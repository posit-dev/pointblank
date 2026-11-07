from __future__ import annotations

import html as html_lib
import inspect
import json
import math
import numbers
import random
import re
import statistics
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import narwhals as nw

from pointblank._utils import _select_df_lib, _with_row_index

if TYPE_CHECKING:
    from great_tables import GT

    from pointblank.validate import Validate

__all__ = [
    "Ablation",
    "AblationVariant",
]

# The row number column used to match the rows that steps failed to the rows of the data
_INDEX = "pb_index_"

_CONTROLS = ("size_matched",)
_ADJUSTMENTS = ("holm",)

_VERDICT_COLORS = {"helps": "#4CA64C", "hurts": "#CF142B", "inconclusive": "#999999"}

# The columns of `Ablation.get_effects()`, with their types
_EFFECT_COLUMNS = {
    "variant": "str",
    "name": "str",
    "strategy": "str",
    "steps": "list",
    "n_removed": "int",
    "f_removed": "float",
    "metric": "str",
    "mean": "float",
    "sd": "float",
    "delta_vs_full": "float",
    "delta_vs_full_lower": "float",
    "delta_vs_full_upper": "float",
    "p_vs_full": "float",
    "p_adj_vs_full": "float",
    "control_mean": "float",
    "delta_vs_control": "float",
    "delta_vs_control_lower": "float",
    "delta_vs_control_upper": "float",
    "p_vs_control": "float",
    "p_adj_vs_control": "float",
    "verdict": "str",
}


@dataclass(frozen=True)
class AblationVariant:
    """
    A version of the training table evaluated in an ablation.

    An [`Ablation`](`pointblank.Ablation`) evaluates a model on several versions of the target
    table of a validation: the full table, the table with the rows that failed some of the steps
    removed (a variant), and the table with the same number of rows removed at random (a control).
    Each time it calls the `evaluate=` function, it passes the version being evaluated as the
    `variant=` argument, so the function can tell them apart (e.g., to log the runs, or to name the
    files that a model is saved to). The variants of an ablation are also listed in its
    [`plan()`](`pointblank.Ablation.plan`) and its results.

    Attributes
    ----------
    id
        A short identifier for the version, unique within the ablation: `"full"` for the full
        table, `"step_<i>"` for a variant of [`drop_each()`](`pointblank.Ablation.drop_each`),
        `"all"` for [`drop_all()`](`pointblank.Ablation.drop_all`), the `name=` (or
        `"steps_<i>_<j>"`) for [`drop_steps()`](`pointblank.Ablation.drop_steps`), and
        `"control_<n>"` for the control that removes `n` random rows.
    strategy
        The kind of version: `"full"`, `"drop_steps"`, `"drop_each"`, `"drop_all"`, or
        `"control"`.
    name
        A name for the version, as shown in reports (e.g., `"Step 2"`).
    steps
        The numbers of the validation steps whose failing rows were removed (empty for the full
        table and for controls).
    n_removed
        The number of rows removed from the table.
    f_removed
        The share of the table's rows that were removed (from `0` to `1`).
    is_control
        Is this a control (random rows removed) rather than a variant?
    group
        Reserved for strategies that remove groups of rows (always `None` for now).
    """

    id: str
    strategy: str
    name: str
    steps: tuple[int, ...] = ()
    n_removed: int = 0
    f_removed: float = 0.0
    is_control: bool = False
    group: Any = None

    def _to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "strategy": self.strategy,
            "name": self.name,
            "steps": list(self.steps),
            "n_removed": self.n_removed,
            "f_removed": self.f_removed,
            "is_control": self.is_control,
            "group": self.group,
        }


@dataclass(frozen=True)
class _Spec:
    """A variant as it was asked for, before the rows it removes are known."""

    id: str
    name: str
    strategy: str
    steps: tuple[int, ...]


class Ablation:
    """
    Measure how the rows that fail validation steps affect a model trained on the data.

    A validation says which rows of a training table fail its checks, but not whether those rows
    matter to a model trained on the table. An ablation finds out: it trains and evaluates the
    model on the full table and on versions of the table with the rows that failed some of the
    steps removed, and compares the results. Pointblank never trains anything itself: you provide
    an `evaluate=` function that takes a training table and returns a score (e.g., the accuracy of
    a model trained on the table, measured on a fixed holdout set).

    Removing rows can change a score just because there are fewer rows to train on, so each
    variant is also compared to a *control* that removes the same number of rows at random. The
    difference between the two is the effect of the rows that failed the steps, as opposed to the
    effect of having less data. Training is often random too (e.g., the initial weights of a
    model), so every version is evaluated `repeats=` times with different seeds. The `r`-th
    evaluation of every version gets the same seed, so the differences between versions are
    compared seed by seed (paired), which cancels out much of the noise.

    An `Ablation` is built in two parts, like a validation: the strategy methods add variants
    ([`drop_each()`](`pointblank.Ablation.drop_each`),
    [`drop_all()`](`pointblank.Ablation.drop_all`), and
    [`drop_steps()`](`pointblank.Ablation.drop_steps`)), and nothing is evaluated until
    [`run()`](`pointblank.Ablation.run`) is called. Use [`plan()`](`pointblank.Ablation.plan`)
    before that to see how many rows each variant removes and how many evaluations the run needs.

    Parameters
    ----------
    validation
        An interrogated validation (a [`Validate`](`pointblank.Validate`) object). Its target table
        is the training table, and its row-level steps (the same ones that
        [`get_row_flags()`](`pointblank.Validate.get_row_flags`) uses) say which rows to remove.
    evaluate
        A function that trains and evaluates a model. It's called as `evaluate(train, seed=seed,
        variant=variant)`, where `train` is the training table with some rows removed (of the same
        type as the validation's target table; a lazy or database table is passed as a lazy
        table), `seed` is an integer to seed any randomness in training with, and `variant` is an
        [`AblationVariant`](`pointblank.AblationVariant`) describing the version of the table. The
        function can leave out the `variant=` parameter if it doesn't need it. It must return a
        number, or a dictionary of named numbers (one per metric). The evaluation data should be a
        fixed holdout set that the function holds on to: it must not depend on `train`.
    metric
        The name of the metric that verdicts are based on. It's needed when `evaluate=` returns a
        dictionary with more than one metric (the others are still reported). When `evaluate=`
        returns a number, this is the name it's given (`"score"` by default).
    higher_is_better
        Is a higher value of `metric=` better (e.g., accuracy) or worse (e.g., an error)? This
        decides whether a variant that raises the metric helps or hurts.
    repeats
        The number of times each version of the table is evaluated, each time with a different
        seed. At least `3` are needed for verdicts. More repeats give narrower intervals but take
        longer to run.
    seed
        The seed of the first repeat: repeat `r` (counting from `0`) passes `seed + r` to
        `evaluate=`. The random rows removed by the controls also depend on it, so a run can be
        reproduced exactly.
    control
        The control that variants are compared to. With `"size_matched"` (the default), each
        variant is compared to the full table with the same number of rows removed at random
        (drawn anew for each repeat). Variants that remove the same number of rows share their
        control. Use `None` to compare variants to the full table only.
    duplicates
        How the duplicate rows found by [`rows_distinct()`](`pointblank.Validate.rows_distinct`)
        steps are removed. With `"keep_first"` (the default) or `"keep_last"`, one copy of each
        group of duplicates is kept and the others are removed. `"flag_all"` removes every copy,
        and `None` leaves `rows_distinct()` steps out of the ablation. These are the same options as
        in [`get_row_flags()`](`pointblank.Validate.get_row_flags`).
    duplicates_order_by
        One or more columns that order the copies of a duplicated row, to decide which copy is
        kept (e.g., `"updated_at"` with `duplicates="keep_last"` keeps the newest record).
    max_removed
        A variant that removes more than this share of the rows (from `0` to `1`) gives a warning,
        as a comparison of such different tables says more about the size of the training table
        than about the quality of the removed rows.
    conf_level
        The confidence level of the intervals, and of the verdicts (a variant helps or hurts if
        its adjusted p-value is below `1 - conf_level`).
    adjust
        How p-values are adjusted for comparing several variants: `"holm"` (the default) uses the
        Holm method, which keeps the chance of any false verdict among the variants below
        `1 - conf_level`. Use `None` to use the unadjusted p-values.
    label
        An optional label for the ablation, shown in its reports.

    Returns
    -------
    Ablation
        An `Ablation` object, with no variants yet.

    Details
    -------
    For each variant and metric, the results give the mean and standard deviation of the metric
    across the repeats, and two paired comparisons: with the full table (`delta_vs_full`) and with
    the control (`delta_vs_control`). Each comparison has a t-interval at `conf_level=` and a
    p-value from a paired t-test, along with the p-value adjusted across the variants. The
    verdict on a variant is based on the metric named in `metric=` and the comparison with the
    control (or with the full table, if `control=None`):

    - `"helps"`: removing the variant's rows makes the metric better, with an adjusted p-value
      below `1 - conf_level`
    - `"hurts"`: removing the rows makes the metric worse, with an adjusted p-value below
      `1 - conf_level`
    - `"inconclusive"`: the difference could be due to chance

    With fewer than 3 repeats, there are no verdicts. The intervals aren't adjusted for multiple
    comparisons, so an interval can leave out zero while the verdict is `"inconclusive"`.

    A variant that removes no rows (e.g., a step that no row failed) is left out of the ablation,
    with a note in the reports. Steps that can't be matched to the rows of the table, such as
    steps with a `pre=` function not marked with [`row_preserving()`](`pointblank.row_preserving`),
    are left out too.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's take a table of temperature readings where some readings were recorded in tenths of a
    degree (so they're ten times too high) and some are missing. The 'model' trained on it is
    simple: it estimates the average temperature from a bootstrap sample of the readings, and it's
    evaluated by its error against the known average of `20` degrees.

    ```{python}
    import pointblank as pb
    import polars as pl

    temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
    for k in range(0, 300, 25):
        temps[k] *= 10
    for k in range(7, 300, 40):
        temps[k] = None
    readings = pl.DataFrame({"temp": temps})

    def evaluate(train, *, seed, variant):
        sample = train.drop_nulls().sample(fraction=1.0, with_replacement=True, seed=seed)
        return {"error": abs(sample["temp"].mean() - 20)}
    ```

    The validation has two steps: one for missing readings and one for readings out of range.

    ```{python}
    validation = (
        pb.Validate(data=readings)
        .col_vals_not_null(columns="temp")
        .col_vals_between(columns="temp", left=-10, right=45, na_pass=True)
        .interrogate()
    )
    ```

    Now we can set up the ablation, with one variant for each step (`drop_each()`) and one for
    both steps (`drop_all()`). The error is the metric, and lower errors are better.

    ```{python}
    ablation = (
        pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
        .drop_each()
        .drop_all()
    )

    ablation.run()
    ```

    Removing the rows that failed step 2 (the readings out of range) cuts the error by much more
    than removing the same number of random rows, so the verdict is that it helps. The missing
    readings were dropped by the model anyway, so removing them makes no real difference and the
    verdict on step 1 is inconclusive. The numbers behind the report are available as a table from
    [`get_effects()`](`pointblank.Ablation.get_effects`).

    See Also
    --------
    Validate.get_row_flags : Get which steps each row failed, as columns added to the table.
    AblationVariant : A version of the training table evaluated in an ablation.
    """

    def __init__(
        self,
        validation: Validate,
        evaluate: Callable[..., Any],
        metric: str | None = None,
        higher_is_better: bool = True,
        repeats: int = 5,
        seed: int = 0,
        control: str | None = "size_matched",
        duplicates: str | None = "keep_first",
        duplicates_order_by: str | list[str] | None = None,
        max_removed: float = 0.5,
        conf_level: float = 0.95,
        adjust: str | None = "holm",
        label: str | None = None,
    ) -> None:
        from pointblank.validate import (
            Validate,
            _check_duplicates_policy,
            _row_result_steps,
            get_row_count,
        )

        if not isinstance(validation, Validate):
            raise TypeError(
                "`validation=` must be a `Validate` object, got "
                f"{type(validation).__name__} instead."
            )
        if getattr(validation, "time_start", None) is None:
            raise ValueError("The validation hasn't been interrogated. Call `interrogate()` first.")
        if not callable(evaluate):
            raise TypeError(
                f"`evaluate=` must be a function, got {type(evaluate).__name__} instead."
            )
        if metric is not None and not isinstance(metric, str):
            raise TypeError(f"`metric=` must be a string, got {type(metric).__name__} instead.")
        if not isinstance(higher_is_better, bool):
            raise TypeError("`higher_is_better=` must be `True` or `False`.")
        if not _is_int(repeats) or repeats < 1:
            raise ValueError(f"`repeats=` must be a positive integer, got {repeats!r}.")
        if not _is_int(seed):
            raise TypeError(f"`seed=` must be an integer, got {seed!r}.")
        if control is not None and control not in _CONTROLS:
            raise ValueError(
                f"`control=` must be one of {', '.join(repr(c) for c in _CONTROLS)}, or `None`; "
                f"got {control!r}."
            )
        if not _is_number(max_removed) or not 0 < max_removed <= 1:
            raise ValueError(f"`max_removed=` must be a number in (0, 1], got {max_removed!r}.")
        if not _is_number(conf_level) or not 0 < conf_level < 1:
            raise ValueError(f"`conf_level=` must be a number in (0, 1), got {conf_level!r}.")
        if adjust is not None and adjust not in _ADJUSTMENTS:
            raise ValueError(
                f"`adjust=` must be one of {', '.join(repr(a) for a in _ADJUSTMENTS)}, or `None`; "
                f"got {adjust!r}."
            )
        if label is not None and not isinstance(label, str):
            raise TypeError(f"`label=` must be a string, got {type(label).__name__} instead.")

        columns = nw.from_native(validation.data).collect_schema().names()
        self._order_by = _check_duplicates_policy(duplicates, duplicates_order_by, columns=columns)
        steps, skipped = _row_result_steps(validation.validation_info, duplicates=duplicates)

        if repeats < 3:
            warnings.warn(
                f"With `repeats={repeats}`, the ablation won't give verdicts on its variants (at "
                "least 3 repeats are needed).",
                UserWarning,
                stacklevel=2,
            )

        self.validation = validation
        self.evaluate = evaluate
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.repeats = repeats
        self.seed = seed
        self.control = control
        self.duplicates = duplicates
        self.duplicates_order_by = duplicates_order_by
        self.max_removed = float(max_removed)
        self.conf_level = float(conf_level)
        self.adjust = adjust
        self.label = label

        self._n_rows = get_row_count(validation.data)
        self._steps = {step.i: step for step in steps}
        self._skipped = dict(skipped)
        self._pass_variant = _accepts_keyword(evaluate, "variant")

        self._specs: list[_Spec] = []
        self._notes: list[str] = []
        self._resolved: tuple[list[tuple[AblationVariant, frozenset[int]]], list[str]] | None = None
        self._reset_results()

    def __repr__(self) -> str:
        n_variants = len(self._specs)
        status = "run" if self._effects is not None else "not run"
        label = f"{self.label!r}: " if self.label else ""
        return (
            f"Ablation({label}{n_variants} variant{'s' if n_variants != 1 else ''}, "
            f"{self.repeats} repeats, control={self.control!r}; {status})"
        )

    def _repr_html_(self) -> str:
        if self._effects is None:
            return self.plan().as_raw_html()
        return self.get_tabular_report().as_raw_html()

    # ------------------------------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------------------------------

    def drop_steps(self, steps: int | list[int], name: str | None = None) -> Ablation:
        """
        Add a variant that removes the rows failing any of the given steps.

        This is the most flexible strategy: it adds one variant, with the rows that failed any of
        `steps=` removed. Use it to test a particular combination of checks (e.g., all the checks
        on one column), or to give a variant a name of its own.

        Parameters
        ----------
        steps
            One or more step numbers. Each must be a row-level step that the ablation can use
            (active, and without a `pre=` function unless that function is marked with
            [`row_preserving()`](`pointblank.row_preserving`)).
        name
            A name for the variant, shown in reports and used as its `id`. By default, it's named
            after the steps (e.g., `"Steps 1, 3"`, with the id `"steps_1_3"`).

        Returns
        -------
        Ablation
            The `Ablation` object, with the variant added.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's a table of temperature readings with some out-of-range values, checked against two
        different ranges. The 'model' estimates the average temperature and is evaluated by its
        error against the true average of `20`.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed, variant):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_between(columns="temp", left=-10, right=45)
            .col_vals_lt(columns="temp", value=100)
            .interrogate()
        )

        ablation = pb.Ablation(
            validation, evaluate=evaluate, metric="error", higher_is_better=False
        ).drop_steps([1, 2], name="Range checks")

        ablation.plan()
        ```

        The plan has the full table, the one variant (removing the rows that failed either
        step), and its control, which removes the same number of rows at random.
        """
        steps = self._check_steps(steps, method="drop_steps")
        if name is None:
            label = "Step" if len(steps) == 1 else "Steps"
            name = f"{label} {', '.join(str(i) for i in steps)}"
            spec_id = "steps_" + "_".join(str(i) for i in steps)
        else:
            if not isinstance(name, str) or not name:
                raise TypeError("`name=` must be a non-empty string.")
            spec_id = name
        self._add_spec(_Spec(id=spec_id, name=name, strategy="drop_steps", steps=steps))
        return self

    def drop_each(self, steps: int | list[int] | None = None) -> Ablation:
        """
        Add a variant for each step, removing only the rows that failed that step.

        Each variant answers the question 'does removing the rows that fail this check help, on
        its own?' Comparing the variants shows which checks matter to the model and which don't.

        Parameters
        ----------
        steps
            The step numbers to add variants for. By default, every row-level step that the
            ablation can use gets one (the steps that can't be used are listed in the notes of the
            reports).

        Returns
        -------
        Ablation
            The `Ablation` object, with the variants added.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's a table of temperature readings with out-of-range and missing values. The 'model'
        estimates the average temperature, ignoring missing readings, and is evaluated by its error
        against the true average of `20`.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        for k in range(7, 300, 40):
            temps[k] = None
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed, variant):
            sample = train.drop_nulls().sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_not_null(columns="temp")
            .col_vals_between(columns="temp", left=-10, right=45, na_pass=True)
            .interrogate()
        )

        ablation = pb.Ablation(
            validation, evaluate=evaluate, metric="error", higher_is_better=False
        ).drop_each()

        ablation.plan()
        ```

        There's a variant for each step. The two variants remove different numbers of rows, so
        each has its own control.
        """
        if steps is None:
            steps = self._all_steps(method="drop_each")
        else:
            steps = self._check_steps(steps, method="drop_each")
        for i in steps:
            self._add_spec(
                _Spec(id=f"step_{i}", name=f"Step {i}", strategy="drop_each", steps=(i,))
            )
        return self

    def drop_all(self, steps: list[int] | None = None) -> Ablation:
        """
        Add a variant that removes the rows failing any step.

        This variant answers the question 'does the validation as a whole help?' Without
        `rows_distinct()` steps, its training table is the one that
        [`get_sundered_data()`](`pointblank.Validate.get_sundered_data`) gives: the rows that
        passed every step.

        Parameters
        ----------
        steps
            The step numbers to combine. By default, every row-level step that the ablation can
            use is combined.

        Returns
        -------
        Ablation
            The `Ablation` object, with the variant added.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's a table of temperature readings with out-of-range and missing values, validated
        with a step for each problem.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        for k in range(7, 300, 40):
            temps[k] = None
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed, variant):
            sample = train.drop_nulls().sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_not_null(columns="temp")
            .col_vals_between(columns="temp", left=-10, right=45, na_pass=True)
            .interrogate()
        )

        ablation = pb.Ablation(
            validation, evaluate=evaluate, metric="error", higher_is_better=False
        ).drop_all()

        ablation.plan()
        ```

        The one variant removes the 20 rows that failed either step.
        """
        if steps is None:
            steps = self._all_steps(method="drop_all")
            spec = _Spec(id="all", name="All steps", strategy="drop_all", steps=steps)
        else:
            steps = self._check_steps(steps, method="drop_all")
            numbers_text = ", ".join(str(i) for i in steps)
            spec = _Spec(
                id="all_" + "_".join(str(i) for i in steps),
                name=f"All of steps {numbers_text}",
                strategy="drop_all",
                steps=steps,
            )
        self._add_spec(spec)
        return self

    def _check_steps(self, steps: Any, method: str) -> tuple[int, ...]:
        if _is_int(steps):
            steps = [steps]
        elif isinstance(steps, (list, tuple, range)):
            steps = list(steps)
        else:
            raise TypeError(
                f"`steps=` must be a step number or a list of them, got {type(steps).__name__}."
            )
        if not steps:
            raise ValueError("`steps=` must have at least one step number.")
        for i in steps:
            if not _is_int(i):
                raise TypeError(f"Step numbers must be integers, got {i!r}.")
            if i not in self._steps:
                raise ValueError(f"`{method}()` can't use step {i}: {self._ineligible_reason(i)}.")
        return tuple(sorted(set(steps)))

    def _all_steps(self, method: str) -> tuple[int, ...]:
        for i, reason in self._skipped.items():
            self._add_note(f"Step {i} isn't used: {reason}.")
        if not self._steps:
            raise ValueError(
                f"`{method}()` has no steps to use: the validation has no row-level steps that "
                "can be matched to the rows of the table."
            )
        return tuple(sorted(self._steps))

    def _ineligible_reason(self, i: int) -> str:
        if i in self._skipped:
            return self._skipped[i]
        step = next((v for v in self.validation.validation_info if v.i == i), None)
        if step is None:
            return f"the validation has no step {i}"
        if not step.active:
            return "it's inactive"
        return "it doesn't check rows"

    def _add_spec(self, spec: _Spec) -> None:
        if spec.id == "full" or spec.id.startswith("control_"):
            raise ValueError(f"The variant name {spec.id!r} is reserved; choose another one.")
        for existing in self._specs:
            if existing.id == spec.id:
                if existing.steps == spec.steps:
                    return
                raise ValueError(f"There's already a variant called {spec.id!r}.")
        for existing in self._specs:
            if existing.steps == spec.steps:
                if existing.name != spec.name:
                    self._add_note(
                        f"{spec.name} isn't used: it removes the same rows as {existing.name}."
                    )
                return
        self._specs.append(spec)
        self._resolved = None
        self._reset_results()

    def _add_note(self, note: str) -> None:
        if note not in self._notes:
            self._notes.append(note)

    # ------------------------------------------------------------------------------------------
    # Rows and tables
    # ------------------------------------------------------------------------------------------

    def _resolve(self) -> tuple[list[tuple[AblationVariant, frozenset[int]]], list[str]]:
        """
        Find the rows each variant removes, leaving out the variants that remove no rows (or every
        row), and give the notes about them.
        """
        if self._resolved is not None:
            return self._resolved

        failed = self._failed_rows(sorted({i for spec in self._specs for i in spec.steps}))
        n_rows = self._n_rows
        variants = []
        notes = []

        for spec in self._specs:
            removed = frozenset().union(*(failed[i] for i in spec.steps))
            n_removed = len(removed)
            if n_removed == 0:
                notes.append(f"{spec.name} isn't used: it removes no rows.")
                continue
            if n_removed >= n_rows:
                notes.append(f"{spec.name} isn't used: it removes every row.")
                continue
            f_removed = n_removed / n_rows
            if f_removed > self.max_removed:
                warnings.warn(
                    f"{spec.name} removes {f_removed:.1%} of the rows, more than "
                    f"`max_removed={self.max_removed:g}`. With that much removed, a comparison "
                    "says more about the size of the training table than about the quality of "
                    "the removed rows.",
                    UserWarning,
                    stacklevel=3,
                )
            variant = AblationVariant(
                id=spec.id,
                strategy=spec.strategy,
                name=spec.name,
                steps=spec.steps,
                n_removed=n_removed,
                f_removed=f_removed,
            )
            variants.append((variant, removed))

        self._resolved = (variants, notes)
        return self._resolved

    def _failed_rows(self, steps: list[int]) -> dict[int, frozenset[int]]:
        """Get the numbers of the rows that failed each step."""
        from pointblank.validate import _step_row_results

        failed = {}
        if not steps:
            return failed

        data_nw = _with_row_index(self.validation.data, name=_INDEX)
        for i in steps:
            results = (
                _step_row_results(
                    self._steps[i],
                    data_nw,
                    index_name=_INDEX,
                    duplicates=self.duplicates,
                    duplicates_order_by=self._order_by,
                )
                .filter(nw.col("pb_failed_"))
                .select(_INDEX)
            )
            if isinstance(results, nw.LazyFrame):
                results = results.collect()
            failed[i] = frozenset(int(k) for k in results[_INDEX].to_list())
        return failed

    def _controls(self, variants: list[AblationVariant]) -> list[AblationVariant]:
        if self.control is None:
            return []
        return [
            AblationVariant(
                id=f"control_{n}",
                strategy="control",
                name=f"{n:,} random rows",
                n_removed=n,
                f_removed=n / self._n_rows,
                is_control=True,
            )
            for n in sorted({v.n_removed for v in variants})
        ]

    def _control_rows(self, n: int, repeat: int) -> frozenset[int]:
        """The random rows that the control removing `n` rows removes in a repeat."""
        rng = random.Random(f"pointblank-ablation:{self.seed}:{repeat}:{n}")
        return frozenset(rng.sample(range(self._n_rows), n))

    def _training_table(self, removed: frozenset[int]) -> Any:
        """The target table without the `removed` rows, of the same type as the target table."""
        data_nw = _with_row_index(self.validation.data, name=_INDEX)
        if removed:
            data_nw = data_nw.filter(~nw.col(_INDEX).is_in(sorted(removed)))
        return data_nw.drop(_INDEX).to_native()

    # ------------------------------------------------------------------------------------------
    # Running
    # ------------------------------------------------------------------------------------------

    def _reset_results(self) -> None:
        self._runs: list[dict[str, Any]] = []
        self._values: dict[tuple[str, int], dict[str, float]] = {}
        self._metrics: list[str] | None = None
        self._primary: str | None = None
        self._variants: list[AblationVariant] = []
        self._control_variants: list[AblationVariant] = []
        self._effects: list[dict[str, Any]] | None = None

    def _full_variant(self) -> AblationVariant:
        return AblationVariant(id="full", strategy="full", name="Full table")

    def plan(self) -> GT:
        """
        Show the versions of the table that the ablation will evaluate.

        The plan is a dry run: it finds the rows that each variant removes, without evaluating
        anything. It lists the full table, each variant, and each control, with the number and
        share of rows they remove, and gives the number of evaluations that
        [`run()`](`pointblank.Ablation.run`) will make in its header. Variants that won't be used
        (e.g., because they remove no rows) are listed in the notes at the bottom.

        Returns
        -------
        GT
            A Great Tables object.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's an ablation of a table of temperature readings, checked for missing values and for
        values out of range, and also for duplicate readings (there are none).

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        for k in range(7, 300, 40):
            temps[k] = None
        readings = pl.DataFrame({"id": range(300), "temp": temps})

        def evaluate(train, *, seed, variant):
            sample = train.drop_nulls().sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_not_null(columns="temp")
            .col_vals_between(columns="temp", left=-10, right=45, na_pass=True)
            .rows_distinct(columns_subset="id")
            .interrogate()
        )

        ablation = pb.Ablation(
            validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=10
        ).drop_each()

        ablation.plan()
        ```

        No row failed step 3, so it has no variant, as the note says. With 10 repeats of the full
        table, two variants, and two controls, the run will make 50 evaluations.
        """
        from great_tables import GT, google_font, html, loc, style

        from pointblank._utils_html import _fmt_raw_html

        variants, notes = self._resolve()
        variant_list = [v for v, _ in variants]
        controls = self._controls(variant_list)
        n_runs = self.repeats * (1 + len(variant_list) + len(controls))

        rows = [
            {
                "group": "Baseline",
                "variant": _variant_html(self._full_variant(), self._steps),
                "removed": _removed_html(0, 0.0),
            }
        ]
        rows += [
            {
                "group": "Variants",
                "variant": _variant_html(v, self._steps),
                "removed": _removed_html(v.n_removed, v.f_removed),
            }
            for v in variant_list
        ]
        rows += [
            {
                "group": "Controls",
                "variant": _variant_html(v, self._steps),
                "removed": _removed_html(v.n_removed, v.f_removed),
            }
            for v in controls
        ]

        subtitle = (
            f"{self._n_rows:,} rows in the training table. Each version is evaluated "
            f"{self.repeats} time{'s' if self.repeats != 1 else ''}, so the run will make "
            f"<strong>{n_runs:,}</strong> evaluations."
        )
        gt_tbl = (
            _fmt_raw_html(
                GT(
                    _frame(rows, ["group", "variant", "removed"]),
                    groupname_col="group",
                    id="pb_ablation_plan",
                ),
                columns=["variant", "removed"],
            )
            .tab_header(
                title=html(_title_html("Ablation plan", self.label)), subtitle=html(subtitle)
            )
            .cols_label(variant="Version", removed="Removed")
            .cols_align(align="right", columns="removed")
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_style(style=style.text(size="12px"), locations=loc.body())
            .tab_style(style=style.text(size="12px", weight="bold"), locations=loc.column_labels())
            .tab_style(
                style=[style.text(weight="bold", size="12px"), style.fill(color="#F6F7F8")],
                locations=loc.row_groups(),
            )
            .cols_width(cases={"variant": "75%", "removed": "25%"})
            .tab_options(table_width="100%")
        )
        for note in [*self._notes, *notes]:
            gt_tbl = gt_tbl.tab_source_note(source_note=html(_note_html(note)))
        return gt_tbl

    def run(self) -> Ablation:
        """
        Evaluate the full table, each variant, and each control.

        The `evaluate=` function is called `repeats=` times for each version of the table, with
        the seeds `seed`, `seed + 1`, and so on. The full table is evaluated first, then each
        variant, then each control. A control is drawn anew for each repeat, so that the
        comparison with it doesn't depend on one random draw of rows. Once the run is done, the
        results are available from [`get_tabular_report()`](`pointblank.Ablation.get_tabular_report`),
        [`get_effects()`](`pointblank.Ablation.get_effects`), and
        [`to_json()`](`pointblank.Ablation.to_json`). Running again repeats every evaluation.

        Returns
        -------
        Ablation
            The `Ablation` object, with the results of the run.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's an ablation of a table of temperature readings with some out-of-range values. The
        `evaluate()` function also returns the number of rows it was given, which is a second
        metric in the results.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed, variant):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return {"error": abs(sample["temp"].mean() - 20), "n_train": train.height}

        validation = (
            pb.Validate(data=readings)
            .col_vals_between(columns="temp", left=-10, right=45)
            .interrogate()
        )

        ablation = (
            pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
            .drop_each()
            .run()
        )

        ablation
        ```

        After the run, the `Ablation` object shows its report. The report shows the metric named
        in `metric=`; the other metrics are in the results of `get_effects()`.
        """
        variants, _ = self._resolve()
        if not self._specs:
            raise ValueError(
                "There are no variants to evaluate. Add some with `drop_each()`, `drop_all()`, "
                "or `drop_steps()` first."
            )
        if not variants:
            raise ValueError(
                "None of the variants remove any rows of the table, so there's nothing to "
                "compare with the full table."
            )

        self._reset_results()
        variant_list = [v for v, _ in variants]
        self._variants = variant_list
        self._control_variants = self._controls(variant_list)

        full = self._full_variant()
        for r in range(self.repeats):
            self._evaluate(full, frozenset(), r)
        for variant, removed in variants:
            for r in range(self.repeats):
                self._evaluate(variant, removed, r)
        for control in self._control_variants:
            for r in range(self.repeats):
                self._evaluate(control, self._control_rows(control.n_removed, r), r)

        self._effects = self._compute_effects()
        return self

    def _evaluate(self, variant: AblationVariant, removed: frozenset[int], repeat: int) -> None:
        seed = self.seed + repeat
        kwargs = {"seed": seed}
        if self._pass_variant:
            kwargs["variant"] = variant
        result = self.evaluate(self._training_table(removed), **kwargs)
        values = self._metric_values(result, variant=variant, seed=seed)
        self._values[(variant.id, repeat)] = values
        self._runs.append(
            {"variant": variant.id, "repeat": repeat, "seed": seed, "metrics": values}
        )

    def _metric_values(self, result: Any, variant: AblationVariant, seed: int) -> dict[str, float]:
        where = f"for the variant {variant.id!r} (seed {seed})"
        if isinstance(result, dict):
            if not result:
                raise ValueError(f"`evaluate=` returned an empty dictionary {where}.")
            values = dict(result)
        elif _is_number(result):
            values = {self.metric or "score": result}
        else:
            raise TypeError(
                "`evaluate=` must return a number or a dictionary of numbers, but it returned "
                f"{type(result).__name__} {where}."
            )

        for name, value in values.items():
            if not isinstance(name, str):
                raise TypeError(f"The metric names from `evaluate=` must be strings, got {name!r}.")
            if not _is_number(value) or not math.isfinite(float(value)):
                raise ValueError(
                    f"`evaluate=` returned {value!r} for the metric {name!r} {where}, but each "
                    "metric must be a finite number."
                )
        values = {name: float(value) for name, value in values.items()}

        if self._metrics is None:
            if self.metric is None and len(values) > 1:
                raise ValueError(
                    f"`evaluate=` returned several metrics ({', '.join(map(repr, values))}), so "
                    "use `metric=` to say which one the verdicts are based on."
                )
            if self.metric is not None and self.metric not in values:
                raise ValueError(
                    f"`evaluate=` didn't return the metric {self.metric!r} named in `metric=` (it "
                    f"returned {', '.join(map(repr, values))})."
                )
            self._metrics = list(values)
            self._primary = self.metric or self._metrics[0]
        elif set(values) != set(self._metrics):
            raise ValueError(
                f"`evaluate=` returned the metrics {', '.join(map(repr, values))} {where}, but "
                f"{', '.join(map(repr, self._metrics))} for the full table. It must return the "
                "same metrics every time."
            )
        return values

    # ------------------------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------------------------

    def _compute_effects(self) -> list[dict[str, Any]]:
        repeats = range(self.repeats)
        controls = {c.n_removed: c for c in self._control_variants}
        rows = []

        for metric in self._metrics:
            full = [self._values[("full", r)][metric] for r in repeats]
            rows.append(
                _effect_row(
                    self._full_variant(),
                    metric=metric,
                    mean=statistics.fmean(full),
                    sd=_sd(full),
                )
            )

            variant_rows = []
            for variant in self._variants:
                values = [self._values[(variant.id, r)][metric] for r in repeats]
                row = _effect_row(
                    variant, metric=metric, mean=statistics.fmean(values), sd=_sd(values)
                )
                _add_comparison(row, "full", values, full, self.conf_level)
                if variant.n_removed in controls:
                    control_id = controls[variant.n_removed].id
                    control = [self._values[(control_id, r)][metric] for r in repeats]
                    row["control_mean"] = statistics.fmean(control)
                    _add_comparison(row, "control", values, control, self.conf_level)
                variant_rows.append(row)

            for comparison in ("full", "control"):
                p_values = [row[f"p_vs_{comparison}"] for row in variant_rows]
                adjusted = _holm(p_values) if self.adjust == "holm" else p_values
                for row, p_adj in zip(variant_rows, adjusted):
                    row[f"p_adj_vs_{comparison}"] = p_adj

            if metric == self._primary:
                for row in variant_rows:
                    row["verdict"] = self._verdict(row)

            rows += variant_rows

        return rows

    def _verdict(self, row: dict[str, Any]) -> str | None:
        if self.repeats < 3:
            return None
        comparison = "control" if self.control is not None else "full"
        p_adj = row[f"p_adj_vs_{comparison}"]
        delta = row[f"delta_vs_{comparison}"]
        if p_adj is None or delta is None:
            return None  # pragma: no cover
        if p_adj < 1 - self.conf_level and delta != 0:
            better = delta > 0 if self.higher_is_better else delta < 0
            return "helps" if better else "hurts"
        return "inconclusive"

    def _check_run(self) -> None:
        if self._effects is None:
            raise ValueError("There are no results yet. Call `run()` first.")

    def get_effects(self) -> Any:
        """
        Get the results of the ablation as a table.

        The table has one row for each version of the training table and each metric: the full
        table first (as the baseline), then the variants. Use it to make plots, to compare
        ablations of different data, or to look at the metrics that the report doesn't show.

        Returns
        -------
        DataFrame
            A Polars DataFrame (or a pandas DataFrame, if Polars isn't installed) with these
            columns:

            - `variant`, `name`, `strategy`: the variant's id, name, and strategy (`"full"` for the
            full table)
            - `steps`: the numbers of the steps whose failing rows were removed, as a list
            - `n_removed`, `f_removed`: the number and share of rows removed
            - `metric`, `mean`, `sd`: the metric, and its mean and standard deviation across the
            repeats
            - `delta_vs_full`, `delta_vs_full_lower`, `delta_vs_full_upper`: the mean paired
            difference from the full table, with its interval at `conf_level=`
            - `p_vs_full`, `p_adj_vs_full`: the p-value of that difference, and the p-value
            adjusted across the variants (as set by `adjust=`)
            - `control_mean`: the mean of the metric for the variant's control
            - `delta_vs_control`, `delta_vs_control_lower`, `delta_vs_control_upper`,
            `p_vs_control`, `p_adj_vs_control`: the same, for the difference from the control
            - `verdict`: `"helps"`, `"hurts"`, or `"inconclusive"` for the metric named in
            `metric=` (missing for other metrics, and with fewer than 3 repeats)

            The comparisons are missing for the full table, and the control columns are missing
            when `control=None`. The intervals and p-values need at least 2 repeats.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Here's an ablation of a table of temperature readings with some out-of-range values.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed, variant):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_between(columns="temp", left=-10, right=45)
            .interrogate()
        )

        ablation = (
            pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
            .drop_each()
            .run()
        )

        ablation.get_effects().select("variant", "mean", "delta_vs_control", "verdict")
        ```

        Removing the 12 rows that failed step 1 lowers the error by about 7 degrees compared
        with removing 12 random rows, so the verdict is that the step helps.
        """
        self._check_run()
        rows = self._effects
        lib = _select_df_lib(preference="polars")

        if lib.__name__ == "polars":
            dtypes = {
                "str": lib.String,
                "int": lib.Int64,
                "float": lib.Float64,
                "list": lib.List(lib.Int64),
            }
            schema = {name: dtypes[kind] for name, kind in _EFFECT_COLUMNS.items()}
            return lib.DataFrame(
                [[row[c] for c in _EFFECT_COLUMNS] for row in rows], schema=schema, orient="row"
            )

        df = lib.DataFrame(
            [[row[c] for c in _EFFECT_COLUMNS] for row in rows], columns=list(_EFFECT_COLUMNS)
        )
        for name, kind in _EFFECT_COLUMNS.items():
            if kind == "float":
                df[name] = df[name].astype("float64")
            elif kind == "int":
                df[name] = df[name].astype("int64")
        return df

    def to_dict(self) -> dict[str, Any]:
        """
        Get the settings and results of the ablation as a JSON-serializable dict.

        Returns
        -------
        dict[str, Any]
            A dictionary with the ablation's settings (`"settings"`), the versions of the table
            that were evaluated (`"variants"`, including the full table and the controls), the
            metrics from every evaluation (`"runs"`, one entry per version and repeat, with its
            seed), the results (`"effects"`, with the same fields as the columns of
            [`get_effects()`](`pointblank.Ablation.get_effects`)), and the notes about variants
            and steps that weren't used (`"notes"`).

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's an ablation of a table of temperature readings, with three repeats.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed, variant):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_between(columns="temp", left=-10, right=45)
            .interrogate()
        )

        ablation = (
            pb.Ablation(
                validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=3
            )
            .drop_each()
            .run()
        )

        ablation.to_dict()["runs"][:4]
        ```

        Each run records the version of the table, the repeat, the seed, and the metrics, so the
        runs can be stored and analyzed again later.
        """
        self._check_run()
        settings = {
            "label": self.label,
            "metric": self._primary,
            "metrics": list(self._metrics),
            "higher_is_better": self.higher_is_better,
            "repeats": self.repeats,
            "seed": self.seed,
            "control": self.control,
            "duplicates": self.duplicates,
            "duplicates_order_by": list(self._order_by) if self._order_by else None,
            "max_removed": self.max_removed,
            "conf_level": self.conf_level,
            "adjust": self.adjust,
            "n_rows": self._n_rows,
            "tbl_name": getattr(self.validation, "tbl_name", None),
        }
        variants = [self._full_variant(), *self._variants, *self._control_variants]
        _, notes = self._resolve()
        return {
            "settings": settings,
            "variants": [v._to_dict() for v in variants],
            "runs": [{**run, "metrics": dict(run["metrics"])} for run in self._runs],
            "effects": [{**row, "steps": list(row["steps"])} for row in self._effects],
            "notes": [*self._notes, *notes],
        }

    def to_json(self, indent: int | None = 2) -> str:
        """
        Get the settings and results of the ablation as a JSON string.

        The structure is the same as that of [`to_dict()`](`pointblank.Ablation.to_dict`). Saving
        it alongside a model's training data makes a record of which checks mattered to the model,
        and of how that was measured.

        Parameters
        ----------
        indent
            The number of spaces to indent nested levels by. Use `None` for the most compact
            output, on a single line.

        Returns
        -------
        str
            The settings and results of the ablation as a JSON string.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's an ablation of a table of temperature readings, with three repeats.

        ```{python}
        import json
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed, variant):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_between(columns="temp", left=-10, right=45)
            .interrogate()
        )

        ablation = (
            pb.Ablation(
                validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=3
            )
            .drop_each()
            .run()
        )

        print(json.dumps(json.loads(ablation.to_json())["settings"], indent=2))
        ```

        The settings record everything needed to repeat the ablation with the same `evaluate()`
        function, including the seed.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def get_tabular_report(self, metric: str | None = None, title: str | None = None) -> GT:
        """
        Get a tabular report of the ablation's results.

        The report has a row for each variant, with the rows it removed, the metric's mean and
        standard deviation across the repeats, and its difference from the control (or from the
        full table, if `control=None`). The difference is shown with its interval, as a bar
        around zero (on the same scale for every variant), and with the verdict as a badge. The
        header gives the metric for the full table and the settings of the run, and notes at the
        bottom list the variants and steps that weren't used.

        Parameters
        ----------
        metric
            The metric to show. By default, it's the one that verdicts are based on (named in
            `metric=` when the ablation was created). Verdicts are only shown for that metric.
        title
            An optional title replacing the default.

        Returns
        -------
        GT
            A Great Tables object.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's an ablation of a table of temperature readings with out-of-range and missing values.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        for k in range(7, 300, 40):
            temps[k] = None
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed, variant):
            sample = train.drop_nulls().sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_not_null(columns="temp")
            .col_vals_between(columns="temp", left=-10, right=45, na_pass=True)
            .interrogate()
        )

        ablation = (
            pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
            .drop_each()
            .run()
        )

        ablation.get_tabular_report(title="Which checks matter for the temperature model?")
        ```

        Removing the readings that are out of range (step 2) helps, while removing the missing
        readings (step 1) makes no clear difference.
        """
        from great_tables import GT, google_font, html, loc, style

        from pointblank._utils_html import _fmt_raw_html

        self._check_run()
        if metric is None:
            metric = self._primary
        elif metric not in self._metrics:
            raise ValueError(
                f"There's no metric {metric!r} in the results (the metrics are "
                f"{', '.join(map(repr, self._metrics))})."
            )

        effects = [row for row in self._effects if row["metric"] == metric]
        baseline, variant_rows = effects[0], effects[1:]
        comparison = "control" if self.control is not None else "full"

        fmt = _NumberFormat([row["mean"] for row in effects])
        bounds = [
            abs(row[f"{key}"])
            for row in variant_rows
            for key in (
                f"delta_vs_{comparison}",
                f"delta_vs_{comparison}_lower",
                f"delta_vs_{comparison}_upper",
            )
            if row[key] is not None
        ]
        scale = max(bounds, default=0.0) or 1.0

        rows = []
        for row in variant_rows:
            variant = next(v for v in self._variants if v.id == row["variant"])
            metric_html = _mono(fmt.value(row["mean"], row["sd"]))
            if comparison == "control":
                metric_html += _small(f"Δ vs full {fmt.delta(row['delta_vs_full'])}")
            rows.append(
                {
                    "variant": _variant_html(variant, self._steps),
                    "removed": _removed_html(row["n_removed"], row["f_removed"]),
                    "metric": metric_html,
                    "effect": _effect_html(row, comparison, fmt, scale),
                }
            )

        vs_text = (
            "removing the same number of random rows"
            if comparison == "control"
            else "the full table"
        )
        # `higher_is_better=` is only known to apply to the metric the verdicts are based on
        direction = ""
        if metric == self._primary:
            direction = f" ({'higher' if self.higher_is_better else 'lower'} is better)"
        subtitle = (
            f"<div>{_esc(metric)} on the full table ({self._n_rows:,} rows): "
            f"<strong>{fmt.value(baseline['mean'], baseline['sd'])}</strong>"
            f"{direction}.</div>"
            f"<div>Each version was evaluated {self.repeats} time"
            f"{'s' if self.repeats != 1 else ''}; Δ is the mean paired difference from "
            f"{vs_text}.</div>"
        )
        if metric != self._primary:
            subtitle += (
                f"<div>Verdicts are based on {_esc(self._primary)}, so none are shown here.</div>"
            )

        effect_label = "Δ vs random" if comparison == "control" else "Δ vs full"
        default_title = _title_html("Ablation report", self.label)
        gt_tbl = (
            _fmt_raw_html(
                GT(
                    _frame(rows, ["variant", "removed", "metric", "effect"]),
                    id="pb_ablation_report",
                ),
                columns=["variant", "removed", "metric", "effect"],
            )
            .tab_header(
                title=html(_esc(title) if title is not None else default_title),
                subtitle=html(subtitle),
            )
            .cols_label(
                variant="Variant", removed="Removed", metric=_esc(metric), effect=effect_label
            )
            .cols_align(align="right", columns=["removed", "metric"])
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_style(style=style.text(size="12px"), locations=loc.body())
            .tab_style(style=style.text(size="12px", weight="bold"), locations=loc.column_labels())
            .cols_width(
                cases={"variant": "40%", "removed": "14%", "metric": "18%", "effect": "28%"}
            )
            .tab_options(table_width="100%")
        )

        _, notes = self._resolve()
        interval_note = f"Intervals are {self.conf_level:.0%} paired t-intervals" + (
            ", not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values."
            if self.adjust == "holm"
            else "; verdicts use unadjusted p-values."
        )
        if self.repeats < 3:
            interval_note += " There are no verdicts with fewer than 3 repeats."
        for note in [interval_note, *self._notes, *notes]:
            gt_tbl = gt_tbl.tab_source_note(source_note=html(_note_html(note)))
        return gt_tbl


# ----------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------


def _sd(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _betacf(a: float, b: float, x: float) -> float:
    """The continued fraction of the incomplete beta function (modified Lentz's method)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 1000):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function, I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_front = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _t_p_value(t: float, df: int) -> float:
    """The two-sided p-value of a t statistic: P(|T| >= |t|) for T with `df` degrees of freedom."""
    if math.isinf(t):
        return 0.0
    return min(1.0, _betainc(df / 2.0, 0.5, df / (df + t * t)))


def _t_quantile(prob: float, df: int) -> float:
    """The `prob` quantile (above 0.5) of the t distribution with `df` degrees of freedom."""
    tail = 2.0 * (1.0 - prob)
    lower, upper = 0.0, 1.0
    while _t_p_value(upper, df) > tail:
        upper *= 2.0
    for _ in range(200):
        middle = (lower + upper) / 2.0
        if _t_p_value(middle, df) > tail:
            lower = middle
        else:
            upper = middle
        if upper - lower < 1e-12 * upper:
            break
    return (lower + upper) / 2.0


def _paired(
    values: list[float], reference: list[float], conf_level: float
) -> tuple[float, float | None, float | None, float | None]:
    """
    Compare paired values: the mean difference, its t-interval, and the p-value of a paired t-test.

    With one pair, there's no interval or p-value. When the differences are all the same, the
    interval is that difference, and the p-value is 0 (or 1, if the difference is 0).
    """
    diffs = [v - r for v, r in zip(values, reference)]
    mean = statistics.fmean(diffs)
    n = len(diffs)
    if n < 2:
        return mean, None, None, None
    se = statistics.stdev(diffs) / math.sqrt(n)
    if se == 0:
        return mean, mean, mean, (0.0 if mean != 0 else 1.0)
    half_width = _t_quantile(1 - (1 - conf_level) / 2, n - 1) * se
    return mean, mean - half_width, mean + half_width, _t_p_value(mean / se, n - 1)


def _holm(p_values: list[float | None]) -> list[float | None]:
    """Adjust p-values with the Holm method (missing p-values are left out and stay missing)."""
    present = sorted((p, k) for k, p in enumerate(p_values) if p is not None)
    m = len(present)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, k) in enumerate(present):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[k] = running
    return adjusted


def _effect_row(variant: AblationVariant, metric: str, mean: float, sd: float | None) -> dict:
    row: dict[str, Any] = {name: None for name in _EFFECT_COLUMNS}
    row.update(
        variant=variant.id,
        name=variant.name,
        strategy=variant.strategy,
        steps=list(variant.steps),
        n_removed=variant.n_removed,
        f_removed=variant.f_removed,
        metric=metric,
        mean=mean,
        sd=sd,
    )
    return row


def _add_comparison(
    row: dict[str, Any],
    comparison: str,
    values: list[float],
    reference: list[float],
    conf_level: float,
) -> None:
    delta, lower, upper, p = _paired(values, reference, conf_level)
    row[f"delta_vs_{comparison}"] = delta
    row[f"delta_vs_{comparison}_lower"] = lower
    row[f"delta_vs_{comparison}_upper"] = upper
    row[f"p_vs_{comparison}"] = p


# ----------------------------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------------------------


class _NumberFormat:
    """Format a metric's values with a number of decimals that suits their size."""

    def __init__(self, values: list[float]) -> None:
        size = max((abs(v) for v in values if v), default=1.0)
        self.decimals = max(0, min(6, 3 - math.floor(math.log10(size))))

    def number(self, x: float) -> str:
        return f"{x:,.{self.decimals}f}".replace("-", "−")

    def delta(self, x: float | None) -> str:
        if x is None:
            return "—"  # pragma: no cover
        return f"{x:+,.{self.decimals}f}".replace("-", "−")

    def value(self, mean: float, sd: float | None) -> str:
        if sd is None:
            return self.number(mean)
        return f"{self.number(mean)} ± {self.number(sd)}"


def _esc(text: Any) -> str:
    return html_lib.escape(str(text), quote=False)


def _code_html(text: str) -> str:
    """Escape text for HTML, showing the parts in backticks as code."""
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", _esc(text))


def _mono(text: str) -> str:
    return f"<span style='font-family: \"IBM Plex Mono\", monospace;'>{text}</span>"


def _small(text: str) -> str:
    return f"<div style='font-size: 10px; color: #777777;'>{text}</div>"


def _title_html(title: str, label: str | None) -> str:
    if label is None:
        return title
    return f"{title}<div style='font-size: 13px; color: #555555;'>{_esc(label)}</div>"


def _note_html(note: str) -> str:
    return f"<span style='font-size: 11px; color: #555555;'>{_code_html(note)}</span>"


def _frame(rows: list[dict[str, str]], columns: list[str]) -> Any:
    lib = _select_df_lib(preference="polars")
    if lib.__name__ == "polars":
        return lib.DataFrame(
            [[row[c] for c in columns] for row in rows], schema=columns, orient="row"
        )
    return lib.DataFrame([[row[c] for c in columns] for row in rows], columns=columns)


def _variant_html(variant: AblationVariant, steps: dict[int, Any], max_briefs: int = 3) -> str:
    """The variant's name, with a line for each step (its brief) below."""
    name = f"<div><strong>{_esc(variant.name)}</strong></div>"
    if variant.strategy == "full":
        return name + _small("Nothing removed")
    if variant.is_control:
        return name + _small("Rows removed at random, drawn anew for each repeat")

    lines = []
    for i in variant.steps[:max_briefs]:
        step = steps[i]
        brief = step.brief or step.autobrief or step.assertion_type
        text = _code_html(brief)
        lines.append(text if len(variant.steps) == 1 else f"{i}: {text}")
    if len(variant.steps) > max_briefs:
        lines.append(f"and {len(variant.steps) - max_briefs} more steps")
    return name + "".join(_small(line) for line in lines)


def _removed_html(n_removed: int, f_removed: float) -> str:
    pct = f_removed * 100
    if pct == 0:
        pct_text = "0%"
    elif pct < 0.1:
        pct_text = "<0.1%"
    else:
        pct_text = f"{pct:.1f}%"
    return f"<div>{_mono(f'{n_removed:,}')}</div>" + _small(_esc(pct_text))


def _badge(text: str, color: str) -> str:
    return (
        f"<span style='background: {color}; color: white; padding: 1px 6px; border-radius: 3px; "
        f"font-size: 10px; font-weight: bold; letter-spacing: 0.3px;'>{text}</span>"
    )


def _interval_svg(
    estimate: float, lower: float | None, upper: float | None, scale: float, color: str
) -> str:
    """A small plot of an interval around zero: a line for the interval and a dot for the estimate."""
    width, height = 100, 12
    half = width / 2

    def x(v: float) -> float:
        return half + v / scale * (half - 4)

    parts = [
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        "style='vertical-align: middle;'>",
        f"<line x1='{half}' y1='0' x2='{half}' y2='{height}' stroke='#BBBBBB' stroke-width='1'/>",
    ]
    if lower is not None and upper is not None:
        parts.append(
            f"<line x1='{x(lower):.1f}' y1='{height / 2}' x2='{x(upper):.1f}' y2='{height / 2}' "
            f"stroke='{color}' stroke-width='2'/>"
        )
    parts.append(f"<circle cx='{x(estimate):.1f}' cy='{height / 2}' r='3' fill='{color}'/>")
    parts.append("</svg>")
    return "".join(parts)


def _effect_html(row: dict[str, Any], comparison: str, fmt: _NumberFormat, scale: float) -> str:
    """The difference from the control (or full table), its interval, and the verdict badge."""
    delta = row[f"delta_vs_{comparison}"]
    lower = row[f"delta_vs_{comparison}_lower"]
    upper = row[f"delta_vs_{comparison}_upper"]
    verdict = row["verdict"]
    color = _VERDICT_COLORS.get(verdict, "#666666")

    text = fmt.delta(delta)
    if lower is not None and upper is not None:
        text += f" <span style='color: #777777;'>[{fmt.delta(lower)}, {fmt.delta(upper)}]</span>"
    plot = _interval_svg(delta, lower, upper, scale, color)
    badge = f" {_badge(verdict, color)}" if verdict is not None else ""
    return f"<div>{_mono(text)}</div><div style='white-space: nowrap;'>{plot}{badge}</div>"


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------


def _is_int(x: Any) -> bool:
    return isinstance(x, numbers.Integral) and not isinstance(x, bool)


def _is_number(x: Any) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _accepts_keyword(fn: Callable[..., Any], name: str) -> bool:
    """Can `fn` be called with the keyword argument `name`? (If unsure, assume it can.)"""
    try:
        parameters = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):  # pragma: no cover
        return True
    return any(
        p.kind is inspect.Parameter.VAR_KEYWORD
        or (p.name == name and p.kind is not inspect.Parameter.POSITIONAL_ONLY)
        for p in parameters
    )
