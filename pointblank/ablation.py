from __future__ import annotations

import csv
import hashlib
import html as html_lib
import inspect
import json
import math
import numbers
import os
import random
import re
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import narwhals as nw

from pointblank._utils import _select_df_lib, _with_row_index

if TYPE_CHECKING:
    from great_tables import GT

    from pointblank.validate import Validate

__all__ = [
    "Ablation",
    "AblationVariant",
    "Stratified",
    "sklearn_evaluator",
]

# The row number column used to match the rows that steps failed to the rows of the data
_INDEX = "pb_index_"

_CONTROLS = ("size_matched",)
_ADJUSTMENTS = ("holm",)
_EXECUTORS = ("thread", "process")
_EXPORT_FORMATS = ("parquet", "indices")

# `drop_groups()` needs `values=` for a column with more distinct values than this
_MAX_GROUPS = 50

# The relative difference between two results that `check_determinism()` takes as rounding
_ROUNDING = 1e-9

# The versions of the cache entries and of the manifest written by `Ablation.export()`
_CACHE_VERSION = 1
_MANIFEST_FORMAT = "pointblank-ablation"
_MANIFEST_VERSION = 1

_ML_INSTALL_HINT = '`pip install "pointblank[ml]"`'

# The parameters that `sweep_step()` can't vary: they don't change which rows fail a step, or they
# change which rows the step checks
_SWEEP_FIXED_PARAMS = ("thresholds", "actions", "brief", "active", "dimension", "pre", "segments")

_VERDICT_COLORS = {
    "helps": "#4CA64C",
    "hurts": "#CF142B",
    "inconclusive": "#999999",
    "redundant": "#E5AB00",
}

# The columns of `Ablation.get_effects()`, with their types
_EFFECT_COLUMNS = {
    "variant": "str",
    "name": "str",
    "strategy": "str",
    "steps": "list",
    "left_out": "int",
    "group": "str",
    "sweep": "str",
    "value": "float",
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
    "delta_added": "float",
    "delta_added_lower": "float",
    "delta_added_upper": "float",
    "p_added": "float",
    "p_adj_added": "float",
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
        `"steps_<i>_<j>"`) for [`drop_steps()`](`pointblank.Ablation.drop_steps`),
        `"all_but_<i>"` for [`drop_all_but_each()`](`pointblank.Ablation.drop_all_but_each`),
        `"<column>=<value>"` for [`drop_groups()`](`pointblank.Ablation.drop_groups`),
        `"<score>:lowest_<q>"` (or `"<score>:below_<value>"`, and the same with `highest` and
        `above`) for [`sweep_score()`](`pointblank.Ablation.sweep_score`), where `<score>` is
        `step_<i>` or the score column, `"step_<i>:<param>=<value>"` for
        [`sweep_step()`](`pointblank.Ablation.sweep_step`), and `"control_<n>"` for the control
        that removes `n` random rows (or `"control_<id>"` for the control of a
        `drop_all_but_each()` variant).
    strategy
        The kind of version: `"full"`, `"drop_steps"`, `"drop_each"`, `"drop_all"`,
        `"drop_all_but_each"`, `"drop_groups"`, `"sweep_score"`, `"sweep_step"`, or `"control"`.
    name
        A name for the version, as shown in reports (e.g., `"Step 2"`).
    steps
        The numbers of the validation steps whose failing rows were removed (empty for the full
        table, for groups, for the points of a sweep, and for most controls).
    n_removed
        The number of rows removed from the table.
    f_removed
        The share of the table's rows that were removed (from `0` to `1`).
    is_control
        Is this a control (random rows removed) rather than a variant?
    group
        For a variant of `drop_groups()`, the value of `group_column` in the rows it removes
        (`None` for the rows where it's missing).
    group_column
        For a variant of `drop_groups()`, the column the rows are grouped by (otherwise `None`).
    left_out
        For a variant of `drop_all_but_each()` (and its control), the step whose rows are kept
        (otherwise `None`).
    sweep
        For a point of a sweep (from `sweep_score()` or `sweep_step()`), the id of the sweep, which
        its points share (e.g., `"step_2:lowest"` or `"step_2:min_val"`; otherwise `None`).
    value
        For a point of a sweep, its value: the share of the scored rows removed, the score
        threshold, or the value of the step's parameter (otherwise `None`).
    """

    id: str
    strategy: str
    name: str
    steps: tuple[int, ...] = ()
    n_removed: int = 0
    f_removed: float = 0.0
    is_control: bool = False
    group: Any = None
    group_column: str | None = None
    left_out: int | None = None
    sweep: str | None = None
    value: float | None = None

    def _to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "strategy": self.strategy,
            "name": self.name,
            "steps": list(self.steps),
            "n_removed": self.n_removed,
            "f_removed": self.f_removed,
            "is_control": self.is_control,
            "group": _json_value(self.group),
            "group_column": self.group_column,
            "left_out": self.left_out,
            "sweep": self.sweep,
            "value": self.value,
        }


@dataclass(frozen=True)
class _Spec:
    """A variant as it was asked for, before the rows it removes are known."""

    id: str
    name: str
    strategy: str
    steps: tuple[int, ...] = ()
    left_out: int | None = None
    group_column: str | None = None
    group: Any = None
    sweep: str | None = None
    value: float | None = None

    @property
    def key(self) -> tuple[Any, ...]:
        """What the variant removes: specs with the same key remove the same rows."""
        if self.sweep is not None:
            # The rows of a sweep's point are found when it's added (`Ablation._sweep_rows`)
            return ("sweep", self.id)
        if self.group_column is not None:
            return ("group", self.group_column, self.group)
        if self.left_out is not None:
            return ("left_out", self.steps, self.left_out)
        return ("steps", self.steps)

    @property
    def base(self) -> tuple[int, ...]:
        """For a `drop_all_but_each()` variant, all the steps (including the one left out)."""
        return tuple(sorted((*self.steps, self.left_out)))


@dataclass
class _Resolved:
    """The variants of an ablation with the rows they remove, and the controls they need."""

    # The variants that are used, with the numbers of the rows they remove
    variants: list[tuple[AblationVariant, frozenset[int]]]
    removed: dict[str, frozenset[int]]
    # Notes about the variants that aren't used
    notes: list[str]
    # The variants that remove the same rows as an earlier version, which aren't evaluated again
    same_as: dict[str, str]
    controls: list[AblationVariant]
    # For each control: the rows it always removes, the number of random rows, and the key of
    # its random draws
    control_draws: dict[str, tuple[frozenset[int], int, str]]
    # The control of each variant
    variant_control: dict[str, str]
    # For each `drop_all_but_each()` variant, the variant that removes the rows of every step
    references: dict[str, str]
    # Every evaluation that a run makes (the version of the table and the repeat)
    tasks: list[tuple[AblationVariant, int]] | None = None


@dataclass(frozen=True)
class Stratified:
    """
    A control that removes random rows in proportion to groups of the table.

    By default, an [`Ablation`](`pointblank.Ablation`) compares each variant with a control that
    removes the same number of rows at random from anywhere in the table. When the table has
    groups that matter to the model (e.g., the source of each record, or the class of a
    classification target), a random draw can take more rows from one group than another just by
    chance. A stratified control removes rows from each group in proportion to its size, so the
    control has the same mix of groups as the full table, and the comparison with it isn't
    blurred by a change in that mix. Pass it to `control=` when creating the ablation.

    Parameters
    ----------
    by
        One or more columns of the training table that define the groups. With several columns,
        each combination of their values is a group, and missing values make a group of their
        own.

    Returns
    -------
    Stratified
        A `Stratified` object, for the `control=` parameter of [`Ablation`](`pointblank.Ablation`).

    Details
    -------
    Each repeat draws its rows anew. The number of rows drawn from each group is its share of the
    rows, rounded so that the total is the number the control removes (the groups with the
    largest remainders get the extra rows). A control for a
    [`drop_all_but_each()`](`pointblank.Ablation.drop_all_but_each`) variant draws in proportion
    to the rows that the variant keeps.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False)
    ```

    Here's a table of temperature readings from two sensors, where sensor `"b"` has a third of the
    readings and some of its readings are out of range. The 'model' estimates the average
    temperature, which is `20` degrees.

    ```{python}
    import pointblank as pb
    import polars as pl

    temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
    for k in range(2, 300, 24):
        temps[k] *= 10
    readings = pl.DataFrame(
        {"sensor": ["a", "a", "b"] * 100, "temp": temps}
    )

    def evaluate(train, *, seed):
        sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
        return abs(sample["temp"].mean() - 20)

    validation = (
        pb.Validate(data=readings)
        .col_vals_between(columns="temp", left=-10, right=45)
        .interrogate()
    )

    ablation = pb.Ablation(
        validation,
        evaluate=evaluate,
        metric="error",
        higher_is_better=False,
        control=pb.ablation.Stratified(by="sensor"),
    ).drop_each()

    ablation.plan()
    ```

    The control removes as many rows as the variant, two thirds of them from sensor `"a"` and one
    third from sensor `"b"`, whatever the variant removes.
    """

    by: str | list[str] | tuple[str, ...]

    def __post_init__(self) -> None:
        by = [self.by] if isinstance(self.by, str) else self.by
        if (
            not isinstance(by, (list, tuple))
            or not by
            or not all(isinstance(c, str) and c for c in by)
        ):
            raise TypeError("`by=` must be a column name or a list of column names.")
        if len(set(by)) != len(by):
            raise ValueError("The columns in `by=` must be different.")
        object.__setattr__(self, "by", tuple(by))

    def __repr__(self) -> str:
        by = self.by[0] if len(self.by) == 1 else list(self.by)
        return f"Stratified(by={by!r})"


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
    [`drop_all()`](`pointblank.Ablation.drop_all`),
    [`drop_steps()`](`pointblank.Ablation.drop_steps`),
    [`drop_all_but_each()`](`pointblank.Ablation.drop_all_but_each`),
    [`drop_groups()`](`pointblank.Ablation.drop_groups`), and the sweeps
    [`sweep_score()`](`pointblank.Ablation.sweep_score`) and
    [`sweep_step()`](`pointblank.Ablation.sweep_step`)), and nothing is evaluated until
    [`run()`](`pointblank.Ablation.run`) is called. Use [`plan()`](`pointblank.Ablation.plan`)
    before that to see how many rows each variant removes and how many evaluations the run needs.
    When training happens elsewhere (e.g., on a cluster), [`export()`](`pointblank.Ablation.export`)
    writes out the versions of the table to train on, and
    [`load_results()`](`pointblank.Ablation.load_results`) reads back the metrics.

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
        fixed holdout set that the function holds on to: it must not depend on `train`. For
        scikit-learn models, [`sklearn_evaluator()`](`pointblank.ablation.sklearn_evaluator`)
        makes this function for you. It can be left out (`None`) if the evaluations are done
        elsewhere, with [`export()`](`pointblank.Ablation.export`) and
        [`load_results()`](`pointblank.Ablation.load_results`).
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
        control. A [`Stratified`](`pointblank.ablation.Stratified`) control (e.g.,
        `pb.ablation.Stratified(by="source")`) removes the random rows in proportion to the groups
        of one or more columns. Use `None` to compare variants to the full table only.
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
    evaluate_version
        An optional version of the `evaluate=` function (e.g., `"v2"`, or a commit hash). It's
        recorded with the results and is part of the cache key, so changing it when the model or
        its training changes makes the cache evaluate everything again.
    cache
        A directory to keep the results of evaluations in. Each result is saved as soon as it's
        made, under a key made from the data, the rows removed, the seed, and `evaluate_version=`.
        An evaluation with the same key is never repeated, so a run that stops partway can be
        started again without losing work, and adding a variant only evaluates what's new. The
        `evaluate=` function itself isn't part of the key, so change `evaluate_version=` (or use
        another directory) when it changes.

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

    The variants of [`drop_all_but_each()`](`pointblank.Ablation.drop_all_but_each`) are judged
    on what their left-out step adds on top of the others instead (the `delta_added` columns),
    and can also be `"redundant"`: the step's rows help on their own, but make no clear
    difference once the other steps' rows are removed.

    With fewer than 3 repeats, there are no verdicts. The intervals aren't adjusted for multiple
    comparisons, so an interval can leave out zero while the verdict is `"inconclusive"`.

    A variant that removes no rows (e.g., a step that no row failed) is left out of the ablation,
    with a note in the reports. Steps that can't be matched to the rows of the table, such as
    steps with a `pre=` function not marked with [`row_preserving()`](`pointblank.row_preserving`),
    are left out too.

    Before a long run, [`check_determinism()`](`pointblank.Ablation.check_determinism`) checks that
    `evaluate=` gives the same result for the same table and seed, which the paired comparisons
    rely on.

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
    yaml_ablation : Set up an ablation from a YAML configuration.
    """

    def __init__(
        self,
        validation: Validate,
        evaluate: Callable[..., Any] | None = None,
        metric: str | None = None,
        higher_is_better: bool = True,
        repeats: int = 5,
        seed: int = 0,
        control: str | Stratified | None = "size_matched",
        duplicates: str | None = "keep_first",
        duplicates_order_by: str | list[str] | None = None,
        max_removed: float = 0.5,
        conf_level: float = 0.95,
        adjust: str | None = "holm",
        label: str | None = None,
        evaluate_version: str | None = None,
        cache: str | os.PathLike[str] | None = None,
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
        if evaluate is not None and not callable(evaluate):
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
        if control is not None and not isinstance(control, Stratified) and control not in _CONTROLS:
            raise ValueError(
                f"`control=` must be one of {', '.join(repr(c) for c in _CONTROLS)}, a "
                f"`Stratified` control, or `None`; got {control!r}."
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
        if evaluate_version is not None and not isinstance(evaluate_version, str):
            raise TypeError(
                "`evaluate_version=` must be a string, got "
                f"{type(evaluate_version).__name__} instead."
            )
        if cache is not None and not isinstance(cache, (str, os.PathLike)):
            raise TypeError(
                f"`cache=` must be the path of a directory, got {type(cache).__name__} instead."
            )
        if cache is not None and Path(cache).exists() and not Path(cache).is_dir():
            raise ValueError(f"`cache=` must be a directory, but {str(cache)!r} is a file.")

        columns = nw.from_native(validation.data).collect_schema().names()
        if isinstance(control, Stratified):
            missing = [c for c in control.by if c not in columns]
            if missing:
                raise ValueError(
                    "The `Stratified` control's columns aren't in the table: "
                    f"{', '.join(map(repr, missing))}."
                )
        self._columns = columns
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
        self.evaluate_version = evaluate_version
        self.cache = Path(cache) if cache is not None else None

        self._n_rows = get_row_count(validation.data)
        self._steps = {step.i: step for step in steps}
        self._skipped = dict(skipped)
        self._pass_variant = evaluate is not None and _accepts_keyword(evaluate, "variant")

        self._specs: list[_Spec] = []
        self._notes: list[str] = []
        self._resolved: _Resolved | None = None
        # The rows of each group, by column (for `drop_groups()`), and of each stratum (for a
        # `Stratified` control)
        self._groups: dict[str, dict[Any, frozenset[int]]] = {}
        self._strata: list[list[int]] | None = None
        # The sweeps (by id: their label and the label of their values), and for each of their
        # points, the rows it removes and a line describing them
        self._sweeps: dict[str, dict[str, Any]] = {}
        self._sweep_rows: dict[str, frozenset[int]] = {}
        self._sweep_details: dict[str, str] = {}
        self._data_fingerprint: str | None = None
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

    def drop_all_but_each(self, steps: list[int] | None = None) -> Ablation:
        """
        Add a variant for each step, removing the rows that failed every other step.

        The variants of [`drop_each()`](`pointblank.Ablation.drop_each`) say whether each check
        helps on its own. This strategy asks a different question: 'does this check add anything,
        once the other checks have been applied?' Two checks can find mostly the same rows, so
        that each helps on its own but either one makes the other redundant. Each variant leaves
        out one step: it removes the rows that failed any of the other steps, and is compared with
        the variant that removes the rows of every step (which this method adds too, as with
        [`drop_all()`](`pointblank.Ablation.drop_all`)).

        Parameters
        ----------
        steps
            The step numbers to use (at least two). By default, every row-level step that the
            ablation can use is included.

        Returns
        -------
        Ablation
            The `Ablation` object, with the variants added.

        Details
        -------
        The effect of a left-out step is what its rows add on top of the other steps: the
        difference between removing the rows of every step and removing the rows of the other
        steps (`delta_added` in [`get_effects()`](`pointblank.Ablation.get_effects`)). Removing
        the step's rows also means removing more rows, so with a control (the default), the
        comparison is with removing the rows of the other steps plus as many random rows as the
        step adds. Each variant has a control of its own for that.

        The verdict is `"helps"` or `"hurts"` when the step's rows make a clear difference on top
        of the others. When they don't, but the step's rows help on their own (its variant from
        `drop_each()` helps), the verdict is `"redundant"`: the other steps already remove what
        matters. So it's best to use this strategy together with `drop_each()`.

        A variant that removes the same rows as another version of the table (e.g., when every
        row the step failed also failed another step) isn't evaluated again, and the results of
        that version are used.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's a table of temperature readings where some readings were recorded in tenths of a
        degree, so they're ten times too high. Step 1 finds all of them, while step 2 has a looser
        limit and only finds the highest ones.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_between(columns="temp", left=-10, right=45)
            .col_vals_lt(columns="temp", value=200)
            .interrogate()
        )

        ablation = (
            pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
            .drop_each()
            .drop_all_but_each()
            .run()
        )

        ablation
        ```

        Both steps help on their own. On top of step 2, step 1 still helps, since it removes the
        bad readings that step 2 misses. But step 2 is redundant: every row it finds also fails
        step 1, so leaving it out removes the same rows as using both steps.
        """
        if steps is None:
            base = self._all_steps(method="drop_all_but_each")
        else:
            base = self._check_steps(steps, method="drop_all_but_each")
        if len(base) < 2:
            raise ValueError(
                "`drop_all_but_each()` needs at least two steps, so that there are other steps "
                "to compare each step with."
            )

        self.drop_all(None if steps is None else list(base))
        numbers = ", ".join(str(i) for i in base)
        for k in base:
            rest = tuple(i for i in base if i != k)
            if steps is None:
                spec_id, name = f"all_but_{k}", f"All steps but step {k}"
            else:
                spec_id = f"all_but_{k}_of_" + "_".join(str(i) for i in base)
                name = f"Steps {numbers} but step {k}"
            self._add_spec(
                _Spec(id=spec_id, name=name, strategy="drop_all_but_each", steps=rest, left_out=k)
            )
        return self

    def drop_groups(self, column: str, values: Any = None) -> Ablation:
        """
        Add a variant for each group of rows, removing the rows of that group.

        The other strategies remove the rows that failed validation steps. This one removes the
        rows with a value in a column (e.g., every record from one source, region, or batch), to
        find out whether a part of the data helps or hurts the model. A group that hurts is worth
        a closer look: it may have a problem that no check has found yet.

        Parameters
        ----------
        column
            The column whose values define the groups.
        values
            The values of `column=` to make variants for: one value or a list of them (`None` in
            a list stands for the rows where the column is missing). By default, every value in
            the column gets a variant, as long as there are no more than 50 of them.

        Returns
        -------
        Ablation
            The `Ablation` object, with the variants added.

        Details
        -------
        The variants are named after the column and value (e.g., `"source = web"`, with the id
        `"source=web"`), and the rows where the column is missing are a group of their own
        (`"source is missing"`). Like any other variant, a group is compared with removing the
        same number of random rows. A group with most of the table's rows gives a warning (see
        `max_removed=`), since without it the training table is mostly a different dataset.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's a table of temperature readings from three sensors, where sensor `"c"` reads five
        degrees too high. No check of the validation finds this, since the readings are all in the
        expected range.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        sensors = ["a", "b", "c"] * 100
        temps = [t + 5 if s == "c" else t for t, s in zip(temps, sensors)]
        readings = pl.DataFrame({"sensor": sensors, "temp": temps})

        def evaluate(train, *, seed):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_between(columns="temp", left=-10, right=45)
            .interrogate()
        )

        ablation = (
            pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
            .drop_groups("sensor")
            .run()
        )

        ablation
        ```

        Removing the readings of sensor `"c"` helps, while removing either of the other sensors
        hurts, since it leaves a larger share of the readings from sensor `"c"`.
        """
        if not isinstance(column, str):
            raise TypeError(f"`column=` must be a column name, got {type(column).__name__}.")
        if column not in self._columns:
            raise ValueError(f"The column {column!r} isn't in the table.")

        groups = self._group_rows(column)
        if values is None:
            values = list(groups)
            if len(values) > _MAX_GROUPS:
                raise ValueError(
                    f"The column {column!r} has {len(values):,} distinct values, more than "
                    f"{_MAX_GROUPS}. Choose the groups to use with `values=`."
                )
        elif isinstance(values, (list, tuple, set)):
            values = list(values)
            if not values:
                raise ValueError("`values=` must have at least one value.")
        else:
            values = [values]

        for value in values:
            try:
                hash(value)
            except TypeError:
                raise TypeError(f"The values in `values=` must be single values, got {value!r}.")
            if value is None:
                spec_id, name = f"{column}=null", f"{column} is missing"
            else:
                spec_id, name = f"{column}={value}", f"{column} = {value}"
            self._add_spec(
                _Spec(
                    id=spec_id,
                    name=name,
                    strategy="drop_groups",
                    group_column=column,
                    group=value,
                )
            )
        return self

    def sweep_score(
        self,
        score: int | str,
        quantiles: list[float] | None = None,
        values: list[float] | None = None,
        remove: str | None = None,
    ) -> Ablation:
        """
        Add a variant for each cutoff of a score, removing the rows on the wrong side of it.

        A score (e.g., from a quality classifier) doesn't say where to draw the line between the
        rows to keep and those to remove. This strategy measures it: each variant removes more of
        the lowest-scoring rows than the last, so the results trace how the model changes as the
        cutoff gets stricter (a dose-response curve). The report plots the curve, along with what
        removing as many random rows does. The cutoffs are given as shares of the scored rows to
        remove (`quantiles=`) or as score values (`values=`).

        Parameters
        ----------
        score
            Where the scores come from: the number of a
            [`col_vals_scored()`](`pointblank.Validate.col_vals_scored`) step that kept its scores
            (`keep_scores=True`, the default), or the name of a numeric column of the table.
        quantiles
            The shares of the scored rows to remove, each greater than `0` and at most `1` (e.g.,
            `0.1` removes the tenth of the rows with the lowest scores). This is the default, with
            the shares `0.05`, `0.1`, `0.2`, `0.3`, `0.4`, and `0.5`.
        values
            Score cutoffs to use instead of `quantiles=`: each variant removes the rows that score
            below the cutoff (or above it, with `remove="highest"`).
        remove
            Which rows to remove: `"lowest"` (the lowest-scoring rows) or `"highest"`. By default,
            it's `"highest"` for a step with only `max_val=` (e.g., a toxicity or perplexity score,
            where a high score is bad) and `"lowest"` otherwise.

        Returns
        -------
        Ablation
            The `Ablation` object, with the variants added.

        Details
        -------
        The rows without a score (a missing value, a row the scorer gave no score, or a row
        outside a segmented step's segment) are never removed, and a note says how many there are.
        With `quantiles=`, the rows are ranked by their score, and ties are broken by row order,
        so each share removes a set number of rows. Each point of the sweep is a variant like any
        other: it's compared with removing the same number of random rows and gets a verdict. A
        point that removes no rows is left out, and a point that removes the same rows as another
        version of the table isn't evaluated again.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's a table of temperature readings, some of them recorded in tenths of a degree, and a
        scorer that rates how plausible each reading is (from `0` to `1`). The 'model' estimates
        the average temperature and is evaluated by its error against the true average of `20`.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        readings = pl.DataFrame({"temp": temps})

        def plausibility(rows):
            return [1 / (1 + abs(row["temp"] - 20) / 5) for row in rows]

        def evaluate(train, *, seed):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_scored(columns="temp", scorer=plausibility, min_val=0.5)
            .interrogate()
        )

        ablation = (
            pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
            .sweep_score(1, quantiles=[0.02, 0.04, 0.1, 0.2])
            .run()
        )

        ablation
        ```

        Removing the lowest-scoring 4% of the rows takes out every bad reading, and the curve below
        the report shows the error dropping to near zero there. Removing more rows than that makes
        no further difference.
        """
        from pointblank.validate import _has_kept_scores

        if isinstance(score, bool) or not isinstance(score, (int, str)):
            raise TypeError(
                "`score=` must be the number of a `col_vals_scored()` step or the name of a "
                f"column, got {type(score).__name__}."
            )
        if quantiles is not None and values is not None:
            raise ValueError("Use either `quantiles=` or `values=`, not both.")
        if remove is not None and remove not in ("lowest", "highest"):
            raise ValueError(f"`remove=` must be 'lowest' or 'highest', got {remove!r}.")

        if isinstance(score, str):
            if score not in self._columns:
                raise ValueError(f"The column {score!r} isn't in the table.")
            scores = self._column_scores(score)
            base_id, base_name = score, score
            remove = remove or "lowest"
        else:
            if score not in self._steps:
                raise ValueError(
                    f"`sweep_score()` can't use step {score}: {self._ineligible_reason(score)}."
                )
            step = self._steps[score]
            if not _has_kept_scores(step):
                raise ValueError(
                    f"`sweep_score()` can't use step {score}: it isn't a `col_vals_scored()` step "
                    "that kept its scores (with `keep_scores=True`)."
                )
            scores = self._step_scores(score)
            base_id, base_name = f"step_{score}", f"Step {score}"
            if remove is None:
                bounds = step.values or {}
                only_max = bounds.get("min_val") is None and bounds.get("max_val") is not None
                remove = "highest" if only_max else "lowest"

        if values is None:
            quantiles = self._check_sweep_values(
                quantiles if quantiles is not None else [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                "quantiles",
            )
            for q in quantiles:
                if not 0 < q <= 1:
                    raise ValueError(
                        f"The values of `quantiles=` must be greater than 0 and at most 1, got {q!r}."
                    )
        else:
            values = self._check_sweep_values(values, "values")

        n_scored = len(scores)
        n_missing = self._n_rows - n_scored
        if n_missing:
            label = f"step {score}" if isinstance(score, int) else f"the column `{score}`"
            self._add_note(
                f"1 row has no score in {label}, so the sweep never removes it."
                if n_missing == 1
                else f"{n_missing:,} rows have no score in {label}, so the sweep never removes "
                "them."
            )

        highest = remove == "highest"
        if values is None:
            sweep_id = f"{base_id}:{remove}"
            self._sweeps[sweep_id] = {
                "label": f"{base_name}, removing the {remove}-scoring rows",
                "axis": "Share of the scored rows removed",
                "percent": True,
            }
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1] if highest else kv[1], kv[0]))
            for q in quantiles:
                n = math.floor(q * n_scored + 0.5)
                pct = f"{q * 100:g}%"
                spec_id = f"{sweep_id}_{_number_text(q)}"
                self._sweep_rows[spec_id] = frozenset(k for k, _ in ranked[:n])
                self._sweep_details[spec_id] = (
                    f"The {n:,} rows with the {remove} scores (of {n_scored:,} scored rows)"
                )
                self._add_spec(
                    _Spec(
                        id=spec_id,
                        name=f"{base_name}: {remove} {pct} of scores",
                        strategy="sweep_score",
                        sweep=sweep_id,
                        value=float(q),
                    )
                )
        else:
            side = "above" if highest else "below"
            sweep_id = f"{base_id}:{side}"
            self._sweeps[sweep_id] = {
                "label": f"{base_name}, removing the rows scoring {side} each cutoff",
                "axis": "Score cutoff",
                "percent": False,
            }
            for v in values:
                text = _number_text(v)
                spec_id = f"{sweep_id}_{text}"
                self._sweep_rows[spec_id] = frozenset(
                    k for k, x in scores.items() if (x > v if highest else x < v)
                )
                self._sweep_details[spec_id] = f"The rows with a score {side} {text}"
                self._add_spec(
                    _Spec(
                        id=spec_id,
                        name=f"{base_name}: scores {side} {text}",
                        strategy="sweep_score",
                        sweep=sweep_id,
                        value=float(v),
                    )
                )
        return self

    def sweep_step(self, step: int, param: str, values: list[float]) -> Ablation:
        """
        Add a variant for each value of a step's parameter, removing the rows that fail it.

        A check's bounds are often picked by hand: is `min_val=0.5` the right cutoff, or would
        `0.3` keep more useful rows? This strategy measures it. For each value, the step is run
        again on the table with `param=` set to that value, and the variant removes the rows that
        fail it. The results trace how the model changes as the check gets stricter, and the
        report plots the curve, along with what removing as many random rows does.

        Parameters
        ----------
        step
            The number of the step to vary. It must be a row-level step that the ablation can use
            (active, and without a `pre=` function unless that function is marked with
            [`row_preserving()`](`pointblank.row_preserving`)).
        param
            The name of the parameter of the step's validation method to vary (e.g., `"min_val"`
            for [`col_vals_scored()`](`pointblank.Validate.col_vals_scored`), or `"right"` for
            [`col_vals_between()`](`pointblank.Validate.col_vals_between`)). It can't be one of
            the parameters that don't change which rows fail (`thresholds=`, `actions=`, `brief=`,
            `active=`, and `dimension=`), nor `pre=` or `segments=`.
        values
            The values of `param=` to use, as numbers.

        Returns
        -------
        Ablation
            The `Ablation` object, with the variants added.

        Details
        -------
        The step is run again as it was set up, with only `param=` changed, so a
        `col_vals_scored()` step takes its scores from the score cache instead of scoring the rows
        again. For a segmented step, only the step's own segment is checked. A value that the
        validation method doesn't accept (e.g., a `min_val=` above the step's `max_val=`) gives an
        error. As with any other variant, a value whose step fails no rows is left out, and a value
        that removes the same rows as another version of the table isn't evaluated again.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's a table of temperature readings, some of them recorded in tenths of a degree, with
        a check that the readings are below an upper limit. The 'model' estimates the average
        temperature and is evaluated by its error against the true average of `20`.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_lt(columns="temp", value=100)
            .interrogate()
        )

        ablation = (
            pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
            .sweep_step(1, param="value", values=[21, 50, 190, 220])
            .run()
        )

        ablation
        ```

        A limit of `50` removes every bad reading. A limit of `21` also removes many good readings
        (the warmer ones), which leaves the average too low, and the limits of `190` and `220` miss
        some of the bad readings.
        """
        from pointblank.validate import Validate, _validation_info_to_step

        if not _is_int(step):
            raise TypeError(f"`step=` must be a step number, got {step!r}.")
        if step not in self._steps:
            raise ValueError(
                f"`sweep_step()` can't use step {step}: {self._ineligible_reason(step)}."
            )
        if not isinstance(param, str) or not param:
            raise TypeError("`param=` must be the name of a parameter of the step's method.")
        values = self._check_sweep_values(values, "values")

        info = self._steps[step]
        found = _validation_info_to_step(info, None, None, [], keep_objects=True)
        if found is None:  # pragma: no cover
            raise ValueError(f"`sweep_step()` can't run step {step} again.")
        method, kwargs = found
        parameters = inspect.signature(getattr(Validate, method)).parameters
        if param in _SWEEP_FIXED_PARAMS or param == "self" or param not in parameters:
            raise ValueError(
                f"`sweep_step()` can't vary {param!r}: it isn't a parameter of `{method}()` that "
                "changes which rows fail."
            )
        # The step is checked on its own: without actions (which could act on the results), and
        # on the segment it was checked on (not every segment of its `segments=`)
        for key in ("thresholds", "actions", "brief", "active", "dimension"):
            kwargs.pop(key, None)
        if info.segments is not None:
            kwargs["segments"] = info.segments

        sweep_id = f"step_{step}:{param}"
        self._sweeps[sweep_id] = {
            "label": f"Step {step}, varying {param}",
            "axis": f"{param} (step {step})",
            "percent": False,
        }
        for v in values:
            text = _number_text(v)
            spec_id = f"{sweep_id}={text}"
            self._sweep_rows[spec_id] = self._rerun_failed_rows(
                method, {**kwargs, param: v}, step=step, setting=f"{param}={text}"
            )
            self._sweep_details[spec_id] = f"The rows that fail step {step} with `{param}={text}`"
            self._add_spec(
                _Spec(
                    id=spec_id,
                    name=f"Step {step} with {param}={text}",
                    strategy="sweep_step",
                    sweep=sweep_id,
                    value=float(v),
                )
            )
        return self

    def _check_sweep_values(self, values: Any, arg: str) -> list[float]:
        if not isinstance(values, (list, tuple, range)):
            raise TypeError(f"`{arg}=` must be a list of numbers, got {type(values).__name__}.")
        values = list(values)
        if not values:
            raise ValueError(f"`{arg}=` must have at least one value.")
        for v in values:
            if not _is_number(v) or not math.isfinite(float(v)):
                raise TypeError(f"The values of `{arg}=` must be finite numbers, got {v!r}.")
        if len(set(values)) != len(values):
            raise ValueError(f"The values of `{arg}=` must be different from each other.")
        return values

    def _step_scores(self, i: int) -> dict[int, float]:
        """The scores of a `col_vals_scored()` step, by row number (rows with no score left out)."""
        from pointblank.validate import _step_row_results

        data_nw = _with_row_index(self.validation.data, name=_INDEX)
        results = _step_row_results(
            self._steps[i],
            data_nw,
            index_name=_INDEX,
            duplicates=self.duplicates,
            duplicates_order_by=self._order_by,
            extra=("pb_score_",),
        ).select(_INDEX, "pb_score_")
        if isinstance(results, nw.LazyFrame):
            results = results.collect()
        return _scores_by_row(results[_INDEX].to_list(), results["pb_score_"].to_list())

    def _column_scores(self, column: str) -> dict[int, float]:
        """The values of a numeric column, by row number (rows with a missing value left out)."""
        frame = _with_row_index(self.validation.data, name=_INDEX).select(_INDEX, column)
        if not frame.collect_schema()[column].is_numeric():
            raise TypeError(f"The column {column!r} must be numeric to be used as a score.")
        if isinstance(frame, nw.LazyFrame):
            frame = frame.collect()
        return _scores_by_row(frame[_INDEX].to_list(), frame[column].to_list())

    def _rerun_failed_rows(
        self, method: str, kwargs: dict[str, Any], step: int, setting: str
    ) -> frozenset[int]:
        """Run a step again on the table with other settings, and get the rows that fail it."""
        from pointblank.validate import Validate, _step_row_results

        try:
            validation = getattr(Validate(data=self.validation.data), method)(**kwargs)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Step {step} can't be run with `{setting}`: {e}") from e
        validation.interrogate(collect_extracts=False)

        data_nw = _with_row_index(self.validation.data, name=_INDEX)
        removed: set[int] = set()
        for info in validation.validation_info:
            if info.eval_error or info.tbl_checked is None:
                raise ValueError(f"Step {step} couldn't be evaluated with `{setting}`.")
            results = (
                _step_row_results(
                    info,
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
            removed.update(int(k) for k in results[_INDEX].to_list())
        return frozenset(removed)

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
                if existing.key == spec.key:
                    return
                raise ValueError(f"There's already a variant called {spec.id!r}.")
        for existing in self._specs:
            if existing.key == spec.key:
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

    def _resolve(self) -> _Resolved:
        """
        Find the rows each variant removes and the controls the variants need, leaving out the
        variants that remove no rows (or every row), and give the notes about them.
        """
        if self._resolved is not None:
            return self._resolved

        failed = self._failed_rows(sorted({i for spec in self._specs for i in spec.steps}))
        n_rows = self._n_rows
        notes: list[str] = []

        usable = []
        for spec in self._specs:
            if spec.sweep is not None:
                removed = self._sweep_rows[spec.id]
            elif spec.group_column is not None:
                removed = self._group_rows(spec.group_column).get(spec.group, frozenset())
            else:
                removed = frozenset().union(*(failed[i] for i in spec.steps))
            if len(removed) >= n_rows:
                notes.append(f"{spec.name} isn't used: it removes every row.")
                continue
            # A `drop_all_but_each()` variant can remove no rows (when the left-out step is the
            # only one that rows failed): it's then the same as the full table
            if not removed and spec.left_out is None:
                notes.append(f"{spec.name} isn't used: it removes no rows.")
                continue
            usable.append((spec, removed))

        by_key = {spec.key: spec for spec, _ in usable}
        variants = []
        references = {}
        for spec, removed in usable:
            if spec.left_out is not None:
                reference = by_key.get(("steps", spec.base))
                if reference is None:
                    reference = next(s for s in self._specs if s.key == ("steps", spec.base))
                    notes.append(
                        f"{spec.name} isn't used: it's compared with {reference.name}, which isn't "
                        "used."
                    )
                    continue
                references[spec.id] = reference.id
            n_removed = len(removed)
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
                group=spec.group,
                group_column=spec.group_column,
                left_out=spec.left_out,
                sweep=spec.sweep,
                value=spec.value,
            )
            variants.append((variant, removed))

        # A `drop_all_but_each()` variant or a sweep's point that removes the same rows as another
        # version of the table isn't evaluated again
        same_as = {}
        first_with_rows: dict[frozenset[int], str] = {}
        for variant, removed in variants:
            if variant.left_out is None and variant.sweep is None:
                first_with_rows.setdefault(removed, variant.id)
        for variant, removed in variants:
            if variant.left_out is None and variant.sweep is None:
                continue
            if not removed:
                same_as[variant.id] = "full"
            elif removed in first_with_rows:
                same_as[variant.id] = first_with_rows[removed]
            else:
                first_with_rows[removed] = variant.id

        controls = []
        control_draws = {}
        variant_control = {}
        removed_by_id = {variant.id: removed for variant, removed in variants}
        if self.control is not None:
            for n in sorted({v.n_removed for v, _ in variants if v.left_out is None}):
                control = AblationVariant(
                    id=f"control_{n}",
                    strategy="control",
                    name=f"{n:,} random rows",
                    n_removed=n,
                    f_removed=n / n_rows,
                    is_control=True,
                )
                controls.append(control)
                control_draws[control.id] = (frozenset(), n, str(n))
            for variant, removed in variants:
                if variant.left_out is None:
                    variant_control[variant.id] = f"control_{variant.n_removed}"
                    continue
                # The control of a `drop_all_but_each()` variant removes its rows, plus as many
                # random rows as the left-out step adds
                reference = removed_by_id[references[variant.id]]
                n_random = len(reference) - len(removed)
                if n_random == 0:
                    continue
                if not removed:
                    variant_control[variant.id] = f"control_{n_random}"
                    continue
                control = AblationVariant(
                    id=f"control_{variant.id}",
                    strategy="control",
                    name=f"{variant.name}, plus {n_random:,} random rows",
                    steps=variant.steps,
                    n_removed=len(reference),
                    f_removed=len(reference) / n_rows,
                    is_control=True,
                    left_out=variant.left_out,
                )
                controls.append(control)
                control_draws[control.id] = (removed, n_random, f"{variant.id}:{n_random}")
                variant_control[variant.id] = control.id

        self._resolved = _Resolved(
            variants=variants,
            removed=removed_by_id,
            notes=notes,
            same_as=same_as,
            controls=controls,
            control_draws=control_draws,
            variant_control=variant_control,
            references=references,
        )
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

    def _group_rows(self, column: str) -> dict[Any, frozenset[int]]:
        """The numbers of the rows with each value of a column, in the order of the values."""
        if column not in self._groups:
            frame = _with_row_index(self.validation.data, name=_INDEX).select(_INDEX, column)
            if isinstance(frame, nw.LazyFrame):
                frame = frame.collect()
            buckets: dict[Any, list[int]] = {}
            try:
                for k, value in zip(frame[_INDEX].to_list(), frame[column].to_list()):
                    buckets.setdefault(_group_value(value), []).append(int(k))
            except TypeError:
                raise TypeError(
                    f"The rows can't be grouped by the column {column!r}: its values aren't "
                    "single values."
                ) from None
            self._groups[column] = {v: frozenset(buckets[v]) for v in _sorted_values(buckets)}
        return self._groups[column]

    def _strata_rows(self) -> list[list[int]]:
        """The numbers of the rows in each group of a `Stratified` control."""
        if self._strata is None:
            by = list(self.control.by)
            frame = _with_row_index(self.validation.data, name=_INDEX).select(_INDEX, *by)
            if isinstance(frame, nw.LazyFrame):
                frame = frame.collect()
            buckets: dict[Any, list[int]] = {}
            columns = [frame[c].to_list() for c in by]
            try:
                for k, *values in zip(frame[_INDEX].to_list(), *columns):
                    key = tuple(_group_value(v) for v in values)
                    buckets.setdefault(key, []).append(int(k))
            except TypeError:
                raise TypeError(
                    "The rows can't be grouped by the columns of the `Stratified` control: their "
                    "values aren't single values."
                ) from None
            self._strata = [sorted(buckets[key]) for key in _sorted_values(buckets)]
        return self._strata

    def _control_rows(self, control_id: str, repeat: int) -> frozenset[int]:
        """The rows that a control removes in a repeat."""
        always, n_random, key = self._resolve().control_draws[control_id]
        stratified = isinstance(self.control, Stratified)
        if stratified:
            key = f"stratified:{json.dumps(list(self.control.by))}:{key}"
        rng = random.Random(f"pointblank-ablation:{self.seed}:{repeat}:{key}")

        if stratified:
            pools = self._strata_rows()
            if always:
                pools = [[k for k in pool if k not in always] for pool in pools]
            picked = []
            for pool, quota in zip(pools, _allocate(n_random, [len(p) for p in pools])):
                if quota:
                    picked.extend(rng.sample(pool, quota))
            return always | frozenset(picked)

        if not always:
            return frozenset(rng.sample(range(self._n_rows), n_random))
        candidates = [k for k in range(self._n_rows) if k not in always]
        return always | frozenset(rng.sample(candidates, n_random))

    def _tasks(self) -> list[tuple[AblationVariant, int]]:
        """
        Every evaluation that a run makes, in order: the full table, each variant, and each
        control, `repeats=` times each (a variant that's the same as an earlier version isn't
        evaluated).
        """
        resolved = self._resolve()
        if resolved.tasks is None:
            repeats = range(self.repeats)
            tasks = [(self._full_variant(), r) for r in repeats]
            for variant, _ in resolved.variants:
                if variant.id not in resolved.same_as:
                    tasks += [(variant, r) for r in repeats]
            for control in resolved.controls:
                tasks += [(control, r) for r in repeats]
            resolved.tasks = tasks
        return resolved.tasks

    def _removed_rows(self, variant: AblationVariant, repeat: int) -> frozenset[int]:
        """The rows that a version of the table removes in a repeat."""
        if variant.is_control:
            return self._control_rows(variant.id, repeat)
        if variant.strategy == "full":
            return frozenset()
        return self._resolve().removed[variant.id]

    def _training_frame(self, removed: frozenset[int]) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
        data_nw = _with_row_index(self.validation.data, name=_INDEX)
        if removed:
            data_nw = data_nw.filter(~nw.col(_INDEX).is_in(sorted(removed)))
        return data_nw.drop(_INDEX)

    def _training_table(self, removed: frozenset[int]) -> Any:
        """The target table without the `removed` rows, of the same type as the target table."""
        return self._training_frame(removed).to_native()

    # ------------------------------------------------------------------------------------------
    # Fingerprints and the cache
    # ------------------------------------------------------------------------------------------

    def _fingerprint(self) -> str:
        """A hash of the training table's schema and values."""
        if self._data_fingerprint is None:
            self._data_fingerprint = _table_fingerprint(self.validation.data)
        return self._data_fingerprint

    def _plan_fingerprint(self) -> str:
        """A hash of the training table and of every evaluation that a run makes."""
        settings = self._settings()
        payload = {
            "data": self._fingerprint(),
            "settings": {
                key: settings[key]
                for key in ("repeats", "seed", "control", "duplicates", "duplicates_order_by")
            },
            "tasks": [
                [variant.id, r, self.seed + r, _rows_hash(self._removed_rows(variant, r))]
                for variant, r in self._tasks()
            ],
            "same_as": self._resolve().same_as,
        }
        return _hash_json(payload)

    def _cache_key(self, removed: frozenset[int], seed: int) -> str:
        return _hash_json(
            {
                "version": _CACHE_VERSION,
                "data": self._fingerprint(),
                "n_rows": self._n_rows,
                "removed": _rows_hash(removed),
                "seed": seed,
                "evaluate_version": self.evaluate_version,
            }
        )

    def _cache_get(self, key: str) -> Any:
        """The result of an evaluation in the cache (`None` if it isn't there)."""
        try:
            with open(self.cache / f"{key}.json", encoding="utf-8") as f:
                entry = json.load(f)
        except (OSError, ValueError):
            return None
        if not isinstance(entry, dict) or not _is_result(entry.get("result")):
            return None
        return entry["result"]

    def _cache_put(self, key: str, result: Any, variant: AblationVariant, seed: int) -> None:
        import tempfile

        self.cache.mkdir(parents=True, exist_ok=True)
        entry = {
            "result": result,
            "variant": variant.id,
            "seed": seed,
            "evaluate_version": self.evaluate_version,
        }
        # Write to a temporary file first, so that an interrupted run never leaves a partial entry
        fd, tmp = tempfile.mkstemp(dir=self.cache, prefix=f".{key}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f)
            os.replace(tmp, self.cache / f"{key}.json")
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

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
        # Were the results read with `load_results()`?
        self._loaded = False

    def _settings(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "metric": self._primary or self.metric,
            "metrics": list(self._metrics) if self._metrics is not None else None,
            "higher_is_better": self.higher_is_better,
            "repeats": self.repeats,
            "seed": self.seed,
            "control": _control_setting(self.control),
            "duplicates": self.duplicates,
            "duplicates_order_by": list(self._order_by) if self._order_by else None,
            "max_removed": self.max_removed,
            "conf_level": self.conf_level,
            "adjust": self.adjust,
            "evaluate_version": self.evaluate_version,
            "n_rows": self._n_rows,
            "tbl_name": getattr(self.validation, "tbl_name", None),
        }

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

        resolved = self._resolve()
        tasks = self._tasks()
        names = {"full": self._full_variant().name, **{v.id: v.name for v, _ in resolved.variants}}

        rows = [
            {
                "group": "Baseline",
                "variant": _variant_html(self._full_variant(), self._steps),
                "removed": _removed_html(0, 0.0),
            }
        ]
        variants = [v for v, _ in resolved.variants]
        for v in sorted(variants, key=self._group_order):
            same_as = resolved.same_as.get(v.id)
            rows.append(
                {
                    "group": self._group_label(v),
                    "variant": _variant_html(
                        v,
                        self._steps,
                        same_as=names[same_as] if same_as else None,
                        detail=self._sweep_details.get(v.id) if v.sweep else None,
                    ),
                    "removed": _removed_html(v.n_removed, v.f_removed),
                }
            )
        rows += [
            {
                "group": "Controls",
                "variant": _variant_html(v, self._steps, control=self.control),
                "removed": _removed_html(v.n_removed, v.f_removed),
            }
            for v in resolved.controls
        ]

        subtitle = (
            f"{self._n_rows:,} rows in the training table. Each version is evaluated "
            f"{self.repeats} time{'s' if self.repeats != 1 else ''}, so the run will make "
            f"<strong>{len(tasks):,}</strong> evaluations."
        )
        if self.cache is not None:
            n_cached = sum(
                self._cache_get(self._cache_key(self._removed_rows(v, r), self.seed + r))
                is not None
                for v, r in tasks
            )
            if n_cached:
                subtitle += f" {n_cached:,} of them are already in the cache."
        notes = resolved.notes
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

    def run(self, max_workers: int | None = None, executor: str = "thread") -> Ablation:
        """
        Evaluate the full table, each variant, and each control.

        The `evaluate=` function is called `repeats=` times for each version of the table, with
        the seeds `seed`, `seed + 1`, and so on. The full table is evaluated first, then each
        variant, then each control. A control is drawn anew for each repeat, so that the
        comparison with it doesn't depend on one random draw of rows. Once the run is done, the
        results are available from [`get_tabular_report()`](`pointblank.Ablation.get_tabular_report`),
        [`get_effects()`](`pointblank.Ablation.get_effects`), and
        [`to_json()`](`pointblank.Ablation.to_json`). Running again repeats every evaluation,
        except those in the cache (see `cache=`).

        Parameters
        ----------
        max_workers
            The number of evaluations to make at the same time. By default (or with `1`), they're
            made one after another. Models that already use several cores (e.g., with
            `n_jobs=-1`) usually gain little from this.
        executor
            How evaluations are made at the same time, when `max_workers=` is more than `1`. With
            `"thread"` (the default), they run in threads of this process, which suits models
            that release Python's global interpreter lock while training (as most libraries
            written in C, C++, or Rust do). With `"process"`, each runs in a separate process:
            the `evaluate=` function must then be one that can be pickled (e.g., defined at the
            top level of a module, rather than in a notebook or a lambda), and the training table
            must be a Polars, pandas, or PyArrow table.

        Returns
        -------
        Ablation
            The `Ablation` object, with the results of the run.

        Details
        -------
        The results are the same whatever `max_workers=` is: the evaluations are made in a
        different order, but each gets the same table and seed, and the results are put back in
        order. If an evaluation fails, the evaluations that haven't started are cancelled and the
        error is raised. With a cache, the results that were already made are kept, so running
        again picks up where the run stopped.

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
            .run(max_workers=4)
        )

        ablation
        ```

        After the run, the `Ablation` object shows its report. The report shows the metric named
        in `metric=`; the other metrics are in the results of `get_effects()`.
        """
        if self.evaluate is None:
            raise ValueError(
                "There's no `evaluate=` function to run the evaluations with. Give one when "
                "creating the `Ablation`, or evaluate the versions of the table elsewhere with "
                "`export()` and `load_results()`."
            )
        if max_workers is not None and (not _is_int(max_workers) or max_workers < 1):
            raise ValueError(f"`max_workers=` must be a positive integer, got {max_workers!r}.")
        if executor not in _EXECUTORS:
            raise ValueError(
                f"`executor=` must be one of {', '.join(repr(e) for e in _EXECUTORS)}; got "
                f"{executor!r}."
            )
        resolved = self._check_variants()
        tasks = self._tasks()
        parallel = max_workers is not None and max_workers > 1
        if parallel and executor == "process":
            self._check_process_executor()

        self._start_results(resolved)
        if parallel:
            outcomes = self._run_parallel(tasks, max_workers=max_workers, executor=executor)
            for (variant, repeat), (values, cached) in zip(tasks, outcomes):
                self._record(variant, repeat, values, cached=cached)
        else:
            for variant, repeat in tasks:
                values, cached = self._task_values(variant, repeat)
                self._record(variant, repeat, values, cached=cached)
        self._finish_results(resolved)
        return self

    def check_determinism(self, seed: int | None = None, tolerance: float = 0.0) -> bool:
        """
        Check that `evaluate=` gives the same result twice for the same table and seed.

        The comparisons of an ablation are paired: each version of the table is evaluated with
        the same seeds, so that the randomness of training (e.g., the initial weights of a model,
        or the order of its training rows) cancels out in the differences. That only works if the
        seed controls all of that randomness. This method evaluates the full table twice with the
        same seed and warns if the results differ, which is worth checking before a long run.

        Parameters
        ----------
        seed
            The seed to evaluate with. By default, it's the `seed=` of the ablation.
        tolerance
            The largest difference in a metric that still counts as the same. Differences from
            floating-point rounding (e.g., from a model that sums its trees in parallel, in
            whichever order they finish) always count as the same: these are relative
            differences of up to `1e-9`.

        Returns
        -------
        bool
            `True` if both evaluations gave the same metrics (within `tolerance=`), `False`
            (with a warning that lists the metrics that differ) if they didn't.

        Examples
        --------
        Here's an `evaluate()` function that forgets to use its seed, so that its bootstrap
        sample is different every time.

        ```{python}
        import warnings
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed):
            sample = train.sample(fraction=1.0, with_replacement=True)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_between(columns="temp", left=-10, right=45)
            .interrogate()
        )

        ablation = pb.Ablation(validation, evaluate=evaluate)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            deterministic = ablation.check_determinism()

        print(deterministic)
        print(caught[0].message)
        ```

        Passing `seed=seed` to `sample()` would fix it, and the check would then return `True`.
        """
        if self.evaluate is None:
            raise ValueError("There's no `evaluate=` function to check.")
        if seed is None:
            seed = self.seed
        elif not _is_int(seed):
            raise TypeError(f"`seed=` must be an integer, got {seed!r}.")
        if not _is_number(tolerance) or tolerance < 0:
            raise ValueError(f"`tolerance=` must be a number of at least 0, got {tolerance!r}.")

        full = self._full_variant()
        first = self._result_values(self._call(full, frozenset(), seed), full, seed)
        second = self._result_values(self._call(full, frozenset(), seed), full, seed)

        if set(first) != set(second):
            details = (
                f"{', '.join(map(repr, first))} the first time and "
                f"{', '.join(map(repr, second))} the second"
            )
            different = True
        else:
            names = [
                name
                for name in first
                if abs(first[name] - second[name])
                > max(tolerance, _ROUNDING * max(abs(first[name]), abs(second[name])))
            ]
            details = "; ".join(f"{name}: {first[name]!r} and {second[name]!r}" for name in names)
            different = bool(names)
        if different:
            warnings.warn(
                "`evaluate=` isn't deterministic: given the full table and the seed "
                f"{seed} twice, it returned different results ({details}). Use the seed for "
                "every source of randomness in training, or the ablation's comparisons will be "
                "noisier than they need to be.",
                UserWarning,
                stacklevel=2,
            )
        return not different

    def export(
        self, path: str | os.PathLike[str], format: str = "parquet", overwrite: bool = False
    ) -> Path:
        """
        Write out the versions of the table to evaluate, for training them elsewhere.

        When the model can't be trained in this Python session (e.g., it's trained on a cluster,
        by another team, or in another language), the ablation can be split in two. This method
        writes every version of the training table that a run would evaluate to a directory,
        with a manifest that says which seed to train each one with. Once the evaluations are
        done, [`load_results()`](`pointblank.Ablation.load_results`) reads the metrics back and
        computes the results, just as [`run()`](`pointblank.Ablation.run`) would.

        Parameters
        ----------
        path
            The directory to write to. It's created if it doesn't exist.
        format
            How the versions of the table are written. With `"parquet"` (the default), each is a
            Parquet file of the table with its rows removed. With `"indices"`, each is a JSON file
            with the numbers of the rows that are removed (counting from `0`, in the order of the
            table), which is much smaller when the table is large and the evaluations can read
            the full table themselves. For database tables, which have no row order of their own,
            the rows are numbered in the order of their values, so `"parquet"` is the safer
            choice.
        overwrite
            Should an earlier export in the same directory be replaced? By default, there's an
            error if the directory already has a manifest.

        Returns
        -------
        Path
            The directory that was written to.

        Details
        -------
        The directory has:

        - `manifest.json`: the settings of the ablation, the versions of the table, every
        evaluation to make (the version, the repeat, the seed, and the file of its table), and
        fingerprints of the data and of the plan, which `load_results()` checks
        - `tables/`: the versions of the table, one file for each variant (shared by its repeats)
        and one for each repeat of each control, since a control's random rows change from one
        repeat to the next
        - `results_template.csv`: a table with a row for each evaluation and an empty column for
        the metric, to be filled in and saved as `results.csv`

        Each evaluation trains a model on its table with its seed and scores it on the same fixed
        holdout data, as the `evaluate=` function of a run would. Add a column to the results for
        each metric, if there's more than one.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's an ablation of a table of temperature readings, with no `evaluate=` function
        since the evaluations are made elsewhere.

        ```{python}
        import tempfile
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        readings = pl.DataFrame({"temp": temps})

        validation = (
            pb.Validate(data=readings)
            .col_vals_between(columns="temp", left=-10, right=45)
            .interrogate()
        )

        ablation = pb.Ablation(validation, metric="error", higher_is_better=False).drop_each()

        directory = ablation.export(tempfile.mkdtemp())

        pl.read_csv(directory / "results_template.csv").head()
        ```

        The template has a row for each of the 15 evaluations (5 repeats of the full table, the
        variant, and its control), each with the seed to train with.
        """
        if format not in _EXPORT_FORMATS:
            raise ValueError(
                f"`format=` must be one of {', '.join(repr(f) for f in _EXPORT_FORMATS)}; got "
                f"{format!r}."
            )
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(f"`path=` must be the path of a directory, got {type(path).__name__}.")
        resolved = self._check_variants()
        tasks = self._tasks()

        root = Path(path)
        if (root / "manifest.json").exists() and not overwrite:
            raise FileExistsError(
                f"There's already an exported ablation in {str(root)!r}. Use `overwrite=True` to "
                "replace it."
            )
        (root / "tables").mkdir(parents=True, exist_ok=True)

        extension = "parquet" if format == "parquet" else "json"
        slugs = _unique_slugs(dict.fromkeys(variant.id for variant, _ in tasks))
        written = set()
        runs = []
        for variant, repeat in tasks:
            # The rows a control removes change with each repeat, while a variant's don't
            if variant.is_control:
                file = f"tables/{slugs[variant.id]}_r{repeat}.{extension}"
                repeats = [repeat]
            else:
                file = f"tables/{slugs[variant.id]}.{extension}"
                repeats = list(range(self.repeats))
            if file not in written:
                removed = self._removed_rows(variant, repeat)
                self._write_table(root / file, removed, format, variant=variant, repeats=repeats)
                written.add(file)
            runs.append(
                {"variant": variant.id, "repeat": repeat, "seed": self.seed + repeat, "file": file}
            )

        columns = ["variant", "repeat", "seed", self.metric or "score"]
        with open(root / "results_template.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            for run in runs:
                writer.writerow([run["variant"], run["repeat"], run["seed"], ""])

        import datetime

        from pointblank import __version__

        manifest = {
            "format": _MANIFEST_FORMAT,
            "format_version": _MANIFEST_VERSION,
            "pointblank_version": __version__,
            "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
            "table_format": format,
            "fingerprints": {"data": self._fingerprint(), "plan": self._plan_fingerprint()},
            "settings": self._settings(),
            "variants": [v._to_dict() for v in self._versions(resolved)],
            "same_as": dict(resolved.same_as),
            "runs": runs,
            "results": {
                "file": "results.csv",
                "template": "results_template.csv",
                "columns": columns,
            },
            "notes": [*self._notes, *resolved.notes],
        }
        with open(root / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        return root

    def load_results(
        self, results: Any, manifest: str | os.PathLike[str] | None = None
    ) -> Ablation:
        """
        Read the results of evaluations made elsewhere, and compute the effects.

        This is the second half of an ablation split with
        [`export()`](`pointblank.Ablation.export`): once each version of the table has been
        trained and scored, this method reads the metrics and computes the results, as
        [`run()`](`pointblank.Ablation.run`) would. The ablation must have the same variants and
        settings as the one that was exported (e.g., it's made by the same code).

        Parameters
        ----------
        results
            The metrics of the evaluations: the path of a CSV file, the path of a directory with a
            `results.csv` file (e.g., the directory written by `export()`), or a table (e.g., a
            Polars or pandas DataFrame). It needs a `variant` column (the id of the version of the
            table), a `seed` column, and a column for each metric. A `repeat` column is checked if
            it's there, and any other column is a metric.
        manifest
            The path of the `manifest.json` file written by `export()` (or of its directory). By
            default, a manifest in the same directory as the results file is used if there is one.
            Its fingerprints are checked against the ablation, which makes sure that the results
            are for the same data and the same versions of the table.

        Returns
        -------
        Ablation
            The `Ablation` object, with the results.

        Details
        -------
        There must be exactly one row for each evaluation in the plan, with a value for every
        metric. The results are checked in the same way as those from `evaluate=`: when there's
        more than one metric, `metric=` must name the one that verdicts are based on.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's an ablation of a table of temperature readings, exported for training elsewhere.

        ```{python}
        import tempfile
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        readings = pl.DataFrame({"temp": temps})

        validation = (
            pb.Validate(data=readings)
            .col_vals_between(columns="temp", left=-10, right=45)
            .interrogate()
        )

        ablation = pb.Ablation(validation, metric="error", higher_is_better=False).drop_each()
        directory = ablation.export(tempfile.mkdtemp())
        ```

        Here, the 'training' happens in the same session, reading each table from the manifest
        and filling in the template, but it could be done anywhere.

        ```{python}
        import json

        manifest = json.loads((directory / "manifest.json").read_text())

        def train_and_score(table, seed):
            sample = table.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        results = pl.DataFrame(
            [
                {
                    "variant": run["variant"],
                    "seed": run["seed"],
                    "error": train_and_score(pl.read_parquet(directory / run["file"]), run["seed"]),
                }
                for run in manifest["runs"]
            ]
        )
        results.write_csv(directory / "results.csv")

        ablation.load_results(directory)
        ```

        The results were found in the directory, next to the manifest, and the report is the same
        as a run with the same evaluations would give.
        """
        resolved = self._check_variants()
        tasks = self._tasks()
        rows, columns, location = _read_results(results)

        manifest_path = None
        if manifest is not None:
            if not isinstance(manifest, (str, os.PathLike)):
                raise TypeError(
                    "`manifest=` must be the path of a manifest file, got "
                    f"{type(manifest).__name__}."
                )
            manifest_path = Path(manifest)
            if manifest_path.is_dir():
                manifest_path = manifest_path / "manifest.json"
            if not manifest_path.exists():
                raise FileNotFoundError(f"There's no manifest at {str(manifest_path)!r}.")
        elif location is not None and (location / "manifest.json").exists():
            manifest_path = location / "manifest.json"
        if manifest_path is not None:
            self._check_manifest(manifest_path)

        for required in ("variant", "seed"):
            if required not in columns:
                raise ValueError(f"The results have no {required!r} column.")
        metrics = [c for c in columns if c not in ("variant", "repeat", "seed")]
        if not metrics:
            raise ValueError(
                "The results have no metrics: they need a column for each metric, along with "
                "'variant' and 'seed'."
            )

        expected = {(variant.id, self.seed + repeat): repeat for variant, repeat in tasks}
        found: dict[tuple[str, int], dict[str, float]] = {}
        for n, row in enumerate(rows, start=1):
            variant_id = row["variant"]
            if variant_id is None or variant_id == "":
                raise ValueError(f"Row {n} of the results has no variant.")
            variant_id = str(variant_id)
            seed = _parse_int(row["seed"], f"Row {n} of the results has the seed")
            key = (variant_id, seed)
            if key not in expected:
                raise ValueError(
                    f"Row {n} of the results is for {variant_id!r} with the seed {seed}, which "
                    "isn't one of the evaluations of the ablation."
                )
            if key in found:
                raise ValueError(
                    f"The results have more than one row for {variant_id!r} with the seed {seed}."
                )
            if row.get("repeat") not in (None, ""):
                repeat = _parse_int(row["repeat"], f"Row {n} of the results has the repeat")
                if repeat != expected[key]:
                    raise ValueError(
                        f"Row {n} of the results has the repeat {repeat}, but the seed {seed} is "
                        f"for repeat {expected[key]}."
                    )
            values = {}
            for metric in metrics:
                value = _parse_float(row[metric])
                if value is None:
                    raise ValueError(f"Row {n} of the results has no value for {metric!r}.")
                if not math.isfinite(value):
                    raise ValueError(
                        f"Row {n} of the results has {value!r} for {metric!r}, but each metric "
                        "must be a finite number."
                    )
                values[metric] = value
            found[key] = values

        missing = [key for key in expected if key not in found]
        if missing:
            shown = ", ".join(f"{v!r} (seed {s})" for v, s in missing[:5])
            more = f", and {len(missing) - 5:,} more" if len(missing) > 5 else ""
            raise ValueError(
                f"The results are missing {len(missing):,} of the {len(expected):,} evaluations: "
                f"{shown}{more}."
            )

        self._start_results(resolved)
        self._loaded = True
        for variant, repeat in tasks:
            values = found[(variant.id, self.seed + repeat)]
            self._record(variant, repeat, values, cached=False, loaded=True)
        self._finish_results(resolved)
        return self

    def _check_variants(self) -> _Resolved:
        resolved = self._resolve()
        if not self._specs:
            raise ValueError(
                "There are no variants to evaluate. Add some with `drop_each()`, `drop_all()`, "
                "or `drop_steps()` first."
            )
        if not resolved.variants:
            raise ValueError(
                "None of the variants remove any rows of the table, so there's nothing to "
                "compare with the full table."
            )
        return resolved

    def _check_process_executor(self) -> None:
        import pickle

        implementation = nw.from_native(self.validation.data).implementation
        if implementation not in (
            nw.Implementation.POLARS,
            nw.Implementation.PANDAS,
            nw.Implementation.PYARROW,
        ):
            raise ValueError(
                '`executor="process"` sends each training table to another process, which works '
                "for Polars, pandas, and PyArrow tables, but not for this table "
                f'({type(self.validation.data).__name__}). Use `executor="thread"` instead.'
            )
        try:
            pickle.dumps(self.evaluate)
        except Exception as e:
            raise TypeError(
                '`executor="process"` needs an `evaluate=` function that can be sent to another '
                "process (e.g., a function defined at the top level of a module, rather than a "
                'lambda or a function defined inside another one). Use `executor="thread"` '
                "instead."
            ) from e

    def _versions(self, resolved: _Resolved) -> list[AblationVariant]:
        """The full table, the variants, and the controls."""
        return [self._full_variant(), *(v for v, _ in resolved.variants), *resolved.controls]

    def _call(self, variant: AblationVariant, removed: frozenset[int], seed: int) -> Any:
        kwargs: dict[str, Any] = {"seed": seed}
        if self._pass_variant:
            kwargs["variant"] = variant
        return self.evaluate(self._training_table(removed), **kwargs)

    def _task_values(self, variant: AblationVariant, repeat: int) -> tuple[dict[str, float], bool]:
        """Evaluate a version of the table (or get the result from the cache)."""
        seed = self.seed + repeat
        removed = self._removed_rows(variant, repeat)
        key = None
        if self.cache is not None:
            key = self._cache_key(removed, seed)
            cached = self._cache_get(key)
            if cached is not None:
                return self._result_values(cached, variant, seed), True
        result = self._call(variant, removed, seed)
        values = self._result_values(result, variant, seed)
        if key is not None:
            self._cache_put(key, _stored_result(result), variant, seed)
        return values, False

    def _run_parallel(
        self, tasks: list[tuple[AblationVariant, int]], max_workers: int, executor: str
    ) -> list[tuple[dict[str, float], bool]]:
        """Make the evaluations in a pool of threads or processes, giving the results in order."""
        import multiprocessing
        from concurrent.futures import (
            FIRST_COMPLETED,
            ProcessPoolExecutor,
            ThreadPoolExecutor,
            wait,
        )

        outcomes: list[Any] = [None] * len(tasks)
        todo = []
        first_with_key: dict[str, int] = {}
        same_as: dict[int, int] = {}
        for k, (variant, repeat) in enumerate(tasks):
            key = None
            if self.cache is not None:
                seed = self.seed + repeat
                key = self._cache_key(self._removed_rows(variant, repeat), seed)
                cached = self._cache_get(key)
                if cached is not None:
                    outcomes[k] = (self._result_values(cached, variant, seed), True)
                    continue
                # Versions with the same rows and seed are only evaluated once
                if key in first_with_key:
                    same_as[k] = first_with_key[key]
                    continue
                first_with_key[key] = k
            todo.append((k, key))

        if executor == "process":
            # Not forked: forking a process that uses threads (as Polars does) can deadlock
            pool: Any = ProcessPoolExecutor(
                max_workers=max_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_set_worker_evaluate,
                initargs=(self.evaluate,),
            )
        else:
            pool = ThreadPoolExecutor(max_workers=max_workers)

        pending = iter(todo)
        running: dict[Any, tuple[int, str | None]] = {}

        def submit() -> None:
            item = next(pending, None)
            if item is None:
                return
            k, key = item
            variant, repeat = tasks[k]
            seed = self.seed + repeat
            removed = self._removed_rows(variant, repeat)
            if executor == "process":
                future = pool.submit(
                    _worker_evaluate,
                    self._training_table(removed),
                    seed,
                    variant if self._pass_variant else None,
                )
            else:
                future = pool.submit(self._call, variant, removed, seed)
            running[future] = (k, key)

        try:
            # Only a few tables wait for a worker at a time, so they don't all sit in memory
            for _ in range(2 * max_workers):
                submit()
            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    k, key = running.pop(future)
                    variant, repeat = tasks[k]
                    seed = self.seed + repeat
                    result = future.result()
                    values = self._result_values(result, variant, seed)
                    if key is not None:
                        self._cache_put(key, _stored_result(result), variant, seed)
                    outcomes[k] = (values, False)
                    submit()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

        for k, first in same_as.items():
            outcomes[k] = (outcomes[first][0], True)
        return outcomes

    def _result_values(self, result: Any, variant: AblationVariant, seed: int) -> dict[str, float]:
        """Check the result of an evaluation, and give its metrics."""
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
        return {name: float(value) for name, value in values.items()}

    def _check_metrics(
        self, values: dict[str, float], variant: AblationVariant, seed: int, loaded: bool
    ) -> None:
        """Check that an evaluation has the same metrics as the first one (the full table)."""
        source = "The results have" if loaded else "`evaluate=` returned"
        if self._metrics is None:
            if self.metric is None and len(values) > 1:
                raise ValueError(
                    f"{source} several metrics ({', '.join(map(repr, values))}), so use "
                    "`metric=` to say which one the verdicts are based on."
                )
            if self.metric is not None and self.metric not in values:
                missing = "The results don't have" if loaded else "`evaluate=` didn't return"
                raise ValueError(
                    f"{missing} the metric {self.metric!r} named in `metric=` (only "
                    f"{', '.join(map(repr, values))})."
                )
            self._metrics = list(values)
            self._primary = self.metric or self._metrics[0]
        elif set(values) != set(self._metrics):
            raise ValueError(
                f"`evaluate=` returned the metrics {', '.join(map(repr, values))} for the variant "
                f"{variant.id!r} (seed {seed}), but {', '.join(map(repr, self._metrics))} for the "
                "full table. It must return the same metrics every time."
            )

    def _start_results(self, resolved: _Resolved) -> None:
        self._reset_results()
        self._variants = [v for v, _ in resolved.variants]
        self._control_variants = list(resolved.controls)

    def _record(
        self,
        variant: AblationVariant,
        repeat: int,
        values: dict[str, float],
        cached: bool,
        loaded: bool = False,
    ) -> None:
        seed = self.seed + repeat
        self._check_metrics(values, variant=variant, seed=seed, loaded=loaded)
        self._values[(variant.id, repeat)] = values
        self._runs.append(
            {
                "variant": variant.id,
                "repeat": repeat,
                "seed": seed,
                "metrics": values,
                "cached": cached,
            }
        )

    def _finish_results(self, resolved: _Resolved) -> None:
        # The variants that weren't evaluated get the results of the version with the same rows
        for variant_id, same_as in resolved.same_as.items():
            for r in range(self.repeats):
                self._values[(variant_id, r)] = self._values[(same_as, r)]
        self._effects = self._compute_effects()

    def _check_manifest(self, path: Path) -> None:
        try:
            with open(path, encoding="utf-8") as f:
                manifest = json.load(f)
        except ValueError:
            raise ValueError(f"{str(path)!r} isn't a valid manifest (it isn't JSON).") from None
        if not isinstance(manifest, dict) or manifest.get("format") != _MANIFEST_FORMAT:
            raise ValueError(f"{str(path)!r} isn't a manifest written by `Ablation.export()`.")
        fingerprints = manifest.get("fingerprints") or {}
        if fingerprints.get("data") != self._fingerprint():
            raise ValueError(
                "The results are for different data: the data fingerprint in the manifest "
                "doesn't match the table of this ablation's validation."
            )
        if fingerprints.get("plan") != self._plan_fingerprint():
            raise ValueError(
                "The results are for a different plan: the versions of the table, the repeats, "
                "or the seed of this ablation aren't the same as those in the manifest."
            )

    def _write_table(
        self,
        file: Path,
        removed: frozenset[int],
        format: str,
        variant: AblationVariant,
        repeats: list[int],
    ) -> None:
        if format == "indices":
            payload = {
                "variant": variant.id,
                "repeats": repeats,
                "n_rows": self._n_rows,
                "removed": sorted(removed),
            }
            with open(file, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            return
        frame = self._training_frame(removed)
        if isinstance(frame, nw.LazyFrame):
            frame = frame.collect()
        if frame.implementation.is_pandas_like():
            frame = nw.from_native(frame.to_native().reset_index(drop=True), eager_only=True)
        frame.write_parquet(file)

    # ------------------------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------------------------

    def _compute_effects(self) -> list[dict[str, Any]]:
        resolved = self._resolve()
        repeats = range(self.repeats)
        rows = []

        def series(version_id: str, metric: str) -> list[float]:
            return [self._values[(version_id, r)][metric] for r in repeats]

        for metric in self._metrics:
            full = series("full", metric)
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
                values = series(variant.id, metric)
                row = _effect_row(
                    variant, metric=metric, mean=statistics.fmean(values), sd=_sd(values)
                )
                _add_comparison(row, "vs_full", values, full, self.conf_level)
                control_id = resolved.variant_control.get(variant.id)
                control = series(control_id, metric) if control_id is not None else None
                if control is not None:
                    row["control_mean"] = statistics.fmean(control)
                if variant.left_out is None:
                    if control is not None:
                        _add_comparison(row, "vs_control", values, control, self.conf_level)
                else:
                    # What the left-out step adds: removing every step's rows compared with
                    # removing the other steps' rows (plus as many random rows, with a control)
                    reference = series(resolved.references[variant.id], metric)
                    baseline = control if control is not None else values
                    _add_comparison(row, "added", reference, baseline, self.conf_level)
                variant_rows.append(row)

            normal = [row for row in variant_rows if row["left_out"] is None]
            left_out = [row for row in variant_rows if row["left_out"] is not None]
            for prefix, family in (
                ("vs_full", variant_rows),
                ("vs_control", normal),
                ("added", left_out),
            ):
                p_values = [row[f"p_{prefix}"] for row in family]
                adjusted = _holm(p_values) if self.adjust == "holm" else p_values
                for row, p_adj in zip(family, adjusted):
                    row[f"p_adj_{prefix}"] = p_adj

            if metric == self._primary:
                comparison = "vs_control" if self.control is not None else "vs_full"
                alone = {}
                for row in normal:
                    row["verdict"] = self._verdict(row, comparison)
                    if len(row["steps"]) == 1:
                        alone[row["steps"][0]] = row["verdict"]
                for row in left_out:
                    verdict = self._verdict(row, "added")
                    # The step's rows help on their own, but add nothing to the other steps
                    if verdict == "inconclusive" and alone.get(row["left_out"]) == "helps":
                        verdict = "redundant"
                    row["verdict"] = verdict

            rows += variant_rows

        return rows

    def _verdict(self, row: dict[str, Any], prefix: str) -> str | None:
        if self.repeats < 3:
            return None
        p_adj = row[f"p_adj_{prefix}"]
        delta = row[f"delta_{prefix}"]
        if p_adj is None or delta is None:
            return None  # pragma: no cover
        if p_adj < 1 - self.conf_level and delta != 0:
            better = delta > 0 if self.higher_is_better else delta < 0
            return "helps" if better else "hurts"
        return "inconclusive"

    def _check_run(self) -> None:
        if self._effects is None:
            raise ValueError("There are no results yet. Call `run()` (or `load_results()`) first.")

    def annotate(self, validation: Validate | None = None) -> Validate:
        """
        Add the results of the ablation to the steps of the validation, as notes.

        Each step with a variant of its own (from [`drop_each()`](`pointblank.Ablation.drop_each`),
        or a [`drop_steps()`](`pointblank.Ablation.drop_steps`) variant with just that step) gets
        a note with the effect of removing its failing rows and the verdict. A step with a
        variant from [`drop_all_but_each()`](`pointblank.Ablation.drop_all_but_each`) also gets
        the effect of its rows on top of the other steps. The notes appear in the validation's
        report, so that the report shows which checks matter to the model, next to the checks
        themselves.

        Parameters
        ----------
        validation
            The validation to add the notes to. By default, it's the validation of the ablation.
            Another validation (e.g., of a newer version of the data, with the same steps) can be
            given, as long as each step used in the ablation is the same kind of check of the
            same column.

        Returns
        -------
        Validate
            The validation, with the notes added (any notes from an earlier `annotate()` are
            replaced).

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's an ablation of a table of temperature readings with out-of-range and missing
        values.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        for k in range(7, 300, 40):
            temps[k] = None
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed):
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

        ablation.annotate()
        ```

        The validation report now has a note on each step, with the change in the error from
        removing its failing rows and the verdict.
        """
        from pointblank.validate import Validate

        self._check_run()
        target = self.validation if validation is None else validation
        if not isinstance(target, Validate):
            raise TypeError(
                f"`validation=` must be a `Validate` object, got {type(target).__name__} instead."
            )
        info = {step.i: step for step in target.validation_info}
        if target is not self.validation:
            for i, step in self._steps.items():
                other = info.get(i)
                if (
                    other is None
                    or other.assertion_type != step.assertion_type
                    or other.column != step.column
                ):
                    raise ValueError(
                        f"The validation doesn't match the ablation's: its step {i} isn't a "
                        f"`{step.assertion_type}()` step of the same column."
                    )

        effects = [row for row in self._effects if row["metric"] == self._primary]
        fmt = _NumberFormat([row["mean"] for row in effects])
        alone = {}
        on_top: dict[int, list[dict[str, Any]]] = {}
        for row in effects:
            if row["strategy"] in ("full", "drop_groups"):
                continue
            if row["left_out"] is not None:
                on_top.setdefault(row["left_out"], []).append(row)
            elif len(row["steps"]) == 1:
                alone[row["steps"][0]] = row

        comparison = "vs_control" if self.control is not None else "vs_full"
        vs_text = "removing as many random rows" if self.control is not None else "the full table"
        direction = "higher" if self.higher_is_better else "lower"
        for i in sorted(set(alone) | set(on_top)):
            if i not in info:
                continue  # pragma: no cover
            parts = []
            if i in alone:
                row = alone[i]
                change = _change_text(fmt, row, comparison)
                parts.append(
                    (
                        f"removing the {row['n_removed']:,} rows that failed this step changes "
                        f"it by {change}, compared with {vs_text}",
                        row["verdict"],
                    )
                )
            for row in on_top.get(i, []):
                others = ", ".join(str(k) for k in row["steps"])
                label = "step" if len(row["steps"]) == 1 else "steps"
                parts.append(
                    (
                        f"on top of {label} {others}, this step's rows change it by "
                        f"{_change_text(fmt, row, 'added')}",
                        row["verdict"],
                    )
                )
            intro = f"Ablation ({self._primary}, {direction} is better): "
            markdown = intro + "; ".join(
                text + (f": **{verdict}**" if verdict else "") for text, verdict in parts
            )
            text = intro + "; ".join(
                text + (f": {verdict}" if verdict else "") for text, verdict in parts
            )
            info[i]._add_note(key="ablation", markdown=markdown + ".", text=text + ".")
        return target

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
            - `left_out`: for a variant of
            [`drop_all_but_each()`](`pointblank.Ablation.drop_all_but_each`), the step whose rows
            are kept
            - `group`: for a variant of [`drop_groups()`](`pointblank.Ablation.drop_groups`), the
            value of the group whose rows are removed, as a string
            - `sweep`, `value`: for a point of a sweep (from
            [`sweep_score()`](`pointblank.Ablation.sweep_score`) or
            [`sweep_step()`](`pointblank.Ablation.sweep_step`)), the id of the sweep and the
            point's value (the share of the scored rows removed, the score cutoff, or the value of
            the step's parameter)
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
            - `delta_added`, `delta_added_lower`, `delta_added_upper`, `p_added`,
            `p_adj_added`: for a variant of `drop_all_but_each()`, the difference that removing
            the left-out step's rows makes on top of the other steps (the variant that removes
            every step's rows, compared with this variant, or with its control)
            - `verdict`: `"helps"`, `"hurts"`, or `"inconclusive"` for the metric named in
            `metric=` (missing for other metrics, and with fewer than 3 repeats), or
            `"redundant"` for a variant of `drop_all_but_each()` whose left-out step helps on its
            own but adds nothing to the other steps

            The comparisons are missing for the full table, and the control columns are missing
            when `control=None`. The intervals and p-values need at least 2 repeats. The verdict
            of a `drop_all_but_each()` variant is based on the `delta_added` columns, and those
            of the other variants on the `delta_vs_control` columns (or the `delta_vs_full`
            columns, when `control=None`).

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
                # A nullable integer type, for the columns with missing values
                df[name] = df[name].astype("Int64" if df[name].isna().any() else "int64")
        return df

    def to_dict(self) -> dict[str, Any]:
        """
        Get the settings and results of the ablation as a JSON-serializable dict.

        Returns
        -------
        dict[str, Any]
            A dictionary with the ablation's settings (`"settings"`), fingerprints of the data and
            of the plan (`"fingerprints"`, as in the manifest of
            [`export()`](`pointblank.Ablation.export`)), the versions of the table that were
            evaluated (`"variants"`, including the full table and the controls), the variants that
            weren't evaluated because they remove the same rows as another version (`"same_as"`,
            from each to that version), the metrics from every evaluation (`"runs"`, one entry per
            version and repeat, with its seed and whether its result came from the cache), the
            results (`"effects"`, with the same fields as the columns of
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

        Each run records the version of the table, the repeat, the seed, the metrics, and whether
        the result came from the cache, so the runs can be stored and analyzed again later.
        """
        self._check_run()
        resolved = self._resolve()
        return {
            "settings": self._settings(),
            "fingerprints": {"data": self._fingerprint(), "plan": self._plan_fingerprint()},
            "variants": [v._to_dict() for v in self._versions(resolved)],
            "same_as": dict(resolved.same_as),
            "runs": [{**run, "metrics": dict(run["metrics"])} for run in self._runs],
            "effects": [{**row, "steps": list(row["steps"])} for row in self._effects],
            "notes": [*self._notes, *resolved.notes],
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
        variants of [`drop_all_but_each()`](`pointblank.Ablation.drop_all_but_each`) are in a
        group of their own, showing what the left-out step's rows add to the other steps. So is
        each sweep (from [`sweep_score()`](`pointblank.Ablation.sweep_score`) or
        [`sweep_step()`](`pointblank.Ablation.sweep_step`)), and a plot of its curve at the
        bottom shows the difference from the full table at each of its values, next to that of
        removing as many random rows. The header gives the metric for the full table and the settings of the run, and notes at the
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
        comparison = "vs_control" if self.control is not None else "vs_full"
        has_left_out = any(row["left_out"] is not None for row in variant_rows)
        has_groups = has_left_out or any(row["sweep"] is not None for row in variant_rows)
        resolved = self._resolve()
        names = {"full": baseline["name"], **{row["variant"]: row["name"] for row in variant_rows}}

        def prefix(row: dict[str, Any]) -> str:
            return "added" if row["left_out"] is not None else comparison

        fmt = _NumberFormat([row["mean"] for row in effects])
        bounds = [
            abs(row[key])
            for row in variant_rows
            for key in (
                f"delta_{prefix(row)}",
                f"delta_{prefix(row)}_lower",
                f"delta_{prefix(row)}_upper",
            )
            if row[key] is not None
        ]
        scale = max(bounds, default=0.0) or 1.0

        rows = []
        variants = {v.id: v for v in self._variants}
        for row in sorted(
            variant_rows, key=lambda row: self._group_order(variants[row["variant"]])
        ):
            variant = variants[row["variant"]]
            same_as = resolved.same_as.get(variant.id)
            metric_html = _mono(fmt.value(row["mean"], row["sd"]))
            if comparison == "vs_control" or variant.left_out is not None:
                metric_html += _small(f"Δ vs full {fmt.delta(row['delta_vs_full'])}")
            rows.append(
                {
                    "group": self._group_label(variant),
                    "variant": _variant_html(
                        variant,
                        self._steps,
                        same_as=names.get(same_as) if same_as else None,
                        detail=self._sweep_details.get(variant.id) if variant.sweep else None,
                    ),
                    "removed": _removed_html(row["n_removed"], row["f_removed"]),
                    "metric": metric_html,
                    "effect": _effect_html(row, prefix(row), fmt, scale),
                }
            )

        if comparison == "vs_control":
            vs_text = "removing the same number of random rows"
            if isinstance(self.control, Stratified):
                vs_text += f" (stratified by {', '.join(self.control.by)})"
        else:
            vs_text = "the full table"
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
            f"{_esc(vs_text)}.</div>"
        )
        if has_left_out:
            others = (
                "the other steps' rows, plus as many random rows"
                if self.control is not None
                else "only the other steps' rows"
            )
            subtitle += (
                "<div>For leaving one out, Δ is what the left-out step's rows add to the other "
                f"steps: removing every step's rows, compared with removing {others}.</div>"
            )
        if self.evaluate_version is not None:
            subtitle += (
                f"<div>Version of <code>evaluate</code>: {_esc(self.evaluate_version)}.</div>"
            )
        if metric != self._primary:
            subtitle += (
                f"<div>Verdicts are based on {_esc(self._primary)}, so none are shown here.</div>"
            )

        effect_label = "Δ vs random" if comparison == "vs_control" else "Δ vs full"
        default_title = _title_html("Ablation report", self.label)
        columns = ["variant", "removed", "metric", "effect"]
        gt_tbl = (
            _fmt_raw_html(
                GT(
                    _frame(rows, ["group", *columns] if has_groups else columns),
                    groupname_col="group" if has_groups else None,
                    id="pb_ablation_report",
                ),
                columns=columns,
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
        if has_groups:
            gt_tbl = gt_tbl.tab_style(
                style=[style.text(weight="bold", size="12px"), style.fill(color="#F6F7F8")],
                locations=loc.row_groups(),
            )

        interval_note = f"Intervals are {self.conf_level:.0%} paired t-intervals" + (
            ", not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values."
            if self.adjust == "holm"
            else "; verdicts use unadjusted p-values."
        )
        if self.repeats < 3:
            interval_note += " There are no verdicts with fewer than 3 repeats."
        source_notes = [interval_note]
        if self._loaded:
            source_notes.append(
                "The results were read with `load_results()`, from evaluations made elsewhere."
            )
        n_cached = sum(run["cached"] for run in self._runs)
        if n_cached:
            source_notes.append(
                f"{n_cached:,} of the {len(self._runs):,} evaluations came from the cache."
            )
        for note in [*source_notes, *self._notes, *resolved.notes]:
            gt_tbl = gt_tbl.tab_source_note(source_note=html(_note_html(note)))
        # A curve for each sweep: how the metric changes with the sweep's value
        for sweep_id, sweep in self._sweeps.items():
            points = [row for row in variant_rows if row["sweep"] == sweep_id]
            if points:
                gt_tbl = gt_tbl.tab_source_note(
                    source_note=html(
                        _sweep_svg(
                            sweep,
                            points,
                            baseline=baseline["mean"],
                            fmt=fmt,
                            metric=metric,
                            direction=direction,
                        )
                    )
                )
        return gt_tbl

    def _group_order(self, variant: AblationVariant) -> tuple[int, float]:
        """Where a variant is listed in the plan and the report: by its group, then its value."""
        if variant.sweep is not None:
            return (2 + list(self._sweeps).index(variant.sweep), variant.value)
        return (int(variant.left_out is not None), 0.0)

    def _group_label(self, variant: AblationVariant) -> str:
        if variant.sweep is not None:
            return f"Sweep: {self._sweeps[variant.sweep]['label']}"
        return "Variants" if variant.left_out is None else "Leave one out"


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
    group = None
    if variant.group_column is not None and variant.group is not None:
        group = str(variant.group)
    row.update(
        variant=variant.id,
        name=variant.name,
        strategy=variant.strategy,
        steps=list(variant.steps),
        left_out=variant.left_out,
        group=group,
        sweep=variant.sweep,
        value=variant.value,
        n_removed=variant.n_removed,
        f_removed=variant.f_removed,
        metric=metric,
        mean=mean,
        sd=sd,
    )
    return row


def _add_comparison(
    row: dict[str, Any],
    prefix: str,
    values: list[float],
    reference: list[float],
    conf_level: float,
) -> None:
    delta, lower, upper, p = _paired(values, reference, conf_level)
    row[f"delta_{prefix}"] = delta
    row[f"delta_{prefix}_lower"] = lower
    row[f"delta_{prefix}_upper"] = upper
    row[f"p_{prefix}"] = p


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


def _variant_html(
    variant: AblationVariant,
    steps: dict[int, Any],
    same_as: str | None = None,
    control: Any = None,
    max_briefs: int = 3,
    detail: str | None = None,
) -> str:
    """The variant's name, with a line for each step (its brief) below (or the `detail=` line)."""
    name = f"<div><strong>{_esc(variant.name)}</strong></div>"
    if variant.strategy == "full":
        return name + _small("Nothing removed")
    if variant.is_control:
        within = ""
        if isinstance(control, Stratified):
            within = f" within each group of {_esc(', '.join(control.by))}"
        if variant.left_out is not None:
            listed = ", ".join(str(i) for i in variant.steps)
            label = "step" if len(variant.steps) == 1 else "steps"
            text = (
                f"The rows of {label} {listed}, plus random rows{within} drawn anew for each repeat"
            )
        else:
            text = f"Rows removed at random{within}, drawn anew for each repeat"
        return name + _small(text)
    if variant.group_column is not None:
        return name + _small("All the rows of this group")

    def brief(i: int) -> str:
        step = steps[i]
        return _code_html(step.brief or step.autobrief or step.assertion_type)

    lines = []
    if detail is not None:
        lines.append(_code_html(detail))
    elif variant.left_out is not None:
        lines.append(f"Leaves out step {variant.left_out}: {brief(variant.left_out)}")
    else:
        for i in variant.steps[:max_briefs]:
            lines.append(brief(i) if len(variant.steps) == 1 else f"{i}: {brief(i)}")
        if len(variant.steps) > max_briefs:
            lines.append(f"and {len(variant.steps) - max_briefs} more steps")
    if same_as is not None:
        lines.append(f"Same rows as {_esc(same_as)}, so it isn't evaluated again")
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


def _effect_html(row: dict[str, Any], prefix: str, fmt: _NumberFormat, scale: float) -> str:
    """A difference (e.g., from the control), its interval, and the verdict badge."""
    delta = row[f"delta_{prefix}"]
    lower = row[f"delta_{prefix}_lower"]
    upper = row[f"delta_{prefix}_upper"]
    verdict = row["verdict"]
    color = _VERDICT_COLORS.get(verdict, "#666666")

    text = fmt.delta(delta)
    if lower is not None and upper is not None:
        text += f" <span style='color: #777777;'>[{fmt.delta(lower)}, {fmt.delta(upper)}]</span>"
    plot = _interval_svg(delta, lower, upper, scale, color)
    badge = f" {_badge(verdict, color)}" if verdict is not None else ""
    return f"<div>{_mono(text)}</div><div style='white-space: nowrap;'>{plot}{badge}</div>"


def _sweep_svg(
    sweep: dict[str, Any],
    points: list[dict[str, Any]],
    baseline: float,
    fmt: _NumberFormat,
    metric: str,
    direction: str,
) -> str:
    """
    A plot of a sweep: the difference from the full table at each of its values (a dot with its
    interval, colored by the verdict), with a dashed line for removing as many random rows.
    """
    width, height = 460, 230
    left, right, top, bottom = 58, 16, 44, 40
    points = sorted(points, key=lambda row: row["value"])
    scale_x = 100.0 if sweep["percent"] else 1.0
    xs = [row["value"] * scale_x for row in points]
    controls = [
        row["control_mean"] - baseline if row["control_mean"] is not None else None
        for row in points
    ]
    ys = [0.0, *controls]
    for row in points:
        ys += [row["delta_vs_full"], row["delta_vs_full_lower"], row["delta_vs_full_upper"]]
    ys = [y for y in ys if y is not None]

    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_min, x_max = x_min - 1, x_max + 1
    x_pad = (x_max - x_min) * 0.04
    x_min, x_max = x_min - x_pad, x_max + x_pad
    y_min, y_max = min(ys), max(ys)
    if y_min == y_max:
        y_min, y_max = y_min - 1, y_max + 1
    ticks = _nice_ticks(y_min, y_max)
    pad = (y_max - y_min) * 0.08
    y_min, y_max = min(y_min - pad, ticks[0]), max(y_max + pad, ticks[-1])

    def px(x: float) -> float:
        return left + (x - x_min) / (x_max - x_min) * (width - left - right)

    def py(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * (height - top - bottom)

    text_style = "font-family: 'IBM Plex Sans', sans-serif; font-size: 10px; fill: #555555;"
    title = f"{sweep['label']}: Δ {metric} vs full{direction}"
    parts = [
        f"<svg width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
        "style='display: block; margin-top: 4px;'>",
        f"<text x='0' y='13' style=\"{text_style} font-size: 11px; fill: #333333;\">"
        f"{_esc(title)}</text>",
        # The axes, and the line for no difference from the full table
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height - bottom}' stroke='#BBBBBB'/>",
        f"<line x1='{left}' y1='{height - bottom}' x2='{width - right}' y2='{height - bottom}' "
        "stroke='#BBBBBB'/>",
        f"<line x1='{left}' y1='{py(0):.1f}' x2='{width - right}' y2='{py(0):.1f}' "
        "stroke='#999999' stroke-width='1'/>",
    ]
    for y in ticks:
        if y != 0:
            parts.append(
                f"<line x1='{left}' y1='{py(y):.1f}' x2='{width - right}' y2='{py(y):.1f}' "
                "stroke='#EEEEEE' stroke-width='1'/>"
            )
        parts.append(
            f"<text x='{left - 5}' y='{py(y) + 3:.1f}' text-anchor='end' "
            f'style="{text_style}">{f"{y:+g}".replace("-", "−")}</text>'
        )
    for x in xs:
        label = f"{x:g}%" if sweep["percent"] else f"{x:g}"
        parts.append(
            f"<text x='{px(x):.1f}' y='{height - bottom + 13}' text-anchor='middle' "
            f'style="{text_style}">{_esc(label)}</text>'
        )
    parts.append(
        f"<text x='{(left + width - right) / 2:.1f}' y='{height - 6}' text-anchor='middle' "
        f'style="{text_style}">{_esc(sweep["axis"])}</text>'
    )

    # Removing as many random rows (when there's a control)
    control_points = [(px(x), py(y)) for x, y in zip(xs, controls) if y is not None]
    if len(control_points) > 1:
        path = " ".join(f"{x:.1f},{y:.1f}" for x, y in control_points)
        parts.append(
            f"<polyline points='{path}' fill='none' stroke='#888888' stroke-width='1.5' "
            "stroke-dasharray='4,3'/>"
        )
    elif control_points:
        x, y = control_points[0]
        parts.append(
            f"<line x1='{x - 8:.1f}' y1='{y:.1f}' x2='{x + 8:.1f}' y2='{y:.1f}' stroke='#888888' "
            "stroke-width='1.5' stroke-dasharray='4,3'/>"
        )

    path = " ".join(f"{px(x):.1f},{py(row['delta_vs_full']):.1f}" for x, row in zip(xs, points))
    parts.append(f"<polyline points='{path}' fill='none' stroke='#333333' stroke-width='1.5'/>")
    for x, row in zip(xs, points):
        color = _VERDICT_COLORS.get(row["verdict"], "#666666")
        lower, upper = row["delta_vs_full_lower"], row["delta_vs_full_upper"]
        if lower is not None and upper is not None:
            parts.append(
                f"<line x1='{px(x):.1f}' y1='{py(lower):.1f}' x2='{px(x):.1f}' "
                f"y2='{py(upper):.1f}' stroke='{color}' stroke-width='2'/>"
            )
        parts.append(
            f"<circle cx='{px(x):.1f}' cy='{py(row['delta_vs_full']):.1f}' r='3.5' "
            f"fill='{color}'><title>{_esc(row['name'])}: "
            f"{_esc(fmt.delta(row['delta_vs_full']))}</title></circle>"
        )
    if control_points:
        parts.append(
            f"<line x1='0' y1='26' x2='18' y2='26' stroke='#888888' stroke-width='1.5' "
            "stroke-dasharray='4,3'/>"
            f"<text x='22' y='29' style=\"{text_style}\">Removing as many random rows</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _nice_ticks(low: float, high: float, n: int = 4) -> list[float]:
    """Round values for the ticks of an axis from `low` to `high` (including zero)."""
    span = high - low
    raw = span / n
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    first = math.floor(low / step + _ROUNDING)
    last = math.ceil(high / step - _ROUNDING)
    return [round(k * step, 12) for k in range(first, last + 1)]


def _change_text(fmt: _NumberFormat, row: dict[str, Any], prefix: str) -> str:
    """A difference with its interval, as text (e.g., for a note)."""
    text = fmt.delta(row[f"delta_{prefix}"])
    lower, upper = row[f"delta_{prefix}_lower"], row[f"delta_{prefix}_upper"]
    if lower is not None and upper is not None:
        text += f" [{fmt.delta(lower)}, {fmt.delta(upper)}]"
    return text


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


def _control_setting(control: Any) -> Any:
    """The `control=` of an ablation, as a JSON value."""
    if isinstance(control, Stratified):
        return {"stratified": list(control.by)}
    return control


def _json_value(value: Any) -> Any:
    """A value of a table (e.g., a group of `drop_groups()`), as a JSON value."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if _is_int(value):
        return int(value)
    if _is_number(value):
        return float(value) if math.isfinite(float(value)) else str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _number_text(value: Any) -> str:
    """A number as text for the id and name of a sweep's point (e.g., `0.1`, or `5` for `5.0`)."""
    x = float(value)
    if x.is_integer() and abs(x) < 1e15:
        return str(int(x))
    return repr(x)


def _scores_by_row(rows: list[Any], scores: list[Any]) -> dict[int, float]:
    """The scores of rows by row number, leaving out the missing ones."""
    result = {}
    for k, score in zip(rows, scores):
        if score is None or type(score).__name__ in ("NAType", "NaTType"):
            continue
        score = float(score)
        if math.isnan(score):
            continue
        result[int(k)] = score
    return result


def _group_value(value: Any) -> Any:
    """A value used to group rows, with every kind of missing value as `None`."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    # pandas' missing values (`pd.NA` and `pd.NaT`), without importing pandas
    if type(value).__name__ in ("NAType", "NaTType"):
        return None
    hash(value)
    return value


def _sorted_values(values: Any) -> list[Any]:
    """Sort the values of a column (or tuples of them), with missing values last."""
    values = list(values)

    def key(v: Any) -> Any:
        if isinstance(v, tuple):
            return tuple(key(x) for x in v)
        return (v is None, 0 if v is None else v)

    def fallback(v: Any) -> Any:
        if isinstance(v, tuple):
            return tuple(fallback(x) for x in v)
        return (v is None, type(v).__name__, repr(v))

    try:
        return sorted(values, key=key)
    except TypeError:
        return sorted(values, key=fallback)


def _allocate(n: int, sizes: list[int]) -> list[int]:
    """
    Split `n` rows among groups of these sizes in proportion to their sizes, with the groups with
    the largest remainders (then the first ones) getting the extra rows.
    """
    total = sum(sizes)
    if total == 0:
        return [0] * len(sizes)
    quotas = [n * size // total for size in sizes]
    remainders = [n * size % total for size in sizes]
    extra = n - sum(quotas)
    for k in sorted(range(len(sizes)), key=lambda k: (-remainders[k], k))[:extra]:
        quotas[k] += 1
    return quotas


def _table_fingerprint(data: Any) -> str:
    """A hash of a table's schema and values (in the order of its rows)."""
    frame = nw.from_native(data)
    schema = {name: str(dtype) for name, dtype in frame.collect_schema().items()}
    digest = hashlib.sha256(json.dumps(schema, sort_keys=False).encode())

    if isinstance(frame, nw.LazyFrame) or frame.implementation not in (
        nw.Implementation.POLARS,
        nw.Implementation.PANDAS,
    ):
        indexed = _with_row_index(data, name=_INDEX)
        if isinstance(indexed, nw.LazyFrame):
            try:
                import polars  # noqa: F401

                backend = "polars"
            except ImportError:  # pragma: no cover
                backend = "pandas"
            indexed = indexed.collect(backend=backend)
        frame = indexed.sort(_INDEX).drop(_INDEX)
        if frame.implementation not in (nw.Implementation.POLARS, nw.Implementation.PANDAS):
            try:
                frame = nw.from_native(frame.to_polars(), eager_only=True)
            except ImportError:  # pragma: no cover
                frame = nw.from_native(frame.to_pandas(), eager_only=True)

    native = frame.to_native()
    digest.update(f"rows:{len(frame)}".encode())
    try:
        if frame.implementation is nw.Implementation.POLARS:
            import array

            import polars as pl

            hashes = native.hash_rows(seed=0, seed_1=1, seed_2=2, seed_3=3)
            digest.update(f"polars:{pl.__version__}".encode())
            digest.update(array.array("Q", hashes.to_list()).tobytes())
        else:
            import pandas as pd

            hashes = pd.util.hash_pandas_object(native, index=False)
            digest.update(f"pandas:{pd.__version__}".encode())
            digest.update(hashes.to_numpy().tobytes())
    except Exception:
        # Values that can't be hashed by the library (e.g., some nested values)
        digest.update(b"repr")
        for row in frame.iter_rows():
            digest.update(repr(row).encode())
    return digest.hexdigest()


def _rows_hash(removed: frozenset[int]) -> str:
    return hashlib.sha256(",".join(map(str, sorted(removed))).encode()).hexdigest()


def _hash_json(payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _is_result(result: Any) -> bool:
    """Is this a valid result of an evaluation (as stored in the cache)?"""

    def metric(value: Any) -> bool:
        return _is_number(value) and math.isfinite(float(value))

    if isinstance(result, dict):
        return bool(result) and all(isinstance(k, str) and metric(v) for k, v in result.items())
    return metric(result)


def _stored_result(result: Any) -> Any:
    """The result of an evaluation, as it's stored in the cache."""
    if isinstance(result, dict):
        return {name: float(value) for name, value in result.items()}
    return float(result)


# The `evaluate=` function of an ablation, in each process of a pool
_WORKER_EVALUATE: Callable[..., Any] | None = None


def _set_worker_evaluate(evaluate: Callable[..., Any]) -> None:
    global _WORKER_EVALUATE
    _WORKER_EVALUATE = evaluate


def _worker_evaluate(train: Any, seed: int, variant: AblationVariant | None) -> Any:
    kwargs: dict[str, Any] = {"seed": seed}
    if variant is not None:
        kwargs["variant"] = variant
    return _WORKER_EVALUATE(train, **kwargs)


def _unique_slugs(ids: Any) -> dict[str, str]:
    """A file name for each id, unique even on file systems that ignore case."""
    slugs: dict[str, str] = {}
    used: set[str] = set()
    for id_ in ids:
        base = re.sub(r"[^A-Za-z0-9._-]+", "_", id_).strip("._") or "variant"
        slug, n = base, 2
        while slug.lower() in used:
            slug, n = f"{base}_{n}", n + 1
        used.add(slug.lower())
        slugs[id_] = slug
    return slugs


def _read_results(results: Any) -> tuple[list[dict[str, Any]], list[str], Path | None]:
    """The rows and columns of a table of results, and the directory it was read from."""
    if isinstance(results, (str, os.PathLike)):
        path = Path(results)
        if path.is_dir():
            path = path / "results.csv"
        if not path.exists():
            raise FileNotFoundError(f"There's no file of results at {str(path)!r}.")
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            rows = list(reader)
        return rows, columns, path.parent

    try:
        frame = nw.from_native(results)
    except TypeError:
        raise TypeError(
            "`results=` must be the path of a CSV file (or of a directory with a `results.csv` "
            f"file) or a table, got {type(results).__name__}."
        ) from None
    if isinstance(frame, nw.LazyFrame):
        frame = frame.collect()
    return list(frame.rows(named=True)), frame.columns, None


def _parse_int(value: Any, context: str) -> int:
    if _is_int(value):
        return int(value)
    try:
        number = float(value.strip()) if isinstance(value, str) else float(value)
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{context} {value!r}, which isn't an integer.")
    return int(number)


def _parse_float(value: Any) -> float | None:
    """A metric from a table of results (`None` if it's missing)."""
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "" or value.lower() in ("na", "nan", "null", "none"):
            return None
        try:
            return float(value)
        except ValueError:
            raise ValueError(
                f"The results have {value!r} as a metric, which isn't a number."
            ) from None
    if _is_number(value):
        number = float(value)
        return None if math.isnan(number) else number
    if type(value).__name__ == "NAType":
        return None
    raise ValueError(f"The results have {value!r} as a metric, which isn't a number.")


# ----------------------------------------------------------------------------------------------
# scikit-learn
# ----------------------------------------------------------------------------------------------

_TASKS = ("auto", "classification", "regression")

# The most distinct values of an integer target for it to be taken as classes, with `task="auto"`
_MAX_INTEGER_CLASSES = 10

# The most categories that a histogram-based gradient boosting model can treat as categorical
_MAX_CATEGORIES = 255


def _import_sklearn() -> Any:
    try:
        import sklearn
    except ImportError:
        raise ImportError(
            "`sklearn_evaluator()` needs scikit-learn, which isn't installed. Install it with "
            f"{_ML_INSTALL_HINT}."
        ) from None
    return sklearn


def _feature_kind(dtype: Any) -> str | None:
    """How a feature of this type is encoded (`None` if it can't be)."""
    if dtype == nw.String or isinstance(dtype, (nw.Categorical, nw.Enum)):
        return "categorical"
    if dtype.is_numeric() or dtype == nw.Boolean:
        return "numeric"
    return None


def sklearn_evaluator(
    holdout: Any,
    target: str,
    features: list[str] | None = None,
    estimator: Any = None,
    task: str = "auto",
    scoring: Any = None,
) -> Callable[..., dict[str, float]]:
    """
    Make an `evaluate=` function that trains a scikit-learn model and scores it on held-out data.

    An [`Ablation`](`pointblank.Ablation`) needs a function that trains a model on a version of
    the training table and scores it. For tabular models, that function is often the same: pick
    the features, encode them, fit a model, and score it on fixed held-out data. This function
    makes one, so that an ablation can start with a few lines. It needs scikit-learn, which can be
    installed with `pip install "pointblank[ml]"`.

    Parameters
    ----------
    holdout
        The held-out data that every version of the training table is scored on, with the same
        columns as the training table (any table that the validation could use). It's never
        ablated. Its rows with a missing target are left out.
    target
        The column to predict.
    features
        The columns to predict from. By default, every column except `target=` with a type that
        can be used: numbers, booleans, strings, and categoricals (there's a warning listing the
        other columns, which are left out).
    estimator
        The scikit-learn estimator (or pipeline) to train. It's cloned for each evaluation, with
        each `random_state` parameter (including those of the steps of a pipeline) set to the
        seed of the evaluation. By default, it's a histogram-based gradient boosting model
        (`HistGradientBoostingClassifier` or `HistGradientBoostingRegressor`), which handles
        missing values and treats string and categorical features as categories.
    task
        The kind of model: `"classification"`, `"regression"`, or `"auto"` (the default), which
        takes a target of strings, booleans, or categories (or of integers with at most 10
        distinct values) as classes, and any other numeric target as a regression.
    scoring
        How the model is scored: the name of a scikit-learn scorer (e.g., `"roc_auc"` or
        `"neg_mean_absolute_error"`), a scorer function `scorer(estimator, X, y)`, or a list or
        dictionary of them to get several metrics. By default, it's `"accuracy"` for
        classification and `"r2"` for regression.

    Returns
    -------
    Callable
        A function for the `evaluate=` parameter of `Ablation`, which returns a dictionary of
        scores (named by the scorers). It can be pickled, so it works with
        `run(executor="process")` as long as the estimator can be pickled.

    Details
    -------
    String and categorical features are encoded as integer codes, with the categories found in
    each version of the training table (a category that's missing from it, like a missing value,
    is encoded as a missing value). Numeric and boolean features are used as numbers. Rows of the
    training table with a missing target are left out of training.

    The scorers of scikit-learn are such that higher is better (that's why the error metrics are
    negated, as in `"neg_mean_absolute_error"`), so the default `higher_is_better=True` of
    `Ablation` is the right one for them. With several scorers, use `metric=` to say which one
    the verdicts are based on.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False)
    ```

    Here's a table of the temperatures of a process at different pressures, where some of the
    temperatures were recorded in the wrong units. The held-out data is clean.

    ```{python}
    import pointblank as pb
    import polars as pl

    def readings(n, offset):
        pressure = [((k + offset) * 37 % 101) / 10 for k in range(n)]
        return pl.DataFrame(
            {"pressure": pressure, "temp": [20 + 3 * p + (k % 7) / 7 for k, p in enumerate(pressure)]}
        )

    train = readings(400, offset=0).with_row_index().with_columns(
        temp=pl.when(pl.col("index") % 10 == 3).then(pl.col("temp") * 9 / 5 + 32).otherwise("temp")
    ).drop("index")
    holdout = readings(200, offset=500)

    validation = (
        pb.Validate(data=train)
        .col_vals_between(columns="temp", left=15, right=60)
        .interrogate()
    )

    ablation = pb.Ablation(
        validation,
        evaluate=pb.ablation.sklearn_evaluator(holdout, target="temp"),
        metric="r2",
    ).drop_each()

    ablation.run()
    ```

    The model is scored with R² by default, for a regression. Removing the readings in the wrong
    units improves it, compared with removing as many random readings.
    """
    _import_sklearn()

    holdout_nw = nw.from_native(holdout)
    if isinstance(holdout_nw, nw.LazyFrame):
        holdout_nw = holdout_nw.collect()
    if not isinstance(target, str):
        raise TypeError(f"`target=` must be a column name, got {type(target).__name__}.")
    schema = holdout_nw.schema
    if target not in schema:
        raise ValueError(f"The holdout data has no column {target!r} (`target=`).")
    if task not in _TASKS:
        raise ValueError(
            f"`task=` must be one of {', '.join(repr(t) for t in _TASKS)}; got {task!r}."
        )

    if features is None:
        features, unused = [], []
        for name, dtype in schema.items():
            if name == target:
                continue
            (features if _feature_kind(dtype) is not None else unused).append(name)
        if unused:
            warnings.warn(
                f"`sklearn_evaluator()` doesn't use the columns {', '.join(map(repr, unused))}, "
                "since their types can't be used as features. Give `features=` to choose the "
                "features yourself.",
                UserWarning,
                stacklevel=2,
            )
    else:
        if isinstance(features, str):
            features = [features]
        features = list(features)
        missing = [c for c in features if c not in schema]
        if missing:
            raise ValueError(
                f"The holdout data has no column {', '.join(map(repr, missing))} (`features=`)."
            )
        if target in features:
            raise ValueError(f"The target {target!r} can't also be a feature.")
        unusable = [c for c in features if _feature_kind(schema[c]) is None]
        if unusable:
            raise TypeError(
                f"The features {', '.join(map(repr, unusable))} have types that can't be used "
                "(only numbers, booleans, strings, and categoricals can)."
            )
    if not features:
        raise ValueError("There are no features to train the model with.")

    holdout_nw = holdout_nw.filter(~nw.col(target).is_null())
    if len(holdout_nw) == 0:
        raise ValueError(f"The holdout data has no rows with a value of {target!r}.")

    target_dtype = schema[target]
    if task == "auto":
        if target_dtype.is_float():
            task = "regression"
        elif target_dtype.is_integer():
            n_unique = holdout_nw[target].n_unique()
            task = "classification" if n_unique <= _MAX_INTEGER_CLASSES else "regression"
        elif _feature_kind(target_dtype) == "categorical" or target_dtype == nw.Boolean:
            task = "classification"
        else:
            raise TypeError(
                f"The target {target!r} has the type {target_dtype}, which isn't a class or a "
                "number. Give `task=` to say what kind of model to train."
            )
    if task == "regression" and not target_dtype.is_numeric():
        raise TypeError(f"A regression needs a numeric target, but {target!r} is {target_dtype}.")

    if scoring is None:
        scoring = "accuracy" if task == "classification" else "r2"
    _scorers(scoring)

    return _SklearnEvaluator(
        holdout=holdout_nw.to_native(),
        target=target,
        features=features,
        kinds={c: _feature_kind(schema[c]) for c in features},
        estimator=estimator,
        task=task,
        scoring=scoring,
    )


def _scorers(scoring: Any) -> dict[str, Any]:
    """The scorers of `scoring=`, by name."""
    from sklearn.metrics import get_scorer

    if isinstance(scoring, str):
        items = {scoring: scoring}
    elif callable(scoring):
        items = {getattr(scoring, "__name__", None) or "score": scoring}
    elif isinstance(scoring, (list, tuple)) and scoring:
        if not all(isinstance(s, str) for s in scoring):
            raise TypeError("A list of scorers in `scoring=` must be a list of scorer names.")
        items = {s: s for s in scoring}
    elif isinstance(scoring, dict) and scoring:
        items = dict(scoring)
    else:
        raise TypeError(
            "`scoring=` must be the name of a scorer, a scorer function, or a list or dictionary "
            f"of them; got {scoring!r}."
        )
    scorers = {}
    for name, scorer in items.items():
        if isinstance(scorer, str):
            try:
                scorer = get_scorer(scorer)
            except ValueError:
                raise ValueError(f"{scorer!r} isn't the name of a scikit-learn scorer.") from None
        elif not callable(scorer):
            raise TypeError(f"The scorer {name!r} must be a scorer name or function.")
        scorers[str(name)] = scorer
    return scorers


class _SklearnEvaluator:
    """The `evaluate=` function made by `sklearn_evaluator()`."""

    def __init__(
        self,
        holdout: Any,
        target: str,
        features: list[str],
        kinds: dict[str, str],
        estimator: Any,
        task: str,
        scoring: Any,
    ) -> None:
        self.holdout = holdout
        self.target = target
        self.features = features
        self.kinds = kinds
        self.estimator = estimator
        self.task = task
        self.scoring = scoring

    def __repr__(self) -> str:
        return (
            f"sklearn_evaluator(target={self.target!r}, features={self.features!r}, "
            f"task={self.task!r}, scoring={self.scoring!r})"
        )

    def __call__(self, train: Any, *, seed: int) -> dict[str, float]:
        import numpy as np

        frame = nw.from_native(train)
        if isinstance(frame, nw.LazyFrame):
            frame = frame.collect()
        missing = [c for c in [*self.features, self.target] if c not in frame.columns]
        if missing:
            raise ValueError(
                f"The training table has no column {', '.join(map(repr, missing))}, which "
                "`sklearn_evaluator()` needs."
            )
        frame = frame.filter(~nw.col(self.target).is_null())
        holdout = nw.from_native(self.holdout, eager_only=True)

        # The categories of each string or categorical feature, from the training table
        categories = {}
        for c in self.features:
            if self.kinds[c] == "categorical":
                values = {str(v) for v in frame[c].to_list() if v is not None}
                categories[c] = {v: k for k, v in enumerate(sorted(values))}

        def encode(table: Any) -> Any:
            columns = []
            for c in self.features:
                values = table[c].to_list()
                if c in categories:
                    codes = categories[c]
                    columns.append(
                        [codes.get(str(v), math.nan) if v is not None else math.nan for v in values]
                    )
                else:
                    columns.append([_to_float(v) for v in values])
            return np.array(columns, dtype=float).T.reshape(len(table), len(self.features))

        def labels(table: Any) -> Any:
            values = table[self.target].to_list()
            if self.task == "regression":
                return np.array([_to_float(v) for v in values], dtype=float)
            return np.array([str(v) if not isinstance(v, (bool, int)) else v for v in values])

        model = self._model(seed, categories)
        model.fit(encode(frame), labels(frame))
        X, y = encode(holdout), labels(holdout)
        return {name: float(scorer(model, X, y)) for name, scorer in _scorers(self.scoring).items()}

    def _model(self, seed: int, categories: dict[str, dict[str, int]]) -> Any:
        from sklearn.base import clone

        if self.estimator is None:
            from sklearn.ensemble import (
                HistGradientBoostingClassifier,
                HistGradientBoostingRegressor,
            )

            mask = [
                c in categories and len(categories[c]) <= _MAX_CATEGORIES for c in self.features
            ]
            cls = (
                HistGradientBoostingClassifier
                if self.task == "classification"
                else HistGradientBoostingRegressor
            )
            return cls(random_state=seed, categorical_features=mask if any(mask) else None)

        model = clone(self.estimator)
        seeds = {
            name: seed
            for name in model.get_params(deep=True)
            if name == "random_state" or name.endswith("__random_state")
        }
        if seeds:
            model.set_params(**seeds)
        return model


def _to_float(value: Any) -> float:
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan
