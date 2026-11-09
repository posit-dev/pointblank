from __future__ import annotations

import csv
import functools
import hashlib
import html as html_lib
import inspect
import itertools
import json
import math
import numbers
import os
import random
import re
import statistics
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import narwhals as nw

from pointblank._report_text import (
    ENGLISH,
    Localizer,
    Num,
    Text,
    capitalized,
    msg,
    report_localizer,
)
from pointblank._utils import _select_df_lib, _with_row_index

if TYPE_CHECKING:
    from great_tables import GT

    from pointblank.thresholds import Thresholds
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
_INTERVALS = ("t", "bootstrap")
_PASS_AS = ("table", "row_index")
# The number of resamples of the repeats for `ci="bootstrap"`, and the fewest repeats it's suited to
_BOOTSTRAP_RESAMPLES = 4000
_BOOTSTRAP_MIN_REPEATS = 10

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

# The parameters of a step that don't change which rows fail it, left out when it's run again
_RERUN_DROPPED_PARAMS = ("thresholds", "actions", "brief", "active", "dimension")

# The defects that `inject()` can put into clean rows (a function can be given too)
_CORRUPTIONS = ("null", "shuffle", "truncate", "duplicate")
_INJECT_RATES = (0.01, 0.05, 0.1)

# The tables that `inject()` can corrupt (it changes the values in memory)
_INJECT_IMPLEMENTATIONS = ("polars", "pandas", "pyarrow")

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
    "corruption": "str",
    "column": "str",
    "n_corrupted": "int",
    "checked_by": "int",
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
        [`sweep_step()`](`pointblank.Ablation.sweep_step`), `"inject:<corruption>:<column>_<rate>"`
        (or `"inject:duplicate_<rate>"`) for [`inject()`](`pointblank.Ablation.inject`),
        `"shapley_<i>_<j>"` for a group of steps that only
        [`shapley()`](`pointblank.Ablation.shapley`) needs, and `"control_<n>"` for the control
        that removes `n` random rows (or `"control_<id>"` for the control of a
        `drop_all_but_each()` variant).
    strategy
        The kind of version: `"full"`, `"drop_steps"`, `"drop_each"`, `"drop_all"`,
        `"drop_all_but_each"`, `"drop_groups"`, `"sweep_score"`, `"sweep_step"`, `"inject"`,
        `"shapley"`, or `"control"`.
    name
        A name for the version, as shown in reports (e.g., `"Step 2"`).
    steps
        The numbers of the validation steps whose failing rows were removed (empty for the full
        table, for groups, for the points of a sweep, for injected defects, and for most
        controls).
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
        threshold, or the value of the step's parameter. For a variant of `inject()`, the share of
        the rows corrupted (otherwise `None`).
    corruption
        For a variant of [`inject()`](`pointblank.Ablation.inject`), the defect put into the rows:
        `"null"`, `"shuffle"`, `"truncate"`, `"duplicate"`, or the name of the corruption function
        (otherwise `None`).
    column
        For a variant of `inject()`, the column whose values were corrupted (`None` for
        `"duplicate"`, and otherwise).
    n_corrupted
        For a variant of `inject()`, the number of rows corrupted (or copied, for `"duplicate"`).
    checked_by
        For a variant of `inject()`, the step that checks for the defect, given as its `step=`
        (otherwise `None`).

    Examples
    --------
    Here's a table of temperature readings, some of them recorded in tenths of a degree (so
    they're ten times too high). The `evaluate()` function takes the `variant=` argument and keeps
    a log of each call, as a real function might do to name the files that its models are saved
    to.

    ```{python}
    import pointblank as pb
    import polars as pl

    temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
    for k in range(0, 300, 25):
        temps[k] *= 10
    readings = pl.DataFrame({"temp": temps})

    calls = []

    def evaluate(train, *, seed, variant):
        calls.append(
            {
                "id": variant.id,
                "strategy": variant.strategy,
                "n_removed": variant.n_removed,
                "is_control": variant.is_control,
                "seed": seed,
                "n_rows": train.height,
            }
        )
        sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
        return abs(sample["temp"].mean() - 20)

    validation = (
        pb.Validate(data=readings)
        .col_vals_between(columns="temp", left=-10, right=45)
        .interrogate()
    )

    ablation = pb.Ablation(
        validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=3
    ).drop_each()
    ablation.run()

    pl.DataFrame(calls)
    ```

    The function was called three times (once for each repeat, with the seeds `0`, `1`, and `2`)
    for each of the three versions of the table: the full table (`"full"`), the variant of
    `drop_each()` that removes the 12 rows failing step 1 (`"step_1"`), and its control, which
    removes 12 random rows (`"control_12"`). The variant and its control both train on 288 rows,
    so the training table alone can't tell them apart, but `variant=` can. The variant always
    removes the same rows, while the control draws a new set of random rows for each repeat.

    See Also
    --------
    Ablation : Measure how the rows that fail validation steps affect a model trained on the data.
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
    corruption: str | None = None
    column: str | None = None
    n_corrupted: int = 0
    checked_by: int | None = None

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
            "corruption": self.corruption,
            "column": self.column,
            "n_corrupted": self.n_corrupted,
            "checked_by": self.checked_by,
        }


@dataclass(frozen=True)
class _Injection:
    """The rows that a variant of `inject()` corrupts, and their new values."""

    kind: str
    label: str
    column: str | None
    rows: tuple[int, ...]
    # The new values of the rows (empty for `"null"` and `"duplicate"`)
    values: tuple[Any, ...]
    step: int | None
    key: str


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
    # The name as a text to translate (when it isn't given by the user)
    text: Text | None = field(default=None, compare=False)

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


def _named(text: Text | str, **kwargs: Any) -> _Spec:
    """A spec named by a text to translate (its `name` is the text in English)."""
    if isinstance(text, str):
        return _Spec(name=text, **kwargs)
    return _Spec(name=ENGLISH.r(text), text=text, **kwargs)


# The notes for the reasons that a step isn't used (from `_row_result_steps()`)
_SKIP_REASON_KEYS = {
    "it couldn't be evaluated": "abl_note_step_eval_error",
    "it uses `pre=` without `pb.row_preserving()`": "abl_note_step_pre",
    "no `duplicates=` policy was given": "abl_note_step_duplicates",
    "its results can't be matched to the table's rows": "abl_note_step_tbl_match",
}


@dataclass
class _Resolved:
    """The variants of an ablation with the rows they remove, and the controls they need."""

    # The variants that are used, with the numbers of the rows they remove
    variants: list[tuple[AblationVariant, frozenset[int]]]
    removed: dict[str, frozenset[int]]
    # Notes about the variants that aren't used
    notes: list[Text]
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
    # For `shapley()`: the variant of each group of steps
    shapley: dict[tuple[int, ...], str] | None = None


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

    Here's a table of temperature readings from two sensors in different places: sensor `"a"` has
    two thirds of the readings and reads about 20 degrees, while sensor `"b"` has the other third
    and reads about 40 degrees. One reading in ten comes from a device that was never calibrated
    (`calibrated_on` is missing), and those read 1.5 degrees too high. The 'model' estimates the
    average temperature across both sensors, and it's evaluated by its error against the true
    average of the calibrated readings. It's deterministic, so the only randomness in the ablation
    is in the rows that the controls remove.

    ```{python}
    import pointblank as pb
    import polars as pl

    sensors = ["a", "a", "b"] * 100
    calibrated_on = [None if k % 10 == 3 else "2026-03-01" for k in range(300)]
    temps = [
        (20 if s == "a" else 40) + (k * 37 % 11 - 5) * 0.6 + (1.5 if c is None else 0)
        for k, (s, c) in enumerate(zip(sensors, calibrated_on))
    ]
    readings = pl.DataFrame({"sensor": sensors, "calibrated_on": calibrated_on, "temp": temps})
    true_average = readings.drop_nulls()["temp"].mean()

    def evaluate(train, *, seed):
        return abs(train["temp"].mean() - true_average)

    validation = (
        pb.Validate(data=readings)
        .col_vals_not_null(columns="calibrated_on")
        .interrogate()
    )

    def ablate(control):
        return (
            pb.Ablation(
                validation,
                evaluate=evaluate,
                metric="error",
                higher_is_better=False,
                control=control,
            )
            .drop_each()
            .run()
        )

    stratified = ablate(pb.ablation.Stratified(by="sensor"))
    stratified
    ```

    Removing the 30 uncalibrated readings helps. Each repeat of the control removes 30 random
    readings, always 20 from sensor `"a"` and 10 from sensor `"b"`, so the control has the same
    mix of sensors as the full table. Here's the same ablation with the default control, which
    removes 30 readings from anywhere in the table:

    ```{python}
    pl.concat(
        [
            ablation.get_effects()
            .filter(pl.col("variant") == "step_1")
            .select(
                pl.lit(name).alias("control"),
                "control_mean",
                "delta_vs_control",
                "delta_vs_control_lower",
                "delta_vs_control_upper",
                "verdict",
            )
            for name, ablation in [
                ("size_matched", ablate("size_matched")),
                ("stratified", stratified),
            ]
        ]
    )
    ```

    The default control's random draws take a different number of readings from each sensor in
    each repeat, and since the sensors read 20 degrees apart, that alone moves the average. The
    noise widens the interval to about [−0.23, +0.003], which includes zero, so the verdict is
    inconclusive. The stratified control removes that source of noise, and its interval of about
    [−0.19, −0.09] is narrow enough for the verdict to be that the step helps. Stratifying is
    worth it when the groups differ in what the model learns from them: with groups that are much
    alike, it makes little difference.

    See Also
    --------
    Ablation : Measure how the rows that fail validation steps affect a model trained on the data.
    Ablation.drop_groups : Remove the rows of each group, one group at a time.
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
    [`drop_groups()`](`pointblank.Ablation.drop_groups`), the sweeps
    [`sweep_score()`](`pointblank.Ablation.sweep_score`) and
    [`sweep_step()`](`pointblank.Ablation.sweep_step`),
    [`inject()`](`pointblank.Ablation.inject`), which corrupts clean rows to measure what a defect
    costs, and [`shapley()`](`pointblank.Ablation.shapley`), which splits the credit among
    overlapping steps), and nothing is evaluated until
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
        function can leave out the `variant=` parameter if it doesn't need it. With
        `pass_as="row_index"`, `train` is the list of the numbers of the rows that are kept
        instead. It must return a number, or a dictionary of named numbers (one per metric). The
        evaluation data should be a fixed holdout set that the function holds on to: it must not
        depend on `train`. For scikit-learn models,
        [`sklearn_evaluator()`](`pointblank.ablation.sklearn_evaluator`) makes this function for
        you. It can be left out (`None`) if the evaluations are done elsewhere, with
        [`export()`](`pointblank.Ablation.export`) and
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
        and `None` leaves `~~pointblank.Validate.rows_distinct()` steps out of the ablation. These
        are the same options as in [`get_row_flags()`](`pointblank.Validate.get_row_flags`).
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
    holdout
        The holdout set that `evaluate=` scores the model on (a table of any kind that
        [`Validate`](`pointblank.Validate`) takes, or the path of a file), to be checked with the
        same validation steps. The holdout isn't used in the ablation (rows are never removed from
        it), but a model can only be scored as well as its holdout allows, so the reports give its
        health score next to that of the training table. The holdout is interrogated when the
        `Ablation` is created, and the results are in the `holdout_validation` attribute (steps
        that call a model, such as those of [`prompt()`](`pointblank.Validate.prompt`), call it
        again for the holdout). With
        [`sklearn_evaluator()`](`pointblank.ablation.sklearn_evaluator`), it's the same table as
        the evaluator's `holdout=`.
    ci
        How the intervals and p-values are computed from the paired differences of the repeats.
        With `"t"` (the default), they come from a paired t-test, which assumes the differences
        are roughly normal. With `"bootstrap"`, they come from 4,000 resamples of the repeats (a
        percentile interval, and a p-value from the share of resampled means on either side of
        zero), which assumes nothing about their distribution but needs more repeats: with fewer
        than 10, it gives intervals that are too narrow (there's a warning).
    pass_as
        What `evaluate=` gets as its first argument. With `"table"` (the default), it's the
        training table with the rows removed. With `"row_index"`, it's a sorted list of the
        numbers of the rows that are kept (counting from `0`, in the order of the table, as in
        the `format="indices"` of [`export()`](`pointblank.Ablation.export`)), for a pipeline that
        loads the training data itself. This can't be used with
        [`inject()`](`pointblank.Ablation.inject`), which changes the values of the table, or
        with [`sklearn_evaluator()`](`pointblank.ablation.sklearn_evaluator`).

    Returns
    -------
    Ablation
        An `Ablation` object, with no variants yet.

    Details
    -------
    For each variant and metric, the results give the mean and standard deviation of the metric
    across the repeats, and two paired comparisons: with the full table (`delta_vs_full`) and with
    the control (`delta_vs_control`). Each comparison has an interval at `conf_level=` and a
    p-value (from a paired t-test, or from the bootstrap with `ci="bootstrap"`), along with the
    p-value adjusted across the variants. The verdict on a variant is based on the metric named in
    `metric=` and the comparison with the control (or with the full table, if `control=None`):

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
        holdout: Any = None,
        ci: str = "t",
        pass_as: str = "table",
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
        if ci not in _INTERVALS:
            raise ValueError(
                f"`ci=` must be one of {', '.join(repr(c) for c in _INTERVALS)}; got {ci!r}."
            )
        if pass_as not in _PASS_AS:
            raise ValueError(
                f"`pass_as=` must be one of {', '.join(repr(p) for p in _PASS_AS)}; got "
                f"{pass_as!r}."
            )
        if pass_as == "row_index" and isinstance(evaluate, _SklearnEvaluator):
            raise ValueError(
                '`sklearn_evaluator()` trains on the training table, so it needs `pass_as="table"`.'
            )

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
        elif ci == "bootstrap" and repeats < _BOOTSTRAP_MIN_REPEATS:
            warnings.warn(
                f'With `repeats={repeats}`, the intervals of `ci="bootstrap"` are too narrow (and '
                "its verdicts too confident), as there are too few repeats to resample. Use at "
                f'least {_BOOTSTRAP_MIN_REPEATS} repeats, or the default `ci="t"`.',
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
        self.ci = ci
        self.pass_as = pass_as
        # The validation's steps, applied to the holdout set
        self.holdout_validation: Validate | None = None
        if holdout is not None:
            try:
                self.holdout_validation = validation.set_tbl(holdout).interrogate()
            except Exception as e:
                raise ValueError(f"The `holdout=` table couldn't be validated ({e}).") from e
            self._holdout_n_rows = get_row_count(self.holdout_validation.data)

        self._n_rows = get_row_count(validation.data)
        self._steps = {step.i: step for step in steps}
        self._skipped = dict(skipped)
        self._pass_variant = evaluate is not None and _accepts_keyword(evaluate, "variant")

        self._specs: list[_Spec] = []
        self._notes: list[Text | str] = []
        # The names of the versions of the table as texts to translate, by id
        self._texts: dict[str, Text] = {"full": msg("abl_name_full")}
        self._resolved: _Resolved | None = None
        # The rows of each group, by column (for `drop_groups()`), and of each stratum (for a
        # `Stratified` control)
        self._groups: dict[str, dict[Any, frozenset[int]]] = {}
        self._strata: list[list[int]] | None = None
        # The sweeps (by id: their label and the label of their values), and for each of their
        # points, the rows it removes and a line describing them
        self._sweeps: dict[str, dict[str, Any]] = {}
        self._sweep_rows: dict[str, frozenset[int]] = {}
        self._sweep_details: dict[str, Text] = {}
        # The rows that each variant of `inject()` corrupts, and for each of them, the share of
        # failing test units that the corruption adds to its step (found when needed)
        self._injections: dict[str, _Injection] = {}
        self._caught: dict[str, float] = {}
        # The steps of `shapley()` and the groups of them it needs evaluated
        self._shapley: dict[str, Any] | None = None
        self._data_fingerprint: str | None = None
        self._reset_results()

    def __repr__(self) -> str:
        n_variants = len(self._specs)
        status = "run" if self._effects is not None else "not run"
        label = f"{self.label!r}: " if self.label else ""
        # (the variants of the groups of steps for Shapley values are made when the ablation runs)
        shapley = (
            f", Shapley values of {len(self._shapley['steps'])} steps" if self._shapley else ""
        )
        return (
            f"Ablation({label}{n_variants} variant{'s' if n_variants != 1 else ''}{shapley}, "
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

        Here's a table of temperature readings with two kinds of bad readings: some were recorded
        in tenths of a degree (so they're ten times too high), and some are the error code `-40`
        that the sensor reports when it fails. Step 1 checks for readings that are too high, and
        step 2 for readings that are too low. The 'model' estimates the average temperature and is
        evaluated by its error against the true average of `20`.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        for k in range(7, 300, 40):
            temps[k] = -40
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed, variant):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_lt(columns="temp", value=100)
            .col_vals_gt(columns="temp", value=-30)
            .interrogate()
        )

        ablation = (
            pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
            .drop_each()
            .drop_steps([1, 2], name="Both range checks")
            .run()
        )

        ablation
        ```

        Removing the readings in tenths (step 1) helps, and removing them along with the error
        codes (`"Both range checks"`) also helps, giving the lowest error: less than a tenth of a
        degree, down from about 6 degrees. But removing only the error codes (step 2) is
        inconclusive, with an interval that includes zero. The readings in tenths pull the average
        up while the error codes pull it down, so they partly offset each other: without the error
        codes, the readings in tenths pull the average up even more. The effect of a combination
        of steps isn't the sum of the effects of its steps, which is why it's worth testing
        combinations like this one.

        See Also
        --------
        Ablation.drop_each : Remove the rows failing each step, one step at a time.
        Ablation.drop_all : Remove the rows failing any step.
        """
        steps = self._check_steps(steps, method="drop_steps")
        text = None
        if name is None:
            text = (
                msg("abl_name_step", i=str(steps[0]))
                if len(steps) == 1
                else msg("abl_name_steps", steps=steps)
            )
            name = ENGLISH.r(text)
            spec_id = "steps_" + "_".join(str(i) for i in steps)
        else:
            if not isinstance(name, str) or not name:
                raise TypeError("`name=` must be a non-empty string.")
            spec_id = name
        self._add_spec(_Spec(id=spec_id, name=name, strategy="drop_steps", steps=steps, text=text))
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

        Here's a table of temperature readings with missing values and with readings that were
        recorded in tenths of a degree (so they're ten times too high). The 'model' estimates the
        average temperature, ignoring missing readings, and is evaluated by its error against the
        true average of `20`. The validation has a step for each problem, and a third step that
        flags readings below `19` degrees as suspect.

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
            .col_vals_gt(columns="temp", value=19, na_pass=True)
            .interrogate()
        )

        ablation = (
            pb.Ablation(
                validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=10
            )
            .drop_each()
            .run()
        )

        ablation
        ```

        Each step tells a different story. Removing the missing readings (step 1) is
        inconclusive, since the model ignores them anyway. Removing the readings in tenths (step 2)
        helps a lot. And removing the readings below `19` degrees (step 3) hurts: they're good
        readings, and without them the estimate is too high. Step 3 is too strict for this model,
        and it removes about a third of the rows. Each variant removes a different number of rows,
        so each has its own control. With so few rows removed by steps 1 and 2, the ablation uses
        `repeats=10` to measure the effects precisely enough.

        See Also
        --------
        Ablation.drop_all_but_each : Remove the rows failing every step but one, for each step.
        Ablation.drop_all : Remove the rows failing any step.
        Ablation.shapley : Share the effect of the validation among its steps.
        """
        if steps is None:
            steps = self._all_steps(method="drop_each")
        else:
            steps = self._check_steps(steps, method="drop_each")
        for i in steps:
            self._add_spec(
                _named(
                    msg("abl_name_step", i=str(i)), id=f"step_{i}", strategy="drop_each", steps=(i,)
                )
            )
        return self

    def drop_all(self, steps: list[int] | None = None) -> Ablation:
        """
        Add a variant that removes the rows failing any step.

        This variant answers the question 'does the validation as a whole help?' Without
        `~~pointblank.Validate.rows_distinct()` steps, its training table is the one that
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

        Here's a table of temperature readings with missing values and with readings that were
        recorded in tenths of a degree (so they're ten times too high). The 'model' estimates the
        average temperature, ignoring missing readings, and is evaluated by its error against the
        true average of `20`. The validation has a step for each problem, and a third step that
        flags readings below `19` degrees as suspect. The ablation tests the validation as a
        whole, and then only its first two steps.

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
            .col_vals_gt(columns="temp", value=19, na_pass=True)
            .interrogate()
        )

        ablation = (
            pb.Ablation(
                validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=10
            )
            .drop_all()
            .drop_all(steps=[1, 2])
            .run()
        )

        ablation
        ```

        The validation as a whole helps: removing the 121 rows that failed any step lowers the
        error from about 7.3 degrees to about 1.2. But the validation does even better without its
        third step: removing only the 20 rows that failed steps 1 or 2 lowers the error to less
        than a tenth of a degree. A validation that helps can still have steps that hurt, and
        [`drop_each()`](`pointblank.Ablation.drop_each`) shows which ones they are.

        See Also
        --------
        Ablation.drop_each : Remove the rows failing each step, one step at a time.
        Ablation.drop_all_but_each : Remove the rows failing every step but one, for each step.
        Ablation.drop_steps : Remove the rows failing any of the given steps.
        """
        if steps is None:
            steps = self._all_steps(method="drop_all")
            spec = _named(msg("abl_name_all_steps"), id="all", strategy="drop_all", steps=steps)
        else:
            steps = self._check_steps(steps, method="drop_all")
            spec = _named(
                msg("abl_name_all_of_steps", steps=steps),
                id="all_" + "_".join(str(i) for i in steps),
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

        See Also
        --------
        Ablation.drop_each : Remove the rows failing each step, one step at a time.
        Ablation.drop_all : Remove the rows failing any step.
        Ablation.shapley : Share the effect of the validation among its steps.
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
        for k in base:
            rest = tuple(i for i in base if i != k)
            if steps is None:
                spec_id, text = f"all_but_{k}", msg("abl_name_all_but_step", k=str(k))
            else:
                spec_id = f"all_but_{k}_of_" + "_".join(str(i) for i in base)
                text = msg("abl_name_steps_but_step", steps=base, k=str(k))
            self._add_spec(
                _named(text, id=spec_id, strategy="drop_all_but_each", steps=rest, left_out=k)
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

        See Also
        --------
        ablation.Stratified : Draw the rows of controls in proportion to groups of rows.
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
                spec_id, text = f"{column}=null", msg("abl_name_group_missing", column=column)
            else:
                spec_id, text = f"{column}={value}", f"{column} = {value}"
            self._add_spec(
                _named(
                    text,
                    id=spec_id,
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

        See Also
        --------
        Ablation.sweep_step : Vary a parameter of a step, removing the rows that fail it.
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
            missing_label = msg("abl_note_no_score_column", column=score)
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
            base_id, base_name = f"step_{score}", msg("abl_name_step", i=str(score))
            missing_label = msg("abl_note_no_score_step", step=str(score))
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
            self._add_note(Text(missing_label.key, (*missing_label.fields, ("n", n_missing))))

        highest = remove == "highest"
        if values is None:
            sweep_id = f"{base_id}:{remove}"
            self._sweeps[sweep_id] = {
                "label": msg(f"abl_sweep_{remove}", base=base_name),
                "axis": msg("abl_axis_scored_removed"),
                "percent": True,
            }
            ranked = sorted(scores.items(), key=lambda kv: (-kv[1] if highest else kv[1], kv[0]))
            for q in quantiles:
                n = math.floor(q * n_scored + 0.5)
                spec_id = f"{sweep_id}_{_number_text(q)}"
                self._sweep_rows[spec_id] = frozenset(k for k, _ in ranked[:n])
                self._sweep_details[spec_id] = msg(f"abl_detail_{remove}", n=n, total=n_scored)
                self._add_spec(
                    _named(
                        msg(f"abl_name_{remove}", base=base_name, pct=Num(q * 100, "g", "%")),
                        id=spec_id,
                        strategy="sweep_score",
                        sweep=sweep_id,
                        value=float(q),
                    )
                )
        else:
            side = "above" if highest else "below"
            sweep_id = f"{base_id}:{side}"
            self._sweeps[sweep_id] = {
                "label": msg(f"abl_sweep_{side}", base=base_name),
                "axis": msg("abl_axis_cutoff"),
                "percent": False,
            }
            for v in values:
                text = _number_text(v)
                spec_id = f"{sweep_id}_{text}"
                self._sweep_rows[spec_id] = frozenset(
                    k for k, x in scores.items() if (x > v if highest else x < v)
                )
                self._sweep_details[spec_id] = msg(f"abl_detail_{side}", value=Num(v))
                self._add_spec(
                    _named(
                        msg(f"abl_name_{side}", base=base_name, value=Num(v)),
                        id=spec_id,
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
        `~~pointblank.Validate.col_vals_scored()` step takes its scores from the score cache instead
        of scoring the rows again. For a segmented step, only the step's own segment is checked. A
        value that the validation method doesn't accept (e.g., a `min_val=` above the step's
        `max_val=`) gives an error. As with any other variant, a value whose step fails no rows is
        left out, and a value that removes the same rows as another version of the table isn't
        evaluated again.

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

        Every limit gets the verdict that it helps, though, since each verdict compares a point of
        the sweep with its own control (removing as many random rows), not with the other points.
        Even the limit of `21` does better than removing 117 random rows, which would leave most of
        the bad readings in. To pick a limit, compare the points' metrics with each other, as the
        curve below the report does: the error is lowest at `50`.

        See Also
        --------
        Ablation.sweep_score : Remove the rows on the wrong side of each cutoff of a score.
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
        for key in _RERUN_DROPPED_PARAMS:
            kwargs.pop(key, None)
        if info.segments is not None:
            kwargs["segments"] = info.segments

        sweep_id = f"step_{step}:{param}"
        self._sweeps[sweep_id] = {
            "label": msg("abl_sweep_step", step=str(step), param=param),
            "axis": msg("abl_axis_param", param=param, step=str(step)),
            "percent": False,
        }
        for v in values:
            text = _number_text(v)
            spec_id = f"{sweep_id}={text}"
            self._sweep_rows[spec_id] = self._rerun_failed_rows(
                method, {**kwargs, param: v}, step=step, setting=f"{param}={text}"
            )
            setting = f"{param}={text}"
            self._sweep_details[spec_id] = msg(
                "abl_detail_step_setting", step=str(step), setting=setting
            )
            self._add_spec(
                _named(
                    msg("abl_name_step_setting", step=str(step), setting=setting),
                    id=spec_id,
                    strategy="sweep_step",
                    sweep=sweep_id,
                    value=float(v),
                )
            )
        return self

    def inject(
        self,
        corruption: str | Callable[[Any], Any],
        column: str | None = None,
        step: int | None = None,
        rates: list[float] | None = None,
        name: str | None = None,
    ) -> Ablation:
        """
        Add a variant for each rate of a defect, put into clean rows of the table.

        The other strategies remove the rows that failed a check, to see whether they matter. This
        one runs the experiment the other way around: it takes rows that pass every check and
        gives them a defect (e.g., a missing value) at a few rates, to see how much the defect
        costs the model. The results trace how the model gets worse as the defect gets more common
        (a dose-response curve), and the report plots the curve. When the defect is one that a
        step checks for (given as `step=`), this measures what a failing test unit of the step
        costs, which [`suggest_thresholds()`](`pointblank.Ablation.suggest_thresholds`) turns into
        thresholds for the step.

        Parameters
        ----------
        corruption
            The defect to put into the rows:

            - `"null"`: replace the value with a missing value
            - `"shuffle"`: swap the values of the corrupted rows among themselves (e.g., to
            shuffle the labels of a classification target)
            - `"truncate"`: cut a text value to the first half of its characters
            - `"duplicate"`: add a copy of each corrupted row (whole rows, so there's no
            `column=`)

            It can also be a function that takes a value and returns the corrupted value (e.g.,
            `lambda x: x * 10`, for a reading recorded in the wrong unit).
        column
            The column to corrupt. By default, it's the column that `step=` checks (if it checks
            a single column).
        step
            The number of the step that checks for the defect (e.g., a
            [`col_vals_not_null()`](`pointblank.Validate.col_vals_not_null`) step, for `"null"`).
            It must be a row-level step that the ablation can use. `suggest_thresholds()` only
            suggests thresholds for the steps given here.
        rates
            The shares of the table's rows to corrupt, each greater than `0` and at most `1`. By
            default, they're `0.01`, `0.05`, and `0.1`.
        name
            A name for the defect, used in the ids and names of the variants (e.g., `"unit_error"`).
            By default, it's `corruption=`, or the name of the function.

        Returns
        -------
        Ablation
            The `Ablation` object, with the variants added.

        Details
        -------
        The corrupted rows are drawn at random (with the ablation's `seed=`) from the clean rows:
        the rows that fail none of the ablation's steps and have a value in `column=` (a text
        value with at least one character, for `"truncate"`). The draws are nested, so the rows
        corrupted at one rate are also corrupted at every higher rate, and they're the same in
        every repeat. A rate that needs more rows than there are clean rows, or that rounds to no
        rows, is left out, with a note. The copies of `"duplicate"` rows are added at the end of
        the table. With `"shuffle"`, the values move around in a ring, so no row keeps its own
        value (though it can get an equal one).

        Each variant is compared with the full table rather than with a control, since no rows are
        removed. A verdict of `"hurts"` means that the defect makes the model clearly worse. The
        table is corrupted in memory, so this strategy needs a Polars, pandas, or PyArrow table
        (or a Polars `LazyFrame`). The versions of the table can be exported with
        [`export()`](`pointblank.Ablation.export`), but only as Parquet files.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's a table of temperature readings with no missing values, and a 'model' that
        estimates the average temperature, filling in a missing reading with `0`. It's evaluated
        by its error against the true average of `20`.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed):
            sample = train.fill_null(0).sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_not_null(columns="temp")
            .interrogate()
        )

        ablation = (
            pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
            .inject("null", step=1)
            .run()
        )

        ablation
        ```

        Each missing reading pulls the estimate down, so the error grows with the share of
        missing readings, by about 0.2 degrees for each percent. That's the cost of a failing
        test unit of step 1, which
        [`suggest_thresholds()`](`pointblank.Ablation.suggest_thresholds`) uses. Making 5% or 10%
        of the readings missing clearly hurts, but the verdict for 1% is inconclusive: its effect
        (about 0.14 degrees) is small next to the noise of the bootstrapped estimate. A small
        effect needs more `repeats=` to be measured.

        See Also
        --------
        Ablation.suggest_thresholds : Suggest thresholds from what failing test units cost.
        """
        if isinstance(corruption, str):
            if corruption not in _CORRUPTIONS:
                raise ValueError(
                    f"`corruption=` must be one of {', '.join(map(repr, _CORRUPTIONS))}, or a "
                    f"function; got {corruption!r}."
                )
            kind = corruption
        elif callable(corruption):
            kind = "function"
        else:
            raise TypeError(
                "`corruption=` must be the name of a defect or a function, got "
                f"{type(corruption).__name__}."
            )
        if name is not None and (not isinstance(name, str) or not name):
            raise TypeError("`name=` must be a non-empty string.")
        label = name or (_function_name(corruption) if kind == "function" else kind)

        if step is not None:
            if not _is_int(step):
                raise TypeError(f"`step=` must be a step number, got {step!r}.")
            if step not in self._steps:
                raise ValueError(
                    f"`inject()` can't use step {step}: {self._ineligible_reason(step)}."
                )
        if kind == "duplicate":
            if column is not None:
                raise ValueError(
                    '`column=` isn\'t used with `corruption="duplicate"`, which copies whole rows.'
                )
        else:
            if column is None and step is not None:
                checked = self._steps[step].column
                if isinstance(checked, list) and len(checked) == 1:
                    checked = checked[0]
                if isinstance(checked, str):
                    column = checked
            if column is None:
                reason = f" (step {step} doesn't check a single column)" if step is not None else ""
                raise ValueError(f"`inject()` needs the column to corrupt, as `column=`{reason}.")
            if not isinstance(column, str):
                raise TypeError(f"`column=` must be the name of a column, got {column!r}.")
            if column not in self._columns:
                raise ValueError(f"The column {column!r} isn't in the table.")

        rates = self._check_sweep_values(list(_INJECT_RATES) if rates is None else rates, "rates")
        for rate in rates:
            if not 0 < rate <= 1:
                raise ValueError(
                    f"The values of `rates=` must be greater than 0 and at most 1, got {rate!r}."
                )
        if self.pass_as == "row_index":
            raise ValueError(
                "`inject()` changes the values of the table, which a list of row numbers can't "
                'describe, so it needs `pass_as="table"`.'
            )
        implementation = nw.from_native(self.validation.data).implementation
        if implementation.name.lower() not in _INJECT_IMPLEMENTATIONS:
            raise ValueError(
                "`inject()` corrupts the table in memory, which works for Polars, pandas, and "
                f"PyArrow tables, but not for this table ({type(self.validation.data).__name__})."
            )

        frame = _with_row_index(self.validation.data, name=_INDEX)
        dtype = None
        if kind != "duplicate":
            frame = frame.select(_INDEX, column)
            dtype = frame.collect_schema()[column]
            if kind == "truncate" and dtype != nw.String:
                raise TypeError(
                    f'`corruption="truncate"` needs a text column, but {column!r} is {dtype}.'
                )
        else:
            frame = frame.select(_INDEX)
        if isinstance(frame, nw.LazyFrame):
            frame = frame.collect()
        rows = [int(k) for k in frame[_INDEX].to_list()]
        values = dict(zip(rows, frame[column].to_list())) if kind != "duplicate" else {}

        # The clean rows: those that fail none of the steps (and have a value to corrupt)
        dirty = frozenset().union(*self._failed_rows(sorted(self._steps)).values())
        pool = [
            k
            for k in rows
            if k not in dirty
            and (
                kind == "duplicate"
                or (not _is_missing(values[k]) and (kind != "truncate" or values[k] != ""))
            )
        ]
        random.Random(f"pointblank-ablation-inject:{self.seed}:{label}:{column}").shuffle(pool)

        sweep_id = f"inject:{label}" if kind == "duplicate" else f"inject:{label}:{column}"
        description: Text | str = {
            "null": msg("abl_defect_null", column=column),
            "shuffle": msg("abl_defect_shuffle", column=column),
            "truncate": msg("abl_defect_truncate", column=column),
            "duplicate": msg("abl_defect_duplicate") if label == "duplicate" else label,
            "function": msg("abl_defect_function", name=label, column=column),
        }[kind]
        self._sweeps[sweep_id] = {
            "label": description,
            "axis": msg("abl_axis_copied" if kind == "duplicate" else "abl_axis_corrupted"),
            "percent": True,
            "inject": True,
        }
        title = capitalized(description)
        for rate in rates:
            n = math.floor(rate * self._n_rows + 0.5)
            pct = Num(rate * 100, "g", "%")
            spec_id = f"{sweep_id}_{_number_text(rate)}"
            spec_text = msg("abl_name_rate", name=title, pct=pct)
            if n < (2 if kind == "shuffle" else 1):
                self._add_note(msg("abl_note_too_few", name=spec_text, pct=pct, total=self._n_rows))
                continue
            if n > len(pool):
                self._add_note(
                    msg("abl_note_needs_clean", name=spec_text, n=n, available=len(pool))
                )
                continue
            chosen = pool[:n]
            new_values = _corrupted_values(kind, corruption, [values.get(k) for k in chosen])
            if new_values:
                _check_corrupted_values(new_values, column, dtype, implementation, kind)
            pairs = sorted(zip(chosen, new_values or [None] * n), key=lambda pair: pair[0])
            injection = _Injection(
                kind=kind,
                label=label,
                column=column,
                rows=tuple(k for k, _ in pairs),
                values=tuple(v for _, v in pairs) if new_values else (),
                step=step,
                key=_hash_json(
                    {
                        "kind": kind,
                        "label": label,
                        "column": column,
                        "rows": _rows_hash(frozenset(chosen)),
                        "values": hashlib.sha256(
                            repr([v for _, v in pairs] if new_values else []).encode("utf-8")
                        ).hexdigest(),
                    }
                ),
            )
            self._injections[spec_id] = injection
            self._caught.pop(spec_id, None)
            detail = {
                "null": msg("abl_detail_null", n=n, column=column),
                "shuffle": msg("abl_detail_shuffle", n=n, column=column),
                "truncate": msg("abl_detail_truncate", n=n, column=column),
                "duplicate": msg("abl_detail_duplicate", n=n),
                "function": msg("abl_detail_function", n=n, name=label, column=column),
            }[kind]
            if step is not None:
                detail = msg("abl_detail_checked_by", detail=detail, step=str(step))
            self._sweep_details[spec_id] = detail
            self._add_spec(
                _named(
                    spec_text,
                    id=spec_id,
                    strategy="inject",
                    sweep=sweep_id,
                    value=float(rate),
                )
            )
        return self

    def shapley(
        self,
        steps: int | list[int] | None = None,
        permutations: int = 200,
        seed: int | None = None,
    ) -> Ablation:
        """
        Split the effect of removing the steps' rows fairly among the steps (Shapley values).

        When steps fail many of the same rows, the variants of
        [`drop_each()`](`pointblank.Ablation.drop_each`) can't say which step deserves the
        credit: each one looks useful on its own, and
        [`drop_all_but_each()`](`pointblank.Ablation.drop_all_but_each`) shows each adding little
        on top of the others. Shapley values settle it the way game theory splits a payout among
        players: each step gets its average contribution over the orders in which the steps could
        be added. Steps that remove the same rows share the credit for them, and the values of
        the steps add up to the effect of removing every step's rows. This method adds the
        variants that the values need (one for each group of steps), and after the run, the
        report shows the values in a group of their own. They're also available from
        [`get_shapley()`](`pointblank.Ablation.get_shapley`).

        Parameters
        ----------
        steps
            The steps to split the effect among, as a list of step numbers. By default, it's every
            step that the ablation can use. Steps that no row failed are left out (with a note),
            and at least two steps must remain.
        permutations
            The number of random orders of the steps to average over, when there are too many
            steps to use every order. With `k` steps, there are `k!` orders: when that's at most
            `permutations=` (with the default of `200`, for up to 5 steps), every group of steps is
            evaluated and the values are exact.
        seed
            The seed of the random orders. By default, it's the `seed=` of the ablation.

        Returns
        -------
        Ablation
            The `Ablation` object, with the variants added (replacing those of an earlier
            `shapley()`).

        Details
        -------
        The value of a group of steps is the effect of removing the rows that fail any of them,
        compared with the control (or with the full table, with `control=None`), as for the
        verdicts of the other variants. The Shapley value of a step is the average of what it adds
        to the value of the steps before it, over the orders of the steps. The exact values need
        every group of the `k` steps (`2^k - 1` of them), and the estimated ones need the groups
        that start each random order. Each group is a variant (with its control) evaluated
        `repeats=` times, so the values can take many evaluations: check
        [`plan()`](`pointblank.Ablation.plan`) first. A variant of another strategy that removes
        the rows of the same steps is reused, and a group that removes the same rows as another
        isn't evaluated again.

        The values are computed for each repeat, so each has an interval and a p-value across the
        repeats, and a verdict like that of a variant (`"helps"`, `"hurts"`, or
        `"inconclusive"`), with the p-values adjusted across the steps. The groups of steps are
        listed in the plan, and in the results of [`get_effects()`](`pointblank.Ablation.get_effects`)
        (with the strategy `"shapley"`), but not in the report.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's a table of temperature readings, some of them recorded in tenths of a degree and
        some with the sentinel value `-999`. Step 1 catches the sentinels, and step 2 catches both
        kinds of bad readings, so the steps overlap.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        for k in range(7, 300, 40):
            temps[k] = -999
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_gt(columns="temp", value=-50)
            .col_vals_between(columns="temp", left=-50, right=50)
            .interrogate()
        )

        ablation = (
            pb.Ablation(
                validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=10
            )
            .shapley()
            .run()
        )

        ablation
        ```

        Both steps remove the sentinels, so they share the credit for them equally. Step 2 also
        gets all the credit for the readings in tenths of a degree, which only it catches, so its
        Shapley value is the larger one. As the note at the bottom says, the two values add up to
        the effect of removing the rows of both steps.

        See Also
        --------
        Ablation.get_shapley : Get the Shapley values of the steps as a table.
        Ablation.drop_all_but_each : Remove the rows failing every step but one, for each step.
        """
        if seed is None:
            seed = self.seed
        elif not _is_int(seed):
            raise TypeError(f"`seed=` must be an integer, got {seed!r}.")
        if not _is_int(permutations) or permutations < 1:
            raise ValueError(f"`permutations=` must be a positive integer, got {permutations!r}.")
        players = (
            self._all_steps("shapley") if steps is None else self._check_steps(steps, "shapley")
        )

        failed = self._failed_rows(list(players))
        for i in players:
            if not failed[i]:
                self._add_note(msg("abl_note_no_shapley", i=str(i)))
        players = tuple(i for i in players if failed[i])
        if len(players) < 2:
            raise ValueError(
                "`shapley()` needs at least two steps that rows failed, to split the effect among."
            )
        if len(frozenset().union(*(failed[i] for i in players))) >= self._n_rows:
            raise ValueError("`shapley()` can't use these steps: together, they remove every row.")

        k = len(players)
        exact = math.factorial(k) <= permutations
        orders = None
        if exact:
            coalitions = [
                c for size in range(1, k + 1) for c in itertools.combinations(players, size)
            ]
        else:
            rng = random.Random(f"pointblank-ablation-shapley:{seed}")
            orders = []
            for _ in range(permutations):
                order = list(players)
                rng.shuffle(order)
                orders.append(tuple(order))
            needed = {tuple(sorted(order[:m])) for order in orders for m in range(1, k + 1)}
            coalitions = sorted(needed, key=lambda c: (len(c), c))

        self._shapley = {
            "steps": players,
            "n_failed": {i: len(failed[i]) for i in players},
            "exact": exact,
            "permutations": permutations,
            "seed": seed,
            "orders": orders,
            "coalitions": coalitions,
        }
        self._resolved = None
        self._reset_results()
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
            key = _SKIP_REASON_KEYS.get(reason)
            self._add_note(
                msg(key, i=str(i)) if key is not None else f"Step {i} isn't used: {reason}."
            )
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

    def _text(self, version: Any) -> Text | str:
        """The name of a version of the table (or of its id), as a text to translate."""
        if isinstance(version, str):
            return self._texts.get(version, version)
        return self._texts.get(version.id, version.name)

    def _add_spec(self, spec: _Spec) -> None:
        if spec.id == "full" or spec.id.startswith(("control_", "shapley_")):
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
                        msg(
                            "abl_note_same_rows",
                            name=spec.text or spec.name,
                            other=existing.text or existing.name,
                        )
                    )
                return
        self._specs.append(spec)
        if spec.text is not None:
            self._texts[spec.id] = spec.text
        self._resolved = None
        self._reset_results()

    def _add_note(self, note: Text | str) -> None:
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

        players = self._shapley["steps"] if self._shapley is not None else ()
        failed = self._failed_rows(
            sorted({i for spec in self._specs for i in spec.steps} | set(players))
        )
        n_rows = self._n_rows
        notes: list[Text] = []

        usable = []
        for spec in self._specs:
            if spec.strategy == "inject":
                # An injected defect removes no rows (it corrupts some)
                usable.append((spec, frozenset()))
                continue
            if spec.sweep is not None:
                removed = self._sweep_rows[spec.id]
            elif spec.group_column is not None:
                removed = self._group_rows(spec.group_column).get(spec.group, frozenset())
            else:
                removed = frozenset().union(*(failed[i] for i in spec.steps))
            if len(removed) >= n_rows:
                notes.append(msg("abl_note_removes_all", name=self._text(spec)))
                continue
            # A `drop_all_but_each()` variant can remove no rows (when the left-out step is the
            # only one that rows failed): it's then the same as the full table
            if not removed and spec.left_out is None:
                notes.append(msg("abl_note_removes_none", name=self._text(spec)))
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
                        msg(
                            "abl_note_reference_unused",
                            name=self._text(spec),
                            reference=self._text(reference),
                        )
                    )
                    continue
                references[spec.id] = reference.id
            n_removed = len(removed)
            f_removed = n_removed / n_rows
            injection = self._injections.get(spec.id) if spec.strategy == "inject" else None
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
                corruption=injection.kind if injection else None,
                column=injection.column if injection else None,
                n_corrupted=len(injection.rows) if injection else 0,
                checked_by=injection.step if injection else None,
            )
            variants.append((variant, removed))

        # The groups of steps of `shapley()`, reusing the variants that remove the same steps' rows
        shapley_ids = None
        if self._shapley is not None:
            shapley_ids = {}
            by_steps = {
                variant.steps: variant.id
                for variant, _ in variants
                if variant.left_out is None
                and variant.sweep is None
                and variant.group_column is None
            }
            for coalition in self._shapley["coalitions"]:
                if coalition in by_steps:
                    shapley_ids[coalition] = by_steps[coalition]
                    continue
                removed = frozenset().union(*(failed[i] for i in coalition))
                text = (
                    msg("abl_name_steps", steps=coalition)
                    if len(coalition) > 1
                    else msg("abl_name_step", i=str(coalition[0]))
                )
                variant = AblationVariant(
                    id="shapley_" + "_".join(map(str, coalition)),
                    strategy="shapley",
                    name=ENGLISH.r(text),
                    steps=coalition,
                    n_removed=len(removed),
                    f_removed=len(removed) / n_rows,
                )
                variants.append((variant, removed))
                self._texts[variant.id] = text
                shapley_ids[coalition] = variant.id

        # A `drop_all_but_each()` variant, a sweep's point, or a group of steps of `shapley()` that
        # removes the same rows as another version of the table isn't evaluated again
        same_as = {}
        first_with_rows: dict[frozenset[int], str] = {}

        def independent(variant: AblationVariant) -> bool:
            return (
                variant.left_out is None and variant.sweep is None and variant.strategy != "shapley"
            )

        for variant, removed in variants:
            if independent(variant):
                first_with_rows.setdefault(removed, variant.id)
        for variant, removed in variants:
            if independent(variant) or variant.strategy == "inject":
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
            for n in sorted(
                {v.n_removed for v, _ in variants if v.left_out is None and v.strategy != "inject"}
            ):
                text = msg("abl_name_random_rows", n=n)
                self._texts[f"control_{n}"] = text
                control = AblationVariant(
                    id=f"control_{n}",
                    strategy="control",
                    name=ENGLISH.r(text),
                    n_removed=n,
                    f_removed=n / n_rows,
                    is_control=True,
                )
                controls.append(control)
                control_draws[control.id] = (frozenset(), n, str(n))
            for variant, removed in variants:
                if variant.strategy == "inject":
                    continue
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
                text = msg("abl_name_plus_random", name=self._text(variant), n=n_random)
                self._texts[f"control_{variant.id}"] = text
                control = AblationVariant(
                    id=f"control_{variant.id}",
                    strategy="control",
                    name=ENGLISH.r(text),
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
            shapley=shapley_ids,
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

    def _training_frame(
        self, removed: frozenset[int], injection: _Injection | None = None
    ) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
        data_nw = _with_row_index(self.validation.data, name=_INDEX)
        if injection is not None:
            data_nw = _apply_injection(data_nw, injection)
        if removed:
            data_nw = data_nw.filter(~nw.col(_INDEX).is_in(sorted(removed)))
        return data_nw.drop(_INDEX)

    def _training_table(self, removed: frozenset[int], injection: _Injection | None = None) -> Any:
        """
        The target table without the `removed` rows (and with the defect of an `inject()`
        variant), of the same type as the target table.
        """
        return self._training_frame(removed, injection).to_native()

    def _injection(self, variant: AblationVariant) -> _Injection | None:
        return self._injections.get(variant.id) if variant.strategy == "inject" else None

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
                [variant.id, r, self.seed + r, self._version_hash(variant, r)]
                for variant, r in self._tasks()
            ],
            "same_as": self._resolve().same_as,
        }
        return _hash_json(payload)

    def _version_hash(self, variant: AblationVariant, repeat: int) -> str:
        """A hash of the version of the table that a variant evaluates in a repeat."""
        injection = self._injection(variant)
        if injection is not None:
            return f"inject:{injection.key}"
        return _rows_hash(self._removed_rows(variant, repeat))

    def _task_key(self, variant: AblationVariant, repeat: int) -> str:
        """The key of an evaluation in the cache."""
        return self._cache_key(
            self._removed_rows(variant, repeat), self.seed + repeat, self._injection(variant)
        )

    def _cache_key(
        self, removed: frozenset[int], seed: int, injection: _Injection | None = None
    ) -> str:
        payload = {
            "version": _CACHE_VERSION,
            "data": self._fingerprint(),
            "n_rows": self._n_rows,
            "removed": _rows_hash(removed),
            "seed": seed,
            "evaluate_version": self.evaluate_version,
        }
        if injection is not None:
            payload["injected"] = injection.key
        return _hash_json(payload)

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
        self._shapley_values: list[dict[str, Any]] | None = None
        self._shapley_total: dict[str, float] = {}
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
            "ci": self.ci,
            "pass_as": self.pass_as,
            "evaluate_version": self.evaluate_version,
            "n_rows": self._n_rows,
            "tbl_name": getattr(self.validation, "tbl_name", None),
        }

    def _full_variant(self) -> AblationVariant:
        return AblationVariant(id="full", strategy="full", name=ENGLISH.r(self._texts["full"]))

    def _holdout_summary(self) -> dict[str, Any] | None:
        """The results of the validation's steps on the holdout set (`None` without one)."""
        if self.holdout_validation is None:
            return None
        steps = []
        for info in self.holdout_validation.validation_info:
            failed = info.eval_error or info.n is None
            steps.append(
                {
                    "step": info.i,
                    "n": None if failed else info.n,
                    "n_failed": None if failed else info.n_failed,
                    "f_failed": None if failed else info.f_failed,
                }
            )
        return {
            "n_rows": self._holdout_n_rows,
            "health_score": self.holdout_validation.get_health_score(),
            "steps": steps,
        }

    def _holdout_html(self, ctx: Localizer = ENGLISH) -> str:
        """A line for the subtitle of a report, with the health score of the holdout set."""
        summary = self._holdout_summary()
        if summary is None:
            return ""
        text = ctx.t(
            "abl_holdout",
            train=ctx.marks(f"{self.validation.get_health_score():.2f}"),
            rows=ctx.t("n_rows", n=summary["n_rows"]),
            holdout=ctx.marks(f"{summary['health_score']:.2f}"),
        )
        return f"<div>{text}</div>"

    def plan(self, lang: str | None = None, locale: str | None = None) -> GT:
        """
        Show the versions of the table that the ablation will evaluate.

        The plan is a dry run: it finds the rows that each variant removes, without evaluating
        anything. It lists the full table, each variant, and each control, with the number and
        share of rows they remove, and gives the number of evaluations that
        [`run()`](`pointblank.Ablation.run`) will make in its header. Variants that won't be used
        (e.g., because they remove no rows) are listed in the notes at the bottom.

        Parameters
        ----------
        lang
            The language of the plan's text, as a two-letter language code (e.g., `"fr"` or
            `"de"`; see the `lang=` argument of [`Validate`](`pointblank.Validate`) for the
            languages). By default, the plan is in the language of the validation.
        locale
            The locale for the numbers in the plan (e.g., `"de-CH"`), which sets the decimal mark
            and the thousands separator. By default, it's the locale of the validation (when the
            plan is in the validation's language), or `lang=`.

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

        See Also
        --------
        Ablation.run : Evaluate the full table, each variant, and each control.
        """
        from great_tables import GT, google_font, html, loc, style

        from pointblank._utils_html import _fmt_raw_html

        ctx = report_localizer(lang, locale, default=self.validation)
        resolved = self._resolve()
        tasks = self._tasks()

        def name(version: AblationVariant | str) -> str:
            return ctx.r(self._text(version))

        full = self._full_variant()
        by_id = {"full": full, **{v.id: v for v, _ in resolved.variants}}
        rows = [
            {
                "group": ctx.t("abl_group_baseline"),
                "variant": _variant_html(full, self._steps, name=name(full), ctx=ctx),
                "removed": _removed_html(0, 0.0, ctx),
            }
        ]
        variants = [v for v, _ in resolved.variants]
        for v in sorted(variants, key=self._group_order):
            same_as = resolved.same_as.get(v.id)
            rows.append(
                {
                    "group": self._group_label(v, ctx),
                    "variant": _variant_html(
                        v,
                        self._steps,
                        same_as=name(by_id[same_as]) if same_as else None,
                        detail=self._sweep_details.get(v.id) if v.sweep else None,
                        name=name(v),
                        ctx=ctx,
                    ),
                    "removed": (
                        _corrupted_html(v, self._n_rows, ctx)
                        if v.strategy == "inject"
                        else _removed_html(v.n_removed, v.f_removed, ctx)
                    ),
                }
            )
        rows += [
            {
                "group": ctx.t("abl_group_controls"),
                "variant": _variant_html(
                    v, self._steps, control=self.control, name=name(v), ctx=ctx
                ),
                "removed": _removed_html(v.n_removed, v.f_removed, ctx),
            }
            for v in resolved.controls
        ]

        lines = [
            ctx.t("abl_plan_rows", n=self._n_rows),
            ctx.t("abl_plan_repeats", n=self.repeats, evaluations=len(tasks)),
        ]
        if self.cache is not None:
            n_cached = sum(self._cache_get(self._task_key(v, r)) is not None for v, r in tasks)
            if n_cached:
                lines.append(ctx.t("abl_plan_cached", n=n_cached))
        subtitle = ctx.sentences(lines)
        if self.holdout_validation is not None:
            subtitle = f"<div>{subtitle}</div>{self._holdout_html(ctx)}"
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
                title=html(_title_html(ctx.t("abl_plan_title"), self.label)),
                subtitle=html(subtitle),
            )
            .cols_label(variant=ctx.t("abl_col_version"), removed=ctx.t("abl_col_removed"))
            .cols_align(align="right", columns="removed")
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="right" if ctx.rtl else "left")
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
            gt_tbl = gt_tbl.tab_source_note(source_note=html(_note_html(ctx.r(note))))
        return _rtl_styles(gt_tbl, ctx)

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

        Here's an ablation of a table of temperature readings, some of them recorded in tenths of
        a degree. The `evaluate()` function also returns the number of rows it was given, which is
        a second metric in the results. With `max_workers=4`, the run makes four evaluations at a
        time.

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

        After the run, the `Ablation` object shows its report, for the metric named in `metric=`.
        Removing the 12 readings in tenths helps. The other metric has a report of its own, from
        `get_tabular_report(metric="n_train")`, and both are in the results of
        [`get_effects()`](`pointblank.Ablation.get_effects`):

        ```{python}
        ablation.get_effects().select("variant", "metric", "mean", "delta_vs_control", "verdict")
        ```

        The full table has 300 rows, and the variant and its control have 288. There's no verdict
        for `n_train`, since verdicts are only for the metric in `metric=`.

        Making the evaluations at the same time doesn't change the results, since each evaluation
        gets the same table and seed either way. Here's the same ablation, run one evaluation after
        another:

        ```{python}
        serial = (
            pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
            .drop_each()
            .run()
        )

        serial.get_effects().equals(ablation.get_effects())
        ```

        The results are identical. Making evaluations at the same time pays off when each one
        takes a while (e.g., training a real model), and when they don't already use every core.

        See Also
        --------
        Ablation.plan : Show the versions of the table that the ablation will evaluate.
        Ablation.check_determinism : Check that `evaluate=` gives the same result twice.
        Ablation.get_effects : Get the results of the ablation as a table.
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

        See Also
        --------
        Ablation.run : Evaluate the full table, each variant, and each control.
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

        See Also
        --------
        Ablation.load_results : Read the results of evaluations made elsewhere.
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
        if format == "indices" and any(v.strategy == "inject" for v, _ in resolved.variants):
            raise ValueError(
                '`format="indices"` can only describe versions of the table with rows removed, '
                'not the corrupted tables of `inject()`. Use `format="parquet"` instead.'
            )

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
            "notes": [ENGLISH.r(note) for note in [*self._notes, *resolved.notes]],
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
            `results.csv` file (e.g., the directory written by `~~pointblank.Ablation.export()`), or
            a table (e.g., a Polars or pandas DataFrame). It needs a `variant` column (the id of the
            version of the table), a `seed` column, and a column for each metric. A `repeat` column
            is checked if it's there, and any other column is a metric.
        manifest
            The path of the `manifest.json` file written by `~~pointblank.Ablation.export()` (or of
            its directory). By default, a manifest in the same directory as the results file is used
            if there is one. Its fingerprints are checked against the ablation, which makes sure
            that the results are for the same data and the same versions of the table.

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

        See Also
        --------
        Ablation.export : Write out the versions of the table, for training them elsewhere.
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
        if not self._specs and self._shapley is None:
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
        # (a list of row numbers can be sent to another process whatever the table is)
        if self.pass_as == "table" and implementation not in (
            nw.Implementation.POLARS,
            nw.Implementation.PANDAS,
            nw.Implementation.PYARROW,
        ):
            raise ValueError(
                '`executor="process"` sends each training table to another process, which works '
                "for Polars, pandas, and PyArrow tables, but not for this table "
                f'({type(self.validation.data).__name__}). Use `executor="thread"` (or '
                '`pass_as="row_index"`) instead.'
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
        return self.evaluate(self._training_input(variant, removed), **kwargs)

    def _training_input(self, variant: AblationVariant, removed: frozenset[int]) -> Any:
        """What `evaluate=` is given: the training table, or the numbers of its kept rows."""
        if self.pass_as == "row_index":
            return [k for k in range(self._n_rows) if k not in removed]
        return self._training_table(removed, self._injection(variant))

    def _task_values(self, variant: AblationVariant, repeat: int) -> tuple[dict[str, float], bool]:
        """Evaluate a version of the table (or get the result from the cache)."""
        seed = self.seed + repeat
        removed = self._removed_rows(variant, repeat)
        key = None
        if self.cache is not None:
            key = self._task_key(variant, repeat)
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
                key = self._task_key(variant, repeat)
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
                    self._training_input(variant, removed),
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
        self._shapley_values = self._compute_shapley() if self._shapley is not None else None

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
        frame = self._training_frame(removed, self._injection(variant))
        if isinstance(frame, nw.LazyFrame):
            frame = frame.collect()
        if frame.implementation.is_pandas_like():
            frame = nw.from_native(frame.to_native().reset_index(drop=True), eager_only=True)
        frame.write_parquet(file)

    # ------------------------------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------------------------------

    def _bootstrap(self) -> list[list[int]] | None:
        """The resamples of the repeats for `ci="bootstrap"` (the same for every comparison)."""
        if self.ci != "bootstrap":
            return None
        return _bootstrap_resamples(self.repeats, self.seed)

    def _compute_effects(self) -> list[dict[str, Any]]:
        resolved = self._resolve()
        repeats = range(self.repeats)
        resamples = self._bootstrap()
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
                _add_comparison(row, "vs_full", values, full, self.conf_level, resamples)
                control_id = resolved.variant_control.get(variant.id)
                control = series(control_id, metric) if control_id is not None else None
                if control is not None:
                    row["control_mean"] = statistics.fmean(control)
                if variant.left_out is None:
                    if control is not None:
                        _add_comparison(
                            row, "vs_control", values, control, self.conf_level, resamples
                        )
                else:
                    # What the left-out step adds: removing every step's rows compared with
                    # removing the other steps' rows (plus as many random rows, with a control)
                    reference = series(resolved.references[variant.id], metric)
                    baseline = control if control is not None else values
                    _add_comparison(row, "added", reference, baseline, self.conf_level, resamples)
                variant_rows.append(row)

            normal = [
                row
                for row in variant_rows
                if row["left_out"] is None and row["strategy"] not in ("inject", "shapley")
            ]
            injected = [row for row in variant_rows if row["strategy"] == "inject"]
            left_out = [row for row in variant_rows if row["left_out"] is not None]
            # The groups of steps of `shapley()` are only there for the Shapley values, so they
            # don't change the adjusted p-values of the other variants
            coalitions = [row for row in variant_rows if row["strategy"] == "shapley"]
            judged = [row for row in variant_rows if row["strategy"] != "shapley"]
            for prefix, family in (
                ("vs_full", judged),
                ("vs_control", normal),
                ("added", left_out),
                ("vs_full", coalitions),
                ("vs_control", coalitions),
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
                # A defect is compared with the full table, as it removes no rows
                for row in injected:
                    row["verdict"] = self._verdict(row, "vs_full")
                for row in left_out:
                    verdict = self._verdict(row, "added")
                    # The step's rows help on their own, but add nothing to the other steps
                    if verdict == "inconclusive" and alone.get(row["left_out"]) == "helps":
                        verdict = "redundant"
                    row["verdict"] = verdict

            rows += variant_rows

        return rows

    def _verdict(self, row: dict[str, Any], prefix: str) -> str | None:
        return self._judge(row[f"p_adj_{prefix}"], row[f"delta_{prefix}"])

    def _judge(self, p_adj: float | None, delta: float | None) -> str | None:
        if self.repeats < 3:
            return None
        if p_adj is None or delta is None:
            return None  # pragma: no cover
        if p_adj < 1 - self.conf_level and delta != 0:
            better = delta > 0 if self.higher_is_better else delta < 0
            return "helps" if better else "hurts"
        return "inconclusive"

    def _compute_shapley(self) -> list[dict[str, Any]]:
        """The Shapley value of each step of `shapley()`, for each metric."""
        resolved = self._resolve()
        info = self._shapley
        players = info["steps"]
        k = len(players)
        repeats = range(self.repeats)
        weights = [
            math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
            for size in range(k)
        ]
        rows = []
        self._shapley_total = {}
        for metric in self._metrics:
            full = [self._values[("full", r)][metric] for r in repeats]
            per_repeat: dict[int, list[float]] = {i: [] for i in players}
            totals = []
            for r in repeats:
                # The value of each group of steps: its effect in this repeat
                value: dict[tuple[int, ...], float] = {(): 0.0}
                for coalition, variant_id in resolved.shapley.items():
                    control_id = resolved.variant_control.get(variant_id)
                    reference = self._values[(control_id, r)][metric] if control_id else full[r]
                    value[coalition] = self._values[(variant_id, r)][metric] - reference
                totals.append(value[players])
                for i in players:
                    if info["exact"]:
                        others = [j for j in players if j != i]
                        phi = 0.0
                        for size in range(k):
                            for before in itertools.combinations(others, size):
                                with_i = tuple(sorted((*before, i)))
                                phi += weights[size] * (value[with_i] - value[before])
                    else:
                        contributions = []
                        for order in info["orders"]:
                            at = order.index(i)
                            before = tuple(sorted(order[:at]))
                            with_i = tuple(sorted(order[: at + 1]))
                            contributions.append(value[with_i] - value[before])
                        phi = statistics.fmean(contributions)
                    per_repeat[i].append(phi)
            self._shapley_total[metric] = statistics.fmean(totals)

            family = []
            for i in players:
                mean, lower, upper, p = _paired(
                    per_repeat[i], [0.0] * self.repeats, self.conf_level, self._bootstrap()
                )
                family.append(
                    {
                        "step": i,
                        "n_failed": info["n_failed"][i],
                        "metric": metric,
                        "shapley": mean,
                        "lower": lower,
                        "upper": upper,
                        "p": p,
                        "p_adj": None,
                        "verdict": None,
                    }
                )
            p_values = [row["p"] for row in family]
            adjusted = _holm(p_values) if self.adjust == "holm" else p_values
            for row, p_adj in zip(family, adjusted):
                row["p_adj"] = p_adj
                if metric == self._primary:
                    row["verdict"] = self._judge(p_adj, row["shapley"])
            rows += family
        return rows

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
        the effect of its rows on top of the other steps, and a step with a Shapley value (from
        [`shapley()`](`pointblank.Ablation.shapley`)) gets that value. The notes appear in the
        validation's report, so that the report shows which checks matter to the model, next to
        the checks themselves. They're written in the language of that validation (its `lang=`),
        with the numbers formatted for its `locale=`.

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

        See Also
        --------
        Ablation.get_tabular_report : Get a tabular report of the ablation's results.
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

        ctx = report_localizer(None, None, default=target)
        effects = [row for row in self._effects if row["metric"] == self._primary]
        fmt = _NumberFormat([row["mean"] for row in effects], ctx)
        alone = {}
        on_top: dict[int, list[dict[str, Any]]] = {}
        for row in effects:
            if row["strategy"] in ("full", "drop_groups", "inject", "shapley"):
                continue
            if row["left_out"] is not None:
                on_top.setdefault(row["left_out"], []).append(row)
            elif len(row["steps"]) == 1:
                alone[row["steps"][0]] = row

        comparison = "vs_control" if self.control is not None else "vs_full"
        alone_key = (
            "abl_annotate_alone_random" if self.control is not None else "abl_annotate_alone_full"
        )
        direction = ctx.t("abl_higher" if self.higher_is_better else "abl_lower")
        shapley = {
            row["step"]: row for row in self._shapley_values or [] if row["metric"] == self._primary
        }
        for i in sorted(set(alone) | set(on_top) | set(shapley)):
            if i not in info:
                continue  # pragma: no cover
            parts = []
            if i in alone:
                row = alone[i]
                change = _change_text(fmt, row, comparison)
                parts.append((ctx.t(alone_key, n=row["n_removed"], change=change), row["verdict"]))
            for row in on_top.get(i, []):
                text = ctx.t(
                    "abl_annotate_on_top",
                    steps=tuple(row["steps"]),
                    change=_change_text(fmt, row, "added"),
                    n=len(row["steps"]),
                )
                parts.append((text, row["verdict"]))
            if i in shapley:
                row = shapley[i]
                value = fmt.delta(row["shapley"])
                if row["lower"] is not None and row["upper"] is not None:
                    value += " " + ctx.interval(fmt.delta(row["lower"]), fmt.delta(row["upper"]))
                parts.append((ctx.t("abl_annotate_shapley", value=value), row["verdict"]))

            def note(bold: bool) -> str:
                texts = []
                for text, verdict in parts:
                    if verdict is not None:
                        verdict = ctx.t(f"abl_verdict_{verdict}")
                        verdict = f"**{verdict}**" if bold else verdict
                        text = ctx.t("abl_annotate_verdict", text=text, verdict=verdict)
                    texts.append(text)
                return ctx.t(
                    "abl_annotate_note",
                    metric=self._primary,
                    direction=direction,
                    parts=ctx.clause_sep.join(texts),
                )

            info[i]._add_note(key="ablation", markdown=note(bold=True), text=note(bold=False))
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
            - `corruption`, `column`, `n_corrupted`, `checked_by`: for a variant of
            [`inject()`](`pointblank.Ablation.inject`), the defect (`"function"` for a function),
            the column it's put into (missing for `"duplicate"`), the number of rows corrupted (or
            copied), and the step that checks for it (`0` and missing for the other variants)
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
            when `control=None` (and for the variants of `inject()`, which remove no rows). The
            intervals and p-values need at least 2 repeats. The verdict of a
            `drop_all_but_each()` variant is based on the `delta_added` columns, that of an
            `inject()` variant on the `delta_vs_full` columns, and those of the other variants on
            the `delta_vs_control` columns (or the `delta_vs_full` columns, when `control=None`).
            The groups of steps that [`shapley()`](`pointblank.Ablation.shapley`) adds have the
            strategy `"shapley"` and no verdict, and their p-values are adjusted on their own;
            the Shapley values themselves are in
            [`get_shapley()`](`pointblank.Ablation.get_shapley`).

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

        See Also
        --------
        Ablation.get_tabular_report : Get a tabular report of the ablation's results.
        Ablation.get_shapley : Get the Shapley values of the steps as a table.
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

    def get_shapley(self) -> Any:
        """
        Get the Shapley values of the steps as a table.

        The Shapley values (from [`shapley()`](`pointblank.Ablation.shapley`)) split the effect
        of removing every step's rows among the steps, giving each step its average contribution
        over the orders in which the steps could be added. This method gives them with their
        intervals and verdicts, for every metric.

        Returns
        -------
        DataFrame
            A Polars DataFrame (or a pandas DataFrame, if Polars isn't installed) with a row for
            each step and metric, and these columns:

            - `step`: the number of the step
            - `n_failed`: the number of rows that failed the step
            - `metric`: the metric
            - `shapley`: the Shapley value of the step, the mean across the repeats
            - `lower`, `upper`: its interval at `conf_level=` (with at least 2 repeats)
            - `p`, `p_adj`: the p-value of the Shapley value (against no effect), and the p-value
            adjusted across the steps (as set by `adjust=`)
            - `verdict`: `"helps"`, `"hurts"`, or `"inconclusive"` for the metric named in
            `metric=` (missing for other metrics, and with fewer than 3 repeats)

            For each metric, the Shapley values add up to the effect of removing the rows of every
            step (compared with the control, or with the full table when `control=None`).

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
        ```

        Here's an ablation of a table of temperature readings with overlapping steps: step 1
        catches the sentinel value `-999`, and step 2 catches the sentinels along with the readings
        in tenths of a degree. There are 10 repeats, since the sentinels make the error vary a lot
        from one repeat to the next.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        for k in range(0, 300, 25):
            temps[k] *= 10
        for k in range(7, 300, 40):
            temps[k] = -999
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed):
            sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_gt(columns="temp", value=-50)
            .col_vals_between(columns="temp", left=-50, right=50)
            .interrogate()
        )

        ablation = (
            pb.Ablation(
                validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=10
            )
            .shapley()
            .run()
        )

        ablation.get_shapley().select("step", "shapley", "lower", "upper", "verdict")
        ```

        Both steps lower the error, and step 2 gets the larger share, since it also catches the
        readings that step 1 misses.

        See Also
        --------
        Ablation.shapley : Split the effect of removing the steps' rows fairly among the steps.
        Ablation.get_effects : Get the results of the ablation as a table.
        """
        self._check_run()
        if self._shapley_values is None:
            raise ValueError("There are no Shapley values: call `shapley()` before the run.")
        columns = {
            "step": "int",
            "n_failed": "int",
            "metric": "str",
            "shapley": "float",
            "lower": "float",
            "upper": "float",
            "p": "float",
            "p_adj": "float",
            "verdict": "str",
        }
        rows = [[row[c] for c in columns] for row in self._shapley_values]
        lib = _select_df_lib(preference="polars")
        if lib.__name__ == "polars":
            dtypes = {"str": lib.String, "int": lib.Int64, "float": lib.Float64}
            schema = {name: dtypes[kind] for name, kind in columns.items()}
            return lib.DataFrame(rows, schema=schema, orient="row")
        df = lib.DataFrame(rows, columns=list(columns))
        for name, kind in columns.items():
            if kind == "float":
                df[name] = df[name].astype("float64")
            elif kind == "int":
                df[name] = df[name].astype("int64")
        return df

    def suggest_thresholds(self, tolerance: float) -> dict[int, Thresholds]:
        """
        Suggest thresholds for the steps, from what their failing test units cost the model.

        Thresholds are usually picked by feel: is 1% of missing values a warning, or an error?
        With the variants of [`inject()`](`pointblank.Ablation.inject`), the ablation has measured
        how much the model gets worse as a defect gets more common. This method turns that into
        thresholds: for each step that checks for an injected defect (given as `step=` to
        `inject()`), the error threshold is the share of failing test units at which the model
        would get worse by `tolerance=`, and the warning threshold is half of that.

        Parameters
        ----------
        tolerance
            How much worse the model can get before it's an error, in the units of the metric that
            the verdicts are based on (e.g., `0.01` for an accuracy that can drop by a percentage
            point).

        Returns
        -------
        dict[int, Thresholds]
            The suggested thresholds, by step number, as [`Thresholds`](`pointblank.Thresholds`)
            objects with the `warning=` and `error=` levels set as shares of the test units. They
            can be given to the steps (e.g., as the `thresholds=` of the validation method).

        Details
        -------
        For each rate of a defect, the step is checked again on the corrupted table, which gives
        the share of its test units that the defect makes fail (beyond those that failed
        already). The cost of a failing test unit is then the slope of the harm to the model
        (how much worse the metric is than on the full table) against that share, fitted through
        zero over the rates. The error threshold is `tolerance=` divided by that slope, rounded to
        two significant digits.

        A step gets no thresholds (with a warning saying why) when the defect makes none of its
        test units fail, when the defect doesn't make the model worse (or when none of the
        verdicts are `"hurts"`, with at least 3 repeats), or when even failing every test unit
        would cost less than `tolerance=`. When a step has several injected defects, its
        thresholds come from the costliest one. The thresholds are only as good as the defects
        are realistic: a defect that's injected into random clean rows can cost less (or more)
        than the same defect in the rows where it really happens.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(report_incl_header=False, report_incl_footer_timings=False)
        ```

        Here's a table of temperature readings with no missing values, and a 'model' that
        estimates the average temperature, filling in a missing reading with `0`. Missing
        readings are injected at three rates.

        ```{python}
        import pointblank as pb
        import polars as pl

        temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
        readings = pl.DataFrame({"temp": temps})

        def evaluate(train, *, seed):
            sample = train.fill_null(0).sample(fraction=1.0, with_replacement=True, seed=seed)
            return abs(sample["temp"].mean() - 20)

        validation = (
            pb.Validate(data=readings)
            .col_vals_not_null(columns="temp")
            .interrogate()
        )

        ablation = (
            pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
            .inject("null", step=1)
            .run()
        )

        ablation.suggest_thresholds(tolerance=0.5)
        ```

        Each percent of missing readings adds about 0.2 degrees to the error, so the error
        threshold is the share of missing readings that adds 0.5 degrees: about 2.5%. The warning
        threshold is half of that. The verdict for the rate of 1% is inconclusive (its effect is
        small next to the noise), but the slope is fitted over all three rates, and the other two
        clearly hurt. A suggestion needs at least one rate that hurts, so for a defect with small
        effects, use rates high enough to measure, or more `repeats=`.

        See Also
        --------
        Ablation.inject : Add a variant for each rate of a defect, put into clean rows of the table.
        """
        from pointblank.thresholds import Thresholds

        self._check_run()
        if not _is_number(tolerance) or not math.isfinite(float(tolerance)):
            raise TypeError(f"`tolerance=` must be a number, got {tolerance!r}.")
        if tolerance <= 0:
            raise ValueError(f"`tolerance=` must be greater than 0, got {tolerance!r}.")
        effects = {
            row["variant"]: row
            for row in self._effects
            if row["metric"] == self._primary and row["strategy"] == "inject"
        }
        by_sweep: dict[str, list[dict[str, Any]]] = {}
        for variant_id, row in effects.items():
            if self._injections[variant_id].step is not None:
                by_sweep.setdefault(row["sweep"], []).append(row)
        if not by_sweep:
            raise ValueError(
                "`suggest_thresholds()` needs `inject()` variants with the step that checks for "
                "the defect, given as `step=`."
            )

        suggested: dict[int, float] = {}
        skipped: dict[int, list[str]] = {}
        for sweep_id, points in by_sweep.items():
            step = self._injections[points[0]["variant"]].step
            label = ENGLISH.r(self._sweeps[sweep_id]["label"])
            shares = [self._caught_share(row["variant"]) for row in points]
            harms = [
                -row["delta_vs_full"] if self.higher_is_better else row["delta_vs_full"]
                for row in points
            ]
            used = [(x, d) for x, d in zip(shares, harms) if x > 0]
            reason = None
            cost = None
            if not used:
                reason = f"the {label} make none of its test units fail"
            else:
                cost = sum(x * d for x, d in used) / sum(x * x for x, _ in used)
                verdicts = [row["verdict"] for row in points if row["verdict"] is not None]
                if cost <= 0:
                    reason = f"the {label} don't make the model worse"
                elif verdicts and "hurts" not in verdicts:
                    reason = f"the {label} don't clearly make the model worse"
                elif tolerance / cost >= 1:
                    reason = (
                        f"even with every test unit failing, the {label} would cost less than "
                        f"`tolerance={tolerance:g}`"
                    )
            if reason is not None:
                skipped.setdefault(step, []).append(reason)
                continue
            error = float(f"{tolerance / cost:.2g}")
            suggested[step] = min(error, suggested.get(step, error))

        if any(step not in suggested for step in skipped):
            lines = [
                f"step {step}: {'; '.join(reasons)}"
                for step, reasons in sorted(skipped.items())
                if step not in suggested
            ]
            warnings.warn(
                "Some steps get no suggested thresholds (" + "; ".join(lines) + ").",
                UserWarning,
                stacklevel=2,
            )
        return {
            step: Thresholds(warning=float(f"{error / 2:.2g}"), error=error)
            for step, error in sorted(suggested.items())
        }

    def _caught_share(self, variant_id: str) -> float:
        """
        The share of a step's test units that an `inject()` variant's defect makes fail (beyond
        those that failed already).
        """
        if variant_id in self._caught:
            return self._caught[variant_id]
        from pointblank.validate import Validate, _validation_info_to_step

        injection = self._injections[variant_id]
        info = self._steps[injection.step]
        found = _validation_info_to_step(info, None, None, [], keep_objects=True)
        if found is None:  # pragma: no cover
            raise ValueError(f"Step {injection.step} can't be checked again.")
        method, kwargs = found
        for key in _RERUN_DROPPED_PARAMS:
            kwargs.pop(key, None)
        if info.segments is not None:
            kwargs["segments"] = info.segments

        table = self._training_table(frozenset(), injection)
        validation = getattr(Validate(data=table), method)(**kwargs)
        validation.interrogate(collect_extracts=False)
        n_failed = sum(step.n_failed or 0 for step in validation.validation_info)
        n = sum(step.n or 0 for step in validation.validation_info)
        before = (info.n_failed or 0) / info.n if info.n else 0.0
        share = max(0.0, n_failed / n - before) if n else 0.0
        self._caught[variant_id] = share
        return share

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
            [`get_effects()`](`pointblank.Ablation.get_effects`)), the Shapley values
            (`"shapley"`, with the same fields as the columns of
            [`get_shapley()`](`pointblank.Ablation.get_shapley`), or `None` without
            [`shapley()`](`pointblank.Ablation.shapley`)), the notes about variants and steps
            that weren't used (`"notes"`), and the results of the steps on the holdout set
            (`"holdout"`, with its number of rows, its health score, and the number and share of
            failing test units of each step, or `None` without `holdout=`).

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

        See Also
        --------
        Ablation.to_json : Get the settings and results of the ablation as a JSON string.
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
            "shapley": (
                [dict(row) for row in self._shapley_values]
                if self._shapley_values is not None
                else None
            ),
            "notes": [ENGLISH.r(note) for note in [*self._notes, *resolved.notes]],
            "holdout": self._holdout_summary(),
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

        See Also
        --------
        Ablation.to_dict : Get the settings and results of the ablation as a JSON-serializable dict.
        """
        return json.dumps(self.to_dict(), indent=indent)

    def get_tabular_report(
        self,
        metric: str | None = None,
        title: str | None = None,
        lang: str | None = None,
        locale: str | None = None,
    ) -> GT:
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
        removing as many random rows. The defects of [`inject()`](`pointblank.Ablation.inject`)
        are shown in the same way, compared with the full table. The Shapley values of
        [`shapley()`](`pointblank.Ablation.shapley`) are in a group of their own, with each
        step's share of the total effect. The header gives the metric for the full table and the
        settings of the run, and notes at the bottom list the variants and steps that weren't
        used.

        Parameters
        ----------
        metric
            The metric to show. By default, it's the one that verdicts are based on (named in
            `metric=` when the ablation was created). Verdicts are only shown for that metric.
        title
            An optional title replacing the default.
        lang
            The language of the report's text, as a two-letter language code (e.g., `"fr"` or
            `"de"`; see the `lang=` argument of [`Validate`](`pointblank.Validate`) for the
            languages). By default, the report is in the language of the validation. The names
            of the variants given in [`drop_steps()`](`pointblank.Ablation.drop_steps`), the
            metrics, and the briefs of the steps are shown as they are.
        locale
            The locale for the numbers in the report (e.g., `"de-CH"`), which sets the decimal
            mark and the thousands separator. By default, it's the locale of the validation (when
            the report is in the validation's language), or `lang=`.

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

        Here's an ablation of a table of temperature readings with missing values and with
        readings that were recorded in tenths of a degree (so they're ten times too high). The
        'model' estimates the average temperature, ignoring missing readings. The `evaluate()`
        function returns two metrics: the error against the true average of `20`, which the
        verdicts are based on, and the estimate itself.

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
            estimate = sample["temp"].mean()
            return {"error": abs(estimate - 20), "estimate": estimate}

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

        Removing the readings in tenths (step 2) helps, while removing the missing readings
        (step 1) makes no clear difference. The error doesn't say in which direction the model
        was wrong, though. The report for the other metric does:

        ```{python}
        ablation.get_tabular_report(metric="estimate")
        ```

        The readings in tenths pull the estimate up, to about 27 degrees, and removing them brings
        it back to about 20. There are no verdicts in this report, since they're based on the
        error. The report can also be in another language, with `lang=` (e.g., `lang="fr"`), and
        use the number formats of another locale, with `locale=`.

        See Also
        --------
        Ablation.get_effects : Get the results of the ablation as a table.
        Ablation.annotate : Add the results of the ablation to the validation, as notes.
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

        # The groups of steps of `shapley()` are shown through the Shapley values
        effects = [
            row for row in self._effects if row["metric"] == metric and row["strategy"] != "shapley"
        ]
        baseline, variant_rows = effects[0], effects[1:]
        has_left_out = any(row["left_out"] is not None for row in variant_rows)
        has_injected = any(row["strategy"] == "inject" for row in variant_rows)
        shapley_rows = [row for row in self._shapley_values or [] if row["metric"] == metric]
        # (injected defects are always compared with the full table)
        compared = shapley_rows or any(row["strategy"] != "inject" for row in variant_rows)
        comparison = "vs_control" if self.control is not None and compared else "vs_full"
        has_groups = (
            has_left_out
            or bool(shapley_rows)
            or any(row["sweep"] is not None for row in variant_rows)
        )
        resolved = self._resolve()
        ctx = report_localizer(lang, locale, default=self.validation)
        variants = {v.id: v for v in self._variants}

        def name(version_id: str) -> str:
            return ctx.r(self._text(variants.get(version_id, version_id)))

        def prefix(row: dict[str, Any]) -> str:
            if row["left_out"] is not None:
                return "added"
            return "vs_full" if row["strategy"] == "inject" else comparison

        fmt = _NumberFormat([row["mean"] for row in effects], ctx)
        bounds = [
            abs(row[key])
            for row in variant_rows
            for key in (
                f"delta_{prefix(row)}",
                f"delta_{prefix(row)}_lower",
                f"delta_{prefix(row)}_upper",
            )
            if row[key] is not None
        ] + [
            abs(row[key])
            for row in shapley_rows
            for key in ("shapley", "lower", "upper")
            if row[key] is not None
        ]
        scale = max(bounds, default=0.0) or 1.0

        rows = []
        for row in sorted(
            variant_rows, key=lambda row: self._group_order(variants[row["variant"]])
        ):
            variant = variants[row["variant"]]
            same_as = resolved.same_as.get(variant.id)
            metric_html = _mono(fmt.value(row["mean"], row["sd"]))
            if prefix(row) != "vs_full":
                delta = fmt.delta(row["delta_vs_full"])
                metric_html += _small(_esc(ctx.t("abl_delta_vs_full", delta=delta)))
            if variant.strategy == "inject":
                removed_html = _corrupted_html(variant, self._n_rows, ctx)
            else:
                removed_html = _removed_html(row["n_removed"], row["f_removed"], ctx)
            rows.append(
                {
                    "group": self._group_label(variant, ctx),
                    "variant": _variant_html(
                        variant,
                        self._steps,
                        same_as=name(same_as) if same_as else None,
                        detail=self._sweep_details.get(variant.id) if variant.sweep else None,
                        name=name(variant.id),
                        ctx=ctx,
                    ),
                    "removed": removed_html,
                    "metric": metric_html,
                    "effect": _effect_html(row, prefix(row), fmt, scale),
                }
            )
        total = self._shapley_total.get(metric) if shapley_rows else None
        for row in shapley_rows:
            i = row["step"]
            step = AblationVariant(
                id=f"shapley_{i}", strategy="shapley", name=f"Step {i}", steps=(i,)
            )
            share = row["shapley"] / total if total and abs(total) > _ROUNDING else None
            rows.append(
                {
                    "group": ctx.t("abl_group_shapley"),
                    "variant": _variant_html(
                        step, self._steps, name=ctx.t("abl_name_step", i=i), ctx=ctx
                    ),
                    "removed": _removed_html(row["n_failed"], row["n_failed"] / self._n_rows, ctx),
                    "metric": (
                        _mono(ctx.pct(share, 0)) + _small(_esc(ctx.t("abl_of_total")))
                        if share is not None
                        else _mono("—")
                    ),
                    "effect": _effect_html(
                        {
                            "delta_shapley": row["shapley"],
                            "delta_shapley_lower": row["lower"],
                            "delta_shapley_upper": row["upper"],
                            "verdict": row["verdict"],
                        },
                        "shapley",
                        fmt,
                        scale,
                    ),
                }
            )

        # `higher_is_better=` is only known to apply to the metric the verdicts are based on
        direction = ""
        full_metric = {
            "metric": _esc(metric),
            "rows": ctx.t("n_rows", n=self._n_rows),
            "value": fmt.value(baseline["mean"], baseline["sd"]),
        }
        if metric == self._primary:
            direction = ctx.t("abl_higher" if self.higher_is_better else "abl_lower")
            lines = [ctx.t("abl_full_metric_direction", direction=direction, **full_metric)]
            direction = f" ({direction})"
        else:
            lines = [ctx.t("abl_full_metric", **full_metric)]
        if comparison != "vs_control":
            lines.append(ctx.t("abl_evaluated_full", n=self.repeats))
        elif isinstance(self.control, Stratified):
            columns = _esc(ctx.list_sep.join(self.control.by))
            lines.append(ctx.t("abl_evaluated_stratified", n=self.repeats, columns=columns))
        else:
            lines.append(ctx.t("abl_evaluated_random", n=self.repeats))
        if has_left_out:
            control = self.control is not None
            lines.append(ctx.t("abl_left_out_control" if control else "abl_left_out"))
        if has_injected and comparison != "vs_full":
            lines.append(ctx.t("abl_injected_vs_full"))
        if shapley_rows:
            lines.append(ctx.t("abl_shapley_share"))
        if self.evaluate_version is not None:
            lines.append(ctx.t("abl_evaluate_version", version=_esc(self.evaluate_version)))
        subtitle = "".join(f"<div>{line}</div>" for line in lines) + self._holdout_html(ctx)
        if metric != self._primary:
            text = ctx.t("abl_verdicts_based_on", metric=_esc(self._primary))
            subtitle += f"<div>{text}</div>"

        effect_label = ctx.t(
            "abl_col_delta_random" if comparison == "vs_control" else "abl_col_delta_full"
        )
        default_title = _title_html(ctx.t("abl_report_title"), self.label)
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
                variant=ctx.t("abl_col_variant"),
                removed=ctx.t("abl_col_removed"),
                metric=_esc(metric),
                effect=effect_label,
            )
            .cols_align(align="right", columns=["removed", "metric"])
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="right" if ctx.rtl else "left")
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

        kind = "bootstrap" if self.ci == "bootstrap" else "t"
        holm = "_holm" if self.adjust == "holm" else ""
        interval_note = ctx.sentences(
            [
                ctx.t(f"abl_intervals_{kind}{holm}", conf=ctx.pct(self.conf_level, 0)),
                ctx.t("abl_no_verdicts") if self.repeats < 3 else "",
            ]
        )
        source_notes = [interval_note]
        if self._loaded:
            source_notes.append(ctx.t("abl_loaded"))
        n_cached = sum(run["cached"] for run in self._runs)
        if n_cached:
            source_notes.append(ctx.t("abl_cached", n=n_cached, total=len(self._runs)))
        if shapley_rows:
            source_notes.append(self._shapley_note(fmt.delta(total), ctx))
        for note in [*source_notes, *self._notes, *resolved.notes]:
            gt_tbl = gt_tbl.tab_source_note(source_note=html(_note_html(ctx.r(note))))
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
                            name=lambda row: name(row["variant"]),
                        )
                    )
                )
        return _rtl_styles(gt_tbl, ctx)

    def _group_order(self, variant: AblationVariant) -> tuple[int, float]:
        """Where a variant is listed in the plan and the report: by its group, then its value."""
        if variant.sweep is not None:
            return (2 + list(self._sweeps).index(variant.sweep), variant.value)
        if variant.strategy == "shapley":
            return (2 + len(self._sweeps), float(len(variant.steps)))
        return (int(variant.left_out is not None), 0.0)

    def _group_label(self, variant: AblationVariant, ctx: Localizer = ENGLISH) -> str:
        if variant.strategy == "inject":
            return ctx.t("abl_group_injected", label=self._sweeps[variant.sweep]["label"])
        if variant.sweep is not None:
            return ctx.t("abl_group_sweep", label=self._sweeps[variant.sweep]["label"])
        if variant.strategy == "shapley":
            return ctx.t("abl_group_shapley_groups")
        return ctx.t(
            "abl_group_variants" if variant.left_out is None else "abl_group_leave_one_out"
        )

    def _shapley_note(self, total: str, ctx: Localizer = ENGLISH) -> str:
        info = self._shapley
        k = len(info["steps"])
        if info["exact"]:
            how = ctx.t("abl_shapley_exact", k=k)
        else:
            how = ctx.t(
                "abl_shapley_estimated", orders=info["permutations"], k=k, seed=str(info["seed"])
            )
        total_key = "abl_shapley_sum_random" if self.control is not None else "abl_shapley_sum_full"
        return ctx.sentences([how, ctx.t(total_key, total=total)])


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
    values: list[float],
    reference: list[float],
    conf_level: float,
    resamples: list[list[int]] | None = None,
) -> tuple[float, float | None, float | None, float | None]:
    """
    Compare paired values: the mean difference, its interval, and its p-value.

    The interval and the p-value are those of a paired t-test or, with `resamples=` (from
    `_bootstrap_resamples()`), the percentile interval of the bootstrap and the share of the
    resampled means on the other side of zero. With one pair, there's no interval or p-value. When
    the differences are all the same, the interval is that difference, and the p-value is 0 (or 1,
    if the difference is 0).
    """
    diffs = [v - r for v, r in zip(values, reference)]
    mean = statistics.fmean(diffs)
    n = len(diffs)
    if n < 2:
        return mean, None, None, None
    se = statistics.stdev(diffs) / math.sqrt(n)
    if se == 0:
        return mean, mean, mean, (0.0 if mean != 0 else 1.0)
    if resamples is not None:
        means = sorted(sum(diffs[k] for k in sample) / n for sample in resamples)
        tail = (1 - conf_level) / 2
        n_below = sum(1 for m in means if m <= 0)
        n_above = sum(1 for m in means if m >= 0)
        p = min(1.0, 2 * (min(n_below, n_above) + 1) / (len(means) + 1))
        return mean, _quantile(means, tail), _quantile(means, 1 - tail), p
    half_width = _t_quantile(1 - (1 - conf_level) / 2, n - 1) * se
    return mean, mean - half_width, mean + half_width, _t_p_value(mean / se, n - 1)


@functools.lru_cache(maxsize=4)
def _bootstrap_resamples(n: int, seed: int) -> list[list[int]]:
    """Resamples (with replacement) of `n` repeats, made the same way for the same seed."""
    rng = random.Random(f"pointblank-ablation-bootstrap:{seed}:{n}")
    return [[rng.randrange(n) for _ in range(n)] for _ in range(_BOOTSTRAP_RESAMPLES)]


def _quantile(ordered: list[float], q: float) -> float:
    """The `q` quantile of sorted values, interpolated between them (R's type 7)."""
    h = (len(ordered) - 1) * q
    below = math.floor(h)
    above = min(below + 1, len(ordered) - 1)
    return ordered[below] + (h - below) * (ordered[above] - ordered[below])


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
        corruption=variant.corruption,
        column=variant.column,
        n_corrupted=variant.n_corrupted,
        checked_by=variant.checked_by,
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
    resamples: list[list[int]] | None = None,
) -> None:
    delta, lower, upper, p = _paired(values, reference, conf_level, resamples)
    row[f"delta_{prefix}"] = delta
    row[f"delta_{prefix}_lower"] = lower
    row[f"delta_{prefix}_upper"] = upper
    row[f"p_{prefix}"] = p


# ----------------------------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------------------------


class _NumberFormat:
    """Format a metric's values with a number of decimals that suits their size."""

    def __init__(self, values: list[float], ctx: Localizer = ENGLISH) -> None:
        size = max((abs(v) for v in values if v), default=1.0)
        self.decimals = max(0, min(6, 3 - math.floor(math.log10(size))))
        self.ctx = ctx

    def number(self, x: float) -> str:
        return self.ctx.marks(f"{x:,.{self.decimals}f}").replace("-", "−")

    def delta(self, x: float | None) -> str:
        if x is None:
            return "—"  # pragma: no cover
        return self.ctx.marks(f"{x:+,.{self.decimals}f}").replace("-", "−")

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


def _rtl_styles(gt_tbl: GT, ctx: Localizer) -> GT:
    """Set a report's text right to left, if its language is written that way."""
    if not ctx.rtl:
        return gt_tbl
    from great_tables import loc, style

    return gt_tbl.tab_style(
        style=style.css("direction: rtl; text-align: right;"),
        locations=[
            loc.header(),
            loc.column_labels(),
            loc.body(),
            loc.row_groups(),
            loc.source_notes(),
        ],
    )


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
    detail: Text | str | None = None,
    name: str | None = None,
    ctx: Localizer = ENGLISH,
) -> str:
    """
    The variant's name (`name=`, written in the report's language), with a line for each step (its
    brief) below (or the `detail=` line).
    """
    name = f"<div><strong>{_esc(variant.name if name is None else name)}</strong></div>"
    if variant.strategy == "full":
        return name + _small(_esc(ctx.t("abl_nothing_removed")))
    if variant.is_control:
        within = isinstance(control, Stratified)
        columns = _esc(ctx.list_sep.join(control.by)) if within else ""
        if variant.left_out is not None:
            key = "abl_control_left_out_within" if within else "abl_control_left_out"
            text = ctx.t(key, steps=tuple(variant.steps), columns=columns, n=len(variant.steps))
        else:
            key = "abl_control_random_within" if within else "abl_control_random"
            text = ctx.t(key, columns=columns)
        return name + _small(text)
    if variant.group_column is not None:
        return name + _small(_esc(ctx.t("abl_group_rows")))

    def brief(i: int) -> str:
        step = steps[i]
        return _code_html(step.brief or step.autobrief or step.assertion_type)

    lines = []
    if detail is not None:
        lines.append(_code_html(ctx.r(detail)))
    elif variant.left_out is not None:
        lines.append(ctx.t("abl_leaves_out", i=variant.left_out, brief=brief(variant.left_out)))
    else:
        for i in variant.steps[:max_briefs]:
            lines.append(brief(i) if len(variant.steps) == 1 else f"{i}: {brief(i)}")
        if len(variant.steps) > max_briefs:
            lines.append(_esc(ctx.t("abl_more_steps", n=len(variant.steps) - max_briefs)))
    if same_as is not None:
        lines.append(_esc(ctx.t("abl_same_rows_as", name=same_as)))
    return name + "".join(_small(line) for line in lines)


def _pct_text(share: float, ctx: Localizer = ENGLISH) -> str:
    pct = share * 100
    if pct == 0:
        return "0%"
    if pct < 0.1:
        return "<" + ctx.marks("0.1%")
    return ctx.marks(f"{pct:.1f}%")


def _removed_html(n_removed: int, f_removed: float, ctx: Localizer = ENGLISH) -> str:
    return f"<div>{_mono(ctx.int(n_removed))}</div>" + _small(_esc(_pct_text(f_removed, ctx)))


def _corrupted_html(variant: AblationVariant, n_rows: int, ctx: Localizer = ENGLISH) -> str:
    """The number and share of the rows that an `inject()` variant corrupts (or copies)."""
    key = "abl_pct_copied" if variant.corruption == "duplicate" else "abl_pct_corrupted"
    share = _pct_text(variant.n_corrupted / n_rows, ctx)
    return f"<div>{_mono(ctx.int(variant.n_corrupted))}</div>" + _small(_esc(ctx.t(key, pct=share)))


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
    ctx = fmt.ctx
    delta = row[f"delta_{prefix}"]
    lower = row[f"delta_{prefix}_lower"]
    upper = row[f"delta_{prefix}_upper"]
    verdict = row["verdict"]
    color = _VERDICT_COLORS.get(verdict, "#666666")

    text = fmt.delta(delta)
    if lower is not None and upper is not None:
        interval = ctx.interval(fmt.delta(lower), fmt.delta(upper))
        text += f" <span style='color: #777777;'>{interval}</span>"
    plot = _interval_svg(delta, lower, upper, scale, color)
    badge = ""
    if verdict is not None:
        badge = f" {_badge(_esc(ctx.t(f'abl_verdict_{verdict}')), color)}"
    return f"<div>{_mono(text)}</div><div style='white-space: nowrap;'>{plot}{badge}</div>"


def _sweep_svg(
    sweep: dict[str, Any],
    points: list[dict[str, Any]],
    baseline: float,
    fmt: _NumberFormat,
    metric: str,
    direction: str,
    name: Callable[[dict[str, Any]], str] | None = None,
) -> str:
    """
    A plot of a sweep: the difference from the full table at each of its values (a dot with its
    interval, colored by the verdict), with a dashed line for removing as many random rows. The
    texts are in the language of `fmt`, with `name=` giving the name of a point's variant.
    """
    ctx = fmt.ctx
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
    title = ctx.t("abl_sweep_title", label=ctx.r(sweep["label"]), metric=metric) + direction
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
            f'style="{text_style}">{ctx.marks(f"{y:+g}").replace("-", "−")}</text>'
        )
    for x in xs:
        label = ctx.marks(f"{x:g}%" if sweep["percent"] else f"{x:g}")
        parts.append(
            f"<text x='{px(x):.1f}' y='{height - bottom + 13}' text-anchor='middle' "
            f'style="{text_style}">{_esc(label)}</text>'
        )
    parts.append(
        f"<text x='{(left + width - right) / 2:.1f}' y='{height - 6}' text-anchor='middle' "
        f'style="{text_style}">{_esc(ctx.r(sweep["axis"]))}</text>'
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
            f"fill='{color}'><title>{_esc(row['name'] if name is None else name(row))}: "
            f"{_esc(fmt.delta(row['delta_vs_full']))}</title></circle>"
        )
    if control_points:
        parts.append(
            f"<line x1='0' y1='26' x2='18' y2='26' stroke='#888888' stroke-width='1.5' "
            "stroke-dasharray='4,3'/>"
            f"<text x='22' y='29' style=\"{text_style}\">{_esc(ctx.t('abl_sweep_legend'))}</text>"
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
        text += " " + fmt.ctx.interval(fmt.delta(lower), fmt.delta(upper))
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


def _is_missing(value: Any) -> bool:
    """Is a value missing (`None`, or a NaN float or timestamp)?"""
    if value is None or type(value).__name__ == "NAType":
        return True
    try:
        return bool(value != value)
    except Exception:
        return False


def _function_name(fn: Callable[..., Any]) -> str:
    name = getattr(fn, "__name__", None)
    if not isinstance(name, str) or not name or name == "<lambda>":
        return "custom"
    return name


def _corrupted_values(kind: str, fn: Any, values: list[Any]) -> list[Any]:
    """The new values of the rows that `inject()` corrupts (none for `"null"` and `"duplicate"`)."""
    if kind == "shuffle":
        # A ring: each row gets the value of the next one
        return values[1:] + values[:1]
    if kind == "truncate":
        return [v[: len(v) // 2] for v in values]
    if kind == "function":
        new = []
        for v in values:
            try:
                new.append(fn(v))
            except Exception as e:
                raise ValueError(
                    f"The `corruption=` function raised an error for the value {v!r}: {e}"
                ) from e
        return new
    return []


def _corrupted_series(
    name: str, values: list[Any], dtype: Any, implementation: Any
) -> nw.Series[Any]:
    series = nw.new_series(name, values, backend=implementation)
    if series.dtype != dtype:
        series = series.cast(dtype)
    return series


def _check_corrupted_values(
    values: list[Any], column: str, dtype: Any, implementation: Any, kind: str
) -> None:
    try:
        _corrupted_series(column, values, dtype, implementation)
    except Exception as e:
        if kind != "function":  # pragma: no cover
            raise
        raise TypeError(
            f"The values from the `corruption=` function can't be put in the column {column!r} "
            f"(of type {dtype}): {e}"
        ) from e


def _apply_injection(frame: Any, injection: _Injection) -> Any:
    """Put the defect of an `inject()` variant into a table (with the row numbers in `_INDEX`)."""
    rows = list(injection.rows)
    if injection.kind == "duplicate":
        return nw.concat([frame, frame.filter(nw.col(_INDEX).is_in(rows))])
    column = injection.column
    if injection.kind == "null":
        return frame.with_columns(
            nw.when(~nw.col(_INDEX).is_in(rows)).then(nw.col(column)).alias(column)
        )
    lazy = isinstance(frame, nw.LazyFrame)
    if lazy:
        frame = frame.collect()
    dtype = frame.collect_schema()[column]
    new = dict(zip(rows, injection.values))
    values = [
        new[int(k)] if int(k) in new else v
        for k, v in zip(frame[_INDEX].to_list(), frame[column].to_list())
    ]
    frame = frame.with_columns(_corrupted_series(column, values, dtype, frame.implementation))
    return frame.lazy() if lazy else frame


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

    Here's a table of the temperatures of a process at different pressures, where one temperature
    in ten was recorded in degrees Fahrenheit rather than Celsius. The held-out data is clean. The
    model predicts the temperature from the pressure, and it's scored in two ways: with R² (which
    the verdicts are based on) and with the mean absolute error.

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

    evaluate = pb.ablation.sklearn_evaluator(
        holdout,
        target="temp",
        features=["pressure"],
        scoring=["r2", "neg_mean_absolute_error"],
    )

    ablation = pb.Ablation(validation, evaluate=evaluate, metric="r2").drop_each().run()

    ablation
    ```

    The default model, a gradient boosting model, is badly thrown off by the readings in degrees
    Fahrenheit: its R² on the held-out data is below zero, so it does worse than always predicting
    the average temperature. Removing those readings brings its R² close to `1`. The report for
    the other metric shows the same change in degrees:

    ```{python}
    ablation.get_tabular_report(metric="neg_mean_absolute_error")
    ```

    The mean absolute error drops from about 6 degrees to about 0.3 (the metric is negated, so
    that higher is better, as for every scikit-learn scorer). Whether a problem in the data
    matters depends on the model, though. Here's the same ablation with a robust linear model,
    which gives less weight to the readings far from the line that it fits:

    ```{python}
    from sklearn.linear_model import HuberRegressor

    evaluate = pb.ablation.sklearn_evaluator(
        holdout,
        target="temp",
        features=["pressure"],
        estimator=HuberRegressor(),
        scoring=["r2", "neg_mean_absolute_error"],
    )

    pb.Ablation(validation, evaluate=evaluate, metric="r2").drop_each().run()
    ```

    This model is barely affected by the readings in degrees Fahrenheit, with an R² of about
    0.999 either way. Removing them still helps, and the verdict says so, since the difference is
    consistent across the repeats. But the difference is tiny (about 0.00005), and the mean absolute
    error only drops from about 0.253 degrees to 0.245. A clear verdict doesn't mean that the
    effect is large, so look at the size of the difference too.

    See Also
    --------
    Ablation : Measure how the rows that fail validation steps affect a model trained on the data.
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
