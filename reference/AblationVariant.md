# AblationVariant

A version of the training table evaluated in an ablation.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L118-L229)

``` python
AblationVariant(
    id,
    strategy,
    name,
    steps=(),
    n_removed=0,
    f_removed=0.0,
    is_control=False,
    group=None,
    group_column=None,
    left_out=None,
    sweep=None,
    value=None,
    corruption=None,
    column=None,
    n_corrupted=0,
    checked_by=None,
)
```

An [`Ablation`](../reference/Ablation.md#pointblank.Ablation) evaluates a model on several versions of the target table of a validation: the full table, the table with the rows that failed some of the steps removed (a variant), and the table with the same number of rows removed at random (a control). Each time it calls the `evaluate=` function, it passes the version being evaluated as the `variant=` argument, so the function can tell them apart (e.g., to log the runs, or to name the files that a model is saved to). The variants of an ablation are also listed in its [`plan()`](../reference/Ablation.plan.md#pointblank.Ablation.plan) and its results.

## Attributes

`id: str`  
A short identifier for the version, unique within the ablation: `"full"` for the full table, `"step_<i>"` for a variant of [`drop_each()`](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each), `"all"` for [`drop_all()`](../reference/Ablation.drop_all.md#pointblank.Ablation.drop_all), the `name=` (or `"steps_<i>_<j>"`) for [`drop_steps()`](../reference/Ablation.drop_steps.md#pointblank.Ablation.drop_steps), `"all_but_<i>"` for [`drop_all_but_each()`](../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each), `"<column>=<value>"` for [`drop_groups()`](../reference/Ablation.drop_groups.md#pointblank.Ablation.drop_groups), `"<score>:lowest_<q>"` (or `"<score>:below_<value>"`, and the same with `highest` and `above`) for [`sweep_score()`](../reference/Ablation.sweep_score.md#pointblank.Ablation.sweep_score), where `<score>` is `step_<i>` or the score column, `"step_<i>:<param>=<value>"` for [`sweep_step()`](../reference/Ablation.sweep_step.md#pointblank.Ablation.sweep_step), `"inject:<corruption>:<column>_<rate>"` (or `"inject:duplicate_<rate>"`) for [`inject()`](../reference/Ablation.inject.md#pointblank.Ablation.inject), `"shapley_<i>_<j>"` for a group of steps that only [`shapley()`](../reference/Ablation.shapley.md#pointblank.Ablation.shapley) needs, and `"control_<n>"` for the control that removes `n` random rows (or `"control_<id>"` for the control of a [drop_all_but_each()](../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each) variant).

`strategy: str`  
The kind of version: `"full"`, `"drop_steps"`, `"drop_each"`, `"drop_all"`, `"drop_all_but_each"`, `"drop_groups"`, `"sweep_score"`, `"sweep_step"`, `"inject"`, `"shapley"`, or `"control"`.

`name: str`  
A name for the version, as shown in reports (e.g., `"Step 2"`).

`steps: tuple[int, …]`  
The numbers of the validation steps whose failing rows were removed (empty for the full table, for groups, for the points of a sweep, for injected defects, and for most controls).

`n_removed: int`  
The number of rows removed from the table.

`f_removed: float`  
The share of the table's rows that were removed (from `0` to `1`).

`is_control: bool`  
Is this a control (random rows removed) rather than a variant?

`group: Any`  
For a variant of [drop_groups()](../reference/Ablation.drop_groups.md#pointblank.Ablation.drop_groups), the value of `group_column` in the rows it removes (`None` for the rows where it's missing).

`group_column: str | None`  
For a variant of [drop_groups()](../reference/Ablation.drop_groups.md#pointblank.Ablation.drop_groups), the column the rows are grouped by (otherwise `None`).

`left_out: int | None`  
For a variant of [drop_all_but_each()](../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each) (and its control), the step whose rows are kept (otherwise `None`).

`sweep: str | None`  
For a point of a sweep (from [sweep_score()](../reference/Ablation.sweep_score.md#pointblank.Ablation.sweep_score) or [sweep_step()](../reference/Ablation.sweep_step.md#pointblank.Ablation.sweep_step)), the id of the sweep, which its points share (e.g., `"step_2:lowest"` or `"step_2:min_val"`; otherwise `None`).

`value: float | None`  
For a point of a sweep, its value: the share of the scored rows removed, the score threshold, or the value of the step's parameter. For a variant of [inject()](../reference/Ablation.inject.md#pointblank.Ablation.inject), the share of the rows corrupted (otherwise `None`).

`corruption: str | None`  
For a variant of [`inject()`](../reference/Ablation.inject.md#pointblank.Ablation.inject), the defect put into the rows: `"null"`, `"shuffle"`, `"truncate"`, `"duplicate"`, or the name of the corruption function (otherwise `None`).

`column: str | None`  
For a variant of [inject()](../reference/Ablation.inject.md#pointblank.Ablation.inject), the column whose values were corrupted (`None` for `"duplicate"`, and otherwise).

`n_corrupted: int`  
For a variant of [inject()](../reference/Ablation.inject.md#pointblank.Ablation.inject), the number of rows corrupted (or copied, for `"duplicate"`).

`checked_by: int | None`  
For a variant of [inject()](../reference/Ablation.inject.md#pointblank.Ablation.inject), the step that checks for the defect, given as its `step=` (otherwise `None`).
