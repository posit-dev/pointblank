# AblationVariant

A version of the training table evaluated in an ablation.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L63-L121)

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
)
```

An [`Ablation`](../reference/Ablation.md#pointblank.Ablation) evaluates a model on several versions of the target table of a validation: the full table, the table with the rows that failed some of the steps removed (a variant), and the table with the same number of rows removed at random (a control). Each time it calls the `evaluate=` function, it passes the version being evaluated as the `variant=` argument, so the function can tell them apart (e.g., to log the runs, or to name the files that a model is saved to). The variants of an ablation are also listed in its [`plan()`](../reference/Ablation.plan.md#pointblank.Ablation.plan) and its results.

## Attributes

`id: str`  
A short identifier for the version, unique within the ablation: `"full"` for the full table, `"step_<i>"` for a variant of [`drop_each()`](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each), `"all"` for [`drop_all()`](../reference/Ablation.drop_all.md#pointblank.Ablation.drop_all), the `name=` (or `"steps_<i>_<j>"`) for [`drop_steps()`](../reference/Ablation.drop_steps.md#pointblank.Ablation.drop_steps), and `"control_<n>"` for the control that removes `n` random rows.

`strategy: str`  
The kind of version: `"full"`, `"drop_steps"`, `"drop_each"`, `"drop_all"`, or `"control"`.

`name: str`  
A name for the version, as shown in reports (e.g., `"Step 2"`).

`steps: tuple[int, …]`  
The numbers of the validation steps whose failing rows were removed (empty for the full table and for controls).

`n_removed: int`  
The number of rows removed from the table.

`f_removed: float`  
The share of the table's rows that were removed (from `0` to `1`).

`is_control: bool`  
Is this a control (random rows removed) rather than a variant?

`group: Any`  
Reserved for strategies that remove groups of rows (always `None` for now).
