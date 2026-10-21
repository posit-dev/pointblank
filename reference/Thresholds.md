# Thresholds

Definition of threshold values.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/thresholds.py#L9-L188)

``` python
Thresholds(
    warning=None,
    error=None,
    critical=None,
)
```

Thresholds are used to set limits on the number of failing test units at different levels. The levels are 'warning', 'error', and 'critical'. These levels correspond to different levels of severity when a threshold is reached. The threshold values can be set as absolute counts or as fractions of the total number of test units. When a threshold is reached, an action can be taken (e.g., displaying a message or calling a function) if there is an associated action defined for that level (defined through the [`Actions`](../reference/Actions.md#pointblank.Actions) class).

## Parameters

`warning: int | float | bool | None = None`  
The threshold for the 'warning' level. This can be an absolute count or a fraction of the total. Using `True` will set this threshold value to `1`.

`error: int | float | bool | None = None`  
The threshold for the 'error' level. This can be an absolute count or a fraction of the total. Using `True` will set this threshold value to `1`.

`critical: int | float | bool | None = None`  
The threshold for the 'critical' level. This can be an absolute count or a fraction of the total. Using `True` will set this threshold value to `1`.

## Returns

`Thresholds`  
A [Thresholds](../reference/Thresholds.md#pointblank.Thresholds) object. This can be used when using the [`Validate`](../reference/Validate.md#pointblank.Validate) class (to set thresholds globally) or when defining validation steps like [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) (so that threshold values are scoped to individual validation steps, overriding any global thresholds).

## Examples

In a data validation workflow, you can set thresholds for the number of failing test units at different levels. For example, you can set a threshold for the 'warning' level when the number of failing test units exceeds 10% of the total number of test units:

``` python
thresholds_1 = pb.Thresholds(warning=0.1)
```

You can also set thresholds for the 'error' and 'critical' levels:

``` python
thresholds_2 = pb.Thresholds(warning=0.1, error=0.2, critical=0.05)
```

Thresholds can also be set as absolute counts. Here's an example where the 'warning' level is set to `5` failing test units:

``` python
thresholds_3 = pb.Thresholds(warning=5)
```

The `thresholds` object can be used to set global thresholds for all validation steps. Or, you can set thresholds for individual validation steps, which will override the global thresholds. Here's a data validation workflow example where we set global thresholds and then override with different thresholds at the [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) step:

``` python
validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="small_table"),
        label="Example Validation",
        thresholds=pb.Thresholds(warning=0.1, error=0.2, critical=0.3)
    )
    .col_vals_not_null(columns=["c", "d"])
    .col_vals_gt(columns="a", value=3, thresholds=pb.Thresholds(warning=5))
    .interrogate()
)

validation
```

[TABLE]

As can be seen, the last step ([`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)) has its own thresholds, which override the global thresholds set at the beginning of the validation workflow (in the [`Validate`](../reference/Validate.md#pointblank.Validate) class).
