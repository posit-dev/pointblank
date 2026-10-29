# Validate.critical()

Get the 'critical' level status for each validation step.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L19326-L19430)

``` python
Validate.critical(
    i=None,
    scalar=False,
)
```

The 'critical' status for a validation step is `True` if the fraction of failing test units meets or exceeds the threshold for the 'critical' level. Otherwise, the status is `False`.

The ascribed name of 'critical' is semantic and is thus simply a status indicator that could be used to trigger some action to be take. Here's how it fits in with other status indicators:

- 'warning': the status obtained by calling [`warning()`](../reference/Validate.warning.md#pointblank.Validate.warning), least severe
- 'error': the status obtained by calling [`error()`](../reference/Validate.error.md#pointblank.Validate.error), middle severity
- 'critical': the status obtained by calling [critical()](../reference/Validate.critical.md#pointblank.Validate.critical), most severe

This method provides a dictionary of the 'critical' status for each validation step. If the `scalar=True` argument is provided and `i=` is a scalar, the value is returned as a scalar instead of a dictionary.

## Parameters

`i: int | list[int] | None = None`  
The validation step number(s) from which the 'critical' status is obtained. Can be provided as a list of integers or a single integer. If `None`, all steps are included.

`scalar: bool = False`  
If `True` and `i=` is a scalar, return the value as a scalar instead of a dictionary.

## Returns

`dict[int, bool] | bool`  
A dictionary of the 'critical' status for each validation step or a scalar value.

## Examples

In the example below, we'll use a simple Polars DataFrame with three columns (`a`, `b`, and `c`). There will be three validation steps, and the first step will have many failing test units, the rest will be completely passing. We've set thresholds here for each of the steps by using `thresholds=(2, 4, 5)`, which means:

- the 'warning' threshold is `2` failing test units
- the 'error' threshold is `4` failing test units
- the 'critical' threshold is `5` failing test units

After interrogation, the [critical()](../reference/Validate.critical.md#pointblank.Validate.critical) method is used to determine the 'critical' status for each validation step.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [2, 4, 4, 7, 2, 3, 8],
        "b": [9, 8, 10, 5, 10, 6, 2],
        "c": ["a", "b", "a", "a", "b", "b", "a"]
    }
)

validation = (
    pb.Validate(data=tbl, thresholds=(2, 4, 5))
    .col_vals_gt(columns="a", value=5)
    .col_vals_lt(columns="b", value=15)
    .col_vals_in_set(columns="c", set=["a", "b"])
    .interrogate()
)

validation.critical()
```

    {1: True, 2: False, 3: False}

The returned dictionary provides the 'critical' status for each validation step. The first step has a `True` value since the number of failing test units meets the threshold for the 'critical' level. The second and third steps have `False` values since the number of failing test units was `0`, which is below the threshold for the 'critical' level.

We can also visually inspect the 'critical' status across all steps by viewing the validation table:

``` python
validation
```

[TABLE]

We can see that there are filled gray, yellow, and red circles in the first step (far right side, in the `W`, `E`, and `C` columns) indicating that the 'warning', 'error', and 'critical' thresholds were met. The other steps have empty gray, yellow, and red circles. This means that thresholds were 'set but not met' in those steps.

If we wanted to check the 'critical' status for a single validation step, we can provide the step number. Also, we could have the value returned as a scalar by setting `scalar=True` (ensuring that `i=` is a scalar).

``` python
validation.critical(i=1)
```

    {1: True}

The returned value is `True`, indicating that the first validation step had the 'critical' threshold met.
