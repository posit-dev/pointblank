# Validate.assert_below_threshold()

Raise an `AssertionError` if validation steps exceed a specified threshold level.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L18131-L18292)

``` python
Validate.assert_below_threshold(
    level="warning",
    i=None,
    message=None,
)
```

The [assert_below_threshold()](../reference/Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold) method checks whether validation steps' failure rates are below a given threshold level (`"warning"`, `"error"`, or `"critical"`). This is particularly useful in automated testing environments where you want to ensure your data quality meets minimum standards before proceeding.

If any validation step exceeds the specified threshold level, an `AssertionError` will be raised with details about which steps failed. If the validation has not yet been interrogated, this method will automatically call [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) with default parameters.

## Parameters

`level: str = "warning"`  
The threshold level to check against, which could be any of `"warning"` (the default), `"error"`, or `"critical"`. An `AssertionError` will be raised if any validation step exceeds this level.

`i: int | None = None`  
Specific validation step number(s) to check. Can be provided as a single integer or a list of integers. If `None` (the default), all steps are checked.

`message: str | None = None`  
Custom error message to use if assertion fails. If `None`, a default message will be generated that lists the specific steps that exceeded the threshold.

## Returns

`None`  

## Raises

`AssertionError`  
If any specified validation step exceeds the given threshold level.

`ValueError`  
If an invalid threshold level is provided.

## Examples

Below are some examples of how to use the [assert_below_threshold()](../reference/Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold) method. First, we'll create a simple Polars DataFrame with two columns (`a` and `b`).

``` python
import polars as pl

tbl = pl.DataFrame({
    "a": [7, 4, 9, 7, 12],
    "b": [9, 8, 10, 5, 10]
})
```

Then a validation plan will be created with thresholds (`warning=0.1`, `error=0.2`, `critical=0.3`). After interrogating, we display the validation report table:

``` python
import pointblank as pb

validation = (
    pb.Validate(data=tbl, thresholds=(0.1, 0.2, 0.3))
    .col_vals_gt(columns="a", value=5)   # 1 failing test unit
    .col_vals_lt(columns="b", value=10)  # 2 failing test units
    .interrogate()
)

validation
```

[TABLE]

Using `assert_below_threshold(level="warning")` will raise an `AssertionError` if any step exceeds the 'warning' threshold:

``` python
try:
    validation.assert_below_threshold(level="warning")
except AssertionError as e:
    print(f"Assertion failed: {e}")
```

    Assertion failed: The following steps exceeded the warning threshold level:
    Step 1: Expect that values in `a` should be > `5`.
    Step 2: Expect that values in `b` should be < `10`.

Check a specific step against the 'critical' threshold using the `i=` parameter:

``` python
validation.assert_below_threshold(level="critical", i=1)  # Won't raise an error
```

As the first step is below the 'critical' threshold (it exceeds the 'warning' and 'error' thresholds), no error is raised and nothing is printed.

We can also provide a custom error message with the `message=` parameter. Let's try that here:

``` python
try:
    validation.assert_below_threshold(
        level="error",
        message="Data quality too low for processing!"
    )
except AssertionError as e:
    print(f"Custom error: {e}")
```

    Custom error: Data quality too low for processing!

## See Also

- [`warning()`](../reference/Validate.warning.md#pointblank.Validate.warning)

: get the 'warning' status for each validation

`step`  

\- [`error()`](../reference/Validate.error.md#pointblank.Validate.error)  
get the 'error' status for each validation step

\- [`critical()`](../reference/Validate.critical.md#pointblank.Validate.critical)  
get the 'critical' status for each

`validation` step  

\- [`assert_passing()`](../reference/Validate.assert_passing.md#pointblank.Validate.assert_passing)  
assert all validations pass

`completely`  
