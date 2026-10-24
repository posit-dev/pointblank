# Validate.above_threshold()

Check if any validation steps exceed a specified threshold level.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L18163-L18302)

``` python
Validate.above_threshold(
    level="warning",
    i=None,
)
```

The [above_threshold()](../reference/Validate.above_threshold.md#pointblank.Validate.above_threshold) method checks whether validation steps exceed a given threshold level. This provides a non-exception-based alternative to [`assert_below_threshold()`](../reference/Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold) for conditional workflow control based on validation results.

This method is useful in scenarios where you want to check if any validation steps failed beyond a certain threshold without raising an exception, allowing for more flexible programmatic responses to validation issues.

## Parameters

`level: str = "warning"`  
The threshold level to check against. Valid options are: `"warning"` (the least severe threshold level), `"error"` (the middle severity threshold level), and `"critical"` (the most severe threshold level). The default is `"warning"`.

`i: int | None = None`  
Specific validation step number(s) to check. If a single integer, checks only that step. If a list of integers, checks all specified steps. If `None` (the default), checks all validation steps. Step numbers are 1-based (first step is `1`, not `0`).

## Returns

`bool`  
`True` if any of the specified validation steps exceed the given threshold level, `False` otherwise.

## Raises

`ValueError`  
If an invalid threshold level is provided.

## Examples

Below are some examples of how to use the [above_threshold()](../reference/Validate.above_threshold.md#pointblank.Validate.above_threshold) method. First, we'll create a simple Polars DataFrame with a single column ([values](../reference/MetadataPackage.md#pointblank.MetadataPackage.values)).

``` python
import polars as pl

tbl = pl.DataFrame({
    "values": [1, 2, 3, 4, 5, 0, -1]
})
```

Then a validation plan will be created with thresholds (`warning=0.1`, `error=0.2`, `critical=0.3`). After interrogating, we display the validation report table:

``` python
import pointblank as pb

validation = (
    pb.Validate(data=tbl, thresholds=(0.1, 0.2, 0.3))
    .col_vals_gt(columns="values", value=0)
    .col_vals_lt(columns="values", value=10)
    .col_vals_between(columns="values", left=0, right=5)
    .interrogate()
)

validation
```

[TABLE]

Let's check if any steps exceed the 'warning' threshold with the [above_threshold()](../reference/Validate.above_threshold.md#pointblank.Validate.above_threshold) method. A message will be printed if that's the case:

``` python
if validation.above_threshold(level="warning"):
    print("Some steps have exceeded the warning threshold")
```

    Some steps have exceeded the warning threshold

Check if only steps 2 and 3 exceed the 'error' threshold through use of the `i=` argument:

``` python
if validation.above_threshold(level="error", i=[2, 3]):
    print("Steps 2 and/or 3 have exceeded the error threshold")
```

You can use this in a workflow to conditionally trigger processes. Here's a snippet of how you might use this in a function:

``` python
def process_data(validation_obj):
    # Only continue processing if validation passes critical thresholds
    if not validation_obj.above_threshold(level="critical"):
        # Continue with processing
        print("Data meets critical quality thresholds, proceeding...")
        return True
    else:
        # Log failure and stop processing
        print("Data fails critical quality checks, aborting...")
        return False
```

Note that this is just a suggestion for how to implement conditional workflow processes. You should adapt this pattern to your specific requirements, which might include different threshold levels, custom logging mechanisms, or integration with your organization's data pipelines and notification systems.

## See Also

- [`assert_below_threshold()`](../reference/Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold)

: a similar

`method` that raises an exception if thresholds are exceeded  

\- [`warning()`](../reference/Validate.warning.md#pointblank.Validate.warning)  
get the 'warning' status for each validation

`step`  

\- [`error()`](../reference/Validate.error.md#pointblank.Validate.error)  
get the 'error' status for each validation step

\- [`critical()`](../reference/Validate.critical.md#pointblank.Validate.critical)  
get the 'critical' status for each

`validation` step  
