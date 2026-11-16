# validate()

Check a table against the contract.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L838-L930)

``` python
validate(data)
```

This is the usual way to apply a contract. It compiles the contract into a validation plan for the table and interrogates it right away. This is the same as calling [`to_validate()`](../reference/pointblank.Contract.to_validate.md) and then [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate). The result is an ordinary interrogated [`Validate`](../reference/Validate.md#pointblank.Validate) object. You can display it as a report, inspect individual steps with [`get_step_report()`](../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report), pull out failing rows with [`get_data_extracts()`](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts), or check the outcome in code.

A failing contract doesn't raise an error or warning here, whatever the contract's `on_violation=` setting is. That setting is applied by [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline). When you call [validate()](../reference/pointblank.Contract.validate.md) yourself, check the result with a method like [`all_passed()`](../reference/Validate.all_passed.md#pointblank.Validate.all_passed), [`assert_passing()`](../reference/Validate.assert_passing.md#pointblank.Validate.assert_passing), or [`assert_below_threshold()`](../reference/Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold). To add checks before the validation runs, use [`to_validate()`](../reference/pointblank.Contract.to_validate.md) instead.

## Parameters

`data: IntoDataFrame`  
The table to validate against the contract. Any table type that [Validate](../reference/Validate.md#pointblank.Validate) accepts can be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

## Returns

`Validate`  
An interrogated [Validate](../reference/Validate.md#pointblank.Validate) object with the results of every contract check.

## Raises

`TypeError`  
If a step's arguments don't fit its validation method (for example, a misspelled parameter name). Step arguments are only checked when the contract is compiled.

## Examples

Here's a contract for a sensor readings feed. Each reading needs a sensor ID, the temperature must be within a plausible range, and the feed must have five rows. Any failing test unit counts as an error.

``` python
import pointblank as pb
import polars as pl

readings_contract = pb.Contract(
    name="sensor_readings",
    owner="iot-platform",
    steps=[
        pb.Step("col_vals_not_null", columns="sensor_id"),
        pb.Step("col_vals_between", columns="temp_c", left=-40, right=60),
        pb.Step("row_count_match", count=5),
    ],
    thresholds=pb.Thresholds(error=1),
)

readings = pl.DataFrame(
    {
        "sensor_id": ["s1", "s2", None, "s4", "s5"],
        "temp_c": [21.5, 19.0, 22.1, 85.0, 20.4],
    }
)

validation = readings_contract.validate(readings)

validation
```

[TABLE]

The row count is right, but the report shows two problems: one reading has no sensor ID, and one temperature (85.0) is out of range. Both steps reach the error threshold. Because [validate()](../reference/pointblank.Contract.validate.md) doesn't raise, a script or scheduled job decides what happens next. For example, it could test whether any step reached the error level:

``` python
validation.above_threshold(level="error")
```

    True

In a pipeline where bad data should stop processing, [`assert_below_threshold()`](../reference/Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold) raises an `AssertionError` in this situation instead.
