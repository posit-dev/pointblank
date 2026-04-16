## Validate.get_json_report()


Get a report of the validation results as a JSON-formatted string.


Usage

``` python
Validate.get_json_report(
    use_fields=None,
    exclude_fields=None,
)
```


The [get_json_report()](Validate.get_json_report.md#pointblank.Validate.get_json_report) method provides a machine-readable report of validation results in JSON format. This is particularly useful for programmatic processing, storing validation results, or integrating with other systems. The report includes detailed information about each validation step, such as assertion type, columns validated, threshold values, test results, and more.

By default, all available validation information fields are included in the report. However, you can customize the fields to include or exclude using the `use_fields=` and `exclude_fields=` parameters.


## Parameters


`use_fields: list[str] | None = None`  
An optional list of specific fields to include in the report. If provided, only these fields will be included in the JSON output. If `None` (the default), all standard validation report fields are included. Have a look at the *Available Report Fields* section below for a list of fields that can be included in the report.

`exclude_fields: list[str] | None = None`  
An optional list of fields to exclude from the report. If provided, these fields will be omitted from the JSON output. If `None` (the default), no fields are excluded. This parameter cannot be used together with `use_fields=`. The *Available Report Fields* provides a listing of fields that can be excluded from the report.


## Returns


`str`  
A JSON-formatted string representing the validation report, with each validation step as an object in the report array.


## Available Report Fields

The JSON report can include any of the standard validation report fields, including:

- `i`: the step number (1-indexed)
- `i_o`: the original step index from the validation plan (pre-expansion)
- `assertion_type`: the type of validation assertion (e.g., `"col_vals_gt"`, etc.)
- `column`: the column being validated (or columns used in certain validations)
- `values`: the comparison values or parameters used in the validation
- `inclusive`: whether the comparison is inclusive (for range-based validations)
- `na_pass`: whether `NA`/`Null` values are considered passing (for certain validations)
- `pre`: preprocessing function applied before validation
- `segments`: data segments to which the validation was applied
- `thresholds`: threshold level statement that was used for the validation step
- `label`: custom label for the validation step
- `brief`: a brief description of the validation step
- `active`: whether the validation step is active
- [all_passed](Validate.all_passed.md#pointblank.Validate.all_passed): whether all test units passed in the step
- [n](Validate.n.md#pointblank.Validate.n): total number of test units
- [n_passed](Validate.n_passed.md#pointblank.Validate.n_passed), [n_failed](Validate.n_failed.md#pointblank.Validate.n_failed): number of test units that passed and failed
- [f_passed](Validate.f_passed.md#pointblank.Validate.f_passed), [f_failed](Validate.f_failed.md#pointblank.Validate.f_failed): Fraction of test units that passed and failed
- [warning](Validate.warning.md#pointblank.Validate.warning), [error](Validate.error.md#pointblank.Validate.error), [critical](Validate.critical.md#pointblank.Validate.critical): whether the namesake threshold level was exceeded (is `null` if threshold not set)
- `time_processed`: when the validation step was processed (ISO 8601 format)
- `proc_duration_s`: the processing duration in seconds


## Examples

Let's create a validation plan with a few validation steps and generate a JSON report of the results:


``` python
import pointblank as pb
import polars as pl

# Create a sample DataFrame
tbl = pl.DataFrame({
    "a": [5, 7, 8, 9],
    "b": [3, 4, 2, 1]
})

# Create and execute a validation plan
validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns="a", value=6)
    .col_vals_lt(columns="b", value=4)
    .interrogate()
)

# Get the full JSON report
json_report = validation.get_json_report()

print(json_report)
```


    [
        {
            "i": 1,
            "i_o": 1,
            "assertion_type": "col_vals_gt",
            "column": "a",
            "values": 6,
            "inclusive": null,
            "na_pass": false,
            "pre": null,
            "segments": null,
            "thresholds": "Thresholds(warning=None, error=None, critical=None)",
            "label": null,
            "brief": null,
            "active": true,
            "all_passed": false,
            "n": 4,
            "n_passed": 3,
            "n_failed": 1,
            "f_passed": 0.75,
            "f_failed": 0.25,
            "warning": null,
            "error": null,
            "critical": null,
            "time_processed": "2026-04-16T03:02:44.339+00:00",
            "proc_duration_s": 0.0069
        },
        {
            "i": 2,
            "i_o": 2,
            "assertion_type": "col_vals_lt",
            "column": "b",
            "values": 4,
            "inclusive": null,
            "na_pass": false,
            "pre": null,
            "segments": null,
            "thresholds": "Thresholds(warning=None, error=None, critical=None)",
            "label": null,
            "brief": null,
            "active": true,
            "all_passed": false,
            "n": 4,
            "n_passed": 3,
            "n_failed": 1,
            "f_passed": 0.75,
            "f_failed": 0.25,
            "warning": null,
            "error": null,
            "critical": null,
            "time_processed": "2026-04-16T03:02:44.342+00:00",
            "proc_duration_s": 0.002689
        }
    ]


You can also customize which fields to include:


``` python
json_report = validation.get_json_report(
    use_fields=["i", "assertion_type", "column", "n_passed", "n_failed"]
)

print(json_report)
```


    [
        {
            "i": 1,
            "assertion_type": "col_vals_gt",
            "column": "a",
            "n_passed": 3,
            "n_failed": 1
        },
        {
            "i": 2,
            "assertion_type": "col_vals_lt",
            "column": "b",
            "n_passed": 3,
            "n_failed": 1
        }
    ]


Or which fields to exclude:


``` python
json_report = validation.get_json_report(
    exclude_fields=[
        "i_o", "thresholds", "pre", "segments", "values",
        "na_pass", "inclusive", "label", "brief", "active",
        "time_processed", "proc_duration_s"
    ]
)

print(json_report)
```


    [
        {
            "i": 1,
            "assertion_type": "col_vals_gt",
            "column": "a",
            "all_passed": false,
            "n": 4,
            "n_passed": 3,
            "n_failed": 1,
            "f_passed": 0.75,
            "f_failed": 0.25,
            "warning": null,
            "error": null,
            "critical": null
        },
        {
            "i": 2,
            "assertion_type": "col_vals_lt",
            "column": "b",
            "all_passed": false,
            "n": 4,
            "n_passed": 3,
            "n_failed": 1,
            "f_passed": 0.75,
            "f_failed": 0.25,
            "warning": null,
            "error": null,
            "critical": null
        }
    ]


The JSON output can be further processed or analyzed programmatically:


``` python
import json

# Parse the JSON report
report_data = json.loads(validation.get_json_report())

# Extract and analyze validation results
failing_steps = [step for step in report_data if step["n_failed"] > 0]
print(f"Number of failing validation steps: {len(failing_steps)}")
```


    Number of failing validation steps: 2


#### See Also

[report](report.md), [failed](failed.md)
