# get_validation_summary()

Access validation summary information when authoring final actions.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L292-L425)

``` python
get_validation_summary()
```

This function provides a convenient way to access summary information about the validation process within a final action. It returns a dictionary with key metrics from the validation process. This function can only be used within callables crafted for the [`FinalActions`](../reference/FinalActions.md#pointblank.FinalActions) class.

## Returns

`dict | None`  
A dictionary containing validation metrics. If called outside of an final action context, this function will return `None`.

## Description Of The Summary Fields

The summary dictionary contains the following fields:

- `n_steps` (`int`): The total number of validation steps.
- `n_passing_steps` (`int`): The number of validation steps where all test units passed.
- `n_failing_steps` (`int`): The number of validation steps that had some failing test units.
- `n_warning_steps` (`int`): The number of steps that exceeded a 'warning' threshold.
- `n_error_steps` (`int`): The number of steps that exceeded an 'error' threshold.
- `n_critical_steps` (`int`): The number of steps that exceeded a 'critical' threshold.
- `list_passing_steps` (`list[int]`): List of step numbers where all test units passed.
- `list_failing_steps` (`list[int]`): List of step numbers for steps having failing test units.
- `dict_n` (`dict`): The number of test units for each validation step.
- `dict_n_passed` (`dict`): The number of test units that passed for each validation step.
- `dict_n_failed` (`dict`): The number of test units that failed for each validation step.
- `dict_f_passed` (`dict`): The fraction of test units that passed for each validation step.
- `dict_f_failed` (`dict`): The fraction of test units that failed for each validation step.
- `dict_warning` (`dict`): The 'warning' level status for each validation step.
- `dict_error` (`dict`): The 'error' level status for each validation step.
- `dict_critical` (`dict`): The 'critical' level status for each validation step.
- `all_passed` (`bool`): Whether or not every validation step had no failing test units.
- `highest_severity` (`str`): The highest severity level encountered during validation. This can be one of the following: `"warning"`, `"error"`, or `"critical"`, `"some failing"`, or `"all passed"`.
- `tbl_row_count` (`int`): The number of rows in the target table.
- `tbl_column_count` (`int`): The number of columns in the target table.
- `tbl_name` (`str`): The name of the target table.
- `validation_duration` (`float`): The duration of the validation in seconds.
- `dimension_scores` (`dict`): The test-unit-weighted health score (`0`-`100`) for each data quality dimension present in the validation (e.g., `{"completeness": 99.2, "validity": 97.8}`).
- `overall_health_score` (`float`): The overall, test-unit-weighted health score (`0`-`100`) across all dimensions.

Note that the summary dictionary is only available within the context of a final action. If called outside of a final action (i.e., when no final action is being executed), this function will return `None`.

## Examples

Final actions are executed after the completion of all validation steps. They provide an opportunity to take appropriate actions based on the overall validation results. Here's an example of a final action function (`send_report()`) that sends an alert when critical validation failures are detected. To keep the example self-contained, the alert is printed; in practice, this is where you'd send an email or post a message to a chat service:

``` python
import pointblank as pb

def send_report():
    summary = pb.get_validation_summary()
    if summary["highest_severity"] == "critical":
        print(f"CRITICAL validation failures in {summary['tbl_name']}")
        print(f"Steps with critical severity: {summary['n_critical_steps']}")

validation = (
    pb.Validate(
        data=pb.load_dataset("small_table"),
        tbl_name="small_table",
        thresholds=(1, 2, 3),
        final_actions=pb.FinalActions(send_report),
    )
    .col_vals_gt(columns="d", value=1000)
    .col_vals_not_null(columns="c")
    .interrogate()
)
```

    CRITICAL validation failures in small_table
    Steps with critical severity: 1

The first step has six failing test units, which reaches the 'critical' threshold of three, so the alert was printed during [interrogate()](../reference/Validate.interrogate.md#pointblank.Validate.interrogate). The second step's two missing values only reach the 'error' threshold. Outside of a final action, [get_validation_summary()](../reference/get_validation_summary.md#pointblank.get_validation_summary) returns `None`.

The [get_validation_summary()](../reference/get_validation_summary.md#pointblank.get_validation_summary) function can also be used to create custom reporting for validation results:

``` python
def log_validation_results():
    summary = pb.get_validation_summary()

    print(f"Validation completed with status: {summary['highest_severity'].upper()}")
    print(f"Steps: {summary['n_steps']} total")
    print(f"  - {summary['n_passing_steps']} passing, {summary['n_failing_steps']} failing")
    print(
        f"  - Severity: {summary['n_warning_steps']} warnings, "
        f"{summary['n_error_steps']} errors, "
        f"{summary['n_critical_steps']} critical"
    )

    if summary["highest_severity"] in ["error", "critical"]:
        print("Action required: Please review failing validation steps!")

validation = (
    pb.Validate(
        data=pb.load_dataset("small_table"),
        tbl_name="small_table",
        thresholds=(1, 2, 3),
        final_actions=pb.FinalActions(log_validation_results),
    )
    .col_vals_gt(columns="d", value=100)
    .col_vals_not_null(columns="c")
    .col_vals_le(columns="a", value=7)
    .interrogate()
)
```

    Validation completed with status: ERROR
    Steps: 3 total
      - 1 passing, 2 failing
      - Severity: 2 warnings, 1 errors, 0 critical
    Action required: Please review failing validation steps!

The first step passes, the second step's two missing values reach the 'error' threshold, and the third step's single failure reaches the 'warning' threshold. A step that reaches a higher threshold also counts toward the lower ones, which is why the severity counts can add up to more than the number of failing steps. Final actions work well with both simple logging and more complex notification systems, allowing you to integrate validation results into your broader data quality workflows.

## See Also

[FinalActions](../reference/FinalActions.md#pointblank.FinalActions)  
Create custom actions that run after all validation steps are done.
