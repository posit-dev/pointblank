# FinalActions

Define actions to be taken after validation is complete.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/thresholds.py#L504-L660)

``` python
FinalActions(*actions)
```

Final actions are executed after all validation steps have been completed. They provide a mechanism to respond to the overall validation results, such as sending alerts when critical failures are detected or generating summary reports.

## Parameters

`*actions`  
One or more actions to execute after validation. An action can be (1) a callable function that will be executed with no arguments, or (2) a string message that will be printed to the console.

## Returns

`FinalActions`  
An [FinalActions](../reference/FinalActions.md#pointblank.FinalActions) object. This can be used when using the [`Validate`](../reference/Validate.md#pointblank.Validate) class (to set final actions for the validation workflow).

## Types Of Actions

Final actions can be defined in two different ways:

1.  **String**: A message to be displayed when the validation is complete.
2.  **Callable**: A function that is called when the validation is complete.

The actions are executed at the end of the validation workflow. When providing a string, it will simply be printed to the console. A callable will also be executed at the time of validation completion. Several strings and callables can be provided to the [FinalActions](../reference/FinalActions.md#pointblank.FinalActions) class, and they will be executed in the order they are provided.

## Crafting Callables With [get_validation_summary()](../reference/get_validation_summary.md#pointblank.get_validation_summary)

When creating a callable function to be used as a final action, you can use the [`get_validation_summary()`](../reference/get_validation_summary.md#pointblank.get_validation_summary) function to retrieve the summary of the validation results. This summary contains information about the validation workflow, including the number of test units, the number of failing test units, and the threshold levels that were exceeded. You can use this information to craft your final action message or to take specific actions based on the validation results.

## Examples

Final actions provide a powerful way to respond to the overall results of a validation workflow. They're especially useful for sending notifications, generating reports, or taking corrective actions based on the complete validation outcome.

The following example shows how to create a final action that checks for critical failures and sends an alert. The `d` column of the `small_table` dataset has many values that aren't greater than `1000`, so the step reaches the 'critical' threshold:

``` python
import pointblank as pb

def send_alert():
    summary = pb.get_validation_summary()
    if summary["highest_severity"] == "critical":
        print(f"ALERT: Critical validation failures found in {summary['tbl_name']}")

validation = (
    pb.Validate(
        data=pb.load_dataset("small_table"),
        tbl_name="small_table",
        thresholds=(1, 2, 3),
        final_actions=pb.FinalActions(send_alert),
    )
    .col_vals_gt(columns="d", value=1000)
    .interrogate()
)
```

    ALERT: Critical validation failures found in small_table

In this example, the `send_alert()` function is defined to check the validation summary for critical failures. Since some were found, the alert message was printed. The function is passed to the [FinalActions](../reference/FinalActions.md#pointblank.FinalActions) class, which ensures it will be executed after all validation steps are complete. Note that we used the [`get_validation_summary()`](../reference/get_validation_summary.md#pointblank.get_validation_summary) function to retrieve the summary of the validation results to help craft the alert message.

Multiple final actions can be provided in a sequence. They will be executed in the order they are specified after all validation steps have completed. Here, a string message is followed by the `send_alert()` function and a second function that prints a count of the failing steps:

``` python
def log_counts():
    summary = pb.get_validation_summary()
    print(f"{summary['n_failing_steps']} of {summary['n_steps']} steps had failures")

validation = (
    pb.Validate(
        data=pb.load_dataset("small_table"),
        tbl_name="small_table",
        thresholds=(1, 2, 3),
        final_actions=pb.FinalActions(
            "Validation complete.",  # a string message
            send_alert,              # a callable function
            log_counts,              # another callable function
        ),
    )
    .col_vals_gt(columns="d", value=1000)
    .col_vals_not_null(columns="c")
    .col_exists(columns="a")
    .interrogate()
)
```

    Validation complete.
    ALERT: Critical validation failures found in small_table
    2 of 3 steps had failures

The three outputs appear in the order the actions were given. A string action is printed as is, while the functions can use the summary to decide what to do. Two of the three steps have failures (`d` has values of `1000` or less and `c` has missing values), and only the `col_exists()` step passes.

## See Also

[get_validation_summary()](../reference/get_validation_summary.md#pointblank.get_validation_summary)  
Get a summary of the validation results inside a final action.

[Actions](../reference/Actions.md#pointblank.Actions)  
Define actions that run when a step exceeds a threshold.
