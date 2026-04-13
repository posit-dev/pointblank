## FinalActions


Define actions to be taken after validation is complete.


Usage

``` python
FinalActions()
```


Final actions are executed after all validation steps have been completed. They provide a mechanism to respond to the overall validation results, such as sending alerts when critical failures are detected or generating summary reports.


## Parameters


`*actions`  
One or more actions to execute after validation. An action can be (1) a callable function that will be executed with no arguments, or (2) a string message that will be printed to the console.


## Returns


`FinalActions`  
An [FinalActions](FinalActions.md#pointblank.FinalActions) object. This can be used when using the <a href="Validate.html#pointblank.Validate" class="gdls-link"><code>Validate</code></a> class (to set final actions for the validation workflow).


## Types Of Actions

Final actions can be defined in two different ways:

1.  **String**: A message to be displayed when the validation is complete.
2.  **Callable**: A function that is called when the validation is complete.

The actions are executed at the end of the validation workflow. When providing a string, it will simply be printed to the console. A callable will also be executed at the time of validation completion. Several strings and callables can be provided to the [FinalActions](FinalActions.md#pointblank.FinalActions) class, and they will be executed in the order they are provided.


## Crafting Callables With [get_validation_summary()](get_validation_summary.md#pointblank.get_validation_summary)

When creating a callable function to be used as a final action, you can use the <a href="get_validation_summary.html#pointblank.get_validation_summary" class="gdls-link"><code>get_validation_summary()</code></a> function to retrieve the summary of the validation results. This summary contains information about the validation workflow, including the number of test units, the number of failing test units, and the threshold levels that were exceeded. You can use this information to craft your final action message or to take specific actions based on the validation results.


## Examples

Final actions provide a powerful way to respond to the overall results of a validation workflow. They're especially useful for sending notifications, generating reports, or taking corrective actions based on the complete validation outcome.

The following example shows how to create a final action that checks for critical failures and sends an alert:

``` python
import pointblank as pb

def send_alert():
    summary = pb.get_validation_summary()
    if summary["highest_severity"] == "critical":
        print(f"ALERT: Critical validation failures found in {summary['tbl_name']}")

validation = (
    pb.Validate(
        data=my_data,
        final_actions=pb.FinalActions(send_alert)
    )
    .col_vals_gt(columns="revenue", value=0)
    .interrogate()
)
```

In this example, the `send_alert()` function is defined to check the validation summary for critical failures. If any are found, an alert message is printed to the console. The function is passed to the [FinalActions](FinalActions.md#pointblank.FinalActions) class, which ensures it will be executed after all validation steps are complete. Note that we used the <a href="get_validation_summary.html#pointblank.get_validation_summary" class="gdls-link"><code>get_validation_summary()</code></a> function to retrieve the summary of the validation results to help craft the alert message.

Multiple final actions can be provided in a sequence. They will be executed in the order they are specified after all validation steps have completed:

``` python
validation = (
    pb.Validate(
        data=my_data,
        final_actions=pb.FinalActions(
            "Validation complete.",  # a string message
            send_alert,              # a callable function
            generate_report          # another callable function
        )
    )
    .col_vals_gt(columns="revenue", value=0)
    .interrogate()
)
```


#### See Also

[The](The.md), [used](used.md)
