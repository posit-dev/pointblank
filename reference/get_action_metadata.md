# get_action_metadata()

Access step-level metadata when authoring custom actions.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L198-L271)

``` python
get_action_metadata()
```

Get the metadata for the validation step where an action was triggered. This can be called by user functions to get the metadata for the current action. This function can only be used within callables crafted for the [`Actions`](../reference/Actions.md#pointblank.Actions) class.

## Returns

`dict | None`  
A dictionary containing the metadata for the current step. If called outside of an action (i.e., when no action is being executed), this function will return `None`.

## Description Of The Metadata Fields

The metadata dictionary contains the following fields for a given validation step:

- `step`: The step number.
- `column`: The column name.
- `value`: The value being compared (only available in certain validation steps).
- `type`: The assertion type (e.g., `"col_vals_gt"`, etc.).
- `time`: The time the validation step was executed (in ISO format).
- `level`: The severity level (`"warning"`, `"error"`, or `"critical"`).
- `level_num`: The severity level as a numeric value (`30`, `40`, or `50`).
- `autobrief`: A localized and brief statement of the expectation for the step.
- `failure_text`: Localized text that explains how the validation step failed.

## Examples

When creating a custom action, you can access the metadata for the current step using the [get_action_metadata()](../reference/get_action_metadata.md#pointblank.get_action_metadata) function. Here's an example of a custom action that logs the metadata for the current step:

``` python
import pointblank as pb

def log_issue():
    metadata = pb.get_action_metadata()
    print(f"Type: {metadata['type']}, Step: {metadata['step']}")

validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="game_revenue", tbl_type="duckdb"),
        thresholds=pb.Thresholds(warning=0.05, error=0.10, critical=0.15),
        actions=pb.Actions(warning=log_issue),
    )
    .col_vals_regex(columns="player_id", pattern=r"[A-Z]{12}[0-9]{3}")
    .col_vals_gt(columns="item_revenue", value=0.05)
    .col_vals_gt(
        columns="session_duration",
        value=15,
    )
    .interrogate()
)

validation
```

    Type: col_vals_gt, Step: 2
    Type: col_vals_gt, Step: 3

[TABLE]

Key pieces to note in the above example:

- `log_issue()` (the custom action) collects [metadata](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.metadata) by calling [get_action_metadata()](../reference/get_action_metadata.md#pointblank.get_action_metadata)
- the [metadata](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.metadata) is a dictionary that is used to craft the log message
- the action is passed as a bare function to the [Actions](../reference/Actions.md#pointblank.Actions) object within the [Validate](../reference/Validate.md#pointblank.Validate) object (placing it within `Validate(actions=)` ensures it's set as an action for every validation step)

## See Also

`Have` a look at [`Actions`](../reference/Actions.md#pointblank.Actions) for more information on how to create custom  

`actions` for validation steps that exceed a set threshold value.  
