# Actions

Definition of action values.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/thresholds.py#L294-L501)

``` python
Actions(
    warning=None,
    error=None,
    critical=None,
    default=None,
    highest_only=True,
)
```

Actions complement threshold values by defining what action should be taken when a threshold level is reached. The action can be a string or a `Callable`. When a string is used, it is interpreted as a message to be displayed. When a `Callable` is used, it will be invoked at interrogation time if the threshold level is met or exceeded.

There are three threshold levels: 'warning', 'error', and 'critical'. These levels correspond to different levels of severity when a threshold is reached. Those thresholds can be defined using the [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) class or various shorthand forms. Actions don't have to be defined for all threshold levels; if an action is not defined for a level in exceedance, no action will be taken. Likewise, there is no negative consequence (other than a no-op) for defining actions for thresholds that don't exist (e.g., setting an action for the 'critical' level when no corresponding 'critical' threshold has been set).

## Parameters

`warning: str | Callable | list[str | Callable] | None = None`  
A string, `Callable`, or list of `Callable`/string values for the 'warning' level. Using `None` means no action should be performed at the 'warning' level.

`error: str | Callable | list[str | Callable] | None = None`  
A string, `Callable`, or list of `Callable`/string values for the 'error' level. Using `None` means no action should be performed at the 'error' level.

`critical: str | Callable | list[str | Callable] | None = None`  
A string, `Callable`, or list of `Callable`/string values for the 'critical' level. Using `None` means no action should be performed at the 'critical' level.

`default: str | Callable | list[str | Callable] | None = None`  
A string, `Callable`, or list of `Callable`/string values for all threshold levels. This parameter can be used to set the same action for all threshold levels. If an action is defined for a specific threshold level, it will override the action set for all levels.

`highest_only: bool = True`  
A boolean value that, when set to `True` (the default), results in executing only the action for the highest threshold level that is exceeded. Useful when you want to ensure that only the most severe action is taken when multiple threshold levels are exceeded.

## Returns

`Actions`  
An [Actions](../reference/Actions.md#pointblank.Actions) object. This can be used when using the [`Validate`](../reference/Validate.md#pointblank.Validate) class (to set actions for meeting different threshold levels globally) or when defining validation steps like [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) (so that actions are scoped to individual validation steps, overriding any globally set actions).

## Types Of Actions

Actions can be defined in different ways:

1.  **String**: A message to be displayed when the threshold level is met or exceeded.
2.  **Callable**: A function that is called when the threshold level is met or exceeded.
3.  **List of Strings/Callables**: Multiple messages or functions to be called when the threshold level is met or exceeded.

The actions are executed at interrogation time when the threshold level assigned to the action is exceeded by the number or proportion of failing test units. When providing a string, it will simply be printed to the console. A callable will also be executed at the time of interrogation. If providing a list of strings or callables, each item in the list will be executed in order. Such a list can contain a mix of strings and callables.

## String Templating

When using a string as an action, you can include placeholders for the following variables:

- `{type}`: The validation step type where the action is executed (e.g., 'col_vals_gt', 'col_vals_lt', etc.)
- `{level}`: The threshold level where the action is executed ('warning', 'error', or 'critical')
- `{step}` or `{i}`: The step number in the validation workflow where the action is executed
- `{col}` or `{column}`: The column name where the action is executed
- `{val}` or `{value}`: An associated value for the validation method (e.g., the value to compare against in a 'col_vals_gt' validation step)
- `{time}`: A datetime value for when the action was executed

The first two placeholders can also be used in uppercase (e.g., `{TYPE}` or `{LEVEL}`) and the corresponding values will be displayed in uppercase. The placeholders are replaced with the actual values during interrogation.

For example, the string `"{LEVEL}: '{type}' threshold exceeded for column {col}."` will be displayed as `"WARNING: 'col_vals_gt' threshold exceeded for column a."` when the 'warning' threshold is exceeded in a 'col_vals_gt' validation step involving column `a`.

## Crafting Callables With [get_action_metadata()](../reference/get_action_metadata.md#pointblank.get_action_metadata)

When creating a callable function to be used as an action, you can use the [`get_action_metadata()`](../reference/get_action_metadata.md#pointblank.get_action_metadata) function to retrieve metadata about the step where the action is executed. This metadata contains information about the validation step, including the step type, level, step number, column name, and associated value. You can use this information to craft your action message or to take specific actions based on the metadata provided.

## Examples

Let's define both threshold values and actions for a data validation workflow. We'll set these thresholds and actions globally for all validation steps. In this specific example, the only actions we'll define are for the 'critical' level:

``` python
import pointblank as pb

validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="game_revenue", tbl_type="duckdb"),
        thresholds=pb.Thresholds(warning=0.05, error=0.10, critical=0.15),
        actions=pb.Actions(critical="Major data quality issue found in step {step}."),
    )
    .col_vals_regex(columns="player_id", pattern=r"[A-Z]{12}[0-9]{3}")
    .col_vals_gt(columns="item_revenue", value=0.05)
    .col_vals_gt(columns="session_duration", value=15)
    .interrogate()
)

validation
```

    Major data quality issue found in step 3.

[TABLE]

Because we set the 'critical' action to display `"Major data quality issue found."` in the console, this message will be displayed if the number of failing test units exceeds the 'critical' threshold (set to 15% of the total number of test units). In step 3 of the validation workflow, the 'critical' threshold is exceeded, so the message is displayed in the console.

Actions can be defined locally for individual validation steps, which will override any global actions set at the beginning of the validation workflow. Here's a variation of the above example where we set global threshold values but assign an action only for an individual validation step:

``` python
def dq_issue():
    from datetime import datetime

    print(f"Data quality issue found ({datetime.now()}).")

validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="game_revenue", tbl_type="duckdb"),
        thresholds=pb.Thresholds(warning=0.05, error=0.10, critical=0.15),
    )
    .col_vals_regex(columns="player_id", pattern=r"[A-Z]{12}[0-9]{3}")
    .col_vals_gt(columns="item_revenue", value=0.05)
    .col_vals_gt(
        columns="session_duration",
        value=15,
        actions=pb.Actions(warning=dq_issue),
    )
    .interrogate()
)

validation
```

    Data quality issue found (2026-10-21 22:48:41.993349).

[TABLE]

In this case, the 'warning' action is set to call the `dq_issue()` function. This action is only executed when the 'warning' threshold is exceeded in the 'session_duration' column. Because all three thresholds are exceeded in step 3, the 'warning' action of executing the function occurs (resulting in a message being printed to the console). If actions were set for the other two threshold levels, they would also be executed.

## See Also

`The` [`get_action_metadata()`](../reference/get_action_metadata.md#pointblank.get_action_metadata) function, which can be used to  

`retrieve` metadata about the step where the action is executed.  
