# Thresholds and actions reference

## Thresholds

`Thresholds(warning=None, error=None, critical=None)`

### Value interpretation

| Value     | Level is reached when...           | Example                |
|-----------|------------------------------------|------------------------|
| `0.05`    | 5% or more of test units fail      | 50 failures in 1000    |
| `5`       | 5 or more test units fail          | Absolute count         |
| `True`    | Any test unit fails                | Same as `1`            |
| `None`    | Level not used                     | No threshold set       |

`Thresholds` also accepts a tuple (`(0.01, 0.05, 0.10)`) or a dict
(`{"warning": 0.01, "error": 0.05}`) wherever `thresholds=` is taken.

### Setting thresholds

```python
# Global -- applies to all steps
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=0.01, error=0.05, critical=0.25),
)

# Per-step -- overrides global for that step
.col_vals_gt(
    columns="amount",
    value=0,
    thresholds=pb.Thresholds(warning=True),  # any failure warns
)

# Shorthand -- integer/float sets warning level
pb.Validate(data=df, thresholds=3)  # warning at 3 failures
```

## Actions

`Actions(warning=None, error=None, critical=None, default=None,
highest_only=True)`

Each level accepts:

- `str` -- message template (printed)
- `Callable` -- function to call
- `list[str | Callable]` -- multiple actions
- `None` -- no action

### Template variables

| Variable          | Description                      |
|-------------------|----------------------------------|
| `{type}`          | Validation method name           |
| `{level}`         | Threshold level triggered        |
| `{step}` or `{i}` | Step number                     |
| `{col}` or `{column}` | Column name                 |
| `{val}` or `{value}` | Comparison value              |
| `{time}`          | Timestamp                        |

Other placeholders (e.g. `{n_failed}`, `{n}`) are not substituted.

### Examples

```python
# String messages
pb.Actions(
    warning="Step {step}: {col} has warnings at {time}",
    error="ERROR in {col}: {type} check failed",
)

# Callable actions
def alert_on_error():
    metadata = pb.get_action_metadata()
    send_email(f"Error in step {metadata['step']}")

pb.Actions(error=alert_on_error)

# Slack notifications
pb.Actions(
    critical=pb.send_slack_notification(
        webhook_url="https://hooks.slack.com/services/...",
    ),
)

# Multiple actions per level
pb.Actions(
    error=[
        "Error in {col}",
        lambda: log_to_database(),
        pb.send_slack_notification(webhook_url="..."),
    ],
)
```

### highest_only

When `True` (default), only fires actions for the highest triggered
level. When `False`, fires all triggered levels.

### get_action_metadata()

Inside an action callable, call `pb.get_action_metadata()` to access
step details:

```python
def my_action():
    meta = pb.get_action_metadata()
    # meta keys: step, column, type, value, level, level_num, time,
    #            autobrief, failure_text
```

## Final actions

`FinalActions(*actions)` -- run after all steps complete.

```python
def summary_check():
    summary = pb.get_validation_summary()
    if summary["n_failing_steps"] > 0:
        create_jira_ticket(summary)

pb.Validate(
    data=df,
    final_actions=pb.FinalActions(summary_check, "Validation complete."),
)
```

String final actions are printed as they are: template variables like
`{time}` are not substituted (use a callable for dynamic messages).

### get_validation_summary()

Inside a final action callable, call `pb.get_validation_summary()`:

```python
def report():
    s = pb.get_validation_summary()
    # s keys: n_steps, n_passing_steps, n_failing_steps,
    #         n_warning_steps, n_error_steps, n_critical_steps,
    #         list_passing_steps, list_failing_steps, all_passed,
    #         highest_severity, overall_health_score, dimension_scores,
    #         tbl_name, tbl_row_count, tbl_column_count,
    #         validation_duration, and dict_* (per-step dicts)
```

## Common patterns

### Warn-then-stop pipeline

```python
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=0.01, critical=0.10),
    actions=pb.Actions(
        warning="Data quality warning: {col}",
        critical=lambda: sys.exit(1),
    ),
)
```

### Slack on any failure

```python
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=True),
    actions=pb.Actions(
        warning=pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/...",
        ),
    ),
)
```

### Send to OpenTelemetry

`emit_otel()` is a final action (use it in `FinalActions`, not `Actions`).
It sends metrics by default, plus trace spans and log records when asked,
after every `interrogate()`:

```python
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=0.05),
    final_actions=pb.FinalActions(
        pb.emit_otel(
            enable_tracing=True,
            extra_attributes={"pipeline": "my-pipeline"},
        ),
    ),
)
```
