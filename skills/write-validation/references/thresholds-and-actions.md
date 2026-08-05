# Thresholds and actions reference

## Thresholds

`Thresholds(warning=None, error=None, critical=None)`

### Value interpretation

| Value     | Meaning                       | Example             |
|-----------|-------------------------------|----------------------|
| `0.05`    | 5% of test units may fail     | 50 failures in 1000 |
| `5`       | At most 5 test units may fail | Absolute count       |
| `True`    | Any failure triggers          | Same as `1`          |
| `None`    | Level not used                | No threshold set     |

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

- `str` -- message template (printed to stdout)
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

# OpenTelemetry
pb.Actions(
    warning=pb.emit_otel(service_name="data-pipeline"),
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
    # meta keys: step, column, type, level, value, time, ...
```

## Final actions

`FinalActions(*actions)` -- run after all steps complete.

```python
def summary_check():
    summary = pb.get_validation_summary()
    if summary["n_failed_steps"] > 0:
        create_jira_ticket(summary)

pb.Validate(
    data=df,
    final_actions=pb.FinalActions(summary_check, "Validation complete at {time}"),
)
```

### get_validation_summary()

Inside a final action callable, call `pb.get_validation_summary()`:

```python
def report():
    s = pb.get_validation_summary()
    # s keys: n_steps, n_passed_steps, n_failed_steps,
    #         warn_count, error_count, critical_count, ...
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

### Log to OpenTelemetry

```python
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=0.05),
    actions=pb.Actions(
        warning=pb.emit_otel(service_name="my-pipeline"),
    ),
)
```
