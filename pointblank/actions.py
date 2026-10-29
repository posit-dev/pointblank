from __future__ import annotations

from datetime import datetime
from typing import Callable

import requests

__all__ = [
    "send_slack_notification",
]


def send_slack_notification(
    webhook_url: str | None = None,
    step_msg: str | None = None,
    summary_msg: str | None = None,
    debug: bool = False,
) -> Callable | None:
    """Create an action that sends a Slack notification about a validation.

    Slack is a common place for a team to hear about problems with its data. The
    `send_slack_notification()` function creates an action that posts a message to a Slack channel
    through an [incoming webhook](https://api.slack.com/messaging/webhooks). The same action works
    in two places:

    1. In [`Actions`](`pointblank.Actions`), it sends a message about a step that exceeded a
    threshold level.
    2. In [`FinalActions`](`pointblank.FinalActions`), it sends a summary of the whole validation
    after the interrogation.

    The action works out from where it runs which kind of message to send. The messages are made
    from templates (`step_msg=` and `summary_msg=`), which can use Slack's Markdown formatting.

    A notification that can't be sent (because of a wrong webhook URL, a network error or a
    template with an unknown variable) doesn't stop the validation. The error is printed as
    `Failed to send Slack notification: ...` instead of being raised.

    Without a webhook URL, nothing is sent to Slack, which is useful for trying out templates:

    - With `webhook_url=None`, the action prints the message that it would send.
    - With `webhook_url=None` and `debug=True`, the function prints previews of both templates
    filled in with sample values. It then returns `None` instead of an action.

    Parameters
    ----------
    webhook_url
        The URL of a Slack incoming webhook (`https://hooks.slack.com/services/...`). Since anyone
        with the URL can post to the channel, it's best read from an environment variable rather
        than written in the code. The default is `None`, which prints the messages instead of
        sending them.
    step_msg
        The template for messages about a step, using the variables in *Template Variables for
        Step Messages* below. The default is `None`, which uses a template with the assertion
        type, the threshold level, the step number, the column and the time.
    summary_msg
        The template for summary messages, using the variables in *Template Variables for Summary
        Messages* below. The default is `None`, which uses a template with the highest threshold
        level reached, the number of steps, the number of failing steps and the time.
    debug
        Whether to print each message, and Slack's response, when it's sent. With
        `webhook_url=None`, the function prints previews of the templates instead (see above). The
        default is `False`.

    Returns
    -------
    Callable or None
        The action, to use in `Actions` or `FinalActions`. With `webhook_url=None` and
        `debug=True`, the function returns `None` after printing the previews.

    Template Variables for Step Messages
    ------------------------------------
    A `step_msg=` template can use these variables:

    - `"{step}"`: The step number.
    - `"{column}"`: The column name.
    - `"{value}"`: The value being compared (only available in certain validation steps).
    - `"{type}"`: The assertion type (e.g., `"col_vals_gt"`, etc.).
    - `"{level}"`: The threshold level (`"warning"`, `"error"`, or `"critical"`).
    - `"{level_num}"`: The threshold level as a number (`30`, `40`, or `50`).
    - `"{autobrief}"`: A localized and brief statement of the expectation for the step.
    - `"{failure_text}"`: Localized text that explains how the validation step failed.
    - `"{time}"`: The time of the notification.

    These are the keys of [`get_action_metadata()`](`pointblank.get_action_metadata`)'s
    dictionary, plus `"{time}"`.

    Template Variables for Summary Messages
    ---------------------------------------
    A `summary_msg=` template can use these variables:

    - `"{n_steps}"`: The total number of validation steps.
    - `"{n_passing_steps}"`: The number of validation steps where all test units passed.
    - `"{n_failing_steps}"`: The number of validation steps that had some failing test units.
    - `"{n_warning_steps}"`: The number of steps that exceeded a 'warning' threshold.
    - `"{n_error_steps}"`: The number of steps that exceeded an 'error' threshold.
    - `"{n_critical_steps}"`: The number of steps that exceeded a 'critical' threshold.
    - `"{all_passed}"`: Whether or not every validation step had no failing test units.
    - `"{highest_severity}"`: The highest threshold level reached in the validation: `"warning"`,
    `"error"` or `"critical"`. If no threshold was exceeded, it is `"some failing"` when some
    test units failed and `"all passed"` otherwise.
    - `"{tbl_row_count}"`: The number of rows in the target table.
    - `"{tbl_column_count}"`: The number of columns in the target table.
    - `"{tbl_name}"`: The name of the target table (`"Unknown"` without a `tbl_name=`).
    - `"{validation_duration}"`: The duration of the validation in seconds.
    - `"{time}"`: The time of the notification.

    These are keys of [`get_validation_summary()`](`pointblank.get_validation_summary`)'s
    dictionary, plus `"{time}"`. Its other keys, such as `"{overall_health_score}"` and
    `"{list_failing_steps}"`, can be used too. Since the templates are filled in with Python's
    `str.format()`, a literal brace has to be doubled (`"{{"` or `"}}"`).

    Examples
    --------
    Let's validate a small table of orders, which has a missing customer name and two amounts
    that aren't positive. The action has no webhook URL, so it prints the message instead of
    sending it. In `Actions(critical=)`, it runs for the steps that exceed the critical
    threshold:

    ```{python}
    import pointblank as pb
    import polars as pl

    orders = pl.DataFrame(
        {
            "amount": [29.99, 150.00, -5.00, 75.50, 0.00],
            "customer": ["Alice", "Bob", None, "Dana", "Eve"],
        }
    )

    notify_slack = pb.send_slack_notification()

    validation = (
        pb.Validate(
            data=orders,
            tbl_name="orders",
            thresholds=(0.05, 0.20, 0.40),
            actions=pb.Actions(critical=notify_slack),
        )
        .col_vals_not_null(columns="customer")
        .col_vals_gt(columns="amount", value=0)
        .interrogate()
    )
    ```

    Only the second step, with 40% of its test units failing, reached the critical threshold, so
    there is one message. The first step (20% failing) exceeded the error threshold, which has no
    action here.

    In `FinalActions`, the same action sends a summary of the validation instead:

    ```{python}
    validation = (
        pb.Validate(
            data=orders,
            tbl_name="orders",
            thresholds=(0.05, 0.20, 0.40),
            final_actions=pb.FinalActions(notify_slack),
        )
        .col_vals_not_null(columns="customer")
        .col_vals_gt(columns="amount", value=0)
        .interrogate()
    )
    ```

    The summary is sent once per interrogation, whether or not any step failed. Both steps failed
    here, and the highest threshold level reached was critical.

    Custom templates can add more detail. With `webhook_url=None` and `debug=True`, the function
    shows previews of the templates with sample values, without a validation:

    ```{python}
    pb.send_slack_notification(
        step_msg=(
            "🚨 *Step {step} ({type})* exceeded the {level} threshold\\n"
            "• Column: {column}\\n"
            "• Details: {failure_text}"
        ),
        summary_msg=(
            "📊 *Validation of {tbl_name}*: {highest_severity}\\n"
            "• Failing steps: {n_failing_steps} of {n_steps}\\n"
            "• Rows: {tbl_row_count}"
        ),
        debug=True,
    )
    ```

    The sample values show where each variable goes. For a real channel, we pass the webhook URL
    and use the action for both kinds of message. With `Actions(default=)`, a step message is sent
    for every step that exceeds a threshold, at its highest level:

    ```python
    import os

    notify_slack = pb.send_slack_notification(webhook_url=os.environ["SLACK_WEBHOOK_URL"])

    validation = (
        pb.Validate(
            data=orders,
            tbl_name="orders",
            thresholds=(0.05, 0.20, 0.40),
            actions=pb.Actions(default=notify_slack),
            final_actions=pb.FinalActions(notify_slack),
        )
        .col_vals_not_null(columns="customer")
        .col_vals_gt(columns="amount", value=0)
        .interrogate()
    )
    ```

    This interrogation posts three messages to the channel: one for each failing step (at the
    error and critical levels), then the summary. Reading the URL from the `SLACK_WEBHOOK_URL`
    environment variable keeps it out of the code.

    See Also
    --------
    Actions : Run actions when a step exceeds a threshold level.
    FinalActions : Run actions after an interrogation.
    get_action_metadata : Get the information that step messages are made from.
    get_validation_summary : Get the information that summary messages are made from.
    emit_otel : Send validation metrics to OpenTelemetry from a final action.
    """
    # Default templates
    default_step = """*Data Validation Alert*
• Type: {type}
• Level: {level}
• Step: {step}
• Column: {column}
• Time: {time}"""

    default_summary = """*Data Validation Summary*
• Highest Severity: {highest_severity}
• Total Steps: {n_steps}
• Failed Steps: {n_failing_steps}
• Time: {time}"""

    # Use provided templates or defaults
    step = step_msg or default_step
    summary = summary_msg or default_summary

    # If called directly (not as a callback), show template preview
    if webhook_url is None and debug:
        # Sample data for step preview with all possible variables
        step_data = {
            "step": 1,
            "column": "column_name",
            "value": 100,
            "type": "col_vals_gt",
            "level": "critical",
            "level_num": 50,
            "autobrief": "Values in column_name must be greater than 100",
            "failure_text": "25% of values failed this test",
            "time": datetime.now(),
        }

        # Sample data for summary preview with all possible variables
        summary_data = {
            "highest_severity": "critical",
            "all_passed": False,
            "n_steps": 5,
            "n_passing_steps": 3,
            "n_failing_steps": 2,
            "n_warning_steps": 1,
            "n_error_steps": 0,
            "n_critical_steps": 1,
            "tbl_name": "example_table",
            "tbl_row_count": 1000,
            "tbl_column_count": 8,
            "validation_duration": 1.23,
            "time": datetime.now(),
        }

        print("\n=== Step Notification Preview ===")
        print(step.format(**step_data))

        print("\n=== Summary Notification Preview ===")
        print(summary.format(**summary_data))

        return None

    def notify() -> None:  # pragma: no cover
        try:
            message = None

            # Try to get summary data first
            try:
                from pointblank import get_validation_summary

                summary_data = get_validation_summary()
                if summary_data is not None:
                    summary_data["time"] = datetime.now()
                    message = summary.format(**summary_data)
            except ImportError:
                pass

            # If no summary, try step data
            if message is None:
                try:
                    from pointblank import get_action_metadata

                    metadata = get_action_metadata()
                    if metadata is not None:
                        metadata["time"] = datetime.now()
                        message = step.format(**metadata)
                except ImportError:
                    pass

            # If still no message, raise error
            if message is None:
                raise ValueError("No validation data available")

            if debug:
                print("\n=== Debug Info ===")
                print("Message Content:")
                print(message)
                print("==================\n")

            if webhook_url is None:
                print("\n=== Dry Run ===")
                print("Would send to Slack:")
                print(message)
                print("==============\n")
                return

            payload = {"text": message}
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()

            if debug:
                print("\n=== Slack Response ===")
                print(f"Status Code: {response.status_code}")
                print(f"Response: {response.text}")
                print("=====================\n")

        except Exception as e:
            error_msg = f"Failed to send Slack notification: {str(e)}"
            if debug:
                print("\n=== Error Details ===")
                print(error_msg)
                print("===================\n")
            else:
                print(error_msg)

    return notify  # pragma: no cover
