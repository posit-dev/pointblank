# emit_otel()

Create a final action that sends a validation's metrics to OpenTelemetry.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/integrations/otel.py#L698-L875)

``` python
emit_otel(
    *,
    meter_name="pointblank",
    enable_metrics=True,
    enable_tracing=False,
    enable_logging=False,
    meter_provider=None,
    tracer_provider=None,
    logger_provider=None,
    metric_prefix="pb.validation",
    log_level="warning",
    extra_attributes=None,
)
```

The [emit_otel()](../reference/emit_otel.md#pointblank.emit_otel) function returns an action for [`FinalActions`](../reference/FinalActions.md#pointblank.FinalActions) that sends validation metrics to OpenTelemetry (OTel) at the end of every [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) call. With the action in the validation plan, every run reports to the monitoring backend without any code after the validation. That suits validations that run on a schedule, such as checks in a data pipeline.

The action sends the metrics of [`OTelExporter`](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter), with these differences:

- It sends metrics only. The `enable_tracing=`, `enable_logging=`, `tracer_provider=`, `logger_provider=` and `log_level=` arguments currently have no effect. Use [`OTelExporter.export()`](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter.export) after the interrogation for traces and log records.
- It doesn't send the step duration histogram (`{prefix}.step.duration`), so there are 11 metrics instead of 12.
- The only attributes are `pb.tbl_name` (when the validation has a `tbl_name=`) and those in `extra_attributes=`.

Without `meter_provider=`, the metrics go to OTel's global meter provider, which discards them until the application sets up the OTel SDK. The OTel packages are an optional dependency, installed with `pip install pointblank[otel]`.

## Parameters

`meter_name: str = "pointblank"`  
The name that identifies Pointblank as the source of the metrics (the OTel instrumentation scope). The default is `"pointblank"`.

`enable_metrics: bool = True`  
Whether to send the metrics. The default is `True`.

`enable_tracing: bool = False`  
Currently has no effect. The default is `False`.

`enable_logging: bool = False`  
Currently has no effect. The default is `False`.

`meter_provider: Any | None = None`  
An OTel `MeterProvider` for the metrics. The default is `None`, which uses the global meter provider.

`tracer_provider: Any | None = None`  
Currently has no effect. The default is `None`.

`logger_provider: Any | None = None`  
Currently has no effect. The default is `None`.

`metric_prefix: str = "pb.validation"`  
The prefix for the names of the metrics. The default is `"pb.validation"`.

`log_level: str = "warning"`  
Currently has no effect. The default is `"warning"`.

`extra_attributes: dict[str, str] | None = None`  
A dictionary of attributes to add to every metric, such as the environment or the name of the pipeline. The default is `None`.

## Returns

`Callable`  
A function to use in [FinalActions](../reference/FinalActions.md#pointblank.FinalActions).

## Raises

`ImportError`  
If the OTel packages aren't installed.

## Examples

Let's add the action to a validation of a small table of orders. The OTel SDK's `InMemoryMetricReader` keeps the metrics in memory so we can look at them (in production, the meter provider would send them to a collector instead):

``` python
import pointblank as pb
import polars as pl
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

reader = InMemoryMetricReader()
meter_provider = MeterProvider(metric_readers=[reader])

orders = pl.DataFrame(
    {
        "amount": [29.99, 150.00, -5.00, 75.50, 0.00],
        "customer": ["Alice", "Bob", None, "Dana", "Eve"],
    }
)

validation = (
    pb.Validate(
        data=orders,
        tbl_name="orders",
        thresholds=(0.05, 0.20, 0.40),
        final_actions=pb.FinalActions(
            pb.emit_otel(meter_provider=meter_provider, extra_attributes={"env": "dev"})
        ),
    )
    .col_vals_not_null(columns="customer")
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)

metrics = reader.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics

for metric in metrics:
    print(metric.name, metric.data.data_points[0].value)
```

    pb.validation.steps.total 2
    pb.validation.steps.passed 0
    pb.validation.steps.failed 2
    pb.validation.test_units.total 10
    pb.validation.test_units.passed 7
    pb.validation.test_units.failed 3
    pb.validation.pass_rate 0.7
    pb.validation.duration 0.005109
    pb.validation.threshold.warning 2
    pb.validation.threshold.error 2
    pb.validation.threshold.critical 1

The metrics were sent during [interrogate()](../reference/Validate.interrogate.md#pointblank.Validate.interrogate). Both steps failed, 7 of the 10 test units passed, and the second step exceeded all three thresholds. Each metric has the `pb.tbl_name` and `env` attributes:

``` python
dict(metrics[0].data.data_points[0].attributes)
```

    {'pb.tbl_name': 'orders', 'env': 'dev'}

Since [FinalActions](../reference/FinalActions.md#pointblank.FinalActions) takes any number of actions, the metrics can be sent along with other notifications. Here, a [`send_slack_notification()`](../reference/send_slack_notification.md#pointblank.send_slack_notification) action without a webhook URL prints the message it would send:

``` python
validation = (
    pb.Validate(
        data=orders,
        tbl_name="orders",
        thresholds=(0.05, 0.20, 0.40),
        final_actions=pb.FinalActions(
            pb.emit_otel(meter_provider=meter_provider),
            pb.send_slack_notification(),
        ),
    )
    .col_vals_not_null(columns="customer")
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)
```

    === Dry Run ===
    Would send to Slack:
    *Data Validation Summary*
    • Highest Severity: critical
    • Total Steps: 2
    • Failed Steps: 2
    • Time: 2026-10-29 12:56:14.594144
    ==============

The actions run in the order they're given, so the metrics are sent before the Slack message. The message summarizes the same results as the metrics: both steps failed, and the highest threshold level reached was critical.

## See Also

[OTelExporter](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter)  
Send metrics, traces and log records after an interrogation.

[FinalActions](../reference/FinalActions.md#pointblank.FinalActions)  
Run actions after an interrogation.

[send_slack_notification()](../reference/send_slack_notification.md#pointblank.send_slack_notification)  
Send a Slack notification from a final action.
