# emit_otel()

Create a final action that sends a validation's results to OpenTelemetry.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/integrations/otel.py#L665-L891)

``` python
emit_otel(
    *,
    meter_name="pointblank",
    meter_version=None,
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

The [emit_otel()](../reference/emit_otel.md#pointblank.emit_otel) function returns an action for [`FinalActions`](../reference/FinalActions.md#pointblank.FinalActions) that sends validation results to OpenTelemetry (OTel) at the end of every [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) call. With the action in the validation plan, every run reports to the monitoring backend without any code after the validation. That suits validations that run on a schedule, such as checks in a data pipeline.

The action sends the same signals as [`OTelExporter.export()`](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter.export) called right after the interrogation: metrics (on by default), trace spans (`enable_tracing=True`) and log records for the steps that exceeded a threshold (`enable_logging=True`), with the same names and attributes. See [`OTelExporter`](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter) for the list of signals. Since the signals are sent during [interrogate()](../reference/Validate.interrogate.md#pointblank.Validate.interrogate), the `pb.validate` span is a child of whatever span is current at that point, such as the span of a pipeline task.

Without `meter_provider=` and `tracer_provider=`, the metrics and traces go to OTel's global providers, which discard them until the application sets up the OTel SDK. Log records have no global provider, so logging needs `logger_provider=`. The OTel packages are an optional dependency, installed with `pip install pointblank[otel]`.

## Parameters

`meter_name: str = "pointblank"`  
The name that identifies Pointblank as the source of the signals (the OTel instrumentation scope), used for the meter, the tracer and the logger. The default is `"pointblank"`.

`meter_version: str | None = None`  
The version of the instrumentation scope. The default is `None`, which uses the installed version of Pointblank.

`enable_metrics: bool = True`  
Whether to send the metrics. The default is `True`.

`enable_tracing: bool = False`  
Whether to send the trace spans. The default is `False`.

`enable_logging: bool = False`  
Whether to send log records for the steps that exceeded a threshold. This needs `logger_provider=`. The default is `False`.

`meter_provider: Any | None = None`  
An OTel `MeterProvider` for the metrics. The default is `None`, which uses the global meter provider.

`tracer_provider: Any | None = None`  
An OTel `TracerProvider` for the trace spans. The default is `None`, which uses the global tracer provider.

`logger_provider: Any | None = None`  
An OTel `LoggerProvider` for the log records, needed when `enable_logging=True`. The default is `None`.

`metric_prefix: str = "pb.validation"`  
The prefix for the names of the metrics. The default is `"pb.validation"`.

`log_level: str = "warning"`  
The lowest threshold level that produces a log record: `"warning"` (every step that exceeded a threshold), `"error"` or `"critical"`. The default is `"warning"`.

`extra_attributes: dict[str, str] | None = None`  
A dictionary of attributes to add to every signal, such as the environment or the name of the pipeline. The default is `None`.

## Returns

`Callable`  
A function to use in [FinalActions](../reference/FinalActions.md#pointblank.FinalActions).

## Raises

`ImportError`  
If the OTel packages aren't installed.

`ValueError`  
If `enable_logging=True` is given without `logger_provider=`, or if `log_level=` isn't one of the accepted values. Both are raised by [emit_otel()](../reference/emit_otel.md#pointblank.emit_otel) itself, when the validation plan is built, rather than during [interrogate()](../reference/Validate.interrogate.md#pointblank.Validate.interrogate).

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
    for point in metric.data.data_points:
        value = point.count if hasattr(point, "count") else point.value
        print(metric.name, value)
```

    pb.validation.steps.total 2
    pb.validation.steps.passed 0
    pb.validation.steps.failed 2
    pb.validation.test_units.total 10
    pb.validation.test_units.passed 7
    pb.validation.test_units.failed 3
    pb.validation.pass_rate 0.7
    pb.validation.duration 0.009726
    pb.validation.threshold.warning 2
    pb.validation.threshold.error 2
    pb.validation.threshold.critical 1
    pb.validation.step.duration 1
    pb.validation.step.duration 1

The metrics were sent during [interrogate()](../reference/Validate.interrogate.md#pointblank.Validate.interrogate). Both steps failed, 7 of the 10 test units passed, and the second step exceeded all three thresholds. The step duration histogram has a data point for each step, and the value printed for it is its count of one observation. Each data point has the `pb.tbl_name` and `env` attributes:

``` python
dict(metrics[0].data.data_points[0].attributes)
```

    {'pb.tbl_name': 'orders', 'env': 'dev'}

To also send traces, give the action `enable_tracing=True` and a tracer provider. Here, an in-memory span exporter keeps the spans:

``` python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

span_exporter = InMemorySpanExporter()
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

validation = (
    pb.Validate(
        data=orders,
        tbl_name="orders",
        thresholds=(0.05, 0.20, 0.40),
        final_actions=pb.FinalActions(
            pb.emit_otel(
                enable_metrics=False, enable_tracing=True, tracer_provider=tracer_provider
            )
        ),
    )
    .col_vals_not_null(columns="customer")
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)

for span in span_exporter.get_finished_spans():
    print(span.name, span.attributes.get("pb.step.assertion_type"))
```

    pb.validate.step col_vals_not_null
    pb.validate.step col_vals_gt
    pb.validate None

There's a `pb.validate.step` span for each step, inside the `pb.validate` span of the whole interrogation, just as with [OTelExporter](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter).

Since [FinalActions](../reference/FinalActions.md#pointblank.FinalActions) takes any number of actions, the signals can be sent along with other notifications. Here, a [`send_slack_notification()`](../reference/send_slack_notification.md#pointblank.send_slack_notification) action without a webhook URL prints the message it would send:

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
    • Time: 2026-11-08 18:15:36.070632
    ==============

The actions run in the order they're given, so the metrics are sent before the Slack message. The message summarizes the same results as the metrics: both steps failed, and the highest threshold level reached was critical.

## See Also

[OTelExporter](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter)  
Send metrics, traces and log records after an interrogation.

[FinalActions](../reference/FinalActions.md#pointblank.FinalActions)  
Run actions after an interrogation.

[send_slack_notification()](../reference/send_slack_notification.md#pointblank.send_slack_notification)  
Send a Slack notification from a final action.
