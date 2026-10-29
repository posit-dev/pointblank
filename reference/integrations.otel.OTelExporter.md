# integrations.otel.OTelExporter

Export Pointblank validation results as OpenTelemetry signals.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/integrations/otel.py#L30-L695)

``` python
integrations.otel.OTelExporter(
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

OpenTelemetry (OTel) is a vendor-neutral standard for application telemetry, which backends such as Grafana, Datadog, New Relic and Prometheus can receive. An [OTelExporter](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter) sends the results of an interrogated [`Validate`](../reference/Validate.md#pointblank.Validate) object to OTel, so that data quality shows up in the same dashboards and alerts as the rest of a pipeline's telemetry. Call [`export()`](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter.export) after [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) to send the results. For an export that happens at the end of every interrogation, use [`emit_otel()`](../reference/emit_otel.md#pointblank.emit_otel) in [`FinalActions`](../reference/FinalActions.md#pointblank.FinalActions) instead (it sends metrics only).

An export can produce three kinds of signal:

- **Metrics** (on by default). There are counters of the steps (`{prefix}.steps.total`, `{prefix}.steps.passed` and `{prefix}.steps.failed`), of the test units (`{prefix}.test_units.total`, `{prefix}.test_units.passed` and `{prefix}.test_units.failed`) and of the steps that exceeded each threshold level (`{prefix}.threshold.warning`, `{prefix}.threshold.error` and `{prefix}.threshold.critical`). There are gauges of the overall pass rate (`{prefix}.pass_rate`) and of the interrogation time in seconds (`{prefix}.duration`), and a histogram of the step processing times (`{prefix}.step.duration`). Here `{prefix}` is the `metric_prefix=` value. Counters add up over exports, while a gauge keeps the latest value.
- **Traces** (`enable_tracing=True`). A `pb.validate` span covers the interrogation, with a `pb.validate.step` child span for each step. A step's span has its test unit counts and fractions and the threshold levels it exceeded. The `pb.validate` span also has a `threshold_breach` event for each threshold level that each step exceeded.
- **Logs** (`enable_logging=True`). There is one log record for each step that exceeded a threshold, at the highest level it exceeded (`WARNING`, `ERROR` or `CRITICAL`). The `log_level=` value sets the lowest level that is logged.

Every signal has the attributes `pb.tbl_name`, `pb.label`, `pb.owner` and `pb.version` (for the ones that are set in the [Validate](../reference/Validate.md#pointblank.Validate) object), plus those in `extra_attributes=`. The signals for individual steps also have `pb.step.i`, `pb.step.assertion_type` and `pb.step.column`.

The signals go to the providers given as `meter_provider=`, `tracer_provider=` and `logger_provider=`. Without them, metrics and traces go to OTel's global providers, which discard everything until the application sets up the OTel SDK (for example, with an OTLP exporter that sends to a collector). Log records have no global provider to fall back on, so logging needs `logger_provider=`.

The OTel packages are an optional dependency, installed with `pip install pointblank[otel]`.

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

`meter_provider: MeterProvider | None = None`  
An OTel `MeterProvider` for the metrics. The default is `None`, which uses the global meter provider.

`tracer_provider: TracerProvider | None = None`  
An OTel `TracerProvider` for the trace spans. The default is `None`, which uses the global tracer provider.

`logger_provider: Any | None = None`  
An OTel `LoggerProvider` for the log records, needed when `enable_logging=True`. The default is `None`.

`metric_prefix: str = "pb.validation"`  
The prefix for the names of the metrics. The default is `"pb.validation"`.

`log_level: str = "warning"`  
The lowest threshold level that produces a log record: `"warning"` (every step that exceeded a threshold), `"error"` or `"critical"`. The default is `"warning"`.

`extra_attributes: dict[str, str] | None = None`  
A dictionary of attributes to add to every signal, such as the environment or the name of the pipeline. The default is `None`.

## Raises

`ImportError`  
If the OTel packages aren't installed.

## Examples

Let's validate a small table of orders, which has a missing customer name and two amounts that aren't positive:

``` python
import pointblank as pb
import polars as pl

orders = pl.DataFrame(
    {
        "order_id": [101, 102, 103, 104, 105],
        "amount": [29.99, 150.00, -5.00, 75.50, 0.00],
        "customer": ["Alice", "Bob", None, "Dana", "Eve"],
    }
)

validation = (
    pb.Validate(data=orders, tbl_name="orders", thresholds=(0.05, 0.20, 0.40))
    .col_vals_not_null(columns="customer")
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)
```

In production, the exporter would use providers that send to a collector. Here, the OTel SDK's `InMemoryMetricReader` keeps the metrics in memory so we can look at them:

``` python
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from pointblank.integrations.otel import OTelExporter

reader = InMemoryMetricReader()
meter_provider = MeterProvider(metric_readers=[reader])

exporter = OTelExporter(meter_provider=meter_provider, extra_attributes={"env": "dev"})
exporter.export(validation)

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
    pb.validation.duration 0.004554
    pb.validation.threshold.warning 2
    pb.validation.threshold.error 2
    pb.validation.threshold.critical 1
    pb.validation.step.duration 1
    pb.validation.step.duration 1

Both steps failed, and 7 of the 10 test units passed, for a pass rate of `0.7`. The first step (20% failing) exceeded the warning and error thresholds, and the second (40% failing) exceeded all three. The step duration histogram has a data point for each step, and the value printed for it is its count of one observation. Every data point has the `pb.tbl_name` and `env` attributes (and the histogram's also have the `pb.step.*` ones):

``` python
dict(metrics[0].data.data_points[0].attributes)
```

    {'pb.tbl_name': 'orders', 'env': 'dev'}

For traces, we give the exporter a `TracerProvider` with an in-memory span exporter. This exporter sends traces only:

``` python
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

span_exporter = InMemorySpanExporter()
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))

OTelExporter(
    enable_metrics=False, enable_tracing=True, tracer_provider=tracer_provider
).export(validation)

for span in span_exporter.get_finished_spans():
    attrs = span.attributes
    print(span.name, attrs.get("pb.step.assertion_type"), attrs.get("pb.step.n_failed"))
```

    pb.validate.step col_vals_not_null 1
    pb.validate.step col_vals_gt 2
    pb.validate None None

The two `pb.validate.step` spans (one per step, with its number of failing test units) end before the `pb.validate` span that contains them, so they're listed first. The `pb.validate` span has five `threshold_breach` events: two for the first step and three for the second. The [`export()`](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter.export) method has an example of sending log records.

## See Also

[emit_otel()](../reference/emit_otel.md#pointblank.emit_otel)  
Send the metrics automatically at the end of an interrogation.

[FinalActions](../reference/FinalActions.md#pointblank.FinalActions)  
Run actions after an interrogation.

## Methods

| Name | Description |
|----|----|
| [export()](#export) | Send the results of an interrogated validation as OTel signals. |

### export()

Send the results of an interrogated validation as OTel signals.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/integrations/otel.py#L241-L364)

``` python
export(validation)
```

The [export()](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter.export) method sends the signals that the exporter has enabled (metrics, traces and log records, as described in [`OTelExporter`](../reference/integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter)) for one interrogated [`Validate`](../reference/Validate.md#pointblank.Validate) object. The same exporter can export any number of validations, such as one for each table in a pipeline. The counters then add up over the validations, and the `pb.tbl_name` attribute keeps the tables apart. Exporting the same validation twice counts it twice.

#### Parameters

`validation: Validate`  
An interrogated [Validate](../reference/Validate.md#pointblank.Validate) object.

#### Returns

`None`  
The signals are sent to the providers.

#### Raises

`ValueError`  
If `validation` hasn't been interrogated, or if logging is enabled and there is no `logger_provider=`. In the second case, the metrics and traces have already been sent when the error is raised.

`ImportError`  
If the OTel packages aren't installed.

#### Examples

Let's send log records for the steps of a validation that exceeded a threshold. The OTel SDK's `InMemoryLogRecordExporter` keeps the records in memory so we can look at them:

``` python
import pointblank as pb
import polars as pl
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    InMemoryLogRecordExporter,
    SimpleLogRecordProcessor,
)
from pointblank.integrations.otel import OTelExporter

log_exporter = InMemoryLogRecordExporter()
logger_provider = LoggerProvider()
logger_provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))

orders = pl.DataFrame(
    {
        "amount": [29.99, 150.00, -5.00, 75.50, 0.00],
        "customer": ["Alice", "Bob", None, "Dana", "Eve"],
    }
)

validation = (
    pb.Validate(data=orders, tbl_name="orders", thresholds=(0.05, 0.20, 0.40))
    .col_vals_not_null(columns="customer")
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)

exporter = OTelExporter(
    enable_metrics=False, enable_logging=True, logger_provider=logger_provider
)
exporter.export(validation)

for log in log_exporter.get_finished_logs():
    print(log.log_record.severity_text, log.log_record.body)
```

    ERROR Validation step 1 (col_vals_not_null on 'customer') exceeded error threshold: 20.00% failed
    CRITICAL Validation step 2 (col_vals_gt on 'amount') exceeded critical threshold: 40.00% failed

Each step has one record, at the highest threshold level it exceeded. With `log_level="critical"`, only the second step's record would be sent.

A pipeline that checks several tables can use one exporter for all of them. Here, we export a second validation, of a table of customers, with a metrics-only exporter that also gets the first validation:

``` python
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

customers = pl.DataFrame({"customer_id": [1, 2, 3]})

validation_2 = (
    pb.Validate(data=customers, tbl_name="customers")
    .col_vals_not_null(columns="customer_id")
    .col_vals_gt(columns="customer_id", value=0)
    .col_vals_lt(columns="customer_id", value=1000)
    .interrogate()
)

reader = InMemoryMetricReader()
exporter = OTelExporter(meter_provider=MeterProvider(metric_readers=[reader]))

exporter.export(validation)
exporter.export(validation_2)

metrics = reader.get_metrics_data().resource_metrics[0].scope_metrics[0].metrics
steps_total = next(m for m in metrics if m.name == "pb.validation.steps.total")

for point in steps_total.data.data_points:
    print(point.attributes["pb.tbl_name"], point.value)
```

    orders 2
    customers 3

The `pb.validation.steps.total` counter has a data point for each table: 2 steps for `orders` and 3 for `customers`. A dashboard can show them separately, or add them up for a total over the pipeline.
