## integrations.otel.OTelExporter


Export Pointblank validation results as OpenTelemetry signals.


Usage

``` python
integrations.otel.OTelExporter()
```


## Parameters


`meter_name: str = ``"pointblank"`  
Name for the OTel Meter. Defaults to `"pointblank"`.

`meter_version: str | None = None`  
Version string for the Meter. Defaults to `"pointblank.__version__"`.

`enable_metrics: bool = ``True`  
Emit OTel metrics (counters, gauges, histograms). Default is `"True"`.

`enable_tracing: bool = ``False`  
Emit OTel trace spans. Default is `"False"`.

`enable_logging: bool = ``False`  
Emit OTel log records for threshold breaches. Default is `"False"`.

`meter_provider: MeterProvider | None = None`  
Custom MeterProvider. If `"None"`, uses the global default.

`tracer_provider: TracerProvider | None = None`  
Custom TracerProvider. If `"None"`, uses the global default.

`logger_provider: Any | None = None`  
Custom LoggerProvider. Required when `"enable_logging=True"`.

`metric_prefix: str = ``"pb.validation"`  
Prefix for all metric instrument names. Default is `"pb.validation"`.

`log_level: str = ``"warning"`  
Minimum severity level for log emission. One of `"warning"`, `"error"`, `"critical"`. Default is `"warning"`.

`extra_attributes: dict[str, str] | None = None`  
Additional key-value pairs attached to all emitted signals.


## Methods

| Name | Description |
|----|----|
| [export()](#export) | Export validation results as OTel signals. |

------------------------------------------------------------------------


#### export()


Export validation results as OTel signals.


Usage

``` python
export(validation)
```


##### Parameters


`validation: Validate`  
A [Validate](Validate.md#pointblank.Validate) object that has been interrogated. If not yet interrogated, raises `ValueError`.
