## emit_otel()


Create an OTel export action for use in [FinalActions](FinalActions.md#pointblank.FinalActions).


Usage

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
    extra_attributes=None
)
```


Returns a callable that, when invoked as a FinalAction after interrogation, exports validation results as OpenTelemetry metrics.


## Parameters


`meter_name: str = ``"pointblank"`  
Name for the OTel Meter. Defaults to `"pointblank"`.

`enable_metrics: bool = ``True`  
Emit OTel metrics (counters, gauges, histograms). Default is `"True"`.

`enable_tracing: bool = ``False`  
Emit OTel trace spans. Default is `"False"`.

`enable_logging: bool = ``False`  
Emit OTel log records for threshold breaches. Default is `"False"`.

`meter_provider: Any | None = None`  
Custom MeterProvider. If `"None"`, uses the global default.

`tracer_provider: Any | None = None`  
Custom TracerProvider. If `"None"`, uses the global default.

`logger_provider: Any | None = None`  
Custom LoggerProvider. Required when `"enable_logging=True"`.

`metric_prefix: str = ``"pb.validation"`  
Prefix for all metric instrument names. Default is `"pb.validation"`.

`log_level: str = ``"warning"`  
Minimum severity level for log emission. One of `"warning"`, `"error"`, `"critical"`. The default is `"warning"`.

`extra_attributes: dict[str, str] | None = None`  
Additional key-value pairs attached to all emitted signals.


## Returns


`Callable`  
A callable suitable for use in [FinalActions](FinalActions.md#pointblank.FinalActions).


## Examples

``` python
validation = (
    pb.Validate(
        data=df,
        tbl_name="sales",
        final_actions=pb.FinalActions(
            pb.emit_otel(enable_metrics=True, enable_tracing=True),
        ),
    )
    .col_vals_not_null(columns="customer_id")
    .interrogate()
)
```
