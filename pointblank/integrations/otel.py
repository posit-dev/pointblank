from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider

    from pointblank.validate import Validate

try:
    from opentelemetry import metrics as otel_metrics
    from opentelemetry import trace as otel_trace

    HAS_OTEL = True
except ImportError:
    HAS_OTEL = False


def _check_otel_installed() -> None:
    """Raise ImportError with install hint if OTel packages are missing."""
    if not HAS_OTEL:
        raise ImportError(
            "OpenTelemetry packages are required for OTel integration. "
            "Install them with: pip install pointblank[otel]"
        )


class OTelExporter:
    """Export Pointblank validation results as OpenTelemetry signals.

    OpenTelemetry (OTel) is a vendor-neutral standard for application telemetry, which backends
    such as Grafana, Datadog, New Relic and Prometheus can receive. An `OTelExporter` sends the
    results of an interrogated [`Validate`](`pointblank.Validate`) object to OTel, so that data
    quality shows up in the same dashboards and alerts as the rest of a pipeline's telemetry. Call
    [`export()`](`pointblank.integrations.otel.OTelExporter.export`) after
    [`interrogate()`](`pointblank.Validate.interrogate`) to send the results. For an export that
    happens at the end of every interrogation, use [`emit_otel()`](`pointblank.emit_otel`) in
    [`FinalActions`](`pointblank.FinalActions`) instead (it sends metrics only).

    An export can produce three kinds of signal:

    - **Metrics** (on by default). There are counters of the steps (`{prefix}.steps.total`,
    `{prefix}.steps.passed` and `{prefix}.steps.failed`), of the test units
    (`{prefix}.test_units.total`, `{prefix}.test_units.passed` and `{prefix}.test_units.failed`)
    and of the steps that exceeded each threshold level (`{prefix}.threshold.warning`,
    `{prefix}.threshold.error` and `{prefix}.threshold.critical`). There are gauges of the overall
    pass rate (`{prefix}.pass_rate`) and of the interrogation time in seconds (`{prefix}.duration`),
    and a histogram of the step processing times (`{prefix}.step.duration`). Here `{prefix}` is the
    `metric_prefix=` value. Counters add up over exports, while a gauge keeps the latest value.
    - **Traces** (`enable_tracing=True`). A `pb.validate` span covers the interrogation, with a
    `pb.validate.step` child span for each step. A step's span has its test unit counts and
    fractions and the threshold levels it exceeded. The `pb.validate` span also has a
    `threshold_breach` event for each threshold level that each step exceeded.
    - **Logs** (`enable_logging=True`). There is one log record for each step that exceeded a
    threshold, at the highest level it exceeded (`WARNING`, `ERROR` or `CRITICAL`). The
    `log_level=` value sets the lowest level that is logged.

    Every signal has the attributes `pb.tbl_name`, `pb.label`, `pb.owner` and `pb.version` (for the
    ones that are set in the `Validate` object), plus those in `extra_attributes=`. The signals for
    individual steps also have `pb.step.i`, `pb.step.assertion_type` and `pb.step.column`.

    The signals go to the providers given as `meter_provider=`, `tracer_provider=` and
    `logger_provider=`. Without them, metrics and traces go to OTel's global providers, which
    discard everything until the application sets up the OTel SDK (for example, with an OTLP
    exporter that sends to a collector). Log records have no global provider to fall back on, so
    logging needs `logger_provider=`.

    The OTel packages are an optional dependency, installed with `pip install pointblank[otel]`.

    Parameters
    ----------
    meter_name
        The name that identifies Pointblank as the source of the signals (the OTel instrumentation
        scope), used for the meter, the tracer and the logger. The default is `"pointblank"`.
    meter_version
        The version of the instrumentation scope. The default is `None`, which uses the installed
        version of Pointblank.
    enable_metrics
        Whether to send the metrics. The default is `True`.
    enable_tracing
        Whether to send the trace spans. The default is `False`.
    enable_logging
        Whether to send log records for the steps that exceeded a threshold. This needs
        `logger_provider=`. The default is `False`.
    meter_provider
        An OTel `MeterProvider` for the metrics. The default is `None`, which uses the global meter
        provider.
    tracer_provider
        An OTel `TracerProvider` for the trace spans. The default is `None`, which uses the global
        tracer provider.
    logger_provider
        An OTel `LoggerProvider` for the log records, needed when `enable_logging=True`. The
        default is `None`.
    metric_prefix
        The prefix for the names of the metrics. The default is `"pb.validation"`.
    log_level
        The lowest threshold level that produces a log record: `"warning"` (every step that
        exceeded a threshold), `"error"` or `"critical"`. The default is `"warning"`.
    extra_attributes
        A dictionary of attributes to add to every signal, such as the environment or the name of
        the pipeline. The default is `None`.

    Raises
    ------
    ImportError
        If the OTel packages aren't installed.

    Examples
    --------
    Let's validate a small table of orders, which has a missing customer name and two amounts
    that aren't positive:

    ```{python}
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

    In production, the exporter would use providers that send to a collector. Here, the OTel SDK's
    `InMemoryMetricReader` keeps the metrics in memory so we can look at them:

    ```{python}
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

    Both steps failed, and 7 of the 10 test units passed, for a pass rate of `0.7`. The first step
    (20% failing) exceeded the warning and error thresholds, and the second (40% failing) exceeded
    all three. The step duration histogram has a data point for each step, and the value printed
    for it is its count of one observation. Every data point has the `pb.tbl_name` and `env`
    attributes (and the histogram's also have the `pb.step.*` ones):

    ```{python}
    dict(metrics[0].data.data_points[0].attributes)
    ```

    For traces, we give the exporter a `TracerProvider` with an in-memory span exporter. This
    exporter sends traces only:

    ```{python}
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

    The two `pb.validate.step` spans (one per step, with its number of failing test units) end
    before the `pb.validate` span that contains them, so they're listed first. The `pb.validate`
    span has five `threshold_breach` events: two for the first step and three for the second. The
    [`export()`](`pointblank.integrations.otel.OTelExporter.export`) method has an example of
    sending log records.

    See Also
    --------
    emit_otel : Send the metrics automatically at the end of an interrogation.
    FinalActions : Run actions after an interrogation.
    """

    def __init__(
        self,
        *,
        meter_name: str = "pointblank",
        meter_version: str | None = None,
        enable_metrics: bool = True,
        enable_tracing: bool = False,
        enable_logging: bool = False,
        meter_provider: MeterProvider | None = None,
        tracer_provider: TracerProvider | None = None,
        logger_provider: Any | None = None,
        metric_prefix: str = "pb.validation",
        log_level: str = "warning",
        extra_attributes: dict[str, str] | None = None,
    ) -> None:
        _check_otel_installed()

        self.meter_name = meter_name
        self.meter_version = meter_version or self._get_pointblank_version()
        self.enable_metrics = enable_metrics
        self.enable_tracing = enable_tracing
        self.enable_logging = enable_logging
        self.meter_provider = meter_provider
        self.tracer_provider = tracer_provider
        self.logger_provider = logger_provider
        self.metric_prefix = metric_prefix
        self.log_level = log_level
        self.extra_attributes = extra_attributes or {}
        self._instruments_initialized = False

    @staticmethod
    def _get_pointblank_version() -> str:
        try:
            import pointblank

            return pointblank.__version__
        except (ImportError, AttributeError):
            return "0.0.0"

    # ── Public API ──────────────────────────────────────────────────

    def export(self, validation: Validate) -> None:
        """Send the results of an interrogated validation as OTel signals.

        The `export()` method sends the signals that the exporter has enabled (metrics, traces and
        log records, as described in [`OTelExporter`](`pointblank.integrations.otel.OTelExporter`))
        for one interrogated [`Validate`](`pointblank.Validate`) object. The same exporter can
        export any number of validations, such as one for each table in a pipeline. The counters
        then add up over the validations, and the `pb.tbl_name` attribute keeps the tables apart.
        Exporting the same validation twice counts it twice.

        Parameters
        ----------
        validation
            An interrogated `Validate` object.

        Returns
        -------
        None
            The signals are sent to the providers.

        Raises
        ------
        ValueError
            If `validation` hasn't been interrogated, or if logging is enabled and there is no
            `logger_provider=`. In the second case, the metrics and traces have already been sent
            when the error is raised.
        ImportError
            If the OTel packages aren't installed.

        Examples
        --------
        Let's send log records for the steps of a validation that exceeded a threshold. The OTel
        SDK's `InMemoryLogRecordExporter` keeps the records in memory so we can look at them:

        ```{python}
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

        Each step has one record, at the highest threshold level it exceeded. With
        `log_level="critical"`, only the second step's record would be sent.

        A pipeline that checks several tables can use one exporter for all of them. Here, we export
        a second validation, of a table of customers, with a metrics-only exporter that also gets
        the first validation:

        ```{python}
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

        The `pb.validation.steps.total` counter has a data point for each table: 2 steps for
        `orders` and 3 for `customers`. A dashboard can show them separately, or add them up for a
        total over the pipeline.
        """
        _check_otel_installed()

        if validation.time_start is None:
            raise ValueError(
                "Validate object must be interrogated before exporting. Call .interrogate() first."
            )

        attrs = self._build_common_attributes(validation)

        if self.enable_metrics:
            self._emit_metrics(validation, attrs)
        if self.enable_tracing:
            self._emit_traces(validation, attrs)
        if self.enable_logging:
            self._emit_logs(validation, attrs)

    def _export_from_summary(self, summary: dict) -> None:
        """Export validation metrics from a FinalActions summary dict."""
        attrs = self._build_common_attributes_from_summary(summary)

        if self.enable_metrics:
            self._emit_metrics_from_summary(summary, attrs)

    # ── Attribute Builders ──────────────────────────────────────────

    def _build_common_attributes(self, validation: Validate) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        if validation.tbl_name is not None:
            attrs["pb.tbl_name"] = validation.tbl_name
        if validation.label is not None:
            attrs["pb.label"] = validation.label
        if getattr(validation, "owner", None) is not None:
            attrs["pb.owner"] = validation.owner
        if getattr(validation, "version", None) is not None:
            attrs["pb.version"] = validation.version
        attrs.update(self.extra_attributes)
        return attrs

    def _build_common_attributes_from_summary(self, summary: dict) -> dict[str, Any]:
        attrs: dict[str, Any] = {}
        tbl_name = summary.get("tbl_name")
        if tbl_name is not None and tbl_name != "Unknown":
            attrs["pb.tbl_name"] = tbl_name
        attrs.update(self.extra_attributes)
        return attrs

    # ── Provider / Instrument Access ────────────────────────────────

    def _get_meter(self):
        provider = self.meter_provider or otel_metrics.get_meter_provider()
        return provider.get_meter(self.meter_name, self.meter_version)

    def _get_tracer(self):
        provider = self.tracer_provider or otel_trace.get_tracer_provider()
        return provider.get_tracer(self.meter_name, self.meter_version)

    def _get_logger(self):
        if self.logger_provider is not None:
            return self.logger_provider.get_logger(self.meter_name, self.meter_version)
        raise ValueError("logger_provider must be set to emit OTel log records.")

    def _ensure_instruments(self, meter) -> None:
        if self._instruments_initialized:
            return

        p = self.metric_prefix

        self._counter_steps_total = meter.create_counter(
            f"{p}.steps.total",
            unit="{step}",
            description="Total validation steps executed",
        )
        self._counter_steps_passed = meter.create_counter(
            f"{p}.steps.passed",
            unit="{step}",
            description="Validation steps fully passing",
        )
        self._counter_steps_failed = meter.create_counter(
            f"{p}.steps.failed",
            unit="{step}",
            description="Validation steps with failures",
        )
        self._counter_units_total = meter.create_counter(
            f"{p}.test_units.total",
            unit="{unit}",
            description="Total test units evaluated",
        )
        self._counter_units_passed = meter.create_counter(
            f"{p}.test_units.passed",
            unit="{unit}",
            description="Test units that passed",
        )
        self._counter_units_failed = meter.create_counter(
            f"{p}.test_units.failed",
            unit="{unit}",
            description="Test units that failed",
        )
        self._gauge_pass_rate = meter.create_gauge(
            f"{p}.pass_rate",
            unit="1",
            description="Overall pass fraction across all steps",
        )
        self._histogram_step_duration = meter.create_histogram(
            f"{p}.step.duration",
            unit="s",
            description="Validation step processing duration",
        )
        self._gauge_duration = meter.create_gauge(
            f"{p}.duration",
            unit="s",
            description="Total wall-clock interrogation time",
        )
        self._counter_threshold_warning = meter.create_counter(
            f"{p}.threshold.warning",
            unit="{step}",
            description="Steps exceeding warning threshold",
        )
        self._counter_threshold_error = meter.create_counter(
            f"{p}.threshold.error",
            unit="{step}",
            description="Steps exceeding error threshold",
        )
        self._counter_threshold_critical = meter.create_counter(
            f"{p}.threshold.critical",
            unit="{step}",
            description="Steps exceeding critical threshold",
        )

        self._instruments_initialized = True

    # ── Metrics Emission ────────────────────────────────────────────

    def _emit_metrics(self, validation: Validate, attrs: dict) -> None:
        meter = self._get_meter()
        self._ensure_instruments(meter)

        vi_list = validation.validation_info

        steps_total = len(vi_list)
        steps_passed = sum(1 for v in vi_list if v.all_passed)
        steps_failed = steps_total - steps_passed
        units_passed = sum(v.n_passed or 0 for v in vi_list)
        units_failed = sum(v.n_failed or 0 for v in vi_list)
        units_total = units_passed + units_failed
        threshold_warning = sum(1 for v in vi_list if v.warning)
        threshold_error = sum(1 for v in vi_list if v.error)
        threshold_critical = sum(1 for v in vi_list if v.critical)

        pass_rate = units_passed / units_total if units_total > 0 else 1.0

        assert validation.time_start is not None and validation.time_end is not None
        duration = (validation.time_end - validation.time_start).total_seconds()

        self._counter_steps_total.add(steps_total, attrs)
        self._counter_steps_passed.add(steps_passed, attrs)
        self._counter_steps_failed.add(steps_failed, attrs)
        self._counter_units_total.add(units_total, attrs)
        self._counter_units_passed.add(units_passed, attrs)
        self._counter_units_failed.add(units_failed, attrs)
        self._gauge_pass_rate.set(pass_rate, attrs)
        self._gauge_duration.set(duration, attrs)
        self._counter_threshold_warning.add(threshold_warning, attrs)
        self._counter_threshold_error.add(threshold_error, attrs)
        self._counter_threshold_critical.add(threshold_critical, attrs)

        # Per-step histogram
        for vi in vi_list:
            step_attrs = {
                **attrs,
                "pb.step.i": vi.i,
                "pb.step.assertion_type": vi.assertion_type or "",
                "pb.step.column": str(vi.column) if vi.column else "",
            }
            if vi.proc_duration_s is not None:
                self._histogram_step_duration.record(vi.proc_duration_s, step_attrs)

    def _emit_metrics_from_summary(self, summary: dict, attrs: dict) -> None:
        meter = self._get_meter()
        self._ensure_instruments(meter)

        n_steps = summary.get("n_steps", 0)
        n_passing = summary.get("n_passing_steps", 0)
        n_failing = summary.get("n_failing_steps", 0)

        dict_n_passed = summary.get("dict_n_passed", {})
        dict_n_failed = summary.get("dict_n_failed", {})

        units_passed = sum(v or 0 for v in dict_n_passed.values())
        units_failed = sum(v or 0 for v in dict_n_failed.values())
        units_total = units_passed + units_failed

        pass_rate = units_passed / units_total if units_total > 0 else 1.0
        duration = summary.get("validation_duration", 0.0)

        n_warning = summary.get("n_warning_steps", 0)
        n_error = summary.get("n_error_steps", 0)
        n_critical = summary.get("n_critical_steps", 0)

        self._counter_steps_total.add(n_steps, attrs)
        self._counter_steps_passed.add(n_passing, attrs)
        self._counter_steps_failed.add(n_failing, attrs)
        self._counter_units_total.add(units_total, attrs)
        self._counter_units_passed.add(units_passed, attrs)
        self._counter_units_failed.add(units_failed, attrs)
        self._gauge_pass_rate.set(pass_rate, attrs)
        self._gauge_duration.set(duration, attrs)
        self._counter_threshold_warning.add(n_warning, attrs)
        self._counter_threshold_error.add(n_error, attrs)
        self._counter_threshold_critical.add(n_critical, attrs)

    # ── Trace Emission ──────────────────────────────────────────────

    def _emit_traces(self, validation: Validate, attrs: dict) -> None:
        tracer = self._get_tracer()

        assert validation.time_start is not None and validation.time_end is not None
        start_ns = int(validation.time_start.timestamp() * 1e9)
        end_ns = int(validation.time_end.timestamp() * 1e9)

        root_attrs = {
            **attrs,
            "pb.steps.total": len(validation.validation_info),
            "pb.all_passed": all(v.all_passed for v in validation.validation_info),
            "pb.duration_s": (validation.time_end - validation.time_start).total_seconds(),
        }

        root_span = tracer.start_span(
            "pb.validate",
            attributes=root_attrs,
            start_time=start_ns,
        )
        root_context = otel_trace.set_span_in_context(root_span)

        # Create child spans for each validation step
        for vi in validation.validation_info:
            child_attrs: dict[str, Any] = {
                **attrs,
                "pb.step.i": vi.i or 0,
                "pb.step.assertion_type": vi.assertion_type or "",
                "pb.step.column": str(vi.column) if vi.column else "",
                "pb.step.n": vi.n or 0,
                "pb.step.n_passed": vi.n_passed or 0,
                "pb.step.n_failed": vi.n_failed or 0,
                "pb.step.f_passed": vi.f_passed or 0.0,
                "pb.step.f_failed": vi.f_failed or 0.0,
                "pb.step.all_passed": bool(vi.all_passed),
            }

            if vi.warning:
                child_attrs["pb.step.warning"] = True
            if vi.error:
                child_attrs["pb.step.error"] = True
            if vi.critical:
                child_attrs["pb.step.critical"] = True
            if vi.proc_duration_s is not None:
                child_attrs["pb.step.duration_s"] = vi.proc_duration_s

            # Calculate child span timing from step metadata
            if vi.time_processed and vi.proc_duration_s is not None:
                end_time = datetime.datetime.fromisoformat(vi.time_processed)
                step_end_ns = int(end_time.timestamp() * 1e9)
                step_start_ns = step_end_ns - int(vi.proc_duration_s * 1e9)
            else:
                step_start_ns = start_ns
                step_end_ns = end_ns

            child_span = tracer.start_span(
                "pb.validate.step",
                attributes=child_attrs,
                start_time=step_start_ns,
                context=root_context,
            )
            child_span.end(end_time=step_end_ns)

        # Record threshold breach events on the root span
        for vi in validation.validation_info:
            for level in ("warning", "error", "critical"):
                if getattr(vi, level, False):
                    root_span.add_event(
                        "threshold_breach",
                        attributes={
                            "pb.step.i": vi.i or 0,
                            "pb.step.assertion_type": vi.assertion_type or "",
                            "pb.step.column": str(vi.column) if vi.column else "",
                            "pb.threshold.level": level,
                            "pb.step.f_failed": vi.f_failed or 0.0,
                        },
                    )

        root_span.end(end_time=end_ns)

    # ── Log Emission ────────────────────────────────────────────────

    def _emit_logs(self, validation: Validate, attrs: dict) -> None:
        from opentelemetry._logs import LogRecord, SeverityNumber

        logger = self._get_logger()

        severity_map = {
            "warning": (SeverityNumber.WARN, "WARNING"),
            "error": (SeverityNumber.ERROR, "ERROR"),
            "critical": (SeverityNumber.FATAL, "CRITICAL"),
        }
        log_level_order = {"warning": 0, "error": 1, "critical": 2}
        min_level = log_level_order.get(self.log_level, 0)

        for vi in validation.validation_info:
            # Find the highest exceeded threshold for this step
            highest_level = None
            for level in ("critical", "error", "warning"):
                if getattr(vi, level, False):
                    highest_level = level
                    break

            if highest_level is None:
                continue

            if log_level_order[highest_level] < min_level:
                continue

            f_failed = vi.f_failed or 0.0
            body = (
                f"Validation step {vi.i} ({vi.assertion_type or 'unknown'} "
                f"on '{vi.column or 'unknown'}') exceeded {highest_level} "
                f"threshold: {f_failed:.2%} failed"
            )

            log_attrs = {
                **attrs,
                "pb.step.i": vi.i or 0,
                "pb.step.assertion_type": vi.assertion_type or "",
                "pb.step.column": str(vi.column) if vi.column else "",
                "pb.step.f_failed": f_failed,
                "pb.threshold.level": highest_level,
            }

            sev_number, sev_text = severity_map[highest_level]

            logger.emit(
                LogRecord(
                    body=body,
                    severity_number=sev_number,
                    severity_text=sev_text,
                    attributes=log_attrs,
                )
            )


def emit_otel(
    *,
    meter_name: str = "pointblank",
    enable_metrics: bool = True,
    enable_tracing: bool = False,
    enable_logging: bool = False,
    meter_provider: Any | None = None,
    tracer_provider: Any | None = None,
    logger_provider: Any | None = None,
    metric_prefix: str = "pb.validation",
    log_level: str = "warning",
    extra_attributes: dict[str, str] | None = None,
) -> Callable:
    """Create a final action that sends a validation's metrics to OpenTelemetry.

    The `emit_otel()` function returns an action for [`FinalActions`](`pointblank.FinalActions`)
    that sends validation metrics to OpenTelemetry (OTel) at the end of every
    [`interrogate()`](`pointblank.Validate.interrogate`) call. With the action in the validation
    plan, every run reports to the monitoring backend without any code after the validation. That
    suits validations that run on a schedule, such as checks in a data pipeline.

    The action sends the metrics of
    [`OTelExporter`](`pointblank.integrations.otel.OTelExporter`), with these differences:

    - It sends metrics only. The `enable_tracing=`, `enable_logging=`, `tracer_provider=`,
    `logger_provider=` and `log_level=` arguments currently have no effect. Use
    [`OTelExporter.export()`](`pointblank.integrations.otel.OTelExporter.export`) after the
    interrogation for traces and log records.
    - It doesn't send the step duration histogram (`{prefix}.step.duration`), so there are 11
    metrics instead of 12.
    - The only attributes are `pb.tbl_name` (when the validation has a `tbl_name=`) and those in
    `extra_attributes=`.

    Without `meter_provider=`, the metrics go to OTel's global meter provider, which discards them
    until the application sets up the OTel SDK. The OTel packages are an optional dependency,
    installed with `pip install pointblank[otel]`.

    Parameters
    ----------
    meter_name
        The name that identifies Pointblank as the source of the metrics (the OTel instrumentation
        scope). The default is `"pointblank"`.
    enable_metrics
        Whether to send the metrics. The default is `True`.
    enable_tracing
        Currently has no effect. The default is `False`.
    enable_logging
        Currently has no effect. The default is `False`.
    meter_provider
        An OTel `MeterProvider` for the metrics. The default is `None`, which uses the global meter
        provider.
    tracer_provider
        Currently has no effect. The default is `None`.
    logger_provider
        Currently has no effect. The default is `None`.
    metric_prefix
        The prefix for the names of the metrics. The default is `"pb.validation"`.
    log_level
        Currently has no effect. The default is `"warning"`.
    extra_attributes
        A dictionary of attributes to add to every metric, such as the environment or the name of
        the pipeline. The default is `None`.

    Returns
    -------
    Callable
        A function to use in `FinalActions`.

    Raises
    ------
    ImportError
        If the OTel packages aren't installed.

    Examples
    --------
    Let's add the action to a validation of a small table of orders. The OTel SDK's
    `InMemoryMetricReader` keeps the metrics in memory so we can look at them (in production,
    the meter provider would send them to a collector instead):

    ```{python}
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

    The metrics were sent during `interrogate()`. Both steps failed, 7 of the 10 test units passed,
    and the second step exceeded all three thresholds. Each metric has the `pb.tbl_name` and `env`
    attributes:

    ```{python}
    dict(metrics[0].data.data_points[0].attributes)
    ```

    Since `FinalActions` takes any number of actions, the metrics can be sent along with other
    notifications. Here, a [`send_slack_notification()`](`pointblank.send_slack_notification`)
    action without a webhook URL prints the message it would send:

    ```{python}
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

    The actions run in the order they're given, so the metrics are sent before the Slack message.
    The message summarizes the same results as the metrics: both steps failed, and the highest
    threshold level reached was critical.

    See Also
    --------
    integrations.otel.OTelExporter : Send metrics, traces and log records after an interrogation.
    FinalActions : Run actions after an interrogation.
    send_slack_notification : Send a Slack notification from a final action.
    """
    _check_otel_installed()

    exporter = OTelExporter(
        meter_name=meter_name,
        enable_metrics=enable_metrics,
        enable_tracing=enable_tracing,
        enable_logging=enable_logging,
        meter_provider=meter_provider,
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        metric_prefix=metric_prefix,
        log_level=log_level,
        extra_attributes=extra_attributes,
    )

    def _otel_final_action() -> None:
        from pointblank import get_validation_summary

        summary = get_validation_summary()
        if summary is None:
            return
        exporter._export_from_summary(summary)

    return _otel_final_action
