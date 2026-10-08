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

    Parameters
    ----------
    meter_name
        Name for the OTel Meter. Defaults to `"pointblank"`.
    meter_version
        Version string for the Meter. Defaults to `"pointblank.__version__"`.
    enable_metrics
        Emit OTel metrics (counters, gauges, histograms). Default is `"True"`.
    enable_tracing
        Emit OTel trace spans. Default is `"False"`.
    enable_logging
        Emit OTel log records for threshold breaches. Default is `"False"`.
    meter_provider
        Custom MeterProvider. If `"None"`, uses the global default.
    tracer_provider
        Custom TracerProvider. If `"None"`, uses the global default.
    logger_provider
        Custom LoggerProvider. Required when `"enable_logging=True"`.
    metric_prefix
        Prefix for all metric instrument names. Default is `"pb.validation"`.
    log_level
        Minimum severity level for log emission. One of `"warning"`, `"error"`, `"critical"`.
        Default is `"warning"`.
    extra_attributes
        Additional key-value pairs attached to all emitted signals.
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
        """Export validation results as OTel signals.

        Parameters
        ----------
        validation
            A `Validate` object that has been interrogated. If not yet interrogated, raises
            `ValueError`.
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
    """Create an OTel export action for use in `FinalActions`.

    Returns a callable that, when invoked as a FinalAction after interrogation, exports validation
    results as OpenTelemetry metrics.

    Parameters
    ----------
    meter_name
        Name for the OTel Meter. Defaults to `"pointblank"`.
    enable_metrics
        Emit OTel metrics (counters, gauges, histograms). Default is `"True"`.
    enable_tracing
        Emit OTel trace spans. Default is `"False"`.
    enable_logging
        Emit OTel log records for threshold breaches. Default is `"False"`.
    meter_provider
        Custom MeterProvider. If `"None"`, uses the global default.
    tracer_provider
        Custom TracerProvider. If `"None"`, uses the global default.
    logger_provider
        Custom LoggerProvider. Required when `"enable_logging=True"`.
    metric_prefix
        Prefix for all metric instrument names. Default is `"pb.validation"`.
    log_level
        Minimum severity level for log emission. One of `"warning"`, `"error"`, `"critical"`.
        The default is `"warning"`.
    extra_attributes
        Additional key-value pairs attached to all emitted signals.

    Returns
    -------
    Callable
        A callable suitable for use in `FinalActions`.

    Examples
    --------
    ```python
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
