import pytest

otel_sdk = pytest.importorskip("opentelemetry.sdk")

from pointblank.integrations.otel import OTelExporter

pytestmark = pytest.mark.otel


# ── Log Emission ────────────────────────────────────────────────────────


def test_logs_emitted_for_threshold_breaches(otel_log_exporter, validation_with_failures):
    exporter, provider = otel_log_exporter
    OTelExporter(
        enable_logging=True,
        enable_metrics=False,
        logger_provider=provider,
    ).export(validation_with_failures)

    logs = exporter.get_finished_logs()
    assert len(logs) >= 1


def test_no_logs_when_all_pass(otel_log_exporter, validation_all_pass):
    exporter, provider = otel_log_exporter
    OTelExporter(
        enable_logging=True,
        enable_metrics=False,
        logger_provider=provider,
    ).export(validation_all_pass)

    logs = exporter.get_finished_logs()
    assert len(logs) == 0


def test_log_body_contains_step_info(otel_log_exporter, validation_with_failures):
    exporter, provider = otel_log_exporter
    OTelExporter(
        enable_logging=True,
        enable_metrics=False,
        logger_provider=provider,
    ).export(validation_with_failures)

    log_bodies = [log.log_record.body for log in exporter.get_finished_logs()]
    assert any("col_vals" in body for body in log_bodies)


def test_log_attributes_include_tbl_name(otel_log_exporter, validation_with_failures):
    exporter, provider = otel_log_exporter
    OTelExporter(
        enable_logging=True,
        enable_metrics=False,
        logger_provider=provider,
    ).export(validation_with_failures)

    for log in exporter.get_finished_logs():
        attrs = dict(log.log_record.attributes)
        assert attrs.get("pb.tbl_name") == "test_failures"


# ── Log Level Filtering ─────────────────────────────────────────────────


def test_log_level_critical_filters_lower(otel_log_exporter, validation_mixed_thresholds):
    """With log_level='critical', only critical-level breaches emit logs."""
    exporter, provider = otel_log_exporter
    OTelExporter(
        enable_logging=True,
        enable_metrics=False,
        logger_provider=provider,
        log_level="critical",
    ).export(validation_mixed_thresholds)

    logs = exporter.get_finished_logs()
    assert len(logs) == 0


def test_log_level_warning_includes_all(otel_log_exporter, validation_with_failures):
    """With log_level='warning' (default), all exceeded thresholds emit logs."""
    exporter, provider = otel_log_exporter
    OTelExporter(
        enable_logging=True,
        enable_metrics=False,
        logger_provider=provider,
        log_level="warning",
    ).export(validation_with_failures)

    logs = exporter.get_finished_logs()
    assert len(logs) == 2


def test_log_level_error_filters_warnings(otel_log_exporter, validation_mixed_thresholds):
    """With log_level='error', warning-only breaches are skipped."""
    exporter, provider = otel_log_exporter
    OTelExporter(
        enable_logging=True,
        enable_metrics=False,
        logger_provider=provider,
        log_level="error",
    ).export(validation_mixed_thresholds)

    logs = exporter.get_finished_logs()
    assert len(logs) == 0
