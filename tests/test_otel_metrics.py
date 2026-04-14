import pytest

otel_sdk = pytest.importorskip("opentelemetry.sdk")

import pointblank as pb
from pointblank.integrations.otel import OTelExporter
from tests.otel_helpers import (
    get_all_metric_names,
    get_counter_value,
    get_data_point_attributes,
    get_gauge_value,
    get_metrics_by_name,
)

pytestmark = pytest.mark.otel


EXPECTED_METRICS = {
    "pb.validation.steps.total",
    "pb.validation.steps.passed",
    "pb.validation.steps.failed",
    "pb.validation.test_units.total",
    "pb.validation.test_units.passed",
    "pb.validation.test_units.failed",
    "pb.validation.pass_rate",
    "pb.validation.step.duration",
    "pb.validation.duration",
    "pb.validation.threshold.warning",
    "pb.validation.threshold.error",
    "pb.validation.threshold.critical",
}


# ── Completeness ────────────────────────────────────────────────────────


def test_all_specified_metrics_emitted(otel_metric_reader, validation_with_failures):
    reader, provider = otel_metric_reader
    exporter = OTelExporter(meter_provider=provider)
    exporter.export(validation_with_failures)
    emitted = get_all_metric_names(reader)
    assert EXPECTED_METRICS.issubset(emitted), f"Missing metrics: {EXPECTED_METRICS - emitted}"


def test_no_unexpected_metrics_emitted(otel_metric_reader, validation_with_failures):
    reader, provider = otel_metric_reader
    exporter = OTelExporter(meter_provider=provider)
    exporter.export(validation_with_failures)
    emitted = get_all_metric_names(reader)
    unexpected = emitted - EXPECTED_METRICS
    assert not unexpected, f"Unexpected metrics: {unexpected}"


# ── Counter Accuracy ────────────────────────────────────────────────────


def test_steps_total(otel_metric_reader, validation_with_failures):
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_with_failures)
    assert get_counter_value(reader, "pb.validation.steps.total") == 2


def test_steps_passed_all_pass(otel_metric_reader, validation_all_pass):
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_all_pass)
    assert get_counter_value(reader, "pb.validation.steps.passed") == 2
    assert get_counter_value(reader, "pb.validation.steps.failed") == 0


def test_steps_failed_with_failures(otel_metric_reader, validation_with_failures):
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_with_failures)
    assert get_counter_value(reader, "pb.validation.steps.failed") == 2


def test_test_units_match_validation_info(otel_metric_reader, validation_with_failures):
    """Counter values must exactly match sum of validation_info n_passed/n_failed."""
    reader, provider = otel_metric_reader
    v = validation_with_failures
    OTelExporter(meter_provider=provider).export(v)

    expected_passed = sum(vi.n_passed for vi in v.validation_info if vi.n_passed is not None)
    expected_failed = sum(vi.n_failed for vi in v.validation_info if vi.n_failed is not None)

    assert get_counter_value(reader, "pb.validation.test_units.passed") == expected_passed
    assert get_counter_value(reader, "pb.validation.test_units.failed") == expected_failed
    assert (
        get_counter_value(reader, "pb.validation.test_units.total")
        == expected_passed + expected_failed
    )


def test_threshold_counters_warning_only(otel_metric_reader, validation_mixed_thresholds):
    """Step with 10% fail rate exceeds only warning (threshold=0.05)."""
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_mixed_thresholds)
    assert get_counter_value(reader, "pb.validation.threshold.warning") == 1
    assert get_counter_value(reader, "pb.validation.threshold.error") == 0
    assert get_counter_value(reader, "pb.validation.threshold.critical") == 0


def test_threshold_counters_all_exceeded(otel_metric_reader, validation_with_failures):
    """Steps with 20%/40% fail rates exceed all thresholds (0.05/0.10/0.30)."""
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_with_failures)
    assert get_counter_value(reader, "pb.validation.threshold.warning") == 2
    assert get_counter_value(reader, "pb.validation.threshold.error") == 2
    assert get_counter_value(reader, "pb.validation.threshold.critical") == 1


# ── Gauge Accuracy ──────────────────────────────────────────────────────


def test_pass_rate_all_pass(otel_metric_reader, validation_all_pass):
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_all_pass)
    assert get_gauge_value(reader, "pb.validation.pass_rate") == 1.0


def test_pass_rate_with_failures(otel_metric_reader, validation_with_failures):
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_with_failures)
    rate = get_gauge_value(reader, "pb.validation.pass_rate")
    assert 0.0 <= rate < 1.0


def test_pass_rate_range(otel_metric_reader, validation_with_failures):
    """Pass rate must always be in [0, 1]."""
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_with_failures)
    rate = get_gauge_value(reader, "pb.validation.pass_rate")
    assert 0.0 <= rate <= 1.0


def test_duration_is_positive(otel_metric_reader, validation_all_pass):
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_all_pass)
    duration = get_gauge_value(reader, "pb.validation.duration")
    assert duration > 0.0


# ── Histogram Accuracy ──────────────────────────────────────────────────


def test_histogram_count_matches_step_count(otel_metric_reader, validation_with_failures):
    """Histogram count = number of validation steps."""
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_with_failures)
    metrics = get_metrics_by_name(reader)
    hist = metrics["pb.validation.step.duration"]
    total_count = sum(dp.count for dp in hist.data.data_points)
    assert total_count == 2


def test_histogram_values_are_positive(otel_metric_reader, validation_all_pass):
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_all_pass)
    metrics = get_metrics_by_name(reader)
    hist = metrics["pb.validation.step.duration"]
    for dp in hist.data.data_points:
        assert dp.min >= 0.0
        assert dp.sum > 0.0


# ── Attributes ──────────────────────────────────────────────────────────


def test_tbl_name_attribute(otel_metric_reader, validation_all_pass):
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_all_pass)
    attrs_list = get_data_point_attributes(reader, "pb.validation.steps.total")
    assert all(a.get("pb.tbl_name") == "test_all_pass" for a in attrs_list)


def test_label_attribute(otel_metric_reader, validation_all_pass):
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_all_pass)
    attrs_list = get_data_point_attributes(reader, "pb.validation.steps.total")
    assert all(a.get("pb.label") == "All-pass fixture" for a in attrs_list)


def test_owner_attribute(otel_metric_reader, validation_with_failures):
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_with_failures)
    attrs_list = get_data_point_attributes(reader, "pb.validation.steps.total")
    assert all(a.get("pb.owner") == "test-team" for a in attrs_list)


def test_version_attribute(otel_metric_reader, validation_with_failures):
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_with_failures)
    attrs_list = get_data_point_attributes(reader, "pb.validation.steps.total")
    assert all(a.get("pb.version") == "1.0.0" for a in attrs_list)


def test_optional_attributes_omitted_when_none(otel_metric_reader, validation_all_pass):
    """Attributes not set on Validate (owner, version) should not appear."""
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_all_pass)
    attrs_list = get_data_point_attributes(reader, "pb.validation.steps.total")
    for attrs in attrs_list:
        assert "pb.owner" not in attrs
        assert "pb.version" not in attrs


def test_extra_attributes_merged(otel_metric_reader, validation_all_pass):
    reader, provider = otel_metric_reader
    OTelExporter(
        meter_provider=provider,
        extra_attributes={"env": "staging", "team": "data-eng"},
    ).export(validation_all_pass)
    attrs_list = get_data_point_attributes(reader, "pb.validation.steps.total")
    for attrs in attrs_list:
        assert attrs["env"] == "staging"
        assert attrs["team"] == "data-eng"


def test_per_step_attributes_on_histogram(otel_metric_reader, validation_with_failures):
    """Histogram data points should carry step-level attributes."""
    reader, provider = otel_metric_reader
    OTelExporter(meter_provider=provider).export(validation_with_failures)
    attrs_list = get_data_point_attributes(reader, "pb.validation.step.duration")
    assertion_types = {a.get("pb.step.assertion_type") for a in attrs_list}
    assert "col_vals_not_null" in assertion_types
    assert "col_vals_gt" in assertion_types


# ── Custom Prefix ───────────────────────────────────────────────────────


def test_custom_prefix(otel_metric_reader, validation_all_pass):
    reader, provider = otel_metric_reader
    OTelExporter(
        meter_provider=provider,
        metric_prefix="myapp.dq",
    ).export(validation_all_pass)
    names = get_all_metric_names(reader)
    assert "myapp.dq.steps.total" in names
    assert "pb.validation.steps.total" not in names
