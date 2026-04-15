import pytest

import pointblank as pb

pytestmark = pytest.mark.otel


# ── Import Guard ────────────────────────────────────────────────────────


def test_export_before_interrogate():
    """Exporting before interrogate() should raise ValueError."""
    otel_sdk = pytest.importorskip("opentelemetry.sdk")
    from pointblank.integrations.otel import OTelExporter

    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"x": [1, 2, 3]})

    validation = pb.Validate(data=df).col_vals_not_null(columns="x")
    # NOT interrogated

    exporter = OTelExporter()
    with pytest.raises(ValueError, match="must be interrogated"):
        exporter.export(validation)


# ── Disabled Signals ────────────────────────────────────────────────────


def test_metrics_disabled(otel_metric_reader, validation_all_pass):
    from pointblank.integrations.otel import OTelExporter
    from tests.otel_helpers import get_all_metric_names

    reader, provider = otel_metric_reader
    OTelExporter(
        enable_metrics=False,
        meter_provider=provider,
    ).export(validation_all_pass)
    assert get_all_metric_names(reader) == set()


def test_tracing_disabled_by_default(otel_span_exporter, validation_all_pass):
    from pointblank.integrations.otel import OTelExporter

    exporter, provider = otel_span_exporter
    OTelExporter(
        enable_tracing=False,
        enable_metrics=False,
        tracer_provider=provider,
    ).export(validation_all_pass)
    assert len(exporter.get_finished_spans()) == 0
