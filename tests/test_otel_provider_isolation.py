import pytest

otel_sdk = pytest.importorskip("opentelemetry.sdk")

import pointblank as pb
from pointblank.integrations.otel import OTelExporter
from tests.otel_helpers import get_all_metric_names, get_counter_value, get_data_point_attributes

pytestmark = pytest.mark.otel


def test_separate_providers_independent():
    """Two OTelExporter instances with different providers don't cross-talk."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    pl = pytest.importorskip("polars")

    reader_a = InMemoryMetricReader()
    provider_a = MeterProvider(metric_readers=[reader_a])

    reader_b = InMemoryMetricReader()
    provider_b = MeterProvider(metric_readers=[reader_b])

    df = pl.DataFrame({"x": [1, 2, 3]})
    v = pb.Validate(data=df, tbl_name="isolation").col_vals_gt(columns="x", value=0).interrogate()

    OTelExporter(meter_provider=provider_a).export(v)

    # Provider B should have no data
    assert get_all_metric_names(reader_b) == set()
    # Provider A should have data
    assert "pb.validation.steps.total" in get_all_metric_names(reader_a)

    provider_a.shutdown()
    provider_b.shutdown()


def test_multiple_exports_accumulate(otel_metric_reader):
    """Counters should accumulate across multiple export() calls."""
    pl = pytest.importorskip("polars")

    reader, provider = otel_metric_reader
    df = pl.DataFrame({"x": [1, 2, 3]})
    v = pb.Validate(data=df, tbl_name="accum").col_vals_gt(columns="x", value=0).interrogate()

    exporter = OTelExporter(meter_provider=provider)
    exporter.export(v)
    exporter.export(v)

    # Counter for steps.total should have accumulated (1 step x 2 exports = 2)
    assert get_counter_value(reader, "pb.validation.steps.total") == 2


def test_exporter_reuse_across_different_validations(otel_metric_reader):
    """Same OTelExporter instance can export different validations."""
    pl = pytest.importorskip("polars")

    reader, provider = otel_metric_reader
    df1 = pl.DataFrame({"x": [1, 2, 3]})
    df2 = pl.DataFrame({"x": [1, None, 3]})

    v1 = pb.Validate(data=df1, tbl_name="tbl_a").col_vals_gt(columns="x", value=0).interrogate()
    v2 = pb.Validate(data=df2, tbl_name="tbl_b").col_vals_not_null(columns="x").interrogate()

    exporter = OTelExporter(meter_provider=provider)
    exporter.export(v1)
    exporter.export(v2)

    # Both should have emitted — verify by checking attribute diversity
    attrs_list = get_data_point_attributes(reader, "pb.validation.steps.total")
    tbl_names = {a.get("pb.tbl_name") for a in attrs_list}
    assert "tbl_a" in tbl_names
    assert "tbl_b" in tbl_names


def test_no_global_provider_mutation():
    """OTelExporter with explicit provider should not alter the global provider."""
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    pl = pytest.importorskip("polars")

    global_before = otel_metrics.get_meter_provider()
    reader = InMemoryMetricReader()
    local_provider = MeterProvider(metric_readers=[reader])

    df = pl.DataFrame({"x": [1, 2, 3]})
    v = pb.Validate(data=df).col_vals_gt(columns="x", value=0).interrogate()
    OTelExporter(meter_provider=local_provider).export(v)

    global_after = otel_metrics.get_meter_provider()
    assert global_before is global_after

    local_provider.shutdown()
