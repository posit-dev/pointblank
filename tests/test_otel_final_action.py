import pytest

otel_sdk = pytest.importorskip("opentelemetry.sdk")

import pointblank as pb
from pointblank.integrations.otel import emit_otel
from tests.otel_helpers import get_all_metric_names

pytestmark = pytest.mark.otel


def test_metrics_emitted_via_final_actions(otel_metric_reader):
    """Metrics should be emitted automatically at end of interrogate()."""
    pl = pytest.importorskip("polars")

    reader, provider = otel_metric_reader
    df = pl.DataFrame({"x": [1, 2, 3]})

    _ = (
        pb.Validate(
            data=df,
            tbl_name="final_action_test",
            final_actions=pb.FinalActions(
                emit_otel(
                    meter_provider=provider,
                    enable_metrics=True,
                ),
            ),
        )
        .col_vals_gt(columns="x", value=0)
        .interrogate()
    )

    names = get_all_metric_names(reader)
    assert "pb.validation.steps.total" in names


def test_emit_otel_combined_with_other_actions(otel_metric_reader):
    """emit_otel() and other FinalActions should both execute."""
    pl = pytest.importorskip("polars")

    reader, provider = otel_metric_reader
    callback_called = []

    def track_callback():
        callback_called.append(True)

    df = pl.DataFrame({"x": [1, 2, 3]})
    _ = (
        pb.Validate(
            data=df,
            tbl_name="combo_test",
            final_actions=pb.FinalActions(
                emit_otel(meter_provider=provider),
                track_callback,
            ),
        )
        .col_vals_gt(columns="x", value=0)
        .interrogate()
    )

    assert len(callback_called) == 1
    assert "pb.validation.steps.total" in get_all_metric_names(reader)
