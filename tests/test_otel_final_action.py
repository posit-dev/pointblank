import pytest

otel_sdk = pytest.importorskip("opentelemetry.sdk")

import pointblank as pb
from pointblank.integrations.otel import OTelExporter, emit_otel
from tests.otel_helpers import get_all_metric_names, get_data_point_attributes

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


def test_otel_final_action_outside_final_actions(otel_metric_reader):
    """Called outside of a final action, the action has no validation and sends nothing."""
    reader, provider = otel_metric_reader
    action = emit_otel(meter_provider=provider, enable_metrics=True)

    action()

    assert "pb.validation.steps.total" not in get_all_metric_names(reader)


# ── Same signals as OTelExporter.export() ──────────────────────────────


def _validate_with(final_action):
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"x": [1, 2, None, 4, 5], "y": [10, -3, 5, 0, 8]})
    return (
        pb.Validate(
            data=df,
            tbl_name="test_failures",
            label="Has-failures fixture",
            owner="test-team",
            version="1.0.0",
            thresholds=(0.05, 0.10, 0.30),
            final_actions=pb.FinalActions(final_action),
        )
        .col_vals_not_null(columns="x")  # 1 failure (20%) → exceeds warning + error
        .col_vals_gt(columns="y", value=0)  # 2 failures (40%) → exceeds all thresholds
        .interrogate()
    )


def _metric_points(reader):
    """Every data point as `{(metric name, attributes): value}`, with histograms as `(count, sum)`."""
    points = {}
    for rm in reader.get_metrics_data().resource_metrics:
        for sm in rm.scope_metrics:
            for metric in sm.metrics:
                for dp in metric.data.data_points:
                    key = (metric.name, frozenset(dp.attributes.items()))
                    points[key] = (dp.count, dp.sum) if hasattr(dp, "count") else dp.value
    return points


def test_metrics_match_export(otel_metric_reader):
    """emit_otel() sends the same metrics, values and attributes as export() (all 12 metrics)."""
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader, provider = otel_metric_reader
    validation = _validate_with(
        emit_otel(meter_provider=provider, extra_attributes={"env": "test"})
    )

    export_reader = InMemoryMetricReader()
    export_provider = MeterProvider(metric_readers=[export_reader])
    OTelExporter(meter_provider=export_provider, extra_attributes={"env": "test"}).export(
        validation
    )

    emitted = _metric_points(reader)
    assert emitted == _metric_points(export_reader)

    names = {name for name, _ in emitted}
    assert len(names) == 12
    assert "pb.validation.step.duration" in names

    attrs = get_data_point_attributes(reader, "pb.validation.steps.total")[0]
    assert attrs == {
        "pb.tbl_name": "test_failures",
        "pb.label": "Has-failures fixture",
        "pb.owner": "test-team",
        "pb.version": "1.0.0",
        "env": "test",
    }
    export_provider.shutdown()


def test_traces_emitted_via_final_actions(otel_span_exporter):
    exporter, provider = otel_span_exporter
    _validate_with(emit_otel(enable_metrics=False, enable_tracing=True, tracer_provider=provider))

    spans = exporter.get_finished_spans()
    roots = [s for s in spans if s.parent is None]
    children = [s for s in spans if s.parent is not None]

    assert [s.name for s in roots] == ["pb.validate"]
    assert [s.name for s in children] == ["pb.validate.step", "pb.validate.step"]
    assert all(s.parent.span_id == roots[0].context.span_id for s in children)
    assert [s.attributes["pb.step.n_failed"] for s in children] == [1, 2]
    assert roots[0].attributes["pb.label"] == "Has-failures fixture"
    assert len([e for e in roots[0].events if e.name == "threshold_breach"]) == 5


def test_spans_nest_under_current_span(otel_span_exporter):
    """The spans are sent during interrogate(), so they join the span that's current there."""
    exporter, provider = otel_span_exporter
    tracer = provider.get_tracer("pipeline")

    with tracer.start_as_current_span("pipeline_task") as task_span:
        _validate_with(
            emit_otel(enable_metrics=False, enable_tracing=True, tracer_provider=provider)
        )

    root = next(s for s in exporter.get_finished_spans() if s.name == "pb.validate")
    assert root.parent.span_id == task_span.get_span_context().span_id


def test_logs_emitted_via_final_actions(otel_log_exporter):
    exporter, provider = otel_log_exporter
    _validate_with(
        emit_otel(
            enable_metrics=False,
            enable_logging=True,
            logger_provider=provider,
            log_level="critical",
        )
    )

    logs = exporter.get_finished_logs()
    # Only the second step reaches the critical threshold
    assert len(logs) == 1
    assert logs[0].log_record.severity_text == "CRITICAL"
    assert logs[0].log_record.attributes["pb.step.i"] == 2


def test_meter_version_is_passed_through(otel_metric_reader):
    reader, provider = otel_metric_reader
    _validate_with(emit_otel(meter_provider=provider, meter_version="9.9.9"))

    scope = reader.get_metrics_data().resource_metrics[0].scope_metrics[0].scope
    assert scope.name == "pointblank"
    assert scope.version == "9.9.9"


def test_invalid_arguments_raise_when_the_plan_is_built():
    """Bad arguments raise from emit_otel(), not from inside interrogate()."""
    with pytest.raises(ValueError, match="needs a `logger_provider=`"):
        emit_otel(enable_logging=True)
    with pytest.raises(ValueError, match="`log_level=` must be one of"):
        emit_otel(log_level="warn")


# ── Final action context ───────────────────────────────────────────────


def test_nested_validation_keeps_outer_context():
    """A final action that runs another validation gets its own summary and validation back."""
    pl = pytest.importorskip("polars")
    from pointblank.validate import _get_current_validation

    seen = {}

    def inner_action():
        seen["inner"] = pb.get_validation_summary()["tbl_name"]

    def outer_action():
        (
            pb.Validate(
                data=pl.DataFrame({"z": [1]}),
                tbl_name="inner",
                final_actions=pb.FinalActions(inner_action),
            )
            .col_vals_gt(columns="z", value=0)
            .interrogate()
        )
        seen["outer_summary"] = pb.get_validation_summary()["tbl_name"]
        seen["outer_validation"] = _get_current_validation().tbl_name

    (
        pb.Validate(
            data=pl.DataFrame({"x": [1, 2]}),
            tbl_name="outer",
            final_actions=pb.FinalActions(outer_action),
        )
        .col_vals_gt(columns="x", value=0)
        .interrogate()
    )

    assert seen == {"inner": "inner", "outer_summary": "outer", "outer_validation": "outer"}
    assert pb.get_validation_summary() is None
    assert _get_current_validation() is None
