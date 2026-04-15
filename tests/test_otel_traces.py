import pytest

otel_sdk = pytest.importorskip("opentelemetry.sdk")

from pointblank.integrations.otel import OTelExporter

pytestmark = pytest.mark.otel


# ── Span Hierarchy ──────────────────────────────────────────────────────


def test_single_root_span(otel_span_exporter, validation_with_failures):
    exporter, provider = otel_span_exporter
    OTelExporter(
        enable_tracing=True,
        enable_metrics=False,
        tracer_provider=provider,
    ).export(validation_with_failures)

    spans = exporter.get_finished_spans()
    root_spans = [s for s in spans if s.parent is None]
    assert len(root_spans) == 1
    assert root_spans[0].name == "pb.validate"


def test_child_span_per_step(otel_span_exporter, validation_with_failures):
    exporter, provider = otel_span_exporter
    OTelExporter(
        enable_tracing=True,
        enable_metrics=False,
        tracer_provider=provider,
    ).export(validation_with_failures)

    spans = exporter.get_finished_spans()
    child_spans = [s for s in spans if s.parent is not None]
    assert len(child_spans) == 2


def test_all_spans_share_trace_id(otel_span_exporter, validation_with_failures):
    exporter, provider = otel_span_exporter
    OTelExporter(
        enable_tracing=True,
        enable_metrics=False,
        tracer_provider=provider,
    ).export(validation_with_failures)

    spans = exporter.get_finished_spans()
    trace_ids = {s.context.trace_id for s in spans}
    assert len(trace_ids) == 1


def test_child_spans_parent_is_root(otel_span_exporter, validation_with_failures):
    exporter, provider = otel_span_exporter
    OTelExporter(
        enable_tracing=True,
        enable_metrics=False,
        tracer_provider=provider,
    ).export(validation_with_failures)

    spans = exporter.get_finished_spans()
    root = [s for s in spans if s.parent is None][0]
    children = [s for s in spans if s.parent is not None]
    for child in children:
        assert child.parent.span_id == root.context.span_id


# ── Span Attributes ─────────────────────────────────────────────────────


def test_root_span_attributes(otel_span_exporter, validation_with_failures):
    exporter, provider = otel_span_exporter
    OTelExporter(
        enable_tracing=True,
        enable_metrics=False,
        tracer_provider=provider,
    ).export(validation_with_failures)

    root = [s for s in exporter.get_finished_spans() if s.parent is None][0]
    assert root.attributes["pb.tbl_name"] == "test_failures"
    assert root.attributes["pb.all_passed"] is False
    assert root.attributes["pb.steps.total"] == 2


def test_child_span_attributes(otel_span_exporter, validation_with_failures):
    exporter, provider = otel_span_exporter
    OTelExporter(
        enable_tracing=True,
        enable_metrics=False,
        tracer_provider=provider,
    ).export(validation_with_failures)

    children = [s for s in exporter.get_finished_spans() if s.parent is not None]
    child_attrs = {s.attributes["pb.step.i"]: s.attributes for s in children}
    assert child_attrs[1]["pb.step.assertion_type"] == "col_vals_not_null"
    assert child_attrs[2]["pb.step.assertion_type"] == "col_vals_gt"


# ── Span Timing ─────────────────────────────────────────────────────────


def test_root_span_duration_positive(otel_span_exporter, validation_all_pass):
    exporter, provider = otel_span_exporter
    OTelExporter(
        enable_tracing=True,
        enable_metrics=False,
        tracer_provider=provider,
    ).export(validation_all_pass)

    root = [s for s in exporter.get_finished_spans() if s.parent is None][0]
    assert root.end_time > root.start_time


def test_child_spans_within_root_bounds(otel_span_exporter, validation_all_pass):
    exporter, provider = otel_span_exporter
    OTelExporter(
        enable_tracing=True,
        enable_metrics=False,
        tracer_provider=provider,
    ).export(validation_all_pass)

    spans = exporter.get_finished_spans()
    root = [s for s in spans if s.parent is None][0]
    children = [s for s in spans if s.parent is not None]
    # Allow 1ms tolerance for float->int nanosecond conversion rounding
    tolerance_ns = 1_000_000
    for child in children:
        assert child.start_time >= root.start_time - tolerance_ns
        assert child.end_time <= root.end_time + tolerance_ns


# ── Span Events ─────────────────────────────────────────────────────────


def test_threshold_breach_events_emitted(otel_span_exporter, validation_with_failures):
    exporter, provider = otel_span_exporter
    OTelExporter(
        enable_tracing=True,
        enable_metrics=False,
        tracer_provider=provider,
    ).export(validation_with_failures)

    root = [s for s in exporter.get_finished_spans() if s.parent is None][0]
    breach_events = [e for e in root.events if e.name == "threshold_breach"]
    assert len(breach_events) >= 1


def test_no_events_when_all_pass(otel_span_exporter, validation_all_pass):
    exporter, provider = otel_span_exporter
    OTelExporter(
        enable_tracing=True,
        enable_metrics=False,
        tracer_provider=provider,
    ).export(validation_all_pass)

    root = [s for s in exporter.get_finished_spans() if s.parent is None][0]
    breach_events = [e for e in root.events if e.name == "threshold_breach"]
    assert len(breach_events) == 0


# ── Context Propagation ─────────────────────────────────────────────────


def test_spans_nest_under_existing_context(otel_span_exporter, validation_all_pass):
    """When a parent span is active, pb.validate becomes its child."""
    from opentelemetry import trace

    exporter, provider = otel_span_exporter
    tracer = provider.get_tracer("test")

    with tracer.start_as_current_span("pipeline.task") as parent_span:
        OTelExporter(
            enable_tracing=True,
            enable_metrics=False,
            tracer_provider=provider,
        ).export(validation_all_pass)

    spans = exporter.get_finished_spans()
    pb_root = [s for s in spans if s.name == "pb.validate"][0]
    assert pb_root.parent is not None
    assert pb_root.parent.span_id == parent_span.get_span_context().span_id
