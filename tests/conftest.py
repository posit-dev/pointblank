import pytest


@pytest.fixture
def half_null_ser():
    """A 1k element half null series. Exists to get around rounding issues."""
    pl = pytest.importorskip("polars")
    data = [None if i % 2 == 0 else i for i in range(1000)]
    return pl.Series("half_null", data)


@pytest.fixture
def pandas_tbl():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"x": [1, 2, 3]})


@pytest.fixture
def polars_tbl():
    pl = pytest.importorskip("polars")
    return pl.DataFrame({"x": [1, 2, 3]})


@pytest.fixture
def ibis_tbl():
    ibis = pytest.importorskip("ibis")
    return ibis.memtable({"x": [1, 2, 3]})


@pytest.fixture
def arrow_tbl():
    pa = pytest.importorskip("pyarrow")
    return pa.Table.from_pydict({"x": [1, 2, 3]})


@pytest.fixture(
    params=[
        pytest.param("pandas_tbl", id="pandas"),
        pytest.param("polars_tbl", id="polars"),
        pytest.param("ibis_tbl", id="ibis"),
        pytest.param("arrow_tbl", id="pyarrow"),
    ]
)
def backend_tbl(request):
    """Parameterized fixture that provides tables for each backend in turn.

    This fixture receives the name of another fixture and calls it via
    request.getfixturevalue(). This allows it to run once per backend.
    """
    fixture_name = request.param
    return request.getfixturevalue(fixture_name)


# ── OTel Test Fixtures ─────────────────────────────────────────────────


@pytest.fixture()
def otel_metric_reader():
    """Provide an in-memory MeterProvider + reader pair.

    Returns `(reader, provider)`. Call `reader.get_metrics_data()` after exporting to inspect
    emitted metrics.
    """
    otel_sdk = pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    yield reader, provider
    provider.shutdown()


@pytest.fixture()
def otel_span_exporter():
    """Provide an in-memory TracerProvider + span exporter pair.

    Returns `(span_exporter, provider)`. Call `span_exporter.get_finished_spans()` after exporting
    to inspect spans.
    """
    otel_sdk = pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield exporter, provider
    provider.shutdown()


@pytest.fixture()
def otel_log_exporter():
    """Provide an in-memory LoggerProvider + log exporter pair.

    Returns `(log_exporter, provider)`. Call `log_exporter.get_finished_logs()` after exporting
    to inspect logs.
    """
    otel_sdk = pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import (
        InMemoryLogRecordExporter,
        SimpleLogRecordProcessor,
    )

    exporter = InMemoryLogRecordExporter()
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(exporter))
    yield exporter, provider
    provider.shutdown()


@pytest.fixture()
def validation_all_pass():
    """A fully-passing Validate object for baseline metric assertions."""
    import pointblank as pb

    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"x": [1, 2, 3, 4, 5], "y": ["a", "b", "c", "d", "e"]})
    return (
        pb.Validate(data=df, tbl_name="test_all_pass", label="All-pass fixture")
        .col_vals_not_null(columns="x")
        .col_vals_gt(columns="x", value=0)
        .interrogate()
    )


@pytest.fixture()
def validation_with_failures():
    """A Validate object with known failures and threshold breaches."""
    import pointblank as pb

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
        )
        .col_vals_not_null(columns="x")  # 1 failure (20%) → exceeds warning + error
        .col_vals_gt(columns="y", value=0)  # 2 failures (40%) → exceeds all thresholds
        .interrogate()
    )


@pytest.fixture()
def validation_mixed_thresholds():
    """A Validate with steps at different threshold severity levels."""
    import pointblank as pb

    pl = pytest.importorskip("polars")
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, None],
        }
    )
    return (
        pb.Validate(
            data=df,
            tbl_name="test_mixed",
            thresholds=(0.05, 0.15, 0.25),
        )
        .col_vals_gt(columns="a", value=0)  # 0% fail → no threshold
        .col_vals_not_null(columns="b")  # 10% fail → warning only
        .interrogate()
    )
