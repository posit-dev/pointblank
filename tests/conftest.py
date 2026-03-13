import pytest


@pytest.fixture
def half_null_ser():
    """A 1k element half null series. Exists to get around rounding issues."""
    pl = pytest.importorskip("polars")
    data = [None if i % 2 == 0 else i for i in range(1000)]
    return pl.Series("half_null", data)


# ── Backend smoke test fixtures ────────────────────────────────────────────────


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
