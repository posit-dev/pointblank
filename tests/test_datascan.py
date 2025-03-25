import pytest
import narwhals as nw

from hypothesis import given, settings
import polars.testing.parametric as pt
from great_tables import GT
import polars as pl

from pointblank.datascan import DataScan, col_summary_tbl
from pointblank._datascan_utils import _compact_0_1_fmt, _compact_decimal_fmt, _compact_integer_fmt


## Setup Strategies:
happy_path_df = pt.dataframes(
    min_size=5, allowed_dtypes=[pl.Int64, pl.Float64, pl.String, pl.Categorical]
)


@given(df=happy_path_df)
def test_datascan_class_parametric(df) -> None:
    scanner = DataScan(data=df)

    df_nw = nw.from_native(df)

    summary_res = scanner.summary_data.to_native()

    ## Go through checks:
    cols = summary_res.select("colname").to_series().to_list()

    msg = "cols must be the same"
    df_cols = df_nw.columns
    assert set(cols) == set(df_cols), msg

    msg = "return type must be same as input"
    assert isinstance(summary_res, type(df)), msg

    msg = "did not return correct amount of summary rows"
    assert len(summary_res) == len(cols)  # only for happy path

    msg = "contains sample data"
    assert "sample_data" in summary_res.columns

    # TODO: Should contain many more cases


@given(df=pt.dataframes(min_size=5))
def test_datascan_json_output(df):
    scanner = DataScan(data=df)

    profile_json = scanner.to_json()

    assert isinstance(profile_json, str)


@given(df=happy_path_df)
@settings(max_examples=5)
def test_col_summary_tbl(df):
    col_summary = col_summary_tbl(df)

    assert isinstance(col_summary, GT)


def test_col_summary_tbl_polars_categorical_column():
    import polars as pl

    log_levels = pl.Enum(["debug", "info", "warning", "error"])

    df_pl = pl.DataFrame(
        {
            "level": ["debug", "info", "debug", "error"],
            "message": [
                "process id: 525",
                "Service started correctly",
                "startup time: 67ms",
                "Cannot connect to DB!",
            ],
        },
        schema_overrides={
            "level": log_levels,
        },
    )

    tabular_output = col_summary_tbl(df_pl)

    assert isinstance(tabular_output, GT)


def test_col_summary_tbl_pandas_snap(snapshot):
    dataset = load_dataset(dataset="small_table", tbl_type="pandas")
    col_summary_html = col_summary_tbl(dataset).as_raw_html()

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(col_summary_html, "col_summary_html_pandas.html")


def test_col_summary_tbl_polars_snap(snapshot):
    dataset = load_dataset(dataset="small_table", tbl_type="polars")
    col_summary_html = col_summary_tbl(dataset).as_raw_html()

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(col_summary_html, "col_summary_html_polars.html")


def test_col_summary_tbl_duckdb_snap(snapshot):
    dataset = load_dataset(dataset="small_table", tbl_type="duckdb")
    col_summary_html = col_summary_tbl(dataset).as_raw_html()

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(col_summary_html, "col_summary_html_duckdb.html")


def test_datascan_class_raises():
    with pytest.raises(TypeError):
        DataScan(data="not a DataFrame or Ibis Table")

    with pytest.raises(TypeError):
        DataScan(data=123)

    with pytest.raises(TypeError):
        DataScan(data=[1, 2, 3])


def test_compact_integer_fmt():
    assert _compact_integer_fmt(value=0) == "0"
    assert _compact_integer_fmt(value=0.4) == "4.0E−1"
    assert _compact_integer_fmt(value=0.6) == "6.0E−1"
    assert _compact_integer_fmt(value=1) == "1"
    assert _compact_integer_fmt(value=43.91) == "44"
    assert _compact_integer_fmt(value=226.1) == "226"
    assert _compact_integer_fmt(value=4362.54) == "4363"
    assert _compact_integer_fmt(value=15321.23) == "1.5E4"


def test_compact_decimal_fmt():
    assert _compact_decimal_fmt(value=0) == "0.00"
    assert _compact_decimal_fmt(value=1) == "1.00"
    assert _compact_decimal_fmt(value=0.0) == "0.00"
    assert _compact_decimal_fmt(value=1.0) == "1.00"
    assert _compact_decimal_fmt(value=0.1) == "0.10"
    assert _compact_decimal_fmt(value=0.5) == "0.50"
    assert _compact_decimal_fmt(value=0.01) == "0.01"
    assert _compact_decimal_fmt(value=0.009) == "9.0E−3"
    assert _compact_decimal_fmt(value=0.000001) == "1.0E−6"
    assert _compact_decimal_fmt(value=0.99) == "0.99"
    assert _compact_decimal_fmt(value=1) == "1.00"
    assert _compact_decimal_fmt(value=43.91) == "43.9"
    assert _compact_decimal_fmt(value=226.1) == "226"
    assert _compact_decimal_fmt(value=4362.54) == "4363"
    assert _compact_decimal_fmt(value=15321.23) == "1.5E4"


def test_compact_0_1_fmt():
    _compact_0_1_fmt(value=0) == "0.0"
    _compact_0_1_fmt(value=1) == "1.0"
    _compact_0_1_fmt(value=0.0) == "0.0"
    _compact_0_1_fmt(value=1.0) == "1.0"
    _compact_0_1_fmt(value=0.1) == "0.1"
    _compact_0_1_fmt(value=0.5) == "0.5"
    _compact_0_1_fmt(value=0.01) == "0.01"
    _compact_0_1_fmt(value=0.009) == "<0.01"
    _compact_0_1_fmt(value=0.000001) == "<0.01"
    _compact_0_1_fmt(value=0.99) == "0.99"
    _compact_0_1_fmt(value=0.991) == ">0.99"
    _compact_0_1_fmt(value=226.1) == "226"


if __name__ == "__main__":
    pytest.main([__file__, "-x"])

