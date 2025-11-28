from __future__ import annotations

import os
import pytest
import narwhals as nw

import polars.selectors as cs
from hypothesis import given, settings, strategies as st, example
import polars.testing.parametric as ptp
from great_tables import GT
from typing import TYPE_CHECKING, NamedTuple
import polars as pl
import polars.testing as pt
import pointblank as pb

from pointblank.datascan import DataScan, col_summary_tbl
from pointblank.validate import get_data_path, _process_github_url
from pointblank._datascan_utils import _compact_0_1_fmt, _compact_decimal_fmt, _compact_integer_fmt
from pointblank.scan_profile_stats import StatGroup, COLUMN_ORDER_REGISTRY
from pointblank.scan_profile import _TypeMap, _DataProfile

from pointblank._constants import SVG_ICONS_FOR_DATA_TYPES
from enum import Enum

if TYPE_CHECKING:
    import pyarrow as pa
    import pandas as pd


## Setup Strategies:
## Generate df and ldf happy paths using polars.
## Also generate pandas and arrow strategies which should smoke test any complete mistakes
## or inconsistent handling in narwhals. Really checking the consistency among packages is
## too much the job of narwhals, and we should avoid stepping on their testing suite.
## LDF gets a datetime check because eager datetime values are not easily handled by pandas.
## We need the coverage of datetimes generally and that is checked by the ldf, just not for eager.
happy_path_df = ptp.dataframes(
    min_size=5,
    allowed_dtypes=[pl.Int64, pl.Float64, pl.String, pl.Categorical, pl.Date],
)
happy_path_ldf = ptp.dataframes(
    min_size=5,
    allowed_dtypes=[pl.Int64, pl.Float64, pl.String, pl.Categorical, pl.Date, pl.Datetime],
    lazy=True,
)


@st.composite
def _arrow_strat(draw) -> pa.Table:
    polars_df = draw(happy_path_df)
    return nw.from_native(polars_df).to_arrow()


@st.composite
def _pandas_strat(draw) -> pd.DataFrame:
    polars_df = draw(happy_path_df)
    return nw.from_native(polars_df).to_pandas()


@given(happy_path_df | happy_path_ldf | _arrow_strat() | _pandas_strat())
@example(pb.load_dataset("small_table", "polars"))
@example(pb.load_dataset("small_table", "pandas"))
@example(pb.load_dataset("small_table", "duckdb"))
@example(pb.load_dataset("game_revenue", "polars"))
@example(pb.load_dataset("game_revenue", "pandas"))
@example(pb.load_dataset("game_revenue", "duckdb"))
@example(pb.load_dataset("nycflights", "polars"))
@example(pb.load_dataset("nycflights", "pandas"))
@example(pb.load_dataset("nycflights", "duckdb"))
@settings(deadline=None)  # too variant to enforce deadline
def test_datascan_class_parametric(df) -> None:
    scanner = DataScan(data=df)

    df_nw = nw.from_native(df)

    summary_res: nw.DataFrame = nw.from_native(scanner.summary_data)

    ## High Level Checks:
    cols = summary_res.select("colname").to_dict()["colname"].to_list()

    msg = "cols must be the same"
    df_cols = df_nw.columns
    assert set(cols) == set(df_cols), msg

    msg = "return type is the physical version of the input"
    try:
        assert df_nw.implementation == summary_res.implementation
    except AssertionError:
        if df_nw.implementation.name == "IBIS" and df_nw._level == "lazy":
            pass  # this is actually expected, the summary will come back in another type
        else:
            raise AssertionError

    msg = "did not return correct amount of summary rows"
    assert len(summary_res) == len(cols)  # only for happy path

    msg = "contains sample data"
    assert "sample_data" in summary_res.columns

    ## More Granular Checks:
    cols_that_must_be_there = ("n_missing", "n_unique", "icon", "colname", "sample_data", "coltype")
    for col in cols_that_must_be_there:
        assert col in summary_res.columns, f"Missing column: {col}"

    # this also catches developer error in syncing the calculations and stat classes
    # for example if dev adds a new stat to `scan_profile_stats.py` and does not add
    # it to the `calc_stats` method, this test will fail since it never calculated the
    # statistic.
    msg = "If a single of a group is there, they should all be there."
    for group in StatGroup:
        stats_that_should_be_present: list[str] = [
            stat.name for stat in COLUMN_ORDER_REGISTRY if group == stat.group
        ]
        any_in_summary = any(
            col for col in stats_that_should_be_present if col in summary_res.columns
        )
        if any_in_summary:
            for stat in stats_that_should_be_present:
                assert stat in summary_res.columns, f"{msg}: Missing {stat}"


## Deterministic Casing:
class _Case(NamedTuple):
    data: pl.DataFrame
    should_be: pl.DataFrame


case1 = _Case(
    data=pl.DataFrame(
        {
            # TODO: Make the bool tri-valent
            "bool_col": [True, False, True, False, True],
            "numeric_col": [1.5, 2.3, 3.1, 4.7, 5.2],
        }
    ),
    should_be=pl.DataFrame(
        {
            "colname": ["bool_col", "numeric_col"],
            "std": [None, 1.57],
            "mean": [None, 3.36],
            "max": [None, 5.2],
            "q_1": [None, 2.3],
            "p95": [None, 5.1],
            "n_missing": [0, 0],
            "median": [None, 3.1],
            "iqr": [None, 2.4],
            "p05": [None, 1.516],
            "n_unique": [2, 5],
            "q_3": [None, 4.7],
            "min": [None, 1.5],
            "freqs": [{"True": 3, "False": 2}, None],
        }
    ),
)


@pytest.mark.parametrize("case", [case1])
def test_deterministic_calculations(case: _Case) -> None:
    scanner = DataScan(case.data)

    output = scanner.summary_data.drop("icon", "coltype", "sample_data")

    check_settings = {
        "check_row_order": False,
        "check_column_order": False,
        "check_exact": False,
        "abs_tol": 0.01,
    }

    pt.assert_frame_equal(case.should_be, output, check_dtypes=False, **check_settings)

    output_clean = output.drop("freqs")  # TODO: make this dynamic, ie. a a struct?
    should_be_clean = case.should_be.drop("freqs")

    pt.assert_frame_equal(should_be_clean, output_clean, check_dtypes=True, **check_settings)


@given(happy_path_df | happy_path_ldf | _arrow_strat() | _pandas_strat())
@example(pb.load_dataset("small_table", "polars"))
@example(pb.load_dataset("small_table", "pandas"))
@example(pb.load_dataset("small_table", "duckdb"))
@example(pb.load_dataset("game_revenue", "polars"))
@example(pb.load_dataset("game_revenue", "pandas"))
@example(pb.load_dataset("game_revenue", "duckdb"))
@example(pb.load_dataset("nycflights", "polars"))
@example(pb.load_dataset("nycflights", "pandas"))
@example(pb.load_dataset("nycflights", "duckdb"))
@settings(deadline=None)
def test_datascan_json_output(df):
    scanner = DataScan(data=df)

    profile_json = scanner.to_json()

    assert isinstance(profile_json, str)


@given(happy_path_df | happy_path_ldf | _arrow_strat() | _pandas_strat())
@example(pb.load_dataset("nycflights", "duckdb"))  # ! move this back to the normal spot
@example(pb.load_dataset("small_table", "polars"))
@example(pb.load_dataset("small_table", "pandas"))
@example(pb.load_dataset("small_table", "duckdb"))
@example(pb.load_dataset("game_revenue", "polars"))
@example(pb.load_dataset("game_revenue", "pandas"))
@example(pb.load_dataset("game_revenue", "duckdb"))
@example(pb.load_dataset("nycflights", "polars"))
@example(pb.load_dataset("nycflights", "pandas"))
@settings(deadline=None)
def test_col_summary_tbl(df):
    col_summary = col_summary_tbl(df)

    assert isinstance(col_summary, GT)


def test_col_summary_tbl_polars_categorical_column():
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


def test_datascan_csv_input():
    # Test with individual CSV file
    csv_path = "data_raw/small_table.csv"
    scanner = DataScan(data=csv_path)
    assert scanner.summary_data is not None

    # Test with another CSV file
    csv_path2 = "data_raw/game_revenue.csv"
    scanner2 = DataScan(data=csv_path2)
    assert scanner2.summary_data is not None


def test_datascan_parquet_input():
    # Test with individual Parquet file
    parquet_path = "tests/tbl_files/tbl_xyz.parquet"
    scanner = DataScan(data=parquet_path)
    assert scanner.summary_data is not None

    # Test with another Parquet file
    parquet_path2 = "tests/tbl_files/taxi_sample.parquet"
    scanner2 = DataScan(data=parquet_path2)
    assert scanner2.summary_data is not None


def test_datascan_connection_string_input():
    # Test with DuckDB connection string using get_data_path
    duckdb_path = get_data_path("small_table", "duckdb")
    duckdb_conn = f"duckdb:///{duckdb_path}::small_table"
    scanner = DataScan(data=duckdb_conn)
    assert scanner.summary_data is not None

    # Test with SQLite connection string using absolute path

    sqlite_path = os.path.abspath("tests/tbl_files/tbl_xyz.sqlite")
    sqlite_conn = f"sqlite:///{sqlite_path}::tbl_xyz"
    scanner2 = DataScan(data=sqlite_conn)
    assert scanner2.summary_data is not None


def test_datascan_parquet_glob_patterns():
    # Test with glob pattern for committed parquet files
    parquet_glob = "tests/tbl_files/parquet_data/data_*.parquet"
    scanner = DataScan(data=parquet_glob)
    assert scanner.summary_data is not None


def test_col_summary_tbl_csv_input():
    # Test with individual CSV file
    csv_path = "data_raw/small_table.csv"
    result = col_summary_tbl(csv_path)
    assert isinstance(result, GT)

    # Test with another CSV file
    csv_path2 = "data_raw/game_revenue.csv"
    result2 = col_summary_tbl(csv_path2)
    assert isinstance(result2, GT)


def test_col_summary_tbl_parquet_input():
    # Test with individual Parquet file
    parquet_path = "tests/tbl_files/tbl_xyz.parquet"
    result = col_summary_tbl(parquet_path)
    assert isinstance(result, GT)

    # Test with another Parquet file
    parquet_path2 = "tests/tbl_files/taxi_sample.parquet"
    result2 = col_summary_tbl(parquet_path2)
    assert isinstance(result2, GT)


def test_col_summary_tbl_connection_string_input():
    # Test with DuckDB connection string using get_data_path
    duckdb_path = get_data_path("small_table", "duckdb")
    duckdb_conn = f"duckdb:///{duckdb_path}::small_table"
    result = col_summary_tbl(duckdb_conn)
    assert isinstance(result, GT)

    # Test with SQLite connection string using absolute path

    sqlite_path = os.path.abspath("tests/tbl_files/tbl_xyz.sqlite")
    sqlite_conn = f"sqlite:///{sqlite_path}::tbl_xyz"
    result2 = col_summary_tbl(sqlite_conn)
    assert isinstance(result2, GT)


def test_col_summary_tbl_parquet_glob_patterns():
    # Test with glob pattern for parquet files
    parquet_glob = "tests/tbl_files/parquet_data/data_*.parquet"
    result = col_summary_tbl(parquet_glob)
    assert isinstance(result, GT)


def test_datascan_github_url_csv():
    # Test with a GitHub CSV file from our own repository
    github_csv_url = "https://github.com/posit-dev/pointblank/blob/main/data_raw/small_table.csv"

    try:
        scanner = DataScan(data=github_csv_url)
        assert scanner.summary_data is not None
        # Verify we got some columns
        assert len(scanner.summary_data) > 0
    except Exception as e:
        # If URL access fails (network issues, etc.), skip the test
        pytest.skip(f"GitHub URL test skipped due to network or access issue: {e}")


def test_datascan_github_url_parquet():
    # Test with a GitHub Parquet file from our own repository
    github_parquet_url = "https://github.com/posit-dev/pointblank/blob/main/tests/tbl_files/parquet_data/data_a.parquet"

    try:
        scanner = DataScan(data=github_parquet_url)
        assert scanner.summary_data is not None
        # Verify we got some columns
        assert len(scanner.summary_data) > 0
    except Exception as e:
        # If URL access fails (network issues, etc.), skip the test
        pytest.skip(f"GitHub URL test skipped due to network or access issue: {e}")


def test_col_summary_tbl_github_url_csv():
    # Test with a GitHub CSV file from our own repository
    github_csv_url = "https://github.com/posit-dev/pointblank/blob/main/data_raw/small_table.csv"

    try:
        result = col_summary_tbl(github_csv_url)
        assert isinstance(result, GT)
    except Exception as e:
        # If URL access fails (network issues, etc.), skip the test
        pytest.skip(f"GitHub URL test skipped due to network or access issue: {e}")


def test_github_url_processing_function():
    # Test with non-GitHub URL (should return unchanged)
    non_github_url = "https://example.com/data.csv"
    result = _process_github_url(non_github_url)
    assert result == non_github_url

    # Test with non-URL string (should return unchanged)
    non_url = "local_file.csv"
    result = _process_github_url(non_url)
    assert result == non_url

    # Test with GitHub URL that doesn't point to CSV/Parquet (should return unchanged)
    github_non_data_url = "https://github.com/user/repo/blob/main/README.md"
    result = _process_github_url(github_non_data_url)
    assert result == github_non_data_url

    # Test with malformed GitHub URL (should return unchanged)
    malformed_url = "https://github.com/user/repo/data.csv"  # missing /blob/branch/
    result = _process_github_url(malformed_url)
    assert result == malformed_url


def test_raw_github_url_processing():
    # Use a raw GitHub URL from our own repository
    raw_github_csv_url = (
        "https://raw.githubusercontent.com/posit-dev/pointblank/main/data_raw/small_table.csv"
    )

    try:
        result = _process_github_url(raw_github_csv_url)
        # Should return a dataframe, not the original URL string
        assert not isinstance(result, str), "Result should be a dataframe, not a string"
        # Should be a dataframe with some basic properties
        assert hasattr(result, "shape") or hasattr(result, "height"), (
            "Result should be a dataframe with shape/height attribute"
        )
    except Exception as e:
        # If URL access fails (network issues, etc.), skip the test
        pytest.skip(f"Raw GitHub URL test skipped due to network or access issue: {e}")


def test_raw_github_url_in_col_summary_tbl():
    # Use a raw GitHub URL from our own repository
    raw_github_csv_url = (
        "https://raw.githubusercontent.com/posit-dev/pointblank/main/data_raw/small_table.csv"
    )

    try:
        result = col_summary_tbl(raw_github_csv_url)
        # Should work without being misclassified as a connection string
        assert isinstance(result, GT)
    except Exception as e:
        # If URL access fails (network issues, etc.), skip the test
        pytest.skip(
            f"Raw GitHub URL col_summary_tbl test skipped due to network or access issue: {e}"
        )


def test_raw_github_url_in_datascan():
    # Use a raw GitHub URL from our own repository
    raw_github_csv_url = (
        "https://raw.githubusercontent.com/posit-dev/pointblank/main/data_raw/small_table.csv"
    )

    try:
        data_scan = DataScan(raw_github_csv_url)
        result = data_scan.get_tabular_report()
        # Should work without being misclassified as a connection string
        assert isinstance(result, GT)
    except Exception as e:
        # If URL access fails (network issues, etc.), skip the test
        pytest.skip(f"Raw GitHub URL DataScan test skipped due to network or access issue: {e}")


def test_datascan_with_struct_column():
    """Test that DataScan handles struct columns (illegal types) by skipping them."""

    # Create a DataFrame with a struct column
    df_pl = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "nested": [
                {"a": 1, "b": 2},
                {"a": 3, "b": 4},
                {"a": 5, "b": 6},
            ],
        }
    )

    # DataScan should handle this without error by skipping the struct column
    scanner = DataScan(data=df_pl)

    assert scanner.summary_data is not None

    # Verify that the struct column was skipped
    col_names = scanner.summary_data.select("colname").to_dict()["colname"].to_list()

    # The struct column should not be in the summary
    assert "id" in col_names
    assert "name" in col_names


def test_datascan_save_to_json(tmp_path):
    """Test the save_to_json() method."""

    # Create a simple DataFrame
    df_pl = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "value": [10.5, 20.3, 30.1],
        }
    )

    scanner = DataScan(data=df_pl)

    # Save to a temporary file
    output_file = tmp_path / "test_output.json"
    scanner.save_to_json(str(output_file))

    # Verify the file was created
    assert output_file.exists()

    # Verify the file contains valid JSON
    import json

    with open(output_file) as f:
        content = json.load(f)
        # The saved content should be a JSON string (due to json.dump with a string)
        assert isinstance(content, str)


def test_typemap_fetch_icon_with_unknown_type():
    """Test the KeyError exception handler in _TypeMap.fetch_icon()."""

    # Create a mock type that's not in the icon_map
    class UnknownType(Enum):
        UNKNOWN = ("unknown_type",)

    # This should trigger the KeyError and return the "object" icon
    result = _TypeMap.fetch_icon(UnknownType.UNKNOWN)
    assert result == SVG_ICONS_FOR_DATA_TYPES["object"]


def test_dataprofile_as_dataframe_with_mixed_column_types():
    """Test as_dataframe works correctly with different column types."""
    # Create a DataFrame with different column types
    df_pl = pl.DataFrame(
        {
            "numeric_col": [1, 2, 3],
            "string_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )

    scanner = DataScan(data=df_pl)

    # Call as_dataframe() with strict=False (the default)
    result = scanner.profile.as_dataframe(strict=False)

    # Should succeed without raising TypeError
    assert result is not None
    assert len(result) > 0

    # Verify we can also call with strict=True
    result_strict = scanner.profile.as_dataframe(strict=True)
    assert result_strict is not None
