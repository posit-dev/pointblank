import pyexpat
import pytest
import pandas as pd
import polars as pl
from unittest.mock import Mock, patch

from pointblank._interrogation import (
    _column_has_null_values,
    _modify_datetime_compare_val,
    _safe_is_nan_or_null_expr,
    _safe_modify_datetime_compare_val,
    ConjointlyValidation,
)
from pointblank.column import Column


@pytest.fixture
def tbl_pd():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": ["4", "5", "6", "7"], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_pl():
    return pl.DataFrame({"x": [1, 2, 3, 4], "y": ["4", "5", "6", "7"], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_pd_distinct():
    return pd.DataFrame(
        {
            "col_1": ["a", "b", "c", "d"],
            "col_2": ["a", "a", "c", "d"],
            "col_3": ["a", "a", "d", "e"],
        }
    )


@pytest.fixture
def tbl_pl_distinct():
    return pl.DataFrame(
        {
            "col_1": ["a", "b", "c", "d"],
            "col_2": ["a", "a", "c", "d"],
            "col_3": ["a", "a", "d", "e"],
        }
    )


COLUMN_LIST = ["x", "y", "z", "pb_is_good_"]

COLUMN_LIST_DISTINCT = ["col_1", "col_2", "col_3", "pb_is_good_"]


def test_safe_modify_datetime_with_collect_schema():
    """Test _safe_modify_datetime_compare_val with a LazyFrame (collect_schema path)."""
    import datetime
    import narwhals as nw

    df = nw.from_native(pl.DataFrame({"date_col": [datetime.date(2023, 6, 1)]})).lazy()
    compare_val = datetime.datetime(2023, 1, 1, 12, 0, 0)

    result = _safe_modify_datetime_compare_val(df, "date_col", compare_val)

    # datetime should be coerced to date to match the column dtype
    assert isinstance(result, datetime.date)
    assert not isinstance(result, datetime.datetime)


def test_safe_modify_datetime_with_schema_attribute():
    """Test _safe_modify_datetime_compare_val with an eager DataFrame (schema path)."""
    import datetime
    import narwhals as nw

    df = nw.from_native(pl.DataFrame({"date_col": [datetime.date(2023, 6, 1)]}))
    compare_val = datetime.datetime(2023, 1, 1, 12, 0, 0)

    result = _safe_modify_datetime_compare_val(df, "date_col", compare_val)

    assert isinstance(result, datetime.date)
    assert not isinstance(result, datetime.datetime)


def test_safe_modify_datetime_fallback_sample_collect():
    """Test fallback to sample collection."""

    # Create mock dataframe without schema methods
    mock_df = Mock()
    del mock_df.collect_schema
    del mock_df.schema

    # Mock head().collect() scenario
    mock_sample = Mock()
    mock_sample.dtypes = {"date_col": "datetime64[ns]"}
    mock_sample.columns = ["date_col"]
    mock_df.head.return_value.collect.return_value = mock_sample

    with patch("pointblank._interrogation._modify_datetime_compare_val") as mock_modify:
        mock_modify.return_value = "modified_value"

        result = _safe_modify_datetime_compare_val(mock_df, "date_col", "2023-01-01")

        assert result == "modified_value"


def test_safe_modify_datetime_fallback_sample_exception():
    """Test exception in sample collection."""

    mock_df = Mock()
    del mock_df.collect_schema
    del mock_df.schema
    mock_df.head.side_effect = Exception("Cannot collect")

    # Should not crash and fall through to next fallback
    result = _safe_modify_datetime_compare_val(mock_df, "date_col", "2023-01-01")
    assert result == "2023-01-01"  # Original value returned


def test_safe_modify_datetime_direct_access_fallback():
    """Test direct dtypes access fallback."""

    mock_df = Mock()
    del mock_df.collect_schema
    del mock_df.schema
    mock_df.head.side_effect = Exception("Cannot collect")

    # Set up direct access
    mock_df.dtypes = {"date_col": "datetime64[ns]"}
    mock_df.columns = ["date_col"]

    with patch("pointblank._interrogation._modify_datetime_compare_val") as mock_modify:
        mock_modify.return_value = "modified_value"

        result = _safe_modify_datetime_compare_val(mock_df, "date_col", "2023-01-01")

        assert result == "modified_value"


def test_safe_modify_datetime_direct_access_exception():
    """Test exception in direct access."""

    mock_df = Mock()
    del mock_df.collect_schema
    del mock_df.schema
    mock_df.head.side_effect = Exception("Cannot collect")

    # Make dtypes access raise exception
    type(mock_df).dtypes = Mock(side_effect=Exception("No dtypes"))

    result = _safe_modify_datetime_compare_val(mock_df, "date_col", "2023-01-01")
    assert result == "2023-01-01"  # Original value returned


def test_safe_modify_datetime_outer_exception():
    """Test outer exception handling."""

    mock_df = Mock()

    # Make the entire try block raise an exception
    mock_df.collect_schema.side_effect = Exception("Major failure")

    result = _safe_modify_datetime_compare_val(mock_df, "date_col", "2023-01-01")
    assert result == "2023-01-01"  # Original value returned


@patch("pointblank._interrogation._get_tbl_type")
def test_pyspark_expression_handling_with_error(mock_get_tbl_type):
    """Test PySpark expression error handling."""

    mock_get_tbl_type.return_value = "pyspark"

    # Create a mock PySpark DataFrame
    mock_df = Mock()

    # Create ConjointlyValidation instance with expression functions
    conjointly = ConjointlyValidation(
        data_tbl=mock_df,
        expressions=[],
        threshold=1.0,
        tbl_type="pyspark",
    )

    # Mock expression functions that will fail
    def failing_expr_fn(df):
        raise Exception("PySpark error")

    def failing_col_expr_fn(df):
        # Mock a column expression that also fails conversion
        mock_col_expr = Mock()
        mock_col_expr.to_pyspark_expr.side_effect = Exception("Conversion error")
        return mock_col_expr

    conjointly.expressions = [failing_expr_fn, failing_col_expr_fn]

    # Mock the PySpark imports and methods
    with patch("pyspark.sql.functions.lit") as mock_lit:
        lit_result = Mock()
        mock_lit.return_value = lit_result
        mock_df.withColumn.return_value = "results_table"

        # This should handle the errors gracefully and return default case
        result = conjointly._get_pyspark_results()

        # Should fall back to default case
        assert result == "results_table"
        # Just verify it was called, don't check the exact mock object
        mock_df.withColumn.assert_called_once()
        args, kwargs = mock_df.withColumn.call_args
        assert args[0] == "pb_is_good_"


def test_pyspark_results_table_creation_default_case():
    """Test default case in PySpark results."""

    mock_df = Mock()

    conjointly = ConjointlyValidation(
        data_tbl=mock_df,
        expressions=[],
        threshold=1.0,
        tbl_type="pyspark",
    )

    # Mock PySpark F.lit for the default case
    with patch("pyspark.sql.functions.lit") as mock_lit:
        mock_lit.return_value = "lit_true"
        mock_df.withColumn.return_value = "results_table"

        result = conjointly._get_pyspark_results()

        assert result == "results_table"
        mock_df.withColumn.assert_called_with("pb_is_good_", "lit_true")


def test_pyspark_nested_exception_print():
    """Test the nested exception print statement."""

    mock_df = Mock()

    conjointly = ConjointlyValidation(
        data_tbl=mock_df,
        expressions=[],
        threshold=1.0,
        tbl_type="pyspark",
    )

    def failing_expr_fn(df):
        raise Exception("First error")

    def failing_nested_expr_fn(df):
        if df is None:
            raise Exception("Second error")
        raise Exception("First error")

    conjointly.expressions = [failing_expr_fn, failing_nested_expr_fn]

    # Mock print to capture the error message
    with patch("builtins.print") as mock_print:
        with patch("pyspark.sql.functions.lit") as mock_lit:
            mock_lit.return_value = "lit_true"
            mock_df.withColumn.return_value = "results_table"

            result = conjointly._get_pyspark_results()

            # Should have printed the error messages
            assert mock_print.call_count >= 1


def test_check_column_has_nulls_attribute_error():
    """Test AttributeError handling in null checking."""

    # Create a mock table without null_count method
    mock_table = Mock()
    del mock_table.select().null_count  # Remove null_count method

    # Mock the select().collect() scenario for LazyFrames
    mock_collected = Mock()
    mock_collected.null_count.return_value = {"test_col": [1]}
    mock_table.select.return_value.collect.return_value = mock_collected

    result = _column_has_null_values(mock_table, "test_col")
    assert result is True


def test_check_column_has_nulls_nested_exceptions():
    """Test nested exception handling in null checking."""

    # Create a mock that raises AttributeError for null_count
    mock_table = Mock()

    # Make standard null_count() method fail
    mock_select_result = Mock()
    del mock_select_result.null_count  # Remove null_count method to trigger AttributeError
    mock_table.select.return_value = mock_select_result

    # Make collect() also fail
    mock_select_result.collect.side_effect = Exception("Collect failed")

    # Mock Narwhals scenario that also fails
    with patch("pointblank._interrogation.nw") as mock_nw:
        mock_nw.col.return_value.is_null.return_value.sum.return_value.alias.return_value = (
            "null_expr"
        )
        mock_table.select.side_effect = [mock_select_result, Exception("Select failed")]

        result = _column_has_null_values(mock_table, "test_col")
        assert result is False  # Last resort returns False


def test_modify_datetime_column_isinstance_check():
    """Test the isinstance check in the _modify_datetime_compare_val() function."""

    mock_column = Mock()
    mock_column.dtype = "datetime64[ns]"

    # Create a Column instance to test the isinstance check
    column_instance = Column("test")

    # This should return the column instance itself
    result = _modify_datetime_compare_val(mock_column, column_instance)
    assert result == column_instance


def test_safe_is_nan_or_null_expr_with_schema_attribute():
    """Test _safe_is_nan_or_null_expr with an eager DataFrame and float column."""
    import narwhals as nw

    df = nw.from_native(pl.DataFrame({"float_col": [1.0, float("nan"), None]}))
    col_expr = nw.col("float_col")

    result = _safe_is_nan_or_null_expr(df, col_expr, "float_col")

    evaluated = df.select(result).to_native()["float_col"].to_list()
    assert evaluated == [False, True, True]


def test_safe_is_nan_or_null_expr_schema_non_numeric():
    """String columns should only get null checks, not NaN checks."""
    import narwhals as nw

    df = nw.from_native(pl.DataFrame({"str_col": ["a", None, "b"]}))
    col_expr = nw.col("str_col")
    result = _safe_is_nan_or_null_expr(df, col_expr, "str_col")

    evaluated = df.select(result).to_native()["str_col"].to_list()
    assert evaluated == [False, True, False]


def test_safe_is_nan_or_null_expr_schema_is_nan_fails():
    """Test _safe_is_nan_or_null_expr falls back to null-only for string columns."""
    import narwhals as nw

    df = nw.from_native(pl.DataFrame({"str_col": ["a", None, "b"]}))
    col_expr = nw.col("str_col")

    result = _safe_is_nan_or_null_expr(df, col_expr, "str_col")

    evaluated = df.select(result).to_native()["str_col"].to_list()
    assert evaluated == [False, True, False]


def test_safe_is_nan_or_null_expr_lazy_polars():
    """Works with a Polars LazyFrame."""
    import narwhals as nw

    df = nw.from_native(pl.DataFrame({"x": [1.0, float("nan"), None]}).lazy())
    col_expr = nw.col("x")
    result = _safe_is_nan_or_null_expr(df, col_expr, "x")

    evaluated = df.select(result).collect().to_native()["x"].to_list()
    assert evaluated == [False, True, True]


def test_safe_is_nan_or_null_expr_eager_polars():
    """Works with an eager Polars DataFrame."""
    import narwhals as nw

    df = nw.from_native(pl.DataFrame({"x": [1.0, float("nan"), None]}))
    col_expr = nw.col("x")
    result = _safe_is_nan_or_null_expr(df, col_expr, "x")

    evaluated = df.select(result).to_native()["x"].to_list()
    assert evaluated == [False, True, True]


def test_safe_is_nan_or_null_expr_eager_pandas():
    """Works with an eager Pandas DataFrame."""
    import narwhals as nw

    df = nw.from_native(pd.DataFrame({"x": [1.0, float("nan"), None]}))
    col_expr = nw.col("x")
    result = _safe_is_nan_or_null_expr(df, col_expr, "x")

    evaluated = df.select(result).to_native()["x"].tolist()
    assert evaluated == [False, True, True]


def test_safe_is_nan_or_null_expr_ibis_sqlite():
    """Works with an Ibis SQLite backend (should only do null checks)."""
    import ibis
    import narwhals as nw

    con = ibis.sqlite.connect()
    t = con.create_table("test", pd.DataFrame({"x": [1.0, None, 3.0]}))
    nw_tbl = nw.from_native(t)
    col_expr = nw.col("x")

    result = _safe_is_nan_or_null_expr(nw_tbl, col_expr, "x")

    evaluated = nw_tbl.select(result).to_native().to_pandas()["x"].tolist()
    assert evaluated == [False, True, False]


def test_safe_modify_datetime_lazy_polars():
    """Works with a Polars LazyFrame."""
    import datetime
    import narwhals as nw

    df = nw.from_native(pl.DataFrame({"d": [datetime.date(2023, 6, 1)]}).lazy())
    result = _safe_modify_datetime_compare_val(df, "d", datetime.datetime(2023, 1, 1, 12, 0))

    assert isinstance(result, datetime.date)
    assert not isinstance(result, datetime.datetime)


def test_safe_modify_datetime_eager_pandas():
    """Works with an eager Pandas DataFrame."""
    import datetime
    import narwhals as nw

    df = nw.from_native(pd.DataFrame({"d": [datetime.date(2023, 6, 1)]}))
    result = _safe_modify_datetime_compare_val(df, "d", datetime.datetime(2023, 1, 1, 12, 0))

    assert isinstance(result, datetime.date)
    assert not isinstance(result, datetime.datetime)


if __name__ == "__main__":
    pytest.main([__file__])
