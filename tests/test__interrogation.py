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
from pointblank.column import Column, col


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
    """Test that _column_has_null_values returns False when select raises an exception."""

    # Create a mock table that raises when select() is called
    mock_table = Mock()
    mock_table.select.side_effect = Exception("Unsupported operation")

    result = _column_has_null_values(mock_table, "test_col")
    assert result is False


def test_check_column_has_nulls_nested_exceptions():
    """Test nested exception handling in null checking."""

    # Create a mock table that raises when select() is called
    mock_table = Mock()
    mock_table.select.side_effect = Exception("Select failed")

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


class TestTblMatch:
    """Tests for the tbl_match() function."""

    def test_matching_pandas_dataframes(self):
        from pointblank._interrogation import tbl_match

        df1 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert tbl_match(df1, df2) is True

    def test_matching_polars_dataframes(self):
        from pointblank._interrogation import tbl_match

        df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        df2 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        assert tbl_match(df1, df2) is True

    def test_different_values(self):
        from pointblank._interrogation import tbl_match

        df1 = pd.DataFrame({"a": [1, 2, 3]})
        df2 = pd.DataFrame({"a": [1, 2, 99]})
        assert tbl_match(df1, df2) is False

    def test_different_column_count(self):
        from pointblank._interrogation import tbl_match

        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"a": [1]})
        assert tbl_match(df1, df2) is False

    def test_different_row_count(self):
        from pointblank._interrogation import tbl_match

        df1 = pd.DataFrame({"a": [1, 2]})
        df2 = pd.DataFrame({"a": [1]})
        assert tbl_match(df1, df2) is False

    def test_different_column_names(self):
        from pointblank._interrogation import tbl_match

        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"b": [1]})
        assert tbl_match(df1, df2) is False

    def test_different_column_order(self):
        from pointblank._interrogation import tbl_match

        df1 = pd.DataFrame({"a": [1], "b": [2]})
        df2 = pd.DataFrame({"b": [2], "a": [1]})
        assert tbl_match(df1, df2) is False

    def test_nan_values_match(self):
        from pointblank._interrogation import tbl_match

        df1 = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
        df2 = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
        assert tbl_match(df1, df2) is True

    def test_none_values_match(self):
        from pointblank._interrogation import tbl_match

        df1 = pl.DataFrame({"a": [1, None, 3]})
        df2 = pl.DataFrame({"a": [1, None, 3]})
        assert tbl_match(df1, df2) is True

    def test_none_vs_value_mismatch(self):
        from pointblank._interrogation import tbl_match

        df1 = pl.DataFrame({"a": [1, None, 3]})
        df2 = pl.DataFrame({"a": [1, 2, 3]})
        assert tbl_match(df1, df2) is False

    def test_empty_dataframes_match(self):
        from pointblank._interrogation import tbl_match

        df1 = pd.DataFrame({"a": pd.Series([], dtype=int)})
        df2 = pd.DataFrame({"a": pd.Series([], dtype=int)})
        assert tbl_match(df1, df2) is True

    def test_cross_backend_pandas_polars(self):
        from pointblank._interrogation import tbl_match

        df_pd = pd.DataFrame({"a": [1, 2, 3]})
        df_pl = pl.DataFrame({"a": [1, 2, 3]})
        assert tbl_match(df_pd, df_pl) is True

    def test_string_columns(self):
        from pointblank._interrogation import tbl_match

        df1 = pl.DataFrame({"s": ["hello", "world"]})
        df2 = pl.DataFrame({"s": ["hello", "world"]})
        assert tbl_match(df1, df2) is True

    def test_string_columns_mismatch(self):
        from pointblank._interrogation import tbl_match

        df1 = pl.DataFrame({"s": ["hello", "world"]})
        df2 = pl.DataFrame({"s": ["hello", "earth"]})
        assert tbl_match(df1, df2) is False


class TestInterrogateNe:
    """Tests for interrogate_ne() null-handling paths."""

    def test_ne_no_nulls_column_compare_polars(self):
        from pointblank._interrogation import interrogate_ne

        df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 99, 3]})
        result = interrogate_ne(df, "a", col("b"), na_pass=False)
        good = result["pb_is_good_"].to_list()
        assert good == [False, True, False]

    def test_ne_no_nulls_literal_polars(self):
        from pointblank._interrogation import interrogate_ne

        df = pl.DataFrame({"a": [1, 2, 3]})
        result = interrogate_ne(df, "a", 2, na_pass=False)
        good = result["pb_is_good_"].to_list()
        assert good == [True, False, True]

    def test_ne_ref_nulls_column_compare_na_pass_false_polars(self):
        """CASE 1: ref column has nulls, compare does not, na_pass=False."""
        from pointblank._interrogation import interrogate_ne

        df = pl.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
        result = interrogate_ne(df, "a", col("b"), na_pass=False)
        good = result["pb_is_good_"].to_list()
        assert good[0] is False
        assert good[1] is False
        assert good[2] is False

    def test_ne_ref_nulls_column_compare_na_pass_true_polars(self):
        """CASE 1: ref column has nulls, compare does not, na_pass=True."""
        from pointblank._interrogation import interrogate_ne

        df = pl.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
        result = interrogate_ne(df, "a", col("b"), na_pass=True)
        good = result["pb_is_good_"].to_list()
        assert good[0] is False
        assert good[1] is True
        assert good[2] is False

    def test_ne_compare_nulls_column_compare_na_pass_false_polars(self):
        """CASE 2: compare column has nulls, ref does not, na_pass=False."""
        from pointblank._interrogation import interrogate_ne

        df = pl.DataFrame({"a": [1, 2, 3], "b": [1, None, 3]})
        result = interrogate_ne(df, "a", col("b"), na_pass=False)
        good = result["pb_is_good_"].to_list()
        assert good[0] is False
        assert good[2] is False

    def test_ne_compare_nulls_column_compare_na_pass_true_polars(self):
        """CASE 2: compare column has nulls, ref does not, na_pass=True."""
        from pointblank._interrogation import interrogate_ne

        df = pl.DataFrame({"a": [1, 2, 3], "b": [1, None, 3]})
        result = interrogate_ne(df, "a", col("b"), na_pass=True)
        good = result["pb_is_good_"].to_list()
        assert good[0] is False
        assert good[1] is True
        assert good[2] is False

    def test_ne_both_nulls_column_compare_na_pass_false_polars(self):
        """CASE 3: both columns have nulls, na_pass=False."""
        from pointblank._interrogation import interrogate_ne

        df = pl.DataFrame({"a": [1, None, None], "b": [None, 2, None]})
        result = interrogate_ne(df, "a", col("b"), na_pass=False)
        good = result["pb_is_good_"].to_list()
        assert good[0] is False
        assert good[1] is False
        assert good[2] is False

    def test_ne_both_nulls_column_compare_na_pass_true_polars(self):
        """CASE 3: both columns have nulls, na_pass=True."""
        from pointblank._interrogation import interrogate_ne

        df = pl.DataFrame({"a": [1, None, None], "b": [None, 2, None]})
        result = interrogate_ne(df, "a", col("b"), na_pass=True)
        good = result["pb_is_good_"].to_list()
        assert good[0] is True
        assert good[1] is True
        assert good[2] is True

    def test_ne_ref_nulls_literal_na_pass_false_pandas(self):
        """Ref column has nulls, literal compare, na_pass=False, Pandas."""
        from pointblank._interrogation import interrogate_ne

        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        result = interrogate_ne(df, "a", 2.0, na_pass=False)
        good = result["pb_is_good_"].tolist()
        assert good[0] is True
        assert good[1] is False
        assert good[2] is True

    def test_ne_ref_nulls_literal_na_pass_true_pandas(self):
        """Ref column has nulls, literal compare, na_pass=True, Pandas."""
        from pointblank._interrogation import interrogate_ne

        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        result = interrogate_ne(df, "a", 2.0, na_pass=True)
        good = result["pb_is_good_"].tolist()
        assert good[0] is True
        assert good[1] is True
        assert good[2] is True

    def test_ne_ref_nulls_literal_na_pass_false_polars(self):
        """Ref column has nulls, literal compare, na_pass=False, Polars."""
        from pointblank._interrogation import interrogate_ne

        df = pl.DataFrame({"a": [1, None, 3]})
        result = interrogate_ne(df, "a", 2, na_pass=False)
        good = result["pb_is_good_"].to_list()
        assert good[0] is True
        assert good[1] is False
        assert good[2] is True

    def test_ne_ref_nulls_literal_na_pass_true_polars(self):
        """Ref column has nulls, literal compare, na_pass=True, Polars."""
        from pointblank._interrogation import interrogate_ne

        df = pl.DataFrame({"a": [1, None, 3]})
        result = interrogate_ne(df, "a", 2, na_pass=True)
        good = result["pb_is_good_"].to_list()
        assert good[0] is True
        assert good[1] is True
        assert good[2] is True

    def test_ne_ref_nulls_column_compare_na_pass_false_pandas(self):
        """CASE 1 on Pandas: ref has nulls, compare does not."""
        from pointblank._interrogation import interrogate_ne

        df = pd.DataFrame({"a": [1.0, None, 3.0], "b": [1.0, 2.0, 3.0]})
        result = interrogate_ne(df, "a", col("b"), na_pass=False)
        good = result["pb_is_good_"].tolist()
        assert good[0] is False
        assert good[2] is False

    def test_ne_compare_nulls_column_compare_pandas(self):
        """CASE 2 on Pandas: compare column has nulls."""
        from pointblank._interrogation import interrogate_ne

        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [1.0, None, 3.0]})
        result = interrogate_ne(df, "a", col("b"), na_pass=False)
        good = result["pb_is_good_"].tolist()
        assert good[1] is False

    def test_ne_both_nulls_column_compare_pandas(self):
        """CASE 3 on Pandas: both columns have nulls."""
        from pointblank._interrogation import interrogate_ne

        df = pd.DataFrame({"a": [1.0, None, None], "b": [None, 2.0, None]})
        result = interrogate_ne(df, "a", col("b"), na_pass=True)
        good = result["pb_is_good_"].tolist()
        assert good[0] is True
        assert good[1] is True
        assert good[2] is True


class TestColPctMissing:
    """Tests for col_pct_missing()."""

    def test_no_missing_values(self):
        from pointblank._interrogation import col_pct_missing

        df = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
        assert col_pct_missing(df, "a", sentinels=[], count_null=True, max_pct=0.5) is True

    def test_all_missing_sentinels(self):
        from pointblank._interrogation import col_pct_missing

        df = pd.DataFrame({"a": [-99, -99, -99, -99, -99]})
        assert col_pct_missing(df, "a", sentinels=[-99], count_null=False, max_pct=0.5) is False

    def test_mixed_sentinels_and_nulls(self):
        from pointblank._interrogation import col_pct_missing

        df = pd.DataFrame({"a": [1.0, -99.0, None, 4.0, 5.0]})
        result = col_pct_missing(df, "a", sentinels=[-99.0], count_null=True, max_pct=0.5)
        assert result is True

    def test_sentinel_only_no_null_counting(self):
        from pointblank._interrogation import col_pct_missing

        df = pd.DataFrame({"a": [1.0, -99.0, None, 4.0, 5.0]})
        result = col_pct_missing(df, "a", sentinels=[-99.0], count_null=False, max_pct=0.3)
        assert result is True

    def test_null_only_no_sentinels(self):
        from pointblank._interrogation import col_pct_missing

        df = pd.DataFrame({"a": [1.0, None, None, 4.0, 5.0]})
        result = col_pct_missing(df, "a", sentinels=[], count_null=True, max_pct=0.3)
        assert result is False

    def test_empty_sentinels_no_null_counting(self):
        from pointblank._interrogation import col_pct_missing

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = col_pct_missing(df, "a", sentinels=[], count_null=False, max_pct=0.0)
        assert result is True

    def test_empty_table(self):
        from pointblank._interrogation import col_pct_missing

        df = pd.DataFrame({"a": pd.Series([], dtype=float)})
        result = col_pct_missing(df, "a", sentinels=[-99], count_null=True, max_pct=0.0)
        assert result is True

    def test_polars_lazy_frame(self):
        from pointblank._interrogation import col_pct_missing

        df = pl.DataFrame({"a": [1, None, 3, None, 5]}).lazy()
        result = col_pct_missing(df, "a", sentinels=[], count_null=True, max_pct=0.5)
        assert result is True


class TestInterrogateWithinSpec:
    """Tests for interrogate_within_spec() checksum-based specs."""

    def test_isbn_valid(self):
        from pointblank._interrogation import interrogate_within_spec

        df = pd.DataFrame({"isbn": ["978-0-306-40615-7", "0-306-40615-2"]})
        result = interrogate_within_spec(
            df, "isbn", {"spec": "isbn"}, na_pass=False
        )
        assert "pb_is_good_" in result.columns

    def test_isbn_with_null_na_pass(self):
        from pointblank._interrogation import interrogate_within_spec

        df = pd.DataFrame({"isbn": ["978-0-306-40615-7", None]})
        result = interrogate_within_spec(
            df, "isbn", {"spec": "isbn"}, na_pass=True
        )
        good = result["pb_is_good_"].tolist()
        assert good[1] is True

    def test_email_regex_spec(self):
        from pointblank._interrogation import interrogate_within_spec

        df = pl.DataFrame({"email": ["test@example.com", "not-an-email", None]})
        result = interrogate_within_spec(
            df, "email", {"spec": "email"}, na_pass=False
        )
        good = result["pb_is_good_"].to_list()
        assert good[0] is True
        assert good[1] is False
        assert good[2] is False

    def test_email_regex_spec_na_pass(self):
        from pointblank._interrogation import interrogate_within_spec

        df = pl.DataFrame({"email": ["test@example.com", None]})
        result = interrogate_within_spec(
            df, "email", {"spec": "email"}, na_pass=True
        )
        good = result["pb_is_good_"].to_list()
        assert good[0] is True
        assert good[1] is True

    def test_url_spec(self):
        from pointblank._interrogation import interrogate_within_spec

        df = pd.DataFrame({"url": ["https://example.com", "not-a-url"]})
        result = interrogate_within_spec(
            df, "url", {"spec": "url"}, na_pass=False
        )
        good = result["pb_is_good_"].tolist()
        assert good[0] is True
        assert good[1] is False

    def test_ipv4_spec(self):
        from pointblank._interrogation import interrogate_within_spec

        df = pl.DataFrame({"ip": ["192.168.1.1", "999.999.999.999"]})
        result = interrogate_within_spec(
            df, "ip", {"spec": "ipv4"}, na_pass=False
        )
        good = result["pb_is_good_"].to_list()
        assert good[0] is True

    def test_unknown_spec_raises(self):
        from pointblank._interrogation import interrogate_within_spec

        df = pd.DataFrame({"a": ["x"]})
        with pytest.raises(ValueError, match="Unknown specification type"):
            interrogate_within_spec(df, "a", {"spec": "bogus_spec"}, na_pass=False)

    def test_postal_code_without_country_raises(self):
        from pointblank._interrogation import interrogate_within_spec

        df = pd.DataFrame({"zip": ["12345"]})
        with pytest.raises(ValueError, match="Country code required"):
            interrogate_within_spec(df, "zip", {"spec": "postal_code"}, na_pass=False)

    def test_isbn_polars(self):
        from pointblank._interrogation import interrogate_within_spec

        df = pl.DataFrame({"isbn": ["978-0-306-40615-7", "invalid-isbn"]})
        result = interrogate_within_spec(
            df, "isbn", {"spec": "isbn"}, na_pass=False
        )
        good = result["pb_is_good_"].to_list()
        assert good[0] is True
        assert good[1] is False

    def test_swift_bic_spec(self):
        from pointblank._interrogation import interrogate_within_spec

        df = pd.DataFrame({"bic": ["DEUTDEFF", "INVALID"]})
        result = interrogate_within_spec(
            df, "bic", {"spec": "swift_bic"}, na_pass=False
        )
        assert "pb_is_good_" in result.columns

    def test_spec_with_country_bracket_syntax(self):
        from pointblank._interrogation import interrogate_within_spec

        df = pd.DataFrame({"iban": ["DE89370400440532013000"]})
        result = interrogate_within_spec(
            df, "iban", {"spec": "iban[DE]"}, na_pass=False
        )
        assert "pb_is_good_" in result.columns


class TestDataFreshness:
    """Tests for data_freshness() and _is_datetime_aware()."""

    def test_fresh_data_passes(self):
        import datetime
        from pointblank._interrogation import data_freshness

        now = datetime.datetime.now()
        df = pd.DataFrame({"ts": [now - datetime.timedelta(minutes=5)]})
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=1),
            reference_time=None,
            timezone=None,
            allow_tz_mismatch=False,
        )
        assert result["passed"] is True

    def test_stale_data_fails(self):
        import datetime
        from pointblank._interrogation import data_freshness

        old_time = datetime.datetime(2020, 1, 1)
        df = pd.DataFrame({"ts": [old_time]})
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=1),
            reference_time=datetime.datetime.now(),
            timezone=None,
            allow_tz_mismatch=False,
        )
        assert result["passed"] is False

    def test_explicit_reference_time(self):
        import datetime
        from pointblank._interrogation import data_freshness

        ref = datetime.datetime(2023, 6, 15, 12, 0, 0)
        data_time = datetime.datetime(2023, 6, 15, 11, 0, 0)
        df = pd.DataFrame({"ts": [data_time]})
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=2),
            reference_time=ref,
            timezone=None,
            allow_tz_mismatch=False,
        )
        assert result["passed"] is True
        assert result["age"] == datetime.timedelta(hours=1)

    def test_empty_column(self):
        import datetime
        from pointblank._interrogation import data_freshness

        df = pl.DataFrame({"ts": pl.Series("ts", [None], dtype=pl.Datetime)})
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=1),
            reference_time=None,
            timezone=None,
            allow_tz_mismatch=False,
        )
        assert result["passed"] is False
        assert result["column_empty"] is True

    def test_timezone_aware_data_naive_ref(self):
        import datetime
        from pointblank._interrogation import data_freshness

        aware_time = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
        df = pd.DataFrame({"ts": [aware_time]})
        naive_ref = datetime.datetime(2023, 6, 15, 13, 0, 0)
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=2),
            reference_time=naive_ref,
            timezone=None,
            allow_tz_mismatch=False,
        )
        assert result.get("tz_warning_key") is not None

    def test_timezone_naive_data_aware_ref(self):
        import datetime
        from pointblank._interrogation import data_freshness

        naive_time = datetime.datetime(2023, 6, 15, 12, 0, 0)
        df = pd.DataFrame({"ts": [naive_time]})
        aware_ref = datetime.datetime(2023, 6, 15, 13, 0, 0, tzinfo=datetime.timezone.utc)
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=2),
            reference_time=aware_ref,
            timezone=None,
            allow_tz_mismatch=False,
        )
        assert result.get("tz_warning_key") is not None

    def test_timezone_with_offset_string(self):
        import datetime
        from pointblank._interrogation import data_freshness

        now = datetime.datetime.now(datetime.timezone.utc)
        df = pd.DataFrame({"ts": [now - datetime.timedelta(minutes=10)]})
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=1),
            reference_time=None,
            timezone="-5",
            allow_tz_mismatch=True,
        )
        assert "passed" in result

    def test_timezone_with_iana_name(self):
        import datetime
        from pointblank._interrogation import data_freshness

        now = datetime.datetime.now(datetime.timezone.utc)
        df = pd.DataFrame({"ts": [now - datetime.timedelta(minutes=10)]})
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=1),
            reference_time=None,
            timezone="US/Eastern",
            allow_tz_mismatch=True,
        )
        assert "passed" in result

    def test_polars_input(self):
        import datetime
        from pointblank._interrogation import data_freshness

        now = datetime.datetime.now()
        df = pl.DataFrame({"ts": [now - datetime.timedelta(minutes=5)]})
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=1),
            reference_time=None,
            timezone=None,
            allow_tz_mismatch=False,
        )
        assert result["passed"] is True

    def test_allow_tz_mismatch_suppresses_warning(self):
        import datetime
        from pointblank._interrogation import data_freshness

        aware_time = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
        df = pd.DataFrame({"ts": [aware_time]})
        naive_ref = datetime.datetime(2023, 6, 15, 13, 0, 0)
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=2),
            reference_time=naive_ref,
            timezone=None,
            allow_tz_mismatch=True,
        )
        assert result.get("tz_warning_key") is None

    def test_aware_data_no_tz_no_ref(self):
        """Data is tz-aware, no timezone specified, ref_time=None -> UTC ref."""
        import datetime
        from pointblank._interrogation import data_freshness

        now_utc = datetime.datetime.now(datetime.timezone.utc)
        df = pd.DataFrame({"ts": [now_utc - datetime.timedelta(minutes=5)]})
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=1),
            reference_time=None,
            timezone=None,
            allow_tz_mismatch=False,
        )
        assert result["passed"] is True

    def test_aware_data_naive_ref_with_timezone(self):
        """Data is tz-aware, ref is naive, timezone provided -> ref gets tz."""
        import datetime
        from pointblank._interrogation import data_freshness

        aware_time = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
        df = pd.DataFrame({"ts": [aware_time]})
        naive_ref = datetime.datetime(2023, 6, 15, 13, 0, 0)
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=2),
            reference_time=naive_ref,
            timezone="US/Eastern",
            allow_tz_mismatch=True,
        )
        assert "passed" in result

    def test_naive_data_aware_ref_with_timezone(self):
        """Data is naive, ref is tz-aware, timezone provided -> data gets tz."""
        import datetime
        from pointblank._interrogation import data_freshness

        naive_time = datetime.datetime(2023, 6, 15, 12, 0, 0)
        df = pd.DataFrame({"ts": [naive_time]})
        aware_ref = datetime.datetime(2023, 6, 15, 13, 0, 0, tzinfo=datetime.timezone.utc)
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=2),
            reference_time=aware_ref,
            timezone="US/Eastern",
            allow_tz_mismatch=True,
        )
        assert "passed" in result

    def test_naive_data_aware_ref_no_timezone(self):
        """Data is naive, ref is tz-aware, no timezone -> ref stripped of tz."""
        import datetime
        from pointblank._interrogation import data_freshness

        naive_time = datetime.datetime(2023, 6, 15, 12, 0, 0)
        df = pd.DataFrame({"ts": [naive_time]})
        aware_ref = datetime.datetime(2023, 6, 15, 13, 0, 0, tzinfo=datetime.timezone.utc)
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=2),
            reference_time=aware_ref,
            timezone=None,
            allow_tz_mismatch=True,
        )
        assert result["passed"] is True

    def test_naive_data_with_timezone_no_ref(self):
        """Data is naive, timezone specified, ref=None -> ref uses timezone."""
        import datetime
        from pointblank._interrogation import data_freshness

        now = datetime.datetime.now()
        df = pd.DataFrame({"ts": [now - datetime.timedelta(minutes=5)]})
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=1),
            reference_time=None,
            timezone="-5",
            allow_tz_mismatch=True,
        )
        assert "passed" in result

    def test_aware_data_naive_ref_no_timezone(self):
        """Data is tz-aware, ref is naive, no timezone -> ref gets UTC."""
        import datetime
        from pointblank._interrogation import data_freshness

        aware_time = datetime.datetime(2023, 6, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
        df = pd.DataFrame({"ts": [aware_time]})
        naive_ref = datetime.datetime(2023, 6, 15, 13, 0, 0)
        result = data_freshness(
            df, "ts",
            max_age=datetime.timedelta(hours=2),
            reference_time=naive_ref,
            timezone=None,
            allow_tz_mismatch=True,
        )
        assert result["passed"] is True


class TestIsDatetimeAware:
    """Tests for _is_datetime_aware()."""

    def test_none_is_not_aware(self):
        from pointblank._interrogation import _is_datetime_aware

        assert _is_datetime_aware(None) is False

    def test_naive_datetime(self):
        import datetime
        from pointblank._interrogation import _is_datetime_aware

        dt = datetime.datetime(2023, 1, 1)
        assert _is_datetime_aware(dt) is False

    def test_aware_datetime(self):
        import datetime
        from pointblank._interrogation import _is_datetime_aware

        dt = datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc)
        assert _is_datetime_aware(dt) is True

    def test_non_datetime_object(self):
        from pointblank._interrogation import _is_datetime_aware

        assert _is_datetime_aware("not a datetime") is False

    def test_integer_is_not_aware(self):
        from pointblank._interrogation import _is_datetime_aware

        assert _is_datetime_aware(42) is False


class TestInterrogateStrLen:
    """Tests for interrogate_str_len()."""

    def test_min_only(self):
        from pointblank._interrogation import interrogate_str_len

        df = pd.DataFrame({"s": ["ab", "abcd", "a"]})
        result = interrogate_str_len(df, "s", {"min_val": 2, "max_val": None}, na_pass=False)
        good = result["pb_is_good_"].tolist()
        assert good == [True, True, False]

    def test_max_only(self):
        from pointblank._interrogation import interrogate_str_len

        df = pd.DataFrame({"s": ["ab", "abcd", "a"]})
        result = interrogate_str_len(df, "s", {"min_val": None, "max_val": 3}, na_pass=False)
        good = result["pb_is_good_"].tolist()
        assert good == [True, False, True]

    def test_min_and_max(self):
        from pointblank._interrogation import interrogate_str_len

        df = pl.DataFrame({"s": ["ab", "abcde", "abc"]})
        result = interrogate_str_len(df, "s", {"min_val": 2, "max_val": 4}, na_pass=False)
        good = result["pb_is_good_"].to_list()
        assert good == [True, False, True]

    def test_na_pass(self):
        from pointblank._interrogation import interrogate_str_len

        df = pl.DataFrame({"s": ["abc", None]})
        result = interrogate_str_len(df, "s", {"min_val": 2, "max_val": 5}, na_pass=True)
        good = result["pb_is_good_"].to_list()
        assert good[1] is True


class TestInterrogateRegex:
    """Tests for interrogate_regex()."""

    def test_regex_string_format(self):
        from pointblank._interrogation import interrogate_regex

        df = pd.DataFrame({"s": ["abc", "def", "ghi"]})
        result = interrogate_regex(df, "s", "^a", na_pass=False)
        good = result["pb_is_good_"].tolist()
        assert good == [True, False, False]

    def test_regex_dict_format_with_inverse(self):
        from pointblank._interrogation import interrogate_regex

        df = pl.DataFrame({"s": ["abc", "def", "ghi"]})
        result = interrogate_regex(
            df, "s", {"pattern": "^a", "inverse": True}, na_pass=False
        )
        good = result["pb_is_good_"].to_list()
        assert good == [False, True, True]


class TestApplyMissingExclusion:
    """Tests for apply_missing_exclusion()."""

    def test_sentinel_exclusion(self):
        from pointblank._interrogation import apply_missing_exclusion

        df = pd.DataFrame({"a": [1, -99, 3], "pb_is_good_": [True, False, True]})
        spec = Mock()
        spec.sentinel_values.return_value = [-99]
        spec.null_is_missing = False

        result = apply_missing_exclusion(df, "a", spec)
        good = result["pb_is_good_"].tolist()
        assert good == [True, True, True]

    def test_null_is_missing_exclusion(self):
        from pointblank._interrogation import apply_missing_exclusion

        df = pd.DataFrame({"a": [1.0, None, 3.0], "pb_is_good_": [True, False, True]})
        spec = Mock()
        spec.sentinel_values.return_value = []
        spec.null_is_missing = True

        result = apply_missing_exclusion(df, "a", spec)
        good = result["pb_is_good_"].tolist()
        assert good == [True, True, True]

    def test_no_exclusion_criteria(self):
        from pointblank._interrogation import apply_missing_exclusion

        df = pd.DataFrame({"a": [1, 2, 3], "pb_is_good_": [True, False, True]})
        spec = Mock()
        spec.sentinel_values.return_value = []
        spec.null_is_missing = False

        result = apply_missing_exclusion(df, "a", spec)
        good = result["pb_is_good_"].tolist()
        assert good == [True, False, True]


class TestInterrogateMissingOnlyCoded:
    """Tests for interrogate_missing_only_coded()."""

    def test_sentinels_pass(self):
        from pointblank._interrogation import interrogate_missing_only_coded

        df = pd.DataFrame({"a": [1, -99, 5, -99]})
        result = interrogate_missing_only_coded(
            df, "a", sentinels=[-99], count_null=False,
            allowed=None, min_val=1, max_val=10,
        )
        good = result["pb_is_good_"].tolist()
        assert good == [True, True, True, True]

    def test_undocumented_code_fails(self):
        from pointblank._interrogation import interrogate_missing_only_coded

        df = pd.DataFrame({"a": [1, -99, 999, 5]})
        result = interrogate_missing_only_coded(
            df, "a", sentinels=[-99], count_null=False,
            allowed=None, min_val=1, max_val=10,
        )
        good = result["pb_is_good_"].tolist()
        assert good[2] is False

    def test_null_as_missing(self):
        from pointblank._interrogation import interrogate_missing_only_coded

        df = pd.DataFrame({"a": [1.0, None, 5.0]})
        result = interrogate_missing_only_coded(
            df, "a", sentinels=[], count_null=True,
            allowed=None, min_val=None, max_val=None,
        )
        good = result["pb_is_good_"].tolist()
        assert good[1] is True
        assert good[0] is False

    def test_allowed_values(self):
        from pointblank._interrogation import interrogate_missing_only_coded

        df = pd.DataFrame({"a": [1, 2, 3, 99]})
        result = interrogate_missing_only_coded(
            df, "a", sentinels=[], count_null=False,
            allowed=[1, 2, 3], min_val=None, max_val=None,
        )
        good = result["pb_is_good_"].tolist()
        assert good == [True, True, True, False]

    def test_no_criteria_all_fail(self):
        from pointblank._interrogation import interrogate_missing_only_coded

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = interrogate_missing_only_coded(
            df, "a", sentinels=[], count_null=False,
            allowed=None, min_val=None, max_val=None,
        )
        good = result["pb_is_good_"].tolist()
        assert good == [False, False, False]


class TestInterrogateMissingConsistent:
    """Tests for interrogate_missing_consistent()."""

    def test_consistent_all_present(self):
        from pointblank._interrogation import interrogate_missing_consistent

        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = interrogate_missing_consistent(
            df, columns=["a", "b"], sentinels=[-99], count_null=True,
        )
        good = result["pb_is_good_"].tolist()
        assert good == [True, True, True]

    def test_consistent_all_missing(self):
        from pointblank._interrogation import interrogate_missing_consistent

        df = pd.DataFrame({"a": [-99, -99], "b": [-99, -99]})
        result = interrogate_missing_consistent(
            df, columns=["a", "b"], sentinels=[-99], count_null=False,
        )
        good = result["pb_is_good_"].tolist()
        assert good == [True, True]

    def test_inconsistent_missing(self):
        from pointblank._interrogation import interrogate_missing_consistent

        df = pd.DataFrame({"a": [-99, 1], "b": [1, -99]})
        result = interrogate_missing_consistent(
            df, columns=["a", "b"], sentinels=[-99], count_null=False,
        )
        good = result["pb_is_good_"].tolist()
        assert good == [False, False]

    def test_null_counting(self):
        from pointblank._interrogation import interrogate_missing_consistent

        df = pd.DataFrame({"a": [None, 1.0], "b": [None, 2.0]})
        result = interrogate_missing_consistent(
            df, columns=["a", "b"], sentinels=[], count_null=True,
        )
        good = result["pb_is_good_"].tolist()
        assert good == [True, True]

    def test_no_sentinels_no_null(self):
        from pointblank._interrogation import interrogate_missing_consistent

        df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = interrogate_missing_consistent(
            df, columns=["a", "b"], sentinels=[], count_null=False,
        )
        good = result["pb_is_good_"].to_list()
        assert good == [True, True]


class TestInterrogateIncreasingDecreasing:
    """Tests for interrogate_increasing() and interrogate_decreasing()."""

    def test_strictly_increasing(self):
        from pointblank._interrogation import interrogate_increasing

        df = pd.DataFrame({"a": [1, 2, 3, 4]})
        result = interrogate_increasing(
            df, "a", allow_stationary=False, decreasing_tol=0, na_pass=False
        )
        good = result["pb_is_good_"].tolist()
        assert good[0] is True
        assert all(g is True for g in good[1:])

    def test_not_increasing(self):
        from pointblank._interrogation import interrogate_increasing

        df = pd.DataFrame({"a": [1, 3, 2, 4]})
        result = interrogate_increasing(
            df, "a", allow_stationary=False, decreasing_tol=0, na_pass=False
        )
        good = result["pb_is_good_"].tolist()
        assert good[2] is False

    def test_increasing_allow_stationary(self):
        from pointblank._interrogation import interrogate_increasing

        df = pd.DataFrame({"a": [1, 2, 2, 3]})
        result = interrogate_increasing(
            df, "a", allow_stationary=True, decreasing_tol=0, na_pass=False
        )
        good = result["pb_is_good_"].tolist()
        assert all(g is True for g in good)

    def test_increasing_with_tolerance(self):
        from pointblank._interrogation import interrogate_increasing

        df = pd.DataFrame({"a": [1, 3, 2.5, 4]})
        result = interrogate_increasing(
            df, "a", allow_stationary=False, decreasing_tol=1.0, na_pass=False
        )
        good = result["pb_is_good_"].tolist()
        assert good[2] is True

    def test_increasing_na_pass(self):
        from pointblank._interrogation import interrogate_increasing

        df = pd.DataFrame({"a": [1.0, None, 3.0, 4.0]})
        result = interrogate_increasing(
            df, "a", allow_stationary=False, decreasing_tol=0, na_pass=True
        )
        good = result["pb_is_good_"].tolist()
        assert good[1] is True

    def test_strictly_decreasing(self):
        from pointblank._interrogation import interrogate_decreasing

        df = pd.DataFrame({"a": [4, 3, 2, 1]})
        result = interrogate_decreasing(
            df, "a", allow_stationary=False, increasing_tol=0, na_pass=False
        )
        good = result["pb_is_good_"].tolist()
        assert all(g is True for g in good)

    def test_decreasing_with_tolerance(self):
        from pointblank._interrogation import interrogate_decreasing

        df = pd.DataFrame({"a": [4, 3, 3.5, 1]})
        result = interrogate_decreasing(
            df, "a", allow_stationary=False, increasing_tol=1.0, na_pass=False
        )
        good = result["pb_is_good_"].tolist()
        assert good[2] is True


class TestCoerceToCommonBackend:
    """Tests for _coerce_to_common_backend()."""

    def test_same_backend_no_conversion(self):
        from pointblank._interrogation import _coerce_to_common_backend

        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [2]})
        result1, result2 = _coerce_to_common_backend(df1, df2)
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)

    def test_polars_to_pandas_conversion(self):
        from pointblank._interrogation import _coerce_to_common_backend

        df_pd = pd.DataFrame({"a": [1, 2]})
        df_pl = pl.DataFrame({"a": [1, 2]})
        result1, result2 = _coerce_to_common_backend(df_pd, df_pl)
        assert isinstance(result1, pd.DataFrame)
        assert isinstance(result2, pd.DataFrame)

    def test_pandas_to_polars_conversion(self):
        from pointblank._interrogation import _coerce_to_common_backend

        df_pl = pl.DataFrame({"a": [1, 2]})
        df_pd = pd.DataFrame({"a": [1, 2]})
        result1, result2 = _coerce_to_common_backend(df_pl, df_pd)
        assert isinstance(result1, pl.DataFrame)
        assert isinstance(result2, pl.DataFrame)


class TestInterrogateInTable:
    """Tests for interrogate_in_table()."""

    def test_mismatched_column_lengths_raises(self):
        from pointblank._interrogation import interrogate_in_table

        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        ref = pd.DataFrame({"x": [1], "y": [2], "z": [3]})
        with pytest.raises(ValueError, match="same length"):
            interrogate_in_table(df, ["a", "b"], ref, ["x", "y", "z"], na_pass=False)


class TestInterrogateEqNe:
    """Tests for interrogate_eq with column-column comparison through null paths."""

    def test_eq_column_compare_polars(self):
        from pointblank._interrogation import interrogate_eq

        df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 99, 3]})
        result = interrogate_eq(df, "a", col("b"), na_pass=False)
        good = result["pb_is_good_"].to_list()
        assert good == [True, False, True]

    def test_eq_column_compare_with_nulls_na_pass_true(self):
        from pointblank._interrogation import interrogate_eq

        df = pl.DataFrame({"a": [1, None, 3], "b": [1, 2, None]})
        result = interrogate_eq(df, "a", col("b"), na_pass=True)
        good = result["pb_is_good_"].to_list()
        assert good[0] is True
        assert good[1] is True
        assert good[2] is True

    def test_eq_literal_compare(self):
        from pointblank._interrogation import interrogate_eq

        df = pd.DataFrame({"a": [1, 2, 3]})
        result = interrogate_eq(df, "a", 2, na_pass=False)
        good = result["pb_is_good_"].tolist()
        assert good == [False, True, False]


if __name__ == "__main__":
    pytest.main([__file__])
