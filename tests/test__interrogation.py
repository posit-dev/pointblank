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
    """Test using collect_schema method."""

    # Create a mock dataframe with collect_schema
    mock_df = Mock()
    mock_schema = {"date_col": "datetime64[ns]"}
    mock_df.collect_schema.return_value = mock_schema

    # Mock the _modify_datetime_compare_val function
    with patch("pointblank._interrogation._modify_datetime_compare_val") as mock_modify:
        mock_modify.return_value = "modified_value"

        result = _safe_modify_datetime_compare_val(mock_df, "date_col", "2023-01-01")

        assert result == "modified_value"
        mock_modify.assert_called_once()


def test_safe_modify_datetime_with_schema_attribute():
    """Test using schema attribute."""

    # Create a mock dataframe with schema attribute
    mock_df = Mock()
    del mock_df.collect_schema  # Remove collect_schema
    mock_df.schema = {"date_col": "datetime64[ns]"}

    with patch("pointblank._interrogation._modify_datetime_compare_val") as mock_modify:
        mock_modify.return_value = "modified_value"

        result = _safe_modify_datetime_compare_val(mock_df, "date_col", "2023-01-01")

        assert result == "modified_value"


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
    """Test _safe_is_nan_or_null_expr() using schema attribute."""

    # Create a mock dataframe with schema attribute
    mock_df = Mock(spec=["schema"])  # spec ensures only 'schema' attribute exists

    # Set up schema attribute with a float column
    mock_df.schema = {"float_col": "Float64"}

    # Create a mock column expression
    mock_col_expr = Mock()
    mock_is_null = Mock()
    mock_is_nan = Mock()
    mock_col_expr.is_null.return_value = mock_is_null
    mock_col_expr.is_nan.return_value = mock_is_nan

    # Mock the OR operation
    mock_is_null.__or__ = Mock(return_value="null_or_nan_check")

    # Call the function
    result = _safe_is_nan_or_null_expr(mock_df, mock_col_expr, "float_col")

    # Should have called is_nan() for numeric type
    assert result == "null_or_nan_check"
    mock_col_expr.is_null.assert_called_once()
    mock_col_expr.is_nan.assert_called_once()


def test_safe_is_nan_or_null_expr_schema_non_numeric():
    """Test _safe_is_nan_or_null_expr() with schema attribute for non-numeric column."""

    # Create a mock dataframe with schema attribute (but not collect_schema)
    mock_df = Mock(spec=["schema"])  # spec ensures only 'schema' attribute exists

    # Set up schema attribute with a string column (non-numeric)
    mock_df.schema = {"string_col": "String"}

    # Create a mock column expression
    mock_col_expr = Mock()
    mock_is_null_result = "null_check"
    mock_col_expr.is_null.return_value = mock_is_null_result

    # Call the function
    result = _safe_is_nan_or_null_expr(mock_df, mock_col_expr, "string_col")

    # Should only check for null (not NaN) for string column
    # The result is what is_null() returned (null_check = column_expr.is_null())
    assert result == mock_is_null_result
    mock_col_expr.is_null.assert_called_once()
    # is_nan should not be called for string column
    mock_col_expr.is_nan.assert_not_called()


def test_safe_is_nan_or_null_expr_schema_is_nan_fails():
    """Test _safe_is_nan_or_null_expr() when is_nan() raises exception."""

    # Create a mock dataframe with schema attribute
    mock_df = Mock(spec=["schema"])  # spec ensures only 'schema' attribute exists
    mock_df.schema = {"float_col": "Float64"}

    # Create a mock column expression where is_nan() fails
    mock_col_expr = Mock()
    mock_is_null_result = "null_check_only"
    mock_col_expr.is_null.return_value = mock_is_null_result
    mock_col_expr.is_nan.side_effect = Exception("is_nan not supported")

    # Call the function
    result = _safe_is_nan_or_null_expr(mock_df, mock_col_expr, "float_col")

    # Should fall back to null check only when is_nan() fails
    # The result is what is_null() returned (since null_check = column_expr.is_null())
    assert result == mock_is_null_result
    mock_col_expr.is_null.assert_called_once()
    mock_col_expr.is_nan.assert_called_once()
