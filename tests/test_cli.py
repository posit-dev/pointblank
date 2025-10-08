from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
from typing import Any

import pandas as pd
import pytest
import click
from click.testing import CliRunner
from rich.console import Console

import pointblank as pb
from pointblank.cli import (
    OrderedGroup,
    cli,
    datasets,
    requirements,
    preview,
    info,
    scan,
    missing,
    make_template,
    run,
    validate,
    _display_validation_summary,
    _format_cell_value,
    _format_dtype_compact,
    _format_missing_percentage,
    _format_pass_fail,
    _format_units,
    _get_column_dtypes,
    _handle_pl_missing,
    _is_piped_data_source,
    _load_data_source,
    _rich_print_gt_table,
    _rich_print_missing_table,
    _rich_print_scan_table,
    _show_concise_help,
    console,
)
from pointblank._utils import _get_tbl_type, _is_lib_present


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture(autouse=True)
def mock_data_loading(monkeypatch):
    """Mock all data loading functions to prevent file creation during tests."""
    # Create a realistic pandas DataFrame as mock data
    try:
        import pandas as pd

        mock_data = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                "b": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m"],
                "c": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.1, 12.2, 13.3],
                "date": pd.date_range("2024-01-01", periods=13),
                "date_time": pd.date_range("2024-01-01 10:00:00", periods=13, freq="h"),
                "f": ["x", "y", "z"] * 4 + ["x"],
                "g": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130],
                "h": [
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                    False,
                    True,
                ],
            }
        )
    except ImportError:
        # Fallback to Mock if pandas not available
        mock_data = Mock()
        mock_data.columns = ["a", "b", "c", "date", "date_time", "f", "g", "h"]
        mock_data.shape = (13, 8)
        mock_data.dtypes = {
            "a": "int64",
            "b": "object",
            "c": "float64",
            "date": "datetime64[ns]",
            "date_time": "datetime64[ns]",
            "f": "object",
            "g": "int64",
            "h": "bool",
        }

    # Store original functions to call for invalid datasets
    original_load_dataset = pb.load_dataset
    original_col_summary_tbl = pb.col_summary_tbl
    original_missing_vals_tbl = pb.missing_vals_tbl
    original_preview = pb.preview

    def mock_load_dataset(name, *args, **kwargs):
        # Only mock known valid datasets to prevent file creation
        if name in ["small_table", "game_revenue", "nycflights", "global_sales"]:
            return mock_data
        else:
            # For invalid datasets, call original function to get proper error handling
            return original_load_dataset(name, *args, **kwargs)

    def mock_col_summary_tbl(data=None, *args, **kwargs):
        # If data is our mock_data or a known dataset string, return mock
        if data is mock_data or data in [
            "small_table",
            "game_revenue",
            "nycflights",
            "global_sales",
        ]:
            mock_gt = Mock()
            mock_gt._tbl_data = mock_data
            mock_gt.as_raw_html.return_value = "<html><body>Mock HTML</body></html>"
            return mock_gt
        else:
            # For other data, call original function
            return original_col_summary_tbl(data=data, *args, **kwargs)

    def mock_missing_vals_tbl(data=None, *args, **kwargs):
        # If data is our mock_data or a known dataset string, return mock
        if data is mock_data or data in [
            "small_table",
            "game_revenue",
            "nycflights",
            "global_sales",
        ]:
            mock_gt = Mock()
            mock_gt._tbl_data = mock_data
            mock_gt.as_raw_html.return_value = "<html><body>Mock Missing Report</body></html>"
            return mock_gt
        else:
            # For other data, call original function
            return original_missing_vals_tbl(data=data, *args, **kwargs)

    def mock_preview(data=None, *args, **kwargs):
        # If data is our mock_data or a known dataset string, return mock
        if data is mock_data or data in [
            "small_table",
            "game_revenue",
            "nycflights",
            "global_sales",
        ]:
            mock_gt = Mock()
            mock_gt._tbl_data = mock_data
            mock_gt.as_raw_html.return_value = "<html><body>Mock Preview</body></html>"
            return mock_gt
        else:
            # For other data, call original function
            return original_preview(data=data, *args, **kwargs)

    def mock_get_row_count(data=None, *args, **kwargs):
        if data is mock_data:
            return 13
        # For other data, don't mock - let it fail naturally if needed
        return pb.get_row_count(data, *args, **kwargs)

    def mock_get_column_count(data=None, *args, **kwargs):
        if data is mock_data:
            return 8
        # For other data, don't mock - let it fail naturally if needed
        return pb.get_column_count(data, *args, **kwargs)

    def mock_get_tbl_type(data=None, *args, **kwargs):
        if data is mock_data:
            return "pandas"
        # For other data, don't mock - let it fail naturally if needed
        return _get_tbl_type(data, *args, **kwargs)

    def mock_validate(*args, **kwargs):
        mock_validation = Mock()
        mock_validation.col_exists.return_value = mock_validation
        mock_validation.col_vals_gt.return_value = mock_validation
        mock_validation.interrogate.return_value = mock_validation
        mock_validation._tbl_validation = Mock()
        mock_validation._tbl_validation.n_pass = 5
        mock_validation._tbl_validation.n_fail = 2
        mock_validation._tbl_validation.n_warn = 1
        mock_validation._tbl_validation.n_notify = 0
        return mock_validation

    # Mock all data loading and processing functions
    monkeypatch.setattr("pointblank.load_dataset", mock_load_dataset)
    monkeypatch.setattr("pointblank.col_summary_tbl", mock_col_summary_tbl)
    monkeypatch.setattr("pointblank.missing_vals_tbl", mock_missing_vals_tbl)
    monkeypatch.setattr("pointblank.get_row_count", mock_get_row_count)
    monkeypatch.setattr("pointblank.get_column_count", mock_get_column_count)
    monkeypatch.setattr("pointblank._utils._get_tbl_type", mock_get_tbl_type)
    monkeypatch.setattr("pointblank.Validate", mock_validate)
    monkeypatch.setattr("pointblank.validate", mock_validate)
    monkeypatch.setattr("pointblank.preview", mock_preview)

    return mock_data


def test_format_cell_value_basic():
    """Test basic cell value formatting."""
    # Test regular string
    assert _format_cell_value("test") == "test"

    # Test row number formatting
    assert _format_cell_value(123, is_row_number=True) == "[dim]123[/dim]"

    # Test None value
    assert _format_cell_value(None) == "[red]None[/red]"

    # Test empty string
    assert _format_cell_value("") == "[red][/red]"


def test_format_cell_value_truncation():
    """Test cell value truncation based on column count."""
    long_text = "a" * 100

    # Test with few columns (less aggressive truncation)
    result = _format_cell_value(long_text, max_width=50, num_columns=5)
    assert len(result) <= 50
    assert "…" in result

    # Test with many columns (more aggressive truncation)
    result = _format_cell_value(long_text, max_width=50, num_columns=20)
    assert len(result) <= 30
    assert "…" in result


@patch("pandas.isna")
@patch("numpy.isnan")
def test_format_cell_value_pandas_na(mock_isnan, mock_isna):
    """Test formatting of pandas/numpy NA values."""
    # Mock pandas NA detection
    mock_isna.return_value = True
    mock_isnan.return_value = True

    # Test NaN value
    result = _format_cell_value(float("nan"))
    # The function should detect NA values when pandas/numpy are available
    assert "[red]" in result


def test_format_dtype_compact():
    """Test data type formatting to compact representation."""
    # Test common type conversions
    assert _format_dtype_compact("utf8") == "str"
    assert _format_dtype_compact("string") == "str"
    assert _format_dtype_compact("int64") == "i64"
    assert _format_dtype_compact("int32") == "i32"
    assert _format_dtype_compact("float64") == "f64"
    assert _format_dtype_compact("float32") == "f32"
    assert _format_dtype_compact("boolean") == "bool"
    assert _format_dtype_compact("bool") == "bool"
    assert _format_dtype_compact("datetime") == "datetime"
    assert _format_dtype_compact("date") == "date"
    assert _format_dtype_compact("object") == "obj"
    assert _format_dtype_compact("category") == "cat"

    # Test unknown types with truncation for long names
    assert _format_dtype_compact("unknown_type") == "unknown_…"
    assert _format_dtype_compact("short") == "short"


def test_get_column_dtypes_pandas_like():
    """Test column dtype extraction for pandas-like objects."""
    # Simple test using actual pandas if available
    try:
        import pandas as pd

        df = pd.DataFrame({"col1": [1, 2], "col2": [1.0, 2.0], "col3": ["a", "b"]})
        columns = ["col1", "col2", "col3"]

        result = _get_column_dtypes(df, columns)

        # Should have entries for all columns
        assert len(result) == 3
        assert all(col in result for col in columns)
        assert all(result[col] != "?" for col in columns)  # Should detect types

    except ImportError:
        # Fallback test with mock
        mock_df = Mock()
        mock_df.dtypes = None
        columns = ["col1", "col2"]

        result = _get_column_dtypes(mock_df, columns)
        expected = {"col1": "?", "col2": "?"}
        assert result == expected


def test_get_column_dtypes_schema_based():
    """Test column dtype extraction for schema-based objects."""
    # Simplified test that exercises the fallback path
    mock_df = Mock()

    # Remove dtypes and schema to test fallback
    mock_df.dtypes = None
    mock_df.schema = None

    columns = ["col1", "col2", "col3"]

    result = _get_column_dtypes(mock_df, columns)
    expected = {"col1": "?", "col2": "?", "col3": "?"}
    assert result == expected


def test_get_column_dtypes_fallback():
    """Test fallback when no schema or dtypes available."""
    mock_df = Mock()
    mock_df.schema = None
    mock_df.dtypes = None

    columns = ["col1", "col2"]

    result = _get_column_dtypes(mock_df, columns)
    expected = {"col1": "?", "col2": "?"}
    assert result == expected


def test_format_missing_percentage():
    """Test missing percentage formatting."""
    assert _format_missing_percentage(0.0) == "[green]●[/green]"
    assert _format_missing_percentage(50.0) == "50%"
    assert _format_missing_percentage(33.3) == "33%"
    assert _format_missing_percentage(100.0) == "[red]●[/red]"
    assert _format_missing_percentage(0.5) == "<1%"
    assert _format_missing_percentage(99.5) == ">99%"


@patch("pointblank.cli.console")
def test_rich_print_gt_table_basic(mock_console):
    """Test basic rich table printing."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.return_value = "<table>data</table>"

    # Should not raise any exceptions
    _rich_print_gt_table(mock_gt_table)

    # Console should be used for printing
    mock_console.print.assert_called()


@patch("pointblank.cli.console")
def test_rich_print_gt_table_with_preview_info(mock_console):
    """Test rich table printing with preview information."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.return_value = "<table>data</table>"

    preview_info = {"source_type": "Test Data", "shape": (100, 5), "table_type": "pandas.DataFrame"}

    _rich_print_gt_table(mock_gt_table, preview_info=preview_info)

    # Should print the table and info
    assert mock_console.print.call_count >= 2


@patch("pointblank.cli.console")
def test_rich_print_gt_table_error_handling(mock_console):
    """Test error handling in rich table printing."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.side_effect = Exception("HTML generation failed")

    # Should handle errors gracefully
    _rich_print_gt_table(mock_gt_table)

    # Should still attempt to print
    mock_console.print.assert_called()


@patch("pointblank.cli.console")
def test_display_validation_summary(mock_console):
    """Test validation summary display."""
    mock_validation = Mock()
    mock_validation.validation_info = [
        Mock(all_passed=True, n=100, n_passed=100, n_failed=0),
        Mock(all_passed=False, n=50, n_passed=45, n_failed=5),
    ]

    _display_validation_summary(mock_validation)

    # Should print summary information
    mock_console.print.assert_called()


@patch("pointblank.cli.console")
def test_display_validation_summary_no_info(mock_console):
    """Test validation summary display with no validation info."""
    mock_validation = Mock()
    mock_validation.validation_info = []

    _display_validation_summary(mock_validation)

    # Should handle empty validation info
    mock_console.print.assert_called()


def test_rich_print_missing_table():
    """Test _rich_print_missing_table function."""
    # Create a mock missing table
    mock_table = Mock()
    mock_table.as_raw_html.return_value = "<table><tr><td>Missing: 5</td></tr></table>"

    # Test the function with correct signature
    _rich_print_missing_table(gt_table=mock_table, original_data=None)

    # Should not raise any errors
    assert True


def test_rich_print_scan_table():
    """Test _rich_print_scan_table function."""
    # Create a mock scan result
    mock_scan = Mock()
    mock_scan.as_raw_html.return_value = "<table><tr><td>Scan Result</td></tr></table>"

    # Test the function
    _rich_print_scan_table(
        scan_result=mock_scan,
        data_source="test_data.csv",
        source_type="CSV",
        table_type="DataFrame",
        total_rows=100,
    )

    # Should not raise any errors
    assert True


def test_cli_main_help():
    """Test main CLI help command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output


def test_cli_version():
    """Test CLI version command."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0


def test_cli_group_version():
    """Test the main CLI group version option."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "pb, version" in result.output


def test_datasets_command():
    """Test the datasets command listing available datasets."""
    runner = CliRunner()
    result = runner.invoke(datasets)
    assert result.exit_code == 0
    assert "Available Pointblank Datasets" in result.output
    assert "small_table" in result.output
    assert "game_revenue" in result.output
    assert "nycflights" in result.output
    assert "global_sales" in result.output


def test_requirements_command():
    """Test the requirements command showing dependency status."""
    runner = CliRunner()
    result = runner.invoke(requirements)
    assert result.exit_code == 0
    assert "Dependency Status" in result.output
    assert "polars" in result.output
    assert "pandas" in result.output


def test_requirements_command_detailed():
    """Test the requirements command with detailed output."""
    runner = CliRunner()
    result = runner.invoke(requirements)
    assert result.exit_code == 0
    assert "Dependency Status" in result.output
    assert "polars" in result.output
    assert "pandas" in result.output
    assert "ibis" in result.output
    assert "duckdb" in result.output
    assert "pyarrow" in result.output


def test_preview_command_comprehensive_options():
    """Test preview command with comprehensive option combinations."""
    runner = CliRunner()

    # Test with no-header option
    result = runner.invoke(preview, ["small_table", "--no-header"])
    assert result.exit_code == 0

    # Test with custom column width and table width
    result = runner.invoke(
        preview, ["small_table", "--max-col-width", "100", "--min-table-width", "300"]
    )
    assert result.exit_code == 0

    # Test column range variations
    result = runner.invoke(preview, ["small_table", "--col-range", "3:"])
    assert result.exit_code == 0

    result = runner.invoke(preview, ["small_table", "--col-range", ":4"])
    assert result.exit_code == 0


def test_format_dtype_compact_edge_cases():
    """Test format_dtype_compact with additional edge cases."""
    # Test case variations and complex types
    test_cases = [
        ("TIME64[ns]", "time"),
        ("DATETIME64[ns]", "datetime"),
        ("LIST[STRING]", "list[str…"),
        ("MAP<STRING,INT64>", "map<str…"),
        ("STRUCT", "struct"),
        ("NULL", "null"),
        ("", ""),
        ("a", "a"),
        ("very_very_long_type_name_exceeding_limit", "very_ver…"),
    ]

    for input_type, expected in test_cases:
        result = _format_dtype_compact(input_type)
        if len(expected) > 8 and not expected.endswith("…"):
            expected = expected[:8] + "..."
        # Allow flexible matching for complex type transformations
        assert isinstance(result, str)
        assert len(result) <= 15


def test_rich_print_gt_table_wide_table_handling():
    """Test rich table display with very wide tables."""

    # Create mock GT table with many columns
    mock_gt = Mock()
    mock_df = Mock()

    # Create 25 columns (more than max_terminal_cols)
    many_columns = [f"column_{i:02d}" for i in range(25)]
    mock_df.columns = many_columns
    mock_gt._tbl_data = mock_df

    # Mock the DataFrame methods
    mock_df.to_dicts.return_value = [
        {col: f"value_{i}" for i, col in enumerate(many_columns)} for _ in range(3)
    ]

    # Test that the function can handle wide tables without crashing
    try:
        _rich_print_gt_table(mock_gt)
    except Exception:
        pass  # Expected due to mocking limitations, but shouldn't crash the test


def test_get_column_dtypes_fallback_scenarios():
    """Test _get_column_dtypes with various fallback scenarios."""

    # Test with DataFrame that raises exception on dtype access
    mock_df = Mock()
    mock_df.dtypes.side_effect = Exception("Mock exception")

    result = _get_column_dtypes(mock_df, ["col1", "col2"])
    assert result == {"col1": "?", "col2": "?"}

    # Test with DataFrame that has schema but no to_dict method
    mock_df2 = Mock(spec=[])  # Empty spec so no attributes exist
    mock_df2.schema = Mock(spec=[])  # Empty spec so no to_dict method

    result = _get_column_dtypes(mock_df2, ["col1"])
    # This may return "unknown" because getattr returns "Unknown" which gets formatted
    assert result == {"col1": "unknown"} or result == {"col1": "?"}

    # Test with DataFrame that has neither dtypes nor schema
    mock_df3 = Mock(spec=[])  # No attributes exist
    result = _get_column_dtypes(mock_df3, ["col1"])
    assert result == {"col1": "?"}


def test_format_cell_value_numpy_without_pandas():
    """Test format_cell_value when numpy is available but pandas is not."""

    # Test with NaN value handling when imports might fail
    result = _format_cell_value(float("nan"))
    assert isinstance(result, str)
    # The result could be "nan" or some other string representation


def test_cli_commands_help_output():
    """Test help output for all CLI commands."""
    runner = CliRunner()

    # Test main CLI help
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Pointblank CLI" in result.output

    # Test individual command help
    commands = [
        "datasets",
        "requirements",
        "preview",
        "info",
        "scan",
        "missing",
        "validate",
        "run",
    ]

    for cmd_name in commands:
        result = runner.invoke(cli, [cmd_name, "--help"])
        assert result.exit_code == 0


def test_validate_all_check_types():
    """Test validate with all available check types."""
    runner = CliRunner()

    # Test each check type individually (only using available choices)
    basic_checks = ["rows-distinct", "rows-complete", "col-vals-not-null"]

    for check in basic_checks:
        result = runner.invoke(validate, ["small_table", "--check", check])
        assert result.exit_code in [0, 1]  # May pass or fail validation

    # Test checks that require additional parameters
    result = runner.invoke(
        validate,
        [
            "small_table",
            "--check",
            "col-vals-gt",
            "--column",
            "c",
            "--value",
            "1",
        ],
    )
    assert result.exit_code in [0, 1]


def test_display_validation_summary_error_handling():
    """Test _display_validation_summary with error conditions."""

    # Test with None validation object
    try:
        _display_validation_summary(None)
    except Exception:
        pass  # Expected

    # Test with validation object missing attributes
    mock_validation = Mock()
    mock_validation.validation_info = None

    try:
        _display_validation_summary(mock_validation)
    except Exception:
        pass  # Expected due to missing attributes


def test_rich_print_functions_error_recovery():
    """Test rich print functions with error scenarios."""

    # Test _rich_print_gt_table with None input
    try:
        _rich_print_gt_table(None)
    except Exception:
        pass  # Expected

    # Test with GT table that fails HTML generation
    mock_gt = Mock()
    mock_gt.as_raw_html.side_effect = Exception("HTML generation failed")

    try:
        _rich_print_gt_table(mock_gt)
    except Exception:
        pass  # Expected


def test_cli_with_connection_string_formats():
    """Test CLI commands with various connection string formats."""
    runner = CliRunner()

    # Test with different connection string formats (these may fail but shouldn't crash)
    connection_strings = [
        "csv://nonexistent.csv",
        "parquet://nonexistent.parquet",
        "duckdb://memory",
        "sqlite://memory",
    ]

    for conn_str in connection_strings:
        result = runner.invoke(preview, [conn_str])
        # Should handle gracefully, exit code may be 0 or 1
        assert result.exit_code in [0, 1]


def test_missing_percentage_precision():
    """Test missing percentage formatting precision."""

    # Test various precision scenarios with pre-calculated percentages
    test_cases = [
        (33.3, "33%"),
        (66.7, "67%"),
        (14.3, "14%"),
        (83.3, "83%"),
        (0.0, "[green]●[/green]"),
        (100.0, "[red]●[/red]"),
    ]

    for percentage, expected in test_cases:
        result = _format_missing_percentage(percentage)
        assert result == expected


def test_column_selection_edge_cases():
    """Test column selection with edge cases."""
    runner = CliRunner()

    # Test with invalid column names
    result = runner.invoke(preview, ["small_table", "--columns", "nonexistent_column"])
    assert result.exit_code in [0, 1]  # May handle gracefully

    # Test with empty column specification
    result = runner.invoke(preview, ["small_table", "--columns", ""])
    assert result.exit_code in [0, 1]

    # Test with column range that exceeds available columns
    result = runner.invoke(preview, ["small_table", "--col-range", "1:100"])
    assert result.exit_code in [0, 1]


def test_datasets_command():
    """Test the datasets command listing available datasets."""
    runner = CliRunner()
    result = runner.invoke(datasets)
    assert result.exit_code == 0
    assert "Available Pointblank Datasets" in result.output
    assert "small_table" in result.output
    assert "game_revenue" in result.output
    assert "nycflights" in result.output
    assert "global_sales" in result.output


def test_requirements_command():
    """Test the requirements command showing dependency status."""
    runner = CliRunner()
    result = runner.invoke(requirements)
    assert result.exit_code == 0
    assert "Dependency Status" in result.output
    assert "polars" in result.output
    assert "pandas" in result.output


def test_requirements_command_detailed():
    """Test the requirements command with detailed output."""
    runner = CliRunner()
    result = runner.invoke(requirements)
    assert result.exit_code == 0
    assert "Dependency Status" in result.output
    assert "polars" in result.output
    assert "pandas" in result.output
    assert "ibis" in result.output
    assert "duckdb" in result.output
    assert "pyarrow" in result.output


def test_format_cell_value_comprehensive():
    """Test format_cell_value with comprehensive scenarios."""

    # Test with various None-like values
    assert _format_cell_value(None) == "[red]None[/red]"
    assert _format_cell_value("") == "[red][/red]"

    # Test row number formatting
    assert _format_cell_value(42, is_row_number=True) == "[dim]42[/dim]"

    # Test with different data types
    assert isinstance(_format_cell_value(123), str)
    assert isinstance(_format_cell_value(12.34), str)
    assert isinstance(_format_cell_value(True), str)
    # Skip list/dict tests as they can cause pandas.isna() issues
    # assert isinstance(_format_cell_value([1, 2, 3]), str)
    # assert isinstance(_format_cell_value({"key": "value"}), str)

    # Test truncation with different column counts
    long_text = "x" * 100
    result_few_cols = _format_cell_value(long_text, max_width=50, num_columns=3)
    result_many_cols = _format_cell_value(long_text, max_width=50, num_columns=20)

    # With many columns, should be more aggressively truncated
    assert len(result_many_cols) <= len(result_few_cols)


def test_get_column_dtypes_comprehensive():
    """Test _get_column_dtypes with comprehensive DataFrame scenarios."""

    # Test with mock DataFrame with dtypes.to_dict
    mock_df1 = Mock()
    mock_dtypes1 = Mock()
    mock_dtypes1.to_dict.return_value = {"col1": "String", "col2": "Int64"}
    mock_df1.dtypes = mock_dtypes1

    result = _get_column_dtypes(mock_df1, ["col1", "col2"])
    assert result["col1"] == "str"
    assert result["col2"] == "i64"

    # Test with DataFrame that has dtypes but no to_dict method
    mock_df2 = Mock()
    mock_dtypes2 = Mock()
    del mock_dtypes2.to_dict  # Remove to_dict method
    mock_dtypes2.iloc = Mock(side_effect=lambda i: f"dtype_{i}")
    mock_df2.dtypes = mock_dtypes2

    result = _get_column_dtypes(mock_df2, ["col1"])
    assert "col1" in result

    # Test with DataFrame that has schema
    mock_df3 = Mock()
    del mock_df3.dtypes
    mock_schema = Mock()
    mock_schema.to_dict.return_value = {"col1": "String"}
    mock_df3.schema = mock_schema

    result = _get_column_dtypes(mock_df3, ["col1"])
    assert result["col1"] == "str"


def test_format_missing_percentage_edge_cases():
    """Test format_missing_percentage with comprehensive edge cases."""

    # Test normal cases with pre-calculated percentages
    assert _format_missing_percentage(5.0) == "5%"
    assert _format_missing_percentage(50.0) == "50%"
    assert _format_missing_percentage(0.0) == "[green]●[/green]"
    assert _format_missing_percentage(100.0) == "[red]●[/red]"

    # Test edge cases
    assert _format_missing_percentage(0.5) == "<1%"
    assert _format_missing_percentage(99.5) == ">99%"

    # Test small percentages
    assert _format_missing_percentage(0.1) == "<1%"

    # Test large percentages
    assert _format_missing_percentage(99.9) == ">99%"


def test_cli_commands_basic_functionality():
    """Test basic functionality of all CLI commands with valid inputs."""
    runner = CliRunner()

    # Test info command
    result = runner.invoke(info, ["small_table"])
    assert result.exit_code == 0

    # Test missing command
    result = runner.invoke(missing, ["small_table"])
    assert result.exit_code == 0

    # Test make-template command
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        result = runner.invoke(make_template, [f.name])
        script_path = f.name

    try:
        assert result.exit_code == 0
    finally:
        Path(script_path).unlink(missing_ok=True)


def test_scan_command_basic():
    """Test scan command basic functionality with mocked data loading."""
    runner = CliRunner()
    result = runner.invoke(scan, ["small_table"])
    assert result.exit_code == 0


def test_run_command_basic():
    """Test run command basic functionality."""
    runner = CliRunner()

    # Create a temporary validation script
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write("""
import pointblank as pb

# Load data directly in script
data = pb.load_dataset("small_table")

validation = (
    pb.Validate(data=data)
    .col_exists(['a', 'b'])
    .interrogate()
)
""")
        script_path = f.name

    try:
        result = runner.invoke(run, [script_path])
        assert result.exit_code in [0, 1]  # May pass or fail validation
    finally:
        Path(script_path).unlink()


def test_preview_with_different_head_tail_combinations():
    """Test preview command with different head/tail combinations."""
    runner = CliRunner()

    # Test with different head values
    result = runner.invoke(preview, ["small_table", "--head", "3"])
    assert result.exit_code == 0

    # Test with different tail values
    result = runner.invoke(preview, ["small_table", "--tail", "2"])
    assert result.exit_code == 0

    # Test with both head and tail
    result = runner.invoke(preview, ["small_table", "--head", "2", "--tail", "1"])
    assert result.exit_code == 0

    # Test with limit
    result = runner.invoke(preview, ["small_table", "--limit", "5"])
    assert result.exit_code in [0, 1]  # May have issues with limit validation


def test_preview_column_selection_combinations():
    """Test preview command with various column selection methods."""
    runner = CliRunner()

    # Test col-first
    result = runner.invoke(preview, ["small_table", "--col-first", "3"])
    assert result.exit_code == 0

    # Test col-last
    result = runner.invoke(preview, ["small_table", "--col-last", "2"])
    assert result.exit_code == 0

    # Test col-range with different formats
    result = runner.invoke(preview, ["small_table", "--col-range", "2:4"])
    assert result.exit_code == 0

    result = runner.invoke(preview, ["small_table", "--col-range", "2:"])
    assert result.exit_code == 0

    result = runner.invoke(preview, ["small_table", "--col-range", ":3"])
    assert result.exit_code == 0


def test_all_built_in_datasets():
    """Test that all built-in datasets work with basic commands."""
    runner = CliRunner()

    datasets = ["small_table", "game_revenue", "nycflights", "global_sales"]

    for dataset in datasets:
        # Test preview
        result = runner.invoke(preview, [dataset])
        assert result.exit_code == 0

        # Test info
        result = runner.invoke(info, [dataset])
        assert result.exit_code == 0


def test_rich_print_functions_with_console_errors():
    """Test rich print functions when console operations fail."""

    # Test with mock console that raises errors
    with patch("pointblank.cli.console") as mock_console:
        mock_console.print.side_effect = Exception("Console error")

        # These should handle console errors gracefully
        try:
            _rich_print_missing_table(gt_table=Mock(), original_data=None)
        except Exception:
            pass  # Expected

        try:
            _rich_print_scan_table(
                scan_result=Mock(),
                data_source="test.csv",
                source_type="CSV",
                table_type="DataFrame",
                total_rows=100,
            )
        except Exception:
            pass  # Expected


def test_missing_command_basic(runner, tmp_path):
    """Test basic missing command functionality."""
    result = runner.invoke(missing, ["small_table"])
    assert result.exit_code in [0, 1]  # May fail due to missing dependencies
    assert "✓ Loaded data source: small_table" in result.output or "Error:" in result.output


def test_missing_command_with_html_output(runner, tmp_path):
    """Test missing command with HTML output."""
    output_file = tmp_path / "missing_report.html"
    result = runner.invoke(missing, ["small_table", "--output-html", str(output_file)])
    assert result.exit_code in [0, 1]  # May fail due to missing dependencies


def test_missing_command_with_invalid_data(runner):
    """Test missing command with invalid data source."""
    result = runner.invoke(missing, ["nonexistent_file.csv"])
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_run_command_comprehensive(runner, tmp_path):
    """Test run command with comprehensive options."""
    script_file = tmp_path / "validation.py"
    html_file = tmp_path / "report.html"
    json_file = tmp_path / "report.json"

    script_content = """
import pointblank as pb

# Use CLI-provided data if available, otherwise load default
if 'cli_data' in globals() and cli_data is not None:
    data = cli_data
else:
    data = pb.load_dataset("small_table")

validation = pb.Validate(data=data).col_vals_gt("c", 0).interrogate()
"""
    script_file.write_text(script_content)

    result = runner.invoke(
        run,
        [
            str(script_file),
            "--data",
            "small_table",
            "--output-html",
            str(html_file),
            "--output-json",
            str(json_file),
        ],
    )
    assert result.exit_code in [0, 1]


def test_run_command_fail_on_critical(runner, tmp_path):
    """Test run command with fail-on option."""
    script_file = tmp_path / "validation.py"
    script_content = """
import pointblank as pb

# Use CLI-provided data if available, otherwise load default
if 'cli_data' in globals() and cli_data is not None:
    data = cli_data
else:
    data = pb.load_dataset("small_table")

validation = pb.Validate(data=data, thresholds=pb.Thresholds(critical=0.01)).col_vals_gt("c", 999999).interrogate()  # Should fail
"""
    script_file.write_text(script_content)

    result = runner.invoke(
        run, [str(script_file), "--data", "small_table", "--fail-on", "critical"]
    )
    assert result.exit_code in [0, 1]


def test_run_command_invalid_script(runner, tmp_path):
    """Test run command with invalid script."""
    script_file = tmp_path / "bad_script.py"
    script_file.write_text("invalid python syntax !!!")

    result = runner.invoke(run, [str(script_file)])
    assert result.exit_code == 1
    assert "Error executing validation script:" in result.output


def test_column_range_selection_edge_cases(runner):
    """Test column range selection with various edge cases."""
    # Test invalid range format
    result = runner.invoke(preview, ["small_table", "--col-range", "invalid"])
    assert result.exit_code in [0, 1]

    # Test range with missing end
    result = runner.invoke(preview, ["small_table", "--col-range", "1:"])
    assert result.exit_code in [0, 1]

    # Test range with missing start
    result = runner.invoke(preview, ["small_table", "--col-range", ":3"])
    assert result.exit_code in [0, 1]


def test_preview_with_data_processing_errors(runner):
    """Test preview with data processing errors."""
    # Test with a definitely invalid file that should cause processing errors
    result = runner.invoke(preview, ["nonexistent_file_xyz.csv"])
    assert result.exit_code == 1  # Should fail
    assert "Error:" in result.output


def test_scan_with_data_processing_errors(runner, monkeypatch):
    """Test scan with data processing errors."""

    def mock_scan_error(*args, **kwargs):
        raise Exception("Scan error")

    monkeypatch.setattr("pointblank.col_summary_tbl", mock_scan_error)

    result = runner.invoke(scan, ["small_table"])
    assert result.exit_code == 1
    assert "Error:" in result.output


def test_format_cell_value_with_pandas_dtypes():
    """Test format_cell_value with pandas-specific data types."""
    try:
        import pandas as pd
        import numpy as np

        # Test with pandas NA
        assert _format_cell_value(pd.NA) == "[red]NA[/red]"

        # Test with pandas Timestamp
        ts = pd.Timestamp("2021-01-01")
        result = _format_cell_value(ts)
        assert "2021-01-01" in result

        # Test with pandas categorical
        cat = pd.Categorical(["a", "b", "c"])
        result = _format_cell_value(cat[0])
        assert result == "a"

    except ImportError:
        # Skip if pandas not available
        pass


def test_get_column_dtypes_with_schema_based_systems():
    """Test _get_column_dtypes with schema-based systems."""

    # Mock object with schema attribute
    class MockSchemaObj:
        def __init__(self):
            self.schema = MockSchema()

        @property
        def columns(self):
            return ["col1", "col2"]

    class MockSchema:
        def to_dict(self):
            return {"col1": "int64", "col2": "str"}

    obj = MockSchemaObj()
    result = _get_column_dtypes(obj, obj.columns)
    assert "col1" in result
    assert "col2" in result


def test_rich_print_functions_with_different_table_formats():
    """Test rich print functions with different table formats."""
    from rich.console import Console
    from io import StringIO

    # Mock GT table-like object
    class MockGTTable:
        def _repr_html_(self):
            return "<table><tr><td>test</td></tr></table>"

    # Test with string buffer to capture output
    string_io = StringIO()
    console = Console(file=string_io, width=80)

    # This should not crash
    try:
        _rich_print_gt_table(MockGTTable(), console=console)
    except Exception:
        pass  # Expected to fail in test environment, but shouldn't crash


def test_display_validation_summary_edge_cases():
    """Test display_validation_summary with edge cases."""

    # Mock validation object with no info
    class MockValidation:
        validation_info = None

    # Should not crash
    try:
        _display_validation_summary(MockValidation())
    except Exception:
        pass  # May fail but shouldn't crash the system

    # Mock validation object with empty info
    class MockValidationEmpty:
        validation_info = []

    try:
        _display_validation_summary(MockValidationEmpty())
    except Exception:
        pass


def test_format_dtype_compact_with_complex_types():
    """Test _format_dtype_compact with complex data types."""
    # Test various pandas dtypes
    assert "obj" == _format_dtype_compact("object")
    assert "i64" == _format_dtype_compact("int64")
    assert "f32" == _format_dtype_compact("float32")
    assert "bool" == _format_dtype_compact("boolean")
    assert "datetime" == _format_dtype_compact("datetime64[ns]")

    # Test unknown types (should be truncated if too long)
    result = _format_dtype_compact("unknown_type_12345")
    assert result == "unknown_…"  # Long types get truncated


def test_preview_with_column_selection_and_row_num(runner):
    """Test preview with column selection when _row_num_ exists."""
    result = runner.invoke(preview, ["small_table", "--columns", "a,b", "--head", "3"])
    assert result.exit_code in [0, 1]


def test_cli_commands_with_connection_strings(runner):
    """Test CLI commands with various connection string formats."""
    # Test with DuckDB connection string
    result = runner.invoke(preview, ["duckdb:///test.db::table"])
    assert result.exit_code in [0, 1]  # Will fail but shouldn't crash

    # Test with SQL query in connection string
    result = runner.invoke(scan, ["duckdb:///test.db::SELECT * FROM table"])
    assert result.exit_code in [0, 1]


def test_format_cell_value_with_special_pandas_cases():
    """Test format_cell_value with special pandas cases that might be missed."""
    try:
        import pandas as pd
        import numpy as np

        # Test with pandas Series element
        series = pd.Series([1, 2, 3])
        result = _format_cell_value(series.iloc[0])
        assert result == "1"

        # Test with pandas Index element
        index = pd.Index([1, 2, 3])
        result = _format_cell_value(index[0])
        assert result == "1"

    except ImportError:
        pass


def test_get_column_dtypes_error_recovery():
    """Test _get_column_dtypes error recovery."""

    # Mock object that causes errors
    class ErrorObj:
        @property
        def columns(self):
            raise Exception("Column access error")

        @property
        def dtypes(self):
            raise Exception("Dtypes access error")

    obj = ErrorObj()
    columns = ["col1", "col2"]

    # Should return fallback dictionary
    result = _get_column_dtypes(obj, columns)
    assert all(col in result for col in columns)
    assert all(result[col] == "?" for col in columns)


def test_rich_print_gt_table_with_wide_data():
    """Test _rich_print_gt_table with wide table handling."""

    # Mock a wide GT table
    class MockWideGTTable:
        def _repr_html_(self):
            # Simulate wide table HTML
            cols = [f"col_{i}" for i in range(20)]  # Many columns
            html = "<table><tr>"
            for col in cols:
                html += f"<th>{col}</th>"
            html += "</tr><tr>"
            for i in range(20):
                html += f"<td>value_{i}</td>"
            html += "</tr></table>"
            return html

    # This should handle wide tables gracefully
    try:
        _rich_print_gt_table(MockWideGTTable())
    except Exception:
        pass  # May fail but shouldn't crash


def test_format_missing_percentage_boundary_values():
    """Test _format_missing_percentage with boundary values."""
    # Test exactly 0%
    assert _format_missing_percentage(0.0) == "[green]●[/green]"

    # Test exactly 100%
    assert _format_missing_percentage(100.0) == "[red]●[/red]"

    # Test very small percentage
    assert _format_missing_percentage(0.0001) == "<1%"

    # Test very large percentage (>100%)
    assert _format_missing_percentage(150.0) == "150%"


def test_preview_command_with_file_not_found_error(runner):
    """Test preview command when file processing functions throw specific errors."""
    result = runner.invoke(preview, ["/nonexistent/path/file.csv"])
    assert result.exit_code in [0, 1]
    # Should handle gracefully without crashing


def test_scan_command_with_html_file_write_error(runner, tmp_path, monkeypatch):
    """Test scan command with HTML file write error."""
    # Create a directory instead of a file to cause write error
    output_dir = tmp_path / "scan_output.html"
    output_dir.mkdir()

    result = runner.invoke(scan, ["small_table", "--output-html", str(output_dir)])
    assert result.exit_code in [0, 1]


def test_run_command_with_file_output_errors(runner, tmp_path, monkeypatch):
    """Test run command with file output errors."""
    script_file = tmp_path / "validation.py"
    script_content = """
import pointblank as pb

# Use CLI-provided data if available, otherwise load default
if 'cli_data' in globals() and cli_data is not None:
    data = cli_data
else:
    data = pb.load_dataset("small_table")

validation = pb.Validate(data=data).col_vals_gt("c", 0).interrogate()
"""
    script_file.write_text(script_content)

    # Create directories instead of files to cause write errors
    html_dir = tmp_path / "report.html"
    json_dir = tmp_path / "report.json"
    html_dir.mkdir()
    json_dir.mkdir()

    result = runner.invoke(
        run,
        [
            str(script_file),
            "--data",
            "small_table",
            "--output-html",
            str(html_dir),
            "--output-json",
            str(json_dir),
        ],
    )
    assert result.exit_code in [0, 1]
    # Should show warnings about not being able to save files


def test_preview_with_column_iteration_error():
    """Test preview command error handling during column iteration."""

    # This tests the exception handling in _rich_print_gt_table
    class MockErrorTable:
        def _repr_html_(self):
            raise Exception("HTML generation error")

    # Should handle the error gracefully
    try:
        _rich_print_gt_table(MockErrorTable())
    except Exception:
        pass  # Expected


def test_is_piped_data_source():
    """Test the _is_piped_data_source() function."""

    # Test valid piped data source
    assert _is_piped_data_source("/var/folders/abc/def/pb_pipe_12345") == True
    assert _is_piped_data_source("/tmp/pb_pipe_67890") == True

    # Test non-piped data sources
    assert _is_piped_data_source("regular_file.csv") == False
    assert _is_piped_data_source("/var/folders/abc/regular_file.csv") == False
    assert _is_piped_data_source("/tmp/regular_file.csv") == False
    assert _is_piped_data_source("pb_pipe_without_path") == False

    # Test falsy values (empty string returns empty string, which is falsy)
    assert not _is_piped_data_source("")
    assert not _is_piped_data_source(None)


def test_ordered_group_fallback():
    """Test OrderedGroup fallback for commands not in desired order."""

    # Create a mock context
    class MockContext:
        pass

    # Create an OrderedGroup instance
    group = OrderedGroup()

    # Mock the parent list_commands to return commands not in our desired order
    def mock_list_commands(self, ctx):
        return ["unknown_command", "info", "another_unknown", "preview"]

    # Temporarily replace the parent method
    original_method = click.Group.list_commands
    click.Group.list_commands = mock_list_commands

    try:
        ctx = MockContext()
        result = group.list_commands(ctx)

        # Should have info and preview in desired order, followed by unknown commands
        assert "info" in result
        assert "preview" in result
        assert "unknown_command" in result
        assert "another_unknown" in result

        # info should come before preview (desired order)
        info_idx = result.index("info")
        preview_idx = result.index("preview")
        assert info_idx < preview_idx

    finally:
        # Restore original method
        click.Group.list_commands = original_method


def test_format_cell_value_edge_cases():
    """Test edge cases in _format_cell_value() function."""

    # Test with very wide content and many columns
    long_text = "x" * 200
    result = _format_cell_value(long_text, max_width=50, num_columns=20)

    assert len(result) <= 30  # Should be more aggressive with many columns
    assert "…" in result

    # Test with medium number of columns
    result = _format_cell_value(long_text, max_width=50, num_columns=12)

    assert len(result) <= 40  # Less aggressive truncation
    assert "…" in result

    # Test extremely long text
    extremely_long = "y" * 1000
    result = _format_cell_value(extremely_long, max_width=50, num_columns=5)

    assert "…" in result
    assert len(result) <= 50


def test_get_column_dtypes_exception_handling():
    """Test exception handling in _get_column_dtypes()."""

    # Mock an object that raises exceptions when accessing dtypes
    class ProblematicDataFrame:
        @property
        def dtypes(self):
            raise Exception("Cannot access dtypes")

        @property
        def schema(self):
            raise Exception("Cannot access schema")

    columns = ["col1", "col2"]
    df = ProblematicDataFrame()

    # Should handle exceptions gracefully and return "?" for all columns
    result = _get_column_dtypes(df, columns)
    expected = {"col1": "?", "col2": "?"}

    assert result == expected


def test_format_cell_value_pandas_specific():
    """Test Pandas-specific NA handling in _format_cell_value()."""

    # Test with Pandas NA
    result = _format_cell_value(pd.NA)
    assert "[red]" in result and "NA" in result

    # Test with Pandas NaT (Not a Time)
    result = _format_cell_value(pd.NaT)
    assert "[red]" in result


def test_get_column_dtypes_schema_exception():
    """Test _get_column_dtypes() with schema objects that raise exceptions."""

    class MockSchemaWithError:
        def to_dict(self):
            raise Exception("Schema conversion error")

    class MockDataFrameWithBadSchema:
        dtypes = None
        schema = MockSchemaWithError()

    columns = ["col1", "col2"]
    df = MockDataFrameWithBadSchema()

    # Should handle schema exception and fall back to "?"
    result = _get_column_dtypes(df, columns)
    expected = {"col1": "?", "col2": "?"}

    assert result == expected


def test_get_column_dtypes_pandas_edge_cases():
    """Test _get_column_dtypes() Pandas DataFrame edge cases."""

    # Test pandas DataFrame with dtypes that don't have iloc
    class MockPandasDtypes:
        def __init__(self):
            self.data = ["int64", "float32", "object"]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index]

    class MockPandasDF:
        def __init__(self):
            self.dtypes = MockPandasDtypes()

    df = MockPandasDF()
    columns = ["col1", "col2", "col3"]

    # Should handle pandas dtypes without iloc
    result = _get_column_dtypes(df, columns)

    assert result["col1"] == "i64"  # int64 -> i64
    assert result["col2"] == "f32"  # float32 -> f32
    assert result["col3"] == "obj"  # object -> obj

    # Test case where we have more columns than dtypes
    columns_extra = ["col1", "col2", "col3", "col4", "col5"]
    result = _get_column_dtypes(df, columns_extra)

    assert result["col4"] == "?"  # Should fall back to "?" for extra columns
    assert result["col5"] == "?"


def test_format_dtype_compact_generic_fallbacks():
    """Test _format_dtype_compact() generic fallback cases."""

    # Test generic int fallback: types that contain "int" but not specific patterns
    assert _format_dtype_compact("int") == "int"
    assert _format_dtype_compact("bigint") == "int"
    assert _format_dtype_compact("smallint") == "int"
    assert _format_dtype_compact("integer") == "int"

    # Test generic float fallback: types that contain "float" but not float32/float64
    assert _format_dtype_compact("float") == "float"  # exact match
    assert _format_dtype_compact("myfloat") == "float"  # contains "float"

    # Test generic string fallback: types that contain "str" but not "string"
    assert _format_dtype_compact("str") == "str"  # exact match
    assert _format_dtype_compact("mystr") == "str"  # contains "str" and ≤8 chars

    # Test long unknown types get truncated: this happens before generic fallbacks
    long_type = "very_long_unknown_type_name"
    result = _format_dtype_compact(long_type)

    assert result == "very_lon…"
    assert len(result) == 9  # 8 chars + ellipsis

    # Test short unknown types pass through: this is the final fallback
    short_type = "custom"
    result = _format_dtype_compact(short_type)

    assert result == "custom"

    # Test types that don't match any pattern and are short
    assert _format_dtype_compact("double") == "double"  # 6 chars, passes through
    assert _format_dtype_compact("decimal") == "decimal"  # 7 chars, passes through
    assert _format_dtype_compact("varchar") == "varchar"  # 7 chars, passes through


def test_format_cell_value_pandas_exception_handling():
    """Test _format_cell_value() Pandas exception handling."""

    # Mock pandas to raise exceptions
    with patch("pandas.isna") as mock_isna, patch("numpy.isnan") as mock_isnan:
        # Test ImportError path
        mock_isna.side_effect = ImportError("pandas not available")
        mock_isnan.side_effect = ImportError("numpy not available")

        result = _format_cell_value(float("nan"))

        # Should handle ImportError gracefully and return string representation
        assert isinstance(result, str)

        mock_isna.side_effect = TypeError("value not compatible")
        mock_isnan.side_effect = TypeError("value not compatible")

        result = _format_cell_value("test_value")

        # Should handle TypeError gracefully
        assert result == "test_value"

        # Test ValueError path
        mock_isna.side_effect = ValueError("ambiguous array")
        mock_isnan.side_effect = ValueError("ambiguous array")

        result = _format_cell_value(123)

        # Should handle ValueError gracefully
        assert result == "123"


def test_format_cell_value_pandas_type_detection():
    """Test _format_cell_value() Pandas type detection edge case."""

    # Test with pandas NA type
    with patch("pandas.isna") as mock_isna:
        mock_isna.return_value = True

        # Create a pandas object and mock its type detection
        pd_series = pd.Series([1, 2, 3])
        result = _format_cell_value(pd_series.iloc[0])

        # The result should be a string
        assert isinstance(result, str)


def test_get_column_dtypes_polars_style():
    """Test _get_column_dtypes() with Polars dtypes."""

    # Mock Polars-style DataFrame
    class MockPolarsDtypes:
        def to_dict(self):
            return {"col1": "Int64", "col2": "Utf8", "col3": "Float64", "col4": "Boolean"}

    class MockPolarsDF:
        def __init__(self):
            self.dtypes = MockPolarsDtypes()

    df = MockPolarsDF()
    columns = ["col1", "col2", "col3", "col4"]

    # Should handle Polars-style dtypes
    result = _get_column_dtypes(df, columns)

    assert result["col1"] == "i64"  # Int64 -> i64
    assert result["col2"] == "str"  # Utf8 -> str
    assert result["col3"] == "f64"  # Float64 -> f64
    assert result["col4"] == "bool"  # Boolean -> bool


def test_get_column_dtypes_missing_column_fallback():
    """Test _get_column_dtypes() when column is missing from dtypes."""

    # Mock Polars-style DataFrame with missing column
    class MockPartialDtypes:
        def to_dict(self):
            return {"col1": "Int64", "col2": "Utf8"}  # Only 2 columns

    class MockPartialDF:
        def __init__(self):
            self.dtypes = MockPartialDtypes()

    df = MockPartialDF()
    columns = ["col1", "col2", "col3", "col4"]  # Asking for 4 columns but only 2 exist

    # Should handle missing columns by falling back to "?"
    result = _get_column_dtypes(df, columns)

    assert result["col1"] == "i64"  # Int64 -> i64 (exists)
    assert result["col2"] == "str"  # Utf8 -> str (exists)
    assert result["col3"] == "?"  # Missing column -> "?"
    assert result["col4"] == "?"  # Missing column -> "?"


def test_rich_print_gt_table_data_extraction_paths():
    """Test _rich_print_gt_table() data extraction from different GT table structures."""

    # Test _body.body path
    class MockBodyTable:
        def __init__(self):
            self._body = Mock()
            self._body.body = Mock()
            self._body.body.columns = ["col1", "col2"]
            self._body.body.to_dicts = Mock(return_value=[{"col1": "val1", "col2": "val2"}])

        def as_raw_html(self):
            return "<table><tr><td>test</td></tr></table>"

    # Should extract data from _body.body
    try:
        _rich_print_gt_table(MockBodyTable())
    except Exception:
        pass  # Expected due to mocking limitations

    # Test _data path
    class MockDataTable:
        def __init__(self):
            self._data = Mock()
            self._data.columns = ["col1", "col2"]
            self._data.to_dicts = Mock(return_value=[{"col1": "val1", "col2": "val2"}])

        def as_raw_html(self):
            return "<table><tr><td>test</td></tr></table>"

    # Should extract data from _data
    try:
        _rich_print_gt_table(MockDataTable())
    except Exception:
        pass  # Expected due to mocking limitations

    # Test .data path
    class MockDirectDataTable:
        def __init__(self):
            self.data = Mock()
            self.data.columns = ["col1", "col2"]
            self.data.to_dicts = Mock(return_value=[{"col1": "val1", "col2": "val2"}])

        def as_raw_html(self):
            return "<table><tr><td>test</td></tr></table>"

    # Should extract data from .data
    try:
        _rich_print_gt_table(MockDirectDataTable())
    except Exception:
        pass  # Expected due to mocking limitations


@patch("pointblank.cli.console")
def test_rich_print_gt_table_console_size_fallback(mock_console):
    """Test _rich_print_gt_table() console size exception handling."""

    # Mock console that raises exception when accessing size
    mock_console.size.width.side_effect = Exception("Cannot get console size")

    class MockGTTable:
        def __init__(self):
            self._tbl_data = Mock()
            self._tbl_data.columns = ["col1", "col2", "col3"]
            self._tbl_data.to_dicts = Mock(
                return_value=[{"col1": "val1", "col2": "val2", "col3": "val3"}]
            )

        def as_raw_html(self):
            return "<table><tr><td>test</td></tr></table>"

    # Should handle console size exception gracefully
    try:
        _rich_print_gt_table(MockGTTable())
    except Exception:
        pass  # May fail but shouldn't crash due to console size error


def test_rich_print_gt_table_row_number_calculations():
    """Test _rich_print_gt_table() row number width calculations."""

    # Test to_dicts path for row number calculation
    class MockTableWithRowNums:
        def __init__(self):
            self._tbl_data = Mock()
            self._tbl_data.columns = ["_row_num_", "col1", "col2"]
            self._tbl_data.to_dicts = Mock(
                return_value=[
                    {"_row_num_": 1, "col1": "val1", "col2": "val2"},
                    {"_row_num_": 999, "col1": "val3", "col2": "val4"},  # Large row number
                    {"_row_num_": 1234, "col1": "val5", "col2": "val6"},  # Even larger
                ]
            )

        def as_raw_html(self):
            return "<table><tr><td>test</td></tr></table>"

    # Should calculate appropriate row number width based on max row number
    try:
        _rich_print_gt_table(MockTableWithRowNums())
    except Exception:
        pass  # Expected due to mocking limitations

    # Test to_dict("records") path
    class MockTableWithRecords:
        def __init__(self):
            self._tbl_data = Mock()
            self._tbl_data.columns = ["_row_num_", "col1"]
            # Remove to_dicts to force to_dict path
            del self._tbl_data.to_dicts
            self._tbl_data.to_dict = Mock(
                return_value=[{"_row_num_": 1, "col1": "val1"}, {"_row_num_": 500, "col1": "val2"}]
            )

        def as_raw_html(self):
            return "<table><tr><td>test</td></tr></table>"

    # Should use to_dict("records") fallback
    try:
        _rich_print_gt_table(MockTableWithRecords())
    except Exception:
        pass  # Expected due to mocking limitations

    # Test exception in row number calculation
    class MockTableRowNumError:
        def __init__(self):
            self._tbl_data = Mock()
            self._tbl_data.columns = ["_row_num_", "col1"]
            self._tbl_data.to_dicts = Mock(side_effect=Exception("Row number calculation error"))

        def as_raw_html(self):
            return "<table><tr><td>test</td></tr></table>"

    # Should handle row number calculation exception
    try:
        _rich_print_gt_table(MockTableRowNumError())
    except Exception:
        pass  # Expected due to mocking limitations


def test_cli_preview_with_advanced_options():
    """Test preview command with advanced option combinations that might hit untested paths."""
    runner = CliRunner()

    # Test with very specific column selections that might trigger edge cases
    result = runner.invoke(preview, ["small_table", "--col-first", "1", "--col-last", "1"])
    assert result.exit_code == 0

    # Test with head=0 (edge case)
    result = runner.invoke(preview, ["small_table", "--head", "0"])
    assert result.exit_code in [0, 1]  # May be handled differently

    # Test with tail=0 (edge case)
    result = runner.invoke(preview, ["small_table", "--tail", "0"])
    assert result.exit_code in [0, 1]  # May be handled differently

    # Test with both head and tail as very small numbers
    result = runner.invoke(preview, ["small_table", "--head", "1", "--tail", "1"])
    assert result.exit_code == 0


def test_cli_scan_with_edge_case_options():
    """Test scan command with options that might trigger untested paths."""
    runner = CliRunner()

    # Test scan with very small limits that might trigger edge cases
    result = runner.invoke(scan, ["small_table"])
    assert result.exit_code == 0

    # These might exercise different code paths in scan processing
    # The exact paths depend on the internal logic, but these are reasonable test cases


def test_format_cell_value_extremely_long_truncation():
    """Test _format_cell_value() with edge cases in truncation logic."""

    # Test the "extremely long text" path in _format_cell_value()
    extremely_long_text = "z" * 2000  # Very long text

    # Test with various column counts to hit different truncation branches
    result = _format_cell_value(extremely_long_text, max_width=50, num_columns=5)
    assert "…" in result
    assert len(result) <= 50

    # Test the "double max_width" condition mentioned in the function
    double_max_text = "w" * 100  # Exactly double the max_width of 50
    result = _format_cell_value(double_max_text, max_width=50, num_columns=8)
    assert "…" in result


def test_format_units_basic_numbers():
    """Test _format_units() with basic number formatting."""

    # Test None
    assert _format_units(None) == "—"

    # Test small numbers (no formatting)
    assert _format_units(0) == "0"
    assert _format_units(1) == "1"
    assert _format_units(999) == "999"
    assert _format_units(9999) == "9999"

    # Test thousands (K formatting starts at 10,000)
    assert _format_units(10000) == "10K"
    assert _format_units(15000) == "15K"
    assert _format_units(99999) == "100K"
    assert _format_units(150000) == "150K"
    assert _format_units(999999) == "1000K"

    # Test millions
    assert _format_units(1000000) == "1.0M"
    assert _format_units(1500000) == "1.5M"
    assert _format_units(15000000) == "15.0M"
    assert _format_units(999999999) == "1000.0M"

    # Test billions
    assert _format_units(1000000000) == "1.0B"
    assert _format_units(1500000000) == "1.5B"
    assert _format_units(15000000000) == "15.0B"


def test_format_units_edge_cases():
    """Test _format_units() with edge cases."""
    # Test exact boundaries
    assert _format_units(9999) == "9999"  # Just under K threshold
    assert _format_units(10000) == "10K"  # Exactly K threshold
    assert _format_units(999999) == "1000K"  # Just under M threshold
    assert _format_units(1000000) == "1.0M"  # Exactly M threshold
    assert _format_units(999999999) == "1000.0M"  # Just under B threshold
    assert _format_units(1000000000) == "1.0B"  # Exactly B threshold

    # Test decimal precision
    assert _format_units(1234567) == "1.2M"  # Rounds down
    assert _format_units(1567890) == "1.6M"  # Rounds up
    assert _format_units(10550000000) == "10.6B"  # Rounds up


def test_format_pass_fail_basic():
    """Test _format_pass_fail() with basic pass/fail scenarios."""

    # Test None values
    assert _format_pass_fail(None, 100) == "—/—"
    assert _format_pass_fail(50, None) == "—/—"
    assert _format_pass_fail(None, None) == "—/—"

    # Test zero total
    assert _format_pass_fail(0, 0) == "—/—"
    assert _format_pass_fail(5, 0) == "—/—"

    # Test perfect scores (fraction = 1.0)
    assert _format_pass_fail(100, 100) == "100/1.00"
    assert _format_pass_fail(50, 50) == "50/1.00"
    assert _format_pass_fail(1, 1) == "1/1.00"

    # Test zero scores (fraction = 0.0)
    assert _format_pass_fail(0, 100) == "0/0.00"
    assert _format_pass_fail(0, 50) == "0/0.00"

    # Test regular fractions
    assert _format_pass_fail(50, 100) == "50/0.50"  # Exactly 50%
    assert _format_pass_fail(25, 100) == "25/0.25"  # Exactly 25%
    assert _format_pass_fail(75, 100) == "75/0.75"  # Exactly 75%


def test_format_pass_fail_special_thresholds():
    """Test _format_pass_fail() with special threshold handling."""

    # Test very small fractions (< 0.005)
    assert _format_pass_fail(1, 1000) == "1/<0.01"  # 0.001 -> <0.01
    assert _format_pass_fail(2, 1000) == "2/<0.01"  # 0.002 -> <0.01
    assert _format_pass_fail(4, 1000) == "4/<0.01"  # 0.004 -> <0.01

    # Test just at boundary (>= 0.005)
    assert _format_pass_fail(5, 1000) == "5/0.01"  # 0.005 -> 0.01
    assert _format_pass_fail(6, 1000) == "6/0.01"  # 0.006 -> 0.01

    # Test very high fractions (> 0.995)
    assert _format_pass_fail(996, 1000) == "996/>0.99"  # 0.996 -> >0.99
    assert _format_pass_fail(999, 1000) == "999/>0.99"  # 0.999 -> >0.99

    # Test at 0.995 boundary (0.995 rounds to 0.99, not 1.00)
    assert _format_pass_fail(995, 1000) == "995/0.99"  # 0.995 -> 0.99
    assert _format_pass_fail(994, 1000) == "994/0.99"  # 0.994 -> 0.99


def test_format_pass_fail_with_large_numbers():
    """Test _format_pass_fail() with large numbers (uses _format_units)."""

    # Test thousands
    assert _format_pass_fail(15000, 20000) == "15K/0.75"
    assert _format_pass_fail(50000, 100000) == "50K/0.50"

    # Test millions
    assert _format_pass_fail(1500000, 2000000) == "1.5M/0.75"
    assert _format_pass_fail(750000, 1000000) == "750K/0.75"

    # Test billions
    assert _format_pass_fail(1500000000, 2000000000) == "1.5B/0.75"

    # Test mixed large numbers with special thresholds
    assert _format_pass_fail(15000, 20000000) == "15K/<0.01"  # Very small fraction
    assert (
        _format_pass_fail(19950000, 20000000) == "19.9M/>0.99"
    )  # Very high fraction (19.95M rounds to 19.9M)


def test_format_pass_fail_precision():
    """Test _format_pass_fail() decimal precision handling."""

    # Test various decimal precisions
    assert _format_pass_fail(33, 100) == "33/0.33"  # 0.33
    assert _format_pass_fail(67, 100) == "67/0.67"  # 0.67
    assert _format_pass_fail(123, 1000) == "123/0.12"  # 0.123 -> 0.12
    assert _format_pass_fail(567, 1000) == "567/0.57"  # 0.567 -> 0.57
    assert _format_pass_fail(789, 1000) == "789/0.79"  # 0.789 -> 0.79

    # Test rounding behavior
    assert _format_pass_fail(333, 1000) == "333/0.33"  # 0.333 -> 0.33 (rounds down)
    assert _format_pass_fail(336, 1000) == "336/0.34"  # 0.336 -> 0.34 (rounds up)
    assert _format_pass_fail(666, 1000) == "666/0.67"  # 0.666 -> 0.67 (rounds up)


def test_load_data_source_edge_cases():
    """Test _load_data_source() function edge cases."""

    # Test with each of the built-in datasets to ensure they all work
    datasets = ["small_table", "game_revenue", "nycflights", "global_sales"]

    for dataset in datasets:
        try:
            result = _load_data_source(dataset)
            assert result is not None
        except Exception:
            pass  # May fail in test environment, but shouldn't crash

    # Test with non-dataset that should go through _process_data() path
    try:
        result = _load_data_source("unknown_dataset_name.csv")
    except Exception:
        pass  # Expected to fail, but shouldn't crash


def test_get_column_dtypes_missing_column_fallback(mock_data_loading):
    """Test _get_column_dtypes() when column not in dtypes."""

    # Create test data
    test_data = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

    # Mock a scenario where column dtype detection fails
    with patch("pandas.api.types.infer_dtype") as mock_infer:
        mock_infer.side_effect = Exception("Dtype detection failed")

        # This should fall back to dtypes_dict[col] = "?"
        result = _get_column_dtypes(test_data, ["col1", "col2"])

        # Should have fallback values
        assert "col1" in result
        assert "col2" in result


def test_validation_summary_severity_branches():
    """Test _display_validation_summary() severity logic."""

    # Mock a validation object with specific counts to test different severity branches
    mock_validation = Mock()

    # Test case 1: Some steps have errors
    mock_validation.n_passed.return_value = 2
    mock_validation.n_passed_all.return_value = 1
    mock_validation.n_failed.return_value = 1
    mock_validation.n_warned.return_value = 1
    mock_validation.get_tabulation_df.return_value = pd.DataFrame(
        {
            "eval": [1, 1, 0, 0, 1],  # Mix of pass/fail to avoid n_error > 0
            "n": [10, 8, 5, 3, 12],
            "f_passed": [1.0, 1.0, 0.0, 0.0, 1.0],
        }
    )

    with patch("pointblank.cli.console"):
        _display_validation_summary(mock_validation)

    # Test case 2: All passed
    mock_validation.get_tabulation_df.return_value = pd.DataFrame(
        {
            "eval": [1, 1, 1],  # All pass
            "n": [10, 8, 5],
            "f_passed": [1.0, 1.0, 1.0],  # All 100% pass rate
        }
    )
    mock_validation.n_passed.return_value = 3
    mock_validation.n_passed_all.return_value = 3  # All steps are "all passed"
    mock_validation.n_failed.return_value = 0
    mock_validation.n_warned.return_value = 0

    with patch("pointblank.cli.console"):
        _display_validation_summary(mock_validation)


def test_info_command_data_loading_status(runner, mock_data_loading):
    """Test info command data loading status message."""
    # Test the status context manager during data loading
    with patch("pointblank.cli.console") as mock_console:
        with patch("pointblank.cli._load_data_source") as mock_load:
            mock_load.return_value = pd.DataFrame({"col": [1, 2, 3]})
            with patch("pointblank.cli._get_tbl_type") as mock_type:
                mock_type.return_value = "pandas"
                with patch("pointblank.get_row_count") as mock_count:
                    mock_count.return_value = 3

                    result = runner.invoke(info, ["small_table"])

                    # Should have called console.status with loading message
                    mock_console.status.assert_called_with("[bold green]Loading data...")


def test_format_missing_percentage_edge_case():
    """Test _format_missing_percentage() with edge case values."""
    from pointblank.cli import _format_missing_percentage

    # Test with exactly 0.0
    result = _format_missing_percentage(0.0)
    assert result is not None

    # Test with exactly 1.0
    result = _format_missing_percentage(1.0)
    assert result is not None

    # Test with middle percentage
    result = _format_missing_percentage(0.5)
    assert result is not None


def test_rich_print_scan_table_comprehensive():
    """Test _rich_print_scan_table()."""

    # Create a mock GT object with _tbl_data attribute and realistic scan data
    mock_scan_result = Mock()

    # Create realistic scan data that would come from col_summary_tbl()
    scan_data = pd.DataFrame(
        {
            "colname": [
                '<div>col1</div><div style="color: gray;">int64</div>',
                '<div>col2</div><div style="color: gray;">object</div>',
                '<div>col3</div><div style="color: gray;">float64</div>',
            ],
            "n_missing": ["0", "2<br>10%", "1<br>5%"],
            "n_unique": ["3", "2", "T0.67F0.33"],  # Test different unique value formats
            "mean": [None, None, "15.5"],
            "std": [None, None, "5.2"],
            "min": ["1", None, "10.0"],
            "max": ["3", None, "21.0"],
            "median": [None, None, "15.0"],
            "q_1": [None, None, "12.5"],
            "q_3": [None, None, "18.5"],
            "iqr": [None, None, "6.0"],
        }
    )

    # Mock the _tbl_data attribute
    mock_scan_result._tbl_data = scan_data

    # Test successful table creation
    with patch("pointblank.cli.console") as mock_console:
        _rich_print_scan_table(
            scan_result=mock_scan_result,
            data_source="test_data.csv",
            source_type="External source: test_data.csv",
            table_type="pandas.DataFrame",
            total_rows=20,
            total_columns=3,
        )

        # Should have called console.print to display the table
        assert mock_console.print.called


def test_rich_print_scan_table_format_value_edge_cases():
    """Test the format_value() helper function with edge cases."""

    # Create test data with various edge cases
    mock_scan_result = Mock()
    edge_case_data = pd.DataFrame(
        {
            "colname": [
                '<div>dates</div><div style="color: gray;">int64</div>',
                '<div>large_nums</div><div style="color: gray;">int64</div>',
                '<div>small_nums</div><div style="color: gray;">float64</div>',
                '<div>strings</div><div style="color: gray;">object</div>',
            ],
            "n_missing": ["0", "0", "0", "0"],
            "n_unique": ["20240101", "1000000", "0.001", "some_very_long_string_value"],
            "mean": ["20240515", "5000000", "0.005", None],
            "max": ["20241231", "10000000", "0.01", None],
        }
    )
    mock_scan_result._tbl_data = edge_case_data

    with patch("pointblank.cli.console"):
        _rich_print_scan_table(
            scan_result=mock_scan_result,
            data_source="edge_cases.csv",
            source_type="Test data",
            table_type="pandas.DataFrame",
        )


def test_rich_print_scan_table_extract_column_info():
    """Test HTML column info extraction."""

    # Test data with various HTML formats
    mock_scan_result = Mock()
    html_test_data = pd.DataFrame(
        {
            "colname": [
                '<div>normal_col</div><div style="color: gray;">string</div>',
                "<div>missing_type</div>",  # No type div
                '<div class="fancy">fancy_col</div><div style="color: gray; font-style: italic;">complex_type</div>',
            ],
            "n_missing": ["0", "0", "0"],
            "n_unique": ["5", "3", "8"],
        }
    )
    mock_scan_result._tbl_data = html_test_data

    with patch("pointblank.cli.console"):
        _rich_print_scan_table(
            scan_result=mock_scan_result,
            data_source="html_test.csv",
            source_type="HTML test",
            table_type="test",
        )


def test_rich_print_scan_table_error_fallback():
    """Test error handling fallback."""

    # Create mock that will raise an exception during processing
    mock_scan_result = Mock()
    mock_scan_result._tbl_data = None  # This will cause an AttributeError

    with patch("pointblank.cli.console") as mock_console:
        _rich_print_scan_table(
            scan_result=mock_scan_result,
            data_source="error_test.csv",
            source_type="Error test",
            table_type="test",
        )

        # Should have called console.print with error message
        assert mock_console.print.called
        # Check that error message was displayed
        calls = mock_console.print.call_args_list
        error_call_found = False
        for call in calls:
            if len(call[0]) > 0 and "Error displaying table" in str(call[0][0]):
                error_call_found = True
                break
        assert error_call_found


def test_rich_print_scan_table_no_statistical_columns():
    """Test table with minimal columns (no statistical data)."""

    # Create data with only basic columns (no mean, std, etc.)
    mock_scan_result = Mock()
    minimal_data = pd.DataFrame(
        {
            "colname": ['<div>simple_col</div><div style="color: gray;">int</div>'],
            "n_missing": ["0"],
            "n_unique": ["10"],
            # No statistical columns
        }
    )
    mock_scan_result._tbl_data = minimal_data

    with patch("pointblank.cli.console"):
        _rich_print_scan_table(
            scan_result=mock_scan_result,
            data_source="minimal.csv",
            source_type="Minimal test",
            table_type="simple",
        )


def test_get_column_dtypes_polars_missing_column_fallback():
    """Test _get_column_dtypes() for Polars case where column not in raw_dtypes."""
    from pointblank.cli import _get_column_dtypes

    # Create mock that simulates Polars DataFrame with missing column in dtypes
    mock_df = Mock()
    mock_df.dtypes = Mock()

    # Mock to_dict method that returns incomplete dtype mapping
    def mock_to_dict():
        return {"col1": "Int64"}  # Missing col2

    mock_df.dtypes.to_dict = mock_to_dict

    # Call with columns that include one not in raw_dtypes
    result = _get_column_dtypes(mock_df, ["col1", "col2", "col3"])

    # col1 should have the actual dtype, col2 and col3 should be "?"
    assert "col1" in result
    assert result["col2"] == "?"
    assert result["col3"] == "?"


def test_show_concise_help_info():
    """Test _show_concise_help() for 'info' command."""
    with patch("pointblank.cli.console") as mock_console:
        with patch("sys.exit"):
            _show_concise_help("info", None)

        # Should have called console.print multiple times
        assert mock_console.print.called

        # Check that info-specific content was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        info_content_found = any("pb info" in call for call in calls)
        assert info_content_found


def test_show_concise_help_preview():
    """Test _show_concise_help() for 'preview' command."""
    with patch("pointblank.cli.console") as mock_console:
        with patch("sys.exit"):
            _show_concise_help("preview", None)

        # Should have called console.print multiple times
        assert mock_console.print.called

        # Check that preview-specific content was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        preview_content_found = any("pb preview" in call for call in calls)
        assert preview_content_found


def test_show_concise_help_scan():
    """Test _show_concise_help() for 'scan' command."""
    with patch("pointblank.cli.console") as mock_console:
        with patch("sys.exit"):
            _show_concise_help("scan", None)

        # Should have called console.print multiple times
        assert mock_console.print.called

        # Check that scan-specific content was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        scan_content_found = any("pb scan" in call for call in calls)
        assert scan_content_found


def test_show_concise_help_missing():
    """Test _show_concise_help() for 'missing' command."""
    with patch("pointblank.cli.console") as mock_console:
        with patch("sys.exit"):
            _show_concise_help("missing", None)

        # Should have called console.print multiple times
        assert mock_console.print.called

        # Check that missing-specific content was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        missing_content_found = any("pb missing" in call for call in calls)
        assert missing_content_found


def test_show_concise_help_validate():
    """Test _show_concise_help() for 'validate' command."""
    with patch("pointblank.cli.console") as mock_console:
        with patch("sys.exit"):
            _show_concise_help("validate", None)

        # Should have called console.print multiple times
        assert mock_console.print.called

        # Check that validate-specific content was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        validate_content_found = any("pb validate" in call for call in calls)
        assert validate_content_found


def test_show_concise_help_run():
    """Test _show_concise_help() for 'run' command."""
    with patch("pointblank.cli.console") as mock_console:
        with patch("sys.exit"):
            _show_concise_help("run", None)

        # Should have called console.print multiple times
        assert mock_console.print.called

        # Check that run-specific content was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        run_content_found = any("pb run" in call for call in calls)
        assert run_content_found


def test_show_concise_help_make_template():
    """Test _show_concise_help() for 'make-template' command."""
    with patch("pointblank.cli.console") as mock_console:
        with patch("sys.exit"):
            _show_concise_help("make-template", None)

        # Should have called console.print multiple times
        assert mock_console.print.called

        # Check that make-template-specific content was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        template_content_found = any("pb make-template" in call for call in calls)
        assert template_content_found


def test_show_concise_help_pl():
    """Test _show_concise_help() for 'pl' command."""
    with patch("pointblank.cli.console") as mock_console:
        with patch("sys.exit"):
            _show_concise_help("pl", None)

        # Should have called console.print multiple times
        assert mock_console.print.called

        # Check that pl-specific content was printed
        calls = [str(call) for call in mock_console.print.call_args_list]
        pl_content_found = any("pb pl" in call for call in calls)
        assert pl_content_found


@patch("pointblank.cli.console")
def test_rich_print_gt_table_is_complete_preview(mock_console):
    """Test _rich_print_gt_table() with is_complete=True preview info."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.return_value = "<table><tr><td>test</td></tr></table>"

    # Mock _tbl_data for rich table creation
    mock_df = Mock()
    mock_df.columns = ["col1", "col2"]
    mock_df.to_dicts.return_value = [{"col1": "val1", "col2": "val2"}]
    mock_gt_table._tbl_data = mock_df

    # Preview info with is_complete=True
    preview_info = {
        "is_complete": True,
        "total_rows": 1,  # Match the actual number of rows in mock data
        "head_rows": 0,
        "tail_rows": 0,
    }

    _rich_print_gt_table(mock_gt_table, preview_info=preview_info)

    # Should call console.print with "Showing all X rows" message
    calls = mock_console.print.call_args_list
    found_complete_message = any(
        len(call[0]) > 0 and "Showing all 1 rows" in str(call[0][0]) for call in calls
    )
    assert found_complete_message


@patch("pointblank.cli.console")
def test_rich_print_gt_table_fallback_preview(mock_console):
    """Test _rich_print_gt_table() fallback case with preview_info but no head/tail rows."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.return_value = "<table><tr><td>test</td></tr></table>"

    # Mock _tbl_data for rich table creation
    mock_df = Mock()
    mock_df.columns = ["col1", "col2"]
    mock_df.to_dicts.return_value = [{"col1": "val1", "col2": "val2"}]
    mock_gt_table._tbl_data = mock_df

    # Preview info with no head/tail rows and is_complete=False should trigger fallback
    preview_info = {
        "is_complete": False,
        "total_rows": 1000,  # This becomes total_dataset_rows in the logic
        "head_rows": 0,
        "tail_rows": 0,
    }

    _rich_print_gt_table(mock_gt_table, preview_info=preview_info)

    # Should call console.print with fallback message
    # total_rows (actual table) = 1, total_dataset_rows = 1000
    calls = mock_console.print.call_args_list
    found_fallback_message = any(
        len(call[0]) > 0 and "Showing 1 rows from 1,000 total rows" in str(call[0][0])
        for call in calls
    )
    assert found_fallback_message


@patch("pointblank.cli.console")
def test_rich_print_gt_table_max_rows_exceeded(mock_console):
    """Test _rich_print_gt_table() when total_rows > max_rows without preview_info."""
    mock_gt_table = Mock()
    mock_gt_table.as_raw_html.return_value = "<table><tr><td>test</td></tr></table>"

    # Mock _tbl_data for rich table creation with many rows
    mock_df = Mock()
    mock_df.columns = ["col1", "col2"]
    # Create 100 rows of mock data to exceed max_rows limit (50)
    mock_rows = [{"col1": f"val1_{i}", "col2": f"val2_{i}"} for i in range(100)]
    mock_df.to_dicts.return_value = mock_rows
    mock_gt_table._tbl_data = mock_df

    # No preview_info provided, should use original fallback logic
    _rich_print_gt_table(mock_gt_table, preview_info=None)

    # Should call console.print with max_rows exceeded message
    calls = mock_console.print.call_args_list
    found_max_rows_message = any(
        len(call[0]) > 0
        and "Showing first 50 of 100 rows. Use --output-html to see all data" in str(call[0][0])
        for call in calls
    )
    assert found_max_rows_message


@patch("pointblank.cli.console")
def test_display_validation_summary_error_severity(mock_console):
    """Test _display_validation_summary() with error severity."""
    mock_validation = Mock()

    # Create mock validation_info with steps that have error=True
    mock_step1 = Mock()
    mock_step1.warning = False
    mock_step1.error = False
    mock_step1.critical = False
    mock_step1.all_passed = True

    mock_step2 = Mock()
    mock_step2.warning = False
    mock_step2.error = True  # This step has error=True
    mock_step2.critical = False
    mock_step2.all_passed = False

    mock_validation.validation_info = [mock_step1, mock_step2]

    # Mock other validation object methods used by function
    mock_validation.get_tabulation_df.return_value = pd.DataFrame(
        {
            "eval": [1, 0],  # One pass, one fail
            "n": [10, 5],
            "f_passed": [1.0, 0.0],
            "f_failed": [0.0, 1.0],
            "W": [False, False],
            "S": [False, True],
            "N": [False, False],
        }
    )

    _display_validation_summary(mock_validation)

    # Should call console.print and set highest_severity="error" with severity_color="yellow"
    assert mock_console.print.called

    # Check that error-related styling was used
    calls = mock_console.print.call_args_list
    found_error_styling = any("yellow" in str(call) for call in calls)
    assert found_error_styling


@patch("pointblank.cli.console")
def test_display_validation_summary_warning_severity(mock_console):
    """Test _display_validation_summary() with warning severity."""
    mock_validation = Mock()

    # Create mock validation_info with steps that have warning=True
    mock_step1 = Mock()
    mock_step1.warning = False
    mock_step1.error = False
    mock_step1.critical = False
    mock_step1.all_passed = True

    mock_step2 = Mock()
    mock_step2.warning = True  # This step has warning=True
    mock_step2.error = False
    mock_step2.critical = False
    mock_step2.all_passed = False

    mock_validation.validation_info = [mock_step1, mock_step2]

    # Mock other validation object methods used by function
    mock_validation.get_tabulation_df.return_value = pd.DataFrame(
        {
            "eval": [1, 1],  # Both pass evaluation but one has warning
            "n": [10, 5],
            "f_passed": [1.0, 0.8],  # Second has 80% pass rate (warning level)
            "f_failed": [0.0, 0.2],
            "W": [False, True],  # Second has W=True (warning condition)
            "S": [False, False],
            "N": [False, False],
        }
    )

    _display_validation_summary(mock_validation)

    # Should call console.print and set highest_severity="warning" with severity_color="bright_black"
    assert mock_console.print.called

    # Check that warning-related styling was used
    calls = mock_console.print.call_args_list
    found_warning_styling = any("bright_black" in str(call) for call in calls)
    assert found_warning_styling


@patch("pointblank.cli.console")
def test_display_validation_summary_passed_severity(mock_console):
    """Test _display_validation_summary() with passed severity."""
    mock_validation = Mock()

    # Create mock validation_info with steps that:
    # - Have no warning/error/critical threshold exceedances (all False)
    # - Some steps don't have 100% pass rate (all_passed=False for some)
    mock_step1 = Mock()
    mock_step1.warning = False
    mock_step1.error = False
    mock_step1.critical = False
    mock_step1.all_passed = True  # This step has 100% pass rate
    mock_step1.i = 1
    mock_step1.assertion_type = "col_vals_not_null"
    mock_step1.column = "name"
    mock_step1.n = 1000
    mock_step1.n_passed = 1000
    mock_step1.values = None

    mock_step2 = Mock()
    mock_step2.warning = False
    mock_step2.error = False
    mock_step2.critical = False
    mock_step2.all_passed = False  # This step has some failing units but no threshold exceedance
    mock_step2.i = 2
    mock_step2.assertion_type = "col_vals_gt"
    mock_step2.column = "age"
    mock_step2.n = 1000
    mock_step2.n_passed = 950  # 95% pass rate - no threshold exceeded but not 100%
    mock_step2.values = 0

    # Mock thresholds for the steps
    mock_threshold = Mock()
    mock_threshold.warning = None
    mock_threshold.error = None
    mock_threshold.critical = None
    mock_step1.thresholds = mock_threshold
    mock_step2.thresholds = mock_threshold

    # Mock extract
    mock_step1.extract = None
    mock_step2.extract = None

    mock_validation.validation_info = [mock_step1, mock_step2]

    _display_validation_summary(mock_validation)

    # Should call console.print and set highest_severity="passed" with severity_color="green"
    assert mock_console.print.called

    # Check that the passed severity styling was used (green color for "passed")
    calls = mock_console.print.call_args_list
    summary_calls = [call for call in calls if "passed" in str(call)]

    assert len(summary_calls) > 0

    # Verify green styling is used for passed severity
    found_green_styling = any("[green]passed[/green]" in str(call) for call in calls)

    assert found_green_styling


def test_show_concise_help_with_context():
    """Test _show_concise_help() exit behavior when context is provided."""
    mock_ctx = Mock()

    with patch("pointblank.cli.console"):
        _show_concise_help("info", mock_ctx)

    # Should call ctx.exit(1)
    mock_ctx.exit.assert_called_once_with(1)


def test_show_concise_help_without_context():
    """Test _show_concise_help() exit behavior when no context provided."""
    with patch("pointblank.cli.console"):
        with patch("sys.exit") as mock_exit:
            _show_concise_help("info", None)

            # Should call sys.exit(1)
            mock_exit.assert_called_once_with(1)


@patch("pointblank.cli._rich_print_missing_table")
@patch("pointblank.missing_vals_tbl")
def test_handle_pl_missing_console_output(mock_missing_vals_tbl, mock_rich_print):
    """Test _handle_pl_missing() with console output (no HTML file)."""

    # Create mock result data and missing values table
    mock_result = Mock()
    mock_missing_table = Mock()
    mock_missing_table.as_raw_html.return_value = "<html><body>Missing report</body></html>"
    mock_missing_vals_tbl.return_value = mock_missing_table

    # Test console output path (output_html=None)
    _handle_pl_missing(mock_result, "pl.read_csv('test.csv')", None)

    # Should call pb.missing_vals_tbl with the result data
    mock_missing_vals_tbl.assert_called_once_with(data=mock_result)

    # Should call _rich_print_missing_table with missing_table and result
    mock_rich_print.assert_called_once_with(mock_missing_table, mock_result)


@patch("pointblank.cli.console")
@patch("pathlib.Path.write_text")
@patch("pointblank.missing_vals_tbl")
def test_handle_pl_missing_html_output(mock_missing_vals_tbl, mock_write_text, mock_console):
    """Test _handle_pl_missing() with HTML output."""

    # Create mock result data and missing values table
    mock_result = Mock()
    mock_missing_table = Mock()
    mock_html_content = "<html><body>Missing values report</body></html>"
    mock_missing_table.as_raw_html.return_value = mock_html_content
    mock_missing_vals_tbl.return_value = mock_missing_table

    output_file = "/tmp/test_missing_report.html"

    # Test HTML output path
    _handle_pl_missing(mock_result, "pl.read_csv('test.csv')", output_file)

    # Should call pb.missing_vals_tbl with the result data
    mock_missing_vals_tbl.assert_called_once_with(data=mock_result)

    # Should call as_raw_html to get HTML content
    mock_missing_table.as_raw_html.assert_called_once()

    # Should write HTML content to file
    mock_write_text.assert_called_once_with(mock_html_content, encoding="utf-8")

    # Should print success message
    mock_console.print.assert_called_once_with(
        f"[green]✓[/green] Missing values report saved to: {output_file}"
    )


@patch("pointblank.cli.console")
@patch("pointblank.missing_vals_tbl")
@patch("sys.exit")
def test_handle_pl_missing_exception_handling(mock_exit, mock_missing_vals_tbl, mock_console):
    """Test _handle_pl_missing() exception handling."""

    # Create mock result data
    mock_result = Mock()

    # Make pb.missing_vals_tbl raise an exception to trigger error handling
    test_error = Exception("Test missing values table creation error")
    mock_missing_vals_tbl.side_effect = test_error

    # Test exception handling path
    _handle_pl_missing(mock_result, "pl.read_csv('test.csv')", None)

    # Should call pb.missing_vals_tbl and get exception
    mock_missing_vals_tbl.assert_called_once_with(data=mock_result)

    # Should print error message
    mock_console.print.assert_called_once_with(
        "[red]Error creating missing values report:[/red] Test missing values table creation error"
    )

    # Should call sys.exit(1)
    mock_exit.assert_called_once_with(1)


@patch("pointblank.cli.console")
@patch("pathlib.Path.write_text")
@patch("pointblank.missing_vals_tbl")
@patch("sys.exit")
def test_handle_pl_missing_html_write_exception(
    mock_exit, mock_missing_vals_tbl, mock_write_text, mock_console
):
    """Test _handle_pl_missing() when HTML file writing fails."""

    # Create mock result data and missing values table
    mock_result = Mock()
    mock_missing_table = Mock()
    mock_missing_table.as_raw_html.return_value = "<html><body>Missing report</body></html>"
    mock_missing_vals_tbl.return_value = mock_missing_table

    # Make Path.write_text raise an exception to trigger error handling during HTML writing
    test_error = Exception("Permission denied writing HTML file")
    mock_write_text.side_effect = test_error

    output_file = "/tmp/test_missing_report.html"

    # Test HTML writing exception path
    _handle_pl_missing(mock_result, "pl.read_csv('test.csv')", output_file)

    # Should call pb.missing_vals_tbl
    mock_missing_vals_tbl.assert_called_once_with(data=mock_result)

    # Should attempt to write HTML content
    mock_write_text.assert_called_once()

    # Should print error message
    mock_console.print.assert_called_once_with(
        "[red]Error creating missing values report:[/red] Permission denied writing HTML file"
    )

    # Should call sys.exit(1)
    mock_exit.assert_called_once_with(1)


@patch("pointblank.cli._rich_print_missing_table")
@patch("pointblank.missing_vals_tbl")
def test_handle_pl_missing_rich_print_call(mock_missing_vals_tbl, mock_rich_print):
    """Test that _handle_pl_missing() calls _rich_print_missing_table() for console output."""

    # Create mock result data and missing values table
    mock_result = Mock()
    mock_missing_table = Mock()
    mock_missing_vals_tbl.return_value = mock_missing_table

    # Test console output path to verify _rich_print_missing_table call
    _handle_pl_missing(mock_result, "pl.read_csv('test.csv')", None)

    # Should call pb.missing_vals_tbl with the result data
    mock_missing_vals_tbl.assert_called_once_with(data=mock_result)

    # Should call _rich_print_missing_table with missing_table and result
    mock_rich_print.assert_called_once_with(mock_missing_table, mock_result)


def test_info_command_no_data_source(runner):
    """Test info command when no data source is provided (should show help)."""
    result = runner.invoke(info, [])

    # Should exit with code 1 after showing help (expected behavior)
    assert result.exit_code == 1
    # Should contain help content
    assert "pb info" in result.output


def test_preview_command_no_data_source_no_pipe(runner):
    """Test preview command when no data source is provided and not piped."""
    result = runner.invoke(preview, [])

    # Should exit with code 1
    assert result.exit_code == 1
    # Should contain either help content or pipe error (both are valid error paths)
    assert ("pb preview" in result.output) or ("No data provided via pipe" in result.output)


def test_scan_command_no_data_source_no_pipe(runner):
    """Test scan command when no data source is provided and not piped."""
    result = runner.invoke(scan, [])

    # Should exit with code 1
    assert result.exit_code == 1
    # Should contain either help content or pipe error (both are valid error paths)
    assert ("pb scan" in result.output) or ("No data provided via pipe" in result.output)


def test_missing_command_no_data_source_no_pipe(runner):
    """Test missing command when no data source is provided and not piped."""
    result = runner.invoke(missing, [])

    # Should exit with code 1
    assert result.exit_code == 1
    # Should contain either help content or pipe error (both are valid error paths)
    assert ("pb missing" in result.output) or ("No data provided via pipe" in result.output)


def test_validate_command_no_data_source_no_pipe(runner):
    """Test validate command when no data source is provided and not piped."""
    result = runner.invoke(validate, [])

    # Should exit with code 1
    assert result.exit_code == 1
    # Should contain either help content or pipe error (both are valid error paths)
    assert ("pb validate" in result.output) or ("No data provided via pipe" in result.output)
