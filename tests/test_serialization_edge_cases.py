import tempfile
from pathlib import Path
from unittest.mock import patch
import pickle

import polars as pl
import pytest

import pointblank as pb


def test_read_file_invalid_object():
    """Test loading a file that doesn't contain a Validate object."""
    # Create a temporary file with non-Validate object
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
        pickle.dump({"not": "validate"}, f)
        temp_path = f.name

    try:
        with pytest.raises(
            RuntimeError, match="Invalid validation file format|Failed to read validation object"
        ):
            pb.read_file(temp_path)
    finally:
        Path(temp_path).unlink()


def test_read_file_missing_file():
    """Test loading a file that doesn't exist."""
    with pytest.raises(FileNotFoundError):
        pb.read_file("nonexistent_file.pkl")


def test_write_file_with_path():
    """Test write_file with specified path parameter."""
    data = pl.DataFrame({"a": [1, 2, 3]})
    validation = pb.Validate(data).col_vals_gt("a", 0)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with path parameter
        filename = "test_validation"
        pb.write_file(validation, filename, path=temp_dir, quiet=True)

        # Check that file was created in the specified path
        expected_path = Path(temp_dir) / f"{filename}.pkl"
        assert expected_path.exists()

        # Verify it can be read back
        loaded_validation = pb.read_file(str(expected_path))
        assert isinstance(loaded_validation, pb.Validate)


def test_write_file_directory_creation():
    """Test that write_file creates directories if they don't exist."""
    data = pl.DataFrame({"a": [1, 2, 3]})
    validation = pb.Validate(data).col_vals_gt("a", 0)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with nested directory that doesn't exist
        nested_path = Path(temp_dir) / "nested" / "subdir"
        pb.write_file(validation, "test_validation", path=str(nested_path), quiet=True)

        # Check that nested directories were created and file exists
        expected_path = nested_path / "test_validation.pkl"
        assert expected_path.exists()


def test_keep_extracts_functionality():
    """Test the keep_extracts parameter functionality."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    # Create a validation and interrogate it to get some results with potential extracts
    validation = (
        pb.Validate(data)
        .col_vals_gt("a", 5)  # This should fail for some rows
        .interrogate()
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with keep_extracts=True
        pb.write_file(
            validation, "test_with_extracts", path=temp_dir, keep_extracts=True, quiet=True
        )

        # Test with keep_extracts=False
        pb.write_file(
            validation, "test_without_extracts", path=temp_dir, keep_extracts=False, quiet=True
        )

        # Both should succeed (basic functionality test)
        extract_path = Path(temp_dir) / "test_with_extracts.pkl"
        no_extract_path = Path(temp_dir) / "test_without_extracts.pkl"

        assert extract_path.exists()
        assert no_extract_path.exists()


def test_quiet_parameter():
    """Test the quiet parameter suppresses output."""
    data = pl.DataFrame({"a": [1, 2, 3]})
    validation = pb.Validate(data).col_vals_gt("a", 0)

    with tempfile.TemporaryDirectory() as temp_dir:
        with patch("builtins.print") as mock_print:
            pb.write_file(validation, "test_quiet", path=temp_dir, quiet=True)

            # Should not have printed anything
            mock_print.assert_not_called()


def test_filename_pkl_extension_handling():
    """Test that .pkl extension is added if not present, but preserved if present."""
    data = pl.DataFrame({"a": [1, 2, 3]})
    validation = pb.Validate(data).col_vals_gt("a", 0)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test without .pkl extension
        pb.write_file(validation, "test_file", path=temp_dir, quiet=True)
        assert (Path(temp_dir) / "test_file.pkl").exists()

        # Test with .pkl extension already present
        pb.write_file(validation, "test_file_with_ext.pkl", path=temp_dir, quiet=True)
        assert (Path(temp_dir) / "test_file_with_ext.pkl").exists()
        # Should not create test_file_with_ext.pkl.pkl
        assert not (Path(temp_dir) / "test_file_with_ext.pkl.pkl").exists()


def test_validation_with_no_data():
    """Test serialization of validation without data table."""
    # Create validation without data
    validation = pb.Validate(None)

    with tempfile.TemporaryDirectory() as temp_dir:
        pb.write_file(validation, "no_data_validation", path=temp_dir, quiet=True)
        expected_path = Path(temp_dir) / "no_data_validation.pkl"
        assert expected_path.exists()

        # Verify it can be loaded
        loaded_validation = pb.read_file(str(expected_path))
        assert isinstance(loaded_validation, pb.Validate)
        assert loaded_validation.data is None


def test_keep_tbl_false_removes_data():
    """Test that keep_tbl=False removes data from saved validation."""
    data = pl.DataFrame({"a": [1, 2, 3]})
    validation = pb.Validate(data).col_vals_gt("a", 0)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save without table data
        pb.write_file(validation, "no_table", path=temp_dir, keep_tbl=False, quiet=True)

        # Load and verify data is None
        loaded_validation = pb.read_file(str(Path(temp_dir) / "no_table.pkl"))
        assert loaded_validation.data is None


def test_keep_tbl_true_preserves_data():
    """Test that keep_tbl=True preserves data in saved validation."""
    data = pl.DataFrame({"a": [1, 2, 3]})
    validation = pb.Validate(data).col_vals_gt("a", 0)

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save with table data
        pb.write_file(validation, "with_table", path=temp_dir, keep_tbl=True, quiet=True)

        # Load and verify data is preserved
        loaded_validation = pb.read_file(str(Path(temp_dir) / "with_table.pkl"))
        assert loaded_validation.data is not None
        assert isinstance(loaded_validation.data, pl.DataFrame)


def test_lambda_function_error():
    """Test that lambda functions cause immediate error."""
    data = pl.DataFrame({"a": [1, 2, 3]})
    validation = pb.Validate(data).col_vals_gt(
        "a", 0, pre=lambda df: df.with_columns(pl.col("a") * 2)
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Should raise ValueError about lambda functions
        with pytest.raises(ValueError, match="Cannot serialize validation object.*lambda"):
            pb.write_file(validation, "lambda_test", path=temp_dir, quiet=True)


def test_database_table_removal_message():
    """Test that database table removal message appears when needed."""
    # Create a mock validation with a database-type table
    data = pl.DataFrame({"a": [1, 2, 3]})
    validation = pb.Validate(data)

    # Mock the _get_tbl_type function to return a database type
    with patch("pointblank.validate._get_tbl_type", return_value="duckdb"):
        with patch("builtins.print") as mock_print:
            with tempfile.TemporaryDirectory() as temp_dir:
                pb.write_file(validation, "test_db", path=temp_dir, keep_tbl=True, quiet=False)

                # Should print database removal message (pragma covered, so just exercising path)
                mock_print.assert_called()
