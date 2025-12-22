import pickle
import sys
from pathlib import Path

import polars as pl
import pytest

import pointblank as pb


# Path to the validation test files
VALIDATIONS_DIR = Path(__file__).parent.parent / "pointblank" / "data" / "validations"


def load_validation_file(filename: str):
    """Load a pickled validation object from the test files directory."""
    # Add the validations directory to path for importing preprocessing_functions
    sys.path.insert(0, str(VALIDATIONS_DIR))

    file_path = VALIDATIONS_DIR / f"{filename}.pkl"
    with open(file_path, "rb") as f:
        return pickle.load(f)


def test_validation_serialization_roundtrip():
    """Test that validation objects can be pickled and unpickled correctly."""
    # Load each test validation file and verify it can be unpickled
    validation_files = [
        "simple_preprocessing",
        "complex_preprocessing",
        "narwhals_function",
        "multiple_steps",
        "pandas_compatible",
        "no_preprocessing",
    ]

    for filename in validation_files:
        validation = load_validation_file(filename)

        # Verify it's a Validate object
        assert isinstance(validation, pb.Validate)

        # Test that we can pickle it again (roundtrip)
        pickled_data = pickle.dumps(validation)
        roundtrip_validation = pickle.loads(pickled_data)

        # Verify the roundtrip object is still a Validate object
        assert isinstance(roundtrip_validation, pb.Validate)

        # Verify basic attributes are preserved
        assert roundtrip_validation.tbl_name == validation.tbl_name
        assert len(roundtrip_validation.validation_info) == len(validation.validation_info)


def test_validation_preprocessing_functions_preserved():
    """Test that preprocessing functions are preserved in serialization."""
    # Test a validation with preprocessing functions
    validation = load_validation_file("simple_preprocessing")

    # Check that preprocessing functions are present
    for step in validation.validation_info:
        if step.pre is not None:
            # Verify the preprocessing function is callable
            assert callable(step.pre)

            # Verify it has the expected function attributes
            assert hasattr(step.pre, "__name__") or hasattr(step.pre, "func")


def test_validation_results_preserved():
    """Test that validation results are preserved after serialization."""
    validation_files = [
        "simple_preprocessing",
        "complex_preprocessing",
        "multiple_steps",
        "no_preprocessing",
    ]

    for filename in validation_files:
        validation = load_validation_file(filename)

        # Verify that validation results exist (should have been interrogated)
        assert hasattr(validation, "validation_info")
        assert validation.validation_info is not None
        assert len(validation.validation_info) > 0

        # Check that we can get reports
        json_report = validation.get_json_report()
        assert isinstance(json_report, str)
        assert len(json_report) > 0


def test_complex_preprocessing_functions():
    """Test that complex preprocessing functions work after deserialization."""
    validation = load_validation_file("complex_preprocessing")

    # Create test data similar to what was used originally
    test_data = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "c": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
            "d": [50, 75, 100, 125, 150, 175, 200, 225, 250, 275],
        }
    )

    # Verify we can create a new validation with similar preprocessing
    # and it behaves consistently
    for step in validation.validation_info:
        if step.pre is not None:
            # Test that the preprocessing function can be applied
            try:
                transformed_data = step.pre(test_data)
                assert isinstance(transformed_data, pl.DataFrame)
                assert len(transformed_data) > 0
            except Exception as e:
                pytest.fail(f"Preprocessing function failed: {e}")


def test_narwhals_function_compatibility():
    """Test that narwhals-based preprocessing functions work after serialization."""
    validation = load_validation_file("narwhals_function")

    # Test that the narwhals function is properly preserved
    test_data = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "d": [50, 75, 100, 125, 150, 175, 200, 225, 250, 275],
        }
    )

    for step in validation.validation_info:
        if step.pre is not None:
            # Apply the narwhals function
            result = step.pre(test_data)
            # The result might be a narwhals DataFrame or polars DataFrame
            assert result is not None
            # Check that we can convert to native
            try:
                import narwhals as nw

                native_result = nw.to_native(result)
                assert isinstance(native_result, (pl.DataFrame, pl.LazyFrame))
            except ImportError:
                # If narwhals is not available, just check it's not None
                assert result is not None


def test_multiple_preprocessing_steps():
    """Test validations with multiple different preprocessing steps."""
    validation = load_validation_file("multiple_steps")

    # Verify that each step can have different preprocessing functions
    preprocessing_functions = []
    for step in validation.validation_info:
        if step.pre is not None:
            preprocessing_functions.append(step.pre)

    # Should have multiple different preprocessing functions
    assert len(preprocessing_functions) >= 2

    # Each should be callable
    for func in preprocessing_functions:
        assert callable(func)
