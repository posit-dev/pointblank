"""
Generate reference validation files for serialization regression testing.

This script creates validation objects with various preprocessing functions
and stores them as pickled files in the validations directory. These files
serve as regression tests to ensure serialization compatibility across versions.
"""

import pickle

# Add the parent directory to Python path to import pointblank
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing_functions import (
    add_computed_column,
    complex_preprocessing,
    double_column_a,
    filter_by_d_gt_100,
    narwhals_median_transform,
    pandas_compatible_transform,
)

import pointblank as pb


def create_test_data():
    """Create a test dataset for validation examples."""
    return pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "c": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
            "d": [50, 75, 100, 125, 150, 175, 200, 225, 250, 275],
        }
    )


def create_validation_examples():
    """Create various validation objects for testing serialization."""
    data = create_test_data()
    validations = {}

    # Basic validation with simple preprocessing
    validations["simple_preprocessing"] = (
        pb.Validate(data, tbl_name="test_data")
        .col_vals_gt("a", value=0, pre=double_column_a)
        .col_vals_in_set("c", set=["x", "y"])
    )

    # Validation with complex preprocessing
    validations["complex_preprocessing"] = (
        pb.Validate(data, tbl_name="test_data")
        .col_vals_gt("a_doubled", value=0, pre=complex_preprocessing)
        .col_vals_gt("d_scaled", value=15, pre=complex_preprocessing)
    )

    # Validation with narwhals function
    validations["narwhals_function"] = pb.Validate(data, tbl_name="test_data").col_vals_gt(
        "a", value=5, pre=narwhals_median_transform
    )

    # Validation with multiple preprocessing steps
    validations["multiple_steps"] = (
        pb.Validate(data, tbl_name="test_data")
        .col_vals_gt("a", value=2, pre=double_column_a)
        .col_vals_in_set("c", set=["x", "y"], pre=filter_by_d_gt_100)
        .col_vals_gt("sum_ab", value=100, pre=add_computed_column)
    )

    # Validation with pandas-compatible function
    validations["pandas_compatible"] = pb.Validate(data, tbl_name="test_data").col_vals_gt(
        "a_plus_b", value=10, pre=pandas_compatible_transform
    )

    # Basic validation without preprocessing (control case)
    validations["no_preprocessing"] = (
        pb.Validate(data, tbl_name="test_data")
        .col_vals_gt("a", value=0)
        .col_vals_lt("d", value=300)
        .col_vals_in_set("c", set=["x", "y"])
    )

    return validations


def save_validation_files(validations, output_dir):
    """Save validation objects as pickled files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, validation in validations.items():
        # Interrogate to populate results
        validation.interrogate()

        # Save the validation object
        file_path = output_path / f"{name}.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(validation, f)

        print(f"Saved {name} validation to {file_path}")

        # Also save as JSON for human readability
        json_path = output_path / f"{name}.json"
        try:
            json_report = validation.get_json_report()
            with open(json_path, "w") as f:
                f.write(json_report)
            print(f"Saved {name} validation JSON to {json_path}")
        except Exception as e:
            print(f"Could not save JSON for {name}: {e}")


if __name__ == "__main__":
    # Create validation examples
    validations = create_validation_examples()

    # Save to the validations directory
    output_dir = Path(__file__).parent
    save_validation_files(validations, output_dir)

    print(f"\nCreated {len(validations)} test validation files in {output_dir}")
    print("These files can be used for regression testing serialization compatibility.")
