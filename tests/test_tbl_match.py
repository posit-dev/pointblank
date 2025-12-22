from __future__ import annotations

import pytest
import narwhals as nw
import pandas as pd
import polars as pl
import ibis
import duckdb
import sqlite3

import tempfile
import os
import datetime
import random

from pointblank import Validate, Thresholds


@pytest.fixture
def polars_basic_table():
    """Basic Polars DataFrame for testing."""

    return pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})


@pytest.fixture
def pandas_basic_table():
    """Basic Pandas DataFrame for testing."""

    return pd.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})


@pytest.fixture
def duckdb_basic_table():
    """Basic DuckDB (Ibis) table for testing."""
    con = ibis.duckdb.connect()

    return con.create_table(
        "basic_table",
        {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]},
        overwrite=True,
    )


def test_tbl_match_identical_polars():
    """Test that identical Polars tables pass validation."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()
    assert validation.n_passed(i=1, scalar=True) == 1
    assert validation.n_failed(i=1, scalar=True) == 0


def test_tbl_match_identical_pandas():
    """Test that identical Pandas tables pass validation."""

    tbl_1 = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()
    assert validation.n_passed(i=1, scalar=True) == 1
    assert validation.n_failed(i=1, scalar=True) == 0


def test_tbl_match_identical_mixed_backends():
    """Test that identical data in different backends passes validation with automatic coercion."""

    tbl_polars = pl.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]}
    )

    tbl_pandas = pd.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]}
    )

    # Test Polars data with Pandas comparison (should auto-convert Pandas to Polars)
    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    # Test Pandas data with Polars comparison (should auto-convert Polars to Pandas)
    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


def test_tbl_match_mixed_backends_with_differences():
    """Test that differences are detected even with mixed backends."""

    tbl_polars = pl.DataFrame(
        {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]}
    )

    tbl_pandas = pd.DataFrame(
        {"a": [1, 2, 3, 5], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]}  # Different value
    )

    # Should fail due to data mismatch (stage 6 failure)
    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert not validation.all_passed()
    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == 1


def test_tbl_match_mixed_backends_complex_types():
    """Test mixed backends with complex data types including NaN."""

    tbl_polars = pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.1, float("nan"), 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )

    tbl_pandas = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.1, float("nan"), 3.3],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )

    # Should pass with automatic backend coercion
    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    # Test the other direction
    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


@pytest.mark.skip(reason="Empty table schema matching has edge case issues: to be fixed")
def test_tbl_match_empty_tables():
    """Test that two empty tables with matching schema pass validation."""

    # Create empty tables with explicit schema
    schema = {"a": pl.Int64, "b": pl.String, "c": pl.Float64}
    tbl_1 = pl.DataFrame(schema=schema)
    tbl_2 = pl.DataFrame(schema=schema)

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_single_row():
    """Test that single-row tables match correctly."""

    tbl_1 = pl.DataFrame({"a": [1], "b": ["x"], "c": [2.5]})
    tbl_2 = pl.DataFrame({"a": [1], "b": ["x"], "c": [2.5]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_with_none_values():
    """Test that tables with None values match correctly."""

    tbl_1 = pl.DataFrame(
        {"a": [1, None, 3, 4], "b": ["w", "x", None, "z"], "c": [4.0, 5.0, 6.0, None]}
    )

    tbl_2 = pl.DataFrame(
        {"a": [1, None, 3, 4], "b": ["w", "x", None, "z"], "c": [4.0, 5.0, 6.0, None]}
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_with_nan_values():
    """Test that tables with NaN values match correctly."""

    tbl_1 = pl.DataFrame({"a": [1.0, float("nan"), 3.0, 4.0], "b": [1.0, 2.0, float("nan"), 4.0]})

    tbl_2 = pl.DataFrame({"a": [1.0, float("nan"), 3.0, 4.0], "b": [1.0, 2.0, float("nan"), 4.0]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_different_column_count_fewer():
    """Test that tables with different column counts fail validation (fewer columns)."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()
    assert validation.n_passed(i=1, scalar=True) == 0
    assert validation.n_failed(i=1, scalar=True) == 1


def test_tbl_match_different_column_count_more():
    """Test that tables with different column counts fail validation (more columns)."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    tbl_2 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": ["w", "x", "y", "z"],
            "c": [4.0, 5.0, 6.0, 7.0],
            "d": [10, 20, 30, 40],
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()
    assert validation.n_failed(i=1, scalar=True) == 1


def test_tbl_match_different_row_count_fewer():
    """Test that tables with different row counts fail validation (fewer rows)."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame({"a": [1, 2, 3], "b": ["w", "x", "y"], "c": [4.0, 5.0, 6.0]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()
    assert validation.n_failed(i=1, scalar=True) == 1


def test_tbl_match_different_row_count_more():
    """Test that tables with different row counts fail validation (more rows)."""

    tbl_1 = pl.DataFrame({"a": [1, 2], "b": ["w", "x"], "c": [4.0, 5.0]})

    tbl_2 = pl.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": ["w", "x", "y", "z", "a"], "c": [4.0, 5.0, 6.0, 7.0, 8.0]}
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()
    assert validation.n_failed(i=1, scalar=True) == 1


def test_tbl_match_empty_vs_nonempty():
    """Test that empty vs non-empty tables fail validation."""

    tbl_1 = pl.DataFrame(
        {"a": [], "b": [], "c": []}, schema={"a": pl.Int64, "b": pl.String, "c": pl.Float64}
    )
    tbl_2 = pl.DataFrame({"a": [1, 2, 3], "b": ["w", "x", "y"], "c": [4.0, 5.0, 6.0]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()


def test_tbl_match_different_column_names():
    """Test that tables with different column names fail validation."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": ["w", "x", "y", "z"],
            "d": [4.0, 5.0, 6.0, 7.0],  # Changed 'c' to 'd'
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()
    assert validation.n_failed(i=1, scalar=True) == 1


def test_tbl_match_completely_different_column_names():
    """Test that tables with completely different column names fail validation."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame({"x": [1, 2, 3, 4], "y": ["w", "x", "y", "z"], "z": [4.0, 5.0, 6.0, 7.0]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()


def test_tbl_match_different_column_order_two_swapped():
    """Test that tables with different column order fail validation (two columns swapped)."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "c": [4.0, 5.0, 6.0, 7.0],  # Swapped order with 'b'
            "b": ["w", "x", "y", "z"],
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()
    assert validation.n_failed(i=1, scalar=True) == 1


def test_tbl_match_different_column_order_completely_reversed():
    """Test that tables with completely reversed column order fail validation."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame({"c": [4.0, 5.0, 6.0, 7.0], "b": ["w", "x", "y", "z"], "a": [1, 2, 3, 4]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()


def test_tbl_match_different_column_case_uppercase():
    """Test that tables with different column name case fail validation (uppercase)."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": ["w", "x", "y", "z"],
            "C": [4.0, 5.0, 6.0, 7.0],  # Changed 'c' to 'C'
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()
    assert validation.n_failed(i=1, scalar=True) == 1


def test_tbl_match_different_column_case_mixed():
    """Test that tables with mixed case column names fail validation."""

    tbl_1 = pl.DataFrame(
        {
            "first_name": [1, 2, 3, 4],
            "last_name": ["w", "x", "y", "z"],
            "email": [4.0, 5.0, 6.0, 7.0],
        }
    )

    tbl_2 = pl.DataFrame(
        {
            "First_Name": [1, 2, 3, 4],  # Changed case
            "Last_Name": ["w", "x", "y", "z"],  # Changed case
            "Email": [4.0, 5.0, 6.0, 7.0],  # Changed case
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()


def test_tbl_match_different_data_one_value():
    """Test that tables with one different value fail validation."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": ["w", "x", "y", "z"],
            "c": [4.0, 5.5, 6.0, 7.0],  # Changed 5.0 to 5.5
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()
    assert validation.n_failed(i=1, scalar=True) == 1


def test_tbl_match_different_data_multiple_values():
    """Test that tables with multiple different values fail validation."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame(
        {
            "a": [1, 2, 99, 4],  # Changed 3 to 99
            "b": ["w", "x", "y", "z"],
            "c": [4.0, 5.5, 6.0, 8.0],  # Changed 5.0 to 5.5 and 7.0 to 8.0
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()


def test_tbl_match_different_data_string_values():
    """Test that tables with different string values fail validation."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": ["w", "x", "Y", "z"],  # Changed "y" to "Y"
            "c": [4.0, 5.0, 6.0, 7.0],
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()


def test_tbl_match_none_vs_value():
    """Test that None vs actual value fails validation."""

    tbl_1 = pl.DataFrame({"a": [1, 2, None, 4], "b": ["w", "x", "y", "z"]})

    tbl_2 = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],  # Changed None to 3
            "b": ["w", "x", "y", "z"],
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()


def test_tbl_match_nan_vs_value():
    """Test that NaN vs actual value fails validation."""

    tbl_1 = pl.DataFrame({"a": [1.0, float("nan"), 3.0, 4.0], "b": [1.0, 2.0, 3.0, 4.0]})

    tbl_2 = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0],  # Changed NaN to 2.0
            "b": [1.0, 2.0, 3.0, 4.0],
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert not validation.all_passed()


def test_tbl_match_with_preprocessing():
    """Test tbl_match() with preprocessing applied to target table."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    # Expected table after preprocessing (filter rows where a > 2)
    tbl_2 = pl.DataFrame({"a": [3, 4], "b": ["y", "z"], "c": [6.0, 7.0]})

    validation = (
        Validate(data=tbl_1)
        .tbl_match(tbl_compare=tbl_2, pre=lambda df: df.filter(pl.col("a") > 2))
        .interrogate()
    )

    assert validation.all_passed()


def test_tbl_match_with_preprocessing_narwhals():
    """Test tbl_match() with preprocessing using Narwhals."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    # Expected table after preprocessing (double values in column 'a')
    tbl_2 = pl.DataFrame({"a": [2, 4, 6, 8], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    validation = (
        Validate(data=tbl_1)
        .tbl_match(tbl_compare=tbl_2, pre=lambda dfn: dfn.with_columns(a=nw.col("a") * 2))
        .interrogate()
    )

    assert validation.all_passed()


def test_tbl_match_with_callable_comparison_table():
    """Test tbl_match() with a callable that returns the comparison table."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    def get_comparison_table():
        return pl.DataFrame(
            {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]}
        )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=get_comparison_table).interrogate()

    assert validation.all_passed()


def test_tbl_match_with_lambda_comparison_table():
    """Test tbl_match() with a lambda that returns the comparison table."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    validation = (
        Validate(data=tbl_1)
        .tbl_match(
            tbl_compare=lambda: pl.DataFrame(
                {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]}
            )
        )
        .interrogate()
    )

    assert validation.all_passed()


def test_tbl_match_with_boolean_columns():
    """Test tbl_match() with boolean columns."""

    tbl_1 = pl.DataFrame({"a": [True, False, True, False], "b": [1, 2, 3, 4]})

    tbl_2 = pl.DataFrame({"a": [True, False, True, False], "b": [1, 2, 3, 4]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_with_datetime_columns():
    """Test tbl_match() with datetime columns."""

    tbl_1 = pl.DataFrame(
        {
            "date": [
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2021, 1, 2),
                datetime.datetime(2021, 1, 3),
            ],
            "value": [10, 20, 30],
        }
    )

    tbl_2 = pl.DataFrame(
        {
            "date": [
                datetime.datetime(2021, 1, 1),
                datetime.datetime(2021, 1, 2),
                datetime.datetime(2021, 1, 3),
            ],
            "value": [10, 20, 30],
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_with_list_columns():
    """Test tbl_match() with list/array columns."""

    tbl_1 = pl.DataFrame({"id": [1, 2, 3], "values": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})

    tbl_2 = pl.DataFrame({"id": [1, 2, 3], "values": [[1, 2, 3], [4, 5, 6], [7, 8, 9]]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_with_large_table():
    """Test tbl_match() with a large table (10,000 rows)."""

    random.seed(14323)

    n = 10000
    tbl_1 = pl.DataFrame(
        {
            "id": list(range(n)),
            "value": [random.random() for _ in range(n)],
            "category": [random.choice(["A", "B", "C", "D"]) for _ in range(n)],
        }
    )

    # Create identical copy
    tbl_2 = tbl_1.clone()

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_with_many_columns():
    """Test tbl_match() with many columns (50 columns)."""

    data = {f"col_{i}": [i, i + 1, i + 2, i + 3] for i in range(50)}

    tbl_1 = pl.DataFrame(data)
    tbl_2 = pl.DataFrame(data)

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_multiple_steps():
    """Test multiple tbl_match() validation steps in sequence."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    tbl_2 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    tbl_3 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    validation = (
        Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).tbl_match(tbl_compare=tbl_3).interrogate()
    )

    assert validation.all_passed()
    assert validation.n() == {1: 1, 2: 1}
    assert validation.n_passed() == {1: 1, 2: 1}


def test_tbl_match_with_other_validations():
    """Test tbl_match() combined with other validation methods."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    tbl_2 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]})

    validation = (
        Validate(data=tbl_1)
        .col_exists(columns=["a", "b", "c"])
        .row_count_match(count=4)
        .col_count_match(count=3)
        .tbl_match(tbl_compare=tbl_2)
        .interrogate()
    )

    assert validation.all_passed()


def test_tbl_match_with_thresholds():
    """Test tbl_match() with threshold levels."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    tbl_2 = pl.DataFrame(
        {
            "a": [1, 2, 3, 5],  # Different value
            "b": ["w", "x", "y", "z"],
        }
    )

    thresholds = Thresholds(warning=1, error=1, critical=1)

    validation = (
        Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2, thresholds=thresholds).interrogate()
    )

    assert not validation.all_passed()
    assert validation.validation_info[0].warning
    assert validation.validation_info[0].error
    assert validation.validation_info[0].critical


def test_tbl_match_with_brief():
    """Test tbl_match() with custom brief description."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    tbl_2 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    validation = (
        Validate(data=tbl_1)
        .tbl_match(tbl_compare=tbl_2, brief="Custom validation description")
        .interrogate()
    )

    assert validation.all_passed()
    assert validation.validation_info[0].brief == "Custom validation description"


def test_tbl_match_with_active_false():
    """Test tbl_match() with active=False (validation should be skipped)."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    tbl_2 = pl.DataFrame(
        {
            "a": [99, 99, 99, 99],  # Completely different
            "b": ["a", "b", "c", "d"],
        }
    )

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2, active=False).interrogate()

    # Since validation is inactive, it should still be in validation_info but not executed
    assert len(validation.validation_info) == 1
    assert validation.validation_info[0].active is False


def test_tbl_match_single_column():
    """Test tbl_match() with single column tables."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4]})
    tbl_2 = pl.DataFrame({"a": [1, 2, 3, 4]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_with_duplicate_rows():
    """Test tbl_match() with tables containing duplicate rows."""

    tbl_1 = pl.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]})

    tbl_2 = pl.DataFrame({"a": [1, 1, 2, 2], "b": ["x", "x", "y", "y"]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_with_special_characters_in_column_names():
    """Test tbl_match() with special characters in column names."""

    tbl_1 = pl.DataFrame({"col_1": [1, 2, 3], "col-2": [4, 5, 6], "col.3": [7, 8, 9]})

    tbl_2 = pl.DataFrame({"col_1": [1, 2, 3], "col-2": [4, 5, 6], "col.3": [7, 8, 9]})

    validation = Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate()

    assert validation.all_passed()


def test_tbl_match_assert_passing():
    """Test tbl_match() with assert_passing() method."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    tbl_2 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    # Should not raise an exception
    (Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate().assert_passing())


def test_tbl_match_assert_passing_failure():
    """Test tbl_match() with assert_passing() method when validation fails."""

    tbl_1 = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})

    tbl_2 = pl.DataFrame(
        {
            "a": [1, 2, 3, 5],  # Different value
            "b": ["w", "x", "y", "z"],
        }
    )

    with pytest.raises(AssertionError):
        (Validate(data=tbl_1).tbl_match(tbl_compare=tbl_2).interrogate().assert_passing())


# ==============================================================================
# Cross-Backend Validation Tests
# ==============================================================================


def test_tbl_match_cross_backend_integers_polars_to_pandas():
    """Test integer columns work across Polars to Pandas conversion."""

    tbl_polars = pl.DataFrame({"int8": [1, 2, 3], "int64": [100, 200, 300]})
    tbl_pandas = pd.DataFrame({"int8": [1, 2, 3], "int64": [100, 200, 300]})

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_integers_pandas_to_polars():
    """Test integer columns work across Pandas to Polars conversion."""

    tbl_pandas = pd.DataFrame({"int8": [1, 2, 3], "int64": [100, 200, 300]})
    tbl_polars = pl.DataFrame({"int8": [1, 2, 3], "int64": [100, 200, 300]})

    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_floats_with_nan():
    """Test float columns with NaN values work across backends."""

    tbl_polars = pl.DataFrame(
        {
            "float32": [1.5, float("nan"), 3.5],
            "float64": [10.123, 20.456, float("nan")],
        }
    )
    tbl_pandas = pd.DataFrame(
        {
            "float32": [1.5, float("nan"), 3.5],
            "float64": [10.123, 20.456, float("nan")],
        }
    )

    # Polars data with Pandas comparison
    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    # Pandas data with Polars comparison
    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_strings():
    """Test string columns work across backends."""

    tbl_polars = pl.DataFrame(
        {
            "strings": ["hello", "world", "test"],
            "empty": ["", "non-empty", ""],
            "special": ["café", "naïve", "résumé"],
        }
    )
    tbl_pandas = pd.DataFrame(
        {
            "strings": ["hello", "world", "test"],
            "empty": ["", "non-empty", ""],
            "special": ["café", "naïve", "résumé"],
        }
    )

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_booleans():
    """Test boolean columns work across backends."""

    tbl_polars = pl.DataFrame(
        {
            "bool1": [True, False, True],
            "bool2": [False, False, True],
            "bool3": [True, True, True],
        }
    )
    tbl_pandas = pd.DataFrame(
        {
            "bool1": [True, False, True],
            "bool2": [False, False, True],
            "bool3": [True, True, True],
        }
    )

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_none_values():
    """Test None/null values work across backends."""

    # Note: For nullable integers, Pandas converts to float, so we use float types
    tbl_polars = pl.DataFrame(
        {
            "nullable_float": [1.0, None, 3.0],
            "nullable_str": ["a", None, "c"],
            "nullable_float2": [1.1, 2.2, None],
        }
    )
    tbl_pandas = pd.DataFrame(
        {
            "nullable_float": [1.0, None, 3.0],
            "nullable_str": ["a", None, "c"],
            "nullable_float2": [1.1, 2.2, None],
        }
    )

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_mixed_nulls_and_nans():
    """Test mixed None and NaN values work across backends."""

    tbl_polars = pl.DataFrame(
        {
            "col1": [1.0, None, float("nan"), 4.0],
            "col2": [None, 2.0, 3.0, float("nan")],
        }
    )
    tbl_pandas = pd.DataFrame(
        {
            "col1": [1.0, None, float("nan"), 4.0],
            "col2": [None, 2.0, 3.0, float("nan")],
        }
    )

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_list_columns():
    """Test list columns work across backends (basic types)."""

    tbl_polars = pl.DataFrame(
        {
            "list_int": [[1, 2], [3, 4], [5, 6]],
            "list_str": [["a", "b"], ["c", "d"], ["e", "f"]],
        }
    )
    tbl_pandas = pd.DataFrame(
        {
            "list_int": [[1, 2], [3, 4], [5, 6]],
            "list_str": [["a", "b"], ["c", "d"], ["e", "f"]],
        }
    )

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_large_dataset():
    """Test cross-backend validation with larger dataset."""
    n = 1000

    random.seed(9823462)

    values = [random.random() for _ in range(n)]

    tbl_polars = pl.DataFrame(
        {
            "id": list(range(n)),
            "value": values,
            "category": [f"cat_{i % 10}" for i in range(n)],
        }
    )
    tbl_pandas = pd.DataFrame(
        {
            "id": list(range(n)),
            "value": values,
            "category": [f"cat_{i % 10}" for i in range(n)],
        }
    )

    # Both tables have same data, test cross-backend comparison
    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_column_order_mismatch():
    """Test that column order mismatches are detected across backends."""
    tbl_polars = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    tbl_pandas = pd.DataFrame({"a": [1, 2], "c": [5, 6], "b": [3, 4]})  # Different order

    # Should fail at stage 4 (column order check)
    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert not validation.all_passed()


def test_tbl_match_cross_backend_column_name_case():
    """Test that column name case differences are detected across backends."""
    tbl_polars = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    tbl_pandas = pd.DataFrame({"A": [1, 2], "b": [3, 4]})  # Different case

    # Should fail at stage 5 (case-sensitive check)
    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert not validation.all_passed()


def test_tbl_match_cross_backend_data_type_mismatch():
    """Test that data type mismatches are detected across backends."""
    tbl_polars = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    tbl_pandas = pd.DataFrame({"a": ["1", "2", "3"], "b": ["x", "y", "z"]})  # Different dtype

    # Should fail at schema check stage
    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert not validation.all_passed()


def test_tbl_match_cross_backend_with_preprocessing():
    """Test cross-backend validation with preprocessing."""

    def sort_by_id(tbl):
        return nw.from_native(tbl).sort("id").to_native()

    tbl_polars = pl.DataFrame({"id": [3, 1, 2], "value": [30, 10, 20]})
    tbl_pandas = pd.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]})

    # With preprocessing to sort both tables
    validation = (
        Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas, pre=sort_by_id).interrogate()
    )

    assert validation.all_passed()


def test_tbl_match_cross_backend_timestamp_strings():
    """Test datetime-like strings work across backends (as strings)."""
    tbl_polars = pl.DataFrame(
        {
            "timestamp_str": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "time_str": ["12:00:00", "13:30:00", "14:45:00"],
        }
    )
    tbl_pandas = pd.DataFrame(
        {
            "timestamp_str": ["2021-01-01", "2021-01-02", "2021-01-03"],
            "time_str": ["12:00:00", "13:30:00", "14:45:00"],
        }
    )

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_edge_case_single_column():
    """Test cross-backend validation with single column."""
    tbl_polars = pl.DataFrame({"only_col": [1, 2, 3, 4, 5]})
    tbl_pandas = pd.DataFrame({"only_col": [1, 2, 3, 4, 5]})

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_edge_case_single_row():
    """Test cross-backend validation with single row."""
    tbl_polars = pl.DataFrame({"a": [1], "b": [2], "c": [3]})
    tbl_pandas = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


def test_tbl_match_cross_backend_wide_table():
    """Test cross-backend validation with many columns."""
    n_cols = 50
    data = {f"col_{i}": [i, i + 1, i + 2] for i in range(n_cols)}

    tbl_polars = pl.DataFrame(data)
    tbl_pandas = pd.DataFrame(data)

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


# ==============================================================================
# Database Backend Tests (DuckDB, SQLite, Ibis)
# ==============================================================================


def test_tbl_match_duckdb_native_vs_polars():
    """Test DuckDB (native) vs Polars comparison."""

    # Create test data
    data = {
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True],
    }

    tbl_polars = pl.DataFrame(data)

    # Create DuckDB native table
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE test_table AS SELECT * FROM tbl_polars")
    tbl_duckdb = con.execute("SELECT * FROM test_table").fetchdf()
    con.close()

    # DuckDB returns Pandas DataFrame, so this tests Pandas<->Polars
    validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_duckdb).interrogate()
    assert validation.all_passed()


def test_tbl_match_duckdb_ibis_vs_polars():
    """Test DuckDB (as Ibis table) vs Polars comparison."""

    # Create test data
    data = {
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"],
    }

    tbl_polars = pl.DataFrame(data)

    # Create DuckDB connection and Ibis table
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE test_table (int_col INTEGER, float_col DOUBLE, str_col VARCHAR)")
    con.execute("INSERT INTO test_table VALUES (1, 1.5, 'a'), (2, 2.5, 'b'), (3, 3.5, 'c')")

    # Create Ibis connection
    ibis_con = ibis.duckdb.connect(":memory:")
    ibis_con.raw_sql("CREATE TABLE test_table (int_col INTEGER, float_col DOUBLE, str_col VARCHAR)")
    ibis_con.raw_sql("INSERT INTO test_table VALUES (1, 1.5, 'a'), (2, 2.5, 'b'), (3, 3.5, 'c')")
    tbl_ibis = ibis_con.table("test_table")

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_ibis).interrogate()
    assert validation.all_passed()

    # Note: Ibis tables are handled through Narwhals, so this should work
    validation = Validate(data=tbl_ibis).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()

    con.close()


def test_tbl_match_duckdb_native_vs_pandas():
    """Test DuckDB (native) vs Pandas comparison."""

    # Create test data
    data = {
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"],
        "bool_col": [True, False, True],
    }

    tbl_pandas = pd.DataFrame(data)

    # Create DuckDB native table and fetch as DataFrame
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE test_table AS SELECT * FROM tbl_pandas")
    tbl_duckdb = con.execute("SELECT * FROM test_table").fetchdf()
    con.close()

    # Both are Pandas DataFrames, should match perfectly
    validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_duckdb).interrogate()
    assert validation.all_passed()


def test_tbl_match_duckdb_ibis_vs_pandas():
    """Test DuckDB (as Ibis table) vs Pandas comparison."""

    # Create test data
    data = {
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"],
    }

    tbl_pandas = pd.DataFrame(data)

    # Create Ibis DuckDB connection and table
    ibis_con = ibis.duckdb.connect(":memory:")
    ibis_con.raw_sql("CREATE TABLE test_table (int_col INTEGER, float_col DOUBLE, str_col VARCHAR)")
    ibis_con.raw_sql("INSERT INTO test_table VALUES (1, 1.5, 'a'), (2, 2.5, 'b'), (3, 3.5, 'c')")
    tbl_ibis = ibis_con.table("test_table")

    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_ibis).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_ibis).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()


def test_tbl_match_sqlite_vs_pandas():
    """Test SQLite vs Pandas comparison."""

    # Create test data
    data = {
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"],
    }

    tbl_pandas = pd.DataFrame(data)

    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # Create SQLite table
        con = sqlite3.connect(db_path)
        tbl_pandas.to_sql("test_table", con, index=False, if_exists="replace")

        # Read back from SQLite
        tbl_sqlite = pd.read_sql("SELECT * FROM test_table", con)
        con.close()

        # Both are Pandas DataFrames
        validation = Validate(data=tbl_sqlite).tbl_match(tbl_compare=tbl_pandas).interrogate()
        assert validation.all_passed()

        validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_sqlite).interrogate()
        assert validation.all_passed()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_tbl_match_sqlite_vs_polars():
    """Test SQLite vs Polars comparison."""

    # Create test data
    data = {
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"],
    }

    tbl_polars = pl.DataFrame(data)

    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # Create SQLite table
        con = sqlite3.connect(db_path)
        tbl_pandas_temp = tbl_polars.to_pandas()
        tbl_pandas_temp.to_sql("test_table", con, index=False, if_exists="replace")

        # Read back from SQLite as Pandas, then compare with Polars
        tbl_sqlite = pd.read_sql("SELECT * FROM test_table", con)
        con.close()

        # Cross-backend: SQLite (Pandas) vs Polars
        validation = Validate(data=tbl_sqlite).tbl_match(tbl_compare=tbl_polars).interrogate()
        assert validation.all_passed()

        validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_sqlite).interrogate()
        assert validation.all_passed()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_tbl_match_sqlite_vs_duckdb_native():
    """Test SQLite vs DuckDB (native) comparison."""

    # Create test data
    data = {
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"],
    }

    tbl_pandas = pd.DataFrame(data)

    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # Create SQLite table
        con_sqlite = sqlite3.connect(db_path)
        tbl_pandas.to_sql("test_table", con_sqlite, index=False, if_exists="replace")
        tbl_sqlite = pd.read_sql("SELECT * FROM test_table", con_sqlite)
        con_sqlite.close()

        # Create DuckDB table
        con_duckdb = duckdb.connect(":memory:")
        con_duckdb.execute("CREATE TABLE test_table AS SELECT * FROM tbl_pandas")
        tbl_duckdb = con_duckdb.execute("SELECT * FROM test_table").fetchdf()
        con_duckdb.close()

        # Both return Pandas DataFrames
        validation = Validate(data=tbl_sqlite).tbl_match(tbl_compare=tbl_duckdb).interrogate()
        assert validation.all_passed()

        validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_sqlite).interrogate()
        assert validation.all_passed()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_tbl_match_sqlite_vs_duckdb_ibis():
    """Test SQLite vs DuckDB (as Ibis table) comparison."""

    # Create test data
    data = {
        "int_col": [1, 2, 3],
        "float_col": [1.5, 2.5, 3.5],
        "str_col": ["a", "b", "c"],
    }

    tbl_pandas = pd.DataFrame(data)

    # Create temporary SQLite database
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as f:
        db_path = f.name

    try:
        # Create SQLite table
        con_sqlite = sqlite3.connect(db_path)
        tbl_pandas.to_sql("test_table", con_sqlite, index=False, if_exists="replace")
        tbl_sqlite = pd.read_sql("SELECT * FROM test_table", con_sqlite)
        con_sqlite.close()

        # Create DuckDB Ibis table
        ibis_con = ibis.duckdb.connect(":memory:")
        ibis_con.raw_sql(
            "CREATE TABLE test_table (int_col INTEGER, float_col DOUBLE, str_col VARCHAR)"
        )
        ibis_con.raw_sql(
            "INSERT INTO test_table VALUES (1, 1.5, 'a'), (2, 2.5, 'b'), (3, 3.5, 'c')"
        )
        tbl_ibis = ibis_con.table("test_table")

        # SQLite (Pandas) vs DuckDB Ibis
        validation = Validate(data=tbl_sqlite).tbl_match(tbl_compare=tbl_ibis).interrogate()
        assert validation.all_passed()

        validation = Validate(data=tbl_ibis).tbl_match(tbl_compare=tbl_sqlite).interrogate()
        assert validation.all_passed()
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


def test_tbl_match_database_with_null_values():
    """Test that database backends handle NULL values correctly."""

    # Create data with NULL values
    data = {
        "int_col": [1, None, 3],
        "float_col": [1.5, 2.5, None],
        "str_col": ["a", None, "c"],
    }

    tbl_polars = pl.DataFrame(data)
    tbl_pandas = pd.DataFrame(data)

    # DuckDB with NULL values
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE test_table AS SELECT * FROM tbl_pandas")
    tbl_duckdb = con.execute("SELECT * FROM test_table").fetchdf()
    con.close()

    # Test DuckDB vs Polars with NULL values
    validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_duckdb).interrogate()
    assert validation.all_passed()


def test_tbl_match_database_large_dataset():
    """Test database backends with larger dataset."""

    # Create larger dataset
    n = 500
    data = {
        "id": list(range(n)),
        "value": [float(i) * 1.5 for i in range(n)],
        "category": [f"cat_{i % 10}" for i in range(n)],
    }

    tbl_polars = pl.DataFrame(data)
    tbl_pandas = pd.DataFrame(data)

    # DuckDB
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE test_table AS SELECT * FROM tbl_pandas")
    tbl_duckdb = con.execute("SELECT * FROM test_table").fetchdf()
    con.close()

    validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_duckdb).interrogate()
    assert validation.all_passed()


def test_tbl_match_duckdb_native_vs_polars():
    """Test DuckDB native connection vs Polars DataFrame."""

    # Create test data
    tbl_polars = pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )

    # Create DuckDB connection and table
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE test_table AS
        SELECT * FROM tbl_polars
    """
    )
    tbl_duckdb = conn.execute("SELECT * FROM test_table").df()

    # Test both directions
    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_duckdb).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()

    conn.close()


def test_tbl_match_duckdb_ibis_vs_polars():
    """Test DuckDB as Ibis table vs Polars DataFrame."""

    # Create test data
    tbl_polars = pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        }
    )

    # Create DuckDB Ibis connection
    conn = ibis.duckdb.connect(":memory:")
    conn.create_table("test_table", tbl_polars.to_pandas())
    tbl_ibis = conn.table("test_table")

    # Test both directions
    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_ibis).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_ibis).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()


def test_tbl_match_duckdb_native_vs_pandas():
    """Test DuckDB native connection vs Pandas DataFrame."""

    # Create test data
    tbl_pandas = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
            "bool_col": [True, False, True],
        }
    )

    # Create DuckDB connection and table
    conn = duckdb.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE test_table AS
        SELECT * FROM tbl_pandas
    """
    )
    tbl_duckdb = conn.execute("SELECT * FROM test_table").df()

    # Test both directions
    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_duckdb).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()

    conn.close()


def test_tbl_match_duckdb_ibis_vs_pandas():
    """Test DuckDB as Ibis table vs Pandas DataFrame."""

    # Create test data
    tbl_pandas = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        }
    )

    # Create DuckDB Ibis connection
    conn = ibis.duckdb.connect(":memory:")
    conn.create_table("test_table", tbl_pandas)
    tbl_ibis = conn.table("test_table")

    # Test both directions
    validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_ibis).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_ibis).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()


def test_tbl_match_sqlite_vs_pandas():
    """Test SQLite (via Ibis) vs Pandas DataFrame."""

    # Create test data
    tbl_pandas = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        }
    )

    # Create temporary SQLite database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        conn = ibis.sqlite.connect(db_path)
        conn.create_table("test_table", tbl_pandas)
        tbl_sqlite = conn.table("test_table")

        # Test both directions
        validation = Validate(data=tbl_pandas).tbl_match(tbl_compare=tbl_sqlite).interrogate()
        assert validation.all_passed()

        validation = Validate(data=tbl_sqlite).tbl_match(tbl_compare=tbl_pandas).interrogate()
        assert validation.all_passed()


def test_tbl_match_sqlite_vs_polars():
    """Test SQLite (via Ibis) vs Polars DataFrame."""

    # Create test data
    tbl_polars = pl.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        }
    )

    # Create temporary SQLite database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        conn = ibis.sqlite.connect(db_path)
        conn.create_table("test_table", tbl_polars.to_pandas())
        tbl_sqlite = conn.table("test_table")

        # Test both directions
        validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_sqlite).interrogate()
        assert validation.all_passed()

        validation = Validate(data=tbl_sqlite).tbl_match(tbl_compare=tbl_polars).interrogate()
        assert validation.all_passed()


def test_tbl_match_sqlite_vs_duckdb_native():
    """Test SQLite (via Ibis) vs DuckDB native connection."""

    # Create test data as Pandas (common format)
    test_data = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        }
    )

    # Create SQLite database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        conn_sqlite = ibis.sqlite.connect(db_path)
        conn_sqlite.create_table("test_table", test_data)
        tbl_sqlite = conn_sqlite.table("test_table")

        # Create DuckDB table
        conn_duckdb = duckdb.connect(":memory:")
        conn_duckdb.execute("CREATE TABLE test_table AS SELECT * FROM test_data")
        tbl_duckdb = conn_duckdb.execute("SELECT * FROM test_table").df()

        # Test both directions
        validation = Validate(data=tbl_sqlite).tbl_match(tbl_compare=tbl_duckdb).interrogate()
        assert validation.all_passed()

        validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_sqlite).interrogate()
        assert validation.all_passed()

        conn_duckdb.close()


def test_tbl_match_sqlite_vs_duckdb_ibis():
    """Test SQLite (via Ibis) vs DuckDB (via Ibis)."""

    # Create test data
    test_data = pd.DataFrame(
        {
            "int_col": [1, 2, 3],
            "float_col": [1.5, 2.5, 3.5],
            "str_col": ["a", "b", "c"],
        }
    )

    # Create SQLite database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        conn_sqlite = ibis.sqlite.connect(db_path)
        conn_sqlite.create_table("test_table", test_data)
        tbl_sqlite = conn_sqlite.table("test_table")

        # Create DuckDB Ibis connection
        conn_duckdb = ibis.duckdb.connect(":memory:")
        conn_duckdb.create_table("test_table", test_data)
        tbl_duckdb = conn_duckdb.table("test_table")

        # Test both directions
        validation = Validate(data=tbl_sqlite).tbl_match(tbl_compare=tbl_duckdb).interrogate()
        assert validation.all_passed()

        validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_sqlite).interrogate()
        assert validation.all_passed()


def test_tbl_match_database_with_null_values():
    """Test database backends with NULL values."""

    # Create test data with nulls
    tbl_pandas = pd.DataFrame(
        {
            "int_col": [1, None, 3],
            "float_col": [1.5, 2.5, None],
            "str_col": ["a", None, "c"],
        }
    )

    # DuckDB with nulls
    conn_duckdb = ibis.duckdb.connect(":memory:")
    conn_duckdb.create_table("test_table", tbl_pandas)
    tbl_duckdb = conn_duckdb.table("test_table")

    # Polars with nulls - NOTE: int_col uses Float64 to match DuckDB's behavior
    # (DuckDB materializes nullable integers as float64)
    tbl_polars = pl.DataFrame(
        {
            "int_col": [1.0, None, 3.0],  # Use float to match DuckDB's schema
            "float_col": [1.5, 2.5, None],
            "str_col": ["a", None, "c"],
        }
    )

    # Test database vs Polars
    validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()

    # Test database vs Pandas
    validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_pandas).interrogate()
    assert validation.all_passed()


def test_tbl_match_database_with_multiple_types():
    """Test database backends with various column types."""

    # Create comprehensive test data
    test_data = pd.DataFrame(
        {
            "int_col": [1, 2, 3, 4, 5],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5],
            "str_col": ["apple", "banana", "cherry", "date", "elderberry"],
            "bool_col": [True, False, True, False, True],
        }
    )

    # Create DuckDB table
    conn_duckdb = ibis.duckdb.connect(":memory:")
    conn_duckdb.create_table("test_table", test_data)
    tbl_duckdb = conn_duckdb.table("test_table")

    # Create Polars table
    tbl_polars = pl.from_pandas(test_data)

    # Test DuckDB Ibis vs Polars
    validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=tbl_polars).interrogate()
    assert validation.all_passed()

    validation = Validate(data=tbl_polars).tbl_match(tbl_compare=tbl_duckdb).interrogate()
    assert validation.all_passed()

    # Test DuckDB Ibis vs Pandas
    validation = Validate(data=tbl_duckdb).tbl_match(tbl_compare=test_data).interrogate()
    assert validation.all_passed()

    validation = Validate(data=test_data).tbl_match(tbl_compare=tbl_duckdb).interrogate()
    assert validation.all_passed()
