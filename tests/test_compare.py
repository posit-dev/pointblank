from __future__ import annotations
import pytest

from pointblank.compare import Compare
from pointblank.validate import get_data_path
import polars.testing.parametric as pt
from hypothesis import given


@pytest.mark.xfail
def test_compare_basic(dfa, dfb) -> None:
    comp = Compare(dfa, dfb)

    comp.compare()

    raise NotImplementedError


def test_compare_csv_input():
    # Test with CSV files
    csv_path1 = "data_raw/small_table.csv"
    csv_path2 = "data_raw/small_table.csv"  # Compare with itself for testing
    comp = Compare(csv_path1, csv_path2)
    comp.compare()  # Need to call compare() to create the scan objects
    assert comp._scana is not None
    assert comp._scanb is not None


def test_compare_parquet_input():
    # Test with Parquet files
    parquet_path1 = "tests/tbl_files/tbl_xyz.parquet"
    parquet_path2 = "tests/tbl_files/tbl_xyz.parquet"  # Compare with itself for testing
    comp = Compare(parquet_path1, parquet_path2)
    comp.compare()  # Need to call compare() to create the scan objects
    assert comp._scana is not None
    assert comp._scanb is not None


def test_compare_connection_string_input():
    # Test with connection strings using get_data_path and absolute path
    duckdb_path = get_data_path("small_table", "duckdb")
    conn1 = f"duckdb:///{duckdb_path}::small_table"

    import os

    sqlite_path = os.path.abspath("tests/tbl_files/tbl_xyz.sqlite")
    conn2 = f"sqlite:///{sqlite_path}::tbl_xyz"

    comp = Compare(conn1, conn2)
    comp.compare()  # Need to call compare() to create the scan objects
    assert comp._scana is not None
    assert comp._scanb is not None


def test_compare_mixed_input_types():
    # Test CSV vs Parquet
    csv_path = "data_raw/small_table.csv"
    parquet_path = "tests/tbl_files/tbl_xyz.parquet"
    comp = Compare(csv_path, parquet_path)
    comp.compare()  # Need to call compare() to create the scan objects
    assert comp._scana is not None
    assert comp._scanb is not None

    # Test connection string vs CSV using get_data_path
    duckdb_path = get_data_path("small_table", "duckdb")
    conn = f"duckdb:///{duckdb_path}::small_table"
    comp2 = Compare(conn, csv_path)
    comp2.compare()  # Need to call compare() to create the scan objects
    assert comp2._scana is not None
    assert comp2._scanb is not None
    comp2.compare()  # Need to call compare() to create the scan objects
    assert comp2._scana is not None
    assert comp2._scanb is not None
