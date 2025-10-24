import pytest
import ibis

from pointblank._interrogation import interrogate_within_spec_db


@pytest.fixture
def vin_test_data_duckdb():
    """Create DuckDB table with VIN test data."""
    con = ibis.connect("duckdb://")

    data = {
        "id": [1, 2, 3, 4, 5],
        "vin": [
            "1HGBH41JXMN109186",  # Valid VIN
            "1M8GDM9AXKP042788",  # Valid VIN
            "1HGBH41JXM0109186",  # Invalid (contains 'O')
            "1HGBH41JXMN10918",  # Invalid (too short)
            None,  # NULL
        ],
    }

    return con.create_table("vin_data", data, overwrite=True)


@pytest.fixture
def vin_test_data_sqlite():
    """Create SQLite table with VIN test data."""
    con = ibis.connect("sqlite://")

    data = {
        "id": [1, 2, 3, 4, 5],
        "vin": [
            "1HGBH41JXMN109186",  # Valid VIN
            "1M8GDM9AXKP042788",  # Valid VIN
            "1HGBH41JXM0109186",  # Invalid (contains 'O')
            "1HGBH41JXMN10918",  # Invalid (too short)
            None,  # NULL
        ],
    }

    return con.create_table("vin_data", data, overwrite=True)


def test_vin_validation_duckdb_basic(vin_test_data_duckdb):
    """Test basic VIN validation with DuckDB (database-native)."""
    result = interrogate_within_spec_db(
        tbl=vin_test_data_duckdb,
        column="vin",
        values={"spec": "vin"},
        na_pass=False,
    )

    # Execute to check results
    result_df = result.execute()

    # First two VINs should be valid
    assert result_df["pb_is_good_"][0] == True
    assert result_df["pb_is_good_"][1] == True

    # Third VIN invalid (contains 'O')
    assert result_df["pb_is_good_"][2] == False

    # Fourth VIN invalid (too short)
    assert result_df["pb_is_good_"][3] == False

    # Fifth is NULL, should fail with na_pass=False
    assert result_df["pb_is_good_"][4] == False


def test_vin_validation_duckdb_na_pass(vin_test_data_duckdb):
    """Test VIN validation with na_pass=True (DuckDB, database-native)."""
    result = interrogate_within_spec_db(
        tbl=vin_test_data_duckdb,
        column="vin",
        values={"spec": "vin"},
        na_pass=True,
    )

    # Execute to check results
    result_df = result.execute()

    # NULL should pass with na_pass=True
    assert result_df["pb_is_good_"][4] == True


def test_vin_validation_sqlite_basic(vin_test_data_sqlite):
    """Test basic VIN validation with SQLite (database-native)."""
    result = interrogate_within_spec_db(
        tbl=vin_test_data_sqlite,
        column="vin",
        values={"spec": "vin"},
        na_pass=False,
    )

    # Execute to check results
    result_df = result.execute()

    # First two VINs should be valid
    assert result_df["pb_is_good_"][0] == True
    assert result_df["pb_is_good_"][1] == True

    # Third VIN invalid (contains 'O')
    assert result_df["pb_is_good_"][2] == False

    # Fourth VIN invalid (too short)
    assert result_df["pb_is_good_"][3] == False

    # Fifth is NULL, should fail with na_pass=False
    assert result_df["pb_is_good_"][4] == False


def test_vin_validation_no_materialization(vin_test_data_duckdb):
    """
    Verify that database-native validation doesn't materialize data.

    This test confirms that the validation is performed as a lazy Ibis expression
    and only executed when explicitly called.
    """
    result = interrogate_within_spec_db(
        tbl=vin_test_data_duckdb,
        column="vin",
        values={"spec": "vin"},
        na_pass=False,
    )

    # Result should still be an Ibis table (not materialized)
    assert hasattr(result, "execute")

    # Should be able to chain more operations without executing
    filtered = result.filter(result["pb_is_good_"] == True)
    assert hasattr(filtered, "execute")

    # Only materialize when we explicitly execute
    materialized = filtered.execute()
    assert len(materialized) == 2  # Only 2 valid VINs


def test_unsupported_spec_raises_error(vin_test_data_duckdb):
    """Test that unsupported specs raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="Database-native validation for 'email'"):
        interrogate_within_spec_db(
            tbl=vin_test_data_duckdb,
            column="vin",
            values={"spec": "email"},
            na_pass=False,
        )


def test_fallback_to_regular_for_non_ibis():
    """Test that non-Ibis tables fall back to regular implementation."""
    import polars as pl
    from pointblank._interrogation import interrogate_within_spec

    # Create a Polars DataFrame
    df = pl.DataFrame(
        {
            "vin": [
                "1HGBH41JXMN109186",  # Valid
                "1HGBH41JXM0109186",  # Invalid (contains 'O')
            ]
        }
    )

    # Should fall back to regular implementation (which will work)
    result = interrogate_within_spec_db(
        tbl=df,
        column="vin",
        values={"spec": "vin"},
        na_pass=False,
    )

    # Result should be a Polars DataFrame (not Ibis)
    assert isinstance(result, pl.DataFrame)
    assert "pb_is_good_" in result.columns
