from __future__ import annotations

import pytest
import polars as pl
import yaml
import tempfile
from pathlib import Path

import pointblank as pb
from pointblank.contract import (
    Contract,
    Step,
    _VALID_VALIDATION_METHODS,
    _dict_to_schema,
    _dict_to_thresholds,
    _schema_to_dict,
    _thresholds_to_dict,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_df():
    """A simple DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
            "amount": [100.0, 200.0, 150.0, 300.0, 250.0],
            "status": ["active", "active", "inactive", "active", "inactive"],
        }
    )


@pytest.fixture
def df_with_nulls():
    """A DataFrame with null values."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, None, 5],
            "name": ["Alice", None, "Charlie", "Dave", None],
            "amount": [100.0, 200.0, None, 300.0, 250.0],
        }
    )


@pytest.fixture
def basic_schema():
    """A simple schema."""
    return pb.Schema(id="Int64", name="String", amount="Float64", status="String")


@pytest.fixture
def basic_contract(basic_schema):
    """A basic contract with schema and steps."""
    return Contract(
        name="test_contract",
        direction="source",
        schema=basic_schema,
        steps=[
            Step("col_vals_not_null", columns=["id", "name"]),
            Step("col_vals_gt", columns="amount", value=0),
        ],
        version="1.0.0",
        owner="test-team",
    )


