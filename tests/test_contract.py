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


# ─── Step Tests ──────────────────────────────────────────────────────────────────


class TestStep:
    """Tests for the Step class."""

    def test_basic_creation(self):
        step = Step("col_vals_gt", columns="amount", value=0)
        assert step.method == "col_vals_gt"
        assert step.kwargs == {"columns": "amount", "value": 0}

    def test_creation_no_kwargs(self):
        step = Step("rows_complete")
        assert step.method == "rows_complete"
        assert step.kwargs == {}

    def test_creation_with_list_columns(self):
        step = Step("col_vals_not_null", columns=["id", "name", "email"])
        assert step.kwargs["columns"] == ["id", "name", "email"]

    def test_creation_with_complex_kwargs(self):
        step = Step(
            "col_vals_between", columns="amount", left=0, right=1000, inclusive=(True, False)
        )
        assert step.kwargs == {
            "columns": "amount",
            "left": 0,
            "right": 1000,
            "inclusive": (True, False),
        }

    def test_repr_with_kwargs(self):
        step = Step("col_vals_gt", columns="amount", value=0)
        repr_str = repr(step)
        assert "Step(" in repr_str
        assert "col_vals_gt" in repr_str
        assert "columns='amount'" in repr_str
        assert "value=0" in repr_str

    def test_repr_no_kwargs(self):
        step = Step("rows_complete")
        assert repr(step) == "Step('rows_complete')"

    def test_equality(self):
        step1 = Step("col_vals_gt", columns="a", value=5)
        step2 = Step("col_vals_gt", columns="a", value=5)
        step3 = Step("col_vals_gt", columns="a", value=10)
        assert step1 == step2
        assert step1 != step3

    def test_equality_different_type(self):
        step = Step("col_vals_gt", columns="a", value=5)
        assert step != "not a step"
        assert step.__eq__("not a step") is NotImplemented

    def test_to_dict_with_kwargs(self):
        step = Step("col_vals_gt", columns="amount", value=0)
        d = step.to_dict()
        assert d == {"col_vals_gt": {"columns": "amount", "value": 0}}

    def test_to_dict_no_kwargs(self):
        step = Step("rows_complete")
        d = step.to_dict()
        assert d == {"rows_complete": {}}

    def test_from_dict(self):
        d = {"col_vals_gt": {"columns": "amount", "value": 0}}
        step = Step.from_dict(d)
        assert step.method == "col_vals_gt"
        assert step.kwargs == {"columns": "amount", "value": 0}

    def test_from_dict_empty_kwargs(self):
        d = {"rows_complete": {}}
        step = Step.from_dict(d)
        assert step.method == "rows_complete"
        assert step.kwargs == {}

    def test_from_dict_none_kwargs(self):
        d = {"rows_complete": None}
        step = Step.from_dict(d)
        assert step.method == "rows_complete"
        assert step.kwargs == {}

    def test_from_dict_multiple_keys_raises(self):
        d = {"col_vals_gt": {"columns": "a"}, "col_vals_lt": {"columns": "b"}}
        with pytest.raises(ValueError, match="exactly one key"):
            Step.from_dict(d)

    def test_from_dict_non_dict_kwargs_raises(self):
        d = {"col_vals_gt": "not_a_dict"}
        with pytest.raises(TypeError, match="Step kwargs must be a dictionary"):
            Step.from_dict(d)

    def test_round_trip(self):
        """Test to_dict -> from_dict round trip."""
        original = Step("col_vals_in_set", columns="status", set=["a", "b", "c"])
        reconstructed = Step.from_dict(original.to_dict())
        assert original == reconstructed

    def test_all_valid_methods(self):
        """Ensure all documented methods are in the valid set."""
        assert "col_vals_gt" in _VALID_VALIDATION_METHODS
        assert "col_vals_lt" in _VALID_VALIDATION_METHODS
        assert "col_vals_between" in _VALID_VALIDATION_METHODS
        assert "rows_distinct" in _VALID_VALIDATION_METHODS
        assert "col_vals_not_null" in _VALID_VALIDATION_METHODS
        assert "col_vals_regex" in _VALID_VALIDATION_METHODS
        assert "col_exists" in _VALID_VALIDATION_METHODS
        assert "rows_complete" in _VALID_VALIDATION_METHODS


