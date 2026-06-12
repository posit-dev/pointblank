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


# ─── Contract Tests ──────────────────────────────────────────────────────────────


class TestContractCreation:
    """Tests for Contract instantiation and validation."""

    def test_basic_creation(self, basic_schema):
        contract = Contract(
            name="test",
            direction="source",
            schema=basic_schema,
            steps=[Step("col_vals_not_null", columns=["id"])],
        )
        assert contract.name == "test"
        assert contract.direction == "source"
        assert contract.schema is not None
        assert len(contract.steps) == 1

    def test_minimal_creation(self):
        """Contract with just a name is valid."""
        contract = Contract(name="minimal")
        assert contract.name == "minimal"
        assert contract.direction == "source"
        assert contract.schema is None
        assert contract.steps == []

    def test_target_direction(self):
        contract = Contract(name="output", direction="target")
        assert contract.direction == "target"

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="must be 'source' or 'target'"):
            Contract(name="bad", direction="invalid")

    def test_invalid_on_violation(self):
        with pytest.raises(ValueError, match="must be 'warn', 'raise', or 'log'"):
            Contract(name="bad", on_violation="explode")

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Contract(name="")

    def test_non_string_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Contract(name=123)  # type: ignore

    def test_steps_not_list_raises(self):
        with pytest.raises(TypeError, match="must be a list"):
            Contract(name="test", steps="not_a_list")  # type: ignore

    def test_steps_non_step_item_raises(self):
        with pytest.raises(TypeError, match="must be Step objects"):
            Contract(name="test", steps=[Step("col_vals_gt", columns="a", value=1), "bad"])

    def test_invalid_method_in_step_raises(self):
        with pytest.raises(ValueError, match="Unknown validation method"):
            Contract(name="test", steps=[Step("nonexistent_method")])

    def test_all_metadata_fields(self):
        contract = Contract(
            name="full",
            direction="target",
            version="2.1.0",
            owner="analytics-team",
            consumers=["ml-team", "bi-team"],
            description="A fully specified contract",
            on_violation="raise",
        )
        assert contract.version == "2.1.0"
        assert contract.owner == "analytics-team"
        assert contract.consumers == ["ml-team", "bi-team"]
        assert contract.description == "A fully specified contract"
        assert contract.on_violation == "raise"

    def test_thresholds(self):
        contract = Contract(
            name="with_thresh",
            thresholds=pb.Thresholds(warning=0.01, error=0.05, critical=0.10),
        )
        assert contract.thresholds.warning == 0.01
        assert contract.thresholds.error == 0.05
        assert contract.thresholds.critical == 0.10


