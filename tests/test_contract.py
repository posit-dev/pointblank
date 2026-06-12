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


class TestContractValidation:
    """Tests for Contract.validate() and Contract.to_validate()."""

    def test_validate_passing(self, simple_df, basic_contract):
        result = basic_contract.validate(simple_df)
        assert result.all_passed()

    def test_validate_failing_null(self, df_with_nulls):
        contract = Contract(
            name="null_check",
            steps=[Step("col_vals_not_null", columns=["id", "name", "amount"])],
        )
        result = contract.validate(df_with_nulls)
        assert not result.all_passed()

    def test_validate_schema_match(self, simple_df):
        contract = Contract(
            name="schema_check",
            schema=pb.Schema(id="Int64", name="String", amount="Float64", status="String"),
        )
        result = contract.validate(simple_df)
        assert result.all_passed()

    def test_validate_schema_mismatch(self, simple_df):
        contract = Contract(
            name="bad_schema",
            schema=pb.Schema(id="String", name="String"),  # id is actually Int64
        )
        result = contract.validate(simple_df)
        assert not result.all_passed()

    def test_validate_with_gt(self, simple_df):
        contract = Contract(
            name="amount_check",
            steps=[Step("col_vals_gt", columns="amount", value=50)],
        )
        result = contract.validate(simple_df)
        assert result.all_passed()

    def test_validate_with_gt_failing(self, simple_df):
        contract = Contract(
            name="amount_check",
            steps=[Step("col_vals_gt", columns="amount", value=200)],
        )
        result = contract.validate(simple_df)
        assert not result.all_passed()

    def test_validate_with_in_set(self, simple_df):
        contract = Contract(
            name="status_check",
            steps=[Step("col_vals_in_set", columns="status", set=["active", "inactive"])],
        )
        result = contract.validate(simple_df)
        assert result.all_passed()

    def test_validate_with_in_set_failing(self, simple_df):
        contract = Contract(
            name="status_check",
            steps=[Step("col_vals_in_set", columns="status", set=["active"])],
        )
        result = contract.validate(simple_df)
        assert not result.all_passed()

    def test_validate_rows_distinct(self, simple_df):
        contract = Contract(
            name="distinct_check",
            steps=[Step("rows_distinct", columns_subset=["id"])],
        )
        result = contract.validate(simple_df)
        assert result.all_passed()

    def test_validate_rows_distinct_failing(self):
        df = pl.DataFrame({"id": [1, 1, 2, 3, 3]})
        contract = Contract(
            name="distinct_check",
            steps=[Step("rows_distinct", columns_subset=["id"])],
        )
        result = contract.validate(df)
        assert not result.all_passed()

    def test_validate_multiple_steps(self, simple_df):
        contract = Contract(
            name="multi_check",
            steps=[
                Step("col_vals_not_null", columns=["id", "name"]),
                Step("col_vals_gt", columns="amount", value=0),
                Step("col_vals_in_set", columns="status", set=["active", "inactive"]),
                Step("rows_distinct", columns_subset=["id"]),
            ],
        )
        result = contract.validate(simple_df)
        assert result.all_passed()

    def test_to_validate_not_interrogated(self, simple_df, basic_contract):
        """to_validate() should NOT interrogate."""
        validation = basic_contract.to_validate(simple_df)
        # Before interrogation, the validation should exist but not have results
        assert validation is not None

    def test_validate_with_regex(self):
        df = pl.DataFrame({"email": ["a@b.com", "c@d.org", "e@f.net"]})
        contract = Contract(
            name="email_check",
            steps=[Step("col_vals_regex", columns="email", pattern=r"^[^@]+@[^@]+\.[^@]+$")],
        )
        result = contract.validate(df)
        assert result.all_passed()

    def test_validate_with_regex_failing(self):
        df = pl.DataFrame({"email": ["a@b.com", "not_an_email", "e@f.net"]})
        contract = Contract(
            name="email_check",
            steps=[Step("col_vals_regex", columns="email", pattern=r"^[^@]+@[^@]+\.[^@]+$")],
        )
        result = contract.validate(df)
        assert not result.all_passed()

    def test_validate_col_exists(self, simple_df):
        contract = Contract(
            name="exists_check",
            steps=[Step("col_exists", columns="id")],
        )
        result = contract.validate(simple_df)
        assert result.all_passed()

    def test_validate_col_exists_failing(self, simple_df):
        contract = Contract(
            name="exists_check",
            steps=[Step("col_exists", columns="nonexistent_column")],
        )
        result = contract.validate(simple_df)
        assert not result.all_passed()

    def test_validate_between(self, simple_df):
        contract = Contract(
            name="between_check",
            steps=[Step("col_vals_between", columns="amount", left=0, right=500)],
        )
        result = contract.validate(simple_df)
        assert result.all_passed()

    def test_validate_between_failing(self, simple_df):
        contract = Contract(
            name="between_check",
            steps=[Step("col_vals_between", columns="amount", left=0, right=100)],
        )
        result = contract.validate(simple_df)
        assert not result.all_passed()


class TestContractOnViolation:
    """Tests for on_violation behavior."""

    def test_on_violation_warn_default(self, basic_contract):
        assert basic_contract.on_violation == "warn"

    def test_on_violation_raise(self, df_with_nulls):
        contract = Contract(
            name="strict",
            steps=[Step("col_vals_not_null", columns=["id"])],
            on_violation="raise",
        )
        # Using Contract.validate() directly doesn't trigger on_violation
        # (it's Pipeline that handles on_violation)
        result = contract.validate(df_with_nulls)
        assert not result.all_passed()

    def test_on_violation_log(self):
        contract = Contract(name="log_mode", on_violation="log")
        assert contract.on_violation == "log"


class TestContractSerialization:
    """Tests for Contract serialization (to_dict, from_dict, YAML)."""

    def test_to_dict_minimal(self):
        contract = Contract(name="minimal")
        d = contract.to_dict()
        assert d == {"name": "minimal"}

    def test_to_dict_full(self, basic_contract):
        d = basic_contract.to_dict()
        assert d["name"] == "test_contract"
        assert d["version"] == "1.0.0"
        assert d["owner"] == "test-team"
        assert "schema" in d
        assert "steps" in d
        assert len(d["steps"]) == 2

    def test_to_dict_target_direction(self):
        contract = Contract(name="out", direction="target")
        d = contract.to_dict()
        assert d["direction"] == "target"

    def test_to_dict_source_direction_omitted(self):
        contract = Contract(name="in", direction="source")
        d = contract.to_dict()
        assert "direction" not in d  # source is default, omitted

    def test_from_dict_minimal(self):
        d = {"name": "minimal"}
        contract = Contract.from_dict(d)
        assert contract.name == "minimal"
        assert contract.direction == "source"
        assert contract.steps == []

    def test_from_dict_full(self):
        d = {
            "name": "full_contract",
            "direction": "target",
            "version": "2.0.0",
            "owner": "data-team",
            "consumers": ["ml", "bi"],
            "description": "Test contract",
            "on_violation": "raise",
            "schema": {"id": "Int64", "name": "String"},
            "steps": [
                {"col_vals_not_null": {"columns": ["id"]}},
                {"col_vals_gt": {"columns": "id", "value": 0}},
            ],
            "thresholds": {"warning": 0.01, "error": 0.05},
        }
        contract = Contract.from_dict(d)
        assert contract.name == "full_contract"
        assert contract.direction == "target"
        assert contract.version == "2.0.0"
        assert contract.owner == "data-team"
        assert contract.consumers == ["ml", "bi"]
        assert contract.description == "Test contract"
        assert contract.on_violation == "raise"
        assert contract.schema is not None
        assert len(contract.steps) == 2
        assert contract.thresholds.warning == 0.01
        assert contract.thresholds.error == 0.05

    def test_from_dict_no_name_raises(self):
        with pytest.raises(ValueError, match="must include a 'name' key"):
            Contract.from_dict({})

    def test_from_dict_steps_not_list_raises(self):
        with pytest.raises(TypeError, match="must be a list"):
            Contract.from_dict({"name": "bad", "steps": "not_a_list"})

    def test_round_trip_dict(self, basic_contract):
        """to_dict -> from_dict should be lossless."""
        d = basic_contract.to_dict()
        loaded = Contract.from_dict(d)
        assert loaded.name == basic_contract.name
        assert loaded.version == basic_contract.version
        assert loaded.owner == basic_contract.owner
        assert loaded.steps == basic_contract.steps

    def test_to_yaml_string(self, basic_contract):
        yaml_str = basic_contract.to_yaml()
        assert "contract:" in yaml_str
        assert "name: test_contract" in yaml_str
        assert "col_vals_not_null:" in yaml_str

    def test_to_yaml_file(self, basic_contract, tmp_path):
        path = str(tmp_path / "contract.yaml")
        basic_contract.to_yaml(path)
        assert Path(path).exists()

        # Verify file content
        with open(path) as f:
            content = yaml.safe_load(f)
        assert content["contract"]["name"] == "test_contract"

    def test_from_yaml(self, basic_contract, tmp_path):
        path = str(tmp_path / "contract.yaml")
        basic_contract.to_yaml(path)

        loaded = Contract.from_yaml(path)
        assert loaded.name == basic_contract.name
        assert loaded.steps == basic_contract.steps

    def test_from_yaml_not_found(self):
        with pytest.raises(FileNotFoundError):
            Contract.from_yaml("/nonexistent/path.yaml")

    def test_from_yaml_empty_file(self, tmp_path):
        path = str(tmp_path / "empty.yaml")
        Path(path).write_text("")
        with pytest.raises(ValueError, match="empty"):
            Contract.from_yaml(path)

    def test_from_yaml_with_contract_key(self, tmp_path):
        """YAML files with a top-level 'contract' key should work."""
        content = {
            "contract": {
                "name": "yaml_test",
                "steps": [{"col_vals_not_null": {"columns": ["id"]}}],
            }
        }
        path = str(tmp_path / "contract.yaml")
        with open(path, "w") as f:
            yaml.dump(content, f)

        loaded = Contract.from_yaml(path)
        assert loaded.name == "yaml_test"
        assert len(loaded.steps) == 1


class TestContractRepr:
    """Tests for Contract __repr__."""

    def test_repr_minimal(self):
        contract = Contract(name="test")
        r = repr(contract)
        assert "Contract(" in r
        assert "name='test'" in r

    def test_repr_with_version(self):
        contract = Contract(name="test", version="1.0.0")
        r = repr(contract)
        assert "version='1.0.0'" in r

    def test_repr_with_schema(self, basic_schema):
        contract = Contract(name="test", schema=basic_schema)
        r = repr(contract)
        assert "schema=<defined>" in r


# ─── Helper Function Tests ───────────────────────────────────────────────────────


class TestHelperFunctions:
    """Tests for _schema_to_dict, _dict_to_schema, etc."""

    def test_schema_to_dict(self):
        schema = pb.Schema(id="Int64", name="String")
        d = _schema_to_dict(schema)
        assert d == {"id": "Int64", "name": "String"}

    def test_dict_to_schema(self):
        d = {"id": "Int64", "name": "String", "amount": "Float64"}
        schema = _dict_to_schema(d)
        assert schema is not None
        # Verify the schema has the right columns
        col_names = [col_name for col_name, _ in schema.columns]
        assert "id" in col_names
        assert "name" in col_names
        assert "amount" in col_names

    def test_thresholds_to_dict(self):
        thresholds = pb.Thresholds(warning=0.01, error=0.05, critical=0.10)
        d = _thresholds_to_dict(thresholds)
        assert d == {"warning": 0.01, "error": 0.05, "critical": 0.10}

    def test_thresholds_to_dict_partial(self):
        thresholds = pb.Thresholds(warning=0.01)
        d = _thresholds_to_dict(thresholds)
        assert d == {"warning": 0.01}

    def test_dict_to_thresholds(self):
        d = {"warning": 0.01, "error": 0.05}
        thresholds = _dict_to_thresholds(d)
        assert thresholds.warning == 0.01
        assert thresholds.error == 0.05
        assert thresholds.critical is None


# ─── Edge Cases ──────────────────────────────────────────────────────────────────


class TestContractEdgeCases:
    """Edge cases and boundary conditions."""

    def test_contract_with_many_steps(self, simple_df):
        """Contract with many different step types."""
        contract = Contract(
            name="many_steps",
            steps=[
                Step("col_exists", columns="id"),
                Step("col_exists", columns="name"),
                Step("col_vals_not_null", columns=["id"]),
                Step("col_vals_gt", columns="id", value=0),
                Step("col_vals_lt", columns="amount", value=1000),
                Step("col_vals_between", columns="amount", left=0, right=500),
                Step("col_vals_in_set", columns="status", set=["active", "inactive"]),
                Step("rows_distinct", columns_subset=["id"]),
            ],
        )
        result = contract.validate(simple_df)
        assert result.all_passed()

    def test_contract_schema_only(self, simple_df):
        """Contract with only schema, no steps."""
        contract = Contract(
            name="schema_only",
            schema=pb.Schema(id="Int64", name="String", amount="Float64", status="String"),
        )
        result = contract.validate(simple_df)
        assert result.all_passed()

    def test_contract_steps_only(self, simple_df):
        """Contract with only steps, no schema."""
        contract = Contract(
            name="steps_only",
            steps=[Step("col_vals_not_null", columns=["id"])],
        )
        result = contract.validate(simple_df)
        assert result.all_passed()

    def test_contract_empty_no_schema_no_steps(self, simple_df):
        """Contract with no schema and no steps."""
        contract = Contract(name="empty")
        # Should still be able to create a validate object
        validation = contract.to_validate(simple_df)
        assert validation is not None

    def test_contract_consumers_single_string(self):
        contract = Contract(name="test", consumers="analytics-team")
        assert contract.consumers == "analytics-team"

    def test_contract_consumers_list(self):
        contract = Contract(name="test", consumers=["ml-team", "bi-team"])
        assert contract.consumers == ["ml-team", "bi-team"]

    def test_step_with_na_pass(self, simple_df):
        """Step with na_pass parameter."""
        contract = Contract(
            name="na_pass_test",
            steps=[Step("col_vals_gt", columns="amount", value=0, na_pass=True)],
        )
        result = contract.validate(simple_df)
        assert result.all_passed()

    def test_yaml_round_trip_complex(self, tmp_path):
        """Full YAML round-trip with complex contract."""
        contract = Contract(
            name="complex_contract",
            direction="target",
            schema=pb.Schema(
                order_id="String",
                amount="Float64",
                currency="String",
                status="String",
            ),
            steps=[
                Step("col_vals_not_null", columns=["order_id", "amount"]),
                Step("col_vals_gt", columns="amount", value=0),
                Step("col_vals_in_set", columns="currency", set=["USD", "EUR", "GBP"]),
                Step("col_vals_in_set", columns="status", set=["pending", "shipped", "delivered"]),
                Step("rows_distinct", columns=["order_id"]),
            ],
            version="2.1.0",
            owner="data-platform",
            consumers=["analytics", "ml"],
            description="Clean order data contract",
            thresholds=pb.Thresholds(warning=0.01, error=0.05, critical=0.10),
            on_violation="raise",
        )

        path = str(tmp_path / "complex.yaml")
        contract.to_yaml(path)
        loaded = Contract.from_yaml(path)

        assert loaded.name == contract.name
        assert loaded.direction == contract.direction
        assert loaded.version == contract.version
        assert loaded.owner == contract.owner
        assert loaded.consumers == contract.consumers
        assert loaded.on_violation == contract.on_violation
        assert loaded.steps == contract.steps
        assert loaded.thresholds.warning == 0.01
        assert loaded.thresholds.error == 0.05
        assert loaded.thresholds.critical == 0.10

    def test_contract_with_thresholds_validates(self, simple_df):
        """Contract with thresholds passes them to Validate."""
        contract = Contract(
            name="thresh_test",
            steps=[Step("col_vals_gt", columns="amount", value=50)],
            thresholds=pb.Thresholds(warning=0.5, error=0.8),
        )
        result = contract.validate(simple_df)
        assert result.all_passed()
