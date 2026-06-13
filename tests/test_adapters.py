from __future__ import annotations

import json
import tempfile

import polars as pl
import pytest

import pointblank as pb
from pointblank.adapters import (
    ContractAdapter,
    ContractImport,
    MappedConstraint,
    export_contract,
    get_adapter,
    import_contract,
    list_adapters,
    register_adapter,
)
from pointblank.adapters._registry import _ADAPTER_REGISTRY


@pytest.fixture
def simple_df():
    """Simple DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Charlie", "Dave", "Eve"],
            "age": [25, 30, 35, 40, 45],
            "email": [
                "alice@example.com",
                "bob@example.com",
                "charlie@example.com",
                "dave@example.com",
                "eve@example.com",
            ],
            "status": ["active", "active", "inactive", "active", "inactive"],
        }
    )


@pytest.fixture
def json_schema_dict():
    """A JSON Schema document as a dict."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "User Profile",
        "description": "Schema for user profile data",
        "type": "object",
        "properties": {
            "age": {"type": "integer", "minimum": 0, "maximum": 150},
            "email": {"type": "string", "format": "email"},
            "status": {"type": "string", "enum": ["active", "inactive", "pending"]},
            "name": {"type": "string", "pattern": "^[A-Za-z ]+$"},
            "score": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 100},
        },
        "required": ["age", "email"],
    }


@pytest.fixture
def frictionless_schema_dict():
    """A Frictionless Table Schema as a dict."""
    return {
        "fields": [
            {
                "name": "id",
                "type": "integer",
                "constraints": {"required": True, "unique": True},
            },
            {
                "name": "name",
                "type": "string",
                "constraints": {"required": True},
            },
            {
                "name": "age",
                "type": "integer",
                "constraints": {"minimum": 0, "maximum": 150},
            },
            {
                "name": "status",
                "type": "string",
                "constraints": {"enum": ["active", "inactive"]},
            },
            {
                "name": "email",
                "type": "string",
                "constraints": {"pattern": r"^[^@]+@[^@]+\.[^@]+$"},
            },
        ],
        "primaryKey": "id",
    }


@pytest.fixture
def frictionless_datapackage_dict(frictionless_schema_dict):
    """A Frictionless Data Package with one resource."""
    return {
        "name": "my-package",
        "resources": [
            {
                "name": "users",
                "path": "users.csv",
                "schema": frictionless_schema_dict,
            },
            {
                "name": "orders",
                "path": "orders.csv",
                "schema": {
                    "fields": [
                        {"name": "order_id", "type": "integer"},
                        {"name": "user_id", "type": "integer"},
                    ]
                },
            },
        ],
    }


class TestRegistry:
    def test_builtin_adapters_registered(self):
        """Built-in adapters should be registered on import."""
        adapters = list_adapters()
        assert "json_schema" in adapters
        assert "frictionless" in adapters

    def test_get_adapter_json_schema(self):
        adapter = get_adapter("json_schema")
        assert adapter.format_name == "json_schema"
        assert adapter.supports_import is True
        assert adapter.supports_export is True

    def test_get_adapter_frictionless(self):
        adapter = get_adapter("frictionless")
        assert adapter.format_name == "frictionless"

    def test_get_adapter_unknown_raises(self):
        with pytest.raises(ValueError, match="No adapter registered"):
            get_adapter("nonexistent_format")

    def test_list_adapters_info(self):
        adapters = list_adapters()
        for name, info in adapters.items():
            assert "class" in info
            assert "file_extensions" in info
            assert "supports_import" in info
            assert "supports_export" in info

    def test_register_custom_adapter(self):
        """Custom adapters can be registered via decorator."""

        @register_adapter("test_custom")
        class TestCustomAdapter(ContractAdapter):
            format_name = "test_custom"
            file_extensions = [".custom"]

            @staticmethod
            def detect(source):
                return False

            def import_contract(self, source, **kwargs):
                return ContractImport(source_format="test_custom")

        assert "test_custom" in list_adapters()
        adapter = get_adapter("test_custom")
        assert adapter.format_name == "test_custom"

        # Cleanup
        del _ADAPTER_REGISTRY["test_custom"]


class TestContractImport:
    def test_to_validate(self, simple_df):
        result = ContractImport(
            source_format="test",
            columns=[("age", "Int64"), ("name", "String")],
            constraints=[
                MappedConstraint(
                    method="col_vals_ge",
                    kwargs={"columns": "age", "value": 0},
                ),
                MappedConstraint(
                    method="col_vals_not_null",
                    kwargs={"columns": "name"},
                ),
            ],
        )
        validation = result.to_validate(data=simple_df)
        # Should have schema check + 2 constraint steps
        assert len(validation.validation_info) == 3

    def test_to_contract(self):
        result = ContractImport(
            source_format="test",
            columns=[("age", "Int64")],
            constraints=[
                MappedConstraint(
                    method="col_vals_ge",
                    kwargs={"columns": "age", "value": 0},
                ),
            ],
            metadata={"description": "Test contract"},
        )
        contract = result.to_contract(name="my_contract")
        assert contract.name == "my_contract"
        assert contract.description == "Test contract"
        assert contract.schema is not None
        assert len(contract.steps) == 1
        assert contract.steps[0].method == "col_vals_ge"

    def test_to_python(self):
        result = ContractImport(
            source_format="test",
            columns=[("age", "Int64")],
            constraints=[
                MappedConstraint(method="col_vals_ge", kwargs={"columns": "age", "value": 0}),
            ],
        )
        code = result.to_python()
        assert "import pointblank as pb" in code
        assert "pb.Validate(data=data)" in code
        assert ".col_vals_ge(" in code
        assert "pb.Schema(" in code

    def test_to_yaml(self):
        result = ContractImport(
            source_format="test",
            columns=[("age", "Int64")],
            constraints=[
                MappedConstraint(method="col_vals_ge", kwargs={"columns": "age", "value": 0}),
            ],
        )
        yaml_str = result.to_yaml()
        assert "col_schema_match" in yaml_str
        assert "col_vals_ge" in yaml_str

    def test_summary(self):
        result = ContractImport(
            source_format="json_schema",
            source_path="/path/to/file.json",
            columns=[("a", "Int64"), ("b", "String")],
            constraints=[MappedConstraint(method="col_vals_ge", kwargs={})],
            warnings=["Some warning"],
            coverage=0.8,
        )
        summary = result.summary()
        assert "json_schema" in summary
        assert "Columns detected: 2" in summary
        assert "Constraints mapped: 1" in summary
        assert "80%" in summary
        assert "Some warning" in summary

    def test_repr(self):
        result = ContractImport(
            source_format="json_schema",
            columns=[("a", "Int64")],
            constraints=[MappedConstraint(method="col_vals_ge", kwargs={})],
            coverage=1.0,
        )
        assert "json_schema" in repr(result)
        assert "columns=1" in repr(result)


class TestJSONSchemaImport:
    def test_import_from_dict(self, json_schema_dict):
        result = import_contract(json_schema_dict, format="json_schema")

        assert result.source_format == "json_schema"
        assert len(result.columns) == 5
        assert result.metadata.get("title") == "User Profile"
        assert result.metadata.get("description") == "Schema for user profile data"

        # Check column dtypes
        col_map = dict(result.columns)
        assert col_map["age"] == "Int64"
        assert col_map["email"] == "String"
        assert col_map["status"] == "String"
        assert col_map["score"] == "Float64"

    def test_import_constraints_minimum_maximum(self, json_schema_dict):
        result = import_contract(json_schema_dict, format="json_schema")

        methods = [(c.method, c.kwargs) for c in result.constraints]

        # age has minimum=0, maximum=150
        assert ("col_vals_ge", {"columns": "age", "value": 0}) in methods
        assert ("col_vals_le", {"columns": "age", "value": 150}) in methods

    def test_import_constraints_exclusive(self, json_schema_dict):
        result = import_contract(json_schema_dict, format="json_schema")
        methods = [(c.method, c.kwargs) for c in result.constraints]

        # score has exclusiveMinimum=0, exclusiveMaximum=100
        assert ("col_vals_gt", {"columns": "score", "value": 0}) in methods
        assert ("col_vals_lt", {"columns": "score", "value": 100}) in methods

    def test_import_constraints_enum(self, json_schema_dict):
        result = import_contract(json_schema_dict, format="json_schema")
        methods = [(c.method, c.kwargs) for c in result.constraints]

        assert (
            "col_vals_in_set",
            {"columns": "status", "set": ["active", "inactive", "pending"]},
        ) in methods

    def test_import_constraints_required(self, json_schema_dict):
        result = import_contract(json_schema_dict, format="json_schema")
        methods = [(c.method, c.kwargs) for c in result.constraints]

        assert ("col_vals_not_null", {"columns": "age"}) in methods
        assert ("col_vals_not_null", {"columns": "email"}) in methods

    def test_import_constraints_pattern(self, json_schema_dict):
        result = import_contract(json_schema_dict, format="json_schema")
        methods = [(c.method, c.kwargs) for c in result.constraints]

        assert ("col_vals_regex", {"columns": "name", "pattern": "^[A-Za-z ]+$"}) in methods

    def test_import_constraints_format_email(self, json_schema_dict):
        result = import_contract(json_schema_dict, format="json_schema")
        methods = [(c.method, c.kwargs) for c in result.constraints]

        assert ("col_vals_within_spec", {"columns": "email", "spec": "email"}) in methods

    def test_import_from_file(self, json_schema_dict):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".schema.json", delete=False) as f:
            json.dump(json_schema_dict, f)
            f.flush()
            result = import_contract(f.name, format="json_schema")

        assert result.source_format == "json_schema"
        assert result.source_path == f.name
        assert len(result.columns) == 5

    def test_import_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            import_contract("/nonexistent/file.schema.json", format="json_schema")

    def test_import_invalid_type(self):
        with pytest.raises(TypeError, match="must be a file path"):
            import_contract(12345, format="json_schema")

    def test_to_validate_end_to_end(self, json_schema_dict, simple_df):
        result = import_contract(json_schema_dict, format="json_schema")
        validation = result.to_validate(data=simple_df)
        validation.interrogate()
        # Should complete without error

    def test_auto_detect_json_schema(self, json_schema_dict):
        """Auto-detection works for JSON Schema dicts."""
        result = import_contract(json_schema_dict)  # no format specified
        assert result.source_format == "json_schema"

    def test_const_constraint(self):
        schema = {
            "type": "object",
            "properties": {
                "version": {"type": "string", "const": "v2"},
            },
        }
        result = import_contract(schema, format="json_schema")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert ("col_vals_eq", {"columns": "version", "value": "v2"}) in methods

    def test_nullable_union_type(self):
        schema = {
            "type": "object",
            "properties": {
                "nickname": {"type": ["string", "null"]},
            },
        }
        result = import_contract(schema, format="json_schema")
        col_map = dict(result.columns)
        assert col_map["nickname"] == "String"


