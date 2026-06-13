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


class TestJSONSchemaExport:
    def test_export_from_contract(self):
        contract = pb.Contract(
            name="test_export",
            schema=pb.Schema(age="Int64", name="String"),
            steps=[
                pb.Step("col_vals_ge", columns="age", value=0),
                pb.Step("col_vals_not_null", columns="name"),
            ],
        )
        result = export_contract(contract, format="json_schema")

        assert result["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert result["title"] == "test_export"
        assert "properties" in result
        assert result["properties"]["age"]["type"] == "integer"
        assert result["properties"]["age"]["minimum"] == 0
        assert "name" in result["required"]

    def test_export_to_file(self):
        contract = pb.Contract(
            name="test_file_export",
            schema=pb.Schema(id="Int64"),
            steps=[],
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            export_contract(contract, f.name, format="json_schema")
            f.flush()

        with open(f.name) as fh:
            data = json.load(fh)

        assert data["title"] == "test_file_export"
        assert "properties" in data

    def test_export_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected a Validate or Contract"):
            export_contract("not a contract", format="json_schema")


class TestFrictionlessImport:
    def test_import_from_dict(self, frictionless_schema_dict):
        result = import_contract(frictionless_schema_dict, format="frictionless")

        assert result.source_format == "frictionless"
        assert len(result.columns) == 5

        col_map = dict(result.columns)
        assert col_map["id"] == "Int64"
        assert col_map["name"] == "String"
        assert col_map["age"] == "Int64"
        assert col_map["status"] == "String"

    def test_import_constraints_required(self, frictionless_schema_dict):
        result = import_contract(frictionless_schema_dict, format="frictionless")
        methods = [(c.method, c.kwargs) for c in result.constraints]

        assert ("col_vals_not_null", {"columns": "id"}) in methods
        assert ("col_vals_not_null", {"columns": "name"}) in methods

    def test_import_constraints_unique(self, frictionless_schema_dict):
        result = import_contract(frictionless_schema_dict, format="frictionless")
        methods = [(c.method, c.kwargs) for c in result.constraints]

        assert ("rows_distinct", {"columns_subset": "id"}) in methods

    def test_import_constraints_min_max(self, frictionless_schema_dict):
        result = import_contract(frictionless_schema_dict, format="frictionless")
        methods = [(c.method, c.kwargs) for c in result.constraints]

        assert ("col_vals_ge", {"columns": "age", "value": 0}) in methods
        assert ("col_vals_le", {"columns": "age", "value": 150}) in methods

    def test_import_constraints_enum(self, frictionless_schema_dict):
        result = import_contract(frictionless_schema_dict, format="frictionless")
        methods = [(c.method, c.kwargs) for c in result.constraints]

        assert (
            "col_vals_in_set",
            {"columns": "status", "set": ["active", "inactive"]},
        ) in methods

    def test_import_constraints_pattern(self, frictionless_schema_dict):
        result = import_contract(frictionless_schema_dict, format="frictionless")
        methods = [(c.method, c.kwargs) for c in result.constraints]

        assert (
            "col_vals_regex",
            {"columns": "email", "pattern": r"^[^@]+@[^@]+\.[^@]+$"},
        ) in methods

    def test_import_primary_key(self, frictionless_schema_dict):
        """Primary key should generate not_null + distinct constraints."""
        result = import_contract(frictionless_schema_dict, format="frictionless")
        methods = [(c.method, c.kwargs) for c in result.constraints]

        # Primary key "id" should have not_null
        not_null_cols = [
            c.kwargs["columns"] for c in result.constraints if c.method == "col_vals_not_null"
        ]
        assert "id" in not_null_cols

    def test_import_from_datapackage(self, frictionless_datapackage_dict):
        """Import from a Data Package selects the first resource by default."""
        result = import_contract(frictionless_datapackage_dict, format="frictionless")
        assert len(result.columns) == 5  # users table

    def test_import_from_datapackage_by_name(self, frictionless_datapackage_dict):
        """Import a specific resource by name."""
        result = import_contract(
            frictionless_datapackage_dict, format="frictionless", resource="orders"
        )
        assert len(result.columns) == 2
        col_names = [name for name, _ in result.columns]
        assert "order_id" in col_names

    def test_import_from_datapackage_by_index(self, frictionless_datapackage_dict):
        result = import_contract(frictionless_datapackage_dict, format="frictionless", resource=1)
        assert len(result.columns) == 2

    def test_import_from_datapackage_invalid_name(self, frictionless_datapackage_dict):
        with pytest.raises(ValueError, match="not found"):
            import_contract(
                frictionless_datapackage_dict, format="frictionless", resource="nonexistent"
            )

    def test_import_from_file(self, frictionless_schema_dict):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(frictionless_schema_dict, f)
            f.flush()
            result = import_contract(f.name, format="frictionless")

        assert result.source_format == "frictionless"
        assert result.source_path == f.name

    def test_import_foreign_key_warning(self):
        """Foreign keys should produce a warning since cross-table is unsupported."""
        schema = {
            "fields": [
                {"name": "user_id", "type": "integer"},
            ],
            "foreignKeys": [
                {
                    "fields": ["user_id"],
                    "reference": {"resource": "users", "fields": ["id"]},
                }
            ],
        }
        result = import_contract(schema, format="frictionless")
        assert len(result.warnings) == 1
        assert "Foreign key" in result.warnings[0]
        assert result.coverage < 1.0

    def test_to_validate_end_to_end(self, frictionless_schema_dict, simple_df):
        result = import_contract(frictionless_schema_dict, format="frictionless")
        validation = result.to_validate(data=simple_df)
        validation.interrogate()

    def test_auto_detect_frictionless(self, frictionless_schema_dict):
        """Auto-detection works for Frictionless dicts."""
        result = import_contract(frictionless_schema_dict)
        assert result.source_format == "frictionless"


class TestFrictionlessExport:
    def test_export_from_contract(self):
        contract = pb.Contract(
            name="test_export",
            schema=pb.Schema(id="Int64", name="String", age="Int64"),
            steps=[
                pb.Step("col_vals_not_null", columns="id"),
                pb.Step("rows_distinct", columns="id"),
                pb.Step("col_vals_ge", columns="age", value=0),
            ],
        )
        result = export_contract(contract, format="frictionless")

        assert "fields" in result
        fields = result["fields"]
        assert len(fields) == 3

        # Check field types
        field_map = {f["name"]: f for f in fields}
        assert field_map["id"]["type"] == "integer"
        assert field_map["name"]["type"] == "string"
        assert field_map["id"]["constraints"]["required"] is True
        assert field_map["id"]["constraints"]["unique"] is True
        assert field_map["age"]["constraints"]["minimum"] == 0

    def test_export_to_file(self):
        contract = pb.Contract(
            name="test",
            schema=pb.Schema(x="Int64"),
            steps=[],
        )
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            export_contract(contract, f.name, format="frictionless")

        with open(f.name) as fh:
            data = json.load(fh)
        assert "fields" in data


class TestImportContractAPI:
    def test_format_required_or_detectable(self):
        """Should raise if format can't be detected."""
        with pytest.raises(ValueError, match="Could not auto-detect"):
            import_contract({"random": "data"})

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="No adapter registered"):
            import_contract("file.txt", format="made_up_format")


class TestExportContractAPI:
    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="No adapter registered"):
            export_contract(pb.Contract(name="x"), format="made_up_format")


class TestRoundTrip:
    def test_json_schema_roundtrip(self):
        """Import JSON Schema -> export -> re-import should produce equivalent constraints."""
        original_schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150},
                "status": {"type": "string", "enum": ["active", "inactive"]},
            },
            "required": ["age"],
        }

        # Import
        imported = import_contract(original_schema, format="json_schema")

        # Create a contract from it
        contract = imported.to_contract(name="roundtrip_test")

        # Export back to JSON Schema
        exported = export_contract(contract, format="json_schema")

        # Re-import
        reimported = import_contract(exported, format="json_schema")

        # Verify same constraints exist (use frozenset for list values)
        def _hashable_kwargs(kwargs):
            items = []
            for k, v in sorted(kwargs.items()):
                items.append((k, tuple(v) if isinstance(v, list) else v))
            return tuple(items)

        original_methods = {(c.method, _hashable_kwargs(c.kwargs)) for c in imported.constraints}
        roundtrip_methods = {(c.method, _hashable_kwargs(c.kwargs)) for c in reimported.constraints}
        assert original_methods == roundtrip_methods

    def test_frictionless_roundtrip(self):
        """Import Frictionless -> export -> re-import should produce equivalent constraints."""
        original_schema = {
            "fields": [
                {
                    "name": "age",
                    "type": "integer",
                    "constraints": {"required": True, "minimum": 0},
                },
                {
                    "name": "status",
                    "type": "string",
                    "constraints": {"enum": ["active", "inactive"]},
                },
            ],
        }

        imported = import_contract(original_schema, format="frictionless")
        contract = imported.to_contract(name="roundtrip_test")
        exported = export_contract(contract, format="frictionless")
        reimported = import_contract(exported, format="frictionless")

        def _hashable_kwargs(kwargs):
            items = []
            for k, v in sorted(kwargs.items()):
                items.append((k, tuple(v) if isinstance(v, list) else v))
            return tuple(items)

        original_methods = {(c.method, _hashable_kwargs(c.kwargs)) for c in imported.constraints}
        roundtrip_methods = {(c.method, _hashable_kwargs(c.kwargs)) for c in reimported.constraints}
        assert original_methods == roundtrip_methods
