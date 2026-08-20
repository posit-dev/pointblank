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
        assert "dbt" in adapters
        assert "odcs" in adapters

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


# ── dbt adapter fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def dbt_schema_dict():
    """A dbt schema.yml document as a dict."""
    return {
        "version": 2,
        "models": [
            {
                "name": "users",
                "description": "User accounts table",
                "columns": [
                    {
                        "name": "id",
                        "data_type": "integer",
                        "data_tests": ["not_null", "unique"],
                    },
                    {
                        "name": "name",
                        "data_type": "varchar",
                        "data_tests": ["not_null"],
                    },
                    {
                        "name": "age",
                        "data_type": "integer",
                    },
                    {
                        "name": "status",
                        "data_type": "string",
                        "data_tests": [
                            {"accepted_values": {"values": ["active", "inactive", "pending"]}}
                        ],
                    },
                    {
                        "name": "email",
                        "data_type": "varchar(256)",
                    },
                ],
            }
        ],
    }


@pytest.fixture
def dbt_schema_legacy_tests():
    """A dbt schema.yml using the legacy 'tests' key."""
    return {
        "version": 2,
        "models": [
            {
                "name": "orders",
                "columns": [
                    {
                        "name": "order_id",
                        "data_type": "integer",
                        "tests": ["not_null", "unique"],
                    },
                    {
                        "name": "user_id",
                        "data_type": "integer",
                        "tests": [
                            {"relationships": {"to": "ref('users')", "field": "id"}},
                        ],
                    },
                ],
            }
        ],
    }


@pytest.fixture
def dbt_sources_dict():
    """A dbt schema.yml with sources instead of models."""
    return {
        "version": 2,
        "sources": [
            {
                "name": "raw",
                "tables": [
                    {
                        "name": "events",
                        "columns": [
                            {
                                "name": "event_id",
                                "data_type": "bigint",
                                "data_tests": ["not_null", "unique"],
                            },
                            {
                                "name": "event_type",
                                "data_type": "string",
                                "data_tests": [
                                    {"accepted_values": {"values": ["click", "view", "purchase"]}}
                                ],
                            },
                        ],
                    }
                ],
            }
        ],
    }


class TestDbtImport:
    def test_import_from_dict(self, dbt_schema_dict):
        result = import_contract(dbt_schema_dict, format="dbt")

        assert result.source_format == "dbt"
        assert len(result.columns) == 5
        assert result.metadata.get("title") == "users"
        assert result.metadata.get("description") == "User accounts table"

    def test_import_column_types(self, dbt_schema_dict):
        result = import_contract(dbt_schema_dict, format="dbt")
        col_map = dict(result.columns)
        assert col_map["id"] == "Int64"
        assert col_map["name"] == "String"
        assert col_map["age"] == "Int64"
        assert col_map["status"] == "String"
        assert col_map["email"] == "String"  # varchar(256) -> String

    def test_import_not_null(self, dbt_schema_dict):
        result = import_contract(dbt_schema_dict, format="dbt")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert ("col_vals_not_null", {"columns": "id"}) in methods
        assert ("col_vals_not_null", {"columns": "name"}) in methods

    def test_import_unique(self, dbt_schema_dict):
        result = import_contract(dbt_schema_dict, format="dbt")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert ("rows_distinct", {"columns_subset": "id"}) in methods

    def test_import_accepted_values(self, dbt_schema_dict):
        result = import_contract(dbt_schema_dict, format="dbt")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert (
            "col_vals_in_set",
            {"columns": "status", "set": ["active", "inactive", "pending"]},
        ) in methods

    def test_import_legacy_tests_key(self, dbt_schema_legacy_tests):
        result = import_contract(dbt_schema_legacy_tests, format="dbt")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert ("col_vals_not_null", {"columns": "order_id"}) in methods
        assert ("rows_distinct", {"columns_subset": "order_id"}) in methods

    def test_import_relationship_warning(self, dbt_schema_legacy_tests):
        result = import_contract(dbt_schema_legacy_tests, format="dbt")
        assert any(
            "relationship" in w.lower() or "cross-table" in w.lower() for w in result.warnings
        )
        assert result.coverage < 1.0

    def test_import_from_sources(self, dbt_sources_dict):
        result = import_contract(dbt_sources_dict, format="dbt")
        assert result.source_format == "dbt"
        assert len(result.columns) == 2
        col_map = dict(result.columns)
        assert col_map["event_id"] == "Int64"

    def test_import_specific_model(self):
        doc = {
            "version": 2,
            "models": [
                {"name": "first", "columns": [{"name": "a"}]},
                {"name": "second", "columns": [{"name": "b"}, {"name": "c"}]},
            ],
        }
        result = import_contract(doc, format="dbt", model="second")
        assert len(result.columns) == 2
        col_names = [name for name, _ in result.columns]
        assert "b" in col_names

    def test_import_model_not_found(self, dbt_schema_dict):
        with pytest.raises(ValueError, match="not found"):
            import_contract(dbt_schema_dict, format="dbt", model="nonexistent")

    def test_import_from_file(self, dbt_schema_dict):
        import yaml as _yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            _yaml.dump(dbt_schema_dict, f)
            f.flush()
            result = import_contract(f.name, format="dbt")

        assert result.source_format == "dbt"
        assert result.source_path == f.name

    def test_import_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            import_contract("/nonexistent/schema.yml", format="dbt")

    def test_import_invalid_type(self):
        with pytest.raises(TypeError, match="must be a file path"):
            import_contract(12345, format="dbt")

    def test_auto_detect_dbt(self, dbt_schema_dict):
        result = import_contract(dbt_schema_dict)
        assert result.source_format == "dbt"

    def test_to_validate_end_to_end(self, dbt_schema_dict, simple_df):
        result = import_contract(dbt_schema_dict, format="dbt")
        validation = result.to_validate(data=simple_df)
        validation.interrogate()

    def test_no_models_or_sources_raises(self):
        with pytest.raises(ValueError, match="No models or source tables"):
            import_contract({"version": 2}, format="dbt")


class TestDbtExport:
    def test_export_from_contract(self):
        contract = pb.Contract(
            name="test_model",
            description="A test model",
            schema=pb.Schema(id="Int64", name="String", age="Int64"),
            steps=[
                pb.Step("col_vals_not_null", columns="id"),
                pb.Step("rows_distinct", columns="id"),
                pb.Step("col_vals_in_set", columns="name", set=["Alice", "Bob"]),
            ],
        )
        result = export_contract(contract, format="dbt")

        assert result["version"] == 2
        assert len(result["models"]) == 1
        model = result["models"][0]
        assert model["name"] == "test_model"
        assert model["description"] == "A test model"

        col_map = {c["name"]: c for c in model["columns"]}
        assert "not_null" in col_map["id"]["data_tests"]
        assert "unique" in col_map["id"]["data_tests"]
        assert col_map["id"]["data_type"] == "integer"

    def test_export_to_file(self):
        import yaml as _yaml

        contract = pb.Contract(
            name="file_test",
            schema=pb.Schema(x="Int64"),
            steps=[],
        )
        with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as f:
            export_contract(contract, f.name, format="dbt")

        with open(f.name) as fh:
            data = _yaml.safe_load(fh)
        assert data["version"] == 2
        assert data["models"][0]["name"] == "file_test"

    def test_export_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected a Validate or Contract"):
            export_contract("not a contract", format="dbt")


class TestDbtRoundTrip:
    def test_dbt_roundtrip(self):
        original = {
            "version": 2,
            "models": [
                {
                    "name": "users",
                    "columns": [
                        {
                            "name": "id",
                            "data_type": "integer",
                            "data_tests": ["not_null", "unique"],
                        },
                        {
                            "name": "status",
                            "data_type": "string",
                            "data_tests": [{"accepted_values": {"values": ["a", "b"]}}],
                        },
                    ],
                }
            ],
        }
        imported = import_contract(original, format="dbt")
        contract = imported.to_contract(name="roundtrip")
        exported = export_contract(contract, format="dbt")
        reimported = import_contract(exported, format="dbt")

        def _hashable_kwargs(kwargs):
            items = []
            for k, v in sorted(kwargs.items()):
                items.append((k, tuple(v) if isinstance(v, list) else v))
            return tuple(items)

        original_methods = {(c.method, _hashable_kwargs(c.kwargs)) for c in imported.constraints}
        roundtrip_methods = {(c.method, _hashable_kwargs(c.kwargs)) for c in reimported.constraints}
        assert original_methods == roundtrip_methods


# ── ODCS adapter fixtures ────────────────────────────────────────────────────


@pytest.fixture
def odcs_v3_dict():
    """An ODCS v3 data contract as a dict."""
    return {
        "kind": "DataContract",
        "apiVersion": "v3.0.0",
        "info": {
            "title": "User Accounts",
            "description": "Contract for user account data",
        },
        "dataset": [
            {
                "table": "users",
                "columns": [
                    {
                        "column": "id",
                        "logicalType": "integer",
                        "isNullable": False,
                        "isUnique": True,
                    },
                    {
                        "column": "name",
                        "logicalType": "string",
                        "isNullable": False,
                    },
                    {
                        "column": "age",
                        "logicalType": "integer",
                        "minimum": 0,
                        "maximum": 150,
                    },
                    {
                        "column": "status",
                        "logicalType": "string",
                        "enum": ["active", "inactive", "pending"],
                    },
                    {
                        "column": "email",
                        "logicalType": "string",
                        "pattern": r"^[^@]+@[^@]+\.[^@]+$",
                    },
                ],
            }
        ],
    }


@pytest.fixture
def odcs_v2_dict():
    """An ODCS v2-style data contract as a dict."""
    return {
        "kind": "DataContract",
        "apiVersion": "v2.2.2",
        "datasetName": "orders",
        "description": "Order data contract",
        "dataset": [
            {
                "table": "orders",
                "columns": [
                    {
                        "column": "order_id",
                        "logicalType": "integer",
                        "isNullable": False,
                        "isPrimaryKey": True,
                    },
                    {
                        "column": "amount",
                        "logicalType": "float",
                        "minimum": 0,
                    },
                ],
            }
        ],
    }


class TestODCSImport:
    def test_import_from_dict_v3(self, odcs_v3_dict):
        result = import_contract(odcs_v3_dict, format="odcs")

        assert result.source_format == "odcs"
        assert result.source_version == "v3.0.0"
        assert len(result.columns) == 5
        assert result.metadata.get("title") == "User Accounts"
        assert result.metadata.get("description") == "Contract for user account data"

    def test_import_column_types(self, odcs_v3_dict):
        result = import_contract(odcs_v3_dict, format="odcs")
        col_map = dict(result.columns)
        assert col_map["id"] == "Int64"
        assert col_map["name"] == "String"
        assert col_map["age"] == "Int64"
        assert col_map["status"] == "String"
        assert col_map["email"] == "String"

    def test_import_not_null(self, odcs_v3_dict):
        result = import_contract(odcs_v3_dict, format="odcs")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert ("col_vals_not_null", {"columns": "id"}) in methods
        assert ("col_vals_not_null", {"columns": "name"}) in methods

    def test_import_unique(self, odcs_v3_dict):
        result = import_contract(odcs_v3_dict, format="odcs")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert ("rows_distinct", {"columns_subset": "id"}) in methods

    def test_import_min_max(self, odcs_v3_dict):
        result = import_contract(odcs_v3_dict, format="odcs")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert ("col_vals_ge", {"columns": "age", "value": 0}) in methods
        assert ("col_vals_le", {"columns": "age", "value": 150}) in methods

    def test_import_enum(self, odcs_v3_dict):
        result = import_contract(odcs_v3_dict, format="odcs")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert (
            "col_vals_in_set",
            {"columns": "status", "set": ["active", "inactive", "pending"]},
        ) in methods

    def test_import_pattern(self, odcs_v3_dict):
        result = import_contract(odcs_v3_dict, format="odcs")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert (
            "col_vals_regex",
            {"columns": "email", "pattern": r"^[^@]+@[^@]+\.[^@]+$"},
        ) in methods

    def test_import_v2(self, odcs_v2_dict):
        result = import_contract(odcs_v2_dict, format="odcs")
        assert result.source_format == "odcs"
        assert result.source_version == "v2.2.2"
        assert result.metadata.get("title") == "orders"

    def test_import_primary_key(self, odcs_v2_dict):
        result = import_contract(odcs_v2_dict, format="odcs")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert ("col_vals_not_null", {"columns": "order_id"}) in methods
        assert ("rows_distinct", {"columns_subset": "order_id"}) in methods

    def test_import_specific_table(self):
        doc = {
            "kind": "DataContract",
            "apiVersion": "v3.0.0",
            "info": {"title": "Multi"},
            "dataset": [
                {"table": "first", "columns": [{"column": "a", "logicalType": "string"}]},
                {"table": "second", "columns": [{"column": "b"}, {"column": "c"}]},
            ],
        }
        result = import_contract(doc, format="odcs", table="second")
        assert len(result.columns) == 2

    def test_import_table_not_found(self, odcs_v3_dict):
        with pytest.raises(ValueError, match="not found"):
            import_contract(odcs_v3_dict, format="odcs", table="nonexistent")

    def test_import_from_file_yaml(self, odcs_v3_dict):
        import yaml as _yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".odcs.yml", delete=False) as f:
            _yaml.dump(odcs_v3_dict, f)
            f.flush()
            result = import_contract(f.name, format="odcs")

        assert result.source_format == "odcs"
        assert result.source_path == f.name

    def test_import_from_file_json(self, odcs_v3_dict):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".odcs.json", delete=False) as f:
            json.dump(odcs_v3_dict, f)
            f.flush()
            result = import_contract(f.name, format="odcs")

        assert result.source_format == "odcs"

    def test_import_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            import_contract("/nonexistent/contract.odcs.yml", format="odcs")

    def test_import_invalid_type(self):
        with pytest.raises(TypeError, match="must be a file path"):
            import_contract(12345, format="odcs")

    def test_auto_detect_odcs(self, odcs_v3_dict):
        result = import_contract(odcs_v3_dict)
        assert result.source_format == "odcs"

    def test_to_validate_end_to_end(self, odcs_v3_dict, simple_df):
        result = import_contract(odcs_v3_dict, format="odcs")
        validation = result.to_validate(data=simple_df)
        validation.interrogate()

    def test_no_dataset_raises(self):
        with pytest.raises(ValueError, match="No 'dataset' section"):
            import_contract({"kind": "DataContract", "apiVersion": "v3.0.0"}, format="odcs")

    def test_minlength_warning(self):
        doc = {
            "kind": "DataContract",
            "apiVersion": "v3.0.0",
            "info": {"title": "test"},
            "dataset": [
                {
                    "table": "t",
                    "columns": [{"column": "name", "logicalType": "string", "minLength": 1}],
                }
            ],
        }
        result = import_contract(doc, format="odcs")
        assert any("minLength" in w for w in result.warnings)
        assert result.coverage < 1.0


class TestODCSExport:
    def test_export_from_contract(self):
        contract = pb.Contract(
            name="test_contract",
            description="A test contract",
            schema=pb.Schema(id="Int64", name="String", age="Int64"),
            steps=[
                pb.Step("col_vals_not_null", columns="id"),
                pb.Step("rows_distinct", columns="id"),
                pb.Step("col_vals_ge", columns="age", value=0),
            ],
        )
        result = export_contract(contract, format="odcs")

        assert result["kind"] == "DataContract"
        assert result["apiVersion"] == "v3.0.0"
        assert result["info"]["title"] == "test_contract"
        assert result["info"]["description"] == "A test contract"

        table = result["dataset"][0]
        col_map = {c["column"]: c for c in table["columns"]}
        assert col_map["id"]["isNullable"] is False
        assert col_map["id"]["isUnique"] is True
        assert col_map["id"]["logicalType"] == "integer"
        assert col_map["age"]["minimum"] == 0

    def test_export_to_file_yaml(self):
        import yaml as _yaml

        contract = pb.Contract(name="file_test", schema=pb.Schema(x="Int64"), steps=[])
        with tempfile.NamedTemporaryFile(suffix=".odcs.yml", delete=False) as f:
            export_contract(contract, f.name, format="odcs")

        with open(f.name) as fh:
            data = _yaml.safe_load(fh)
        assert data["kind"] == "DataContract"

    def test_export_to_file_json(self):
        contract = pb.Contract(name="file_test", schema=pb.Schema(x="Int64"), steps=[])
        with tempfile.NamedTemporaryFile(suffix=".odcs.json", delete=False) as f:
            export_contract(contract, f.name, format="odcs")

        with open(f.name) as fh:
            data = json.load(fh)
        assert data["kind"] == "DataContract"

    def test_export_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected a Validate or Contract"):
            export_contract("not a contract", format="odcs")


class TestODCSRoundTrip:
    def test_odcs_roundtrip(self):
        original = {
            "kind": "DataContract",
            "apiVersion": "v3.0.0",
            "info": {"title": "test"},
            "dataset": [
                {
                    "table": "users",
                    "columns": [
                        {
                            "column": "id",
                            "logicalType": "integer",
                            "isNullable": False,
                            "isUnique": True,
                        },
                        {"column": "age", "logicalType": "integer", "minimum": 0, "maximum": 150},
                        {"column": "status", "logicalType": "string", "enum": ["a", "b"]},
                    ],
                }
            ],
        }
        imported = import_contract(original, format="odcs")
        contract = imported.to_contract(name="roundtrip")
        exported = export_contract(contract, format="odcs")
        reimported = import_contract(exported, format="odcs")

        def _hashable_kwargs(kwargs):
            items = []
            for k, v in sorted(kwargs.items()):
                items.append((k, tuple(v) if isinstance(v, list) else v))
            return tuple(items)

        original_methods = {(c.method, _hashable_kwargs(c.kwargs)) for c in imported.constraints}
        roundtrip_methods = {(c.method, _hashable_kwargs(c.kwargs)) for c in reimported.constraints}
        assert original_methods == roundtrip_methods


# ---------------------------------------------------------------------------
# MappedConstraint.__repr__
# ---------------------------------------------------------------------------


def test_mapped_constraint_repr_no_kwargs():
    mc = MappedConstraint(method="col_vals_not_null")
    r = repr(mc)
    assert "MappedConstraint" in r
    assert "col_vals_not_null" in r


def test_mapped_constraint_repr_with_kwargs():
    mc = MappedConstraint(method="col_vals_gt", kwargs={"columns": "age", "value": 0})
    r = repr(mc)
    assert "col_vals_gt" in r
    assert "columns=" in r
    assert "value=" in r


# ---------------------------------------------------------------------------
# ContractImport.__repr__
# ---------------------------------------------------------------------------


def test_contract_import_repr():
    ci = ContractImport(
        source_format="test_fmt",
        columns=[("a", "Int64"), ("b", "String")],
        constraints=[MappedConstraint("col_vals_not_null", kwargs={"columns": "a"})],
        coverage=0.75,
    )
    r = repr(ci)
    assert "ContractImport" in r
    assert "test_fmt" in r
    assert "2" in r
    assert "75%" in r


# ---------------------------------------------------------------------------
# ContractImport.summary()
# ---------------------------------------------------------------------------


def test_summary_minimal():
    ci = ContractImport(source_format="fmt")
    s = ci.summary()
    assert "fmt" in s
    assert "Columns detected: 0" in s
    assert "Constraints mapped: 0" in s


def test_summary_with_source_path_and_version():
    ci = ContractImport(
        source_format="fmt",
        source_path="/some/file.yml",
        source_version="2.0",
        columns=[("x", "Int64")],
    )
    s = ci.summary()
    assert "/some/file.yml" in s
    assert "2.0" in s
    assert "Columns detected: 1" in s


def test_summary_with_warnings():
    ci = ContractImport(
        source_format="fmt",
        warnings=["unknown constraint foo", "another warning"],
        coverage=0.5,
    )
    s = ci.summary()
    assert "Warnings" in s
    assert "unknown constraint foo" in s
    assert "50%" in s


# ---------------------------------------------------------------------------
# ContractImport.to_python()
# ---------------------------------------------------------------------------


def test_to_python_basic():
    ci = ContractImport(
        source_format="fmt",
        columns=[("id", "Int64")],
        constraints=[MappedConstraint("col_vals_not_null", kwargs={"columns": "id"})],
    )
    code = ci.to_python()
    assert "import pointblank as pb" in code
    assert "col_schema_match" in code
    assert "col_vals_not_null" in code
    assert "interrogate()" in code


def test_to_python_no_columns_no_constraints():
    ci = ContractImport(source_format="fmt")
    code = ci.to_python()
    assert "import pointblank as pb" in code
    assert "col_schema_match" not in code
    assert "interrogate()" in code


# ---------------------------------------------------------------------------
# ContractImport.to_yaml()
# ---------------------------------------------------------------------------


def test_to_yaml_basic():
    ci = ContractImport(
        source_format="fmt",
        columns=[("age", "Int64")],
        constraints=[MappedConstraint("col_vals_ge", kwargs={"columns": "age", "value": 0})],
    )
    yml = ci.to_yaml()
    assert "validation" in yml
    assert "col_schema_match" in yml
    assert "col_vals_ge" in yml


def test_to_yaml_no_typed_columns():
    ci = ContractImport(
        source_format="fmt",
        columns=[("age", None)],
        constraints=[MappedConstraint("col_vals_not_null", kwargs={"columns": "age"})],
    )
    yml = ci.to_yaml()
    assert "col_schema_match" not in yml
    assert "col_vals_not_null" in yml


# ---------------------------------------------------------------------------
# ContractImport.to_validate() — unknown method skipped with warning
# ---------------------------------------------------------------------------


def test_to_validate_unknown_method_adds_warning(simple_df):
    ci = ContractImport(
        source_format="fmt",
        constraints=[MappedConstraint("no_such_method_xyz", kwargs={"columns": "id"})],
    )
    ci.to_validate(data=simple_df)
    assert any("no_such_method_xyz" in w for w in ci.warnings)


# ---------------------------------------------------------------------------
# ContractImport.to_contract()
# ---------------------------------------------------------------------------


def test_to_contract_uses_metadata_description():
    ci = ContractImport(
        source_format="fmt",
        columns=[("id", "Int64")],
        constraints=[MappedConstraint("col_vals_not_null", kwargs={"columns": "id"})],
        metadata={"description": "My contract description"},
    )
    contract = ci.to_contract(name="meta_test")
    assert contract.description == "My contract description"


def test_to_contract_no_typed_columns_no_schema():
    ci = ContractImport(
        source_format="fmt",
        columns=[("id", None)],
        constraints=[],
    )
    contract = ci.to_contract()
    assert contract.schema is None


# ---------------------------------------------------------------------------
# ContractAdapter class-level attributes
# ---------------------------------------------------------------------------


def test_contract_adapter_defaults():
    adapter = ContractAdapter()
    assert adapter.format_name == ""
    assert adapter.file_extensions == []
    assert adapter.supports_import is True
    assert adapter.supports_export is True


# ---------------------------------------------------------------------------
# register_adapter — no format_name raises ValueError
# ---------------------------------------------------------------------------


def test_register_adapter_no_name_raises():
    with pytest.raises(ValueError, match="must have a `format_name`"):

        @register_adapter()
        class BadAdapter(ContractAdapter):
            format_name = ""


# ---------------------------------------------------------------------------
# register_adapter — used without parentheses (class passed directly)
# ---------------------------------------------------------------------------


def test_register_adapter_no_parens():
    @register_adapter
    class NoParensAdapter(ContractAdapter):
        format_name = "_test_no_parens_adapter"
        file_extensions = [".npa"]
        supports_export = False

    assert "_test_no_parens_adapter" in _ADAPTER_REGISTRY
    _ADAPTER_REGISTRY.pop("_test_no_parens_adapter", None)


# ---------------------------------------------------------------------------
# list_adapters() returns expected structure
# ---------------------------------------------------------------------------


def test_list_adapters_returns_known_format():
    result = list_adapters()
    assert isinstance(result, dict)
    assert "odcs" in result
    entry = result["odcs"]
    assert "class" in entry
    assert "file_extensions" in entry
    assert "supports_import" in entry
    assert "supports_export" in entry


# ---------------------------------------------------------------------------
# get_adapter — unknown format raises ValueError
# ---------------------------------------------------------------------------


def test_get_adapter_unknown_raises():
    with pytest.raises(ValueError, match="No adapter registered"):
        get_adapter("_no_such_format_xyz_")


# ---------------------------------------------------------------------------
# _detect_format — extension loop (file path ending in registered extension)
# ---------------------------------------------------------------------------


def test_detect_format_extension_loop():
    from pointblank.adapters._registry import _detect_format

    detected = _detect_format("myfile.odcs.json")
    assert detected == "odcs"


# ---------------------------------------------------------------------------
# import_contract — auto-detect fails raises ValueError
# ---------------------------------------------------------------------------


def test_import_contract_auto_detect_fails():
    with pytest.raises(ValueError, match="Could not auto-detect format"):
        import_contract({"completely": "unknown"})


# ---------------------------------------------------------------------------
# import_contract — adapter that does not support import raises ValueError
# ---------------------------------------------------------------------------


def test_import_contract_adapter_no_import_support():
    @register_adapter("_test_no_import_fmt")
    class NoImportAdapter(ContractAdapter):
        format_name = "_test_no_import_fmt"
        supports_import = False

        @staticmethod
        def detect(source):
            return False

    try:
        with pytest.raises(ValueError, match="does not support import"):
            import_contract({"x": 1}, format="_test_no_import_fmt")
    finally:
        _ADAPTER_REGISTRY.pop("_test_no_import_fmt", None)


# ---------------------------------------------------------------------------
# export_contract — adapter that does not support export raises ValueError
# ---------------------------------------------------------------------------


def test_export_contract_adapter_no_export_support():
    @register_adapter("_test_no_export_fmt")
    class NoExportAdapter(ContractAdapter):
        format_name = "_test_no_export_fmt"
        supports_export = False

        @staticmethod
        def detect(source):
            return False

    contract = pb.Contract(name="x", schema=None, steps=[])
    try:
        with pytest.raises(ValueError, match="does not support export"):
            export_contract(contract, format="_test_no_export_fmt")
    finally:
        _ADAPTER_REGISTRY.pop("_test_no_export_fmt", None)


# ---------------------------------------------------------------------------
# Frictionless — additional coverage
# ---------------------------------------------------------------------------


class TestFrictionlessAdditional:
    def test_import_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            import_contract("/nonexistent/schema.json", format="frictionless")

    def test_import_invalid_type(self):
        with pytest.raises(TypeError, match="must be a file path"):
            import_contract(12345, format="frictionless")

    def test_export_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Expected a Validate or Contract"):
            export_contract("not valid", format="frictionless")

    def test_detect_string_valid_json_file(self, frictionless_schema_dict):
        adapter = get_adapter("frictionless")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(frictionless_schema_dict, f)
            fname = f.name
        assert adapter.detect(fname) is True

    def test_detect_string_nonexistent(self):
        adapter = get_adapter("frictionless")
        assert adapter.detect("/nonexistent/file.json") is False

    def test_detect_resources_dict(self):
        adapter = get_adapter("frictionless")
        assert adapter.detect({"resources": [{"name": "x"}]}) is True

    def test_detect_no_fields_no_resources(self):
        adapter = get_adapter("frictionless")
        assert adapter.detect({"other": "data"}) is False

    def test_detect_invalid_json_file(self):
        adapter = get_adapter("frictionless")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            fname = f.name
        assert adapter.detect(fname) is False

    def test_detect_non_string_non_dict(self):
        adapter = get_adapter("frictionless")
        assert adapter.detect(42) is False

    def test_import_empty_resources_raises(self):
        with pytest.raises(ValueError, match="no resources"):
            import_contract({"resources": []}, format="frictionless")

    def test_import_resource_index_out_of_range(self, frictionless_datapackage_dict):
        with pytest.raises(IndexError, match="out of range"):
            import_contract(frictionless_datapackage_dict, format="frictionless", resource=99)

    def test_import_resource_invalid_key_type(self, frictionless_datapackage_dict):
        with pytest.raises(TypeError, match="resource must be str or int"):
            import_contract(frictionless_datapackage_dict, format="frictionless", resource=1.5)

    def test_import_resource_no_schema_fields(self):
        doc = {"resources": [{"name": "x", "schema": {"other_key": []}}]}
        with pytest.raises(ValueError, match="schema.fields"):
            import_contract(doc, format="frictionless")

    def test_import_neither_fields_nor_resources_raises(self):
        with pytest.raises(ValueError, match="neither"):
            import_contract({"something_else": True}, format="frictionless")

    def test_import_description_metadata(self):
        schema = {
            "fields": [{"name": "x", "type": "string"}],
            "description": "My table description",
        }
        result = import_contract(schema, format="frictionless")
        assert result.metadata.get("description") == "My table description"

    def test_import_composite_primary_key(self):
        schema = {
            "fields": [
                {"name": "a", "type": "integer"},
                {"name": "b", "type": "string"},
            ],
            "primaryKey": ["a", "b"],
        }
        result = import_contract(schema, format="frictionless")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert ("col_vals_not_null", {"columns": "a"}) in methods
        assert ("col_vals_not_null", {"columns": "b"}) in methods
        distinct = [c for c in result.constraints if c.method == "rows_distinct"]
        assert any(isinstance(c.kwargs.get("columns_subset"), list) for c in distinct)

    def test_import_field_any_type(self):
        schema = {"fields": [{"name": "x", "type": "any"}]}
        result = import_contract(schema, format="frictionless")
        col_map = dict(result.columns)
        assert col_map["x"] is None

    def test_export_from_validate(self, simple_df):
        v = pb.Validate(data=simple_df)
        v.col_vals_not_null(columns="id")
        v.col_vals_ge(columns="age", value=0)
        v.interrogate()
        result = export_contract(v, format="frictionless")
        assert "fields" in result
        field_names = [f["name"] for f in result["fields"]]
        assert "id" in field_names

    def test_export_from_validate_with_set(self, simple_df):
        v = pb.Validate(data=simple_df)
        v.col_vals_in_set(columns="status", set=["active", "inactive"])
        v.interrogate()
        result = export_contract(v, format="frictionless")
        assert "fields" in result

    def test_pb_dtype_to_frictionless_type_int(self):
        from pointblank.adapters._frictionless import _pb_dtype_to_frictionless_type
        assert _pb_dtype_to_frictionless_type("Int64") == "integer"

    def test_pb_dtype_to_frictionless_type_float(self):
        from pointblank.adapters._frictionless import _pb_dtype_to_frictionless_type
        assert _pb_dtype_to_frictionless_type("Float64") == "number"
        assert _pb_dtype_to_frictionless_type("Double") == "number"
        assert _pb_dtype_to_frictionless_type("Decimal") == "number"

    def test_pb_dtype_to_frictionless_type_string(self):
        from pointblank.adapters._frictionless import _pb_dtype_to_frictionless_type
        assert _pb_dtype_to_frictionless_type("String") == "string"
        assert _pb_dtype_to_frictionless_type("Utf8") == "string"
        assert _pb_dtype_to_frictionless_type("Object") == "string"

    def test_pb_dtype_to_frictionless_type_bool(self):
        from pointblank.adapters._frictionless import _pb_dtype_to_frictionless_type
        assert _pb_dtype_to_frictionless_type("Boolean") == "boolean"

    def test_pb_dtype_to_frictionless_type_datetime(self):
        from pointblank.adapters._frictionless import _pb_dtype_to_frictionless_type
        assert _pb_dtype_to_frictionless_type("Datetime") == "datetime"
        assert _pb_dtype_to_frictionless_type("Timestamp") == "datetime"

    def test_pb_dtype_to_frictionless_type_date(self):
        from pointblank.adapters._frictionless import _pb_dtype_to_frictionless_type
        assert _pb_dtype_to_frictionless_type("Date") == "date"

    def test_pb_dtype_to_frictionless_type_time(self):
        from pointblank.adapters._frictionless import _pb_dtype_to_frictionless_type
        assert _pb_dtype_to_frictionless_type("Time") == "time"

    def test_pb_dtype_to_frictionless_type_duration(self):
        from pointblank.adapters._frictionless import _pb_dtype_to_frictionless_type
        assert _pb_dtype_to_frictionless_type("Duration") == "duration"

    def test_pb_dtype_to_frictionless_type_unknown(self):
        from pointblank.adapters._frictionless import _pb_dtype_to_frictionless_type
        assert _pb_dtype_to_frictionless_type("UnknownType") is None

    def test_apply_step_to_fields_new_column(self):
        from pointblank.adapters._frictionless import _apply_step_to_fields
        fields = []
        field_map = {}
        _apply_step_to_fields("col_vals_not_null", {"columns": "new_col"}, field_map, fields)
        assert len(fields) == 1
        assert fields[0]["name"] == "new_col"
        assert fields[0]["constraints"]["required"] is True

    def test_apply_step_to_fields_list_columns(self):
        from pointblank.adapters._frictionless import _apply_step_to_fields
        fields = []
        field_map = {}
        _apply_step_to_fields("col_vals_not_null", {"columns": ["a", "b"]}, field_map, fields)
        assert len(fields) == 2

    def test_apply_step_to_fields_no_columns(self):
        from pointblank.adapters._frictionless import _apply_step_to_fields
        fields = []
        field_map = {}
        _apply_step_to_fields("col_vals_not_null", {}, field_map, fields)
        assert len(fields) == 0

    def test_apply_step_to_fields_invalid_column_type(self):
        from pointblank.adapters._frictionless import _apply_step_to_fields
        fields = []
        field_map = {}
        _apply_step_to_fields("col_vals_not_null", {"columns": 42}, field_map, fields)
        assert len(fields) == 0

    def test_apply_step_rows_distinct(self):
        from pointblank.adapters._frictionless import _apply_step_to_fields
        fields = [{"name": "id"}]
        field_map = {"id": fields[0]}
        _apply_step_to_fields("rows_distinct", {"columns_subset": "id"}, field_map, fields)
        assert fields[0]["constraints"]["unique"] is True

    def test_apply_step_col_vals_le(self):
        from pointblank.adapters._frictionless import _apply_step_to_fields
        fields = [{"name": "age"}]
        field_map = {"age": fields[0]}
        _apply_step_to_fields("col_vals_le", {"columns": "age", "value": 100}, field_map, fields)
        assert fields[0]["constraints"]["maximum"] == 100

    def test_apply_step_col_vals_in_set(self):
        from pointblank.adapters._frictionless import _apply_step_to_fields
        fields = [{"name": "status"}]
        field_map = {"status": fields[0]}
        _apply_step_to_fields(
            "col_vals_in_set", {"columns": "status", "set": ["a", "b"]}, field_map, fields
        )
        assert fields[0]["constraints"]["enum"] == ["a", "b"]

    def test_apply_step_col_vals_regex(self):
        from pointblank.adapters._frictionless import _apply_step_to_fields
        fields = [{"name": "email"}]
        field_map = {"email": fields[0]}
        _apply_step_to_fields(
            "col_vals_regex", {"columns": "email", "pattern": r"\S+@\S+"}, field_map, fields
        )
        assert fields[0]["constraints"]["pattern"] == r"\S+@\S+"

    def test_extract_validate_step_kwargs_list_values(self):
        from pointblank.adapters._frictionless import _extract_validate_step_kwargs_from_info

        class FakeStep:
            column = "status"
            values = ["a", "b", "c"]
            val_info = None

        kwargs = _extract_validate_step_kwargs_from_info(FakeStep())
        assert kwargs["columns"] == "status"
        assert kwargs["set"] == ["a", "b", "c"]

    def test_extract_validate_step_kwargs_scalar_value(self):
        from pointblank.adapters._frictionless import _extract_validate_step_kwargs_from_info

        class FakeStep:
            column = "age"
            values = 10
            val_info = None

        kwargs = _extract_validate_step_kwargs_from_info(FakeStep())
        assert kwargs["value"] == 10

    def test_extract_validate_step_kwargs_pattern(self):
        from pointblank.adapters._frictionless import _extract_validate_step_kwargs_from_info

        class FakeStep:
            column = "email"
            values = None
            val_info = {"pattern": r"\S+@\S+"}

        kwargs = _extract_validate_step_kwargs_from_info(FakeStep())
        assert kwargs["pattern"] == r"\S+@\S+"

    def test_extract_validate_step_kwargs_no_attrs(self):
        from pointblank.adapters._frictionless import _extract_validate_step_kwargs_from_info

        class FakeStep:
            pass

        kwargs = _extract_validate_step_kwargs_from_info(FakeStep())
        assert kwargs == {}

    def test_extract_validate_step_kwargs_tuple_values(self):
        from pointblank.adapters._frictionless import _extract_validate_step_kwargs_from_info

        class FakeStep:
            column = "x"
            values = (1, 2)
            val_info = {}

        kwargs = _extract_validate_step_kwargs_from_info(FakeStep())
        assert kwargs["set"] == [1, 2]


# ---------------------------------------------------------------------------
# ODCS — additional coverage
# ---------------------------------------------------------------------------


class TestODCSAdditional:
    def test_normalize_odcs_type_with_precision(self):
        from pointblank.adapters._odcs import _normalize_odcs_type
        assert _normalize_odcs_type("varchar(255)") == "String"
        assert _normalize_odcs_type("numeric(10,2)") == "Float64"
        assert _normalize_odcs_type("TIMESTAMP_NTZ") == "Datetime"

    def test_detect_string_valid_yaml_file(self, odcs_v3_dict):
        import yaml as _yaml
        adapter = get_adapter("odcs")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            _yaml.dump(odcs_v3_dict, f)
            fname = f.name
        assert adapter.detect(fname) is True

    def test_detect_string_nonexistent(self):
        adapter = get_adapter("odcs")
        assert adapter.detect("/nonexistent/contract.yml") is False

    def test_detect_non_string_non_dict(self):
        adapter = get_adapter("odcs")
        assert adapter.detect(42) is False

    def test_detect_invalid_yaml_file(self):
        adapter = get_adapter("odcs")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(": : }{{{")
            fname = f.name
        result = adapter.detect(fname)
        assert isinstance(result, bool)

    def test_detect_json_file(self, odcs_v3_dict):
        adapter = get_adapter("odcs")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(odcs_v3_dict, f)
            fname = f.name
        assert adapter.detect(fname) is True

    def test_is_odcs_apiversion_dataset(self):
        from pointblank.adapters._odcs import _is_odcs
        assert _is_odcs({"apiVersion": "v3.0.0", "dataset": []}) is True
        assert _is_odcs({"apiVersion": "v3.0.0", "schema": []}) is True
        assert _is_odcs({"something": "else"}) is False

    def test_schema_flat_list_v3(self):
        doc = {
            "kind": "DataContract",
            "apiVersion": "v3.0.0",
            "info": {"title": "test"},
            "schema": [
                {"column": "id", "logicalType": "integer"},
                {"column": "name", "logicalType": "string"},
            ],
        }
        result = import_contract(doc, format="odcs")
        assert len(result.columns) == 2
        col_map = dict(result.columns)
        assert col_map["id"] == "Int64"

    def test_import_values_field(self):
        doc = {
            "kind": "DataContract",
            "apiVersion": "v3.0.0",
            "info": {"title": "test"},
            "dataset": [
                {
                    "table": "t",
                    "columns": [
                        {"column": "status", "logicalType": "string", "values": ["a", "b", "c"]}
                    ],
                }
            ],
        }
        result = import_contract(doc, format="odcs")
        methods = [(c.method, c.kwargs) for c in result.constraints]
        assert ("col_vals_in_set", {"columns": "status", "set": ["a", "b", "c"]}) in methods

    def test_import_checks_warning(self):
        doc = {
            "kind": "DataContract",
            "apiVersion": "v3.0.0",
            "info": {"title": "test"},
            "dataset": [
                {
                    "table": "t",
                    "columns": [
                        {
                            "column": "x",
                            "logicalType": "integer",
                            "checks": [{"name": "custom_check", "value": 0}],
                        }
                    ],
                }
            ],
        }
        result = import_contract(doc, format="odcs")
        assert any("custom check" in w for w in result.warnings)
        assert result.coverage < 1.0

    def test_import_maxlength_warning(self):
        doc = {
            "kind": "DataContract",
            "apiVersion": "v3.0.0",
            "info": {"title": "test"},
            "dataset": [
                {
                    "table": "t",
                    "columns": [{"column": "name", "logicalType": "string", "maxLength": 100}],
                }
            ],
        }
        result = import_contract(doc, format="odcs")
        assert any("maxLength" in w for w in result.warnings)

    def test_import_table_by_name_key(self):
        doc = {
            "kind": "DataContract",
            "apiVersion": "v3.0.0",
            "info": {"title": "test"},
            "dataset": [
                {"name": "first_table", "columns": [{"column": "a", "logicalType": "string"}]},
            ],
        }
        result = import_contract(doc, format="odcs", table="first_table")
        assert len(result.columns) == 1

    def test_import_doc_not_dict_raises(self):
        import yaml as _yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("- item1\n- item2\n")
            fname = f.name
        with pytest.raises(ValueError, match="must be a YAML/JSON mapping"):
            import_contract(fname, format="odcs")

    def test_export_from_validate(self, simple_df):
        v = pb.Validate(data=simple_df)
        v.col_vals_not_null(columns="id")
        v.col_vals_ge(columns="age", value=0)
        v.interrogate()
        result = export_contract(v, format="odcs")
        assert result["kind"] == "DataContract"
        table = result["dataset"][0]
        col_map = {c["column"]: c for c in table["columns"]}
        assert col_map["id"]["isNullable"] is False

    def test_export_from_validate_in_set(self, simple_df):
        v = pb.Validate(data=simple_df)
        v.col_vals_in_set(columns="status", set=["active", "inactive"])
        v.col_vals_regex(columns="email", pattern=r".+@.+")
        v.col_vals_le(columns="age", value=150)
        v.interrogate()
        result = export_contract(v, format="odcs")
        assert result["kind"] == "DataContract"

    def test_pb_dtype_to_odcs_type_all_branches(self):
        from pointblank.adapters._odcs import _pb_dtype_to_odcs_type
        assert _pb_dtype_to_odcs_type("Int64") == "integer"
        assert _pb_dtype_to_odcs_type("Float64") == "number"
        assert _pb_dtype_to_odcs_type("Double") == "number"
        assert _pb_dtype_to_odcs_type("Decimal") == "number"
        assert _pb_dtype_to_odcs_type("String") == "string"
        assert _pb_dtype_to_odcs_type("Utf8") == "string"
        assert _pb_dtype_to_odcs_type("Object") == "string"
        assert _pb_dtype_to_odcs_type("Boolean") == "boolean"
        assert _pb_dtype_to_odcs_type("Datetime") == "timestamp"
        assert _pb_dtype_to_odcs_type("Timestamp") == "timestamp"
        assert _pb_dtype_to_odcs_type("Date") == "date"
        assert _pb_dtype_to_odcs_type("Time") == "time"
        assert _pb_dtype_to_odcs_type("UnknownType") == "string"

    def test_apply_step_to_odcs_columns_not_null(self):
        from pointblank.adapters._odcs import _apply_step_to_odcs_columns
        col_defs = []
        col_map = {}
        _apply_step_to_odcs_columns("col_vals_not_null", {"columns": "a"}, col_map, col_defs)
        assert col_map["a"]["isNullable"] is False

    def test_apply_step_to_odcs_columns_distinct(self):
        from pointblank.adapters._odcs import _apply_step_to_odcs_columns
        col_defs = []
        col_map = {}
        _apply_step_to_odcs_columns(
            "rows_distinct", {"columns_subset": "a"}, col_map, col_defs
        )
        assert col_map["a"]["isUnique"] is True

    def test_apply_step_to_odcs_columns_in_set(self):
        from pointblank.adapters._odcs import _apply_step_to_odcs_columns
        col_defs = []
        col_map = {}
        _apply_step_to_odcs_columns(
            "col_vals_in_set", {"columns": "b", "set": ["x", "y"]}, col_map, col_defs
        )
        assert col_map["b"]["enum"] == ["x", "y"]

    def test_apply_step_to_odcs_columns_regex(self):
        from pointblank.adapters._odcs import _apply_step_to_odcs_columns
        col_defs = []
        col_map = {}
        _apply_step_to_odcs_columns(
            "col_vals_regex", {"columns": "c", "pattern": r"\d+"}, col_map, col_defs
        )
        assert col_map["c"]["pattern"] == r"\d+"

    def test_apply_step_to_odcs_columns_ge_le(self):
        from pointblank.adapters._odcs import _apply_step_to_odcs_columns
        col_defs = []
        col_map = {}
        _apply_step_to_odcs_columns("col_vals_ge", {"columns": "d", "value": 0}, col_map, col_defs)
        assert col_map["d"]["minimum"] == 0
        _apply_step_to_odcs_columns(
            "col_vals_le", {"columns": "d", "value": 100}, col_map, col_defs
        )
        assert col_map["d"]["maximum"] == 100

    def test_apply_step_no_columns(self):
        from pointblank.adapters._odcs import _apply_step_to_odcs_columns
        col_defs = []
        col_map = {}
        _apply_step_to_odcs_columns("col_vals_not_null", {}, col_map, col_defs)
        assert len(col_defs) == 0

    def test_apply_step_list_columns(self):
        from pointblank.adapters._odcs import _apply_step_to_odcs_columns
        col_defs = []
        col_map = {}
        _apply_step_to_odcs_columns(
            "col_vals_not_null", {"columns": ["a", "b"]}, col_map, col_defs
        )
        assert len(col_defs) == 2

    def test_apply_step_invalid_column_type(self):
        from pointblank.adapters._odcs import _apply_step_to_odcs_columns
        col_defs = []
        col_map = {}
        _apply_step_to_odcs_columns("col_vals_not_null", {"columns": 42}, col_map, col_defs)
        assert len(col_defs) == 0

    def test_extract_validate_step_kwargs_list(self):
        from pointblank.adapters._odcs import _extract_validate_step_kwargs

        class FakeStep:
            column = "status"
            values = ["a", "b"]

        kwargs = _extract_validate_step_kwargs(FakeStep())
        assert kwargs["columns"] == "status"
        assert kwargs["set"] == ["a", "b"]

    def test_extract_validate_step_kwargs_tuple(self):
        from pointblank.adapters._odcs import _extract_validate_step_kwargs

        class FakeStep:
            column = "x"
            values = ("p", "q")

        kwargs = _extract_validate_step_kwargs(FakeStep())
        assert kwargs["set"] == ["p", "q"]

    def test_extract_validate_step_kwargs_scalar(self):
        from pointblank.adapters._odcs import _extract_validate_step_kwargs

        class FakeStep:
            column = "age"
            values = 10

        kwargs = _extract_validate_step_kwargs(FakeStep())
        assert kwargs["value"] == 10

    def test_extract_validate_step_kwargs_no_attrs(self):
        from pointblank.adapters._odcs import _extract_validate_step_kwargs

        class FakeStep:
            pass

        kwargs = _extract_validate_step_kwargs(FakeStep())
        assert kwargs == {}


# ---------------------------------------------------------------------------
# dbt — additional coverage
# ---------------------------------------------------------------------------


class TestDbtAdditional:
    def test_normalize_dbt_type_with_precision(self):
        from pointblank.adapters._dbt import _normalize_dbt_type
        assert _normalize_dbt_type("varchar(256)") == "String"
        assert _normalize_dbt_type("numeric(10,2)") == "Float64"
        assert _normalize_dbt_type("TIMESTAMP_NTZ") == "Datetime"

    def test_detect_string_valid_yaml_file(self, dbt_schema_dict):
        import yaml as _yaml
        adapter = get_adapter("dbt")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            _yaml.dump(dbt_schema_dict, f)
            fname = f.name
        assert adapter.detect(fname) is True

    def test_detect_string_nonexistent(self):
        adapter = get_adapter("dbt")
        assert adapter.detect("/nonexistent/schema.yml") is False

    def test_detect_non_string_non_dict(self):
        adapter = get_adapter("dbt")
        assert adapter.detect(42) is False

    def test_detect_invalid_yaml_file(self):
        adapter = get_adapter("dbt")
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write(": : }{{{")
            fname = f.name
        result = adapter.detect(fname)
        assert isinstance(result, bool)

    def test_is_dbt_schema_models(self):
        from pointblank.adapters._dbt import _is_dbt_schema
        assert _is_dbt_schema({"models": []}) is True

    def test_is_dbt_schema_sources(self):
        from pointblank.adapters._dbt import _is_dbt_schema
        assert _is_dbt_schema({"sources": []}) is True

    def test_is_dbt_schema_false(self):
        from pointblank.adapters._dbt import _is_dbt_schema
        assert _is_dbt_schema({"other": "data"}) is False

    def test_import_doc_not_dict_raises(self):
        import yaml as _yaml
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            f.write("- item1\n- item2\n")
            fname = f.name
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            import_contract(fname, format="dbt")

    def test_import_unknown_string_test_warning(self):
        doc = {
            "version": 2,
            "models": [
                {
                    "name": "test_model",
                    "columns": [{"name": "col", "data_tests": ["not_null", "some_custom_test"]}],
                }
            ],
        }
        result = import_contract(doc, format="dbt")
        assert any("some_custom_test" in w for w in result.warnings)

    def test_import_dict_test_not_null(self):
        doc = {
            "version": 2,
            "models": [
                {"name": "m", "columns": [{"name": "col", "data_tests": [{"not_null": {}}]}]}
            ],
        }
        result = import_contract(doc, format="dbt")
        methods = [c.method for c in result.constraints]
        assert "col_vals_not_null" in methods

    def test_import_dict_test_unique(self):
        doc = {
            "version": 2,
            "models": [
                {"name": "m", "columns": [{"name": "col", "data_tests": [{"unique": {}}]}]}
            ],
        }
        result = import_contract(doc, format="dbt")
        methods = [c.method for c in result.constraints]
        assert "rows_distinct" in methods

    def test_import_dict_test_unknown_warns(self):
        doc = {
            "version": 2,
            "models": [
                {
                    "name": "m",
                    "columns": [
                        {"name": "col", "data_tests": [{"custom_test": {"param": "val"}}]}
                    ],
                }
            ],
        }
        result = import_contract(doc, format="dbt")
        assert any("custom_test" in w for w in result.warnings)

    def test_import_unrecognized_test_format(self):
        doc = {
            "version": 2,
            "models": [{"name": "m", "columns": [{"name": "col", "data_tests": [42]}]}],
        }
        result = import_contract(doc, format="dbt")
        assert any("unrecognized" in w.lower() for w in result.warnings)

    def test_import_model_level_unique_combo(self):
        doc = {
            "version": 2,
            "models": [
                {
                    "name": "m",
                    "columns": [{"name": "a"}, {"name": "b"}],
                    "data_tests": [{"unique": {"combination_of_columns": ["a", "b"]}}],
                }
            ],
        }
        result = import_contract(doc, format="dbt")
        distinct = [c for c in result.constraints if c.method == "rows_distinct"]
        assert len(distinct) == 1
        assert distinct[0].kwargs["columns_subset"] == ["a", "b"]

    def test_import_model_level_unique_columns_key(self):
        doc = {
            "version": 2,
            "models": [
                {
                    "name": "m",
                    "columns": [{"name": "a"}, {"name": "b"}],
                    "data_tests": [{"unique": {"columns": ["a", "b"]}}],
                }
            ],
        }
        result = import_contract(doc, format="dbt")
        distinct = [c for c in result.constraints if c.method == "rows_distinct"]
        assert len(distinct) == 1

    def test_import_model_level_unique_no_columns_warns(self):
        doc = {
            "version": 2,
            "models": [
                {
                    "name": "m",
                    "columns": [{"name": "a"}],
                    "data_tests": [{"unique": {}}],
                }
            ],
        }
        result = import_contract(doc, format="dbt")
        assert any("no columns" in w.lower() for w in result.warnings)

    def test_import_model_level_unknown_test(self):
        doc = {
            "version": 2,
            "models": [
                {
                    "name": "m",
                    "columns": [{"name": "a"}],
                    "data_tests": [{"custom_model_test": {"param": "val"}}],
                }
            ],
        }
        result = import_contract(doc, format="dbt")
        assert any("custom_model_test" in w for w in result.warnings)

    def test_import_model_level_non_dict_test(self):
        doc = {
            "version": 2,
            "models": [
                {
                    "name": "m",
                    "columns": [{"name": "a"}],
                    "data_tests": ["simple_model_test"],
                }
            ],
        }
        result = import_contract(doc, format="dbt")
        assert any("skipped" in w.lower() for w in result.warnings)

    def test_export_from_validate(self, simple_df):
        v = pb.Validate(data=simple_df)
        v.col_vals_not_null(columns="id")
        v.col_vals_in_set(columns="status", set=["active", "inactive"])
        v.interrogate()
        result = export_contract(v, format="dbt")
        assert result["version"] == 2
        assert len(result["models"]) == 1
        model = result["models"][0]
        assert "columns" in model

    def test_export_from_validate_extra_steps(self, simple_df):
        v = pb.Validate(data=simple_df)
        v.col_vals_ge(columns="age", value=0)
        v.col_vals_le(columns="age", value=200)
        v.interrogate()
        result = export_contract(v, format="dbt")
        assert result["version"] == 2

    def test_pb_dtype_to_dbt_type_all_branches(self):
        from pointblank.adapters._dbt import _pb_dtype_to_dbt_type
        assert _pb_dtype_to_dbt_type("Int64") == "integer"
        assert _pb_dtype_to_dbt_type("Float64") == "float"
        assert _pb_dtype_to_dbt_type("Double") == "float"
        assert _pb_dtype_to_dbt_type("Decimal") == "float"
        assert _pb_dtype_to_dbt_type("String") == "string"
        assert _pb_dtype_to_dbt_type("Utf8") == "string"
        assert _pb_dtype_to_dbt_type("Object") == "string"
        assert _pb_dtype_to_dbt_type("Boolean") == "boolean"
        assert _pb_dtype_to_dbt_type("Datetime") == "timestamp"
        assert _pb_dtype_to_dbt_type("Timestamp") == "timestamp"
        assert _pb_dtype_to_dbt_type("Date") == "date"
        assert _pb_dtype_to_dbt_type("Time") == "time"
        assert _pb_dtype_to_dbt_type("UnknownType") == "string"

    def test_apply_step_to_dbt_columns_not_null(self):
        from pointblank.adapters._dbt import _apply_step_to_dbt_columns
        columns = []
        col_map = {}
        _apply_step_to_dbt_columns("col_vals_not_null", {"columns": "a"}, col_map, columns)
        assert "not_null" in col_map["a"]["data_tests"]

    def test_apply_step_to_dbt_columns_distinct(self):
        from pointblank.adapters._dbt import _apply_step_to_dbt_columns
        columns = []
        col_map = {}
        _apply_step_to_dbt_columns("rows_distinct", {"columns_subset": "a"}, col_map, columns)
        assert "unique" in col_map["a"]["data_tests"]

    def test_apply_step_to_dbt_columns_in_set(self):
        from pointblank.adapters._dbt import _apply_step_to_dbt_columns
        columns = []
        col_map = {}
        _apply_step_to_dbt_columns(
            "col_vals_in_set", {"columns": "b", "set": ["x", "y"]}, col_map, columns
        )
        tests = col_map["b"]["data_tests"]
        assert any(isinstance(t, dict) and "accepted_values" in t for t in tests)

    def test_apply_step_no_columns(self):
        from pointblank.adapters._dbt import _apply_step_to_dbt_columns
        columns = []
        col_map = {}
        _apply_step_to_dbt_columns("col_vals_not_null", {}, col_map, columns)
        assert len(columns) == 0

    def test_apply_step_list_columns(self):
        from pointblank.adapters._dbt import _apply_step_to_dbt_columns
        columns = []
        col_map = {}
        _apply_step_to_dbt_columns(
            "col_vals_not_null", {"columns": ["a", "b"]}, col_map, columns
        )
        assert len(columns) == 2

    def test_apply_step_invalid_column_type(self):
        from pointblank.adapters._dbt import _apply_step_to_dbt_columns
        columns = []
        col_map = {}
        _apply_step_to_dbt_columns("col_vals_not_null", {"columns": 42}, col_map, columns)
        assert len(columns) == 0

    def test_apply_step_not_null_no_duplicate(self):
        from pointblank.adapters._dbt import _apply_step_to_dbt_columns
        columns = []
        col_map = {}
        _apply_step_to_dbt_columns("col_vals_not_null", {"columns": "a"}, col_map, columns)
        _apply_step_to_dbt_columns("col_vals_not_null", {"columns": "a"}, col_map, columns)
        assert col_map["a"]["data_tests"].count("not_null") == 1

    def test_apply_step_unique_no_duplicate(self):
        from pointblank.adapters._dbt import _apply_step_to_dbt_columns
        columns = []
        col_map = {}
        _apply_step_to_dbt_columns("rows_distinct", {"columns_subset": "a"}, col_map, columns)
        _apply_step_to_dbt_columns("rows_distinct", {"columns_subset": "a"}, col_map, columns)
        assert col_map["a"]["data_tests"].count("unique") == 1

    def test_extract_validate_step_kwargs_list(self):
        from pointblank.adapters._dbt import _extract_validate_step_kwargs

        class FakeStep:
            column = "status"
            values = ["a", "b"]

        kwargs = _extract_validate_step_kwargs(FakeStep())
        assert kwargs["columns"] == "status"
        assert kwargs["set"] == ["a", "b"]

    def test_extract_validate_step_kwargs_tuple(self):
        from pointblank.adapters._dbt import _extract_validate_step_kwargs

        class FakeStep:
            column = "x"
            values = ("p", "q")

        kwargs = _extract_validate_step_kwargs(FakeStep())
        assert kwargs["set"] == ["p", "q"]

    def test_extract_validate_step_kwargs_scalar(self):
        from pointblank.adapters._dbt import _extract_validate_step_kwargs

        class FakeStep:
            column = "age"
            values = 5

        kwargs = _extract_validate_step_kwargs(FakeStep())
        assert kwargs["value"] == 5

    def test_extract_validate_step_kwargs_no_attrs(self):
        from pointblank.adapters._dbt import _extract_validate_step_kwargs

        class FakeStep:
            pass

        kwargs = _extract_validate_step_kwargs(FakeStep())
        assert kwargs == {}
