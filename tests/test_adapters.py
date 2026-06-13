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


