
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


