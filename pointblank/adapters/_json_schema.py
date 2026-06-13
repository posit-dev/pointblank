from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pointblank.adapters._base import ContractAdapter, ContractImport, MappedConstraint
from pointblank.adapters._registry import register_adapter

# Mapping from JSON Schema types to Pointblank/Narwhals dtype strings
_JSON_SCHEMA_TYPE_MAP: dict[str, str] = {
    "integer": "Int64",
    "number": "Float64",
    "string": "String",
    "boolean": "Boolean",
}

# Mapping from JSON Schema "format" to col_vals_within_spec specs
_JSON_SCHEMA_FORMAT_MAP: dict[str, str] = {
    "email": "email",
    "uri": "url",
    "uri-reference": "url",
    "ipv4": "ipv4",
    "ipv6": "ipv6",
}


@register_adapter("json_schema")
class JSONSchemaAdapter(ContractAdapter):
    """Adapter for JSON Schema (Draft 2020-12 and earlier drafts).

    Supports import from JSON Schema files or dicts, and export of Pointblank validations back to
    JSON Schema format.
    """

    format_name = "json_schema"
    file_extensions = [".schema.json"]
    supports_import = True
    supports_export = True

    @staticmethod
    def detect(source: Any) -> bool:
        """Detect if the source is a JSON Schema document."""
        if isinstance(source, dict):
            return "$schema" in source or ("type" in source and "properties" in source)

        if isinstance(source, str):
            path = Path(source)
            if path.suffix == ".json" and path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                    return "$schema" in data or ("type" in data and "properties" in data)
                except (json.JSONDecodeError, OSError):
                    return False

        return False

