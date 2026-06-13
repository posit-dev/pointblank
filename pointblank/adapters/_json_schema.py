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


