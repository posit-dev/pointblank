from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pointblank.metadata._types import MetadataImport

__all__ = ["export_metadata"]

# Reverse mapping from Pointblank dtypes to Frictionless types
_DTYPE_TO_FRICTIONLESS: dict[str, str] = {
    "Int8": "integer",
    "Int16": "integer",
    "Int32": "integer",
    "Int64": "integer",
    "UInt8": "integer",
    "UInt16": "integer",
    "UInt32": "integer",
    "UInt64": "integer",
    "Float32": "number",
    "Float64": "number",
    "String": "string",
    "Boolean": "boolean",
    "Date": "date",
    "Datetime": "datetime",
    "Time": "time",
    "Duration": "duration",
}


