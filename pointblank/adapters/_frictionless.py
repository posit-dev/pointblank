from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pointblank.adapters._base import ContractAdapter, ContractImport, MappedConstraint
from pointblank.adapters._registry import register_adapter

# Mapping from Frictionless field types to Pointblank/Narwhals dtype strings
_FRICTIONLESS_TYPE_MAP: dict[str, str] = {
    "integer": "Int64",
    "number": "Float64",
    "string": "String",
    "boolean": "Boolean",
    "date": "Date",
    "datetime": "Datetime",
    "time": "Time",
    "duration": "Duration",
    "year": "Int64",
    "yearmonth": "String",
    "object": "String",
    "array": "String",
    "geopoint": "String",
    "geojson": "String",
    "any": None,
}


@register_adapter("frictionless")
class FrictionlessAdapter(ContractAdapter):
    """Adapter for Frictionless Data Table Schema.

    Supports import from Frictionless Table Schema (standalone or within a Data Package descriptor),
    and export back to Table Schema format.

    Take a look at https://specs.frictionlessdata.io/table-schema/ for the specification details.
    """

    format_name = "frictionless"
    file_extensions = [".resource.json", ".datapackage.json"]
    supports_import = True
    supports_export = True

    @staticmethod
    def detect(source: Any) -> bool:
        """Detect if the source is a Frictionless Table Schema or Data Package."""
        if isinstance(source, dict):
            # Table Schema has "fields" at top level
            if "fields" in source and isinstance(source["fields"], list):
                return True
            # Data Package has "resources"
            if "resources" in source:
                return True
            return False

        if isinstance(source, str):
            path = Path(source)
            if path.suffix == ".json" and path.exists():
                try:
                    with open(path) as f:
                        data = json.load(f)
                    return (
                        "fields" in data and isinstance(data.get("fields"), list)
                    ) or "resources" in data
                except (json.JSONDecodeError, OSError):
                    return False

        return False

