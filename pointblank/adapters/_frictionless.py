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

    def import_contract(self, source: Any, **kwargs: Any) -> ContractImport:
        """Import a Frictionless Table Schema or Data Package.

        Parameters
        ----------
        source
            A file path (str) to a JSON file, or a dict with the schema content.
        resource
            For Data Packages with multiple resources, the name or index of the resource to import.
            Defaults to the first resource.
        **kwargs
            Additional options (e.g., `resource="my_table"`).

        Returns
        -------
        ContractImport
            The import result.
        """
        source_path = None

        if isinstance(source, str):
            source_path = source
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"Frictionless schema file not found: {source}")
            with open(path) as f:
                doc = json.load(f)
        elif isinstance(source, dict):
            doc = source
        else:
            raise TypeError(
                f"Frictionless source must be a file path (str) or dict, "
                f"got {type(source).__name__}"
            )

        # If it's a Data Package, extract the Table Schema from a resource
        table_schema = self._extract_table_schema(doc, **kwargs)

        return self._parse_table_schema(table_schema, source_path=source_path)

    def export_contract(
        self,
        validation_or_contract: Any,
        destination: str | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        """Export a Validate or Contract to Frictionless Table Schema format.

        Parameters
        ----------
        validation_or_contract
            A `Validate` or `Contract` object.
        destination
            Optional file path to write the Table Schema JSON.
        **kwargs
            Not currently used.

        Returns
        -------
        dict
            The Frictionless Table Schema as a dict.
        """
        from pointblank.contract import Contract
        from pointblank.validate import Validate

        if isinstance(validation_or_contract, Contract):
            table_schema = self._export_from_contract(validation_or_contract)
        elif isinstance(validation_or_contract, Validate):
            table_schema = self._export_from_validate(validation_or_contract)
        else:
            raise TypeError(
                f"Expected a Validate or Contract object, "
                f"got {type(validation_or_contract).__name__}"
            )

        if destination is not None:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(table_schema, f, indent=2)

        return table_schema

