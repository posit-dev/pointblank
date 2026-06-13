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

    def import_contract(self, source: Any, **kwargs: Any) -> ContractImport:
        """Import a JSON Schema document into a ContractImport.

        Parameters
        ----------
        source
            A file path (str) to a .json file, or a dict with the schema content.
        **kwargs
            Not currently used.

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
                raise FileNotFoundError(f"JSON Schema file not found: {source}")
            with open(path) as f:
                schema_doc = json.load(f)
        elif isinstance(source, dict):
            schema_doc = source
        else:
            raise TypeError(
                f"JSON Schema source must be a file path (str) or dict, got {type(source).__name__}"
            )

        return self._parse_schema(schema_doc, source_path=source_path)

    def export_contract(
        self,
        validation_or_contract: Any,
        destination: str | None = None,
        **kwargs: Any,
    ) -> str | dict[str, Any]:
        """Export a Validate or Contract to JSON Schema format.

        Parameters
        ----------
        validation_or_contract
            A `Validate` or `Contract` object.
        destination
            Optional file path to write the JSON Schema.
        **kwargs
            Not currently used.

        Returns
        -------
        dict
            The JSON Schema document as a dict.
        """
        from pointblank.contract import Contract
        from pointblank.validate import Validate

        if isinstance(validation_or_contract, Contract):
            schema_doc = self._export_from_contract(validation_or_contract)
        elif isinstance(validation_or_contract, Validate):
            schema_doc = self._export_from_validate(validation_or_contract)
        else:
            raise TypeError(
                f"Expected a Validate or Contract object, got {type(validation_or_contract).__name__}"
            )

        if destination is not None:
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(schema_doc, f, indent=2)

        return schema_doc

