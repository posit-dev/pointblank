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

# Mapping from JSON Schema "format" to `col_vals_within_spec()` specs
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

    def _parse_schema(
        self, schema_doc: dict[str, Any], source_path: str | None = None
    ) -> ContractImport:
        """Parse a JSON Schema dict into a ContractImport."""
        columns: list[tuple[str, str | None]] = []
        constraints: list[MappedConstraint] = []
        warnings: list[str] = []
        metadata: dict[str, Any] = {}

        # Extract metadata
        if "title" in schema_doc:
            metadata["title"] = schema_doc["title"]
        if "description" in schema_doc:
            metadata["description"] = schema_doc["description"]

        # Detect schema version
        source_version = schema_doc.get("$schema")

        # Extract properties (the main column definitions)
        properties = schema_doc.get("properties", {})
        required_fields = set(schema_doc.get("required", []))

        total_constraints = 0

        for prop_name, prop_schema in properties.items():
            # Determine dtype from type
            prop_type = prop_schema.get("type")
            dtype = None
            if isinstance(prop_type, str):
                dtype = _JSON_SCHEMA_TYPE_MAP.get(prop_type)
            elif isinstance(prop_type, list):
                # Union type like ["string", "null"] — use the non-null type
                non_null = [t for t in prop_type if t != "null"]
                if non_null:
                    dtype = _JSON_SCHEMA_TYPE_MAP.get(non_null[0])

            columns.append((prop_name, dtype))

            # Required fields -> col_vals_not_null
            if prop_name in required_fields:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_not_null",
                        kwargs={"columns": prop_name},
                        source_description=f"required field: {prop_name}",
                    )
                )

            # minimum -> col_vals_ge
            if "minimum" in prop_schema:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_ge",
                        kwargs={"columns": prop_name, "value": prop_schema["minimum"]},
                        source_description=f"minimum: {prop_schema['minimum']}",
                    )
                )

            # maximum -> col_vals_le
            if "maximum" in prop_schema:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_le",
                        kwargs={"columns": prop_name, "value": prop_schema["maximum"]},
                        source_description=f"maximum: {prop_schema['maximum']}",
                    )
                )

            # exclusiveMinimum -> col_vals_gt
            if "exclusiveMinimum" in prop_schema:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_gt",
                        kwargs={
                            "columns": prop_name,
                            "value": prop_schema["exclusiveMinimum"],
                        },
                        source_description=f"exclusiveMinimum: {prop_schema['exclusiveMinimum']}",
                    )
                )

            # exclusiveMaximum -> col_vals_lt
            if "exclusiveMaximum" in prop_schema:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_lt",
                        kwargs={
                            "columns": prop_name,
                            "value": prop_schema["exclusiveMaximum"],
                        },
                        source_description=f"exclusiveMaximum: {prop_schema['exclusiveMaximum']}",
                    )
                )

            # enum -> col_vals_in_set
            if "enum" in prop_schema:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_in_set",
                        kwargs={"columns": prop_name, "set": prop_schema["enum"]},
                        source_description=f"enum: {prop_schema['enum']}",
                    )
                )

            # pattern -> col_vals_regex
            if "pattern" in prop_schema:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_regex",
                        kwargs={"columns": prop_name, "pattern": prop_schema["pattern"]},
                        source_description=f"pattern: {prop_schema['pattern']}",
                    )
                )

            # format -> col_vals_within_spec (where applicable)
            if "format" in prop_schema:
                fmt = prop_schema["format"]
                spec = _JSON_SCHEMA_FORMAT_MAP.get(fmt)
                if spec:
                    total_constraints += 1
                    constraints.append(
                        MappedConstraint(
                            method="col_vals_within_spec",
                            kwargs={"columns": prop_name, "spec": spec},
                            source_description=f"format: {fmt}",
                        )
                    )
                else:
                    total_constraints += 1
                    warnings.append(
                        f"Column '{prop_name}': JSON Schema format '{fmt}' has no "
                        f"Pointblank equivalent — skipped."
                    )

            # const -> col_vals_eq
            if "const" in prop_schema:
                total_constraints += 1
                constraints.append(
                    MappedConstraint(
                        method="col_vals_eq",
                        kwargs={"columns": prop_name, "value": prop_schema["const"]},
                        source_description=f"const: {prop_schema['const']}",
                    )
                )

        # Calculate coverage
        coverage = 1.0
        if total_constraints > 0:
            mapped_count = total_constraints - len(warnings)
            coverage = mapped_count / total_constraints

        return ContractImport(
            source_format="json_schema",
            source_path=source_path,
            source_version=source_version,
            columns=columns,
            constraints=constraints,
            metadata=metadata,
            warnings=warnings,
            coverage=coverage,
        )

    def _export_from_contract(self, contract: Any) -> dict[str, Any]:
        """Export a Contract object to JSON Schema."""

        schema_doc: dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "title": contract.name,
        }

        if contract.description:
            schema_doc["description"] = contract.description

        properties: dict[str, Any] = {}
        required: list[str] = []

        # From the contract's Schema
        if contract.schema is not None and contract.schema.columns is not None:
            for col_name, col_dtype in contract.schema.columns:
                prop: dict[str, Any] = {}
                if col_dtype:
                    json_type = _pb_dtype_to_json_type(str(col_dtype))
                    if json_type:
                        prop["type"] = json_type
                properties[col_name] = prop

        # From the contract's Steps
        for step in contract.steps:
            _apply_step_to_properties(step.method, step.kwargs, properties, required)

        if properties:
            schema_doc["properties"] = properties
        if required:
            schema_doc["required"] = required

        return schema_doc

    def _export_from_validate(self, validation: Any) -> dict[str, Any]:
        """Export a Validate object to JSON Schema.

        This is a best-effort export. It scans the validation steps and maps them back to JSON
        Schema constraints where possible.
        """
        schema_doc: dict[str, Any] = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
        }

        if hasattr(validation, "_tbl_name") and validation._tbl_name:
            schema_doc["title"] = validation._tbl_name

        properties: dict[str, Any] = {}
        required: list[str] = []

        # Walk through validation steps
        for step in validation.validation_info:
            method = step.assertion_type
            kwargs = _extract_step_kwargs_from_info(step)
            _apply_step_to_properties(method, kwargs, properties, required)

        if properties:
            schema_doc["properties"] = properties
        if required:
            schema_doc["required"] = required

        return schema_doc


def _pb_dtype_to_json_type(dtype: str) -> str | None:
    """Map a Pointblank/Narwhals dtype string to a JSON Schema type."""
    dtype_lower = dtype.lower()
    if "int" in dtype_lower:
        return "integer"
    if "float" in dtype_lower or "double" in dtype_lower or "decimal" in dtype_lower:
        return "number"
    if "str" in dtype_lower or "utf8" in dtype_lower or "object" in dtype_lower:
        return "string"
    if "bool" in dtype_lower:
        return "boolean"
    return None


