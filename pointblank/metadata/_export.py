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


def export_metadata(
    source: MetadataImport,
    destination: str | Path | None = None,
    format: str = "frictionless",
    **kwargs: Any,
) -> dict[str, Any] | str:
    """Export metadata to an external standard format.

    Converts a MetadataImport object to a standards-compliant representation (e.g., Frictionless
    Table Schema) and optionally writes it to a file.

    Parameters
    ----------
    source
        The MetadataImport object to export.
    destination
        Optional file path to write the output. If `None`, returns the result as a dict (for JSON
        formats) or string.
    format
        Target format. Currently supported: `"frictionless"`.
    **kwargs
        Additional format-specific options.

    Returns
    -------
    dict | str
        The exported metadata as a dict (JSON formats) or string.

    Raises
    ------
    ValueError
        If the format is not supported.
    """
    format = format.lower().strip()

    if format in ("frictionless", "table_schema"):
        result = _export_to_frictionless(source, **kwargs)
    else:
        raise ValueError(
            f"Unsupported export format: '{format}'. Currently supported: 'frictionless'."
        )

    if destination is not None:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


def _export_to_frictionless(
    meta: MetadataImport,
    include_constraints: bool = True,
    **kwargs: Any,
) -> dict[str, Any]:
    """Export MetadataImport to Frictionless Table Schema format.

    Parameters
    ----------
    meta
        The metadata to export.
    include_constraints
        Whether to include field constraints in the output. Default is `True`.

    Returns
    -------
    dict
        A Frictionless Table Schema dict.
    """
    fields: list[dict[str, Any]] = []
    primary_key: list[str] = []

    for var in meta.variables:
        field_def: dict[str, Any] = {"name": var.name}

        # Type
        frictionless_type = _DTYPE_TO_FRICTIONLESS.get(var.dtype or "String", "string")
        field_def["type"] = frictionless_type

        # Title (label)
        if var.label:
            field_def["title"] = var.label

        # Description
        if var.description:
            field_def["description"] = var.description

        # Format
        if var.display_format:
            field_def["format"] = var.display_format

        # Constraints
        if include_constraints:
            constraints: dict[str, Any] = {}

            if var.required:
                constraints["required"] = True
            if var.unique:
                constraints["unique"] = True
            if var.min_val is not None:
                constraints["minimum"] = var.min_val
            if var.max_val is not None:
                constraints["maximum"] = var.max_val
            if var.min_length is not None:
                constraints["minLength"] = var.min_length
            if var.max_length is not None:
                constraints["maxLength"] = var.max_length
            if var.pattern is not None:
                constraints["pattern"] = var.pattern
            if var.allowed_values is not None:
                constraints["enum"] = var.allowed_values

            if constraints:
                field_def["constraints"] = constraints

        # Missing values (field-level)
        if var.missing_values:
            field_def["missingValues"] = [""] + [str(v) for v in var.missing_values]

        # Track primary key candidates (required + unique)
        if var.required and var.unique:
            primary_key.append(var.name)

        fields.append(field_def)

    # Build the Table Schema
    table_schema: dict[str, Any] = {"fields": fields}

    if primary_key:
        table_schema["primaryKey"] = primary_key[0] if len(primary_key) == 1 else primary_key

    if meta.dataset_label:
        table_schema["title"] = meta.dataset_label
    if meta.dataset_description:
        table_schema["description"] = meta.dataset_description

    return table_schema
