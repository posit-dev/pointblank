from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pointblank.metadata._types import (
    Codelist,
    CodelistEntry,
    MetadataImport,
    MetadataPackage,
    MissingValueCode,
    VariableMetadata,
)

__all__ = [
    "_read_frictionless_metadata",
    "_read_csvw_metadata",
]

# Mapping from Frictionless field types to Pointblank dtype strings
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
    "any": "String",
}

# Mapping from CSVW datatype base values to Pointblank dtype strings
_CSVW_DATATYPE_MAP: dict[str, str] = {
    "integer": "Int64",
    "int": "Int64",
    "long": "Int64",
    "short": "Int64",
    "byte": "Int64",
    "nonNegativeInteger": "Int64",
    "positiveInteger": "Int64",
    "unsignedInt": "Int64",
    "unsignedLong": "Int64",
    "unsignedShort": "Int64",
    "float": "Float64",
    "double": "Float64",
    "decimal": "Float64",
    "number": "Float64",
    "string": "String",
    "normalizedString": "String",
    "token": "String",
    "anyURI": "String",
    "boolean": "Boolean",
    "date": "Date",
    "dateTime": "Datetime",
    "datetime": "Datetime",
    "time": "Time",
    "duration": "Duration",
    "gDay": "String",
    "gMonth": "String",
    "gYear": "Int64",
    "gYearMonth": "String",
    "gMonthDay": "String",
    "hexBinary": "String",
    "base64Binary": "String",
    "anyAtomicType": "String",
    "json": "String",
    "xml": "String",
    "html": "String",
}


def _read_frictionless_metadata(
    path: Path,
    resource: str | int | None = None,
    **kwargs: Any,
) -> MetadataImport | MetadataPackage:
    """Read metadata from a Frictionless Table Schema or Data Package.

    Supports both standalone Table Schema files and full Data Package descriptors
    (`datapackage.json`). For Data Packages with multiple resources, returns a `MetadataPackage`.

    Parameters
    ----------
    path
        Path to the JSON file (Table Schema or Data Package descriptor).
    resource
        For Data Packages: name or index of a specific resource to import.
        If None and the package has multiple resources, returns a `MetadataPackage`.
    **kwargs
        Additional options (currently unused).

    Returns
    -------
    MetadataImport | MetadataPackage
        A `MetadataImport` for single-resource files or when a specific resource
        is selected, or a `MetadataPackage` for multi-resource packages.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON is not a valid Frictionless schema or package.
    """
    if not path.exists():
        raise FileNotFoundError(f"Frictionless schema file not found: {path}")

    with open(path) as f:
        doc = json.load(f)

    # Determine if this is a Table Schema or a Data Package
    if "fields" in doc and isinstance(doc["fields"], list):
        # Standalone Table Schema
        return _parse_frictionless_table_schema(doc, source_path=str(path))

    elif "resources" in doc:
        # Data Package
        resources = doc["resources"]
        if not resources:
            raise ValueError("Data Package has no resources.")

        # If a specific resource is requested, return a single MetadataImport
        if resource is not None:
            schema = _extract_resource_schema(resources, resource)
            resource_name = (
                resources[resource].get("name", f"resource_{resource}")
                if isinstance(resource, int)
                else resource
            )
            meta = _parse_frictionless_table_schema(
                schema, source_path=str(path), dataset_name=resource_name
            )
            meta.dataset_description = doc.get("description")
            return meta

        # Single resource → return MetadataImport
        if len(resources) == 1:
            res = resources[0]
            schema = res.get("schema", {})
            if "fields" not in schema:
                raise ValueError("Resource has no 'schema.fields'.")
            meta = _parse_frictionless_table_schema(
                schema,
                source_path=str(path),
                dataset_name=res.get("name", path.stem),
            )
            meta.dataset_description = res.get("description") or doc.get("description")
            return meta

        # Multiple resources → return MetadataPackage
        items: dict[str, MetadataImport] = {}
        for i, res in enumerate(resources):
            res_name = res.get("name", f"resource_{i}")
            schema = res.get("schema", {})
            if "fields" in schema:
                meta = _parse_frictionless_table_schema(
                    schema,
                    source_path=str(path),
                    dataset_name=res_name,
                )
                meta.dataset_description = res.get("description")
                items[res_name] = meta

        return MetadataPackage(
            name=doc.get("name"),
            description=doc.get("description"),
            version=doc.get("version"),
            items=items,
        )

    else:
        raise ValueError(
            "JSON document is neither a Frictionless Table Schema (no 'fields') "
            "nor a Data Package (no 'resources')."
        )


def _extract_resource_schema(
    resources: list[dict[str, Any]], resource_key: str | int
) -> dict[str, Any]:
    """Extract the schema from a specific resource in a Data Package."""
    if isinstance(resource_key, int):
        if resource_key >= len(resources):
            raise IndexError(
                f"Resource index {resource_key} out of range "
                f"(package has {len(resources)} resources)."
            )
        res = resources[resource_key]
    elif isinstance(resource_key, str):
        res = None
        for r in resources:
            if r.get("name") == resource_key:
                res = r
                break
        if res is None:
            available = [r.get("name", f"<index {i}>") for i, r in enumerate(resources)]
            raise ValueError(f"Resource '{resource_key}' not found. Available: {available}")
    else:
        raise TypeError(f"resource must be str or int, got {type(resource_key).__name__}")

    schema = res.get("schema", {})
    if "fields" not in schema:
        raise ValueError(f"Resource has no 'schema.fields'. Got keys: {list(schema.keys())}")
    return schema


def _parse_frictionless_table_schema(
    table_schema: dict[str, Any],
    source_path: str | None = None,
    dataset_name: str | None = None,
) -> MetadataImport:
    """Parse a Frictionless Table Schema dict into a `MetadataImport`.

    Extracts field definitions, constraints, primary keys, foreign keys, and missing value
    specifications.
    """
    variables: list[VariableMetadata] = []
    codelists: dict[str, Codelist] = {}
    missing_codes: dict[str, list[MissingValueCode]] = {}

    fields = table_schema.get("fields", [])
    primary_key = table_schema.get("primaryKey", [])
    if isinstance(primary_key, str):
        primary_key = [primary_key]

    # Package-level missing values (apply to all fields unless overridden)
    package_missing = table_schema.get("missingValues", [""])

    for field_def in fields:
        field_name = field_def.get("name", "")
        field_type = field_def.get("type", "any")
        field_format = field_def.get("format")
        field_description = field_def.get("description")
        field_title = field_def.get("title")
        field_rdf_type = field_def.get("rdfType")

        dtype = _FRICTIONLESS_TYPE_MAP.get(field_type, "String")

        # Parse constraints
        constraints = field_def.get("constraints", {})

        required = constraints.get("required", False)
        unique = constraints.get("unique", False)
        min_val = constraints.get("minimum")
        max_val = constraints.get("maximum")
        min_length = constraints.get("minLength")
        max_length = constraints.get("maxLength")
        pattern = constraints.get("pattern")
        enum = constraints.get("enum")

        # Primary key columns are implicitly required and unique
        if field_name in primary_key:
            required = True
            unique = True

        # Create codelist from enum values
        codelist_ref = None
        if enum:
            cl_name = f"{field_name}_enum"
            codelist_ref = cl_name
            codelists[cl_name] = Codelist(
                name=cl_name,
                label=f"Allowed values for {field_name}",
                source="Frictionless Table Schema",
                codes=[CodelistEntry(value=v, label=str(v)) for v in enum],
            )

        # Field-level missing values
        field_missing = field_def.get("missingValues")
        missing_vals = None
        if field_missing is not None:
            # Field-level overrides package-level
            missing_vals = [v for v in field_missing if v != ""]
        elif package_missing and any(v != "" for v in package_missing):
            missing_vals = [v for v in package_missing if v != ""]

        if missing_vals:
            missing_codes[field_name] = [
                MissingValueCode(
                    value=v,
                    label=f"Missing value marker: {v!r}",
                    category="user_missing",
                )
                for v in missing_vals
            ]

        variables.append(
            VariableMetadata(
                name=field_name,
                label=field_title,
                description=field_description,
                dtype=dtype,
                required=required,
                unique=unique,
                min_val=float(min_val) if min_val is not None else None,
                max_val=float(max_val) if max_val is not None else None,
                min_length=min_length,
                max_length=max_length,
                pattern=pattern,
                allowed_values=enum,
                codelist_ref=codelist_ref,
                missing_values=missing_vals,
                display_format=field_format,
            )
        )

    return MetadataImport(
        source_format="frictionless",
        source_path=source_path,
        dataset_name=dataset_name,
        dataset_label=table_schema.get("title"),
        dataset_description=table_schema.get("description"),
        variables=variables,
        codelists=codelists,
        missing_value_codes=missing_codes,
    )


