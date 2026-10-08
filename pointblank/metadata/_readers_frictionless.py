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


def _read_csvw_metadata(path: Path, **kwargs: Any) -> MetadataImport | MetadataPackage:
    """Read metadata from a CSVW (CSV on the Web) metadata document.

    CSVW defines how to describe CSV files using JSON-LD metadata. This reader extracts column
    definitions, datatypes, constraints, and null markers.

    Parameters
    ----------
    path
        Path to the CSVW metadata JSON-LD file.
    **kwargs
        Additional options (currently unused).

    Returns
    -------
    MetadataImport | MetadataPackage
        A `MetadataImport` for single-table CSVW, or a `MetadataPackage` for multi-table groups.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON is not a valid CSVW metadata document.

    References
    ----------
    - https://www.w3.org/TR/tabular-metadata/
    - https://www.w3.org/TR/tabular-data-primer/
    """
    if not path.exists():
        raise FileNotFoundError(f"CSVW metadata file not found: {path}")

    with open(path) as f:
        doc = json.load(f)

    # CSVW can be a Table (has "tableSchema") or a TableGroup (has "tables")
    if "tables" in doc:
        # TableGroup — multiple tables
        tables = doc["tables"]
        if len(tables) == 1:
            return _parse_csvw_table(tables[0], source_path=str(path))

        items: dict[str, MetadataImport] = {}
        for i, table in enumerate(tables):
            table_url = table.get("url", f"table_{i}")
            name = Path(table_url).stem if table_url else f"table_{i}"
            items[name] = _parse_csvw_table(table, source_path=str(path))

        return MetadataPackage(
            name=doc.get("dc:title") or doc.get("rdfs:label"),
            description=doc.get("dc:description"),
            items=items,
        )

    elif "tableSchema" in doc or "url" in doc:
        # Single Table
        return _parse_csvw_table(doc, source_path=str(path))

    else:
        raise ValueError(
            "JSON document is not a valid CSVW metadata file. "
            "Expected 'tables' (TableGroup) or 'tableSchema'/'url' (Table)."
        )


def _parse_csvw_table(
    table_doc: dict[str, Any],
    source_path: str | None = None,
) -> MetadataImport:
    """Parse a single CSVW Table definition into a `MetadataImport`."""
    variables: list[VariableMetadata] = []
    codelists: dict[str, Codelist] = {}

    table_schema = table_doc.get("tableSchema", {})
    columns = table_schema.get("columns", [])

    # Table-level null markers
    table_null = table_doc.get("null", table_schema.get("null", [""]))
    if isinstance(table_null, str):
        table_null = [table_null]

    # Primary key
    primary_key = table_schema.get("primaryKey", [])
    if isinstance(primary_key, str):
        primary_key = [primary_key]

    # Dataset name from URL or table properties
    table_url = table_doc.get("url")
    dataset_name = None
    if table_url:
        dataset_name = Path(table_url).stem

    missing_codes: dict[str, list[MissingValueCode]] = {}

    for col_def in columns:
        col_name = col_def.get("name") or col_def.get("titles")
        if isinstance(col_name, list):
            col_name = col_name[0] if col_name else None
        if not col_name:
            continue

        # Skip virtual columns (computed, not in the CSV)
        if col_def.get("virtual", False):
            continue

        # Suppress columns (columns that should be suppressed in output)
        if col_def.get("suppressOutput", False):
            continue

        # Datatype
        datatype = col_def.get("datatype")
        dtype = "String"
        min_val = None
        max_val = None
        min_length = None
        max_length = None
        pattern = None

        if datatype:
            if isinstance(datatype, str):
                dtype = _CSVW_DATATYPE_MAP.get(datatype, "String")
            elif isinstance(datatype, dict):
                base = datatype.get("base", "string")
                dtype = _CSVW_DATATYPE_MAP.get(base, "String")

                # Constraints within the datatype object
                min_val_raw = datatype.get("minimum") or datatype.get("minInclusive")
                max_val_raw = datatype.get("maximum") or datatype.get("maxInclusive")
                if datatype.get("minExclusive") is not None:
                    min_val_raw = datatype["minExclusive"]
                if datatype.get("maxExclusive") is not None:
                    max_val_raw = datatype["maxExclusive"]

                if min_val_raw is not None:
                    try:
                        min_val = float(min_val_raw)
                    except (TypeError, ValueError):
                        pass
                if max_val_raw is not None:
                    try:
                        max_val = float(max_val_raw)
                    except (TypeError, ValueError):
                        pass

                min_length = datatype.get("minLength")
                max_length = datatype.get("maxLength") or datatype.get("length")
                pattern = datatype.get("format")

        # Required
        required = col_def.get("required", False)
        if col_name in primary_key:
            required = True

        # Title and description
        title = col_def.get("titles")
        if isinstance(title, list):
            title = title[0] if title else None
        elif isinstance(title, dict):
            # Language map: {"en": "Title"}
            title = next(iter(title.values()), None)

        description = col_def.get("dc:description") or col_def.get("rdfs:comment")

        # Column-level null markers
        col_null = col_def.get("null")
        if col_null is not None:
            if isinstance(col_null, str):
                col_null = [col_null]
            null_vals = [v for v in col_null if v != ""]
        else:
            null_vals = [v for v in table_null if v != ""]

        missing_vals = null_vals if null_vals else None
        if missing_vals:
            missing_codes[col_name] = [
                MissingValueCode(
                    value=v,
                    label=f"Null marker: {v!r}",
                    category="system_missing",
                )
                for v in missing_vals
            ]

        # Separator means this is a list-valued column
        separator = col_def.get("separator")

        variables.append(
            VariableMetadata(
                name=col_name,
                label=title if title != col_name else None,
                description=description,
                dtype=dtype,
                required=required,
                unique=col_name in primary_key,
                min_val=min_val,
                max_val=max_val,
                min_length=min_length,
                max_length=max_length,
                pattern=pattern,
                missing_values=missing_vals,
            )
        )

    return MetadataImport(
        source_format="csvw",
        source_path=source_path,
        dataset_name=dataset_name,
        dataset_label=table_doc.get("dc:title") or table_schema.get("dc:title"),
        dataset_description=table_doc.get("dc:description"),
        variables=variables,
        codelists=codelists,
        missing_value_codes=missing_codes,
    )
