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
    """Export metadata as a Frictionless Table Schema.

    The `export_metadata()` function is the reverse of
    [`import_metadata()`](`pointblank.import_metadata`): it writes a
    [`MetadataImport`](`pointblank.MetadataImport`) out in a standard format. The only format
    currently supported is the
    [Frictionless Table Schema](https://specs.frictionlessdata.io/table-schema/), a JSON
    description of a table's fields that many data tools can read. Because the metadata can
    come from any of the formats that `import_metadata()` reads, this is also a way to convert
    between them, for example to get a Table Schema from a CSVW file or from the variable
    definitions in a Define-XML document.

    Each variable becomes a field, with its dtype mapped to a Frictionless type (`"integer"`,
    `"number"`, `"string"`, `"date"` and so on) and its label and description used as the field's
    `title` and `description`. The constraints `required`, `unique`, `minimum`, `maximum`,
    `minLength`, `maxLength`, `pattern` and `enum` are written for the variables that have them,
    and declared missing values become the field's `missingValues`. Variables that are both
    required and unique make up the schema's `primaryKey`, and the dataset label and description
    become the schema's `title` and `description`.

    Parameters
    ----------
    source
        The metadata to export. To export one dataset from a
        [`MetadataPackage`](`pointblank.MetadataPackage`), select it first (as in
        `package["DM"]`).
    destination
        A file path to write the schema to, as indented JSON. Any missing parent directories are
        created. If `None` (the default), nothing is written. The schema is returned in both cases.
    format
        The output format. Only `"frictionless"` is supported (`"table_schema"` is accepted as
        another name for it).
    **kwargs
        Options for the exporter. `include_constraints=False` leaves out the `constraints` of
        every field, giving a schema with only the field names, types, titles and missing values.

    Returns
    -------
    dict
        The Table Schema as a dictionary.

    Raises
    ------
    ValueError
        If the format isn't supported.

    Examples
    --------
    Let's import the bundled CSVW metadata for a table of weather observations and export it as a
    Frictionless Table Schema:

    ```{python}
    import json
    import pointblank as pb

    meta = pb.import_metadata(pb.load_metadata_example("weather_csvw.json"))

    schema = pb.export_metadata(meta)

    print(json.dumps(schema["fields"][:3], indent=2))
    ```

    The first three fields show how the CSVW column definitions were carried over: the titles,
    the types (a CSVW `decimal` is a Frictionless `number`), and the constraints. The remaining
    keys of the schema describe the table as a whole:

    ```{python}
    print(schema["primaryKey"])
    print(schema["title"])
    ```

    The two required and unique columns form the primary key, and the title comes from the
    dataset label in the CSVW file.

    Give a `destination=` to write the schema to a file. Here it's written to a temporary
    directory and then imported again:

    ```{python}
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "weather_schema.json"

        pb.export_metadata(meta, destination=path, include_constraints=False)

        reimported = pb.import_metadata(path)

    print(reimported.variable_names)
    print(reimported.get_variable("temperature_c").min_val)
    ```

    The file is a standalone Table Schema, so `import_metadata()` reads it back with the same
    variables. With `include_constraints=False` the range of `temperature_c` wasn't written, so
    its `min_val` is now `None`. That's useful when you want to share the structure of a table
    without the rules for its values.

    See Also
    --------
    import_metadata : Read metadata from a file.
    MetadataImport : The metadata that is exported.
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
