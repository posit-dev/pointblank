from __future__ import annotations

from typing import Any

from pointblank.adapters._base import ContractImport
from pointblank.adapters._registry import _detect_format, get_adapter


def import_contract(source: Any, *, format: str | None = None, **kwargs: Any) -> ContractImport:
    """Import a contract/schema from an external format.

    Reads an external schema definition (JSON Schema, Frictionless Table Schema, dbt schema.yml,
    Pandera schema, Pydantic model, etc.) and produces a `ContractImport` with validation steps
    mapped to Pointblank methods.

    Parameters
    ----------
    source
        The source to import from. Can be: (1) a file path (str) to a schema/contract file, (2) a
        Python dict with schema content already loaded, or (3) a Python object (e.g., a Pandera
        `DataFrameSchema` or Pydantic model class).
    format
        The format identifier (e.g., `"json_schema"`, `"frictionless"`, `"dbt"`, etc.). If `None`,
        the format is auto-detected from file extension or content.
    **kwargs
        Format-specific options passed to the adapter.

    Returns
    -------
    ContractImport
        An object containing the imported columns, constraints, and methods to generate `Validate`
        objects, `Contract` objects, Python code, or YAML.

    Raises
    ------
    ValueError
        If the format cannot be detected or no adapter is registered for it.

    Examples
    --------
    ```python
    import pointblank as pb

    # Import from JSON Schema
    result = pb.import_contract("user_profile.schema.json", format="json_schema")
    validation = result.to_validate(data=my_df)
    validation.interrogate()

    # Auto-detect format
    result = pb.import_contract("datapackage.json")

    # Import from a dict
    schema_dict = {"type": "object", "properties": {"age": {"type": "integer", "minimum": 0}}}
    result = pb.import_contract(schema_dict, format="json_schema")
    ```
    """
    if format is None:
        format = _detect_format(source)
        if format is None:
            raise ValueError(
                f"Could not auto-detect format for source: {source!r}. "
                "Please specify the format explicitly using the format= parameter."
            )

    adapter = get_adapter(format)

    if not adapter.supports_import:
        raise ValueError(f"Adapter '{format}' does not support import.")

    return adapter.import_contract(source, **kwargs)


