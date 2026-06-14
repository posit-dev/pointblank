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


