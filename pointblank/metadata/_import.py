from __future__ import annotations

from pathlib import Path
from typing import Any

from pointblank.metadata._types import MetadataImport, MetadataPackage

__all__ = ["import_metadata"]

# Mapping of format strings to reader functions
_FORMAT_REGISTRY: dict[str, str] = {
    "spss": "_readers_stats",
    "sav": "_readers_stats",
    "xpt": "_readers_stats",
    "sas": "_readers_stats",
    "stata": "_readers_stats",
    "dta": "_readers_stats",
    "frictionless": "_readers_frictionless",
    "datapackage": "_readers_frictionless",
    "table_schema": "_readers_frictionless",
    "csvw": "_readers_frictionless",
    "cdisc_define": "_readers_cdisc",
    "define_xml": "_readers_cdisc",
    "cdisc_ct": "_readers_cdisc",
    "cdisc_sdtm": "_sdtm_validate",
    "cdisc_adam": "_adam_validate",
}

# File extension to format mapping for auto-detection
_EXTENSION_MAP: dict[str, str] = {
    ".sav": "spss",
    ".zsav": "spss",
    ".xpt": "xpt",
    ".sas7bdat": "sas",
    ".dta": "stata",
}

# XML files that may need content-based detection
_XML_FORMATS: set[str] = {"cdisc_define", "define_xml", "cdisc_ct"}


def _detect_format(path: str | Path) -> str:
    """Detect the metadata format from a file path.

    Parameters
    ----------
    path
        Path to the metadata file.

    Returns
    -------
    str
        Detected format identifier.

    Raises
    ------
    ValueError
        If the format cannot be determined from the file extension.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in _EXTENSION_MAP:
        return _EXTENSION_MAP[suffix]

    # For JSON files, peek at the content to detect the format
    if suffix == ".json":
        return _detect_json_format(p)

    # For XML files, peek at the content to detect CDISC format
    if suffix == ".xml":
        return _detect_xml_format(p)

    raise ValueError(
        f"Cannot auto-detect metadata format from extension '{suffix}'. "
        f"Please specify the format= parameter explicitly. "
        f"Supported extensions: {sorted(_EXTENSION_MAP.keys())}, .json "
        f"(auto-detected as frictionless or csvw), and .xml (CDISC)."
    )


def _detect_json_format(path: Path) -> str:
    """Detect whether a JSON file is Frictionless or CSVW.

    Parameters
    ----------
    path
        Path to the JSON file.

    Returns
    -------
    str
        Either `"frictionless"` or `"csvw"`.

    Raises
    ------
    ValueError
        If the JSON format cannot be determined.
    """
    import json

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        with open(path) as f:
            doc = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {path} — {e}") from None

    if not isinstance(doc, dict):
        raise ValueError(f"Expected a JSON object, got {type(doc).__name__}")

    # Frictionless: has "fields" (Table Schema) or "resources" (Data Package)
    if "fields" in doc and isinstance(doc.get("fields"), list):
        return "frictionless"
    if "resources" in doc:
        return "frictionless"

    # CSVW: has "tables" (TableGroup) or "tableSchema" (Table)
    if "tables" in doc:
        return "csvw"
    if "tableSchema" in doc:
        return "csvw"
    if "url" in doc and ("dialect" in doc or "tableSchema" in doc):
        return "csvw"

    # Filename heuristics
    name_lower = path.name.lower()
    if "datapackage" in name_lower or "table-schema" in name_lower:
        return "frictionless"
    if "csv-metadata" in name_lower or "csvw" in name_lower:
        return "csvw"

    raise ValueError(
        f"Cannot auto-detect JSON format for '{path.name}'. "
        f"Please specify format='frictionless' or format='csvw' explicitly."
    )


