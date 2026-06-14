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


