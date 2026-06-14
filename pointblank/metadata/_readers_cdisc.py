from __future__ import annotations

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
    "_read_define_xml_metadata",
    "_read_cdisc_ct_metadata",
]

# Define-XML namespaces (supports 2.0 and 2.1)
_DEFINE_NS_20 = {
    "odm": "http://www.cdisc.org/ns/odm/v1.3",
    "def": "http://www.cdisc.org/ns/def/v2.0",
    "xlink": "http://www.w3.org/1999/xlink",
    "arm": "http://www.cdisc.org/ns/arm/v1.0",
}

_DEFINE_NS_21 = {
    "odm": "http://www.cdisc.org/ns/odm/v1.3",
    "def": "http://www.cdisc.org/ns/def/v2.1",
    "xlink": "http://www.w3.org/1999/xlink",
    "arm": "http://www.cdisc.org/ns/arm/v1.0",
}

# NCI/EVS namespace for Controlled Terminology
_CT_NS = {
    "odm": "http://www.cdisc.org/ns/odm/v1.3",
    "nciodm": "http://ncicb.nci.nih.gov/xml/odm/EVS/CDISC",
    "xlink": "http://www.w3.org/1999/xlink",
}

# CDISC data type to Pointblank dtype mapping
_CDISC_TYPE_MAP: dict[str, str] = {
    "text": "String",
    "integer": "Int64",
    "float": "Float64",
    "double": "Float64",
    "date": "Date",
    "time": "String",
    "datetime": "Datetime",
    "partialDate": "String",
    "partialTime": "String",
    "partialDatetime": "String",
    "durationDatetime": "String",
    "intervalDatetime": "String",
    "incompleteDatetime": "String",
    "incompleteDate": "String",
    "incompleteTime": "String",
    "URI": "String",
    "boolean": "Boolean",
}


def _ensure_lxml() -> None:
    """Check that lxml is available, raise helpful error if not."""
    try:
        import lxml.etree  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'lxml' package is required for CDISC XML parsing. "
            "Install it with: pip install lxml"
        ) from None


def _detect_define_version(root) -> tuple[dict[str, str], str]:
    """Detect the Define-XML version from the root element.

    Parameters
    ----------
    root
        The lxml root element.

    Returns
    -------
    tuple
        (namespace_dict, version_string)
    """
    # Check namespace declarations on root
    nsmap = root.nsmap

    # Look for def namespace version
    for prefix, uri in nsmap.items():
        if "def/v2.1" in uri:
            return _DEFINE_NS_21, "2.1"
        if "def/v2.0" in uri:
            return _DEFINE_NS_20, "2.0"

    # Fallback: check for DefineVersion attribute
    define_version = root.get("def:DefineVersion") or root.get("DefineVersion")
    if define_version:
        if define_version.startswith("2.1"):
            return _DEFINE_NS_21, "2.1"
        return _DEFINE_NS_20, "2.0"

    # Default to 2.0
    return _DEFINE_NS_20, "2.0"


