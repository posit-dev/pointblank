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


def _detect_xml_format(path: Path) -> str:
    """Detect the CDISC XML format by examining the root element and namespaces.

    Parameters
    ----------
    path
        Path to the XML file.

    Returns
    -------
    str
        Detected format: `"cdisc_define"` or `"cdisc_ct"`.

    Raises
    ------
    ValueError
        If the XML format cannot be determined.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    # Read just enough of the file to determine the format
    # Use iterparse to avoid loading the entire file
    try:
        from lxml import etree
    except ImportError:
        raise ImportError(
            "The 'lxml' package is required for XML format detection. "
            "Install it with: pip install lxml"
        ) from None

    try:
        # Parse just the root element
        context = etree.iterparse(str(path), events=("start",))
        _, root = next(context)
    except Exception as e:
        raise ValueError(f"Cannot parse XML file '{path.name}': {e}") from None

    nsmap = root.nsmap

    # Check for Define-XML namespace (def:)
    for uri in nsmap.values():
        if uri and "cdisc.org/ns/def" in uri:
            return "cdisc_define"

    # Check for NCI/EVS namespace (indicates Controlled Terminology)
    for uri in nsmap.values():
        if uri and "ncicb.nci.nih.gov" in uri:
            return "cdisc_ct"

    # Filename heuristics
    name_lower = path.name.lower()
    if "define" in name_lower:
        return "cdisc_define"
    if any(x in name_lower for x in ("sdtm", "adam", "send", "terminology", "_ct")):
        return "cdisc_ct"

    # If it has ODM namespace, treat as CT (generic ODM)
    for uri in nsmap.values():
        if uri and "cdisc.org/ns/odm" in uri:
            return "cdisc_ct"

    raise ValueError(
        f"Cannot auto-detect XML format for '{path.name}'. "
        f"Please specify format='cdisc_define' or format='cdisc_ct' explicitly."
    )


def import_metadata(
    source: str | Path | Any,
    format: str | None = None,
    **kwargs: Any,
) -> MetadataImport | MetadataPackage:
    """Import metadata from an external standard or file.

    Reads metadata definitions from statistical package files (SPSS, SAS, Stata), standards
    documents (CDISC Define-XML, Frictionless), or scientific formats (NetCDF/CF) and returns a
    structured representation that can be converted to Pointblank validation workflows.

    Parameters
    ----------
    source
        Path to a metadata file, or an object containing metadata (e.g., an xarray Dataset). For
        file paths, the format will be auto-detected from the extension if not specified.
    format
        Explicit format identifier. If None, auto-detected from the file extension. Supported
        formats: `"spss"`, `"sav"`, `"xpt"`, `"sas"`, `"stata"`, `"dta"`, `"frictionless"`,
        `"datapackage"`, `"table_schema"`, `"csvw"`, `"cdisc_define"`, `"define_xml"`, `"cdisc_ct"`.
    **kwargs
        Additional format-specific options passed to the reader.

    Returns
    -------
    MetadataImport | MetadataPackage
        A MetadataImport for single-dataset sources, or a MetadataPackage for multi-dataset sources
        (e.g., multi-domain CDISC studies).

    Raises
    ------
    ValueError
        If the format cannot be determined or is not supported.
    ImportError
        If the required optional dependency is not installed.

    Examples
    --------
    Import SPSS metadata and generate validation:

    ```python
    import pointblank as pb

    meta = pb.import_metadata("survey_data.sav")
    meta.summary()

    # Convert to a validation workflow
    validation = meta.to_validate(data=df).interrogate()
    ```

    Import SAS Transport metadata:

    ```python
    meta = pb.import_metadata("clinical_data.xpt", format="xpt")
    schema = meta.to_schema()
    ```
    """
    # Resolve path
    if isinstance(source, (str, Path)):
        path = Path(source)

        # Auto-detect format if not specified
        if format is None:
            format = _detect_format(path)

        # Normalize format name
        format = format.lower().strip()

        # Route to the appropriate reader
        if format in ("spss", "sav"):
            from pointblank.metadata._readers_stats import _read_spss_metadata

            return _read_spss_metadata(path, **kwargs)

        elif format in ("xpt", "sas"):
            from pointblank.metadata._readers_stats import _read_xpt_metadata

            return _read_xpt_metadata(path, **kwargs)

        elif format in ("stata", "dta"):
            from pointblank.metadata._readers_stats import _read_stata_metadata

            return _read_stata_metadata(path, **kwargs)

        elif format in ("frictionless", "datapackage", "table_schema"):
            from pointblank.metadata._readers_frictionless import (
                _read_frictionless_metadata,
            )

            return _read_frictionless_metadata(path, **kwargs)

        elif format == "csvw":
            from pointblank.metadata._readers_frictionless import _read_csvw_metadata

            return _read_csvw_metadata(path, **kwargs)

        elif format in ("cdisc_define", "define_xml"):
            from pointblank.metadata._readers_cdisc import _read_define_xml_metadata

            return _read_define_xml_metadata(path, **kwargs)

        elif format == "cdisc_ct":
            from pointblank.metadata._readers_cdisc import _read_cdisc_ct_metadata

            return _read_cdisc_ct_metadata(path, **kwargs)

        elif format == "cdisc_sdtm":
            from pointblank.metadata._sdtm_validate import sdtm_to_metadata

            # For SDTM, the "source" can be a domain code string or file path
            # If a domain kwarg is provided, use that; otherwise try to infer
            domain = kwargs.pop("domain", None)
            if domain is None:
                raise ValueError(
                    "format='cdisc_sdtm' requires a domain= parameter "
                    "(e.g., domain='DM', domain='AE')."
                )
            return sdtm_to_metadata(domain=domain, **kwargs)

        elif format == "cdisc_adam":
            from pointblank.metadata._adam_validate import adam_to_metadata

            # For ADaM, the "source" can be a dataset name or file path
            dataset = kwargs.pop("dataset", None)
            if dataset is None:
                raise ValueError(
                    "format='cdisc_adam' requires a dataset= parameter "
                    "(e.g., dataset='ADSL', dataset='BDS')."
                )
            return adam_to_metadata(dataset=dataset, **kwargs)

        else:
            raise ValueError(
                f"Unsupported metadata format: '{format}'. "
                f"Currently supported: 'spss', 'xpt', 'stata', 'frictionless', 'csvw', "
                f"'cdisc_define', 'cdisc_ct', 'cdisc_sdtm', 'cdisc_adam'. "
                f"Future support planned for: 'netcdf', 'ddi'."
            )
    else:
        raise TypeError(
            f"Expected a file path (str or Path), got {type(source).__name__}. "
            f"Object-based import (e.g., from xarray Datasets) is planned for a future release."
        )
