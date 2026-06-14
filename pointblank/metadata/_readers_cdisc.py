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


def _read_define_xml_metadata(
    path: str | Path,
    dataset: str | None = None,
    **kwargs: Any,
) -> MetadataImport | MetadataPackage:
    """Read metadata from a CDISC Define-XML file.

    Extracts ItemGroup (dataset) definitions, ItemDef (variable) definitions, CodeList definitions,
    and Where Clause conditions from Define-XML 2.0/2.1.

    Parameters
    ----------
    path
        Path to the Define-XML file.
    dataset
        If provided, return metadata only for this specific dataset/domain. If `None` and multiple
        datasets exist, returns a MetadataPackage.

    Returns
    -------
    MetadataImport | MetadataPackage
        A `MetadataImport` for a single dataset, or `MetadataPackage` for multiple.
    """
    _ensure_lxml()
    from lxml import etree

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Define-XML file not found: {path}")

    # Parse the XML
    tree = etree.parse(str(path))  # noqa: S320
    root = tree.getroot()

    # Detect Define-XML version and get appropriate namespaces
    ns, version = _detect_define_version(root)

    # Extract study-level info
    study_el = root.find(".//odm:Study", ns)
    study_oid = study_el.get("OID") if study_el is not None else None

    # Find the MetaDataVersion element
    mdv = root.find(".//odm:Study/odm:MetaDataVersion", ns)
    if mdv is None:
        # Try without Study wrapper (some exports flatten)
        mdv = root.find(".//odm:MetaDataVersion", ns)
    if mdv is None:
        raise ValueError(f"No MetaDataVersion element found in {path.name}")

    # Extract all CodeLists
    codelists = _parse_codelists(mdv, ns)

    # Extract all ItemDefs (variable definitions)
    item_defs = _parse_item_defs(mdv, ns, codelists)

    # Extract ItemGroups (datasets)
    item_groups = _parse_item_groups(mdv, ns, item_defs, codelists)

    # If a specific dataset is requested, return just that one
    if dataset is not None:
        dataset_upper = dataset.upper()
        if dataset_upper not in item_groups:
            available = sorted(item_groups.keys())
            raise KeyError(
                f"Dataset '{dataset}' not found in Define-XML. Available datasets: {available}"
            )
        meta = item_groups[dataset_upper]
        meta.source_path = str(path)
        meta.source_version = f"Define-XML {version}"
        meta.study_id = study_oid
        return meta

    # If there's only one dataset, return it directly
    if len(item_groups) == 1:
        meta = next(iter(item_groups.values()))
        meta.source_path = str(path)
        meta.source_version = f"Define-XML {version}"
        meta.study_id = study_oid
        return meta

    # Multiple datasets → MetadataPackage
    for meta in item_groups.values():
        meta.source_path = str(path)
        meta.source_version = f"Define-XML {version}"
        meta.study_id = study_oid

    return MetadataPackage(
        name=study_oid or path.stem,
        items=item_groups,
        description=f"CDISC Define-XML {version} study metadata",
        version=version,
    )


def _parse_codelists(mdv, ns: dict[str, str]) -> dict[str, Codelist]:
    """Parse `CodeList` elements from `MetaDataVersion`.

    Parameters
    ----------
    mdv
        The `MetaDataVersion` XML element.
    ns
        Namespace dictionary.

    Returns
    -------
    dict
        Mapping of `CodeList` OID to `Codelist` object.
    """
    codelists: dict[str, Codelist] = {}

    for cl_el in mdv.findall("odm:CodeList", ns):
        oid = cl_el.get("OID", "")
        name = cl_el.get("Name", oid)
        data_type = cl_el.get("DataType", "text")

        entries: list[CodelistEntry] = []
        for item in cl_el.findall("odm:CodeListItem", ns):
            value = item.get("CodedValue", "")
            # Coerce to int/float based on DataType
            if data_type in ("integer",):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass
            elif data_type in ("float", "double"):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass

            # Get the Decode (display label)
            decode_el = item.find("odm:Decode/odm:TranslatedText", ns)
            label = decode_el.text if decode_el is not None and decode_el.text else value

            # Check for NCI code (extensible attribute)
            # nci:ExtCodeID or similar
            entry = CodelistEntry(value=value, label=str(label))
            entries.append(entry)

        # Also check for EnumeratedItem (no Decode needed, value = label)
        for item in cl_el.findall("odm:EnumeratedItem", ns):
            value = item.get("CodedValue", "")
            if data_type in ("integer",):
                try:
                    value = int(value)
                except (ValueError, TypeError):
                    pass
            elif data_type in ("float", "double"):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    pass
            entries.append(CodelistEntry(value=value, label=str(value)))

        codelists[oid] = Codelist(
            name=name,
            codes=entries,
            label=name,
            source="CDISC Define-XML",
        )

    return codelists


def _parse_item_defs(
    mdv, ns: dict[str, str], codelists: dict[str, Codelist]
) -> dict[str, dict[str, Any]]:
    """Parse `ItemDef` elements into a lookup dict.

    Parameters
    ----------
    mdv
        The `MetaDataVersion` XML element.
    ns
        Namespace dictionary.
    codelists
        Already-parsed codelists for cross-referencing.

    Returns
    -------
    dict
        Mapping of `ItemDef` OID to a dict of variable properties.
    """
    item_defs: dict[str, dict[str, Any]] = {}

    for item_el in mdv.findall("odm:ItemDef", ns):
        oid = item_el.get("OID", "")
        name = item_el.get("Name", "")
        data_type = item_el.get("DataType", "text")
        length_str = item_el.get("Length")
        sig_digits_str = item_el.get("SignificantDigits")
        label = item_el.get("Comment", "")

        # Get def:Label attribute (Define-XML standard)
        if not label and "def" in ns:
            def_label = item_el.get(f"{{{ns['def']}}}Label", "")
            if def_label:
                label = def_label

        # Get the Description/TranslatedText (overrides if present)
        desc_el = item_el.find("odm:Description/odm:TranslatedText", ns)
        if desc_el is not None and desc_el.text:
            label = desc_el.text

        # Map CDISC data type to Pointblank dtype
        dtype = _CDISC_TYPE_MAP.get(data_type.lower(), "String")

        # Get CodeList reference (standard ODM or Define-XML namespace)
        cl_ref_el = item_el.find("odm:CodeListRef", ns)
        if cl_ref_el is None and "def" in ns:
            cl_ref_el = item_el.find(f"{{{ns['def']}}}CodeListRef", ns)
        codelist_oid = cl_ref_el.get("CodeListOID") if cl_ref_el is not None else None

        # Get Origin (CRF, Derived, Assigned, Protocol)
        origin = None
        origin_el = item_el.find(f"{{{ns['def']}}}Origin", ns) if "def" in ns else None
        if origin_el is None:
            # Try alternative path
            origin_el = item_el.find("def:Origin", ns)
        if origin_el is not None:
            origin = origin_el.get("Type")

        # Get computational method reference (for derived variables)
        comp_method = None
        if origin == "Derived":
            # In Define-XML 2.1, methods are linked via MethodOID
            method_ref = item_el.find(f"{{{ns['def']}}}MethodRef", ns)
            if method_ref is None:
                method_ref = item_el.find("def:MethodRef", ns)
            # Store MethodOID for later resolution if needed
            if method_ref is not None:
                comp_method = method_ref.get("MethodOID")

        item_defs[oid] = {
            "name": name,
            "label": label,
            "dtype": dtype,
            "data_type_raw": data_type,
            "length": int(length_str) if length_str else None,
            "significant_digits": int(sig_digits_str) if sig_digits_str else None,
            "codelist_oid": codelist_oid,
            "origin": origin,
            "computational_method": comp_method,
        }

    return item_defs


