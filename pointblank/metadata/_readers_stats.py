from __future__ import annotations

from pathlib import Path
from typing import Any

from pointblank.metadata._types import (
    Codelist,
    CodelistEntry,
    MetadataImport,
    MissingValueCode,
    VariableMetadata,
)

__all__ = [
    "_read_spss_metadata",
    "_read_xpt_metadata",
    "_read_stata_metadata",
]


def _ensure_pyreadstat():
    """Check that pyreadstat is available, raise a helpful error if not."""
    try:
        import pyreadstat

        return pyreadstat
    except ImportError:
        raise ImportError(
            "The 'pyreadstat' package is required for importing metadata from "
            "SPSS, SAS, and Stata files. Install it with:\n\n"
            "  pip install pyreadstat\n\n"
            "Or install pointblank with the stats extra:\n\n"
            "  pip install pointblank[stats]"
        ) from None


def _spss_type_to_dtype(readstat_type: str, original_format: str | None) -> str:
    """Map SPSS variable type to Pointblank dtype string.

    Parameters
    ----------
    readstat_type
        The readstat type string: `"double"` or `"string"`.
    original_format
        SPSS format string (e.g., `"F8.2"`, `"A20"`, `"DATE11"`).

    Returns
    -------
    str
        Pointblank dtype string.
    """
    if readstat_type == "string":
        return "String"

    # Numeric type - check original format for date/time types
    if original_format:
        fmt_upper = original_format.upper()
        if any(d in fmt_upper for d in ("DATE", "ADATE", "EDATE", "SDATE", "JDATE")):
            return "Date"
        if "TIME" in fmt_upper or "DTIME" in fmt_upper:
            return "Time"
        if "DATETIME" in fmt_upper:
            return "Datetime"
        # Check if integer format (no decimal places or .0)
        if "." in original_format:
            parts = original_format.split(".")
            if len(parts) == 2 and parts[1] == "0":
                return "Int64"

    return "Float64"


def _sas_type_to_dtype(var_type: str, format_str: str | None) -> str:
    """Map SAS variable type to Pointblank dtype string.

    Parameters
    ----------
    var_type
        SAS variable type (`"numeric"` or `"character"`).
    format_str
        SAS format string (e.g., `"DATE9."`, `"DATETIME20."`, `"$CHAR200."`).

    Returns
    -------
    str
        Pointblank dtype string.
    """
    if var_type == "character":
        return "String"

    # Numeric type - check format
    if format_str:
        fmt_upper = format_str.upper().rstrip(".")
        if any(d in fmt_upper for d in ("DATE", "DDMMYY", "MMDDYY", "YYMMDD", "JULIAN")):
            return "Date"
        if "TIME" in fmt_upper:
            return "Time"
        if "DATETIME" in fmt_upper:
            return "Datetime"

    return "Float64"


def _stata_type_to_dtype(stata_type: str) -> str:
    """Map Stata variable type to Pointblank dtype string.

    Parameters
    ----------
    stata_type
        Stata type string (e.g., `"byte"`, `"int"`, `"long"`, `"float"`, `"double"`, `"strXX"`).

    Returns
    -------
    str
        Pointblank dtype string.
    """
    if stata_type.startswith("str"):
        return "String"
    if stata_type in ("byte", "int", "long"):
        return "Int64"
    if stata_type in ("float", "double"):
        return "Float64"
    return "String"


def _read_spss_metadata(path: Path, **kwargs: Any) -> MetadataImport:
    """Read metadata from an SPSS `.sav` file.

    Extracts variable names, labels, types, value labels, missing value codes, and display formats
    without loading the full data.

    Parameters
    ----------
    path
        Path to the `.sav` file.
    **kwargs
        Additional options (currently unused).

    Returns
    -------
    MetadataImport
        Structured metadata from the SPSS file.
    """
    pyreadstat = _ensure_pyreadstat()

    # Read metadata only (no data loaded)
    _, meta = pyreadstat.read_sav(str(path), metadataonly=True)

    variables: list[VariableMetadata] = []
    codelists: dict[str, Codelist] = {}
    missing_codes: dict[str, list[MissingValueCode]] = {}

    for i, col_name in enumerate(meta.column_names):
        # Basic info
        label = meta.column_names_to_labels.get(col_name)

        # Type info from pyreadstat
        readstat_type = meta.readstat_variable_types.get(col_name, "double")
        original_format = meta.original_variable_types.get(col_name)

        # Determine dtype
        dtype = _spss_type_to_dtype(readstat_type, original_format)

        # Value labels
        value_labels = None
        allowed_values = None
        codelist_ref = None

        if col_name in meta.variable_value_labels:
            val_labels = meta.variable_value_labels[col_name]
            if val_labels:
                value_labels = val_labels
                allowed_values = list(val_labels.keys())

                # Create a codelist for this variable
                cl_name = f"{col_name}_values"
                codelist_ref = cl_name
                codelists[cl_name] = Codelist(
                    name=cl_name,
                    label=f"Value labels for {col_name}",
                    source="SPSS .sav",
                    codes=[CodelistEntry(value=v, label=lbl) for v, lbl in val_labels.items()],
                )

        # Missing values
        missing_vals = None
        if hasattr(meta, "missing_ranges") and col_name in meta.missing_ranges:
            raw_missing = meta.missing_ranges[col_name]
            if raw_missing:
                missing_vals = []
                mv_codes = []
                for item in raw_missing:
                    if isinstance(item, dict):
                        # Range missing: {"lo": x, "hi": y}
                        lo = item.get("lo")
                        hi = item.get("hi")
                        if lo == hi:
                            missing_vals.append(lo)
                            mv_codes.append(
                                MissingValueCode(
                                    value=lo,
                                    label=f"User-defined missing ({lo})",
                                    category="user_missing",
                                )
                            )
                        else:
                            # Range — we store both endpoints
                            missing_vals.extend([lo, hi])
                            mv_codes.append(
                                MissingValueCode(
                                    value=f"{lo} to {hi}",
                                    label=f"User-defined missing range ({lo} to {hi})",
                                    category="user_missing",
                                )
                            )
                    else:
                        missing_vals.append(item)
                        mv_codes.append(
                            MissingValueCode(
                                value=item,
                                label=f"User-defined missing ({item})",
                                category="user_missing",
                            )
                        )
                if mv_codes:
                    missing_codes[col_name] = mv_codes

        # Determine max_length for string variables
        max_length = None
        if readstat_type == "string":
            # Try to get from variable_storage_width or from original format (A20 → 20)
            if hasattr(meta, "variable_storage_width"):
                width = meta.variable_storage_width.get(col_name)
                if width:
                    max_length = width
            if max_length is None and original_format:
                # Parse format like "A20" to get length
                fmt = original_format.upper()
                if fmt.startswith("A"):
                    try:
                        max_length = int(fmt[1:])
                    except ValueError:
                        pass

        # Build variable metadata
        variables.append(
            VariableMetadata(
                name=col_name,
                label=label if label else None,
                dtype=dtype,
                max_length=max_length,
                allowed_values=allowed_values,
                value_labels=value_labels,
                missing_values=missing_vals,
                codelist_ref=codelist_ref,
                display_format=original_format,
            )
        )

    return MetadataImport(
        source_format="spss",
        source_path=str(path),
        dataset_name=path.stem,
        dataset_label=getattr(meta, "file_label", None) or None,
        creation_date=getattr(meta, "creation_time", None),
        variables=variables,
        codelists=codelists,
        missing_value_codes=missing_codes,
    )


def _read_xpt_metadata(path: Path, **kwargs: Any) -> MetadataImport:
    """Read metadata from a SAS Transport (`.xpt`) file.

    Extracts variable names, labels, types, lengths, and formats.

    Parameters
    ----------
    path
        Path to the `.xpt` file.
    **kwargs
        Additional options (currently unused).

    Returns
    -------
    MetadataImport
        Structured metadata from the SAS Transport file.
    """
    pyreadstat = _ensure_pyreadstat()

    # Read metadata only
    _, meta = pyreadstat.read_xport(str(path), metadataonly=True)

    variables: list[VariableMetadata] = []

    for col_name in meta.column_names:
        # Get label
        label = meta.column_names_to_labels.get(col_name)

        # Get type info - readstat_variable_types gives "string" or "double"
        readstat_type = meta.readstat_variable_types.get(col_name, "double")
        original_format = meta.original_variable_types.get(col_name)

        # Determine dtype
        is_string = readstat_type == "string"
        if is_string:
            dtype = "String"
        else:
            dtype = _sas_type_to_dtype("numeric", original_format)

        # Variable length (significant for SAS/CDISC compliance)
        max_length = None
        if hasattr(meta, "variable_storage_width"):
            width = meta.variable_storage_width.get(col_name)
            if width and is_string:
                max_length = width

        variables.append(
            VariableMetadata(
                name=col_name,
                label=label if label else None,
                dtype=dtype,
                max_length=max_length,
                display_format=original_format,
            )
        )

    return MetadataImport(
        source_format="xpt",
        source_path=str(path),
        dataset_name=getattr(meta, "table_name", None) or path.stem.upper(),
        dataset_label=getattr(meta, "file_label", None) or None,
        variables=variables,
    )


def _read_stata_metadata(path: Path, **kwargs: Any) -> MetadataImport:
    """Read metadata from a Stata `.dta` file.

    Extracts variable names, labels, types, value labels, and formats.

    Parameters
    ----------
    path
        Path to the `.dta` file.
    **kwargs
        Additional options (currently unused).

    Returns
    -------
    MetadataImport
        Structured metadata from the Stata file.
    """
    pyreadstat = _ensure_pyreadstat()

    # Read metadata only
    _, meta = pyreadstat.read_dta(str(path), metadataonly=True)

    variables: list[VariableMetadata] = []
    codelists: dict[str, Codelist] = {}

    for col_name in meta.column_names:
        # Basic info
        label = meta.column_names_to_labels.get(col_name)

        # Type info - readstat gives "double" or "string"
        readstat_type = meta.readstat_variable_types.get(col_name, "double")
        original_format = meta.original_variable_types.get(col_name)

        if readstat_type == "string":
            dtype = "String"
        else:
            dtype = "Float64"

        # Value labels
        value_labels = None
        allowed_values = None
        codelist_ref = None

        if col_name in meta.variable_value_labels:
            val_labels = meta.variable_value_labels[col_name]
            if val_labels:
                value_labels = val_labels
                allowed_values = list(val_labels.keys())

                # Create codelist
                cl_name = f"{col_name}_values"
                codelist_ref = cl_name
                codelists[cl_name] = Codelist(
                    name=cl_name,
                    label=f"Value labels for {col_name}",
                    source="Stata .dta",
                    codes=[CodelistEntry(value=v, label=lbl) for v, lbl in val_labels.items()],
                )

        # Max length for strings - parse from original format like "%-9s"
        max_length = None
        if readstat_type == "string" and original_format:
            # Stata format like "%-9s" or "%9s"
            fmt = original_format.replace("%", "").replace("-", "").replace("s", "")
            try:
                max_length = int(fmt)
            except ValueError:
                pass

        variables.append(
            VariableMetadata(
                name=col_name,
                label=label if label else None,
                dtype=dtype,
                max_length=max_length,
                allowed_values=allowed_values,
                value_labels=value_labels,
                codelist_ref=codelist_ref,
            )
        )

    return MetadataImport(
        source_format="stata",
        source_path=str(path),
        dataset_name=path.stem,
        dataset_label=getattr(meta, "file_label", None) or None,
        variables=variables,
        codelists=codelists,
    )
