from __future__ import annotations

from pathlib import Path
from typing import Any

from importlib.resources import files

from pointblank.metadata._types import MetadataImport, MetadataPackage

__all__ = ["import_metadata", "load_metadata_example"]

# Bundled metadata example files, mapped to a short description and the format that
# `import_metadata()` uses to read them. These ship with the package so the documentation
# examples (and your own experimentation) can run against real files without any external
# downloads.
_METADATA_EXAMPLES: dict[str, tuple[str, str]] = {
    "define.xml": (
        "cdisc_define",
        "CDISC Define-XML 2.0 document describing the DM and AE domains of study XYZ789.",
    ),
    "sdtm_ct.xml": (
        "cdisc_ct",
        "CDISC SDTM Controlled Terminology package with a handful of common codelists "
        "(SEX, RACE, severity, etc.).",
    ),
    "datapackage.json": (
        "frictionless",
        "Frictionless Data Package describing a 'transactions' table with typed columns "
        "and constraints.",
    ),
    "table_schema.json": (
        "table_schema",
        "Standalone Frictionless Table Schema for a sensor-readings table.",
    ),
    "weather_csvw.json": (
        "csvw",
        "W3C CSVW (CSV on the Web) metadata for a weather-observations table.",
    ),
    "dm.xpt": (
        "xpt",
        "SAS Transport (XPT) file containing a small SDTM Demographics (DM) dataset.",
    ),
}

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


def load_metadata_example(name: str) -> Path:
    """Get the path to a bundled metadata example file.

    Pointblank ships a small collection of metadata files in the formats that
    [`import_metadata()`](`pointblank.import_metadata`) reads: CDISC Define-XML, CDISC Controlled
    Terminology, a Frictionless Data Package and Table Schema, CSVW, and SAS Transport (XPT). This
    function returns the path to one of them, so you can try out metadata import (and run the
    examples in this documentation) without supplying files of your own.

    Parameters
    ----------
    name
        The file name of the example, such as `"define.xml"`. See the *Available Examples* section
        for the full list. An unknown name raises a `ValueError` that lists the valid names.

    Returns
    -------
    Path
        The path to the bundled file, which can be passed directly to `import_metadata()`.

    Available Examples
    ------------------
    | Name | Format | Description |
    |------|--------|-------------|
    | `"define.xml"` | `cdisc_define` | Define-XML 2.0 document with the DM and AE domains |
    | `"sdtm_ct.xml"` | `cdisc_ct` | SDTM Controlled Terminology with common codelists |
    | `"datapackage.json"` | `frictionless` | Frictionless Data Package for a transactions table |
    | `"table_schema.json"` | `table_schema` | Frictionless Table Schema for sensor readings |
    | `"weather_csvw.json"` | `csvw` | W3C CSVW metadata for weather observations |
    | `"dm.xpt"` | `xpt` | SAS Transport file with a small SDTM Demographics dataset |

    The XML formats need the `lxml` package and the XPT file needs `pyreadstat`. The JSON formats
    don't need any extra packages.

    Examples
    --------
    Get the path to the bundled Define-XML document:

    ```{python}
    import pointblank as pb

    define_path = pb.load_metadata_example("define.xml")

    print(define_path.name)
    print(define_path.exists())
    ```

    The result is a `pathlib.Path` to a file inside the installed package. Pass it to
    [`import_metadata()`](`pointblank.import_metadata`) to read the metadata:

    ```{python}
    package = pb.import_metadata(define_path)

    for name in package.keys():
        meta = package[name]
        print(f"{name}: {meta.dataset_label} ({len(meta)} variables)")
    ```

    The format was detected from the file's contents, and because the document describes two
    datasets the result is a [`MetadataPackage`](`pointblank.MetadataPackage`) with one entry per
    dataset.

    See Also
    --------
    import_metadata : Read metadata from a file, such as one of these examples.
    """
    if name not in _METADATA_EXAMPLES:
        available = "\n".join(
            f"- `{fname}` ({fmt}): {desc}" for fname, (fmt, desc) in _METADATA_EXAMPLES.items()
        )
        raise ValueError(
            f"The metadata example `{name}` is not valid. Choose one of the following:\n{available}"
        )

    resource = files("pointblank.data") / "metadata_examples" / name

    return Path(str(resource))


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
    except ImportError:  # pragma: no cover
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
    """Import metadata from a statistical-package file or a metadata standard.

    Many datasets come with a description of their variables: the variable labels, value labels
    and missing-value codes stored in an SPSS, Stata or SAS file, the field types and constraints in
    a Frictionless or CSVW schema, or the variable definitions and codelists in a CDISC Define-XML
    document. The `import_metadata()` function reads that description (not the data itself) into a
    [`MetadataImport`](`pointblank.MetadataImport`) object, with one
    [`VariableMetadata`](`pointblank.VariableMetadata`) per variable and any codelists as
    [`Codelist`](`pointblank.Codelist`) objects.

    The imported metadata is most useful as the starting point for validation. Its
    [`to_validate()`](`pointblank.MetadataImport.to_validate`) method turns the declared types and
    constraints into a [`Validate`](`pointblank.Validate`) plan,
    [`to_schema()`](`pointblank.MetadataImport.to_schema`) gives a
    [`Schema`](`pointblank.Schema`) of the column types, and
    [`missing_specs()`](`pointblank.MetadataImport.missing_specs`) turns declared missing-value
    codes into [`MissingSpec`](`pointblank.MissingSpec`) objects. Use
    [`export_metadata()`](`pointblank.export_metadata`) to go the other way and write metadata out
    as a Frictionless Table Schema.

    A source that describes several datasets (a Define-XML document for a whole study, a Data
    Package with several resources, or a Controlled Terminology file with several codelists) gives a
    [`MetadataPackage`](`pointblank.MetadataPackage`) instead, which holds one `MetadataImport` per
    dataset.

    Parameters
    ----------
    source
        The path to the metadata file, as a string or a `pathlib.Path`. Other kinds of objects
        raise a `TypeError`.
    format
        The format of the file. If `None` (the default), the format is detected from the file: by
        extension for `.sav`, `.zsav`, `.dta` and `.xpt` files, and by content for `.json` files
        (Frictionless or CSVW) and `.xml` files (Define-XML or Controlled Terminology). Other
        extensions need an explicit format. See the *Supported Formats* section for the values.
    **kwargs
        Options for the reader of the chosen format. The Define-XML reader takes `dataset=` (such
        as `"DM"`) to return only that dataset as a `MetadataImport`. The Controlled Terminology
        reader takes `codelist=` to return only one codelist, and the Frictionless reader takes
        `resource=` (a name or an index) to pick one resource from a Data Package. The
        `"cdisc_sdtm"` and `"cdisc_adam"` formats need `domain=` and `dataset=`, respectively, and
        also take `study_id=`.

    Returns
    -------
    MetadataImport | MetadataPackage
        A `MetadataImport` for a source that describes a single dataset, or a `MetadataPackage`
        for a source that describes several.

    Raises
    ------
    ValueError
        If the format can't be detected or isn't supported, or if a required option such as
        `domain=` is missing.
    FileNotFoundError
        If the file doesn't exist.
    ImportError
        If the format needs an optional package that isn't installed.

    Supported Formats
    -----------------
    | `format=` | Reads | Needs |
    |-----------|-------|-------|
    | `"frictionless"` (or `"datapackage"`, `"table_schema"`) | Frictionless schema (`.json`) | |
    | `"csvw"` | W3C CSV on the Web metadata (`.json`) | |
    | `"cdisc_define"` (or `"define_xml"`) | CDISC Define-XML 2.0 or 2.1 (`.xml`) | `lxml` |
    | `"cdisc_ct"` | CDISC Controlled Terminology in ODM-XML (`.xml`) | `lxml` |
    | `"spss"` (or `"sav"`) | SPSS data file (`.sav`, `.zsav`) | `pyreadstat` |
    | `"stata"` (or `"dta"`) | Stata data file (`.dta`) | `pyreadstat` |
    | `"xpt"` (or `"sas"`) | SAS Transport file (`.xpt`) | `pyreadstat` |
    | `"cdisc_sdtm"` | A built-in SDTM domain template | |
    | `"cdisc_adam"` | A built-in ADaM dataset template | |

    For SPSS, Stata and SAS files only the metadata is read, not the data. The last two formats
    don't read a file at all: `source=` isn't used (any string will do), and the template named
    by `domain=` or `dataset=` is returned, as with
    [`sdtm_to_metadata()`](`pointblank.sdtm_to_metadata`) and
    [`adam_to_metadata()`](`pointblank.adam_to_metadata`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```
    Pointblank bundles a few metadata files to experiment with, which can be found with
    [`load_metadata_example()`](`pointblank.load_metadata_example`). Let's import a Frictionless
    Data Package that describes a table of sales transactions:

    ```{python}
    import pointblank as pb

    meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))

    meta
    ```

    The format was detected from the file's contents. The metadata describes eight variables and
    one codelist (the allowed values of `category`). Each variable's metadata can be looked up by
    name:

    ```{python}
    amount = meta.get_variable("amount")

    print(amount.dtype, amount.required, amount.min_val, amount.max_val)
    ```

    The `amount` variable must be a non-null float from `0.01` to `99999.99`. Constraints like these
    can be checked against a table with
    [`to_validate()`](`pointblank.MetadataImport.to_validate`), which builds a validation plan from
    the metadata:

    ```{python}
    import datetime
    import polars as pl

    transactions = pl.DataFrame(
        {
            "transaction_id": ["TXN-00001", "TXN-00002", "TXN-00003", "TXN-00004"],
            "customer_id": ["CUST-101", "CUST-102", "CUST-101", "CUST-103"],
            "amount": [25.50, 1200.00, 0.00, 89.99],
            "quantity": [1, 3, 2, 1],
            "category": ["food", "electronics", "home", "toys"],
            "sale_date": [datetime.date(2024, 5, d) for d in (1, 1, 2, 3)],
            "discount_pct": [0.0, 10.0, 0.0, 5.0],
            "email": ["a@example.com", "b@example.com", "c@example.com", "not-an-email"],
        }
    )

    meta.to_validate(data=transactions).interrogate()
    ```

    The table has the declared column types, so the schema check in step 1 passes. Three
    later steps find a failing row each: the `amount` of `0.00` is below the minimum, `"toys"`
    isn't one of the allowed categories, and `"not-an-email"` doesn't match the pattern given for
    `email`.

    A Define-XML document usually describes all the datasets in a study, so importing one gives a
    [`MetadataPackage`](`pointblank.MetadataPackage`):

    ```{python}
    package = pb.import_metadata(pb.load_metadata_example("define.xml"))

    package
    ```

    The package holds the Demographics (`DM`) and Adverse Events (`AE`) datasets, and
    `package["DM"]` gives the `MetadataImport` for one of them. If only one dataset is needed, the
    `dataset=` option of the Define-XML reader returns it directly:

    ```{python}
    dm = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="DM")

    print(dm.dataset_label)
    print(dm.variable_names)
    ```

    Options like `dataset=` are passed through to the reader for the file's format. The
    *Supported Formats* section lists the formats, and the `**kwargs` parameter describes the
    options that each reader accepts.

    See Also
    --------
    load_metadata_example : Get the path to a bundled metadata file.
    export_metadata : Export metadata as a Frictionless Table Schema.
    MetadataImport : The metadata for a single dataset.
    MetadataPackage : The metadata for several datasets.
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
