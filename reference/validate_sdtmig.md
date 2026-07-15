## validate_sdtmig()


Validate SDTM datasets against the SDTMIG rule catalog and return a conformance report.


Usage

``` python
validate_sdtmig(
    datasets, version="3-4", ct_packages=None, define_xml=None, study_id=None
)
```


Runs the bundled SDTMIG 3.4 rule catalog (426 rules) against the provided SDTM domain datasets using Pointblank's built-in conformance engine. No external tools, subprocesses, network calls, or CDISC CORE installation are required.

The catalog covers seven rule types:

- **RECORD_CHECK** -- per-row value checks (controlled terminology, ISO 8601 dates, ranges, uniqueness constraints). These rules produce row-level findings accessible via [findings_df()](ConformanceReport.md#pointblank.ConformanceReport.findings_df) and [get_findings_table()](ConformanceReport.md#pointblank.ConformanceReport.get_findings_table).
- **VARIABLE_METADATA_CHECK** -- variable presence and ordering (e.g., USUBJID must appear before domain-specific variables).
- **DATASET_METADATA_CHECK** -- dataset-level attributes (sort keys, required sort order).
- **DATASET_CONTENTS_CHECK** -- dataset-level value constraints (e.g., all rows in a domain must share the same STUDYID).
- **DOMAIN_PRESENCE_CHECK** -- required or prohibited domain presence (e.g., DM must be present, RELREC must not appear in an SDTM-only package).
- **DEFINE_ITEM_METADATA_CHECK** -- variable declarations in the Define-XML (activated only when `define_xml` is supplied).
- **DEFINE_CODELIST_CHECK** -- codelist declarations in the Define-XML (activated only when `define_xml` is supplied).


## Controlled Terminology

By default the most recent bundled CT package (`sdtm-ct-2024-09-27`) is used. Codelist checks are case-insensitive: a value of `"beats/min"` matches a term `"BEATS/MIN"`. SAS/XPT missing values (empty strings `""`) are treated as null and skipped, so they do not generate false positives for codelist or format rules.


## Supp- And Relrec Handling

Supplemental Qualifiers (`SUPP--`) datasets use `RDOMAIN` instead of `DOMAIN` and have a fixed non-standard structure, so they are automatically excluded from catch-all rules (rules with no explicit domain list). RELREC is similarly excluded.


## Parameters


`datasets: dict`  
Mapping of SDTM domain name to a DataFrame. Keys are matched case-insensitively (`"DM"` and `"dm"` are equivalent). Accepts Polars, pandas, or any narwhals-compatible DataFrame. Include all domains relevant to your submission; rules that require a domain not in the mapping are marked `not_applicable`.

`version: str = ``"3-4"`  
SDTMIG version string. Accepts either dot or hyphen notation (`"3.4"` or `"3-4"`). Currently only `"3-4"` has a bundled catalog.

`ct_packages: list[str] | None = None`  
One or more CT package slugs to load (e.g., `["sdtm-ct-2024-09-27"]`). When `None` (the default) the most recent bundled package is used automatically.

`define_xml: Any = None`  
Optional Define-XML metadata, supplied as a file path (`str` or `pathlib.Path`) or a pre-parsed [MetadataPackage](MetadataPackage.md#pointblank.MetadataPackage) object. When provided, `DEFINE_ITEM_METADATA_CHECK` and `DEFINE_CODELIST_CHECK` rules become active; without it they are marked `not_applicable`.

`study_id: str | None = None`  
Optional study identifier (e.g., `"CDISCPILOT01"`) shown in the report header.


## Returns


`ConformanceReport`  
A built-in engine report ([is_rules](ConformanceReport.md#pointblank.ConformanceReport.is_rules) is `True`). In Jupyter and Quarto notebooks the object renders automatically as the rule-level summary table. Call [get_tabular_report()](Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report) for the `GT` object, [get_findings_table()](ConformanceReport.md#pointblank.ConformanceReport.get_findings_table) for a record-level drill-down, or [findings_df()](ConformanceReport.md#pointblank.ConformanceReport.findings_df) for a Polars DataFrame of failing rows.


## Examples

Validate a study from in-memory Polars DataFrames:

``` python
import pointblank as pb

report = pb.validate_sdtmig({"DM": dm, "AE": ae, "LB": lb})
report  # renders the rule summary table in a notebook
```

Drill down to the individual failing records:

``` python
report.get_findings_table()  # styled record-level table
report.findings_df()         # Polars DataFrame for programmatic use
```

Load from XPT files using pyreadstat:

``` python
import pyreadstat, polars as pl

def load(path):
    df, _ = pyreadstat.read_xport(path)
    return pl.from_pandas(df)

report = pb.validate_sdtmig({
    "DM": load("sdtm/dm.xpt"),
    "AE": load("sdtm/ae.xpt"),
    "LB": load("sdtm/lb.xpt"),
}, study_id="STUDY001")
```
