## ConformanceReport


The result of a CDISC conformance validation run.


Usage

``` python
ConformanceReport(
    validations=dict(),
    package=None,
    agency=None,
    core=None,
    native_result=None
)
```


A [ConformanceReport](ConformanceReport.md#pointblank.ConformanceReport) is returned by <a href="validate_sdtmig.html#pointblank.validate_sdtmig" class="gdls-link"><code>validate_sdtmig()</code></a> and <a href="SubmissionPackage.html#pointblank.SubmissionPackage.validate_conformance" class="gdls-link"><code>SubmissionPackage.validate_conformance()</code></a>. It exists in one of two forms depending on the engine used:

- **Built-in rules engine** ([is_rules](ConformanceReport.md#pointblank.ConformanceReport.is_rules) is `True`) -- produced by Pointblank's SDTMIG rule catalog. Each rule is evaluated against the supplied datasets and receives one of five statuses: `"pass"`, `"fail"`, `"error"`, `"not_applicable"`, or `"not_supported"`. Row-level findings (the individual failing records) are collected for RECORD_CHECK rules and accessible via <a href="ConformanceReport.html#pointblank.ConformanceReport.findings_df" class="gdls-link"><code>findings_df()</code></a> and <a href="ConformanceReport.html#pointblank.ConformanceReport.get_findings_table" class="gdls-link"><code>get_findings_table()</code></a>.
- **CDISC CORE** ([is_core](ConformanceReport.md#pointblank.ConformanceReport.is_core) is `True`) -- produced by the external CDISC CORE command-line engine. Rule-keyed findings and run provenance are exposed via [findings()](ConformanceReport.md#pointblank.ConformanceReport.findings) and [rules()](ConformanceReport.md#pointblank.ConformanceReport.rules).

In a Jupyter or Quarto notebook the report renders automatically as a color-coded rule summary table (calling `_repr_html_()` is equivalent to `get_tabular_report()._repr_html_()`).


## Parameters


`validations: dict[str, Validate] = dict()`    
Reserved for legacy use; not populated by the built-in engine.

`package: SubmissionPackage | None = None`  
The [SubmissionPackage](SubmissionPackage.md#pointblank.SubmissionPackage) the report was produced from, if any.

`agency: str | None = None`  
The agency rule-set selector used for the run (`None` for CDISC base rules).

`core: ParsedCoreReport | None = None`  
The parsed CDISC CORE report (CORE form only). `None` for built-in engine reports.

`native_result: NativeConformanceResult | None = None`  
The `NativeConformanceResult` produced by the rules engine (native form only).


## Attributes

| Name | Description |
|----|----|
| [is_core](#is_core) | Whether this report wraps CDISC CORE engine results (vs. built-in engine results). |
| [is_rules](#is_rules) | Whether this report was produced by Pointblank's built-in rule-based conformance engine. |
| [n_datasets](#n_datasets) | Number of datasets validated. |

------------------------------------------------------------------------


#### is_core


Whether this report wraps CDISC CORE engine results (vs. built-in engine results).


`is_core: bool`


------------------------------------------------------------------------


#### is_rules


Whether this report was produced by Pointblank's built-in rule-based conformance engine.


`is_rules: bool`


------------------------------------------------------------------------


#### n_datasets


Number of datasets validated.


`n_datasets: int`


## Methods

| Name | Description |
|----|----|
| [all_passed()](#all_passed) | Whether the run reported no conformance failures. |
| [findings()](#findings) | Return the row-level findings. |
| [findings_df()](#findings_df) | Return all row-level findings as a Polars DataFrame. |
| [from_core_report()](#from_core_report) | Build a CORE-backed [ConformanceReport](ConformanceReport.md#pointblank.ConformanceReport) from a CDISC CORE JSON report. |
| [get_findings_table()](#get_findings_table) | Build a record-level findings table as a styled Great Tables object. |
| [get_tabular_report()](#get_tabular_report) | Build a rule-level conformance summary table as a styled Great Tables object. |
| [get_validation()](#get_validation) | Get the [Validate](Validate.md#pointblank.Validate) object for a single dataset (case-insensitive). |
| [issues()](#issues) | Return the conformance issues found. |
| [rules()](#rules) | Return the per-rule run results. |
| [summary()](#summary) | Return a summary of the conformance run. |
| [to_excel()](#to_excel) | Save the conformance report as an Excel workbook. |
| [to_json()](#to_json) | Save the conformance report as a JSON file. |

------------------------------------------------------------------------


#### all_passed()


Whether the run reported no conformance failures.


Usage

``` python
all_passed()
```


For built-in engine reports, this is `True` when every check in every dataset passed with no failing test units. For CORE reports, this is `True` when no rule reported an issue or execution error.


------------------------------------------------------------------------


#### findings()


Return the row-level findings.


Usage

``` python
findings()
```


For CORE reports, returns `CoreFinding` objects from CORE's `Issue_Details`. For built-in engine reports, returns `NativeRowFinding` objects. For Validate-based reports, returns an empty list.


------------------------------------------------------------------------


#### findings_df()


Return all row-level findings as a Polars DataFrame.


Usage

``` python
findings_df()
```


Each row represents one failing record captured during the conformance run. Use this method for programmatic analysis (filtering by rule, grouping by subject, exporting to CSV, or joining back to the source datasets to investigate root causes).

Only `RECORD_CHECK` and `DATASET_CONTENTS_CHECK` rules produce row-level findings; rules that check metadata or domain presence (e.g., `VARIABLE_METADATA_CHECK`, `DOMAIN_PRESENCE_CHECK`) report a finding count in [get_tabular_report()](Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report) but do not appear here. To see the visual findings table call [get_findings_table()](ConformanceReport.md#pointblank.ConformanceReport.get_findings_table) instead.

Findings are capped at **100 rows per rule** to bound memory use on large datasets. The `n_issues` value shown in [get_tabular_report()](Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report) always reflects the true total count for a rule, even when more than 100 records failed.


##### Returns


`polars.DataFrame`  
One row per captured finding with the following columns:

- `rule_id`: CDISC CORE rule identifier (e.g., `"SDTM-007"`).
- `dataset`: The SDTM domain the failing record belongs to (e.g., `"AE"`).
- `row_index`: 0-based row position of the failing record in the source dataset.
- `usubjid`: Unique Subject Identifier from the `"USUBJID"` column, if present.
- `checked_column`: The specific variable that violated the rule (e.g., `"SEX"`).
- `checked_value`: The actual value of `checked_column` in that row.
- `description`: Human-readable rule description. Derived first from the rule's operations; falls back to the conditions tree for rules with no explicit operations (e.g., range checks like `AGE < 0`).
- `checked_value`: The actual value of `checked_column` in that row.
- `description`: Human-readable rule description.

Returns an empty DataFrame (with the same schema) when all rules pass.


##### Raises


`TypeError`  
If called on a CDISC CORE-backed report. Use [findings()](ConformanceReport.md#pointblank.ConformanceReport.findings) instead, which returns a list of `CoreFinding` objects.


------------------------------------------------------------------------


#### from_core_report()


Build a CORE-backed [ConformanceReport](ConformanceReport.md#pointblank.ConformanceReport) from a CDISC CORE JSON report.


Usage

``` python
from_core_report(report, package=None, agency=None)
```


##### Parameters


`report: dict | ParsedCoreReport`  
Either a raw CORE JSON report (`dict`, as produced by `core validate -of JSON`) or an already-parsed [`ParsedCoreReport`](%60pointblank.metadata._cdisc_core.ParsedCoreReport%60).

`package: SubmissionPackage | None = None`  
The [SubmissionPackage](SubmissionPackage.md#pointblank.SubmissionPackage) the run was produced from, if any.

`agency: str | None = None`  
The agency rule-set selector used for the run.


##### Returns


`ConformanceReport`  
A report in CORE form ([is_core](ConformanceReport.md#pointblank.ConformanceReport.is_core) is `True`).


------------------------------------------------------------------------


#### get_findings_table()


Build a record-level findings table as a styled Great Tables object.


Usage

``` python
get_findings_table()
```


Returns one row per failing record captured by Pointblank's built-in rules engine. This is the drill-down companion to [get_tabular_report()](Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report): where the tabular report shows one row per rule with an aggregate issue count, the findings table shows the individual offending records so reviewers can trace violations back to specific subjects and variables.


##### Table Layout

The table has two column spanners:

- **Rule**: `Domain` and `Description` identify which rule fired and in which domain.

- **Finding**: `USUBJID`, `Column`, `Row`, and `Value` identify the specific record.

  - `USUBJID`: the unique subject identifier (e.g., `"CDISCPILOT01-01-001"`).
  - `Column`: the variable that violated the rule (e.g., `"SEX"`).
  - `Row`: 1-based row number of the failing record in the source domain dataset.
  - `Value`: the actual value found in `Column` for that row.

The header shows the standard and version (e.g., `SDTMIG 3-4`) alongside a breakdown of how many rules passed, failed, and were not applicable across the full run.

A narrow red bar on the left edge of each row marks it as a failure, consistent with the color coding in [get_tabular_report()](Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report).


##### Findings Cap

At most 100 findings per rule are shown. When a rule has more than 100 failing records the table shows the first 100; the true total is always visible in [get_tabular_report()](Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report).


##### Returns


`GT`  
A styled `great_tables.GT` object. Renders automatically in Jupyter and Quarto notebooks.


##### Raises


`TypeError`  
If called on a CDISC CORE-backed report. The findings table is only available for built-in engine results.

`ValueError`  
If there are no row-level findings to display (i.e., all applicable rules passed).


------------------------------------------------------------------------


#### get_tabular_report()


Build a rule-level conformance summary table as a styled Great Tables object.


Usage

``` python
get_tabular_report()
```


Returns one row per rule in the catalog, summarizing whether each rule passed, failed, was not applicable, or could not be evaluated. This is the high-level overview; use [get_findings_table()](ConformanceReport.md#pointblank.ConformanceReport.get_findings_table) or [findings_df()](ConformanceReport.md#pointblank.ConformanceReport.findings_df) to drill into the individual failing records.


##### Table Layout

Each row contains:

- A colored status bar on the left edge: green for pass, red for fail, amber for error, and grey for not-applicable or not-supported.
- `Rule` -- CDISC CORE rule identifier (e.g., `"SDTM-007"`).
- `Domain` -- The SDTM domain(s) the rule targets. Rules that apply to every domain show a comma-separated list; rules targeting all SUPP- datasets show `"SUPP--"`.
- `Type` -- The rule category: `Record`, `Variable`, `Metadata`, `Domain`, `Dataset`, `Define`, or [Codelist](Codelist.md#pointblank.Codelist).
- `Issues` -- Count of failing records or dataset-level violations. Shown in bold red when non-zero. This count always reflects the true total, even when the findings table caps display at 100 rows per rule.
- `Description` -- Human-readable explanation of what the rule checks.

Rows are sorted by severity: failing rules appear first, followed by errors, passing rules, not-applicable rules, and unsupported rule types.

The table header shows `"CDISC Conformance"` with a `PASS` or `FAIL` badge, and a subtitle line with the standard, version, and a count breakdown (e.g., `SDTMIG 3-4 · 410 passed · 4 failed · 12 n/a`).


##### Returns


`GT`  
A styled `great_tables.GT` object set in IBM Plex Sans / IBM Plex Mono. Renders automatically in Jupyter and Quarto notebooks; call `._repr_html_()` to get the HTML string directly. This is the same object produced by `_repr_html_()`.


##### Raises


`TypeError`  
If called on a CDISC CORE-backed report. The tabular report is only available for built-in engine results.


------------------------------------------------------------------------


#### get_validation()


Get the [Validate](Validate.md#pointblank.Validate) object for a single dataset (case-insensitive).


Usage

``` python
get_validation(name)
```


------------------------------------------------------------------------


#### issues()


Return the conformance issues found.


Usage

``` python
issues(severity=None, status=None)
```


##### Parameters


`severity: str | None = None`  
(Built-in engine reports only.) Optional severity filter: `"warning"`, `"error"`, or `"critical"`. Requires thresholds to have been set on the run. If `None`, all steps with failing test units are returned.

`status: str | None = None`  
(CORE reports only.) Optional rule-status filter, e.g. `"ISSUE REPORTED"` or `"EXECUTION ERROR"`. If `None`, all reported issues are returned.


##### Returns


`list[dict]`  
For a **built-in engine** report, one dict per failing step, with keys `dataset`, `step`, `assertion`, `column`, [n_failed](Validate.n_failed.md#pointblank.Validate.n_failed), [n](Validate.n.md#pointblank.Validate.n), and `severity`.

For a **CORE** report, one dict per (dataset, rule) with reported issues, with keys `dataset`, `rule_id`, `message`, [issues](ConformanceReport.md#pointblank.ConformanceReport.issues) (count), and `status`.


------------------------------------------------------------------------


#### rules()


Return the per-rule run results.


Usage

``` python
rules(status=None)
```


For CORE reports, returns `CoreRuleResult` objects. For built-in engine reports, returns `NativeRuleResult` objects. For Validate-based reports, returns an empty list.


##### Parameters


`status: str | None = None`  
Optional status filter. For CORE: e.g. `"SUCCESS"`, `"SKIPPED"`. For built-in engine reports: `"pass"`, `"fail"`, `"error"`, `"not_applicable"`, `"not_supported"`.


------------------------------------------------------------------------


#### summary()


Return a summary of the conformance run.


Usage

``` python
summary()
```


##### Returns


`dict`  
For a **built-in engine** report, a mapping of dataset name to a dict with keys `n_steps`, `n_steps_failed`, [n_failed](Validate.n_failed.md#pointblank.Validate.n_failed) (failing test units), and [all_passed](Validate.all_passed.md#pointblank.Validate.all_passed).

For a **CORE** report, a single dict with keys [standard](SubmissionPackage.md#pointblank.SubmissionPackage.standard), `version`, `engine_version`, `n_rules`, `status_counts` (rule counts by run status), `n_issues` (total reported issues), [n_datasets](ConformanceReport.md#pointblank.ConformanceReport.n_datasets), and [all_passed](Validate.all_passed.md#pointblank.Validate.all_passed).


------------------------------------------------------------------------


#### to_excel()


Save the conformance report as an Excel workbook.


Usage

``` python
to_excel(path)
```


For CORE reports the workbook contains sheets `Issue_Summary`, `Issue_Details`, `Rules_Report`, and `Conformance_Details`. For built-in engine reports the workbook contains `Issues` and `Summary`.

Requires the `openpyxl` package (`pip install openpyxl` or `pip install 'pointblank[excel]'`).


##### Parameters


`path: str | Path`  
Destination path (including filename). Parent directories are created if needed.


##### Returns


`Path`  
The path written.


##### Raises


`ImportError`  
If `openpyxl` or `pandas` are not installed.


------------------------------------------------------------------------


#### to_json()


Save the conformance report as a JSON file.


Usage

``` python
to_json(path)
```


For CORE reports the output mirrors the original CORE JSON structure (`Conformance_Details`, `Dataset_Details`, `Issue_Summary`, `Issue_Details`, `Rules_Report`), making the file readable by anything that parses a standard CORE report. For built-in engine reports the file contains [summary](ContractImport.md#pointblank.ContractImport.summary) and [issues](ConformanceReport.md#pointblank.ConformanceReport.issues) keys.


##### Parameters


`path: str | Path`  
Destination path (including filename). Parent directories are created if needed.


##### Returns


`Path`  
The path written.
