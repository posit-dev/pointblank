# Changelog

This changelog is generated automatically from [GitHub Releases](https://github.com/posit-dev/pointblank/releases).

## v1.1.1

*2026-10-15* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v1.1.1)

Packaging fix release: no functional changes from v1.1.0.

## v1.1.0

*2026-10-15* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v1.1.0)

Pointblank `v1.1.0` adds table comparison and makes validation results more correct across backends. The new `pb.compare()` function, the `tbl_reconcile()` validation step, and the `pb compare` CLI command show exactly how two tables differ: row by row, column by column, and in aggregate. `tbl_match()` now runs on the same engine, with full step reports. This release also fixes how missing values, Ibis tables, and expression errors are handled in many validation methods. Steps that can't be evaluated now report an evaluation error instead of a misleading failure.

#### New Features

- **Table comparison with `pb.compare()`** -- Aligns two tables by key, by position, or as multisets (`keys="*"`). Each row is classified as match, changed, missing, extra, or duplicate key, and schema differences are reported too. You can set numeric and datetime tolerances, normalize strings, map renamed columns with `column_map=`, and compare a subset of columns. The result is a [Comparison](./reference/Comparison.md#pointblank.Comparison) object with counts, row samples, a per-column summary, JSON export, and a Great Tables report in all 40 reporting languages. Inputs are scanned lazily, and `partitions=` limits memory use on very large tables. ([\#443](https://github.com/posit-dev/pointblank/issues/443))
- **`tbl_reconcile()` validation step** -- Reconciles aggregates (sum, count, mean, min, max) between two tables, either by key group or as grand totals. ([\#443](https://github.com/posit-dev/pointblank/issues/443))
- **`pb compare` CLI command** -- Compares two tables from the command line, with HTML or JSON output, `--engine duckdb`, and exit codes for CI use (`0` match, `1` differences, `2` error). ([\#443](https://github.com/posit-dev/pointblank/issues/443))
- **A richer `tbl_match()`** -- Now uses the comparison engine and accepts the same settings. It gains a full step report, failing-row extracts from both tables, sundering, a compact settings badge, and `to_code()`/`to_yaml()` support. **Note:** each aligned row now counts as one test unit (before, the whole step was one unit), so fractional thresholds now refer to the share of failing rows. Pass/fail verdicts are unchanged. To fail on any difference as before, use `thresholds=1`. ([\#443](https://github.com/posit-dev/pointblank/issues/443))

#### Enhancements

- Anthropic is no longer a required dependency. It's now installed only with the `pointblank[generate]` extra, so core installs are lighter. Thanks [@timkpaine](https://github.com/timkpaine)! ([\#437](https://github.com/posit-dev/pointblank/issues/437))
- `col_vals_increasing()` and `col_vals_decreasing()` gain an `order_by=` parameter, so they now work on database tables, Ibis, and PySpark. Each value is now compared with the last non-missing value before it. ([\#451](https://github.com/posit-dev/pointblank/issues/451))
- `col_vals_not_in_set()` gains an `na_pass=` parameter. Both set-membership methods now document how they handle missing values and `NaN`. ([\#450](https://github.com/posit-dev/pointblank/issues/450))
- `to_code()` and `to_yaml()` now include `col_vals_in_table()` steps (with a placeholder for the reference table). ([\#461](https://github.com/posit-dev/pointblank/issues/461))
- `col_vals_in_table(ref_table=)` now accepts file paths and connection strings. ([\#463](https://github.com/posit-dev/pointblank/issues/463))
- `col_vals_in_table()` now matches join key types across backends and handles numeric dtype mismatches in composite keys. ([\#431](https://github.com/posit-dev/pointblank/issues/431))
- A `pre=` function with a `dfn` parameter now always receives a Narwhals DataFrame, whether it's a lambda, a named function, or another callable. This now also works with [col()](./reference/col.md#pointblank.col) selectors and `segments=`. ([\#458](https://github.com/posit-dev/pointblank/issues/458))

#### Bug Fixes

**Evaluation errors instead of misleading failures**

- Invalid expressions in `col_vals_expr()` and `conjointly()` are now reported as evaluation errors. ([\#448](https://github.com/posit-dev/pointblank/issues/448))
- Comparison steps on columns with incompatible types are now reported as evaluation errors. ([\#459](https://github.com/posit-dev/pointblank/issues/459))
- [prompt()](./reference/Validate.prompt.md#pointblank.Validate.prompt) steps now record an evaluation error when an AI provider package is missing or every model request fails (e.g., an invalid API key). Before, every row was reported as failing. ([\#456](https://github.com/posit-dev/pointblank/issues/456))
- Step reports now explain evaluation errors, and failing rows are now extracted for Ibis tables. ([\#452](https://github.com/posit-dev/pointblank/issues/452))
- Tables are now converted to a common backend without changing their dtypes, and conversion failures are reported. ([\#453](https://github.com/posit-dev/pointblank/issues/453))

**Missing values**

- Missing values now fail `col_vals_not_in_set()` on every backend. Thanks [@raashish1601](https://github.com/raashish1601)! ([\#432](https://github.com/posit-dev/pointblank/issues/432))
- `col_vals_eq()` and `col_vals_ne()` now count missing values consistently when comparing against another column. Thanks [@raashish1601](https://github.com/raashish1601)! ([\#433](https://github.com/posit-dev/pointblank/issues/433), [\#449](https://github.com/posit-dev/pointblank/issues/449))
- Rows where a `col_vals_expr()` expression evaluates to null now count as failing. Thanks [@raashish1601](https://github.com/raashish1601)! ([\#435](https://github.com/posit-dev/pointblank/issues/435))

**Backend support**

- `col_vals_increasing()` and `col_vals_decreasing()` now support Polars LazyFrames. Thanks [@raashish1601](https://github.com/raashish1601)! ([\#434](https://github.com/posit-dev/pointblank/issues/434))
- `tbl_match()` now materializes Ibis memtables and other Ibis backends. Thanks [@raashish1601](https://github.com/raashish1601)! ([\#436](https://github.com/posit-dev/pointblank/issues/436))
- `conjointly()` expressions now evaluate on Ibis tables. Thanks [@raashish1601](https://github.com/raashish1601)! ([\#438](https://github.com/posit-dev/pointblank/issues/438))
- `pb.load_dataset()` no longer raises a permission error for DuckDB datasets on Windows. Thanks [@matt-humphrey](https://github.com/matt-humphrey)! ([\#245](https://github.com/posit-dev/pointblank/issues/245))

**Reports**

- Step reports can now be built for passing steps that have no extracts. Thanks [@raashish1601](https://github.com/raashish1601)! ([\#440](https://github.com/posit-dev/pointblank/issues/440))
- [get_dataframe_report()](./reference/Validate.get_dataframe_report.md#pointblank.Validate.get_dataframe_report) now supports steps with several columns and serializes non-string values for DuckDB. Thanks [@cristianchiriac](https://github.com/cristianchiriac)! ([\#444](https://github.com/posit-dev/pointblank/issues/444), [\#445](https://github.com/posit-dev/pointblank/issues/445))
- DuckDB report values are now stored as JSON text, full composite keys are reported, and `step_evaluated` is filled in. ([\#454](https://github.com/posit-dev/pointblank/issues/454))
- [col_summary_tbl()](./reference/col_summary_tbl.md#pointblank.col_summary_tbl) no longer errors on boolean columns, including ones with missing values. ([\#466](https://github.com/posit-dev/pointblank/issues/466))

**Data generation**

- [generate_dataset()](./reference/generate_dataset.md#pointblank.generate_dataset) now respects `unique=True` for coherent presets. ([\#447](https://github.com/posit-dev/pointblank/issues/447))

**CDISC conformance**

- The bundled controlled terminology is now generated from the official NCI EVS file for CDISC CT 2024-09-27. Before, only 3 of 19 bundled codelists matched the official ones, so official values (e.g., NY = U, SEX = INTERSEX) were rejected. COUNTRY now covers all ISO 3166-1 alpha-3 codes, and RACE accepts MULTIPLE. ([\#460](https://github.com/posit-dev/pointblank/issues/460))
- The AEACN, FREQ, and EPOCH codelists have been corrected, and ten rules that contradicted SDTMIG 3.4 have been removed (along with the AEREL and AETOXGR codelist rules). ([\#455](https://github.com/posit-dev/pointblank/issues/455), [\#457](https://github.com/posit-dev/pointblank/issues/457), [\#460](https://github.com/posit-dev/pointblank/issues/460))
- SDTMIG 3.4 presence and not-null rules now follow the IG's core designations. ([\#464](https://github.com/posit-dev/pointblank/issues/464))
- Define-XML extensions of extensible CDISC codelists are now accepted in conformance checks. ([\#462](https://github.com/posit-dev/pointblank/issues/462))

#### Documentation

- The lambda `pre=` example uses a lambda function again. Thanks [@dpprdan](https://github.com/dpprdan)! ([\#330](https://github.com/posit-dev/pointblank/issues/330))

## v1.0.1

*2026-10-07* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v1.0.1)

Pointblank `v1.0.1` is a maintenance release that makes Pointblank fully compatible with Polars 2.0. Polars 2.0 no longer silently converts between integers and floats in membership checks, and it requires equal-height frames for horizontal concatenation. Both changes caused errors in several validation methods and in schema step reports. This release fixes those errors across the board and corrects row ordering and row numbering for Polars LazyFrames. Polars `1.33.0` and later remain supported. Many thanks to [@FBruzzesi](https://github.com/FBruzzesi) for spotting these issues in Narwhals CI and contributing the first set of fixes!

#### Bug Fixes

- `col_vals_in_set()` and `col_vals_not_in_set()` work again on Polars 2.0 when integer columns are checked against float values or float columns against integer values, including mixed sets such as `[1, 2.5]`. ([\#429](https://github.com/posit-dev/pointblank/issues/429), [\#430](https://github.com/posit-dev/pointblank/issues/430))
- Step reports for `col_schema_match()` no longer raise a `ShapeError` on Polars 2.0 when the table and the expected schema have different numbers of columns. ([\#429](https://github.com/posit-dev/pointblank/issues/429))
- Mismatches between integer, float and Decimal types no longer raise errors on Polars 2.0 in `col_vals_in_table()`, `col_pct_missing()`, `col_missing_only_coded()`, `col_missing_consistent()`, [missing_vals_tbl()](./reference/missing_vals_tbl.md#pointblank.missing_vals_tbl), the `missing=` argument of validation methods, segment groups, or the native CDISC conformance rule engine. ([\#430](https://github.com/posit-dev/pointblank/issues/430))
- [get_sundered_data()](./reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) now returns rows in their original order for Polars LazyFrames. Polars 2.0's streaming engine could otherwise reorder them. ([\#430](https://github.com/posit-dev/pointblank/issues/430))
- [get_data_extracts()](./reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) now reports correct row numbers for Polars LazyFrames. Previously, rows were numbered by the sort order of the first column, on every Polars version. ([\#430](https://github.com/posit-dev/pointblank/issues/430))

#### Maintenance

- CI now runs the test suite against the latest Polars and Narwhals releases as well as the minimum supported versions, so new releases of either library are tested as soon as they come out. ([\#430](https://github.com/posit-dev/pointblank/issues/430))

## v1.0.0

*2026-10-06* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v1.0.0)

Pointblank reaches `v1.0.0` with new ways to reuse and compose validation logic, a new string-length validation method, and full compatibility with Great Tables `v1.0.0`. The new [Steps](./reference/Steps.md#pointblank.Steps) builder and [Validate.add_steps()](./reference/Validate.add_steps.md#pointblank.Validate.add_steps) let you define step libraries once and combine them across validation plans, with options to remap column names, override thresholds, or skip specific steps when importing. The new `col_vals_str_len()` method checks that string lengths fall within minimum and/or maximum bounds. This release also restores validation reports under Great Tables `v1.0.0`, which now escapes HTML in table cells by default.

#### New Features

- **Composable validation steps**: the new `pb.Steps()` builder records validation steps without attaching them to a table, and [Validate.add_steps()](./reference/Validate.add_steps.md#pointblank.Validate.add_steps) adds them to any validation plan. When importing steps, you can remap columns with `columns_map=`, override `thresholds=` or `active=`, and drop steps by name or position with `exclude=`. Existing [Validate](./reference/Validate.md#pointblank.Validate) objects can also serve as step sources, and [Steps](./reference/Steps.md#pointblank.Steps) objects show a formatted HTML summary in notebooks. A new *Composable Steps* article in the User Guide covers the workflow. ([\#424](https://github.com/posit-dev/pointblank/issues/424))
- **`col_vals_str_len()` validation method**: checks whether the character length of string values in one or more columns falls between `min_val=` and/or `max_val=`. It supports the usual `na_pass=`, `pre=`, `segments=`, `thresholds=`, and `actions=` options, has translated autobriefs, and is also available on [Steps](./reference/Steps.md#pointblank.Steps). ([\#422](https://github.com/posit-dev/pointblank/issues/422))

#### Bug Fixes

- Validation reports now render correctly with Great Tables `v1.0.0`. Previously, the HTML in the report's cells (status icons, badges, and labels) appeared as escaped markup. The fix also keeps working with earlier Great Tables versions. ([\#426](https://github.com/posit-dev/pointblank/issues/426))
- The MCP server's prompts now import `Message` from FastMCP's current module path, fixing an import error with recent FastMCP releases.
- `sqlglot` is now restricted to versions below `30.18`, since `30.18` and later omit table names in `DROP` statements and break DuckDB and SQLite backends. ([\#425](https://github.com/posit-dev/pointblank/issues/425))

#### Documentation

- Added a glossary to the reference section of the User Guide that defines key Pointblank terms.
- LLM examples in the skills docs now use Anthropic model IDs that are still served. ([\#423](https://github.com/posit-dev/pointblank/issues/423))
- Clarified the failing-case example in the length validation docs.

#### Maintenance

- Test coverage was expanded substantially across the validation, MCP, metadata, conformance, adapter, scan profile, and data generation modules.

## v0.27.0

*2026-08-13* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.27.0)

Pointblank `v0.27.0` brings referential integrity validation, statistical drift detection, and expanded data profiling capabilities. The new `col_vals_in_table()` validation method enables cross-table foreign key checks, while [DataScan.compare()](./reference/DataScan.compare.md#pointblank.DataScan.compare) and the [DataScanDiff](./reference/DataScanDiff.md#pointblank.DataScanDiff) class make it possible to detect schema and statistical drift between data profiles over time. Several skills are now packaged with the library for immediate use, contract adapters received important additions and fixes, and the MCP integration has been cleaned up with a leaner dependency footprint.

#### New Features

- **Referential integrity validation with `col_vals_in_table()`** -- New validation method that checks whether each value (or composite key) in a column exists in a reference table column. Supports single and multi-column keys, `na_pass`, segmentation, and all standard thresholds and actions, making foreign key-style consistency checks a first-class validation step. ([\#419](https://github.com/posit-dev/pointblank/issues/419))

- **Statistical drift measures via [DataScanDiff](./reference/DataScanDiff.md#pointblank.DataScanDiff)** -- `DataScan.compare(baseline)` now returns a [DataScanDiff](./reference/DataScanDiff.md#pointblank.DataScanDiff) object that provides programmatic access to schema changes (columns added, removed, or type-changed) and per-column statistical drift metrics -- Population Stability Index (PSI) for numeric and categorical columns, and Kolmogorov-Smirnov statistic for numerics (with a pure-Python fallback when SciPy is unavailable). Results are surfaced via `get_tabular_report()`, which renders a styled Great Tables report with schema and drift columns. ([\#421](https://github.com/posit-dev/pointblank/issues/421))

- **Packaged skills** -- A set of skills is now distributed with the package (e.g., defining contracts, drafting validations, scanning and profiling data, writing validations, and more). ([\#417](https://github.com/posit-dev/pointblank/issues/417))

#### Enhancements

- **Extended [DataScan](./reference/DataScan.md#pointblank.DataScan) class** -- [DataScan](./reference/DataScan.md#pointblank.DataScan) can now be serialized and restored without access to the original data. New methods include `to_dict()`, `to_json()`, [save_to_json()](./reference/DataScan.save_to_json.md#pointblank.DataScan.save_to_json), `from_dict()`, [from_json()](./reference/DataScan.from_json.md#pointblank.DataScan.from_json), and [load_from_json()](./reference/DataScan.load_from_json.md#pointblank.DataScan.load_from_json) for round-trip persistence of data profiles. Stored scans can then be compared against new scans using `compare()` to track drift over time. ([\#420](https://github.com/posit-dev/pointblank/issues/420))

- **Contract adapter additions and fixes** -- The contract adapter system received expanded functionality and correctness fixes across the supported formats (dbt, Frictionless, JSON Schema, ODCS). Adapter import and export coverage was improved, and [ContractImport.to_contract()](./reference/ContractImport.md#pointblank.ContractImport.to_contract) and [ContractImport.to_python()](./reference/ContractImport.md#pointblank.ContractImport.to_python) output is now more reliable. ([\#418](https://github.com/posit-dev/pointblank/issues/418))

#### Bug Fixes

- Removed the direct `mcp[cli]` dependency and replaced it with an updated `fastmcp>=2.14.2` pin, resolving installation conflicts in environments that use the MCP optional extra. ([\#416](https://github.com/posit-dev/pointblank/issues/416))

#### New Contributors

- [@FBruzzezi](https://github.com/FBruzzezi) made their first contribution in [\#416](https://github.com/posit-dev/pointblank/issues/416).

## v0.26.0

*2026-07-27* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/0.26.0)

Pointblank `v0.26.0` is a major release focused on clinical data standards, data quality measurement, and AI-assisted validation editing. This release introduces comprehensive CDISC conformance validation with a built-in rule engine and bundled controlled terminology, a metadata import/export system supporting SDTM, ADaM, Frictionless, and CSVW formats, and structured missingness handling through a new [MissingSpec](./reference/MissingSpec.md#pointblank.MissingSpec) type. Validation steps are now automatically tagged with data quality dimensions and rolled up into health scores, and the new [EditValidation](./reference/EditValidation.md#pointblank.EditValidation) class lets you modify existing validation plans through interaction with an LLM.

##### New Features

- **Data quality dimensions and health scoring** -- Every validation step is now tagged with a data quality dimension (Completeness, Validity, Uniqueness, Consistency, Timeliness, or Volume) inferred automatically from the assertion type. Dimensions are overridable per step via `dimension=` or remappable globally via `pb.config(dimension_map=...)`. New methods [get_dimension_scores()](./reference/Validate.get_dimension_scores.md#pointblank.Validate.get_dimension_scores), [get_health_score()](./reference/Validate.get_health_score.md#pointblank.Validate.get_health_score), [get_scorecard()](./reference/Validate.get_scorecard.md#pointblank.Validate.get_scorecard), and [get_validation_summary()](./reference/get_validation_summary.md#pointblank.get_validation_summary) expose test-unit-weighted scores, with [assert_dimension_scores()](./reference/Validate.assert_dimension_scores.md#pointblank.Validate.assert_dimension_scores) available to fail a run when a dimension falls below a threshold. The validation report shows color-coded dimension badges on each step number with a health-score summary in the footer. ([\#409](https://github.com/posit-dev/pointblank/issues/409))

- **CDISC submission conformance** -- A new conformance validation framework provides a rule engine, bundled SDTMIG 3.4 rules with SDTM controlled terminology, and public APIs for validating study submission packages. Includes [ConformanceReport](./reference/ConformanceReport.md#pointblank.ConformanceReport) with JSON and Excel export, [findings_df()](./reference/ConformanceReport.md#pointblank.ConformanceReport.findings_df) and [get_findings_table()](./reference/ConformanceReport.md#pointblank.ConformanceReport.get_findings_table) for detailed inspection, and integration with the CDISC CORE engine. ([\#413](https://github.com/posit-dev/pointblank/issues/413))

- **Metadata standards** -- New metadata import, export, and validation workflows for clinical data standards. The [import_metadata()](./reference/import_metadata.md#pointblank.import_metadata) function reads Frictionless Table Schema, CSVW, CDISC Define-XML, CDISC Controlled Terminology, and SPSS/SAS/Stata files into a unified [MetadataImport](./reference/MetadataImport.md#pointblank.MetadataImport) object. Built-in SDTM and ADaM templates enable domain-level structure validation with [validate_sdtm()](./reference/validate_sdtm.md#pointblank.validate_sdtm) and [validate_adam()](./reference/validate_adam.md#pointblank.validate_adam). Metadata can be exported back to Frictionless format and converted to Pointblank schemas. ([\#403](https://github.com/posit-dev/pointblank/issues/403))

- **Schema enforcement API** -- A contract import/export system with a modular adapter architecture. `import_contract()` and `export_contract()` translate between external schema formats (JSON Schema, Frictionless Table Schema) and Pointblank validations or contracts. Custom adapters can be registered via the `[@register](https://github.com/register)_adapter` decorator. ([\#402](https://github.com/posit-dev/pointblank/issues/402))

- **Structured missingness** -- A new [MissingSpec](./reference/MissingSpec.md#pointblank.MissingSpec) type maps sentinel values to reasons, and a `missing=` parameter on `col_vals_*()` methods excludes coded values from checks. Four dedicated validation steps are added: `col_pct_missing()`, `col_missing_coded()`, `col_missing_only_coded()`, and `col_missing_consistent()`. ([\#404](https://github.com/posit-dev/pointblank/issues/404))

- **AI validation editor** -- The new [EditValidation](./reference/EditValidation.md#pointblank.EditValidation) class lets you modify an existing validation plan from a natural-language instruction, with [diff()](./reference/EditValidation.md#pointblank.EditValidation.diff), [changed_steps()](./reference/EditValidation.md#pointblank.EditValidation.changed_steps), [preview()](./reference/preview.md#pointblank.preview), and [accept()](./reference/EditValidation.md#pointblank.EditValidation.accept) methods for reviewing changes. Plan serialization via [Validate.to_code()](./reference/Validate.to_code.md#pointblank.Validate.to_code) and [Validate.to_yaml()](./reference/Validate.to_yaml.md#pointblank.Validate.to_yaml) is fully round-trippable. Also adds [Validate.suggest_improvements()](./reference/Validate.suggest_improvements.md#pointblank.Validate.suggest_improvements), [Validate.from_prompt()](./reference/Validate.from_prompt.md#pointblank.Validate.from_prompt), `Validate.from_plans()`, a syntax-lint guardrail with auto-reprompt, and a `pb edit` CLI command. ([\#408](https://github.com/posit-dev/pointblank/issues/408))

- **[float_field()](./reference/float_field.md#pointblank.float_field) `precision=` parameter** -- Generated float values can now be rounded to a specified number of decimal places, useful for prices, scores, and other fields where full floating-point precision is not desired. ([\#412](https://github.com/posit-dev/pointblank/issues/412))

##### Bug Fixes

- Per-row validation checks on Polars LazyFrames are no longer 8-11× slower than v0.24.0. The result-tallying code now computes row, pass, fail, and null counts in a single aggregation instead of issuing three to four separate `.collect()` calls that re-executed the entire lazy plan. ([\#407](https://github.com/posit-dev/pointblank/issues/407))

- Segmented validation on Polars LazyFrames no longer raises an `UnboundLocalError` for `seg_tuples()`. Segment expressions now work consistently for both eager and lazy DataFrames. ([\#414](https://github.com/posit-dev/pointblank/issues/414))

##### Documentation

- CDISC validation user guide examples are now fully executable against sample metadata documents bundled with the package via the new [load_metadata_example()](./reference/load_metadata_example.md#pointblank.load_metadata_example) function. ([\#406](https://github.com/posit-dev/pointblank/issues/406))

## v0.25.0

*2026-06-12* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.25.0)

Pointblank `v0.25.0` introduces powerful new capabilities for data pipeline governance and synthetic data generation. This release adds boundary enforcement through contracts and pipelines, letting you validate data at both the source and target of a transformation. Rich schema inference automatically detects constraints from existing tables (including ranges, categorical values, and semantic presets) enabling realistic synthetic data generation. The [prompt()](./reference/Validate.prompt.md#pointblank.Validate.prompt) validation method now accepts multi-modal attachments (images and PDFs), giving the LLM visual context when evaluating data quality.

#### New Features

- **Rich schema inference** -- `Schema.from_table()` and [schema_from_tbl()](./reference/schema_from_tbl.md#pointblank.schema_from_tbl) automatically extract comprehensive constraints from any supported table, including numeric ranges, categorical values, nullability, date bounds, and 25+ semantic presets (email, address, UUID, etc.) to power realistic synthetic data generation. ([\#391](https://github.com/posit-dev/pointblank/issues/391))

- **Multi-modal attachments in [prompt()](./reference/Validate.prompt.md#pointblank.Validate.prompt)** -- The [prompt()](./reference/Validate.prompt.md#pointblank.Validate.prompt) validation method now accepts an `attachments=` parameter for images and PDFs (local files or URLs), providing visual context to the LLM for richer data validation against brand guides, schema diagrams, or reference documents. ([\#393](https://github.com/posit-dev/pointblank/issues/393))

- **Boundary enforcement with contracts and pipelines** -- New [Contract](./reference/Contract.md#pointblank.Contract) and [Pipeline](./reference/Pipeline.md#pointblank.Pipeline) classes enable dual-boundary validation at the source and target of data transformations. Contracts define schema expectations and validation steps with ownership metadata, while pipelines orchestrate the full validate-transform-validate workflow with short-circuit optimization. ([\#401](https://github.com/posit-dev/pointblank/issues/401))

#### Bug Fixes

- Lazy frames are no longer prematurely collected during interrogation, preserving the performance benefits of deferred evaluation with Polars LazyFrames. ([\#398](https://github.com/posit-dev/pointblank/issues/398))

- `rows_distinct()` now correctly handles rows containing null values instead of dropping them from validation results. ([\#399](https://github.com/posit-dev/pointblank/issues/399))

- The MCP server now starts reliably across different versions of the `fastmcp` library by using version-aware instantiation with fallback strategies. ([\#400](https://github.com/posit-dev/pointblank/issues/400))

## v0.24.0

*2026-04-22* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.24.0)

### New Features

- Added the [get_dataframe_report()](./reference/Validate.get_dataframe_report.md#pointblank.Validate.get_dataframe_report) method to [Validate](./reference/Validate.md#pointblank.Validate) for writing validation results to a DataFrame (supports Polars, Pandas, and DuckDB output). ([@Meghansaha](https://github.com/Meghansaha), [\#341](https://github.com/posit-dev/pointblank/issues/341))
- Added OpenTelemetry (OTel) integration for exporting validation results as metrics, traces, and logs. Available as an optional dependency via the `otel` extra. ([\#383](https://github.com/posit-dev/pointblank/issues/383))
- Added Azure OpenAI as a model provider for both [Validate.prompt()](./reference/Validate.prompt.md#pointblank.Validate.prompt) and [DraftValidation](./reference/DraftValidation.md#pointblank.DraftValidation). ([@howardbaik](https://github.com/howardbaik), [\#385](https://github.com/posit-dev/pointblank/issues/385), [\#387](https://github.com/posit-dev/pointblank/issues/387))

### Docs

- Reorganized demos into the new docs structure. ([\#381](https://github.com/posit-dev/pointblank/issues/381))
- Added a User Guide page on OpenTelemetry integration. ([\#383](https://github.com/posit-dev/pointblank/issues/383))
- Added Azure OpenAI usage and setup instructions to the User Guide. ([\#385](https://github.com/posit-dev/pointblank/issues/385), [\#387](https://github.com/posit-dev/pointblank/issues/387))

### New Contributors

- [@howardbaik](https://github.com/howardbaik) made their first contribution in [\#385](https://github.com/posit-dev/pointblank/issues/385)

Full Changelog: [v0.23.0…v0.24.0](https://github.com/posit-dev/pointblank/compare/v0.23.0...v0.24.0)

## v0.23.0

*2026-04-01* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.23.0)

### Features

- We now run pyrefly to infer types. ([@tylerriccio33](https://github.com/tylerriccio33), [\#374](https://github.com/posit-dev/pointblank/issues/374))
- Data generation now offers a locale code string preset. ([\#375](https://github.com/posit-dev/pointblank/issues/375))
- A credit card provider is now a string preset for data generation. ([\#377](https://github.com/posit-dev/pointblank/issues/377))

### Fixes:

- Audit the performance of lazy transformations by [@tylerriccio33](https://github.com/tylerriccio33) in https://github.com/posit-dev/pointblank/pull/373
- Aggregation and Reference Column Hardening by [@tylerriccio33](https://github.com/tylerriccio33) in https://github.com/posit-dev/pointblank/pull/372

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.22.0…v0.23.0](https://github.com/posit-dev/pointblank/compare/v0.22.0...v0.23.0)

## v0.22.0

*2026-02-26* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.22.0)

### New Features

- Expanded [generate_dataset()](./reference/generate_dataset.md#pointblank.generate_dataset) country support with 29 new countries (EC, PA, SA, UA, AM, AZ, BO, CM, DO, GE, GT, HN, IL, JM, JO, KH, KZ, LB, MD, MM, MZ, NP, PY, RS, RW, SV, TZ, UY, UZ), bringing the total to 100 supported countries, along with expanded street lists and normalized industry names across existing countries. ([\#370](https://github.com/posit-dev/pointblank/issues/370), [\#371](https://github.com/posit-dev/pointblank/issues/371))
- Added [profile_fields()](./reference/profile_fields.md#pointblank.profile_fields) as a composite helper that creates a dictionary of person-profile `StringField` objects for direct unpacking into a [Schema()](./reference/Schema.md#pointblank.Schema). Supports three tiers (`"minimal"`, `"standard"`, `"full"`), name splitting, column include/exclude, and a `prefix=` parameter for namespacing. ([\#367](https://github.com/posit-dev/pointblank/issues/367))
- Significantly expanded YAML workflow support with governance metadata (`owner`, `consumers`, `version`), `final_actions` and `reference` top-level keys, aggregate validation methods ([col_sum_gt](./reference/Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt), [col_avg_le](./reference/Validate.col_avg_le.md#pointblank.Validate.col_avg_le), etc.), `col_pct_null`, `data_freshness`, shortcut syntax for the `active` parameter, unknown-key validation to catch typos, and YAML-to-Python roundtrip fidelity. ([\#369](https://github.com/posit-dev/pointblank/issues/369))

### Bug Fixes

- Fixed [preview()](./reference/preview.md#pointblank.preview) failing on tables with duration (timedelta) columns by casting them to strings before display. ([\#368](https://github.com/posit-dev/pointblank/issues/368))

### Docs

- Added YAML reference and validation workflow guides covering governance metadata and aggregate methods. ([\#369](https://github.com/posit-dev/pointblank/issues/369))
- Updated data generation documentation with [profile_fields()](./reference/profile_fields.md#pointblank.profile_fields) usage and 100-country support. ([\#367](https://github.com/posit-dev/pointblank/issues/367), [\#370](https://github.com/posit-dev/pointblank/issues/370), [\#371](https://github.com/posit-dev/pointblank/issues/371))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.21.0…v0.22.0](https://github.com/posit-dev/pointblank/compare/v0.21.0...v0.22.0)

## v0.21.0

*2026-02-19* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.21.0)

### New Features

- Significantly enhanced [generate_dataset()](./reference/generate_dataset.md#pointblank.generate_dataset) with locale mixing (`country=` now accepts lists or weighted dicts), frequency-weighted sampling via a 4-tier system, a `user_agent` preset with country-weighted browser selection, presets for hash digests (`md5`, `sha1`, `sha256`), barcodes (`ean8`, `ean13`), date strings (`date_between`, `date_range`, `future_date`, `past_date`), and ISO 3166-1 country codes (`country_code_2`, `country_code_3`), country-specific license plate formats, industry-coherent person-company pairing, a [generate_dataset](./reference/generate_dataset.md#pointblank.generate_dataset) pytest fixture with automatic per-test seeding, and expanded country support (bringing the total to 71). ([\#352](https://github.com/posit-dev/pointblank/issues/352), [\#354](https://github.com/posit-dev/pointblank/issues/354), [\#355](https://github.com/posit-dev/pointblank/issues/355), [\#358](https://github.com/posit-dev/pointblank/issues/358), [\#360](https://github.com/posit-dev/pointblank/issues/360), [\#361](https://github.com/posit-dev/pointblank/issues/361), [\#362](https://github.com/posit-dev/pointblank/issues/362), [\#363](https://github.com/posit-dev/pointblank/issues/363), [\#364](https://github.com/posit-dev/pointblank/issues/364), [\#365](https://github.com/posit-dev/pointblank/issues/365), [\#366](https://github.com/posit-dev/pointblank/issues/366))
- Added expressions support for the `active=` parameter to provide for more flexible and dynamic validation workflows. ([\#349](https://github.com/posit-dev/pointblank/issues/349))

### Docs

- Added validation report notes output display in documentation examples. ([\#350](https://github.com/posit-dev/pointblank/issues/350))
- Improved docstrings for [generate_dataset()](./reference/generate_dataset.md#pointblank.generate_dataset) and all `*_field()` functions. ([\#353](https://github.com/posit-dev/pointblank/issues/353))
- Fixed broken links in `quickstart.qmd` by removing incorrect path prefixes. ([@Meghansaha](https://github.com/Meghansaha), [\#357](https://github.com/posit-dev/pointblank/issues/357))
- Updated data generation documentation with new presets and features. ([\#359](https://github.com/posit-dev/pointblank/issues/359))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.20.0…v0.21.0](https://github.com/posit-dev/pointblank/compare/v0.20.0...v0.21.0)

## v0.20.0

*2026-02-04* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.20.0)

### New Features

- Added the [generate_dataset()](./reference/generate_dataset.md#pointblank.generate_dataset) function for creating realistic synthetic test data from [Schema](./reference/Schema.md#pointblank.Schema) definitions, with a suite of field helper functions ([int_field()](./reference/int_field.md#pointblank.int_field), [string_field()](./reference/string_field.md#pointblank.string_field), [date_field()](./reference/date_field.md#pointblank.date_field), `name_field()`, etc.) and support for 50 country locales via the `country=` parameter. ([\#348](https://github.com/posit-dev/pointblank/issues/348))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.19.0…v0.20.0](https://github.com/posit-dev/pointblank/compare/v0.19.0...v0.20.0)

## v0.19.0

*2026-01-21* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.19.0)

### New Features

- We can now specify `owner`, `consumers`, and `version` metadata for data validations within the [Validate](./reference/Validate.md#pointblank.Validate) class. ([\#344](https://github.com/posit-dev/pointblank/issues/344))
- The new `data_freshness()` validation method has been added for easily checking the freshness of data. ([\#345](https://github.com/posit-dev/pointblank/issues/345))

### Fixes

- Added the `.step_report()` capability for the aggregate validation methods (e.g., [col_sum_gt()](./reference/Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt), [col_sd_lt()](./reference/Validate.col_sd_lt.md#pointblank.Validate.col_sd_lt), etc.). ([\#343](https://github.com/posit-dev/pointblank/issues/343))
- Updated SVG icons for all of the aggregate validation methods. ([\#346](https://github.com/posit-dev/pointblank/issues/346))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.18.0…v0.19.0](https://github.com/posit-dev/pointblank/compare/v0.18.0...v0.19.0)

## v0.18.0

*2026-01-13* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.18.0)

### New Features

- Added several validation methods that work on aggregates of column values (e.g.,[col_sum_gt()](./reference/Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt), [col_sd_lt()](./reference/Validate.col_sd_lt.md#pointblank.Validate.col_sd_lt), etc.). ([@tylerriccio33](https://github.com/tylerriccio33), [\#332](https://github.com/posit-dev/pointblank/issues/332))

### Fixes

- Implemented fix for handling datetime strings in `_apply_segments()`. ([\#337](https://github.com/posit-dev/pointblank/issues/337))
- Added reference data as a default for aggregation comparison values. ([@tylerriccio33](https://github.com/tylerriccio33), [\#339](https://github.com/posit-dev/pointblank/issues/339))
- Add experimental gradual typing for development. ([@tylerriccio33](https://github.com/tylerriccio33), [\#198](https://github.com/posit-dev/pointblank/issues/198), [\#338](https://github.com/posit-dev/pointblank/issues/338))
- Improved the display of aggregation-type methods in the validation report. ([\#342](https://github.com/posit-dev/pointblank/issues/342))

### Docs

- Updated the reference API with aggregate check methods. ([\#340](https://github.com/posit-dev/pointblank/issues/340))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.17.0…v0.18.0](https://github.com/posit-dev/pointblank/compare/v0.17.0...v0.18.0)

## v0.17.0

*2025-12-02* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.17.0)

### New Features

- All `col_schema_match()` validation steps are now accompanied by notes that describe the results. ([\#325](https://github.com/posit-dev/pointblank/issues/325))
- Notes appear in validation reports notes when columns are not found or none are resolved by use of column selectors. ([\#326](https://github.com/posit-dev/pointblank/issues/326))
- Validation reports now have informative notes for steps that use the `pre=` parameter. ([\#328](https://github.com/posit-dev/pointblank/issues/328))
- New options are available (in `get_tabular_report()` and globally in `config()`) for enabling/disabling validation report footer sections. ([\#327](https://github.com/posit-dev/pointblank/issues/327))
- Added the `test-core` target for running core tests (large time savings compared to running all tests). ([\#331](https://github.com/posit-dev/pointblank/issues/331))
- New validation method added, `col_pct_null()`, for checking the percentage of Null values in a column. ([\#290](https://github.com/posit-dev/pointblank/issues/290), [@tylerriccio33](https://github.com/tylerriccio33))

### Fixes

- Removed a duplicate paragraph in the docs. ([\#329](https://github.com/posit-dev/pointblank/issues/329), [@dpprdan](https://github.com/dpprdan))

### New Contributors

- [@dpprdan](https://github.com/dpprdan) made their first contribution in https://github.com/posit-dev/pointblank/pull/329

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.16.0…v0.17.0](https://github.com/posit-dev/pointblank/compare/v0.16.0...v0.17.0)

## v0.16.0

*2025-11-18* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.16.0)

### New Features

- Use of local thresholds settings will now produce notes at the bottom of the validation report table. ([\#318](https://github.com/posit-dev/pointblank/issues/318))
- Added the [print_database_tables()](./reference/print_database_tables.md#pointblank.print_database_tables) function for printing a list of tables in a database connection. ([\#322](https://github.com/posit-dev/pointblank/issues/322), [@Meghansaha](https://github.com/Meghansaha))
- Added YAML support for several recently added validation methods. ([\#312](https://github.com/posit-dev/pointblank/issues/312))
- There are now report translations for all official EU languages. ([\#314](https://github.com/posit-dev/pointblank/issues/314))

### Fixes

- Ensured that Pointblank is compatible with Python 3.14 ([\#320](https://github.com/posit-dev/pointblank/issues/320))

### Docs

- Revised project website navigation to put the User Guide front and center. ([\#307](https://github.com/posit-dev/pointblank/issues/307), [\#321](https://github.com/posit-dev/pointblank/issues/321))
- Modified logo used in website. ([\#308](https://github.com/posit-dev/pointblank/issues/308))
- Added both the `llms.txt` and `llms-full.txt` files to the project website. ([\#310](https://github.com/posit-dev/pointblank/issues/310))
- Updated documentation with newly-added validation methods. ([\#311](https://github.com/posit-dev/pointblank/issues/311))
- Added a PDF version of the User Guide to the website ([\#313](https://github.com/posit-dev/pointblank/issues/313))
- Improved the appearance of the website's sidebar button (visible at constrained screen widths). ([\#315](https://github.com/posit-dev/pointblank/issues/315))
- Fixed broken contributor info links. ([\#316](https://github.com/posit-dev/pointblank/issues/316), [@Meghansaha](https://github.com/Meghansaha))

### New Contributors

- [@Meghansaha](https://github.com/Meghansaha) made their first contribution in https://github.com/posit-dev/pointblank/pull/316

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.15.0…v0.16.0](https://github.com/posit-dev/pointblank/compare/v0.15.0...v0.16.0)

## v0.15.0

*2025-10-28* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.15.0)

### New Features

- Several new validation methods were incorporated: `col_vals_within_spec()`, `col_vals_increasing()`, `col_vals_decreasing()`, and `tbl_match()`. ([\#304](https://github.com/posit-dev/pointblank/issues/304), [\#305](https://github.com/posit-dev/pointblank/issues/305), [\#306](https://github.com/posit-dev/pointblank/issues/306))
- The option to ignore SSL verification was added to the [DraftValidation](./reference/DraftValidation.md#pointblank.DraftValidation) class via the new `verify_ssl=` parameter. ([\#302](https://github.com/posit-dev/pointblank/issues/302))
- Added the (internal for now) 'notes' functionality for validation step, which will allow for useful information in individual validation steps to be available post-interrogation. ([\#303](https://github.com/posit-dev/pointblank/issues/303))

### Fixes

- Altered the HTML tags of 'brief' text in validation reports for better display across different publishing environments. ([\#299](https://github.com/posit-dev/pointblank/issues/299))

### Docs

- Added a User Guide page on validation reports. ([\#300](https://github.com/posit-dev/pointblank/issues/300))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.14.0…v0.15.0](https://github.com/posit-dev/pointblank/compare/v0.14.0...v0.15.0)

## v0.14.0

*2025-10-10* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.14.0)

### New Features

- Added the `.prompt()` validation method to enable data validation using LLMs, evaluating data in rows against natural language descriptions instead of programmatic rules. ([\#287](https://github.com/posit-dev/pointblank/issues/287))
- The new [write_file()](./reference/write_file.md#pointblank.write_file) and [read_file()](./reference/read_file.md#pointblank.read_file) functions allow for writing/reading validation objects to and from disk. ([\#291](https://github.com/posit-dev/pointblank/issues/291))
- More translation languages added (`"id"`, `"uk"`, `"he"`, `"th"`, and `"fa"`). ([\#293](https://github.com/posit-dev/pointblank/issues/293))

### Docs

- Summary fields in the [get_validation_summary()](./reference/get_validation_summary.md#pointblank.get_validation_summary) docs examples were updated. ([@jrycw](https://github.com/jrycw), [\#284](https://github.com/posit-dev/pointblank/issues/284))
- The [yaml_to_python()](./reference/yaml_to_python.md#pointblank.yaml_to_python) function now has published docs on the project website. ([@jrycw](https://github.com/jrycw), [\#288](https://github.com/posit-dev/pointblank/issues/288))
- There's now a Posit badge in the header of the project website. ([\#292](https://github.com/posit-dev/pointblank/issues/292))

### Fixes

- Zero-row tables are now better handled in the validation workflow. ([\#297](https://github.com/posit-dev/pointblank/issues/297))

### Chores

- Added many tests to improve code coverage. ([\#294](https://github.com/posit-dev/pointblank/issues/294), [\#298](https://github.com/posit-dev/pointblank/issues/298))
- New tests and documentation now available for validations involving datetime + timezone comparisons. ([\#289](https://github.com/posit-dev/pointblank/issues/289))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.13.4…v0.14.0](https://github.com/posit-dev/pointblank/compare/v0.13.4...v0.14.0)

## v0.13.4

*2025-09-19* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.13.4)

### New Features

- Add Enum support to `col_vals_in_set()` and `col_vals_not_in_set()`. ([\#280](https://github.com/posit-dev/pointblank/issues/280))
- The `col_vals_regex()` validation method now has the `inverse=` parameter to enable negative regex matching. ([\#282](https://github.com/posit-dev/pointblank/issues/282))

### Fixes

- The usage of `pre=` is now better isolated to the steps using it. ([\#283](https://github.com/posit-dev/pointblank/issues/283))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.13.3…v0.13.4](https://github.com/posit-dev/pointblank/compare/v0.13.3...v0.13.4)

## v0.13.3

*2025-09-18* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.13.3)

### New Features

- Added the `namespaces=` parameter to [yaml_interrogate()](./reference/yaml_interrogate.md#pointblank.yaml_interrogate) to facilitate authoring of custom actions. ([@mark-druffel](https://github.com/mark-druffel), [\#277](https://github.com/posit-dev/pointblank/issues/277))

### Fixes

- Fixed error when using `brief=` with `segments=seg_group()`. ([\#275](https://github.com/posit-dev/pointblank/issues/275))

### New Contributors

- [@mark-druffel](https://github.com/mark-druffel) made their first contribution in https://github.com/posit-dev/pointblank/pull/277

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.13.2…v0.13.3](https://github.com/posit-dev/pointblank/compare/v0.13.2...v0.13.3)

## v0.13.2

*2025-09-05* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.13.2)

### New Features

- String-based comparisons are now possible in the `col_vals_eq()` and `col_vals_ne()` validation methods. ([\#272](https://github.com/posit-dev/pointblank/issues/272))

### Fixes

- We can now enable passing of `NaN` values during interrogation when `na_pass=True`. ([\#271](https://github.com/posit-dev/pointblank/issues/271))
- A `col_vals_expr()`-based step can now be used with the [get_step_report()](./reference/Validate.get_step_report.md#pointblank.Validate.get_step_report) method. ([\#273](https://github.com/posit-dev/pointblank/issues/273))

### Chores

- Fixed all CI test failures and warnings. ([\#270](https://github.com/posit-dev/pointblank/issues/270))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.13.1…v0.13.2](https://github.com/posit-dev/pointblank/compare/v0.13.1...v0.13.2)

## v0.13.1

*2025-08-29* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.13.1)

### Fixes

- Several MCP server issues have been fixed. ([\#265](https://github.com/posit-dev/pointblank/issues/265))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.13.0…v0.13.1](https://github.com/posit-dev/pointblank/compare/v0.13.0...v0.13.1)

## v0.13.0

*2025-08-28* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.13.0)

### New Features

- Incorporated a Model Context Protocol (MCP) server implementation. ([\#239](https://github.com/posit-dev/pointblank/issues/239), [\#264](https://github.com/posit-dev/pointblank/issues/264), [@pipaber](https://github.com/pipaber))

### Docs

- New installation instructions for adding Pointblank to pixi projects. ([\#263](https://github.com/posit-dev/pointblank/issues/263), [@gregorywaynepower](https://github.com/gregorywaynepower))

### Chores

- Added Makefile pre-commit targets. ([\#262](https://github.com/posit-dev/pointblank/issues/262), [@tylerriccio33](https://github.com/tylerriccio33))
- Reorganized tests and added new ones. ([\#261](https://github.com/posit-dev/pointblank/issues/261))

### New Contributors

- [@pipaber](https://github.com/pipaber) made their first contribution in https://github.com/posit-dev/pointblank/pull/239

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.12.2…v0.13.0](https://github.com/posit-dev/pointblank/compare/v0.12.2...v0.13.0)

## v0.12.2

*2025-08-16* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.12.2)

### New Features

- Added the [set_tbl()](./reference/Validate.set_tbl.md#pointblank.Validate.set_tbl) method and also the `set_tbl=` parameter to [yaml_interrogate()](./reference/yaml_interrogate.md#pointblank.yaml_interrogate). ([\#260](https://github.com/posit-dev/pointblank/issues/260))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.12.1…v0.12.2](https://github.com/posit-dev/pointblank/compare/v0.12.1...v0.12.2)

## v0.12.1

*2025-08-11* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.12.1)

### New Features

- Pointblank now internally handles all Ibis tables using Narwhals. ([\#257](https://github.com/posit-dev/pointblank/issues/257))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.12.0…v0.12.1](https://github.com/posit-dev/pointblank/compare/v0.12.0...v0.12.1)

## v0.12.0

*2025-08-07* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.12.0)

### New Features

- The segmentation feature was expanded by way of the new [seg_group()](./reference/seg_group.md#pointblank.seg_group) helper function. ([\#243](https://github.com/posit-dev/pointblank/issues/243))
- You can now validate Spark DataFrames without using Ibis (we now internally process those using Narwhals). ([\#256](https://github.com/posit-dev/pointblank/issues/256))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.6…v0.12.0](https://github.com/posit-dev/pointblank/compare/v0.11.6...v0.12.0)

## v0.11.6

*2025-07-29* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.6)

### Docs

- Enhanced YAML documentation by creating two new articles for the User Guide. ([\#249](https://github.com/posit-dev/pointblank/issues/249))

### Fixes

- Specify `ignore_nulls=True` in `nw.all_horizontal`. ([@MarcoGorelli](https://github.com/MarcoGorelli), [\#251](https://github.com/posit-dev/pointblank/issues/251))

### New Contributors

- [@MarcoGorelli](https://github.com/MarcoGorelli) made their first contribution in https://github.com/posit-dev/pointblank/pull/251

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.5…v0.11.6](https://github.com/posit-dev/pointblank/compare/v0.11.5...v0.11.6)

## v0.11.5

*2025-07-24* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.5)

- Added `pyyaml` as a dependency.

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.4…v0.11.5](https://github.com/posit-dev/pointblank/compare/v0.11.4...v0.11.5)

## v0.11.4

*2025-07-23* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.4)

### New Features

- It's now possible to define YAML configuration files for validation workflows and run the YAML validations through [yaml_interrogate()](./reference/yaml_interrogate.md#pointblank.yaml_interrogate) or in the Pointblank CLI via `pb run` ([\#247](https://github.com/posit-dev/pointblank/issues/247), [\#248](https://github.com/posit-dev/pointblank/issues/248))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.3…v0.11.4](https://github.com/posit-dev/pointblank/compare/v0.11.3...v0.11.4)

## v0.11.3

*2025-07-18* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.3)

### New Features

- The CLI has the new `pb pl` command, allowing you to run Polars at the command line. ([\#246](https://github.com/posit-dev/pointblank/issues/246))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.2…v0.11.3](https://github.com/posit-dev/pointblank/compare/v0.11.2...v0.11.3)

## v0.11.2

*2025-07-08* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.2)

### Fixes

- The display of the CLI's `pb run` output is much improved and more closely follows the tabular HTML output of the Python API. ([\#240](https://github.com/posit-dev/pointblank/issues/240))
- For the CLI's `pb validate` and `pb run` commands, the extract limit is now correctly applied to written CSVs (while displayed extracts will always have a limit of 10). ([\#242](https://github.com/posit-dev/pointblank/issues/242))

### Documentation

- Improved the CLI articles in the User Guide. ([\#236](https://github.com/posit-dev/pointblank/issues/236))
- Updated the CLI demonstration gif for the project website. ([\#241](https://github.com/posit-dev/pointblank/issues/241))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.1…v0.11.2](https://github.com/posit-dev/pointblank/compare/v0.11.1...v0.11.2)

## v0.11.1

*2025-06-26* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.1)

### New Features

- Have consistent input ingestion for all classes/functions with a `data=` parameter. ([\#223](https://github.com/posit-dev/pointblank/issues/223))
- Enable data to be obtained via GitHub URLs. ([\#224](https://github.com/posit-dev/pointblank/issues/224))
- Allow CLI's `pb validate` to perform multiple validations. ([\#231](https://github.com/posit-dev/pointblank/issues/231))
- For CLI's `pb validate` allow `--column` to use column index. ([\#233](https://github.com/posit-dev/pointblank/issues/233))

### Fixes

- Use `_process_data()` to centralize data ingest functionality. ([\#225](https://github.com/posit-dev/pointblank/issues/225), [\#226](https://github.com/posit-dev/pointblank/issues/226))
- Improve display of tables in CLI. ([\#227](https://github.com/posit-dev/pointblank/issues/227))
- Simplify use of extracts in CLI. ([\#228](https://github.com/posit-dev/pointblank/issues/228))
- Rework `pb validate` to primarily be a validation script runner. ([\#229](https://github.com/posit-dev/pointblank/issues/229))
- Make CLI table styling consistent across all commands. ([\#234](https://github.com/posit-dev/pointblank/issues/234))
- Improve messages with `pb validate <data>` default and add `--list-checks` option. ([\#230](https://github.com/posit-dev/pointblank/issues/230))

### Documentation

- Update VHS-based terminal recordings. ([\#232](https://github.com/posit-dev/pointblank/issues/232))
- Create two new CLI utility articles. ([\#235](https://github.com/posit-dev/pointblank/issues/235))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.11.0…v0.11.1](https://github.com/posit-dev/pointblank/compare/v0.11.0...v0.11.1)

## v0.11.0

*2025-06-20* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.11.0)

### New Features

- The `pb` CLI utility has been added to enable: (1) quick data quality checks, (2) exploration of data at the command line, and (3) easy integration with shell scripts and automation workflows. ([\#221](https://github.com/posit-dev/pointblank/issues/221))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.10.0…v0.11.0](https://github.com/posit-dev/pointblank/compare/v0.10.0...v0.11.0)

## v0.10.0

*2025-06-18* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.10.0)

### New Features

- It's now possible to validate Polars LazyFrames. ([\#200](https://github.com/posit-dev/pointblank/issues/200))
- Paths to CSV and Parquet files can now be used as inputs for [Validate](./reference/Validate.md#pointblank.Validate) and [preview()](./reference/preview.md#pointblank.preview). ([\#213](https://github.com/posit-dev/pointblank/issues/213), [\#214](https://github.com/posit-dev/pointblank/issues/214), [\#217](https://github.com/posit-dev/pointblank/issues/217))
- The [get_data_path()](./reference/get_data_path.md#pointblank.get_data_path) function was added so that paths to internal CSV and Parquet example datasets can be accessed. ([\#215](https://github.com/posit-dev/pointblank/issues/215))
- Data connection strings can be used directly with [Validate](./reference/Validate.md#pointblank.Validate) to connect to DB tables via Ibis. ([\#216](https://github.com/posit-dev/pointblank/issues/216))

### Fixes

- The [DataScan](./reference/DataScan.md#pointblank.DataScan) class was refactored to expose data and statistics consistently. ([@tylerriccio33](https://github.com/tylerriccio33), [\#94](https://github.com/posit-dev/pointblank/issues/94))
- Pass/fail result counting during interrogation is now more computationally efficient. ([\#203](https://github.com/posit-dev/pointblank/issues/203))
- Validation steps using `col_vals_expr()` can now correctly return data extracts (and produce a CSV button in the validation report table). ([@zilto](https://github.com/zilto), [\#197](https://github.com/posit-dev/pointblank/issues/197))
- A dependency on Pandas during the rendering validation report tables was eliminated. ([\#220](https://github.com/posit-dev/pointblank/issues/220))
- An unwanted scrollbar in the API reference pages (obscuring text for parameters) was removed. ([@matt-humphrey](https://github.com/matt-humphrey), [\#218](https://github.com/posit-dev/pointblank/issues/218))

### Documentation

- The project website's API reference underwent several layout and typesetting changes for the better. ([\#204](https://github.com/posit-dev/pointblank/issues/204), [\#205](https://github.com/posit-dev/pointblank/issues/205), [\#208](https://github.com/posit-dev/pointblank/issues/208), [\#209](https://github.com/posit-dev/pointblank/issues/209))
- Terminology throughout the documentation was improved so that users can better distinguish column- and row-based validation methods. ([\#199](https://github.com/posit-dev/pointblank/issues/199))
- A few more Pointblank demos were added to the project website's Examples page. ([\#210](https://github.com/posit-dev/pointblank/issues/210))
- Several posts were added to the Pointblog. ([\#177](https://github.com/posit-dev/pointblank/issues/177), [\#202](https://github.com/posit-dev/pointblank/issues/202))
- Improved the interlinking in the User Guide thanks to new functionality in quartodoc. ([\#191](https://github.com/posit-dev/pointblank/issues/191), [\#212](https://github.com/posit-dev/pointblank/issues/212))

### New Contributors

- [@zilto](https://github.com/zilto) made their first contribution in https://github.com/posit-dev/pointblank/pull/197
- [@matt-humphrey](https://github.com/matt-humphrey) made their first contribution in https://github.com/posit-dev/pointblank/pull/218

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.6…v0.10.0](https://github.com/posit-dev/pointblank/compare/v0.9.6...v0.10.0)

## v0.9.6

*2025-05-23* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.6)

### Minor Improvements and bug Fixes

- Added the [above_threshold()](./reference/Validate.above_threshold.md#pointblank.Validate.above_threshold) method to determine whether steps exceeded a specific threshold level. ([\#184](https://github.com/posit-dev/pointblank/issues/184))
- There's now support for BigQuery Ibis-backend tables. ([\#190](https://github.com/posit-dev/pointblank/issues/190))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.5…v0.9.6](https://github.com/posit-dev/pointblank/compare/v0.9.5...v0.9.6)

## v0.9.5

*2025-05-20* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.5)

### Minor improvements and bug fixes

- Added the [assert_below_threshold()](./reference/Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold) method to raise an `AssertionError` if validation steps exceed a specified threshold level. ([\#183](https://github.com/posit-dev/pointblank/issues/183))
- We now allow [assert_passing()](./reference/Validate.assert_passing.md#pointblank.Validate.assert_passing) to use [interrogate()](./reference/Validate.interrogate.md#pointblank.Validate.interrogate) when needed. ([\#182](https://github.com/posit-dev/pointblank/issues/182))
- The printing of a [Schema](./reference/Schema.md#pointblank.Schema) object no longer errors if a column doesn't have a declared data type. ([\#181](https://github.com/posit-dev/pointblank/issues/181))

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.4…v0.9.5](https://github.com/posit-dev/pointblank/compare/v0.9.4...v0.9.5)

## v0.9.4

*2025-05-08* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.4)

### New Features and Fixes

- Add the `global_sales` dataset, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/174
- Use Python 3.8 and 3.9 compatible type aliases, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/175
- Incorporate templating variables for segments in `brief=`, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/176

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.2…v0.9.4](https://github.com/posit-dev/pointblank/compare/v0.9.2...v0.9.4)

## v0.9.2

*2025-05-06* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.2)

### What's Changed

- Added the [specially()](./reference/Validate.specially.md#pointblank.Validate.specially) validation method, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/172

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.1…v0.9.2](https://github.com/posit-dev/pointblank/compare/v0.9.1...v0.9.2)

## v0.9.1

*2025-05-02* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.1)

### New Features

- Added the `rows_complete()` validation method, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/171

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.9.0…v0.9.1](https://github.com/posit-dev/pointblank/compare/v0.9.0...v0.9.1)

## v0.9.0

*2025-04-28* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.9.0)

### New Features

- There's now support for data segmentation within a wide variety of validation methods (with the new `segments=` argument), by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/170
- Implemented parallelization of tests to decrease the overall duration of tests, by [@tylerriccio33](https://github.com/tylerriccio33) in https://github.com/posit-dev/pointblank/pull/165

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.7…v0.9.0](https://github.com/posit-dev/pointblank/compare/v0.8.7...v0.9.0)

## v0.8.7

*2025-04-21* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.7)

### New Features

- Allow `None` in `col_vals_in_set()` by [@tylerriccio33](https://github.com/tylerriccio33), https://github.com/posit-dev/pointblank/pull/162

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.6…v0.8.7](https://github.com/posit-dev/pointblank/compare/v0.8.6...v0.8.7)

## v0.8.6

*2025-04-16* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.6)

### New Features

- Added the `conjointly()` validation method, by [@rich-iannone](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/159, https://github.com/posit-dev/pointblank/pull/160)

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.5…v0.8.6](https://github.com/posit-dev/pointblank/compare/v0.8.5...v0.8.6)

## v0.8.5

*2025-04-15* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.5)

### New Features

- Added step report functionality for `rows_distinct()` validation steps, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/157

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.4…v0.8.5](https://github.com/posit-dev/pointblank/compare/v0.8.4...v0.8.5)

## v0.8.4

*2025-04-12* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.4)

### New Features

- All step reports are now translated according to the `Validate(lang=)` value, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/155

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.3…v0.8.4](https://github.com/posit-dev/pointblank/compare/v0.8.3...v0.8.4)

## v0.8.3

*2025-04-10* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.3)

### New Features

- Added Arabic, Hindi, and Greek translations for automatically-generated briefs and for the validation report table, by [@rich-iannone](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/150, https://github.com/posit-dev/pointblank/pull/151, https://github.com/posit-dev/pointblank/pull/152)

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.2…v0.8.3](https://github.com/posit-dev/pointblank/compare/v0.8.2...v0.8.3)

## v0.8.2

*2025-04-09* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.2)

### New Features and Fixes

- Added the [send_slack_notification()](./reference/send_slack_notification.md#pointblank.send_slack_notification) function for creating a Slack notification action, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/147
- Updated the `get_api_text()` utility function, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/148

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.1…v0.8.2](https://github.com/posit-dev/pointblank/compare/v0.8.1...v0.8.2)

## v0.8.1

*2025-04-08* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.1)

### New Features and Fixes

- Added the ability to define final actions for validation (post interrogation), by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/146
- Refined translations for the `zh-Hant` lang, by [@jrycw](https://github.com/jrycw) in https://github.com/posit-dev/pointblank/pull/143

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.8.0…v0.8.1](https://github.com/posit-dev/pointblank/compare/v0.8.0...v0.8.1)

## v0.8.0

*2025-04-03* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.8.0)

### New Features

- Added `highest_level=` and `default=` parameters to the [Actions](./reference/Actions.md#pointblank.Actions) class, by [@rich-iannone](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/124, https://github.com/posit-dev/pointblank/pull/126)
- The `brief=` text now appears in the validation report table, plus we enabled more templating features for `brief=`, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/127
- Add the global setting option for `brief=` in [Validate](./reference/Validate.md#pointblank.Validate), by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/128
- We now allow for flexible validations using dates or datetimes, by [@rich-iannone](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/134, https://github.com/posit-dev/pointblank/pull/135, https://github.com/posit-dev/pointblank/pull/136)
- The `lang=` value in [Validate](./reference/Validate.md#pointblank.Validate) now translates the validation report table to `lang=`'s spoken language, and several more language translations were added, by [@rich-iannone](https://github.com/rich-iannone) (https://github.com/posit-dev/pointblank/pull/131, https://github.com/posit-dev/pointblank/pull/138, https://github.com/posit-dev/pointblank/pull/139)
- We can now better customize the header in step reports through `.get_step_report(header=...)`, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/142

### Fixes and Documentation

- There's now a configurable limit (with set default) on extract rows (no matter which scheme was used for their collection), by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/130
- Renamed internal references to threshold levels, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/140
- Enhanced the docs for [Validate](./reference/Validate.md#pointblank.Validate) and the validation methods, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/137

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.7.3…v0.8.0](https://github.com/posit-dev/pointblank/compare/v0.7.3...v0.8.0)

## v0.7.3

*2025-03-26* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.7.3)

### New Features

- Added the `limit=` parameter to [get_step_report()](./reference/Validate.get_step_report.md#pointblank.Validate.get_step_report), by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/115
- The new `min_tbl_width=` arg in [preview()](./reference/preview.md#pointblank.preview) can improve the display of narrow preview tables by default, and, allows for customization, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/117
- Added the [get_action_metadata()](./reference/get_action_metadata.md#pointblank.get_action_metadata) function to help users make more powerful [Actions](./reference/Actions.md#pointblank.Actions) callables, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/118
- We now include failure text in the `_ValidationInfo` object after interrogation, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/119
- Basic customization of the header in [get_step_report()](./reference/Validate.get_step_report.md#pointblank.Validate.get_step_report) is now possible, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/120
- Added the ability to select a subset of columns in row-based step reports, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/123

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.7.2…v0.7.3](https://github.com/posit-dev/pointblank/compare/v0.7.2...v0.7.3)

## v0.7.2

*2025-03-24* · [GitHub](https://github.com/posit-dev/pointblank/releases/tag/v0.7.2)

### New Features and Fixes

- Pointblank now supports the Traditional Chinese (`zh-Hant`) locale (for localization of autobriefs), by [@jrycw](https://github.com/jrycw) in https://github.com/posit-dev/pointblank/pull/109
- Allow validations to work with dates/datetime comparisons across columns, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/110
- Revised [preview()](./reference/preview.md#pointblank.preview) for better operability with the Ibis PySpark backend, by [@rich-iannone](https://github.com/rich-iannone) in https://github.com/posit-dev/pointblank/pull/112

### New Contributors

- [@jrycw](https://github.com/jrycw) made their first contribution in https://github.com/posit-dev/pointblank/pull/109

**Full Changelog**: [https://github.com/posit-dev/pointblank/compare/v0.7.1…v0.7.2](https://github.com/posit-dev/pointblank/compare/v0.7.1...v0.7.2)
