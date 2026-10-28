# Glossary

This page explains the terms used throughout the Pointblank documentation, in alphabetical order. Where a term matches something in Pointblank's API, the name of the class, function, or method is shown in `code`. Terms in **bold** have their own entry on this page.

Action  
Something Pointblank does when a step's failures reach a **threshold** level, such as printing a message or running a function you supply. You set actions with the [Actions](../../reference/Actions.md#pointblank.Actions) class, either for the whole validation (`Validate(actions=...)`) or for a single step. Messages can include placeholders like `{step}`, `{col}`, and `{level}`, which are filled in when the action runs. By default, only the action for the most serious level reached is run. Inside a function used as an action, [get_action_metadata()](../../reference/get_action_metadata.md#pointblank.get_action_metadata) gives details such as the step number, the column, and the level reached. See also **final action**. [Learn more](../validation-plan/actions.md).

Active / inactive step  
Whether a **validation step** runs. A step with `active=False` is skipped, but it still appears in the **validation report** so that step numbers don't shift. You can also give `active=` a function that decides at run time. The helpers [has_columns()](../../reference/has_columns.md#pointblank.has_columns) and [has_rows()](../../reference/has_rows.md#pointblank.has_rows) make this easy: for example, `active=pb.has_columns("email")` runs the step only if the table has an `email` column.

ADaM (Analysis Data Model)  
A **CDISC** standard for the analysis-ready datasets used in clinical trials, which are built from **SDTM** data. Common examples are ADSL (one row per trial subject) and ADAE (adverse events). Pointblank can check that an ADaM dataset has the columns the standard requires, using [validate_adam_structure()](../../reference/validate_adam_structure.md#pointblank.validate_adam_structure) or [validate_adam()](../../reference/validate_adam.md#pointblank.validate_adam). [Learn more](../metadata-import/cdisc-validation.md).

Aggregate check  
A check on a summary of a whole column, such as its sum, average, or standard deviation, instead of on each value. Examples include [col_sum_gt()](../../reference/Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt), [col_avg_eq()](../../reference/Validate.col_avg_eq.md#pointblank.Validate.col_avg_eq), and [col_sd_lt()](../../reference/Validate.col_sd_lt.md#pointblank.Validate.col_sd_lt). Because the whole column is summarized into one number, an aggregate check has a single **test unit**. Use `tol=` to allow some leeway (see **tolerance**), and set **reference data** to compare with the same summary of another table. [Learn more](../validation-plan/validation-methods.md).

AI check  
A **validation step** where a large language model (LLM) reads each row and judges whether it follows a rule you describe in plain words. This helps with rules that are hard to write as code, like "the description should mention the product's color". Added with [prompt()](../../reference/Validate.prompt.md#pointblank.Validate.prompt). [Learn more](../validation-plan/validation-methods.md).

AI Validation Editor  
A tool that changes an existing **validation plan** based on a plain-English instruction, such as "also check that the email column has no missing values". It uses an LLM to write the revised plan. With the [EditValidation](../../reference/EditValidation.md#pointblank.EditValidation) class you can see what changed ([diff()](../../reference/EditValidation.md#pointblank.EditValidation.diff) or [review()](../../reference/EditValidation.md#pointblank.EditValidation.review)) and then get the new plan with [accept()](../../reference/EditValidation.md#pointblank.EditValidation.accept). It's also available from the **CLI** as `pb edit`. See also **DraftValidation**. [Learn more](../advanced-validation/ai-validation-editor.md).

Assertion  
A way to stop your code when data doesn't pass validation. After building a plan, call [assert_passing()](../../reference/Validate.assert_passing.md#pointblank.Validate.assert_passing) to raise an error if any **test unit** failed, or [assert_below_threshold()](../../reference/Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold) to raise an error if a **threshold** level was reached. Both run **interrogation** first if it hasn't happened yet. This is handy in data pipelines and tests. There's also [assert_dimension_scores()](../../reference/Validate.assert_dimension_scores.md#pointblank.Validate.assert_dimension_scores), which raises an error if the score for any **quality dimension** is below a minimum you set. If you'd rather get `True` or `False` than an error, use a **status check**. [Learn more](../advanced-validation/assertions.md).

Assistant  
A chat window where you can ask an LLM questions about your data and about Pointblank. You start it with `assistant()`, and it opens in your web browser or in the terminal. If you give it a table, a summary of that table is shared with the model so its answers fit your data.

Backend  
The library or database that holds your table. Pointblank works with Polars, Pandas, and PySpark tables, and (through the Ibis library) with databases such as DuckDB, PostgreSQL, MySQL, SQLite, SQL Server, Snowflake, Databricks, and BigQuery. It can also read CSV and Parquet files directly, and database tables given as a **connection string**. You don't need to tell Pointblank which backend you're using, since it works this out from the data you pass in. See also **Narwhals**. [Learn more](../getting-started/installation.md).

Boundary enforcement  
See **pipeline**.

Brief  
A short description of what a **validation step** checks, shown in the STEP column of the **validation report**. You can write your own (`brief="..."`) or let Pointblank write one for you (`brief=True`). Briefs can include placeholders such as `{col}` (the column name), `{step}` (the step number), and `{auto}` (Pointblank's generated description). Generated briefs use the **report language**. [Learn more](../validation-plan/briefs.md).

CDISC (Clinical Data Interchange Standards Consortium)  
The organization that sets data standards for clinical trials, including **SDTM**, **ADaM**, and **Define-XML**. Pointblank can read CDISC metadata, check that datasets have the expected structure, and check a set of trial datasets against CDISC's rules. It can do that last part with its own built-in rules ([validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig)) or by running the **CORE** engine. [Learn more](../metadata-import/cdisc-validation.md).

CLI (command-line interface)  
The `pb` command, which you run in a terminal to use Pointblank without writing Python. `pb preview`, `pb scan`, `pb missing`, and `pb info` let you look at a table. `pb validate` runs quick one-off checks, and `pb run` runs a full validation written in Python or YAML. `pb compare` runs a **table comparison**, `pb make-template` creates a starter file, and `pb edit` opens the **AI Validation Editor**. See also **exit code**. [Learn more](../the-pointblank-cli/cli-reference.md).

Coded missing value  
A stand-in value, such as `-99` or `"UNK"`, that means a value is missing for a known reason (for example, the person refused to answer or wasn't asked). The [MissingSpec](../../reference/MissingSpec.md#pointblank.MissingSpec) class lists which codes a column uses and what each one means. Validation methods like [col_missing_coded()](../../reference/Validate.col_missing_coded.md#pointblank.Validate.col_missing_coded) and the **missing values table** can then treat these codes as missing data. See also **null flavor**. [Learn more](../data-inspection/missing-vals-tbl.md).

Codelist  
A named list of allowed values, often with a label for each (for example, `"M"` means Male and `"F"` means Female). Codelists usually come from a **metadata import**, such as CDISC **controlled terminology** or SPSS value labels. Pointblank represents them with the [Codelist](../../reference/Codelist.md#pointblank.Codelist) class. [Learn more](../metadata-import/metadata-import.md).

Column comparison  
Checking each value in a column against the value of another column in the same row, rather than against a fixed value. Wrap the other column's name in [col()](../../reference/col.md#pointblank.col): for example, `col_vals_lt(columns="a", value=pb.col("c"))` checks that each value of `a` is less than the value of `c` in the same row. [Learn more](../validation-plan/validation-methods.md).

Column selection  
How you tell a **validation step** which columns to check. You can give a single column name, a list of names, or a helper that picks columns for you: [starts_with()](../../reference/starts_with.md#pointblank.starts_with), [ends_with()](../../reference/ends_with.md#pointblank.ends_with), [contains()](../../reference/contains.md#pointblank.contains), [`matches()`](../../reference/matches.md#pointblank.matches), [everything()](../../reference/everything.md#pointblank.everything), [first_n()](../../reference/first_n.md#pointblank.first_n), or [last_n()](../../reference/last_n.md#pointblank.last_n). Helpers can be combined inside [col()](../../reference/col.md#pointblank.col) using `&` (and), `|` (or), and `-` (but not). Each column that's picked gets its own step. [Learn more](../validation-plan/column-selection-patterns.md).

Column summary table  
A table that describes each column of your data: its type, how many values are missing, how many are distinct, and statistics like the mean and quartiles. You make one with [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl). See also **DataScan**, **preview**, and **missing values table**. [Learn more](../data-inspection/col-summary-tbl.md).

Comparison  
See **table comparison**.

Composable steps  
A reusable set of **validation steps** that isn't tied to any particular table. You build the set with the [Steps](../../reference/Steps.md#pointblank.Steps) class, using the same method names as [Validate](../../reference/Validate.md#pointblank.Validate), and add it to any validation with `Validate.add_steps()`. This lets you write common checks once and reuse them for many tables. [Learn more](../advanced-validation/composable-steps.md).

Conformance report  
The results of checking a set of clinical trial datasets against **CDISC** rules, given by [validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig) and [validate_cdisc_submission()](../../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission) as a [ConformanceReport](../../reference/ConformanceReport.md#pointblank.ConformanceReport). Its table shows one row per rule, with a status of pass, fail, error, not applicable, or not supported. The *findings* (one for each record that broke a rule, with the column and value at fault) are available as a table with [get_findings_table()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.get_findings_table) or as a DataFrame with [findings_df()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.findings_df). [Learn more](../metadata-import/cdisc-submission-conformance.md).

Conjointly  
A single **validation step** that checks several conditions on each row at once. A row passes only if every condition is true for that row. This is different from adding one step per condition, where each condition is judged on its own. Added with [conjointly()](../../reference/Validate.conjointly.md#pointblank.Validate.conjointly). [Learn more](../validation-plan/validation-methods.md).

Connection string  
A piece of text that points to a database table, such as `"duckdb:///path/to/db.ddb::table_name"`. The part before `::` says how to reach the database, and the part after it names the table. You can pass a connection string as the data for [Validate](../../reference/Validate.md#pointblank.Validate) or for **CLI** commands, instead of a table you've already loaded. [Learn more](../the-pointblank-cli/cli-reference.md).

Contract  
A written-down agreement about what a table should look like. It combines a **schema** (the expected columns and their types), a list of rules, and details like the owner and version. You create one with the [Contract](../../reference/Contract.md#pointblank.Contract) class and check a table against it with `.validate(data)`. Contracts can be saved as YAML, imported from other formats (such as JSON Schema, Frictionless, dbt, and ODCS) with [`import_contract()`](../../reference/import_contract.md#pointblank.import_contract), and exported with [`export_contract()`](../../reference/export_contract.md#pointblank.export_contract). See also **pipeline**, **contract import**, and **violation handling**. [Learn more](../contracts-and-pipelines/contracts.md).

Contract adapter  
The part of Pointblank that translates between a **contract** and one outside format. There are built-in adapters for JSON Schema, Frictionless, dbt, and ODCS, and [list_adapters()](../../reference/list_adapters.md#pointblank.list_adapters) shows which ones are available. For another format, you can write your own adapter by making a subclass of [ContractAdapter](../../reference/ContractAdapter.md#pointblank.ContractAdapter) and registering it with `@register_adapter`. [`import_contract()`](../../reference/import_contract.md#pointblank.import_contract) and [`export_contract()`](../../reference/export_contract.md#pointblank.export_contract) can then use it like any other. [Learn more](../contracts-and-pipelines/custom-adapters.md).

Contract import  
What you get back from [`import_contract()`](../../reference/import_contract.md#pointblank.import_contract): a [ContractImport](../../reference/ContractImport.md#pointblank.ContractImport) object holding the checks that were translated from an outside format. Nothing is checked yet. Its `coverage` is the share of the outside rules that could be translated, and `warnings` lists any that couldn't. From there, `to_validate(data)` gives a **validation plan**, [to_contract()](../../reference/ContractImport.md#pointblank.ContractImport.to_contract) gives a **contract**, and [to_python()](../../reference/ContractImport.md#pointblank.ContractImport.to_python) or [`to_yaml()`](../../reference/ContractImport.md#pointblank.ContractImport.to_yaml) give code. [Learn more](../contracts-and-pipelines/importing-contracts.md).

Controlled terminology (CT)  
CDISC's published lists of the allowed values for clinical trial data, such as the codes for sex, race, or units of measure. They're released as dated packages a few times a year, so it's good practice to use the same version your study was set up with. `import_metadata(..., format="cdisc_ct")` reads a package as a set of **codelists**, and [validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig) checks values against a package that comes with Pointblank (or one you pick with `ct_packages=`). Some codelists are *extensible*, meaning a study may add its own values. [Learn more](../metadata-import/cdisc-validation.md).

CORE (CDISC Conformance Rules Engine)  
CDISC's official, open-source program for checking clinical trial data against CDISC's rules. You install it separately, and [validate_cdisc_submission()](../../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission) runs it for you and turns its output into a [ConformanceReport](../../reference/ConformanceReport.md#pointblank.ConformanceReport). Pointblank also has built-in rules that don't need CORE (see **CDISC**). [Learn more](../metadata-import/cdisc-submission-conformance.md).

Critical  
See **warning, error, and critical**.

Custom check  
A **validation step** that uses a function you write, for rules that none of the built-in validation methods cover. Your function receives the table and returns `True` or `False` values showing what passed. Added with [specially()](../../reference/Validate.specially.md#pointblank.Validate.specially). [Learn more](../validation-plan/validation-methods.md).

Data coherence  
In **test data generation**, the way related columns are made to fit together within each row. For example, an email address is made from the person's name, and the address, city, postcode, phone number, and coordinates all belong to the same place. This happens on its own when related **presets** are used together, and it still holds when rows from several countries are mixed. [Learn more](../test-data-generation/test-data-generation.md).

Data extract  
The rows that failed a **validation step**, saved during **interrogation** so you can look at them afterwards. You get them with [Validate.get_data_extracts()](../../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts). Only steps that check values row by row produce extracts. Options on [interrogate()](../../reference/Validate.interrogate.md#pointblank.Validate.interrogate), such as `extract_limit=`, control how many rows are kept. See also **sundering**. [Learn more](../post-interrogation/extracts.md).

Data freshness  
Whether a table has recent data. [data_freshness()](../../reference/Validate.data_freshness.md#pointblank.Validate.data_freshness) checks that the latest date or time in a column is no older than `max_age=`, given as text like `"6h"`, `"2d"`, or `"1d 12h"`. The **validation report** shows how old the data was, which helps tell a small delay from a pipeline that has stopped. [Learn more](../validation-plan/validation-methods.md).

DataScan  
An object that holds a column-by-column profile of a table, with statistics like counts, missing values, and averages. It's what the **column summary table** is built from. You can save a [DataScan](../../reference/DataScan.md#pointblank.DataScan) to a JSON file, load it again later, and compare two scans to spot **drift**. [Learn more](../data-inspection/col-summary-tbl.md).

Define-XML  
A **CDISC** file format (written in XML) that describes a clinical trial dataset: its column names, labels, types, and allowed values. Pointblank can read Define-XML files with [import_metadata()](../../reference/import_metadata.md#pointblank.import_metadata) and can pass them to **CORE**. [Learn more](../metadata-import/cdisc-validation.md).

Direction  
See **source and target**.

Domain  
See **SDTM**.

DraftValidation  
A tool that uses an LLM to write a first draft of a **validation plan** for your table. It sends the model a summary of your table (column types, counts, and basic statistics) rather than the full data, and gives back Python code for you to review and adjust. Available through the [DraftValidation](../../reference/DraftValidation.md#pointblank.DraftValidation) class. See also **AI Validation Editor**. [Learn more](../advanced-validation/draft-validation.md).

Drift  
A change in a table between two points in time, like a column being added or removed, or the typical values in a column shifting. You can look for drift by comparing two **DataScan** objects with [DataScan.compare()](../../reference/DataScan.compare.md#pointblank.DataScan.compare). It returns a [DataScanDiff](../../reference/DataScanDiff.md#pointblank.DataScanDiff), which lists the columns that were added, removed, or changed, along with **drift scores**. [Learn more](../data-inspection/col-summary-tbl.md).

Drift score  
A number that measures how far a column's values have shifted between two **DataScan** objects. [DataScan.compare()](../../reference/DataScan.compare.md#pointblank.DataScan.compare) gives two kinds. The PSI (Population Stability Index) works for numeric and categorical columns: under 0.1 means little change, and over 0.25 means a large shift. The KS (Kolmogorov-Smirnov) test works for numeric columns, and a small p-value (say, under 0.05) means the two sets of values differ. Drift scores need the original data, so they can't be worked out for a scan loaded from a JSON file. [Learn more](../data-inspection/col-summary-tbl.md).

Error  
See **warning, error, and critical**.

Exit code  
The number a command gives back when it finishes, which scripts and CI systems use to tell whether it succeeded (`0` means success). `pb validate --exit-code` gives a non-zero code if any check fails, and `pb run --fail-on` does so when a chosen **threshold** level is reached. `pb compare` gives `0` if the tables match, `1` if they differ, and `2` if there was an error. [Learn more](../the-pointblank-cli/cli-reference.md).

Expression check  
A **validation step** that checks each row against a condition written in your table library's own syntax. For example, with Polars you might check `pl.col("end") > pl.col("start")`. Added with [col_vals_expr()](../../reference/Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr). [Learn more](../advanced-validation/expressions.md).

Field  
A description of one column for **test data generation**, made with a helper such as [int_field()](../../reference/int_field.md#pointblank.int_field), [float_field()](../../reference/float_field.md#pointblank.float_field), [string_field()](../../reference/string_field.md#pointblank.string_field), [bool_field()](../../reference/bool_field.md#pointblank.bool_field), or [date_field()](../../reference/date_field.md#pointblank.date_field). The options say what values are allowed, for example `min_val=`, `max_val=`, `allowed=`, `pattern=`, `preset=`, or `unique=True`, depending on the type. You put fields together in a [Schema](../../reference/Schema.md#pointblank.Schema) and pass it to [generate_dataset()](../../reference/generate_dataset.md#pointblank.generate_dataset). [Learn more](../test-data-generation/test-data-generation.md).

Final action  
An **action** that runs once, after all the **validation steps** are finished, whatever the results were. Final actions are useful for sending a summary, such as a message to a chat channel ([send_slack_notification()](../../reference/send_slack_notification.md#pointblank.send_slack_notification) makes one for Slack). You set them with the [FinalActions](../../reference/FinalActions.md#pointblank.FinalActions) class. Inside a final action, [get_validation_summary()](../../reference/get_validation_summary.md#pointblank.get_validation_summary) gives you the results. [Learn more](../validation-plan/actions.md).

Global settings  
Default settings that apply to every validation in your Python session, changed with `pb.config()`. They control what the **validation report** includes (for example, the header, footer, and timings) and how the **health score** is calculated. Each call to `pb.config()` sets every option, so any option you leave out goes back to its default.

Governance metadata  
Details about who looks after a validation and who relies on it, given with `owner=`, `consumers=`, and `version=` in [Validate](../../reference/Validate.md#pointblank.Validate) (the same fields can be set in a **YAML workflow** or a **contract**). They're shown in the footer of the **validation report** and don't change how the checks run. [Learn more](../post-interrogation/validation-reports.md).

Great Tables  
The Python library Pointblank uses to make its tables, such as the **validation report**, the **step report**, and the **column summary table**. These come back as Great Tables `GT` objects, so you can show them, turn them into HTML with `as_raw_html()`, or save them as an image with `gtsave()`.

Health score  
A single number from 0 to 100 that sums up data quality: the share of all **test units** that passed. Pointblank also works out a score for each **quality dimension**. Get the scores with [get_health_score()](../../reference/Validate.get_health_score.md#pointblank.Validate.get_health_score) and [get_dimension_scores()](../../reference/Validate.get_dimension_scores.md#pointblank.Validate.get_dimension_scores), or show them with [get_scorecard()](../../reference/Validate.get_scorecard.md#pointblank.Validate.get_scorecard). They appear in the **validation report** when you use `incl_dimensions=True`. [Learn more](../post-interrogation/quality-dimensions-and-scoring.md).

Interrogation  
Running the checks in a **validation plan**, which happens when you call `.interrogate()`. Nothing is checked until then. Interrogation runs every step, counts what passed and failed, compares the counts to the **thresholds**, runs any **actions**, and saves **data extracts**. [Learn more](../validation-plan/validation-overview.md).

JSON report  
The validation results as JSON text, which other programs can read. You get it with [Validate.get_json_report()](../../reference/Validate.get_json_report.md#pointblank.Validate.get_json_report), and you can choose which details to include with `use_fields=` or leave some out with `exclude_fields=`. See also **validation report**.

MCP server  
A way for AI assistants (such as the AI chat in VS Code or Positron) to use Pointblank for you. MCP (Model Context Protocol) is a standard way for AI tools to connect to other programs. Through it, an assistant can load data, build and run validations, and show previews and summaries. [Learn more](../mcp-server/mcp-quick-start.md).

Metadata import  
Reading a description of a dataset from a file made by another tool, and turning it into Pointblank checks. [import_metadata()](../../reference/import_metadata.md#pointblank.import_metadata) reads files from SPSS, SAS, and Stata, Frictionless Data Packages, CSV on the Web (CSVW), and **CDISC** standards like **Define-XML**. You can turn the result into a validation plan with `to_validate(data)` or into a **schema** with [to_schema()](../../reference/MetadataImport.md#pointblank.MetadataImport.to_schema). [Learn more](../metadata-import/metadata-import.md).

Missing values table  
A table showing where the missing values are in your data, made with [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl). By default it splits the rows into chunks and shades each chunk by how much of each column is missing. If you describe your **coded missing values** with `missing=`, it can also break them down by reason. [Learn more](../data-inspection/missing-vals-tbl.md).

`na_pass`  
An option on many validation methods that decides whether missing values (nulls) count as passing. The default, `na_pass=False`, counts them as failures. Set `na_pass=True` when missing values are acceptable. [Learn more](../validation-plan/validation-methods.md).

Narwhals  
The library Pointblank uses to work with tables from different **backends** in the same way. It's why a validation you write once works whether your table is in Polars, Pandas, or a database. Pointblank installs it for you, so you won't usually need to use it directly.

Null flavor  
A standard code from the HL7 and **CDISC** standards that says why a value is missing. For example, `"NI"` means no information, `"UNK"` means unknown, and `"ASKU"` means asked but unknown. [MissingSpec.from_cdisc_null_flavors()](../../reference/MissingSpec.md#pointblank.MissingSpec.from_cdisc_null_flavors) sets up all 14 codes for you. See also **coded missing value**. [Learn more](../metadata-import/cdisc-validation.md).

OpenTelemetry integration  
OpenTelemetry is a widely used standard for sending measurements from programs to monitoring tools such as Grafana or Datadog. Adding [emit_otel()](../../reference/emit_otel.md#pointblank.emit_otel) as a **final action** sends numbers like pass rates and **threshold** breaches after each validation. This lets you track data quality on the same dashboards as the rest of your systems. [Learn more](../integrations/otel-integration.md).

Partitions  
A way to compare tables that are too large to fit in memory. With `partitions=N`, [`compare()`](../../reference/compare.md#pointblank.compare) (and [tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match) and [tbl_reconcile()](../../reference/Validate.tbl_reconcile.md#pointblank.Validate.tbl_reconcile)) works through the tables in N passes, each covering a share of the keys from both tables. The results are the same as in a single pass, but less memory is needed at once. More partitions use less memory but take longer. See also **table comparison**. [Learn more](../table-comparison/comparing-large-files.md).

Pipeline  
A data-processing step with **contracts** on both sides. A [Pipeline](../../reference/Pipeline.md#pointblank.Pipeline) checks the incoming data (the source), runs your transformation function, then checks the result (the target), all in one call to `run(data, transform)`. By default, if the incoming data fails, the transformation is skipped. The [PipelineResult](../../reference/PipelineResult.md#pointblank.PipelineResult) that comes back holds both validations and the transformed data, and its [passed](../../reference/PipelineResult.md#pointblank.PipelineResult.passed) is `True` only if both checks passed. Checking data at these two points is sometimes called *boundary enforcement*. See also **source and target** and **violation handling**. [Learn more](../contracts-and-pipelines/pipelines.md).

Plan serialization  
Turning a **validation plan** into text so it can be saved, shared, reviewed, or rebuilt later. [Validate.to_code()](../../reference/Validate.to_code.md#pointblank.Validate.to_code) gives Python code, [Validate.to_yaml()](../../reference/Validate.to_yaml.md#pointblank.Validate.to_yaml) gives a **YAML workflow**, and [Validate.to_json_schema()](../../reference/Validate.to_json_schema.md#pointblank.Validate.to_json_schema) gives a JSON Schema document. See also **round trip**. [Learn more](../advanced-validation/plan-serialization.md).

Preprocessing  
Changing the table for just one step before it's checked, using the `pre=` option. You give `pre=` a function that takes the table and returns a changed version, for example with some rows filtered out or a new column added. The original data and other steps are not affected. The number of **test units** is counted after the change. [Learn more](../validation-plan/preprocessing.md).

Preset  
A ready-made kind of realistic text for **test data generation**, chosen with `string_field(preset=...)`. There are presets for names, email addresses, street addresses, cities, phone numbers, company names, URLs, credit card numbers, and more. [profile_fields()](../../reference/profile_fields.md#pointblank.profile_fields) gives a whole set of person-related presets at once. Presets follow the `country=` given to [generate_dataset()](../../reference/generate_dataset.md#pointblank.generate_dataset), and related presets fit together within a row (see **data coherence**). [Learn more](../test-data-generation/test-data-generation.md).

Preview  
A quick look at a table showing the first and last few rows along with each column's type, made with [preview()](../../reference/preview.md#pointblank.preview). See also **column summary table**. [Learn more](../data-inspection/preview.md).

`python:` block  
A way to write Python code inside a **YAML workflow**, for settings that need it, such as `tbl`, `pre`, `expr`, and `actions`. For safety, the code runs in a limited environment: basic built-in functions, `Path`, `pb` (Pointblank), `pl` (Polars), and `pd` (Pandas) are available, but other modules can't be imported. [Learn more](../yaml/yaml-reference.md).

Quality dimension  
A category describing what kind of data quality a step measures. The categories are *completeness*, *validity*, *uniqueness*, *consistency*, *timeliness*, and *volume*. Each validation method has a default category (for example, [col_vals_not_null()](../../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) measures completeness), and you can change it with `dimension=`. Each dimension gets its own score, and together they make up the **health score**. [Learn more](../post-interrogation/quality-dimensions-and-scoring.md).

Reconciliation  
Checking that two tables add up to the same numbers even though their rows differ, for example that a summary table agrees with the detailed records it was built from. [tbl_reconcile()](../../reference/Validate.tbl_reconcile.md#pointblank.Validate.tbl_reconcile) groups both tables by the `keys=` columns, totals the `measures=` columns (with `agg="sum"` by default, or `"count"`, `"mean"`, `"min"`, or `"max"`), and compares the results group by group. Each group is one **test unit**. See also **table comparison**. [Learn more](../table-comparison/comparing-tables.md).

Reference data  
A second table, such as last month's data, that **aggregate checks** can compare with. You set it with `Validate(reference=...)`. When an aggregate check is given no `value=`, it compares with the same column in the reference data, so `col_sum_eq(columns="revenue", tol=50)` checks that the total revenue is within 50 of the reference table's total. To compare with a different column of the reference data, use `value=pb.ref("column_name")`. [Learn more](../validation-plan/validation-methods.md).

Referential check  
A check that every value in a column also appears in a column of another table, for example that every `customer_id` in an orders table exists in the customers table. Added with [col_vals_in_table()](../../reference/Validate.col_vals_in_table.md#pointblank.Validate.col_vals_in_table). [Learn more](../validation-plan/validation-methods.md).

Report language  
The language used for the text in reports and in generated **briefs**, set with `Validate(lang=...)`. Forty languages are available. A separate option, `locale=`, controls how numbers are formatted (for example, whether a comma or a period is used as the decimal mark). [Learn more](../validation-plan/briefs.md).

Round trip  
Turning something into another form and back again, ending up with the same thing. For example, a **validation plan** saved with [`to_yaml()`](../../reference/Validate.to_yaml.md#pointblank.Validate.to_yaml) and loaded again with [yaml_interrogate()](../../reference/yaml_interrogate.md#pointblank.yaml_interrogate), or a **contract** imported from JSON Schema and exported back. Anything that both forms can express survives the round trip. Things that only one form can express, such as a `pre=` function, are replaced with a placeholder or left out. [Learn more](../advanced-validation/plan-serialization.md).

Row-based check  
A check on whole rows rather than on single values. [rows_distinct()](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) checks that no row appears more than once, and [rows_complete()](../../reference/Validate.rows_complete.md#pointblank.Validate.rows_complete) checks that no row has missing values. Both can look at just some of the columns with `columns_subset=`. [Learn more](../validation-plan/validation-methods.md).

Row status  
The result for one row in a **table comparison**: `match` (in both tables, with equal values), `changed` (in both tables, but at least one value differs), `missing` (only in the source table), `extra` (only in the target table), or `dup_key` (its key appears more than once, so it can't be matched up). The counts are in the [status_counts](../../reference/Comparison.status_counts.md#pointblank.Comparison.status_counts) of a [Comparison](../../reference/Comparison.md#pointblank.Comparison). See also **source and target**. [Learn more](../table-comparison/comparing-tables.md).

Schema  
A description of the columns a table should have: their names and, optionally, their types. You write one with the [Schema](../../reference/Schema.md#pointblank.Schema) class and check a table against it with [col_schema_match()](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match), with options for how strict to be about column order, capital letters, and types. [schema_from_tbl()](../../reference/schema_from_tbl.md#pointblank.schema_from_tbl) builds a schema from an existing table. [Learn more](../advanced-validation/schema-validation.md).

Scorecard  
A compact table showing the **health score** and the score for each **quality dimension**, made with [Validate.get_scorecard()](../../reference/Validate.get_scorecard.md#pointblank.Validate.get_scorecard). [Learn more](../post-interrogation/quality-dimensions-and-scoring.md).

SDTM (Study Data Tabulation Model)  
A **CDISC** standard for organizing the data collected in a clinical trial into tables called *domains*, such as DM (demographics), AE (adverse events), and LB (lab results). Pointblank can check that a domain has the columns the standard requires ([validate_sdtm_structure()](../../reference/validate_sdtm_structure.md#pointblank.validate_sdtm_structure)), and it can check datasets against the SDTM rules ([validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig)). [Learn more](../metadata-import/cdisc-validation.md).

SDTMIG (SDTM Implementation Guide)  
CDISC's detailed rules for how data should be laid out in **SDTM** domains: which columns each domain needs, what values they can hold, and how the domains relate to each other. [validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig) checks a set of domains against the SDTMIG 3.4 rules, without needing **CORE**, and gives back a **conformance report**. [Learn more](../metadata-import/cdisc-submission-conformance.md).

Seed  
A number that makes randomly generated data repeatable: `generate_dataset(schema, seed=23)` gives the same rows each time with the same version of Pointblank. The [generate_dataset](../../reference/generate_dataset.md#pointblank.generate_dataset) pytest fixture picks a seed from each test's name, so every test gets its own data, which stays the same from one run to the next. [Learn more](../test-data-generation/test-data-generation.md).

Segment  
A group of rows that gets checked and reported on its own. Using `segments=` splits one step into several, one per group. For example, `segments="region"` makes one step for each region in the table, and `segments=("region", ["East", "West"])` makes steps for just those two regions. To treat several values as one group, use [seg_group()](../../reference/seg_group.md#pointblank.seg_group). [Learn more](../validation-plan/segmentation.md).

Severity level  
See **warning, error, and critical**.

Source and target  
The two sides of a **table comparison** or a **pipeline**. In a table comparison, the source is the table you start from and the target is the one checked against it. With [tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match), the table being validated is the target, and `tbl_compare=` is the source. In a pipeline, the source is the data going into your transformation and the target is what comes out, and each can have its own **contract**. A contract's `direction=` (`"source"` or `"target"`) records which side it's meant for. [Learn more](../contracts-and-pipelines/pipelines.md).

Specification check  
A check that text values follow a well-known format, such as email addresses, URLs, postal codes, credit card numbers, ISBNs, VINs, or IBANs. Added with [col_vals_within_spec()](../../reference/Validate.col_vals_within_spec.md#pointblank.Validate.col_vals_within_spec), for example with `spec="email"` or `spec="postal_code[USA]"`. [Learn more](../validation-plan/validation-methods.md).

Status check  
A method that tells you how a validation went, as `True` or `False`, without stopping your code. [`all_passed()`](../../reference/Validate.all_passed.md#pointblank.Validate.all_passed) tells you whether every **test unit** passed, and [above_threshold()](../../reference/Validate.above_threshold.md#pointblank.Validate.above_threshold), [warning()](../../reference/Validate.warning.md#pointblank.Validate.warning), [error()](../../reference/Validate.error.md#pointblank.Validate.error), and [critical()](../../reference/Validate.critical.md#pointblank.Validate.critical) tell you whether **threshold** levels were reached. See also **assertion**. [Learn more](../advanced-validation/assertions.md).

Step  
Usually short for **validation step**. In a **contract**, the [Step](../../reference/Step.md#pointblank.Step) class describes one validation method call by its name and arguments (for example, `Step("col_vals_not_null", columns="id")`), so that it can be saved as YAML or JSON. [Learn more](../contracts-and-pipelines/contracts.md).

Step report  
A detailed view of a single **validation step**, showing what it checked and, where it makes sense, the rows that failed. You get one with [Validate.get_step_report()](../../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report). See also **validation report**. [Learn more](../post-interrogation/step-reports.md).

Structured missingness  
See **coded missing value**.

Sundering  
Splitting a table in two after validation: the rows that passed every row-by-row check, and the rows that failed at least one. You get either part with [Validate.get_sundered_data()](../../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data), using `type="pass"` or `type="fail"`. See also **data extract**. [Learn more](../post-interrogation/sundering.md).

Table comparison  
Finding the differences between two tables. [`compare()`](../../reference/compare.md#pointblank.compare) matches up the rows of a source table and a target table, gives each row a **row status** (such as `changed` or `missing`), and reports any differences in their columns. It returns a [Comparison](../../reference/Comparison.md#pointblank.Comparison) object, which counts as `True` only if the tables match. Rows can be matched by key columns (`keys="id"`), by position (the default), or regardless of order (`keys="*"`), and small differences can be allowed with `tolerance=` or ignored with `normalize=` (for example, letter case). As a **validation step**, [tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match) runs the same comparison with each row as a **test unit**, and [tbl_reconcile()](../../reference/Validate.tbl_reconcile.md#pointblank.Validate.tbl_reconcile) compares totals instead (see **reconciliation**). From the **CLI**, use `pb compare`. To check only the number of rows or columns, use [row_count_match()](../../reference/Validate.row_count_match.md#pointblank.Validate.row_count_match) or [col_count_match()](../../reference/Validate.col_count_match.md#pointblank.Validate.col_count_match). See also **source and target** and **partitions**. [Learn more](../table-comparison/comparing-tables.md).

Test data generation  
Making realistic fake tables for testing and demos. You describe each column in a [Schema](../../reference/Schema.md#pointblank.Schema) with **fields**, such as `int_field(min_val=1)` or `string_field(preset="email")`, and then create rows with [generate_dataset()](../../reference/generate_dataset.md#pointblank.generate_dataset). Use `country=` for data that fits one or more countries, and `seed=` to get the same rows every time. When Pointblank is installed, a [generate_dataset](../../reference/generate_dataset.md#pointblank.generate_dataset) fixture is also available in pytest. Not to be confused with **test datasets**. See also **preset**, **data coherence**, and **seed**. [Learn more](../test-data-generation/test-data-generation.md).

Test dataset  
One of the sample tables that comes with Pointblank, loaded with [load_dataset()](../../reference/load_dataset.md#pointblank.load_dataset). The choices are `small_table`, `game_revenue`, `nycflights`, and `global_sales`, and `tbl_type=` sets which **backend** you get (for example, `"polars"`, `"pandas"`, or `"duckdb"`). They're used in examples throughout the documentation. [Learn more](../getting-started/quickstart.md).

Test unit  
A single thing that passes or fails in a **validation step**. When a step checks the values in a column, each value is one test unit, so checking a column in a 100-row table gives 100 test units. When a step checks the table or column as a whole (like [col_exists()](../../reference/Validate.col_exists.md#pointblank.Validate.col_exists) or [col_sum_gt()](../../reference/Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt)), there is just one test unit. The pass and fail counts in the **validation report** are counts of test units. [Learn more](../getting-started/quickstart.md).

Threshold  
A limit on how many failing **test units** a step can have before it's flagged. A threshold can be a fraction (`0.1` means 10% of test units) or a count (`5` means five test units); any value of 1 or more is treated as a count. There are three levels: **warning, error, and critical**. A level is reached when the failures meet or go past its threshold. You set thresholds for the whole validation (`Validate(thresholds=...)`) or for a single step, either with the [Thresholds](../../reference/Thresholds.md#pointblank.Thresholds) class or with a shortcut like `(0.1, 0.2, 0.3)`. [Learn more](../validation-plan/thresholds.md).

Tolerance  
How far a result can be from the expected one and still pass. **Aggregate checks** and [col_pct_null()](../../reference/Validate.col_pct_null.md#pointblank.Validate.col_pct_null) take `tol=`: for example, `col_avg_eq(columns="x", value=10, tol=0.5)` passes for any average from 9.5 to 10.5. In a **table comparison**, `tolerance=` sets how much two values may differ, as an absolute amount (`atol`), as a share of the source value (`rtol`), or, for dates and times, as a length of time like `"1s"`. [Learn more](../validation-plan/validation-methods.md).

Trend check  
A check that a column's values go up ([col_vals_increasing()](../../reference/Validate.col_vals_increasing.md#pointblank.Validate.col_vals_increasing)) or down ([col_vals_decreasing()](../../reference/Validate.col_vals_decreasing.md#pointblank.Validate.col_vals_decreasing)) from one row to the next, as in a time series. Repeated values fail unless `allow_stationary=True`. Rows are compared in the table's own order, or in the order given by `order_by=`, which database tables need since their rows have no fixed order. [Learn more](../validation-plan/validation-methods.md).

Validate  
The main class in Pointblank. A [Validate](../../reference/Validate.md#pointblank.Validate) object holds your table, settings like **thresholds** and **actions**, and the **validation plan**. You add checks by calling validation methods on it, one after another, and run them with `.interrogate()`. Afterwards, the same object gives you reports, **data extracts**, and results. [Learn more](../validation-plan/validation-overview.md).

Validation method  
A method of [Validate](../../reference/Validate.md#pointblank.Validate) that adds a check to the **validation plan**, such as [col_vals_gt()](../../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) (values are greater than some number) or [rows_distinct()](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) (no duplicate rows). Each call adds one or more **validation steps**. [Learn more](../validation-plan/validation-methods.md).

Validation plan  
The list of checks you've added to a [Validate](../../reference/Validate.md#pointblank.Validate) object. Building the plan doesn't check anything yet: it only records what should be checked. The checks happen during **interrogation**. Plans can also be written in YAML (see **YAML workflow**). [Learn more](../validation-plan/validation-overview.md).

Validation report  
The table that summarizes the results of every **validation step** after **interrogation**. It shows each step's description, which columns it checked, how many **test units** passed and failed, and which **threshold** levels were reached. You get it with [Validate.get_tabular_report()](../../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report). See also **step report**, **JSON report**, and **report language**. [Learn more](../post-interrogation/validation-reports.md).

Validation step  
One check in a **validation plan**. Each call to a **validation method** creates at least one step: one for each column it checks and one for each **segment**. Steps are numbered from 1. After **interrogation**, each step records how many **test units** passed and failed and which **threshold** levels were reached. [Learn more](../validation-plan/validation-methods.md).

Validation template  
A **YAML workflow** written to be used with many tables. Leave the data out with `tbl: null`, then give a table each time it runs with `yaml_interrogate(..., set_tbl=...)`. This way, a set of checks written once can be run on, say, each quarter's data. [Learn more](../yaml/yaml-validation-workflows.md).

Violation handling  
What happens when data fails a **contract** inside a **pipeline**, set with the contract's `on_violation=`. With `"warn"` (the default), Python shows a warning; with `"raise"`, an error stops the run; and with `"log"`, a message is written to the `pointblank.contract` logger. [Learn more](../contracts-and-pipelines/contracts.md).

Warning, error, and critical  
The three **threshold** levels, from least to most serious. The names are only labels, so what each one means is up to you. For example, you might send a note to your team at the warning level and stop a data pipeline at the critical level. The **validation report** shows which levels each step reached in its W, E, and C columns. [Learn more](../validation-plan/thresholds.md).

YAML workflow  
A validation written in a YAML file instead of Python code. The file lists the data to use, the **validation steps**, the **thresholds**, and other settings. This makes validations easy to store in version control and share with people who don't write Python. You run one with [yaml_interrogate()](../../reference/yaml_interrogate.md#pointblank.yaml_interrogate) or with `pb run` on the command line, and you can convert one to Python with [yaml_to_python()](../../reference/yaml_to_python.md#pointblank.yaml_to_python). See also **validation template** and **`python:` block**. [Learn more](../yaml/yaml-validation-workflows.md).
