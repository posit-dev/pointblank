# Glossary

This page explains the terms used throughout the Pointblank documentation, in alphabetical order. Where a term matches something in Pointblank's API, the name of the class, function, or method is shown in `code`. Terms in **bold** have their own entry on this page.

Action  
Something Pointblank does when a step's failures reach a **threshold** level, such as printing a message or running a function you supply. You set actions with the [Actions](../../reference/Actions.md#pointblank.Actions) class, either for the whole validation (`Validate(actions=...)`) or for a single step. Messages can include placeholders like `{step}`, `{col}`, and `{level}`, which are filled in when the action runs. By default, only the action for the most serious level reached is run. See also **final action**. [Learn more](../../user-guide/validation-plan/actions.md).

Active / inactive step  
Whether a **validation step** runs. A step with `active=False` is skipped, but it still appears in the **validation report** so that step numbers don't shift. You can also give `active=` a function that decides at run time. The helpers [has_columns()](../../reference/has_columns.md#pointblank.has_columns) and [has_rows()](../../reference/has_rows.md#pointblank.has_rows) make this easy: for example, `active=pb.has_columns("email")` runs the step only if the table has an `email` column.

ADaM (Analysis Data Model)  
A **CDISC** standard for the analysis-ready datasets used in clinical trials, which are built from **SDTM** data. Common examples are ADSL (one row per trial subject) and ADAE (adverse events). Pointblank can check that an ADaM dataset has the columns the standard requires, using [validate_adam_structure()](../../reference/validate_adam_structure.md#pointblank.validate_adam_structure) or [validate_adam()](../../reference/validate_adam.md#pointblank.validate_adam). [Learn more](../../user-guide/metadata-import/cdisc-validation.md).

Aggregate check  
A check on a summary of a whole column, such as its sum, average, or standard deviation, instead of on each value. Examples include [col_sum_gt()](../../reference/Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt), [col_avg_eq()](../../reference/Validate.col_avg_eq.md#pointblank.Validate.col_avg_eq), and [col_sd_lt()](../../reference/Validate.col_sd_lt.md#pointblank.Validate.col_sd_lt). Because the whole column is summarized into one number, an aggregate check has a single **test unit**. [Learn more](../../user-guide/validation-plan/validation-methods.md).

AI check  
A **validation step** where a large language model (LLM) reads each row and judges whether it follows a rule you describe in plain words. This helps with rules that are hard to write as code, like "the description should mention the product's color". Added with [prompt()](../../reference/Validate.prompt.md#pointblank.Validate.prompt). [Learn more](../../user-guide/validation-plan/validation-methods.md).

AI Validation Editor  
A tool that changes an existing **validation plan** based on a plain-English instruction, such as "also check that the email column has no missing values". It uses an LLM to write the revised plan. With the [EditValidation](../../reference/EditValidation.md#pointblank.EditValidation) class you can see what changed ([diff()](../../reference/EditValidation.md#pointblank.EditValidation.diff) or [review()](../../reference/EditValidation.md#pointblank.EditValidation.review)) and then get the new plan with [accept()](../../reference/EditValidation.md#pointblank.EditValidation.accept). It's also available from the **CLI** as `pb edit`. See also **DraftValidation**. [Learn more](../../user-guide/advanced-validation/ai-validation-editor.md).

Assertion  
A way to stop your code when data doesn't pass validation. After building a plan, call [assert_passing()](../../reference/Validate.assert_passing.md#pointblank.Validate.assert_passing) to raise an error if any **test unit** failed, or [assert_below_threshold()](../../reference/Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold) to raise an error if a **threshold** level was reached. Both run **interrogation** first if it hasn't happened yet. This is handy in data pipelines and tests. If you'd rather get `True` or `False` than an error, use a **status check**. [Learn more](../../user-guide/advanced-validation/assertions.md).

Assistant  
A chat window where you can ask an LLM questions about your data and about Pointblank. You start it with `assistant()`, and it opens in your web browser or in the terminal. If you give it a table, a summary of that table is shared with the model so its answers fit your data.

Backend  
The library or database that holds your table. Pointblank works with Polars, Pandas, and PySpark tables, and (through the Ibis library) with databases such as DuckDB, PostgreSQL, MySQL, SQLite, SQL Server, Snowflake, Databricks, and BigQuery. It can also read CSV and Parquet files directly. You don't need to tell Pointblank which backend you're using, since it works this out from the data you pass in. [Learn more](../../user-guide/getting-started/installation.md).

Brief  
A short description of what a **validation step** checks, shown in the STEP column of the **validation report**. You can write your own (`brief="..."`) or let Pointblank write one for you (`brief=True`). Briefs can include placeholders such as `{col}` (the column name), `{step}` (the step number), and `{auto}` (Pointblank's generated description). Generated briefs use the **report language**. [Learn more](../../user-guide/validation-plan/briefs.md).

CDISC (Clinical Data Interchange Standards Consortium)  
The organization that sets data standards for clinical trials, including **SDTM**, **ADaM**, and **Define-XML**. Pointblank can read CDISC metadata, check that datasets have the expected structure, and check a set of trial datasets against CDISC's rules. It can do that last part with its own built-in rules ([validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig)) or by running the **CORE** engine. [Learn more](../../user-guide/metadata-import/cdisc-validation.md).

CLI (command-line interface)  
The `pb` command, which you run in a terminal to use Pointblank without writing Python. `pb preview`, `pb scan`, `pb missing`, and `pb info` let you look at a table. `pb validate` runs quick one-off checks, and `pb run` runs a full validation written in Python or YAML. `pb make-template` creates a starter file, and `pb edit` opens the **AI Validation Editor**. [Learn more](../../user-guide/the-pointblank-cli/cli-reference.md).

Coded missing value  
A stand-in value, such as `-99` or `"UNK"`, that means a value is missing for a known reason (for example, the person refused to answer or wasn't asked). The [MissingSpec](../../reference/MissingSpec.md#pointblank.MissingSpec) class lists which codes a column uses and what each one means. Validation methods like [col_missing_coded()](../../reference/Validate.col_missing_coded.md#pointblank.Validate.col_missing_coded) and the **missing values table** can then treat these codes as missing data. See also **null flavor**. [Learn more](../../user-guide/data-inspection/missing-vals-tbl.md).

Codelist  
A named list of allowed values, often with a label for each (for example, `"M"` means Male and `"F"` means Female). Codelists usually come from a **metadata import**, such as CDISC controlled terminology or SPSS value labels. Pointblank represents them with the [Codelist](../../reference/Codelist.md#pointblank.Codelist) class. [Learn more](../../user-guide/metadata-import/metadata-import.md).

Column selection  
How you tell a **validation step** which columns to check. You can give a single column name, a list of names, or a helper that picks columns for you: [starts_with()](../../reference/starts_with.md#pointblank.starts_with), [ends_with()](../../reference/ends_with.md#pointblank.ends_with), [contains()](../../reference/contains.md#pointblank.contains), [matches()](../../reference/matches.md#pointblank.matches), [everything()](../../reference/everything.md#pointblank.everything), [first_n()](../../reference/first_n.md#pointblank.first_n), or [last_n()](../../reference/last_n.md#pointblank.last_n). Helpers can be combined inside [col()](../../reference/col.md#pointblank.col) using `&` (and), `|` (or), and `-` (but not). Each column that's picked gets its own step. [Learn more](../../user-guide/validation-plan/column-selection-patterns.md).

Column summary table  
A table that describes each column of your data: its type, how many values are missing, how many are distinct, and statistics like the mean and quartiles. You make one with [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl). See also **DataScan**, **preview**, and **missing values table**. [Learn more](../../user-guide/data-inspection/col-summary-tbl.md).

Composable steps  
A reusable set of **validation steps** that isn't tied to any particular table. You build the set with the [Steps](../../reference/Steps.md#pointblank.Steps) class, using the same method names as [Validate](../../reference/Validate.md#pointblank.Validate), and add it to any validation with `Validate.add_steps()`. This lets you write common checks once and reuse them for many tables. [Learn more](../../user-guide/advanced-validation/composable-steps.md).

Conjointly  
A single **validation step** that checks several conditions on each row at once. A row passes only if every condition is true for that row. This is different from adding one step per condition, where each condition is judged on its own. Added with [conjointly()](../../reference/Validate.conjointly.md#pointblank.Validate.conjointly). [Learn more](../../user-guide/validation-plan/validation-methods.md).

Contract  
A written-down agreement about what a table should look like. It combines a **schema** (the expected columns and their types), a list of rules, and details like the owner and version. You create one with the [Contract](../../reference/Contract.md#pointblank.Contract) class and check a table against it with `.validate(data)`. Contracts can be saved as YAML, imported from other formats (such as JSON Schema, Frictionless, dbt, and ODCS) with `import_contract()`, and exported with `export_contract()`. See also **pipeline**. [Learn more](../../user-guide/contracts-and-pipelines/contracts.md).

CORE (CDISC Conformance Rules Engine)  
CDISC's official, open-source program for checking clinical trial data against CDISC's rules. You install it separately, and [validate_cdisc_submission()](../../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission) runs it for you and turns its output into a [ConformanceReport](../../reference/ConformanceReport.md#pointblank.ConformanceReport). Pointblank also has built-in rules that don't need CORE (see **CDISC**). [Learn more](../../user-guide/metadata-import/cdisc-submission-conformance.md).

Critical  
See **warning, error, and critical**.

Custom check  
A **validation step** that uses a function you write, for rules that none of the built-in validation methods cover. Your function receives the table and returns `True` or `False` values showing what passed. Added with [specially()](../../reference/Validate.specially.md#pointblank.Validate.specially). [Learn more](../../user-guide/validation-plan/validation-methods.md).

Data extract  
The rows that failed a **validation step**, saved during **interrogation** so you can look at them afterwards. You get them with [Validate.get_data_extracts()](../../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts). Only steps that check values row by row produce extracts. Options on [interrogate()](../../reference/Validate.interrogate.md#pointblank.Validate.interrogate), such as `extract_limit=`, control how many rows are kept. See also **sundering**. [Learn more](../../user-guide/post-interrogation/extracts.md).

DataScan  
An object that holds a column-by-column profile of a table, with statistics like counts, missing values, and averages. It's what the **column summary table** is built from. You can save a [DataScan](../../reference/DataScan.md#pointblank.DataScan) to a JSON file, load it again later, and compare two scans to spot **drift**. [Learn more](../../user-guide/data-inspection/col-summary-tbl.md).

Define-XML  
A **CDISC** file format (written in XML) that describes a clinical trial dataset: its column names, labels, types, and allowed values. Pointblank can read Define-XML files with [import_metadata()](../../reference/import_metadata.md#pointblank.import_metadata) and can pass them to **CORE**. [Learn more](../../user-guide/metadata-import/cdisc-validation.md).

DraftValidation  
A tool that uses an LLM to write a first draft of a **validation plan** for your table. It sends the model a summary of your table (column types, counts, and basic statistics) rather than the full data, and gives back Python code for you to review and adjust. Available through the [DraftValidation](../../reference/DraftValidation.md#pointblank.DraftValidation) class. See also **AI Validation Editor**. [Learn more](../../user-guide/advanced-validation/draft-validation.md).

Drift  
A change in a table between two points in time, like a column being added or removed, or the typical values in a column shifting. You can look for drift by comparing two **DataScan** objects with [DataScan.compare()](../../reference/DataScan.compare.md#pointblank.DataScan.compare). [Learn more](../../user-guide/data-inspection/col-summary-tbl.md).

Error  
See **warning, error, and critical**.

Expression check  
A **validation step** that checks each row against a condition written in your table library's own syntax. For example, with Polars you might check `pl.col("end") > pl.col("start")`. Added with [col_vals_expr()](../../reference/Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr). [Learn more](../../user-guide/advanced-validation/expressions.md).

Final action  
An **action** that runs once, after all the **validation steps** are finished, whatever the results were. Final actions are useful for sending a summary, such as a message to a chat channel. You set them with the [FinalActions](../../reference/FinalActions.md#pointblank.FinalActions) class. Inside a final action, [get_validation_summary()](../../reference/get_validation_summary.md#pointblank.get_validation_summary) gives you the results. [Learn more](../../user-guide/validation-plan/actions.md).

Global settings  
Default settings that apply to every validation in your Python session, changed with `pb.config()`. They control what the **validation report** includes (for example, the header, footer, and timings) and how the **health score** is calculated. Each call to `pb.config()` sets every option, so any option you leave out goes back to its default.

Health score  
A single number from 0 to 100 that sums up data quality: the share of all **test units** that passed. Pointblank also works out a score for each **quality dimension**. Get the scores with [get_health_score()](../../reference/Validate.get_health_score.md#pointblank.Validate.get_health_score) and [get_dimension_scores()](../../reference/Validate.get_dimension_scores.md#pointblank.Validate.get_dimension_scores), or show them with [get_scorecard()](../../reference/Validate.get_scorecard.md#pointblank.Validate.get_scorecard). They appear in the **validation report** when you use `incl_dimensions=True`. [Learn more](../../user-guide/post-interrogation/quality-dimensions-and-scoring.md).

Interrogation  
Running the checks in a **validation plan**, which happens when you call `.interrogate()`. Nothing is checked until then. Interrogation runs every step, counts what passed and failed, compares the counts to the **thresholds**, runs any **actions**, and saves **data extracts**. [Learn more](../../user-guide/validation-plan/validation-overview.md).

JSON report  
The validation results as JSON text, which other programs can read. You get it with [Validate.get_json_report()](../../reference/Validate.get_json_report.md#pointblank.Validate.get_json_report), and you can choose which details to include with `use_fields=` or leave some out with `exclude_fields=`. See also **validation report**.

MCP server  
A way for AI assistants (such as the AI chat in VS Code or Positron) to use Pointblank for you. MCP (Model Context Protocol) is a standard way for AI tools to connect to other programs. Through it, an assistant can load data, build and run validations, and show previews and summaries. [Learn more](../../user-guide/mcp-server/mcp-quick-start.md).

Metadata import  
Reading a description of a dataset from a file made by another tool, and turning it into Pointblank checks. [import_metadata()](../../reference/import_metadata.md#pointblank.import_metadata) reads files from SPSS, SAS, and Stata, Frictionless Data Packages, CSV on the Web (CSVW), and **CDISC** standards like **Define-XML**. You can turn the result into a validation plan with `to_validate(data)` or into a **schema** with [to_schema()](../../reference/MetadataImport.md#pointblank.MetadataImport.to_schema). [Learn more](../../user-guide/metadata-import/metadata-import.md).

Missing values table  
A table showing where the missing values are in your data, made with [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl). By default it splits the rows into chunks and shades each chunk by how much of each column is missing. If you describe your **coded missing values** with `missing=`, it can also break them down by reason. [Learn more](../../user-guide/data-inspection/missing-vals-tbl.md).

`na_pass`  
An option on many validation methods that decides whether missing values (nulls) count as passing. The default, `na_pass=False`, counts them as failures. Set `na_pass=True` when missing values are acceptable. [Learn more](../../user-guide/validation-plan/validation-methods.md).

Null flavor  
A standard code from the HL7 and **CDISC** standards that says why a value is missing. For example, `"NI"` means no information, `"UNK"` means unknown, and `"ASKU"` means asked but unknown. [MissingSpec.from_cdisc_null_flavors()](../../reference/MissingSpec.md#pointblank.MissingSpec.from_cdisc_null_flavors) sets up all 14 codes for you. See also **coded missing value**. [Learn more](../../user-guide/metadata-import/cdisc-validation.md).

OpenTelemetry integration  
OpenTelemetry is a widely used standard for sending measurements from programs to monitoring tools such as Grafana or Datadog. Adding [emit_otel()](../../reference/emit_otel.md#pointblank.emit_otel) as a **final action** sends numbers like pass rates and **threshold** breaches after each validation. This lets you track data quality on the same dashboards as the rest of your systems. [Learn more](../../user-guide/integrations/otel-integration.md).

Pipeline  
A data-processing step with **contracts** on both sides. A [Pipeline](../../reference/Pipeline.md#pointblank.Pipeline) checks the incoming data (the source), runs your transformation function, then checks the result (the target), all in one call to `run(data, transform)`. By default, if the incoming data fails, the transformation is skipped. [Learn more](../../user-guide/contracts-and-pipelines/pipelines.md).

Plan serialization  
Turning a **validation plan** into text so it can be saved, shared, reviewed, or rebuilt later. [Validate.to_code()](../../reference/Validate.to_code.md#pointblank.Validate.to_code) gives Python code, [Validate.to_yaml()](../../reference/Validate.to_yaml.md#pointblank.Validate.to_yaml) gives a **YAML workflow**, and [Validate.to_json_schema()](../../reference/Validate.to_json_schema.md#pointblank.Validate.to_json_schema) gives a JSON Schema document. [Learn more](../../user-guide/advanced-validation/plan-serialization.md).

Preprocessing  
Changing the table for just one step before it's checked, using the `pre=` option. You give `pre=` a function that takes the table and returns a changed version, for example with some rows filtered out or a new column added. The original data and other steps are not affected. The number of **test units** is counted after the change. [Learn more](../../user-guide/validation-plan/preprocessing.md).

Preview  
A quick look at a table showing the first and last few rows along with each column's type, made with [preview()](../../reference/preview.md#pointblank.preview). See also **column summary table**. [Learn more](../../user-guide/data-inspection/preview.md).

Quality dimension  
A category describing what kind of data quality a step measures. The categories are *completeness*, *validity*, *uniqueness*, *consistency*, *timeliness*, and *volume*. Each validation method has a default category (for example, [col_vals_not_null()](../../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) measures completeness), and you can change it with `dimension=`. Each dimension gets its own score, and together they make up the **health score**. [Learn more](../../user-guide/post-interrogation/quality-dimensions-and-scoring.md).

Referential check  
A check that every value in a column also appears in a column of another table, for example that every `customer_id` in an orders table exists in the customers table. Added with [col_vals_in_table()](../../reference/Validate.col_vals_in_table.md#pointblank.Validate.col_vals_in_table). [Learn more](../../user-guide/validation-plan/validation-methods.md).

Report language  
The language used for the text in reports and in generated **briefs**, set with `Validate(lang=...)`. Forty languages are available. A separate option, `locale=`, controls how numbers are formatted (for example, whether a comma or a period is used as the decimal mark). [Learn more](../../user-guide/validation-plan/briefs.md).

Schema  
A description of the columns a table should have: their names and, optionally, their types. You write one with the [Schema](../../reference/Schema.md#pointblank.Schema) class and check a table against it with [col_schema_match()](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match), with options for how strict to be about column order, capital letters, and types. [schema_from_tbl()](../../reference/schema_from_tbl.md#pointblank.schema_from_tbl) builds a schema from an existing table. [Learn more](../../user-guide/advanced-validation/schema-validation.md).

Scorecard  
A compact table showing the **health score** and the score for each **quality dimension**, made with [Validate.get_scorecard()](../../reference/Validate.get_scorecard.md#pointblank.Validate.get_scorecard). [Learn more](../../user-guide/post-interrogation/quality-dimensions-and-scoring.md).

SDTM (Study Data Tabulation Model)  
A **CDISC** standard for organizing the data collected in a clinical trial into tables called *domains*, such as DM (demographics), AE (adverse events), and LB (lab results). Pointblank can check that a domain has the columns the standard requires ([validate_sdtm_structure()](../../reference/validate_sdtm_structure.md#pointblank.validate_sdtm_structure)), and it can check datasets against the SDTM rules ([validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig)). [Learn more](../../user-guide/metadata-import/cdisc-validation.md).

Segment  
A group of rows that gets checked and reported on its own. Using `segments=` splits one step into several, one per group. For example, `segments="region"` makes one step for each region in the table, and `segments=("region", ["East", "West"])` makes steps for just those two regions. To treat several values as one group, use [seg_group()](../../reference/seg_group.md#pointblank.seg_group). [Learn more](../../user-guide/validation-plan/segmentation.md).

Status check  
A method that tells you how a validation went, as `True` or `False`, without stopping your code. `all_passed()` tells you whether every **test unit** passed, and [above_threshold()](../../reference/Validate.above_threshold.md#pointblank.Validate.above_threshold), [warning()](../../reference/Validate.warning.md#pointblank.Validate.warning), [error()](../../reference/Validate.error.md#pointblank.Validate.error), and [critical()](../../reference/Validate.critical.md#pointblank.Validate.critical) tell you whether **threshold** levels were reached. See also **assertion**. [Learn more](../../user-guide/advanced-validation/assertions.md).

Step  
See **validation step**.

Step report  
A detailed view of a single **validation step**, showing what it checked and, where it makes sense, the rows that failed. You get one with [Validate.get_step_report()](../../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report). See also **validation report**. [Learn more](../../user-guide/post-interrogation/step-reports.md).

Sundering  
Splitting a table in two after validation: the rows that passed every row-by-row check, and the rows that failed at least one. You get either part with [Validate.get_sundered_data()](../../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data), using `type="pass"` or `type="fail"`. See also **data extract**. [Learn more](../../user-guide/post-interrogation/sundering.md).

Table comparison  
Checks on the table as a whole: whether it matches another table exactly ([tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match)) or has the expected number of rows or columns ([row_count_match()](../../reference/Validate.row_count_match.md#pointblank.Validate.row_count_match) and [col_count_match()](../../reference/Validate.col_count_match.md#pointblank.Validate.col_count_match)). [Learn more](../../user-guide/validation-plan/validation-methods.md).

Test data generation  
Making realistic fake tables for testing and demos. You describe each column in a [Schema](../../reference/Schema.md#pointblank.Schema) with field helpers, such as `int_field(min_val=1)` or `string_field(preset="email")`, and then create rows with [generate_dataset()](../../reference/generate_dataset.md#pointblank.generate_dataset). When Pointblank is installed, a [generate_dataset](../../reference/generate_dataset.md#pointblank.generate_dataset) fixture is also available in pytest. Not to be confused with **test datasets**. [Learn more](../../user-guide/test-data-generation/test-data-generation.md).

Test dataset  
One of the sample tables that comes with Pointblank, loaded with [load_dataset()](../../reference/load_dataset.md#pointblank.load_dataset). The choices are `small_table`, `game_revenue`, `nycflights`, and `global_sales`, and `tbl_type=` sets which **backend** you get (for example, `"polars"`, `"pandas"`, or `"duckdb"`). They're used in examples throughout the documentation. [Learn more](../../user-guide/getting-started/quickstart.md).

Test unit  
A single thing that passes or fails in a **validation step**. When a step checks the values in a column, each value is one test unit, so checking a column in a 100-row table gives 100 test units. When a step checks the table or column as a whole (like [col_exists()](../../reference/Validate.col_exists.md#pointblank.Validate.col_exists) or [col_sum_gt()](../../reference/Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt)), there is just one test unit. The pass and fail counts in the **validation report** are counts of test units. [Learn more](../../user-guide/getting-started/quickstart.md).

Threshold  
A limit on how many failing **test units** a step can have before it's flagged. A threshold can be a fraction (`0.1` means 10% of test units) or a count (`5` means five test units); any value of 1 or more is treated as a count. There are three levels: **warning, error, and critical**. A level is reached when the failures meet or go past its threshold. You set thresholds for the whole validation (`Validate(thresholds=...)`) or for a single step, either with the [Thresholds](../../reference/Thresholds.md#pointblank.Thresholds) class or with a shortcut like `(0.1, 0.2, 0.3)`. [Learn more](../../user-guide/validation-plan/thresholds.md).

Validate  
The main class in Pointblank. A [Validate](../../reference/Validate.md#pointblank.Validate) object holds your table, settings like **thresholds** and **actions**, and the **validation plan**. You add checks by calling validation methods on it, one after another, and run them with `.interrogate()`. Afterwards, the same object gives you reports, **data extracts**, and results. [Learn more](../../user-guide/validation-plan/validation-overview.md).

Validation method  
A method of [Validate](../../reference/Validate.md#pointblank.Validate) that adds a check to the **validation plan**, such as [col_vals_gt()](../../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) (values are greater than some number) or [rows_distinct()](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) (no duplicate rows). Each call adds one or more **validation steps**. [Learn more](../../user-guide/validation-plan/validation-methods.md).

Validation plan  
The list of checks you've added to a [Validate](../../reference/Validate.md#pointblank.Validate) object. Building the plan doesn't check anything yet: it only records what should be checked. The checks happen during **interrogation**. Plans can also be written in YAML (see **YAML workflow**). [Learn more](../../user-guide/validation-plan/validation-overview.md).

Validation report  
The table that summarizes the results of every **validation step** after **interrogation**. It shows each step's description, which columns it checked, how many **test units** passed and failed, and which **threshold** levels were reached. You get it with [Validate.get_tabular_report()](../../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report). See also **step report**, **JSON report**, and **report language**. [Learn more](../../user-guide/post-interrogation/validation-reports.md).

Validation step  
One check in a **validation plan**. Each call to a **validation method** creates at least one step: one for each column it checks and one for each **segment**. Steps are numbered from 1. After **interrogation**, each step records how many **test units** passed and failed and which **threshold** levels were reached. [Learn more](../../user-guide/validation-plan/validation-methods.md).

Warning, error, and critical  
The three **threshold** levels, from least to most serious. The names are only labels, so what each one means is up to you. For example, you might send a note to your team at the warning level and stop a data pipeline at the critical level. The **validation report** shows which levels each step reached in its W, E, and C columns. [Learn more](../../user-guide/validation-plan/thresholds.md).

YAML workflow  
A validation written in a YAML file instead of Python code. The file lists the data to use, the **validation steps**, the **thresholds**, and other settings. This makes validations easy to store in version control and share with people who don't write Python. You run one with [yaml_interrogate()](../../reference/yaml_interrogate.md#pointblank.yaml_interrogate) or with `pb run` on the command line, and you can convert one to Python with [yaml_to_python()](../../reference/yaml_to_python.md#pointblank.yaml_to_python). [Learn more](../../user-guide/yaml/yaml-validation-workflows.md).
