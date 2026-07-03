# API Reference


## Validate


When performing data validation, use the `Validate` class to get the process started. It takes the target table and options for metadata and failure thresholds (using the `Thresholds` class or shorthands). The `Validate` class has numerous methods for defining validation steps and for obtaining post-interrogation metrics and data.


[Validate](Validate.md#pointblank.Validate)  
Workflow for defining a set of validations on a table and interrogating for results.

[Thresholds](Thresholds.md#pointblank.Thresholds)  
Definition of threshold values.

[Actions](Actions.md#pointblank.Actions)  
Definition of action values.

[FinalActions](FinalActions.md#pointblank.FinalActions)  
Define actions to be taken after validation is complete.

[Schema](Schema.md#pointblank.Schema)  
Definition of a schema object.

[DraftValidation](DraftValidation.md#pointblank.DraftValidation)  
Draft a validation plan for a given table using an LLM.

[EditValidation](EditValidation.md#pointblank.EditValidation)  
Edit an existing validation plan with a plain-English instruction using an LLM.

[MissingSpec](MissingSpec.md#pointblank.MissingSpec)  
Specification for structured missing values in a column.


## Contracts and Pipelines


Use `Contract` and `Step` to define declarative data quality contracts that specify what valid data looks like. Use `Pipeline` to enforce contracts at both boundaries of a data transformation (source and target), producing a `PipelineResult` with full introspection into what passed and what failed.


[Contract](Contract.md#pointblank.Contract)  
A declarative boundary contract for pipeline data.

[Step](Step.md#pointblank.Step)  
A single validation step in a Contract, defined declaratively.

[Pipeline](Pipeline.md#pointblank.Pipeline)  
Binds source and target contracts into a pipeline boundary enforcement unit.

[PipelineResult](PipelineResult.md#pointblank.PipelineResult)  
Result of a pipeline boundary validation run.


## Contract Import/Export


Import external schema definitions (JSON Schema, Frictionless Table Schema, and more) into Pointblank validation workflows, or export Pointblank contracts to those formats. Use `import_contract()` as the entry point, `export_contract()` for the reverse, and `register_adapter()` to add support for custom formats.


[import_contract()](import_contract.md#pointblank.import_contract)  
Import a contract/schema from an external format.

[export_contract()](export_contract.md#pointblank.export_contract)  
Export a Pointblank validation or contract to an external format.

[list_adapters()](list_adapters.md#pointblank.list_adapters)  
List all registered adapters with their capabilities.

[register_adapter()](register_adapter.md#pointblank.register_adapter)  
Register a contract adapter class.

[ContractImport](ContractImport.md#pointblank.ContractImport)  
Result of importing an external contract/schema.

[ContractAdapter](ContractAdapter.md#pointblank.ContractAdapter)  
Base class for contract import/export adapters.


## Validation Steps


Validation steps are sequential validations on the target data. Call `Validate`'s validation methods to build up a validation plan: a collection of steps that provides good validation coverage.


[Validate.col_vals_gt()](Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)  
Are column data greater than a fixed value or data in another column?

[Validate.col_vals_lt()](Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt)  
Are column data less than a fixed value or data in another column?

[Validate.col_vals_ge()](Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge)  
Are column data greater than or equal to a fixed value or data in another column?

[Validate.col_vals_le()](Validate.col_vals_le.md#pointblank.Validate.col_vals_le)  
Are column data less than or equal to a fixed value or data in another column?

[Validate.col_vals_eq()](Validate.col_vals_eq.md#pointblank.Validate.col_vals_eq)  
Are column data equal to a fixed value or data in another column?

[Validate.col_vals_ne()](Validate.col_vals_ne.md#pointblank.Validate.col_vals_ne)  
Are column data not equal to a fixed value or data in another column?

[Validate.col_vals_between()](Validate.col_vals_between.md#pointblank.Validate.col_vals_between)  
Do column data lie between two specified values or data in other columns?

[Validate.col_vals_outside()](Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside)  
Do column data lie outside of two specified values or data in other columns?

[Validate.col_vals_in_set()](Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set)  
Validate whether column values are in a set of values.

[Validate.col_vals_not_in_set()](Validate.col_vals_not_in_set.md#pointblank.Validate.col_vals_not_in_set)  
Validate whether column values are not in a set of values.

[Validate.col_vals_increasing()](Validate.col_vals_increasing.md#pointblank.Validate.col_vals_increasing)  
Are column data increasing by row?

[Validate.col_vals_decreasing()](Validate.col_vals_decreasing.md#pointblank.Validate.col_vals_decreasing)  
Are column data decreasing by row?

[Validate.col_vals_null()](Validate.col_vals_null.md#pointblank.Validate.col_vals_null)  
Validate whether values in a column are Null.

[Validate.col_vals_not_null()](Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null)  
Validate whether values in a column are not Null.

[Validate.col_vals_regex()](Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex)  
Validate whether column values match a regular expression pattern.

[Validate.col_vals_within_spec()](Validate.col_vals_within_spec.md#pointblank.Validate.col_vals_within_spec)  
Validate whether column values fit within a specification.

[Validate.col_vals_expr()](Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr)  
Validate column values using a custom expression.

[Validate.col_exists()](Validate.col_exists.md#pointblank.Validate.col_exists)  
Validate whether one or more columns exist in the table.

[Validate.col_pct_null()](Validate.col_pct_null.md#pointblank.Validate.col_pct_null)  
Validate whether a column has a specific percentage of Null values.

[Validate.col_pct_missing()](Validate.col_pct_missing.md#pointblank.Validate.col_pct_missing)  
Validate that the percentage of *structured* missing values stays within a limit.

[Validate.col_missing_coded()](Validate.col_missing_coded.md#pointblank.Validate.col_missing_coded)  
Validate that all missing values in a column are *coded* (no uncoded nulls).

[Validate.col_missing_only_coded()](Validate.col_missing_only_coded.md#pointblank.Validate.col_missing_only_coded)  
Validate that a column contains only documented codes and legitimate values.

[Validate.col_missing_consistent()](Validate.col_missing_consistent.md#pointblank.Validate.col_missing_consistent)  
Validate that related columns share a consistent missingness pattern for a given reason.

[Validate.rows_distinct()](Validate.rows_distinct.md#pointblank.Validate.rows_distinct)  
Validate whether rows in the table are distinct.

[Validate.rows_complete()](Validate.rows_complete.md#pointblank.Validate.rows_complete)  
Validate whether row data are complete by having no missing values.

[Validate.col_schema_match()](Validate.col_schema_match.md#pointblank.Validate.col_schema_match)  
Do columns in the table (and their types) match a predefined schema?

[Validate.row_count_match()](Validate.row_count_match.md#pointblank.Validate.row_count_match)  
Validate whether the row count of the table matches a specified count.

[Validate.col_count_match()](Validate.col_count_match.md#pointblank.Validate.col_count_match)  
Validate whether the column count of the table matches a specified count.

[Validate.data_freshness()](Validate.data_freshness.md#pointblank.Validate.data_freshness)  
Validate that data in a datetime column is not older than a specified maximum age.

[Validate.tbl_match()](Validate.tbl_match.md#pointblank.Validate.tbl_match)  
Validate whether the target table matches a comparison table.

[Validate.conjointly()](Validate.conjointly.md#pointblank.Validate.conjointly)  
Perform multiple row-wise validations for joint validity.

[Validate.specially()](Validate.specially.md#pointblank.Validate.specially)  
Perform a specialized validation with customized logic.

[Validate.prompt()](Validate.prompt.md#pointblank.Validate.prompt)  
Validate rows using AI/LLM-powered analysis.


## Aggregation Steps


These validation methods check aggregated column values (sums, averages, standard deviations) against fixed values or column references.


[Validate.col_sum_gt()](Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt)  
Does the column sum satisfy a greater than comparison?

[Validate.col_sum_lt()](Validate.col_sum_lt.md#pointblank.Validate.col_sum_lt)  
Does the column sum satisfy a less than comparison?

[Validate.col_sum_ge()](Validate.col_sum_ge.md#pointblank.Validate.col_sum_ge)  
Does the column sum satisfy a greater than or equal to comparison?

[Validate.col_sum_le()](Validate.col_sum_le.md#pointblank.Validate.col_sum_le)  
Does the column sum satisfy a less than or equal to comparison?

[Validate.col_sum_eq()](Validate.col_sum_eq.md#pointblank.Validate.col_sum_eq)  
Does the column sum satisfy an equal to comparison?

[Validate.col_avg_gt()](Validate.col_avg_gt.md#pointblank.Validate.col_avg_gt)  
Does the column average satisfy a greater than comparison?

[Validate.col_avg_lt()](Validate.col_avg_lt.md#pointblank.Validate.col_avg_lt)  
Does the column average satisfy a less than comparison?

[Validate.col_avg_ge()](Validate.col_avg_ge.md#pointblank.Validate.col_avg_ge)  
Does the column average satisfy a greater than or equal to comparison?

[Validate.col_avg_le()](Validate.col_avg_le.md#pointblank.Validate.col_avg_le)  
Does the column average satisfy a less than or equal to comparison?

[Validate.col_avg_eq()](Validate.col_avg_eq.md#pointblank.Validate.col_avg_eq)  
Does the column average satisfy an equal to comparison?

[Validate.col_sd_gt()](Validate.col_sd_gt.md#pointblank.Validate.col_sd_gt)  
Does the column standard deviation satisfy a greater than comparison?

[Validate.col_sd_lt()](Validate.col_sd_lt.md#pointblank.Validate.col_sd_lt)  
Does the column standard deviation satisfy a less than comparison?

[Validate.col_sd_ge()](Validate.col_sd_ge.md#pointblank.Validate.col_sd_ge)  
Does the column standard deviation satisfy a greater than or equal to comparison?

[Validate.col_sd_le()](Validate.col_sd_le.md#pointblank.Validate.col_sd_le)  
Does the column standard deviation satisfy a less than or equal to comparison?

[Validate.col_sd_eq()](Validate.col_sd_eq.md#pointblank.Validate.col_sd_eq)  
Does the column standard deviation satisfy an equal to comparison?


## Column Selection


Use the `col()` function along with column selection helpers to flexibly select columns for validation. Combine `col()` with `starts_with()`, `matches()`, etc. for selecting multiple target columns.


[col()](col.md#pointblank.col)  
Helper function for referencing a column in the input table.

[starts_with()](starts_with.md#pointblank.starts_with)  
Select columns that start with specified text.

[ends_with()](ends_with.md#pointblank.ends_with)  
Select columns that end with specified text.

[contains()](contains.md#pointblank.contains)  
Select columns that contain specified text.

[matches()](matches.md#pointblank.matches)  
Select columns that match a specified regular expression pattern.

[everything()](everything.md#pointblank.everything)  
Select all columns.

[first_n()](first_n.md#pointblank.first_n)  
Select the first `n` columns in the column list.

[last_n()](last_n.md#pointblank.last_n)  
Select the last `n` columns in the column list.

[expr_col()](expr_col.md#pointblank.expr_col)  
Create a column expression for use in `conjointly()` validation.


## Segment Groups


Combine multiple values into a single segment using `seg_*()` helper functions.


[seg_group()](seg_group.md#pointblank.seg_group)  
Group together values for segmentation.


## Interrogation and Reporting


The validation plan is executed when `interrogate()` is called. After interrogation, view validation reports, extract metrics, or split data based on results.


[Validate.interrogate()](Validate.interrogate.md#pointblank.Validate.interrogate)  
Execute each validation step against the table and store the results.

[Validate.set_tbl()](Validate.set_tbl.md#pointblank.Validate.set_tbl)  
Set or replace the table associated with the Validate object.

[Validate.get_tabular_report()](Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report)  
Validation report as a GT table.

[Validate.get_step_report()](Validate.get_step_report.md#pointblank.Validate.get_step_report)  
Get a detailed report for a single validation step.

[Validate.get_json_report()](Validate.get_json_report.md#pointblank.Validate.get_json_report)  
Get a report of the validation results as a JSON-formatted string.

[Validate.get_dataframe_report()](Validate.get_dataframe_report.md#pointblank.Validate.get_dataframe_report)  
Get a report of the validation results as a DataFrame.

[Validate.to_code()](Validate.to_code.md#pointblank.Validate.to_code)  
Render this validation plan as canonical Pointblank Python code.

[Validate.to_yaml()](Validate.to_yaml.md#pointblank.Validate.to_yaml)  
Serialize this validation plan to a `yaml_interrogate()`-compatible YAML config.

[Validate.suggest_improvements()](Validate.suggest_improvements.md#pointblank.Validate.suggest_improvements)  
Propose AI-generated improvements to this validation plan.

[Validate.from_prompt()](Validate.from_prompt.md#pointblank.Validate.from_prompt)  
Build a validation plan for this table from a natural-language prompt.

[Validate.get_sundered_data()](Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data)  
Get the data that passed or failed the validation steps.

[Validate.get_data_extracts()](Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts)  
Get the rows that failed for each validation step.

[Validate.all_passed()](Validate.all_passed.md#pointblank.Validate.all_passed)  
Determine if every validation step passed perfectly, with no failing test units.

[Validate.assert_passing()](Validate.assert_passing.md#pointblank.Validate.assert_passing)  
Raise an `AssertionError` if all tests are not passing.

[Validate.assert_below_threshold()](Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold)  
Raise an `AssertionError` if validation steps exceed a specified threshold level.

[Validate.assert_dimension_scores()](Validate.assert_dimension_scores.md#pointblank.Validate.assert_dimension_scores)  
Raise an `AssertionError` if any dimension's health score falls below a minimum.

[Validate.above_threshold()](Validate.above_threshold.md#pointblank.Validate.above_threshold)  
Check if any validation steps exceed a specified threshold level.

[Validate.get_dimension_scores()](Validate.get_dimension_scores.md#pointblank.Validate.get_dimension_scores)  
Get per-dimension health scores from the validation results.

[Validate.get_health_score()](Validate.get_health_score.md#pointblank.Validate.get_health_score)  
Get the overall data quality health score from the validation results.

[Validate.get_scorecard()](Validate.get_scorecard.md#pointblank.Validate.get_scorecard)  
Get a data quality scorecard as a GT table.

[Validate.n()](Validate.n.md#pointblank.Validate.n)  
Provides a dictionary of the number of test units for each validation step.

[Validate.n_passed()](Validate.n_passed.md#pointblank.Validate.n_passed)  
Provides a dictionary of the number of test units that passed for each validation step.

[Validate.n_failed()](Validate.n_failed.md#pointblank.Validate.n_failed)  
Provides a dictionary of the number of test units that failed for each validation step.

[Validate.f_passed()](Validate.f_passed.md#pointblank.Validate.f_passed)  
Provides a dictionary of the fraction of test units that passed for each validation step.

[Validate.f_failed()](Validate.f_failed.md#pointblank.Validate.f_failed)  
Provides a dictionary of the fraction of test units that failed for each validation step.

[Validate.warning()](Validate.warning.md#pointblank.Validate.warning)  
Get the 'warning' level status for each validation step.

[Validate.error()](Validate.error.md#pointblank.Validate.error)  
Get the 'error' level status for each validation step.

[Validate.critical()](Validate.critical.md#pointblank.Validate.critical)  
Get the 'critical' level status for each validation step.


## Inspection and Assistance


Functions for getting to grips with a new data table. Use `DataScan` for a quick overview, `preview()` for first/last rows, `col_summary_tbl()` for column summaries, and `missing_vals_tbl()` for missing value analysis.


[DataScan](DataScan.md#pointblank.DataScan)  
Get a summary of a dataset.

[preview()](preview.md#pointblank.preview)  
Display a table preview that shows some rows from the top, some from the bottom.

[col_summary_tbl()](col_summary_tbl.md#pointblank.col_summary_tbl)  
Generate a column-level summary table of a dataset.

[missing_vals_tbl()](missing_vals_tbl.md#pointblank.missing_vals_tbl)  
Display a table that shows the missing values in the input table.

[load_dataset()](load_dataset.md#pointblank.load_dataset)  
Load a dataset hosted in the library as specified table type.

[get_data_path()](get_data_path.md#pointblank.get_data_path)  
Get the file path to a dataset included with the Pointblank package.

[connect_to_table()](connect_to_table.md#pointblank.connect_to_table)  
Connect to a database table using a connection string.

[print_database_tables()](print_database_tables.md#pointblank.print_database_tables)  
List all tables in a database from a connection string.


## Table Pre-checks


Helper functions for use with the `active=` parameter of validation methods. These inspect the target table before a step runs and conditionally skip the step when preconditions are not met.


[has_columns()](has_columns.md#pointblank.has_columns)  
Check whether one or more columns exist in a table.

[has_rows()](has_rows.md#pointblank.has_rows)  
Check whether a table has a certain number of rows.


## YAML


Functions for using YAML to orchestrate validation workflows.


[yaml_interrogate()](yaml_interrogate.md#pointblank.yaml_interrogate)  
Execute a YAML-based validation workflow.

[validate_yaml()](validate_yaml.md#pointblank.validate_yaml)  
Validate YAML configuration against the expected structure.

[yaml_to_python()](yaml_to_python.md#pointblank.yaml_to_python)  
Convert YAML validation configuration to equivalent Python code.


## Utility Functions


Functions for accessing metadata about the target data and managing configuration.


[get_column_count()](get_column_count.md#pointblank.get_column_count)  
Get the number of columns in a table.

[get_row_count()](get_row_count.md#pointblank.get_row_count)  
Get the number of rows in a table.

[get_action_metadata()](get_action_metadata.md#pointblank.get_action_metadata)  
Access step-level metadata when authoring custom actions.

[get_validation_summary()](get_validation_summary.md#pointblank.get_validation_summary)  
Access validation summary information when authoring final actions.

[write_file()](write_file.md#pointblank.write_file)  
Write a Validate object to disk as a serialized file.

[read_file()](read_file.md#pointblank.read_file)  
Read a Validate object from disk that was previously saved with `write_file()`.

[ref()](ref.md#pointblank.ref)  
Reference a column from the reference data for aggregate comparisons.


## Test Data Generation


Generate synthetic test data based on schema definitions. Use `generate_dataset()` to create data from a Schema object, or `schema_from_tbl()` to infer a generation-ready schema from an existing table (Polars, Pandas, or Ibis/DuckDB).


[generate_dataset()](generate_dataset.md#pointblank.generate_dataset)  
Generate synthetic test data from a schema.

[schema_from_tbl()](schema_from_tbl.md#pointblank.schema_from_tbl)  
Create a Schema from an existing table with inferred Field constraints.

[int_field()](int_field.md#pointblank.int_field)  
Create an integer column specification for use in a schema.

[float_field()](float_field.md#pointblank.float_field)  
Create a floating-point column specification for use in a schema.

[string_field()](string_field.md#pointblank.string_field)  
Create a string column specification for use in a schema.

[bool_field()](bool_field.md#pointblank.bool_field)  
Create a boolean column specification for use in a schema.

[date_field()](date_field.md#pointblank.date_field)  
Create a date column specification for use in a schema.

[datetime_field()](datetime_field.md#pointblank.datetime_field)  
Create a datetime column specification for use in a schema.

[time_field()](time_field.md#pointblank.time_field)  
Create a time column specification for use in a schema.

[duration_field()](duration_field.md#pointblank.duration_field)  
Create a duration column specification for use in a schema.

[profile_fields()](profile_fields.md#pointblank.profile_fields)  
Create a dict of string field specifications representing a person profile.


## Prebuilt Actions


Prebuilt action functions for common notification patterns.


[send_slack_notification()](send_slack_notification.md#pointblank.send_slack_notification)  
Create a Slack notification function using a webhook URL.

[emit_otel()](emit_otel.md#pointblank.emit_otel)  
Create an OTel export action for use in `FinalActions`.


## Metadata Import/Export


Import variable-level metadata from external data standards files (CDISC Define-XML, Controlled Terminology, SPSS `.sav`, SAS XPORT, Stata `.dta`, and more) and export metadata to various formats. Use `import_metadata()` as the entry point and `export_metadata()` for the reverse.


[import_metadata()](import_metadata.md#pointblank.import_metadata)  
Import metadata from an external standard or file.

[export_metadata()](export_metadata.md#pointblank.export_metadata)  
Export metadata to an external standard format.

[MetadataImport](MetadataImport.md#pointblank.MetadataImport)  
Parsed metadata from an external standard.

[MetadataPackage](MetadataPackage.md#pointblank.MetadataPackage)  
A collection of `MetadataImport` objects from a multi-dataset source.

[VariableMetadata](VariableMetadata.md#pointblank.VariableMetadata)  
Metadata for a single variable/column, as imported from an external standard.

[Codelist](Codelist.md#pointblank.Codelist)  
A controlled terminology / value set from an external standard.

[CodelistEntry](CodelistEntry.md#pointblank.CodelistEntry)  
A single entry in a codelist (controlled terminology).

[MissingValueCode](MissingValueCode.md#pointblank.MissingValueCode)  
A structured missing value definition from an external standard.


## SDTM Validation


Validate clinical datasets against CDISC SDTM domain templates. Use `validate_sdtm()` to generate a full `Validate` workflow, or `validate_sdtm_structure()` for a quick structural conformance check. Retrieve domain templates with `get_sdtm_domain()` and `list_sdtm_domains()`.


[validate_sdtm()](validate_sdtm.md#pointblank.validate_sdtm)  
Generate a comprehensive SDTM validation workflow for a dataset.

[validate_sdtm_structure()](validate_sdtm_structure.md#pointblank.validate_sdtm_structure)  
Validate the structural conformance of a dataset against an SDTM domain template.

[sdtm_to_metadata()](sdtm_to_metadata.md#pointblank.sdtm_to_metadata)  
Convert an SDTM domain template to a `MetadataImport` object.

[get_sdtm_domain()](get_sdtm_domain.md#pointblank.get_sdtm_domain)  
Get the SDTM template for a specific domain.

[list_sdtm_domains()](list_sdtm_domains.md#pointblank.list_sdtm_domains)  
List all available SDTM domain codes.

[SDTMDomainTemplate](SDTMDomainTemplate.md#pointblank.SDTMDomainTemplate)  
Structural template for an SDTM domain.

[SDTMVariableSpec](SDTMVariableSpec.md#pointblank.SDTMVariableSpec)  
Specification for a single variable in an SDTM domain template.


## ADaM Validation


Validate analysis datasets against CDISC ADaM templates. Use `validate_adam()` to generate a full `Validate` workflow, or `validate_adam_structure()` for a quick structural conformance check. Retrieve dataset templates with `get_adam_dataset()` and `list_adam_datasets()`.


[validate_adam()](validate_adam.md#pointblank.validate_adam)  
Generate a comprehensive ADaM validation workflow for a dataset.

[validate_adam_structure()](validate_adam_structure.md#pointblank.validate_adam_structure)  
Validate structural conformance of a dataset against an ADaM template.

[adam_to_metadata()](adam_to_metadata.md#pointblank.adam_to_metadata)  
Convert an ADaM dataset template to a MetadataImport object.

[get_adam_dataset()](get_adam_dataset.md#pointblank.get_adam_dataset)  
Get the ADaM template for a specific dataset.

[list_adam_datasets()](list_adam_datasets.md#pointblank.list_adam_datasets)  
List all available ADaM dataset template names.

[ADaMDatasetTemplate](ADaMDatasetTemplate.md#pointblank.ADaMDatasetTemplate)  
Structural template for an ADaM dataset.

[ADaMVariableSpec](ADaMVariableSpec.md#pointblank.ADaMVariableSpec)  
Specification for a single variable in an ADaM dataset template.


## Integrations


Classes for integrating Pointblank with external observability and monitoring systems. Use `OTelExporter` to export validation results as OpenTelemetry metrics, traces, and logs.


[integrations.otel.OTelExporter](integrations.otel.OTelExporter.md#pointblank.integrations.otel.OTelExporter)  
Export Pointblank validation results as OpenTelemetry signals.
