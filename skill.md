---
name: pointblank
description: >
  Find out if your data is what you think it is. Use when writing Python code that uses the pointblank package.
license: MIT
compatibility: Requires Python >=3.10.
---

# Pointblank

Find out if your data is what you think it is.

## Installation

```bash
pip install pointblank
```

## API overview

### Validate

When performing data validation, use the `Validate` class to get the process started. It takes the target table and options for metadata and failure thresholds (using the `Thresholds` class or shorthands). The `Validate` class has numerous methods for defining validation steps and for obtaining post-interrogation metrics and data.


- `Validate`: Workflow for defining a set of validations on a table and interrogating for results
- `Thresholds`: Definition of threshold values
- `Actions`: Definition of action values
- `FinalActions`: Define actions to be taken after validation is complete
- `Schema`: Definition of a schema object
- `DraftValidation`: Draft a validation plan for a given table using an LLM
- `EditValidation`: Edit an existing validation plan with a plain-English instruction using an LLM
- `MissingSpec`: Specification for structured missing values in a column

### Contracts and Pipelines

Use `Contract` and `Step` to define declarative data quality contracts that specify what valid data looks like. Use `Pipeline` to enforce contracts at both boundaries of a data transformation (source and target), producing a `PipelineResult` with full introspection into what passed and what failed.


- `Contract`: A declarative boundary contract for pipeline data
- `Step`: A single validation step in a Contract, defined declaratively
- `Pipeline`: Binds source and target contracts into a pipeline boundary enforcement unit
- `PipelineResult`: Result of a pipeline boundary validation run

### Contract Import/Export

Import external schema definitions (JSON Schema, Frictionless Table Schema, and more) into Pointblank validation workflows, or export Pointblank contracts to those formats. Use `import_contract()` as the entry point, `export_contract()` for the reverse, and `register_adapter()` to add support for custom formats.


- `import_contract`: Import a contract/schema from an external format
- `export_contract`: Export a Pointblank validation or contract to an external format
- `list_adapters`: List all registered adapters with their capabilities
- `register_adapter`: Register a contract adapter class
- `ContractImport`: Result of importing an external contract/schema
- `ContractAdapter`: Base class for contract import/export adapters

### Validation Steps

Validation steps are sequential validations on the target data. Call `Validate`'s validation methods to build up a validation plan: a collection of steps that provides good validation coverage.


- `Validate.col_vals_gt`
- `Validate.col_vals_lt`
- `Validate.col_vals_ge`
- `Validate.col_vals_le`
- `Validate.col_vals_eq`
- `Validate.col_vals_ne`
- `Validate.col_vals_between`
- `Validate.col_vals_outside`
- `Validate.col_vals_in_set`
- `Validate.col_vals_not_in_set`
- `Validate.col_vals_increasing`
- `Validate.col_vals_decreasing`
- `Validate.col_vals_null`
- `Validate.col_vals_not_null`
- `Validate.col_vals_regex`
- `Validate.col_vals_within_spec`
- `Validate.col_vals_expr`
- `Validate.col_exists`
- `Validate.col_pct_null`
- `Validate.col_pct_missing`
- `Validate.col_missing_coded`
- `Validate.col_missing_only_coded`
- `Validate.col_missing_consistent`
- `Validate.rows_distinct`
- `Validate.rows_complete`
- `Validate.col_schema_match`
- `Validate.row_count_match`
- `Validate.col_count_match`
- `Validate.data_freshness`
- `Validate.tbl_match`
- `Validate.conjointly`
- `Validate.specially`
- `Validate.prompt`

### Aggregation Steps

These validation methods check aggregated column values (sums, averages, standard deviations) against fixed values or column references.


- `Validate.col_sum_gt`
- `Validate.col_sum_lt`
- `Validate.col_sum_ge`
- `Validate.col_sum_le`
- `Validate.col_sum_eq`
- `Validate.col_avg_gt`
- `Validate.col_avg_lt`
- `Validate.col_avg_ge`
- `Validate.col_avg_le`
- `Validate.col_avg_eq`
- `Validate.col_sd_gt`
- `Validate.col_sd_lt`
- `Validate.col_sd_ge`
- `Validate.col_sd_le`
- `Validate.col_sd_eq`

### Column Selection

Use the `col()` function along with column selection helpers to flexibly select columns for validation. Combine `col()` with `starts_with()`, `matches()`, etc. for selecting multiple target columns.


- `col`: Helper function for referencing a column in the input table
- `starts_with`: Select columns that start with specified text
- `ends_with`: Select columns that end with specified text
- `contains`: Select columns that contain specified text
- `matches`: Select columns that match a specified regular expression pattern
- `everything`: Select all columns
- `first_n`: Select the first `n` columns in the column list
- `last_n`: Select the last `n` columns in the column list
- `expr_col`: Create a column expression for use in `conjointly()` validation

### Segment Groups

Combine multiple values into a single segment using `seg_*()` helper functions.


- `seg_group`: Group together values for segmentation

### Interrogation and Reporting

The validation plan is executed when `interrogate()` is called. After interrogation, view validation reports, extract metrics, or split data based on results.


- `Validate.interrogate`
- `Validate.set_tbl`
- `Validate.get_tabular_report`
- `Validate.get_step_report`
- `Validate.get_json_report`
- `Validate.get_dataframe_report`
- `Validate.to_code`
- `Validate.to_yaml`
- `Validate.suggest_improvements`
- `Validate.from_prompt`
- `Validate.get_sundered_data`
- `Validate.get_data_extracts`
- `Validate.all_passed`
- `Validate.assert_passing`
- `Validate.assert_below_threshold`
- `Validate.assert_dimension_scores`
- `Validate.above_threshold`
- `Validate.get_dimension_scores`
- `Validate.get_health_score`
- `Validate.get_scorecard`
- `Validate.n`
- `Validate.n_passed`
- `Validate.n_failed`
- `Validate.f_passed`
- `Validate.f_failed`
- `Validate.warning`
- `Validate.error`
- `Validate.critical`

### Inspection and Assistance

Functions for getting to grips with a new data table. Use `DataScan` for a quick overview, `preview()` for first/last rows, `col_summary_tbl()` for column summaries, and `missing_vals_tbl()` for missing value analysis.


- `DataScan`: Get a summary of a dataset
- `preview`: Display a table preview that shows some rows from the top, some from the bottom
- `col_summary_tbl`: Generate a column-level summary table of a dataset
- `missing_vals_tbl`: Display a table that shows the missing values in the input table
- `load_dataset`: Load a dataset hosted in the library as specified table type
- `get_data_path`: Get the file path to a dataset included with the Pointblank package
- `connect_to_table`: Connect to a database table using a connection string
- `print_database_tables`: List all tables in a database from a connection string

### Table Pre-checks

Helper functions for use with the `active=` parameter of validation methods. These inspect the target table before a step runs and conditionally skip the step when preconditions are not met.


- `has_columns`: Check whether one or more columns exist in a table
- `has_rows`: Check whether a table has a certain number of rows

### YAML

Functions for using YAML to orchestrate validation workflows.


- `yaml_interrogate`: Execute a YAML-based validation workflow
- `validate_yaml`: Validate YAML configuration against the expected structure
- `yaml_to_python`: Convert YAML validation configuration to equivalent Python code

### Utility Functions

Functions for accessing metadata about the target data and managing configuration.


- `get_column_count`: Get the number of columns in a table
- `get_row_count`: Get the number of rows in a table
- `get_action_metadata`: Access step-level metadata when authoring custom actions
- `get_validation_summary`: Access validation summary information when authoring final actions
- `write_file`: Write a Validate object to disk as a serialized file
- `read_file`: Read a Validate object from disk that was previously saved with `write_file()`
- `ref`: Reference a column from the reference data for aggregate comparisons

### Test Data Generation

Generate synthetic test data based on schema definitions. Use `generate_dataset()` to create data from a Schema object, or `schema_from_tbl()` to infer a generation-ready schema from an existing table (Polars, Pandas, or Ibis/DuckDB).


- `generate_dataset`: Generate synthetic test data from a schema
- `schema_from_tbl`: Create a Schema from an existing table with inferred Field constraints
- `int_field`: Create an integer column specification for use in a schema
- `float_field`: Create a floating-point column specification for use in a schema
- `string_field`: Create a string column specification for use in a schema
- `bool_field`: Create a boolean column specification for use in a schema
- `date_field`: Create a date column specification for use in a schema
- `datetime_field`: Create a datetime column specification for use in a schema
- `time_field`: Create a time column specification for use in a schema
- `duration_field`: Create a duration column specification for use in a schema
- `profile_fields`: Create a dict of string field specifications representing a person profile

### Prebuilt Actions

Prebuilt action functions for common notification patterns.


- `send_slack_notification`: Create a Slack notification function using a webhook URL
- `emit_otel`: Create an OTel export action for use in `FinalActions`

### Metadata Import/Export

Import variable-level metadata from external data standards files (CDISC Define-XML, Controlled Terminology, SPSS `.sav`, SAS XPORT, Stata `.dta`, and more) and export metadata to various formats. Use `import_metadata()` as the entry point and `export_metadata()` for the reverse.


- `import_metadata`: Import metadata from an external standard or file
- `export_metadata`: Export metadata to an external standard format
- `MetadataImport`: Parsed metadata from an external standard
- `MetadataPackage`: A collection of `MetadataImport` objects from a multi-dataset source
- `VariableMetadata`: Metadata for a single variable/column, as imported from an external standard
- `Codelist`: A controlled terminology / value set from an external standard
- `CodelistEntry`: A single entry in a codelist (controlled terminology)
- `MissingValueCode`: A structured missing value definition from an external standard

### SDTM Validation

Validate clinical datasets against CDISC SDTM domain templates. Use `validate_sdtm()` to generate a full `Validate` workflow, or `validate_sdtm_structure()` for a quick structural conformance check. Retrieve domain templates with `get_sdtm_domain()` and `list_sdtm_domains()`.


- `validate_sdtm`: Generate a comprehensive SDTM validation workflow for a dataset
- `validate_sdtm_structure`: Validate the structural conformance of a dataset against an SDTM domain template
- `sdtm_to_metadata`: Convert an SDTM domain template to a `MetadataImport` object
- `get_sdtm_domain`: Get the SDTM template for a specific domain
- `list_sdtm_domains`: List all available SDTM domain codes
- `SDTMDomainTemplate`: Structural template for an SDTM domain
- `SDTMVariableSpec`: Specification for a single variable in an SDTM domain template

### ADaM Validation

Validate analysis datasets against CDISC ADaM templates. Use `validate_adam()` to generate a full `Validate` workflow, or `validate_adam_structure()` for a quick structural conformance check. Retrieve dataset templates with `get_adam_dataset()` and `list_adam_datasets()`.


- `validate_adam`: Generate a comprehensive ADaM validation workflow for a dataset
- `validate_adam_structure`: Validate structural conformance of a dataset against an ADaM template
- `adam_to_metadata`: Convert an ADaM dataset template to a MetadataImport object
- `get_adam_dataset`: Get the ADaM template for a specific dataset
- `list_adam_datasets`: List all available ADaM dataset template names
- `ADaMDatasetTemplate`: Structural template for an ADaM dataset
- `ADaMVariableSpec`: Specification for a single variable in an ADaM dataset template

### CDISC Submission Conformance

Validate SDTM datasets for CDISC conformance. `validate_sdtmig()` is the primary entry point: pass a dictionary of domain DataFrames and receive a `ConformanceReport` with 426 SDTMIG 3.4 rules evaluated in-process. For full submission-package checks (cross-dataset referential integrity, SUPP-- linkage, Define-XML) use `SubmissionPackage`. For the authoritative CDISC-certified rule set, use `validate_cdisc_submission()` (requires the CORE CLI) or `SubmissionPackage.validate_conformance(engine="core")`. All paths return a `ConformanceReport`.


- `validate_sdtmig`: Validate SDTM datasets against the SDTMIG rule catalog and return a conformance report
- `validate_cdisc_submission`: Validate a CDISC submission with the CDISC CORE engine, in one call
- `SubmissionPackage`: A data-level model of a study submission package for CDISC conformance validation
- `ConformanceReport`: The result of a CDISC conformance validation run

### Integrations

Classes for integrating Pointblank with external observability and monitoring systems. Use `OTelExporter` to export validation results as OpenTelemetry metrics, traces, and logs.


- `integrations.otel.OTelExporter`

## Resources

- [Full documentation](https://posit-dev.github.io/pointblank/)
- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs
