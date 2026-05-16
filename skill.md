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

### Validation Steps

Validation steps are sequential validations on the target data. Call Validate's validation methods to build up a validation plan: a collection of steps that provides good validation coverage.


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
- `Validate.get_sundered_data`
- `Validate.get_data_extracts`
- `Validate.all_passed`
- `Validate.assert_passing`
- `Validate.assert_below_threshold`
- `Validate.above_threshold`
- `Validate.n`
- `Validate.n_passed`
- `Validate.n_failed`
- `Validate.f_passed`
- `Validate.f_failed`
- `Validate.warning`
- `Validate.error`
- `Validate.critical`

### Inspection and Assistance

Functions for getting to grips with a new data table. Use DataScan for a quick overview, `preview()` for first/last rows, `col_summary_tbl()` for column summaries, and `missing_vals_tbl()` for missing value analysis.


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

### Integrations

Classes for integrating Pointblank with external observability and monitoring systems. Use `OTelExporter` to export validation results as OpenTelemetry metrics, traces, and logs.


- `integrations.otel.OTelExporter`

## Resources

- [llms.txt](llms.txt) — Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) — Comprehensive documentation for LLMs
