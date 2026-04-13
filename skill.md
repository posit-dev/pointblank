# Pointblank

Find out if your data is what you think it is.


# Installation

``` bash
pip install pointblank
```


# API overview


## Validate

When performing data validation, use the [Validate](reference/Validate.html#pointblank.Validate) class to get the process started. It takes the target table and options for metadata and failure thresholds (using the [Thresholds](reference/Thresholds.html#pointblank.Thresholds) class or shorthands). The [Validate](reference/Validate.html#pointblank.Validate) class has numerous methods for defining validation steps and for obtaining post-interrogation metrics and data.

- [Validate](reference/Validate.html#pointblank.Validate): Workflow for defining a set of validations on a table and interrogating for results
- [Thresholds](reference/Thresholds.html#pointblank.Thresholds): Definition of threshold values
- [Actions](reference/Actions.html#pointblank.Actions): Definition of action values
- [FinalActions](reference/FinalActions.html#pointblank.FinalActions): Define actions to be taken after validation is complete
- [Schema](reference/Schema.html#pointblank.Schema): Definition of a schema object
- [DraftValidation](reference/DraftValidation.html#pointblank.DraftValidation): Draft a validation plan for a given table using an LLM


## Validation Steps

Validation steps are sequential validations on the target data. Call Validate's validation methods to build up a validation plan: a collection of steps that provides good validation coverage.

- [Validate.col_vals_gt](reference/Validate.col_vals_gt.html#pointblank.Validate.col_vals_gt)
- [Validate.col_vals_lt](reference/Validate.col_vals_lt.html#pointblank.Validate.col_vals_lt)
- [Validate.col_vals_ge](reference/Validate.col_vals_ge.html#pointblank.Validate.col_vals_ge)
- [Validate.col_vals_le](reference/Validate.col_vals_le.html#pointblank.Validate.col_vals_le)
- [Validate.col_vals_eq](reference/Validate.col_vals_eq.html#pointblank.Validate.col_vals_eq)
- [Validate.col_vals_ne](reference/Validate.col_vals_ne.html#pointblank.Validate.col_vals_ne)
- [Validate.col_vals_between](reference/Validate.col_vals_between.html#pointblank.Validate.col_vals_between)
- [Validate.col_vals_outside](reference/Validate.col_vals_outside.html#pointblank.Validate.col_vals_outside)
- [Validate.col_vals_in_set](reference/Validate.col_vals_in_set.html#pointblank.Validate.col_vals_in_set)
- [Validate.col_vals_not_in_set](reference/Validate.col_vals_not_in_set.html#pointblank.Validate.col_vals_not_in_set)
- [Validate.col_vals_increasing](reference/Validate.col_vals_increasing.html#pointblank.Validate.col_vals_increasing)
- [Validate.col_vals_decreasing](reference/Validate.col_vals_decreasing.html#pointblank.Validate.col_vals_decreasing)
- [Validate.col_vals_null](reference/Validate.col_vals_null.html#pointblank.Validate.col_vals_null)
- [Validate.col_vals_not_null](reference/Validate.col_vals_not_null.html#pointblank.Validate.col_vals_not_null)
- [Validate.col_vals_regex](reference/Validate.col_vals_regex.html#pointblank.Validate.col_vals_regex)
- [Validate.col_vals_within_spec](reference/Validate.col_vals_within_spec.html#pointblank.Validate.col_vals_within_spec)
- [Validate.col_vals_expr](reference/Validate.col_vals_expr.html#pointblank.Validate.col_vals_expr)
- [Validate.col_exists](reference/Validate.col_exists.html#pointblank.Validate.col_exists)
- [Validate.col_pct_null](reference/Validate.col_pct_null.html#pointblank.Validate.col_pct_null)
- [Validate.rows_distinct](reference/Validate.rows_distinct.html#pointblank.Validate.rows_distinct)
- [Validate.rows_complete](reference/Validate.rows_complete.html#pointblank.Validate.rows_complete)
- [Validate.col_schema_match](reference/Validate.col_schema_match.html#pointblank.Validate.col_schema_match)
- [Validate.row_count_match](reference/Validate.row_count_match.html#pointblank.Validate.row_count_match)
- [Validate.col_count_match](reference/Validate.col_count_match.html#pointblank.Validate.col_count_match)
- [Validate.data_freshness](reference/Validate.data_freshness.html#pointblank.Validate.data_freshness)
- [Validate.tbl_match](reference/Validate.tbl_match.html#pointblank.Validate.tbl_match)
- [Validate.conjointly](reference/Validate.conjointly.html#pointblank.Validate.conjointly)
- [Validate.specially](reference/Validate.specially.html#pointblank.Validate.specially)
- [Validate.prompt](reference/Validate.prompt.html#pointblank.Validate.prompt)


## Aggregation Steps

These validation methods check aggregated column values (sums, averages, standard deviations) against fixed values or column references.

- [Validate.col_sum_gt](reference/Validate.col_sum_gt.html#pointblank.Validate.col_sum_gt)
- [Validate.col_sum_lt](reference/Validate.col_sum_lt.html#pointblank.Validate.col_sum_lt)
- [Validate.col_sum_ge](reference/Validate.col_sum_ge.html#pointblank.Validate.col_sum_ge)
- [Validate.col_sum_le](reference/Validate.col_sum_le.html#pointblank.Validate.col_sum_le)
- [Validate.col_sum_eq](reference/Validate.col_sum_eq.html#pointblank.Validate.col_sum_eq)
- [Validate.col_avg_gt](reference/Validate.col_avg_gt.html#pointblank.Validate.col_avg_gt)
- [Validate.col_avg_lt](reference/Validate.col_avg_lt.html#pointblank.Validate.col_avg_lt)
- [Validate.col_avg_ge](reference/Validate.col_avg_ge.html#pointblank.Validate.col_avg_ge)
- [Validate.col_avg_le](reference/Validate.col_avg_le.html#pointblank.Validate.col_avg_le)
- [Validate.col_avg_eq](reference/Validate.col_avg_eq.html#pointblank.Validate.col_avg_eq)
- [Validate.col_sd_gt](reference/Validate.col_sd_gt.html#pointblank.Validate.col_sd_gt)
- [Validate.col_sd_lt](reference/Validate.col_sd_lt.html#pointblank.Validate.col_sd_lt)
- [Validate.col_sd_ge](reference/Validate.col_sd_ge.html#pointblank.Validate.col_sd_ge)
- [Validate.col_sd_le](reference/Validate.col_sd_le.html#pointblank.Validate.col_sd_le)
- [Validate.col_sd_eq](reference/Validate.col_sd_eq.html#pointblank.Validate.col_sd_eq)


## Column Selection

Use the [col()](reference/col.html#pointblank.col) function along with column selection helpers to flexibly select columns for validation. Combine [col()](reference/col.html#pointblank.col) with [starts_with()](reference/starts_with.html#pointblank.starts_with), [matches()](reference/matches.html#pointblank.matches), etc. for selecting multiple target columns.

- [col](reference/col.html#pointblank.col): Helper function for referencing a column in the input table
- [starts_with](reference/starts_with.html#pointblank.starts_with): Select columns that start with specified text
- [ends_with](reference/ends_with.html#pointblank.ends_with): Select columns that end with specified text
- [contains](reference/contains.html#pointblank.contains): Select columns that contain specified text
- [matches](reference/matches.html#pointblank.matches): Select columns that match a specified regular expression pattern
- [everything](reference/everything.html#pointblank.everything): Select all columns
- [first_n](reference/first_n.html#pointblank.first_n): Select the first [n](reference/Validate.n.html#pointblank.Validate.n) columns in the column list
- [last_n](reference/last_n.html#pointblank.last_n): Select the last [n](reference/Validate.n.html#pointblank.Validate.n) columns in the column list
- [expr_col](reference/expr_col.html#pointblank.expr_col): Create a column expression for use in [conjointly()](reference/Validate.conjointly.html#pointblank.Validate.conjointly) validation


## Segment Groups

Combine multiple values into a single segment using `seg_*()` helper functions.

- [seg_group](reference/seg_group.html#pointblank.seg_group): Group together values for segmentation


## Interrogation and Reporting

The validation plan is executed when [interrogate()](reference/Validate.interrogate.html#pointblank.Validate.interrogate) is called. After interrogation, view validation reports, extract metrics, or split data based on results.

- [Validate.interrogate](reference/Validate.interrogate.html#pointblank.Validate.interrogate)
- [Validate.set_tbl](reference/Validate.set_tbl.html#pointblank.Validate.set_tbl)
- [Validate.get_tabular_report](reference/Validate.get_tabular_report.html#pointblank.Validate.get_tabular_report)
- [Validate.get_step_report](reference/Validate.get_step_report.html#pointblank.Validate.get_step_report)
- [Validate.get_json_report](reference/Validate.get_json_report.html#pointblank.Validate.get_json_report)
- [Validate.get_sundered_data](reference/Validate.get_sundered_data.html#pointblank.Validate.get_sundered_data)
- [Validate.get_data_extracts](reference/Validate.get_data_extracts.html#pointblank.Validate.get_data_extracts)
- [Validate.all_passed](reference/Validate.all_passed.html#pointblank.Validate.all_passed)
- [Validate.assert_passing](reference/Validate.assert_passing.html#pointblank.Validate.assert_passing)
- [Validate.assert_below_threshold](reference/Validate.assert_below_threshold.html#pointblank.Validate.assert_below_threshold)
- [Validate.above_threshold](reference/Validate.above_threshold.html#pointblank.Validate.above_threshold)
- [Validate.n](reference/Validate.n.html#pointblank.Validate.n)
- [Validate.n_passed](reference/Validate.n_passed.html#pointblank.Validate.n_passed)
- [Validate.n_failed](reference/Validate.n_failed.html#pointblank.Validate.n_failed)
- [Validate.f_passed](reference/Validate.f_passed.html#pointblank.Validate.f_passed)
- [Validate.f_failed](reference/Validate.f_failed.html#pointblank.Validate.f_failed)
- [Validate.warning](reference/Validate.warning.html#pointblank.Validate.warning)
- [Validate.error](reference/Validate.error.html#pointblank.Validate.error)
- [Validate.critical](reference/Validate.critical.html#pointblank.Validate.critical)


## Inspection and Assistance

Functions for getting to grips with a new data table. Use DataScan for a quick overview, [preview()](reference/preview.html#pointblank.preview) for first/last rows, [col_summary_tbl()](reference/col_summary_tbl.html#pointblank.col_summary_tbl) for column summaries, and [missing_vals_tbl()](reference/missing_vals_tbl.html#pointblank.missing_vals_tbl) for missing value analysis.

- [DataScan](reference/DataScan.html#pointblank.DataScan): Get a summary of a dataset
- [preview](reference/preview.html#pointblank.preview): Display a table preview that shows some rows from the top, some from the bottom
- [col_summary_tbl](reference/col_summary_tbl.html#pointblank.col_summary_tbl): Generate a column-level summary table of a dataset
- [missing_vals_tbl](reference/missing_vals_tbl.html#pointblank.missing_vals_tbl): Display a table that shows the missing values in the input table
- [load_dataset](reference/load_dataset.html#pointblank.load_dataset): Load a dataset hosted in the library as specified table type
- [get_data_path](reference/get_data_path.html#pointblank.get_data_path): Get the file path to a dataset included with the Pointblank package
- [connect_to_table](reference/connect_to_table.html#pointblank.connect_to_table): Connect to a database table using a connection string
- [print_database_tables](reference/print_database_tables.html#pointblank.print_database_tables): List all tables in a database from a connection string


## Table Pre-checks

Helper functions for use with the `active=` parameter of validation methods. These inspect the target table before a step runs and conditionally skip the step when preconditions are not met.

- [has_columns](reference/has_columns.html#pointblank.has_columns): Check whether one or more columns exist in a table
- [has_rows](reference/has_rows.html#pointblank.has_rows): Check whether a table has a certain number of rows


## YAML

Functions for using YAML to orchestrate validation workflows.

- [yaml_interrogate](reference/yaml_interrogate.html#pointblank.yaml_interrogate): Execute a YAML-based validation workflow
- [validate_yaml](reference/validate_yaml.html#pointblank.validate_yaml): Validate YAML configuration against the expected structure
- [yaml_to_python](reference/yaml_to_python.html#pointblank.yaml_to_python): Convert YAML validation configuration to equivalent Python code


## Utility Functions

Functions for accessing metadata about the target data and managing configuration.

- [get_column_count](reference/get_column_count.html#pointblank.get_column_count): Get the number of columns in a table
- [get_row_count](reference/get_row_count.html#pointblank.get_row_count): Get the number of rows in a table
- [get_action_metadata](reference/get_action_metadata.html#pointblank.get_action_metadata): Access step-level metadata when authoring custom actions
- [get_validation_summary](reference/get_validation_summary.html#pointblank.get_validation_summary): Access validation summary information when authoring final actions
- [write_file](reference/write_file.html#pointblank.write_file): Write a Validate object to disk as a serialized file
- [read_file](reference/read_file.html#pointblank.read_file): Read a Validate object from disk that was previously saved with [write_file()](reference/write_file.html#pointblank.write_file)
- [ref](reference/ref.html#pointblank.ref): Reference a column from the reference data for aggregate comparisons


## Test Data Generation

Generate synthetic test data based on schema definitions. Use [generate_dataset()](reference/generate_dataset.html#pointblank.generate_dataset) to create data from a Schema object.

- [generate_dataset](reference/generate_dataset.html#pointblank.generate_dataset): Generate synthetic test data from a schema
- [int_field](reference/int_field.html#pointblank.int_field): Create an integer column specification for use in a schema
- [float_field](reference/float_field.html#pointblank.float_field): Create a floating-point column specification for use in a schema
- [string_field](reference/string_field.html#pointblank.string_field): Create a string column specification for use in a schema
- [bool_field](reference/bool_field.html#pointblank.bool_field): Create a boolean column specification for use in a schema
- [date_field](reference/date_field.html#pointblank.date_field): Create a date column specification for use in a schema
- [datetime_field](reference/datetime_field.html#pointblank.datetime_field): Create a datetime column specification for use in a schema
- [time_field](reference/time_field.html#pointblank.time_field): Create a time column specification for use in a schema
- [duration_field](reference/duration_field.html#pointblank.duration_field): Create a duration column specification for use in a schema
- [profile_fields](reference/profile_fields.html#pointblank.profile_fields): Create a dict of string field specifications representing a person profile


## Prebuilt Actions

Prebuilt action functions for common notification patterns.

- [send_slack_notification](reference/send_slack_notification.html#pointblank.send_slack_notification): Create a Slack notification function using a webhook URL


# Resources

- [llms.txt](llms.txt) -- Indexed API reference for LLMs
- [llms-full.txt](llms-full.txt) -- Comprehensive documentation for LLMs


## Reuse


MIT
