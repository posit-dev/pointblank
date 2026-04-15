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


## Validation Steps


Validation steps are sequential validations on the target data. Call Validate's validation methods to build up a validation plan: a collection of steps that provides good validation coverage.


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

[Validate.above_threshold()](Validate.above_threshold.md#pointblank.Validate.above_threshold)  
Check if any validation steps exceed a specified threshold level.

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


Functions for getting to grips with a new data table. Use DataScan for a quick overview, `preview()` for first/last rows, `col_summary_tbl()` for column summaries, and `missing_vals_tbl()` for missing value analysis.


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


Generate synthetic test data based on schema definitions. Use `generate_dataset()` to create data from a Schema object.


[generate_dataset()](generate_dataset.md#pointblank.generate_dataset)  
Generate synthetic test data from a schema.

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
