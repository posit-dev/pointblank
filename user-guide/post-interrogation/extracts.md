# Data Extracts

When validating data, identifying exactly which rows failed is critical for diagnosing and resolving data quality issues. This is where *data extracts* come in. Data extracts consist of target table rows containing at least one cell that failed validation. While the validation report provides an overview of pass/fail statistics, data extracts give you the actual problematic records for deeper investigation.

This article will cover:

- which validation methods collect data extracts
- multiple ways to access and work with data extracts
- practical examples of using extracts for data quality improvement
- advanced techniques for analyzing extract patterns

## The Validation Methods that Work with Data Extracts

The following validation methods operate on column values and will have rows extracted when there are failing test units in those rows:

- [`Validate.col_vals_gt()`](../../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)
- [`Validate.col_vals_lt()`](../../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt)
- [`Validate.col_vals_ge()`](../../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge)
- [`Validate.col_vals_le()`](../../reference/Validate.col_vals_le.md#pointblank.Validate.col_vals_le)
- [`Validate.col_vals_eq()`](../../reference/Validate.col_vals_eq.md#pointblank.Validate.col_vals_eq)
- [`Validate.col_vals_ne()`](../../reference/Validate.col_vals_ne.md#pointblank.Validate.col_vals_ne)
- [`Validate.col_vals_between()`](../../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between)
- [`Validate.col_vals_outside()`](../../reference/Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside)
- [`Validate.col_vals_in_set()`](../../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set)
- [`Validate.col_vals_not_in_set()`](../../reference/Validate.col_vals_not_in_set.md#pointblank.Validate.col_vals_not_in_set)
- [`Validate.col_vals_in_table()`](../../reference/Validate.col_vals_in_table.md#pointblank.Validate.col_vals_in_table)
- [`Validate.col_vals_null()`](../../reference/Validate.col_vals_null.md#pointblank.Validate.col_vals_null)
- [`Validate.col_vals_not_null()`](../../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null)
- [`Validate.col_vals_regex()`](../../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex)
- [`Validate.col_vals_expr()`](../../reference/Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr)
- [`Validate.conjointly()`](../../reference/Validate.conjointly.md#pointblank.Validate.conjointly)

These row-based validation methods will also have rows extracted should there be failing rows:

- [`Validate.rows_distinct()`](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct)
- [`Validate.rows_complete()`](../../reference/Validate.rows_complete.md#pointblank.Validate.rows_complete)

Note that some validation methods like [`Validate.col_exists()`](../../reference/Validate.col_exists.md#pointblank.Validate.col_exists) and [`Validate.col_schema_match()`](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) don't generate data extracts because they validate structural aspects of the table rather than checking column values.

## Accessing Data Extracts

There are three primary ways to access data extracts in Pointblank:

1.  the **CSV** buttons in validation reports
2.  through the [`Validate.get_data_extracts()`](../../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) method
3.  inspecting a subset of failed rows in step reports

Let's explore each approach using examples.

### CSV Data from Validation Reports

Data extracts are embedded within validation report tables. Let's look at an example, using the `small_table` dataset, where data extracts are collected in a single validation step due to failing test units:

``` python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_vals_lt( columns="d", value=3000)
    .interrogate()
)

validation
```

[TABLE]

The single validation step checks whether values in `d` are less than `3000`. Within that column, values range from `108.34` to `9999.99` so it makes sense that we can see 4 failing test units in the `FAIL` column.

If you look at the far right of the validation report you'll find there's a `CSV` button. Pressing it initiates the download of a CSV file, and that file contains the data extract for this validation step. The `CSV` button only appears when:

1.  there is a non-zero number of failing test units
2.  the validation step is based on the use of a column-value or a row-based validation method (the methods outlined in the section entitled *The Validation Methods that Work with Data Extracts*)

Access to CSV data for the test unit errors is useful when the validation report is shared with other data quality stakeholders, since it is easily accessible and doesn't require further use of Pointblank. The stakeholder can simply open the downloaded CSV in their preferred spreadsheet software, import it into a different analysis environment like R or Julia, or process it with any tool that supports CSV files. This cross-platform compatibility makes the CSV export particularly valuable in mixed-language data teams where different members might be working with different tools.

### [get_data_extracts()](../../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts)

For programmatic access to data extracts, Pointblank provides the [`Validate.get_data_extracts()`](../../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) method. This allows you to work with extract data directly in your Python workflow:

``` python
# Get data extracts from step 1
extract_1 = validation.get_data_extracts(i=1, frame=True)

extract_1
```

shape: (4, 9)

| \_row_num\_ | date_time | date | a | b | c | d | e | f |
|----|----|----|----|----|----|----|----|----|
| u32 | datetime\[μs\] | date | i64 | str | i64 | f64 | bool | str |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | "1-bcd-345" | 3 | 3423.29 | true | "high" |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | "5-egh-163" | 8 | 9999.99 | true | "low" |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | "5-jdo-903" | null | 3892.4 | false | "mid" |
| 6 | 2016-01-11 06:15:00 | 2016-01-11 | 4 | "2-dhe-923" | 4 | 3291.03 | true | "mid" |

The extracted table is of the same type (a Polars DataFrame) as the target table. Previously we used [load_dataset()](../../reference/load_dataset.md#pointblank.load_dataset) with the `tbl_type="polars"` option to fetch the dataset in that form.

Note these important details about using [`Validate.get_data_extracts()`](../../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts):

- the parameter `i=1` corresponds to the step number shown in the validation report (1-indexed, not 0-indexed)
- setting `frame=True` returns the data as a DataFrame rather than a dictionary (only works when `i` is a single integer)
- the extract includes all columns from the original data, not just the column being validated
- an additional `_row_num_` column is added to identify the original row positions

### Step Reports

Step reports provide another way to access and visualize failing data. When you generate a step report for a validation step that has failing rows, those failing rows are displayed directly in the report:

``` python
# Get a step report for the first validation step
step_report = validation.get_step_report(i=1)

step_report
```

| Report for Validation Step 1 ASSERTION `d < 3000` **4** / **13** TEST UNIT FAILURES IN COLUMN **6** EXTRACT OF ALL **4** ROWS (WITH TEST UNIT FAILURES IN RED): |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 6 | 2016-01-11 06:15:00 | 2016-01-11 | 4 | 2-dhe-923 | 4 | 3291.03 | True | mid |

Step reports offer several advantages for working with data extracts as they:

1.  provide immediate visual context by highlighting the specific column being validated
2.  format the data for better readability, especially useful when sharing results with colleagues
3.  include additional metadata about the validation step and failure statistics

For steps with many failures, you can customize how many rows to display:

``` python
# Limit to just 2 rows of failing data
limited_report = validation.get_step_report(i=1, limit=2)

limited_report
```

| Report for Validation Step 1 ASSERTION `d < 3000` **4** / **13** TEST UNIT FAILURES IN COLUMN **6** EXTRACT OF FIRST **2** ROWS (WITH TEST UNIT FAILURES IN RED): |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |

Step reports are particularly valuable when you want to quickly inspect the failing data without extracting it into a separate DataFrame. They provide a bridge between the high-level validation report and the detailed data extracts.

## Viewing Data Extracts with [preview()](../../reference/preview.md#pointblank.preview)

To get a consistent HTML representation of any data extract (regardless of the table type), we can use the [preview()](../../reference/preview.md#pointblank.preview) function:

``` python
pb.preview(data=extract_1)
```

| PolarsRows4Columns9 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 6 | 2016-01-11 06:15:00 | 2016-01-11 | 4 | 2-dhe-923 | 4 | 3291.03 | True | mid |

The view is optimized for readability, with column names and data types displayed in a compact format. Notice that the `_row_num_` column is now part of the table stub and doesn't steal focus from the table's original columns.

The [preview()](../../reference/preview.md#pointblank.preview) function is designed to provide the head and tail (5 rows each) of the table so very large extracts won't overflow the display.

## Working with Multiple Validation Steps

When validating data with multiple steps, you can extract failing rows from any step or combine extracts from multiple steps:

``` python
# Create a validation with multiple steps
multi_validation = (
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_vals_gt(columns="a", value=3)                                  # Step 1
    .col_vals_lt(columns="d", value=3000)                               # Step 2
    .col_vals_regex(columns="b", pattern="^[0-9]-[a-z]{3}-[0-9]{3}$")   # Step 3
    .interrogate()
)

multi_validation
```

[TABLE]

### Extracting Data from a Specific Step

You can access extracts from any specific validation step:

``` python
# Get extracts from step 2 (`d < 3000` validation)
less_than_failures = multi_validation.get_data_extracts(i=2, frame=True)

less_than_failures
```

shape: (4, 9)

| \_row_num\_ | date_time | date | a | b | c | d | e | f |
|----|----|----|----|----|----|----|----|----|
| u32 | datetime\[μs\] | date | i64 | str | i64 | f64 | bool | str |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | "1-bcd-345" | 3 | 3423.29 | true | "high" |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | "5-egh-163" | 8 | 9999.99 | true | "low" |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | "5-jdo-903" | null | 3892.4 | false | "mid" |
| 6 | 2016-01-11 06:15:00 | 2016-01-11 | 4 | "2-dhe-923" | 4 | 3291.03 | true | "mid" |

Using `frame=True` means that returned value will be a DataFrame (not a dictionary that contains a single DataFrame).

If a step has no failing rows, an empty DataFrame will be returned:

``` python
# Get extracts from step 3 (regex check)
regex_failures = multi_validation.get_data_extracts(i=3, frame=True)

regex_failures
```

shape: (0, 9)

| \_row_num\_ | date_time      | date | a   | b   | c   | d   | e    | f   |
|-------------|----------------|------|-----|-----|-----|-----|------|-----|
| u32         | datetime\[μs\] | date | i64 | str | i64 | f64 | bool | str |

### Getting All Extracts at Once

To retrieve extracts from all steps with failures in one command:

``` python
# Get all extracts ()
all_extracts = multi_validation.get_data_extracts()

# Display the step numbers that have extracts
print(f"Steps with data extracts: {list(all_extracts.keys())}")
```

    Steps with data extracts: [1, 2, 3]

A dictionary of DataFrames is returned and only steps with failures will appear in this dictionary.

### Getting Specific Extracts

You can also retrieve data extracts from several specified steps as a dictionary:

``` python
# Get extracts from steps 1 and 2 as a dictionary
extract_dict = multi_validation.get_data_extracts(i=[1, 2])

# The keys are the step numbers
print(f"Dictionary keys: {list(extract_dict.keys())}")

# Get the number of failing rows in each extract
for step, extract in extract_dict.items():
    print(f"Step {step}: {len(extract)} failing rows")
```

    Dictionary keys: [1, 2]
    Step 1: 7 failing rows
    Step 2: 4 failing rows

Note that `frame=True` cannot be used when retrieving multiple extracts.

## Applications of Data Extracts

Once you have extracted the failing data, there are numerous ways to analyze and use this information to improve data quality. Let's explore some practical applications.

### Finding Patterns Across Validation Steps

You can analyze patterns across different validation steps by combining extracts:

``` python
# Get a consolidated view of all rows that failed any validation
all_failure_rows = set()
for step, extract in all_extracts.items():
    if len(extract) > 0:
        all_failure_rows.update(extract["_row_num_"])

print(f"Total unique rows with failures: {len(all_failure_rows)}")
print(f"Row numbers with failures: {sorted(all_failure_rows)}")
```

    Total unique rows with failures: 8
    Row numbers with failures: [1, 2, 4, 6, 9, 10, 12, 13]

### Identifying Rows with Multiple Failures

You might want to find rows that failed multiple validation checks, as these often represent more serious data quality issues:

``` python
# Get row numbers from each extract
step1_rows = set(multi_validation.get_data_extracts(i=1, frame=True)["_row_num_"])
step2_rows = set(multi_validation.get_data_extracts(i=2, frame=True)["_row_num_"])

# Find rows that failed both validations
common_failures = step1_rows.intersection(step2_rows)
print(f"Rows failing both step 1 and step 2: {common_failures}")
```

    Rows failing both step 1 and step 2: {1, 2, 4}

### Statistical Analysis of Failing Values

Once you have data extracts, you can perform statistical analysis to identify patterns in the failing data:

``` python
# Get extracts from step 2
d_value_failures = multi_validation.get_data_extracts(i=2, frame=True)

# Basic statistical analysis of the failing values
if len(d_value_failures) > 0:
    print(f"Min failing value: {d_value_failures['d'].min()}")
    print(f"Max failing value: {d_value_failures['d'].max()}")
    print(f"Mean failing value: {d_value_failures['d'].mean()}")
```

    Min failing value: 3291.03
    Max failing value: 9999.99
    Mean failing value: 5151.6775

These analysis techniques help you thoroughly investigate data quality issues by examining failing data from multiple perspectives. Rather than treating failures as isolated incidents, you can identify patterns that might indicate systematic problems in your data pipeline.

### Detailed Analysis with [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl)

For a more comprehensive view of the statistical properties of your extract data, you can use the [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function:

``` python
# Get extracts from step 2
d_value_failures = multi_validation.get_data_extracts(i=2, frame=True)

# Generate a comprehensive statistical summary of the failing data
pb.col_summary_tbl(d_value_failures)
```

| PolarsRows4Columns9 |  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|  | Column | NA | UQ | Mean | SD | Min | P5 | Q1 | Med | Q3 | P95 | Max | IQR |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>numeric\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="numeric" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"\>\</polygon\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>\_row_num\_\</div\>\<div style='font-size: 11px; color: gray;'\>UInt32\</div\> | 0\<br\>0 | 4\<br\>1 | 3.25 | 2.22 | 1 | 1.01 | 1.75 | 3 | 4.5 | 5.7 | 6 | 2.75 |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>date\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="date" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#007D00" stroke-width="4" fill="#2CCF00" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="D" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<path d="M0,0 L34.716,0 C40.788,0 46.332,0.968 51.348,2.904 C56.364,4.84 60.632,7.722 64.152,11.55 C67.672,15.378 70.4,20.174 72.336,25.938 C74.272,31.702 75.24,38.412 75.24,46.068 C75.24,53.724 74.272,60.434 72.336,66.198 C70.4,71.962 67.672,76.758 64.152,80.586 C60.632,84.414 56.364,87.296 51.348,89.232 C46.332,91.168 40.788,92.136 34.716,92.136 L0,92.136 L0,0 Z M34.716,74.316 C40.7,74.316 45.408,72.644 48.84,69.3 C52.272,65.956 53.988,60.588 53.988,53.196 L53.988,38.94 C53.988,31.548 52.272,26.18 48.84,22.836 C45.408,19.492 40.7,17.82 34.716,17.82 L20.064,17.82 L20.064,74.316 L34.716,74.316 Z"\>\</path\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>date_time\</div\>\<div style='font-size: 11px; color: gray;'\>Datetime(time_unit='us', time_zone=None)\</div\> | 0\<br\>0 | 4\<br\>1 | \- | \- | 2016\<br\>01\<br\>04 00:32:00 | \- | \- | \- | \- | \- | 2016\<br\>01\<br\>11 06:15:00 | \- |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>date\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="date" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#007D00" stroke-width="4" fill="#2CCF00" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="D" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<path d="M0,0 L34.716,0 C40.788,0 46.332,0.968 51.348,2.904 C56.364,4.84 60.632,7.722 64.152,11.55 C67.672,15.378 70.4,20.174 72.336,25.938 C74.272,31.702 75.24,38.412 75.24,46.068 C75.24,53.724 74.272,60.434 72.336,66.198 C70.4,71.962 67.672,76.758 64.152,80.586 C60.632,84.414 56.364,87.296 51.348,89.232 C46.332,91.168 40.788,92.136 34.716,92.136 L0,92.136 L0,0 Z M34.716,74.316 C40.7,74.316 45.408,72.644 48.84,69.3 C52.272,65.956 53.988,60.588 53.988,53.196 L53.988,38.94 C53.988,31.548 52.272,26.18 48.84,22.836 C45.408,19.492 40.7,17.82 34.716,17.82 L20.064,17.82 L20.064,74.316 L34.716,74.316 Z"\>\</path\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>date\</div\>\<div style='font-size: 11px; color: gray;'\>Date\</div\> | 0\<br\>0 | 3\<br\>0.75 | \- | \- | 2016\<br\>01\<br\>04 | \- | \- | \- | \- | \- | 2016\<br\>01\<br\>11 | \- |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>numeric\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="numeric" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"\>\</polygon\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>a\</div\>\<div style='font-size: 11px; color: gray;'\>Int64\</div\> | 0\<br\>0 | 3\<br\>0.75 | 2.75 | 0.96 | 2 | 2 | 2 | 2.5 | 3.25 | 3.85 | 4 | 1.25 |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>string\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="string" transform="translate(4.000000, 4.500000)"\> \<rect id="square" stroke="#9A8700" stroke-width="4" fill="#CFB600" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="S" transform="translate(49.516000, 38.280000)" fill="#FFFFFF" fill-rule="nonzero"\> \<path d="M34.98,95.304 C26.884,95.304 20.02,93.918 14.388,91.146 C8.756,88.374 3.96,84.788 0,80.388 L13.2,67.056 C19.448,74.096 27.148,77.616 36.3,77.616 C41.228,77.616 44.88,76.604 47.256,74.58 C49.632,72.556 50.82,69.872 50.82,66.528 C50.82,63.976 50.116,61.842 48.708,60.126 C47.3,58.41 44.44,57.244 40.128,56.628 L31.02,55.44 C21.252,54.208 14.102,51.26 9.57,46.596 C5.038,41.932 2.772,35.728 2.772,27.984 C2.772,23.848 3.564,20.064 5.148,16.632 C6.732,13.2 8.998,10.252 11.946,7.788 C14.894,5.324 18.502,3.41 22.77,2.046 C27.038,0.682 31.9,0 37.356,0 C44.308,0 50.424,1.1 55.704,3.3 C60.984,5.5 65.516,8.712 69.3,12.936 L55.968,26.4 C53.768,23.848 51.106,21.758 47.982,20.13 C44.858,18.502 40.92,17.688 36.168,17.688 C31.68,17.688 28.336,18.48 26.136,20.064 C23.936,21.648 22.836,23.848 22.836,26.664 C22.836,29.832 23.694,32.12 25.41,33.528 C27.126,34.936 29.92,35.948 33.792,36.564 L42.9,38.016 C52.404,39.512 59.444,42.438 64.02,46.794 C68.596,51.15 70.884,57.332 70.884,65.34 C70.884,69.74 70.092,73.788 68.508,77.484 C66.924,81.18 64.614,84.348 61.578,86.988 C58.542,89.628 54.78,91.674 50.292,93.126 C45.804,94.578 40.7,95.304 34.98,95.304 Z"\>\</path\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>b\</div\>\<div style='font-size: 11px; color: gray;'\>String\</div\> | 0\<br\>0 | 4\<br\>1 | 9 | 0 | 9 | 9 | 9 | 9 | 9 | 9 | 9 | 0 |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>numeric\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="numeric" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"\>\</polygon\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>c\</div\>\<div style='font-size: 11px; color: gray;'\>Int64\</div\> | 1\<br\>0.25 | 4\<br\>1 | 5 | 2.65 | 3 | 3.01 | 3.5 | 4 | 6 | 7.6 | 8 | 2.5 |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>numeric\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="numeric" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"\>\</polygon\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>d\</div\>\<div style='font-size: 11px; color: gray;'\>Float64\</div\> | 0\<br\>0 | 4\<br\>1 | 5,151.68 | 3,242.49 | 3291.03 | 3,293.01 | 3,390.22 | 3,657.85 | 5,419.3 | 9,083.85 | 9999.99 | 2,029.07 |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>boolean\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="boolean" transform="translate(4.000000, 4.339342)"\> \<rect id="square" stroke="#9C3E00" stroke-width="4" fill="#CF6100" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="T" transform="translate(32.322000, 29.972000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="29.326 11.61 29.326 60.028 16.254 60.028 16.254 11.61 0 11.61 0 0 45.58 0 45.58 11.61"\>\</polygon\> \</g\> \<g id="F" transform="translate(99.122000, 81.972000)" fill="#FFFFFF" fill-rule="nonzero"\> \<polygon points="0 60.028 0 0 39.818 0 39.818 11.61 13.072 11.61 13.072 23.908 35.862 23.908 35.862 35.432 13.072 35.432 13.072 60.028"\>\</polygon\> \</g\> \<rect id="line" stroke="#FFFFFF" fill="#FFFFFF" transform="translate(82.022437, 84.377940) rotate(-315.000000) translate(-82.022437, -84.377940) " x="78.0224369" y="25.37794" width="8" height="118"\>\</rect\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>e\</div\>\<div style='font-size: 11px; color: gray;'\>Boolean\</div\> | 0\<br\>0 | \<span style='font-weight: bold;'\>T\</span\>0.75\<br\>\<span style='font-weight: bold;'\>F\</span\>0.25 | \- | \- | \- | \- | \- | \- | \- | \- | \- | \- |
| \<svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: \#FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"\> \<title\>string\</title\> \<g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"\> \<g id="string" transform="translate(4.000000, 4.500000)"\> \<rect id="square" stroke="#9A8700" stroke-width="4" fill="#CFB600" x="2" y="2" width="168" height="168" rx="8"\>\</rect\> \<g id="S" transform="translate(49.516000, 38.280000)" fill="#FFFFFF" fill-rule="nonzero"\> \<path d="M34.98,95.304 C26.884,95.304 20.02,93.918 14.388,91.146 C8.756,88.374 3.96,84.788 0,80.388 L13.2,67.056 C19.448,74.096 27.148,77.616 36.3,77.616 C41.228,77.616 44.88,76.604 47.256,74.58 C49.632,72.556 50.82,69.872 50.82,66.528 C50.82,63.976 50.116,61.842 48.708,60.126 C47.3,58.41 44.44,57.244 40.128,56.628 L31.02,55.44 C21.252,54.208 14.102,51.26 9.57,46.596 C5.038,41.932 2.772,35.728 2.772,27.984 C2.772,23.848 3.564,20.064 5.148,16.632 C6.732,13.2 8.998,10.252 11.946,7.788 C14.894,5.324 18.502,3.41 22.77,2.046 C27.038,0.682 31.9,0 37.356,0 C44.308,0 50.424,1.1 55.704,3.3 C60.984,5.5 65.516,8.712 69.3,12.936 L55.968,26.4 C53.768,23.848 51.106,21.758 47.982,20.13 C44.858,18.502 40.92,17.688 36.168,17.688 C31.68,17.688 28.336,18.48 26.136,20.064 C23.936,21.648 22.836,23.848 22.836,26.664 C22.836,29.832 23.694,32.12 25.41,33.528 C27.126,34.936 29.92,35.948 33.792,36.564 L42.9,38.016 C52.404,39.512 59.444,42.438 64.02,46.794 C68.596,51.15 70.884,57.332 70.884,65.34 C70.884,69.74 70.092,73.788 68.508,77.484 C66.924,81.18 64.614,84.348 61.578,86.988 C58.542,89.628 54.78,91.674 50.292,93.126 C45.804,94.578 40.7,95.304 34.98,95.304 Z"\>\</path\> \</g\> \</g\> \</g\> \</svg\> | \<div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'\>f\</div\>\<div style='font-size: 11px; color: gray;'\>String\</div\> | 0\<br\>0 | 3\<br\>0.75 | 3.25 | 0.5 | 3 | 3 | 3 | 3 | 3.25 | 3.85 | 4 | 0.25 |
| String columns statistics regard the string's length. |  |  |  |  |  |  |  |  |  |  |  |  |  |

This statistical overview provides:

1.  a count of values (including missing values)
2.  type information for each column
3.  distribution metrics like min, max, mean, and quartiles for numeric columns
4.  frequency of common values for categorical columns
5.  missing value counts and proportions

Using [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) on data extracts lets you quickly understand the characteristics of failing data without writing custom analysis code. This approach is particularly valuable when:

- You need to understand the statistical properties of failing records
- You want to compare distributions of failing vs passing data
- You're looking for anomalies or unexpected patterns within the failing rows

For example, if values failing a validation check are concentrated at certain quantiles or have an unusual distribution shape, this might indicate a systematic data collection or processing issue rather than random errors.

## Using Extracts for Data Quality Improvement

Data extracts are especially valuable for:

1.  **Root Cause Analysis**: examining the full context of failing rows to understand why they failed
2.  **Data Cleaning**: creating targeted cleanup scripts that focus only on problematic records
3.  **Feedback Loops**: sharing specific examples with data providers to improve upstream quality
4.  **Pattern Recognition**: identifying systemic issues by analyzing groups of failing records

Here's an example of using extracts to create a corrective action plan:

``` python
import polars as pl

# Create a new sample of an extract DF
sample_extract = pl.DataFrame({
    "id": range(1, 11),
    "value": [3500, 4200, 3800, 9800, 5500, 7200, 8300, 4100, 7600, 3200],
    "category": ["A", "B", "A", "C", "B", "A", "C", "B", "A", "B"],
    "region": [
        "South", "South", "North", "East", "South",
        "South", "East", "South", "West", "South"
    ]
})

# Identify which regions have the most failures
region_counts = (
    sample_extract
    .group_by("region")
    .agg(pl.len().alias("failure_count"))
    .sort("failure_count", descending=True)
)

region_counts
```

shape: (4, 2)

| region  | failure_count |
|---------|---------------|
| str     | u32           |
| "South" | 6             |
| "East"  | 2             |
| "West"  | 1             |
| "North" | 1             |

Analysis shows that 6 out of 10 failing records (60%) are from the `"South"` region, making it the highest priority area for data quality investigation. This suggests a potential systemic issue with data collection or processing in that specific region.

## Best Practices for Working with Data Extracts

When incorporating data extracts into your data quality workflow:

1.  Use extracts for investigation, not just reporting: the real value is in the insights you gain from analyzing the problematic data

2.  Combine with other Pointblank features: data extracts work well with step reports and can inform threshold settings for future validations

3.  Consider sampling for very large datasets: if your extracts contain thousands of rows, focus your investigation on a representative sample

4.  Look beyond individual validation steps: cross-reference extracts from different steps to identify complex issues that span multiple validation rules

5.  Document patterns in failing data: record and share insights about common failure modes to build organizational knowledge about data quality issues.

By integrating these practices into your data validation workflow, you'll transform data extracts from simple error lists into powerful diagnostic tools. The most successful data quality initiatives treat extracts as the starting point for investigation rather than the end result of validation. When systematically analyzed and documented, patterns in failing data can reveal underlying issues in data systems, collection methods, or business processes that might otherwise remain hidden. Remember that the ultimate goal isn't just to identify problematic records, but to use that information to implement targeted improvements that prevent similar issues from occurring in the future.

## Conclusion

Data extracts bridge the gap between high-level validation statistics and the detailed context needed to fix data quality issues. By providing access to the actual failing records, Pointblank enables you to:

- pinpoint exactly which data points caused validation failures
- understand the full context around problematic values
- develop targeted strategies for data cleanup and quality improvement
- communicate specific examples to stakeholders

Whether you're accessing extracts through CSV downloads, the [`Validate.get_data_extracts()`](../../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) method, or step reports, this feature provides the detail needed to move from identifying problems to implementing solutions.
