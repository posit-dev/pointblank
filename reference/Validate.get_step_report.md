# Validate.get_step_report()

Get a detailed report for a single validation step.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L21888-L22280)

``` python
Validate.get_step_report(
    i,
    columns_subset=None,
    header=":default:",
    limit=10,
)
```

The [get_step_report()](../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report) method returns a report of what went well--or what failed spectacularly--for a given validation step. The report includes a summary of the validation step and a detailed breakdown of the interrogation results. The report is presented as a GT table object, which can be displayed in a notebook or exported to an HTML file.

> **Warning**
>
> The [get_step_report()](../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report) method is still experimental. Please report any issues you encounter in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).

## Parameters

`i: int`  
The step number for which to get the report.

`columns_subset: str | list[str] | Column | None = None`  
The columns to display in a step report that shows errors in the input table. By default all columns are shown (`None`). If a subset of columns is desired, we can provide a list of column names, a string with a single column name, a `Column` object, or a `ColumnSelector` object. The last two options allow for more flexible column selection using column selector functions. Errors are raised if the column names provided don't match any columns in the table (when provided as a string or list of strings) or if column selector expressions don't resolve to any columns.

`header: str = ":default:"`  
Options for customizing the header of the step report. The default is the `":default:"` value which produces a header with a standard title and set of details underneath. Aside from this default, free text can be provided for the header. This will be interpreted as Markdown text and transformed internally to HTML. You can provide one of two templating elements: `{title}` and `{details}`. The default header has the template `"{title}{details}"` so you can easily start from that and modify as you see fit. If you don't want a header at all, you can set `header=None` to remove it entirely.

`limit: int | None = 10`  
The number of rows to display for those validation steps that check values in rows (the `col_vals_*()` validation steps). The default is `10` rows and the limit can be removed entirely by setting `limit=None`.

## Returns

`GT`  
A GT table object that represents the detailed report for the validation step.

## Types Of Step Reports

The [get_step_report()](../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report) method produces a report based on the *type* of validation step. The following column-value or row-based validation step validation methods will produce a report that shows the rows of the data that failed:

- [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)
- [`col_vals_ge()`](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge)
- [`col_vals_lt()`](../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt)
- [`col_vals_le()`](../reference/Validate.col_vals_le.md#pointblank.Validate.col_vals_le)
- [`col_vals_eq()`](../reference/Validate.col_vals_eq.md#pointblank.Validate.col_vals_eq)
- [`col_vals_ne()`](../reference/Validate.col_vals_ne.md#pointblank.Validate.col_vals_ne)
- [`col_vals_between()`](../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between)
- [`col_vals_outside()`](../reference/Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside)
- [`col_vals_in_set()`](../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set)
- [`col_vals_not_in_set()`](../reference/Validate.col_vals_not_in_set.md#pointblank.Validate.col_vals_not_in_set)
- [`col_vals_increasing()`](../reference/Validate.col_vals_increasing.md#pointblank.Validate.col_vals_increasing)
- [`col_vals_decreasing()`](../reference/Validate.col_vals_decreasing.md#pointblank.Validate.col_vals_decreasing)
- [`col_vals_null()`](../reference/Validate.col_vals_null.md#pointblank.Validate.col_vals_null)
- [`col_vals_not_null()`](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null)
- [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex)
- [`col_vals_within_spec()`](../reference/Validate.col_vals_within_spec.md#pointblank.Validate.col_vals_within_spec)
- [`col_vals_str_len()`](../reference/Validate.col_vals_str_len.md#pointblank.Validate.col_vals_str_len)
- [`col_vals_in_table()`](../reference/Validate.col_vals_in_table.md#pointblank.Validate.col_vals_in_table)
- [`col_vals_expr()`](../reference/Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr)
- [`col_missing_coded()`](../reference/Validate.col_missing_coded.md#pointblank.Validate.col_missing_coded)
- [`col_missing_only_coded()`](../reference/Validate.col_missing_only_coded.md#pointblank.Validate.col_missing_only_coded)
- [`col_missing_consistent()`](../reference/Validate.col_missing_consistent.md#pointblank.Validate.col_missing_consistent)
- [`conjointly()`](../reference/Validate.conjointly.md#pointblank.Validate.conjointly)
- [`prompt()`](../reference/Validate.prompt.md#pointblank.Validate.prompt)
- [`rows_complete()`](../reference/Validate.rows_complete.md#pointblank.Validate.rows_complete)

The [`rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) validation step will produce a report that shows duplicate rows (or duplicate values in one or a set of columns as defined in that method's `columns_subset=` parameter.

The [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) validation step will produce a report that shows the schema of the data table and the schema of the validation step. The report will indicate whether the schemas match or not.

The [`tbl_match()`](../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match) and [`tbl_reconcile()`](../reference/Validate.tbl_reconcile.md#pointblank.Validate.tbl_reconcile) validation steps will produce a report comparing the target table to the comparison table, with samples of the rows (or key groups) that differ.

The remaining validation steps check the table, or a column, as a whole. Each produces a summary report with the measured value(s) beside the expected value(s):

- the aggregate methods (e.g., [`col_sum_gt()`](../reference/Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt), [`col_avg_eq()`](../reference/Validate.col_avg_eq.md#pointblank.Validate.col_avg_eq), [`col_sd_le()`](../reference/Validate.col_sd_le.md#pointblank.Validate.col_sd_le)): the computed aggregate value
- [`col_exists()`](../reference/Validate.col_exists.md#pointblank.Validate.col_exists): the column's position in the table (and, if it's missing, the closest-matching column name)
- [`row_count_match()`](../reference/Validate.row_count_match.md#pointblank.Validate.row_count_match) and [`col_count_match()`](../reference/Validate.col_count_match.md#pointblank.Validate.col_count_match): the row or column count
- [`col_pct_null()`](../reference/Validate.col_pct_null.md#pointblank.Validate.col_pct_null) and [`col_pct_missing()`](../reference/Validate.col_pct_missing.md#pointblank.Validate.col_pct_missing): the number and percentage of null or missing values
- [`data_freshness()`](../reference/Validate.data_freshness.md#pointblank.Validate.data_freshness): the most recent datetime value and its age
- [`specially()`](../reference/Validate.specially.md#pointblank.Validate.specially): the number of passing and failing test units

## Examples

Let's create a validation plan with a few validation steps and interrogate the data. With that, we'll have a look at the validation reporting table for the entire collection of steps and what went well or what failed.

``` python
import pointblank as pb

validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="small_table", tbl_type="pandas"),
        tbl_name="small_table",
        label="Example for the get_step_report() method",
        thresholds=(1, 0.20, 0.40)
    )
    .col_vals_lt(columns="d", value=3500)
    .col_vals_between(columns="c", left=1, right=8)
    .col_vals_gt(columns="a", value=3)
    .col_vals_regex(columns="b", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}")
    .interrogate()
)

validation
```

[TABLE]

There were four validation steps performed, where the first three steps had failing test units and the last step had no failures. Let's get a detailed report for the first step by using the [get_step_report()](../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report) method.

``` python
validation.get_step_report(i=1)
```

| Report for Validation Step 1 ✗ ASSERTION `d < 3500` **2** / **13** TEST UNIT FAILURES IN COLUMN **6** EXTRACT OF ALL **2** ROWS (WITH TEST UNIT FAILURES IN RED): |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *datetime64\[ns\]* | date *datetime64\[ns\]* | a *int64* | b *object* | c *float64* | d *float64* | e *bool* | f *object* |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 00:00:00 | 3 | 5-egh-163 | 8.0 | 9999.99 | True | low |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 00:00:00 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |

The report for the first step is displayed. The report includes a summary of the validation step and a detailed breakdown of the interrogation results. The report provides details on what the validation step was checking, the extent to which the test units failed, and a table that shows the failing rows of the data with the column of interest highlighted.

The second and third steps also had failing test units. Reports for those steps can be viewed by using `get_step_report(i=2)` and `get_step_report(i=3)` respectively.

The final step did not have any failing test units. A report for the final step can still be viewed by using `get_step_report(i=4)`. The report will indicate that every test unit passed and a prview of the target table will be provided.

``` python
validation.get_step_report(i=4)
```

| Report for Validation Step 4 ✓ ASSERTION `b matches regex [0-9]-[a-z]{3}-[0-9]{3}` **13** TEST UNITS *ALL PASSED* IN COLUMN **4** PREVIEW OF TARGET TABLE: |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *datetime64\[ns\]* | date *datetime64\[ns\]* | a *int64* | b *object* | c *float64* | d *float64* | e *bool* | f *object* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 00:00:00 | 2 | 1-bcd-345 | 3.0 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 00:00:00 | 3 | 5-egh-163 | 8.0 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 00:00:00 | 6 | 8-kdg-938 | 3.0 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 00:00:00 | 2 | 5-jdo-903 | NA | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 00:00:00 | 8 | 3-ldm-038 | 7.0 | 283.94 | True | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 00:00:00 | 3 | 5-bce-642 | 9.0 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 00:00:00 | 3 | 5-bce-642 | 9.0 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 00:00:00 | 4 | 2-dmx-010 | 7.0 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 00:00:00 | 2 | 7-dmx-010 | 8.0 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 00:00:00 | 1 | 3-dka-303 | NA | 2230.09 | True | high |

If you'd like to trim down the number of columns shown in the report, you can provide a subset of columns to display. For example, if you only want to see the columns `a`, `b`, and `c`, you can provide those column names as a list.

``` python
validation.get_step_report(i=1, columns_subset=["a", "b", "c"])
```

| Report for Validation Step 1 ✗ ASSERTION `d < 3500` **2** / **13** TEST UNIT FAILURES IN COLUMN **6** EXTRACT OF ALL **2** ROWS: |  |  |  |
|----|----|----|----|
|  | a *int64* | b *object* | c *float64* |
| 2 | 3 | 5-egh-163 | 8.0 |
| 4 | 2 | 5-jdo-903 | None |

If you'd like to increase or reduce the maximum number of rows shown in the report, you can provide a different value for the `limit` parameter. For example, if you'd like to see only up to 5 rows, you can set `limit=5`.

``` python
validation.get_step_report(i=3, limit=5)
```

| Report for Validation Step 3 ✗ ASSERTION `a > 3` **7** / **13** TEST UNIT FAILURES IN COLUMN **3** EXTRACT OF FIRST **5** ROWS (WITH TEST UNIT FAILURES IN RED): |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *datetime64\[ns\]* | date *datetime64\[ns\]* | a *int64* | b *object* | c *float64* | d *float64* | e *bool* | f *object* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 00:00:00 | 2 | 1-bcd-345 | 3.0 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 00:00:00 | 3 | 5-egh-163 | 8.0 | 9999.99 | True | low |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 00:00:00 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 00:00:00 | 3 | 5-bce-642 | 9.0 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 00:00:00 | 3 | 5-bce-642 | 9.0 | 837.93 | False | high |

Step 3 actually had 7 failing test units, but only the first 5 rows are shown in the step report because of the `limit=5` parameter.
