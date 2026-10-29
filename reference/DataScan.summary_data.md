# DataScan.summary_data

The column statistics as a DataFrame.

`DataScan.summary_data: Any`

The report from [`get_tabular_report()`](../reference/DataScan.get_tabular_report.md#pointblank.DataScan.get_tabular_report) is for reading, while [summary_data](../reference/DataScan.summary_data.md#pointblank.DataScan.summary_data) holds the same statistics as data. That's useful for finding columns that need attention (the ones with missing values, or with a single distinct value), for building validation steps from what was found, or for passing the profile to another tool.

The statistics are described in the [`DataScan`](../reference/DataScan.md#pointblank.DataScan) documentation. A statistic that doesn't apply to a column's type is null (for example, `mean` for a date column). The order of the statistic columns isn't fixed, so select columns by name.

## Examples

Here's the summary for the numeric columns of the `small_table` dataset.

``` python
import pointblank as pb
import polars as pl

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

summary = pb.DataScan(small_table).summary_data

(
    summary
    .filter(pl.col("coltype").is_in(["Int64", "Float64"]))
    .select("colname", "n_missing", "n_unique", "mean", "min", "max")
)
```

shape: (3, 6)

| colname | n_missing | n_unique | mean        | min      | max       |
|---------|-----------|----------|-------------|----------|-----------|
| str     | i64       | i64      | f64         | str      | str       |
| "a"     | 0         | 7        | 3.769231    | "1"      | "8"       |
| "c"     | 2         | 7        | 5.727273    | "2"      | "9"       |
| "d"     | 0         | 12       | 2304.702308 | "108.34" | "9999.99" |

Column `c` is the only one with missing values. The `min` and `max` columns are strings, because they also hold the dates from the date columns.

Since this is an ordinary DataFrame, the profile can lead to further checks. This finds the columns with one distinct value fewer than the table has rows:

``` python
summary.filter(pl.col("n_unique") == small_table.height - 1).get_column("colname").to_list()
```

    ['date_time', 'b', 'd']

Three columns that look like they should identify each row (a timestamp, a code, and an amount) each have exactly one repeated value. That's a sign of a duplicated row, which [`Validate.rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) would confirm.
