# DataScan

Get a summary of a dataset.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L31-L1245)

``` python
DataScan(
    data,
    tbl_name=None,
)
```

Before writing validation rules for a table, it helps to know what's in it: which columns have missing values, how many distinct values each column has, and what ranges the numbers and dates fall in. A [DataScan](../reference/DataScan.md#pointblank.DataScan) collects these facts for every column in one pass. You can look at the result as a report, work with it as a table, or save it to compare against later.

Once a scan is made, it can be used in several ways:

- [`get_tabular_report()`](../reference/DataScan.get_tabular_report.md#pointblank.DataScan.get_tabular_report) shows the summary as a table with one row per column (this is also what [`col_summary_tbl()`](../reference/col_summary_tbl.md#pointblank.col_summary_tbl) returns)
- [`summary_data`](../reference/DataScan.summary_data.md#pointblank.DataScan.summary_data) gives the statistics as a DataFrame, for filtering, sorting, or building checks on them
- [`to_dict()`](../reference/DataScan.to_dict.md#pointblank.DataScan.to_dict), [`to_json()`](../reference/DataScan.to_json.md#pointblank.DataScan.to_json), and [`save_to_json()`](../reference/DataScan.save_to_json.md#pointblank.DataScan.save_to_json) store the profile, which [`from_dict()`](../reference/DataScan.from_dict.md#pointblank.DataScan.from_dict), [`from_json()`](../reference/DataScan.from_json.md#pointblank.DataScan.from_json), and [`load_from_json()`](../reference/DataScan.load_from_json.md#pointblank.DataScan.load_from_json) restore without the original data
- [`compare()`](../reference/DataScan.compare.md#pointblank.DataScan.compare) compares two scans and reports schema changes and drift in each column's statistics, as a [`DataScanDiff`](../reference/DataScanDiff.md#pointblank.DataScanDiff)

The scan stores the profile, not the data. To compare the rows of two tables rather than their profiles, use [`compare()`](../reference/compare.md#pointblank.compare).

> **Warning**
>
> The [DataScan()](../reference/DataScan.md#pointblank.DataScan) class is still experimental. Please report any issues you encounter in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).

## Parameters

`data: Any`  
The data to scan and summarize. This could be a DataFrame object, an Ibis table object, a CSV file path, a Parquet file path, a GitHub URL pointing to a CSV or Parquet file, or a database connection string.

`tbl_name: str | None = None`  
Optionally, the name of the table could be provided as `tbl_name`. It's shown in the header of the report, stored with the profile, and used to label the two scans in a [`DataScanDiff`](../reference/DataScanDiff.md#pointblank.DataScanDiff) report.

## Measures Of Missingness And Distinctness

For each column, the following measures are provided:

- `n_missing`: the number of missing values in the column
- `n_unique`: the number of distinct values in the column (if there are missing values, they count as one more distinct value)

The report also shows each count as a fraction of the number of rows in the table.

## Statistics For Numeric And String Columns

For numeric and string columns, the following statistics are provided. For string columns, they describe the lengths of the strings (in characters), not the strings themselves, which makes it easy to spot truncated values or codes of an unexpected length.

- `mean`: the mean
- `std`: the standard deviation
- `min`: the minimum
- `p05`: the 5th percentile
- `q_1`: the first quartile
- `median`: the median
- `q_3`: the third quartile
- `p95`: the 95th percentile
- `max`: the maximum
- `iqr`: the interquartile range (`q_3 - q_1`)

## Statistics For Date, Datetime, And Time Columns

For date, datetime, and time columns, the following statistics are provided:

- `min`: the earliest value in the column
- `max`: the latest value in the column

## Statistics For Boolean Columns

For Boolean columns, `freqs` gives the number of `True` and `False` values (as a dictionary, e.g., `{"True": 9, "False": 4}`). The report shows these as percentages.

## Supported Column Types

Numeric columns (integers, floats and decimals), string columns (including categorical and enum columns), date, datetime and time columns, and Boolean columns are summarized. Columns of other types (e.g., lists, arrays, structs, durations and binary data) can't be summarized, so they're listed in the report with no statistics, and in the JSON under `skipped_columns`. pandas stores decimals, times, and Booleans with missing values as `object` columns, which can't be summarized either.

## Returns

`DataScan`  
A DataScan object.

## Examples

Let's scan the `small_table` dataset, which has 13 rows and 8 columns of several types.

``` python
import pointblank as pb

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

scan = pb.DataScan(small_table, tbl_name="small_table")

scan.get_tabular_report()
```

[TABLE]

The report has one row per column. Column `c` has 2 missing values, `d` spans a wide range (from about 108 to 9,999.99), and the string column `b` always has 9 characters, which suggests a fixed-format code. Column `e` is Boolean, so it shows the share of `True` and `False` values instead of statistics.

The same statistics are available as a DataFrame through [summary_data](../reference/DataScan.summary_data.md#pointblank.DataScan.summary_data), so they can be filtered like any other table. Here are the columns with missing values:

``` python
import polars as pl

(
    scan.summary_data
    .filter(pl.col("n_missing") > 0)
    .select("colname", "coltype", "n_missing", "n_unique")
)
```

shape: (1, 4)

| colname | coltype | n_missing | n_unique |
|---------|---------|-----------|----------|
| str     | str     | i64       | i64      |
| "c"     | "Int64" | 2         | 7        |

A scan can also be kept as a baseline. If its profile is saved now with [`save_to_json()`](../reference/DataScan.save_to_json.md#pointblank.DataScan.save_to_json), a later scan of a new version of the table can be compared against it with [`compare()`](../reference/DataScan.compare.md#pointblank.DataScan.compare), which reports added and removed columns, type changes, and changes in each statistic.

## See Also

[col_summary_tbl()](../reference/col_summary_tbl.md#pointblank.col_summary_tbl)  
Get the same report directly from a table.

[missing_vals_tbl()](../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl)  
See where in a table the missing values are.

[preview()](../reference/preview.md#pointblank.preview)  
Show the first and last rows of a table.

[DataScanDiff](../reference/DataScanDiff.md#pointblank.DataScanDiff)  
The result of comparing two scans.
