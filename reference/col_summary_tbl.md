# col_summary_tbl()

Generate a column-level summary table of a dataset.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L2006-L2113)

``` python
col_summary_tbl(
    data,
    tbl_name=None,
)
```

The [col_summary_tbl()](../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function generates a summary table of a dataset, focusing on providing column-level information about the dataset. The summary includes the following information:

- the type of the table (e.g., `"polars"`, `"pandas"`, etc.)
- the number of rows and columns in the table
- column-level information, including:
  - the column name
  - the column type
  - measures of missingness and distinctness
  - descriptive stats and quantiles
  - statistics for datetime columns

The summary table is returned as a GT object, which can be displayed in a notebook or saved to an HTML file.

> **Warning**
>
> The [col_summary_tbl()](../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function is still experimental. Please report any issues you encounter in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).

## Parameters

`data: Any`  
The table to summarize, which could be a DataFrame object, an Ibis table object, a CSV file path, a Parquet file path, or a database connection string. Read the *Supported Input Table Types* section for details on the supported table types.

`tbl_name: str | None = None`  
Optionally, the name of the table could be provided as `tbl_name=`.

## Returns

`GT`  
A GT object that displays the column-level summaries of the table.

## Supported Column Types

Numeric columns (integers, floats and decimals), string columns (including categorical and enum columns), date, datetime and time columns, and Boolean columns are summarized. Columns of other types (e.g., lists, arrays, structs, durations and binary data) can't be summarized, so they get a row marked *unsupported type* with no statistics. pandas stores decimals, times, and Booleans with missing values as `object` columns, which can't be summarized either.

## Supported Input Table Types

The `data=` parameter can be given any of the following table types:

- Polars DataFrame (`"polars"`)
- Pandas DataFrame (`"pandas"`)
- DuckDB table (`"duckdb"`)\*
- MySQL table (`"mysql"`)\*
- PostgreSQL table (`"postgresql"`)\*
- SQLite table (`"sqlite"`)\*
- Parquet table (`"parquet"`)\*
- CSV files (string path or `pathlib.Path` object with `.csv` extension)
- Parquet files (string path, `pathlib.Path` object, glob pattern, directory with `.parquet` extension, or partitioned dataset)
- GitHub URLs (direct links to CSV or Parquet files on GitHub)
- Database connection strings (URI format with optional table specification)

The table types marked with an asterisk need to be prepared as Ibis tables (with type of `ibis.expr.types.relations.Table`). Furthermore, using [col_summary_tbl()](../reference/col_summary_tbl.md#pointblank.col_summary_tbl) with these types of tables requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a Polars or Pandas DataFrame, the availability of Ibis is not needed.

## Examples

It's easy to get a column-level summary of a table using the [col_summary_tbl()](../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function. Here's an example using the `small_table` dataset (itself loaded using the [`load_dataset()`](../reference/load_dataset.md#pointblank.load_dataset) function):

``` python
import pointblank as pb

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

pb.col_summary_tbl(data=small_table)
```

[TABLE]

This table used above was a Polars DataFrame, but the [col_summary_tbl()](../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function works with any table supported by `pointblank`, including Pandas DataFrames and Ibis backend tables. Here's an example using a DuckDB table handled by Ibis:

``` python
nycflights = pb.load_dataset(dataset="nycflights", tbl_type="duckdb")

pb.col_summary_tbl(data=nycflights, tbl_name="nycflights")
```

    /opt/hostedtoolcache/Python/3.11.17/x64/lib/python3.11/site-packages/great_tables/_tbl_data.py:902: UserWarning: PyArrow Table support is currently experimental.
      warnings.warn("PyArrow Table support is currently experimental.")

[TABLE]

The `tbl_name=` value appears in the report's title, which helps tell reports apart when several tables are summarized. An Ibis table is read into memory before its statistics are computed, so a very large table might be better summarized from a sample. To keep the summary rather than just display it (to query it, save it, or compare it with a later version of the table), create a [`DataScan`](../reference/DataScan.md#pointblank.DataScan) instead, which produces this same report through its [`get_tabular_report()`](../reference/DataScan.get_tabular_report.md#pointblank.DataScan.get_tabular_report) method.
