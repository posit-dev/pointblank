# missing_vals_tbl()

Display a table that shows the missing values in the input table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L3058-L3608)

``` python
missing_vals_tbl(
    data,
    missing=None,
    as_heatmap=False,
)
```

The [missing_vals_tbl()](../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) function generates a table that shows the missing values in the input table. The table is displayed using the Great Tables API, which allows for further customization of the table's appearance if so desired.

By default, missingness is treated as binary (a value is either Null or it isn't) and the function renders a sector-based heatmap of the proportion of Null values across the rows of each column. When a `missing=` mapping of columns to [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) objects is supplied, the function instead renders a *structured missingness* breakdown: one row per column with the count and percentage of complete values and of each missing *reason* (e.g., `refused`, `not_asked`). Declared (coded) reasons are grouped under a "Missing Reasons" spanner and keep their raw input form as labels; actual `Null`/`None`/`NA` values (which are not part of the spec) are tallied in a fixed "Null" column at the far right (styled like "Complete"), so they aren't mistaken for declared reasons.

Note that supplying `missing=` produces a *different report* than the default view: it is a distinct visualization (a per-reason breakdown table, or a per-reason heatmap with `as_heatmap=True`), not an annotated version of the default sector heatmap. The report titles differ accordingly ("Missing Values" for the default, "Missing Values by Reason" or "Missing Pattern Heatmap" for the structured views), and the shared header/title styling makes the family resemblance clear.

## Parameters

`data: Any`  
The table for which to display the missing values. This could be a DataFrame object, an Ibis table object, a CSV file path, a Parquet file path, or a database connection string. Read the *Supported Input Table Types* section for details on the supported table types.

`missing: dict[str, MissingSpec] | None = None`  
An optional dictionary mapping column names to [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) objects. When provided, the function renders a structured breakdown of missingness by reason for the specified columns (rather than the default sector heatmap). The reason columns are the union of reasons across the supplied specs; a reason that isn't defined for a given column is shown as an em dash (not applicable), as distinct from a defined-but-unobserved reason (shown as `0 (0%)`).

`as_heatmap: bool = False`  
Only applies when `missing=` is provided. When `True`, render the per-reason proportions as a color-coded heatmap (cells shaded from light to dark by the proportion missing) instead of the count/percentage text breakdown. Default is `False`.

## Returns

`GT`  
A GT object that displays the table of missing values in the input table.

## Supported Input Table Types

The `data=` parameter can be given any of the following table types:

- Polars DataFrame (`"polars"`)
- Pandas DataFrame (`"pandas"`)
- PySpark table (`"pyspark"`)
- DuckDB table (`"duckdb"`)\*
- MySQL table (`"mysql"`)\*
- PostgreSQL table (`"postgresql"`)\*
- SQLite table (`"sqlite"`)\*
- Microsoft SQL Server table (`"mssql"`)\*
- Snowflake table (`"snowflake"`)\*
- Databricks table (`"databricks"`)\*
- BigQuery table (`"bigquery"`)\*
- Parquet table (`"parquet"`)\*
- CSV files (string path or `pathlib.Path` object with `.csv` extension)
- Parquet files (string path, `pathlib.Path` object, glob pattern, directory with `.parquet` extension, or partitioned dataset)
- Database connection strings (URI format with optional table specification)

The table types marked with an asterisk need to be prepared as Ibis tables (with type of `ibis.expr.types.relations.Table`). Furthermore, using [missing_vals_tbl()](../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) with these types of tables requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a Polars or Pandas DataFrame, the availability of Ibis is not needed.

## The Missing Values Table

The missing values table shows the proportion of missing values in each column of the input table. The table is divided into sectors, with each sector representing a range of rows in the table. The proportion of missing values in each sector is calculated for each column. The table is displayed using the Great Tables API, which allows for further customization of the table's appearance.

To ensure that the table can scale to tables with many columns, each row in the reporting table represents a column in the input table. There are 10 sectors shown in the table, where the first sector represents the first 10% of the rows, the second sector represents the next 10% of the rows, and so on. Any sectors that are light blue indicate that there are no missing values in that sector. If there are missing values, the proportion of missing values is shown by a gray color (light gray for low proportions, dark gray to black for very high proportions).

## Examples

The [missing_vals_tbl()](../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) function is useful for quickly identifying columns with missing values in a table. Here's an example using the `nycflights` dataset (loaded as a Polars DataFrame using the [`load_dataset()`](../reference/load_dataset.md#pointblank.load_dataset) function):

``` python
import pointblank as pb

nycflights = pb.load_dataset("nycflights", tbl_type="polars")

pb.missing_vals_tbl(nycflights)
```

| Missing Values   46,595 in total |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|
| PolarsRows336,776Columns18 |  |  |  |  |  |  |  |  |  |  |
| Column | Row Sector |  |  |  |  |  |  |  |  |  |
|  | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| year |  |  |  |  |  |  |  |  |  |  |
| month |  |  |  |  |  |  |  |  |  |  |
| day |  |  |  |  |  |  |  |  |  |  |
| dep_time |  |  |  |  |  |  |  |  |  |  |
| sched_dep_time |  |  |  |  |  |  |  |  |  |  |
| dep_delay |  |  |  |  |  |  |  |  |  |  |
| arr_time |  |  |  |  |  |  |  |  |  |  |
| sched_arr_time |  |  |  |  |  |  |  |  |  |  |
| arr_delay |  |  |  |  |  |  |  |  |  |  |
| carrier |  |  |  |  |  |  |  |  |  |  |
| flight |  |  |  |  |  |  |  |  |  |  |
| tailnum |  |  |  |  |  |  |  |  |  |  |
| origin |  |  |  |  |  |  |  |  |  |  |
| dest |  |  |  |  |  |  |  |  |  |  |
| air_time |  |  |  |  |  |  |  |  |  |  |
| distance |  |  |  |  |  |  |  |  |  |  |
| hour |  |  |  |  |  |  |  |  |  |  |
| minute |  |  |  |  |  |  |  |  |  |  |
|  NO MISSING VALUES     PROPORTION MISSING:   0% 100% ROW SECTORS 1 - 33677 33678 - 67354 67355 - 101031 101032 - 134708 134709 - 168385 168386 - 202062 202063 - 235739 235740 - 269416 269417 - 303093 303094 - 336776 |  |  |  |  |  |  |  |  |  |  |

The table shows the proportion of missing values in each column of the `nycflights` dataset. The table is divided into sectors, with each sector representing a range of rows in the table (with around 34,000 rows per sector). The proportion of missing values in each sector is calculated for each column. The various shades of gray indicate the proportion of missing values in each sector. Many columns have no missing values at all, and those sectors are colored light blue.
