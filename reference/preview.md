# preview()

Display a table preview that shows some rows from the top, some from the bottom.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L2093-L2356)

``` python
preview(
    data,
    columns_subset=None,
    n_head=5,
    n_tail=5,
    limit=50,
    show_row_numbers=True,
    max_col_width=250,
    min_tbl_width=500,
    incl_header=None,
)
```

To get a quick look at the data in a table, we can use the [preview()](../reference/preview.md#pointblank.preview) function to display a preview of the table. The function shows a subset of the rows from the start and end of the table, with the number of rows from the start and end determined by the `n_head=` and `n_tail=` parameters (set to `5` by default). This function works with any table that is supported by the `pointblank` library, including Pandas, Polars, and Ibis backend tables (e.g., DuckDB, MySQL, PostgreSQL, SQLite, Parquet, etc.).

The view is optimized for readability, with column names and data types displayed in a compact format. The column widths are sized to fit the column names, dtypes, and column content up to a configurable maximum width of `max_col_width=` pixels. The table can be scrolled horizontally to view even very large datasets. Since the output is a Great Tables (`GT`) object, it can be further customized using the `great_tables` API.

## Parameters

`data: Any`  
The table to preview, which could be a DataFrame object, an Ibis table object, a CSV file path, a Parquet file path, or a database connection string. When providing a CSV or Parquet file path (as a string or `pathlib.Path` object), the file will be automatically loaded using an available DataFrame library (Polars or Pandas). Parquet input also supports glob patterns, directories containing .parquet files, and Spark-style partitioned datasets. Connection strings enable direct database access via Ibis with optional table specification using the `::table_name` suffix. Read the *Supported Input Table Types* section for details on the supported table types.

`columns_subset: str | list[str] | Column | None = None`  
The columns to display in the table, by default `None` (all columns are shown). This can be a string, a list of strings, a `Column` object, or a `ColumnSelector` object. The latter two options allow for more flexible column selection using column selector functions. Errors are raised if the column names provided don't match any columns in the table (when provided as a string or list of strings) or if column selector expressions don't resolve to any columns.

`n_head: int = 5`  
The number of rows to show from the start of the table. Set to `5` by default.

`n_tail: int = 5`  
The number of rows to show from the end of the table. Set to `5` by default.

`limit: int = 50`  
The limit value for the sum of `n_head=` and `n_tail=` (the total number of rows shown). If the sum of `n_head=` and `n_tail=` exceeds the limit, an error is raised. The default value is `50`.

`show_row_numbers: bool = True`  
Should row numbers be shown? The numbers shown reflect the row numbers of the head and tail in the input `data=` table. By default, this is set to `True`.

`max_col_width: int = 250`  
The maximum width of the columns (in pixels) before the text is truncated. The default value is `250` (`"250px"`).

`min_tbl_width: int = 500`  
The minimum width of the table in pixels. If the sum of the column widths is less than this value, the all columns are sized up to reach this minimum width value. The default value is `500` (`"500px"`).

`incl_header: bool | None = None`  
Should the table include a header with the table type and table dimensions? Set to `True` by default.

## Returns

`GT`  
A GT object that displays the preview of the table.

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

The table types marked with an asterisk need to be prepared as Ibis tables (with type of `ibis.expr.types.relations.Table`). Furthermore, using [preview()](../reference/preview.md#pointblank.preview) with these types of tables requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a Polars or Pandas DataFrame, the availability of Ibis is not needed.

To use a CSV file, ensure that a string or `pathlib.Path` object with a `.csv` extension is provided. The file will be automatically detected and loaded using the best available DataFrame library. The loading preference is Polars first, then Pandas as a fallback.

Connection strings follow database URL formats and must also specify a table using the `::table_name` suffix. Examples include:

    "duckdb:///path/to/database.ddb::table_name"
    "sqlite:///path/to/database.db::table_name"
    "postgresql://user:password@localhost:5432/database::table_name"
    "mysql://user:password@localhost:3306/database::table_name"
    "bigquery://project/dataset::table_name"
    "snowflake://user:password@account/database/schema::table_name"

When using connection strings, the Ibis library with the appropriate backend driver is required.

## Examples

It's easy to preview a table using the [preview()](../reference/preview.md#pointblank.preview) function. Here's an example using the `small_table` dataset (itself loaded using the [`load_dataset()`](../reference/load_dataset.md#pointblank.load_dataset) function):

``` python
import pointblank as pb

small_table_polars = pb.load_dataset("small_table")

pb.preview(small_table_polars)
```

| PolarsRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | None | 2230.09 | True | high |

This table is a Polars DataFrame, but the [preview()](../reference/preview.md#pointblank.preview) function works with any table supported by `pointblank`, including Pandas DataFrames and Ibis backend tables. Here's an example using a DuckDB table handled by Ibis:

``` python
small_table_duckdb = pb.load_dataset("small_table", tbl_type="duckdb")

pb.preview(small_table_duckdb)
```

| DuckDBRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *timestamp* | date *date* | a *int64* | b *string* | c *int64* | d *float64* | e *boolean* | f *string* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | NULL | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | NULL | 2230.09 | True | high |

The blue dividing line marks the end of the first `n_head=` rows and the start of the last `n_tail=` rows.

We can adjust the number of rows shown from the start and end of the table by setting the `n_head=` and `n_tail=` parameters. Let's enlarge each of these to `10`:

``` python
pb.preview(small_table_polars, n_head=10, n_tail=10)
```

| PolarsRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 6 | 2016-01-11 06:15:00 | 2016-01-11 | 4 | 2-dhe-923 | 4 | 3291.03 | True | mid |
| 7 | 2016-01-15 18:46:00 | 2016-01-15 | 7 | 1-knw-093 | 3 | 843.34 | True | high |
| 8 | 2016-01-17 11:27:00 | 2016-01-17 | 4 | 5-boe-639 | 2 | 1035.64 | False | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | None | 2230.09 | True | high |

In the above case, the entire dataset is shown since the sum of `n_head=` and `n_tail=` is greater than the number of rows in the table (which is 13).

The `columns_subset=` parameter can be used to show only specific columns in the table. You can provide a list of column names to make the selection. Let's try that with the `"game_revenue"` dataset as a Pandas DataFrame:

``` python
game_revenue_pandas = pb.load_dataset("game_revenue", tbl_type="pandas")

pb.preview(game_revenue_pandas, columns_subset=["player_id", "item_name", "item_revenue"])
```

| PandasRows2,000Columns11 |  |  |  |
|----|----|----|----|
|  | player_id *object* | item_name *object* | item_revenue *float64* |
| 1 | ECPANOIXLZHF896 | offer2 | 8.99 |
| 2 | ECPANOIXLZHF896 | gems3 | 22.49 |
| 3 | ECPANOIXLZHF896 | gold7 | 107.99 |
| 4 | ECPANOIXLZHF896 | ad_20sec | 0.76 |
| 5 | ECPANOIXLZHF896 | ad_5sec | 0.03 |
| 1996 | NAOJRDMCSEBI281 | ad_survey | 1.332 |
| 1997 | NAOJRDMCSEBI281 | ad_survey | 1.35 |
| 1998 | RMOSWHJGELCI675 | ad_5sec | 0.03 |
| 1999 | RMOSWHJGELCI675 | offer5 | 26.09 |
| 2000 | GJCXNTWEBIPQ369 | ad_5sec | 0.12 |

Alternatively, we can use column selector functions like [`starts_with()`](../reference/starts_with.md#pointblank.starts_with) and [`matches()`](../reference/matches.md#pointblank.matches)\` to select columns based on text or patterns:

``` python
pb.preview(game_revenue_pandas, n_head=2, n_tail=2, columns_subset=pb.starts_with("session"))
```

| PandasRows2,000Columns11 |  |  |  |
|----|----|----|----|
|  | session_id *object* | session_start *datetime64\[ns, UTC\]* | session_duration *float64* |
| 1 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 16.3 |
| 2 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 16.3 |
| 1999 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 8.4 |
| 2000 | GJCXNTWEBIPQ369-9elq67md | 2015-01-21 03:59:23+00:00 | 18.5 |

Multiple column selector functions can be combined within [`col()`](../reference/col.md#pointblank.col) using operators like `|` and `&`:

``` python
pb.preview(
  game_revenue_pandas,
  n_head=2,
  n_tail=2,
  columns_subset=pb.col(pb.starts_with("item") | pb.matches("player"))
)
```

| PandasRows2,000Columns11 |  |  |  |  |
|----|----|----|----|----|
|  | player_id *object* | item_type *object* | item_name *object* | item_revenue *float64* |
| 1 | ECPANOIXLZHF896 | iap | offer2 | 8.99 |
| 2 | ECPANOIXLZHF896 | iap | gems3 | 22.49 |
| 1999 | RMOSWHJGELCI675 | iap | offer5 | 26.09 |
| 2000 | GJCXNTWEBIPQ369 | ad | ad_5sec | 0.12 |

### Working with CSV Files

The [preview()](../reference/preview.md#pointblank.preview) function can directly accept CSV file paths, making it easy to preview data stored in CSV files without manual loading:

``` python
# Get a path to a CSV file from the package data
csv_path = pb.get_data_path("global_sales", "csv")

pb.preview(csv_path)
```

| PolarsRows50,000Columns20 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | product_id *String* | product_category *String* | customer_id *String* | customer_segment *String* | region *String* | country *String* | city *String* | timestamp *Datetime* | quarter *String* | month *Int64* | year *Int64* | price *Float64* | quantity *Int64* | status *String* | email *String* | revenue *Float64* | tax *Float64* | total *Float64* | payment_method *String* | sales_channel *String* |
| 1 | 98b70df0 | Manufacturing | cf3b13c7 | Government | Asia Pacific | Australia | Melbourne | 2021-12-25 19:00:00 | 2021-Q4 | 12 | 2021 | 186.0 | 7 | returned | user1651@test.org | 1302.0 | 127.45 | 1429.45 | Apple Pay | Partner |
| 2 | 9d09fef5 | Manufacturing | 08b5db12 | Consumer | Europe | France | Nice | 2022-06-12 17:25:00 | 2022-Q2 | 6 | 2022 | 137.03 | 8 | returned | user5200@company.io | 1096.24 | 222.52 | 1318.76 | PayPal | Distributor |
| 3 | 8ac6b077 | Retail | 41079b2e | Consumer | Europe | France | Toulouse | 2023-05-06 09:09:00 | 2023-Q2 | 5 | 2023 | 330.08 | 4 | shipped | user9180@mockdata.com | 1320.32 | 260.89 | 1581.21 | PayPal | Phone |
| 4 | 13d2df9d | Healthcare | b421eece | Consumer | North America | USA | Miami | 2023-10-11 16:53:00 | 2023-Q4 | 10 | 2023 | 420.09 | 3 | shipped | user1636@example.com | 1260.27 | 103.99 | 1364.26 | Bank Transfer | Phone |
| 5 | 98b70df0 | Manufacturing | 5906a04f | SMB | North America | Canada | Calgary | 2022-05-05 01:53:00 | 2022-Q2 | 5 | 2022 | 187.77 | 3 | delivered | user9971@mockdata.com | 563.31 | 75.73 | 639.04 | Credit Card | Phone |
| 49996 | 53a36468 | Finance | 966a8bbe | Government | Asia Pacific | Australia | Melbourne | 2023-11-04 14:45:00 | 2023-Q4 | 11 | 2023 | 198.18 | 1 | pending | user8593@test.org | 198.18 | 18.3 | 216.48 | Google Pay | Partner |
| 49997 | a42fd1ff | Healthcare | ff8933e4 | SMB | Asia Pacific | Japan | Kyoto | 2023-04-27 17:27:00 | 2023-Q2 | 4 | 2023 | 419.72 | 2 | returned | user5448@company.io | 839.44 | 90.49 | 929.93 | Google Pay | Partner |
| 49998 | bbf158d2 | Technology | f0c0af3f | Enterprise | North America | USA | Los Angeles | 2021-04-24 23:15:00 | 2021-Q2 | 4 | 2021 | 302.52 | 1 | pending | user1463@test.org | 302.52 | 21.68 | 324.2 | Bank Transfer | Online |
| 49999 | 2a0866de | Healthcare | 5b27ba59 | SMB | Europe | France | Nice | 2023-12-30 19:44:00 | 2023-Q4 | 12 | 2023 | 433.82 | 5 | pending | user4167@test.org | 2169.1 | 448.87 | 2617.97 | Credit Card | Online |
| 50000 | 6260f67c | Technology | 482c1d84 | Consumer | Asia Pacific | Japan | Kyoto | 2021-12-05 09:49:00 | 2021-Q4 | 12 | 2021 | 400.31 | 8 | returned | user4238@example.com | 3202.48 | 339.84 | 3542.32 | Apple Pay | Distributor |

You can also use a Path object to specify the CSV file:

``` python
from pathlib import Path

csv_file = Path(pb.get_data_path("game_revenue", "csv"))

pb.preview(csv_file, n_head=3, n_tail=3)
```

| PolarsRows2,000Columns11 |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|
|  | player_id *String* | session_id *String* | session_start *Datetime* | time *Datetime* | item_type *String* | item_name *String* | item_revenue *Float64* | session_duration *Float64* | start_day *Date* | acquisition *String* | country *String* |
| 1 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:31:27+00:00 | iap | offer2 | 8.99 | 16.3 | 2015-01-01 | google | Germany |
| 2 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:36:57+00:00 | iap | gems3 | 22.49 | 16.3 | 2015-01-01 | google | Germany |
| 3 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:37:45+00:00 | iap | gold7 | 107.99 | 16.3 | 2015-01-01 | google | Germany |
| 1998 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:40:00+00:00 | ad | ad_5sec | 0.03 | 8.4 | 2015-01-10 | other_campaign | France |
| 1999 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:47:12+00:00 | iap | offer5 | 26.09 | 8.4 | 2015-01-10 | other_campaign | France |
| 2000 | GJCXNTWEBIPQ369 | GJCXNTWEBIPQ369-9elq67md | 2015-01-21 03:59:23+00:00 | 2015-01-21 04:06:29+00:00 | ad | ad_5sec | 0.12 | 18.5 | 2015-01-14 | organic | United States |

### Working with Parquet Files

The [preview()](../reference/preview.md#pointblank.preview) function can directly accept Parquet files and datasets in various formats:

``` python
# Single Parquet file from package data
parquet_path = pb.get_data_path("nycflights", "parquet")

pb.preview(parquet_path)
```

| PolarsRows336,776Columns18 |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|----|
|  | year *Int64* | month *Int64* | day *Int64* | dep_time *Int64* | sched_dep_time *Int64* | dep_delay *Int64* | arr_time *Int64* | sched_arr_time *Int64* | arr_delay *Int64* | carrier *String* | flight *Int64* | tailnum *String* | origin *String* | dest *String* | air_time *Int64* | distance *Int64* | hour *Int64* | minute *Int64* |
| 1 | 2013 | 1 | 1 | 517 | 515 | 2 | 830 | 819 | 11 | UA | 1545 | N14228 | EWR | IAH | 227 | 1400 | 5 | 15 |
| 2 | 2013 | 1 | 1 | 533 | 529 | 4 | 850 | 830 | 20 | UA | 1714 | N24211 | LGA | IAH | 227 | 1416 | 5 | 29 |
| 3 | 2013 | 1 | 1 | 542 | 540 | 2 | 923 | 850 | 33 | AA | 1141 | N619AA | JFK | MIA | 160 | 1089 | 5 | 40 |
| 4 | 2013 | 1 | 1 | 544 | 545 | -1 | 1004 | 1022 | -18 | B6 | 725 | N804JB | JFK | BQN | 183 | 1576 | 5 | 45 |
| 5 | 2013 | 1 | 1 | 554 | 600 | -6 | 812 | 837 | -25 | DL | 461 | N668DN | LGA | ATL | 116 | 762 | 6 | 0 |
| 336772 | 2013 | 9 | 30 | None | 1455 | None | None | 1634 | None | 9E | 3393 | None | JFK | DCA | None | 213 | 14 | 55 |
| 336773 | 2013 | 9 | 30 | None | 2200 | None | None | 2312 | None | 9E | 3525 | None | LGA | SYR | None | 198 | 22 | 0 |
| 336774 | 2013 | 9 | 30 | None | 1210 | None | None | 1330 | None | MQ | 3461 | N535MQ | LGA | BNA | None | 764 | 12 | 10 |
| 336775 | 2013 | 9 | 30 | None | 1159 | None | None | 1344 | None | MQ | 3572 | N511MQ | LGA | CLE | None | 419 | 11 | 59 |
| 336776 | 2013 | 9 | 30 | None | 840 | None | None | 1020 | None | MQ | 3531 | N839MQ | LGA | RDU | None | 431 | 8 | 40 |

You can also use glob patterns and directories:

``` python
# Multiple Parquet files with glob patterns
pb.preview("data/sales_*.parquet")

# Directory containing Parquet files
pb.preview("parquet_data/")

# Partitioned Parquet dataset
pb.preview("sales_data/")  # Auto-discovers partition columns
```

### Working with Database Connection Strings

The [preview()](../reference/preview.md#pointblank.preview) function supports database connection strings for direct preview of database tables. Connection strings must specify a table using the `::table_name` suffix:

``` python
# Get path to a DuckDB database file from package data
duckdb_path = pb.get_data_path("game_revenue", "duckdb")

pb.preview(f"duckdb:///{duckdb_path}::game_revenue")
```

| DuckDBRows2,000Columns11 |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|
|  | player_id *string* | session_id *string* | session_start *timestamp* | time *timestamp* | item_type *string* | item_name *string* | item_revenue *float64* | session_duration *float64* | start_day *date* | acquisition *string* | country *string* |
| 1 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:31:27+00:00 | iap | offer2 | 8.99 | 16.3 | 2015-01-01 | google | Germany |
| 2 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:36:57+00:00 | iap | gems3 | 22.49 | 16.3 | 2015-01-01 | google | Germany |
| 3 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:37:45+00:00 | iap | gold7 | 107.99 | 16.3 | 2015-01-01 | google | Germany |
| 4 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:42:33+00:00 | ad | ad_20sec | 0.76 | 16.3 | 2015-01-01 | google | Germany |
| 5 | ECPANOIXLZHF896 | ECPANOIXLZHF896-hdu9jkls | 2015-01-01 11:50:02+00:00 | 2015-01-01 11:55:20+00:00 | ad | ad_5sec | 0.03 | 35.2 | 2015-01-01 | google | Germany |
| 1996 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:02:50+00:00 | ad | ad_survey | 1.332 | 25.8 | 2015-01-11 | organic | Norway |
| 1997 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:22:14+00:00 | ad | ad_survey | 1.35 | 25.8 | 2015-01-11 | organic | Norway |
| 1998 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:40:00+00:00 | ad | ad_5sec | 0.03 | 8.4 | 2015-01-10 | other_campaign | France |
| 1999 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:47:12+00:00 | iap | offer5 | 26.09 | 8.4 | 2015-01-10 | other_campaign | France |
| 2000 | GJCXNTWEBIPQ369 | GJCXNTWEBIPQ369-9elq67md | 2015-01-21 03:59:23+00:00 | 2015-01-21 04:06:29+00:00 | ad | ad_5sec | 0.12 | 18.5 | 2015-01-14 | organic | United States |

For comprehensive documentation on supported connection string formats, error handling, and installation requirements, see the [`connect_to_table()`](../reference/connect_to_table.md#pointblank.connect_to_table) function.
