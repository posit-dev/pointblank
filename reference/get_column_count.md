## get_column_count()


Get the number of columns in a table.


Usage

``` python
get_column_count(data)
```


The [get_column_count()](get_column_count.md#pointblank.get_column_count) function returns the number of columns in a table. The function works with any table that is supported by the `pointblank` library, including Pandas, Polars, and Ibis backend tables (e.g., DuckDB, MySQL, PostgreSQL, SQLite, Parquet, etc.). It also supports direct input of CSV files, Parquet files, and database connection strings.


## Parameters


`data: Any`  
The table for which to get the column count, which could be a DataFrame object, an Ibis table object, a CSV file path, a Parquet file path, or a database connection string. Read the *Supported Input Table Types* section for details on the supported table types.


## Returns


`int`  
The number of columns in the table.


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

The table types marked with an asterisk need to be prepared as Ibis tables (with type of `ibis.expr.types.relations.Table`). Furthermore, using [get_column_count()](get_column_count.md#pointblank.get_column_count) with these types of tables requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a Polars or Pandas DataFrame, the availability of Ibis is not needed.

To use a CSV file, ensure that a string or `pathlib.Path` object with a `.csv` extension is provided. The file will be automatically detected and loaded using the best available DataFrame library. The loading preference is Polars first, then Pandas as a fallback.

GitHub URLs pointing to CSV or Parquet files are automatically detected and converted to raw content URLs for downloading. The URL format should be: `https://github.com/user/repo/blob/branch/path/file.csv` or `https://github.com/user/repo/blob/branch/path/file.parquet`

Connection strings follow database URL formats and must also specify a table using the `::table_name` suffix. Examples include:

    "duckdb:///path/to/database.ddb::table_name"
    "sqlite:///path/to/database.db::table_name"
    "postgresql://user:password@localhost:5432/database::table_name"
    "mysql://user:password@localhost:3306/database::table_name"
    "bigquery://project/dataset::table_name"
    "snowflake://user:password@account/database/schema::table_name"

When using connection strings, the Ibis library with the appropriate backend driver is required.


## Examples

To get the number of columns in a table, we can use the [get_column_count()](get_column_count.md#pointblank.get_column_count) function. Here's an example using the `small_table` dataset (itself loaded using the <a href="load_dataset.html#pointblank.load_dataset" class="gdls-link"><code>load_dataset()</code></a> function):


``` python
import pointblank as pb

small_table_polars = pb.load_dataset("small_table")

pb.get_column_count(small_table_polars)
```


    8


This table is a Polars DataFrame, but the [get_column_count()](get_column_count.md#pointblank.get_column_count) function works with any table supported by `pointblank`, including Pandas DataFrames and Ibis backend tables. Here's an example using a DuckDB table handled by Ibis:


``` python
small_table_duckdb = pb.load_dataset("small_table", tbl_type="duckdb")

pb.get_column_count(small_table_duckdb)
```


    8


##### Working with CSV Files

The [get_column_count()](get_column_count.md#pointblank.get_column_count) function can directly accept CSV file paths:


``` python
# Get a path to a CSV file from the package data
csv_path = pb.get_data_path("global_sales", "csv")

pb.get_column_count(csv_path)
```


    20


##### Working with Parquet Files

The function supports various Parquet input formats:


``` python
# Single Parquet file from package data
parquet_path = pb.get_data_path("nycflights", "parquet")

pb.get_column_count(parquet_path)
```


    18


You can also use glob patterns and directories:

``` python
# Multiple Parquet files with glob patterns
pb.get_column_count("data/sales_*.parquet")

# Directory containing Parquet files
pb.get_column_count("parquet_data/")

# Partitioned Parquet dataset
pb.get_column_count("sales_data/")  # Auto-discovers partition columns
```


##### Working with Database Connection Strings

The function supports database connection strings for direct access to database tables:


``` python
# Get path to a DuckDB database file from package data
duckdb_path = pb.get_data_path("game_revenue", "duckdb")

pb.get_column_count(f"duckdb:///{duckdb_path}::game_revenue")
```


    11


The function always returns the number of columns in the table as an integer value, which is `8` for the `small_table` dataset.
