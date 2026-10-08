# Data backends reference

Pointblank validates tables from multiple backends through a
unified API. The validation methods work identically regardless
of the backend.

## Supported backends

| Backend       | Input type              | Extra dependency     |
|---------------|-------------------------|----------------------|
| Polars        | `pl.DataFrame`, `pl.LazyFrame` | `polars`       |
| Pandas        | `pd.DataFrame`          | `pandas`             |
| DuckDB        | `ibis.Table`            | `ibis-framework[duckdb]` |
| PostgreSQL    | `ibis.Table`            | `ibis-framework[postgres]` |
| MySQL         | `ibis.Table`            | `ibis-framework[mysql]` |
| SQLite        | `ibis.Table`            | `ibis-framework[sqlite]` |
| Snowflake     | `ibis.Table`            | `ibis-framework[snowflake]` |
| CSV files     | File path string        | (none)               |
| Parquet files | File path string        | (none)               |

## Connecting to databases

Use `pb.connect_to_table()` with a connection string. Append
`::table_name` to specify the table:

```python
import pointblank as pb

# DuckDB
tbl = pb.connect_to_table("duckdb:///warehouse.db::sales")

# PostgreSQL
tbl = pb.connect_to_table("postgresql://user:pass@host:5432/db::orders")

# MySQL
tbl = pb.connect_to_table("mysql://user:pass@host:3306/db::customers")

# SQLite
tbl = pb.connect_to_table("sqlite:///local.db::events")

# Snowflake
tbl = pb.connect_to_table("snowflake://user:pass@account/db/schema::table")
```

## Using file paths directly

Pass CSV or Parquet paths as the `data` argument:

```python
validation = (
    pb.Validate(data="data/orders.csv")
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)

# Parquet files
validation = (
    pb.Validate(data="s3://bucket/data.parquet")
    .col_vals_not_null(columns="id")
    .interrogate()
)
```

## Built-in datasets

Pointblank includes example datasets for testing:

```python
df = pb.load_dataset("small_table")               # default: Polars
df = pb.load_dataset("game_revenue", tbl_type="pandas")
df = pb.load_dataset("nycflights", tbl_type="duckdb")
df = pb.load_dataset("global_sales", tbl_type="polars")
```

Available datasets: `small_table`, `game_revenue`, `nycflights`,
`global_sales`.

Available types: `"polars"` (default), `"pandas"`, `"duckdb"`.

## Listing database tables

```python
pb.print_database_tables("duckdb:///warehouse.db")
```

## Utility functions

```python
pb.get_row_count(df)      # row count for any backend
pb.get_column_count(df)   # column count for any backend
pb.preview(df)            # quick visual preview as GT table
```
