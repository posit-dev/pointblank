# connect_to_table()

Connect to a database table using a connection string.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L4988-L5164)

``` python
connect_to_table(connection_string)
```

This utility function tests whether a connection string leads to a valid table and returns the table object if successful. It provides helpful error messages when no table is specified or when backend dependencies are missing.

## Parameters

`connection_string: str`  
A database connection string with a required table specification using the `::table_name` suffix. Supported formats are outlined in the *Supported Connection String Formats* section.

## Returns

`Any`  
An Ibis table object for the specified database table.

## Supported Connection String Formats

The `connection_string` parameter must include a valid connection string with a table name specified using the `::` syntax. Here are some examples on how to format connection strings for various backends:

    DuckDB:     "duckdb:///path/to/database.ddb::table_name"
    SQLite:     "sqlite:///path/to/database.db::table_name"
    PostgreSQL: "postgresql://user:password@localhost:5432/database::table_name"
    MySQL:      "mysql://user:password@localhost:3306/database::table_name"
    BigQuery:   "bigquery://project/dataset::table_name"
    Snowflake:  "snowflake://user:password@account/database/schema::table_name"

If the connection string does not include a table name, the function will attempt to connect to the database and list available tables, providing guidance on how to specify a table.

## Examples

Connect to a DuckDB table:

``` python
import pointblank as pb

# Get path to a DuckDB database file from package data
duckdb_path = pb.get_data_path("game_revenue", "duckdb")

# Connect to the `game_revenue` table in the DuckDB database
game_revenue = pb.connect_to_table(f"duckdb:///{duckdb_path}::game_revenue")

# Use with the `preview()` function
pb.preview(game_revenue)
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

Here are some backend-specific connection examples:

``` python
# PostgreSQL
pg_table = pb.connect_to_table(
    "postgresql://user:password@localhost:5432/warehouse::customer_data"
)

# SQLite
sqlite_table = pb.connect_to_table("sqlite:///local_data.db::products")

# BigQuery
bq_table = pb.connect_to_table("bigquery://my-project/analytics::daily_metrics")
```

This function requires the Ibis library with appropriate backend drivers:

``` bash
# You can install a set of common backends:
pip install 'ibis-framework[duckdb,postgres,mysql,sqlite]'

# ...or specific backends as needed:
pip install 'ibis-framework[duckdb]'    # for DuckDB
pip install 'ibis-framework[postgres]'  # for PostgreSQL
```

## See Also

print_database_tables : List all available tables in a database for discovery
