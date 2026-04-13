## print_database_tables()


List all tables in a database from a connection string.


Usage

``` python
print_database_tables(connection_string)
```


The [print_database_tables()](print_database_tables.md#pointblank.print_database_tables) function connects to a database and returns a list of all available tables. This is particularly useful for discovering what tables exist in a database before connecting to a specific table with \`connect_to_table(). The function automatically filters out temporary Ibis tables (memtables) to show only user tables. It supports all database backends available through Ibis, including DuckDB, SQLite, PostgreSQL, MySQL, BigQuery, and Snowflake.


## Parameters


`connection_string: str`  
A database connection string *without* the `::table_name` suffix. Example: `"duckdb:///path/to/database.ddb"`.


## Returns


`list[str]`  
List of table names, excluding temporary Ibis tables.


#### See Also

- [connect_to_table()](connect_to_table.md): Connect to a database table with full connection string documentation
