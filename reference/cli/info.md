# pb info


Display information about a data source.


``` bash
pb info [DATA_SOURCE]
```


Shows table type, dimensions, column names, and data types.

DATA_SOURCE can be:

- CSV file path (e.g., data.csv)
- Parquet file path or pattern (e.g., data.parquet, data/\*.parquet)
- GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
- Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
- Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)


<span class="gd-details-chevron" aria-hidden="true"></span>Full --help output


    Usage: pb info [OPTIONS] [DATA_SOURCE]

      Display information about a data source.

      Shows table type, dimensions, column names, and data types.

      DATA_SOURCE can be:

      - CSV file path (e.g., data.csv)
      - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
      - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
      - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
      - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)

    Options:
      --help  Show this message and exit.


# Arguments


`DATA_SOURCE: TEXT`  
Optional.
