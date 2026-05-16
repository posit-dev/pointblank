# pb missing


Generate a missing values report for a data table.


``` bash
pb missing [OPTIONS] [DATA_SOURCE]
```


DATA_SOURCE can be:

- CSV file path (e.g., data.csv)
- Parquet file path or pattern (e.g., data.parquet, data/\*.parquet)
- GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
- Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
- Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)
- Piped data from pb pl command


<span class="gd-details-chevron" aria-hidden="true"></span>Full --help output


    Usage: pb missing [OPTIONS] [DATA_SOURCE]

      Generate a missing values report for a data table.

      DATA_SOURCE can be:

      - CSV file path (e.g., data.csv)
      - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
      - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
      - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
      - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)
      - Piped data from pb pl command

    Options:
      --output-html PATH  Save HTML output to file
      --help              Show this message and exit.


# Arguments


`DATA_SOURCE: TEXT`  
Optional.


# Options


`-output-html: PATH`  
Save HTML output to file
