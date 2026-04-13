# pb scan


    Usage: pb scan [OPTIONS] [DATA_SOURCE]

      Generate a data scan profile report.

      Produces a comprehensive data profile including:

      - Column types and distributions
      - Missing value patterns
      - Basic statistics
      - Data quality indicators

      DATA_SOURCE can be:

      - CSV file path (e.g., data.csv)
      - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
      - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
      - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
      - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)
      - Piped data from pb pl command

    Options:
      --output-html PATH  Save HTML scan report to file
      -c, --columns TEXT  Comma-separated list of columns to scan
      --help              Show this message and exit.
