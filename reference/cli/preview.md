# pb preview


Preview a data table showing head and tail rows.


``` bash
pb preview [OPTIONS] [DATA_SOURCE]
```


DATA_SOURCE can be:

- CSV file path (e.g., data.csv)
- Parquet file path or pattern (e.g., data.parquet, data/\*.parquet)
- GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
- Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
- Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)
- Piped data from pb pl command

COLUMN SELECTION OPTIONS:

For tables with many columns, use these options to control which columns are displayed:

- `--columns`: Specify exact columns (`--columns` "name,age,email")
- `--col-range`: Select column range (`--col-range` "1:10", `--col-range` "5:", `--col-range` ":15")
- `--col-first`: Show first N columns (`--col-first` 5)
- `--col-last`: Show last N columns (`--col-last` 3)

Tables with \>15 columns automatically show first 7 and last 7 columns with indicators.


<span class="gd-details-chevron" aria-hidden="true"></span>Full --help output


    Usage: pb preview [OPTIONS] [DATA_SOURCE]

      Preview a data table showing head and tail rows.

      DATA_SOURCE can be:

      - CSV file path (e.g., data.csv)
      - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
      - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
      - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
      - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)
      - Piped data from pb pl command

      COLUMN SELECTION OPTIONS:

      For tables with many columns, use these options to control which columns are
      displayed:

      - --columns: Specify exact columns (--columns "name,age,email")
      - --col-range: Select column range (--col-range "1:10", --col-range "5:", --col-range ":15")
      - --col-first: Show first N columns (--col-first 5)
      - --col-last: Show last N columns (--col-last 3)

      Tables with >15 columns automatically show first 7 and last 7 columns with
      indicators.

    Options:
      --columns TEXT             Comma-separated list of columns to display
      --col-range TEXT           Column range like '1:10' or '5:' or ':15'
                                 (1-based indexing)
      --col-first INTEGER        Show first N columns
      --col-last INTEGER         Show last N columns
      --head INTEGER             Number of rows from the top (default: 5)
      --tail INTEGER             Number of rows from the bottom (default: 5)
      --limit INTEGER            Maximum total rows to display (default: 50)
      --no-row-numbers           Hide row numbers
      --max-col-width INTEGER    Maximum column width in pixels (default: 250)
      --min-table-width INTEGER  Minimum table width in pixels (default: 500)
      --no-header                Hide table header
      --output-html PATH         Save HTML output to file
      --help                     Show this message and exit.


# Arguments


`DATA_SOURCE: TEXT`  
Optional.


# Options


`-columns: TEXT`  
Comma-separated list of columns to display

`-col-range: TEXT`  
Column range like `1:10` or `5:` or `:15` (1-based indexing)

`-col-first: INTEGER`  
Show first N columns

`-col-last: INTEGER`  
Show last N columns

`-head: INTEGER = 5`  
Number of rows from the top (default: 5)

`-tail: INTEGER = 5`  
Number of rows from the bottom (default: 5)

`-limit: INTEGER = 50`  
Maximum total rows to display (default: 50)

`-no-row-numbers`  
Hide row numbers

`-max-col-width: INTEGER = 250`  
Maximum column width in pixels (default: 250)

`-min-table-width: INTEGER = 500`  
Minimum table width in pixels (default: 500)

`-no-header`  
Hide table header

`-output-html: PATH`  
Save HTML output to file
