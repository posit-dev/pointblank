# pb validate


Perform single or multiple data validations.


``` bash
pb validate [OPTIONS] [DATA_SOURCE]
```


Run one or more validation checks on your data in a single command. Use multiple `--check` options to perform multiple validations.

DATA_SOURCE can be:

- CSV file path (e.g., data.csv)
- Parquet file path or pattern (e.g., data.parquet, data/\*.parquet)
- GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
- Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
- Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)

AVAILABLE CHECK_TYPES:

Require no additional options:

- rows-distinct: Check if all rows in the dataset are unique (no duplicates)
- rows-complete: Check if all rows are complete (no missing values in any column)

Require `--column`:

- col-exists: Check if a specific column exists in the dataset
- col-vals-not-null: Check if all values in a column are not null/missing

Require `--column` and `--value`:

- col-vals-gt: Check if column values are greater than a fixed value
- col-vals-ge: Check if column values are greater than or equal to a fixed value
- col-vals-lt: Check if column values are less than a fixed value
- col-vals-le: Check if column values are less than or equal to a fixed value

Require `--column` and `--set`:

- col-vals-in-set: Check if column values are in an allowed set

Use `--list-checks` to see all available validation methods with examples. The default CHECK_TYPE is `rows-distinct` which checks for duplicate rows.


<span class="gd-details-chevron" aria-hidden="true"></span>Full --help output


    Usage: pb validate [OPTIONS] [DATA_SOURCE]

      Perform single or multiple data validations.

      Run one or more validation checks on your data in a single command. Use
      multiple --check options to perform multiple validations.

      DATA_SOURCE can be:

      - CSV file path (e.g., data.csv)
      - Parquet file path or pattern (e.g., data.parquet, data/*.parquet)
      - GitHub URL to CSV/Parquet (e.g., https://github.com/user/repo/blob/main/data.csv)
      - Database connection string (e.g., duckdb:///path/to/db.ddb::table_name)
      - Dataset name from pointblank (small_table, game_revenue, nycflights, global_sales)

      AVAILABLE CHECK_TYPES:

      Require no additional options:

      - rows-distinct: Check if all rows in the dataset are unique (no duplicates)
      - rows-complete: Check if all rows are complete (no missing values in any column)

      Require --column:

      - col-exists: Check if a specific column exists in the dataset
      - col-vals-not-null: Check if all values in a column are not null/missing

      Require --column and --value:

      - col-vals-gt: Check if column values are greater than a fixed value
      - col-vals-ge: Check if column values are greater than or equal to a fixed value
      - col-vals-lt: Check if column values are less than a fixed value
      - col-vals-le: Check if column values are less than or equal to a fixed value

      Require --column and --set:

      - col-vals-in-set: Check if column values are in an allowed set

      Use --list-checks to see all available validation methods with examples. The
      default CHECK_TYPE is 'rows-distinct' which checks for duplicate rows.

      Examples:

      pb validate data.csv                               # Uses default validation (rows-distinct)
      pb validate data.csv --list-checks                 # Show all available checks
      pb validate data.csv --check rows-distinct
      pb validate data.csv --check rows-distinct --show-extract
      pb validate data.csv --check rows-distinct --write-extract failing_rows_folder
      pb validate data.csv --check rows-distinct --exit-code
      pb validate data.csv --check col-exists --column price
      pb validate data.csv --check col-vals-not-null --column email
      pb validate data.csv --check col-vals-gt --column score --value 50
      pb validate data.csv --check col-vals-in-set --column status --set "active,inactive,pending"

      Multiple validations in one command: pb validate data.csv --check rows-
      distinct --check rows-complete

    Options:
      --list-checks         List available validation checks and exit
      --check CHECK_TYPE    Type of validation check to perform. Can be used
                            multiple times for multiple checks.
      --column TEXT         Column name or integer position as #N (1-based index)
                            for validation.
      --set TEXT            Comma-separated allowed values for col-vals-in-set
                            checks.
      --value FLOAT         Numeric value for comparison checks.
      --show-extract        Show extract of failing rows if validation fails
      --write-extract TEXT  Save failing rows to folder. Provide base name for
                            folder.
      --limit INTEGER       Maximum number of failing rows to save to CSV
                            (default: 500)
      --exit-code           Exit with non-zero code if validation fails
      --help                Show this message and exit.


# Arguments


`DATA_SOURCE: TEXT`  
Optional.


# Options


`-list-checks`  
List available validation checks and exit

`-check: CHOICE`  
Type of validation check to perform. Can be used multiple times for multiple checks.

`-column: TEXT`  
Column name or integer position as \#N (1-based index) for validation.

`-set: TEXT`  
Comma-separated allowed values for col-vals-in-set checks.

`-value: FLOAT`  
Numeric value for comparison checks.

`-show-extract`  
Show extract of failing rows if validation fails

`-write-extract: TEXT`  
Save failing rows to folder. Provide base name for folder.

`-limit: INTEGER = 500`  
Maximum number of failing rows to save to CSV (default: 500)

`-exit-code`  
Exit with non-zero code if validation fails


# Examples

``` bash

pb validate data.csv                               # Uses default validation (rows-distinct)
pb validate data.csv --list-checks                 # Show all available checks
pb validate data.csv --check rows-distinct
pb validate data.csv --check rows-distinct --show-extract
pb validate data.csv --check rows-distinct --write-extract failing_rows_folder
pb validate data.csv --check rows-distinct --exit-code
pb validate data.csv --check col-exists --column price
pb validate data.csv --check col-vals-not-null --column email
pb validate data.csv --check col-vals-gt --column score --value 50
pb validate data.csv --check col-vals-in-set --column status --set "active,inactive,pending"

Multiple validations in one command:
pb validate data.csv --check rows-distinct --check rows-complete
```
