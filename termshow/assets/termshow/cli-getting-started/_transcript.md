```console
> # Getting Started with the Pointblank CLI
>
> # See all available commands for the Pointblank CLI
> pb --help
Usage: pb [OPTIONS] COMMAND [ARGS]...

  Pointblank CLI: Data validation and quality tools for data engineers.

  Use this CLI to validate data quality, explore datasets, and generate
  comprehensive reports for CSV, Parquet, and database sources. Suitable for
  data pipelines, ETL validation, and exploratory data analysis from the
  command line.

  Quick Examples:

    pb preview data.csv              Preview your data
    pb scan data.csv                 Generate data profile
    pb validate data.csv             Run basic validation

  Use pb COMMAND --help for detailed help on any command.

Options:
  -v, --version  Show the version and exit.
  -h, --help     Show this message and exit.

Commands:
  info           Display information about a data source.
  preview        Preview a data table showing head and tail rows.
  scan           Generate a data scan profile report.
  missing        Generate a missing values report for a data table.
  validate       Perform single or multiple data validations.
  run            Run a Pointblank validation script or YAML configuration.
  compare        Compare two tables row by row.
  make-template  Create a validation script or YAML configuration template.
  pl             Execute Polars expressions and display results.
  datasets       List available built-in datasets.
  requirements   Check installed dependencies and their availability.
  edit           Edit an existing validation plan with a natural-language...
>
> # There are a number of datasets you can use for testing
> pb datasets
Available Pointblank Datasets

  Dataset Name   Dimensions                  Description
 ──────────────────────────────────────────────────────────────────────────────────
  small_table    13 rows × 8 columns         Small demo dataset for testing
  game_revenue   2,000 rows × 11 columns     Game development company revenue data
  nycflights     336,776 rows × 18 columns   NYC airport flights data from 2013
  global_sales   50,000 rows × 20 columns    Global sales data across regions


Use these dataset names directly with any pb CLI command.
Example: pb preview small_table
>
> # Learn about all the `pb validate` options
> pb validate --help
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
>
> # Your first validation: check for duplicates with rows-distinct
> pb validate small_table --check rows-distinct
✓ Loaded data source: small_table
✓ rows-distinct validation completed

Validation Result: Rows Distinct

  Property            Value
 ───────────────────────────────────
  Data Source         small_table
  Check Type          rows-distinct
  Total Rows Tested   13
  Passing Rows        11
  Failing Rows        2
  Result              ✗ FAILED
  Duplicate Rows      2 found0m


╭────────────────────────────────────────────────────────────╮
│ ✗ Validation FAILED: 2 duplicate rows found in small_table │
│ 💡 Tip: Use --show-extract to see the failing rows         │
╰────────────────────────────────────────────────────────────╯
>
> # You can validate any sort of dataset you want: CSV, Parquet, or a DB table
```
