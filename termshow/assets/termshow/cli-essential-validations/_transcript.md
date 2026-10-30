```console
> # Essential Data Validation with Pointblank CLI
>
> # 1. Get the table dimensions, just to know what you're working with
> pb info small_table

Data Source Information

  Property     Value
 ──────────────────────────
  Source       small_table
  Table Type   polars
  Rows         13
  Columns      8

>
> # 2. Check for duplicate rows (the most common data quality issue)
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
> # 3. Find missing values in important columns
> pb validate small_table --check col-vals-not-null --column c
✓ Loaded data source: small_table
✓ col-vals-not-null validation completed

Validation Result: Column Values Not Null

  Property            Value
 ───────────────────────────────────────
  Data Source         small_table
  Check Type          col-vals-not-null
  Column              c
  Total Rows Tested   13
  Passing Rows        11
  Failing Rows        2
  Result           0m   ✗ FAILED
  Null Values         2 found


╭───────────────────────────────────────────────────────────────────────╮
│ ✗ Validation FAILED: 2 null values found in column 'c' in small_table │
│ 💡 Tip: Use --show-extract to see the failing rows                    │
╰───────────────────────────────────────────────────────────────────────╯
>
> # 4. When validation fails, see exactly which rows are problematic with --show-extract
> pb validate small_table --check col-vals-not-null --column c --show-extract
✓ Loaded data source: small_table
✓ col-vals-not-null validation completed

Validation Result: Column Values Not Null

  Property            Value
 ───────────────────────────────────────
  Data Source         small_table
  Check Type          col-vals-not-null
  Column              c
  Total Rows Tested   13
  Passing Rows        11
  Failing Rows        2
  Result           0m   ✗ FAILED
  Null Values         2 found


Extract of failing rows (null values in 'c'):
Showing all 2 rows with null values


           date_time             date         a       b           c       d         e        f
           <datetime>            <date>       <i64>   <str>       <i64>   <f64>     <bool>   <str>
 ──────────────────────────────────────────────────────────────────────────────────────────────────
       4   2016-01-06 17:23:00   2016-01-06   2       5-jdo-903   None    3892.4    False    mid
      13   2016-01-30 11:23:00   2016-01-30   1       3-dka-303   None    2230.09   True     high


╭───────────────────────────────────────────────────────────────────────╮
│ ✗ Validation FAILED: 2 null values found in column 'c' in small_table │
╰───────────────────────────────────────────────────────────────────────╯
>
> # That is just a sampling of the essentials with duplicates, nulls, with data extracts
```
