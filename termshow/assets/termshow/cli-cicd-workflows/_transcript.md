```console
> # Example of a Production CI/CD Data Validation Workflow with the Pointblank CLI
>
> # Pipeline Step 1: Checking for duplicates while using exit codes (this fails)
> pb validate small_table --check rows-distinct --exit-code
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
Exiting with non-zero code due to validation failure
>
> # Pipeline Step 2: Checking for null values in an important column (this passes)
> pb validate small_table --check col-vals-not-null --column date_time --exit-code
✓ Loaded data source: small_table
✓ col-vals-not-null validation completed

Validation Result: Column Values Not Null

  Property            Value
 ───────────────────────────────────────
  Data Source         small_table
  Check Type          col-vals-not-null
  Column              date_time
  Total Rows Tested   13
  Passing Rows        13
  Failing Rows        0
  Result           0m   ✓ PASSED
  Null Values         None found


╭────────────────────────────────────────────────────────────────────────────────╮
│ ✓ Validation PASSED: No null values found in column 'date_time' in small_table │
╰────────────────────────────────────────────────────────────────────────────────╯
>
> # Pipeline Step 3: Checking for values greater than 2 (this one fails)
> pb validate small_table --check col-vals-gt --column a --value 2 --exit-code
✓ Loaded data source: small_table
✓ col-vals-gt validation completed

Validation Result: Column Values
Greater Than

  Property            Value
 ─────────────────────────────────────
  Data Source         small_table
  Check Type          col-vals-gt
  Column              a
  Comparison Value    > 2.0
  Total Rows Tested   13
  Passing Rows        9
  Failing Rows        4
  Result              ✗ FAILED
  Invalid Values      4 values <= 2.0


╭─────────────────────────────────────────────────────────────────────────╮
│ ✗ Validation FAILED: 4 values <= 2.0 found in column 'a' in small_table │
│ 💡 Tip: Use --show-extract to see the failing rows                      │
╰─────────────────────────────────────────────────────────────────────────╯
Exiting with non-zero code due to validation failure
>
> # Make use of --exit-code in GitHub Actions or any other automation platform
> # Exit codes: 0 = success, 1 = validation failed
```
