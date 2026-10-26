# Skills

A skill is a package of structured files that teaches an AI coding agent how to work with a specific tool or framework. This project ships multiple skills -- use the switcher below to browse each one. Install a skill in your agent and it will be able to run commands, edit configuration, write content, and troubleshoot problems without step-by-step guidance from you.

▶ Install these skills

**Any agent** -- install all with [npx](https://github.com/vercel-labs/skills):

``` bash
npx skills add https://posit-dev.github.io/pointblank/
```

**CLI** -- install all skills in a project:

``` bash
great-docs skill install pointblank
```

**Codex / OpenCode**

``` text
Tell the agent to fetch these skill files:
https://posit-dev.github.io/pointblank/.well-known/agent-skills/pointblank/SKILL.md
https://posit-dev.github.io/pointblank/.well-known/agent-skills/write-validation/SKILL.md
https://posit-dev.github.io/pointblank/.well-known/agent-skills/validate-yaml/SKILL.md
https://posit-dev.github.io/pointblank/.well-known/agent-skills/draft-validation/SKILL.md
https://posit-dev.github.io/pointblank/.well-known/agent-skills/define-contracts/SKILL.md
https://posit-dev.github.io/pointblank/.well-known/agent-skills/scan-and-profile/SKILL.md
https://posit-dev.github.io/pointblank/.well-known/agent-skills/generate-data/SKILL.md
```

Or browse the skill files below.

pointblank

write-validation

validate-yaml

draft-validation

define-contracts

scan-and-profile

generate-data

### SKILL LAYOUT

``` gd-skills-tree

pointblank/
├── SKILL.md
└── references/
    ├── column-selectors.md
    ├── data-backends.md
    └── validation-methods.md
```

### SKILL.md

``` gd-skills-raw
```

```` markdown
---
name: pointblank
description: >
  Validate DataFrames and database tables with Pointblank. Covers the
  Validate workflow (plan, interrogate, report), column selectors, data
  backends (Polars, Pandas, DuckDB, databases via Ibis), threshold
  levels, actions, and result extraction. Use when building, running,
  or troubleshooting data-validation pipelines.
license: MIT
compatibility: Requires Python >=3.10.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - data-validation
    - data-quality
    - polars
    - pandas
    - duckdb
    - ibis
---

# Pointblank

A data-validation library for Python. Define validation steps against a
table, execute them with `interrogate()`, then inspect results as a
rich tabular report or programmatically via pass/fail counts and
data extracts.

## Quick start

```python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset("small_table"))
    .col_vals_gt(columns="d", value=100)
    .col_vals_not_null(columns="date_time")
    .col_vals_in_set(columns="f", set=["low", "mid", "high"])
    .interrogate()
)

validation  # displays the HTML validation report
```

## Skill directory structure

This skill ships with companion files for agent consumption:

```
skills/pointblank/
+-- SKILL.md                    <- This file
+-- references/
    +-- validation-methods.md   <- All col_vals_* / row / schema methods
    +-- column-selectors.md     <- col(), starts_with(), matches(), etc.
    +-- data-backends.md        <- Polars, Pandas, DuckDB, Ibis, files
```

## When to use what

| I want to...                         | Use                                  |
| ------------------------------------ | ------------------------------------ |
| Check column values meet a condition | `col_vals_gt/lt/eq/...`              |
| Ensure no nulls in a column          | `col_vals_not_null`                  |
| Check values are in an allowed set   | `col_vals_in_set`                    |
| Match a regex pattern                | `col_vals_regex`                     |
| Validate table schema                | `col_schema_match`                   |
| Check row/column counts              | `row_count_match`, `col_count_match` |
| Find duplicate rows                  | `rows_distinct`                      |
| Check data freshness                 | `data_freshness`                     |
| Combine multiple conditions          | `conjointly`                         |
| Run a custom check                   | `specially`                          |
| Use LLM-based validation             | `prompt`                             |
| Select columns by pattern            | `starts_with`, `contains`, `matches` |
| Set failure thresholds               | `Thresholds`                         |
| Trigger actions on failure           | `Actions`, `FinalActions`            |
| Get failing rows                     | `get_data_extracts`                  |
| Split data into pass/fail            | `get_sundered_data`                  |
| Profile a dataset first              | `DataScan`                           |
| Compare two tables (diff)            | `compare`, `tbl_match(keys=...)`     |
| Define validation in YAML            | `yaml_interrogate`                   |
| Enforce contracts in a pipeline      | `Contract`, `Pipeline`               |
| Generate test data                   | `Schema.generate`, field classes     |
| Draft validation with an LLM         | `DraftValidation`                    |

## Core concepts

### The Validate workflow

Every validation follows three phases:

1. **Plan** -- Create a `Validate` object with a data source and chain
   validation methods to define steps.
2. **Interrogate** -- Call `.interrogate()` to execute all steps against
   the data.
3. **Report** -- View results with the built-in HTML report (just
   evaluate the object), or extract results programmatically.

```python
import pointblank as pb

validation = (
    pb.Validate(data=df, tbl_name="orders", label="Daily order check")
    .col_vals_gt(columns="amount", value=0)
    .col_vals_not_null(columns="customer_id")
    .col_vals_between(columns="quantity", left=1, right=1000)
    .interrogate()
)
```

### Data backends

Pointblank works with multiple table types through the same API:

| Backend     | How to supply data                                                  |
| ----------- | ------------------------------------------------------------------- |
| Polars      | `pl.DataFrame` or `pl.LazyFrame`                                    |
| Pandas      | `pd.DataFrame`                                                      |
| DuckDB      | `ibis.Table` via `pb.connect_to_table("duckdb://path.db::table")`   |
| PostgreSQL  | `ibis.Table` via `pb.connect_to_table("postgresql://...")`          |
| MySQL       | `ibis.Table` via `pb.connect_to_table("mysql://...")`               |
| SQLite      | `ibis.Table` via `pb.connect_to_table("sqlite://...")`              |
| Snowflake   | `ibis.Table` via `pb.connect_to_table("snowflake://...")`           |
| CSV/Parquet | File path string: `"data/orders.csv"`, `"s3://bucket/file.parquet"` |

```python
tbl = pb.connect_to_table("duckdb:///warehouse.db::sales")
validation = pb.Validate(data=tbl).col_vals_gt(columns="revenue", value=0).interrogate()
```

### Column selectors

Instead of naming columns one by one, use selectors to target groups:

```python
from pointblank import col, starts_with, ends_with, contains, matches, everything

# All columns starting with "price"
.col_vals_gt(columns=starts_with("price"), value=0)

# Combine selectors with operators
.col_vals_not_null(columns=starts_with("id") | ends_with("_key"))

# Exclude columns
.col_vals_not_null(columns=everything() - matches("_tmp$"))
```

Selectors: `col()`, `starts_with()`, `ends_with()`, `contains()`,
`matches()`, `everything()`, `first_n()`, `last_n()`.

Operators: `&` (and), `|` (or), `-` (difference), `~` (not).

### Thresholds and actions

Set failure thresholds at three severity levels:

```python
validation = (
    pb.Validate(
        data=df,
        thresholds=pb.Thresholds(warning=0.05, error=0.10, critical=0.25),
        actions=pb.Actions(
            warning="Step {step}: {col} has warnings",
            critical=pb.send_slack_notification(webhook_url="..."),
        ),
    )
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)
```

Threshold values: `<1` = fraction of failing rows, `>=1` = absolute
count, `True` = any failure (equivalent to 1).

Per-step thresholds override the global setting.

### Extracting results

```python
validation.all_passed()          # True if every step passed
validation.n_passed(i=1)         # count of passing units in step 1
validation.f_failed(i=2)         # fraction of failing units in step 2

# Get the rows that failed step 1
extracts = validation.get_data_extracts(i=1, frame=True)

# Split data into passing and failing subsets
pass_df = validation.get_sundered_data(type="pass")
fail_df = validation.get_sundered_data(type="fail")

# JSON report for downstream systems
json_str = validation.get_json_report()
```

### Reporting

```python
# Full HTML report (default display)
validation.get_tabular_report()

# Per-step detail report
validation.get_step_report(i=1, limit=20)

# Customize report sections
validation.get_tabular_report(
    title="Nightly Checks",
    incl_header=True,
    incl_footer=True,
    incl_footer_timings=True,
)
```

### Serialization

Save and reload validation objects for auditing or scheduling:

```python
pb.write_file(validation, filename="nightly_check.pb")
restored = pb.read_file("nightly_check.pb")
```

## Related skills

| Skill            | When to use it                                    |
| ---------------- | ------------------------------------------------- |
| write-validation | Detailed guidance on choosing and composing steps |
| define-contracts | Contract and Pipeline boundary validation         |
| scan-and-profile | Profile data before writing validation rules      |
| validate-yaml    | Define validation plans in YAML                   |
| generate-data    | Create synthetic test data from schemas           |
| draft-validation | LLM-assisted validation drafting and editing      |

## Gotchas

1. **Call `.interrogate()` last.** Validation methods only define steps;
   nothing executes until `interrogate()` is called.
2. **Column selectors are case-insensitive by default.** Pass
   `case_sensitive=True` to `starts_with()`, `contains()`, etc. if
   needed.
3. **Threshold fractions vs counts.** A threshold of `0.05` means 5% of
   rows may fail; a threshold of `5` means at most 5 rows may fail.
4. **`na_pass=False` is the default.** Null values count as failures
   unless you set `na_pass=True`.
5. **Database tables require Ibis.** Use `pb.connect_to_table()` with a
   connection string to get an Ibis table object.
6. **File paths work directly.** Pass `"data.csv"` or `"data.parquet"`
   as `data=` and Pointblank reads it automatically.
7. **Reports render as HTML.** In notebooks, just evaluate the Validate
   object. In scripts, call `get_tabular_report()` explicitly.
````

### references/column-selectors.md

``` gd-skills-raw
```

```` markdown
# Column selectors reference

Column selectors let you target multiple columns by pattern instead
of listing them individually.

## Available selectors

| Selector            | Description                              | Example                       |
|---------------------|------------------------------------------|-------------------------------|
| `col("name")`       | Explicit column name                    | `col("revenue")`             |
| `starts_with(text)` | Columns whose name starts with text     | `starts_with("price")`       |
| `ends_with(text)`   | Columns whose name ends with text       | `ends_with("_id")`           |
| `contains(text)`    | Columns whose name contains text        | `contains("amount")`         |
| `matches(pattern)`  | Columns matching a regex                | `matches(r"^col_\d+")`       |
| `everything()`      | All columns                             | `everything()`               |
| `first_n(n)`        | First n columns                         | `first_n(3)`                 |
| `last_n(n)`         | Last n columns                          | `last_n(2)`                  |

## Parameters

All text-based selectors accept `case_sensitive: bool = False`.

`first_n` and `last_n` accept `offset: int = 0` to skip columns
before counting.

## Combining selectors with operators

| Operator | Meaning    | Example                                  |
|----------|------------|------------------------------------------|
| `\|`     | Union      | `starts_with("a") \| starts_with("b")`   |
| `&`      | Intersect  | `contains("price") & ends_with("_usd")`  |
| `-`      | Difference | `everything() - matches("_tmp$")`        |
| `~`      | Negate     | `~contains("debug")`                     |

## Expression columns (for conjointly)

`expr_col()` creates column expressions for use in `conjointly()`:

```python
from pointblank import expr_col

.conjointly(
    lambda v: v.col_vals_expr(expr_col("start") < expr_col("end")),
    lambda v: v.col_vals_gt(columns="duration", value=0),
)
```

`expr_col` supports: `>`, `<`, `==`, `!=`, `>=`, `<=`, `+`, `-`,
`*`, `/`, `is_null()`, `is_not_null()`, `&`, `|`.

## Reference columns

`ref()` references a column in the reference table for aggregate
comparisons:

```python
validation = (
    pb.Validate(data=current_df, reference=previous_df)
    .col_sum_eq(columns="revenue", value=ref("revenue"))
    .interrogate()
)
```

When `value=None` in aggregate methods and a reference table is set,
`ref(column)` is used automatically.

## Usage in validation methods

Selectors work in any method that accepts `columns`:

```python
import pointblank as pb
from pointblank import starts_with, ends_with, everything

(
    pb.Validate(data=df)
    .col_vals_gt(columns=starts_with("price"), value=0)
    .col_vals_not_null(columns=everything() - ends_with("_notes"))
    .col_vals_in_set(columns="status", set=["active", "inactive"])
    .interrogate()
)
```
````

### references/data-backends.md

``` gd-skills-raw
```

```` markdown
# Data backends reference

Pointblank validates tables from multiple backends through a
unified API. The validation methods work identically regardless
of the backend.

## Supported backends

| Backend       | Input type              | Extra dependency     |
|---------------|-------------------------|----------------------|
| Polars        | `pl.DataFrame`, `pl.LazyFrame` | `polars`       |
| Pandas        | `pd.DataFrame`          | `pandas`             |
| DuckDB        | `ibis.Table`            | `ibis-framework[duckdb]` |
| PostgreSQL    | `ibis.Table`            | `ibis-framework[postgres]` |
| MySQL         | `ibis.Table`            | `ibis-framework[mysql]` |
| SQLite        | `ibis.Table`            | `ibis-framework[sqlite]` |
| Snowflake     | `ibis.Table`            | `ibis-framework[snowflake]` |
| CSV files     | File path string        | (none)               |
| Parquet files | File path string        | (none)               |

## Connecting to databases

Use `pb.connect_to_table()` with a connection string. Append
`::table_name` to specify the table:

```python
import pointblank as pb

# DuckDB
tbl = pb.connect_to_table("duckdb:///warehouse.db::sales")

# PostgreSQL
tbl = pb.connect_to_table("postgresql://user:pass@host:5432/db::orders")

# MySQL
tbl = pb.connect_to_table("mysql://user:pass@host:3306/db::customers")

# SQLite
tbl = pb.connect_to_table("sqlite:///local.db::events")

# Snowflake
tbl = pb.connect_to_table("snowflake://user:pass@account/db/schema::table")
```

## Using file paths directly

Pass CSV or Parquet paths as the `data` argument:

```python
validation = (
    pb.Validate(data="data/orders.csv")
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)

# Parquet files
validation = (
    pb.Validate(data="s3://bucket/data.parquet")
    .col_vals_not_null(columns="id")
    .interrogate()
)
```

## Built-in datasets

Pointblank includes example datasets for testing:

```python
df = pb.load_dataset("small_table")               # default: Polars
df = pb.load_dataset("game_revenue", tbl_type="pandas")
df = pb.load_dataset("nycflights", tbl_type="duckdb")
df = pb.load_dataset("global_sales", tbl_type="polars")
```

Available datasets: `small_table`, `game_revenue`, `nycflights`,
`global_sales`.

Available types: `"polars"` (default), `"pandas"`, `"duckdb"`.

## Listing database tables

```python
pb.print_database_tables("duckdb:///warehouse.db")
```

## Utility functions

```python
pb.get_row_count(df)      # row count for any backend
pb.get_column_count(df)   # column count for any backend
pb.preview(df)            # quick visual preview as GT table
```
````

### references/validation-methods.md

``` gd-skills-raw
```

```` markdown
# Validation methods reference

Complete list of validation methods available on `Validate`.

## Value comparison methods

All share a common signature pattern:

```python
.col_vals_gt(
    columns,              # str, list[str], or column selector
    value,                # numeric value or col("other_column")
    na_pass=False,        # treat nulls as passing?
    missing=None,         # MissingSpec for structured missingness
    pre=None,             # callable to transform data before check
    segments=None,        # segment the check by group values
    thresholds=None,      # per-step thresholds override
    actions=None,         # per-step actions override
    brief=None,           # custom brief text or False to suppress
    active=True,          # bool or callable to conditionally skip
    dimension=None,       # tag for dimensional scoring
)
```

| Method              | Checks that values are...            |
|---------------------|--------------------------------------|
| `col_vals_gt`       | greater than `value`                 |
| `col_vals_lt`       | less than `value`                    |
| `col_vals_ge`       | greater than or equal to `value`     |
| `col_vals_le`       | less than or equal to `value`        |
| `col_vals_eq`       | equal to `value`                     |
| `col_vals_ne`       | not equal to `value`                 |

## Range methods

```python
.col_vals_between(columns, left, right, inclusive=(True, True), ...)
.col_vals_outside(columns, left, right, inclusive=(True, True), ...)
```

`inclusive` controls boundary inclusion: `(True, True)` = closed
interval, `(False, False)` = open interval.

## Set membership

```python
.col_vals_in_set(columns, set=["a", "b", "c"], ...)
.col_vals_not_in_set(columns, set=["x", "y"], ...)
```

## Monotonicity

```python
.col_vals_increasing(columns, allow_stationary=False, decreasing_tol=None, ...)
.col_vals_decreasing(columns, allow_stationary=False, increasing_tol=None, ...)
```

## Null checks

```python
.col_vals_null(columns, ...)      # all values must be null
.col_vals_not_null(columns, ...)  # no values may be null
```

No `na_pass` or `missing` parameters on these methods.

## Pattern and spec matching

```python
.col_vals_regex(columns, pattern="^[A-Z]{3}$", inverse=False, ...)
.col_vals_within_spec(columns, spec="email", ...)
```

## Expression-based

```python
from pointblank import expr_col

.col_vals_expr(expr_col("price") * expr_col("qty") > 0, ...)
```

## Aggregate comparison methods

Pattern: `col_{agg}_{comp}()` where agg is `sum`, `avg`, or `sd`
and comp is `eq`, `gt`, `ge`, `lt`, or `le`.

```python
.col_sum_gt(columns, value=1000, tol=0, ...)
.col_avg_between(columns, value=50.0, tol=0.5, ...)
.col_sd_lt(columns, value=10, ...)
```

When a reference table is set and `value=None`, automatically
compares against the reference column.

## Null percentage

```python
.col_pct_null(columns, p=0.05, tol=0, ...)     # null % <= p
.col_pct_missing(columns, missing=spec, max_pct=0.10, ...)
```

## Structural checks

```python
.col_exists(columns, ...)
.col_schema_match(schema, complete=True, in_order=True, ...)
.col_count_match(count=10, inverse=False, ...)
.row_count_match(count=1000, tol=0, inverse=False, ...)
.rows_distinct(columns_subset=None, ...)
.rows_complete(columns_subset=None, ...)
```

## Data freshness

```python
.data_freshness(column="updated_at", max_age="2h", reference_time=None, ...)
```

`max_age` accepts strings like `"2h"`, `"30m"`, `"1d"`, or
`datetime.timedelta` objects.

## Table comparison

```python
.tbl_match(tbl_compare=other_df, ...)
```

## Compound and custom checks

```python
# Multiple conditions must all hold for each row
.conjointly(
    lambda v: v.col_vals_gt(columns="a", value=0),
    lambda v: v.col_vals_lt(columns="a", value=100),
)

# Fully custom check
.specially(lambda tbl: len(tbl) > 0)
```

## LLM-based validation

```python
.prompt(
    prompt="Check if product names are appropriate",
    model="anthropic:claude-sonnet-4-6",
    columns_subset=["product_name"],
    batch_size=1000,
    max_concurrent=3,
)
```

## Common parameters

| Parameter    | Type                    | Description                              |
|--------------|-------------------------|------------------------------------------|
| `columns`    | `str \| list \| selector` | Target column(s)                       |
| `value`      | `numeric \| col()`      | Comparison value or column reference     |
| `na_pass`    | `bool`                  | Treat nulls as passing (default `False`) |
| `missing`    | `MissingSpec`           | Structured missingness definition        |
| `pre`        | `Callable`              | Transform data before check              |
| `segments`   | `SegmentSpec`           | Segment validation by groups             |
| `thresholds` | `Thresholds`            | Per-step failure thresholds              |
| `actions`    | `Actions`               | Per-step actions on threshold breach     |
| `brief`      | `str \| bool`           | Step description text                    |
| `active`     | `bool \| Callable`      | Conditionally skip step                  |
| `dimension`  | `str`                   | Tag for dimensional scoring              |
````

### SKILL LAYOUT

``` gd-skills-tree

write-validation/
├── SKILL.md
└── references/
    ├── step-selection-guide.md
    └── thresholds-and-actions.md
```

### SKILL.md

``` gd-skills-raw
```

```` markdown
---
name: write-validation
description: >
  Write data-validation plans with Pointblank. Covers choosing the
  right validation methods for each data quality concern, composing
  multi-step plans, setting thresholds and actions, using segments,
  conditional steps, handling nulls and missing values, and extracting
  results. Use when building or improving a validation workflow.
license: MIT
compatibility: Requires Python >=3.10, pointblank installed.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - data-validation
    - data-quality
    - validation-plan
    - thresholds
    - actions
---

# Write Validation

Skill for composing effective data-validation plans with Pointblank.
A validation plan is a sequence of steps that check different aspects
of your data, organized to catch issues early and report clearly.

## Quick start

```python
import pointblank as pb

validation = (
    pb.Validate(
        data=df,
        tbl_name="orders",
        label="Order validation",
        thresholds=pb.Thresholds(warning=0.01, error=0.05),
    )
    # Structural checks first
    .col_exists(columns=["order_id", "amount", "status"])
    .row_count_match(count=1000, tol=50)

    # Value checks
    .col_vals_gt(columns="amount", value=0)
    .col_vals_not_null(columns="order_id")
    .col_vals_in_set(columns="status", set=["pending", "shipped", "delivered"])

    # Execute
    .interrogate()
)
```

## Skill directory structure

```
skills/write-validation/
+-- SKILL.md                       <- This file
+-- references/
    +-- step-selection-guide.md    <- Which method for which concern
    +-- thresholds-and-actions.md  <- Configuring failure responses
```

## When to use what

| Data quality concern              | Recommended method                   |
| --------------------------------- | ------------------------------------ |
| Column exists                     | `col_exists`                         |
| Schema matches expectations       | `col_schema_match`                   |
| Expected row count                | `row_count_match`                    |
| Expected column count             | `col_count_match`                    |
| No null values                    | `col_vals_not_null`                  |
| All null (placeholder column)     | `col_vals_null`                      |
| No duplicate rows                 | `rows_distinct`                      |
| All rows complete (no nulls)      | `rows_complete`                      |
| Values above/below a bound        | `col_vals_gt/lt/ge/le`               |
| Values in a numeric range         | `col_vals_between`                   |
| Values outside a range            | `col_vals_outside`                   |
| Values from an allowed set        | `col_vals_in_set`                    |
| Values not in a forbidden set     | `col_vals_not_in_set`                |
| Values match a pattern            | `col_vals_regex`                     |
| Values match a format (email etc) | `col_vals_within_spec`               |
| Values are monotonically ordered  | `col_vals_increasing/decreasing`     |
| Null percentage within budget     | `col_pct_null`                       |
| Structured missingness            | `col_pct_missing` with `MissingSpec` |
| Aggregate comparison (sum, avg)   | `col_sum_eq`, `col_avg_gt`, etc.     |
| Data is recent enough             | `data_freshness`                     |
| Table matches another table       | `tbl_match` (`keys=`, `tolerance=`)  |
| Multiple conditions per row       | `conjointly`                         |
| Custom logic                      | `specially`                          |
| LLM-based semantic check          | `prompt`                             |

## Core concepts

### Ordering your steps

Organize validation steps from structural to semantic:

1. **Schema checks** -- `col_exists`, `col_schema_match`,
   `col_count_match`, `row_count_match`
2. **Completeness checks** -- `col_vals_not_null`, `rows_complete`,
   `col_pct_null`
3. **Uniqueness checks** -- `rows_distinct`
4. **Value-range checks** -- `col_vals_gt`, `col_vals_between`, etc.
5. **Format checks** -- `col_vals_regex`, `col_vals_within_spec`
6. **Set membership** -- `col_vals_in_set`, `col_vals_not_in_set`
7. **Cross-column checks** -- `conjointly`, `col_vals_expr`
8. **Aggregate checks** -- `col_sum_eq`, `col_avg_gt`
9. **Freshness checks** -- `data_freshness`
10. **Custom/semantic checks** -- `specially`, `prompt`

This order means structural problems surface first before
value-level checks run.

### Handling nulls

By default, null values count as failures in value checks. Control
this per step:

```python
# Nulls count as failures (default)
.col_vals_gt(columns="amount", value=0)

# Nulls are treated as passing
.col_vals_gt(columns="amount", value=0, na_pass=True)
```

For structured missingness (sentinel values like -999, "N/A"):

```python
missing_spec = pb.MissingSpec(
    reasons={-999: "not collected", -1: "redacted"},
    null_is_missing=True,
    null_reason="unknown",
)

.col_vals_gt(columns="measurement", value=0, missing=missing_spec)
.col_pct_missing(columns="measurement", missing=missing_spec, max_pct=0.10)
```

### Segmented validation

Break validation into groups to see which segments fail:

```python
.col_vals_gt(
    columns="revenue",
    value=0,
    segments=pb.seg_group(["region_a", "region_b", "region_c"]),
)
```

### Conditional steps

Skip steps dynamically based on the data:

```python
# Only run if the column exists
.col_vals_gt(
    columns="new_feature",
    value=0,
    active=pb.has_columns("new_feature"),
)

# Only run if the table has enough rows
.rows_distinct(active=pb.has_rows(min=100))
```

### Pre-processing data

Transform data before a check with `pre`:

```python
.col_vals_gt(
    columns="price",
    value=0,
    pre=lambda df: df.filter(pl.col("status") == "active"),
)
```

### Thresholds

Set thresholds at three severity levels:

```python
# Global thresholds (apply to all steps)
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=0.01, error=0.05, critical=0.25),
)

# Per-step override
.col_vals_gt(
    columns="amount",
    value=0,
    thresholds=pb.Thresholds(warning=5, error=20),
)
```

- Values `< 1`: fraction of failing test units (e.g., `0.05` = 5%)
- Values `>= 1`: absolute count of failures (e.g., `5` = 5 rows)
- `True`: any failure triggers (equivalent to `1`)

### Actions

Trigger responses when thresholds are exceeded:

```python
pb.Actions(
    warning="Warning: {col} step {step} at {time}",
    error=lambda: send_alert("Data quality error"),
    critical=[
        pb.send_slack_notification(webhook_url="https://..."),
        "Critical failure in {col}",
    ],
    highest_only=True,  # only fire the highest triggered level
)
```

Template variables: `{type}`, `{level}`, `{step}`, `{col}`,
`{val}`, `{time}`.

### Final actions

Run after all steps complete, with access to the full summary:

```python
def check_overall(summary=None):
    summary = pb.get_validation_summary()
    if summary and summary["n_failed_steps"] > 0:
        send_report(summary)

pb.Validate(
    data=df,
    final_actions=pb.FinalActions(check_overall),
)
```

### Comparing against reference data

Track drift by comparing current data to a reference table:

```python
validation = (
    pb.Validate(data=current_df, reference=previous_df)
    .col_sum_eq(columns="revenue")    # value=None -> uses ref()
    .col_avg_eq(columns="quantity")
    .interrogate()
)
```

### Extracting results

```python
# Did everything pass?
validation.all_passed()

# Counts and fractions by step
validation.n_passed(i=1, scalar=True)
validation.f_failed(i=[1, 2, 3])

# Get failing rows for a step
extracts = validation.get_data_extracts(i=1, frame=True)

# Split into pass/fail subsets
pass_df = validation.get_sundered_data(type="pass")
fail_df = validation.get_sundered_data(type="fail")

# Machine-readable report
json_report = validation.get_json_report()
```

### Saving and reloading

```python
pb.write_file(validation, filename="daily_check.pb")
restored = pb.read_file("daily_check.pb")
```

## Workflows

### Building a validation plan from scratch

1. Profile the data with `pb.DataScan(data=df)` to understand
   distributions, nulls, and types.
2. Start with structural checks (`col_exists`, `col_schema_match`).
3. Add completeness checks (`col_vals_not_null`, `rows_complete`).
4. Add value constraints based on domain knowledge.
5. Set thresholds appropriate to the use case.
6. Run `interrogate()` and review the report.
7. Iterate: adjust thresholds, add missing checks, remove noisy ones.

### Adding checks to an existing plan

Read the current validation code, identify uncovered columns or
concerns, and add steps in the appropriate position (structural
before semantic). Preserve the existing threshold/action
configuration unless changing it is part of the task.

### Diagnosing validation failures

1. Run `interrogate()` and check the report.
2. Use `get_data_extracts(i=N, frame=True)` to see failing rows.
3. Use `get_step_report(i=N)` for a detailed per-step view.
4. Determine whether the check or the data is wrong.
5. Fix the check (adjust value/threshold) or flag the data issue.

## Gotchas

1. **`.interrogate()` must be called.** Steps are declarative until
   executed.
2. **`na_pass` defaults to `False`.** Nulls fail value checks unless
   you opt in.
3. **Threshold `0.05` vs `5`.** Fractional = percentage, integer =
   count.
4. **`col_vals_between` is closed by default.** Pass
   `inclusive=(False, False)` for an open interval.
5. **`conjointly` takes lambdas, not method calls.** Each argument is
   `lambda v: v.col_vals_gt(...)`.
6. **Aggregate methods compare one value per column**, not per row.
   They produce a single pass/fail per column.
7. **`segments` splits the check into sub-groups.** Each segment is
   reported separately in the validation report.
````

### references/step-selection-guide.md

``` gd-skills-raw
```

```` markdown
# Step selection guide

Decision tree for choosing the right validation method.

## By data type

### Numeric columns

| Concern                     | Method                                  |
|-----------------------------|-----------------------------------------|
| Positive values only        | `col_vals_gt(columns, value=0)`         |
| Within a range              | `col_vals_between(columns, left, right)`|
| Not zero                    | `col_vals_ne(columns, value=0)`         |
| Sum matches expected        | `col_sum_eq(columns, value=total)`      |
| Average within tolerance    | `col_avg_between(columns, value, tol)`  |
| Standard deviation bounded  | `col_sd_lt(columns, value=max_sd)`      |
| Monotonically increasing    | `col_vals_increasing(columns)`          |
| Null percentage under limit | `col_pct_null(columns, p=0.05)`         |

### String columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| Matches a pattern           | `col_vals_regex(columns, pattern)`        |
| Valid email/URL/phone/etc.  | `col_vals_within_spec(columns, spec)`     |
| From a known set            | `col_vals_in_set(columns, set)`           |
| Not a forbidden value       | `col_vals_not_in_set(columns, set)`       |
| Not null                    | `col_vals_not_null(columns)`              |

### Date/datetime columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| After a cutoff date         | `col_vals_gt(columns, value=cutoff)`      |
| Within a date range         | `col_vals_between(columns, left, right)`  |
| Data is recent              | `data_freshness(column, max_age="2h")`    |
| Chronologically ordered     | `col_vals_increasing(columns)`            |

### Boolean columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| All true                    | `col_vals_eq(columns, value=True)`        |
| All false                   | `col_vals_eq(columns, value=False)`       |

## By concern type

### Completeness

```python
.col_vals_not_null(columns="required_field")
.rows_complete()                          # no nulls in any column
.rows_complete(columns_subset=["a", "b"]) # no nulls in a, b
.col_pct_null(columns="optional", p=0.20) # at most 20% null
```

### Uniqueness

```python
.rows_distinct()                             # all rows unique
.rows_distinct(columns_subset=["id"])        # id column unique
.rows_distinct(columns_subset=["a", "b"])    # composite unique
```

### Consistency (cross-column)

```python
from pointblank import expr_col

.conjointly(
    lambda v: v.col_vals_expr(expr_col("start") < expr_col("end")),
    lambda v: v.col_vals_gt(columns="duration", value=0),
)
```

### Referential (against another table)

```python
pb.Validate(data=current, reference=previous)
.col_sum_eq(columns="total")       # sums match reference
.col_avg_eq(columns="avg_price")   # averages match reference
.tbl_match(tbl_compare=expected)   # tables are identical
```

## Available spec values for col_vals_within_spec

Use these with `.col_vals_within_spec(columns, spec="...")`:

- `"email"` -- valid email addresses
- `"url"` -- valid URLs
- `"ipv4"` -- IPv4 addresses
- `"ipv6"` -- IPv6 addresses
- `"phone"` -- phone numbers (E.164)

## MissingSpec for structured missingness

When sentinel values represent missing data:

```python
spec = pb.MissingSpec(
    reasons={
        -999: "not collected",
        -1: "redacted",
        "N/A": "not applicable",
    },
    categories={
        "system": ["not collected"],
        "policy": ["redacted", "not applicable"],
    },
    null_is_missing=True,
    null_reason="unknown",
    description="Clinical trial data missingness codes",
)
```

Use with `missing=spec` in value checks, or standalone:

```python
.col_pct_missing(columns="lab_value", missing=spec, max_pct=0.10,
                 reason="not collected", category="system")
```
````

### references/thresholds-and-actions.md

``` gd-skills-raw
```

```` markdown
# Thresholds and actions reference

## Thresholds

`Thresholds(warning=None, error=None, critical=None)`

### Value interpretation

| Value     | Meaning                       | Example             |
|-----------|-------------------------------|----------------------|
| `0.05`    | 5% of test units may fail     | 50 failures in 1000 |
| `5`       | At most 5 test units may fail | Absolute count       |
| `True`    | Any failure triggers          | Same as `1`          |
| `None`    | Level not used                | No threshold set     |

### Setting thresholds

```python
# Global -- applies to all steps
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=0.01, error=0.05, critical=0.25),
)

# Per-step -- overrides global for that step
.col_vals_gt(
    columns="amount",
    value=0,
    thresholds=pb.Thresholds(warning=True),  # any failure warns
)

# Shorthand -- integer/float sets warning level
pb.Validate(data=df, thresholds=3)  # warning at 3 failures
```

## Actions

`Actions(warning=None, error=None, critical=None, default=None,
highest_only=True)`

Each level accepts:

- `str` -- message template (printed to stdout)
- `Callable` -- function to call
- `list[str | Callable]` -- multiple actions
- `None` -- no action

### Template variables

| Variable          | Description                      |
|-------------------|----------------------------------|
| `{type}`          | Validation method name           |
| `{level}`         | Threshold level triggered        |
| `{step}` or `{i}` | Step number                     |
| `{col}` or `{column}` | Column name                 |
| `{val}` or `{value}` | Comparison value              |
| `{time}`          | Timestamp                        |

### Examples

```python
# String messages
pb.Actions(
    warning="Step {step}: {col} has warnings at {time}",
    error="ERROR in {col}: {type} check failed",
)

# Callable actions
def alert_on_error():
    metadata = pb.get_action_metadata()
    send_email(f"Error in step {metadata['step']}")

pb.Actions(error=alert_on_error)

# Slack notifications
pb.Actions(
    critical=pb.send_slack_notification(
        webhook_url="https://hooks.slack.com/services/...",
    ),
)

# OpenTelemetry
pb.Actions(
    warning=pb.emit_otel(service_name="data-pipeline"),
)

# Multiple actions per level
pb.Actions(
    error=[
        "Error in {col}",
        lambda: log_to_database(),
        pb.send_slack_notification(webhook_url="..."),
    ],
)
```

### highest_only

When `True` (default), only fires actions for the highest triggered
level. When `False`, fires all triggered levels.

### get_action_metadata()

Inside an action callable, call `pb.get_action_metadata()` to access
step details:

```python
def my_action():
    meta = pb.get_action_metadata()
    # meta keys: step, column, type, level, value, time, ...
```

## Final actions

`FinalActions(*actions)` -- run after all steps complete.

```python
def summary_check():
    summary = pb.get_validation_summary()
    if summary["n_failed_steps"] > 0:
        create_jira_ticket(summary)

pb.Validate(
    data=df,
    final_actions=pb.FinalActions(summary_check, "Validation complete at {time}"),
)
```

### get_validation_summary()

Inside a final action callable, call `pb.get_validation_summary()`:

```python
def report():
    s = pb.get_validation_summary()
    # s keys: n_steps, n_passed_steps, n_failed_steps,
    #         warn_count, error_count, critical_count, ...
```

## Common patterns

### Warn-then-stop pipeline

```python
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=0.01, critical=0.10),
    actions=pb.Actions(
        warning="Data quality warning: {col}",
        critical=lambda: sys.exit(1),
    ),
)
```

### Slack on any failure

```python
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=True),
    actions=pb.Actions(
        warning=pb.send_slack_notification(
            webhook_url="https://hooks.slack.com/services/...",
        ),
    ),
)
```

### Log to OpenTelemetry

```python
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=0.05),
    actions=pb.Actions(
        warning=pb.emit_otel(service_name="my-pipeline"),
    ),
)
```
````

### SKILL LAYOUT

``` gd-skills-tree

validate-yaml/
├── SKILL.md
└── references/
    └── yaml-schema.md
```

### SKILL.md

``` gd-skills-raw
```

```` markdown
---
name: validate-yaml
description: >
  Define Pointblank validation plans in YAML instead of Python code.
  Covers the YAML schema, validate_yaml() for syntax checking,
  yaml_interrogate() for execution, yaml_to_python() for code
  generation, and data source configuration. Use when defining
  validation plans declaratively or sharing them across teams.
license: MIT
compatibility: Requires Python >=3.10, pointblank installed.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - data-validation
    - yaml
    - declarative
    - configuration
---

# Validate YAML

Skill for defining data-validation plans in YAML. YAML-based plans
are declarative, version-controllable, and shareable across teams
without requiring Python knowledge.

## Quick start

```yaml
# validation.yaml
tbl: "data/orders.csv"
tbl_name: orders
label: Order validation
thresholds:
  warning: 0.01
  error: 0.05

steps:
  - method: col_vals_gt
    columns: amount
    value: 0
  - method: col_vals_not_null
    columns: order_id
  - method: col_vals_in_set
    columns: status
    set: [pending, shipped, delivered]
```

```python
import pointblank as pb

# Execute the YAML plan
validation = pb.yaml_interrogate("validation.yaml")
validation.get_tabular_report()
```

## Skill directory structure

```
skills/validate-yaml/
+-- SKILL.md                    <- This file
+-- references/
    +-- yaml-schema.md          <- Full YAML key reference
```

## When to use what

| I want to...                        | Use                                |
| ----------------------------------- | ---------------------------------- |
| Check YAML syntax without running   | `validate_yaml()`                  |
| Execute a YAML validation plan      | `yaml_interrogate()`               |
| Convert YAML to Python code         | `yaml_to_python()`                 |
| Override the data source at runtime | `yaml_interrogate(set_tbl=df)`     |
| Use custom functions in YAML steps  | `yaml_interrogate(namespaces=...)` |

## Core concepts

### YAML structure

A YAML validation plan has two required keys (`tbl` and `steps`)
and several optional keys:

```yaml
# Required
tbl: "path/to/data.csv" # data source
steps: # validation steps
  - method: col_vals_gt
    columns: amount
    value: 0

# Optional metadata
tbl_name: orders
label: Daily order check
owner: data-team
consumers: [analytics, reporting]
version: "1.0"
lang: en
locale: en_US
df_library: polars # polars (default) or pandas

# Optional thresholds and actions
thresholds:
  warning: 0.01
  error: 0.05
  critical: 0.25

actions:
  warning: "Warning: {col} failed at {time}"
  error: "Error in step {step}: {col}"

final_actions:
  - "Validation complete"

# Optional brief setting
brief: true

# Optional reference table
reference: "path/to/reference.csv"

# Optional missing specs
missing_specs:
  measurement:
    reasons:
      -999: not collected
      -1: redacted
    null_is_missing: true
```

### Data sources in YAML

The `tbl` key accepts:

| Value                          | Interpreted as               |
| ------------------------------ | ---------------------------- |
| `"data.csv"`                   | CSV file path                |
| `"data.parquet"`               | Parquet file path            |
| `"duckdb:///db.ddb::table"`    | DuckDB connection string     |
| `"postgresql://...::table"`    | PostgreSQL connection string |
| `"sqlite:///db.sqlite::table"` | SQLite connection string     |

### Steps in YAML

Each step is a dictionary with `method` and the method's parameters:

```yaml
steps:
  # Value comparison
  - method: col_vals_gt
    columns: amount
    value: 0
    na_pass: true

  # Range check
  - method: col_vals_between
    columns: score
    left: 0
    right: 100
    inclusive: [true, true]

  # Set membership
  - method: col_vals_in_set
    columns: status
    set: [active, inactive, pending]

  # Pattern match
  - method: col_vals_regex
    columns: email
    pattern: ".+@.+\\..+"

  # Null check
  - method: col_vals_not_null
    columns: [id, name, email]

  # Schema match
  - method: col_schema_match
    schema:
      id: Int64
      name: String
      amount: Float64
    complete: true
    in_order: true

  # Row count
  - method: row_count_match
    count: 1000
    tol: 50

  # Structural
  - method: rows_distinct
    columns_subset: [id]

  - method: rows_complete

  # Per-step thresholds
  - method: col_vals_gt
    columns: revenue
    value: 0
    thresholds:
      warning: 5
      error: 20

  # Conditional step
  - method: col_vals_gt
    columns: new_feature
    value: 0
    active: false
```

### Validating YAML syntax

Check that a YAML file is well-formed before running:

```python
pb.validate_yaml("validation.yaml")  # raises on errors
```

### Executing a YAML plan

```python
# Run from file
validation = pb.yaml_interrogate("validation.yaml")

# Override the data source
validation = pb.yaml_interrogate("validation.yaml", set_tbl=my_df)

# Provide custom namespaces for functions
validation = pb.yaml_interrogate(
    "validation.yaml",
    namespaces={"my_module": my_module},
)
```

### Converting YAML to Python

Generate equivalent Python code from a YAML plan:

```python
python_code = pb.yaml_to_python("validation.yaml")
print(python_code)
```

Output:

```python
import pointblank as pb

validation = (
    pb.Validate(
        data="data/orders.csv",
        tbl_name="orders",
        label="Daily order check",
        thresholds=pb.Thresholds(warning=0.01, error=0.05),
    )
    .col_vals_gt(columns="amount", value=0)
    .col_vals_not_null(columns="order_id")
    .col_vals_in_set(columns="status", set=["pending", "shipped", "delivered"])
    .interrogate()
)
```

## Workflows

### Creating a YAML validation plan

1. Profile the data to understand its shape and types.
2. Write the YAML file with `tbl` and `steps`.
3. Run `pb.validate_yaml()` to check syntax.
4. Run `pb.yaml_interrogate()` to execute.
5. Review the report and iterate.

### Sharing plans across teams

1. Define the plan in YAML.
2. Commit to version control.
3. Team members execute with `pb.yaml_interrogate()`.
4. Override the data source with `set_tbl=` as needed.

### Migrating from YAML to Python

1. Run `pb.yaml_to_python("plan.yaml")` to generate code.
2. Review and customize the generated Python.
3. Add features not available in YAML (e.g., `pre` transforms,
   `specially` with custom callables).

## Gotchas

1. **Escape regex backslashes.** YAML requires `\\` for a literal
   backslash: `pattern: "\\d+"`.
2. **Lists use YAML syntax.** Write `set: [a, b, c]` or use the
   block form with `- a`.
3. **`tbl` is required in the file** but can be overridden with
   `set_tbl=` at runtime.
4. **`inclusive` is a list, not a tuple.** Write
   `inclusive: [true, true]` in YAML.
5. **Not all parameters are available.** `pre` (callable transforms),
   `specially`, and `conjointly` with lambdas require Python code.
   Use `yaml_to_python()` to migrate when you need these features.
6. **`df_library` defaults to `"polars"`.** Set to `"pandas"` if
   your downstream code expects Pandas DataFrames.
````

### references/yaml-schema.md

``` gd-skills-raw
```

```` markdown
# YAML schema reference

Complete reference for all keys in a Pointblank YAML validation
plan.

## Top-level keys

| Key              | Type                | Required | Default    | Description                        |
|------------------|---------------------|----------|------------|------------------------------------|
| `tbl`            | `str`               | yes      | --         | Data source (path or connection)   |
| `steps`          | `list[dict]`        | yes      | --         | Validation steps                   |
| `tbl_name`       | `str`               | no       | `null`     | Display name for the table         |
| `label`          | `str`               | no       | `null`     | Validation plan label              |
| `owner`          | `str`               | no       | `null`     | Data owner identifier              |
| `consumers`      | `str \| list[str]`  | no       | `null`     | Data consumers                     |
| `version`        | `str`               | no       | `null`     | Plan version                       |
| `lang`           | `str`               | no       | `null`     | Report language code               |
| `locale`         | `str`               | no       | `null`     | Locale for value formatting        |
| `df_library`     | `str`               | no       | `"polars"` | DataFrame library (`polars`/`pandas`) |
| `brief`          | `bool \| str`       | no       | `null`     | Global brief setting               |
| `reference`      | `str`               | no       | `null`     | Reference table path               |
| `thresholds`     | `dict`              | no       | `null`     | Global thresholds                  |
| `actions`        | `dict`              | no       | `null`     | Global actions                     |
| `final_actions`  | `list[str]`         | no       | `null`     | Post-validation actions            |
| `missing_specs`  | `dict`              | no       | `null`     | Named MissingSpec definitions      |

## Step keys

Each step in the `steps` list is a dictionary:

| Key           | Type                | Required | Description                     |
|---------------|---------------------|----------|---------------------------------|
| `method`      | `str`               | yes      | Validation method name          |
| `columns`     | `str \| list[str]`  | varies   | Target column(s)                |
| `value`       | `any`               | varies   | Comparison value                |
| `left`        | `number`            | varies   | Left bound (between/outside)    |
| `right`       | `number`            | varies   | Right bound (between/outside)   |
| `inclusive`    | `list[bool]`        | no       | Boundary inclusion              |
| `set`         | `list`              | varies   | Allowed/forbidden values        |
| `pattern`     | `str`               | varies   | Regex pattern                   |
| `spec`        | `str`               | varies   | Format spec (email, url, etc.)  |
| `schema`      | `dict`              | varies   | Schema definition               |
| `count`       | `int`               | varies   | Expected count                  |
| `tol`         | `number`            | no       | Tolerance for count checks      |
| `na_pass`     | `bool`              | no       | Treat nulls as passing          |
| `inverse`     | `bool`              | no       | Invert the check                |
| `complete`    | `bool`              | no       | Schema completeness             |
| `in_order`    | `bool`              | no       | Schema column ordering          |
| `columns_subset` | `list[str]`      | no       | Column subset for distinct/complete |
| `thresholds`  | `dict`              | no       | Per-step thresholds             |
| `actions`     | `dict`              | no       | Per-step actions                |
| `brief`       | `str \| bool`       | no       | Step brief text                 |
| `active`      | `bool`              | no       | Enable/disable step             |
| `dimension`   | `str`               | no       | Dimensional scoring tag         |

## Thresholds format

```yaml
thresholds:
  warning: 0.01     # fraction (< 1) or count (>= 1)
  error: 0.05
  critical: 0.25
```

## Actions format

```yaml
actions:
  warning: "Warning: {col} step {step}"
  error: "Error in {col} at {time}"
  critical: "CRITICAL: {type} failed for {col}"
```

Template variables: `{type}`, `{level}`, `{step}`, `{i}`, `{col}`,
`{column}`, `{val}`, `{value}`, `{time}`.

## Missing specs format

```yaml
missing_specs:
  column_name:
    reasons:
      -999: not collected
      -1: redacted
    categories:
      system: [not collected]
      policy: [redacted]
    null_is_missing: true
    null_reason: unknown
    description: Clinical data missingness
```

## Data source formats

```yaml
# File paths
tbl: "data/orders.csv"
tbl: "data/sales.parquet"

# Database connections (append ::table_name)
tbl: "duckdb:///warehouse.db::sales"
tbl: "postgresql://user:pass@host:5432/db::orders"
tbl: "mysql://user:pass@host:3306/db::customers"
tbl: "sqlite:///local.db::events"
```

## Complete example

```yaml
tbl: "duckdb:///warehouse.db::daily_orders"
tbl_name: daily_orders
label: "Daily order quality check"
owner: data-engineering
consumers: [analytics, finance]
version: "2.1"
lang: en
df_library: polars

thresholds:
  warning: 0.01
  error: 0.05
  critical: 0.25

actions:
  warning: "Step {step}: {col} warning at {time}"
  critical: "CRITICAL failure in {col}"

final_actions:
  - "Validation run completed"

missing_specs:
  amount:
    reasons:
      -1: refunded
    null_is_missing: true

steps:
  - method: col_schema_match
    schema:
      order_id: Int64
      customer_id: Int64
      amount: Float64
      status: String
      created_at: Datetime
    complete: true
    in_order: false

  - method: row_count_match
    count: 1000
    tol: 200

  - method: col_vals_not_null
    columns: [order_id, customer_id, status]

  - method: rows_distinct
    columns_subset: [order_id]

  - method: col_vals_gt
    columns: amount
    value: 0
    na_pass: true

  - method: col_vals_in_set
    columns: status
    set: [pending, processing, shipped, delivered, cancelled]

  - method: col_vals_regex
    columns: customer_id
    pattern: "\\d+"

  - method: col_pct_null
    columns: amount
    p: 0.05
```
````

### SKILL LAYOUT

``` gd-skills-tree

draft-validation/
├── SKILL.md
└── references/
    └── providers-reference.md
```

### SKILL.md

``` gd-skills-raw
```

```` markdown
---
name: draft-validation
description: >
  Use LLMs to draft and edit Pointblank validation plans. Covers
  DraftValidation for generating plans from data, EditValidation for
  modifying existing plans with natural language, and the interactive
  assistant() chat interface. Supports Anthropic, OpenAI, Ollama,
  Bedrock, and Azure OpenAI providers. Use when bootstrapping
  validation for a new dataset or refining existing plans.
license: MIT
compatibility: Requires Python >=3.10, pointblank installed, plus an LLM provider SDK.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - llm
    - ai-assisted
    - validation-drafting
    - code-generation
    - anthropic
    - openai
---

# Draft Validation

Skill for using LLMs to bootstrap and refine data-validation plans.
Instead of writing every check by hand, describe your data and let
an LLM generate a starting plan, then iterate with natural language
instructions.

## Quick start

```python
import pointblank as pb

# Draft a validation plan from data
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
)

# View the generated code
print(draft.code)

# Check syntax
draft.validate_syntax()
```

## Skill directory structure

```
skills/draft-validation/
+-- SKILL.md                        <- This file
+-- references/
    +-- providers-reference.md      <- LLM provider configuration
```

## When to use what

| I want to...                             | Use                       |
| ---------------------------------------- | ------------------------- |
| Generate a validation plan from data     | `DraftValidation`         |
| Edit an existing plan with instructions  | `EditValidation`          |
| Chat interactively about validation      | `assistant()`             |
| See what the LLM generated               | `draft.code`              |
| Check generated code is valid            | `draft.validate_syntax()` |
| See what changed in an edit              | `edit.diff()`             |
| Accept an edit and get a Validate object | `edit.accept()`           |

## Core concepts

### DraftValidation

Give data to an LLM and get back a validation plan:

```python
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
    api_key=None,          # uses env var by default
    max_reprompts=1,       # retries on invalid code
)
```

The LLM analyzes the data's columns, types, distributions, and
patterns to generate appropriate validation steps.

```python
# The raw LLM response
draft.response

# The extracted Python code
draft.code

# Check if the code is valid Python
draft.validate_syntax()  # True/False

# See which steps were generated
draft.changed_steps()    # list of step dicts
```

### EditValidation

Modify an existing validation plan with natural language:

```python
# From an existing Validate object
edit = pb.EditValidation(
    validation=existing_validation,
    instruction="Add a check that order_id is unique and amount is positive",
    model="anthropic:claude-sonnet-4-6",
)

# From Python code string
edit = pb.EditValidation(
    validation=code_string,
    instruction="Remove the regex check and add a between check for age",
    model="openai:gpt-6-luna",
)

# From a YAML file
edit = pb.EditValidation(
    validation="validation.yaml",
    instruction="Add threshold warnings at 5%",
    model="anthropic:claude-sonnet-4-6",
)
```

Working with edits:

```python
# See the generated code
edit.to_code()

# See what changed
edit.diff()

# See which steps were modified
edit.changed_steps()

# Accept the edit and get a Validate object
validation = edit.accept()
validation.interrogate()
```

You can supply data to the edit for context:

```python
edit = pb.EditValidation(
    validation=existing_validation,
    instruction="Add checks for the new columns",
    model="anthropic:claude-sonnet-4-6",
    data=updated_df,
)
```

### Interactive assistant

Chat with an LLM about data validation:

```python
# Browser-based chat (default)
pb.assistant(
    model="anthropic:claude-sonnet-4-6",
    data=df,
    tbl_name="orders",
)

# Terminal-based chat
pb.assistant(
    model="anthropic:claude-sonnet-4-6",
    data=df,
    display="terminal",
)
```

The assistant can:

- Suggest validation steps for your data
- Explain Pointblank concepts and methods
- Help debug validation failures
- Generate code snippets

### Model string format

All LLM features use the format `"provider:model_name"`:

```python
# Anthropic
model="anthropic:claude-sonnet-4-6"
model="anthropic:claude-haiku-5-5"

# OpenAI
model="openai:gpt-6-luna"
model="openai:gpt-5.6-sol"

# Ollama (local)
model="ollama:llama3"
model="ollama:mistral"

# AWS Bedrock
model="bedrock:anthropic.claude-sonnet-5-5"

# Azure OpenAI
model="azure-openai:my-deployment-name"
```

### API key handling

By default, the API key is read from environment variables:

| Provider     | Environment variable   |
| ------------ | ---------------------- |
| Anthropic    | `ANTHROPIC_API_KEY`    |
| OpenAI       | `OPENAI_API_KEY`       |
| Ollama       | (no key needed)        |
| Bedrock      | AWS credentials        |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` |

Or pass explicitly:

```python
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
    api_key="sk-...",
)
```

## Workflows

### Bootstrapping validation for a new dataset

1. Load your data.
2. Run `pb.DraftValidation(data=df, model="...")`.
3. Review the generated code with `draft.code`.
4. Check syntax with `draft.validate_syntax()`.
5. Copy the code into your project and customize.
6. Run `interrogate()` and iterate.

### Iterating on a validation plan

1. Start with a draft or existing validation.
2. Use `EditValidation` with natural language instructions.
3. Review changes with `edit.diff()`.
4. Accept with `edit.accept()` or iterate with another edit.

### Interactive exploration

1. Start `pb.assistant(model="...", data=df)`.
2. Ask questions about your data and validation needs.
3. Copy suggested code into your project.

## Gotchas

1. **LLM output is not guaranteed correct.** Always review generated
   code before using in production.
2. **`validate_syntax()` checks Python syntax, not semantics.** The
   code may parse but still have incorrect method calls.
3. **`max_reprompts` controls retries.** If the LLM generates invalid
   code, it will retry up to this many times.
4. **Ollama runs locally.** No API key needed but the model must be
   downloaded first with `ollama pull`.
5. **`accept()` returns an uninterrogated Validate object.** Call
   `.interrogate()` to execute.
6. **The assistant requires a running display.** `"browser"` opens a
   web interface; `"terminal"` uses the console. Neither works in
   non-interactive environments.
7. **Large tables may be sampled.** The LLM sees a profile/sample of
   the data, not every row.

## Related skills

| Skill            | When to use it                              |
| ---------------- | ------------------------------------------- |
| pointblank       | Full Validate workflow after drafting       |
| write-validation | Manual validation plan composition          |
| scan-and-profile | Profile data before asking the LLM to draft |
````

### references/providers-reference.md

``` gd-skills-raw
```

```` markdown
# LLM providers reference

Configuration for each supported LLM provider.

## Provider model strings

Format: `"provider:model_name"`

### Anthropic

```python
model="anthropic:claude-sonnet-4-6"
model="anthropic:claude-haiku-5-5"
model="anthropic:claude-opus-4-8"
```

Environment variable: `ANTHROPIC_API_KEY`

Install: `pip install "pointblank[generate]"` (includes chatlas and every provider's SDK)

### OpenAI

```python
model="openai:gpt-6-luna"
model="openai:gpt-5.6-sol"
```

Environment variable: `OPENAI_API_KEY`

Install: `pip install "pointblank[generate]"`

### Ollama (local)

```python
model="ollama:llama3"
model="ollama:mistral"
model="ollama:codellama"
```

No API key needed. Requires Ollama running locally, and the `openai` package (`pip install "pointblank[generate]"`).

Setup:
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3
```

### AWS Bedrock

```python
model="bedrock:anthropic.claude-sonnet-5-5"
model="bedrock:anthropic.claude-haiku-5-5"
```

Uses AWS credentials from the environment (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION) or AWS profiles.

Install: `pip install "pointblank[generate]"` (Bedrock uses `anthropic[bedrock]`, which brings in boto3)

### Azure OpenAI

```python
model="azure-openai:my-deployment-name"
```

Environment variables:
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_ENDPOINT`
- `OPENAI_API_VERSION` (e.g., `2024-06-01`)

Install: `pip install "pointblank[generate]"`

## API key options

### Environment variables (recommended)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

### Explicit in code

```python
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
    api_key="sk-ant-...",
)
```

### SSL verification

For environments with custom certificates:

```python
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
    verify_ssl=False,
)
```

## Feature support by function

| Feature            | DraftValidation | EditValidation | assistant() |
|--------------------|:-:|:-:|:-:|
| Anthropic          | yes | yes | yes |
| OpenAI             | yes | yes | yes |
| Ollama             | yes | yes | yes |
| Bedrock            | yes | yes | yes |
| Azure OpenAI       | yes | yes | no  |
| `api_key` param    | yes | yes | yes |
| `verify_ssl` param | yes | yes | no  |
| `max_reprompts`    | yes | yes | no  |
| Browser display    | no  | no  | yes |
| Terminal display   | no  | no  | yes |
````

### SKILL LAYOUT

``` gd-skills-tree

define-contracts/
├── SKILL.md
└── references/
    ├── contract-reference.md
    └── pipeline-reference.md
```

### SKILL.md

``` gd-skills-raw
```

```` markdown
---
name: define-contracts
description: >
  Define data contracts and pipeline validation with Pointblank.
  Covers Contract, Step, Schema, Pipeline, and PipelineResult for
  enforcing structural and semantic expectations at data boundaries.
  Use when setting up source/target contracts, pipeline validation,
  or contract serialization to YAML.
license: MIT
compatibility: Requires Python >=3.10, pointblank installed.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - data-contracts
    - pipeline-validation
    - schema
    - data-quality
---

# Define Contracts

Skill for defining data contracts that enforce expectations at the
boundaries of data pipelines. A contract declares what a dataset
must look like (schema) and what properties it must satisfy (steps),
along with metadata about ownership and violation behavior.

## Quick start

```python
import pointblank as pb

contract = pb.Contract(
    name="orders-source",
    direction="source",
    schema=pb.Schema(
        order_id="Int64",
        amount="Float64",
        status="String",
    ),
    steps=[
        pb.Step("col_vals_gt", columns="amount", value=0),
        pb.Step("col_vals_not_null", columns="order_id"),
        pb.Step("col_vals_in_set", columns="status",
                set=["pending", "shipped", "delivered"]),
    ],
    on_violation="raise",
)

# Validate data against the contract
validation = contract.validate(data=df)
```

## Skill directory structure

```
skills/define-contracts/
+-- SKILL.md                     <- This file
+-- references/
    +-- contract-reference.md    <- Contract, Step, on_violation details
    +-- pipeline-reference.md    <- Pipeline, PipelineResult details
```

## When to use what

| I want to...                          | Use                    |
| ------------------------------------- | ---------------------- |
| Declare expected table structure      | `Schema`               |
| Declare a semantic check as a step    | `Step`                 |
| Bundle schema + steps into a contract | `Contract`             |
| Validate data at pipeline ingestion   | `Pipeline` with source |
| Validate data after transformation    | `Pipeline` with target |
| Validate both source and target       | `Pipeline` with both   |
| Serialize a contract to YAML          | `contract.to_yaml()`   |
| Load a contract from YAML             | `Contract.from_yaml()` |
| Warn on violation without stopping    | `on_violation="warn"`  |
| Raise an exception on violation       | `on_violation="raise"` |
| Log violations silently               | `on_violation="log"`   |

## Core concepts

### Contract

A `Contract` bundles:

- **name** -- identifier for the contract
- **direction** -- `"source"` (incoming data) or `"target"` (output)
- **schema** -- expected column names and types
- **steps** -- list of `Step` objects defining semantic checks
- **on_violation** -- what to do when validation fails

```python
contract = pb.Contract(
    name="customer-data",
    direction="source",
    schema=pb.Schema(
        id="Int64",
        name="String",
        email="String",
        age="Int32",
    ),
    steps=[
        pb.Step("col_vals_not_null", columns="id"),
        pb.Step("col_vals_gt", columns="age", value=0),
        pb.Step("col_vals_regex", columns="email",
                pattern=r".+@.+\..+"),
        pb.Step("rows_distinct", columns_subset=["id"]),
    ],
    version="1.0",
    owner="data-team",
    consumers=["analytics", "ml-pipeline"],
    description="Customer master data contract",
    on_violation="raise",
)
```

### Step

A `Step` is a declarative representation of a validation method call:

```python
pb.Step("col_vals_gt", columns="amount", value=0)
pb.Step("col_vals_between", columns="score", left=0, right=100)
pb.Step("col_vals_in_set", columns="status", set=["a", "b", "c"])
pb.Step("col_vals_not_null", columns="id")
pb.Step("rows_distinct", columns_subset=["id"])
pb.Step("col_schema_match", schema=my_schema, complete=True)
pb.Step("row_count_match", count=1000, tol=50)
```

The `method` argument is any Validate method name. All remaining
keyword arguments are passed to that method.

### Schema

Define expected table structure:

```python
# From keyword arguments
schema = pb.Schema(id="Int64", name="String", age="Int32")

# From a dictionary
schema = pb.Schema({"id": "Int64", "name": "String"})

# From a list of tuples
schema = pb.Schema([("id", "Int64"), ("name", "String")])

# Column names only (no type checking)
schema = pb.Schema(["id", "name", "age"])

# Infer from an existing table
schema = pb.schema_from_tbl(df)
schema = pb.Schema.from_table(df, infer_constraints=True)
```

### Validating against a contract

```python
# Returns a Validate object (already interrogated)
validation = contract.validate(data=df)

# Check results
validation.all_passed()
validation.get_tabular_report()
```

Or convert to a Validate object for further customization:

```python
v = contract.to_validate(data=df)
# Add more steps if needed
v = v.col_vals_gt(columns="extra_col", value=0)
v = v.interrogate()
```

### on_violation behavior

| Value     | Behavior                             |
| --------- | ------------------------------------ |
| `"warn"`  | Print a warning message (default)    |
| `"raise"` | Raise an exception if any step fails |
| `"log"`   | Log the violation silently           |

### Pipeline

A `Pipeline` orchestrates source and target contract validation
around a data transformation:

```python
source_contract = pb.Contract(
    name="raw-orders",
    direction="source",
    schema=pb.Schema(id="Int64", amount="Float64"),
    steps=[pb.Step("col_vals_not_null", columns="id")],
    on_violation="raise",
)

target_contract = pb.Contract(
    name="clean-orders",
    direction="target",
    schema=pb.Schema(id="Int64", amount="Float64", is_valid="Boolean"),
    steps=[
        pb.Step("col_vals_gt", columns="amount", value=0),
        pb.Step("col_vals_not_null", columns="is_valid"),
    ],
    on_violation="warn",
)

pipeline = pb.Pipeline(
    source=source_contract,
    target=target_contract,
    label="Order cleaning pipeline",
    short_circuit=True,  # skip target if source fails
)

def transform(df):
    return df.with_columns(is_valid=pl.col("amount") > 0)

result = pipeline.run(data=raw_df, transform=transform)
```

### PipelineResult

```python
result.passed                 # True if both source and target passed
result.source_passed          # True if source contract passed
result.target_passed          # True if target contract passed
result.source_validation      # Validate object for source
result.target_validation      # Validate object for target
result.transform_output       # the transformed data
result.get_report()           # summary report string
```

### Serialization

```python
# Save contract to YAML
contract.to_yaml("contracts/orders-source.yaml")

# Load contract from YAML
contract = pb.Contract.from_yaml("contracts/orders-source.yaml")

# Dictionary round-trip
d = contract.to_dict()
contract = pb.Contract.from_dict(d)

# Pipeline serialization
pipeline.to_yaml("pipelines/order-cleaning.yaml")
pipeline = pb.Pipeline.from_yaml("pipelines/order-cleaning.yaml")
```

## Workflows

### Setting up a new contract

1. Profile the data with `pb.DataScan(data=df)` to understand its
   shape, types, and distributions.
2. Infer a starting schema: `schema = pb.schema_from_tbl(df)`.
3. Define steps for the semantic rules your domain requires.
4. Choose `on_violation` based on criticality.
5. Test with `contract.validate(data=df)`.
6. Serialize to YAML for version control.

### Adding contracts to an existing pipeline

1. Define source and target contracts.
2. Wrap the transformation in a `Pipeline`.
3. Use `short_circuit=True` to skip the transform when source
   validation fails.
4. Check `result.passed` to gate downstream processing.

### Evolving contracts over time

When schema or rules change:

1. Update the schema and steps in the contract YAML.
2. Bump the `version` field.
3. Test against representative data.
4. Communicate changes to `consumers`.

## Gotchas

1. **`direction` is metadata, not enforcement.** It documents intent
   but doesn't change validation behavior.
2. **`on_violation="raise"` stops execution.** Use `"warn"` or
   `"log"` when you want to continue despite failures.
3. **`short_circuit=True` skips target validation** if source
   validation fails. Set to `False` to always run both.
4. **Schema type strings are backend-specific.** Use the dtype names
   from your backend (e.g., `"Int64"` for Polars, `"int64"` for
   Pandas).
5. **`to_validate()` does not call `interrogate()`.** Call it yourself
   if you add steps. Use `validate()` for automatic interrogation.
6. **Steps reference method names as strings.** Typos in method names
   surface at validation time, not at contract creation.
````

### references/contract-reference.md

``` gd-skills-raw
```

```` markdown
# Contract reference

## Contract constructor

```python
pb.Contract(
    name: str,                                    # required
    direction: Literal["source", "target"] = "source",
    schema: Schema | None = None,
    steps: list[Step] = [],
    version: str | None = None,
    owner: str | None = None,
    consumers: str | list[str] | None = None,
    description: str | None = None,
    thresholds: Thresholds | None = None,
    on_violation: Literal["warn", "raise", "log"] = "warn",
)
```

## Contract methods

| Method                 | Returns        | Description                    |
|------------------------|----------------|--------------------------------|
| `validate(data)`       | `Validate`     | Interrogate data immediately   |
| `to_validate(data)`    | `Validate`     | Build Validate without running |
| `to_dict()`            | `dict`         | Serialize to dictionary        |
| `from_dict(cls, data)` | `Contract`     | Deserialize from dictionary    |
| `from_yaml(cls, path)` | `Contract`     | Load from YAML file            |
| `to_yaml(path=None)`   | `str \| None`  | Save to YAML (or return str)   |

## Step constructor

```python
pb.Step(method: str, **kwargs)
```

`method` is the name of any Validate method (e.g.,
`"col_vals_gt"`, `"rows_distinct"`). All other keyword arguments
are forwarded to that method.

## Step methods

| Method                 | Returns    | Description                |
|------------------------|------------|----------------------------|
| `to_dict()`            | `dict`     | Serialize to dictionary    |
| `from_dict(cls, data)` | `Step`     | Deserialize from dictionary|

## Valid step methods

- `col_vals_gt`, `col_vals_lt`, `col_vals_ge`, `col_vals_le`,
  `col_vals_eq`, `col_vals_ne`
- `col_vals_between`, `col_vals_outside`
- `col_vals_in_set`, `col_vals_not_in_set`
- `col_vals_null`, `col_vals_not_null`
- `col_vals_regex`, `col_vals_within_spec`, `col_vals_expr`
- `col_vals_increasing`, `col_vals_decreasing`
- `col_exists`, `col_schema_match`
- `col_count_match`, `row_count_match`
- `col_pct_null`, `col_pct_missing`
- `rows_distinct`, `rows_complete`
- `data_freshness`, `tbl_match`
- `conjointly`, `specially`

## YAML format

```yaml
name: orders-source
direction: source
version: "1.0"
owner: data-team
consumers:
  - analytics
  - ml-pipeline
description: Order data source contract
on_violation: raise

schema:
  order_id: Int64
  amount: Float64
  status: String

steps:
  - method: col_vals_not_null
    columns: order_id
  - method: col_vals_gt
    columns: amount
    value: 0
  - method: col_vals_in_set
    columns: status
    set: [pending, shipped, delivered]
  - method: rows_distinct
    columns_subset: [order_id]

thresholds:
  warning: 0.01
  error: 0.05
```

## on_violation behavior

| Value     | On failure...                                 |
|-----------|-----------------------------------------------|
| `"warn"`  | Prints warning to stderr, continues           |
| `"raise"` | Raises `ContractViolationError`               |
| `"log"`   | Logs via Python `logging` module              |
````

### references/pipeline-reference.md

``` gd-skills-raw
```

```` markdown
# Pipeline reference

## Pipeline constructor

```python
pb.Pipeline(
    source: Contract | None = None,
    target: Contract | None = None,
    thresholds: Thresholds | None = None,
    actions: Actions | None = None,
    final_actions: FinalActions | None = None,
    label: str | None = None,
    short_circuit: bool = True,
)
```

## Pipeline methods

| Method                   | Returns          | Description                       |
|--------------------------|------------------|-----------------------------------|
| `run(data, transform)`   | `PipelineResult` | Full pipeline: source + transform + target |
| `validate_source(data)`  | `Validate`       | Run source contract only          |
| `validate_target(data)`  | `Validate`       | Run target contract only          |
| `to_dict()`              | `dict`           | Serialize to dictionary           |
| `from_dict(cls, data)`   | `Pipeline`       | Deserialize from dictionary       |
| `from_yaml(cls, path)`   | `Pipeline`       | Load from YAML file               |
| `to_yaml(path=None)`     | `str \| None`    | Save to YAML (or return str)      |

## PipelineResult

| Attribute / Method    | Type              | Description                    |
|-----------------------|-------------------|--------------------------------|
| `source_validation`   | `Validate \| None`| Source validation result       |
| `target_validation`   | `Validate \| None`| Target validation result       |
| `transform_output`    | `Any`             | Output of the transform        |
| `source_passed`       | `bool`            | Source contract passed?        |
| `target_passed`       | `bool`            | Target contract passed?        |
| `passed`              | `bool`            | Both passed?                   |
| `get_report()`        | `str`             | Summary report text            |

## Pipeline execution flow

```
1. Validate source contract against input data
   |
   +-- If source fails and short_circuit=True -> stop, return result
   |
2. Run transform(data) -> transformed_data
   |
3. Validate target contract against transformed_data
   |
4. Return PipelineResult
```

## short_circuit behavior

| `short_circuit` | Source fails        | Source passes          |
|-----------------|---------------------|------------------------|
| `True`          | Skip transform+target| Run transform+target  |
| `False`         | Run transform+target | Run transform+target  |

## YAML format

```yaml
label: Order cleaning pipeline
short_circuit: true

source:
  name: raw-orders
  direction: source
  schema:
    id: Int64
    amount: Float64
  steps:
    - method: col_vals_not_null
      columns: id
  on_violation: raise

target:
  name: clean-orders
  direction: target
  schema:
    id: Int64
    amount: Float64
    is_valid: Boolean
  steps:
    - method: col_vals_gt
      columns: amount
      value: 0
  on_violation: warn
```

## Patterns

### Source-only pipeline

```python
pipeline = pb.Pipeline(source=source_contract)
result = pipeline.run(data=df, transform=lambda d: d)
```

### Target-only pipeline

```python
pipeline = pb.Pipeline(target=target_contract)
result = pipeline.run(data=raw_df, transform=my_transform)
```

### Accessing validation reports

```python
result = pipeline.run(data=df, transform=transform)

if not result.passed:
    if not result.source_passed:
        result.source_validation.get_tabular_report()
    if not result.target_passed:
        result.target_validation.get_tabular_report()
```
````

### SKILL LAYOUT

``` gd-skills-tree

scan-and-profile/
├── SKILL.md
└── references/
    ├── datascan-reference.md
    └── schema-inference.md
```

### SKILL.md

``` gd-skills-raw
```

```` markdown
---
name: scan-and-profile
description: >
  Profile and scan datasets with Pointblank before writing validation
  rules. Covers DataScan for column-level statistics, Schema inference
  with schema_from_tbl(), missing values analysis with missing_vals_tbl(),
  and table previewing. Use when exploring a new dataset or understanding
  data distributions before validation.
license: MIT
compatibility: Requires Python >=3.10, pointblank installed.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - data-profiling
    - data-scan
    - schema
    - data-exploration
    - missing-values
---

# Scan and Profile

Skill for profiling datasets before writing validation rules.
Understanding your data's shape, types, distributions, and
missingness patterns helps you write targeted, effective
validation plans.

## Quick start

```python
import pointblank as pb

# Scan a dataset
scan = pb.DataScan(data=df, tbl_name="orders")
scan.get_tabular_report()  # rich HTML summary
```

## Skill directory structure

```
skills/scan-and-profile/
+-- SKILL.md                      <- This file
+-- references/
    +-- datascan-reference.md     <- DataScan details and output
    +-- schema-inference.md       <- Schema inference and construction
```

## When to use what

| I want to...                     | Use                                     |
| -------------------------------- | --------------------------------------- |
| Get a full column-level profile  | `DataScan`                              |
| See column types and basic stats | `DataScan.get_tabular_report()`         |
| Export profile as JSON           | `DataScan.to_json()`                    |
| Infer a schema from data         | `schema_from_tbl()`                     |
| Infer a schema with constraints  | `Schema.from_table()`                   |
| See a quick preview of the table | `preview()`                             |
| Analyze missing values           | `missing_vals_tbl()`                    |
| Get row/column counts            | `get_row_count()`, `get_column_count()` |

## Core concepts

### DataScan

`DataScan` produces a comprehensive profile of every column in a
dataset:

```python
scan = pb.DataScan(data=df, tbl_name="monthly_sales")

# View as HTML table
scan.get_tabular_report()

# Include sample data in the report
scan.get_tabular_report(show_sample_data=True)

# Access raw summary data
scan.summary_data

# Export to JSON
json_str = scan.to_json()
scan.save_to_json("profile_output.json")
```

The report includes per-column:

- Data type
- Count of non-null values
- Missingness (count and percentage)
- Distinct value count
- Negative / zero / positive value counts (numeric)
- Descriptive statistics (mean, median, std, min, max)
- Quantiles (Q1, Q3, IQR)

### Shortcut: col_summary_tbl

For a quick column summary without creating a DataScan object:

```python
pb.col_summary_tbl(data=df, tbl_name="orders")
```

### Table preview

Quick visual preview of the first and last rows:

```python
pb.preview(data=df, n_head=5, n_tail=5)

# Customize
pb.preview(
    data=df,
    columns_subset=["id", "name", "amount"],
    n_head=10,
    n_tail=3,
    limit=50,
    show_row_numbers=True,
    max_col_width=250,
)
```

### Missing values analysis

Dedicated analysis of missingness patterns:

```python
# Basic missing values table
pb.missing_vals_tbl(data=df)

# As a heatmap
pb.missing_vals_tbl(data=df, as_heatmap=True)

# With structured missingness definitions
missing_specs = {
    "measurement": pb.MissingSpec(
        reasons={-999: "not collected", -1: "redacted"},
    ),
    "notes": pb.MissingSpec(
        reasons={"N/A": "not applicable"},
    ),
}
pb.missing_vals_tbl(data=df, missing=missing_specs)
```

### Schema inference

Infer a schema from an existing table:

```python
# Basic inference (column names and types)
schema = pb.schema_from_tbl(df)
print(schema.get_column_list())
print(schema.get_dtype_list())

# With constraint inference
schema = pb.Schema.from_table(
    df,
    infer_constraints=True,       # infer value ranges, sets, etc.
    categorical_threshold=20,     # columns with <= 20 distinct values
    detect_presets=True,           # detect email, URL, etc. patterns
    sample_size=None,             # sample rows for inference (None=all)
)
```

### Constructing schemas manually

```python
# From keyword arguments
schema = pb.Schema(id="Int64", name="String", amount="Float64")

# From a dictionary
schema = pb.Schema({"id": "Int64", "name": "String"})

# From a list of tuples
schema = pb.Schema([("id", "Int64"), ("name", "String")])

# Column names only (type checking skipped)
schema = pb.Schema(["id", "name", "amount"])
```

### Schema inspection

```python
schema.get_column_list()   # ["id", "name", "amount"]
schema.get_dtype_list()    # ["Int64", "String", "Float64"]
```

### Quick counts

```python
pb.get_row_count(df)       # number of rows
pb.get_column_count(df)    # number of columns
```

## Workflows

### Profiling a new dataset

1. Load or connect to the data.
2. Run `pb.preview(data)` for a quick look.
3. Run `pb.DataScan(data=df).get_tabular_report()` for full stats.
4. Run `pb.missing_vals_tbl(data=df)` to understand missingness.
5. Infer a schema: `schema = pb.schema_from_tbl(df)`.
6. Use the profile to inform validation rules.

### From profile to validation plan

1. Profile the data with `DataScan`.
2. Note columns with high missingness -- add `col_pct_null` checks.
3. Note columns with few distinct values -- add `col_vals_in_set`.
4. Note numeric ranges -- add `col_vals_between` checks.
5. Infer schema and use in `col_schema_match`.
6. Build the validation plan with the `write-validation` skill.

### Comparing profiles over time

```python
scan_today = pb.DataScan(data=today_df)
scan_yesterday = pb.DataScan(data=yesterday_df)

# Compare by exporting to JSON
scan_today.save_to_json("profile_today.json")
scan_yesterday.save_to_json("profile_yesterday.json")
```

## Gotchas

1. **DataScan reads the full table.** For large datasets, consider
   sampling first.
2. **Schema type names are backend-specific.** Polars uses `"Int64"`,
   Pandas uses `"int64"`. Use `schema_from_tbl()` to get the right
   names automatically.
3. **`schema_from_tbl` infers from current data.** If the data has
   unexpected types (e.g., string column with numbers), the inferred
   schema reflects that.
4. **`missing_vals_tbl` only shows null by default.** Pass
   `MissingSpec` definitions to include sentinel values.
5. **`preview()` returns a GT table object.** In notebooks it renders
   automatically; in scripts, you may need to display it.

## Related skills

| Skill            | When to use it                               |
| ---------------- | -------------------------------------------- |
| pointblank       | Full Validate workflow overview              |
| write-validation | Build validation plans from profile insights |
| generate-data    | Create synthetic data matching a schema      |
````

### references/datascan-reference.md

``` gd-skills-raw
```

```` markdown
# DataScan reference

## Constructor

```python
pb.DataScan(
    data: Any,              # DataFrame, Ibis table, or file path
    tbl_name: str | None = None,
)
```

## Properties

| Property       | Type   | Description                          |
|----------------|--------|--------------------------------------|
| `summary_data` | `dict` | Raw column-level statistics          |

## Methods

| Method                                   | Returns  | Description                     |
|------------------------------------------|----------|---------------------------------|
| `get_tabular_report(show_sample_data=False)` | `GT` | HTML summary table              |
| `to_json()`                              | `str`    | JSON string of profile          |
| `save_to_json(output_file)`              | `None`   | Write JSON to file              |

## Column statistics in summary_data

For each column, the summary includes:

| Statistic       | Description                              |
|-----------------|------------------------------------------|
| `dtype`          | Column data type                        |
| `n_non_null`     | Count of non-null values                |
| `n_null`         | Count of null values                    |
| `pct_null`       | Percentage null                         |
| `n_distinct`     | Count of distinct values                |
| `pct_distinct`   | Percentage distinct                     |
| `n_negative`     | Count of negative values (numeric)      |
| `n_zero`         | Count of zero values (numeric)          |
| `n_positive`     | Count of positive values (numeric)      |
| `mean`           | Arithmetic mean (numeric)               |
| `median`         | Median value (numeric)                  |
| `std`            | Standard deviation (numeric)            |
| `min`            | Minimum value                           |
| `max`            | Maximum value                           |
| `q1`             | First quartile (numeric)                |
| `q3`             | Third quartile (numeric)                |

## Shortcut function

```python
pb.col_summary_tbl(data=df, tbl_name="my_table")
```

Equivalent to `DataScan(data, tbl_name).get_tabular_report()`.

## missing_vals_tbl

```python
pb.missing_vals_tbl(
    data: Any,
    missing: dict[str, MissingSpec] | None = None,
    as_heatmap: bool = False,
) -> GT
```

Analyzes null and missing values across all columns. Returns a GT
table showing missingness counts and patterns.

With `as_heatmap=True`, renders a visual heatmap of missingness
across rows and columns.

## preview

```python
pb.preview(
    data: Any,
    columns_subset: list[str] | None = None,
    n_head: int = 5,
    n_tail: int = 5,
    limit: int = 50,
    show_row_numbers: bool = True,
    max_col_width: int = 250,
    min_tbl_width: int = 500,
    incl_header: bool | None = None,
) -> GT
```

## Utility functions

```python
pb.get_row_count(data: Any) -> int
pb.get_column_count(data: Any) -> int
```
````

### references/schema-inference.md

``` gd-skills-raw
```

```` markdown
# Schema inference reference

## schema_from_tbl

```python
pb.schema_from_tbl(
    tbl: Any,
    *,
    infer_constraints: bool = True,
    categorical_threshold: int = 20,
    detect_presets: bool = True,
    sample_size: int | None = None,
) -> Schema
```

Creates a Schema from an existing table with optional constraint
inference.

### Parameters

| Parameter                | Default | Description                         |
|--------------------------|---------|-------------------------------------|
| `infer_constraints`      | `True`  | Infer value ranges, allowed sets    |
| `categorical_threshold`  | `20`    | Max distinct values for categorical |
| `detect_presets`         | `True`  | Detect email, URL, phone patterns   |
| `sample_size`            | `None`  | Rows to sample (None = all)         |

## Schema.from_table

Class method with the same parameters:

```python
schema = pb.Schema.from_table(
    df,
    infer_constraints=True,
    categorical_threshold=20,
    detect_presets=True,
    sample_size=1000,
)
```

## Schema constructor

```python
# Keyword arguments
pb.Schema(id="Int64", name="String", amount="Float64")

# Dictionary
pb.Schema({"id": "Int64", "name": "String"})

# List of tuples
pb.Schema([("id", "Int64"), ("name", "String")])

# Column names only
pb.Schema(["id", "name", "amount"])

# From existing table
pb.Schema(tbl=df)
```

## Schema methods

| Method                          | Returns        | Description                  |
|---------------------------------|----------------|------------------------------|
| `get_column_list()`             | `list[str]`    | Column names                 |
| `get_dtype_list()`              | `list[str]`    | Data type strings            |
| `get_schema_coerced(to=None)`   | `Schema`       | Coerced to a target backend  |
| `generate(n=100, ...)`          | `DataFrame`    | Generate synthetic data      |

## Common dtype strings by backend

| Concept   | Polars       | Pandas       |
|-----------|-------------|--------------|
| Integer   | `Int8/16/32/64`, `UInt8/16/32/64` | `int8/16/32/64`, `uint8/16/32/64` |
| Float     | `Float32/64` | `float32/64` |
| String    | `String`     | `object`, `string` |
| Boolean   | `Boolean`    | `bool`       |
| Date      | `Date`       | `datetime64[ns]` |
| Datetime  | `Datetime`   | `datetime64[ns]` |
| Duration  | `Duration`   | `timedelta64[ns]` |

## Using inferred schema in validation

```python
schema = pb.schema_from_tbl(df)

validation = (
    pb.Validate(data=new_df)
    .col_schema_match(
        schema=schema,
        complete=True,       # all columns must be present
        in_order=True,       # column order must match
        full_match_dtypes=True,
    )
    .interrogate()
)
```

### col_schema_match parameters

| Parameter                  | Default | Description                         |
|----------------------------|---------|-------------------------------------|
| `schema`                   | required | Schema object to match against     |
| `complete`                 | `True`  | All schema columns must exist       |
| `in_order`                 | `True`  | Column order must match             |
| `case_sensitive_colnames`  | `True`  | Column name comparison              |
| `case_sensitive_dtypes`    | `True`  | Dtype string comparison             |
| `full_match_dtypes`        | `True`  | Exact dtype match required          |
````

### SKILL LAYOUT

``` gd-skills-tree

generate-data/
├── SKILL.md
└── references/
    ├── field-reference.md
    └── presets-reference.md
```

### SKILL.md

``` gd-skills-raw
```

```` markdown
---
name: generate-data
description: >
  Generate synthetic datasets with Pointblank using Schema and field
  classes. Covers IntField, FloatField, StringField, BoolField,
  DateField, DatetimeField, and more. Supports presets (name, email,
  address, etc.), country-specific data, nullable columns, unique
  constraints, and profile_fields() for person data. Use when creating
  test data, fixtures, or synthetic datasets for validation testing.
license: MIT
compatibility: Requires Python >=3.10, pointblank installed.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - data-generation
    - synthetic-data
    - test-data
    - schema
    - faker
---

# Generate Data

Skill for creating synthetic datasets from schema definitions and
field specifications. Useful for testing validation rules, creating
fixtures, generating demo data, and prototyping pipelines.

## Quick start

```python
import pointblank as pb

schema = pb.Schema(
    id=pb.int_field(min_val=1, max_val=10000, unique=True),
    name=pb.string_field(preset="name"),
    email=pb.string_field(preset="email"),
    age=pb.int_field(min_val=18, max_val=95),
    score=pb.float_field(min_val=0.0, max_val=100.0, precision=2),
    active=pb.bool_field(p_true=0.8),
)

df = schema.generate(n=1000, seed=42)
```

## Skill directory structure

```
skills/generate-data/
+-- SKILL.md                    <- This file
+-- references/
    +-- field-reference.md      <- All field types and parameters
    +-- presets-reference.md    <- Available string presets
```

## When to use what

| I want to...                        | Use                                   |
| ----------------------------------- | ------------------------------------- |
| Generate a dataset from a schema    | `schema.generate()`                   |
| Generate without creating a Schema  | `pb.generate_dataset()`               |
| Define integer columns              | `int_field()`                         |
| Define float columns                | `float_field()`                       |
| Define string columns with patterns | `string_field(pattern=)`              |
| Define string columns with presets  | `string_field(preset=)`               |
| Define boolean columns              | `bool_field()`                        |
| Define date columns                 | `date_field()`                        |
| Define datetime columns             | `datetime_field()`                    |
| Define time columns                 | `time_field()`                        |
| Define duration columns             | `duration_field()`                    |
| Add person profile fields           | `profile_fields()`                    |
| Generate country-specific data      | `generate(country="DE")`              |
| Make columns nullable               | `nullable=True, null_probability=0.1` |
| Ensure unique values                | `unique=True`                         |
| Use a custom generator function     | `generator=my_func`                   |

## Core concepts

### Schema-based generation

Define columns using field classes, then generate:

```python
schema = pb.Schema(
    order_id=pb.int_field(min_val=1, max_val=99999, unique=True),
    product=pb.string_field(allowed=["Widget A", "Widget B", "Gadget"]),
    quantity=pb.int_field(min_val=1, max_val=100),
    price=pb.float_field(min_val=0.99, max_val=999.99, precision=2),
    shipped=pb.bool_field(p_true=0.7),
    order_date=pb.date_field(min_date="2024-01-01", max_date="2024-12-31"),
)

df = schema.generate(n=500, seed=42, output="polars")
```

### generate() parameters

| Parameter  | Default    | Description                             |
| ---------- | ---------- | --------------------------------------- |
| `n`        | `100`      | Number of rows to generate              |
| `seed`     | `None`     | Random seed for reproducibility         |
| `output`   | `"polars"` | Output format: `"polars"` or `"pandas"` |
| `country`  | `"US"`     | Country code for locale-aware data      |
| `shuffle`  | `True`     | Shuffle rows after generation           |
| `weighted` | `True`     | Use weighted distributions              |

### generate_dataset() convenience function

```python
df = pb.generate_dataset(schema, n=500, seed=42)
```

### Nullable columns

Any field type supports nulls:

```python
pb.int_field(min_val=0, max_val=100, nullable=True, null_probability=0.1)
pb.string_field(preset="email", nullable=True, null_probability=0.05)
```

### Unique constraints

Ensure all generated values are distinct:

```python
pb.int_field(min_val=1, max_val=10000, unique=True)
pb.string_field(preset="email", unique=True)
```

### Allowed values (categorical)

Restrict to a specific set of values:

```python
pb.int_field(allowed=[1, 2, 3, 5, 8, 13])
pb.float_field(allowed=[0.5, 1.0, 1.5, 2.0])
pb.string_field(allowed=["low", "medium", "high"])
```

### String patterns

Generate strings matching a pattern:

```python
pb.string_field(pattern=r"[A-Z]{3}-\d{4}")    # "ABC-1234"
pb.string_field(pattern=r"INV-\d{6}")          # "INV-003847"
pb.string_field(pattern=r"[a-z]{5,10}")        # random lowercase
```

### String presets

Use built-in presets for realistic data:

```python
pb.string_field(preset="name")           # full names
pb.string_field(preset="email")          # email addresses
pb.string_field(preset="address")        # street addresses
pb.string_field(preset="city")           # city names
pb.string_field(preset="phone_number")   # phone numbers
pb.string_field(preset="company")        # company names
pb.string_field(preset="job")            # job titles
pb.string_field(preset="url")            # URLs
pb.string_field(preset="uuid4")          # UUIDs
pb.string_field(preset="iban")           # IBANs
pb.string_field(preset="ssn")            # SSNs
```

Presets produce country-specific data when `country` is set.

### Profile fields

Generate person-related fields as a group:

```python
fields = pb.profile_fields(
    set="standard",        # "standard" or "extended"
    split_name=True,       # first_name + last_name vs full name
    include=None,          # specific fields to include
    exclude=None,          # specific fields to exclude
    prefix=None,           # prefix for field names
)

schema = pb.Schema(
    id=pb.int_field(min_val=1, max_val=99999, unique=True),
    **fields,
)

df = schema.generate(n=100, country="US")
```

### Custom generators

Supply your own generator function:

```python
import random

def custom_sku():
    return f"SKU-{random.randint(1000, 9999)}"

schema = pb.Schema(
    sku=pb.string_field(generator=custom_sku),
)
```

### Country-specific generation

Over 100 countries supported:

```python
# German names, addresses, phone numbers
df = schema.generate(n=100, country="DE")

# Japanese
df = schema.generate(n=100, country="JP")

# Brazilian
df = schema.generate(n=100, country="BR")
```

## Workflows

### Creating test data for validation rules

1. Define the schema matching your production table.
2. Use field constraints to generate realistic ranges.
3. Add some nullable columns to test null handling.
4. Generate the dataset.
5. Run your validation plan against it.

```python
schema = pb.Schema(
    id=pb.int_field(min_val=1, max_val=10000, unique=True),
    amount=pb.float_field(min_val=-10, max_val=1000, precision=2),
    status=pb.string_field(allowed=["active", "inactive", "INVALID"]),
    email=pb.string_field(preset="email", nullable=True, null_probability=0.1),
)

test_df = schema.generate(n=500, seed=42)

validation = (
    pb.Validate(data=test_df)
    .col_vals_gt(columns="amount", value=0)
    .col_vals_in_set(columns="status", set=["active", "inactive"])
    .col_vals_not_null(columns="email")
    .interrogate()
)
```

### Generating fixtures from an existing table

```python
# Infer schema from real data
schema = pb.Schema.from_table(
    production_df,
    infer_constraints=True,
    categorical_threshold=20,
)

# Generate synthetic version
fixture = schema.generate(n=100, seed=1)
```

## Gotchas

1. **`unique=True` needs a large enough range.** If `max_val - min_val`
   < `n`, generation will fail for integer fields.
2. **Only one of `preset`, `pattern`, `allowed` per StringField.**
   They are mutually exclusive.
3. **Presets require `faker` to be installed.** Install with
   `pip install pointblank[faker]` or `pip install faker`.
4. **`seed` makes generation reproducible** but the same seed with
   different `n` produces different data (not a prefix of larger).
5. **`output` only supports `"polars"` and `"pandas"`.** For other
   formats, convert after generation.
6. **`null_probability=0` with `nullable=True`** generates no nulls.
   Set the probability to get actual null values.
````

### references/field-reference.md

``` gd-skills-raw
```

```` markdown
# Field reference

All field types for synthetic data generation.

## Common parameters (all field types)

| Parameter          | Type               | Default  | Description                  |
|--------------------|--------------------|----------|------------------------------|
| `nullable`         | `bool`             | `False`  | Allow null values            |
| `null_probability` | `float`            | `0.0`    | Fraction of nulls (0.0-1.0)  |
| `unique`           | `bool`             | `False`  | All values must be distinct  |
| `generator`        | `Callable \| None` | `None`   | Custom generator function    |

## IntField / int_field()

```python
pb.int_field(
    min_val: int | None = None,
    max_val: int | None = None,
    allowed: list[int] | None = None,
    dtype: str = "Int64",         # Int8/16/32/64, UInt8/16/32/64
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## FloatField / float_field()

```python
pb.float_field(
    min_val: float | None = None,
    max_val: float | None = None,
    allowed: list[float] | None = None,
    precision: int | None = None,   # decimal places
    dtype: str = "Float64",         # Float32 or Float64
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## StringField / string_field()

```python
pb.string_field(
    min_length: int | None = None,
    max_length: int | None = None,
    pattern: str | None = None,     # regex pattern to match
    preset: str | None = None,      # named preset (e.g., "email")
    allowed: list[str] | None = None,
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

Only one of `preset`, `pattern`, or `allowed` may be set.

## BoolField / bool_field()

```python
pb.bool_field(
    p_true: float = 0.5,           # probability of True
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## DateField / date_field()

```python
pb.date_field(
    min_date: str | date | None = None,    # "2024-01-01" or date()
    max_date: str | date | None = None,
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## DatetimeField / datetime_field()

```python
pb.datetime_field(
    min_date: str | datetime | None = None,
    max_date: str | datetime | None = None,
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## TimeField / time_field()

```python
pb.time_field(
    min_time: str | time | None = None,
    max_time: str | time | None = None,
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## DurationField / duration_field()

```python
pb.duration_field(
    min_duration: str | timedelta | None = None,
    max_duration: str | timedelta | None = None,
    nullable=False, null_probability=0.0, unique=False, generator=None,
)
```

## profile_fields()

```python
pb.profile_fields(
    set: str = "standard",        # "standard" or "extended"
    split_name: bool = True,      # first_name + last_name vs name
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    prefix: str | None = None,
) -> dict[str, StringField]
```

Returns a dict of StringField objects suitable for `**`-unpacking
into `Schema()`.

Standard set includes: name (or first_name + last_name), email,
phone_number, address, city, country.

Extended set adds: company, job, url, and more.
````

### references/presets-reference.md

``` gd-skills-raw
```

```` markdown
# String presets reference

Available preset values for `string_field(preset="...")`. Presets
generate realistic, locale-aware data using Faker under the hood.

## Person

| Preset          | Example output             |
|-----------------|----------------------------|
| `name`          | "John Smith"               |
| `first_name`    | "John"                     |
| `last_name`     | "Smith"                    |
| `prefix`        | "Mr."                      |
| `suffix`        | "Jr."                      |

## Contact

| Preset          | Example output             |
|-----------------|----------------------------|
| `email`         | "john.smith@example.com"   |
| `phone_number`  | "+1-555-123-4567"          |
| `url`           | "https://example.com"      |

## Address

| Preset          | Example output             |
|-----------------|----------------------------|
| `address`       | "123 Main St, Apt 4"       |
| `city`          | "New York"                 |
| `state`         | "California"               |
| `zipcode`       | "90210"                    |
| `country`       | "United States"            |
| `street_address`| "123 Main Street"          |

## Business

| Preset          | Example output             |
|-----------------|----------------------------|
| `company`       | "Acme Corporation"         |
| `job`           | "Software Engineer"        |
| `catch_phrase`  | "Innovative solutions"     |

## Internet

| Preset          | Example output             |
|-----------------|----------------------------|
| `user_name`     | "jsmith42"                 |
| `domain_name`   | "example.com"              |
| `ipv4`          | "192.168.1.1"              |
| `ipv6`          | "2001:db8::1"              |
| `mac_address`   | "00:1A:2B:3C:4D:5E"       |

## Identifiers

| Preset          | Example output             |
|-----------------|----------------------------|
| `uuid4`         | "a1b2c3d4-e5f6-..."       |
| `iban`          | "DE89 3704 0044 0532 ..."  |
| `ssn`           | "123-45-6789"              |
| `license_plate` | "ABC-1234"                 |

## Text

| Preset          | Example output             |
|-----------------|----------------------------|
| `text`          | "Lorem ipsum dolor..."     |
| `sentence`      | "The quick brown fox."     |
| `paragraph`     | "Lorem ipsum dolor sit..." |
| `word`          | "lorem"                    |

## Finance

| Preset              | Example output         |
|---------------------|------------------------|
| `credit_card_number` | "4111111111111111"     |
| `currency_code`      | "USD"                 |
| `cryptocurrency_code`| "BTC"                 |

## Country-specific behavior

Presets produce locale-appropriate data based on the `country`
parameter passed to `generate()`:

```python
schema = pb.Schema(
    name=pb.string_field(preset="name"),
    city=pb.string_field(preset="city"),
    phone=pb.string_field(preset="phone_number"),
)

# US data
us_df = schema.generate(n=100, country="US")

# German data
de_df = schema.generate(n=100, country="DE")

# Japanese data
jp_df = schema.generate(n=100, country="JP")
```

Over 100 country codes are supported, using standard ISO 3166-1
alpha-2 codes.
````
