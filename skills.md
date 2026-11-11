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
https://posit-dev.github.io/pointblank/.well-known/agent-skills/check-training-data/SKILL.md
```

Or browse the skill files below.

pointblank

write-validation

validate-yaml

draft-validation

define-contracts

scan-and-profile

generate-data

check-training-data

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
  backends (Polars, Pandas, PySpark, DuckDB, databases via Ibis),
  threshold levels, actions, and result extraction (including per-row
  failure flags). Use when building, running, or troubleshooting
  data-validation pipelines.
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
    - pyspark
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
    +-- data-backends.md        <- Polars, Pandas, PySpark, DuckDB, Ibis, files
```

## When to use what

| I want to...                         | Use                                       |
| ------------------------------------ | ----------------------------------------- |
| Check column values meet a condition | `col_vals_gt/lt/eq/...`                   |
| Ensure no nulls in a column          | `col_vals_not_null`                       |
| Check values are in an allowed set   | `col_vals_in_set`, `col_vals_in_table`    |
| Match a regex pattern                | `col_vals_regex`                          |
| Check string lengths                 | `col_vals_str_len`                        |
| Validate table schema                | `col_schema_match`                        |
| Check row/column counts              | `row_count_match`, `col_count_match`      |
| Find duplicate rows                  | `rows_distinct`, `rows_near_distinct`     |
| Check data freshness                 | `data_freshness`                          |
| Check coded missing values           | `MissingSpec`, `col_missing_*`            |
| Check column sums/means/SDs          | `col_sum_*`, `col_avg_*`, `col_sd_*`      |
| Reconcile totals with a source       | `tbl_reconcile`                           |
| Combine multiple conditions          | `conjointly`                              |
| Run a custom check                   | `specially`                               |
| Use LLM-based validation             | `prompt`, `calibrate_prompt`              |
| Score rows with a model/function     | `col_vals_scored`, `pb.scorers`           |
| Check text quality, language, PII    | `col_vals_text_quality`, `col_vals_in_language`, `col_vals_no_pii` |
| Check text, chat, or RAG data        | `col_vals_token_len`, `col_vals_chat_format`, `pb.rag` |
| Reuse a bundle of steps              | `Steps`, `add_steps`                      |
| Select columns by pattern            | `starts_with`, `contains`, `matches`      |
| Set failure thresholds               | `Thresholds`                              |
| Trigger actions on failure           | `Actions`, `FinalActions`                 |
| Get failing rows                     | `get_data_extracts`                       |
| Split data into pass/fail            | `get_sundered_data`                       |
| Flag the steps each row failed       | `get_row_flags`                           |
| Profile a dataset first              | `DataScan`                                |
| Compare two tables (diff)            | `compare`, `tbl_match(keys=...)`          |
| Define validation in YAML            | `yaml_interrogate`                        |
| Enforce contracts in a pipeline      | `Contract`, `Pipeline`                    |
| Generate test data                   | `Schema.generate`, field classes          |
| Draft validation with an LLM         | `DraftValidation`                         |
| See which checks matter to a model   | `Ablation`                                |
| Check splits or synthetic data       | `check_splits`, `check_synthetic`         |
| Write a dataset card                 | `to_data_card`                            |

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
| DuckDB      | `ibis.Table` via `pb.connect_to_table("duckdb://...::table")`       |
| PostgreSQL  | `ibis.Table` via `pb.connect_to_table("postgresql://...")`          |
| MySQL       | `ibis.Table` via `pb.connect_to_table("mysql://...")`               |
| SQLite      | `ibis.Table` via `pb.connect_to_table("sqlite://...")`              |
| Snowflake   | `ibis.Table` via `pb.connect_to_table("snowflake://...")`           |
| PySpark     | `pyspark.sql.DataFrame`                                             |
| CSV/Parquet | File path string: `"data/orders.csv"`, `"s3://bucket/file.parquet"` |

```python
tbl = pb.connect_to_table("duckdb:///path/to/warehouse.ddb::sales")
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

# Keep every row, with a pb_failed_<i>_ column per step plus
# pb_n_failed_ and pb_row_score_ (share of the row's checks passed)
flags = validation.get_row_flags()

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

| Skill               | When to use it                                    |
| ------------------- | ------------------------------------------------- |
| write-validation    | Detailed guidance on choosing and composing steps |
| define-contracts    | Contract and Pipeline boundary validation         |
| scan-and-profile    | Profile data before writing validation rules      |
| validate-yaml       | Define validation plans in YAML                   |
| generate-data       | Create synthetic test data from schemas           |
| draft-validation    | LLM-assisted validation drafting and editing      |
| check-training-data | Text/RAG/ML data checks, ablation, data cards     |

## Gotchas

1. **Call `.interrogate()` last.** Validation methods only define steps;
   nothing executes until `interrogate()` is called.
2. **Column selectors are case-insensitive by default.** Pass
   `case_sensitive=True` to `starts_with()`, `contains()`, etc. if
   needed.
3. **Threshold fractions vs counts.** A threshold of `0.05` is reached
   when 5% or more of test units fail; a threshold of `5` is reached
   when 5 or more fail.
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

`expr_col()` creates backend-agnostic column expressions for use in
`conjointly()`. Each argument of `conjointly()` is a callable that takes
the table and returns an expression:

```python
from pointblank import expr_col

.conjointly(
    lambda df: expr_col("start") < expr_col("end"),
    lambda df: expr_col("duration") > 0,
)
```

`expr_col` supports: `>`, `<`, `==`, `!=`, `>=`, `<=`, `+`, `-`,
`*`, `/`, `is_null()`, `is_not_null()`, `&`, `|`. It doesn't work in
`col_vals_expr()`, which needs a native Polars/Narwhals/Ibis
expression (e.g. `pl.col("start") < pl.col("end")`).

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

Selectors work in the `col_vals_*` methods (except `col_vals_scored`),
`col_exists`, `col_pct_null`, `col_pct_missing`, `col_missing_coded`,
and `col_missing_only_coded`. The aggregate methods (`col_sum_gt`, ...),
`col_missing_consistent`, and the text checks (`col_vals_scored`,
`rows_near_distinct`, `rows_not_contaminated`) take column names only:

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

| Backend       | Input type              | Install extra             |
|---------------|-------------------------|---------------------------|
| Polars        | `pl.DataFrame`, `pl.LazyFrame` | `pointblank[pl]`   |
| Pandas        | `pd.DataFrame`          | `pointblank[pd]`          |
| PySpark       | `pyspark.sql.DataFrame` | `pointblank[pyspark]`     |
| DuckDB        | `ibis.Table`            | `pointblank[duckdb]`      |
| PostgreSQL    | `ibis.Table`            | `pointblank[postgres]`    |
| MySQL         | `ibis.Table`            | `pointblank[mysql]`       |
| SQLite        | `ibis.Table`            | `pointblank[sqlite]`      |
| SQL Server    | `ibis.Table`            | `pointblank[mssql]`       |
| Snowflake     | `ibis.Table`            | `pointblank[snowflake]`   |
| BigQuery      | `ibis.Table`            | `pointblank[bigquery]`    |
| Databricks    | `ibis.Table`            | `pointblank[databricks]`  |
| CSV files     | File path string        | (none)                    |
| Parquet files | File path string        | (none)                    |

A PySpark DataFrame is passed to `Validate(data=...)` as it is; the
checks run on Spark. The text and scoring checks (`col_vals_scored`,
`rows_near_distinct`, `col_vals_chat_format`, `rows_not_contaminated`)
collect the distinct values of their columns to Python for a database
or PySpark table and join the results back, so a sampling `pre=` keeps
large tables cheap.

## Connecting to databases

Use `pb.connect_to_table()` with a connection string. Append
`::table_name` to specify the table:

```python
import pointblank as pb

# DuckDB
tbl = pb.connect_to_table("duckdb:///path/to/warehouse.ddb::sales")

# PostgreSQL
tbl = pb.connect_to_table("postgresql://user:pass@host:5432/db::orders")

# MySQL
tbl = pb.connect_to_table("mysql://user:pass@host:3306/db::customers")

# SQLite
tbl = pb.connect_to_table("sqlite:///path/to/local.db::events")

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
pb.print_database_tables("duckdb:///path/to/warehouse.ddb")  # returns a list of table names
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
    value,                # numeric/date value, col("other_column")
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

`value` can be a number, a `datetime.date`/`datetime` (or an ISO
string such as `"2016-01-10"`) for date columns, or `col("other")` to
compare against another column. These methods can't check boolean
columns; use `col_vals_expr(expr=pl.col("flag"))` instead.

## Range methods

```python
.col_vals_between(columns, left, right, inclusive=(True, True), na_pass=False, ...)
.col_vals_outside(columns, left, right, inclusive=(True, True), na_pass=False, ...)
```

`inclusive` controls boundary inclusion: `(True, True)` = closed
interval, `(False, False)` = open interval.

## Set membership

```python
.col_vals_in_set(columns, set=["a", "b", "c"], ...)          # no na_pass
.col_vals_not_in_set(columns, set=["x", "y"], na_pass=False, ...)

# Values must appear in a column of another table (no segments)
.col_vals_in_table(columns, ref_table=lookup_df, ref_column="code", na_pass=False, ...)
```

## Monotonicity

```python
.col_vals_increasing(columns, allow_stationary=False, decreasing_tol=None, order_by=None, ...)
.col_vals_decreasing(columns, allow_stationary=False, increasing_tol=None, order_by=None, ...)
```

## Null checks

```python
.col_vals_null(columns, ...)      # all values must be null
.col_vals_not_null(columns, ...)  # no values may be null
```

No `na_pass` or `missing` parameters on these methods.

## Pattern, spec, and length matching

```python
.col_vals_regex(columns, pattern="^[A-Z]{3}$", na_pass=False, inverse=False, ...)
.col_vals_within_spec(columns, spec="email", na_pass=False, ...)
.col_vals_str_len(columns, min_val=None, max_val=None, na_pass=False, ...)
```

Specs: `"isbn"`, `"vin"`, `"postal_code[<CC>]"` (e.g. `"postal_code[US]"`),
`"credit_card"`, `"iban[<CC>]"`, `"swift"`, `"phone"`, `"email"`,
`"url"`, `"ipv4"`, `"ipv6"`, `"mac"`. A postal code always needs a
country code.

## Expression-based

```python
import polars as pl

.col_vals_expr(expr=pl.col("price") * pl.col("qty") > 0, ...)
.col_vals_expr(expr=lambda df: df["price"] > 0, ...)
```

`expr` is a Polars, Narwhals, or Ibis expression (matching the table
type), or a callable that takes the table. There's no `columns`
parameter.

## Aggregate comparison methods

Pattern: `col_{agg}_{comp}()` where agg is `sum`, `avg`, or `sd` and
comp is `eq`, `gt`, `ge`, `lt`, or `le` (15 methods; there are no
`_between` variants).

```python
.col_sum_gt(columns, value=1000, tol=0, ...)
.col_avg_le(columns, value=50.0, tol=0.5, ...)
.col_sd_lt(columns, value=10, ...)
```

These take only `columns`, `value`, `tol`, `thresholds`, `brief`,
`actions`, `active`, and `dimension` (no `pre`, `segments`, or
`na_pass`). When `Validate(reference=...)` is set and `value=None`,
they compare against the same column of the reference table; use
`value=pb.ref("other_col")` for a differently named column.

## Missing values

```python
.col_pct_null(columns, p=0.05, tol=0.05, ...)  # null share in [p - tol, p + tol]
.col_pct_missing(columns, missing=spec, max_pct=0.10, reason=None, category=None, ...)

# Structured missingness (spec = pb.MissingSpec(reasons={-99: "refused"}))
.col_missing_coded(columns, missing=spec, ...)          # no raw nulls, only codes
.col_missing_only_coded(columns, missing=spec, allowed=None, min_val=None, max_val=None, ...)
.col_missing_consistent(columns=["a", "b"], missing=spec, when_reason="not_asked", ...)
```

`col_pct_null` and `col_pct_missing` have no `pre`/`segments`. Pass
`missing=spec` to the value-comparison methods so coded values aren't
compared as real values.

## Structural checks

```python
.col_exists(columns, ...)
.col_schema_match(schema, complete=True, in_order=True,
                  case_sensitive_colnames=True, case_sensitive_dtypes=True,
                  full_match_dtypes=True, ...)
.col_count_match(count=10, inverse=False, ...)        # count can be a table
.row_count_match(count=1000, tol=0, inverse=False, ...)
.rows_distinct(columns_subset=None, ...)
.rows_complete(columns_subset=None, ...)
```

## Data freshness

```python
.data_freshness(column="updated_at", max_age="24 hours", reference_time=None,
                timezone=None, allow_tz_mismatch=False, allow_future=False, ...)
```

`max_age` accepts strings like `"24 hours"`, `"30 minutes"`, `"2 weeks"`
(short forms like `"2h"` and `"1d"` also work) or `datetime.timedelta`
objects. Data dated later than the reference time fails unless
`allow_future=` allows it (a duration such as `"1m"` for clock skew,
or `True` for any amount).

## Table comparison

```python
# Row-by-row match against another table
.tbl_match(
    tbl_compare=other_df,
    keys=None,            # match rows by key instead of position
    columns=None,         # subset of columns to compare
    column_map=None,      # {"name_in_compare": "name_in_target"}
    tolerance=None,       # number, {"atol": .., "rtol": ..}, or per-column dict
    normalize=None,       # {"name": ["strip", "lower"]}, {"x": [{"round": 2}]}
    null_equal=True,
    schema="strict",      # or "common"
    dup_keys="flag",      # or "compare"
    order_by=None,
    partitions=None,
)

# Totals by key (or grand totals) must agree with a source table
.tbl_reconcile(
    tbl_compare=source_df,
    keys="region",        # None for grand totals
    measures="amount",
    agg="sum",
    tolerance=None,
    column_map={"amount": "amt"},  # name in tbl_compare -> name in target
    partitions=None,
)
```

## Compound and custom checks

```python
import polars as pl
from pointblank import expr_col

# Multiple conditions must all hold for each row; each argument is a
# callable taking the table and returning a boolean expression
.conjointly(
    lambda df: pl.col("a") > 0,
    lambda df: pl.col("a") < 100,
    lambda df: expr_col("a") < expr_col("b"),   # backend-agnostic form
)

# Fully custom check: a callable taking the table (or nothing) and
# returning boolean results
.specially(expr=lambda tbl: len(tbl) > 0)
```

Bare expressions (not wrapped in a callable) don't work in
`conjointly()`.

## Text and ML data checks

```python
# Each row scored by a model or function; passes if the score is in bounds
.col_vals_scored(columns="text", scorer=my_scorer, min_val=None, max_val=None,
                 na_pass=False, cache=True, keep_scores=True, ...)

# Near-duplicate rows (MinHash over word n-grams, or exact after normalizing)
.rows_near_distinct(columns="text", method="minhash", threshold=0.85, ngram=5, num_perm=128, ...)

# Text checks: each adds col_vals_scored() or col_vals_regex() steps,
# one set per column, with a brief for each
.col_vals_text_quality(columns="text", min_words=50, max_words=100_000, ...)  # 8 Gopher filters
.col_vals_in_language(columns="text", expected="en", expected_column=None,
                      min_confidence=0.5, backend="langid", ...)
.col_vals_token_len(columns="text", tokenizer="whitespace", min_val=None, max_val=4096, ...)
.col_vals_no_pii(columns="text", entities=("email", "phone", "credit_card"), ...)

# Chat conversations (list of messages or JSON text) are well formed
.col_vals_chat_format(columns="messages", roles=("system", "user", "assistant"),
                      alternate=True, first_role=None, last_role=None,
                      allow_empty=False, role_key="role", content_key="content",
                      na_pass=False, ...)

# Rows don't overlap with a reference (e.g. benchmark questions)
.rows_not_contaminated(columns="text", reference=benchmark_df,
                       reference_columns="question", method="ngram",
                       ngram=None, threshold=0.8, num_perm=128, ...)
```

`ngram=None` in `rows_not_contaminated()` means 13 words. A scorer
takes a batch (a list of row dicts) and returns a list of numbers or
`None`; see the `check-training-data` skill for the `pb.scorers`
adapters, the text checks, and the `pb.rag` step bundles.

## LLM-based validation

```python
.prompt(
    prompt="Check if product names are appropriate",
    model="anthropic:claude-sonnet-4-6",   # or a list of models
    columns_subset=["product_name"],
    attachments=None,
    batch_size=1000,
    max_concurrent=3,
    aggregate="majority",                  # or "all"/"any", for a list of models
)

# After interrogation: agreement of step 3 with hand labels
validation.calibrate_prompt(i=3, labels="human_label")  # column or list
```

`calibrate_prompt()` returns a dict (agreement share, Cohen's kappa,
disagreement counts, and per-model agreement when `model=` is a list)
and adds it as a note on the step.

## Common parameters

| Parameter    | Type                    | Description                              |
|--------------|-------------------------|------------------------------------------|
| `columns`    | `str \| list \| selector` | Target column(s)                       |
| `value`      | `numeric \| col()`      | Comparison value or column reference     |
| `na_pass`    | `bool`                  | Treat nulls as passing (default `False`) |
| `missing`    | `MissingSpec`           | Structured missingness definition        |
| `pre`        | `Callable`              | Transform data before check              |
| `segments`   | `str \| tuple`          | Segment validation by groups             |
| `thresholds` | `Thresholds`            | Per-step failure thresholds              |
| `actions`    | `Actions`               | Per-step actions on threshold breach     |
| `brief`      | `str \| bool`           | Step description text                    |
| `active`     | `bool \| Callable`      | Conditionally skip step                  |
| `dimension`  | `str`                   | Tag for dimensional scoring              |

`segments="f"` makes one step per distinct value of `f`;
`segments=("f", ["low", "high"])` picks the values; and
`segments=("f", pb.seg_group([["low", "mid"], ["high"]]))` combines
values into groups. Wrap a `pre=` function in `pb.row_preserving()`
when it only filters or changes rows, so the step still counts in
`get_sundered_data()` and `get_row_flags()`.
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
| No near-duplicate text            | `rows_near_distinct`                 |
| All rows complete (no nulls)      | `rows_complete`                      |
| Values above/below a bound        | `col_vals_gt/lt/ge/le`               |
| Values in a numeric range         | `col_vals_between`                   |
| Values outside a range            | `col_vals_outside`                   |
| Values from an allowed set        | `col_vals_in_set`                    |
| Values found in a lookup table    | `col_vals_in_table`                  |
| Values not in a forbidden set     | `col_vals_not_in_set`                |
| Values match a pattern            | `col_vals_regex`                     |
| String length within bounds       | `col_vals_str_len`                   |
| Values match a format (email etc) | `col_vals_within_spec`               |
| Values are monotonically ordered  | `col_vals_increasing/decreasing`     |
| Null percentage within budget     | `col_pct_null`                       |
| Structured missingness            | `col_pct_missing`, `col_missing_coded`, `col_missing_only_coded`, `col_missing_consistent` with `MissingSpec` |
| Aggregate comparison (sum, avg)   | `col_sum_eq`, `col_avg_gt`, etc.     |
| Data is recent enough             | `data_freshness`                     |
| Table matches another table       | `tbl_match` (`keys=`, `tolerance=`)  |
| Totals agree with a source table  | `tbl_reconcile`                      |
| Multiple conditions per row       | `conjointly`                         |
| Custom logic                      | `specially`                          |
| LLM-based semantic check          | `prompt`                             |
| Score from a model or function    | `col_vals_scored`                    |
| Chat data well formed             | `col_vals_chat_format`               |
| No overlap with a benchmark       | `rows_not_contaminated`              |

For text corpora, RAG data, and model training sets, see the
`check-training-data` skill.

## Core concepts

### Ordering your steps

Organize validation steps from structural to semantic:

1. **Schema checks** -- `col_exists`, `col_schema_match`,
   `col_count_match`, `row_count_match`
2. **Completeness checks** -- `col_vals_not_null`, `rows_complete`,
   `col_pct_null`, `col_missing_*`
3. **Uniqueness checks** -- `rows_distinct`, `rows_near_distinct`
4. **Value-range checks** -- `col_vals_gt`, `col_vals_between`, etc.
5. **Format checks** -- `col_vals_regex`, `col_vals_within_spec`,
   `col_vals_str_len`, `col_vals_chat_format`
6. **Set membership** -- `col_vals_in_set`, `col_vals_not_in_set`,
   `col_vals_in_table`
7. **Cross-column checks** -- `conjointly`, `col_vals_expr`
8. **Aggregate checks** -- `col_sum_eq`, `col_avg_gt`, `tbl_reconcile`
9. **Freshness checks** -- `data_freshness`
10. **Custom/semantic checks** -- `specially`, `prompt`,
    `col_vals_scored`, `rows_not_contaminated`

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

Break validation into groups to see which segments fail. Each
segment becomes its own step in the report:

```python
# One step per distinct value of "region"
.col_vals_gt(columns="revenue", value=0, segments="region")

# Only some values
.col_vals_gt(columns="revenue", value=0, segments=("region", ["north", "south"]))

# Combine values into groups (seg_group goes inside the tuple)
.col_vals_gt(
    columns="revenue",
    value=0,
    segments=("region", pb.seg_group([["north", "south"], ["east", "west"]])),
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

Steps with `pre=` are left out of `get_sundered_data()` and
`get_row_flags()`, since their rows can't be matched to the target
table. Wrap a function that only filters, reorders, or changes values
in `pb.row_preserving(...)` to keep the step in.

### Reusable bundles of steps

`pb.Steps` records steps (with the same methods as `Validate`) that
can be added to any validation with `add_steps()`:

```python
id_checks = (
    pb.Steps()
    .col_vals_not_null(columns="id")
    .rows_distinct(columns_subset="id")
)

(
    pb.Validate(data=df)
    .add_steps(id_checks)
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)
```

`add_steps(*steps, active=None, thresholds=None, exclude=None,
columns_map=None)` can override the bundle's thresholds, skip steps,
or rename columns. The `pb.rag` functions return `Steps` bundles too.

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

Template variables: `{type}`, `{level}`, `{step}` (or `{i}`), `{col}`
(or `{column}`), `{val}` (or `{value}`), `{time}`.

### Final actions

Run after all steps complete, with access to the full summary:

```python
def check_overall():
    summary = pb.get_validation_summary()
    if summary and summary["n_failing_steps"] > 0:
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

# Every row, flagged with the steps it failed (pb_failed_<i>_ columns)
flags = validation.get_row_flags()

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
5. **`conjointly` takes callables that return expressions.** Each
   argument is like `lambda df: pl.col("a") > 0` (or
   `lambda df: expr_col("a") > 0`), not a validation method call.
6. **Aggregate methods compare one value per column**, not per row.
   They produce a single pass/fail per column, and they take no `pre=`
   or `segments=`.
7. **`segments` splits the check into sub-groups.** Each segment is
   reported separately in the validation report. `pb.seg_group()` is
   only valid inside a `(column, ...)` tuple.
8. **Comparison methods can't check boolean columns.** Use
   `col_vals_expr(expr=pl.col("flag"))` (or `~pl.col("flag")`) instead
   of `col_vals_eq(columns="flag", value=True)`.
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
| Average within tolerance    | `col_avg_eq(columns, value, tol=0.5)`   |
| Standard deviation bounded  | `col_sd_lt(columns, value=max_sd)`      |
| Monotonically increasing    | `col_vals_increasing(columns)`          |
| Null share near a target    | `col_pct_null(columns, p=0.05, tol=0.05)`|

### String columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| Matches a pattern           | `col_vals_regex(columns, pattern)`        |
| Valid email/URL/phone/etc.  | `col_vals_within_spec(columns, spec)`     |
| Length within bounds        | `col_vals_str_len(columns, min_val, max_val)` |
| From a known set            | `col_vals_in_set(columns, set)`           |
| Exists in a lookup table    | `col_vals_in_table(columns, ref_table, ref_column)` |
| Not a forbidden value       | `col_vals_not_in_set(columns, set)`       |
| Not null                    | `col_vals_not_null(columns)`              |

### Date/datetime columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| After a cutoff date         | `col_vals_gt(columns, value=dt.date(2024, 1, 1))` |
| Within a date range         | `col_vals_between(columns, left="2024-01-01", right="2024-12-31")` |
| Data is recent              | `data_freshness(column, max_age="2 hours")` |
| Chronologically ordered     | `col_vals_increasing(columns)`            |

### Boolean columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| All true                    | `col_vals_expr(expr=pl.col("flag"))`      |
| All false                   | `col_vals_expr(expr=~pl.col("flag"))`     |

The comparison and set methods (`col_vals_eq`, `col_vals_in_set`, ...)
don't accept boolean columns; use `col_vals_expr()` as above.

### Text columns (corpora, chat data)

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| Gopher quality filters      | `col_vals_text_quality(columns="text")`   |
| In the expected language    | `col_vals_in_language(columns="text", expected="en")` |
| Length in model tokens      | `col_vals_token_len(columns="text", tokenizer=..., max_val=4096)` |
| No emails/phones/card nums  | `col_vals_no_pii(columns="text")`         |
| No near duplicates          | `rows_near_distinct(columns="text")`      |
| Model/function score bound  | `col_vals_scored(columns, scorer, min_val)` |
| Chat messages well formed   | `col_vals_chat_format(columns="messages")` |
| No benchmark overlap        | `rows_not_contaminated(columns, reference)` |

See the `check-training-data` skill for details.

## By concern type

### Completeness

```python
.col_vals_not_null(columns="required_field")
.rows_complete()                          # no nulls in any column
.rows_complete(columns_subset=["a", "b"]) # no nulls in a, b
.col_pct_null(columns="optional", p=0.10, tol=0.10) # 0-20% null
```

### Uniqueness

```python
.rows_distinct()                             # all rows unique
.rows_distinct(columns_subset=["id"])        # id column unique
.rows_distinct(columns_subset=["a", "b"])    # composite unique
```

### Consistency (cross-column)

```python
import polars as pl

# Every row must satisfy all the conditions
.conjointly(
    lambda df: pl.col("start") < pl.col("end"),
    lambda df: pl.col("duration") > 0,
)

# A single cross-column expression
.col_vals_expr(expr=pl.col("start") < pl.col("end"))

# Compare against another column
.col_vals_gt(columns="end", value=pb.col("start"))
```

Use `pb.expr_col("start") < pb.expr_col("end")` inside the
`conjointly()` lambdas for a backend-agnostic form.

### Referential (against another table)

```python
pb.Validate(data=current, reference=previous)
.col_sum_eq(columns="total")       # sums match reference
.col_avg_eq(columns="avg_price")   # averages match reference
.tbl_match(tbl_compare=expected)   # tables are identical
.tbl_match(tbl_compare=expected, keys="id", tolerance=0.01)  # by key
.tbl_reconcile(tbl_compare=source, keys="region", measures="amount")
.col_vals_in_table(columns="customer_id", ref_table=customers,
                   ref_column="id")  # foreign key
```

## Available spec values for col_vals_within_spec

Use these with `.col_vals_within_spec(columns, spec="...")`:

- `"email"` -- email addresses
- `"url"` -- URLs
- `"phone"` -- phone numbers
- `"ipv4"`, `"ipv6"` -- IP addresses
- `"mac"` -- MAC addresses
- `"isbn"` -- 10- and 13-digit ISBNs
- `"vin"` -- vehicle identification numbers
- `"credit_card"` -- card numbers (Luhn check)
- `"swift"` -- SWIFT/BIC codes
- `"postal_code[<CC>]"` -- postal codes for a country, e.g.
  `"postal_code[US]"`, `"postal_code[CA]"` (a country is required;
  `"zip"` on its own raises an error)
- `"iban[<CC>]"` -- IBANs for a country, e.g. `"iban[DE]"`

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

| Value     | Level is reached when...           | Example                |
|-----------|------------------------------------|------------------------|
| `0.05`    | 5% or more of test units fail      | 50 failures in 1000    |
| `5`       | 5 or more test units fail          | Absolute count         |
| `True`    | Any test unit fails                | Same as `1`            |
| `None`    | Level not used                     | No threshold set       |

`Thresholds` also accepts a tuple (`(0.01, 0.05, 0.10)`) or a dict
(`{"warning": 0.01, "error": 0.05}`) wherever `thresholds=` is taken.

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

- `str` -- message template (printed)
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

Other placeholders (e.g. `{n_failed}`, `{n}`) are not substituted.

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
    # meta keys: step, column, type, value, level, level_num, time,
    #            autobrief, failure_text
```

## Final actions

`FinalActions(*actions)` -- run after all steps complete.

```python
def summary_check():
    summary = pb.get_validation_summary()
    if summary["n_failing_steps"] > 0:
        create_jira_ticket(summary)

pb.Validate(
    data=df,
    final_actions=pb.FinalActions(summary_check, "Validation complete."),
)
```

String final actions are printed as they are: template variables like
`{time}` are not substituted (use a callable for dynamic messages).

### get_validation_summary()

Inside a final action callable, call `pb.get_validation_summary()`:

```python
def report():
    s = pb.get_validation_summary()
    # s keys: n_steps, n_passing_steps, n_failing_steps,
    #         n_warning_steps, n_error_steps, n_critical_steps,
    #         list_passing_steps, list_failing_steps, all_passed,
    #         highest_severity, overall_health_score, dimension_scores,
    #         tbl_name, tbl_row_count, tbl_column_count,
    #         validation_duration, and dict_* (per-step dicts)
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

### Send to OpenTelemetry

`emit_otel()` is a final action (use it in `FinalActions`, not `Actions`).
It sends metrics by default, plus trace spans and log records when asked,
after every `interrogate()`:

```python
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=0.05),
    final_actions=pb.FinalActions(
        pb.emit_otel(
            enable_tracing=True,
            extra_attributes={"pipeline": "my-pipeline"},
        ),
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
  Covers the YAML schema, validate_yaml() for checking a config,
  yaml_interrogate() for execution, yaml_to_python() for code
  generation, Validate.to_yaml() for export, Python code in YAML
  (allow_python), data source configuration, and the ablation block
  run by yaml_ablation(). Use when defining validation plans
  declaratively or sharing them across teams.
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
tbl: small_table
tbl_name: small_table
label: Small table checks
thresholds:
  warning: 0.01
  error: 0.05

steps:
  - col_vals_gt:
      columns: d
      value: 100
  - col_vals_not_null:
      columns: date_time
  - col_vals_in_set:
      columns: f
      set: [low, mid, high]
  - rows_distinct
```

```python
import pointblank as pb

# Execute the YAML plan
validation = pb.yaml_interrogate("validation.yaml")
validation.get_tabular_report()
```

From the command line: `pb run validation.yaml`.

## Skill directory structure

```
skills/validate-yaml/
+-- SKILL.md                    <- This file
+-- references/
    +-- yaml-schema.md          <- Full YAML key reference
```

## When to use what

| I want to...                         | Use                                         |
| ------------------------------------ | ------------------------------------------- |
| Check a config's top-level structure | `validate_yaml()`                           |
| Execute a YAML validation plan       | `yaml_interrogate()`, `pb run`              |
| Convert YAML to Python code          | `yaml_to_python()`                          |
| Export a Python validation to YAML   | `Validate.to_yaml()`                        |
| Override the data source at runtime  | `yaml_interrogate(set_tbl=df)`              |
| Run `pre:`, lambdas, `python:` blocks | `yaml_interrogate(allow_python=True)`      |
| Make modules available to that code  | `yaml_interrogate(namespaces=["math"])`     |
| Keep an ablation with the validation | `ablation:` block + `yaml_ablation()`       |

## Core concepts

### YAML structure

A YAML validation plan has two required keys (`tbl` and `steps`)
and several optional keys. Unknown top-level keys raise an error.

```yaml
# Required
tbl: data/orders.csv           # data source
steps:                         # validation steps
  - col_vals_gt:
      columns: amount
      value: 0

# Optional metadata
tbl_name: orders
label: Daily order check
owner: data-team
consumers: [analytics, reporting]
version: "1.0"
lang: en
locale: en-US
df_library: polars             # polars (default), pandas, or duckdb

# Optional thresholds and actions
thresholds:
  warning: 0.01
  error: 0.05
  critical: 0.25

actions:
  warning: "Step {step} ({type}) on {col}: warning at {time}"
  error: "Error in step {step}: {col}"
  highest_only: true

final_actions:
  - "Validation complete."     # printed after all steps run

# Optional brief setting
brief: true

# Optional reference table (for aggregate checks with no value)
reference: data/orders_last_week.csv

# Optional named missing-value specs (used by `missing: survey` in steps)
missing_specs:
  survey:
    reasons:
      -99: refused
      -98: not asked
    null_is_missing: true

# Optional ablation (ignored by yaml_interrogate; see below)
ablation:
  metric: mae
  strategies:
    - drop_each
```

### Data sources in YAML

The `tbl` key accepts:

| Value                               | Interpreted as                             |
| ----------------------------------- | ------------------------------------------ |
| `small_table`                       | A built-in dataset                         |
| `data.csv`, `data.parquet`          | A file path (read with `df_library`)       |
| `"duckdb:///path/to/db.ddb::table"` | A database connection string               |
| `null`                              | A template; pass the table with `set_tbl=` |
| `python: \|` block                  | Code that makes the table (`allow_python`) |

```yaml
tbl:
  python: |
    pl.read_csv("orders.csv").filter(pl.col("region") == "EU")
```

### Steps in YAML

Each step is a mapping of one method name to its parameters, or just
the method name for a step with no parameters:

```yaml
steps:
  # Value comparison
  - col_vals_gt:
      columns: amount
      value: 0
      na_pass: true

  # Compare with another column (no allow_python needed)
  - col_vals_lt:
      columns: discount
      value: pb.col("amount")

  # Range check
  - col_vals_between:
      columns: score
      left: 0
      right: 100
      inclusive: [true, true]

  # Set membership
  - col_vals_in_set:
      columns: status
      set: [active, inactive, pending]

  # Pattern match (double backslashes in double-quoted strings)
  - col_vals_regex:
      columns: code
      pattern: "^\\d-[a-z]{3}-\\d{3}$"

  # Null checks
  - col_vals_not_null:
      columns: [id, name, email]
  - col_pct_null:
      columns: email
      p: 0.05            # null share must be in [p - tol, p + tol]
      tol: 0.05

  # Schema match: a `columns:` list of [name, type] pairs or bare names
  - col_schema_match:
      schema:
        columns:
          - [id, Int64]
          - [name, String]
          - amount             # name only, no type check
      complete: true
      in_order: true

  # Row count, distinct and complete rows
  - row_count_match:
      count: 1000
      tol: 50
  - rows_distinct
  - rows_complete:
      columns_subset: [id, name]

  # Aggregates
  - col_sum_gt:
      columns: revenue
      value: 0

  # Freshness
  - data_freshness:
      column: updated_at
      max_age: 24 hours

  # Coded missing values (a name from `missing_specs:`)
  - col_missing_coded:
      columns: q1
      missing: survey

  # Segments: one step per value, or picked values
  - col_vals_gt:
      columns: revenue
      value: 0
      segments: region
  - col_vals_gt:
      columns: revenue
      value: 0
      segments: [[region, [north, south]]]

  # Per-step thresholds, actions, brief, and a conditional step
  - col_vals_gt:
      columns: new_feature
      value: 0
      thresholds:
        warning: 5
        error: 20
      actions:
        warning: "Step {step} warning"
      brief: "new_feature is positive"
      active: pb.has_columns("new_feature")
```

Text and ML checks have YAML forms too. A `col_vals_scored` scorer is
written as one of the `pb.scorers` adapters with its settings:

```yaml
  - col_vals_scored:
      columns: text
      scorer:
        text_stat:
          stat: word_count
      min_val: 50
  - rows_near_distinct:
      columns: text
      threshold: 0.9
  - rows_not_contaminated:
      columns: text
      reference: benchmark.csv     # a file, a list of texts, or a dict of them by name
      reference_columns: question
  - col_vals_chat_format:
      columns: messages
```

### Python code in YAML

Code outside a small declarative subset isn't run unless you allow it,
since it can do anything the Python session can. Without permission,
`yaml_interrogate()` raises `YAMLValidationError`.

Always allowed (no `allow_python` needed): `pb.load_dataset()`,
`pb.col()`, `pb.ref()`, `pb.seg_group()`, `pb.has_columns()`,
`pb.has_rows()`, the column selectors, `Decimal(...)`,
`datetime.date/time/datetime/timedelta(...)`, and `ZoneInfo(...)`.

Needs `allow_python=True`: `pre:`, `col_vals_expr`'s `expr:`,
`conjointly`'s `expressions:`, `specially`'s `expr:`, callable actions,
a scorer of your own, and any other `python:` block.

```yaml
steps:
  - col_vals_gt:
      columns: d
      value: 100
      pre: |
        lambda df: df.filter(pl.col("f") == "high")
  - conjointly:
      expressions:
        - "lambda df: pl.col('d') > pl.col('a')"
        - "lambda df: pl.col('a') > 0"
  - specially:
      expr: "lambda df: len(df) > 0"
  - col_vals_gt:
      columns: d
      value:
        python: math.floor(99.9)
```

```python
pb.yaml_interrogate("trusted.yaml", allow_python=True, namespaces=["math"])
```

`namespaces=` takes module names (`["math"]`) or a mapping of aliases
to module names (`{"np": "numpy"}`); `pb`, `pl`, `pd`, `Path`,
`datetime`, `Decimal`, and `ZoneInfo` are already available. From the
CLI: `pb run trusted.yaml --allow-python`. Only allow Python for configs
you trust.

### Checking a YAML config

```python
pb.validate_yaml("validation.yaml")  # returns None; raises YAMLValidationError
```

It checks the YAML syntax, the top-level keys, that `tbl` and a
non-empty `steps` list are present, the thresholds and actions, and
any `ablation:` block, without loading data. It doesn't check the
steps themselves. `yaml_to_python()` catches an unknown method name
or a malformed step without loading data; a misspelled parameter is
only caught when `yaml_interrogate()` calls the method.

### Executing a YAML plan

```python
# Run from a file or a YAML string
validation = pb.yaml_interrogate("validation.yaml")

# Override the data source (works with `tbl: null` templates)
validation = pb.yaml_interrogate("validation.yaml", set_tbl=my_df)
```

### Converting between YAML and Python

```python
print(pb.yaml_to_python("validation.yaml"))   # Python code in a ```python fence

yaml_text = validation.to_yaml()              # a Validate object as YAML
validation.to_yaml("validation.yaml")         # ...also written to a file
```

`to_yaml()` writes the table's name as `tbl:`, so rerun the file with
`set_tbl=` when the table isn't a built-in dataset or a path.

### Ablation in YAML

An `ablation:` block describes an `Ablation` of the plan's validation
(see the `check-training-data` skill). `yaml_interrogate()` ignores
it; `yaml_ablation()` interrogates the validation and adds the variants:

```yaml
ablation:
  evaluate: my_models.training:evaluate   # module:function (needs allow_python)
  metric: mae
  higher_is_better: false
  repeats: 5
  strategies:
    - drop_each
    - drop_steps:
        steps: [1, 2]
        name: Without either step
    - drop_groups:
        column: sensor
    - sweep_step:
        step: 1
        param: right
        values: [30, 45, 60]
    - inject:
        corruption: "null"     # quote it, or YAML reads a null
        step: 2
```

```python
# Pass the function from Python (no allow_python needed) ...
ablation = pb.yaml_ablation("plan.yaml", evaluate=evaluate, set_tbl=train)
ablation.plan()   # what will run
ablation.run()

# ... or let the block's `evaluate:` be imported
ablation = pb.yaml_ablation("plan.yaml", allow_python=True)
```

## Workflows

### Creating a YAML validation plan

1. Profile the data to understand its shape and types.
2. Write the YAML file with `tbl` and `steps` (or build the plan in
   Python and call `to_yaml()`).
3. Run `pb.validate_yaml()` to check its structure, then
   `pb.yaml_to_python()` to catch unknown methods without loading data.
4. Run `pb.yaml_interrogate()` (or `pb run`) to execute.
5. Review the report and iterate.

### Sharing plans across teams

1. Define the plan in YAML, with `tbl: null` if it's a template.
2. Commit to version control.
3. Team members execute with `pb.yaml_interrogate(..., set_tbl=df)`.

### Migrating from YAML to Python

1. Run `pb.yaml_to_python("plan.yaml")` to generate code.
2. Review and customize the generated Python.

## Gotchas

1. **Steps are `- method:` keys, not `method:` fields.** Write
   `- col_vals_gt:` with the parameters indented below it.
2. **Escape regex backslashes.** In a double-quoted YAML string,
   write `\\d` for `\d`.
3. **Quote values YAML would misread.** `"null"` for the `inject`
   corruption, `"1.0"` for a version, and a type with a comma such as
   `"Datetime(time_unit='us', time_zone=None)"`.
4. **`inclusive` is a list, not a tuple.** Write
   `inclusive: [true, true]`.
5. **`pre`, `conjointly`, `specially`, and `col_vals_expr` need
   `allow_python=True`.** `value: pb.col("x")` and
   `active: pb.has_rows(min=5)` don't.
6. **Give `columns:` as names.** Use a name or a list of names; a
   column selector in `columns:` isn't expanded in YAML (it is in
   Python).
7. **`col_schema_match` takes `schema: {columns: [...]}`**, not a
   mapping of names to types.
8. **Action strings support `{step}`/`{i}`, `{col}`/`{column}`,
   `{val}`/`{value}`, `{type}`, `{level}`, and `{time}`.** Final
   action strings are printed as they are, with no templating.
9. **`df_library` defaults to `"polars"`.** Set it to `"pandas"` or
   `"duckdb"` to read files into those instead.
````

### references/yaml-schema.md

``` gd-skills-raw
```

```` markdown
# YAML schema reference

Complete reference for the keys in a Pointblank YAML validation
plan. Unknown top-level keys raise a `YAMLValidationError`.

## Top-level keys

| Key              | Type                | Required | Default    | Description                        |
|------------------|---------------------|----------|------------|------------------------------------|
| `tbl`            | `str \| null \| python:` | yes | --         | Data source (dataset, path, connection, or `null` with `set_tbl=`) |
| `steps`          | `list`              | yes      | --         | Validation steps (non-empty)       |
| `tbl_name`       | `str`               | no       | `null`     | Display name for the table         |
| `label`          | `str`               | no       | `null`     | Validation plan label              |
| `owner`          | `str`               | no       | `null`     | Data owner identifier              |
| `consumers`      | `str \| list[str]`  | no       | `null`     | Data consumers                     |
| `version`        | `str`               | no       | `null`     | Plan version                       |
| `lang`           | `str`               | no       | `null`     | Report language code               |
| `locale`         | `str`               | no       | `null`     | Locale for value formatting        |
| `df_library`     | `str`               | no       | `"polars"` | `polars`, `pandas`, or `duckdb`    |
| `brief`          | `bool \| str`       | no       | `null`     | Global brief setting               |
| `reference`      | `str \| python:`    | no       | `null`     | Reference table (same forms as `tbl`) |
| `thresholds`     | `dict`              | no       | `null`     | Global thresholds                  |
| `actions`        | `dict`              | no       | `null`     | Global actions                     |
| `final_actions`  | `list`              | no       | `null`     | Post-validation actions            |
| `missing_specs`  | `dict`              | no       | `null`     | Named `MissingSpec` definitions    |
| `ablation`       | `dict`              | no       | `null`     | Ablation settings (`yaml_ablation()` only) |

## Steps

Each item in `steps` is a mapping of exactly one method name to its
parameters, or a bare method name for a step with no parameters:

```yaml
steps:
  - rows_distinct
  - col_vals_gt:
      columns: amount
      value: 0
```

The parameters are those of the `Validate` method, with these YAML
forms:

| Parameter        | YAML form                                                         |
|------------------|-------------------------------------------------------------------|
| `columns`        | A name or a list of names                                         |
| `value`, `left`, `right` | A literal, an ISO date string, or `pb.col("other")` / `pb.ref("col")` |
| `set`            | A list                                                            |
| `inclusive`      | A two-item list, e.g. `[true, false]`                             |
| `tol`            | A number, or a two-item `[lower, upper]` list                     |
| `schema`         | `{columns: [[name, type], name, ...]}` (for `col_schema_match`)   |
| `segments`       | A column name, or a list of names and `[column, value]` pairs     |
| `missing`        | A name from `missing_specs:`, or an inline spec mapping           |
| `scorer`         | `{<pb.scorers adapter>: {<settings>}}`, or a `python:` block       |
| `thresholds`     | `{warning: .., error: .., critical: ..}`                          |
| `actions`        | Same form as the top-level `actions`                              |
| `active`         | `true`/`false`, `pb.has_columns(...)`, or `pb.has_rows(...)`      |
| `pre`            | A lambda as a string (needs `allow_python=True`)                  |
| `expr`           | A string of code (`col_vals_expr`, `specially`; needs `allow_python=True`) |
| `expressions`    | A list of lambda strings (`conjointly`; needs `allow_python=True`) |

The adapters for `scorer:` are `embedding_similarity`, `reranker`,
`classifier`, `langid`, `perplexity`, `text_stat`, and `token_count`.

## Python code

A `python:` block (or a string of code in `pre`, `expr`, and
`expressions`) is run only with `yaml_interrogate(..., allow_python=True)`
or `pb run ... --allow-python`:

```yaml
tbl:
  python: |
    pl.read_csv("orders.csv").filter(pl.col("region") == "EU")
steps:
  - col_vals_gt:
      columns: amount
      value:
        python: math.floor(99.9)    # with namespaces=["math"]
```

These are evaluated without `allow_python`: `pb.load_dataset()`,
`pb.col()`, `pb.ref()`, `pb.seg_group()`, `pb.has_columns()`,
`pb.has_rows()`, the column selectors, `Decimal(...)`,
`datetime.date/time/datetime/timedelta(...)`, and `ZoneInfo(...)`.

## Thresholds format

```yaml
thresholds:
  warning: 0.01     # fraction (< 1) or count (>= 1); reached at >=
  error: 0.05
  critical: 0.25
```

## Actions format

```yaml
actions:
  warning: "Warning: {col} step {step}"
  error: "Error in {col} at {time}"
  critical: "CRITICAL: {type} failed for {col}"
  highest_only: true          # only the highest level reached runs
```

The keys are `warning`, `error`, `critical`, `default`, and
`highest_only`. Strings are printed with these template variables
filled in: `{type}`, `{level}`, `{step}`, `{i}`, `{col}`, `{column}`,
`{val}`, `{value}`, `{time}`. A callable action is a `python:` block.

`final_actions` is a list of strings (printed as they are) or `python:`
blocks.

## Missing specs format

`missing_specs` holds named specs that steps refer to with
`missing: <name>`:

```yaml
missing_specs:
  survey:
    reasons:
      -99: refused
      -98: not asked
    categories:
      nonresponse: [refused]
      design: [not asked]
    null_is_missing: true
    null_reason: unknown
    description: Survey nonresponse codes

steps:
  - col_missing_coded:
      columns: q1
      missing: survey
  - col_pct_missing:
      columns: q1
      missing: survey
      max_pct: 0.10
```

## Ablation format

Used by `yaml_ablation()` and ignored by `yaml_interrogate()`. The keys
are the parameters of `Ablation` other than the validation:
`evaluate`, `evaluate_version`, `metric`, `higher_is_better`, `repeats`,
`seed`, `control`, `duplicates`, `duplicates_order_by`, `max_removed`,
`conf_level`, `adjust`, `label`, `cache`, `holdout`, `ci`, `pass_as`,
and `strategies`.

```yaml
ablation:
  evaluate: my_models.training:evaluate   # needs allow_python, or pass evaluate=
  metric: mae
  higher_is_better: false
  repeats: 5
  control:
    stratified: sensor        # or size_matched (default), or null
  cache: .ablation_cache
  holdout: data/holdout.parquet
  strategies:
    - drop_each
    - drop_all_but_each
    - drop_steps:
        steps: [1, 2]
        name: Without steps 1 and 2
    - drop_groups:
        column: sensor
    - sweep_step:
        step: 1
        param: right
        values: [30, 45, 60]
    - inject:
        corruption: "null"    # null, shuffle, truncate, or duplicate
        step: 2
```

The strategies (and their parameters) are `drop_each` (`steps`),
`drop_all` (`steps`), `drop_steps` (`steps`, `name`),
`drop_all_but_each` (`steps`), `drop_groups` (`column`, `values`),
`sweep_score` (`score`, `quantiles`, `values`, `remove`), `sweep_step`
(`step`, `param`, `values`), `inject` (`corruption`, `column`, `step`,
`rates`, `name`), and `shapley` (`steps`, `permutations`, `seed`).

## Data source formats

```yaml
# Built-in datasets
tbl: small_table

# File paths (read with df_library)
tbl: data/orders.csv
tbl: data/sales.parquet

# Database connections (append ::table_name)
tbl: "duckdb:///path/to/warehouse.ddb::sales"
tbl: "postgresql://user:pass@host:5432/db::orders"
tbl: "sqlite:///path/to/local.db::events"

# A template: the table is given with set_tbl=
tbl: null
```

## Complete example

```yaml
tbl: "duckdb:///path/to/warehouse.ddb::daily_orders"
tbl_name: daily_orders
label: "Daily order quality check"
owner: data-engineering
consumers: [analytics, finance]
version: "2.1"
lang: en

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
  amount_codes:
    reasons:
      -1: refunded
    null_is_missing: true

steps:
  - col_schema_match:
      schema:
        columns:                  # Ibis type names for a database table
          - [order_id, int64]
          - [customer_id, int64]
          - [amount, float64]
          - [status, string]
          - created_at
      complete: true
      in_order: false

  - row_count_match:
      count: 1000
      tol: 200

  - col_vals_not_null:
      columns: [order_id, customer_id, status]

  - rows_distinct:
      columns_subset: [order_id]

  - col_vals_gt:
      columns: amount
      value: 0
      missing: amount_codes

  - col_vals_in_set:
      columns: status
      set: [pending, processing, shipped, delivered, cancelled]

  - col_pct_null:
      columns: amount
      p: 0
      tol: 0.05

  - data_freshness:
      column: created_at
      max_age: 24 hours
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
  modifying existing plans with natural language (and reviewing the
  change before accepting it), and the interactive assistant() chat
  interface. Supports Anthropic, OpenAI, Ollama, Bedrock, Azure OpenAI,
  and Cohere providers. Use when bootstrapping validation for a new
  dataset or refining existing plans.
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

# Check that it parses and uses only real validation methods
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

| I want to...                             | Use                                   |
| ---------------------------------------- | ------------------------------------- |
| Generate a validation plan from data     | `DraftValidation`                     |
| Edit an existing plan with instructions  | `EditValidation`                      |
| Compare two plans without a model        | `EditValidation.from_plans()`         |
| Chat interactively about validation      | `assistant()`                         |
| See what the LLM generated               | `draft.code`, `edit.to_code()`        |
| Check generated code is valid            | `validate_syntax()`                   |
| See what changed in an edit              | `edit.diff()`, `edit.changed_steps()` |
| Review the changes as a table            | `edit.review()`                       |
| Accept an edit and get a Validate object | `edit.accept()`                       |

## Core concepts

### DraftValidation

Give data to an LLM and get back a validation plan:

```python
draft = pb.DraftValidation(
    data=df,
    model="anthropic:claude-sonnet-4-6",
    api_key=None,          # uses env var by default
    verify_ssl=True,
    max_reprompts=1,       # retries when the plan fails validate_syntax()
)
```

The model doesn't see the rows. It gets a JSON summary of the table
from `DataScan` (row and column counts, column names and types, missing
counts, summary statistics, and a few sample values), so large tables
work too.

```python
# The raw LLM response
draft.response

# The extracted Python code
draft.code

# Does the code parse, and does every col_*/row_*/rows_* method exist?
draft.validate_syntax()  # True/False

# The drafted steps as change records
draft.changed_steps()    # [{"action": "add", "method": ..., "new": ...}, ...]
```

The drafted code uses the placeholder `your_data` in `Validate(data=...)`
and ends with `.interrogate()`, so replace `your_data` with your table
before running it.

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

# From a YAML file (or a .py file, or a YAML string)
edit = pb.EditValidation(
    validation="validation.yaml",
    instruction="Add threshold warnings at 5%",
    model="anthropic:claude-sonnet-4-6",
)
```

Every input is turned into Pointblank code first (YAML through
`yaml_to_python()`), so the revised plan is always Python code with the
`your_data` placeholder.

Working with edits:

```python
# The revised plan as code
edit.to_code()

# A unified diff of the two plans' code
print(edit.diff())

# The changed steps (ignores reformatting); unchanged steps aren't listed
edit.changed_steps()
# [{'action': 'modify', 'method': 'col_vals_gt',
#   'old': "col_vals_gt(columns='d', value=100)",
#   'new': "col_vals_gt(columns='d', value=50)"}, ...]

# A great_tables table of the changes, colored by kind
edit.review()

# Accept the edit and get a Validate object (not yet interrogated)
validation = edit.accept()
validation.interrogate()
```

Supply data to the edit so the model sees a summary of the table (for
edits like choosing a new bound). That table is also the one `accept()`
uses when it's given none:

```python
edit = pb.EditValidation(
    validation=existing_validation,
    instruction="Add checks for the new columns",
    model="anthropic:claude-sonnet-4-6",
    data=updated_df,
)
```

`accept(data=...)` binds a table to `your_data`, so the same revised
plan can be run on other data. Before running anything it applies the
`validate_syntax()` check and raises a `ValueError` naming the problem
if it fails. It removes any `.interrogate()` call, runs the code with
`exec()`, and returns the `Validate` assigned to `validation`.

### Comparing two plans without a model

`EditValidation.from_plans()` builds an `EditValidation` from two plans
you already have, with no LLM, API key, or `chatlas` needed. It's
handy in code review for a change to a saved plan:

```python
small_table = pb.load_dataset("small_table")

original = pb.Validate(data=small_table).col_vals_gt(columns="d", value=100).rows_distinct()
revised = (
    pb.Validate(data=small_table)
    .col_vals_gt(columns="d", value=50)
    .col_vals_not_null(columns="a")
)

edit = pb.EditValidation.from_plans(original, revised, instruction="Loosen d; require a")

edit.review()                                       # instruction is the subtitle
validation = edit.accept(data=small_table).interrogate()
```

Either plan can be a `Validate` object, a code string, a YAML string,
or a `.py`/`.yaml` path. An object from `from_plans()` has no `data=`,
so pass the table to `accept()`.

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
model="anthropic:claude-opus-4-8"

# OpenAI
model="openai:gpt-6-luna"
model="openai:gpt-5.6-sol"

# Ollama (local)
model="ollama:llama3"
model="ollama:mistral"

# AWS Bedrock
model="bedrock:anthropic.claude-sonnet-5-5"

# Azure OpenAI (the part after the colon is the deployment id)
model="azure-openai:my-deployment-name"

# Cohere
model="cohere:command-a-03-2025"
```

### API key handling

By default, the API key is read from environment variables:

| Provider     | Environment variable                                           |
| ------------ | -------------------------------------------------------------- |
| Anthropic    | `ANTHROPIC_API_KEY`                                            |
| OpenAI       | `OPENAI_API_KEY`                                               |
| Ollama       | (no key needed)                                                |
| Bedrock      | AWS credentials                                                |
| Azure OpenAI | `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `OPENAI_API_VERSION` |
| Cohere       | `CO_API_KEY` (or `COHERE_API_KEY`)                             |

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
4. Check it with `draft.validate_syntax()`.
5. Copy the code into your project, replace `your_data`, and customize.
6. Run `interrogate()` and iterate.

### Iterating on a validation plan

1. Start with a draft or existing validation.
2. Use `EditValidation` with natural language instructions.
3. Review changes with `edit.review()` or `edit.diff()`.
4. Accept with `edit.accept()` or iterate with another edit.

### Reviewing a change to a saved plan

1. Load the old and new versions (files, YAML, or `Validate` objects).
2. Run `pb.EditValidation.from_plans(old, new)`.
3. Read `changed_steps()` or `review()`.

### Interactive exploration

1. Start `pb.assistant(model="...", data=df)`.
2. Ask questions about your data and validation needs.
3. Copy suggested code into your project.

## Gotchas

1. **LLM output is not guaranteed correct.** Always review generated
   code before using in production.
2. **`validate_syntax()` checks that the code parses and that every
   `col_*`/`row_*`/`rows_*` method exists.** It doesn't run the code,
   so a wrong argument name or value is only caught by `accept()` or
   at interrogation.
3. **`max_reprompts` controls retries.** When the plan fails that
   check, the model is told the problem and asked again, up to this
   many times. Each retry is another request; `0` takes the first reply.
4. **`accept()` runs the revised code with `exec()`.** Read it with
   `to_code()` or `review()` first, as you would any code you didn't
   write.
5. **`accept()` returns an uninterrogated Validate object.** Call
   `.interrogate()` to execute. It needs a table, from `accept(data=)`
   or the `EditValidation`'s `data=`.
6. **Ollama runs locally.** No API key needed but the model must be
   downloaded first with `ollama pull`.
7. **The assistant requires a running display.** `"browser"` opens a
   web interface; `"terminal"` uses the console. Neither works in
   non-interactive environments.
8. **The model sees a summary, not the rows.** Bounds and sets it
   suggests come from summary statistics and a few sample values, so
   check them against the data.

## Related skills

| Skill            | When to use it                              |
| ---------------- | ------------------------------------------- |
| pointblank       | Full Validate workflow after drafting       |
| write-validation | Manual validation plan composition          |
| scan-and-profile | Profile data before asking the LLM to draft |
| validate-yaml    | Keep the accepted plan as YAML              |
````

### references/providers-reference.md

``` gd-skills-raw
```

```` markdown
# LLM providers reference

Configuration for each supported LLM provider. Everything is installed
with `pip install "pointblank[generate]"` (chatlas, the Anthropic SDK
with Bedrock support, the OpenAI SDK, and Shiny for the assistant).

## Provider model strings

Format: `"provider:model_name"`

### Anthropic

```python
model="anthropic:claude-sonnet-4-6"
model="anthropic:claude-opus-4-8"
```

Environment variable: `ANTHROPIC_API_KEY`

### OpenAI

```python
model="openai:gpt-6-luna"
model="openai:gpt-5.6-sol"
```

Environment variable: `OPENAI_API_KEY`

### Ollama (local)

```python
model="ollama:llama3"
model="ollama:mistral"
model="ollama:codellama"
```

No API key needed. Requires Ollama running locally.

Setup:
```bash
# Install Ollama: https://ollama.ai
ollama pull llama3
```

### AWS Bedrock

```python
model="bedrock:anthropic.claude-sonnet-5-5"
```

Uses AWS credentials from the environment (AWS_ACCESS_KEY_ID,
AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION) or AWS profiles; `api_key`
and `verify_ssl` aren't used. Bedrock goes through `anthropic[bedrock]`,
which brings in boto3.

### Azure OpenAI

```python
model="azure-openai:my-deployment-name"   # the Azure deployment id
```

Environment variables:
- `AZURE_OPENAI_API_KEY` (or pass `api_key=`)
- `AZURE_OPENAI_ENDPOINT` (e.g., `https://<resource>.openai.azure.com`)
- `OPENAI_API_VERSION` (e.g., `2024-06-01`)

A missing endpoint or API version raises a `ValueError`.

### Cohere

```python
model="cohere:command-a-03-2025"
```

Environment variable: `CO_API_KEY` (or `COHERE_API_KEY`). Requests go
to Cohere's OpenAI-compatible API, so this uses the OpenAI SDK.

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
| Azure OpenAI       | yes | yes | yes |
| Cohere             | yes | yes | yes |
| `api_key` param    | yes | yes | yes |
| `verify_ssl` param | yes | yes | no  |
| `max_reprompts`    | yes | yes | no  |
| `data` param       | required | optional | optional |
| Browser display    | no  | no  | yes |
| Terminal display   | no  | no  | yes |

`EditValidation.from_plans()` doesn't call a model, so it needs no
provider at all.
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
  enforcing structural and semantic expectations at data boundaries,
  plus YAML serialization and import/export of external contract
  formats (JSON Schema, Frictionless, dbt, ODCS). Use when setting up
  source/target contracts, pipeline validation, or contract files.
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
    on_violation="raise",  # applied when a Pipeline enforces the contract
)

# Validate data against the contract (returns an interrogated Validate;
# never warns or raises, whatever on_violation is)
validation = contract.validate(data=df)
validation.assert_passing()  # raise if any test unit failed
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
| Convert to JSON Schema, dbt, etc.     | `export_contract()`    |
| Read JSON Schema, dbt, ODCS, etc.     | `import_contract()`    |
| Add reusable steps to a contract      | `to_validate()` + `add_steps()` |
| Warn on violation without stopping    | `on_violation="warn"`  |
| Raise an exception on violation       | `on_violation="raise"` |
| Log violations instead of warning     | `on_violation="log"`   |

## Core concepts

### Contract

A `Contract` bundles:

- **name** -- identifier for the contract
- **direction** -- `"source"` (incoming data) or `"target"` (output)
- **schema** -- expected column names and types
- **steps** -- list of `Step` objects defining semantic checks
- **on_violation** -- what a `Pipeline` does when validation fails

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

The `method` argument names a Validate method, and the keyword
arguments are passed to it. A contract accepts only these methods
(any other name raises a `ValueError` when the contract is created):
the `col_vals_*` comparison, range, set, null, regex, `within_spec`,
`increasing`/`decreasing`, and `expr` methods; `col_exists`,
`col_pct_null`, `col_schema_match`, `col_count_match`,
`row_count_match`, `rows_distinct`, `rows_complete`; and `conjointly`,
whose expressions go in a list:
`pb.Step("conjointly", exprs=[lambda df: pl.col("a") > 0])`.

For other checks (aggregates, `data_freshness`, `tbl_match`, text
checks, ...), compile the contract and chain them on, or add a
`pb.Steps` bundle:

```python
v = (
    contract.to_validate(data=df)
    .add_steps(pb.Steps().col_vals_not_null(columns="email"))
    .data_freshness(column="updated_at", max_age="24 hours")
    .interrogate()
)
```

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

# Take the names and types from an existing table
schema = pb.Schema(tbl=df)
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

`on_violation` applies only when a `Pipeline` enforces the contract
(`run()`, `validate_source()`, `validate_target()`). A boundary fails
when any test unit fails, whatever the thresholds.

| Value     | Behavior                                                   |
| --------- | ---------------------------------------------------------- |
| `"warn"`  | Issue a `UserWarning` (default)                            |
| `"raise"` | Raise `ContractViolationError` (a `RuntimeError`)          |
| `"log"`   | Write a warning to the `"pointblank.contract"` logger      |

The error has `contract`, `validation`, and (from `run()`) `result`
attributes, the last being the `PipelineResult` of the run.

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
result.passed                 # True if every validated boundary passed
result.source_passed          # True if source contract passed
result.target_passed          # True if target contract passed
result.source_validation      # Validate object for source
result.target_validation      # Validate object for target
result.transform_output       # the transformed data
result.get_report()           # summary report string
```

### Serialization

```python
# Save contract to YAML (also returns the YAML string)
yaml_text = contract.to_yaml("contracts/orders-source.yaml")

# Load contract from YAML
contract = pb.Contract.from_yaml("contracts/orders-source.yaml")

# Dictionary round-trip
d = contract.to_dict()
contract = pb.Contract.from_dict(d)

# Pipeline serialization
pipeline.to_yaml("pipelines/order-cleaning.yaml")
pipeline = pb.Pipeline.from_yaml("pipelines/order-cleaning.yaml")
```

Step arguments must be plain data to be written to YAML; a lambda, an
expression, or a column selector makes `to_yaml()` raise a
`ValueError`.

### Other contract formats

```python
# To JSON Schema, Frictionless Table Schema, dbt, or ODCS (a dict, or
# written to a file with a second `destination` argument); a Validate
# object can be exported too
spec = pb.export_contract(contract, format="json_schema")   # also "frictionless", "dbt", "odcs"

# From one of those formats (a dict, a string, or a file path)
imported = pb.import_contract("orders.schema.json", format="json_schema")
print(imported.summary())          # columns found, constraints mapped, coverage
contract = imported.to_contract(name="orders")
```

Rules with no equivalent in the other format are left out of an
export, and the import summary reports what couldn't be mapped.

## Workflows

### Setting up a new contract

1. Profile the data with `pb.DataScan(data=df)` to understand its
   shape, types, and distributions.
2. Take a starting schema from the data: `schema = pb.Schema(tbl=df)`.
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
2. **`on_violation` only acts in a `Pipeline`.** `contract.validate()`
   returns the validation without warning or raising; check it with
   `all_passed()` or `assert_passing()`.
3. **`short_circuit=True` skips the transform and target validation**
   if source validation fails. Set to `False` to always run both (a
   source with `on_violation="raise"` then raises after the target
   is validated).
4. **A pipeline's `thresholds=` replace the contracts' thresholds**;
   they aren't merged. Thresholds don't decide whether a boundary
   passes: any failing test unit fails it.
5. **Schema type strings are backend-specific.** Use the dtype names
   from your backend (e.g., `"Int64"` for Polars, `"int64"` for
   Pandas).
6. **`to_validate()` does not call `interrogate()`.** Call it yourself
   if you add steps. Use `validate()` for automatic interrogation.
7. **Only some methods can be contract steps.** An unsupported or
   misspelled method name raises a `ValueError` when the `Contract` is
   created; a misspelled argument raises a `TypeError` when it's
   compiled into a validation.
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

| Method                 | Returns        | Description                              |
|------------------------|----------------|------------------------------------------|
| `validate(data)`       | `Validate`     | Compile and interrogate (never raises)   |
| `to_validate(data)`    | `Validate`     | Compile without interrogating            |
| `to_dict()`            | `dict`         | Serialize to dictionary                  |
| `from_dict(cls, data)` | `Contract`     | Deserialize from dictionary              |
| `from_yaml(cls, path)` | `Contract`     | Load from YAML file                      |
| `to_yaml(path=None)`   | `str`          | YAML string (also written to `path`)     |

The compiled validation has `tbl_name=` set to the contract name, the
label `"Contract: <name> v<version>"`, the contract's thresholds,
owner, consumers, and version, a `col_schema_match()` step first (if
there's a schema, with its strict defaults), and then the steps.

## Step constructor

```python
pb.Step(method: str, **kwargs)
```

`method` is the name of a Validate method (e.g. `"col_vals_gt"`,
`"rows_distinct"`), and the keyword arguments are passed to it. The
step's parts are its `method` and `kwargs` attributes. `conjointly`
takes its expressions as a list in `exprs=`:

```python
pb.Step("conjointly", exprs=[lambda df: pl.col("a") > 0, lambda df: pl.col("a") < 10])
```

## Step methods

| Method                 | Returns    | Description                |
|------------------------|------------|----------------------------|
| `to_dict()`            | `dict`     | Serialize to dictionary    |
| `from_dict(cls, data)` | `Step`     | Deserialize from dictionary|

## Valid step methods

A `Contract` accepts only these (others raise a `ValueError` when the
contract is created):

- `col_vals_gt`, `col_vals_lt`, `col_vals_ge`, `col_vals_le`,
  `col_vals_eq`, `col_vals_ne`
- `col_vals_between`, `col_vals_outside`
- `col_vals_in_set`, `col_vals_not_in_set`
- `col_vals_null`, `col_vals_not_null`
- `col_vals_regex`, `col_vals_within_spec`, `col_vals_expr`
- `col_vals_increasing`, `col_vals_decreasing`
- `col_exists`, `col_pct_null`, `col_schema_match`
- `col_count_match`, `row_count_match`
- `rows_distinct`, `rows_complete`
- `conjointly`

For anything else, chain it onto `contract.to_validate(data)` or use
`add_steps()` with a `pb.Steps` bundle (which can hold any method).

## YAML format

`to_yaml()` writes the contract under a `contract:` key. Unset fields
and the default `direction`/`on_violation` are left out; each step is a
mapping of the method name to its arguments:

```yaml
contract:
  name: orders-source
  version: '1.0'
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
  - col_vals_not_null:
      columns: order_id
  - col_vals_gt:
      columns: amount
      value: 0
  - col_vals_in_set:
      columns: status
      set: [pending, shipped, delivered]
  - rows_distinct:
      columns_subset: [order_id]
  thresholds:
    warning: 0.01
    error: 0.05
```

Step arguments must be plain data (strings, numbers, booleans, dates,
lists, dicts); step-level `Thresholds` and tuples are converted
automatically. A lambda, expression, or column selector makes
`to_yaml()` raise a `ValueError` naming where it is.

## Import and export

```python
pb.export_contract(contract_or_validation, destination=None, *, format=...)  # -> dict or str
pb.import_contract(source, *, format=None)                                   # -> ContractImport
```

Formats: `"json_schema"`, `"frictionless"`, `"dbt"`, `"odcs"`.
`ContractImport` has `summary()`, `coverage`, `to_contract(name=...)`,
`to_validate(...)`, `to_python()`, and `to_yaml()`.

## on_violation behavior

Applied only by `Pipeline.run()`, `validate_source()`, and
`validate_target()` when a boundary has any failing test unit:

| Value     | On failure...                                              |
|-----------|------------------------------------------------------------|
| `"warn"`  | Issues a `UserWarning`, continues                          |
| `"raise"` | Raises `ContractViolationError` (a `RuntimeError`)         |
| `"log"`   | Writes a warning to the `"pointblank.contract"` logger     |

`ContractViolationError` has `contract`, `validation`, and `result`
attributes (`result` is the `PipelineResult` when raised by `run()`,
otherwise `None`).
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
    target: Contract | None = None,       # at least one of source/target
    thresholds: Thresholds | None = None, # replaces the contracts' thresholds
    actions: Actions | None = None,       # used for both boundaries
    final_actions: FinalActions | None = None,  # run once per boundary
    label: str | None = None,             # validations get "<label> [source]" etc.
    short_circuit: bool = True,
)
```

## Pipeline methods

| Method                   | Returns          | Description                       |
|--------------------------|------------------|-----------------------------------|
| `run(data, transform)`   | `PipelineResult` | Source check + transform + target check |
| `validate_source(data)`  | `Validate`       | Run source contract only          |
| `validate_target(data)`  | `Validate`       | Run target contract only          |
| `to_dict()`              | `dict`           | Serialize to dictionary           |
| `from_dict(cls, data)`   | `Pipeline`       | Deserialize from dictionary       |
| `from_yaml(cls, path)`   | `Pipeline`       | Load from YAML file               |
| `to_yaml(path=None)`     | `str`            | YAML string (also written to `path`) |

All three running methods apply the failing contract's `on_violation`.

## PipelineResult

| Attribute / Method    | Type              | Description                    |
|-----------------------|-------------------|--------------------------------|
| `source_validation`   | `Validate \| None`| Source validation result       |
| `target_validation`   | `Validate \| None`| Target validation (None if skipped) |
| `transform_output`    | `Any`             | Output of the transform (None if skipped) |
| `source_passed`       | `bool`            | Source contract passed?        |
| `target_passed`       | `bool`            | Target contract passed?        |
| `passed`              | `bool`            | All validated boundaries passed? |
| `get_report()`        | `str`             | Summary report text            |

A boundary passes only if every test unit passes (thresholds don't
matter). A boundary that wasn't validated counts as passing, so a
target skipped after a failing source has `target_passed=True` while
`passed` is `False`.

## Pipeline execution flow

```
1. Validate source contract against input data
   |
   +-- If source fails and short_circuit=True -> apply on_violation, stop
   |
2. Run transform(data) -> transformed_data
   |
3. Validate target contract against transformed_data
   |
4. Apply on_violation for any failing boundary; return PipelineResult
```

## short_circuit behavior

| `short_circuit` | Source fails                     | Source passes          |
|-----------------|----------------------------------|------------------------|
| `True`          | Skip transform + target          | Run transform + target |
| `False`         | Run transform + target; a `"raise"` source raises after the target | Run transform + target |

## YAML format

```yaml
pipeline:
  label: Order cleaning pipeline
  short_circuit: true
source:
  name: raw-orders
  on_violation: raise
  schema:
    id: Int64
    amount: Float64
  steps:
  - col_vals_not_null:
      columns: id
target:
  name: clean-orders
  direction: target
  schema:
    id: Int64
    amount: Float64
    is_valid: Boolean
  steps:
  - col_vals_gt:
      columns: amount
      value: 0
```

## Patterns

### Source-only pipeline (ingestion guard)

```python
pipeline = pb.Pipeline(source=source_contract)
validation = pipeline.validate_source(df)
```

### Target-only pipeline (output guard)

```python
pipeline = pb.Pipeline(target=target_contract)
result = pipeline.run(data=raw_df, transform=my_transform)
```

### Handling a raised violation

```python
try:
    result = pipeline.run(data=df, transform=transform)
except pb.ContractViolationError as e:
    print(e.contract.name)
    e.validation.get_tabular_report()
    result = e.result           # the PipelineResult of the run
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
  rules. Covers DataScan for column-level statistics, comparing a scan
  with a saved baseline to find drift (DataScan.compare), schemas taken
  from a table, missing values analysis with missing_vals_tbl(), and
  table previewing. Use when exploring a new dataset, understanding
  data distributions before validation, or checking a refreshed table
  against an earlier profile.
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
    - drift
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

| I want to...                         | Use                                     |
| ------------------------------------ | --------------------------------------- |
| Get a full column-level profile      | `DataScan`                              |
| See column types and basic stats     | `DataScan.get_tabular_report()`         |
| Export a profile as JSON or a dict   | `to_json()`, `save_to_json()`, `to_dict()` |
| Bring back a saved profile           | `DataScan.load_from_json()`, `from_json()`, `from_dict()` |
| Compare a table's profile with a baseline | `DataScan.compare()`               |
| Find which rows of two tables differ | `pb.compare()` (rows, not profiles)     |
| Get a schema for `col_schema_match`  | `Schema(tbl=df)`                        |
| Infer constraints for generating data | `schema_from_tbl()`, `Schema.from_table()` |
| See a quick preview of the table     | `preview()`                             |
| Analyze missing values               | `missing_vals_tbl()`                    |
| Get row/column counts                | `get_row_count()`, `get_column_count()` |

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

# The profile as a Polars DataFrame, one row per column
scan.summary_data

# Export
json_str = scan.to_json()
scan.save_to_json("profile_output.json")
profile = scan.to_dict()      # {"metadata": ..., "columns": ...}
```

`summary_data` has these columns: `colname`, `coltype`, `icon`,
`n_missing`, `n_unique`, `mean`, `std`, `min`, `p05`, `q_1`, `median`,
`q_3`, `p95`, `max`, `iqr`, `freqs`, and `sample_data`. For a string
column the statistics are of the string lengths; `freqs` holds the
value counts of a Boolean column.

### Comparing with a baseline

A table that's refreshed regularly can change in ways no single row
shows: a column dropped, a type changed, values shifted. Save a
profile of a version known to be good, then compare later scans with
it. The scan you call `compare()` on is the current one:

```python
pb.DataScan(data=good_df).save_to_json("profile_baseline.json")

baseline = pb.DataScan.load_from_json("profile_baseline.json")
current = pb.DataScan(data=new_df)

diff = current.compare(baseline)       # a DataScanDiff

diff.has_changes                       # True if anything differs
diff.row_count_diff                    # (baseline rows, current rows)
diff.columns_added                     # only in the current scan
diff.columns_removed                   # only in the baseline
diff.columns_type_changed
diff.column_diffs                      # per column: status, stat_diffs, drift_scores
diff.to_dict()
diff.get_tabular_report()
```

Each statistic is compared exactly, so any change is reported with
both values. Drift scores show how far the distribution moved: numeric
columns get `psi` (population stability index), `ks_statistic`, and
`ks_p_value`, and other columns get a `psi` from their value counts. A
PSI below 0.1 is usually read as little shift, 0.1 to 0.25 as moderate,
and above 0.25 as large.

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

# With coded missing values, keyed by column name
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

### Schemas from a table

There are two kinds, for two jobs:

```python
# Column names and dtypes, for col_schema_match() and contracts
schema = pb.Schema(tbl=df)
schema.get_column_list()   # ["id", "name", "amount"]
schema.get_dtype_list()    # ["Int64", "String", "Float64"]

# Field objects with constraints inferred from the values, for
# generating synthetic data with schema.generate()
gen_schema = pb.schema_from_tbl(
    df,
    infer_constraints=True,       # ranges, null rates, uniqueness, allowed sets
    categorical_threshold=20,     # <= 20 distinct values becomes `allowed=`
    detect_presets=True,          # match email, URL, etc. by name and values
    sample_size=None,             # sample rows first (None = all)
)
synthetic = gen_schema.generate(n=1000, seed=23)
```

`pb.Schema.from_table()` is the same as `schema_from_tbl()`.

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
5. Take a schema: `schema = pb.Schema(tbl=df)`.
6. Use the profile to inform validation rules.

### From profile to validation plan

1. Profile the data with `DataScan`.
2. Note columns with high missingness -- add `col_pct_null` checks.
3. Note columns with few distinct values -- add `col_vals_in_set`.
4. Note numeric ranges -- add `col_vals_between` checks.
5. Use `pb.Schema(tbl=df)` in `col_schema_match`.
6. Build the validation plan with the `write-validation` skill.

### Watching a refreshed table for drift

1. Scan a version known to be good and `save_to_json()` it.
2. On each refresh, scan the new data and `compare()` it with
   `DataScan.load_from_json(...)`.
3. Check `diff.has_changes`, the added, removed, and type-changed
   columns, and the drift scores; show `diff.get_tabular_report()`.
4. Turn anything that matters into a validation step.

## Gotchas

1. **DataScan reads the full table.** For large datasets, consider
   sampling first.
2. **Schema type names are backend-specific.** Polars uses `"Int64"`,
   Pandas uses `"int64"`, and Ibis uses `"int64"`. `Schema(tbl=df)`
   gets the right names automatically.
3. **`schema_from_tbl()` is for generating data, not for
   `col_schema_match()`.** Its fields hold short dtypes (`"Datetime"`
   rather than `"Datetime(time_unit='us', time_zone=None)"`), so a
   schema match with the default `full_match_dtypes=True` can fail on
   the same table. Use `Schema(tbl=df)` to validate.
4. **A restored profile has no values.** After `load_from_json()`,
   `from_json()`, or `from_dict()`, `compare()` still compares the
   statistics, but only Boolean columns get drift scores. Compare two
   scans made from data to get them for every column.
5. **`compare()` direction matters.** Call it on the current scan and
   pass the baseline; "added" means in the current scan only.
6. **`missing_vals_tbl` only shows null by default.** Pass
   `MissingSpec` definitions to include sentinel values.
7. **`preview()` returns a GT table object.** In notebooks it renders
   automatically; in scripts, you may need to display it.

## Related skills

| Skill               | When to use it                                   |
| ------------------- | ------------------------------------------------ |
| pointblank          | Full Validate workflow overview                  |
| write-validation    | Build validation plans from profile insights     |
| generate-data       | Create synthetic data matching a schema          |
| draft-validation    | Have an LLM draft a plan from the profile        |
| check-training-data | Drift between train/test splits (`check_splits`) |
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

| Property       | Type              | Description                        |
|----------------|-------------------|------------------------------------|
| `summary_data` | Polars DataFrame  | The profile, one row per column    |

## Methods

| Method                                        | Returns        | Description                       |
|-----------------------------------------------|----------------|-----------------------------------|
| `get_tabular_report(*, show_sample_data=False)` | `GT`         | HTML summary table                |
| `to_json()`                                   | `str`          | JSON string of the profile        |
| `save_to_json(output_file)`                   | `None`         | Write the JSON to a file          |
| `to_dict()`                                   | `dict`         | `{"metadata": ..., "columns": ...}` |
| `DataScan.from_json(json_string)`             | `DataScan`     | Restore from a JSON string        |
| `DataScan.load_from_json(input_file)`         | `DataScan`     | Restore from a JSON file          |
| `DataScan.from_dict(d)`                       | `DataScan`     | Restore from `to_dict()` output   |
| `compare(baseline)`                           | `DataScanDiff` | Differences from a baseline scan  |

The metadata has `table_name`, `row_count`, `columns`, and
`skipped_columns`.

## Columns of summary_data

| Column        | Description                                         |
|---------------|-----------------------------------------------------|
| `colname`     | Column name                                         |
| `coltype`     | Column data type                                    |
| `icon`        | Type icon used in the report                        |
| `n_missing`   | Count of missing values                             |
| `n_unique`    | Count of distinct values                            |
| `mean`        | Mean                                                |
| `std`         | Standard deviation                                  |
| `min`         | Minimum                                             |
| `p05`         | 5th percentile                                      |
| `q_1`         | First quartile                                      |
| `median`      | Median                                              |
| `q_3`         | Third quartile                                      |
| `p95`         | 95th percentile                                     |
| `max`         | Maximum                                             |
| `iqr`         | Interquartile range                                 |
| `freqs`       | Value counts (Boolean columns)                      |
| `sample_data` | A few sample values                                 |

For string columns the statistics describe the string lengths.

## DataScanDiff

Returned by `current_scan.compare(baseline_scan)`:

| Attribute / Method        | Description                                          |
|---------------------------|------------------------------------------------------|
| `has_changes`             | `True` if anything differs                           |
| `row_count_diff`          | `(baseline rows, current rows)`                      |
| `columns_added`           | Columns only in the current scan                     |
| `columns_removed`         | Columns only in the baseline                         |
| `columns_type_changed`    | Columns whose type differs                           |
| `column_diffs`            | One record per column: `colname`, `coltype_baseline`, `coltype_current`, `status` (`unchanged`, `changed`, `type_changed`, `added`, `removed`), `stat_diffs` (`{stat: (baseline, current)}`), `drift_scores` |
| `to_dict()`               | `row_count`, `columns_added`, `columns_removed`, `columns_type_changed`, `stat_diffs`, `drift_scores` |
| `get_tabular_report()`    | GT report of the differences                         |

Drift scores (zero scores are left out):

| Column kind | Scores                                         |
|-------------|------------------------------------------------|
| Numeric     | `psi` (ten bins at the baseline's deciles; needs at least ten values on each side), `ks_statistic`, `ks_p_value` (SciPy if installed, else an approximation) |
| Other       | `psi` from the frequency of each distinct value |

Drift scores need the values, so a scan restored from JSON or a dict
gives them only for Boolean columns.

## Shortcut function

```python
pb.col_summary_tbl(data=df, tbl_name="my_table")
```

Equivalent to `DataScan(data, tbl_name).get_tabular_report()`.

## missing_vals_tbl

```python
pb.missing_vals_tbl(
    data: Any,
    missing: dict[str, MissingSpec] | None = None,   # keyed by column name
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
    columns_subset: str | list[str] | Column | None = None,
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

There are two ways to get a schema from a table:

| Call                     | Holds                              | Use it for                        |
|--------------------------|------------------------------------|-----------------------------------|
| `pb.Schema(tbl=df)`      | Column names and full dtype strings | `col_schema_match()`, contracts  |
| `pb.schema_from_tbl(df)` | Field objects with inferred constraints | `schema.generate()`, `generate_dataset()` |

## Schema(tbl=...)

```python
schema = pb.Schema(tbl=df)
schema.get_column_list()   # ['date_time', 'date', 'a', ...]
schema.get_dtype_list()    # ["Datetime(time_unit='us', time_zone=None)", 'Date', 'Int64', ...]
```

The dtypes are the backend's own names (Polars `Int64`, Pandas
`int64`, Ibis `int64`), so the schema matches the table it came from.

## schema_from_tbl

```python
pb.schema_from_tbl(
    tbl: Any,
    *,
    infer_constraints: bool = True,
    categorical_threshold: int | float = 20,
    detect_presets: bool = True,
    sample_size: int | None = None,
) -> Schema
```

Creates a Schema of Field objects (`IntField`, `StringField`, ...)
whose constraints come from the values, ready for generating
synthetic data that looks like the table.

### Parameters

| Parameter                | Default | Description                         |
|--------------------------|---------|-------------------------------------|
| `infer_constraints`      | `True`  | Infer min/max, null rates, uniqueness, allowed sets; `False` behaves like `Schema(tbl=df)` |
| `categorical_threshold`  | `20`    | Max distinct values (int) or fraction of rows (float in 0-1) for a categorical `allowed=` list |
| `detect_presets`         | `True`  | Match string columns to presets (email, URL, phone) by name and values |
| `sample_size`            | `None`  | Rows to sample first (None = all)   |

```python
gen_schema = pb.schema_from_tbl(df)
synthetic = gen_schema.generate(n=1000, seed=23)
```

## Schema.from_table

Class method with the same parameters as `schema_from_tbl()`:

```python
schema = pb.Schema.from_table(df, sample_size=1000)
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
| `get_dtype_list()`              | `list`         | Dtype strings (Field objects for an inferred schema) |
| `get_tbl_type()`                | `str`          | Table type the schema came from |
| `get_schema_coerced(to=None)`   | `Schema`       | Coerced to a target backend  |
| `generate(n=100, seed=None, output="polars", country="US", shuffle=True, weighted=True)` | DataFrame or dict | Generate synthetic data |

## Common dtype strings by backend

| Concept   | Polars       | Pandas       |
|-----------|-------------|--------------|
| Integer   | `Int8/16/32/64`, `UInt8/16/32/64` | `int8/16/32/64`, `uint8/16/32/64` |
| Float     | `Float32/64` | `float32/64` |
| String    | `String`     | `object`, `string` |
| Boolean   | `Boolean`    | `bool`       |
| Date      | `Date`       | `datetime64[ns]` |
| Datetime  | `Datetime(time_unit='us', time_zone=None)` | `datetime64[ns]` |
| Duration  | `Duration`   | `timedelta64[ns]` |

## Using a table's schema in validation

```python
schema = pb.Schema(tbl=df)

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
| `full_match_dtypes`        | `True`  | Exact dtype match required (`False` allows a substring, so `"Int"` matches `"Int64"`) |
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
  constraints, profile_fields() for person data, schemas inferred from
  a real table, and check_synthetic() for comparing synthetic data with
  the real data it imitates. Use when creating test data, fixtures, or
  synthetic datasets for validation testing.
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
| Mix several countries               | `generate(country=["DE", "FR"])`      |
| Imitate an existing table           | `schema_from_tbl(df).generate()`      |
| Check synthetic data against real   | `check_synthetic()`                   |
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
| `output`   | `"polars"` | `"polars"`, `"pandas"`, or `"dict"`     |
| `country`  | `"US"`     | A country code, a list of them, or a dict of codes to weights |
| `shuffle`  | `True`     | Shuffle rows after generation           |
| `weighted` | `True`     | Use weighted distributions              |

### generate_dataset() convenience function

```python
df = pb.generate_dataset(schema, n=500, seed=42)
```

It takes the same parameters as `generate()`.

### Nullable columns

Any field type supports nulls (`null_probability` needs `nullable=True`):

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
pb.string_field(preset="address")        # full addresses
pb.string_field(preset="city")           # city names
pb.string_field(preset="postcode")       # postal codes
pb.string_field(preset="phone_number")   # phone numbers
pb.string_field(preset="company")        # company names
pb.string_field(preset="job")            # job titles
pb.string_field(preset="url")            # URLs
pb.string_field(preset="uuid4")          # UUIDs
pb.string_field(preset="iban")           # IBANs
pb.string_field(preset="ssn")            # SSNs
```

Presets produce country-specific data when `country` is set, and the
presets in a row agree with each other (the email is built from the
name, the city, state, and postcode go together). See
`references/presets-reference.md` for the full list; an unknown preset
raises a `ValueError` that lists them.

### Profile fields

Generate person-related fields as a group:

```python
fields = pb.profile_fields(
    set="standard",        # "minimal", "standard" or "full"
    split_name=True,       # first_name + last_name vs full name
    include=None,          # specific fields to include
    exclude=None,          # specific fields to exclude
    prefix=None,           # prefix for field names (each prefix is a separate person)
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

Over 100 countries supported (ISO 3166-1 alpha-2 codes):

```python
# German names, addresses, phone numbers
df = schema.generate(n=100, country="DE")

# Japanese
df = schema.generate(n=100, country="JP")

# Brazilian
df = schema.generate(n=100, country="BR")

# A mix: rows spread evenly over the countries, or by weight
df = schema.generate(n=100, country=["DE", "FR"])
df = schema.generate(n=100, country={"JP": 0.7, "BR": 0.3})
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
# Infer field constraints from real data
schema = pb.schema_from_tbl(
    production_df,
    infer_constraints=True,
    categorical_threshold=20,
)

# Generate synthetic version
fixture = schema.generate(n=100, seed=1)
```

`pb.Schema.from_table()` is the same as `schema_from_tbl()`. Each field
gets its range, null rate, uniqueness, and (for few distinct values) an
`allowed=` list from the data, and string columns are matched to
presets by name and values.

### Checking synthetic data against the real data

`check_synthetic()` compares synthetic data with its seed (the real
data it imitates) for fidelity (each column's distribution, by PSI and
KS), diversity (repeated rows, and distinct-n for `text=` columns), and
copies of seed rows (exact, or near copies of `text=` columns):

```python
real = pb.load_dataset("small_table")
synthetic = pb.schema_from_tbl(real).generate(n=50, seed=1)

check = pb.check_synthetic(synthetic, real)

check.ok                    # False if there's any issue
check.issues                # e.g. "The distribution of 'd' ... differs ... (PSI 2.65)."
check.fidelity              # per-column psi, ks, level, missing shares
check.get_tabular_report()
check.get_copies()          # synthetic rows that copy a seed row
```

Inferred fields keep each column's range but not its shape, so numeric
columns often fail fidelity; tighten those fields (or use `allowed=`)
when the distribution matters. See the `check-training-data` skill for
more on synthetic data used for training.

## Gotchas

1. **`unique=True` needs a large enough range.** If `max_val - min_val`
   < `n`, generation will fail for integer fields.
2. **Only one of `preset`, `pattern`, `allowed` per StringField.**
   They are mutually exclusive.
3. **Presets don't need Faker.** Their data ships with Pointblank,
   so no extra install is needed.
4. **`seed` makes generation reproducible for a given `n`.** With a
   larger `n`, most columns start with the smaller dataset's rows, but
   not all (`email` addresses come out differently), so don't rely on
   a smaller dataset being the first rows of a larger one.
5. **`output` is `"polars"`, `"pandas"`, or `"dict"`.** For other
   formats, convert after generation.
6. **`null_probability` needs `nullable=True`.** A probability above 0
   without it raises a `ValueError`, and `nullable=True` with the
   default `null_probability=0.0` generates no nulls.

## Related skills

| Skill               | When to use it                                   |
| ------------------- | ------------------------------------------------ |
| pointblank          | Validate the generated data                      |
| scan-and-profile    | Profile the real table the schema imitates       |
| write-validation    | Write the plan that the test data exercises      |
| check-training-data | Check synthetic data used for training a model   |
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

A `null_probability` above 0 needs `nullable=True` (otherwise a
`ValueError` is raised).

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

Only one of `preset`, `pattern`, or `allowed` may be set. See
`presets-reference.md` for the presets.

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
    set: str = "standard",        # "minimal", "standard" or "full"
    split_name: bool = True,      # first_name + last_name vs name
    include: list[str] | None = None,
    exclude: list[str] | None = None,
    prefix: str | None = None,
) -> dict[str, StringField]
```

Returns a dict of StringField objects suitable for `**`-unpacking
into `Schema()`.

Minimal set: first_name + last_name (or name), email, phone_number.

Standard set: first_name + last_name (or name), email, city, state,
postcode, phone_number.

Full set: the standard set plus address, company, job.

Each `prefix=` makes a separate person: two prefixed profiles in one
schema (e.g., `customer_` and `seller_`) get different names,
locations and employers in each row. Other preset columns whose names
start with the prefix belong to that profile.
````

### references/presets-reference.md

``` gd-skills-raw
```

```` markdown
# String presets reference

Available preset values for `string_field(preset="...")`. The data
behind them ships with Pointblank (no Faker needed) and is
country-aware. An unknown preset raises a `ValueError` listing these.

The example outputs are for `country="US"`.

## Person and contact

| Preset          | Example output                      |
|-----------------|-------------------------------------|
| `name`          | "Kenneth Cole"                      |
| `name_full`     | "Mr. Kenneth Cole"                  |
| `first_name`    | "Kenneth"                           |
| `last_name`     | "Cole"                              |
| `gender`        | "male"                              |
| `email`         | "kenneth.cole50@mail.com"           |
| `phone_number`  | "(423) 886-9589"                    |
| `user_name`     | "kenneth_cole"                      |

## Location

| Preset           | Example output                                   |
|------------------|--------------------------------------------------|
| `address`        | "7705 Chapel Avenue, Suite 407, Nashville, Tennessee 37287" |
| `city`           | "Nashville"                                      |
| `state`          | "Tennessee"                                      |
| `postcode`       | "37287"                                          |
| `country`        | "United States"                                  |
| `country_code_2` | "US"                                             |
| `country_code_3` | "USA"                                            |
| `latitude`       | "36.218438"                                      |
| `longitude`      | "-86.968814"                                     |
| `locale_code`    | "en_US"                                          |

## Business

| Preset          | Example output                              |
|-----------------|---------------------------------------------|
| `company`       | "Classic Data"                              |
| `job`           | "Software Engineer"                         |
| `catch_phrase`  | "Seamless technologies that revolutionize"  |

## Internet

| Preset          | Example output                               |
|-----------------|----------------------------------------------|
| `url`           | "https://www.been.site"                      |
| `domain_name`   | "would.io"                                   |
| `ipv4`          | "120.233.157.90"                             |
| `ipv6`          | "c984:fe56:9e90:ef80:ee3e:4924:567f:e3a2"    |
| `password`      | "0EAIMIEDU7d2"                               |
| `user_agent`    | "Mozilla/5.0 (iPad; CPU OS 17_3_1 like ..."  |

## Identifiers and barcodes

| Preset          | Example output                           |
|-----------------|------------------------------------------|
| `uuid4`         | "fa9abf41-41cc-41a9-aeee-074eeb6b7fae"   |
| `md5`           | "249d006e264f848f313d204597f680d0"       |
| `sha1`          | "19d20679952d7d83593766d778d93c873850969a" |
| `sha256`        | "7dd89f77d239bf2f79ed44ef5372b8b4..."    |
| `ssn`           | "166-55-6405"                            |
| `license_plate` | "W72-51B"                                |
| `ean8`          | "12500925"                               |
| `ean13`         | "1171082961608"                          |

## Finance

| Preset                 | Example output              |
|------------------------|-----------------------------|
| `credit_card_number`   | "5232860129040479"          |
| `credit_card_provider` | "Mastercard"                |
| `iban`                 | "US52947179637462169994"    |
| `currency_code`        | "USD"                       |

## Text

| Preset          | Example output                                   |
|-----------------|--------------------------------------------------|
| `text`          | "His number all come then water two them ..."    |
| `sentence`      | "See their was long get she more two work."      |
| `paragraph`     | "Place which only point him might still ..."     |
| `word`          | "most"                                           |

## Dates and times (as strings)

| Preset             | Example output              |
|--------------------|-----------------------------|
| `date_this_year`   | "2026-06-28"                |
| `date_this_decade` | "2021-04-27"                |
| `date_between`     | "2025-03-17"                |
| `date_range`       | "2007-08-01 - 2010-02-22"   |
| `future_date`      | "2027-07-27"                |
| `past_date`        | "2020-07-28"                |
| `time`             | "07:09:32"                  |

For date and time columns with a real type, use `date_field()`,
`datetime_field()`, or `time_field()` instead.

## Files and misc

| Preset           | Example output        |
|------------------|-----------------------|
| `color_name`     | "Coral"               |
| `file_name`      | "look.tar"            |
| `file_extension` | "yaml"                |
| `mime_type`      | "video/x-matroska"    |

## Coherent rows

Related presets in one row agree with each other: the `email` and
`user_name` are built from the name, the `address`, `city`, `state`,
and `postcode` describe one place, and the credit card number matches
its provider. With `profile_fields(prefix=...)`, each prefix is a
separate person.

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

# Several countries
mixed_df = schema.generate(n=100, country={"DE": 0.5, "FR": 0.3, "IT": 0.2})
```

Over 100 country codes are supported, using standard ISO 3166-1
alpha-2 codes.
````

### SKILL LAYOUT

``` gd-skills-tree

check-training-data/
├── SKILL.md
└── references/
    ├── ablation-reference.md
    ├── scorers-reference.md
    └── text-and-rag-reference.md
```

### SKILL.md

``` gd-skills-raw
```

```` markdown
---
name: check-training-data
description: >
  Check data for training and evaluating models with Pointblank: web
  text (heuristic quality filters, language, PII, token length), scored
  steps (col_vals_scored and pb.scorers), near duplicates, benchmark
  contamination, chat transcripts, RAG corpora and evaluation sets
  (pb.rag), LLM-judged rows with calibration, per-row flags and
  weights, data ablation (which checks help a model), train/test split
  checks, synthetic data checks, and dataset cards. Use when cleaning
  or auditing a corpus or a training table, or deciding which checks
  are worth enforcing.
license: MIT
compatibility: >
  Requires Python >=3.10, pointblank installed. Some scorers need extra
  packages (langid, tiktoken, transformers, sentence-transformers,
  cohere); sklearn_evaluator needs pointblank[ml].
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - training-data
    - llm
    - text
    - rag
    - ablation
    - data-cards
---

# Check Training Data

Skill for checking the data that a model is trained or evaluated on,
and for measuring whether the checks matter to the model. The checks
are ordinary validation steps, so they get the usual report, thresholds,
actions, and YAML form; what this skill adds is the steps made for
text and ML data and the tools that use their results.

## Quick start

```python
import pointblank as pb
import polars as pl

docs = pl.DataFrame({"doc_id": [...], "text": [...]})

validation = (
    pb.Validate(data=docs, tbl_name="web_docs")
    .col_vals_text_quality(columns="text")          # 8 Gopher quality steps
    .col_vals_no_pii(columns="text")                # email, phone, credit card
    .rows_near_distinct(columns="text")             # MinHash near duplicates
    .interrogate()
)

flags = validation.get_row_flags()                  # pb_failed_<i>_ per step
clean = flags.filter(pl.col("pb_n_failed_") == 0)   # keeps one copy of each duplicate
```

## Skill directory structure

```
skills/check-training-data/
+-- SKILL.md                         <- This file
+-- references/
    +-- text-and-rag-reference.md    <- text checks, pb.rag, text row checks
    +-- scorers-reference.md         <- col_vals_scored and pb.scorers
    +-- ablation-reference.md        <- Ablation, strategies, results, YAML
```

## When to use what

| I want to...                                       | Use                                         |
| -------------------------------------------------- | ------------------------------------------- |
| Filter web text like the Gopher models             | `col_vals_text_quality(columns)`            |
| Check text is in the expected language             | `col_vals_in_language(columns, expected="en")` |
| Find emails, phones, card numbers, IPs, SSNs, IBANs | `col_vals_no_pii(columns, entities=...)`   |
| Bound text length in model tokens                  | `col_vals_token_len(columns, tokenizer, max_val=...)` |
| Bound a score from any function or model           | `col_vals_scored(columns, scorer, min_val/max_val)` |
| Use a ready-made scorer (embeddings, reranker, classifier, ...) | `pb.scorers.*`                  |
| Find near-duplicate documents                      | `rows_near_distinct(columns)`               |
| Find benchmark or test text in training data       | `rows_not_contaminated(columns, reference)` |
| Check chat transcripts for fine-tuning             | `col_vals_chat_format(columns)`             |
| Check a RAG corpus of chunks                       | `add_steps(pb.rag.corpus_checks(...))`      |
| Check a RAG evaluation set                         | `add_steps(pb.rag.eval_checks(...))`        |
| Judge rows with an LLM                             | `prompt(prompt, model)`                     |
| Measure how far to trust the LLM judge             | `calibrate_prompt(i, labels)`               |
| See which checks each row failed, or weight rows   | `get_row_flags()`                           |
| Use a filtering `pre=` step in flags and ablation  | `pre=pb.row_preserving(fn)`                 |
| Measure whether removing failing rows helps a model | `pb.Ablation(validation, evaluate, metric)` |
| Write an ablation in YAML                          | `pb.yaml_ablation(yaml, evaluate=...)`      |
| Check splits for leaks, label coverage, drift      | `pb.check_splits(train=, valid=, test=)`    |
| Check synthetic data against its seed              | `pb.check_synthetic(synthetic, seed)`       |
| Write a Hugging Face card or a datasheet           | `validation.to_data_card()`                 |

## Core concepts

### Text checks

Four validation methods check text. Each adds ordinary steps
(`col_vals_scored()` or `col_vals_regex()`, listed under those names in
the report, with a brief that says what each checks), one set per
column of `columns=`:

```python
validation = (
    pb.Validate(data=corpus)
    .col_vals_text_quality(columns="text", min_words=10)
    .col_vals_no_pii(columns="text", entities=["email", "phone", "ip_address"])
    .col_vals_token_len(columns="text", tokenizer="whitespace", max_val=4096)
    .interrogate()
)
```

- `col_vals_text_quality()` adds eight steps (word count, mean word
  length, symbol ratio, bullet lines, ellipsis lines, alphabetic words,
  stop words, repeated lines) with the Gopher bounds as defaults. Set a
  bound to `None` to leave its check out.
- `col_vals_no_pii()` adds one regex step per entity (`email`, `phone`,
  `credit_card` by default; also `ip_address`, `us_ssn`, `iban`).
- `col_vals_in_language()` scores each text's probability of being in
  `expected=` (or the language in `expected_column=`) and needs the
  `langid` package (or `backend="fasttext"` with `model=`).
- `col_vals_token_len()` takes a Hugging Face tokenizer name,
  `"tiktoken:cl100k_base"`, `"whitespace"`, a tokenizer object, or a
  function.

They're on `pb.Steps` too, and work in YAML with plain settings.

See `references/text-and-rag-reference.md` for every parameter.

### Scored steps

`col_vals_scored()` passes rows to a scorer in batches and checks each
score against `min_val=` and/or `max_val=` (inclusive). A scorer is a
function that takes a list of row dicts (the values of `columns=`) and
returns a number (or `None`) for each:

```python
def n_words(batch):
    return [len(row["review"].split()) for row in batch]

# a stable name and a version for the score cache (a lambda or a
# nested function needs a name to be cached on disk)
n_words_scorer = pb.scorers.from_function(n_words, name="n_words", version="1")

validation = (
    pb.Validate(data=reviews)
    .col_vals_scored(columns="review", scorer=n_words, min_val=3)
    .col_vals_scored(columns="review", scorer=n_words_scorer, min_val=3,
                     cache=".scores", na_pass=True)
    .col_vals_scored(columns="review",
                     scorer=pb.scorers.text_stat("mean_word_length"), max_val=10)
    .interrogate()
)
```

- Rows with a missing value aren't scored; they fail unless
  `na_pass=True`.
- `cache=True` (default) keeps scores in memory for the session; a
  directory keeps them in `scores.sqlite` between sessions (the scorer
  needs a stable name). `pb.scorers.clear_cache()` empties the memory.
- `keep_scores=True` (default) puts the scores in the extracts and in
  `get_row_flags()` as `pb_score_<i>_`, and lets
  `Ablation.sweep_score()` use them.
- Several columns: `columns=["query", "passage"]` gives each row dict
  both values (a reranker needs this).

The adapters in `pb.scorers` (`embedding_similarity`, `reranker`,
`classifier`, `langid`, `perplexity`, `token_count`, `text_stat`) wrap
common models. See `references/scorers-reference.md`.

### Near duplicates and contamination

```python
validation = (
    pb.Validate(data=train)
    .rows_near_distinct(columns="text", threshold=0.85)
    .rows_not_contaminated(columns="text", reference=benchmark,
                           reference_columns="question")
    .interrogate()
)
```

- `rows_near_distinct()` normalizes text (case, punctuation,
  whitespace) and, with `method="minhash"` (default), groups texts whose
  word 5-gram Jaccard similarity is at least `threshold=`. Every row of
  a group fails the step.
- `rows_not_contaminated()` fails rows that share a 13-word n-gram with
  any reference text (`method="ngram"`), are the same once normalized
  (`"exact"`), or are similar by MinHash (`"minhash"`). A reference text
  shorter than `ngram=` words matches a row that contains it whole. The
  reference can be a table, a file path, a list of texts, or a dict of
  tables by name.

### Chat transcripts

```python
validation = (
    pb.Validate(data=chats)
    .col_vals_chat_format(columns="messages", last_role="assistant")
    .interrogate()
)
```

Each value is a list of `{"role", "content"}` messages (a Polars list of
structs, a pandas list of dicts, or JSON text). A row fails for an
unknown role, a system message that isn't first, turns that don't
alternate (`alternate=True`), the wrong `first_role=`/`last_role=`, or
empty content (unless `allow_empty=True`).

### RAG corpora and evaluation sets (pb.rag)

```python
import datetime

corpus_validation = (
    pb.Validate(data=chunks, tbl_name="chunks")
    .add_steps(
        pb.rag.corpus_checks(
            "text",
            chunk_id="chunk_id",
            metadata="source",
            updated_at="updated_at",
            max_age="365 days",
            reference_time=datetime.date(2026, 11, 1),
        )
    )
    .interrogate()
)

eval_validation = (
    pb.Validate(data=queries)
    .add_steps(pb.rag.eval_checks("query", gold_doc_id="doc_id", corpus=corpus))
    .interrogate()
)
```

`corpus_checks()` checks for empty chunks, repeated chunk ids, missing
metadata, near-duplicate chunks, stale chunks, and (with
`min_tokens=`/`max_tokens=`) chunk length. `eval_checks()` checks for
empty and repeated queries, missing gold document ids, and gold
documents that aren't in the corpus (`corpus_id=` names its id column
if it differs); with `passage=`, `scorer=`, and `min_relevance=`, it also
checks that each passage answers its query.

### LLM-judged rows

`prompt()` sends rows to a model (or a list of models, combined by
`aggregate="majority"`) and passes or fails each one.
`calibrate_prompt()` then compares the step with labels given by a
person (`True` should pass, `False` should fail, `None` unlabeled):

```python
validation = (
    pb.Validate(data=reviews)
    .prompt(
        "The review describes the product, not the delivery.",
        model="anthropic:claude-sonnet-4-6",
        columns_subset="review",
    )
    .interrogate()
)

validation.calibrate_prompt(1, labels="human_label")
# {'n_labeled': ..., 'agreement': ..., 'kappa': ..., 'true_pass': ...,
#  'false_pass': ..., 'false_fail': ..., 'true_fail': ...}
```

A kappa above 0.8 is strong agreement and below 0.4 is weak. The label
column must not be in `columns_subset=` (the model would see the
answers). Label a few dozen rows, including some that should fail.

### Using the results: flags and weights

`get_row_flags()` returns the whole table with `pb_failed_<i>_` for
each row-level step, `pb_score_<i>_` for scored steps, `pb_n_failed_`,
and `pb_row_score_` (the weighted share of the row's checks that
passed, from 0 to 1):

```python
flags = validation.get_row_flags()

clean = flags.filter(pl.col("pb_n_failed_") == 0)          # drop failing rows
weights = flags["pb_row_score_"]                            # or weight them
no_email = flags.filter(~pl.col("pb_failed_9_"))           # step 9 (emails) only
```

Duplicate groups from `rows_distinct()`/`rows_near_distinct()` keep
one copy unflagged (`duplicates="keep_first"`, or `"keep_last"` with
`duplicates_order_by=`, or `"flag_all"`).

Steps with `pre=` are left out of the flags, sundering, and ablation,
because a `pre=` function can change the rows. Mark a function that
only filters, reorders, or changes values with `pb.row_preserving()`:

```python
.col_vals_gt(
    columns="amount", value=0,
    pre=pb.row_preserving(lambda df: df.filter(pl.col("status") == "paid")),
)
```

### Data ablation

An `Ablation` trains the model on versions of the training table with
the failing rows of some steps removed, compares each version with a
control that removes as many random rows, and gives a verdict:
*helps*, *hurts*, *inconclusive*, or (for `drop_all_but_each()`)
*redundant*.

```python
temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10        # readings in the wrong units
for k in range(7, 300, 40):
    temps[k] = None       # missing readings
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed):
    sample = train.drop_nulls().sample(fraction=1.0, with_replacement=True, seed=seed)
    return {"error": abs(sample["temp"].mean() - 20)}

validation = (
    pb.Validate(data=readings)
    .col_vals_not_null(columns="temp")
    .col_vals_between(columns="temp", left=-10, right=45, na_pass=True)
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .drop_each()
    .drop_all()
)
ablation.plan()      # what will be evaluated, without evaluating
ablation.run()       # shows the report
ablation.get_effects().select("name", "delta_vs_control", "p_adj_vs_control", "verdict")
```

Here, removing the out-of-range readings (step 2) helps, and removing
the missing ones (step 1) is inconclusive, since `evaluate` drops nulls
anyway.

- `evaluate(train, *, seed)` trains on `train` (same table type as the
  validation's data), scores on fixed held-out data, and returns a
  number or a dict of numbers. It can also take `variant=`.
- `pb.ablation.sklearn_evaluator(holdout, target=..., scoring=...)`
  makes one for tabular scikit-learn models (needs `pointblank[ml]`).
- The other strategies: `drop_steps()`, `drop_all_but_each()`,
  `drop_groups(column)`, `sweep_step(step, param, values)`,
  `sweep_score(step, quantiles=...)`, `inject(corruption, step=,
  rates=)`, and `shapley()`. Then `suggest_thresholds(tolerance=)`,
  `annotate()`, `export()`/`load_results()`, and `to_json()`.

See `references/ablation-reference.md`.

### Split checks

```python
check = pb.check_splits(train=train, test=test, columns="text",
                        label="topic", near=True)

check.ok                       # False if there are issues
check.issues                   # e.g. "2 rows of 'test' (16.7%) are also in 'train' ..."
check.get_overlap("train", "test")   # the leaked rows (1-based row numbers)
check.get_tabular_report()
```

It checks overlap (exact, or near with `near=True`), label coverage and
shares (with `label=`), and drift of `drift_columns=` (PSI and KS).
Use `splits={"train": ..., "valid": ..., "test": ...}` for other names.

### Synthetic data

`pb.check_synthetic(synthetic, seed, text=...)` checks fidelity,
diversity, and copies of seed rows (see the `generate-data` skill).

### Data cards

```python
card = validation.to_data_card(
    ablation=ablation,                 # adds an Ablation Evidence section
    description="Hourly temperature readings from three sites.",
    license="cc-by-4.0",
    language="en",
    path="README.md",                  # optional; the card is returned either way
)
```

`format="huggingface"` (default) gives YAML front matter and the
Hugging Face template sections plus *Data Quality*;
`format="datasheet"` answers the *Datasheets for Datasets* questions.
The curator comes from `owner=`, uses from `consumers=`, and version
from `version=` of the validation. Unknown parts say
"[More Information Needed]".

## Workflows

### Cleaning a text corpus

1. Add `col_vals_text_quality()` (adjust `min_words=` for short
   texts), `col_vals_no_pii()`, and `rows_near_distinct()`.
2. Add `col_vals_in_language()` or scorers (`classifier`, `perplexity`) if the
   packages are available.
3. Interrogate and read the report: which checks fail the most rows?
4. Use `get_row_flags()` to drop rows, or `pb_row_score_` to weight
   them.
5. Check the training split against benchmarks with
   `rows_not_contaminated()` and against the test split with
   `check_splits()`.

### Deciding which checks to enforce

1. Interrogate the training table with the candidate checks.
2. Write `evaluate` (or use `sklearn_evaluator`) on fixed held-out
   data, and run `ablation.check_determinism()`.
3. `Ablation(...).drop_each().drop_all_but_each().run()`.
4. Enforce the checks that help, drop those that hurt (they remove
   good data), and drop those that are redundant.
5. Tune limits with `sweep_step()`/`sweep_score()`; set thresholds
   with `inject()` and `suggest_thresholds()`.
6. `ablation.annotate()` and `to_data_card(ablation=ablation)` to
   record the evidence.

### Releasing a dataset

1. `pb.check_splits()` for leaks and label coverage.
2. Validate each split; give the validation `owner=`, `consumers=`,
   and `version=`.
3. `to_data_card(path="README.md", license=..., source=...,
   column_descriptions=...)`.

## Gotchas

1. **`get_sundered_data()` leaves out duplicate steps.** It skips
   `rows_distinct()` and `rows_near_distinct()`, so its passing rows
   keep every copy. Filter `get_row_flags()` instead, which keeps one
   copy of each group.
2. **The step counts every copy, the flags keep one.** A pair of near
   duplicates is 2 failing test units in the report, but only the
   second copy is flagged by `get_row_flags()`.
3. **Missing values fail scored steps.** They aren't scored, and fail
   unless `na_pass=True`. In `rows_not_contaminated()` and the regex
   steps of `col_vals_no_pii()`, they pass.
4. **Lambda scorers cache only in memory.** A disk `cache=` needs a
   stable scorer name: use `pb.scorers.from_function(fn, name=...)`,
   and change `version=` when the model changes.
5. **`pre=` steps are left out of flags and ablation** unless the
   function is wrapped in `pb.row_preserving()` (it must keep the
   `pb_index_` column and not aggregate rows).
6. **The heuristic filters assume English web documents.** The default
   `min_words=50` fails short texts, and `min_stop_words=2` uses
   English stop words (set `stop_words=` for other languages).
7. **Ablation verdicts need at least 3 repeats** and are given only for
   `metric=` (other metrics in the dict are evaluated but get no
   verdict). Pick the metric before running.
8. **`evaluate` must use its seed** for all randomness; check with
   `ablation.check_determinism()`. Adding a variant clears the
   results, so call `run()` again.
9. **`yaml_ablation()` imports `evaluate:` only with
   `allow_python=True`.** Otherwise pass `evaluate=` directly.
10. **`to_data_card()` needs an interrogated validation.**

## Related skills

| Skill            | When to use it                                       |
| ---------------- | ---------------------------------------------------- |
| pointblank       | The general API, reports, and post-interrogation     |
| write-validation | Choosing and combining the steps                     |
| validate-yaml    | Scored steps and ablations in YAML                   |
| scan-and-profile | Profiling a table and comparing it with a baseline   |
| generate-data    | Synthetic data and `check_synthetic()`               |
| draft-validation | Drafting a plan with an LLM                          |
````

### references/ablation-reference.md

``` gd-skills-raw
```

```` markdown
# Ablation reference

## Constructor

```python
pb.Ablation(
    validation,                  # an interrogated Validate; its table is the training table
    evaluate=None,               # evaluate(train, *, seed[, variant]) -> number or dict
    metric=None,                 # the metric for verdicts ("score" for a single number)
    higher_is_better=True,
    repeats=5,                   # at least 3 for verdicts
    seed=0,                      # repeat r gets seed + r
    control="size_matched",      # or pb.ablation.Stratified(by=...), or None
    duplicates="keep_first",     # "keep_last", "flag_all", or None (leave rows_distinct out)
    duplicates_order_by=None,
    max_removed=0.5,             # warn when a variant removes more than this share
    conf_level=0.95,
    adjust="holm",               # or None
    label=None,
    evaluate_version=None,       # part of the cache key; change it with the model
    cache=None,                  # a directory of evaluation results
    holdout=None,                # checked with the same steps (never ablated)
    ci="t",                      # or "bootstrap" (needs 10+ repeats)
    pass_as="table",             # or "row_index" (sorted list of kept row numbers)
)
```

### The evaluate function

- Called as `evaluate(train, seed=seed, variant=variant)`; `variant=`
  can be left out of the signature.
- `train` has the same type as the validation's table (lazy and
  database tables are passed lazily). With `pass_as="row_index"`, it's
  the list of kept row numbers (from 0).
- Return a number or a dict of numbers. Score on a fixed holdout that
  doesn't depend on `train`, and use `seed` for every random choice.
- `variant` is an `AblationVariant` with `id`, `strategy`, `name`,
  `steps`, `n_removed`, `f_removed`, `is_control`, `group`,
  `group_column`, `left_out`, `sweep`, `value`, `corruption`, `column`,
  `n_corrupted`, and `checked_by`.

### sklearn_evaluator

```python
evaluate = pb.ablation.sklearn_evaluator(
    holdout,                     # fixed held-out table, same columns as train
    target="y",
    features=None,               # default: usable columns except target
    estimator=None,              # default: HistGradientBoosting{Classifier,Regressor}
    task="auto",                 # "classification", "regression", or "auto"
    scoring=None,                # "accuracy"/"r2" by default; a name, a scorer, or a list/dict
)
```

Needs `pip install "pointblank[ml]"`. The estimator is cloned for each
evaluation with every `random_state` set to the seed. scikit-learn
scorers are higher-is-better (errors are negated), so keep
`higher_is_better=True`. It can't be used with `pass_as="row_index"`.

## Strategies (add variants)

Each returns the `Ablation`, so they chain. Adding a variant clears
earlier results.

| Method | Variant(s) |
|--------|-----------|
| `drop_each(steps=None)` | One per step: remove its failing rows |
| `drop_all()` | Remove rows failing any step |
| `drop_steps(steps, name=None)` | Remove rows failing any of `steps` |
| `drop_all_but_each()` | One per step: remove rows failing every *other* step (what the left-out step adds) |
| `drop_groups(column, values=None)` | One per value of `column`: remove its rows |
| `sweep_step(step, param, values)` | One per value of a step parameter, e.g. `sweep_step(2, "right", [30, 45, 100])` |
| `sweep_score(score, quantiles=None, values=None, remove=None)` | Remove shares (or cutoffs) of the scores of a `col_vals_scored` step or a numeric column |
| `inject(corruption, column=None, step=None, rates=None, name=None)` | Corrupt clean rows at each rate: `"null"`, `"shuffle"`, `"truncate"`, `"duplicate"`, or a function of a column's values |
| `shapley(steps=None, permutations=200, seed=None)` | The combinations needed for each step's Shapley value |

Only row-level steps can be ablated: table-level steps, inactive steps,
and steps with an unmarked `pre=` are left out with a note (mark a
filtering `pre=` with `pb.row_preserving()`).

`inject()` needs a Polars, pandas, or PyArrow table and
`pass_as="table"`; its variants are compared with the full table (no
control).

## Running and results

| Method | Returns / does |
|--------|----------------|
| `plan(lang=None, locale=None)` | GT table of what will be evaluated |
| `check_determinism(seed=None, tolerance=0.0)` | `bool`; warns if the same seed gives different results |
| `run(max_workers=None, executor="thread")` | Evaluates; returns the `Ablation` (which displays its report). `executor="process"` needs a top-level `evaluate` |
| `get_tabular_report(metric=None, title=None, lang=None, locale=None)` | GT report |
| `get_effects()` | Polars DataFrame, one row per variant and metric |
| `get_shapley()` | Shapley values with intervals and p-values |
| `suggest_thresholds(tolerance)` | `{step: Thresholds}` from `inject()` variants (warns for steps without a suggestion) |
| `annotate(validation=None)` | Adds a note with the effect to each ablated step; returns the validation |
| `export(path, format="parquet", overwrite=False)` | Writes `manifest.json`, `tables/`, `results_template.csv` (`format="indices"` writes row numbers instead) |
| `load_results(results, manifest=None)` | Reads results (a directory, CSV, or table) and computes the effects |
| `to_dict()` / `to_json()` | Settings, fingerprints, variants, runs, effects, Shapley values, notes, holdout |
| `holdout_validation` | The interrogated holdout (with `holdout=`) |

### get_effects() columns

`variant`, `name`, `strategy`, `steps`, `left_out`, `group`, `sweep`,
`value`, `n_removed`, `f_removed`, `corruption`, `column`,
`n_corrupted`, `checked_by`, `metric`, `mean`, `sd`, `delta_vs_full`
(+ `_lower`, `_upper`), `p_vs_full`, `p_adj_vs_full`, `control_mean`,
`delta_vs_control` (+ `_lower`, `_upper`), `p_vs_control`,
`p_adj_vs_control`, `delta_added` (+ `_lower`, `_upper`), `p_added`,
`p_adj_added`, `verdict`.

### Verdicts

| Verdict | Meaning |
|---------|---------|
| `helps` | Better than its control (adjusted p < `1 - conf_level`) |
| `hurts` | Worse than its control (the removed rows were good data) |
| `inconclusive` | The difference could be chance |
| `redundant` | `drop_all_but_each()`: the step's rows help alone but add nothing to the other steps |

Each variant and its control use the same seeds, and differences are
paired by seed (a paired t-interval by default). Verdicts are given only
for `metric=`, and only with 3 or more repeats.

## Evaluating elsewhere

```python
plan = pb.Ablation(validation, metric="error", higher_is_better=False, repeats=3).drop_each()
directory = plan.export("ablation-export")
# ... train on each file in manifest.json, write results.csv with
#     variant, repeat, seed, and a column per metric ...
plan.load_results(directory)
```

## YAML

```yaml
tbl: small_table
steps:
- col_vals_gt:
    columns: d
    value: 1000
- col_vals_not_null:
    columns: c
ablation:
  evaluate: my_model:evaluate     # imported only with allow_python=True
  metric: score
  repeats: 3
  strategies:
  - drop_each
  - drop_groups:
      column: f
```

```python
ablation = pb.yaml_ablation(yaml_text, evaluate=evaluate)   # or allow_python=True
ablation.run()
```

`yaml_ablation(yaml, evaluate=None, set_tbl=None, holdout=None,
namespaces=None, allow_python=False)` interrogates the validation and
returns the `Ablation`, ready to run. Strategy arguments are given by
name (`- sweep_step: {step: 1, param: value, values: [500, 1000]}`);
an `inject` corruption must be one of the four names. `yaml_interrogate()`
ignores the `ablation:` block, and `yaml_to_python()` writes it out as
Python.
````

### references/scorers-reference.md

``` gd-skills-raw
```

```` markdown
# Scorers reference

## col_vals_scored

```python
.col_vals_scored(
    columns,              # a column, or a list (each row dict gets all of them)
    scorer,               # a function of a batch of rows, or a pb.scorers adapter
    min_val=None,         # lowest passing score (inclusive)
    max_val=None,         # highest passing score (inclusive); give at least one
    na_pass=False,        # rows with a missing value, or with no score
    cache=True,           # True (memory), a directory (scores.sqlite), or False
    keep_scores=True,     # pb_score_ in extracts, get_row_flags(), sweep_score()
    pre=None, segments=None, thresholds=None, actions=None,
    brief=None, active=True,
)
```

A scorer takes a list of row dicts (`{"review": ..., "stars": ...}`)
and returns one number (or `None`) per row, in order. Rows with a
missing value in any of `columns=` aren't passed to it.

```python
def stars_match(batch):
    return [
        1.0 if (row["stars"] >= 4) == ("great" in row["review"].lower()) else 0.0
        for row in batch
    ]

pb.Validate(data=reviews).col_vals_scored(
    columns=["review", "stars"], scorer=stars_match, min_val=1, na_pass=True
)
```

## The Scorer protocol

Any callable is a scorer. Three optional attributes are read:

| Attribute    | Use                                                       |
|--------------|-----------------------------------------------------------|
| `name`       | Identifies the scorer in the cache and the report          |
| `version`    | Part of the cache key (change it when the model changes)  |
| `batch_size` | Rows per call (256 if absent)                             |

`pb.scorers.from_function(fn, name=None, version=None, batch_size=None,
per_row=False)` sets them. With `per_row=True`, `fn` takes one row
dict and returns its score:

```python
n_chars = pb.scorers.from_function(lambda row: len(row["text"]), name="n_chars", per_row=True)
```

## The score cache

Scores are cached by the scorer's name and version and the row's
values, so re-interrogating (as an `Ablation` does) doesn't call the
model again. They're written after each batch, so an interrupted run
resumes.

- `cache=True`: in memory for the session; `pb.scorers.clear_cache()`
  empties it.
- `cache="dir"`: an SQLite `dir/scores.sqlite`, shared between
  sessions. Needs a stable name: the adapters have one (unless given a
  model object), and so does a function defined at the top level of a
  module. A lambda or nested function needs `from_function(name=...)`.
- `cache=False`: score every time.

## Adapters (pb.scorers)

All take `name=`, `version=`, and `batch_size=`, and a `column=` that
can be left out when the step has one column.

| Adapter | Scores | Needs |
|---------|--------|-------|
| `text_stat(stat, stop_words=None)` | A text statistic (below) | nothing |
| `token_count(tokenizer)` | Number of tokens | depends on the tokenizer (see `col_vals_token_len()`) |
| `langid(expected=, expected_column=, backend="langid", model=None)` | Probability of the expected language(s) | `langid`, or `fasttext` + `model=`, or a function |
| `embedding_similarity(reference, embedder)` | Cosine similarity to the closest reference text | `sentence-transformers` for a model name; or an object with `encode()`; or a function of a list of texts returning vectors |
| `classifier(hf_model, label=None, function_to_apply=None, device=None)` | A text classifier's score for `label=` (no label for one-output models) | `transformers` + a backend for a model name; or a pipeline |
| `reranker(provider="cohere", model=None, query=None, document=None, api_key=None)` | Relevance of a document to a query (two columns) | `cohere` + `CO_API_KEY`, or `provider="sentence-transformers"` |
| `perplexity(fn=None, kenlm_model=None, per_row=True)` | Perplexity under a language model | a function, or `kenlm` + a model path |

Notes:

- `classifier()` with a function instead of a pipeline calls it as
  `fn(texts)`, adding `top_k=None`, `truncation=True`, and
  `function_to_apply=` (if set) only for the keywords the function
  takes; it returns, for each text, a list of
  `{"label": ..., "score": ...}` dicts.
- `langid()` with a function backend: `fn(text)` returns
  `{lang: prob}` or `[(lang, prob), ...]`.
- `reranker()` scores from different models aren't on one scale; pick
  `min_val=` for the model.

```python
def letter_counts(texts):          # stands in for an embedding model
    return [[t.count(c) for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts]

pb.Validate(data=docs).col_vals_scored(
    columns="text",
    scorer=pb.scorers.embedding_similarity("checking data in tables", embedder=letter_counts),
    min_val=0.8,
)
```

## text_stat statistics

Words are split on whitespace; lines on newlines (blank lines left
out).

| `stat`             | Score                                          |
|--------------------|------------------------------------------------|
| `char_count`       | Characters                                     |
| `word_count`       | Words                                          |
| `mean_word_length` | Mean characters per word                       |
| `symbol_ratio`     | `#` symbols and ellipses per word              |
| `bullet_lines`     | Share of lines starting with a bullet          |
| `ellipsis_lines`   | Share of lines ending with an ellipsis         |
| `alpha_words`      | Share of words with a letter                   |
| `stop_words`       | Distinct stop words (`stop_words=` to change them) |
| `repeated_lines`   | Share of lines repeating an earlier line       |

A text without words (or lines) has no score for the share statistics.

## Scores after interrogation

- `validation.get_row_flags()` has `pb_score_<i>_` per scored step.
- The extracts of failing rows have a `pb_score_` column.
- `Ablation.sweep_score(i, quantiles=[...])` removes growing shares of
  the scored rows: the highest-scoring ones for a step with only
  `max_val=`, the lowest otherwise (or set `remove=`). `values=` gives
  score cutoffs instead, and a column name in place of `i` uses that
  column as the score.

## YAML

```yaml
steps:
- col_vals_scored:
    columns: text
    scorer:
      text_stat:
        stat: word_count
    min_val: 50
```

The adapter's settings must be a mapping (`text_stat: word_count`
raises a `YAMLValidationError`). A scorer of your own needs
`allow_python=True`; see the `validate-yaml` skill.
````

### references/text-and-rag-reference.md

``` gd-skills-raw
```

```` markdown
# Text and RAG checks reference

The text checks are validation methods (on `Validate` and `pb.Steps`)
that add ordinary steps: `col_vals_scored()` or `col_vals_regex()`
steps, listed under those names in the report, each with a brief that
says what it checks. They take `columns=` (a column or a list, with a
set of steps for each) and also `pre=`, `segments=`, `thresholds=`,
`actions=`, `brief=`, `active=`, and `dimension=`, applied to each step
they add. By default, each step's brief is generated; a string `brief=`
can include it as `"{auto}"`, and `brief=False` leaves it out. They work
in YAML like any other method (plain settings only, so no function
backends or tokenizers).

The `pb.rag` functions return a `pb.Steps` bundle, added to a
validation with `add_steps()`. They take `pre=`, `segments=`,
`thresholds=`, `actions=`, and `active=`.

## col_vals_text_quality

```python
.col_vals_text_quality(
    columns,
    min_words=50, max_words=100_000,
    min_mean_word_length=3, max_mean_word_length=10,
    max_symbol_ratio=0.1,
    max_bullet_lines=0.9,
    max_ellipsis_lines=0.3,
    min_alpha_words=0.8,
    min_stop_words=2,
    max_repeated_lines=0.3,
    stop_words=None,
    na_pass=True,
)
```

The quality filters of the Gopher models (Rae et al., 2021), as eight
`col_vals_scored()` steps with `text_stat()` scorers:

| Step | Statistic          | Passes when                                   |
|------|--------------------|-----------------------------------------------|
| 1    | `word_count`       | between `min_words` and `max_words`           |
| 2    | `mean_word_length` | between the two bounds (characters)           |
| 3    | `symbol_ratio`     | `#` and ellipses per word <= `max_symbol_ratio` |
| 4    | `bullet_lines`     | share of lines starting with a bullet <= bound |
| 5    | `ellipsis_lines`   | share of lines ending with an ellipsis <= bound |
| 6    | `alpha_words`      | share of words with a letter >= bound         |
| 7    | `stop_words`       | distinct stop words >= `min_stop_words`       |
| 8    | `repeated_lines`   | share of lines repeating an earlier line <= bound |

Set a bound to `None` to drop its check (a two-bound check is dropped
when both are `None`). Missing values pass (`na_pass=True`), as do
texts without the words or lines a check needs (step 1 catches
those). The default stop words are `the be to of and that have with`;
set `stop_words=` for other languages.

## col_vals_no_pii

```python
.col_vals_no_pii(columns, entities=("email", "phone", "credit_card"), na_pass=True)
```

One `col_vals_regex(inverse=True)` step per column and entity. The
patterns use only portable regex features, so they run in databases
and PySpark.

| Entity        | Finds                                              |
|---------------|----------------------------------------------------|
| `email`       | Email addresses                                    |
| `phone`       | North American phone numbers (`(555) 123-4567`, `+1 555.123.4567`) |
| `credit_card` | Visa, Mastercard, Amex, Discover numbers (no checksum) |
| `ip_address`  | IPv4 addresses                                     |
| `us_ssn`      | US Social Security numbers (`123-45-6789`)         |
| `iban`        | IBANs                                              |

`entities=` can also be a dict of your own patterns by name. The
patterns find likely matches; for names and addresses, use a named
entity model through `col_vals_scored()`.

## col_vals_in_language

```python
.col_vals_in_language(
    columns,
    expected=None,            # "en" or ["en", "fr"]
    expected_column=None,     # or a column with each row's language
    min_confidence=0.5,
    backend="langid",         # "langid", "fasttext", or a function
    model=None,               # fastText model path (e.g., "lid.176.bin")
    na_pass=False,
    cache=True,
)
```

One `col_vals_scored()` step per column, with the
`pb.scorers.langid()` scorer. The
`"langid"` backend needs the `langid` package and `"fasttext"` needs
`fasttext`. A function backend takes a text and returns
`{language: probability}` or `[(language, probability), ...]`. `"en"`
matches `"en_US"`.

## col_vals_token_len

```python
.col_vals_token_len(columns, tokenizer, min_val=None, max_val=None,
                    na_pass=False, cache=True)
```

One `col_vals_scored()` step per column, with the
`pb.scorers.token_count()` scorer. For a length in characters, use
`col_vals_str_len()`.

| Tokenizer                    | Needs          |
|------------------------------|----------------|
| `"gpt2"`, `"bert-base-uncased"`, ... (Hugging Face name) | `transformers` |
| `"tiktoken:cl100k_base"`, `"tiktoken:gpt-4o"` | `tiktoken`   |
| `"whitespace"`               | nothing (counts words) |
| An object with `encode()`    | its library    |
| A function returning tokens or a count | nothing |

## rows_near_distinct

```python
.rows_near_distinct(columns, method="minhash", threshold=0.85, ngram=5, num_perm=128)
```

Texts are normalized (lower case, no punctuation, single spaces).
`method="exact"` groups texts that are then the same; `"minhash"`
groups texts whose word n-gram Jaccard similarity is at least
`threshold=`. Several columns are joined by newlines. Every row of a
group fails; `get_row_flags()` leaves one copy unflagged.

## rows_not_contaminated

```python
.rows_not_contaminated(
    columns,
    reference,                 # table, path, list of tables, dict of tables, or list of texts
    reference_columns=None,    # default: the same names as columns=
    method="ngram",            # "ngram", "exact", or "minhash"
    ngram=None,                # 13 for "ngram", 5 for "minhash"
    threshold=0.8,             # "minhash" only
    num_perm=128,
)
```

- `"ngram"`: a row fails if it shares any `ngram`-word n-gram with a
  reference text. A reference text shorter than `ngram` words matches a
  row that contains it whole.
- `"exact"`: the normalized texts are the same.
- `"minhash"`: Jaccard similarity of n-grams >= `threshold`.

Missing values pass. The first matching reference text is in a
`pb_match_` column of the extracts, and the note counts overlaps per
reference (name them with a dict). The reference is held in memory.

## col_vals_chat_format

```python
.col_vals_chat_format(
    columns,
    roles=("system", "user", "assistant"),
    alternate=True,       # no role follows itself (after a first system message)
    first_role=None,      # e.g., "user"
    last_role=None,       # e.g., "assistant"
    allow_empty=False,
    role_key="role",
    content_key="content",
    na_pass=False,
)
```

Values are lists of messages (Polars list of structs, pandas list of
dicts) or JSON text, either a list or an object with `"messages"`. A
system message may only be first.

## pb.rag.corpus_checks

```python
pb.rag.corpus_checks(
    text,
    chunk_id=None,
    metadata=None,             # column or list, e.g. ["source", "acl"]
    min_tokens=None, max_tokens=None, tokenizer="whitespace",
    near_duplicates=True,      # or a threshold
    updated_at=None, max_age=None,   # "365 days", "2 weeks", or timedelta
    reference_time=None,       # default: when the steps are made (UTC)
)
```

| Step               | With                         | Method                 |
|--------------------|------------------------------|------------------------|
| Chunk has text     | always                       | `col_vals_regex`       |
| Ids are unique     | `chunk_id=`                  | `rows_distinct`        |
| Metadata present   | `metadata=` (one step each)  | `col_vals_not_null`    |
| Token length       | `min_tokens=` or `max_tokens=` | `col_vals_token_len`   |
| No near duplicates | `near_duplicates=True`       | `rows_near_distinct`   |
| Fresh enough       | `updated_at=` and `max_age=` | `col_vals_ge`          |

The freshness cutoff is fixed when the steps are made, so pass
`reference_time=` for reproducible results.

## pb.rag.eval_checks

```python
pb.rag.eval_checks(
    query,
    gold_doc_id=None,
    corpus=None, corpus_id=None,     # corpus_id defaults to gold_doc_id's name
    passage=None, scorer=None, min_relevance=None,
    near_duplicates=True,
)
```

| Step                      | With                                  | Method              |
|---------------------------|---------------------------------------|---------------------|
| Query has text            | always                                | `col_vals_regex`    |
| No near-duplicate queries | `near_duplicates=True`                | `rows_near_distinct` |
| Has a gold document       | `gold_doc_id=`                        | `col_vals_not_null` |
| Gold document in corpus   | `corpus=`                             | `col_vals_in_table` |
| Passage is relevant       | `passage=`, `scorer=`, `min_relevance=` | `col_vals_scored` |

A relevance scorer gets the query and the passage, e.g.
`pb.scorers.reranker(provider="cohere")`.
````
