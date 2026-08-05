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
