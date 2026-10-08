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
