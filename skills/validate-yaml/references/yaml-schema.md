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
