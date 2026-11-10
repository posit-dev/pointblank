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
