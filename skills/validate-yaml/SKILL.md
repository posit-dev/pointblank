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
