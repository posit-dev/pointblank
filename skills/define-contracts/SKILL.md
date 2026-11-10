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
