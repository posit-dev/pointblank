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
