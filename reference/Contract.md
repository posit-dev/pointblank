# Contract

A declarative boundary contract for pipeline data.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L376-L1384)

``` python
Contract(
    name,
    direction="source",
    schema=None,
    steps=list(),
    version=None,
    owner=None,
    consumers=None,
    description=None,
    thresholds=None,
    on_violation="warn",
)
```

A [Contract](../reference/Contract.md#pointblank.Contract) is a written-down agreement about what a table must look like at a specific point in a data pipeline. It bundles three things into one object: a [`Schema`](../reference/Schema.md#pointblank.Schema) with the expected column names and types (the structural expectations), a list of [`Step`](../reference/Step.md#pointblank.Step) objects with the rules the values must follow (the semantic expectations), and metadata saying who owns the data, who depends on it, and which version of the agreement is in force.

With a plain [`Validate`](../reference/Validate.md#pointblank.Validate) chain, the rules exist only as code that runs against one table. A contract holds the same rules as data. It isn't tied to any table, so you can apply it to each day's delivery of a feed, check it into version control, review changes to it like any other change, and share it as a YAML file. Teams on both sides of a handoff can then refer to the same definition of "valid". The producer of a table can check its output against the contract before publishing, and the consumer can check its input against the same contract on arrival.

Use a contract when the expectations matter more than any one validation run: data crossing a team or system boundary, inputs and outputs of a scheduled job, or a dataset with named owners and downstream consumers. For a one-off look at a table, a [Validate](../reference/Validate.md#pointblank.Validate) chain is simpler.

Validating data against a contract gives you an ordinary interrogated [Validate](../reference/Validate.md#pointblank.Validate) object. That means everything you already use with validations works here: the tabular report, step reports, [`all_passed()`](../reference/Validate.all_passed.md#pointblank.Validate.all_passed), [`assert_passing()`](../reference/Validate.assert_passing.md#pointblank.Validate.assert_passing), data extracts, and so on.

## Parameters

`name: str`  
A short, unique name for the contract (e.g., `"raw_clickstream_feed"`). It becomes the table name in the validation report and is part of the report label. It must be a non-empty string.

`direction: Literal["source", "target"] = "source"`  
Where the contract sits relative to a transformation: `"source"` (the default) for data flowing in, or `"target"` for the data product flowing out. The direction is descriptive metadata that appears in violation messages from [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline). It doesn't change what gets checked.

`schema: Schema | None = None`  
An optional [`Schema`](../reference/Schema.md#pointblank.Schema) giving the expected column names and data types. When a schema is given, it's checked first, with [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) and its default settings. These defaults are strict: the table must have exactly the schema's columns, in the same order, with matching types.

`steps: list[Step] = list()`  
A list of [`Step`](../reference/Step.md#pointblank.Step) objects, each naming a validation method and its arguments (e.g., `pb.Step("col_vals_gt", columns="amount", value=0)`). They run in list order after the schema check. Only a subset of validation methods can be used. See the *Supported Step Methods* section below.

`version: str | None = None`  
A version string for the contract, such as `"1.2.0"`. Semantic versioning is a good convention, but any string is accepted. The version is added to the report label (e.g., `"Contract: orders v1.2.0"`) and recorded in the report's governance metadata.

`owner: str | None = None`  
The person or team responsible for the contract and the data it describes. It's recorded in the report's governance metadata.

`consumers: str | list[str] | None = None`  
Who depends on data meeting this contract, as a string or a list of strings. It's recorded in the report's governance metadata.

`description: str | None = None`  
An optional longer description of the contract's purpose. It's kept with the contract and serialized with it, but doesn't appear in validation reports.

`thresholds: Thresholds | None = None`  
Default [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) for the validation built from this contract. These set when a failing step is considered a 'warning', 'error', or 'critical' issue. If the contract is used in a [Pipeline](../reference/Pipeline.md#pointblank.Pipeline) that has its own thresholds, the pipeline's thresholds take precedence.

`on_violation: Literal["warn", "raise", "log"] = "warn"`  
What a [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline) should do when data fails this contract: `"warn"` (the default) issues a `UserWarning`, `"raise"` raises a [`ContractViolationError`](../reference/ContractViolationError.md#pointblank.ContractViolationError) (a `RuntimeError`), and `"log"` writes a warning to the `"pointblank.contract"` logger. This setting applies only when the contract is enforced through a [Pipeline](../reference/Pipeline.md#pointblank.Pipeline). Calling [`validate()`](../reference/pointblank.Contract.validate.md) directly always returns the validation without warning or raising, so check the result yourself (e.g., with [`assert_passing()`](../reference/Validate.assert_passing.md#pointblank.Validate.assert_passing)).

## Returns

`Contract`  
A [Contract](../reference/Contract.md#pointblank.Contract) object. Apply it to data with [`validate()`](../reference/pointblank.Contract.validate.md), or with [`to_validate()`](../reference/pointblank.Contract.to_validate.md) when you want to add more steps before interrogating.

## How A Contract Becomes A Validation

A contract holds no data and runs nothing until it's applied to a table. Calling [`to_validate()`](../reference/pointblank.Contract.to_validate.md) (or [`validate()`](../reference/pointblank.Contract.validate.md), which also interrogates) compiles the contract into a [Validate](../reference/Validate.md#pointblank.Validate) object:

1.  a [Validate](../reference/Validate.md#pointblank.Validate) is created for the table, with `tbl_name=` set to the contract name, a label of the form `"Contract: <name> v<version>"`, and the contract's `thresholds=`, `owner=`, `consumers=`, and `version=`
2.  if the contract has a schema, a [col_schema_match()](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) step is added first
3.  each [`Step`](../reference/Step.md#pointblank.Step) is added in order, by calling the validation method it names with its stored arguments

A step that lists several columns expands into one validation step per column, just as it would in a [Validate](../reference/Validate.md#pointblank.Validate) chain. Mistakes in a step's arguments (a misspelled parameter name, for example) surface when the contract is compiled, not when it's created.

## Supported Step Methods

These validation methods can be used in a contract's steps:

- column value checks: [col_vals_gt](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt), [col_vals_lt](../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt), [col_vals_ge](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge), [col_vals_le](../reference/Validate.col_vals_le.md#pointblank.Validate.col_vals_le), [col_vals_eq](../reference/Validate.col_vals_eq.md#pointblank.Validate.col_vals_eq), [col_vals_ne](../reference/Validate.col_vals_ne.md#pointblank.Validate.col_vals_ne), [col_vals_between](../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between), [col_vals_outside](../reference/Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside), [col_vals_in_set](../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set), [col_vals_not_in_set](../reference/Validate.col_vals_not_in_set.md#pointblank.Validate.col_vals_not_in_set), [col_vals_increasing](../reference/Validate.col_vals_increasing.md#pointblank.Validate.col_vals_increasing), [col_vals_decreasing](../reference/Validate.col_vals_decreasing.md#pointblank.Validate.col_vals_decreasing), [col_vals_null](../reference/Validate.col_vals_null.md#pointblank.Validate.col_vals_null), [col_vals_not_null](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null), [col_vals_regex](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex), [col_vals_within_spec](../reference/Validate.col_vals_within_spec.md#pointblank.Validate.col_vals_within_spec), and [col_vals_expr](../reference/Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr)
- column-level checks: [col_exists](../reference/Validate.col_exists.md#pointblank.Validate.col_exists) and [col_pct_null](../reference/Validate.col_pct_null.md#pointblank.Validate.col_pct_null)
- table-level checks: [col_schema_match](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match), [col_count_match](../reference/Validate.col_count_match.md#pointblank.Validate.col_count_match), [row_count_match](../reference/Validate.row_count_match.md#pointblank.Validate.row_count_match), [rows_distinct](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct), and [rows_complete](../reference/Validate.rows_complete.md#pointblank.Validate.rows_complete)
- multi-condition checks: [conjointly](../reference/Validate.conjointly.md#pointblank.Validate.conjointly), with its expressions given as a list in `exprs=`

Each takes the same arguments as the [`Validate`](../reference/Validate.md#pointblank.Validate) method of the same name. A step that names any other method is rejected with a `ValueError` when the contract is created. For steps that a contract can't express, compile the contract with [`to_validate()`](../reference/pointblank.Contract.to_validate.md) and chain the extra methods onto the result.

## Serialization

Contracts are meant to live outside your code, so they round-trip through YAML and plain dictionaries. [`to_yaml()`](../reference/pointblank.Contract.to_yaml.md) and [`from_yaml()`](../reference/pointblank.Contract.from_yaml.md) work with YAML files, and [`to_dict()`](../reference/pointblank.Contract.to_dict.md) and [`from_dict()`](../reference/pointblank.Contract.from_dict.md) work with dictionaries (useful for JSON or a configuration system). The schema is stored as a mapping of column names to type strings, the thresholds as a mapping of levels to values, and each step as a single-key mapping from method name to arguments. Step arguments must be plain data (strings, numbers, booleans, dates, lists, and dictionaries) for a contract to be written to YAML, although step-level [Thresholds](../reference/Thresholds.md#pointblank.Thresholds) objects and tuples are converted to dictionaries automatically. A contract with other objects in its steps, such as a column selector (e.g., `pb.starts_with("amt_")`), a Polars expression, or a function, can't be written: [to_yaml()](../reference/pointblank.Contract.to_yaml.md) raises a `ValueError` that says where the object is. Keep contracts like that in Python.

## Working With Other Pointblank Tools

A contract is the building block for several other parts of the API:

- [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline) pairs a source contract and a target contract around a transformation function. It validates the input, runs the transformation, validates the output, and applies each contract's `on_violation=` setting. The results come back in a [`PipelineResult`](../reference/PipelineResult.md#pointblank.PipelineResult).
- [`export_contract()`](../reference/export_contract.md#pointblank.export_contract) converts a contract into an external format (JSON Schema, Frictionless Table Schema, dbt, or the Open Data Contract Standard), and [`import_contract()`](../reference/import_contract.md#pointblank.import_contract) reads one of those formats. Call [`to_contract()`](../reference/ContractImport.md#pointblank.ContractImport.to_contract) on the result to get a [Contract](../reference/Contract.md#pointblank.Contract). Each format supports a different set of rules, so checks with no equivalent in the target format are left out of the export.
- Because [`to_validate()`](../reference/pointblank.Contract.to_validate.md) returns an ordinary [Validate](../reference/Validate.md#pointblank.Validate) object, a contract can be combined with reusable [`Steps`](../reference/Steps.md#pointblank.Steps) libraries through [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps).

## Examples

Let's write a contract for a daily orders extract. The schema fixes the four columns and their types, and the steps say that IDs must be present, each order must appear once, amounts must be positive, and the status must be one of three known values. Ownership and versioning go in the metadata, and the thresholds flag any failing test unit as a warning and a failure rate of 10% or more as an error.

``` python
import pointblank as pb
import polars as pl

orders_contract = pb.Contract(
    name="orders",
    version="1.2.0",
    owner="sales-data-team",
    consumers=["finance", "analytics"],
    description="Daily order extract delivered by the sales platform.",
    schema=pb.Schema(
        order_id="String",
        customer_id="Int64",
        amount="Float64",
        status="String",
    ),
    steps=[
        pb.Step("col_vals_not_null", columns=["order_id", "customer_id"]),
        pb.Step("rows_distinct", columns_subset=["order_id"]),
        pb.Step("col_vals_gt", columns="amount", value=0),
        pb.Step("col_vals_in_set", columns="status", set=["pending", "shipped", "delivered"]),
    ],
    thresholds=pb.Thresholds(warning=1, error=0.10),
)

orders_contract
```

    Contract(name='orders', direction='source', version='1.2.0', schema=<defined>, steps=4)

The contract itself is just a definition. Printing it shows a short summary: the name, direction, version, whether a schema is set, and the number of steps.

Now let's apply the contract to a day's extract that meets every expectation. The [`validate()`](../reference/pointblank.Contract.validate.md) method compiles the contract against the table and interrogates it, returning a [Validate](../reference/Validate.md#pointblank.Validate) object that displays as a validation report.

``` python
orders_day_1 = pl.DataFrame(
    {
        "order_id": ["A1", "A2", "A3", "A4"],
        "customer_id": [101, 102, 101, 103],
        "amount": [25.00, 110.50, 42.00, 8.99],
        "status": ["pending", "shipped", "delivered", "shipped"],
    }
)

orders_contract.validate(orders_day_1)
```

[TABLE]

The report header shows the label built from the contract (`Contract: orders v1.2.0`) along with the owner, consumers, and version. The first step is the schema check, followed by one step per contract rule. The [col_vals_not_null](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) step names two columns, so it becomes two validation steps. All six steps pass.

The next day's extract has problems: a missing customer ID, a repeated order ID, a negative amount, and a status the contract doesn't allow. The same contract catches all of them.

``` python
orders_day_2 = pl.DataFrame(
    {
        "order_id": ["A5", "A6", "A6", "A8"],
        "customer_id": [104, None, 101, 105],
        "amount": [25.00, -5.00, 42.00, 8.99],
        "status": ["pending", "shipped", "returned", "shipped"],
    }
)

validation = orders_contract.validate(orders_day_2)

validation
```

[TABLE]

The schema still matches, but four of the rule steps have failing test units. With only four rows, one failure is already 25% of the table, so those steps exceed both the warning and the error thresholds. The [rows_distinct](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step counts two failures because both rows with order `"A6"` are duplicates. Since the result is an ordinary [Validate](../reference/Validate.md#pointblank.Validate) object, you can check it in code, for example to stop a scheduled job:

``` python
validation.all_passed()
```

    False

To share the contract with another team or keep it under version control, write it out as YAML. Called without a path, [`to_yaml()`](../reference/pointblank.Contract.to_yaml.md) returns the YAML text.

``` python
print(orders_contract.to_yaml())
```

    contract:
      name: orders
      version: 1.2.0
      owner: sales-data-team
      consumers:
      - finance
      - analytics
      description: Daily order extract delivered by the sales platform.
      schema:
        order_id: String
        customer_id: Int64
        amount: Float64
        status: String
      steps:
      - col_vals_not_null:
          columns:
          - order_id
          - customer_id
      - rows_distinct:
          columns_subset:
          - order_id
      - col_vals_gt:
          columns: amount
          value: 0
      - col_vals_in_set:
          columns: status
          set:
          - pending
          - shipped
          - delivered
      thresholds:
        warning: 1
        error: 0.1

The YAML lists the metadata, schema, steps, and thresholds in a form that's easy to read and review. [`from_yaml()`](../reference/pointblank.Contract.from_yaml.md) reads it back into an equivalent [Contract](../reference/Contract.md#pointblank.Contract).

Finally, a contract can be the starting point for a stricter, local validation. Here, [`to_validate()`](../reference/pointblank.Contract.to_validate.md) compiles the contract without interrogating, and we add one more check (orders of 100 or less) before calling [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate).

``` python
(
    orders_contract.to_validate(orders_day_1)
    .col_vals_le(columns="amount", value=100)
    .interrogate()
)
```

[TABLE]

The six contract steps pass as before, and the added seventh step flags the order of 110.50. The contract itself is unchanged, so other users of it aren't affected by the extra rule.

## See Also

[Step](../reference/Step.md#pointblank.Step)  
A single declarative validation step, used in a contract's `steps=`.

[Schema](../reference/Schema.md#pointblank.Schema)  
The column names and types used in a contract's `schema=`.

[Pipeline](../reference/Pipeline.md#pointblank.Pipeline)  
Enforce source and target contracts around a transformation.

[export_contract()](../reference/export_contract.md#pointblank.export_contract)  
Convert a contract to JSON Schema, Frictionless, dbt, or ODCS.

[import_contract()](../reference/import_contract.md#pointblank.import_contract)  
Read an external format, then call [to_contract()](../reference/ContractImport.md#pointblank.ContractImport.to_contract) on it.

## Functions

| Name | Description |
|----|----|
| [validate()](pointblank.Contract.validate.md) | Check a table against the contract. |
| [to_validate()](pointblank.Contract.to_validate.md) | Compile the contract into a validation plan for a table, without running it. |
| [to_yaml()](pointblank.Contract.to_yaml.md) | Write the contract as YAML. |
| [from_yaml()](pointblank.Contract.from_yaml.md) | Read a contract from a YAML file. |
| [to_dict()](pointblank.Contract.to_dict.md) | Convert the contract to a dictionary. |
| [from_dict()](pointblank.Contract.from_dict.md) | Create a contract from a dictionary. |
