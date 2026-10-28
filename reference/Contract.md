# Contract

A declarative boundary contract for pipeline data.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L360-L1346)

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
What a [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline) should do when data fails this contract: `"warn"` (the default) issues a `UserWarning`, `"raise"` raises a `RuntimeError`, and `"log"` writes a warning to the `"pointblank.contract"` logger. This setting applies only when the contract is enforced through a [Pipeline](../reference/Pipeline.md#pointblank.Pipeline). Calling [`validate()`](../reference/Contract.md#pointblank.Contract.validate) directly always returns the validation without warning or raising, so check the result yourself (e.g., with [`assert_passing()`](../reference/Validate.assert_passing.md#pointblank.Validate.assert_passing)).

## Returns

`Contract`  
A [Contract](../reference/Contract.md#pointblank.Contract) object. Apply it to data with [`validate()`](../reference/Contract.md#pointblank.Contract.validate), or with [`to_validate()`](../reference/Contract.md#pointblank.Contract.to_validate) when you want to add more steps before interrogating.

## How A Contract Becomes A Validation

A contract holds no data and runs nothing until it's applied to a table. Calling [`to_validate()`](../reference/Contract.md#pointblank.Contract.to_validate) (or [`validate()`](../reference/Contract.md#pointblank.Contract.validate), which also interrogates) compiles the contract into a [Validate](../reference/Validate.md#pointblank.Validate) object:

1.  a [Validate](../reference/Validate.md#pointblank.Validate) is created for the table, with `tbl_name=` set to the contract name, a label of the form `"Contract: <name> v<version>"`, and the contract's `thresholds=`, `owner=`, `consumers=`, and `version=`
2.  if the contract has a schema, a `col_schema_match()` step is added first
3.  each [`Step`](../reference/Step.md#pointblank.Step) is added in order, by calling the validation method it names with its stored arguments

A step that lists several columns expands into one validation step per column, just as it would in a [Validate](../reference/Validate.md#pointblank.Validate) chain. Mistakes in a step's arguments (a misspelled parameter name, for example) surface when the contract is compiled, not when it's created.

## Supported Step Methods

These validation methods can be used in a contract's steps:

- column value checks: `col_vals_gt`, `col_vals_lt`, `col_vals_ge`, `col_vals_le`, `col_vals_eq`, `col_vals_ne`, `col_vals_between`, `col_vals_outside`, `col_vals_in_set`, `col_vals_not_in_set`, `col_vals_increasing`, `col_vals_decreasing`, `col_vals_null`, `col_vals_not_null`, `col_vals_regex`, `col_vals_within_spec`, and `col_vals_expr`
- column-level checks: `col_exists` and `col_pct_null`
- table-level checks: `col_schema_match`, `col_count_match`, `row_count_match`, `rows_distinct`, and `rows_complete`
- multi-condition checks: `conjointly`, with its expressions given as a list in `exprs=`

Each takes the same arguments as the [`Validate`](../reference/Validate.md#pointblank.Validate) method of the same name. A step that names any other method is rejected with a `ValueError` when the contract is created. For steps that a contract can't express, compile the contract with [`to_validate()`](../reference/Contract.md#pointblank.Contract.to_validate) and chain the extra methods onto the result.

## Serialization

Contracts are meant to live outside your code, so they round-trip through YAML and plain dictionaries. [`to_yaml()`](../reference/Contract.md#pointblank.Contract.to_yaml) and [`from_yaml()`](../reference/Contract.md#pointblank.Contract.from_yaml) work with YAML files, and [`to_dict()`](../reference/Contract.md#pointblank.Contract.to_dict) and [`from_dict()`](../reference/Contract.md#pointblank.Contract.from_dict) work with dictionaries (useful for JSON or a configuration system). The schema is stored as a mapping of column names to type strings, the thresholds as a mapping of levels to values, and each step as a single-key mapping from method name to arguments. Step arguments must be plain data (strings, numbers, booleans, dates, lists, and dictionaries) for a contract to be written to YAML, although step-level [Thresholds](../reference/Thresholds.md#pointblank.Thresholds) objects and tuples are converted to dictionaries automatically. A contract with other objects in its steps, such as a column selector (e.g., `pb.starts_with("amt_")`), a Polars expression, or a function, can't be written: `to_yaml()` raises a `ValueError` that says where the object is. Keep contracts like that in Python.

## Working With Other Pointblank Tools

A contract is the building block for several other parts of the API:

- [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline) pairs a source contract and a target contract around a transformation function. It validates the input, runs the transformation, validates the output, and applies each contract's `on_violation=` setting. The results come back in a [`PipelineResult`](../reference/PipelineResult.md#pointblank.PipelineResult).
- [`export_contract()`](../reference/export_contract.md#pointblank.export_contract) converts a contract into an external format (JSON Schema, Frictionless Table Schema, dbt, or the Open Data Contract Standard), and [`import_contract()`](../reference/import_contract.md#pointblank.import_contract) reads one of those formats. Call [`to_contract()`](../reference/ContractImport.md#pointblank.ContractImport.to_contract) on the result to get a [Contract](../reference/Contract.md#pointblank.Contract). Each format supports a different set of rules, so checks with no equivalent in the target format are left out of the export.
- Because [`to_validate()`](../reference/Contract.md#pointblank.Contract.to_validate) returns an ordinary [Validate](../reference/Validate.md#pointblank.Validate) object, a contract can be combined with reusable [`Steps`](../reference/Steps.md#pointblank.Steps) libraries through [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps).

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

Now let's apply the contract to a day's extract that meets every expectation. The [`validate()`](../reference/Contract.md#pointblank.Contract.validate) method compiles the contract against the table and interrogates it, returning a [Validate](../reference/Validate.md#pointblank.Validate) object that displays as a validation report.

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

The report header shows the label built from the contract (`Contract: orders v1.2.0`) along with the owner, consumers, and version. The first step is the schema check, followed by one step per contract rule. The `col_vals_not_null` step names two columns, so it becomes two validation steps. All six steps pass.

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

The schema still matches, but four of the rule steps have failing test units. With only four rows, one failure is already 25% of the table, so those steps exceed both the warning and the error thresholds. The `rows_distinct` step counts two failures because both rows with order `"A6"` are duplicates. Since the result is an ordinary [Validate](../reference/Validate.md#pointblank.Validate) object, you can check it in code, for example to stop a scheduled job:

``` python
validation.all_passed()
```

    False

To share the contract with another team or keep it under version control, write it out as YAML. Called without a path, [`to_yaml()`](../reference/Contract.md#pointblank.Contract.to_yaml) returns the YAML text.

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

The YAML lists the metadata, schema, steps, and thresholds in a form that's easy to read and review. [`from_yaml()`](../reference/Contract.md#pointblank.Contract.from_yaml) reads it back into an equivalent [Contract](../reference/Contract.md#pointblank.Contract).

Finally, a contract can be the starting point for a stricter, local validation. Here, [`to_validate()`](../reference/Contract.md#pointblank.Contract.to_validate) compiles the contract without interrogating, and we add one more check (orders of 100 or less) before calling [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate).

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

`export_contract`  
Convert a contract to JSON Schema, Frictionless, dbt, or ODCS.

`import_contract`  
Read an external contract format, then call [to_contract()](../reference/ContractImport.md#pointblank.ContractImport.to_contract) on the result.

## Methods

| Name | Description |
|----|----|
| [validate()](#validate) | Check a table against the contract. |
| [to_validate()](#to_validate) | Compile the contract into a validation plan for a table, without running it. |
| [to_yaml()](#to_yaml) | Write the contract as YAML. |
| [from_yaml()](#from_yaml) | Read a contract from a YAML file. |
| [to_dict()](#to_dict) | Convert the contract to a dictionary. |
| [from_dict()](#from_dict) | Create a contract from a dictionary. |

### validate()

Check a table against the contract.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L811-L903)

``` python
validate(data)
```

This is the usual way to apply a contract. It compiles the contract into a validation plan for the table and interrogates it right away. This is the same as calling [`to_validate()`](../reference/Contract.md#pointblank.Contract.to_validate) and then [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate). The result is an ordinary interrogated [`Validate`](../reference/Validate.md#pointblank.Validate) object. You can display it as a report, inspect individual steps with [`get_step_report()`](../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report), pull out failing rows with [`get_data_extracts()`](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts), or check the outcome in code.

A failing contract doesn't raise an error or warning here, whatever the contract's `on_violation=` setting is. That setting is applied by [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline). When you call [validate()](../reference/Contract.md#pointblank.Contract.validate) yourself, check the result with a method like [`all_passed()`](../reference/Validate.all_passed.md#pointblank.Validate.all_passed), [`assert_passing()`](../reference/Validate.assert_passing.md#pointblank.Validate.assert_passing), or [`assert_below_threshold()`](../reference/Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold). To add checks before the validation runs, use [`to_validate()`](../reference/Contract.md#pointblank.Contract.to_validate) instead.

#### Parameters

`data: IntoDataFrame`  
The table to validate against the contract. Any table type that [Validate](../reference/Validate.md#pointblank.Validate) accepts can be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

#### Returns

`Validate`  
An interrogated [Validate](../reference/Validate.md#pointblank.Validate) object with the results of every contract check.

#### Raises

`TypeError`  
If a step's arguments don't fit its validation method (for example, a misspelled parameter name). Step arguments are only checked when the contract is compiled.

#### Examples

Here's a contract for a sensor readings feed. Each reading needs a sensor ID, the temperature must be within a plausible range, and the feed must have five rows. Any failing test unit counts as an error.

``` python
import pointblank as pb
import polars as pl

readings_contract = pb.Contract(
    name="sensor_readings",
    owner="iot-platform",
    steps=[
        pb.Step("col_vals_not_null", columns="sensor_id"),
        pb.Step("col_vals_between", columns="temp_c", left=-40, right=60),
        pb.Step("row_count_match", count=5),
    ],
    thresholds=pb.Thresholds(error=1),
)

readings = pl.DataFrame(
    {
        "sensor_id": ["s1", "s2", None, "s4", "s5"],
        "temp_c": [21.5, 19.0, 22.1, 85.0, 20.4],
    }
)

validation = readings_contract.validate(readings)

validation
```

[TABLE]

The row count is right, but the report shows two problems: one reading has no sensor ID, and one temperature (85.0) is out of range. Both steps reach the error threshold. Because [validate()](../reference/Contract.md#pointblank.Contract.validate) doesn't raise, a script or scheduled job decides what happens next. For example, it could test whether any step reached the error level:

``` python
validation.above_threshold(level="error")
```

    True

In a pipeline where bad data should stop processing, [`assert_below_threshold()`](../reference/Validate.assert_below_threshold.md#pointblank.Validate.assert_below_threshold) raises an `AssertionError` in this situation instead.

### to_validate()

Compile the contract into a validation plan for a table, without running it.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L692-L809)

``` python
to_validate(data)
```

This method turns the contract into a [`Validate`](../reference/Validate.md#pointblank.Validate) object for the given table. The validation is set up with the contract's name, label, thresholds, and governance metadata (owner, consumers, and version). Then the schema check (if the contract has a schema) and each of the contract's steps are added in order. Nothing is checked yet: the plan runs when you call [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) on it.

Use this method instead of [`validate()`](../reference/Contract.md#pointblank.Contract.validate) when you want to do something with the plan before it runs. Most often that's adding checks that are local to one use of the data, or ones a contract can't express, such as [`col_vals_str_len()`](../reference/Validate.col_vals_str_len.md#pointblank.Validate.col_vals_str_len) or [`specially()`](../reference/Validate.specially.md#pointblank.Validate.specially). You can also attach [`Actions`](../reference/Actions.md#pointblank.Actions) and final actions this way, or combine the contract with a reusable [`Steps`](../reference/Steps.md#pointblank.Steps) library via [`add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps). The contract itself isn't changed, so these additions don't affect anyone else using it.

#### Parameters

`data: IntoDataFrame`  
The table to validate against the contract. Any table type that [Validate](../reference/Validate.md#pointblank.Validate) accepts can be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

#### Returns

`Validate`  
A [Validate](../reference/Validate.md#pointblank.Validate) object holding the contract's checks, not yet interrogated.

#### Raises

`TypeError`  
If a step's arguments don't fit its validation method (for example, a misspelled parameter name). Step arguments are only checked when the contract is compiled.

#### Examples

Let's define a small contract for a table of products: a two-column schema, plus rules that product IDs are present and prices are positive.

``` python
import pointblank as pb
import polars as pl

products_contract = pb.Contract(
    name="products",
    version="2.0.0",
    schema=pb.Schema(product_id="String", price="Float64"),
    steps=[
        pb.Step("col_vals_not_null", columns="product_id"),
        pb.Step("col_vals_gt", columns="price", value=0),
    ],
)

products = pl.DataFrame(
    {
        "product_id": ["P-100", "P-101", "P-102", "X-103"],
        "price": [19.99, 5.49, 120.00, 7.25],
    }
)
```

Our team also wants product IDs to start with `"P-"`. That rule isn't part of the shared contract, so we compile the contract with `to_validate()`, add a [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) step, and then interrogate.

``` python
(
    products_contract.to_validate(products)
    .col_vals_regex(columns="product_id", pattern=r"^P-")
    .interrogate()
)
```

[TABLE]

The first three steps come from the contract (the schema check, then the two rules) and all pass. The fourth step is our addition, and it flags the ID `"X-103"`.

### to_yaml()

Write the contract as YAML.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L1248-L1336)

``` python
to_yaml(path=None)
```

YAML is the natural way to store and share a contract. It can sit in version control next to the pipeline code, so changes to the rules go through review, and other teams can read it without running Python. This method writes the contract under a top-level `contract:` key, with the fields in a fixed order (name and metadata first, then the schema, steps, and thresholds). [`from_yaml()`](../reference/Contract.md#pointblank.Contract.from_yaml) reads the file back into an equivalent [Contract](../reference/Contract.md#pointblank.Contract).

Fields that aren't set, and the default `direction=` and `on_violation=` values, are left out of the YAML (see [`to_dict()`](../reference/Contract.md#pointblank.Contract.to_dict)). Step arguments must be plain data (strings, numbers, booleans, dates, lists, and dictionaries) so that `from_yaml()` can read them back. Step-level thresholds are converted to dictionaries automatically, but column selectors, expressions, and functions can't be written.

#### Parameters

`path: str | Path | None = None`  
An optional path to write the YAML to. Any missing parent directories are created, and an existing file is overwritten. If `None` (the default), nothing is written.

#### Returns

`str`  
The YAML text. It's returned whether or not `path=` is given.

#### Raises

`ValueError`  
If a step argument can't be represented in YAML (for example, a column selector such as `pb.starts_with("amt_")` or a Polars expression). The message names the step and argument. Nothing is written to `path=` in this case.

#### Examples

Let's define a contract for a table of web sessions and get its YAML representation.

``` python
import pointblank as pb

sessions_contract = pb.Contract(
    name="web_sessions",
    version="3.1.0",
    owner="web-analytics",
    consumers=["marketing", "product"],
    on_violation="raise",
    schema=pb.Schema(session_id="String", duration_s="Int64", device="String"),
    steps=[
        pb.Step("col_vals_ge", columns="duration_s", value=0),
        pb.Step("col_vals_in_set", columns="device", set=["desktop", "mobile", "tablet"]),
    ],
)

print(sessions_contract.to_yaml())
```

    contract:
      name: web_sessions
      version: 3.1.0
      owner: web-analytics
      consumers:
      - marketing
      - product
      on_violation: raise
      schema:
        session_id: String
        duration_s: Int64
        device: String
      steps:
      - col_vals_ge:
          columns: duration_s
          value: 0
      - col_vals_in_set:
          columns: device
          set:
          - desktop
          - mobile
          - tablet

The output lists the metadata first, including `on_violation` because it isn't the default, then the schema and the steps. Each step is written as its method name with the arguments nested below it.

To save the contract, pass a file path. The parent directory is created if needed, and the YAML text is still returned. Reading the file back gives a contract equal to the original:

``` python
import tempfile
from pathlib import Path

contract_path = Path(tempfile.mkdtemp()) / "contracts" / "web_sessions.yaml"

_ = sessions_contract.to_yaml(contract_path)

pb.Contract.from_yaml(contract_path) == sessions_contract
```

    True

### from_yaml()

Read a contract from a YAML file.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L1121-L1246)

``` python
from_yaml(path)
```

Keeping contracts in YAML files separates the definition of "valid data" from the code that checks it. The files are easy to read, can be reviewed and versioned like any other change, and can be shared between the teams that produce and consume a dataset. This method loads such a file into a [Contract](../reference/Contract.md#pointblank.Contract), ready to use with [`validate()`](../reference/Contract.md#pointblank.Contract.validate) or a [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline).

The file can hold the contract under a top-level `contract:` key, as [`to_yaml()`](../reference/Contract.md#pointblank.Contract.to_yaml) writes it, or have the contract fields at the top level. The fields are the contract's parameter names (see [`from_dict()`](../reference/Contract.md#pointblank.Contract.from_dict) for the expected structure). A hand-written contract file might look like this:

``` yaml
contract:
  name: payments
  version: 1.0.0
  owner: billing-team
  schema:
    payment_id: String
    amount: Float64
  steps:
  - col_vals_not_null:
      columns: payment_id
  - col_vals_gt:
      columns: amount
      value: 0
  thresholds:
    warning: 1
```

#### Parameters

`path: str | Path`  
The path to the YAML file.

#### Returns

`Contract`  
A new [Contract](../reference/Contract.md#pointblank.Contract) object.

#### Raises

`FileNotFoundError`  
If no file exists at `path`.

`ValueError`  
If the file is empty, if it has no contract `name`, or if a field has an invalid value (see [`from_dict()`](../reference/Contract.md#pointblank.Contract.from_dict)).

#### Examples

Let's write the contract file shown above to a temporary directory, then load it.

``` python
import pointblank as pb
import polars as pl
import tempfile
from pathlib import Path

contract_yaml = '''
contract:
  name: payments
  version: 1.0.0
  owner: billing-team
  schema:
    payment_id: String
    amount: Float64
  steps:
  - col_vals_not_null:
      columns: payment_id
  - col_vals_gt:
      columns: amount
      value: 0
  thresholds:
    warning: 1
'''

contract_path = Path(tempfile.mkdtemp()) / "payments.yaml"
contract_path.write_text(contract_yaml)

payments_contract = pb.Contract.from_yaml(contract_path)

payments_contract
```

    Contract(name='payments', direction='source', version='1.0.0', schema=<defined>, steps=2)

The loaded contract works like one defined in Python. Here it is applied to a small payments table:

``` python
payments = pl.DataFrame({"payment_id": ["T1", "T2", "T3"], "amount": [50.0, 0.0, 12.5]})

payments_contract.validate(payments)
```

[TABLE]

The report label, `Contract: payments v1.0.0`, comes from the file. The payment with an amount of `0.0` fails the `col_vals_gt` step and triggers the warning threshold.

### to_dict()

Convert the contract to a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L905-L989)

``` python
to_dict()
```

This method gives you the contract as plain Python data (strings, numbers, lists, and dictionaries), ready to be written as JSON, stored in a database or configuration system, or compared with another contract's definition. It's the same structure that [`to_yaml()`](../reference/Contract.md#pointblank.Contract.to_yaml) writes, but without the top-level `contract` key. [`from_dict()`](../reference/Contract.md#pointblank.Contract.from_dict) turns it back into a [Contract](../reference/Contract.md#pointblank.Contract).

The dictionary always has a `"name"` key. Other fields are only included when they are set. `direction=` is left out when it's `"source"` and `on_violation=` when it's `"warn"`, because those are the defaults. The schema becomes a mapping of column names to type strings, each step becomes a single-key mapping from method name to arguments (see [`Step.to_dict()`](../reference/Step.md#pointblank.Step.to_dict)), and the thresholds become a mapping with the levels that are set.

#### Returns

`dict[str, Any]`  
A dictionary representation of the contract.

#### Examples

Let's create a contract for a target table and convert it to a dictionary.

``` python
import pointblank as pb

accounts_contract = pb.Contract(
    name="accounts",
    direction="target",
    version="1.0.0",
    schema=pb.Schema(account_id="Int64", email="String"),
    steps=[
        pb.Step("rows_distinct", columns_subset="account_id"),
        pb.Step("col_vals_regex", columns="email", pattern=r"^[^@]+@[^@]+$"),
    ],
    thresholds=pb.Thresholds(warning=0.01),
)

accounts_contract.to_dict()
```

    {'name': 'accounts',
     'direction': 'target',
     'version': '1.0.0',
     'schema': {'account_id': 'Int64', 'email': 'String'},
     'steps': [{'rows_distinct': {'columns_subset': 'account_id'}},
      {'col_vals_regex': {'columns': 'email', 'pattern': '^[^@]+@[^@]+$'}}],
     'thresholds': {'warning': 0.01}}

`direction` is included because `"target"` isn't the default, while `on_violation`, `owner`, and the other unset fields are left out. Only the warning level of the thresholds is set, so it's the only one listed.

A dictionary like this converts directly to JSON with the standard library:

``` python
import json

print(json.dumps(accounts_contract.to_dict(), indent=2))
```

    {
      "name": "accounts",
      "direction": "target",
      "version": "1.0.0",
      "schema": {
        "account_id": "Int64",
        "email": "String"
      },
      "steps": [
        {
          "rows_distinct": {
            "columns_subset": "account_id"
          }
        },
        {
          "col_vals_regex": {
            "columns": "email",
            "pattern": "^[^@]+@[^@]+$"
          }
        }
      ],
      "thresholds": {
        "warning": 0.01
      }
    }

### from_dict()

Create a contract from a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/contract.py#L991-L1119)

``` python
from_dict(data)
```

This is the reverse of [`to_dict()`](../reference/Contract.md#pointblank.Contract.to_dict). Use it when the contract comes from somewhere other than a YAML file: a JSON document, an API response, a database record, or a dictionary built in code. To read a YAML file directly, use [`from_yaml()`](../reference/Contract.md#pointblank.Contract.from_yaml).

The dictionary uses the contract's parameter names as keys. Only `"name"` is required. `"schema"` should map column names to type strings, `"steps"` should be a list of single-key mappings from method name to arguments, and `"thresholds"` should map any of `"warning"`, `"error"`, and `"critical"` to a value. The parsed contract is checked as usual, so an invalid `direction`, `on_violation`, or step method raises an error.

#### Parameters

`data: dict[str, Any]`  
A dictionary describing a contract, in the form produced by [`to_dict()`](../reference/Contract.md#pointblank.Contract.to_dict).

#### Returns

`Contract`  
A new [Contract](../reference/Contract.md#pointblank.Contract) object.

#### Raises

`ValueError`  
If `"name"` is missing or empty, if `"direction"` or `"on_violation"` has an invalid value, or if a step names a method that contracts don't support.

`TypeError`  
If `"steps"` isn't a list, or if a step's arguments aren't a dictionary.

#### Examples

Suppose a contract definition arrives as JSON from a configuration service. After parsing, it's a dictionary that `from_dict()` can turn into a [Contract](../reference/Contract.md#pointblank.Contract).

``` python
import json
import pointblank as pb

contract_json = '''
{
  "name": "shipments",
  "version": "0.3.0",
  "owner": "logistics",
  "schema": {"shipment_id": "String", "weight_kg": "Float64"},
  "steps": [
    {"col_vals_not_null": {"columns": "shipment_id"}},
    {"col_vals_between": {"columns": "weight_kg", "left": 0, "right": 1000}}
  ],
  "thresholds": {"warning": 1, "error": 0.05}
}
'''

shipments_contract = pb.Contract.from_dict(json.loads(contract_json))

shipments_contract
```

    Contract(name='shipments', direction='source', version='0.3.0', schema=<defined>, steps=2)

The result is a regular [Contract](../reference/Contract.md#pointblank.Contract), ready to use on data:

``` python
import polars as pl

shipments = pl.DataFrame(
    {"shipment_id": ["S1", "S2", "S3"], "weight_kg": [12.5, 980.0, 1250.0]}
)

shipments_contract.validate(shipments)
```

[TABLE]

The schema check and the not-null check pass, and the range check flags the shipment weighing 1250 kg.
