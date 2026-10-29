# Pipeline

Enforce data contracts at both ends of a data transformation.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L280-L1441)

``` python
Pipeline(
    source=None,
    target=None,
    thresholds=None,
    actions=None,
    final_actions=None,
    label=None,
    short_circuit=True,
)
```

Most data problems are introduced at a boundary: an upstream system starts sending nulls or a new category, or a transformation step drops, duplicates, or miscalculates rows. A [Pipeline](../reference/Pipeline.md#pointblank.Pipeline) guards those boundaries. It pairs a *source* [`Contract`](../reference/Contract.md#pointblank.Contract), describing the data a transformation expects to receive, with a *target* contract, describing the data it promises to produce. Data is validated against the source contract on the way in and against the target contract on the way out, so a failure tells you whether the input was bad or the transformation went wrong.

The pipeline doesn't run or schedule your transformation logic beyond calling the function you give it. The transformation can use any DataFrame library, and the pipeline can sit inside any orchestration tool. You can also check just one boundary with [`validate_source()`](../reference/Pipeline.md#pointblank.Pipeline.validate_source) or [`validate_target()`](../reference/Pipeline.md#pointblank.Pipeline.validate_target), for example when the transformation runs somewhere else. Only one of the two contracts is required, so a pipeline can also act as an ingestion guard (source only) or an output guard (target only).

## Parameters

`source: Contract | None = None`  
The [`Contract`](../reference/Contract.md#pointblank.Contract) that incoming data must meet. Leave it as `None` for a pipeline that only checks output.

`target: Contract | None = None`  
The [`Contract`](../reference/Contract.md#pointblank.Contract) that the transformed data must meet. Leave it as `None` for a pipeline that only checks input. At least one of `source=` and `target=` must be given.

`thresholds: Thresholds | None = None`  
[`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) to use for both boundary validations. When given, they replace each contract's own `thresholds=` (they aren't merged), which lets the same contracts be applied more or less strictly in different settings. Thresholds set on individual steps still take precedence. Thresholds determine the warning, error, and critical levels shown in the reports and the [`Actions`](../reference/Actions.md#pointblank.Actions) they trigger, but not whether a boundary passes (see *Passing and Failing* below).

`actions: Actions | None = None`  
[`Actions`](../reference/Actions.md#pointblank.Actions) to take when a step in either boundary validation reaches a threshold level. The same actions are used for both boundaries.

`final_actions: FinalActions | None = None`  
[`FinalActions`](../reference/FinalActions.md#pointblank.FinalActions) to run when a boundary validation finishes. They run once for each boundary that's validated, so twice in a full run.

`label: str | None = None`  
A label for the pipeline. Each boundary validation is labeled `"<label> [source]"` or `"<label> [target]"`. Without a label, the contract name is used instead, as in `"Pipeline: raw_orders [source]"`.

`short_circuit: bool = True`  
Whether [`run()`](../reference/Pipeline.md#pointblank.Pipeline.run) should stop when the source validation fails. With `True` (the default), the transformation isn't called and the target isn't validated, so bad input never reaches your transformation code or anything it writes to. Use `False` to always run the transformation and validate both boundaries, for example to see every problem at once while developing.

## Returns

`Pipeline`  
A [Pipeline](../reference/Pipeline.md#pointblank.Pipeline) object. Use [`run()`](../reference/Pipeline.md#pointblank.Pipeline.run) to validate, transform, and validate again in one call, getting back a [`PipelineResult`](../reference/PipelineResult.md#pointblank.PipelineResult).

## Raises

`TypeError`  
If `source=` or `target=` is given and isn't a [Contract](../reference/Contract.md#pointblank.Contract).

`ValueError`  
If neither `source=` nor `target=` is given.

## How A Pipeline Runs

Calling [`run()`](../reference/Pipeline.md#pointblank.Pipeline.run) with a table and a transformation function goes through these steps:

1.  the source contract is compiled into a [`Validate`](../reference/Validate.md#pointblank.Validate) object for the table and interrogated
2.  if the source validation has any failing test units and `short_circuit=True`, the run stops here
3.  the transformation function is called with the original table, and its output is kept
4.  the target contract is compiled for the output and interrogated

Each boundary validation is built the same way as in [`Contract.to_validate()`](../reference/Contract.md#pointblank.Contract.to_validate): the table name is the contract name, the contract's owner, consumers, and version are recorded, the schema check (if any) comes first, and the steps follow in order. The pipeline adds its own label, thresholds, and actions. If one of the contracts is missing, its boundary is simply skipped.

## Passing And Failing

A boundary *passes* only when every test unit in every step passes, which is the same test as [`Validate.all_passed()`](../reference/Validate.all_passed.md#pointblank.Validate.all_passed). Thresholds don't change this: a single failing row makes the boundary fail, even if it's well below the warning level. For decisions based on threshold levels instead, inspect the boundary validations directly, for example with [`above_threshold()`](../reference/Validate.above_threshold.md#pointblank.Validate.above_threshold).

When a boundary fails, the pipeline applies that contract's `on_violation=` setting. With `"warn"` (the default), a `UserWarning` is issued. With `"log"`, a warning-level message is written to the `"pointblank.contract"` logger. With `"raise"`, [`validate_source()`](../reference/Pipeline.md#pointblank.Pipeline.validate_source) and [`validate_target()`](../reference/Pipeline.md#pointblank.Pipeline.validate_target) raise a `RuntimeError`. [`run()`](../reference/Pipeline.md#pointblank.Pipeline.run) is different: it always returns a [`PipelineResult`](../reference/PipelineResult.md#pointblank.PipelineResult) so that both validations can be inspected, so a `"raise"` violation is recorded in the result without an exception or warning. To stop a job after [run()](../reference/Pipeline.md#pointblank.Pipeline.run), check [`PipelineResult.passed`](../reference/PipelineResult.md#pointblank.PipelineResult.passed).

## Serialization

A pipeline can be stored as YAML with [`to_yaml()`](../reference/Pipeline.md#pointblank.Pipeline.to_yaml) and read back with [`from_yaml()`](../reference/Pipeline.md#pointblank.Pipeline.from_yaml), or converted to and from a dictionary with [`to_dict()`](../reference/Pipeline.md#pointblank.Pipeline.to_dict) and [`from_dict()`](../reference/Pipeline.md#pointblank.Pipeline.from_dict). The stored form holds the label, thresholds, and `short_circuit=` setting under a `pipeline` key, plus the two contracts under `source` and `target`. Actions and final actions are functions or messages tied to your environment, so they aren't stored. Pass them again when you recreate the pipeline. The transformation isn't part of the pipeline either: it's given to [run()](../reference/Pipeline.md#pointblank.Pipeline.run) each time.

## Examples

Let's build a pipeline around a transformation that prepares raw order records for reporting. The raw records store amounts in cents with lowercase currency codes. The source contract says what the transformation needs from them: an order ID on every record, no negative amounts, and a known currency.

``` python
import pointblank as pb
import polars as pl

raw_orders_contract = pb.Contract(
    name="raw_orders",
    steps=[
        pb.Step("col_vals_not_null", columns="order_id"),
        pb.Step("col_vals_ge", columns="amount_cents", value=0),
        pb.Step("col_vals_in_set", columns="currency", set=["usd", "eur", "gbp"]),
    ],
)
```

The target contract describes what the transformation promises to deliver: a fixed schema, one row per order, positive amounts, and uppercase currency codes.

``` python
orders_contract = pb.Contract(
    name="orders",
    direction="target",
    schema=pb.Schema(order_id="String", amount="Float64", currency="String"),
    steps=[
        pb.Step("rows_distinct", columns_subset="order_id"),
        pb.Step("col_vals_gt", columns="amount", value=0),
        pb.Step("col_vals_in_set", columns="currency", set=["USD", "EUR", "GBP"]),
    ],
)

pipeline = pb.Pipeline(source=raw_orders_contract, target=orders_contract, label="Orders")

pipeline
```

    Pipeline(
      label='Orders',
      source='raw_orders',
      target='orders',
      short_circuit=True
    )

The transformation is an ordinary function that takes a table and returns a table. This one removes zero-amount test orders, converts cents to whole currency units, and uppercases the currency codes.

``` python
def prepare_orders(df: pl.DataFrame) -> pl.DataFrame:
    return df.filter(pl.col("amount_cents") > 0).select(
        "order_id",
        amount=pl.col("amount_cents") / 100,
        currency=pl.col("currency").str.to_uppercase(),
    )
```

Now we can give the pipeline a day of raw orders and the transformation function. [`run()`](../reference/Pipeline.md#pointblank.Pipeline.run) validates the input, transforms it, and validates the output.

``` python
raw_orders = pl.DataFrame(
    {
        "order_id": ["A1", "A2", "A3", "A4"],
        "amount_cents": [2500, 0, 11050, 899],
        "currency": ["usd", "eur", "usd", "gbp"],
    }
)

result = pipeline.run(data=raw_orders, transform=prepare_orders)

result
```

    PipelineResult(
      source=validated,
      target=validated,
      passed=True
    )

Both boundaries pass, so the [`PipelineResult`](../reference/PipelineResult.md#pointblank.PipelineResult) reports `passed=True`. Its `transform_output` attribute holds the prepared table, ready to be written out:

``` python
result.transform_output
```

shape: (3, 3)

| order_id | amount | currency |
|----------|--------|----------|
| str      | f64    | str      |
| "A1"     | 25.0   | "USD"    |
| "A3"     | 110.5  | "USD"    |
| "A4"     | 8.99   | "GBP"    |

Each boundary validation is kept as an interrogated [Validate](../reference/Validate.md#pointblank.Validate) object, so the full report is available. Here's the target validation, with the label built from the pipeline label and the contract's direction.

``` python
result.target_validation
```

[TABLE]

The next day's raw orders have problems: a record with no order ID, a negative amount, and a currency the transformation can't handle. Since `short_circuit=` is `True` by default, the pipeline stops after the source validation fails. The source contract uses the default `on_violation="warn"`, so a warning is also issued.

``` python
bad_raw_orders = pl.DataFrame(
    {
        "order_id": ["B1", None, "B3", "B4"],
        "amount_cents": [1200, 3400, -500, 760],
        "currency": ["usd", "usd", "cad", "eur"],
    }
)

result = pipeline.run(data=bad_raw_orders, transform=prepare_orders)

print(f"passed: {result.passed}")
print(f"transform output: {result.transform_output}")
```

    passed: False
    transform output: None

    UserWarning: Contract 'raw_orders' (source) violated: validation has failing steps.

The transformation never ran, so there's no output and no target validation. The source validation shows why the input was rejected, with one failing test unit in each step:

``` python
result.source_validation
```

[TABLE]

The batch was caught at the source boundary, so nothing reached the target table. Once the upstream data is fixed, the same pipeline can be run again on it without any changes.

## See Also

[PipelineResult](../reference/PipelineResult.md#pointblank.PipelineResult)  
The result of [run()](../reference/Pipeline.md#pointblank.Pipeline.run), holding both boundary validations.

[Contract](../reference/Contract.md#pointblank.Contract)  
Define the source and target contracts.

[Step](../reference/Step.md#pointblank.Step)  
A single validation rule within a contract.

## Methods

| Name | Description |
|----|----|
| [run()](#run) | Validate the input, transform it, and validate the output. |
| [validate_source()](#validate_source) | Check incoming data against the source contract. |
| [validate_target()](#validate_target) | Check transformed data against the target contract. |
| [to_yaml()](#to_yaml) | Write the pipeline as YAML. |
| [from_yaml()](#from_yaml) | Read a pipeline from a YAML file. |
| [to_dict()](#to_dict) | Convert the pipeline to a dictionary. |
| [from_dict()](#from_dict) | Create a pipeline from a dictionary. |

### run()

Validate the input, transform it, and validate the output.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L756-L936)

``` python
run(
    data,
    transform,
)
```

This is the main way to use a pipeline. It checks `data` against the source contract, calls `transform` on it, checks the output against the target contract, and returns everything in a [`PipelineResult`](../reference/PipelineResult.md#pointblank.PipelineResult): the two interrogated validations, the transformed data, and pass/fail properties for each boundary and overall. A failure is attributed to a boundary, so you can tell whether the input was bad or the transformation went wrong.

If the source validation has any failing test units and the pipeline has `short_circuit=True` (the default), the run stops there. The transformation isn't called, and the result has no transformed data and no target validation. With `short_circuit=False`, the transformation is called and the target is validated whatever the outcome of the source validation. If the pipeline has only one contract, the other boundary is skipped.

When a boundary fails, its contract's `on_violation=` setting is applied: `"warn"` issues a `UserWarning` and `"log"` writes to the `"pointblank.contract"` logger. A `"raise"` violation doesn't raise here, though, because [run()](../reference/Pipeline.md#pointblank.Pipeline.run) always returns its result so that both boundaries can be inspected. To stop a job when the pipeline fails, check the result's [`passed`](../reference/PipelineResult.md#pointblank.PipelineResult.passed) property, or use [`validate_source()`](../reference/Pipeline.md#pointblank.Pipeline.validate_source) and [`validate_target()`](../reference/Pipeline.md#pointblank.Pipeline.validate_target), which do raise.

#### Parameters

`data: IntoDataFrame`  
The input table. Any table type that [Validate](../reference/Validate.md#pointblank.Validate) accepts can be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

`transform: Callable[[Any], Any]`  
A function that takes the input table and returns the transformed table. It's called with `data` itself, and its return value is validated against the target contract, so it must also be a table that [Validate](../reference/Validate.md#pointblank.Validate) accepts. The function is called at most once per run. An exception raised by it isn't caught, so it propagates from [run()](../reference/Pipeline.md#pointblank.Pipeline.run).

#### Returns

`PipelineResult`  
The results of the run. `source_validation` and `target_validation` hold the interrogated [Validate](../reference/Validate.md#pointblank.Validate) objects (or `None` for a boundary that wasn't validated), and `transform_output` holds the transformed table (or `None` if the run stopped early).

#### Examples

Let's set up a pipeline for a customer table. The raw data must have an ID and a sign-up date for every customer, and the cleaned output must have one row per customer and a two-letter country code.

``` python
import pointblank as pb
import polars as pl

pipeline = pb.Pipeline(
    source=pb.Contract(
        name="raw_customers",
        steps=[pb.Step("col_vals_not_null", columns=["customer_id", "signup_date"])],
    ),
    target=pb.Contract(
        name="customers",
        direction="target",
        steps=[
            pb.Step("rows_distinct", columns_subset="customer_id"),
            pb.Step("col_vals_regex", columns="country", pattern=r"^[A-Z]{2}$"),
        ],
    ),
)

raw_customers = pl.DataFrame(
    {
        "customer_id": [1, 2, 3, 3],
        "signup_date": ["2025-01-04", "2025-01-09", "2025-02-11", "2025-02-11"],
        "country": ["us", "ca", "gb", "gb"],
    }
)
```

The raw table meets the source contract, but our transformation has a bug: it uppercases the country codes but forgets to remove the duplicated record for customer 3.

``` python
def clean_customers(df):
    return df.with_columns(pl.col("country").str.to_uppercase())

result = pipeline.run(data=raw_customers, transform=clean_customers)

print(f"source passed: {result.source_passed}")
print(f"target passed: {result.target_passed}")
```

    source passed: True
    target passed: False

    UserWarning: Contract 'customers' (target) violated: validation has failing steps.

The input was fine and the output wasn't, which points to the transformation. In the target validation, the `rows_distinct` step flags both rows for customer 3. The default `on_violation="warn"` setting also issued a warning.

``` python
result.target_validation
```

[TABLE]

During development, it's often useful to see the problems at both boundaries in a single run, even when the input is bad. A pipeline with `short_circuit=False` always runs the transformation and the target validation. Here, the input is missing a sign-up date, and the corrected transformation runs anyway:

``` python
dev_pipeline = pb.Pipeline(
    source=pipeline.source, target=pipeline.target, short_circuit=False
)

raw_customers_2 = pl.DataFrame(
    {
        "customer_id": [4, 5],
        "signup_date": ["2025-03-01", None],
        "country": ["fr", "de"],
    }
)

result = dev_pipeline.run(
    data=raw_customers_2,
    transform=lambda df: clean_customers(df).unique(subset="customer_id"),
)

print(result.get_report())
```

    Pipeline Boundary Validation Results
    ========================================

    Source Boundary: FAILED
    ------------------------------
      Steps: 2
      Test units passed: 3
      Test units failed: 1

    Target Boundary: PASSED
    ------------------------------
      Steps: 2
      Test units passed: 4
      Test units failed: 0

    Overall: FAILED

    UserWarning: Contract 'raw_customers' (source) violated: validation has failing steps.

The summary from [`get_report()`](../reference/PipelineResult.md#pointblank.PipelineResult.get_report) shows that the source boundary failed with one failing test unit, while the target boundary passed. Overall, the run failed.

### validate_source()

Check incoming data against the source contract.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L551-L649)

``` python
validate_source(data)
```

This method validates one table against the pipeline's source contract and returns the interrogated [`Validate`](../reference/Validate.md#pointblank.Validate) object. Use it when you want an ingestion check without running a transformation through the pipeline: for example, when data lands in one task of an orchestrated workflow and is transformed in another, or when the transformation is written in SQL or another tool. Pair it with [`validate_target()`](../reference/Pipeline.md#pointblank.Pipeline.validate_target) to check the other boundary.

The validation is built the same way as in [`run()`](../reference/Pipeline.md#pointblank.Pipeline.run), with the pipeline's label, thresholds, and actions. If the data fails, the source contract's `on_violation=` setting is applied. Unlike [run()](../reference/Pipeline.md#pointblank.Pipeline.run), this method lets a `"raise"` violation propagate, which makes it a simple gate: with `on_violation="raise"`, the code after the call only runs when the data meets the contract.

#### Parameters

`data: IntoDataFrame`  
The incoming table. Any table type that [Validate](../reference/Validate.md#pointblank.Validate) accepts can be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

#### Returns

`Validate`  
An interrogated [Validate](../reference/Validate.md#pointblank.Validate) object with the results of every source contract check.

#### Raises

`ValueError`  
If the pipeline has no source contract.

`RuntimeError`  
If the data fails the source contract and the contract has `on_violation="raise"`.

#### Examples

Let's make a pipeline that guards the ingestion of sensor readings. The source contract uses `on_violation="raise"`, so readings that don't meet it are rejected with an error.

``` python
import pointblank as pb
import polars as pl

readings_contract = pb.Contract(
    name="sensor_readings",
    steps=[
        pb.Step("col_vals_not_null", columns="sensor_id"),
        pb.Step("col_vals_between", columns="temp_c", left=-40, right=60),
    ],
    on_violation="raise",
)

ingest = pb.Pipeline(source=readings_contract, label="Sensor ingest")
```

A batch of good readings passes, and we get the validation back:

``` python
readings = pl.DataFrame({"sensor_id": ["s1", "s2", "s3"], "temp_c": [21.5, 19.0, 22.1]})

ingest.validate_source(readings)
```

[TABLE]

A batch with an impossible temperature raises a `RuntimeError` naming the contract, so a job that loads the data can stop before anything is written.

``` python
bad_readings = pl.DataFrame({"sensor_id": ["s1", "s2"], "temp_c": [21.5, 85.0]})

try:
    ingest.validate_source(bad_readings)
except RuntimeError as e:
    print(f"Rejected: {e}")
```

    Rejected: Contract 'sensor_readings' (source) violated: validation has failing steps.

The message names the contract and its direction. If the contract used `on_violation="warn"` or `"log"` instead, the call would return the failing validation rather than raising, leaving it to the caller to decide what to do.

### validate_target()

Check transformed data against the target contract.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L651-L754)

``` python
validate_target(data)
```

This method validates one table against the pipeline's target contract and returns the interrogated [`Validate`](../reference/Validate.md#pointblank.Validate) object. It's the outbound counterpart of [`validate_source()`](../reference/Pipeline.md#pointblank.Pipeline.validate_source). Use it to check the output of a transformation that runs outside the pipeline (for example, a table produced by a SQL model or a separate task) before it's published to the people and systems that depend on it.

The validation is built the same way as in [`run()`](../reference/Pipeline.md#pointblank.Pipeline.run), with the pipeline's label, thresholds, and actions. If the data fails, the target contract's `on_violation=` setting is applied, and a `"raise"` violation propagates as a `RuntimeError`.

#### Parameters

`data: IntoDataFrame`  
The transformed table. Any table type that [Validate](../reference/Validate.md#pointblank.Validate) accepts can be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

#### Returns

`Validate`  
An interrogated [Validate](../reference/Validate.md#pointblank.Validate) object with the results of every target contract check.

#### Raises

`ValueError`  
If the pipeline has no target contract.

`RuntimeError`  
If the data fails the target contract and the contract has `on_violation="raise"`.

#### Examples

Suppose a daily revenue summary is built by a separate SQL job, and we want to check it before it goes to a dashboard. The target contract requires one row per day and non-negative revenue. It uses `on_violation="log"`, so failures are logged rather than raised.

``` python
import pointblank as pb
import polars as pl

summary_contract = pb.Contract(
    name="daily_revenue",
    direction="target",
    steps=[
        pb.Step("rows_distinct", columns_subset="day"),
        pb.Step("col_vals_ge", columns="revenue", value=0),
    ],
    on_violation="log",
)

publish = pb.Pipeline(target=summary_contract, label="Revenue summary")
```

Here's the job's output, which has a repeated day:

``` python
summary = pl.DataFrame(
    {
        "day": ["2025-03-01", "2025-03-02", "2025-03-02"],
        "revenue": [1520.0, 1785.5, 1785.5],
    }
)

validation = publish.validate_target(summary)

validation
```

    Contract 'daily_revenue' (target) violated: validation has failing steps.

[TABLE]

The `rows_distinct` step fails for both copies of `"2025-03-02"`, and a message is written to the `"pointblank.contract"` logger. Because the method returns the validation, the job can decide what to do with the result:

``` python
validation.all_passed()
```

    False

Here the job would hold the summary back until the duplicate day is resolved. To make a failure stop the job instead, set `on_violation="raise"` on the contract, which turns a failing validation into a `RuntimeError`.

### to_yaml()

Write the pipeline as YAML.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L1352-L1429)

``` python
to_yaml(path=None)
```

YAML is a convenient way to keep a pipeline's contracts and settings under version control next to the transformation code, so changes to them go through review. This method writes the pipeline settings under a `pipeline:` key, followed by the contracts under `source:` and `target:`, in the structure described in [`to_dict()`](../reference/Pipeline.md#pointblank.Pipeline.to_dict). [`from_yaml()`](../reference/Pipeline.md#pointblank.Pipeline.from_yaml) reads the file back into an equivalent [Pipeline](../reference/Pipeline.md#pointblank.Pipeline).

The pipeline's actions and final actions aren't written. Contract step arguments must be plain data (strings, numbers, booleans, dates, lists, and dictionaries), as for [`Contract.to_yaml()`](../reference/Contract.md#pointblank.Contract.to_yaml).

#### Parameters

`path: str | None = None`  
An optional path to write the YAML to. Any missing parent directories are created, and an existing file is overwritten. If `None` (the default), nothing is written.

#### Returns

`str`  
The YAML text. It's returned whether or not `path=` is given.

#### Raises

`ValueError`  
If a contract step argument can't be represented in YAML (for example, a column selector or a Polars expression). The message names the step and argument. Nothing is written to `path=` in this case.

#### Examples

Let's create a pipeline for a product catalog and get its YAML representation.

``` python
import pointblank as pb

catalog_pipeline = pb.Pipeline(
    source=pb.Contract(
        name="supplier_feed",
        steps=[pb.Step("col_vals_not_null", columns=["sku", "price"])],
    ),
    target=pb.Contract(
        name="catalog",
        direction="target",
        schema=pb.Schema(sku="String", price="Float64"),
        steps=[pb.Step("rows_distinct", columns_subset="sku")],
        on_violation="raise",
    ),
    thresholds=pb.Thresholds(warning=1, error=0.05),
    label="Catalog refresh",
)

print(catalog_pipeline.to_yaml())
```

    pipeline:
      label: Catalog refresh
      thresholds:
        warning: 1
        error: 0.05
    source:
      name: supplier_feed
      steps:
      - col_vals_not_null:
          columns:
          - sku
          - price
    target:
      name: catalog
      direction: target
      on_violation: raise
      schema:
        sku: String
        price: Float64
      steps:
      - rows_distinct:
          columns_subset: sku

To save the file, pass a path, for example `catalog_pipeline.to_yaml("contracts/catalog_pipeline.yaml")`. The YAML text is still returned.

### from_yaml()

Read a pipeline from a YAML file.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L1222-L1350)

``` python
from_yaml(path)
```

Storing a pipeline in YAML keeps the definition of what valid input and output look like separate from the transformation code, where it can be reviewed and versioned on its own. This method loads a file written by [`to_yaml()`](../reference/Pipeline.md#pointblank.Pipeline.to_yaml), or written by hand in the same structure (see [`from_dict()`](../reference/Pipeline.md#pointblank.Pipeline.from_dict) for the details). A hand-written file might look like this:

``` yaml
pipeline:
  label: Shipments
source:
  name: raw_shipments
  steps:
  - col_vals_not_null:
      columns: shipment_id
target:
  name: shipments
  steps:
  - col_vals_between:
      columns: weight_kg
      left: 0
      right: 1000
```

The transformation, actions, and final actions aren't part of the file. Pass the transformation to [`run()`](../reference/Pipeline.md#pointblank.Pipeline.run) as usual.

#### Parameters

`path: str`  
The path to the YAML file.

#### Returns

`Pipeline`  
A new [Pipeline](../reference/Pipeline.md#pointblank.Pipeline) object.

#### Raises

`FileNotFoundError`  
If no file exists at `path`.

`ValueError`  
If the file is empty, if it has neither a `source` nor a `target` contract, or if a contract is invalid (see [`Contract.from_dict()`](../reference/Contract.md#pointblank.Contract.from_dict)).

#### Examples

Let's write the pipeline file shown above to a temporary directory, then load it.

``` python
import pointblank as pb
import polars as pl
import tempfile
from pathlib import Path

pipeline_yaml = '''
pipeline:
  label: Shipments
source:
  name: raw_shipments
  steps:
  - col_vals_not_null:
      columns: shipment_id
target:
  name: shipments
  steps:
  - col_vals_between:
      columns: weight_kg
      left: 0
      right: 1000
'''

pipeline_path = Path(tempfile.mkdtemp()) / "shipments.yaml"
pipeline_path.write_text(pipeline_yaml)

shipments_pipeline = pb.Pipeline.from_yaml(pipeline_path)

shipments_pipeline
```

    Pipeline(
      label='Shipments',
      source='raw_shipments',
      target='shipments',
      short_circuit=True
    )

The loaded pipeline works like one defined in Python. Here it runs a transformation that converts weights from pounds to kilograms:

``` python
raw_shipments = pl.DataFrame({"shipment_id": ["S1", "S2"], "weight_lb": [26.5, 410.0]})

result = shipments_pipeline.run(
    data=raw_shipments,
    transform=lambda df: df.select(
        "shipment_id", weight_kg=pl.col("weight_lb") * 0.4536
    ),
)

result.passed
```

    True

Both boundaries pass, since the heavier shipment (410 lb, or about 186 kg) is well within the target's range of 0 to 1000 kg. The transformation stays in Python while the contracts live in the file, so the rules can be changed without touching the job's code.

### to_dict()

Convert the pipeline to a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L1028-L1098)

``` python
to_dict()
```

This method gives you the pipeline's definition as plain Python data, ready to be written as JSON or stored in a configuration system. It's the same structure that [`to_yaml()`](../reference/Pipeline.md#pointblank.Pipeline.to_yaml) writes, and [`from_dict()`](../reference/Pipeline.md#pointblank.Pipeline.from_dict) turns it back into a [Pipeline](../reference/Pipeline.md#pointblank.Pipeline).

The dictionary has up to three keys. `"pipeline"` holds the pipeline's own settings: the `"label"` and `"thresholds"` (when set) and `"short_circuit"` (only when it's `False`, since `True` is the default). This entry is always present, even when it's empty. `"source"` and `"target"` hold the contracts, in the form produced by [`Contract.to_dict()`](../reference/Contract.md#pointblank.Contract.to_dict), and each is left out if the pipeline doesn't have that contract. The pipeline's actions and final actions aren't included.

#### Returns

`dict[str, Any]`  
A dictionary representation of the pipeline.

#### Examples

Let's create a pipeline with a label, thresholds, and short-circuiting turned off, and convert it to a dictionary.

``` python
import pointblank as pb

pipeline = pb.Pipeline(
    source=pb.Contract(
        name="raw_events", steps=[pb.Step("col_vals_not_null", columns="event_id")]
    ),
    target=pb.Contract(
        name="events",
        direction="target",
        steps=[pb.Step("rows_distinct", columns_subset="event_id")],
    ),
    thresholds=pb.Thresholds(warning=0.01, error=0.05),
    label="Event stream",
    short_circuit=False,
)

pipeline.to_dict()
```

    {'pipeline': {'label': 'Event stream',
      'thresholds': {'warning': 0.01, 'error': 0.05},
      'short_circuit': False},
     'source': {'name': 'raw_events',
      'steps': [{'col_vals_not_null': {'columns': 'event_id'}}]},
     'target': {'name': 'events',
      'direction': 'target',
      'steps': [{'rows_distinct': {'columns_subset': 'event_id'}}]}}

The pipeline settings come first, followed by each contract. The source contract has no `direction` key because `"source"` is the default direction.

### from_dict()

Create a pipeline from a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L1100-L1220)

``` python
from_dict(data)
```

This is the reverse of [`to_dict()`](../reference/Pipeline.md#pointblank.Pipeline.to_dict). Use it when the pipeline definition comes from somewhere other than a YAML file, such as a JSON document or a configuration service. To read a YAML file directly, use [`from_yaml()`](../reference/Pipeline.md#pointblank.Pipeline.from_yaml).

The dictionary can have a `"pipeline"` entry with any of `"label"`, `"thresholds"` (a mapping of `"warning"`, `"error"`, and `"critical"` to values), and `"short_circuit"`. Settings that are missing get their defaults. The contracts go under `"source"` and `"target"` in the form read by [`Contract.from_dict()`](../reference/Contract.md#pointblank.Contract.from_dict). A contract without a `"direction"` key is given the direction that matches its position, so `direction:` doesn't need to be repeated in hand-written definitions. Actions and final actions can't be given in the dictionary. To use them, create a new pipeline with the parsed contracts and settings plus your actions.

#### Parameters

`data: dict[str, Any]`  
A dictionary describing a pipeline, in the form produced by [`to_dict()`](../reference/Pipeline.md#pointblank.Pipeline.to_dict).

#### Returns

`Pipeline`  
A new [Pipeline](../reference/Pipeline.md#pointblank.Pipeline) object.

#### Raises

`ValueError`  
If the dictionary has neither a `"source"` nor a `"target"` contract, or if a contract is invalid (see [`Contract.from_dict()`](../reference/Contract.md#pointblank.Contract.from_dict)).

#### Examples

Here's a pipeline definition written as JSON. It only has a target contract, so it acts as an output guard.

``` python
import json
import pointblank as pb
import polars as pl

pipeline_json = '''
{
  "pipeline": {"label": "Invoice export", "thresholds": {"warning": 1}},
  "target": {
    "name": "invoices",
    "steps": [
      {"col_vals_not_null": {"columns": "invoice_id"}},
      {"col_vals_gt": {"columns": "total", "value": 0}}
    ]
  }
}
'''

export_pipeline = pb.Pipeline.from_dict(json.loads(pipeline_json))

export_pipeline
```

    Pipeline(
      label='Invoice export',
      target='invoices',
      short_circuit=True
    )

The target contract didn't specify a direction, so it was given `"target"`. The pipeline can now check an export:

``` python
invoices = pl.DataFrame({"invoice_id": ["I-1", "I-2", "I-3"], "total": [120.0, 0.0, 75.5]})

export_pipeline.validate_target(invoices)
```

    UserWarning: Contract 'invoices' (target) violated: validation has failing steps.

[TABLE]

The invoice with a total of `0.0` fails the `col_vals_gt` step and reaches the warning threshold set in the pipeline definition.
