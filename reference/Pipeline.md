# Pipeline

Enforce data contracts at both ends of a data transformation.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L373-L1563)

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

The pipeline doesn't run or schedule your transformation logic beyond calling the function you give it. The transformation can use any DataFrame library, and the pipeline can sit inside any orchestration tool. You can also check just one boundary with [`validate_source()`](../reference/pointblank.Pipeline.validate_source.md) or [`validate_target()`](../reference/pointblank.Pipeline.validate_target.md), for example when the transformation runs somewhere else. Only one of the two contracts is required, so a pipeline can also act as an ingestion guard (source only) or an output guard (target only).

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
Whether [`run()`](../reference/pointblank.Pipeline.run.md) should stop when the source validation fails. With `True` (the default), the transformation isn't called and the target isn't validated, so bad input never reaches your transformation code or anything it writes to. Use `False` to always run the transformation and validate both boundaries, for example to see every problem at once while developing. A source contract with `on_violation="raise"` still raises with `False`, but only after the target has been validated.

## Returns

`Pipeline`  
A [Pipeline](../reference/Pipeline.md#pointblank.Pipeline) object. Use [`run()`](../reference/pointblank.Pipeline.run.md) to validate, transform, and validate again in one call, getting back a [`PipelineResult`](../reference/PipelineResult.md#pointblank.PipelineResult).

## Raises

`TypeError`  
If `source=` or `target=` is given and isn't a [Contract](../reference/Contract.md#pointblank.Contract).

`ValueError`  
If neither `source=` nor `target=` is given.

## How A Pipeline Runs

Calling [`run()`](../reference/pointblank.Pipeline.run.md) with a table and a transformation function goes through these steps:

1.  the source contract is compiled into a [`Validate`](../reference/Validate.md#pointblank.Validate) object for the table and interrogated
2.  if the source validation has any failing test units and `short_circuit=True`, the run stops here
3.  the transformation function is called with the original table, and its output is kept
4.  the target contract is compiled for the output and interrogated

Each boundary validation is built the same way as in [`Contract.to_validate()`](../reference/pointblank.Contract.to_validate.md): the table name is the contract name, the contract's owner, consumers, and version are recorded, the schema check (if any) comes first, and the steps follow in order. The pipeline adds its own label, thresholds, and actions. If one of the contracts is missing, its boundary is simply skipped.

## Passing And Failing

A boundary *passes* only when every test unit in every step passes, which is the same test as [`Validate.all_passed()`](../reference/Validate.all_passed.md#pointblank.Validate.all_passed). Thresholds don't change this: a single failing row makes the boundary fail, even if it's well below the warning level. For decisions based on threshold levels instead, inspect the boundary validations directly, for example with [`above_threshold()`](../reference/Validate.above_threshold.md#pointblank.Validate.above_threshold).

When a boundary fails, the pipeline applies that contract's `on_violation=` setting. With `"warn"` (the default), a `UserWarning` is issued. With `"log"`, a warning-level message is written to the `"pointblank.contract"` logger. With `"raise"`, a [`ContractViolationError`](../reference/ContractViolationError.md#pointblank.ContractViolationError) (a `RuntimeError`) is raised, which stops the code that called the pipeline. When it's raised by [`run()`](../reference/pointblank.Pipeline.run.md), its `result` attribute holds the [`PipelineResult`](../reference/PipelineResult.md#pointblank.PipelineResult) of the run, so both validations can still be inspected.

## Serialization

A pipeline can be stored as YAML with [`to_yaml()`](../reference/pointblank.Pipeline.to_yaml.md) and read back with [`from_yaml()`](../reference/pointblank.Pipeline.from_yaml.md), or converted to and from a dictionary with [`to_dict()`](../reference/pointblank.Pipeline.to_dict.md) and [`from_dict()`](../reference/pointblank.Pipeline.from_dict.md). The stored form holds the label, thresholds, and `short_circuit=` setting under a `pipeline` key, plus the two contracts under `source` and `target`. Actions and final actions are functions or messages tied to your environment, so they aren't stored. Pass them again when you recreate the pipeline. The transformation isn't part of the pipeline either: it's given to [run()](../reference/pointblank.Pipeline.run.md) each time.

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

Now we can give the pipeline a day of raw orders and the transformation function. [`run()`](../reference/pointblank.Pipeline.run.md) validates the input, transforms it, and validates the output.

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
The result of [run()](../reference/pointblank.Pipeline.run.md), holding both boundary validations.

[Contract](../reference/Contract.md#pointblank.Contract)  
Define the source and target contracts.

[Step](../reference/Step.md#pointblank.Step)  
A single validation rule within a contract.

## Functions

| Name | Description |
|----|----|
| [run()](pointblank.Pipeline.run.md) | Validate the input, transform it, and validate the output. |
| [validate_source()](pointblank.Pipeline.validate_source.md) | Check incoming data against the source contract. |
| [validate_target()](pointblank.Pipeline.validate_target.md) | Check transformed data against the target contract. |
| [to_yaml()](pointblank.Pipeline.to_yaml.md) | Write the pipeline as YAML. |
| [from_yaml()](pointblank.Pipeline.from_yaml.md) | Read a pipeline from a YAML file. |
| [to_dict()](pointblank.Pipeline.to_dict.md) | Convert the pipeline to a dictionary. |
| [from_dict()](pointblank.Pipeline.from_dict.md) | Create a pipeline from a dictionary. |
