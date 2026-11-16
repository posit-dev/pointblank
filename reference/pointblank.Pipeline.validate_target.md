# validate_target()

Check transformed data against the target contract.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L746-L851)

``` python
validate_target(data)
```

This method validates one table against the pipeline's target contract and returns the interrogated [`Validate`](../reference/Validate.md#pointblank.Validate) object. It's the outbound counterpart of [`validate_source()`](../reference/pointblank.Pipeline.validate_source.md). Use it to check the output of a transformation that runs outside the pipeline (for example, a table produced by a SQL model or a separate task) before it's published to the people and systems that depend on it.

The validation is built the same way as in [`run()`](../reference/pointblank.Pipeline.run.md), with the pipeline's label, thresholds, and actions. If the data fails, the target contract's `on_violation=` setting is applied, and a `"raise"` violation propagates as a [`ContractViolationError`](../reference/ContractViolationError.md#pointblank.ContractViolationError).

## Parameters

`data: IntoDataFrame`  
The transformed table. Any table type that [Validate](../reference/Validate.md#pointblank.Validate) accepts can be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

## Returns

`Validate`  
An interrogated [Validate](../reference/Validate.md#pointblank.Validate) object with the results of every target contract check.

## Raises

`ValueError`  
If the pipeline has no target contract.

`ContractViolationError`  
If the data fails the target contract and the contract has `on_violation="raise"`. It's a subclass of `RuntimeError`, and its `validation` attribute holds the failing validation.

## Examples

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

The [rows_distinct](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step fails for both copies of `"2025-03-02"`, and a message is written to the `"pointblank.contract"` logger. Because the method returns the validation, the job can decide what to do with the result:

``` python
validation.all_passed()
```

    False

Here the job would hold the summary back until the duplicate day is resolved. To make a failure stop the job instead, set `on_violation="raise"` on the contract, which turns a failing validation into a [ContractViolationError](../reference/ContractViolationError.md#pointblank.ContractViolationError).
