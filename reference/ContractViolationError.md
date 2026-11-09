# ContractViolationError

The error raised when data fails a contract with `on_violation="raise"`.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L283-L370)

``` python
ContractViolationError(
    message,
    contract,
    validation,
    result=None,
)
```

A [`Pipeline`](../reference/Pipeline.md#pointblank.Pipeline) raises this error when a boundary validation has any failing test units and that boundary's [`Contract`](../reference/Contract.md#pointblank.Contract) was created with `on_violation="raise"`. It's raised by [`run()`](../reference/Pipeline.md#pointblank.Pipeline.run), [`validate_source()`](../reference/Pipeline.md#pointblank.Pipeline.validate_source), and [`validate_target()`](../reference/Pipeline.md#pointblank.Pipeline.validate_target). The message names the contract and its direction, and the error carries the failing validation, so the code that catches it can report or log the details before stopping.

It's a subclass of `RuntimeError`, so `except RuntimeError:` catches it as well.

## Attributes

`contract`  
The [`Contract`](../reference/Contract.md#pointblank.Contract) that the data failed.

`validation`  
The interrogated [`Validate`](../reference/Validate.md#pointblank.Validate) object for the failing boundary. Display it to see the validation report.

`result`  
The [`PipelineResult`](../reference/PipelineResult.md#pointblank.PipelineResult) of the run, holding everything the run produced before it stopped (the source validation, and the transformed data and target validation if they were reached). It's `None` when the error comes from [validate_source()](../reference/Pipeline.md#pointblank.Pipeline.validate_source) or [validate_target()](../reference/Pipeline.md#pointblank.Pipeline.validate_target).

## Examples

Let's make a pipeline whose target contract must hold for the output to be published. It uses `on_violation="raise"`, so a failing output stops the run with an error.

``` python
import pointblank as pb
import polars as pl

pipeline = pb.Pipeline(
    target=pb.Contract(
        name="prices",
        direction="target",
        steps=[pb.Step("col_vals_gt", columns="price", value=0)],
        on_violation="raise",
    ),
)

raw_prices = pl.DataFrame({"item": ["a", "b", "c"], "price": [4.5, 0.0, 12.0]})
```

The transformation passes a zero price through, so the run raises. The error holds the result of the run, which still has the transformed data and the target validation:

``` python
try:
    pipeline.run(data=raw_prices, transform=lambda df: df)
except pb.ContractViolationError as e:
    print(e)
    failed_run = e.result

failed_run.target_validation
```

    Contract 'prices' (target) violated: validation has failing steps.

[TABLE]

The report shows the one failing test unit in the [col_vals_gt](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) step, the row for item `"b"`. Nothing after the [run()](../reference/Pipeline.md#pointblank.Pipeline.run) call was reached, so the prices weren't published.

## See Also

[Pipeline](../reference/Pipeline.md#pointblank.Pipeline)  
Enforce contracts at both ends of a transformation.

[Contract](../reference/Contract.md#pointblank.Contract)  
Set `on_violation=` to choose what happens when data fails.
