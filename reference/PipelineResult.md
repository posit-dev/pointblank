# PipelineResult

The outcome of running a data pipeline.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L17-L261)

``` python
PipelineResult(
    source_validation=None,
    target_validation=None,
    transform_output=None,
    _source_passed=None,
    _target_passed=None,
)
```

A [PipelineResult](../reference/PipelineResult.md#pointblank.PipelineResult) is what [`Pipeline.run()`](../reference/Pipeline.md#pointblank.Pipeline.run) returns. It collects everything a run produced: the interrogated validation for each boundary, the transformed data, and pass/fail properties for a quick go/no-go decision. Keeping all of it in one object means the code that runs a pipeline can decide what to do next (publish the output, stop the job, or alert someone) and still have the full reports at hand to explain the decision.

You don't normally create a [PipelineResult](../reference/PipelineResult.md#pointblank.PipelineResult) yourself. Get one from [run()](../reference/Pipeline.md#pointblank.Pipeline.run) and read its attributes and properties.

## Attributes

`source_validation: Validate | None`  
The interrogated [`Validate`](../reference/Validate.md#pointblank.Validate) object for the source boundary, or `None` if the pipeline has no source contract. Display it to see the validation report, or use any of its methods (e.g., [`get_step_report()`](../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report) or [`get_data_extracts()`](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts)) to dig into failures.

`target_validation: Validate | None`  
The interrogated [Validate](../reference/Validate.md#pointblank.Validate) object for the target boundary. It's `None` if the pipeline has no target contract, or if the run stopped after a failing source validation (with `short_circuit=True`).

`transform_output: Any`  
The table returned by the transformation function, or `None` if the run stopped before the transformation was called. This is the data to publish when the result passes.

## Passing And Failing

The [`passed`](../reference/PipelineResult.md#pointblank.PipelineResult.passed), [`source_passed`](../reference/PipelineResult.md#pointblank.PipelineResult.source_passed), and [`target_passed`](../reference/PipelineResult.md#pointblank.PipelineResult.target_passed) properties give a boolean verdict. A boundary passes when every test unit in every step passes (the same test as [`Validate.all_passed()`](../reference/Validate.all_passed.md#pointblank.Validate.all_passed)), so thresholds don't affect it. A boundary that wasn't validated counts as passing. That covers a pipeline without that contract, and also a target that was skipped because the source failed: in that case [target_passed](../reference/PipelineResult.md#pointblank.PipelineResult.target_passed) is `True`, but [passed](../reference/PipelineResult.md#pointblank.PipelineResult.passed) is still `False` because of the source. For a verdict based on threshold levels instead, use the validations directly, for example with [`above_threshold()`](../reference/Validate.above_threshold.md#pointblank.Validate.above_threshold).

## Examples

Let's run a small pipeline that turns raw temperature readings in Fahrenheit into Celsius. The source contract requires a station for every reading, and the target contract requires temperatures in a plausible range.

``` python
import pointblank as pb
import polars as pl

pipeline = pb.Pipeline(
    source=pb.Contract(
        name="raw_readings", steps=[pb.Step("col_vals_not_null", columns="station")]
    ),
    target=pb.Contract(
        name="readings",
        direction="target",
        steps=[pb.Step("col_vals_between", columns="temp_c", left=-60, right=60)],
    ),
)

raw_readings = pl.DataFrame({"station": ["N1", "N2", "S1"], "temp_f": [68.0, 41.0, 95.0]})

result = pipeline.run(
    data=raw_readings,
    transform=lambda df: df.select("station", temp_c=(pl.col("temp_f") - 32) * 5 / 9),
)

result
```

    PipelineResult(
      source=validated,
      target=validated,
      passed=True
    )

Printing the result shows which boundaries were validated and whether the run passed. In a scheduled job, [passed](../reference/PipelineResult.md#pointblank.PipelineResult.passed) is the natural thing to branch on:

``` python
if result.passed:
    print(result.transform_output)
else:
    print("Not publishing: the pipeline failed.")
```

    shape: (3, 2)
    ┌─────────┬────────┐
    │ station ┆ temp_c │
    │ ---     ┆ ---    │
    │ str     ┆ f64    │
    ╞═════════╪════════╡
    │ N1      ┆ 20.0   │
    │ N2      ┆ 5.0    │
    │ S1      ┆ 35.0   │
    └─────────┴────────┘

For a plain-text summary of both boundaries, suitable for a log or a notification, use [`get_report()`](../reference/PipelineResult.md#pointblank.PipelineResult.get_report):

``` python
print(result.get_report())
```

    Pipeline Boundary Validation Results
    ========================================

    Source Boundary: PASSED
    ------------------------------
      Steps: 1
      Test units passed: 3
      Test units failed: 0

    Target Boundary: PASSED
    ------------------------------
      Steps: 1
      Test units passed: 3
      Test units failed: 0

    Overall: PASSED

The full validation report for either boundary is in `source_validation` or `target_validation`:

``` python
result.target_validation
```

[TABLE]

All three converted readings pass the target's single step, which is why [passed](../reference/PipelineResult.md#pointblank.PipelineResult.passed) is `True`. When a run fails, these reports show which steps failed and on which rows, and [`get_data_extracts()`](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) pulls out the failing rows themselves.

## See Also

[Pipeline](../reference/Pipeline.md#pointblank.Pipeline)  
Define the boundaries and run the transformation.

[all_passed()](../reference/Validate.all_passed.md#pointblank.Validate.all_passed)  
The test used to decide whether a boundary passes.

## Attributes

| Name | Description |
|----|----|
| [passed](#passed) | Whether the whole pipeline run passed. |
| [source_passed](#source_passed) | Whether the source boundary validation passed. |
| [target_passed](#target_passed) | Whether the target boundary validation passed. |

### passed

Whether the whole pipeline run passed.

`passed: bool`

This is `True` only when both [`source_passed`](../reference/PipelineResult.md#pointblank.PipelineResult.source_passed) and [`target_passed`](../reference/PipelineResult.md#pointblank.PipelineResult.target_passed) are `True`, meaning that every test unit at each validated boundary passed. A run that stopped early because the source failed is `False`. Thresholds don't affect it, so a single failing test unit is enough to make it `False`.

### source_passed

Whether the source boundary validation passed.

`source_passed: bool`

This is `True` when every test unit in the source validation passed, or when the pipeline has no source contract. Thresholds don't affect it.

### target_passed

Whether the target boundary validation passed.

`target_passed: bool`

This is `True` when every test unit in the target validation passed. It's also `True` when the target wasn't validated, either because the pipeline has no target contract or because the run stopped after a failing source validation. Check [`passed`](../reference/PipelineResult.md#pointblank.PipelineResult.passed) for the overall outcome.

## Methods

| Name | Description |
|----|----|
| [get_report()](#get_report) | Summarize both boundary validations as plain text. |

### get_report()

Summarize both boundary validations as plain text.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L187-L253)

``` python
get_report()
```

This method returns a short text summary of the run: for each boundary that was validated, whether it passed, the number of validation steps, and the total numbers of passing and failing test units, followed by the overall outcome. A boundary that wasn't validated is left out. Text works well where an HTML report doesn't, such as a log file, a terminal, a chat notification, or the description of a failed job. For the full report of a boundary, display the result's `source_validation` or `target_validation` attribute instead.

#### Returns

`str`  
A multi-line summary. Use `print()` to display it.

#### Examples

Let's run a pipeline whose target contract catches a mistake in the transformation: it should keep only active accounts, but the filter was left out.

``` python
import pointblank as pb
import polars as pl

pipeline = pb.Pipeline(
    source=pb.Contract(
        name="raw_accounts", steps=[pb.Step("col_vals_not_null", columns="account_id")]
    ),
    target=pb.Contract(
        name="active_accounts",
        direction="target",
        steps=[pb.Step("col_vals_eq", columns="status", value="active")],
        on_violation="log",
    ),
)

raw_accounts = pl.DataFrame(
    {"account_id": [1, 2, 3, 4], "status": ["active", "closed", "active", "closed"]}
)

result = pipeline.run(data=raw_accounts, transform=lambda df: df)

print(result.get_report())
```

    Contract 'active_accounts' (target) violated: validation has failing steps.

    Pipeline Boundary Validation Results
    ========================================

    Source Boundary: PASSED
    ------------------------------
      Steps: 1
      Test units passed: 4
      Test units failed: 0

    Target Boundary: FAILED
    ------------------------------
      Steps: 1
      Test units passed: 2
      Test units failed: 2

    Overall: FAILED

The source boundary passed with all four test units. The target boundary failed with two failing test units, one for each closed account, so the run failed overall.
