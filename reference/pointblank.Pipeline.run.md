# run()

Validate the input, transform it, and validate the output.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L853-L1050)

``` python
run(
    data,
    transform,
)
```

This is the main way to use a pipeline. It checks `data` against the source contract, calls `transform` on it, checks the output against the target contract, and returns everything in a [`PipelineResult`](../reference/PipelineResult.md#pointblank.PipelineResult): the two interrogated validations, the transformed data, and pass/fail properties for each boundary and overall. A failure is attributed to a boundary, so you can tell whether the input was bad or the transformation went wrong.

If the source validation has any failing test units and the pipeline has `short_circuit=True` (the default), the run stops there. The transformation isn't called, and the result has no transformed data and no target validation. With `short_circuit=False`, the transformation is called and the target is validated whatever the outcome of the source validation. If the pipeline has only one contract, the other boundary is skipped.

When a boundary fails, its contract's `on_violation=` setting is applied: `"warn"` issues a `UserWarning` and `"log"` writes to the `"pointblank.contract"` logger. With `"raise"`, `run()` raises a [`ContractViolationError`](../reference/ContractViolationError.md#pointblank.ContractViolationError) instead of returning. A failing source raises right away with `short_circuit=True`, and after the target has been validated with `short_circuit=False`. The error's `result` attribute holds the [`PipelineResult`](../reference/PipelineResult.md#pointblank.PipelineResult) of the run, with the validations and any transformed data. With `"warn"` or `"log"`, check the result's [`passed`](../reference/PipelineResult.md#pointblank.PipelineResult.passed) property to stop a job when the pipeline fails.

## Parameters

`data: IntoDataFrame`  
The input table. Any table type that [Validate](../reference/Validate.md#pointblank.Validate) accepts can be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

`transform: Callable[[Any], Any]`  
A function that takes the input table and returns the transformed table. It's called with `data` itself, and its return value is validated against the target contract, so it must also be a table that [Validate](../reference/Validate.md#pointblank.Validate) accepts. The function is called at most once per run. An exception raised by it isn't caught, so it propagates from `run()`.

## Returns

`PipelineResult`  
The results of the run. `source_validation` and `target_validation` hold the interrogated [Validate](../reference/Validate.md#pointblank.Validate) objects (or `None` for a boundary that wasn't validated), and `transform_output` holds the transformed table (or `None` if the run stopped early).

## Raises

`ContractViolationError`  
If a boundary fails and its contract has `on_violation="raise"`. If both boundaries fail that way (possible with `short_circuit=False`), the error is for the source contract. Either way, the error's `result` attribute holds the [PipelineResult](../reference/PipelineResult.md#pointblank.PipelineResult).

## Examples

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

The input was fine and the output wasn't, which points to the transformation. In the target validation, the [rows_distinct](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step flags both rows for customer 3. The default `on_violation="warn"` setting also issued a warning.

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

    UserWarning: Contract 'raw_customers' (source) violated: validation has failing steps.

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

The summary from [`get_report()`](../reference/PipelineResult.md#pointblank.PipelineResult.get_report) shows that the source boundary failed with one failing test unit, while the target boundary passed. Overall, the run failed.
