# validate_source()

Check incoming data against the source contract.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/pipeline.py#L644-L744)

``` python
validate_source(data)
```

This method validates one table against the pipeline's source contract and returns the interrogated [`Validate`](../reference/Validate.md#pointblank.Validate) object. Use it when you want an ingestion check without running a transformation through the pipeline: for example, when data lands in one task of an orchestrated workflow and is transformed in another, or when the transformation is written in SQL or another tool. Pair it with [`validate_target()`](../reference/pointblank.Pipeline.validate_target.md) to check the other boundary.

The validation is built the same way as in [`run()`](../reference/pointblank.Pipeline.run.md), with the pipeline's label, thresholds, and actions. If the data fails, the source contract's `on_violation=` setting is applied. With `on_violation="raise"`, this makes the method a simple gate: the code after the call only runs when the data meets the contract.

## Parameters

`data: IntoDataFrame`  
The incoming table. Any table type that [Validate](../reference/Validate.md#pointblank.Validate) accepts can be used (e.g., a Polars or Pandas DataFrame, or an Ibis table).

## Returns

`Validate`  
An interrogated [Validate](../reference/Validate.md#pointblank.Validate) object with the results of every source contract check.

## Raises

`ValueError`  
If the pipeline has no source contract.

`ContractViolationError`  
If the data fails the source contract and the contract has `on_violation="raise"`. It's a subclass of `RuntimeError`, and its `validation` attribute holds the failing validation.

## Examples

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

A batch with an impossible temperature raises a [ContractViolationError](../reference/ContractViolationError.md#pointblank.ContractViolationError) (a kind of `RuntimeError`) naming the contract, so a job that loads the data can stop before anything is written.

``` python
bad_readings = pl.DataFrame({"sensor_id": ["s1", "s2"], "temp_c": [21.5, 85.0]})

try:
    ingest.validate_source(bad_readings)
except RuntimeError as e:
    print(f"Rejected: {e}")
```

    Rejected: Contract 'sensor_readings' (source) violated: validation has failing steps.

The message names the contract and its direction. If the contract used `on_violation="warn"` or `"log"` instead, the call would return the failing validation rather than raising, leaving it to the caller to decide what to do.
