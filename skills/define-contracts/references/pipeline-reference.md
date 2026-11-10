# Pipeline reference

## Pipeline constructor

```python
pb.Pipeline(
    source: Contract | None = None,
    target: Contract | None = None,       # at least one of source/target
    thresholds: Thresholds | None = None, # replaces the contracts' thresholds
    actions: Actions | None = None,       # used for both boundaries
    final_actions: FinalActions | None = None,  # run once per boundary
    label: str | None = None,             # validations get "<label> [source]" etc.
    short_circuit: bool = True,
)
```

## Pipeline methods

| Method                   | Returns          | Description                       |
|--------------------------|------------------|-----------------------------------|
| `run(data, transform)`   | `PipelineResult` | Source check + transform + target check |
| `validate_source(data)`  | `Validate`       | Run source contract only          |
| `validate_target(data)`  | `Validate`       | Run target contract only          |
| `to_dict()`              | `dict`           | Serialize to dictionary           |
| `from_dict(cls, data)`   | `Pipeline`       | Deserialize from dictionary       |
| `from_yaml(cls, path)`   | `Pipeline`       | Load from YAML file               |
| `to_yaml(path=None)`     | `str`            | YAML string (also written to `path`) |

All three running methods apply the failing contract's `on_violation`.

## PipelineResult

| Attribute / Method    | Type              | Description                    |
|-----------------------|-------------------|--------------------------------|
| `source_validation`   | `Validate \| None`| Source validation result       |
| `target_validation`   | `Validate \| None`| Target validation (None if skipped) |
| `transform_output`    | `Any`             | Output of the transform (None if skipped) |
| `source_passed`       | `bool`            | Source contract passed?        |
| `target_passed`       | `bool`            | Target contract passed?        |
| `passed`              | `bool`            | All validated boundaries passed? |
| `get_report()`        | `str`             | Summary report text            |

A boundary passes only if every test unit passes (thresholds don't
matter). A boundary that wasn't validated counts as passing, so a
target skipped after a failing source has `target_passed=True` while
`passed` is `False`.

## Pipeline execution flow

```
1. Validate source contract against input data
   |
   +-- If source fails and short_circuit=True -> apply on_violation, stop
   |
2. Run transform(data) -> transformed_data
   |
3. Validate target contract against transformed_data
   |
4. Apply on_violation for any failing boundary; return PipelineResult
```

## short_circuit behavior

| `short_circuit` | Source fails                     | Source passes          |
|-----------------|----------------------------------|------------------------|
| `True`          | Skip transform + target          | Run transform + target |
| `False`         | Run transform + target; a `"raise"` source raises after the target | Run transform + target |

## YAML format

```yaml
pipeline:
  label: Order cleaning pipeline
  short_circuit: true
source:
  name: raw-orders
  on_violation: raise
  schema:
    id: Int64
    amount: Float64
  steps:
  - col_vals_not_null:
      columns: id
target:
  name: clean-orders
  direction: target
  schema:
    id: Int64
    amount: Float64
    is_valid: Boolean
  steps:
  - col_vals_gt:
      columns: amount
      value: 0
```

## Patterns

### Source-only pipeline (ingestion guard)

```python
pipeline = pb.Pipeline(source=source_contract)
validation = pipeline.validate_source(df)
```

### Target-only pipeline (output guard)

```python
pipeline = pb.Pipeline(target=target_contract)
result = pipeline.run(data=raw_df, transform=my_transform)
```

### Handling a raised violation

```python
try:
    result = pipeline.run(data=df, transform=transform)
except pb.ContractViolationError as e:
    print(e.contract.name)
    e.validation.get_tabular_report()
    result = e.result           # the PipelineResult of the run
```

### Accessing validation reports

```python
result = pipeline.run(data=df, transform=transform)

if not result.passed:
    if not result.source_passed:
        result.source_validation.get_tabular_report()
    if not result.target_passed:
        result.target_validation.get_tabular_report()
```
