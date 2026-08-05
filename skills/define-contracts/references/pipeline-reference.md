# Pipeline reference

## Pipeline constructor

```python
pb.Pipeline(
    source: Contract | None = None,
    target: Contract | None = None,
    thresholds: Thresholds | None = None,
    actions: Actions | None = None,
    final_actions: FinalActions | None = None,
    label: str | None = None,
    short_circuit: bool = True,
)
```

## Pipeline methods

| Method                   | Returns          | Description                       |
|--------------------------|------------------|-----------------------------------|
| `run(data, transform)`   | `PipelineResult` | Full pipeline: source + transform + target |
| `validate_source(data)`  | `Validate`       | Run source contract only          |
| `validate_target(data)`  | `Validate`       | Run target contract only          |
| `to_dict()`              | `dict`           | Serialize to dictionary           |
| `from_dict(cls, data)`   | `Pipeline`       | Deserialize from dictionary       |
| `from_yaml(cls, path)`   | `Pipeline`       | Load from YAML file               |
| `to_yaml(path=None)`     | `str \| None`    | Save to YAML (or return str)      |

## PipelineResult

| Attribute / Method    | Type              | Description                    |
|-----------------------|-------------------|--------------------------------|
| `source_validation`   | `Validate \| None`| Source validation result       |
| `target_validation`   | `Validate \| None`| Target validation result       |
| `transform_output`    | `Any`             | Output of the transform        |
| `source_passed`       | `bool`            | Source contract passed?        |
| `target_passed`       | `bool`            | Target contract passed?        |
| `passed`              | `bool`            | Both passed?                   |
| `get_report()`        | `str`             | Summary report text            |

## Pipeline execution flow

```
1. Validate source contract against input data
   |
   +-- If source fails and short_circuit=True -> stop, return result
   |
2. Run transform(data) -> transformed_data
   |
3. Validate target contract against transformed_data
   |
4. Return PipelineResult
```

## short_circuit behavior

| `short_circuit` | Source fails        | Source passes          |
|-----------------|---------------------|------------------------|
| `True`          | Skip transform+target| Run transform+target  |
| `False`         | Run transform+target | Run transform+target  |

## YAML format

```yaml
label: Order cleaning pipeline
short_circuit: true

source:
  name: raw-orders
  direction: source
  schema:
    id: Int64
    amount: Float64
  steps:
    - method: col_vals_not_null
      columns: id
  on_violation: raise

target:
  name: clean-orders
  direction: target
  schema:
    id: Int64
    amount: Float64
    is_valid: Boolean
  steps:
    - method: col_vals_gt
      columns: amount
      value: 0
  on_violation: warn
```

## Patterns

### Source-only pipeline

```python
pipeline = pb.Pipeline(source=source_contract)
result = pipeline.run(data=df, transform=lambda d: d)
```

### Target-only pipeline

```python
pipeline = pb.Pipeline(target=target_contract)
result = pipeline.run(data=raw_df, transform=my_transform)
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
