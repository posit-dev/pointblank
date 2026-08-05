# Contract reference

## Contract constructor

```python
pb.Contract(
    name: str,                                    # required
    direction: Literal["source", "target"] = "source",
    schema: Schema | None = None,
    steps: list[Step] = [],
    version: str | None = None,
    owner: str | None = None,
    consumers: str | list[str] | None = None,
    description: str | None = None,
    thresholds: Thresholds | None = None,
    on_violation: Literal["warn", "raise", "log"] = "warn",
)
```

## Contract methods

| Method                 | Returns        | Description                    |
|------------------------|----------------|--------------------------------|
| `validate(data)`       | `Validate`     | Interrogate data immediately   |
| `to_validate(data)`    | `Validate`     | Build Validate without running |
| `to_dict()`            | `dict`         | Serialize to dictionary        |
| `from_dict(cls, data)` | `Contract`     | Deserialize from dictionary    |
| `from_yaml(cls, path)` | `Contract`     | Load from YAML file            |
| `to_yaml(path=None)`   | `str \| None`  | Save to YAML (or return str)   |

## Step constructor

```python
pb.Step(method: str, **kwargs)
```

`method` is the name of any Validate method (e.g.,
`"col_vals_gt"`, `"rows_distinct"`). All other keyword arguments
are forwarded to that method.

## Step methods

| Method                 | Returns    | Description                |
|------------------------|------------|----------------------------|
| `to_dict()`            | `dict`     | Serialize to dictionary    |
| `from_dict(cls, data)` | `Step`     | Deserialize from dictionary|

## Valid step methods

- `col_vals_gt`, `col_vals_lt`, `col_vals_ge`, `col_vals_le`,
  `col_vals_eq`, `col_vals_ne`
- `col_vals_between`, `col_vals_outside`
- `col_vals_in_set`, `col_vals_not_in_set`
- `col_vals_null`, `col_vals_not_null`
- `col_vals_regex`, `col_vals_within_spec`, `col_vals_expr`
- `col_vals_increasing`, `col_vals_decreasing`
- `col_exists`, `col_schema_match`
- `col_count_match`, `row_count_match`
- `col_pct_null`, `col_pct_missing`
- `rows_distinct`, `rows_complete`
- `data_freshness`, `tbl_match`
- `conjointly`, `specially`

## YAML format

```yaml
name: orders-source
direction: source
version: "1.0"
owner: data-team
consumers:
  - analytics
  - ml-pipeline
description: Order data source contract
on_violation: raise

schema:
  order_id: Int64
  amount: Float64
  status: String

steps:
  - method: col_vals_not_null
    columns: order_id
  - method: col_vals_gt
    columns: amount
    value: 0
  - method: col_vals_in_set
    columns: status
    set: [pending, shipped, delivered]
  - method: rows_distinct
    columns_subset: [order_id]

thresholds:
  warning: 0.01
  error: 0.05
```

## on_violation behavior

| Value     | On failure...                                 |
|-----------|-----------------------------------------------|
| `"warn"`  | Prints warning to stderr, continues           |
| `"raise"` | Raises `ContractViolationError`               |
| `"log"`   | Logs via Python `logging` module              |
