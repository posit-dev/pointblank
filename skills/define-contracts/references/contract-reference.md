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

| Method                 | Returns        | Description                              |
|------------------------|----------------|------------------------------------------|
| `validate(data)`       | `Validate`     | Compile and interrogate (never raises)   |
| `to_validate(data)`    | `Validate`     | Compile without interrogating            |
| `to_dict()`            | `dict`         | Serialize to dictionary                  |
| `from_dict(cls, data)` | `Contract`     | Deserialize from dictionary              |
| `from_yaml(cls, path)` | `Contract`     | Load from YAML file                      |
| `to_yaml(path=None)`   | `str`          | YAML string (also written to `path`)     |

The compiled validation has `tbl_name=` set to the contract name, the
label `"Contract: <name> v<version>"`, the contract's thresholds,
owner, consumers, and version, a `col_schema_match()` step first (if
there's a schema, with its strict defaults), and then the steps.

## Step constructor

```python
pb.Step(method: str, **kwargs)
```

`method` is the name of a Validate method (e.g. `"col_vals_gt"`,
`"rows_distinct"`), and the keyword arguments are passed to it. The
step's parts are its `method` and `kwargs` attributes. `conjointly`
takes its expressions as a list in `exprs=`:

```python
pb.Step("conjointly", exprs=[lambda df: pl.col("a") > 0, lambda df: pl.col("a") < 10])
```

## Step methods

| Method                 | Returns    | Description                |
|------------------------|------------|----------------------------|
| `to_dict()`            | `dict`     | Serialize to dictionary    |
| `from_dict(cls, data)` | `Step`     | Deserialize from dictionary|

## Valid step methods

A `Contract` accepts only these (others raise a `ValueError` when the
contract is created):

- `col_vals_gt`, `col_vals_lt`, `col_vals_ge`, `col_vals_le`,
  `col_vals_eq`, `col_vals_ne`
- `col_vals_between`, `col_vals_outside`
- `col_vals_in_set`, `col_vals_not_in_set`
- `col_vals_null`, `col_vals_not_null`
- `col_vals_regex`, `col_vals_within_spec`, `col_vals_expr`
- `col_vals_increasing`, `col_vals_decreasing`
- `col_exists`, `col_pct_null`, `col_schema_match`
- `col_count_match`, `row_count_match`
- `rows_distinct`, `rows_complete`
- `conjointly`

For anything else, chain it onto `contract.to_validate(data)` or use
`add_steps()` with a `pb.Steps` bundle (which can hold any method).

## YAML format

`to_yaml()` writes the contract under a `contract:` key. Unset fields
and the default `direction`/`on_violation` are left out; each step is a
mapping of the method name to its arguments:

```yaml
contract:
  name: orders-source
  version: '1.0'
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
  - col_vals_not_null:
      columns: order_id
  - col_vals_gt:
      columns: amount
      value: 0
  - col_vals_in_set:
      columns: status
      set: [pending, shipped, delivered]
  - rows_distinct:
      columns_subset: [order_id]
  thresholds:
    warning: 0.01
    error: 0.05
```

Step arguments must be plain data (strings, numbers, booleans, dates,
lists, dicts); step-level `Thresholds` and tuples are converted
automatically. A lambda, expression, or column selector makes
`to_yaml()` raise a `ValueError` naming where it is.

## Import and export

```python
pb.export_contract(contract_or_validation, destination=None, *, format=...)  # -> dict or str
pb.import_contract(source, *, format=None)                                   # -> ContractImport
```

Formats: `"json_schema"`, `"frictionless"`, `"dbt"`, `"odcs"`.
`ContractImport` has `summary()`, `coverage`, `to_contract(name=...)`,
`to_validate(...)`, `to_python()`, and `to_yaml()`.

## on_violation behavior

Applied only by `Pipeline.run()`, `validate_source()`, and
`validate_target()` when a boundary has any failing test unit:

| Value     | On failure...                                              |
|-----------|------------------------------------------------------------|
| `"warn"`  | Issues a `UserWarning`, continues                          |
| `"raise"` | Raises `ContractViolationError` (a `RuntimeError`)         |
| `"log"`   | Writes a warning to the `"pointblank.contract"` logger     |

`ContractViolationError` has `contract`, `validation`, and `result`
attributes (`result` is the `PipelineResult` when raised by `run()`,
otherwise `None`).
