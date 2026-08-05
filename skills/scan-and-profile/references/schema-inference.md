# Schema inference reference

## schema_from_tbl

```python
pb.schema_from_tbl(
    tbl: Any,
    *,
    infer_constraints: bool = True,
    categorical_threshold: int = 20,
    detect_presets: bool = True,
    sample_size: int | None = None,
) -> Schema
```

Creates a Schema from an existing table with optional constraint
inference.

### Parameters

| Parameter                | Default | Description                         |
|--------------------------|---------|-------------------------------------|
| `infer_constraints`      | `True`  | Infer value ranges, allowed sets    |
| `categorical_threshold`  | `20`    | Max distinct values for categorical |
| `detect_presets`         | `True`  | Detect email, URL, phone patterns   |
| `sample_size`            | `None`  | Rows to sample (None = all)         |

## Schema.from_table

Class method with the same parameters:

```python
schema = pb.Schema.from_table(
    df,
    infer_constraints=True,
    categorical_threshold=20,
    detect_presets=True,
    sample_size=1000,
)
```

## Schema constructor

```python
# Keyword arguments
pb.Schema(id="Int64", name="String", amount="Float64")

# Dictionary
pb.Schema({"id": "Int64", "name": "String"})

# List of tuples
pb.Schema([("id", "Int64"), ("name", "String")])

# Column names only
pb.Schema(["id", "name", "amount"])

# From existing table
pb.Schema(tbl=df)
```

## Schema methods

| Method                          | Returns        | Description                  |
|---------------------------------|----------------|------------------------------|
| `get_column_list()`             | `list[str]`    | Column names                 |
| `get_dtype_list()`              | `list[str]`    | Data type strings            |
| `get_schema_coerced(to=None)`   | `Schema`       | Coerced to a target backend  |
| `generate(n=100, ...)`          | `DataFrame`    | Generate synthetic data      |

## Common dtype strings by backend

| Concept   | Polars       | Pandas       |
|-----------|-------------|--------------|
| Integer   | `Int8/16/32/64`, `UInt8/16/32/64` | `int8/16/32/64`, `uint8/16/32/64` |
| Float     | `Float32/64` | `float32/64` |
| String    | `String`     | `object`, `string` |
| Boolean   | `Boolean`    | `bool`       |
| Date      | `Date`       | `datetime64[ns]` |
| Datetime  | `Datetime`   | `datetime64[ns]` |
| Duration  | `Duration`   | `timedelta64[ns]` |

## Using inferred schema in validation

```python
schema = pb.schema_from_tbl(df)

validation = (
    pb.Validate(data=new_df)
    .col_schema_match(
        schema=schema,
        complete=True,       # all columns must be present
        in_order=True,       # column order must match
        full_match_dtypes=True,
    )
    .interrogate()
)
```

### col_schema_match parameters

| Parameter                  | Default | Description                         |
|----------------------------|---------|-------------------------------------|
| `schema`                   | required | Schema object to match against     |
| `complete`                 | `True`  | All schema columns must exist       |
| `in_order`                 | `True`  | Column order must match             |
| `case_sensitive_colnames`  | `True`  | Column name comparison              |
| `case_sensitive_dtypes`    | `True`  | Dtype string comparison             |
| `full_match_dtypes`        | `True`  | Exact dtype match required          |
