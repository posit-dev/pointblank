# Schema inference reference

There are two ways to get a schema from a table:

| Call                     | Holds                              | Use it for                        |
|--------------------------|------------------------------------|-----------------------------------|
| `pb.Schema(tbl=df)`      | Column names and full dtype strings | `col_schema_match()`, contracts  |
| `pb.schema_from_tbl(df)` | Field objects with inferred constraints | `schema.generate()`, `generate_dataset()` |

## Schema(tbl=...)

```python
schema = pb.Schema(tbl=df)
schema.get_column_list()   # ['date_time', 'date', 'a', ...]
schema.get_dtype_list()    # ["Datetime(time_unit='us', time_zone=None)", 'Date', 'Int64', ...]
```

The dtypes are the backend's own names (Polars `Int64`, Pandas
`int64`, Ibis `int64`), so the schema matches the table it came from.

## schema_from_tbl

```python
pb.schema_from_tbl(
    tbl: Any,
    *,
    infer_constraints: bool = True,
    categorical_threshold: int | float = 20,
    detect_presets: bool = True,
    sample_size: int | None = None,
) -> Schema
```

Creates a Schema of Field objects (`IntField`, `StringField`, ...)
whose constraints come from the values, ready for generating
synthetic data that looks like the table.

### Parameters

| Parameter                | Default | Description                         |
|--------------------------|---------|-------------------------------------|
| `infer_constraints`      | `True`  | Infer min/max, null rates, uniqueness, allowed sets; `False` behaves like `Schema(tbl=df)` |
| `categorical_threshold`  | `20`    | Max distinct values (int) or fraction of rows (float in 0-1) for a categorical `allowed=` list |
| `detect_presets`         | `True`  | Match string columns to presets (email, URL, phone) by name and values |
| `sample_size`            | `None`  | Rows to sample first (None = all)   |

```python
gen_schema = pb.schema_from_tbl(df)
synthetic = gen_schema.generate(n=1000, seed=23)
```

## Schema.from_table

Class method with the same parameters as `schema_from_tbl()`:

```python
schema = pb.Schema.from_table(df, sample_size=1000)
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
| `get_dtype_list()`              | `list`         | Dtype strings (Field objects for an inferred schema) |
| `get_tbl_type()`                | `str`          | Table type the schema came from |
| `get_schema_coerced(to=None)`   | `Schema`       | Coerced to a target backend  |
| `generate(n=100, seed=None, output="polars", country="US", shuffle=True, weighted=True)` | DataFrame or dict | Generate synthetic data |

## Common dtype strings by backend

| Concept   | Polars       | Pandas       |
|-----------|-------------|--------------|
| Integer   | `Int8/16/32/64`, `UInt8/16/32/64` | `int8/16/32/64`, `uint8/16/32/64` |
| Float     | `Float32/64` | `float32/64` |
| String    | `String`     | `object`, `string` |
| Boolean   | `Boolean`    | `bool`       |
| Date      | `Date`       | `datetime64[ns]` |
| Datetime  | `Datetime(time_unit='us', time_zone=None)` | `datetime64[ns]` |
| Duration  | `Duration`   | `timedelta64[ns]` |

## Using a table's schema in validation

```python
schema = pb.Schema(tbl=df)

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
| `full_match_dtypes`        | `True`  | Exact dtype match required (`False` allows a substring, so `"Int"` matches `"Int64"`) |
