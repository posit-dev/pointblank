# schema_from_tbl()

Create a Schema from an existing table with inferred Field constraints.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/schema.py#L1568-L1635)

``` python
schema_from_tbl(
    tbl,
    *,
    infer_constraints=True,
    categorical_threshold=20,
    detect_presets=True,
    sample_size=None,
)
```

This is the functional form of `Schema.from_table()`. It inspects the actual values in the table to infer rich constraints (min/max, uniqueness, null rates, allowed values, presets) suitable for synthetic data generation via `schema.generate()` or [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset).

## Parameters

`tbl: Any`  
A Polars DataFrame, Pandas DataFrame, or Ibis table (DuckDB, SQLite, etc.).

`infer_constraints: bool = True`  
When `True` (default), inspect values to infer min/max, uniqueness, null rates, etc. When `False`, behave like `Schema(tbl=df)` (dtype only).

`categorical_threshold: int | float = 20`  
If a column has \<= this many unique values (int) or this fraction of total rows (float between `0` and `1`), treat it as categorical and populate `allowed=`. Default is `20`.

`detect_presets: bool = True`  
Attempt to match string columns to known generation presets (e.g., email, url, phone_number) based on column name heuristics and value validation. Default is `True`.

`sample_size: int | None = None`  
If set, sample this many rows before analysis (useful for very large tables). `None` means use all rows.

## Returns

`Schema`  
A Schema populated with Field objects containing inferred constraints, ready for use with `schema.generate()` or [generate_dataset()](../reference/generate_dataset.md#pointblank.generate_dataset).

## Examples

``` python
import pointblank as pb
import polars as pl

df = pl.DataFrame({
    "user_id": list(range(1, 51)),
    "email": [f"user{i}@example.com" for i in range(50)],
    "age": [20 + i % 50 for i in range(50)],
    "status": ["active", "pending", "inactive"] * 16 + ["active", "pending"],
})

schema = pb.schema_from_tbl(df)
print(schema)
```

    Pointblank Schema
      user_id: IntField(dtype='Int64', nullable=False, null_probability=0.0, unique=True, min_val=1, max_val=50, allowed=None)
      email: StringField(dtype='String', nullable=False, null_probability=0.0, unique=True, min_length=None, max_length=None, pattern=None, preset='email', allowed=None)
      age: IntField(dtype='Int64', nullable=False, null_probability=0.0, unique=True, min_val=20, max_val=69, allowed=None)
      status: StringField(dtype='String', nullable=False, null_probability=0.0, unique=False, min_length=None, max_length=None, pattern=None, preset=None, allowed=['active', 'inactive', 'pending'])

Generate synthetic data matching the original's characteristics:

``` python
pb.preview(schema.generate(n=10, seed=23))
```

| PolarsRows10Columns4 |  |  |  |  |
|----|----|----|----|----|
|  | user_id *Int64* | email *String* | age *Int64* | status *String* |
| 1 | 50 | doris.martin@yandex.com | 69 | inactive |
| 2 | 19 | ngonzalez74@yahoo.com | 38 | active |
| 3 | 6 | jessica379@protonmail.com | 25 | active |
| 4 | 2 | george_evans@yahoo.com | 21 | pending |
| 5 | 38 | p_williams@outlook.com | 57 | inactive |
| 6 | 20 | andreamitchell@mail.com | 39 | inactive |
| 7 | 28 | maria.valentine@mail.com | 47 | inactive |
| 8 | 25 | vwalker@gmail.com | 44 | pending |
| 9 | 34 | brenda.lopez@zoho.com | 53 | inactive |
| 10 | 23 | laurendavis@aol.com | 42 | active |
