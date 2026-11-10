# schema_from_tbl()

Create a Schema from an existing table with inferred Field constraints.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/schema.py#L1616-L1696)

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
Attempt to match string columns to known generation presets (e.g., email, url, phone_number) based on column name heuristics and value validation. Default is `True`. A column whose name matches a place or organization preset (`state`, `city`, `country`, `company`, etc.) but holds a small set of repeated values is treated as categorical instead, keeping its values in `allowed=`.

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

The inferred schema has a field for each column. The `user_id` and `age` fields take their ranges from the data and are marked unique (every value in those columns is distinct), the `email` column is matched to the `"email"` preset by its name and values, and `status` has few enough distinct values to become a categorical field with an `allowed=` list.

Let's generate synthetic data matching the original's characteristics:

``` python
pb.preview(schema.generate(n=10, seed=23))
```

| PolarsRows10Columns4 |  |  |  |  |
|----|----|----|----|----|
|  | user_id *Int64* | email *String* | age *Int64* | status *String* |
| 1 | 40 | doris.martin@yandex.com | 32 | inactive |
| 2 | 20 | ngonzalez74@yahoo.com | 68 | inactive |
| 3 | 23 | jessica379@protonmail.com | 62 | pending |
| 4 | 35 | george_evans@yahoo.com | 44 | inactive |
| 5 | 34 | p_williams@outlook.com | 24 | active |
| 6 | 28 | andreamitchell@mail.com | 45 | inactive |
| 7 | 50 | maria.valentine@mail.com | 66 | inactive |
| 8 | 9 | vwalker@gmail.com | 51 | inactive |
| 9 | 13 | brenda.lopez@zoho.com | 34 | inactive |
| 10 | 24 | laurendavis@aol.com | 60 | active |

The new rows follow the inferred constraints: the IDs lie between 1 and 50, the ages between 20 and 69, and every `status` is one of the three original values. The emails are new, realistic addresses from the preset rather than copies of the original `user{i}@example.com` values.
