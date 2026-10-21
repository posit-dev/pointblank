# Check the Schema of a Table

[TABLE]

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": ["apple", "banana", "cherry", "date"],
        "b": [1, 6, 3, 5],
        "c": [1.1, 2.2, 3.3, 4.4],
    }
)

# Use the Schema class to define the column schema as loosely or rigorously as required
schema = pb.Schema(
    columns=[
        ("a", "String"),          # Column 'a' has dtype 'String'
        ("b", ["Int", "Int64"]),  # Column 'b' has dtype 'Int' or 'Int64'
        ("c", )                   # Column 'c' follows 'b' but we don't specify a dtype here
    ]
)

# Use the `col_schema_match()` validation method to perform the schema check
validation = (
    pb.Validate(data=tbl)
    .col_schema_match(schema=schema)
    .interrogate()
)

validation
```
