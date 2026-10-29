# Validate.set_tbl()

Set or replace the table associated with the Validate object.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L6162-L6300)

``` python
Validate.set_tbl(
    tbl,
    tbl_name=None,
    label=None,
)
```

This method allows you to replace the table associated with a Validate object with a different (but presumably similar) table. This is useful when you want to apply the same validation plan to multiple tables or when you have a validation workflow defined but want to swap in a different data source.

## Parameters

`tbl: Any`  
The table to replace the existing table with. This can be any supported table type including DataFrame objects, Ibis table objects, CSV file paths, Parquet file paths, GitHub URLs, or database connection strings. The same table type constraints apply as in the [Validate](../reference/Validate.md#pointblank.Validate) constructor.

`tbl_name: str | None = None`  
An optional name to assign to the new input table object. If no value is provided, the existing table name will be retained.

`label: str | None = None`  
An optional label for the validation plan. If no value is provided, the existing label will be retained.

## Returns

`Validate`  
A new [Validate](../reference/Validate.md#pointblank.Validate) object with the replacement table.

## When To Use

The [set_tbl()](../reference/Validate.set_tbl.md#pointblank.Validate.set_tbl) method is particularly useful in scenarios where you have:

- multiple similar tables that need the same validation checks
- a template validation workflow that should be applied to different data sources
- YAML-defined validations where you want to override the table specified in the YAML

The [set_tbl()](../reference/Validate.set_tbl.md#pointblank.Validate.set_tbl) method creates a copy of the validation object with the new table, so the original validation object remains unchanged. This allows you to reuse validation plans across multiple tables without interference.

## Examples

We will first create two similar tables for our future validation plans.

``` python
import pointblank as pb
import polars as pl

# Create two similar tables
table_1 = pl.DataFrame({
    "x": [1, 2, 3, 4, 5],
    "y": [5, 4, 3, 2, 1],
    "z": ["a", "b", "c", "d", "e"]
})

table_2 = pl.DataFrame({
    "x": [2, 4, 6, 8, 10],
    "y": [10, 8, 6, 4, 2],
    "z": ["f", "g", "h", "i", "j"]
})
```

Create a validation plan with the first table.

``` python
validation_table_1 = (
    pb.Validate(
        data=table_1,
        tbl_name="Table 1",
        label="Validation applied to the first table"
    )
    .col_vals_gt(columns="x", value=0)
    .col_vals_lt(columns="y", value=10)
)
```

Now apply the same validation plan to the second table.

``` python
validation_table_2 = (
    validation_table_1
    .set_tbl(
        tbl=table_2,
        tbl_name="Table 2",
        label="Validation applied to the second table"
    )
)
```

Here is the interrogation of the first table:

``` python
validation_table_1.interrogate()
```

[TABLE]

And the second table:

``` python
validation_table_2.interrogate()
```

[TABLE]

Every value of `y` in the first table is below `10`, but the second table has a `10`, so the second step has one failing test unit there. The plan was written once and applied to both tables, each with its own name and label. This is handy for tables that share a structure, such as monthly extracts or one table per region.
