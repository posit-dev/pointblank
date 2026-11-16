# to_schema()

Get a [Schema](../reference/Schema.md#pointblank.Schema) of the variables' names and types.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L706-L738)

``` python
to_schema()
```

The schema has a column for each variable, in order, with the variable's `dtype` (or `"String"` if the dtype isn't known). It can be used with [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) to check that a table has the expected columns and types. Only the names and types are included: constraints such as ranges and allowed values aren't part of a schema, and are checked by the steps that [`to_validate()`](../reference/pointblank.MetadataImport.to_validate.md) adds after its schema check.

## Returns

`Schema`  
A [`Schema`](../reference/Schema.md#pointblank.Schema) with one column per variable.

## Examples

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))

meta.to_schema()
```

    Schema(columns=[('transaction_id', 'String'), ('customer_id', 'String'), ('amount', 'Float64'), ('quantity', 'Int64'), ('category', 'String'), ('sale_date', 'Date'), ('discount_pct', 'Float64'), ('email', 'String')])

The schema lists the eight columns of the transactions table with their types. The dtype names are those of Polars, so a schema check against a table from another backend fails on the type names (pandas, for example, has `object` and `int64` where Polars has `String` and `Int64`). Convert such a table to Polars before checking it.
