# Schema

Definition of a schema object.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/schema.py#L20-L731)

``` python
Schema(
    columns=None,
    tbl=None,
    **kwargs,
)
```

The schema object defines the structure of a table. Once it is defined, the object can be used in a validation workflow, using [Validate](../reference/Validate.md#pointblank.Validate) and its methods, to ensure that the structure of a table matches the expected schema. The validation method that works with the schema object is called [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match).

A schema for a table can be constructed with the [Schema](../reference/Schema.md#pointblank.Schema) class in a number of ways:

1.  providing a list of column names to `columns=` (to check only the column names)
2.  using a list of one- or two-element tuples in `columns=` (to check both column names and optionally dtypes, should be in the form of `[(column_name, dtype), ...]`)
3.  providing a dictionary to `columns=`, where the keys are column names and the values are dtypes
4.  providing individual column arguments in the form of keyword arguments (constructed as `column_name=dtype`)

The schema object can also be constructed by providing a DataFrame or Ibis table object (using the `tbl=` parameter) and the schema will be collected from either type of object. The schema object can be printed to display the column names and dtypes. Note that if `tbl=` is provided then there shouldn't be any other inputs provided through either `columns=` or `**kwargs`.

## Parameters

`columns: str | list[str] | list[tuple[str, str]] | list[tuple[str]] | dict[str, str] | None = None`  
A list of strings (representing column names), a list of tuples (for column names and column dtypes), or a dictionary containing column and dtype information. If any of these inputs are provided here, it will take precedence over any column arguments provided via `**kwargs`.

`tbl: Any | None = None`  
A DataFrame (Polars or Pandas) or an Ibis table object from which the schema will be collected. Read the *Supported Input Table Types* section for details on the supported table types.

`**kwargs`  
Individual column arguments that are in the form of `column=dtype` or `column=[dtype1, dtype2, ...]`. These will be ignored if the `columns=` parameter is not `None`.

## Returns

`Schema`  
A schema object.

## Supported Input Table Types

The `tbl=` parameter, if used, can be given any of the following table types:

- Polars DataFrame (`"polars"`)
- Pandas DataFrame (`"pandas"`)
- PySpark table (`"pyspark"`)
- DuckDB table (`"duckdb"`)\*
- MySQL table (`"mysql"`)\*
- PostgreSQL table (`"postgresql"`)\*
- SQLite table (`"sqlite"`)\*
- Microsoft SQL Server table (`"mssql"`)\*
- Snowflake table (`"snowflake"`)\*
- Databricks table (`"databricks"`)\*
- BigQuery table (`"bigquery"`)\*
- Parquet table (`"parquet"`)\*

The table types marked with an asterisk need to be prepared as Ibis tables (with type of `ibis.expr.types.relations.Table`). Furthermore, using `Schema(tbl=)` with these types of tables requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a Polars or Pandas DataFrame, the availability of Ibis is not needed.

## Additional Notes On Schema Construction

While there is flexibility in how a schema can be constructed, there is the potential for some confusion. So let's go through each of the methods of constructing a schema in more detail and single out some important points.

When providing a list of column names to `columns=`, a [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) validation step will only check the column names. Any arguments pertaining to dtypes will be ignored.

When using a list of tuples in `columns=`, the tuples could contain the column name and dtype or just the column name. This construction allows for more flexibility in constructing the schema as some columns will be checked for dtypes and others will not. This method is the only way to have mixed checks of column names and dtypes in [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match).

When providing a dictionary to `columns=`, the keys are the column names and the values are the dtypes. This method of input is useful in those cases where you might already have a dictionary of column names and dtypes that you want to use as the schema.

If using individual column arguments in the form of keyword arguments, the column names are the keyword arguments and the dtypes are the values. This method emphasizes readability and is perhaps more convenient when manually constructing a schema with a small number of columns.

Finally, multiple dtypes can be provided for a single column by providing a list or tuple of dtypes in place of a scalar string value. Having multiple dtypes for a column allows for the dtype check via [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) to make multiple attempts at matching the column dtype. Should any of the dtypes match the column dtype, that part of the schema check will pass. Here are some examples of how you could provide single and multiple dtypes for a column:

``` python
# list of tuples
schema_1 = pb.Schema(columns=[("name", "String"), ("age", ["Float64", "Int64"])])

# dictionary
schema_2 = pb.Schema(columns={"name": "String", "age": ["Float64", "Int64"]})

# keyword arguments
schema_3 = pb.Schema(name="String", age=["Float64", "Int64"])
```

All of the above examples will construct the same schema object.

## Examples

A schema can be constructed via the [Schema](../reference/Schema.md#pointblank.Schema) class in multiple ways. Let's use the following Polars DataFrame as a basis for constructing a schema:

``` python
import pointblank as pb
import polars as pl

df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "height": [5.6, 6.0, 5.8]
})
```

You could provide `Schema(columns=)` a list of tuples containing column names and data types:

``` python
schema = pb.Schema(columns=[("name", "String"), ("age", "Int64"), ("height", "Float64")])
```

Alternatively, a dictionary containing column names and dtypes also works:

``` python
schema = pb.Schema(columns={"name": "String", "age": "Int64", "height": "Float64"})
```

Another input method involves using individual column arguments in the form of keyword arguments:

``` python
schema = pb.Schema(name="String", age="Int64", height="Float64")
```

Finally, could also provide a DataFrame (Polars and Pandas) or an Ibis table object to `tbl=` and the schema will be collected:

``` python
schema = pb.Schema(tbl=df)
```

Whichever method you choose, you can verify the schema inputs by printing the `schema` object:

``` python
print(schema)
```

    Pointblank Schema
      name: String
      age: Int64
      height: Float64

The [Schema](../reference/Schema.md#pointblank.Schema) object can be used to validate the structure of a table against the schema. The relevant [Validate](../reference/Validate.md#pointblank.Validate) method for this is [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match). In a validation workflow, you'll have a target table (defined at the beginning of the workflow) and you might want to ensure that your expectations of the table structure are met. The [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) method works with a [Schema](../reference/Schema.md#pointblank.Schema) object to validate the structure of the table. Here's an example of how you could use [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) in a validation workflow:

``` python
# Define the schema
schema = pb.Schema(name="String", age="Int64", height="Float64")

# Define a validation that checks the schema against the table (`df`)
validation = (
    pb.Validate(data=df)
    .col_schema_match(schema=schema)
    .interrogate()
)

# Display the validation results
validation
```

[TABLE]

The [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) validation method will validate the structure of the table against the schema during interrogation. If the structure of the table does not match the schema, the single test unit will fail. In this case, the defined schema matched the structure of the table, so the validation passed.

We can also choose to check only the column names of the target table. This can be done by providing a simplified [Schema](../reference/Schema.md#pointblank.Schema) object, which is given a list of column names:

``` python
schema = pb.Schema(columns=["name", "age", "height"])

validation = (
    pb.Validate(data=df)
    .col_schema_match(schema=schema)
    .interrogate()
)

validation
```

[TABLE]

In this case, the schema only checks the column names of the table against the schema during interrogation. If the column names of the table do not match the schema, the single test unit will fail. In this case, the defined schema matched the column names of the table, so the validation passed.

If you wanted to check column names and dtypes only for a subset of columns (and just the column names for the rest), you could use a list of mixed one- or two-item tuples in `columns=`:

``` python
schema = pb.Schema(columns=[("name", "String"), ("age", ), ("height", )])

validation = (
    pb.Validate(data=df)
    .col_schema_match(schema=schema)
    .interrogate()
)

validation
```

[TABLE]

Not specifying a dtype for a column (as is the case for the `age` and `height` columns in the above example) will only check the column name.

There may also be the case where you want to check the column names and specify multiple dtypes for a column to have several attempts at matching the dtype. This can be done by providing a list of dtypes where there would normally be a single dtype:

``` python
schema = pb.Schema(
  columns=[("name", "String"), ("age", ["Float64", "Int64"]), ("height", "Float64")]
)

validation = (
    pb.Validate(data=df)
    .col_schema_match(schema=schema)
    .interrogate()
)

validation
```

[TABLE]

For the `age` column, the schema will check for both `Float64` and `Int64` dtypes. If either of these dtypes is found in the column, the portion of the schema check will succeed.

## See Also

`The` [`col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) validation method, where a  

[Schema](../reference/Schema.md#pointblank.Schema) object is used in a validation workflow.  
