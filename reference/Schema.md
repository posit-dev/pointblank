## Schema


Definition of a schema object.


Usage

``` python
Schema()
```


The schema object defines the structure of a table. Once it is defined, the object can be used in a validation workflow, using [Validate](Validate.md#pointblank.Validate) and its methods, to ensure that the structure of a table matches the expected schema. The validation method that works with the schema object is called <a href="Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>col_schema_match()</code></a>.

A schema for a table can be constructed with the [Schema](Schema.md#pointblank.Schema) class in a number of ways:

1.  providing a list of column names to `columns=` (to check only the column names)
2.  using a list of one- or two-element tuples in `columns=` (to check both column names and optionally dtypes, should be in the form of `[(column_name, dtype), ...]`)
3.  providing a dictionary to `columns=`, where the keys are column names and the values are dtypes
4.  providing individual column arguments in the form of keyword arguments (constructed as `column_name=dtype`)

The schema object can also be constructed by providing a DataFrame or Ibis table object (using the `tbl=` parameter) and the schema will be collected from either type of object. The schema object can be printed to display the column names and dtypes. Note that if `tbl=` is provided then there shouldn't be any other inputs provided through either `columns=` or `**kwargs`.


## Parameters


`columns: str | list[str] | list[tuple[str, str]] | list[tuple[str]] | dict[str, str] | None`  
`= None`  
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

When providing a list of column names to `columns=`, a <a href="Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>col_schema_match()</code></a> validation step will only check the column names. Any arguments pertaining to dtypes will be ignored.

When using a list of tuples in `columns=`, the tuples could contain the column name and dtype or just the column name. This construction allows for more flexibility in constructing the schema as some columns will be checked for dtypes and others will not. This method is the only way to have mixed checks of column names and dtypes in <a href="Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>col_schema_match()</code></a>.

When providing a dictionary to `columns=`, the keys are the column names and the values are the dtypes. This method of input is useful in those cases where you might already have a dictionary of column names and dtypes that you want to use as the schema.

If using individual column arguments in the form of keyword arguments, the column names are the keyword arguments and the dtypes are the values. This method emphasizes readability and is perhaps more convenient when manually constructing a schema with a small number of columns.

Finally, multiple dtypes can be provided for a single column by providing a list or tuple of dtypes in place of a scalar string value. Having multiple dtypes for a column allows for the dtype check via <a href="Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>col_schema_match()</code></a> to make multiple attempts at matching the column dtype. Should any of the dtypes match the column dtype, that part of the schema check will pass. Here are some examples of how you could provide single and multiple dtypes for a column:

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

A schema can be constructed via the [Schema](Schema.md#pointblank.Schema) class in multiple ways. Let's use the following Polars DataFrame as a basis for constructing a schema:


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


The [Schema](Schema.md#pointblank.Schema) object can be used to validate the structure of a table against the schema. The relevant [Validate](Validate.md#pointblank.Validate) method for this is <a href="Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>col_schema_match()</code></a>. In a validation workflow, you'll have a target table (defined at the beginning of the workflow) and you might want to ensure that your expectations of the table structure are met. The <a href="Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>col_schema_match()</code></a> method works with a [Schema](Schema.md#pointblank.Schema) object to validate the structure of the table. Here's an example of how you could use <a href="Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>col_schema_match()</code></a> in a validation workflow:


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


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-type_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">STEP</th>
<th id="pb_tbl-columns_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">COLUMNS</th>
<th id="pb_tbl-values_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">VALUES</th>
<th id="pb_tbl-tbl" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">TBL</th>
<th id="pb_tbl-eval" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EVAL</th>
<th id="pb_tbl-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNITS</th>
<th id="pb_tbl-pass" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">PASS</th>
<th id="pb_tbl-fail" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">FAIL</th>
<th id="pb_tbl-w_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">W</th>
<th id="pb_tbl-e_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">E</th>
<th id="pb_tbl-c_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">C</th>
<th id="pb_tbl-extract_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EXT</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3NjaGVtYV9tYXRjaDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfc2NoZW1hX21hdGNoIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4zMTAzNDUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01My43MTIyMzQsMzkuNzg4NTI2OCBMNTQuMjEyMjM0LDU2LjI4ODUyNjggTDQyLjIxMjIzNCw1Ni43ODg1MjY4IEw0Mi4yMTIyMzQsMzkuNzg4NTI2OCBMNTMuNzEyMjM0LDM5Ljc4ODUyNjggWiBNMzkuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCw1Ni43ODg1MjY4IEwyNy43MTIyMzQsNTYuNzg4NTI2OCBMMjcuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCwzOS43ODg1MjY4IFogTTI1LjIxMjIzNCwzOS43ODg1MjY4IEwyNS4yMTIyMzQsNTYuNzg4NTI2OCBMMTMuNzEyMjM0LDU2Ljc4ODUyNjggTDEzLjIxMjIzNCw0MC4yODg1MjY4IEwyNS4yMTIyMzQsMzkuNzg4NTI2OCBaIiBpZD0iY29sdW1uc19zY2hlbWEiIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8ZyBpZD0idmVydGljYWxfZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMwLjAwMDAwMCwgMjkuMDAwMDAwKSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIj4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIyLjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSIyLjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUiPjwvbGluZT4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSI1LjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSI1LjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUtQ29weSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00MS43MTIyMzQsOS4wMTQ2NjkzNSBMNDEuNzEyMjM0LDI3LjAxNDY2OTQgTDUzLjcxMjIzNCwyNy4wMTQ2Njk0IEM1NC4yNjIyMzQsMjcuMDE0NjY5NCA1NC43MTIyMzQsMjYuNTY0NjY5NCA1NC43MTIyMzQsMjYuMDE0NjY5NCBMNTQuNzEyMjM0LDEwLjAxNDY2OTQgQzU0LjcxMjIzNCw5LjQ2NDY2OTM1IDU0LjI2MjIzNCw5LjAxNDY2OTM1IDUzLjcxMjIzNCw5LjAxNDY2OTM1IEw0MS43MTIyMzQsOS4wMTQ2NjkzNSBaIE0yNy4yMTIyMzQsOS4wMTQ2NjkzNSBDMjcuMjEyMjM0LDkuMDE0NjY5MzUgMjcuMjEyMjM0LDE1LjAxNDY2OTQgMjcuMjEyMjM0LDI3LjAxNDY2OTQgTDQwLjIxMjIzNCwyNy4wMTQ2Njk0IEw0MC4yMTIyMzQsOS4wMTQ2NjkzNSBDMzEuNTQ1NTY3Myw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IFogTTEzLjcxMjIzNCw5LjAxNDY2OTM1IEMxMy4xNjIyMzQsOS4wMTQ2NjkzNSAxMi43MTIyMzQsOS40NjQ2NjkzNSAxMi43MTIyMzQsMTAuMDE0NjY5NCBMMTIuNzEyMjM0LDI2LjAxNDY2OTQgQzEyLjcxMjIzNCwyNi41NjQ2Njk0IDEzLjE2MjIzNCwyNy4wMTQ2Njk0IDEzLjcxMjIzNCwyNy4wMTQ2Njk0IEwyNS43MTIyMzQsMjcuMDE0NjY5NCBMMjUuNzEyMjM0LDkuMDE0NjY5MzUgTDEzLjcxMjIzNCw5LjAxNDY2OTM1IFoiIGlkPSJjb2x1bW5zX3JlYWwiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_schema_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">SCHEMA</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(schema_check)</span> <span style="color:#4CA64C;">✓</span> Schema validation <strong>passed</strong>.</p>
Schema Comparison


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">name</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">age</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Int64</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">height</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">height</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('name', 'String'), ('age', 'Int64'), ('height', 'Float64')]</code>
</div></td>
</tr>
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
Schema Match Settings

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64


</div></td>
</tr>
</tfoot>

</table>

</div></td>
</tr>
</tfoot>

</table>


The <a href="Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>col_schema_match()</code></a> validation method will validate the structure of the table against the schema during interrogation. If the structure of the table does not match the schema, the single test unit will fail. In this case, the defined schema matched the structure of the table, so the validation passed.

We can also choose to check only the column names of the target table. This can be done by providing a simplified [Schema](Schema.md#pointblank.Schema) object, which is given a list of column names:


``` python
schema = pb.Schema(columns=["name", "age", "height"])

validation = (
    pb.Validate(data=df)
    .col_schema_match(schema=schema)
    .interrogate()
)

validation
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-type_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">STEP</th>
<th id="pb_tbl-columns_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">COLUMNS</th>
<th id="pb_tbl-values_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">VALUES</th>
<th id="pb_tbl-tbl" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">TBL</th>
<th id="pb_tbl-eval" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EVAL</th>
<th id="pb_tbl-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNITS</th>
<th id="pb_tbl-pass" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">PASS</th>
<th id="pb_tbl-fail" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">FAIL</th>
<th id="pb_tbl-w_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">W</th>
<th id="pb_tbl-e_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">E</th>
<th id="pb_tbl-c_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">C</th>
<th id="pb_tbl-extract_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EXT</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3NjaGVtYV9tYXRjaDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfc2NoZW1hX21hdGNoIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4zMTAzNDUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01My43MTIyMzQsMzkuNzg4NTI2OCBMNTQuMjEyMjM0LDU2LjI4ODUyNjggTDQyLjIxMjIzNCw1Ni43ODg1MjY4IEw0Mi4yMTIyMzQsMzkuNzg4NTI2OCBMNTMuNzEyMjM0LDM5Ljc4ODUyNjggWiBNMzkuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCw1Ni43ODg1MjY4IEwyNy43MTIyMzQsNTYuNzg4NTI2OCBMMjcuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCwzOS43ODg1MjY4IFogTTI1LjIxMjIzNCwzOS43ODg1MjY4IEwyNS4yMTIyMzQsNTYuNzg4NTI2OCBMMTMuNzEyMjM0LDU2Ljc4ODUyNjggTDEzLjIxMjIzNCw0MC4yODg1MjY4IEwyNS4yMTIyMzQsMzkuNzg4NTI2OCBaIiBpZD0iY29sdW1uc19zY2hlbWEiIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8ZyBpZD0idmVydGljYWxfZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMwLjAwMDAwMCwgMjkuMDAwMDAwKSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIj4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIyLjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSIyLjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUiPjwvbGluZT4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSI1LjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSI1LjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUtQ29weSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00MS43MTIyMzQsOS4wMTQ2NjkzNSBMNDEuNzEyMjM0LDI3LjAxNDY2OTQgTDUzLjcxMjIzNCwyNy4wMTQ2Njk0IEM1NC4yNjIyMzQsMjcuMDE0NjY5NCA1NC43MTIyMzQsMjYuNTY0NjY5NCA1NC43MTIyMzQsMjYuMDE0NjY5NCBMNTQuNzEyMjM0LDEwLjAxNDY2OTQgQzU0LjcxMjIzNCw5LjQ2NDY2OTM1IDU0LjI2MjIzNCw5LjAxNDY2OTM1IDUzLjcxMjIzNCw5LjAxNDY2OTM1IEw0MS43MTIyMzQsOS4wMTQ2NjkzNSBaIE0yNy4yMTIyMzQsOS4wMTQ2NjkzNSBDMjcuMjEyMjM0LDkuMDE0NjY5MzUgMjcuMjEyMjM0LDE1LjAxNDY2OTQgMjcuMjEyMjM0LDI3LjAxNDY2OTQgTDQwLjIxMjIzNCwyNy4wMTQ2Njk0IEw0MC4yMTIyMzQsOS4wMTQ2NjkzNSBDMzEuNTQ1NTY3Myw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IFogTTEzLjcxMjIzNCw5LjAxNDY2OTM1IEMxMy4xNjIyMzQsOS4wMTQ2NjkzNSAxMi43MTIyMzQsOS40NjQ2NjkzNSAxMi43MTIyMzQsMTAuMDE0NjY5NCBMMTIuNzEyMjM0LDI2LjAxNDY2OTQgQzEyLjcxMjIzNCwyNi41NjQ2Njk0IDEzLjE2MjIzNCwyNy4wMTQ2Njk0IDEzLjcxMjIzNCwyNy4wMTQ2Njk0IEwyNS43MTIyMzQsMjcuMDE0NjY5NCBMMjUuNzEyMjM0LDkuMDE0NjY5MzUgTDEzLjcxMjIzNCw5LjAxNDY2OTM1IFoiIGlkPSJjb2x1bW5zX3JlYWwiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_schema_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">SCHEMA</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(schema_check)</span> <span style="color:#4CA64C;">✓</span> Schema validation <strong>passed</strong>.</p>
Schema Comparison


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">name</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">age</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">height</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">height</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('name',), ('age',), ('height',)]</code>
</div></td>
</tr>
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
Schema Match Settings

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64


</div></td>
</tr>
</tfoot>

</table>

</div></td>
</tr>
</tfoot>

</table>


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


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-type_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">STEP</th>
<th id="pb_tbl-columns_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">COLUMNS</th>
<th id="pb_tbl-values_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">VALUES</th>
<th id="pb_tbl-tbl" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">TBL</th>
<th id="pb_tbl-eval" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EVAL</th>
<th id="pb_tbl-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNITS</th>
<th id="pb_tbl-pass" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">PASS</th>
<th id="pb_tbl-fail" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">FAIL</th>
<th id="pb_tbl-w_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">W</th>
<th id="pb_tbl-e_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">E</th>
<th id="pb_tbl-c_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">C</th>
<th id="pb_tbl-extract_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EXT</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3NjaGVtYV9tYXRjaDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfc2NoZW1hX21hdGNoIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4zMTAzNDUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01My43MTIyMzQsMzkuNzg4NTI2OCBMNTQuMjEyMjM0LDU2LjI4ODUyNjggTDQyLjIxMjIzNCw1Ni43ODg1MjY4IEw0Mi4yMTIyMzQsMzkuNzg4NTI2OCBMNTMuNzEyMjM0LDM5Ljc4ODUyNjggWiBNMzkuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCw1Ni43ODg1MjY4IEwyNy43MTIyMzQsNTYuNzg4NTI2OCBMMjcuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCwzOS43ODg1MjY4IFogTTI1LjIxMjIzNCwzOS43ODg1MjY4IEwyNS4yMTIyMzQsNTYuNzg4NTI2OCBMMTMuNzEyMjM0LDU2Ljc4ODUyNjggTDEzLjIxMjIzNCw0MC4yODg1MjY4IEwyNS4yMTIyMzQsMzkuNzg4NTI2OCBaIiBpZD0iY29sdW1uc19zY2hlbWEiIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8ZyBpZD0idmVydGljYWxfZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMwLjAwMDAwMCwgMjkuMDAwMDAwKSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIj4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIyLjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSIyLjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUiPjwvbGluZT4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSI1LjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSI1LjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUtQ29weSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00MS43MTIyMzQsOS4wMTQ2NjkzNSBMNDEuNzEyMjM0LDI3LjAxNDY2OTQgTDUzLjcxMjIzNCwyNy4wMTQ2Njk0IEM1NC4yNjIyMzQsMjcuMDE0NjY5NCA1NC43MTIyMzQsMjYuNTY0NjY5NCA1NC43MTIyMzQsMjYuMDE0NjY5NCBMNTQuNzEyMjM0LDEwLjAxNDY2OTQgQzU0LjcxMjIzNCw5LjQ2NDY2OTM1IDU0LjI2MjIzNCw5LjAxNDY2OTM1IDUzLjcxMjIzNCw5LjAxNDY2OTM1IEw0MS43MTIyMzQsOS4wMTQ2NjkzNSBaIE0yNy4yMTIyMzQsOS4wMTQ2NjkzNSBDMjcuMjEyMjM0LDkuMDE0NjY5MzUgMjcuMjEyMjM0LDE1LjAxNDY2OTQgMjcuMjEyMjM0LDI3LjAxNDY2OTQgTDQwLjIxMjIzNCwyNy4wMTQ2Njk0IEw0MC4yMTIyMzQsOS4wMTQ2NjkzNSBDMzEuNTQ1NTY3Myw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IFogTTEzLjcxMjIzNCw5LjAxNDY2OTM1IEMxMy4xNjIyMzQsOS4wMTQ2NjkzNSAxMi43MTIyMzQsOS40NjQ2NjkzNSAxMi43MTIyMzQsMTAuMDE0NjY5NCBMMTIuNzEyMjM0LDI2LjAxNDY2OTQgQzEyLjcxMjIzNCwyNi41NjQ2Njk0IDEzLjE2MjIzNCwyNy4wMTQ2Njk0IDEzLjcxMjIzNCwyNy4wMTQ2Njk0IEwyNS43MTIyMzQsMjcuMDE0NjY5NCBMMjUuNzEyMjM0LDkuMDE0NjY5MzUgTDEzLjcxMjIzNCw5LjAxNDY2OTM1IFoiIGlkPSJjb2x1bW5zX3JlYWwiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_schema_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">SCHEMA</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(schema_check)</span> <span style="color:#4CA64C;">✓</span> Schema validation <strong>passed</strong>.</p>
Schema Comparison


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">name</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">age</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">height</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">height</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('name', 'String'), ('age',), ('height',)]</code>
</div></td>
</tr>
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
Schema Match Settings

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64


</div></td>
</tr>
</tfoot>

</table>

</div></td>
</tr>
</tfoot>

</table>


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


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-type_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">STEP</th>
<th id="pb_tbl-columns_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">COLUMNS</th>
<th id="pb_tbl-values_upd" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">VALUES</th>
<th id="pb_tbl-tbl" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">TBL</th>
<th id="pb_tbl-eval" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EVAL</th>
<th id="pb_tbl-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNITS</th>
<th id="pb_tbl-pass" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">PASS</th>
<th id="pb_tbl-fail" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">FAIL</th>
<th id="pb_tbl-w_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">W</th>
<th id="pb_tbl-e_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">E</th>
<th id="pb_tbl-c_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">C</th>
<th id="pb_tbl-extract_upd" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: #666666; font-weight: bold" scope="col">EXT</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3NjaGVtYV9tYXRjaDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfc2NoZW1hX21hdGNoIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4zMTAzNDUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01My43MTIyMzQsMzkuNzg4NTI2OCBMNTQuMjEyMjM0LDU2LjI4ODUyNjggTDQyLjIxMjIzNCw1Ni43ODg1MjY4IEw0Mi4yMTIyMzQsMzkuNzg4NTI2OCBMNTMuNzEyMjM0LDM5Ljc4ODUyNjggWiBNMzkuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCw1Ni43ODg1MjY4IEwyNy43MTIyMzQsNTYuNzg4NTI2OCBMMjcuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCwzOS43ODg1MjY4IFogTTI1LjIxMjIzNCwzOS43ODg1MjY4IEwyNS4yMTIyMzQsNTYuNzg4NTI2OCBMMTMuNzEyMjM0LDU2Ljc4ODUyNjggTDEzLjIxMjIzNCw0MC4yODg1MjY4IEwyNS4yMTIyMzQsMzkuNzg4NTI2OCBaIiBpZD0iY29sdW1uc19zY2hlbWEiIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8ZyBpZD0idmVydGljYWxfZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMwLjAwMDAwMCwgMjkuMDAwMDAwKSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIj4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIyLjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSIyLjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUiPjwvbGluZT4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSI1LjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSI1LjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUtQ29weSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00MS43MTIyMzQsOS4wMTQ2NjkzNSBMNDEuNzEyMjM0LDI3LjAxNDY2OTQgTDUzLjcxMjIzNCwyNy4wMTQ2Njk0IEM1NC4yNjIyMzQsMjcuMDE0NjY5NCA1NC43MTIyMzQsMjYuNTY0NjY5NCA1NC43MTIyMzQsMjYuMDE0NjY5NCBMNTQuNzEyMjM0LDEwLjAxNDY2OTQgQzU0LjcxMjIzNCw5LjQ2NDY2OTM1IDU0LjI2MjIzNCw5LjAxNDY2OTM1IDUzLjcxMjIzNCw5LjAxNDY2OTM1IEw0MS43MTIyMzQsOS4wMTQ2NjkzNSBaIE0yNy4yMTIyMzQsOS4wMTQ2NjkzNSBDMjcuMjEyMjM0LDkuMDE0NjY5MzUgMjcuMjEyMjM0LDE1LjAxNDY2OTQgMjcuMjEyMjM0LDI3LjAxNDY2OTQgTDQwLjIxMjIzNCwyNy4wMTQ2Njk0IEw0MC4yMTIyMzQsOS4wMTQ2NjkzNSBDMzEuNTQ1NTY3Myw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IFogTTEzLjcxMjIzNCw5LjAxNDY2OTM1IEMxMy4xNjIyMzQsOS4wMTQ2NjkzNSAxMi43MTIyMzQsOS40NjQ2NjkzNSAxMi43MTIyMzQsMTAuMDE0NjY5NCBMMTIuNzEyMjM0LDI2LjAxNDY2OTQgQzEyLjcxMjIzNCwyNi41NjQ2Njk0IDEzLjE2MjIzNCwyNy4wMTQ2Njk0IDEzLjcxMjIzNCwyNy4wMTQ2Njk0IEwyNS43MTIyMzQsMjcuMDE0NjY5NCBMMjUuNzEyMjM0LDkuMDE0NjY5MzUgTDEzLjcxMjIzNCw5LjAxNDY2OTM1IFoiIGlkPSJjb2x1bW5zX3JlYWwiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_schema_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">SCHEMA</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(schema_check)</span> <span style="color:#4CA64C;">✓</span> Schema validation <strong>passed</strong>.</p>
Schema Comparison


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">name</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">age</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64 | <span style="text-decoration: underline; text-decoration-color: #4CA64C; text-underline-offset: 3px;">Int64</span></td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">height</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">height</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('name', 'String'), ('age', ['Float64', 'Int64']), ('height', 'Float64')]</code>
</div></td>
</tr>
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
Schema Match Settings

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64


</div></td>
</tr>
</tfoot>

</table>

</div></td>
</tr>
</tfoot>

</table>


For the `age` column, the schema will check for both `Float64` and `Int64` dtypes. If either of these dtypes is found in the column, the portion of the schema check will succeed.


#### See Also

[The](The.md)
