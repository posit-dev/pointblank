## Validate.tbl_match()


Validate whether the target table matches a comparison table.


Usage

``` python
Validate.tbl_match(
    tbl_compare,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True
)
```


The [tbl_match()](Validate.tbl_match.md#pointblank.Validate.tbl_match) method checks whether the target table's composition matches that of a comparison table. The validation performs a comprehensive comparison using progressively stricter checks (from least to most stringent):

1.  **Column count match**: both tables must have the same number of columns
2.  **Row count match**: both tables must have the same number of rows
3.  **Schema match (loose)**: column names and dtypes match (case-insensitive, any order)
4.  **Schema match (order)**: columns in the correct order (case-insensitive names)
5.  **Schema match (exact)**: column names match exactly (case-sensitive, correct order)
6.  **Data match**: values in corresponding cells must be identical

This progressive approach helps identify exactly where tables differ. The validation will fail at the first check that doesn't pass, making it easier to diagnose mismatches. This validation operates over a single test unit (pass/fail for complete table match).


## Parameters


`tbl_compare: Any`  
The comparison table to validate against. This can be a DataFrame object (Polars or Pandas), an Ibis table object, or a callable that returns a table. If a callable is provided, it will be executed during interrogation to obtain the comparison table.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table. Have a look at the *Preprocessing* section for more information on how to use this argument.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect. Look at the *Thresholds* section for information on how to set threshold levels.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the <a href="Actions.html#pointblank.Actions" class="gdls-link"><code>Actions</code></a> class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = ``True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like <a href="has_columns.html#pointblank.has_columns" class="gdls-link"><code>has_columns()</code></a> and <a href="has_rows.html#pointblank.has_rows" class="gdls-link"><code>has_rows()</code></a> can be used here to conditionally activate a step based on properties of the target table.


## Returns


`Validate`  
The [Validate](Validate.md#pointblank.Validate) object with the added validation step.


## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Note that the same preprocessing is **not** applied to the comparison table; only the target table is preprocessed. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](Validate.md#pointblank.Validate) object or used in subsequent validation steps.


## Thresholds

The `thresholds=` parameter is used to set the failure-condition levels for the validation step. If they are set here at the step level, these thresholds will override any thresholds set at the global level in `Validate(thresholds=...)`.

There are three threshold levels: 'warning', 'error', and 'critical'. The threshold values can either be set as a proportion failing of all test units (a value between `0` to `1`), or, the absolute number of failing test units (as integer that's `1` or greater).

Thresholds can be defined using one of these input schemes:

1.  use the <a href="Thresholds.html#pointblank.Thresholds" class="gdls-link"><code>Thresholds</code></a> class (the most direct way to create thresholds)
2.  provide a tuple of 1-3 values, where position `0` is the 'warning' level, position `1` is the 'error' level, and position `2` is the 'critical' level
3.  create a dictionary of 1-3 value entries; the valid keys: are 'warning', 'error', and 'critical'
4.  a single integer/float value denoting absolute number or fraction of failing test units for the 'warning' level only

If the number of failing test units exceeds set thresholds, the validation step will be marked as 'warning', 'error', or 'critical'. All of the threshold levels don't need to be set, you're free to set any combination of them.

Aside from reporting failure conditions, thresholds can be used to determine the actions to take for each level of failure (using the `actions=` parameter).


## Cross-Backend Validation

The [tbl_match()](Validate.tbl_match.md#pointblank.Validate.tbl_match) method supports **automatic backend coercion** when comparing tables from different backends (e.g., comparing a Polars DataFrame against a Pandas DataFrame, or comparing database tables from DuckDB/SQLite against in-memory DataFrames). When tables with different backends are detected, the comparison table is automatically converted to match the data table's backend before validation proceeds.

**Certified Backend Combinations:**

All combinations of the following backends have been tested and certified to work (in both directions):

- Pandas DataFrame
- Polars DataFrame
- DuckDB (native)
- DuckDB (as Ibis table)
- SQLite (via Ibis)

Note that database backends (DuckDB, SQLite, PostgreSQL, MySQL, Snowflake, BigQuery) are automatically materialized during validation:

- if comparing **against Polars**: materialized to Polars
- if comparing **against Pandas**: materialized to Pandas
- if **both tables are database backends**: both materialized to Polars

This ensures optimal performance and type consistency.

**Data Types That Work Best in Cross-Backend Validation:**

- numeric types: int, float columns (including proper NaN handling)
- string types: text columns with consistent encodings
- boolean types: True/False values
- null values: `None` and `NaN` are treated as equivalent across backends
- list columns: nested list structures (with basic types)

**Known Limitations:**

While many data types work well in cross-backend validation, there are some known limitations to be aware of:

- date/datetime types: When converting between Polars and Pandas, date objects may be represented differently. For example, `datetime.date` objects in Pandas may become `pd.Timestamp` objects when converted from Polars, leading to false mismatches. To work around this, ensure both tables use the same datetime representation before comparison.
- custom types: User-defined types or complex nested structures may not convert cleanly between backends and could cause unexpected comparison failures.
- categorical types: Categorical/factor columns may have different internal representations across backends.
- timezone-aware datetimes: Timezone handling differs between backends and may cause comparison issues.

Here are some ideas to overcome such limitations:

- for date/datetime columns, consider using `pre=` preprocessing to normalize representations before comparison.
- when working with custom types, manually convert tables to the same backend before using [tbl_match()](Validate.tbl_match.md#pointblank.Validate.tbl_match).
- use the same datetime precision (e.g., milliseconds vs microseconds) in both tables.


## Examples

For the examples here, we'll create two simple tables to demonstrate the [tbl_match()](Validate.tbl_match.md#pointblank.Validate.tbl_match) validation.


``` python
import pointblank as pb
import polars as pl

# Create the first table
tbl_1 = pl.DataFrame({
    "a": [1, 2, 3, 4],
    "b": ["w", "x", "y", "z"],
    "c": [4.0, 5.0, 6.0, 7.0]
})

# Create an identical table
tbl_2 = pl.DataFrame({
    "a": [1, 2, 3, 4],
    "b": ["w", "x", "y", "z"],
    "c": [4.0, 5.0, 6.0, 7.0]
})

pb.preview(tbl_1)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="4" class="gt_heading gt_title gt_font_normal"><div>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">4</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">3</span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_preview_tbl-_row_num_" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"></th>
<th id="pb_preview_tbl-a" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

a

<em>Int64</em>

</div></th>
<th id="pb_preview_tbl-b" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

b

<em>String</em>

</div></th>
<th id="pb_preview_tbl-c" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

c

<em>Float64</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">w</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4.0</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">x</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5.0</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">y</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6.0</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">z</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7.0</td>
</tr>
</tbody>
</table>


Let's validate that `tbl_1` matches `tbl_2`. Since these tables are identical, the validation should pass.


``` python
validation = (
    pb.Validate(data=tbl_1)
    .tbl_match(tbl_compare=tbl_2)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+dGJsX21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InRibF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNzU4NjIxKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8ZyBpZD0iZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ2LjAyNjYxMSwgMjAuNzEwMTIyKSByb3RhdGUoLTkwLjAwMDAwMCkgdHJhbnNsYXRlKC00Ni4wMjY2MTEsIC0yMC43MTAxMjIpIHRyYW5zbGF0ZSg0Mi41MjY2MTEsIDE2LjIxMDEyMikiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLWxpbmVjYXA9InNxdWFyZSI+CiAgICAgICAgICAgICAgICA8bGluZSB4MT0iMi4yMTIyMzM5NyIgeTE9IjAuNTE0NjY5MzUzIiB4Mj0iMi4yMTIyMzM5NyIgeTI9IjcuNTg1NzM3MTYiIGlkPSJMaW5lIj48L2xpbmU+CiAgICAgICAgICAgICAgICA8bGluZSB4MT0iNS4yMTIyMzM5NyIgeTE9IjAuNTE0NjY5MzUzIiB4Mj0iNS4yMTIyMzM5NyIgeTI9IjcuNTg1NzM3MTYiIGlkPSJMaW5lIj48L2xpbmU+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPGcgaWQ9ImVxdWFsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyMS4zOTc4NTcsIDQ1LjMxOTIxNykgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMjEuMzk3ODU3LCAtNDUuMzE5MjE3KSB0cmFuc2xhdGUoMTcuODk3ODU3LCA0MC44MTkyMTcpIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yMS4zODgyNDE5LDcuNzc4Njk3ODMgQzIxLjM1ODQyOTgsNy43NzkzNTE3NyAyMS4zMjg3MDQsNy43ODIxNjM2NyAyMS4yOTkyOTk2LDcuNzg3MTE5MTQgTDkuMDkwMTQ4MjQsNy43ODcxMTkxNCBDOC43NTAyOTQzMSw3Ljc4NzE1MzEyIDguNDc0Nzk2NzcsOC4wNjI2NTA2NyA4LjQ3NDc2Mjc5LDguNDAyNTA0NiBMOC40NzQ3NjI3OSwxNi4yOTkxNDk4IEM4LjQ2Mzc3ODc0LDE2LjM2NTYwNjEgOC40NjM3Nzg3NCwxNi40MzM0MTQ5IDguNDc0NzYyNzksMTYuNDk5ODcxMyBMOC40NzQ3NjI3OSwyNC45MTQ1NDYxIEM4LjQ2Mzc3ODc0LDI0Ljk4MTAwMjUgOC40NjM3Nzg3NCwyNS4wNDg4MTEyIDguNDc0NzYyNzksMjUuMTE1MjY3NiBMOC40NzQ3NjI3OSwzMy4wMTc5MjI2IEM4LjQ3NDc5Njc3LDMzLjM1Nzc3NjYgOC43NTAyOTQzMSwzMy42MzMyNzQxIDkuMDkwMTQ4MjQsMzMuNjMzMzA4MSBMMjEuMjk0NDkxNiwzMy42MzMzMDgxIEMyMS4zNjA5NDc5LDMzLjY0NDI5MjEgMjEuNDI4NzU2NywzMy42NDQyOTIxIDIxLjQ5NTIxMzEsMzMuNjMzMzA4MSBMMzMuNzA1NTY2MywzMy42MzMzMDgxIEMzNC4wNDU0MjAyLDMzLjYzMzI3NDEgMzQuMzIwOTE3OCwzMy4zNTc3NzY2IDM0LjMyMDk1MTcsMzMuMDE3OTIyNiBMMzQuMzIwOTUxNywyNS4xMjEyNzc1IEMzNC4zMzE5MzU4LDI1LjA1NDgyMTEgMzQuMzMxOTM1OCwyNC45ODcwMTIzIDM0LjMyMDk1MTcsMjQuOTIwNTU2IEwzNC4zMjA5NTE3LDE2LjUwNTg4MTEgQzM0LjMzMTkzNTgsMTYuNDM5NDI0OCAzNC4zMzE5MzU4LDE2LjM3MTYxNiAzNC4zMjA5NTE3LDE2LjMwNTE1OTYgTDM0LjMyMDk1MTcsOC40MDI1MDQ2IEMzNC4zMjA5MTc4LDguMDYyNjUwNjcgMzQuMDQ1NDIwMiw3Ljc4NzE1MzEyIDMzLjcwNTU2NjMsNy43ODcxMTkxNCBMMjEuNDkyODA5NCw3Ljc4NzExOTE0IEMyMS40NTgyNTY1LDcuNzgxMzQzNjkgMjEuNDIzMjczNiw3Ljc3ODY5NzgzIDIxLjM4ODI0MTksNy43Nzg2OTc4MyBaIE05LjcwNTUzMzY5LDkuMDE3ODkwMDUgTDIwLjc4MjQ3MTgsOS4wMTc4OTAwNSBMMjAuNzgyNDcxOCwxNS43ODcxMyBMOS43MDU1MzM2OSwxNS43ODcxMyBMOS43MDU1MzM2OSw5LjAxNzg5MDA1IFogTTIyLjAxMzI0MjcsOS4wMTc4OTAwNSBMMzMuMDkwMTgwOCw5LjAxNzg5MDA1IEwzMy4wOTAxODA4LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDkuMDE3ODkwMDUgWiBNOS43MDU1MzM2OSwxNy4wMTc5MDA5IEwyMC43ODI0NzE4LDE3LjAxNzkwMDkgTDIwLjc4MjQ3MTgsMjQuNDAyNTI2MyBMOS43MDU1MzM2OSwyNC40MDI1MjYzIEw5LjcwNTUzMzY5LDE3LjAxNzkwMDkgWiBNMjIuMDEzMjQyNywxNy4wMTc5MDA5IEwzMy4wOTAxODA4LDE3LjAxNzkwMDkgTDMzLjA5MDE4MDgsMjQuNDAyNTI2MyBMMjIuMDEzMjQyNywyNC40MDI1MjYzIEwyMi4wMTMyNDI3LDE3LjAxNzkwMDkgWiBNOS43MDU1MzM2OSwyNS42MzMyOTcyIEwyMC43ODI0NzE4LDI1LjYzMzI5NzIgTDIwLjc4MjQ3MTgsMzIuNDAyNTM3MiBMOS43MDU1MzM2OSwzMi40MDI1MzcyIEw5LjcwNTUzMzY5LDI1LjYzMzI5NzIgWiBNMjIuMDEzMjQyNywyNS42MzMyOTcyIEwzMy4wOTAxODA4LDI1LjYzMzI5NzIgTDMzLjA5MDE4MDgsMzIuNDAyNTM3MiBMMjIuMDEzMjQyNywzMi40MDI1MzcyIEwyMi4wMTMyNDI3LDI1LjYzMzI5NzIgWiIgaWQ9InRhYmxlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00Ni4wMTY5OTUzLDMyLjM4Nzc5MjYgQzQ1Ljk4NzE4MzIsMzIuMzg4NDQ2NSA0NS45NTc0NTc1LDMyLjM5MTI1ODQgNDUuOTI4MDUzLDMyLjM5NjIxMzkgTDMzLjcxODkwMTYsMzIuMzk2MjEzOSBDMzMuMzc5MDQ3NywzMi4zOTYyNDc5IDMzLjEwMzU1MDIsMzIuNjcxNzQ1NCAzMy4xMDM1MTYyLDMzLjAxMTU5OTMgTDMzLjEwMzUxNjIsNDAuOTA4MjQ0NSBDMzMuMDkyNTMyMiw0MC45NzQ3MDA5IDMzLjA5MjUzMjIsNDEuMDQyNTA5NyAzMy4xMDM1MTYyLDQxLjEwODk2NiBMMzMuMTAzNTE2Miw0OS41MjM2NDA4IEMzMy4wOTI1MzIyLDQ5LjU5MDA5NzIgMzMuMDkyNTMyMiw0OS42NTc5MDYgMzMuMTAzNTE2Miw0OS43MjQzNjI0IEwzMy4xMDM1MTYyLDU3LjYyNzAxNzQgQzMzLjEwMzU1MDIsNTcuOTY2ODcxMyAzMy4zNzkwNDc3LDU4LjI0MjM2ODkgMzMuNzE4OTAxNiw1OC4yNDI0MDI4IEw0NS45MjMyNDUsNTguMjQyNDAyOCBDNDUuOTg5NzAxNCw1OC4yNTMzODY5IDQ2LjA1NzUxMDEsNTguMjUzMzg2OSA0Ni4xMjM5NjY1LDU4LjI0MjQwMjggTDU4LjMzNDMxOTcsNTguMjQyNDAyOCBDNTguNjc0MTczNiw1OC4yNDIzNjg5IDU4Ljk0OTY3MTIsNTcuOTY2ODcxMyA1OC45NDk3MDUxLDU3LjYyNzAxNzQgTDU4Ljk0OTcwNTEsNDkuNzMwMzcyMiBDNTguOTYwNjg5Miw0OS42NjM5MTU4IDU4Ljk2MDY4OTIsNDkuNTk2MTA3MSA1OC45NDk3MDUxLDQ5LjUyOTY1MDcgTDU4Ljk0OTcwNTEsNDEuMTE0OTc1OSBDNTguOTYwNjg5Miw0MS4wNDg1MTk1IDU4Ljk2MDY4OTIsNDAuOTgwNzEwNyA1OC45NDk3MDUxLDQwLjkxNDI1NDQgTDU4Ljk0OTcwNTEsMzMuMDExNTk5MyBDNTguOTQ5NjcxMiwzMi42NzE3NDU0IDU4LjY3NDE3MzYsMzIuMzk2MjQ3OSA1OC4zMzQzMTk3LDMyLjM5NjIxMzkgTDQ2LjEyMTU2MjgsMzIuMzk2MjEzOSBDNDYuMDg3MDA5OSwzMi4zOTA0Mzg0IDQ2LjA1MjAyNywzMi4zODc3OTI2IDQ2LjAxNjk5NTMsMzIuMzg3NzkyNiBaIE0zNC4zMzQyODcxLDMzLjYyNjk4NDggTDQ1LjQxMTIyNTIsMzMuNjI2OTg0OCBMNDUuNDExMjI1Miw0MC4zOTYyMjQ4IEwzNC4zMzQyODcxLDQwLjM5NjIyNDggTDM0LjMzNDI4NzEsMzMuNjI2OTg0OCBaIE00Ni42NDE5OTYxLDMzLjYyNjk4NDggTDU3LjcxODkzNDIsMzMuNjI2OTg0OCBMNTcuNzE4OTM0Miw0MC4zOTYyMjQ4IEw0Ni42NDE5OTYxLDQwLjM5NjIyNDggTDQ2LjY0MTk5NjEsMzMuNjI2OTg0OCBaIE0zNC4zMzQyODcxLDQxLjYyNjk5NTcgTDQ1LjQxMTIyNTIsNDEuNjI2OTk1NyBMNDUuNDExMjI1Miw0OS4wMTE2MjExIEwzNC4zMzQyODcxLDQ5LjAxMTYyMTEgTDM0LjMzNDI4NzEsNDEuNjI2OTk1NyBaIE00Ni42NDE5OTYxLDQxLjYyNjk5NTcgTDU3LjcxODkzNDIsNDEuNjI2OTk1NyBMNTcuNzE4OTM0Miw0OS4wMTE2MjExIEw0Ni42NDE5OTYxLDQ5LjAxMTYyMTEgTDQ2LjY0MTk5NjEsNDEuNjI2OTk1NyBaIE0zNC4zMzQyODcxLDUwLjI0MjM5MiBMNDUuNDExMjI1Miw1MC4yNDIzOTIgTDQ1LjQxMTIyNTIsNTcuMDExNjMxOSBMMzQuMzM0Mjg3MSw1Ny4wMTE2MzE5IEwzNC4zMzQyODcxLDUwLjI0MjM5MiBaIE00Ni42NDE5OTYxLDUwLjI0MjM5MiBMNTcuNzE4OTM0Miw1MC4yNDIzOTIgTDU3LjcxODkzNDIsNTcuMDExNjMxOSBMNDYuNjQxOTk2MSw1Ny4wMTE2MzE5IEw0Ni42NDE5OTYxLDUwLjI0MjM5MiBaIiBpZD0idGFibGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

tbl_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">None</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">EXTERNAL TABLE</td>
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
</tbody>
</table>


The validation table shows that the single test unit passed, indicating that the two tables match completely.

Now, let's create a table with a slight difference and see what happens.


``` python
# Create a table with one different value
tbl_3 = pl.DataFrame({
    "a": [1, 2, 3, 4],
    "b": ["w", "x", "y", "z"],
    "c": [4.0, 5.5, 6.0, 7.0]  # Changed 5.0 to 5.5
})

validation = (
    pb.Validate(data=tbl_1)
    .tbl_match(tbl_compare=tbl_3)
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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+dGJsX21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InRibF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNzU4NjIxKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8ZyBpZD0iZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ2LjAyNjYxMSwgMjAuNzEwMTIyKSByb3RhdGUoLTkwLjAwMDAwMCkgdHJhbnNsYXRlKC00Ni4wMjY2MTEsIC0yMC43MTAxMjIpIHRyYW5zbGF0ZSg0Mi41MjY2MTEsIDE2LjIxMDEyMikiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLWxpbmVjYXA9InNxdWFyZSI+CiAgICAgICAgICAgICAgICA8bGluZSB4MT0iMi4yMTIyMzM5NyIgeTE9IjAuNTE0NjY5MzUzIiB4Mj0iMi4yMTIyMzM5NyIgeTI9IjcuNTg1NzM3MTYiIGlkPSJMaW5lIj48L2xpbmU+CiAgICAgICAgICAgICAgICA8bGluZSB4MT0iNS4yMTIyMzM5NyIgeTE9IjAuNTE0NjY5MzUzIiB4Mj0iNS4yMTIyMzM5NyIgeTI9IjcuNTg1NzM3MTYiIGlkPSJMaW5lIj48L2xpbmU+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPGcgaWQ9ImVxdWFsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyMS4zOTc4NTcsIDQ1LjMxOTIxNykgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMjEuMzk3ODU3LCAtNDUuMzE5MjE3KSB0cmFuc2xhdGUoMTcuODk3ODU3LCA0MC44MTkyMTcpIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yMS4zODgyNDE5LDcuNzc4Njk3ODMgQzIxLjM1ODQyOTgsNy43NzkzNTE3NyAyMS4zMjg3MDQsNy43ODIxNjM2NyAyMS4yOTkyOTk2LDcuNzg3MTE5MTQgTDkuMDkwMTQ4MjQsNy43ODcxMTkxNCBDOC43NTAyOTQzMSw3Ljc4NzE1MzEyIDguNDc0Nzk2NzcsOC4wNjI2NTA2NyA4LjQ3NDc2Mjc5LDguNDAyNTA0NiBMOC40NzQ3NjI3OSwxNi4yOTkxNDk4IEM4LjQ2Mzc3ODc0LDE2LjM2NTYwNjEgOC40NjM3Nzg3NCwxNi40MzM0MTQ5IDguNDc0NzYyNzksMTYuNDk5ODcxMyBMOC40NzQ3NjI3OSwyNC45MTQ1NDYxIEM4LjQ2Mzc3ODc0LDI0Ljk4MTAwMjUgOC40NjM3Nzg3NCwyNS4wNDg4MTEyIDguNDc0NzYyNzksMjUuMTE1MjY3NiBMOC40NzQ3NjI3OSwzMy4wMTc5MjI2IEM4LjQ3NDc5Njc3LDMzLjM1Nzc3NjYgOC43NTAyOTQzMSwzMy42MzMyNzQxIDkuMDkwMTQ4MjQsMzMuNjMzMzA4MSBMMjEuMjk0NDkxNiwzMy42MzMzMDgxIEMyMS4zNjA5NDc5LDMzLjY0NDI5MjEgMjEuNDI4NzU2NywzMy42NDQyOTIxIDIxLjQ5NTIxMzEsMzMuNjMzMzA4MSBMMzMuNzA1NTY2MywzMy42MzMzMDgxIEMzNC4wNDU0MjAyLDMzLjYzMzI3NDEgMzQuMzIwOTE3OCwzMy4zNTc3NzY2IDM0LjMyMDk1MTcsMzMuMDE3OTIyNiBMMzQuMzIwOTUxNywyNS4xMjEyNzc1IEMzNC4zMzE5MzU4LDI1LjA1NDgyMTEgMzQuMzMxOTM1OCwyNC45ODcwMTIzIDM0LjMyMDk1MTcsMjQuOTIwNTU2IEwzNC4zMjA5NTE3LDE2LjUwNTg4MTEgQzM0LjMzMTkzNTgsMTYuNDM5NDI0OCAzNC4zMzE5MzU4LDE2LjM3MTYxNiAzNC4zMjA5NTE3LDE2LjMwNTE1OTYgTDM0LjMyMDk1MTcsOC40MDI1MDQ2IEMzNC4zMjA5MTc4LDguMDYyNjUwNjcgMzQuMDQ1NDIwMiw3Ljc4NzE1MzEyIDMzLjcwNTU2NjMsNy43ODcxMTkxNCBMMjEuNDkyODA5NCw3Ljc4NzExOTE0IEMyMS40NTgyNTY1LDcuNzgxMzQzNjkgMjEuNDIzMjczNiw3Ljc3ODY5NzgzIDIxLjM4ODI0MTksNy43Nzg2OTc4MyBaIE05LjcwNTUzMzY5LDkuMDE3ODkwMDUgTDIwLjc4MjQ3MTgsOS4wMTc4OTAwNSBMMjAuNzgyNDcxOCwxNS43ODcxMyBMOS43MDU1MzM2OSwxNS43ODcxMyBMOS43MDU1MzM2OSw5LjAxNzg5MDA1IFogTTIyLjAxMzI0MjcsOS4wMTc4OTAwNSBMMzMuMDkwMTgwOCw5LjAxNzg5MDA1IEwzMy4wOTAxODA4LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDkuMDE3ODkwMDUgWiBNOS43MDU1MzM2OSwxNy4wMTc5MDA5IEwyMC43ODI0NzE4LDE3LjAxNzkwMDkgTDIwLjc4MjQ3MTgsMjQuNDAyNTI2MyBMOS43MDU1MzM2OSwyNC40MDI1MjYzIEw5LjcwNTUzMzY5LDE3LjAxNzkwMDkgWiBNMjIuMDEzMjQyNywxNy4wMTc5MDA5IEwzMy4wOTAxODA4LDE3LjAxNzkwMDkgTDMzLjA5MDE4MDgsMjQuNDAyNTI2MyBMMjIuMDEzMjQyNywyNC40MDI1MjYzIEwyMi4wMTMyNDI3LDE3LjAxNzkwMDkgWiBNOS43MDU1MzM2OSwyNS42MzMyOTcyIEwyMC43ODI0NzE4LDI1LjYzMzI5NzIgTDIwLjc4MjQ3MTgsMzIuNDAyNTM3MiBMOS43MDU1MzM2OSwzMi40MDI1MzcyIEw5LjcwNTUzMzY5LDI1LjYzMzI5NzIgWiBNMjIuMDEzMjQyNywyNS42MzMyOTcyIEwzMy4wOTAxODA4LDI1LjYzMzI5NzIgTDMzLjA5MDE4MDgsMzIuNDAyNTM3MiBMMjIuMDEzMjQyNywzMi40MDI1MzcyIEwyMi4wMTMyNDI3LDI1LjYzMzI5NzIgWiIgaWQ9InRhYmxlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00Ni4wMTY5OTUzLDMyLjM4Nzc5MjYgQzQ1Ljk4NzE4MzIsMzIuMzg4NDQ2NSA0NS45NTc0NTc1LDMyLjM5MTI1ODQgNDUuOTI4MDUzLDMyLjM5NjIxMzkgTDMzLjcxODkwMTYsMzIuMzk2MjEzOSBDMzMuMzc5MDQ3NywzMi4zOTYyNDc5IDMzLjEwMzU1MDIsMzIuNjcxNzQ1NCAzMy4xMDM1MTYyLDMzLjAxMTU5OTMgTDMzLjEwMzUxNjIsNDAuOTA4MjQ0NSBDMzMuMDkyNTMyMiw0MC45NzQ3MDA5IDMzLjA5MjUzMjIsNDEuMDQyNTA5NyAzMy4xMDM1MTYyLDQxLjEwODk2NiBMMzMuMTAzNTE2Miw0OS41MjM2NDA4IEMzMy4wOTI1MzIyLDQ5LjU5MDA5NzIgMzMuMDkyNTMyMiw0OS42NTc5MDYgMzMuMTAzNTE2Miw0OS43MjQzNjI0IEwzMy4xMDM1MTYyLDU3LjYyNzAxNzQgQzMzLjEwMzU1MDIsNTcuOTY2ODcxMyAzMy4zNzkwNDc3LDU4LjI0MjM2ODkgMzMuNzE4OTAxNiw1OC4yNDI0MDI4IEw0NS45MjMyNDUsNTguMjQyNDAyOCBDNDUuOTg5NzAxNCw1OC4yNTMzODY5IDQ2LjA1NzUxMDEsNTguMjUzMzg2OSA0Ni4xMjM5NjY1LDU4LjI0MjQwMjggTDU4LjMzNDMxOTcsNTguMjQyNDAyOCBDNTguNjc0MTczNiw1OC4yNDIzNjg5IDU4Ljk0OTY3MTIsNTcuOTY2ODcxMyA1OC45NDk3MDUxLDU3LjYyNzAxNzQgTDU4Ljk0OTcwNTEsNDkuNzMwMzcyMiBDNTguOTYwNjg5Miw0OS42NjM5MTU4IDU4Ljk2MDY4OTIsNDkuNTk2MTA3MSA1OC45NDk3MDUxLDQ5LjUyOTY1MDcgTDU4Ljk0OTcwNTEsNDEuMTE0OTc1OSBDNTguOTYwNjg5Miw0MS4wNDg1MTk1IDU4Ljk2MDY4OTIsNDAuOTgwNzEwNyA1OC45NDk3MDUxLDQwLjkxNDI1NDQgTDU4Ljk0OTcwNTEsMzMuMDExNTk5MyBDNTguOTQ5NjcxMiwzMi42NzE3NDU0IDU4LjY3NDE3MzYsMzIuMzk2MjQ3OSA1OC4zMzQzMTk3LDMyLjM5NjIxMzkgTDQ2LjEyMTU2MjgsMzIuMzk2MjEzOSBDNDYuMDg3MDA5OSwzMi4zOTA0Mzg0IDQ2LjA1MjAyNywzMi4zODc3OTI2IDQ2LjAxNjk5NTMsMzIuMzg3NzkyNiBaIE0zNC4zMzQyODcxLDMzLjYyNjk4NDggTDQ1LjQxMTIyNTIsMzMuNjI2OTg0OCBMNDUuNDExMjI1Miw0MC4zOTYyMjQ4IEwzNC4zMzQyODcxLDQwLjM5NjIyNDggTDM0LjMzNDI4NzEsMzMuNjI2OTg0OCBaIE00Ni42NDE5OTYxLDMzLjYyNjk4NDggTDU3LjcxODkzNDIsMzMuNjI2OTg0OCBMNTcuNzE4OTM0Miw0MC4zOTYyMjQ4IEw0Ni42NDE5OTYxLDQwLjM5NjIyNDggTDQ2LjY0MTk5NjEsMzMuNjI2OTg0OCBaIE0zNC4zMzQyODcxLDQxLjYyNjk5NTcgTDQ1LjQxMTIyNTIsNDEuNjI2OTk1NyBMNDUuNDExMjI1Miw0OS4wMTE2MjExIEwzNC4zMzQyODcxLDQ5LjAxMTYyMTEgTDM0LjMzNDI4NzEsNDEuNjI2OTk1NyBaIE00Ni42NDE5OTYxLDQxLjYyNjk5NTcgTDU3LjcxODkzNDIsNDEuNjI2OTk1NyBMNTcuNzE4OTM0Miw0OS4wMTE2MjExIEw0Ni42NDE5OTYxLDQ5LjAxMTYyMTEgTDQ2LjY0MTk5NjEsNDEuNjI2OTk1NyBaIE0zNC4zMzQyODcxLDUwLjI0MjM5MiBMNDUuNDExMjI1Miw1MC4yNDIzOTIgTDQ1LjQxMTIyNTIsNTcuMDExNjMxOSBMMzQuMzM0Mjg3MSw1Ny4wMTE2MzE5IEwzNC4zMzQyODcxLDUwLjI0MjM5MiBaIE00Ni42NDE5OTYxLDUwLjI0MjM5MiBMNTcuNzE4OTM0Miw1MC4yNDIzOTIgTDU3LjcxODkzNDIsNTcuMDExNjMxOSBMNDYuNjQxOTk2MSw1Ny4wMTE2MzE5IEw0Ni42NDE5OTYxLDUwLjI0MjM5MiBaIiBpZD0idGFibGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

tbl_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">None</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">EXTERNAL TABLE</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody>
</table>


The validation table shows that the single test unit failed because the tables don't match (one value is different in column `c`).
