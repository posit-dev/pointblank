## Validate.conjointly()


Perform multiple row-wise validations for joint validity.


Usage

``` python
Validate.conjointly(
    *exprs, pre=None, thresholds=None, actions=None, brief=None, active=True
)
```


The [conjointly()](Validate.conjointly.md#pointblank.Validate.conjointly) validation method checks whether each row in the table passes multiple validation conditions simultaneously. This enables compound validation logic where a test unit (typically a row) must satisfy all specified conditions to pass the validation.

This method accepts multiple validation expressions as callables, which should return boolean expressions when applied to the data. You can use lambdas that incorporate Polars/Pandas/Ibis expressions (based on the target table type) or create more complex validation functions. The validation will operate over the number of test units that is equal to the number of rows in the table (determined after any `pre=` mutation has been applied).


## Parameters


`*exprs: Callable`  
Multiple validation expressions provided as callable functions. Each callable should accept a table as its single argument and return a boolean expression or Series/Column that evaluates to boolean values for each row.

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

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](Validate.md#pointblank.Validate) object or used in subsequent validation steps.


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


## Examples

For the examples here, we'll use a simple Polars DataFrame with three numeric columns (`a`, `b`, and `c`). The table is shown below:


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [5, 7, 1, 3, 9, 4],
        "b": [6, 3, 0, 5, 8, 2],
        "c": [10, 4, 8, 9, 10, 5],
    }
)

pb.preview(tbl)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_preview_tbl-_row_num_" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"></th>
<th id="pb_preview_tbl-a" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

a

<em>Int64</em>

</div></th>
<th id="pb_preview_tbl-b" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

b

<em>Int64</em>

</div></th>
<th id="pb_preview_tbl-c" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

c

<em>Int64</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">10</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">0</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">8</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">9</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">9</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">8</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">10</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">6</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
</tr>
</tbody>
</table>


Let's validate that the values in each row satisfy multiple conditions simultaneously:

1.  Column `a` should be greater than 2
2.  Column `b` should be less than 7
3.  The sum of `a` and `b` should be less than the value in column `c`

We'll use [conjointly()](Validate.conjointly.md#pointblank.Validate.conjointly) to check all these conditions together:


``` python
validation = (
    pb.Validate(data=tbl)
    .conjointly(
        lambda df: pl.col("a") > 2,
        lambda df: pl.col("b") < 7,
        lambda df: pl.col("a") + pl.col("b") < pl.col("c")
    )
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29uam9pbnRseTwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb25qb2ludGx5IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4yNDEzNzkpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01MS44NDg1OTc2LDEyIEwxNS41NzU4NzAzLDEyIEMxMy45OTg2MzI5LDEyIDEyLjcxMjIzNCwxMy4yODYzOTg5IDEyLjcxMjIzNCwxNC44NjM2MzY0IEwxMi43MTIyMzQsNTEuMTM2MzYzNiBDMTIuNzEyMjM0LDUyLjcxMzYwMTEgMTMuOTk4NjMyOSw1NCAxNS41NzU4NzAzLDU0IEw1MS44NDg1OTc2LDU0IEM1My40MjU4MzUxLDU0IDU0LjcxMjIzNCw1Mi43MTM2MDExIDU0LjcxMjIzNCw1MS4xMzYzNjM2IEw1NC43MTIyMzQsMTQuODYzNjM2NCBDNTQuNzEyMjM0LDEzLjI4NjM5ODkgNTMuNDI1ODM1MSwxMiA1MS44NDg1OTc2LDEyIFogTTM3LjA3MjIzNCw0NCBMMjAuMjcyMjM0LDQ0IEwyMC4yNzIyMzQsNDIgTDM3LjA3MjIzNCw0MiBMMzcuMDcyMjM0LDQ0IFogTTM3LjA3MjIzNCwzNCBMMjAuMjcyMjM0LDM0IEwyMC4yNzIyMzQsMzIgTDM3LjA3MjIzNCwzMiBMMzcuMDcyMjM0LDM0IFogTTM3LjA3MjIzNCwyNCBMMjAuMjcyMjM0LDI0IEwyMC4yNzIyMzQsMjIgTDM3LjA3MjIzNCwyMiBMMzcuMDcyMjM0LDI0IFogTTQ3LjkyMzMyNzksNDEuNzczNDM4IEw0NS41NzA2NzE5LDQ1Ljc3MzQzOCBDNDUuNDQyNzAyOSw0NS45OTYwOTQgNDUuMjM5MjY1LDQ2LjE0ODQzOCA0NS4wMDk1Nzc5LDQ2LjE4NzUgQzQ0Ljk3MDIwMjksNDYuMTk1MzEzIDQ0LjkyNzU0NjksNDYuMTk5MjE5IDQ0Ljg4NDg5LDQ2LjE5OTIxOSBDNDQuNzAxMTQsNDYuMTk5MjE5IDQ0LjUyMDY3MTksNDYuMTI4OTA2IDQ0LjM3MzAxNSw0NS45OTIxODggTDQyLjE4NzcwMjksNDMuOTkyMTg4IEM0MS44MjAyMDI5LDQzLjY1NjI1IDQxLjc1MTI5NjksNDMuMDI3MzQ0IDQyLjAzMzQ4NCw0Mi41ODk4NDQgQzQyLjMxNTY3MTksNDIuMTUyMzQ0IDQyLjg0Mzk1MjksNDIuMDcwMzEzIDQzLjIxMTQ1MjksNDIuNDA2MjUgTDQ0LjY5Nzg1OSw0My43Njk1MzEgTDQ2LjU0ODQ4NCw0MC42MjUgQzQ2LjgxNDI2NSw0MC4xNzE4NzUgNDcuMzM1OTg0LDQwLjA2MjUgNDcuNzE2NjA5LDQwLjM3ODkwNiBDNDguMDk3MjM0LDQwLjY5NTMxMyA0OC4xODkxMDksNDEuMzIwMzEzIDQ3LjkyMzMyNzksNDEuNzczNDM4IFogTTQ3LjkyMzMyNzksMzEuNzczNDM4IEw0NS41NzA2NzE5LDM1Ljc3MzQzOCBDNDUuNDQyNzAyOSwzNS45OTYwOTQgNDUuMjM5MjY1LDM2LjE0ODQzOCA0NS4wMDk1Nzc5LDM2LjE4NzUgQzQ0Ljk3MDIwMjksMzYuMTk1MzEzIDQ0LjkyNzU0NjksMzYuMTk5MjE5IDQ0Ljg4NDg5LDM2LjE5OTIxOSBDNDQuNzAxMTQsMzYuMTk5MjE5IDQ0LjUyMDY3MTksMzYuMTI4OTA2IDQ0LjM3MzAxNSwzNS45OTIxODggTDQyLjE4NzcwMjksMzMuOTkyMTg4IEM0MS44MjAyMDI5LDMzLjY1NjI1IDQxLjc1MTI5NjksMzMuMDI3MzQ0IDQyLjAzMzQ4NCwzMi41ODk4NDQgQzQyLjMxNTY3MTksMzIuMTUyMzQ0IDQyLjg0Mzk1MjksMzIuMDcwMzEzIDQzLjIxMTQ1MjksMzIuNDA2MjUgTDQ0LjY5Nzg1OSwzMy43Njk1MzEgTDQ2LjU0ODQ4NCwzMC42Mjg5MDYgQzQ2LjgxNDI2NSwzMC4xNzU3ODEgNDcuMzM1OTg0LDMwLjA2MjUgNDcuNzE2NjA5LDMwLjM4MjgxMyBDNDguMDk3MjM0LDMwLjY5OTIxOSA0OC4xODkxMDksMzEuMzIwMzEzIDQ3LjkyMzMyNzksMzEuNzczNDM4IFogTTQ3LjkyMzMyNzksMjEuNzczNDM4IEw0NS41NzA2NzE5LDI1Ljc3MzQzOCBDNDUuNDQyNzAyOSwyNS45OTYwOTQgNDUuMjM5MjY1LDI2LjE0ODQzOCA0NS4wMDk1Nzc5LDI2LjE4NzUgQzQ0Ljk3MDIwMjksMjYuMTk1MzEzIDQ0LjkyNzU0NjksMjYuMTk5MjE5IDQ0Ljg4NDg5LDI2LjE5OTIxOSBDNDQuNzAxMTQsMjYuMTk5MjE5IDQ0LjUyMDY3MTksMjYuMTI4OTA2IDQ0LjM3MzAxNSwyNS45OTIxODggTDQyLjE4NzcwMjksMjMuOTkyMTg4IEM0MS44MjAyMDI5LDIzLjY1NjI1IDQxLjc1MTI5NjksMjMuMDI3MzQ0IDQyLjAzMzQ4NCwyMi41ODk4NDQgQzQyLjMxNTY3MTksMjIuMTUyMzQ0IDQyLjg0Mzk1MjksMjIuMDcwMzEzIDQzLjIxMTQ1MjksMjIuNDA2MjUgTDQ0LjY5Nzg1OSwyMy43Njk1MzEgTDQ2LjU0ODQ4NCwyMC42MjUgQzQ2LjgxNDI2NSwyMC4xNzE4NzUgNDcuMzM1OTg0LDIwLjA2MjUgNDcuNzE2NjA5LDIwLjM3ODkwNiBDNDguMDk3MjM0LDIwLjY5OTIxOSA0OC4xODkxMDksMjEuMzIwMzEzIDQ3LjkyMzMyNzksMjEuNzczNDM4IFoiIGlkPSJjb25qb2ludCIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

conjointly()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">COLUMN EXPR</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">6</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
0.17</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
0.83</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


The validation table shows that not all rows satisfy all three conditions together. For a row to pass the conjoint validation, all three conditions must be true for that row.

We can also use preprocessing to filter the data before applying the conjoint validation:


``` python
# Define preprocessing function for serialization compatibility
def filter_by_c_gt_5(df):
    return df.filter(pl.col("c") > 5)

validation = (
    pb.Validate(data=tbl)
    .conjointly(
        lambda df: pl.col("a") > 2,
        lambda df: pl.col("b") < 7,
        lambda df: pl.col("a") + pl.col("b") < pl.col("c"),
        pre=filter_by_c_gt_5
    )
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29uam9pbnRseTwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb25qb2ludGx5IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4yNDEzNzkpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01MS44NDg1OTc2LDEyIEwxNS41NzU4NzAzLDEyIEMxMy45OTg2MzI5LDEyIDEyLjcxMjIzNCwxMy4yODYzOTg5IDEyLjcxMjIzNCwxNC44NjM2MzY0IEwxMi43MTIyMzQsNTEuMTM2MzYzNiBDMTIuNzEyMjM0LDUyLjcxMzYwMTEgMTMuOTk4NjMyOSw1NCAxNS41NzU4NzAzLDU0IEw1MS44NDg1OTc2LDU0IEM1My40MjU4MzUxLDU0IDU0LjcxMjIzNCw1Mi43MTM2MDExIDU0LjcxMjIzNCw1MS4xMzYzNjM2IEw1NC43MTIyMzQsMTQuODYzNjM2NCBDNTQuNzEyMjM0LDEzLjI4NjM5ODkgNTMuNDI1ODM1MSwxMiA1MS44NDg1OTc2LDEyIFogTTM3LjA3MjIzNCw0NCBMMjAuMjcyMjM0LDQ0IEwyMC4yNzIyMzQsNDIgTDM3LjA3MjIzNCw0MiBMMzcuMDcyMjM0LDQ0IFogTTM3LjA3MjIzNCwzNCBMMjAuMjcyMjM0LDM0IEwyMC4yNzIyMzQsMzIgTDM3LjA3MjIzNCwzMiBMMzcuMDcyMjM0LDM0IFogTTM3LjA3MjIzNCwyNCBMMjAuMjcyMjM0LDI0IEwyMC4yNzIyMzQsMjIgTDM3LjA3MjIzNCwyMiBMMzcuMDcyMjM0LDI0IFogTTQ3LjkyMzMyNzksNDEuNzczNDM4IEw0NS41NzA2NzE5LDQ1Ljc3MzQzOCBDNDUuNDQyNzAyOSw0NS45OTYwOTQgNDUuMjM5MjY1LDQ2LjE0ODQzOCA0NS4wMDk1Nzc5LDQ2LjE4NzUgQzQ0Ljk3MDIwMjksNDYuMTk1MzEzIDQ0LjkyNzU0NjksNDYuMTk5MjE5IDQ0Ljg4NDg5LDQ2LjE5OTIxOSBDNDQuNzAxMTQsNDYuMTk5MjE5IDQ0LjUyMDY3MTksNDYuMTI4OTA2IDQ0LjM3MzAxNSw0NS45OTIxODggTDQyLjE4NzcwMjksNDMuOTkyMTg4IEM0MS44MjAyMDI5LDQzLjY1NjI1IDQxLjc1MTI5NjksNDMuMDI3MzQ0IDQyLjAzMzQ4NCw0Mi41ODk4NDQgQzQyLjMxNTY3MTksNDIuMTUyMzQ0IDQyLjg0Mzk1MjksNDIuMDcwMzEzIDQzLjIxMTQ1MjksNDIuNDA2MjUgTDQ0LjY5Nzg1OSw0My43Njk1MzEgTDQ2LjU0ODQ4NCw0MC42MjUgQzQ2LjgxNDI2NSw0MC4xNzE4NzUgNDcuMzM1OTg0LDQwLjA2MjUgNDcuNzE2NjA5LDQwLjM3ODkwNiBDNDguMDk3MjM0LDQwLjY5NTMxMyA0OC4xODkxMDksNDEuMzIwMzEzIDQ3LjkyMzMyNzksNDEuNzczNDM4IFogTTQ3LjkyMzMyNzksMzEuNzczNDM4IEw0NS41NzA2NzE5LDM1Ljc3MzQzOCBDNDUuNDQyNzAyOSwzNS45OTYwOTQgNDUuMjM5MjY1LDM2LjE0ODQzOCA0NS4wMDk1Nzc5LDM2LjE4NzUgQzQ0Ljk3MDIwMjksMzYuMTk1MzEzIDQ0LjkyNzU0NjksMzYuMTk5MjE5IDQ0Ljg4NDg5LDM2LjE5OTIxOSBDNDQuNzAxMTQsMzYuMTk5MjE5IDQ0LjUyMDY3MTksMzYuMTI4OTA2IDQ0LjM3MzAxNSwzNS45OTIxODggTDQyLjE4NzcwMjksMzMuOTkyMTg4IEM0MS44MjAyMDI5LDMzLjY1NjI1IDQxLjc1MTI5NjksMzMuMDI3MzQ0IDQyLjAzMzQ4NCwzMi41ODk4NDQgQzQyLjMxNTY3MTksMzIuMTUyMzQ0IDQyLjg0Mzk1MjksMzIuMDcwMzEzIDQzLjIxMTQ1MjksMzIuNDA2MjUgTDQ0LjY5Nzg1OSwzMy43Njk1MzEgTDQ2LjU0ODQ4NCwzMC42Mjg5MDYgQzQ2LjgxNDI2NSwzMC4xNzU3ODEgNDcuMzM1OTg0LDMwLjA2MjUgNDcuNzE2NjA5LDMwLjM4MjgxMyBDNDguMDk3MjM0LDMwLjY5OTIxOSA0OC4xODkxMDksMzEuMzIwMzEzIDQ3LjkyMzMyNzksMzEuNzczNDM4IFogTTQ3LjkyMzMyNzksMjEuNzczNDM4IEw0NS41NzA2NzE5LDI1Ljc3MzQzOCBDNDUuNDQyNzAyOSwyNS45OTYwOTQgNDUuMjM5MjY1LDI2LjE0ODQzOCA0NS4wMDk1Nzc5LDI2LjE4NzUgQzQ0Ljk3MDIwMjksMjYuMTk1MzEzIDQ0LjkyNzU0NjksMjYuMTk5MjE5IDQ0Ljg4NDg5LDI2LjE5OTIxOSBDNDQuNzAxMTQsMjYuMTk5MjE5IDQ0LjUyMDY3MTksMjYuMTI4OTA2IDQ0LjM3MzAxNSwyNS45OTIxODggTDQyLjE4NzcwMjksMjMuOTkyMTg4IEM0MS44MjAyMDI5LDIzLjY1NjI1IDQxLjc1MTI5NjksMjMuMDI3MzQ0IDQyLjAzMzQ4NCwyMi41ODk4NDQgQzQyLjMxNTY3MTksMjIuMTUyMzQ0IDQyLjg0Mzk1MjksMjIuMDcwMzEzIDQzLjIxMTQ1MjksMjIuNDA2MjUgTDQ0LjY5Nzg1OSwyMy43Njk1MzEgTDQ2LjU0ODQ4NCwyMC42MjUgQzQ2LjgxNDI2NSwyMC4xNzE4NzUgNDcuMzM1OTg0LDIwLjA2MjUgNDcuNzE2NjA5LDIwLjM3ODkwNiBDNDguMDk3MjM0LDIwLjY5OTIxOSA0OC4xODkxMDksMjEuMzIwMzEzIDQ3LjkyMzMyNzksMjEuNzczNDM4IFoiIGlkPSJjb25qb2ludCIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

conjointly()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">COLUMN EXPR</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJtb2RpZmllZCIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9Im1vZGlmaWVkIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC41NzAxNDcpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgeD0iMC4xMjUxMzI1MDYiIHk9IjAiIHdpZHRoPSIyMy43NDk3MzUiIGhlaWdodD0iMjMuNzg5NDczNyIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzAwMDAwMCIgeD0iMTUuMjIxNTYyNiIgeT0iMTQuODAyNTg4NSIgd2lkdGg9IjYuMzQwODk4MjEiIGhlaWdodD0iNi4zNDA4OTgyMSIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzlBN0NCNCIgeD0iMTYuMTI2NDMwOSIgeT0iMTUuNzA3NDU2OCIgd2lkdGg9IjQuNTMxMTYxNTgiIGhlaWdodD0iNC41MzExNjE1OCIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">4</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
0.25</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">3<br />
0.75</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(pre_applied)</span> Precondition applied: table dimensions <span style="font-family: monospace;">[6 rows, 3 columns]</span> → <span style="font-family: monospace;">[<strong>4</strong> rows, 3 columns]</span>.</p></td>
</tr>
</tfoot>

</table>


This allows for more complex validation scenarios where the data is first prepared and then validated against multiple conditions simultaneously.

Or, you can use the backend-agnostic column expression helper <a href="expr_col.html#pointblank.expr_col" class="gdls-link"><code>expr_col()</code></a> to write expressions that work across different table backends:


``` python
tbl = pl.DataFrame(
    {
        "a": [5, 7, 1, 3, 9, 4],
        "b": [6, 3, 0, 5, 8, 2],
        "c": [10, 4, 8, 9, 10, 5],
    }
)

# Using backend-agnostic syntax with expr_col()
validation = (
    pb.Validate(data=tbl)
    .conjointly(
        lambda df: pb.expr_col("a") > 2,
        lambda df: pb.expr_col("b") < 7,
        lambda df: pb.expr_col("a") + pb.expr_col("b") < pb.expr_col("c")
    )
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29uam9pbnRseTwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb25qb2ludGx5IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4yNDEzNzkpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01MS44NDg1OTc2LDEyIEwxNS41NzU4NzAzLDEyIEMxMy45OTg2MzI5LDEyIDEyLjcxMjIzNCwxMy4yODYzOTg5IDEyLjcxMjIzNCwxNC44NjM2MzY0IEwxMi43MTIyMzQsNTEuMTM2MzYzNiBDMTIuNzEyMjM0LDUyLjcxMzYwMTEgMTMuOTk4NjMyOSw1NCAxNS41NzU4NzAzLDU0IEw1MS44NDg1OTc2LDU0IEM1My40MjU4MzUxLDU0IDU0LjcxMjIzNCw1Mi43MTM2MDExIDU0LjcxMjIzNCw1MS4xMzYzNjM2IEw1NC43MTIyMzQsMTQuODYzNjM2NCBDNTQuNzEyMjM0LDEzLjI4NjM5ODkgNTMuNDI1ODM1MSwxMiA1MS44NDg1OTc2LDEyIFogTTM3LjA3MjIzNCw0NCBMMjAuMjcyMjM0LDQ0IEwyMC4yNzIyMzQsNDIgTDM3LjA3MjIzNCw0MiBMMzcuMDcyMjM0LDQ0IFogTTM3LjA3MjIzNCwzNCBMMjAuMjcyMjM0LDM0IEwyMC4yNzIyMzQsMzIgTDM3LjA3MjIzNCwzMiBMMzcuMDcyMjM0LDM0IFogTTM3LjA3MjIzNCwyNCBMMjAuMjcyMjM0LDI0IEwyMC4yNzIyMzQsMjIgTDM3LjA3MjIzNCwyMiBMMzcuMDcyMjM0LDI0IFogTTQ3LjkyMzMyNzksNDEuNzczNDM4IEw0NS41NzA2NzE5LDQ1Ljc3MzQzOCBDNDUuNDQyNzAyOSw0NS45OTYwOTQgNDUuMjM5MjY1LDQ2LjE0ODQzOCA0NS4wMDk1Nzc5LDQ2LjE4NzUgQzQ0Ljk3MDIwMjksNDYuMTk1MzEzIDQ0LjkyNzU0NjksNDYuMTk5MjE5IDQ0Ljg4NDg5LDQ2LjE5OTIxOSBDNDQuNzAxMTQsNDYuMTk5MjE5IDQ0LjUyMDY3MTksNDYuMTI4OTA2IDQ0LjM3MzAxNSw0NS45OTIxODggTDQyLjE4NzcwMjksNDMuOTkyMTg4IEM0MS44MjAyMDI5LDQzLjY1NjI1IDQxLjc1MTI5NjksNDMuMDI3MzQ0IDQyLjAzMzQ4NCw0Mi41ODk4NDQgQzQyLjMxNTY3MTksNDIuMTUyMzQ0IDQyLjg0Mzk1MjksNDIuMDcwMzEzIDQzLjIxMTQ1MjksNDIuNDA2MjUgTDQ0LjY5Nzg1OSw0My43Njk1MzEgTDQ2LjU0ODQ4NCw0MC42MjUgQzQ2LjgxNDI2NSw0MC4xNzE4NzUgNDcuMzM1OTg0LDQwLjA2MjUgNDcuNzE2NjA5LDQwLjM3ODkwNiBDNDguMDk3MjM0LDQwLjY5NTMxMyA0OC4xODkxMDksNDEuMzIwMzEzIDQ3LjkyMzMyNzksNDEuNzczNDM4IFogTTQ3LjkyMzMyNzksMzEuNzczNDM4IEw0NS41NzA2NzE5LDM1Ljc3MzQzOCBDNDUuNDQyNzAyOSwzNS45OTYwOTQgNDUuMjM5MjY1LDM2LjE0ODQzOCA0NS4wMDk1Nzc5LDM2LjE4NzUgQzQ0Ljk3MDIwMjksMzYuMTk1MzEzIDQ0LjkyNzU0NjksMzYuMTk5MjE5IDQ0Ljg4NDg5LDM2LjE5OTIxOSBDNDQuNzAxMTQsMzYuMTk5MjE5IDQ0LjUyMDY3MTksMzYuMTI4OTA2IDQ0LjM3MzAxNSwzNS45OTIxODggTDQyLjE4NzcwMjksMzMuOTkyMTg4IEM0MS44MjAyMDI5LDMzLjY1NjI1IDQxLjc1MTI5NjksMzMuMDI3MzQ0IDQyLjAzMzQ4NCwzMi41ODk4NDQgQzQyLjMxNTY3MTksMzIuMTUyMzQ0IDQyLjg0Mzk1MjksMzIuMDcwMzEzIDQzLjIxMTQ1MjksMzIuNDA2MjUgTDQ0LjY5Nzg1OSwzMy43Njk1MzEgTDQ2LjU0ODQ4NCwzMC42Mjg5MDYgQzQ2LjgxNDI2NSwzMC4xNzU3ODEgNDcuMzM1OTg0LDMwLjA2MjUgNDcuNzE2NjA5LDMwLjM4MjgxMyBDNDguMDk3MjM0LDMwLjY5OTIxOSA0OC4xODkxMDksMzEuMzIwMzEzIDQ3LjkyMzMyNzksMzEuNzczNDM4IFogTTQ3LjkyMzMyNzksMjEuNzczNDM4IEw0NS41NzA2NzE5LDI1Ljc3MzQzOCBDNDUuNDQyNzAyOSwyNS45OTYwOTQgNDUuMjM5MjY1LDI2LjE0ODQzOCA0NS4wMDk1Nzc5LDI2LjE4NzUgQzQ0Ljk3MDIwMjksMjYuMTk1MzEzIDQ0LjkyNzU0NjksMjYuMTk5MjE5IDQ0Ljg4NDg5LDI2LjE5OTIxOSBDNDQuNzAxMTQsMjYuMTk5MjE5IDQ0LjUyMDY3MTksMjYuMTI4OTA2IDQ0LjM3MzAxNSwyNS45OTIxODggTDQyLjE4NzcwMjksMjMuOTkyMTg4IEM0MS44MjAyMDI5LDIzLjY1NjI1IDQxLjc1MTI5NjksMjMuMDI3MzQ0IDQyLjAzMzQ4NCwyMi41ODk4NDQgQzQyLjMxNTY3MTksMjIuMTUyMzQ0IDQyLjg0Mzk1MjksMjIuMDcwMzEzIDQzLjIxMTQ1MjksMjIuNDA2MjUgTDQ0LjY5Nzg1OSwyMy43Njk1MzEgTDQ2LjU0ODQ4NCwyMC42MjUgQzQ2LjgxNDI2NSwyMC4xNzE4NzUgNDcuMzM1OTg0LDIwLjA2MjUgNDcuNzE2NjA5LDIwLjM3ODkwNiBDNDguMDk3MjM0LDIwLjY5OTIxOSA0OC4xODkxMDksMjEuMzIwMzEzIDQ3LjkyMzMyNzksMjEuNzczNDM4IFoiIGlkPSJjb25qb2ludCIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

conjointly()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">COLUMN EXPR</td>
MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">6</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
0.17</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
0.83</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


Using <a href="expr_col.html#pointblank.expr_col" class="gdls-link"><code>expr_col()</code></a> allows your validation code to work consistently across Pandas, Polars, and Ibis table backends without changes, making your validation pipelines more portable.


#### See Also

[Look](Look.md), [information](information.md)
