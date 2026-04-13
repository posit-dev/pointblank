## Validate.col_exists()


Validate whether one or more columns exist in the table.


Usage

``` python
Validate.col_exists(
    columns, thresholds=None, actions=None, brief=None, active=True
)
```


The [col_exists()](Validate.col_exists.md#pointblank.Validate.col_exists) method checks whether one or more columns exist in the target table. The only requirement is specification of the column names. Each validation step or expectation will operate over a single test unit, which is whether the column exists or not.


## Parameters


`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a> with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect. Look at the *Thresholds* section for information on how to set threshold levels.

`actions: Actions | None = None`  
Optional actions to take when the validation step(s) meets or exceeds any set threshold levels. If provided, the <a href="Actions.html#pointblank.Actions" class="gdls-link"><code>Actions</code></a> class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = ``True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like <a href="has_columns.html#pointblank.has_columns" class="gdls-link"><code>has_columns()</code></a> and <a href="has_rows.html#pointblank.has_rows" class="gdls-link"><code>has_rows()</code></a> can be used here to conditionally activate a step based on properties of the target table.


## Returns


`Validate`  
The [Validate](Validate.md#pointblank.Validate) object with the added validation step.


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

For the examples here, we'll use a simple Polars DataFrame with a string columns (`a`) and a numeric column (`b`). The table is shown below:


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": ["apple", "banana", "cherry", "date"],
        "b": [1, 6, 3, 5],
    }
)

pb.preview(tbl)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_preview_tbl-_row_num_" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"></th>
<th id="pb_preview_tbl-a" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

a

<em>String</em>

</div></th>
<th id="pb_preview_tbl-b" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

b

<em>Int64</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">1</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">apple</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">banana</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">cherry</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">4</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">date</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
</tr>
</tbody>
</table>


Let's validate that the columns `a` and `b` actually exist in the table. We'll determine if this validation had any failing test units (each validation will have a single test unit).


``` python
validation = (
    pb.Validate(data=tbl)
    .col_exists(columns=["a", "b"])
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
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
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
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


Printing the `validation` object shows the validation table in an HTML viewing environment. The validation table shows two entries (one check per column) generated by the [col_exists()](Validate.col_exists.md#pointblank.Validate.col_exists) validation step. Both steps passed since both columns provided in `columns=` are present in the table.

Now, let's check for the existence of a different set of columns.


``` python
validation = (
    pb.Validate(data=tbl)
    .col_exists(columns=["b", "c"])
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
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
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
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


The validation table reports one passing validation step (the check for column `b`) and one failing validation step (the check for column `c`, which doesn't exist).
