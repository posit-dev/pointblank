## expr_col()


Create a column expression for use in [conjointly()](Validate.conjointly.md#pointblank.Validate.conjointly) validation.


Usage

``` python
expr_col(column_name)
```


This function returns a ColumnExpression object that supports operations like `>`, `<`, `+`, etc. for use in <a href="Validate.conjointly.html#pointblank.Validate.conjointly" class="gdls-link"><code>conjointly()</code></a> validation expressions.


## Parameters


`column_name: str`  
The name of the column to reference.


## Returns


`ColumnExpression`  
A column expression that can be used in comparisons and operations.


## Examples

Let's say we have a table with three columns: `a`, `b`, and `c`. We want to validate that:

- The values in column `a` are greater than `2`.
- The values in column `b` are less than `7`.
- The sum of columns `a` and `b` is less than the values in column `c`.

We can use the [expr_col()](expr_col.md#pointblank.expr_col) function to create a column expression for each of these conditions.


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

# Using expr_col() to create backend-agnostic validation expressions
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


The above code creates a validation object that checks the specified conditions using the [expr_col()](expr_col.md#pointblank.expr_col) function. The resulting validation table will show whether each condition was satisfied for each row in the table.


#### See Also

[The](The.md), [function](function.md)
