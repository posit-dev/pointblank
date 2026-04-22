## Validate.col_pct_null()


Validate whether a column has a specific percentage of Null values.


Usage

``` python
Validate.col_pct_null(
    columns, p, tol=0, thresholds=None, actions=None, brief=None, active=True
)
```


The [col_pct_null()](Validate.col_pct_null.md#pointblank.Validate.col_pct_null) validation method checks whether the percentage of Null values in a column matches a specified percentage `p=` (within an optional tolerance `tol=`). This validation operates at the column level, generating a single validation step per column that passes or fails based on whether the actual percentage of Null values falls within the acceptable range defined by `p ± tol`.


## Parameters


`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a> with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`p: float`  
The expected percentage of Null values in the column, expressed as a decimal between `0.0` and `1.0`. For example, `p=0.5` means 50% of values should be Null.

`tol: Tolerance = ``0`  
The tolerance allowed when comparing the actual percentage of Null values to the expected percentage `p=`. The validation passes if the actual percentage falls within the range `[p - tol, p + tol]`. Default is `0`, meaning an exact match is required. See the *Tolerance* section for details on all supported formats (absolute, relative, symmetric, and asymmetric bounds).

`thresholds: int | float | None | bool | tuple | dict | Thresholds = None`  
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


## Tolerance

The `tol=` parameter accepts several different formats to specify the acceptable deviation from the expected percentage `p=`. The tolerance can be expressed as:

1.  *single integer* (absolute tolerance): the exact number of test units that can deviate. For example, `tol=2` means the actual count can differ from the expected count by up to 2 units in either direction.

2.  *single float between 0 and 1* (relative tolerance): a proportion of the expected count. For example, if the expected count is 50 and `tol=0.1`, the acceptable range is 45 to 55 (50 ± 10% of 50 = 50 ± 5).

3.  *tuple of two integers* (absolute bounds): explicitly specify the lower and upper bounds as absolute deviations. For example, `tol=(1, 3)` means the actual count can be 1 unit below or 3 units above the expected count.

4.  *tuple of two floats between 0 and 1* (relative bounds): explicitly specify the lower and upper bounds as proportional deviations. For example, `tol=(0.05, 0.15)` means the lower bound is 5% below and the upper bound is 15% above the expected count.

When using a single value (integer or float), the tolerance is applied symmetrically in both directions. When using a tuple, you can specify asymmetric tolerances where the lower and upper bounds differ.


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

For the examples here, we'll use a simple Polars DataFrame with three columns (`a`, `b`, and `c`) that have different percentages of Null values. The table is shown below:


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [1, 2, 3, 4, 5, 6, 7, 8],
        "b": [1, None, 3, None, 5, None, 7, None],
        "c": [None, None, None, None, None, None, 1, 2],
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
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">6</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">8</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">8</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
</tr>
</tbody>
</table>


Let's validate that column `a` has 0% Null values (i.e., no Null values at all).


``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="a", p=0.0)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5wY3RfbnVsbDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19udWxsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxLjAwMDAwMCwgMS41ODE3MTcpIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICA8cGF0aCBkPSJNNTUsMCBDNTcuNDg1MjgxMywwIDU5LjczNTI4MTMsMS4wMDczNTkzMSA2MS4zNjM5NjEsMi42MzYwMzg5NyBDNjIuOTkyNjQwNyw0LjI2NDcxODYzIDY0LDYuNTE0NzE4NjMgNjQsOSBMNjQsOSBMNjQsNjQgTDksNjQgQzYuNTE0NzE4NjIsNjQgNC4yNjQ3MTg2Miw2Mi45OTI2NDA3IDIuNjM2MDM4OTcsNjEuMzYzOTYxIEMxLjAwNzM1OTMxLDU5LjczNTI4MTQgMCw1Ny40ODUyODE0IDAsNTUgTDAsNTUgTDAsOSBDMCw2LjUxNDcxODYzIDEuMDA3MzU5MzEsNC4yNjQ3MTg2MyAyLjYzNjAzODk3LDIuNjM2MDM4OTcgQzQuMjY0NzE4NjIsMS4wMDczNTkzMSA2LjUxNDcxODYyLDAgOSwwIEw5LDAgTDU1LDAgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NS40MTQ2MjI4LDQxLjUgQzQzLjI4NzcyNTQsNDEuNSA0MS42MjUyMjU0LDQwLjc1MzU3MTUgNDAuNzQ1MjAzLDM5LjQ1MzY4MzIgQzM5Ljg2NTE4MDUsNDAuNzUzNTcxNSAzOC4yMDI2ODA1LDQxLjUgMzYuMDc1NzgzNSw0MS41IEMzMy4yNjM5NTI5LDQxLjUgMjguODAyMzQ1OCwzOS4wNjc3NDU0IDI4LjgwMjM0NTgsMzIuNTQyODU3MiBDMjguODAyMzQ1OCwyOS41MjMyMTQzIDI5Ljc1NDQ2NjIsMjIuNSAzOC41NzM3NzQ0LDIyLjUgQzM4Ljg3Mjc2OTgsMjIuNSAzOS4xMTY2MzE1LDIyLjc0Mzg2MTcgMzkuMTE2NjMxNSwyMy4wNDI4NTcxIEwzOS4xMTY2MzE1LDI0LjEyODU3MTQgQzM5LjExNjYzMTUsMjQuNDI3NTY2OCAzOC44NzI3Njk4LDI0LjY3MTQyODYgMzguNTczNzc0NCwyNC42NzE0Mjg2IEMzNS45NTkxNTM5LDI0LjY3MTQyODYgMzMuMTQ1MjAyOSwyNy4xMzU0OTEzIDMzLjE0NTIwMjksMzIuNTQyODU3MiBDMzMuMTQ1MjAyOSwzNS4wNTE0NTEgMzQuMDEyNTAxOSwzOC43ODU3MTQzIDM2LjQwMjM0NTgsMzguNzg1NzE0MyBDMzguMDk0NTMzNiwzOC43ODU3MTQzIDM5LjA2MTQ5NzgsMzcuMDIzNTQ5IDM5LjY1MzEyNjksMzUuNTc3MzQzOSBDMzguNTc1ODk0OCwzMy43NjY0MDYxIDM4LjAzMDkxNzIsMzIuMDY5OTc3NSAzOC4wMzA5MTcyLDMwLjUzNDcwOTcgQzM4LjAzMDkxNzIsMjcuMDA4MjU5MiAzOS40OTE5NjYyLDI1Ljc1NzE0MjkgNDAuNzQ1MjAzLDI1Ljc1NzE0MjkgQzQxLjk5ODQzOTcsMjUuNzU3MTQyOSA0My40NTk0ODg2LDI3LjAwODI1OTIgNDMuNDU5NDg4NiwzMC41MzQ3MDk3IEM0My40NTk0ODg2LDMyLjA5MTE4MzIgNDIuOTI5MzU1LDMzLjc0NTIwMSA0MS44MzkzOTk0LDM1LjU3NzM0MzkgQzQyLjcxOTQyMTgsMzcuNzA2MzYxNyA0My44MTE0OTc5LDM4Ljc4NTcxNDMgNDUuMDg4MDYwMSwzOC43ODU3MTQzIEM0Ny40Nzc5MDQsMzguNzg1NzE0MyA0OC4zNDUyMDMsMzUuMDUxNDUxIDQ4LjM0NTIwMywzMi41NDI4NTcyIEM0OC4zNDUyMDMsMjcuMTM1NDkxMyA0NS41MzEyNTE5LDI0LjY3MTQyODYgNDIuOTE2NjMxNSwyNC42NzE0Mjg2IEM0Mi42MTU1MTU3LDI0LjY3MTQyODYgNDIuMzczNzc0NCwyNC40Mjc1NjY4IDQyLjM3Mzc3NDQsMjQuMTI4NTcxNCBMNDIuMzczNzc0NCwyMy4wNDI4NTcxIEM0Mi4zNzM3NzQ0LDIyLjc0Mzg2MTcgNDIuNjE1NTE1NywyMi41IDQyLjkxNjYzMTUsMjIuNSBDNTEuNzM1OTM5NywyMi41IDUyLjY4ODA2MDEsMjkuNTIzMjE0MyA1Mi42ODgwNjAxLDMyLjU0Mjg1NzIgQzUyLjY4ODA2MDEsMzkuMDY3NzQ1NCA0OC4yMjY0NTI5LDQxLjUgNDUuNDE0NjIyOCw0MS41IFoiIGlkPSJvbWVnYSIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICAgICAgPGcgaWQ9InBlcmNlbnQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDExLjI2ODUwOCwgMjMuODU0MzczKSIgZmlsbD0iIzAwMDAwMCI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS44OTkyMDU1MywxNy4yMDM3OTg4IEMxLjc4NzIwNTUzLDE3LjEyMzc5ODggMS42NjQsMTcuMDI5NjI3NCAxLjU4NCwxNi44OTM2Mjc0IEMxLjUwNCwxNi43NTc2Mjc0IDEuNDY0LDE2LjYwOTYyNzQgMS40NjQsMTYuNDQ5NjI3NCBDMS40NjQsMTYuMjQxNjI3NCAxLjUzNiwxNi4wMzM2Mjc0IDEuNjgsMTUuODI1NjI3NCBMMTIuMjQsMC40ODk2Mjc0MzQgQzEyLjQzMiwwLjE4NTYyNzQzNCAxMi41OTQyNjYyLDAgMTIuOTQ2MjY2MiwwIEMxMy4xNTQyNjYyLDAgMTMuNDcyLDAuMDg5NjI3NDM0IDEzLjY4LDAuMjAxNjI3NDM0IEMxNC4wNDgsMC40MjU2Mjc0MzQgMTQuMjMyLDAuNjgxNjI3NDM0IDE0LjIzMiwwLjk2OTYyNzQzNCBDMTQuMjMyLDEuMTQ1NjI3NDMgMTQuMTYsMS4zNDU2Mjc0MyAxNC4wMTYsMS41Njk2Mjc0MyBMMy40MzIsMTYuOTc3NjI3NCBDMy4xNzYsMTcuMjgxNjI3NCAyLjg4LDE3LjQzMzYyNzQgMi41NDQsMTcuNDMzNjI3NCBDMi4zMzYsMTcuNDMzNjI3NCAyLjEzOTIwNTUzLDE3LjM0Nzc5ODggMS44OTkyMDU1MywxNy4yMDM3OTg4IFogTTMuODY0LDcuNDczNjI3NDMgQzMuMTc2LDcuNDczNjI3NDMgMi41MzYsNy4zMDU2Mjc0MyAxLjk0NCw2Ljk2OTYyNzQzIEMxLjM1Miw2LjYzMzYyNzQzIDAuODgsNi4xODE2Mjc0MyAwLjUyOCw1LjYxMzYyNzQzIEMwLjE3Niw1LjA0NTYyNzQzIDAsNC40MjU2Mjc0MyAwLDMuNzUzNjI3NDMgQzAsMy4wODE2Mjc0MyAwLjE3MiwyLjQ2MTYyNzQzIDAuNTE2LDEuODkzNjI3NDMgQzAuODYsMS4zMjU2Mjc0MyAxLjMyOCwwLjg3NzYyNzQzNCAxLjkyLDAuNTQ5NjI3NDM0IEMyLjUxMiwwLjIyMTYyNzQzNCAzLjE2LDAuMDU3NjI3NDM0IDMuODY0LDAuMDU3NjI3NDM0IEM0LjU2OCwwLjA1NzYyNzQzNCA1LjIxNiwwLjIyMTYyNzQzNCA1LjgwOCwwLjU0OTYyNzQzNCBDNi40LDAuODc3NjI3NDM0IDYuODY0LDEuMzI1NjI3NDMgNy4yLDEuODkzNjI3NDMgQzcuNTM2LDIuNDYxNjI3NDMgNy43MDQsMy4wODE2Mjc0MyA3LjcwNCwzLjc1MzYyNzQzIEM3LjcwNCw0LjQyNTYyNzQzIDcuNTMyLDUuMDQ1NjI3NDMgNy4xODgsNS42MTM2Mjc0MyBDNi44NDQsNi4xODE2Mjc0MyA2LjM4LDYuNjMzNjI3NDMgNS43OTYsNi45Njk2Mjc0MyBDNS4yMTIsNy4zMDU2Mjc0MyA0LjU2OCw3LjQ3MzYyNzQzIDMuODY0LDcuNDczNjI3NDMgWiBNMy44NjQsNS42OTc2Mjc0MyBDNC40MDgsNS42OTc2Mjc0MyA0Ljg1Miw1LjUxMzYyNzQzIDUuMTk2LDUuMTQ1NjI3NDMgQzUuNTQsNC43Nzc2Mjc0MyA1LjcxMiw0LjMxMzYyNzQzIDUuNzEyLDMuNzUzNjI3NDMgQzUuNzEyLDMuMTc3NjI3NDMgNS41NCwyLjcwNTYyNzQzIDUuMTk2LDIuMzM3NjI3NDMgQzQuODUyLDEuOTY5NjI3NDMgNC40MDgsMS43ODU2Mjc0MyAzLjg2NCwxLjc4NTYyNzQzIEMzLjMwNCwxLjc4NTYyNzQzIDIuODQ4LDEuOTY5NjI3NDMgMi40OTYsMi4zMzc2Mjc0MyBDMi4xNDQsMi43MDU2Mjc0MyAxLjk2OCwzLjE3NzYyNzQzIDEuOTY4LDMuNzUzNjI3NDMgQzEuOTY4LDQuMzEzNjI3NDMgMi4xNDQsNC43Nzc2Mjc0MyAyLjQ5Niw1LjE0NTYyNzQzIEMyLjg0OCw1LjUxMzYyNzQzIDMuMzA0LDUuNjk3NjI3NDMgMy44NjQsNS42OTc2Mjc0MyBaIE0xMS45NTIsMTcuMzg1NjI3NCBDMTEuMjQ4LDE3LjM4NTYyNzQgMTAuNiwxNy4yMTc2Mjc0IDEwLjAwOCwxNi44ODE2Mjc0IEM5LjQxNiwxNi41NDU2Mjc0IDguOTQ4LDE2LjA5MzYyNzQgOC42MDQsMTUuNTI1NjI3NCBDOC4yNiwxNC45NTc2Mjc0IDguMDg4LDE0LjMzNzYyNzQgOC4wODgsMTMuNjY1NjI3NCBDOC4wODgsMTIuOTkzNjI3NCA4LjI2LDEyLjM3MzYyNzQgOC42MDQsMTEuODA1NjI3NCBDOC45NDgsMTEuMjM3NjI3NCA5LjQxNiwxMC43ODk2Mjc0IDEwLjAwOCwxMC40NjE2Mjc0IEMxMC42LDEwLjEzMzYyNzQgMTEuMjQ4LDkuOTY5NjI3NDMgMTEuOTUyLDkuOTY5NjI3NDMgQzEyLjY1Niw5Ljk2OTYyNzQzIDEzLjMsMTAuMTMzNjI3NCAxMy44ODQsMTAuNDYxNjI3NCBDMTQuNDY4LDEwLjc4OTYyNzQgMTQuOTI4LDExLjIzNzYyNzQgMTUuMjY0LDExLjgwNTYyNzQgQzE1LjYsMTIuMzczNjI3NCAxNS43NjgsMTIuOTkzNjI3NCAxNS43NjgsMTMuNjY1NjI3NCBDMTUuNzY4LDE0LjMzNzYyNzQgMTUuNTk2LDE0Ljk1NzYyNzQgMTUuMjUyLDE1LjUyNTYyNzQgQzE0LjkwOCwxNi4wOTM2Mjc0IDE0LjQ0NCwxNi41NDU2Mjc0IDEzLjg2LDE2Ljg4MTYyNzQgQzEzLjI3NiwxNy4yMTc2Mjc0IDEyLjY0LDE3LjM4NTYyNzQgMTEuOTUyLDE3LjM4NTYyNzQgWiBNMTEuOTUyLDE1LjYwOTYyNzQgQzEyLjQ4LDE1LjYwOTYyNzQgMTIuOTIsMTUuNDI5NjI3NCAxMy4yNzIsMTUuMDY5NjI3NCBDMTMuNjI0LDE0LjcwOTYyNzQgMTMuOCwxNC4yNDE2Mjc0IDEzLjgsMTMuNjY1NjI3NCBDMTMuOCwxMy4xMDU2Mjc0IDEzLjYyNCwxMi42NDE2Mjc0IDEzLjI3MiwxMi4yNzM2Mjc0IEMxMi45MiwxMS45MDU2Mjc0IDEyLjQ4LDExLjcyMTYyNzQgMTEuOTUyLDExLjcyMTYyNzQgQzExLjM5MiwxMS43MjE2Mjc0IDEwLjkzMiwxMS45MDU2Mjc0IDEwLjU3MiwxMi4yNzM2Mjc0IEMxMC4yMTIsMTIuNjQxNjI3NCAxMC4wMzIsMTMuMTA1NjI3NCAxMC4wMzIsMTMuNjY1NjI3NCBDMTAuMDMyLDE0LjI0MTYyNzQgMTAuMjEyLDE0LjcwOTYyNzQgMTAuNTcyLDE1LjA2OTYyNzQgQzEwLjkzMiwxNS40Mjk2Mjc0IDExLjM5MiwxNS42MDk2Mjc0IDExLjk1MiwxNS42MDk2Mjc0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_pct_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">p = 0.0<br />
tol = 0</td>
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


Printing the `validation` object shows the validation table in an HTML viewing environment. The validation table shows the single entry that corresponds to the validation step created by using [col_pct_null()](Validate.col_pct_null.md#pointblank.Validate.col_pct_null). The validation passed since column `a` has no Null values.

Now, let's check that column `b` has exactly 50% Null values.


``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="b", p=0.5)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5wY3RfbnVsbDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19udWxsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxLjAwMDAwMCwgMS41ODE3MTcpIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICA8cGF0aCBkPSJNNTUsMCBDNTcuNDg1MjgxMywwIDU5LjczNTI4MTMsMS4wMDczNTkzMSA2MS4zNjM5NjEsMi42MzYwMzg5NyBDNjIuOTkyNjQwNyw0LjI2NDcxODYzIDY0LDYuNTE0NzE4NjMgNjQsOSBMNjQsOSBMNjQsNjQgTDksNjQgQzYuNTE0NzE4NjIsNjQgNC4yNjQ3MTg2Miw2Mi45OTI2NDA3IDIuNjM2MDM4OTcsNjEuMzYzOTYxIEMxLjAwNzM1OTMxLDU5LjczNTI4MTQgMCw1Ny40ODUyODE0IDAsNTUgTDAsNTUgTDAsOSBDMCw2LjUxNDcxODYzIDEuMDA3MzU5MzEsNC4yNjQ3MTg2MyAyLjYzNjAzODk3LDIuNjM2MDM4OTcgQzQuMjY0NzE4NjIsMS4wMDczNTkzMSA2LjUxNDcxODYyLDAgOSwwIEw5LDAgTDU1LDAgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NS40MTQ2MjI4LDQxLjUgQzQzLjI4NzcyNTQsNDEuNSA0MS42MjUyMjU0LDQwLjc1MzU3MTUgNDAuNzQ1MjAzLDM5LjQ1MzY4MzIgQzM5Ljg2NTE4MDUsNDAuNzUzNTcxNSAzOC4yMDI2ODA1LDQxLjUgMzYuMDc1NzgzNSw0MS41IEMzMy4yNjM5NTI5LDQxLjUgMjguODAyMzQ1OCwzOS4wNjc3NDU0IDI4LjgwMjM0NTgsMzIuNTQyODU3MiBDMjguODAyMzQ1OCwyOS41MjMyMTQzIDI5Ljc1NDQ2NjIsMjIuNSAzOC41NzM3NzQ0LDIyLjUgQzM4Ljg3Mjc2OTgsMjIuNSAzOS4xMTY2MzE1LDIyLjc0Mzg2MTcgMzkuMTE2NjMxNSwyMy4wNDI4NTcxIEwzOS4xMTY2MzE1LDI0LjEyODU3MTQgQzM5LjExNjYzMTUsMjQuNDI3NTY2OCAzOC44NzI3Njk4LDI0LjY3MTQyODYgMzguNTczNzc0NCwyNC42NzE0Mjg2IEMzNS45NTkxNTM5LDI0LjY3MTQyODYgMzMuMTQ1MjAyOSwyNy4xMzU0OTEzIDMzLjE0NTIwMjksMzIuNTQyODU3MiBDMzMuMTQ1MjAyOSwzNS4wNTE0NTEgMzQuMDEyNTAxOSwzOC43ODU3MTQzIDM2LjQwMjM0NTgsMzguNzg1NzE0MyBDMzguMDk0NTMzNiwzOC43ODU3MTQzIDM5LjA2MTQ5NzgsMzcuMDIzNTQ5IDM5LjY1MzEyNjksMzUuNTc3MzQzOSBDMzguNTc1ODk0OCwzMy43NjY0MDYxIDM4LjAzMDkxNzIsMzIuMDY5OTc3NSAzOC4wMzA5MTcyLDMwLjUzNDcwOTcgQzM4LjAzMDkxNzIsMjcuMDA4MjU5MiAzOS40OTE5NjYyLDI1Ljc1NzE0MjkgNDAuNzQ1MjAzLDI1Ljc1NzE0MjkgQzQxLjk5ODQzOTcsMjUuNzU3MTQyOSA0My40NTk0ODg2LDI3LjAwODI1OTIgNDMuNDU5NDg4NiwzMC41MzQ3MDk3IEM0My40NTk0ODg2LDMyLjA5MTE4MzIgNDIuOTI5MzU1LDMzLjc0NTIwMSA0MS44MzkzOTk0LDM1LjU3NzM0MzkgQzQyLjcxOTQyMTgsMzcuNzA2MzYxNyA0My44MTE0OTc5LDM4Ljc4NTcxNDMgNDUuMDg4MDYwMSwzOC43ODU3MTQzIEM0Ny40Nzc5MDQsMzguNzg1NzE0MyA0OC4zNDUyMDMsMzUuMDUxNDUxIDQ4LjM0NTIwMywzMi41NDI4NTcyIEM0OC4zNDUyMDMsMjcuMTM1NDkxMyA0NS41MzEyNTE5LDI0LjY3MTQyODYgNDIuOTE2NjMxNSwyNC42NzE0Mjg2IEM0Mi42MTU1MTU3LDI0LjY3MTQyODYgNDIuMzczNzc0NCwyNC40Mjc1NjY4IDQyLjM3Mzc3NDQsMjQuMTI4NTcxNCBMNDIuMzczNzc0NCwyMy4wNDI4NTcxIEM0Mi4zNzM3NzQ0LDIyLjc0Mzg2MTcgNDIuNjE1NTE1NywyMi41IDQyLjkxNjYzMTUsMjIuNSBDNTEuNzM1OTM5NywyMi41IDUyLjY4ODA2MDEsMjkuNTIzMjE0MyA1Mi42ODgwNjAxLDMyLjU0Mjg1NzIgQzUyLjY4ODA2MDEsMzkuMDY3NzQ1NCA0OC4yMjY0NTI5LDQxLjUgNDUuNDE0NjIyOCw0MS41IFoiIGlkPSJvbWVnYSIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICAgICAgPGcgaWQ9InBlcmNlbnQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDExLjI2ODUwOCwgMjMuODU0MzczKSIgZmlsbD0iIzAwMDAwMCI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS44OTkyMDU1MywxNy4yMDM3OTg4IEMxLjc4NzIwNTUzLDE3LjEyMzc5ODggMS42NjQsMTcuMDI5NjI3NCAxLjU4NCwxNi44OTM2Mjc0IEMxLjUwNCwxNi43NTc2Mjc0IDEuNDY0LDE2LjYwOTYyNzQgMS40NjQsMTYuNDQ5NjI3NCBDMS40NjQsMTYuMjQxNjI3NCAxLjUzNiwxNi4wMzM2Mjc0IDEuNjgsMTUuODI1NjI3NCBMMTIuMjQsMC40ODk2Mjc0MzQgQzEyLjQzMiwwLjE4NTYyNzQzNCAxMi41OTQyNjYyLDAgMTIuOTQ2MjY2MiwwIEMxMy4xNTQyNjYyLDAgMTMuNDcyLDAuMDg5NjI3NDM0IDEzLjY4LDAuMjAxNjI3NDM0IEMxNC4wNDgsMC40MjU2Mjc0MzQgMTQuMjMyLDAuNjgxNjI3NDM0IDE0LjIzMiwwLjk2OTYyNzQzNCBDMTQuMjMyLDEuMTQ1NjI3NDMgMTQuMTYsMS4zNDU2Mjc0MyAxNC4wMTYsMS41Njk2Mjc0MyBMMy40MzIsMTYuOTc3NjI3NCBDMy4xNzYsMTcuMjgxNjI3NCAyLjg4LDE3LjQzMzYyNzQgMi41NDQsMTcuNDMzNjI3NCBDMi4zMzYsMTcuNDMzNjI3NCAyLjEzOTIwNTUzLDE3LjM0Nzc5ODggMS44OTkyMDU1MywxNy4yMDM3OTg4IFogTTMuODY0LDcuNDczNjI3NDMgQzMuMTc2LDcuNDczNjI3NDMgMi41MzYsNy4zMDU2Mjc0MyAxLjk0NCw2Ljk2OTYyNzQzIEMxLjM1Miw2LjYzMzYyNzQzIDAuODgsNi4xODE2Mjc0MyAwLjUyOCw1LjYxMzYyNzQzIEMwLjE3Niw1LjA0NTYyNzQzIDAsNC40MjU2Mjc0MyAwLDMuNzUzNjI3NDMgQzAsMy4wODE2Mjc0MyAwLjE3MiwyLjQ2MTYyNzQzIDAuNTE2LDEuODkzNjI3NDMgQzAuODYsMS4zMjU2Mjc0MyAxLjMyOCwwLjg3NzYyNzQzNCAxLjkyLDAuNTQ5NjI3NDM0IEMyLjUxMiwwLjIyMTYyNzQzNCAzLjE2LDAuMDU3NjI3NDM0IDMuODY0LDAuMDU3NjI3NDM0IEM0LjU2OCwwLjA1NzYyNzQzNCA1LjIxNiwwLjIyMTYyNzQzNCA1LjgwOCwwLjU0OTYyNzQzNCBDNi40LDAuODc3NjI3NDM0IDYuODY0LDEuMzI1NjI3NDMgNy4yLDEuODkzNjI3NDMgQzcuNTM2LDIuNDYxNjI3NDMgNy43MDQsMy4wODE2Mjc0MyA3LjcwNCwzLjc1MzYyNzQzIEM3LjcwNCw0LjQyNTYyNzQzIDcuNTMyLDUuMDQ1NjI3NDMgNy4xODgsNS42MTM2Mjc0MyBDNi44NDQsNi4xODE2Mjc0MyA2LjM4LDYuNjMzNjI3NDMgNS43OTYsNi45Njk2Mjc0MyBDNS4yMTIsNy4zMDU2Mjc0MyA0LjU2OCw3LjQ3MzYyNzQzIDMuODY0LDcuNDczNjI3NDMgWiBNMy44NjQsNS42OTc2Mjc0MyBDNC40MDgsNS42OTc2Mjc0MyA0Ljg1Miw1LjUxMzYyNzQzIDUuMTk2LDUuMTQ1NjI3NDMgQzUuNTQsNC43Nzc2Mjc0MyA1LjcxMiw0LjMxMzYyNzQzIDUuNzEyLDMuNzUzNjI3NDMgQzUuNzEyLDMuMTc3NjI3NDMgNS41NCwyLjcwNTYyNzQzIDUuMTk2LDIuMzM3NjI3NDMgQzQuODUyLDEuOTY5NjI3NDMgNC40MDgsMS43ODU2Mjc0MyAzLjg2NCwxLjc4NTYyNzQzIEMzLjMwNCwxLjc4NTYyNzQzIDIuODQ4LDEuOTY5NjI3NDMgMi40OTYsMi4zMzc2Mjc0MyBDMi4xNDQsMi43MDU2Mjc0MyAxLjk2OCwzLjE3NzYyNzQzIDEuOTY4LDMuNzUzNjI3NDMgQzEuOTY4LDQuMzEzNjI3NDMgMi4xNDQsNC43Nzc2Mjc0MyAyLjQ5Niw1LjE0NTYyNzQzIEMyLjg0OCw1LjUxMzYyNzQzIDMuMzA0LDUuNjk3NjI3NDMgMy44NjQsNS42OTc2Mjc0MyBaIE0xMS45NTIsMTcuMzg1NjI3NCBDMTEuMjQ4LDE3LjM4NTYyNzQgMTAuNiwxNy4yMTc2Mjc0IDEwLjAwOCwxNi44ODE2Mjc0IEM5LjQxNiwxNi41NDU2Mjc0IDguOTQ4LDE2LjA5MzYyNzQgOC42MDQsMTUuNTI1NjI3NCBDOC4yNiwxNC45NTc2Mjc0IDguMDg4LDE0LjMzNzYyNzQgOC4wODgsMTMuNjY1NjI3NCBDOC4wODgsMTIuOTkzNjI3NCA4LjI2LDEyLjM3MzYyNzQgOC42MDQsMTEuODA1NjI3NCBDOC45NDgsMTEuMjM3NjI3NCA5LjQxNiwxMC43ODk2Mjc0IDEwLjAwOCwxMC40NjE2Mjc0IEMxMC42LDEwLjEzMzYyNzQgMTEuMjQ4LDkuOTY5NjI3NDMgMTEuOTUyLDkuOTY5NjI3NDMgQzEyLjY1Niw5Ljk2OTYyNzQzIDEzLjMsMTAuMTMzNjI3NCAxMy44ODQsMTAuNDYxNjI3NCBDMTQuNDY4LDEwLjc4OTYyNzQgMTQuOTI4LDExLjIzNzYyNzQgMTUuMjY0LDExLjgwNTYyNzQgQzE1LjYsMTIuMzczNjI3NCAxNS43NjgsMTIuOTkzNjI3NCAxNS43NjgsMTMuNjY1NjI3NCBDMTUuNzY4LDE0LjMzNzYyNzQgMTUuNTk2LDE0Ljk1NzYyNzQgMTUuMjUyLDE1LjUyNTYyNzQgQzE0LjkwOCwxNi4wOTM2Mjc0IDE0LjQ0NCwxNi41NDU2Mjc0IDEzLjg2LDE2Ljg4MTYyNzQgQzEzLjI3NiwxNy4yMTc2Mjc0IDEyLjY0LDE3LjM4NTYyNzQgMTEuOTUyLDE3LjM4NTYyNzQgWiBNMTEuOTUyLDE1LjYwOTYyNzQgQzEyLjQ4LDE1LjYwOTYyNzQgMTIuOTIsMTUuNDI5NjI3NCAxMy4yNzIsMTUuMDY5NjI3NCBDMTMuNjI0LDE0LjcwOTYyNzQgMTMuOCwxNC4yNDE2Mjc0IDEzLjgsMTMuNjY1NjI3NCBDMTMuOCwxMy4xMDU2Mjc0IDEzLjYyNCwxMi42NDE2Mjc0IDEzLjI3MiwxMi4yNzM2Mjc0IEMxMi45MiwxMS45MDU2Mjc0IDEyLjQ4LDExLjcyMTYyNzQgMTEuOTUyLDExLjcyMTYyNzQgQzExLjM5MiwxMS43MjE2Mjc0IDEwLjkzMiwxMS45MDU2Mjc0IDEwLjU3MiwxMi4yNzM2Mjc0IEMxMC4yMTIsMTIuNjQxNjI3NCAxMC4wMzIsMTMuMTA1NjI3NCAxMC4wMzIsMTMuNjY1NjI3NCBDMTAuMDMyLDE0LjI0MTYyNzQgMTAuMjEyLDE0LjcwOTYyNzQgMTAuNTcyLDE1LjA2OTYyNzQgQzEwLjkzMiwxNS40Mjk2Mjc0IDExLjM5MiwxNS42MDk2Mjc0IDExLjk1MiwxNS42MDk2Mjc0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_pct_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">p = 0.5<br />
tol = 0</td>
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


This validation also passes, as column `b` has exactly 4 out of 8 values as Null (50%).

Finally, let's validate column `c` with a tolerance. Column `c` has 75% Null values, so we'll check if it's approximately 70% Null with a tolerance of 10%.


``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="c", p=0.70, tol=0.10)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5wY3RfbnVsbDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19udWxsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxLjAwMDAwMCwgMS41ODE3MTcpIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICA8cGF0aCBkPSJNNTUsMCBDNTcuNDg1MjgxMywwIDU5LjczNTI4MTMsMS4wMDczNTkzMSA2MS4zNjM5NjEsMi42MzYwMzg5NyBDNjIuOTkyNjQwNyw0LjI2NDcxODYzIDY0LDYuNTE0NzE4NjMgNjQsOSBMNjQsOSBMNjQsNjQgTDksNjQgQzYuNTE0NzE4NjIsNjQgNC4yNjQ3MTg2Miw2Mi45OTI2NDA3IDIuNjM2MDM4OTcsNjEuMzYzOTYxIEMxLjAwNzM1OTMxLDU5LjczNTI4MTQgMCw1Ny40ODUyODE0IDAsNTUgTDAsNTUgTDAsOSBDMCw2LjUxNDcxODYzIDEuMDA3MzU5MzEsNC4yNjQ3MTg2MyAyLjYzNjAzODk3LDIuNjM2MDM4OTcgQzQuMjY0NzE4NjIsMS4wMDczNTkzMSA2LjUxNDcxODYyLDAgOSwwIEw5LDAgTDU1LDAgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NS40MTQ2MjI4LDQxLjUgQzQzLjI4NzcyNTQsNDEuNSA0MS42MjUyMjU0LDQwLjc1MzU3MTUgNDAuNzQ1MjAzLDM5LjQ1MzY4MzIgQzM5Ljg2NTE4MDUsNDAuNzUzNTcxNSAzOC4yMDI2ODA1LDQxLjUgMzYuMDc1NzgzNSw0MS41IEMzMy4yNjM5NTI5LDQxLjUgMjguODAyMzQ1OCwzOS4wNjc3NDU0IDI4LjgwMjM0NTgsMzIuNTQyODU3MiBDMjguODAyMzQ1OCwyOS41MjMyMTQzIDI5Ljc1NDQ2NjIsMjIuNSAzOC41NzM3NzQ0LDIyLjUgQzM4Ljg3Mjc2OTgsMjIuNSAzOS4xMTY2MzE1LDIyLjc0Mzg2MTcgMzkuMTE2NjMxNSwyMy4wNDI4NTcxIEwzOS4xMTY2MzE1LDI0LjEyODU3MTQgQzM5LjExNjYzMTUsMjQuNDI3NTY2OCAzOC44NzI3Njk4LDI0LjY3MTQyODYgMzguNTczNzc0NCwyNC42NzE0Mjg2IEMzNS45NTkxNTM5LDI0LjY3MTQyODYgMzMuMTQ1MjAyOSwyNy4xMzU0OTEzIDMzLjE0NTIwMjksMzIuNTQyODU3MiBDMzMuMTQ1MjAyOSwzNS4wNTE0NTEgMzQuMDEyNTAxOSwzOC43ODU3MTQzIDM2LjQwMjM0NTgsMzguNzg1NzE0MyBDMzguMDk0NTMzNiwzOC43ODU3MTQzIDM5LjA2MTQ5NzgsMzcuMDIzNTQ5IDM5LjY1MzEyNjksMzUuNTc3MzQzOSBDMzguNTc1ODk0OCwzMy43NjY0MDYxIDM4LjAzMDkxNzIsMzIuMDY5OTc3NSAzOC4wMzA5MTcyLDMwLjUzNDcwOTcgQzM4LjAzMDkxNzIsMjcuMDA4MjU5MiAzOS40OTE5NjYyLDI1Ljc1NzE0MjkgNDAuNzQ1MjAzLDI1Ljc1NzE0MjkgQzQxLjk5ODQzOTcsMjUuNzU3MTQyOSA0My40NTk0ODg2LDI3LjAwODI1OTIgNDMuNDU5NDg4NiwzMC41MzQ3MDk3IEM0My40NTk0ODg2LDMyLjA5MTE4MzIgNDIuOTI5MzU1LDMzLjc0NTIwMSA0MS44MzkzOTk0LDM1LjU3NzM0MzkgQzQyLjcxOTQyMTgsMzcuNzA2MzYxNyA0My44MTE0OTc5LDM4Ljc4NTcxNDMgNDUuMDg4MDYwMSwzOC43ODU3MTQzIEM0Ny40Nzc5MDQsMzguNzg1NzE0MyA0OC4zNDUyMDMsMzUuMDUxNDUxIDQ4LjM0NTIwMywzMi41NDI4NTcyIEM0OC4zNDUyMDMsMjcuMTM1NDkxMyA0NS41MzEyNTE5LDI0LjY3MTQyODYgNDIuOTE2NjMxNSwyNC42NzE0Mjg2IEM0Mi42MTU1MTU3LDI0LjY3MTQyODYgNDIuMzczNzc0NCwyNC40Mjc1NjY4IDQyLjM3Mzc3NDQsMjQuMTI4NTcxNCBMNDIuMzczNzc0NCwyMy4wNDI4NTcxIEM0Mi4zNzM3NzQ0LDIyLjc0Mzg2MTcgNDIuNjE1NTE1NywyMi41IDQyLjkxNjYzMTUsMjIuNSBDNTEuNzM1OTM5NywyMi41IDUyLjY4ODA2MDEsMjkuNTIzMjE0MyA1Mi42ODgwNjAxLDMyLjU0Mjg1NzIgQzUyLjY4ODA2MDEsMzkuMDY3NzQ1NCA0OC4yMjY0NTI5LDQxLjUgNDUuNDE0NjIyOCw0MS41IFoiIGlkPSJvbWVnYSIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICAgICAgPGcgaWQ9InBlcmNlbnQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDExLjI2ODUwOCwgMjMuODU0MzczKSIgZmlsbD0iIzAwMDAwMCI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS44OTkyMDU1MywxNy4yMDM3OTg4IEMxLjc4NzIwNTUzLDE3LjEyMzc5ODggMS42NjQsMTcuMDI5NjI3NCAxLjU4NCwxNi44OTM2Mjc0IEMxLjUwNCwxNi43NTc2Mjc0IDEuNDY0LDE2LjYwOTYyNzQgMS40NjQsMTYuNDQ5NjI3NCBDMS40NjQsMTYuMjQxNjI3NCAxLjUzNiwxNi4wMzM2Mjc0IDEuNjgsMTUuODI1NjI3NCBMMTIuMjQsMC40ODk2Mjc0MzQgQzEyLjQzMiwwLjE4NTYyNzQzNCAxMi41OTQyNjYyLDAgMTIuOTQ2MjY2MiwwIEMxMy4xNTQyNjYyLDAgMTMuNDcyLDAuMDg5NjI3NDM0IDEzLjY4LDAuMjAxNjI3NDM0IEMxNC4wNDgsMC40MjU2Mjc0MzQgMTQuMjMyLDAuNjgxNjI3NDM0IDE0LjIzMiwwLjk2OTYyNzQzNCBDMTQuMjMyLDEuMTQ1NjI3NDMgMTQuMTYsMS4zNDU2Mjc0MyAxNC4wMTYsMS41Njk2Mjc0MyBMMy40MzIsMTYuOTc3NjI3NCBDMy4xNzYsMTcuMjgxNjI3NCAyLjg4LDE3LjQzMzYyNzQgMi41NDQsMTcuNDMzNjI3NCBDMi4zMzYsMTcuNDMzNjI3NCAyLjEzOTIwNTUzLDE3LjM0Nzc5ODggMS44OTkyMDU1MywxNy4yMDM3OTg4IFogTTMuODY0LDcuNDczNjI3NDMgQzMuMTc2LDcuNDczNjI3NDMgMi41MzYsNy4zMDU2Mjc0MyAxLjk0NCw2Ljk2OTYyNzQzIEMxLjM1Miw2LjYzMzYyNzQzIDAuODgsNi4xODE2Mjc0MyAwLjUyOCw1LjYxMzYyNzQzIEMwLjE3Niw1LjA0NTYyNzQzIDAsNC40MjU2Mjc0MyAwLDMuNzUzNjI3NDMgQzAsMy4wODE2Mjc0MyAwLjE3MiwyLjQ2MTYyNzQzIDAuNTE2LDEuODkzNjI3NDMgQzAuODYsMS4zMjU2Mjc0MyAxLjMyOCwwLjg3NzYyNzQzNCAxLjkyLDAuNTQ5NjI3NDM0IEMyLjUxMiwwLjIyMTYyNzQzNCAzLjE2LDAuMDU3NjI3NDM0IDMuODY0LDAuMDU3NjI3NDM0IEM0LjU2OCwwLjA1NzYyNzQzNCA1LjIxNiwwLjIyMTYyNzQzNCA1LjgwOCwwLjU0OTYyNzQzNCBDNi40LDAuODc3NjI3NDM0IDYuODY0LDEuMzI1NjI3NDMgNy4yLDEuODkzNjI3NDMgQzcuNTM2LDIuNDYxNjI3NDMgNy43MDQsMy4wODE2Mjc0MyA3LjcwNCwzLjc1MzYyNzQzIEM3LjcwNCw0LjQyNTYyNzQzIDcuNTMyLDUuMDQ1NjI3NDMgNy4xODgsNS42MTM2Mjc0MyBDNi44NDQsNi4xODE2Mjc0MyA2LjM4LDYuNjMzNjI3NDMgNS43OTYsNi45Njk2Mjc0MyBDNS4yMTIsNy4zMDU2Mjc0MyA0LjU2OCw3LjQ3MzYyNzQzIDMuODY0LDcuNDczNjI3NDMgWiBNMy44NjQsNS42OTc2Mjc0MyBDNC40MDgsNS42OTc2Mjc0MyA0Ljg1Miw1LjUxMzYyNzQzIDUuMTk2LDUuMTQ1NjI3NDMgQzUuNTQsNC43Nzc2Mjc0MyA1LjcxMiw0LjMxMzYyNzQzIDUuNzEyLDMuNzUzNjI3NDMgQzUuNzEyLDMuMTc3NjI3NDMgNS41NCwyLjcwNTYyNzQzIDUuMTk2LDIuMzM3NjI3NDMgQzQuODUyLDEuOTY5NjI3NDMgNC40MDgsMS43ODU2Mjc0MyAzLjg2NCwxLjc4NTYyNzQzIEMzLjMwNCwxLjc4NTYyNzQzIDIuODQ4LDEuOTY5NjI3NDMgMi40OTYsMi4zMzc2Mjc0MyBDMi4xNDQsMi43MDU2Mjc0MyAxLjk2OCwzLjE3NzYyNzQzIDEuOTY4LDMuNzUzNjI3NDMgQzEuOTY4LDQuMzEzNjI3NDMgMi4xNDQsNC43Nzc2Mjc0MyAyLjQ5Niw1LjE0NTYyNzQzIEMyLjg0OCw1LjUxMzYyNzQzIDMuMzA0LDUuNjk3NjI3NDMgMy44NjQsNS42OTc2Mjc0MyBaIE0xMS45NTIsMTcuMzg1NjI3NCBDMTEuMjQ4LDE3LjM4NTYyNzQgMTAuNiwxNy4yMTc2Mjc0IDEwLjAwOCwxNi44ODE2Mjc0IEM5LjQxNiwxNi41NDU2Mjc0IDguOTQ4LDE2LjA5MzYyNzQgOC42MDQsMTUuNTI1NjI3NCBDOC4yNiwxNC45NTc2Mjc0IDguMDg4LDE0LjMzNzYyNzQgOC4wODgsMTMuNjY1NjI3NCBDOC4wODgsMTIuOTkzNjI3NCA4LjI2LDEyLjM3MzYyNzQgOC42MDQsMTEuODA1NjI3NCBDOC45NDgsMTEuMjM3NjI3NCA5LjQxNiwxMC43ODk2Mjc0IDEwLjAwOCwxMC40NjE2Mjc0IEMxMC42LDEwLjEzMzYyNzQgMTEuMjQ4LDkuOTY5NjI3NDMgMTEuOTUyLDkuOTY5NjI3NDMgQzEyLjY1Niw5Ljk2OTYyNzQzIDEzLjMsMTAuMTMzNjI3NCAxMy44ODQsMTAuNDYxNjI3NCBDMTQuNDY4LDEwLjc4OTYyNzQgMTQuOTI4LDExLjIzNzYyNzQgMTUuMjY0LDExLjgwNTYyNzQgQzE1LjYsMTIuMzczNjI3NCAxNS43NjgsMTIuOTkzNjI3NCAxNS43NjgsMTMuNjY1NjI3NCBDMTUuNzY4LDE0LjMzNzYyNzQgMTUuNTk2LDE0Ljk1NzYyNzQgMTUuMjUyLDE1LjUyNTYyNzQgQzE0LjkwOCwxNi4wOTM2Mjc0IDE0LjQ0NCwxNi41NDU2Mjc0IDEzLjg2LDE2Ljg4MTYyNzQgQzEzLjI3NiwxNy4yMTc2Mjc0IDEyLjY0LDE3LjM4NTYyNzQgMTEuOTUyLDE3LjM4NTYyNzQgWiBNMTEuOTUyLDE1LjYwOTYyNzQgQzEyLjQ4LDE1LjYwOTYyNzQgMTIuOTIsMTUuNDI5NjI3NCAxMy4yNzIsMTUuMDY5NjI3NCBDMTMuNjI0LDE0LjcwOTYyNzQgMTMuOCwxNC4yNDE2Mjc0IDEzLjgsMTMuNjY1NjI3NCBDMTMuOCwxMy4xMDU2Mjc0IDEzLjYyNCwxMi42NDE2Mjc0IDEzLjI3MiwxMi4yNzM2Mjc0IEMxMi45MiwxMS45MDU2Mjc0IDEyLjQ4LDExLjcyMTYyNzQgMTEuOTUyLDExLjcyMTYyNzQgQzExLjM5MiwxMS43MjE2Mjc0IDEwLjkzMiwxMS45MDU2Mjc0IDEwLjU3MiwxMi4yNzM2Mjc0IEMxMC4yMTIsMTIuNjQxNjI3NCAxMC4wMzIsMTMuMTA1NjI3NCAxMC4wMzIsMTMuNjY1NjI3NCBDMTAuMDMyLDE0LjI0MTYyNzQgMTAuMjEyLDE0LjcwOTYyNzQgMTAuNTcyLDE1LjA2OTYyNzQgQzEwLjkzMiwxNS40Mjk2Mjc0IDExLjM5MiwxNS42MDk2Mjc0IDExLjk1MiwxNS42MDk2Mjc0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_pct_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">p = 0.7<br />
tol = 0.1</td>
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


This validation passes because the actual percentage (75%) falls within the acceptable range of 60% to 80% (70% ± 10%).

The `tol=` parameter supports multiple formats to express tolerance. Let's explore all the different ways to specify tolerance using column `b`, which has exactly 50% Null values (4 out of 8 values).

*Using an absolute tolerance (integer)*: Specify the exact number of rows that can deviate. With `tol=1`, we allow the count to differ by 1 row in either direction.


``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="b", p=0.375, tol=1)  # Expect 3 nulls, allow ±1 (range: 2-4)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5wY3RfbnVsbDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19udWxsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxLjAwMDAwMCwgMS41ODE3MTcpIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICA8cGF0aCBkPSJNNTUsMCBDNTcuNDg1MjgxMywwIDU5LjczNTI4MTMsMS4wMDczNTkzMSA2MS4zNjM5NjEsMi42MzYwMzg5NyBDNjIuOTkyNjQwNyw0LjI2NDcxODYzIDY0LDYuNTE0NzE4NjMgNjQsOSBMNjQsOSBMNjQsNjQgTDksNjQgQzYuNTE0NzE4NjIsNjQgNC4yNjQ3MTg2Miw2Mi45OTI2NDA3IDIuNjM2MDM4OTcsNjEuMzYzOTYxIEMxLjAwNzM1OTMxLDU5LjczNTI4MTQgMCw1Ny40ODUyODE0IDAsNTUgTDAsNTUgTDAsOSBDMCw2LjUxNDcxODYzIDEuMDA3MzU5MzEsNC4yNjQ3MTg2MyAyLjYzNjAzODk3LDIuNjM2MDM4OTcgQzQuMjY0NzE4NjIsMS4wMDczNTkzMSA2LjUxNDcxODYyLDAgOSwwIEw5LDAgTDU1LDAgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NS40MTQ2MjI4LDQxLjUgQzQzLjI4NzcyNTQsNDEuNSA0MS42MjUyMjU0LDQwLjc1MzU3MTUgNDAuNzQ1MjAzLDM5LjQ1MzY4MzIgQzM5Ljg2NTE4MDUsNDAuNzUzNTcxNSAzOC4yMDI2ODA1LDQxLjUgMzYuMDc1NzgzNSw0MS41IEMzMy4yNjM5NTI5LDQxLjUgMjguODAyMzQ1OCwzOS4wNjc3NDU0IDI4LjgwMjM0NTgsMzIuNTQyODU3MiBDMjguODAyMzQ1OCwyOS41MjMyMTQzIDI5Ljc1NDQ2NjIsMjIuNSAzOC41NzM3NzQ0LDIyLjUgQzM4Ljg3Mjc2OTgsMjIuNSAzOS4xMTY2MzE1LDIyLjc0Mzg2MTcgMzkuMTE2NjMxNSwyMy4wNDI4NTcxIEwzOS4xMTY2MzE1LDI0LjEyODU3MTQgQzM5LjExNjYzMTUsMjQuNDI3NTY2OCAzOC44NzI3Njk4LDI0LjY3MTQyODYgMzguNTczNzc0NCwyNC42NzE0Mjg2IEMzNS45NTkxNTM5LDI0LjY3MTQyODYgMzMuMTQ1MjAyOSwyNy4xMzU0OTEzIDMzLjE0NTIwMjksMzIuNTQyODU3MiBDMzMuMTQ1MjAyOSwzNS4wNTE0NTEgMzQuMDEyNTAxOSwzOC43ODU3MTQzIDM2LjQwMjM0NTgsMzguNzg1NzE0MyBDMzguMDk0NTMzNiwzOC43ODU3MTQzIDM5LjA2MTQ5NzgsMzcuMDIzNTQ5IDM5LjY1MzEyNjksMzUuNTc3MzQzOSBDMzguNTc1ODk0OCwzMy43NjY0MDYxIDM4LjAzMDkxNzIsMzIuMDY5OTc3NSAzOC4wMzA5MTcyLDMwLjUzNDcwOTcgQzM4LjAzMDkxNzIsMjcuMDA4MjU5MiAzOS40OTE5NjYyLDI1Ljc1NzE0MjkgNDAuNzQ1MjAzLDI1Ljc1NzE0MjkgQzQxLjk5ODQzOTcsMjUuNzU3MTQyOSA0My40NTk0ODg2LDI3LjAwODI1OTIgNDMuNDU5NDg4NiwzMC41MzQ3MDk3IEM0My40NTk0ODg2LDMyLjA5MTE4MzIgNDIuOTI5MzU1LDMzLjc0NTIwMSA0MS44MzkzOTk0LDM1LjU3NzM0MzkgQzQyLjcxOTQyMTgsMzcuNzA2MzYxNyA0My44MTE0OTc5LDM4Ljc4NTcxNDMgNDUuMDg4MDYwMSwzOC43ODU3MTQzIEM0Ny40Nzc5MDQsMzguNzg1NzE0MyA0OC4zNDUyMDMsMzUuMDUxNDUxIDQ4LjM0NTIwMywzMi41NDI4NTcyIEM0OC4zNDUyMDMsMjcuMTM1NDkxMyA0NS41MzEyNTE5LDI0LjY3MTQyODYgNDIuOTE2NjMxNSwyNC42NzE0Mjg2IEM0Mi42MTU1MTU3LDI0LjY3MTQyODYgNDIuMzczNzc0NCwyNC40Mjc1NjY4IDQyLjM3Mzc3NDQsMjQuMTI4NTcxNCBMNDIuMzczNzc0NCwyMy4wNDI4NTcxIEM0Mi4zNzM3NzQ0LDIyLjc0Mzg2MTcgNDIuNjE1NTE1NywyMi41IDQyLjkxNjYzMTUsMjIuNSBDNTEuNzM1OTM5NywyMi41IDUyLjY4ODA2MDEsMjkuNTIzMjE0MyA1Mi42ODgwNjAxLDMyLjU0Mjg1NzIgQzUyLjY4ODA2MDEsMzkuMDY3NzQ1NCA0OC4yMjY0NTI5LDQxLjUgNDUuNDE0NjIyOCw0MS41IFoiIGlkPSJvbWVnYSIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICAgICAgPGcgaWQ9InBlcmNlbnQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDExLjI2ODUwOCwgMjMuODU0MzczKSIgZmlsbD0iIzAwMDAwMCI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS44OTkyMDU1MywxNy4yMDM3OTg4IEMxLjc4NzIwNTUzLDE3LjEyMzc5ODggMS42NjQsMTcuMDI5NjI3NCAxLjU4NCwxNi44OTM2Mjc0IEMxLjUwNCwxNi43NTc2Mjc0IDEuNDY0LDE2LjYwOTYyNzQgMS40NjQsMTYuNDQ5NjI3NCBDMS40NjQsMTYuMjQxNjI3NCAxLjUzNiwxNi4wMzM2Mjc0IDEuNjgsMTUuODI1NjI3NCBMMTIuMjQsMC40ODk2Mjc0MzQgQzEyLjQzMiwwLjE4NTYyNzQzNCAxMi41OTQyNjYyLDAgMTIuOTQ2MjY2MiwwIEMxMy4xNTQyNjYyLDAgMTMuNDcyLDAuMDg5NjI3NDM0IDEzLjY4LDAuMjAxNjI3NDM0IEMxNC4wNDgsMC40MjU2Mjc0MzQgMTQuMjMyLDAuNjgxNjI3NDM0IDE0LjIzMiwwLjk2OTYyNzQzNCBDMTQuMjMyLDEuMTQ1NjI3NDMgMTQuMTYsMS4zNDU2Mjc0MyAxNC4wMTYsMS41Njk2Mjc0MyBMMy40MzIsMTYuOTc3NjI3NCBDMy4xNzYsMTcuMjgxNjI3NCAyLjg4LDE3LjQzMzYyNzQgMi41NDQsMTcuNDMzNjI3NCBDMi4zMzYsMTcuNDMzNjI3NCAyLjEzOTIwNTUzLDE3LjM0Nzc5ODggMS44OTkyMDU1MywxNy4yMDM3OTg4IFogTTMuODY0LDcuNDczNjI3NDMgQzMuMTc2LDcuNDczNjI3NDMgMi41MzYsNy4zMDU2Mjc0MyAxLjk0NCw2Ljk2OTYyNzQzIEMxLjM1Miw2LjYzMzYyNzQzIDAuODgsNi4xODE2Mjc0MyAwLjUyOCw1LjYxMzYyNzQzIEMwLjE3Niw1LjA0NTYyNzQzIDAsNC40MjU2Mjc0MyAwLDMuNzUzNjI3NDMgQzAsMy4wODE2Mjc0MyAwLjE3MiwyLjQ2MTYyNzQzIDAuNTE2LDEuODkzNjI3NDMgQzAuODYsMS4zMjU2Mjc0MyAxLjMyOCwwLjg3NzYyNzQzNCAxLjkyLDAuNTQ5NjI3NDM0IEMyLjUxMiwwLjIyMTYyNzQzNCAzLjE2LDAuMDU3NjI3NDM0IDMuODY0LDAuMDU3NjI3NDM0IEM0LjU2OCwwLjA1NzYyNzQzNCA1LjIxNiwwLjIyMTYyNzQzNCA1LjgwOCwwLjU0OTYyNzQzNCBDNi40LDAuODc3NjI3NDM0IDYuODY0LDEuMzI1NjI3NDMgNy4yLDEuODkzNjI3NDMgQzcuNTM2LDIuNDYxNjI3NDMgNy43MDQsMy4wODE2Mjc0MyA3LjcwNCwzLjc1MzYyNzQzIEM3LjcwNCw0LjQyNTYyNzQzIDcuNTMyLDUuMDQ1NjI3NDMgNy4xODgsNS42MTM2Mjc0MyBDNi44NDQsNi4xODE2Mjc0MyA2LjM4LDYuNjMzNjI3NDMgNS43OTYsNi45Njk2Mjc0MyBDNS4yMTIsNy4zMDU2Mjc0MyA0LjU2OCw3LjQ3MzYyNzQzIDMuODY0LDcuNDczNjI3NDMgWiBNMy44NjQsNS42OTc2Mjc0MyBDNC40MDgsNS42OTc2Mjc0MyA0Ljg1Miw1LjUxMzYyNzQzIDUuMTk2LDUuMTQ1NjI3NDMgQzUuNTQsNC43Nzc2Mjc0MyA1LjcxMiw0LjMxMzYyNzQzIDUuNzEyLDMuNzUzNjI3NDMgQzUuNzEyLDMuMTc3NjI3NDMgNS41NCwyLjcwNTYyNzQzIDUuMTk2LDIuMzM3NjI3NDMgQzQuODUyLDEuOTY5NjI3NDMgNC40MDgsMS43ODU2Mjc0MyAzLjg2NCwxLjc4NTYyNzQzIEMzLjMwNCwxLjc4NTYyNzQzIDIuODQ4LDEuOTY5NjI3NDMgMi40OTYsMi4zMzc2Mjc0MyBDMi4xNDQsMi43MDU2Mjc0MyAxLjk2OCwzLjE3NzYyNzQzIDEuOTY4LDMuNzUzNjI3NDMgQzEuOTY4LDQuMzEzNjI3NDMgMi4xNDQsNC43Nzc2Mjc0MyAyLjQ5Niw1LjE0NTYyNzQzIEMyLjg0OCw1LjUxMzYyNzQzIDMuMzA0LDUuNjk3NjI3NDMgMy44NjQsNS42OTc2Mjc0MyBaIE0xMS45NTIsMTcuMzg1NjI3NCBDMTEuMjQ4LDE3LjM4NTYyNzQgMTAuNiwxNy4yMTc2Mjc0IDEwLjAwOCwxNi44ODE2Mjc0IEM5LjQxNiwxNi41NDU2Mjc0IDguOTQ4LDE2LjA5MzYyNzQgOC42MDQsMTUuNTI1NjI3NCBDOC4yNiwxNC45NTc2Mjc0IDguMDg4LDE0LjMzNzYyNzQgOC4wODgsMTMuNjY1NjI3NCBDOC4wODgsMTIuOTkzNjI3NCA4LjI2LDEyLjM3MzYyNzQgOC42MDQsMTEuODA1NjI3NCBDOC45NDgsMTEuMjM3NjI3NCA5LjQxNiwxMC43ODk2Mjc0IDEwLjAwOCwxMC40NjE2Mjc0IEMxMC42LDEwLjEzMzYyNzQgMTEuMjQ4LDkuOTY5NjI3NDMgMTEuOTUyLDkuOTY5NjI3NDMgQzEyLjY1Niw5Ljk2OTYyNzQzIDEzLjMsMTAuMTMzNjI3NCAxMy44ODQsMTAuNDYxNjI3NCBDMTQuNDY4LDEwLjc4OTYyNzQgMTQuOTI4LDExLjIzNzYyNzQgMTUuMjY0LDExLjgwNTYyNzQgQzE1LjYsMTIuMzczNjI3NCAxNS43NjgsMTIuOTkzNjI3NCAxNS43NjgsMTMuNjY1NjI3NCBDMTUuNzY4LDE0LjMzNzYyNzQgMTUuNTk2LDE0Ljk1NzYyNzQgMTUuMjUyLDE1LjUyNTYyNzQgQzE0LjkwOCwxNi4wOTM2Mjc0IDE0LjQ0NCwxNi41NDU2Mjc0IDEzLjg2LDE2Ljg4MTYyNzQgQzEzLjI3NiwxNy4yMTc2Mjc0IDEyLjY0LDE3LjM4NTYyNzQgMTEuOTUyLDE3LjM4NTYyNzQgWiBNMTEuOTUyLDE1LjYwOTYyNzQgQzEyLjQ4LDE1LjYwOTYyNzQgMTIuOTIsMTUuNDI5NjI3NCAxMy4yNzIsMTUuMDY5NjI3NCBDMTMuNjI0LDE0LjcwOTYyNzQgMTMuOCwxNC4yNDE2Mjc0IDEzLjgsMTMuNjY1NjI3NCBDMTMuOCwxMy4xMDU2Mjc0IDEzLjYyNCwxMi42NDE2Mjc0IDEzLjI3MiwxMi4yNzM2Mjc0IEMxMi45MiwxMS45MDU2Mjc0IDEyLjQ4LDExLjcyMTYyNzQgMTEuOTUyLDExLjcyMTYyNzQgQzExLjM5MiwxMS43MjE2Mjc0IDEwLjkzMiwxMS45MDU2Mjc0IDEwLjU3MiwxMi4yNzM2Mjc0IEMxMC4yMTIsMTIuNjQxNjI3NCAxMC4wMzIsMTMuMTA1NjI3NCAxMC4wMzIsMTMuNjY1NjI3NCBDMTAuMDMyLDE0LjI0MTYyNzQgMTAuMjEyLDE0LjcwOTYyNzQgMTAuNTcyLDE1LjA2OTYyNzQgQzEwLjkzMiwxNS40Mjk2Mjc0IDExLjM5MiwxNS42MDk2Mjc0IDExLjk1MiwxNS42MDk2Mjc0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_pct_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">p = 0.375<br />
tol = 1</td>
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


This passes because column `b` has 4 Null values, which falls within the acceptable range of 2 to 4 (3 ± 1).

*Using a relative tolerance (float)*: Specify the tolerance as a proportion of the expected count. With `tol=0.25`, we allow a 25% deviation from the expected count.


``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="b", p=0.375, tol=0.25)  # Expect 3 nulls, allow ±25% (range: 2.25-3.75)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5wY3RfbnVsbDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19udWxsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxLjAwMDAwMCwgMS41ODE3MTcpIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICA8cGF0aCBkPSJNNTUsMCBDNTcuNDg1MjgxMywwIDU5LjczNTI4MTMsMS4wMDczNTkzMSA2MS4zNjM5NjEsMi42MzYwMzg5NyBDNjIuOTkyNjQwNyw0LjI2NDcxODYzIDY0LDYuNTE0NzE4NjMgNjQsOSBMNjQsOSBMNjQsNjQgTDksNjQgQzYuNTE0NzE4NjIsNjQgNC4yNjQ3MTg2Miw2Mi45OTI2NDA3IDIuNjM2MDM4OTcsNjEuMzYzOTYxIEMxLjAwNzM1OTMxLDU5LjczNTI4MTQgMCw1Ny40ODUyODE0IDAsNTUgTDAsNTUgTDAsOSBDMCw2LjUxNDcxODYzIDEuMDA3MzU5MzEsNC4yNjQ3MTg2MyAyLjYzNjAzODk3LDIuNjM2MDM4OTcgQzQuMjY0NzE4NjIsMS4wMDczNTkzMSA2LjUxNDcxODYyLDAgOSwwIEw5LDAgTDU1LDAgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NS40MTQ2MjI4LDQxLjUgQzQzLjI4NzcyNTQsNDEuNSA0MS42MjUyMjU0LDQwLjc1MzU3MTUgNDAuNzQ1MjAzLDM5LjQ1MzY4MzIgQzM5Ljg2NTE4MDUsNDAuNzUzNTcxNSAzOC4yMDI2ODA1LDQxLjUgMzYuMDc1NzgzNSw0MS41IEMzMy4yNjM5NTI5LDQxLjUgMjguODAyMzQ1OCwzOS4wNjc3NDU0IDI4LjgwMjM0NTgsMzIuNTQyODU3MiBDMjguODAyMzQ1OCwyOS41MjMyMTQzIDI5Ljc1NDQ2NjIsMjIuNSAzOC41NzM3NzQ0LDIyLjUgQzM4Ljg3Mjc2OTgsMjIuNSAzOS4xMTY2MzE1LDIyLjc0Mzg2MTcgMzkuMTE2NjMxNSwyMy4wNDI4NTcxIEwzOS4xMTY2MzE1LDI0LjEyODU3MTQgQzM5LjExNjYzMTUsMjQuNDI3NTY2OCAzOC44NzI3Njk4LDI0LjY3MTQyODYgMzguNTczNzc0NCwyNC42NzE0Mjg2IEMzNS45NTkxNTM5LDI0LjY3MTQyODYgMzMuMTQ1MjAyOSwyNy4xMzU0OTEzIDMzLjE0NTIwMjksMzIuNTQyODU3MiBDMzMuMTQ1MjAyOSwzNS4wNTE0NTEgMzQuMDEyNTAxOSwzOC43ODU3MTQzIDM2LjQwMjM0NTgsMzguNzg1NzE0MyBDMzguMDk0NTMzNiwzOC43ODU3MTQzIDM5LjA2MTQ5NzgsMzcuMDIzNTQ5IDM5LjY1MzEyNjksMzUuNTc3MzQzOSBDMzguNTc1ODk0OCwzMy43NjY0MDYxIDM4LjAzMDkxNzIsMzIuMDY5OTc3NSAzOC4wMzA5MTcyLDMwLjUzNDcwOTcgQzM4LjAzMDkxNzIsMjcuMDA4MjU5MiAzOS40OTE5NjYyLDI1Ljc1NzE0MjkgNDAuNzQ1MjAzLDI1Ljc1NzE0MjkgQzQxLjk5ODQzOTcsMjUuNzU3MTQyOSA0My40NTk0ODg2LDI3LjAwODI1OTIgNDMuNDU5NDg4NiwzMC41MzQ3MDk3IEM0My40NTk0ODg2LDMyLjA5MTE4MzIgNDIuOTI5MzU1LDMzLjc0NTIwMSA0MS44MzkzOTk0LDM1LjU3NzM0MzkgQzQyLjcxOTQyMTgsMzcuNzA2MzYxNyA0My44MTE0OTc5LDM4Ljc4NTcxNDMgNDUuMDg4MDYwMSwzOC43ODU3MTQzIEM0Ny40Nzc5MDQsMzguNzg1NzE0MyA0OC4zNDUyMDMsMzUuMDUxNDUxIDQ4LjM0NTIwMywzMi41NDI4NTcyIEM0OC4zNDUyMDMsMjcuMTM1NDkxMyA0NS41MzEyNTE5LDI0LjY3MTQyODYgNDIuOTE2NjMxNSwyNC42NzE0Mjg2IEM0Mi42MTU1MTU3LDI0LjY3MTQyODYgNDIuMzczNzc0NCwyNC40Mjc1NjY4IDQyLjM3Mzc3NDQsMjQuMTI4NTcxNCBMNDIuMzczNzc0NCwyMy4wNDI4NTcxIEM0Mi4zNzM3NzQ0LDIyLjc0Mzg2MTcgNDIuNjE1NTE1NywyMi41IDQyLjkxNjYzMTUsMjIuNSBDNTEuNzM1OTM5NywyMi41IDUyLjY4ODA2MDEsMjkuNTIzMjE0MyA1Mi42ODgwNjAxLDMyLjU0Mjg1NzIgQzUyLjY4ODA2MDEsMzkuMDY3NzQ1NCA0OC4yMjY0NTI5LDQxLjUgNDUuNDE0NjIyOCw0MS41IFoiIGlkPSJvbWVnYSIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICAgICAgPGcgaWQ9InBlcmNlbnQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDExLjI2ODUwOCwgMjMuODU0MzczKSIgZmlsbD0iIzAwMDAwMCI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS44OTkyMDU1MywxNy4yMDM3OTg4IEMxLjc4NzIwNTUzLDE3LjEyMzc5ODggMS42NjQsMTcuMDI5NjI3NCAxLjU4NCwxNi44OTM2Mjc0IEMxLjUwNCwxNi43NTc2Mjc0IDEuNDY0LDE2LjYwOTYyNzQgMS40NjQsMTYuNDQ5NjI3NCBDMS40NjQsMTYuMjQxNjI3NCAxLjUzNiwxNi4wMzM2Mjc0IDEuNjgsMTUuODI1NjI3NCBMMTIuMjQsMC40ODk2Mjc0MzQgQzEyLjQzMiwwLjE4NTYyNzQzNCAxMi41OTQyNjYyLDAgMTIuOTQ2MjY2MiwwIEMxMy4xNTQyNjYyLDAgMTMuNDcyLDAuMDg5NjI3NDM0IDEzLjY4LDAuMjAxNjI3NDM0IEMxNC4wNDgsMC40MjU2Mjc0MzQgMTQuMjMyLDAuNjgxNjI3NDM0IDE0LjIzMiwwLjk2OTYyNzQzNCBDMTQuMjMyLDEuMTQ1NjI3NDMgMTQuMTYsMS4zNDU2Mjc0MyAxNC4wMTYsMS41Njk2Mjc0MyBMMy40MzIsMTYuOTc3NjI3NCBDMy4xNzYsMTcuMjgxNjI3NCAyLjg4LDE3LjQzMzYyNzQgMi41NDQsMTcuNDMzNjI3NCBDMi4zMzYsMTcuNDMzNjI3NCAyLjEzOTIwNTUzLDE3LjM0Nzc5ODggMS44OTkyMDU1MywxNy4yMDM3OTg4IFogTTMuODY0LDcuNDczNjI3NDMgQzMuMTc2LDcuNDczNjI3NDMgMi41MzYsNy4zMDU2Mjc0MyAxLjk0NCw2Ljk2OTYyNzQzIEMxLjM1Miw2LjYzMzYyNzQzIDAuODgsNi4xODE2Mjc0MyAwLjUyOCw1LjYxMzYyNzQzIEMwLjE3Niw1LjA0NTYyNzQzIDAsNC40MjU2Mjc0MyAwLDMuNzUzNjI3NDMgQzAsMy4wODE2Mjc0MyAwLjE3MiwyLjQ2MTYyNzQzIDAuNTE2LDEuODkzNjI3NDMgQzAuODYsMS4zMjU2Mjc0MyAxLjMyOCwwLjg3NzYyNzQzNCAxLjkyLDAuNTQ5NjI3NDM0IEMyLjUxMiwwLjIyMTYyNzQzNCAzLjE2LDAuMDU3NjI3NDM0IDMuODY0LDAuMDU3NjI3NDM0IEM0LjU2OCwwLjA1NzYyNzQzNCA1LjIxNiwwLjIyMTYyNzQzNCA1LjgwOCwwLjU0OTYyNzQzNCBDNi40LDAuODc3NjI3NDM0IDYuODY0LDEuMzI1NjI3NDMgNy4yLDEuODkzNjI3NDMgQzcuNTM2LDIuNDYxNjI3NDMgNy43MDQsMy4wODE2Mjc0MyA3LjcwNCwzLjc1MzYyNzQzIEM3LjcwNCw0LjQyNTYyNzQzIDcuNTMyLDUuMDQ1NjI3NDMgNy4xODgsNS42MTM2Mjc0MyBDNi44NDQsNi4xODE2Mjc0MyA2LjM4LDYuNjMzNjI3NDMgNS43OTYsNi45Njk2Mjc0MyBDNS4yMTIsNy4zMDU2Mjc0MyA0LjU2OCw3LjQ3MzYyNzQzIDMuODY0LDcuNDczNjI3NDMgWiBNMy44NjQsNS42OTc2Mjc0MyBDNC40MDgsNS42OTc2Mjc0MyA0Ljg1Miw1LjUxMzYyNzQzIDUuMTk2LDUuMTQ1NjI3NDMgQzUuNTQsNC43Nzc2Mjc0MyA1LjcxMiw0LjMxMzYyNzQzIDUuNzEyLDMuNzUzNjI3NDMgQzUuNzEyLDMuMTc3NjI3NDMgNS41NCwyLjcwNTYyNzQzIDUuMTk2LDIuMzM3NjI3NDMgQzQuODUyLDEuOTY5NjI3NDMgNC40MDgsMS43ODU2Mjc0MyAzLjg2NCwxLjc4NTYyNzQzIEMzLjMwNCwxLjc4NTYyNzQzIDIuODQ4LDEuOTY5NjI3NDMgMi40OTYsMi4zMzc2Mjc0MyBDMi4xNDQsMi43MDU2Mjc0MyAxLjk2OCwzLjE3NzYyNzQzIDEuOTY4LDMuNzUzNjI3NDMgQzEuOTY4LDQuMzEzNjI3NDMgMi4xNDQsNC43Nzc2Mjc0MyAyLjQ5Niw1LjE0NTYyNzQzIEMyLjg0OCw1LjUxMzYyNzQzIDMuMzA0LDUuNjk3NjI3NDMgMy44NjQsNS42OTc2Mjc0MyBaIE0xMS45NTIsMTcuMzg1NjI3NCBDMTEuMjQ4LDE3LjM4NTYyNzQgMTAuNiwxNy4yMTc2Mjc0IDEwLjAwOCwxNi44ODE2Mjc0IEM5LjQxNiwxNi41NDU2Mjc0IDguOTQ4LDE2LjA5MzYyNzQgOC42MDQsMTUuNTI1NjI3NCBDOC4yNiwxNC45NTc2Mjc0IDguMDg4LDE0LjMzNzYyNzQgOC4wODgsMTMuNjY1NjI3NCBDOC4wODgsMTIuOTkzNjI3NCA4LjI2LDEyLjM3MzYyNzQgOC42MDQsMTEuODA1NjI3NCBDOC45NDgsMTEuMjM3NjI3NCA5LjQxNiwxMC43ODk2Mjc0IDEwLjAwOCwxMC40NjE2Mjc0IEMxMC42LDEwLjEzMzYyNzQgMTEuMjQ4LDkuOTY5NjI3NDMgMTEuOTUyLDkuOTY5NjI3NDMgQzEyLjY1Niw5Ljk2OTYyNzQzIDEzLjMsMTAuMTMzNjI3NCAxMy44ODQsMTAuNDYxNjI3NCBDMTQuNDY4LDEwLjc4OTYyNzQgMTQuOTI4LDExLjIzNzYyNzQgMTUuMjY0LDExLjgwNTYyNzQgQzE1LjYsMTIuMzczNjI3NCAxNS43NjgsMTIuOTkzNjI3NCAxNS43NjgsMTMuNjY1NjI3NCBDMTUuNzY4LDE0LjMzNzYyNzQgMTUuNTk2LDE0Ljk1NzYyNzQgMTUuMjUyLDE1LjUyNTYyNzQgQzE0LjkwOCwxNi4wOTM2Mjc0IDE0LjQ0NCwxNi41NDU2Mjc0IDEzLjg2LDE2Ljg4MTYyNzQgQzEzLjI3NiwxNy4yMTc2Mjc0IDEyLjY0LDE3LjM4NTYyNzQgMTEuOTUyLDE3LjM4NTYyNzQgWiBNMTEuOTUyLDE1LjYwOTYyNzQgQzEyLjQ4LDE1LjYwOTYyNzQgMTIuOTIsMTUuNDI5NjI3NCAxMy4yNzIsMTUuMDY5NjI3NCBDMTMuNjI0LDE0LjcwOTYyNzQgMTMuOCwxNC4yNDE2Mjc0IDEzLjgsMTMuNjY1NjI3NCBDMTMuOCwxMy4xMDU2Mjc0IDEzLjYyNCwxMi42NDE2Mjc0IDEzLjI3MiwxMi4yNzM2Mjc0IEMxMi45MiwxMS45MDU2Mjc0IDEyLjQ4LDExLjcyMTYyNzQgMTEuOTUyLDExLjcyMTYyNzQgQzExLjM5MiwxMS43MjE2Mjc0IDEwLjkzMiwxMS45MDU2Mjc0IDEwLjU3MiwxMi4yNzM2Mjc0IEMxMC4yMTIsMTIuNjQxNjI3NCAxMC4wMzIsMTMuMTA1NjI3NCAxMC4wMzIsMTMuNjY1NjI3NCBDMTAuMDMyLDE0LjI0MTYyNzQgMTAuMjEyLDE0LjcwOTYyNzQgMTAuNTcyLDE1LjA2OTYyNzQgQzEwLjkzMiwxNS40Mjk2Mjc0IDExLjM5MiwxNS42MDk2Mjc0IDExLjk1MiwxNS42MDk2Mjc0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_pct_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">p = 0.375<br />
tol = 0.25</td>
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


This passes because 4 Null values falls within the acceptable range (3 ± 0.75 calculates to 2.25 to 3.75, which rounds down to 2 to 3 rows).

*Using asymmetric absolute bounds (tuple of integers)*: Specify different lower and upper bounds as absolute values. With `tol=(0, 2)`, we allow no deviation below but up to 2 rows above the expected count.


``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="b", p=0.25, tol=(0, 2))  # Expect 2 Nulls, allow +0/-2 (range: 2-4)
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
MSA0MS44MzkzOTk0LDM1LjU3NzM0MzkgQzQyLjcxOTQyMTgsMzcuNzA2MzYxNyA0My44MTE0OTc5LDM4Ljc4NTcxNDMgNDUuMDg4MDYwMSwzOC43ODU3MTQzIEM0Ny40Nzc5MDQsMzguNzg1NzE0MyA0OC4zNDUyMDMsMzUuMDUxNDUxIDQ4LjM0NTIwMywzMi41NDI4NTcyIEM0OC4zNDUyMDMsMjcuMTM1NDkxMyA0NS41MzEyNTE5LDI0LjY3MTQyODYgNDIuOTE2NjMxNSwyNC42NzE0Mjg2IEM0Mi42MTU1MTU3LDI0LjY3MTQyODYgNDIuMzczNzc0NCwyNC40Mjc1NjY4IDQyLjM3Mzc3NDQsMjQuMTI4NTcxNCBMNDIuMzczNzc0NCwyMy4wNDI4NTcxIEM0Mi4zNzM3NzQ0LDIyLjc0Mzg2MTcgNDIuNjE1NTE1NywyMi41IDQyLjkxNjYzMTUsMjIuNSBDNTEuNzM1OTM5NywyMi41IDUyLjY4ODA2MDEsMjkuNTIzMjE0MyA1Mi42ODgwNjAxLDMyLjU0Mjg1NzIgQzUyLjY4ODA2MDEsMzkuMDY3NzQ1NCA0OC4yMjY0NTI5LDQxLjUgNDUuNDE0NjIyOCw0MS41IFoiIGlkPSJvbWVnYSIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICAgICAgPGcgaWQ9InBlcmNlbnQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDExLjI2ODUwOCwgMjMuODU0MzczKSIgZmlsbD0iIzAwMDAwMCI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS44OTkyMDU1MywxNy4yMDM3OTg4IEMxLjc4NzIwNTUzLDE3LjEyMzc5ODggMS42NjQsMTcuMDI5NjI3NCAxLjU4NCwxNi44OTM2Mjc0IEMxLjUwNCwxNi43NTc2Mjc0IDEuNDY0LDE2LjYwOTYyNzQgMS40NjQsMTYuNDQ5NjI3NCBDMS40NjQsMTYuMjQxNjI3NCAxLjUzNiwxNi4wMzM2Mjc0IDEuNjgsMTUuODI1NjI3NCBMMTIuMjQsMC40ODk2Mjc0MzQgQzEyLjQzMiwwLjE4NTYyNzQzNCAxMi41OTQyNjYyLDAgMTIuOTQ2MjY2MiwwIEMxMy4xNTQyNjYyLDAgMTMuNDcyLDAuMDg5NjI3NDM0IDEzLjY4LDAuMjAxNjI3NDM0IEMxNC4wNDgsMC40MjU2Mjc0MzQgMTQuMjMyLDAuNjgxNjI3NDM0IDE0LjIzMiwwLjk2OTYyNzQzNCBDMTQuMjMyLDEuMTQ1NjI3NDMgMTQuMTYsMS4zNDU2Mjc0MyAxNC4wMTYsMS41Njk2Mjc0MyBMMy40MzIsMTYuOTc3NjI3NCBDMy4xNzYsMTcuMjgxNjI3NCAyLjg4LDE3LjQzMzYyNzQgMi41NDQsMTcuNDMzNjI3NCBDMi4zMzYsMTcuNDMzNjI3NCAyLjEzOTIwNTUzLDE3LjM0Nzc5ODggMS44OTkyMDU1MywxNy4yMDM3OTg4IFogTTMuODY0LDcuNDczNjI3NDMgQzMuMTc2LDcuNDczNjI3NDMgMi41MzYsNy4zMDU2Mjc0MyAxLjk0NCw2Ljk2OTYyNzQzIEMxLjM1Miw2LjYzMzYyNzQzIDAuODgsNi4xODE2Mjc0MyAwLjUyOCw1LjYxMzYyNzQzIEMwLjE3Niw1LjA0NTYyNzQzIDAsNC40MjU2Mjc0MyAwLDMuNzUzNjI3NDMgQzAsMy4wODE2Mjc0MyAwLjE3MiwyLjQ2MTYyNzQzIDAuNTE2LDEuODkzNjI3NDMgQzAuODYsMS4zMjU2Mjc0MyAxLjMyOCwwLjg3NzYyNzQzNCAxLjkyLDAuNTQ5NjI3NDM0IEMyLjUxMiwwLjIyMTYyNzQzNCAzLjE2LDAuMDU3NjI3NDM0IDMuODY0LDAuMDU3NjI3NDM0IEM0LjU2OCwwLjA1NzYyNzQzNCA1LjIxNiwwLjIyMTYyNzQzNCA1LjgwOCwwLjU0OTYyNzQzNCBDNi40LDAuODc3NjI3NDM0IDYuODY0LDEuMzI1NjI3NDMgNy4yLDEuODkzNjI3NDMgQzcuNTM2LDIuNDYxNjI3NDMgNy43MDQsMy4wODE2Mjc0MyA3LjcwNCwzLjc1MzYyNzQzIEM3LjcwNCw0LjQyNTYyNzQzIDcuNTMyLDUuMDQ1NjI3NDMgNy4xODgsNS42MTM2Mjc0MyBDNi44NDQsNi4xODE2Mjc0MyA2LjM4LDYuNjMzNjI3NDMgNS43OTYsNi45Njk2Mjc0MyBDNS4yMTIsNy4zMDU2Mjc0MyA0LjU2OCw3LjQ3MzYyNzQzIDMuODY0LDcuNDczNjI3NDMgWiBNMy44NjQsNS42OTc2Mjc0MyBDNC40MDgsNS42OTc2Mjc0MyA0Ljg1Miw1LjUxMzYyNzQzIDUuMTk2LDUuMTQ1NjI3NDMgQzUuNTQsNC43Nzc2Mjc0MyA1LjcxMiw0LjMxMzYyNzQzIDUuNzEyLDMuNzUzNjI3NDMgQzUuNzEyLDMuMTc3NjI3NDMgNS41NCwyLjcwNTYyNzQzIDUuMTk2LDIuMzM3NjI3NDMgQzQuODUyLDEuOTY5NjI3NDMgNC40MDgsMS43ODU2Mjc0MyAzLjg2NCwxLjc4NTYyNzQzIEMzLjMwNCwxLjc4NTYyNzQzIDIuODQ4LDEuOTY5NjI3NDMgMi40OTYsMi4zMzc2Mjc0MyBDMi4xNDQsMi43MDU2Mjc0MyAxLjk2OCwzLjE3NzYyNzQzIDEuOTY4LDMuNzUzNjI3NDMgQzEuOTY4LDQuMzEzNjI3NDMgMi4xNDQsNC43Nzc2Mjc0MyAyLjQ5Niw1LjE0NTYyNzQzIEMyLjg0OCw1LjUxMzYyNzQzIDMuMzA0LDUuNjk3NjI3NDMgMy44NjQsNS42OTc2Mjc0MyBaIE0xMS45NTIsMTcuMzg1NjI3NCBDMTEuMjQ4LDE3LjM4NTYyNzQgMTAuNiwxNy4yMTc2Mjc0IDEwLjAwOCwxNi44ODE2Mjc0IEM5LjQxNiwxNi41NDU2Mjc0IDguOTQ4LDE2LjA5MzYyNzQgOC42MDQsMTUuNTI1NjI3NCBDOC4yNiwxNC45NTc2Mjc0IDguMDg4LDE0LjMzNzYyNzQgOC4wODgsMTMuNjY1NjI3NCBDOC4wODgsMTIuOTkzNjI3NCA4LjI2LDEyLjM3MzYyNzQgOC42MDQsMTEuODA1NjI3NCBDOC45NDgsMTEuMjM3NjI3NCA5LjQxNiwxMC43ODk2Mjc0IDEwLjAwOCwxMC40NjE2Mjc0IEMxMC42LDEwLjEzMzYyNzQgMTEuMjQ4LDkuOTY5NjI3NDMgMTEuOTUyLDkuOTY5NjI3NDMgQzEyLjY1Niw5Ljk2OTYyNzQzIDEzLjMsMTAuMTMzNjI3NCAxMy44ODQsMTAuNDYxNjI3NCBDMTQuNDY4LDEwLjc4OTYyNzQgMTQuOTI4LDExLjIzNzYyNzQgMTUuMjY0LDExLjgwNTYyNzQgQzE1LjYsMTIuMzczNjI3NCAxNS43NjgsMTIuOTkzNjI3NCAxNS43NjgsMTMuNjY1NjI3NCBDMTUuNzY4LDE0LjMzNzYyNzQgMTUuNTk2LDE0Ljk1NzYyNzQgMTUuMjUyLDE1LjUyNTYyNzQgQzE0LjkwOCwxNi4wOTM2Mjc0IDE0LjQ0NCwxNi41NDU2Mjc0IDEzLjg2LDE2Ljg4MTYyNzQgQzEzLjI3NiwxNy4yMTc2Mjc0IDEyLjY0LDE3LjM4NTYyNzQgMTEuOTUyLDE3LjM4NTYyNzQgWiBNMTEuOTUyLDE1LjYwOTYyNzQgQzEyLjQ4LDE1LjYwOTYyNzQgMTIuOTIsMTUuNDI5NjI3NCAxMy4yNzIsMTUuMDY5NjI3NCBDMTMuNjI0LDE0LjcwOTYyNzQgMTMuOCwxNC4yNDE2Mjc0IDEzLjgsMTMuNjY1NjI3NCBDMTMuOCwxMy4xMDU2Mjc0IDEzLjYyNCwxMi42NDE2Mjc0IDEzLjI3MiwxMi4yNzM2Mjc0IEMxMi45MiwxMS45MDU2Mjc0IDEyLjQ4LDExLjcyMTYyNzQgMTEuOTUyLDExLjcyMTYyNzQgQzExLjM5MiwxMS43MjE2Mjc0IDEwLjkzMiwxMS45MDU2Mjc0IDEwLjU3MiwxMi4yNzM2Mjc0IEMxMC4yMTIsMTIuNjQxNjI3NCAxMC4wMzIsMTMuMTA1NjI3NCAxMC4wMzIsMTMuNjY1NjI3NCBDMTAuMDMyLDE0LjI0MTYyNzQgMTAuMjEyLDE0LjcwOTYyNzQgMTAuNTcyLDE1LjA2OTYyNzQgQzEwLjkzMiwxNS40Mjk2Mjc0IDExLjM5MiwxNS42MDk2Mjc0IDExLjk1MiwxNS42MDk2Mjc0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_pct_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">p = 0.25<br />
tol = (0, 2)</td>
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


This passes because 4 Null values falls within the acceptable range of 2 to 4.

*Using asymmetric relative bounds (tuple of floats)*: Specify different lower and upper bounds as proportions. With `tol=(0.1, 0.3)`, we allow 10% below and 30% above the expected count.


``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="b", p=0.375, tol=(0.1, 0.3))  # Expect 3 Nulls, allow -10%/+30%
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5wY3RfbnVsbDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19udWxsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxLjAwMDAwMCwgMS41ODE3MTcpIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICA8cGF0aCBkPSJNNTUsMCBDNTcuNDg1MjgxMywwIDU5LjczNTI4MTMsMS4wMDczNTkzMSA2MS4zNjM5NjEsMi42MzYwMzg5NyBDNjIuOTkyNjQwNyw0LjI2NDcxODYzIDY0LDYuNTE0NzE4NjMgNjQsOSBMNjQsOSBMNjQsNjQgTDksNjQgQzYuNTE0NzE4NjIsNjQgNC4yNjQ3MTg2Miw2Mi45OTI2NDA3IDIuNjM2MDM4OTcsNjEuMzYzOTYxIEMxLjAwNzM1OTMxLDU5LjczNTI4MTQgMCw1Ny40ODUyODE0IDAsNTUgTDAsNTUgTDAsOSBDMCw2LjUxNDcxODYzIDEuMDA3MzU5MzEsNC4yNjQ3MTg2MyAyLjYzNjAzODk3LDIuNjM2MDM4OTcgQzQuMjY0NzE4NjIsMS4wMDczNTkzMSA2LjUxNDcxODYyLDAgOSwwIEw5LDAgTDU1LDAgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NS40MTQ2MjI4LDQxLjUgQzQzLjI4NzcyNTQsNDEuNSA0MS42MjUyMjU0LDQwLjc1MzU3MTUgNDAuNzQ1MjAzLDM5LjQ1MzY4MzIgQzM5Ljg2NTE4MDUsNDAuNzUzNTcxNSAzOC4yMDI2ODA1LDQxLjUgMzYuMDc1NzgzNSw0MS41IEMzMy4yNjM5NTI5LDQxLjUgMjguODAyMzQ1OCwzOS4wNjc3NDU0IDI4LjgwMjM0NTgsMzIuNTQyODU3MiBDMjguODAyMzQ1OCwyOS41MjMyMTQzIDI5Ljc1NDQ2NjIsMjIuNSAzOC41NzM3NzQ0LDIyLjUgQzM4Ljg3Mjc2OTgsMjIuNSAzOS4xMTY2MzE1LDIyLjc0Mzg2MTcgMzkuMTE2NjMxNSwyMy4wNDI4NTcxIEwzOS4xMTY2MzE1LDI0LjEyODU3MTQgQzM5LjExNjYzMTUsMjQuNDI3NTY2OCAzOC44NzI3Njk4LDI0LjY3MTQyODYgMzguNTczNzc0NCwyNC42NzE0Mjg2IEMzNS45NTkxNTM5LDI0LjY3MTQyODYgMzMuMTQ1MjAyOSwyNy4xMzU0OTEzIDMzLjE0NTIwMjksMzIuNTQyODU3MiBDMzMuMTQ1MjAyOSwzNS4wNTE0NTEgMzQuMDEyNTAxOSwzOC43ODU3MTQzIDM2LjQwMjM0NTgsMzguNzg1NzE0MyBDMzguMDk0NTMzNiwzOC43ODU3MTQzIDM5LjA2MTQ5NzgsMzcuMDIzNTQ5IDM5LjY1MzEyNjksMzUuNTc3MzQzOSBDMzguNTc1ODk0OCwzMy43NjY0MDYxIDM4LjAzMDkxNzIsMzIuMDY5OTc3NSAzOC4wMzA5MTcyLDMwLjUzNDcwOTcgQzM4LjAzMDkxNzIsMjcuMDA4MjU5MiAzOS40OTE5NjYyLDI1Ljc1NzE0MjkgNDAuNzQ1MjAzLDI1Ljc1NzE0MjkgQzQxLjk5ODQzOTcsMjUuNzU3MTQyOSA0My40NTk0ODg2LDI3LjAwODI1OTIgNDMuNDU5NDg4NiwzMC41MzQ3MDk3IEM0My40NTk0ODg2LDMyLjA5MTE4MzIgNDIuOTI5MzU1LDMzLjc0NTIwMSA0MS44MzkzOTk0LDM1LjU3NzM0MzkgQzQyLjcxOTQyMTgsMzcuNzA2MzYxNyA0My44MTE0OTc5LDM4Ljc4NTcxNDMgNDUuMDg4MDYwMSwzOC43ODU3MTQzIEM0Ny40Nzc5MDQsMzguNzg1NzE0MyA0OC4zNDUyMDMsMzUuMDUxNDUxIDQ4LjM0NTIwMywzMi41NDI4NTcyIEM0OC4zNDUyMDMsMjcuMTM1NDkxMyA0NS41MzEyNTE5LDI0LjY3MTQyODYgNDIuOTE2NjMxNSwyNC42NzE0Mjg2IEM0Mi42MTU1MTU3LDI0LjY3MTQyODYgNDIuMzczNzc0NCwyNC40Mjc1NjY4IDQyLjM3Mzc3NDQsMjQuMTI4NTcxNCBMNDIuMzczNzc0NCwyMy4wNDI4NTcxIEM0Mi4zNzM3NzQ0LDIyLjc0Mzg2MTcgNDIuNjE1NTE1NywyMi41IDQyLjkxNjYzMTUsMjIuNSBDNTEuNzM1OTM5NywyMi41IDUyLjY4ODA2MDEsMjkuNTIzMjE0MyA1Mi42ODgwNjAxLDMyLjU0Mjg1NzIgQzUyLjY4ODA2MDEsMzkuMDY3NzQ1NCA0OC4yMjY0NTI5LDQxLjUgNDUuNDE0NjIyOCw0MS41IFoiIGlkPSJvbWVnYSIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICAgICAgPGcgaWQ9InBlcmNlbnQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDExLjI2ODUwOCwgMjMuODU0MzczKSIgZmlsbD0iIzAwMDAwMCI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS44OTkyMDU1MywxNy4yMDM3OTg4IEMxLjc4NzIwNTUzLDE3LjEyMzc5ODggMS42NjQsMTcuMDI5NjI3NCAxLjU4NCwxNi44OTM2Mjc0IEMxLjUwNCwxNi43NTc2Mjc0IDEuNDY0LDE2LjYwOTYyNzQgMS40NjQsMTYuNDQ5NjI3NCBDMS40NjQsMTYuMjQxNjI3NCAxLjUzNiwxNi4wMzM2Mjc0IDEuNjgsMTUuODI1NjI3NCBMMTIuMjQsMC40ODk2Mjc0MzQgQzEyLjQzMiwwLjE4NTYyNzQzNCAxMi41OTQyNjYyLDAgMTIuOTQ2MjY2MiwwIEMxMy4xNTQyNjYyLDAgMTMuNDcyLDAuMDg5NjI3NDM0IDEzLjY4LDAuMjAxNjI3NDM0IEMxNC4wNDgsMC40MjU2Mjc0MzQgMTQuMjMyLDAuNjgxNjI3NDM0IDE0LjIzMiwwLjk2OTYyNzQzNCBDMTQuMjMyLDEuMTQ1NjI3NDMgMTQuMTYsMS4zNDU2Mjc0MyAxNC4wMTYsMS41Njk2Mjc0MyBMMy40MzIsMTYuOTc3NjI3NCBDMy4xNzYsMTcuMjgxNjI3NCAyLjg4LDE3LjQzMzYyNzQgMi41NDQsMTcuNDMzNjI3NCBDMi4zMzYsMTcuNDMzNjI3NCAyLjEzOTIwNTUzLDE3LjM0Nzc5ODggMS44OTkyMDU1MywxNy4yMDM3OTg4IFogTTMuODY0LDcuNDczNjI3NDMgQzMuMTc2LDcuNDczNjI3NDMgMi41MzYsNy4zMDU2Mjc0MyAxLjk0NCw2Ljk2OTYyNzQzIEMxLjM1Miw2LjYzMzYyNzQzIDAuODgsNi4xODE2Mjc0MyAwLjUyOCw1LjYxMzYyNzQzIEMwLjE3Niw1LjA0NTYyNzQzIDAsNC40MjU2Mjc0MyAwLDMuNzUzNjI3NDMgQzAsMy4wODE2Mjc0MyAwLjE3MiwyLjQ2MTYyNzQzIDAuNTE2LDEuODkzNjI3NDMgQzAuODYsMS4zMjU2Mjc0MyAxLjMyOCwwLjg3NzYyNzQzNCAxLjkyLDAuNTQ5NjI3NDM0IEMyLjUxMiwwLjIyMTYyNzQzNCAzLjE2LDAuMDU3NjI3NDM0IDMuODY0LDAuMDU3NjI3NDM0IEM0LjU2OCwwLjA1NzYyNzQzNCA1LjIxNiwwLjIyMTYyNzQzNCA1LjgwOCwwLjU0OTYyNzQzNCBDNi40LDAuODc3NjI3NDM0IDYuODY0LDEuMzI1NjI3NDMgNy4yLDEuODkzNjI3NDMgQzcuNTM2LDIuNDYxNjI3NDMgNy43MDQsMy4wODE2Mjc0MyA3LjcwNCwzLjc1MzYyNzQzIEM3LjcwNCw0LjQyNTYyNzQzIDcuNTMyLDUuMDQ1NjI3NDMgNy4xODgsNS42MTM2Mjc0MyBDNi44NDQsNi4xODE2Mjc0MyA2LjM4LDYuNjMzNjI3NDMgNS43OTYsNi45Njk2Mjc0MyBDNS4yMTIsNy4zMDU2Mjc0MyA0LjU2OCw3LjQ3MzYyNzQzIDMuODY0LDcuNDczNjI3NDMgWiBNMy44NjQsNS42OTc2Mjc0MyBDNC40MDgsNS42OTc2Mjc0MyA0Ljg1Miw1LjUxMzYyNzQzIDUuMTk2LDUuMTQ1NjI3NDMgQzUuNTQsNC43Nzc2Mjc0MyA1LjcxMiw0LjMxMzYyNzQzIDUuNzEyLDMuNzUzNjI3NDMgQzUuNzEyLDMuMTc3NjI3NDMgNS41NCwyLjcwNTYyNzQzIDUuMTk2LDIuMzM3NjI3NDMgQzQuODUyLDEuOTY5NjI3NDMgNC40MDgsMS43ODU2Mjc0MyAzLjg2NCwxLjc4NTYyNzQzIEMzLjMwNCwxLjc4NTYyNzQzIDIuODQ4LDEuOTY5NjI3NDMgMi40OTYsMi4zMzc2Mjc0MyBDMi4xNDQsMi43MDU2Mjc0MyAxLjk2OCwzLjE3NzYyNzQzIDEuOTY4LDMuNzUzNjI3NDMgQzEuOTY4LDQuMzEzNjI3NDMgMi4xNDQsNC43Nzc2Mjc0MyAyLjQ5Niw1LjE0NTYyNzQzIEMyLjg0OCw1LjUxMzYyNzQzIDMuMzA0LDUuNjk3NjI3NDMgMy44NjQsNS42OTc2Mjc0MyBaIE0xMS45NTIsMTcuMzg1NjI3NCBDMTEuMjQ4LDE3LjM4NTYyNzQgMTAuNiwxNy4yMTc2Mjc0IDEwLjAwOCwxNi44ODE2Mjc0IEM5LjQxNiwxNi41NDU2Mjc0IDguOTQ4LDE2LjA5MzYyNzQgOC42MDQsMTUuNTI1NjI3NCBDOC4yNiwxNC45NTc2Mjc0IDguMDg4LDE0LjMzNzYyNzQgOC4wODgsMTMuNjY1NjI3NCBDOC4wODgsMTIuOTkzNjI3NCA4LjI2LDEyLjM3MzYyNzQgOC42MDQsMTEuODA1NjI3NCBDOC45NDgsMTEuMjM3NjI3NCA5LjQxNiwxMC43ODk2Mjc0IDEwLjAwOCwxMC40NjE2Mjc0IEMxMC42LDEwLjEzMzYyNzQgMTEuMjQ4LDkuOTY5NjI3NDMgMTEuOTUyLDkuOTY5NjI3NDMgQzEyLjY1Niw5Ljk2OTYyNzQzIDEzLjMsMTAuMTMzNjI3NCAxMy44ODQsMTAuNDYxNjI3NCBDMTQuNDY4LDEwLjc4OTYyNzQgMTQuOTI4LDExLjIzNzYyNzQgMTUuMjY0LDExLjgwNTYyNzQgQzE1LjYsMTIuMzczNjI3NCAxNS43NjgsMTIuOTkzNjI3NCAxNS43NjgsMTMuNjY1NjI3NCBDMTUuNzY4LDE0LjMzNzYyNzQgMTUuNTk2LDE0Ljk1NzYyNzQgMTUuMjUyLDE1LjUyNTYyNzQgQzE0LjkwOCwxNi4wOTM2Mjc0IDE0LjQ0NCwxNi41NDU2Mjc0IDEzLjg2LDE2Ljg4MTYyNzQgQzEzLjI3NiwxNy4yMTc2Mjc0IDEyLjY0LDE3LjM4NTYyNzQgMTEuOTUyLDE3LjM4NTYyNzQgWiBNMTEuOTUyLDE1LjYwOTYyNzQgQzEyLjQ4LDE1LjYwOTYyNzQgMTIuOTIsMTUuNDI5NjI3NCAxMy4yNzIsMTUuMDY5NjI3NCBDMTMuNjI0LDE0LjcwOTYyNzQgMTMuOCwxNC4yNDE2Mjc0IDEzLjgsMTMuNjY1NjI3NCBDMTMuOCwxMy4xMDU2Mjc0IDEzLjYyNCwxMi42NDE2Mjc0IDEzLjI3MiwxMi4yNzM2Mjc0IEMxMi45MiwxMS45MDU2Mjc0IDEyLjQ4LDExLjcyMTYyNzQgMTEuOTUyLDExLjcyMTYyNzQgQzExLjM5MiwxMS43MjE2Mjc0IDEwLjkzMiwxMS45MDU2Mjc0IDEwLjU3MiwxMi4yNzM2Mjc0IEMxMC4yMTIsMTIuNjQxNjI3NCAxMC4wMzIsMTMuMTA1NjI3NCAxMC4wMzIsMTMuNjY1NjI3NCBDMTAuMDMyLDE0LjI0MTYyNzQgMTAuMjEyLDE0LjcwOTYyNzQgMTAuNTcyLDE1LjA2OTYyNzQgQzEwLjkzMiwxNS40Mjk2Mjc0IDExLjM5MiwxNS42MDk2Mjc0IDExLjk1MiwxNS42MDk2Mjc0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_pct_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">p = 0.375<br />
tol = (0.1, 0.3)</td>
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


This passes because 4 Null values falls within the acceptable range (3 - 0.3 to 3 + 0.9 calculates to 2.7 to 3.9, which rounds down to 2 to 3 rows).
