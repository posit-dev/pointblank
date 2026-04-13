## Validate.col_sd_ge()


Does the column standard deviation satisfy a greater than or equal to comparison?


Usage

``` python
Validate.col_sd_ge(
    columns,
    value=None,
    tol=0,
    thresholds=None,
    brief=False,
    actions=None,
    active=True
)
```


The [col_sd_ge()](Validate.col_sd_ge.md#pointblank.Validate.col_sd_ge) validation method checks whether the standard deviation of values in a column is at least a specified `value=`. This is an aggregation-based validation where the entire column is reduced to a single standard deviation value that is then compared against the target. The comparison used in this function is `standard deviation(column) >= value`.

Unlike row-level validations (e.g., [col_vals_gt()](Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)), this method treats the entire column as a single test unit. The validation either passes completely (if the aggregated value satisfies the comparison) or fails completely.


## Parameters


`columns: _PBUnresolvedColumn`  
A single column or a list of columns to validate. If multiple columns are supplied, there will be a separate validation step generated for each column. The columns must contain numeric data for the standard deviation to be computed.

`value: float | Column | ReferenceColumn | None = None`  
The value to compare the column standard deviation against. This can be: (1) a numeric literal (`int` or `float`), (2) a <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a> object referencing another column whose standard deviation will be used for comparison, (3) a <a href="ref.html#pointblank.ref" class="gdls-link"><code>ref()</code></a> object referencing a column in reference data (when `Validate(reference=)` has been set), or (4) `None` to automatically compare against the same column in reference data (shorthand for `ref(column_name)` when reference data is set).

`tol: Tolerance = ``0`  
A tolerance value for the comparison. The default is `0`, meaning exact comparison. When set to a positive value, the comparison becomes more lenient. For example, with `tol=0.5`, a standard deviation that differs from the target by up to `0.5` will still pass. The `tol=` parameter expands the acceptable range for the comparison. For [col_sd_ge()](Validate.col_sd_ge.md#pointblank.Validate.col_sd_ge), a tolerance of `tol=0.5` would mean the standard deviation can be within `0.5` of the target value and still pass validation.

`thresholds: float | bool | tuple | dict | Thresholds | None = None`  
Failure threshold levels so that the validation step can react accordingly when failing test units are level. Since this is an aggregation-based validation with only one test unit, threshold values typically should be set as absolute counts (e.g., `1`) to indicate pass/fail, or as proportions where any value less than `1.0` means failure is acceptable.

`brief: str | bool = ``False`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the <a href="Actions.html#pointblank.Actions" class="gdls-link"><code>Actions</code></a> class should be used to define the actions.

`active: bool | Callable = ``True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like <a href="has_columns.html#pointblank.has_columns" class="gdls-link"><code>has_columns()</code></a> and <a href="has_rows.html#pointblank.has_rows" class="gdls-link"><code>has_rows()</code></a> can be used here to conditionally activate a step based on properties of the target table.


## Returns


`Validate`  
The [Validate](Validate.md#pointblank.Validate) object with the added validation step.


## Using Reference Data

The [col_sd_ge()](Validate.col_sd_ge.md#pointblank.Validate.col_sd_ge) method supports comparing column aggregations against reference data. This is useful for validating that statistical properties remain consistent across different versions of a dataset, or for comparing current data against historical baselines.

To use reference data, set the `reference=` parameter when creating the [Validate](Validate.md#pointblank.Validate) object:

``` python
validation = (
    pb.Validate(data=current_data, reference=baseline_data)
    .col_sd_ge(columns="revenue")  # Compares sum(current.revenue) vs sum(baseline.revenue)
    .interrogate()
)
```

When `value=None` and reference data is set, the method automatically compares against the same column in the reference data. You can also explicitly specify reference columns using the [ref()](ref.md#pointblank.ref) helper:

``` python
.col_sd_ge(columns="revenue", value=pb.ref("baseline_revenue"))
```


## Understanding Tolerance

The `tol=` parameter allows for fuzzy comparisons, which is especially important for floating-point aggregations where exact equality is often unreliable.

The `tol=` parameter expands the acceptable range for the comparison. For [col_sd_ge()](Validate.col_sd_ge.md#pointblank.Validate.col_sd_ge), a tolerance of `tol=0.5` would mean the standard deviation can be within `0.5` of the target value and still pass validation.

For equality comparisons (`col_*_eq`), the tolerance creates a range `[value - tol, value + tol]` within which the aggregation is considered valid. For inequality comparisons, the tolerance shifts the comparison boundary.


## Thresholds

The `thresholds=` parameter is used to set the failure-condition levels for the validation step. If they are set here at the step level, these thresholds will override any thresholds set at the global level in `Validate(thresholds=...)`.

There are three threshold levels: 'warning', 'error', and 'critical'. Since aggregation validations operate on a single test unit (the aggregated value), threshold values are typically set as absolute counts:

- `thresholds=1` means any failure triggers a 'warning'
- `thresholds=(1, 1, 1)` means any failure triggers all three levels

Thresholds can be defined using one of these input schemes:

1.  use the <a href="Thresholds.html#pointblank.Thresholds" class="gdls-link"><code>Thresholds</code></a> class (the most direct way to create thresholds)
2.  provide a tuple of 1-3 values, where position `0` is the 'warning' level, position `1` is the 'error' level, and position `2` is the 'critical' level
3.  create a dictionary of 1-3 value entries; the valid keys: are 'warning', 'error', and 'critical'
4.  a single integer/float value denoting absolute number or fraction of failing test units for the 'warning' level only


## Examples

For the examples, we'll use a simple Polars DataFrame with numeric columns. The table is shown below:


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [1, 2, 3, 4, 5],
        "b": [2, 2, 2, 2, 2],
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
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
</tr>
</tbody>
</table>


Let's validate that the standard deviation of column `a` is at least `2`:


``` python
validation = (
    pb.Validate(data=tbl)
    .col_sd_ge(columns="a", value=2)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc2RfZ2U8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3NkX2dlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxLjUwMDAwMCwgMS41MDAwMDApIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICA8cGF0aCBkPSJNNTUsMCBDNTcuNDg1MjgxMywwIDU5LjczNTI4MTMsMS4wMDczNTkzMSA2MS4zNjM5NjEsMi42MzYwMzg5NyBDNjIuOTkyNjQwNyw0LjI2NDcxODYzIDY0LDYuNTE0NzE4NjMgNjQsOSBMNjQsOSBMNjQsNjQgTDksNjQgQzYuNTE0NzE4NjIsNjQgNC4yNjQ3MTg2Miw2Mi45OTI2NDA3IDIuNjM2MDM4OTcsNjEuMzYzOTYxIEMxLjAwNzM1OTMxLDU5LjczNTI4MTQgMCw1Ny40ODUyODE0IDAsNTUgTDAsNTUgTDAsOSBDMCw2LjUxNDcxODYzIDEuMDA3MzU5MzEsNC4yNjQ3MTg2MyAyLjYzNjAzODk3LDIuNjM2MDM4OTcgQzQuMjY0NzE4NjIsMS4wMDczNTkzMSA2LjUxNDcxODYyLDAgOSwwIEw5LDAgTDU1LDAgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJzZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjQuMjQ0MDAwLCA0Ni4zNzQwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCwxMC43NjYgQzIuNjY5MzMzMzMsMTAuNzY2IDIuMDMsMTAuNjQyMzMzMyAxLjQ3LDEwLjM5NSBDMC45MSwxMC4xNDc2NjY3IDAuNDY2NjY2NjY3LDkuODE0IDAuMTQsOS4zOTQgQzAuMDQ2NjY2NjY2Nyw5LjI3MjY2NjY3IDAsOS4xNTEzMzMzMyAwLDkuMDMgQzAsOC44MjQ2NjY2NyAwLjA5OCw4LjY3MDY2NjY3IDAuMjk0LDguNTY4IEMwLjM3OCw4LjUyMTMzMzMzIDAuNDY2NjY2NjY3LDguNDk4IDAuNTYsOC40OTggQzAuNzE4NjY2NjY3LDguNDk4IDAuODgyLDguNTgyIDEuMDUsOC43NSBDMS4zNTgsOS4wNzY2NjY2NyAxLjY5ODY2NjY3LDkuMzE0NjY2NjcgMi4wNzIsOS40NjQgQzIuNDQ1MzMzMzMsOS42MTMzMzMzMyAyLjg3NDY2NjY3LDkuNjg4IDMuMzYsOS42ODggQzMuODA4LDkuNjg4IDQuMTc0MzMzMzMsOS42MTU2NjY2NyA0LjQ1OSw5LjQ3MSBDNC43NDM2NjY2Nyw5LjMyNjMzMzMzIDQuODg2LDkuMTI4IDQuODg2LDguODc2IEM0Ljg4Niw4LjY4OTMzMzMzIDQuODMyMzMzMzMsOC41MzUzMzMzMyA0LjcyNSw4LjQxNCBDNC42MTc2NjY2Nyw4LjI5MjY2NjY3IDQuNDMzMzMzMzMsOC4xNzYgNC4xNzIsOC4wNjQgQzMuOTEwNjY2NjcsNy45NTIgMy41LDcuODAyNjY2NjcgMi45NCw3LjYxNiBDMS45OTczMzMzMyw3LjMxNzMzMzMzIDEuMzMyMzMzMzMsNy4wMDkzMzMzMyAwLjk0NSw2LjY5MiBDMC41NTc2NjY2NjcsNi4zNzQ2NjY2NyAwLjM2NCw1Ljk1OTMzMzMzIDAuMzY0LDUuNDQ2IEMwLjM2NCw0Ljg5NTMzMzMzIDAuNjA0MzMzMzMzLDQuNDU0MzMzMzMgMS4wODUsNC4xMjMgQzEuNTY1NjY2NjcsMy43OTE2NjY2NyAyLjIxNjY2NjY3LDMuNjI2IDMuMDM4LDMuNjI2IEMzLjY0NDY2NjY3LDMuNjI2IDQuMjA3LDMuNzI2MzMzMzMgNC43MjUsMy45MjcgQzUuMjQzLDQuMTI3NjY2NjcgNS42MjgsNC40MDA2NjY2NyA1Ljg4LDQuNzQ2IEM1Ljk2NCw0Ljg1OCA2LjAwNiw0Ljk3NDY2NjY3IDYuMDA2LDUuMDk2IEM2LjAwNiw1LjI1NDY2NjY3IDUuOTIyLDUuMzkgNS43NTQsNS41MDIgQzUuNjMyNjY2NjcsNS41NzY2NjY2NyA1LjUxMTMzMzMzLDUuNjE0IDUuMzksNS42MTQgQzUuMjAzMzMzMzMsNS42MTQgNS4wMjYsNS41MyA0Ljg1OCw1LjM2MiBDNC42MzQsNS4xMzggNC4zNzUsNC45NzIzMzMzMyA0LjA4MSw0Ljg2NSBDMy43ODcsNC43NTc2NjY2NyAzLjQzNDY2NjY3LDQuNzA0IDMuMDI0LDQuNzA0IEMyLjU4NTMzMzMzLDQuNzA0IDIuMjUxNjY2NjcsNC43NjkzMzMzMyAyLjAyMyw0LjkgQzEuNzk0MzMzMzMsNS4wMzA2NjY2NyAxLjY4LDUuMjE3MzMzMzMgMS42OCw1LjQ2IEMxLjY4LDUuNjQ2NjY2NjcgMS43MzEzMzMzMyw1Ljc5NiAxLjgzNCw1LjkwOCBDMS45MzY2NjY2Nyw2LjAyIDIuMTE0LDYuMTI3MzMzMzMgMi4zNjYsNi4yMyBDMi42MTgsNi4zMzI2NjY2NyAzLjAzOCw2LjQ3NzMzMzMzIDMuNjI2LDYuNjY0IEM0LjI3OTMzMzMzLDYuODY5MzMzMzMgNC43OTAzMzMzMyw3LjA3NDY2NjY3IDUuMTU5LDcuMjggQzUuNTI3NjY2NjcsNy40ODUzMzMzMyA1Ljc5MTMzMzMzLDcuNzE0IDUuOTUsNy45NjYgQzYuMTA4NjY2NjcsOC4yMTggNi4xODgsOC41MjEzMzMzMyA2LjE4OCw4Ljg3NiBDNi4xODgsOS40NDUzMzMzMyA1LjkzMzY2NjY3LDkuOTAyNjY2NjcgNS40MjUsMTAuMjQ4IEM0LjkxNjMzMzMzLDEwLjU5MzMzMzMgNC4yMzczMzMzMywxMC43NjYgMy4zODgsMTAuNzY2IFoiIGlkPSJzIiAvPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTExLjA4OCwxMC43NjYgQzEwLjMzMiwxMC43NjYgOS42NzQsMTAuNjIzNjY2NyA5LjExNCwxMC4zMzkgQzguNTU0LDEwLjA1NDMzMzMgOC4xMiw5LjY0NiA3LjgxMiw5LjExNCBDNy41MDQsOC41ODIgNy4zNSw3Ljk1MiA3LjM1LDcuMjI0IEM3LjM1LDYuNTI0IDcuNDk3LDUuOTAxIDcuNzkxLDUuMzU1IEM4LjA4NSw0LjgwOSA4LjUxNjY2NjY3LDQuMzg0MzMzMzMgOS4wODYsNC4wODEgQzkuNjU1MzMzMzMsMy43Nzc2NjY2NyAxMC4zMzIsMy42MjYgMTEuMTE2LDMuNjI2IEMxMS43NTA2NjY3LDMuNjI2IDEyLjI4OTY2NjcsMy43NDAzMzMzMyAxMi43MzMsMy45NjkgQzEzLjE3NjMzMzMsNC4xOTc2NjY2NyAxMy41OCw0LjUyMiAxMy45NDQsNC45NDIgTDEzLjk1OCwwLjY1OCBDMTMuOTU4LDAuNDUyNjY2NjY3IDE0LjAxNjMzMzMsMC4yOTE2NjY2NjcgMTQuMTMzLDAuMTc1IEMxNC4yNDk2NjY3LDAuMDU4MzMzMzMzMyAxNC40MTA2NjY3LDAgMTQuNjE2LDAgQzE0LjgxMiwwIDE0Ljk2NiwwLjA1ODMzMzMzMzMgMTUuMDc4LDAuMTc1IEMxNS4xOSwwLjI5MTY2NjY2NyAxNS4yNDYsMC40NTI2NjY2NjcgMTUuMjQ2LDAuNjU4IEwxNS4yNDYsMTAuMTA4IEMxNS4yNDYsMTAuMzEzMzMzMyAxNS4xOSwxMC40NzQzMzMzIDE1LjA3OCwxMC41OTEgQzE0Ljk2NiwxMC43MDc2NjY3IDE0LjgxMiwxMC43NjYgMTQuNjE2LDEwLjc2NiBDMTQuNDEwNjY2NywxMC43NjYgMTQuMjQ5NjY2NywxMC43MDc2NjY3IDE0LjEzMywxMC41OTEgQzE0LjAxNjMzMzMsMTAuNDc0MzMzMyAxMy45NTgsMTAuMzEzMzMzMyAxMy45NTgsMTAuMTA4IEwxMy45NTgsOS40MjIgQzEzLjMwNDY2NjcsMTAuMzE4IDEyLjM0OCwxMC43NjYgMTEuMDg4LDEwLjc2NiBaIE0xMS4yMjgsOS42ODggQzEyLjA3NzMzMzMsOS42ODggMTIuNzQ0NjY2Nyw5LjQ2NjMzMzMzIDEzLjIzLDkuMDIzIEMxMy43MTUzMzMzLDguNTc5NjY2NjcgMTMuOTU4LDcuOTcwNjY2NjcgMTMuOTU4LDcuMTk2IEMxMy45NTgsNi40MzA2NjY2NyAxMy43MDEzMzMzLDUuODI0IDEzLjE4OCw1LjM3NiBDMTIuNjc0NjY2Nyw0LjkyOCAxMi4wMjYsNC43MDQgMTEuMjQyLDQuNzA0IEMxMC40Myw0LjcwNCA5Ljc5NTMzMzMzLDQuOTMyNjY2NjcgOS4zMzgsNS4zOSBDOC44ODA2NjY2Nyw1Ljg0NzMzMzMzIDguNjUyLDYuNDU4NjY2NjcgOC42NTIsNy4yMjQgQzguNjUyLDguMDI2NjY2NjcgOC44NzgzMzMzMyw4LjYzOCA5LjMzMSw5LjA1OCBDOS43ODM2NjY2Nyw5LjQ3OCAxMC40MTYsOS42ODggMTEuMjI4LDkuNjg4IFoiIGlkPSJkIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC45NTIzODEsMTAgTDE5LjA0NzYxOTEsMTAgQzE2LjgxMzMzMzMsMTAgMTUsMTEuODEzMzMzMyAxNSwxNC4wNDc2MTkgTDE1LDM5Ljk1MjM4MSBDMTUsNDIuMTg2NjY2NyAxNi44MTMzMzMzLDQ0IDE5LjA0NzYxOTEsNDQgTDQ0Ljk1MjM4MSw0NCBDNDcuMTg2NjY2Nyw0NCA0OSw0Mi4xODY2NjY3IDQ5LDM5Ljk1MjM4MSBMNDksMTQuMDQ3NjE5IEM0OSwxMS44MTMzMzMzIDQ3LjE4NjY2NjcsMTAgNDQuOTUyMzgxLDEwIFogTTQwLjkwNDc2MTksMzguMzMzMzMzMyBMMjMuMDk1MjM4MSwzOC4zMzMzMzMzIEwyMy4wOTUyMzgxLDM2LjcxNDI4NTcgTDQwLjkwNDc2MTksMzYuNzE0Mjg1NyBMNDAuOTA0NzYxOSwzOC4zMzMzMzMzIFogTTI0LjI4NTIzODEsMzMuMzc5MDQ3NiBMMjMuNTI0Mjg1NywzMS45NTQyODU3IEwzNy41NDUyMzgxLDI0LjU3MTQyODYgTDIzLjUyNDI4NTcsMTcuMTg4NTcxNCBMMjQuMjg1MjM4MSwxNS43NjM4MDk1IEw0MS4wMjYxOTA1LDI0LjU3MTQyODYgTDI0LjI4NTIzODEsMzMuMzc5MDQ3NiBaIiBpZD0iZ3JlYXRlcl90aGFuX2VxdWFsIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_sd_ge()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">2</td>
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


The validation result shows whether the standard deviation comparison passed or failed. Since this is an aggregation-based validation, there is exactly one test unit per column.

When validating multiple columns, each column gets its own validation step:


``` python
validation = (
    pb.Validate(data=tbl)
    .col_sd_ge(columns=["a", "b"], value=2)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc2RfZ2U8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3NkX2dlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxLjUwMDAwMCwgMS41MDAwMDApIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICA8cGF0aCBkPSJNNTUsMCBDNTcuNDg1MjgxMywwIDU5LjczNTI4MTMsMS4wMDczNTkzMSA2MS4zNjM5NjEsMi42MzYwMzg5NyBDNjIuOTkyNjQwNyw0LjI2NDcxODYzIDY0LDYuNTE0NzE4NjMgNjQsOSBMNjQsOSBMNjQsNjQgTDksNjQgQzYuNTE0NzE4NjIsNjQgNC4yNjQ3MTg2Miw2Mi45OTI2NDA3IDIuNjM2MDM4OTcsNjEuMzYzOTYxIEMxLjAwNzM1OTMxLDU5LjczNTI4MTQgMCw1Ny40ODUyODE0IDAsNTUgTDAsNTUgTDAsOSBDMCw2LjUxNDcxODYzIDEuMDA3MzU5MzEsNC4yNjQ3MTg2MyAyLjYzNjAzODk3LDIuNjM2MDM4OTcgQzQuMjY0NzE4NjIsMS4wMDczNTkzMSA2LjUxNDcxODYyLDAgOSwwIEw5LDAgTDU1LDAgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJzZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjQuMjQ0MDAwLCA0Ni4zNzQwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCwxMC43NjYgQzIuNjY5MzMzMzMsMTAuNzY2IDIuMDMsMTAuNjQyMzMzMyAxLjQ3LDEwLjM5NSBDMC45MSwxMC4xNDc2NjY3IDAuNDY2NjY2NjY3LDkuODE0IDAuMTQsOS4zOTQgQzAuMDQ2NjY2NjY2Nyw5LjI3MjY2NjY3IDAsOS4xNTEzMzMzMyAwLDkuMDMgQzAsOC44MjQ2NjY2NyAwLjA5OCw4LjY3MDY2NjY3IDAuMjk0LDguNTY4IEMwLjM3OCw4LjUyMTMzMzMzIDAuNDY2NjY2NjY3LDguNDk4IDAuNTYsOC40OTggQzAuNzE4NjY2NjY3LDguNDk4IDAuODgyLDguNTgyIDEuMDUsOC43NSBDMS4zNTgsOS4wNzY2NjY2NyAxLjY5ODY2NjY3LDkuMzE0NjY2NjcgMi4wNzIsOS40NjQgQzIuNDQ1MzMzMzMsOS42MTMzMzMzMyAyLjg3NDY2NjY3LDkuNjg4IDMuMzYsOS42ODggQzMuODA4LDkuNjg4IDQuMTc0MzMzMzMsOS42MTU2NjY2NyA0LjQ1OSw5LjQ3MSBDNC43NDM2NjY2Nyw5LjMyNjMzMzMzIDQuODg2LDkuMTI4IDQuODg2LDguODc2IEM0Ljg4Niw4LjY4OTMzMzMzIDQuODMyMzMzMzMsOC41MzUzMzMzMyA0LjcyNSw4LjQxNCBDNC42MTc2NjY2Nyw4LjI5MjY2NjY3IDQuNDMzMzMzMzMsOC4xNzYgNC4xNzIsOC4wNjQgQzMuOTEwNjY2NjcsNy45NTIgMy41LDcuODAyNjY2NjcgMi45NCw3LjYxNiBDMS45OTczMzMzMyw3LjMxNzMzMzMzIDEuMzMyMzMzMzMsNy4wMDkzMzMzMyAwLjk0NSw2LjY5MiBDMC41NTc2NjY2NjcsNi4zNzQ2NjY2NyAwLjM2NCw1Ljk1OTMzMzMzIDAuMzY0LDUuNDQ2IEMwLjM2NCw0Ljg5NTMzMzMzIDAuNjA0MzMzMzMzLDQuNDU0MzMzMzMgMS4wODUsNC4xMjMgQzEuNTY1NjY2NjcsMy43OTE2NjY2NyAyLjIxNjY2NjY3LDMuNjI2IDMuMDM4LDMuNjI2IEMzLjY0NDY2NjY3LDMuNjI2IDQuMjA3LDMuNzI2MzMzMzMgNC43MjUsMy45MjcgQzUuMjQzLDQuMTI3NjY2NjcgNS42MjgsNC40MDA2NjY2NyA1Ljg4LDQuNzQ2IEM1Ljk2NCw0Ljg1OCA2LjAwNiw0Ljk3NDY2NjY3IDYuMDA2LDUuMDk2IEM2LjAwNiw1LjI1NDY2NjY3IDUuOTIyLDUuMzkgNS43NTQsNS41MDIgQzUuNjMyNjY2NjcsNS41NzY2NjY2NyA1LjUxMTMzMzMzLDUuNjE0IDUuMzksNS42MTQgQzUuMjAzMzMzMzMsNS42MTQgNS4wMjYsNS41MyA0Ljg1OCw1LjM2MiBDNC42MzQsNS4xMzggNC4zNzUsNC45NzIzMzMzMyA0LjA4MSw0Ljg2NSBDMy43ODcsNC43NTc2NjY2NyAzLjQzNDY2NjY3LDQuNzA0IDMuMDI0LDQuNzA0IEMyLjU4NTMzMzMzLDQuNzA0IDIuMjUxNjY2NjcsNC43NjkzMzMzMyAyLjAyMyw0LjkgQzEuNzk0MzMzMzMsNS4wMzA2NjY2NyAxLjY4LDUuMjE3MzMzMzMgMS42OCw1LjQ2IEMxLjY4LDUuNjQ2NjY2NjcgMS43MzEzMzMzMyw1Ljc5NiAxLjgzNCw1LjkwOCBDMS45MzY2NjY2Nyw2LjAyIDIuMTE0LDYuMTI3MzMzMzMgMi4zNjYsNi4yMyBDMi42MTgsNi4zMzI2NjY2NyAzLjAzOCw2LjQ3NzMzMzMzIDMuNjI2LDYuNjY0IEM0LjI3OTMzMzMzLDYuODY5MzMzMzMgNC43OTAzMzMzMyw3LjA3NDY2NjY3IDUuMTU5LDcuMjggQzUuNTI3NjY2NjcsNy40ODUzMzMzMyA1Ljc5MTMzMzMzLDcuNzE0IDUuOTUsNy45NjYgQzYuMTA4NjY2NjcsOC4yMTggNi4xODgsOC41MjEzMzMzMyA2LjE4OCw4Ljg3NiBDNi4xODgsOS40NDUzMzMzMyA1LjkzMzY2NjY3LDkuOTAyNjY2NjcgNS40MjUsMTAuMjQ4IEM0LjkxNjMzMzMzLDEwLjU5MzMzMzMgNC4yMzczMzMzMywxMC43NjYgMy4zODgsMTAuNzY2IFoiIGlkPSJzIiAvPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTExLjA4OCwxMC43NjYgQzEwLjMzMiwxMC43NjYgOS42NzQsMTAuNjIzNjY2NyA5LjExNCwxMC4zMzkgQzguNTU0LDEwLjA1NDMzMzMgOC4xMiw5LjY0NiA3LjgxMiw5LjExNCBDNy41MDQsOC41ODIgNy4zNSw3Ljk1MiA3LjM1LDcuMjI0IEM3LjM1LDYuNTI0IDcuNDk3LDUuOTAxIDcuNzkxLDUuMzU1IEM4LjA4NSw0LjgwOSA4LjUxNjY2NjY3LDQuMzg0MzMzMzMgOS4wODYsNC4wODEgQzkuNjU1MzMzMzMsMy43Nzc2NjY2NyAxMC4zMzIsMy42MjYgMTEuMTE2LDMuNjI2IEMxMS43NTA2NjY3LDMuNjI2IDEyLjI4OTY2NjcsMy43NDAzMzMzMyAxMi43MzMsMy45NjkgQzEzLjE3NjMzMzMsNC4xOTc2NjY2NyAxMy41OCw0LjUyMiAxMy45NDQsNC45NDIgTDEzLjk1OCwwLjY1OCBDMTMuOTU4LDAuNDUyNjY2NjY3IDE0LjAxNjMzMzMsMC4yOTE2NjY2NjcgMTQuMTMzLDAuMTc1IEMxNC4yNDk2NjY3LDAuMDU4MzMzMzMzMyAxNC40MTA2NjY3LDAgMTQuNjE2LDAgQzE0LjgxMiwwIDE0Ljk2NiwwLjA1ODMzMzMzMzMgMTUuMDc4LDAuMTc1IEMxNS4xOSwwLjI5MTY2NjY2NyAxNS4yNDYsMC40NTI2NjY2NjcgMTUuMjQ2LDAuNjU4IEwxNS4yNDYsMTAuMTA4IEMxNS4yNDYsMTAuMzEzMzMzMyAxNS4xOSwxMC40NzQzMzMzIDE1LjA3OCwxMC41OTEgQzE0Ljk2NiwxMC43MDc2NjY3IDE0LjgxMiwxMC43NjYgMTQuNjE2LDEwLjc2NiBDMTQuNDEwNjY2NywxMC43NjYgMTQuMjQ5NjY2NywxMC43MDc2NjY3IDE0LjEzMywxMC41OTEgQzE0LjAxNjMzMzMsMTAuNDc0MzMzMyAxMy45NTgsMTAuMzEzMzMzMyAxMy45NTgsMTAuMTA4IEwxMy45NTgsOS40MjIgQzEzLjMwNDY2NjcsMTAuMzE4IDEyLjM0OCwxMC43NjYgMTEuMDg4LDEwLjc2NiBaIE0xMS4yMjgsOS42ODggQzEyLjA3NzMzMzMsOS42ODggMTIuNzQ0NjY2Nyw5LjQ2NjMzMzMzIDEzLjIzLDkuMDIzIEMxMy43MTUzMzMzLDguNTc5NjY2NjcgMTMuOTU4LDcuOTcwNjY2NjcgMTMuOTU4LDcuMTk2IEMxMy45NTgsNi40MzA2NjY2NyAxMy43MDEzMzMzLDUuODI0IDEzLjE4OCw1LjM3NiBDMTIuNjc0NjY2Nyw0LjkyOCAxMi4wMjYsNC43MDQgMTEuMjQyLDQuNzA0IEMxMC40Myw0LjcwNCA5Ljc5NTMzMzMzLDQuOTMyNjY2NjcgOS4zMzgsNS4zOSBDOC44ODA2NjY2Nyw1Ljg0NzMzMzMzIDguNjUyLDYuNDU4NjY2NjcgOC42NTIsNy4yMjQgQzguNjUyLDguMDI2NjY2NjcgOC44NzgzMzMzMyw4LjYzOCA5LjMzMSw5LjA1OCBDOS43ODM2NjY2Nyw5LjQ3OCAxMC40MTYsOS42ODggMTEuMjI4LDkuNjg4IFoiIGlkPSJkIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC45NTIzODEsMTAgTDE5LjA0NzYxOTEsMTAgQzE2LjgxMzMzMzMsMTAgMTUsMTEuODEzMzMzMyAxNSwxNC4wNDc2MTkgTDE1LDM5Ljk1MjM4MSBDMTUsNDIuMTg2NjY2NyAxNi44MTMzMzMzLDQ0IDE5LjA0NzYxOTEsNDQgTDQ0Ljk1MjM4MSw0NCBDNDcuMTg2NjY2Nyw0NCA0OSw0Mi4xODY2NjY3IDQ5LDM5Ljk1MjM4MSBMNDksMTQuMDQ3NjE5IEM0OSwxMS44MTMzMzMzIDQ3LjE4NjY2NjcsMTAgNDQuOTUyMzgxLDEwIFogTTQwLjkwNDc2MTksMzguMzMzMzMzMyBMMjMuMDk1MjM4MSwzOC4zMzMzMzMzIEwyMy4wOTUyMzgxLDM2LjcxNDI4NTcgTDQwLjkwNDc2MTksMzYuNzE0Mjg1NyBMNDAuOTA0NzYxOSwzOC4zMzMzMzMzIFogTTI0LjI4NTIzODEsMzMuMzc5MDQ3NiBMMjMuNTI0Mjg1NywzMS45NTQyODU3IEwzNy41NDUyMzgxLDI0LjU3MTQyODYgTDIzLjUyNDI4NTcsMTcuMTg4NTcxNCBMMjQuMjg1MjM4MSwxNS43NjM4MDk1IEw0MS4wMjYxOTA1LDI0LjU3MTQyODYgTDI0LjI4NTIzODEsMzMuMzc5MDQ3NiBaIiBpZD0iZ3JlYXRlcl90aGFuX2VxdWFsIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_sd_ge()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">2</td>
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
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc2RfZ2U8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3NkX2dlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxLjUwMDAwMCwgMS41MDAwMDApIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICA8cGF0aCBkPSJNNTUsMCBDNTcuNDg1MjgxMywwIDU5LjczNTI4MTMsMS4wMDczNTkzMSA2MS4zNjM5NjEsMi42MzYwMzg5NyBDNjIuOTkyNjQwNyw0LjI2NDcxODYzIDY0LDYuNTE0NzE4NjMgNjQsOSBMNjQsOSBMNjQsNjQgTDksNjQgQzYuNTE0NzE4NjIsNjQgNC4yNjQ3MTg2Miw2Mi45OTI2NDA3IDIuNjM2MDM4OTcsNjEuMzYzOTYxIEMxLjAwNzM1OTMxLDU5LjczNTI4MTQgMCw1Ny40ODUyODE0IDAsNTUgTDAsNTUgTDAsOSBDMCw2LjUxNDcxODYzIDEuMDA3MzU5MzEsNC4yNjQ3MTg2MyAyLjYzNjAzODk3LDIuNjM2MDM4OTcgQzQuMjY0NzE4NjIsMS4wMDczNTkzMSA2LjUxNDcxODYyLDAgOSwwIEw5LDAgTDU1LDAgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJzZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjQuMjQ0MDAwLCA0Ni4zNzQwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCwxMC43NjYgQzIuNjY5MzMzMzMsMTAuNzY2IDIuMDMsMTAuNjQyMzMzMyAxLjQ3LDEwLjM5NSBDMC45MSwxMC4xNDc2NjY3IDAuNDY2NjY2NjY3LDkuODE0IDAuMTQsOS4zOTQgQzAuMDQ2NjY2NjY2Nyw5LjI3MjY2NjY3IDAsOS4xNTEzMzMzMyAwLDkuMDMgQzAsOC44MjQ2NjY2NyAwLjA5OCw4LjY3MDY2NjY3IDAuMjk0LDguNTY4IEMwLjM3OCw4LjUyMTMzMzMzIDAuNDY2NjY2NjY3LDguNDk4IDAuNTYsOC40OTggQzAuNzE4NjY2NjY3LDguNDk4IDAuODgyLDguNTgyIDEuMDUsOC43NSBDMS4zNTgsOS4wNzY2NjY2NyAxLjY5ODY2NjY3LDkuMzE0NjY2NjcgMi4wNzIsOS40NjQgQzIuNDQ1MzMzMzMsOS42MTMzMzMzMyAyLjg3NDY2NjY3LDkuNjg4IDMuMzYsOS42ODggQzMuODA4LDkuNjg4IDQuMTc0MzMzMzMsOS42MTU2NjY2NyA0LjQ1OSw5LjQ3MSBDNC43NDM2NjY2Nyw5LjMyNjMzMzMzIDQuODg2LDkuMTI4IDQuODg2LDguODc2IEM0Ljg4Niw4LjY4OTMzMzMzIDQuODMyMzMzMzMsOC41MzUzMzMzMyA0LjcyNSw4LjQxNCBDNC42MTc2NjY2Nyw4LjI5MjY2NjY3IDQuNDMzMzMzMzMsOC4xNzYgNC4xNzIsOC4wNjQgQzMuOTEwNjY2NjcsNy45NTIgMy41LDcuODAyNjY2NjcgMi45NCw3LjYxNiBDMS45OTczMzMzMyw3LjMxNzMzMzMzIDEuMzMyMzMzMzMsNy4wMDkzMzMzMyAwLjk0NSw2LjY5MiBDMC41NTc2NjY2NjcsNi4zNzQ2NjY2NyAwLjM2NCw1Ljk1OTMzMzMzIDAuMzY0LDUuNDQ2IEMwLjM2NCw0Ljg5NTMzMzMzIDAuNjA0MzMzMzMzLDQuNDU0MzMzMzMgMS4wODUsNC4xMjMgQzEuNTY1NjY2NjcsMy43OTE2NjY2NyAyLjIxNjY2NjY3LDMuNjI2IDMuMDM4LDMuNjI2IEMzLjY0NDY2NjY3LDMuNjI2IDQuMjA3LDMuNzI2MzMzMzMgNC43MjUsMy45MjcgQzUuMjQzLDQuMTI3NjY2NjcgNS42MjgsNC40MDA2NjY2NyA1Ljg4LDQuNzQ2IEM1Ljk2NCw0Ljg1OCA2LjAwNiw0Ljk3NDY2NjY3IDYuMDA2LDUuMDk2IEM2LjAwNiw1LjI1NDY2NjY3IDUuOTIyLDUuMzkgNS43NTQsNS41MDIgQzUuNjMyNjY2NjcsNS41NzY2NjY2NyA1LjUxMTMzMzMzLDUuNjE0IDUuMzksNS42MTQgQzUuMjAzMzMzMzMsNS42MTQgNS4wMjYsNS41MyA0Ljg1OCw1LjM2MiBDNC42MzQsNS4xMzggNC4zNzUsNC45NzIzMzMzMyA0LjA4MSw0Ljg2NSBDMy43ODcsNC43NTc2NjY2NyAzLjQzNDY2NjY3LDQuNzA0IDMuMDI0LDQuNzA0IEMyLjU4NTMzMzMzLDQuNzA0IDIuMjUxNjY2NjcsNC43NjkzMzMzMyAyLjAyMyw0LjkgQzEuNzk0MzMzMzMsNS4wMzA2NjY2NyAxLjY4LDUuMjE3MzMzMzMgMS42OCw1LjQ2IEMxLjY4LDUuNjQ2NjY2NjcgMS43MzEzMzMzMyw1Ljc5NiAxLjgzNCw1LjkwOCBDMS45MzY2NjY2Nyw2LjAyIDIuMTE0LDYuMTI3MzMzMzMgMi4zNjYsNi4yMyBDMi42MTgsNi4zMzI2NjY2NyAzLjAzOCw2LjQ3NzMzMzMzIDMuNjI2LDYuNjY0IEM0LjI3OTMzMzMzLDYuODY5MzMzMzMgNC43OTAzMzMzMyw3LjA3NDY2NjY3IDUuMTU5LDcuMjggQzUuNTI3NjY2NjcsNy40ODUzMzMzMyA1Ljc5MTMzMzMzLDcuNzE0IDUuOTUsNy45NjYgQzYuMTA4NjY2NjcsOC4yMTggNi4xODgsOC41MjEzMzMzMyA2LjE4OCw4Ljg3NiBDNi4xODgsOS40NDUzMzMzMyA1LjkzMzY2NjY3LDkuOTAyNjY2NjcgNS40MjUsMTAuMjQ4IEM0LjkxNjMzMzMzLDEwLjU5MzMzMzMgNC4yMzczMzMzMywxMC43NjYgMy4zODgsMTAuNzY2IFoiIGlkPSJzIiAvPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTExLjA4OCwxMC43NjYgQzEwLjMzMiwxMC43NjYgOS42NzQsMTAuNjIzNjY2NyA5LjExNCwxMC4zMzkgQzguNTU0LDEwLjA1NDMzMzMgOC4xMiw5LjY0NiA3LjgxMiw5LjExNCBDNy41MDQsOC41ODIgNy4zNSw3Ljk1MiA3LjM1LDcuMjI0IEM3LjM1LDYuNTI0IDcuNDk3LDUuOTAxIDcuNzkxLDUuMzU1IEM4LjA4NSw0LjgwOSA4LjUxNjY2NjY3LDQuMzg0MzMzMzMgOS4wODYsNC4wODEgQzkuNjU1MzMzMzMsMy43Nzc2NjY2NyAxMC4zMzIsMy42MjYgMTEuMTE2LDMuNjI2IEMxMS43NTA2NjY3LDMuNjI2IDEyLjI4OTY2NjcsMy43NDAzMzMzMyAxMi43MzMsMy45NjkgQzEzLjE3NjMzMzMsNC4xOTc2NjY2NyAxMy41OCw0LjUyMiAxMy45NDQsNC45NDIgTDEzLjk1OCwwLjY1OCBDMTMuOTU4LDAuNDUyNjY2NjY3IDE0LjAxNjMzMzMsMC4yOTE2NjY2NjcgMTQuMTMzLDAuMTc1IEMxNC4yNDk2NjY3LDAuMDU4MzMzMzMzMyAxNC40MTA2NjY3LDAgMTQuNjE2LDAgQzE0LjgxMiwwIDE0Ljk2NiwwLjA1ODMzMzMzMzMgMTUuMDc4LDAuMTc1IEMxNS4xOSwwLjI5MTY2NjY2NyAxNS4yNDYsMC40NTI2NjY2NjcgMTUuMjQ2LDAuNjU4IEwxNS4yNDYsMTAuMTA4IEMxNS4yNDYsMTAuMzEzMzMzMyAxNS4xOSwxMC40NzQzMzMzIDE1LjA3OCwxMC41OTEgQzE0Ljk2NiwxMC43MDc2NjY3IDE0LjgxMiwxMC43NjYgMTQuNjE2LDEwLjc2NiBDMTQuNDEwNjY2NywxMC43NjYgMTQuMjQ5NjY2NywxMC43MDc2NjY3IDE0LjEzMywxMC41OTEgQzE0LjAxNjMzMzMsMTAuNDc0MzMzMyAxMy45NTgsMTAuMzEzMzMzMyAxMy45NTgsMTAuMTA4IEwxMy45NTgsOS40MjIgQzEzLjMwNDY2NjcsMTAuMzE4IDEyLjM0OCwxMC43NjYgMTEuMDg4LDEwLjc2NiBaIE0xMS4yMjgsOS42ODggQzEyLjA3NzMzMzMsOS42ODggMTIuNzQ0NjY2Nyw5LjQ2NjMzMzMzIDEzLjIzLDkuMDIzIEMxMy43MTUzMzMzLDguNTc5NjY2NjcgMTMuOTU4LDcuOTcwNjY2NjcgMTMuOTU4LDcuMTk2IEMxMy45NTgsNi40MzA2NjY2NyAxMy43MDEzMzMzLDUuODI0IDEzLjE4OCw1LjM3NiBDMTIuNjc0NjY2Nyw0LjkyOCAxMi4wMjYsNC43MDQgMTEuMjQyLDQuNzA0IEMxMC40Myw0LjcwNCA5Ljc5NTMzMzMzLDQuOTMyNjY2NjcgOS4zMzgsNS4zOSBDOC44ODA2NjY2Nyw1Ljg0NzMzMzMzIDguNjUyLDYuNDU4NjY2NjcgOC42NTIsNy4yMjQgQzguNjUyLDguMDI2NjY2NjcgOC44NzgzMzMzMyw4LjYzOCA5LjMzMSw5LjA1OCBDOS43ODM2NjY2Nyw5LjQ3OCAxMC40MTYsOS42ODggMTEuMjI4LDkuNjg4IFoiIGlkPSJkIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC45NTIzODEsMTAgTDE5LjA0NzYxOTEsMTAgQzE2LjgxMzMzMzMsMTAgMTUsMTEuODEzMzMzMyAxNSwxNC4wNDc2MTkgTDE1LDM5Ljk1MjM4MSBDMTUsNDIuMTg2NjY2NyAxNi44MTMzMzMzLDQ0IDE5LjA0NzYxOTEsNDQgTDQ0Ljk1MjM4MSw0NCBDNDcuMTg2NjY2Nyw0NCA0OSw0Mi4xODY2NjY3IDQ5LDM5Ljk1MjM4MSBMNDksMTQuMDQ3NjE5IEM0OSwxMS44MTMzMzMzIDQ3LjE4NjY2NjcsMTAgNDQuOTUyMzgxLDEwIFogTTQwLjkwNDc2MTksMzguMzMzMzMzMyBMMjMuMDk1MjM4MSwzOC4zMzMzMzMzIEwyMy4wOTUyMzgxLDM2LjcxNDI4NTcgTDQwLjkwNDc2MTksMzYuNzE0Mjg1NyBMNDAuOTA0NzYxOSwzOC4zMzMzMzMzIFogTTI0LjI4NTIzODEsMzMuMzc5MDQ3NiBMMjMuNTI0Mjg1NywzMS45NTQyODU3IEwzNy41NDUyMzgxLDI0LjU3MTQyODYgTDIzLjUyNDI4NTcsMTcuMTg4NTcxNCBMMjQuMjg1MjM4MSwxNS43NjM4MDk1IEw0MS4wMjYxOTA1LDI0LjU3MTQyODYgTDI0LjI4NTIzODEsMzMuMzc5MDQ3NiBaIiBpZD0iZ3JlYXRlcl90aGFuX2VxdWFsIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_sd_ge()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">2</td>
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


Using tolerance for flexible comparisons:


``` python
validation = (
    pb.Validate(data=tbl)
    .col_sd_ge(columns="a", value=2, tol=1.0)
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
zMzMgMS42OCw1LjQ2IEMxLjY4LDUuNjQ2NjY2NjcgMS43MzEzMzMzMyw1Ljc5NiAxLjgzNCw1LjkwOCBDMS45MzY2NjY2Nyw2LjAyIDIuMTE0LDYuMTI3MzMzMzMgMi4zNjYsNi4yMyBDMi42MTgsNi4zMzI2NjY2NyAzLjAzOCw2LjQ3NzMzMzMzIDMuNjI2LDYuNjY0IEM0LjI3OTMzMzMzLDYuODY5MzMzMzMgNC43OTAzMzMzMyw3LjA3NDY2NjY3IDUuMTU5LDcuMjggQzUuNTI3NjY2NjcsNy40ODUzMzMzMyA1Ljc5MTMzMzMzLDcuNzE0IDUuOTUsNy45NjYgQzYuMTA4NjY2NjcsOC4yMTggNi4xODgsOC41MjEzMzMzMyA2LjE4OCw4Ljg3NiBDNi4xODgsOS40NDUzMzMzMyA1LjkzMzY2NjY3LDkuOTAyNjY2NjcgNS40MjUsMTAuMjQ4IEM0LjkxNjMzMzMzLDEwLjU5MzMzMzMgNC4yMzczMzMzMywxMC43NjYgMy4zODgsMTAuNzY2IFoiIGlkPSJzIiAvPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTExLjA4OCwxMC43NjYgQzEwLjMzMiwxMC43NjYgOS42NzQsMTAuNjIzNjY2NyA5LjExNCwxMC4zMzkgQzguNTU0LDEwLjA1NDMzMzMgOC4xMiw5LjY0NiA3LjgxMiw5LjExNCBDNy41MDQsOC41ODIgNy4zNSw3Ljk1MiA3LjM1LDcuMjI0IEM3LjM1LDYuNTI0IDcuNDk3LDUuOTAxIDcuNzkxLDUuMzU1IEM4LjA4NSw0LjgwOSA4LjUxNjY2NjY3LDQuMzg0MzMzMzMgOS4wODYsNC4wODEgQzkuNjU1MzMzMzMsMy43Nzc2NjY2NyAxMC4zMzIsMy42MjYgMTEuMTE2LDMuNjI2IEMxMS43NTA2NjY3LDMuNjI2IDEyLjI4OTY2NjcsMy43NDAzMzMzMyAxMi43MzMsMy45NjkgQzEzLjE3NjMzMzMsNC4xOTc2NjY2NyAxMy41OCw0LjUyMiAxMy45NDQsNC45NDIgTDEzLjk1OCwwLjY1OCBDMTMuOTU4LDAuNDUyNjY2NjY3IDE0LjAxNjMzMzMsMC4yOTE2NjY2NjcgMTQuMTMzLDAuMTc1IEMxNC4yNDk2NjY3LDAuMDU4MzMzMzMzMyAxNC40MTA2NjY3LDAgMTQuNjE2LDAgQzE0LjgxMiwwIDE0Ljk2NiwwLjA1ODMzMzMzMzMgMTUuMDc4LDAuMTc1IEMxNS4xOSwwLjI5MTY2NjY2NyAxNS4yNDYsMC40NTI2NjY2NjcgMTUuMjQ2LDAuNjU4IEwxNS4yNDYsMTAuMTA4IEMxNS4yNDYsMTAuMzEzMzMzMyAxNS4xOSwxMC40NzQzMzMzIDE1LjA3OCwxMC41OTEgQzE0Ljk2NiwxMC43MDc2NjY3IDE0LjgxMiwxMC43NjYgMTQuNjE2LDEwLjc2NiBDMTQuNDEwNjY2NywxMC43NjYgMTQuMjQ5NjY2NywxMC43MDc2NjY3IDE0LjEzMywxMC41OTEgQzE0LjAxNjMzMzMsMTAuNDc0MzMzMyAxMy45NTgsMTAuMzEzMzMzMyAxMy45NTgsMTAuMTA4IEwxMy45NTgsOS40MjIgQzEzLjMwNDY2NjcsMTAuMzE4IDEyLjM0OCwxMC43NjYgMTEuMDg4LDEwLjc2NiBaIE0xMS4yMjgsOS42ODggQzEyLjA3NzMzMzMsOS42ODggMTIuNzQ0NjY2Nyw5LjQ2NjMzMzMzIDEzLjIzLDkuMDIzIEMxMy43MTUzMzMzLDguNTc5NjY2NjcgMTMuOTU4LDcuOTcwNjY2NjcgMTMuOTU4LDcuMTk2IEMxMy45NTgsNi40MzA2NjY2NyAxMy43MDEzMzMzLDUuODI0IDEzLjE4OCw1LjM3NiBDMTIuNjc0NjY2Nyw0LjkyOCAxMi4wMjYsNC43MDQgMTEuMjQyLDQuNzA0IEMxMC40Myw0LjcwNCA5Ljc5NTMzMzMzLDQuOTMyNjY2NjcgOS4zMzgsNS4zOSBDOC44ODA2NjY2Nyw1Ljg0NzMzMzMzIDguNjUyLDYuNDU4NjY2NjcgOC42NTIsNy4yMjQgQzguNjUyLDguMDI2NjY2NjcgOC44NzgzMzMzMyw4LjYzOCA5LjMzMSw5LjA1OCBDOS43ODM2NjY2Nyw5LjQ3OCAxMC40MTYsOS42ODggMTEuMjI4LDkuNjg4IFoiIGlkPSJkIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC45NTIzODEsMTAgTDE5LjA0NzYxOTEsMTAgQzE2LjgxMzMzMzMsMTAgMTUsMTEuODEzMzMzMyAxNSwxNC4wNDc2MTkgTDE1LDM5Ljk1MjM4MSBDMTUsNDIuMTg2NjY2NyAxNi44MTMzMzMzLDQ0IDE5LjA0NzYxOTEsNDQgTDQ0Ljk1MjM4MSw0NCBDNDcuMTg2NjY2Nyw0NCA0OSw0Mi4xODY2NjY3IDQ5LDM5Ljk1MjM4MSBMNDksMTQuMDQ3NjE5IEM0OSwxMS44MTMzMzMzIDQ3LjE4NjY2NjcsMTAgNDQuOTUyMzgxLDEwIFogTTQwLjkwNDc2MTksMzguMzMzMzMzMyBMMjMuMDk1MjM4MSwzOC4zMzMzMzMzIEwyMy4wOTUyMzgxLDM2LjcxNDI4NTcgTDQwLjkwNDc2MTksMzYuNzE0Mjg1NyBMNDAuOTA0NzYxOSwzOC4zMzMzMzMzIFogTTI0LjI4NTIzODEsMzMuMzc5MDQ3NiBMMjMuNTI0Mjg1NywzMS45NTQyODU3IEwzNy41NDUyMzgxLDI0LjU3MTQyODYgTDIzLjUyNDI4NTcsMTcuMTg4NTcxNCBMMjQuMjg1MjM4MSwxNS43NjM4MDk1IEw0MS4wMjYxOTA1LDI0LjU3MTQyODYgTDI0LjI4NTIzODEsMzMuMzc5MDQ3NiBaIiBpZD0iZ3JlYXRlcl90aGFuX2VxdWFsIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_sd_ge()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">2<br />
tol=1.0</td>
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
