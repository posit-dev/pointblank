## Validate.col_sum_gt()


Does the column sum satisfy a greater than comparison?


Usage

``` python
Validate.col_sum_gt(
    columns,
    value=None,
    tol=0,
    thresholds=None,
    brief=False,
    actions=None,
    active=True
)
```


The [col_sum_gt()](Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt) validation method checks whether the sum of values in a column is greater than a specified `value=`. This is an aggregation-based validation where the entire column is reduced to a single sum value that is then compared against the target. The comparison used in this function is `sum(column) > value`.

Unlike row-level validations (e.g., [col_vals_gt()](Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)), this method treats the entire column as a single test unit. The validation either passes completely (if the aggregated value satisfies the comparison) or fails completely.


## Parameters


`columns: _PBUnresolvedColumn`  
A single column or a list of columns to validate. If multiple columns are supplied, there will be a separate validation step generated for each column. The columns must contain numeric data for the sum to be computed.

`value: float | Column | ReferenceColumn | None = None`  
The value to compare the column sum against. This can be: (1) a numeric literal (`int` or `float`), (2) a <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a> object referencing another column whose sum will be used for comparison, (3) a <a href="ref.html#pointblank.ref" class="gdls-link"><code>ref()</code></a> object referencing a column in reference data (when `Validate(reference=)` has been set), or (4) `None` to automatically compare against the same column in reference data (shorthand for `ref(column_name)` when reference data is set).

`tol: Tolerance = ``0`  
A tolerance value for the comparison. The default is `0`, meaning exact comparison. When set to a positive value, the comparison becomes more lenient. For example, with `tol=0.5`, a sum that differs from the target by up to `0.5` will still pass. The `tol=` parameter expands the acceptable range for the comparison. For [col_sum_gt()](Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt), a tolerance of `tol=0.5` would mean the sum can be within `0.5` of the target value and still pass validation.

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

The [col_sum_gt()](Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt) method supports comparing column aggregations against reference data. This is useful for validating that statistical properties remain consistent across different versions of a dataset, or for comparing current data against historical baselines.

To use reference data, set the `reference=` parameter when creating the [Validate](Validate.md#pointblank.Validate) object:

``` python
validation = (
    pb.Validate(data=current_data, reference=baseline_data)
    .col_sum_gt(columns="revenue")  # Compares sum(current.revenue) vs sum(baseline.revenue)
    .interrogate()
)
```

When `value=None` and reference data is set, the method automatically compares against the same column in the reference data. You can also explicitly specify reference columns using the [ref()](ref.md#pointblank.ref) helper:

``` python
.col_sum_gt(columns="revenue", value=pb.ref("baseline_revenue"))
```


## Understanding Tolerance

The `tol=` parameter allows for fuzzy comparisons, which is especially important for floating-point aggregations where exact equality is often unreliable.

The `tol=` parameter expands the acceptable range for the comparison. For [col_sum_gt()](Validate.col_sum_gt.md#pointblank.Validate.col_sum_gt), a tolerance of `tol=0.5` would mean the sum can be within `0.5` of the target value and still pass validation.

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


Let's validate that the sum of column `a` is greater than `15`:


``` python
validation = (
    pb.Validate(data=tbl)
    .col_sum_gt(columns="a", value=15)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc3VtX2d0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9zdW1fZ3QiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InN1bSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTguODg5MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCw3LjE0IEMyLjY2OTMzMzMzLDcuMTQgMi4wMyw3LjAxNjMzMzMzIDEuNDcsNi43NjkgQzAuOTEsNi41MjE2NjY2NyAwLjQ2NjY2NjY2Nyw2LjE4OCAwLjE0LDUuNzY4IEMwLjA0NjY2NjY2NjcsNS42NDY2NjY2NyAwLDUuNTI1MzMzMzMgMCw1LjQwNCBDMCw1LjE5ODY2NjY3IDAuMDk4LDUuMDQ0NjY2NjcgMC4yOTQsNC45NDIgQzAuMzc4LDQuODk1MzMzMzMgMC40NjY2NjY2NjcsNC44NzIgMC41Niw0Ljg3MiBDMC43MTg2NjY2NjcsNC44NzIgMC44ODIsNC45NTYgMS4wNSw1LjEyNCBDMS4zNTgsNS40NTA2NjY2NyAxLjY5ODY2NjY3LDUuNjg4NjY2NjcgMi4wNzIsNS44MzggQzIuNDQ1MzMzMzMsNS45ODczMzMzMyAyLjg3NDY2NjY3LDYuMDYyIDMuMzYsNi4wNjIgQzMuODA4LDYuMDYyIDQuMTc0MzMzMzMsNS45ODk2NjY2NyA0LjQ1OSw1Ljg0NSBDNC43NDM2NjY2Nyw1LjcwMDMzMzMzIDQuODg2LDUuNTAyIDQuODg2LDUuMjUgQzQuODg2LDUuMDYzMzMzMzMgNC44MzIzMzMzMyw0LjkwOTMzMzMzIDQuNzI1LDQuNzg4IEM0LjYxNzY2NjY3LDQuNjY2NjY2NjcgNC40MzMzMzMzMyw0LjU1IDQuMTcyLDQuNDM4IEMzLjkxMDY2NjY3LDQuMzI2IDMuNSw0LjE3NjY2NjY3IDIuOTQsMy45OSBDMS45OTczMzMzMywzLjY5MTMzMzMzIDEuMzMyMzMzMzMsMy4zODMzMzMzMyAwLjk0NSwzLjA2NiBDMC41NTc2NjY2NjcsMi43NDg2NjY2NyAwLjM2NCwyLjMzMzMzMzMzIDAuMzY0LDEuODIgQzAuMzY0LDEuMjY5MzMzMzMgMC42MDQzMzMzMzMsMC44MjgzMzMzMzMgMS4wODUsMC40OTcgQzEuNTY1NjY2NjcsMC4xNjU2NjY2NjcgMi4yMTY2NjY2NywwIDMuMDM4LDAgQzMuNjQ0NjY2NjcsMCA0LjIwNywwLjEwMDMzMzMzMyA0LjcyNSwwLjMwMSBDNS4yNDMsMC41MDE2NjY2NjcgNS42MjgsMC43NzQ2NjY2NjcgNS44OCwxLjEyIEM1Ljk2NCwxLjIzMiA2LjAwNiwxLjM0ODY2NjY3IDYuMDA2LDEuNDcgQzYuMDA2LDEuNjI4NjY2NjcgNS45MjIsMS43NjQgNS43NTQsMS44NzYgQzUuNjMyNjY2NjcsMS45NTA2NjY2NyA1LjUxMTMzMzMzLDEuOTg4IDUuMzksMS45ODggQzUuMjAzMzMzMzMsMS45ODggNS4wMjYsMS45MDQgNC44NTgsMS43MzYgQzQuNjM0LDEuNTEyIDQuMzc1LDEuMzQ2MzMzMzMgNC4wODEsMS4yMzkgQzMuNzg3LDEuMTMxNjY2NjcgMy40MzQ2NjY2NywxLjA3OCAzLjAyNCwxLjA3OCBDMi41ODUzMzMzMywxLjA3OCAyLjI1MTY2NjY3LDEuMTQzMzMzMzMgMi4wMjMsMS4yNzQgQzEuNzk0MzMzMzMsMS40MDQ2NjY2NyAxLjY4LDEuNTkxMzMzMzMgMS42OCwxLjgzNCBDMS42OCwyLjAyMDY2NjY3IDEuNzMxMzMzMzMsMi4xNyAxLjgzNCwyLjI4MiBDMS45MzY2NjY2NywyLjM5NCAyLjExNCwyLjUwMTMzMzMzIDIuMzY2LDIuNjA0IEMyLjYxOCwyLjcwNjY2NjY3IDMuMDM4LDIuODUxMzMzMzMgMy42MjYsMy4wMzggQzQuMjc5MzMzMzMsMy4yNDMzMzMzMyA0Ljc5MDMzMzMzLDMuNDQ4NjY2NjcgNS4xNTksMy42NTQgQzUuNTI3NjY2NjcsMy44NTkzMzMzMyA1Ljc5MTMzMzMzLDQuMDg4IDUuOTUsNC4zNCBDNi4xMDg2NjY2Nyw0LjU5MiA2LjE4OCw0Ljg5NTMzMzMzIDYuMTg4LDUuMjUgQzYuMTg4LDUuODE5MzMzMzMgNS45MzM2NjY2Nyw2LjI3NjY2NjY3IDUuNDI1LDYuNjIyIEM0LjkxNjMzMzMzLDYuOTY3MzMzMzMgNC4yMzczMzMzMyw3LjE0IDMuMzg4LDcuMTQgWiIgaWQ9InMiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTAuNTU2LDcuMTQgQzkuNjIyNjY2NjcsNy4xNCA4Ljg5NDY2NjY3LDYuODcxNjY2NjcgOC4zNzIsNi4zMzUgQzcuODQ5MzMzMzMsNS43OTgzMzMzMyA3LjU4OCw1LjA1ODY2NjY3IDcuNTg4LDQuMTE2IEw3LjU4OCwwLjY1OCBDNy41ODgsMC40NTI2NjY2NjcgNy42NDYzMzMzMywwLjI5MTY2NjY2NyA3Ljc2MywwLjE3NSBDNy44Nzk2NjY2NywwLjA1ODMzMzMzMzMgOC4wMzYsMCA4LjIzMiwwIEM4LjQzNzMzMzMzLDAgOC41OTYsMC4wNTgzMzMzMzMzIDguNzA4LDAuMTc1IEM4LjgyLDAuMjkxNjY2NjY3IDguODc2LDAuNDUyNjY2NjY3IDguODc2LDAuNjU4IEw4Ljg3Niw0LjExNiBDOC44NzYsNC43MzIgOS4wNDE2NjY2Nyw1LjIxMDMzMzMzIDkuMzczLDUuNTUxIEM5LjcwNDMzMzMzLDUuODkxNjY2NjcgMTAuMTY4NjY2Nyw2LjA2MiAxMC43NjYsNi4wNjIgQzExLjQ5NCw2LjA2MiAxMi4wNzk2NjY3LDUuODI4NjY2NjcgMTIuNTIzLDUuMzYyIEMxMi45NjYzMzMzLDQuODk1MzMzMzMgMTMuMTg4LDQuMjc0NjY2NjcgMTMuMTg4LDMuNSBMMTMuMTg4LDAuNjU4IEMxMy4xODgsMC40NTI2NjY2NjcgMTMuMjQ2MzMzMywwLjI5MTY2NjY2NyAxMy4zNjMsMC4xNzUgQzEzLjQ3OTY2NjcsMC4wNTgzMzMzMzMzIDEzLjY0MDY2NjcsMCAxMy44NDYsMCBDMTQuMDUxMzMzMywwIDE0LjIwNzY2NjcsMC4wNTYgMTQuMzE1LDAuMTY4IEMxNC40MjIzMzMzLDAuMjggMTQuNDc2LDAuNDQzMzMzMzMzIDE0LjQ3NiwwLjY1OCBMMTQuNDc2LDYuNDgyIEMxNC40NzYsNi42OTY2NjY2NyAxNC40MjIzMzMzLDYuODYgMTQuMzE1LDYuOTcyIEMxNC4yMDc2NjY3LDcuMDg0IDE0LjA1MTMzMzMsNy4xNCAxMy44NDYsNy4xNCBDMTMuNjQwNjY2Nyw3LjE0IDEzLjQ4Miw3LjA4MTY2NjY3IDEzLjM3LDYuOTY1IEMxMy4yNTgsNi44NDgzMzMzMyAxMy4yMDIsNi42ODczMzMzMyAxMy4yMDIsNi40ODIgTDEzLjIwMiw1Ljc5NiBDMTIuNTAyLDYuNjkyIDExLjYyLDcuMTQgMTAuNTU2LDcuMTQgWiIgaWQ9InUiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTYuODQyLDcuMTQgQzE2LjYzNjY2NjcsNy4xNCAxNi40NzgsNy4wODE2NjY2NyAxNi4zNjYsNi45NjUgQzE2LjI1NCw2Ljg0ODMzMzMzIDE2LjE5OCw2LjY4NzMzMzMzIDE2LjE5OCw2LjQ4MiBMMTYuMTk4LDAuNjU4IEMxNi4xOTgsMC40NTI2NjY2NjcgMTYuMjU0LDAuMjkxNjY2NjY3IDE2LjM2NiwwLjE3NSBDMTYuNDc4LDAuMDU4MzMzMzMzMyAxNi42MzY2NjY3LDAgMTYuODQyLDAgQzE3LjA0NzMzMzMsMCAxNy4yMDgzMzMzLDAuMDU4MzMzMzMzMyAxNy4zMjUsMC4xNzUgQzE3LjQ0MTY2NjcsMC4yOTE2NjY2NjcgMTcuNSwwLjQ1MjY2NjY2NyAxNy41LDAuNjU4IEwxNy41LDEuMzAyIEMxNy43MzMzMzMzLDAuODkxMzMzMzMzIDE4LjAzNjY2NjcsMC41NzE2NjY2NjcgMTguNDEsMC4zNDMgQzE4Ljc4MzMzMzMsMC4xMTQzMzMzMzMgMTkuMTgsMCAxOS42LDAgQzIwLjAyOTMzMzMsMCAyMC40MDczMzMzLDAuMTE0MzMzMzMzIDIwLjczNCwwLjM0MyBDMjEuMDYwNjY2NywwLjU3MTY2NjY2NyAyMS4zMDMzMzMzLDAuODkxMzMzMzMzIDIxLjQ2MiwxLjMwMiBDMjIuMTQzMzMzMywwLjQzNCAyMi45MTMzMzMzLDAgMjMuNzcyLDAgQzI0LjUwOTMzMzMsMCAyNS4wNzE2NjY3LDAuMjM4IDI1LjQ1OSwwLjcxNCBDMjUuODQ2MzMzMywxLjE5IDI2LjA0LDEuODgwNjY2NjcgMjYuMDQsMi43ODYgTDI2LjA0LDYuNDgyIEMyNi4wNCw2LjY4NzMzMzMzIDI1Ljk4MTY2NjcsNi44NDgzMzMzMyAyNS44NjUsNi45NjUgQzI1Ljc0ODMzMzMsNy4wODE2NjY2NyAyNS41OTIsNy4xNCAyNS4zOTYsNy4xNCBDMjUuMTkwNjY2Nyw3LjE0IDI1LjAzMiw3LjA4NCAyNC45Miw2Ljk3MiBDMjQuODA4LDYuODYgMjQuNzUyLDYuNjk2NjY2NjcgMjQuNzUyLDYuNDgyIEwyNC43NTIsMi43MyBDMjQuNzUyLDIuMTc5MzMzMzMgMjQuNjU4NjY2NywxLjc2NjMzMzMzIDI0LjQ3MiwxLjQ5MSBDMjQuMjg1MzMzMywxLjIxNTY2NjY3IDIzLjk3NzMzMzMsMS4wNzggMjMuNTQ4LDEuMDc4IEMyMi44MDEzMzMzLDEuMDc4IDIyLjIzNjY2NjcsMS41MDI2NjY2NyAyMS44NTQsMi4zNTIgTDIxLjg1NCw2LjQ4MiBDMjEuODU0LDYuNjg3MzMzMzMgMjEuNzk1NjY2Nyw2Ljg0ODMzMzMzIDIxLjY3OSw2Ljk2NSBDMjEuNTYyMzMzMyw3LjA4MTY2NjY3IDIxLjQwMTMzMzMsNy4xNCAyMS4xOTYsNy4xNCBDMjAuOTkwNjY2Nyw3LjE0IDIwLjgzNDMzMzMsNy4wODQgMjAuNzI3LDYuOTcyIEMyMC42MTk2NjY3LDYuODYgMjAuNTY2LDYuNjk2NjY2NjcgMjAuNTY2LDYuNDgyIEwyMC41NjYsMi45NjggQzIwLjU2NiwyLjM1MiAyMC40NjMzMzMzLDEuODgzIDIwLjI1OCwxLjU2MSBDMjAuMDUyNjY2NywxLjIzOSAxOS43NTQsMS4wNzggMTkuMzYyLDEuMDc4IEMxOC44MTEzMzMzLDEuMDc4IDE4LjM2MzMzMzMsMS4yOTUgMTguMDE4LDEuNzI5IEMxNy42NzI2NjY3LDIuMTYzIDE3LjUsMi43MjUzMzMzMyAxNy41LDMuNDE2IEwxNy41LDYuNDgyIEMxNy41LDYuNjg3MzMzMzMgMTcuNDQxNjY2Nyw2Ljg0ODMzMzMzIDE3LjMyNSw2Ljk2NSBDMTcuMjA4MzMzMyw3LjA4MTY2NjY3IDE3LjA0NzMzMzMsNy4xNCAxNi44NDIsNy4xNCBaIiBpZD0ibSIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuOTUyMzgxLDEwIEwxOS4wNDc2MTkxLDEwIEMxNi44MTMzMzMzLDEwIDE1LDExLjgxMzMzMzMgMTUsMTQuMDQ3NjE5IEwxNSwzOS45NTIzODEgQzE1LDQyLjE4NjY2NjcgMTYuODEzMzMzMyw0NCAxOS4wNDc2MTkxLDQ0IEw0NC45NTIzODEsNDQgQzQ3LjE4NjY2NjcsNDQgNDksNDIuMTg2NjY2NyA0OSwzOS45NTIzODEgTDQ5LDE0LjA0NzYxOSBDNDksMTEuODEzMzMzMyA0Ny4xODY2NjY3LDEwIDQ0Ljk1MjM4MSwxMCBaIE0yNi43OTQ3NjE5LDM2LjU2ODU3MTQgTDI1Ljg3MTkwNDgsMzUuMjQwOTUyNCBMMzcuODYwOTUyNCwyNyBMMjUuODcxOTA0OCwxOC43NTkwNDc2IEwyNi43OTQ3NjE5LDE3LjQzMTQyODYgTDQwLjcxODU3MTQsMjcgTDI2Ljc5NDc2MTksMzYuNTY4NTcxNCBaIiBpZD0iZ3JlYXRlcl90aGFuIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_sum_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">15</td>
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


The validation result shows whether the sum comparison passed or failed. Since this is an aggregation-based validation, there is exactly one test unit per column.

When validating multiple columns, each column gets its own validation step:


``` python
validation = (
    pb.Validate(data=tbl)
    .col_sum_gt(columns=["a", "b"], value=15)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc3VtX2d0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9zdW1fZ3QiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InN1bSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTguODg5MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCw3LjE0IEMyLjY2OTMzMzMzLDcuMTQgMi4wMyw3LjAxNjMzMzMzIDEuNDcsNi43NjkgQzAuOTEsNi41MjE2NjY2NyAwLjQ2NjY2NjY2Nyw2LjE4OCAwLjE0LDUuNzY4IEMwLjA0NjY2NjY2NjcsNS42NDY2NjY2NyAwLDUuNTI1MzMzMzMgMCw1LjQwNCBDMCw1LjE5ODY2NjY3IDAuMDk4LDUuMDQ0NjY2NjcgMC4yOTQsNC45NDIgQzAuMzc4LDQuODk1MzMzMzMgMC40NjY2NjY2NjcsNC44NzIgMC41Niw0Ljg3MiBDMC43MTg2NjY2NjcsNC44NzIgMC44ODIsNC45NTYgMS4wNSw1LjEyNCBDMS4zNTgsNS40NTA2NjY2NyAxLjY5ODY2NjY3LDUuNjg4NjY2NjcgMi4wNzIsNS44MzggQzIuNDQ1MzMzMzMsNS45ODczMzMzMyAyLjg3NDY2NjY3LDYuMDYyIDMuMzYsNi4wNjIgQzMuODA4LDYuMDYyIDQuMTc0MzMzMzMsNS45ODk2NjY2NyA0LjQ1OSw1Ljg0NSBDNC43NDM2NjY2Nyw1LjcwMDMzMzMzIDQuODg2LDUuNTAyIDQuODg2LDUuMjUgQzQuODg2LDUuMDYzMzMzMzMgNC44MzIzMzMzMyw0LjkwOTMzMzMzIDQuNzI1LDQuNzg4IEM0LjYxNzY2NjY3LDQuNjY2NjY2NjcgNC40MzMzMzMzMyw0LjU1IDQuMTcyLDQuNDM4IEMzLjkxMDY2NjY3LDQuMzI2IDMuNSw0LjE3NjY2NjY3IDIuOTQsMy45OSBDMS45OTczMzMzMywzLjY5MTMzMzMzIDEuMzMyMzMzMzMsMy4zODMzMzMzMyAwLjk0NSwzLjA2NiBDMC41NTc2NjY2NjcsMi43NDg2NjY2NyAwLjM2NCwyLjMzMzMzMzMzIDAuMzY0LDEuODIgQzAuMzY0LDEuMjY5MzMzMzMgMC42MDQzMzMzMzMsMC44MjgzMzMzMzMgMS4wODUsMC40OTcgQzEuNTY1NjY2NjcsMC4xNjU2NjY2NjcgMi4yMTY2NjY2NywwIDMuMDM4LDAgQzMuNjQ0NjY2NjcsMCA0LjIwNywwLjEwMDMzMzMzMyA0LjcyNSwwLjMwMSBDNS4yNDMsMC41MDE2NjY2NjcgNS42MjgsMC43NzQ2NjY2NjcgNS44OCwxLjEyIEM1Ljk2NCwxLjIzMiA2LjAwNiwxLjM0ODY2NjY3IDYuMDA2LDEuNDcgQzYuMDA2LDEuNjI4NjY2NjcgNS45MjIsMS43NjQgNS43NTQsMS44NzYgQzUuNjMyNjY2NjcsMS45NTA2NjY2NyA1LjUxMTMzMzMzLDEuOTg4IDUuMzksMS45ODggQzUuMjAzMzMzMzMsMS45ODggNS4wMjYsMS45MDQgNC44NTgsMS43MzYgQzQuNjM0LDEuNTEyIDQuMzc1LDEuMzQ2MzMzMzMgNC4wODEsMS4yMzkgQzMuNzg3LDEuMTMxNjY2NjcgMy40MzQ2NjY2NywxLjA3OCAzLjAyNCwxLjA3OCBDMi41ODUzMzMzMywxLjA3OCAyLjI1MTY2NjY3LDEuMTQzMzMzMzMgMi4wMjMsMS4yNzQgQzEuNzk0MzMzMzMsMS40MDQ2NjY2NyAxLjY4LDEuNTkxMzMzMzMgMS42OCwxLjgzNCBDMS42OCwyLjAyMDY2NjY3IDEuNzMxMzMzMzMsMi4xNyAxLjgzNCwyLjI4MiBDMS45MzY2NjY2NywyLjM5NCAyLjExNCwyLjUwMTMzMzMzIDIuMzY2LDIuNjA0IEMyLjYxOCwyLjcwNjY2NjY3IDMuMDM4LDIuODUxMzMzMzMgMy42MjYsMy4wMzggQzQuMjc5MzMzMzMsMy4yNDMzMzMzMyA0Ljc5MDMzMzMzLDMuNDQ4NjY2NjcgNS4xNTksMy42NTQgQzUuNTI3NjY2NjcsMy44NTkzMzMzMyA1Ljc5MTMzMzMzLDQuMDg4IDUuOTUsNC4zNCBDNi4xMDg2NjY2Nyw0LjU5MiA2LjE4OCw0Ljg5NTMzMzMzIDYuMTg4LDUuMjUgQzYuMTg4LDUuODE5MzMzMzMgNS45MzM2NjY2Nyw2LjI3NjY2NjY3IDUuNDI1LDYuNjIyIEM0LjkxNjMzMzMzLDYuOTY3MzMzMzMgNC4yMzczMzMzMyw3LjE0IDMuMzg4LDcuMTQgWiIgaWQ9InMiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTAuNTU2LDcuMTQgQzkuNjIyNjY2NjcsNy4xNCA4Ljg5NDY2NjY3LDYuODcxNjY2NjcgOC4zNzIsNi4zMzUgQzcuODQ5MzMzMzMsNS43OTgzMzMzMyA3LjU4OCw1LjA1ODY2NjY3IDcuNTg4LDQuMTE2IEw3LjU4OCwwLjY1OCBDNy41ODgsMC40NTI2NjY2NjcgNy42NDYzMzMzMywwLjI5MTY2NjY2NyA3Ljc2MywwLjE3NSBDNy44Nzk2NjY2NywwLjA1ODMzMzMzMzMgOC4wMzYsMCA4LjIzMiwwIEM4LjQzNzMzMzMzLDAgOC41OTYsMC4wNTgzMzMzMzMzIDguNzA4LDAuMTc1IEM4LjgyLDAuMjkxNjY2NjY3IDguODc2LDAuNDUyNjY2NjY3IDguODc2LDAuNjU4IEw4Ljg3Niw0LjExNiBDOC44NzYsNC43MzIgOS4wNDE2NjY2Nyw1LjIxMDMzMzMzIDkuMzczLDUuNTUxIEM5LjcwNDMzMzMzLDUuODkxNjY2NjcgMTAuMTY4NjY2Nyw2LjA2MiAxMC43NjYsNi4wNjIgQzExLjQ5NCw2LjA2MiAxMi4wNzk2NjY3LDUuODI4NjY2NjcgMTIuNTIzLDUuMzYyIEMxMi45NjYzMzMzLDQuODk1MzMzMzMgMTMuMTg4LDQuMjc0NjY2NjcgMTMuMTg4LDMuNSBMMTMuMTg4LDAuNjU4IEMxMy4xODgsMC40NTI2NjY2NjcgMTMuMjQ2MzMzMywwLjI5MTY2NjY2NyAxMy4zNjMsMC4xNzUgQzEzLjQ3OTY2NjcsMC4wNTgzMzMzMzMzIDEzLjY0MDY2NjcsMCAxMy44NDYsMCBDMTQuMDUxMzMzMywwIDE0LjIwNzY2NjcsMC4wNTYgMTQuMzE1LDAuMTY4IEMxNC40MjIzMzMzLDAuMjggMTQuNDc2LDAuNDQzMzMzMzMzIDE0LjQ3NiwwLjY1OCBMMTQuNDc2LDYuNDgyIEMxNC40NzYsNi42OTY2NjY2NyAxNC40MjIzMzMzLDYuODYgMTQuMzE1LDYuOTcyIEMxNC4yMDc2NjY3LDcuMDg0IDE0LjA1MTMzMzMsNy4xNCAxMy44NDYsNy4xNCBDMTMuNjQwNjY2Nyw3LjE0IDEzLjQ4Miw3LjA4MTY2NjY3IDEzLjM3LDYuOTY1IEMxMy4yNTgsNi44NDgzMzMzMyAxMy4yMDIsNi42ODczMzMzMyAxMy4yMDIsNi40ODIgTDEzLjIwMiw1Ljc5NiBDMTIuNTAyLDYuNjkyIDExLjYyLDcuMTQgMTAuNTU2LDcuMTQgWiIgaWQ9InUiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTYuODQyLDcuMTQgQzE2LjYzNjY2NjcsNy4xNCAxNi40NzgsNy4wODE2NjY2NyAxNi4zNjYsNi45NjUgQzE2LjI1NCw2Ljg0ODMzMzMzIDE2LjE5OCw2LjY4NzMzMzMzIDE2LjE5OCw2LjQ4MiBMMTYuMTk4LDAuNjU4IEMxNi4xOTgsMC40NTI2NjY2NjcgMTYuMjU0LDAuMjkxNjY2NjY3IDE2LjM2NiwwLjE3NSBDMTYuNDc4LDAuMDU4MzMzMzMzMyAxNi42MzY2NjY3LDAgMTYuODQyLDAgQzE3LjA0NzMzMzMsMCAxNy4yMDgzMzMzLDAuMDU4MzMzMzMzMyAxNy4zMjUsMC4xNzUgQzE3LjQ0MTY2NjcsMC4yOTE2NjY2NjcgMTcuNSwwLjQ1MjY2NjY2NyAxNy41LDAuNjU4IEwxNy41LDEuMzAyIEMxNy43MzMzMzMzLDAuODkxMzMzMzMzIDE4LjAzNjY2NjcsMC41NzE2NjY2NjcgMTguNDEsMC4zNDMgQzE4Ljc4MzMzMzMsMC4xMTQzMzMzMzMgMTkuMTgsMCAxOS42LDAgQzIwLjAyOTMzMzMsMCAyMC40MDczMzMzLDAuMTE0MzMzMzMzIDIwLjczNCwwLjM0MyBDMjEuMDYwNjY2NywwLjU3MTY2NjY2NyAyMS4zMDMzMzMzLDAuODkxMzMzMzMzIDIxLjQ2MiwxLjMwMiBDMjIuMTQzMzMzMywwLjQzNCAyMi45MTMzMzMzLDAgMjMuNzcyLDAgQzI0LjUwOTMzMzMsMCAyNS4wNzE2NjY3LDAuMjM4IDI1LjQ1OSwwLjcxNCBDMjUuODQ2MzMzMywxLjE5IDI2LjA0LDEuODgwNjY2NjcgMjYuMDQsMi43ODYgTDI2LjA0LDYuNDgyIEMyNi4wNCw2LjY4NzMzMzMzIDI1Ljk4MTY2NjcsNi44NDgzMzMzMyAyNS44NjUsNi45NjUgQzI1Ljc0ODMzMzMsNy4wODE2NjY2NyAyNS41OTIsNy4xNCAyNS4zOTYsNy4xNCBDMjUuMTkwNjY2Nyw3LjE0IDI1LjAzMiw3LjA4NCAyNC45Miw2Ljk3MiBDMjQuODA4LDYuODYgMjQuNzUyLDYuNjk2NjY2NjcgMjQuNzUyLDYuNDgyIEwyNC43NTIsMi43MyBDMjQuNzUyLDIuMTc5MzMzMzMgMjQuNjU4NjY2NywxLjc2NjMzMzMzIDI0LjQ3MiwxLjQ5MSBDMjQuMjg1MzMzMywxLjIxNTY2NjY3IDIzLjk3NzMzMzMsMS4wNzggMjMuNTQ4LDEuMDc4IEMyMi44MDEzMzMzLDEuMDc4IDIyLjIzNjY2NjcsMS41MDI2NjY2NyAyMS44NTQsMi4zNTIgTDIxLjg1NCw2LjQ4MiBDMjEuODU0LDYuNjg3MzMzMzMgMjEuNzk1NjY2Nyw2Ljg0ODMzMzMzIDIxLjY3OSw2Ljk2NSBDMjEuNTYyMzMzMyw3LjA4MTY2NjY3IDIxLjQwMTMzMzMsNy4xNCAyMS4xOTYsNy4xNCBDMjAuOTkwNjY2Nyw3LjE0IDIwLjgzNDMzMzMsNy4wODQgMjAuNzI3LDYuOTcyIEMyMC42MTk2NjY3LDYuODYgMjAuNTY2LDYuNjk2NjY2NjcgMjAuNTY2LDYuNDgyIEwyMC41NjYsMi45NjggQzIwLjU2NiwyLjM1MiAyMC40NjMzMzMzLDEuODgzIDIwLjI1OCwxLjU2MSBDMjAuMDUyNjY2NywxLjIzOSAxOS43NTQsMS4wNzggMTkuMzYyLDEuMDc4IEMxOC44MTEzMzMzLDEuMDc4IDE4LjM2MzMzMzMsMS4yOTUgMTguMDE4LDEuNzI5IEMxNy42NzI2NjY3LDIuMTYzIDE3LjUsMi43MjUzMzMzMyAxNy41LDMuNDE2IEwxNy41LDYuNDgyIEMxNy41LDYuNjg3MzMzMzMgMTcuNDQxNjY2Nyw2Ljg0ODMzMzMzIDE3LjMyNSw2Ljk2NSBDMTcuMjA4MzMzMyw3LjA4MTY2NjY3IDE3LjA0NzMzMzMsNy4xNCAxNi44NDIsNy4xNCBaIiBpZD0ibSIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuOTUyMzgxLDEwIEwxOS4wNDc2MTkxLDEwIEMxNi44MTMzMzMzLDEwIDE1LDExLjgxMzMzMzMgMTUsMTQuMDQ3NjE5IEwxNSwzOS45NTIzODEgQzE1LDQyLjE4NjY2NjcgMTYuODEzMzMzMyw0NCAxOS4wNDc2MTkxLDQ0IEw0NC45NTIzODEsNDQgQzQ3LjE4NjY2NjcsNDQgNDksNDIuMTg2NjY2NyA0OSwzOS45NTIzODEgTDQ5LDE0LjA0NzYxOSBDNDksMTEuODEzMzMzMyA0Ny4xODY2NjY3LDEwIDQ0Ljk1MjM4MSwxMCBaIE0yNi43OTQ3NjE5LDM2LjU2ODU3MTQgTDI1Ljg3MTkwNDgsMzUuMjQwOTUyNCBMMzcuODYwOTUyNCwyNyBMMjUuODcxOTA0OCwxOC43NTkwNDc2IEwyNi43OTQ3NjE5LDE3LjQzMTQyODYgTDQwLjcxODU3MTQsMjcgTDI2Ljc5NDc2MTksMzYuNTY4NTcxNCBaIiBpZD0iZ3JlYXRlcl90aGFuIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_sum_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">15</td>
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc3VtX2d0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9zdW1fZ3QiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InN1bSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTguODg5MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCw3LjE0IEMyLjY2OTMzMzMzLDcuMTQgMi4wMyw3LjAxNjMzMzMzIDEuNDcsNi43NjkgQzAuOTEsNi41MjE2NjY2NyAwLjQ2NjY2NjY2Nyw2LjE4OCAwLjE0LDUuNzY4IEMwLjA0NjY2NjY2NjcsNS42NDY2NjY2NyAwLDUuNTI1MzMzMzMgMCw1LjQwNCBDMCw1LjE5ODY2NjY3IDAuMDk4LDUuMDQ0NjY2NjcgMC4yOTQsNC45NDIgQzAuMzc4LDQuODk1MzMzMzMgMC40NjY2NjY2NjcsNC44NzIgMC41Niw0Ljg3MiBDMC43MTg2NjY2NjcsNC44NzIgMC44ODIsNC45NTYgMS4wNSw1LjEyNCBDMS4zNTgsNS40NTA2NjY2NyAxLjY5ODY2NjY3LDUuNjg4NjY2NjcgMi4wNzIsNS44MzggQzIuNDQ1MzMzMzMsNS45ODczMzMzMyAyLjg3NDY2NjY3LDYuMDYyIDMuMzYsNi4wNjIgQzMuODA4LDYuMDYyIDQuMTc0MzMzMzMsNS45ODk2NjY2NyA0LjQ1OSw1Ljg0NSBDNC43NDM2NjY2Nyw1LjcwMDMzMzMzIDQuODg2LDUuNTAyIDQuODg2LDUuMjUgQzQuODg2LDUuMDYzMzMzMzMgNC44MzIzMzMzMyw0LjkwOTMzMzMzIDQuNzI1LDQuNzg4IEM0LjYxNzY2NjY3LDQuNjY2NjY2NjcgNC40MzMzMzMzMyw0LjU1IDQuMTcyLDQuNDM4IEMzLjkxMDY2NjY3LDQuMzI2IDMuNSw0LjE3NjY2NjY3IDIuOTQsMy45OSBDMS45OTczMzMzMywzLjY5MTMzMzMzIDEuMzMyMzMzMzMsMy4zODMzMzMzMyAwLjk0NSwzLjA2NiBDMC41NTc2NjY2NjcsMi43NDg2NjY2NyAwLjM2NCwyLjMzMzMzMzMzIDAuMzY0LDEuODIgQzAuMzY0LDEuMjY5MzMzMzMgMC42MDQzMzMzMzMsMC44MjgzMzMzMzMgMS4wODUsMC40OTcgQzEuNTY1NjY2NjcsMC4xNjU2NjY2NjcgMi4yMTY2NjY2NywwIDMuMDM4LDAgQzMuNjQ0NjY2NjcsMCA0LjIwNywwLjEwMDMzMzMzMyA0LjcyNSwwLjMwMSBDNS4yNDMsMC41MDE2NjY2NjcgNS42MjgsMC43NzQ2NjY2NjcgNS44OCwxLjEyIEM1Ljk2NCwxLjIzMiA2LjAwNiwxLjM0ODY2NjY3IDYuMDA2LDEuNDcgQzYuMDA2LDEuNjI4NjY2NjcgNS45MjIsMS43NjQgNS43NTQsMS44NzYgQzUuNjMyNjY2NjcsMS45NTA2NjY2NyA1LjUxMTMzMzMzLDEuOTg4IDUuMzksMS45ODggQzUuMjAzMzMzMzMsMS45ODggNS4wMjYsMS45MDQgNC44NTgsMS43MzYgQzQuNjM0LDEuNTEyIDQuMzc1LDEuMzQ2MzMzMzMgNC4wODEsMS4yMzkgQzMuNzg3LDEuMTMxNjY2NjcgMy40MzQ2NjY2NywxLjA3OCAzLjAyNCwxLjA3OCBDMi41ODUzMzMzMywxLjA3OCAyLjI1MTY2NjY3LDEuMTQzMzMzMzMgMi4wMjMsMS4yNzQgQzEuNzk0MzMzMzMsMS40MDQ2NjY2NyAxLjY4LDEuNTkxMzMzMzMgMS42OCwxLjgzNCBDMS42OCwyLjAyMDY2NjY3IDEuNzMxMzMzMzMsMi4xNyAxLjgzNCwyLjI4MiBDMS45MzY2NjY2NywyLjM5NCAyLjExNCwyLjUwMTMzMzMzIDIuMzY2LDIuNjA0IEMyLjYxOCwyLjcwNjY2NjY3IDMuMDM4LDIuODUxMzMzMzMgMy42MjYsMy4wMzggQzQuMjc5MzMzMzMsMy4yNDMzMzMzMyA0Ljc5MDMzMzMzLDMuNDQ4NjY2NjcgNS4xNTksMy42NTQgQzUuNTI3NjY2NjcsMy44NTkzMzMzMyA1Ljc5MTMzMzMzLDQuMDg4IDUuOTUsNC4zNCBDNi4xMDg2NjY2Nyw0LjU5MiA2LjE4OCw0Ljg5NTMzMzMzIDYuMTg4LDUuMjUgQzYuMTg4LDUuODE5MzMzMzMgNS45MzM2NjY2Nyw2LjI3NjY2NjY3IDUuNDI1LDYuNjIyIEM0LjkxNjMzMzMzLDYuOTY3MzMzMzMgNC4yMzczMzMzMyw3LjE0IDMuMzg4LDcuMTQgWiIgaWQ9InMiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTAuNTU2LDcuMTQgQzkuNjIyNjY2NjcsNy4xNCA4Ljg5NDY2NjY3LDYuODcxNjY2NjcgOC4zNzIsNi4zMzUgQzcuODQ5MzMzMzMsNS43OTgzMzMzMyA3LjU4OCw1LjA1ODY2NjY3IDcuNTg4LDQuMTE2IEw3LjU4OCwwLjY1OCBDNy41ODgsMC40NTI2NjY2NjcgNy42NDYzMzMzMywwLjI5MTY2NjY2NyA3Ljc2MywwLjE3NSBDNy44Nzk2NjY2NywwLjA1ODMzMzMzMzMgOC4wMzYsMCA4LjIzMiwwIEM4LjQzNzMzMzMzLDAgOC41OTYsMC4wNTgzMzMzMzMzIDguNzA4LDAuMTc1IEM4LjgyLDAuMjkxNjY2NjY3IDguODc2LDAuNDUyNjY2NjY3IDguODc2LDAuNjU4IEw4Ljg3Niw0LjExNiBDOC44NzYsNC43MzIgOS4wNDE2NjY2Nyw1LjIxMDMzMzMzIDkuMzczLDUuNTUxIEM5LjcwNDMzMzMzLDUuODkxNjY2NjcgMTAuMTY4NjY2Nyw2LjA2MiAxMC43NjYsNi4wNjIgQzExLjQ5NCw2LjA2MiAxMi4wNzk2NjY3LDUuODI4NjY2NjcgMTIuNTIzLDUuMzYyIEMxMi45NjYzMzMzLDQuODk1MzMzMzMgMTMuMTg4LDQuMjc0NjY2NjcgMTMuMTg4LDMuNSBMMTMuMTg4LDAuNjU4IEMxMy4xODgsMC40NTI2NjY2NjcgMTMuMjQ2MzMzMywwLjI5MTY2NjY2NyAxMy4zNjMsMC4xNzUgQzEzLjQ3OTY2NjcsMC4wNTgzMzMzMzMzIDEzLjY0MDY2NjcsMCAxMy44NDYsMCBDMTQuMDUxMzMzMywwIDE0LjIwNzY2NjcsMC4wNTYgMTQuMzE1LDAuMTY4IEMxNC40MjIzMzMzLDAuMjggMTQuNDc2LDAuNDQzMzMzMzMzIDE0LjQ3NiwwLjY1OCBMMTQuNDc2LDYuNDgyIEMxNC40NzYsNi42OTY2NjY2NyAxNC40MjIzMzMzLDYuODYgMTQuMzE1LDYuOTcyIEMxNC4yMDc2NjY3LDcuMDg0IDE0LjA1MTMzMzMsNy4xNCAxMy44NDYsNy4xNCBDMTMuNjQwNjY2Nyw3LjE0IDEzLjQ4Miw3LjA4MTY2NjY3IDEzLjM3LDYuOTY1IEMxMy4yNTgsNi44NDgzMzMzMyAxMy4yMDIsNi42ODczMzMzMyAxMy4yMDIsNi40ODIgTDEzLjIwMiw1Ljc5NiBDMTIuNTAyLDYuNjkyIDExLjYyLDcuMTQgMTAuNTU2LDcuMTQgWiIgaWQ9InUiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTYuODQyLDcuMTQgQzE2LjYzNjY2NjcsNy4xNCAxNi40NzgsNy4wODE2NjY2NyAxNi4zNjYsNi45NjUgQzE2LjI1NCw2Ljg0ODMzMzMzIDE2LjE5OCw2LjY4NzMzMzMzIDE2LjE5OCw2LjQ4MiBMMTYuMTk4LDAuNjU4IEMxNi4xOTgsMC40NTI2NjY2NjcgMTYuMjU0LDAuMjkxNjY2NjY3IDE2LjM2NiwwLjE3NSBDMTYuNDc4LDAuMDU4MzMzMzMzMyAxNi42MzY2NjY3LDAgMTYuODQyLDAgQzE3LjA0NzMzMzMsMCAxNy4yMDgzMzMzLDAuMDU4MzMzMzMzMyAxNy4zMjUsMC4xNzUgQzE3LjQ0MTY2NjcsMC4yOTE2NjY2NjcgMTcuNSwwLjQ1MjY2NjY2NyAxNy41LDAuNjU4IEwxNy41LDEuMzAyIEMxNy43MzMzMzMzLDAuODkxMzMzMzMzIDE4LjAzNjY2NjcsMC41NzE2NjY2NjcgMTguNDEsMC4zNDMgQzE4Ljc4MzMzMzMsMC4xMTQzMzMzMzMgMTkuMTgsMCAxOS42LDAgQzIwLjAyOTMzMzMsMCAyMC40MDczMzMzLDAuMTE0MzMzMzMzIDIwLjczNCwwLjM0MyBDMjEuMDYwNjY2NywwLjU3MTY2NjY2NyAyMS4zMDMzMzMzLDAuODkxMzMzMzMzIDIxLjQ2MiwxLjMwMiBDMjIuMTQzMzMzMywwLjQzNCAyMi45MTMzMzMzLDAgMjMuNzcyLDAgQzI0LjUwOTMzMzMsMCAyNS4wNzE2NjY3LDAuMjM4IDI1LjQ1OSwwLjcxNCBDMjUuODQ2MzMzMywxLjE5IDI2LjA0LDEuODgwNjY2NjcgMjYuMDQsMi43ODYgTDI2LjA0LDYuNDgyIEMyNi4wNCw2LjY4NzMzMzMzIDI1Ljk4MTY2NjcsNi44NDgzMzMzMyAyNS44NjUsNi45NjUgQzI1Ljc0ODMzMzMsNy4wODE2NjY2NyAyNS41OTIsNy4xNCAyNS4zOTYsNy4xNCBDMjUuMTkwNjY2Nyw3LjE0IDI1LjAzMiw3LjA4NCAyNC45Miw2Ljk3MiBDMjQuODA4LDYuODYgMjQuNzUyLDYuNjk2NjY2NjcgMjQuNzUyLDYuNDgyIEwyNC43NTIsMi43MyBDMjQuNzUyLDIuMTc5MzMzMzMgMjQuNjU4NjY2NywxLjc2NjMzMzMzIDI0LjQ3MiwxLjQ5MSBDMjQuMjg1MzMzMywxLjIxNTY2NjY3IDIzLjk3NzMzMzMsMS4wNzggMjMuNTQ4LDEuMDc4IEMyMi44MDEzMzMzLDEuMDc4IDIyLjIzNjY2NjcsMS41MDI2NjY2NyAyMS44NTQsMi4zNTIgTDIxLjg1NCw2LjQ4MiBDMjEuODU0LDYuNjg3MzMzMzMgMjEuNzk1NjY2Nyw2Ljg0ODMzMzMzIDIxLjY3OSw2Ljk2NSBDMjEuNTYyMzMzMyw3LjA4MTY2NjY3IDIxLjQwMTMzMzMsNy4xNCAyMS4xOTYsNy4xNCBDMjAuOTkwNjY2Nyw3LjE0IDIwLjgzNDMzMzMsNy4wODQgMjAuNzI3LDYuOTcyIEMyMC42MTk2NjY3LDYuODYgMjAuNTY2LDYuNjk2NjY2NjcgMjAuNTY2LDYuNDgyIEwyMC41NjYsMi45NjggQzIwLjU2NiwyLjM1MiAyMC40NjMzMzMzLDEuODgzIDIwLjI1OCwxLjU2MSBDMjAuMDUyNjY2NywxLjIzOSAxOS43NTQsMS4wNzggMTkuMzYyLDEuMDc4IEMxOC44MTEzMzMzLDEuMDc4IDE4LjM2MzMzMzMsMS4yOTUgMTguMDE4LDEuNzI5IEMxNy42NzI2NjY3LDIuMTYzIDE3LjUsMi43MjUzMzMzMyAxNy41LDMuNDE2IEwxNy41LDYuNDgyIEMxNy41LDYuNjg3MzMzMzMgMTcuNDQxNjY2Nyw2Ljg0ODMzMzMzIDE3LjMyNSw2Ljk2NSBDMTcuMjA4MzMzMyw3LjA4MTY2NjY3IDE3LjA0NzMzMzMsNy4xNCAxNi44NDIsNy4xNCBaIiBpZD0ibSIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuOTUyMzgxLDEwIEwxOS4wNDc2MTkxLDEwIEMxNi44MTMzMzMzLDEwIDE1LDExLjgxMzMzMzMgMTUsMTQuMDQ3NjE5IEwxNSwzOS45NTIzODEgQzE1LDQyLjE4NjY2NjcgMTYuODEzMzMzMyw0NCAxOS4wNDc2MTkxLDQ0IEw0NC45NTIzODEsNDQgQzQ3LjE4NjY2NjcsNDQgNDksNDIuMTg2NjY2NyA0OSwzOS45NTIzODEgTDQ5LDE0LjA0NzYxOSBDNDksMTEuODEzMzMzMyA0Ny4xODY2NjY3LDEwIDQ0Ljk1MjM4MSwxMCBaIE0yNi43OTQ3NjE5LDM2LjU2ODU3MTQgTDI1Ljg3MTkwNDgsMzUuMjQwOTUyNCBMMzcuODYwOTUyNCwyNyBMMjUuODcxOTA0OCwxOC43NTkwNDc2IEwyNi43OTQ3NjE5LDE3LjQzMTQyODYgTDQwLjcxODU3MTQsMjcgTDI2Ljc5NDc2MTksMzYuNTY4NTcxNCBaIiBpZD0iZ3JlYXRlcl90aGFuIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_sum_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">15</td>
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
    .col_sum_gt(columns="a", value=15, tol=1.0)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc3VtX2d0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9zdW1fZ3QiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InN1bSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTguODg5MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCw3LjE0IEMyLjY2OTMzMzMzLDcuMTQgMi4wMyw3LjAxNjMzMzMzIDEuNDcsNi43NjkgQzAuOTEsNi41MjE2NjY2NyAwLjQ2NjY2NjY2Nyw2LjE4OCAwLjE0LDUuNzY4IEMwLjA0NjY2NjY2NjcsNS42NDY2NjY2NyAwLDUuNTI1MzMzMzMgMCw1LjQwNCBDMCw1LjE5ODY2NjY3IDAuMDk4LDUuMDQ0NjY2NjcgMC4yOTQsNC45NDIgQzAuMzc4LDQuODk1MzMzMzMgMC40NjY2NjY2NjcsNC44NzIgMC41Niw0Ljg3MiBDMC43MTg2NjY2NjcsNC44NzIgMC44ODIsNC45NTYgMS4wNSw1LjEyNCBDMS4zNTgsNS40NTA2NjY2NyAxLjY5ODY2NjY3LDUuNjg4NjY2NjcgMi4wNzIsNS44MzggQzIuNDQ1MzMzMzMsNS45ODczMzMzMyAyLjg3NDY2NjY3LDYuMDYyIDMuMzYsNi4wNjIgQzMuODA4LDYuMDYyIDQuMTc0MzMzMzMsNS45ODk2NjY2NyA0LjQ1OSw1Ljg0NSBDNC43NDM2NjY2Nyw1LjcwMDMzMzMzIDQuODg2LDUuNTAyIDQuODg2LDUuMjUgQzQuODg2LDUuMDYzMzMzMzMgNC44MzIzMzMzMyw0LjkwOTMzMzMzIDQuNzI1LDQuNzg4IEM0LjYxNzY2NjY3LDQuNjY2NjY2NjcgNC40MzMzMzMzMyw0LjU1IDQuMTcyLDQuNDM4IEMzLjkxMDY2NjY3LDQuMzI2IDMuNSw0LjE3NjY2NjY3IDIuOTQsMy45OSBDMS45OTczMzMzMywzLjY5MTMzMzMzIDEuMzMyMzMzMzMsMy4zODMzMzMzMyAwLjk0NSwzLjA2NiBDMC41NTc2NjY2NjcsMi43NDg2NjY2NyAwLjM2NCwyLjMzMzMzMzMzIDAuMzY0LDEuODIgQzAuMzY0LDEuMjY5MzMzMzMgMC42MDQzMzMzMzMsMC44MjgzMzMzMzMgMS4wODUsMC40OTcgQzEuNTY1NjY2NjcsMC4xNjU2NjY2NjcgMi4yMTY2NjY2NywwIDMuMDM4LDAgQzMuNjQ0NjY2NjcsMCA0LjIwNywwLjEwMDMzMzMzMyA0LjcyNSwwLjMwMSBDNS4yNDMsMC41MDE2NjY2NjcgNS42MjgsMC43NzQ2NjY2NjcgNS44OCwxLjEyIEM1Ljk2NCwxLjIzMiA2LjAwNiwxLjM0ODY2NjY3IDYuMDA2LDEuNDcgQzYuMDA2LDEuNjI4NjY2NjcgNS45MjIsMS43NjQgNS43NTQsMS44NzYgQzUuNjMyNjY2NjcsMS45NTA2NjY2NyA1LjUxMTMzMzMzLDEuOTg4IDUuMzksMS45ODggQzUuMjAzMzMzMzMsMS45ODggNS4wMjYsMS45MDQgNC44NTgsMS43MzYgQzQuNjM0LDEuNTEyIDQuMzc1LDEuMzQ2MzMzMzMgNC4wODEsMS4yMzkgQzMuNzg3LDEuMTMxNjY2NjcgMy40MzQ2NjY2NywxLjA3OCAzLjAyNCwxLjA3OCBDMi41ODUzMzMzMywxLjA3OCAyLjI1MTY2NjY3LDEuMTQzMzMzMzMgMi4wMjMsMS4yNzQgQzEuNzk0MzMzMzMsMS40MDQ2NjY2NyAxLjY4LDEuNTkxMzMzMzMgMS42OCwxLjgzNCBDMS42OCwyLjAyMDY2NjY3IDEuNzMxMzMzMzMsMi4xNyAxLjgzNCwyLjI4MiBDMS45MzY2NjY2NywyLjM5NCAyLjExNCwyLjUwMTMzMzMzIDIuMzY2LDIuNjA0IEMyLjYxOCwyLjcwNjY2NjY3IDMuMDM4LDIuODUxMzMzMzMgMy42MjYsMy4wMzggQzQuMjc5MzMzMzMsMy4yNDMzMzMzMyA0Ljc5MDMzMzMzLDMuNDQ4NjY2NjcgNS4xNTksMy42NTQgQzUuNTI3NjY2NjcsMy44NTkzMzMzMyA1Ljc5MTMzMzMzLDQuMDg4IDUuOTUsNC4zNCBDNi4xMDg2NjY2Nyw0LjU5MiA2LjE4OCw0Ljg5NTMzMzMzIDYuMTg4LDUuMjUgQzYuMTg4LDUuODE5MzMzMzMgNS45MzM2NjY2Nyw2LjI3NjY2NjY3IDUuNDI1LDYuNjIyIEM0LjkxNjMzMzMzLDYuOTY3MzMzMzMgNC4yMzczMzMzMyw3LjE0IDMuMzg4LDcuMTQgWiIgaWQ9InMiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTAuNTU2LDcuMTQgQzkuNjIyNjY2NjcsNy4xNCA4Ljg5NDY2NjY3LDYuODcxNjY2NjcgOC4zNzIsNi4zMzUgQzcuODQ5MzMzMzMsNS43OTgzMzMzMyA3LjU4OCw1LjA1ODY2NjY3IDcuNTg4LDQuMTE2IEw3LjU4OCwwLjY1OCBDNy41ODgsMC40NTI2NjY2NjcgNy42NDYzMzMzMywwLjI5MTY2NjY2NyA3Ljc2MywwLjE3NSBDNy44Nzk2NjY2NywwLjA1ODMzMzMzMzMgOC4wMzYsMCA4LjIzMiwwIEM4LjQzNzMzMzMzLDAgOC41OTYsMC4wNTgzMzMzMzMzIDguNzA4LDAuMTc1IEM4LjgyLDAuMjkxNjY2NjY3IDguODc2LDAuNDUyNjY2NjY3IDguODc2LDAuNjU4IEw4Ljg3Niw0LjExNiBDOC44NzYsNC43MzIgOS4wNDE2NjY2Nyw1LjIxMDMzMzMzIDkuMzczLDUuNTUxIEM5LjcwNDMzMzMzLDUuODkxNjY2NjcgMTAuMTY4NjY2Nyw2LjA2MiAxMC43NjYsNi4wNjIgQzExLjQ5NCw2LjA2MiAxMi4wNzk2NjY3LDUuODI4NjY2NjcgMTIuNTIzLDUuMzYyIEMxMi45NjYzMzMzLDQuODk1MzMzMzMgMTMuMTg4LDQuMjc0NjY2NjcgMTMuMTg4LDMuNSBMMTMuMTg4LDAuNjU4IEMxMy4xODgsMC40NTI2NjY2NjcgMTMuMjQ2MzMzMywwLjI5MTY2NjY2NyAxMy4zNjMsMC4xNzUgQzEzLjQ3OTY2NjcsMC4wNTgzMzMzMzMzIDEzLjY0MDY2NjcsMCAxMy44NDYsMCBDMTQuMDUxMzMzMywwIDE0LjIwNzY2NjcsMC4wNTYgMTQuMzE1LDAuMTY4IEMxNC40MjIzMzMzLDAuMjggMTQuNDc2LDAuNDQzMzMzMzMzIDE0LjQ3NiwwLjY1OCBMMTQuNDc2LDYuNDgyIEMxNC40NzYsNi42OTY2NjY2NyAxNC40MjIzMzMzLDYuODYgMTQuMzE1LDYuOTcyIEMxNC4yMDc2NjY3LDcuMDg0IDE0LjA1MTMzMzMsNy4xNCAxMy44NDYsNy4xNCBDMTMuNjQwNjY2Nyw3LjE0IDEzLjQ4Miw3LjA4MTY2NjY3IDEzLjM3LDYuOTY1IEMxMy4yNTgsNi44NDgzMzMzMyAxMy4yMDIsNi42ODczMzMzMyAxMy4yMDIsNi40ODIgTDEzLjIwMiw1Ljc5NiBDMTIuNTAyLDYuNjkyIDExLjYyLDcuMTQgMTAuNTU2LDcuMTQgWiIgaWQ9InUiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTYuODQyLDcuMTQgQzE2LjYzNjY2NjcsNy4xNCAxNi40NzgsNy4wODE2NjY2NyAxNi4zNjYsNi45NjUgQzE2LjI1NCw2Ljg0ODMzMzMzIDE2LjE5OCw2LjY4NzMzMzMzIDE2LjE5OCw2LjQ4MiBMMTYuMTk4LDAuNjU4IEMxNi4xOTgsMC40NTI2NjY2NjcgMTYuMjU0LDAuMjkxNjY2NjY3IDE2LjM2NiwwLjE3NSBDMTYuNDc4LDAuMDU4MzMzMzMzMyAxNi42MzY2NjY3LDAgMTYuODQyLDAgQzE3LjA0NzMzMzMsMCAxNy4yMDgzMzMzLDAuMDU4MzMzMzMzMyAxNy4zMjUsMC4xNzUgQzE3LjQ0MTY2NjcsMC4yOTE2NjY2NjcgMTcuNSwwLjQ1MjY2NjY2NyAxNy41LDAuNjU4IEwxNy41LDEuMzAyIEMxNy43MzMzMzMzLDAuODkxMzMzMzMzIDE4LjAzNjY2NjcsMC41NzE2NjY2NjcgMTguNDEsMC4zNDMgQzE4Ljc4MzMzMzMsMC4xMTQzMzMzMzMgMTkuMTgsMCAxOS42LDAgQzIwLjAyOTMzMzMsMCAyMC40MDczMzMzLDAuMTE0MzMzMzMzIDIwLjczNCwwLjM0MyBDMjEuMDYwNjY2NywwLjU3MTY2NjY2NyAyMS4zMDMzMzMzLDAuODkxMzMzMzMzIDIxLjQ2MiwxLjMwMiBDMjIuMTQzMzMzMywwLjQzNCAyMi45MTMzMzMzLDAgMjMuNzcyLDAgQzI0LjUwOTMzMzMsMCAyNS4wNzE2NjY3LDAuMjM4IDI1LjQ1OSwwLjcxNCBDMjUuODQ2MzMzMywxLjE5IDI2LjA0LDEuODgwNjY2NjcgMjYuMDQsMi43ODYgTDI2LjA0LDYuNDgyIEMyNi4wNCw2LjY4NzMzMzMzIDI1Ljk4MTY2NjcsNi44NDgzMzMzMyAyNS44NjUsNi45NjUgQzI1Ljc0ODMzMzMsNy4wODE2NjY2NyAyNS41OTIsNy4xNCAyNS4zOTYsNy4xNCBDMjUuMTkwNjY2Nyw3LjE0IDI1LjAzMiw3LjA4NCAyNC45Miw2Ljk3MiBDMjQuODA4LDYuODYgMjQuNzUyLDYuNjk2NjY2NjcgMjQuNzUyLDYuNDgyIEwyNC43NTIsMi43MyBDMjQuNzUyLDIuMTc5MzMzMzMgMjQuNjU4NjY2NywxLjc2NjMzMzMzIDI0LjQ3MiwxLjQ5MSBDMjQuMjg1MzMzMywxLjIxNTY2NjY3IDIzLjk3NzMzMzMsMS4wNzggMjMuNTQ4LDEuMDc4IEMyMi44MDEzMzMzLDEuMDc4IDIyLjIzNjY2NjcsMS41MDI2NjY2NyAyMS44NTQsMi4zNTIgTDIxLjg1NCw2LjQ4MiBDMjEuODU0LDYuNjg3MzMzMzMgMjEuNzk1NjY2Nyw2Ljg0ODMzMzMzIDIxLjY3OSw2Ljk2NSBDMjEuNTYyMzMzMyw3LjA4MTY2NjY3IDIxLjQwMTMzMzMsNy4xNCAyMS4xOTYsNy4xNCBDMjAuOTkwNjY2Nyw3LjE0IDIwLjgzNDMzMzMsNy4wODQgMjAuNzI3LDYuOTcyIEMyMC42MTk2NjY3LDYuODYgMjAuNTY2LDYuNjk2NjY2NjcgMjAuNTY2LDYuNDgyIEwyMC41NjYsMi45NjggQzIwLjU2NiwyLjM1MiAyMC40NjMzMzMzLDEuODgzIDIwLjI1OCwxLjU2MSBDMjAuMDUyNjY2NywxLjIzOSAxOS43NTQsMS4wNzggMTkuMzYyLDEuMDc4IEMxOC44MTEzMzMzLDEuMDc4IDE4LjM2MzMzMzMsMS4yOTUgMTguMDE4LDEuNzI5IEMxNy42NzI2NjY3LDIuMTYzIDE3LjUsMi43MjUzMzMzMyAxNy41LDMuNDE2IEwxNy41LDYuNDgyIEMxNy41LDYuNjg3MzMzMzMgMTcuNDQxNjY2Nyw2Ljg0ODMzMzMzIDE3LjMyNSw2Ljk2NSBDMTcuMjA4MzMzMyw3LjA4MTY2NjY3IDE3LjA0NzMzMzMsNy4xNCAxNi44NDIsNy4xNCBaIiBpZD0ibSIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuOTUyMzgxLDEwIEwxOS4wNDc2MTkxLDEwIEMxNi44MTMzMzMzLDEwIDE1LDExLjgxMzMzMzMgMTUsMTQuMDQ3NjE5IEwxNSwzOS45NTIzODEgQzE1LDQyLjE4NjY2NjcgMTYuODEzMzMzMyw0NCAxOS4wNDc2MTkxLDQ0IEw0NC45NTIzODEsNDQgQzQ3LjE4NjY2NjcsNDQgNDksNDIuMTg2NjY2NyA0OSwzOS45NTIzODEgTDQ5LDE0LjA0NzYxOSBDNDksMTEuODEzMzMzMyA0Ny4xODY2NjY3LDEwIDQ0Ljk1MjM4MSwxMCBaIE0yNi43OTQ3NjE5LDM2LjU2ODU3MTQgTDI1Ljg3MTkwNDgsMzUuMjQwOTUyNCBMMzcuODYwOTUyNCwyNyBMMjUuODcxOTA0OCwxOC43NTkwNDc2IEwyNi43OTQ3NjE5LDE3LjQzMTQyODYgTDQwLjcxODU3MTQsMjcgTDI2Ljc5NDc2MTksMzYuNTY4NTcxNCBaIiBpZD0iZ3JlYXRlcl90aGFuIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_sum_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">15<br />
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
