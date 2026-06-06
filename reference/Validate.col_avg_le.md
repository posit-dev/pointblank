## Validate.col_avg_le()


Does the column average satisfy a less than or equal to comparison?


Usage

``` python
Validate.col_avg_le(
    columns,
    value=None,
    tol=0,
    thresholds=None,
    brief=False,
    actions=None,
    active=True
)
```


The [col_avg_le()](Validate.col_avg_le.md#pointblank.Validate.col_avg_le) validation method checks whether the average of values in a column is at most a specified `value=`. This is an aggregation-based validation where the entire column is reduced to a single average value that is then compared against the target. The comparison used in this function is `average(column) <= value`.

Unlike row-level validations (e.g., [col_vals_gt()](Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)), this method treats the entire column as a single test unit. The validation either passes completely (if the aggregated value satisfies the comparison) or fails completely.


## Parameters


`columns: _PBUnresolvedColumn`  
A single column or a list of columns to validate. If multiple columns are supplied, there will be a separate validation step generated for each column. The columns must contain numeric data for the average to be computed.

`value: float | Column | ReferenceColumn | None = None`  
The value to compare the column average against. This can be: (1) a numeric literal (`int` or `float`), (2) a <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a> object referencing another column whose average will be used for comparison, (3) a <a href="ref.html#pointblank.ref" class="gdls-link"><code>ref()</code></a> object referencing a column in reference data (when `Validate(reference=)` has been set), or (4) `None` to automatically compare against the same column in reference data (shorthand for `ref(column_name)` when reference data is set).

`tol: Tolerance = ``0`  
A tolerance value for the comparison. The default is `0`, meaning exact comparison. When set to a positive value, the comparison becomes more lenient. For example, with `tol=0.5`, a average that differs from the target by up to `0.5` will still pass. The `tol=` parameter expands the acceptable range for the comparison. For [col_avg_le()](Validate.col_avg_le.md#pointblank.Validate.col_avg_le), a tolerance of `tol=0.5` would mean the average can be within `0.5` of the target value and still pass validation.

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

The [col_avg_le()](Validate.col_avg_le.md#pointblank.Validate.col_avg_le) method supports comparing column aggregations against reference data. This is useful for validating that statistical properties remain consistent across different versions of a dataset, or for comparing current data against historical baselines.

To use reference data, set the `reference=` parameter when creating the [Validate](Validate.md#pointblank.Validate) object:

``` python
validation = (
    pb.Validate(data=current_data, reference=baseline_data)
    .col_avg_le(columns="revenue")  # Compares sum(current.revenue) vs sum(baseline.revenue)
    .interrogate()
)
```

When `value=None` and reference data is set, the method automatically compares against the same column in the reference data. You can also explicitly specify reference columns using the [ref()](ref.md#pointblank.ref) helper:

``` python
.col_avg_le(columns="revenue", value=pb.ref("baseline_revenue"))
```


## Understanding Tolerance

The `tol=` parameter allows for fuzzy comparisons, which is especially important for floating-point aggregations where exact equality is often unreliable.

The `tol=` parameter expands the acceptable range for the comparison. For [col_avg_le()](Validate.col_avg_le.md#pointblank.Validate.col_avg_le), a tolerance of `tol=0.5` would mean the average can be within `0.5` of the target value and still pass validation.

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


Let's validate that the average of column `a` is at most `3`:


``` python
validation = (
    pb.Validate(data=tbl)
    .col_avg_le(columns="a", value=3)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfYXZnX2xlPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9hdmdfbGUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9ImF2ZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjAuMDU4MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yLjk0LDcuMTQgQzIuMDM0NjY2NjcsNy4xNCAxLjMxODMzMzMzLDYuOTM0NjY2NjcgMC43OTEsNi41MjQgQzAuMjYzNjY2NjY3LDYuMTEzMzMzMzMgMCw1LjU1OCAwLDQuODU4IEMwLDQuMTU4IDAuMjc3NjY2NjY3LDMuNjAyNjY2NjcgMC44MzMsMy4xOTIgQzEuMzg4MzMzMzMsMi43ODEzMzMzMyAyLjE0NjY2NjY3LDIuNTc2IDMuMTA4LDIuNTc2IEMzLjYxMiwyLjU3NiA0LjA2MjMzMzMzLDIuNjQxMzMzMzMgNC40NTksMi43NzIgQzQuODU1NjY2NjcsMi45MDI2NjY2NyA1LjE3NTMzMzMzLDMuMDggNS40MTgsMy4zMDQgTDUuNDE4LDIuNjA0IEM1LjQxOCwyLjEgNS4yNTcsMS43MTUgNC45MzUsMS40NDkgQzQuNjEzLDEuMTgzIDQuMTM5MzMzMzMsMS4wNSAzLjUxNCwxLjA1IEMyLjY4MzMzMzMzLDEuMDUgMS45NjQ2NjY2NywxLjI0MTMzMzMzIDEuMzU4LDEuNjI0IEMxLjE4MDY2NjY3LDEuNzI2NjY2NjcgMS4wMjY2NjY2NywxLjc3OCAwLjg5NiwxLjc3OCBDMC43NzQ2NjY2NjcsMS43NzggMC42NTMzMzMzMzMsMS43MjY2NjY2NyAwLjUzMiwxLjYyNCBDMC40MjkzMzMzMzMsMS41MjEzMzMzMyAwLjM3OCwxLjQgMC4zNzgsMS4yNiBDMC4zNzgsMS4xMDEzMzMzMyAwLjQ1NzMzMzMzMywwLjk1NjY2NjY2NyAwLjYxNiwwLjgyNiBDMC45NTIsMC41NzQgMS4zNzY2NjY2NywwLjM3MzMzMzMzMyAxLjg5LDAuMjI0IEMyLjQwMzMzMzMzLDAuMDc0NjY2NjY2NyAyLjk0LDAgMy41LDAgQzQuNTE3MzMzMzMsMCA1LjMxMDY2NjY3LDAuMjMxIDUuODgsMC42OTMgQzYuNDQ5MzMzMzMsMS4xNTUgNi43MzQsMS44MDEzMzMzMyA2LjczNCwyLjYzMiBMNi43MzQsNi41MSBDNi43MzQsNi43MTUzMzMzMyA2LjY4MDMzMzMzLDYuODcxNjY2NjcgNi41NzMsNi45NzkgQzYuNDY1NjY2NjcsNy4wODYzMzMzMyA2LjMwOTMzMzMzLDcuMTQgNi4xMDQsNy4xNCBDNS44OTg2NjY2Nyw3LjE0IDUuNzQsNy4wODYzMzMzMyA1LjYyOCw2Ljk3OSBDNS41MTYsNi44NzE2NjY2NyA1LjQ2LDYuNzI0NjY2NjcgNS40Niw2LjUzOCBMNS40Niw2LjE0NiBDNC45LDYuODA4NjY2NjcgNC4wNiw3LjE0IDIuOTQsNy4xNCBaIE0zLjE2NCw2LjE0NiBDMy44NDUzMzMzMyw2LjE0NiA0LjM5MzY2NjY3LDYuMDIyMzMzMzMgNC44MDksNS43NzUgQzUuMjI0MzMzMzMsNS41Mjc2NjY2NyA1LjQzMiw1LjIwMzMzMzMzIDUuNDMyLDQuODAyIEM1LjQzMiw0LjQxIDUuMjMzNjY2NjcsNC4xMDQzMzMzMyA0LjgzNywzLjg4NSBDNC40NDAzMzMzMywzLjY2NTY2NjY3IDMuODkyLDMuNTU2IDMuMTkyLDMuNTU2IEMyLjU4NTMzMzMzLDMuNTU2IDIuMTE4NjY2NjcsMy42NjggMS43OTIsMy44OTIgQzEuNDY1MzMzMzMsNC4xMTYgMS4zMDIsNC40MzggMS4zMDIsNC44NTggQzEuMzAyLDUuMjY4NjY2NjcgMS40NjMsNS41ODYgMS43ODUsNS44MSBDMi4xMDcsNi4wMzQgMi41NjY2NjY2Nyw2LjE0NiAzLjE2NCw2LjE0NiBaIiBpZD0iYSIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS40OTQsNy4xNCBDMTAuNTMyNjY2Nyw3LjE0IDkuODUxMzMzMzMsNi41NjEzMzMzMyA5LjQ1LDUuNDA0IEw3Ljc5OCwwLjc4NCBDNy43NywwLjcwOTMzMzMzMyA3Ljc1NiwwLjY0NCA3Ljc1NiwwLjU4OCBDNy43NTYsMC40NjY2NjY2NjcgNy43OTgsMC4zNTcgNy44ODIsMC4yNTkgQzcuOTY2LDAuMTYxIDguMDc4LDAuMDg4NjY2NjY2NyA4LjIxOCwwLjA0MiBDOC4zMDIsMC4wMTQgOC4zODYsMCA4LjQ3LDAgQzguNjEsMCA4LjczMTMzMzMzLDAuMDM1IDguODM0LDAuMTA1IEM4LjkzNjY2NjY3LDAuMTc1IDkuMDA2NjY2NjcsMC4yNzA2NjY2NjcgOS4wNDQsMC4zOTIgTDEwLjczOCw1LjEzOCBDMTAuODU5MzMzMyw1LjQ4MzMzMzMzIDEwLjk3Niw1LjcyNiAxMS4wODgsNS44NjYgQzExLjIsNi4wMDYgMTEuMzM1MzMzMyw2LjA3NiAxMS40OTQsNi4wNzYgQzExLjY1MjY2NjcsNi4wNzYgMTEuNzg4LDYuMDA4MzMzMzMgMTEuOSw1Ljg3MyBDMTIuMDEyLDUuNzM3NjY2NjcgMTIuMTI4NjY2Nyw1LjUwMiAxMi4yNSw1LjE2NiBMMTMuOTQ0LDAuMzkyIEMxMy45OTA2NjY3LDAuMjcwNjY2NjY3IDE0LjA2MywwLjE3NzMzMzMzMyAxNC4xNjEsMC4xMTIgQzE0LjI1OSwwLjA0NjY2NjY2NjcgMTQuMzc4LDAuMDE0IDE0LjUxOCwwLjAxNCBDMTQuNzMyNjY2NywwLjAxNCAxNC45MDUzMzMzLDAuMDcyMzMzMzMzMyAxNS4wMzYsMC4xODkgQzE1LjE2NjY2NjcsMC4zMDU2NjY2NjcgMTUuMjMyLDAuNDQzMzMzMzMzIDE1LjIzMiwwLjYwMiBDMTUuMjMyLDAuNjg2IDE1LjIyMjY2NjcsMC43NDY2NjY2NjcgMTUuMjA0LDAuNzg0IEwxMy41MzgsNS40MzIgQzEzLjMyMzMzMzMsNi4wMTA2NjY2NyAxMy4wNTAzMzMzLDYuNDQgMTIuNzE5LDYuNzIgQzEyLjM4NzY2NjcsNyAxMS45NzkzMzMzLDcuMTQgMTEuNDk0LDcuMTQgWiIgaWQ9InYiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTkuODEsMTAuNTE0IEMxOS4xMTkzMzMzLDEwLjUxNCAxOC40OTE2NjY3LDEwLjQxMTMzMzMgMTcuOTI3LDEwLjIwNiBDMTcuMzYyMzMzMywxMC4wMDA2NjY3IDE2LjkwNzMzMzMsOS43MTEzMzMzMyAxNi41NjIsOS4zMzggQzE2LjQ2ODY2NjcsOS4yMzUzMzMzMyAxNi40MjIsOS4xMjMzMzMzMyAxNi40MjIsOS4wMDIgQzE2LjQyMiw4Ljg0MzMzMzMzIDE2LjUxMDY2NjcsOC42ODkzMzMzMyAxNi42ODgsOC41NCBDMTYuNzcyLDguNDY1MzMzMzMgMTYuODc5MzMzMyw4LjQyOCAxNy4wMSw4LjQyOCBDMTcuMjA2LDguNDI4IDE3LjM5MjY2NjcsOC41MjEzMzMzMyAxNy41Nyw4LjcwOCBDMTcuNzk0LDguOTQxMzMzMzMgMTguMDg1NjY2Nyw5LjEyNTY2NjY3IDE4LjQ0NSw5LjI2MSBDMTguODA0MzMzMyw5LjM5NjMzMzMzIDE5LjIyNjY2NjcsOS40NjQgMTkuNzEyLDkuNDY0IEMyMC41NjEzMzMzLDkuNDY0IDIxLjIxNDY2NjcsOS4yNDQ2NjY2NyAyMS42NzIsOC44MDYgQzIyLjEyOTMzMzMsOC4zNjczMzMzMyAyMi4zNTMzMzMzLDcuNzQyIDIyLjM0NCw2LjkzIEwyMi4zNDQsNS44NjYgQzIyLjAyNjY2NjcsNi4yNjczMzMzMyAyMS42MjUzMzMzLDYuNTg3IDIxLjE0LDYuODI1IEMyMC42NTQ2NjY3LDcuMDYzIDIwLjA4NTMzMzMsNy4xODIgMTkuNDMyLDcuMTgyIEMxOC4yNTYsNy4xODIgMTcuMzQ2LDYuODY3IDE2LjcwMiw2LjIzNyBDMTYuMDU4LDUuNjA3IDE1LjczNiw0Ljc1MDY2NjY3IDE1LjczNiwzLjY2OCBDMTUuNzM2LDIuNTU3MzMzMzMgMTYuMDYyNjY2NywxLjY2ODMzMzMzIDE2LjcxNiwxLjAwMSBDMTcuMzY5MzMzMywwLjMzMzY2NjY2NyAxOC4zMDI2NjY3LDAgMTkuNTE2LDAgQzIwLjA5NDY2NjcsMCAyMC42MjIsMC4xMTY2NjY2NjcgMjEuMDk4LDAuMzUgQzIxLjU3NCwwLjU4MzMzMzMzMyAyMS45ODkzMzMzLDAuOTE0NjY2NjY3IDIyLjM0NCwxLjM0NCBMMjIuMzQ0LDAuNjMgQzIyLjM0NCwwLjQ0MzMzMzMzMyAyMi40MDcsMC4yOTE2NjY2NjcgMjIuNTMzLDAuMTc1IEMyMi42NTksMC4wNTgzMzMzMzMzIDIyLjgxNTMzMzMsMCAyMy4wMDIsMCBDMjMuMTk4LDAgMjMuMzU0MzMzMywwLjA1NiAyMy40NzEsMC4xNjggQzIzLjU4NzY2NjcsMC4yOCAyMy42NDYsMC40MzQgMjMuNjQ2LDAuNjMgTDIzLjY0Niw2LjkzIEMyMy42NDYsNy42NDg2NjY2NyAyMy40ODczMzMzLDguMjc4NjY2NjcgMjMuMTcsOC44MiBDMjIuODUyNjY2Nyw5LjM2MTMzMzMzIDIyLjQwNDY2NjcsOS43NzkgMjEuODI2LDEwLjA3MyBDMjEuMjQ3MzMzMywxMC4zNjcgMjAuNTc1MzMzMywxMC41MTQgMTkuODEsMTAuNTE0IFogTTE5LjYxNCw2LjA5IEMyMC4xMDg2NjY3LDYuMDkgMjAuNTYzNjY2Nyw1Ljk5NjY2NjY3IDIwLjk3OSw1LjgxIEMyMS4zOTQzMzMzLDUuNjIzMzMzMzMgMjEuNzI1NjY2Nyw1LjM0MzMzMzMzIDIxLjk3Myw0Ljk3IEMyMi4yMjAzMzMzLDQuNTk2NjY2NjcgMjIuMzQ0LDQuMTM5MzMzMzMgMjIuMzQ0LDMuNTk4IEMyMi4zNDQsMi44NTEzMzMzMyAyMi4xMDYsMi4yNDQ2NjY2NyAyMS42MywxLjc3OCBDMjEuMTU0LDEuMzExMzMzMzMgMjAuNDk2LDEuMDc4IDE5LjY1NiwxLjA3OCBDMTguODI1MzMzMywxLjA3OCAxOC4xODEzMzMzLDEuMzA0MzMzMzMgMTcuNzI0LDEuNzU3IEMxNy4yNjY2NjY3LDIuMjA5NjY2NjcgMTcuMDM4LDIuODQ2NjY2NjcgMTcuMDM4LDMuNjY4IEMxNy4wMzgsNC40MTQ2NjY2NyAxNy4yNTk2NjY3LDUuMDA1IDE3LjcwMyw1LjQzOSBDMTguMTQ2MzMzMyw1Ljg3MyAxOC43ODMzMzMzLDYuMDkgMTkuNjE0LDYuMDkgWiIgaWQ9ImciIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPHBhdGggZD0iTTQ2LjY4MTgxODIsMTAgTDE3LjMxODE4MTgsMTAgQzE2LjA0MTM3MDYsMTAgMTUsMTEuMDQxMzcwNSAxNSwxMi4zMTgxODE4IEwxNSw0MS42ODE4MTgyIEMxNSw0Mi45NTg2Mjk1IDE2LjA0MTM3MDYsNDQgMTcuMzE4MTgxOCw0NCBMNDYuNjgxODE4Miw0NCBDNDcuOTU4NjI5NSw0NCA0OSw0Mi45NTg2Mjk1IDQ5LDQxLjY4MTgxODIgTDQ5LDEyLjMxODE4MTggQzQ5LDExLjA0MTM3MDUgNDcuOTU4NjI5NSwxMCA0Ni42ODE4MTgyLDEwIFogTTM4LjU4MDI1NTcsMTYuMjc4NDA5MSBMMzkuMzI4ODM1MiwxNy42MzA2ODE4IEwyNi42MzYxODYzLDI0LjY4MTgxODIgTDM5LjMyODgzNTIsMzEuNzMyOTU0NSBMMzguNTgwMjU1NywzMy4wODUyMjczIEwyMy40NTQ3MjI4LDI0LjY4MTgxODIgTDM4LjU4MDI1NTcsMTYuMjc4NDA5MSBaIE00MC41LDM3LjgxODE4MTggTDIzLjUsMzcuODE4MTgxOCBMMjMuNSwzNi4yNzI3MjczIEw0MC41LDM2LjI3MjcyNzMgTDQwLjUsMzcuODE4MTgxOCBaIiBpZD0ibGVzc190aGFuX2VxdWFsIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_avg_le()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">3</td>
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


The validation result shows whether the average comparison passed or failed. Since this is an aggregation-based validation, there is exactly one test unit per column.

When validating multiple columns, each column gets its own validation step:


``` python
validation = (
    pb.Validate(data=tbl)
    .col_avg_le(columns=["a", "b"], value=3)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfYXZnX2xlPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9hdmdfbGUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9ImF2ZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjAuMDU4MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yLjk0LDcuMTQgQzIuMDM0NjY2NjcsNy4xNCAxLjMxODMzMzMzLDYuOTM0NjY2NjcgMC43OTEsNi41MjQgQzAuMjYzNjY2NjY3LDYuMTEzMzMzMzMgMCw1LjU1OCAwLDQuODU4IEMwLDQuMTU4IDAuMjc3NjY2NjY3LDMuNjAyNjY2NjcgMC44MzMsMy4xOTIgQzEuMzg4MzMzMzMsMi43ODEzMzMzMyAyLjE0NjY2NjY3LDIuNTc2IDMuMTA4LDIuNTc2IEMzLjYxMiwyLjU3NiA0LjA2MjMzMzMzLDIuNjQxMzMzMzMgNC40NTksMi43NzIgQzQuODU1NjY2NjcsMi45MDI2NjY2NyA1LjE3NTMzMzMzLDMuMDggNS40MTgsMy4zMDQgTDUuNDE4LDIuNjA0IEM1LjQxOCwyLjEgNS4yNTcsMS43MTUgNC45MzUsMS40NDkgQzQuNjEzLDEuMTgzIDQuMTM5MzMzMzMsMS4wNSAzLjUxNCwxLjA1IEMyLjY4MzMzMzMzLDEuMDUgMS45NjQ2NjY2NywxLjI0MTMzMzMzIDEuMzU4LDEuNjI0IEMxLjE4MDY2NjY3LDEuNzI2NjY2NjcgMS4wMjY2NjY2NywxLjc3OCAwLjg5NiwxLjc3OCBDMC43NzQ2NjY2NjcsMS43NzggMC42NTMzMzMzMzMsMS43MjY2NjY2NyAwLjUzMiwxLjYyNCBDMC40MjkzMzMzMzMsMS41MjEzMzMzMyAwLjM3OCwxLjQgMC4zNzgsMS4yNiBDMC4zNzgsMS4xMDEzMzMzMyAwLjQ1NzMzMzMzMywwLjk1NjY2NjY2NyAwLjYxNiwwLjgyNiBDMC45NTIsMC41NzQgMS4zNzY2NjY2NywwLjM3MzMzMzMzMyAxLjg5LDAuMjI0IEMyLjQwMzMzMzMzLDAuMDc0NjY2NjY2NyAyLjk0LDAgMy41LDAgQzQuNTE3MzMzMzMsMCA1LjMxMDY2NjY3LDAuMjMxIDUuODgsMC42OTMgQzYuNDQ5MzMzMzMsMS4xNTUgNi43MzQsMS44MDEzMzMzMyA2LjczNCwyLjYzMiBMNi43MzQsNi41MSBDNi43MzQsNi43MTUzMzMzMyA2LjY4MDMzMzMzLDYuODcxNjY2NjcgNi41NzMsNi45NzkgQzYuNDY1NjY2NjcsNy4wODYzMzMzMyA2LjMwOTMzMzMzLDcuMTQgNi4xMDQsNy4xNCBDNS44OTg2NjY2Nyw3LjE0IDUuNzQsNy4wODYzMzMzMyA1LjYyOCw2Ljk3OSBDNS41MTYsNi44NzE2NjY2NyA1LjQ2LDYuNzI0NjY2NjcgNS40Niw2LjUzOCBMNS40Niw2LjE0NiBDNC45LDYuODA4NjY2NjcgNC4wNiw3LjE0IDIuOTQsNy4xNCBaIE0zLjE2NCw2LjE0NiBDMy44NDUzMzMzMyw2LjE0NiA0LjM5MzY2NjY3LDYuMDIyMzMzMzMgNC44MDksNS43NzUgQzUuMjI0MzMzMzMsNS41Mjc2NjY2NyA1LjQzMiw1LjIwMzMzMzMzIDUuNDMyLDQuODAyIEM1LjQzMiw0LjQxIDUuMjMzNjY2NjcsNC4xMDQzMzMzMyA0LjgzNywzLjg4NSBDNC40NDAzMzMzMywzLjY2NTY2NjY3IDMuODkyLDMuNTU2IDMuMTkyLDMuNTU2IEMyLjU4NTMzMzMzLDMuNTU2IDIuMTE4NjY2NjcsMy42NjggMS43OTIsMy44OTIgQzEuNDY1MzMzMzMsNC4xMTYgMS4zMDIsNC40MzggMS4zMDIsNC44NTggQzEuMzAyLDUuMjY4NjY2NjcgMS40NjMsNS41ODYgMS43ODUsNS44MSBDMi4xMDcsNi4wMzQgMi41NjY2NjY2Nyw2LjE0NiAzLjE2NCw2LjE0NiBaIiBpZD0iYSIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS40OTQsNy4xNCBDMTAuNTMyNjY2Nyw3LjE0IDkuODUxMzMzMzMsNi41NjEzMzMzMyA5LjQ1LDUuNDA0IEw3Ljc5OCwwLjc4NCBDNy43NywwLjcwOTMzMzMzMyA3Ljc1NiwwLjY0NCA3Ljc1NiwwLjU4OCBDNy43NTYsMC40NjY2NjY2NjcgNy43OTgsMC4zNTcgNy44ODIsMC4yNTkgQzcuOTY2LDAuMTYxIDguMDc4LDAuMDg4NjY2NjY2NyA4LjIxOCwwLjA0MiBDOC4zMDIsMC4wMTQgOC4zODYsMCA4LjQ3LDAgQzguNjEsMCA4LjczMTMzMzMzLDAuMDM1IDguODM0LDAuMTA1IEM4LjkzNjY2NjY3LDAuMTc1IDkuMDA2NjY2NjcsMC4yNzA2NjY2NjcgOS4wNDQsMC4zOTIgTDEwLjczOCw1LjEzOCBDMTAuODU5MzMzMyw1LjQ4MzMzMzMzIDEwLjk3Niw1LjcyNiAxMS4wODgsNS44NjYgQzExLjIsNi4wMDYgMTEuMzM1MzMzMyw2LjA3NiAxMS40OTQsNi4wNzYgQzExLjY1MjY2NjcsNi4wNzYgMTEuNzg4LDYuMDA4MzMzMzMgMTEuOSw1Ljg3MyBDMTIuMDEyLDUuNzM3NjY2NjcgMTIuMTI4NjY2Nyw1LjUwMiAxMi4yNSw1LjE2NiBMMTMuOTQ0LDAuMzkyIEMxMy45OTA2NjY3LDAuMjcwNjY2NjY3IDE0LjA2MywwLjE3NzMzMzMzMyAxNC4xNjEsMC4xMTIgQzE0LjI1OSwwLjA0NjY2NjY2NjcgMTQuMzc4LDAuMDE0IDE0LjUxOCwwLjAxNCBDMTQuNzMyNjY2NywwLjAxNCAxNC45MDUzMzMzLDAuMDcyMzMzMzMzMyAxNS4wMzYsMC4xODkgQzE1LjE2NjY2NjcsMC4zMDU2NjY2NjcgMTUuMjMyLDAuNDQzMzMzMzMzIDE1LjIzMiwwLjYwMiBDMTUuMjMyLDAuNjg2IDE1LjIyMjY2NjcsMC43NDY2NjY2NjcgMTUuMjA0LDAuNzg0IEwxMy41MzgsNS40MzIgQzEzLjMyMzMzMzMsNi4wMTA2NjY2NyAxMy4wNTAzMzMzLDYuNDQgMTIuNzE5LDYuNzIgQzEyLjM4NzY2NjcsNyAxMS45NzkzMzMzLDcuMTQgMTEuNDk0LDcuMTQgWiIgaWQ9InYiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTkuODEsMTAuNTE0IEMxOS4xMTkzMzMzLDEwLjUxNCAxOC40OTE2NjY3LDEwLjQxMTMzMzMgMTcuOTI3LDEwLjIwNiBDMTcuMzYyMzMzMywxMC4wMDA2NjY3IDE2LjkwNzMzMzMsOS43MTEzMzMzMyAxNi41NjIsOS4zMzggQzE2LjQ2ODY2NjcsOS4yMzUzMzMzMyAxNi40MjIsOS4xMjMzMzMzMyAxNi40MjIsOS4wMDIgQzE2LjQyMiw4Ljg0MzMzMzMzIDE2LjUxMDY2NjcsOC42ODkzMzMzMyAxNi42ODgsOC41NCBDMTYuNzcyLDguNDY1MzMzMzMgMTYuODc5MzMzMyw4LjQyOCAxNy4wMSw4LjQyOCBDMTcuMjA2LDguNDI4IDE3LjM5MjY2NjcsOC41MjEzMzMzMyAxNy41Nyw4LjcwOCBDMTcuNzk0LDguOTQxMzMzMzMgMTguMDg1NjY2Nyw5LjEyNTY2NjY3IDE4LjQ0NSw5LjI2MSBDMTguODA0MzMzMyw5LjM5NjMzMzMzIDE5LjIyNjY2NjcsOS40NjQgMTkuNzEyLDkuNDY0IEMyMC41NjEzMzMzLDkuNDY0IDIxLjIxNDY2NjcsOS4yNDQ2NjY2NyAyMS42NzIsOC44MDYgQzIyLjEyOTMzMzMsOC4zNjczMzMzMyAyMi4zNTMzMzMzLDcuNzQyIDIyLjM0NCw2LjkzIEwyMi4zNDQsNS44NjYgQzIyLjAyNjY2NjcsNi4yNjczMzMzMyAyMS42MjUzMzMzLDYuNTg3IDIxLjE0LDYuODI1IEMyMC42NTQ2NjY3LDcuMDYzIDIwLjA4NTMzMzMsNy4xODIgMTkuNDMyLDcuMTgyIEMxOC4yNTYsNy4xODIgMTcuMzQ2LDYuODY3IDE2LjcwMiw2LjIzNyBDMTYuMDU4LDUuNjA3IDE1LjczNiw0Ljc1MDY2NjY3IDE1LjczNiwzLjY2OCBDMTUuNzM2LDIuNTU3MzMzMzMgMTYuMDYyNjY2NywxLjY2ODMzMzMzIDE2LjcxNiwxLjAwMSBDMTcuMzY5MzMzMywwLjMzMzY2NjY2NyAxOC4zMDI2NjY3LDAgMTkuNTE2LDAgQzIwLjA5NDY2NjcsMCAyMC42MjIsMC4xMTY2NjY2NjcgMjEuMDk4LDAuMzUgQzIxLjU3NCwwLjU4MzMzMzMzMyAyMS45ODkzMzMzLDAuOTE0NjY2NjY3IDIyLjM0NCwxLjM0NCBMMjIuMzQ0LDAuNjMgQzIyLjM0NCwwLjQ0MzMzMzMzMyAyMi40MDcsMC4yOTE2NjY2NjcgMjIuNTMzLDAuMTc1IEMyMi42NTksMC4wNTgzMzMzMzMzIDIyLjgxNTMzMzMsMCAyMy4wMDIsMCBDMjMuMTk4LDAgMjMuMzU0MzMzMywwLjA1NiAyMy40NzEsMC4xNjggQzIzLjU4NzY2NjcsMC4yOCAyMy42NDYsMC40MzQgMjMuNjQ2LDAuNjMgTDIzLjY0Niw2LjkzIEMyMy42NDYsNy42NDg2NjY2NyAyMy40ODczMzMzLDguMjc4NjY2NjcgMjMuMTcsOC44MiBDMjIuODUyNjY2Nyw5LjM2MTMzMzMzIDIyLjQwNDY2NjcsOS43NzkgMjEuODI2LDEwLjA3MyBDMjEuMjQ3MzMzMywxMC4zNjcgMjAuNTc1MzMzMywxMC41MTQgMTkuODEsMTAuNTE0IFogTTE5LjYxNCw2LjA5IEMyMC4xMDg2NjY3LDYuMDkgMjAuNTYzNjY2Nyw1Ljk5NjY2NjY3IDIwLjk3OSw1LjgxIEMyMS4zOTQzMzMzLDUuNjIzMzMzMzMgMjEuNzI1NjY2Nyw1LjM0MzMzMzMzIDIxLjk3Myw0Ljk3IEMyMi4yMjAzMzMzLDQuNTk2NjY2NjcgMjIuMzQ0LDQuMTM5MzMzMzMgMjIuMzQ0LDMuNTk4IEMyMi4zNDQsMi44NTEzMzMzMyAyMi4xMDYsMi4yNDQ2NjY2NyAyMS42MywxLjc3OCBDMjEuMTU0LDEuMzExMzMzMzMgMjAuNDk2LDEuMDc4IDE5LjY1NiwxLjA3OCBDMTguODI1MzMzMywxLjA3OCAxOC4xODEzMzMzLDEuMzA0MzMzMzMgMTcuNzI0LDEuNzU3IEMxNy4yNjY2NjY3LDIuMjA5NjY2NjcgMTcuMDM4LDIuODQ2NjY2NjcgMTcuMDM4LDMuNjY4IEMxNy4wMzgsNC40MTQ2NjY2NyAxNy4yNTk2NjY3LDUuMDA1IDE3LjcwMyw1LjQzOSBDMTguMTQ2MzMzMyw1Ljg3MyAxOC43ODMzMzMzLDYuMDkgMTkuNjE0LDYuMDkgWiIgaWQ9ImciIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPHBhdGggZD0iTTQ2LjY4MTgxODIsMTAgTDE3LjMxODE4MTgsMTAgQzE2LjA0MTM3MDYsMTAgMTUsMTEuMDQxMzcwNSAxNSwxMi4zMTgxODE4IEwxNSw0MS42ODE4MTgyIEMxNSw0Mi45NTg2Mjk1IDE2LjA0MTM3MDYsNDQgMTcuMzE4MTgxOCw0NCBMNDYuNjgxODE4Miw0NCBDNDcuOTU4NjI5NSw0NCA0OSw0Mi45NTg2Mjk1IDQ5LDQxLjY4MTgxODIgTDQ5LDEyLjMxODE4MTggQzQ5LDExLjA0MTM3MDUgNDcuOTU4NjI5NSwxMCA0Ni42ODE4MTgyLDEwIFogTTM4LjU4MDI1NTcsMTYuMjc4NDA5MSBMMzkuMzI4ODM1MiwxNy42MzA2ODE4IEwyNi42MzYxODYzLDI0LjY4MTgxODIgTDM5LjMyODgzNTIsMzEuNzMyOTU0NSBMMzguNTgwMjU1NywzMy4wODUyMjczIEwyMy40NTQ3MjI4LDI0LjY4MTgxODIgTDM4LjU4MDI1NTcsMTYuMjc4NDA5MSBaIE00MC41LDM3LjgxODE4MTggTDIzLjUsMzcuODE4MTgxOCBMMjMuNSwzNi4yNzI3MjczIEw0MC41LDM2LjI3MjcyNzMgTDQwLjUsMzcuODE4MTgxOCBaIiBpZD0ibGVzc190aGFuX2VxdWFsIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_avg_le()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">3</td>
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfYXZnX2xlPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9hdmdfbGUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9ImF2ZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjAuMDU4MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yLjk0LDcuMTQgQzIuMDM0NjY2NjcsNy4xNCAxLjMxODMzMzMzLDYuOTM0NjY2NjcgMC43OTEsNi41MjQgQzAuMjYzNjY2NjY3LDYuMTEzMzMzMzMgMCw1LjU1OCAwLDQuODU4IEMwLDQuMTU4IDAuMjc3NjY2NjY3LDMuNjAyNjY2NjcgMC44MzMsMy4xOTIgQzEuMzg4MzMzMzMsMi43ODEzMzMzMyAyLjE0NjY2NjY3LDIuNTc2IDMuMTA4LDIuNTc2IEMzLjYxMiwyLjU3NiA0LjA2MjMzMzMzLDIuNjQxMzMzMzMgNC40NTksMi43NzIgQzQuODU1NjY2NjcsMi45MDI2NjY2NyA1LjE3NTMzMzMzLDMuMDggNS40MTgsMy4zMDQgTDUuNDE4LDIuNjA0IEM1LjQxOCwyLjEgNS4yNTcsMS43MTUgNC45MzUsMS40NDkgQzQuNjEzLDEuMTgzIDQuMTM5MzMzMzMsMS4wNSAzLjUxNCwxLjA1IEMyLjY4MzMzMzMzLDEuMDUgMS45NjQ2NjY2NywxLjI0MTMzMzMzIDEuMzU4LDEuNjI0IEMxLjE4MDY2NjY3LDEuNzI2NjY2NjcgMS4wMjY2NjY2NywxLjc3OCAwLjg5NiwxLjc3OCBDMC43NzQ2NjY2NjcsMS43NzggMC42NTMzMzMzMzMsMS43MjY2NjY2NyAwLjUzMiwxLjYyNCBDMC40MjkzMzMzMzMsMS41MjEzMzMzMyAwLjM3OCwxLjQgMC4zNzgsMS4yNiBDMC4zNzgsMS4xMDEzMzMzMyAwLjQ1NzMzMzMzMywwLjk1NjY2NjY2NyAwLjYxNiwwLjgyNiBDMC45NTIsMC41NzQgMS4zNzY2NjY2NywwLjM3MzMzMzMzMyAxLjg5LDAuMjI0IEMyLjQwMzMzMzMzLDAuMDc0NjY2NjY2NyAyLjk0LDAgMy41LDAgQzQuNTE3MzMzMzMsMCA1LjMxMDY2NjY3LDAuMjMxIDUuODgsMC42OTMgQzYuNDQ5MzMzMzMsMS4xNTUgNi43MzQsMS44MDEzMzMzMyA2LjczNCwyLjYzMiBMNi43MzQsNi41MSBDNi43MzQsNi43MTUzMzMzMyA2LjY4MDMzMzMzLDYuODcxNjY2NjcgNi41NzMsNi45NzkgQzYuNDY1NjY2NjcsNy4wODYzMzMzMyA2LjMwOTMzMzMzLDcuMTQgNi4xMDQsNy4xNCBDNS44OTg2NjY2Nyw3LjE0IDUuNzQsNy4wODYzMzMzMyA1LjYyOCw2Ljk3OSBDNS41MTYsNi44NzE2NjY2NyA1LjQ2LDYuNzI0NjY2NjcgNS40Niw2LjUzOCBMNS40Niw2LjE0NiBDNC45LDYuODA4NjY2NjcgNC4wNiw3LjE0IDIuOTQsNy4xNCBaIE0zLjE2NCw2LjE0NiBDMy44NDUzMzMzMyw2LjE0NiA0LjM5MzY2NjY3LDYuMDIyMzMzMzMgNC44MDksNS43NzUgQzUuMjI0MzMzMzMsNS41Mjc2NjY2NyA1LjQzMiw1LjIwMzMzMzMzIDUuNDMyLDQuODAyIEM1LjQzMiw0LjQxIDUuMjMzNjY2NjcsNC4xMDQzMzMzMyA0LjgzNywzLjg4NSBDNC40NDAzMzMzMywzLjY2NTY2NjY3IDMuODkyLDMuNTU2IDMuMTkyLDMuNTU2IEMyLjU4NTMzMzMzLDMuNTU2IDIuMTE4NjY2NjcsMy42NjggMS43OTIsMy44OTIgQzEuNDY1MzMzMzMsNC4xMTYgMS4zMDIsNC40MzggMS4zMDIsNC44NTggQzEuMzAyLDUuMjY4NjY2NjcgMS40NjMsNS41ODYgMS43ODUsNS44MSBDMi4xMDcsNi4wMzQgMi41NjY2NjY2Nyw2LjE0NiAzLjE2NCw2LjE0NiBaIiBpZD0iYSIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS40OTQsNy4xNCBDMTAuNTMyNjY2Nyw3LjE0IDkuODUxMzMzMzMsNi41NjEzMzMzMyA5LjQ1LDUuNDA0IEw3Ljc5OCwwLjc4NCBDNy43NywwLjcwOTMzMzMzMyA3Ljc1NiwwLjY0NCA3Ljc1NiwwLjU4OCBDNy43NTYsMC40NjY2NjY2NjcgNy43OTgsMC4zNTcgNy44ODIsMC4yNTkgQzcuOTY2LDAuMTYxIDguMDc4LDAuMDg4NjY2NjY2NyA4LjIxOCwwLjA0MiBDOC4zMDIsMC4wMTQgOC4zODYsMCA4LjQ3LDAgQzguNjEsMCA4LjczMTMzMzMzLDAuMDM1IDguODM0LDAuMTA1IEM4LjkzNjY2NjY3LDAuMTc1IDkuMDA2NjY2NjcsMC4yNzA2NjY2NjcgOS4wNDQsMC4zOTIgTDEwLjczOCw1LjEzOCBDMTAuODU5MzMzMyw1LjQ4MzMzMzMzIDEwLjk3Niw1LjcyNiAxMS4wODgsNS44NjYgQzExLjIsNi4wMDYgMTEuMzM1MzMzMyw2LjA3NiAxMS40OTQsNi4wNzYgQzExLjY1MjY2NjcsNi4wNzYgMTEuNzg4LDYuMDA4MzMzMzMgMTEuOSw1Ljg3MyBDMTIuMDEyLDUuNzM3NjY2NjcgMTIuMTI4NjY2Nyw1LjUwMiAxMi4yNSw1LjE2NiBMMTMuOTQ0LDAuMzkyIEMxMy45OTA2NjY3LDAuMjcwNjY2NjY3IDE0LjA2MywwLjE3NzMzMzMzMyAxNC4xNjEsMC4xMTIgQzE0LjI1OSwwLjA0NjY2NjY2NjcgMTQuMzc4LDAuMDE0IDE0LjUxOCwwLjAxNCBDMTQuNzMyNjY2NywwLjAxNCAxNC45MDUzMzMzLDAuMDcyMzMzMzMzMyAxNS4wMzYsMC4xODkgQzE1LjE2NjY2NjcsMC4zMDU2NjY2NjcgMTUuMjMyLDAuNDQzMzMzMzMzIDE1LjIzMiwwLjYwMiBDMTUuMjMyLDAuNjg2IDE1LjIyMjY2NjcsMC43NDY2NjY2NjcgMTUuMjA0LDAuNzg0IEwxMy41MzgsNS40MzIgQzEzLjMyMzMzMzMsNi4wMTA2NjY2NyAxMy4wNTAzMzMzLDYuNDQgMTIuNzE5LDYuNzIgQzEyLjM4NzY2NjcsNyAxMS45NzkzMzMzLDcuMTQgMTEuNDk0LDcuMTQgWiIgaWQ9InYiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTkuODEsMTAuNTE0IEMxOS4xMTkzMzMzLDEwLjUxNCAxOC40OTE2NjY3LDEwLjQxMTMzMzMgMTcuOTI3LDEwLjIwNiBDMTcuMzYyMzMzMywxMC4wMDA2NjY3IDE2LjkwNzMzMzMsOS43MTEzMzMzMyAxNi41NjIsOS4zMzggQzE2LjQ2ODY2NjcsOS4yMzUzMzMzMyAxNi40MjIsOS4xMjMzMzMzMyAxNi40MjIsOS4wMDIgQzE2LjQyMiw4Ljg0MzMzMzMzIDE2LjUxMDY2NjcsOC42ODkzMzMzMyAxNi42ODgsOC41NCBDMTYuNzcyLDguNDY1MzMzMzMgMTYuODc5MzMzMyw4LjQyOCAxNy4wMSw4LjQyOCBDMTcuMjA2LDguNDI4IDE3LjM5MjY2NjcsOC41MjEzMzMzMyAxNy41Nyw4LjcwOCBDMTcuNzk0LDguOTQxMzMzMzMgMTguMDg1NjY2Nyw5LjEyNTY2NjY3IDE4LjQ0NSw5LjI2MSBDMTguODA0MzMzMyw5LjM5NjMzMzMzIDE5LjIyNjY2NjcsOS40NjQgMTkuNzEyLDkuNDY0IEMyMC41NjEzMzMzLDkuNDY0IDIxLjIxNDY2NjcsOS4yNDQ2NjY2NyAyMS42NzIsOC44MDYgQzIyLjEyOTMzMzMsOC4zNjczMzMzMyAyMi4zNTMzMzMzLDcuNzQyIDIyLjM0NCw2LjkzIEwyMi4zNDQsNS44NjYgQzIyLjAyNjY2NjcsNi4yNjczMzMzMyAyMS42MjUzMzMzLDYuNTg3IDIxLjE0LDYuODI1IEMyMC42NTQ2NjY3LDcuMDYzIDIwLjA4NTMzMzMsNy4xODIgMTkuNDMyLDcuMTgyIEMxOC4yNTYsNy4xODIgMTcuMzQ2LDYuODY3IDE2LjcwMiw2LjIzNyBDMTYuMDU4LDUuNjA3IDE1LjczNiw0Ljc1MDY2NjY3IDE1LjczNiwzLjY2OCBDMTUuNzM2LDIuNTU3MzMzMzMgMTYuMDYyNjY2NywxLjY2ODMzMzMzIDE2LjcxNiwxLjAwMSBDMTcuMzY5MzMzMywwLjMzMzY2NjY2NyAxOC4zMDI2NjY3LDAgMTkuNTE2LDAgQzIwLjA5NDY2NjcsMCAyMC42MjIsMC4xMTY2NjY2NjcgMjEuMDk4LDAuMzUgQzIxLjU3NCwwLjU4MzMzMzMzMyAyMS45ODkzMzMzLDAuOTE0NjY2NjY3IDIyLjM0NCwxLjM0NCBMMjIuMzQ0LDAuNjMgQzIyLjM0NCwwLjQ0MzMzMzMzMyAyMi40MDcsMC4yOTE2NjY2NjcgMjIuNTMzLDAuMTc1IEMyMi42NTksMC4wNTgzMzMzMzMzIDIyLjgxNTMzMzMsMCAyMy4wMDIsMCBDMjMuMTk4LDAgMjMuMzU0MzMzMywwLjA1NiAyMy40NzEsMC4xNjggQzIzLjU4NzY2NjcsMC4yOCAyMy42NDYsMC40MzQgMjMuNjQ2LDAuNjMgTDIzLjY0Niw2LjkzIEMyMy42NDYsNy42NDg2NjY2NyAyMy40ODczMzMzLDguMjc4NjY2NjcgMjMuMTcsOC44MiBDMjIuODUyNjY2Nyw5LjM2MTMzMzMzIDIyLjQwNDY2NjcsOS43NzkgMjEuODI2LDEwLjA3MyBDMjEuMjQ3MzMzMywxMC4zNjcgMjAuNTc1MzMzMywxMC41MTQgMTkuODEsMTAuNTE0IFogTTE5LjYxNCw2LjA5IEMyMC4xMDg2NjY3LDYuMDkgMjAuNTYzNjY2Nyw1Ljk5NjY2NjY3IDIwLjk3OSw1LjgxIEMyMS4zOTQzMzMzLDUuNjIzMzMzMzMgMjEuNzI1NjY2Nyw1LjM0MzMzMzMzIDIxLjk3Myw0Ljk3IEMyMi4yMjAzMzMzLDQuNTk2NjY2NjcgMjIuMzQ0LDQuMTM5MzMzMzMgMjIuMzQ0LDMuNTk4IEMyMi4zNDQsMi44NTEzMzMzMyAyMi4xMDYsMi4yNDQ2NjY2NyAyMS42MywxLjc3OCBDMjEuMTU0LDEuMzExMzMzMzMgMjAuNDk2LDEuMDc4IDE5LjY1NiwxLjA3OCBDMTguODI1MzMzMywxLjA3OCAxOC4xODEzMzMzLDEuMzA0MzMzMzMgMTcuNzI0LDEuNzU3IEMxNy4yNjY2NjY3LDIuMjA5NjY2NjcgMTcuMDM4LDIuODQ2NjY2NjcgMTcuMDM4LDMuNjY4IEMxNy4wMzgsNC40MTQ2NjY2NyAxNy4yNTk2NjY3LDUuMDA1IDE3LjcwMyw1LjQzOSBDMTguMTQ2MzMzMyw1Ljg3MyAxOC43ODMzMzMzLDYuMDkgMTkuNjE0LDYuMDkgWiIgaWQ9ImciIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPHBhdGggZD0iTTQ2LjY4MTgxODIsMTAgTDE3LjMxODE4MTgsMTAgQzE2LjA0MTM3MDYsMTAgMTUsMTEuMDQxMzcwNSAxNSwxMi4zMTgxODE4IEwxNSw0MS42ODE4MTgyIEMxNSw0Mi45NTg2Mjk1IDE2LjA0MTM3MDYsNDQgMTcuMzE4MTgxOCw0NCBMNDYuNjgxODE4Miw0NCBDNDcuOTU4NjI5NSw0NCA0OSw0Mi45NTg2Mjk1IDQ5LDQxLjY4MTgxODIgTDQ5LDEyLjMxODE4MTggQzQ5LDExLjA0MTM3MDUgNDcuOTU4NjI5NSwxMCA0Ni42ODE4MTgyLDEwIFogTTM4LjU4MDI1NTcsMTYuMjc4NDA5MSBMMzkuMzI4ODM1MiwxNy42MzA2ODE4IEwyNi42MzYxODYzLDI0LjY4MTgxODIgTDM5LjMyODgzNTIsMzEuNzMyOTU0NSBMMzguNTgwMjU1NywzMy4wODUyMjczIEwyMy40NTQ3MjI4LDI0LjY4MTgxODIgTDM4LjU4MDI1NTcsMTYuMjc4NDA5MSBaIE00MC41LDM3LjgxODE4MTggTDIzLjUsMzcuODE4MTgxOCBMMjMuNSwzNi4yNzI3MjczIEw0MC41LDM2LjI3MjcyNzMgTDQwLjUsMzcuODE4MTgxOCBaIiBpZD0ibGVzc190aGFuX2VxdWFsIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_avg_le()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">3</td>
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


Using tolerance for flexible comparisons:


``` python
validation = (
    pb.Validate(data=tbl)
    .col_avg_le(columns="a", value=3, tol=1.0)
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfYXZnX2xlPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9hdmdfbGUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9ImF2ZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjAuMDU4MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yLjk0LDcuMTQgQzIuMDM0NjY2NjcsNy4xNCAxLjMxODMzMzMzLDYuOTM0NjY2NjcgMC43OTEsNi41MjQgQzAuMjYzNjY2NjY3LDYuMTEzMzMzMzMgMCw1LjU1OCAwLDQuODU4IEMwLDQuMTU4IDAuMjc3NjY2NjY3LDMuNjAyNjY2NjcgMC44MzMsMy4xOTIgQzEuMzg4MzMzMzMsMi43ODEzMzMzMyAyLjE0NjY2NjY3LDIuNTc2IDMuMTA4LDIuNTc2IEMzLjYxMiwyLjU3NiA0LjA2MjMzMzMzLDIuNjQxMzMzMzMgNC40NTksMi43NzIgQzQuODU1NjY2NjcsMi45MDI2NjY2NyA1LjE3NTMzMzMzLDMuMDggNS40MTgsMy4zMDQgTDUuNDE4LDIuNjA0IEM1LjQxOCwyLjEgNS4yNTcsMS43MTUgNC45MzUsMS40NDkgQzQuNjEzLDEuMTgzIDQuMTM5MzMzMzMsMS4wNSAzLjUxNCwxLjA1IEMyLjY4MzMzMzMzLDEuMDUgMS45NjQ2NjY2NywxLjI0MTMzMzMzIDEuMzU4LDEuNjI0IEMxLjE4MDY2NjY3LDEuNzI2NjY2NjcgMS4wMjY2NjY2NywxLjc3OCAwLjg5NiwxLjc3OCBDMC43NzQ2NjY2NjcsMS43NzggMC42NTMzMzMzMzMsMS43MjY2NjY2NyAwLjUzMiwxLjYyNCBDMC40MjkzMzMzMzMsMS41MjEzMzMzMyAwLjM3OCwxLjQgMC4zNzgsMS4yNiBDMC4zNzgsMS4xMDEzMzMzMyAwLjQ1NzMzMzMzMywwLjk1NjY2NjY2NyAwLjYxNiwwLjgyNiBDMC45NTIsMC41NzQgMS4zNzY2NjY2NywwLjM3MzMzMzMzMyAxLjg5LDAuMjI0IEMyLjQwMzMzMzMzLDAuMDc0NjY2NjY2NyAyLjk0LDAgMy41LDAgQzQuNTE3MzMzMzMsMCA1LjMxMDY2NjY3LDAuMjMxIDUuODgsMC42OTMgQzYuNDQ5MzMzMzMsMS4xNTUgNi43MzQsMS44MDEzMzMzMyA2LjczNCwyLjYzMiBMNi43MzQsNi41MSBDNi43MzQsNi43MTUzMzMzMyA2LjY4MDMzMzMzLDYuODcxNjY2NjcgNi41NzMsNi45NzkgQzYuNDY1NjY2NjcsNy4wODYzMzMzMyA2LjMwOTMzMzMzLDcuMTQgNi4xMDQsNy4xNCBDNS44OTg2NjY2Nyw3LjE0IDUuNzQsNy4wODYzMzMzMyA1LjYyOCw2Ljk3OSBDNS41MTYsNi44NzE2NjY2NyA1LjQ2LDYuNzI0NjY2NjcgNS40Niw2LjUzOCBMNS40Niw2LjE0NiBDNC45LDYuODA4NjY2NjcgNC4wNiw3LjE0IDIuOTQsNy4xNCBaIE0zLjE2NCw2LjE0NiBDMy44NDUzMzMzMyw2LjE0NiA0LjM5MzY2NjY3LDYuMDIyMzMzMzMgNC44MDksNS43NzUgQzUuMjI0MzMzMzMsNS41Mjc2NjY2NyA1LjQzMiw1LjIwMzMzMzMzIDUuNDMyLDQuODAyIEM1LjQzMiw0LjQxIDUuMjMzNjY2NjcsNC4xMDQzMzMzMyA0LjgzNywzLjg4NSBDNC40NDAzMzMzMywzLjY2NTY2NjY3IDMuODkyLDMuNTU2IDMuMTkyLDMuNTU2IEMyLjU4NTMzMzMzLDMuNTU2IDIuMTE4NjY2NjcsMy42NjggMS43OTIsMy44OTIgQzEuNDY1MzMzMzMsNC4xMTYgMS4zMDIsNC40MzggMS4zMDIsNC44NTggQzEuMzAyLDUuMjY4NjY2NjcgMS40NjMsNS41ODYgMS43ODUsNS44MSBDMi4xMDcsNi4wMzQgMi41NjY2NjY2Nyw2LjE0NiAzLjE2NCw2LjE0NiBaIiBpZD0iYSIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS40OTQsNy4xNCBDMTAuNTMyNjY2Nyw3LjE0IDkuODUxMzMzMzMsNi41NjEzMzMzMyA5LjQ1LDUuNDA0IEw3Ljc5OCwwLjc4NCBDNy43NywwLjcwOTMzMzMzMyA3Ljc1NiwwLjY0NCA3Ljc1NiwwLjU4OCBDNy43NTYsMC40NjY2NjY2NjcgNy43OTgsMC4zNTcgNy44ODIsMC4yNTkgQzcuOTY2LDAuMTYxIDguMDc4LDAuMDg4NjY2NjY2NyA4LjIxOCwwLjA0MiBDOC4zMDIsMC4wMTQgOC4zODYsMCA4LjQ3LDAgQzguNjEsMCA4LjczMTMzMzMzLDAuMDM1IDguODM0LDAuMTA1IEM4LjkzNjY2NjY3LDAuMTc1IDkuMDA2NjY2NjcsMC4yNzA2NjY2NjcgOS4wNDQsMC4zOTIgTDEwLjczOCw1LjEzOCBDMTAuODU5MzMzMyw1LjQ4MzMzMzMzIDEwLjk3Niw1LjcyNiAxMS4wODgsNS44NjYgQzExLjIsNi4wMDYgMTEuMzM1MzMzMyw2LjA3NiAxMS40OTQsNi4wNzYgQzExLjY1MjY2NjcsNi4wNzYgMTEuNzg4LDYuMDA4MzMzMzMgMTEuOSw1Ljg3MyBDMTIuMDEyLDUuNzM3NjY2NjcgMTIuMTI4NjY2Nyw1LjUwMiAxMi4yNSw1LjE2NiBMMTMuOTQ0LDAuMzkyIEMxMy45OTA2NjY3LDAuMjcwNjY2NjY3IDE0LjA2MywwLjE3NzMzMzMzMyAxNC4xNjEsMC4xMTIgQzE0LjI1OSwwLjA0NjY2NjY2NjcgMTQuMzc4LDAuMDE0IDE0LjUxOCwwLjAxNCBDMTQuNzMyNjY2NywwLjAxNCAxNC45MDUzMzMzLDAuMDcyMzMzMzMzMyAxNS4wMzYsMC4xODkgQzE1LjE2NjY2NjcsMC4zMDU2NjY2NjcgMTUuMjMyLDAuNDQzMzMzMzMzIDE1LjIzMiwwLjYwMiBDMTUuMjMyLDAuNjg2IDE1LjIyMjY2NjcsMC43NDY2NjY2NjcgMTUuMjA0LDAuNzg0IEwxMy41MzgsNS40MzIgQzEzLjMyMzMzMzMsNi4wMTA2NjY2NyAxMy4wNTAzMzMzLDYuNDQgMTIuNzE5LDYuNzIgQzEyLjM4NzY2NjcsNyAxMS45NzkzMzMzLDcuMTQgMTEuNDk0LDcuMTQgWiIgaWQ9InYiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTkuODEsMTAuNTE0IEMxOS4xMTkzMzMzLDEwLjUxNCAxOC40OTE2NjY3LDEwLjQxMTMzMzMgMTcuOTI3LDEwLjIwNiBDMTcuMzYyMzMzMywxMC4wMDA2NjY3IDE2LjkwNzMzMzMsOS43MTEzMzMzMyAxNi41NjIsOS4zMzggQzE2LjQ2ODY2NjcsOS4yMzUzMzMzMyAxNi40MjIsOS4xMjMzMzMzMyAxNi40MjIsOS4wMDIgQzE2LjQyMiw4Ljg0MzMzMzMzIDE2LjUxMDY2NjcsOC42ODkzMzMzMyAxNi42ODgsOC41NCBDMTYuNzcyLDguNDY1MzMzMzMgMTYuODc5MzMzMyw4LjQyOCAxNy4wMSw4LjQyOCBDMTcuMjA2LDguNDI4IDE3LjM5MjY2NjcsOC41MjEzMzMzMyAxNy41Nyw4LjcwOCBDMTcuNzk0LDguOTQxMzMzMzMgMTguMDg1NjY2Nyw5LjEyNTY2NjY3IDE4LjQ0NSw5LjI2MSBDMTguODA0MzMzMyw5LjM5NjMzMzMzIDE5LjIyNjY2NjcsOS40NjQgMTkuNzEyLDkuNDY0IEMyMC41NjEzMzMzLDkuNDY0IDIxLjIxNDY2NjcsOS4yNDQ2NjY2NyAyMS42NzIsOC44MDYgQzIyLjEyOTMzMzMsOC4zNjczMzMzMyAyMi4zNTMzMzMzLDcuNzQyIDIyLjM0NCw2LjkzIEwyMi4zNDQsNS44NjYgQzIyLjAyNjY2NjcsNi4yNjczMzMzMyAyMS42MjUzMzMzLDYuNTg3IDIxLjE0LDYuODI1IEMyMC42NTQ2NjY3LDcuMDYzIDIwLjA4NTMzMzMsNy4xODIgMTkuNDMyLDcuMTgyIEMxOC4yNTYsNy4xODIgMTcuMzQ2LDYuODY3IDE2LjcwMiw2LjIzNyBDMTYuMDU4LDUuNjA3IDE1LjczNiw0Ljc1MDY2NjY3IDE1LjczNiwzLjY2OCBDMTUuNzM2LDIuNTU3MzMzMzMgMTYuMDYyNjY2NywxLjY2ODMzMzMzIDE2LjcxNiwxLjAwMSBDMTcuMzY5MzMzMywwLjMzMzY2NjY2NyAxOC4zMDI2NjY3LDAgMTkuNTE2LDAgQzIwLjA5NDY2NjcsMCAyMC42MjIsMC4xMTY2NjY2NjcgMjEuMDk4LDAuMzUgQzIxLjU3NCwwLjU4MzMzMzMzMyAyMS45ODkzMzMzLDAuOTE0NjY2NjY3IDIyLjM0NCwxLjM0NCBMMjIuMzQ0LDAuNjMgQzIyLjM0NCwwLjQ0MzMzMzMzMyAyMi40MDcsMC4yOTE2NjY2NjcgMjIuNTMzLDAuMTc1IEMyMi42NTksMC4wNTgzMzMzMzMzIDIyLjgxNTMzMzMsMCAyMy4wMDIsMCBDMjMuMTk4LDAgMjMuMzU0MzMzMywwLjA1NiAyMy40NzEsMC4xNjggQzIzLjU4NzY2NjcsMC4yOCAyMy42NDYsMC40MzQgMjMuNjQ2LDAuNjMgTDIzLjY0Niw2LjkzIEMyMy42NDYsNy42NDg2NjY2NyAyMy40ODczMzMzLDguMjc4NjY2NjcgMjMuMTcsOC44MiBDMjIuODUyNjY2Nyw5LjM2MTMzMzMzIDIyLjQwNDY2NjcsOS43NzkgMjEuODI2LDEwLjA3MyBDMjEuMjQ3MzMzMywxMC4zNjcgMjAuNTc1MzMzMywxMC41MTQgMTkuODEsMTAuNTE0IFogTTE5LjYxNCw2LjA5IEMyMC4xMDg2NjY3LDYuMDkgMjAuNTYzNjY2Nyw1Ljk5NjY2NjY3IDIwLjk3OSw1LjgxIEMyMS4zOTQzMzMzLDUuNjIzMzMzMzMgMjEuNzI1NjY2Nyw1LjM0MzMzMzMzIDIxLjk3Myw0Ljk3IEMyMi4yMjAzMzMzLDQuNTk2NjY2NjcgMjIuMzQ0LDQuMTM5MzMzMzMgMjIuMzQ0LDMuNTk4IEMyMi4zNDQsMi44NTEzMzMzMyAyMi4xMDYsMi4yNDQ2NjY2NyAyMS42MywxLjc3OCBDMjEuMTU0LDEuMzExMzMzMzMgMjAuNDk2LDEuMDc4IDE5LjY1NiwxLjA3OCBDMTguODI1MzMzMywxLjA3OCAxOC4xODEzMzMzLDEuMzA0MzMzMzMgMTcuNzI0LDEuNzU3IEMxNy4yNjY2NjY3LDIuMjA5NjY2NjcgMTcuMDM4LDIuODQ2NjY2NjcgMTcuMDM4LDMuNjY4IEMxNy4wMzgsNC40MTQ2NjY2NyAxNy4yNTk2NjY3LDUuMDA1IDE3LjcwMyw1LjQzOSBDMTguMTQ2MzMzMyw1Ljg3MyAxOC43ODMzMzMzLDYuMDkgMTkuNjE0LDYuMDkgWiIgaWQ9ImciIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPHBhdGggZD0iTTQ2LjY4MTgxODIsMTAgTDE3LjMxODE4MTgsMTAgQzE2LjA0MTM3MDYsMTAgMTUsMTEuMDQxMzcwNSAxNSwxMi4zMTgxODE4IEwxNSw0MS42ODE4MTgyIEMxNSw0Mi45NTg2Mjk1IDE2LjA0MTM3MDYsNDQgMTcuMzE4MTgxOCw0NCBMNDYuNjgxODE4Miw0NCBDNDcuOTU4NjI5NSw0NCA0OSw0Mi45NTg2Mjk1IDQ5LDQxLjY4MTgxODIgTDQ5LDEyLjMxODE4MTggQzQ5LDExLjA0MTM3MDUgNDcuOTU4NjI5NSwxMCA0Ni42ODE4MTgyLDEwIFogTTM4LjU4MDI1NTcsMTYuMjc4NDA5MSBMMzkuMzI4ODM1MiwxNy42MzA2ODE4IEwyNi42MzYxODYzLDI0LjY4MTgxODIgTDM5LjMyODgzNTIsMzEuNzMyOTU0NSBMMzguNTgwMjU1NywzMy4wODUyMjczIEwyMy40NTQ3MjI4LDI0LjY4MTgxODIgTDM4LjU4MDI1NTcsMTYuMjc4NDA5MSBaIE00MC41LDM3LjgxODE4MTggTDIzLjUsMzcuODE4MTgxOCBMMjMuNSwzNi4yNzI3MjczIEw0MC41LDM2LjI3MjcyNzMgTDQwLjUsMzcuODE4MTgxOCBaIiBpZD0ibGVzc190aGFuX2VxdWFsIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_avg_le()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">3<br />
tol=1.0</td>
i41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
