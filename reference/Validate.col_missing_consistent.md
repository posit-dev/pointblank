## Validate.col_missing_consistent()


Validate that related columns share a consistent missingness pattern for a given reason.


Usage

``` python
Validate.col_missing_consistent(
    columns,
    missing,
    when_reason,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None
)
```


The [col_missing_consistent()](Validate.col_missing_consistent.md#pointblank.Validate.col_missing_consistent) method checks that, across a set of related columns, the "missing for a specific reason" status is *consistent*: for each row, either *none* of the columns are missing for `when_reason=`, or *all* of them are. This is useful for structured survey or clinical data where a skip pattern should propagate across related fields -- for example, if a question wasn't asked (`"not_asked"`) then all of its dependent fields should also be coded `"not_asked"`.

A value is considered "missing for the reason" when it is one of the sentinel values mapped to `when_reason=` in the <a href="MissingSpec.html#pointblank.MissingSpec" class="gdls-link"><code>MissingSpec</code></a> (and, when the reason is the spec's [null_reason](MissingSpec.md#pointblank.MissingSpec.null_reason) and `null_is_missing=True`, an actual null). This validation operates over the number of test units equal to the number of rows in the table. A row fails when some -- but not all -- of the columns are missing for the given reason.


## Parameters


`columns: list[str]`  
A list of related columns to check for consistent missingness.

`missing: MissingSpec`  
A <a href="MissingSpec.html#pointblank.MissingSpec" class="gdls-link"><code>MissingSpec</code></a> describing the sentinel values and their reasons for the columns.

`when_reason: str`  
The reason label whose presence should be consistent across `columns=`. If one column in a row is missing for this reason, all of them should be.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment).

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the <a href="Actions.html#pointblank.Actions" class="gdls-link"><code>Actions</code></a> class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = ``True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged).

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is inferred automatically from the assertion type. This label appears in the validation report and feeds the overall and per-dimension health scores.


## Returns


`Validate`  
The [Validate](Validate.md#pointblank.Validate) object with the added validation step.


## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.


## Segmentation

The `segments=` argument allows for the segmentation of a validation step into multiple segments. This is useful for applying the same validation step to different subsets of the data. The segmentation can be done based on a single column or specific fields within a column. Providing a single column name results in a separate validation step for each unique value in that column; a tuple of `(column, values)` restricts segmentation to the listed values. The segmentation is performed after any `pre=` preprocessing.


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

Here, `income_source` and `income_amount` should both be coded `"not_asked"` (`-99`) together when the income question wasn't asked. The last row is inconsistent -- only one field is coded `-99`:


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "income_source": [1, -99, 2, -99],
        "income_amount": [50000, -99, 42000, 38000],
    }
)

income_missing = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"})

validation = (
    pb.Validate(data=tbl)
    .col_missing_consistent(
        columns=["income_source", "income_amount"],
        missing=income_missing,
        when_reason="not_asked",
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX21pc3NpbmdfY29uc2lzdGVudDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfbWlzc2luZ19jb25zaXN0ZW50IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC41NTE3MjQpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00MC42MTIwODA1LDQ3LjAzNzgzNCBDMzcuNDY5MjM0OCw0Ny4wMzc4MzQgMzUuMDEyNjEzOSw0NS45MzQ4NjEzIDMzLjcxMjIzNCw0NC4wMTQwNTk3IEMzMi40MTE4NTQxLDQ1LjkzNDg2MTMgMjkuOTU1MjMzMSw0Ny4wMzc4MzQgMjYuODEyMzg4Myw0Ny4wMzc4MzQgQzIyLjY1NzQzOTcsNDcuMDM3ODM0IDE2LjA2NDY3MTIsNDMuNDQzNzcyMyAxNi4wNjQ2NzEyLDMzLjgwMjE2MTkgQzE2LjA2NDY3MTIsMjkuMzQwMTM2MSAxNy40NzE1ODc5LDE4Ljk2MjE2NiAzMC41MDM1ODYyLDE4Ljk2MjE2NiBDMzAuOTQ1NDAxOCwxOC45NjIxNjYgMzEuMzA1NzQ4MSwxOS4zMjI1MTI0IDMxLjMwNTc0ODEsMTkuNzY0MzI3OSBMMzEuMzA1NzQ4MSwyMS4zNjg2NTE4IEMzMS4zMDU3NDgxLDIxLjgxMDQ2NzQgMzAuOTQ1NDAxOCwyMi4xNzA4MTM4IDMwLjUwMzU4NjIsMjIuMTcwODEzOCBDMjYuNjQwMDQ4NiwyMi4xNzA4MTM4IDIyLjQ4MTk2NjgsMjUuODExODc3NCAyMi40ODE5NjY4LDMzLjgwMjE2MTkgQzIyLjQ4MTk2NjgsMzcuNTA5MDI3NyAyMy43NjM1NDU2LDQzLjAyNzAyNDMgMjcuMjk0OTM4NCw0My4wMjcwMjQzIEMyOS43OTU0MjgsNDMuMDI3MDI0MyAzMS4yMjQyNzksNDAuNDIzMTMxMiAzMi4wOTg1MDk1LDM4LjI4NjEyMjEgQzMwLjUwNjcxOTQsMzUuNjEwMTU5NiAyOS43MDE0MjQzLDMzLjEwMzQwMzUgMjkuNzAxNDI0MywzMC44MzQ3ODkyIEMyOS43MDE0MjQzLDI1LjYyMzg3MDcgMzEuODYwMzY3NywyMy43NzUxMzc3IDMzLjcxMjIzNCwyMy43NzUxMzc3IEMzNS41NjQxMDAyLDIzLjc3NTEzNzcgMzcuNzIzMDQzNywyNS42MjM4NzA3IDM3LjcyMzA0MzcsMzAuODM0Nzg5MiBDMzcuNzIzMDQzNywzMy4xMzQ3MzgzIDM2LjkzOTY4MjgsMzUuNTc4ODI1NSAzNS4zMjkwOTE2LDM4LjI4NjEyMjEgQzM2LjYyOTQ3MTUsNDEuNDMyMTAwOSAzOC4yNDMxOTYsNDMuMDI3MDI0MyA0MC4xMjk1Mjk1LDQzLjAyNzAyNDMgQzQzLjY2MDkyMjMsNDMuMDI3MDI0MyA0NC45NDI1MDEyLDM3LjUwOTAyNzcgNDQuOTQyNTAxMiwzMy44MDIxNjE5IEM0NC45NDI1MDEyLDI1LjgxMTg3NzQgNDAuNzg0NDE5MywyMi4xNzA4MTM4IDM2LjkyMDg4MTcsMjIuMTcwODEzOCBDMzYuNDc1OTMyOSwyMi4xNzA4MTM4IDM2LjExODcxOTgsMjEuODEwNDY3NCAzNi4xMTg3MTk4LDIxLjM2ODY1MTggTDM2LjExODcxOTgsMTkuNzY0MzI3OSBDMzYuMTE4NzE5OCwxOS4zMjI1MTI0IDM2LjQ3NTkzMjksMTguOTYyMTY2IDM2LjkyMDg4MTcsMTguOTYyMTY2IEM0OS45NTI4ODAxLDE4Ljk2MjE2NiA1MS4zNTk3OTY3LDI5LjM0MDEzNjEgNTEuMzU5Nzk2NywzMy44MDIxNjE5IEM1MS4zNTk3OTY3LDQzLjQ0Mzc3MjMgNDQuNzY3MDI4Miw0Ny4wMzc4MzQgNDAuNjEyMDgwNSw0Ny4wMzc4MzQgWiIgaWQ9Im9tZWdhIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_missing_consistent()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">income_source, income_amount</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"><span style="font-weight: 600; letter-spacing: 0.5px;">CONSISTENT</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">4</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">3<br />
0.75</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
0.25</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">
<hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(missing_spec)</span> <strong>Missing codes:</strong> <code>-99</code>→not_asked, <code>-98</code>→refused, <code>null</code>→unknown. Consistency required for reason <code>not_asked</code></p></td>
</tr>
</tfoot>

</table>


The validation reports one failing test unit: the final row, where `income_source` is coded `-99` (`"not_asked"`) but `income_amount` is a real value.
