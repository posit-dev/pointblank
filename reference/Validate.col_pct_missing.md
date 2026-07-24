## Validate.col_pct_missing()


Validate that the percentage of *structured* missing values stays within a limit.


Usage

``` python
Validate.col_pct_missing(
    columns,
    missing,
    max_pct,
    reason=None,
    category=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None
)
```


The [col_pct_missing()](Validate.col_pct_missing.md#pointblank.Validate.col_pct_missing) validation method checks whether the percentage of missing values in a column is at most `max_pct=`. Unlike <a href="Validate.col_pct_null.html#pointblank.Validate.col_pct_null" class="gdls-link"><code>col_pct_null()</code></a>, which only considers actual null values, this method uses a <a href="MissingSpec.html#pointblank.MissingSpec" class="gdls-link"><code>MissingSpec</code></a> to define which values count as missing: declared sentinel values (e.g., `-99` for `"refused"`) and, when `null_is_missing=True`, actual null values. This validation operates at the column level, generating a single validation step per column that passes when the missing percentage does not exceed `max_pct=`.

You can narrow the check to a single reason (via `reason=`) or a category of reasons (via `category=`), making it possible to assert things like "at most 10% of values were refused" or "at most 15% are item nonresponse".


## Parameters


`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a> with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`missing: MissingSpec`  
A <a href="MissingSpec.html#pointblank.MissingSpec" class="gdls-link"><code>MissingSpec</code></a> describing the sentinel values (and their reasons) that encode missingness for this column.

`max_pct: float`  
The maximum allowable percentage of missing values, expressed as a decimal between `0.0` and `1.0`. For example, `max_pct=0.20` means at most 20% of values may be missing.

`reason: str | None = None`  
If provided, only count missing values whose reason matches this label. Cannot be combined with `category=`.

`category: str | None = None`  
If provided, only count missing values whose reason falls in this category (as defined in `MissingSpec.categories`). Cannot be combined with `reason=`.

`thresholds: int | float | None | bool | tuple | dict | Thresholds = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect.

`actions: Actions | None = None`  
Optional actions to take when the validation step(s) meets or exceeds any set threshold levels. If provided, the <a href="Actions.html#pointblank.Actions" class="gdls-link"><code>Actions</code></a> class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = ``True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged).

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is inferred automatically from the assertion type. This label appears in the validation report and feeds the overall and per-dimension health scores.


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

Survey data often encodes missingness with sentinel values rather than nulls. Here, the `age` column uses `-99` (`"not_asked"`), `-98` (`"refused"`), and `-97` (`"dont_know"`):


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {"age": [34, -98, 41, -99, 29, -98, 55, 38]},
)

age_missing = pb.MissingSpec(
    reasons={-99: "not_asked", -98: "refused", -97: "dont_know"},
    categories={"item_nonresponse": ["refused", "dont_know"]},
)

validation = (
    pb.Validate(data=tbl)
    .col_pct_missing(columns="age", missing=age_missing, max_pct=0.5)
    .col_pct_missing(columns="age", missing=age_missing, reason="refused", max_pct=0.30)
    .interrogate()
)

validation
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_tbl p { margin: 0; padding: 0; }
 #pb_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 90%; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_tbl .gt_left { text-align: left; }
 #pb_tbl .gt_center { text-align: center; }
 #pb_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_tbl .gt_font_normal { font-weight: normal; }
 #pb_tbl .gt_font_bold { font-weight: bold; }
 #pb_tbl .gt_font_italic { font-style: italic; }
 #pb_tbl .gt_super { font-size: 65%; }
 #pb_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5wY3RfbWlzc2luZzwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfcGN0X21pc3NpbmciIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMDAwMDAwLCAxLjU4MTcxNykiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InBlcmNlbnQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDExLjI2ODUwOCwgMjMuODU0MzczKSIgZmlsbD0iIzAwMDAwMCI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS44OTkyMDU1MywxNy4yMDM3OTg4IEMxLjc4NzIwNTUzLDE3LjEyMzc5ODggMS42NjQsMTcuMDI5NjI3NCAxLjU4NCwxNi44OTM2Mjc0IEMxLjUwNCwxNi43NTc2Mjc0IDEuNDY0LDE2LjYwOTYyNzQgMS40NjQsMTYuNDQ5NjI3NCBDMS40NjQsMTYuMjQxNjI3NCAxLjUzNiwxNi4wMzM2Mjc0IDEuNjgsMTUuODI1NjI3NCBMMTIuMjQsMC40ODk2Mjc0MzQgQzEyLjQzMiwwLjE4NTYyNzQzNCAxMi41OTQyNjYyLDAgMTIuOTQ2MjY2MiwwIEMxMy4xNTQyNjYyLDAgMTMuNDcyLDAuMDg5NjI3NDM0IDEzLjY4LDAuMjAxNjI3NDM0IEMxNC4wNDgsMC40MjU2Mjc0MzQgMTQuMjMyLDAuNjgxNjI3NDM0IDE0LjIzMiwwLjk2OTYyNzQzNCBDMTQuMjMyLDEuMTQ1NjI3NDMgMTQuMTYsMS4zNDU2Mjc0MyAxNC4wMTYsMS41Njk2Mjc0MyBMMy40MzIsMTYuOTc3NjI3NCBDMy4xNzYsMTcuMjgxNjI3NCAyLjg4LDE3LjQzMzYyNzQgMi41NDQsMTcuNDMzNjI3NCBDMi4zMzYsMTcuNDMzNjI3NCAyLjEzOTIwNTUzLDE3LjM0Nzc5ODggMS44OTkyMDU1MywxNy4yMDM3OTg4IFogTTMuODY0LDcuNDczNjI3NDMgQzMuMTc2LDcuNDczNjI3NDMgMi41MzYsNy4zMDU2Mjc0MyAxLjk0NCw2Ljk2OTYyNzQzIEMxLjM1Miw2LjYzMzYyNzQzIDAuODgsNi4xODE2Mjc0MyAwLjUyOCw1LjYxMzYyNzQzIEMwLjE3Niw1LjA0NTYyNzQzIDAsNC40MjU2Mjc0MyAwLDMuNzUzNjI3NDMgQzAsMy4wODE2Mjc0MyAwLjE3MiwyLjQ2MTYyNzQzIDAuNTE2LDEuODkzNjI3NDMgQzAuODYsMS4zMjU2Mjc0MyAxLjMyOCwwLjg3NzYyNzQzNCAxLjkyLDAuNTQ5NjI3NDM0IEMyLjUxMiwwLjIyMTYyNzQzNCAzLjE2LDAuMDU3NjI3NDM0IDMuODY0LDAuMDU3NjI3NDM0IEM0LjU2OCwwLjA1NzYyNzQzNCA1LjIxNiwwLjIyMTYyNzQzNCA1LjgwOCwwLjU0OTYyNzQzNCBDNi40LDAuODc3NjI3NDM0IDYuODY0LDEuMzI1NjI3NDMgNy4yLDEuODkzNjI3NDMgQzcuNTM2LDIuNDYxNjI3NDMgNy43MDQsMy4wODE2Mjc0MyA3LjcwNCwzLjc1MzYyNzQzIEM3LjcwNCw0LjQyNTYyNzQzIDcuNTMyLDUuMDQ1NjI3NDMgNy4xODgsNS42MTM2Mjc0MyBDNi44NDQsNi4xODE2Mjc0MyA2LjM4LDYuNjMzNjI3NDMgNS43OTYsNi45Njk2Mjc0MyBDNS4yMTIsNy4zMDU2Mjc0MyA0LjU2OCw3LjQ3MzYyNzQzIDMuODY0LDcuNDczNjI3NDMgWiBNMy44NjQsNS42OTc2Mjc0MyBDNC40MDgsNS42OTc2Mjc0MyA0Ljg1Miw1LjUxMzYyNzQzIDUuMTk2LDUuMTQ1NjI3NDMgQzUuNTQsNC43Nzc2Mjc0MyA1LjcxMiw0LjMxMzYyNzQzIDUuNzEyLDMuNzUzNjI3NDMgQzUuNzEyLDMuMTc3NjI3NDMgNS41NCwyLjcwNTYyNzQzIDUuMTk2LDIuMzM3NjI3NDMgQzQuODUyLDEuOTY5NjI3NDMgNC40MDgsMS43ODU2Mjc0MyAzLjg2NCwxLjc4NTYyNzQzIEMzLjMwNCwxLjc4NTYyNzQzIDIuODQ4LDEuOTY5NjI3NDMgMi40OTYsMi4zMzc2Mjc0MyBDMi4xNDQsMi43MDU2Mjc0MyAxLjk2OCwzLjE3NzYyNzQzIDEuOTY4LDMuNzUzNjI3NDMgQzEuOTY4LDQuMzEzNjI3NDMgMi4xNDQsNC43Nzc2Mjc0MyAyLjQ5Niw1LjE0NTYyNzQzIEMyLjg0OCw1LjUxMzYyNzQzIDMuMzA0LDUuNjk3NjI3NDMgMy44NjQsNS42OTc2Mjc0MyBaIE0xMS45NTIsMTcuMzg1NjI3NCBDMTEuMjQ4LDE3LjM4NTYyNzQgMTAuNiwxNy4yMTc2Mjc0IDEwLjAwOCwxNi44ODE2Mjc0IEM5LjQxNiwxNi41NDU2Mjc0IDguOTQ4LDE2LjA5MzYyNzQgOC42MDQsMTUuNTI1NjI3NCBDOC4yNiwxNC45NTc2Mjc0IDguMDg4LDE0LjMzNzYyNzQgOC4wODgsMTMuNjY1NjI3NCBDOC4wODgsMTIuOTkzNjI3NCA4LjI2LDEyLjM3MzYyNzQgOC42MDQsMTEuODA1NjI3NCBDOC45NDgsMTEuMjM3NjI3NCA5LjQxNiwxMC43ODk2Mjc0IDEwLjAwOCwxMC40NjE2Mjc0IEMxMC42LDEwLjEzMzYyNzQgMTEuMjQ4LDkuOTY5NjI3NDMgMTEuOTUyLDkuOTY5NjI3NDMgQzEyLjY1Niw5Ljk2OTYyNzQzIDEzLjMsMTAuMTMzNjI3NCAxMy44ODQsMTAuNDYxNjI3NCBDMTQuNDY4LDEwLjc4OTYyNzQgMTQuOTI4LDExLjIzNzYyNzQgMTUuMjY0LDExLjgwNTYyNzQgQzE1LjYsMTIuMzczNjI3NCAxNS43NjgsMTIuOTkzNjI3NCAxNS43NjgsMTMuNjY1NjI3NCBDMTUuNzY4LDE0LjMzNzYyNzQgMTUuNTk2LDE0Ljk1NzYyNzQgMTUuMjUyLDE1LjUyNTYyNzQgQzE0LjkwOCwxNi4wOTM2Mjc0IDE0LjQ0NCwxNi41NDU2Mjc0IDEzLjg2LDE2Ljg4MTYyNzQgQzEzLjI3NiwxNy4yMTc2Mjc0IDEyLjY0LDE3LjM4NTYyNzQgMTEuOTUyLDE3LjM4NTYyNzQgWiBNMTEuOTUyLDE1LjYwOTYyNzQgQzEyLjQ4LDE1LjYwOTYyNzQgMTIuOTIsMTUuNDI5NjI3NCAxMy4yNzIsMTUuMDY5NjI3NCBDMTMuNjI0LDE0LjcwOTYyNzQgMTMuOCwxNC4yNDE2Mjc0IDEzLjgsMTMuNjY1NjI3NCBDMTMuOCwxMy4xMDU2Mjc0IDEzLjYyNCwxMi42NDE2Mjc0IDEzLjI3MiwxMi4yNzM2Mjc0IEMxMi45MiwxMS45MDU2Mjc0IDEyLjQ4LDExLjcyMTYyNzQgMTEuOTUyLDExLjcyMTYyNzQgQzExLjM5MiwxMS43MjE2Mjc0IDEwLjkzMiwxMS45MDU2Mjc0IDEwLjU3MiwxMi4yNzM2Mjc0IEMxMC4yMTIsMTIuNjQxNjI3NCAxMC4wMzIsMTMuMTA1NjI3NCAxMC4wMzIsMTMuNjY1NjI3NCBDMTAuMDMyLDE0LjI0MTYyNzQgMTAuMjEyLDE0LjcwOTYyNzQgMTAuNTcyLDE1LjA2OTYyNzQgQzEwLjkzMiwxNS40Mjk2Mjc0IDExLjM5MiwxNS42MDk2Mjc0IDExLjk1MiwxNS42MDk2Mjc0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_pct_missing()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">age</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">≤ 0.5</td>
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5wY3RfbWlzc2luZzwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfcGN0X21pc3NpbmciIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMDAwMDAwLCAxLjU4MTcxNykiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InBlcmNlbnQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDExLjI2ODUwOCwgMjMuODU0MzczKSIgZmlsbD0iIzAwMDAwMCI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS44OTkyMDU1MywxNy4yMDM3OTg4IEMxLjc4NzIwNTUzLDE3LjEyMzc5ODggMS42NjQsMTcuMDI5NjI3NCAxLjU4NCwxNi44OTM2Mjc0IEMxLjUwNCwxNi43NTc2Mjc0IDEuNDY0LDE2LjYwOTYyNzQgMS40NjQsMTYuNDQ5NjI3NCBDMS40NjQsMTYuMjQxNjI3NCAxLjUzNiwxNi4wMzM2Mjc0IDEuNjgsMTUuODI1NjI3NCBMMTIuMjQsMC40ODk2Mjc0MzQgQzEyLjQzMiwwLjE4NTYyNzQzNCAxMi41OTQyNjYyLDAgMTIuOTQ2MjY2MiwwIEMxMy4xNTQyNjYyLDAgMTMuNDcyLDAuMDg5NjI3NDM0IDEzLjY4LDAuMjAxNjI3NDM0IEMxNC4wNDgsMC40MjU2Mjc0MzQgMTQuMjMyLDAuNjgxNjI3NDM0IDE0LjIzMiwwLjk2OTYyNzQzNCBDMTQuMjMyLDEuMTQ1NjI3NDMgMTQuMTYsMS4zNDU2Mjc0MyAxNC4wMTYsMS41Njk2Mjc0MyBMMy40MzIsMTYuOTc3NjI3NCBDMy4xNzYsMTcuMjgxNjI3NCAyLjg4LDE3LjQzMzYyNzQgMi41NDQsMTcuNDMzNjI3NCBDMi4zMzYsMTcuNDMzNjI3NCAyLjEzOTIwNTUzLDE3LjM0Nzc5ODggMS44OTkyMDU1MywxNy4yMDM3OTg4IFogTTMuODY0LDcuNDczNjI3NDMgQzMuMTc2LDcuNDczNjI3NDMgMi41MzYsNy4zMDU2Mjc0MyAxLjk0NCw2Ljk2OTYyNzQzIEMxLjM1Miw2LjYzMzYyNzQzIDAuODgsNi4xODE2Mjc0MyAwLjUyOCw1LjYxMzYyNzQzIEMwLjE3Niw1LjA0NTYyNzQzIDAsNC40MjU2Mjc0MyAwLDMuNzUzNjI3NDMgQzAsMy4wODE2Mjc0MyAwLjE3MiwyLjQ2MTYyNzQzIDAuNTE2LDEuODkzNjI3NDMgQzAuODYsMS4zMjU2Mjc0MyAxLjMyOCwwLjg3NzYyNzQzNCAxLjkyLDAuNTQ5NjI3NDM0IEMyLjUxMiwwLjIyMTYyNzQzNCAzLjE2LDAuMDU3NjI3NDM0IDMuODY0LDAuMDU3NjI3NDM0IEM0LjU2OCwwLjA1NzYyNzQzNCA1LjIxNiwwLjIyMTYyNzQzNCA1LjgwOCwwLjU0OTYyNzQzNCBDNi40LDAuODc3NjI3NDM0IDYuODY0LDEuMzI1NjI3NDMgNy4yLDEuODkzNjI3NDMgQzcuNTM2LDIuNDYxNjI3NDMgNy43MDQsMy4wODE2Mjc0MyA3LjcwNCwzLjc1MzYyNzQzIEM3LjcwNCw0LjQyNTYyNzQzIDcuNTMyLDUuMDQ1NjI3NDMgNy4xODgsNS42MTM2Mjc0MyBDNi44NDQsNi4xODE2Mjc0MyA2LjM4LDYuNjMzNjI3NDMgNS43OTYsNi45Njk2Mjc0MyBDNS4yMTIsNy4zMDU2Mjc0MyA0LjU2OCw3LjQ3MzYyNzQzIDMuODY0LDcuNDczNjI3NDMgWiBNMy44NjQsNS42OTc2Mjc0MyBDNC40MDgsNS42OTc2Mjc0MyA0Ljg1Miw1LjUxMzYyNzQzIDUuMTk2LDUuMTQ1NjI3NDMgQzUuNTQsNC43Nzc2Mjc0MyA1LjcxMiw0LjMxMzYyNzQzIDUuNzEyLDMuNzUzNjI3NDMgQzUuNzEyLDMuMTc3NjI3NDMgNS41NCwyLjcwNTYyNzQzIDUuMTk2LDIuMzM3NjI3NDMgQzQuODUyLDEuOTY5NjI3NDMgNC40MDgsMS43ODU2Mjc0MyAzLjg2NCwxLjc4NTYyNzQzIEMzLjMwNCwxLjc4NTYyNzQzIDIuODQ4LDEuOTY5NjI3NDMgMi40OTYsMi4zMzc2Mjc0MyBDMi4xNDQsMi43MDU2Mjc0MyAxLjk2OCwzLjE3NzYyNzQzIDEuOTY4LDMuNzUzNjI3NDMgQzEuOTY4LDQuMzEzNjI3NDMgMi4xNDQsNC43Nzc2Mjc0MyAyLjQ5Niw1LjE0NTYyNzQzIEMyLjg0OCw1LjUxMzYyNzQzIDMuMzA0LDUuNjk3NjI3NDMgMy44NjQsNS42OTc2Mjc0MyBaIE0xMS45NTIsMTcuMzg1NjI3NCBDMTEuMjQ4LDE3LjM4NTYyNzQgMTAuNiwxNy4yMTc2Mjc0IDEwLjAwOCwxNi44ODE2Mjc0IEM5LjQxNiwxNi41NDU2Mjc0IDguOTQ4LDE2LjA5MzYyNzQgOC42MDQsMTUuNTI1NjI3NCBDOC4yNiwxNC45NTc2Mjc0IDguMDg4LDE0LjMzNzYyNzQgOC4wODgsMTMuNjY1NjI3NCBDOC4wODgsMTIuOTkzNjI3NCA4LjI2LDEyLjM3MzYyNzQgOC42MDQsMTEuODA1NjI3NCBDOC45NDgsMTEuMjM3NjI3NCA5LjQxNiwxMC43ODk2Mjc0IDEwLjAwOCwxMC40NjE2Mjc0IEMxMC42LDEwLjEzMzYyNzQgMTEuMjQ4LDkuOTY5NjI3NDMgMTEuOTUyLDkuOTY5NjI3NDMgQzEyLjY1Niw5Ljk2OTYyNzQzIDEzLjMsMTAuMTMzNjI3NCAxMy44ODQsMTAuNDYxNjI3NCBDMTQuNDY4LDEwLjc4OTYyNzQgMTQuOTI4LDExLjIzNzYyNzQgMTUuMjY0LDExLjgwNTYyNzQgQzE1LjYsMTIuMzczNjI3NCAxNS43NjgsMTIuOTkzNjI3NCAxNS43NjgsMTMuNjY1NjI3NCBDMTUuNzY4LDE0LjMzNzYyNzQgMTUuNTk2LDE0Ljk1NzYyNzQgMTUuMjUyLDE1LjUyNTYyNzQgQzE0LjkwOCwxNi4wOTM2Mjc0IDE0LjQ0NCwxNi41NDU2Mjc0IDEzLjg2LDE2Ljg4MTYyNzQgQzEzLjI3NiwxNy4yMTc2Mjc0IDEyLjY0LDE3LjM4NTYyNzQgMTEuOTUyLDE3LjM4NTYyNzQgWiBNMTEuOTUyLDE1LjYwOTYyNzQgQzEyLjQ4LDE1LjYwOTYyNzQgMTIuOTIsMTUuNDI5NjI3NCAxMy4yNzIsMTUuMDY5NjI3NCBDMTMuNjI0LDE0LjcwOTYyNzQgMTMuOCwxNC4yNDE2Mjc0IDEzLjgsMTMuNjY1NjI3NCBDMTMuOCwxMy4xMDU2Mjc0IDEzLjYyNCwxMi42NDE2Mjc0IDEzLjI3MiwxMi4yNzM2Mjc0IEMxMi45MiwxMS45MDU2Mjc0IDEyLjQ4LDExLjcyMTYyNzQgMTEuOTUyLDExLjcyMTYyNzQgQzExLjM5MiwxMS43MjE2Mjc0IDEwLjkzMiwxMS45MDU2Mjc0IDEwLjU3MiwxMi4yNzM2Mjc0IEMxMC4yMTIsMTIuNjQxNjI3NCAxMC4wMzIsMTMuMTA1NjI3NCAxMC4wMzIsMTMuNjY1NjI3NCBDMTAuMDMyLDE0LjI0MTYyNzQgMTAuMjEyLDE0LjcwOTYyNzQgMTAuNTcyLDE1LjA2OTYyNzQgQzEwLjkzMiwxNS40Mjk2Mjc0IDExLjM5MiwxNS42MDk2Mjc0IDExLjk1MiwxNS42MDk2Mjc0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_pct_missing()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">age</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">≤ 0.3</td>
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
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">
<hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(missing_spec)</span> <strong>Missing codes:</strong> <code>-99</code>→not_asked, <code>-98</code>→refused, <code>-97</code>→dont_know, <code>null</code>→unknown</p>
<p>Step 2 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(missing_spec)</span> <strong>Missing codes:</strong> <code>-99</code>→not_asked, <code>-98</code>→refused, <code>-97</code>→dont_know, <code>null</code>→unknown. Counting reason <code>refused</code></p></td>
</tr>
</tfoot>

</table>
