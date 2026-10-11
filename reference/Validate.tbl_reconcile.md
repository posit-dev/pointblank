# Validate.tbl_reconcile()


Validate that aggregates of the target table reconcile with a comparison table.


Usage

``` python
Validate.tbl_reconcile(
    tbl_compare,
    keys=None,
    measures=None,
    agg="sum",
    tolerance=None,
    column_map=None,
    partitions=None,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```


The [tbl_reconcile()](Validate.tbl_reconcile.md#pointblank.Validate.tbl_reconcile) method checks that the target table agrees with a comparison table (the *source*, or expected data) on aggregate values: totals, counts, averages, minimums, or maximums of *measure* columns, computed per group of *key* columns (or over the whole table). It's the check to reach for when the two tables don't have the same rows but must add up to the same numbers, for example a warehouse fact table against the source system it was loaded from, or a monthly summary against the detailed transactions.

Each key group is a test unit: it passes if the group is in both tables and all of its aggregates are equal (within the tolerance), and fails if an aggregate differs, if the group is missing from the target, or if it's an extra group in the target. To compare the rows themselves rather than aggregates, use <a href="../reference/Validate.tbl_match.html#pointblank.Validate.tbl_match" class="gdls-link"><code>tbl_match()</code></a>.


## Parameters


`tbl_compare: Any`  
The comparison table (the expected data) to reconcile against. This can be a DataFrame object (Polars or Pandas), an Ibis table object, a path to a CSV or Parquet file, or a callable that returns a table.

`keys: str | list[str] | None = None`  
The column(s) to group by. Aggregates are compared group by group. If `None` (the default), the aggregates are computed over the whole table and compared as a single group (a grand total).

`measures: str | list[str] | None = None`  
The column(s) to aggregate. Required unless `agg="count"`, which (without `measures=`) compares the number of rows in each group.

`agg: str | dict[str, str | list[str]] = ``"sum"`  
The aggregation(s): `"sum"` (the default), `"count"` (the number of non-null values), `"mean"`, `"min"`, or `"max"`. Use a dict to aggregate measures differently, mapping a measure to an aggregation or a list of aggregations (e.g., `{"amount": ["sum", "max"], "qty": "sum"}`).

`tolerance: Any = None`  
Tolerance for comparing the aggregates. A number is an absolute tolerance for all aggregates. A dict with `"atol"` and/or `"rtol"` keys applies to all aggregates, where values are equal if `|expected - actual| <= atol + rtol * |expected|`. A dict mapping measures to either of the above sets per-measure tolerances (applied to all of a measure's aggregates other than counts).

`column_map: dict[str, str] | None = None`  
A mapping of comparison-table column names to target column names, for keys or measures that are named differently in the target.

`partitions: int | None = None`  
Compare the aggregated tables in this many passes, each covering a disjoint subset of the key values, to bound memory use when there are very many groups. Requires `keys=`.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table. Have a look at the *Preprocessing* section for more information on how to use this argument.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect. Look at the *Thresholds* section for information on how to set threshold levels.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the <a href="../reference/Actions.html#pointblank.Actions" class="gdls-link"><code>Actions</code></a> class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = ``True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like <a href="../reference/has_columns.html#pointblank.has_columns" class="gdls-link"><code>has_columns()</code></a> and <a href="../reference/has_rows.html#pointblank.has_rows" class="gdls-link"><code>has_rows()</code></a> can be used here to conditionally activate a step based on properties of the target table.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is inferred automatically from the assertion type. This label appears in the validation report and feeds the overall and per-dimension health scores.


## Returns


`Validate`  
The [Validate](Validate.md#pointblank.Validate) object with the added validation step.


## Test Units

Each key group found in either table is a test unit (with `keys=None`, there's a single unit for the whole table). A unit passes if its aggregates are equal (within the tolerance) in both tables. Rows with null keys form a group of their own, which is matched with the null-key group of the other table.

The data extract for the step (from <a href="../reference/Validate.get_data_extracts.html#pointblank.Validate.get_data_extracts" class="gdls-link"><code>get_data_extracts()</code></a>) contains the failing groups: the key column(s), a `_status_` column (`"changed"`, `"missing"`, or `"extra"`), and a pair of `<measure>_<agg>_source`/`<measure>_<agg>_target` columns for each aggregate. The step report shows the full comparison of the aggregates. Since test units are groups rather than rows, this step doesn't take part in <a href="../reference/Validate.get_sundered_data.html#pointblank.Validate.get_sundered_data" class="gdls-link"><code>get_sundered_data()</code></a>.


## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Note that the same preprocessing is **not** applied to the comparison table; only the target table is preprocessed. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](Validate.md#pointblank.Validate) object or used in subsequent validation steps.


## Thresholds

The `thresholds=` parameter is used to set the failure-condition levels for the validation step. If they are set here at the step level, these thresholds will override any thresholds set at the global level in `Validate(thresholds=...)`.

There are three threshold levels: 'warning', 'error', and 'critical'. The threshold values can either be set as a proportion failing of all test units (a value between `0` to `1`), or, the absolute number of failing test units (as integer that's `1` or greater).

Thresholds can be defined using one of these input schemes:

1.  use the <a href="../reference/Thresholds.html#pointblank.Thresholds" class="gdls-link"><code>Thresholds</code></a> class (the most direct way to create thresholds)
2.  provide a tuple of 1-3 values, where position `0` is the 'warning' level, position `1` is the 'error' level, and position `2` is the 'critical' level
3.  create a dictionary of 1-3 value entries; the valid keys: are 'warning', 'error', and 'critical'
4.  a single integer/float value denoting absolute number or fraction of failing test units for the 'warning' level only

If the number of failing test units exceeds set thresholds, the validation step will be marked as 'warning', 'error', or 'critical'. All of the threshold levels don't need to be set, you're free to set any combination of them.

Aside from reporting failure conditions, thresholds can be used to determine the actions to take for each level of failure (using the `actions=` parameter).


## Examples

Let's reconcile a small table of loaded orders against the source system's orders. The loaded table names the amount column differently, has a different number of rows (one order was split into two rows), and is missing the orders for one region.


``` python
import pointblank as pb
import polars as pl

source_orders = pl.DataFrame(
    {
        "region": ["north", "north", "south", "east", "west"],
        "amount": [100.0, 250.0, 75.5, 30.0, 12.0],
    }
)

loaded_orders = pl.DataFrame(
    {
        "region": ["north", "north", "north", "south", "east"],
        "amt": [100.0, 200.0, 50.0, 75.5, 31.0],
    }
)

validation = (
    pb.Validate(data=loaded_orders)
    .tbl_reconcile(
        tbl_compare=source_orders,
        keys="region",
        measures="amount",
        column_map={"amount": "amt"},
    )
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
 #pb_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+dGJsX3JlY29uY2lsZTwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ0YmxfcmVjb25jaWxlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC43NTg2MjEpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yMS4zODgyNDE5LDcuNzc4Njk3ODMgQzIxLjM1ODQyOTgsNy43NzkzNTE3NyAyMS4zMjg3MDQsNy43ODIxNjM2NyAyMS4yOTkyOTk2LDcuNzg3MTE5MTQgTDkuMDkwMTQ4MjQsNy43ODcxMTkxNCBDOC43NTAyOTQzMSw3Ljc4NzE1MzEyIDguNDc0Nzk2NzcsOC4wNjI2NTA2NyA4LjQ3NDc2Mjc5LDguNDAyNTA0NiBMOC40NzQ3NjI3OSwxNi4yOTkxNDk4IEM4LjQ2Mzc3ODc0LDE2LjM2NTYwNjEgOC40NjM3Nzg3NCwxNi40MzM0MTQ5IDguNDc0NzYyNzksMTYuNDk5ODcxMyBMOC40NzQ3NjI3OSwyNC45MTQ1NDYxIEM4LjQ2Mzc3ODc0LDI0Ljk4MTAwMjUgOC40NjM3Nzg3NCwyNS4wNDg4MTEyIDguNDc0NzYyNzksMjUuMTE1MjY3NiBMOC40NzQ3NjI3OSwzMy4wMTc5MjI2IEM4LjQ3NDc5Njc3LDMzLjM1Nzc3NjYgOC43NTAyOTQzMSwzMy42MzMyNzQxIDkuMDkwMTQ4MjQsMzMuNjMzMzA4MSBMMjEuMjk0NDkxNiwzMy42MzMzMDgxIEMyMS4zNjA5NDc5LDMzLjY0NDI5MjEgMjEuNDI4NzU2NywzMy42NDQyOTIxIDIxLjQ5NTIxMzEsMzMuNjMzMzA4MSBMMzMuNzA1NTY2MywzMy42MzMzMDgxIEMzNC4wNDU0MjAyLDMzLjYzMzI3NDEgMzQuMzIwOTE3OCwzMy4zNTc3NzY2IDM0LjMyMDk1MTcsMzMuMDE3OTIyNiBMMzQuMzIwOTUxNywyNS4xMjEyNzc1IEMzNC4zMzE5MzU4LDI1LjA1NDgyMTEgMzQuMzMxOTM1OCwyNC45ODcwMTIzIDM0LjMyMDk1MTcsMjQuOTIwNTU2IEwzNC4zMjA5NTE3LDE2LjUwNTg4MTEgQzM0LjMzMTkzNTgsMTYuNDM5NDI0OCAzNC4zMzE5MzU4LDE2LjM3MTYxNiAzNC4zMjA5NTE3LDE2LjMwNTE1OTYgTDM0LjMyMDk1MTcsOC40MDI1MDQ2IEMzNC4zMjA5MTc4LDguMDYyNjUwNjcgMzQuMDQ1NDIwMiw3Ljc4NzE1MzEyIDMzLjcwNTU2NjMsNy43ODcxMTkxNCBMMjEuNDkyODA5NCw3Ljc4NzExOTE0IEMyMS40NTgyNTY1LDcuNzgxMzQzNjkgMjEuNDIzMjczNiw3Ljc3ODY5NzgzIDIxLjM4ODI0MTksNy43Nzg2OTc4MyBaIE05LjcwNTUzMzY5LDkuMDE3ODkwMDUgTDIwLjc4MjQ3MTgsOS4wMTc4OTAwNSBMMjAuNzgyNDcxOCwxNS43ODcxMyBMOS43MDU1MzM2OSwxNS43ODcxMyBMOS43MDU1MzM2OSw5LjAxNzg5MDA1IFogTTIyLjAxMzI0MjcsOS4wMTc4OTAwNSBMMzMuMDkwMTgwOCw5LjAxNzg5MDA1IEwzMy4wOTAxODA4LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDkuMDE3ODkwMDUgWiBNOS43MDU1MzM2OSwxNy4wMTc5MDA5IEwyMC43ODI0NzE4LDE3LjAxNzkwMDkgTDIwLjc4MjQ3MTgsMjQuNDAyNTI2MyBMOS43MDU1MzM2OSwyNC40MDI1MjYzIEw5LjcwNTUzMzY5LDE3LjAxNzkwMDkgWiBNMjIuMDEzMjQyNywxNy4wMTc5MDA5IEwzMy4wOTAxODA4LDE3LjAxNzkwMDkgTDMzLjA5MDE4MDgsMjQuNDAyNTI2MyBMMjIuMDEzMjQyNywyNC40MDI1MjYzIEwyMi4wMTMyNDI3LDE3LjAxNzkwMDkgWiBNOS43MDU1MzM2OSwyNS42MzMyOTcyIEwyMC43ODI0NzE4LDI1LjYzMzI5NzIgTDIwLjc4MjQ3MTgsMzIuNDAyNTM3MiBMOS43MDU1MzM2OSwzMi40MDI1MzcyIEw5LjcwNTUzMzY5LDI1LjYzMzI5NzIgWiBNMjIuMDEzMjQyNywyNS42MzMyOTcyIEwzMy4wOTAxODA4LDI1LjYzMzI5NzIgTDMzLjA5MDE4MDgsMzIuNDAyNTM3MiBMMjIuMDEzMjQyNywzMi40MDI1MzcyIEwyMi4wMTMyNDI3LDI1LjYzMzI5NzIgWiIgaWQ9InRhYmxlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMuNTIwMCwgNi4wNDAwKSBzY2FsZSgwLjcpIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMjEuMzg4MjQxOSw3Ljc3ODY5NzgzIEMyMS4zNTg0Mjk4LDcuNzc5MzUxNzcgMjEuMzI4NzA0LDcuNzgyMTYzNjcgMjEuMjk5Mjk5Niw3Ljc4NzExOTE0IEw5LjA5MDE0ODI0LDcuNzg3MTE5MTQgQzguNzUwMjk0MzEsNy43ODcxNTMxMiA4LjQ3NDc5Njc3LDguMDYyNjUwNjcgOC40NzQ3NjI3OSw4LjQwMjUwNDYgTDguNDc0NzYyNzksMTYuMjk5MTQ5OCBDOC40NjM3Nzg3NCwxNi4zNjU2MDYxIDguNDYzNzc4NzQsMTYuNDMzNDE0OSA4LjQ3NDc2Mjc5LDE2LjQ5OTg3MTMgTDguNDc0NzYyNzksMjQuOTE0NTQ2MSBDOC40NjM3Nzg3NCwyNC45ODEwMDI1IDguNDYzNzc4NzQsMjUuMDQ4ODExMiA4LjQ3NDc2Mjc5LDI1LjExNTI2NzYgTDguNDc0NzYyNzksMzMuMDE3OTIyNiBDOC40NzQ3OTY3NywzMy4zNTc3NzY2IDguNzUwMjk0MzEsMzMuNjMzMjc0MSA5LjA5MDE0ODI0LDMzLjYzMzMwODEgTDIxLjI5NDQ5MTYsMzMuNjMzMzA4MSBDMjEuMzYwOTQ3OSwzMy42NDQyOTIxIDIxLjQyODc1NjcsMzMuNjQ0MjkyMSAyMS40OTUyMTMxLDMzLjYzMzMwODEgTDMzLjcwNTU2NjMsMzMuNjMzMzA4MSBDMzQuMDQ1NDIwMiwzMy42MzMyNzQxIDM0LjMyMDkxNzgsMzMuMzU3Nzc2NiAzNC4zMjA5NTE3LDMzLjAxNzkyMjYgTDM0LjMyMDk1MTcsMjUuMTIxMjc3NSBDMzQuMzMxOTM1OCwyNS4wNTQ4MjExIDM0LjMzMTkzNTgsMjQuOTg3MDEyMyAzNC4zMjA5NTE3LDI0LjkyMDU1NiBMMzQuMzIwOTUxNywxNi41MDU4ODExIEMzNC4zMzE5MzU4LDE2LjQzOTQyNDggMzQuMzMxOTM1OCwxNi4zNzE2MTYgMzQuMzIwOTUxNywxNi4zMDUxNTk2IEwzNC4zMjA5NTE3LDguNDAyNTA0NiBDMzQuMzIwOTE3OCw4LjA2MjY1MDY3IDM0LjA0NTQyMDIsNy43ODcxNTMxMiAzMy43MDU1NjYzLDcuNzg3MTE5MTQgTDIxLjQ5MjgwOTQsNy43ODcxMTkxNCBDMjEuNDU4MjU2NSw3Ljc4MTM0MzY5IDIxLjQyMzI3MzYsNy43Nzg2OTc4MyAyMS4zODgyNDE5LDcuNzc4Njk3ODMgWiBNOS43MDU1MzM2OSw5LjAxNzg5MDA1IEwyMC43ODI0NzE4LDkuMDE3ODkwMDUgTDIwLjc4MjQ3MTgsMTUuNzg3MTMgTDkuNzA1NTMzNjksMTUuNzg3MTMgTDkuNzA1NTMzNjksOS4wMTc4OTAwNSBaIE0yMi4wMTMyNDI3LDkuMDE3ODkwMDUgTDMzLjA5MDE4MDgsOS4wMTc4OTAwNSBMMzMuMDkwMTgwOCwxNS43ODcxMyBMMjIuMDEzMjQyNywxNS43ODcxMyBMMjIuMDEzMjQyNyw5LjAxNzg5MDA1IFogTTkuNzA1NTMzNjksMTcuMDE3OTAwOSBMMjAuNzgyNDcxOCwxNy4wMTc5MDA5IEwyMC43ODI0NzE4LDI0LjQwMjUyNjMgTDkuNzA1NTMzNjksMjQuNDAyNTI2MyBMOS43MDU1MzM2OSwxNy4wMTc5MDA5IFogTTIyLjAxMzI0MjcsMTcuMDE3OTAwOSBMMzMuMDkwMTgwOCwxNy4wMTc5MDA5IEwzMy4wOTAxODA4LDI0LjQwMjUyNjMgTDIyLjAxMzI0MjcsMjQuNDAyNTI2MyBMMjIuMDEzMjQyNywxNy4wMTc5MDA5IFogTTkuNzA1NTMzNjksMjUuNjMzMjk3MiBMMjAuNzgyNDcxOCwyNS42MzMyOTcyIEwyMC43ODI0NzE4LDMyLjQwMjUzNzIgTDkuNzA1NTMzNjksMzIuNDAyNTM3MiBMOS43MDU1MzM2OSwyNS42MzMyOTcyIFogTTIyLjAxMzI0MjcsMjUuNjMzMjk3MiBMMzMuMDkwMTgwOCwyNS42MzMyOTcyIEwzMy4wOTAxODA4LDMyLjQwMjUzNzIgTDIyLjAxMzI0MjcsMzIuNDAyNTM3MiBMMjIuMDEzMjQyNywyNS42MzMyOTcyIFoiIGlkPSJ0YWJsZSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzMi4wMjAwLCA2LjA0MDApIHNjYWxlKDAuNykiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yMy43NSwzOC4wMCBMMTMuMjUsMzguMDAgTDE5LjEwLDQ0LjUwIEwxMy4yNSw1MS4wMCBMMjMuNzUsNTEuMDAiIGlkPSJzdW0iIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIxLjYiIHN0cm9rZS1saW5lam9pbj0ibWl0ZXIiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiIGZpbGw9Im5vbmUiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Mi4yNSwzOC4wMCBMNDEuNzUsMzguMDAgTDQ3LjYwLDQ0LjUwIEw0MS43NSw1MS4wMCBMNTIuMjUsNTEuMDAiIGlkPSJzdW0iIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIxLjYiIHN0cm9rZS1saW5lam9pbj0ibWl0ZXIiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiIGZpbGw9Im5vbmUiIC8+CiAgICAgICAgICAgIDxnIGlkPSJlcXVhbCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIj48bGluZSB4MT0iMjguNzUiIHkxPSI0My4wMCIgeDI9IjM2Ljc1IiB5Mj0iNDMuMDAiPjwvbGluZT48bGluZSB4MT0iMjguNzUiIHkxPSI0Ni4wMCIgeDI9IjM2Ljc1IiB5Mj0iNDYuMDAiPjwvbGluZT48L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

tbl_reconcile()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">None</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">EXTERNAL TABLE<br />
<span style="font-size: 8px; font-weight: 600; letter-spacing: 0.5px; color: #5A6B7B;">sum · by region</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">4</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.50</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.50</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


Of the four regions, `north` and `south` reconcile; `east` has a different total, and `west` is missing from the loaded orders. The step report shows the details:


``` python
validation.get_step_report(i=1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_comparison_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_comparison_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_comparison_tbl p { margin: 0; padding: 0; }
 #pb_comparison_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: 100%; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_comparison_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_comparison_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_comparison_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_comparison_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_comparison_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_comparison_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_comparison_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_comparison_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_comparison_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_comparison_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_comparison_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_comparison_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_comparison_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_comparison_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_comparison_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_comparison_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_comparison_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_comparison_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_comparison_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_comparison_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_comparison_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_comparison_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_comparison_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_comparison_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_left { text-align: left; }
 #pb_comparison_tbl .gt_center { text-align: center; }
 #pb_comparison_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_comparison_tbl .gt_font_normal { font-weight: normal; }
 #pb_comparison_tbl .gt_font_bold { font-weight: bold; }
 #pb_comparison_tbl .gt_font_italic { font-style: italic; }
 #pb_comparison_tbl .gt_super { font-size: 65%; }
 #pb_comparison_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_comparison_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 100%" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<colgroup>
<col style="width: 30%" />
<col style="width: 70%" />
</colgroup>
<thead>
<tr class="gt_heading">
<th colspan="2" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #CF142B;">✗</span>


ASSERTION <span style="border-style: solid; border-width: thin; border-color: lightblue; padding-left: 2px; padding-right: 2px;"><code style="color: #303030; background-color: transparent; position: relative; bottom: 1px;">tbl_reconcile(keys='region', measures='amount', agg='sum', column_map={'amount': 'amt'})</code></span>

<strong>tbl_compare</strong> (4 groups) → <strong>target</strong> (3 groups), aligned by key <code>region</code><br />
<span style="background:#CF142B; color:white; padding:1px 6px; border-radius:3px; font-size:11px; font-weight:bold; letter-spacing:0.3px;">DIFFERENT</span> 2 of 4 groups fail (50.0%)<br />


<br />
<span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#4CA64C; margin-right:4px; margin-left:4px;"></span>match <strong>2</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#E8A33D; margin-right:4px; margin-left:4px;"></span>changed <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#CF142B; margin-right:4px; margin-left:4px;"></span>missing <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#3D7CC9; margin-right:4px; margin-left:4px;"></span>extra <strong>0</strong></span><br />
<span style="border:1px solid #555; color:#555; padding:0 5px; border-radius:3px; font-size:11px; white-space:nowrap;"><code>sum</code> by <code>region</code></span>


</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="2" class="gt_group_heading" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Column mismatches (1 of 1 columns)</th>
</tr>

<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">amount_sum</td>
<td class="gt_row gt_left" style="font-size: 12px"><div style="display:inline-block; vertical-align:middle; width:120px; height:7px; background:#EEE; border-radius:2px; margin-right:8px; margin-left:8px; overflow:hidden;">


1 of 3 matched groups differ (33.3%)</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" class="gt_group_heading" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Changed groups (1 group)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">east</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount_sum:</span> <span style="text-decoration:line-through; color:#999;">30</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">31</span> <span style="color:#888; font-size:11px;">(+1)</span></span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" class="gt_group_heading" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Missing from target (1 group)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">west</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount_sum:</span> 12</span></td>
</tr>
</tbody>
</table>


Grand totals don't need any keys. Here, the total amount matches within a tolerance of `20`, and so the step passes:


``` python
(
    pb.Validate(data=loaded_orders)
    .tbl_reconcile(
        tbl_compare=source_orders,
        measures="amount",
        column_map={"amount": "amt"},
        tolerance=20,
    )
    .interrogate()
    .all_passed()
)
```


    True
