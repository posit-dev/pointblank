## Validate.get_data_extracts()


Get the rows that failed for each validation step.


Usage

``` python
Validate.get_data_extracts(
    i=None,
    frame=False,
)
```


After the <a href="Validate.interrogate.html#pointblank.Validate.interrogate" class="gdls-link"><code>interrogate()</code></a> method has been called, the [get_data_extracts()](Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) method can be used to extract the rows that failed in each column-value or row-based validation step (e.g., <a href="Validate.col_vals_gt.html#pointblank.Validate.col_vals_gt" class="gdls-link"><code>col_vals_gt()</code></a>, <a href="Validate.rows_distinct.html#pointblank.Validate.rows_distinct" class="gdls-link"><code>rows_distinct()</code></a>, etc.). The method returns a dictionary of tables containing the rows that failed in every validation step. If `frame=True` and `i=` is a scalar, the value is conveniently returned as a table (forgoing the dictionary structure).


## Parameters


`i: int | list[int] | None = None`  
The validation step number(s) from which the failed rows are obtained. Can be provided as a list of integers or a single integer. If `None`, all steps are included.

`frame: bool = ``False`  
If `True` and `i=` is a scalar, return the value as a DataFrame instead of a dictionary.


## Returns


`dict[int, Any] | Any`  
A dictionary of tables containing the rows that failed in every compatible validation step. Alternatively, it can be a DataFrame if `frame=True` and `i=` is a scalar.


## Compatible Validation Methods For Yielding Extracted Rows

The following validation methods operate on column values and will have rows extracted when there are failing test units.

- <a href="Validate.col_vals_gt.html#pointblank.Validate.col_vals_gt" class="gdls-link"><code>col_vals_gt()</code></a>
- <a href="Validate.col_vals_ge.html#pointblank.Validate.col_vals_ge" class="gdls-link"><code>col_vals_ge()</code></a>
- <a href="Validate.col_vals_lt.html#pointblank.Validate.col_vals_lt" class="gdls-link"><code>col_vals_lt()</code></a>
- <a href="Validate.col_vals_le.html#pointblank.Validate.col_vals_le" class="gdls-link"><code>col_vals_le()</code></a>
- <a href="Validate.col_vals_eq.html#pointblank.Validate.col_vals_eq" class="gdls-link"><code>col_vals_eq()</code></a>
- <a href="Validate.col_vals_ne.html#pointblank.Validate.col_vals_ne" class="gdls-link"><code>col_vals_ne()</code></a>
- <a href="Validate.col_vals_between.html#pointblank.Validate.col_vals_between" class="gdls-link"><code>col_vals_between()</code></a>
- <a href="Validate.col_vals_outside.html#pointblank.Validate.col_vals_outside" class="gdls-link"><code>col_vals_outside()</code></a>
- <a href="Validate.col_vals_in_set.html#pointblank.Validate.col_vals_in_set" class="gdls-link"><code>col_vals_in_set()</code></a>
- <a href="Validate.col_vals_not_in_set.html#pointblank.Validate.col_vals_not_in_set" class="gdls-link"><code>col_vals_not_in_set()</code></a>
- <a href="Validate.col_vals_increasing.html#pointblank.Validate.col_vals_increasing" class="gdls-link"><code>col_vals_increasing()</code></a>
- <a href="Validate.col_vals_decreasing.html#pointblank.Validate.col_vals_decreasing" class="gdls-link"><code>col_vals_decreasing()</code></a>
- <a href="Validate.col_vals_null.html#pointblank.Validate.col_vals_null" class="gdls-link"><code>col_vals_null()</code></a>
- <a href="Validate.col_vals_not_null.html#pointblank.Validate.col_vals_not_null" class="gdls-link"><code>col_vals_not_null()</code></a>
- <a href="Validate.col_vals_regex.html#pointblank.Validate.col_vals_regex" class="gdls-link"><code>col_vals_regex()</code></a>
- <a href="Validate.col_vals_within_spec.html#pointblank.Validate.col_vals_within_spec" class="gdls-link"><code>col_vals_within_spec()</code></a>
- <a href="Validate.col_vals_expr.html#pointblank.Validate.col_vals_expr" class="gdls-link"><code>col_vals_expr()</code></a>
- <a href="Validate.conjointly.html#pointblank.Validate.conjointly" class="gdls-link"><code>conjointly()</code></a>
- <a href="Validate.prompt.html#pointblank.Validate.prompt" class="gdls-link"><code>prompt()</code></a>

An extracted row for these validation methods means that a test unit failed for that row in the validation step.

These row-based validation methods will also have rows extracted should there be failing rows:

- <a href="Validate.rows_distinct.html#pointblank.Validate.rows_distinct" class="gdls-link"><code>rows_distinct()</code></a>
- <a href="Validate.rows_complete.html#pointblank.Validate.rows_complete" class="gdls-link"><code>rows_complete()</code></a>

The extracted rows are a subset of the original table and are useful for further analysis or for understanding the nature of the failing test units.


## Examples

Let's perform a series of validation steps on a Polars DataFrame. We'll use the <a href="Validate.col_vals_gt.html#pointblank.Validate.col_vals_gt" class="gdls-link"><code>col_vals_gt()</code></a> in the first step, <a href="Validate.col_vals_lt.html#pointblank.Validate.col_vals_lt" class="gdls-link"><code>col_vals_lt()</code></a> in the second step, and <a href="Validate.col_vals_ge.html#pointblank.Validate.col_vals_ge" class="gdls-link"><code>col_vals_ge()</code></a> in the third step. The <a href="Validate.interrogate.html#pointblank.Validate.interrogate" class="gdls-link"><code>interrogate()</code></a> method executes the validation; then, we can extract the rows that failed for each validation step.


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [5, 6, 5, 3, 6, 1],
        "b": [1, 2, 1, 5, 2, 6],
        "c": [3, 7, 2, 6, 3, 1],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns="a", value=4)
    .col_vals_lt(columns="c", value=5)
    .col_vals_ge(columns="b", value=1)
    .interrogate()
)

validation.get_data_extracts()
```


    {1: shape: (2, 4)
     ┌───────────┬─────┬─────┬─────┐
     │ _row_num_ ┆ a   ┆ b   ┆ c   │
     │ ---       ┆ --- ┆ --- ┆ --- │
     │ u32       ┆ i64 ┆ i64 ┆ i64 │
     ╞═══════════╪═════╪═════╪═════╡
     │ 4         ┆ 3   ┆ 5   ┆ 6   │
     │ 6         ┆ 1   ┆ 6   ┆ 1   │
     └───────────┴─────┴─────┴─────┘,
     2: shape: (2, 4)
     ┌───────────┬─────┬─────┬─────┐
     │ _row_num_ ┆ a   ┆ b   ┆ c   │
     │ ---       ┆ --- ┆ --- ┆ --- │
     │ u32       ┆ i64 ┆ i64 ┆ i64 │
     ╞═══════════╪═════╪═════╪═════╡
     │ 2         ┆ 6   ┆ 2   ┆ 7   │
     │ 4         ┆ 3   ┆ 5   ┆ 6   │
     └───────────┴─────┴─────┴─────┘,
     3: shape: (0, 4)
     ┌───────────┬─────┬─────┬─────┐
     │ _row_num_ ┆ a   ┆ b   ┆ c   │
     │ ---       ┆ --- ┆ --- ┆ --- │
     │ u32       ┆ i64 ┆ i64 ┆ i64 │
     ╞═══════════╪═════╪═════╪═════╡
     └───────────┴─────┴─────┴─────┘}


The [get_data_extracts()](Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) method returns a dictionary of tables, where each table contains a subset of rows from the table. These are the rows that failed for each validation step.

In the first step, the<a href="Validate.col_vals_gt.html#pointblank.Validate.col_vals_gt" class="gdls-link"><code>col_vals_gt()</code></a> method was used to check if the values in column `a` were greater than `4`. The extracted table shows the rows where this condition was not met; look at the `a` column: all values are less than `4`.

In the second step, the <a href="Validate.col_vals_lt.html#pointblank.Validate.col_vals_lt" class="gdls-link"><code>col_vals_lt()</code></a> method was used to check if the values in column `c` were less than `5`. In the extracted two-row table, we see that the values in column `c` are greater than `5`.

The third step (<a href="Validate.col_vals_ge.html#pointblank.Validate.col_vals_ge" class="gdls-link"><code>col_vals_ge()</code></a>) checked if the values in column `b` were greater than or equal to `1`. There were no failing test units, so the extracted table is empty (i.e., has columns but no rows).

The `i=` argument can be used to narrow down the extraction to one or more steps. For example, to extract the rows that failed in the first step only:


``` python
validation.get_data_extracts(i=1)
```


    {1: shape: (2, 4)
     ┌───────────┬─────┬─────┬─────┐
     │ _row_num_ ┆ a   ┆ b   ┆ c   │
     │ ---       ┆ --- ┆ --- ┆ --- │
     │ u32       ┆ i64 ┆ i64 ┆ i64 │
     ╞═══════════╪═════╪═════╪═════╡
     │ 4         ┆ 3   ┆ 5   ┆ 6   │
     │ 6         ┆ 1   ┆ 6   ┆ 1   │
     └───────────┴─────┴─────┴─────┘}


Note that the first validation step is indexed at `1` (not `0`). This 1-based indexing is in place here to match the step numbers reported in the validation table. What we get back is still a dictionary, but it only contains one table (the one for the first step).

If you want to get the extracted table as a DataFrame, set `frame=True` and provide a scalar value for `i`. For example, to get the extracted table for the second step as a DataFrame:


``` python
pb.preview(validation.get_data_extracts(i=2, frame=True))
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_preview_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_preview_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_preview_tbl p { margin: 0; padding: 0; }
 #pb_preview_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_preview_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_preview_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_preview_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_preview_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_preview_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_preview_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_preview_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: solid; border-left-width: 1px; border-left-color: #F2F2F2; border-right-style: solid; border-right-width: 1px; border-right-color: #F2F2F2; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_preview_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_preview_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_preview_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_preview_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_preview_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_preview_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_preview_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_preview_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_preview_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_preview_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: solid; border-left-width: 1px; border-left-color: #E9E9E9; border-right-style: solid; border-right-width: 1px; border-right-color: #E9E9E9; vertical-align: middle; overflow-x: hidden; }
 #pb_preview_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_preview_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_preview_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_preview_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_preview_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_preview_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_preview_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_preview_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_preview_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_preview_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_preview_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_preview_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_preview_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_preview_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_preview_tbl .gt_left { text-align: left; }
 #pb_preview_tbl .gt_center { text-align: center; }
 #pb_preview_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_preview_tbl .gt_font_normal { font-weight: normal; }
 #pb_preview_tbl .gt_font_bold { font-weight: bold; }
 #pb_preview_tbl .gt_font_italic { font-style: italic; }
 #pb_preview_tbl .gt_super { font-size: 65%; }
 #pb_preview_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_preview_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_preview_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_preview_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_preview_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_preview_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

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
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6</td>
</tr>
</tbody>
</table>


The extracted table is now a DataFrame, which can serve as a more convenient format for further analysis or visualization. We further used the <a href="preview.html#pointblank.preview" class="gdls-link"><code>preview()</code></a> function to show the DataFrame in an HTML view.
