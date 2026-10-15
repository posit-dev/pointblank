# seg_group()


Group together values for segmentation.


Usage

``` python
seg_group(values)
```


Many validation methods have a `segments=` argument that can be used to specify one or more columns, or certain values within a column, to create segments for validation (e.g., <a href="../reference/Validate.col_vals_gt.html#pointblank.Validate.col_vals_gt" class="gdls-link"><code>col_vals_gt()</code></a>, <a href="../reference/Validate.col_vals_regex.html#pointblank.Validate.col_vals_regex" class="gdls-link"><code>col_vals_regex()</code></a>, etc.). When passing in a column, or a tuple with a column and certain values, a segment will be created for each individual value within the column or given values. The [seg_group()](seg_group.md#pointblank.seg_group) selector enables values to be grouped together into a segment. For example, if you were to create a segment for a column "region", investigating just "North" and "South" regions, a typical segment would look like:

`segments=("region", ["North", "South"])`

This would create two validation steps, one for each of the regions. If you wanted to group these two regions into a single segment, you could use the [seg_group()](seg_group.md#pointblank.seg_group) function like this:

`segments=("region", pb.seg_group(["North", "South"]))`

You could create a second segment for "East" and "West" regions like this:

`segments=("region", pb.seg_group([["North", "South"], ["East", "West"]]))`

There will be a validation step created for every segment. Note that if there aren't any segments created using [seg_group()](seg_group.md#pointblank.seg_group) (or any other segment expression), the validation step will fail to be evaluated during the interrogation process. Such a failure to evaluate will be reported in the validation results but it won't affect the interrogation process overall (i.e., the process won't be halted).


## Parameters


`values: list[Any]`  
A list of values to be grouped into a segment. This can be a single list or a list of lists.


## Returns


`Segment`  
A `Segment` object, which can be used to combine values into a segment.


## Examples

Let's say we're analyzing sales from our local bookstore, and want to check the number of books sold for the month exceeds a certain threshold. We could pass in the argument `segments="genre"`, which would return a segment for each unique genre in the datasets. We could also pass in `segments=("genre", ["Fantasy", "Science Fiction"])`, to only create segments for those two genres. However, if we wanted to group these two genres into a single segment, we could use the [seg_group()](seg_group.md#pointblank.seg_group) function.


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "title": [
            "The Hobbit",
            "Harry Potter and the Sorcerer's Stone",
            "The Lord of the Rings",
            "A Game of Thrones",
            "The Name of the Wind",
            "The Girl with the Dragon Tattoo",
            "The Da Vinci Code",
            "The Hitchhiker's Guide to the Galaxy",
            "The Martian",
            "Brave New World"
        ],
        "genre": [
            "Fantasy",
            "Fantasy",
            "Fantasy",
            "Fantasy",
            "Fantasy",
            "Mystery",
            "Mystery",
            "Science Fiction",
            "Science Fiction",
            "Science Fiction",
        ],
        "units_sold": [875, 932, 756, 623, 445, 389, 678, 534, 712, 598],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(
        columns="units_sold",
        value=500,
        segments=("genre", pb.seg_group(["Fantasy", "Science Fiction"]))
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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>genre / ['Fantasy', 'Science Fiction']</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTAtSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMC1jb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMC1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0wLWdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">units_sold</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">500</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTAtc2VnbWVudGVkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0wLXNlZ21lbnRlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTAtUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0icGJfdGJsLXRibC0wLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjQwMDk0ODE2LDE0LjM0OTg1MjcgQzIuMzY5MTE2MDIsMTQuMzQ5ODUyNyAwLjcxNTk0ODE2MywxNi4wMDI4NTg2IDAuNzE1OTQ4MTYzLDE4LjAzNDg1MjcgQzAuNzE1OTQ4MTYzLDIwLjA2Njg0NjcgMi4zNjkxMTYwMiwyMS43MTk4NTI3IDQuNDAwOTQ4MTYsMjEuNzE5ODUyNyBDNi40MzI3ODAzMSwyMS43MTk4NTI3IDguMDg1OTQ4MTYsMjAuMDY2ODQ2NyA4LjA4NTk0ODE2LDE4LjAzNDg1MjcgQzguMDg1OTQ4MTYsMTYuMDAyODU4NiA2LjQzMjcyNjMzLDE0LjM0OTg1MjcgNC40MDA5NDgxNiwxNC4zNDk4NTI3IFogTTQuNDAwOTQ4MTYsMjAuOTgyODIwMyBDMi43NzU0MTc2NywyMC45ODI4MjAzIDEuNDUyOTgwNTUsMTkuNjYwMzgzMSAxLjQ1Mjk4MDU1LDE4LjAzNDg1MjcgQzEuNDUyOTgwNTUsMTYuNDA5MzIyMiAyLjc3NTQxNzY3LDE1LjA4Njg4NSA0LjQwMDk0ODE2LDE1LjA4Njg4NSBDNi4wMjY0Nzg2NSwxNS4wODY4ODUgNy4zNDg5MTU3OCwxNi40MDkzMjIyIDcuMzQ4OTE1NzgsMTguMDM0ODUyNyBDNy4zNDg5MTU3OCwxOS42NjAzODMxIDYuMDI2NDc4NjUsMjAuOTgyODIwMyA0LjQwMDk0ODE2LDIwLjk4MjgyMDMgWiIgaWQ9InBiX3RibC10YmwtMC1TaGFwZSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDMzMzYwMSwtMC42NDAyODc1ODggTDE4LjQzMzM2MDEsMTQuNTg4Mjk1MiBMOS41MjEzMTEzMSwxNC41ODgyOTUyIEwxMS4zNzE2MTUxLDE1Ljk4MDk5NyBDMTEuNjU3NjE2NSwxNi4xOTQ4NzY1IDExLjcxNDgxNjksMTYuNjAwMjUxOCAxMS41MDA5Mzc0LDE2Ljg4NjI1MzIgQzExLjI4NzA1ODUsMTcuMTcyMjU0NiAxMC44ODE2ODI3LDE3LjIyOTQ1NSAxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDEwLjU5NTY4MTMsMTcuMDE1NTc1NSBMNy4yMzMzMDEyNiwxNC40Njg5MjA4IEM3LjA2NjY3NDI5LDE0LjM0OTU0NjQgNi45NjcxOTU1OSwxNC4xNTU1NjMyIDYuOTY3MTk1NTksMTMuOTUxNjMxNiBDNi45NjcxOTU1OSwxMy43NDc3MDA2IDcuMDY2Njc0MjksMTMuNTUzNzE2OCA3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDcuMjMzMzAxMjYsMTMuNDM0MzQyNCBMMTAuNTk1NjgxMywxMC44ODc2ODc3IEMxMC42Nzc3NTEsMTAuODI1NTEzNyAxMC43NzIyNTYxLDEwLjc4NTcyMjIgMTAuODc0MjIxNiwxMC43NjgzMTMyIEMxMC44OTkwOTE2LDEwLjc2MDg1MjIgMTAuOTI2NDQ3OCwxMC43NTMzOTE4IDEwLjk1MzgwNDYsMTAuNzQ4NDE3NSBDMTEuMjUyMjQwNywxMC43MDg2MjYgMTEuNTM4MjQyMSwxMC44ODI3MTQxIDExLjYzNzcyMDgsMTEuMTY2MjI4IEMxMS43Mzk2ODYzLDExLjQ0OTc0MjYgMTEuNjI3NzcyOSwxMS43NjU1ODcgMTEuMzcxNjE1MSwxMS45MjIyNjYxIEwxMS4zNzE2MTUxLDExLjkyMjI2NjEgTDkuNTIxMzExMzEsMTMuMzE0OTY3OSBMMTcuMTU5MzUzMywxMy4zMTQ3MTI0IEwxNy4xNTk0NTc3LDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsMC42MzM2MTQ4MjggTDYuOTI4MzUzMywtMC42NDAyODc1ODggTDE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IFoiIGlkPSJwYl90YmwtdGJsLTAtYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-eval"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-test_units">8</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-pass">7<br />
0.88</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-fail">1<br />
0.12</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-w_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC" headers="pb_tbl-e_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-c_upd">--</td>
<td class="gt_row gt_center" style="height: 40px" headers="pb_tbl-extract_upd">CSV</td>
</tr>
</tbody>
</table>


What's more, we can create multiple segments, combining the genres in different ways.


``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(
        columns="units_sold",
        value=500,
        segments=("genre", pb.seg_group([
            ["Fantasy", "Science Fiction"],
            ["Fantasy", "Mystery"],
            ["Mystery", "Science Fiction"]
        ]))
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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>genre / ['Fantasy', 'Science Fiction']</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTAtSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMC1jb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMC1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0wLWdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">units_sold</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">500</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTAtc2VnbWVudGVkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0wLXNlZ21lbnRlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTAtUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0icGJfdGJsLXRibC0wLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjQwMDk0ODE2LDE0LjM0OTg1MjcgQzIuMzY5MTE2MDIsMTQuMzQ5ODUyNyAwLjcxNTk0ODE2MywxNi4wMDI4NTg2IDAuNzE1OTQ4MTYzLDE4LjAzNDg1MjcgQzAuNzE1OTQ4MTYzLDIwLjA2Njg0NjcgMi4zNjkxMTYwMiwyMS43MTk4NTI3IDQuNDAwOTQ4MTYsMjEuNzE5ODUyNyBDNi40MzI3ODAzMSwyMS43MTk4NTI3IDguMDg1OTQ4MTYsMjAuMDY2ODQ2NyA4LjA4NTk0ODE2LDE4LjAzNDg1MjcgQzguMDg1OTQ4MTYsMTYuMDAyODU4NiA2LjQzMjcyNjMzLDE0LjM0OTg1MjcgNC40MDA5NDgxNiwxNC4zNDk4NTI3IFogTTQuNDAwOTQ4MTYsMjAuOTgyODIwMyBDMi43NzU0MTc2NywyMC45ODI4MjAzIDEuNDUyOTgwNTUsMTkuNjYwMzgzMSAxLjQ1Mjk4MDU1LDE4LjAzNDg1MjcgQzEuNDUyOTgwNTUsMTYuNDA5MzIyMiAyLjc3NTQxNzY3LDE1LjA4Njg4NSA0LjQwMDk0ODE2LDE1LjA4Njg4NSBDNi4wMjY0Nzg2NSwxNS4wODY4ODUgNy4zNDg5MTU3OCwxNi40MDkzMjIyIDcuMzQ4OTE1NzgsMTguMDM0ODUyNyBDNy4zNDg5MTU3OCwxOS42NjAzODMxIDYuMDI2NDc4NjUsMjAuOTgyODIwMyA0LjQwMDk0ODE2LDIwLjk4MjgyMDMgWiIgaWQ9InBiX3RibC10YmwtMC1TaGFwZSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDMzMzYwMSwtMC42NDAyODc1ODggTDE4LjQzMzM2MDEsMTQuNTg4Mjk1MiBMOS41MjEzMTEzMSwxNC41ODgyOTUyIEwxMS4zNzE2MTUxLDE1Ljk4MDk5NyBDMTEuNjU3NjE2NSwxNi4xOTQ4NzY1IDExLjcxNDgxNjksMTYuNjAwMjUxOCAxMS41MDA5Mzc0LDE2Ljg4NjI1MzIgQzExLjI4NzA1ODUsMTcuMTcyMjU0NiAxMC44ODE2ODI3LDE3LjIyOTQ1NSAxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDEwLjU5NTY4MTMsMTcuMDE1NTc1NSBMNy4yMzMzMDEyNiwxNC40Njg5MjA4IEM3LjA2NjY3NDI5LDE0LjM0OTU0NjQgNi45NjcxOTU1OSwxNC4xNTU1NjMyIDYuOTY3MTk1NTksMTMuOTUxNjMxNiBDNi45NjcxOTU1OSwxMy43NDc3MDA2IDcuMDY2Njc0MjksMTMuNTUzNzE2OCA3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDcuMjMzMzAxMjYsMTMuNDM0MzQyNCBMMTAuNTk1NjgxMywxMC44ODc2ODc3IEMxMC42Nzc3NTEsMTAuODI1NTEzNyAxMC43NzIyNTYxLDEwLjc4NTcyMjIgMTAuODc0MjIxNiwxMC43NjgzMTMyIEMxMC44OTkwOTE2LDEwLjc2MDg1MjIgMTAuOTI2NDQ3OCwxMC43NTMzOTE4IDEwLjk1MzgwNDYsMTAuNzQ4NDE3NSBDMTEuMjUyMjQwNywxMC43MDg2MjYgMTEuNTM4MjQyMSwxMC44ODI3MTQxIDExLjYzNzcyMDgsMTEuMTY2MjI4IEMxMS43Mzk2ODYzLDExLjQ0OTc0MjYgMTEuNjI3NzcyOSwxMS43NjU1ODcgMTEuMzcxNjE1MSwxMS45MjIyNjYxIEwxMS4zNzE2MTUxLDExLjkyMjI2NjEgTDkuNTIxMzExMzEsMTMuMzE0OTY3OSBMMTcuMTU5MzUzMywxMy4zMTQ3MTI0IEwxNy4xNTk0NTc3LDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsMC42MzM2MTQ4MjggTDYuOTI4MzUzMywtMC42NDAyODc1ODggTDE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IFoiIGlkPSJwYl90YmwtdGJsLTAtYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-eval"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-test_units">8</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-pass">7<br />
0.88</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-fail">1<br />
0.12</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-w_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC" headers="pb_tbl-e_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-c_upd">--</td>
<td class="gt_row gt_center" style="height: 40px" headers="pb_tbl-extract_upd">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap" headers="pb_tbl-type_upd"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>genre / ['Fantasy', 'Mystery']</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTEtSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMS1jb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMS1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0xLWdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">units_sold</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">500</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTEtc2VnbWVudGVkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0xLXNlZ21lbnRlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTEtUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0icGJfdGJsLXRibC0xLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjQwMDk0ODE2LDE0LjM0OTg1MjcgQzIuMzY5MTE2MDIsMTQuMzQ5ODUyNyAwLjcxNTk0ODE2MywxNi4wMDI4NTg2IDAuNzE1OTQ4MTYzLDE4LjAzNDg1MjcgQzAuNzE1OTQ4MTYzLDIwLjA2Njg0NjcgMi4zNjkxMTYwMiwyMS43MTk4NTI3IDQuNDAwOTQ4MTYsMjEuNzE5ODUyNyBDNi40MzI3ODAzMSwyMS43MTk4NTI3IDguMDg1OTQ4MTYsMjAuMDY2ODQ2NyA4LjA4NTk0ODE2LDE4LjAzNDg1MjcgQzguMDg1OTQ4MTYsMTYuMDAyODU4NiA2LjQzMjcyNjMzLDE0LjM0OTg1MjcgNC40MDA5NDgxNiwxNC4zNDk4NTI3IFogTTQuNDAwOTQ4MTYsMjAuOTgyODIwMyBDMi43NzU0MTc2NywyMC45ODI4MjAzIDEuNDUyOTgwNTUsMTkuNjYwMzgzMSAxLjQ1Mjk4MDU1LDE4LjAzNDg1MjcgQzEuNDUyOTgwNTUsMTYuNDA5MzIyMiAyLjc3NTQxNzY3LDE1LjA4Njg4NSA0LjQwMDk0ODE2LDE1LjA4Njg4NSBDNi4wMjY0Nzg2NSwxNS4wODY4ODUgNy4zNDg5MTU3OCwxNi40MDkzMjIyIDcuMzQ4OTE1NzgsMTguMDM0ODUyNyBDNy4zNDg5MTU3OCwxOS42NjAzODMxIDYuMDI2NDc4NjUsMjAuOTgyODIwMyA0LjQwMDk0ODE2LDIwLjk4MjgyMDMgWiIgaWQ9InBiX3RibC10YmwtMS1TaGFwZSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDMzMzYwMSwtMC42NDAyODc1ODggTDE4LjQzMzM2MDEsMTQuNTg4Mjk1MiBMOS41MjEzMTEzMSwxNC41ODgyOTUyIEwxMS4zNzE2MTUxLDE1Ljk4MDk5NyBDMTEuNjU3NjE2NSwxNi4xOTQ4NzY1IDExLjcxNDgxNjksMTYuNjAwMjUxOCAxMS41MDA5Mzc0LDE2Ljg4NjI1MzIgQzExLjI4NzA1ODUsMTcuMTcyMjU0NiAxMC44ODE2ODI3LDE3LjIyOTQ1NSAxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDEwLjU5NTY4MTMsMTcuMDE1NTc1NSBMNy4yMzMzMDEyNiwxNC40Njg5MjA4IEM3LjA2NjY3NDI5LDE0LjM0OTU0NjQgNi45NjcxOTU1OSwxNC4xNTU1NjMyIDYuOTY3MTk1NTksMTMuOTUxNjMxNiBDNi45NjcxOTU1OSwxMy43NDc3MDA2IDcuMDY2Njc0MjksMTMuNTUzNzE2OCA3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDcuMjMzMzAxMjYsMTMuNDM0MzQyNCBMMTAuNTk1NjgxMywxMC44ODc2ODc3IEMxMC42Nzc3NTEsMTAuODI1NTEzNyAxMC43NzIyNTYxLDEwLjc4NTcyMjIgMTAuODc0MjIxNiwxMC43NjgzMTMyIEMxMC44OTkwOTE2LDEwLjc2MDg1MjIgMTAuOTI2NDQ3OCwxMC43NTMzOTE4IDEwLjk1MzgwNDYsMTAuNzQ4NDE3NSBDMTEuMjUyMjQwNywxMC43MDg2MjYgMTEuNTM4MjQyMSwxMC44ODI3MTQxIDExLjYzNzcyMDgsMTEuMTY2MjI4IEMxMS43Mzk2ODYzLDExLjQ0OTc0MjYgMTEuNjI3NzcyOSwxMS43NjU1ODcgMTEuMzcxNjE1MSwxMS45MjIyNjYxIEwxMS4zNzE2MTUxLDExLjkyMjI2NjEgTDkuNTIxMzExMzEsMTMuMzE0OTY3OSBMMTcuMTU5MzUzMywxMy4zMTQ3MTI0IEwxNy4xNTk0NTc3LDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsMC42MzM2MTQ4MjggTDYuOTI4MzUzMywtMC42NDAyODc1ODggTDE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IFoiIGlkPSJwYl90YmwtdGJsLTEtYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-eval"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-test_units">7</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-pass">5<br />
0.71</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-fail">2<br />
0.29</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-w_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC" headers="pb_tbl-e_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-c_upd">--</td>
<td class="gt_row gt_center" style="height: 40px" headers="pb_tbl-extract_upd">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap" headers="pb_tbl-type_upd"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>genre / ['Mystery', 'Science Fiction']</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTItSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMi1jb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMi1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0yLWdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">units_sold</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">500</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTItc2VnbWVudGVkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0yLXNlZ21lbnRlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTItUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0icGJfdGJsLXRibC0yLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjQwMDk0ODE2LDE0LjM0OTg1MjcgQzIuMzY5MTE2MDIsMTQuMzQ5ODUyNyAwLjcxNTk0ODE2MywxNi4wMDI4NTg2IDAuNzE1OTQ4MTYzLDE4LjAzNDg1MjcgQzAuNzE1OTQ4MTYzLDIwLjA2Njg0NjcgMi4zNjkxMTYwMiwyMS43MTk4NTI3IDQuNDAwOTQ4MTYsMjEuNzE5ODUyNyBDNi40MzI3ODAzMSwyMS43MTk4NTI3IDguMDg1OTQ4MTYsMjAuMDY2ODQ2NyA4LjA4NTk0ODE2LDE4LjAzNDg1MjcgQzguMDg1OTQ4MTYsMTYuMDAyODU4NiA2LjQzMjcyNjMzLDE0LjM0OTg1MjcgNC40MDA5NDgxNiwxNC4zNDk4NTI3IFogTTQuNDAwOTQ4MTYsMjAuOTgyODIwMyBDMi43NzU0MTc2NywyMC45ODI4MjAzIDEuNDUyOTgwNTUsMTkuNjYwMzgzMSAxLjQ1Mjk4MDU1LDE4LjAzNDg1MjcgQzEuNDUyOTgwNTUsMTYuNDA5MzIyMiAyLjc3NTQxNzY3LDE1LjA4Njg4NSA0LjQwMDk0ODE2LDE1LjA4Njg4NSBDNi4wMjY0Nzg2NSwxNS4wODY4ODUgNy4zNDg5MTU3OCwxNi40MDkzMjIyIDcuMzQ4OTE1NzgsMTguMDM0ODUyNyBDNy4zNDg5MTU3OCwxOS42NjAzODMxIDYuMDI2NDc4NjUsMjAuOTgyODIwMyA0LjQwMDk0ODE2LDIwLjk4MjgyMDMgWiIgaWQ9InBiX3RibC10YmwtMi1TaGFwZSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDMzMzYwMSwtMC42NDAyODc1ODggTDE4LjQzMzM2MDEsMTQuNTg4Mjk1MiBMOS41MjEzMTEzMSwxNC41ODgyOTUyIEwxMS4zNzE2MTUxLDE1Ljk4MDk5NyBDMTEuNjU3NjE2NSwxNi4xOTQ4NzY1IDExLjcxNDgxNjksMTYuNjAwMjUxOCAxMS41MDA5Mzc0LDE2Ljg4NjI1MzIgQzExLjI4NzA1ODUsMTcuMTcyMjU0NiAxMC44ODE2ODI3LDE3LjIyOTQ1NSAxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDEwLjU5NTY4MTMsMTcuMDE1NTc1NSBMNy4yMzMzMDEyNiwxNC40Njg5MjA4IEM3LjA2NjY3NDI5LDE0LjM0OTU0NjQgNi45NjcxOTU1OSwxNC4xNTU1NjMyIDYuOTY3MTk1NTksMTMuOTUxNjMxNiBDNi45NjcxOTU1OSwxMy43NDc3MDA2IDcuMDY2Njc0MjksMTMuNTUzNzE2OCA3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDcuMjMzMzAxMjYsMTMuNDM0MzQyNCBMMTAuNTk1NjgxMywxMC44ODc2ODc3IEMxMC42Nzc3NTEsMTAuODI1NTEzNyAxMC43NzIyNTYxLDEwLjc4NTcyMjIgMTAuODc0MjIxNiwxMC43NjgzMTMyIEMxMC44OTkwOTE2LDEwLjc2MDg1MjIgMTAuOTI2NDQ3OCwxMC43NTMzOTE4IDEwLjk1MzgwNDYsMTAuNzQ4NDE3NSBDMTEuMjUyMjQwNywxMC43MDg2MjYgMTEuNTM4MjQyMSwxMC44ODI3MTQxIDExLjYzNzcyMDgsMTEuMTY2MjI4IEMxMS43Mzk2ODYzLDExLjQ0OTc0MjYgMTEuNjI3NzcyOSwxMS43NjU1ODcgMTEuMzcxNjE1MSwxMS45MjIyNjYxIEwxMS4zNzE2MTUxLDExLjkyMjI2NjEgTDkuNTIxMzExMzEsMTMuMzE0OTY3OSBMMTcuMTU5MzUzMywxMy4zMTQ3MTI0IEwxNy4xNTk0NTc3LDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsMC42MzM2MTQ4MjggTDYuOTI4MzUzMywtMC42NDAyODc1ODggTDE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IFoiIGlkPSJwYl90YmwtdGJsLTItYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-eval"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-test_units">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-pass">4<br />
0.80</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-fail">1<br />
0.20</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-w_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC" headers="pb_tbl-e_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-c_upd">--</td>
<td class="gt_row gt_center" style="height: 40px" headers="pb_tbl-extract_upd">CSV</td>
</tr>
</tbody>
</table>
