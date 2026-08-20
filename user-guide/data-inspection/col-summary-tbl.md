# Column Summaries

While previewing a table with [preview()](../../reference/preview.md#pointblank.preview) is undoubtedly a good thing to do, sometimes you need more. This is where summarizing a table comes in. When you view a summary of a table, the column-by-column info can quickly increase your understanding of a dataset. Plus, it allows you to quickly catch anomalies in your data (e.g., the maximum value of a column could be far outside the realm of possibility).

Pointblank provides a function to make it extremely easy to view column-level summaries in a single table. That function is called [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) and, just like [preview()](../../reference/preview.md#pointblank.preview) does, it supports the use of any table that Pointblank can use for validation. And no matter what the input data is, the resultant reporting table is consistent in its design and construction.


# Trying out [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl)

The function only requires a table. Let's use the `small_table` dataset (a very simple table) to start us off:


``` python
import pointblank as pb

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

pb.col_summary_tbl(small_table)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#yetjrwbxqr table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#yetjrwbxqr thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#yetjrwbxqr p { margin: 0; padding: 0; }
 #yetjrwbxqr .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #yetjrwbxqr .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #yetjrwbxqr .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #yetjrwbxqr .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #yetjrwbxqr .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #yetjrwbxqr .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yetjrwbxqr .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #yetjrwbxqr .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #yetjrwbxqr .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #yetjrwbxqr .gt_column_spanner_outer:first-child { padding-left: 0; }
 #yetjrwbxqr .gt_column_spanner_outer:last-child { padding-right: 0; }
 #yetjrwbxqr .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #yetjrwbxqr .gt_spanner_row { border-bottom-style: hidden; }
 #yetjrwbxqr .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #yetjrwbxqr .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #yetjrwbxqr .gt_from_md> :first-child { margin-top: 0; }
 #yetjrwbxqr .gt_from_md> :last-child { margin-bottom: 0; }
 #yetjrwbxqr .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #yetjrwbxqr .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #yetjrwbxqr .gt_indent_1 { text-indent: 5px; }
 #yetjrwbxqr .gt_indent_2 { text-indent: calc(5px * 2); }
 #yetjrwbxqr .gt_indent_3 { text-indent: calc(5px * 3); }
 #yetjrwbxqr .gt_indent_4 { text-indent: calc(5px * 4); }
 #yetjrwbxqr .gt_indent_5 { text-indent: calc(5px * 5); }
 #yetjrwbxqr .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #yetjrwbxqr .gt_row_group_first td { border-top-width: 2px; }
 #yetjrwbxqr .gt_row_group_first th { border-top-width: 2px; }
 #yetjrwbxqr .gt_striped { color: #333333; background-color: #F4F4F4; }
 #yetjrwbxqr .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yetjrwbxqr .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #yetjrwbxqr .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #yetjrwbxqr .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yetjrwbxqr .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #yetjrwbxqr .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #yetjrwbxqr .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #yetjrwbxqr .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yetjrwbxqr .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #yetjrwbxqr .gt_left { text-align: left; }
 #yetjrwbxqr .gt_center { text-align: center; }
 #yetjrwbxqr .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #yetjrwbxqr .gt_font_normal { font-weight: normal; }
 #yetjrwbxqr .gt_font_bold { font-weight: bold; }
 #yetjrwbxqr .gt_font_italic { font-style: italic; }
 #yetjrwbxqr .gt_super { font-size: 65%; }
 #yetjrwbxqr .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yetjrwbxqr .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #yetjrwbxqr .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yetjrwbxqr .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #yetjrwbxqr .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #yetjrwbxqr .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal"><div>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">13</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">8</span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="icon" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="colname" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Column</th>
<th id="n_missing" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">NA</th>
<th id="n_unique" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">UQ</th>
<th id="mean" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">Mean</th>
<th id="std" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">SD</th>
<th id="min" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">Min</th>
<th id="p05" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">P<span style="font-size: 0.75em; vertical-align: sub; position: relative; line-height: 0.5em;">5</span></th>
<th id="q_1" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">Q<span style="font-size: 0.75em; vertical-align: sub; position: relative; line-height: 0.5em;">1</span></th>
<th id="median" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">Med</th>
<th id="q_3" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">Q<span style="font-size: 0.75em; vertical-align: sub; position: relative; line-height: 0.5em;">3</span></th>
<th id="p95" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">P<span style="font-size: 0.75em; vertical-align: sub; position: relative; line-height: 0.5em;">95</span></th>
<th id="max" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">Max</th>
<th id="iqr" class="gt_col_heading gt_columns_bottom_border gt_center" style="text-align: right;" scope="col">IQR</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPmRhdGU8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJkYXRlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjMDA3RDAwIiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiMyQ0NGMDAiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9IkQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0wLDAgTDM0LjcxNiwwIEM0MC43ODgsMCA0Ni4zMzIsMC45NjggNTEuMzQ4LDIuOTA0IEM1Ni4zNjQsNC44NCA2MC42MzIsNy43MjIgNjQuMTUyLDExLjU1IEM2Ny42NzIsMTUuMzc4IDcwLjQsMjAuMTc0IDcyLjMzNiwyNS45MzggQzc0LjI3MiwzMS43MDIgNzUuMjQsMzguNDEyIDc1LjI0LDQ2LjA2OCBDNzUuMjQsNTMuNzI0IDc0LjI3Miw2MC40MzQgNzIuMzM2LDY2LjE5OCBDNzAuNCw3MS45NjIgNjcuNjcyLDc2Ljc1OCA2NC4xNTIsODAuNTg2IEM2MC42MzIsODQuNDE0IDU2LjM2NCw4Ny4yOTYgNTEuMzQ4LDg5LjIzMiBDNDYuMzMyLDkxLjE2OCA0MC43ODgsOTIuMTM2IDM0LjcxNiw5Mi4xMzYgTDAsOTIuMTM2IEwwLDAgWiBNMzQuNzE2LDc0LjMxNiBDNDAuNyw3NC4zMTYgNDUuNDA4LDcyLjY0NCA0OC44NCw2OS4zIEM1Mi4yNzIsNjUuOTU2IDUzLjk4OCw2MC41ODggNTMuOTg4LDUzLjE5NiBMNTMuOTg4LDM4Ljk0IEM1My45ODgsMzEuNTQ4IDUyLjI3MiwyNi4xOCA0OC44NCwyMi44MzYgQzQ1LjQwOCwxOS40OTIgNDAuNywxNy44MiAzNC43MTYsMTcuODIgTDIwLjA2NCwxNy44MiBMMjAuMDY0LDc0LjMxNiBMMzQuNzE2LDc0LjMxNiBaIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
date_time

Datetime(time_unit='us', time_zone=None)
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">12<br />
0.92</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2016<br />
01<br />
04 00:32:00</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2016<br />
01<br />
30 11:23:00</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPmRhdGU8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJkYXRlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjMDA3RDAwIiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiMyQ0NGMDAiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9IkQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0wLDAgTDM0LjcxNiwwIEM0MC43ODgsMCA0Ni4zMzIsMC45NjggNTEuMzQ4LDIuOTA0IEM1Ni4zNjQsNC44NCA2MC42MzIsNy43MjIgNjQuMTUyLDExLjU1IEM2Ny42NzIsMTUuMzc4IDcwLjQsMjAuMTc0IDcyLjMzNiwyNS45MzggQzc0LjI3MiwzMS43MDIgNzUuMjQsMzguNDEyIDc1LjI0LDQ2LjA2OCBDNzUuMjQsNTMuNzI0IDc0LjI3Miw2MC40MzQgNzIuMzM2LDY2LjE5OCBDNzAuNCw3MS45NjIgNjcuNjcyLDc2Ljc1OCA2NC4xNTIsODAuNTg2IEM2MC42MzIsODQuNDE0IDU2LjM2NCw4Ny4yOTYgNTEuMzQ4LDg5LjIzMiBDNDYuMzMyLDkxLjE2OCA0MC43ODgsOTIuMTM2IDM0LjcxNiw5Mi4xMzYgTDAsOTIuMTM2IEwwLDAgWiBNMzQuNzE2LDc0LjMxNiBDNDAuNyw3NC4zMTYgNDUuNDA4LDcyLjY0NCA0OC44NCw2OS4zIEM1Mi4yNzIsNjUuOTU2IDUzLjk4OCw2MC41ODggNTMuOTg4LDUzLjE5NiBMNTMuOTg4LDM4Ljk0IEM1My45ODgsMzEuNTQ4IDUyLjI3MiwyNi4xOCA0OC44NCwyMi44MzYgQzQ1LjQwOCwxOS40OTIgNDAuNywxNy44MiAzNC43MTYsMTcuODIgTDIwLjA2NCwxNy44MiBMMjAuMDY0LDc0LjMxNiBMMzQuNzE2LDc0LjMxNiBaIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
date

Date
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">11<br />
0.85</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2016<br />
01<br />
04</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2016<br />
01<br />
30</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
a

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">7<br />
0.54</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3.77</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2.09</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1.06</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">7.4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">8</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPnN0cmluZzwvdGl0bGU+CiAgICA8ZyBpZD0iaWNvbiIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InN0cmluZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNC4wMDAwMDAsIDQuNTAwMDAwKSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJzcXVhcmUiIHN0cm9rZT0iIzlBODcwMCIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSIjQ0ZCNjAwIiB4PSIyIiB5PSIyIiB3aWR0aD0iMTY4IiBoZWlnaHQ9IjE2OCIgcng9IjgiIC8+CiAgICAgICAgICAgIDxnIGlkPSJTIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0OS41MTYwMDAsIDM4LjI4MDAwMCkiIGZpbGw9IiNGRkZGRkYiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMzQuOTgsOTUuMzA0IEMyNi44ODQsOTUuMzA0IDIwLjAyLDkzLjkxOCAxNC4zODgsOTEuMTQ2IEM4Ljc1Niw4OC4zNzQgMy45Niw4NC43ODggMCw4MC4zODggTDEzLjIsNjcuMDU2IEMxOS40NDgsNzQuMDk2IDI3LjE0OCw3Ny42MTYgMzYuMyw3Ny42MTYgQzQxLjIyOCw3Ny42MTYgNDQuODgsNzYuNjA0IDQ3LjI1Niw3NC41OCBDNDkuNjMyLDcyLjU1NiA1MC44Miw2OS44NzIgNTAuODIsNjYuNTI4IEM1MC44Miw2My45NzYgNTAuMTE2LDYxLjg0MiA0OC43MDgsNjAuMTI2IEM0Ny4zLDU4LjQxIDQ0LjQ0LDU3LjI0NCA0MC4xMjgsNTYuNjI4IEwzMS4wMiw1NS40NCBDMjEuMjUyLDU0LjIwOCAxNC4xMDIsNTEuMjYgOS41Nyw0Ni41OTYgQzUuMDM4LDQxLjkzMiAyLjc3MiwzNS43MjggMi43NzIsMjcuOTg0IEMyLjc3MiwyMy44NDggMy41NjQsMjAuMDY0IDUuMTQ4LDE2LjYzMiBDNi43MzIsMTMuMiA4Ljk5OCwxMC4yNTIgMTEuOTQ2LDcuNzg4IEMxNC44OTQsNS4zMjQgMTguNTAyLDMuNDEgMjIuNzcsMi4wNDYgQzI3LjAzOCwwLjY4MiAzMS45LDAgMzcuMzU2LDAgQzQ0LjMwOCwwIDUwLjQyNCwxLjEgNTUuNzA0LDMuMyBDNjAuOTg0LDUuNSA2NS41MTYsOC43MTIgNjkuMywxMi45MzYgTDU1Ljk2OCwyNi40IEM1My43NjgsMjMuODQ4IDUxLjEwNiwyMS43NTggNDcuOTgyLDIwLjEzIEM0NC44NTgsMTguNTAyIDQwLjkyLDE3LjY4OCAzNi4xNjgsMTcuNjg4IEMzMS42OCwxNy42ODggMjguMzM2LDE4LjQ4IDI2LjEzNiwyMC4wNjQgQzIzLjkzNiwyMS42NDggMjIuODM2LDIzLjg0OCAyMi44MzYsMjYuNjY0IEMyMi44MzYsMjkuODMyIDIzLjY5NCwzMi4xMiAyNS40MSwzMy41MjggQzI3LjEyNiwzNC45MzYgMjkuOTIsMzUuOTQ4IDMzLjc5MiwzNi41NjQgTDQyLjksMzguMDE2IEM1Mi40MDQsMzkuNTEyIDU5LjQ0NCw0Mi40MzggNjQuMDIsNDYuNzk0IEM2OC41OTYsNTEuMTUgNzAuODg0LDU3LjMzMiA3MC44ODQsNjUuMzQgQzcwLjg4NCw2OS43NCA3MC4wOTIsNzMuNzg4IDY4LjUwOCw3Ny40ODQgQzY2LjkyNCw4MS4xOCA2NC42MTQsODQuMzQ4IDYxLjU3OCw4Ni45ODggQzU4LjU0Miw4OS42MjggNTQuNzgsOTEuNjc0IDUwLjI5Miw5My4xMjYgQzQ1LjgwNCw5NC41NzggNDAuNyw5NS4zMDQgMzQuOTgsOTUuMzA0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
b

String
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">12<br />
0.92</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
c

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2<br />
0.15</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">7<br />
0.54</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">5.73</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2.72</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2.05</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">7</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">8</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">5</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
d

Float64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">12<br />
0.92</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,304.7</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,631.36</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">108.34</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">118.88</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">837.93</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,035.64</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3,291.03</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">6,335.44</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">9999.99</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,453.1</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPmJvb2xlYW48L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJib29sZWFuIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjOUMzRTAwIiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNDRjYxMDAiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9IlQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMyLjMyMjAwMCwgMjkuOTcyMDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjkuMzI2IDExLjYxIDI5LjMyNiA2MC4wMjggMTYuMjU0IDYwLjAyOCAxNi4yNTQgMTEuNjEgMCAxMS42MSAwIDAgNDUuNTggMCA0NS41OCAxMS42MSI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxnIGlkPSJGIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg5OS4xMjIwMDAsIDgxLjk3MjAwMCkiIGZpbGw9IiNGRkZGRkYiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBwb2ludHM9IjAgNjAuMDI4IDAgMCAzOS44MTggMCAzOS44MTggMTEuNjEgMTMuMDcyIDExLjYxIDEzLjA3MiAyMy45MDggMzUuODYyIDIzLjkwOCAzNS44NjIgMzUuNDMyIDEzLjA3MiAzNS40MzIgMTMuMDcyIDYwLjAyOCI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxyZWN0IGlkPSJsaW5lIiBzdHJva2U9IiNGRkZGRkYiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDgyLjAyMjQzNywgODQuMzc3OTQwKSByb3RhdGUoLTMxNS4wMDAwMDApIHRyYW5zbGF0ZSgtODIuMDIyNDM3LCAtODQuMzc3OTQwKSAiIHg9Ijc4LjAyMjQzNjkiIHk9IjI1LjM3Nzk0IiB3aWR0aD0iOCIgaGVpZ2h0PSIxMTgiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
e

Boolean
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;"><span style="font-weight: bold;">T</span>0.62<br />
<span style="font-weight: bold;">F</span>0.38</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">-</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPnN0cmluZzwvdGl0bGU+CiAgICA8ZyBpZD0iaWNvbiIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InN0cmluZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNC4wMDAwMDAsIDQuNTAwMDAwKSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJzcXVhcmUiIHN0cm9rZT0iIzlBODcwMCIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSIjQ0ZCNjAwIiB4PSIyIiB5PSIyIiB3aWR0aD0iMTY4IiBoZWlnaHQ9IjE2OCIgcng9IjgiIC8+CiAgICAgICAgICAgIDxnIGlkPSJTIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0OS41MTYwMDAsIDM4LjI4MDAwMCkiIGZpbGw9IiNGRkZGRkYiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMzQuOTgsOTUuMzA0IEMyNi44ODQsOTUuMzA0IDIwLjAyLDkzLjkxOCAxNC4zODgsOTEuMTQ2IEM4Ljc1Niw4OC4zNzQgMy45Niw4NC43ODggMCw4MC4zODggTDEzLjIsNjcuMDU2IEMxOS40NDgsNzQuMDk2IDI3LjE0OCw3Ny42MTYgMzYuMyw3Ny42MTYgQzQxLjIyOCw3Ny42MTYgNDQuODgsNzYuNjA0IDQ3LjI1Niw3NC41OCBDNDkuNjMyLDcyLjU1NiA1MC44Miw2OS44NzIgNTAuODIsNjYuNTI4IEM1MC44Miw2My45NzYgNTAuMTE2LDYxLjg0MiA0OC43MDgsNjAuMTI2IEM0Ny4zLDU4LjQxIDQ0LjQ0LDU3LjI0NCA0MC4xMjgsNTYuNjI4IEwzMS4wMiw1NS40NCBDMjEuMjUyLDU0LjIwOCAxNC4xMDIsNTEuMjYgOS41Nyw0Ni41OTYgQzUuMDM4LDQxLjkzMiAyLjc3MiwzNS43MjggMi43NzIsMjcuOTg0IEMyLjc3MiwyMy44NDggMy41NjQsMjAuMDY0IDUuMTQ4LDE2LjYzMiBDNi43MzIsMTMuMiA4Ljk5OCwxMC4yNTIgMTEuOTQ2LDcuNzg4IEMxNC44OTQsNS4zMjQgMTguNTAyLDMuNDEgMjIuNzcsMi4wNDYgQzI3LjAzOCwwLjY4MiAzMS45LDAgMzcuMzU2LDAgQzQ0LjMwOCwwIDUwLjQyNCwxLjEgNTUuNzA0LDMuMyBDNjAuOTg0LDUuNSA2NS41MTYsOC43MTIgNjkuMywxMi45MzYgTDU1Ljk2OCwyNi40IEM1My43NjgsMjMuODQ4IDUxLjEwNiwyMS43NTggNDcuOTgyLDIwLjEzIEM0NC44NTgsMTguNTAyIDQwLjkyLDE3LjY4OCAzNi4xNjgsMTcuNjg4IEMzMS42OCwxNy42ODggMjguMzM2LDE4LjQ4IDI2LjEzNiwyMC4wNjQgQzIzLjkzNiwyMS42NDggMjIuODM2LDIzLjg0OCAyMi44MzYsMjYuNjY0IEMyMi44MzYsMjkuODMyIDIzLjY5NCwzMi4xMiAyNS40MSwzMy41MjggQzI3LjEyNiwzNC45MzYgMjkuOTIsMzUuOTQ4IDMzLjc5MiwzNi41NjQgTDQyLjksMzguMDE2IEM1Mi40MDQsMzkuNTEyIDU5LjQ0NCw0Mi40MzggNjQuMDIsNDYuNzk0IEM2OC41OTYsNTEuMTUgNzAuODg0LDU3LjMzMiA3MC44ODQsNjUuMzQgQzcwLjg4NCw2OS43NCA3MC4wOTIsNzMuNzg4IDY4LjUwOCw3Ny40ODQgQzY2LjkyNCw4MS4xOCA2NC42MTQsODQuMzQ4IDYxLjU3OCw4Ni45ODggQzU4LjU0Miw4OS42MjggNTQuNzgsOTEuNjc0IDUwLjI5Miw5My4xMjYgQzQ1LjgwNCw5NC41NzggNDAuNyw5NS4zMDQgMzQuOTgsOTUuMzA0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
f

String
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">3<br />
0.23</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3.46</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0.52</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1</td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote">String columns statistics regard the string's length.</td>
</tr>
</tfoot>

</table>


The header provides the type of table we're looking at (`POLARS`, since this is a Polars DataFrame) and the table dimensions. The rest of the table focuses on the column-level summaries. As such, each row represents a summary of a column in the `small_table` dataset. There's a lot of information in this summary table to digest. Some of it is intuitive since this sort of table summarization isn't all that uncommon, but other aspects of it could also give some pause. So we'll carefully wade through how to interpret this report.


# Data Categories in the Column Summary Table

On the left side of the table are icons of different colors. These represent categories that the columns fall into. There are only five categories and columns can only be of one type. The categories (and their letter marks) are:

- `N`: numeric
- `S`: string-based
- `D`: date/datetime
- `T/F`: boolean
- `O`: object

The numeric category (`N`) takes data types such as floats and integers. The `S` category is for string-based columns. Date or datetime values are lumped into the `D` category. Boolean columns (`T/F`) have their own category and are *not* considered numeric (e.g., `0`/`1`). The `O` category is a catchall for all other types of columns. Given the disparity of these categories and that we want them in the same table, some statistical measures will be sensible for certain column categories but not for others. Given that, we'll explain how each category is represented in the column summary table.


# Numeric Data

Three columns in `small_table` are numeric: `a` (`Int64`), `c` (`Int64`), and `d` (`Float64`). The common measures of the missing count/proportion (`NA`) and the unique value count/proportion (`UQ`) are provided for the numeric data type. For these two measures, the top number is the absolute count of missing values and the count of unique values. The bottom number is a proportion of the absolute count divided by the row count; this makes each proportion a value between `0` and `1` (bounds included).

The next two columns represent the mean (`Mean`) and the standard deviation (`SD`). The minumum (`Min`), maximum, (`Max`) and a set of quantiles occupy the next few columns (includes `P5`, `Q1`, `Med` for median, `Q3`, and `P95`). Finally, the interquartile range (`IQR`: `Q3` - `Q1`) is the last measure provided.


# String Data

String data is present in `small_table`, being in columns `b` and `f`. The missing value (`NA`) and uniqueness (`UQ`) measures are accounted for here. The statistical measures are all based on string lengths, so what happens is that all strings in a column are converted to those numeric values and a subset of stats values is presented. To avoid some understandable confusion when reading the table, the stats values in each of the cells with values are annotated with the text `"SL"`. It makes less sense to provide a full suite of quantile values so only the minimum (`Min`), median (`Med`), and maximum (`Max`) are provided.


# Date/Datetime Data and Boolean Data

We see that in the first two rows of our summary table there are summaries of the `date_time` and `date` columns. The summaries we provide for a date/datetime category (notice the green `D` to the left of the column names) are:

1.  the missing count/proportion (`NA`)
2.  the unique value count/proportion (`UQ`)
3.  the minimum and maximum dates/datetimes

One column, `e`, is of the `Boolean` type. Because columns of this type could only have `True`, `False`, or missing values, we provide summary data for missingness (under `NA`) and proportions of `True` and `False` values (under `UQ`).


# Using [DataScan](../../reference/DataScan.md#pointblank.DataScan) Directly

The [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function is a convenience wrapper around the <a href="../../reference/DataScan.html#pointblank.DataScan" class="gdls-link"><code>DataScan</code></a> class. When you need more than a visual report (for example, to save the profile for later comparison), you can work with [DataScan](../../reference/DataScan.md#pointblank.DataScan) directly:


``` python
small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

scan = pb.DataScan(data=small_table, tbl_name="small_table")
```


The [DataScan](../../reference/DataScan.md#pointblank.DataScan) object computes the same column-level statistics shown in the summary table. You can access the profile as a dictionary with [to_dict()](../../reference/Step.md#pointblank.Step.to_dict), export it as JSON with [to_json()](../../reference/DataScan.to_json.md#pointblank.DataScan.to_json), or render the same tabular report with [get_tabular_report()](../../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report).


# Saving and Loading Profiles

A [DataScan](../../reference/DataScan.md#pointblank.DataScan) profile can be saved to disk and loaded back later. This is useful for establishing baselines that you compare against in the future.


``` python
# Save the profile to a JSON file
scan.save_to_json("small_table_profile.json")

# Later, load it back without needing the original data
loaded = pb.DataScan.load_from_json("small_table_profile.json")
```


The [from_json()](../../reference/DataScan.from_json.md#pointblank.DataScan.from_json) classmethod works with JSON strings directly, and [from_dict()](../../reference/Step.md#pointblank.Step.from_dict) accepts the dictionary format produced by [to_dict()](../../reference/Step.md#pointblank.Step.to_dict). All three approaches produce a fully restored [DataScan](../../reference/DataScan.md#pointblank.DataScan) that retains the column names, types, statistics, and sample data from the original scan.


# Comparing Profiles for Drift

When your data changes over time, you can compare two [DataScan](../../reference/DataScan.md#pointblank.DataScan) profiles to detect drift. The [compare()](../../reference/DataScan.compare.md#pointblank.DataScan.compare) method identifies schema changes (columns added, removed, or with changed types) and statistical shifts in common columns.


``` python
import polars as pl

# Simulate two versions of a dataset
orders_v1 = pl.DataFrame({
    "order_id": [1, 2, 3, 4, 5],
    "amount": [10.0, 25.0, 15.0, 30.0, 20.0],
    "status": ["paid", "paid", "refund", "paid", "paid"],
})

orders_v2 = pl.DataFrame({
    "order_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "amount": [10.0, 25.0, 15.0, 30.0, 20.0, 150.0, 200.0, 175.0],
    "status": ["paid", "paid", "refund", "paid", "paid", "paid", "paid", "paid"],
    "region": ["US", "EU", "US", "EU", "US", "US", "EU", "EU"],
})

baseline = pb.DataScan(data=orders_v1, tbl_name="orders_v1")
current = pb.DataScan(data=orders_v2, tbl_name="orders_v2")

diff = current.compare(baseline)
```


The returned <a href="../../reference/DataScanDiff.html#pointblank.DataScanDiff" class="gdls-link"><code>DataScanDiff</code></a> object provides programmatic access to the changes:


``` python
print("Has changes:", diff.has_changes)
print("Columns added:", diff.columns_added)
print("Row count (baseline vs current):", diff.row_count_diff)
```


    Has changes: True
    Columns added: ['region']
    Row count (baseline vs current): (5, 8)


You can also get the full comparison as a dictionary with [to_dict()](../../reference/Step.md#pointblank.Step.to_dict), or view it as a styled report:


``` python
diff.get_tabular_report()
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#zqlquuhbqi table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#zqlquuhbqi thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#zqlquuhbqi p { margin: 0; padding: 0; }
 #zqlquuhbqi .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #zqlquuhbqi .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #zqlquuhbqi .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #zqlquuhbqi .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #zqlquuhbqi .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #zqlquuhbqi .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zqlquuhbqi .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #zqlquuhbqi .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #zqlquuhbqi .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #zqlquuhbqi .gt_column_spanner_outer:first-child { padding-left: 0; }
 #zqlquuhbqi .gt_column_spanner_outer:last-child { padding-right: 0; }
 #zqlquuhbqi .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #zqlquuhbqi .gt_spanner_row { border-bottom-style: hidden; }
 #zqlquuhbqi .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #zqlquuhbqi .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #zqlquuhbqi .gt_from_md> :first-child { margin-top: 0; }
 #zqlquuhbqi .gt_from_md> :last-child { margin-bottom: 0; }
 #zqlquuhbqi .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #zqlquuhbqi .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #zqlquuhbqi .gt_indent_1 { text-indent: 5px; }
 #zqlquuhbqi .gt_indent_2 { text-indent: calc(5px * 2); }
 #zqlquuhbqi .gt_indent_3 { text-indent: calc(5px * 3); }
 #zqlquuhbqi .gt_indent_4 { text-indent: calc(5px * 4); }
 #zqlquuhbqi .gt_indent_5 { text-indent: calc(5px * 5); }
 #zqlquuhbqi .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #zqlquuhbqi .gt_row_group_first td { border-top-width: 2px; }
 #zqlquuhbqi .gt_row_group_first th { border-top-width: 2px; }
 #zqlquuhbqi .gt_striped { color: #333333; background-color: #F4F4F4; }
 #zqlquuhbqi .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zqlquuhbqi .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #zqlquuhbqi .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #zqlquuhbqi .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zqlquuhbqi .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #zqlquuhbqi .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #zqlquuhbqi .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #zqlquuhbqi .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zqlquuhbqi .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #zqlquuhbqi .gt_left { text-align: left; }
 #zqlquuhbqi .gt_center { text-align: center; }
 #zqlquuhbqi .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #zqlquuhbqi .gt_font_normal { font-weight: normal; }
 #zqlquuhbqi .gt_font_bold { font-weight: bold; }
 #zqlquuhbqi .gt_font_italic { font-style: italic; }
 #zqlquuhbqi .gt_super { font-size: 65%; }
 #zqlquuhbqi .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zqlquuhbqi .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #zqlquuhbqi .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zqlquuhbqi .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #zqlquuhbqi .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #zqlquuhbqi .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="width:100%;" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr class="gt_heading">
<th colspan="6" class="gt_heading gt_title gt_font_normal">Profile Comparison: orders_v1 vs orders_v2</th>
</tr>
<tr class="gt_heading">
<th colspan="6" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Row count: 5 (baseline) vs 8 (current)</th>
</tr>
<tr class="gt_col_headings">
<th id="column" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Column</th>
<th id="status" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Status</th>
<th id="type_baseline" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Type (Baseline)</th>
<th id="type_current" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Type (Current)</th>
<th id="stat_changes" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Changed Statistics</th>
<th id="drift" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Drift Scores</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">order_id</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Stats Changed</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Int64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Int64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">n_unique: 5 -> 8<br />
mean: 3 -> 4.5<br />
median: 3 -> 4.5<br />
std: 1.581 -> 2.449<br />
max: 5 -> 8<br />
p05: 1.02 -> 1.035<br />
q_1: 2 -> 2.75<br />
q_3: 4 -> 6.25<br />
p95: 4.8 -> 7.65<br />
iqr: 2 -> 3.5</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">KS: 0.3750 (p=0.6963)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">amount</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Stats Changed</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Float64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Float64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">n_unique: 5 -> 8<br />
mean: 20 -> 78.12<br />
median: 20 -> 27.5<br />
std: 7.906 -> 81.54<br />
max: 30 -> 200<br />
p05: 10.1 -> 10.18<br />
q_1: 15 -> 18.75<br />
q_3: 25 -> 156.2<br />
p95: 29 -> 191.2<br />
iqr: 10 -> 137.5</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">KS: 0.3750 (p=0.6963)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">status</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Stats Changed</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">String</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">String</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">mean: 4.4 -> 4.25<br />
std: 0.8944 -> 0.7071<br />
p95: 5.6 -> 5.3</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">PSI: 0.0420</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">region</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Added</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px"></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">String</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px"></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px"></td>
</tr>
</tbody>
</table>


The report shows each column's status (OK, Added, Removed, Stats Changed, or Type Changed) along with any statistics that shifted between the baseline and current profiles. This makes it straightforward to spot when your data's shape or distribution has changed in ways that might affect downstream analyses or validation rules.


## Statistical Drift Measures

When both the baseline and current [DataScan](../../reference/DataScan.md#pointblank.DataScan) objects have their original data attached (i.e., they haven't been loaded from disk), the comparison also computes formal drift statistics for each column:

- **PSI (Population Stability Index)**: Measures how much a distribution has shifted. Computed for both numeric and categorical columns. Values below 0.1 indicate no meaningful drift, 0.1 to 0.25 suggests moderate drift, and above 0.25 signals a significant population shift.
- **KS (Kolmogorov-Smirnov) test**: For numeric columns only, this measures the maximum distance between two empirical CDFs and provides a p-value. A small p-value (e.g., below 0.05) indicates the two distributions are statistically different.

These scores appear in the `drift_scores` field of the comparison dictionary and in the "Drift Scores" column of the generated report. Let's look at a more dramatic drift example to see them in action:


``` python
# Baseline: normal revenue range
revenue_v1 = pl.DataFrame({
    "amount": [10.0, 25.0, 15.0, 30.0, 20.0, 12.0, 18.0, 22.0, 28.0, 35.0,
               14.0, 26.0, 19.0, 31.0, 23.0, 11.0, 17.0, 29.0, 33.0, 21.0],
    "category": (["electronics"] * 8 + ["clothing"] * 7 + ["food"] * 5),
})

# Current: large orders have appeared, category mix shifted
revenue_v2 = pl.DataFrame({
    "amount": [10.0, 25.0, 15.0, 30.0, 20.0, 150.0, 200.0, 175.0, 180.0, 160.0,
               14.0, 26.0, 190.0, 170.0, 165.0, 11.0, 17.0, 155.0, 185.0, 195.0],
    "category": (["electronics"] * 12 + ["clothing"] * 3 + ["food"] * 5),
})

baseline = pb.DataScan(data=revenue_v1, tbl_name="revenue_v1")
current = pb.DataScan(data=revenue_v2, tbl_name="revenue_v2")

diff = current.compare(baseline)
diff.get_tabular_report()
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#aonekqiyjo table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#aonekqiyjo thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#aonekqiyjo p { margin: 0; padding: 0; }
 #aonekqiyjo .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #aonekqiyjo .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #aonekqiyjo .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #aonekqiyjo .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #aonekqiyjo .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #aonekqiyjo .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #aonekqiyjo .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #aonekqiyjo .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #aonekqiyjo .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #aonekqiyjo .gt_column_spanner_outer:first-child { padding-left: 0; }
 #aonekqiyjo .gt_column_spanner_outer:last-child { padding-right: 0; }
 #aonekqiyjo .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #aonekqiyjo .gt_spanner_row { border-bottom-style: hidden; }
 #aonekqiyjo .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #aonekqiyjo .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #aonekqiyjo .gt_from_md> :first-child { margin-top: 0; }
 #aonekqiyjo .gt_from_md> :last-child { margin-bottom: 0; }
 #aonekqiyjo .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #aonekqiyjo .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #aonekqiyjo .gt_indent_1 { text-indent: 5px; }
 #aonekqiyjo .gt_indent_2 { text-indent: calc(5px * 2); }
 #aonekqiyjo .gt_indent_3 { text-indent: calc(5px * 3); }
 #aonekqiyjo .gt_indent_4 { text-indent: calc(5px * 4); }
 #aonekqiyjo .gt_indent_5 { text-indent: calc(5px * 5); }
 #aonekqiyjo .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #aonekqiyjo .gt_row_group_first td { border-top-width: 2px; }
 #aonekqiyjo .gt_row_group_first th { border-top-width: 2px; }
 #aonekqiyjo .gt_striped { color: #333333; background-color: #F4F4F4; }
 #aonekqiyjo .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #aonekqiyjo .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #aonekqiyjo .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #aonekqiyjo .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #aonekqiyjo .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #aonekqiyjo .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #aonekqiyjo .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #aonekqiyjo .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #aonekqiyjo .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #aonekqiyjo .gt_left { text-align: left; }
 #aonekqiyjo .gt_center { text-align: center; }
 #aonekqiyjo .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #aonekqiyjo .gt_font_normal { font-weight: normal; }
 #aonekqiyjo .gt_font_bold { font-weight: bold; }
 #aonekqiyjo .gt_font_italic { font-style: italic; }
 #aonekqiyjo .gt_super { font-size: 65%; }
 #aonekqiyjo .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #aonekqiyjo .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #aonekqiyjo .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #aonekqiyjo .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #aonekqiyjo .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #aonekqiyjo .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="width:100%;" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<colgroup>
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
<col style="width: 16%" />
</colgroup>
<thead>
<tr class="gt_heading">
<th colspan="6" class="gt_heading gt_title gt_font_normal">Profile Comparison: revenue_v1 vs revenue_v2</th>
</tr>
<tr class="gt_heading">
<th colspan="6" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Row count: 20 (baseline) vs 20 (current)</th>
</tr>
<tr class="gt_col_headings">
<th id="column" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Column</th>
<th id="status" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Status</th>
<th id="type_baseline" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Type (Baseline)</th>
<th id="type_current" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Type (Current)</th>
<th id="stat_changes" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Changed Statistics</th>
<th id="drift" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Drift Scores</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">amount</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Stats Changed</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Float64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Float64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">mean: 21.95 -> 104.7<br />
median: 21.5 -> 152.5<br />
std: 7.536 -> 80.83<br />
max: 35 -> 200<br />
q_1: 16.5 -> 19.25<br />
q_3: 28.25 -> 176.2<br />
p95: 33.1 -> 195.2<br />
iqr: 11.75 -> 157</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">PSI: 2.9414<br />
KS: 0.5500 (p=0.0026)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">category</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">Stats Changed</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">String</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">String</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">mean: 8.2 -> 8.8<br />
median: 8 -> 11<br />
std: 2.821 -> 3.037</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px">PSI: 0.2506</td>
</tr>
</tbody>
</table>


The drift scores are also available programmatically through [to_dict()](../../reference/Step.md#pointblank.Step.to_dict):


``` python
drift = diff.to_dict().get("drift_scores", {})
for col_name, scores in drift.items():
    print(f"{col_name}: {scores}")
```


    amount: {'psi': 2.941363, 'ks_statistic': 0.55, 'ks_p_value': 0.002571}
    category: {'psi': 0.250553}


When a [DataScan](../../reference/DataScan.md#pointblank.DataScan) is loaded from a saved JSON file (with no original data attached), drift scores cannot be computed. The comparison still reports schema changes and raw stat differences, but the `drift_scores` field will be empty. If you need drift scores in a persist-and-compare workflow, keep the original data available when calling [compare()](../../reference/DataScan.compare.md#pointblank.DataScan.compare).


## A Typical Workflow

In practice, drift detection works best as a two-phase process. First, you establish a baseline profile once your data quality checks pass and save it to disk. Then, on each subsequent pipeline run, you load that baseline and compare it against a fresh scan of the incoming data.

``` python
# On the first run: establish the baseline
baseline = pb.DataScan(data=production_table, tbl_name="orders")
baseline.save_to_json("orders_baseline.json")

# On subsequent runs: compare against the baseline
baseline = pb.DataScan.load_from_json("orders_baseline.json")
current = pb.DataScan(data=production_table, tbl_name="orders")
diff = current.compare(baseline)

if diff.has_changes:
    print("Data drift detected!")
    print(diff.to_dict())
```

The key thing to notice is that the baseline is loaded from a saved JSON file, so it won't have original data attached. This means drift scores (PSI and KS) won't be computed for the loaded baseline. If you need those scores, keep the baseline data accessible and create a fresh [DataScan](../../reference/DataScan.md#pointblank.DataScan) from it rather than loading from JSON.

The [has_changes](../../reference/DataScanDiff.md#pointblank.DataScanDiff.has_changes) property gives you a simple boolean for CI pipelines or alerting. So if anything shifted (schema or statistics) you can easily flag it. For more granular decisions, inspect `columns_added`, `columns_removed`, `columns_type_changed`, and the per-column `stat_diffs` and `drift_scores` in the object provided by [to_dict()](../../reference/Step.md#pointblank.Step.to_dict).


# Conclusion

The [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function and the [DataScan](../../reference/DataScan.md#pointblank.DataScan) class give you two levels of access to column-level profiling. The function is a quick, visual summary you can drop into a notebook to understand a dataset at a glance. The class provides the same statistics in a structured form that you can save, load, and compare over time. When your data evolves, [compare()](../../reference/DataScan.compare.md#pointblank.DataScan.compare) catches schema drift (columns added, removed, or retyped) and statistical drift (shifts in means, quantiles, null rates, and frequency distributions), optionally backed by formal measures like PSI and the KS test. Together, these tools make it straightforward to move from initial data discovery to ongoing data quality monitoring.
