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
#zjtmujoiqi table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#zjtmujoiqi thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#zjtmujoiqi p { margin: 0; padding: 0; }
 #zjtmujoiqi .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #zjtmujoiqi .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #zjtmujoiqi .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #zjtmujoiqi .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #zjtmujoiqi .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #zjtmujoiqi .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zjtmujoiqi .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #zjtmujoiqi .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #zjtmujoiqi .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #zjtmujoiqi .gt_column_spanner_outer:first-child { padding-left: 0; }
 #zjtmujoiqi .gt_column_spanner_outer:last-child { padding-right: 0; }
 #zjtmujoiqi .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #zjtmujoiqi .gt_spanner_row { border-bottom-style: hidden; }
 #zjtmujoiqi .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #zjtmujoiqi .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #zjtmujoiqi .gt_from_md> :first-child { margin-top: 0; }
 #zjtmujoiqi .gt_from_md> :last-child { margin-bottom: 0; }
 #zjtmujoiqi .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #zjtmujoiqi .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #zjtmujoiqi .gt_indent_1 { text-indent: 5px; }
 #zjtmujoiqi .gt_indent_2 { text-indent: calc(5px * 2); }
 #zjtmujoiqi .gt_indent_3 { text-indent: calc(5px * 3); }
 #zjtmujoiqi .gt_indent_4 { text-indent: calc(5px * 4); }
 #zjtmujoiqi .gt_indent_5 { text-indent: calc(5px * 5); }
 #zjtmujoiqi .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #zjtmujoiqi .gt_row_group_first td { border-top-width: 2px; }
 #zjtmujoiqi .gt_row_group_first th { border-top-width: 2px; }
 #zjtmujoiqi .gt_striped { color: #333333; background-color: #F4F4F4; }
 #zjtmujoiqi .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zjtmujoiqi .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #zjtmujoiqi .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #zjtmujoiqi .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #zjtmujoiqi .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #zjtmujoiqi .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #zjtmujoiqi .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #zjtmujoiqi .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zjtmujoiqi .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #zjtmujoiqi .gt_left { text-align: left; }
 #zjtmujoiqi .gt_center { text-align: center; }
 #zjtmujoiqi .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #zjtmujoiqi .gt_font_normal { font-weight: normal; }
 #zjtmujoiqi .gt_font_bold { font-weight: bold; }
 #zjtmujoiqi .gt_font_italic { font-style: italic; }
 #zjtmujoiqi .gt_super { font-size: 65%; }
 #zjtmujoiqi .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zjtmujoiqi .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #zjtmujoiqi .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #zjtmujoiqi .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #zjtmujoiqi .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #zjtmujoiqi .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" id="title" class="gt_heading gt_title gt_font_normal"><div>

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
<td class="gt_row gt_left" style="font-family: IBM Plex Mono" headers="icon"><svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"> <title>date</title> <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="date" transform="translate(4.000000, 4.339342)"> <rect id="square" stroke="#007D00" stroke-width="4" fill="#2CCF00" x="2" y="2" width="168" height="168" rx="8"></rect> <g id="D" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"> <path d="M0,0 L34.716,0 C40.788,0 46.332,0.968 51.348,2.904 C56.364,4.84 60.632,7.722 64.152,11.55 C67.672,15.378 70.4,20.174 72.336,25.938 C74.272,31.702 75.24,38.412 75.24,46.068 C75.24,53.724 74.272,60.434 72.336,66.198 C70.4,71.962 67.672,76.758 64.152,80.586 C60.632,84.414 56.364,87.296 51.348,89.232 C46.332,91.168 40.788,92.136 34.716,92.136 L0,92.136 L0,0 Z M34.716,74.316 C40.7,74.316 45.408,72.644 48.84,69.3 C52.272,65.956 53.988,60.588 53.988,53.196 L53.988,38.94 C53.988,31.548 52.272,26.18 48.84,22.836 C45.408,19.492 40.7,17.82 34.716,17.82 L20.064,17.82 L20.064,74.316 L34.716,74.316 Z"></path> </g> </g> </g> </svg></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px" headers="colname"><div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>date_time</div><div style='font-size: 11px; color: gray;'>Datetime(time_unit='us', time_zone=None)</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_missing">0<br>0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_unique">12<br>0.92</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="mean">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="std">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="min">2016<br>01<br>04 00:32:00</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p05">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_1">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="median">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_3">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p95">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="max">2016<br>01<br>30 11:23:00</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="iqr">-</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono" headers="icon"><svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"> <title>date</title> <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="date" transform="translate(4.000000, 4.339342)"> <rect id="square" stroke="#007D00" stroke-width="4" fill="#2CCF00" x="2" y="2" width="168" height="168" rx="8"></rect> <g id="D" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"> <path d="M0,0 L34.716,0 C40.788,0 46.332,0.968 51.348,2.904 C56.364,4.84 60.632,7.722 64.152,11.55 C67.672,15.378 70.4,20.174 72.336,25.938 C74.272,31.702 75.24,38.412 75.24,46.068 C75.24,53.724 74.272,60.434 72.336,66.198 C70.4,71.962 67.672,76.758 64.152,80.586 C60.632,84.414 56.364,87.296 51.348,89.232 C46.332,91.168 40.788,92.136 34.716,92.136 L0,92.136 L0,0 Z M34.716,74.316 C40.7,74.316 45.408,72.644 48.84,69.3 C52.272,65.956 53.988,60.588 53.988,53.196 L53.988,38.94 C53.988,31.548 52.272,26.18 48.84,22.836 C45.408,19.492 40.7,17.82 34.716,17.82 L20.064,17.82 L20.064,74.316 L34.716,74.316 Z"></path> </g> </g> </g> </svg></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px" headers="colname"><div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>date</div><div style='font-size: 11px; color: gray;'>Date</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_missing">0<br>0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_unique">11<br>0.85</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="mean">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="std">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="min">2016<br>01<br>04</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p05">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_1">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="median">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_3">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p95">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="max">2016<br>01<br>30</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="iqr">-</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono" headers="icon"><svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"> <title>numeric</title> <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="numeric" transform="translate(4.000000, 4.339342)"> <rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"></rect> <g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"> <polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"></polygon> </g> </g> </g> </svg></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px" headers="colname"><div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>a</div><div style='font-size: 11px; color: gray;'>Int64</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_missing">0<br>0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_unique">7<br>0.54</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="mean">3.77</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="std">2.09</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="min">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p05">1.06</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_1">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="median">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_3">4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p95">7.4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="max">8</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="iqr">2</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono" headers="icon"><svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"> <title>string</title> <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="string" transform="translate(4.000000, 4.500000)"> <rect id="square" stroke="#9A8700" stroke-width="4" fill="#CFB600" x="2" y="2" width="168" height="168" rx="8"></rect> <g id="S" transform="translate(49.516000, 38.280000)" fill="#FFFFFF" fill-rule="nonzero"> <path d="M34.98,95.304 C26.884,95.304 20.02,93.918 14.388,91.146 C8.756,88.374 3.96,84.788 0,80.388 L13.2,67.056 C19.448,74.096 27.148,77.616 36.3,77.616 C41.228,77.616 44.88,76.604 47.256,74.58 C49.632,72.556 50.82,69.872 50.82,66.528 C50.82,63.976 50.116,61.842 48.708,60.126 C47.3,58.41 44.44,57.244 40.128,56.628 L31.02,55.44 C21.252,54.208 14.102,51.26 9.57,46.596 C5.038,41.932 2.772,35.728 2.772,27.984 C2.772,23.848 3.564,20.064 5.148,16.632 C6.732,13.2 8.998,10.252 11.946,7.788 C14.894,5.324 18.502,3.41 22.77,2.046 C27.038,0.682 31.9,0 37.356,0 C44.308,0 50.424,1.1 55.704,3.3 C60.984,5.5 65.516,8.712 69.3,12.936 L55.968,26.4 C53.768,23.848 51.106,21.758 47.982,20.13 C44.858,18.502 40.92,17.688 36.168,17.688 C31.68,17.688 28.336,18.48 26.136,20.064 C23.936,21.648 22.836,23.848 22.836,26.664 C22.836,29.832 23.694,32.12 25.41,33.528 C27.126,34.936 29.92,35.948 33.792,36.564 L42.9,38.016 C52.404,39.512 59.444,42.438 64.02,46.794 C68.596,51.15 70.884,57.332 70.884,65.34 C70.884,69.74 70.092,73.788 68.508,77.484 C66.924,81.18 64.614,84.348 61.578,86.988 C58.542,89.628 54.78,91.674 50.292,93.126 C45.804,94.578 40.7,95.304 34.98,95.304 Z"></path> </g> </g> </g> </svg></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px" headers="colname"><div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>b</div><div style='font-size: 11px; color: gray;'>String</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_missing">0<br>0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_unique">12<br>0.92</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="mean">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="std">0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="min">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p05">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_1">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="median">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_3">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p95">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="max">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="iqr">0</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono" headers="icon"><svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"> <title>numeric</title> <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="numeric" transform="translate(4.000000, 4.339342)"> <rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"></rect> <g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"> <polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"></polygon> </g> </g> </g> </svg></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px" headers="colname"><div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>c</div><div style='font-size: 11px; color: gray;'>Int64</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_missing">2<br>0.15</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_unique">7<br>0.54</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="mean">5.73</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="std">2.72</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="min">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p05">2.05</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_1">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="median">7</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_3">8</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p95">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="max">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="iqr">5</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono" headers="icon"><svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"> <title>numeric</title> <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="numeric" transform="translate(4.000000, 4.339342)"> <rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"></rect> <g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"> <polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"></polygon> </g> </g> </g> </svg></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px" headers="colname"><div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>d</div><div style='font-size: 11px; color: gray;'>Float64</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_missing">0<br>0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_unique">12<br>0.92</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="mean">2,304.7</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="std">2,631.36</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="min">108.34</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p05">118.88</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_1">837.93</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="median">1,035.64</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_3">3,291.03</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p95">6,335.44</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="max">9999.99</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="iqr">2,453.1</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono" headers="icon"><svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"> <title>boolean</title> <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="boolean" transform="translate(4.000000, 4.339342)"> <rect id="square" stroke="#9C3E00" stroke-width="4" fill="#CF6100" x="2" y="2" width="168" height="168" rx="8"></rect> <g id="T" transform="translate(32.322000, 29.972000)" fill="#FFFFFF" fill-rule="nonzero"> <polygon points="29.326 11.61 29.326 60.028 16.254 60.028 16.254 11.61 0 11.61 0 0 45.58 0 45.58 11.61"></polygon> </g> <g id="F" transform="translate(99.122000, 81.972000)" fill="#FFFFFF" fill-rule="nonzero"> <polygon points="0 60.028 0 0 39.818 0 39.818 11.61 13.072 11.61 13.072 23.908 35.862 23.908 35.862 35.432 13.072 35.432 13.072 60.028"></polygon> </g> <rect id="line" stroke="#FFFFFF" fill="#FFFFFF" transform="translate(82.022437, 84.377940) rotate(-315.000000) translate(-82.022437, -84.377940) " x="78.0224369" y="25.37794" width="8" height="118"></rect> </g> </g> </svg></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px" headers="colname"><div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>e</div><div style='font-size: 11px; color: gray;'>Boolean</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_missing">0<br>0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_unique"><span style='font-weight: bold;'>T</span>0.62<br><span style='font-weight: bold;'>F</span>0.38</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="mean">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="std">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="min">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p05">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_1">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="median">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_3">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p95">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="max">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="iqr">-</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono" headers="icon"><svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"> <title>string</title> <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="string" transform="translate(4.000000, 4.500000)"> <rect id="square" stroke="#9A8700" stroke-width="4" fill="#CFB600" x="2" y="2" width="168" height="168" rx="8"></rect> <g id="S" transform="translate(49.516000, 38.280000)" fill="#FFFFFF" fill-rule="nonzero"> <path d="M34.98,95.304 C26.884,95.304 20.02,93.918 14.388,91.146 C8.756,88.374 3.96,84.788 0,80.388 L13.2,67.056 C19.448,74.096 27.148,77.616 36.3,77.616 C41.228,77.616 44.88,76.604 47.256,74.58 C49.632,72.556 50.82,69.872 50.82,66.528 C50.82,63.976 50.116,61.842 48.708,60.126 C47.3,58.41 44.44,57.244 40.128,56.628 L31.02,55.44 C21.252,54.208 14.102,51.26 9.57,46.596 C5.038,41.932 2.772,35.728 2.772,27.984 C2.772,23.848 3.564,20.064 5.148,16.632 C6.732,13.2 8.998,10.252 11.946,7.788 C14.894,5.324 18.502,3.41 22.77,2.046 C27.038,0.682 31.9,0 37.356,0 C44.308,0 50.424,1.1 55.704,3.3 C60.984,5.5 65.516,8.712 69.3,12.936 L55.968,26.4 C53.768,23.848 51.106,21.758 47.982,20.13 C44.858,18.502 40.92,17.688 36.168,17.688 C31.68,17.688 28.336,18.48 26.136,20.064 C23.936,21.648 22.836,23.848 22.836,26.664 C22.836,29.832 23.694,32.12 25.41,33.528 C27.126,34.936 29.92,35.948 33.792,36.564 L42.9,38.016 C52.404,39.512 59.444,42.438 64.02,46.794 C68.596,51.15 70.884,57.332 70.884,65.34 C70.884,69.74 70.092,73.788 68.508,77.484 C66.924,81.18 64.614,84.348 61.578,86.988 C58.542,89.628 54.78,91.674 50.292,93.126 C45.804,94.578 40.7,95.304 34.98,95.304 Z"></path> </g> </g> </g> </svg></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px" headers="colname"><div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>f</div><div style='font-size: 11px; color: gray;'>String</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_missing">0<br>0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_unique">3<br>0.23</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="mean">3.46</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="std">0.52</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="min">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p05">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_1">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="median">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_3">4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p95">4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="max">4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="iqr">1</td>
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


The [DataScan](../../reference/DataScan.md#pointblank.DataScan) object computes the same column-level statistics shown in the summary table. You can access the profile as a dictionary with `to_dict()`, export it as JSON with `to_json()`, or render the same tabular report with `get_tabular_report()`.


# Saving and Loading Profiles

A [DataScan](../../reference/DataScan.md#pointblank.DataScan) profile can be saved to disk and loaded back later. This is useful for establishing baselines that you compare against in the future.


``` python
# Save the profile to a JSON file
scan.save_to_json("small_table_profile.json")

# Later, load it back without needing the original data
loaded = pb.DataScan.load_from_json("small_table_profile.json")
```


The [from_json()](../../reference/DataScan.from_json.md#pointblank.DataScan.from_json) classmethod works with JSON strings directly, and `from_dict()` accepts the dictionary format produced by `to_dict()`. All three approaches produce a fully restored [DataScan](../../reference/DataScan.md#pointblank.DataScan) that retains the column names, types, statistics, and sample data from the original scan.


# Comparing Profiles for Drift

When your data changes over time, you can compare two [DataScan](../../reference/DataScan.md#pointblank.DataScan) profiles to detect drift. The `compare()` method identifies schema changes (columns added, removed, or with changed types) and statistical shifts in common columns.


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


You can also get the full comparison as a dictionary with `to_dict()`, or view it as a styled report:


``` python
diff.get_tabular_report()
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#puznwliyqq table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#puznwliyqq thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#puznwliyqq p { margin: 0; padding: 0; }
 #puznwliyqq .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #puznwliyqq .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #puznwliyqq .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #puznwliyqq .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #puznwliyqq .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #puznwliyqq .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #puznwliyqq .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #puznwliyqq .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #puznwliyqq .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #puznwliyqq .gt_column_spanner_outer:first-child { padding-left: 0; }
 #puznwliyqq .gt_column_spanner_outer:last-child { padding-right: 0; }
 #puznwliyqq .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #puznwliyqq .gt_spanner_row { border-bottom-style: hidden; }
 #puznwliyqq .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #puznwliyqq .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #puznwliyqq .gt_from_md> :first-child { margin-top: 0; }
 #puznwliyqq .gt_from_md> :last-child { margin-bottom: 0; }
 #puznwliyqq .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #puznwliyqq .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #puznwliyqq .gt_indent_1 { text-indent: 5px; }
 #puznwliyqq .gt_indent_2 { text-indent: calc(5px * 2); }
 #puznwliyqq .gt_indent_3 { text-indent: calc(5px * 3); }
 #puznwliyqq .gt_indent_4 { text-indent: calc(5px * 4); }
 #puznwliyqq .gt_indent_5 { text-indent: calc(5px * 5); }
 #puznwliyqq .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #puznwliyqq .gt_row_group_first td { border-top-width: 2px; }
 #puznwliyqq .gt_row_group_first th { border-top-width: 2px; }
 #puznwliyqq .gt_striped { color: #333333; background-color: #F4F4F4; }
 #puznwliyqq .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #puznwliyqq .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #puznwliyqq .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #puznwliyqq .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #puznwliyqq .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #puznwliyqq .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #puznwliyqq .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #puznwliyqq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #puznwliyqq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #puznwliyqq .gt_left { text-align: left; }
 #puznwliyqq .gt_center { text-align: center; }
 #puznwliyqq .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #puznwliyqq .gt_font_normal { font-weight: normal; }
 #puznwliyqq .gt_font_bold { font-weight: bold; }
 #puznwliyqq .gt_font_italic { font-style: italic; }
 #puznwliyqq .gt_super { font-size: 65%; }
 #puznwliyqq .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #puznwliyqq .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #puznwliyqq .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #puznwliyqq .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #puznwliyqq .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #puznwliyqq .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="width:100%;" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
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
<th colspan="6" id="title" class="gt_heading gt_title gt_font_normal">Profile Comparison: orders_v1 vs orders_v2</th>
</tr>
<tr class="gt_heading">
<th colspan="6" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Row count: 5 (baseline) vs 8 (current)</th>
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
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="column">order_id</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="status">Stats Changed</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_baseline">Int64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_current">Int64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="stat_changes">n_unique: 5 -> 8<br />
mean: 3 -> 4.5<br />
median: 3 -> 4.5<br />
std: 1.581 -> 2.449<br />
max: 5 -> 8<br />
p05: 1.02 -> 1.035<br />
q_1: 2 -> 2.75<br />
q_3: 4 -> 6.25<br />
p95: 4.8 -> 7.65<br />
iqr: 2 -> 3.5</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="drift">KS: 0.3750 (p=0.6963)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="column">amount</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="status">Stats Changed</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_baseline">Float64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_current">Float64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="stat_changes">n_unique: 5 -> 8<br />
mean: 20 -> 78.12<br />
median: 20 -> 27.5<br />
std: 7.906 -> 81.54<br />
max: 30 -> 200<br />
p05: 10.1 -> 10.18<br />
q_1: 15 -> 18.75<br />
q_3: 25 -> 156.2<br />
p95: 29 -> 191.2<br />
iqr: 10 -> 137.5</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="drift">KS: 0.3750 (p=0.6963)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="column">status</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="status">Stats Changed</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_baseline">String</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_current">String</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="stat_changes">mean: 4.4 -> 4.25<br />
std: 0.8944 -> 0.7071<br />
p95: 5.6 -> 5.3</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="drift">PSI: 0.0420</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="column">region</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="status">Added</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_baseline"></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_current">String</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="stat_changes"></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="drift"></td>
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
#lhopahdsgg table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#lhopahdsgg thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#lhopahdsgg p { margin: 0; padding: 0; }
 #lhopahdsgg .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #lhopahdsgg .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #lhopahdsgg .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #lhopahdsgg .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #lhopahdsgg .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #lhopahdsgg .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lhopahdsgg .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #lhopahdsgg .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #lhopahdsgg .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #lhopahdsgg .gt_column_spanner_outer:first-child { padding-left: 0; }
 #lhopahdsgg .gt_column_spanner_outer:last-child { padding-right: 0; }
 #lhopahdsgg .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #lhopahdsgg .gt_spanner_row { border-bottom-style: hidden; }
 #lhopahdsgg .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #lhopahdsgg .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #lhopahdsgg .gt_from_md> :first-child { margin-top: 0; }
 #lhopahdsgg .gt_from_md> :last-child { margin-bottom: 0; }
 #lhopahdsgg .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #lhopahdsgg .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #lhopahdsgg .gt_indent_1 { text-indent: 5px; }
 #lhopahdsgg .gt_indent_2 { text-indent: calc(5px * 2); }
 #lhopahdsgg .gt_indent_3 { text-indent: calc(5px * 3); }
 #lhopahdsgg .gt_indent_4 { text-indent: calc(5px * 4); }
 #lhopahdsgg .gt_indent_5 { text-indent: calc(5px * 5); }
 #lhopahdsgg .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #lhopahdsgg .gt_row_group_first td { border-top-width: 2px; }
 #lhopahdsgg .gt_row_group_first th { border-top-width: 2px; }
 #lhopahdsgg .gt_striped { color: #333333; background-color: #F4F4F4; }
 #lhopahdsgg .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lhopahdsgg .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #lhopahdsgg .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #lhopahdsgg .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #lhopahdsgg .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #lhopahdsgg .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #lhopahdsgg .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #lhopahdsgg .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lhopahdsgg .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #lhopahdsgg .gt_left { text-align: left; }
 #lhopahdsgg .gt_center { text-align: center; }
 #lhopahdsgg .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #lhopahdsgg .gt_font_normal { font-weight: normal; }
 #lhopahdsgg .gt_font_bold { font-weight: bold; }
 #lhopahdsgg .gt_font_italic { font-style: italic; }
 #lhopahdsgg .gt_super { font-size: 65%; }
 #lhopahdsgg .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lhopahdsgg .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #lhopahdsgg .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #lhopahdsgg .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #lhopahdsgg .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #lhopahdsgg .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="width:100%;" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
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
<th colspan="6" id="title" class="gt_heading gt_title gt_font_normal">Profile Comparison: revenue_v1 vs revenue_v2</th>
</tr>
<tr class="gt_heading">
<th colspan="6" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border">Row count: 20 (baseline) vs 20 (current)</th>
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
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="column">amount</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="status">Stats Changed</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_baseline">Float64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_current">Float64</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="stat_changes">mean: 21.95 -> 104.7<br />
median: 21.5 -> 152.5<br />
std: 7.536 -> 80.83<br />
max: 35 -> 200<br />
q_1: 16.5 -> 19.25<br />
q_3: 28.25 -> 176.2<br />
p95: 33.1 -> 195.2<br />
iqr: 11.75 -> 157</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="drift">PSI: 2.9414<br />
KS: 0.5500 (p=0.0026)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="column">category</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="status">Stats Changed</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_baseline">String</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="type_current">String</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="stat_changes">mean: 8.2 -> 8.8<br />
median: 8 -> 11<br />
std: 2.821 -> 3.037</td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 11px" headers="drift">PSI: 0.2506</td>
</tr>
</tbody>
</table>


The drift scores are also available programmatically through `to_dict()`:


``` python
drift = diff.to_dict().get("drift_scores", {})
for col_name, scores in drift.items():
    print(f"{col_name}: {scores}")
```


    amount: {'psi': 2.941363, 'ks_statistic': 0.55, 'ks_p_value': 0.002571}
    category: {'psi': 0.250553}


When a [DataScan](../../reference/DataScan.md#pointblank.DataScan) is loaded from a saved JSON file (with no original data attached), drift scores cannot be computed. The comparison still reports schema changes and raw stat differences, but the `drift_scores` field will be empty. If you need drift scores in a persist-and-compare workflow, keep the original data available when calling `compare()`.


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

The [has_changes](../../reference/DataScanDiff.md#pointblank.DataScanDiff.has_changes) property gives you a simple boolean for CI pipelines or alerting. So if anything shifted (schema or statistics) you can easily flag it. For more granular decisions, inspect `columns_added`, `columns_removed`, `columns_type_changed`, and the per-column `stat_diffs` and `drift_scores` in the object provided by `to_dict()`.


# Conclusion

The [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function and the [DataScan](../../reference/DataScan.md#pointblank.DataScan) class give you two levels of access to column-level profiling. The function is a quick, visual summary you can drop into a notebook to understand a dataset at a glance. The class provides the same statistics in a structured form that you can save, load, and compare over time. When your data evolves, `compare()` catches schema drift (columns added, removed, or retyped) and statistical drift (shifts in means, quantiles, null rates, and frequency distributions), optionally backed by formal measures like PSI and the KS test. Together, these tools make it straightforward to move from initial data discovery to ongoing data quality monitoring.
