# Missing Values Reporting

Sometimes values just aren't there: they're missing. This can either be expected or another thing to worry about. Either way, we can dig a little deeper if need be and use the [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) function to generate a summary table that can elucidate how many values are missing, and roughly where.


# Using and Understanding [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl)

The missing values table is arranged a lot like the column summary table (generated via the [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function) in that columns of the input table are arranged as rows in the reporting table. Let's use [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) on the `nycflights` dataset, which has a lot of missing values:


``` python
import pointblank as pb

nycflights = pb.load_dataset(dataset="nycflights", tbl_type="polars")

pb.missing_vals_tbl(nycflights)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#ioctopvoym table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#ioctopvoym thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#ioctopvoym p { margin: 0; padding: 0; }
 #ioctopvoym .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #ioctopvoym .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #ioctopvoym .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #ioctopvoym .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #ioctopvoym .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #ioctopvoym .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ioctopvoym .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #ioctopvoym .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #ioctopvoym .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #ioctopvoym .gt_column_spanner_outer:first-child { padding-left: 0; }
 #ioctopvoym .gt_column_spanner_outer:last-child { padding-right: 0; }
 #ioctopvoym .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #ioctopvoym .gt_spanner_row { border-bottom-style: hidden; }
 #ioctopvoym .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #ioctopvoym .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #ioctopvoym .gt_from_md> :first-child { margin-top: 0; }
 #ioctopvoym .gt_from_md> :last-child { margin-bottom: 0; }
 #ioctopvoym .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #ioctopvoym .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #ioctopvoym .gt_indent_1 { text-indent: 5px; }
 #ioctopvoym .gt_indent_2 { text-indent: calc(5px * 2); }
 #ioctopvoym .gt_indent_3 { text-indent: calc(5px * 3); }
 #ioctopvoym .gt_indent_4 { text-indent: calc(5px * 4); }
 #ioctopvoym .gt_indent_5 { text-indent: calc(5px * 5); }
 #ioctopvoym .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #ioctopvoym .gt_row_group_first td { border-top-width: 2px; }
 #ioctopvoym .gt_row_group_first th { border-top-width: 2px; }
 #ioctopvoym .gt_striped { color: #333333; background-color: #F4F4F4; }
 #ioctopvoym .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ioctopvoym .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #ioctopvoym .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #ioctopvoym .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #ioctopvoym .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #ioctopvoym .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #ioctopvoym .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #ioctopvoym .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ioctopvoym .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #ioctopvoym .gt_left { text-align: left; }
 #ioctopvoym .gt_center { text-align: center; }
 #ioctopvoym .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #ioctopvoym .gt_font_normal { font-weight: normal; }
 #ioctopvoym .gt_font_bold { font-weight: bold; }
 #ioctopvoym .gt_font_italic { font-style: italic; }
 #ioctopvoym .gt_super { font-size: 65%; }
 #ioctopvoym .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ioctopvoym .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #ioctopvoym .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #ioctopvoym .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #ioctopvoym .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #ioctopvoym .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="11" id="title" class="gt_heading gt_title gt_font_normal">Missing Values   <span style="font-size: 14px; text-transform: uppercase; color: #333333">46,595 in total</span></th>
</tr>
<tr class="gt_heading">
<th colspan="11" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">336,776</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">18</span>

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th rowspan="2" id="columns" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: black; font-size: 16px" scope="col">Column</th>
<th colspan="10" id="Row-Sector" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">Row Sector</th>
</tr>
<tr class="gt_col_headings">
<th id="1" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">1</th>
<th id="2" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">2</th>
<th id="3" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">3</th>
<th id="4" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">4</th>
<th id="5" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">5</th>
<th id="6" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">6</th>
<th id="7" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">7</th>
<th id="8" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">8</th>
<th id="9" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">9</th>
<th id="10" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">10</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">year</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">month</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">day</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">dep_time</td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #343434; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">sched_dep_time</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">dep_delay</td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #343434; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">arr_time</td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #272727; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">sched_arr_time</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">arr_delay</td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #101010; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">carrier</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">flight</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">tailnum</td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #6c6c6c; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #ababab; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #2a2a2a; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #333333; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #434343; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #494949; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #4a4a4a; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #6d6d6d; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">origin</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">dest</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">air_time</td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #101010; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">distance</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">hour</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">minute</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="11" class="gt_sourcenote">


<span style="font-size: 10px;">NO MISSING VALUES</span><span style="font-size: 10px;">     PROPORTION MISSING:  </span>

0%


100%


ROW SECTORS

<ol>
<li>1 - 33677</li>
<li>33678 - 67354</li>
<li>67355 - 101031</li>
<li>101032 - 134708</li>
<li>134709 - 168385</li>
<li>168386 - 202062</li>
<li>202063 - 235739</li>
<li>235740 - 269416</li>
<li>269417 - 303093</li>
<li>303094 - 336776</li>
</ol>
</div></td>
</tr>
</tfoot>

</table>


There are 18 columns in `nycflights` and they're arranged down the missing values table as rows. To the right we see column headers indicating 10 columns that are row sectors. Row sectors are groups of rows and each sector contains a tenth of the total rows in the table. The leftmost sectors are the rows at the top of the table whereas the sectors on the right are closer to the bottom. If you'd like to know which rows make up each row sector, there are details on this in the table footer area (click the `ROW SECTORS` text or the disclosure triangle).

Now that we know about row sectors, we need to understand the visuals here. A light blue cell indicates there are no (`0`) missing values within a given row sector of a column. For `nycflights` we can see that several columns have no missing values at all (i.e., the light blue color makes up the entire row in the missing values table).

When there are missing values in a column's row sector, you'll be met with a grayscale color. The proportion of missing values corresponds to the color ramp from light gray to solid black. Interestingly, most of the columns that have missing values appear to be related to each other in terms of the extent of missing values (i.e., the appearance in the reporting table looks roughly the same, indicating a sort of systematic missingness). These columns are `dep_time`, `dep_delay`, `arr_time`, `arr_delay`, and `air_time`.

The odd column out with regard to the distribution of missing values is `tailnum`. By scanning the row and observing that the grayscale color values are all a little different we see that the degree of missingness of more variable and not related to the other columns containing missing values.


# Missing Value Tables from the Other Datasets

The `small_table` dataset has only 13 rows to it. Let's use that as a Pandas DataFrame with [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl):


``` python
small_table = pb.load_dataset(dataset="small_table", tbl_type="pandas")

pb.missing_vals_tbl(small_table)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#jcsgeddbea table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#jcsgeddbea thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#jcsgeddbea p { margin: 0; padding: 0; }
 #jcsgeddbea .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #jcsgeddbea .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #jcsgeddbea .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #jcsgeddbea .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #jcsgeddbea .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #jcsgeddbea .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jcsgeddbea .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #jcsgeddbea .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #jcsgeddbea .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #jcsgeddbea .gt_column_spanner_outer:first-child { padding-left: 0; }
 #jcsgeddbea .gt_column_spanner_outer:last-child { padding-right: 0; }
 #jcsgeddbea .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #jcsgeddbea .gt_spanner_row { border-bottom-style: hidden; }
 #jcsgeddbea .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #jcsgeddbea .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #jcsgeddbea .gt_from_md> :first-child { margin-top: 0; }
 #jcsgeddbea .gt_from_md> :last-child { margin-bottom: 0; }
 #jcsgeddbea .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #jcsgeddbea .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #jcsgeddbea .gt_indent_1 { text-indent: 5px; }
 #jcsgeddbea .gt_indent_2 { text-indent: calc(5px * 2); }
 #jcsgeddbea .gt_indent_3 { text-indent: calc(5px * 3); }
 #jcsgeddbea .gt_indent_4 { text-indent: calc(5px * 4); }
 #jcsgeddbea .gt_indent_5 { text-indent: calc(5px * 5); }
 #jcsgeddbea .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #jcsgeddbea .gt_row_group_first td { border-top-width: 2px; }
 #jcsgeddbea .gt_row_group_first th { border-top-width: 2px; }
 #jcsgeddbea .gt_striped { color: #333333; background-color: #F4F4F4; }
 #jcsgeddbea .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jcsgeddbea .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #jcsgeddbea .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #jcsgeddbea .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #jcsgeddbea .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #jcsgeddbea .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #jcsgeddbea .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #jcsgeddbea .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jcsgeddbea .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #jcsgeddbea .gt_left { text-align: left; }
 #jcsgeddbea .gt_center { text-align: center; }
 #jcsgeddbea .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #jcsgeddbea .gt_font_normal { font-weight: normal; }
 #jcsgeddbea .gt_font_bold { font-weight: bold; }
 #jcsgeddbea .gt_font_italic { font-style: italic; }
 #jcsgeddbea .gt_super { font-size: 65%; }
 #jcsgeddbea .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jcsgeddbea .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #jcsgeddbea .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #jcsgeddbea .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #jcsgeddbea .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #jcsgeddbea .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="11" id="title" class="gt_heading gt_title gt_font_normal">Missing Values   <span style="font-size: 14px; text-transform: uppercase; color: #333333">2 in total</span></th>
</tr>
<tr class="gt_heading">
<th colspan="11" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div>

<span style="background-color: #150458; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #150458; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Pandas</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">13</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">8</span>

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th rowspan="2" id="columns" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: black; font-size: 16px" scope="col">Column</th>
<th colspan="10" id="Row-Sector" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">Row Sector</th>
</tr>
<tr class="gt_col_headings">
<th id="1" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">1</th>
<th id="2" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">2</th>
<th id="3" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">3</th>
<th id="4" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">4</th>
<th id="5" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">5</th>
<th id="6" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">6</th>
<th id="7" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">7</th>
<th id="8" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">8</th>
<th id="9" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">9</th>
<th id="10" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">10</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">date_time</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">date</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">a</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">b</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">c</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">d</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">e</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">f</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="11" class="gt_sourcenote">


<span style="font-size: 10px;">NO MISSING VALUES</span><span style="font-size: 10px;">     PROPORTION MISSING:  </span>

0%


100%


ROW SECTORS

<ol>
<li>1 - 1</li>
<li>2 - 2</li>
<li>3 - 3</li>
<li>4 - 4</li>
<li>5 - 5</li>
<li>6 - 6</li>
<li>7 - 7</li>
<li>8 - 8</li>
<li>9 - 9</li>
<li>10 - 13</li>
</ol>
</div></td>
</tr>
</tfoot>

</table>


It appears that only column `c` has missing values. And since the table is very small in terms of row count, most of the row sectors contain only a single row.

The `game_revenue` dataset has *no* missing values. And this can be easily proven by using [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) with it:


``` python
game_revenue = pb.load_dataset(dataset="game_revenue", tbl_type="duckdb")

pb.missing_vals_tbl(game_revenue)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#obwxywluwp table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#obwxywluwp thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#obwxywluwp p { margin: 0; padding: 0; }
 #obwxywluwp .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #obwxywluwp .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #obwxywluwp .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #obwxywluwp .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #obwxywluwp .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #obwxywluwp .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #obwxywluwp .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #obwxywluwp .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #obwxywluwp .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #obwxywluwp .gt_column_spanner_outer:first-child { padding-left: 0; }
 #obwxywluwp .gt_column_spanner_outer:last-child { padding-right: 0; }
 #obwxywluwp .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #obwxywluwp .gt_spanner_row { border-bottom-style: hidden; }
 #obwxywluwp .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #obwxywluwp .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #obwxywluwp .gt_from_md> :first-child { margin-top: 0; }
 #obwxywluwp .gt_from_md> :last-child { margin-bottom: 0; }
 #obwxywluwp .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #obwxywluwp .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #obwxywluwp .gt_indent_1 { text-indent: 5px; }
 #obwxywluwp .gt_indent_2 { text-indent: calc(5px * 2); }
 #obwxywluwp .gt_indent_3 { text-indent: calc(5px * 3); }
 #obwxywluwp .gt_indent_4 { text-indent: calc(5px * 4); }
 #obwxywluwp .gt_indent_5 { text-indent: calc(5px * 5); }
 #obwxywluwp .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #obwxywluwp .gt_row_group_first td { border-top-width: 2px; }
 #obwxywluwp .gt_row_group_first th { border-top-width: 2px; }
 #obwxywluwp .gt_striped { color: #333333; background-color: #F4F4F4; }
 #obwxywluwp .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #obwxywluwp .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #obwxywluwp .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #obwxywluwp .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #obwxywluwp .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #obwxywluwp .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #obwxywluwp .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #obwxywluwp .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #obwxywluwp .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #obwxywluwp .gt_left { text-align: left; }
 #obwxywluwp .gt_center { text-align: center; }
 #obwxywluwp .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #obwxywluwp .gt_font_normal { font-weight: normal; }
 #obwxywluwp .gt_font_bold { font-weight: bold; }
 #obwxywluwp .gt_font_italic { font-style: italic; }
 #obwxywluwp .gt_super { font-size: 65%; }
 #obwxywluwp .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #obwxywluwp .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #obwxywluwp .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #obwxywluwp .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #obwxywluwp .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #obwxywluwp .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="11" id="title" class="gt_heading gt_title gt_font_normal">Missing Values <span style="color:#4CA64C;">✓</span></th>
</tr>
<tr class="gt_heading">
<th colspan="11" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div>

<span style="background-color: #000000; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #000000; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">DuckDB</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">2,000</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">11</span>

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th rowspan="2" id="columns" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: black; font-size: 16px" scope="col">Column</th>
<th colspan="10" id="Row-Sector" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">Row Sector</th>
</tr>
<tr class="gt_col_headings">
<th id="1" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">1</th>
<th id="2" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">2</th>
<th id="3" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">3</th>
<th id="4" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">4</th>
<th id="5" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">5</th>
<th id="6" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">6</th>
<th id="7" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">7</th>
<th id="8" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">8</th>
<th id="9" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">9</th>
<th id="10" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: black; font-size: 16px" scope="col">10</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">player_id</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">session_id</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">session_start</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">time</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">item_type</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">item_name</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">item_revenue</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">session_duration</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">start_day</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">acquisition</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">country</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 1"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 2"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 3"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 4"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 5"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 6"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 7"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 8"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 9"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue" headers="Row-Sector 10"></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="11" class="gt_sourcenote">


<span style="font-size: 10px;">NO MISSING VALUES</span><span style="font-size: 10px;">     PROPORTION MISSING:  </span>

0%


100%


ROW SECTORS

<ol>
<li>1 - 200</li>
<li>201 - 400</li>
<li>401 - 600</li>
<li>601 - 800</li>
<li>801 - 1000</li>
<li>1001 - 1200</li>
<li>1201 - 1400</li>
<li>1401 - 1600</li>
<li>1601 - 1800</li>
<li>1801 - 2000</li>
</ol>
</div></td>
</tr>
</tfoot>

</table>


We see nothing but light blue in this report! The header also indicates that there are no missing values by displaying a large green check mark (the other report tables provided a count of total missing values across all columns).


# Structured Missingness by Reason

So far we've treated missingness as binary: a value is either `Null` or it isn't. But real-world data often encodes *why* a value is absent. Survey data distinguishes *refused* from *not asked* from *don't know*; clinical and statistical-package data use sentinel codes like `-99`, `".A"`, or `"NOT DONE"`. Pointblank captures this with the <a href="../../reference/MissingSpec.html#pointblank.MissingSpec" class="gdls-link"><code>MissingSpec</code></a> class, which maps sentinel values to human-readable *reasons*.

When you pass a `missing=` mapping of column names to [MissingSpec](../../reference/MissingSpec.md#pointblank.MissingSpec) objects, [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) switches from the sector heatmap to a *structured breakdown*: one row per column with the count and percentage of complete values and of each missing reason.

> **Note: Supplying `missing=` produces a different report**
>
> The structured breakdown is a *distinct visualization*, not an annotated version of the default sector heatmap. Adding `missing=` changes the table's whole layout. The report title changes too (from "Missing Values" to "Missing Values by Reason", or "Missing Pattern Heatmap" with `as_heatmap=True`), and the shared title styling and monospaced column list keep the two views recognizably part of the same family.


``` python
import polars as pl

survey = pl.DataFrame(
    {
        "age": [34, -98, 41, -99, 29, -98, 55, None],
        "income": [50000, -99, -1, None, 42000, -99, 38000, 61000],
    }
)

specs = {
    "age": pb.MissingSpec(reasons={-99: "not_asked", -98: "refused", -97: "dont_know"}),
    "income": pb.MissingSpec(reasons={-99: "not_asked", -1: "below_threshold"}),
}

pb.missing_vals_tbl(survey, missing=specs)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#yjihkytvmp table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#yjihkytvmp thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#yjihkytvmp p { margin: 0; padding: 0; }
 #yjihkytvmp .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #yjihkytvmp .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #yjihkytvmp .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #yjihkytvmp .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #yjihkytvmp .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #yjihkytvmp .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yjihkytvmp .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #yjihkytvmp .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #yjihkytvmp .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #yjihkytvmp .gt_column_spanner_outer:first-child { padding-left: 0; }
 #yjihkytvmp .gt_column_spanner_outer:last-child { padding-right: 0; }
 #yjihkytvmp .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #yjihkytvmp .gt_spanner_row { border-bottom-style: hidden; }
 #yjihkytvmp .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #yjihkytvmp .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #yjihkytvmp .gt_from_md> :first-child { margin-top: 0; }
 #yjihkytvmp .gt_from_md> :last-child { margin-bottom: 0; }
 #yjihkytvmp .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #yjihkytvmp .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #yjihkytvmp .gt_indent_1 { text-indent: 5px; }
 #yjihkytvmp .gt_indent_2 { text-indent: calc(5px * 2); }
 #yjihkytvmp .gt_indent_3 { text-indent: calc(5px * 3); }
 #yjihkytvmp .gt_indent_4 { text-indent: calc(5px * 4); }
 #yjihkytvmp .gt_indent_5 { text-indent: calc(5px * 5); }
 #yjihkytvmp .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #yjihkytvmp .gt_row_group_first td { border-top-width: 2px; }
 #yjihkytvmp .gt_row_group_first th { border-top-width: 2px; }
 #yjihkytvmp .gt_striped { color: #333333; background-color: #F4F4F4; }
 #yjihkytvmp .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yjihkytvmp .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #yjihkytvmp .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #yjihkytvmp .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yjihkytvmp .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #yjihkytvmp .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #yjihkytvmp .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #yjihkytvmp .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yjihkytvmp .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #yjihkytvmp .gt_left { text-align: left; }
 #yjihkytvmp .gt_center { text-align: center; }
 #yjihkytvmp .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #yjihkytvmp .gt_font_normal { font-weight: normal; }
 #yjihkytvmp .gt_font_bold { font-weight: bold; }
 #yjihkytvmp .gt_font_italic { font-style: italic; }
 #yjihkytvmp .gt_super { font-size: 65%; }
 #yjihkytvmp .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yjihkytvmp .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #yjihkytvmp .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yjihkytvmp .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #yjihkytvmp .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #yjihkytvmp .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="width:100%;" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<colgroup>
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr class="gt_heading">
<th colspan="7" id="title" class="gt_heading gt_title gt_font_normal">Missing Values by Reason</th>
</tr>
<tr class="gt_heading">
<th colspan="7" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">8</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">2</span>

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th rowspan="2" id="columns" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Column</th>
<th rowspan="2" id="complete" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">Complete</th>
<th colspan="4" id="Missing-Reasons" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">Missing Reasons</th>
<th rowspan="2" id="null" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col">Null</th>
</tr>
<tr class="gt_col_headings">
<th id="not_asked" class="gt_col_heading gt_columns_bottom_border gt_right" style="font-family: IBM Plex Mono; font-size: 12px" scope="col">not_asked</th>
<th id="refused" class="gt_col_heading gt_columns_bottom_border gt_right" style="font-family: IBM Plex Mono; font-size: 12px" scope="col">refused</th>
<th id="dont_know" class="gt_col_heading gt_columns_bottom_border gt_right" style="font-family: IBM Plex Mono; font-size: 12px" scope="col">dont_know</th>
<th id="below_threshold" class="gt_col_heading gt_columns_bottom_border gt_right" style="font-family: IBM Plex Mono; font-size: 12px" scope="col">below_threshold</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">age</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="complete">4 (50%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="Missing-Reasons not_asked">1 (12%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="Missing-Reasons refused">2 (25%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="Missing-Reasons dont_know">0 (0%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="Missing-Reasons below_threshold">--</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="null">1 (12%)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">income</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="complete">4 (50%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="Missing-Reasons not_asked">2 (25%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="Missing-Reasons refused">--</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="Missing-Reasons dont_know">--</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="Missing-Reasons below_threshold">1 (12%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px" headers="null">1 (12%)</td>
</tr>
</tbody>
</table>


Each [MissingSpec](../../reference/MissingSpec.md#pointblank.MissingSpec) declares the sentinel values for a column and the reason each one represents. Those declared (coded) reasons are grouped under the **Missing Reasons** spanner. By default, actual `Null` values are also counted as missing; because those are raw `Null`/`None`/`NA` values and *not* part of the spec, they're tallied in a fixed **Null** column at the far right (styled like **Complete**), rather than as a reason. Set `null_is_missing=False` on the spec if raw nulls should be treated as real values instead -- then there's no **Null** column at all.

The reason columns are the *union* of reasons across all the specs you provide. When a reason isn't defined for a particular column, that cell shows an em dash (`--`) rather than `0`. This signals "not applicable to this column", as distinct from a reason that *is* defined but simply wasn't observed (which shows `0 (0%)`).


## Viewing the pattern as a heatmap

For a more visual read of *where* missingness concentrates, pass `as_heatmap=True`. The reason columns are then shaded from light to dark by the proportion missing:


``` python
pb.missing_vals_tbl(survey, missing=specs, as_heatmap=True)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#tpowbtmqvi table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#tpowbtmqvi thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#tpowbtmqvi p { margin: 0; padding: 0; }
 #tpowbtmqvi .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #tpowbtmqvi .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #tpowbtmqvi .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #tpowbtmqvi .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #tpowbtmqvi .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #tpowbtmqvi .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #tpowbtmqvi .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #tpowbtmqvi .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #tpowbtmqvi .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #tpowbtmqvi .gt_column_spanner_outer:first-child { padding-left: 0; }
 #tpowbtmqvi .gt_column_spanner_outer:last-child { padding-right: 0; }
 #tpowbtmqvi .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #tpowbtmqvi .gt_spanner_row { border-bottom-style: hidden; }
 #tpowbtmqvi .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #tpowbtmqvi .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #tpowbtmqvi .gt_from_md> :first-child { margin-top: 0; }
 #tpowbtmqvi .gt_from_md> :last-child { margin-bottom: 0; }
 #tpowbtmqvi .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #tpowbtmqvi .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #tpowbtmqvi .gt_indent_1 { text-indent: 5px; }
 #tpowbtmqvi .gt_indent_2 { text-indent: calc(5px * 2); }
 #tpowbtmqvi .gt_indent_3 { text-indent: calc(5px * 3); }
 #tpowbtmqvi .gt_indent_4 { text-indent: calc(5px * 4); }
 #tpowbtmqvi .gt_indent_5 { text-indent: calc(5px * 5); }
 #tpowbtmqvi .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #tpowbtmqvi .gt_row_group_first td { border-top-width: 2px; }
 #tpowbtmqvi .gt_row_group_first th { border-top-width: 2px; }
 #tpowbtmqvi .gt_striped { color: #333333; background-color: #F4F4F4; }
 #tpowbtmqvi .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #tpowbtmqvi .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #tpowbtmqvi .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #tpowbtmqvi .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #tpowbtmqvi .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #tpowbtmqvi .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #tpowbtmqvi .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #tpowbtmqvi .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #tpowbtmqvi .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #tpowbtmqvi .gt_left { text-align: left; }
 #tpowbtmqvi .gt_center { text-align: center; }
 #tpowbtmqvi .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #tpowbtmqvi .gt_font_normal { font-weight: normal; }
 #tpowbtmqvi .gt_font_bold { font-weight: bold; }
 #tpowbtmqvi .gt_font_italic { font-style: italic; }
 #tpowbtmqvi .gt_super { font-size: 65%; }
 #tpowbtmqvi .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #tpowbtmqvi .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #tpowbtmqvi .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #tpowbtmqvi .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #tpowbtmqvi .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #tpowbtmqvi .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="width:100%;" aria-labelledby="title" aria-describedby="subtitle" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<colgroup>
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
<col style="width: 14%" />
</colgroup>
<thead>
<tr class="gt_heading">
<th colspan="7" id="title" class="gt_heading gt_title gt_font_normal">Missing Pattern Heatmap</th>
</tr>
<tr class="gt_heading">
<th colspan="7" id="subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">8</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">2</span>

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th rowspan="2" id="columns" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">Column</th>
<th rowspan="2" id="complete" class="gt_col_heading gt_columns_bottom_border gt_center" scope="col">Complete</th>
<th colspan="4" id="Missing-Reasons" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">Missing Reasons</th>
<th rowspan="2" id="null" class="gt_col_heading gt_columns_bottom_border gt_center" scope="col">Null</th>
</tr>
<tr class="gt_col_headings">
<th id="not_asked" class="gt_col_heading gt_columns_bottom_border gt_center" style="font-family: IBM Plex Mono; font-size: 12px" scope="col">not_asked</th>
<th id="refused" class="gt_col_heading gt_columns_bottom_border gt_center" style="font-family: IBM Plex Mono; font-size: 12px" scope="col">refused</th>
<th id="dont_know" class="gt_col_heading gt_columns_bottom_border gt_center" style="font-family: IBM Plex Mono; font-size: 12px" scope="col">dont_know</th>
<th id="below_threshold" class="gt_col_heading gt_columns_bottom_border gt_center" style="font-family: IBM Plex Mono; font-size: 12px" scope="col">below_threshold</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">age</td>
<td class="gt_row gt_center" headers="complete">50%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #d3d3d3" headers="Missing-Reasons not_asked">12%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #b2b2b2" headers="Missing-Reasons refused">25%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5" headers="Missing-Reasons dont_know">0%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #FFFFFF" headers="Missing-Reasons below_threshold">--</td>
<td class="gt_row gt_center" headers="null">12%</td>
</tr>
<tr>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 12px" headers="columns">income</td>
<td class="gt_row gt_center" headers="complete">50%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #b2b2b2" headers="Missing-Reasons not_asked">25%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #FFFFFF" headers="Missing-Reasons refused">--</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #FFFFFF" headers="Missing-Reasons dont_know">--</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #d3d3d3" headers="Missing-Reasons below_threshold">12%</td>
<td class="gt_row gt_center" headers="null">12%</td>
</tr>
</tbody>
</table>


## Pre-built specs for common standards

You don't always have to define reasons by hand. [MissingSpec](../../reference/MissingSpec.md#pointblank.MissingSpec) provides factory methods for common encodings, including CDISC/HL7 null flavors and SAS special missing values:


``` python
cdisc = pb.MissingSpec.from_cdisc_null_flavors()
print("NASK ->", cdisc.reason_for("NASK"))   # not_asked
print("UNK  ->", cdisc.reason_for("UNK"))     # unknown
```


    NASK -> not_asked
    UNK  -> unknown


When metadata is imported from SPSS, Stata, or SAS files (see the *Metadata Import* section), <a href="../../reference/MetadataImport.html#pointblank.MetadataImport" class="gdls-link"><code>MetadataImport.missing_specs()</code></a> auto-generates a `{column: MissingSpec}` mapping from the variables' declared missing values, ready to pass straight to [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl).

> **Note: Note**
>
> The same [MissingSpec](../../reference/MissingSpec.md#pointblank.MissingSpec) objects power missingness-aware *validation*, not just reporting. You can pass `missing=` to the `col_vals_*()` methods (to exclude sentinel values from a check) and use the dedicated <a href="../../reference/Validate.col_pct_missing.html#pointblank.Validate.col_pct_missing" class="gdls-link"><code>col_pct_missing()</code></a>, <a href="../../reference/Validate.col_missing_coded.html#pointblank.Validate.col_missing_coded" class="gdls-link"><code>col_missing_coded()</code></a>, <a href="../../reference/Validate.col_missing_only_coded.html#pointblank.Validate.col_missing_only_coded" class="gdls-link"><code>col_missing_only_coded()</code></a>, and <a href="../../reference/Validate.col_missing_consistent.html#pointblank.Validate.col_missing_consistent" class="gdls-link"><code>col_missing_consistent()</code></a> validation steps. See the *Validation Methods* article for details.
