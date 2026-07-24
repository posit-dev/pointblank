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
#vmeykimzng table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#vmeykimzng thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#vmeykimzng p { margin: 0; padding: 0; }
 #vmeykimzng .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #vmeykimzng .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #vmeykimzng .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #vmeykimzng .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #vmeykimzng .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #vmeykimzng .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #vmeykimzng .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #vmeykimzng .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #vmeykimzng .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #vmeykimzng .gt_column_spanner_outer:first-child { padding-left: 0; }
 #vmeykimzng .gt_column_spanner_outer:last-child { padding-right: 0; }
 #vmeykimzng .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #vmeykimzng .gt_spanner_row { border-bottom-style: hidden; }
 #vmeykimzng .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #vmeykimzng .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #vmeykimzng .gt_from_md> :first-child { margin-top: 0; }
 #vmeykimzng .gt_from_md> :last-child { margin-bottom: 0; }
 #vmeykimzng .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #vmeykimzng .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #vmeykimzng .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #vmeykimzng .gt_row_group_first td { border-top-width: 2px; }
 #vmeykimzng .gt_row_group_first th { border-top-width: 2px; }
 #vmeykimzng .gt_striped { color: #333333; background-color: #F4F4F4; }
 #vmeykimzng .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #vmeykimzng .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #vmeykimzng .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #vmeykimzng .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #vmeykimzng .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #vmeykimzng .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #vmeykimzng .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #vmeykimzng .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #vmeykimzng .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #vmeykimzng .gt_left { text-align: left; }
 #vmeykimzng .gt_center { text-align: center; }
 #vmeykimzng .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #vmeykimzng .gt_font_normal { font-weight: normal; }
 #vmeykimzng .gt_font_bold { font-weight: bold; }
 #vmeykimzng .gt_font_italic { font-style: italic; }
 #vmeykimzng .gt_super { font-size: 65%; }
 #vmeykimzng .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #vmeykimzng .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #vmeykimzng .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #vmeykimzng .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #vmeykimzng .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #vmeykimzng .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="11" class="gt_heading gt_title gt_font_normal">Missing Values   <span style="font-size: 14px; text-transform: uppercase; color: #333333">46,595 in total</span></th>
</tr>
<tr class="gt_heading">
<th colspan="11" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div>

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
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">year</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">month</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">day</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">dep_time</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #373737; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">sched_dep_time</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">dep_delay</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #373737; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">arr_time</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #272727; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">sched_arr_time</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">arr_delay</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #0b0b0b; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">carrier</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">flight</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">tailnum</td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #747474; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #b2b2b2; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #2b2b2b; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #363636; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #484848; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #4f4f4f; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #505050; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #757575; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">origin</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">dest</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">air_time</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #FFFFFF; background-color: #0b0b0b; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">distance</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">hour</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">minute</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
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
#wevvmdgkia table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#wevvmdgkia thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#wevvmdgkia p { margin: 0; padding: 0; }
 #wevvmdgkia .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #wevvmdgkia .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #wevvmdgkia .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #wevvmdgkia .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #wevvmdgkia .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #wevvmdgkia .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wevvmdgkia .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #wevvmdgkia .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #wevvmdgkia .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #wevvmdgkia .gt_column_spanner_outer:first-child { padding-left: 0; }
 #wevvmdgkia .gt_column_spanner_outer:last-child { padding-right: 0; }
 #wevvmdgkia .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #wevvmdgkia .gt_spanner_row { border-bottom-style: hidden; }
 #wevvmdgkia .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #wevvmdgkia .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #wevvmdgkia .gt_from_md> :first-child { margin-top: 0; }
 #wevvmdgkia .gt_from_md> :last-child { margin-bottom: 0; }
 #wevvmdgkia .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #wevvmdgkia .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #wevvmdgkia .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #wevvmdgkia .gt_row_group_first td { border-top-width: 2px; }
 #wevvmdgkia .gt_row_group_first th { border-top-width: 2px; }
 #wevvmdgkia .gt_striped { color: #333333; background-color: #F4F4F4; }
 #wevvmdgkia .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wevvmdgkia .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #wevvmdgkia .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #wevvmdgkia .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wevvmdgkia .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #wevvmdgkia .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #wevvmdgkia .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #wevvmdgkia .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wevvmdgkia .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #wevvmdgkia .gt_left { text-align: left; }
 #wevvmdgkia .gt_center { text-align: center; }
 #wevvmdgkia .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #wevvmdgkia .gt_font_normal { font-weight: normal; }
 #wevvmdgkia .gt_font_bold { font-weight: bold; }
 #wevvmdgkia .gt_font_italic { font-style: italic; }
 #wevvmdgkia .gt_super { font-size: 65%; }
 #wevvmdgkia .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wevvmdgkia .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #wevvmdgkia .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wevvmdgkia .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #wevvmdgkia .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #wevvmdgkia .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="11" class="gt_heading gt_title gt_font_normal">Missing Values   <span style="font-size: 14px; text-transform: uppercase; color: #333333">2 in total</span></th>
</tr>
<tr class="gt_heading">
<th colspan="11" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div>

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
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">date_time</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">date</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">a</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">b</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">c</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #808080; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">d</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">e</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">f</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
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
#towbwlpxaa table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#towbwlpxaa thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#towbwlpxaa p { margin: 0; padding: 0; }
 #towbwlpxaa .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #towbwlpxaa .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #towbwlpxaa .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #towbwlpxaa .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #towbwlpxaa .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #towbwlpxaa .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #towbwlpxaa .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #towbwlpxaa .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #towbwlpxaa .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #towbwlpxaa .gt_column_spanner_outer:first-child { padding-left: 0; }
 #towbwlpxaa .gt_column_spanner_outer:last-child { padding-right: 0; }
 #towbwlpxaa .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #towbwlpxaa .gt_spanner_row { border-bottom-style: hidden; }
 #towbwlpxaa .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #towbwlpxaa .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #towbwlpxaa .gt_from_md> :first-child { margin-top: 0; }
 #towbwlpxaa .gt_from_md> :last-child { margin-bottom: 0; }
 #towbwlpxaa .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #towbwlpxaa .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #towbwlpxaa .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #towbwlpxaa .gt_row_group_first td { border-top-width: 2px; }
 #towbwlpxaa .gt_row_group_first th { border-top-width: 2px; }
 #towbwlpxaa .gt_striped { color: #333333; background-color: #F4F4F4; }
 #towbwlpxaa .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #towbwlpxaa .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #towbwlpxaa .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #towbwlpxaa .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #towbwlpxaa .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #towbwlpxaa .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #towbwlpxaa .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #towbwlpxaa .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #towbwlpxaa .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #towbwlpxaa .gt_left { text-align: left; }
 #towbwlpxaa .gt_center { text-align: center; }
 #towbwlpxaa .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #towbwlpxaa .gt_font_normal { font-weight: normal; }
 #towbwlpxaa .gt_font_bold { font-weight: bold; }
 #towbwlpxaa .gt_font_italic { font-style: italic; }
 #towbwlpxaa .gt_super { font-size: 65%; }
 #towbwlpxaa .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #towbwlpxaa .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #towbwlpxaa .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #towbwlpxaa .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #towbwlpxaa .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #towbwlpxaa .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="11" class="gt_heading gt_title gt_font_normal">Missing Values <span style="color:#4CA64C;">✓</span></th>
</tr>
<tr class="gt_heading">
<th colspan="11" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div>

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
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">player_id</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">session_id</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">session_start</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">time</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">item_type</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">item_name</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">item_revenue</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">session_duration</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">start_day</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">acquisition</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px">country</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5; border-left: 1px solid #F0F0F0; border-right: 1px solid #F0F0F0; height: 20px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; background-color: lightblue"></td>
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
#mlnztjtask table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#mlnztjtask thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#mlnztjtask p { margin: 0; padding: 0; }
 #mlnztjtask .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #mlnztjtask .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #mlnztjtask .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #mlnztjtask .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #mlnztjtask .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #mlnztjtask .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #mlnztjtask .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #mlnztjtask .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #mlnztjtask .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #mlnztjtask .gt_column_spanner_outer:first-child { padding-left: 0; }
 #mlnztjtask .gt_column_spanner_outer:last-child { padding-right: 0; }
 #mlnztjtask .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #mlnztjtask .gt_spanner_row { border-bottom-style: hidden; }
 #mlnztjtask .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #mlnztjtask .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #mlnztjtask .gt_from_md> :first-child { margin-top: 0; }
 #mlnztjtask .gt_from_md> :last-child { margin-bottom: 0; }
 #mlnztjtask .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #mlnztjtask .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #mlnztjtask .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #mlnztjtask .gt_row_group_first td { border-top-width: 2px; }
 #mlnztjtask .gt_row_group_first th { border-top-width: 2px; }
 #mlnztjtask .gt_striped { color: #333333; background-color: #F4F4F4; }
 #mlnztjtask .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #mlnztjtask .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #mlnztjtask .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #mlnztjtask .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #mlnztjtask .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #mlnztjtask .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #mlnztjtask .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #mlnztjtask .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #mlnztjtask .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #mlnztjtask .gt_left { text-align: left; }
 #mlnztjtask .gt_center { text-align: center; }
 #mlnztjtask .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #mlnztjtask .gt_font_normal { font-weight: normal; }
 #mlnztjtask .gt_font_bold { font-weight: bold; }
 #mlnztjtask .gt_font_italic { font-style: italic; }
 #mlnztjtask .gt_super { font-size: 65%; }
 #mlnztjtask .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #mlnztjtask .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #mlnztjtask .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #mlnztjtask .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #mlnztjtask .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #mlnztjtask .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="width:100%;" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
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
<th colspan="7" class="gt_heading gt_title gt_font_normal">Missing Values by Reason</th>
</tr>
<tr class="gt_heading">
<th colspan="7" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div>

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
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 12px">age</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">4 (50%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">1 (12%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">2 (25%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">0 (0%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">--</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">1 (12%)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 12px">income</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">4 (50%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">2 (25%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">--</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">--</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">1 (12%)</td>
<td class="gt_row gt_right" style="font-family: IBM Plex Mono; font-size: 12px">1 (12%)</td>
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
#qjrihliazg table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#qjrihliazg thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#qjrihliazg p { margin: 0; padding: 0; }
 #qjrihliazg .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #qjrihliazg .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #qjrihliazg .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #qjrihliazg .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #qjrihliazg .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #qjrihliazg .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qjrihliazg .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #qjrihliazg .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #qjrihliazg .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #qjrihliazg .gt_column_spanner_outer:first-child { padding-left: 0; }
 #qjrihliazg .gt_column_spanner_outer:last-child { padding-right: 0; }
 #qjrihliazg .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #qjrihliazg .gt_spanner_row { border-bottom-style: hidden; }
 #qjrihliazg .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #qjrihliazg .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #qjrihliazg .gt_from_md> :first-child { margin-top: 0; }
 #qjrihliazg .gt_from_md> :last-child { margin-bottom: 0; }
 #qjrihliazg .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #qjrihliazg .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #qjrihliazg .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #qjrihliazg .gt_row_group_first td { border-top-width: 2px; }
 #qjrihliazg .gt_row_group_first th { border-top-width: 2px; }
 #qjrihliazg .gt_striped { color: #333333; background-color: #F4F4F4; }
 #qjrihliazg .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qjrihliazg .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #qjrihliazg .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #qjrihliazg .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #qjrihliazg .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #qjrihliazg .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #qjrihliazg .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #qjrihliazg .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qjrihliazg .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #qjrihliazg .gt_left { text-align: left; }
 #qjrihliazg .gt_center { text-align: center; }
 #qjrihliazg .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #qjrihliazg .gt_font_normal { font-weight: normal; }
 #qjrihliazg .gt_font_bold { font-weight: bold; }
 #qjrihliazg .gt_font_italic { font-style: italic; }
 #qjrihliazg .gt_super { font-size: 65%; }
 #qjrihliazg .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qjrihliazg .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #qjrihliazg .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #qjrihliazg .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #qjrihliazg .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #qjrihliazg .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="width:100%;" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
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
<th colspan="7" class="gt_heading gt_title gt_font_normal">Missing Pattern Heatmap</th>
</tr>
<tr class="gt_heading">
<th colspan="7" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div>

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
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 12px">age</td>
<td class="gt_row gt_center">50%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #d6d6d6">12%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #b8b8b8">25%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #f5f5f5">0%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #FFFFFF">--</td>
<td class="gt_row gt_center">12%</td>
</tr>
<tr>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 12px">income</td>
<td class="gt_row gt_center">50%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #b8b8b8">25%</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #FFFFFF">--</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #FFFFFF">--</td>
<td class="gt_row gt_center" style="color: #000000; background-color: #d6d6d6">12%</td>
<td class="gt_row gt_center">12%</td>
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
