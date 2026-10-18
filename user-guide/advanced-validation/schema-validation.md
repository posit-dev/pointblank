# Schema Validation

Schema validation in Pointblank allows you to verify that your data conforms to an expected structure and type specification. This is particularly useful when ensuring data consistency across systems or validating incoming data against predefined requirements.

Let's first look at the dataset we'll use for the first example:


``` python
import pointblank as pb

# Preview the small_table dataset we'll use throughout this guide
pb.preview(pb.load_dataset(dataset="small_table", tbl_type="polars"))
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
 #pb_preview_tbl .gt_subtitle { color: #333333; font-size: 12px; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
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
 #pb_preview_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_preview_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_preview_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_preview_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_preview_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
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

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_preview_tbl-title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="9" id="pb_preview_tbl-title" class="gt_heading gt_title gt_font_normal"><div>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">13</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">8</span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_preview_tbl-_row_num_" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"></th>
<th id="pb_preview_tbl-date_time" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

date_time

<em>Datetime</em>

</div></th>
<th id="pb_preview_tbl-date" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

date

<em>Date</em>

</div></th>
<th id="pb_preview_tbl-a" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

a

<em>Int64</em>

</div></th>
<th id="pb_preview_tbl-b" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

b

<em>String</em>

</div></th>
<th id="pb_preview_tbl-c" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

c

<em>Int64</em>

</div></th>
<th id="pb_preview_tbl-d" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

d

<em>Float64</em>

</div></th>
<th id="pb_preview_tbl-e" class="gt_col_heading gt_columns_bottom_border gt_center" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

e

<em>Boolean</em>

</div></th>
<th id="pb_preview_tbl-f" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

f

<em>String</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date_time">2016-01-04 11:00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date">2016-01-04</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-a">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-b">1-bcd-345</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-c">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-d">3423.29</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-e">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-f">high</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date_time">2016-01-04 00:32:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date">2016-01-04</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-a">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-b">5-egh-163</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-c">8</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-d">9999.99</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-e">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-f">low</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date_time">2016-01-05 13:32:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date">2016-01-05</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-a">6</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-b">8-kdg-938</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-c">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-d">2343.23</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-e">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-f">high</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date_time">2016-01-06 17:23:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date">2016-01-06</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-a">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-b">5-jdo-903</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159" headers="pb_preview_tbl-c">None</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-d">3892.4</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-e">False</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-f">mid</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-date_time">2016-01-09 12:36:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-date">2016-01-09</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-a">8</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-b">3-ldm-038</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-c">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-d">283.94</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-e">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-f">low</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">9</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date_time">2016-01-20 04:30:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date">2016-01-20</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-a">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-b">5-bce-642</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-c">9</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-d">837.93</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-e">False</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-f">high</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">10</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date_time">2016-01-20 04:30:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date">2016-01-20</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-a">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-b">5-bce-642</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-c">9</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-d">837.93</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-e">False</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-f">high</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">11</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date_time">2016-01-26 20:07:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date">2016-01-26</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-a">4</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-b">2-dmx-010</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-c">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-d">833.98</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-e">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-f">low</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">12</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date_time">2016-01-28 02:51:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date">2016-01-28</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-a">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-b">7-dmx-010</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-c">8</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-d">108.34</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-e">False</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-f">low</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">13</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date_time">2016-01-30 11:23:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-date">2016-01-30</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-a">1</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-b">3-dka-303</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159" headers="pb_preview_tbl-c">None</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-d">2230.09</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-e">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-f">high</td>
</tr>
</tbody>
</table>


# Schema Definition and Validation

A schema in Pointblank is created using the [Schema](../../reference/Schema.md#pointblank.Schema) class which defines the expected structure of a table. Once created, you apply schema validation through the <a href="../../reference/Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>Validate.col_schema_match()</code></a> validation step.


``` python
# Create a schema definition matching small_table structure
schema = pb.Schema(
    columns=[
        ("date_time",),   # Only check column name
        ("date",),        # Only check column name
        ("a", "Int64"),   # Check name and type
        ("b", "String"),  # Check name and type
        ("c", "Int64"),   # Check name and type
        ("d", "Float64"), # Check name and type
        ("e", "Boolean"), # Check name and type
        ("f",),           # Only check column name
    ]
)

# Validate the small_table against the schema
small_table_validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="small_table", tbl_type="polars"),
        label="Schema validation of `small_table`.",
    )
    .col_schema_match(schema=schema)
    .interrogate()
)

small_table_validation
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

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_tbl-title" aria-describedby="pb_tbl-subtitle" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" id="pb_tbl-title" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" id="pb_tbl-subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Schema validation of `small_table`.</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span>

</div></th>
</tr>
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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3NjaGVtYV9tYXRjaDwvdGl0bGU+CiAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTAtSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMC1jb2xfc2NoZW1hX21hdGNoIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4zMTAzNDUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0wLXJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01My43MTIyMzQsMzkuNzg4NTI2OCBMNTQuMjEyMjM0LDU2LjI4ODUyNjggTDQyLjIxMjIzNCw1Ni43ODg1MjY4IEw0Mi4yMTIyMzQsMzkuNzg4NTI2OCBMNTMuNzEyMjM0LDM5Ljc4ODUyNjggWiBNMzkuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCw1Ni43ODg1MjY4IEwyNy43MTIyMzQsNTYuNzg4NTI2OCBMMjcuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCwzOS43ODg1MjY4IFogTTI1LjIxMjIzNCwzOS43ODg1MjY4IEwyNS4yMTIyMzQsNTYuNzg4NTI2OCBMMTMuNzEyMjM0LDU2Ljc4ODUyNjggTDEzLjIxMjIzNCw0MC4yODg1MjY4IEwyNS4yMTIyMzQsMzkuNzg4NTI2OCBaIiBpZD0icGJfdGJsLXR5cGVfdXBkLTAtY29sdW1uc19zY2hlbWEiIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTAtdmVydGljYWxfZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMwLjAwMDAwMCwgMjkuMDAwMDAwKSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIj4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIyLjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSIyLjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9InBiX3RibC10eXBlX3VwZC0wLUxpbmUiPjwvbGluZT4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSI1LjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSI1LjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9InBiX3RibC10eXBlX3VwZC0wLUxpbmUtQ29weSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00MS43MTIyMzQsOS4wMTQ2NjkzNSBMNDEuNzEyMjM0LDI3LjAxNDY2OTQgTDUzLjcxMjIzNCwyNy4wMTQ2Njk0IEM1NC4yNjIyMzQsMjcuMDE0NjY5NCA1NC43MTIyMzQsMjYuNTY0NjY5NCA1NC43MTIyMzQsMjYuMDE0NjY5NCBMNTQuNzEyMjM0LDEwLjAxNDY2OTQgQzU0LjcxMjIzNCw5LjQ2NDY2OTM1IDU0LjI2MjIzNCw5LjAxNDY2OTM1IDUzLjcxMjIzNCw5LjAxNDY2OTM1IEw0MS43MTIyMzQsOS4wMTQ2NjkzNSBaIE0yNy4yMTIyMzQsOS4wMTQ2NjkzNSBDMjcuMjEyMjM0LDkuMDE0NjY5MzUgMjcuMjEyMjM0LDE1LjAxNDY2OTQgMjcuMjEyMjM0LDI3LjAxNDY2OTQgTDQwLjIxMjIzNCwyNy4wMTQ2Njk0IEw0MC4yMTIyMzQsOS4wMTQ2NjkzNSBDMzEuNTQ1NTY3Myw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IFogTTEzLjcxMjIzNCw5LjAxNDY2OTM1IEMxMy4xNjIyMzQsOS4wMTQ2NjkzNSAxMi43MTIyMzQsOS40NjQ2NjkzNSAxMi43MTIyMzQsMTAuMDE0NjY5NCBMMTIuNzEyMjM0LDI2LjAxNDY2OTQgQzEyLjcxMjIzNCwyNi41NjQ2Njk0IDEzLjE2MjIzNCwyNy4wMTQ2Njk0IDEzLjcxMjIzNCwyNy4wMTQ2Njk0IEwyNS43MTIyMzQsMjcuMDE0NjY5NCBMMjUuNzEyMjM0LDkuMDE0NjY5MzUgTDEzLjcxMjIzNCw5LjAxNDY2OTM1IFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMC1jb2x1bW5zX3JlYWwiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_schema_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">SCHEMA</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTAtdW5jaGFuZ2VkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0wLXVuY2hhbmdlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC41MDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTAtUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0icGJfdGJsLXRibC0wLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMy45NjM4MTg5LDguNjk5MzM1IEMxMy45MzY0NjIxLDguNzA0MzA5MjUgMTMuOTA5MTA1OSw4LjcxMTc2OTY4IDEzLjg4NDIzNTksOC43MTkyMzA3NCBDMTMuNzgyMjcwNCw4LjczNjYzOTY3IDEzLjY4Nzc2NTQsOC43NzY0MzExNSAxMy42MDU2OTU2LDguODM4NjA1MTggTDEwLjI0MzMxNTYsMTEuMzg1MjU5OCBDMTAuMDc2Njg4NiwxMS41MDQ2MzQzIDkuOTc3MjA5OTMsMTEuNjk4NjE4MSA5Ljk3NzIwOTkzLDExLjkwMjU0OTEgQzkuOTc3MjA5OTMsMTIuMTA2NDgwNyAxMC4wNzY2ODg2LDEyLjMwMDQ2MzkgMTAuMjQzMzE1NiwxMi40MTk4MzgzIEwxMy42MDU2OTU2LDE0Ljk2NjQ5MyBDMTMuODkxNjk3LDE1LjE4MDM3MjUgMTQuMjk3MDcyOSwxNS4xMjMxNzIxIDE0LjUxMDk1MTcsMTQuODM3MTcwNyBDMTQuNzI0ODMxMywxNC41NTExNjkyIDE0LjY2NzYzMDksMTQuMTQ1Nzk0IDE0LjM4MTYyOTQsMTMuOTMxOTE0NSBMMTIuNTMxMzI1NywxMi41MzkyMTI3IEwyMS44ODEyNDk1LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTEuMjY1ODg1NCBMMTIuNTMxMzI1NywxMS4yNjU4ODU0IEwxNC4zODE2Mjk0LDkuODczMTgzNjQgQzE0LjYzNzc4NzIsOS43MTY1MDQ1MyAxNC43NDk3MDA2LDkuNDAwNjYwMTQgMTQuNjQ3NzM1MSw5LjExNzE0NTUzIEMxNC41NDgyNTY0LDguODMzNjMxNTYgMTQuMjYyMjU1LDguNjU5NTQzNTIgMTMuOTYzODE4OSw4LjY5OTMzNSBaIiBpZD0icGJfdGJsLXRibC0wLWFycm93IiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNS45MjkyMzAsIDExLjg5NDczNykgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE1LjkyOTIzMCwgLTExLjg5NDczNykgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-eval"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-test_units">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-pass">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-fail">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-w_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC" headers="pb_tbl-e_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-c_upd">--</td>
<td class="gt_row gt_center" style="height: 40px" headers="pb_tbl-extract_upd">--</td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">
<hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(schema_check)</span> <span style="color:#4CA64C;">✓</span> Schema validation <strong>passed</strong>.</p>
Schema Comparison


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_step_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
<p>#pb_step_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
tr { background-color: transparent; }
#pb_step_tbl p { margin: 0; padding: 0; }
#pb_step_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
#pb_step_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
#pb_step_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
#pb_step_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
#pb_step_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
#pb_step_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
#pb_step_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
#pb_step_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
#pb_step_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
#pb_step_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
#pb_step_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
#pb_step_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
#pb_step_tbl .gt_spanner_row { border-bottom-style: hidden; }
#pb_step_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
#pb_step_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
#pb_step_tbl .gt_from_md> :first-child { margin-top: 0; }
#pb_step_tbl .gt_from_md> :last-child { margin-bottom: 0; }
#pb_step_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
#pb_step_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
#pb_step_tbl .gt_indent_1 { text-indent: 5px; }
#pb_step_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
#pb_step_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
#pb_step_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
#pb_step_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
#pb_step_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
#pb_step_tbl .gt_row_group_first td { border-top-width: 2px; }
#pb_step_tbl .gt_row_group_first th { border-top-width: 2px; }
#pb_step_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
#pb_step_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
#pb_step_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
#pb_step_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
#pb_step_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
#pb_step_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
#pb_step_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
#pb_step_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
#pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
#pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
#pb_step_tbl .gt_left { text-align: left; }
#pb_step_tbl .gt_center { text-align: center; }
#pb_step_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
#pb_step_tbl .gt_font_normal { font-weight: normal; }
#pb_step_tbl .gt_font_bold { font-weight: bold; }
#pb_step_tbl .gt_font_italic { font-style: italic; }
#pb_step_tbl .gt_super { font-size: 65%; }
#pb_step_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
#pb_step_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
#pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
#pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
#pb_step_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
#pb_step_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }</p>
</style>
<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">date_time</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Datetime(time_unit='us', time_zone=None)</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">date_time</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">date</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Date</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">date</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">a</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">a</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">b</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">b</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">String</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">5</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">c</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">5</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">c</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">6</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">d</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Float64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">6</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">d</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Float64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">7</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">e</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Boolean</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">7</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">e</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Boolean</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">8</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">f</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">8</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">f</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('date_time',), ('date',), ('a', 'Int64'), ('b', 'String'), ('c', 'Int64'), ('d', 'Float64'), ('e', 'Boolean'), ('f',)]</code>
</div></td>
</tr>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote"> 

Schema Match Settings

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64


</div></td>
</tr>
</tfoot>

</table>

</div></td>
</tr>
</tfoot>

</table>


The output shows the validation passed successfully. When all columns have the correct names and types as specified in the schema, the validation passes with a single passing test unit. If there were discrepancies, this would fail, but the basic output wouldn't show specific issues.

For detailed information about validation results, use <a href="../../reference/Validate.get_step_report.html#pointblank.Validate.get_step_report" class="gdls-link"><code>Validate.get_step_report()</code></a>:


``` python
small_table_validation.get_step_report(i=1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_step_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_step_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_step_tbl p { margin: 0; padding: 0; }
 #pb_step_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_step_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_step_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_step_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_step_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_step_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_step_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_step_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_step_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_step_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_step_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_step_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_step_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_step_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_step_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_step_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_step_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_step_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_step_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_step_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_step_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_step_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_step_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_step_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_step_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_left { text-align: left; }
 #pb_step_tbl .gt_center { text-align: center; }
 #pb_step_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_step_tbl .gt_font_normal { font-weight: normal; }
 #pb_step_tbl .gt_font_bold { font-weight: bold; }
 #pb_step_tbl .gt_font_italic { font-style: italic; }
 #pb_step_tbl .gt_super { font-size: 65%; }
 #pb_step_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_step_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_step_tbl-title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="8" id="pb_step_tbl-title" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #4CA64C;">✓</span>


COLUMN SCHEMA MATCH

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">date_time</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Datetime(time_unit='us', time_zone=None)</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">date_time</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">date</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Date</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">date</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">a</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">a</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">b</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">b</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">String</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">5</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">c</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">5</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">c</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">6</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">d</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Float64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">6</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">d</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Float64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">7</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">e</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Boolean</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">7</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">e</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Boolean</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">8</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">f</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">8</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">f</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('date_time',), ('date',), ('a', 'Int64'), ('b', 'String'), ('c', 'Int64'), ('d', 'Float64'), ('e', 'Boolean'), ('f',)]</code>
</div></td>
</tr>
</tfoot>

</table>


The step report provides specific details about which columns were checked and whether they matched the schema, helping diagnose issues when validation fails.


# Schema Components and Column Types

When defining a schema, you need to specify column names and optionally their data types. By default, Pointblank enforces strict validation where:

- all columns in your table must match the specified schema
- column order must match the schema
- column types are case-sensitive
- type names must match exactly

The schema definition accepts column types as string representations, which vary depending on your data source:

- `string`: Character data (may also be `"String"`, `"varchar"`, `"character"`, etc.)
- `integer`: Integer values (may also be `"Int64"`, `"int"`, `"bigint"`, etc.)
- `numeric`: Numeric values including integers and floating-point numbers (may also be `"Float64"`, `"double"`, `"decimal"`, etc.)
- `boolean`: Logical values (`True`/`False`) (may also be `"Boolean"`, `"bool"`, etc.)
- `datetime`: Date and time values (may also be `"Datetime"`, `"timestamp"`, etc.)
- `date`: Date values (may also be `"Date"`, etc.)
- `time`: Time values

For specific database engines or DataFrame libraries, you may need to use their exact type names (like `"VARCHAR(255)"` for SQL databases or `"Int64"` for Polars integers).


# Discovering Column Types

To easily determine the correct type string for columns in your data, Pointblank provides two helpful functions:


``` python
import polars as pl
from datetime import date

# Define a sample dataframe
sample_df = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "join_date": [date(2020, 1, 1), date(2021, 3, 15), date(2022, 7, 10)]
})
```


``` python
# Method 1: Using `preview()` with `show_types=True` to see column types
pb.preview(sample_df)
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
 #pb_preview_tbl .gt_subtitle { color: #333333; font-size: 12px; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
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
 #pb_preview_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_preview_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_preview_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_preview_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_preview_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
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

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_preview_tbl-title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="4" id="pb_preview_tbl-title" class="gt_heading gt_title gt_font_normal"><div>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">3</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">3</span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_preview_tbl-_row_num_" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"></th>
<th id="pb_preview_tbl-id" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

id

<em>Int64</em>

</div></th>
<th id="pb_preview_tbl-name" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

name

<em>String</em>

</div></th>
<th id="pb_preview_tbl-join_date" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

join_date

<em>Date</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-id">1</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-name">Alice</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-join_date">2020-01-01</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-id">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-name">Bob</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-join_date">2021-03-15</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-id">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-name">Charlie</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-join_date">2022-07-10</td>
</tr>
</tbody>
</table>


``` python
# Method 2: Using `col_summary_tbl()` which shows column types and other details
pb.col_summary_tbl(sample_df)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#yjoddnwcrj table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#yjoddnwcrj thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#yjoddnwcrj p { margin: 0; padding: 0; }
 #yjoddnwcrj .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #yjoddnwcrj .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #yjoddnwcrj .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #yjoddnwcrj .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #yjoddnwcrj .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #yjoddnwcrj .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yjoddnwcrj .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #yjoddnwcrj .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #yjoddnwcrj .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #yjoddnwcrj .gt_column_spanner_outer:first-child { padding-left: 0; }
 #yjoddnwcrj .gt_column_spanner_outer:last-child { padding-right: 0; }
 #yjoddnwcrj .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #yjoddnwcrj .gt_spanner_row { border-bottom-style: hidden; }
 #yjoddnwcrj .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #yjoddnwcrj .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #yjoddnwcrj .gt_from_md> :first-child { margin-top: 0; }
 #yjoddnwcrj .gt_from_md> :last-child { margin-bottom: 0; }
 #yjoddnwcrj .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #yjoddnwcrj .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #yjoddnwcrj .gt_indent_1 { text-indent: 5px; }
 #yjoddnwcrj .gt_indent_2 { text-indent: calc(5px * 2); }
 #yjoddnwcrj .gt_indent_3 { text-indent: calc(5px * 3); }
 #yjoddnwcrj .gt_indent_4 { text-indent: calc(5px * 4); }
 #yjoddnwcrj .gt_indent_5 { text-indent: calc(5px * 5); }
 #yjoddnwcrj .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #yjoddnwcrj .gt_row_group_first td { border-top-width: 2px; }
 #yjoddnwcrj .gt_row_group_first th { border-top-width: 2px; }
 #yjoddnwcrj .gt_striped { color: #333333; background-color: #F4F4F4; }
 #yjoddnwcrj .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yjoddnwcrj .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #yjoddnwcrj .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #yjoddnwcrj .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #yjoddnwcrj .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #yjoddnwcrj .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #yjoddnwcrj .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #yjoddnwcrj .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yjoddnwcrj .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #yjoddnwcrj .gt_left { text-align: left; }
 #yjoddnwcrj .gt_center { text-align: center; }
 #yjoddnwcrj .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #yjoddnwcrj .gt_font_normal { font-weight: normal; }
 #yjoddnwcrj .gt_font_bold { font-weight: bold; }
 #yjoddnwcrj .gt_font_italic { font-style: italic; }
 #yjoddnwcrj .gt_super { font-size: 65%; }
 #yjoddnwcrj .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yjoddnwcrj .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #yjoddnwcrj .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #yjoddnwcrj .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #yjoddnwcrj .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #yjoddnwcrj .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" id="title" class="gt_heading gt_title gt_font_normal"><div>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">3</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">3</span>

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
<td class="gt_row gt_left" style="font-family: IBM Plex Mono" headers="icon"><svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"> <title>numeric</title> <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="numeric" transform="translate(4.000000, 4.339342)"> <rect id="square" stroke="#640177" stroke-width="4" fill="#A400CF" x="2" y="2" width="168" height="168" rx="8"></rect> <g id="N" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"> <polygon points="28.116 48.576 19.272 30.096 18.876 30.096 18.876 92.136 0 92.136 0 0 21.912 0 47.124 43.56 55.968 62.04 56.364 62.04 56.364 0 75.24 0 75.24 92.136 53.328 92.136"></polygon> </g> </g> </g> </svg></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px" headers="colname"><div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>id</div><div style='font-size: 11px; color: gray;'>Int64</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_missing">0<br>0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_unique">3<br>1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="mean">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="std">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="min">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p05">1.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_1">1.5</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="median">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_3">2.5</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p95">2.9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="max">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="iqr">1</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono" headers="icon"><svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"> <title>string</title> <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="string" transform="translate(4.000000, 4.500000)"> <rect id="square" stroke="#9A8700" stroke-width="4" fill="#CFB600" x="2" y="2" width="168" height="168" rx="8"></rect> <g id="S" transform="translate(49.516000, 38.280000)" fill="#FFFFFF" fill-rule="nonzero"> <path d="M34.98,95.304 C26.884,95.304 20.02,93.918 14.388,91.146 C8.756,88.374 3.96,84.788 0,80.388 L13.2,67.056 C19.448,74.096 27.148,77.616 36.3,77.616 C41.228,77.616 44.88,76.604 47.256,74.58 C49.632,72.556 50.82,69.872 50.82,66.528 C50.82,63.976 50.116,61.842 48.708,60.126 C47.3,58.41 44.44,57.244 40.128,56.628 L31.02,55.44 C21.252,54.208 14.102,51.26 9.57,46.596 C5.038,41.932 2.772,35.728 2.772,27.984 C2.772,23.848 3.564,20.064 5.148,16.632 C6.732,13.2 8.998,10.252 11.946,7.788 C14.894,5.324 18.502,3.41 22.77,2.046 C27.038,0.682 31.9,0 37.356,0 C44.308,0 50.424,1.1 55.704,3.3 C60.984,5.5 65.516,8.712 69.3,12.936 L55.968,26.4 C53.768,23.848 51.106,21.758 47.982,20.13 C44.858,18.502 40.92,17.688 36.168,17.688 C31.68,17.688 28.336,18.48 26.136,20.064 C23.936,21.648 22.836,23.848 22.836,26.664 C22.836,29.832 23.694,32.12 25.41,33.528 C27.126,34.936 29.92,35.948 33.792,36.564 L42.9,38.016 C52.404,39.512 59.444,42.438 64.02,46.794 C68.596,51.15 70.884,57.332 70.884,65.34 C70.884,69.74 70.092,73.788 68.508,77.484 C66.924,81.18 64.614,84.348 61.578,86.988 C58.542,89.628 54.78,91.674 50.292,93.126 C45.804,94.578 40.7,95.304 34.98,95.304 Z"></path> </g> </g> </g> </svg></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px" headers="colname"><div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>name</div><div style='font-size: 11px; color: gray;'>String</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_missing">0<br>0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_unique">3<br>1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="mean">5</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="std">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="min">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p05">3.02</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_1">4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="median">5</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_3">6</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p95">6.8</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="max">7</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="iqr">2</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono" headers="icon"><svg width="24px" viewBox="0 0 180 181" version="1.1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" style="background: #FFFFFF; display: block; vertical-align: middle; margin: auto; padding-top: 0.5px;"> <title>date</title> <g id="icon" stroke="none" stroke-width="1" fill="none" fill-rule="evenodd"> <g id="date" transform="translate(4.000000, 4.339342)"> <rect id="square" stroke="#007D00" stroke-width="4" fill="#2CCF00" x="2" y="2" width="168" height="168" rx="8"></rect> <g id="D" transform="translate(49.664000, 39.864000)" fill="#FFFFFF" fill-rule="nonzero"> <path d="M0,0 L34.716,0 C40.788,0 46.332,0.968 51.348,2.904 C56.364,4.84 60.632,7.722 64.152,11.55 C67.672,15.378 70.4,20.174 72.336,25.938 C74.272,31.702 75.24,38.412 75.24,46.068 C75.24,53.724 74.272,60.434 72.336,66.198 C70.4,71.962 67.672,76.758 64.152,80.586 C60.632,84.414 56.364,87.296 51.348,89.232 C46.332,91.168 40.788,92.136 34.716,92.136 L0,92.136 L0,0 Z M34.716,74.316 C40.7,74.316 45.408,72.644 48.84,69.3 C52.272,65.956 53.988,60.588 53.988,53.196 L53.988,38.94 C53.988,31.548 52.272,26.18 48.84,22.836 C45.408,19.492 40.7,17.82 34.716,17.82 L20.064,17.82 L20.064,74.316 L34.716,74.316 Z"></path> </g> </g> </g> </svg></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px" headers="colname"><div style='font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;'>join_date</div><div style='font-size: 11px; color: gray;'>Date</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_missing">0<br>0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="n_unique">3<br>1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="mean">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="std">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="min">2020<br>01<br>01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p05">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_1">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="median">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="q_3">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="p95">-</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="max">2022<br>07<br>10</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;" headers="iqr">-</td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote">String columns statistics regard the string's length.</td>
</tr>
</tfoot>

</table>


These functions help you identify the exact type strings to use in your schema definitions, eliminating guesswork and ensuring compatibility with your data source.


# Creating a Schema

You can create a schema in four different ways, each with its own advantages. All schema objects can be printed to display their column names and data types.


## 1. Using a List of Tuples with `columns=`

This approach allows for mixed validation: some columns checked for both name and type, others only for name:


``` python
schema_tuples = pb.Schema(

    # List of tuples approach: flexible for mixed type/name checking ---
    columns=[
        ("name", "String"), # Check name and type
        ("age", "Int64"),   # Check name and type
        ("height",)         # Check name only
    ]
)

print(schema_tuples)
```


    Pointblank Schema
      name: String
      age: Int64
      height: <ANY>


This is the only method that allows checking just column names for some columns while checking both names and types for others.


## 2. Using a Dictionary with `columns=`

This approach is often the most readable when defining a schema manually, especially for larger schemas:


``` python
schema_dict = pb.Schema(

    # Dictionary approach (keys are column names, values are data types) ---
    columns={
        "name": "String",
        "age": "Int64",
        "height": "Float64",
        "created_at": "Datetime"
    }
)

print(schema_dict)
```


    Pointblank Schema
      name: String
      age: Int64
      height: Float64
      created_at: Datetime


With this method, you must always provide both column names (as keys) and their types (as values).


## 3. Using Keyword Arguments

For more readable code with a small number of columns:


``` python
schema_kwargs = pb.Schema(

    # Keyword arguments approach (more readable for simple schemas) ---
    name="String",
    age="Int64",
    height="Float64"
)

print(schema_kwargs)
```


    Pointblank Schema
      name: String
      age: Int64
      height: Float64


Like the dictionary method, this approach requires both column names and types.


## 4. Extracting from an Existing Table with `tbl=`

You can automatically extract a schema from an existing table:


``` python
import polars as pl

# Create a sample dataframe
df = pl.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "height": [5.6, 6.0, 5.8]
})

# Extract schema from table
schema_from_table = pb.Schema(tbl=df)

print(schema_from_table)
```


    Pointblank Schema
      name: String
      age: Int64
      height: Float64


This is especially useful when you want to validate that future data matches the structure of a reference dataset.


# Multiple Data Types for a Column

You can specify multiple acceptable types for a column by providing a list of types:


``` python
# Schema with multiple possible types for a column
schema_multi_types = pb.Schema(
    columns={
        "name": "String",
        "age": ["Int64", "Float64"],  # Accept either integer or float
        "active": "Boolean"
    }
)

print(schema_multi_types)
```


    Pointblank Schema
      name: String
      age: ['Int64', 'Float64']
      active: Boolean


This is useful when working with data sources that might represent the same information in different ways (e.g., integers sometimes stored as floats).


# Schema Validation Options

When using [col_schema_match()](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match), you can customize validation behavior with several important options:

| Option | Default | Description |
|----|----|----|
| `complete` | `True` | Require exact column presence (no extra columns allowed) |
| `in_order` | `True` | Enforce column order |
| `case_sensitive_colnames` | `True` | Make column name matching case-sensitive |
| `case_sensitive_dtypes` | `True` | Make data type matching case-sensitive |
| `full_match_dtypes` | `True` | Require exact (not partial) type name matches |


## Controlling Column Presence

By default, <a href="../../reference/Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>Validate.col_schema_match()</code></a> requires a complete match between the schema's columns and the table's columns. You can make this more flexible:


``` python
# Create a sample table
users_table_extra = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
    "extra_col": ["a", "b", "c"]  # Extra column not in schema
})

# Create a schema
schema = pb.Schema(
    columns={"id": "Int64", "name": "String", "age": "Int64"}
)

# Validate without requiring all columns to be present
validation = (
    pb.Validate(data=users_table_extra)
    .col_schema_match(
        schema=schema,

        # Allow schema columns to be a subset ---
        complete=False
    )
    .interrogate()
)

validation.get_step_report(i=1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_step_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_step_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_step_tbl p { margin: 0; padding: 0; }
 #pb_step_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_step_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_step_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_step_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_step_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_step_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_step_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_step_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_step_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_step_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_step_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_step_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_step_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_step_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_step_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_step_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_step_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_step_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_step_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_step_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_step_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_step_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_step_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_step_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_step_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_left { text-align: left; }
 #pb_step_tbl .gt_center { text-align: center; }
 #pb_step_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_step_tbl .gt_font_normal { font-weight: normal; }
 #pb_step_tbl .gt_font_bold { font-weight: bold; }
 #pb_step_tbl .gt_font_italic { font-style: italic; }
 #pb_step_tbl .gt_super { font-size: 65%; }
 #pb_step_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_step_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_step_tbl-title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="8" id="pb_step_tbl-title" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #4CA64C;">✓</span>


COLUMN SCHEMA MATCH

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">id</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">id</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">name</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">String</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">age</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">extra_col</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5; background-color: #FFC1C159" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp"></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; background-color: #FFC1C159" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp"></td>
<td class="gt_row gt_left" style="background-color: #FFC1C159" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; background-color: #FFC1C159" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp"></td>
<td class="gt_row gt_left" style="background-color: #FFC1C159" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('id', 'Int64'), ('name', 'String'), ('age', 'Int64')]</code>
</div></td>
</tr>
</tfoot>

</table>


## Column Order Enforcement

You can control whether column order matters in your validation:


``` python
# Create a sample table
users_table = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],
})

# Create a schema
schema = pb.Schema(
    columns={"name": "String", "age": "Int64", "id": "Int64"}
)

# Validate without enforcing column order
validation = (
    pb.Validate(data=users_table)
    .col_schema_match(
        schema=schema,

        # Don't enforce column order ---
        in_order=False
    )
    .interrogate()
)

validation.get_step_report(i=1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_step_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_step_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_step_tbl p { margin: 0; padding: 0; }
 #pb_step_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_step_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_step_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_step_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_step_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_step_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_step_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_step_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_step_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_step_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_step_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_step_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_step_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_step_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_step_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_step_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_step_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_step_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_step_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_step_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_step_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_step_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_step_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_step_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_step_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_left { text-align: left; }
 #pb_step_tbl .gt_center { text-align: center; }
 #pb_step_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_step_tbl .gt_font_normal { font-weight: normal; }
 #pb_step_tbl .gt_font_bold { font-weight: bold; }
 #pb_step_tbl .gt_font_italic { font-style: italic; }
 #pb_step_tbl .gt_super { font-size: 65%; }
 #pb_step_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_step_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_step_tbl-title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="8" id="pb_step_tbl-title" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #4CA64C;">✓</span>


COLUMN SCHEMA MATCH

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">id</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">id</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">name</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">String</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">age</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('name', 'String'), ('age', 'Int64'), ('id', 'Int64')]</code>
</div></td>
</tr>
</tfoot>

</table>


## Case Sensitivity

Control whether column names and data types are case-sensitive:


``` python
# Create schema with different case charactistics
case_schema = pb.Schema(
    columns={"ID": "int64", "NAME": "string", "AGE": "int64"}
)

# Create validation with case-insensitive column names and types
validation = (
    pb.Validate(data=users_table)
    .col_schema_match(
        schema=case_schema,

        # Ignore case in column names ---
        case_sensitive_colnames=False,

        # Ignore case in data type names ---
        case_sensitive_dtypes=False
    )
    .interrogate()
)

validation.get_step_report(i=1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_step_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_step_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_step_tbl p { margin: 0; padding: 0; }
 #pb_step_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_step_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_step_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_step_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_step_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_step_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_step_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_step_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_step_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_step_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_step_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_step_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_step_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_step_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_step_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_step_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_step_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_step_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_step_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_step_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_step_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_step_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_step_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_step_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_step_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_left { text-align: left; }
 #pb_step_tbl .gt_center { text-align: center; }
 #pb_step_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_step_tbl .gt_font_normal { font-weight: normal; }
 #pb_step_tbl .gt_font_bold { font-weight: bold; }
 #pb_step_tbl .gt_font_italic { font-style: italic; }
 #pb_step_tbl .gt_super { font-size: 65%; }
 #pb_step_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_step_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_step_tbl-title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="8" id="pb_step_tbl-title" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #4CA64C;">✓</span>


COLUMN SCHEMA MATCH

COMPLETE

IN ORDER

COLUMN = column

DTYPE = dtype

float ≠ float64

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">id</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">ID</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">NAME</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">string</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">AGE</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('ID', 'int64'), ('NAME', 'string'), ('AGE', 'int64')]</code>
</div></td>
</tr>
</tfoot>

</table>


## Type Matching Precision

Control how strictly data types must match:


``` python
# Create schema with simplified type names
type_schema = pb.Schema(

    # Using simplified type names ---
    columns={"id": "int", "name": "str", "age": "int"}
)

# Allow partial type matches
validation = (
    pb.Validate(data=users_table)
    .col_schema_match(
        schema=type_schema,

        # Ignore case in data type names ---
        case_sensitive_dtypes=False,

        # Allow partial type name matches ---
        full_match_dtypes=False
    )
    .interrogate()
)

validation.get_step_report(i=1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_step_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_step_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_step_tbl p { margin: 0; padding: 0; }
 #pb_step_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_step_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_step_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_step_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_step_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_step_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_step_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_step_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_step_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_step_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_step_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_step_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_step_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_step_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_step_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_step_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_step_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_step_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_step_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_step_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_step_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_step_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_step_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_step_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_step_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_left { text-align: left; }
 #pb_step_tbl .gt_center { text-align: center; }
 #pb_step_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_step_tbl .gt_font_normal { font-weight: normal; }
 #pb_step_tbl .gt_font_bold { font-weight: bold; }
 #pb_step_tbl .gt_font_italic { font-style: italic; }
 #pb_step_tbl .gt_super { font-size: 65%; }
 #pb_step_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_step_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_step_tbl-title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="8" id="pb_step_tbl-title" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #4CA64C;">✓</span>


COLUMN SCHEMA MATCH

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE = dtype

float = float64

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">id</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">id</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">int</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">name</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">str</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">age</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">int</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('id', 'int'), ('name', 'str'), ('age', 'int')]</code>
</div></td>
</tr>
</tfoot>

</table>


# Common Schema Validation Patterns

This section explores common patterns for applying schema validation to different scenarios. Each pattern addresses specific validation needs you might encounter when working with real-world data. We'll examine the step reports (<a href="../../reference/Validate.get_step_report.html#pointblank.Validate.get_step_report" class="gdls-link"><code>Validate.get_step_report()</code></a>) for these validations since they provide more detailed information about what was checked and how the validation performed, offering an intuitive way to understand the results beyond simple pass/fail indicators.


## Structural Validation Only

When you only care about column names but not their types:


``` python
# Create a schema with only column names
structure_schema = pb.Schema(
    columns=["id", "name", "age", "extra_col"]
)

# Validate structure only
validation = (
    pb.Validate(data=users_table_extra)
    .col_schema_match(schema=structure_schema)
    .interrogate()
)

validation.get_step_report(i=1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_step_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_step_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_step_tbl p { margin: 0; padding: 0; }
 #pb_step_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_step_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_step_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_step_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_step_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_step_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_step_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_step_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_step_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_step_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_step_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_step_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_step_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_step_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_step_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_step_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_step_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_step_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_step_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_step_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_step_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_step_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_step_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_step_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_step_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_left { text-align: left; }
 #pb_step_tbl .gt_center { text-align: center; }
 #pb_step_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_step_tbl .gt_font_normal { font-weight: normal; }
 #pb_step_tbl .gt_font_bold { font-weight: bold; }
 #pb_step_tbl .gt_font_italic { font-style: italic; }
 #pb_step_tbl .gt_super { font-size: 65%; }
 #pb_step_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_step_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_step_tbl-title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="8" id="pb_step_tbl-title" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #4CA64C;">✓</span>


COLUMN SCHEMA MATCH

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">id</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">id</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">name</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">age</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">extra_col</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">extra_col</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('id',), ('name',), ('age',), ('extra_col',)]</code>
</div></td>
</tr>
</tfoot>

</table>


## Mixed Validation

Validate types for critical columns but just presence for others:


``` python
# Mixed validation for different columns
mixed_schema = pb.Schema(
    columns=[
        ("id", "Int64"),     # Check name and type
        ("name", "String"),  # Check name and type
        ("age",),            # Check name only
        ("extra_col",)       # Check name only
    ]
)

# Validate with mixed approach
validation = (
    pb.Validate(data=users_table_extra)
    .col_schema_match(schema=mixed_schema)
    .interrogate()
)

validation.get_step_report(i=1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_step_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_step_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_step_tbl p { margin: 0; padding: 0; }
 #pb_step_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_step_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_step_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_step_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_step_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_step_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_step_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_step_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_step_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_step_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_step_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_step_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_step_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_step_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_step_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_step_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_step_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_step_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_step_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_step_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_step_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_step_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_step_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_step_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_step_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_left { text-align: left; }
 #pb_step_tbl .gt_center { text-align: center; }
 #pb_step_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_step_tbl .gt_font_normal { font-weight: normal; }
 #pb_step_tbl .gt_font_bold { font-weight: bold; }
 #pb_step_tbl .gt_font_italic { font-style: italic; }
 #pb_step_tbl .gt_super { font-size: 65%; }
 #pb_step_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_step_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_step_tbl-title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="8" id="pb_step_tbl-title" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #4CA64C;">✓</span>


COLUMN SCHEMA MATCH

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">id</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">id</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">name</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">String</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">age</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">extra_col</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">extra_col</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">--</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('id', 'Int64'), ('name', 'String'), ('age',), ('extra_col',)]</code>
</div></td>
</tr>
</tfoot>

</table>


## Progressive Schema Evolution

As your data evolves, you might need to adapt your validation approach:


``` python
# Original schema
original_schema = pb.Schema(
    columns={
        "id": "Int64",
        "name": "String"
    }
)

# New data with additional columns
evolved_data = pl.DataFrame({
    "id": [1, 2, 3],
    "name": ["Alice", "Bob", "Charlie"],
    "age": [25, 30, 35],           # New column
    "active": [True, False, True]  # New column
})

# Validate with flexible parameters
validation = (
    pb.Validate(evolved_data)
    .col_schema_match(
        schema=original_schema,

        # Allow extra columns ---
        complete=False,

        # Don't enforce order ---
        in_order=False
    )
    .interrogate()
)

validation.get_step_report(i=1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_step_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_step_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_step_tbl p { margin: 0; padding: 0; }
 #pb_step_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_step_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_step_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_step_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_step_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_step_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_step_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_step_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_step_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_step_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_step_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_step_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_step_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_step_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_step_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_step_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_step_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_step_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_step_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_step_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_step_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_step_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_step_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_step_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_step_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_step_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_left { text-align: left; }
 #pb_step_tbl .gt_center { text-align: center; }
 #pb_step_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_step_tbl .gt_font_normal { font-weight: normal; }
 #pb_step_tbl .gt_font_bold { font-weight: bold; }
 #pb_step_tbl .gt_font_italic { font-style: italic; }
 #pb_step_tbl .gt_super { font-size: 65%; }
 #pb_step_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_step_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_step_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_step_tbl-title" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="8" id="pb_step_tbl-title" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #4CA64C;">✓</span>


COLUMN SCHEMA MATCH

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64

</div></th>
</tr>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">id</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">id</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">name</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">String</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp"></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp"></td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp"></td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">active</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Boolean</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp"></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp"></td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp"></td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('id', 'Int64'), ('name', 'String')]</code>
</div></td>
</tr>
</tfoot>

</table>


# Integrating with Larger Validation Workflows

Schema validation is often just one part of a comprehensive data validation strategy. You can combine schema checks with other validation steps:


``` python
# Define a schema
schema = pb.Schema(
    columns={
        "id": "Int64",
        "name": "String",
        "age": "Int64"
    }
)

# Create a validation plan
validation = (
    pb.Validate(
        users_table,
        label="User data validation",
        thresholds=pb.Thresholds(warning=0.05, error=0.1)
    )

    # Add schema validation ---
    .col_schema_match(schema=schema)

    # Add other validation steps ---
    .col_vals_not_null(columns="id")
    .col_vals_gt(columns="age", value=26)
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

<table class="gt_table" style="table-layout: fixed;; width: 0px" aria-labelledby="pb_tbl-title" aria-describedby="pb_tbl-subtitle" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" id="pb_tbl-title" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" id="pb_tbl-subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">User data validation</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span><span style="background-color: #AAAAAA; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; border: solid 1px #AAAAAA; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">WARNING</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #AAAAAA; padding: 2px 15px 2px 15px; font-size: smaller; margin-right: 5px;">0.05</span><span style="background-color: #EBBC14; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #EBBC14; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">ERROR</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #EBBC14; padding: 2px 15px 2px 15px; font-size: smaller; margin-right: 5px;">0.1</span><span style="background-color: #FF3300; color: white; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 1px; border: solid 1px #FF3300; font-weight: bold; padding: 2px 15px 2px 15px; font-size: smaller;">CRITICAL</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #FF3300; padding: 2px 15px 2px 15px; font-size: smaller;">--</span></span>

</div></th>
</tr>
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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3NjaGVtYV9tYXRjaDwvdGl0bGU+CiAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTAtSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMC1jb2xfc2NoZW1hX21hdGNoIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4zMTAzNDUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0wLXJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01My43MTIyMzQsMzkuNzg4NTI2OCBMNTQuMjEyMjM0LDU2LjI4ODUyNjggTDQyLjIxMjIzNCw1Ni43ODg1MjY4IEw0Mi4yMTIyMzQsMzkuNzg4NTI2OCBMNTMuNzEyMjM0LDM5Ljc4ODUyNjggWiBNMzkuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCw1Ni43ODg1MjY4IEwyNy43MTIyMzQsNTYuNzg4NTI2OCBMMjcuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCwzOS43ODg1MjY4IFogTTI1LjIxMjIzNCwzOS43ODg1MjY4IEwyNS4yMTIyMzQsNTYuNzg4NTI2OCBMMTMuNzEyMjM0LDU2Ljc4ODUyNjggTDEzLjIxMjIzNCw0MC4yODg1MjY4IEwyNS4yMTIyMzQsMzkuNzg4NTI2OCBaIiBpZD0icGJfdGJsLXR5cGVfdXBkLTAtY29sdW1uc19zY2hlbWEiIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTAtdmVydGljYWxfZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMwLjAwMDAwMCwgMjkuMDAwMDAwKSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIj4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIyLjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSIyLjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9InBiX3RibC10eXBlX3VwZC0wLUxpbmUiPjwvbGluZT4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSI1LjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSI1LjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9InBiX3RibC10eXBlX3VwZC0wLUxpbmUtQ29weSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00MS43MTIyMzQsOS4wMTQ2NjkzNSBMNDEuNzEyMjM0LDI3LjAxNDY2OTQgTDUzLjcxMjIzNCwyNy4wMTQ2Njk0IEM1NC4yNjIyMzQsMjcuMDE0NjY5NCA1NC43MTIyMzQsMjYuNTY0NjY5NCA1NC43MTIyMzQsMjYuMDE0NjY5NCBMNTQuNzEyMjM0LDEwLjAxNDY2OTQgQzU0LjcxMjIzNCw5LjQ2NDY2OTM1IDU0LjI2MjIzNCw5LjAxNDY2OTM1IDUzLjcxMjIzNCw5LjAxNDY2OTM1IEw0MS43MTIyMzQsOS4wMTQ2NjkzNSBaIE0yNy4yMTIyMzQsOS4wMTQ2NjkzNSBDMjcuMjEyMjM0LDkuMDE0NjY5MzUgMjcuMjEyMjM0LDE1LjAxNDY2OTQgMjcuMjEyMjM0LDI3LjAxNDY2OTQgTDQwLjIxMjIzNCwyNy4wMTQ2Njk0IEw0MC4yMTIyMzQsOS4wMTQ2NjkzNSBDMzEuNTQ1NTY3Myw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IFogTTEzLjcxMjIzNCw5LjAxNDY2OTM1IEMxMy4xNjIyMzQsOS4wMTQ2NjkzNSAxMi43MTIyMzQsOS40NjQ2NjkzNSAxMi43MTIyMzQsMTAuMDE0NjY5NCBMMTIuNzEyMjM0LDI2LjAxNDY2OTQgQzEyLjcxMjIzNCwyNi41NjQ2Njk0IDEzLjE2MjIzNCwyNy4wMTQ2Njk0IDEzLjcxMjIzNCwyNy4wMTQ2Njk0IEwyNS43MTIyMzQsMjcuMDE0NjY5NCBMMjUuNzEyMjM0LDkuMDE0NjY5MzUgTDEzLjcxMjIzNCw5LjAxNDY2OTM1IFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMC1jb2x1bW5zX3JlYWwiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_schema_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">SCHEMA</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTAtdW5jaGFuZ2VkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0wLXVuY2hhbmdlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC41MDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTAtUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0icGJfdGJsLXRibC0wLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMy45NjM4MTg5LDguNjk5MzM1IEMxMy45MzY0NjIxLDguNzA0MzA5MjUgMTMuOTA5MTA1OSw4LjcxMTc2OTY4IDEzLjg4NDIzNTksOC43MTkyMzA3NCBDMTMuNzgyMjcwNCw4LjczNjYzOTY3IDEzLjY4Nzc2NTQsOC43NzY0MzExNSAxMy42MDU2OTU2LDguODM4NjA1MTggTDEwLjI0MzMxNTYsMTEuMzg1MjU5OCBDMTAuMDc2Njg4NiwxMS41MDQ2MzQzIDkuOTc3MjA5OTMsMTEuNjk4NjE4MSA5Ljk3NzIwOTkzLDExLjkwMjU0OTEgQzkuOTc3MjA5OTMsMTIuMTA2NDgwNyAxMC4wNzY2ODg2LDEyLjMwMDQ2MzkgMTAuMjQzMzE1NiwxMi40MTk4MzgzIEwxMy42MDU2OTU2LDE0Ljk2NjQ5MyBDMTMuODkxNjk3LDE1LjE4MDM3MjUgMTQuMjk3MDcyOSwxNS4xMjMxNzIxIDE0LjUxMDk1MTcsMTQuODM3MTcwNyBDMTQuNzI0ODMxMywxNC41NTExNjkyIDE0LjY2NzYzMDksMTQuMTQ1Nzk0IDE0LjM4MTYyOTQsMTMuOTMxOTE0NSBMMTIuNTMxMzI1NywxMi41MzkyMTI3IEwyMS44ODEyNDk1LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTEuMjY1ODg1NCBMMTIuNTMxMzI1NywxMS4yNjU4ODU0IEwxNC4zODE2Mjk0LDkuODczMTgzNjQgQzE0LjYzNzc4NzIsOS43MTY1MDQ1MyAxNC43NDk3MDA2LDkuNDAwNjYwMTQgMTQuNjQ3NzM1MSw5LjExNzE0NTUzIEMxNC41NDgyNTY0LDguODMzNjMxNTYgMTQuMjYyMjU1LDguNjU5NTQzNTIgMTMuOTYzODE4OSw4LjY5OTMzNSBaIiBpZD0icGJfdGJsLXRibC0wLWFycm93IiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNS45MjkyMzAsIDExLjg5NDczNykgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE1LjkyOTIzMCwgLTExLjg5NDczNykgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-eval"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-test_units">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-pass">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-fail">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-w_upd"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC" headers="pb_tbl-e_upd"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-c_upd">--</td>
<td class="gt_row gt_center" style="height: 40px" headers="pb_tbl-extract_upd">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9InBiX3RibC10eXBlX3VwZC0xLUljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTEtY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icGJfdGJsLXR5cGVfdXBkLTEtcmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0icGJfdGJsLXR5cGVfdXBkLTEtb21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0icGJfdGJsLXR5cGVfdXBkLTEtbGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9InBiX3RibC10eXBlX3VwZC0xLWxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">id</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTEtdW5jaGFuZ2VkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0xLXVuY2hhbmdlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC41MDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTEtUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0icGJfdGJsLXRibC0xLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMy45NjM4MTg5LDguNjk5MzM1IEMxMy45MzY0NjIxLDguNzA0MzA5MjUgMTMuOTA5MTA1OSw4LjcxMTc2OTY4IDEzLjg4NDIzNTksOC43MTkyMzA3NCBDMTMuNzgyMjcwNCw4LjczNjYzOTY3IDEzLjY4Nzc2NTQsOC43NzY0MzExNSAxMy42MDU2OTU2LDguODM4NjA1MTggTDEwLjI0MzMxNTYsMTEuMzg1MjU5OCBDMTAuMDc2Njg4NiwxMS41MDQ2MzQzIDkuOTc3MjA5OTMsMTEuNjk4NjE4MSA5Ljk3NzIwOTkzLDExLjkwMjU0OTEgQzkuOTc3MjA5OTMsMTIuMTA2NDgwNyAxMC4wNzY2ODg2LDEyLjMwMDQ2MzkgMTAuMjQzMzE1NiwxMi40MTk4MzgzIEwxMy42MDU2OTU2LDE0Ljk2NjQ5MyBDMTMuODkxNjk3LDE1LjE4MDM3MjUgMTQuMjk3MDcyOSwxNS4xMjMxNzIxIDE0LjUxMDk1MTcsMTQuODM3MTcwNyBDMTQuNzI0ODMxMywxNC41NTExNjkyIDE0LjY2NzYzMDksMTQuMTQ1Nzk0IDE0LjM4MTYyOTQsMTMuOTMxOTE0NSBMMTIuNTMxMzI1NywxMi41MzkyMTI3IEwyMS44ODEyNDk1LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTEuMjY1ODg1NCBMMTIuNTMxMzI1NywxMS4yNjU4ODU0IEwxNC4zODE2Mjk0LDkuODczMTgzNjQgQzE0LjYzNzc4NzIsOS43MTY1MDQ1MyAxNC43NDk3MDA2LDkuNDAwNjYwMTQgMTQuNjQ3NzM1MSw5LjExNzE0NTUzIEMxNC41NDgyNTY0LDguODMzNjMxNTYgMTQuMjYyMjU1LDguNjU5NTQzNTIgMTMuOTYzODE4OSw4LjY5OTMzNSBaIiBpZD0icGJfdGJsLXRibC0xLWFycm93IiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNS45MjkyMzAsIDExLjg5NDczNykgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE1LjkyOTIzMCwgLTExLjg5NDczNykgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-eval"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-test_units">3</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-pass">3<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-fail">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-w_upd"><span style="color: #AAAAAA;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC" headers="pb_tbl-e_upd"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-c_upd">--</td>
<td class="gt_row gt_center" style="height: 40px" headers="pb_tbl-extract_upd">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #EBBC14; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#EBBC14</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTItSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMi1jb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMi1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0yLWdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">age</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">26</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTItdW5jaGFuZ2VkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0yLXVuY2hhbmdlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC41MDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTItUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0icGJfdGJsLXRibC0yLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMy45NjM4MTg5LDguNjk5MzM1IEMxMy45MzY0NjIxLDguNzA0MzA5MjUgMTMuOTA5MTA1OSw4LjcxMTc2OTY4IDEzLjg4NDIzNTksOC43MTkyMzA3NCBDMTMuNzgyMjcwNCw4LjczNjYzOTY3IDEzLjY4Nzc2NTQsOC43NzY0MzExNSAxMy42MDU2OTU2LDguODM4NjA1MTggTDEwLjI0MzMxNTYsMTEuMzg1MjU5OCBDMTAuMDc2Njg4NiwxMS41MDQ2MzQzIDkuOTc3MjA5OTMsMTEuNjk4NjE4MSA5Ljk3NzIwOTkzLDExLjkwMjU0OTEgQzkuOTc3MjA5OTMsMTIuMTA2NDgwNyAxMC4wNzY2ODg2LDEyLjMwMDQ2MzkgMTAuMjQzMzE1NiwxMi40MTk4MzgzIEwxMy42MDU2OTU2LDE0Ljk2NjQ5MyBDMTMuODkxNjk3LDE1LjE4MDM3MjUgMTQuMjk3MDcyOSwxNS4xMjMxNzIxIDE0LjUxMDk1MTcsMTQuODM3MTcwNyBDMTQuNzI0ODMxMywxNC41NTExNjkyIDE0LjY2NzYzMDksMTQuMTQ1Nzk0IDE0LjM4MTYyOTQsMTMuOTMxOTE0NSBMMTIuNTMxMzI1NywxMi41MzkyMTI3IEwyMS44ODEyNDk1LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTEuMjY1ODg1NCBMMTIuNTMxMzI1NywxMS4yNjU4ODU0IEwxNC4zODE2Mjk0LDkuODczMTgzNjQgQzE0LjYzNzc4NzIsOS43MTY1MDQ1MyAxNC43NDk3MDA2LDkuNDAwNjYwMTQgMTQuNjQ3NzM1MSw5LjExNzE0NTUzIEMxNC41NDgyNTY0LDguODMzNjMxNTYgMTQuMjYyMjU1LDguNjU5NTQzNTIgMTMuOTYzODE4OSw4LjY5OTMzNSBaIiBpZD0icGJfdGJsLXRibC0yLWFycm93IiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNS45MjkyMzAsIDExLjg5NDczNykgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE1LjkyOTIzMCwgLTExLjg5NDczNykgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-eval"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-test_units">3</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-pass">2<br />
0.67</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-fail">1<br />
0.33</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-w_upd"><span style="color: #AAAAAA;">●</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC" headers="pb_tbl-e_upd"><span style="color: #EBBC14;">●</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-c_upd">--</td>
<td class="gt_row gt_center" style="height: 40px" headers="pb_tbl-extract_upd">CSV</td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">
<hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(schema_check)</span> <span style="color:#4CA64C;">✓</span> Schema validation <strong>passed</strong>.</p>
Schema Comparison


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_step_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }
<p>#pb_step_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
tr { background-color: transparent; }
#pb_step_tbl p { margin: 0; padding: 0; }
#pb_step_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
#pb_step_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
#pb_step_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
#pb_step_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
#pb_step_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
#pb_step_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
#pb_step_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
#pb_step_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
#pb_step_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
#pb_step_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
#pb_step_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
#pb_step_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
#pb_step_tbl .gt_spanner_row { border-bottom-style: hidden; }
#pb_step_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
#pb_step_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
#pb_step_tbl .gt_from_md> :first-child { margin-top: 0; }
#pb_step_tbl .gt_from_md> :last-child { margin-bottom: 0; }
#pb_step_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
#pb_step_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
#pb_step_tbl .gt_indent_1 { text-indent: 5px; }
#pb_step_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
#pb_step_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
#pb_step_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
#pb_step_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
#pb_step_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
#pb_step_tbl .gt_row_group_first td { border-top-width: 2px; }
#pb_step_tbl .gt_row_group_first th { border-top-width: 2px; }
#pb_step_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
#pb_step_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
#pb_step_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
#pb_step_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
#pb_step_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
#pb_step_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
#pb_step_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
#pb_step_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
#pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
#pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
#pb_step_tbl .gt_left { text-align: left; }
#pb_step_tbl .gt_center { text-align: center; }
#pb_step_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
#pb_step_tbl .gt_font_normal { font-weight: normal; }
#pb_step_tbl .gt_font_bold { font-weight: bold; }
#pb_step_tbl .gt_font_italic { font-style: italic; }
#pb_step_tbl .gt_super { font-size: 65%; }
#pb_step_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
#pb_step_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
#pb_step_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
#pb_step_tbl .gt_sourcenote { font-size: 12px; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
#pb_step_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
#pb_step_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }</p>
</style>
<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings gt_spanner_row">
<th colspan="3" id="pb_step_tbl-TARGET" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">TARGET</th>
<th colspan="5" id="pb_step_tbl-EXPECTED" class="gt_center gt_columns_top_border gt_column_spanner_outer" scope="colgroup">EXPECTED</th>
</tr>
<tr class="gt_col_headings">
<th id="pb_step_tbl-index_target" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-dtype_target" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-index_exp" class="gt_col_heading gt_columns_bottom_border gt_right" scope="col"></th>
<th id="pb_step_tbl-col_name_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">COLUMN</th>
<th id="pb_step_tbl-col_name_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
<th id="pb_step_tbl-dtype_exp" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col">DATA TYPE</th>
<th id="pb_step_tbl-dtype_exp_correct" class="gt_col_heading gt_columns_bottom_border gt_left" scope="col"></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">id</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">id</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">name</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">name</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">String</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px" headers="pb_step_tbl-TARGET pb_step_tbl-index_target">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-col_name_target">age</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-TARGET pb_step_tbl-dtype_target">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5" headers="pb_step_tbl-EXPECTED pb_step_tbl-index_exp">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp">age</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-col_name_exp_correct"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp">Int64</td>
<td class="gt_row gt_left" headers="pb_step_tbl-EXPECTED pb_step_tbl-dtype_exp_correct"><span style="color: #4CA64C;">✓</span></td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote">

Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('id', 'Int64'), ('name', 'String'), ('age', 'Int64')]</code>
</div></td>
</tr>
<tr class="gt_sourcenotes">
<td colspan="8" class="gt_sourcenote"> 

Schema Match Settings

COMPLETE

IN ORDER

COLUMN ≠ column

DTYPE ≠ dtype

float ≠ float64


</div></td>
</tr>
</tfoot>

</table>

</div></td>
</tr>
</tfoot>

</table>


This approach allows you to first validate the structure of your data and then check specific business rules or constraints.


# Best Practices

1.  Define schemas early: document and define expected data structures early in your data workflow.

2.  Choose the right creation method:

    - use `columns=<dict>` for readability with many columns
    - use `columns=<list of tuples>` for mixed name/type validation
    - use `kwargs` for small schemas with simple column names
    - use `tbl=` to extract schemas from reference datasets

3.  Be deliberate about strictness: choose validation parameters based on your specific needs:

    - strict validation (`complete=True`) for critical data interfaces
    - flexible validation (`complete=False`, `in_order=False`) for evolving datasets

4.  Reuse schemas: create schema definitions that can be reused across multiple validation contexts.

5.  Version control schemas: as your data evolves, maintain versions of your schemas to track changes.

6.  Extract schemas from reference data: when you have a 'golden' dataset that represents your ideal structure, use `Schema(tbl=reference_data)` to extract its schema.

7.  Consider type flexibility: use multiple types per column (`["Int64", "Float64"]`) when working with data from diverse sources.

8.  Combine with targeted validation: use schema validation for structural checks and add specific validation steps for business rules.


# Conclusion

Schema validation provides a powerful mechanism for ensuring your data adheres to expected structural requirements. It serves as an excellent first line of defense in your data validation strategy, verifying that the data you're working with has the expected shape before applying more detailed business rule validations.

The [Schema](../../reference/Schema.md#pointblank.Schema) class offers multiple ways to define schemas, from manual specification with dictionaries or keyword arguments to automatic extraction from reference tables. When combined with the flexible options of <a href="../../reference/Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>Validate.col_schema_match()</code></a>, you can implement validation approaches ranging from strict structural enforcement to more flexible evolution-friendly checks.

By understanding the different schema creation methods and validation options, you can efficiently validate the structure of your data tables and ensure they meet your requirements before processing.
