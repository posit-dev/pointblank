# Custom Validation with specially()


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
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-10-15|18:48:32</span>

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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+c3BlY2lhbGx5PC90aXRsZT4KICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMC1JY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InBiX3RibC10eXBlX3VwZC0wLXNwZWNpYWxseSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMjA2ODk3KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMC1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTAtc3RhciIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOC41MDAwMDAsIDguNTAwMDAwKSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yNSwwIEMyNC41ODc0NDg0LDAgMjQuMjE3NDUxNywwLjI1NDAwMjg0NyAyNC4wNjgzNTksMC42Mzg2NzE5IEwxNy45MDIzNDQsMTYuNTM1MTU2IEwwLjk0OTIxODc1LDE3LjQwMDM5MSBDMC41MzYxMjQ0MDksMTcuNDIxMzAxMyAwLjE3ODUyNDU4LDE3LjY5NDM1MjMgMC4wNDk1NDQ2Mzk1LDE4LjA4NzM1MTUgQy0wLjA3OTQzNTMwMTIsMTguNDgwMzUwNyAwLjA0NjgyMDQ1MiwxOC45MTIyMDAyIDAuMzY3MTg3NSwxOS4xNzM4MjggTDEzLjU2ODM1OSwyOS45NjY3OTcgTDkuMjMyNDIxOSw0Ni4zNDM3NSBDOS4xMjY0Njk2Myw0Ni43NDI4MDA5IDkuMjc2NjMwNTgsNDcuMTY1OTQzMyA5LjYxMDQyNjk4LDQ3LjQwODk0MDIgQzkuOTQ0MjIzMzgsNDcuNjUxOTM3IDEwLjM5MzAzNDUsNDcuNjY0ODM0IDEwLjc0MDIzNCw0Ny40NDE0MDYgTDI1LDM4LjI4OTA2MiBMMzkuMjU5NzY2LDQ3LjQ0MTQwNiBDMzkuNjA2OTY1NSw0Ny42NjQ4MzM5IDQwLjA1NTc3NjYsNDcuNjUxOTM2OSA0MC4zODk1NzI5LDQ3LjQwODk0MDEgQzQwLjcyMzM2OTMsNDcuMTY1OTQzMiA0MC44NzM1MzAyLDQ2Ljc0MjgwMDkgNDAuNzY3NTc4LDQ2LjM0Mzc1IEwzNi40MzE2NDEsMjkuOTY2Nzk3IEw0OS42MzI4MTIsMTkuMTczODI4IEM0OS45NTMxNzksMTguOTEyMjAwMiA1MC4wNzk0MzQ4LDE4LjQ4MDM1MDcgNDkuOTUwNDU0OSwxOC4wODczNTE2IEM0OS44MjE0NzUsMTcuNjk0MzUyNCA0OS40NjM4NzUzLDE3LjQyMTMwMTQgNDkuMDUwNzgxLDE3LjQwMDM5MSBMMzIuMDk3NjU2LDE2LjUzNTE1NiBMMjUuOTMxNjQxLDAuNjM4NjcxOSBDMjUuNzgyNTQ4MywwLjI1NDAwMjg0NyAyNS40MTI1NTE2LDAgMjUsMCBaIE0yNSwzLjc2MzY3MTkgTDMwLjQ2Njc5NywxNy44NjEzMjggQzMwLjYwOTY4OSwxOC4yMjkxNDE2IDMwLjk1NTQ5NjIsMTguNDc4NTUxNSAzMS4zNDk2MDksMTguNDk4MDQ3IEw0Ni4zNTkzNzUsMTkuMjY1NjI1IEwzNC42Njc5NjksMjguODI2MTcyIEMzNC4zNjQ2MDU0LDI5LjA3NDIxMTQgMzQuMjM0MDQ5MywyOS40NzY1Njc5IDM0LjMzMzk4NCwyOS44NTU0NjkgTDM4LjE3NTc4MSw0NC4zNjkxNDEgTDI1LjU0MTAxNiwzNi4yNTc4MTIgQzI1LjIxMTQ3ODksMzYuMDQ1ODUzNiAyNC43ODg1MjExLDM2LjA0NTg1MzYgMjQuNDU4OTg0LDM2LjI1NzgxMiBMMTEuODI0MjE5LDQ0LjM2OTE0MSBMMTUuNjY2MDE2LDI5Ljg1NTQ2OSBDMTUuNzY1OTUwNywyOS40NzY1Njc5IDE1LjYzNTM5NDYsMjkuMDc0MjExNCAxNS4zMzIwMzEsMjguODI2MTcyIEwzLjY0MDYyNSwxOS4yNjU2MjUgTDE4LjY1MDM5MSwxOC40OTgwNDcgQzE5LjA0NDUwMzgsMTguNDc4NTUxNSAxOS4zOTAzMTEsMTguMjI5MTQxNiAxOS41MzMyMDMsMTcuODYxMzI4IEwyNSwzLjc2MzY3MTkgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0wLVNoYXBlIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

specially()


<p>All values in column 'a' should be within 2 std devs of mean</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd"></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">EXPR</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTAtdW5jaGFuZ2VkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0wLXVuY2hhbmdlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC41MDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTAtUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0icGJfdGJsLXRibC0wLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMy45NjM4MTg5LDguNjk5MzM1IEMxMy45MzY0NjIxLDguNzA0MzA5MjUgMTMuOTA5MTA1OSw4LjcxMTc2OTY4IDEzLjg4NDIzNTksOC43MTkyMzA3NCBDMTMuNzgyMjcwNCw4LjczNjYzOTY3IDEzLjY4Nzc2NTQsOC43NzY0MzExNSAxMy42MDU2OTU2LDguODM4NjA1MTggTDEwLjI0MzMxNTYsMTEuMzg1MjU5OCBDMTAuMDc2Njg4NiwxMS41MDQ2MzQzIDkuOTc3MjA5OTMsMTEuNjk4NjE4MSA5Ljk3NzIwOTkzLDExLjkwMjU0OTEgQzkuOTc3MjA5OTMsMTIuMTA2NDgwNyAxMC4wNzY2ODg2LDEyLjMwMDQ2MzkgMTAuMjQzMzE1NiwxMi40MTk4MzgzIEwxMy42MDU2OTU2LDE0Ljk2NjQ5MyBDMTMuODkxNjk3LDE1LjE4MDM3MjUgMTQuMjk3MDcyOSwxNS4xMjMxNzIxIDE0LjUxMDk1MTcsMTQuODM3MTcwNyBDMTQuNzI0ODMxMywxNC41NTExNjkyIDE0LjY2NzYzMDksMTQuMTQ1Nzk0IDE0LjM4MTYyOTQsMTMuOTMxOTE0NSBMMTIuNTMxMzI1NywxMi41MzkyMTI3IEwyMS44ODEyNDk1LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTEuMjY1ODg1NCBMMTIuNTMxMzI1NywxMS4yNjU4ODU0IEwxNC4zODE2Mjk0LDkuODczMTgzNjQgQzE0LjYzNzc4NzIsOS43MTY1MDQ1MyAxNC43NDk3MDA2LDkuNDAwNjYwMTQgMTQuNjQ3NzM1MSw5LjExNzE0NTUzIEMxNC41NDgyNTY0LDguODMzNjMxNTYgMTQuMjYyMjU1LDguNjU5NTQzNTIgMTMuOTYzODE4OSw4LjY5OTMzNSBaIiBpZD0icGJfdGJsLXRibC0wLWFycm93IiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNS45MjkyMzAsIDExLjg5NDczNykgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE1LjkyOTIzMCwgLTExLjg5NDczNykgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-eval"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-test_units">2000</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-pass">1947<br />
0.97</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-fail">53<br />
0.03</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-w_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC" headers="pb_tbl-e_upd">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-c_upd">--</td>
<td class="gt_row gt_center" style="height: 40px" headers="pb_tbl-extract_upd">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+c3BlY2lhbGx5PC90aXRsZT4KICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMS1JY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InBiX3RibC10eXBlX3VwZC0xLXNwZWNpYWxseSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMjA2ODk3KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMS1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTEtc3RhciIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOC41MDAwMDAsIDguNTAwMDAwKSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yNSwwIEMyNC41ODc0NDg0LDAgMjQuMjE3NDUxNywwLjI1NDAwMjg0NyAyNC4wNjgzNTksMC42Mzg2NzE5IEwxNy45MDIzNDQsMTYuNTM1MTU2IEwwLjk0OTIxODc1LDE3LjQwMDM5MSBDMC41MzYxMjQ0MDksMTcuNDIxMzAxMyAwLjE3ODUyNDU4LDE3LjY5NDM1MjMgMC4wNDk1NDQ2Mzk1LDE4LjA4NzM1MTUgQy0wLjA3OTQzNTMwMTIsMTguNDgwMzUwNyAwLjA0NjgyMDQ1MiwxOC45MTIyMDAyIDAuMzY3MTg3NSwxOS4xNzM4MjggTDEzLjU2ODM1OSwyOS45NjY3OTcgTDkuMjMyNDIxOSw0Ni4zNDM3NSBDOS4xMjY0Njk2Myw0Ni43NDI4MDA5IDkuMjc2NjMwNTgsNDcuMTY1OTQzMyA5LjYxMDQyNjk4LDQ3LjQwODk0MDIgQzkuOTQ0MjIzMzgsNDcuNjUxOTM3IDEwLjM5MzAzNDUsNDcuNjY0ODM0IDEwLjc0MDIzNCw0Ny40NDE0MDYgTDI1LDM4LjI4OTA2MiBMMzkuMjU5NzY2LDQ3LjQ0MTQwNiBDMzkuNjA2OTY1NSw0Ny42NjQ4MzM5IDQwLjA1NTc3NjYsNDcuNjUxOTM2OSA0MC4zODk1NzI5LDQ3LjQwODk0MDEgQzQwLjcyMzM2OTMsNDcuMTY1OTQzMiA0MC44NzM1MzAyLDQ2Ljc0MjgwMDkgNDAuNzY3NTc4LDQ2LjM0Mzc1IEwzNi40MzE2NDEsMjkuOTY2Nzk3IEw0OS42MzI4MTIsMTkuMTczODI4IEM0OS45NTMxNzksMTguOTEyMjAwMiA1MC4wNzk0MzQ4LDE4LjQ4MDM1MDcgNDkuOTUwNDU0OSwxOC4wODczNTE2IEM0OS44MjE0NzUsMTcuNjk0MzUyNCA0OS40NjM4NzUzLDE3LjQyMTMwMTQgNDkuMDUwNzgxLDE3LjQwMDM5MSBMMzIuMDk3NjU2LDE2LjUzNTE1NiBMMjUuOTMxNjQxLDAuNjM4NjcxOSBDMjUuNzgyNTQ4MywwLjI1NDAwMjg0NyAyNS40MTI1NTE2LDAgMjUsMCBaIE0yNSwzLjc2MzY3MTkgTDMwLjQ2Njc5NywxNy44NjEzMjggQzMwLjYwOTY4OSwxOC4yMjkxNDE2IDMwLjk1NTQ5NjIsMTguNDc4NTUxNSAzMS4zNDk2MDksMTguNDk4MDQ3IEw0Ni4zNTkzNzUsMTkuMjY1NjI1IEwzNC42Njc5NjksMjguODI2MTcyIEMzNC4zNjQ2MDU0LDI5LjA3NDIxMTQgMzQuMjM0MDQ5MywyOS40NzY1Njc5IDM0LjMzMzk4NCwyOS44NTU0NjkgTDM4LjE3NTc4MSw0NC4zNjkxNDEgTDI1LjU0MTAxNiwzNi4yNTc4MTIgQzI1LjIxMTQ3ODksMzYuMDQ1ODUzNiAyNC43ODg1MjExLDM2LjA0NTg1MzYgMjQuNDU4OTg0LDM2LjI1NzgxMiBMMTEuODI0MjE5LDQ0LjM2OTE0MSBMMTUuNjY2MDE2LDI5Ljg1NTQ2OSBDMTUuNzY1OTUwNywyOS40NzY1Njc5IDE1LjYzNTM5NDYsMjkuMDc0MjExNCAxNS4zMzIwMzEsMjguODI2MTcyIEwzLjY0MDYyNSwxOS4yNjU2MjUgTDE4LjY1MDM5MSwxOC40OTgwNDcgQzE5LjA0NDUwMzgsMTguNDc4NTUxNSAxOS4zOTAzMTEsMTguMjI5MTQxNiAxOS41MzMyMDMsMTcuODYxMzI4IEwyNSwzLjc2MzY3MTkgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0xLVNoYXBlIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

specially()


<p>All values in column 'c' should be within 3 std devs of mean</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd"></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">EXPR</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTEtdW5jaGFuZ2VkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0xLXVuY2hhbmdlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC41MDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTEtUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0icGJfdGJsLXRibC0xLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMy45NjM4MTg5LDguNjk5MzM1IEMxMy45MzY0NjIxLDguNzA0MzA5MjUgMTMuOTA5MTA1OSw4LjcxMTc2OTY4IDEzLjg4NDIzNTksOC43MTkyMzA3NCBDMTMuNzgyMjcwNCw4LjczNjYzOTY3IDEzLjY4Nzc2NTQsOC43NzY0MzExNSAxMy42MDU2OTU2LDguODM4NjA1MTggTDEwLjI0MzMxNTYsMTEuMzg1MjU5OCBDMTAuMDc2Njg4NiwxMS41MDQ2MzQzIDkuOTc3MjA5OTMsMTEuNjk4NjE4MSA5Ljk3NzIwOTkzLDExLjkwMjU0OTEgQzkuOTc3MjA5OTMsMTIuMTA2NDgwNyAxMC4wNzY2ODg2LDEyLjMwMDQ2MzkgMTAuMjQzMzE1NiwxMi40MTk4MzgzIEwxMy42MDU2OTU2LDE0Ljk2NjQ5MyBDMTMuODkxNjk3LDE1LjE4MDM3MjUgMTQuMjk3MDcyOSwxNS4xMjMxNzIxIDE0LjUxMDk1MTcsMTQuODM3MTcwNyBDMTQuNzI0ODMxMywxNC41NTExNjkyIDE0LjY2NzYzMDksMTQuMTQ1Nzk0IDE0LjM4MTYyOTQsMTMuOTMxOTE0NSBMMTIuNTMxMzI1NywxMi41MzkyMTI3IEwyMS44ODEyNDk1LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTEuMjY1ODg1NCBMMTIuNTMxMzI1NywxMS4yNjU4ODU0IEwxNC4zODE2Mjk0LDkuODczMTgzNjQgQzE0LjYzNzc4NzIsOS43MTY1MDQ1MyAxNC43NDk3MDA2LDkuNDAwNjYwMTQgMTQuNjQ3NzM1MSw5LjExNzE0NTUzIEMxNC41NDgyNTY0LDguODMzNjMxNTYgMTQuMjYyMjU1LDguNjU5NTQzNTIgMTMuOTYzODE4OSw4LjY5OTMzNSBaIiBpZD0icGJfdGJsLXRibC0xLWFycm93IiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNS45MjkyMzAsIDExLjg5NDczNykgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE1LjkyOTIzMCwgLTExLjg5NDczNykgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-eval"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-test_units">2000</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-pass">2000<br />
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

<span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin-left: 10px; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-10-15 18:48:32 UTC</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">< 1 s</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 1px 5px -1px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-10-15 18:48:32 UTC</span>
</div></td>
</tr>
</tfoot>

</table>


``` python
import pointblank as pb
import polars as pl

def within_std_deviations(df, column, n_std=2):
    """Check if all values are within n standard deviations of the mean"""
    mean_val = df[column].mean()
    std_val = df[column].std()

    lower_bound = mean_val - (n_std * std_val)
    upper_bound = mean_val + (n_std * std_val)

    # Add a boolean column and return the modified DataFrame
    return df.with_columns(
        pl.col(column).is_between(lower_bound, upper_bound, closed="both").alias("validation_result")
    )

validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="game_revenue", tbl_type="polars")
    )
    .specially(
        expr=lambda df: within_std_deviations(df, column="session_duration", n_std=2),
        brief="All values in column 'a' should be within 2 std devs of mean"
    )
    .specially(
        expr=lambda df: within_std_deviations(df, column="session_duration", n_std=3),
        brief="All values in column 'c' should be within 3 std devs of mean"
    )
    .interrogate()
)

validation
```

Preview of Input Table


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
<th colspan="12" id="pb_preview_tbl-title" class="gt_heading gt_title gt_font_normal"><div>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">2,000</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">11</span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_preview_tbl-_row_num_" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"></th>
<th id="pb_preview_tbl-player_id" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

player_id

<em>String</em>

</div></th>
<th id="pb_preview_tbl-session_id" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

session_id

<em>String</em>

</div></th>
<th id="pb_preview_tbl-session_start" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

session_start

<em>Datetime</em>

</div></th>
<th id="pb_preview_tbl-time" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

time

<em>Datetime</em>

</div></th>
<th id="pb_preview_tbl-item_type" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

item_type

<em>String</em>

</div></th>
<th id="pb_preview_tbl-item_name" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

item_name

<em>String</em>

</div></th>
<th id="pb_preview_tbl-item_revenue" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

item_revenue

<em>Float64</em>

</div></th>
<th id="pb_preview_tbl-session_duration" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

session_duration

<em>Float64</em>

</div></th>
<th id="pb_preview_tbl-start_day" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

start_day

<em>Date</em>

</div></th>
<th id="pb_preview_tbl-acquisition" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

acquisition

<em>String</em>

</div></th>
<th id="pb_preview_tbl-country" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

country

<em>String</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">1</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-player_id">ECPANOIXLZHF896</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_id">ECPANOIXLZHF896-eol2j8bs</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_start">2015-01-01 01:31:03+00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-time">2015-01-01 01:31:27+00:00</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_type">iap</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_name">offer2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_revenue">8.99</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_duration">16.3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-start_day">2015-01-01</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-acquisition">google</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-country">Germany</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-player_id">ECPANOIXLZHF896</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_id">ECPANOIXLZHF896-eol2j8bs</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_start">2015-01-01 01:31:03+00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-time">2015-01-01 01:36:57+00:00</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_type">iap</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_name">gems3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_revenue">22.49</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_duration">16.3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-start_day">2015-01-01</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-acquisition">google</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-country">Germany</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-player_id">ECPANOIXLZHF896</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_id">ECPANOIXLZHF896-eol2j8bs</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_start">2015-01-01 01:31:03+00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-time">2015-01-01 01:37:45+00:00</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_type">iap</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_name">gold7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_revenue">107.99</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_duration">16.3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-start_day">2015-01-01</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-acquisition">google</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-country">Germany</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">4</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-player_id">ECPANOIXLZHF896</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_id">ECPANOIXLZHF896-eol2j8bs</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_start">2015-01-01 01:31:03+00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-time">2015-01-01 01:42:33+00:00</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_type">ad</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_name">ad_20sec</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_revenue">0.76</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_duration">16.3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-start_day">2015-01-01</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-acquisition">google</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-country">Germany</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">5</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-player_id">ECPANOIXLZHF896</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-session_id">ECPANOIXLZHF896-hdu9jkls</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-session_start">2015-01-01 11:50:02+00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-time">2015-01-01 11:55:20+00:00</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-item_type">ad</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-item_name">ad_5sec</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-item_revenue">0.03</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-session_duration">35.2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-start_day">2015-01-01</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-acquisition">google</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80" headers="pb_preview_tbl-country">Germany</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">1996</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-player_id">NAOJRDMCSEBI281</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_id">NAOJRDMCSEBI281-j2vs9ilp</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_start">2015-01-21 01:57:50+00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-time">2015-01-21 02:02:50+00:00</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_type">ad</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_name">ad_survey</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_revenue">1.332</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_duration">25.8</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-start_day">2015-01-11</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-acquisition">organic</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-country">Norway</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">1997</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-player_id">NAOJRDMCSEBI281</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_id">NAOJRDMCSEBI281-j2vs9ilp</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_start">2015-01-21 01:57:50+00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-time">2015-01-21 02:22:14+00:00</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_type">ad</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_name">ad_survey</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_revenue">1.35</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_duration">25.8</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-start_day">2015-01-11</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-acquisition">organic</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-country">Norway</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">1998</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-player_id">RMOSWHJGELCI675</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_id">RMOSWHJGELCI675-vbhcsmtr</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_start">2015-01-21 02:39:48+00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-time">2015-01-21 02:40:00+00:00</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_type">ad</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_name">ad_5sec</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_revenue">0.03</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_duration">8.4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-start_day">2015-01-10</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-acquisition">other_campaign</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-country">France</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">1999</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-player_id">RMOSWHJGELCI675</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_id">RMOSWHJGELCI675-vbhcsmtr</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_start">2015-01-21 02:39:48+00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-time">2015-01-21 02:47:12+00:00</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_type">iap</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_name">offer5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_revenue">26.09</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_duration">8.4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-start_day">2015-01-10</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-acquisition">other_campaign</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-country">France</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80" headers="pb_preview_tbl-_row_num_">2000</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-player_id">GJCXNTWEBIPQ369</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_id">GJCXNTWEBIPQ369-9elq67md</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_start">2015-01-21 03:59:23+00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-time">2015-01-21 04:06:29+00:00</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_type">ad</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_name">ad_5sec</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-item_revenue">0.12</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-session_duration">18.5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-start_day">2015-01-14</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-acquisition">organic</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E" headers="pb_preview_tbl-country">United States</td>
</tr>
</tbody>
</table>
