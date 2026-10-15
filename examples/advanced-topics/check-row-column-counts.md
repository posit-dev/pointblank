# Verifying Row and Column Counts


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
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-10-15|18:47:50</span>

<span style="background-color: #000000; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #000000; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">DuckDB</span>

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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2NvdW50X21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMC1JY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InBiX3RibC10eXBlX3VwZC0wLWNvbF9jb3VudF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMjc1ODYyKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMC1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuNjM1Mjc2MiwxMy4wMTQ2Njk0IEw0NC42MzUyNzYyLDU0LjAxNDU4ODYgTDQyLjUzNTcxNTQsNTQuMDE0MjQ5MyBDNDIuNDE3MDgyMiw1NC4wMTEyMTc0IDQyLjEwMDgyMTksNTMuOTk1ODkyMiA0MS44ODE4NDg5LDUzLjc3OTE0ODEgQzQxLjc1NjYwMTUsNTMuNjU1MTc1NiA0MS42ODI4OTUzLDUzLjQ2MDMyMiA0MS42ODI4OTUzLDUzLjE4MTMzNiBMNDEuNjgyODk1Myw1My4xODEzMzYgTDQxLjY4Mjg5NTMsMTMuODQ4MDAyNyBDNDEuNjgyODk1MywxMy41NjUxNzMxIDQxLjc1ODkwOTQsMTMuMzY4ODI4MiA0MS44ODcwMzIxLDEzLjI0NTEyMzUgQzQyLjEyNjgxMiwxMy4wMTM2MTIgNDIuNDc2NjY1NCwxMy4wMTQ2Njk0IDQyLjU2Mzg0NzYsMTMuMDE0NjY5NCBMNDQuNjM1Mjc2MiwxMy4wMTQ2Njk0IFogTTUwLjYzMDYwMDYsMTMuMDE0NjY5NCBMNTAuNjMwNjAwNiw1NC4wMTQ2Njk0IEw0Ny43MzUxODk5LDU0LjAxNDY2OTQgTDQ3LjczNTE4OTksMTMuMDE0NjY5NCBMNTAuNjMwNjAwNiwxMy4wMTQ2Njk0IFogTTUzLjczMDUxNDMsMTMuMDE0NzUwMiBMNTUuODMwMDA4OCwxMy4wMTUwODk0IEM1NS45NDQwNTY2LDEzLjAxODA2NjcgNTYuMjM3NDgxMywxMy4wMzQxNTM0IDU2LjQ1NjAxNjUsMTMuMjI0MjgzOCBDNTYuNTcxNDgyOCwxMy4zMjQ3NDIxIDU2LjY1MjY2MTIsMTMuNDgwNzA3MiA1Ni42NzU4Mzg5LDEzLjcwNDE3MDcgTDU2LjY3NTgzODksMTMuNzA0MTcwNyBMNTYuNjgyODk1Myw1My4xODEzMzYgQzU2LjY4Mjg5NTMsNTMuNDYwMzIyIDU2LjYwOTE4OSw1My42NTUxNzU2IDU2LjQ4Mzk0MTYsNTMuNzc5MTQ4MSBDNTYuMjY0OTY4Nyw1My45OTU4OTIyIDU1Ljk0ODcwODMsNTQuMDExMjE3NCA1NS44MzAwNzUyLDU0LjAxNDI0OTMgTDU1LjgzMDA3NTIsNTQuMDE0MjQ5MyBMNTMuNzMwNTE0Myw1NC4wMTQ1ODg2IEw1My43MzA1MTQzLDEzLjAxNDc1MDIgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0wLXJvd3NfdHdvIiBzdHJva2U9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDkuMTgyODk1LCAzMy41MTQ2NjkpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC00OS4xODI4OTUsIC0zMy41MTQ2NjkpICIgLz4KICAgICAgICAgICAgPGcgaWQ9InBiX3RibC10eXBlX3VwZC0wLXZlcnRpY2FsX2VxdWFsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzNC4xMzUxOTUsIDMzLjcyNjkwMykgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuMTM1MTk1LCAtMzMuNzI2OTAzKSB0cmFuc2xhdGUoMzAuNjM1MTk1LCAyOS4yMjY5MDMpIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0icGJfdGJsLXR5cGVfdXBkLTAtTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0icGJfdGJsLXR5cGVfdXBkLTAtTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS41OTMxODYzLDEyLjUxNDY2OTQgQzExLjM4MzY2MjUsMTIuNTE0NjY5NCAxMC4yMTIyMzQsMTIuNTY0NjY5NCAxMC4yMTIyMzQsMTMuODQ4MDAyNyBMMTAuMjEyMjM0LDUzLjE4MTMzNiBDMTAuMjEyMjM0LDU0LjQ2NDY2OTQgMTEuMzgzNjYyNSw1NC41MTQ2Njk0IDExLjU5MzE4NjMsNTQuNTE0NjY5NCBMMTQuMTY0NjE0OSw1NC41MTQ2Njk0IEwxNC4xNjQ2MTQ5LDEyLjUxNDY2OTQgTDExLjU5MzE4NjMsMTIuNTE0NjY5NCBaIE0yMC4xNzIxNzcxLDEyLjUxNDY2OTQgTDIwLjE3MjE3NzEsNTQuNTE0NjY5NCBMMTYuMjUyMjkwOCw1NC41MTQ2Njk0IEwxNi4yNTIyOTA4LDU0LjUxNDY2OTQgTDE2LjI1MjI5MDgsMTIuNTE0NjY5NCBMMTYuMjUyMjkwOCwxMi41MTQ2Njk0IEwyMC4xNzIxNzcxLDEyLjUxNDY2OTQgWiBNMjQuODY1NjE0OSwxMi41MTUwOTA0IEMyNS4xNDQ4Nzg2LDEyLjUyMTc2MyAyNi4yMTIyMzQsMTIuNjIzMDAyNyAyNi4yMTIyMzQsMTMuODQ4MDAyNyBMMjYuMjEyMjM0LDEzLjg0ODAwMjcgTDI2LjIxMjIzNCw1My4xODEzMzYgQzI2LjIxMjIzNCw1NC40NjQ2Njk0IDI1LjA0MDgwNTQsNTQuNTE0NjY5NCAyNC44MzEyODE2LDU0LjUxNDY2OTQgTDI0LjgzMTI4MTYsNTQuNTE0NjY5NCBMMjIuMjU5ODUzLDU0LjUxNDY2OTQgTDIyLjI1OTg1MywxMi41MTQ2Njk0IFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMC1yb3dzX29uZSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxOC4yMTIyMzQsIDMzLjUxNDY2OSkgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE4LjIxMjIzNCwgLTMzLjUxNDY2OSkgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_count_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">11</td>
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
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+cm93X2NvdW50X21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMS1JY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InBiX3RibC10eXBlX3VwZC0xLXJvd19jb3VudF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNzkzMTAzKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMS1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMjkuMTY0NjE0OSwyOC40ODUzMzA2IEwyOS4xNjQ2MTQ5LDY5LjQ4NTMzMDYgTDI3LjA1NDkwODQsNjkuNDg1MDQ5MyBDMjYuOTI5NjE3OSw2OS40ODI2OTQ0IDI2LjYyOTUwODUsNjkuNDYwNjY0NSAyNi40MTYzNzA4LDY5LjI1NDg3NjUgQzI2LjI4ODI0ODEsNjkuMTMxMTcxOCAyNi4yMTIyMzQsNjguOTM0ODI2OSAyNi4yMTIyMzQsNjguNjUxOTk3MyBMMjYuMjEyMjM0LDY4LjY1MTk5NzMgTDI2LjIxMjIzNCwyOS4zMTg2NjQgQzI2LjIxMjIzNCwyOS4wMzU4MzQ0IDI2LjI4ODI0ODEsMjguODM5NDg5NSAyNi40MTYzNzA4LDI4LjcxNTc4NDggQzI2LjY1NjE1MDcsMjguNDg0MjczMyAyNy4wMDYwMDQxLDI4LjQ4NTMzMDYgMjcuMDkzMTg2MywyOC40ODUzMzA2IEwyOS4xNjQ2MTQ5LDI4LjQ4NTMzMDYgWiBNMzUuMTU5OTM5MywyOC40ODUzMzA2IEwzNS4xNTk5MzkzLDY5LjQ4NTMzMDYgTDMyLjI2NDUyODYsNjkuNDg1MzMwNiBMMzIuMjY0NTI4NiwyOC40ODUzMzA2IEwzNS4xNTk5MzkzLDI4LjQ4NTMzMDYgWiBNNDAuMzU5MzQ3NSwyOC40ODU3NTA3IEM0MC40NzMzOTUzLDI4LjQ4ODcyOCA0MC43NjY4MiwyOC41MDQ4MTQ2IDQwLjk4NTM1NTIsMjguNjk0OTQ1MSBDNDEuMTAwODIxNSwyOC43OTU0MDM0IDQxLjE4MTk5OTksMjguOTUxMzY4NCA0MS4yMDUxNzc2LDI5LjE3NDgzMiBMNDEuMjA1MTc3NiwyOS4xNzQ4MzIgTDQxLjIxMjIzNCw2OC42NTE5OTczIEM0MS4yMTIyMzQsNjguOTMwOTgzMyA0MS4xMzg1Mjc3LDY5LjEyNTgzNjkgNDEuMDEzMjgwMyw2OS4yNDk4MDk0IEM0MC43OTQzMDc0LDY5LjQ2NjU1MzUgNDAuNDc4MDQ3LDY5LjQ4MTg3ODcgNDAuMzU5NDEzOSw2OS40ODQ5MTA2IEw0MC4zNTk0MTM5LDY5LjQ4NDkxMDYgTDM4LjI1OTg1Myw2OS40ODUyNDk4IEwzOC4yNTk4NTMsMjguNDg1NDExNCBaIiBpZD0icGJfdGJsLXR5cGVfdXBkLTEtcm93c190d28iIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzMy43MTIyMzQsIDQ4Ljk4NTMzMSkgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMzMuNzEyMjM0LCAtNDguOTg1MzMxKSAiIC8+CiAgICAgICAgICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMS12ZXJ0aWNhbF9lcXVhbCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzAuMDAwMDAwLCAyOS4zODA1NzApIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0icGJfdGJsLXR5cGVfdXBkLTEtTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0icGJfdGJsLXR5cGVfdXBkLTEtTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yNy4wOTMxODYzLC0yLjk4NTMzMDY1IEMyNi44ODM2NjI1LC0yLjk4NTMzMDY1IDI1LjcxMjIzNCwtMi45MzUzMzA2NSAyNS43MTIyMzQsLTEuNjUxOTk3MzEgTDI1LjcxMjIzNCwzNy42ODEzMzYgQzI1LjcxMjIzNCwzOC45NjQ2Njk0IDI2Ljg4MzY2MjUsMzkuMDE0NjY5NCAyNy4wOTMxODYzLDM5LjAxNDY2OTQgTDI5LjY2NDYxNDksMzkuMDE0NjY5NCBMMjkuNjY0NjE0OSwtMi45ODUzMzA2NSBMMjcuMDkzMTg2MywtMi45ODUzMzA2NSBaIE0zNS42NzIxNzcxLC0yLjk4NTMzMDY1IEwzNS42NzIxNzcxLDM5LjAxNDY2OTQgTDMxLjc1MjI5MDgsMzkuMDE0NjY5NCBMMzEuNzUyMjkwOCwzOS4wMTQ2Njk0IEwzMS43NTIyOTA4LC0yLjk4NTMzMDY1IEwzMS43NTIyOTA4LC0yLjk4NTMzMDY1IEwzNS42NzIxNzcxLC0yLjk4NTMzMDY1IFogTTQwLjM2NTYxNDksLTIuOTg0OTA5NiBDNDAuNjQ0ODc4NiwtMi45NzgyMzY5OCA0MS43MTIyMzQsLTIuODc2OTk3MzEgNDEuNzEyMjM0LC0xLjY1MTk5NzMxIEw0MS43MTIyMzQsLTEuNjUxOTk3MzEgTDQxLjcxMjIzNCwzNy42ODEzMzYgQzQxLjcxMjIzNCwzOC45NjQ2Njk0IDQwLjU0MDgwNTQsMzkuMDE0NjY5NCA0MC4zMzEyODE2LDM5LjAxNDY2OTQgTDQwLjMzMTI4MTYsMzkuMDE0NjY5NCBMMzcuNzU5ODUzLDM5LjAxNDY2OTQgTDM3Ljc1OTg1MywtMi45ODUzMzA2NSBaIiBpZD0icGJfdGJsLXR5cGVfdXBkLTEtcm93c19vbmUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuNzEyMjM0LCAxOC4wMTQ2NjkpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTMzLjcxMjIzNCwgLTE4LjAxNDY2OSkgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

row_count_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">2000</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTEtdW5jaGFuZ2VkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0xLXVuY2hhbmdlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC41MDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTEtUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0icGJfdGJsLXRibC0xLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMy45NjM4MTg5LDguNjk5MzM1IEMxMy45MzY0NjIxLDguNzA0MzA5MjUgMTMuOTA5MTA1OSw4LjcxMTc2OTY4IDEzLjg4NDIzNTksOC43MTkyMzA3NCBDMTMuNzgyMjcwNCw4LjczNjYzOTY3IDEzLjY4Nzc2NTQsOC43NzY0MzExNSAxMy42MDU2OTU2LDguODM4NjA1MTggTDEwLjI0MzMxNTYsMTEuMzg1MjU5OCBDMTAuMDc2Njg4NiwxMS41MDQ2MzQzIDkuOTc3MjA5OTMsMTEuNjk4NjE4MSA5Ljk3NzIwOTkzLDExLjkwMjU0OTEgQzkuOTc3MjA5OTMsMTIuMTA2NDgwNyAxMC4wNzY2ODg2LDEyLjMwMDQ2MzkgMTAuMjQzMzE1NiwxMi40MTk4MzgzIEwxMy42MDU2OTU2LDE0Ljk2NjQ5MyBDMTMuODkxNjk3LDE1LjE4MDM3MjUgMTQuMjk3MDcyOSwxNS4xMjMxNzIxIDE0LjUxMDk1MTcsMTQuODM3MTcwNyBDMTQuNzI0ODMxMywxNC41NTExNjkyIDE0LjY2NzYzMDksMTQuMTQ1Nzk0IDE0LjM4MTYyOTQsMTMuOTMxOTE0NSBMMTIuNTMxMzI1NywxMi41MzkyMTI3IEwyMS44ODEyNDk1LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTEuMjY1ODg1NCBMMTIuNTMxMzI1NywxMS4yNjU4ODU0IEwxNC4zODE2Mjk0LDkuODczMTgzNjQgQzE0LjYzNzc4NzIsOS43MTY1MDQ1MyAxNC43NDk3MDA2LDkuNDAwNjYwMTQgMTQuNjQ3NzM1MSw5LjExNzE0NTUzIEMxNC41NDgyNTY0LDguODMzNjMxNTYgMTQuMjYyMjU1LDguNjU5NTQzNTIgMTMuOTYzODE4OSw4LjY5OTMzNSBaIiBpZD0icGJfdGJsLXRibC0xLWFycm93IiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNS45MjkyMzAsIDExLjg5NDczNykgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE1LjkyOTIzMCwgLTExLjg5NDczNykgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" /></td>
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
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+cm93X2NvdW50X21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMi1JY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InBiX3RibC10eXBlX3VwZC0yLXJvd19jb3VudF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNzkzMTAzKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMi1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMjkuMTY0NjE0OSwyOC40ODUzMzA2IEwyOS4xNjQ2MTQ5LDY5LjQ4NTMzMDYgTDI3LjA1NDkwODQsNjkuNDg1MDQ5MyBDMjYuOTI5NjE3OSw2OS40ODI2OTQ0IDI2LjYyOTUwODUsNjkuNDYwNjY0NSAyNi40MTYzNzA4LDY5LjI1NDg3NjUgQzI2LjI4ODI0ODEsNjkuMTMxMTcxOCAyNi4yMTIyMzQsNjguOTM0ODI2OSAyNi4yMTIyMzQsNjguNjUxOTk3MyBMMjYuMjEyMjM0LDY4LjY1MTk5NzMgTDI2LjIxMjIzNCwyOS4zMTg2NjQgQzI2LjIxMjIzNCwyOS4wMzU4MzQ0IDI2LjI4ODI0ODEsMjguODM5NDg5NSAyNi40MTYzNzA4LDI4LjcxNTc4NDggQzI2LjY1NjE1MDcsMjguNDg0MjczMyAyNy4wMDYwMDQxLDI4LjQ4NTMzMDYgMjcuMDkzMTg2MywyOC40ODUzMzA2IEwyOS4xNjQ2MTQ5LDI4LjQ4NTMzMDYgWiBNMzUuMTU5OTM5MywyOC40ODUzMzA2IEwzNS4xNTk5MzkzLDY5LjQ4NTMzMDYgTDMyLjI2NDUyODYsNjkuNDg1MzMwNiBMMzIuMjY0NTI4NiwyOC40ODUzMzA2IEwzNS4xNTk5MzkzLDI4LjQ4NTMzMDYgWiBNNDAuMzU5MzQ3NSwyOC40ODU3NTA3IEM0MC40NzMzOTUzLDI4LjQ4ODcyOCA0MC43NjY4MiwyOC41MDQ4MTQ2IDQwLjk4NTM1NTIsMjguNjk0OTQ1MSBDNDEuMTAwODIxNSwyOC43OTU0MDM0IDQxLjE4MTk5OTksMjguOTUxMzY4NCA0MS4yMDUxNzc2LDI5LjE3NDgzMiBMNDEuMjA1MTc3NiwyOS4xNzQ4MzIgTDQxLjIxMjIzNCw2OC42NTE5OTczIEM0MS4yMTIyMzQsNjguOTMwOTgzMyA0MS4xMzg1Mjc3LDY5LjEyNTgzNjkgNDEuMDEzMjgwMyw2OS4yNDk4MDk0IEM0MC43OTQzMDc0LDY5LjQ2NjU1MzUgNDAuNDc4MDQ3LDY5LjQ4MTg3ODcgNDAuMzU5NDEzOSw2OS40ODQ5MTA2IEw0MC4zNTk0MTM5LDY5LjQ4NDkxMDYgTDM4LjI1OTg1Myw2OS40ODUyNDk4IEwzOC4yNTk4NTMsMjguNDg1NDExNCBaIiBpZD0icGJfdGJsLXR5cGVfdXBkLTItcm93c190d28iIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzMy43MTIyMzQsIDQ4Ljk4NTMzMSkgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMzMuNzEyMjM0LCAtNDguOTg1MzMxKSAiIC8+CiAgICAgICAgICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMi12ZXJ0aWNhbF9lcXVhbCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzAuMDAwMDAwLCAyOS4zODA1NzApIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0icGJfdGJsLXR5cGVfdXBkLTItTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0icGJfdGJsLXR5cGVfdXBkLTItTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yNy4wOTMxODYzLC0yLjk4NTMzMDY1IEMyNi44ODM2NjI1LC0yLjk4NTMzMDY1IDI1LjcxMjIzNCwtMi45MzUzMzA2NSAyNS43MTIyMzQsLTEuNjUxOTk3MzEgTDI1LjcxMjIzNCwzNy42ODEzMzYgQzI1LjcxMjIzNCwzOC45NjQ2Njk0IDI2Ljg4MzY2MjUsMzkuMDE0NjY5NCAyNy4wOTMxODYzLDM5LjAxNDY2OTQgTDI5LjY2NDYxNDksMzkuMDE0NjY5NCBMMjkuNjY0NjE0OSwtMi45ODUzMzA2NSBMMjcuMDkzMTg2MywtMi45ODUzMzA2NSBaIE0zNS42NzIxNzcxLC0yLjk4NTMzMDY1IEwzNS42NzIxNzcxLDM5LjAxNDY2OTQgTDMxLjc1MjI5MDgsMzkuMDE0NjY5NCBMMzEuNzUyMjkwOCwzOS4wMTQ2Njk0IEwzMS43NTIyOTA4LC0yLjk4NTMzMDY1IEwzMS43NTIyOTA4LC0yLjk4NTMzMDY1IEwzNS42NzIxNzcxLC0yLjk4NTMzMDY1IFogTTQwLjM2NTYxNDksLTIuOTg0OTA5NiBDNDAuNjQ0ODc4NiwtMi45NzgyMzY5OCA0MS43MTIyMzQsLTIuODc2OTk3MzEgNDEuNzEyMjM0LC0xLjY1MTk5NzMxIEw0MS43MTIyMzQsLTEuNjUxOTk3MzEgTDQxLjcxMjIzNCwzNy42ODEzMzYgQzQxLjcxMjIzNCwzOC45NjQ2Njk0IDQwLjU0MDgwNTQsMzkuMDE0NjY5NCA0MC4zMzEyODE2LDM5LjAxNDY2OTQgTDQwLjMzMTI4MTYsMzkuMDE0NjY5NCBMMzcuNzU5ODUzLDM5LjAxNDY2OTQgTDM3Ljc1OTg1MywtMi45ODUzMzA2NSBaIiBpZD0icGJfdGJsLXR5cGVfdXBkLTItcm93c19vbmUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuNzEyMjM0LCAxOC4wMTQ2NjkpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTMzLjcxMjIzNCwgLTE4LjAxNDY2OSkgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

row_count_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">≠ 0</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTItdW5jaGFuZ2VkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0yLXVuY2hhbmdlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC41MDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTItUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0icGJfdGJsLXRibC0yLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMy45NjM4MTg5LDguNjk5MzM1IEMxMy45MzY0NjIxLDguNzA0MzA5MjUgMTMuOTA5MTA1OSw4LjcxMTc2OTY4IDEzLjg4NDIzNTksOC43MTkyMzA3NCBDMTMuNzgyMjcwNCw4LjczNjYzOTY3IDEzLjY4Nzc2NTQsOC43NzY0MzExNSAxMy42MDU2OTU2LDguODM4NjA1MTggTDEwLjI0MzMxNTYsMTEuMzg1MjU5OCBDMTAuMDc2Njg4NiwxMS41MDQ2MzQzIDkuOTc3MjA5OTMsMTEuNjk4NjE4MSA5Ljk3NzIwOTkzLDExLjkwMjU0OTEgQzkuOTc3MjA5OTMsMTIuMTA2NDgwNyAxMC4wNzY2ODg2LDEyLjMwMDQ2MzkgMTAuMjQzMzE1NiwxMi40MTk4MzgzIEwxMy42MDU2OTU2LDE0Ljk2NjQ5MyBDMTMuODkxNjk3LDE1LjE4MDM3MjUgMTQuMjk3MDcyOSwxNS4xMjMxNzIxIDE0LjUxMDk1MTcsMTQuODM3MTcwNyBDMTQuNzI0ODMxMywxNC41NTExNjkyIDE0LjY2NzYzMDksMTQuMTQ1Nzk0IDE0LjM4MTYyOTQsMTMuOTMxOTE0NSBMMTIuNTMxMzI1NywxMi41MzkyMTI3IEwyMS44ODEyNDk1LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTEuMjY1ODg1NCBMMTIuNTMxMzI1NywxMS4yNjU4ODU0IEwxNC4zODE2Mjk0LDkuODczMTgzNjQgQzE0LjYzNzc4NzIsOS43MTY1MDQ1MyAxNC43NDk3MDA2LDkuNDAwNjYwMTQgMTQuNjQ3NzM1MSw5LjExNzE0NTUzIEMxNC41NDgyNTY0LDguODMzNjMxNTYgMTQuMjYyMjU1LDguNjU5NTQzNTIgMTMuOTYzODE4OSw4LjY5OTMzNSBaIiBpZD0icGJfdGJsLXRibC0yLWFycm93IiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNS45MjkyMzAsIDExLjg5NDczNykgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE1LjkyOTIzMCwgLTExLjg5NDczNykgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" /></td>
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
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">4</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2NvdW50X21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMy1JY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InBiX3RibC10eXBlX3VwZC0zLWNvbF9jb3VudF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMjc1ODYyKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMy1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuNjM1Mjc2MiwxMy4wMTQ2Njk0IEw0NC42MzUyNzYyLDU0LjAxNDU4ODYgTDQyLjUzNTcxNTQsNTQuMDE0MjQ5MyBDNDIuNDE3MDgyMiw1NC4wMTEyMTc0IDQyLjEwMDgyMTksNTMuOTk1ODkyMiA0MS44ODE4NDg5LDUzLjc3OTE0ODEgQzQxLjc1NjYwMTUsNTMuNjU1MTc1NiA0MS42ODI4OTUzLDUzLjQ2MDMyMiA0MS42ODI4OTUzLDUzLjE4MTMzNiBMNDEuNjgyODk1Myw1My4xODEzMzYgTDQxLjY4Mjg5NTMsMTMuODQ4MDAyNyBDNDEuNjgyODk1MywxMy41NjUxNzMxIDQxLjc1ODkwOTQsMTMuMzY4ODI4MiA0MS44ODcwMzIxLDEzLjI0NTEyMzUgQzQyLjEyNjgxMiwxMy4wMTM2MTIgNDIuNDc2NjY1NCwxMy4wMTQ2Njk0IDQyLjU2Mzg0NzYsMTMuMDE0NjY5NCBMNDQuNjM1Mjc2MiwxMy4wMTQ2Njk0IFogTTUwLjYzMDYwMDYsMTMuMDE0NjY5NCBMNTAuNjMwNjAwNiw1NC4wMTQ2Njk0IEw0Ny43MzUxODk5LDU0LjAxNDY2OTQgTDQ3LjczNTE4OTksMTMuMDE0NjY5NCBMNTAuNjMwNjAwNiwxMy4wMTQ2Njk0IFogTTUzLjczMDUxNDMsMTMuMDE0NzUwMiBMNTUuODMwMDA4OCwxMy4wMTUwODk0IEM1NS45NDQwNTY2LDEzLjAxODA2NjcgNTYuMjM3NDgxMywxMy4wMzQxNTM0IDU2LjQ1NjAxNjUsMTMuMjI0MjgzOCBDNTYuNTcxNDgyOCwxMy4zMjQ3NDIxIDU2LjY1MjY2MTIsMTMuNDgwNzA3MiA1Ni42NzU4Mzg5LDEzLjcwNDE3MDcgTDU2LjY3NTgzODksMTMuNzA0MTcwNyBMNTYuNjgyODk1Myw1My4xODEzMzYgQzU2LjY4Mjg5NTMsNTMuNDYwMzIyIDU2LjYwOTE4OSw1My42NTUxNzU2IDU2LjQ4Mzk0MTYsNTMuNzc5MTQ4MSBDNTYuMjY0OTY4Nyw1My45OTU4OTIyIDU1Ljk0ODcwODMsNTQuMDExMjE3NCA1NS44MzAwNzUyLDU0LjAxNDI0OTMgTDU1LjgzMDA3NTIsNTQuMDE0MjQ5MyBMNTMuNzMwNTE0Myw1NC4wMTQ1ODg2IEw1My43MzA1MTQzLDEzLjAxNDc1MDIgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0zLXJvd3NfdHdvIiBzdHJva2U9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDkuMTgyODk1LCAzMy41MTQ2NjkpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC00OS4xODI4OTUsIC0zMy41MTQ2NjkpICIgLz4KICAgICAgICAgICAgPGcgaWQ9InBiX3RibC10eXBlX3VwZC0zLXZlcnRpY2FsX2VxdWFsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzNC4xMzUxOTUsIDMzLjcyNjkwMykgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuMTM1MTk1LCAtMzMuNzI2OTAzKSB0cmFuc2xhdGUoMzAuNjM1MTk1LCAyOS4yMjY5MDMpIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0icGJfdGJsLXR5cGVfdXBkLTMtTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0icGJfdGJsLXR5cGVfdXBkLTMtTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS41OTMxODYzLDEyLjUxNDY2OTQgQzExLjM4MzY2MjUsMTIuNTE0NjY5NCAxMC4yMTIyMzQsMTIuNTY0NjY5NCAxMC4yMTIyMzQsMTMuODQ4MDAyNyBMMTAuMjEyMjM0LDUzLjE4MTMzNiBDMTAuMjEyMjM0LDU0LjQ2NDY2OTQgMTEuMzgzNjYyNSw1NC41MTQ2Njk0IDExLjU5MzE4NjMsNTQuNTE0NjY5NCBMMTQuMTY0NjE0OSw1NC41MTQ2Njk0IEwxNC4xNjQ2MTQ5LDEyLjUxNDY2OTQgTDExLjU5MzE4NjMsMTIuNTE0NjY5NCBaIE0yMC4xNzIxNzcxLDEyLjUxNDY2OTQgTDIwLjE3MjE3NzEsNTQuNTE0NjY5NCBMMTYuMjUyMjkwOCw1NC41MTQ2Njk0IEwxNi4yNTIyOTA4LDU0LjUxNDY2OTQgTDE2LjI1MjI5MDgsMTIuNTE0NjY5NCBMMTYuMjUyMjkwOCwxMi41MTQ2Njk0IEwyMC4xNzIxNzcxLDEyLjUxNDY2OTQgWiBNMjQuODY1NjE0OSwxMi41MTUwOTA0IEMyNS4xNDQ4Nzg2LDEyLjUyMTc2MyAyNi4yMTIyMzQsMTIuNjIzMDAyNyAyNi4yMTIyMzQsMTMuODQ4MDAyNyBMMjYuMjEyMjM0LDEzLjg0ODAwMjcgTDI2LjIxMjIzNCw1My4xODEzMzYgQzI2LjIxMjIzNCw1NC40NjQ2Njk0IDI1LjA0MDgwNTQsNTQuNTE0NjY5NCAyNC44MzEyODE2LDU0LjUxNDY2OTQgTDI0LjgzMTI4MTYsNTQuNTE0NjY5NCBMMjIuMjU5ODUzLDU0LjUxNDY2OTQgTDIyLjI1OTg1MywxMi41MTQ2Njk0IFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMy1yb3dzX29uZSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxOC4yMTIyMzQsIDMzLjUxNDY2OSkgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE4LjIxMjIzNCwgLTMzLjUxNDY2OSkgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_count_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">11</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-tbl"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJwYl90YmwtdGJsLTMtdW5jaGFuZ2VkIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0icGJfdGJsLXRibC0zLXVuY2hhbmdlZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC41MDAwMDAsIDAuNTcwMTQ3KSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJwYl90YmwtdGJsLTMtUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0icGJfdGJsLXRibC0zLVNoYXBlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMy45NjM4MTg5LDguNjk5MzM1IEMxMy45MzY0NjIxLDguNzA0MzA5MjUgMTMuOTA5MTA1OSw4LjcxMTc2OTY4IDEzLjg4NDIzNTksOC43MTkyMzA3NCBDMTMuNzgyMjcwNCw4LjczNjYzOTY3IDEzLjY4Nzc2NTQsOC43NzY0MzExNSAxMy42MDU2OTU2LDguODM4NjA1MTggTDEwLjI0MzMxNTYsMTEuMzg1MjU5OCBDMTAuMDc2Njg4NiwxMS41MDQ2MzQzIDkuOTc3MjA5OTMsMTEuNjk4NjE4MSA5Ljk3NzIwOTkzLDExLjkwMjU0OTEgQzkuOTc3MjA5OTMsMTIuMTA2NDgwNyAxMC4wNzY2ODg2LDEyLjMwMDQ2MzkgMTAuMjQzMzE1NiwxMi40MTk4MzgzIEwxMy42MDU2OTU2LDE0Ljk2NjQ5MyBDMTMuODkxNjk3LDE1LjE4MDM3MjUgMTQuMjk3MDcyOSwxNS4xMjMxNzIxIDE0LjUxMDk1MTcsMTQuODM3MTcwNyBDMTQuNzI0ODMxMywxNC41NTExNjkyIDE0LjY2NzYzMDksMTQuMTQ1Nzk0IDE0LjM4MTYyOTQsMTMuOTMxOTE0NSBMMTIuNTMxMzI1NywxMi41MzkyMTI3IEwyMS44ODEyNDk1LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTEuMjY1ODg1NCBMMTIuNTMxMzI1NywxMS4yNjU4ODU0IEwxNC4zODE2Mjk0LDkuODczMTgzNjQgQzE0LjYzNzc4NzIsOS43MTY1MDQ1MyAxNC43NDk3MDA2LDkuNDAwNjYwMTQgMTQuNjQ3NzM1MSw5LjExNzE0NTUzIEMxNC41NDgyNTY0LDguODMzNjMxNTYgMTQuMjYyMjU1LDguNjU5NTQzNTIgMTMuOTYzODE4OSw4LjY5OTMzNSBaIiBpZD0icGJfdGJsLXRibC0zLWFycm93IiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNS45MjkyMzAsIDExLjg5NDczNykgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE1LjkyOTIzMCwgLTExLjg5NDczNykgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" /></td>
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

<span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin-left: 10px; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-10-15 18:47:50 UTC</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">< 1 s</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 1px 5px -1px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-10-15 18:47:50 UTC</span>
</div></td>
</tr>
</tfoot>

</table>


``` python
import pointblank as pb

validation = (
    pb.Validate(
        data=pb.load_dataset(dataset="game_revenue", tbl_type="duckdb")
    )
    .col_count_match(count=11)                       # expect 11 columns in the table
    .row_count_match(count=2000)                     # expect 2,000 rows in the table
    .row_count_match(count=0, inverse=True)          # expect that the table has rows
    .col_count_match(                                # compare column count against
        count=pb.load_dataset(                       # that of another table
            dataset="game_revenue", tbl_type="pandas"
        )
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

<span style="background-color: #000000; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #000000; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">DuckDB</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">2,000</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">11</span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_preview_tbl-_row_num_" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"></th>
<th id="pb_preview_tbl-player_id" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

player_id

<em>string</em>

</div></th>
<th id="pb_preview_tbl-session_id" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

session_id

<em>string</em>

</div></th>
<th id="pb_preview_tbl-session_start" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

session_start

<em>timestamp</em>

</div></th>
<th id="pb_preview_tbl-time" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

time

<em>timestamp</em>

</div></th>
<th id="pb_preview_tbl-item_type" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

item_type

<em>string</em>

</div></th>
<th id="pb_preview_tbl-item_name" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

item_name

<em>string</em>

</div></th>
<th id="pb_preview_tbl-item_revenue" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

item_revenue

<em>float64</em>

</div></th>
<th id="pb_preview_tbl-session_duration" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

session_duration

<em>float64</em>

</div></th>
<th id="pb_preview_tbl-start_day" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

start_day

<em>date</em>

</div></th>
<th id="pb_preview_tbl-acquisition" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

acquisition

<em>string</em>

</div></th>
<th id="pb_preview_tbl-country" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

country

<em>string</em>

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
