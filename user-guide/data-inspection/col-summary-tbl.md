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
</tbody><tfoot class="gt_sourcenotes">
<tr>
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
