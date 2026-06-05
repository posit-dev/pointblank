## col_summary_tbl()


Generate a column-level summary table of a dataset.


Usage

``` python
col_summary_tbl(
    data,
    tbl_name=None,
)
```


The [col_summary_tbl()](col_summary_tbl.md#pointblank.col_summary_tbl) function generates a summary table of a dataset, focusing on providing column-level information about the dataset. The summary includes the following information:

- the type of the table (e.g., `"polars"`, `"pandas"`, etc.)
- the number of rows and columns in the table
- column-level information, including:
  - the column name
  - the column type
  - measures of missingness and distinctness
  - descriptive stats and quantiles
  - statistics for datetime columns

The summary table is returned as a GT object, which can be displayed in a notebook or saved to an HTML file.

> **Warning: Warning**
>
> The [col_summary_tbl()](col_summary_tbl.md#pointblank.col_summary_tbl) function is still experimental. Please report any issues you encounter in the [Pointblank issue tracker](https://github.com/posit-dev/pointblank/issues).


## Parameters


`data: Any`  
The table to summarize, which could be a DataFrame object, an Ibis table object, a CSV file path, a Parquet file path, or a database connection string. Read the *Supported Input Table Types* section for details on the supported table types.

`tbl_name: str | None = None`  
Optionally, the name of the table could be provided as `tbl_name=`.


## Returns


`GT`  
A GT object that displays the column-level summaries of the table.


## Supported Input Table Types

The `data=` parameter can be given any of the following table types:

- Polars DataFrame (`"polars"`)
- Pandas DataFrame (`"pandas"`)
- DuckDB table (`"duckdb"`)\*
- MySQL table (`"mysql"`)\*
- PostgreSQL table (`"postgresql"`)\*
- SQLite table (`"sqlite"`)\*
- Parquet table (`"parquet"`)\*
- CSV files (string path or `pathlib.Path` object with `.csv` extension)
- Parquet files (string path, `pathlib.Path` object, glob pattern, directory with `.parquet` extension, or partitioned dataset)
- GitHub URLs (direct links to CSV or Parquet files on GitHub)
- Database connection strings (URI format with optional table specification)

The table types marked with an asterisk need to be prepared as Ibis tables (with type of `ibis.expr.types.relations.Table`). Furthermore, using [col_summary_tbl()](col_summary_tbl.md#pointblank.col_summary_tbl) with these types of tables requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a Polars or Pandas DataFrame, the availability of Ibis is not needed.


## Examples

It's easy to get a column-level summary of a table using the [col_summary_tbl()](col_summary_tbl.md#pointblank.col_summary_tbl) function. Here's an example using the `small_table` dataset (itself loaded using the <a href="load_dataset.html#pointblank.load_dataset" class="gdls-link"><code>load_dataset()</code></a> function):


``` python
import pointblank as pb

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

pb.col_summary_tbl(data=small_table)
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


This table used above was a Polars DataFrame, but the [col_summary_tbl()](col_summary_tbl.md#pointblank.col_summary_tbl) function works with any table supported by `pointblank`, including Pandas DataFrames and Ibis backend tables. Here's an example using a DuckDB table handled by Ibis:


``` python
nycflights = pb.load_dataset(dataset="nycflights", tbl_type="duckdb")

pb.col_summary_tbl(data=nycflights, tbl_name="nycflights")
```


    /opt/hostedtoolcache/Python/3.11.15/x64/lib/python3.11/site-packages/great_tables/_tbl_data.py:818: UserWarning: PyArrow Table support is currently experimental.
      warnings.warn("PyArrow Table support is currently experimental.")


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal"><div>

<span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">336,776</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">18</span>

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
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
year

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,013</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,013</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,013</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,013</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,013</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,013</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,013</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,013</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
month

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">12<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">6.55</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">3.41</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">6.77</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">10</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">12</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">12</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">6</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
day

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">31<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">15.71</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">8.77</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">8</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">15.64</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">23</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">29</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">31</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">15</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
dep_time

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">8255<br />
0.02</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1319<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,349.11</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">488.28</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">514</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">907</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,381.15</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,744</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,112</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,400</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">837</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
sched_dep_time

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1021<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,344.25</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">467.34</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">106</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">545</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">906</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,377.13</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,729</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,050</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,359</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">823</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
dep_delay

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">8255<br />
0.02</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">528<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">12.64</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">40.21</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">−43</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">−13</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">−5</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">−1.58</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">11</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">88</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,301</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">16</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
arr_time

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">8713<br />
0.03</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1412<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,502.05</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">533.26</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">10</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,104</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,539.25</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,940</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,248</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,400</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">836</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
sched_arr_time

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1163<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,536.38</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">497.46</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">15</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,124</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,575.93</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,945</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,246</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,359</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">821</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
arr_delay

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">9430<br />
0.03</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">578<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">6.9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">44.63</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">−86</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">−48</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">−17</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">−4.92</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">14</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">91</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,272</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">31</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPnN0cmluZzwvdGl0bGU+CiAgICA8ZyBpZD0iaWNvbiIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InN0cmluZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNC4wMDAwMDAsIDQuNTAwMDAwKSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJzcXVhcmUiIHN0cm9rZT0iIzlBODcwMCIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSIjQ0ZCNjAwIiB4PSIyIiB5PSIyIiB3aWR0aD0iMTY4IiBoZWlnaHQ9IjE2OCIgcng9IjgiIC8+CiAgICAgICAgICAgIDxnIGlkPSJTIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0OS41MTYwMDAsIDM4LjI4MDAwMCkiIGZpbGw9IiNGRkZGRkYiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMzQuOTgsOTUuMzA0IEMyNi44ODQsOTUuMzA0IDIwLjAyLDkzLjkxOCAxNC4zODgsOTEuMTQ2IEM4Ljc1Niw4OC4zNzQgMy45Niw4NC43ODggMCw4MC4zODggTDEzLjIsNjcuMDU2IEMxOS40NDgsNzQuMDk2IDI3LjE0OCw3Ny42MTYgMzYuMyw3Ny42MTYgQzQxLjIyOCw3Ny42MTYgNDQuODgsNzYuNjA0IDQ3LjI1Niw3NC41OCBDNDkuNjMyLDcyLjU1NiA1MC44Miw2OS44NzIgNTAuODIsNjYuNTI4IEM1MC44Miw2My45NzYgNTAuMTE2LDYxLjg0MiA0OC43MDgsNjAuMTI2IEM0Ny4zLDU4LjQxIDQ0LjQ0LDU3LjI0NCA0MC4xMjgsNTYuNjI4IEwzMS4wMiw1NS40NCBDMjEuMjUyLDU0LjIwOCAxNC4xMDIsNTEuMjYgOS41Nyw0Ni41OTYgQzUuMDM4LDQxLjkzMiAyLjc3MiwzNS43MjggMi43NzIsMjcuOTg0IEMyLjc3MiwyMy44NDggMy41NjQsMjAuMDY0IDUuMTQ4LDE2LjYzMiBDNi43MzIsMTMuMiA4Ljk5OCwxMC4yNTIgMTEuOTQ2LDcuNzg4IEMxNC44OTQsNS4zMjQgMTguNTAyLDMuNDEgMjIuNzcsMi4wNDYgQzI3LjAzOCwwLjY4MiAzMS45LDAgMzcuMzU2LDAgQzQ0LjMwOCwwIDUwLjQyNCwxLjEgNTUuNzA0LDMuMyBDNjAuOTg0LDUuNSA2NS41MTYsOC43MTIgNjkuMywxMi45MzYgTDU1Ljk2OCwyNi40IEM1My43NjgsMjMuODQ4IDUxLjEwNiwyMS43NTggNDcuOTgyLDIwLjEzIEM0NC44NTgsMTguNTAyIDQwLjkyLDE3LjY4OCAzNi4xNjgsMTcuNjg4IEMzMS42OCwxNy42ODggMjguMzM2LDE4LjQ4IDI2LjEzNiwyMC4wNjQgQzIzLjkzNiwyMS42NDggMjIuODM2LDIzLjg0OCAyMi44MzYsMjYuNjY0IEMyMi44MzYsMjkuODMyIDIzLjY5NCwzMi4xMiAyNS40MSwzMy41MjggQzI3LjEyNiwzNC45MzYgMjkuOTIsMzUuOTQ4IDMzLjc5MiwzNi41NjQgTDQyLjksMzguMDE2IEM1Mi40MDQsMzkuNTEyIDU5LjQ0NCw0Mi40MzggNjQuMDIsNDYuNzk0IEM2OC41OTYsNTEuMTUgNzAuODg0LDU3LjMzMiA3MC44ODQsNjUuMzQgQzcwLjg4NCw2OS43NCA3MC4wOTIsNzMuNzg4IDY4LjUwOCw3Ny40ODQgQzY2LjkyNCw4MS4xOCA2NC42MTQsODQuMzQ4IDYxLjU3OCw4Ni45ODggQzU4LjU0Miw4OS42MjggNTQuNzgsOTEuNjc0IDUwLjI5Miw5My4xMjYgQzQ1LjgwNCw5NC41NzggNDAuNyw5NS4zMDQgMzQuOTgsOTUuMzA0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
carrier

String
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">16<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
flight

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">3844<br />
0.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,971.92</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,632.47</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">553</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,499.04</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3,465</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">4,695</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">8,500</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,912</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPnN0cmluZzwvdGl0bGU+CiAgICA8ZyBpZD0iaWNvbiIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InN0cmluZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNC4wMDAwMDAsIDQuNTAwMDAwKSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJzcXVhcmUiIHN0cm9rZT0iIzlBODcwMCIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSIjQ0ZCNjAwIiB4PSIyIiB5PSIyIiB3aWR0aD0iMTY4IiBoZWlnaHQ9IjE2OCIgcng9IjgiIC8+CiAgICAgICAgICAgIDxnIGlkPSJTIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0OS41MTYwMDAsIDM4LjI4MDAwMCkiIGZpbGw9IiNGRkZGRkYiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMzQuOTgsOTUuMzA0IEMyNi44ODQsOTUuMzA0IDIwLjAyLDkzLjkxOCAxNC4zODgsOTEuMTQ2IEM4Ljc1Niw4OC4zNzQgMy45Niw4NC43ODggMCw4MC4zODggTDEzLjIsNjcuMDU2IEMxOS40NDgsNzQuMDk2IDI3LjE0OCw3Ny42MTYgMzYuMyw3Ny42MTYgQzQxLjIyOCw3Ny42MTYgNDQuODgsNzYuNjA0IDQ3LjI1Niw3NC41OCBDNDkuNjMyLDcyLjU1NiA1MC44Miw2OS44NzIgNTAuODIsNjYuNTI4IEM1MC44Miw2My45NzYgNTAuMTE2LDYxLjg0MiA0OC43MDgsNjAuMTI2IEM0Ny4zLDU4LjQxIDQ0LjQ0LDU3LjI0NCA0MC4xMjgsNTYuNjI4IEwzMS4wMiw1NS40NCBDMjEuMjUyLDU0LjIwOCAxNC4xMDIsNTEuMjYgOS41Nyw0Ni41OTYgQzUuMDM4LDQxLjkzMiAyLjc3MiwzNS43MjggMi43NzIsMjcuOTg0IEMyLjc3MiwyMy44NDggMy41NjQsMjAuMDY0IDUuMTQ4LDE2LjYzMiBDNi43MzIsMTMuMiA4Ljk5OCwxMC4yNTIgMTEuOTQ2LDcuNzg4IEMxNC44OTQsNS4zMjQgMTguNTAyLDMuNDEgMjIuNzcsMi4wNDYgQzI3LjAzOCwwLjY4MiAzMS45LDAgMzcuMzU2LDAgQzQ0LjMwOCwwIDUwLjQyNCwxLjEgNTUuNzA0LDMuMyBDNjAuOTg0LDUuNSA2NS41MTYsOC43MTIgNjkuMywxMi45MzYgTDU1Ljk2OCwyNi40IEM1My43NjgsMjMuODQ4IDUxLjEwNiwyMS43NTggNDcuOTgyLDIwLjEzIEM0NC44NTgsMTguNTAyIDQwLjkyLDE3LjY4OCAzNi4xNjgsMTcuNjg4IEMzMS42OCwxNy42ODggMjguMzM2LDE4LjQ4IDI2LjEzNiwyMC4wNjQgQzIzLjkzNiwyMS42NDggMjIuODM2LDIzLjg0OCAyMi44MzYsMjYuNjY0IEMyMi44MzYsMjkuODMyIDIzLjY5NCwzMi4xMiAyNS40MSwzMy41MjggQzI3LjEyNiwzNC45MzYgMjkuOTIsMzUuOTQ4IDMzLjc5MiwzNi41NjQgTDQyLjksMzguMDE2IEM1Mi40MDQsMzkuNTEyIDU5LjQ0NCw0Mi40MzggNjQuMDIsNDYuNzk0IEM2OC41OTYsNTEuMTUgNzAuODg0LDU3LjMzMiA3MC44ODQsNjUuMzQgQzcwLjg4NCw2OS43NCA3MC4wOTIsNzMuNzg4IDY4LjUwOCw3Ny40ODQgQzY2LjkyNCw4MS4xOCA2NC42MTQsODQuMzQ4IDYxLjU3OCw4Ni45ODggQzU4LjU0Miw4OS42MjggNTQuNzgsOTEuNjc0IDUwLjI5Miw5My4xMjYgQzQ1LjgwNCw5NC41NzggNDAuNyw5NS4zMDQgMzQuOTgsOTUuMzA0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
tailnum

String
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">2512<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">4044<br />
0.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">6</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0.07</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">5</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">6</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">6</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">6</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">6</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">6</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">6</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPnN0cmluZzwvdGl0bGU+CiAgICA8ZyBpZD0iaWNvbiIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InN0cmluZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNC4wMDAwMDAsIDQuNTAwMDAwKSI+CiAgICAgICAgICAgIDxyZWN0IGlkPSJzcXVhcmUiIHN0cm9rZT0iIzlBODcwMCIgc3Ryb2tlLXdpZHRoPSI0IiBmaWxsPSIjQ0ZCNjAwIiB4PSIyIiB5PSIyIiB3aWR0aD0iMTY4IiBoZWlnaHQ9IjE2OCIgcng9IjgiIC8+CiAgICAgICAgICAgIDxnIGlkPSJTIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0OS41MTYwMDAsIDM4LjI4MDAwMCkiIGZpbGw9IiNGRkZGRkYiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMzQuOTgsOTUuMzA0IEMyNi44ODQsOTUuMzA0IDIwLjAyLDkzLjkxOCAxNC4zODgsOTEuMTQ2IEM4Ljc1Niw4OC4zNzQgMy45Niw4NC43ODggMCw4MC4zODggTDEzLjIsNjcuMDU2IEMxOS40NDgsNzQuMDk2IDI3LjE0OCw3Ny42MTYgMzYuMyw3Ny42MTYgQzQxLjIyOCw3Ny42MTYgNDQuODgsNzYuNjA0IDQ3LjI1Niw3NC41OCBDNDkuNjMyLDcyLjU1NiA1MC44Miw2OS44NzIgNTAuODIsNjYuNTI4IEM1MC44Miw2My45NzYgNTAuMTE2LDYxLjg0MiA0OC43MDgsNjAuMTI2IEM0Ny4zLDU4LjQxIDQ0LjQ0LDU3LjI0NCA0MC4xMjgsNTYuNjI4IEwzMS4wMiw1NS40NCBDMjEuMjUyLDU0LjIwOCAxNC4xMDIsNTEuMjYgOS41Nyw0Ni41OTYgQzUuMDM4LDQxLjkzMiAyLjc3MiwzNS43MjggMi43NzIsMjcuOTg0IEMyLjc3MiwyMy44NDggMy41NjQsMjAuMDY0IDUuMTQ4LDE2LjYzMiBDNi43MzIsMTMuMiA4Ljk5OCwxMC4yNTIgMTEuOTQ2LDcuNzg4IEMxNC44OTQsNS4zMjQgMTguNTAyLDMuNDEgMjIuNzcsMi4wNDYgQzI3LjAzOCwwLjY4MiAzMS45LDAgMzcuMzU2LDAgQzQ0LjMwOCwwIDUwLjQyNCwxLjEgNTUuNzA0LDMuMyBDNjAuOTg0LDUuNSA2NS41MTYsOC43MTIgNjkuMywxMi45MzYgTDU1Ljk2OCwyNi40IEM1My43NjgsMjMuODQ4IDUxLjEwNiwyMS43NTggNDcuOTgyLDIwLjEzIEM0NC44NTgsMTguNTAyIDQwLjkyLDE3LjY4OCAzNi4xNjgsMTcuNjg4IEMzMS42OCwxNy42ODggMjguMzM2LDE4LjQ4IDI2LjEzNiwyMC4wNjQgQzIzLjkzNiwyMS42NDggMjIuODM2LDIzLjg0OCAyMi44MzYsMjYuNjY0IEMyMi44MzYsMjkuODMyIDIzLjY5NCwzMi4xMiAyNS40MSwzMy41MjggQzI3LjEyNiwzNC45MzYgMjkuOTIsMzUuOTQ4IDMzLjc5MiwzNi41NjQgTDQyLjksMzguMDE2IEM1Mi40MDQsMzkuNTEyIDU5LjQ0NCw0Mi40MzggNjQuMDIsNDYuNzk0IEM2OC41OTYsNTEuMTUgNzAuODg0LDU3LjMzMiA3MC44ODQsNjUuMzQgQzcwLjg4NCw2OS43NCA3MC4wOTIsNzMuNzg4IDY4LjUwOCw3Ny40ODQgQzY2LjkyNCw4MS4xOCA2NC42MTQsODQuMzQ4IDYxLjU3OCw4Ni45ODggQzU4LjU0Miw4OS42MjggNTQuNzgsOTEuNjc0IDUwLjI5Miw5My4xMjYgQzQ1LjgwNCw5NC41NzggNDAuNyw5NS4zMDQgMzQuOTgsOTUuMzA0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
origin

String
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">3<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
</tr>
<tr>
uNTAyLDMuNDEgMjIuNzcsMi4wNDYgQzI3LjAzOCwwLjY4MiAzMS45LDAgMzcuMzU2LDAgQzQ0LjMwOCwwIDUwLjQyNCwxLjEgNTUuNzA0LDMuMyBDNjAuOTg0LDUuNSA2NS41MTYsOC43MTIgNjkuMywxMi45MzYgTDU1Ljk2OCwyNi40IEM1My43NjgsMjMuODQ4IDUxLjEwNiwyMS43NTggNDcuOTgyLDIwLjEzIEM0NC44NTgsMTguNTAyIDQwLjkyLDE3LjY4OCAzNi4xNjgsMTcuNjg4IEMzMS42OCwxNy42ODggMjguMzM2LDE4LjQ4IDI2LjEzNiwyMC4wNjQgQzIzLjkzNiwyMS42NDggMjIuODM2LDIzLjg0OCAyMi44MzYsMjYuNjY0IEMyMi44MzYsMjkuODMyIDIzLjY5NCwzMi4xMiAyNS40MSwzMy41MjggQzI3LjEyNiwzNC45MzYgMjkuOTIsMzUuOTQ4IDMzLjc5MiwzNi41NjQgTDQyLjksMzguMDE2IEM1Mi40MDQsMzkuNTEyIDU5LjQ0NCw0Mi40MzggNjQuMDIsNDYuNzk0IEM2OC41OTYsNTEuMTUgNzAuODg0LDU3LjMzMiA3MC44ODQsNjUuMzQgQzcwLjg4NCw2OS43NCA3MC4wOTIsNzMuNzg4IDY4LjUwOCw3Ny40ODQgQzY2LjkyNCw4MS4xOCA2NC42MTQsODQuMzQ4IDYxLjU3OCw4Ni45ODggQzU4LjU0Miw4OS42MjggNTQuNzgsOTEuNjc0IDUwLjI5Miw5My4xMjYgQzQ1LjgwNCw5NC41NzggNDAuNyw5NS4zMDQgMzQuOTgsOTUuMzA0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
dest

String
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">105<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
air_time

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">9430<br />
0.03</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">510<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">150.69</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">93.69</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">20</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">31</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">82</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">129.35</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">192</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">339</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">695</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">110</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
distance

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">214<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,039.91</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">733.23</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">17</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">116</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">502</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">861.05</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">1,389</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">2,475</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">4,983</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">887</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
hour

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">20<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">13.18</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">4.66</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">1</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">5</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">9</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">13.4</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">17</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">20</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">23</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">8</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgdmlld2JveD0iMCAwIDE4MCAxODEiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7IGRpc3BsYXk6IGJsb2NrOyB2ZXJ0aWNhbC1hbGlnbjogbWlkZGxlOyBtYXJnaW46IGF1dG87IHBhZGRpbmctdG9wOiAwLjVweDsiPgogICAgPHRpdGxlPm51bWVyaWM8L3RpdGxlPgogICAgPGcgaWQ9Imljb24iIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJudW1lcmljIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg0LjAwMDAwMCwgNC4zMzkzNDIpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9InNxdWFyZSIgc3Ryb2tlPSIjNjQwMTc3IiBzdHJva2Utd2lkdGg9IjQiIGZpbGw9IiNBNDAwQ0YiIHg9IjIiIHk9IjIiIHdpZHRoPSIxNjgiIGhlaWdodD0iMTY4IiByeD0iOCIgLz4KICAgICAgICAgICAgPGcgaWQ9Ik4iIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ5LjY2NDAwMCwgMzkuODY0MDAwKSIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwb2x5Z29uIHBvaW50cz0iMjguMTE2IDQ4LjU3NiAxOS4yNzIgMzAuMDk2IDE4Ljg3NiAzMC4wOTYgMTguODc2IDkyLjEzNiAwIDkyLjEzNiAwIDAgMjEuOTEyIDAgNDcuMTI0IDQzLjU2IDU1Ljk2OCA2Mi4wNCA1Ni4zNjQgNjIuMDQgNTYuMzY0IDAgNzUuMjQgMCA3NS4yNCA5Mi4xMzYgNTMuMzI4IDkyLjEzNiI+PC9wb2x5Z29uPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; font-size: 12px"><div style="font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden;">
minute

Int64
</div></td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0<br />
0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">60<br />
<.01</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">26.23</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">19.3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">0</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">8</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">28.3</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-left: 1px dashed #E5E5E5; font-size: 10px;">44</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">58</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">59</td>
<td class="gt_row gt_center" style="text-align: right; font-family: IBM Plex Mono; border-right: 1px solid #D3D3D3; border-left: 1px dashed #E5E5E5; font-size: 10px;">36</td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote">String columns statistics regard the string's length.</td>
</tr>
</tfoot>

</table>
