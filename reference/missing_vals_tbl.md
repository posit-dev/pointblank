## missing_vals_tbl()


Display a table that shows the missing values in the input table.


Usage

``` python
missing_vals_tbl(
    data,
    missing=None,
    as_heatmap=False,
)
```


The [missing_vals_tbl()](missing_vals_tbl.md#pointblank.missing_vals_tbl) function generates a table that shows the missing values in the input table. The table is displayed using the Great Tables API, which allows for further customization of the table's appearance if so desired.

By default, missingness is treated as binary (a value is either Null or it isn't) and the function renders a sector-based heatmap of the proportion of Null values across the rows of each column. When a `missing=` mapping of columns to <a href="MissingSpec.html#pointblank.MissingSpec" class="gdls-link"><code>MissingSpec</code></a> objects is supplied, the function instead renders a *structured missingness* breakdown: one row per column with the count and percentage of complete values and of each missing *reason* (e.g., `refused`, `not_asked`). Declared (coded) reasons are grouped under a "Missing Reasons" spanner and keep their raw input form as labels; actual `Null`/`None`/`NA` values (which are not part of the spec) are tallied in a fixed "Null" column at the far right (styled like "Complete"), so they aren't mistaken for declared reasons.

Note that supplying `missing=` produces a *different report* than the default view: it is a distinct visualization (a per-reason breakdown table, or a per-reason heatmap with `as_heatmap=True`), not an annotated version of the default sector heatmap. The report titles differ accordingly ("Missing Values" for the default, "Missing Values by Reason" or "Missing Pattern Heatmap" for the structured views), and the shared header/title styling makes the family resemblance clear.


## Parameters


`data: Any`  
The table for which to display the missing values. This could be a DataFrame object, an Ibis table object, a CSV file path, a Parquet file path, or a database connection string. Read the *Supported Input Table Types* section for details on the supported table types.

`missing: dict[str, MissingSpec] | None = None`  
An optional dictionary mapping column names to <a href="MissingSpec.html#pointblank.MissingSpec" class="gdls-link"><code>MissingSpec</code></a> objects. When provided, the function renders a structured breakdown of missingness by reason for the specified columns (rather than the default sector heatmap). The reason columns are the union of reasons across the supplied specs; a reason that isn't defined for a given column is shown as an em dash (not applicable), as distinct from a defined-but-unobserved reason (shown as `0 (0%)`).

`as_heatmap: bool = ``False`  
Only applies when `missing=` is provided. When `True`, render the per-reason proportions as a color-coded heatmap (cells shaded from light to dark by the proportion missing) instead of the count/percentage text breakdown. Default is `False`.


## Returns


`GT`  
A GT object that displays the table of missing values in the input table.


## Supported Input Table Types

The `data=` parameter can be given any of the following table types:

- Polars DataFrame (`"polars"`)
- Pandas DataFrame (`"pandas"`)
- PySpark table (`"pyspark"`)
- DuckDB table (`"duckdb"`)\*
- MySQL table (`"mysql"`)\*
- PostgreSQL table (`"postgresql"`)\*
- SQLite table (`"sqlite"`)\*
- Microsoft SQL Server table (`"mssql"`)\*
- Snowflake table (`"snowflake"`)\*
- Databricks table (`"databricks"`)\*
- BigQuery table (`"bigquery"`)\*
- Parquet table (`"parquet"`)\*
- CSV files (string path or `pathlib.Path` object with `.csv` extension)
- Parquet files (string path, `pathlib.Path` object, glob pattern, directory with `.parquet` extension, or partitioned dataset)
- Database connection strings (URI format with optional table specification)

The table types marked with an asterisk need to be prepared as Ibis tables (with type of `ibis.expr.types.relations.Table`). Furthermore, using [missing_vals_tbl()](missing_vals_tbl.md#pointblank.missing_vals_tbl) with these types of tables requires the Ibis library (`v9.5.0` or above) to be installed. If the input table is a Polars or Pandas DataFrame, the availability of Ibis is not needed.


## The Missing Values Table

The missing values table shows the proportion of missing values in each column of the input table. The table is divided into sectors, with each sector representing a range of rows in the table. The proportion of missing values in each sector is calculated for each column. The table is displayed using the Great Tables API, which allows for further customization of the table's appearance.

To ensure that the table can scale to tables with many columns, each row in the reporting table represents a column in the input table. There are 10 sectors shown in the table, where the first sector represents the first 10% of the rows, the second sector represents the next 10% of the rows, and so on. Any sectors that are light blue indicate that there are no missing values in that sector. If there are missing values, the proportion of missing values is shown by a gray color (light gray for low proportions, dark gray to black for very high proportions).


## Examples

The [missing_vals_tbl()](missing_vals_tbl.md#pointblank.missing_vals_tbl) function is useful for quickly identifying columns with missing values in a table. Here's an example using the `nycflights` dataset (loaded as a Polars DataFrame using the <a href="load_dataset.html#pointblank.load_dataset" class="gdls-link"><code>load_dataset()</code></a> function):


``` python
import pointblank as pb

nycflights = pb.load_dataset("nycflights", tbl_type="polars")

pb.missing_vals_tbl(nycflights)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#wywbznuttf table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#wywbznuttf thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#wywbznuttf p { margin: 0; padding: 0; }
 #wywbznuttf .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #wywbznuttf .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #wywbznuttf .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #wywbznuttf .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #wywbznuttf .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #wywbznuttf .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wywbznuttf .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #wywbznuttf .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #wywbznuttf .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #wywbznuttf .gt_column_spanner_outer:first-child { padding-left: 0; }
 #wywbznuttf .gt_column_spanner_outer:last-child { padding-right: 0; }
 #wywbznuttf .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #wywbznuttf .gt_spanner_row { border-bottom-style: hidden; }
 #wywbznuttf .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #wywbznuttf .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #wywbznuttf .gt_from_md> :first-child { margin-top: 0; }
 #wywbznuttf .gt_from_md> :last-child { margin-bottom: 0; }
 #wywbznuttf .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #wywbznuttf .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #wywbznuttf .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #wywbznuttf .gt_row_group_first td { border-top-width: 2px; }
 #wywbznuttf .gt_row_group_first th { border-top-width: 2px; }
 #wywbznuttf .gt_striped { color: #333333; background-color: #F4F4F4; }
 #wywbznuttf .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wywbznuttf .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #wywbznuttf .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #wywbznuttf .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #wywbznuttf .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #wywbznuttf .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #wywbznuttf .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #wywbznuttf .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wywbznuttf .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #wywbznuttf .gt_left { text-align: left; }
 #wywbznuttf .gt_center { text-align: center; }
 #wywbznuttf .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #wywbznuttf .gt_font_normal { font-weight: normal; }
 #wywbznuttf .gt_font_bold { font-weight: bold; }
 #wywbznuttf .gt_font_italic { font-style: italic; }
 #wywbznuttf .gt_super { font-size: 65%; }
 #wywbznuttf .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wywbznuttf .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #wywbznuttf .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #wywbznuttf .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #wywbznuttf .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #wywbznuttf .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
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


The table shows the proportion of missing values in each column of the `nycflights` dataset. The table is divided into sectors, with each sector representing a range of rows in the table (with around 34,000 rows per sector). The proportion of missing values in each sector is calculated for each column. The various shades of gray indicate the proportion of missing values in each sector. Many columns have no missing values at all, and those sectors are colored light blue.
