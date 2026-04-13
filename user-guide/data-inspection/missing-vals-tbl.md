# Missing Values Reporting

Sometimes values just aren't there: they're missing. This can either be expected or another thing to worry about. Either way, we can dig a little deeper if need be and use the [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) function to generate a summary table that can elucidate how many values are missing, and roughly where.


# Using and Understanding [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl)

The missing values table is arranged a lot like the column summary table (generated via the [col_summary_tbl()](../../reference/col_summary_tbl.md#pointblank.col_summary_tbl) function) in that columns of the input table are arranged as rows in the reporting table. Let's use [missing_vals_tbl()](../../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl) on the `nycflights` dataset, which has a lot of missing values:


``` python
import pointblank as pb

nycflights = pb.load_dataset(dataset="nycflights", tbl_type="polars")

pb.missing_vals_tbl(nycflights)
```


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
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="11" class="gt_sourcenote"><div style="display: flex; align-items: center; padding-bottom: 10px;">


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
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="11" class="gt_sourcenote"><div style="display: flex; align-items: center; padding-bottom: 10px;">


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
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="11" class="gt_sourcenote"><div style="display: flex; align-items: center; padding-bottom: 10px;">


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
