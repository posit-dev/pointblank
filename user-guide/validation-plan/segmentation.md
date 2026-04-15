# Segmentation

When validating data, you often need to analyze specific subsets or segments of your data separately. Maybe you want to ensure that data quality meets standards in each geographic region, for each product category, or across different time periods. This is where the `segments=` argument can be useful.

Data segmentation lets you split a validation step into multiple segments, with each segment receiving its own validation step. Rather than validating an entire table at once, you could instead validate different partitions separately and get separate results for each.

The `segments=` argument is available in many validation methods; typically it's in those methods that check values within rows, and those methods that examine entire rows (<a href="../../reference/Validate.rows_distinct.html#pointblank.Validate.rows_distinct" class="gdls-link"><code>Validate.rows_distinct()</code></a>, <a href="../../reference/Validate.rows_complete.html#pointblank.Validate.rows_complete" class="gdls-link"><code>Validate.rows_complete()</code></a>). When you use it, Pointblank will:

1.  split your data according to your segmentation criteria
2.  run the validation separately on each segment
3.  report results individually for each segment

Let's explore how to use the `segments=` argument through a few practical examples.


# Basic Segmentation by Column Values

The simplest way to segment data is by the unique values in a column. For the upcoming example, we'll use the `small_table` dataset, which contains a categorical-value column called `f`.

First, let's preview the dataset:


``` python
table = pb.load_dataset()

pb.preview(table)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="9" class="gt_heading gt_title gt_font_normal"><div>

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
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-04 11:00:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-04</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1-bcd-345</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3423.29</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">high</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-04 00:32:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-04</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5-egh-163</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">8</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">9999.99</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">low</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-05 13:32:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-05</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">8-kdg-938</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2343.23</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">high</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-06 17:23:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-06</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5-jdo-903</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3892.4</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">False</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">mid</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">2016-01-09 12:36:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">2016-01-09</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">8</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">3-ldm-038</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">283.94</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">low</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">9</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-20 04:30:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-20</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5-bce-642</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">9</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">837.93</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">False</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">high</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">10</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-20 04:30:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-20</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5-bce-642</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">9</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">837.93</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">False</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">high</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">11</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-26 20:07:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-26</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2-dmx-010</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">833.98</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">low</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">12</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-28 02:51:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-28</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7-dmx-010</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">8</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">108.34</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">False</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">low</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">13</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-30 11:23:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-30</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3-dka-303</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: #B22222; background-color: #FFC1C159">None</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2230.09</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">high</td>
</tr>
</tbody>
</table>


Now, let's validate that values in column `d` are greater than `100`, but we'll also segment the validation by the categorical values in column `f`:


``` python
validation_1 = (
    pb.Validate(
        data=pb.load_dataset(),
        tbl_name="small_table",
        label="Segmented validation by category"
    )
    .col_vals_gt(
        columns="d", value=100,

        # Segment by unique values in column `f` ---
        segments="f"
    )
    .interrogate()
)

validation_1
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Segmented validation by category</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">Polars</span><span style="background-color: none; color: #222222; padding: 0.5em 0.5em; position: inherit; margin: 5px 10px 5px -4px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">small_table</span>

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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>f / high</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">6</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">6<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>f / low</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>f / mid</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">2</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody>
</table>


In the validation report, notice that instead of a single validation step, we have multiple steps: one for each unique value in the `f` column. The segmentation is clearly indicated in the `STEP` column with labels like `SEGMENT f / high`, making it easy to identify which segment each validation result belongs to. This clear labeling helps when reviewing reports, especially with complex validations that use multiple segmentation criteria.


# Segmenting on Specific Values

Sometimes you don't want to segment on all unique values in a column, but only on specific ones of interest. You can do this by providing a tuple with the column name and a list of values:


``` python
validation_2 = (
    pb.Validate(
        data=pb.load_dataset(),
        tbl_name="small_table",
        label="Segmented validation on specific categories"
    )
    .col_vals_gt(
        columns="d",
        value=100,
        segments=("f", ["low", "high"])  # Only segment on "low" and "high" values in column `f`
    )
    .interrogate()
)

validation_2
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Segmented validation on specific categories</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">Polars</span><span style="background-color: none; color: #222222; padding: 0.5em 0.5em; position: inherit; margin: 5px 10px 5px -4px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">small_table</span>

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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>f / low</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>f / high</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">6</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">6<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody>
</table>


In this example, we only create validation steps for the `"low"` and `"high"` segments, ignoring any rows with `f` equal to `"mid"`.


# Multiple Segmentation Criteria

For more complex segmentation, you can provide a list of columns or column-value tuples. This creates segments based on combinations of criteria:


``` python
validation_3 = (
    pb.Validate(
        data=pb.load_dataset(),
        tbl_name="small_table",
        label="Multiple segmentation criteria"
    )
    .col_vals_gt(
        columns="d",
        value=100,

        # Segment by values in `f` AND specific values in `a` ---
        segments=["f", ("a", [1, 2])]
    )
    .interrogate()
)

validation_3
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Multiple segmentation criteria</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">Polars</span><span style="background-color: none; color: #222222; padding: 0.5em 0.5em; position: inherit; margin: 5px 10px 5px -4px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">small_table</span>

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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>f / high</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
LTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">6</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">6<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>f / low</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">5<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>f / mid</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">2</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">4</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>a / 1</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">5</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>a / 2</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">3</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">3<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody>
</table>


This creates validation steps for each combination of values in column `f` and the specified values in column `a`.


# Segmentation with Preprocessing

You can combine segmentation with preprocessing for powerful and flexible validations. All preprocessing is applied before segmentation occurs, which means you can create derived columns to segment on:


``` python
import polars as pl

# Define preprocessing function for creating a categorical column
def add_d_category_column(df):
    return df.with_columns(
        d_category=pl.when(pl.col("d") > 150).then(pl.lit("high")).otherwise(pl.lit("low"))
    )

validation_4 = (
    pb.Validate(
        data=pb.load_dataset(tbl_type="polars"),
        tbl_name="small_table",
        label="Segmentation with preprocessing",
    )
    .col_vals_gt(
        columns="d", value=100,

        # Create a column containing categorical values ---
        pre=add_d_category_column,

        # Segment by the computed column `d_category` generated via `pre=` ---
        segments="d_category",
    )
    .interrogate()
)

validation_4
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Segmentation with preprocessing</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">Polars</span><span style="background-color: none; color: #222222; padding: 0.5em 0.5em; position: inherit; margin: 5px 10px 5px -4px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">small_table</span>

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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>d_category / high</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">12</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">12<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>d_category / low</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">100</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">1</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(pre_applied)</span> Precondition applied: table dimensions <span style="font-family: monospace;">[13 rows, 8 columns]</span> → <span style="font-family: monospace;">[13 rows, <strong>9</strong> columns]</span>.</p>
<p>Step 2 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(pre_applied)</span> Precondition applied: table dimensions <span style="font-family: monospace;">[13 rows, 8 columns]</span> → <span style="font-family: monospace;">[13 rows, <strong>9</strong> columns]</span>.</p></td>
</tr>
</tfoot>

</table>


In this example, we first create a derived column `d_category` based on whether `d` is greater than `150`. Then, we segment our validation based on this derived column by using `segments="d_category"`.


# When to Use Segmentation

Segmentation is particularly useful when:

1.  Data quality standards vary by group: different regions, product lines, or customer segments might have different acceptable thresholds
2.  Identifying problem areas: segmentation helps pinpoint exactly where data quality issues exist, rather than just knowing that some issue exists somewhere in the data
3.  Generating detailed reports: by segmenting, you get more granular reporting that can be shared with different stakeholders responsible for different parts of the data
4.  Tracking improvements over time: segmented validations make it easier to see if data quality is improving in specific areas that were previously problematic

By using segmentation strategically in these scenarios, you can transform your data validation from a simple pass/fail system into a much more nuanced diagnostic tool that provides actionable insights about data quality across different dimensions. This targeted approach not only helps identify issues more precisely but also enables more effective communication of data quality metrics to relevant stakeholders.


# Segmentation vs. Multiple Validation Steps

So why use segmentation instead of just creating separate validation steps for each segment using filtering in the `pre=` argument? Well, segmentation offers several nice advantages:

1.  Conciseness: you define your validation logic once, not repeatedly for each segment
2.  Consistency: we can be certain that the same validation is applied uniformly across segments
3.  Clarity: the validation report will clearly organize results by segment (with extra labeling)
4.  Convenience: there's no need to manually extract and filter subsets of your data

Segmentation can end of simplifying your validation code while also providing more structured and informative reporting about different portions of your data.


# Practical Example: Validating Sales Data by Region and Product Type

Let's see a more realistic example where we validate sales data segmented by both region and product type:


``` python
import pandas as pd
import numpy as np

# Create a sample sales dataset
np.random.seed(123)

# Create a simple sales dataset
sales_data = pd.DataFrame({
    "region": np.random.choice(["North", "South", "East", "West"], 100),
    "product_type": np.random.choice(["Electronics", "Clothing", "Food"], 100),
    "units_sold": np.random.randint(5, 100, 100),
    "revenue": np.random.uniform(100, 10000, 100),
    "cost": np.random.uniform(50, 5000, 100)
})

# Calculate profit
sales_data["profit"] = sales_data["revenue"] - sales_data["cost"]
sales_data["profit_margin"] = sales_data["profit"] / sales_data["revenue"]

# Preview the dataset
pb.preview(sales_data)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="8" class="gt_heading gt_title gt_font_normal"><div>

<span style="background-color: #150458; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #150458; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Pandas</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">100</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">7</span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_preview_tbl-_row_num_" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"></th>
<th id="pb_preview_tbl-region" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

region

<em>str</em>

</div></th>
<th id="pb_preview_tbl-product_type" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

product_type

<em>str</em>

</div></th>
<th id="pb_preview_tbl-units_sold" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

units_sold

<em>int64</em>

</div></th>
<th id="pb_preview_tbl-revenue" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

revenue

<em>float64</em>

</div></th>
<th id="pb_preview_tbl-cost" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

cost

<em>float64</em>

</div></th>
<th id="pb_preview_tbl-profit" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

profit

<em>float64</em>

</div></th>
<th id="pb_preview_tbl-profit_margin" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

profit_margin

<em>float64</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">1</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">East</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">Clothing</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">55</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">8428.654356103547</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1363.5197435071943</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7065.134612596353</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">0.8382280627607168</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">South</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">Electronics</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6589.7066024003025</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3824.069456121553</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2765.6371462787497</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">0.41969048292246663</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">East</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">Food</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">23</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4680.5819759229435</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4122.545156369359</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">558.0368195535848</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">0.11922381071929586</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">4</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">East</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">Clothing</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">51</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5693.611988153584</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1797.3122335569797</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3896.2997545966045</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">0.6843282897927435</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">5</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">North</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">Clothing</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">50</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">4296.763518753258</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">4872.448283639371</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">-575.684764886113</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; border-bottom: 2px solid #6699CC80">-0.13398102138354426</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">96</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">West</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">Clothing</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">85</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6551.261354681658</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">936.7119894981438</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5614.549365183515</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">0.8570180704470368</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">97</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">South</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">Electronics</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">29</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">9543.579639173184</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2779.779531480257</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6763.800107692927</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">0.7087277901396456</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">98</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">East</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">Food</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">20</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4822.302251263769</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2833.48720726181</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1988.815044001959</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">0.41242023837903463</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">99</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">North</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">Clothing</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">54</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">8801.046116310079</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2185.8559620190636</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6615.1901542910155</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">0.7516368016788095</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">100</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">North</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">Clothing</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">85</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7942.857049695305</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1834.7969383843642</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6108.060111310941</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">0.7690003827458094</td>
</tr>
</tbody>
</table>


Now, let's validate that profit margins are above 20% across different regions and product types:


``` python
validation_5 = (
    pb.Validate(
        data=sales_data,
        tbl_name="sales_data",
        label="Sales data validation by region and product"
    )
    .col_vals_gt(
        columns="profit_margin",
        value=0.2,
        segments=["region", "product_type"],
        brief="Profit margin > 20% check"
    )
    .interrogate()
)

validation_5
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Sales data validation by region and product</span>

<span style="background-color: #150458; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px #150458; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">Pandas</span><span style="background-color: none; color: #222222; padding: 0.5em 0.5em; position: inherit; margin: 5px 10px 5px -4px; border: solid 1px #150458; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">sales_data</span>

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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>region / East</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()


<p>Profit margin > 20% check</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">profit_margin</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0.2</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">30</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">20<br />
0.67</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">10<br />
0.33</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>region / North</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()


<p>Profit margin > 20% check</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">profit_margin</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0.2</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">25</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">17<br />
0.68</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">8<br />
0.32</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>region / South</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()


<p>Profit margin > 20% check</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">profit_margin</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0.2</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">21</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">18<br />
0.86</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">3<br />
0.14</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">4</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>region / West</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()


<p>Profit margin > 20% check</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">profit_margin</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0.2</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">24</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">16<br />
0.67</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">8<br />
0.33</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">5</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>product_type / Clothing</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()


<p>Profit margin > 20% check</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">profit_margin</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0.2</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">38</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">28<br />
0.74</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">10<br />
0.26</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">6</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>product_type / Electronics</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()


<p>Profit margin > 20% check</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">profit_margin</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0.2</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">33</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">21<br />
0.64</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">12<br />
0.36</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">7</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; overflow-x: visible; white-space: nowrap"><div style="margin-top: 0px; margin-bottom: 0px; white-space: pre; font-size: 8px; color: darkblue; padding-bottom: 4px; ">
<strong><span style="font-family: Helvetica, arial, sans-serif;">SEGMENT  </span></strong><span>product_type / Food</span>

<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()


<p>Profit margin > 20% check</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">profit_margin</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0.2</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJzZWdtZW50ZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJzZWdtZW50ZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMTguNDA1MTA5NSwxNC4zNTE4ODg5IEMyMC40NDA1MTQyLDE0LjM1MTg4ODkgMjIuMDkwNTM1OCwxNi4wMDE5MTA1IDIyLjA5MDUzNTgsMTguMDM3MzE1MiBDMjIuMDkwNTM1OCwyMC4wNzI3MiAyMC40NDA1MTQyLDIxLjcyMjc0MTYgMTguNDA1MTA5NSwyMS43MjI3NDE2IEMxNi4zNjk3MDQ3LDIxLjcyMjc0MTYgMTQuNzE5NjgzMSwyMC4wNzI3MiAxNC43MTk2ODMxLDE4LjAzNzMxNTIgQzE0LjcxOTY4MzEsMTYuMDAxOTEwNSAxNi4zNjk3MDQ3LDE0LjM1MTg4ODkgMTguNDA1MTA5NSwxNC4zNTE4ODg5IFogTTE4LjQwNTEwOTUsMTQuOTc4NzYyMSBDMTYuNzE1OTE3MiwxNC45Nzg3NjIxIDE1LjM0NjU1NjMsMTYuMzQ4MTIzIDE1LjM0NjU1NjMsMTguMDM3MzE1MiBDMTUuMzQ2NTU2MywxOC43Njk3Mzg1IDE1LjYwNDAwMSwxOS40NDIwMzIyIDE2LjAzMzMzNTcsMTkuOTY4NjQxOSBMMjAuMzM2NDM2MSwxNS42NjU1NDE1IEMxOS44MDk4MjY0LDE1LjIzNjIwNjggMTkuMTM3NTMyNywxNC45Nzg3NjIxIDE4LjQwNTEwOTUsMTQuOTc4NzYyMSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">29</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">22<br />
0.76</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">7<br />
0.24</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


This validation gives us a detailed breakdown of profit margin performance across the different regions and product types, making it easy to identify areas that need attention.


# Best Practices for Segmentation

Effective data segmentation requires thoughtful planning about how to divide your data in ways that make sense for your validation needs. When implementing segmentation in your data validation workflow, consider these key principles:

1.  Choose meaningful segments: select segmentation columns that align with your business logic and organizational structure

2.  Use preprocessing when needed: if your raw data doesn't have good segmentation columns, create them through preprocessing (with the `pre=` argument)

3.  Combine with actions: for critical segments, define segment-specific actions using the `actions=` parameter to respond to validation failures.

By implementing these best practices, you'll create more targeted, maintainable, and actionable data validations. Segmentation becomes most powerful when it aligns with natural divisions in your data and analytical processes, allowing for more precise identification of quality issues while maintaining a unified validation framework.


# Conclusion

Data segmentation can make your validations more targeted and informative. By dividing your data into meaningful segments, you can identify quality issues with greater precision, apply appropriate validation standards to different parts of your data, and generate more actionable reports.

The `segments=` parameter transforms validation from a monolithic process into a granular assessment of data quality across various dimensions of your dataset. Whether you're dealing with regional differences, product categories, time periods, or any other meaningful divisions in your data, segmentation makes it possible to validate each portion according to its specific requirements while maintaining the simplicity of a unified validation framework.
