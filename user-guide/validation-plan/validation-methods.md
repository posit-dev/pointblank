# Validation Methods

Pointblank provides a comprehensive suite of validation methods to verify different aspects of your data. Each method creates a validation step that becomes part of your validation plan.

These validation methods cover everything from checking column values against thresholds to validating the table structure and detecting duplicates. Combined into validation steps, they form the foundation of your data quality workflow.

Pointblank provides [over 40 validation methods](https://posit-dev.github.io/pointblank/reference/#validation-steps) to handle diverse data quality requirements. These are grouped into five main categories:

1.  Column Value Validations
2.  Row-based Validations
3.  Table Structure Validations
4.  AI-Powered Validations
5.  Aggregate Validations

Within each of these categories, we'll walk through several examples showing how each validation method creates steps in your validation plan.

And we'll use the `small_table` dataset for all of our examples. Here's a preview of it:


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
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-09 12:36:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-09</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">8</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3-ldm-038</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">283.94</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">low</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">6</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-11 06:15:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-11</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2-dhe-923</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3291.03</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">mid</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-15 18:46:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-15</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1-knw-093</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">843.34</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">True</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">high</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">8</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-17 11:27:00</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2016-01-17</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5-boe-639</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1035.64</td>
<td class="gt_row gt_center" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">False</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">low</td>
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


# Validation Methods to Validation Steps

In Pointblank, validation *methods* become validation *steps* when you add them to a validation plan. Each method creates a distinct step that performs a specific check on your data.

Here's a simple example showing how three validation methods create three validation steps:


``` python
import pointblank as pb

(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))

    # Step 1: Check that values in column `a` are greater than 2 ---
    .col_vals_gt(columns="a", value=2, brief="Values in 'a' must exceed 2.")

    # Step 2: Check that column 'date' exists in the table ---
    .col_exists(columns="date", brief="Column 'date' must exist.")

    # Step 3: Check that the table has exactly 13 rows ---
    .row_count_match(count=13, brief="Table should have exactly 13 rows.")
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()


<p>Values in 'a' must exceed 2.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">2</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">9<br />
0.69</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">4<br />
0.31</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()


<p>Column 'date' must exist.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+cm93X2NvdW50X21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InJvd19jb3VudF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNzkzMTAzKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMjkuMTY0NjE0OSwyOC40ODUzMzA2IEwyOS4xNjQ2MTQ5LDY5LjQ4NTMzMDYgTDI3LjA1NDkwODQsNjkuNDg1MDQ5MyBDMjYuOTI5NjE3OSw2OS40ODI2OTQ0IDI2LjYyOTUwODUsNjkuNDYwNjY0NSAyNi40MTYzNzA4LDY5LjI1NDg3NjUgQzI2LjI4ODI0ODEsNjkuMTMxMTcxOCAyNi4yMTIyMzQsNjguOTM0ODI2OSAyNi4yMTIyMzQsNjguNjUxOTk3MyBMMjYuMjEyMjM0LDY4LjY1MTk5NzMgTDI2LjIxMjIzNCwyOS4zMTg2NjQgQzI2LjIxMjIzNCwyOS4wMzU4MzQ0IDI2LjI4ODI0ODEsMjguODM5NDg5NSAyNi40MTYzNzA4LDI4LjcxNTc4NDggQzI2LjY1NjE1MDcsMjguNDg0MjczMyAyNy4wMDYwMDQxLDI4LjQ4NTMzMDYgMjcuMDkzMTg2MywyOC40ODUzMzA2IEwyOS4xNjQ2MTQ5LDI4LjQ4NTMzMDYgWiBNMzUuMTU5OTM5MywyOC40ODUzMzA2IEwzNS4xNTk5MzkzLDY5LjQ4NTMzMDYgTDMyLjI2NDUyODYsNjkuNDg1MzMwNiBMMzIuMjY0NTI4NiwyOC40ODUzMzA2IEwzNS4xNTk5MzkzLDI4LjQ4NTMzMDYgWiBNNDAuMzU5MzQ3NSwyOC40ODU3NTA3IEM0MC40NzMzOTUzLDI4LjQ4ODcyOCA0MC43NjY4MiwyOC41MDQ4MTQ2IDQwLjk4NTM1NTIsMjguNjk0OTQ1MSBDNDEuMTAwODIxNSwyOC43OTU0MDM0IDQxLjE4MTk5OTksMjguOTUxMzY4NCA0MS4yMDUxNzc2LDI5LjE3NDgzMiBMNDEuMjA1MTc3NiwyOS4xNzQ4MzIgTDQxLjIxMjIzNCw2OC42NTE5OTczIEM0MS4yMTIyMzQsNjguOTMwOTgzMyA0MS4xMzg1Mjc3LDY5LjEyNTgzNjkgNDEuMDEzMjgwMyw2OS4yNDk4MDk0IEM0MC43OTQzMDc0LDY5LjQ2NjU1MzUgNDAuNDc4MDQ3LDY5LjQ4MTg3ODcgNDAuMzU5NDEzOSw2OS40ODQ5MTA2IEw0MC4zNTk0MTM5LDY5LjQ4NDkxMDYgTDM4LjI1OTg1Myw2OS40ODUyNDk4IEwzOC4yNTk4NTMsMjguNDg1NDExNCBaIiBpZD0icm93c190d28iIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzMy43MTIyMzQsIDQ4Ljk4NTMzMSkgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMzMuNzEyMjM0LCAtNDguOTg1MzMxKSAiIC8+CiAgICAgICAgICAgIDxnIGlkPSJ2ZXJ0aWNhbF9lcXVhbCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzAuMDAwMDAwLCAyOS4zODA1NzApIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yNy4wOTMxODYzLC0yLjk4NTMzMDY1IEMyNi44ODM2NjI1LC0yLjk4NTMzMDY1IDI1LjcxMjIzNCwtMi45MzUzMzA2NSAyNS43MTIyMzQsLTEuNjUxOTk3MzEgTDI1LjcxMjIzNCwzNy42ODEzMzYgQzI1LjcxMjIzNCwzOC45NjQ2Njk0IDI2Ljg4MzY2MjUsMzkuMDE0NjY5NCAyNy4wOTMxODYzLDM5LjAxNDY2OTQgTDI5LjY2NDYxNDksMzkuMDE0NjY5NCBMMjkuNjY0NjE0OSwtMi45ODUzMzA2NSBMMjcuMDkzMTg2MywtMi45ODUzMzA2NSBaIE0zNS42NzIxNzcxLC0yLjk4NTMzMDY1IEwzNS42NzIxNzcxLDM5LjAxNDY2OTQgTDMxLjc1MjI5MDgsMzkuMDE0NjY5NCBMMzEuNzUyMjkwOCwzOS4wMTQ2Njk0IEwzMS43NTIyOTA4LC0yLjk4NTMzMDY1IEwzMS43NTIyOTA4LC0yLjk4NTMzMDY1IEwzNS42NzIxNzcxLC0yLjk4NTMzMDY1IFogTTQwLjM2NTYxNDksLTIuOTg0OTA5NiBDNDAuNjQ0ODc4NiwtMi45NzgyMzY5OCA0MS43MTIyMzQsLTIuODc2OTk3MzEgNDEuNzEyMjM0LC0xLjY1MTk5NzMxIEw0MS43MTIyMzQsLTEuNjUxOTk3MzEgTDQxLjcxMjIzNCwzNy42ODEzMzYgQzQxLjcxMjIzNCwzOC45NjQ2Njk0IDQwLjU0MDgwNTQsMzkuMDE0NjY5NCA0MC4zMzEyODE2LDM5LjAxNDY2OTQgTDQwLjMzMTI4MTYsMzkuMDE0NjY5NCBMMzcuNzU5ODUzLDM5LjAxNDY2OTQgTDM3Ljc1OTg1MywtMi45ODUzMzA2NSBaIiBpZD0icm93c19vbmUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuNzEyMjM0LCAxOC4wMTQ2NjkpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTMzLjcxMjIzNCwgLTE4LjAxNDY2OSkgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

row_count_match()


<p>Table should have exactly 13 rows.</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">13</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


Each validation method produces one step in the validation report above. When combined, these steps form a complete validation plan that systematically checks different aspects of your data quality.


# Common Arguments

Most validation methods in Pointblank share a set of common arguments that provide consistency and flexibility across different validation types:

- `columns=`: specifies which column(s) to validate (used in column-based validations)
- `pre=`: allows data transformation before validation
- `segments=`: enables validation across different data subsets
- `thresholds=`: sets acceptable failure thresholds
- `actions=`: defines actions to take when validations fail
- `brief=`: provides a description of what the validation is checking
- `active=`: determines if the validation step should be executed (default is `True`)
- `na_pass=`: controls how missing values are handled (only for column value validation methods)

For column validation methods, the `na_pass=` parameter determines whether missing values (Null/None/NA) should pass validation (this parameter is covered in a later section).

These arguments follow a consistent pattern across validation methods, so you don't need to memorize different parameter sets for each function. This systematic approach makes Pointblank more intuitive to work with as you build increasingly complex validation plans.

We'll cover most of these common arguments in their own dedicated sections later in the **User Guide**, as some of them represent a deeper topic worthy of focused attention.


# 1. Column Value Validations

These methods check individual values within columns against specific criteria:

- **Comparison checks** (<a href="../../reference/Validate.col_vals_gt.html#pointblank.Validate.col_vals_gt" class="gdls-link"><code>Validate.col_vals_gt()</code></a>, <a href="../../reference/Validate.col_vals_lt.html#pointblank.Validate.col_vals_lt" class="gdls-link"><code>Validate.col_vals_lt()</code></a>, etc.) for comparing values to thresholds or other columns

- **Range checks** (<a href="../../reference/Validate.col_vals_between.html#pointblank.Validate.col_vals_between" class="gdls-link"><code>Validate.col_vals_between()</code></a>, <a href="../../reference/Validate.col_vals_outside.html#pointblank.Validate.col_vals_outside" class="gdls-link"><code>Validate.col_vals_outside()</code></a>) for verifying that values fall within or outside specific ranges

- **Set membership checks** (<a href="../../reference/Validate.col_vals_in_set.html#pointblank.Validate.col_vals_in_set" class="gdls-link"><code>Validate.col_vals_in_set()</code></a>, <a href="../../reference/Validate.col_vals_not_in_set.html#pointblank.Validate.col_vals_not_in_set" class="gdls-link"><code>Validate.col_vals_not_in_set()</code></a>) for validating values against predefined sets

- **Null value checks** (<a href="../../reference/Validate.col_vals_null.html#pointblank.Validate.col_vals_null" class="gdls-link"><code>Validate.col_vals_null()</code></a>, <a href="../../reference/Validate.col_vals_not_null.html#pointblank.Validate.col_vals_not_null" class="gdls-link"><code>Validate.col_vals_not_null()</code></a>) for testing presence or absence of null values

- **Pattern matching checks** (<a href="../../reference/Validate.col_vals_regex.html#pointblank.Validate.col_vals_regex" class="gdls-link"><code>Validate.col_vals_regex()</code></a>, <a href="../../reference/Validate.col_vals_within_spec.html#pointblank.Validate.col_vals_within_spec" class="gdls-link"><code>Validate.col_vals_within_spec()</code></a>) for validating text patterns with regular expressions or against standard specifications

- **Trending value checks** (<a href="../../reference/Validate.col_vals_increasing.html#pointblank.Validate.col_vals_increasing" class="gdls-link"><code>Validate.col_vals_increasing()</code></a>, <a href="../../reference/Validate.col_vals_decreasing.html#pointblank.Validate.col_vals_decreasing" class="gdls-link"><code>Validate.col_vals_decreasing()</code></a>) for verifying that values increase or decrease as you move down the rows

- **Custom expression checks** (<a href="../../reference/Validate.col_vals_expr.html#pointblank.Validate.col_vals_expr" class="gdls-link"><code>Validate.col_vals_expr()</code></a>) for complex validations using custom expressions

Now let's look at some key examples from select categories of column value validations.


## Comparison Checks

Let's start with a simple example of how <a href="../../reference/Validate.col_vals_gt.html#pointblank.Validate.col_vals_gt" class="gdls-link"><code>Validate.col_vals_gt()</code></a> might be used to check if the values in a column are greater than a specified value.


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_vals_gt(columns="a", value=5)
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
MzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">5</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">3<br />
0.23</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">10<br />
0.77</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


If you're checking data in a column that contains Null/`None`/`NA` values and you'd like to disregard those values (i.e., let them pass validation), you can use `na_pass=True`. The following example checks values in column `c` of `small_table`, which contains two `None` values:


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_vals_le(columns="c", value=10, na_pass=True)
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19sZTwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19sZSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNTEsMTAgTDEzLDEwIEMxMS4zNDc2NTYsMTAgMTAsMTEuMzQ3NjU2IDEwLDEzIEwxMCw1MSBDMTAsNTIuNjUyMzQ0IDExLjM0NzY1Niw1NCAxMyw1NCBMNTEsNTQgQzUyLjY1MjM0NCw1NCA1NCw1Mi42NTIzNDQgNTQsNTEgTDU0LDEzIEM1NCwxMS4zNDc2NTYgNTIuNjUyMzQ0LDEwIDUxLDEwIFogTTQwLjUxNTYyNSwxOC4xMjUgTDQxLjQ4NDM3NSwxOS44NzUgTDI1LjA1ODU5NCwyOSBMNDEuNDg0Mzc1LDM4LjEyNSBMNDAuNTE1NjI1LDM5Ljg3NSBMMjAuOTQxNDA2LDI5IEw0MC41MTU2MjUsMTguMTI1IFogTTQzLDQ2IEwyMSw0NiBMMjEsNDQgTDQzLDQ0IEw0Myw0NiBaIiBpZD0ibGVzc190aGFuX2VxdWFsIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_le()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">10</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">13<br />
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


In the above validation table, we see that all test units passed. If we didn't use `na_pass=True` there would be 2 failing test units, one for each `None` value in the `c` column.

It's possible to check against column values against values in an adjacent column. To do this, supply the `value=` argument with the column name within the [col()](../../reference/col.md#pointblank.col) helper function. Here's an example of that:


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_vals_lt(columns="a", value=pb.col("c"))
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19sdDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19sdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNTEsMTAgTDEzLDEwIEMxMS4zNDc2NTYsMTAgMTAsMTEuMzQ3NjU2IDEwLDEzIEwxMCw1MSBDMTAsNTIuNjUyMzQ0IDExLjM0NzY1Niw1NCAxMyw1NCBMNTEsNTQgQzUyLjY1MjM0NCw1NCA1NCw1Mi42NTIzNDQgNTQsNTEgTDU0LDEzIEM1NCwxMS4zNDc2NTYgNTIuNjUyMzQ0LDEwIDUxLDEwIFogTTM4LjcwNzAzMSw0NS4yOTI5NjkgTDM3LjI5Mjk2OSw0Ni43MDcwMzEgTDIyLjU4NTkzOCwzMiBMMzcuMjkyOTY5LDE3LjI5Mjk2OSBMMzguNzA3MDMxLDE4LjcwNzAzMSBMMjUuNDE0MDYzLDMyIEwzOC43MDcwMzEsNDUuMjkyOTY5IFoiIGlkPSJsZXNzX3RoYW4iIGZpbGw9IiMwMDAwMDAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_lt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">6<br />
0.46</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">7<br />
0.54</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


This validation checks that values in column `a` are less than values in column `c`.


## Checking of Missing Values

A very common thing to validate is that there are no Null/NA/missing values in a column. The <a href="../../reference/Validate.col_vals_not_null.html#pointblank.Validate.col_vals_not_null" class="gdls-link"><code>Validate.col_vals_not_null()</code></a> method checks for the presence of missing values:


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_vals_not_null(columns="a")
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0ib21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0ibGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">13<br />
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


Column `a` has no missing values and the above validation proves this.


## Checking Percentage of Missing Values

While <a href="../../reference/Validate.col_vals_not_null.html#pointblank.Validate.col_vals_not_null" class="gdls-link"><code>Validate.col_vals_not_null()</code></a> ensures there are no missing values at all, sometimes you need to validate that missing values match a specific percentage. The <a href="../../reference/Validate.col_pct_null.html#pointblank.Validate.col_pct_null" class="gdls-link"><code>Validate.col_pct_null()</code></a> method checks whether the percentage of missing values in a column matches an expected value:


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_pct_null(columns="c", p=0.15, tol=0.05)  # Expect ~15% missing values (±5%)
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5wY3RfbnVsbDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19udWxsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxLjAwMDAwMCwgMS41ODE3MTcpIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICA8cGF0aCBkPSJNNTUsMCBDNTcuNDg1MjgxMywwIDU5LjczNTI4MTMsMS4wMDczNTkzMSA2MS4zNjM5NjEsMi42MzYwMzg5NyBDNjIuOTkyNjQwNyw0LjI2NDcxODYzIDY0LDYuNTE0NzE4NjMgNjQsOSBMNjQsOSBMNjQsNjQgTDksNjQgQzYuNTE0NzE4NjIsNjQgNC4yNjQ3MTg2Miw2Mi45OTI2NDA3IDIuNjM2MDM4OTcsNjEuMzYzOTYxIEMxLjAwNzM1OTMxLDU5LjczNTI4MTQgMCw1Ny40ODUyODE0IDAsNTUgTDAsNTUgTDAsOSBDMCw2LjUxNDcxODYzIDEuMDA3MzU5MzEsNC4yNjQ3MTg2MyAyLjYzNjAzODk3LDIuNjM2MDM4OTcgQzQuMjY0NzE4NjIsMS4wMDczNTkzMSA2LjUxNDcxODYyLDAgOSwwIEw5LDAgTDU1LDAgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NS40MTQ2MjI4LDQxLjUgQzQzLjI4NzcyNTQsNDEuNSA0MS42MjUyMjU0LDQwLjc1MzU3MTUgNDAuNzQ1MjAzLDM5LjQ1MzY4MzIgQzM5Ljg2NTE4MDUsNDAuNzUzNTcxNSAzOC4yMDI2ODA1LDQxLjUgMzYuMDc1NzgzNSw0MS41IEMzMy4yNjM5NTI5LDQxLjUgMjguODAyMzQ1OCwzOS4wNjc3NDU0IDI4LjgwMjM0NTgsMzIuNTQyODU3MiBDMjguODAyMzQ1OCwyOS41MjMyMTQzIDI5Ljc1NDQ2NjIsMjIuNSAzOC41NzM3NzQ0LDIyLjUgQzM4Ljg3Mjc2OTgsMjIuNSAzOS4xMTY2MzE1LDIyLjc0Mzg2MTcgMzkuMTE2NjMxNSwyMy4wNDI4NTcxIEwzOS4xMTY2MzE1LDI0LjEyODU3MTQgQzM5LjExNjYzMTUsMjQuNDI3NTY2OCAzOC44NzI3Njk4LDI0LjY3MTQyODYgMzguNTczNzc0NCwyNC42NzE0Mjg2IEMzNS45NTkxNTM5LDI0LjY3MTQyODYgMzMuMTQ1MjAyOSwyNy4xMzU0OTEzIDMzLjE0NTIwMjksMzIuNTQyODU3MiBDMzMuMTQ1MjAyOSwzNS4wNTE0NTEgMzQuMDEyNTAxOSwzOC43ODU3MTQzIDM2LjQwMjM0NTgsMzguNzg1NzE0MyBDMzguMDk0NTMzNiwzOC43ODU3MTQzIDM5LjA2MTQ5NzgsMzcuMDIzNTQ5IDM5LjY1MzEyNjksMzUuNTc3MzQzOSBDMzguNTc1ODk0OCwzMy43NjY0MDYxIDM4LjAzMDkxNzIsMzIuMDY5OTc3NSAzOC4wMzA5MTcyLDMwLjUzNDcwOTcgQzM4LjAzMDkxNzIsMjcuMDA4MjU5MiAzOS40OTE5NjYyLDI1Ljc1NzE0MjkgNDAuNzQ1MjAzLDI1Ljc1NzE0MjkgQzQxLjk5ODQzOTcsMjUuNzU3MTQyOSA0My40NTk0ODg2LDI3LjAwODI1OTIgNDMuNDU5NDg4NiwzMC41MzQ3MDk3IEM0My40NTk0ODg2LDMyLjA5MTE4MzIgNDIuOTI5MzU1LDMzLjc0NTIwMSA0MS44MzkzOTk0LDM1LjU3NzM0MzkgQzQyLjcxOTQyMTgsMzcuNzA2MzYxNyA0My44MTE0OTc5LDM4Ljc4NTcxNDMgNDUuMDg4MDYwMSwzOC43ODU3MTQzIEM0Ny40Nzc5MDQsMzguNzg1NzE0MyA0OC4zNDUyMDMsMzUuMDUxNDUxIDQ4LjM0NTIwMywzMi41NDI4NTcyIEM0OC4zNDUyMDMsMjcuMTM1NDkxMyA0NS41MzEyNTE5LDI0LjY3MTQyODYgNDIuOTE2NjMxNSwyNC42NzE0Mjg2IEM0Mi42MTU1MTU3LDI0LjY3MTQyODYgNDIuMzczNzc0NCwyNC40Mjc1NjY4IDQyLjM3Mzc3NDQsMjQuMTI4NTcxNCBMNDIuMzczNzc0NCwyMy4wNDI4NTcxIEM0Mi4zNzM3NzQ0LDIyLjc0Mzg2MTcgNDIuNjE1NTE1NywyMi41IDQyLjkxNjYzMTUsMjIuNSBDNTEuNzM1OTM5NywyMi41IDUyLjY4ODA2MDEsMjkuNTIzMjE0MyA1Mi42ODgwNjAxLDMyLjU0Mjg1NzIgQzUyLjY4ODA2MDEsMzkuMDY3NzQ1NCA0OC4yMjY0NTI5LDQxLjUgNDUuNDE0NjIyOCw0MS41IFoiIGlkPSJvbWVnYSIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICAgICAgPGcgaWQ9InBlcmNlbnQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDExLjI2ODUwOCwgMjMuODU0MzczKSIgZmlsbD0iIzAwMDAwMCI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS44OTkyMDU1MywxNy4yMDM3OTg4IEMxLjc4NzIwNTUzLDE3LjEyMzc5ODggMS42NjQsMTcuMDI5NjI3NCAxLjU4NCwxNi44OTM2Mjc0IEMxLjUwNCwxNi43NTc2Mjc0IDEuNDY0LDE2LjYwOTYyNzQgMS40NjQsMTYuNDQ5NjI3NCBDMS40NjQsMTYuMjQxNjI3NCAxLjUzNiwxNi4wMzM2Mjc0IDEuNjgsMTUuODI1NjI3NCBMMTIuMjQsMC40ODk2Mjc0MzQgQzEyLjQzMiwwLjE4NTYyNzQzNCAxMi41OTQyNjYyLDAgMTIuOTQ2MjY2MiwwIEMxMy4xNTQyNjYyLDAgMTMuNDcyLDAuMDg5NjI3NDM0IDEzLjY4LDAuMjAxNjI3NDM0IEMxNC4wNDgsMC40MjU2Mjc0MzQgMTQuMjMyLDAuNjgxNjI3NDM0IDE0LjIzMiwwLjk2OTYyNzQzNCBDMTQuMjMyLDEuMTQ1NjI3NDMgMTQuMTYsMS4zNDU2Mjc0MyAxNC4wMTYsMS41Njk2Mjc0MyBMMy40MzIsMTYuOTc3NjI3NCBDMy4xNzYsMTcuMjgxNjI3NCAyLjg4LDE3LjQzMzYyNzQgMi41NDQsMTcuNDMzNjI3NCBDMi4zMzYsMTcuNDMzNjI3NCAyLjEzOTIwNTUzLDE3LjM0Nzc5ODggMS44OTkyMDU1MywxNy4yMDM3OTg4IFogTTMuODY0LDcuNDczNjI3NDMgQzMuMTc2LDcuNDczNjI3NDMgMi41MzYsNy4zMDU2Mjc0MyAxLjk0NCw2Ljk2OTYyNzQzIEMxLjM1Miw2LjYzMzYyNzQzIDAuODgsNi4xODE2Mjc0MyAwLjUyOCw1LjYxMzYyNzQzIEMwLjE3Niw1LjA0NTYyNzQzIDAsNC40MjU2Mjc0MyAwLDMuNzUzNjI3NDMgQzAsMy4wODE2Mjc0MyAwLjE3MiwyLjQ2MTYyNzQzIDAuNTE2LDEuODkzNjI3NDMgQzAuODYsMS4zMjU2Mjc0MyAxLjMyOCwwLjg3NzYyNzQzNCAxLjkyLDAuNTQ5NjI3NDM0IEMyLjUxMiwwLjIyMTYyNzQzNCAzLjE2LDAuMDU3NjI3NDM0IDMuODY0LDAuMDU3NjI3NDM0IEM0LjU2OCwwLjA1NzYyNzQzNCA1LjIxNiwwLjIyMTYyNzQzNCA1LjgwOCwwLjU0OTYyNzQzNCBDNi40LDAuODc3NjI3NDM0IDYuODY0LDEuMzI1NjI3NDMgNy4yLDEuODkzNjI3NDMgQzcuNTM2LDIuNDYxNjI3NDMgNy43MDQsMy4wODE2Mjc0MyA3LjcwNCwzLjc1MzYyNzQzIEM3LjcwNCw0LjQyNTYyNzQzIDcuNTMyLDUuMDQ1NjI3NDMgNy4xODgsNS42MTM2Mjc0MyBDNi44NDQsNi4xODE2Mjc0MyA2LjM4LDYuNjMzNjI3NDMgNS43OTYsNi45Njk2Mjc0MyBDNS4yMTIsNy4zMDU2Mjc0MyA0LjU2OCw3LjQ3MzYyNzQzIDMuODY0LDcuNDczNjI3NDMgWiBNMy44NjQsNS42OTc2Mjc0MyBDNC40MDgsNS42OTc2Mjc0MyA0Ljg1Miw1LjUxMzYyNzQzIDUuMTk2LDUuMTQ1NjI3NDMgQzUuNTQsNC43Nzc2Mjc0MyA1LjcxMiw0LjMxMzYyNzQzIDUuNzEyLDMuNzUzNjI3NDMgQzUuNzEyLDMuMTc3NjI3NDMgNS41NCwyLjcwNTYyNzQzIDUuMTk2LDIuMzM3NjI3NDMgQzQuODUyLDEuOTY5NjI3NDMgNC40MDgsMS43ODU2Mjc0MyAzLjg2NCwxLjc4NTYyNzQzIEMzLjMwNCwxLjc4NTYyNzQzIDIuODQ4LDEuOTY5NjI3NDMgMi40OTYsMi4zMzc2Mjc0MyBDMi4xNDQsMi43MDU2Mjc0MyAxLjk2OCwzLjE3NzYyNzQzIDEuOTY4LDMuNzUzNjI3NDMgQzEuOTY4LDQuMzEzNjI3NDMgMi4xNDQsNC43Nzc2Mjc0MyAyLjQ5Niw1LjE0NTYyNzQzIEMyLjg0OCw1LjUxMzYyNzQzIDMuMzA0LDUuNjk3NjI3NDMgMy44NjQsNS42OTc2Mjc0MyBaIE0xMS45NTIsMTcuMzg1NjI3NCBDMTEuMjQ4LDE3LjM4NTYyNzQgMTAuNiwxNy4yMTc2Mjc0IDEwLjAwOCwxNi44ODE2Mjc0IEM5LjQxNiwxNi41NDU2Mjc0IDguOTQ4LDE2LjA5MzYyNzQgOC42MDQsMTUuNTI1NjI3NCBDOC4yNiwxNC45NTc2Mjc0IDguMDg4LDE0LjMzNzYyNzQgOC4wODgsMTMuNjY1NjI3NCBDOC4wODgsMTIuOTkzNjI3NCA4LjI2LDEyLjM3MzYyNzQgOC42MDQsMTEuODA1NjI3NCBDOC45NDgsMTEuMjM3NjI3NCA5LjQxNiwxMC43ODk2Mjc0IDEwLjAwOCwxMC40NjE2Mjc0IEMxMC42LDEwLjEzMzYyNzQgMTEuMjQ4LDkuOTY5NjI3NDMgMTEuOTUyLDkuOTY5NjI3NDMgQzEyLjY1Niw5Ljk2OTYyNzQzIDEzLjMsMTAuMTMzNjI3NCAxMy44ODQsMTAuNDYxNjI3NCBDMTQuNDY4LDEwLjc4OTYyNzQgMTQuOTI4LDExLjIzNzYyNzQgMTUuMjY0LDExLjgwNTYyNzQgQzE1LjYsMTIuMzczNjI3NCAxNS43NjgsMTIuOTkzNjI3NCAxNS43NjgsMTMuNjY1NjI3NCBDMTUuNzY4LDE0LjMzNzYyNzQgMTUuNTk2LDE0Ljk1NzYyNzQgMTUuMjUyLDE1LjUyNTYyNzQgQzE0LjkwOCwxNi4wOTM2Mjc0IDE0LjQ0NCwxNi41NDU2Mjc0IDEzLjg2LDE2Ljg4MTYyNzQgQzEzLjI3NiwxNy4yMTc2Mjc0IDEyLjY0LDE3LjM4NTYyNzQgMTEuOTUyLDE3LjM4NTYyNzQgWiBNMTEuOTUyLDE1LjYwOTYyNzQgQzEyLjQ4LDE1LjYwOTYyNzQgMTIuOTIsMTUuNDI5NjI3NCAxMy4yNzIsMTUuMDY5NjI3NCBDMTMuNjI0LDE0LjcwOTYyNzQgMTMuOCwxNC4yNDE2Mjc0IDEzLjgsMTMuNjY1NjI3NCBDMTMuOCwxMy4xMDU2Mjc0IDEzLjYyNCwxMi42NDE2Mjc0IDEzLjI3MiwxMi4yNzM2Mjc0IEMxMi45MiwxMS45MDU2Mjc0IDEyLjQ4LDExLjcyMTYyNzQgMTEuOTUyLDExLjcyMTYyNzQgQzExLjM5MiwxMS43MjE2Mjc0IDEwLjkzMiwxMS45MDU2Mjc0IDEwLjU3MiwxMi4yNzM2Mjc0IEMxMC4yMTIsMTIuNjQxNjI3NCAxMC4wMzIsMTMuMTA1NjI3NCAxMC4wMzIsMTMuNjY1NjI3NCBDMTAuMDMyLDE0LjI0MTYyNzQgMTAuMjEyLDE0LjcwOTYyNzQgMTAuNTcyLDE1LjA2OTYyNzQgQzEwLjkzMiwxNS40Mjk2Mjc0IDExLjM5MiwxNS42MDk2Mjc0IDExLjk1MiwxNS42MDk2Mjc0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_pct_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">p = 0.15<br />
tol = 0.05</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


This validation checks that approximately 15% of values in column `c` are missing, allowing a tolerance of ±5% (so the acceptable range is 10-20%). The `tol=` parameter can accept various formats including absolute counts or percentage ranges:


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_pct_null(columns="c", p=0.15, tol=(0.05, 0.10))  # Asymmetric tolerance: -5%/+10%
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5wY3RfbnVsbDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19udWxsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxLjAwMDAwMCwgMS41ODE3MTcpIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICA8cGF0aCBkPSJNNTUsMCBDNTcuNDg1MjgxMywwIDU5LjczNTI4MTMsMS4wMDczNTkzMSA2MS4zNjM5NjEsMi42MzYwMzg5NyBDNjIuOTkyNjQwNyw0LjI2NDcxODYzIDY0LDYuNTE0NzE4NjMgNjQsOSBMNjQsOSBMNjQsNjQgTDksNjQgQzYuNTE0NzE4NjIsNjQgNC4yNjQ3MTg2Miw2Mi45OTI2NDA3IDIuNjM2MDM4OTcsNjEuMzYzOTYxIEMxLjAwNzM1OTMxLDU5LjczNTI4MTQgMCw1Ny40ODUyODE0IDAsNTUgTDAsNTUgTDAsOSBDMCw2LjUxNDcxODYzIDEuMDA3MzU5MzEsNC4yNjQ3MTg2MyAyLjYzNjAzODk3LDIuNjM2MDM4OTcgQzQuMjY0NzE4NjIsMS4wMDczNTkzMSA2LjUxNDcxODYyLDAgOSwwIEw5LDAgTDU1LDAgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NS40MTQ2MjI4LDQxLjUgQzQzLjI4NzcyNTQsNDEuNSA0MS42MjUyMjU0LDQwLjc1MzU3MTUgNDAuNzQ1MjAzLDM5LjQ1MzY4MzIgQzM5Ljg2NTE4MDUsNDAuNzUzNTcxNSAzOC4yMDI2ODA1LDQxLjUgMzYuMDc1NzgzNSw0MS41IEMzMy4yNjM5NTI5LDQxLjUgMjguODAyMzQ1OCwzOS4wNjc3NDU0IDI4LjgwMjM0NTgsMzIuNTQyODU3MiBDMjguODAyMzQ1OCwyOS41MjMyMTQzIDI5Ljc1NDQ2NjIsMjIuNSAzOC41NzM3NzQ0LDIyLjUgQzM4Ljg3Mjc2OTgsMjIuNSAzOS4xMTY2MzE1LDIyLjc0Mzg2MTcgMzkuMTE2NjMxNSwyMy4wNDI4NTcxIEwzOS4xMTY2MzE1LDI0LjEyODU3MTQgQzM5LjExNjYzMTUsMjQuNDI3NTY2OCAzOC44NzI3Njk4LDI0LjY3MTQyODYgMzguNTczNzc0NCwyNC42NzE0Mjg2IEMzNS45NTkxNTM5LDI0LjY3MTQyODYgMzMuMTQ1MjAyOSwyNy4xMzU0OTEzIDMzLjE0NTIwMjksMzIuNTQyODU3MiBDMzMuMTQ1MjAyOSwzNS4wNTE0NTEgMzQuMDEyNTAxOSwzOC43ODU3MTQzIDM2LjQwMjM0NTgsMzguNzg1NzE0MyBDMzguMDk0NTMzNiwzOC43ODU3MTQzIDM5LjA2MTQ5NzgsMzcuMDIzNTQ5IDM5LjY1MzEyNjksMzUuNTc3MzQzOSBDMzguNTc1ODk0OCwzMy43NjY0MDYxIDM4LjAzMDkxNzIsMzIuMDY5OTc3NSAzOC4wMzA5MTcyLDMwLjUzNDcwOTcgQzM4LjAzMDkxNzIsMjcuMDA4MjU5MiAzOS40OTE5NjYyLDI1Ljc1NzE0MjkgNDAuNzQ1MjAzLDI1Ljc1NzE0MjkgQzQxLjk5ODQzOTcsMjUuNzU3MTQyOSA0My40NTk0ODg2LDI3LjAwODI1OTIgNDMuNDU5NDg4NiwzMC41MzQ3MDk3IEM0My40NTk0ODg2LDMyLjA5MTE4MzIgNDIuOTI5MzU1LDMzLjc0NTIwMSA0MS44MzkzOTk0LDM1LjU3NzM0MzkgQzQyLjcxOTQyMTgsMzcuNzA2MzYxNyA0My44MTE0OTc5LDM4Ljc4NTcxNDMgNDUuMDg4MDYwMSwzOC43ODU3MTQzIEM0Ny40Nzc5MDQsMzguNzg1NzE0MyA0OC4zNDUyMDMsMzUuMDUxNDUxIDQ4LjM0NTIwMywzMi41NDI4NTcyIEM0OC4zNDUyMDMsMjcuMTM1NDkxMyA0NS41MzEyNTE5LDI0LjY3MTQyODYgNDIuOTE2NjMxNSwyNC42NzE0Mjg2IEM0Mi42MTU1MTU3LDI0LjY3MTQyODYgNDIuMzczNzc0NCwyNC40Mjc1NjY4IDQyLjM3Mzc3NDQsMjQuMTI4NTcxNCBMNDIuMzczNzc0NCwyMy4wNDI4NTcxIEM0Mi4zNzM3NzQ0LDIyLjc0Mzg2MTcgNDIuNjE1NTE1NywyMi41IDQyLjkxNjYzMTUsMjIuNSBDNTEuNzM1OTM5NywyMi41IDUyLjY4ODA2MDEsMjkuNTIzMjE0MyA1Mi42ODgwNjAxLDMyLjU0Mjg1NzIgQzUyLjY4ODA2MDEsMzkuMDY3NzQ1NCA0OC4yMjY0NTI5LDQxLjUgNDUuNDE0NjIyOCw0MS41IFoiIGlkPSJvbWVnYSIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICAgICAgPGcgaWQ9InBlcmNlbnQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDExLjI2ODUwOCwgMjMuODU0MzczKSIgZmlsbD0iIzAwMDAwMCI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS44OTkyMDU1MywxNy4yMDM3OTg4IEMxLjc4NzIwNTUzLDE3LjEyMzc5ODggMS42NjQsMTcuMDI5NjI3NCAxLjU4NCwxNi44OTM2Mjc0IEMxLjUwNCwxNi43NTc2Mjc0IDEuNDY0LDE2LjYwOTYyNzQgMS40NjQsMTYuNDQ5NjI3NCBDMS40NjQsMTYuMjQxNjI3NCAxLjUzNiwxNi4wMzM2Mjc0IDEuNjgsMTUuODI1NjI3NCBMMTIuMjQsMC40ODk2Mjc0MzQgQzEyLjQzMiwwLjE4NTYyNzQzNCAxMi41OTQyNjYyLDAgMTIuOTQ2MjY2MiwwIEMxMy4xNTQyNjYyLDAgMTMuNDcyLDAuMDg5NjI3NDM0IDEzLjY4LDAuMjAxNjI3NDM0IEMxNC4wNDgsMC40MjU2Mjc0MzQgMTQuMjMyLDAuNjgxNjI3NDM0IDE0LjIzMiwwLjk2OTYyNzQzNCBDMTQuMjMyLDEuMTQ1NjI3NDMgMTQuMTYsMS4zNDU2Mjc0MyAxNC4wMTYsMS41Njk2Mjc0MyBMMy40MzIsMTYuOTc3NjI3NCBDMy4xNzYsMTcuMjgxNjI3NCAyLjg4LDE3LjQzMzYyNzQgMi41NDQsMTcuNDMzNjI3NCBDMi4zMzYsMTcuNDMzNjI3NCAyLjEzOTIwNTUzLDE3LjM0Nzc5ODggMS44OTkyMDU1MywxNy4yMDM3OTg4IFogTTMuODY0LDcuNDczNjI3NDMgQzMuMTc2LDcuNDczNjI3NDMgMi41MzYsNy4zMDU2Mjc0MyAxLjk0NCw2Ljk2OTYyNzQzIEMxLjM1Miw2LjYzMzYyNzQzIDAuODgsNi4xODE2Mjc0MyAwLjUyOCw1LjYxMzYyNzQzIEMwLjE3Niw1LjA0NTYyNzQzIDAsNC40MjU2Mjc0MyAwLDMuNzUzNjI3NDMgQzAsMy4wODE2Mjc0MyAwLjE3MiwyLjQ2MTYyNzQzIDAuNTE2LDEuODkzNjI3NDMgQzAuODYsMS4zMjU2Mjc0MyAxLjMyOCwwLjg3NzYyNzQzNCAxLjkyLDAuNTQ5NjI3NDM0IEMyLjUxMiwwLjIyMTYyNzQzNCAzLjE2LDAuMDU3NjI3NDM0IDMuODY0LDAuMDU3NjI3NDM0IEM0LjU2OCwwLjA1NzYyNzQzNCA1LjIxNiwwLjIyMTYyNzQzNCA1LjgwOCwwLjU0OTYyNzQzNCBDNi40LDAuODc3NjI3NDM0IDYuODY0LDEuMzI1NjI3NDMgNy4yLDEuODkzNjI3NDMgQzcuNTM2LDIuNDYxNjI3NDMgNy43MDQsMy4wODE2Mjc0MyA3LjcwNCwzLjc1MzYyNzQzIEM3LjcwNCw0LjQyNTYyNzQzIDcuNTMyLDUuMDQ1NjI3NDMgNy4xODgsNS42MTM2Mjc0MyBDNi44NDQsNi4xODE2Mjc0MyA2LjM4LDYuNjMzNjI3NDMgNS43OTYsNi45Njk2Mjc0MyBDNS4yMTIsNy4zMDU2Mjc0MyA0LjU2OCw3LjQ3MzYyNzQzIDMuODY0LDcuNDczNjI3NDMgWiBNMy44NjQsNS42OTc2Mjc0MyBDNC40MDgsNS42OTc2Mjc0MyA0Ljg1Miw1LjUxMzYyNzQzIDUuMTk2LDUuMTQ1NjI3NDMgQzUuNTQsNC43Nzc2Mjc0MyA1LjcxMiw0LjMxMzYyNzQzIDUuNzEyLDMuNzUzNjI3NDMgQzUuNzEyLDMuMTc3NjI3NDMgNS41NCwyLjcwNTYyNzQzIDUuMTk2LDIuMzM3NjI3NDMgQzQuODUyLDEuOTY5NjI3NDMgNC40MDgsMS43ODU2Mjc0MyAzLjg2NCwxLjc4NTYyNzQzIEMzLjMwNCwxLjc4NTYyNzQzIDIuODQ4LDEuOTY5NjI3NDMgMi40OTYsMi4zMzc2Mjc0MyBDMi4xNDQsMi43MDU2Mjc0MyAxLjk2OCwzLjE3NzYyNzQzIDEuOTY4LDMuNzUzNjI3NDMgQzEuOTY4LDQuMzEzNjI3NDMgMi4xNDQsNC43Nzc2Mjc0MyAyLjQ5Niw1LjE0NTYyNzQzIEMyLjg0OCw1LjUxMzYyNzQzIDMuMzA0LDUuNjk3NjI3NDMgMy44NjQsNS42OTc2Mjc0MyBaIE0xMS45NTIsMTcuMzg1NjI3NCBDMTEuMjQ4LDE3LjM4NTYyNzQgMTAuNiwxNy4yMTc2Mjc0IDEwLjAwOCwxNi44ODE2Mjc0IEM5LjQxNiwxNi41NDU2Mjc0IDguOTQ4LDE2LjA5MzYyNzQgOC42MDQsMTUuNTI1NjI3NCBDOC4yNiwxNC45NTc2Mjc0IDguMDg4LDE0LjMzNzYyNzQgOC4wODgsMTMuNjY1NjI3NCBDOC4wODgsMTIuOTkzNjI3NCA4LjI2LDEyLjM3MzYyNzQgOC42MDQsMTEuODA1NjI3NCBDOC45NDgsMTEuMjM3NjI3NCA5LjQxNiwxMC43ODk2Mjc0IDEwLjAwOCwxMC40NjE2Mjc0IEMxMC42LDEwLjEzMzYyNzQgMTEuMjQ4LDkuOTY5NjI3NDMgMTEuOTUyLDkuOTY5NjI3NDMgQzEyLjY1Niw5Ljk2OTYyNzQzIDEzLjMsMTAuMTMzNjI3NCAxMy44ODQsMTAuNDYxNjI3NCBDMTQuNDY4LDEwLjc4OTYyNzQgMTQuOTI4LDExLjIzNzYyNzQgMTUuMjY0LDExLjgwNTYyNzQgQzE1LjYsMTIuMzczNjI3NCAxNS43NjgsMTIuOTkzNjI3NCAxNS43NjgsMTMuNjY1NjI3NCBDMTUuNzY4LDE0LjMzNzYyNzQgMTUuNTk2LDE0Ljk1NzYyNzQgMTUuMjUyLDE1LjUyNTYyNzQgQzE0LjkwOCwxNi4wOTM2Mjc0IDE0LjQ0NCwxNi41NDU2Mjc0IDEzLjg2LDE2Ljg4MTYyNzQgQzEzLjI3NiwxNy4yMTc2Mjc0IDEyLjY0LDE3LjM4NTYyNzQgMTEuOTUyLDE3LjM4NTYyNzQgWiBNMTEuOTUyLDE1LjYwOTYyNzQgQzEyLjQ4LDE1LjYwOTYyNzQgMTIuOTIsMTUuNDI5NjI3NCAxMy4yNzIsMTUuMDY5NjI3NCBDMTMuNjI0LDE0LjcwOTYyNzQgMTMuOCwxNC4yNDE2Mjc0IDEzLjgsMTMuNjY1NjI3NCBDMTMuOCwxMy4xMDU2Mjc0IDEzLjYyNCwxMi42NDE2Mjc0IDEzLjI3MiwxMi4yNzM2Mjc0IEMxMi45MiwxMS45MDU2Mjc0IDEyLjQ4LDExLjcyMTYyNzQgMTEuOTUyLDExLjcyMTYyNzQgQzExLjM5MiwxMS43MjE2Mjc0IDEwLjkzMiwxMS45MDU2Mjc0IDEwLjU3MiwxMi4yNzM2Mjc0IEMxMC4yMTIsMTIuNjQxNjI3NCAxMC4wMzIsMTMuMTA1NjI3NCAxMC4wMzIsMTMuNjY1NjI3NCBDMTAuMDMyLDE0LjI0MTYyNzQgMTAuMjEyLDE0LjcwOTYyNzQgMTAuNTcyLDE1LjA2OTYyNzQgQzEwLjkzMiwxNS40Mjk2Mjc0IDExLjM5MiwxNS42MDk2Mjc0IDExLjk1MiwxNS42MDk2Mjc0IFoiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_pct_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">p = 0.15<br />
tol = (0.05, 0.1)</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


## Checking Strings with Regexes

A regular expression (regex) validation via the <a href="../../reference/Validate.col_vals_regex.html#pointblank.Validate.col_vals_regex" class="gdls-link"><code>Validate.col_vals_regex()</code></a> validation method checks if values in a column match a specified pattern. Here's an example with two validation steps, each checking text values in a column:


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_vals_regex(columns="b", pattern=r"^\d-[a-z]{3}-\d{3}$")
    .col_vals_regex(columns="f", pattern=r"high|low|mid")
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfcmVnZXg8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfcmVnZXgiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjAzNDQ4MykiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InJlZ2V4X3N5bWJvbHMiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4LjAwMDAwMCwgMTIuMDAwMDAwKSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjE3NDM0NTA4LDMzLjAxMzU4MiBDMS45NDg5NTMyOCwzMy4wMTM1ODIgMC4xMzgwMDY5MjMsMzQuODI0NTI4NCAwLjEzODAwNjkyMywzNy4wNDk5MjAyIEMwLjEzODAwNjkyMywzOS4yNzUzMTIgMS45NDg5NTMyOCw0MS4wODYyNTgzIDQuMTc0MzQ1MDgsNDEuMDg2MjU4MyBDNi4zOTk3MzY4OCw0MS4wODYyNTgzIDguMjEwNjgzMjQsMzkuMjc1MzEyIDguMjEwNjgzMjQsMzcuMDQ5OTIwMiBDOC4yMTA2ODMyNCwzNC44MjQ1Mjg0IDYuMzk5NzM2ODgsMzMuMDEzNTgyIDQuMTc0MzQ1MDgsMzMuMDEzNTgyIFoiIGlkPSJmdWxsX3N0b3AiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMjMuOTQ3OTcxOCwyMy4zMTc1NDAyIEwyMS41NjI4MjY0LDIzLjMxNzU0MDIgQzIxLjIzNDQwMzIsMjMuMzE3NTQwMiAyMC45NjY1NDAxLDIzLjA1MjAwNjcgMjAuOTY2NTQwMSwyMi43MjEyNTM4IEwyMC45NjY1NDAxLDE1LjEwMjI5NzkgTDE0LjM0NDUwMDQsMTguODg3MzE5MiBDMTQuMDYyNjYyMSwxOS4wNTAzNjYgMTMuNzAxNjI5MiwxOC45NTI1MzggMTMuNTM2MjUzMywxOC42NzA2OTkxIEwxMi4zNDM2ODA2LDE2LjY0NDI1NzUgQzEyLjI2MjE1NywxNi41MDY4MzIgMTIuMjM4ODY0MiwxNi4zNDM3ODUyIDEyLjI4MDc5MDksMTYuMTkwMDU0OSBDMTIuMzIwMzg3OSwxNi4wMzYzMjUxIDEyLjQyMDU0NTUsMTUuOTA1ODg3NCAxMi41NTc5NzEsMTUuODI2NjkyOSBMMTkuMTgwMDEwMSwxMS45ODgwOTk0IEwxMi41NTc5NzEsOC4xNTE4MzUxMSBDMTIuNDIwNTQ1NSw4LjA3MjY0MTEyIDEyLjMyMDM4NzksNy45Mzk4NzQzOSAxMi4yODA3OTA5LDcuNzg2MTQ0MDEgQzEyLjIzODg2NDIsNy42MzI0MTQyMyAxMi4yNjIxNTcsNy40NjkzNjY4OSAxMi4zNDEzNTA5LDcuMzMxOTQxMzcgTDEzLjUzMzkyMzcsNS4zMDU0OTk3NSBDMTMuNjk5MzAwMSw1LjAyMzY2MTQzIDE0LjA2MjY2MjEsNC45MjgxNjE5OSAxNC4zNDQ1MDA0LDUuMDkxMjA5MzQgTDIwLjk2NjU0MDEsOC44NzM5MDA5MSBMMjAuOTY2NTQwMSwxLjI1NDk0NTAxIEMyMC45NjY1NDAxLDAuOTI2NTIxODE4IDIxLjIzNDQwMzIsMC42NTg2NTg2NTggMjEuNTYyODI2NCwwLjY1ODY1ODY1OCBMMjMuOTQ3OTcxOCwwLjY1ODY1ODY1OCBDMjQuMjc4NzI0NywwLjY1ODY1ODY1OCAyNC41NDQyNTgyLDAuOTI2NTIxODE4IDI0LjU0NDI1ODIsMS4yNTQ5NDUwMSBMMjQuNTQ0MjU4Miw4Ljg3MzkwMDkxIEwzMS4xNjYyOTc5LDUuMDkxMjA5MzQgQzMxLjQ0ODEzNjIsNC45MjgxNjE5OSAzMS44MDkxNjkxLDUuMDIzNjYxNDMgMzEuOTc0NTQ1NSw1LjMwNTQ5OTc1IEwzMy4xNjcxMTgyLDcuMzMxOTQxMzcgQzMzLjI0ODY0MTMsNy40NjkzNjY4OSAzMy4yNzE5MzQxLDcuNjMyNDE0MjMgMzMuMjMwMDA3NCw3Ljc4NjE0NDAxIEMzMy4xOTA0MTA0LDcuOTM5ODc0MzkgMzMuMDkwMjUyOCw4LjA3MjY0MTEyIDMyLjk1MjgyNzgsOC4xNTE4MzUxMSBMMjYuMzMwNzg4MiwxMS45ODgwOTk0IEwzMi45NTI4Mjc4LDE1LjgyNDM2MzggQzMzLjA4NzkyMzcsMTUuOTA1ODg3NCAzMy4xODgwODEzLDE2LjAzNjMyNTEgMzMuMjMwMDA3NCwxNi4xOTAwNTQ5IEMzMy4yNjk2MDUsMTYuMzQzNzg1MiAzMy4yNDg2NDEzLDE2LjUwNjgzMiAzMy4xNjcxMTgyLDE2LjY0NDI1NzUgTDMxLjk3NDU0NTUsMTguNjcwNjk5MSBDMzEuODA5MTY5MSwxOC45NTI1MzggMzEuNDQ4MTM2MiwxOS4wNTAzNjYgMzEuMTY2Mjk3OSwxOC44ODQ5ODk1IEwyNC41NDQyNTgyLDE1LjEwMjI5NzkgTDI0LjU0NDI1ODIsMjIuNzIxMjUzOCBDMjQuNTQ0MjU4MiwyMy4wNTIwMDY3IDI0LjI3ODcyNDcsMjMuMzE3NTQwMiAyMy45NDc5NzE4LDIzLjMxNzU0MDIgWiIgaWQ9ImFzdGVyaXNrIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_regex()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">^\d-[a-z]{3}-\d{3}$</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">13<br />
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfcmVnZXg8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfcmVnZXgiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjAzNDQ4MykiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InJlZ2V4X3N5bWJvbHMiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4LjAwMDAwMCwgMTIuMDAwMDAwKSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjE3NDM0NTA4LDMzLjAxMzU4MiBDMS45NDg5NTMyOCwzMy4wMTM1ODIgMC4xMzgwMDY5MjMsMzQuODI0NTI4NCAwLjEzODAwNjkyMywzNy4wNDk5MjAyIEMwLjEzODAwNjkyMywzOS4yNzUzMTIgMS45NDg5NTMyOCw0MS4wODYyNTgzIDQuMTc0MzQ1MDgsNDEuMDg2MjU4MyBDNi4zOTk3MzY4OCw0MS4wODYyNTgzIDguMjEwNjgzMjQsMzkuMjc1MzEyIDguMjEwNjgzMjQsMzcuMDQ5OTIwMiBDOC4yMTA2ODMyNCwzNC44MjQ1Mjg0IDYuMzk5NzM2ODgsMzMuMDEzNTgyIDQuMTc0MzQ1MDgsMzMuMDEzNTgyIFoiIGlkPSJmdWxsX3N0b3AiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMjMuOTQ3OTcxOCwyMy4zMTc1NDAyIEwyMS41NjI4MjY0LDIzLjMxNzU0MDIgQzIxLjIzNDQwMzIsMjMuMzE3NTQwMiAyMC45NjY1NDAxLDIzLjA1MjAwNjcgMjAuOTY2NTQwMSwyMi43MjEyNTM4IEwyMC45NjY1NDAxLDE1LjEwMjI5NzkgTDE0LjM0NDUwMDQsMTguODg3MzE5MiBDMTQuMDYyNjYyMSwxOS4wNTAzNjYgMTMuNzAxNjI5MiwxOC45NTI1MzggMTMuNTM2MjUzMywxOC42NzA2OTkxIEwxMi4zNDM2ODA2LDE2LjY0NDI1NzUgQzEyLjI2MjE1NywxNi41MDY4MzIgMTIuMjM4ODY0MiwxNi4zNDM3ODUyIDEyLjI4MDc5MDksMTYuMTkwMDU0OSBDMTIuMzIwMzg3OSwxNi4wMzYzMjUxIDEyLjQyMDU0NTUsMTUuOTA1ODg3NCAxMi41NTc5NzEsMTUuODI2NjkyOSBMMTkuMTgwMDEwMSwxMS45ODgwOTk0IEwxMi41NTc5NzEsOC4xNTE4MzUxMSBDMTIuNDIwNTQ1NSw4LjA3MjY0MTEyIDEyLjMyMDM4NzksNy45Mzk4NzQzOSAxMi4yODA3OTA5LDcuNzg2MTQ0MDEgQzEyLjIzODg2NDIsNy42MzI0MTQyMyAxMi4yNjIxNTcsNy40NjkzNjY4OSAxMi4zNDEzNTA5LDcuMzMxOTQxMzcgTDEzLjUzMzkyMzcsNS4zMDU0OTk3NSBDMTMuNjk5MzAwMSw1LjAyMzY2MTQzIDE0LjA2MjY2MjEsNC45MjgxNjE5OSAxNC4zNDQ1MDA0LDUuMDkxMjA5MzQgTDIwLjk2NjU0MDEsOC44NzM5MDA5MSBMMjAuOTY2NTQwMSwxLjI1NDk0NTAxIEMyMC45NjY1NDAxLDAuOTI2NTIxODE4IDIxLjIzNDQwMzIsMC42NTg2NTg2NTggMjEuNTYyODI2NCwwLjY1ODY1ODY1OCBMMjMuOTQ3OTcxOCwwLjY1ODY1ODY1OCBDMjQuMjc4NzI0NywwLjY1ODY1ODY1OCAyNC41NDQyNTgyLDAuOTI2NTIxODE4IDI0LjU0NDI1ODIsMS4yNTQ5NDUwMSBMMjQuNTQ0MjU4Miw4Ljg3MzkwMDkxIEwzMS4xNjYyOTc5LDUuMDkxMjA5MzQgQzMxLjQ0ODEzNjIsNC45MjgxNjE5OSAzMS44MDkxNjkxLDUuMDIzNjYxNDMgMzEuOTc0NTQ1NSw1LjMwNTQ5OTc1IEwzMy4xNjcxMTgyLDcuMzMxOTQxMzcgQzMzLjI0ODY0MTMsNy40NjkzNjY4OSAzMy4yNzE5MzQxLDcuNjMyNDE0MjMgMzMuMjMwMDA3NCw3Ljc4NjE0NDAxIEMzMy4xOTA0MTA0LDcuOTM5ODc0MzkgMzMuMDkwMjUyOCw4LjA3MjY0MTEyIDMyLjk1MjgyNzgsOC4xNTE4MzUxMSBMMjYuMzMwNzg4MiwxMS45ODgwOTk0IEwzMi45NTI4Mjc4LDE1LjgyNDM2MzggQzMzLjA4NzkyMzcsMTUuOTA1ODg3NCAzMy4xODgwODEzLDE2LjAzNjMyNTEgMzMuMjMwMDA3NCwxNi4xOTAwNTQ5IEMzMy4yNjk2MDUsMTYuMzQzNzg1MiAzMy4yNDg2NDEzLDE2LjUwNjgzMiAzMy4xNjcxMTgyLDE2LjY0NDI1NzUgTDMxLjk3NDU0NTUsMTguNjcwNjk5MSBDMzEuODA5MTY5MSwxOC45NTI1MzggMzEuNDQ4MTM2MiwxOS4wNTAzNjYgMzEuMTY2Mjk3OSwxOC44ODQ5ODk1IEwyNC41NDQyNTgyLDE1LjEwMjI5NzkgTDI0LjU0NDI1ODIsMjIuNzIxMjUzOCBDMjQuNTQ0MjU4MiwyMy4wNTIwMDY3IDI0LjI3ODcyNDcsMjMuMzE3NTQwMiAyMy45NDc5NzE4LDIzLjMxNzU0MDIgWiIgaWQ9ImFzdGVyaXNrIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_regex()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">f</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">high|low|mid</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">13<br />
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


## Checking Strings Against Specifications

The <a href="../../reference/Validate.col_vals_within_spec.html#pointblank.Validate.col_vals_within_spec" class="gdls-link"><code>Validate.col_vals_within_spec()</code></a> method validates column values against common data specifications like email addresses, URLs, postal codes, credit card numbers, ISBNs, VINs, and IBANs. This is particularly useful when you need to validate that text data conforms to standard formats:


``` python
import polars as pl

# Create a sample table with various data types
sample_data = pl.DataFrame({
    "isbn": ["978-0-306-40615-7", "0-306-40615-2", "invalid"],
    "email": ["test@example.com", "user@domain.co.uk", "not-an-email"],
    "zip": ["12345", "90210", "invalid"]
})

(
    pb.Validate(data=sample_data)
    .col_vals_within_spec(columns="isbn", spec="isbn")
    .col_vals_within_spec(columns="email", spec="email")
    .col_vals_within_spec(columns="zip", spec="postal_code[US]")
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
jgwOTEzMiwzNi44Mjc5MDIgMTMuNzgzMjAzLDM2LjI5Njg3NSBaIE0zNC4yMTA5MzgsMzYuMzEyNSBDMzYuMTc0MDY0LDM2Ljg0MTcyMSAzNy45NDk0MTMsMzcuNTA3NjM5IDM5LjQ4MjQyMiwzOC4yODEyNSBDMzYuOTAyNDk5LDQxLjA3NzI4NSAzMy41ODg5MjcsNDMuMTgxOTg5IDI5LjgyODEyNSw0NC4yNTk3NjYgQzMwLjkxMzUxOSw0My4xMjI3MTYgMzEuODgxMzYzLDQxLjcwMDE2NiAzMi43MDExNzIsNDAuMDYwNTQ3IEMzMy4yNzI0OTgsMzguOTE3ODk0IDMzLjc3MzYzNCwzNy42NTcyOSAzNC4yMTA5MzgsMzYuMzEyNSBaIiBpZD0iU2hhcGUiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_within_spec()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">isbn</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">isbn</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">3</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.67</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
0.33</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfd2l0aGluX3NwZWM8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfd2l0aGluX3NwZWMiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjUxNzI0MSkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9Imdsb2JlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg5LjcxMjIzNCwgOS4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTI0LDAuOTE5OTIxOSBDMTEuMjY1MTEzLDAuOTE5OTIxOSAwLjkxOTkyMTksMTEuMjY1MTEzIDAuOTE5OTIxOSwyNCBDMC45MTk5MjE5LDM2LjczNDg4NyAxMS4yNjUxMTMsNDcuMDgwMDc4IDI0LDQ3LjA4MDA3OCBDMzYuNzM0ODg3LDQ3LjA4MDA3OCA0Ny4wODAwNzgsMzYuNzM0ODg3IDQ3LjA4MDA3OCwyNCBDNDcuMDgwMDc4LDExLjI2NTExMyAzNi43MzQ4ODcsMC45MTk5MjE5IDI0LDAuOTE5OTIxOSBaIE0yMywzLjAzNzEwOTQgTDIzLDEyLjk3MDcwMyBDMjAuNDMyNTEsMTIuOTEwMTEgMTcuOTkxNDQ1LDEyLjYyMzAyMiAxNS43NDAyMzQsMTIuMTUyMzQ0IEMxNi4xMzY2MjcsMTAuOTUxNSAxNi41ODcxMDMsOS44MzU1NTkgMTcuMDg3ODkxLDguODMzOTg0NCBDMTguNzQwODI1LDUuNTI4MTE1NiAyMC44MzcyODYsMy41MTYwNDk4IDIzLDMuMDM3MTA5NCBaIE0yNSwzLjAzNzEwOTQgQzI3LjE2MjcxNCwzLjUxNjA0OTggMjkuMjU5MTc1LDUuNTI4MTE1NiAzMC45MTIxMDksOC44MzM5ODQ0IEMzMS40MTQ0OTYsOS44Mzg3NTcgMzEuODY2Mzc5LDEwLjk1ODgwNiAzMi4yNjM2NzIsMTIuMTY0MDYyIEMzMC4wMTUyNjksMTIuNjMwMDM3IDI3LjU3MDksMTIuOTExMzc3IDI1LDEyLjk3MDcwMyBMMjUsMy4wMzcxMDk0IFogTTE4LjE3MTg3NSwzLjc0MDIzNDQgQzE3LjA4NjQ4MSw0Ljg3NzI4NDUgMTYuMTE4NjM3LDYuMjk5ODM0NCAxNS4yOTg4MjgsNy45Mzk0NTMxIEMxNC43Mjc1MDIsOS4wODIxMDYgMTQuMjI2MzY2LDEwLjM0MjcxIDEzLjc4OTA2MiwxMS42ODc1IEMxMS44MjU5MzYsMTEuMTU4Mjc5IDEwLjA1MDU4NywxMC40OTIzNjEgOC41MTc1NzgxLDkuNzE4NzUgQzExLjA5NzUwMSw2LjkyMjcxNTEgMTQuNDExMDczLDQuODE4MDEwOSAxOC4xNzE4NzUsMy43NDAyMzQ0IFogTTI5LjgyODEyNSwzLjc0MDIzNDQgQzMzLjU4NTI4OSw0LjgxNjk2ODEgMzYuODk1NzM3LDYuOTE4OTYzNiAzOS40NzQ2MDksOS43MTA5MzggQzM3Ljk2NDI1LDEwLjQ5ODY2OCAzNi4xOTA4NjgsMTEuMTcyMDk4IDM0LjIxNjc5NywxMS43MDMxMjUgQzMzLjc3ODM1MywxMC4zNTI0MDkgMzMuMjc0NzEyLDkuMDg2NTM0IDMyLjcwMTE3Miw3LjkzOTQ1MzEgQzMxLjg4MTM2Myw2LjI5OTgzNDQgMzAuOTEzNTE5LDQuODc3Mjg0NSAyOS44MjgxMjUsMy43NDAyMzQ0IFogTTQwLjc4MzIwMywxMS4yNzM0MzggQzQzLjI4MDMxOSwxNC41NjMyNTQgNDQuODQ5NTkxLDE4LjU5NjU0NCA0NS4wNTQ2ODgsMjMgTDM2LjAxMzY3MiwyMyBDMzUuOTQwNjg2LDE5LjY0MjY5NyAzNS41MTE1ODEsMTYuNDcyODQzIDM0Ljc3NzM0NCwxMy42MzI4MTIgQzM3LjAyMTE2MiwxMy4wMjU3ODggMzkuMDQzNTY0LDEyLjIzMDM1NyA0MC43ODMyMDMsMTEuMjczNDM4IFogTTcuMjA1MDc4MSwxMS4yODkwNjIgQzguOTYzNTM2MiwxMi4yMjI3NTIgMTAuOTg5MzAxLDEzLjAwODc5IDEzLjIyNjU2MiwxMy42MTUyMzQgQzEyLjQ4OTYzMywxNi40NTk2NzEgMTIuMDU5NDYyLDE5LjYzNTkwNCAxMS45ODYzMjgsMjMgTDIuOTQ1MzEyNSwyMyBDMy4xNTAwODU2LDE4LjYwMzQ4NSA0LjcxNDg3MjcsMTQuNTc2MDc4IDcuMjA1MDc4MSwxMS4yODkwNjIgWiBNMTUuMTc1NzgxLDE0LjA4NTkzOCBDMTcuNjA4MTI0LDE0LjYwMzQ3OSAyMC4yMzcxNDUsMTQuOTExNjkyIDIzLDE0Ljk3MjY1NiBMMjMsMjMgTDEzLjk4NjMyOCwyMyBDMTQuMDYwNzI1LDE5Ljc4NzM2OSAxNC40ODA3NDMsMTYuNzYyMjcxIDE1LjE3NTc4MSwxNC4wODU5MzggWiBNMzIuODI4MTI1LDE0LjA5OTYwOSBDMzMuNTIxMDg4LDE2Ljc3MjYgMzMuOTM5NDAxLDE5Ljc5Mjc5NiAzNC4wMTM2NzIsMjMgTDI1LDIzIEwyNSwxNC45NzI2NTYgQzI3Ljc2NDQ1NywxNC45MTMzOTMgMzAuMzk2NDc3LDE0LjYxMjI3MSAzMi44MjgxMjUsMTQuMDk5NjA5IFogTTIuOTQ1MzEyNSwyNSBMMTEuOTg2MzI4LDI1IEMxMi4wNTkzMTQsMjguMzU3MzAzIDEyLjQ4ODQxOSwzMS41MjcxNTYgMTMuMjIyNjU2LDM0LjM2NzE4OCBDMTAuOTc4ODM4LDM0Ljk3NDIxMiA4Ljk1NjQzNjMsMzUuNzY5NjQzIDcuMjE2Nzk2OSwzNi43MjY1NjIgQzQuNzE5NjgwNiwzMy40MzY3NDYgMy4xNTA0MDg4LDI5LjQwMzQ1NiAyLjk0NTMxMjUsMjUgWiBNMTMuOTg2MzI4LDI1IEwyMywyNSBMMjMsMzMuMDI3MzQ0IEMyMC4yMzU1NDMsMzMuMDg2NjA3IDE3LjYwMzUyMywzMy4zODc3MjkgMTUuMTcxODc1LDMzLjkwMDM5MSBDMTQuNDc4OTEyLDMxLjIyNzQgMTQuMDYwNTk5LDI4LjIwNzIwNCAxMy45ODYzMjgsMjUgWiBNMjUsMjUgTDM0LjAxMzY3MiwyNSBDMzMuOTM5Mjc1LDI4LjIxMjYzMSAzMy41MTkyNTcsMzEuMjM3NzI5IDMyLjgyNDIxOSwzMy45MTQwNjIgQzMwLjM5MTg3NiwzMy4zOTY1MjEgMjcuNzYyODU1LDMzLjA4ODMwOCAyNSwzMy4wMjczNDQgTDI1LDI1IFogTTM2LjAxMzY3MiwyNSBMNDUuMDU0Njg4LDI1IEM0NC44NDk5MTQsMjkuMzk2NTE1IDQzLjI4NTEyNywzMy40MjM5MjIgNDAuNzk0OTIyLDM2LjcxMDkzOCBDMzkuMDM2NDY0LDM1Ljc3NzI0OCAzNy4wMTA2OTksMzQuOTkxMjEgMzQuNzczNDM4LDM0LjM4NDc2NiBDMzUuNTEwMzY3LDMxLjU0MDMyOSAzNS45NDA1MzgsMjguMzY0MDk2IDM2LjAxMzY3MiwyNSBaIE0yMywzNS4wMjkyOTcgTDIzLDQ0Ljk2Mjg5MSBDMjAuODM3Mjg2LDQ0LjQ4Mzk1IDE4Ljc0MDgyNSw0Mi40NzE4ODQgMTcuMDg3ODkxLDM5LjE2NjAxNiBDMTYuNTg1NTA0LDM4LjE2MTI0MyAxNi4xMzM2MjEsMzcuMDQxMTk0IDE1LjczNjMyOCwzNS44MzU5MzggQzE3Ljk4NDczMSwzNS4zNjk5NjMgMjAuNDI5MSwzNS4wODg2MjMgMjMsMzUuMDI5Mjk3IFogTTI1LDM1LjAyOTI5NyBDMjcuNTY3NDksMzUuMDg5ODkgMzAuMDA4NTU1LDM1LjM3Njk3OCAzMi4yNTk3NjYsMzUuODQ3NjU2IEMzMS44NjMzNzMsMzcuMDQ4NSAzMS40MTI4OTcsMzguMTY0NDQgMzAuOTEyMTA5LDM5LjE2NjAxNiBDMjkuMjU5MTc1LDQyLjQ3MTg4NCAyNy4xNjI3MTQsNDQuNDgzOTUgMjUsNDQuOTYyODkxIEwyNSwzNS4wMjkyOTcgWiBNMTMuNzgzMjAzLDM2LjI5Njg3NSBDMTQuMjIxNjQ3LDM3LjY0NzU5MSAxNC43MjUyODgsMzguOTEzNDY2IDE1LjI5ODgyOCw0MC4wNjA1NDcgQzE2LjExODYzNyw0MS43MDAxNjYgMTcuMDg2NDgxLDQzLjEyMjcxNiAxOC4xNzE4NzUsNDQuMjU5NzY2IEMxNC40MTQ3MTEsNDMuMTgzMDMyIDExLjEwNDI2Myw0MS4wODEwMzYgOC41MjUzOTA2LDM4LjI4OTA2MiBDMTAuMDM1NzUsMzcuNTAxMzMyIDExLjgwOTEzMiwzNi44Mjc5MDIgMTMuNzgzMjAzLDM2LjI5Njg3NSBaIE0zNC4yMTA5MzgsMzYuMzEyNSBDMzYuMTc0MDY0LDM2Ljg0MTcyMSAzNy45NDk0MTMsMzcuNTA3NjM5IDM5LjQ4MjQyMiwzOC4yODEyNSBDMzYuOTAyNDk5LDQxLjA3NzI4NSAzMy41ODg5MjcsNDMuMTgxOTg5IDI5LjgyODEyNSw0NC4yNTk3NjYgQzMwLjkxMzUxOSw0My4xMjI3MTYgMzEuODgxMzYzLDQxLjcwMDE2NiAzMi43MDExNzIsNDAuMDYwNTQ3IEMzMy4yNzI0OTgsMzguOTE3ODk0IDMzLjc3MzYzNCwzNy42NTcyOSAzNC4yMTA5MzgsMzYuMzEyNSBaIiBpZD0iU2hhcGUiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_within_spec()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">email</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">email</td>
4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">3</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.67</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
0.33</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">3</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfd2l0aGluX3NwZWM8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfd2l0aGluX3NwZWMiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjUxNzI0MSkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9Imdsb2JlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg5LjcxMjIzNCwgOS4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTI0LDAuOTE5OTIxOSBDMTEuMjY1MTEzLDAuOTE5OTIxOSAwLjkxOTkyMTksMTEuMjY1MTEzIDAuOTE5OTIxOSwyNCBDMC45MTk5MjE5LDM2LjczNDg4NyAxMS4yNjUxMTMsNDcuMDgwMDc4IDI0LDQ3LjA4MDA3OCBDMzYuNzM0ODg3LDQ3LjA4MDA3OCA0Ny4wODAwNzgsMzYuNzM0ODg3IDQ3LjA4MDA3OCwyNCBDNDcuMDgwMDc4LDExLjI2NTExMyAzNi43MzQ4ODcsMC45MTk5MjE5IDI0LDAuOTE5OTIxOSBaIE0yMywzLjAzNzEwOTQgTDIzLDEyLjk3MDcwMyBDMjAuNDMyNTEsMTIuOTEwMTEgMTcuOTkxNDQ1LDEyLjYyMzAyMiAxNS43NDAyMzQsMTIuMTUyMzQ0IEMxNi4xMzY2MjcsMTAuOTUxNSAxNi41ODcxMDMsOS44MzU1NTkgMTcuMDg3ODkxLDguODMzOTg0NCBDMTguNzQwODI1LDUuNTI4MTE1NiAyMC44MzcyODYsMy41MTYwNDk4IDIzLDMuMDM3MTA5NCBaIE0yNSwzLjAzNzEwOTQgQzI3LjE2MjcxNCwzLjUxNjA0OTggMjkuMjU5MTc1LDUuNTI4MTE1NiAzMC45MTIxMDksOC44MzM5ODQ0IEMzMS40MTQ0OTYsOS44Mzg3NTcgMzEuODY2Mzc5LDEwLjk1ODgwNiAzMi4yNjM2NzIsMTIuMTY0MDYyIEMzMC4wMTUyNjksMTIuNjMwMDM3IDI3LjU3MDksMTIuOTExMzc3IDI1LDEyLjk3MDcwMyBMMjUsMy4wMzcxMDk0IFogTTE4LjE3MTg3NSwzLjc0MDIzNDQgQzE3LjA4NjQ4MSw0Ljg3NzI4NDUgMTYuMTE4NjM3LDYuMjk5ODM0NCAxNS4yOTg4MjgsNy45Mzk0NTMxIEMxNC43Mjc1MDIsOS4wODIxMDYgMTQuMjI2MzY2LDEwLjM0MjcxIDEzLjc4OTA2MiwxMS42ODc1IEMxMS44MjU5MzYsMTEuMTU4Mjc5IDEwLjA1MDU4NywxMC40OTIzNjEgOC41MTc1NzgxLDkuNzE4NzUgQzExLjA5NzUwMSw2LjkyMjcxNTEgMTQuNDExMDczLDQuODE4MDEwOSAxOC4xNzE4NzUsMy43NDAyMzQ0IFogTTI5LjgyODEyNSwzLjc0MDIzNDQgQzMzLjU4NTI4OSw0LjgxNjk2ODEgMzYuODk1NzM3LDYuOTE4OTYzNiAzOS40NzQ2MDksOS43MTA5MzggQzM3Ljk2NDI1LDEwLjQ5ODY2OCAzNi4xOTA4NjgsMTEuMTcyMDk4IDM0LjIxNjc5NywxMS43MDMxMjUgQzMzLjc3ODM1MywxMC4zNTI0MDkgMzMuMjc0NzEyLDkuMDg2NTM0IDMyLjcwMTE3Miw3LjkzOTQ1MzEgQzMxLjg4MTM2Myw2LjI5OTgzNDQgMzAuOTEzNTE5LDQuODc3Mjg0NSAyOS44MjgxMjUsMy43NDAyMzQ0IFogTTQwLjc4MzIwMywxMS4yNzM0MzggQzQzLjI4MDMxOSwxNC41NjMyNTQgNDQuODQ5NTkxLDE4LjU5NjU0NCA0NS4wNTQ2ODgsMjMgTDM2LjAxMzY3MiwyMyBDMzUuOTQwNjg2LDE5LjY0MjY5NyAzNS41MTE1ODEsMTYuNDcyODQzIDM0Ljc3NzM0NCwxMy42MzI4MTIgQzM3LjAyMTE2MiwxMy4wMjU3ODggMzkuMDQzNTY0LDEyLjIzMDM1NyA0MC43ODMyMDMsMTEuMjczNDM4IFogTTcuMjA1MDc4MSwxMS4yODkwNjIgQzguOTYzNTM2MiwxMi4yMjI3NTIgMTAuOTg5MzAxLDEzLjAwODc5IDEzLjIyNjU2MiwxMy42MTUyMzQgQzEyLjQ4OTYzMywxNi40NTk2NzEgMTIuMDU5NDYyLDE5LjYzNTkwNCAxMS45ODYzMjgsMjMgTDIuOTQ1MzEyNSwyMyBDMy4xNTAwODU2LDE4LjYwMzQ4NSA0LjcxNDg3MjcsMTQuNTc2MDc4IDcuMjA1MDc4MSwxMS4yODkwNjIgWiBNMTUuMTc1NzgxLDE0LjA4NTkzOCBDMTcuNjA4MTI0LDE0LjYwMzQ3OSAyMC4yMzcxNDUsMTQuOTExNjkyIDIzLDE0Ljk3MjY1NiBMMjMsMjMgTDEzLjk4NjMyOCwyMyBDMTQuMDYwNzI1LDE5Ljc4NzM2OSAxNC40ODA3NDMsMTYuNzYyMjcxIDE1LjE3NTc4MSwxNC4wODU5MzggWiBNMzIuODI4MTI1LDE0LjA5OTYwOSBDMzMuNTIxMDg4LDE2Ljc3MjYgMzMuOTM5NDAxLDE5Ljc5Mjc5NiAzNC4wMTM2NzIsMjMgTDI1LDIzIEwyNSwxNC45NzI2NTYgQzI3Ljc2NDQ1NywxNC45MTMzOTMgMzAuMzk2NDc3LDE0LjYxMjI3MSAzMi44MjgxMjUsMTQuMDk5NjA5IFogTTIuOTQ1MzEyNSwyNSBMMTEuOTg2MzI4LDI1IEMxMi4wNTkzMTQsMjguMzU3MzAzIDEyLjQ4ODQxOSwzMS41MjcxNTYgMTMuMjIyNjU2LDM0LjM2NzE4OCBDMTAuOTc4ODM4LDM0Ljk3NDIxMiA4Ljk1NjQzNjMsMzUuNzY5NjQzIDcuMjE2Nzk2OSwzNi43MjY1NjIgQzQuNzE5NjgwNiwzMy40MzY3NDYgMy4xNTA0MDg4LDI5LjQwMzQ1NiAyLjk0NTMxMjUsMjUgWiBNMTMuOTg2MzI4LDI1IEwyMywyNSBMMjMsMzMuMDI3MzQ0IEMyMC4yMzU1NDMsMzMuMDg2NjA3IDE3LjYwMzUyMywzMy4zODc3MjkgMTUuMTcxODc1LDMzLjkwMDM5MSBDMTQuNDc4OTEyLDMxLjIyNzQgMTQuMDYwNTk5LDI4LjIwNzIwNCAxMy45ODYzMjgsMjUgWiBNMjUsMjUgTDM0LjAxMzY3MiwyNSBDMzMuOTM5Mjc1LDI4LjIxMjYzMSAzMy41MTkyNTcsMzEuMjM3NzI5IDMyLjgyNDIxOSwzMy45MTQwNjIgQzMwLjM5MTg3NiwzMy4zOTY1MjEgMjcuNzYyODU1LDMzLjA4ODMwOCAyNSwzMy4wMjczNDQgTDI1LDI1IFogTTM2LjAxMzY3MiwyNSBMNDUuMDU0Njg4LDI1IEM0NC44NDk5MTQsMjkuMzk2NTE1IDQzLjI4NTEyNywzMy40MjM5MjIgNDAuNzk0OTIyLDM2LjcxMDkzOCBDMzkuMDM2NDY0LDM1Ljc3NzI0OCAzNy4wMTA2OTksMzQuOTkxMjEgMzQuNzczNDM4LDM0LjM4NDc2NiBDMzUuNTEwMzY3LDMxLjU0MDMyOSAzNS45NDA1MzgsMjguMzY0MDk2IDM2LjAxMzY3MiwyNSBaIE0yMywzNS4wMjkyOTcgTDIzLDQ0Ljk2Mjg5MSBDMjAuODM3Mjg2LDQ0LjQ4Mzk1IDE4Ljc0MDgyNSw0Mi40NzE4ODQgMTcuMDg3ODkxLDM5LjE2NjAxNiBDMTYuNTg1NTA0LDM4LjE2MTI0MyAxNi4xMzM2MjEsMzcuMDQxMTk0IDE1LjczNjMyOCwzNS44MzU5MzggQzE3Ljk4NDczMSwzNS4zNjk5NjMgMjAuNDI5MSwzNS4wODg2MjMgMjMsMzUuMDI5Mjk3IFogTTI1LDM1LjAyOTI5NyBDMjcuNTY3NDksMzUuMDg5ODkgMzAuMDA4NTU1LDM1LjM3Njk3OCAzMi4yNTk3NjYsMzUuODQ3NjU2IEMzMS44NjMzNzMsMzcuMDQ4NSAzMS40MTI4OTcsMzguMTY0NDQgMzAuOTEyMTA5LDM5LjE2NjAxNiBDMjkuMjU5MTc1LDQyLjQ3MTg4NCAyNy4xNjI3MTQsNDQuNDgzOTUgMjUsNDQuOTYyODkxIEwyNSwzNS4wMjkyOTcgWiBNMTMuNzgzMjAzLDM2LjI5Njg3NSBDMTQuMjIxNjQ3LDM3LjY0NzU5MSAxNC43MjUyODgsMzguOTEzNDY2IDE1LjI5ODgyOCw0MC4wNjA1NDcgQzE2LjExODYzNyw0MS43MDAxNjYgMTcuMDg2NDgxLDQzLjEyMjcxNiAxOC4xNzE4NzUsNDQuMjU5NzY2IEMxNC40MTQ3MTEsNDMuMTgzMDMyIDExLjEwNDI2Myw0MS4wODEwMzYgOC41MjUzOTA2LDM4LjI4OTA2MiBDMTAuMDM1NzUsMzcuNTAxMzMyIDExLjgwOTEzMiwzNi44Mjc5MDIgMTMuNzgzMjAzLDM2LjI5Njg3NSBaIE0zNC4yMTA5MzgsMzYuMzEyNSBDMzYuMTc0MDY0LDM2Ljg0MTcyMSAzNy45NDk0MTMsMzcuNTA3NjM5IDM5LjQ4MjQyMiwzOC4yODEyNSBDMzYuOTAyNDk5LDQxLjA3NzI4NSAzMy41ODg5MjcsNDMuMTgxOTg5IDI5LjgyODEyNSw0NC4yNTk3NjYgQzMwLjkxMzUxOSw0My4xMjI3MTYgMzEuODgxMzYzLDQxLjcwMDE2NiAzMi43MDExNzIsNDAuMDYwNTQ3IEMzMy4yNzI0OTgsMzguOTE3ODk0IDMzLjc3MzYzNCwzNy42NTcyOSAzNC4yMTA5MzgsMzYuMzEyNSBaIiBpZD0iU2hhcGUiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_within_spec()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">zip</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">postal_code[US]</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">3</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.67</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
0.33</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


## Checking for Trending Values

The <a href="../../reference/Validate.col_vals_increasing.html#pointblank.Validate.col_vals_increasing" class="gdls-link"><code>Validate.col_vals_increasing()</code></a> and <a href="../../reference/Validate.col_vals_decreasing.html#pointblank.Validate.col_vals_decreasing" class="gdls-link"><code>Validate.col_vals_decreasing()</code></a> validation methods check whether column values are increasing or decreasing as you move down the rows. These are useful for validating time series data, sequential identifiers, or any data where you expect monotonic trends:


``` python
import polars as pl

# Create a sample table with increasing and decreasing values
trend_data = pl.DataFrame({
    "id": [1, 2, 3, 4, 5],
    "temperature": [20, 22, 25, 28, 30],
    "countdown": [100, 80, 60, 40, 20]
})

(
    pb.Validate(data=trend_data)
    .col_vals_increasing(columns="id")
    .col_vals_increasing(columns="temperature")
    .col_vals_decreasing(columns="countdown")
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfaW5jcmVhc2luZzwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19pbmNyZWFzaW5nIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4xMDM0NDgpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJpbmNyZWFzaW5nIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMy43MTIyMzQsIDEzLjAwMDAwMCkiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMzEuMiwwIEwzMS4yLDEuNiBMMzcuMjUsMS42IEwyNy4yLDExLjY1IEwyNC41NzUsOS4wMjUgTDI0LDguNDc1IEwyMy40MjUsOS4wMjUgTDE0LjQsMTguMDUgTDExLjc3NSwxNS40MjUgTDExLjIsMTQuODc1IEwxMC42MjUsMTUuNDI1IEwwLjIyNSwyNS44MjUgTDEuMzc1LDI2Ljk3NSBMMTEuMiwxNy4xNSBMMTMuODI1LDE5Ljc3NSBMMTQuNCwyMC4zMjUgTDE0Ljk3NSwxOS43NzUgTDI0LDEwLjc1IEwyNi42MjUsMTMuMzc1IEwyNy4yLDEzLjkyNSBMMjcuNzc1LDEzLjM3NSBMMzguNCwyLjc1IEwzOC40LDguOCBMNDAsOC44IEw0MCwwIEwzMS4yLDAgWiBNMzMuNiwxMS4yIEwzMy42LDQwIEwzNS4yLDQwIEwzNS4yLDExLjIgTDMzLjYsMTEuMiBaIE0zOC40LDEyIEwzOC40LDQwIEw0MCw0MCBMNDAsMTIgTDM4LjQsMTIgWiBNMjQsMTYgTDI0LDQwIEwyNS42LDQwIEwyNS42LDE2IEwyNCwxNiBaIE0yOC44LDE2IEwyOC44LDQwIEwzMC40LDQwIEwzMC40LDE2IEwyOC44LDE2IFogTTE5LjIsMTkuMiBMMTkuMiw0MCBMMjAuOCw0MCBMMjAuOCwxOS4yIEwxOS4yLDE5LjIgWiBNOS42LDIyLjQgTDkuNiw0MCBMMTEuMiw0MCBMMTEuMiwyMi40IEw5LjYsMjIuNCBaIE0xNC40LDI0IEwxNC40LDQwIEwxNiw0MCBMMTYsMjQgTDE0LjQsMjQgWiBNNC44LDI3LjIgTDQuOCw0MCBMNi40LDQwIEw2LjQsMjcuMiBMNC44LDI3LjIgWiBNMCwzMC40IEwwLDQwIEwxLjYsNDAgTDEuNiwzMC40IEwwLDMwLjQgWiIgaWQ9IlNoYXBlIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_increasing()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">id</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfaW5jcmVhc2luZzwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19pbmNyZWFzaW5nIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4xMDM0NDgpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJpbmNyZWFzaW5nIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMy43MTIyMzQsIDEzLjAwMDAwMCkiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMzEuMiwwIEwzMS4yLDEuNiBMMzcuMjUsMS42IEwyNy4yLDExLjY1IEwyNC41NzUsOS4wMjUgTDI0LDguNDc1IEwyMy40MjUsOS4wMjUgTDE0LjQsMTguMDUgTDExLjc3NSwxNS40MjUgTDExLjIsMTQuODc1IEwxMC42MjUsMTUuNDI1IEwwLjIyNSwyNS44MjUgTDEuMzc1LDI2Ljk3NSBMMTEuMiwxNy4xNSBMMTMuODI1LDE5Ljc3NSBMMTQuNCwyMC4zMjUgTDE0Ljk3NSwxOS43NzUgTDI0LDEwLjc1IEwyNi42MjUsMTMuMzc1IEwyNy4yLDEzLjkyNSBMMjcuNzc1LDEzLjM3NSBMMzguNCwyLjc1IEwzOC40LDguOCBMNDAsOC44IEw0MCwwIEwzMS4yLDAgWiBNMzMuNiwxMS4yIEwzMy42LDQwIEwzNS4yLDQwIEwzNS4yLDExLjIgTDMzLjYsMTEuMiBaIE0zOC40LDEyIEwzOC40LDQwIEw0MCw0MCBMNDAsMTIgTDM4LjQsMTIgWiBNMjQsMTYgTDI0LDQwIEwyNS42LDQwIEwyNS42LDE2IEwyNCwxNiBaIE0yOC44LDE2IEwyOC44LDQwIEwzMC40LDQwIEwzMC40LDE2IEwyOC44LDE2IFogTTE5LjIsMTkuMiBMMTkuMiw0MCBMMjAuOCw0MCBMMjAuOCwxOS4yIEwxOS4yLDE5LjIgWiBNOS42LDIyLjQgTDkuNiw0MCBMMTEuMiw0MCBMMTEuMiwyMi40IEw5LjYsMjIuNCBaIE0xNC40LDI0IEwxNC40LDQwIEwxNiw0MCBMMTYsMjQgTDE0LjQsMjQgWiBNNC44LDI3LjIgTDQuOCw0MCBMNi40LDQwIEw2LjQsMjcuMiBMNC44LDI3LjIgWiBNMCwzMC40IEwwLDQwIEwxLjYsNDAgTDEuNiwzMC40IEwwLDMwLjQgWiIgaWQ9IlNoYXBlIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_increasing()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">temperature</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfZGVjcmVhc2luZzwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19kZWNyZWFzaW5nIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC41ODYyMDcpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJkZWNyZWFzaW5nIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMy43MTIyMzQsIDEyLjUwMDAwMCkiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMS4zNzUsMC4yMjUgTDAuMjI1LDEuMzc1IEwxMi4yMjUsMTMuMzc1IEwxMi44LDEzLjkyNSBMMTMuMzc1LDEzLjM3NSBMMTYsMTAuNzUgTDI1LjAyNSwxOS43NzUgTDI1LjYsMjAuMzI1IEwyNi4xNzUsMTkuNzc1IEwyOC44LDE3LjE1IEwzNy4yNSwyNS42IEwzMiwyNS42IEwzMiwyNy4yIEw0MCwyNy4yIEw0MCwxOC40IEwzOC40LDE4LjQgTDM4LjQsMjQuNDUgTDI5LjM3NSwxNS40MjUgTDI4LjgsMTQuODc1IEwyOC4yMjUsMTUuNDI1IEwyNS42LDE4LjA1IEwxNi41NzUsOS4wMjUgTDE2LDguNDc1IEwxNS40MjUsOS4wMjUgTDEyLjgsMTEuNjUgTDEuMzc1LDAuMjI1IFogTTAsNi40IEwwLDQwIEwxLjYsNDAgTDEuNiw2LjQgTDAsNi40IFogTTQuOCwxMS4yIEw0LjgsNDAgTDYuNCw0MCBMNi40LDExLjIgTDQuOCwxMS4yIFogTTkuNiwxNiBMOS42LDQwIEwxMS4yLDQwIEwxMS4yLDE2IEw5LjYsMTYgWiBNMTQuNCwxNiBMMTQuNCw0MCBMMTYsNDAgTDE2LDE2IEwxNC40LDE2IFogTTE5LjIsMTkuMiBMMTkuMiw0MCBMMjAuOCw0MCBMMjAuOCwxOS4yIEwxOS4yLDE5LjIgWiBNMjguOCwyMi40IEwyOC44LDQwIEwzMC40LDQwIEwzMC40LDIyLjQgTDI4LjgsMjIuNCBaIE0yNCwyNCBMMjQsNDAgTDI1LjYsNDAgTDI1LjYsMjQgTDI0LDI0IFogTTMzLjYsMzAuNCBMMzMuNiw0MCBMMzUuMiw0MCBMMzUuMiwzMC40IEwzMy42LDMwLjQgWiBNMzguNCwzMC40IEwzOC40LDQwIEw0MCw0MCBMNDAsMzAuNCBMMzguNCwzMC40IFoiIGlkPSJTaGFwZSIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_decreasing()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">countdown</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
yNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


The `allow_stationary=` parameter lets you control whether consecutive identical values should pass validation. By default, stationary values (e.g., `[1, 2, 2, 3]`) will fail the increasing check, but setting `allow_stationary=True` will allow them to pass.


## Handling Missing Values with `na_pass=`

When validating columns containing Null/None/NA values, you can control how these missing values are treated with the `na_pass=` parameter:


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_vals_le(columns="c", value=10, na_pass=True)
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19sZTwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19sZSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNTEsMTAgTDEzLDEwIEMxMS4zNDc2NTYsMTAgMTAsMTEuMzQ3NjU2IDEwLDEzIEwxMCw1MSBDMTAsNTIuNjUyMzQ0IDExLjM0NzY1Niw1NCAxMyw1NCBMNTEsNTQgQzUyLjY1MjM0NCw1NCA1NCw1Mi42NTIzNDQgNTQsNTEgTDU0LDEzIEM1NCwxMS4zNDc2NTYgNTIuNjUyMzQ0LDEwIDUxLDEwIFogTTQwLjUxNTYyNSwxOC4xMjUgTDQxLjQ4NDM3NSwxOS44NzUgTDI1LjA1ODU5NCwyOSBMNDEuNDg0Mzc1LDM4LjEyNSBMNDAuNTE1NjI1LDM5Ljg3NSBMMjAuOTQxNDA2LDI5IEw0MC41MTU2MjUsMTguMTI1IFogTTQzLDQ2IEwyMSw0NiBMMjEsNDQgTDQzLDQ0IEw0Myw0NiBaIiBpZD0ibGVzc190aGFuX2VxdWFsIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_le()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">10</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">13<br />
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


In the above example, column `c` contains two `None` values, but all test units pass because we set `na_pass=True`. Without this setting, those two values would fail the validation.

In summary, `na_pass=` works like this:

- `na_pass=True`: missing values pass validation regardless of the condition being tested
- `na_pass=False` (the default): missing values fail validation


# 2. Row-based Validations

Row-based validations focus on examining properties that span across entire rows rather than individual columns. These are essential for detecting issues that can't be found by looking at columns in isolation:

- <a href="../../reference/Validate.rows_distinct.html#pointblank.Validate.rows_distinct" class="gdls-link"><code>Validate.rows_distinct()</code></a>: ensures no duplicate rows exist in the table
- <a href="../../reference/Validate.rows_complete.html#pointblank.Validate.rows_complete" class="gdls-link"><code>Validate.rows_complete()</code></a>: verifies that no rows contain any missing values

These row-level validations are particularly valuable for ensuring data integrity and completeness at the record level, which is crucial for many analytical and operational data applications.


## Checking Row Distinctness

Here's an example where we check for duplicate rows with <a href="../../reference/Validate.rows_distinct.html#pointblank.Validate.rows_distinct" class="gdls-link"><code>Validate.rows_distinct()</code></a>:


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .rows_distinct()
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+cm93c19kaXN0aW5jdDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJyb3dzX2Rpc3RpbmN0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC40ODI3NTkpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJub19nZW1pbmkiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE3LjAwMDAwMCwgMTMuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMy42NjcwNTYxOSw2LjYyMTA3NTA4IEMzLjEyNTEwMTA0LDYuNjQwNjYzODYgMi42NzQ1NTk3NCw3LjA0NzY3NDQ0IDIuNjAyNzM0MzIsNy41ODUyNzY4MiBDMi41Mjg3MzIyOCw4LjEyMjg3OTIgMi44NTMwMzUyNiw4LjYzNDM2MjcgMy4zNzEwNDg0OCw4Ljc5NzYwMjM5IEM0LjQwNDg5OTA5LDkuMTU0NTUyODYgNi43MDExMzU1Myw5Ljg3MDYzMDIxIDkuODY1ODA1OTUsMTAuMzY0NzAyIEw5Ljg2NTgwNTk1LDMwLjczNjk5NzYgQzYuNjU5NzgxMzcsMzEuMjMzMjQ1OCA0LjM3MjI1MTA0LDMxLjk2NDU1OSAzLjM3MTA0ODQ4LDMyLjMyMTUwOTUgQzIuNzg5OTE1NTUsMzIuNTI4Mjc5NyAyLjQ4NTIwMTczLDMzLjE2ODE3ODQgMi42OTE5NzE5NiwzMy43NDkzMTE0IEMyLjg5ODc0MjIsMzQuMzMwNDQ0MyAzLjUzODY0MDk0LDM0LjYzNTE1ODEgNC4xMTk3NzM4NywzNC40MjgzODc5IEM1LjU0OTc1MjE3LDMzLjkxOTA4MDggMTAuMjAzMTY3OCwzMi40NjA4MDcyIDE2LjUxNzI3MzQsMzIuNDYwODA3MiBDMjIuNzk0Mzc4MSwzMi40NjA4MDcyIDI3LjU1MDA5MDEsMzMuODkwNzg1NSAyOS4wNTQwNzA2LDM0LjQxMDk3NTcgQzI5LjYzNTIwMzYsMzQuNjEzMzkyNiAzMC4yNzA3NDk1LDM0LjMwNDMyNiAzMC40NzMxNjY0LDMzLjcyMzE5MzEgQzMwLjY3NTU4MzMsMzMuMTQyMDYwMSAzMC4zNjY1MTY3LDMyLjUwNjUxNDIgMjkuNzg1MzgzOCwzMi4zMDQwOTczIEMyOC43NDkzNTY4LDMxLjk0NDk3MDQgMjYuNDMxMzU1NCwzMS4yNDQxMjgzIDIzLjIzODM4OTcsMzAuNzU0NDA5OCBMMjMuMjM4Mzg5NywxMC4zODIxMTQzIEMyNi40NDQ0MTQzLDkuODg4MDQyNDMgMjguNzQ1MDA0LDkuMTYxMDgyNTkgMjkuNzY3OTcxNiw4Ljc5NzYwMjM5IEMzMC4zNDkxMDQ1LDguNTkwODMyMTUgMzAuNjUzODE4NCw3Ljk1MDkzMzQxIDMwLjQ0NzA0ODEsNy4zNjk4MDA0OCBDMzAuMjQwMjc3OSw2Ljc4ODY2NzU1IDI5LjYwMDM3OTEsNi40ODM5NTM3MyAyOS4wMTkyNDYyLDYuNjkwNzIzOTYgQzI3LjU1ODc5NjMsNy4yMDg3Mzc3NCAyMi45MTYyNjM3LDguNjU4MzA0NjQgMTYuNjM5MTU4OSw4LjY1ODMwNDY0IEMxMC4zNzA3NjAzLDguNjU4MzA0NjQgNS42MTI4NzE4OCw3LjIxMzA5MDUxIDQuMTAyMzYxNjYsNi42OTA3MjM5NiBDMy45NjMwNjM5MSw2LjYzODQ4NzMgMy44MTUwNjAwNSw2LjYxNDU0NTQ4IDMuNjY3MDU2MTksNi42MjEwNzUwOCBaIE0xMi4wOTQ1Njk5LDEwLjY0MzI5NzUgQzEzLjQ5NDA3NzEsMTAuNzg5MTI1IDE1LjAxOTgyMjYsMTAuODg3MDY4NiAxNi42MzkxNTg5LDEwLjg4NzA2ODYgQzE4LjE5OTcyODksMTAuODg3MDY4NiAxOS42NjIzNTUyLDEwLjc5NzgzMTEgMjEuMDA5NjI1NywxMC42NjA3MDk4IEwyMS4wMDk2MjU3LDMwLjQ1ODQwMjEgQzE5LjYyMzE3OCwzMC4zMTY5MjggMTguMTE5MTk3NSwzMC4yMzIwNDMzIDE2LjUxNzI3MzQsMzAuMjMyMDQzMyBDMTQuOTMyNzYxNSwzMC4yMzIwNDMzIDEzLjQ1NzA3NjMsMzAuMzE5MTA0NCAxMi4wOTQ1Njk5LDMwLjQ1ODQwMjEgTDEyLjA5NDU2OTksMTAuNjQzMjk3NSBaIiBpZD0iZ2VtaW5pIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTE2LjY2MDUzNTQsLTUuMDU5Mjk0OTkgQzE2LjkzNjY3NzgsLTUuMDU5Mjk0OTkgMTcuMTYwNTM1NCwtNC44MzU0MzczNyAxNy4xNjA1MzU0LC00LjU1OTI5NDk5IEwxNy4xNjA1MzU0LDQ0LjU2ODc1MDkgQzE3LjE2MDUzNTQsNDQuODQ0ODkzMyAxNi45MzY2Nzc4LDQ1LjA2ODc1MDkgMTYuNjYwNTM1NCw0NS4wNjg3NTA5IEMxNi4zODQzOTMsNDUuMDY4NzUwOSAxNi4xNjA1MzU0LDQ0Ljg0NDg5MzMgMTYuMTYwNTM1NCw0NC41Njg3NTA5IEwxNi4xNjA1MzU0LC00LjU1OTI5NDk5IEMxNi4xNjA1MzU0LC00LjgzNTQzNzM3IDE2LjM4NDM5MywtNS4wNTkyOTQ5OSAxNi42NjA1MzU0LC01LjA1OTI5NDk5IFoiIGlkPSJsaW5lX2JsYWNrIiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNi42NjA1MzUsIDIwLjAwNDcyOCkgcm90YXRlKC0zMjAuMDAwMDAwKSB0cmFuc2xhdGUoLTE2LjY2MDUzNSwgLTIwLjAwNDcyOCkgIiAvPgogICAgICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4LjU2MDAzMiwgMTkuMTU4MDMxKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMTguNTYwMDMyLCAtMTkuMTU4MDMxKSAiIHBvaW50cz0iMTguMDYwMDMxNiAtNC40NTM2NjczNSAxOS4wNjAwMzE2IC00LjQ1MzY2NzM1IDE5LjA2MDAzMTYgNDIuNzY5NzI5OSAxOC4wNjAwMzE2IDQyLjc2OTcyOTkiPjwvcG9seWdvbj4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

rows_distinct()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">ALL COLUMNS</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">9<br />
0.69</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.15</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


We can also adapt the <a href="../../reference/Validate.rows_distinct.html#pointblank.Validate.rows_distinct" class="gdls-link"><code>Validate.rows_distinct()</code></a> check to use a single column or a subset of columns. To do that, we need to use the `columns_subset=` parameter. Here's an example of that:


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .rows_distinct(columns_subset="b")
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+cm93c19kaXN0aW5jdDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJyb3dzX2Rpc3RpbmN0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC40ODI3NTkpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJub19nZW1pbmkiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE3LjAwMDAwMCwgMTMuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMy42NjcwNTYxOSw2LjYyMTA3NTA4IEMzLjEyNTEwMTA0LDYuNjQwNjYzODYgMi42NzQ1NTk3NCw3LjA0NzY3NDQ0IDIuNjAyNzM0MzIsNy41ODUyNzY4MiBDMi41Mjg3MzIyOCw4LjEyMjg3OTIgMi44NTMwMzUyNiw4LjYzNDM2MjcgMy4zNzEwNDg0OCw4Ljc5NzYwMjM5IEM0LjQwNDg5OTA5LDkuMTU0NTUyODYgNi43MDExMzU1Myw5Ljg3MDYzMDIxIDkuODY1ODA1OTUsMTAuMzY0NzAyIEw5Ljg2NTgwNTk1LDMwLjczNjk5NzYgQzYuNjU5NzgxMzcsMzEuMjMzMjQ1OCA0LjM3MjI1MTA0LDMxLjk2NDU1OSAzLjM3MTA0ODQ4LDMyLjMyMTUwOTUgQzIuNzg5OTE1NTUsMzIuNTI4Mjc5NyAyLjQ4NTIwMTczLDMzLjE2ODE3ODQgMi42OTE5NzE5NiwzMy43NDkzMTE0IEMyLjg5ODc0MjIsMzQuMzMwNDQ0MyAzLjUzODY0MDk0LDM0LjYzNTE1ODEgNC4xMTk3NzM4NywzNC40MjgzODc5IEM1LjU0OTc1MjE3LDMzLjkxOTA4MDggMTAuMjAzMTY3OCwzMi40NjA4MDcyIDE2LjUxNzI3MzQsMzIuNDYwODA3MiBDMjIuNzk0Mzc4MSwzMi40NjA4MDcyIDI3LjU1MDA5MDEsMzMuODkwNzg1NSAyOS4wNTQwNzA2LDM0LjQxMDk3NTcgQzI5LjYzNTIwMzYsMzQuNjEzMzkyNiAzMC4yNzA3NDk1LDM0LjMwNDMyNiAzMC40NzMxNjY0LDMzLjcyMzE5MzEgQzMwLjY3NTU4MzMsMzMuMTQyMDYwMSAzMC4zNjY1MTY3LDMyLjUwNjUxNDIgMjkuNzg1MzgzOCwzMi4zMDQwOTczIEMyOC43NDkzNTY4LDMxLjk0NDk3MDQgMjYuNDMxMzU1NCwzMS4yNDQxMjgzIDIzLjIzODM4OTcsMzAuNzU0NDA5OCBMMjMuMjM4Mzg5NywxMC4zODIxMTQzIEMyNi40NDQ0MTQzLDkuODg4MDQyNDMgMjguNzQ1MDA0LDkuMTYxMDgyNTkgMjkuNzY3OTcxNiw4Ljc5NzYwMjM5IEMzMC4zNDkxMDQ1LDguNTkwODMyMTUgMzAuNjUzODE4NCw3Ljk1MDkzMzQxIDMwLjQ0NzA0ODEsNy4zNjk4MDA0OCBDMzAuMjQwMjc3OSw2Ljc4ODY2NzU1IDI5LjYwMDM3OTEsNi40ODM5NTM3MyAyOS4wMTkyNDYyLDYuNjkwNzIzOTYgQzI3LjU1ODc5NjMsNy4yMDg3Mzc3NCAyMi45MTYyNjM3LDguNjU4MzA0NjQgMTYuNjM5MTU4OSw4LjY1ODMwNDY0IEMxMC4zNzA3NjAzLDguNjU4MzA0NjQgNS42MTI4NzE4OCw3LjIxMzA5MDUxIDQuMTAyMzYxNjYsNi42OTA3MjM5NiBDMy45NjMwNjM5MSw2LjYzODQ4NzMgMy44MTUwNjAwNSw2LjYxNDU0NTQ4IDMuNjY3MDU2MTksNi42MjEwNzUwOCBaIE0xMi4wOTQ1Njk5LDEwLjY0MzI5NzUgQzEzLjQ5NDA3NzEsMTAuNzg5MTI1IDE1LjAxOTgyMjYsMTAuODg3MDY4NiAxNi42MzkxNTg5LDEwLjg4NzA2ODYgQzE4LjE5OTcyODksMTAuODg3MDY4NiAxOS42NjIzNTUyLDEwLjc5NzgzMTEgMjEuMDA5NjI1NywxMC42NjA3MDk4IEwyMS4wMDk2MjU3LDMwLjQ1ODQwMjEgQzE5LjYyMzE3OCwzMC4zMTY5MjggMTguMTE5MTk3NSwzMC4yMzIwNDMzIDE2LjUxNzI3MzQsMzAuMjMyMDQzMyBDMTQuOTMyNzYxNSwzMC4yMzIwNDMzIDEzLjQ1NzA3NjMsMzAuMzE5MTA0NCAxMi4wOTQ1Njk5LDMwLjQ1ODQwMjEgTDEyLjA5NDU2OTksMTAuNjQzMjk3NSBaIiBpZD0iZ2VtaW5pIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTE2LjY2MDUzNTQsLTUuMDU5Mjk0OTkgQzE2LjkzNjY3NzgsLTUuMDU5Mjk0OTkgMTcuMTYwNTM1NCwtNC44MzU0MzczNyAxNy4xNjA1MzU0LC00LjU1OTI5NDk5IEwxNy4xNjA1MzU0LDQ0LjU2ODc1MDkgQzE3LjE2MDUzNTQsNDQuODQ0ODkzMyAxNi45MzY2Nzc4LDQ1LjA2ODc1MDkgMTYuNjYwNTM1NCw0NS4wNjg3NTA5IEMxNi4zODQzOTMsNDUuMDY4NzUwOSAxNi4xNjA1MzU0LDQ0Ljg0NDg5MzMgMTYuMTYwNTM1NCw0NC41Njg3NTA5IEwxNi4xNjA1MzU0LC00LjU1OTI5NDk5IEMxNi4xNjA1MzU0LC00LjgzNTQzNzM3IDE2LjM4NDM5MywtNS4wNTkyOTQ5OSAxNi42NjA1MzU0LC01LjA1OTI5NDk5IFoiIGlkPSJsaW5lX2JsYWNrIiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNi42NjA1MzUsIDIwLjAwNDcyOCkgcm90YXRlKC0zMjAuMDAwMDAwKSB0cmFuc2xhdGUoLTE2LjY2MDUzNSwgLTIwLjAwNDcyOCkgIiAvPgogICAgICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4LjU2MDAzMiwgMTkuMTU4MDMxKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMTguNTYwMDMyLCAtMTkuMTU4MDMxKSAiIHBvaW50cz0iMTguMDYwMDMxNiAtNC40NTM2NjczNSAxOS4wNjAwMzE2IC00LjQ1MzY2NzM1IDE5LjA2MDAzMTYgNDIuNzY5NzI5OSAxOC4wNjAwMzE2IDQyLjc2OTcyOTkiPjwvcG9seWdvbj4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

rows_distinct()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">11<br />
0.85</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.15</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


## Checking Row Completeness

Another important validation is checking for complete rows: rows that have no missing values across all columns or a specified subset of columns. The <a href="../../reference/Validate.rows_complete.html#pointblank.Validate.rows_complete" class="gdls-link"><code>Validate.rows_complete()</code></a> validation method performs this check.

Here's an example checking if all rows in the table are complete (have no missing values in any column):


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .rows_complete()
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+cm93c19jb21wbGV0ZTwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJyb3dzX2NvbXBsZXRlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC45NjU1MTcpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJjb21wbGV0ZV9tZSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTIuNTAwMDAwLCA5LjUwMDAwMCkiIGZpbGw9IiMwMDAwMDAiPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTgsMCBMOCwxMCBMMTYsMTAgTDE2LDE4IEwyNiwxOCBMMjYsMTAgTDM0LDEwIEwzNCwwIEw4LDAgWiBNMTAsMiBMMTYsMiBMMTYsOCBMMTAsOCBMMTAsMiBaIE0xOCwyIEwyNCwyIEwyNCw4IEwxOCw4IEwxOCwyIFogTTI2LDIgTDMyLDIgTDMyLDggTDI2LDggTDI2LDIgWiBNMTgsMTAgTDI0LDEwIEwyNCwxNiBMMTgsMTYgTDE4LDEwIFogTTAsMjEgTDAsNDcgTDQyLDQ3IEw0MiwyMSBMMzIsMjEgTDMyLDI5IEwyNCwyOSBMMjQsMzcgTDE4LDM3IEwxOCwyOSBMMTAsMjkgTDEwLDIxIEwwLDIxIFogTTIsMjMgTDgsMjMgTDgsMjkgTDIsMjkgTDIsMjMgWiBNMzQsMjMgTDQwLDIzIEw0MCwyOSBMMzQsMjkgTDM0LDIzIFogTTIsMzEgTDgsMzEgTDgsMzcgTDIsMzcgTDIsMzEgWiBNMTAsMzEgTDE2LDMxIEwxNiwzNyBMMTAsMzcgTDEwLDMxIFogTTI2LDMxIEwzMiwzMSBMMzIsMzcgTDI2LDM3IEwyNiwzMSBaIE0zNCwzMSBMNDAsMzEgTDQwLDM3IEwzNCwzNyBMMzQsMzEgWiBNMiwzOSBMOCwzOSBMOCw0NSBMMiw0NSBMMiwzOSBaIE0xMCwzOSBMMTYsMzkgTDE2LDQ1IEwxMCw0NSBMMTAsMzkgWiBNMTgsMzkgTDI0LDM5IEwyNCw0NSBMMTgsNDUgTDE4LDM5IFogTTI2LDM5IEwzMiwzOSBMMzIsNDUgTDI2LDQ1IEwyNiwzOSBaIE0zNCwzOSBMNDAsMzkgTDQwLDQ1IEwzNCw0NSBMMzQsMzkgWiIgaWQ9IlNoYXBlIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMjIuNDU2NjQ3NiwxOC4zNTgxNyBDMjIuOTI1Mzk3NiwxOC4yOTU2NyAyMy4zNzQ2MTY2LDE4LjU2OTEwOCAyMy41MzA4NjY2LDE5LjAxNDQyIEMyMy42OTEwMjI2LDE5LjQ1OTczMyAyMy41MTUyNDE2LDE5Ljk1NTgyNiAyMy4xMTI4OTc2LDIwLjIwMTkyIEwyMy4xMTI4OTc2LDIwLjIwMTkyIEwyMC4yMDY2NDc2LDIyLjM4OTQyIEwyNS43OTg5Mjg2LDIyLjM4OTMxMjMgTDI1Ljc5ODkyODYsMjQuMzg5MzEyMyBMMjAuMjA2NjQ3NiwyNC4zODk0MiBMMjMuMTEyODk3NiwyNi41NzY5MiBDMjMuNTYyMTE2NiwyNi45MTI4NTggMjMuNjUxOTYwNiwyNy41NDk1NzYgMjMuMzE2MDIyNiwyNy45OTg3OTUgQzIyLjk4MDA4NTYsMjguNDQ4MDE0IDIyLjM0MzM2NjYsMjguNTM3ODU4IDIxLjg5NDE0NzYsMjguMjAxOTIgTDIxLjg5NDE0NzYsMjguMjAxOTIgTDE2LjYxMjg5NzYsMjQuMjAxOTIgQzE2LjM1MTE3ODYsMjQuMDE0NDIgMTYuMTk0OTI4NiwyMy43MDk3MzMgMTYuMTk0OTI4NiwyMy4zODk0MiBDMTYuMTk0OTI4NiwyMy4wNjkxMDggMTYuMzUxMTc4NiwyMi43NjQ0MiAxNi42MTI4OTc2LDIyLjU3NjkyIEwxNi42MTI4OTc2LDIyLjU3NjkyIEwyMS44OTQxNDc2LDE4LjU3NjkyIEMyMi4wMjMwNTM2LDE4LjQ3OTI2NCAyMi4xNzE0OTE2LDE4LjQxNjc2NCAyMi4zMzE2NDc2LDE4LjM4OTQyIEMyMi4zNzA3MTA2LDE4LjM3NzcwMSAyMi40MTM2Nzg2LDE4LjM2NTk4MyAyMi40NTY2NDc2LDE4LjM1ODE3IFoiIGlkPSJhcnJvd19yaWdodCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjAuOTk3MzkzLCAyMy4zNzcxNDkpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTIwLjk5NzM5MywgLTIzLjM3NzE0OSkgIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

rows_complete()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">ALL COLUMNS</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">13</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">11<br />
0.85</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.15</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


As the report indicates, there are some incomplete rows in the table.


# 3. Table Structure Validations

Table structure validations ensure that the overall architecture of your data meets expectations. These structural checks form a foundation for more detailed data quality assessments:

- <a href="../../reference/Validate.col_exists.html#pointblank.Validate.col_exists" class="gdls-link"><code>Validate.col_exists()</code></a>: verifies a column exists in the table
- <a href="../../reference/Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>Validate.col_schema_match()</code></a>: ensures table matches a defined schema
- <a href="../../reference/Validate.col_count_match.html#pointblank.Validate.col_count_match" class="gdls-link"><code>Validate.col_count_match()</code></a>: confirms the table has the expected number of columns
- <a href="../../reference/Validate.row_count_match.html#pointblank.Validate.row_count_match" class="gdls-link"><code>Validate.row_count_match()</code></a>: verifies the table has the expected number of rows
- <a href="../../reference/Validate.tbl_match.html#pointblank.Validate.tbl_match" class="gdls-link"><code>Validate.tbl_match()</code></a>: validates that the target table matches a comparison table
- <a href="../../reference/Validate.data_freshness.html#pointblank.Validate.data_freshness" class="gdls-link"><code>Validate.data_freshness()</code></a>: checks that data is recent and not stale

These structural validations provide essential checks on the fundamental organization of your data tables, ensuring they have the expected dimensions and components needed for reliable data analysis.


## Checking Column Presence

If you need to check for the presence of individual columns, the [Validate.col_exists()](../../reference/Validate.col_exists.md#pointblank.Validate.col_exists) validation method is useful. In this example, we check whether the `date` column is present in the table:


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_exists(columns="date")
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2V4aXN0czwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfZXhpc3RzIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC44Mjc1ODYpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxyZWN0IGlkPSJjb2x1bW4iIGZpbGw9IiMwMDAwMDAiIHg9IjEyLjIxMTcxNTMiIHk9IjEyLjAxNDY2OTQiIHdpZHRoPSIyMCIgaGVpZ2h0PSI0MiIgcng9IjEiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC4zMTc3MTE0LDQzLjAxNDY2OTQgTDQ0LjMxNzcxMTQsNDAuNTEzNjkyOCBMNDYuODE4Njg4LDQwLjUxMzY5MjggTDQ2LjgxODY4OCw0My4wMTQ2Njk0IEw0NC4zMTc3MTE0LDQzLjAxNDY2OTQgWiBNNDQuMzE3NzExNCwzOC4wMDAwMjA5IEw0NC4zMTc3MTE0LDM3LjMxNDQ3NCBDNDQuMzE3NzExNCwzNS42OTc5Mjk1IDQ0LjkzOTc3NTUsMzQuMTc4NzM5IDQ2LjE4MzkyMjQsMzIuNzU2ODU2OSBMNDYuOTgzNzI3MSwzMS44MzAwOTkgQzQ4LjMxMjUwOTcsMzAuMzA2NjUzOSA0OC45NzY4OTExLDI5LjA1ODI5NCA0OC45NzY4OTExLDI4LjA4NDk4MTkgQzQ4Ljk3Njg5MTEsMjcuMzMxNzIyOSA0OC42ODQ5MDE5LDI2LjczNTA0OTIgNDguMTAwOTE0NiwyNi4yOTQ5NDI4IEM0Ny41MTY5MjczLDI1Ljg1NDgzNjQgNDYuNzI5ODI1OCwyNS42MzQ3ODY1IDQ1LjczOTU4NjQsMjUuNjM0Nzg2NSBDNDQuNDQ0NjU4MSwyNS42MzQ3ODY1IDQzLjA2OTM0NjMsMjUuOTQ3OTM0NSA0MS42MTM2MDk5LDI2LjU3NDIzOTcgTDQxLjYxMzYwOTksMjQuNDU0MTIyNSBDNDMuMTc5MzcyOSwyMy45ODAxNjE4IDQ0LjY0MzU1MSwyMy43NDMxODUgNDYuMDA2MTg4LDIzLjc0MzE4NSBDNDcuNzMyNzU5MSwyMy43NDMxODUgNDkuMTAzODM5MiwyNC4xMzAzODgxIDUwLjExOTQ2OTIsMjQuOTA0ODA2MSBDNTEuMTM1MDk5MywyNS42NzkyMjQgNTEuNjQyOTA2NywyNi43MjY1NzY4IDUxLjY0MjkwNjcsMjguMDQ2ODk1OSBDNTEuNjQyOTA2NywyOC43OTE2OTEzIDUxLjQ5NjkxMjEsMjkuNDMyNzk4MiA1MS4yMDQ5MTg1LDI5Ljk3MDIzNTggQzUwLjkxMjkyNDgsMzAuNTA3NjczMyA1MC4zNTIyMjA4LDMxLjE2OTkzODkgNDkuNTIyNzg5NiwzMS45NTcwNTIyIEw0OC43MzU2ODAyLDMyLjY5MzM4MDMgQzQ3Ljk0ODU2NjksMzMuNDM4MTc1NyA0Ny40MzIyOTYsMzQuMDYyMzU1NiA0Ny4xODY4NTIxLDM0LjU2NTkzODkgQzQ2Ljk0MTQwODEsMzUuMDY5NTIyMSA0Ni44MTg2ODgsMzUuNzQ4NzE0NiA0Ni44MTg2ODgsMzYuNjAzNTM2NSBMNDYuODE4Njg4LDM4LjAwMDAyMDkgTDQ0LjMxNzcxMTQsMzguMDAwMDIwOSBaIiBpZD0iPyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_exists()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


That column is present, so the single test unit of this validation step is a passing one.


## Checking the Table Schema

For deeper checks of table structure, a schema validation can be performed with the <a href="../../reference/Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>Validate.col_schema_match()</code></a> validation method, where the goal is to check whether the structure of a table matches an expected schema. To define an expected table schema, we need to use the [Schema](../../reference/Schema.md#pointblank.Schema) class. Here is a simple example that (1) prepares a schema consisting of column names, (2) uses that `schema` object in a <a href="../../reference/Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>Validate.col_schema_match()</code></a> validation step:


``` python
schema = pb.Schema(columns=["date_time", "date", "a", "b", "c", "d", "e", "f"])

(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_schema_match(schema=schema)
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3NjaGVtYV9tYXRjaDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfc2NoZW1hX21hdGNoIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4zMTAzNDUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01My43MTIyMzQsMzkuNzg4NTI2OCBMNTQuMjEyMjM0LDU2LjI4ODUyNjggTDQyLjIxMjIzNCw1Ni43ODg1MjY4IEw0Mi4yMTIyMzQsMzkuNzg4NTI2OCBMNTMuNzEyMjM0LDM5Ljc4ODUyNjggWiBNMzkuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCw1Ni43ODg1MjY4IEwyNy43MTIyMzQsNTYuNzg4NTI2OCBMMjcuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCwzOS43ODg1MjY4IFogTTI1LjIxMjIzNCwzOS43ODg1MjY4IEwyNS4yMTIyMzQsNTYuNzg4NTI2OCBMMTMuNzEyMjM0LDU2Ljc4ODUyNjggTDEzLjIxMjIzNCw0MC4yODg1MjY4IEwyNS4yMTIyMzQsMzkuNzg4NTI2OCBaIiBpZD0iY29sdW1uc19zY2hlbWEiIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8ZyBpZD0idmVydGljYWxfZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMwLjAwMDAwMCwgMjkuMDAwMDAwKSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIj4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIyLjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSIyLjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUiPjwvbGluZT4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSI1LjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSI1LjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUtQ29weSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00MS43MTIyMzQsOS4wMTQ2NjkzNSBMNDEuNzEyMjM0LDI3LjAxNDY2OTQgTDUzLjcxMjIzNCwyNy4wMTQ2Njk0IEM1NC4yNjIyMzQsMjcuMDE0NjY5NCA1NC43MTIyMzQsMjYuNTY0NjY5NCA1NC43MTIyMzQsMjYuMDE0NjY5NCBMNTQuNzEyMjM0LDEwLjAxNDY2OTQgQzU0LjcxMjIzNCw5LjQ2NDY2OTM1IDU0LjI2MjIzNCw5LjAxNDY2OTM1IDUzLjcxMjIzNCw5LjAxNDY2OTM1IEw0MS43MTIyMzQsOS4wMTQ2NjkzNSBaIE0yNy4yMTIyMzQsOS4wMTQ2NjkzNSBDMjcuMjEyMjM0LDkuMDE0NjY5MzUgMjcuMjEyMjM0LDE1LjAxNDY2OTQgMjcuMjEyMjM0LDI3LjAxNDY2OTQgTDQwLjIxMjIzNCwyNy4wMTQ2Njk0IEw0MC4yMTIyMzQsOS4wMTQ2NjkzNSBDMzEuNTQ1NTY3Myw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IFogTTEzLjcxMjIzNCw5LjAxNDY2OTM1IEMxMy4xNjIyMzQsOS4wMTQ2NjkzNSAxMi43MTIyMzQsOS40NjQ2NjkzNSAxMi43MTIyMzQsMTAuMDE0NjY5NCBMMTIuNzEyMjM0LDI2LjAxNDY2OTQgQzEyLjcxMjIzNCwyNi41NjQ2Njk0IDEzLjE2MjIzNCwyNy4wMTQ2Njk0IDEzLjcxMjIzNCwyNy4wMTQ2Njk0IEwyNS43MTIyMzQsMjcuMDE0NjY5NCBMMjUuNzEyMjM0LDkuMDE0NjY5MzUgTDEzLjcxMjIzNCw5LjAxNDY2OTM1IFoiIGlkPSJjb2x1bW5zX3JlYWwiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_schema_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">SCHEMA</td>
AxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(schema_check)</span> <span style="color:#4CA64C;">✓</span> Schema validation <strong>passed</strong>.</p>
Schema Comparison


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
<td class="gt_row gt_right" style="font-size: 13px">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date_time</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Datetime(time_unit='us', time_zone=None)</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date_time</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Date</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">5</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">5</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">6</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">6</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">7</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">e</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Boolean</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">7</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">e</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">8</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">f</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">8</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">f</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left"></td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('date_time',), ('date',), ('a',), ('b',), ('c',), ('d',), ('e',), ('f',)]</code>
</div></td>
</tr>
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
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


The <a href="../../reference/Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>Validate.col_schema_match()</code></a> validation step will only have a single test unit (signifying pass or fail). We can see in the above validation report that the column schema validation passed.

More often, a schema will be defined using column names and column types. We can do that by using a list of tuples in the `columns=` parameter of [Schema](../../reference/Schema.md#pointblank.Schema). Here's an example of that approach in action:


``` python
schema = pb.Schema(
    columns=[
        ("date_time", "Datetime(time_unit='us', time_zone=None)"),
        ("date", "Date"),
        ("a", "Int64"),
        ("b", "String"),
        ("c", "Int64"),
        ("d", "Float64"),
        ("e", "Boolean"),
        ("f", "String"),
    ]
)

(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_schema_match(schema=schema)
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3NjaGVtYV9tYXRjaDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfc2NoZW1hX21hdGNoIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC4zMTAzNDUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxLjAxNDY2OTM1IEM1OS4xOTc1MTUzLDEuMDE0NjY5MzUgNjEuNDQ3NTE1MywyLjAyMjAyODY3IDYzLjA3NjE5NSwzLjY1MDcwODMyIEM2NC43MDQ4NzQ3LDUuMjc5Mzg3OTggNjUuNzEyMjM0LDcuNTI5Mzg3OTggNjUuNzEyMjM0LDEwLjAxNDY2OTQgTDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsNjUuMDE0NjY5NCBMMTAuNzEyMjM0LDY1LjAxNDY2OTQgQzguMjI2OTUyNTksNjUuMDE0NjY5NCA1Ljk3Njk1MjU5LDY0LjAwNzMxIDQuMzQ4MjcyOTQsNjIuMzc4NjMwNCBDMi43MTk1OTMyOCw2MC43NDk5NTA3IDEuNzEyMjMzOTcsNTguNDk5OTUwNyAxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsNTYuMDE0NjY5NCBMMS43MTIyMzM5NywxMC4wMTQ2Njk0IEMxLjcxMjIzMzk3LDcuNTI5Mzg3OTggMi43MTk1OTMyOCw1LjI3OTM4Nzk4IDQuMzQ4MjcyOTQsMy42NTA3MDgzMiBDNS45NzY5NTI1OSwyLjAyMjAyODY3IDguMjI2OTUyNTksMS4wMTQ2NjkzNSAxMC43MTIyMzQsMS4wMTQ2NjkzNSBMMTAuNzEyMjM0LDEuMDE0NjY5MzUgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01My43MTIyMzQsMzkuNzg4NTI2OCBMNTQuMjEyMjM0LDU2LjI4ODUyNjggTDQyLjIxMjIzNCw1Ni43ODg1MjY4IEw0Mi4yMTIyMzQsMzkuNzg4NTI2OCBMNTMuNzEyMjM0LDM5Ljc4ODUyNjggWiBNMzkuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCw1Ni43ODg1MjY4IEwyNy43MTIyMzQsNTYuNzg4NTI2OCBMMjcuNzEyMjM0LDM5Ljc4ODUyNjggTDM5LjcxMjIzNCwzOS43ODg1MjY4IFogTTI1LjIxMjIzNCwzOS43ODg1MjY4IEwyNS4yMTIyMzQsNTYuNzg4NTI2OCBMMTMuNzEyMjM0LDU2Ljc4ODUyNjggTDEzLjIxMjIzNCw0MC4yODg1MjY4IEwyNS4yMTIyMzQsMzkuNzg4NTI2OCBaIiBpZD0iY29sdW1uc19zY2hlbWEiIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8ZyBpZD0idmVydGljYWxfZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMwLjAwMDAwMCwgMjkuMDAwMDAwKSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2UtbGluZWNhcD0ic3F1YXJlIj4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSIyLjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSIyLjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUiPjwvbGluZT4KICAgICAgICAgICAgICAgIDxsaW5lIHgxPSI1LjIxMjIzMzk3IiB5MT0iMC41MTQ2NjkzNTMiIHgyPSI1LjIxMjIzMzk3IiB5Mj0iNy41ODU3MzcxNiIgaWQ9IkxpbmUtQ29weSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00MS43MTIyMzQsOS4wMTQ2NjkzNSBMNDEuNzEyMjM0LDI3LjAxNDY2OTQgTDUzLjcxMjIzNCwyNy4wMTQ2Njk0IEM1NC4yNjIyMzQsMjcuMDE0NjY5NCA1NC43MTIyMzQsMjYuNTY0NjY5NCA1NC43MTIyMzQsMjYuMDE0NjY5NCBMNTQuNzEyMjM0LDEwLjAxNDY2OTQgQzU0LjcxMjIzNCw5LjQ2NDY2OTM1IDU0LjI2MjIzNCw5LjAxNDY2OTM1IDUzLjcxMjIzNCw5LjAxNDY2OTM1IEw0MS43MTIyMzQsOS4wMTQ2NjkzNSBaIE0yNy4yMTIyMzQsOS4wMTQ2NjkzNSBDMjcuMjEyMjM0LDkuMDE0NjY5MzUgMjcuMjEyMjM0LDE1LjAxNDY2OTQgMjcuMjEyMjM0LDI3LjAxNDY2OTQgTDQwLjIxMjIzNCwyNy4wMTQ2Njk0IEw0MC4yMTIyMzQsOS4wMTQ2NjkzNSBDMzEuNTQ1NTY3Myw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IDI3LjIxMjIzNCw5LjAxNDY2OTM1IFogTTEzLjcxMjIzNCw5LjAxNDY2OTM1IEMxMy4xNjIyMzQsOS4wMTQ2NjkzNSAxMi43MTIyMzQsOS40NjQ2NjkzNSAxMi43MTIyMzQsMTAuMDE0NjY5NCBMMTIuNzEyMjM0LDI2LjAxNDY2OTQgQzEyLjcxMjIzNCwyNi41NjQ2Njk0IDEzLjE2MjIzNCwyNy4wMTQ2Njk0IDEzLjcxMjIzNCwyNy4wMTQ2Njk0IEwyNS43MTIyMzQsMjcuMDE0NjY5NCBMMjUuNzEyMjM0LDkuMDE0NjY5MzUgTDEzLjcxMjIzNCw5LjAxNDY2OTM1IFoiIGlkPSJjb2x1bW5zX3JlYWwiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_schema_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">SCHEMA</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(schema_check)</span> <span style="color:#4CA64C;">✓</span> Schema validation <strong>passed</strong>.</p>
Schema Comparison


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
<td class="gt_row gt_right" style="font-size: 13px">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date_time</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Datetime(time_unit='us', time_zone=None)</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">1</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date_time</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Datetime(time_unit='us', time_zone=None)</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Date</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">2</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">date</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Date</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">3</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Int64</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">4</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">5</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Int64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">5</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Int64</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">6</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">6</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">d</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Float64</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">7</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">e</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Boolean</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">7</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">e</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">Boolean</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
<tr>
<td class="gt_row gt_right" style="font-size: 13px">8</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">f</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_right" style="font-size: 13px; border-left: 3px double #E5E5E5">8</td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">f</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
<td class="gt_row gt_left" style="color: black; font-family: IBM Plex Mono; font-size: 13px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">String</td>
<td class="gt_row gt_left"><span style="color: #4CA64C;">✓</span></td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
Supplied Column Schema:

<code style="color: #303030; font-family: monospace; font-size: 8px;">[('date_time', "Datetime(time_unit='us', time_zone=None)"), ('date', 'Date'), ('a', 'Int64'), ('b', 'String'), ('c', 'Int64'), ('d', 'Float64'), ('e', 'Boolean'), ('f', 'String')]</code>
</div></td>
</tr>
<tr>
<td colspan="8" class="gt_sourcenote"><div style="padding-bottom: 2px;">
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


The <a href="../../reference/Validate.col_schema_match.html#pointblank.Validate.col_schema_match" class="gdls-link"><code>Validate.col_schema_match()</code></a> validation method has several boolean parameters for making the checks less stringent:

- `complete=`: requires exact column matching (all expected columns must exist, no extra columns allowed)
- `in_order=`: enforces that columns appear in the same order as defined in the schema
- `case_sensitive_colnames=`: column names must match with exact letter case
- `case_sensitive_dtypes=`: data type strings must match with exact letter case

These parameters all default to `True`, providing strict schema validation. Setting any to `False` relaxes the validation requirements, making the checks more flexible when exact matching isn't necessary or practical for your use case.


## Comparing Tables with [tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match)

The <a href="../../reference/Validate.tbl_match.html#pointblank.Validate.tbl_match" class="gdls-link"><code>Validate.tbl_match()</code></a> validation method provides a comprehensive way to verify that two tables are identical. It performs a progressive series of checks, from least to most stringent:

1.  Column count match
2.  Row count match
3.  Schema match (loose - case-insensitive, any order)
4.  Schema match (order - columns in correct order)
5.  Schema match (exact - case-sensitive, correct order)
6.  Data match (cell-by-cell comparison)

This progressive approach helps identify exactly where tables differ. Here's an example comparing the `small_table` dataset with itself:


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .tbl_match(tbl_compare=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+dGJsX21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InRibF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNzU4NjIxKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8ZyBpZD0iZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ2LjAyNjYxMSwgMjAuNzEwMTIyKSByb3RhdGUoLTkwLjAwMDAwMCkgdHJhbnNsYXRlKC00Ni4wMjY2MTEsIC0yMC43MTAxMjIpIHRyYW5zbGF0ZSg0Mi41MjY2MTEsIDE2LjIxMDEyMikiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLWxpbmVjYXA9InNxdWFyZSI+CiAgICAgICAgICAgICAgICA8bGluZSB4MT0iMi4yMTIyMzM5NyIgeTE9IjAuNTE0NjY5MzUzIiB4Mj0iMi4yMTIyMzM5NyIgeTI9IjcuNTg1NzM3MTYiIGlkPSJMaW5lIj48L2xpbmU+CiAgICAgICAgICAgICAgICA8bGluZSB4MT0iNS4yMTIyMzM5NyIgeTE9IjAuNTE0NjY5MzUzIiB4Mj0iNS4yMTIyMzM5NyIgeTI9IjcuNTg1NzM3MTYiIGlkPSJMaW5lIj48L2xpbmU+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPGcgaWQ9ImVxdWFsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyMS4zOTc4NTcsIDQ1LjMxOTIxNykgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMjEuMzk3ODU3LCAtNDUuMzE5MjE3KSB0cmFuc2xhdGUoMTcuODk3ODU3LCA0MC44MTkyMTcpIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yMS4zODgyNDE5LDcuNzc4Njk3ODMgQzIxLjM1ODQyOTgsNy43NzkzNTE3NyAyMS4zMjg3MDQsNy43ODIxNjM2NyAyMS4yOTkyOTk2LDcuNzg3MTE5MTQgTDkuMDkwMTQ4MjQsNy43ODcxMTkxNCBDOC43NTAyOTQzMSw3Ljc4NzE1MzEyIDguNDc0Nzk2NzcsOC4wNjI2NTA2NyA4LjQ3NDc2Mjc5LDguNDAyNTA0NiBMOC40NzQ3NjI3OSwxNi4yOTkxNDk4IEM4LjQ2Mzc3ODc0LDE2LjM2NTYwNjEgOC40NjM3Nzg3NCwxNi40MzM0MTQ5IDguNDc0NzYyNzksMTYuNDk5ODcxMyBMOC40NzQ3NjI3OSwyNC45MTQ1NDYxIEM4LjQ2Mzc3ODc0LDI0Ljk4MTAwMjUgOC40NjM3Nzg3NCwyNS4wNDg4MTEyIDguNDc0NzYyNzksMjUuMTE1MjY3NiBMOC40NzQ3NjI3OSwzMy4wMTc5MjI2IEM4LjQ3NDc5Njc3LDMzLjM1Nzc3NjYgOC43NTAyOTQzMSwzMy42MzMyNzQxIDkuMDkwMTQ4MjQsMzMuNjMzMzA4MSBMMjEuMjk0NDkxNiwzMy42MzMzMDgxIEMyMS4zNjA5NDc5LDMzLjY0NDI5MjEgMjEuNDI4NzU2NywzMy42NDQyOTIxIDIxLjQ5NTIxMzEsMzMuNjMzMzA4MSBMMzMuNzA1NTY2MywzMy42MzMzMDgxIEMzNC4wNDU0MjAyLDMzLjYzMzI3NDEgMzQuMzIwOTE3OCwzMy4zNTc3NzY2IDM0LjMyMDk1MTcsMzMuMDE3OTIyNiBMMzQuMzIwOTUxNywyNS4xMjEyNzc1IEMzNC4zMzE5MzU4LDI1LjA1NDgyMTEgMzQuMzMxOTM1OCwyNC45ODcwMTIzIDM0LjMyMDk1MTcsMjQuOTIwNTU2IEwzNC4zMjA5NTE3LDE2LjUwNTg4MTEgQzM0LjMzMTkzNTgsMTYuNDM5NDI0OCAzNC4zMzE5MzU4LDE2LjM3MTYxNiAzNC4zMjA5NTE3LDE2LjMwNTE1OTYgTDM0LjMyMDk1MTcsOC40MDI1MDQ2IEMzNC4zMjA5MTc4LDguMDYyNjUwNjcgMzQuMDQ1NDIwMiw3Ljc4NzE1MzEyIDMzLjcwNTU2NjMsNy43ODcxMTkxNCBMMjEuNDkyODA5NCw3Ljc4NzExOTE0IEMyMS40NTgyNTY1LDcuNzgxMzQzNjkgMjEuNDIzMjczNiw3Ljc3ODY5NzgzIDIxLjM4ODI0MTksNy43Nzg2OTc4MyBaIE05LjcwNTUzMzY5LDkuMDE3ODkwMDUgTDIwLjc4MjQ3MTgsOS4wMTc4OTAwNSBMMjAuNzgyNDcxOCwxNS43ODcxMyBMOS43MDU1MzM2OSwxNS43ODcxMyBMOS43MDU1MzM2OSw5LjAxNzg5MDA1IFogTTIyLjAxMzI0MjcsOS4wMTc4OTAwNSBMMzMuMDkwMTgwOCw5LjAxNzg5MDA1IEwzMy4wOTAxODA4LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDkuMDE3ODkwMDUgWiBNOS43MDU1MzM2OSwxNy4wMTc5MDA5IEwyMC43ODI0NzE4LDE3LjAxNzkwMDkgTDIwLjc4MjQ3MTgsMjQuNDAyNTI2MyBMOS43MDU1MzM2OSwyNC40MDI1MjYzIEw5LjcwNTUzMzY5LDE3LjAxNzkwMDkgWiBNMjIuMDEzMjQyNywxNy4wMTc5MDA5IEwzMy4wOTAxODA4LDE3LjAxNzkwMDkgTDMzLjA5MDE4MDgsMjQuNDAyNTI2MyBMMjIuMDEzMjQyNywyNC40MDI1MjYzIEwyMi4wMTMyNDI3LDE3LjAxNzkwMDkgWiBNOS43MDU1MzM2OSwyNS42MzMyOTcyIEwyMC43ODI0NzE4LDI1LjYzMzI5NzIgTDIwLjc4MjQ3MTgsMzIuNDAyNTM3MiBMOS43MDU1MzM2OSwzMi40MDI1MzcyIEw5LjcwNTUzMzY5LDI1LjYzMzI5NzIgWiBNMjIuMDEzMjQyNywyNS42MzMyOTcyIEwzMy4wOTAxODA4LDI1LjYzMzI5NzIgTDMzLjA5MDE4MDgsMzIuNDAyNTM3MiBMMjIuMDEzMjQyNywzMi40MDI1MzcyIEwyMi4wMTMyNDI3LDI1LjYzMzI5NzIgWiIgaWQ9InRhYmxlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00Ni4wMTY5OTUzLDMyLjM4Nzc5MjYgQzQ1Ljk4NzE4MzIsMzIuMzg4NDQ2NSA0NS45NTc0NTc1LDMyLjM5MTI1ODQgNDUuOTI4MDUzLDMyLjM5NjIxMzkgTDMzLjcxODkwMTYsMzIuMzk2MjEzOSBDMzMuMzc5MDQ3NywzMi4zOTYyNDc5IDMzLjEwMzU1MDIsMzIuNjcxNzQ1NCAzMy4xMDM1MTYyLDMzLjAxMTU5OTMgTDMzLjEwMzUxNjIsNDAuOTA4MjQ0NSBDMzMuMDkyNTMyMiw0MC45NzQ3MDA5IDMzLjA5MjUzMjIsNDEuMDQyNTA5NyAzMy4xMDM1MTYyLDQxLjEwODk2NiBMMzMuMTAzNTE2Miw0OS41MjM2NDA4IEMzMy4wOTI1MzIyLDQ5LjU5MDA5NzIgMzMuMDkyNTMyMiw0OS42NTc5MDYgMzMuMTAzNTE2Miw0OS43MjQzNjI0IEwzMy4xMDM1MTYyLDU3LjYyNzAxNzQgQzMzLjEwMzU1MDIsNTcuOTY2ODcxMyAzMy4zNzkwNDc3LDU4LjI0MjM2ODkgMzMuNzE4OTAxNiw1OC4yNDI0MDI4IEw0NS45MjMyNDUsNTguMjQyNDAyOCBDNDUuOTg5NzAxNCw1OC4yNTMzODY5IDQ2LjA1NzUxMDEsNTguMjUzMzg2OSA0Ni4xMjM5NjY1LDU4LjI0MjQwMjggTDU4LjMzNDMxOTcsNTguMjQyNDAyOCBDNTguNjc0MTczNiw1OC4yNDIzNjg5IDU4Ljk0OTY3MTIsNTcuOTY2ODcxMyA1OC45NDk3MDUxLDU3LjYyNzAxNzQgTDU4Ljk0OTcwNTEsNDkuNzMwMzcyMiBDNTguOTYwNjg5Miw0OS42NjM5MTU4IDU4Ljk2MDY4OTIsNDkuNTk2MTA3MSA1OC45NDk3MDUxLDQ5LjUyOTY1MDcgTDU4Ljk0OTcwNTEsNDEuMTE0OTc1OSBDNTguOTYwNjg5Miw0MS4wNDg1MTk1IDU4Ljk2MDY4OTIsNDAuOTgwNzEwNyA1OC45NDk3MDUxLDQwLjkxNDI1NDQgTDU4Ljk0OTcwNTEsMzMuMDExNTk5MyBDNTguOTQ5NjcxMiwzMi42NzE3NDU0IDU4LjY3NDE3MzYsMzIuMzk2MjQ3OSA1OC4zMzQzMTk3LDMyLjM5NjIxMzkgTDQ2LjEyMTU2MjgsMzIuMzk2MjEzOSBDNDYuMDg3MDA5OSwzMi4zOTA0Mzg0IDQ2LjA1MjAyNywzMi4zODc3OTI2IDQ2LjAxNjk5NTMsMzIuMzg3NzkyNiBaIE0zNC4zMzQyODcxLDMzLjYyNjk4NDggTDQ1LjQxMTIyNTIsMzMuNjI2OTg0OCBMNDUuNDExMjI1Miw0MC4zOTYyMjQ4IEwzNC4zMzQyODcxLDQwLjM5NjIyNDggTDM0LjMzNDI4NzEsMzMuNjI2OTg0OCBaIE00Ni42NDE5OTYxLDMzLjYyNjk4NDggTDU3LjcxODkzNDIsMzMuNjI2OTg0OCBMNTcuNzE4OTM0Miw0MC4zOTYyMjQ4IEw0Ni42NDE5OTYxLDQwLjM5NjIyNDggTDQ2LjY0MTk5NjEsMzMuNjI2OTg0OCBaIE0zNC4zMzQyODcxLDQxLjYyNjk5NTcgTDQ1LjQxMTIyNTIsNDEuNjI2OTk1NyBMNDUuNDExMjI1Miw0OS4wMTE2MjExIEwzNC4zMzQyODcxLDQ5LjAxMTYyMTEgTDM0LjMzNDI4NzEsNDEuNjI2OTk1NyBaIE00Ni42NDE5OTYxLDQxLjYyNjk5NTcgTDU3LjcxODkzNDIsNDEuNjI2OTk1NyBMNTcuNzE4OTM0Miw0OS4wMTE2MjExIEw0Ni42NDE5OTYxLDQ5LjAxMTYyMTEgTDQ2LjY0MTk5NjEsNDEuNjI2OTk1NyBaIE0zNC4zMzQyODcxLDUwLjI0MjM5MiBMNDUuNDExMjI1Miw1MC4yNDIzOTIgTDQ1LjQxMTIyNTIsNTcuMDExNjMxOSBMMzQuMzM0Mjg3MSw1Ny4wMTE2MzE5IEwzNC4zMzQyODcxLDUwLjI0MjM5MiBaIE00Ni42NDE5OTYxLDUwLjI0MjM5MiBMNTcuNzE4OTM0Miw1MC4yNDIzOTIgTDU3LjcxODkzNDIsNTcuMDExNjMxOSBMNDYuNjQxOTk2MSw1Ny4wMTE2MzE5IEw0Ni42NDE5OTYxLDUwLjI0MjM5MiBaIiBpZD0idGFibGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

tbl_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">None</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">EXTERNAL TABLE</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


This validation method is especially useful for:

- Verifying that data transformations preserve expected properties
- Comparing production data against a golden dataset
- Ensuring data consistency across different environments
- Validating that imported data matches source data


## Checking Counts of Row and Columns

Row and column count validations check the number of rows and columns in a table.

Using <a href="../../reference/Validate.row_count_match.html#pointblank.Validate.row_count_match" class="gdls-link"><code>Validate.row_count_match()</code></a> checks whether the number of rows in a table matches a specified count.


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .row_count_match(count=13)
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+cm93X2NvdW50X21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InJvd19jb3VudF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNzkzMTAzKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMjkuMTY0NjE0OSwyOC40ODUzMzA2IEwyOS4xNjQ2MTQ5LDY5LjQ4NTMzMDYgTDI3LjA1NDkwODQsNjkuNDg1MDQ5MyBDMjYuOTI5NjE3OSw2OS40ODI2OTQ0IDI2LjYyOTUwODUsNjkuNDYwNjY0NSAyNi40MTYzNzA4LDY5LjI1NDg3NjUgQzI2LjI4ODI0ODEsNjkuMTMxMTcxOCAyNi4yMTIyMzQsNjguOTM0ODI2OSAyNi4yMTIyMzQsNjguNjUxOTk3MyBMMjYuMjEyMjM0LDY4LjY1MTk5NzMgTDI2LjIxMjIzNCwyOS4zMTg2NjQgQzI2LjIxMjIzNCwyOS4wMzU4MzQ0IDI2LjI4ODI0ODEsMjguODM5NDg5NSAyNi40MTYzNzA4LDI4LjcxNTc4NDggQzI2LjY1NjE1MDcsMjguNDg0MjczMyAyNy4wMDYwMDQxLDI4LjQ4NTMzMDYgMjcuMDkzMTg2MywyOC40ODUzMzA2IEwyOS4xNjQ2MTQ5LDI4LjQ4NTMzMDYgWiBNMzUuMTU5OTM5MywyOC40ODUzMzA2IEwzNS4xNTk5MzkzLDY5LjQ4NTMzMDYgTDMyLjI2NDUyODYsNjkuNDg1MzMwNiBMMzIuMjY0NTI4NiwyOC40ODUzMzA2IEwzNS4xNTk5MzkzLDI4LjQ4NTMzMDYgWiBNNDAuMzU5MzQ3NSwyOC40ODU3NTA3IEM0MC40NzMzOTUzLDI4LjQ4ODcyOCA0MC43NjY4MiwyOC41MDQ4MTQ2IDQwLjk4NTM1NTIsMjguNjk0OTQ1MSBDNDEuMTAwODIxNSwyOC43OTU0MDM0IDQxLjE4MTk5OTksMjguOTUxMzY4NCA0MS4yMDUxNzc2LDI5LjE3NDgzMiBMNDEuMjA1MTc3NiwyOS4xNzQ4MzIgTDQxLjIxMjIzNCw2OC42NTE5OTczIEM0MS4yMTIyMzQsNjguOTMwOTgzMyA0MS4xMzg1Mjc3LDY5LjEyNTgzNjkgNDEuMDEzMjgwMyw2OS4yNDk4MDk0IEM0MC43OTQzMDc0LDY5LjQ2NjU1MzUgNDAuNDc4MDQ3LDY5LjQ4MTg3ODcgNDAuMzU5NDEzOSw2OS40ODQ5MTA2IEw0MC4zNTk0MTM5LDY5LjQ4NDkxMDYgTDM4LjI1OTg1Myw2OS40ODUyNDk4IEwzOC4yNTk4NTMsMjguNDg1NDExNCBaIiBpZD0icm93c190d28iIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzMy43MTIyMzQsIDQ4Ljk4NTMzMSkgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMzMuNzEyMjM0LCAtNDguOTg1MzMxKSAiIC8+CiAgICAgICAgICAgIDxnIGlkPSJ2ZXJ0aWNhbF9lcXVhbCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzAuMDAwMDAwLCAyOS4zODA1NzApIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yNy4wOTMxODYzLC0yLjk4NTMzMDY1IEMyNi44ODM2NjI1LC0yLjk4NTMzMDY1IDI1LjcxMjIzNCwtMi45MzUzMzA2NSAyNS43MTIyMzQsLTEuNjUxOTk3MzEgTDI1LjcxMjIzNCwzNy42ODEzMzYgQzI1LjcxMjIzNCwzOC45NjQ2Njk0IDI2Ljg4MzY2MjUsMzkuMDE0NjY5NCAyNy4wOTMxODYzLDM5LjAxNDY2OTQgTDI5LjY2NDYxNDksMzkuMDE0NjY5NCBMMjkuNjY0NjE0OSwtMi45ODUzMzA2NSBMMjcuMDkzMTg2MywtMi45ODUzMzA2NSBaIE0zNS42NzIxNzcxLC0yLjk4NTMzMDY1IEwzNS42NzIxNzcxLDM5LjAxNDY2OTQgTDMxLjc1MjI5MDgsMzkuMDE0NjY5NCBMMzEuNzUyMjkwOCwzOS4wMTQ2Njk0IEwzMS43NTIyOTA4LC0yLjk4NTMzMDY1IEwzMS43NTIyOTA4LC0yLjk4NTMzMDY1IEwzNS42NzIxNzcxLC0yLjk4NTMzMDY1IFogTTQwLjM2NTYxNDksLTIuOTg0OTA5NiBDNDAuNjQ0ODc4NiwtMi45NzgyMzY5OCA0MS43MTIyMzQsLTIuODc2OTk3MzEgNDEuNzEyMjM0LC0xLjY1MTk5NzMxIEw0MS43MTIyMzQsLTEuNjUxOTk3MzEgTDQxLjcxMjIzNCwzNy42ODEzMzYgQzQxLjcxMjIzNCwzOC45NjQ2Njk0IDQwLjU0MDgwNTQsMzkuMDE0NjY5NCA0MC4zMzEyODE2LDM5LjAxNDY2OTQgTDQwLjMzMTI4MTYsMzkuMDE0NjY5NCBMMzcuNzU5ODUzLDM5LjAxNDY2OTQgTDM3Ljc1OTg1MywtMi45ODUzMzA2NSBaIiBpZD0icm93c19vbmUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuNzEyMjM0LCAxOC4wMTQ2NjkpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTMzLjcxMjIzNCwgLTE4LjAxNDY2OSkgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

row_count_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">13</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


The <a href="../../reference/Validate.col_count_match.html#pointblank.Validate.col_count_match" class="gdls-link"><code>Validate.col_count_match()</code></a> validation method checks if the number of columns in a table matches a specified count.


``` python
(
    pb.Validate(data=pb.load_dataset(dataset="small_table", tbl_type="polars"))
    .col_count_match(count=8)
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX2NvdW50X21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9jb3VudF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMjc1ODYyKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuNjM1Mjc2MiwxMy4wMTQ2Njk0IEw0NC42MzUyNzYyLDU0LjAxNDU4ODYgTDQyLjUzNTcxNTQsNTQuMDE0MjQ5MyBDNDIuNDE3MDgyMiw1NC4wMTEyMTc0IDQyLjEwMDgyMTksNTMuOTk1ODkyMiA0MS44ODE4NDg5LDUzLjc3OTE0ODEgQzQxLjc1NjYwMTUsNTMuNjU1MTc1NiA0MS42ODI4OTUzLDUzLjQ2MDMyMiA0MS42ODI4OTUzLDUzLjE4MTMzNiBMNDEuNjgyODk1Myw1My4xODEzMzYgTDQxLjY4Mjg5NTMsMTMuODQ4MDAyNyBDNDEuNjgyODk1MywxMy41NjUxNzMxIDQxLjc1ODkwOTQsMTMuMzY4ODI4MiA0MS44ODcwMzIxLDEzLjI0NTEyMzUgQzQyLjEyNjgxMiwxMy4wMTM2MTIgNDIuNDc2NjY1NCwxMy4wMTQ2Njk0IDQyLjU2Mzg0NzYsMTMuMDE0NjY5NCBMNDQuNjM1Mjc2MiwxMy4wMTQ2Njk0IFogTTUwLjYzMDYwMDYsMTMuMDE0NjY5NCBMNTAuNjMwNjAwNiw1NC4wMTQ2Njk0IEw0Ny43MzUxODk5LDU0LjAxNDY2OTQgTDQ3LjczNTE4OTksMTMuMDE0NjY5NCBMNTAuNjMwNjAwNiwxMy4wMTQ2Njk0IFogTTUzLjczMDUxNDMsMTMuMDE0NzUwMiBMNTUuODMwMDA4OCwxMy4wMTUwODk0IEM1NS45NDQwNTY2LDEzLjAxODA2NjcgNTYuMjM3NDgxMywxMy4wMzQxNTM0IDU2LjQ1NjAxNjUsMTMuMjI0MjgzOCBDNTYuNTcxNDgyOCwxMy4zMjQ3NDIxIDU2LjY1MjY2MTIsMTMuNDgwNzA3MiA1Ni42NzU4Mzg5LDEzLjcwNDE3MDcgTDU2LjY3NTgzODksMTMuNzA0MTcwNyBMNTYuNjgyODk1Myw1My4xODEzMzYgQzU2LjY4Mjg5NTMsNTMuNDYwMzIyIDU2LjYwOTE4OSw1My42NTUxNzU2IDU2LjQ4Mzk0MTYsNTMuNzc5MTQ4MSBDNTYuMjY0OTY4Nyw1My45OTU4OTIyIDU1Ljk0ODcwODMsNTQuMDExMjE3NCA1NS44MzAwNzUyLDU0LjAxNDI0OTMgTDU1LjgzMDA3NTIsNTQuMDE0MjQ5MyBMNTMuNzMwNTE0Myw1NC4wMTQ1ODg2IEw1My43MzA1MTQzLDEzLjAxNDc1MDIgWiIgaWQ9InJvd3NfdHdvIiBzdHJva2U9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDkuMTgyODk1LCAzMy41MTQ2NjkpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC00OS4xODI4OTUsIC0zMy41MTQ2NjkpICIgLz4KICAgICAgICAgICAgPGcgaWQ9InZlcnRpY2FsX2VxdWFsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzNC4xMzUxOTUsIDMzLjcyNjkwMykgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuMTM1MTk1LCAtMzMuNzI2OTAzKSB0cmFuc2xhdGUoMzAuNjM1MTk1LCAyOS4yMjY5MDMpIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS41OTMxODYzLDEyLjUxNDY2OTQgQzExLjM4MzY2MjUsMTIuNTE0NjY5NCAxMC4yMTIyMzQsMTIuNTY0NjY5NCAxMC4yMTIyMzQsMTMuODQ4MDAyNyBMMTAuMjEyMjM0LDUzLjE4MTMzNiBDMTAuMjEyMjM0LDU0LjQ2NDY2OTQgMTEuMzgzNjYyNSw1NC41MTQ2Njk0IDExLjU5MzE4NjMsNTQuNTE0NjY5NCBMMTQuMTY0NjE0OSw1NC41MTQ2Njk0IEwxNC4xNjQ2MTQ5LDEyLjUxNDY2OTQgTDExLjU5MzE4NjMsMTIuNTE0NjY5NCBaIE0yMC4xNzIxNzcxLDEyLjUxNDY2OTQgTDIwLjE3MjE3NzEsNTQuNTE0NjY5NCBMMTYuMjUyMjkwOCw1NC41MTQ2Njk0IEwxNi4yNTIyOTA4LDU0LjUxNDY2OTQgTDE2LjI1MjI5MDgsMTIuNTE0NjY5NCBMMTYuMjUyMjkwOCwxMi41MTQ2Njk0IEwyMC4xNzIxNzcxLDEyLjUxNDY2OTQgWiBNMjQuODY1NjE0OSwxMi41MTUwOTA0IEMyNS4xNDQ4Nzg2LDEyLjUyMTc2MyAyNi4yMTIyMzQsMTIuNjIzMDAyNyAyNi4yMTIyMzQsMTMuODQ4MDAyNyBMMjYuMjEyMjM0LDEzLjg0ODAwMjcgTDI2LjIxMjIzNCw1My4xODEzMzYgQzI2LjIxMjIzNCw1NC40NjQ2Njk0IDI1LjA0MDgwNTQsNTQuNTE0NjY5NCAyNC44MzEyODE2LDU0LjUxNDY2OTQgTDI0LjgzMTI4MTYsNTQuNTE0NjY5NCBMMjIuMjU5ODUzLDU0LjUxNDY2OTQgTDIyLjI1OTg1MywxMi41MTQ2Njk0IFoiIGlkPSJyb3dzX29uZSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxOC4yMTIyMzQsIDMzLjUxNDY2OSkgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE4LjIxMjIzNCwgLTMzLjUxNDY2OSkgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_count_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">8</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


Expectations on column and row counts can be useful in certain situations and they align nicely with schema checks.


## Validating Data Freshness

Late or missing data is one of the most common (and costly) data quality issues in production systems. When data pipelines fail silently or experience delays, downstream analytics and ML models can produce stale or misleading results. The <a href="../../reference/Validate.data_freshness.html#pointblank.Validate.data_freshness" class="gdls-link"><code>Validate.data_freshness()</code></a> validation method helps catch these issues early by verifying that your data contains recent records.

Data freshness validation works by checking a datetime column against a maximum allowed age. If the most recent timestamp in that column is older than the specified threshold, the validation fails. This simple check can prevent major downstream problems caused by stale data.

Here's an example that validates data is no older than 2 days:


``` python
import polars as pl
from datetime import datetime, timedelta

# Simulate a data feed that should be updated daily
recent_data = pl.DataFrame({
    "event": ["login", "purchase", "logout", "signup"],
    "event_time": [
        datetime.now() - timedelta(hours=1),
        datetime.now() - timedelta(hours=6),
        datetime.now() - timedelta(hours=12),
        datetime.now() - timedelta(hours=18),
    ],
    "user_id": [101, 102, 103, 104]
})

(
    pb.Validate(data=recent_data)
    .data_freshness(column="event_time", max_age="2d")
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5kYXRhX2ZyZXNobmVzczwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJkYXRhX2ZyZXNobmVzcyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICA8Y2lyY2xlIGlkPSJjbG9jay1mYWNlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgY3g9IjMyIiBjeT0iMzIiIHI9IjIwIj48L2NpcmNsZT4KICAgICAgICAgICAgPGxpbmUgeDE9IjMyIiB5MT0iMzIiIHgyPSIzMiIgeTI9IjE3LjIzOTQ1NDMiIGlkPSJob3VyLWhhbmQiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiPjwvbGluZT4KICAgICAgICAgICAgPGxpbmUgeDE9IjMyIiB5MT0iMzIiIHgyPSI0Mi41IiB5Mj0iMzIiIGlkPSJtaW51dGUtaGFuZCIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCI+PC9saW5lPgogICAgICAgICAgICA8Y2lyY2xlIGlkPSJjZW50ZXItZG90IiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIGN4PSIzMiIgY3k9IjMyIiByPSIyIj48L2NpcmNsZT4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

data_freshness()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">event_time</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">2d</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(freshness_details)</span> ✓ Most recent data: <code>2026-04-15 22:48:29</code> (age: <strong>1.0h</strong>, max allowed: 2d)</p></td>
</tr>
</tfoot>

</table>


The `max_age=` parameter accepts a flexible string format: `"30m"` for 30 minutes, `"6h"` for 6 hours, `"2d"` for 2 days, or `"1w"` for 1 week. You can also combine units: `"1d 12h"` for 1.5 days.

When validation succeeds, the report includes details about the data's age in the footer. When it fails, you'll see exactly how old the most recent data is and what threshold was exceeded. This context helps quickly diagnose whether you're dealing with a minor delay or a major pipeline failure.

Data freshness validation is particularly valuable for:

- monitoring ETL pipelines to catch failures before they cascade to reports and dashboards
- validating data feeds to ensure third-party data sources are delivering as expected
- including freshness checks in automated data quality tests as part of continuous integration
- building alerting systems that trigger notifications when critical data becomes stale

You might wonder why not just use <a href="../../reference/Validate.col_vals_gt.html#pointblank.Validate.col_vals_gt" class="gdls-link"><code>Validate.col_vals_gt()</code></a> with a datetime threshold. While that approach works, <a href="../../reference/Validate.data_freshness.html#pointblank.Validate.data_freshness" class="gdls-link"><code>Validate.data_freshness()</code></a> offers several advantages: the method name clearly communicates your intent, the `max_age=` string format (e.g., `"2d"`) is more readable than datetime arithmetic, it auto-generates meaningful validation briefs, the report footer shows helpful context about actual data age and thresholds, and timezone mismatches between your data and comparison time are handled gracefully with informative warnings.

> **Note: Note**
>
> When comparing timezone-aware and timezone-naive datetimes, Pointblank will include a warning in the validation report. For consistent results, ensure your data and comparison times use compatible timezone settings.


# 4. AI-Powered Validations

AI-powered validations use Large Language Models (LLMs) to validate data based on natural language criteria. This opens up new possibilities for complex validation rules that are difficult to express with traditional programmatic methods.


## Validating with Natural Language Prompts

The <a href="../../reference/Validate.prompt.html#pointblank.Validate.prompt" class="gdls-link"><code>Validate.prompt()</code></a> validation method allows you to describe validation criteria in plain language. The LLM interprets your prompt and evaluates each row, producing pass/fail results just like other Pointblank validation methods.

This is particularly useful for:

- Semantic checks (e.g., "descriptions should mention a product name")
- Context-dependent validation (e.g., "prices should be reasonable for the product category")
- Subjective quality assessments (e.g., "comments should be professional and constructive")
- Complex rules that would require extensive regex patterns or custom functions

Here's a simple example that validates whether text descriptions contain specific information:


``` python
import polars as pl

# Create sample data with product descriptions
products = pl.DataFrame({
    "product": ["Widget A", "Gadget B", "Tool C"],
    "description": [
        "High-quality widget made in USA",
        "Innovative gadget with warranty",
        "Professional tool"
    ],
    "price": [29.99, 49.99, 19.99]
})

# Validate that descriptions mention quality or features
(
    pb.Validate(data=products)
    .prompt(
        prompt="Each description should mention either quality, features, or warranty",
        columns_subset=["description"],
        model="anthropic:claude-opus-4-6"
    )
    .interrogate()
)
```


The `columns_subset=` parameter lets you specify which columns to include in the validation, improving performance and reducing API costs by only sending relevant data to the LLM.

**Note:** To use <a href="../../reference/Validate.prompt.html#pointblank.Validate.prompt" class="gdls-link"><code>Validate.prompt()</code></a>, you need to have the appropriate API credentials configured for your chosen LLM provider (Anthropic, OpenAI, Ollama, or AWS Bedrock).


# 5. Aggregate Validations

Aggregate validations operate on column-level statistics rather than individual row values. These methods compute an aggregate value (such as sum, average, or standard deviation) from a column and compare it against an expected value. Unlike row-level validations where each row is a test unit, aggregate validations treat the entire column as a single test unit that either passes or fails.

Pointblank provides three families of aggregate validation methods:

- **Sum validations** (<a href="../../reference/Validate.col_sum_eq.html#pointblank.Validate.col_sum_eq" class="gdls-link"><code>Validate.col_sum_eq()</code></a>, <a href="../../reference/Validate.col_sum_gt.html#pointblank.Validate.col_sum_gt" class="gdls-link"><code>Validate.col_sum_gt()</code></a>, <a href="../../reference/Validate.col_sum_lt.html#pointblank.Validate.col_sum_lt" class="gdls-link"><code>Validate.col_sum_lt()</code></a>, <a href="../../reference/Validate.col_sum_ge.html#pointblank.Validate.col_sum_ge" class="gdls-link"><code>Validate.col_sum_ge()</code></a>, <a href="../../reference/Validate.col_sum_le.html#pointblank.Validate.col_sum_le" class="gdls-link"><code>Validate.col_sum_le()</code></a>) for validating the sum of column values

- **Average validations** (<a href="../../reference/Validate.col_avg_eq.html#pointblank.Validate.col_avg_eq" class="gdls-link"><code>Validate.col_avg_eq()</code></a>, <a href="../../reference/Validate.col_avg_gt.html#pointblank.Validate.col_avg_gt" class="gdls-link"><code>Validate.col_avg_gt()</code></a>, <a href="../../reference/Validate.col_avg_lt.html#pointblank.Validate.col_avg_lt" class="gdls-link"><code>Validate.col_avg_lt()</code></a>, <a href="../../reference/Validate.col_avg_ge.html#pointblank.Validate.col_avg_ge" class="gdls-link"><code>Validate.col_avg_ge()</code></a>, <a href="../../reference/Validate.col_avg_le.html#pointblank.Validate.col_avg_le" class="gdls-link"><code>Validate.col_avg_le()</code></a>) for validating the mean of column values

- **Standard deviation validations** (<a href="../../reference/Validate.col_sd_eq.html#pointblank.Validate.col_sd_eq" class="gdls-link"><code>Validate.col_sd_eq()</code></a>, <a href="../../reference/Validate.col_sd_gt.html#pointblank.Validate.col_sd_gt" class="gdls-link"><code>Validate.col_sd_gt()</code></a>, <a href="../../reference/Validate.col_sd_lt.html#pointblank.Validate.col_sd_lt" class="gdls-link"><code>Validate.col_sd_lt()</code></a>, <a href="../../reference/Validate.col_sd_ge.html#pointblank.Validate.col_sd_ge" class="gdls-link"><code>Validate.col_sd_ge()</code></a>, <a href="../../reference/Validate.col_sd_le.html#pointblank.Validate.col_sd_le" class="gdls-link"><code>Validate.col_sd_le()</code></a>) for validating the standard deviation of column values

Each family supports the five comparison operators: equal to (`_eq`), greater than (`_gt`), less than (`_lt`), greater than or equal to (`_ge`), and less than or equal to (`_le`).


## Validating Column Sums

Here's an example validating that the sum of column `a` equals 55:


``` python
import polars as pl

agg_data = pl.DataFrame({
    "a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "b": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
})

(
    pb.Validate(data=agg_data)
    .col_sum_eq(columns="a", value=55)
    .col_sum_gt(columns="b", value=500)
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc3VtX2VxPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9zdW1fZXEiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InN1bSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTguODg5MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCw3LjE0IEMyLjY2OTMzMzMzLDcuMTQgMi4wMyw3LjAxNjMzMzMzIDEuNDcsNi43NjkgQzAuOTEsNi41MjE2NjY2NyAwLjQ2NjY2NjY2Nyw2LjE4OCAwLjE0LDUuNzY4IEMwLjA0NjY2NjY2NjcsNS42NDY2NjY2NyAwLDUuNTI1MzMzMzMgMCw1LjQwNCBDMCw1LjE5ODY2NjY3IDAuMDk4LDUuMDQ0NjY2NjcgMC4yOTQsNC45NDIgQzAuMzc4LDQuODk1MzMzMzMgMC40NjY2NjY2NjcsNC44NzIgMC41Niw0Ljg3MiBDMC43MTg2NjY2NjcsNC44NzIgMC44ODIsNC45NTYgMS4wNSw1LjEyNCBDMS4zNTgsNS40NTA2NjY2NyAxLjY5ODY2NjY3LDUuNjg4NjY2NjcgMi4wNzIsNS44MzggQzIuNDQ1MzMzMzMsNS45ODczMzMzMyAyLjg3NDY2NjY3LDYuMDYyIDMuMzYsNi4wNjIgQzMuODA4LDYuMDYyIDQuMTc0MzMzMzMsNS45ODk2NjY2NyA0LjQ1OSw1Ljg0NSBDNC43NDM2NjY2Nyw1LjcwMDMzMzMzIDQuODg2LDUuNTAyIDQuODg2LDUuMjUgQzQuODg2LDUuMDYzMzMzMzMgNC44MzIzMzMzMyw0LjkwOTMzMzMzIDQuNzI1LDQuNzg4IEM0LjYxNzY2NjY3LDQuNjY2NjY2NjcgNC40MzMzMzMzMyw0LjU1IDQuMTcyLDQuNDM4IEMzLjkxMDY2NjY3LDQuMzI2IDMuNSw0LjE3NjY2NjY3IDIuOTQsMy45OSBDMS45OTczMzMzMywzLjY5MTMzMzMzIDEuMzMyMzMzMzMsMy4zODMzMzMzMyAwLjk0NSwzLjA2NiBDMC41NTc2NjY2NjcsMi43NDg2NjY2NyAwLjM2NCwyLjMzMzMzMzMzIDAuMzY0LDEuODIgQzAuMzY0LDEuMjY5MzMzMzMgMC42MDQzMzMzMzMsMC44MjgzMzMzMzMgMS4wODUsMC40OTcgQzEuNTY1NjY2NjcsMC4xNjU2NjY2NjcgMi4yMTY2NjY2NywwIDMuMDM4LDAgQzMuNjQ0NjY2NjcsMCA0LjIwNywwLjEwMDMzMzMzMyA0LjcyNSwwLjMwMSBDNS4yNDMsMC41MDE2NjY2NjcgNS42MjgsMC43NzQ2NjY2NjcgNS44OCwxLjEyIEM1Ljk2NCwxLjIzMiA2LjAwNiwxLjM0ODY2NjY3IDYuMDA2LDEuNDcgQzYuMDA2LDEuNjI4NjY2NjcgNS45MjIsMS43NjQgNS43NTQsMS44NzYgQzUuNjMyNjY2NjcsMS45NTA2NjY2NyA1LjUxMTMzMzMzLDEuOTg4IDUuMzksMS45ODggQzUuMjAzMzMzMzMsMS45ODggNS4wMjYsMS45MDQgNC44NTgsMS43MzYgQzQuNjM0LDEuNTEyIDQuMzc1LDEuMzQ2MzMzMzMgNC4wODEsMS4yMzkgQzMuNzg3LDEuMTMxNjY2NjcgMy40MzQ2NjY2NywxLjA3OCAzLjAyNCwxLjA3OCBDMi41ODUzMzMzMywxLjA3OCAyLjI1MTY2NjY3LDEuMTQzMzMzMzMgMi4wMjMsMS4yNzQgQzEuNzk0MzMzMzMsMS40MDQ2NjY2NyAxLjY4LDEuNTkxMzMzMzMgMS42OCwxLjgzNCBDMS42OCwyLjAyMDY2NjY3IDEuNzMxMzMzMzMsMi4xNyAxLjgzNCwyLjI4MiBDMS45MzY2NjY2NywyLjM5NCAyLjExNCwyLjUwMTMzMzMzIDIuMzY2LDIuNjA0IEMyLjYxOCwyLjcwNjY2NjY3IDMuMDM4LDIuODUxMzMzMzMgMy42MjYsMy4wMzggQzQuMjc5MzMzMzMsMy4yNDMzMzMzMyA0Ljc5MDMzMzMzLDMuNDQ4NjY2NjcgNS4xNTksMy42NTQgQzUuNTI3NjY2NjcsMy44NTkzMzMzMyA1Ljc5MTMzMzMzLDQuMDg4IDUuOTUsNC4zNCBDNi4xMDg2NjY2Nyw0LjU5MiA2LjE4OCw0Ljg5NTMzMzMzIDYuMTg4LDUuMjUgQzYuMTg4LDUuODE5MzMzMzMgNS45MzM2NjY2Nyw2LjI3NjY2NjY3IDUuNDI1LDYuNjIyIEM0LjkxNjMzMzMzLDYuOTY3MzMzMzMgNC4yMzczMzMzMyw3LjE0IDMuMzg4LDcuMTQgWiIgaWQ9InMiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTAuNTU2LDcuMTQgQzkuNjIyNjY2NjcsNy4xNCA4Ljg5NDY2NjY3LDYuODcxNjY2NjcgOC4zNzIsNi4zMzUgQzcuODQ5MzMzMzMsNS43OTgzMzMzMyA3LjU4OCw1LjA1ODY2NjY3IDcuNTg4LDQuMTE2IEw3LjU4OCwwLjY1OCBDNy41ODgsMC40NTI2NjY2NjcgNy42NDYzMzMzMywwLjI5MTY2NjY2NyA3Ljc2MywwLjE3NSBDNy44Nzk2NjY2NywwLjA1ODMzMzMzMzMgOC4wMzYsMCA4LjIzMiwwIEM4LjQzNzMzMzMzLDAgOC41OTYsMC4wNTgzMzMzMzMzIDguNzA4LDAuMTc1IEM4LjgyLDAuMjkxNjY2NjY3IDguODc2LDAuNDUyNjY2NjY3IDguODc2LDAuNjU4IEw4Ljg3Niw0LjExNiBDOC44NzYsNC43MzIgOS4wNDE2NjY2Nyw1LjIxMDMzMzMzIDkuMzczLDUuNTUxIEM5LjcwNDMzMzMzLDUuODkxNjY2NjcgMTAuMTY4NjY2Nyw2LjA2MiAxMC43NjYsNi4wNjIgQzExLjQ5NCw2LjA2MiAxMi4wNzk2NjY3LDUuODI4NjY2NjcgMTIuNTIzLDUuMzYyIEMxMi45NjYzMzMzLDQuODk1MzMzMzMgMTMuMTg4LDQuMjc0NjY2NjcgMTMuMTg4LDMuNSBMMTMuMTg4LDAuNjU4IEMxMy4xODgsMC40NTI2NjY2NjcgMTMuMjQ2MzMzMywwLjI5MTY2NjY2NyAxMy4zNjMsMC4xNzUgQzEzLjQ3OTY2NjcsMC4wNTgzMzMzMzMzIDEzLjY0MDY2NjcsMCAxMy44NDYsMCBDMTQuMDUxMzMzMywwIDE0LjIwNzY2NjcsMC4wNTYgMTQuMzE1LDAuMTY4IEMxNC40MjIzMzMzLDAuMjggMTQuNDc2LDAuNDQzMzMzMzMzIDE0LjQ3NiwwLjY1OCBMMTQuNDc2LDYuNDgyIEMxNC40NzYsNi42OTY2NjY2NyAxNC40MjIzMzMzLDYuODYgMTQuMzE1LDYuOTcyIEMxNC4yMDc2NjY3LDcuMDg0IDE0LjA1MTMzMzMsNy4xNCAxMy44NDYsNy4xNCBDMTMuNjQwNjY2Nyw3LjE0IDEzLjQ4Miw3LjA4MTY2NjY3IDEzLjM3LDYuOTY1IEMxMy4yNTgsNi44NDgzMzMzMyAxMy4yMDIsNi42ODczMzMzMyAxMy4yMDIsNi40ODIgTDEzLjIwMiw1Ljc5NiBDMTIuNTAyLDYuNjkyIDExLjYyLDcuMTQgMTAuNTU2LDcuMTQgWiIgaWQ9InUiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTYuODQyLDcuMTQgQzE2LjYzNjY2NjcsNy4xNCAxNi40NzgsNy4wODE2NjY2NyAxNi4zNjYsNi45NjUgQzE2LjI1NCw2Ljg0ODMzMzMzIDE2LjE5OCw2LjY4NzMzMzMzIDE2LjE5OCw2LjQ4MiBMMTYuMTk4LDAuNjU4IEMxNi4xOTgsMC40NTI2NjY2NjcgMTYuMjU0LDAuMjkxNjY2NjY3IDE2LjM2NiwwLjE3NSBDMTYuNDc4LDAuMDU4MzMzMzMzMyAxNi42MzY2NjY3LDAgMTYuODQyLDAgQzE3LjA0NzMzMzMsMCAxNy4yMDgzMzMzLDAuMDU4MzMzMzMzMyAxNy4zMjUsMC4xNzUgQzE3LjQ0MTY2NjcsMC4yOTE2NjY2NjcgMTcuNSwwLjQ1MjY2NjY2NyAxNy41LDAuNjU4IEwxNy41LDEuMzAyIEMxNy43MzMzMzMzLDAuODkxMzMzMzMzIDE4LjAzNjY2NjcsMC41NzE2NjY2NjcgMTguNDEsMC4zNDMgQzE4Ljc4MzMzMzMsMC4xMTQzMzMzMzMgMTkuMTgsMCAxOS42LDAgQzIwLjAyOTMzMzMsMCAyMC40MDczMzMzLDAuMTE0MzMzMzMzIDIwLjczNCwwLjM0MyBDMjEuMDYwNjY2NywwLjU3MTY2NjY2NyAyMS4zMDMzMzMzLDAuODkxMzMzMzMzIDIxLjQ2MiwxLjMwMiBDMjIuMTQzMzMzMywwLjQzNCAyMi45MTMzMzMzLDAgMjMuNzcyLDAgQzI0LjUwOTMzMzMsMCAyNS4wNzE2NjY3LDAuMjM4IDI1LjQ1OSwwLjcxNCBDMjUuODQ2MzMzMywxLjE5IDI2LjA0LDEuODgwNjY2NjcgMjYuMDQsMi43ODYgTDI2LjA0LDYuNDgyIEMyNi4wNCw2LjY4NzMzMzMzIDI1Ljk4MTY2NjcsNi44NDgzMzMzMyAyNS44NjUsNi45NjUgQzI1Ljc0ODMzMzMsNy4wODE2NjY2NyAyNS41OTIsNy4xNCAyNS4zOTYsNy4xNCBDMjUuMTkwNjY2Nyw3LjE0IDI1LjAzMiw3LjA4NCAyNC45Miw2Ljk3MiBDMjQuODA4LDYuODYgMjQuNzUyLDYuNjk2NjY2NjcgMjQuNzUyLDYuNDgyIEwyNC43NTIsMi43MyBDMjQuNzUyLDIuMTc5MzMzMzMgMjQuNjU4NjY2NywxLjc2NjMzMzMzIDI0LjQ3MiwxLjQ5MSBDMjQuMjg1MzMzMywxLjIxNTY2NjY3IDIzLjk3NzMzMzMsMS4wNzggMjMuNTQ4LDEuMDc4IEMyMi44MDEzMzMzLDEuMDc4IDIyLjIzNjY2NjcsMS41MDI2NjY2NyAyMS44NTQsMi4zNTIgTDIxLjg1NCw2LjQ4MiBDMjEuODU0LDYuNjg3MzMzMzMgMjEuNzk1NjY2Nyw2Ljg0ODMzMzMzIDIxLjY3OSw2Ljk2NSBDMjEuNTYyMzMzMyw3LjA4MTY2NjY3IDIxLjQwMTMzMzMsNy4xNCAyMS4xOTYsNy4xNCBDMjAuOTkwNjY2Nyw3LjE0IDIwLjgzNDMzMzMsNy4wODQgMjAuNzI3LDYuOTcyIEMyMC42MTk2NjY3LDYuODYgMjAuNTY2LDYuNjk2NjY2NjcgMjAuNTY2LDYuNDgyIEwyMC41NjYsMi45NjggQzIwLjU2NiwyLjM1MiAyMC40NjMzMzMzLDEuODgzIDIwLjI1OCwxLjU2MSBDMjAuMDUyNjY2NywxLjIzOSAxOS43NTQsMS4wNzggMTkuMzYyLDEuMDc4IEMxOC44MTEzMzMzLDEuMDc4IDE4LjM2MzMzMzMsMS4yOTUgMTguMDE4LDEuNzI5IEMxNy42NzI2NjY3LDIuMTYzIDE3LjUsMi43MjUzMzMzMyAxNy41LDMuNDE2IEwxNy41LDYuNDgyIEMxNy41LDYuNjg3MzMzMzMgMTcuNDQxNjY2Nyw2Ljg0ODMzMzMzIDE3LjMyNSw2Ljk2NSBDMTcuMjA4MzMzMyw3LjA4MTY2NjY3IDE3LjA0NzMzMzMsNy4xNCAxNi44NDIsNy4xNCBaIiBpZD0ibSIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgICAgICA8cGF0aCBkPSJNNDYuNjgxODE4MiwxMCBMMTcuMzE4MTgxOCwxMCBDMTYuMDQxMzcwNiwxMCAxNSwxMS4wNDEzNzA1IDE1LDEyLjMxODE4MTggTDE1LDQxLjY4MTgxODIgQzE1LDQyLjk1ODYyOTUgMTYuMDQxMzcwNiw0NCAxNy4zMTgxODE4LDQ0IEw0Ni42ODE4MTgyLDQ0IEM0Ny45NTg2Mjk1LDQ0IDQ5LDQyLjk1ODYyOTUgNDksNDEuNjgxODE4MiBMNDksMTIuMzE4MTgxOCBDNDksMTEuMDQxMzcwNSA0Ny45NTg2Mjk1LDEwIDQ2LjY4MTgxODIsMTAgWiBNNDIuMDQ1NDU0NiwzMC44NjM2MzY0IEwyMS45NTQ1NDU1LDMwLjg2MzYzNjQgTDIxLjk1NDU0NTUsMjkuMzE4MTgxOCBMNDIuMDQ1NDU0NiwyOS4zMTgxODE4IEw0Mi4wNDU0NTQ2LDMwLjg2MzYzNjQgWiBNNDIuMDQ1NDU0NiwyNC42ODE4MTgyIEwyMS45NTQ1NDU1LDI0LjY4MTgxODIgTDIxLjk1NDU0NTUsMjMuMTM2MzYzNiBMNDIuMDQ1NDU0NiwyMy4xMzYzNjM2IEw0Mi4wNDU0NTQ2LDI0LjY4MTgxODIgWiIgaWQ9ImVxdWFscyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_sum_eq()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">55</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc3VtX2d0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9zdW1fZ3QiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InN1bSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTguODg5MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCw3LjE0IEMyLjY2OTMzMzMzLDcuMTQgMi4wMyw3LjAxNjMzMzMzIDEuNDcsNi43NjkgQzAuOTEsNi41MjE2NjY2NyAwLjQ2NjY2NjY2Nyw2LjE4OCAwLjE0LDUuNzY4IEMwLjA0NjY2NjY2NjcsNS42NDY2NjY2NyAwLDUuNTI1MzMzMzMgMCw1LjQwNCBDMCw1LjE5ODY2NjY3IDAuMDk4LDUuMDQ0NjY2NjcgMC4yOTQsNC45NDIgQzAuMzc4LDQuODk1MzMzMzMgMC40NjY2NjY2NjcsNC44NzIgMC41Niw0Ljg3MiBDMC43MTg2NjY2NjcsNC44NzIgMC44ODIsNC45NTYgMS4wNSw1LjEyNCBDMS4zNTgsNS40NTA2NjY2NyAxLjY5ODY2NjY3LDUuNjg4NjY2NjcgMi4wNzIsNS44MzggQzIuNDQ1MzMzMzMsNS45ODczMzMzMyAyLjg3NDY2NjY3LDYuMDYyIDMuMzYsNi4wNjIgQzMuODA4LDYuMDYyIDQuMTc0MzMzMzMsNS45ODk2NjY2NyA0LjQ1OSw1Ljg0NSBDNC43NDM2NjY2Nyw1LjcwMDMzMzMzIDQuODg2LDUuNTAyIDQuODg2LDUuMjUgQzQuODg2LDUuMDYzMzMzMzMgNC44MzIzMzMzMyw0LjkwOTMzMzMzIDQuNzI1LDQuNzg4IEM0LjYxNzY2NjY3LDQuNjY2NjY2NjcgNC40MzMzMzMzMyw0LjU1IDQuMTcyLDQuNDM4IEMzLjkxMDY2NjY3LDQuMzI2IDMuNSw0LjE3NjY2NjY3IDIuOTQsMy45OSBDMS45OTczMzMzMywzLjY5MTMzMzMzIDEuMzMyMzMzMzMsMy4zODMzMzMzMyAwLjk0NSwzLjA2NiBDMC41NTc2NjY2NjcsMi43NDg2NjY2NyAwLjM2NCwyLjMzMzMzMzMzIDAuMzY0LDEuODIgQzAuMzY0LDEuMjY5MzMzMzMgMC42MDQzMzMzMzMsMC44MjgzMzMzMzMgMS4wODUsMC40OTcgQzEuNTY1NjY2NjcsMC4xNjU2NjY2NjcgMi4yMTY2NjY2NywwIDMuMDM4LDAgQzMuNjQ0NjY2NjcsMCA0LjIwNywwLjEwMDMzMzMzMyA0LjcyNSwwLjMwMSBDNS4yNDMsMC41MDE2NjY2NjcgNS42MjgsMC43NzQ2NjY2NjcgNS44OCwxLjEyIEM1Ljk2NCwxLjIzMiA2LjAwNiwxLjM0ODY2NjY3IDYuMDA2LDEuNDcgQzYuMDA2LDEuNjI4NjY2NjcgNS45MjIsMS43NjQgNS43NTQsMS44NzYgQzUuNjMyNjY2NjcsMS45NTA2NjY2NyA1LjUxMTMzMzMzLDEuOTg4IDUuMzksMS45ODggQzUuMjAzMzMzMzMsMS45ODggNS4wMjYsMS45MDQgNC44NTgsMS43MzYgQzQuNjM0LDEuNTEyIDQuMzc1LDEuMzQ2MzMzMzMgNC4wODEsMS4yMzkgQzMuNzg3LDEuMTMxNjY2NjcgMy40MzQ2NjY2NywxLjA3OCAzLjAyNCwxLjA3OCBDMi41ODUzMzMzMywxLjA3OCAyLjI1MTY2NjY3LDEuMTQzMzMzMzMgMi4wMjMsMS4yNzQgQzEuNzk0MzMzMzMsMS40MDQ2NjY2NyAxLjY4LDEuNTkxMzMzMzMgMS42OCwxLjgzNCBDMS42OCwyLjAyMDY2NjY3IDEuNzMxMzMzMzMsMi4xNyAxLjgzNCwyLjI4MiBDMS45MzY2NjY2NywyLjM5NCAyLjExNCwyLjUwMTMzMzMzIDIuMzY2LDIuNjA0IEMyLjYxOCwyLjcwNjY2NjY3IDMuMDM4LDIuODUxMzMzMzMgMy42MjYsMy4wMzggQzQuMjc5MzMzMzMsMy4yNDMzMzMzMyA0Ljc5MDMzMzMzLDMuNDQ4NjY2NjcgNS4xNTksMy42NTQgQzUuNTI3NjY2NjcsMy44NTkzMzMzMyA1Ljc5MTMzMzMzLDQuMDg4IDUuOTUsNC4zNCBDNi4xMDg2NjY2Nyw0LjU5MiA2LjE4OCw0Ljg5NTMzMzMzIDYuMTg4LDUuMjUgQzYuMTg4LDUuODE5MzMzMzMgNS45MzM2NjY2Nyw2LjI3NjY2NjY3IDUuNDI1LDYuNjIyIEM0LjkxNjMzMzMzLDYuOTY3MzMzMzMgNC4yMzczMzMzMyw3LjE0IDMuMzg4LDcuMTQgWiIgaWQ9InMiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTAuNTU2LDcuMTQgQzkuNjIyNjY2NjcsNy4xNCA4Ljg5NDY2NjY3LDYuODcxNjY2NjcgOC4zNzIsNi4zMzUgQzcuODQ5MzMzMzMsNS43OTgzMzMzMyA3LjU4OCw1LjA1ODY2NjY3IDcuNTg4LDQuMTE2IEw3LjU4OCwwLjY1OCBDNy41ODgsMC40NTI2NjY2NjcgNy42NDYzMzMzMywwLjI5MTY2NjY2NyA3Ljc2MywwLjE3NSBDNy44Nzk2NjY2NywwLjA1ODMzMzMzMzMgOC4wMzYsMCA4LjIzMiwwIEM4LjQzNzMzMzMzLDAgOC41OTYsMC4wNTgzMzMzMzMzIDguNzA4LDAuMTc1IEM4LjgyLDAuMjkxNjY2NjY3IDguODc2LDAuNDUyNjY2NjY3IDguODc2LDAuNjU4IEw4Ljg3Niw0LjExNiBDOC44NzYsNC43MzIgOS4wNDE2NjY2Nyw1LjIxMDMzMzMzIDkuMzczLDUuNTUxIEM5LjcwNDMzMzMzLDUuODkxNjY2NjcgMTAuMTY4NjY2Nyw2LjA2MiAxMC43NjYsNi4wNjIgQzExLjQ5NCw2LjA2MiAxMi4wNzk2NjY3LDUuODI4NjY2NjcgMTIuNTIzLDUuMzYyIEMxMi45NjYzMzMzLDQuODk1MzMzMzMgMTMuMTg4LDQuMjc0NjY2NjcgMTMuMTg4LDMuNSBMMTMuMTg4LDAuNjU4IEMxMy4xODgsMC40NTI2NjY2NjcgMTMuMjQ2MzMzMywwLjI5MTY2NjY2NyAxMy4zNjMsMC4xNzUgQzEzLjQ3OTY2NjcsMC4wNTgzMzMzMzMzIDEzLjY0MDY2NjcsMCAxMy44NDYsMCBDMTQuMDUxMzMzMywwIDE0LjIwNzY2NjcsMC4wNTYgMTQuMzE1LDAuMTY4IEMxNC40MjIzMzMzLDAuMjggMTQuNDc2LDAuNDQzMzMzMzMzIDE0LjQ3NiwwLjY1OCBMMTQuNDc2LDYuNDgyIEMxNC40NzYsNi42OTY2NjY2NyAxNC40MjIzMzMzLDYuODYgMTQuMzE1LDYuOTcyIEMxNC4yMDc2NjY3LDcuMDg0IDE0LjA1MTMzMzMsNy4xNCAxMy44NDYsNy4xNCBDMTMuNjQwNjY2Nyw3LjE0IDEzLjQ4Miw3LjA4MTY2NjY3IDEzLjM3LDYuOTY1IEMxMy4yNTgsNi44NDgzMzMzMyAxMy4yMDIsNi42ODczMzMzMyAxMy4yMDIsNi40ODIgTDEzLjIwMiw1Ljc5NiBDMTIuNTAyLDYuNjkyIDExLjYyLDcuMTQgMTAuNTU2LDcuMTQgWiIgaWQ9InUiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTYuODQyLDcuMTQgQzE2LjYzNjY2NjcsNy4xNCAxNi40NzgsNy4wODE2NjY2NyAxNi4zNjYsNi45NjUgQzE2LjI1NCw2Ljg0ODMzMzMzIDE2LjE5OCw2LjY4NzMzMzMzIDE2LjE5OCw2LjQ4MiBMMTYuMTk4LDAuNjU4IEMxNi4xOTgsMC40NTI2NjY2NjcgMTYuMjU0LDAuMjkxNjY2NjY3IDE2LjM2NiwwLjE3NSBDMTYuNDc4LDAuMDU4MzMzMzMzMyAxNi42MzY2NjY3LDAgMTYuODQyLDAgQzE3LjA0NzMzMzMsMCAxNy4yMDgzMzMzLDAuMDU4MzMzMzMzMyAxNy4zMjUsMC4xNzUgQzE3LjQ0MTY2NjcsMC4yOTE2NjY2NjcgMTcuNSwwLjQ1MjY2NjY2NyAxNy41LDAuNjU4IEwxNy41LDEuMzAyIEMxNy43MzMzMzMzLDAuODkxMzMzMzMzIDE4LjAzNjY2NjcsMC41NzE2NjY2NjcgMTguNDEsMC4zNDMgQzE4Ljc4MzMzMzMsMC4xMTQzMzMzMzMgMTkuMTgsMCAxOS42LDAgQzIwLjAyOTMzMzMsMCAyMC40MDczMzMzLDAuMTE0MzMzMzMzIDIwLjczNCwwLjM0MyBDMjEuMDYwNjY2NywwLjU3MTY2NjY2NyAyMS4zMDMzMzMzLDAuODkxMzMzMzMzIDIxLjQ2MiwxLjMwMiBDMjIuMTQzMzMzMywwLjQzNCAyMi45MTMzMzMzLDAgMjMuNzcyLDAgQzI0LjUwOTMzMzMsMCAyNS4wNzE2NjY3LDAuMjM4IDI1LjQ1OSwwLjcxNCBDMjUuODQ2MzMzMywxLjE5IDI2LjA0LDEuODgwNjY2NjcgMjYuMDQsMi43ODYgTDI2LjA0LDYuNDgyIEMyNi4wNCw2LjY4NzMzMzMzIDI1Ljk4MTY2NjcsNi44NDgzMzMzMyAyNS44NjUsNi45NjUgQzI1Ljc0ODMzMzMsNy4wODE2NjY2NyAyNS41OTIsNy4xNCAyNS4zOTYsNy4xNCBDMjUuMTkwNjY2Nyw3LjE0IDI1LjAzMiw3LjA4NCAyNC45Miw2Ljk3MiBDMjQuODA4LDYuODYgMjQuNzUyLDYuNjk2NjY2NjcgMjQuNzUyLDYuNDgyIEwyNC43NTIsMi43MyBDMjQuNzUyLDIuMTc5MzMzMzMgMjQuNjU4NjY2NywxLjc2NjMzMzMzIDI0LjQ3MiwxLjQ5MSBDMjQuMjg1MzMzMywxLjIxNTY2NjY3IDIzLjk3NzMzMzMsMS4wNzggMjMuNTQ4LDEuMDc4IEMyMi44MDEzMzMzLDEuMDc4IDIyLjIzNjY2NjcsMS41MDI2NjY2NyAyMS44NTQsMi4zNTIgTDIxLjg1NCw2LjQ4MiBDMjEuODU0LDYuNjg3MzMzMzMgMjEuNzk1NjY2Nyw2Ljg0ODMzMzMzIDIxLjY3OSw2Ljk2NSBDMjEuNTYyMzMzMyw3LjA4MTY2NjY3IDIxLjQwMTMzMzMsNy4xNCAyMS4xOTYsNy4xNCBDMjAuOTkwNjY2Nyw3LjE0IDIwLjgzNDMzMzMsNy4wODQgMjAuNzI3LDYuOTcyIEMyMC42MTk2NjY3LDYuODYgMjAuNTY2LDYuNjk2NjY2NjcgMjAuNTY2LDYuNDgyIEwyMC41NjYsMi45NjggQzIwLjU2NiwyLjM1MiAyMC40NjMzMzMzLDEuODgzIDIwLjI1OCwxLjU2MSBDMjAuMDUyNjY2NywxLjIzOSAxOS43NTQsMS4wNzggMTkuMzYyLDEuMDc4IEMxOC44MTEzMzMzLDEuMDc4IDE4LjM2MzMzMzMsMS4yOTUgMTguMDE4LDEuNzI5IEMxNy42NzI2NjY3LDIuMTYzIDE3LjUsMi43MjUzMzMzMyAxNy41LDMuNDE2IEwxNy41LDYuNDgyIEMxNy41LDYuNjg3MzMzMzMgMTcuNDQxNjY2Nyw2Ljg0ODMzMzMzIDE3LjMyNSw2Ljk2NSBDMTcuMjA4MzMzMyw3LjA4MTY2NjY3IDE3LjA0NzMzMzMsNy4xNCAxNi44NDIsNy4xNCBaIiBpZD0ibSIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuOTUyMzgxLDEwIEwxOS4wNDc2MTkxLDEwIEMxNi44MTMzMzMzLDEwIDE1LDExLjgxMzMzMzMgMTUsMTQuMDQ3NjE5IEwxNSwzOS45NTIzODEgQzE1LDQyLjE4NjY2NjcgMTYuODEzMzMzMyw0NCAxOS4wNDc2MTkxLDQ0IEw0NC45NTIzODEsNDQgQzQ3LjE4NjY2NjcsNDQgNDksNDIuMTg2NjY2NyA0OSwzOS45NTIzODEgTDQ5LDE0LjA0NzYxOSBDNDksMTEuODEzMzMzMyA0Ny4xODY2NjY3LDEwIDQ0Ljk1MjM4MSwxMCBaIE0yNi43OTQ3NjE5LDM2LjU2ODU3MTQgTDI1Ljg3MTkwNDgsMzUuMjQwOTUyNCBMMzcuODYwOTUyNCwyNyBMMjUuODcxOTA0OCwxOC43NTkwNDc2IEwyNi43OTQ3NjE5LDE3LjQzMTQyODYgTDQwLjcxODU3MTQsMjcgTDI2Ljc5NDc2MTksMzYuNTY4NTcxNCBaIiBpZD0iZ3JlYXRlcl90aGFuIiBmaWxsPSIjMDAwMDAwIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_sum_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">500</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


## Validating Column Averages

Average validations are useful for ensuring that typical values remain within expected bounds:


``` python
(
    pb.Validate(data=agg_data)
    .col_avg_eq(columns="a", value=5.5)
    .col_avg_ge(columns="b", value=50)
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfYXZnX2VxPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9hdmdfZXEiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9ImF2ZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjAuMDU4MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yLjk0LDcuMTQgQzIuMDM0NjY2NjcsNy4xNCAxLjMxODMzMzMzLDYuOTM0NjY2NjcgMC43OTEsNi41MjQgQzAuMjYzNjY2NjY3LDYuMTEzMzMzMzMgMCw1LjU1OCAwLDQuODU4IEMwLDQuMTU4IDAuMjc3NjY2NjY3LDMuNjAyNjY2NjcgMC44MzMsMy4xOTIgQzEuMzg4MzMzMzMsMi43ODEzMzMzMyAyLjE0NjY2NjY3LDIuNTc2IDMuMTA4LDIuNTc2IEMzLjYxMiwyLjU3NiA0LjA2MjMzMzMzLDIuNjQxMzMzMzMgNC40NTksMi43NzIgQzQuODU1NjY2NjcsMi45MDI2NjY2NyA1LjE3NTMzMzMzLDMuMDggNS40MTgsMy4zMDQgTDUuNDE4LDIuNjA0IEM1LjQxOCwyLjEgNS4yNTcsMS43MTUgNC45MzUsMS40NDkgQzQuNjEzLDEuMTgzIDQuMTM5MzMzMzMsMS4wNSAzLjUxNCwxLjA1IEMyLjY4MzMzMzMzLDEuMDUgMS45NjQ2NjY2NywxLjI0MTMzMzMzIDEuMzU4LDEuNjI0IEMxLjE4MDY2NjY3LDEuNzI2NjY2NjcgMS4wMjY2NjY2NywxLjc3OCAwLjg5NiwxLjc3OCBDMC43NzQ2NjY2NjcsMS43NzggMC42NTMzMzMzMzMsMS43MjY2NjY2NyAwLjUzMiwxLjYyNCBDMC40MjkzMzMzMzMsMS41MjEzMzMzMyAwLjM3OCwxLjQgMC4zNzgsMS4yNiBDMC4zNzgsMS4xMDEzMzMzMyAwLjQ1NzMzMzMzMywwLjk1NjY2NjY2NyAwLjYxNiwwLjgyNiBDMC45NTIsMC41NzQgMS4zNzY2NjY2NywwLjM3MzMzMzMzMyAxLjg5LDAuMjI0IEMyLjQwMzMzMzMzLDAuMDc0NjY2NjY2NyAyLjk0LDAgMy41LDAgQzQuNTE3MzMzMzMsMCA1LjMxMDY2NjY3LDAuMjMxIDUuODgsMC42OTMgQzYuNDQ5MzMzMzMsMS4xNTUgNi43MzQsMS44MDEzMzMzMyA2LjczNCwyLjYzMiBMNi43MzQsNi41MSBDNi43MzQsNi43MTUzMzMzMyA2LjY4MDMzMzMzLDYuODcxNjY2NjcgNi41NzMsNi45NzkgQzYuNDY1NjY2NjcsNy4wODYzMzMzMyA2LjMwOTMzMzMzLDcuMTQgNi4xMDQsNy4xNCBDNS44OTg2NjY2Nyw3LjE0IDUuNzQsNy4wODYzMzMzMyA1LjYyOCw2Ljk3OSBDNS41MTYsNi44NzE2NjY2NyA1LjQ2LDYuNzI0NjY2NjcgNS40Niw2LjUzOCBMNS40Niw2LjE0NiBDNC45LDYuODA4NjY2NjcgNC4wNiw3LjE0IDIuOTQsNy4xNCBaIE0zLjE2NCw2LjE0NiBDMy44NDUzMzMzMyw2LjE0NiA0LjM5MzY2NjY3LDYuMDIyMzMzMzMgNC44MDksNS43NzUgQzUuMjI0MzMzMzMsNS41Mjc2NjY2NyA1LjQzMiw1LjIwMzMzMzMzIDUuNDMyLDQuODAyIEM1LjQzMiw0LjQxIDUuMjMzNjY2NjcsNC4xMDQzMzMzMyA0LjgzNywzLjg4NSBDNC40NDAzMzMzMywzLjY2NTY2NjY3IDMuODkyLDMuNTU2IDMuMTkyLDMuNTU2IEMyLjU4NTMzMzMzLDMuNTU2IDIuMTE4NjY2NjcsMy42NjggMS43OTIsMy44OTIgQzEuNDY1MzMzMzMsNC4xMTYgMS4zMDIsNC40MzggMS4zMDIsNC44NTggQzEuMzAyLDUuMjY4NjY2NjcgMS40NjMsNS41ODYgMS43ODUsNS44MSBDMi4xMDcsNi4wMzQgMi41NjY2NjY2Nyw2LjE0NiAzLjE2NCw2LjE0NiBaIiBpZD0iYSIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS40OTQsNy4xNCBDMTAuNTMyNjY2Nyw3LjE0IDkuODUxMzMzMzMsNi41NjEzMzMzMyA5LjQ1LDUuNDA0IEw3Ljc5OCwwLjc4NCBDNy43NywwLjcwOTMzMzMzMyA3Ljc1NiwwLjY0NCA3Ljc1NiwwLjU4OCBDNy43NTYsMC40NjY2NjY2NjcgNy43OTgsMC4zNTcgNy44ODIsMC4yNTkgQzcuOTY2LDAuMTYxIDguMDc4LDAuMDg4NjY2NjY2NyA4LjIxOCwwLjA0MiBDOC4zMDIsMC4wMTQgOC4zODYsMCA4LjQ3LDAgQzguNjEsMCA4LjczMTMzMzMzLDAuMDM1IDguODM0LDAuMTA1IEM4LjkzNjY2NjY3LDAuMTc1IDkuMDA2NjY2NjcsMC4yNzA2NjY2NjcgOS4wNDQsMC4zOTIgTDEwLjczOCw1LjEzOCBDMTAuODU5MzMzMyw1LjQ4MzMzMzMzIDEwLjk3Niw1LjcyNiAxMS4wODgsNS44NjYgQzExLjIsNi4wMDYgMTEuMzM1MzMzMyw2LjA3NiAxMS40OTQsNi4wNzYgQzExLjY1MjY2NjcsNi4wNzYgMTEuNzg4LDYuMDA4MzMzMzMgMTEuOSw1Ljg3MyBDMTIuMDEyLDUuNzM3NjY2NjcgMTIuMTI4NjY2Nyw1LjUwMiAxMi4yNSw1LjE2NiBMMTMuOTQ0LDAuMzkyIEMxMy45OTA2NjY3LDAuMjcwNjY2NjY3IDE0LjA2MywwLjE3NzMzMzMzMyAxNC4xNjEsMC4xMTIgQzE0LjI1OSwwLjA0NjY2NjY2NjcgMTQuMzc4LDAuMDE0IDE0LjUxOCwwLjAxNCBDMTQuNzMyNjY2NywwLjAxNCAxNC45MDUzMzMzLDAuMDcyMzMzMzMzMyAxNS4wMzYsMC4xODkgQzE1LjE2NjY2NjcsMC4zMDU2NjY2NjcgMTUuMjMyLDAuNDQzMzMzMzMzIDE1LjIzMiwwLjYwMiBDMTUuMjMyLDAuNjg2IDE1LjIyMjY2NjcsMC43NDY2NjY2NjcgMTUuMjA0LDAuNzg0IEwxMy41MzgsNS40MzIgQzEzLjMyMzMzMzMsNi4wMTA2NjY2NyAxMy4wNTAzMzMzLDYuNDQgMTIuNzE5LDYuNzIgQzEyLjM4NzY2NjcsNyAxMS45NzkzMzMzLDcuMTQgMTEuNDk0LDcuMTQgWiIgaWQ9InYiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTkuODEsMTAuNTE0IEMxOS4xMTkzMzMzLDEwLjUxNCAxOC40OTE2NjY3LDEwLjQxMTMzMzMgMTcuOTI3LDEwLjIwNiBDMTcuMzYyMzMzMywxMC4wMDA2NjY3IDE2LjkwNzMzMzMsOS43MTEzMzMzMyAxNi41NjIsOS4zMzggQzE2LjQ2ODY2NjcsOS4yMzUzMzMzMyAxNi40MjIsOS4xMjMzMzMzMyAxNi40MjIsOS4wMDIgQzE2LjQyMiw4Ljg0MzMzMzMzIDE2LjUxMDY2NjcsOC42ODkzMzMzMyAxNi42ODgsOC41NCBDMTYuNzcyLDguNDY1MzMzMzMgMTYuODc5MzMzMyw4LjQyOCAxNy4wMSw4LjQyOCBDMTcuMjA2LDguNDI4IDE3LjM5MjY2NjcsOC41MjEzMzMzMyAxNy41Nyw4LjcwOCBDMTcuNzk0LDguOTQxMzMzMzMgMTguMDg1NjY2Nyw5LjEyNTY2NjY3IDE4LjQ0NSw5LjI2MSBDMTguODA0MzMzMyw5LjM5NjMzMzMzIDE5LjIyNjY2NjcsOS40NjQgMTkuNzEyLDkuNDY0IEMyMC41NjEzMzMzLDkuNDY0IDIxLjIxNDY2NjcsOS4yNDQ2NjY2NyAyMS42NzIsOC44MDYgQzIyLjEyOTMzMzMsOC4zNjczMzMzMyAyMi4zNTMzMzMzLDcuNzQyIDIyLjM0NCw2LjkzIEwyMi4zNDQsNS44NjYgQzIyLjAyNjY2NjcsNi4yNjczMzMzMyAyMS42MjUzMzMzLDYuNTg3IDIxLjE0LDYuODI1IEMyMC42NTQ2NjY3LDcuMDYzIDIwLjA4NTMzMzMsNy4xODIgMTkuNDMyLDcuMTgyIEMxOC4yNTYsNy4xODIgMTcuMzQ2LDYuODY3IDE2LjcwMiw2LjIzNyBDMTYuMDU4LDUuNjA3IDE1LjczNiw0Ljc1MDY2NjY3IDE1LjczNiwzLjY2OCBDMTUuNzM2LDIuNTU3MzMzMzMgMTYuMDYyNjY2NywxLjY2ODMzMzMzIDE2LjcxNiwxLjAwMSBDMTcuMzY5MzMzMywwLjMzMzY2NjY2NyAxOC4zMDI2NjY3LDAgMTkuNTE2LDAgQzIwLjA5NDY2NjcsMCAyMC42MjIsMC4xMTY2NjY2NjcgMjEuMDk4LDAuMzUgQzIxLjU3NCwwLjU4MzMzMzMzMyAyMS45ODkzMzMzLDAuOTE0NjY2NjY3IDIyLjM0NCwxLjM0NCBMMjIuMzQ0LDAuNjMgQzIyLjM0NCwwLjQ0MzMzMzMzMyAyMi40MDcsMC4yOTE2NjY2NjcgMjIuNTMzLDAuMTc1IEMyMi42NTksMC4wNTgzMzMzMzMzIDIyLjgxNTMzMzMsMCAyMy4wMDIsMCBDMjMuMTk4LDAgMjMuMzU0MzMzMywwLjA1NiAyMy40NzEsMC4xNjggQzIzLjU4NzY2NjcsMC4yOCAyMy42NDYsMC40MzQgMjMuNjQ2LDAuNjMgTDIzLjY0Niw2LjkzIEMyMy42NDYsNy42NDg2NjY2NyAyMy40ODczMzMzLDguMjc4NjY2NjcgMjMuMTcsOC44MiBDMjIuODUyNjY2Nyw5LjM2MTMzMzMzIDIyLjQwNDY2NjcsOS43NzkgMjEuODI2LDEwLjA3MyBDMjEuMjQ3MzMzMywxMC4zNjcgMjAuNTc1MzMzMywxMC41MTQgMTkuODEsMTAuNTE0IFogTTE5LjYxNCw2LjA5IEMyMC4xMDg2NjY3LDYuMDkgMjAuNTYzNjY2Nyw1Ljk5NjY2NjY3IDIwLjk3OSw1LjgxIEMyMS4zOTQzMzMzLDUuNjIzMzMzMzMgMjEuNzI1NjY2Nyw1LjM0MzMzMzMzIDIxLjk3Myw0Ljk3IEMyMi4yMjAzMzMzLDQuNTk2NjY2NjcgMjIuMzQ0LDQuMTM5MzMzMzMgMjIuMzQ0LDMuNTk4IEMyMi4zNDQsMi44NTEzMzMzMyAyMi4xMDYsMi4yNDQ2NjY2NyAyMS42MywxLjc3OCBDMjEuMTU0LDEuMzExMzMzMzMgMjAuNDk2LDEuMDc4IDE5LjY1NiwxLjA3OCBDMTguODI1MzMzMywxLjA3OCAxOC4xODEzMzMzLDEuMzA0MzMzMzMgMTcuNzI0LDEuNzU3IEMxNy4yNjY2NjY3LDIuMjA5NjY2NjcgMTcuMDM4LDIuODQ2NjY2NjcgMTcuMDM4LDMuNjY4IEMxNy4wMzgsNC40MTQ2NjY2NyAxNy4yNTk2NjY3LDUuMDA1IDE3LjcwMyw1LjQzOSBDMTguMTQ2MzMzMyw1Ljg3MyAxOC43ODMzMzMzLDYuMDkgMTkuNjE0LDYuMDkgWiIgaWQ9ImciIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPHBhdGggZD0iTTQ2LjY4MTgxODIsMTAgTDE3LjMxODE4MTgsMTAgQzE2LjA0MTM3MDYsMTAgMTUsMTEuMDQxMzcwNSAxNSwxMi4zMTgxODE4IEwxNSw0MS42ODE4MTgyIEMxNSw0Mi45NTg2Mjk1IDE2LjA0MTM3MDYsNDQgMTcuMzE4MTgxOCw0NCBMNDYuNjgxODE4Miw0NCBDNDcuOTU4NjI5NSw0NCA0OSw0Mi45NTg2Mjk1IDQ5LDQxLjY4MTgxODIgTDQ5LDEyLjMxODE4MTggQzQ5LDExLjA0MTM3MDUgNDcuOTU4NjI5NSwxMCA0Ni42ODE4MTgyLDEwIFogTTQyLjA0NTQ1NDYsMzAuODYzNjM2NCBMMjEuOTU0NTQ1NSwzMC44NjM2MzY0IEwyMS45NTQ1NDU1LDI5LjMxODE4MTggTDQyLjA0NTQ1NDYsMjkuMzE4MTgxOCBMNDIuMDQ1NDU0NiwzMC44NjM2MzY0IFogTTQyLjA0NTQ1NDYsMjQuNjgxODE4MiBMMjEuOTU0NTQ1NSwyNC42ODE4MTgyIEwyMS45NTQ1NDU1LDIzLjEzNjM2MzYgTDQyLjA0NTQ1NDYsMjMuMTM2MzYzNiBMNDIuMDQ1NDU0NiwyNC42ODE4MTgyIFoiIGlkPSJlcXVhbHMiIGZpbGw9IiMwMDAwMDAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_avg_eq()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">5.5</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfYXZnX2dlPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9hdmdfZ2UiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9ImF2ZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjAuMDU4MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yLjk0LDcuMTQgQzIuMDM0NjY2NjcsNy4xNCAxLjMxODMzMzMzLDYuOTM0NjY2NjcgMC43OTEsNi41MjQgQzAuMjYzNjY2NjY3LDYuMTEzMzMzMzMgMCw1LjU1OCAwLDQuODU4IEMwLDQuMTU4IDAuMjc3NjY2NjY3LDMuNjAyNjY2NjcgMC44MzMsMy4xOTIgQzEuMzg4MzMzMzMsMi43ODEzMzMzMyAyLjE0NjY2NjY3LDIuNTc2IDMuMTA4LDIuNTc2IEMzLjYxMiwyLjU3NiA0LjA2MjMzMzMzLDIuNjQxMzMzMzMgNC40NTksMi43NzIgQzQuODU1NjY2NjcsMi45MDI2NjY2NyA1LjE3NTMzMzMzLDMuMDggNS40MTgsMy4zMDQgTDUuNDE4LDIuNjA0IEM1LjQxOCwyLjEgNS4yNTcsMS43MTUgNC45MzUsMS40NDkgQzQuNjEzLDEuMTgzIDQuMTM5MzMzMzMsMS4wNSAzLjUxNCwxLjA1IEMyLjY4MzMzMzMzLDEuMDUgMS45NjQ2NjY2NywxLjI0MTMzMzMzIDEuMzU4LDEuNjI0IEMxLjE4MDY2NjY3LDEuNzI2NjY2NjcgMS4wMjY2NjY2NywxLjc3OCAwLjg5NiwxLjc3OCBDMC43NzQ2NjY2NjcsMS43NzggMC42NTMzMzMzMzMsMS43MjY2NjY2NyAwLjUzMiwxLjYyNCBDMC40MjkzMzMzMzMsMS41MjEzMzMzMyAwLjM3OCwxLjQgMC4zNzgsMS4yNiBDMC4zNzgsMS4xMDEzMzMzMyAwLjQ1NzMzMzMzMywwLjk1NjY2NjY2NyAwLjYxNiwwLjgyNiBDMC45NTIsMC41NzQgMS4zNzY2NjY2NywwLjM3MzMzMzMzMyAxLjg5LDAuMjI0IEMyLjQwMzMzMzMzLDAuMDc0NjY2NjY2NyAyLjk0LDAgMy41LDAgQzQuNTE3MzMzMzMsMCA1LjMxMDY2NjY3LDAuMjMxIDUuODgsMC42OTMgQzYuNDQ5MzMzMzMsMS4xNTUgNi43MzQsMS44MDEzMzMzMyA2LjczNCwyLjYzMiBMNi43MzQsNi41MSBDNi43MzQsNi43MTUzMzMzMyA2LjY4MDMzMzMzLDYuODcxNjY2NjcgNi41NzMsNi45NzkgQzYuNDY1NjY2NjcsNy4wODYzMzMzMyA2LjMwOTMzMzMzLDcuMTQgNi4xMDQsNy4xNCBDNS44OTg2NjY2Nyw3LjE0IDUuNzQsNy4wODYzMzMzMyA1LjYyOCw2Ljk3OSBDNS41MTYsNi44NzE2NjY2NyA1LjQ2LDYuNzI0NjY2NjcgNS40Niw2LjUzOCBMNS40Niw2LjE0NiBDNC45LDYuODA4NjY2NjcgNC4wNiw3LjE0IDIuOTQsNy4xNCBaIE0zLjE2NCw2LjE0NiBDMy44NDUzMzMzMyw2LjE0NiA0LjM5MzY2NjY3LDYuMDIyMzMzMzMgNC44MDksNS43NzUgQzUuMjI0MzMzMzMsNS41Mjc2NjY2NyA1LjQzMiw1LjIwMzMzMzMzIDUuNDMyLDQuODAyIEM1LjQzMiw0LjQxIDUuMjMzNjY2NjcsNC4xMDQzMzMzMyA0LjgzNywzLjg4NSBDNC40NDAzMzMzMywzLjY2NTY2NjY3IDMuODkyLDMuNTU2IDMuMTkyLDMuNTU2IEMyLjU4NTMzMzMzLDMuNTU2IDIuMTE4NjY2NjcsMy42NjggMS43OTIsMy44OTIgQzEuNDY1MzMzMzMsNC4xMTYgMS4zMDIsNC40MzggMS4zMDIsNC44NTggQzEuMzAyLDUuMjY4NjY2NjcgMS40NjMsNS41ODYgMS43ODUsNS44MSBDMi4xMDcsNi4wMzQgMi41NjY2NjY2Nyw2LjE0NiAzLjE2NCw2LjE0NiBaIiBpZD0iYSIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS40OTQsNy4xNCBDMTAuNTMyNjY2Nyw3LjE0IDkuODUxMzMzMzMsNi41NjEzMzMzMyA5LjQ1LDUuNDA0IEw3Ljc5OCwwLjc4NCBDNy43NywwLjcwOTMzMzMzMyA3Ljc1NiwwLjY0NCA3Ljc1NiwwLjU4OCBDNy43NTYsMC40NjY2NjY2NjcgNy43OTgsMC4zNTcgNy44ODIsMC4yNTkgQzcuOTY2LDAuMTYxIDguMDc4LDAuMDg4NjY2NjY2NyA4LjIxOCwwLjA0MiBDOC4zMDIsMC4wMTQgOC4zODYsMCA4LjQ3LDAgQzguNjEsMCA4LjczMTMzMzMzLDAuMDM1IDguODM0LDAuMTA1IEM4LjkzNjY2NjY3LDAuMTc1IDkuMDA2NjY2NjcsMC4yNzA2NjY2NjcgOS4wNDQsMC4zOTIgTDEwLjczOCw1LjEzOCBDMTAuODU5MzMzMyw1LjQ4MzMzMzMzIDEwLjk3Niw1LjcyNiAxMS4wODgsNS44NjYgQzExLjIsNi4wMDYgMTEuMzM1MzMzMyw2LjA3NiAxMS40OTQsNi4wNzYgQzExLjY1MjY2NjcsNi4wNzYgMTEuNzg4LDYuMDA4MzMzMzMgMTEuOSw1Ljg3MyBDMTIuMDEyLDUuNzM3NjY2NjcgMTIuMTI4NjY2Nyw1LjUwMiAxMi4yNSw1LjE2NiBMMTMuOTQ0LDAuMzkyIEMxMy45OTA2NjY3LDAuMjcwNjY2NjY3IDE0LjA2MywwLjE3NzMzMzMzMyAxNC4xNjEsMC4xMTIgQzE0LjI1OSwwLjA0NjY2NjY2NjcgMTQuMzc4LDAuMDE0IDE0LjUxOCwwLjAxNCBDMTQuNzMyNjY2NywwLjAxNCAxNC45MDUzMzMzLDAuMDcyMzMzMzMzMyAxNS4wMzYsMC4xODkgQzE1LjE2NjY2NjcsMC4zMDU2NjY2NjcgMTUuMjMyLDAuNDQzMzMzMzMzIDE1LjIzMiwwLjYwMiBDMTUuMjMyLDAuNjg2IDE1LjIyMjY2NjcsMC43NDY2NjY2NjcgMTUuMjA0LDAuNzg0IEwxMy41MzgsNS40MzIgQzEzLjMyMzMzMzMsNi4wMTA2NjY2NyAxMy4wNTAzMzMzLDYuNDQgMTIuNzE5LDYuNzIgQzEyLjM4NzY2NjcsNyAxMS45NzkzMzMzLDcuMTQgMTEuNDk0LDcuMTQgWiIgaWQ9InYiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTkuODEsMTAuNTE0IEMxOS4xMTkzMzMzLDEwLjUxNCAxOC40OTE2NjY3LDEwLjQxMTMzMzMgMTcuOTI3LDEwLjIwNiBDMTcuMzYyMzMzMywxMC4wMDA2NjY3IDE2LjkwNzMzMzMsOS43MTEzMzMzMyAxNi41NjIsOS4zMzggQzE2LjQ2ODY2NjcsOS4yMzUzMzMzMyAxNi40MjIsOS4xMjMzMzMzMyAxNi40MjIsOS4wMDIgQzE2LjQyMiw4Ljg0MzMzMzMzIDE2LjUxMDY2NjcsOC42ODkzMzMzMyAxNi42ODgsOC41NCBDMTYuNzcyLDguNDY1MzMzMzMgMTYuODc5MzMzMyw4LjQyOCAxNy4wMSw4LjQyOCBDMTcuMjA2LDguNDI4IDE3LjM5MjY2NjcsOC41MjEzMzMzMyAxNy41Nyw4LjcwOCBDMTcuNzk0LDguOTQxMzMzMzMgMTguMDg1NjY2Nyw5LjEyNTY2NjY3IDE4LjQ0NSw5LjI2MSBDMTguODA0MzMzMyw5LjM5NjMzMzMzIDE5LjIyNjY2NjcsOS40NjQgMTkuNzEyLDkuNDY0IEMyMC41NjEzMzMzLDkuNDY0IDIxLjIxNDY2NjcsOS4yNDQ2NjY2NyAyMS42NzIsOC44MDYgQzIyLjEyOTMzMzMsOC4zNjczMzMzMyAyMi4zNTMzMzMzLDcuNzQyIDIyLjM0NCw2LjkzIEwyMi4zNDQsNS44NjYgQzIyLjAyNjY2NjcsNi4yNjczMzMzMyAyMS42MjUzMzMzLDYuNTg3IDIxLjE0LDYuODI1IEMyMC42NTQ2NjY3LDcuMDYzIDIwLjA4NTMzMzMsNy4xODIgMTkuNDMyLDcuMTgyIEMxOC4yNTYsNy4xODIgMTcuMzQ2LDYuODY3IDE2LjcwMiw2LjIzNyBDMTYuMDU4LDUuNjA3IDE1LjczNiw0Ljc1MDY2NjY3IDE1LjczNiwzLjY2OCBDMTUuNzM2LDIuNTU3MzMzMzMgMTYuMDYyNjY2NywxLjY2ODMzMzMzIDE2LjcxNiwxLjAwMSBDMTcuMzY5MzMzMywwLjMzMzY2NjY2NyAxOC4zMDI2NjY3LDAgMTkuNTE2LDAgQzIwLjA5NDY2NjcsMCAyMC42MjIsMC4xMTY2NjY2NjcgMjEuMDk4LDAuMzUgQzIxLjU3NCwwLjU4MzMzMzMzMyAyMS45ODkzMzMzLDAuOTE0NjY2NjY3IDIyLjM0NCwxLjM0NCBMMjIuMzQ0LDAuNjMgQzIyLjM0NCwwLjQ0MzMzMzMzMyAyMi40MDcsMC4yOTE2NjY2NjcgMjIuNTMzLDAuMTc1IEMyMi42NTksMC4wNTgzMzMzMzMzIDIyLjgxNTMzMzMsMCAyMy4wMDIsMCBDMjMuMTk4LDAgMjMuMzU0MzMzMywwLjA1NiAyMy40NzEsMC4xNjggQzIzLjU4NzY2NjcsMC4yOCAyMy42NDYsMC40MzQgMjMuNjQ2LDAuNjMgTDIzLjY0Niw2LjkzIEMyMy42NDYsNy42NDg2NjY2NyAyMy40ODczMzMzLDguMjc4NjY2NjcgMjMuMTcsOC44MiBDMjIuODUyNjY2Nyw5LjM2MTMzMzMzIDIyLjQwNDY2NjcsOS43NzkgMjEuODI2LDEwLjA3MyBDMjEuMjQ3MzMzMywxMC4zNjcgMjAuNTc1MzMzMywxMC41MTQgMTkuODEsMTAuNTE0IFogTTE5LjYxNCw2LjA5IEMyMC4xMDg2NjY3LDYuMDkgMjAuNTYzNjY2Nyw1Ljk5NjY2NjY3IDIwLjk3OSw1LjgxIEMyMS4zOTQzMzMzLDUuNjIzMzMzMzMgMjEuNzI1NjY2Nyw1LjM0MzMzMzMzIDIxLjk3Myw0Ljk3IEMyMi4yMjAzMzMzLDQuNTk2NjY2NjcgMjIuMzQ0LDQuMTM5MzMzMzMgMjIuMzQ0LDMuNTk4IEMyMi4zNDQsMi44NTEzMzMzMyAyMi4xMDYsMi4yNDQ2NjY2NyAyMS42MywxLjc3OCBDMjEuMTU0LDEuMzExMzMzMzMgMjAuNDk2LDEuMDc4IDE5LjY1NiwxLjA3OCBDMTguODI1MzMzMywxLjA3OCAxOC4xODEzMzMzLDEuMzA0MzMzMzMgMTcuNzI0LDEuNzU3IEMxNy4yNjY2NjY3LDIuMjA5NjY2NjcgMTcuMDM4LDIuODQ2NjY2NjcgMTcuMDM4LDMuNjY4IEMxNy4wMzgsNC40MTQ2NjY2NyAxNy4yNTk2NjY3LDUuMDA1IDE3LjcwMyw1LjQzOSBDMTguMTQ2MzMzMyw1Ljg3MyAxOC43ODMzMzMzLDYuMDkgMTkuNjE0LDYuMDkgWiIgaWQ9ImciIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPHBhdGggZD0iTTQ0Ljk1MjM4MSwxMCBMMTkuMDQ3NjE5MSwxMCBDMTYuODEzMzMzMywxMCAxNSwxMS44MTMzMzMzIDE1LDE0LjA0NzYxOSBMMTUsMzkuOTUyMzgxIEMxNSw0Mi4xODY2NjY3IDE2LjgxMzMzMzMsNDQgMTkuMDQ3NjE5MSw0NCBMNDQuOTUyMzgxLDQ0IEM0Ny4xODY2NjY3LDQ0IDQ5LDQyLjE4NjY2NjcgNDksMzkuOTUyMzgxIEw0OSwxNC4wNDc2MTkgQzQ5LDExLjgxMzMzMzMgNDcuMTg2NjY2NywxMCA0NC45NTIzODEsMTAgWiBNNDAuOTA0NzYxOSwzOC4zMzMzMzMzIEwyMy4wOTUyMzgxLDM4LjMzMzMzMzMgTDIzLjA5NTIzODEsMzYuNzE0Mjg1NyBMNDAuOTA0NzYxOSwzNi43MTQyODU3IEw0MC45MDQ3NjE5LDM4LjMzMzMzMzMgWiBNMjQuMjg1MjM4MSwzMy4zNzkwNDc2IEwyMy41MjQyODU3LDMxLjk1NDI4NTcgTDM3LjU0NTIzODEsMjQuNTcxNDI4NiBMMjMuNTI0Mjg1NywxNy4xODg1NzE0IEwyNC4yODUyMzgxLDE1Ljc2MzgwOTUgTDQxLjAyNjE5MDUsMjQuNTcxNDI4NiBMMjQuMjg1MjM4MSwzMy4zNzkwNDc2IFoiIGlkPSJncmVhdGVyX3RoYW5fZXF1YWwiIGZpbGw9IiMwMDAwMDAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_avg_ge()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">50</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


## Validating Standard Deviations

Standard deviation validations help ensure data variability is within expected ranges:


``` python
(
    pb.Validate(data=agg_data)
    .col_sd_gt(columns="a", value=2)
    .col_sd_lt(columns="b", value=35)
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
TgsNi40MzA2NjY2NyAxMy43MDEzMzMzLDUuODI0IDEzLjE4OCw1LjM3NiBDMTIuNjc0NjY2Nyw0LjkyOCAxMi4wMjYsNC43MDQgMTEuMjQyLDQuNzA0IEMxMC40Myw0LjcwNCA5Ljc5NTMzMzMzLDQuOTMyNjY2NjcgOS4zMzgsNS4zOSBDOC44ODA2NjY2Nyw1Ljg0NzMzMzMzIDguNjUyLDYuNDU4NjY2NjcgOC42NTIsNy4yMjQgQzguNjUyLDguMDI2NjY2NjcgOC44NzgzMzMzMyw4LjYzOCA5LjMzMSw5LjA1OCBDOS43ODM2NjY2Nyw5LjQ3OCAxMC40MTYsOS42ODggMTEuMjI4LDkuNjg4IFoiIGlkPSJkIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00NC45NTIzODEsMTAgTDE5LjA0NzYxOTEsMTAgQzE2LjgxMzMzMzMsMTAgMTUsMTEuODEzMzMzMyAxNSwxNC4wNDc2MTkgTDE1LDM5Ljk1MjM4MSBDMTUsNDIuMTg2NjY2NyAxNi44MTMzMzMzLDQ0IDE5LjA0NzYxOTEsNDQgTDQ0Ljk1MjM4MSw0NCBDNDcuMTg2NjY2Nyw0NCA0OSw0Mi4xODY2NjY3IDQ5LDM5Ljk1MjM4MSBMNDksMTQuMDQ3NjE5IEM0OSwxMS44MTMzMzMzIDQ3LjE4NjY2NjcsMTAgNDQuOTUyMzgxLDEwIFogTTI2Ljc5NDc2MTksMzYuNTY4NTcxNCBMMjUuODcxOTA0OCwzNS4yNDA5NTI0IEwzNy44NjA5NTI0LDI3IEwyNS44NzE5MDQ4LDE4Ljc1OTA0NzYgTDI2Ljc5NDc2MTksMTcuNDMxNDI4NiBMNDAuNzE4NTcxNCwyNyBMMjYuNzk0NzYxOSwzNi41Njg1NzE0IFoiIGlkPSJncmVhdGVyX3RoYW4iIGZpbGw9IiMwMDAwMDAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_sd_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">2</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
MzEzMzMzMyAxNS4xOSwxMC40NzQzMzMzIDE1LjA3OCwxMC41OTEgQzE0Ljk2NiwxMC43MDc2NjY3IDE0LjgxMiwxMC43NjYgMTQuNjE2LDEwLjc2NiBDMTQuNDEwNjY2NywxMC43NjYgMTQuMjQ5NjY2NywxMC43MDc2NjY3IDE0LjEzMywxMC41OTEgQzE0LjAxNjMzMzMsMTAuNDc0MzMzMyAxMy45NTgsMTAuMzEzMzMzMyAxMy45NTgsMTAuMTA4IEwxMy45NTgsOS40MjIgQzEzLjMwNDY2NjcsMTAuMzE4IDEyLjM0OCwxMC43NjYgMTEuMDg4LDEwLjc2NiBaIE0xMS4yMjgsOS42ODggQzEyLjA3NzMzMzMsOS42ODggMTIuNzQ0NjY2Nyw5LjQ2NjMzMzMzIDEzLjIzLDkuMDIzIEMxMy43MTUzMzMzLDguNTc5NjY2NjcgMTMuOTU4LDcuOTcwNjY2NjcgMTMuOTU4LDcuMTk2IEMxMy45NTgsNi40MzA2NjY2NyAxMy43MDEzMzMzLDUuODI0IDEzLjE4OCw1LjM3NiBDMTIuNjc0NjY2Nyw0LjkyOCAxMi4wMjYsNC43MDQgMTEuMjQyLDQuNzA0IEMxMC40Myw0LjcwNCA5Ljc5NTMzMzMzLDQuOTMyNjY2NjcgOS4zMzgsNS4zOSBDOC44ODA2NjY2Nyw1Ljg0NzMzMzMzIDguNjUyLDYuNDU4NjY2NjcgOC42NTIsNy4yMjQgQzguNjUyLDguMDI2NjY2NjcgOC44NzgzMzMzMyw4LjYzOCA5LjMzMSw5LjA1OCBDOS43ODM2NjY2Nyw5LjQ3OCAxMC40MTYsOS42ODggMTEuMjI4LDkuNjg4IFoiIGlkPSJkIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00Ni42ODE4MTgyLDEwIEwxNy4zMTgxODE4LDEwIEMxNi4wNDEzNzA2LDEwIDE1LDExLjA0MTM3MDUgMTUsMTIuMzE4MTgxOCBMMTUsNDEuNjgxODE4MiBDMTUsNDIuOTU4NjI5NSAxNi4wNDEzNzA2LDQ0IDE3LjMxODE4MTgsNDQgTDQ2LjY4MTgxODIsNDQgQzQ3Ljk1ODYyOTUsNDQgNDksNDIuOTU4NjI5NSA0OSw0MS42ODE4MTgyIEw0OSwxMi4zMTgxODE4IEM0OSwxMS4wNDEzNzA1IDQ3Ljk1ODYyOTUsMTAgNDYuNjgxODE4MiwxMCBaIE0zNy4xODI3MDU4LDM3LjI3MTgzOTcgTDM2LjA5MDAyMTUsMzguMzY0NTI0IEwyNC43MjU0OTc2LDI3IEwzNi4wOTAwMjE1LDE1LjYzNTQ3NiBMMzcuMTgyNzA1OCwxNi43MjgxNjAzIEwyNi45MTA4NjY5LDI3IEwzNy4xODI3MDU4LDM3LjI3MTgzOTcgWiIgaWQ9Imxlc3NfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_sd_lt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">35</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


## Using Tolerance for Fuzzy Comparisons

Floating-point arithmetic can introduce small precision errors, making exact equality comparisons unreliable. The `tol=` parameter allows for fuzzy comparisons by specifying an acceptable tolerance:


``` python
(
    pb.Validate(data=agg_data)
    .col_avg_eq(columns="a", value=5.5, tol=0.01)  # Pass if average is within ±0.01 of 5.5
    .col_sum_eq(columns="b", value=550, tol=1)    # Pass if sum is within ±1 of 550
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfYXZnX2VxPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9hdmdfZXEiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9ImF2ZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjAuMDU4MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yLjk0LDcuMTQgQzIuMDM0NjY2NjcsNy4xNCAxLjMxODMzMzMzLDYuOTM0NjY2NjcgMC43OTEsNi41MjQgQzAuMjYzNjY2NjY3LDYuMTEzMzMzMzMgMCw1LjU1OCAwLDQuODU4IEMwLDQuMTU4IDAuMjc3NjY2NjY3LDMuNjAyNjY2NjcgMC44MzMsMy4xOTIgQzEuMzg4MzMzMzMsMi43ODEzMzMzMyAyLjE0NjY2NjY3LDIuNTc2IDMuMTA4LDIuNTc2IEMzLjYxMiwyLjU3NiA0LjA2MjMzMzMzLDIuNjQxMzMzMzMgNC40NTksMi43NzIgQzQuODU1NjY2NjcsMi45MDI2NjY2NyA1LjE3NTMzMzMzLDMuMDggNS40MTgsMy4zMDQgTDUuNDE4LDIuNjA0IEM1LjQxOCwyLjEgNS4yNTcsMS43MTUgNC45MzUsMS40NDkgQzQuNjEzLDEuMTgzIDQuMTM5MzMzMzMsMS4wNSAzLjUxNCwxLjA1IEMyLjY4MzMzMzMzLDEuMDUgMS45NjQ2NjY2NywxLjI0MTMzMzMzIDEuMzU4LDEuNjI0IEMxLjE4MDY2NjY3LDEuNzI2NjY2NjcgMS4wMjY2NjY2NywxLjc3OCAwLjg5NiwxLjc3OCBDMC43NzQ2NjY2NjcsMS43NzggMC42NTMzMzMzMzMsMS43MjY2NjY2NyAwLjUzMiwxLjYyNCBDMC40MjkzMzMzMzMsMS41MjEzMzMzMyAwLjM3OCwxLjQgMC4zNzgsMS4yNiBDMC4zNzgsMS4xMDEzMzMzMyAwLjQ1NzMzMzMzMywwLjk1NjY2NjY2NyAwLjYxNiwwLjgyNiBDMC45NTIsMC41NzQgMS4zNzY2NjY2NywwLjM3MzMzMzMzMyAxLjg5LDAuMjI0IEMyLjQwMzMzMzMzLDAuMDc0NjY2NjY2NyAyLjk0LDAgMy41LDAgQzQuNTE3MzMzMzMsMCA1LjMxMDY2NjY3LDAuMjMxIDUuODgsMC42OTMgQzYuNDQ5MzMzMzMsMS4xNTUgNi43MzQsMS44MDEzMzMzMyA2LjczNCwyLjYzMiBMNi43MzQsNi41MSBDNi43MzQsNi43MTUzMzMzMyA2LjY4MDMzMzMzLDYuODcxNjY2NjcgNi41NzMsNi45NzkgQzYuNDY1NjY2NjcsNy4wODYzMzMzMyA2LjMwOTMzMzMzLDcuMTQgNi4xMDQsNy4xNCBDNS44OTg2NjY2Nyw3LjE0IDUuNzQsNy4wODYzMzMzMyA1LjYyOCw2Ljk3OSBDNS41MTYsNi44NzE2NjY2NyA1LjQ2LDYuNzI0NjY2NjcgNS40Niw2LjUzOCBMNS40Niw2LjE0NiBDNC45LDYuODA4NjY2NjcgNC4wNiw3LjE0IDIuOTQsNy4xNCBaIE0zLjE2NCw2LjE0NiBDMy44NDUzMzMzMyw2LjE0NiA0LjM5MzY2NjY3LDYuMDIyMzMzMzMgNC44MDksNS43NzUgQzUuMjI0MzMzMzMsNS41Mjc2NjY2NyA1LjQzMiw1LjIwMzMzMzMzIDUuNDMyLDQuODAyIEM1LjQzMiw0LjQxIDUuMjMzNjY2NjcsNC4xMDQzMzMzMyA0LjgzNywzLjg4NSBDNC40NDAzMzMzMywzLjY2NTY2NjY3IDMuODkyLDMuNTU2IDMuMTkyLDMuNTU2IEMyLjU4NTMzMzMzLDMuNTU2IDIuMTE4NjY2NjcsMy42NjggMS43OTIsMy44OTIgQzEuNDY1MzMzMzMsNC4xMTYgMS4zMDIsNC40MzggMS4zMDIsNC44NTggQzEuMzAyLDUuMjY4NjY2NjcgMS40NjMsNS41ODYgMS43ODUsNS44MSBDMi4xMDcsNi4wMzQgMi41NjY2NjY2Nyw2LjE0NiAzLjE2NCw2LjE0NiBaIiBpZD0iYSIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS40OTQsNy4xNCBDMTAuNTMyNjY2Nyw3LjE0IDkuODUxMzMzMzMsNi41NjEzMzMzMyA5LjQ1LDUuNDA0IEw3Ljc5OCwwLjc4NCBDNy43NywwLjcwOTMzMzMzMyA3Ljc1NiwwLjY0NCA3Ljc1NiwwLjU4OCBDNy43NTYsMC40NjY2NjY2NjcgNy43OTgsMC4zNTcgNy44ODIsMC4yNTkgQzcuOTY2LDAuMTYxIDguMDc4LDAuMDg4NjY2NjY2NyA4LjIxOCwwLjA0MiBDOC4zMDIsMC4wMTQgOC4zODYsMCA4LjQ3LDAgQzguNjEsMCA4LjczMTMzMzMzLDAuMDM1IDguODM0LDAuMTA1IEM4LjkzNjY2NjY3LDAuMTc1IDkuMDA2NjY2NjcsMC4yNzA2NjY2NjcgOS4wNDQsMC4zOTIgTDEwLjczOCw1LjEzOCBDMTAuODU5MzMzMyw1LjQ4MzMzMzMzIDEwLjk3Niw1LjcyNiAxMS4wODgsNS44NjYgQzExLjIsNi4wMDYgMTEuMzM1MzMzMyw2LjA3NiAxMS40OTQsNi4wNzYgQzExLjY1MjY2NjcsNi4wNzYgMTEuNzg4LDYuMDA4MzMzMzMgMTEuOSw1Ljg3MyBDMTIuMDEyLDUuNzM3NjY2NjcgMTIuMTI4NjY2Nyw1LjUwMiAxMi4yNSw1LjE2NiBMMTMuOTQ0LDAuMzkyIEMxMy45OTA2NjY3LDAuMjcwNjY2NjY3IDE0LjA2MywwLjE3NzMzMzMzMyAxNC4xNjEsMC4xMTIgQzE0LjI1OSwwLjA0NjY2NjY2NjcgMTQuMzc4LDAuMDE0IDE0LjUxOCwwLjAxNCBDMTQuNzMyNjY2NywwLjAxNCAxNC45MDUzMzMzLDAuMDcyMzMzMzMzMyAxNS4wMzYsMC4xODkgQzE1LjE2NjY2NjcsMC4zMDU2NjY2NjcgMTUuMjMyLDAuNDQzMzMzMzMzIDE1LjIzMiwwLjYwMiBDMTUuMjMyLDAuNjg2IDE1LjIyMjY2NjcsMC43NDY2NjY2NjcgMTUuMjA0LDAuNzg0IEwxMy41MzgsNS40MzIgQzEzLjMyMzMzMzMsNi4wMTA2NjY2NyAxMy4wNTAzMzMzLDYuNDQgMTIuNzE5LDYuNzIgQzEyLjM4NzY2NjcsNyAxMS45NzkzMzMzLDcuMTQgMTEuNDk0LDcuMTQgWiIgaWQ9InYiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTkuODEsMTAuNTE0IEMxOS4xMTkzMzMzLDEwLjUxNCAxOC40OTE2NjY3LDEwLjQxMTMzMzMgMTcuOTI3LDEwLjIwNiBDMTcuMzYyMzMzMywxMC4wMDA2NjY3IDE2LjkwNzMzMzMsOS43MTEzMzMzMyAxNi41NjIsOS4zMzggQzE2LjQ2ODY2NjcsOS4yMzUzMzMzMyAxNi40MjIsOS4xMjMzMzMzMyAxNi40MjIsOS4wMDIgQzE2LjQyMiw4Ljg0MzMzMzMzIDE2LjUxMDY2NjcsOC42ODkzMzMzMyAxNi42ODgsOC41NCBDMTYuNzcyLDguNDY1MzMzMzMgMTYuODc5MzMzMyw4LjQyOCAxNy4wMSw4LjQyOCBDMTcuMjA2LDguNDI4IDE3LjM5MjY2NjcsOC41MjEzMzMzMyAxNy41Nyw4LjcwOCBDMTcuNzk0LDguOTQxMzMzMzMgMTguMDg1NjY2Nyw5LjEyNTY2NjY3IDE4LjQ0NSw5LjI2MSBDMTguODA0MzMzMyw5LjM5NjMzMzMzIDE5LjIyNjY2NjcsOS40NjQgMTkuNzEyLDkuNDY0IEMyMC41NjEzMzMzLDkuNDY0IDIxLjIxNDY2NjcsOS4yNDQ2NjY2NyAyMS42NzIsOC44MDYgQzIyLjEyOTMzMzMsOC4zNjczMzMzMyAyMi4zNTMzMzMzLDcuNzQyIDIyLjM0NCw2LjkzIEwyMi4zNDQsNS44NjYgQzIyLjAyNjY2NjcsNi4yNjczMzMzMyAyMS42MjUzMzMzLDYuNTg3IDIxLjE0LDYuODI1IEMyMC42NTQ2NjY3LDcuMDYzIDIwLjA4NTMzMzMsNy4xODIgMTkuNDMyLDcuMTgyIEMxOC4yNTYsNy4xODIgMTcuMzQ2LDYuODY3IDE2LjcwMiw2LjIzNyBDMTYuMDU4LDUuNjA3IDE1LjczNiw0Ljc1MDY2NjY3IDE1LjczNiwzLjY2OCBDMTUuNzM2LDIuNTU3MzMzMzMgMTYuMDYyNjY2NywxLjY2ODMzMzMzIDE2LjcxNiwxLjAwMSBDMTcuMzY5MzMzMywwLjMzMzY2NjY2NyAxOC4zMDI2NjY3LDAgMTkuNTE2LDAgQzIwLjA5NDY2NjcsMCAyMC42MjIsMC4xMTY2NjY2NjcgMjEuMDk4LDAuMzUgQzIxLjU3NCwwLjU4MzMzMzMzMyAyMS45ODkzMzMzLDAuOTE0NjY2NjY3IDIyLjM0NCwxLjM0NCBMMjIuMzQ0LDAuNjMgQzIyLjM0NCwwLjQ0MzMzMzMzMyAyMi40MDcsMC4yOTE2NjY2NjcgMjIuNTMzLDAuMTc1IEMyMi42NTksMC4wNTgzMzMzMzMzIDIyLjgxNTMzMzMsMCAyMy4wMDIsMCBDMjMuMTk4LDAgMjMuMzU0MzMzMywwLjA1NiAyMy40NzEsMC4xNjggQzIzLjU4NzY2NjcsMC4yOCAyMy42NDYsMC40MzQgMjMuNjQ2LDAuNjMgTDIzLjY0Niw2LjkzIEMyMy42NDYsNy42NDg2NjY2NyAyMy40ODczMzMzLDguMjc4NjY2NjcgMjMuMTcsOC44MiBDMjIuODUyNjY2Nyw5LjM2MTMzMzMzIDIyLjQwNDY2NjcsOS43NzkgMjEuODI2LDEwLjA3MyBDMjEuMjQ3MzMzMywxMC4zNjcgMjAuNTc1MzMzMywxMC41MTQgMTkuODEsMTAuNTE0IFogTTE5LjYxNCw2LjA5IEMyMC4xMDg2NjY3LDYuMDkgMjAuNTYzNjY2Nyw1Ljk5NjY2NjY3IDIwLjk3OSw1LjgxIEMyMS4zOTQzMzMzLDUuNjIzMzMzMzMgMjEuNzI1NjY2Nyw1LjM0MzMzMzMzIDIxLjk3Myw0Ljk3IEMyMi4yMjAzMzMzLDQuNTk2NjY2NjcgMjIuMzQ0LDQuMTM5MzMzMzMgMjIuMzQ0LDMuNTk4IEMyMi4zNDQsMi44NTEzMzMzMyAyMi4xMDYsMi4yNDQ2NjY2NyAyMS42MywxLjc3OCBDMjEuMTU0LDEuMzExMzMzMzMgMjAuNDk2LDEuMDc4IDE5LjY1NiwxLjA3OCBDMTguODI1MzMzMywxLjA3OCAxOC4xODEzMzMzLDEuMzA0MzMzMzMgMTcuNzI0LDEuNzU3IEMxNy4yNjY2NjY3LDIuMjA5NjY2NjcgMTcuMDM4LDIuODQ2NjY2NjcgMTcuMDM4LDMuNjY4IEMxNy4wMzgsNC40MTQ2NjY2NyAxNy4yNTk2NjY3LDUuMDA1IDE3LjcwMyw1LjQzOSBDMTguMTQ2MzMzMyw1Ljg3MyAxOC43ODMzMzMzLDYuMDkgMTkuNjE0LDYuMDkgWiIgaWQ9ImciIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPHBhdGggZD0iTTQ2LjY4MTgxODIsMTAgTDE3LjMxODE4MTgsMTAgQzE2LjA0MTM3MDYsMTAgMTUsMTEuMDQxMzcwNSAxNSwxMi4zMTgxODE4IEwxNSw0MS42ODE4MTgyIEMxNSw0Mi45NTg2Mjk1IDE2LjA0MTM3MDYsNDQgMTcuMzE4MTgxOCw0NCBMNDYuNjgxODE4Miw0NCBDNDcuOTU4NjI5NSw0NCA0OSw0Mi45NTg2Mjk1IDQ5LDQxLjY4MTgxODIgTDQ5LDEyLjMxODE4MTggQzQ5LDExLjA0MTM3MDUgNDcuOTU4NjI5NSwxMCA0Ni42ODE4MTgyLDEwIFogTTQyLjA0NTQ1NDYsMzAuODYzNjM2NCBMMjEuOTU0NTQ1NSwzMC44NjM2MzY0IEwyMS45NTQ1NDU1LDI5LjMxODE4MTggTDQyLjA0NTQ1NDYsMjkuMzE4MTgxOCBMNDIuMDQ1NDU0NiwzMC44NjM2MzY0IFogTTQyLjA0NTQ1NDYsMjQuNjgxODE4MiBMMjEuOTU0NTQ1NSwyNC42ODE4MTgyIEwyMS45NTQ1NDU1LDIzLjEzNjM2MzYgTDQyLjA0NTQ1NDYsMjMuMTM2MzYzNiBMNDIuMDQ1NDU0NiwyNC42ODE4MTgyIFoiIGlkPSJlcXVhbHMiIGZpbGw9IiMwMDAwMDAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_avg_eq()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">5.5<br />
tol=0.01</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc3VtX2VxPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9zdW1fZXEiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InN1bSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTguODg5MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCw3LjE0IEMyLjY2OTMzMzMzLDcuMTQgMi4wMyw3LjAxNjMzMzMzIDEuNDcsNi43NjkgQzAuOTEsNi41MjE2NjY2NyAwLjQ2NjY2NjY2Nyw2LjE4OCAwLjE0LDUuNzY4IEMwLjA0NjY2NjY2NjcsNS42NDY2NjY2NyAwLDUuNTI1MzMzMzMgMCw1LjQwNCBDMCw1LjE5ODY2NjY3IDAuMDk4LDUuMDQ0NjY2NjcgMC4yOTQsNC45NDIgQzAuMzc4LDQuODk1MzMzMzMgMC40NjY2NjY2NjcsNC44NzIgMC41Niw0Ljg3MiBDMC43MTg2NjY2NjcsNC44NzIgMC44ODIsNC45NTYgMS4wNSw1LjEyNCBDMS4zNTgsNS40NTA2NjY2NyAxLjY5ODY2NjY3LDUuNjg4NjY2NjcgMi4wNzIsNS44MzggQzIuNDQ1MzMzMzMsNS45ODczMzMzMyAyLjg3NDY2NjY3LDYuMDYyIDMuMzYsNi4wNjIgQzMuODA4LDYuMDYyIDQuMTc0MzMzMzMsNS45ODk2NjY2NyA0LjQ1OSw1Ljg0NSBDNC43NDM2NjY2Nyw1LjcwMDMzMzMzIDQuODg2LDUuNTAyIDQuODg2LDUuMjUgQzQuODg2LDUuMDYzMzMzMzMgNC44MzIzMzMzMyw0LjkwOTMzMzMzIDQuNzI1LDQuNzg4IEM0LjYxNzY2NjY3LDQuNjY2NjY2NjcgNC40MzMzMzMzMyw0LjU1IDQuMTcyLDQuNDM4IEMzLjkxMDY2NjY3LDQuMzI2IDMuNSw0LjE3NjY2NjY3IDIuOTQsMy45OSBDMS45OTczMzMzMywzLjY5MTMzMzMzIDEuMzMyMzMzMzMsMy4zODMzMzMzMyAwLjk0NSwzLjA2NiBDMC41NTc2NjY2NjcsMi43NDg2NjY2NyAwLjM2NCwyLjMzMzMzMzMzIDAuMzY0LDEuODIgQzAuMzY0LDEuMjY5MzMzMzMgMC42MDQzMzMzMzMsMC44MjgzMzMzMzMgMS4wODUsMC40OTcgQzEuNTY1NjY2NjcsMC4xNjU2NjY2NjcgMi4yMTY2NjY2NywwIDMuMDM4LDAgQzMuNjQ0NjY2NjcsMCA0LjIwNywwLjEwMDMzMzMzMyA0LjcyNSwwLjMwMSBDNS4yNDMsMC41MDE2NjY2NjcgNS42MjgsMC43NzQ2NjY2NjcgNS44OCwxLjEyIEM1Ljk2NCwxLjIzMiA2LjAwNiwxLjM0ODY2NjY3IDYuMDA2LDEuNDcgQzYuMDA2LDEuNjI4NjY2NjcgNS45MjIsMS43NjQgNS43NTQsMS44NzYgQzUuNjMyNjY2NjcsMS45NTA2NjY2NyA1LjUxMTMzMzMzLDEuOTg4IDUuMzksMS45ODggQzUuMjAzMzMzMzMsMS45ODggNS4wMjYsMS45MDQgNC44NTgsMS43MzYgQzQuNjM0LDEuNTEyIDQuMzc1LDEuMzQ2MzMzMzMgNC4wODEsMS4yMzkgQzMuNzg3LDEuMTMxNjY2NjcgMy40MzQ2NjY2NywxLjA3OCAzLjAyNCwxLjA3OCBDMi41ODUzMzMzMywxLjA3OCAyLjI1MTY2NjY3LDEuMTQzMzMzMzMgMi4wMjMsMS4yNzQgQzEuNzk0MzMzMzMsMS40MDQ2NjY2NyAxLjY4LDEuNTkxMzMzMzMgMS42OCwxLjgzNCBDMS42OCwyLjAyMDY2NjY3IDEuNzMxMzMzMzMsMi4xNyAxLjgzNCwyLjI4MiBDMS45MzY2NjY2NywyLjM5NCAyLjExNCwyLjUwMTMzMzMzIDIuMzY2LDIuNjA0IEMyLjYxOCwyLjcwNjY2NjY3IDMuMDM4LDIuODUxMzMzMzMgMy42MjYsMy4wMzggQzQuMjc5MzMzMzMsMy4yNDMzMzMzMyA0Ljc5MDMzMzMzLDMuNDQ4NjY2NjcgNS4xNTksMy42NTQgQzUuNTI3NjY2NjcsMy44NTkzMzMzMyA1Ljc5MTMzMzMzLDQuMDg4IDUuOTUsNC4zNCBDNi4xMDg2NjY2Nyw0LjU5MiA2LjE4OCw0Ljg5NTMzMzMzIDYuMTg4LDUuMjUgQzYuMTg4LDUuODE5MzMzMzMgNS45MzM2NjY2Nyw2LjI3NjY2NjY3IDUuNDI1LDYuNjIyIEM0LjkxNjMzMzMzLDYuOTY3MzMzMzMgNC4yMzczMzMzMyw3LjE0IDMuMzg4LDcuMTQgWiIgaWQ9InMiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTAuNTU2LDcuMTQgQzkuNjIyNjY2NjcsNy4xNCA4Ljg5NDY2NjY3LDYuODcxNjY2NjcgOC4zNzIsNi4zMzUgQzcuODQ5MzMzMzMsNS43OTgzMzMzMyA3LjU4OCw1LjA1ODY2NjY3IDcuNTg4LDQuMTE2IEw3LjU4OCwwLjY1OCBDNy41ODgsMC40NTI2NjY2NjcgNy42NDYzMzMzMywwLjI5MTY2NjY2NyA3Ljc2MywwLjE3NSBDNy44Nzk2NjY2NywwLjA1ODMzMzMzMzMgOC4wMzYsMCA4LjIzMiwwIEM4LjQzNzMzMzMzLDAgOC41OTYsMC4wNTgzMzMzMzMzIDguNzA4LDAuMTc1IEM4LjgyLDAuMjkxNjY2NjY3IDguODc2LDAuNDUyNjY2NjY3IDguODc2LDAuNjU4IEw4Ljg3Niw0LjExNiBDOC44NzYsNC43MzIgOS4wNDE2NjY2Nyw1LjIxMDMzMzMzIDkuMzczLDUuNTUxIEM5LjcwNDMzMzMzLDUuODkxNjY2NjcgMTAuMTY4NjY2Nyw2LjA2MiAxMC43NjYsNi4wNjIgQzExLjQ5NCw2LjA2MiAxMi4wNzk2NjY3LDUuODI4NjY2NjcgMTIuNTIzLDUuMzYyIEMxMi45NjYzMzMzLDQuODk1MzMzMzMgMTMuMTg4LDQuMjc0NjY2NjcgMTMuMTg4LDMuNSBMMTMuMTg4LDAuNjU4IEMxMy4xODgsMC40NTI2NjY2NjcgMTMuMjQ2MzMzMywwLjI5MTY2NjY2NyAxMy4zNjMsMC4xNzUgQzEzLjQ3OTY2NjcsMC4wNTgzMzMzMzMzIDEzLjY0MDY2NjcsMCAxMy44NDYsMCBDMTQuMDUxMzMzMywwIDE0LjIwNzY2NjcsMC4wNTYgMTQuMzE1LDAuMTY4IEMxNC40MjIzMzMzLDAuMjggMTQuNDc2LDAuNDQzMzMzMzMzIDE0LjQ3NiwwLjY1OCBMMTQuNDc2LDYuNDgyIEMxNC40NzYsNi42OTY2NjY2NyAxNC40MjIzMzMzLDYuODYgMTQuMzE1LDYuOTcyIEMxNC4yMDc2NjY3LDcuMDg0IDE0LjA1MTMzMzMsNy4xNCAxMy44NDYsNy4xNCBDMTMuNjQwNjY2Nyw3LjE0IDEzLjQ4Miw3LjA4MTY2NjY3IDEzLjM3LDYuOTY1IEMxMy4yNTgsNi44NDgzMzMzMyAxMy4yMDIsNi42ODczMzMzMyAxMy4yMDIsNi40ODIgTDEzLjIwMiw1Ljc5NiBDMTIuNTAyLDYuNjkyIDExLjYyLDcuMTQgMTAuNTU2LDcuMTQgWiIgaWQ9InUiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTYuODQyLDcuMTQgQzE2LjYzNjY2NjcsNy4xNCAxNi40NzgsNy4wODE2NjY2NyAxNi4zNjYsNi45NjUgQzE2LjI1NCw2Ljg0ODMzMzMzIDE2LjE5OCw2LjY4NzMzMzMzIDE2LjE5OCw2LjQ4MiBMMTYuMTk4LDAuNjU4IEMxNi4xOTgsMC40NTI2NjY2NjcgMTYuMjU0LDAuMjkxNjY2NjY3IDE2LjM2NiwwLjE3NSBDMTYuNDc4LDAuMDU4MzMzMzMzMyAxNi42MzY2NjY3LDAgMTYuODQyLDAgQzE3LjA0NzMzMzMsMCAxNy4yMDgzMzMzLDAuMDU4MzMzMzMzMyAxNy4zMjUsMC4xNzUgQzE3LjQ0MTY2NjcsMC4yOTE2NjY2NjcgMTcuNSwwLjQ1MjY2NjY2NyAxNy41LDAuNjU4IEwxNy41LDEuMzAyIEMxNy43MzMzMzMzLDAuODkxMzMzMzMzIDE4LjAzNjY2NjcsMC41NzE2NjY2NjcgMTguNDEsMC4zNDMgQzE4Ljc4MzMzMzMsMC4xMTQzMzMzMzMgMTkuMTgsMCAxOS42LDAgQzIwLjAyOTMzMzMsMCAyMC40MDczMzMzLDAuMTE0MzMzMzMzIDIwLjczNCwwLjM0MyBDMjEuMDYwNjY2NywwLjU3MTY2NjY2NyAyMS4zMDMzMzMzLDAuODkxMzMzMzMzIDIxLjQ2MiwxLjMwMiBDMjIuMTQzMzMzMywwLjQzNCAyMi45MTMzMzMzLDAgMjMuNzcyLDAgQzI0LjUwOTMzMzMsMCAyNS4wNzE2NjY3LDAuMjM4IDI1LjQ1OSwwLjcxNCBDMjUuODQ2MzMzMywxLjE5IDI2LjA0LDEuODgwNjY2NjcgMjYuMDQsMi43ODYgTDI2LjA0LDYuNDgyIEMyNi4wNCw2LjY4NzMzMzMzIDI1Ljk4MTY2NjcsNi44NDgzMzMzMyAyNS44NjUsNi45NjUgQzI1Ljc0ODMzMzMsNy4wODE2NjY2NyAyNS41OTIsNy4xNCAyNS4zOTYsNy4xNCBDMjUuMTkwNjY2Nyw3LjE0IDI1LjAzMiw3LjA4NCAyNC45Miw2Ljk3MiBDMjQuODA4LDYuODYgMjQuNzUyLDYuNjk2NjY2NjcgMjQuNzUyLDYuNDgyIEwyNC43NTIsMi43MyBDMjQuNzUyLDIuMTc5MzMzMzMgMjQuNjU4NjY2NywxLjc2NjMzMzMzIDI0LjQ3MiwxLjQ5MSBDMjQuMjg1MzMzMywxLjIxNTY2NjY3IDIzLjk3NzMzMzMsMS4wNzggMjMuNTQ4LDEuMDc4IEMyMi44MDEzMzMzLDEuMDc4IDIyLjIzNjY2NjcsMS41MDI2NjY2NyAyMS44NTQsMi4zNTIgTDIxLjg1NCw2LjQ4MiBDMjEuODU0LDYuNjg3MzMzMzMgMjEuNzk1NjY2Nyw2Ljg0ODMzMzMzIDIxLjY3OSw2Ljk2NSBDMjEuNTYyMzMzMyw3LjA4MTY2NjY3IDIxLjQwMTMzMzMsNy4xNCAyMS4xOTYsNy4xNCBDMjAuOTkwNjY2Nyw3LjE0IDIwLjgzNDMzMzMsNy4wODQgMjAuNzI3LDYuOTcyIEMyMC42MTk2NjY3LDYuODYgMjAuNTY2LDYuNjk2NjY2NjcgMjAuNTY2LDYuNDgyIEwyMC41NjYsMi45NjggQzIwLjU2NiwyLjM1MiAyMC40NjMzMzMzLDEuODgzIDIwLjI1OCwxLjU2MSBDMjAuMDUyNjY2NywxLjIzOSAxOS43NTQsMS4wNzggMTkuMzYyLDEuMDc4IEMxOC44MTEzMzMzLDEuMDc4IDE4LjM2MzMzMzMsMS4yOTUgMTguMDE4LDEuNzI5IEMxNy42NzI2NjY3LDIuMTYzIDE3LjUsMi43MjUzMzMzMyAxNy41LDMuNDE2IEwxNy41LDYuNDgyIEMxNy41LDYuNjg3MzMzMzMgMTcuNDQxNjY2Nyw2Ljg0ODMzMzMzIDE3LjMyNSw2Ljk2NSBDMTcuMjA4MzMzMyw3LjA4MTY2NjY3IDE3LjA0NzMzMzMsNy4xNCAxNi44NDIsNy4xNCBaIiBpZD0ibSIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgICAgICA8cGF0aCBkPSJNNDYuNjgxODE4MiwxMCBMMTcuMzE4MTgxOCwxMCBDMTYuMDQxMzcwNiwxMCAxNSwxMS4wNDEzNzA1IDE1LDEyLjMxODE4MTggTDE1LDQxLjY4MTgxODIgQzE1LDQyLjk1ODYyOTUgMTYuMDQxMzcwNiw0NCAxNy4zMTgxODE4LDQ0IEw0Ni42ODE4MTgyLDQ0IEM0Ny45NTg2Mjk1LDQ0IDQ5LDQyLjk1ODYyOTUgNDksNDEuNjgxODE4MiBMNDksMTIuMzE4MTgxOCBDNDksMTEuMDQxMzcwNSA0Ny45NTg2Mjk1LDEwIDQ2LjY4MTgxODIsMTAgWiBNNDIuMDQ1NDU0NiwzMC44NjM2MzY0IEwyMS45NTQ1NDU1LDMwLjg2MzYzNjQgTDIxLjk1NDU0NTUsMjkuMzE4MTgxOCBMNDIuMDQ1NDU0NiwyOS4zMTgxODE4IEw0Mi4wNDU0NTQ2LDMwLjg2MzYzNjQgWiBNNDIuMDQ1NDU0NiwyNC42ODE4MTgyIEwyMS45NTQ1NDU1LDI0LjY4MTgxODIgTDIxLjk1NDU0NTUsMjMuMTM2MzYzNiBMNDIuMDQ1NDU0NiwyMy4xMzYzNjM2IEw0Mi4wNDU0NTQ2LDI0LjY4MTgxODIgWiIgaWQ9ImVxdWFscyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_sum_eq()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">550<br />
tol=1</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


For equality comparisons, the tolerance creates a range `[value - tol, value + tol]` within which the aggregate is considered valid.


## Comparing Against Reference Data

Aggregate validations shine when comparing current data against a baseline or reference dataset. This is invaluable for detecting drift in data properties over time:


``` python
# Current data
current_data = pl.DataFrame({"revenue": [100, 200, 150, 175, 125]})

# Historical baseline
baseline_data = pl.DataFrame({"revenue": [95, 205, 145, 180, 130]})

(
    pb.Validate(data=current_data, reference=baseline_data)
    .col_sum_eq(columns="revenue", tol=50)   # Compare sums with tolerance
    .col_avg_eq(columns="revenue", tol=5)    # Compare averages with tolerance
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfc3VtX2VxPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9zdW1fZXEiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InN1bSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTguODg5MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zLjM4OCw3LjE0IEMyLjY2OTMzMzMzLDcuMTQgMi4wMyw3LjAxNjMzMzMzIDEuNDcsNi43NjkgQzAuOTEsNi41MjE2NjY2NyAwLjQ2NjY2NjY2Nyw2LjE4OCAwLjE0LDUuNzY4IEMwLjA0NjY2NjY2NjcsNS42NDY2NjY2NyAwLDUuNTI1MzMzMzMgMCw1LjQwNCBDMCw1LjE5ODY2NjY3IDAuMDk4LDUuMDQ0NjY2NjcgMC4yOTQsNC45NDIgQzAuMzc4LDQuODk1MzMzMzMgMC40NjY2NjY2NjcsNC44NzIgMC41Niw0Ljg3MiBDMC43MTg2NjY2NjcsNC44NzIgMC44ODIsNC45NTYgMS4wNSw1LjEyNCBDMS4zNTgsNS40NTA2NjY2NyAxLjY5ODY2NjY3LDUuNjg4NjY2NjcgMi4wNzIsNS44MzggQzIuNDQ1MzMzMzMsNS45ODczMzMzMyAyLjg3NDY2NjY3LDYuMDYyIDMuMzYsNi4wNjIgQzMuODA4LDYuMDYyIDQuMTc0MzMzMzMsNS45ODk2NjY2NyA0LjQ1OSw1Ljg0NSBDNC43NDM2NjY2Nyw1LjcwMDMzMzMzIDQuODg2LDUuNTAyIDQuODg2LDUuMjUgQzQuODg2LDUuMDYzMzMzMzMgNC44MzIzMzMzMyw0LjkwOTMzMzMzIDQuNzI1LDQuNzg4IEM0LjYxNzY2NjY3LDQuNjY2NjY2NjcgNC40MzMzMzMzMyw0LjU1IDQuMTcyLDQuNDM4IEMzLjkxMDY2NjY3LDQuMzI2IDMuNSw0LjE3NjY2NjY3IDIuOTQsMy45OSBDMS45OTczMzMzMywzLjY5MTMzMzMzIDEuMzMyMzMzMzMsMy4zODMzMzMzMyAwLjk0NSwzLjA2NiBDMC41NTc2NjY2NjcsMi43NDg2NjY2NyAwLjM2NCwyLjMzMzMzMzMzIDAuMzY0LDEuODIgQzAuMzY0LDEuMjY5MzMzMzMgMC42MDQzMzMzMzMsMC44MjgzMzMzMzMgMS4wODUsMC40OTcgQzEuNTY1NjY2NjcsMC4xNjU2NjY2NjcgMi4yMTY2NjY2NywwIDMuMDM4LDAgQzMuNjQ0NjY2NjcsMCA0LjIwNywwLjEwMDMzMzMzMyA0LjcyNSwwLjMwMSBDNS4yNDMsMC41MDE2NjY2NjcgNS42MjgsMC43NzQ2NjY2NjcgNS44OCwxLjEyIEM1Ljk2NCwxLjIzMiA2LjAwNiwxLjM0ODY2NjY3IDYuMDA2LDEuNDcgQzYuMDA2LDEuNjI4NjY2NjcgNS45MjIsMS43NjQgNS43NTQsMS44NzYgQzUuNjMyNjY2NjcsMS45NTA2NjY2NyA1LjUxMTMzMzMzLDEuOTg4IDUuMzksMS45ODggQzUuMjAzMzMzMzMsMS45ODggNS4wMjYsMS45MDQgNC44NTgsMS43MzYgQzQuNjM0LDEuNTEyIDQuMzc1LDEuMzQ2MzMzMzMgNC4wODEsMS4yMzkgQzMuNzg3LDEuMTMxNjY2NjcgMy40MzQ2NjY2NywxLjA3OCAzLjAyNCwxLjA3OCBDMi41ODUzMzMzMywxLjA3OCAyLjI1MTY2NjY3LDEuMTQzMzMzMzMgMi4wMjMsMS4yNzQgQzEuNzk0MzMzMzMsMS40MDQ2NjY2NyAxLjY4LDEuNTkxMzMzMzMgMS42OCwxLjgzNCBDMS42OCwyLjAyMDY2NjY3IDEuNzMxMzMzMzMsMi4xNyAxLjgzNCwyLjI4MiBDMS45MzY2NjY2NywyLjM5NCAyLjExNCwyLjUwMTMzMzMzIDIuMzY2LDIuNjA0IEMyLjYxOCwyLjcwNjY2NjY3IDMuMDM4LDIuODUxMzMzMzMgMy42MjYsMy4wMzggQzQuMjc5MzMzMzMsMy4yNDMzMzMzMyA0Ljc5MDMzMzMzLDMuNDQ4NjY2NjcgNS4xNTksMy42NTQgQzUuNTI3NjY2NjcsMy44NTkzMzMzMyA1Ljc5MTMzMzMzLDQuMDg4IDUuOTUsNC4zNCBDNi4xMDg2NjY2Nyw0LjU5MiA2LjE4OCw0Ljg5NTMzMzMzIDYuMTg4LDUuMjUgQzYuMTg4LDUuODE5MzMzMzMgNS45MzM2NjY2Nyw2LjI3NjY2NjY3IDUuNDI1LDYuNjIyIEM0LjkxNjMzMzMzLDYuOTY3MzMzMzMgNC4yMzczMzMzMyw3LjE0IDMuMzg4LDcuMTQgWiIgaWQ9InMiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTAuNTU2LDcuMTQgQzkuNjIyNjY2NjcsNy4xNCA4Ljg5NDY2NjY3LDYuODcxNjY2NjcgOC4zNzIsNi4zMzUgQzcuODQ5MzMzMzMsNS43OTgzMzMzMyA3LjU4OCw1LjA1ODY2NjY3IDcuNTg4LDQuMTE2IEw3LjU4OCwwLjY1OCBDNy41ODgsMC40NTI2NjY2NjcgNy42NDYzMzMzMywwLjI5MTY2NjY2NyA3Ljc2MywwLjE3NSBDNy44Nzk2NjY2NywwLjA1ODMzMzMzMzMgOC4wMzYsMCA4LjIzMiwwIEM4LjQzNzMzMzMzLDAgOC41OTYsMC4wNTgzMzMzMzMzIDguNzA4LDAuMTc1IEM4LjgyLDAuMjkxNjY2NjY3IDguODc2LDAuNDUyNjY2NjY3IDguODc2LDAuNjU4IEw4Ljg3Niw0LjExNiBDOC44NzYsNC43MzIgOS4wNDE2NjY2Nyw1LjIxMDMzMzMzIDkuMzczLDUuNTUxIEM5LjcwNDMzMzMzLDUuODkxNjY2NjcgMTAuMTY4NjY2Nyw2LjA2MiAxMC43NjYsNi4wNjIgQzExLjQ5NCw2LjA2MiAxMi4wNzk2NjY3LDUuODI4NjY2NjcgMTIuNTIzLDUuMzYyIEMxMi45NjYzMzMzLDQuODk1MzMzMzMgMTMuMTg4LDQuMjc0NjY2NjcgMTMuMTg4LDMuNSBMMTMuMTg4LDAuNjU4IEMxMy4xODgsMC40NTI2NjY2NjcgMTMuMjQ2MzMzMywwLjI5MTY2NjY2NyAxMy4zNjMsMC4xNzUgQzEzLjQ3OTY2NjcsMC4wNTgzMzMzMzMzIDEzLjY0MDY2NjcsMCAxMy44NDYsMCBDMTQuMDUxMzMzMywwIDE0LjIwNzY2NjcsMC4wNTYgMTQuMzE1LDAuMTY4IEMxNC40MjIzMzMzLDAuMjggMTQuNDc2LDAuNDQzMzMzMzMzIDE0LjQ3NiwwLjY1OCBMMTQuNDc2LDYuNDgyIEMxNC40NzYsNi42OTY2NjY2NyAxNC40MjIzMzMzLDYuODYgMTQuMzE1LDYuOTcyIEMxNC4yMDc2NjY3LDcuMDg0IDE0LjA1MTMzMzMsNy4xNCAxMy44NDYsNy4xNCBDMTMuNjQwNjY2Nyw3LjE0IDEzLjQ4Miw3LjA4MTY2NjY3IDEzLjM3LDYuOTY1IEMxMy4yNTgsNi44NDgzMzMzMyAxMy4yMDIsNi42ODczMzMzMyAxMy4yMDIsNi40ODIgTDEzLjIwMiw1Ljc5NiBDMTIuNTAyLDYuNjkyIDExLjYyLDcuMTQgMTAuNTU2LDcuMTQgWiIgaWQ9InUiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTYuODQyLDcuMTQgQzE2LjYzNjY2NjcsNy4xNCAxNi40NzgsNy4wODE2NjY2NyAxNi4zNjYsNi45NjUgQzE2LjI1NCw2Ljg0ODMzMzMzIDE2LjE5OCw2LjY4NzMzMzMzIDE2LjE5OCw2LjQ4MiBMMTYuMTk4LDAuNjU4IEMxNi4xOTgsMC40NTI2NjY2NjcgMTYuMjU0LDAuMjkxNjY2NjY3IDE2LjM2NiwwLjE3NSBDMTYuNDc4LDAuMDU4MzMzMzMzMyAxNi42MzY2NjY3LDAgMTYuODQyLDAgQzE3LjA0NzMzMzMsMCAxNy4yMDgzMzMzLDAuMDU4MzMzMzMzMyAxNy4zMjUsMC4xNzUgQzE3LjQ0MTY2NjcsMC4yOTE2NjY2NjcgMTcuNSwwLjQ1MjY2NjY2NyAxNy41LDAuNjU4IEwxNy41LDEuMzAyIEMxNy43MzMzMzMzLDAuODkxMzMzMzMzIDE4LjAzNjY2NjcsMC41NzE2NjY2NjcgMTguNDEsMC4zNDMgQzE4Ljc4MzMzMzMsMC4xMTQzMzMzMzMgMTkuMTgsMCAxOS42LDAgQzIwLjAyOTMzMzMsMCAyMC40MDczMzMzLDAuMTE0MzMzMzMzIDIwLjczNCwwLjM0MyBDMjEuMDYwNjY2NywwLjU3MTY2NjY2NyAyMS4zMDMzMzMzLDAuODkxMzMzMzMzIDIxLjQ2MiwxLjMwMiBDMjIuMTQzMzMzMywwLjQzNCAyMi45MTMzMzMzLDAgMjMuNzcyLDAgQzI0LjUwOTMzMzMsMCAyNS4wNzE2NjY3LDAuMjM4IDI1LjQ1OSwwLjcxNCBDMjUuODQ2MzMzMywxLjE5IDI2LjA0LDEuODgwNjY2NjcgMjYuMDQsMi43ODYgTDI2LjA0LDYuNDgyIEMyNi4wNCw2LjY4NzMzMzMzIDI1Ljk4MTY2NjcsNi44NDgzMzMzMyAyNS44NjUsNi45NjUgQzI1Ljc0ODMzMzMsNy4wODE2NjY2NyAyNS41OTIsNy4xNCAyNS4zOTYsNy4xNCBDMjUuMTkwNjY2Nyw3LjE0IDI1LjAzMiw3LjA4NCAyNC45Miw2Ljk3MiBDMjQuODA4LDYuODYgMjQuNzUyLDYuNjk2NjY2NjcgMjQuNzUyLDYuNDgyIEwyNC43NTIsMi43MyBDMjQuNzUyLDIuMTc5MzMzMzMgMjQuNjU4NjY2NywxLjc2NjMzMzMzIDI0LjQ3MiwxLjQ5MSBDMjQuMjg1MzMzMywxLjIxNTY2NjY3IDIzLjk3NzMzMzMsMS4wNzggMjMuNTQ4LDEuMDc4IEMyMi44MDEzMzMzLDEuMDc4IDIyLjIzNjY2NjcsMS41MDI2NjY2NyAyMS44NTQsMi4zNTIgTDIxLjg1NCw2LjQ4MiBDMjEuODU0LDYuNjg3MzMzMzMgMjEuNzk1NjY2Nyw2Ljg0ODMzMzMzIDIxLjY3OSw2Ljk2NSBDMjEuNTYyMzMzMyw3LjA4MTY2NjY3IDIxLjQwMTMzMzMsNy4xNCAyMS4xOTYsNy4xNCBDMjAuOTkwNjY2Nyw3LjE0IDIwLjgzNDMzMzMsNy4wODQgMjAuNzI3LDYuOTcyIEMyMC42MTk2NjY3LDYuODYgMjAuNTY2LDYuNjk2NjY2NjcgMjAuNTY2LDYuNDgyIEwyMC41NjYsMi45NjggQzIwLjU2NiwyLjM1MiAyMC40NjMzMzMzLDEuODgzIDIwLjI1OCwxLjU2MSBDMjAuMDUyNjY2NywxLjIzOSAxOS43NTQsMS4wNzggMTkuMzYyLDEuMDc4IEMxOC44MTEzMzMzLDEuMDc4IDE4LjM2MzMzMzMsMS4yOTUgMTguMDE4LDEuNzI5IEMxNy42NzI2NjY3LDIuMTYzIDE3LjUsMi43MjUzMzMzMyAxNy41LDMuNDE2IEwxNy41LDYuNDgyIEMxNy41LDYuNjg3MzMzMzMgMTcuNDQxNjY2Nyw2Ljg0ODMzMzMzIDE3LjMyNSw2Ljk2NSBDMTcuMjA4MzMzMyw3LjA4MTY2NjY3IDE3LjA0NzMzMzMsNy4xNCAxNi44NDIsNy4xNCBaIiBpZD0ibSIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgICAgICA8cGF0aCBkPSJNNDYuNjgxODE4MiwxMCBMMTcuMzE4MTgxOCwxMCBDMTYuMDQxMzcwNiwxMCAxNSwxMS4wNDEzNzA1IDE1LDEyLjMxODE4MTggTDE1LDQxLjY4MTgxODIgQzE1LDQyLjk1ODYyOTUgMTYuMDQxMzcwNiw0NCAxNy4zMTgxODE4LDQ0IEw0Ni42ODE4MTgyLDQ0IEM0Ny45NTg2Mjk1LDQ0IDQ5LDQyLjk1ODYyOTUgNDksNDEuNjgxODE4MiBMNDksMTIuMzE4MTgxOCBDNDksMTEuMDQxMzcwNSA0Ny45NTg2Mjk1LDEwIDQ2LjY4MTgxODIsMTAgWiBNNDIuMDQ1NDU0NiwzMC44NjM2MzY0IEwyMS45NTQ1NDU1LDMwLjg2MzYzNjQgTDIxLjk1NDU0NTUsMjkuMzE4MTgxOCBMNDIuMDQ1NDU0NiwyOS4zMTgxODE4IEw0Mi4wNDU0NTQ2LDMwLjg2MzYzNjQgWiBNNDIuMDQ1NDU0NiwyNC42ODE4MTgyIEwyMS45NTQ1NDU1LDI0LjY4MTgxODIgTDIxLjk1NDU0NTUsMjMuMTM2MzYzNiBMNDIuMDQ1NDU0NiwyMy4xMzYzNjM2IEw0Mi4wNDU0NTQ2LDI0LjY4MTgxODIgWiIgaWQ9ImVxdWFscyIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_sum_eq()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">revenue</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">ref('revenue')<br />
tol=50</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfYXZnX2VxPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF9hdmdfZXEiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTAwMDAwLCAxLjUwMDAwMCkiIGZpbGwtcnVsZT0ibm9uemVybyI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01NSwwIEM1Ny40ODUyODEzLDAgNTkuNzM1MjgxMywxLjAwNzM1OTMxIDYxLjM2Mzk2MSwyLjYzNjAzODk3IEM2Mi45OTI2NDA3LDQuMjY0NzE4NjMgNjQsNi41MTQ3MTg2MyA2NCw5IEw2NCw5IEw2NCw2NCBMOSw2NCBDNi41MTQ3MTg2Miw2NCA0LjI2NDcxODYyLDYyLjk5MjY0MDcgMi42MzYwMzg5Nyw2MS4zNjM5NjEgQzEuMDA3MzU5MzEsNTkuNzM1MjgxNCAwLDU3LjQ4NTI4MTQgMCw1NSBMMCw1NSBMMCw5IEMwLDYuNTE0NzE4NjMgMS4wMDczNTkzMSw0LjI2NDcxODYzIDIuNjM2MDM4OTcsMi42MzYwMzg5NyBDNC4yNjQ3MTg2MiwxLjAwNzM1OTMxIDYuNTE0NzE4NjIsMCA5LDAgTDksMCBMNTUsMCBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9ImF2ZyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMjAuMDU4MDAwLCA1MC4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yLjk0LDcuMTQgQzIuMDM0NjY2NjcsNy4xNCAxLjMxODMzMzMzLDYuOTM0NjY2NjcgMC43OTEsNi41MjQgQzAuMjYzNjY2NjY3LDYuMTEzMzMzMzMgMCw1LjU1OCAwLDQuODU4IEMwLDQuMTU4IDAuMjc3NjY2NjY3LDMuNjAyNjY2NjcgMC44MzMsMy4xOTIgQzEuMzg4MzMzMzMsMi43ODEzMzMzMyAyLjE0NjY2NjY3LDIuNTc2IDMuMTA4LDIuNTc2IEMzLjYxMiwyLjU3NiA0LjA2MjMzMzMzLDIuNjQxMzMzMzMgNC40NTksMi43NzIgQzQuODU1NjY2NjcsMi45MDI2NjY2NyA1LjE3NTMzMzMzLDMuMDggNS40MTgsMy4zMDQgTDUuNDE4LDIuNjA0IEM1LjQxOCwyLjEgNS4yNTcsMS43MTUgNC45MzUsMS40NDkgQzQuNjEzLDEuMTgzIDQuMTM5MzMzMzMsMS4wNSAzLjUxNCwxLjA1IEMyLjY4MzMzMzMzLDEuMDUgMS45NjQ2NjY2NywxLjI0MTMzMzMzIDEuMzU4LDEuNjI0IEMxLjE4MDY2NjY3LDEuNzI2NjY2NjcgMS4wMjY2NjY2NywxLjc3OCAwLjg5NiwxLjc3OCBDMC43NzQ2NjY2NjcsMS43NzggMC42NTMzMzMzMzMsMS43MjY2NjY2NyAwLjUzMiwxLjYyNCBDMC40MjkzMzMzMzMsMS41MjEzMzMzMyAwLjM3OCwxLjQgMC4zNzgsMS4yNiBDMC4zNzgsMS4xMDEzMzMzMyAwLjQ1NzMzMzMzMywwLjk1NjY2NjY2NyAwLjYxNiwwLjgyNiBDMC45NTIsMC41NzQgMS4zNzY2NjY2NywwLjM3MzMzMzMzMyAxLjg5LDAuMjI0IEMyLjQwMzMzMzMzLDAuMDc0NjY2NjY2NyAyLjk0LDAgMy41LDAgQzQuNTE3MzMzMzMsMCA1LjMxMDY2NjY3LDAuMjMxIDUuODgsMC42OTMgQzYuNDQ5MzMzMzMsMS4xNTUgNi43MzQsMS44MDEzMzMzMyA2LjczNCwyLjYzMiBMNi43MzQsNi41MSBDNi43MzQsNi43MTUzMzMzMyA2LjY4MDMzMzMzLDYuODcxNjY2NjcgNi41NzMsNi45NzkgQzYuNDY1NjY2NjcsNy4wODYzMzMzMyA2LjMwOTMzMzMzLDcuMTQgNi4xMDQsNy4xNCBDNS44OTg2NjY2Nyw3LjE0IDUuNzQsNy4wODYzMzMzMyA1LjYyOCw2Ljk3OSBDNS41MTYsNi44NzE2NjY2NyA1LjQ2LDYuNzI0NjY2NjcgNS40Niw2LjUzOCBMNS40Niw2LjE0NiBDNC45LDYuODA4NjY2NjcgNC4wNiw3LjE0IDIuOTQsNy4xNCBaIE0zLjE2NCw2LjE0NiBDMy44NDUzMzMzMyw2LjE0NiA0LjM5MzY2NjY3LDYuMDIyMzMzMzMgNC44MDksNS43NzUgQzUuMjI0MzMzMzMsNS41Mjc2NjY2NyA1LjQzMiw1LjIwMzMzMzMzIDUuNDMyLDQuODAyIEM1LjQzMiw0LjQxIDUuMjMzNjY2NjcsNC4xMDQzMzMzMyA0LjgzNywzLjg4NSBDNC40NDAzMzMzMywzLjY2NTY2NjY3IDMuODkyLDMuNTU2IDMuMTkyLDMuNTU2IEMyLjU4NTMzMzMzLDMuNTU2IDIuMTE4NjY2NjcsMy42NjggMS43OTIsMy44OTIgQzEuNDY1MzMzMzMsNC4xMTYgMS4zMDIsNC40MzggMS4zMDIsNC44NTggQzEuMzAyLDUuMjY4NjY2NjcgMS40NjMsNS41ODYgMS43ODUsNS44MSBDMi4xMDcsNi4wMzQgMi41NjY2NjY2Nyw2LjE0NiAzLjE2NCw2LjE0NiBaIiBpZD0iYSIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0xMS40OTQsNy4xNCBDMTAuNTMyNjY2Nyw3LjE0IDkuODUxMzMzMzMsNi41NjEzMzMzMyA5LjQ1LDUuNDA0IEw3Ljc5OCwwLjc4NCBDNy43NywwLjcwOTMzMzMzMyA3Ljc1NiwwLjY0NCA3Ljc1NiwwLjU4OCBDNy43NTYsMC40NjY2NjY2NjcgNy43OTgsMC4zNTcgNy44ODIsMC4yNTkgQzcuOTY2LDAuMTYxIDguMDc4LDAuMDg4NjY2NjY2NyA4LjIxOCwwLjA0MiBDOC4zMDIsMC4wMTQgOC4zODYsMCA4LjQ3LDAgQzguNjEsMCA4LjczMTMzMzMzLDAuMDM1IDguODM0LDAuMTA1IEM4LjkzNjY2NjY3LDAuMTc1IDkuMDA2NjY2NjcsMC4yNzA2NjY2NjcgOS4wNDQsMC4zOTIgTDEwLjczOCw1LjEzOCBDMTAuODU5MzMzMyw1LjQ4MzMzMzMzIDEwLjk3Niw1LjcyNiAxMS4wODgsNS44NjYgQzExLjIsNi4wMDYgMTEuMzM1MzMzMyw2LjA3NiAxMS40OTQsNi4wNzYgQzExLjY1MjY2NjcsNi4wNzYgMTEuNzg4LDYuMDA4MzMzMzMgMTEuOSw1Ljg3MyBDMTIuMDEyLDUuNzM3NjY2NjcgMTIuMTI4NjY2Nyw1LjUwMiAxMi4yNSw1LjE2NiBMMTMuOTQ0LDAuMzkyIEMxMy45OTA2NjY3LDAuMjcwNjY2NjY3IDE0LjA2MywwLjE3NzMzMzMzMyAxNC4xNjEsMC4xMTIgQzE0LjI1OSwwLjA0NjY2NjY2NjcgMTQuMzc4LDAuMDE0IDE0LjUxOCwwLjAxNCBDMTQuNzMyNjY2NywwLjAxNCAxNC45MDUzMzMzLDAuMDcyMzMzMzMzMyAxNS4wMzYsMC4xODkgQzE1LjE2NjY2NjcsMC4zMDU2NjY2NjcgMTUuMjMyLDAuNDQzMzMzMzMzIDE1LjIzMiwwLjYwMiBDMTUuMjMyLDAuNjg2IDE1LjIyMjY2NjcsMC43NDY2NjY2NjcgMTUuMjA0LDAuNzg0IEwxMy41MzgsNS40MzIgQzEzLjMyMzMzMzMsNi4wMTA2NjY2NyAxMy4wNTAzMzMzLDYuNDQgMTIuNzE5LDYuNzIgQzEyLjM4NzY2NjcsNyAxMS45NzkzMzMzLDcuMTQgMTEuNDk0LDcuMTQgWiIgaWQ9InYiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMTkuODEsMTAuNTE0IEMxOS4xMTkzMzMzLDEwLjUxNCAxOC40OTE2NjY3LDEwLjQxMTMzMzMgMTcuOTI3LDEwLjIwNiBDMTcuMzYyMzMzMywxMC4wMDA2NjY3IDE2LjkwNzMzMzMsOS43MTEzMzMzMyAxNi41NjIsOS4zMzggQzE2LjQ2ODY2NjcsOS4yMzUzMzMzMyAxNi40MjIsOS4xMjMzMzMzMyAxNi40MjIsOS4wMDIgQzE2LjQyMiw4Ljg0MzMzMzMzIDE2LjUxMDY2NjcsOC42ODkzMzMzMyAxNi42ODgsOC41NCBDMTYuNzcyLDguNDY1MzMzMzMgMTYuODc5MzMzMyw4LjQyOCAxNy4wMSw4LjQyOCBDMTcuMjA2LDguNDI4IDE3LjM5MjY2NjcsOC41MjEzMzMzMyAxNy41Nyw4LjcwOCBDMTcuNzk0LDguOTQxMzMzMzMgMTguMDg1NjY2Nyw5LjEyNTY2NjY3IDE4LjQ0NSw5LjI2MSBDMTguODA0MzMzMyw5LjM5NjMzMzMzIDE5LjIyNjY2NjcsOS40NjQgMTkuNzEyLDkuNDY0IEMyMC41NjEzMzMzLDkuNDY0IDIxLjIxNDY2NjcsOS4yNDQ2NjY2NyAyMS42NzIsOC44MDYgQzIyLjEyOTMzMzMsOC4zNjczMzMzMyAyMi4zNTMzMzMzLDcuNzQyIDIyLjM0NCw2LjkzIEwyMi4zNDQsNS44NjYgQzIyLjAyNjY2NjcsNi4yNjczMzMzMyAyMS42MjUzMzMzLDYuNTg3IDIxLjE0LDYuODI1IEMyMC42NTQ2NjY3LDcuMDYzIDIwLjA4NTMzMzMsNy4xODIgMTkuNDMyLDcuMTgyIEMxOC4yNTYsNy4xODIgMTcuMzQ2LDYuODY3IDE2LjcwMiw2LjIzNyBDMTYuMDU4LDUuNjA3IDE1LjczNiw0Ljc1MDY2NjY3IDE1LjczNiwzLjY2OCBDMTUuNzM2LDIuNTU3MzMzMzMgMTYuMDYyNjY2NywxLjY2ODMzMzMzIDE2LjcxNiwxLjAwMSBDMTcuMzY5MzMzMywwLjMzMzY2NjY2NyAxOC4zMDI2NjY3LDAgMTkuNTE2LDAgQzIwLjA5NDY2NjcsMCAyMC42MjIsMC4xMTY2NjY2NjcgMjEuMDk4LDAuMzUgQzIxLjU3NCwwLjU4MzMzMzMzMyAyMS45ODkzMzMzLDAuOTE0NjY2NjY3IDIyLjM0NCwxLjM0NCBMMjIuMzQ0LDAuNjMgQzIyLjM0NCwwLjQ0MzMzMzMzMyAyMi40MDcsMC4yOTE2NjY2NjcgMjIuNTMzLDAuMTc1IEMyMi42NTksMC4wNTgzMzMzMzMzIDIyLjgxNTMzMzMsMCAyMy4wMDIsMCBDMjMuMTk4LDAgMjMuMzU0MzMzMywwLjA1NiAyMy40NzEsMC4xNjggQzIzLjU4NzY2NjcsMC4yOCAyMy42NDYsMC40MzQgMjMuNjQ2LDAuNjMgTDIzLjY0Niw2LjkzIEMyMy42NDYsNy42NDg2NjY2NyAyMy40ODczMzMzLDguMjc4NjY2NjcgMjMuMTcsOC44MiBDMjIuODUyNjY2Nyw5LjM2MTMzMzMzIDIyLjQwNDY2NjcsOS43NzkgMjEuODI2LDEwLjA3MyBDMjEuMjQ3MzMzMywxMC4zNjcgMjAuNTc1MzMzMywxMC41MTQgMTkuODEsMTAuNTE0IFogTTE5LjYxNCw2LjA5IEMyMC4xMDg2NjY3LDYuMDkgMjAuNTYzNjY2Nyw1Ljk5NjY2NjY3IDIwLjk3OSw1LjgxIEMyMS4zOTQzMzMzLDUuNjIzMzMzMzMgMjEuNzI1NjY2Nyw1LjM0MzMzMzMzIDIxLjk3Myw0Ljk3IEMyMi4yMjAzMzMzLDQuNTk2NjY2NjcgMjIuMzQ0LDQuMTM5MzMzMzMgMjIuMzQ0LDMuNTk4IEMyMi4zNDQsMi44NTEzMzMzMyAyMi4xMDYsMi4yNDQ2NjY2NyAyMS42MywxLjc3OCBDMjEuMTU0LDEuMzExMzMzMzMgMjAuNDk2LDEuMDc4IDE5LjY1NiwxLjA3OCBDMTguODI1MzMzMywxLjA3OCAxOC4xODEzMzMzLDEuMzA0MzMzMzMgMTcuNzI0LDEuNzU3IEMxNy4yNjY2NjY3LDIuMjA5NjY2NjcgMTcuMDM4LDIuODQ2NjY2NjcgMTcuMDM4LDMuNjY4IEMxNy4wMzgsNC40MTQ2NjY2NyAxNy4yNTk2NjY3LDUuMDA1IDE3LjcwMyw1LjQzOSBDMTguMTQ2MzMzMyw1Ljg3MyAxOC43ODMzMzMzLDYuMDkgMTkuNjE0LDYuMDkgWiIgaWQ9ImciIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPHBhdGggZD0iTTQ2LjY4MTgxODIsMTAgTDE3LjMxODE4MTgsMTAgQzE2LjA0MTM3MDYsMTAgMTUsMTEuMDQxMzcwNSAxNSwxMi4zMTgxODE4IEwxNSw0MS42ODE4MTgyIEMxNSw0Mi45NTg2Mjk1IDE2LjA0MTM3MDYsNDQgMTcuMzE4MTgxOCw0NCBMNDYuNjgxODE4Miw0NCBDNDcuOTU4NjI5NSw0NCA0OSw0Mi45NTg2Mjk1IDQ5LDQxLjY4MTgxODIgTDQ5LDEyLjMxODE4MTggQzQ5LDExLjA0MTM3MDUgNDcuOTU4NjI5NSwxMCA0Ni42ODE4MTgyLDEwIFogTTQyLjA0NTQ1NDYsMzAuODYzNjM2NCBMMjEuOTU0NTQ1NSwzMC44NjM2MzY0IEwyMS45NTQ1NDU1LDI5LjMxODE4MTggTDQyLjA0NTQ1NDYsMjkuMzE4MTgxOCBMNDIuMDQ1NDU0NiwzMC44NjM2MzY0IFogTTQyLjA0NTQ1NDYsMjQuNjgxODE4MiBMMjEuOTU0NTQ1NSwyNC42ODE4MTgyIEwyMS45NTQ1NDU1LDIzLjEzNjM2MzYgTDQyLjA0NTQ1NDYsMjMuMTM2MzYzNiBMNDIuMDQ1NDU0NiwyNC42ODE4MTgyIFoiIGlkPSJlcXVhbHMiIGZpbGw9IiMwMDAwMDAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_avg_eq()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">revenue</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">ref('revenue')<br />
tol=5</td>
OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


When `value=None` (the default) and reference data is set, aggregate methods automatically compare against the same column in the reference data.


# 6. Custom Validations with [specially()](../../reference/Validate.specially.md#pointblank.Validate.specially)

While Pointblank provides over 40 built-in validation methods, there are times when you need to implement custom validation logic that goes beyond these standard checks. The <a href="../../reference/Validate.specially.html#pointblank.Validate.specially" class="gdls-link"><code>Validate.specially()</code></a> method gives you complete flexibility to create bespoke validations for domain-specific business rules, complex multi-column checks, or cross-dataset referential integrity constraints.


## Basic Custom Validations

The [specially()](../../reference/Validate.specially.md#pointblank.Validate.specially) method accepts a callable function that performs your custom validation logic. The function should return boolean values indicating whether each test unit passes:


``` python
import polars as pl

simple_tbl = pl.DataFrame({
    "a": [5, 7, 1, 3, 9, 4],
    "b": [6, 3, 0, 5, 8, 2]
})

# Custom validation: sum of two columns must be positive
def validate_sum_positive(data):
    return data.select(pl.col("a") + pl.col("b") > 0)

(
    pb.Validate(data=simple_tbl)
    .specially(
        expr=validate_sum_positive,
        brief="Sum of columns 'a' and 'b' must be positive"
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+c3BlY2lhbGx5PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InNwZWNpYWxseSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMjA2ODk3KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8ZyBpZD0ic3RhciIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOC41MDAwMDAsIDguNTAwMDAwKSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yNSwwIEMyNC41ODc0NDg0LDAgMjQuMjE3NDUxNywwLjI1NDAwMjg0NyAyNC4wNjgzNTksMC42Mzg2NzE5IEwxNy45MDIzNDQsMTYuNTM1MTU2IEwwLjk0OTIxODc1LDE3LjQwMDM5MSBDMC41MzYxMjQ0MDksMTcuNDIxMzAxMyAwLjE3ODUyNDU4LDE3LjY5NDM1MjMgMC4wNDk1NDQ2Mzk1LDE4LjA4NzM1MTUgQy0wLjA3OTQzNTMwMTIsMTguNDgwMzUwNyAwLjA0NjgyMDQ1MiwxOC45MTIyMDAyIDAuMzY3MTg3NSwxOS4xNzM4MjggTDEzLjU2ODM1OSwyOS45NjY3OTcgTDkuMjMyNDIxOSw0Ni4zNDM3NSBDOS4xMjY0Njk2Myw0Ni43NDI4MDA5IDkuMjc2NjMwNTgsNDcuMTY1OTQzMyA5LjYxMDQyNjk4LDQ3LjQwODk0MDIgQzkuOTQ0MjIzMzgsNDcuNjUxOTM3IDEwLjM5MzAzNDUsNDcuNjY0ODM0IDEwLjc0MDIzNCw0Ny40NDE0MDYgTDI1LDM4LjI4OTA2MiBMMzkuMjU5NzY2LDQ3LjQ0MTQwNiBDMzkuNjA2OTY1NSw0Ny42NjQ4MzM5IDQwLjA1NTc3NjYsNDcuNjUxOTM2OSA0MC4zODk1NzI5LDQ3LjQwODk0MDEgQzQwLjcyMzM2OTMsNDcuMTY1OTQzMiA0MC44NzM1MzAyLDQ2Ljc0MjgwMDkgNDAuNzY3NTc4LDQ2LjM0Mzc1IEwzNi40MzE2NDEsMjkuOTY2Nzk3IEw0OS42MzI4MTIsMTkuMTczODI4IEM0OS45NTMxNzksMTguOTEyMjAwMiA1MC4wNzk0MzQ4LDE4LjQ4MDM1MDcgNDkuOTUwNDU0OSwxOC4wODczNTE2IEM0OS44MjE0NzUsMTcuNjk0MzUyNCA0OS40NjM4NzUzLDE3LjQyMTMwMTQgNDkuMDUwNzgxLDE3LjQwMDM5MSBMMzIuMDk3NjU2LDE2LjUzNTE1NiBMMjUuOTMxNjQxLDAuNjM4NjcxOSBDMjUuNzgyNTQ4MywwLjI1NDAwMjg0NyAyNS40MTI1NTE2LDAgMjUsMCBaIE0yNSwzLjc2MzY3MTkgTDMwLjQ2Njc5NywxNy44NjEzMjggQzMwLjYwOTY4OSwxOC4yMjkxNDE2IDMwLjk1NTQ5NjIsMTguNDc4NTUxNSAzMS4zNDk2MDksMTguNDk4MDQ3IEw0Ni4zNTkzNzUsMTkuMjY1NjI1IEwzNC42Njc5NjksMjguODI2MTcyIEMzNC4zNjQ2MDU0LDI5LjA3NDIxMTQgMzQuMjM0MDQ5MywyOS40NzY1Njc5IDM0LjMzMzk4NCwyOS44NTU0NjkgTDM4LjE3NTc4MSw0NC4zNjkxNDEgTDI1LjU0MTAxNiwzNi4yNTc4MTIgQzI1LjIxMTQ3ODksMzYuMDQ1ODUzNiAyNC43ODg1MjExLDM2LjA0NTg1MzYgMjQuNDU4OTg0LDM2LjI1NzgxMiBMMTEuODI0MjE5LDQ0LjM2OTE0MSBMMTUuNjY2MDE2LDI5Ljg1NTQ2OSBDMTUuNzY1OTUwNywyOS40NzY1Njc5IDE1LjYzNTM5NDYsMjkuMDc0MjExNCAxNS4zMzIwMzEsMjguODI2MTcyIEwzLjY0MDYyNSwxOS4yNjU2MjUgTDE4LjY1MDM5MSwxOC40OTgwNDcgQzE5LjA0NDUwMzgsMTguNDc4NTUxNSAxOS4zOTAzMTEsMTguMjI5MTQxNiAxOS41MzMyMDMsMTcuODYxMzI4IEwyNSwzLjc2MzY3MTkgWiIgaWQ9IlNoYXBlIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

specially()


<p>Sum of columns 'a' and 'b' must be positive</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">EXPR</td>
gxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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


This validation passes because all rows have a positive sum for columns `a` and `b`. The [specially()](../../reference/Validate.specially.md#pointblank.Validate.specially) method provides the flexibility to implement any validation logic you can express in Python, making it a powerful tool for custom data quality checks.


## Cross-Dataset Referential Integrity

One powerful use case for [specially()](../../reference/Validate.specially.md#pointblank.Validate.specially) is validating relationships between multiple datasets. This is particularly valuable for checking foreign key constraints, conditional existence rules, and cardinality relationships that span multiple tables.


### Foreign Key Validation

Verify that all keys in one dataset exist in another:


``` python
# Create related datasets: Orders and OrderDetails
orders = pl.DataFrame({
    "order_id": [1, 2, 3, 4, 5],
    "customer_id": ["A", "B", "A", "C", "B"],
    "status": ["completed", "pending", "completed", "cancelled", "completed"]
})

order_details = pl.DataFrame({
    "detail_id": [101, 102, 103, 104, 105, 106, 107, 108, 109],
    "order_id": [1, 1, 1, 2, 3, 3, 4, 5, 5],
    "product_id": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"],
    "quantity": [2, 1, 3, 1, 2, 1, 1, 2, 1]
})

# Validate foreign key constraint
def check_foreign_key(df):
    """Check if all order_ids in order_details exist in orders table"""
    valid_order_ids = orders.select("order_id")
    # Semi join returns only rows with matching keys
    return df.join(valid_order_ids, on="order_id", how="semi").height == df.height

(
    pb.Validate(data=order_details, tbl_name="order_details")
    .specially(
        expr=check_foreign_key,
        brief="All order_ids must exist in orders table"
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+c3BlY2lhbGx5PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InNwZWNpYWxseSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMjA2ODk3KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8ZyBpZD0ic3RhciIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOC41MDAwMDAsIDguNTAwMDAwKSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yNSwwIEMyNC41ODc0NDg0LDAgMjQuMjE3NDUxNywwLjI1NDAwMjg0NyAyNC4wNjgzNTksMC42Mzg2NzE5IEwxNy45MDIzNDQsMTYuNTM1MTU2IEwwLjk0OTIxODc1LDE3LjQwMDM5MSBDMC41MzYxMjQ0MDksMTcuNDIxMzAxMyAwLjE3ODUyNDU4LDE3LjY5NDM1MjMgMC4wNDk1NDQ2Mzk1LDE4LjA4NzM1MTUgQy0wLjA3OTQzNTMwMTIsMTguNDgwMzUwNyAwLjA0NjgyMDQ1MiwxOC45MTIyMDAyIDAuMzY3MTg3NSwxOS4xNzM4MjggTDEzLjU2ODM1OSwyOS45NjY3OTcgTDkuMjMyNDIxOSw0Ni4zNDM3NSBDOS4xMjY0Njk2Myw0Ni43NDI4MDA5IDkuMjc2NjMwNTgsNDcuMTY1OTQzMyA5LjYxMDQyNjk4LDQ3LjQwODk0MDIgQzkuOTQ0MjIzMzgsNDcuNjUxOTM3IDEwLjM5MzAzNDUsNDcuNjY0ODM0IDEwLjc0MDIzNCw0Ny40NDE0MDYgTDI1LDM4LjI4OTA2MiBMMzkuMjU5NzY2LDQ3LjQ0MTQwNiBDMzkuNjA2OTY1NSw0Ny42NjQ4MzM5IDQwLjA1NTc3NjYsNDcuNjUxOTM2OSA0MC4zODk1NzI5LDQ3LjQwODk0MDEgQzQwLjcyMzM2OTMsNDcuMTY1OTQzMiA0MC44NzM1MzAyLDQ2Ljc0MjgwMDkgNDAuNzY3NTc4LDQ2LjM0Mzc1IEwzNi40MzE2NDEsMjkuOTY2Nzk3IEw0OS42MzI4MTIsMTkuMTczODI4IEM0OS45NTMxNzksMTguOTEyMjAwMiA1MC4wNzk0MzQ4LDE4LjQ4MDM1MDcgNDkuOTUwNDU0OSwxOC4wODczNTE2IEM0OS44MjE0NzUsMTcuNjk0MzUyNCA0OS40NjM4NzUzLDE3LjQyMTMwMTQgNDkuMDUwNzgxLDE3LjQwMDM5MSBMMzIuMDk3NjU2LDE2LjUzNTE1NiBMMjUuOTMxNjQxLDAuNjM4NjcxOSBDMjUuNzgyNTQ4MywwLjI1NDAwMjg0NyAyNS40MTI1NTE2LDAgMjUsMCBaIE0yNSwzLjc2MzY3MTkgTDMwLjQ2Njc5NywxNy44NjEzMjggQzMwLjYwOTY4OSwxOC4yMjkxNDE2IDMwLjk1NTQ5NjIsMTguNDc4NTUxNSAzMS4zNDk2MDksMTguNDk4MDQ3IEw0Ni4zNTkzNzUsMTkuMjY1NjI1IEwzNC42Njc5NjksMjguODI2MTcyIEMzNC4zNjQ2MDU0LDI5LjA3NDIxMTQgMzQuMjM0MDQ5MywyOS40NzY1Njc5IDM0LjMzMzk4NCwyOS44NTU0NjkgTDM4LjE3NTc4MSw0NC4zNjkxNDEgTDI1LjU0MTAxNiwzNi4yNTc4MTIgQzI1LjIxMTQ3ODksMzYuMDQ1ODUzNiAyNC43ODg1MjExLDM2LjA0NTg1MzYgMjQuNDU4OTg0LDM2LjI1NzgxMiBMMTEuODI0MjE5LDQ0LjM2OTE0MSBMMTUuNjY2MDE2LDI5Ljg1NTQ2OSBDMTUuNzY1OTUwNywyOS40NzY1Njc5IDE1LjYzNTM5NDYsMjkuMDc0MjExNCAxNS4zMzIwMzEsMjguODI2MTcyIEwzLjY0MDYyNSwxOS4yNjU2MjUgTDE4LjY1MDM5MSwxOC40OTgwNDcgQzE5LjA0NDUwMzgsMTguNDc4NTUxNSAxOS4zOTAzMTEsMTguMjI5MTQxNiAxOS41MzMyMDMsMTcuODYxMzI4IEwyNSwzLjc2MzY3MTkgWiIgaWQ9IlNoYXBlIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

specially()


<p>All order_ids must exist in orders table</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">EXPR</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


This validation ensures referential integrity by confirming that every `order_id` in the `order_details` table has a corresponding record in the `orders` table. The use of a semi-join makes this check efficient, as it only verifies the existence of matching keys without returning full joined data.


### Conditional Existence Checks

Implement "if X then Y must exist" logic across datasets:


``` python
def check_completed_orders_have_details(df):
    """Completed orders must have at least one detail record"""
    completed_orders = df.filter(pl.col("status") == "completed")
    order_ids_with_details = order_details.select("order_id").unique()

    # Check each completed order has matching details
    return completed_orders.join(
        order_ids_with_details,
        on="order_id",
        how="left"
    ).with_columns(
        pl.col("order_id").is_not_null().alias("has_details")
    ).select("has_details")

(
    pb.Validate(data=orders, tbl_name="orders")
    .specially(
        expr=check_completed_orders_have_details,
        brief="Completed orders must have detail records"
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
uMjU5NzY2LDQ3LjQ0MTQwNiBDMzkuNjA2OTY1NSw0Ny42NjQ4MzM5IDQwLjA1NTc3NjYsNDcuNjUxOTM2OSA0MC4zODk1NzI5LDQ3LjQwODk0MDEgQzQwLjcyMzM2OTMsNDcuMTY1OTQzMiA0MC44NzM1MzAyLDQ2Ljc0MjgwMDkgNDAuNzY3NTc4LDQ2LjM0Mzc1IEwzNi40MzE2NDEsMjkuOTY2Nzk3IEw0OS42MzI4MTIsMTkuMTczODI4IEM0OS45NTMxNzksMTguOTEyMjAwMiA1MC4wNzk0MzQ4LDE4LjQ4MDM1MDcgNDkuOTUwNDU0OSwxOC4wODczNTE2IEM0OS44MjE0NzUsMTcuNjk0MzUyNCA0OS40NjM4NzUzLDE3LjQyMTMwMTQgNDkuMDUwNzgxLDE3LjQwMDM5MSBMMzIuMDk3NjU2LDE2LjUzNTE1NiBMMjUuOTMxNjQxLDAuNjM4NjcxOSBDMjUuNzgyNTQ4MywwLjI1NDAwMjg0NyAyNS40MTI1NTE2LDAgMjUsMCBaIE0yNSwzLjc2MzY3MTkgTDMwLjQ2Njc5NywxNy44NjEzMjggQzMwLjYwOTY4OSwxOC4yMjkxNDE2IDMwLjk1NTQ5NjIsMTguNDc4NTUxNSAzMS4zNDk2MDksMTguNDk4MDQ3IEw0Ni4zNTkzNzUsMTkuMjY1NjI1IEwzNC42Njc5NjksMjguODI2MTcyIEMzNC4zNjQ2MDU0LDI5LjA3NDIxMTQgMzQuMjM0MDQ5MywyOS40NzY1Njc5IDM0LjMzMzk4NCwyOS44NTU0NjkgTDM4LjE3NTc4MSw0NC4zNjkxNDEgTDI1LjU0MTAxNiwzNi4yNTc4MTIgQzI1LjIxMTQ3ODksMzYuMDQ1ODUzNiAyNC43ODg1MjExLDM2LjA0NTg1MzYgMjQuNDU4OTg0LDM2LjI1NzgxMiBMMTEuODI0MjE5LDQ0LjM2OTE0MSBMMTUuNjY2MDE2LDI5Ljg1NTQ2OSBDMTUuNzY1OTUwNywyOS40NzY1Njc5IDE1LjYzNTM5NDYsMjkuMDc0MjExNCAxNS4zMzIwMzEsMjguODI2MTcyIEwzLjY0MDYyNSwxOS4yNjU2MjUgTDE4LjY1MDM5MSwxOC40OTgwNDcgQzE5LjA0NDUwMzgsMTguNDc4NTUxNSAxOS4zOTAzMTEsMTguMjI5MTQxNiAxOS41MzMyMDMsMTcuODYxMzI4IEwyNSwzLjc2MzY3MTkgWiIgaWQ9IlNoYXBlIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

specially()


<p>Completed orders must have detail records</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">EXPR</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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


This validation implements conditional business logic: only orders with a `completed` status are required to have detail records. This pattern is common in real-world scenarios where certain records trigger mandatory relationships while others don't. The validation returns a boolean for each completed order, allowing you to see exactly which records pass or fail.


### Cardinality Constraints

Validate that relationships between datasets follow specific cardinality rules:


``` python
def check_quantity_ratio(df):
    """Each order should have exactly 3x quantity units in details"""
    order_counts = orders.group_by("order_id").agg(pl.lit(1).alias("order_count"))

    detail_quantities = order_details.group_by("order_id").agg(
        pl.col("quantity").sum().alias("total_quantity")
    )

    joined = order_counts.join(detail_quantities, on="order_id", how="left")

    return joined.with_columns(
        (pl.col("total_quantity") == pl.col("order_count") * 3).alias("valid_ratio")
    ).select("valid_ratio")


(
    pb.Validate(data=orders, tbl_name="orders")
    .specially(
        expr=check_quantity_ratio,
        brief="Each order should have 3x quantity units in details",
        thresholds=(0.4, 0.7),  # Allow some flexibility
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; background-color: #AAAAAA; color: transparent; font-size: 0px">#AAAAAA</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
OTQzMiA0MC44NzM1MzAyLDQ2Ljc0MjgwMDkgNDAuNzY3NTc4LDQ2LjM0Mzc1IEwzNi40MzE2NDEsMjkuOTY2Nzk3IEw0OS42MzI4MTIsMTkuMTczODI4IEM0OS45NTMxNzksMTguOTEyMjAwMiA1MC4wNzk0MzQ4LDE4LjQ4MDM1MDcgNDkuOTUwNDU0OSwxOC4wODczNTE2IEM0OS44MjE0NzUsMTcuNjk0MzUyNCA0OS40NjM4NzUzLDE3LjQyMTMwMTQgNDkuMDUwNzgxLDE3LjQwMDM5MSBMMzIuMDk3NjU2LDE2LjUzNTE1NiBMMjUuOTMxNjQxLDAuNjM4NjcxOSBDMjUuNzgyNTQ4MywwLjI1NDAwMjg0NyAyNS40MTI1NTE2LDAgMjUsMCBaIE0yNSwzLjc2MzY3MTkgTDMwLjQ2Njc5NywxNy44NjEzMjggQzMwLjYwOTY4OSwxOC4yMjkxNDE2IDMwLjk1NTQ5NjIsMTguNDc4NTUxNSAzMS4zNDk2MDksMTguNDk4MDQ3IEw0Ni4zNTkzNzUsMTkuMjY1NjI1IEwzNC42Njc5NjksMjguODI2MTcyIEMzNC4zNjQ2MDU0LDI5LjA3NDIxMTQgMzQuMjM0MDQ5MywyOS40NzY1Njc5IDM0LjMzMzk4NCwyOS44NTU0NjkgTDM4LjE3NTc4MSw0NC4zNjkxNDEgTDI1LjU0MTAxNiwzNi4yNTc4MTIgQzI1LjIxMTQ3ODksMzYuMDQ1ODUzNiAyNC43ODg1MjExLDM2LjA0NTg1MzYgMjQuNDU4OTg0LDM2LjI1NzgxMiBMMTEuODI0MjE5LDQ0LjM2OTE0MSBMMTUuNjY2MDE2LDI5Ljg1NTQ2OSBDMTUuNzY1OTUwNywyOS40NzY1Njc5IDE1LjYzNTM5NDYsMjkuMDc0MjExNCAxNS4zMzIwMzEsMjguODI2MTcyIEwzLjY0MDYyNSwxOS4yNjU2MjUgTDE4LjY1MDM5MSwxOC40OTgwNDcgQzE5LjA0NDUwMzgsMTguNDc4NTUxNSAxOS4zOTAzMTEsMTguMjI5MTQxNiAxOS41MzMyMDMsMTcuODYxMzI4IEwyNSwzLjc2MzY3MTkgWiIgaWQ9IlNoYXBlIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

specially()


<p>Each order should have 3x quantity units in details</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">EXPR</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.40</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">3<br />
0.60</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><span style="color: #AAAAAA;">●</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC"><span style="color: #EBBC14;">○</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(local_thresholds)</span> Step-specific thresholds set with <span style="font-family: monospace;"><span style="color: #AAAAAA; font-weight: bold;">W</span>:0.4|<span style="color: #EBBC14; font-weight: bold;">E</span>:0.7</span>.</p></td>
</tr>
</tfoot>

</table>


Cardinality constraints like this validate that the relationship between datasets follows expected patterns. In this example, we expect each order to have a specific quantity ratio in the detail records. Note the use of `thresholds=` to allow some flexibility (not every order needs to meet this requirement perfectly, but too many violations would indicate a data quality issue).


### Composite Keys with Business Logic

Validate complex relationships involving multiple columns and conditional logic:


``` python
# More complex scenario with composite keys
employees = pl.DataFrame({
    "dept_id": ["D1", "D1", "D2", "D2", "D3"],
    "emp_id": ["E001", "E002", "E003", "E004", "E005"],
    "emp_name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "is_manager": [True, False, True, False, False]
})

projects = pl.DataFrame({
    "project_id": ["P1", "P2", "P3", "P4"],
    "dept_id": ["D1", "D2", "D1", "D3"],
    "manager_emp_id": ["E001", "E003", "E001", "E005"]
})

def check_project_manager_validity(df):
    """Project managers must be valid managers in their department"""
    validation_result = df.join(
        employees,
        left_on=["dept_id", "manager_emp_id"],
        right_on=["dept_id", "emp_id"],
        how="left"
    ).with_columns(
        # Manager must exist in dept AND have manager status
        ((pl.col("emp_name").is_not_null()) & (pl.col("is_manager") == True)).alias("valid_manager")
    ).select("valid_manager")

    return validation_result

(
    pb.Validate(data=projects, tbl_name="projects")
    .specially(
        expr=check_project_manager_validity,
        brief="Project managers must be valid managers in their department"
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
SIjRkZGRkZGIiAvPgogICAgICAgICAgICA8ZyBpZD0ic3RhciIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOC41MDAwMDAsIDguNTAwMDAwKSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yNSwwIEMyNC41ODc0NDg0LDAgMjQuMjE3NDUxNywwLjI1NDAwMjg0NyAyNC4wNjgzNTksMC42Mzg2NzE5IEwxNy45MDIzNDQsMTYuNTM1MTU2IEwwLjk0OTIxODc1LDE3LjQwMDM5MSBDMC41MzYxMjQ0MDksMTcuNDIxMzAxMyAwLjE3ODUyNDU4LDE3LjY5NDM1MjMgMC4wNDk1NDQ2Mzk1LDE4LjA4NzM1MTUgQy0wLjA3OTQzNTMwMTIsMTguNDgwMzUwNyAwLjA0NjgyMDQ1MiwxOC45MTIyMDAyIDAuMzY3MTg3NSwxOS4xNzM4MjggTDEzLjU2ODM1OSwyOS45NjY3OTcgTDkuMjMyNDIxOSw0Ni4zNDM3NSBDOS4xMjY0Njk2Myw0Ni43NDI4MDA5IDkuMjc2NjMwNTgsNDcuMTY1OTQzMyA5LjYxMDQyNjk4LDQ3LjQwODk0MDIgQzkuOTQ0MjIzMzgsNDcuNjUxOTM3IDEwLjM5MzAzNDUsNDcuNjY0ODM0IDEwLjc0MDIzNCw0Ny40NDE0MDYgTDI1LDM4LjI4OTA2MiBMMzkuMjU5NzY2LDQ3LjQ0MTQwNiBDMzkuNjA2OTY1NSw0Ny42NjQ4MzM5IDQwLjA1NTc3NjYsNDcuNjUxOTM2OSA0MC4zODk1NzI5LDQ3LjQwODk0MDEgQzQwLjcyMzM2OTMsNDcuMTY1OTQzMiA0MC44NzM1MzAyLDQ2Ljc0MjgwMDkgNDAuNzY3NTc4LDQ2LjM0Mzc1IEwzNi40MzE2NDEsMjkuOTY2Nzk3IEw0OS42MzI4MTIsMTkuMTczODI4IEM0OS45NTMxNzksMTguOTEyMjAwMiA1MC4wNzk0MzQ4LDE4LjQ4MDM1MDcgNDkuOTUwNDU0OSwxOC4wODczNTE2IEM0OS44MjE0NzUsMTcuNjk0MzUyNCA0OS40NjM4NzUzLDE3LjQyMTMwMTQgNDkuMDUwNzgxLDE3LjQwMDM5MSBMMzIuMDk3NjU2LDE2LjUzNTE1NiBMMjUuOTMxNjQxLDAuNjM4NjcxOSBDMjUuNzgyNTQ4MywwLjI1NDAwMjg0NyAyNS40MTI1NTE2LDAgMjUsMCBaIE0yNSwzLjc2MzY3MTkgTDMwLjQ2Njc5NywxNy44NjEzMjggQzMwLjYwOTY4OSwxOC4yMjkxNDE2IDMwLjk1NTQ5NjIsMTguNDc4NTUxNSAzMS4zNDk2MDksMTguNDk4MDQ3IEw0Ni4zNTkzNzUsMTkuMjY1NjI1IEwzNC42Njc5NjksMjguODI2MTcyIEMzNC4zNjQ2MDU0LDI5LjA3NDIxMTQgMzQuMjM0MDQ5MywyOS40NzY1Njc5IDM0LjMzMzk4NCwyOS44NTU0NjkgTDM4LjE3NTc4MSw0NC4zNjkxNDEgTDI1LjU0MTAxNiwzNi4yNTc4MTIgQzI1LjIxMTQ3ODksMzYuMDQ1ODUzNiAyNC43ODg1MjExLDM2LjA0NTg1MzYgMjQuNDU4OTg0LDM2LjI1NzgxMiBMMTEuODI0MjE5LDQ0LjM2OTE0MSBMMTUuNjY2MDE2LDI5Ljg1NTQ2OSBDMTUuNzY1OTUwNywyOS40NzY1Njc5IDE1LjYzNTM5NDYsMjkuMDc0MjExNCAxNS4zMzIwMzEsMjguODI2MTcyIEwzLjY0MDYyNSwxOS4yNjU2MjUgTDE4LjY1MDM5MSwxOC40OTgwNDcgQzE5LjA0NDUwMzgsMTguNDc4NTUxNSAxOS4zOTAzMTEsMTguMjI5MTQxNiAxOS41MzMyMDMsMTcuODYxMzI4IEwyNSwzLjc2MzY3MTkgWiIgaWQ9IlNoYXBlIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

specially()


<p>Project managers must be valid managers in their department</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">EXPR</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">4</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">3<br />
0.75</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
0.25</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
</tbody>
</table>


This example demonstrates validation using composite keys (both `dept_id` and `emp_id`) combined with conditional business logic (checking the `is_manager` flag). Such validations are common in enterprise systems where relationships must satisfy multiple constraints simultaneously. The validation reveals that one project (`P4`) fails because employee `E005` is not a manager, even though they exist in the same department.


## Reusable Validation Factories

For validations you'll use repeatedly, create factory functions that generate customized validators:


``` python
def make_foreign_key_validator(reference_table, key_columns):
    """Factory function to create reusable foreign key validators"""
    def validate_fk(df):
        if isinstance(key_columns, str):
            keys = [key_columns]
        else:
            keys = key_columns

        ref_keys = reference_table.select(keys).unique()
        matched = df.join(ref_keys, on=keys, how="semi")
        return matched.height == df.height

    return validate_fk

# Use the factory across multiple validations
(
    pb.Validate(data=order_details, tbl_name="order_details")
    .specially(
        expr=make_foreign_key_validator(orders, "order_id"),
        brief="FK constraint: order_id → orders"
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+c3BlY2lhbGx5PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InNwZWNpYWxseSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMjA2ODk3KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8ZyBpZD0ic3RhciIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOC41MDAwMDAsIDguNTAwMDAwKSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0yNSwwIEMyNC41ODc0NDg0LDAgMjQuMjE3NDUxNywwLjI1NDAwMjg0NyAyNC4wNjgzNTksMC42Mzg2NzE5IEwxNy45MDIzNDQsMTYuNTM1MTU2IEwwLjk0OTIxODc1LDE3LjQwMDM5MSBDMC41MzYxMjQ0MDksMTcuNDIxMzAxMyAwLjE3ODUyNDU4LDE3LjY5NDM1MjMgMC4wNDk1NDQ2Mzk1LDE4LjA4NzM1MTUgQy0wLjA3OTQzNTMwMTIsMTguNDgwMzUwNyAwLjA0NjgyMDQ1MiwxOC45MTIyMDAyIDAuMzY3MTg3NSwxOS4xNzM4MjggTDEzLjU2ODM1OSwyOS45NjY3OTcgTDkuMjMyNDIxOSw0Ni4zNDM3NSBDOS4xMjY0Njk2Myw0Ni43NDI4MDA5IDkuMjc2NjMwNTgsNDcuMTY1OTQzMyA5LjYxMDQyNjk4LDQ3LjQwODk0MDIgQzkuOTQ0MjIzMzgsNDcuNjUxOTM3IDEwLjM5MzAzNDUsNDcuNjY0ODM0IDEwLjc0MDIzNCw0Ny40NDE0MDYgTDI1LDM4LjI4OTA2MiBMMzkuMjU5NzY2LDQ3LjQ0MTQwNiBDMzkuNjA2OTY1NSw0Ny42NjQ4MzM5IDQwLjA1NTc3NjYsNDcuNjUxOTM2OSA0MC4zODk1NzI5LDQ3LjQwODk0MDEgQzQwLjcyMzM2OTMsNDcuMTY1OTQzMiA0MC44NzM1MzAyLDQ2Ljc0MjgwMDkgNDAuNzY3NTc4LDQ2LjM0Mzc1IEwzNi40MzE2NDEsMjkuOTY2Nzk3IEw0OS42MzI4MTIsMTkuMTczODI4IEM0OS45NTMxNzksMTguOTEyMjAwMiA1MC4wNzk0MzQ4LDE4LjQ4MDM1MDcgNDkuOTUwNDU0OSwxOC4wODczNTE2IEM0OS44MjE0NzUsMTcuNjk0MzUyNCA0OS40NjM4NzUzLDE3LjQyMTMwMTQgNDkuMDUwNzgxLDE3LjQwMDM5MSBMMzIuMDk3NjU2LDE2LjUzNTE1NiBMMjUuOTMxNjQxLDAuNjM4NjcxOSBDMjUuNzgyNTQ4MywwLjI1NDAwMjg0NyAyNS40MTI1NTE2LDAgMjUsMCBaIE0yNSwzLjc2MzY3MTkgTDMwLjQ2Njc5NywxNy44NjEzMjggQzMwLjYwOTY4OSwxOC4yMjkxNDE2IDMwLjk1NTQ5NjIsMTguNDc4NTUxNSAzMS4zNDk2MDksMTguNDk4MDQ3IEw0Ni4zNTkzNzUsMTkuMjY1NjI1IEwzNC42Njc5NjksMjguODI2MTcyIEMzNC4zNjQ2MDU0LDI5LjA3NDIxMTQgMzQuMjM0MDQ5MywyOS40NzY1Njc5IDM0LjMzMzk4NCwyOS44NTU0NjkgTDM4LjE3NTc4MSw0NC4zNjkxNDEgTDI1LjU0MTAxNiwzNi4yNTc4MTIgQzI1LjIxMTQ3ODksMzYuMDQ1ODUzNiAyNC43ODg1MjExLDM2LjA0NTg1MzYgMjQuNDU4OTg0LDM2LjI1NzgxMiBMMTEuODI0MjE5LDQ0LjM2OTE0MSBMMTUuNjY2MDE2LDI5Ljg1NTQ2OSBDMTUuNzY1OTUwNywyOS40NzY1Njc5IDE1LjYzNTM5NDYsMjkuMDc0MjExNCAxNS4zMzIwMzEsMjguODI2MTcyIEwzLjY0MDYyNSwxOS4yNjU2MjUgTDE4LjY1MDM5MSwxOC40OTgwNDcgQzE5LjA0NDUwMzgsMTguNDc4NTUxNSAxOS4zOTAzMTEsMTguMjI5MTQxNiAxOS41MzMyMDMsMTcuODYxMzI4IEwyNSwzLjc2MzY3MTkgWiIgaWQ9IlNoYXBlIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

specially()


<p>FK constraint: order_id → orders</p>
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">EXPR</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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
</tbody>
</table>


Factory functions like `make_foreign_key_validator()` make your validation code more maintainable and reusable. Once defined, you can use the same factory to validate different foreign key relationships across your entire data pipeline, ensuring consistency in how these constraints are checked. This pattern is particularly valuable in production environments where you validate multiple related tables.


## When to Use [specially()](../../reference/Validate.specially.md#pointblank.Validate.specially)

The [specially()](../../reference/Validate.specially.md#pointblank.Validate.specially) method is ideal for:

- cross-dataset validations: foreign keys, referential integrity, conditional existence
- complex business rules: multi-column checks, conditional logic, domain-specific constraints
- custom statistical tests: validations requiring calculations not covered by built-in methods
- SQL-style checks: converting complex SQL queries into validation steps
- prototype validations: testing new validation patterns before implementing them as dedicated methods

By combining [specially()](../../reference/Validate.specially.md#pointblank.Validate.specially) with Pointblank's built-in validation methods, you can create comprehensive data quality checks that address both standard and highly specific validation requirements.


# Conclusion

In this article, we've explored the various types of validation methods that Pointblank offers for ensuring data quality. These methods provide a framework for validating column values, checking row properties, verifying table structures, using AI for complex semantic validations, and validating aggregate statistics across columns. By combining these validation methods into comprehensive plans, you can systematically test your data against business rules and quality expectations. And this all helps to ensure your data remains reliable and trustworthy.
