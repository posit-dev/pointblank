# Preprocessing

While the available validation methods can do a lot for you, there's likewise a lot of things you *can't* easily do with them. What if you wanted to validate that

- string lengths in a column are less than 10 characters?
- the median of values in a column is less than the median of values in another column?
- there are at least three instances of every categorical value in a column?

These constitute more sophisticated validation requirements, yet such examinations are quite prevalent in practice. Rather than expanding our library to encompass every conceivable validation scenario (a pursuit that would yield an unwieldy and potentially infinite collection) we instead employ a more elegant approach. By transforming the table under examination through judicious preprocessing and exposing key metrics, we may subsequently employ the existing collection of validation methods. This compositional strategy affords us considerable analytical power while maintaining conceptual clarity and implementation parsimony.

Central to this approach is the idea of composability. Pointblank makes it easy to safely transform the target table for a given validation via the `pre=` argument. Any computed columns are available for the (short) lifetime of the validation step during interrogation. This composability means:

1.  we can validate on different forms of the initial dataset (e.g., validating on aggregate forms, validating on calculated columns, etc.)
2.  there's no need to start an entirely new validation process for each transformed version of the data (i.e., one tabular report could be produced instead of several)

This compositional paradigm allows us to use data transformation effectively within our validation workflows, maintaining both flexibility and clarity in our data quality assessments.


# Transforming Data with Lambda Functions

Now, through examples, let's look at the process of performing the validations mentioned above. We'll use the `small_table` dataset for all of the examples. Here it is in its entirety:


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


In getting to grips with the basics, we'll try to validate that string lengths in the `b` column are less than 10 characters. We can't directly use the <a href="../../reference/Validate.col_vals_lt.html#pointblank.Validate.col_vals_lt" class="gdls-link"><code>Validate.col_vals_lt()</code></a> validation method with that column because it is meant to be used with a column of numeric values. Let's just give that method what it needs and create a column with string lengths!

The target table is a Polars DataFrame so we'll provide a function that uses the Polars API to add in that numeric column:


``` python
import polars as pl

# Define a preprocessing function that gets string lengths from column `b`
def add_string_length_column(df):
    return df.with_columns(string_lengths=pl.col("b").str.len_chars())

(
    pb.Validate(
        data=pb.load_dataset(dataset="small_table", tbl_type="polars"),
        tbl_name="small_table",
        label="String lengths"
    )
    .col_vals_lt(

        # The generated column, via `pre=` (see below) ---
        columns="string_lengths",

        # The string length value to be less than ---
        value=10,

        # The preprocessing function that modifies the table ---
        pre=add_string_length_column
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">String lengths</span>

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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19sdDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19sdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNTEsMTAgTDEzLDEwIEMxMS4zNDc2NTYsMTAgMTAsMTEuMzQ3NjU2IDEwLDEzIEwxMCw1MSBDMTAsNTIuNjUyMzQ0IDExLjM0NzY1Niw1NCAxMyw1NCBMNTEsNTQgQzUyLjY1MjM0NCw1NCA1NCw1Mi42NTIzNDQgNTQsNTEgTDU0LDEzIEM1NCwxMS4zNDc2NTYgNTIuNjUyMzQ0LDEwIDUxLDEwIFogTTM4LjcwNzAzMSw0NS4yOTI5NjkgTDM3LjI5Mjk2OSw0Ni43MDcwMzEgTDIyLjU4NTkzOCwzMiBMMzcuMjkyOTY5LDE3LjI5Mjk2OSBMMzguNzA3MDMxLDE4LjcwNzAzMSBMMjUuNDE0MDYzLDMyIEwzOC43MDcwMzEsNDUuMjkyOTY5IFoiIGlkPSJsZXNzX3RoYW4iIGZpbGw9IiMwMDAwMDAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_lt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"><span style="text-decoration: underline; text-decoration-color: #9A7CB4; text-decoration-thickness: 1px; text-underline-offset: 3px;">string_lengths</span></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">10</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJtb2RpZmllZCIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9Im1vZGlmaWVkIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC41NzAxNDcpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgeD0iMC4xMjUxMzI1MDYiIHk9IjAiIHdpZHRoPSIyMy43NDk3MzUiIGhlaWdodD0iMjMuNzg5NDczNyIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzAwMDAwMCIgeD0iMTUuMjIxNTYyNiIgeT0iMTQuODAyNTg4NSIgd2lkdGg9IjYuMzQwODk4MjEiIGhlaWdodD0iNi4zNDA4OTgyMSIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzlBN0NCNCIgeD0iMTYuMTI2NDMwOSIgeT0iMTUuNzA3NDU2OCIgd2lkdGg9IjQuNTMxMTYxNTgiIGhlaWdodD0iNC41MzExNjE1OCIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
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
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(pre_applied)</span> Precondition applied: table dimensions <span style="font-family: monospace;">[13 rows, 8 columns]</span> → <span style="font-family: monospace;">[13 rows, <strong>9</strong> columns]</span>.</p>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(syn_target_col)</span> Synthetic target column <code style="font-family: "IBM Plex Mono", monospace;">string_lengths</code> created via preprocessing.</p></td>
</tr>
</tfoot>

</table>


The validation was successfully constructed and we can see from the validation report table that all strings in `b` had lengths less than 10 characters. Also note that the icon under the `TBL` column is no longer a rightward-facing arrow, but one that is indicative of a transformation taking place.

Let's examine the transformation approach more closely. In the previous example, we're not directly testing the `b` column itself. Instead, we're validating the `string_lengths` column that was generated by the lambda function provided to `pre=`. The Polars API's `with_columns()` method does the heavy lifting, creating numerical values that represent each string's length in the original column.

That transformation occurs only during interrogation and only for that validation step. Any prior or subsequent steps would normally use the as-provided `small_table`. Having the possibility for data transformation being isolated at the step level means that you don't have to generate separate validation plans for each form of the data, you're free to fluidly transform the target table as necessary for perform validations on different representations of the data.


# Using Custom Functions for Preprocessing

While lambda functions work well for simple transformations, custom named functions can make your validation code more organized and reusable, especially for complex preprocessing logic. Let's implement the same string length validation using a dedicated function:


``` python
def add_string_lengths(df):
    # This generates string length from a column `b`; the new column with
    # the values is called `string_lengths` (will be placed as the last column)
    return df.with_columns(string_lengths=pl.col("b").str.len_chars())

(
    pb.Validate(
        data=pb.load_dataset(dataset="small_table", tbl_type="polars"),
        tbl_name="small_table",
        label="String lengths for column `b`."
    )
    .col_vals_lt(

        # Use of a column selector function to select the last column ---
        columns=pb.last_n(1),

        # The string length to be less than ---
        value=10,

        # Custom function for generating string lengths in a new column ---
        pre=add_string_lengths
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">String lengths for column `b`.</span>

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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19sdDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19sdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNTEsMTAgTDEzLDEwIEMxMS4zNDc2NTYsMTAgMTAsMTEuMzQ3NjU2IDEwLDEzIEwxMCw1MSBDMTAsNTIuNjUyMzQ0IDExLjM0NzY1Niw1NCAxMyw1NCBMNTEsNTQgQzUyLjY1MjM0NCw1NCA1NCw1Mi42NTIzNDQgNTQsNTEgTDU0LDEzIEM1NCwxMS4zNDc2NTYgNTIuNjUyMzQ0LDEwIDUxLDEwIFogTTM4LjcwNzAzMSw0NS4yOTI5NjkgTDM3LjI5Mjk2OSw0Ni43MDcwMzEgTDIyLjU4NTkzOCwzMiBMMzcuMjkyOTY5LDE3LjI5Mjk2OSBMMzguNzA3MDMxLDE4LjcwNzAzMSBMMjUuNDE0MDYzLDMyIEwzOC43MDcwMzEsNDUuMjkyOTY5IFoiIGlkPSJsZXNzX3RoYW4iIGZpbGw9IiMwMDAwMDAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_lt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"><span style="text-decoration: underline; text-decoration-color: #9A7CB4; text-decoration-thickness: 1px; text-underline-offset: 3px;">string_lengths</span></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">10</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJtb2RpZmllZCIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9Im1vZGlmaWVkIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC41NzAxNDcpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgeD0iMC4xMjUxMzI1MDYiIHk9IjAiIHdpZHRoPSIyMy43NDk3MzUiIGhlaWdodD0iMjMuNzg5NDczNyIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzAwMDAwMCIgeD0iMTUuMjIxNTYyNiIgeT0iMTQuODAyNTg4NSIgd2lkdGg9IjYuMzQwODk4MjEiIGhlaWdodD0iNi4zNDA4OTgyMSIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzlBN0NCNCIgeD0iMTYuMTI2NDMwOSIgeT0iMTUuNzA3NDU2OCIgd2lkdGg9IjQuNTMxMTYxNTgiIGhlaWdodD0iNC41MzExNjE1OCIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
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
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(pre_applied)</span> Precondition applied: table dimensions <span style="font-family: monospace;">[13 rows, 8 columns]</span> → <span style="font-family: monospace;">[13 rows, <strong>9</strong> columns]</span>.</p>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(syn_target_col)</span> Synthetic target column <code style="font-family: "IBM Plex Mono", monospace;">string_lengths</code> created via preprocessing.</p></td>
</tr>
</tfoot>

</table>


The column-generating logic was placed in the `add_string_lengths()` function, which is then passed to `pre=`. Notice we're using `pb.last_n(1)` in the `columns` parameter. This is a convenient column selector that targets the last column in the DataFrame, which in our case is the newly created `string_lengths` column. This saves us from having to explicitly write out the column name, making our code more adaptable if column names change. Despite not specifying the name directly, you'll still see the actual column name (`string_lengths`) displayed in the validation report.


# Creating Parameterized Preprocessing Functions

So far we've used simple functions and lambdas, but sometimes you may want to create more flexible preprocessing functions that can be configured with parameters. Let's create a reusable function that can calculate string lengths for any column:


``` python
def string_length_calculator(column_name):
    """Returns a preprocessing function that calculates string lengths for the specified column."""
    def preprocessor(df):
        return df.with_columns(string_lengths=pl.col(column_name).str.len_chars())
    return preprocessor

# Validate string lengths in column b
(
    pb.Validate(
        data=pb.load_dataset(dataset="small_table", tbl_type="polars"),
        tbl_name="small_table",
        label="String lengths for column `b`."
    )
    .col_vals_lt(
        columns=pb.last_n(1),
        value=10,
        pre=string_length_calculator(column_name="b")
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">String lengths for column `b`.</span>

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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19sdDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19sdCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNTEsMTAgTDEzLDEwIEMxMS4zNDc2NTYsMTAgMTAsMTEuMzQ3NjU2IDEwLDEzIEwxMCw1MSBDMTAsNTIuNjUyMzQ0IDExLjM0NzY1Niw1NCAxMyw1NCBMNTEsNTQgQzUyLjY1MjM0NCw1NCA1NCw1Mi42NTIzNDQgNTQsNTEgTDU0LDEzIEM1NCwxMS4zNDc2NTYgNTIuNjUyMzQ0LDEwIDUxLDEwIFogTTM4LjcwNzAzMSw0NS4yOTI5NjkgTDM3LjI5Mjk2OSw0Ni43MDcwMzEgTDIyLjU4NTkzOCwzMiBMMzcuMjkyOTY5LDE3LjI5Mjk2OSBMMzguNzA3MDMxLDE4LjcwNzAzMSBMMjUuNDE0MDYzLDMyIEwzOC43MDcwMzEsNDUuMjkyOTY5IFoiIGlkPSJsZXNzX3RoYW4iIGZpbGw9IiMwMDAwMDAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_lt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"><span style="text-decoration: underline; text-decoration-color: #9A7CB4; text-decoration-thickness: 1px; text-underline-offset: 3px;">string_lengths</span></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">10</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJtb2RpZmllZCIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9Im1vZGlmaWVkIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC41NzAxNDcpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgeD0iMC4xMjUxMzI1MDYiIHk9IjAiIHdpZHRoPSIyMy43NDk3MzUiIGhlaWdodD0iMjMuNzg5NDczNyIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzAwMDAwMCIgeD0iMTUuMjIxNTYyNiIgeT0iMTQuODAyNTg4NSIgd2lkdGg9IjYuMzQwODk4MjEiIGhlaWdodD0iNi4zNDA4OTgyMSIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzlBN0NCNCIgeD0iMTYuMTI2NDMwOSIgeT0iMTUuNzA3NDU2OCIgd2lkdGg9IjQuNTMxMTYxNTgiIGhlaWdodD0iNC41MzExNjE1OCIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
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
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(pre_applied)</span> Precondition applied: table dimensions <span style="font-family: monospace;">[13 rows, 8 columns]</span> → <span style="font-family: monospace;">[13 rows, <strong>9</strong> columns]</span>.</p>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(syn_target_col)</span> Synthetic target column <code style="font-family: "IBM Plex Mono", monospace;">string_lengths</code> created via preprocessing.</p></td>
</tr>
</tfoot>

</table>


This pattern is called a *function factory*, which is a function that creates and returns another function. The outer function (`string_length_calculator()`) accepts parameters that customize the behavior of the returned preprocessing function. The inner function (`preprocessor()`) is what actually gets called during validation.

This approach offers several benefits as it:

- creates reusable, configurable preprocessing functions
- keeps your validation code DRY
- allows you to separate configuration from implementation
- enables easy application of the same transformation to different columns

You could extend this pattern to create even more sophisticated preprocessing functions with multiple parameters, default values, and complex logic.


# Using Narwhals to Preprocess Many Types of DataFrames

In this previous example we used a Polars table. You might have a situation where you perform data validation variously on Pandas and Polars DataFrames. This is where Narwhals becomes handy: it provides a single, consistent API that works across multiple DataFrame types, eliminating the need to learn and switch between different APIs depending on your data source.

Let's obtain `small_table` as a Pandas DataFrame. We'll construct a validation step to verify that the median of column `c` is greater than the median in column `a`.


``` python
import narwhals as nw

# Define preprocessing function using Narwhals for cross-backend compatibility
def get_median_columns_c_and_a(df):
    return nw.from_native(df).select(nw.median("c"), nw.median("a"))

(
    pb.Validate(
        data=pb.load_dataset(dataset="small_table", tbl_type="pandas"),
        tbl_name="small_table",
        label="Median comparison.",
    )
    .col_vals_gt(
        columns="c",
        value=pb.col("a"),

        # Using Narwhals to modify the table; generates table with columns `c` and `a` ---
        pre=get_median_columns_c_and_a
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Median comparison.</span>

<span style="background-color: #150458; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px #150458; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">Pandas</span><span style="background-color: none; color: #222222; padding: 0.5em 0.5em; position: inherit; margin: 5px 10px 5px -4px; border: solid 1px #150458; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">small_table</span>

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
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">c</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJtb2RpZmllZCIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9Im1vZGlmaWVkIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC41NzAxNDcpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgeD0iMC4xMjUxMzI1MDYiIHk9IjAiIHdpZHRoPSIyMy43NDk3MzUiIGhlaWdodD0iMjMuNzg5NDczNyIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzAwMDAwMCIgeD0iMTUuMjIxNTYyNiIgeT0iMTQuODAyNTg4NSIgd2lkdGg9IjYuMzQwODk4MjEiIGhlaWdodD0iNi4zNDA4OTgyMSIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzlBN0NCNCIgeD0iMTYuMTI2NDMwOSIgeT0iMTUuNzA3NDU2OCIgd2lkdGg9IjQuNTMxMTYxNTgiIGhlaWdodD0iNC41MzExNjE1OCIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
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
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(pre_applied)</span> Precondition applied: table dimensions <span style="font-family: monospace;">[13 rows, 8 columns]</span> → <span style="font-family: monospace;">[<strong>1</strong> row, <strong>2</strong> columns]</span>.</p></td>
</tr>
</tfoot>

</table>


The goal is to check that the median value of `c` is greater than the corresponding median of column `a`, which we set up through the `columns=` and `value=` parameters in the <a href="../../reference/Validate.col_vals_gt.html#pointblank.Validate.col_vals_gt" class="gdls-link"><code>Validate.col_vals_gt()</code></a> method.

There's a bit to unpack here so let's look at at the lambda function first. Narwhals can translate a Pandas DataFrame to a Narwhals DataFrame with its `from_native()` function. After that initiating step, you're free to use the Narwhals API (which is modeled on a subset of the Polars API) to do the necessary data transformation. In this case, we are getting the medians of the `c` and `a` columns and ending up with a one-row, two-column table.

We should note that the transformed table is, perhaps surprisingly, a Narwhals DataFrame (we didn't have to go back to a Pandas DataFrame by using `.to_native()`). Pointblank is able to work directly with the Narwhals DataFrame for validation purposes, which makes the workflow more concise.

One more thing to note: Pointblank provides a convenient syntactic sugar for working with Narwhals. If you name the lambda parameter `dfn` instead of `df`, the system automatically applies `nw.from_native()` to the input DataFrame first. This lets you write more concise code without having to explicitly convert the DataFrame to a Narwhals format.


# Swapping in a Totally Different DataFrame

Sometimes data validation requires looking at completely transformed versions of your data (such as aggregated summaries, pivoted views, or even reference tables). While this approach goes against the typical paradigm of validating a single *target table*, there are legitimate use cases where you might need to validate properties that only emerge after significant transformations.

Let's now try to prepare the final validation scenario, checking that there are at least three instances of every categorical value in column `f` (which contains string values in the set of `"low"`, `"mid"`, and `"high"`). This time, we'll prepare the transformed table (transformed by Polars expressions) outside of the Pointblank code.


``` python
data_original = pb.load_dataset(dataset="small_table", tbl_type="polars")
data_transformed = data_original.group_by("f").len(name="n")

data_transformed
```


shape: (3, 2)

| f      | n   |
|--------|-----|
| str    | u32 |
| "low"  | 5   |
| "high" | 6   |
| "mid"  | 2   |


Then, we'll plug in the `data_transformed` DataFrame with a preprocessing function:


``` python
# Define preprocessing function to use the transformed data
def use_transformed_data(df):
    return data_transformed

(
    pb.Validate(
        data=data_original,
        tbl_name="small_table",
        label="Category counts.",
    )
    .col_vals_ge(
        columns="n",
        value=3,
        pre=use_transformed_data
    )
    .interrogate()
)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Category counts.</span>

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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19nZTwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19nZSIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNNDMuNTIzODA5NSw0Ni42NjY2NjY3IEwyMC40NzYxOTA1LDQ2LjY2NjY2NjcgTDIwLjQ3NjE5MDUsNDQuNTcxNDI4NiBMNDMuNTIzODA5NSw0NC41NzE0Mjg2IEw0My41MjM4MDk1LDQ2LjY2NjY2NjcgWiBNMjIuMDE2MTkwNSw0MC4yNTUyMzgxIEwyMS4wMzE0Mjg2LDM4LjQxMTQyODYgTDM5LjE3NjE5MDUsMjguODU3MTQyOSBMMjEuMDMxNDI4NiwxOS4zMDI4NTcxIEwyMi4wMTYxOTA1LDE3LjQ1OTA0NzYgTDQzLjY4MDk1MjQsMjguODU3MTQyOSBMMjIuMDE2MTkwNSw0MC4yNTUyMzgxIFoiIGlkPSJncmVhdGVyX3RoYW5fZXF1YWwiIGZpbGw9IiMwMDAwMDAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_ge()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden"><span style="text-decoration: underline; text-decoration-color: #9A7CB4; text-decoration-thickness: 1px; text-underline-offset: 3px;">n</span></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">3</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjRweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjQgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJtb2RpZmllZCIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9Im1vZGlmaWVkIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC41NzAxNDcpIj4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgeD0iMC4xMjUxMzI1MDYiIHk9IjAiIHdpZHRoPSIyMy43NDk3MzUiIGhlaWdodD0iMjMuNzg5NDczNyIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzAwMDAwMCIgeD0iMTUuMjIxNTYyNiIgeT0iMTQuODAyNTg4NSIgd2lkdGg9IjYuMzQwODk4MjEiIGhlaWdodD0iNi4zNDA4OTgyMSIgLz4KICAgICAgICAgICAgPHJlY3QgaWQ9IlJlY3RhbmdsZSIgZmlsbD0iIzlBN0NCNCIgeD0iMTYuMTI2NDMwOSIgeT0iMTUuNzA3NDU2OCIgd2lkdGg9IjQuNTMxMTYxNTgiIGhlaWdodD0iNC41MzExNjE1OCIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQuNDAwOTQ4MTYsMTQuMzQ5ODUyNyBDMi4zNjkxMTYwMiwxNC4zNDk4NTI3IDAuNzE1OTQ4MTYzLDE2LjAwMjg1ODYgMC43MTU5NDgxNjMsMTguMDM0ODUyNyBDMC43MTU5NDgxNjMsMjAuMDY2ODQ2NyAyLjM2OTExNjAyLDIxLjcxOTg1MjcgNC40MDA5NDgxNiwyMS43MTk4NTI3IEM2LjQzMjc4MDMxLDIxLjcxOTg1MjcgOC4wODU5NDgxNiwyMC4wNjY4NDY3IDguMDg1OTQ4MTYsMTguMDM0ODUyNyBDOC4wODU5NDgxNiwxNi4wMDI4NTg2IDYuNDMyNzI2MzMsMTQuMzQ5ODUyNyA0LjQwMDk0ODE2LDE0LjM0OTg1MjcgWiBNNC40MDA5NDgxNiwyMC45ODI4MjAzIEMyLjc3NTQxNzY3LDIwLjk4MjgyMDMgMS40NTI5ODA1NSwxOS42NjAzODMxIDEuNDUyOTgwNTUsMTguMDM0ODUyNyBDMS40NTI5ODA1NSwxNi40MDkzMjIyIDIuNzc1NDE3NjcsMTUuMDg2ODg1IDQuNDAwOTQ4MTYsMTUuMDg2ODg1IEM2LjAyNjQ3ODY1LDE1LjA4Njg4NSA3LjM0ODkxNTc4LDE2LjQwOTMyMjIgNy4zNDg5MTU3OCwxOC4wMzQ4NTI3IEM3LjM0ODkxNTc4LDE5LjY2MDM4MzEgNi4wMjY0Nzg2NSwyMC45ODI4MjAzIDQuNDAwOTQ4MTYsMjAuOTgyODIwMyBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTE4LjQzMzM2MDEsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLDE0LjU4ODI5NTIgTDkuNTIxMzExMzEsMTQuNTg4Mjk1MiBMMTEuMzcxNjE1MSwxNS45ODA5OTcgQzExLjY1NzYxNjUsMTYuMTk0ODc2NSAxMS43MTQ4MTY5LDE2LjYwMDI1MTggMTEuNTAwOTM3NCwxNi44ODYyNTMyIEMxMS4yODcwNTg1LDE3LjE3MjI1NDYgMTAuODgxNjgyNywxNy4yMjk0NTUgMTAuNTk1NjgxMywxNy4wMTU1NzU1IEwxMC41OTU2ODEzLDE3LjAxNTU3NTUgTDcuMjMzMzAxMjYsMTQuNDY4OTIwOCBDNy4wNjY2NzQyOSwxNC4zNDk1NDY0IDYuOTY3MTk1NTksMTQuMTU1NTYzMiA2Ljk2NzE5NTU5LDEzLjk1MTYzMTYgQzYuOTY3MTk1NTksMTMuNzQ3NzAwNiA3LjA2NjY3NDI5LDEzLjU1MzcxNjggNy4yMzMzMDEyNiwxMy40MzQzNDI0IEw3LjIzMzMwMTI2LDEzLjQzNDM0MjQgTDEwLjU5NTY4MTMsMTAuODg3Njg3NyBDMTAuNjc3NzUxLDEwLjgyNTUxMzcgMTAuNzcyMjU2MSwxMC43ODU3MjIyIDEwLjg3NDIyMTYsMTAuNzY4MzEzMiBDMTAuODk5MDkxNiwxMC43NjA4NTIyIDEwLjkyNjQ0NzgsMTAuNzUzMzkxOCAxMC45NTM4MDQ2LDEwLjc0ODQxNzUgQzExLjI1MjI0MDcsMTAuNzA4NjI2IDExLjUzODI0MjEsMTAuODgyNzE0MSAxMS42Mzc3MjA4LDExLjE2NjIyOCBDMTEuNzM5Njg2MywxMS40NDk3NDI2IDExLjYyNzc3MjksMTEuNzY1NTg3IDExLjM3MTYxNTEsMTEuOTIyMjY2MSBMMTEuMzcxNjE1MSwxMS45MjIyNjYxIEw5LjUyMTMxMTMxLDEzLjMxNDk2NzkgTDE3LjE1OTM1MzMsMTMuMzE0NzEyNCBMMTcuMTU5NDU3NywwLjYzMzYxNDgyOCBMNi45MjgzNTMzLDAuNjMzNjE0ODI4IEw2LjkyODM1MzMsLTAuNjQwMjg3NTg4IEwxOC40MzMzNjAxLC0wLjY0MDI4NzU4OCBaIiBpZD0iYXJyb3ciIGZpbGw9IiMwMDAwMDAiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEyLjY4MDg1NywgOC4yNTIyODYpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTEyLjY4MDg1NywgLTguMjUyMjg2KSAiIC8+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" /></td>
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
</tbody><tfoot class="gt_sourcenotes">
<tr>
<td colspan="14" class="gt_sourcenote" style="text-align: left;"><hr />
<strong>Notes</strong>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(pre_applied)</span> Precondition applied: table dimensions <span style="font-family: monospace;">[13 rows, 8 columns]</span> → <span style="font-family: monospace;">[<strong>3</strong> rows, <strong>2</strong> columns]</span>.</p>
<p>Step 1 <span style="font-family: "IBM Plex Mono", monospace; font-size: smaller;">(syn_target_col)</span> Synthetic target column [n](../../reference/Validate.n.md#pointblank.Validate.n) created via preprocessing.</p></td>
</tr>
</tfoot>

</table>


We can see from the validation report table that there are three test units. This corresponds to a row for each of the categorical value counts. From the report, we find that two of the three test units are passing test units (turns out there are only two instances of `"mid"` in column `f`).

Note that the swapped-in table can be any table type that Pointblank supports, like a Polars DataFrame (as shown here), a Pandas DataFrame, a Narwhals DataFrame, or any other compatible format. This flexibility allows you to validate properties of your data that might only be apparent after significant reshaping or aggregation.


# Conclusion

The preprocessing capabilities in Pointblank provide the power and flexibility for validating complex data properties beyond what's directly possible with the standard validation methods. Through the `pre=` parameter, you can:

- transform your data on-the-fly with computed columns
- generate aggregated metrics to validate statistical properties
- work seamlessly across different DataFrame types using Narwhals
- swap in completely different tables when validating properties that emerge only after transformation

By combining these preprocessing techniques with Pointblank's validation methods, you can create comprehensive data quality checks that address virtually any validation scenario without needing an endless library of specialized validation functions. This composable approach keeps your validation code concise while allowing you to verify even the most complex data quality requirements.

Remember that preprocessing happens just for the specific validation step, keeping your validation plan organized and maintaining the integrity of your original data throughout the rest of the validation process.
