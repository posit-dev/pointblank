## Validate.col_vals_outside()


Do column data lie outside of two specified values or data in other columns?


Usage

``` python
Validate.col_vals_outside(
    columns,
    left,
    right,
    inclusive=(True, True),
    na_pass=False,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True
)
```


The [col_vals_between()](Validate.col_vals_between.md#pointblank.Validate.col_vals_between) validation method checks whether column values in a table *do not* fall within a certain range. The range is specified with three arguments: `left=`, `right=`, and `inclusive=`. The `left=` and `right=` values specify the lower and upper bounds. These bounds can be specified as literal values or as column names provided within <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a>. The validation will operate over the number of test units that is equal to the number of rows in the table (determined after any `pre=` mutation has been applied).


## Parameters


`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a> with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`left: float | int | Column`  
The lower bound of the range. This can be a single value or a single column name given in <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a>. The latter option allows for a column-to-column comparison for this bound. See the *What Can Be Used in `left=` and `right=`?* section for details on this.

`right: float | int | Column`  
The upper bound of the range. This can be a single value or a single column name given in <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a>. The latter option allows for a column-to-column comparison for this bound. See the *What Can Be Used in `left=` and `right=`?* section for details on this.

`inclusive: tuple[bool, bool] = (True, True)`    
A tuple of two boolean values indicating whether the comparison should be inclusive. The position of the boolean values correspond to the `left=` and `right=` values, respectively. By default, both values are `True`.

`na_pass: bool = ``False`  
Should any encountered None, NA, or Null values be considered as passing test units? By default, this is `False`. Set to `True` to pass test units with missing values.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table. Have a look at the *Preprocessing* section for more information on how to use this argument.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment). Can be a single column name, a tuple that specifies a column name and its corresponding values to segment on, or a combination of both (provided as a list). Read the *Segmentation* section for usage information.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect. Look at the *Thresholds* section for information on how to set threshold levels.

`actions: Actions | None = None`  
Optional actions to take when the validation step(s) meets or exceeds any set threshold levels. If provided, the <a href="Actions.html#pointblank.Actions" class="gdls-link"><code>Actions</code></a> class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = ``True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like <a href="has_columns.html#pointblank.has_columns" class="gdls-link"><code>has_columns()</code></a> and <a href="has_rows.html#pointblank.has_rows" class="gdls-link"><code>has_rows()</code></a> can be used here to conditionally activate a step based on properties of the target table.


## Returns


`Validate`  
The [Validate](Validate.md#pointblank.Validate) object with the added validation step.


## What Can Be Used In `left=` And `right=`?

The `left=` and `right=` arguments both allow for a variety of input types. The most common are:

- a single numeric value
- a single date or datetime value
- A <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a> object that represents a column in the target table

When supplying a number as the basis of comparison, keep in mind that all resolved columns must also be numeric. Should you have columns that are of the date or datetime types, you can supply a date or datetime value within `left=` and `right=`. There is flexibility in how you provide the date or datetime values for the bounds; they can be:

- string-based dates or datetimes (e.g., `"2023-10-01"`, `"2023-10-01 13:45:30"`, etc.)
- date or datetime objects using the `datetime` module (e.g., `datetime.date(2023, 10, 1)`, `datetime.datetime(2023, 10, 1, 13, 45, 30)`, etc.)

Finally, when supplying a column name in either `left=` or `right=` (or both), it must be specified within <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a>. This facilitates column-to-column comparisons and, crucially, the columns being compared to either/both of the bounds must be of the same type as the column data (e.g., all numeric, all dates, etc.).


## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Note that you can refer to columns via `columns=` and `left=col(...)`/`right=col(...)` that are expected to be present in the transformed table, but may not exist in the table before preprocessing. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](Validate.md#pointblank.Validate) object or used in subsequent validation steps.


## Segmentation

The `segments=` argument allows for the segmentation of a validation step into multiple segments. This is useful for applying the same validation step to different subsets of the data. The segmentation can be done based on a single column or specific fields within a column.

Providing a single column name will result in a separate validation step for each unique value in that column. For example, if you have a column called `"region"` with values `"North"`, `"South"`, and `"East"`, the validation step will be applied separately to each region.

Alternatively, you can provide a tuple that specifies a column name and its corresponding values to segment on. For example, if you have a column called `"date"` and you want to segment on only specific dates, you can provide a tuple like `("date", ["2023-01-01", "2023-01-02"])`. Any other values in the column will be disregarded (i.e., no validation steps will be created for them).

A list with a combination of column names and tuples can be provided as well. This allows for more complex segmentation scenarios. The following inputs are both valid:

    # Segments from all unique values in the `region` column
    # and specific dates in the `date` column
    segments=["region", ("date", ["2023-01-01", "2023-01-02"])]

    # Segments from all unique values in the `region` and `date` columns
    segments=["region", "date"]

The segmentation is performed during interrogation, and the resulting validation steps will be numbered sequentially. Each segment will have its own validation step, and the results will be reported separately. This allows for a more granular analysis of the data and helps identify issues within specific segments.

Importantly, the segmentation process will be performed after any preprocessing of the data table. Because of this, one can conceivably use the `pre=` argument to generate a column that can be used for segmentation. For example, you could create a new column called `"segment"` through use of `pre=` and then use that column for segmentation.


## Thresholds

The `thresholds=` parameter is used to set the failure-condition levels for the validation step. If they are set here at the step level, these thresholds will override any thresholds set at the global level in `Validate(thresholds=...)`.

There are three threshold levels: 'warning', 'error', and 'critical'. The threshold values can either be set as a proportion failing of all test units (a value between `0` to `1`), or, the absolute number of failing test units (as integer that's `1` or greater).

Thresholds can be defined using one of these input schemes:

1.  use the <a href="Thresholds.html#pointblank.Thresholds" class="gdls-link"><code>Thresholds</code></a> class (the most direct way to create thresholds)
2.  provide a tuple of 1-3 values, where position `0` is the 'warning' level, position `1` is the 'error' level, and position `2` is the 'critical' level
3.  create a dictionary of 1-3 value entries; the valid keys: are 'warning', 'error', and 'critical'
4.  a single integer/float value denoting absolute number or fraction of failing test units for the 'warning' level only

If the number of failing test units exceeds set thresholds, the validation step will be marked as 'warning', 'error', or 'critical'. All of the threshold levels don't need to be set, you're free to set any combination of them.

Aside from reporting failure conditions, thresholds can be used to determine the actions to take for each level of failure (using the `actions=` parameter).


## Examples

For the examples here, we'll use a simple Polars DataFrame with three numeric columns (`a`, `b`, and `c`). The table is shown below:


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [5, 6, 5, 7, 5, 5],
        "b": [2, 3, 6, 4, 3, 6],
        "c": [9, 8, 8, 9, 9, 7],
    }
)

pb.preview(tbl)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_preview_tbl-_row_num_" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"></th>
<th id="pb_preview_tbl-a" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

a

<em>Int64</em>

</div></th>
<th id="pb_preview_tbl-b" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

b

<em>Int64</em>

</div></th>
<th id="pb_preview_tbl-c" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

c

<em>Int64</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">9</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">8</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">8</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">9</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">9</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">6</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7</td>
</tr>
</tbody>
</table>


Let's validate that values in column `a` are all outside the fixed boundary values of `1` and `4`. We'll determine if this validation had any failing test units (there are six test units, one for each row).


``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_outside(columns="a", left=1, right=4)
    .interrogate()
)

validation
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfb3V0c2lkZTwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19vdXRzaWRlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC42ODk2NTUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJvdXRzaWRlX3JhbmdlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMS4wMDAwMDAsIDIxLjAwMDAwMCkiIGZpbGw9IiMwMDAwMDAiPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTEyLjk5MzQ4NCwwLjk2ODc1IEMxMS45NjIyMzQsMS4wODIwMzEgMTEuMTg4Nzk3LDEuOTY0ODQ0IDExLjIxMjIzNCwzIEwxMS4yMTIyMzQsMjEgQzExLjIwMDUxNSwyMS43MjI2NTYgMTEuNTc5NDIyLDIyLjM5MDYyNSAxMi4yMDQ0MjIsMjIuNzUzOTA2IEMxMi44MjU1MTUsMjMuMTIxMDk0IDEzLjU5ODk1MywyMy4xMjEwOTQgMTQuMjIwMDQ3LDIyLjc1MzkwNiBDMTQuODQ1MDQ3LDIyLjM5MDYyNSAxNS4yMjM5NTMsMjEuNzIyNjU2IDE1LjIxMjIzNCwyMSBMMTUuMjEyMjM0LDMgQzE1LjIyMDA0NywyLjQ1NzAzMSAxNS4wMDkxMDksMS45Mzc1IDE0LjYyNjI5NywxLjU1NDY4OCBDMTQuMjQzNDg0LDEuMTcxODc1IDEzLjcyMzk1MywwLjk2MDkzOCAxMy4xODA5ODQsMC45Njg3NSBDMTMuMTE4NDg0LDAuOTY0ODQ0IDEzLjA1NTk4NCwwLjk2NDg0NCAxMi45OTM0ODQsMC45Njg3NSBaIE0zMi45OTM0ODQsMC45Njg3NSBDMzEuOTYyMjM0LDEuMDgyMDMxIDMxLjE4ODc5NywxLjk2NDg0NCAzMS4yMTIyMzQsMyBMMzEuMjEyMjM0LDIxIEMzMS4yMDA1MTUsMjEuNzIyNjU2IDMxLjU3OTQyMiwyMi4zOTA2MjUgMzIuMjA0NDIyLDIyLjc1MzkwNiBDMzIuODI1NTE1LDIzLjEyMTA5NCAzMy41OTg5NTMsMjMuMTIxMDk0IDM0LjIyMDA0NywyMi43NTM5MDYgQzM0Ljg0NTA0NywyMi4zOTA2MjUgMzUuMjIzOTUzLDIxLjcyMjY1NiAzNS4yMTIyMzQsMjEgTDM1LjIxMjIzNCwzIEMzNS4yMjAwNDcsMi40NTcwMzEgMzUuMDA5MTA5LDEuOTM3NSAzNC42MjYyOTcsMS41NTQ2ODggQzM0LjI0MzQ4NCwxLjE3MTg3NSAzMy43MjM5NTMsMC45NjA5MzggMzMuMTgwOTg0LDAuOTY4NzUgQzMzLjExODQ4NCwwLjk2NDg0NCAzMy4wNTU5ODQsMC45NjQ4NDQgMzIuOTkzNDg0LDAuOTY4NzUgWiBNMTcuMjEyMjM0LDEgQzE2LjY2MTQ1MywxIDE2LjIxMjIzNCwxLjQ0OTIxOSAxNi4yMTIyMzQsMiBDMTYuMjEyMjM0LDIuNTUwNzgxIDE2LjY2MTQ1MywzIDE3LjIxMjIzNCwzIEMxNy43NjMwMTUsMyAxOC4yMTIyMzQsMi41NTA3ODEgMTguMjEyMjM0LDIgQzE4LjIxMjIzNCwxLjQ0OTIxOSAxNy43NjMwMTUsMSAxNy4yMTIyMzQsMSBaIE0yMS4yMTIyMzQsMSBDMjAuNjYxNDUzLDEgMjAuMjEyMjM0LDEuNDQ5MjE5IDIwLjIxMjIzNCwyIEMyMC4yMTIyMzQsMi41NTA3ODEgMjAuNjYxNDUzLDMgMjEuMjEyMjM0LDMgQzIxLjc2MzAxNSwzIDIyLjIxMjIzNCwyLjU1MDc4MSAyMi4yMTIyMzQsMiBDMjIuMjEyMjM0LDEuNDQ5MjE5IDIxLjc2MzAxNSwxIDIxLjIxMjIzNCwxIFogTTI1LjIxMjIzNCwxIEMyNC42NjE0NTMsMSAyNC4yMTIyMzQsMS40NDkyMTkgMjQuMjEyMjM0LDIgQzI0LjIxMjIzNCwyLjU1MDc4MSAyNC42NjE0NTMsMyAyNS4yMTIyMzQsMyBDMjUuNzYzMDE1LDMgMjYuMjEyMjM0LDIuNTUwNzgxIDI2LjIxMjIzNCwyIEMyNi4yMTIyMzQsMS40NDkyMTkgMjUuNzYzMDE1LDEgMjUuMjEyMjM0LDEgWiBNMjkuMjEyMjM0LDEgQzI4LjY2MTQ1MywxIDI4LjIxMjIzNCwxLjQ0OTIxOSAyOC4yMTIyMzQsMiBDMjguMjEyMjM0LDIuNTUwNzgxIDI4LjY2MTQ1MywzIDI5LjIxMjIzNCwzIEMyOS43NjMwMTUsMyAzMC4yMTIyMzQsMi41NTA3ODEgMzAuMjEyMjM0LDIgQzMwLjIxMjIzNCwxLjQ0OTIxOSAyOS43NjMwMTUsMSAyOS4yMTIyMzQsMSBaIE0xNy4yMTIyMzQsMjEgQzE2LjY2MTQ1MywyMSAxNi4yMTIyMzQsMjEuNDQ5MjE5IDE2LjIxMjIzNCwyMiBDMTYuMjEyMjM0LDIyLjU1MDc4MSAxNi42NjE0NTMsMjMgMTcuMjEyMjM0LDIzIEMxNy43NjMwMTUsMjMgMTguMjEyMjM0LDIyLjU1MDc4MSAxOC4yMTIyMzQsMjIgQzE4LjIxMjIzNCwyMS40NDkyMTkgMTcuNzYzMDE1LDIxIDE3LjIxMjIzNCwyMSBaIE0yMS4yMTIyMzQsMjEgQzIwLjY2MTQ1MywyMSAyMC4yMTIyMzQsMjEuNDQ5MjE5IDIwLjIxMjIzNCwyMiBDMjAuMjEyMjM0LDIyLjU1MDc4MSAyMC42NjE0NTMsMjMgMjEuMjEyMjM0LDIzIEMyMS43NjMwMTUsMjMgMjIuMjEyMjM0LDIyLjU1MDc4MSAyMi4yMTIyMzQsMjIgQzIyLjIxMjIzNCwyMS40NDkyMTkgMjEuNzYzMDE1LDIxIDIxLjIxMjIzNCwyMSBaIE0yNS4yMTIyMzQsMjEgQzI0LjY2MTQ1MywyMSAyNC4yMTIyMzQsMjEuNDQ5MjE5IDI0LjIxMjIzNCwyMiBDMjQuMjEyMjM0LDIyLjU1MDc4MSAyNC42NjE0NTMsMjMgMjUuMjEyMjM0LDIzIEMyNS43NjMwMTUsMjMgMjYuMjEyMjM0LDIyLjU1MDc4MSAyNi4yMTIyMzQsMjIgQzI2LjIxMjIzNCwyMS40NDkyMTkgMjUuNzYzMDE1LDIxIDI1LjIxMjIzNCwyMSBaIE0yOS4yMTIyMzQsMjEgQzI4LjY2MTQ1MywyMSAyOC4yMTIyMzQsMjEuNDQ5MjE5IDI4LjIxMjIzNCwyMiBDMjguMjEyMjM0LDIyLjU1MDc4MSAyOC42NjE0NTMsMjMgMjkuMjEyMjM0LDIzIEMyOS43NjMwMTUsMjMgMzAuMjEyMjM0LDIyLjU1MDc4MSAzMC4yMTIyMzQsMjIgQzMwLjIxMjIzNCwyMS40NDkyMTkgMjkuNzYzMDE1LDIxIDI5LjIxMjIzNCwyMSBaIiBpZD0ic21hbGxfcmFuZ2UiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik02Ljg2ODUwNTksNi45Nzg1NzkzMyBDNy4zMzcyNTU5LDYuOTE2MDc5MzMgNy43ODY0NzQ5LDcuMTg5NTE3MzMgNy45NDI3MjQ5LDcuNjM0ODI5MzMgQzguMTAyODgwOSw4LjA4MDE0MjMzIDcuOTI3MDk5OSw4LjU3NjIzNTMzIDcuNTI0NzU1OSw4LjgyMjMyOTMzIEw3LjUyNDc1NTksOC44MjIzMjkzMyBMNC42MTg1MDU5LDExLjAwOTgyOTMgTDEwLjIxMDc4NjksMTEuMDA5NzIxNiBMMTAuMjEwNzg2OSwxMy4wMDk3MjE2IEw0LjYxODUwNTksMTMuMDA5ODI5MyBMNy41MjQ3NTU5LDE1LjE5NzMyOTMgQzcuOTczOTc0OSwxNS41MzMyNjczIDguMDYzODE4OSwxNi4xNjk5ODUzIDcuNzI3ODgwOSwxNi42MTkyMDQzIEM3LjM5MTk0MzksMTcuMDY4NDIzMyA2Ljc1NTIyNDksMTcuMTU4MjY3MyA2LjMwNjAwNTksMTYuODIyMzI5MyBMNi4zMDYwMDU5LDE2LjgyMjMyOTMgTDEuMDI0NzU1OSwxMi44MjIzMjkzIEMwLjc2MzAzNjg5NiwxMi42MzQ4MjkzIDAuNjA2Nzg2ODk2LDEyLjMzMDE0MjMgMC42MDY3ODY4OTYsMTIuMDA5ODI5MyBDMC42MDY3ODY4OTYsMTEuNjg5NTE3MyAwLjc2MzAzNjg5NiwxMS4zODQ4MjkzIDEuMDI0NzU1OSwxMS4xOTczMjkzIEwxLjAyNDc1NTksMTEuMTk3MzI5MyBMNi4zMDYwMDU5LDcuMTk3MzI5MzMgQzYuNDM0OTExOSw3LjA5OTY3MzMzIDYuNTgzMzQ5OSw3LjAzNzE3MzMzIDYuNzQzNTA1OSw3LjAwOTgyOTMzIEM2Ljc4MjU2ODksNi45OTgxMTAzMyA2LjgyNTUzNjksNi45ODYzOTIzMyA2Ljg2ODUwNTksNi45Nzg1NzkzMyBaIiBpZD0iYXJyb3dfbGVmdCIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00Mi4zNjkwMjQ2LDYuOTgxMDIwNTggQzQyLjgzNzc3NDYsNi45MTg1MjA1OCA0My4yODY5OTM2LDcuMTkxOTU4NTggNDMuNDQzMjQzNiw3LjYzNzI3MDU4IEM0My42MDMzOTk2LDguMDgyNTgzNTggNDMuNDI3NjE4Niw4LjU3ODY3NjU4IDQzLjAyNTI3NDYsOC44MjQ3NzA1OCBMNDMuMDI1Mjc0Niw4LjgyNDc3MDU4IEw0MC4xMTkwMjQ2LDExLjAxMjI3MDYgTDQ1LjcxMTMwNTYsMTEuMDEyMTYyOSBMNDUuNzExMzA1NiwxMy4wMTIxNjI5IEw0MC4xMTkwMjQ2LDEzLjAxMjI3MDYgTDQzLjAyNTI3NDYsMTUuMTk5NzcwNiBDNDMuNDc0NDkzNiwxNS41MzU3MDg2IDQzLjU2NDMzNzYsMTYuMTcyNDI2NiA0My4yMjgzOTk2LDE2LjYyMTY0NTYgQzQyLjg5MjQ2MjYsMTcuMDcwODY0NiA0Mi4yNTU3NDM2LDE3LjE2MDcwODYgNDEuODA2NTI0NiwxNi44MjQ3NzA2IEw0MS44MDY1MjQ2LDE2LjgyNDc3MDYgTDM2LjUyNTI3NDYsMTIuODI0NzcwNiBDMzYuMjYzNTU1NiwxMi42MzcyNzA2IDM2LjEwNzMwNTYsMTIuMzMyNTgzNiAzNi4xMDczMDU2LDEyLjAxMjI3MDYgQzM2LjEwNzMwNTYsMTEuNjkxOTU4NiAzNi4yNjM1NTU2LDExLjM4NzI3MDYgMzYuNTI1Mjc0NiwxMS4xOTk3NzA2IEwzNi41MjUyNzQ2LDExLjE5OTc3MDYgTDQxLjgwNjUyNDYsNy4xOTk3NzA1OCBDNDEuOTM1NDMwNiw3LjEwMjExNDU4IDQyLjA4Mzg2ODYsNy4wMzk2MTQ1OCA0Mi4yNDQwMjQ2LDcuMDEyMjcwNTggQzQyLjI4MzA4NzYsNy4wMDA1NTE1OCA0Mi4zMjYwNTU2LDYuOTg4ODMzNTggNDIuMzY5MDI0Niw2Ljk4MTAyMDU4IFoiIGlkPSJhcnJvd19yaWdodCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDAuOTA5NzcwLCAxMi4wMDAwMDApIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC00MC45MDk3NzAsIC0xMi4wMDAwMDApICIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_outside()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">a</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">[1, 4]</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
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


Printing the `validation` object shows the validation table in an HTML viewing environment. The validation table shows the single entry that corresponds to the validation step created by using [col_vals_outside()](Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside). All test units passed, and there are no failing test units.

Aside from checking a column against two literal values representing the lower and upper bounds, we can also provide column names to the `left=` and/or `right=` arguments (by using the helper function <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a>. In this way, we can perform three additional comparison types:

1.  `left=column`, `right=column`
2.  `left=literal`, `right=column`
3.  `left=column`, `right=literal`

For the next example, we'll use [col_vals_outside()](Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside) to check whether the values in column `b` are outside of the range formed by the corresponding values in columns `a` (lower bound) and `c` (upper bound).


``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_outside(columns="b", left=pb.col("a"), right=pb.col("c"))
    .interrogate()
)

validation
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfb3V0c2lkZTwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19vdXRzaWRlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC42ODk2NTUpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJvdXRzaWRlX3JhbmdlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMS4wMDAwMDAsIDIxLjAwMDAwMCkiIGZpbGw9IiMwMDAwMDAiPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTEyLjk5MzQ4NCwwLjk2ODc1IEMxMS45NjIyMzQsMS4wODIwMzEgMTEuMTg4Nzk3LDEuOTY0ODQ0IDExLjIxMjIzNCwzIEwxMS4yMTIyMzQsMjEgQzExLjIwMDUxNSwyMS43MjI2NTYgMTEuNTc5NDIyLDIyLjM5MDYyNSAxMi4yMDQ0MjIsMjIuNzUzOTA2IEMxMi44MjU1MTUsMjMuMTIxMDk0IDEzLjU5ODk1MywyMy4xMjEwOTQgMTQuMjIwMDQ3LDIyLjc1MzkwNiBDMTQuODQ1MDQ3LDIyLjM5MDYyNSAxNS4yMjM5NTMsMjEuNzIyNjU2IDE1LjIxMjIzNCwyMSBMMTUuMjEyMjM0LDMgQzE1LjIyMDA0NywyLjQ1NzAzMSAxNS4wMDkxMDksMS45Mzc1IDE0LjYyNjI5NywxLjU1NDY4OCBDMTQuMjQzNDg0LDEuMTcxODc1IDEzLjcyMzk1MywwLjk2MDkzOCAxMy4xODA5ODQsMC45Njg3NSBDMTMuMTE4NDg0LDAuOTY0ODQ0IDEzLjA1NTk4NCwwLjk2NDg0NCAxMi45OTM0ODQsMC45Njg3NSBaIE0zMi45OTM0ODQsMC45Njg3NSBDMzEuOTYyMjM0LDEuMDgyMDMxIDMxLjE4ODc5NywxLjk2NDg0NCAzMS4yMTIyMzQsMyBMMzEuMjEyMjM0LDIxIEMzMS4yMDA1MTUsMjEuNzIyNjU2IDMxLjU3OTQyMiwyMi4zOTA2MjUgMzIuMjA0NDIyLDIyLjc1MzkwNiBDMzIuODI1NTE1LDIzLjEyMTA5NCAzMy41OTg5NTMsMjMuMTIxMDk0IDM0LjIyMDA0NywyMi43NTM5MDYgQzM0Ljg0NTA0NywyMi4zOTA2MjUgMzUuMjIzOTUzLDIxLjcyMjY1NiAzNS4yMTIyMzQsMjEgTDM1LjIxMjIzNCwzIEMzNS4yMjAwNDcsMi40NTcwMzEgMzUuMDA5MTA5LDEuOTM3NSAzNC42MjYyOTcsMS41NTQ2ODggQzM0LjI0MzQ4NCwxLjE3MTg3NSAzMy43MjM5NTMsMC45NjA5MzggMzMuMTgwOTg0LDAuOTY4NzUgQzMzLjExODQ4NCwwLjk2NDg0NCAzMy4wNTU5ODQsMC45NjQ4NDQgMzIuOTkzNDg0LDAuOTY4NzUgWiBNMTcuMjEyMjM0LDEgQzE2LjY2MTQ1MywxIDE2LjIxMjIzNCwxLjQ0OTIxOSAxNi4yMTIyMzQsMiBDMTYuMjEyMjM0LDIuNTUwNzgxIDE2LjY2MTQ1MywzIDE3LjIxMjIzNCwzIEMxNy43NjMwMTUsMyAxOC4yMTIyMzQsMi41NTA3ODEgMTguMjEyMjM0LDIgQzE4LjIxMjIzNCwxLjQ0OTIxOSAxNy43NjMwMTUsMSAxNy4yMTIyMzQsMSBaIE0yMS4yMTIyMzQsMSBDMjAuNjYxNDUzLDEgMjAuMjEyMjM0LDEuNDQ5MjE5IDIwLjIxMjIzNCwyIEMyMC4yMTIyMzQsMi41NTA3ODEgMjAuNjYxNDUzLDMgMjEuMjEyMjM0LDMgQzIxLjc2MzAxNSwzIDIyLjIxMjIzNCwyLjU1MDc4MSAyMi4yMTIyMzQsMiBDMjIuMjEyMjM0LDEuNDQ5MjE5IDIxLjc2MzAxNSwxIDIxLjIxMjIzNCwxIFogTTI1LjIxMjIzNCwxIEMyNC42NjE0NTMsMSAyNC4yMTIyMzQsMS40NDkyMTkgMjQuMjEyMjM0LDIgQzI0LjIxMjIzNCwyLjU1MDc4MSAyNC42NjE0NTMsMyAyNS4yMTIyMzQsMyBDMjUuNzYzMDE1LDMgMjYuMjEyMjM0LDIuNTUwNzgxIDI2LjIxMjIzNCwyIEMyNi4yMTIyMzQsMS40NDkyMTkgMjUuNzYzMDE1LDEgMjUuMjEyMjM0LDEgWiBNMjkuMjEyMjM0LDEgQzI4LjY2MTQ1MywxIDI4LjIxMjIzNCwxLjQ0OTIxOSAyOC4yMTIyMzQsMiBDMjguMjEyMjM0LDIuNTUwNzgxIDI4LjY2MTQ1MywzIDI5LjIxMjIzNCwzIEMyOS43NjMwMTUsMyAzMC4yMTIyMzQsMi41NTA3ODEgMzAuMjEyMjM0LDIgQzMwLjIxMjIzNCwxLjQ0OTIxOSAyOS43NjMwMTUsMSAyOS4yMTIyMzQsMSBaIE0xNy4yMTIyMzQsMjEgQzE2LjY2MTQ1MywyMSAxNi4yMTIyMzQsMjEuNDQ5MjE5IDE2LjIxMjIzNCwyMiBDMTYuMjEyMjM0LDIyLjU1MDc4MSAxNi42NjE0NTMsMjMgMTcuMjEyMjM0LDIzIEMxNy43NjMwMTUsMjMgMTguMjEyMjM0LDIyLjU1MDc4MSAxOC4yMTIyMzQsMjIgQzE4LjIxMjIzNCwyMS40NDkyMTkgMTcuNzYzMDE1LDIxIDE3LjIxMjIzNCwyMSBaIE0yMS4yMTIyMzQsMjEgQzIwLjY2MTQ1MywyMSAyMC4yMTIyMzQsMjEuNDQ5MjE5IDIwLjIxMjIzNCwyMiBDMjAuMjEyMjM0LDIyLjU1MDc4MSAyMC42NjE0NTMsMjMgMjEuMjEyMjM0LDIzIEMyMS43NjMwMTUsMjMgMjIuMjEyMjM0LDIyLjU1MDc4MSAyMi4yMTIyMzQsMjIgQzIyLjIxMjIzNCwyMS40NDkyMTkgMjEuNzYzMDE1LDIxIDIxLjIxMjIzNCwyMSBaIE0yNS4yMTIyMzQsMjEgQzI0LjY2MTQ1MywyMSAyNC4yMTIyMzQsMjEuNDQ5MjE5IDI0LjIxMjIzNCwyMiBDMjQuMjEyMjM0LDIyLjU1MDc4MSAyNC42NjE0NTMsMjMgMjUuMjEyMjM0LDIzIEMyNS43NjMwMTUsMjMgMjYuMjEyMjM0LDIyLjU1MDc4MSAyNi4yMTIyMzQsMjIgQzI2LjIxMjIzNCwyMS40NDkyMTkgMjUuNzYzMDE1LDIxIDI1LjIxMjIzNCwyMSBaIE0yOS4yMTIyMzQsMjEgQzI4LjY2MTQ1MywyMSAyOC4yMTIyMzQsMjEuNDQ5MjE5IDI4LjIxMjIzNCwyMiBDMjguMjEyMjM0LDIyLjU1MDc4MSAyOC42NjE0NTMsMjMgMjkuMjEyMjM0LDIzIEMyOS43NjMwMTUsMjMgMzAuMjEyMjM0LDIyLjU1MDc4MSAzMC4yMTIyMzQsMjIgQzMwLjIxMjIzNCwyMS40NDkyMTkgMjkuNzYzMDE1LDIxIDI5LjIxMjIzNCwyMSBaIiBpZD0ic21hbGxfcmFuZ2UiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik02Ljg2ODUwNTksNi45Nzg1NzkzMyBDNy4zMzcyNTU5LDYuOTE2MDc5MzMgNy43ODY0NzQ5LDcuMTg5NTE3MzMgNy45NDI3MjQ5LDcuNjM0ODI5MzMgQzguMTAyODgwOSw4LjA4MDE0MjMzIDcuOTI3MDk5OSw4LjU3NjIzNTMzIDcuNTI0NzU1OSw4LjgyMjMyOTMzIEw3LjUyNDc1NTksOC44MjIzMjkzMyBMNC42MTg1MDU5LDExLjAwOTgyOTMgTDEwLjIxMDc4NjksMTEuMDA5NzIxNiBMMTAuMjEwNzg2OSwxMy4wMDk3MjE2IEw0LjYxODUwNTksMTMuMDA5ODI5MyBMNy41MjQ3NTU5LDE1LjE5NzMyOTMgQzcuOTczOTc0OSwxNS41MzMyNjczIDguMDYzODE4OSwxNi4xNjk5ODUzIDcuNzI3ODgwOSwxNi42MTkyMDQzIEM3LjM5MTk0MzksMTcuMDY4NDIzMyA2Ljc1NTIyNDksMTcuMTU4MjY3MyA2LjMwNjAwNTksMTYuODIyMzI5MyBMNi4zMDYwMDU5LDE2LjgyMjMyOTMgTDEuMDI0NzU1OSwxMi44MjIzMjkzIEMwLjc2MzAzNjg5NiwxMi42MzQ4MjkzIDAuNjA2Nzg2ODk2LDEyLjMzMDE0MjMgMC42MDY3ODY4OTYsMTIuMDA5ODI5MyBDMC42MDY3ODY4OTYsMTEuNjg5NTE3MyAwLjc2MzAzNjg5NiwxMS4zODQ4MjkzIDEuMDI0NzU1OSwxMS4xOTczMjkzIEwxLjAyNDc1NTksMTEuMTk3MzI5MyBMNi4zMDYwMDU5LDcuMTk3MzI5MzMgQzYuNDM0OTExOSw3LjA5OTY3MzMzIDYuNTgzMzQ5OSw3LjAzNzE3MzMzIDYuNzQzNTA1OSw3LjAwOTgyOTMzIEM2Ljc4MjU2ODksNi45OTgxMTAzMyA2LjgyNTUzNjksNi45ODYzOTIzMyA2Ljg2ODUwNTksNi45Nzg1NzkzMyBaIiBpZD0iYXJyb3dfbGVmdCIgLz4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00Mi4zNjkwMjQ2LDYuOTgxMDIwNTggQzQyLjgzNzc3NDYsNi45MTg1MjA1OCA0My4yODY5OTM2LDcuMTkxOTU4NTggNDMuNDQzMjQzNiw3LjYzNzI3MDU4IEM0My42MDMzOTk2LDguMDgyNTgzNTggNDMuNDI3NjE4Niw4LjU3ODY3NjU4IDQzLjAyNTI3NDYsOC44MjQ3NzA1OCBMNDMuMDI1Mjc0Niw4LjgyNDc3MDU4IEw0MC4xMTkwMjQ2LDExLjAxMjI3MDYgTDQ1LjcxMTMwNTYsMTEuMDEyMTYyOSBMNDUuNzExMzA1NiwxMy4wMTIxNjI5IEw0MC4xMTkwMjQ2LDEzLjAxMjI3MDYgTDQzLjAyNTI3NDYsMTUuMTk5NzcwNiBDNDMuNDc0NDkzNiwxNS41MzU3MDg2IDQzLjU2NDMzNzYsMTYuMTcyNDI2NiA0My4yMjgzOTk2LDE2LjYyMTY0NTYgQzQyLjg5MjQ2MjYsMTcuMDcwODY0NiA0Mi4yNTU3NDM2LDE3LjE2MDcwODYgNDEuODA2NTI0NiwxNi44MjQ3NzA2IEw0MS44MDY1MjQ2LDE2LjgyNDc3MDYgTDM2LjUyNTI3NDYsMTIuODI0NzcwNiBDMzYuMjYzNTU1NiwxMi42MzcyNzA2IDM2LjEwNzMwNTYsMTIuMzMyNTgzNiAzNi4xMDczMDU2LDEyLjAxMjI3MDYgQzM2LjEwNzMwNTYsMTEuNjkxOTU4NiAzNi4yNjM1NTU2LDExLjM4NzI3MDYgMzYuNTI1Mjc0NiwxMS4xOTk3NzA2IEwzNi41MjUyNzQ2LDExLjE5OTc3MDYgTDQxLjgwNjUyNDYsNy4xOTk3NzA1OCBDNDEuOTM1NDMwNiw3LjEwMjExNDU4IDQyLjA4Mzg2ODYsNy4wMzk2MTQ1OCA0Mi4yNDQwMjQ2LDcuMDEyMjcwNTggQzQyLjI4MzA4NzYsNy4wMDA1NTE1OCA0Mi4zMjYwNTU2LDYuOTg4ODMzNTggNDIuMzY5MDI0Niw2Ljk4MTAyMDU4IFoiIGlkPSJhcnJvd19yaWdodCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoNDAuOTA5NzcwLCAxMi4wMDAwMDApIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC00MC45MDk3NzAsIC0xMi4wMDAwMDApICIgLz4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_outside()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">b</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">[a, c]</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">6</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">4<br />
0.67</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.33</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


The validation table reports two failing test units. The specific failing cases are:

- Row 2: `b` is `6` and the bounds are `5` (`a`) and `8` (`c`).
- Row 5: `b` is `6` and the bounds are `5` (`a`) and `7` (`c`).
