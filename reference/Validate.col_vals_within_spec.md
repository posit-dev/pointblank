## Validate.col_vals_within_spec()


Validate whether column values fit within a specification.


Usage

``` python
Validate.col_vals_within_spec(
    columns,
    spec,
    na_pass=False,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True
)
```


The [col_vals_within_spec()](Validate.col_vals_within_spec.md#pointblank.Validate.col_vals_within_spec) validation method checks whether column values in a table correspond to a specification (`spec=`) type (details of which are available in the *Specifications* section). Specifications include common data types like email addresses, URLs, postal codes, vehicle identification numbers (VINs), International Bank Account Numbers (IBANs), and more. This validation will operate over the number of test units that is equal to the number of rows in the table.


## Parameters


`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use <a href="col.html#pointblank.col" class="gdls-link"><code>col()</code></a> with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`spec: str`  
A specification string for defining the specification type. Examples are `"email"`, `"url"`, and `"postal_code[USA]"`. See the *Specifications* section for all available options.

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


## Specifications

A specification type must be used with the `spec=` argument. This is a string-based keyword that corresponds to the type of data in the specified columns. The following keywords can be used:

- `"isbn"`: The International Standard Book Number (ISBN) is a unique numerical identifier for books. This keyword validates both 10-digit and 13-digit ISBNs.

- `"vin"`: A vehicle identification number (VIN) is a unique code used by the automotive industry to identify individual motor vehicles.

- `"postal_code[<country_code>]"`: A postal code (also known as postcodes, PIN, or ZIP codes) is a series of letters, digits, or both included in a postal address. Because the coding varies by country, a country code in either the 2-letter (ISO 3166-1 alpha-2) or 3-letter (ISO 3166-1 alpha-3) format needs to be supplied (e.g., `"postal_code[US]"` or `"postal_code[USA]"`). The keyword alias `"zip"` can be used for US ZIP codes.

- `"credit_card"`: A credit card number can be validated across a variety of issuers. The validation uses the Luhn algorithm.

- `"iban[<country_code>]"`: The International Bank Account Number (IBAN) is a system of identifying bank accounts across countries. Because the length and coding varies by country, a country code needs to be supplied (e.g., `"iban[DE]"` or `"iban[DEU]"`).

- `"swift"`: Business Identifier Codes (also known as SWIFT-BIC, BIC, or SWIFT code) are unique identifiers for financial and non-financial institutions.

- `"phone"`, `"email"`, `"url"`, `"ipv4"`, `"ipv6"`, `"mac"`: Phone numbers, email addresses, Internet URLs, IPv4 or IPv6 addresses, and MAC addresses can be validated with their respective keywords.

Only a single `spec=` value should be provided per function call.


## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Note that you can refer to a column via `columns=` that is expected to be present in the transformed table, but may not exist in the table before preprocessing. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](Validate.md#pointblank.Validate) object or used in subsequent validation steps.


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

For the examples here, we'll use a simple Polars DataFrame with an email column. The table is shown below:


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "email": [
            "user@example.com",
            "admin@test.org",
            "invalid-email",
            "contact@company.co.uk",
        ],
    }
)

pb.preview(tbl)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_col_headings">
<th id="pb_preview_tbl-_row_num_" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"></th>
<th id="pb_preview_tbl-email" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"><div>

email

<em>String</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">1</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">user@example.com</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">admin@test.org</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">invalid-email</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">4</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">contact@company.co.uk</td>
</tr>
</tbody>
</table>


Let's validate that all of the values in the `email` column are valid email addresses. We'll determine if this validation had any failing test units (there are four test units, one for each row).


``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_within_spec(columns="email", spec="email")
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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfd2l0aGluX3NwZWM8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfd2l0aGluX3NwZWMiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjUxNzI0MSkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9Imdsb2JlIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSg5LjcxMjIzNCwgOS4wMDAwMDApIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTI0LDAuOTE5OTIxOSBDMTEuMjY1MTEzLDAuOTE5OTIxOSAwLjkxOTkyMTksMTEuMjY1MTEzIDAuOTE5OTIxOSwyNCBDMC45MTk5MjE5LDM2LjczNDg4NyAxMS4yNjUxMTMsNDcuMDgwMDc4IDI0LDQ3LjA4MDA3OCBDMzYuNzM0ODg3LDQ3LjA4MDA3OCA0Ny4wODAwNzgsMzYuNzM0ODg3IDQ3LjA4MDA3OCwyNCBDNDcuMDgwMDc4LDExLjI2NTExMyAzNi43MzQ4ODcsMC45MTk5MjE5IDI0LDAuOTE5OTIxOSBaIE0yMywzLjAzNzEwOTQgTDIzLDEyLjk3MDcwMyBDMjAuNDMyNTEsMTIuOTEwMTEgMTcuOTkxNDQ1LDEyLjYyMzAyMiAxNS43NDAyMzQsMTIuMTUyMzQ0IEMxNi4xMzY2MjcsMTAuOTUxNSAxNi41ODcxMDMsOS44MzU1NTkgMTcuMDg3ODkxLDguODMzOTg0NCBDMTguNzQwODI1LDUuNTI4MTE1NiAyMC44MzcyODYsMy41MTYwNDk4IDIzLDMuMDM3MTA5NCBaIE0yNSwzLjAzNzEwOTQgQzI3LjE2MjcxNCwzLjUxNjA0OTggMjkuMjU5MTc1LDUuNTI4MTE1NiAzMC45MTIxMDksOC44MzM5ODQ0IEMzMS40MTQ0OTYsOS44Mzg3NTcgMzEuODY2Mzc5LDEwLjk1ODgwNiAzMi4yNjM2NzIsMTIuMTY0MDYyIEMzMC4wMTUyNjksMTIuNjMwMDM3IDI3LjU3MDksMTIuOTExMzc3IDI1LDEyLjk3MDcwMyBMMjUsMy4wMzcxMDk0IFogTTE4LjE3MTg3NSwzLjc0MDIzNDQgQzE3LjA4NjQ4MSw0Ljg3NzI4NDUgMTYuMTE4NjM3LDYuMjk5ODM0NCAxNS4yOTg4MjgsNy45Mzk0NTMxIEMxNC43Mjc1MDIsOS4wODIxMDYgMTQuMjI2MzY2LDEwLjM0MjcxIDEzLjc4OTA2MiwxMS42ODc1IEMxMS44MjU5MzYsMTEuMTU4Mjc5IDEwLjA1MDU4NywxMC40OTIzNjEgOC41MTc1NzgxLDkuNzE4NzUgQzExLjA5NzUwMSw2LjkyMjcxNTEgMTQuNDExMDczLDQuODE4MDEwOSAxOC4xNzE4NzUsMy43NDAyMzQ0IFogTTI5LjgyODEyNSwzLjc0MDIzNDQgQzMzLjU4NTI4OSw0LjgxNjk2ODEgMzYuODk1NzM3LDYuOTE4OTYzNiAzOS40NzQ2MDksOS43MTA5MzggQzM3Ljk2NDI1LDEwLjQ5ODY2OCAzNi4xOTA4NjgsMTEuMTcyMDk4IDM0LjIxNjc5NywxMS43MDMxMjUgQzMzLjc3ODM1MywxMC4zNTI0MDkgMzMuMjc0NzEyLDkuMDg2NTM0IDMyLjcwMTE3Miw3LjkzOTQ1MzEgQzMxLjg4MTM2Myw2LjI5OTgzNDQgMzAuOTEzNTE5LDQuODc3Mjg0NSAyOS44MjgxMjUsMy43NDAyMzQ0IFogTTQwLjc4MzIwMywxMS4yNzM0MzggQzQzLjI4MDMxOSwxNC41NjMyNTQgNDQuODQ5NTkxLDE4LjU5NjU0NCA0NS4wNTQ2ODgsMjMgTDM2LjAxMzY3MiwyMyBDMzUuOTQwNjg2LDE5LjY0MjY5NyAzNS41MTE1ODEsMTYuNDcyODQzIDM0Ljc3NzM0NCwxMy42MzI4MTIgQzM3LjAyMTE2MiwxMy4wMjU3ODggMzkuMDQzNTY0LDEyLjIzMDM1NyA0MC43ODMyMDMsMTEuMjczNDM4IFogTTcuMjA1MDc4MSwxMS4yODkwNjIgQzguOTYzNTM2MiwxMi4yMjI3NTIgMTAuOTg5MzAxLDEzLjAwODc5IDEzLjIyNjU2MiwxMy42MTUyMzQgQzEyLjQ4OTYzMywxNi40NTk2NzEgMTIuMDU5NDYyLDE5LjYzNTkwNCAxMS45ODYzMjgsMjMgTDIuOTQ1MzEyNSwyMyBDMy4xNTAwODU2LDE4LjYwMzQ4NSA0LjcxNDg3MjcsMTQuNTc2MDc4IDcuMjA1MDc4MSwxMS4yODkwNjIgWiBNMTUuMTc1NzgxLDE0LjA4NTkzOCBDMTcuNjA4MTI0LDE0LjYwMzQ3OSAyMC4yMzcxNDUsMTQuOTExNjkyIDIzLDE0Ljk3MjY1NiBMMjMsMjMgTDEzLjk4NjMyOCwyMyBDMTQuMDYwNzI1LDE5Ljc4NzM2OSAxNC40ODA3NDMsMTYuNzYyMjcxIDE1LjE3NTc4MSwxNC4wODU5MzggWiBNMzIuODI4MTI1LDE0LjA5OTYwOSBDMzMuNTIxMDg4LDE2Ljc3MjYgMzMuOTM5NDAxLDE5Ljc5Mjc5NiAzNC4wMTM2NzIsMjMgTDI1LDIzIEwyNSwxNC45NzI2NTYgQzI3Ljc2NDQ1NywxNC45MTMzOTMgMzAuMzk2NDc3LDE0LjYxMjI3MSAzMi44MjgxMjUsMTQuMDk5NjA5IFogTTIuOTQ1MzEyNSwyNSBMMTEuOTg2MzI4LDI1IEMxMi4wNTkzMTQsMjguMzU3MzAzIDEyLjQ4ODQxOSwzMS41MjcxNTYgMTMuMjIyNjU2LDM0LjM2NzE4OCBDMTAuOTc4ODM4LDM0Ljk3NDIxMiA4Ljk1NjQzNjMsMzUuNzY5NjQzIDcuMjE2Nzk2OSwzNi43MjY1NjIgQzQuNzE5NjgwNiwzMy40MzY3NDYgMy4xNTA0MDg4LDI5LjQwMzQ1NiAyLjk0NTMxMjUsMjUgWiBNMTMuOTg2MzI4LDI1IEwyMywyNSBMMjMsMzMuMDI3MzQ0IEMyMC4yMzU1NDMsMzMuMDg2NjA3IDE3LjYwMzUyMywzMy4zODc3MjkgMTUuMTcxODc1LDMzLjkwMDM5MSBDMTQuNDc4OTEyLDMxLjIyNzQgMTQuMDYwNTk5LDI4LjIwNzIwNCAxMy45ODYzMjgsMjUgWiBNMjUsMjUgTDM0LjAxMzY3MiwyNSBDMzMuOTM5Mjc1LDI4LjIxMjYzMSAzMy41MTkyNTcsMzEuMjM3NzI5IDMyLjgyNDIxOSwzMy45MTQwNjIgQzMwLjM5MTg3NiwzMy4zOTY1MjEgMjcuNzYyODU1LDMzLjA4ODMwOCAyNSwzMy4wMjczNDQgTDI1LDI1IFogTTM2LjAxMzY3MiwyNSBMNDUuMDU0Njg4LDI1IEM0NC44NDk5MTQsMjkuMzk2NTE1IDQzLjI4NTEyNywzMy40MjM5MjIgNDAuNzk0OTIyLDM2LjcxMDkzOCBDMzkuMDM2NDY0LDM1Ljc3NzI0OCAzNy4wMTA2OTksMzQuOTkxMjEgMzQuNzczNDM4LDM0LjM4NDc2NiBDMzUuNTEwMzY3LDMxLjU0MDMyOSAzNS45NDA1MzgsMjguMzY0MDk2IDM2LjAxMzY3MiwyNSBaIE0yMywzNS4wMjkyOTcgTDIzLDQ0Ljk2Mjg5MSBDMjAuODM3Mjg2LDQ0LjQ4Mzk1IDE4Ljc0MDgyNSw0Mi40NzE4ODQgMTcuMDg3ODkxLDM5LjE2NjAxNiBDMTYuNTg1NTA0LDM4LjE2MTI0MyAxNi4xMzM2MjEsMzcuMDQxMTk0IDE1LjczNjMyOCwzNS44MzU5MzggQzE3Ljk4NDczMSwzNS4zNjk5NjMgMjAuNDI5MSwzNS4wODg2MjMgMjMsMzUuMDI5Mjk3IFogTTI1LDM1LjAyOTI5NyBDMjcuNTY3NDksMzUuMDg5ODkgMzAuMDA4NTU1LDM1LjM3Njk3OCAzMi4yNTk3NjYsMzUuODQ3NjU2IEMzMS44NjMzNzMsMzcuMDQ4NSAzMS40MTI4OTcsMzguMTY0NDQgMzAuOTEyMTA5LDM5LjE2NjAxNiBDMjkuMjU5MTc1LDQyLjQ3MTg4NCAyNy4xNjI3MTQsNDQuNDgzOTUgMjUsNDQuOTYyODkxIEwyNSwzNS4wMjkyOTcgWiBNMTMuNzgzMjAzLDM2LjI5Njg3NSBDMTQuMjIxNjQ3LDM3LjY0NzU5MSAxNC43MjUyODgsMzguOTEzNDY2IDE1LjI5ODgyOCw0MC4wNjA1NDcgQzE2LjExODYzNyw0MS43MDAxNjYgMTcuMDg2NDgxLDQzLjEyMjcxNiAxOC4xNzE4NzUsNDQuMjU5NzY2IEMxNC40MTQ3MTEsNDMuMTgzMDMyIDExLjEwNDI2Myw0MS4wODEwMzYgOC41MjUzOTA2LDM4LjI4OTA2MiBDMTAuMDM1NzUsMzcuNTAxMzMyIDExLjgwOTEzMiwzNi44Mjc5MDIgMTMuNzgzMjAzLDM2LjI5Njg3NSBaIE0zNC4yMTA5MzgsMzYuMzEyNSBDMzYuMTc0MDY0LDM2Ljg0MTcyMSAzNy45NDk0MTMsMzcuNTA3NjM5IDM5LjQ4MjQyMiwzOC4yODEyNSBDMzYuOTAyNDk5LDQxLjA3NzI4NSAzMy41ODg5MjcsNDMuMTgxOTg5IDI5LjgyODEyNSw0NC4yNTk3NjYgQzMwLjkxMzUxOSw0My4xMjI3MTYgMzEuODgxMzYzLDQxLjcwMDE2NiAzMi43MDExNzIsNDAuMDYwNTQ3IEMzMy4yNzI0OTgsMzguOTE3ODk0IDMzLjc3MzYzNCwzNy42NTcyOSAzNC4yMTA5MzgsMzYuMzEyNSBaIiBpZD0iU2hhcGUiIC8+CiAgICAgICAgICAgIDwvZz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_within_spec()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">email</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">email</td>
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
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


The validation table shows that one test unit failed (the invalid email address in row 3).
