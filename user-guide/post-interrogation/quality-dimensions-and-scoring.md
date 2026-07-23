# Quality Dimensions & Scoring

Beyond knowing *which* validation steps passed or failed, it's often useful to understand data quality along broad, well-understood **dimensions** (completeness, validity, uniqueness, and so on) and to roll everything up into a single **health score** that a governance stakeholder can track over time.

Pointblank does this automatically. Every validation step is tagged with a data quality dimension (inferred from what the step checks), and after interrogation you can obtain per-dimension scores and an overall health score. Nothing about how checks *run* changes; this is a labeling and aggregation layer over results you already have.


# The Six Dimensions

Each validation step belongs to one of six data quality dimensions:

| Dimension | What it measures | Example methods |
|----|----|----|
| **Completeness** | Presence of required values | [col_vals_not_null()](../../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null), [col_pct_null()](../../reference/Validate.col_pct_null.md#pointblank.Validate.col_pct_null), [rows_complete()](../../reference/Validate.rows_complete.md#pointblank.Validate.rows_complete) |
| **Validity** | Values conform to rules, ranges, formats, or schema | [col_vals_gt()](../../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt), [col_vals_regex()](../../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex), [col_vals_in_set()](../../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set), [col_schema_match()](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) |
| **Uniqueness** | Absence of duplicate rows | [rows_distinct()](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) |
| **Consistency** | Internal agreement across columns, rows, or tables | [conjointly()](../../reference/Validate.conjointly.md#pointblank.Validate.conjointly), [col_missing_consistent()](../../reference/Validate.col_missing_consistent.md#pointblank.Validate.col_missing_consistent), [tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match) |
| **Timeliness** | Data recency / freshness | [data_freshness()](../../reference/Validate.data_freshness.md#pointblank.Validate.data_freshness) |
| **Volume** | Expected row and column counts | [row_count_match()](../../reference/Validate.row_count_match.md#pointblank.Validate.row_count_match), [col_count_match()](../../reference/Validate.col_count_match.md#pointblank.Validate.col_count_match) |

The dimension for each step is inferred from its assertion type, so existing validations gain dimensions with no changes on your part.


# Dimensions in the Validation Report

The dimension display in the validation report is **opt-in**: pass `incl_dimensions=True` to <a href="../../reference/Validate.get_tabular_report.html#pointblank.Validate.get_tabular_report" class="gdls-link"><code>get_tabular_report()</code></a> (or set it globally with `pb.config(report_incl_dimensions=True)`). Consider a validation of some sales data that touches several dimensions:


``` python
import pointblank as pb
import polars as pl

sales = pl.DataFrame({
    "order_id": [1, 2, 3, 4, 5, 6, 7, 8, 8, 10],   # one duplicate (8)
    "amount":   [120.0, -5.0, 47.5, 0.0, 30.0, 155.0, 175.0, 95.0, 205.0, None],
    "email":    ["a@ex.com", "invalid", "c@ex.com", "d@ex.com", "nope",
                 "f@ex.com", "g@ex.com", "h@ex.com", "i@ex.com", "j@ex.com"],
    "status":   ["paid", "paid", "refund", "paid", "pending",
                 "paid", "paid", "refund", "paid", "paid"],
})

validation = (
    pb.Validate(data=sales, tbl_name="sales", label="Sales data quality")
    .col_vals_not_null(columns="amount")                                # Completeness
    .col_vals_gt(columns="amount", value=0)                             # Validity
    .col_vals_regex(columns="email", pattern=r"^[^@]+@[^@]+\.[^@]+$")   # Validity
    .col_vals_in_set(columns="status", set=["paid", "refund", "pending"])  # Validity
    .rows_distinct(columns_subset=["order_id"])                         # Uniqueness
    .row_count_match(count=10)                                          # Volume
    .interrogate()
)

validation.get_tabular_report(incl_dimensions=True)
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">Sales data quality</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">Polars</span><span style="background-color: none; color: #222222; padding: 0.5em 0.5em; position: inherit; margin: 5px 10px 5px -4px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 15px 2px 15px; font-size: 10px;">sales</span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_tbl-status_color" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
<th id="pb_tbl-i" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col"></th>
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
<td class="gt_row gt_left" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold"><div style="position: relative; text-align: right;">
<span title="Completeness" style="position: absolute; top: -11px; left: 3px; background-color: #3C6E9A; color: #FFFFFF; font-size: 7px; font-weight: 700; padding: 1px 3px; border-radius: 3px; line-height: 1; letter-spacing: 0.5px; font-family: 'IBM Plex Sans', sans-serif;">CM</span>1
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0ib21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0ibGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">amount</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">10</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">9<br />
0.90</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
0.10</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_left" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold"><div style="position: relative; text-align: right;">
<span title="Validity" style="position: absolute; top: -11px; left: 3px; background-color: #4E9A6B; color: #FFFFFF; font-size: 7px; font-weight: 700; padding: 1px 3px; border-radius: 3px; line-height: 1; letter-spacing: 0.5px; font-family: 'IBM Plex Sans', sans-serif;">VA</span>2
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">amount</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">10</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">7<br />
0.70</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">3<br />
0.30</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_left" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold"><div style="position: relative; text-align: right;">
<span title="Validity" style="position: absolute; top: -11px; left: 3px; background-color: #4E9A6B; color: #FFFFFF; font-size: 7px; font-weight: 700; padding: 1px 3px; border-radius: 3px; line-height: 1; letter-spacing: 0.5px; font-family: 'IBM Plex Sans', sans-serif;">VA</span>3
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfcmVnZXg8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfcmVnZXgiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjAzNDQ4MykiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPGcgaWQ9InJlZ2V4X3N5bWJvbHMiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4LjAwMDAwMCwgMTIuMDAwMDAwKSIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik00LjE3NDM0NTA4LDMzLjAxMzU4MiBDMS45NDg5NTMyOCwzMy4wMTM1ODIgMC4xMzgwMDY5MjMsMzQuODI0NTI4NCAwLjEzODAwNjkyMywzNy4wNDk5MjAyIEMwLjEzODAwNjkyMywzOS4yNzUzMTIgMS45NDg5NTMyOCw0MS4wODYyNTgzIDQuMTc0MzQ1MDgsNDEuMDg2MjU4MyBDNi4zOTk3MzY4OCw0MS4wODYyNTgzIDguMjEwNjgzMjQsMzkuMjc1MzEyIDguMjEwNjgzMjQsMzcuMDQ5OTIwMiBDOC4yMTA2ODMyNCwzNC44MjQ1Mjg0IDYuMzk5NzM2ODgsMzMuMDEzNTgyIDQuMTc0MzQ1MDgsMzMuMDEzNTgyIFoiIGlkPSJmdWxsX3N0b3AiIC8+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMjMuOTQ3OTcxOCwyMy4zMTc1NDAyIEwyMS41NjI4MjY0LDIzLjMxNzU0MDIgQzIxLjIzNDQwMzIsMjMuMzE3NTQwMiAyMC45NjY1NDAxLDIzLjA1MjAwNjcgMjAuOTY2NTQwMSwyMi43MjEyNTM4IEwyMC45NjY1NDAxLDE1LjEwMjI5NzkgTDE0LjM0NDUwMDQsMTguODg3MzE5MiBDMTQuMDYyNjYyMSwxOS4wNTAzNjYgMTMuNzAxNjI5MiwxOC45NTI1MzggMTMuNTM2MjUzMywxOC42NzA2OTkxIEwxMi4zNDM2ODA2LDE2LjY0NDI1NzUgQzEyLjI2MjE1NywxNi41MDY4MzIgMTIuMjM4ODY0MiwxNi4zNDM3ODUyIDEyLjI4MDc5MDksMTYuMTkwMDU0OSBDMTIuMzIwMzg3OSwxNi4wMzYzMjUxIDEyLjQyMDU0NTUsMTUuOTA1ODg3NCAxMi41NTc5NzEsMTUuODI2NjkyOSBMMTkuMTgwMDEwMSwxMS45ODgwOTk0IEwxMi41NTc5NzEsOC4xNTE4MzUxMSBDMTIuNDIwNTQ1NSw4LjA3MjY0MTEyIDEyLjMyMDM4NzksNy45Mzk4NzQzOSAxMi4yODA3OTA5LDcuNzg2MTQ0MDEgQzEyLjIzODg2NDIsNy42MzI0MTQyMyAxMi4yNjIxNTcsNy40NjkzNjY4OSAxMi4zNDEzNTA5LDcuMzMxOTQxMzcgTDEzLjUzMzkyMzcsNS4zMDU0OTk3NSBDMTMuNjk5MzAwMSw1LjAyMzY2MTQzIDE0LjA2MjY2MjEsNC45MjgxNjE5OSAxNC4zNDQ1MDA0LDUuMDkxMjA5MzQgTDIwLjk2NjU0MDEsOC44NzM5MDA5MSBMMjAuOTY2NTQwMSwxLjI1NDk0NTAxIEMyMC45NjY1NDAxLDAuOTI2NTIxODE4IDIxLjIzNDQwMzIsMC42NTg2NTg2NTggMjEuNTYyODI2NCwwLjY1ODY1ODY1OCBMMjMuOTQ3OTcxOCwwLjY1ODY1ODY1OCBDMjQuMjc4NzI0NywwLjY1ODY1ODY1OCAyNC41NDQyNTgyLDAuOTI2NTIxODE4IDI0LjU0NDI1ODIsMS4yNTQ5NDUwMSBMMjQuNTQ0MjU4Miw4Ljg3MzkwMDkxIEwzMS4xNjYyOTc5LDUuMDkxMjA5MzQgQzMxLjQ0ODEzNjIsNC45MjgxNjE5OSAzMS44MDkxNjkxLDUuMDIzNjYxNDMgMzEuOTc0NTQ1NSw1LjMwNTQ5OTc1IEwzMy4xNjcxMTgyLDcuMzMxOTQxMzcgQzMzLjI0ODY0MTMsNy40NjkzNjY4OSAzMy4yNzE5MzQxLDcuNjMyNDE0MjMgMzMuMjMwMDA3NCw3Ljc4NjE0NDAxIEMzMy4xOTA0MTA0LDcuOTM5ODc0MzkgMzMuMDkwMjUyOCw4LjA3MjY0MTEyIDMyLjk1MjgyNzgsOC4xNTE4MzUxMSBMMjYuMzMwNzg4MiwxMS45ODgwOTk0IEwzMi45NTI4Mjc4LDE1LjgyNDM2MzggQzMzLjA4NzkyMzcsMTUuOTA1ODg3NCAzMy4xODgwODEzLDE2LjAzNjMyNTEgMzMuMjMwMDA3NCwxNi4xOTAwNTQ5IEMzMy4yNjk2MDUsMTYuMzQzNzg1MiAzMy4yNDg2NDEzLDE2LjUwNjgzMiAzMy4xNjcxMTgyLDE2LjY0NDI1NzUgTDMxLjk3NDU0NTUsMTguNjcwNjk5MSBDMzEuODA5MTY5MSwxOC45NTI1MzggMzEuNDQ4MTM2MiwxOS4wNTAzNjYgMzEuMTY2Mjk3OSwxOC44ODQ5ODk1IEwyNC41NDQyNTgyLDE1LjEwMjI5NzkgTDI0LjU0NDI1ODIsMjIuNzIxMjUzOCBDMjQuNTQ0MjU4MiwyMy4wNTIwMDY3IDI0LjI3ODcyNDcsMjMuMzE3NTQwMiAyMy45NDc5NzE4LDIzLjMxNzU0MDIgWiIgaWQ9ImFzdGVyaXNrIiAvPgogICAgICAgICAgICA8L2c+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_regex()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">email</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">^[^@]+@[^@]+\.[^@]+$</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">10</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">8<br />
0.80</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.20</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_left" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold"><div style="position: relative; text-align: right;">
<span title="Validity" style="position: absolute; top: -11px; left: 3px; background-color: #4E9A6B; color: #FFFFFF; font-size: 7px; font-weight: 700; padding: 1px 3px; border-radius: 3px; line-height: 1; letter-spacing: 0.5px; font-family: 'IBM Plex Sans', sans-serif;">VA</span>4
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfaW5fc2V0PC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9ImNvbF92YWxzX2luX3NldCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuMTcyNDE0KSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMSBDNTkuMTk3NTE1MywxIDYxLjQ0NzUxNTMsMi4wMDczNTkzMSA2My4wNzYxOTUsMy42MzYwMzg5NyBDNjQuNzA0ODc0Nyw1LjI2NDcxODYzIDY1LjcxMjIzNCw3LjUxNDcxODYzIDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsNjUgTDEwLjcxMjIzNCw2NSBDOC4yMjY5NTI1OSw2NSA1Ljk3Njk1MjU5LDYzLjk5MjY0MDcgNC4zNDgyNzI5NCw2Mi4zNjM5NjEgQzIuNzE5NTkzMjgsNjAuNzM1MjgxNCAxLjcxMjIzMzk3LDU4LjQ4NTI4MTQgMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5Nyw1NiBMMS43MTIyMzM5NywxMCBDMS43MTIyMzM5Nyw3LjUxNDcxODYzIDIuNzE5NTkzMjgsNS4yNjQ3MTg2MyA0LjM0ODI3Mjk0LDMuNjM2MDM4OTcgQzUuOTc2OTUyNTksMi4wMDczNTkzMSA4LjIyNjk1MjU5LDEgMTAuNzEyMjM0LDEgTDEwLjcxMjIzNCwxIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDQuMTI3OTY5LDQxLjE1MzgzODIgTDMxLjA4MTQ1NjgsNDEuMTUzODM4MiBDMjkuOTUxMDc0OCw0MS4xNTM2NDI5IDI4Ljg4MjcwNTIsNDAuOTI1NjEzNCAyNy45MDc5ODg4LDQwLjUxMzY5NTMgQzI2LjQ0Njc0NDIsMzkuODk2MDEzNiAyNS4xOTg0OSwzOC44NTk5Njg1IDI0LjMxODk4OTQsMzcuNTU3NzA5OSBDMjMuODc5MjM5MSwzNi45MDY3MjcgMjMuNTMxNDgxOCwzNi4xODk5MjMzIDIzLjI5MzY4NjYsMzUuNDI1MjY3NSBDMjMuMjEzMDIxNywzNS4xNjU4OSAyMy4xNDYwMjg5LDM0LjkwMDU1NTQgMjMuMDkxMzQwOSwzNC42MzA3Mjg2IEw0NC4xMjc4NzE0LDM0LjYzMDcyODYgQzQ1LjAyODQ2NiwzNC42MzA2MzA5IDQ1Ljc1ODY0ODgsMzMuOTAwNDQ4MSA0NS43NTg2NDg4LDMyLjk5OTg1MzUgQzQ1Ljc1ODY0ODgsMzIuMDk5MjU4OSA0NS4wMjg0NjYsMzEuMzY5MDc2MSA0NC4xMjc4NzE0LDMxLjM2OTA3NjEgTDIzLjA5MDU1OTYsMzEuMzY5MDc2MSBDMjMuMTk5MDU2NywzMC44MzM3MTk0IDIzLjM1OTcwMjgsMzAuMzE4MDg5NCAyMy41Njc1MTczLDI5LjgyNjQ4MzEgQzI0LjE4NTE5OSwyOC4zNjUyMzg2IDI1LjIyMTI0NDIsMjcuMTE2OTg0NCAyNi41MjM2MDA0LDI2LjIzNzQ4MzggQzI3LjE3NDU4MzMsMjUuNzk3NzMzNCAyNy44OTEzODcsMjUuNDQ5OTc2MiAyOC42NTYwNDI4LDI1LjIxMjI3ODYgQzI5LjQyMDg5MzksMjQuOTc0NDgzMyAzMC4yMzM0OTk0LDI0Ljg0NTk2NjUgMzEuMDgxMzU5MSwyNC44NDU5NjY1IEw0NC4xMjc3NzM3LDI0Ljg0NTk2NjUgQzQ1LjAyODM2ODMsMjQuODQ1OTY2NSA0NS43NTg2NDg4LDI0LjExNTc4MzcgNDUuNzU4NjQ4OCwyMy4yMTUxODkxIEM0NS43NTg2NDg4LDIyLjMxNDU5NDUgNDUuMDI4MzY4MywyMS41ODQ0MTE3IDQ0LjEyNzc3MzcsMjEuNTg0NDExNyBMMzEuMDgxMzU5MSwyMS41ODQ0MTE3IEMyOS41MDk2NjQzLDIxLjU4NDQxMTcgMjguMDAzOTg1OCwyMS45MDM4NDgzIDI2LjYzNzM3MTEsMjIuNDgyMDc2NSBDMjQuNTg2NjY3OCwyMy4zNDk4NTgzIDIyLjg0NjkwNDksMjQuNzk1MDg3MSAyMS42MTYzMjY3LDI2LjYxNjI5NiBDMjAuMzg1NjUwOCwyOC40MzYyMzU0IDE5LjY2NTEzNiwzMC42NDEzMzQ3IDE5LjY2NTgxOTEsMzMuMDAwMDQ4OCBDMTkuNjY1ODE5MSwzNC41NzE3NDM2IDE5Ljk4NTI1NjMsMzYuMDc3NDIyMiAyMC41NjM1ODIyLDM3LjQ0NDAzNjkgQzIxLjQzMTI2NjMsMzkuNDk0NzQwMiAyMi44NzY1OTI3LDQxLjIzNDUwMzEgMjQuNjk3NzA0LDQyLjQ2NTA4MTMgQzI2LjUxNzY0MzQsNDMuNjk1NzU3MiAyOC43MjI3NDI3LDQ0LjQxNTU4ODMgMzEuMDgxNDU2OCw0NC40MTU1ODgzIEw0NC4xMjc4NzE0LDQ0LjQxNTU4ODMgQzQ1LjAyODQ2Niw0NC40MTU1ODgzIDQ1Ljc1ODY0ODgsNDMuNjg1NDA1NSA0NS43NTg2NDg4LDQyLjc4NDgxMDkgQzQ1Ljc1ODY0ODgsNDEuODg0MjE2MyA0NS4wMjg1NjM2LDQxLjE1MzgzODIgNDQuMTI3OTY5LDQxLjE1MzgzODIgWiIgaWQ9InNldF9vZiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

col_vals_in_set()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">status</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">paid, refund, pending</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">10</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">10<br />
1.00</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">0<br />
0.00</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">--</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_left" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold"><div style="position: relative; text-align: right;">
<span title="Uniqueness" style="position: absolute; top: -11px; left: 3px; background-color: #8E6FB5; color: #FFFFFF; font-size: 7px; font-weight: 700; padding: 1px 3px; border-radius: 3px; line-height: 1; letter-spacing: 0.5px; font-family: 'IBM Plex Sans', sans-serif;">UQ</span>5
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+cm93c19kaXN0aW5jdDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJyb3dzX2Rpc3RpbmN0IiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjAwMDAwMCwgMC40ODI3NTkpIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU2LjcxMjIzNCwxIEM1OS4xOTc1MTUzLDEgNjEuNDQ3NTE1MywyLjAwNzM1OTMxIDYzLjA3NjE5NSwzLjYzNjAzODk3IEM2NC43MDQ4NzQ3LDUuMjY0NzE4NjMgNjUuNzEyMjM0LDcuNTE0NzE4NjMgNjUuNzEyMjM0LDEwIEw2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCw2NSBMMTAuNzEyMjM0LDY1IEM4LjIyNjk1MjU5LDY1IDUuOTc2OTUyNTksNjMuOTkyNjQwNyA0LjM0ODI3Mjk0LDYyLjM2Mzk2MSBDMi43MTk1OTMyOCw2MC43MzUyODE0IDEuNzEyMjMzOTcsNTguNDg1MjgxNCAxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDU2IEwxLjcxMjIzMzk3LDEwIEMxLjcxMjIzMzk3LDcuNTE0NzE4NjMgMi43MTk1OTMyOCw1LjI2NDcxODYzIDQuMzQ4MjcyOTQsMy42MzYwMzg5NyBDNS45NzY5NTI1OSwyLjAwNzM1OTMxIDguMjI2OTUyNTksMSAxMC43MTIyMzQsMSBMMTAuNzEyMjM0LDEgWiIgaWQ9InJlY3RhbmdsZSIgc3Ryb2tlPSIjMDAwMDAwIiBzdHJva2Utd2lkdGg9IjIiIGZpbGw9IiNGRkZGRkYiIC8+CiAgICAgICAgICAgIDxnIGlkPSJub19nZW1pbmkiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE3LjAwMDAwMCwgMTMuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMy42NjcwNTYxOSw2LjYyMTA3NTA4IEMzLjEyNTEwMTA0LDYuNjQwNjYzODYgMi42NzQ1NTk3NCw3LjA0NzY3NDQ0IDIuNjAyNzM0MzIsNy41ODUyNzY4MiBDMi41Mjg3MzIyOCw4LjEyMjg3OTIgMi44NTMwMzUyNiw4LjYzNDM2MjcgMy4zNzEwNDg0OCw4Ljc5NzYwMjM5IEM0LjQwNDg5OTA5LDkuMTU0NTUyODYgNi43MDExMzU1Myw5Ljg3MDYzMDIxIDkuODY1ODA1OTUsMTAuMzY0NzAyIEw5Ljg2NTgwNTk1LDMwLjczNjk5NzYgQzYuNjU5NzgxMzcsMzEuMjMzMjQ1OCA0LjM3MjI1MTA0LDMxLjk2NDU1OSAzLjM3MTA0ODQ4LDMyLjMyMTUwOTUgQzIuNzg5OTE1NTUsMzIuNTI4Mjc5NyAyLjQ4NTIwMTczLDMzLjE2ODE3ODQgMi42OTE5NzE5NiwzMy43NDkzMTE0IEMyLjg5ODc0MjIsMzQuMzMwNDQ0MyAzLjUzODY0MDk0LDM0LjYzNTE1ODEgNC4xMTk3NzM4NywzNC40MjgzODc5IEM1LjU0OTc1MjE3LDMzLjkxOTA4MDggMTAuMjAzMTY3OCwzMi40NjA4MDcyIDE2LjUxNzI3MzQsMzIuNDYwODA3MiBDMjIuNzk0Mzc4MSwzMi40NjA4MDcyIDI3LjU1MDA5MDEsMzMuODkwNzg1NSAyOS4wNTQwNzA2LDM0LjQxMDk3NTcgQzI5LjYzNTIwMzYsMzQuNjEzMzkyNiAzMC4yNzA3NDk1LDM0LjMwNDMyNiAzMC40NzMxNjY0LDMzLjcyMzE5MzEgQzMwLjY3NTU4MzMsMzMuMTQyMDYwMSAzMC4zNjY1MTY3LDMyLjUwNjUxNDIgMjkuNzg1MzgzOCwzMi4zMDQwOTczIEMyOC43NDkzNTY4LDMxLjk0NDk3MDQgMjYuNDMxMzU1NCwzMS4yNDQxMjgzIDIzLjIzODM4OTcsMzAuNzU0NDA5OCBMMjMuMjM4Mzg5NywxMC4zODIxMTQzIEMyNi40NDQ0MTQzLDkuODg4MDQyNDMgMjguNzQ1MDA0LDkuMTYxMDgyNTkgMjkuNzY3OTcxNiw4Ljc5NzYwMjM5IEMzMC4zNDkxMDQ1LDguNTkwODMyMTUgMzAuNjUzODE4NCw3Ljk1MDkzMzQxIDMwLjQ0NzA0ODEsNy4zNjk4MDA0OCBDMzAuMjQwMjc3OSw2Ljc4ODY2NzU1IDI5LjYwMDM3OTEsNi40ODM5NTM3MyAyOS4wMTkyNDYyLDYuNjkwNzIzOTYgQzI3LjU1ODc5NjMsNy4yMDg3Mzc3NCAyMi45MTYyNjM3LDguNjU4MzA0NjQgMTYuNjM5MTU4OSw4LjY1ODMwNDY0IEMxMC4zNzA3NjAzLDguNjU4MzA0NjQgNS42MTI4NzE4OCw3LjIxMzA5MDUxIDQuMTAyMzYxNjYsNi42OTA3MjM5NiBDMy45NjMwNjM5MSw2LjYzODQ4NzMgMy44MTUwNjAwNSw2LjYxNDU0NTQ4IDMuNjY3MDU2MTksNi42MjEwNzUwOCBaIE0xMi4wOTQ1Njk5LDEwLjY0MzI5NzUgQzEzLjQ5NDA3NzEsMTAuNzg5MTI1IDE1LjAxOTgyMjYsMTAuODg3MDY4NiAxNi42MzkxNTg5LDEwLjg4NzA2ODYgQzE4LjE5OTcyODksMTAuODg3MDY4NiAxOS42NjIzNTUyLDEwLjc5NzgzMTEgMjEuMDA5NjI1NywxMC42NjA3MDk4IEwyMS4wMDk2MjU3LDMwLjQ1ODQwMjEgQzE5LjYyMzE3OCwzMC4zMTY5MjggMTguMTE5MTk3NSwzMC4yMzIwNDMzIDE2LjUxNzI3MzQsMzAuMjMyMDQzMyBDMTQuOTMyNzYxNSwzMC4yMzIwNDMzIDEzLjQ1NzA3NjMsMzAuMzE5MTA0NCAxMi4wOTQ1Njk5LDMwLjQ1ODQwMjEgTDEyLjA5NDU2OTksMTAuNjQzMjk3NSBaIiBpZD0iZ2VtaW5pIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiAvPgogICAgICAgICAgICAgICAgPHBhdGggZD0iTTE2LjY2MDUzNTQsLTUuMDU5Mjk0OTkgQzE2LjkzNjY3NzgsLTUuMDU5Mjk0OTkgMTcuMTYwNTM1NCwtNC44MzU0MzczNyAxNy4xNjA1MzU0LC00LjU1OTI5NDk5IEwxNy4xNjA1MzU0LDQ0LjU2ODc1MDkgQzE3LjE2MDUzNTQsNDQuODQ0ODkzMyAxNi45MzY2Nzc4LDQ1LjA2ODc1MDkgMTYuNjYwNTM1NCw0NS4wNjg3NTA5IEMxNi4zODQzOTMsNDUuMDY4NzUwOSAxNi4xNjA1MzU0LDQ0Ljg0NDg5MzMgMTYuMTYwNTM1NCw0NC41Njg3NTA5IEwxNi4xNjA1MzU0LC00LjU1OTI5NDk5IEMxNi4xNjA1MzU0LC00LjgzNTQzNzM3IDE2LjM4NDM5MywtNS4wNTkyOTQ5OSAxNi42NjA1MzU0LC01LjA1OTI5NDk5IFoiIGlkPSJsaW5lX2JsYWNrIiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNi42NjA1MzUsIDIwLjAwNDcyOCkgcm90YXRlKC0zMjAuMDAwMDAwKSB0cmFuc2xhdGUoLTE2LjY2MDUzNSwgLTIwLjAwNDcyOCkgIiAvPgogICAgICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE4LjU2MDAzMiwgMTkuMTU4MDMxKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMTguNTYwMDMyLCAtMTkuMTU4MDMxKSAiIHBvaW50cz0iMTguMDYwMDMxNiAtNC40NTM2NjczNSAxOS4wNjAwMzE2IC00LjQ1MzY2NzM1IDE5LjA2MDAzMTYgNDIuNzY5NzI5OSAxOC4wNjAwMzE2IDQyLjc2OTcyOTkiPjwvcG9seWdvbj4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

rows_distinct()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">order_id</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">10</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">8<br />
0.80</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.20</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C; color: transparent; font-size: 0px">#4CA64C</td>
<td class="gt_row gt_left" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold"><div style="position: relative; text-align: right;">
<span title="Volume" style="position: absolute; top: -11px; left: 3px; background-color: #7A7A8C; color: #FFFFFF; font-size: 7px; font-weight: 700; padding: 1px 3px; border-radius: 3px; line-height: 1; letter-spacing: 0.5px; font-family: 'IBM Plex Sans', sans-serif;">VO</span>6
</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+cm93X2NvdW50X21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InJvd19jb3VudF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNzkzMTAzKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNMjkuMTY0NjE0OSwyOC40ODUzMzA2IEwyOS4xNjQ2MTQ5LDY5LjQ4NTMzMDYgTDI3LjA1NDkwODQsNjkuNDg1MDQ5MyBDMjYuOTI5NjE3OSw2OS40ODI2OTQ0IDI2LjYyOTUwODUsNjkuNDYwNjY0NSAyNi40MTYzNzA4LDY5LjI1NDg3NjUgQzI2LjI4ODI0ODEsNjkuMTMxMTcxOCAyNi4yMTIyMzQsNjguOTM0ODI2OSAyNi4yMTIyMzQsNjguNjUxOTk3MyBMMjYuMjEyMjM0LDY4LjY1MTk5NzMgTDI2LjIxMjIzNCwyOS4zMTg2NjQgQzI2LjIxMjIzNCwyOS4wMzU4MzQ0IDI2LjI4ODI0ODEsMjguODM5NDg5NSAyNi40MTYzNzA4LDI4LjcxNTc4NDggQzI2LjY1NjE1MDcsMjguNDg0MjczMyAyNy4wMDYwMDQxLDI4LjQ4NTMzMDYgMjcuMDkzMTg2MywyOC40ODUzMzA2IEwyOS4xNjQ2MTQ5LDI4LjQ4NTMzMDYgWiBNMzUuMTU5OTM5MywyOC40ODUzMzA2IEwzNS4xNTk5MzkzLDY5LjQ4NTMzMDYgTDMyLjI2NDUyODYsNjkuNDg1MzMwNiBMMzIuMjY0NTI4NiwyOC40ODUzMzA2IEwzNS4xNTk5MzkzLDI4LjQ4NTMzMDYgWiBNNDAuMzU5MzQ3NSwyOC40ODU3NTA3IEM0MC40NzMzOTUzLDI4LjQ4ODcyOCA0MC43NjY4MiwyOC41MDQ4MTQ2IDQwLjk4NTM1NTIsMjguNjk0OTQ1MSBDNDEuMTAwODIxNSwyOC43OTU0MDM0IDQxLjE4MTk5OTksMjguOTUxMzY4NCA0MS4yMDUxNzc2LDI5LjE3NDgzMiBMNDEuMjA1MTc3NiwyOS4xNzQ4MzIgTDQxLjIxMjIzNCw2OC42NTE5OTczIEM0MS4yMTIyMzQsNjguOTMwOTgzMyA0MS4xMzg1Mjc3LDY5LjEyNTgzNjkgNDEuMDEzMjgwMyw2OS4yNDk4MDk0IEM0MC43OTQzMDc0LDY5LjQ2NjU1MzUgNDAuNDc4MDQ3LDY5LjQ4MTg3ODcgNDAuMzU5NDEzOSw2OS40ODQ5MTA2IEw0MC4zNTk0MTM5LDY5LjQ4NDkxMDYgTDM4LjI1OTg1Myw2OS40ODUyNDk4IEwzOC4yNTk4NTMsMjguNDg1NDExNCBaIiBpZD0icm93c190d28iIHN0cm9rZT0iIzAwMDAwMCIgZmlsbC1ydWxlPSJub256ZXJvIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzMy43MTIyMzQsIDQ4Ljk4NTMzMSkgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMzMuNzEyMjM0LCAtNDguOTg1MzMxKSAiIC8+CiAgICAgICAgICAgIDxnIGlkPSJ2ZXJ0aWNhbF9lcXVhbCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzAuMDAwMDAwLCAyOS4zODA1NzApIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yNy4wOTMxODYzLC0yLjk4NTMzMDY1IEMyNi44ODM2NjI1LC0yLjk4NTMzMDY1IDI1LjcxMjIzNCwtMi45MzUzMzA2NSAyNS43MTIyMzQsLTEuNjUxOTk3MzEgTDI1LjcxMjIzNCwzNy42ODEzMzYgQzI1LjcxMjIzNCwzOC45NjQ2Njk0IDI2Ljg4MzY2MjUsMzkuMDE0NjY5NCAyNy4wOTMxODYzLDM5LjAxNDY2OTQgTDI5LjY2NDYxNDksMzkuMDE0NjY5NCBMMjkuNjY0NjE0OSwtMi45ODUzMzA2NSBMMjcuMDkzMTg2MywtMi45ODUzMzA2NSBaIE0zNS42NzIxNzcxLC0yLjk4NTMzMDY1IEwzNS42NzIxNzcxLDM5LjAxNDY2OTQgTDMxLjc1MjI5MDgsMzkuMDE0NjY5NCBMMzEuNzUyMjkwOCwzOS4wMTQ2Njk0IEwzMS43NTIyOTA4LC0yLjk4NTMzMDY1IEwzMS43NTIyOTA4LC0yLjk4NTMzMDY1IEwzNS42NzIxNzcxLC0yLjk4NTMzMDY1IFogTTQwLjM2NTYxNDksLTIuOTg0OTA5NiBDNDAuNjQ0ODc4NiwtMi45NzgyMzY5OCA0MS43MTIyMzQsLTIuODc2OTk3MzEgNDEuNzEyMjM0LC0xLjY1MTk5NzMxIEw0MS43MTIyMzQsLTEuNjUxOTk3MzEgTDQxLjcxMjIzNCwzNy42ODEzMzYgQzQxLjcxMjIzNCwzOC45NjQ2Njk0IDQwLjU0MDgwNTQsMzkuMDE0NjY5NCA0MC4zMzEyODE2LDM5LjAxNDY2OTQgTDQwLjMzMTI4MTYsMzkuMDE0NjY5NCBMMzcuNzU5ODUzLDM5LjAxNDY2OTQgTDM3Ljc1OTg1MywtMi45ODUzMzA2NSBaIiBpZD0icm93c19vbmUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuNzEyMjM0LCAxOC4wMTQ2NjkpIHJvdGF0ZSgtOTAuMDAwMDAwKSB0cmFuc2xhdGUoLTMzLjcxMjIzNCwgLTE4LjAxNDY2OSkgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" />

row_count_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">10</td>
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
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">

<span style="font-weight: 600; color: #888888; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; padding-right: 8px; vertical-align: middle;">Health Score</span><span style="display: inline-block; padding: 1px 9px; border-radius: 5px; background-color: #A15C00; color: #FFFFFF; font-weight: 700; font-size: 13px; vertical-align: middle; --pb-scale: 1.076; --pb-glow: 16px; --pb-glow-color: #A15C00; animation: pb-health-pulse 0.95s ease-in-out infinite;">84%</span>

<span style="font-weight: 600; color: #888888; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; padding-right: 8px; vertical-align: middle;">Dimension Scores</span><span style="display: inline-block; margin: 2px 10px 2px 0; white-space: nowrap;"><span style="background-color: #3C6E9A; color: #FFFFFF; padding: 2px 7px; border-radius: 8px; font-size: 10px; font-weight: 600; line-height: 1; display: inline-block; vertical-align: middle;">Completeness</span><span style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #444444; padding-left: 5px; vertical-align: middle;">90%</span></span><span style="display: inline-block; margin: 2px 10px 2px 0; white-space: nowrap;"><span style="background-color: #4E9A6B; color: #FFFFFF; padding: 2px 7px; border-radius: 8px; font-size: 10px; font-weight: 600; line-height: 1; display: inline-block; vertical-align: middle;">Validity</span><span style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #444444; padding-left: 5px; vertical-align: middle;">83.33%</span></span><span style="display: inline-block; margin: 2px 10px 2px 0; white-space: nowrap;"><span style="background-color: #8E6FB5; color: #FFFFFF; padding: 2px 7px; border-radius: 8px; font-size: 10px; font-weight: 600; line-height: 1; display: inline-block; vertical-align: middle;">Uniqueness</span><span style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #444444; padding-left: 5px; vertical-align: middle;">80%</span></span><span style="display: inline-block; margin: 2px 10px 2px 0; white-space: nowrap;"><span style="background-color: #7A7A8C; color: #FFFFFF; padding: 2px 7px; border-radius: 8px; font-size: 10px; font-weight: 600; line-height: 1; display: inline-block; vertical-align: middle;">Volume</span><span style="font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #444444; padding-left: 5px; vertical-align: middle;">100%</span></span>

</div></td>
</tr>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">

<span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin-left: 10px; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-07-22 23:24:43 UTC</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">< 1 s</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 1px 5px -1px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-07-22 23:24:43 UTC</span>
</div></td>
</tr>
</tfoot>

</table>


Two things in the report now relate to dimensions:

- **A dimension badge on each step number.** Each step's number carries a small, color-coded two-letter badge in its top-left corner (`CM` completeness, `VA` validity, `UQ` uniqueness, `CS` consistency, `TM` timeliness, `VO` volume). Hover over a badge to see the full dimension name. The badge is compact by design, so it doesn't widen the report.
- **A health-score summary in the footer.** Below the table you'll find the overall **Health Score** followed by a per-dimension breakdown, with each dimension's color reinforcing the badges above.

The scores themselves (below) are always available programmatically, whether or not the display is enabled.


# Overriding a Step's Dimension

Automatic inference covers the common cases, but you can set a dimension explicitly with the `dimension=` parameter on any validation method. This is useful for multi-faceted checks. For example, treating a particular range check as a *consistency* rule rather than plain validity:


``` python
validation_override = (
    pb.Validate(data=sales, tbl_name="sales")
    .col_vals_gt(columns="amount", value=0, dimension="consistency")
    .interrogate()
)

validation_override.get_dimension_scores()
```


    {'consistency': 70.0}


You can also remap dimensions globally (for every step of a given type) with [`config()`](%60pointblank.config%60):


``` python
pb.config(dimension_map={"col_vals_gt": "consistency"})

# Now `col_vals_gt` steps are categorized as "consistency" everywhere
remapped = (
    pb.Validate(data=sales)
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)
print(remapped.validation_info[0].dimension)

pb.config()  # reset back to defaults
```


    consistency


    PointblankConfig(report_incl_header=True, report_incl_footer=True, report_incl_footer_timings=True, report_incl_footer_notes=True, report_incl_dimensions=False, preview_incl_header=True, dimension_map=None, dimension_weights=None, dimension_thresholds=None)


An explicit `dimension=` on a step always takes precedence over the global map.


# Health Scores

After interrogation, two methods surface the scores:


``` python
validation.get_dimension_scores()
```


    {'completeness': 90.0, 'validity': 83.33, 'uniqueness': 80.0, 'volume': 100.0}


``` python
validation.get_health_score()
```


    84.31


Scores are **test-unit weighted**: a dimension's score is the total number of passing test units divided by the total number of test units across its steps, expressed as a percentage. The overall health score is the same calculation across every step. Because it's weighted by test units (not by step count), the score reflects data volume. A failing check over a large table pulls the score down more than one over a small table.

> **Note: Note**
>
> Only steps that produced a pass/fail result contribute to scoring. Steps that haven't been interrogated, inactive steps (`active=False`), and steps that could not be evaluated (for example, a check that references a nonexistent column) are all excluded, so a broken check doesn't distort the score.


## Weighting Dimensions

Some organizations consider certain dimensions more critical than others. Provide per-dimension weights with `config(dimension_weights=...)` to scale each dimension's contribution to the overall score (a dimension not listed keeps a weight of `1.0`):


``` python
pb.config(dimension_weights={"completeness": 3.0})

validation_weighted = (
    pb.Validate(data=sales)
    .col_vals_not_null(columns="amount")   # Completeness, weighted 3x
    .col_vals_gt(columns="amount", value=0)  # Validity
    .interrogate()
)
print(validation_weighted.get_health_score())

pb.config()  # reset back to defaults
```


    85.0


    PointblankConfig(report_incl_header=True, report_incl_footer=True, report_incl_footer_timings=True, report_incl_footer_notes=True, report_incl_dimensions=False, preview_incl_header=True, dimension_map=None, dimension_weights=None, dimension_thresholds=None)


# The Scorecard

For a compact, standalone summary (well suited to dashboards or an executive overview) use <a href="../../reference/Validate.get_scorecard.html#pointblank.Validate.get_scorecard" class="gdls-link"><code>get_scorecard()</code></a>. It shows the overall health score prominently along with a per-dimension breakdown (a color-coded bar, the score, and passing/total test units):


``` python
validation.get_scorecard()
```


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_title gt_font_normal">Dimension Scores -- <code>sales</code></th>
</tr>
<tr class="gt_heading">
<th colspan="3" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div style="padding-top: 4px;">
<span style="font-weight: 600; color: #444444; font-size: 13px;">Health Score:</span> <span style="font-weight: 700; font-size: 22px; color: #A15C00;">84%</span>
</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_scorecard-dimension" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">DIMENSION</th>
<th id="pb_scorecard-score" class="gt_col_heading gt_columns_bottom_border gt_left" style="color: #666666; font-weight: bold" scope="col">SCORE</th>
<th id="pb_scorecard-test_units" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: #666666; font-weight: bold" scope="col">UNITS</th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_left" style="height: 34px"><span style="background-color: #3C6E9A; color: #FFFFFF; padding: 3px 9px; border-radius: 9px; font-size: 11px; font-weight: 600; white-space: nowrap; line-height: 1; display: inline-block; vertical-align: middle;">Completeness</span></td>
<td class="gt_row gt_left" style="height: 34px"><div style="display: flex; align-items: center; gap: 8px;">


<span style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #444444; font-weight: 600;">90%</span>
</div></td>
<td class="gt_row gt_right" style="height: 34px; color: black; font-family: IBM Plex Mono; font-size: 12px">9 / 10</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 34px"><span style="background-color: #4E9A6B; color: #FFFFFF; padding: 3px 9px; border-radius: 9px; font-size: 11px; font-weight: 600; white-space: nowrap; line-height: 1; display: inline-block; vertical-align: middle;">Validity</span></td>
<td class="gt_row gt_left" style="height: 34px"><div style="display: flex; align-items: center; gap: 8px;">


<span style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #444444; font-weight: 600;">83.33%</span>
</div></td>
<td class="gt_row gt_right" style="height: 34px; color: black; font-family: IBM Plex Mono; font-size: 12px">25 / 30</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 34px"><span style="background-color: #8E6FB5; color: #FFFFFF; padding: 3px 9px; border-radius: 9px; font-size: 11px; font-weight: 600; white-space: nowrap; line-height: 1; display: inline-block; vertical-align: middle;">Uniqueness</span></td>
<td class="gt_row gt_left" style="height: 34px"><div style="display: flex; align-items: center; gap: 8px;">


<span style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #444444; font-weight: 600;">80%</span>
</div></td>
<td class="gt_row gt_right" style="height: 34px; color: black; font-family: IBM Plex Mono; font-size: 12px">8 / 10</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 34px"><span style="background-color: #7A7A8C; color: #FFFFFF; padding: 3px 9px; border-radius: 9px; font-size: 11px; font-weight: 600; white-space: nowrap; line-height: 1; display: inline-block; vertical-align: middle;">Volume</span></td>
<td class="gt_row gt_left" style="height: 34px"><div style="display: flex; align-items: center; gap: 8px;">


<span style="font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: #444444; font-weight: 600;">100%</span>
</div></td>
<td class="gt_row gt_right" style="height: 34px; color: black; font-family: IBM Plex Mono; font-size: 12px">1 / 1</td>
</tr>
</tbody>
</table>


The scorecard is a [Great Tables](https://github.com/posit-dev/great-tables) object, so you can display it directly, export it to HTML with `.as_raw_html()`, or save it to an image file with `.gtsave()`.


# Enforcing Minimum Scores

In automated pipelines and CI you often want to *fail the run* when a dimension slips below an acceptable level. Call <a href="../../reference/Validate.assert_dimension_scores.html#pointblank.Validate.assert_dimension_scores" class="gdls-link"><code>assert_dimension_scores()</code></a> with per-dimension minimums; it raises an `AssertionError` if any dimension is below its minimum (here, uniqueness is `80%`):


``` python
try:
    validation.assert_dimension_scores(thresholds={"uniqueness": 95})
except AssertionError as e:
    print(e)
```


    Dimension health score(s) below the required minimum: uniqueness (80% < 95%)


Dimensions present in the thresholds but absent from the validation are ignored, and a call where every dimension meets its minimum simply returns without raising. You can also set the minimums globally with `config(dimension_thresholds=...)`, and then call the method without arguments:


``` python
pb.config(dimension_thresholds={"completeness": 95})

try:
    validation.assert_dimension_scores()
except AssertionError as e:
    print(e)

pb.config()  # reset back to defaults
```


    Dimension health score(s) below the required minimum: completeness (90% < 95%)


    PointblankConfig(report_incl_header=True, report_incl_footer=True, report_incl_footer_timings=True, report_incl_footer_notes=True, report_incl_dimensions=False, preview_incl_header=True, dimension_map=None, dimension_weights=None, dimension_thresholds=None)


# Accessing Scores Programmatically

The scores are also included in the summary available to <a href="../../reference/FinalActions.html#pointblank.FinalActions" class="gdls-link"><code>FinalActions</code></a> via <a href="../../reference/get_validation_summary.html#pointblank.get_validation_summary" class="gdls-link"><code>get_validation_summary()</code></a>, under the `dimension_scores` and `overall_health_score` keys. This makes it easy to log or trend the health score over time:


``` python
def log_health():
    summary = pb.get_validation_summary()
    print(f"Overall health score: {summary['overall_health_score']}")
    print(f"By dimension: {summary['dimension_scores']}")

(
    pb.Validate(data=sales, final_actions=pb.FinalActions(log_health))
    .col_vals_not_null(columns="amount")
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)
```


    Overall health score: 80.0
    By dimension: {'completeness': 90.0, 'validity': 70.0}


<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_title gt_font_normal" style="text-align: left; color: #444444; font-size: 28px; font-weight: bold;">Pointblank Validation</th>
</tr>
<tr class="gt_heading">
<th colspan="14" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border" style="text-align: left;"><div>
<span style="text-decoration-style: solid; text-decoration-color: #ADD8E6; text-decoration-line: underline; text-underline-position: under; color: #333333; font-variant-numeric: tabular-nums; padding-left: 4px; margin-right: 5px; padding-right: 2px;">2026-07-22|23:24:43</span>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span>

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
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+Y29sX3ZhbHNfbm90X251bGw8L3RpdGxlPgogICAgPGcgaWQ9Ikljb25zIiBzdHJva2U9Im5vbmUiIHN0cm9rZS13aWR0aD0iMSIgZmlsbD0ibm9uZSIgZmlsbC1ydWxlPSJldmVub2RkIj4KICAgICAgICA8ZyBpZD0iY29sX3ZhbHNfbm90X251bGwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuMDAwMDAwLCAwLjU1MTcyNCkiPgogICAgICAgICAgICA8cGF0aCBkPSJNNTYuNzEyMjM0LDEgQzU5LjE5NzUxNTMsMSA2MS40NDc1MTUzLDIuMDA3MzU5MzEgNjMuMDc2MTk1LDMuNjM2MDM4OTcgQzY0LjcwNDg3NDcsNS4yNjQ3MTg2MyA2NS43MTIyMzQsNy41MTQ3MTg2MyA2NS43MTIyMzQsMTAgTDY1LjcxMjIzNCwxMCBMNjUuNzEyMjM0LDY1IEwxMC43MTIyMzQsNjUgQzguMjI2OTUyNTksNjUgNS45NzY5NTI1OSw2My45OTI2NDA3IDQuMzQ4MjcyOTQsNjIuMzYzOTYxIEMyLjcxOTU5MzI4LDYwLjczNTI4MTQgMS43MTIyMzM5Nyw1OC40ODUyODE0IDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsNTYgTDEuNzEyMjMzOTcsMTAgQzEuNzEyMjMzOTcsNy41MTQ3MTg2MyAyLjcxOTU5MzI4LDUuMjY0NzE4NjMgNC4zNDgyNzI5NCwzLjYzNjAzODk3IEM1Ljk3Njk1MjU5LDIuMDA3MzU5MzEgOC4yMjY5NTI1OSwxIDEwLjcxMjIzNCwxIEwxMC43MTIyMzQsMSBaIiBpZD0icmVjdGFuZ2xlIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS13aWR0aD0iMiIgZmlsbD0iI0ZGRkZGRiIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTQwLjYxMjA4MDUsNDcuMDM3ODM0IEMzNy40NjkyMzQ4LDQ3LjAzNzgzNCAzNS4wMTI2MTM5LDQ1LjkzNDg2MTMgMzMuNzEyMjM0LDQ0LjAxNDA1OTcgQzMyLjQxMTg1NDEsNDUuOTM0ODYxMyAyOS45NTUyMzMxLDQ3LjAzNzgzNCAyNi44MTIzODgzLDQ3LjAzNzgzNCBDMjIuNjU3NDM5Nyw0Ny4wMzc4MzQgMTYuMDY0NjcxMiw0My40NDM3NzIzIDE2LjA2NDY3MTIsMzMuODAyMTYxOSBDMTYuMDY0NjcxMiwyOS4zNDAxMzYxIDE3LjQ3MTU4NzksMTguOTYyMTY2IDMwLjUwMzU4NjIsMTguOTYyMTY2IEMzMC45NDU0MDE4LDE4Ljk2MjE2NiAzMS4zMDU3NDgxLDE5LjMyMjUxMjQgMzEuMzA1NzQ4MSwxOS43NjQzMjc5IEwzMS4zMDU3NDgxLDIxLjM2ODY1MTggQzMxLjMwNTc0ODEsMjEuODEwNDY3NCAzMC45NDU0MDE4LDIyLjE3MDgxMzggMzAuNTAzNTg2MiwyMi4xNzA4MTM4IEMyNi42NDAwNDg2LDIyLjE3MDgxMzggMjIuNDgxOTY2OCwyNS44MTE4Nzc0IDIyLjQ4MTk2NjgsMzMuODAyMTYxOSBDMjIuNDgxOTY2OCwzNy41MDkwMjc3IDIzLjc2MzU0NTYsNDMuMDI3MDI0MyAyNy4yOTQ5Mzg0LDQzLjAyNzAyNDMgQzI5Ljc5NTQyOCw0My4wMjcwMjQzIDMxLjIyNDI3OSw0MC40MjMxMzEyIDMyLjA5ODUwOTUsMzguMjg2MTIyMSBDMzAuNTA2NzE5NCwzNS42MTAxNTk2IDI5LjcwMTQyNDMsMzMuMTAzNDAzNSAyOS43MDE0MjQzLDMwLjgzNDc4OTIgQzI5LjcwMTQyNDMsMjUuNjIzODcwNyAzMS44NjAzNjc3LDIzLjc3NTEzNzcgMzMuNzEyMjM0LDIzLjc3NTEzNzcgQzM1LjU2NDEwMDIsMjMuNzc1MTM3NyAzNy43MjMwNDM3LDI1LjYyMzg3MDcgMzcuNzIzMDQzNywzMC44MzQ3ODkyIEMzNy43MjMwNDM3LDMzLjEzNDczODMgMzYuOTM5NjgyOCwzNS41Nzg4MjU1IDM1LjMyOTA5MTYsMzguMjg2MTIyMSBDMzYuNjI5NDcxNSw0MS40MzIxMDA5IDM4LjI0MzE5Niw0My4wMjcwMjQzIDQwLjEyOTUyOTUsNDMuMDI3MDI0MyBDNDMuNjYwOTIyMyw0My4wMjcwMjQzIDQ0Ljk0MjUwMTIsMzcuNTA5MDI3NyA0NC45NDI1MDEyLDMzLjgwMjE2MTkgQzQ0Ljk0MjUwMTIsMjUuODExODc3NCA0MC43ODQ0MTkzLDIyLjE3MDgxMzggMzYuOTIwODgxNywyMi4xNzA4MTM4IEMzNi40NzU5MzI5LDIyLjE3MDgxMzggMzYuMTE4NzE5OCwyMS44MTA0Njc0IDM2LjExODcxOTgsMjEuMzY4NjUxOCBMMzYuMTE4NzE5OCwxOS43NjQzMjc5IEMzNi4xMTg3MTk4LDE5LjMyMjUxMjQgMzYuNDc1OTMyOSwxOC45NjIxNjYgMzYuOTIwODgxNywxOC45NjIxNjYgQzQ5Ljk1Mjg4MDEsMTguOTYyMTY2IDUxLjM1OTc5NjcsMjkuMzQwMTM2MSA1MS4zNTk3OTY3LDMzLjgwMjE2MTkgQzUxLjM1OTc5NjcsNDMuNDQzNzcyMyA0NC43NjcwMjgyLDQ3LjAzNzgzNCA0MC42MTIwODA1LDQ3LjAzNzgzNCBaIiBpZD0ib21lZ2EiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTMzLDcuOTM1OTc3MDUgQzMzLjI3NjE0MjQsNy45MzU5NzcwNSAzMy41LDguMTU5ODM0NjcgMzMuNSw4LjQzNTk3NzA1IEwzMy41LDU3LjU2NDAyMyBDMzMuNSw1Ny44NDAxNjUzIDMzLjI3NjE0MjQsNTguMDY0MDIzIDMzLDU4LjA2NDAyMyBDMzIuNzIzODU3Niw1OC4wNjQwMjMgMzIuNSw1Ny44NDAxNjUzIDMyLjUsNTcuNTY0MDIzIEwzMi41LDguNDM1OTc3MDUgQzMyLjUsOC4xNTk4MzQ2NyAzMi43MjM4NTc2LDcuOTM1OTc3MDUgMzMsNy45MzU5NzcwNSBaIiBpZD0ibGluZV9ibGFjayIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMzMuMDAwMDAwLCAzMy4wMDAwMDApIHJvdGF0ZSgtMzIwLjAwMDAwMCkgdHJhbnNsYXRlKC0zMy4wMDAwMDAsIC0zMy4wMDAwMDApICIgLz4KICAgICAgICAgICAgPHBvbHlnb24gaWQ9ImxpbmVfd2hpdGUiIGZpbGw9IiNGRkZGRkYiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDM0Ljg5OTQ5NiwgMzIuMTUzMzAzKSByb3RhdGUoLTMyMC4wMDAwMDApIHRyYW5zbGF0ZSgtMzQuODk5NDk2LCAtMzIuMTUzMzAzKSAiIHBvaW50cz0iMzQuMzk5NDk2MiA4LjU0MTYwNDY5IDM1LjM5OTQ5NjIgOC41NDE2MDQ2OSAzNS4zOTk0OTYyIDU1Ljc2NTAwMTkgMzQuMzk5NDk2MiA1NS43NjUwMDE5Ij48L3BvbHlnb24+CiAgICAgICAgPC9nPgogICAgPC9nPgo8L3N2Zz4=" />

col_vals_not_null()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">amount</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">10</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">9<br />
0.90</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">1<br />
0.10</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
<tr>
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">2</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9ImJhY2tncm91bmQ6ICNGRkZGRkY7Ij4KICAgIDx0aXRsZT5jb2xfdmFsc19ndDwvdGl0bGU+CiAgICA8ZyBpZD0iSWNvbnMiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJjb2xfdmFsc19ndCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS41MDAwMDAsIDEuNTAwMDAwKSIgZmlsbC1ydWxlPSJub256ZXJvIj4KICAgICAgICAgICAgPHBhdGggZD0iTTU1LDAgQzU3LjQ4NTI4MTMsMCA1OS43MzUyODEzLDEuMDA3MzU5MzEgNjEuMzYzOTYxLDIuNjM2MDM4OTcgQzYyLjk5MjY0MDcsNC4yNjQ3MTg2MyA2NCw2LjUxNDcxODYzIDY0LDkgTDY0LDkgTDY0LDY0IEw5LDY0IEM2LjUxNDcxODYyLDY0IDQuMjY0NzE4NjIsNjIuOTkyNjQwNyAyLjYzNjAzODk3LDYxLjM2Mzk2MSBDMS4wMDczNTkzMSw1OS43MzUyODE0IDAsNTcuNDg1MjgxNCAwLDU1IEwwLDU1IEwwLDkgQzAsNi41MTQ3MTg2MyAxLjAwNzM1OTMxLDQuMjY0NzE4NjMgMi42MzYwMzg5NywyLjYzNjAzODk3IEM0LjI2NDcxODYyLDEuMDA3MzU5MzEgNi41MTQ3MTg2MiwwIDksMCBMOSwwIEw1NSwwIFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNDguNzYxOTA0OCwxMCBMMTUuMjM4MDk1MywxMCBDMTIuMzQ2NjY2NywxMCAxMCwxMi4zNDY2NjY3IDEwLDE1LjIzODA5NTIgTDEwLDQ4Ljc2MTkwNDggQzEwLDUxLjY1MzMzMzMgMTIuMzQ2NjY2Nyw1NCAxNS4yMzgwOTUzLDU0IEw0OC43NjE5MDQ4LDU0IEM1MS42NTMzMzMzLDU0IDU0LDUxLjY1MzMzMzMgNTQsNDguNzYxOTA0OCBMNTQsMTUuMjM4MDk1MiBDNTQsMTIuMzQ2NjY2NyA1MS42NTMzMzMzLDEwIDQ4Ljc2MTkwNDgsMTAgWiBNMjUuMjYzODA5NSw0NC4zODI4NTcxIEwyNC4wNjk1MjM4LDQyLjY2NDc2MTkgTDM5LjU4NDc2MTksMzIgTDI0LjA2OTUyMzgsMjEuMzM1MjM4MSBMMjUuMjYzODA5NSwxOS42MTcxNDI5IEw0My4yODI4NTcyLDMyIEwyNS4yNjM4MDk1LDQ0LjM4Mjg1NzEgWiIgaWQ9ImdyZWF0ZXJfdGhhbiIgZmlsbD0iIzAwMDAwMCIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

col_vals_gt()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">amount</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">0</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">10</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">7<br />
0.70</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">3<br />
0.30</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody><tfoot>
<tr class="gt_sourcenotes">
<td colspan="14" class="gt_sourcenote" style="text-align: left;">

<span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin-left: 10px; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-07-22 23:24:43 UTC</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; margin-right: 5px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">< 1 s</span><span style="background-color: #FFF; color: #444; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 1px 5px -1px; border: solid 1px #999999; font-variant-numeric: tabular-nums; border-radius: 0; padding: 2px 10px 2px 10px;">2026-07-22 23:24:43 UTC</span>
</div></td>
</tr>
</tfoot>

</table>


# Localized Dimensions

Like the rest of the validation report, dimension labels and the health-score summary are localized. When you set a reporting language (e.g., `Validate(..., lang="fr")`), the dimension names in badge tooltips, the footer summary, and the scorecard are translated automatically. The two-letter badge codes stay language-neutral (matching the report's other short codes such as `TBL`, `EVAL`, and `W`/`E`/`C`), with the full localized name available on hover.
