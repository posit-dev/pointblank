# Comparing Two Tables

A very common data question is *how do these two tables differ?* Maybe you refactored a pipeline and want to know whether its output changed, migrated a table from one database to another, or received a new extract of a data feed. In each case there's a *source* table (the reference, or the data you expect) and a *target* table (the data you want to check), and you want to know:

- which records are missing from the target
- which records in the target are extra
- which records have values that differ, and in which columns
- whether the columns themselves (names, types, order) changed

Pointblank answers all of these with the `compare()` function, and the same comparison is available as a validation step through [Validate.tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match) and on the command line through `pb compare`.


# A First Comparison

Let's make two small tables of orders. The target is a later extract of the same data: one order had its amount corrected, one order is gone, and a new order has appeared.


``` python
import polars as pl

source = pl.DataFrame(
    {
        "order_id": [1001, 1002, 1003, 1004, 1005],
        "customer": ["Ana", "Ben", "Cho", "Dev", "Eli"],
        "amount": [25.00, 40.50, 12.75, 99.99, 5.00],
        "status": ["shipped", "shipped", "pending", "shipped", "cancelled"],
    }
)

target = pl.DataFrame(
    {
        "order_id": [1001, 1002, 1004, 1005, 1006],
        "customer": ["Ana", "Ben", "Dev", "Eli", "Fay"],
        "amount": [25.00, 45.50, 99.99, 5.00, 18.25],
        "status": ["shipped", "shipped", "shipped", "cancelled", "pending"],
    }
)
```


Since every order has a unique `order_id`, we'll use it as the *key* to line up the rows of the two tables:


``` python
import pointblank as pb

comparison = pb.compare(source, target, keys="order_id")

comparison
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_comparison_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_comparison_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_comparison_tbl p { margin: 0; padding: 0; }
 #pb_comparison_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: 100%; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_comparison_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_comparison_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_comparison_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_comparison_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_comparison_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_comparison_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_comparison_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_comparison_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_comparison_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_comparison_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_comparison_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_comparison_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_comparison_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_comparison_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_comparison_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_comparison_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_comparison_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_comparison_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_comparison_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_comparison_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_comparison_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_comparison_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_comparison_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_comparison_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_left { text-align: left; }
 #pb_comparison_tbl .gt_center { text-align: center; }
 #pb_comparison_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_comparison_tbl .gt_font_normal { font-weight: normal; }
 #pb_comparison_tbl .gt_font_bold { font-weight: bold; }
 #pb_comparison_tbl .gt_font_italic { font-style: italic; }
 #pb_comparison_tbl .gt_super { font-size: 65%; }
 #pb_comparison_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_comparison_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 100%" aria-labelledby="pb_comparison_tbl-title" aria-describedby="pb_comparison_tbl-subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<colgroup>
<col style="width: 30%" />
<col style="width: 70%" />
</colgroup>
<thead>
<tr class="gt_heading">
<th colspan="2" id="pb_comparison_tbl-title" class="gt_heading gt_title gt_font_normal">Table Comparison</th>
</tr>
<tr class="gt_heading">
<th colspan="2" id="pb_comparison_tbl-subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div style="font-size:13px; line-height:1.9;">
<strong>source</strong> (5 rows) → <strong>target</strong> (5 rows), aligned by key <code>order_id</code><br />
<span style="background:#CF142B; color:white; padding:1px 6px; border-radius:3px; font-size:11px; font-weight:bold; letter-spacing:0.3px;">DIFFERENT</span> 3 of 6 rows fail (50.0%)<br />


<br />
<span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#4CA64C; margin-right:4px; margin-left:4px;"></span>match <strong>3</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#E8A33D; margin-right:4px; margin-left:4px;"></span>changed <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#CF142B; margin-right:4px; margin-left:4px;"></span>missing <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#3D7CC9; margin-right:4px; margin-left:4px;"></span>extra <strong>1</strong></span>
</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="2" id="pb_comparison_tbl-Column-mismatches-1-of-3-columns" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Column mismatches (1 of 3 columns)</th>
</tr>

<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Column-mismatches-1-of-3-columns">amount</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Column-mismatches-1-of-3-columns"><div style="display:inline-block; vertical-align:middle; width:120px; height:7px; background:#EEE; border-radius:2px; margin-right:8px; margin-left:8px; overflow:hidden;">


1 of 4 matched rows differ (25.0%)</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Changed-rows-1-row" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Changed rows (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Changed-rows-1-row">1002</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Changed-rows-1-row"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">40.5</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">45.5</span> <span style="color:#888; font-size:11px;">(+5)</span></span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Missing-from-target-1-row" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Missing from target (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Missing-from-target-1-row">1003</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Missing-from-target-1-row"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">customer:</span> Cho</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 12.75</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">status:</span> pending</span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Extra-in-target-1-row" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Extra in target (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Extra-in-target-1-row">1006</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Extra-in-target-1-row"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">customer:</span> Fay</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 18.25</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">status:</span> pending</span></td>
</tr>
</tbody>
</table>


The report's header gives the verdict and a breakdown of the rows. Below it, each section shows one kind of difference: which columns have mismatched values (and how often), then samples of the changed, missing, and extra rows. For changed rows, only the differing values are shown, as *old → new* with the numeric difference alongside.


# Row Statuses

Every row that can be aligned between the two tables gets exactly one status:

| Status | Meaning |
|----|----|
| `match` | the row is in both tables and all compared values are equal |
| `changed` | the row is in both tables but at least one compared value differs |
| `missing` | the row is in the source but not in the target |
| `extra` | the row is in the target but not in the source |
| `dup_key` | the row's key appears more than once in either table, so it can't be aligned |

The counts are available directly on the [Comparison](../../reference/Comparison.md#pointblank.Comparison) object:


``` python
comparison.status_counts
```


    {'match': 3, 'changed': 1, 'missing': 1, 'extra': 1, 'dup_key': 0}


``` python
comparison.n_changed, comparison.n_missing, comparison.n_extra
```


    (1, 1, 1)


A [Comparison](../../reference/Comparison.md#pointblank.Comparison) is truthy only if the tables match, so it can be used directly in a test or an `if` statement:


``` python
bool(comparison)
```


    False


# Getting the Differing Rows

Use the [rows()](../../reference/Comparison.rows.md#pointblank.Comparison.rows) method to get the rows with a given status as a table. Changed rows come with a `<column>_source` and a `<column>_target` column for each compared column:


``` python
comparison.rows("changed")
```


shape: (1, 7)

| order_id | customer_source | customer_target | amount_source | amount_target | status_source | status_target |
|----|----|----|----|----|----|----|
| i64 | str | str | f64 | f64 | str | str |
| 1002 | "Ben" | "Ben" | 40.5 | 45.5 | "shipped" | "shipped" |


Missing rows come from the source and extra rows from the target:


``` python
comparison.rows("missing")
```


shape: (1, 4)

| order_id | customer | amount | status    |
|----------|----------|--------|-----------|
| i64      | str      | f64    | str       |
| 1003     | "Cho"    | 12.75  | "pending" |


``` python
comparison.rows("extra")
```


shape: (1, 4)

| order_id | customer | amount | status    |
|----------|----------|--------|-----------|
| i64      | str      | f64    | str       |
| 1006     | "Fay"    | 18.25  | "pending" |


Only up to `limit=` rows (100 by default) are collected, so this is safe even for very large tables. For a per-column view of the mismatches, use [column_summary()](../../reference/Comparison.column_summary.md#pointblank.Comparison.column_summary):


``` python
comparison.column_summary()
```


shape: (3, 6)

| column     | target_column | dtype_source | dtype_target | n_mismatch | f_mismatch |
|------------|---------------|--------------|--------------|------------|------------|
| str        | str           | str          | str          | i64        | f64        |
| "customer" | "customer"    | "String"     | "String"     | 0          | 0.0        |
| "amount"   | "amount"      | "Float64"    | "Float64"    | 1          | 0.25       |
| "status"   | "status"      | "String"     | "String"     | 0          | 0.0        |


# How Rows Are Aligned

The `keys=` argument decides how the rows of the two tables are lined up:

- **By key** (`keys="order_id"`, or a list for a composite key): rows are matched on the key values, so their order doesn't matter. This is the most informative mode, and the one to use whenever the data has an identifier.
- **By position** (`keys=None`, the default): the first row of the source is compared with the first row of the target, and so on. This suits outputs that are supposed to be identical row for row.
- **As multisets** (`keys="*"`): whole rows are compared regardless of their order, with duplicates counted. This is for tables that have no key at all.

Here's what positional alignment makes of the same two tables. Since a row was removed and another added, most rows no longer line up:


``` python
pb.compare(source, target).status_counts
```


    {'match': 1, 'changed': 4, 'missing': 0, 'extra': 0, 'dup_key': 0}


# Tolerances, Normalization, and Nulls

Not every difference matters. Numeric and datetime columns can be compared with a tolerance, given for all numeric columns or per column. An absolute tolerance of `5` absorbs the corrected amount:


``` python
pb.compare(source, target, keys="order_id", tolerance={"amount": 5}).n_changed
```


    0


Tolerances can be absolute (`atol`), relative to the source value (`rtol`), or both, as in `tolerance={"amount": {"atol": 0.01, "rtol": 1e-6}}`. Two values are equal if `|source - target| <= atol + rtol * |source|`. Datetime tolerances are durations like `"1s"`, `"500ms"`, or `"2m"`.

String values can be normalized before comparing with `normalize=`. For example, `normalize={"customer": ["strip", "lower"]}` ignores surrounding whitespace and letter case (the available operations are `"strip"`, `"lower"`, `"upper"`, `"collapse_whitespace"`, and `{"round": n}`). Reports always show the original values.

By default, two nulls count as equal, and NaN values in floating-point columns are treated as nulls. Use `null_equal=False` to count two nulls as a difference.


# Schema Differences

Column differences are reported in the *Schema differences* section of the report and in the `schema_diff` attribute. Suppose the target renamed a column and added another:


``` python
target_v2 = target.rename({"customer": "customer_name"}).with_columns(
    region=pl.lit("EU")
)

comparison_v2 = pb.compare(source, target_v2, keys="order_id")

comparison_v2.schema_diff
```


    SchemaDiff(only_in_source=['customer'], only_in_target=['customer_name', 'region'], dtype_changed=[], renamed=[], case_only=[], reordered=False)


With the default `schema="strict"`, any schema difference means the tables don't match and every row counts as failing. Two arguments help here. `column_map=` pairs up renamed columns, and `schema="common"` compares only the columns that both tables share (still reporting the schema differences, but without failing any rows):


``` python
pb.compare(
    source,
    target_v2,
    keys="order_id",
    column_map={"customer": "customer_name"},
    schema="common",
)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_comparison_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_comparison_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_comparison_tbl p { margin: 0; padding: 0; }
 #pb_comparison_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: 100%; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_comparison_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_comparison_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_comparison_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_comparison_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_comparison_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_comparison_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_comparison_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_comparison_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_comparison_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_comparison_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_comparison_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_comparison_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_comparison_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_comparison_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_comparison_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_comparison_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_comparison_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_comparison_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_comparison_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_comparison_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_comparison_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_comparison_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_comparison_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_comparison_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_left { text-align: left; }
 #pb_comparison_tbl .gt_center { text-align: center; }
 #pb_comparison_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_comparison_tbl .gt_font_normal { font-weight: normal; }
 #pb_comparison_tbl .gt_font_bold { font-weight: bold; }
 #pb_comparison_tbl .gt_font_italic { font-style: italic; }
 #pb_comparison_tbl .gt_super { font-size: 65%; }
 #pb_comparison_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_comparison_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 100%" aria-labelledby="pb_comparison_tbl-title" aria-describedby="pb_comparison_tbl-subtitle" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<colgroup>
<col style="width: 30%" />
<col style="width: 70%" />
</colgroup>
<thead>
<tr class="gt_heading">
<th colspan="2" id="pb_comparison_tbl-title" class="gt_heading gt_title gt_font_normal">Table Comparison</th>
</tr>
<tr class="gt_heading">
<th colspan="2" id="pb_comparison_tbl-subtitle" class="gt_heading gt_subtitle gt_font_normal gt_bottom_border"><div style="font-size:13px; line-height:1.9;">
<strong>source</strong> (5 rows) → <strong>target</strong> (5 rows), aligned by key <code>order_id</code><br />
<span style="background:#CF142B; color:white; padding:1px 6px; border-radius:3px; font-size:11px; font-weight:bold; letter-spacing:0.3px;">DIFFERENT</span> 3 of 6 rows fail (50.0%)<br />


<br />
<span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#4CA64C; margin-right:4px; margin-left:4px;"></span>match <strong>3</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#E8A33D; margin-right:4px; margin-left:4px;"></span>changed <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#CF142B; margin-right:4px; margin-left:4px;"></span>missing <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#3D7CC9; margin-right:4px; margin-left:4px;"></span>extra <strong>1</strong></span><br />
<span style="border:1px solid #555; color:#555; padding:0 5px; border-radius:3px; font-size:11px; white-space:nowrap;">schema: common columns</span>
</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="2" id="pb_comparison_tbl-Schema-differences" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Schema differences</th>
</tr>

<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Schema-differences">region</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Schema-differences"><span style="border:1px solid #3D7CC9; color:#3D7CC9; padding:0 5px; border-radius:3px; font-size:11px; white-space:nowrap;">only in target</span></td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Schema-differences">customer</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Schema-differences">renamed → <code>customer_name</code> (via <code>column_map=</code>)</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Column-mismatches-1-of-3-columns" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Column mismatches (1 of 3 columns)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Column-mismatches-1-of-3-columns">amount</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Column-mismatches-1-of-3-columns"><div style="display:inline-block; vertical-align:middle; width:120px; height:7px; background:#EEE; border-radius:2px; margin-right:8px; margin-left:8px; overflow:hidden;">


1 of 4 matched rows differ (25.0%)</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Changed-rows-1-row" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Changed rows (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Changed-rows-1-row">1002</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Changed-rows-1-row"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">40.5</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">45.5</span> <span style="color:#888; font-size:11px;">(+5)</span></span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Missing-from-target-1-row" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Missing from target (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Missing-from-target-1-row">1003</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Missing-from-target-1-row"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">customer:</span> Cho</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 12.75</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">status:</span> pending</span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Extra-in-target-1-row" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Extra in target (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Extra-in-target-1-row">1006</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Extra-in-target-1-row"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">customer_name:</span> Fay</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 18.25</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">status:</span> pending</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">region:</span> EU</span></td>
</tr>
</tbody>
</table>


To compare just some of the columns, list them in `columns=`. The schema check is then limited to the keys and those columns.


# Duplicate Keys

A key that appears more than once in either table can't be aligned unambiguously, so its rows get the `dup_key` status (they always count as failing). By default the report lists the duplicated keys with their number of occurrences in each table. With `dup_keys="compare"`, the report also compares the duplicated rows pairwise to show which columns differ between them.


# Table Comparison as a Validation Step

The [tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match) validation method uses the same comparison, which makes it a validation step like any other: it can have thresholds and actions, and it shows up in the validation report. The table being validated is the target, and `tbl_compare=` is the source. Every aligned row is a *test unit*, so thresholds can be set as a proportion of rows:


``` python
validation = (
    pb.Validate(data=target, tbl_name="orders", thresholds=(0.1, 0.25))
    .tbl_match(tbl_compare=source, keys="order_id")
    .interrogate()
)

validation
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_tbl p { margin: 0; padding: 0; }
 #pb_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 90%; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_tbl .gt_left { text-align: left; }
 #pb_tbl .gt_center { text-align: center; }
 #pb_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_tbl .gt_font_normal { font-weight: normal; }
 #pb_tbl .gt_font_bold { font-weight: bold; }
 #pb_tbl .gt_font_italic { font-style: italic; }
 #pb_tbl .gt_super { font-size: 65%; }
 #pb_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

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
<td class="gt_row gt_left" style="height: 40px; background-color: #EBBC14; color: transparent; font-size: 0px" headers="pb_tbl-status_color">#EBBC14</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold" headers="pb_tbl-i">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-type_upd"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+dGJsX21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJwYl90YmwtdHlwZV91cGQtMC1JY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InBiX3RibC10eXBlX3VwZC0wLXRibF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNzU4NjIxKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJwYl90YmwtdHlwZV91cGQtMC1yZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8ZyBpZD0icGJfdGJsLXR5cGVfdXBkLTAtZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ2LjAyNjYxMSwgMjAuNzEwMTIyKSByb3RhdGUoLTkwLjAwMDAwMCkgdHJhbnNsYXRlKC00Ni4wMjY2MTEsIC0yMC43MTAxMjIpIHRyYW5zbGF0ZSg0Mi41MjY2MTEsIDE2LjIxMDEyMikiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLWxpbmVjYXA9InNxdWFyZSI+CiAgICAgICAgICAgICAgICA8bGluZSB4MT0iMi4yMTIyMzM5NyIgeTE9IjAuNTE0NjY5MzUzIiB4Mj0iMi4yMTIyMzM5NyIgeTI9IjcuNTg1NzM3MTYiIGlkPSJwYl90YmwtdHlwZV91cGQtMC1MaW5lIj48L2xpbmU+CiAgICAgICAgICAgICAgICA8bGluZSB4MT0iNS4yMTIyMzM5NyIgeTE9IjAuNTE0NjY5MzUzIiB4Mj0iNS4yMTIyMzM5NyIgeTI9IjcuNTg1NzM3MTYiIGlkPSJwYl90YmwtdHlwZV91cGQtMC1MaW5lIj48L2xpbmU+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPGcgaWQ9InBiX3RibC10eXBlX3VwZC0wLWVxdWFsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyMS4zOTc4NTcsIDQ1LjMxOTIxNykgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMjEuMzk3ODU3LCAtNDUuMzE5MjE3KSB0cmFuc2xhdGUoMTcuODk3ODU3LCA0MC44MTkyMTcpIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0icGJfdGJsLXR5cGVfdXBkLTAtTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0icGJfdGJsLXR5cGVfdXBkLTAtTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yMS4zODgyNDE5LDcuNzc4Njk3ODMgQzIxLjM1ODQyOTgsNy43NzkzNTE3NyAyMS4zMjg3MDQsNy43ODIxNjM2NyAyMS4yOTkyOTk2LDcuNzg3MTE5MTQgTDkuMDkwMTQ4MjQsNy43ODcxMTkxNCBDOC43NTAyOTQzMSw3Ljc4NzE1MzEyIDguNDc0Nzk2NzcsOC4wNjI2NTA2NyA4LjQ3NDc2Mjc5LDguNDAyNTA0NiBMOC40NzQ3NjI3OSwxNi4yOTkxNDk4IEM4LjQ2Mzc3ODc0LDE2LjM2NTYwNjEgOC40NjM3Nzg3NCwxNi40MzM0MTQ5IDguNDc0NzYyNzksMTYuNDk5ODcxMyBMOC40NzQ3NjI3OSwyNC45MTQ1NDYxIEM4LjQ2Mzc3ODc0LDI0Ljk4MTAwMjUgOC40NjM3Nzg3NCwyNS4wNDg4MTEyIDguNDc0NzYyNzksMjUuMTE1MjY3NiBMOC40NzQ3NjI3OSwzMy4wMTc5MjI2IEM4LjQ3NDc5Njc3LDMzLjM1Nzc3NjYgOC43NTAyOTQzMSwzMy42MzMyNzQxIDkuMDkwMTQ4MjQsMzMuNjMzMzA4MSBMMjEuMjk0NDkxNiwzMy42MzMzMDgxIEMyMS4zNjA5NDc5LDMzLjY0NDI5MjEgMjEuNDI4NzU2NywzMy42NDQyOTIxIDIxLjQ5NTIxMzEsMzMuNjMzMzA4MSBMMzMuNzA1NTY2MywzMy42MzMzMDgxIEMzNC4wNDU0MjAyLDMzLjYzMzI3NDEgMzQuMzIwOTE3OCwzMy4zNTc3NzY2IDM0LjMyMDk1MTcsMzMuMDE3OTIyNiBMMzQuMzIwOTUxNywyNS4xMjEyNzc1IEMzNC4zMzE5MzU4LDI1LjA1NDgyMTEgMzQuMzMxOTM1OCwyNC45ODcwMTIzIDM0LjMyMDk1MTcsMjQuOTIwNTU2IEwzNC4zMjA5NTE3LDE2LjUwNTg4MTEgQzM0LjMzMTkzNTgsMTYuNDM5NDI0OCAzNC4zMzE5MzU4LDE2LjM3MTYxNiAzNC4zMjA5NTE3LDE2LjMwNTE1OTYgTDM0LjMyMDk1MTcsOC40MDI1MDQ2IEMzNC4zMjA5MTc4LDguMDYyNjUwNjcgMzQuMDQ1NDIwMiw3Ljc4NzE1MzEyIDMzLjcwNTU2NjMsNy43ODcxMTkxNCBMMjEuNDkyODA5NCw3Ljc4NzExOTE0IEMyMS40NTgyNTY1LDcuNzgxMzQzNjkgMjEuNDIzMjczNiw3Ljc3ODY5NzgzIDIxLjM4ODI0MTksNy43Nzg2OTc4MyBaIE05LjcwNTUzMzY5LDkuMDE3ODkwMDUgTDIwLjc4MjQ3MTgsOS4wMTc4OTAwNSBMMjAuNzgyNDcxOCwxNS43ODcxMyBMOS43MDU1MzM2OSwxNS43ODcxMyBMOS43MDU1MzM2OSw5LjAxNzg5MDA1IFogTTIyLjAxMzI0MjcsOS4wMTc4OTAwNSBMMzMuMDkwMTgwOCw5LjAxNzg5MDA1IEwzMy4wOTAxODA4LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDkuMDE3ODkwMDUgWiBNOS43MDU1MzM2OSwxNy4wMTc5MDA5IEwyMC43ODI0NzE4LDE3LjAxNzkwMDkgTDIwLjc4MjQ3MTgsMjQuNDAyNTI2MyBMOS43MDU1MzM2OSwyNC40MDI1MjYzIEw5LjcwNTUzMzY5LDE3LjAxNzkwMDkgWiBNMjIuMDEzMjQyNywxNy4wMTc5MDA5IEwzMy4wOTAxODA4LDE3LjAxNzkwMDkgTDMzLjA5MDE4MDgsMjQuNDAyNTI2MyBMMjIuMDEzMjQyNywyNC40MDI1MjYzIEwyMi4wMTMyNDI3LDE3LjAxNzkwMDkgWiBNOS43MDU1MzM2OSwyNS42MzMyOTcyIEwyMC43ODI0NzE4LDI1LjYzMzI5NzIgTDIwLjc4MjQ3MTgsMzIuNDAyNTM3MiBMOS43MDU1MzM2OSwzMi40MDI1MzcyIEw5LjcwNTUzMzY5LDI1LjYzMzI5NzIgWiBNMjIuMDEzMjQyNywyNS42MzMyOTcyIEwzMy4wOTAxODA4LDI1LjYzMzI5NzIgTDMzLjA5MDE4MDgsMzIuNDAyNTM3MiBMMjIuMDEzMjQyNywzMi40MDI1MzcyIEwyMi4wMTMyNDI3LDI1LjYzMzI5NzIgWiIgaWQ9InBiX3RibC10eXBlX3VwZC0wLXRhYmxlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00Ni4wMTY5OTUzLDMyLjM4Nzc5MjYgQzQ1Ljk4NzE4MzIsMzIuMzg4NDQ2NSA0NS45NTc0NTc1LDMyLjM5MTI1ODQgNDUuOTI4MDUzLDMyLjM5NjIxMzkgTDMzLjcxODkwMTYsMzIuMzk2MjEzOSBDMzMuMzc5MDQ3NywzMi4zOTYyNDc5IDMzLjEwMzU1MDIsMzIuNjcxNzQ1NCAzMy4xMDM1MTYyLDMzLjAxMTU5OTMgTDMzLjEwMzUxNjIsNDAuOTA4MjQ0NSBDMzMuMDkyNTMyMiw0MC45NzQ3MDA5IDMzLjA5MjUzMjIsNDEuMDQyNTA5NyAzMy4xMDM1MTYyLDQxLjEwODk2NiBMMzMuMTAzNTE2Miw0OS41MjM2NDA4IEMzMy4wOTI1MzIyLDQ5LjU5MDA5NzIgMzMuMDkyNTMyMiw0OS42NTc5MDYgMzMuMTAzNTE2Miw0OS43MjQzNjI0IEwzMy4xMDM1MTYyLDU3LjYyNzAxNzQgQzMzLjEwMzU1MDIsNTcuOTY2ODcxMyAzMy4zNzkwNDc3LDU4LjI0MjM2ODkgMzMuNzE4OTAxNiw1OC4yNDI0MDI4IEw0NS45MjMyNDUsNTguMjQyNDAyOCBDNDUuOTg5NzAxNCw1OC4yNTMzODY5IDQ2LjA1NzUxMDEsNTguMjUzMzg2OSA0Ni4xMjM5NjY1LDU4LjI0MjQwMjggTDU4LjMzNDMxOTcsNTguMjQyNDAyOCBDNTguNjc0MTczNiw1OC4yNDIzNjg5IDU4Ljk0OTY3MTIsNTcuOTY2ODcxMyA1OC45NDk3MDUxLDU3LjYyNzAxNzQgTDU4Ljk0OTcwNTEsNDkuNzMwMzcyMiBDNTguOTYwNjg5Miw0OS42NjM5MTU4IDU4Ljk2MDY4OTIsNDkuNTk2MTA3MSA1OC45NDk3MDUxLDQ5LjUyOTY1MDcgTDU4Ljk0OTcwNTEsNDEuMTE0OTc1OSBDNTguOTYwNjg5Miw0MS4wNDg1MTk1IDU4Ljk2MDY4OTIsNDAuOTgwNzEwNyA1OC45NDk3MDUxLDQwLjkxNDI1NDQgTDU4Ljk0OTcwNTEsMzMuMDExNTk5MyBDNTguOTQ5NjcxMiwzMi42NzE3NDU0IDU4LjY3NDE3MzYsMzIuMzk2MjQ3OSA1OC4zMzQzMTk3LDMyLjM5NjIxMzkgTDQ2LjEyMTU2MjgsMzIuMzk2MjEzOSBDNDYuMDg3MDA5OSwzMi4zOTA0Mzg0IDQ2LjA1MjAyNywzMi4zODc3OTI2IDQ2LjAxNjk5NTMsMzIuMzg3NzkyNiBaIE0zNC4zMzQyODcxLDMzLjYyNjk4NDggTDQ1LjQxMTIyNTIsMzMuNjI2OTg0OCBMNDUuNDExMjI1Miw0MC4zOTYyMjQ4IEwzNC4zMzQyODcxLDQwLjM5NjIyNDggTDM0LjMzNDI4NzEsMzMuNjI2OTg0OCBaIE00Ni42NDE5OTYxLDMzLjYyNjk4NDggTDU3LjcxODkzNDIsMzMuNjI2OTg0OCBMNTcuNzE4OTM0Miw0MC4zOTYyMjQ4IEw0Ni42NDE5OTYxLDQwLjM5NjIyNDggTDQ2LjY0MTk5NjEsMzMuNjI2OTg0OCBaIE0zNC4zMzQyODcxLDQxLjYyNjk5NTcgTDQ1LjQxMTIyNTIsNDEuNjI2OTk1NyBMNDUuNDExMjI1Miw0OS4wMTE2MjExIEwzNC4zMzQyODcxLDQ5LjAxMTYyMTEgTDM0LjMzNDI4NzEsNDEuNjI2OTk1NyBaIE00Ni42NDE5OTYxLDQxLjYyNjk5NTcgTDU3LjcxODkzNDIsNDEuNjI2OTk1NyBMNTcuNzE4OTM0Miw0OS4wMTE2MjExIEw0Ni42NDE5OTYxLDQ5LjAxMTYyMTEgTDQ2LjY0MTk5NjEsNDEuNjI2OTk1NyBaIE0zNC4zMzQyODcxLDUwLjI0MjM5MiBMNDUuNDExMjI1Miw1MC4yNDIzOTIgTDQ1LjQxMTIyNTIsNTcuMDExNjMxOSBMMzQuMzM0Mjg3MSw1Ny4wMTE2MzE5IEwzNC4zMzQyODcxLDUwLjI0MjM5MiBaIE00Ni42NDE5OTYxLDUwLjI0MjM5MiBMNTcuNzE4OTM0Miw1MC4yNDIzOTIgTDU3LjcxODkzNDIsNTcuMDExNjMxOSBMNDYuNjQxOTk2MSw1Ny4wMTE2MzE5IEw0Ni42NDE5OTYxLDUwLjI0MjM5MiBaIiBpZD0icGJfdGJsLXR5cGVfdXBkLTAtdGFibGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

tbl_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-columns_upd">None</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden" headers="pb_tbl-values_upd">EXTERNAL TABLE<br />
<span style="font-size: 8px; font-weight: 600; letter-spacing: 0.5px; color: #5A6B7B;">key: order_id</span></td>
4IDEzLjg4NDIzNTksOC43MTkyMzA3NCBDMTMuNzgyMjcwNCw4LjczNjYzOTY3IDEzLjY4Nzc2NTQsOC43NzY0MzExNSAxMy42MDU2OTU2LDguODM4NjA1MTggTDEwLjI0MzMxNTYsMTEuMzg1MjU5OCBDMTAuMDc2Njg4NiwxMS41MDQ2MzQzIDkuOTc3MjA5OTMsMTEuNjk4NjE4MSA5Ljk3NzIwOTkzLDExLjkwMjU0OTEgQzkuOTc3MjA5OTMsMTIuMTA2NDgwNyAxMC4wNzY2ODg2LDEyLjMwMDQ2MzkgMTAuMjQzMzE1NiwxMi40MTk4MzgzIEwxMy42MDU2OTU2LDE0Ljk2NjQ5MyBDMTMuODkxNjk3LDE1LjE4MDM3MjUgMTQuMjk3MDcyOSwxNS4xMjMxNzIxIDE0LjUxMDk1MTcsMTQuODM3MTcwNyBDMTQuNzI0ODMxMywxNC41NTExNjkyIDE0LjY2NzYzMDksMTQuMTQ1Nzk0IDE0LjM4MTYyOTQsMTMuOTMxOTE0NSBMMTIuNTMxMzI1NywxMi41MzkyMTI3IEwyMS44ODEyNDk1LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTEuMjY1ODg1NCBMMTIuNTMxMzI1NywxMS4yNjU4ODU0IEwxNC4zODE2Mjk0LDkuODczMTgzNjQgQzE0LjYzNzc4NzIsOS43MTY1MDQ1MyAxNC43NDk3MDA2LDkuNDAwNjYwMTQgMTQuNjQ3NzM1MSw5LjExNzE0NTUzIEMxNC41NDgyNTY0LDguODMzNjMxNTYgMTQuMjYyMjU1LDguNjU5NTQzNTIgMTMuOTYzODE4OSw4LjY5OTMzNSBaIiBpZD0icGJfdGJsLXRibC0wLWFycm93IiBmaWxsPSIjMDAwMDAwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNS45MjkyMzAsIDExLjg5NDczNykgcm90YXRlKC0xODAuMDAwMDAwKSB0cmFuc2xhdGUoLTE1LjkyOTIzMCwgLTExLjg5NDczNykgIiAvPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-eval"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px" headers="pb_tbl-test_units">6</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-pass">3<br />
0.50</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5" headers="pb_tbl-fail">3<br />
0.50</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3" headers="pb_tbl-w_upd"><span style="color: #AAAAAA;">●</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC" headers="pb_tbl-e_upd"><span style="color: #EBBC14;">●</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3" headers="pb_tbl-c_upd">--</td>
<td class="gt_row gt_center" style="height: 40px" headers="pb_tbl-extract_upd">CSV</td>
</tr>
</tbody>
</table>


> **Note: Note**
>
> In Pointblank `v1.0.1` and earlier, [tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match) had a single test unit for the whole table. Steps still pass or fail in the same situations, but fractional thresholds now refer to the share of differing rows. To react to *any* difference, as before, use an absolute threshold of `1`.

Three of the six aligned rows differ, exceeding both the 'warning' and 'error' thresholds. The step report holds the full comparison:


``` python
validation.get_step_report(i=1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_comparison_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_comparison_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_comparison_tbl p { margin: 0; padding: 0; }
 #pb_comparison_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: 100%; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_comparison_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_comparison_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_comparison_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_comparison_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_comparison_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_comparison_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_comparison_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_comparison_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_comparison_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_comparison_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_comparison_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_comparison_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_comparison_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_comparison_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_comparison_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_comparison_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_comparison_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_comparison_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_comparison_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_comparison_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_comparison_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_comparison_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_comparison_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_comparison_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_left { text-align: left; }
 #pb_comparison_tbl .gt_center { text-align: center; }
 #pb_comparison_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_comparison_tbl .gt_font_normal { font-weight: normal; }
 #pb_comparison_tbl .gt_font_bold { font-weight: bold; }
 #pb_comparison_tbl .gt_font_italic { font-style: italic; }
 #pb_comparison_tbl .gt_super { font-size: 65%; }
 #pb_comparison_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_comparison_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 100%" aria-labelledby="pb_comparison_tbl-title" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<colgroup>
<col style="width: 30%" />
<col style="width: 70%" />
</colgroup>
<thead>
<tr class="gt_heading">
<th colspan="2" id="pb_comparison_tbl-title" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #CF142B;">✗</span>


ASSERTION <span style="border-style: solid; border-width: thin; border-color: lightblue; padding-left: 2px; padding-right: 2px;"><code style="color: #303030; background-color: transparent; position: relative; bottom: 1px;">tbl_match(keys='order_id')</code></span>

<strong>tbl_compare</strong> (5 rows) → <strong>orders</strong> (5 rows), aligned by key <code>order_id</code><br />
<span style="background:#CF142B; color:white; padding:1px 6px; border-radius:3px; font-size:11px; font-weight:bold; letter-spacing:0.3px;">DIFFERENT</span> 3 of 6 rows fail (50.0%)<br />


<br />
<span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#4CA64C; margin-right:4px; margin-left:4px;"></span>match <strong>3</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#E8A33D; margin-right:4px; margin-left:4px;"></span>changed <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#CF142B; margin-right:4px; margin-left:4px;"></span>missing <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#3D7CC9; margin-right:4px; margin-left:4px;"></span>extra <strong>1</strong></span>


</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="2" id="pb_comparison_tbl-Column-mismatches-1-of-3-columns" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Column mismatches (1 of 3 columns)</th>
</tr>

<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Column-mismatches-1-of-3-columns">amount</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Column-mismatches-1-of-3-columns"><div style="display:inline-block; vertical-align:middle; width:120px; height:7px; background:#EEE; border-radius:2px; margin-right:8px; margin-left:8px; overflow:hidden;">


1 of 4 matched rows differ (25.0%)</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Changed-rows-1-row" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Changed rows (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Changed-rows-1-row">1002</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Changed-rows-1-row"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount:</span> <span style="text-decoration:line-through; color:#999;">40.5</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">45.5</span> <span style="color:#888; font-size:11px;">(+5)</span></span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Missing-from-target-1-row" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Missing from target (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Missing-from-target-1-row">1003</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Missing-from-target-1-row"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">customer:</span> Cho</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 12.75</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">status:</span> pending</span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Extra-in-target-1-row" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Extra in target (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Extra-in-target-1-row">1006</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Extra-in-target-1-row"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">customer:</span> Fay</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount:</span> 18.25</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">status:</span> pending</span></td>
</tr>
</tbody>
</table>


The data extract for the step contains the failing rows from both tables, with a `_status_` column and source/target value pairs:


``` python
validation.get_data_extracts(i=1, frame=True)
```


shape: (3, 8)

| order_id | \_status\_ | customer_source | customer_target | amount_source | amount_target | status_source | status_target |
|----|----|----|----|----|----|----|----|
| i64 | str | str | str | f64 | f64 | str | str |
| 1002 | "changed" | "Ben" | "Ben" | 40.5 | 45.5 | "shipped" | "shipped" |
| 1003 | "missing" | "Cho" | null | 12.75 | null | "pending" | null |
| 1006 | "extra" | null | "Fay" | null | 18.25 | null | "pending" |


All the comparison arguments (`keys=`, `tolerance=`, `schema=`, and so on) are available in [tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match), and they can be written in YAML too:

``` yaml
tbl: orders_target.csv
steps:
  - tbl_match:
      tbl_compare: orders_source.csv
      keys: order_id
      tolerance:
        amount: 0.01
```


# Reconciling Aggregates

Sometimes two tables shouldn't have the same rows but should add up to the same numbers: a warehouse table loaded from a source system, a summary table built from detailed transactions, or two systems that record the same activity at different levels of detail. For these, the [tbl_reconcile()](../../reference/Validate.tbl_reconcile.md#pointblank.Validate.tbl_reconcile) validation method compares *aggregates* instead of rows. It groups both tables by the `keys=` columns, aggregates the `measures=` columns (with `agg="sum"` by default, or `"count"`, `"mean"`, `"min"`, or `"max"`), and compares the results group by group. Each group is a test unit.

Here, the orders in the target have been loaded at a finer grain than in the source, so we reconcile the total amount per customer:


``` python
order_lines = pl.DataFrame(
    {
        "order_id": [1001, 1002, 1002, 1004, 1005, 1006],
        "customer": ["Ana", "Ben", "Ben", "Dev", "Eli", "Fay"],
        "amount": [25.00, 40.50, 5.00, 99.99, 5.00, 18.25],
    }
)

validation = (
    pb.Validate(data=order_lines, tbl_name="order_lines")
    .tbl_reconcile(tbl_compare=source, keys="customer", measures="amount")
    .interrogate()
)

validation.get_step_report(i=1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_comparison_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_comparison_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_comparison_tbl p { margin: 0; padding: 0; }
 #pb_comparison_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: 100%; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_comparison_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_comparison_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_comparison_tbl .gt_subtitle { color: #333333; font-size: 85%; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_comparison_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_comparison_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_comparison_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_comparison_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_comparison_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_comparison_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_comparison_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_comparison_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_comparison_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_comparison_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_comparison_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; overflow-x: hidden; }
 #pb_comparison_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_comparison_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_comparison_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_comparison_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_comparison_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_comparison_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_comparison_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_comparison_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_comparison_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_comparison_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_comparison_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_left { text-align: left; }
 #pb_comparison_tbl .gt_center { text-align: center; }
 #pb_comparison_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_comparison_tbl .gt_font_normal { font-weight: normal; }
 #pb_comparison_tbl .gt_font_bold { font-weight: bold; }
 #pb_comparison_tbl .gt_font_italic { font-style: italic; }
 #pb_comparison_tbl .gt_super { font-size: 65%; }
 #pb_comparison_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_comparison_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_comparison_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_comparison_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_comparison_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 100%" aria-labelledby="pb_comparison_tbl-title" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<colgroup>
<col style="width: 30%" />
<col style="width: 70%" />
</colgroup>
<thead>
<tr class="gt_heading">
<th colspan="2" id="pb_comparison_tbl-title" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #CF142B;">✗</span>


ASSERTION <span style="border-style: solid; border-width: thin; border-color: lightblue; padding-left: 2px; padding-right: 2px;"><code style="color: #303030; background-color: transparent; position: relative; bottom: 1px;">tbl_reconcile(keys='customer', measures='amount', agg='sum')</code></span>

<strong>tbl_compare</strong> (5 groups) → <strong>order_lines</strong> (5 groups), aligned by key <code>customer</code><br />
<span style="background:#CF142B; color:white; padding:1px 6px; border-radius:3px; font-size:11px; font-weight:bold; letter-spacing:0.3px;">DIFFERENT</span> 3 of 6 groups fail (50.0%)<br />


<br />
<span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#4CA64C; margin-right:4px; margin-left:4px;"></span>match <strong>3</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#E8A33D; margin-right:4px; margin-left:4px;"></span>changed <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#CF142B; margin-right:4px; margin-left:4px;"></span>missing <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#3D7CC9; margin-right:4px; margin-left:4px;"></span>extra <strong>1</strong></span><br />
<span style="border:1px solid #555; color:#555; padding:0 5px; border-radius:3px; font-size:11px; white-space:nowrap;"><code>sum</code> by <code>customer</code></span>


</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="2" id="pb_comparison_tbl-Column-mismatches-1-of-1-columns" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Column mismatches (1 of 1 columns)</th>
</tr>

<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Column-mismatches-1-of-1-columns">amount_sum</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Column-mismatches-1-of-1-columns"><div style="display:inline-block; vertical-align:middle; width:120px; height:7px; background:#EEE; border-radius:2px; margin-right:8px; margin-left:8px; overflow:hidden;">


1 of 4 matched groups differ (25.0%)</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Changed-groups-1-group" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Changed groups (1 group)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Changed-groups-1-group">Ben</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Changed-groups-1-group"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">amount_sum:</span> <span style="text-decoration:line-through; color:#999;">40.5</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">45.5</span> <span style="color:#888; font-size:11px;">(+5)</span></span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Missing-from-target-1-group" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Missing from target (1 group)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Missing-from-target-1-group">Cho</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Missing-from-target-1-group"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount_sum:</span> 12.75</span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" id="pb_comparison_tbl-Extra-in-target-1-group" class="gt_group_heading" scope="colgroup" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Extra in target (1 group)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px" headers="pb_comparison_tbl-Extra-in-target-1-group">Fay</td>
<td class="gt_row gt_left" style="font-size: 12px" headers="pb_comparison_tbl-Extra-in-target-1-group"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">amount_sum:</span> 18.25</span></td>
</tr>
</tbody>
</table>


`Ben`'s two order lines add up to more than his order in the source, `Cho` is missing, and `Fay` is new. Without `keys=`, [tbl_reconcile()](../../reference/Validate.tbl_reconcile.md#pointblank.Validate.tbl_reconcile) compares grand totals (a single test unit), and `agg="count"` without `measures=` compares the number of rows in each group. Tolerances, `column_map=`, and `partitions=` work as they do in [tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match). Rows with a null key form a group of their own, which is matched with the other table's null-key group.


# Exporting the Results

The results can be serialized with `to_dict()` or `to_json()`, which include the counts, the schema differences, and (by default) samples of the differing rows. If the data could contain sensitive values, leave the samples out:


``` python
print(comparison.to_json(samples=False))
```


    {
      "source_name": "source",
      "target_name": "target",
      "mode": "keyed",
      "keys": [
        "order_id"
      ],
      "compared_columns": [
        "customer",
        "amount",
        "status"
      ],
      "column_map": {},
      "schema_mode": "strict",
      "null_equal": true,
      "partitions": 1,
      "n_source_rows": 5,
      "n_target_rows": 5,
      "n": 6,
      "n_passed": 3,
      "n_failed": 3,
      "identical": false,
      "status_counts": {
        "match": 3,
        "changed": 1,
        "missing": 1,
        "extra": 1,
        "dup_key": 0
      },
      "column_mismatches": {
        "customer": 0,
        "amount": 1,
        "status": 0
      },
      "schema_diff": {
        "matches": true,
        "only_in_source": [],
        "only_in_target": [],
        "dtype_changed": [],
        "renamed": [],
        "case_only": [],
        "reordered": false
      }
    }


The tabular report can be saved as HTML, like any Great Tables table:

``` python
comparison.get_tabular_report().write_raw_html("orders_diff.html")
```


# From the Command Line

The `pb compare` command runs a comparison on files, database tables, or built-in datasets, prints a summary, and can save the HTML report and the JSON results:

``` bash
pb compare orders_source.csv orders_target.csv --key order_id --output-html diff.html
```

Its exit code makes it easy to use in scripts and CI pipelines: `0` if the tables match, `1` if they differ, and `2` if there was an error. Use `--format json` to print the results as JSON. All the comparison settings have command-line options (e.g., `--tolerance amount=0.01`, `--schema common`); see the [CLI reference](../../user-guide/the-pointblank-cli/cli-reference.md) for the full list.

For tables that are too large to fit in memory, see [Comparing Large Files](../../user-guide/table-comparison/comparing-large-files.md).
