# Validate.tbl_match()


Validate whether the target table matches a comparison table.


Usage

``` python
Validate.tbl_match(
    tbl_compare,
    keys=None,
    columns=None,
    column_map=None,
    tolerance=None,
    normalize=None,
    null_equal=True,
    schema="strict",
    dup_keys="flag",
    order_by=None,
    partitions=None,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```


The [tbl_match()](Validate.tbl_match.md#pointblank.Validate.tbl_match) method compares the target table, row by row, against a comparison table that holds the expected data (the *source*). Rows of the two tables are aligned and every aligned row becomes a test unit: it passes if the row is present in both tables with equal values, and fails if its values changed, if it's missing from the target, if it's an extra row in the target, or if its key is duplicated. Schema differences (columns added, removed, retyped, or reordered) are checked too.

Rows are aligned in one of three ways:

- **by key** (`keys=` given): rows are matched on one or more key columns, so row order doesn't matter; this is the most informative mode
- **by position** (`keys=None`, the default): the first row of each table is compared, then the second, and so on
- **as multisets** (`keys="*"`): rows are compared as whole values regardless of order, with duplicates counted (for tables without a key)

The comparison runs inside the table's own engine (it is backend-agnostic, and database tables are compared in the database). Use <a href="../reference/Validate.get_step_report.html#pointblank.Validate.get_step_report" class="gdls-link"><code>get_step_report()</code></a> to see a detailed report of the differences, and <a href="../reference/Validate.get_data_extracts.html#pointblank.Validate.get_data_extracts" class="gdls-link"><code>get_data_extracts()</code></a> to get the failing rows. The same comparison is available outside of a validation with <a href="../reference/compare.html#pointblank.compare" class="gdls-link"><code>compare()</code></a>.


## Parameters


`tbl_compare: Any`  
The comparison table (the expected data) to validate against. This can be a DataFrame object (Polars or Pandas), an Ibis table object, a path to a CSV or Parquet file, or a callable that returns a table. If a callable is provided, it will be executed during interrogation to obtain the comparison table.

`keys: str | list[str] | None = None`  
One or more columns that uniquely identify a row, used to align the rows of the two tables. Key names refer to the comparison table (use `column_map=` if the target names them differently). If `None` (the default), rows are aligned by position. Use `"*"` to compare the tables as multisets of rows.

`columns: str | list[str] | None = None`  
The comparison-table columns whose values should be compared. By default, all columns present in both tables (other than the keys) are compared. Giving `columns=` also limits the schema check to the keys and these columns.

`column_map: dict[str, str] | None = None`  
A mapping of comparison-table column names to target column names, for renamed columns.

`tolerance: Any = None`  
Tolerance for numeric and datetime comparisons. A number is an absolute tolerance for all numeric columns. A dict with `"atol"` and/or `"rtol"` keys applies to all numeric columns, where values are equal if `|expected - actual| <= atol + rtol * |expected|`. A dict mapping column names to either of the above sets per-column tolerances. Datetime tolerances are given as a `datetime.timedelta` or a string like `"1s"` or `"500ms"`.

`normalize: dict[str, Any] | None = None`  
Per-column transformations applied to both tables before comparing, as a dict mapping a column name to a list of operations: `"strip"`, `"lower"`, `"upper"`, `"collapse_whitespace"`, or `{"round": n}`. The string operations only apply to string columns and `round` only to numeric ones.

`null_equal: bool = ``True`  
Should two null values count as equal? By default, they do. NaN values in floating-point columns are treated as null.

`schema: Literal[``"strict", `<span class="st">`"common"``]`</span>` = ``"strict"`  
How schema differences affect the result. With `"strict"` (the default), any schema difference (other than renames given in `column_map=`) means the tables don't match, and every test unit fails. With `"common"`, only the shared columns are compared and schema differences are reported without failing any test units.

`dup_keys: Literal[``"flag", `<span class="st">`"compare"``]`</span>` = ``"flag"`  
Rows whose key appears more than once in either table always fail. With `"flag"` (the default), the step report lists the duplicated keys. With `"compare"`, the step report also compares the duplicated rows pairwise to show which columns differ.

`order_by: str | list[str] | None = None`  
Column(s) defining the row order for positional alignment on backends without an inherent row order (e.g., PySpark or database tables). Without it, such tables are collected and compared in the order in which their rows are returned; use `keys=` for a reliable alignment. Ignored when `keys=` is given.

`partitions: int | None = None`  
Compare the tables in this many passes, each covering a disjoint subset of the key values (by key hash), to bound memory use for very large tables. Each pass reads the tables again, so this trades time for memory; the results are identical to a single-pass comparison. Requires `keys=` (not `"*"`), and Polars, DuckDB, or Ibis tables (including CSV/Parquet paths). Sundering isn't available for the step when partitions are used.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table. Have a look at the *Preprocessing* section for more information on how to use this argument.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect. Look at the *Thresholds* section for information on how to set threshold levels.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the <a href="../reference/Actions.html#pointblank.Actions" class="gdls-link"><code>Actions</code></a> class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = ``True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like <a href="../reference/has_columns.html#pointblank.has_columns" class="gdls-link"><code>has_columns()</code></a> and <a href="../reference/has_rows.html#pointblank.has_rows" class="gdls-link"><code>has_rows()</code></a> can be used here to conditionally activate a step based on properties of the target table.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is inferred automatically from the assertion type. This label appears in the validation report and feeds the overall and per-dimension health scores.


## Returns


`Validate`  
The [Validate](Validate.md#pointblank.Validate) object with the added validation step.


## Test Units

Each aligned row is a test unit. With `keys=`, that's every key value found in either table (a duplicated key counts as many units as its larger number of occurrences); by position, it's the number of rows in the longer table; as multisets, it's every row copy. A test unit passes only if its row is in both tables with all compared values equal. If the strict schema check fails, all test units fail. When both tables are empty, the step has a single test unit, which passes if the schemas match.

Because failures are counted per row, thresholds can be expressed as proportions (e.g., a [warning](Validate.warning.md#pointblank.Validate.warning) when more than 1% of rows differ). To fail on any difference, use an absolute threshold of `1`.

> **Important: Important**
>
> In Pointblank `v1.0.1` and earlier, [tbl_match()](Validate.tbl_match.md#pointblank.Validate.tbl_match) had a single test unit that failed if the tables differed in any way. Whether a step passes or fails is unchanged, but the counts (`n`, `n_passed`, `n_failed`, and the fractions) now reflect rows. This changes the meaning of fractional thresholds: `thresholds=0.1` used to be exceeded by any difference, and is now exceeded only when at least 10% of the rows fail. To keep the earlier behavior of reacting to any difference, use an absolute threshold of `1` (e.g., `thresholds=1` or `Thresholds(warning=1)`).

The data extract for the step (from <a href="../reference/Validate.get_data_extracts.html#pointblank.Validate.get_data_extracts" class="gdls-link"><code>get_data_extracts()</code></a>) contains the failing rows of *both* tables: the key column(s), a `_status_` column (`"changed"`, `"missing"`, `"extra"`, or `"dup_key"`), and a pair of `<column>_source`/`<column>_target` columns for each compared column. For <a href="../reference/Validate.get_sundered_data.html#pointblank.Validate.get_sundered_data" class="gdls-link"><code>get_sundered_data()</code></a>, target rows with a matching row in the comparison table pass; rows missing from the target can't be part of the sundered data. Sundering for this step is available with in-memory tables (not lazy or database tables) and isn't available in multiset mode.


## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Note that the same preprocessing is **not** applied to the comparison table; only the target table is preprocessed. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](Validate.md#pointblank.Validate) object or used in subsequent validation steps.


## Thresholds

The `thresholds=` parameter is used to set the failure-condition levels for the validation step. If they are set here at the step level, these thresholds will override any thresholds set at the global level in `Validate(thresholds=...)`.

There are three threshold levels: 'warning', 'error', and 'critical'. The threshold values can either be set as a proportion failing of all test units (a value between `0` to `1`), or, the absolute number of failing test units (as integer that's `1` or greater).

Thresholds can be defined using one of these input schemes:

1.  use the <a href="../reference/Thresholds.html#pointblank.Thresholds" class="gdls-link"><code>Thresholds</code></a> class (the most direct way to create thresholds)
2.  provide a tuple of 1-3 values, where position `0` is the 'warning' level, position `1` is the 'error' level, and position `2` is the 'critical' level
3.  create a dictionary of 1-3 value entries; the valid keys: are 'warning', 'error', and 'critical'
4.  a single integer/float value denoting absolute number or fraction of failing test units for the 'warning' level only

If the number of failing test units exceeds set thresholds, the validation step will be marked as 'warning', 'error', or 'critical'. All of the threshold levels don't need to be set, you're free to set any combination of them.

Aside from reporting failure conditions, thresholds can be used to determine the actions to take for each level of failure (using the `actions=` parameter).


## Cross-Backend Validation

The two tables can come from different backends (e.g., a Polars DataFrame and a Pandas DataFrame, or a database table and a CSV file). Tables on the same backend are compared in place (so two tables in the same database are compared inside the database). Tables on different backends are both converted to Polars (or to Pandas, if Polars isn't installed) before the comparison. Key columns whose types differ between the tables (e.g., integer keys that became floats in Pandas because of a missing value) are cast to a common type.

Values in columns whose types differ are compared numerically when both columns are numeric and as strings otherwise; note that such differences also fail the strict schema check (use `schema="common"` to only report them).


## Examples

For the examples here, we'll create two simple tables to demonstrate the [tbl_match()](Validate.tbl_match.md#pointblank.Validate.tbl_match) validation.


``` python
import pointblank as pb
import polars as pl

# Create the first table
tbl_1 = pl.DataFrame({
    "a": [1, 2, 3, 4],
    "b": ["w", "x", "y", "z"],
    "c": [4.0, 5.0, 6.0, 7.0]
})

# Create an identical table
tbl_2 = pl.DataFrame({
    "a": [1, 2, 3, 4],
    "b": ["w", "x", "y", "z"],
    "c": [4.0, 5.0, 6.0, 7.0]
})

pb.preview(tbl_1)
```


<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans&display=swap');
#pb_preview_tbl table {
          font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Helvetica Neue', 'Fira Sans', 'Droid Sans', Arial, sans-serif;
          -webkit-font-smoothing: antialiased;
          -moz-osx-font-smoothing: grayscale;
        }

#pb_preview_tbl thead, tbody, tfoot, tr, td, th { border-style: none; }
 tr { background-color: transparent; }
#pb_preview_tbl p { margin: 0; padding: 0; }
 #pb_preview_tbl .gt_table { display: table; border-collapse: collapse; line-height: normal; margin-left: auto; margin-right: auto; color: #333333; font-size: 16px; font-weight: normal; font-style: normal; background-color: #FFFFFF; width: auto; border-top-style: solid; border-top-width: 2px; border-top-color: #A8A8A8; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #A8A8A8; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; }
 #pb_preview_tbl .gt_caption { padding-top: 4px; padding-bottom: 4px; }
 #pb_preview_tbl .gt_title { color: #333333; font-size: 125%; font-weight: initial; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; border-bottom-color: #FFFFFF; border-bottom-width: 0; }
 #pb_preview_tbl .gt_subtitle { color: #333333; font-size: 12px; font-weight: initial; padding-top: 3px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; border-top-color: #FFFFFF; border-top-width: 0; }
 #pb_preview_tbl .gt_heading { background-color: #FFFFFF; text-align: left; border-bottom-color: #FFFFFF; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_preview_tbl .gt_bottom_border { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_preview_tbl .gt_col_headings { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; }
 #pb_preview_tbl .gt_col_heading { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; border-left-style: solid; border-left-width: 1px; border-left-color: #F2F2F2; border-right-style: solid; border-right-width: 1px; border-right-color: #F2F2F2; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; padding-left: 5px; padding-right: 5px; overflow-x: hidden; }
 #pb_preview_tbl .gt_column_spanner_outer { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: normal; text-transform: inherit; padding-top: 0; padding-bottom: 0; padding-left: 4px; padding-right: 4px; }
 #pb_preview_tbl .gt_column_spanner_outer:first-child { padding-left: 0; }
 #pb_preview_tbl .gt_column_spanner_outer:last-child { padding-right: 0; }
 #pb_preview_tbl .gt_column_spanner { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: bottom; padding-top: 5px; padding-bottom: 5px; overflow-x: hidden; display: inline-block; width: 100%; }
 #pb_preview_tbl .gt_spanner_row { border-bottom-style: hidden; }
 #pb_preview_tbl .gt_group_heading { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 1px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 1px; border-right-color: #D3D3D3; vertical-align: middle; text-align: left; }
 #pb_preview_tbl .gt_empty_group_heading { padding: 0.5px; color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; vertical-align: middle; }
 #pb_preview_tbl .gt_from_md> :first-child { margin-top: 0; }
 #pb_preview_tbl .gt_from_md> :last-child { margin-bottom: 0; }
 #pb_preview_tbl .gt_row { padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; margin: 10px; border-top-style: solid; border-top-width: 1px; border-top-color: #D3D3D3; border-left-style: solid; border-left-width: 1px; border-left-color: #E9E9E9; border-right-style: solid; border-right-width: 1px; border-right-color: #E9E9E9; vertical-align: middle; overflow-x: hidden; }
 #pb_preview_tbl .gt_stub { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; }
 #pb_preview_tbl .gt_indent_1 { text-indent: 5px; }
 #pb_preview_tbl .gt_indent_2 { text-indent: calc(5px * 2); }
 #pb_preview_tbl .gt_indent_3 { text-indent: calc(5px * 3); }
 #pb_preview_tbl .gt_indent_4 { text-indent: calc(5px * 4); }
 #pb_preview_tbl .gt_indent_5 { text-indent: calc(5px * 5); }
 #pb_preview_tbl .gt_stub_row_group { color: #333333; background-color: #FFFFFF; font-size: 100%; font-weight: initial; text-transform: inherit; border-right-style: solid; border-right-width: 2px; border-right-color: #D3D3D3; padding-left: 5px; padding-right: 5px; vertical-align: top; }
 #pb_preview_tbl .gt_row_group_first td { border-top-width: 2px; }
 #pb_preview_tbl .gt_row_group_first th { border-top-width: 2px; }
 #pb_preview_tbl .gt_striped { color: #333333; background-color: #F4F4F4; }
 #pb_preview_tbl .gt_table_body { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_preview_tbl .gt_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_preview_tbl .gt_first_summary_row { border-top-style: solid; border-top-width: 2px; border-top-color: #D3D3D3; }
 #pb_preview_tbl .gt_last_summary_row_top { border-bottom-style: solid; border-bottom-width: 2px; border-bottom-color: #D3D3D3; }
 #pb_preview_tbl .gt_grand_summary_row { color: #333333; background-color: #FFFFFF; text-transform: inherit; padding-top: 8px; padding-bottom: 8px; padding-left: 5px; padding-right: 5px; }
 #pb_preview_tbl .gt_first_grand_summary_row_bottom { border-top-style: double; border-top-width: 6px; border-top-color: #D3D3D3; }
 #pb_preview_tbl .gt_last_grand_summary_row_top { border-bottom-style: double; border-bottom-width: 6px; border-bottom-color: #D3D3D3; }
 #pb_preview_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_preview_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_preview_tbl .gt_left { text-align: left; }
 #pb_preview_tbl .gt_center { text-align: center; }
 #pb_preview_tbl .gt_right { text-align: right; font-variant-numeric: tabular-nums; }
 #pb_preview_tbl .gt_font_normal { font-weight: normal; }
 #pb_preview_tbl .gt_font_bold { font-weight: bold; }
 #pb_preview_tbl .gt_font_italic { font-style: italic; }
 #pb_preview_tbl .gt_super { font-size: 65%; }
 #pb_preview_tbl .gt_footnotes { color: font-color(#FFFFFF); background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_preview_tbl .gt_footnote { margin: 0px; font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; }
 #pb_preview_tbl .gt_sourcenotes { color: #333333; background-color: #FFFFFF; border-bottom-style: none; border-bottom-width: 2px; border-bottom-color: #D3D3D3; border-left-style: none; border-left-width: 2px; border-left-color: #D3D3D3; border-right-style: none; border-right-width: 2px; border-right-color: #D3D3D3; }
 #pb_preview_tbl .gt_sourcenote { font-size: 90%; padding-top: 4px; padding-bottom: 4px; padding-left: 5px; padding-right: 5px; text-align: left; }
 #pb_preview_tbl .gt_footnote_marks { font-size: 75%; vertical-align: 0.4em; position: initial; }
 #pb_preview_tbl .gt_asterisk { font-size: 100%; vertical-align: 0; }
 
</style>

<table class="gt_table" style="table-layout: fixed;; width: 0px" data-quarto-disable-processing="true" data-quarto-bootstrap="false">
<thead>
<tr class="gt_heading">
<th colspan="4" class="gt_heading gt_title gt_font_normal"><div>

<span style="background-color: #0075FF; color: #FFFFFF; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 10px 5px 0px; border: solid 1px #0075FF; font-weight: bold; padding: 2px 10px 2px 10px; font-size: 10px;">Polars</span><span style="background-color: #eecbff; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 5px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">Rows</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #eecbff; padding: 2px 15px 2px 15px; font-size: 10px;">4</span><span style="background-color: #BDE7B4; color: #333333; padding: 0.5em 0.5em; position: inherit; text-transform: uppercase; margin: 5px 0px 5px 3px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">Columns</span><span style="background-color: none; color: #333333; padding: 0.5em 0.5em; position: inherit; margin: 5px 0px 5px -4px; font-weight: bold; border: solid 1px #BDE7B4; padding: 2px 15px 2px 15px; font-size: 10px;">3</span>

</div></th>
</tr>
<tr class="gt_col_headings">
<th id="pb_preview_tbl-_row_num_" class="gt_col_heading gt_columns_bottom_border gt_right" style="color: gray20; font-family: IBM Plex Mono; font-size: 12px" scope="col"></th>
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

<em>Float64</em>

</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">1</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">1</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">w</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4.0</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">2</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">2</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">x</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">5.0</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">3</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">3</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">y</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">6.0</td>
</tr>
<tr>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E; color: gray; font-family: IBM Plex Mono; font-size: 10px; border-right: 2px solid #6699CC80">4</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">4</td>
<td class="gt_row gt_left" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">z</td>
<td class="gt_row gt_right" style="height: 14px; padding: 4px; white-space: nowrap; text-overflow: ellipsis; overflow: hidden; color: black; font-family: IBM Plex Mono; font-size: 12px; border-top: 1px solid #E9E9E; border-bottom: 1px solid #E9E9E">7.0</td>
</tr>
</tbody>
</table>


Let's validate that `tbl_1` matches `tbl_2`. Since these tables are identical, the validation should pass.


``` python
validation = (
    pb.Validate(data=tbl_1)
    .tbl_match(tbl_compare=tbl_2)
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
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">4</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">4<br />
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


The validation table shows that all four test units (one per row) passed, indicating that the two tables match completely.

Now, let's compare against a table that has one different value, is missing a row, and has an extra row. Since the rows can be identified by column `a`, we'll use it as the key.


``` python
tbl_3 = pl.DataFrame({
    "a": [1, 2, 4, 5],
    "b": ["w", "x", "z", "v"],
    "c": [4.0, 5.5, 7.0, 8.0]  # Changed 5.0 to 5.5
})

validation = (
    pb.Validate(data=tbl_3)
    .tbl_match(tbl_compare=tbl_1, keys="a")
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
<td class="gt_row gt_left" style="height: 40px; background-color: #4CA64C66; color: transparent; font-size: 0px">#4CA64C66</td>
<td class="gt_row gt_right" style="height: 40px; color: #666666; font-size: 13px; font-weight: bold">1</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px"><div style="margin: 0; padding: 0; display: inline-block; height: 30px; vertical-align: middle; width: 16%;">
<img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzBweCIgaGVpZ2h0PSIzMHB4IiB2aWV3Ym94PSIwIDAgNjcgNjciIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8dGl0bGU+dGJsX21hdGNoPC90aXRsZT4KICAgIDxnIGlkPSJJY29ucyIgc3Ryb2tlPSJub25lIiBzdHJva2Utd2lkdGg9IjEiIGZpbGw9Im5vbmUiIGZpbGwtcnVsZT0iZXZlbm9kZCI+CiAgICAgICAgPGcgaWQ9InRibF9tYXRjaCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDAuNzU4NjIxKSI+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik01Ni43MTIyMzQsMS4wMTQ2NjkzNSBDNTkuMTk3NTE1MywxLjAxNDY2OTM1IDYxLjQ0NzUxNTMsMi4wMjIwMjg2NyA2My4wNzYxOTUsMy42NTA3MDgzMiBDNjQuNzA0ODc0Nyw1LjI3OTM4Nzk4IDY1LjcxMjIzNCw3LjUyOTM4Nzk4IDY1LjcxMjIzNCwxMC4wMTQ2Njk0IEw2NS43MTIyMzQsMTAuMDE0NjY5NCBMNjUuNzEyMjM0LDY1LjAxNDY2OTQgTDEwLjcxMjIzNCw2NS4wMTQ2Njk0IEM4LjIyNjk1MjU5LDY1LjAxNDY2OTQgNS45NzY5NTI1OSw2NC4wMDczMSA0LjM0ODI3Mjk0LDYyLjM3ODYzMDQgQzIuNzE5NTkzMjgsNjAuNzQ5OTUwNyAxLjcxMjIzMzk3LDU4LjQ5OTk1MDcgMS43MTIyMzM5Nyw1Ni4wMTQ2Njk0IEwxLjcxMjIzMzk3LDU2LjAxNDY2OTQgTDEuNzEyMjMzOTcsMTAuMDE0NjY5NCBDMS43MTIyMzM5Nyw3LjUyOTM4Nzk4IDIuNzE5NTkzMjgsNS4yNzkzODc5OCA0LjM0ODI3Mjk0LDMuNjUwNzA4MzIgQzUuOTc2OTUyNTksMi4wMjIwMjg2NyA4LjIyNjk1MjU5LDEuMDE0NjY5MzUgMTAuNzEyMjM0LDEuMDE0NjY5MzUgTDEwLjcxMjIzNCwxLjAxNDY2OTM1IFoiIGlkPSJyZWN0YW5nbGUiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLXdpZHRoPSIyIiBmaWxsPSIjRkZGRkZGIiAvPgogICAgICAgICAgICA8ZyBpZD0iZXF1YWwiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ2LjAyNjYxMSwgMjAuNzEwMTIyKSByb3RhdGUoLTkwLjAwMDAwMCkgdHJhbnNsYXRlKC00Ni4wMjY2MTEsIC0yMC43MTAxMjIpIHRyYW5zbGF0ZSg0Mi41MjY2MTEsIDE2LjIxMDEyMikiIHN0cm9rZT0iIzAwMDAwMCIgc3Ryb2tlLWxpbmVjYXA9InNxdWFyZSI+CiAgICAgICAgICAgICAgICA8bGluZSB4MT0iMi4yMTIyMzM5NyIgeTE9IjAuNTE0NjY5MzUzIiB4Mj0iMi4yMTIyMzM5NyIgeTI9IjcuNTg1NzM3MTYiIGlkPSJMaW5lIj48L2xpbmU+CiAgICAgICAgICAgICAgICA8bGluZSB4MT0iNS4yMTIyMzM5NyIgeTE9IjAuNTE0NjY5MzUzIiB4Mj0iNS4yMTIyMzM5NyIgeTI9IjcuNTg1NzM3MTYiIGlkPSJMaW5lIj48L2xpbmU+CiAgICAgICAgICAgIDwvZz4KICAgICAgICAgICAgPGcgaWQ9ImVxdWFsIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyMS4zOTc4NTcsIDQ1LjMxOTIxNykgcm90YXRlKC05MC4wMDAwMDApIHRyYW5zbGF0ZSgtMjEuMzk3ODU3LCAtNDUuMzE5MjE3KSB0cmFuc2xhdGUoMTcuODk3ODU3LCA0MC44MTkyMTcpIiBzdHJva2U9IiMwMDAwMDAiIHN0cm9rZS1saW5lY2FwPSJzcXVhcmUiPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjIuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjIuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICAgICAgPGxpbmUgeDE9IjUuMjEyMjMzOTciIHkxPSIwLjUxNDY2OTM1MyIgeDI9IjUuMjEyMjMzOTciIHkyPSI3LjU4NTczNzE2IiBpZD0iTGluZSI+PC9saW5lPgogICAgICAgICAgICA8L2c+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik0yMS4zODgyNDE5LDcuNzc4Njk3ODMgQzIxLjM1ODQyOTgsNy43NzkzNTE3NyAyMS4zMjg3MDQsNy43ODIxNjM2NyAyMS4yOTkyOTk2LDcuNzg3MTE5MTQgTDkuMDkwMTQ4MjQsNy43ODcxMTkxNCBDOC43NTAyOTQzMSw3Ljc4NzE1MzEyIDguNDc0Nzk2NzcsOC4wNjI2NTA2NyA4LjQ3NDc2Mjc5LDguNDAyNTA0NiBMOC40NzQ3NjI3OSwxNi4yOTkxNDk4IEM4LjQ2Mzc3ODc0LDE2LjM2NTYwNjEgOC40NjM3Nzg3NCwxNi40MzM0MTQ5IDguNDc0NzYyNzksMTYuNDk5ODcxMyBMOC40NzQ3NjI3OSwyNC45MTQ1NDYxIEM4LjQ2Mzc3ODc0LDI0Ljk4MTAwMjUgOC40NjM3Nzg3NCwyNS4wNDg4MTEyIDguNDc0NzYyNzksMjUuMTE1MjY3NiBMOC40NzQ3NjI3OSwzMy4wMTc5MjI2IEM4LjQ3NDc5Njc3LDMzLjM1Nzc3NjYgOC43NTAyOTQzMSwzMy42MzMyNzQxIDkuMDkwMTQ4MjQsMzMuNjMzMzA4MSBMMjEuMjk0NDkxNiwzMy42MzMzMDgxIEMyMS4zNjA5NDc5LDMzLjY0NDI5MjEgMjEuNDI4NzU2NywzMy42NDQyOTIxIDIxLjQ5NTIxMzEsMzMuNjMzMzA4MSBMMzMuNzA1NTY2MywzMy42MzMzMDgxIEMzNC4wNDU0MjAyLDMzLjYzMzI3NDEgMzQuMzIwOTE3OCwzMy4zNTc3NzY2IDM0LjMyMDk1MTcsMzMuMDE3OTIyNiBMMzQuMzIwOTUxNywyNS4xMjEyNzc1IEMzNC4zMzE5MzU4LDI1LjA1NDgyMTEgMzQuMzMxOTM1OCwyNC45ODcwMTIzIDM0LjMyMDk1MTcsMjQuOTIwNTU2IEwzNC4zMjA5NTE3LDE2LjUwNTg4MTEgQzM0LjMzMTkzNTgsMTYuNDM5NDI0OCAzNC4zMzE5MzU4LDE2LjM3MTYxNiAzNC4zMjA5NTE3LDE2LjMwNTE1OTYgTDM0LjMyMDk1MTcsOC40MDI1MDQ2IEMzNC4zMjA5MTc4LDguMDYyNjUwNjcgMzQuMDQ1NDIwMiw3Ljc4NzE1MzEyIDMzLjcwNTU2NjMsNy43ODcxMTkxNCBMMjEuNDkyODA5NCw3Ljc4NzExOTE0IEMyMS40NTgyNTY1LDcuNzgxMzQzNjkgMjEuNDIzMjczNiw3Ljc3ODY5NzgzIDIxLjM4ODI0MTksNy43Nzg2OTc4MyBaIE05LjcwNTUzMzY5LDkuMDE3ODkwMDUgTDIwLjc4MjQ3MTgsOS4wMTc4OTAwNSBMMjAuNzgyNDcxOCwxNS43ODcxMyBMOS43MDU1MzM2OSwxNS43ODcxMyBMOS43MDU1MzM2OSw5LjAxNzg5MDA1IFogTTIyLjAxMzI0MjcsOS4wMTc4OTAwNSBMMzMuMDkwMTgwOCw5LjAxNzg5MDA1IEwzMy4wOTAxODA4LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDE1Ljc4NzEzIEwyMi4wMTMyNDI3LDkuMDE3ODkwMDUgWiBNOS43MDU1MzM2OSwxNy4wMTc5MDA5IEwyMC43ODI0NzE4LDE3LjAxNzkwMDkgTDIwLjc4MjQ3MTgsMjQuNDAyNTI2MyBMOS43MDU1MzM2OSwyNC40MDI1MjYzIEw5LjcwNTUzMzY5LDE3LjAxNzkwMDkgWiBNMjIuMDEzMjQyNywxNy4wMTc5MDA5IEwzMy4wOTAxODA4LDE3LjAxNzkwMDkgTDMzLjA5MDE4MDgsMjQuNDAyNTI2MyBMMjIuMDEzMjQyNywyNC40MDI1MjYzIEwyMi4wMTMyNDI3LDE3LjAxNzkwMDkgWiBNOS43MDU1MzM2OSwyNS42MzMyOTcyIEwyMC43ODI0NzE4LDI1LjYzMzI5NzIgTDIwLjc4MjQ3MTgsMzIuNDAyNTM3MiBMOS43MDU1MzM2OSwzMi40MDI1MzcyIEw5LjcwNTUzMzY5LDI1LjYzMzI5NzIgWiBNMjIuMDEzMjQyNywyNS42MzMyOTcyIEwzMy4wOTAxODA4LDI1LjYzMzI5NzIgTDMzLjA5MDE4MDgsMzIuNDAyNTM3MiBMMjIuMDEzMjQyNywzMi40MDI1MzcyIEwyMi4wMTMyNDI3LDI1LjYzMzI5NzIgWiIgaWQ9InRhYmxlIiBmaWxsPSIjMDAwMDAwIiBmaWxsLXJ1bGU9Im5vbnplcm8iIC8+CiAgICAgICAgICAgIDxwYXRoIGQ9Ik00Ni4wMTY5OTUzLDMyLjM4Nzc5MjYgQzQ1Ljk4NzE4MzIsMzIuMzg4NDQ2NSA0NS45NTc0NTc1LDMyLjM5MTI1ODQgNDUuOTI4MDUzLDMyLjM5NjIxMzkgTDMzLjcxODkwMTYsMzIuMzk2MjEzOSBDMzMuMzc5MDQ3NywzMi4zOTYyNDc5IDMzLjEwMzU1MDIsMzIuNjcxNzQ1NCAzMy4xMDM1MTYyLDMzLjAxMTU5OTMgTDMzLjEwMzUxNjIsNDAuOTA4MjQ0NSBDMzMuMDkyNTMyMiw0MC45NzQ3MDA5IDMzLjA5MjUzMjIsNDEuMDQyNTA5NyAzMy4xMDM1MTYyLDQxLjEwODk2NiBMMzMuMTAzNTE2Miw0OS41MjM2NDA4IEMzMy4wOTI1MzIyLDQ5LjU5MDA5NzIgMzMuMDkyNTMyMiw0OS42NTc5MDYgMzMuMTAzNTE2Miw0OS43MjQzNjI0IEwzMy4xMDM1MTYyLDU3LjYyNzAxNzQgQzMzLjEwMzU1MDIsNTcuOTY2ODcxMyAzMy4zNzkwNDc3LDU4LjI0MjM2ODkgMzMuNzE4OTAxNiw1OC4yNDI0MDI4IEw0NS45MjMyNDUsNTguMjQyNDAyOCBDNDUuOTg5NzAxNCw1OC4yNTMzODY5IDQ2LjA1NzUxMDEsNTguMjUzMzg2OSA0Ni4xMjM5NjY1LDU4LjI0MjQwMjggTDU4LjMzNDMxOTcsNTguMjQyNDAyOCBDNTguNjc0MTczNiw1OC4yNDIzNjg5IDU4Ljk0OTY3MTIsNTcuOTY2ODcxMyA1OC45NDk3MDUxLDU3LjYyNzAxNzQgTDU4Ljk0OTcwNTEsNDkuNzMwMzcyMiBDNTguOTYwNjg5Miw0OS42NjM5MTU4IDU4Ljk2MDY4OTIsNDkuNTk2MTA3MSA1OC45NDk3MDUxLDQ5LjUyOTY1MDcgTDU4Ljk0OTcwNTEsNDEuMTE0OTc1OSBDNTguOTYwNjg5Miw0MS4wNDg1MTk1IDU4Ljk2MDY4OTIsNDAuOTgwNzEwNyA1OC45NDk3MDUxLDQwLjkxNDI1NDQgTDU4Ljk0OTcwNTEsMzMuMDExNTk5MyBDNTguOTQ5NjcxMiwzMi42NzE3NDU0IDU4LjY3NDE3MzYsMzIuMzk2MjQ3OSA1OC4zMzQzMTk3LDMyLjM5NjIxMzkgTDQ2LjEyMTU2MjgsMzIuMzk2MjEzOSBDNDYuMDg3MDA5OSwzMi4zOTA0Mzg0IDQ2LjA1MjAyNywzMi4zODc3OTI2IDQ2LjAxNjk5NTMsMzIuMzg3NzkyNiBaIE0zNC4zMzQyODcxLDMzLjYyNjk4NDggTDQ1LjQxMTIyNTIsMzMuNjI2OTg0OCBMNDUuNDExMjI1Miw0MC4zOTYyMjQ4IEwzNC4zMzQyODcxLDQwLjM5NjIyNDggTDM0LjMzNDI4NzEsMzMuNjI2OTg0OCBaIE00Ni42NDE5OTYxLDMzLjYyNjk4NDggTDU3LjcxODkzNDIsMzMuNjI2OTg0OCBMNTcuNzE4OTM0Miw0MC4zOTYyMjQ4IEw0Ni42NDE5OTYxLDQwLjM5NjIyNDggTDQ2LjY0MTk5NjEsMzMuNjI2OTg0OCBaIE0zNC4zMzQyODcxLDQxLjYyNjk5NTcgTDQ1LjQxMTIyNTIsNDEuNjI2OTk1NyBMNDUuNDExMjI1Miw0OS4wMTE2MjExIEwzNC4zMzQyODcxLDQ5LjAxMTYyMTEgTDM0LjMzNDI4NzEsNDEuNjI2OTk1NyBaIE00Ni42NDE5OTYxLDQxLjYyNjk5NTcgTDU3LjcxODkzNDIsNDEuNjI2OTk1NyBMNTcuNzE4OTM0Miw0OS4wMTE2MjExIEw0Ni42NDE5OTYxLDQ5LjAxMTYyMTEgTDQ2LjY0MTk5NjEsNDEuNjI2OTk1NyBaIE0zNC4zMzQyODcxLDUwLjI0MjM5MiBMNDUuNDExMjI1Miw1MC4yNDIzOTIgTDQ1LjQxMTIyNTIsNTcuMDExNjMxOSBMMzQuMzM0Mjg3MSw1Ny4wMTE2MzE5IEwzNC4zMzQyODcxLDUwLjI0MjM5MiBaIE00Ni42NDE5OTYxLDUwLjI0MjM5MiBMNTcuNzE4OTM0Miw1MC4yNDIzOTIgTDU3LjcxODkzNDIsNTcuMDExNjMxOSBMNDYuNjQxOTk2MSw1Ny4wMTE2MzE5IEw0Ni42NDE5OTYxLDUwLjI0MjM5MiBaIiBpZD0idGFibGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" />

tbl_match()

</div></td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">None</td>
<td class="gt_row gt_left" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5; white-space: nowrap; text-overflow: ellipsis; overflow: hidden">EXTERNAL TABLE<br />
<span style="font-size: 8px; font-weight: 600; letter-spacing: 0.5px; color: #5A6B7B;">key: a</span></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3"><img src="data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjVweCIgaGVpZ2h0PSIyNXB4IiB2aWV3Ym94PSIwIDAgMjUgMjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgc3R5bGU9InZlcnRpY2FsLWFsaWduOiBtaWRkbGU7Ij4KICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJ1bmNoYW5nZWQiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuNTAwMDAwLCAwLjU3MDE0NykiPgogICAgICAgICAgICA8cmVjdCBpZD0iUmVjdGFuZ2xlIiB4PSIwLjEyNTEzMjUwNiIgeT0iMCIgd2lkdGg9IjIzLjc0OTczNSIgaGVpZ2h0PSIyMy43ODk0NzM3IiAvPgogICAgICAgICAgICA8cGF0aCBkPSJNNS44MDM3NTA0Niw4LjE4MTk0NzM2IEMzLjc3MTkxODMyLDguMTgxOTQ3MzYgMi4xMTg3NTA0Niw5LjgzNDk1MzI4IDIuMTE4NzUwNDYsMTEuODY2OTQ3NCBDMi4xMTg3NTA0NiwxMy44OTg5NDE0IDMuNzcxOTE4MzIsMTUuNTUxOTQ3NCA1LjgwMzc1MDQ2LDE1LjU1MTk0NzQgQzcuODM1NTgyNiwxNS41NTE5NDc0IDkuNDg4NzUwNDYsMTMuODk4OTQxNCA5LjQ4ODc1MDQ2LDExLjg2Njk0NzQgQzkuNDg4NzUwNDYsOS44MzQ5NTMyOCA3LjgzNTUyODYzLDguMTgxOTQ3MzYgNS44MDM3NTA0Niw4LjE4MTk0NzM2IFogTTUuODAzNzUwNDYsMTQuODE0OTE1IEM0LjE3ODIxOTk3LDE0LjgxNDkxNSAyLjg1NTc4Mjg1LDEzLjQ5MjQ3NzggMi44NTU3ODI4NSwxMS44NjY5NDc0IEMyLjg1NTc4Mjg1LDEwLjI0MTQxNjkgNC4xNzgyMTk5Nyw4LjkxODk3OTc1IDUuODAzNzUwNDYsOC45MTg5Nzk3NSBDNy40MjkyODA5NSw4LjkxODk3OTc1IDguNzUxNzE4MDcsMTAuMjQxNDE2OSA4Ljc1MTcxODA3LDExLjg2Njk0NzQgQzguNzUxNzE4MDcsMTMuNDkyNDc3OCA3LjQyOTI4MDk1LDE0LjgxNDkxNSA1LjgwMzc1MDQ2LDE0LjgxNDkxNSBaIiBpZD0iU2hhcGUiIGZpbGw9IiMwMDAwMDAiIGZpbGwtcnVsZT0ibm9uemVybyIgLz4KICAgICAgICAgICAgPHBhdGggZD0iTTEzLjk2MzgxODksOC42OTkzMzUgQzEzLjkzNjQ2MjEsOC43MDQzMDkyNSAxMy45MDkxMDU5LDguNzExNzY5NjggMTMuODg0MjM1OSw4LjcxOTIzMDc0IEMxMy43ODIyNzA0LDguNzM2NjM5NjcgMTMuNjg3NzY1NCw4Ljc3NjQzMTE1IDEzLjYwNTY5NTYsOC44Mzg2MDUxOCBMMTAuMjQzMzE1NiwxMS4zODUyNTk4IEMxMC4wNzY2ODg2LDExLjUwNDYzNDMgOS45NzcyMDk5MywxMS42OTg2MTgxIDkuOTc3MjA5OTMsMTEuOTAyNTQ5MSBDOS45NzcyMDk5MywxMi4xMDY0ODA3IDEwLjA3NjY4ODYsMTIuMzAwNDYzOSAxMC4yNDMzMTU2LDEyLjQxOTgzODMgTDEzLjYwNTY5NTYsMTQuOTY2NDkzIEMxMy44OTE2OTcsMTUuMTgwMzcyNSAxNC4yOTcwNzI5LDE1LjEyMzE3MjEgMTQuNTEwOTUxNywxNC44MzcxNzA3IEMxNC43MjQ4MzEzLDE0LjU1MTE2OTIgMTQuNjY3NjMwOSwxNC4xNDU3OTQgMTQuMzgxNjI5NCwxMy45MzE5MTQ1IEwxMi41MzEzMjU3LDEyLjUzOTIxMjcgTDIxLjg4MTI0OTUsMTIuNTM5MjEyNyBMMjEuODgxMjQ5NSwxMS4yNjU4ODU0IEwxMi41MzEzMjU3LDExLjI2NTg4NTQgTDE0LjM4MTYyOTQsOS44NzMxODM2NCBDMTQuNjM3Nzg3Miw5LjcxNjUwNDUzIDE0Ljc0OTcwMDYsOS40MDA2NjAxNCAxNC42NDc3MzUxLDkuMTE3MTQ1NTMgQzE0LjU0ODI1NjQsOC44MzM2MzE1NiAxNC4yNjIyNTUsOC42NTk1NDM1MiAxMy45NjM4MTg5LDguNjk5MzM1IFoiIGlkPSJhcnJvdyIgZmlsbD0iIzAwMDAwMCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTUuOTI5MjMwLCAxMS44OTQ3MzcpIHJvdGF0ZSgtMTgwLjAwMDAwMCkgdHJhbnNsYXRlKC0xNS45MjkyMzAsIC0xMS44OTQ3MzcpICIgLz4KICAgICAgICA8L2c+CiAgICA8L2c+Cjwvc3ZnPg==" /></td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3"><span style="color:#4CA64C;">✓</span></td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px">5</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">2<br />
0.40</td>
<td class="gt_row gt_right" style="height: 40px; color: black; font-family: IBM Plex Mono; font-size: 11px; border-left: 1px dashed #E5E5E5">3<br />
0.60</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-left: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC">--</td>
<td class="gt_row gt_center" style="height: 40px; background-color: #FCFCFC; border-right: 1px solid #D3D3D3">--</td>
<td class="gt_row gt_center" style="height: 40px">CSV</td>
</tr>
</tbody>
</table>


Three of the five aligned rows failed: one changed, one missing from the target, and one extra. The step report shows exactly what differs.


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

<table class="gt_table" style="table-layout: fixed;; width: 100%" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
<colgroup>
<col style="width: 30%" />
<col style="width: 70%" />
</colgroup>
<thead>
<tr class="gt_heading">
<th colspan="2" class="gt_heading gt_title gt_font_normal">Report for Validation Step 1 <span style="color: #CF142B;">✗</span>


ASSERTION <span style="border-style: solid; border-width: thin; border-color: lightblue; padding-left: 2px; padding-right: 2px;"><code style="color: #303030; background-color: transparent; position: relative; bottom: 1px;">tbl_match(keys='a')</code></span>

<strong>tbl_compare</strong> (4 rows) → <strong>target</strong> (4 rows), aligned by key <code>a</code><br />
<span style="background:#CF142B; color:white; padding:1px 6px; border-radius:3px; font-size:11px; font-weight:bold; letter-spacing:0.3px;">DIFFERENT</span> 3 of 5 rows fail (60.0%)<br />


<br />
<span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#4CA64C; margin-right:4px; margin-left:4px;"></span>match <strong>2</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#E8A33D; margin-right:4px; margin-left:4px;"></span>changed <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#CF142B; margin-right:4px; margin-left:4px;"></span>missing <strong>1</strong></span><span style="white-space:nowrap; margin-right:12px;"><span style="display:inline-block; width:9px; height:9px; border-radius:2px; background:#3D7CC9; margin-right:4px; margin-left:4px;"></span>extra <strong>1</strong></span>


</div></th>
</tr>
</thead>
<tbody class="gt_table_body">
<tr class="gt_group_heading_row">
<th colspan="2" class="gt_group_heading" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Column mismatches (1 of 2 columns)</th>
</tr>

<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">c</td>
<td class="gt_row gt_left" style="font-size: 12px"><div style="display:inline-block; vertical-align:middle; width:120px; height:7px; background:#EEE; border-radius:2px; margin-right:8px; margin-left:8px; overflow:hidden;">


1 of 3 matched rows differ (33.3%)</td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" class="gt_group_heading" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Changed rows (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">2</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:14px;"><span style="color:#666;">c:</span> <span style="text-decoration:line-through; color:#999;">5</span> → <span style="background:#FDF1DC; padding:0 3px; border-radius:2px;">5.5</span> <span style="color:#888; font-size:11px;">(+0.5)</span></span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" class="gt_group_heading" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Missing from target (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">3</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">b:</span> y</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">c:</span> 6</span></td>
</tr>
<tr class="gt_group_heading_row">
<td colspan="2" class="gt_group_heading" style="font-size: 12px; font-weight: bold; background-color: #F6F7F8">Extra in target (1 row)</td>
</tr>
<tr>
<td class="gt_row gt_left" style="font-family: IBM Plex Mono; font-size: 12px">5</td>
<td class="gt_row gt_left" style="font-size: 12px"><span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">b:</span> v</span> <span style="white-space:nowrap; margin-right:10px;"><span style="color:#666;">c:</span> 8</span></td>
</tr>
</tbody>
</table>
