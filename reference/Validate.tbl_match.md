# Validate.tbl_match()

Validate whether the target table matches a comparison table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L15219-L15580)

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

The `tbl_match()` method compares the target table, row by row, against a comparison table that holds the expected data (the *source*). Rows of the two tables are aligned and every aligned row becomes a test unit: it passes if the row is present in both tables with equal values, and fails if its values changed, if it's missing from the target, if it's an extra row in the target, or if its key is duplicated. Schema differences (columns added, removed, retyped, or reordered) are checked too.

Rows are aligned in one of three ways:

- **by key** (`keys=` given): rows are matched on one or more key columns, so row order doesn't matter; this is the most informative mode
- **by position** (`keys=None`, the default): the first row of each table is compared, then the second, and so on
- **as multisets** (`keys="*"`): rows are compared as whole values regardless of order, with duplicates counted (for tables without a key)

The comparison runs inside the table's own engine (it is backend-agnostic, and database tables are compared in the database). Use [`get_step_report()`](../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report) to see a detailed report of the differences, and [`get_data_extracts()`](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) to get the failing rows. The same comparison is available outside of a validation with [`compare()`](../reference/compare.md#pointblank.compare).

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

`null_equal: bool = True`  
Should two null values count as equal? By default, they do. NaN values in floating-point columns are treated as null.

`schema: Literal["strict", "common"] = "strict"`  
How schema differences affect the result. With `"strict"` (the default), any schema difference (other than renames given in `column_map=`) means the tables don't match, and every test unit fails. With `"common"`, only the shared columns are compared and schema differences are reported without failing any test units.

`dup_keys: Literal["flag", "compare"] = "flag"`  
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
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like [`has_columns()`](../reference/has_columns.md#pointblank.has_columns) and [`has_rows()`](../reference/has_rows.md#pointblank.has_rows) can be used here to conditionally activate a step based on properties of the target table.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is inferred automatically from the assertion type. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Test Units

Each aligned row is a test unit. With `keys=`, that's every key value found in either table (a duplicated key counts as many units as its larger number of occurrences); by position, it's the number of rows in the longer table; as multisets, it's every row copy. A test unit passes only if its row is in both tables with all compared values equal. If the strict schema check fails, all test units fail. When both tables are empty, the step has a single test unit, which passes if the schemas match.

Because failures are counted per row, thresholds can be expressed as proportions (e.g., a [warning](../reference/Validate.warning.md#pointblank.Validate.warning) when more than 1% of rows differ). To fail on any difference, use an absolute threshold of `1`.

> **Important**
>
> In Pointblank `v1.0.1` and earlier, `tbl_match()` had a single test unit that failed if the tables differed in any way. Whether a step passes or fails is unchanged, but the counts (`n`, `n_passed`, `n_failed`, and the fractions) now reflect rows. This changes the meaning of fractional thresholds: `thresholds=0.1` used to be exceeded by any difference, and is now exceeded only when at least 10% of the rows fail. To keep the earlier behavior of reacting to any difference, use an absolute threshold of `1` (e.g., `thresholds=1` or `Thresholds(warning=1)`).

The data extract for the step (from [`get_data_extracts()`](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts)) contains the failing rows of *both* tables: the key column(s), a `_status_` column (`"changed"`, `"missing"`, `"extra"`, or `"dup_key"`), and a pair of `<column>_source`/`<column>_target` columns for each compared column. For [`get_sundered_data()`](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data), target rows with a matching row in the comparison table pass; rows missing from the target can't be part of the sundered data. Sundering for this step is available with in-memory tables (not lazy or database tables) and isn't available in multiset mode.

## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Note that the same preprocessing is **not** applied to the comparison table; only the target table is preprocessed. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](../reference/Validate.md#pointblank.Validate) object or used in subsequent validation steps.

## Thresholds

The `thresholds=` parameter is used to set the failure-condition levels for the validation step. If they are set here at the step level, these thresholds will override any thresholds set at the global level in `Validate(thresholds=...)`.

There are three threshold levels: 'warning', 'error', and 'critical'. The threshold values can either be set as a proportion failing of all test units (a value between `0` to `1`), or, the absolute number of failing test units (as integer that's `1` or greater).

Thresholds can be defined using one of these input schemes:

1.  use the [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) class (the most direct way to create thresholds)
2.  provide a tuple of 1-3 values, where position `0` is the 'warning' level, position `1` is the 'error' level, and position `2` is the 'critical' level
3.  create a dictionary of 1-3 value entries; the valid keys: are 'warning', 'error', and 'critical'
4.  a single integer/float value denoting absolute number or fraction of failing test units for the 'warning' level only

If the number of failing test units exceeds set thresholds, the validation step will be marked as 'warning', 'error', or 'critical'. All of the threshold levels don't need to be set, you're free to set any combination of them.

Aside from reporting failure conditions, thresholds can be used to determine the actions to take for each level of failure (using the `actions=` parameter).

## Cross-Backend Validation

The two tables can come from different backends (e.g., a Polars DataFrame and a Pandas DataFrame, or a database table and a CSV file). Tables on the same backend are compared in place (so two tables in the same database are compared inside the database). Tables on different backends are both converted to Polars (or to Pandas, if Polars isn't installed) before the comparison. Key columns whose types differ between the tables (e.g., integer keys that became floats in Pandas because of a missing value) are cast to a common type.

Values in columns whose types differ are compared numerically when both columns are numeric and as strings otherwise; note that such differences also fail the strict schema check (use `schema="common"` to only report them).

## Examples

For the examples here, we'll create two simple tables to demonstrate the `tbl_match()` validation.

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

| PolarsRows4Columns3 |           |            |             |
|---------------------|-----------|------------|-------------|
|                     | a *Int64* | b *String* | c *Float64* |
| 1                   | 1         | w          | 4.0         |
| 2                   | 2         | x          | 5.0         |
| 3                   | 3         | y          | 6.0         |
| 4                   | 4         | z          | 7.0         |

Let's validate that `tbl_1` matches `tbl_2`. Since these tables are identical, the validation should pass.

``` python
validation = (
    pb.Validate(data=tbl_1)
    .tbl_match(tbl_compare=tbl_2)
    .interrogate()
)

validation
```

[TABLE]

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

[TABLE]

Three of the five aligned rows failed: one changed, one missing from the target, and one extra. The step report shows exactly what differs.

``` python
validation.get_step_report(i=1)
```

[TABLE]
