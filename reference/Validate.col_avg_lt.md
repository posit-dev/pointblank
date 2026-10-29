# Validate.col_avg_lt()

Does the column average satisfy a less than comparison?

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L1111-L1148)

``` python
Validate.col_avg_lt(
    columns,
    value=None,
    tol=0,
    thresholds=None,
    brief=False,
    actions=None,
    active=True,
    dimension=None,
)
```

The [col_avg_lt()](../reference/Validate.col_avg_lt.md#pointblank.Validate.col_avg_lt) validation method checks whether the average of values in a column is less than a specified `value=`. This is an aggregation-based validation where the entire column is reduced to a single average value that is then compared against the target. The comparison used in this function is `average(column) < value`.

Unlike row-level validations (e.g., `col_vals_gt()`), this method treats the entire column as a single test unit. The validation either passes completely (if the aggregated value satisfies the comparison) or fails completely.

## Parameters

`columns: _PBUnresolvedColumn`  
A single column or a list of columns to validate. If multiple columns are supplied, there will be a separate validation step generated for each column. The columns must contain numeric data for the average to be computed.

`value: float | Column | ReferenceColumn | None = None`  
The value to compare the column average against. This can be: (1) a numeric literal (`int` or `float`), (2) a [`col()`](../reference/col.md#pointblank.col) object referencing another column whose average will be used for comparison, (3) a [`ref()`](../reference/ref.md#pointblank.ref) object referencing a column in reference data (when `Validate(reference=)` has been set), or (4) `None` to automatically compare against the same column in reference data (shorthand for `ref(column_name)` when reference data is set).

`tol: Tolerance = 0`  
A tolerance value for the comparison. The default is `0`, meaning exact comparison. When set to a positive value, the comparison becomes more lenient. For example, with `tol=0.5`, a average that differs from the target by up to `0.5` will still pass. The `tol=` parameter expands the acceptable range for the comparison. For [col_avg_lt()](../reference/Validate.col_avg_lt.md#pointblank.Validate.col_avg_lt), a tolerance of `tol=0.5` would mean the average can be within `0.5` of the target value and still pass validation.

`thresholds: float | bool | tuple | dict | Thresholds | None = None`  
Failure threshold levels so that the validation step can react accordingly when failing test units are level. Since this is an aggregation-based validation with only one test unit, threshold values typically should be set as absolute counts (e.g., `1`) to indicate pass/fail, or as proportions where any value less than `1.0` means failure is acceptable.

`brief: str | bool = False`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like [`has_columns()`](../reference/has_columns.md#pointblank.has_columns) and [`has_rows()`](../reference/has_rows.md#pointblank.has_rows) can be used here to conditionally activate a step based on properties of the target table.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the step is assigned to `"validity"`, like the other aggregate checks. A check that's really about how much data there is (such as a sum of quantities against an expected total) might be better labeled `"volume"` or `"consistency"`. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Using Reference Data

The [col_avg_lt()](../reference/Validate.col_avg_lt.md#pointblank.Validate.col_avg_lt) method supports comparing column aggregations against reference data. This is useful for validating that statistical properties remain consistent across different versions of a dataset, or for comparing current data against historical baselines.

To use reference data, set the `reference=` parameter when creating the [Validate](../reference/Validate.md#pointblank.Validate) object:

``` python
validation = (
    pb.Validate(data=current_data, reference=baseline_data)
    .col_avg_lt(columns="revenue")  # Compares sum(current.revenue) vs sum(baseline.revenue)
    .interrogate()
)
```

When `value=None` and reference data is set, the method automatically compares against the same column in the reference data. You can also explicitly specify reference columns using the [ref()](../reference/ref.md#pointblank.ref) helper:

``` python
.col_avg_lt(columns="revenue", value=pb.ref("baseline_revenue"))
```

## Understanding Tolerance

The `tol=` parameter allows for fuzzy comparisons, which is especially important for floating-point aggregations where exact equality is often unreliable.

The `tol=` parameter expands the acceptable range for the comparison. For [col_avg_lt()](../reference/Validate.col_avg_lt.md#pointblank.Validate.col_avg_lt), a tolerance of `tol=0.5` would mean the average can be within `0.5` of the target value and still pass validation.

For equality comparisons (`col_*_eq`), the tolerance creates a range `[value - tol, value + tol]` within which the aggregation is considered valid. For inequality comparisons, the tolerance shifts the comparison boundary.

## Thresholds

The `thresholds=` parameter is used to set the failure-condition levels for the validation step. If they are set here at the step level, these thresholds will override any thresholds set at the global level in `Validate(thresholds=...)`.

There are three threshold levels: 'warning', 'error', and 'critical'. Since aggregation validations operate on a single test unit (the aggregated value), threshold values are typically set as absolute counts:

- `thresholds=1` means any failure triggers a 'warning'
- `thresholds=(1, 1, 1)` means any failure triggers all three levels

Thresholds can be defined using one of these input schemes:

1.  use the [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) class (the most direct way to create thresholds)
2.  provide a tuple of 1-3 values, where position `0` is the 'warning' level, position `1` is the 'error' level, and position `2` is the 'critical' level
3.  create a dictionary of 1-3 value entries; the valid keys: are 'warning', 'error', and 'critical'
4.  a single integer/float value denoting absolute number or fraction of failing test units for the 'warning' level only

## Examples

For the examples, we'll use a simple Polars DataFrame with numeric columns. The table is shown below:

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [1, 2, 3, 4, 5],
        "b": [2, 2, 2, 2, 2],
    }
)

pb.preview(tbl)
```

|     | a *Int64* | b *Int64* |
|-----|-----------|-----------|
| 1   | 1         | 2         |
| 2   | 2         | 2         |
| 3   | 3         | 2         |
| 4   | 4         | 2         |
| 5   | 5         | 2         |

Let's validate that the average of column `a` is less than `3`:

``` python
validation = (
    pb.Validate(data=tbl)
    .col_avg_lt(columns="a", value=3)
    .interrogate()
)

validation
```

[TABLE]

The validation result shows whether the average comparison passed or failed. Since this is an aggregation-based validation, there is exactly one test unit per column.

When validating multiple columns, each column gets its own validation step:

``` python
validation = (
    pb.Validate(data=tbl)
    .col_avg_lt(columns=["a", "b"], value=3)
    .interrogate()
)

validation
```

[TABLE]

Using tolerance for flexible comparisons:

``` python
validation = (
    pb.Validate(data=tbl)
    .col_avg_lt(columns="a", value=3, tol=1.0)
    .interrogate()
)

validation
```

[TABLE]
