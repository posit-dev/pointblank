# Validate.col_sd_gt()

Does the column standard deviation satisfy a greater than comparison?

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L1261-L1301)

``` python
Validate.col_sd_gt(
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

The [col_sd_gt()](../reference/Validate.col_sd_gt.md#pointblank.Validate.col_sd_gt) validation method checks whether the standard deviation of a column is greater than a target `value=`. The whole column is reduced to a single number and compared with the target (`standard deviation(column) > value`), so the step has exactly one test unit, which either passes or fails.

Aggregate checks catch problems that row-level checks can't see. Every value in a column can be within its allowed range while the column as a whole is wrong: a daily total that doesn't match the control figure from the source system, an average that has shifted after a change upstream, or a sensor whose readings have stopped varying (a standard deviation near zero). Use row-level methods like [`col_vals_between()`](../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between) to check individual values, and an aggregate method to check what they add up to. The target can be a fixed number, the same aggregate of another column, or the same aggregate of a column in reference data. The last option makes these methods a good fit for comparing a new load of a table with the last one that was known to be good.

Null values are left out of the computation, and the standard deviation is the sample standard deviation (dividing by `n - 1`). The column must be numeric, and a non-numeric column raises an error during interrogation. Unlike the row-level methods, the aggregate methods don't take `pre=` or `segments=`: the standard deviation is computed over the whole table, so filter the table before passing it to [Validate](../reference/Validate.md#pointblank.Validate) if you need the standard deviation of a subset.

## Parameters

`columns: _PBUnresolvedColumn`  
A single column or a list of columns to validate. If multiple columns are supplied, there will be a separate validation step generated for each column. The columns must contain numeric data for the standard deviation to be computed.

`value: float | Column | ReferenceColumn | None = None`  
The target to compare the column standard deviation against. This can be: (1) a number, (2) a [`col()`](../reference/col.md#pointblank.col) object naming another column, in which case the target is the standard deviation of that column in the same table, (3) a [`ref()`](../reference/ref.md#pointblank.ref) object naming a column in the reference data (set with `Validate(reference=)`), in which case the target is the standard deviation of that column there, or (4) `None` (the default), which is shorthand for [ref()](../reference/ref.md#pointblank.ref) of the same column. Using `None` without reference data raises a `ValueError`. [col()](../reference/col.md#pointblank.col) takes a single column name here, not a column selector.

`tol: Tolerance = 0`  
The tolerance around the target. The default is `0`, an exact comparison. A value below `1` is a fraction of the target's size (`tol=0.05` allows 5%), and a value of `1` or more is an absolute amount (`tol=2` allows `2` either way). A tuple of two values, `(lower, upper)`, sets the two sides separately, with each value read the same way. See the *Understanding Tolerance* section for how the tolerance applies to each comparison.

`thresholds: float | bool | tuple | dict | Thresholds | None = None`  
Failure threshold levels for the step. They override any thresholds set in `Validate(thresholds=)`. Because the step has a single test unit, every level that's set is reached when the step fails. See the *Thresholds* section for details.

`brief: str | bool = False`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. Inspection functions like [`has_columns()`](../reference/has_columns.md#pointblank.has_columns) and [`has_rows()`](../reference/has_rows.md#pointblank.has_rows) can be used here to conditionally activate a step based on properties of the target table.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the step is assigned to `"validity"`, like the other aggregate checks. A check that's really about how much data there is (such as a sum of quantities against an expected total) might be better labeled `"volume"` or `"consistency"`. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Understanding Tolerance

Exact comparisons are often too strict for aggregates. A sum of floating-point values can differ from the expected total in the last decimal places, and an average or standard deviation computed from real data is rarely a round number. The `tol=` parameter turns the target into a range:

- with `tol=0.05`, the allowance is 5% of the target (`0.05 * abs(target)`)
- with `tol=2`, the allowance is `2`
- with `tol=(0.1, 0)`, the allowance is 10% below the target and nothing above it

Because values below `1` are relative, an absolute allowance smaller than `1` can't be given directly. Express it as a fraction of the target instead (for a target of `50`, `tol=0.01` allows `0.5`). A relative tolerance has no effect when the target is `0`.

How the allowance is used depends on the comparison. For `eq`, the aggregate passes if it's anywhere from `target - lower` to `target + upper`. For `gt` and `ge`, it's compared with `target - lower`, so the tolerance lowers the bar. For `lt` and `le`, it's compared with `target + upper`, which raises the ceiling. In those four cases the other side of the tolerance isn't used. For [col_sd_gt()](../reference/Validate.col_sd_gt.md#pointblank.Validate.col_sd_gt), the step passes when `standard deviation(column) > target - lower`.

## Thresholds

The `thresholds=` parameter is used to set the failure-condition levels for the validation step. If they are set here at the step level, these thresholds will override any thresholds set at the global level in `Validate(thresholds=...)`.

An aggregate step has a single test unit, so the fraction of failing test units is either `0` or `1`. Any level that's set, whether as a count (such as `1`) or as a fraction (such as `0.1`), is reached when the step fails, so the levels can't be graded within one step. Set only the level that a failure should reach: `thresholds=pb.Thresholds(error=1)` makes a failure an 'error' without a 'warning', and `thresholds=(1, 1, 1)` makes it reach all three levels.

Thresholds can be defined using one of these input schemes:

1.  use the [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) class (the most direct way to create thresholds)
2.  provide a tuple of 1-3 values, where position `0` is the 'warning' level, position `1` is the 'error' level, and position `2` is the 'critical' level
3.  create a dictionary of 1-3 value entries; the valid keys: are 'warning', 'error', and 'critical'
4.  a single integer/float value denoting absolute number or fraction of failing test units for the 'warning' level only

## Examples

For the examples, we'll use a simple Polars DataFrame with two numeric columns. The table is shown below:

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

Let's validate that the standard deviation of column `a` is greater than `1`:

``` python
validation = (
    pb.Validate(data=tbl)
    .col_sd_gt(columns="a", value=1)
    .interrogate()
)

validation
```

[TABLE]

The standard deviation of `a` is `1.58114`, and `1.58114 > 1` holds, so the step's one test unit passes.

When validating multiple columns, each column gets its own validation step:

``` python
validation = (
    pb.Validate(data=tbl)
    .col_sd_gt(columns=["a", "b"], value=1)
    .interrogate()
)

validation
```

[TABLE]

Each column gets its own step: step 1 is for `a` and step 2 is for `b`. The standard deviation of `b` is `0`, so step 2 fails. The report doesn't show the computed standard deviation, but the step report does:

``` python
validation.get_step_report(i=2)
```

| Report for Validation Step 2 ✗ ASSERTION `SD(b) > 1` The aggregate value for column `b` does not satisfy the condition. |  |  |  |  |
|----|----|----|----|----|
| ACTUAL | EXPECTED | TOL | DIFFERENCE |  |
| 0 | 1 | -- | -1 | ✗ |

The step report shows the standard deviation computed for `b` (*ACTUAL*), the target (*EXPECTED*), the tolerance (*TOL*, none here), and *DIFFERENCE*, which is how far the computed value is from the boundary it had to meet. Here that's `-1`, with positive values above the boundary and negative values below it.

Next, a target that the standard deviation of `a` just misses, first with an exact comparison and then with a 5% tolerance:

``` python
validation = (
    pb.Validate(data=tbl)
    .col_sd_gt(columns="a", value=1.65)
    .col_sd_gt(columns="a", value=1.65, tol=0.05)
    .interrogate()
)

validation
```

[TABLE]

The standard deviation of `a` is `1.58114`, so `1.58114 > 1.65` doesn't hold, if only just, and step 1 fails. In step 2, `tol=0.05` allows 5% of the target (`0.0825`), so the standard deviation only has to be greater than `1.5675`, and the step passes.

The target can also be the same aggregate of another column, given with [`col()`](../reference/col.md#pointblank.col):

``` python
validation = (
    pb.Validate(data=tbl)
    .col_sd_gt(columns="a", value=pb.col("b"))
    .interrogate()
)

validation
```

[TABLE]

Here the target is the standard deviation of `b` (`0`) rather than a fixed number. The standard deviation of `a` is `1.58114`, so `1.58114 > 0` holds and the step passes. Both aggregates are computed from the same table when the validation is interrogated.

Finally, the target can come from reference data, such as an earlier load of the same table that passed its checks:

``` python
current = pl.DataFrame({"revenue": [120.0, 95.5, 143.2, 88.0]})
baseline = pl.DataFrame({"revenue": [110.0, 101.0, 150.0, 90.0]})

validation = (
    pb.Validate(data=current, reference=baseline)
    .col_sd_gt(columns="revenue")
    .col_sd_gt(columns="revenue", value=pb.ref("revenue"), tol=0.05)
    .interrogate()
)

validation
```

[TABLE]

With `reference=` set, leaving out `value=` compares against the same column in the reference table. Step 1 checks `25.0684 > 26.1454` (the standard deviation of `revenue` in `current`, then in `baseline`), so it fails. Step 2 names the target column with [`ref()`](../reference/ref.md#pointblank.ref), which you'd need if the column had a different name in the reference table, and allows a 5% tolerance, so it passes. Comparing each new load of a table with the last one known to be good like this catches drift that no single row would reveal.
