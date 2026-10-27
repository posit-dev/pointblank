# Validate.col_pct_null()

Validate whether a column has a specific percentage of Null values.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L11797-L12104)

``` python
Validate.col_pct_null(
    columns,
    p,
    tol=0,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The [col_pct_null()](../reference/Validate.col_pct_null.md#pointblank.Validate.col_pct_null) validation method checks whether the percentage of Null values in a column matches a specified percentage `p=` (within an optional tolerance `tol=`). This validation operates at the column level, generating a single validation step per column that passes or fails based on whether the actual percentage of Null values falls within the acceptable range defined by `p ± tol`.

## Parameters

`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use [`col()`](../reference/col.md#pointblank.col) with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`p: float`  
The expected percentage of Null values in the column, expressed as a decimal between `0.0` and `1.0`. For example, `p=0.5` means 50% of values should be Null.

`tol: Tolerance = 0`  
The tolerance allowed when comparing the actual percentage of Null values to the expected percentage `p=`. The validation passes if the actual percentage falls within the range `[p - tol, p + tol]`. Default is `0`, meaning an exact match is required. See the *Tolerance* section for details on all supported formats (absolute, relative, symmetric, and asymmetric bounds).

`thresholds: int | float | None | bool | tuple | dict | Thresholds = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect. Look at the *Thresholds* section for information on how to set threshold levels.

`actions: Actions | None = None`  
Optional actions to take when the validation step(s) meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged). A callable can also be provided; it will receive the data table as its single argument and must return a boolean value. The callable is evaluated *before* any `pre=` processing. Inspection functions like [`has_columns()`](../reference/has_columns.md#pointblank.has_columns) and [`has_rows()`](../reference/has_rows.md#pointblank.has_rows) can be used here to conditionally activate a step based on properties of the target table.

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is inferred automatically from the assertion type. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Tolerance

The `tol=` parameter accepts several different formats to specify the acceptable deviation from the expected percentage `p=`. The tolerance can be expressed as:

1.  *single integer* (absolute tolerance): the exact number of test units that can deviate. For example, `tol=2` means the actual count can differ from the expected count by up to 2 units in either direction.

2.  *single float between 0 and 1* (relative tolerance): a proportion of the expected count. For example, if the expected count is 50 and `tol=0.1`, the acceptable range is 45 to 55 (50 ± 10% of 50 = 50 ± 5).

3.  *tuple of two integers* (absolute bounds): explicitly specify the lower and upper bounds as absolute deviations. For example, `tol=(1, 3)` means the actual count can be 1 unit below or 3 units above the expected count.

4.  *tuple of two floats between 0 and 1* (relative bounds): explicitly specify the lower and upper bounds as proportional deviations. For example, `tol=(0.05, 0.15)` means the lower bound is 5% below and the upper bound is 15% above the expected count.

When using a single value (integer or float), the tolerance is applied symmetrically in both directions. When using a tuple, you can specify asymmetric tolerances where the lower and upper bounds differ.

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

## Examples

For the examples here, we'll use a simple Polars DataFrame with three columns (`a`, `b`, and `c`) that have different percentages of Null values. The table is shown below:

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [1, 2, 3, 4, 5, 6, 7, 8],
        "b": [1, None, 3, None, 5, None, 7, None],
        "c": [None, None, None, None, None, None, 1, 2],
    }
)

pb.preview(tbl)
```

|     | a *Int64* | b *Int64* | c *Int64* |
|-----|-----------|-----------|-----------|
| 1   | 1         | 1         | None      |
| 2   | 2         | None      | None      |
| 3   | 3         | 3         | None      |
| 4   | 4         | None      | None      |
| 5   | 5         | 5         | None      |
| 6   | 6         | None      | None      |
| 7   | 7         | 7         | 1         |
| 8   | 8         | None      | 2         |

Let's validate that column `a` has 0% Null values (i.e., no Null values at all).

``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="a", p=0.0)
    .interrogate()
)

validation
```

[TABLE]

Printing the `validation` object shows the validation table in an HTML viewing environment. The validation table shows the single entry that corresponds to the validation step created by using [col_pct_null()](../reference/Validate.col_pct_null.md#pointblank.Validate.col_pct_null). The validation passed since column `a` has no Null values.

Now, let's check that column `b` has exactly 50% Null values.

``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="b", p=0.5)
    .interrogate()
)

validation
```

[TABLE]

This validation also passes, as column `b` has exactly 4 out of 8 values as Null (50%).

Finally, let's validate column `c` with a tolerance. Column `c` has 75% Null values, so we'll check if it's approximately 70% Null with a tolerance of 10%.

``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="c", p=0.70, tol=0.10)
    .interrogate()
)

validation
```

[TABLE]

This validation passes because the actual percentage (75%) falls within the acceptable range of 60% to 80% (70% ± 10%).

The `tol=` parameter supports multiple formats to express tolerance. Let's explore all the different ways to specify tolerance using column `b`, which has exactly 50% Null values (4 out of 8 values).

*Using an absolute tolerance (integer)*: Specify the exact number of rows that can deviate. With `tol=1`, we allow the count to differ by 1 row in either direction.

``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="b", p=0.375, tol=1)  # Expect 3 nulls, allow ±1 (range: 2-4)
    .interrogate()
)

validation
```

[TABLE]

This passes because column `b` has 4 Null values, which falls within the acceptable range of 2 to 4 (3 ± 1).

*Using a relative tolerance (float)*: Specify the tolerance as a proportion of the expected count. With `tol=0.25`, we allow a 25% deviation from the expected count.

``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="b", p=0.375, tol=0.25)  # Expect 3 nulls, allow ±25% (range: 2.25-3.75)
    .interrogate()
)

validation
```

[TABLE]

This passes because 4 Null values falls within the acceptable range (3 ± 0.75 calculates to 2.25 to 3.75, which rounds down to 2 to 3 rows).

*Using asymmetric absolute bounds (tuple of integers)*: Specify different lower and upper bounds as absolute values. With `tol=(0, 2)`, we allow no deviation below but up to 2 rows above the expected count.

``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="b", p=0.25, tol=(0, 2))  # Expect 2 Nulls, allow +0/-2 (range: 2-4)
    .interrogate()
)

validation
```

[TABLE]

This passes because 4 Null values falls within the acceptable range of 2 to 4.

*Using asymmetric relative bounds (tuple of floats)*: Specify different lower and upper bounds as proportions. With `tol=(0.1, 0.3)`, we allow 10% below and 30% above the expected count.

``` python
validation = (
    pb.Validate(data=tbl)
    .col_pct_null(columns="b", p=0.375, tol=(0.1, 0.3))  # Expect 3 Nulls, allow -10%/+30%
    .interrogate()
)

validation
```

[TABLE]

This passes because 4 Null values falls within the acceptable range (3 - 0.3 to 3 + 0.9 calculates to 2.7 to 3.9, which rounds down to 2 to 3 rows).
