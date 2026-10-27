# Validate.conjointly()

Perform multiple row-wise validations for joint validity.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L15788-L16057)

``` python
Validate.conjointly(
    *exprs,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The [conjointly()](../reference/Validate.conjointly.md#pointblank.Validate.conjointly) validation method checks whether each row in the table passes multiple validation conditions simultaneously. This enables compound validation logic where a test unit (typically a row) must satisfy all specified conditions to pass the validation.

This method accepts multiple validation expressions as callables, which should return boolean expressions when applied to the data. You can use lambdas that incorporate Polars/Pandas/Ibis expressions (based on the target table type) or create more complex validation functions. The validation will operate over the number of test units that is equal to the number of rows in the table (determined after any `pre=` mutation has been applied).

## Parameters

`*exprs: Callable`  
Multiple validation expressions provided as callable functions. Each callable should accept a table as its single argument and return a boolean expression or Series/Column that evaluates to boolean values for each row.

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

## Missing Values And Evaluation Errors

A row where any expression evaluates to a missing value (e.g., a comparison on a null) counts as a failing test unit. To let such rows pass, allow for them in the expression itself (e.g., `lambda df: (pl.col("a") > 1) | pl.col("a").is_null()`).

If any of the expressions can't be evaluated against the table (e.g., one refers to a column that doesn't exist), the step is marked as having an evaluation error and a note says which expression failed and why. The other validation steps are still interrogated.

## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](../reference/Validate.md#pointblank.Validate) object or used in subsequent validation steps.

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

For the examples here, we'll use a simple Polars DataFrame with three numeric columns (`a`, `b`, and `c`). The table is shown below:

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [5, 7, 1, 3, 9, 4],
        "b": [6, 3, 0, 5, 8, 2],
        "c": [10, 4, 8, 9, 10, 5],
    }
)

pb.preview(tbl)
```

|     | a *Int64* | b *Int64* | c *Int64* |
|-----|-----------|-----------|-----------|
| 1   | 5         | 6         | 10        |
| 2   | 7         | 3         | 4         |
| 3   | 1         | 0         | 8         |
| 4   | 3         | 5         | 9         |
| 5   | 9         | 8         | 10        |
| 6   | 4         | 2         | 5         |

Let's validate that the values in each row satisfy multiple conditions simultaneously:

1.  Column `a` should be greater than 2
2.  Column `b` should be less than 7
3.  The sum of `a` and `b` should be less than the value in column `c`

We'll use [conjointly()](../reference/Validate.conjointly.md#pointblank.Validate.conjointly) to check all these conditions together:

``` python
validation = (
    pb.Validate(data=tbl)
    .conjointly(
        lambda df: pl.col("a") > 2,
        lambda df: pl.col("b") < 7,
        lambda df: pl.col("a") + pl.col("b") < pl.col("c")
    )
    .interrogate()
)

validation
```

[TABLE]

The validation table shows that not all rows satisfy all three conditions together. For a row to pass the conjoint validation, all three conditions must be true for that row.

We can also use preprocessing to filter the data before applying the conjoint validation:

``` python
# Define preprocessing function for serialization compatibility
def filter_by_c_gt_5(df):
    return df.filter(pl.col("c") > 5)

validation = (
    pb.Validate(data=tbl)
    .conjointly(
        lambda df: pl.col("a") > 2,
        lambda df: pl.col("b") < 7,
        lambda df: pl.col("a") + pl.col("b") < pl.col("c"),
        pre=filter_by_c_gt_5
    )
    .interrogate()
)

validation
```

[TABLE]

This allows for more complex validation scenarios where the data is first prepared and then validated against multiple conditions simultaneously.

Or, you can use the backend-agnostic column expression helper [`expr_col()`](../reference/expr_col.md#pointblank.expr_col) to write expressions that work across different table backends:

``` python
tbl = pl.DataFrame(
    {
        "a": [5, 7, 1, 3, 9, 4],
        "b": [6, 3, 0, 5, 8, 2],
        "c": [10, 4, 8, 9, 10, 5],
    }
)

# Using backend-agnostic syntax with expr_col()
validation = (
    pb.Validate(data=tbl)
    .conjointly(
        lambda df: pb.expr_col("a") > 2,
        lambda df: pb.expr_col("b") < 7,
        lambda df: pb.expr_col("a") + pb.expr_col("b") < pb.expr_col("c")
    )
    .interrogate()
)

validation
```

[TABLE]

Using [`expr_col()`](../reference/expr_col.md#pointblank.expr_col) allows your validation code to work consistently across Pandas, Polars, and Ibis table backends without changes, making your validation pipelines more portable.

## See Also

`Look` at the documentation of the [`expr_col()`](../reference/expr_col.md#pointblank.expr_col) function for more  

`information` on how to use it with different table backends.  
