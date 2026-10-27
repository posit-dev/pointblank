# Validate.col_vals_increasing()

Are column data increasing by row?

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L9608-L9837)

``` python
Validate.col_vals_increasing(
    columns,
    allow_stationary=False,
    decreasing_tol=None,
    na_pass=False,
    order_by=None,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The [col_vals_increasing()](../reference/Validate.col_vals_increasing.md#pointblank.Validate.col_vals_increasing) validation method checks whether column values in a table are increasing when moving down a table. There are options for allowing missing values in the target column, allowing stationary phases (where consecutive values don't change), and even one for allowing decreasing movements up to a certain threshold. This validation will operate over the number of test units that is equal to the number of rows in the table (determined after any `pre=` mutation has been applied).

## Parameters

`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use [`col()`](../reference/col.md#pointblank.col) with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`allow_stationary: bool = False`  
An option to allow pauses in increasing values. For example, if the values for the test units are `[80, 82, 82, 85, 88]` then the third unit (`82`, appearing a second time) would be marked as failing when `allow_stationary` is `False`. Using `allow_stationary=True` will result in all the test units in `[80, 82, 82, 85, 88]` to be marked as passing.

`decreasing_tol: float | None = None`  
An optional threshold value that allows for movement of numerical values in the negative direction. By default this is `None` but using a numerical value will set the absolute threshold of negative travel allowed across numerical test units. Note that setting a value here also has the effect of setting `allow_stationary` to `True`.

`na_pass: bool = False`  
Should any encountered None, NA, or Null values be considered as passing test units? By default, this is `False`. Set to `True` to pass test units with missing values.

`order_by: str | list[str] | None = None`  
The column (or list of columns) that puts the rows in order before the values are compared, in ascending order. If `None` (the default), the table's own row order is used. Tables without an inherent row order (e.g., database tables, Ibis tables, or PySpark DataFrames) need `order_by=`. See the *Row Order and Missing Values* section.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table. Have a look at the *Preprocessing* section for more information on how to use this argument.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment). Can be a single column name, a tuple that specifies a column name and its corresponding values to segment on, or a combination of both (provided as a list). Read the *Segmentation* section for usage information.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
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

## Row Order And Missing Values

Each value is compared with the value before it, so the rows need an order. Polars and pandas DataFrames (and Polars LazyFrames) have one, which is used unless `order_by=` is given. Database tables, Ibis tables, and PySpark DataFrames have no inherent row order, so `order_by=` is needed for them; without it, the step is marked as having an evaluation error. The results are reported against the table's own rows either way. If the `order_by=` columns have ties, the order of the tied rows (and so the result) may vary.

A missing value fails unless `na_pass=True`, and it doesn't break the sequence: the next non-missing value is compared with the last non-missing value before it. For example, with `[1, 2, None, 5]` the `5` is compared with the `2`. The first non-missing value has nothing to be compared with, so it always passes.

## Examples

For the examples here, we'll use a simple Polars DataFrame with a numeric column (`a`). The table is shown below:

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [1, 2, 3, 4, 5, 6],
        "b": [1, 2, 2, 3, 4, 5],
        "c": [1, 2, 1, 3, 4, 5],
    }
)

pb.preview(tbl)
```

|     | a *Int64* | b *Int64* | c *Int64* |
|-----|-----------|-----------|-----------|
| 1   | 1         | 1         | 1         |
| 2   | 2         | 2         | 2         |
| 3   | 3         | 2         | 1         |
| 4   | 4         | 3         | 3         |
| 5   | 5         | 4         | 4         |
| 6   | 6         | 5         | 5         |

Let's validate that values in column `a` are increasing. We'll determine if this validation had any failing test units (there are six test units, one for each row).

``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_increasing(columns="a")
    .interrogate()
)

validation
```

[TABLE]

The validation passed as all values in column `a` are increasing. Now let's check column `b` which has a stationary value:

``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_increasing(columns="b")
    .interrogate()
)

validation
```

[TABLE]

This validation fails at the third row because the value `2` is repeated. If we want to allow stationary values, we can use `allow_stationary=True`:

``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_increasing(columns="b", allow_stationary=True)
    .interrogate()
)

validation
```

[TABLE]
