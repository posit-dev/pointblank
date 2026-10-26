# Validate.col_vals_lt()

Are column data less than a fixed value or data in another column?

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L6715-L7019)

``` python
Validate.col_vals_lt(
    columns,
    value,
    na_pass=False,
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

The [col_vals_lt()](../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt) validation method checks whether column values in a table are *less than* a specified `value=` (the exact comparison used in this function is `col_val < value`). The `value=` can be specified as a single, literal value or as a column name given in [`col()`](../reference/col.md#pointblank.col). This validation will operate over the number of test units that is equal to the number of rows in the table (determined after any `pre=` mutation has been applied).

## Parameters

`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use [`col()`](../reference/col.md#pointblank.col) with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`value: float | int | Column`  
The value to compare against. This can be a single value or a single column name given in [`col()`](../reference/col.md#pointblank.col). The latter option allows for a column-to-column comparison. For more information on which types of values are allowed, see the *What Can Be Used in `value=`?* section.

`na_pass: bool = False`  
Should any encountered None, NA, or Null values be considered as passing test units? By default, this is `False`. Set to `True` to pass test units with missing values.

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

## What Can Be Used In `value=`?

The `value=` argument allows for a variety of input types. The most common are:

- a single numeric value
- a single date or datetime value
- A [`col()`](../reference/col.md#pointblank.col) object that represents a column name

When supplying a number as the basis of comparison, keep in mind that all resolved columns must also be numeric. Should you have columns that are of the date or datetime types, you can supply a date or datetime value as the `value=` argument. There is flexibility in how you provide the date or datetime value, as it can be:

- a string-based date or datetime (e.g., `"2023-10-01"`, `"2023-10-01 13:45:30"`, etc.)
- a date or datetime object using the `datetime` module (e.g., `datetime.date(2023, 10, 1)`, `datetime.datetime(2023, 10, 1, 13, 45, 30)`, etc.)

Finally, when supplying a column name in the `value=` argument, it must be specified within [`col()`](../reference/col.md#pointblank.col). This is a column-to-column comparison and, crucially, the columns being compared must be of the same type (e.g., both numeric, both date, etc.).

## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Note that you can refer to columns via `columns=` and `value=col(...)` that are expected to be present in the transformed table, but may not exist in the table before preprocessing. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](../reference/Validate.md#pointblank.Validate) object or used in subsequent validation steps.

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
        "a": [5, 6, 5, 9, 7, 5],
        "b": [1, 2, 1, 2, 2, 2],
        "c": [2, 1, 1, 4, 3, 4],
    }
)

pb.preview(tbl)
```

|     | a *Int64* | b *Int64* | c *Int64* |
|-----|-----------|-----------|-----------|
| 1   | 5         | 1         | 2         |
| 2   | 6         | 2         | 1         |
| 3   | 5         | 1         | 1         |
| 4   | 9         | 2         | 4         |
| 5   | 7         | 2         | 3         |
| 6   | 5         | 2         | 4         |

Let's validate that values in column `a` are all less than the value of `10`. We'll determine if this validation had any failing test units (there are six test units, one for each row).

``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_lt(columns="a", value=10)
    .interrogate()
)

validation
```

[TABLE]

Printing the `validation` object shows the validation table in an HTML viewing environment. The validation table shows the single entry that corresponds to the validation step created by using [col_vals_lt()](../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt). All test units passed, and there are no failing test units.

Aside from checking a column against a literal value, we can also use a column name in the `value=` argument (with the helper function [`col()`](../reference/col.md#pointblank.col) to perform a column-to-column comparison. For the next example, we'll use [col_vals_lt()](../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt) to check whether the values in column `b` are less than values in column `c`.

``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_lt(columns="b", value=pb.col("c"))
    .interrogate()
)

validation
```

[TABLE]

The validation table reports two failing test units. The specific failing cases are:

- Row 1: `b` is `2` and `c` is `1`.
- Row 2: `b` is `1` and `c` is `1`.
