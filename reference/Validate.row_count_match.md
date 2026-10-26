# Validate.row_count_match()

Validate whether the row count of the table matches a specified count.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L14143-L14373)

``` python
Validate.row_count_match(
    count,
    tol=0,
    inverse=False,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The [row_count_match()](../reference/Validate.row_count_match.md#pointblank.Validate.row_count_match) method checks whether the row count of the target table matches a specified count. This validation will operate over a single test unit, which is whether the row count matches the specified count.

We also have the option to invert the validation step by setting `inverse=True`. This will make the expectation that the row count of the target table *does not* match the specified count.

## Parameters

`count: int | Any`  
The expected row count of the table. This can be an integer value, a Polars or Pandas DataFrame object, or an Ibis backend table. If a DataFrame/table is provided, the row count of that object will be used as the expected count.

`tol: Tolerance = 0`  
The tolerance allowable for the row count match. This can be specified as a single numeric value (integer or float) or as a tuple of two integers representing the lower and upper bounds of the tolerance range. If a single integer value (greater than 1) is provided, it represents the absolute bounds of the tolerance, ie. plus or minus the value. If a float value (between 0-1) is provided, it represents the relative tolerance, ie. plus or minus the relative percentage of the target. If a tuple is provided, it represents the lower and upper absolute bounds of the tolerance range. See the examples for more.

`inverse: bool = False`  
Should the validation step be inverted? If `True`, then the expectation is that the row count of the target table should not match the specified `count=` value.

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

For the examples here, we'll use the built in dataset `"small_table"`. The table can be obtained by calling `load_dataset("small_table")`.

``` python
import pointblank as pb

small_table = pb.load_dataset("small_table")

pb.preview(small_table)
```

| PolarsRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | None | 2230.09 | True | high |

Let's validate that the number of rows in the table matches a fixed value. In this case, we will use the value `13` as the expected row count.

``` python
validation = (
    pb.Validate(data=small_table)
    .row_count_match(count=13)
    .interrogate()
)

validation
```

[TABLE]

The validation table shows that the expectation value of `13` matches the actual count of rows in the target table. So, the single test unit passed.

Let's modify our example to show the different ways we can allow some tolerance to our validation by using the `tol` argument.

``` python
smaller_small_table = small_table.sample(n = 12) # within the lower bound
validation = (
    pb.Validate(data=smaller_small_table)
    .row_count_match(count=13,tol=(2, 0)) # minus 2 but plus 0, ie. 11-13
    .interrogate()
)

validation

validation = (
    pb.Validate(data=smaller_small_table)
    .row_count_match(count=13,tol=.05) # .05% tolerance of 13
    .interrogate()
)

even_smaller_table = small_table.sample(n = 2)
validation = (
    pb.Validate(data=even_smaller_table)
    .row_count_match(count=13,tol=5) # plus or minus 5; this test will fail
    .interrogate()
)

validation
```

[TABLE]
