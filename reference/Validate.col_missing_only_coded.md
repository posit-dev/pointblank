# Validate.col_missing_only_coded()

Validate that a column contains only documented codes and legitimate values.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L12490-L12700)

``` python
Validate.col_missing_only_coded(
    columns,
    missing,
    allowed=None,
    min_val=None,
    max_val=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The [col_missing_only_coded()](../reference/Validate.col_missing_only_coded.md#pointblank.Validate.col_missing_only_coded) method checks that every value in a column is *accounted for*: it is either a declared missing-value code (a sentinel in the [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec), or a null when `null_is_missing=True`), or a legitimate "real" value. Legitimate real values are defined by `allowed=` (an explicit set) and/or a `[min_val, max_val]` range. Any value that is neither a documented code nor a legitimate real value is flagged -- this catches *undocumented* sentinel codes (e.g., a stray `-95`) that aren't part of the spec.

At least one of `allowed=`, `min_val=`, or `max_val=` must be provided so that legitimate real values can be distinguished from undocumented codes. This validation operates over the number of test units equal to the number of rows in the table.

## Parameters

`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use [`col()`](../reference/col.md#pointblank.col) with column selectors to specify one or more columns.

`missing: MissingSpec`  
A [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) declaring the documented sentinel codes.

`allowed: Collection[Any] | None = None`  
An explicit set of legitimate real values. A value in this set passes. Can be combined with `min_val=`/`max_val=` (a value passes if it satisfies either constraint).

`min_val: float | int | None = None`  
Lower bound (inclusive) of the legitimate real-value range.

`max_val: float | int | None = None`  
Upper bound (inclusive) of the legitimate real-value range.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment).

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`.

`actions: Actions | None = None`  
Optional actions to take when the validation step meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

`brief: str | bool | None = None`  
An optional brief description of the validation step that will be displayed in the reporting table. You can use the templating elements like `"{step}"` to insert the step number, or `"{auto}"` to include an automatically generated brief. If `True` the entire brief will be automatically generated. If `None` (the default) then there won't be a brief.

`active: bool | Callable = True`  
A boolean value or callable that determines whether the validation step should be active. Using `False` will make the validation step inactive (still reporting its presence and keeping indexes for the steps unchanged).

`dimension: str | None = None`  
An optional data quality dimension to categorize this validation step for health scoring. One of `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"` (or any custom string). If `None` (the default), the dimension is inferred automatically from the assertion type. This label appears in the validation report and feeds the overall and per-dimension health scores.

## Returns

`Validate`  
The [Validate](../reference/Validate.md#pointblank.Validate) object with the added validation step.

## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

## Segmentation

The `segments=` argument allows for the segmentation of a validation step into multiple segments. This is useful for applying the same validation step to different subsets of the data. The segmentation can be done based on a single column or specific fields within a column. Providing a single column name results in a separate validation step for each unique value in that column; a tuple of `(column, values)` restricts segmentation to the listed values. The segmentation is performed after any `pre=` preprocessing.

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

The `age` column should contain real ages in `[0, 120]` or the documented codes `-99`/`-98`. The value `-95` is an *undocumented* code and should be flagged:

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame({"age": [34, -98, 41, -95, 29, -99, 55]})

age_missing = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"})

validation = (
    pb.Validate(data=tbl)
    .col_missing_only_coded(columns="age", missing=age_missing, min_val=0, max_val=120)
    .interrogate()
)

validation
```

[TABLE]

The validation reports one failing test unit: the row where `age` is `-95`, which is neither a real age in range nor a declared sentinel.
