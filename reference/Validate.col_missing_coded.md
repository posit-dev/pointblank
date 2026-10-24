# Validate.col_missing_coded()

Validate that all missing values in a column are *coded* (no uncoded nulls).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L12195-L12381)

``` python
Validate.col_missing_coded(
    columns,
    missing,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The [col_missing_coded()](../reference/Validate.col_missing_coded.md#pointblank.Validate.col_missing_coded) validation method checks that every absent value in a column is expressed with an explicit missing-value code, rather than a raw null. Under the structured missingness model (see [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec)), every absence should carry a *reason* -- encoded as a sentinel value such as `-99` for `"not_asked"`. A raw null represents *uncoded* (unknown) missingness, so this validation treats raw nulls as failing test units while declared sentinel values and real values pass.

This validation operates over the number of test units equal to the number of rows in the table (determined after any `pre=` mutation has been applied).

## Parameters

`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use [`col()`](../reference/col.md#pointblank.col) with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`missing: MissingSpec`  
A [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) describing the sentinel values (and their reasons) that encode missingness for this column. The spec documents which codes are considered valid expressions of missingness.

`pre: Callable | None = None`  
An optional preprocessing function or lambda to apply to the data table during interrogation. This function should take a table as input and return a modified table.

`segments: SegmentSpec | None = None`  
An optional directive on segmentation, which serves to split a validation step into multiple (one step per segment).

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`.

`actions: Actions | None = None`  
Optional actions to take when the validation step(s) meets or exceeds any set threshold levels. If provided, the [`Actions`](../reference/Actions.md#pointblank.Actions) class should be used to define the actions.

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

Here, the `age` column codes its missingness with sentinel values, except for one row that has a raw null (an uncoded absence):

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame({"age": [34, -98, 41, None, 29, -99, 55, 38]})

age_missing = pb.MissingSpec(
    reasons={-99: "not_asked", -98: "refused", -97: "dont_know"},
)

validation = (
    pb.Validate(data=tbl)
    .col_missing_coded(columns="age", missing=age_missing)
    .interrogate()
)

validation
```

[TABLE]

The validation reports a single failing test unit: the row where `age` is a raw null, which represents missingness without a documented reason.
