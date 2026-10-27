# Validate.col_missing_consistent()

Validate that related columns share a consistent missingness pattern for a given reason.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L13222-L13429)

``` python
Validate.col_missing_consistent(
    columns,
    missing,
    when_reason,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The [col_missing_consistent()](../reference/Validate.col_missing_consistent.md#pointblank.Validate.col_missing_consistent) method checks that, across a set of related columns, the "missing for a specific reason" status is *consistent*: for each row, either *none* of the columns are missing for `when_reason=`, or *all* of them are. This is useful for structured survey or clinical data where a skip pattern should propagate across related fields -- for example, if a question wasn't asked (`"not_asked"`) then all of its dependent fields should also be coded `"not_asked"`.

A value is considered "missing for the reason" when it is one of the sentinel values mapped to `when_reason=` in the [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) (and, when the reason is the spec's `null_reason` and `null_is_missing=True`, an actual null). This validation operates over the number of test units equal to the number of rows in the table. A row fails when some -- but not all -- of the columns are missing for the given reason.

## Parameters

`columns: list[str]`  
A list of related columns to check for consistent missingness.

`missing: MissingSpec`  
A [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) describing the sentinel values and their reasons for the columns.

`when_reason: str`  
The reason label whose presence should be consistent across `columns=`. If one column in a row is missing for this reason, all of them should be.

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

Here, `income_source` and `income_amount` should both be coded `"not_asked"` (`-99`) together when the income question wasn't asked. The last row is inconsistent -- only one field is coded `-99`:

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "income_source": [1, -99, 2, -99],
        "income_amount": [50000, -99, 42000, 38000],
    }
)

income_missing = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"})

validation = (
    pb.Validate(data=tbl)
    .col_missing_consistent(
        columns=["income_source", "income_amount"],
        missing=income_missing,
        when_reason="not_asked",
    )
    .interrogate()
)

validation
```

[TABLE]

The validation reports one failing test unit: the final row, where `income_source` is coded `-99` (`"not_asked"`) but `income_amount` is a real value.
