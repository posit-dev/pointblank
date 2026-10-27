# Validate.col_pct_missing()

Validate that the percentage of *structured* missing values stays within a limit.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L12106-L12312)

``` python
Validate.col_pct_missing(
    columns,
    missing,
    max_pct,
    reason=None,
    category=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The [col_pct_missing()](../reference/Validate.col_pct_missing.md#pointblank.Validate.col_pct_missing) validation method checks whether the percentage of missing values in a column is at most `max_pct=`. Unlike [`col_pct_null()`](../reference/Validate.col_pct_null.md#pointblank.Validate.col_pct_null), which only considers actual null values, this method uses a [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) to define which values count as missing: declared sentinel values (e.g., `-99` for `"refused"`) and, when `null_is_missing=True`, actual null values. This validation operates at the column level, generating a single validation step per column that passes when the missing percentage does not exceed `max_pct=`.

You can narrow the check to a single reason (via `reason=`) or a category of reasons (via `category=`), making it possible to assert things like "at most 10% of values were refused" or "at most 15% are item nonresponse".

## Parameters

`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use [`col()`](../reference/col.md#pointblank.col) with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`missing: MissingSpec`  
A [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) describing the sentinel values (and their reasons) that encode missingness for this column.

`max_pct: float`  
The maximum allowable percentage of missing values, expressed as a decimal between `0.0` and `1.0`. For example, `max_pct=0.20` means at most 20% of values may be missing.

`reason: str | None = None`  
If provided, only count missing values whose reason matches this label. Cannot be combined with `category=`.

`category: str | None = None`  
If provided, only count missing values whose reason falls in this category (as defined in `MissingSpec.categories`). Cannot be combined with `reason=`.

`thresholds: int | float | None | bool | tuple | dict | Thresholds = None`  
Set threshold failure levels for reporting and reacting to exceedences of the levels. The thresholds are set at the step level and will override any global thresholds set in `Validate(thresholds=...)`. The default is `None`, which means that no thresholds will be set locally and global thresholds (if any) will take effect.

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

Survey data often encodes missingness with sentinel values rather than nulls. Here, the `age` column uses `-99` (`"not_asked"`), `-98` (`"refused"`), and `-97` (`"dont_know"`):

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {"age": [34, -98, 41, -99, 29, -98, 55, 38]},
)

age_missing = pb.MissingSpec(
    reasons={-99: "not_asked", -98: "refused", -97: "dont_know"},
    categories={"item_nonresponse": ["refused", "dont_know"]},
)

validation = (
    pb.Validate(data=tbl)
    .col_pct_missing(columns="age", missing=age_missing, max_pct=0.5)
    .col_pct_missing(columns="age", missing=age_missing, reason="refused", max_pct=0.30)
    .interrogate()
)

validation
```

[TABLE]
