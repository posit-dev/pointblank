# Validate.col_vals_str_len()

Validate whether the length of string values falls within specified bounds.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L11430-L11638)

``` python
Validate.col_vals_str_len(
    columns,
    min_val=None,
    max_val=None,
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

The `col_vals_str_len()` validation method checks whether the character length of string values in a column meets the specified minimum and/or maximum bounds. This validation will operate over the number of test units that is equal to the number of rows in the table (determined after any `pre=` mutation has been applied).

## Parameters

`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use [`col()`](../reference/col.md#pointblank.col) with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

`min_val: int | None = None`  
The minimum acceptable string length (inclusive). If `None`, no lower bound is applied. At least one of `min_val=` or `max_val=` must be provided.

`max_val: int | None = None`  
The maximum acceptable string length (inclusive). If `None`, no upper bound is applied. At least one of `min_val=` or `max_val=` must be provided.

`na_pass: bool = False`  
Should any encountered None, NA, or Null values be considered as passing test units? By default, this is `False`. Set to `True` to pass test units with missing values.

`missing: MissingSpec | None = None`  
An optional [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) declaring the codes that stand in for missing values in the column (such as `-99` for a survey question that was refused). Rows holding one of these codes are excluded from the check (they count as passing test units), so only the real values are validated, and a note in the validation report lists the codes. By default a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) also treats nulls as missing, so nulls pass whatever the value of `na_pass=`. Create it with `null_is_missing=False` to leave nulls to `na_pass=`.

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

## Raises

`ValueError`  
If neither `min_val=` nor `max_val=` is provided.

## Examples

For the examples here, we'll use a simple Polars DataFrame with a string column (`a`). The table is shown below:

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": ["short", "medium str", "a very long string value", "ok"],
    }
)

pb.preview(tbl)
```

|     | a *String*               |
|-----|--------------------------|
| 1   | short                    |
| 2   | medium str               |
| 3   | a very long string value |
| 4   | ok                       |

Let's validate that all values in column `a` have a string length between 2 and 10 characters.

``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_str_len(columns="a", min_val=2, max_val=10)
    .interrogate()
)

validation
```

[TABLE]

The validation table shows one failing test unit: `"a very long string value"` (24 chars) exceeds `max_val=10`. The value `"ok"` (2 chars) passes since the lower bound is inclusive.

We can also validate with only a minimum length:

``` python
validation = (
    pb.Validate(data=tbl)
    .col_vals_str_len(columns="a", min_val=3)
    .interrogate()
)

validation
```

[TABLE]

Without an upper bound, the long string now passes, but `"ok"` fails because its two characters fall short of `min_val=3`. So there's again one failing test unit out of four.
