# Validate.col_exists()

Validate whether one or more columns exist in the table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L11614-L11799)

``` python
Validate.col_exists(
    columns,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The [col_exists()](../reference/Validate.col_exists.md#pointblank.Validate.col_exists) method checks whether one or more columns exist in the target table. The only requirement is specification of the column names. Each validation step or expectation will operate over a single test unit, which is whether the column exists or not.

## Parameters

`columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals`  
A single column or a list of columns to validate. Can also use [`col()`](../reference/col.md#pointblank.col) with column selectors to specify one or more columns. If multiple columns are supplied or resolved, there will be a separate validation step generated for each column.

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

For the examples here, we'll use a simple Polars DataFrame with a string columns (`a`) and a numeric column (`b`). The table is shown below:

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": ["apple", "banana", "cherry", "date"],
        "b": [1, 6, 3, 5],
    }
)

pb.preview(tbl)
```

|     | a *String* | b *Int64* |
|-----|------------|-----------|
| 1   | apple      | 1         |
| 2   | banana     | 6         |
| 3   | cherry     | 3         |
| 4   | date       | 5         |

Let's validate that the columns `a` and `b` actually exist in the table. We'll determine if this validation had any failing test units (each validation will have a single test unit).

``` python
validation = (
    pb.Validate(data=tbl)
    .col_exists(columns=["a", "b"])
    .interrogate()
)

validation
```

[TABLE]

Printing the `validation` object shows the validation table in an HTML viewing environment. The validation table shows two entries (one check per column) generated by the [col_exists()](../reference/Validate.col_exists.md#pointblank.Validate.col_exists) validation step. Both steps passed since both columns provided in `columns=` are present in the table.

Now, let's check for the existence of a different set of columns.

``` python
validation = (
    pb.Validate(data=tbl)
    .col_exists(columns=["b", "c"])
    .interrogate()
)

validation
```

[TABLE]

The validation table reports one passing validation step (the check for column `b`) and one failing validation step (the check for column `c`, which doesn't exist).
