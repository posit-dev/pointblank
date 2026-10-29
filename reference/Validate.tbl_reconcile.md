# Validate.tbl_reconcile()

Validate that aggregates of the target table reconcile with a comparison table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L15903-L16201)

``` python
Validate.tbl_reconcile(
    tbl_compare,
    keys=None,
    measures=None,
    agg="sum",
    tolerance=None,
    column_map=None,
    partitions=None,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The `tbl_reconcile()` method checks that the target table agrees with a comparison table (the *source*, or expected data) on aggregate values: totals, counts, averages, minimums, or maximums of *measure* columns, computed per group of *key* columns (or over the whole table). It's the check to reach for when the two tables don't have the same rows but must add up to the same numbers, for example a warehouse fact table against the source system it was loaded from, or a monthly summary against the detailed transactions.

Each key group is a test unit: it passes if the group is in both tables and all of its aggregates are equal (within the tolerance), and fails if an aggregate differs, if the group is missing from the target, or if it's an extra group in the target. To compare the rows themselves rather than aggregates, use [`tbl_match()`](../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match).

## Parameters

`tbl_compare: Any`  
The comparison table (the expected data) to reconcile against. This can be a DataFrame object (Polars or Pandas), an Ibis table object, a path to a CSV or Parquet file, or a callable that returns a table.

`keys: str | list[str] | None = None`  
The column(s) to group by. Aggregates are compared group by group. If `None` (the default), the aggregates are computed over the whole table and compared as a single group (a grand total).

`measures: str | list[str] | None = None`  
The column(s) to aggregate. Required unless `agg="count"`, which (without `measures=`) compares the number of rows in each group.

`agg: str | dict[str, str | list[str]] = "sum"`  
The aggregation(s): `"sum"` (the default), `"count"` (the number of non-null values), `"mean"`, `"min"`, or `"max"`. Use a dict to aggregate measures differently, mapping a measure to an aggregation or a list of aggregations (e.g., `{"amount": ["sum", "max"], "qty": "sum"}`).

`tolerance: Any = None`  
Tolerance for comparing the aggregates. A number is an absolute tolerance for all aggregates. A dict with `"atol"` and/or `"rtol"` keys applies to all aggregates, where values are equal if `|expected - actual| <= atol + rtol * |expected|`. A dict mapping measures to either of the above sets per-measure tolerances (applied to all of a measure's aggregates other than counts).

`column_map: dict[str, str] | None = None`  
A mapping of comparison-table column names to target column names, for keys or measures that are named differently in the target.

`partitions: int | None = None`  
Compare the aggregated tables in this many passes, each covering a disjoint subset of the key values, to bound memory use when there are very many groups. Requires `keys=`.

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

## Test Units

Each key group found in either table is a test unit (with `keys=None`, there's a single unit for the whole table). A unit passes if its aggregates are equal (within the tolerance) in both tables. Rows with null keys form a group of their own, which is matched with the null-key group of the other table.

The data extract for the step (from [`get_data_extracts()`](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts)) contains the failing groups: the key column(s), a `_status_` column (`"changed"`, `"missing"`, or `"extra"`), and a pair of `<measure>_<agg>_source`/`<measure>_<agg>_target` columns for each aggregate. The step report shows the full comparison of the aggregates. Since test units are groups rather than rows, this step doesn't take part in [`get_sundered_data()`](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data).

## Preprocessing

The `pre=` argument allows for a preprocessing function or lambda to be applied to the data table during interrogation. This function should take a table as input and return a modified table. This is useful for performing any necessary transformations or filtering on the data before the validation step is applied.

The preprocessing function can be any callable that takes a table as input and returns a modified table. For example, you could use a lambda function to filter the table based on certain criteria or to apply a transformation to the data. Note that the same preprocessing is **not** applied to the comparison table; only the target table is preprocessed. Regarding the lifetime of the transformed table, it only exists during the validation step and is not stored in the [Validate](../reference/Validate.md#pointblank.Validate) object or used in subsequent validation steps.

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

Let's reconcile a small table of loaded orders against the source system's orders. The loaded table names the amount column differently, has a different number of rows (one order was split into two rows), and is missing the orders for one region.

``` python
import pointblank as pb
import polars as pl

source_orders = pl.DataFrame(
    {
        "region": ["north", "north", "south", "east", "west"],
        "amount": [100.0, 250.0, 75.5, 30.0, 12.0],
    }
)

loaded_orders = pl.DataFrame(
    {
        "region": ["north", "north", "north", "south", "east"],
        "amt": [100.0, 200.0, 50.0, 75.5, 31.0],
    }
)

validation = (
    pb.Validate(data=loaded_orders)
    .tbl_reconcile(
        tbl_compare=source_orders,
        keys="region",
        measures="amount",
        column_map={"amount": "amt"},
    )
    .interrogate()
)

validation
```

[TABLE]

Of the four regions, `north` and `south` reconcile; `east` has a different total, and `west` is missing from the loaded orders. The step report shows the details:

``` python
validation.get_step_report(i=1)
```

[TABLE]

Grand totals don't need any keys. Here, the total amount matches within a tolerance of `20`, and so the step passes:

``` python
(
    pb.Validate(data=loaded_orders)
    .tbl_reconcile(
        tbl_compare=source_orders,
        measures="amount",
        column_map={"amount": "amt"},
        tolerance=20,
    )
    .interrogate()
    .all_passed()
)
```

    True

The method chain ends with `all_passed()`, which returns `True`. The totals are `467.5` in the source and `456.5` in the loaded table, a difference of `11` that falls within the tolerance. A grand total can hide offsetting differences, though: the missing `west` orders and the larger `east` total partly cancel out, which is why the per-region check above is the stricter one.
