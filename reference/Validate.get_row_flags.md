# Validate.get_row_flags()

Get the target table with columns flagging the validation steps each row failed.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L22098-L22349)

``` python
Validate.get_row_flags(
    duplicates="keep_first",
    duplicates_order_by=None,
)
```

Where [`get_sundered_data()`](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) splits the table into the rows that passed every step and those that failed any of them, [get_row_flags()](../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags) keeps every row and says *which* steps each one failed. The returned table is the target table with these columns added:

- `pb_failed_<i>_`, one for each row-level step (where `<i>` is the step number): `True` if the row failed step `<i>`
- `pb_score_<i>_`, one for each [`col_vals_scored()`](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) step that keeps its scores (the default, `keep_scores=True`): the row's score in step `<i>`, missing if the row wasn't scored
- `pb_n_failed_`: the number of steps the row failed
- `pb_row_score_`: the share of the row's checks that passed (from `0` to `1`), weighted by the dimension weights set with `config(dimension_weights=...)` just as in [`get_health_score()`](../reference/Validate.get_health_score.md#pointblank.Validate.get_health_score)

This is useful when removing every row that failed a check is too blunt. You can use the flags to see which problems occur together, filter on just the checks you care about, stratify a sample on them, or use `pb_row_score_` to weight rows (e.g., as sample weights when training a model) instead of dropping them.

## Details

The steps that get flag columns are the same ones used by [get_sundered_data()](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data): steps that check rows or cells down a column, that are active, and that don't use `pre=` (unless the `pre=` function is marked with [`row_preserving()`](../reference/row_preserving.md#pointblank.row_preserving)). A row that a step didn't check (e.g., a row outside the segment of a step that uses `segments=`, or a row that a row-preserving `pre=` function filtered out) isn't flagged by it, and isn't counted in its `pb_row_score_`. A row that no step checked has a missing `pb_row_score_`.

Unlike [get_sundered_data()](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data), steps that use [`rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) are included by default: one copy of each group of duplicate rows (the first one) isn't flagged, and the other copies are. So a `rows_distinct()` flag marks the rows that removing duplicates would remove. Use `duplicates=` and `duplicates_order_by=` to choose which copy is kept.

A row-level step that can't be used is reported in a warning: that's a step that uses an unmarked `pre=` function (it can change the rows, so its results may not line up with the table's rows), a step that couldn't be evaluated, or a `rows_distinct()` step when `duplicates=None`.

## Parameters

`duplicates: str | None = "keep_first"`  
How to flag the duplicate rows found by `rows_distinct()` steps. With `"keep_first"` (the default) or `"keep_last"`, one copy of each group of duplicates (the first or last one, in row order or in the order given by `duplicates_order_by=`) isn't flagged. `"flag_all"` flags every copy, as the step itself counts them, and `None` leaves `rows_distinct()` steps out.

`duplicates_order_by: str | list[str] | None = None`  
One or more columns that order the copies of a duplicated row, to decide which copy isn't flagged. For example, use `duplicates="keep_last", duplicates_order_by="updated_at"` to keep the newest record. Ties are broken by row order, and a copy with a missing value in one of these columns is kept only if every copy is missing that value. Without this, the copies are taken in row order (for a database table, which has no row order of its own, rows are ordered by their values, so the choice is arbitrary but the same from run to run).

## Returns

`Any`  
The target table, of the same type, with the flag and summary columns added.

## Examples

Let's validate a table of orders with a check on the amounts, a check for missing values, and a check for duplicated order IDs.

``` python
import pointblank as pb
import polars as pl

orders = pl.DataFrame(
    {
        "order_id": [1, 2, 2, 3, 4],
        "amount": [25.0, -5.0, -5.0, None, 40.0],
        "status": ["paid", "paid", "paid", "open", None],
    }
)

validation = (
    pb.Validate(data=orders)
    .col_vals_gt(columns="amount", value=0)
    .rows_complete()
    .rows_distinct(columns_subset="order_id")
    .interrogate()
)

pb.preview(validation.get_row_flags())
```

|  | order_id *Int64* | amount *Float64* | status *String* | pb_failed_1\_ *Boolean* | pb_failed_2\_ *Boolean* | pb_failed_3\_ *Boolean* | pb_n_failed\_ *Int64* | pb_row_score\_ *Float64* |
|----|----|----|----|----|----|----|----|----|
| 1 | 1 | 25.0 | paid | False | False | False | 0 | 1.0 |
| 2 | 2 | -5.0 | paid | True | False | False | 1 | 0.6666666666666666 |
| 3 | 2 | -5.0 | paid | True | False | True | 2 | 0.3333333333333333 |
| 4 | 3 | None | open | True | True | False | 2 | 0.3333333333333333 |
| 5 | 4 | 40.0 | None | False | True | False | 1 | 0.6666666666666666 |

Each step has a flag column. Both copies of order `2` have a negative amount, so both fail step 1, but only the second copy is flagged by step 3, since removing duplicates would keep the first one. Order `3` fails step 1 because its amount is missing (missing values fail `col_vals_gt()` unless `na_pass=True`) and it also fails step 2. Order `1` passed every check, so it has a `pb_row_score_` of `1`.

## See Also

[get_sundered_data()](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data)  
Split the table into the rows that passed and failed the steps.

[get_health_score()](../reference/Validate.get_health_score.md#pointblank.Validate.get_health_score)  
Get a single score for the whole table.
