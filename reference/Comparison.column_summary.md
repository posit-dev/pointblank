# Comparison.column_summary()

Get a per-column summary of value mismatches.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L2173-L2263)

``` python
Comparison.column_summary()
```

A row counts as `"changed"` if any compared column differs, so the row statuses don't say which columns are responsible. [column_summary()](../reference/Comparison.column_summary.md#pointblank.Comparison.column_summary) does. It's often the quickest way to find the cause of a difference. A single column with mismatches in every row suggests a change in how that column is computed or formatted, while mismatches spread thinly across many columns suggest changed rows.

The summary also lists each column's data type on both sides and the target column it was compared against, which shows the effect of `column_map=`. The counts are computed when the comparison is made, so this doesn't query the tables again.

## Returns

`Any`  
A table (Polars if available, else Pandas) with one row per compared column: the source and target column names and data types, the number of matched rows whose values differ (`n_mismatch`), and that number as a fraction of matched rows (`f_mismatch`). Matched rows are those with a status of `"match"` or `"changed"`. In multiset mode, there are no matched pairs, so the last two columns are null.

## Examples

Here, a rewritten job produces an orders table where the totals are rounded to whole units, and one order has a different status.

``` python
import pointblank as pb
import polars as pl

orders_old = pl.DataFrame(
    {
        "order_id": [1, 2, 3, 4],
        "status": ["shipped", "shipped", "pending", "shipped"],
        "total": [19.99, 45.50, 12.00, 80.25],
    }
)
orders_new = pl.DataFrame(
    {
        "order_id": [1, 2, 3, 4],
        "status": ["shipped", "shipped", "cancelled", "shipped"],
        "total": [20.0, 46.0, 12.0, 80.0],
    }
)

comparison = pb.compare(orders_old, orders_new, keys="order_id")

comparison.column_summary()
```

shape: (2, 6)

| column   | target_column | dtype_source | dtype_target | n_mismatch | f_mismatch |
|----------|---------------|--------------|--------------|------------|------------|
| str      | str           | str          | str          | i64        | f64        |
| "status" | "status"      | "String"     | "String"     | 1          | 0.25       |
| "total"  | "total"       | "Float64"    | "Float64"    | 3          | 0.75       |

`total` differs in three of the four orders, which points to a systematic change (the rounding), while `status` differs in a single order. All four rows are `"changed"`, but the summary separates the two causes. Once the rounding is accepted (for instance, with `tolerance={"total": 0.5}`), only the status change is left.
