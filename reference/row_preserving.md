# row_preserving()

Mark a `pre=` function as one that keeps the rows of the table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/preprocessing.py#L40-L130)

``` python
row_preserving(fn)
```

Validation methods have a `pre=` argument for a function that changes the table before a step checks it. Such a function can do anything to the table (e.g., aggregate it, or join another table to it), so Pointblank can't match the rows that a step with `pre=` checked to the rows of the target table. Because of this, steps with `pre=` are left out of [`get_row_flags()`](../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags) and [`get_sundered_data()`](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data). Wrapping the function in [row_preserving()](../reference/row_preserving.md#pointblank.row_preserving) says that it only filters, reorders, or changes the values of rows, so that its step can be used there too.

For a marked function, Pointblank adds a column called `pb_index_` (the row number) to the table before calling the function, and uses it to match the rows the function returns to the rows of the target table. The function must keep this column (if it selects columns, include `pb_index_` among them) and must not change its values. Each row of its result must come from a different row of the table, so the function can't aggregate rows, join a table that repeats rows, or combine the table with itself. If its result is missing the `pb_index_` column, or has repeated or unknown values in it, interrogation raises an error naming the step. These checks can't tell a filter from an aggregation that keeps one of each group's rows, along with its `pb_index_` value (e.g., taking the first row of each group), so only mark functions that keep the rows. Rows that the function filters out weren't checked by the step, so they don't count as failing it.

## Parameters

`fn: Callable`  
The function to mark: a function or lambda taking a table and returning a table, as for `pre=`. A function with a parameter named `dfn` is given a Narwhals DataFrame, as with any `pre=` function.

## Returns

`Callable`  
The function, marked as keeping the rows of the table. It behaves just like `fn`.

## Examples

Let's validate the order amounts in a table, checking only the orders that were paid. The `pre=` function filters the table to those rows, and [row_preserving()](../reference/row_preserving.md#pointblank.row_preserving) marks it as one that keeps the table's rows (it only removes some of them).

``` python
import pointblank as pb
import polars as pl

orders = pl.DataFrame(
    {
        "order_id": [1, 2, 3, 4],
        "amount": [25.0, -5.0, -10.0, 40.0],
        "status": ["paid", "paid", "open", "paid"],
    }
)

validation = (
    pb.Validate(data=orders)
    .col_vals_gt(
        columns="amount",
        value=0,
        pre=pb.row_preserving(lambda df: df.filter(pl.col("status") == "paid")),
    )
    .interrogate()
)

pb.preview(validation.get_row_flags())
```

|  | order_id *Int64* | amount *Float64* | status *String* | pb_failed_1\_ *Boolean* | pb_n_failed\_ *Int64* | pb_row_score\_ *Float64* |
|----|----|----|----|----|----|----|
| 1 | 1 | 25.0 | paid | False | 0 | 1.0 |
| 2 | 2 | -5.0 | paid | True | 1 | 0.0 |
| 3 | 3 | -10.0 | open | False | 0 | None |
| 4 | 4 | 40.0 | paid | False | 0 | 1.0 |

The step checked the three paid orders and order `2` failed it. Order `3` also has a negative amount, but it wasn't checked (the `pre=` function filtered it out), so it isn't flagged. Without [row_preserving()](../reference/row_preserving.md#pointblank.row_preserving), the step would have no flag column at all.

## See Also

[get_row_flags()](../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags)  
Get which steps each row failed, as columns added to the table.

[get_sundered_data()](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data)  
Split the table into the rows that passed and failed the steps.
