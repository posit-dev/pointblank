from __future__ import annotations

import functools
from typing import Any, Callable

import narwhals as nw

__all__ = [
    "row_preserving",
]

# The column that numbers the rows of the target table for a `row_preserving()` function, so that
# the rows it returns can be matched to the rows of the target table
_ROW_INDEX = "pb_index_"


class _RowPreservingFunction:
    """
    A `pre=` function marked as keeping the table's rows (see `row_preserving()`).

    It's a callable object rather than a closure so that it can be pickled when the function it
    wraps can be. It looks like the wrapped function to `inspect` (`__wrapped__`, `__name__`,
    etc.), so that its signature (a `dfn` parameter) and its source code are those of the function.
    """

    def __init__(self, fn: Callable) -> None:
        functools.update_wrapper(self, fn)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.__wrapped__(*args, **kwargs)

    def __repr__(self) -> str:
        return f"row_preserving({self.__wrapped__!r})"


class _RowPreservingError(ValueError):
    """Raised when a `row_preserving()` function didn't keep the table's rows."""


def row_preserving(fn: Callable) -> Callable:
    """
    Mark a `pre=` function as one that keeps the rows of the table.

    Validation methods have a `pre=` argument for a function that changes the table before a
    step checks it. Such a function can do anything to the table (e.g., aggregate it, or join
    another table to it), so Pointblank can't match the rows that a step with `pre=` checked to the
    rows of the target table. Because of this, steps with `pre=` are left out of
    [`get_row_flags()`](`pointblank.Validate.get_row_flags`) and
    [`get_sundered_data()`](`pointblank.Validate.get_sundered_data`). Wrapping the function in
    `row_preserving()` says that it only filters, reorders, or changes the values of rows, so that
    its step can be used there too.

    For a marked function, Pointblank adds a column called `pb_index_` (the row number) to the
    table before calling the function, and uses it to match the rows the function returns to the
    rows of the target table. The function must keep this column (if it selects columns, include
    `pb_index_` among them) and must not change its values. Each row of its result must come from
    a different row of the table, so the function can't aggregate rows, join a table that repeats
    rows, or combine the table with itself. If its result is missing the `pb_index_` column, or has
    repeated or unknown values in it, interrogation raises an error naming the step. These checks
    can't tell a filter from an aggregation that keeps one of each group's rows, along with its
    `pb_index_` value (e.g., taking the first row of each group), so only mark functions that keep
    the rows. Rows that the function filters out weren't checked by the step, so they don't count
    as failing it.

    Parameters
    ----------
    fn
        The function to mark: a function or lambda taking a table and returning a table, as for
        `pre=`. A function with a parameter named `dfn` is given a Narwhals DataFrame, as with any
        `pre=` function.

    Returns
    -------
    Callable
        The function, marked as keeping the rows of the table. It behaves just like `fn`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's validate the order amounts in a table, checking only the orders that were paid. The
    `pre=` function filters the table to those rows, and `row_preserving()` marks it as one that
    keeps the table's rows (it only removes some of them).

    ```{python}
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

    The step checked the three paid orders and order `2` failed it. Order `3` also has a negative
    amount, but it wasn't checked (the `pre=` function filtered it out), so it isn't flagged. Without
    `row_preserving()`, the step would have no flag column at all.

    See Also
    --------
    Validate.get_row_flags : Get which steps each row failed, as columns added to the table.
    Validate.get_sundered_data : Split the table into the rows that passed and failed the steps.
    """
    if not callable(fn):
        raise TypeError(
            f"`row_preserving()` needs a function to mark, got {type(fn).__name__} instead."
        )
    if isinstance(fn, _RowPreservingFunction):
        return fn
    return _RowPreservingFunction(fn)


def _is_row_preserving(pre: Any) -> bool:
    """Check whether a `pre=` function is marked with `row_preserving()`."""
    return isinstance(pre, _RowPreservingFunction)


def _check_row_preserving_result(tbl: Any, n_rows: int, step: int | None) -> None:
    """
    Check that the result of a `row_preserving()` function kept the rows of the table.

    The `pb_index_` column numbered the `n_rows` rows of the table given to the function, so the
    result must have that column, as integers with no missing or repeated values, each the number
    of a row of the table.
    """
    tbl_nw = nw.from_native(tbl)
    schema = tbl_nw.collect_schema()

    def error(problem: str) -> _RowPreservingError:
        return _RowPreservingError(
            f"The `pre=` function of step {step} is marked with `pb.row_preserving()`, but {problem}. "
            "A row-preserving function can filter, reorder, and change the values of rows, but it "
            f"must keep the `{_ROW_INDEX}` column that Pointblank adds to the table, and each row "
            "it returns must come from a different row of the table (so it can't aggregate rows, "
            "or join or concatenate tables that repeat rows)."
        )

    if _ROW_INDEX not in schema.names():
        raise error(f"the table it returned has no `{_ROW_INDEX}` column")
    if not schema[_ROW_INDEX].is_integer():
        raise error(f"it changed the `{_ROW_INDEX}` column to {schema[_ROW_INDEX]}")

    index = nw.col(_ROW_INDEX)
    stats = tbl_nw.select(
        n_values=index.count(),
        n_missing=index.null_count(),
        n_unique=index.n_unique(),
        min_value=index.min(),
        max_value=index.max(),
    )
    if isinstance(stats, nw.LazyFrame):
        stats = stats.collect()
    stats = stats.rows(named=True)[0]

    if stats["n_missing"]:
        raise error(f"the `{_ROW_INDEX}` column it returned has missing values")
    if stats["n_values"] == 0:
        return
    if stats["n_unique"] != stats["n_values"]:
        raise error(f"the `{_ROW_INDEX}` column it returned has repeated values")
    if stats["min_value"] < 0 or stats["max_value"] >= n_rows:
        raise error(f"the `{_ROW_INDEX}` column it returned has values that aren't row numbers")
