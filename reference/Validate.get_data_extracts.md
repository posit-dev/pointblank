# Validate.get_data_extracts()

Get the rows that failed for each validation step.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L19211-L19355)

``` python
Validate.get_data_extracts(
    i=None,
    frame=False,
)
```

After the [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) method has been called, the [get_data_extracts()](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) method can be used to extract the rows that failed in each column-value or row-based validation step (e.g., [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt), [`rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct), etc.). The method returns a dictionary of tables containing the rows that failed in every validation step. If `frame=True` and `i=` is a scalar, the value is conveniently returned as a table (forgoing the dictionary structure).

## Parameters

`i: int | list[int] | None = None`  
The validation step number(s) from which the failed rows are obtained. Can be provided as a list of integers or a single integer. If `None`, all steps are included.

`frame: bool = False`  
If `True` and `i=` is a scalar, return the value as a DataFrame instead of a dictionary.

## Returns

`dict[int, Any] | Any`  
A dictionary of tables containing the rows that failed in every compatible validation step. Alternatively, it can be a DataFrame if `frame=True` and `i=` is a scalar.

## Compatible Validation Methods For Yielding Extracted Rows

The following validation methods operate on column values and will have rows extracted when there are failing test units.

- [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)
- [`col_vals_ge()`](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge)
- [`col_vals_lt()`](../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt)
- [`col_vals_le()`](../reference/Validate.col_vals_le.md#pointblank.Validate.col_vals_le)
- [`col_vals_eq()`](../reference/Validate.col_vals_eq.md#pointblank.Validate.col_vals_eq)
- [`col_vals_ne()`](../reference/Validate.col_vals_ne.md#pointblank.Validate.col_vals_ne)
- [`col_vals_between()`](../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between)
- [`col_vals_outside()`](../reference/Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside)
- [`col_vals_in_set()`](../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set)
- [`col_vals_not_in_set()`](../reference/Validate.col_vals_not_in_set.md#pointblank.Validate.col_vals_not_in_set)
- [`col_vals_increasing()`](../reference/Validate.col_vals_increasing.md#pointblank.Validate.col_vals_increasing)
- [`col_vals_decreasing()`](../reference/Validate.col_vals_decreasing.md#pointblank.Validate.col_vals_decreasing)
- [`col_vals_null()`](../reference/Validate.col_vals_null.md#pointblank.Validate.col_vals_null)
- [`col_vals_not_null()`](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null)
- [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex)
- [`col_vals_within_spec()`](../reference/Validate.col_vals_within_spec.md#pointblank.Validate.col_vals_within_spec)
- [`col_vals_expr()`](../reference/Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr)
- [`conjointly()`](../reference/Validate.conjointly.md#pointblank.Validate.conjointly)
- [`prompt()`](../reference/Validate.prompt.md#pointblank.Validate.prompt)

An extracted row for these validation methods means that a test unit failed for that row in the validation step.

These row-based validation methods will also have rows extracted should there be failing rows:

- [`rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct)
- [`rows_complete()`](../reference/Validate.rows_complete.md#pointblank.Validate.rows_complete)

The extracted rows are a subset of the original table and are useful for further analysis or for understanding the nature of the failing test units.

## Examples

Let's perform a series of validation steps on a Polars DataFrame. We'll use the [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) in the first step, [`col_vals_lt()`](../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt) in the second step, and [`col_vals_ge()`](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge) in the third step. The [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) method executes the validation; then, we can extract the rows that failed for each validation step.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [5, 6, 5, 3, 6, 1],
        "b": [1, 2, 1, 5, 2, 6],
        "c": [3, 7, 2, 6, 3, 1],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns="a", value=4)
    .col_vals_lt(columns="c", value=5)
    .col_vals_ge(columns="b", value=1)
    .interrogate()
)

validation.get_data_extracts()
```

    {1: shape: (2, 4)
     ┌───────────┬─────┬─────┬─────┐
     │ _row_num_ ┆ a   ┆ b   ┆ c   │
     │ ---       ┆ --- ┆ --- ┆ --- │
     │ u32       ┆ i64 ┆ i64 ┆ i64 │
     ╞═══════════╪═════╪═════╪═════╡
     │ 4         ┆ 3   ┆ 5   ┆ 6   │
     │ 6         ┆ 1   ┆ 6   ┆ 1   │
     └───────────┴─────┴─────┴─────┘,
     2: shape: (2, 4)
     ┌───────────┬─────┬─────┬─────┐
     │ _row_num_ ┆ a   ┆ b   ┆ c   │
     │ ---       ┆ --- ┆ --- ┆ --- │
     │ u32       ┆ i64 ┆ i64 ┆ i64 │
     ╞═══════════╪═════╪═════╪═════╡
     │ 2         ┆ 6   ┆ 2   ┆ 7   │
     │ 4         ┆ 3   ┆ 5   ┆ 6   │
     └───────────┴─────┴─────┴─────┘,
     3: shape: (0, 4)
     ┌───────────┬─────┬─────┬─────┐
     │ _row_num_ ┆ a   ┆ b   ┆ c   │
     │ ---       ┆ --- ┆ --- ┆ --- │
     │ u32       ┆ i64 ┆ i64 ┆ i64 │
     ╞═══════════╪═════╪═════╪═════╡
     └───────────┴─────┴─────┴─────┘}

The [get_data_extracts()](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts) method returns a dictionary of tables, where each table contains a subset of rows from the table. These are the rows that failed for each validation step.

In the first step, the[`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) method was used to check if the values in column `a` were greater than `4`. The extracted table shows the rows where this condition was not met; look at the `a` column: all values are less than `4`.

In the second step, the [`col_vals_lt()`](../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt) method was used to check if the values in column `c` were less than `5`. In the extracted two-row table, we see that the values in column `c` are greater than `5`.

The third step ([`col_vals_ge()`](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge)) checked if the values in column `b` were greater than or equal to `1`. There were no failing test units, so the extracted table is empty (i.e., has columns but no rows).

The `i=` argument can be used to narrow down the extraction to one or more steps. For example, to extract the rows that failed in the first step only:

``` python
validation.get_data_extracts(i=1)
```

    {1: shape: (2, 4)
     ┌───────────┬─────┬─────┬─────┐
     │ _row_num_ ┆ a   ┆ b   ┆ c   │
     │ ---       ┆ --- ┆ --- ┆ --- │
     │ u32       ┆ i64 ┆ i64 ┆ i64 │
     ╞═══════════╪═════╪═════╪═════╡
     │ 4         ┆ 3   ┆ 5   ┆ 6   │
     │ 6         ┆ 1   ┆ 6   ┆ 1   │
     └───────────┴─────┴─────┴─────┘}

Note that the first validation step is indexed at `1` (not `0`). This 1-based indexing is in place here to match the step numbers reported in the validation table. What we get back is still a dictionary, but it only contains one table (the one for the first step).

If you want to get the extracted table as a DataFrame, set `frame=True` and provide a scalar value for `i`. For example, to get the extracted table for the second step as a DataFrame:

``` python
pb.preview(validation.get_data_extracts(i=2, frame=True))
```

|     | a *Int64* | b *Int64* | c *Int64* |
|-----|-----------|-----------|-----------|
| 2   | 6         | 2         | 7         |
| 4   | 3         | 5         | 6         |

The extracted table is now a DataFrame, which can serve as a more convenient format for further analysis or visualization. We further used the [`preview()`](../reference/preview.md#pointblank.preview) function to show the DataFrame in an HTML view.
