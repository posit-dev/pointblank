# last_n()

Select the last `n` columns in the column list.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/column.py#L1573-L1729)

``` python
last_n(
    n,
    offset=0,
)
```

Many validation methods have a `columns=` argument that can be used to specify the columns for validation (e.g., [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt), [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex), etc.). The [last_n()](../reference/last_n.md#pointblank.last_n) selector function can be used to select *n* columns positioned at the end of the column list. So if the set of table columns consists of

`[age, rev_01, rev_02, profit_01, profit_02]`

and you want to validate the last two columns, you can use `columns=last_n(2)`. This will select the `profit_01` and `profit_02` columns and a validation step will be created for each.

The `offset=` parameter can be used to skip a certain number of columns from the end of the column list. So if you want to select the third and fourth columns from the end, you can use `columns=last_n(2, offset=2)`.

## Parameters

`n: int`  
The number of columns to select from the end of the column list. Should be a positive integer value. If `n` is greater than the number of columns in the table, all columns will be selected.

`offset: int = 0`  
The offset from the end of the column list. The default is `0`. If `offset` is greater than the number of columns in the table, no columns will be selected.

## Returns

`LastN`  
A `LastN` object, which can be used to select the last `n` columns.

## Relevant Validation Methods Where [last_n()](../reference/last_n.md#pointblank.last_n) Can Be Used

This selector function can be used in the `columns=` argument of the following validation methods:

- [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)
- [`col_vals_lt()`](../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt)
- [`col_vals_ge()`](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge)
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
- [`col_exists()`](../reference/Validate.col_exists.md#pointblank.Validate.col_exists)

The [last_n()](../reference/last_n.md#pointblank.last_n) selector function doesn't need to be used in isolation. Read the next section for information on how to compose it with other column selectors for more refined ways to select columns.

## Additional Flexibilty Through Composition With Other Column Selectors

The [last_n()](../reference/last_n.md#pointblank.last_n) function can be composed with other column selectors to create fine-grained column selections. For example, to select all column names starting with "rev" along with the last two columns, you can use the [last_n()](../reference/last_n.md#pointblank.last_n) and [`starts_with()`](../reference/starts_with.md#pointblank.starts_with) functions together. The only condition is that the expressions are wrapped in the [`col()`](../reference/col.md#pointblank.col) function, like this:

``` python
col(last_n(2) | starts_with("rev"))
```

There are four operators that can be used to compose column selectors:

- `&` (*and*)
- `|` (*or*)
- `-` (*difference*)
- `~` (*not*)

The `&` operator is used to select columns that satisfy both conditions. The `|` operator is used to select columns that satisfy either condition. The `-` operator is used to select columns that satisfy the first condition but not the second. The `~` operator is used to select columns that don't satisfy the condition. As many selector functions can be used as needed and the operators can be combined to create complex column selection criteria (parentheses can be used to group conditions and control the order of evaluation).

## Examples

Suppose we have a table with columns `name`, `paid_2021`, `paid_2022`, `paid_2023`, and `paid_2024` and we'd like to validate that the values in the last four columns are greater than `10`. We can use the [last_n()](../reference/last_n.md#pointblank.last_n) column selector function to specify that the last four columns in the table are the columns to validate.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
        "paid_2021": [17.94, 16.55, 17.85],
        "paid_2022": [18.62, 16.95, 18.25],
        "paid_2023": [19.29, 17.75, 18.35],
        "paid_2024": [20.73, 18.35, 20.10],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns=pb.last_n(4), value=10)
    .interrogate()
)

validation
```

[TABLE]

From the results of the validation table we get four validation steps. The values in all those columns were all greater than `10`.

We can also use the [last_n()](../reference/last_n.md#pointblank.last_n) function in combination with other column selectors (within [`col()`](../reference/col.md#pointblank.col)) to create more complex column selection criteria (i.e., to select columns that satisfy multiple conditions). For example, to select the last four columns but also omit those columns that end with `"2023"`, we can use the `-` operator to combine column selectors.

``` python
tbl = pl.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
        "paid_2021": [17.94, 16.55, 17.85],
        "paid_2022": [18.62, 16.95, 18.25],
        "paid_2023": [19.29, 17.75, 18.35],
        "paid_2024": [20.73, 18.35, 20.10],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns=pb.col(pb.last_n(4) - pb.ends_with("2023")), value=10)
    .interrogate()
)

validation
```

[TABLE]

From the results of the validation table we get three validation steps, one for `paid_2021`, `paid_2022`, and `paid_2024`.
