# everything()

Select all columns.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/column.py#L1277-L1411)

``` python
everything()
```

Many validation methods have a `columns=` argument that can be used to specify the columns for validation (e.g., [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt), [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex), etc.). The [everything()](../reference/everything.md#pointblank.everything) selector function can be used to select every column in the table. If you have a table with six columns and they're all suitable for a specific type of validation, you can use `columns=everything())` and all six columns will be selected for validation.

## Returns

`Everything`  
An `Everything` object, which can be used to select all columns.

## Relevant Validation Methods Where [everything()](../reference/everything.md#pointblank.everything) Can Be Used

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

The [everything()](../reference/everything.md#pointblank.everything) selector function doesn't need to be used in isolation. Read the next section for information on how to compose it with other column selectors for more refined ways to select columns.

## Additional Flexibilty Through Composition With Other Column Selectors

The [everything()](../reference/everything.md#pointblank.everything) function can be composed with other column selectors to create fine-grained column selections. For example, to select all column names except those having starting with "id\_", you can use the [everything()](../reference/everything.md#pointblank.everything) and [`starts_with()`](../reference/starts_with.md#pointblank.starts_with) functions together. The only condition is that the expressions are wrapped in the [`col()`](../reference/col.md#pointblank.col) function, like this:

``` python
col(everything() - starts_with("id_"))
```

There are four operators that can be used to compose column selectors:

- `&` (*and*)
- `|` (*or*)
- `-` (*difference*)
- `~` (*not*)

The `&` operator is used to select columns that satisfy both conditions. The `|` operator is used to select columns that satisfy either condition. The `-` operator is used to select columns that satisfy the first condition but not the second. The `~` operator is used to select columns that don't satisfy the condition. As many selector functions can be used as needed and the operators can be combined to create complex column selection criteria (parentheses can be used to group conditions and control the order of evaluation).

## Examples

Suppose we have a table with several numeric columns and we'd like to validate that all these columns have less than `1000`. We can use the [everything()](../reference/everything.md#pointblank.everything) column selector function to select all columns for validation.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "2023_hours": [182, 168, 175],
        "2024_hours": [200, 165, 190],
        "2023_pay_total": [19.29, 17.75, 18.35],
        "2024_pay_total": [20.73, 18.35, 20.10],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_lt(columns=pb.everything(), value=1000)
    .interrogate()
)

validation
```

[TABLE]

From the results of the validation table we get four validation steps, one each column in the table. The values in every column were all lower than `1000`.

We can also use the [everything()](../reference/everything.md#pointblank.everything) function in combination with other column selectors (within [`col()`](../reference/col.md#pointblank.col)) to create more complex column selection criteria (i.e., to select columns that satisfy multiple conditions). For example, to select every column except those that begin with `"2023"` we can use the `-` operator to combine column selectors.

``` python
tbl = pl.DataFrame(
    {
        "2023_hours": [182, 168, 175],
        "2024_hours": [200, 165, 190],
        "2023_pay_total": [19.29, 17.75, 18.35],
        "2024_pay_total": [20.73, 18.35, 20.10],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_lt(columns=pb.col(pb.everything() - pb.starts_with("2023")), value=1000)
    .interrogate()
)

validation
```

[TABLE]

From the results of the validation table we get two validation steps, one for `2024_hours` and one for `2024_pay_total`.
