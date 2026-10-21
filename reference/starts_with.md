# starts_with()

Select columns that start with specified text.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/column.py#L626-L786)

``` python
starts_with(
    text,
    case_sensitive=False,
)
```

Many validation methods have a `columns=` argument that can be used to specify the columns for validation (e.g., [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt), [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex), etc.). The [starts_with()](../reference/starts_with.md#pointblank.starts_with) selector function can be used to select one or more columns that start with some specified text. So if the set of table columns consists of

`[name_first, name_last, age, address]`

and you want to validate columns that start with `"name"`, you can use `columns=starts_with("name")`. This will select the `name_first` and `name_last` columns.

There will be a validation step created for every resolved column. Note that if there aren't any columns resolved from using [starts_with()](../reference/starts_with.md#pointblank.starts_with) (or any other expression using selector functions), the validation step will fail to be evaluated during the interrogation process. Such a failure to evaluate will be reported in the validation results but it won't affect the interrogation process overall (i.e., the process won't be halted).

## Parameters

`text: str`  
The text that the column name should start with.

`case_sensitive: bool = False`  
Whether column names should be treated as case-sensitive. The default is `False`.

## Returns

`StartsWith`  
A `StartsWith` object, which can be used to select columns that start with the specified text.

## Relevant Validation Methods Where [starts_with()](../reference/starts_with.md#pointblank.starts_with) Can Be Used

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

The [starts_with()](../reference/starts_with.md#pointblank.starts_with) selector function doesn't need to be used in isolation. Read the next section for information on how to compose it with other column selectors for more refined ways to select columns.

## Additional Flexibilty Through Composition With Other Column Selectors

The [starts_with()](../reference/starts_with.md#pointblank.starts_with) function can be composed with other column selectors to create fine-grained column selections. For example, to select columns that start with `"a"` and end with `"e"`, you can use the [starts_with()](../reference/starts_with.md#pointblank.starts_with) and [`ends_with()`](../reference/ends_with.md#pointblank.ends_with) functions together. The only condition is that the expressions are wrapped in the [`col()`](../reference/col.md#pointblank.col) function, like this:

``` python
col(starts_with("a") & ends_with("e"))
```

There are four operators that can be used to compose column selectors:

- `&` (*and*)
- `|` (*or*)
- `-` (*difference*)
- `~` (*not*)

The `&` operator is used to select columns that satisfy both conditions. The `|` operator is used to select columns that satisfy either condition. The `-` operator is used to select columns that satisfy the first condition but not the second. The `~` operator is used to select columns that don't satisfy the condition. As many selector functions can be used as needed and the operators can be combined to create complex column selection criteria (parentheses can be used to group conditions and control the order of evaluation).

## Examples

Suppose we have a table with columns `name`, `paid_2021`, `paid_2022`, and `person_id` and we'd like to validate that the values in columns that start with `"paid"` are greater than `10`. We can use the [starts_with()](../reference/starts_with.md#pointblank.starts_with) column selector function to specify the columns that start with `"paid"` as the columns to validate.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
        "paid_2021": [16.32, 16.25, 15.75],
        "paid_2022": [18.62, 16.95, 18.25],
        "person_id": ["A123", "B456", "C789"],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns=pb.starts_with("paid"), value=10)
    .interrogate()
)

validation
```

[TABLE]

From the results of the validation table we get two validation steps, one for `paid_2021` and one for `paid_2022`. The values in both columns were all greater than `10`.

We can also use the [starts_with()](../reference/starts_with.md#pointblank.starts_with) function in combination with other column selectors (within [`col()`](../reference/col.md#pointblank.col)) to create more complex column selection criteria (i.e., to select columns that satisfy multiple conditions). For example, to select columns that start with `"paid"` and match the text `"2023"` or `"2024"`, we can use the `&` operator to combine column selectors.

``` python
tbl = pl.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
        "hours_2022": [160, 180, 160],
        "hours_2023": [182, 168, 175],
        "hours_2024": [200, 165, 190],
        "paid_2022": [18.62, 16.95, 18.25],
        "paid_2023": [19.29, 17.75, 18.35],
        "paid_2024": [20.73, 18.35, 20.10],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(
        columns=pb.col(pb.starts_with("paid") & pb.matches("23|24")),
        value=10
    )
    .interrogate()
)

validation
```

[TABLE]

From the results of the validation table we get two validation steps, one for `paid_2023` and one for `paid_2024`.
