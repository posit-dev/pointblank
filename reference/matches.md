# matches()

Select columns that match a specified regular expression pattern.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/column.py#L1113-L1273)

``` python
matches(
    pattern,
    case_sensitive=False,
)
```

Many validation methods have a `columns=` argument that can be used to specify the columns for validation (e.g., [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt), [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex), etc.). The `matches()` selector function can be used to select one or more columns matching a provided regular expression pattern. So if the set of table columns consists of

`[rev_01, rev_02, profit_01, profit_02, age]`

and you want to validate columns that have two digits at the end of the name, you can use `columns=matches(r"[0-9]{2}$")`. This will select the `rev_01`, `rev_02`, `profit_01`, and `profit_02` columns.

There will be a validation step created for every resolved column. Note that if there aren't any columns resolved from using `matches()` (or any other expression using selector functions), the validation step will fail to be evaluated during the interrogation process. Such a failure to evaluate will be reported in the validation results but it won't affect the interrogation process overall (i.e., the process won't be halted).

## Parameters

`pattern: str`  
The regular expression pattern that the column name should match.

`case_sensitive: bool = False`  
Whether column names should be treated as case-sensitive. The default is `False`.

## Returns

`Matches`  
A `Matches` object, which can be used to select columns that match the specified pattern.

## Relevant Validation Methods Where `matches()` Can Be Used

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

The `matches()` selector function doesn't need to be used in isolation. Read the next section for information on how to compose it with other column selectors for more refined ways to select columns.

## Additional Flexibilty Through Composition With Other Column Selectors

The `matches()` function can be composed with other column selectors to create fine-grained column selections. For example, to select columns that have the text starting with five digits and end with `"_id"`, you can use the `matches()` and [`ends_with()`](../reference/ends_with.md#pointblank.ends_with) functions together. The only condition is that the expressions are wrapped in the [`col()`](../reference/col.md#pointblank.col) function, like this:

``` python
col(matches(r"^[0-9]{5}") & ends_with("_id"))
```

There are four operators that can be used to compose column selectors:

- `&` (*and*)
- `|` (*or*)
- `-` (*difference*)
- `~` (*not*)

The `&` operator is used to select columns that satisfy both conditions. The `|` operator is used to select columns that satisfy either condition. The `-` operator is used to select columns that satisfy the first condition but not the second. The `~` operator is used to select columns that don't satisfy the condition. As many selector functions can be used as needed and the operators can be combined to create complex column selection criteria (parentheses can be used to group conditions and control the order of evaluation).

## Examples

Suppose we have a table with columns `name`, `id_old`, `new_identifier`, and `pay_2021` and we'd like to validate that text values in columns having `"id"` or `"identifier"` in the name have a specific syntax. We can use the `matches()` column selector function to specify the columns that match the pattern.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
        "id_old": ["ID0021", "ID0032", "ID0043"],
        "new_identifier": ["ID9054", "ID9065", "ID9076"],
        "pay_2021": [16.32, 16.25, 15.75],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_regex(columns=pb.matches("id|identifier"), pattern=r"ID[0-9]{4}")
    .interrogate()
)

validation
```

[TABLE]

From the results of the validation table we get two validation steps, one for `id_old` and one for `new_identifier`. The values in both columns all match the pattern `"ID[0-9]{4}"`.

We can also use the `matches()` function in combination with other column selectors (within [`col()`](../reference/col.md#pointblank.col)) to create more complex column selection criteria (i.e., to select columns that satisfy multiple conditions). For example, to select columns that contain `"pay"` and match the text `"2023"` or `"2024"`, we can use the `&` operator to combine column selectors.

``` python
tbl = pl.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
        "2022_hours": [160, 180, 160],
        "2023_hours": [182, 168, 175],
        "2024_hours": [200, 165, 190],
        "2022_pay_total": [18.62, 16.95, 18.25],
        "2023_pay_total": [19.29, 17.75, 18.35],
        "2024_pay_total": [20.73, 18.35, 20.10],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(
        columns=pb.col(pb.contains("pay") & pb.matches("2023|2024")),
        value=10
    )
    .interrogate()
)

validation
```

[TABLE]

From the results of the validation table we get two validation steps, one for `2023_pay_total` and one for `2024_pay_total`.
