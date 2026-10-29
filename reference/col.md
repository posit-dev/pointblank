# col()

Helper function for referencing a column in the input table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/column.py#L250-L545)

``` python
col(exprs)
```

Many of the validation methods (i.e., `col_vals_*()` methods) in Pointblank have a `value=` argument. These validations are comparisons between column values and a literal value, or, between column values and adjacent values in another column. The [col()](../reference/col.md#pointblank.col) helper function is used to specify that it is a column being referenced, not a literal value.

The [col()](../reference/col.md#pointblank.col) doesn't check that the column exists in the input table. It acts to signal that the value being compared is a column value. During validation (i.e., when [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) is called), Pointblank will then check that the column exists in the input table.

For creating expressions to use with the `conjointly()` validation method, use the [`expr_col()`](../reference/expr_col.md#pointblank.expr_col) function instead.

## Parameters

`exprs: str | ColumnSelector | ColumnSelectorNarwhals | nw.selectors.Selector`  
Either the name of a single column in the target table, provided as a string, or, an expression involving column selector functions (e.g., `starts_with("a")`, `ends_with("e") | starts_with("a")`, etc.).

## Returns

`Column | ColumnLiteral | ColumnSelectorNarwhals:`  
A column object or expression representing the column reference.

## Usage With The `columns=` Argument

The [col()](../reference/col.md#pointblank.col) function can be used in the `columns=` argument of the following validation methods:

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

If specifying a single column with certainty (you have the exact name), [col()](../reference/col.md#pointblank.col) is not necessary since you can just pass the column name as a string (though it is still valid to use `col("column_name")`, if preferred). However, if you want to select columns based on complex logic involving multiple column selector functions (e.g., columns that start with `"a"` but don't end with `"e"`), you need to use [col()](../reference/col.md#pointblank.col) to wrap expressions involving column selector functions and logical operators such as `&`, `|`, `-`, and `~`.

Here is an example of such usage with the [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) validation method:

``` python
col_vals_gt(columns=col(starts_with("a") & ~ends_with("e")), value=10)
```

If using only a single column selector function, you can pass the function directly to the `columns=` argument of the validation method, or, you can use [col()](../reference/col.md#pointblank.col) to wrap the function (either is valid though the first is more concise). Here is an example of that simpler usage:

``` python
col_vals_gt(columns=starts_with("a"), value=10)
```

## Usage With The `value=`, `left=`, And `right=` Arguments

The [col()](../reference/col.md#pointblank.col) function can be used in the `value=` argument of the following validation methods

- [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)
- [`col_vals_lt()`](../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt)
- [`col_vals_ge()`](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge)
- [`col_vals_le()`](../reference/Validate.col_vals_le.md#pointblank.Validate.col_vals_le)
- [`col_vals_eq()`](../reference/Validate.col_vals_eq.md#pointblank.Validate.col_vals_eq)
- [`col_vals_ne()`](../reference/Validate.col_vals_ne.md#pointblank.Validate.col_vals_ne)

and in the `left=` and `right=` arguments (either or both) of these two validation methods

- [`col_vals_between()`](../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between)
- [`col_vals_outside()`](../reference/Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside)

You cannot use column selector functions such as [`starts_with()`](../reference/starts_with.md#pointblank.starts_with) in either of the `value=`, `left=`, or `right=` arguments since there would be no guarantee that a single column will be resolved from the target table with this approach. The [col()](../reference/col.md#pointblank.col) function is used to signal that the value being compared is a column value and not a literal value.

## Available Selectors

There is a collection of selectors available in pointblank, allowing you to select columns based on attributes of column names and positions. The selectors are:

- [`starts_with()`](../reference/starts_with.md#pointblank.starts_with)
- [`ends_with()`](../reference/ends_with.md#pointblank.ends_with)
- [`contains()`](../reference/contains.md#pointblank.contains)
- [`matches()`](../reference/matches.md#pointblank.matches)
- [`everything()`](../reference/everything.md#pointblank.everything)
- [`first_n()`](../reference/first_n.md#pointblank.first_n)
- [`last_n()`](../reference/last_n.md#pointblank.last_n)

Alternatively, we support selectors from the Narwhals library! Those selectors can additionally take advantage of the data types of the columns. The selectors are:

- `boolean()`
- `by_dtype()`
- `categorical()`
- `matches()`
- `numeric()`
- `string()`

Have a look at the [Narwhals API documentation on selectors](https://narwhals-dev.github.io/narwhals/api-reference/selectors/) for more information.

## Examples

Suppose we have a table with columns `a` and `b` and we'd like to validate that the values in column `a` are greater than the values in column `b`. We can use the [col()](../reference/col.md#pointblank.col) helper function to reference the comparison column when creating the validation step.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [5, 6, 5, 7, 6, 5],
        "b": [4, 2, 3, 3, 4, 3],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns="a", value=pb.col("b"))
    .interrogate()
)

validation
```

[TABLE]

From results of the validation table it can be seen that values in `a` were greater than values in `b` for every row (or test unit). Using `value=pb.col("b")` specified that the greater-than comparison is across columns, not with a fixed literal value.

If you want to select an arbitrary set of columns upon which to base a validation, you can use column selector functions (e.g., [`starts_with()`](../reference/starts_with.md#pointblank.starts_with), [`ends_with()`](../reference/ends_with.md#pointblank.ends_with), etc.) to specify columns in the `columns=` argument of a validation method. Let's use the [`starts_with()`](../reference/starts_with.md#pointblank.starts_with) column selector function to select columns that start with `"paid"` and validate that the values in those columns are greater than `10`.

``` python
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
    .col_vals_gt(columns=pb.col(pb.starts_with("paid")), value=10)
    .interrogate()
)

validation
```

[TABLE]

In the above example the [col()](../reference/col.md#pointblank.col) function contains the invocation of the [`starts_with()`](../reference/starts_with.md#pointblank.starts_with) column selector function. This is not strictly necessary when using a single column selector function, so `columns=pb.starts_with("paid")` would be equivalent usage here. However, the use of [col()](../reference/col.md#pointblank.col) is required when using multiple column selector functions with logical operators. Here is an example of that more complex usage:

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
        columns=pb.col(pb.starts_with("paid") & pb.matches("2023|2024")),
        value=10
    )
    .interrogate()
)

validation
```

[TABLE]

In the above example the [col()](../reference/col.md#pointblank.col) function contains the invocation of the [`starts_with()`](../reference/starts_with.md#pointblank.starts_with) and [`matches()`](../reference/matches.md#pointblank.matches) column selector functions, combined with the `&` operator. This is necessary to specify the set of columns that start with `"paid"` *and* match the text `"2023"` or `"2024"`.

If you'd like to take advantage of Narwhals selectors, that's also possible. Here is an example of using the `numeric()` column selector function to select all numeric columns for validation, checking that their values are greater than `0`.

``` python
import narwhals.selectors as ncs

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
    .col_vals_ge(columns=pb.col(ncs.numeric()), value=0)
    .interrogate()
)

validation
```

[TABLE]

In the above example the [col()](../reference/col.md#pointblank.col) function contains the invocation of the `numeric()` column selector function from Narwhals. As with the other selectors, this is not strictly necessary when using a single column selector, so `columns=ncs.numeric()` would also be fine here.

Narwhals selectors can also use operators to combine multiple selectors. Here is an example of using the `numeric()` and [`matches()`](../reference/matches.md#pointblank.matches) selectors together to select all numeric columns that fit a specific pattern.

``` python
tbl = pl.DataFrame(
    {
        "name": ["Alice", "Bob", "Charlie"],
        "2022_status": ["ft", "ft", "pt"],
        "2023_status": ["ft", "pt", "ft"],
        "2024_status": ["ft", "pt", "ft"],
        "2022_pay_total": [18.62, 16.95, 18.25],
        "2023_pay_total": [19.29, 17.75, 18.35],
        "2024_pay_total": [20.73, 18.35, 20.10],
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_lt(columns=pb.col(ncs.numeric() & ncs.matches("2023|2024")), value=30)
    .interrogate()
)

validation
```

[TABLE]

In the above example the [col()](../reference/col.md#pointblank.col) function contains the invocation of the `numeric()` and [`matches()`](../reference/matches.md#pointblank.matches) column selector functions from Narwhals, combined with the `&` operator. This is necessary to specify the set of columns that are numeric *and* match the text `"2023"` or `"2024"`.

## See Also

[expr_col()](../reference/expr_col.md#pointblank.expr_col)  
Create a column expression for use in `conjointly()`.
