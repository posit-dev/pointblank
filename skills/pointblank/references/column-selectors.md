# Column selectors reference

Column selectors let you target multiple columns by pattern instead
of listing them individually.

## Available selectors

| Selector            | Description                              | Example                       |
|---------------------|------------------------------------------|-------------------------------|
| `col("name")`       | Explicit column name                    | `col("revenue")`             |
| `starts_with(text)` | Columns whose name starts with text     | `starts_with("price")`       |
| `ends_with(text)`   | Columns whose name ends with text       | `ends_with("_id")`           |
| `contains(text)`    | Columns whose name contains text        | `contains("amount")`         |
| `matches(pattern)`  | Columns matching a regex                | `matches(r"^col_\d+")`       |
| `everything()`      | All columns                             | `everything()`               |
| `first_n(n)`        | First n columns                         | `first_n(3)`                 |
| `last_n(n)`         | Last n columns                          | `last_n(2)`                  |

## Parameters

All text-based selectors accept `case_sensitive: bool = False`.

`first_n` and `last_n` accept `offset: int = 0` to skip columns
before counting.

## Combining selectors with operators

| Operator | Meaning    | Example                                  |
|----------|------------|------------------------------------------|
| `\|`     | Union      | `starts_with("a") \| starts_with("b")`   |
| `&`      | Intersect  | `contains("price") & ends_with("_usd")`  |
| `-`      | Difference | `everything() - matches("_tmp$")`        |
| `~`      | Negate     | `~contains("debug")`                     |

## Expression columns (for conjointly)

`expr_col()` creates backend-agnostic column expressions for use in
`conjointly()`. Each argument of `conjointly()` is a callable that takes
the table and returns an expression:

```python
from pointblank import expr_col

.conjointly(
    lambda df: expr_col("start") < expr_col("end"),
    lambda df: expr_col("duration") > 0,
)
```

`expr_col` supports: `>`, `<`, `==`, `!=`, `>=`, `<=`, `+`, `-`,
`*`, `/`, `is_null()`, `is_not_null()`, `&`, `|`. It doesn't work in
`col_vals_expr()`, which needs a native Polars/Narwhals/Ibis
expression (e.g. `pl.col("start") < pl.col("end")`).

## Reference columns

`ref()` references a column in the reference table for aggregate
comparisons:

```python
validation = (
    pb.Validate(data=current_df, reference=previous_df)
    .col_sum_eq(columns="revenue", value=ref("revenue"))
    .interrogate()
)
```

When `value=None` in aggregate methods and a reference table is set,
`ref(column)` is used automatically.

## Usage in validation methods

Selectors work in the `col_vals_*` methods (except `col_vals_scored`),
`col_exists`, `col_pct_null`, `col_pct_missing`, `col_missing_coded`,
and `col_missing_only_coded`. The aggregate methods (`col_sum_gt`, ...),
`col_missing_consistent`, and the text checks (`col_vals_scored`,
`rows_near_distinct`, `rows_not_contaminated`) take column names only:

```python
import pointblank as pb
from pointblank import starts_with, ends_with, everything

(
    pb.Validate(data=df)
    .col_vals_gt(columns=starts_with("price"), value=0)
    .col_vals_not_null(columns=everything() - ends_with("_notes"))
    .col_vals_in_set(columns="status", set=["active", "inactive"])
    .interrogate()
)
```
