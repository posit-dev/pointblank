# Column Selection Patterns

Data validation often requires working with columns in flexible ways. Pointblank offers two powerful approaches:

1.  Applying validation rules across multiple columns: validate many columns with a single rule
2.  Comparing values between columns: create validations that compare values across different columns

This guide covers both approaches in detail with practical examples.

## Part 1: Applying Rules Across Multiple Columns

Many of Pointblank's validation methods perform column-level checks. These methods provide the `columns=` parameter, which accepts not just a single column name but multiple columns through various selection methods.

Why is this useful? Often you'll want to perform the same validation check (e.g., checking that numerical values are all positive) across multiple columns. Rather than defining the same rules multiple times, you can map the validation across those columns in a single step.

Let's explore this using the `game_revenue` dataset:

| PolarsRows2,000Columns11 |  |  |  |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|----|----|----|
|  | player_id *String* | session_id *String* | session_start *Datetime* | time *Datetime* | item_type *String* | item_name *String* | item_revenue *Float64* | session_duration *Float64* | start_day *Date* | acquisition *String* | country *String* |
| 1 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:31:27+00:00 | iap | offer2 | 8.99 | 16.3 | 2015-01-01 | google | Germany |
| 2 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:36:57+00:00 | iap | gems3 | 22.49 | 16.3 | 2015-01-01 | google | Germany |
| 3 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:37:45+00:00 | iap | gold7 | 107.99 | 16.3 | 2015-01-01 | google | Germany |
| 4 | ECPANOIXLZHF896 | ECPANOIXLZHF896-eol2j8bs | 2015-01-01 01:31:03+00:00 | 2015-01-01 01:42:33+00:00 | ad | ad_20sec | 0.76 | 16.3 | 2015-01-01 | google | Germany |
| 5 | ECPANOIXLZHF896 | ECPANOIXLZHF896-hdu9jkls | 2015-01-01 11:50:02+00:00 | 2015-01-01 11:55:20+00:00 | ad | ad_5sec | 0.03 | 35.2 | 2015-01-01 | google | Germany |
| 1996 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:02:50+00:00 | ad | ad_survey | 1.332 | 25.8 | 2015-01-11 | organic | Norway |
| 1997 | NAOJRDMCSEBI281 | NAOJRDMCSEBI281-j2vs9ilp | 2015-01-21 01:57:50+00:00 | 2015-01-21 02:22:14+00:00 | ad | ad_survey | 1.35 | 25.8 | 2015-01-11 | organic | Norway |
| 1998 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:40:00+00:00 | ad | ad_5sec | 0.03 | 8.4 | 2015-01-10 | other_campaign | France |
| 1999 | RMOSWHJGELCI675 | RMOSWHJGELCI675-vbhcsmtr | 2015-01-21 02:39:48+00:00 | 2015-01-21 02:47:12+00:00 | iap | offer5 | 26.09 | 8.4 | 2015-01-10 | other_campaign | France |
| 2000 | GJCXNTWEBIPQ369 | GJCXNTWEBIPQ369-9elq67md | 2015-01-21 03:59:23+00:00 | 2015-01-21 04:06:29+00:00 | ad | ad_5sec | 0.12 | 18.5 | 2015-01-14 | organic | United States |

### Using a List of Column Names

The simplest way to validate multiple columns is to provide a list to the `columns=` parameter. In the `game_revenue` dataset, we have two columns with numerical data: `item_revenue` and `session_duration`. If we expect all values in both columns to be greater than `0`, we can write:

``` python
import pointblank as pb

(
    pb.Validate(data=pb.load_dataset("game_revenue"))
    .col_vals_gt(
        columns=["item_revenue", "session_duration"],
        value=0
    )
    .interrogate()
)
```

[TABLE]

The validation report shows two validation steps were created from a single method call! All validation parameters are shared across all generated steps, including thresholds and briefs:

``` python
(
    pb.Validate(data=pb.load_dataset("game_revenue"))
    .col_vals_gt(
        columns=["item_revenue", "session_duration"],
        value=0,
        thresholds=(0.1, 0.2, 0.3),
        brief="`{col}` must be greater than zero."
    )
    .interrogate()
)
```

[TABLE]

In this example, you can see that the validation report displays customized briefs for each column ("`item_revenue` must be greater than zero." and "`session_duration` must be greater than zero."), automatically substituting the column name using the `{col}` placeholder in the brief template. This feature is particularly helpful when reviewing reports, as it provides clear, human-readable descriptions of what each validation step is checking. When working with multiple columns through a single validation call, these dynamically generated briefs make your validation reports more understandable for both technical and non-technical stakeholders.

### Using Pointblank's Column Selectors

For more advanced column selection, Pointblank provides selector functions that resolve columns based on:

- text patterns in column names
- column position
- column data type

Two common selectors, [starts_with()](../../reference/starts_with.md#pointblank.starts_with) and [ends_with()](../../reference/ends_with.md#pointblank.ends_with), resolve columns based on text patterns in column names.

The `game_revenue` dataset has three columns starting with "item": `item_type`, `item_name`, and `item_revenue`. Let's check that these columns contain no missing values:

``` python
(
    pb.Validate(data=pb.load_dataset("game_revenue"))
    .col_vals_not_null(columns=pb.starts_with("item"))
    .interrogate()
)
```

[TABLE]

Three validation steps were automatically created because three columns matched the pattern.

The complete list of column selectors includes:

- [starts_with()](../../reference/starts_with.md#pointblank.starts_with)
- [ends_with()](../../reference/ends_with.md#pointblank.ends_with)
- [contains()](../../reference/contains.md#pointblank.contains)
- `matches()`
- [everything()](../../reference/everything.md#pointblank.everything)
- [first_n()](../../reference/first_n.md#pointblank.first_n)
- [last_n()](../../reference/last_n.md#pointblank.last_n)

### Combining Column Selectors

Column selectors can be combined for more powerful selection. To do this, use the [col()](../../reference/col.md#pointblank.col) helper function with logical operators:

- `&` (*and*)
- `|` (*or*)
- `-` (*difference*)
- `~` (*not*)

For example, to select all columns except the first four:

``` python
col_selection = pb.col(pb.everything() - pb.first_n(4))

(
    pb.Validate(data=pb.load_dataset("game_revenue"))
    .col_vals_not_null(
        columns=col_selection,
        thresholds=(1, 0.05, 0.1)
    )
    .interrogate()
)
```

[TABLE]

This selects every column except the first four, resulting in seven validation steps.

### Narwhals Selectors

Pointblank also supports column selectors from the [Narwhals](https://narwhals-dev.github.io/narwhals/) library, which include:

- `matches()`
- `by_dtype()`
- `boolean()`
- `categorical()`
- `datetime()`
- `numeric()`
- `string()`

Here's an example selecting all numeric columns:

``` python
import narwhals.selectors as ncs

(
    pb.Validate(data=pb.load_dataset("game_revenue"))
    .col_vals_gt(
        columns=ncs.numeric(),
        value=0
    )
    .interrogate()
)
```

[TABLE]

And selecting all string columns matching "item\_":

``` python
(
    pb.Validate(data=pb.load_dataset("game_revenue"))
    .col_vals_not_null(columns=pb.col(ncs.string() & ncs.matches("item_")))
    .interrogate()
)
```

[TABLE]

This example demonstrates the power of combining Narwhals selectors with logical operators. By using `ncs.string()` to select string columns and then filtering with `ncs.matches("item_")`, we can precisely target text columns with specific naming patterns. This type of targeted selection is particularly valuable when working with wide datasets that have consistent column naming conventions, allowing you to apply appropriate validation rules to logically grouped columns without explicitly listing each one.

### Caveats for Using Column Selectors

While column selectors are powerful, there are some caveats. If a selector doesn't match any columns, the validation won't fail but will show an 'explosion' in the report:

``` python
(
    pb.Validate(data=pb.load_dataset("game_revenue"))
    .col_vals_not_null(columns=pb.starts_with("items"))
    .col_vals_gt(columns="item_revenue", value=0)
    .interrogate()
)
```

[TABLE]

Notice that although there was a problem with Step 1 (that should be addressed), the interrogation did move on to Step 2 without complication.

To mitigate uncertainty, include validation steps that check for the existence of key columns with [`Validate.col_exists()`](../../reference/Validate.col_exists.md#pointblank.Validate.col_exists) or verify the schema with [`Validate.col_schema_match()`](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match).

## Part 2: Comparing Values Between Columns

Sometimes you need to compare values across different columns rather than against fixed values. Pointblank enables this through the [col()](../../reference/col.md#pointblank.col) helper function.

Let's look at examples using the `small_table` dataset:

| PolarsRows13Columns8 |  |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|----|
|  | date_time *Datetime* | date *Date* | a *Int64* | b *String* | c *Int64* | d *Float64* | e *Boolean* | f *String* |
| 1 | 2016-01-04 11:00:00 | 2016-01-04 | 2 | 1-bcd-345 | 3 | 3423.29 | True | high |
| 2 | 2016-01-04 00:32:00 | 2016-01-04 | 3 | 5-egh-163 | 8 | 9999.99 | True | low |
| 3 | 2016-01-05 13:32:00 | 2016-01-05 | 6 | 8-kdg-938 | 3 | 2343.23 | True | high |
| 4 | 2016-01-06 17:23:00 | 2016-01-06 | 2 | 5-jdo-903 | None | 3892.4 | False | mid |
| 5 | 2016-01-09 12:36:00 | 2016-01-09 | 8 | 3-ldm-038 | 7 | 283.94 | True | low |
| 6 | 2016-01-11 06:15:00 | 2016-01-11 | 4 | 2-dhe-923 | 4 | 3291.03 | True | mid |
| 7 | 2016-01-15 18:46:00 | 2016-01-15 | 7 | 1-knw-093 | 3 | 843.34 | True | high |
| 8 | 2016-01-17 11:27:00 | 2016-01-17 | 4 | 5-boe-639 | 2 | 1035.64 | False | low |
| 9 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 10 | 2016-01-20 04:30:00 | 2016-01-20 | 3 | 5-bce-642 | 9 | 837.93 | False | high |
| 11 | 2016-01-26 20:07:00 | 2016-01-26 | 4 | 2-dmx-010 | 7 | 833.98 | True | low |
| 12 | 2016-01-28 02:51:00 | 2016-01-28 | 2 | 7-dmx-010 | 8 | 108.34 | False | low |
| 13 | 2016-01-30 11:23:00 | 2016-01-30 | 1 | 3-dka-303 | None | 2230.09 | True | high |

### Using [col()](../../reference/col.md#pointblank.col) to Specify a Comparison Column

While we typically use validation methods to compare column values against fixed values:

``` python
...
.col_vals_gt(columns="a", value=2, ...)
...
```

We can also compare values between columns by using [col()](../../reference/col.md#pointblank.col) in the `value=` parameter:

``` python
...
.col_vals_gt(columns="a", value=pb.col("x"), ...)
...
```

This checks that each value in column `a` is greater than the corresponding value in column `x`. Here's a concrete example:

``` python
(
    pb.Validate(data=pb.load_dataset("small_table"))
    .col_vals_gt(
        columns="d",
        value=pb.col("c")
    )
    .interrogate()
)
```

[TABLE]

Notice that the validation report shows both column names (`d` and `c`). There are two failing test units because of missing values in column `c`. When comparing across columns, missing values in either column can cause failures.

To handle missing values, use `na_pass=True`:

``` python
(
    pb.Validate(data=pb.load_dataset("small_table"))
    .col_vals_gt(
        columns="d",
        value=pb.col("c"),
        na_pass=True
    )
    .interrogate()
)
```

[TABLE]

Now all tests pass.

The following validation methods accept a [col()](../../reference/col.md#pointblank.col) expression in their `value=` parameter:

- [`Validate.col_vals_gt()`](../../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt)
- [`Validate.col_vals_lt()`](../../reference/Validate.col_vals_lt.md#pointblank.Validate.col_vals_lt)
- [`Validate.col_vals_ge()`](../../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge)
- [`Validate.col_vals_le()`](../../reference/Validate.col_vals_le.md#pointblank.Validate.col_vals_le)
- [`Validate.col_vals_eq()`](../../reference/Validate.col_vals_eq.md#pointblank.Validate.col_vals_eq)
- [`Validate.col_vals_ne()`](../../reference/Validate.col_vals_ne.md#pointblank.Validate.col_vals_ne)

### Using [col()](../../reference/col.md#pointblank.col) in Range Checks

For range validations via [`Validate.col_vals_between()`](../../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between) and [`Validate.col_vals_outside()`](../../reference/Validate.col_vals_outside.md#pointblank.Validate.col_vals_outside) you can use a mix of column references and fixed values:

``` python
(
    pb.Validate(data=pb.load_dataset("small_table"))
    .col_vals_between(
        columns="d",
        left=pb.col("c"),
        right=10_000,
        na_pass=True
    )
    .interrogate()
)
```

[TABLE]

The validation report shows the range as `[c, 10000]`, indicating that the lower bound comes from column `c` while the upper bound is fixed at `10000`.

## Advanced Examples: Combining Both Approaches

The true power comes from combining both approaches: validating multiple columns and using cross-column comparisons:

``` python
validation = (
    pb.Validate(data=pb.load_dataset("small_table"))
    .col_vals_gt(
        columns=["c", "d"],
        value=pb.col("a"),
        na_pass=True
    )
    .interrogate()
)

validation
```

[TABLE]

This creates validation steps checking that values in both columns `d` and `e` are greater than their corresponding values in column `a`.

## Conclusion

Pointblank provides flexible approaches to working with columns:

1.  Column selection: validate multiple columns with a single validation rule
2.  Cross-column comparison: compare values between columns

These capabilities allow you to:

- write more concise validation code
- apply consistent validation rules across similar columns
- create dynamic validations that check relationships between columns
- build comprehensive data quality checks with minimal code

By getting familiar with these techniques, you can create more elegant and powerful validation plans while also reducing repetition in your code.
