# Step selection guide

Decision tree for choosing the right validation method.

## By data type

### Numeric columns

| Concern                     | Method                                  |
|-----------------------------|-----------------------------------------|
| Positive values only        | `col_vals_gt(columns, value=0)`         |
| Within a range              | `col_vals_between(columns, left, right)`|
| Not zero                    | `col_vals_ne(columns, value=0)`         |
| Sum matches expected        | `col_sum_eq(columns, value=total)`      |
| Average within tolerance    | `col_avg_eq(columns, value, tol=0.5)`   |
| Standard deviation bounded  | `col_sd_lt(columns, value=max_sd)`      |
| Monotonically increasing    | `col_vals_increasing(columns)`          |
| Null share near a target    | `col_pct_null(columns, p=0.05, tol=0.05)`|

### String columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| Matches a pattern           | `col_vals_regex(columns, pattern)`        |
| Valid email/URL/phone/etc.  | `col_vals_within_spec(columns, spec)`     |
| Length within bounds        | `col_vals_str_len(columns, min_val, max_val)` |
| From a known set            | `col_vals_in_set(columns, set)`           |
| Exists in a lookup table    | `col_vals_in_table(columns, ref_table, ref_column)` |
| Not a forbidden value       | `col_vals_not_in_set(columns, set)`       |
| Not null                    | `col_vals_not_null(columns)`              |

### Date/datetime columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| After a cutoff date         | `col_vals_gt(columns, value=dt.date(2024, 1, 1))` |
| Within a date range         | `col_vals_between(columns, left="2024-01-01", right="2024-12-31")` |
| Data is recent              | `data_freshness(column, max_age="2 hours")` |
| Chronologically ordered     | `col_vals_increasing(columns)`            |

### Boolean columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| All true                    | `col_vals_expr(expr=pl.col("flag"))`      |
| All false                   | `col_vals_expr(expr=~pl.col("flag"))`     |

The comparison and set methods (`col_vals_eq`, `col_vals_in_set`, ...)
don't accept boolean columns; use `col_vals_expr()` as above.

### Text columns (corpora, chat data)

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| Gopher quality filters      | `add_steps(pb.text.heuristic_filters("text"))` |
| In the expected language    | `add_steps(pb.text.in_language("text", expected="en"))` |
| Length in model tokens      | `add_steps(pb.text.token_length("text", tokenizer=..., max_val=4096))` |
| No emails/phones/card nums  | `add_steps(pb.text.no_pii("text"))`       |
| No near duplicates          | `rows_near_distinct(columns="text")`      |
| Model/function score bound  | `col_vals_scored(columns, scorer, min_val)` |
| Chat messages well formed   | `col_vals_chat_format(columns="messages")` |
| No benchmark overlap        | `rows_not_contaminated(columns, reference)` |

See the `check-training-data` skill for details.

## By concern type

### Completeness

```python
.col_vals_not_null(columns="required_field")
.rows_complete()                          # no nulls in any column
.rows_complete(columns_subset=["a", "b"]) # no nulls in a, b
.col_pct_null(columns="optional", p=0.10, tol=0.10) # 0-20% null
```

### Uniqueness

```python
.rows_distinct()                             # all rows unique
.rows_distinct(columns_subset=["id"])        # id column unique
.rows_distinct(columns_subset=["a", "b"])    # composite unique
```

### Consistency (cross-column)

```python
import polars as pl

# Every row must satisfy all the conditions
.conjointly(
    lambda df: pl.col("start") < pl.col("end"),
    lambda df: pl.col("duration") > 0,
)

# A single cross-column expression
.col_vals_expr(expr=pl.col("start") < pl.col("end"))

# Compare against another column
.col_vals_gt(columns="end", value=pb.col("start"))
```

Use `pb.expr_col("start") < pb.expr_col("end")` inside the
`conjointly()` lambdas for a backend-agnostic form.

### Referential (against another table)

```python
pb.Validate(data=current, reference=previous)
.col_sum_eq(columns="total")       # sums match reference
.col_avg_eq(columns="avg_price")   # averages match reference
.tbl_match(tbl_compare=expected)   # tables are identical
.tbl_match(tbl_compare=expected, keys="id", tolerance=0.01)  # by key
.tbl_reconcile(tbl_compare=source, keys="region", measures="amount")
.col_vals_in_table(columns="customer_id", ref_table=customers,
                   ref_column="id")  # foreign key
```

## Available spec values for col_vals_within_spec

Use these with `.col_vals_within_spec(columns, spec="...")`:

- `"email"` -- email addresses
- `"url"` -- URLs
- `"phone"` -- phone numbers
- `"ipv4"`, `"ipv6"` -- IP addresses
- `"mac"` -- MAC addresses
- `"isbn"` -- 10- and 13-digit ISBNs
- `"vin"` -- vehicle identification numbers
- `"credit_card"` -- card numbers (Luhn check)
- `"swift"` -- SWIFT/BIC codes
- `"postal_code[<CC>]"` -- postal codes for a country, e.g.
  `"postal_code[US]"`, `"postal_code[CA]"` (a country is required;
  `"zip"` on its own raises an error)
- `"iban[<CC>]"` -- IBANs for a country, e.g. `"iban[DE]"`

## MissingSpec for structured missingness

When sentinel values represent missing data:

```python
spec = pb.MissingSpec(
    reasons={
        -999: "not collected",
        -1: "redacted",
        "N/A": "not applicable",
    },
    categories={
        "system": ["not collected"],
        "policy": ["redacted", "not applicable"],
    },
    null_is_missing=True,
    null_reason="unknown",
    description="Clinical trial data missingness codes",
)
```

Use with `missing=spec` in value checks, or standalone:

```python
.col_pct_missing(columns="lab_value", missing=spec, max_pct=0.10,
                 reason="not collected", category="system")
```
