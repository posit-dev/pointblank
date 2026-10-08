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
| Average within tolerance    | `col_avg_between(columns, value, tol)`  |
| Standard deviation bounded  | `col_sd_lt(columns, value=max_sd)`      |
| Monotonically increasing    | `col_vals_increasing(columns)`          |
| Null percentage under limit | `col_pct_null(columns, p=0.05)`         |

### String columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| Matches a pattern           | `col_vals_regex(columns, pattern)`        |
| Valid email/URL/phone/etc.  | `col_vals_within_spec(columns, spec)`     |
| From a known set            | `col_vals_in_set(columns, set)`           |
| Not a forbidden value       | `col_vals_not_in_set(columns, set)`       |
| Not null                    | `col_vals_not_null(columns)`              |

### Date/datetime columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| After a cutoff date         | `col_vals_gt(columns, value=cutoff)`      |
| Within a date range         | `col_vals_between(columns, left, right)`  |
| Data is recent              | `data_freshness(column, max_age="2h")`    |
| Chronologically ordered     | `col_vals_increasing(columns)`            |

### Boolean columns

| Concern                     | Method                                    |
|-----------------------------|-------------------------------------------|
| All true                    | `col_vals_eq(columns, value=True)`        |
| All false                   | `col_vals_eq(columns, value=False)`       |

## By concern type

### Completeness

```python
.col_vals_not_null(columns="required_field")
.rows_complete()                          # no nulls in any column
.rows_complete(columns_subset=["a", "b"]) # no nulls in a, b
.col_pct_null(columns="optional", p=0.20) # at most 20% null
```

### Uniqueness

```python
.rows_distinct()                             # all rows unique
.rows_distinct(columns_subset=["id"])        # id column unique
.rows_distinct(columns_subset=["a", "b"])    # composite unique
```

### Consistency (cross-column)

```python
from pointblank import expr_col

.conjointly(
    lambda v: v.col_vals_expr(expr_col("start") < expr_col("end")),
    lambda v: v.col_vals_gt(columns="duration", value=0),
)
```

### Referential (against another table)

```python
pb.Validate(data=current, reference=previous)
.col_sum_eq(columns="total")       # sums match reference
.col_avg_eq(columns="avg_price")   # averages match reference
.tbl_match(tbl_compare=expected)   # tables are identical
```

## Available spec values for col_vals_within_spec

Use these with `.col_vals_within_spec(columns, spec="...")`:

- `"email"` -- valid email addresses
- `"url"` -- valid URLs
- `"ipv4"` -- IPv4 addresses
- `"ipv6"` -- IPv6 addresses
- `"phone"` -- phone numbers (E.164)

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
