# Validation methods reference

Complete list of validation methods available on `Validate`.

## Value comparison methods

All share a common signature pattern:

```python
.col_vals_gt(
    columns,              # str, list[str], or column selector
    value,                # numeric value or col("other_column")
    na_pass=False,        # treat nulls as passing?
    missing=None,         # MissingSpec for structured missingness
    pre=None,             # callable to transform data before check
    segments=None,        # segment the check by group values
    thresholds=None,      # per-step thresholds override
    actions=None,         # per-step actions override
    brief=None,           # custom brief text or False to suppress
    active=True,          # bool or callable to conditionally skip
    dimension=None,       # tag for dimensional scoring
)
```

| Method              | Checks that values are...            |
|---------------------|--------------------------------------|
| `col_vals_gt`       | greater than `value`                 |
| `col_vals_lt`       | less than `value`                    |
| `col_vals_ge`       | greater than or equal to `value`     |
| `col_vals_le`       | less than or equal to `value`        |
| `col_vals_eq`       | equal to `value`                     |
| `col_vals_ne`       | not equal to `value`                 |

## Range methods

```python
.col_vals_between(columns, left, right, inclusive=(True, True), ...)
.col_vals_outside(columns, left, right, inclusive=(True, True), ...)
```

`inclusive` controls boundary inclusion: `(True, True)` = closed
interval, `(False, False)` = open interval.

## Set membership

```python
.col_vals_in_set(columns, set=["a", "b", "c"], ...)
.col_vals_not_in_set(columns, set=["x", "y"], ...)
```

## Monotonicity

```python
.col_vals_increasing(columns, allow_stationary=False, decreasing_tol=None, ...)
.col_vals_decreasing(columns, allow_stationary=False, increasing_tol=None, ...)
```

## Null checks

```python
.col_vals_null(columns, ...)      # all values must be null
.col_vals_not_null(columns, ...)  # no values may be null
```

No `na_pass` or `missing` parameters on these methods.

## Pattern and spec matching

```python
.col_vals_regex(columns, pattern="^[A-Z]{3}$", inverse=False, ...)
.col_vals_within_spec(columns, spec="email", ...)
```

## Expression-based

```python
from pointblank import expr_col

.col_vals_expr(expr_col("price") * expr_col("qty") > 0, ...)
```

## Aggregate comparison methods

Pattern: `col_{agg}_{comp}()` where agg is `sum`, `avg`, or `sd`
and comp is `eq`, `gt`, `ge`, `lt`, or `le`.

```python
.col_sum_gt(columns, value=1000, tol=0, ...)
.col_avg_between(columns, value=50.0, tol=0.5, ...)
.col_sd_lt(columns, value=10, ...)
```

When a reference table is set and `value=None`, automatically
compares against the reference column.

## Null percentage

```python
.col_pct_null(columns, p=0.05, tol=0, ...)     # null % <= p
.col_pct_missing(columns, missing=spec, max_pct=0.10, ...)
```

## Structural checks

```python
.col_exists(columns, ...)
.col_schema_match(schema, complete=True, in_order=True, ...)
.col_count_match(count=10, inverse=False, ...)
.row_count_match(count=1000, tol=0, inverse=False, ...)
.rows_distinct(columns_subset=None, ...)
.rows_complete(columns_subset=None, ...)
```

## Data freshness

```python
.data_freshness(column="updated_at", max_age="2h", reference_time=None, ...)
```

`max_age` accepts strings like `"2h"`, `"30m"`, `"1d"`, or
`datetime.timedelta` objects.

## Table comparison

```python
.tbl_match(tbl_compare=other_df, ...)
```

## Compound and custom checks

```python
# Multiple conditions must all hold for each row
.conjointly(
    lambda v: v.col_vals_gt(columns="a", value=0),
    lambda v: v.col_vals_lt(columns="a", value=100),
)

# Fully custom check
.specially(lambda tbl: len(tbl) > 0)
```

## LLM-based validation

```python
.prompt(
    prompt="Check if product names are appropriate",
    model="anthropic:claude-sonnet-4-6",
    columns_subset=["product_name"],
    batch_size=1000,
    max_concurrent=3,
)
```

## Common parameters

| Parameter    | Type                    | Description                              |
|--------------|-------------------------|------------------------------------------|
| `columns`    | `str \| list \| selector` | Target column(s)                       |
| `value`      | `numeric \| col()`      | Comparison value or column reference     |
| `na_pass`    | `bool`                  | Treat nulls as passing (default `False`) |
| `missing`    | `MissingSpec`           | Structured missingness definition        |
| `pre`        | `Callable`              | Transform data before check              |
| `segments`   | `SegmentSpec`           | Segment validation by groups             |
| `thresholds` | `Thresholds`            | Per-step failure thresholds              |
| `actions`    | `Actions`               | Per-step actions on threshold breach     |
| `brief`      | `str \| bool`           | Step description text                    |
| `active`     | `bool \| Callable`      | Conditionally skip step                  |
| `dimension`  | `str`                   | Tag for dimensional scoring              |
