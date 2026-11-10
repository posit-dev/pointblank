# Validation methods reference

Complete list of validation methods available on `Validate`.

## Value comparison methods

All share a common signature pattern:

```python
.col_vals_gt(
    columns,              # str, list[str], or column selector
    value,                # numeric/date value, col("other_column")
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

`value` can be a number, a `datetime.date`/`datetime` (or an ISO
string such as `"2016-01-10"`) for date columns, or `col("other")` to
compare against another column. These methods can't check boolean
columns; use `col_vals_expr(expr=pl.col("flag"))` instead.

## Range methods

```python
.col_vals_between(columns, left, right, inclusive=(True, True), na_pass=False, ...)
.col_vals_outside(columns, left, right, inclusive=(True, True), na_pass=False, ...)
```

`inclusive` controls boundary inclusion: `(True, True)` = closed
interval, `(False, False)` = open interval.

## Set membership

```python
.col_vals_in_set(columns, set=["a", "b", "c"], ...)          # no na_pass
.col_vals_not_in_set(columns, set=["x", "y"], na_pass=False, ...)

# Values must appear in a column of another table (no segments)
.col_vals_in_table(columns, ref_table=lookup_df, ref_column="code", na_pass=False, ...)
```

## Monotonicity

```python
.col_vals_increasing(columns, allow_stationary=False, decreasing_tol=None, order_by=None, ...)
.col_vals_decreasing(columns, allow_stationary=False, increasing_tol=None, order_by=None, ...)
```

## Null checks

```python
.col_vals_null(columns, ...)      # all values must be null
.col_vals_not_null(columns, ...)  # no values may be null
```

No `na_pass` or `missing` parameters on these methods.

## Pattern, spec, and length matching

```python
.col_vals_regex(columns, pattern="^[A-Z]{3}$", na_pass=False, inverse=False, ...)
.col_vals_within_spec(columns, spec="email", na_pass=False, ...)
.col_vals_str_len(columns, min_val=None, max_val=None, na_pass=False, ...)
```

Specs: `"isbn"`, `"vin"`, `"postal_code[<CC>]"` (e.g. `"postal_code[US]"`),
`"credit_card"`, `"iban[<CC>]"`, `"swift"`, `"phone"`, `"email"`,
`"url"`, `"ipv4"`, `"ipv6"`, `"mac"`. A postal code always needs a
country code.

## Expression-based

```python
import polars as pl

.col_vals_expr(expr=pl.col("price") * pl.col("qty") > 0, ...)
.col_vals_expr(expr=lambda df: df["price"] > 0, ...)
```

`expr` is a Polars, Narwhals, or Ibis expression (matching the table
type), or a callable that takes the table. There's no `columns`
parameter.

## Aggregate comparison methods

Pattern: `col_{agg}_{comp}()` where agg is `sum`, `avg`, or `sd` and
comp is `eq`, `gt`, `ge`, `lt`, or `le` (15 methods; there are no
`_between` variants).

```python
.col_sum_gt(columns, value=1000, tol=0, ...)
.col_avg_le(columns, value=50.0, tol=0.5, ...)
.col_sd_lt(columns, value=10, ...)
```

These take only `columns`, `value`, `tol`, `thresholds`, `brief`,
`actions`, `active`, and `dimension` (no `pre`, `segments`, or
`na_pass`). When `Validate(reference=...)` is set and `value=None`,
they compare against the same column of the reference table; use
`value=pb.ref("other_col")` for a differently named column.

## Missing values

```python
.col_pct_null(columns, p=0.05, tol=0.05, ...)  # null share in [p - tol, p + tol]
.col_pct_missing(columns, missing=spec, max_pct=0.10, reason=None, category=None, ...)

# Structured missingness (spec = pb.MissingSpec(reasons={-99: "refused"}))
.col_missing_coded(columns, missing=spec, ...)          # no raw nulls, only codes
.col_missing_only_coded(columns, missing=spec, allowed=None, min_val=None, max_val=None, ...)
.col_missing_consistent(columns=["a", "b"], missing=spec, when_reason="not_asked", ...)
```

`col_pct_null` and `col_pct_missing` have no `pre`/`segments`. Pass
`missing=spec` to the value-comparison methods so coded values aren't
compared as real values.

## Structural checks

```python
.col_exists(columns, ...)
.col_schema_match(schema, complete=True, in_order=True,
                  case_sensitive_colnames=True, case_sensitive_dtypes=True,
                  full_match_dtypes=True, ...)
.col_count_match(count=10, inverse=False, ...)        # count can be a table
.row_count_match(count=1000, tol=0, inverse=False, ...)
.rows_distinct(columns_subset=None, ...)
.rows_complete(columns_subset=None, ...)
```

## Data freshness

```python
.data_freshness(column="updated_at", max_age="24 hours", reference_time=None,
                timezone=None, allow_tz_mismatch=False, allow_future=False, ...)
```

`max_age` accepts strings like `"24 hours"`, `"30 minutes"`, `"2 weeks"`
(short forms like `"2h"` and `"1d"` also work) or `datetime.timedelta`
objects. Data dated later than the reference time fails unless
`allow_future=` allows it (a duration such as `"1m"` for clock skew,
or `True` for any amount).

## Table comparison

```python
# Row-by-row match against another table
.tbl_match(
    tbl_compare=other_df,
    keys=None,            # match rows by key instead of position
    columns=None,         # subset of columns to compare
    column_map=None,      # {"name_in_compare": "name_in_target"}
    tolerance=None,       # number, {"atol": .., "rtol": ..}, or per-column dict
    normalize=None,       # {"name": ["strip", "lower"]}, {"x": [{"round": 2}]}
    null_equal=True,
    schema="strict",      # or "common"
    dup_keys="flag",      # or "compare"
    order_by=None,
    partitions=None,
)

# Totals by key (or grand totals) must agree with a source table
.tbl_reconcile(
    tbl_compare=source_df,
    keys="region",        # None for grand totals
    measures="amount",
    agg="sum",
    tolerance=None,
    column_map={"amount": "amt"},  # name in tbl_compare -> name in target
    partitions=None,
)
```

## Compound and custom checks

```python
import polars as pl
from pointblank import expr_col

# Multiple conditions must all hold for each row; each argument is a
# callable taking the table and returning a boolean expression
.conjointly(
    lambda df: pl.col("a") > 0,
    lambda df: pl.col("a") < 100,
    lambda df: expr_col("a") < expr_col("b"),   # backend-agnostic form
)

# Fully custom check: a callable taking the table (or nothing) and
# returning boolean results
.specially(expr=lambda tbl: len(tbl) > 0)
```

Bare expressions (not wrapped in a callable) don't work in
`conjointly()`.

## Text and ML data checks

```python
# Each row scored by a model or function; passes if the score is in bounds
.col_vals_scored(columns="text", scorer=my_scorer, min_val=None, max_val=None,
                 na_pass=False, cache=True, keep_scores=True, ...)

# Near-duplicate rows (MinHash over word n-grams, or exact after normalizing)
.rows_near_distinct(columns="text", method="minhash", threshold=0.85, ngram=5, num_perm=128, ...)

# Text checks: each adds col_vals_scored() or col_vals_regex() steps,
# one set per column, with a brief for each
.col_vals_text_quality(columns="text", min_words=50, max_words=100_000, ...)  # 8 Gopher filters
.col_vals_in_language(columns="text", expected="en", expected_column=None,
                      min_confidence=0.5, backend="langid", ...)
.col_vals_token_len(columns="text", tokenizer="whitespace", min_val=None, max_val=4096, ...)
.col_vals_no_pii(columns="text", entities=("email", "phone", "credit_card"), ...)

# Chat conversations (list of messages or JSON text) are well formed
.col_vals_chat_format(columns="messages", roles=("system", "user", "assistant"),
                      alternate=True, first_role=None, last_role=None,
                      allow_empty=False, role_key="role", content_key="content",
                      na_pass=False, ...)

# Rows don't overlap with a reference (e.g. benchmark questions)
.rows_not_contaminated(columns="text", reference=benchmark_df,
                       reference_columns="question", method="ngram",
                       ngram=None, threshold=0.8, num_perm=128, ...)
```

`ngram=None` in `rows_not_contaminated()` means 13 words. A scorer
takes a batch (a list of row dicts) and returns a list of numbers or
`None`; see the `check-training-data` skill for the `pb.scorers`
adapters, the text checks, and the `pb.rag` step bundles.

## LLM-based validation

```python
.prompt(
    prompt="Check if product names are appropriate",
    model="anthropic:claude-sonnet-4-6",   # or a list of models
    columns_subset=["product_name"],
    attachments=None,
    batch_size=1000,
    max_concurrent=3,
    aggregate="majority",                  # or "all"/"any", for a list of models
)

# After interrogation: agreement of step 3 with hand labels
validation.calibrate_prompt(i=3, labels="human_label")  # column or list
```

`calibrate_prompt()` returns a dict (agreement share, Cohen's kappa,
disagreement counts, and per-model agreement when `model=` is a list)
and adds it as a note on the step.

## Common parameters

| Parameter    | Type                    | Description                              |
|--------------|-------------------------|------------------------------------------|
| `columns`    | `str \| list \| selector` | Target column(s)                       |
| `value`      | `numeric \| col()`      | Comparison value or column reference     |
| `na_pass`    | `bool`                  | Treat nulls as passing (default `False`) |
| `missing`    | `MissingSpec`           | Structured missingness definition        |
| `pre`        | `Callable`              | Transform data before check              |
| `segments`   | `str \| tuple`          | Segment validation by groups             |
| `thresholds` | `Thresholds`            | Per-step failure thresholds              |
| `actions`    | `Actions`               | Per-step actions on threshold breach     |
| `brief`      | `str \| bool`           | Step description text                    |
| `active`     | `bool \| Callable`      | Conditionally skip step                  |
| `dimension`  | `str`                   | Tag for dimensional scoring              |

`segments="f"` makes one step per distinct value of `f`;
`segments=("f", ["low", "high"])` picks the values; and
`segments=("f", pb.seg_group([["low", "mid"], ["high"]]))` combines
values into groups. Wrap a `pre=` function in `pb.row_preserving()`
when it only filters or changes rows, so the step still counts in
`get_sundered_data()` and `get_row_flags()`.
