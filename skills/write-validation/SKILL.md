---
name: write-validation
description: >
  Write data-validation plans with Pointblank. Covers choosing the
  right validation methods for each data quality concern, composing
  multi-step plans, setting thresholds and actions, using segments,
  conditional steps, handling nulls and missing values, and extracting
  results. Use when building or improving a validation workflow.
license: MIT
compatibility: Requires Python >=3.10, pointblank installed.
metadata:
  author: rich-iannone
  version: "1.0"
  tags:
    - data-validation
    - data-quality
    - validation-plan
    - thresholds
    - actions
---

# Write Validation

Skill for composing effective data-validation plans with Pointblank.
A validation plan is a sequence of steps that check different aspects
of your data, organized to catch issues early and report clearly.

## Quick start

```python
import pointblank as pb

validation = (
    pb.Validate(
        data=df,
        tbl_name="orders",
        label="Order validation",
        thresholds=pb.Thresholds(warning=0.01, error=0.05),
    )
    # Structural checks first
    .col_exists(columns=["order_id", "amount", "status"])
    .row_count_match(count=1000, tol=50)

    # Value checks
    .col_vals_gt(columns="amount", value=0)
    .col_vals_not_null(columns="order_id")
    .col_vals_in_set(columns="status", set=["pending", "shipped", "delivered"])

    # Execute
    .interrogate()
)
```

## Skill directory structure

```
skills/write-validation/
+-- SKILL.md                       <- This file
+-- references/
    +-- step-selection-guide.md    <- Which method for which concern
    +-- thresholds-and-actions.md  <- Configuring failure responses
```

## When to use what

| Data quality concern              | Recommended method                   |
| --------------------------------- | ------------------------------------ |
| Column exists                     | `col_exists`                         |
| Schema matches expectations       | `col_schema_match`                   |
| Expected row count                | `row_count_match`                    |
| Expected column count             | `col_count_match`                    |
| No null values                    | `col_vals_not_null`                  |
| All null (placeholder column)     | `col_vals_null`                      |
| No duplicate rows                 | `rows_distinct`                      |
| No near-duplicate text            | `rows_near_distinct`                 |
| All rows complete (no nulls)      | `rows_complete`                      |
| Values above/below a bound        | `col_vals_gt/lt/ge/le`               |
| Values in a numeric range         | `col_vals_between`                   |
| Values outside a range            | `col_vals_outside`                   |
| Values from an allowed set        | `col_vals_in_set`                    |
| Values found in a lookup table    | `col_vals_in_table`                  |
| Values not in a forbidden set     | `col_vals_not_in_set`                |
| Values match a pattern            | `col_vals_regex`                     |
| String length within bounds       | `col_vals_str_len`                   |
| Values match a format (email etc) | `col_vals_within_spec`               |
| Values are monotonically ordered  | `col_vals_increasing/decreasing`     |
| Null percentage within budget     | `col_pct_null`                       |
| Structured missingness            | `col_pct_missing`, `col_missing_coded`, `col_missing_only_coded`, `col_missing_consistent` with `MissingSpec` |
| Aggregate comparison (sum, avg)   | `col_sum_eq`, `col_avg_gt`, etc.     |
| Data is recent enough             | `data_freshness`                     |
| Table matches another table       | `tbl_match` (`keys=`, `tolerance=`)  |
| Totals agree with a source table  | `tbl_reconcile`                      |
| Multiple conditions per row       | `conjointly`                         |
| Custom logic                      | `specially`                          |
| LLM-based semantic check          | `prompt`                             |
| Score from a model or function    | `col_vals_scored`                    |
| Chat data well formed             | `col_vals_chat_format`               |
| No overlap with a benchmark       | `rows_not_contaminated`              |

For text corpora, RAG data, and model training sets, see the
`check-training-data` skill.

## Core concepts

### Ordering your steps

Organize validation steps from structural to semantic:

1. **Schema checks** -- `col_exists`, `col_schema_match`,
   `col_count_match`, `row_count_match`
2. **Completeness checks** -- `col_vals_not_null`, `rows_complete`,
   `col_pct_null`, `col_missing_*`
3. **Uniqueness checks** -- `rows_distinct`, `rows_near_distinct`
4. **Value-range checks** -- `col_vals_gt`, `col_vals_between`, etc.
5. **Format checks** -- `col_vals_regex`, `col_vals_within_spec`,
   `col_vals_str_len`, `col_vals_chat_format`
6. **Set membership** -- `col_vals_in_set`, `col_vals_not_in_set`,
   `col_vals_in_table`
7. **Cross-column checks** -- `conjointly`, `col_vals_expr`
8. **Aggregate checks** -- `col_sum_eq`, `col_avg_gt`, `tbl_reconcile`
9. **Freshness checks** -- `data_freshness`
10. **Custom/semantic checks** -- `specially`, `prompt`,
    `col_vals_scored`, `rows_not_contaminated`

This order means structural problems surface first before
value-level checks run.

### Handling nulls

By default, null values count as failures in value checks. Control
this per step:

```python
# Nulls count as failures (default)
.col_vals_gt(columns="amount", value=0)

# Nulls are treated as passing
.col_vals_gt(columns="amount", value=0, na_pass=True)
```

For structured missingness (sentinel values like -999, "N/A"):

```python
missing_spec = pb.MissingSpec(
    reasons={-999: "not collected", -1: "redacted"},
    null_is_missing=True,
    null_reason="unknown",
)

.col_vals_gt(columns="measurement", value=0, missing=missing_spec)
.col_pct_missing(columns="measurement", missing=missing_spec, max_pct=0.10)
```

### Segmented validation

Break validation into groups to see which segments fail. Each
segment becomes its own step in the report:

```python
# One step per distinct value of "region"
.col_vals_gt(columns="revenue", value=0, segments="region")

# Only some values
.col_vals_gt(columns="revenue", value=0, segments=("region", ["north", "south"]))

# Combine values into groups (seg_group goes inside the tuple)
.col_vals_gt(
    columns="revenue",
    value=0,
    segments=("region", pb.seg_group([["north", "south"], ["east", "west"]])),
)
```

### Conditional steps

Skip steps dynamically based on the data:

```python
# Only run if the column exists
.col_vals_gt(
    columns="new_feature",
    value=0,
    active=pb.has_columns("new_feature"),
)

# Only run if the table has enough rows
.rows_distinct(active=pb.has_rows(min=100))
```

### Pre-processing data

Transform data before a check with `pre`:

```python
.col_vals_gt(
    columns="price",
    value=0,
    pre=lambda df: df.filter(pl.col("status") == "active"),
)
```

Steps with `pre=` are left out of `get_sundered_data()` and
`get_row_flags()`, since their rows can't be matched to the target
table. Wrap a function that only filters, reorders, or changes values
in `pb.row_preserving(...)` to keep the step in.

### Reusable bundles of steps

`pb.Steps` records steps (with the same methods as `Validate`) that
can be added to any validation with `add_steps()`:

```python
id_checks = (
    pb.Steps()
    .col_vals_not_null(columns="id")
    .rows_distinct(columns_subset="id")
)

(
    pb.Validate(data=df)
    .add_steps(id_checks)
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)
```

`add_steps(*steps, active=None, thresholds=None, exclude=None,
columns_map=None)` can override the bundle's thresholds, skip steps,
or rename columns. The `pb.text` and `pb.rag` functions return `Steps`
bundles too.

### Thresholds

Set thresholds at three severity levels:

```python
# Global thresholds (apply to all steps)
pb.Validate(
    data=df,
    thresholds=pb.Thresholds(warning=0.01, error=0.05, critical=0.25),
)

# Per-step override
.col_vals_gt(
    columns="amount",
    value=0,
    thresholds=pb.Thresholds(warning=5, error=20),
)
```

- Values `< 1`: fraction of failing test units (e.g., `0.05` = 5%)
- Values `>= 1`: absolute count of failures (e.g., `5` = 5 rows)
- `True`: any failure triggers (equivalent to `1`)

### Actions

Trigger responses when thresholds are exceeded:

```python
pb.Actions(
    warning="Warning: {col} step {step} at {time}",
    error=lambda: send_alert("Data quality error"),
    critical=[
        pb.send_slack_notification(webhook_url="https://..."),
        "Critical failure in {col}",
    ],
    highest_only=True,  # only fire the highest triggered level
)
```

Template variables: `{type}`, `{level}`, `{step}` (or `{i}`), `{col}`
(or `{column}`), `{val}` (or `{value}`), `{time}`.

### Final actions

Run after all steps complete, with access to the full summary:

```python
def check_overall():
    summary = pb.get_validation_summary()
    if summary and summary["n_failing_steps"] > 0:
        send_report(summary)

pb.Validate(
    data=df,
    final_actions=pb.FinalActions(check_overall),
)
```

### Comparing against reference data

Track drift by comparing current data to a reference table:

```python
validation = (
    pb.Validate(data=current_df, reference=previous_df)
    .col_sum_eq(columns="revenue")    # value=None -> uses ref()
    .col_avg_eq(columns="quantity")
    .interrogate()
)
```

### Extracting results

```python
# Did everything pass?
validation.all_passed()

# Counts and fractions by step
validation.n_passed(i=1, scalar=True)
validation.f_failed(i=[1, 2, 3])

# Get failing rows for a step
extracts = validation.get_data_extracts(i=1, frame=True)

# Split into pass/fail subsets
pass_df = validation.get_sundered_data(type="pass")
fail_df = validation.get_sundered_data(type="fail")

# Every row, flagged with the steps it failed (pb_failed_<i>_ columns)
flags = validation.get_row_flags()

# Machine-readable report
json_report = validation.get_json_report()
```

### Saving and reloading

```python
pb.write_file(validation, filename="daily_check.pb")
restored = pb.read_file("daily_check.pb")
```

## Workflows

### Building a validation plan from scratch

1. Profile the data with `pb.DataScan(data=df)` to understand
   distributions, nulls, and types.
2. Start with structural checks (`col_exists`, `col_schema_match`).
3. Add completeness checks (`col_vals_not_null`, `rows_complete`).
4. Add value constraints based on domain knowledge.
5. Set thresholds appropriate to the use case.
6. Run `interrogate()` and review the report.
7. Iterate: adjust thresholds, add missing checks, remove noisy ones.

### Adding checks to an existing plan

Read the current validation code, identify uncovered columns or
concerns, and add steps in the appropriate position (structural
before semantic). Preserve the existing threshold/action
configuration unless changing it is part of the task.

### Diagnosing validation failures

1. Run `interrogate()` and check the report.
2. Use `get_data_extracts(i=N, frame=True)` to see failing rows.
3. Use `get_step_report(i=N)` for a detailed per-step view.
4. Determine whether the check or the data is wrong.
5. Fix the check (adjust value/threshold) or flag the data issue.

## Gotchas

1. **`.interrogate()` must be called.** Steps are declarative until
   executed.
2. **`na_pass` defaults to `False`.** Nulls fail value checks unless
   you opt in.
3. **Threshold `0.05` vs `5`.** Fractional = percentage, integer =
   count.
4. **`col_vals_between` is closed by default.** Pass
   `inclusive=(False, False)` for an open interval.
5. **`conjointly` takes callables that return expressions.** Each
   argument is like `lambda df: pl.col("a") > 0` (or
   `lambda df: expr_col("a") > 0`), not a validation method call.
6. **Aggregate methods compare one value per column**, not per row.
   They produce a single pass/fail per column, and they take no `pre=`
   or `segments=`.
7. **`segments` splits the check into sub-groups.** Each segment is
   reported separately in the validation report. `pb.seg_group()` is
   only valid inside a `(column, ...)` tuple.
8. **Comparison methods can't check boolean columns.** Use
   `col_vals_expr(expr=pl.col("flag"))` (or `~pl.col("flag")`) instead
   of `col_vals_eq(columns="flag", value=True)`.
