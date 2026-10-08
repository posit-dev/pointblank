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
| All rows complete (no nulls)      | `rows_complete`                      |
| Values above/below a bound        | `col_vals_gt/lt/ge/le`               |
| Values in a numeric range         | `col_vals_between`                   |
| Values outside a range            | `col_vals_outside`                   |
| Values from an allowed set        | `col_vals_in_set`                    |
| Values not in a forbidden set     | `col_vals_not_in_set`                |
| Values match a pattern            | `col_vals_regex`                     |
| Values match a format (email etc) | `col_vals_within_spec`               |
| Values are monotonically ordered  | `col_vals_increasing/decreasing`     |
| Null percentage within budget     | `col_pct_null`                       |
| Structured missingness            | `col_pct_missing` with `MissingSpec` |
| Aggregate comparison (sum, avg)   | `col_sum_eq`, `col_avg_gt`, etc.     |
| Data is recent enough             | `data_freshness`                     |
| Table matches another table       | `tbl_match`                          |
| Multiple conditions per row       | `conjointly`                         |
| Custom logic                      | `specially`                          |
| LLM-based semantic check          | `prompt`                             |

## Core concepts

### Ordering your steps

Organize validation steps from structural to semantic:

1. **Schema checks** -- `col_exists`, `col_schema_match`,
   `col_count_match`, `row_count_match`
2. **Completeness checks** -- `col_vals_not_null`, `rows_complete`,
   `col_pct_null`
3. **Uniqueness checks** -- `rows_distinct`
4. **Value-range checks** -- `col_vals_gt`, `col_vals_between`, etc.
5. **Format checks** -- `col_vals_regex`, `col_vals_within_spec`
6. **Set membership** -- `col_vals_in_set`, `col_vals_not_in_set`
7. **Cross-column checks** -- `conjointly`, `col_vals_expr`
8. **Aggregate checks** -- `col_sum_eq`, `col_avg_gt`
9. **Freshness checks** -- `data_freshness`
10. **Custom/semantic checks** -- `specially`, `prompt`

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

Break validation into groups to see which segments fail:

```python
.col_vals_gt(
    columns="revenue",
    value=0,
    segments=pb.seg_group(["region_a", "region_b", "region_c"]),
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

Template variables: `{type}`, `{level}`, `{step}`, `{col}`,
`{val}`, `{time}`.

### Final actions

Run after all steps complete, with access to the full summary:

```python
def check_overall(summary=None):
    summary = pb.get_validation_summary()
    if summary and summary["n_failed_steps"] > 0:
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
5. **`conjointly` takes lambdas, not method calls.** Each argument is
   `lambda v: v.col_vals_gt(...)`.
6. **Aggregate methods compare one value per column**, not per row.
   They produce a single pass/fail per column.
7. **`segments` splits the check into sub-groups.** Each segment is
   reported separately in the validation report.
