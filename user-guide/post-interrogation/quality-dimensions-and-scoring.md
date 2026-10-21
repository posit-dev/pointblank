# Quality Dimensions & Scoring

Beyond knowing *which* validation steps passed or failed, it's often useful to understand data quality along broad, well-understood **dimensions** (completeness, validity, uniqueness, and so on) and to roll everything up into a single **health score** that a governance stakeholder can track over time.

Pointblank does this automatically. Every validation step is tagged with a data quality dimension (inferred from what the step checks), and after interrogation you can obtain per-dimension scores and an overall health score. Nothing about how checks *run* changes; this is a labeling and aggregation layer over results you already have.

## The Six Dimensions

Each validation step belongs to one of six data quality dimensions:

| Dimension | What it measures | Example methods |
|----|----|----|
| **Completeness** | Presence of required values | [col_vals_not_null()](../../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null), [col_pct_null()](../../reference/Validate.col_pct_null.md#pointblank.Validate.col_pct_null), [rows_complete()](../../reference/Validate.rows_complete.md#pointblank.Validate.rows_complete) |
| **Validity** | Values conform to rules, ranges, formats, or schema | [col_vals_gt()](../../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt), [col_vals_regex()](../../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex), [col_vals_in_set()](../../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set), [col_schema_match()](../../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match) |
| **Uniqueness** | Absence of duplicate rows | [rows_distinct()](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) |
| **Consistency** | Internal agreement across columns, rows, or tables | [conjointly()](../../reference/Validate.conjointly.md#pointblank.Validate.conjointly), [col_missing_consistent()](../../reference/Validate.col_missing_consistent.md#pointblank.Validate.col_missing_consistent), [col_vals_in_table()](../../reference/Validate.col_vals_in_table.md#pointblank.Validate.col_vals_in_table), [tbl_match()](../../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match) |
| **Timeliness** | Data recency / freshness | [data_freshness()](../../reference/Validate.data_freshness.md#pointblank.Validate.data_freshness) |
| **Volume** | Expected row and column counts | [row_count_match()](../../reference/Validate.row_count_match.md#pointblank.Validate.row_count_match), [col_count_match()](../../reference/Validate.col_count_match.md#pointblank.Validate.col_count_match) |

The dimension for each step is inferred from its assertion type, so existing validations gain dimensions with no changes on your part.

## Dimensions in the Validation Report

The dimension display in the validation report is **opt-in**: pass `incl_dimensions=True` to [`get_tabular_report()`](../../reference/Validate.get_tabular_report.md#pointblank.Validate.get_tabular_report) (or set it globally with `pb.config(report_incl_dimensions=True)`). Consider a validation of some sales data that touches several dimensions:

``` python
import pointblank as pb
import polars as pl

sales = pl.DataFrame({
    "order_id": [1, 2, 3, 4, 5, 6, 7, 8, 8, 10],   # one duplicate (8)
    "amount":   [120.0, -5.0, 47.5, 0.0, 30.0, 155.0, 175.0, 95.0, 205.0, None],
    "email":    ["a@ex.com", "invalid", "c@ex.com", "d@ex.com", "nope",
                 "f@ex.com", "g@ex.com", "h@ex.com", "i@ex.com", "j@ex.com"],
    "status":   ["paid", "paid", "refund", "paid", "pending",
                 "paid", "paid", "refund", "paid", "paid"],
})

validation = (
    pb.Validate(data=sales, tbl_name="sales", label="Sales data quality")
    .col_vals_not_null(columns="amount")                                # Completeness
    .col_vals_gt(columns="amount", value=0)                             # Validity
    .col_vals_regex(columns="email", pattern=r"^[^@]+@[^@]+\.[^@]+$")   # Validity
    .col_vals_in_set(columns="status", set=["paid", "refund", "pending"])  # Validity
    .rows_distinct(columns_subset=["order_id"])                         # Uniqueness
    .row_count_match(count=10)                                          # Volume
    .interrogate()
)

validation.get_tabular_report(incl_dimensions=True)
```

[TABLE]

Two things in the report now relate to dimensions:

- **A dimension badge on each step number.** Each step's number carries a small, color-coded two-letter badge in its top-left corner (`CM` completeness, `VA` validity, `UQ` uniqueness, `CS` consistency, `TM` timeliness, `VO` volume). Hover over a badge to see the full dimension name. The badge is compact by design, so it doesn't widen the report.
- **A health-score summary in the footer.** Below the table you'll find the overall **Health Score** followed by a per-dimension breakdown, with each dimension's color reinforcing the badges above.

The scores themselves (below) are always available programmatically, whether or not the display is enabled.

## Overriding a Step's Dimension

Automatic inference covers the common cases, but you can set a dimension explicitly with the `dimension=` parameter on any validation method. This is useful for multi-faceted checks. For example, treating a particular range check as a *consistency* rule rather than plain validity:

``` python
validation_override = (
    pb.Validate(data=sales, tbl_name="sales")
    .col_vals_gt(columns="amount", value=0, dimension="consistency")
    .interrogate()
)

validation_override.get_dimension_scores()
```

    {'consistency': 70.0}

You can also remap dimensions globally (for every step of a given type) with `config()`:

``` python
pb.config(dimension_map={"col_vals_gt": "consistency"})

# Now `col_vals_gt` steps are categorized as "consistency" everywhere
remapped = (
    pb.Validate(data=sales)
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)
print(remapped.validation_info[0].dimension)

pb.config()  # reset back to defaults
```

    consistency

    PointblankConfig(report_incl_header=True, report_incl_footer=True, report_incl_footer_timings=True, report_incl_footer_notes=True, report_incl_dimensions=False, preview_incl_header=True, dimension_map=None, dimension_weights=None, dimension_thresholds=None)

An explicit `dimension=` on a step always takes precedence over the global map.

## Health Scores

After interrogation, two methods surface the scores:

``` python
validation.get_dimension_scores()
```

    {'completeness': 90.0, 'validity': 83.33, 'uniqueness': 80.0, 'volume': 100.0}

``` python
validation.get_health_score()
```

    84.31

Scores are **test-unit weighted**: a dimension's score is the total number of passing test units divided by the total number of test units across its steps, expressed as a percentage. The overall health score is the same calculation across every step. Because it's weighted by test units (not by step count), the score reflects data volume. A failing check over a large table pulls the score down more than one over a small table.

> **Note**
>
> Only steps that produced a pass/fail result contribute to scoring. Steps that haven't been interrogated, inactive steps (`active=False`), and steps that could not be evaluated (for example, a check that references a nonexistent column) are all excluded, so a broken check doesn't distort the score.

### Weighting Dimensions

Some organizations consider certain dimensions more critical than others. Provide per-dimension weights with `config(dimension_weights=...)` to scale each dimension's contribution to the overall score (a dimension not listed keeps a weight of `1.0`):

``` python
pb.config(dimension_weights={"completeness": 3.0})

validation_weighted = (
    pb.Validate(data=sales)
    .col_vals_not_null(columns="amount")   # Completeness, weighted 3x
    .col_vals_gt(columns="amount", value=0)  # Validity
    .interrogate()
)
print(validation_weighted.get_health_score())

pb.config()  # reset back to defaults
```

    85.0

    PointblankConfig(report_incl_header=True, report_incl_footer=True, report_incl_footer_timings=True, report_incl_footer_notes=True, report_incl_dimensions=False, preview_incl_header=True, dimension_map=None, dimension_weights=None, dimension_thresholds=None)

## The Scorecard

For a compact, standalone summary (well suited to dashboards or an executive overview) use [`get_scorecard()`](../../reference/Validate.get_scorecard.md#pointblank.Validate.get_scorecard). It shows the overall health score prominently along with a per-dimension breakdown (a color-coded bar, the score, and passing/total test units):

``` python
validation.get_scorecard()
```

| Dimension Scores -- `sales` |        |         |
|----------------------------|--------|---------|
| Health Score: 84%          |        |         |
| DIMENSION                  | SCORE  | UNITS   |
| Completeness               | 90%    | 9 / 10  |
| Validity                   | 83.33% | 25 / 30 |
| Uniqueness                 | 80%    | 8 / 10  |
| Volume                     | 100%   | 1 / 1   |

The scorecard is a [Great Tables](https://github.com/posit-dev/great-tables) object, so you can display it directly, export it to HTML with `.as_raw_html()`, or save it to an image file with `.gtsave()`.

## Enforcing Minimum Scores

In automated pipelines and CI you often want to *fail the run* when a dimension slips below an acceptable level. Call [`assert_dimension_scores()`](../../reference/Validate.assert_dimension_scores.md#pointblank.Validate.assert_dimension_scores) with per-dimension minimums; it raises an `AssertionError` if any dimension is below its minimum (here, uniqueness is `80%`):

``` python
try:
    validation.assert_dimension_scores(thresholds={"uniqueness": 95})
except AssertionError as e:
    print(e)
```

    Dimension health score(s) below the required minimum: uniqueness (80% < 95%)

Dimensions present in the thresholds but absent from the validation are ignored, and a call where every dimension meets its minimum simply returns without raising. You can also set the minimums globally with `config(dimension_thresholds=...)`, and then call the method without arguments:

``` python
pb.config(dimension_thresholds={"completeness": 95})

try:
    validation.assert_dimension_scores()
except AssertionError as e:
    print(e)

pb.config()  # reset back to defaults
```

    Dimension health score(s) below the required minimum: completeness (90% < 95%)

    PointblankConfig(report_incl_header=True, report_incl_footer=True, report_incl_footer_timings=True, report_incl_footer_notes=True, report_incl_dimensions=False, preview_incl_header=True, dimension_map=None, dimension_weights=None, dimension_thresholds=None)

## Accessing Scores Programmatically

The scores are also included in the summary available to [`FinalActions`](../../reference/FinalActions.md#pointblank.FinalActions) via [`get_validation_summary()`](../../reference/get_validation_summary.md#pointblank.get_validation_summary), under the `dimension_scores` and `overall_health_score` keys. This makes it easy to log or trend the health score over time:

``` python
def log_health():
    summary = pb.get_validation_summary()
    print(f"Overall health score: {summary['overall_health_score']}")
    print(f"By dimension: {summary['dimension_scores']}")

(
    pb.Validate(data=sales, final_actions=pb.FinalActions(log_health))
    .col_vals_not_null(columns="amount")
    .col_vals_gt(columns="amount", value=0)
    .interrogate()
)
```

    Overall health score: 80.0
    By dimension: {'completeness': 90.0, 'validity': 70.0}

[TABLE]

## Localized Dimensions

Like the rest of the validation report, dimension labels and the health-score summary are localized. When you set a reporting language (e.g., `Validate(..., lang="fr")`), the dimension names in badge tooltips, the footer summary, and the scorecard are translated automatically. The two-letter badge codes stay language-neutral (matching the report's other short codes such as `TBL`, `EVAL`, and `W`/`E`/`C`), with the full localized name available on hover.
