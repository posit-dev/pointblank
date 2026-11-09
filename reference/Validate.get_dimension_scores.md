# Validate.get_dimension_scores()

Get per-dimension health scores from the validation results.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L22574-L22683)

``` python
Validate.get_dimension_scores(by=None)
```

Each validation step is associated with a data quality dimension (e.g., `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"`), either inferred automatically from the assertion type or set explicitly via the `dimension=` parameter on a validation method. This method rolls the per-step results up into a test-unit-weighted pass rate (`0`-`100`) for each dimension present in the plan.

Scores are weighted by test units, so larger tables and steps with more test units contribute proportionally more to each dimension's score. Only steps that have been interrogated contribute; inactive steps are excluded.

## Parameters

`by: str | None = None`  
The name of a column that steps are segmented by (with `segments=`), to get the scores of each segment rather than of the whole validation. Only the steps segmented by that column count toward the scores of a segment (a step without segments applies to the whole table, so it's left out), and a segment whose steps have no results (being inactive, or not evaluated) is left out too. It's an error if no step is segmented by the column. See the *Scores by Segment* section for details.

## Returns

`dict`  
A dictionary mapping each dimension name to its score (a percentage from `0` to `100`). Returns an empty dictionary if the validation has not been interrogated. With `by=`, a dictionary mapping each value of the column (in the order of the steps) to such a dictionary for its segment (a group of values made with [`seg_group()`](../reference/seg_group.md#pointblank.seg_group) is a tuple).

## Scores By Segment

A table often holds several kinds of data whose quality differs: the languages of a multilingual corpus, the regions of a sales table, or the sources of a feed. Steps with `segments=` give results for each kind, and `by=` rolls them up into scores for each, which says which part of the data needs work. The same `by=` is in [`get_health_score()`](../reference/Validate.get_health_score.md#pointblank.Validate.get_health_score), [`assert_dimension_scores()`](../reference/Validate.assert_dimension_scores.md#pointblank.Validate.assert_dimension_scores), and [`get_scorecard()`](../reference/Validate.get_scorecard.md#pointblank.Validate.get_scorecard).

## Examples

Let's validate the `small_table` dataset with three steps, each of which falls into a different dimension: a not-null check (completeness), a range check (validity), and a check for duplicate rows (uniqueness).

``` python
import pointblank as pb

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

validation = (
    pb.Validate(data=small_table)
    .col_vals_not_null(columns="c")
    .col_vals_gt(columns="d", value=1000)
    .rows_distinct()
    .interrogate()
)

validation.get_dimension_scores()
```

    {'completeness': 84.62, 'validity': 53.85, 'uniqueness': 84.62}

Each step has 13 test units (one per row). Completeness scores 84.62 because 11 of the 13 values in `c` are present, and uniqueness gets the same score from the 2 duplicated rows. Validity is lowest: only 7 of the 13 values in `d` are above 1000. A breakdown like this points to the kind of problem to fix, which the single number from [`get_health_score()`](../reference/Validate.get_health_score.md#pointblank.Validate.get_health_score) doesn't.

When the steps are segmented, `by=` gives the scores of each segment. Here, the first two steps are segmented by the values of `f`.

``` python
validation = (
    pb.Validate(data=small_table)
    .col_vals_not_null(columns="c", segments="f")
    .col_vals_gt(columns="d", value=1000, segments="f")
    .rows_distinct()
    .interrogate()
)

validation.get_dimension_scores(by="f")
```

    {'high': {'completeness': 83.33, 'validity': 50.0},
     'low': {'completeness': 100.0, 'validity': 40.0},
     'mid': {'completeness': 50.0, 'validity': 100.0}}

The rows with `f` of `"low"` have no missing values in `c`, but only 2 of their 5 values in `d` are above 1000. The [rows_distinct()](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step isn't segmented, so it's not part of the scores of any segment.

## See Also

[get_health_score()](../reference/Validate.get_health_score.md#pointblank.Validate.get_health_score)  
Get a single overall score across all dimensions.

[get_scorecard()](../reference/Validate.get_scorecard.md#pointblank.Validate.get_scorecard)  
Get a scorecard of the scores.
