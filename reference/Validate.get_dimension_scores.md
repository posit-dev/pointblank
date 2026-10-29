# Validate.get_dimension_scores()

Get per-dimension health scores from the validation results.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L20236-L20295)

``` python
Validate.get_dimension_scores()
```

Each validation step is associated with a data quality dimension (e.g., `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"`), either inferred automatically from the assertion type or set explicitly via the `dimension=` parameter on a validation method. This method rolls the per-step results up into a test-unit-weighted pass rate (`0`-`100`) for each dimension present in the plan.

Scores are weighted by test units, so larger tables and steps with more test units contribute proportionally more to each dimension's score. Only steps that have been interrogated contribute; inactive steps are excluded.

## Returns

`dict[str, float]`  
A dictionary mapping each dimension name to its score (a percentage from `0` to `100`). Returns an empty dictionary if the validation has not been interrogated.

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

## See Also

[get_health_score()](../reference/Validate.get_health_score.md#pointblank.Validate.get_health_score)  
Get a single overall score across all dimensions.
