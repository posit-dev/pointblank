## Validate.get_dimension_scores()


Get per-dimension health scores from the validation results.


Usage

``` python
Validate.get_dimension_scores()
```


Each validation step is associated with a data quality dimension (e.g., `"completeness"`, `"validity"`, `"uniqueness"`, `"consistency"`, `"timeliness"`, or `"volume"`), either inferred automatically from the assertion type or set explicitly via the `dimension=` parameter on a validation method. This method rolls the per-step results up into a test-unit-weighted pass rate (`0`-`100`) for each dimension present in the plan.

Scores are weighted by test units, so larger tables and steps with more test units contribute proportionally more to each dimension's score. Only steps that have been interrogated contribute; inactive steps are excluded.


## Returns


`dict[str, float]`  
A dictionary mapping each dimension name to its score (a percentage from `0` to `100`). Returns an empty dictionary if the validation has not been interrogated.


## Examples

``` python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset("small_table"))
    .col_vals_not_null(columns="c")
    .col_vals_gt(columns="d", value=0)
    .rows_distinct()
    .interrogate()
)

validation.get_dimension_scores()
```


#### See Also

[Use](Use.md), [score](score.md)
