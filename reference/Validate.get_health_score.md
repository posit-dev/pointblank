## Validate.get_health_score()


Get the overall data quality health score from the validation results.


Usage

``` python
Validate.get_health_score()
```


The health score is a single number (a percentage from `0` to `100`) that summarizes the overall quality of the data across all validation steps. It is computed as a test-unit weighted pass rate: the total number of passing test units divided by the total number of test units. This means larger tables and steps operating over more rows contribute proportionally more to the score, so it tracks data volume rather than mere step count.

Per-dimension weights can be set globally via [`config(dimension_weights=...)`](%60pointblank.config%60) for organizations that consider some dimensions (e.g., completeness) more critical than others. A weight scales that dimension's test-unit contribution to the score (so a dimension's influence is its weight times its test-unit count); the returned score is always within `[0, 100]`.


## Returns


`float`  
The overall health score as a percentage from `0` to `100`. Returns `100.0` if the validation has not been interrogated or contains no scorable steps.


## Examples

``` python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset("small_table"))
    .col_vals_not_null(columns="c")
    .col_vals_gt(columns="d", value=0)
    .interrogate()
)

validation.get_health_score()
```


## See Also

[](%60~Use%60) <a href="Validate.get_dimension_scores.html#pointblank.Validate.get_dimension_scores" class="gdls-link"><code>get_dimension_scores()</code></a> for a per-dimension  

[](%60~breakdown%60) of the score.
