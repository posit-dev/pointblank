# Validate.get_health_score()

Get the overall data quality health score from the validation results.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L20287-L20358)

``` python
Validate.get_health_score()
```

The health score is a single number (a percentage from `0` to `100`) that summarizes the overall quality of the data across all validation steps. It is computed as a test-unit weighted pass rate: the total number of passing test units divided by the total number of test units. This means larger tables and steps operating over more rows contribute proportionally more to the score, so it tracks data volume rather than mere step count.

Per-dimension weights can be set globally via `config(dimension_weights=...)` for organizations that consider some dimensions (e.g., completeness) more critical than others. A weight scales that dimension's test-unit contribution to the score (so a dimension's influence is its weight times its test-unit count); the returned score is always within `[0, 100]`.

## Returns

`float`  
The overall health score as a percentage from `0` to `100`. Returns `100.0` if the validation has not been interrogated or contains no scorable steps.

## Examples

Let's validate the `small_table` dataset with a not-null check, a range check, and a check for duplicate rows, then get the overall score.

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

validation.get_health_score()
```

    74.36

The three steps have 39 test units in all, and 29 of them passed, for a score of 74.36. If missing values matter more to you than the other problems, give completeness a larger weight:

``` python
pb.config(dimension_weights={"completeness": 3})

validation.get_health_score()
```

    78.46

The completeness step now counts three times, so its 84.62 pass rate pulls the score up to 78.46. Note that each call to `config()` sets every option, so any other settings you rely on (such as those for the report) need to be included in the same call. Calling `pb.config()` with no arguments goes back to the defaults, with all dimensions weighted equally.

## See Also

[get_dimension_scores()](../reference/Validate.get_dimension_scores.md#pointblank.Validate.get_dimension_scores)  
Get a per-dimension breakdown of the score.
