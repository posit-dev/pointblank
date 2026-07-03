## Validate.assert_dimension_scores()


Raise an `AssertionError` if any dimension's health score falls below a minimum.


Usage

``` python
Validate.assert_dimension_scores(
    thresholds=None,
    message=None,
)
```


The [assert_dimension_scores()](Validate.assert_dimension_scores.md#pointblank.Validate.assert_dimension_scores) method checks each data quality dimension's score (from <a href="Validate.get_dimension_scores.html#pointblank.Validate.get_dimension_scores" class="gdls-link"><code>get_dimension_scores()</code></a>) against a minimum acceptable value. This is useful in automated testing and CI environments where you want to fail the run when, say, the completeness score drops below `95`.


## Parameters


`thresholds: dict[str, float] | None = None`  
A mapping of dimension name to a minimum acceptable score (`0`-`100`). If `None`, the minimums set via [`config(dimension_thresholds=...)`](%60pointblank.config%60) are used. A dimension present in the thresholds but absent from the validation is ignored.

`message: str | None = None`  
Custom error message to use if the assertion fails. If `None`, a default message that lists the offending dimensions (with actual vs. required scores) is generated.


## Returns


`None`  


## Raises


`AssertionError`  
If any dimension's score is below its specified minimum.


## Examples

``` python
import pointblank as pb

validation = (
    pb.Validate(data=pb.load_dataset("small_table"))
    .col_vals_not_null(columns="c")
    .interrogate()
)

# Fail if the completeness score is below 95
validation.assert_dimension_scores(thresholds={"completeness": 95})
```


#### See Also

[Use](Use.md), [scores](scores.md), [thresholds](thresholds.md)
