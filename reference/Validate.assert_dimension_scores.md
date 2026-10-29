# Validate.assert_dimension_scores()

Raise an `AssertionError` if any dimension's health score falls below a minimum.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L20370-L20484)

``` python
Validate.assert_dimension_scores(
    thresholds=None,
    message=None,
)
```

The [assert_dimension_scores()](../reference/Validate.assert_dimension_scores.md#pointblank.Validate.assert_dimension_scores) method checks each data quality dimension's score (from [`get_dimension_scores()`](../reference/Validate.get_dimension_scores.md#pointblank.Validate.get_dimension_scores)) against a minimum acceptable value. This is useful in automated testing and CI environments where you want to fail the run when, say, the completeness score drops below `95`.

## Parameters

`thresholds: dict[str, float] | None = None`  
A mapping of dimension name to a minimum acceptable score (`0`-`100`). If `None`, the minimums set via `config(dimension_thresholds=...)` are used. A dimension present in the thresholds but absent from the validation is ignored.

`message: str | None = None`  
Custom error message to use if the assertion fails. If `None`, a default message that lists the offending dimensions (with actual vs. required scores) is generated.

## Returns

`None`  

## Raises

`AssertionError`  
If any dimension's score is below its specified minimum.

## Examples

Let's validate the `small_table` dataset with a not-null check (a completeness step) and a range check (a validity step).

``` python
import pointblank as pb

small_table = pb.load_dataset(dataset="small_table", tbl_type="polars")

validation = (
    pb.Validate(data=small_table)
    .col_vals_not_null(columns="c")
    .col_vals_gt(columns="d", value=1000)
    .interrogate()
)

validation.get_dimension_scores()
```

    {'completeness': 84.62, 'validity': 53.85}

Requiring completeness of at least 80 and validity of at least 50 passes, so nothing happens:

``` python
validation.assert_dimension_scores(thresholds={"completeness": 80, "validity": 50})
```

Raising the completeness minimum to 95 makes the assertion fail. The `AssertionError` is caught here so that its message can be shown:

``` python
try:
    validation.assert_dimension_scores(thresholds={"completeness": 95, "validity": 50})
except AssertionError as e:
    print(e)
```

    Dimension health score(s) below the required minimum: completeness (84.62% < 95%)

The message lists each dimension below its minimum, with the actual and required scores. In a test suite or a CI job, leave the error uncaught so that it fails the run. To use the same minimums everywhere, set them once with `config(dimension_thresholds=...)` and call [assert_dimension_scores()](../reference/Validate.assert_dimension_scores.md#pointblank.Validate.assert_dimension_scores) without `thresholds=`.

## See Also

[get_dimension_scores()](../reference/Validate.get_dimension_scores.md#pointblank.Validate.get_dimension_scores)  
Get the scores without raising.

`config`  
Set default per-dimension thresholds globally.
