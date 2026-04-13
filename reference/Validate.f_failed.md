## Validate.f_failed()


Provides a dictionary of the fraction of test units that failed for each validation step.


Usage

``` python
Validate.f_failed(
    i=None,
    scalar=False,
)
```


A measure of the fraction of test units that failed is provided by the [f_failed](Validate.f_failed.md#pointblank.Validate.f_failed) attribute. This is the fraction of test units that failed the validation step over the total number of test units. Given this is a fractional value, it will always be in the range of `0` to `1`.

Test units are the atomic units of the validation process. Different validations can have different numbers of test units. For example, a validation that checks for the presence of a column in a table will have a single test unit. A validation that checks for the presence of a value in a column will have as many test units as there are rows in the table.

This method provides a dictionary of the fraction of failing test units for each validation step. If the `scalar=True` argument is provided and `i=` is a scalar, the value is returned as a scalar instead of a dictionary. Furthermore, a value obtained here will be the complement to the analogous value returned by the <a href="Validate.f_passed.html#pointblank.Validate.f_passed" class="gdls-link"><code>f_passed()</code></a> method (i.e., `1 - f_passed()`).


## Parameters


`i: int | list[int] | None = None`  
The validation step number(s) from which the fraction of failing test units is obtained. Can be provided as a list of integers or a single integer. If `None`, all steps are included.

`scalar: bool = ``False`  
If `True` and `i=` is a scalar, return the value as a scalar instead of a dictionary.


## Returns


`dict[int, float] | float`  
A dictionary of the fraction of failing test units for each validation step or a scalar value.


## Examples

In the example below, we'll use a simple Polars DataFrame with three columns (`a`, `b`, and `c`). There will be three validation steps, all having some failing test units. After interrogation, the [f_failed()](Validate.f_failed.md#pointblank.Validate.f_failed) method is used to determine the fraction of failing test units for each validation step.


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [7, 4, 9, 7, 12, 3, 10],
        "b": [9, 8, 10, 5, 10, 6, 2],
        "c": ["a", "b", "c", "a", "b", "d", "c"]
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns="a", value=5)
    .col_vals_gt(columns="b", value=pb.col("a"))
    .col_vals_in_set(columns="c", set=["a", "b"])
    .interrogate()
)

validation.f_failed()
```


    {1: 0.2857142857142857, 2: 0.42857142857142855, 3: 0.42857142857142855}


The returned dictionary shows the fraction of failing test units for each validation step. The values are all greater than `0` since there were failing test units in each step.

If we wanted to check the fraction of failing test units for a single validation step, we can provide the step number. Also, we could have the value returned as a scalar by setting `scalar=True` (ensuring that `i=` is a scalar).


``` python
validation.f_failed(i=1)
```


    {1: 0.2857142857142857}


The returned value is the proportion of failing test units for the first validation step (2 failing test units out of 7 total test units).
