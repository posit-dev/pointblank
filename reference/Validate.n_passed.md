## Validate.n_passed()


Provides a dictionary of the number of test units that passed for each validation step.


Usage

``` python
Validate.n_passed(
    i=None,
    scalar=False,
)
```


The [n_passed()](Validate.n_passed.md#pointblank.Validate.n_passed) method provides the number of test units that passed for each validation step. This is the number of test units that passed in the the validation step. It is always some integer value between `0` and the total number of test units.

Test units are the atomic units of the validation process. Different validations can have different numbers of test units. For example, a validation that checks for the presence of a column in a table will have a single test unit. A validation that checks for the presence of a value in a column will have as many test units as there are rows in the table.

The method provides a dictionary of the number of passing test units for each validation step. If the `scalar=True` argument is provided and `i=` is a scalar, the value is returned as a scalar instead of a dictionary. Furthermore, a value obtained here will be the complement to the analogous value returned by the <a href="Validate.n_passed.html#pointblank.Validate.n_passed" class="gdls-link"><code>n_passed()</code></a> method (i.e., `n - n_failed`).


## Parameters


`i: int | list[int] | None = None`  
The validation step number(s) from which the number of passing test units is obtained. Can be provided as a list of integers or a single integer. If `None`, all steps are included.

`scalar: bool = ``False`  
If `True` and `i=` is a scalar, return the value as a scalar instead of a dictionary.


## Returns


`dict[int, int] | int`  
A dictionary of the number of passing test units for each validation step or a scalar value.


## Examples

In the example below, we'll use a simple Polars DataFrame with three columns (`a`, `b`, and `c`). There will be three validation steps and, as it turns out, all of them will have failing test units. After interrogation, the [n_passed()](Validate.n_passed.md#pointblank.Validate.n_passed) method is used to determine the number of passing test units for each validation step.


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [7, 4, 9, 7, 12],
        "b": [9, 8, 10, 5, 10],
        "c": ["a", "b", "c", "a", "b"]
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns="a", value=5)
    .col_vals_gt(columns="b", value=pb.col("a"))
    .col_vals_in_set(columns="c", set=["a", "b"])
    .interrogate()
)

validation.n_passed()
```


    {1: 4, 2: 3, 3: 4}


The returned dictionary shows that all validation steps had no passing test units (each value was less than `5`, which is the total number of test units for each step).

If we wanted to check the number of passing test units for a single validation step, we can provide the step number. Also, we could forego the dictionary and get a scalar value by setting `scalar=True` (ensuring that `i=` is a scalar).


``` python
validation.n_passed(i=1)
```


    {1: 4}


The returned value of `4` is the number of passing test units for the first validation step.
