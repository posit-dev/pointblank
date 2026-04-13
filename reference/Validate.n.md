## Validate.n()


Provides a dictionary of the number of test units for each validation step.


Usage

``` python
Validate.n(
    i=None,
    scalar=False,
)
```


The [n()](Validate.n.md#pointblank.Validate.n) method provides the number of test units for each validation step. This is the total number of test units that were evaluated in the validation step. It is always an integer value.

Test units are the atomic units of the validation process. Different validations can have different numbers of test units. For example, a validation that checks for the presence of a column in a table will have a single test unit. A validation that checks for the presence of a value in a column will have as many test units as there are rows in the table.

The method provides a dictionary of the number of test units for each validation step. If the `scalar=True` argument is provided and `i=` is a scalar, the value is returned as a scalar instead of a dictionary. The total number of test units for a validation step is the sum of the number of passing and failing test units (i.e., `n = n_passed + n_failed`).


## Parameters


`i: int | list[int] | None = None`  
The validation step number(s) from which the number of test units is obtained. Can be provided as a list of integers or a single integer. If `None`, all steps are included.

`scalar: bool = ``False`  
If `True` and `i=` is a scalar, return the value as a scalar instead of a dictionary.


## Returns


`dict[int, int] | int`  
A dictionary of the number of test units for each validation step or a scalar value.


## Examples

Different types of validation steps can have different numbers of test units. In the example below, we'll use a simple Polars DataFrame with three columns (`a`, `b`, and `c`). There will be three validation steps, and the number of test units for each step will be a little bit different.


``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [1, 2, 9, 5],
        "b": [5, 6, 10, 3],
        "c": ["a", "b", "a", "a"],
    }
)

# Define a preprocessing function
def filter_by_a_gt_1(df):
    return df.filter(pl.col("a") > 1)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns="a", value=0)
    .col_exists(columns="b")
    .col_vals_lt(columns="b", value=9, pre=filter_by_a_gt_1)
    .interrogate()
)
```


The first validation step checks that all values in column `a` are greater than `0`. Let's use the [n()](Validate.n.md#pointblank.Validate.n) method to determine the number of test units this validation step.


``` python
validation.n(i=1, scalar=True)
```


    4


The returned value of `4` is the number of test units for the first validation step. This value is the same as the number of rows in the table.

The second validation step checks for the existence of column `b`. Using the [n()](Validate.n.md#pointblank.Validate.n) method we can get the number of test units for this the second step.


``` python
validation.n(i=2, scalar=True)
```


    1


There's a single test unit here because the validation step is checking for the presence of a single column.

The third validation step checks that all values in column `b` are less than `9` after filtering the table to only include rows where the value in column `a` is greater than `1`. Because the table is filtered, the number of test units will be less than the total number of rows in the input table. Let's prove this by using the [n()](Validate.n.md#pointblank.Validate.n) method.


``` python
validation.n(i=3, scalar=True)
```


    3


The returned value of `3` is the number of test units for the third validation step. When using the `pre=` argument, the input table can be mutated before performing the validation. The [n()](Validate.n.md#pointblank.Validate.n) method is a good way to determine whether the mutation performed as expected.

In all of these examples, the `scalar=True` argument was used to return the value as a scalar integer value. If `scalar=False`, the method will return a dictionary with an entry for the validation step number (from the `i=` argument) and the number of test units. Futhermore, leaving out the `i=` argument altogether will return a dictionary with filled with the number of test units for each validation step. Here's what that looks like:


``` python
validation.n()
```


    {1: 4, 2: 1, 3: 3}
