## Validate.assert_passing()


Raise an `AssertionError` if all tests are not passing.


Usage

``` python
Validate.assert_passing()
```


The [assert_passing()](Validate.assert_passing.md#pointblank.Validate.assert_passing) method will raise an `AssertionError` if a test does not pass. This method simply wraps [all_passed](Validate.all_passed.md#pointblank.Validate.all_passed) for more ready use in test suites. The step number and assertion made is printed in the `AssertionError` message if a failure occurs, ensuring some details are preserved.

If the validation has not yet been interrogated, this method will automatically call <a href="Validate.interrogate.html#pointblank.Validate.interrogate" class="gdls-link"><code>interrogate()</code></a> with default parameters before checking for passing tests.


## Raises


`AssertionError`  
If any validation step has failing test units.


## Examples

In the example below, we'll use a simple Polars DataFrame with three columns (`a`, `b`, and `c`). There will be three validation steps, and the second step will have a failing test unit (the value `10` isn't less than `9`). The [assert_passing()](Validate.assert_passing.md#pointblank.Validate.assert_passing) method is used to assert that all validation steps passed perfectly, automatically performing the interrogation if needed.


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

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns="a", value=0)
    .col_vals_lt(columns="b", value=9) # this assertion is false
    .col_vals_in_set(columns="c", set=["a", "b"])
)

# No need to call [`interrogate()`](`pointblank.Validate.interrogate`) explicitly
validation.assert_passing()
```


    ---------------------------------------------------------------------------
    AssertionError                            Traceback (most recent call last)
    Cell In[1], line 20
         16     .col_vals_in_set(columns="c", set=["a", "b"])
         17 )
         18 
         19 # No need to call [`interrogate()`](`pointblank.Validate.interrogate`) explicitly
    ---> 20 validation.assert_passing()

    File ~/work/pointblank/pointblank/pointblank/validate.py:14702, in Validate.assert_passing(self)
      14698             ]
      14699             msg = "The following assertions failed:\n" + "\n".join(
      14700                 [f"- Step {i + 1}: {autobrief}" for i, autobrief in failed_steps]
      14701             )
    > 14702             raise AssertionError(msg)

    AssertionError: The following assertions failed:
    - Step 2: Expect that values in `b` should be < `9`.
