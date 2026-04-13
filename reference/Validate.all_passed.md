## Validate.all_passed()


Determine if every validation step passed perfectly, with no failing test units.


Usage

``` python
Validate.all_passed()
```


The [all_passed()](Validate.all_passed.md#pointblank.Validate.all_passed) method determines if every validation step passed perfectly, with no failing test units. This method is useful for quickly checking if the table passed all validation steps with flying colors. If there's even a single failing test unit in any validation step, this method will return `False`.

This validation metric might be overly stringent for some validation plans where failing test units are generally expected (and the strategy is to monitor data quality over time). However, the value of [all_passed()](Validate.all_passed.md#pointblank.Validate.all_passed) could be suitable for validation plans designed to ensure that every test unit passes perfectly (e.g., checks for column presence, null-checking tests, etc.).


## Returns


`bool`  
`True` if all validation steps had no failing test units, `False` otherwise.


## Examples

In the example below, we'll use a simple Polars DataFrame with three columns (`a`, `b`, and `c`). There will be three validation steps, and the second step will have a failing test unit (the value `10` isn't less than `9`). After interrogation, the [all_passed()](Validate.all_passed.md#pointblank.Validate.all_passed) method is used to determine if all validation steps passed perfectly.


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
    .col_vals_lt(columns="b", value=9)
    .col_vals_in_set(columns="c", set=["a", "b"])
    .interrogate()
)

validation.all_passed()
```


    False


The returned value is `False` since the second validation step had a failing test unit. If it weren't for that one failing test unit, the return value would have been `True`.
