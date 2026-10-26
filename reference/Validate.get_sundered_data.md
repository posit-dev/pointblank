# Validate.get_sundered_data()

Get the data that passed or failed the validation steps.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L20393-L20557)

``` python
Validate.get_sundered_data(type="pass")
```

Validation of the data is one thing but, sometimes, you want to use the best part of the input dataset for something else. The [get_sundered_data()](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) method works with a [Validate](../reference/Validate.md#pointblank.Validate) object that has been interrogated (i.e., the [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) method was used). We can get either the 'pass' data piece (rows with no failing test units across all column-value based validation functions), or, the 'fail' data piece (rows with at least one failing test unit across the same series of validations).

## Details

There are some caveats to sundering. The validation steps considered for this splitting will only involve steps where:

- of certain check types, where test units are cells checked down a column (e.g., the `col_vals_*()` methods)
- `active=` is not set to `False`
- `pre=` has not been given an expression for modifying the input table

So long as these conditions are met, the data will be split into two constituent tables: one with the rows that passed all validation steps and another with the rows that failed at least one validation step.

## Parameters

`type: str = "pass"`  
The type of data to return. Options are `"pass"` or `"fail"`, where the former returns a table only containing rows where test units always passed validation steps, and the latter returns a table only containing rows had test units that failed in at least one validation step.

## Returns

`Any`  
A table containing the data that passed or failed the validation steps.

## Examples

Let's create a [Validate](../reference/Validate.md#pointblank.Validate) object with three validation steps and then interrogate the data.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [7, 6, 9, 7, 3, 2],
        "b": [9, 8, 10, 5, 10, 6],
        "c": ["c", "d", "a", "b", "a", "b"]
    }
)

validation = (
    pb.Validate(data=tbl)
    .col_vals_gt(columns="a", value=5)
    .col_vals_in_set(columns="c", set=["a", "b"])
    .interrogate()
)

validation
```

[TABLE]

From the validation table, we can see that the first and second steps each had 4 passing test units. A failing test unit will mark the entire row as failing in the context of the [get_sundered_data()](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) method. We can use this method to get the rows of data that passed the during interrogation.

``` python
pb.preview(validation.get_sundered_data())
```

|     | a *Int64* | b *Int64* | c *String* |
|-----|-----------|-----------|------------|
| 1   | 9         | 10        | a          |
| 2   | 7         | 5         | b          |

The returned DataFrame contains the rows that passed all validation steps (we passed this object to [`preview()`](../reference/preview.md#pointblank.preview) to show it in an HTML view). From the six-row input DataFrame, the first two rows and the last two rows had test units that failed validation. Thus the middle two rows are the only ones that passed all validation steps and that's what we see in the returned DataFrame.
