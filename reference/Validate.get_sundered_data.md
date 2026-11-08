# Validate.get_sundered_data()

Get the data that passed or failed the validation steps.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/validate.py#L23351-L23561)

``` python
Validate.get_sundered_data(
    type="pass",
    duplicates=None,
    duplicates_order_by=None,
)
```

Validation of the data is one thing but, sometimes, you want to use the best part of the input dataset for something else. The [get_sundered_data()](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) method works with a [Validate](../reference/Validate.md#pointblank.Validate) object that has been interrogated (i.e., the [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate) method was used). We can get either the 'pass' data piece (rows with no failing test units across all column-value based validation functions), or, the 'fail' data piece (rows with at least one failing test unit across the same series of validations).

## Details

There are some caveats to sundering. The validation steps considered for this splitting will only involve steps where:

- of certain check types, where test units are rows or cells checked down a column (e.g., the `col_vals_*()` methods and `rows_complete()`)
- `active=` is not set to `False`
- `pre=` has not been given an expression for modifying the input table, unless it's marked with [`row_preserving()`](../reference/row_preserving.md#pointblank.row_preserving)

So long as these conditions are met, the data will be split into two constituent tables: one with the rows that passed all validation steps and another with the rows that failed at least one validation step. Steps that use `segments=` are included, with each segment's results counting only for the rows in that segment. Likewise, a step with a [row_preserving()](../reference/row_preserving.md#pointblank.row_preserving) `pre=` function only counts for the rows that the function returned.

Steps that use [`rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) are a special case. That step counts every copy of a duplicated row as failing, since it can't know which of them is the 'wrong' one. Such steps are only used for sundering if you say which copy to keep with `duplicates=`: the kept copy goes to the 'pass' piece (if it passes the other steps) and the other copies go to the 'fail' piece. With `duplicates="flag_all"`, every copy goes to the 'fail' piece. The same goes for the groups of near duplicates found by [`rows_near_distinct()`](../reference/Validate.rows_near_distinct.md#pointblank.Validate.rows_near_distinct) steps.

## Parameters

`type: str = "pass"`  
The type of data to return. Options are `"pass"` or `"fail"`, where the former returns a table only containing rows where test units always passed validation steps, and the latter returns a table only containing rows had test units that failed in at least one validation step.

`duplicates: str | None = None`  
How to treat the duplicate rows found by `rows_distinct()` (and `rows_near_distinct()`) steps. The default `None` leaves those steps out of the sundering. Use `"keep_first"` or `"keep_last"` to keep one copy of each group of duplicates (the first or last one, in row order or in the order given by `duplicates_order_by=`), or `"flag_all"` to treat every copy as failing.

`duplicates_order_by: str | list[str] | None = None`  
One or more columns that order the copies of a duplicated row, to decide which copy is kept with `duplicates="keep_first"` or `duplicates="keep_last"`. For example, use `duplicates="keep_last", duplicates_order_by="updated_at"` to keep the newest record. Ties are broken by row order, and a copy with a missing value in one of these columns is kept only if every copy is missing that value. Without this, the copies are taken in row order (for a database table, which has no row order of its own, rows are ordered by their values, so the choice is arbitrary but the same from run to run).

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

A [`rows_distinct()`](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step only takes part in the sundering if we say which copy of a duplicated row to keep. Here, the record with an `id` of `2` appears twice, and we keep the copy with the latest `version`:

``` python
tbl_versions = pl.DataFrame(
    {
        "id": [1, 2, 2, 3],
        "version": [1, 1, 2, 1],
    }
)

validation_versions = (
    pb.Validate(data=tbl_versions)
    .rows_distinct(columns_subset="id")
    .interrogate()
)

pb.preview(
    validation_versions.get_sundered_data(
        duplicates="keep_last", duplicates_order_by="version"
    )
)
```

|     | id *Int64* | version *Int64* |
|-----|------------|-----------------|
| 1   | 1          | 1               |
| 2   | 2          | 2               |
| 3   | 3          | 1               |

The validation report would count both copies of `id` `2` as failing, but with `duplicates="keep_last"` the copy with `version` `2` is kept in the 'pass' piece, and only the older copy would be in the 'fail' piece. Without `duplicates=`, the `rows_distinct()` step would be left out and all four rows would be returned.

## See Also

[get_row_flags()](../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags)  
Get which steps each row failed, as columns added to the table.

[get_data_extracts()](../reference/Validate.get_data_extracts.md#pointblank.Validate.get_data_extracts)  
Get the rows that failed each validation step.
