# ref()

Reference a column from the reference data for aggregate comparisons.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/column.py#L548-L628)

``` python
ref(column_name)
```

This function is used with aggregate validation methods (like [col_sum_eq](../reference/Validate.col_sum_eq.md#pointblank.Validate.col_sum_eq), [col_avg_gt](../reference/Validate.col_avg_gt.md#pointblank.Validate.col_avg_gt), etc.) to compare the aggregate value of a column in the main data against the aggregate value of a column in the reference data.

To use this function, you must first set the reference data on the [Validate](../reference/Validate.md#pointblank.Validate) object using the `reference=` parameter in the constructor.

## Parameters

`column_name: str`  
The name of the column in the reference data to compute the aggregate from.

## Returns

`ReferenceColumn`  
A reference column marker that indicates the value should be computed from the reference data.

## Examples

Suppose we have two DataFrames: a current data table and a reference (historical) table. We want to validate that the sum of a column in the current data matches the sum of the same column in the reference data.

``` python
import pointblank as pb
import polars as pl

# Current data
current_data = pl.DataFrame({"sales": [100, 200, 300]})

# Reference (historical) data
reference_data = pl.DataFrame({"sales": [100, 200, 300]})

validation = (
    pb.Validate(data=current_data, reference=reference_data)
    .col_sum_eq("sales", pb.ref("sales"))
    .interrogate()
)

validation
```

[TABLE]

The `sales` column sums to `600` in both tables, so the single test unit passes.

You can also compare different columns or use tolerance:

``` python
current_data = pl.DataFrame({"revenue": [105, 205, 305]})
reference_data = pl.DataFrame({"sales": [100, 200, 300]})

# Check if revenue sum is within 10% of sales sum
validation = (
    pb.Validate(data=current_data, reference=reference_data)
    .col_sum_eq("revenue", pb.ref("sales"), tol=0.1)
    .interrogate()
)

validation
```

[TABLE]

The `revenue` column sums to `615`, which is `15` more than the `600` of the reference `sales` column. That's within the relative tolerance of 10% (`60`), so the step passes even though the sums aren't equal.

## See Also

[col()](../reference/col.md#pointblank.col)  
Reference columns within the same table.
