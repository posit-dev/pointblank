# expr_col()

Create a column expression for use in `conjointly()` validation.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/column.py#L2016-L2083)

``` python
expr_col(column_name)
```

This function returns a ColumnExpression object that supports operations like `>`, `<`, `+`, etc. for use in [`conjointly()`](../reference/Validate.conjointly.md#pointblank.Validate.conjointly) validation expressions.

## Parameters

`column_name: str`  
The name of the column to reference.

## Returns

`ColumnExpression`  
A column expression that can be used in comparisons and operations.

## Examples

Let's say we have a table with three columns: `a`, `b`, and `c`. We want to validate that:

- The values in column `a` are greater than `2`.
- The values in column `b` are less than `7`.
- The sum of columns `a` and `b` is less than the values in column `c`.

We can use the [expr_col()](../reference/expr_col.md#pointblank.expr_col) function to create a column expression for each of these conditions.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame(
    {
        "a": [5, 7, 1, 3, 9, 4],
        "b": [6, 3, 0, 5, 8, 2],
        "c": [10, 4, 8, 9, 10, 5],
    }
)

# Using expr_col() to create backend-agnostic validation expressions
validation = (
    pb.Validate(data=tbl)
    .conjointly(
        lambda df: pb.expr_col("a") > 2,
        lambda df: pb.expr_col("b") < 7,
        lambda df: pb.expr_col("a") + pb.expr_col("b") < pb.expr_col("c")
    )
    .interrogate()
)

validation
```

[TABLE]

The above code creates a validation object that checks the specified conditions using the [expr_col()](../reference/expr_col.md#pointblank.expr_col) function. The resulting validation table will show whether each condition was satisfied for each row in the table.

## See Also

[conjointly()](../reference/Validate.conjointly.md#pointblank.Validate.conjointly)  
The validation step where these expressions are used.
