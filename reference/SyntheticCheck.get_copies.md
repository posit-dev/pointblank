# SyntheticCheck.get_copies()

Get the synthetic rows that copy a seed row.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/synthetic.py#L476-L517)

``` python
SyntheticCheck.get_copies()
```

## Returns

`DataFrame`  
A Polars DataFrame (or a pandas one, if Polars isn't installed) with a row for each synthetic row that copies a seed row: the row numbers in each table (from 1), the kind of copy (`"exact"`, or `"near"` for a row whose text is the same or nearly the same), and the values of the columns that were compared (from the synthetic data).

## Examples

``` python
import pointblank as pb
import polars as pl

seed = pl.DataFrame({"name": ["Ana", "Ben", "Cleo"], "age": [34, 51, 27]})
synthetic = pl.DataFrame({"name": ["Dan", "Ben", "Eve"], "age": [40, 51, 27]})

pb.check_synthetic(synthetic, seed).get_copies()
```

shape: (1, 5)

| seed_row | synthetic_row | copy    | name  | age |
|----------|---------------|---------|-------|-----|
| i64      | i64           | str     | str   | i64 |
| 2        | 2             | "exact" | "Ben" | 51  |

The second synthetic row is the same as the second seed row. The third has the age of the third seed row but another name, so it isn't a copy.
