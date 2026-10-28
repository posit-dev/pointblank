# Comparison.column_summary()

Get a per-column summary of value mismatches.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L1385-L1423)

``` python
Comparison.column_summary()
```

## Returns

`Any`  
A table (Polars if available, else Pandas) with one row per compared column: the source and target column names and data types, the number of matched rows whose values differ (`n_mismatch`), and that number as a fraction of matched rows (`f_mismatch`).
