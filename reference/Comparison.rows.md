# Comparison.rows()

Get a sample of rows with a given status.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L861-L895)

``` python
Comparison.rows(
    status,
    limit=100,
)
```

Only up to `limit=` rows are collected, so this is safe to use on very large tables.

## Parameters

`status: Literal["changed", "missing", "extra", "dup_key", "match"]`  
The row status to retrieve:

- `"changed"`: the keys, then a `<column>_source` and a `<column>_target` column for each compared column
- `"missing"`: source rows not in the target (source column names)
- `"extra"`: target rows not in the source (target column names)
- `"dup_key"`: duplicated key values with their occurrence counts in each table (`n_source`, `n_target`)
- `"match"`: the keys of matching rows

`limit: int = 100`  
The maximum number of rows to return.

## Returns

`Any`  
A table of the same type as the (collected) input tables. In positional mode, rows are identified by a 1-based `_row_num_` column.
