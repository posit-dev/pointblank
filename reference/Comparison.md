# Comparison

The result of comparing a source table with a target table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/_compare.py#L490-L1516)

``` python
Comparison()
```

[Comparison](../reference/Comparison.md#pointblank.Comparison) objects are created by [`compare()`](../reference/compare.md#pointblank.compare). They hold the row status counts, per-column mismatch counts, and the schema differences, and can retrieve samples of differing rows on demand. A [Comparison](../reference/Comparison.md#pointblank.Comparison) is truthy only when the tables match (see [`identical`](../reference/Comparison.identical.md#pointblank.Comparison.identical)), so `if pb.compare(a, b):` reads naturally.

## Attributes

`source_name, target_name`  
Labels for the two tables.

`keys`  
The key columns used to align rows (empty for positional alignment).

`mode`  
`"keyed"`, `"positional"`, or `"multiset"` (for `keys="*"`).

`schema_diff`  
A [`SchemaDiff`](../reference/SchemaDiff.md#pointblank.SchemaDiff) describing schema differences.

`n_source_rows, n_target_rows`  
Row counts of the two tables.

`n_match, n_changed, n_missing, n_extra, n_dup_key`  
Number of test units (aligned rows) with each status.

`column_mismatches`  
A dict mapping each compared source column to the number of matched rows where its values differ.
