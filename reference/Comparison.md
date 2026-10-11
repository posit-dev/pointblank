# Comparison


The result of comparing a source table with a target table.


Usage

``` python
Comparison()
```


[Comparison](Comparison.md#pointblank.Comparison) objects are created by <a href="../reference/compare.html#pointblank.compare" class="gdls-link"><code>compare()</code></a>. They hold the row status counts, per-column mismatch counts, and the schema differences, and can retrieve samples of differing rows on demand.


## Attributes


`source_name, target_name`  
Labels for the two tables.

`keys`  
The key columns used to align rows (empty for positional alignment).

`mode`  
`"keyed"` or `"positional"`.

`schema_diff`  
A <a href="../reference/SchemaDiff.html#pointblank.SchemaDiff" class="gdls-link"><code>SchemaDiff</code></a> describing schema differences.

`n_source_rows, n_target_rows`  
Row counts of the two tables.

`n_match, n_changed, n_missing, n_extra, n_dup_key`  
Number of test units (aligned rows) with each status.

`column_mismatches`  
A dict mapping each compared source column to the number of matched rows where its values differ.
