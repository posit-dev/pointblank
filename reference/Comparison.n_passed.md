# Comparison.n_passed

The number of test units that passed.

`Comparison.n_passed: int`

A test unit passes if its status is `"match"`, which means the row is in both tables with the same values in every compared column (within any tolerance set in [`compare()`](../reference/compare.md#pointblank.compare)). If the schema check failed (see [`schema_ok`](../reference/Comparison.schema_ok.md#pointblank.Comparison.schema_ok)), no test unit passes and `n_passed` is `0`, whatever the row statuses are.

`n_passed` and [`n_failed`](../reference/Comparison.n_failed.md#pointblank.Comparison.n_failed) are the counts that [`Validate.tbl_match()`](../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match) uses for its test units, so they work with the same thresholds and fractions as any other validation step.

## Examples

Two of these four stations have the same reading in both tables.

``` python
import pointblank as pb
import polars as pl

readings_v1 = pl.DataFrame(
    {"station": ["S1", "S2", "S3", "S4"], "temp": [12.1, 14.0, 9.8, 11.2]}
)
readings_v2 = pl.DataFrame(
    {"station": ["S1", "S2", "S3", "S4"], "temp": [12.1, 14.3, 9.8, 11.5]}
)

comparison = pb.compare(readings_v1, readings_v2, keys="station")

comparison.n_passed, comparison.n_passed / comparison.n
```

    (2, 0.5)

Half of the test units passed. The two changed rows are available with [`rows("changed")`](../reference/Comparison.rows.md#pointblank.Comparison.rows).
