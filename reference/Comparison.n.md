# Comparison.n

The number of test units (aligned rows).

`Comparison.n: int`

A comparison aligns the rows of the two tables and gives each aligned row one status (see the *Test Units and Row Statuses* section of [`compare()`](../reference/compare.md#pointblank.compare)). `n` is the total, so it's always the sum of the values in [`status_counts`](../reference/Comparison.status_counts.md#pointblank.Comparison.status_counts) and of [`n_passed`](../reference/Comparison.n_passed.md#pointblank.Comparison.n_passed) and [`n_failed`](../reference/Comparison.n_failed.md#pointblank.Comparison.n_failed). It's the denominator to use for any fraction of rows that pass or fail.

`n` isn't the row count of either table. A row that's in both tables counts once, rows in only one table count once each, and a duplicated key counts as many units as the larger of its two occurrence counts. The row counts of the two tables are available as `n_source_rows` and `n_target_rows`.

## Examples

These two tables of weather station readings have four rows each, but only three stations in common.

``` python
import pointblank as pb
import polars as pl

readings_v1 = pl.DataFrame(
    {"station": ["S1", "S2", "S3", "S4"], "temp": [12.1, 14.0, 9.8, 11.2]}
)
readings_v2 = pl.DataFrame(
    {"station": ["S1", "S2", "S3", "S5"], "temp": [12.1, 14.3, 9.8, 8.0]}
)

comparison = pb.compare(readings_v1, readings_v2, keys="station")

comparison.n, comparison.n_source_rows, comparison.n_target_rows
```

    (5, 4, 4)

There are five test units: the three shared stations, `S4` (only in the source), and `S5` (only in the target). Dividing by `n` gives the fraction of rows that failed:

``` python
comparison.n_failed / comparison.n
```

    0.6

Three of the five units failed: `S2` changed, `S4` is missing, and `S5` is extra.
