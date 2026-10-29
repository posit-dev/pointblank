# Comparison.status_counts

The number of test units with each row status.

`Comparison.status_counts: dict[str, int]`

Where [`n_passed`](../reference/Comparison.n_passed.md#pointblank.Comparison.n_passed) and [`n_failed`](../reference/Comparison.n_failed.md#pointblank.Comparison.n_failed) say how many rows differ, [status_counts](../reference/Comparison.status_counts.md#pointblank.Comparison.status_counts) says how they differ. The statuses are described in the *Test Units and Row Statuses* section of [`compare()`](../reference/compare.md#pointblank.compare). The counts are computed when the comparison is made, so reading them doesn't touch the tables again.

The counts are the row statuses alone and aren't affected by the schema check. If the schema check fails, there can be `"match"` rows even though no test units pass.

## Examples

In this example, station `S3` appears twice in the source table.

``` python
import pointblank as pb
import polars as pl

readings_v1 = pl.DataFrame(
    {"station": ["S1", "S2", "S3", "S3", "S4"], "temp": [12.1, 14.0, 9.8, 9.9, 11.2]}
)
readings_v2 = pl.DataFrame(
    {"station": ["S1", "S2", "S3", "S5"], "temp": [12.1, 14.3, 9.8, 8.0]}
)

comparison = pb.compare(readings_v1, readings_v2, keys="station")

comparison.status_counts
```

    {'match': 1, 'changed': 1, 'missing': 1, 'extra': 1, 'dup_key': 2}

Each status has a different cause. `S2` changed, `S4` is missing from the target, and `S5` is extra. The duplicated `S3` key counts as two `"dup_key"` units, since it occurs twice in the source. Those rows can't be paired up, so they're not compared. To see the rows behind any of these counts, pass the status to [`rows()`](../reference/Comparison.rows.md#pointblank.Comparison.rows), as in `comparison.rows("dup_key")`.
