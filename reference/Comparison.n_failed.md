# Comparison.n_failed

The number of test units that failed.

`Comparison.n_failed: int`

A test unit fails if its status is `"changed"`, `"missing"`, `"extra"`, or `"dup_key"`. If the schema check failed (see [`schema_ok`](../reference/Comparison.schema_ok.md#pointblank.Comparison.schema_ok)), every test unit fails, so `n_failed` equals [`n`](../reference/Comparison.n.md#pointblank.Comparison.n). It's always `n - n_passed`.

A count of zero failures, along with a passing schema check, is what makes a comparison [`identical`](../reference/Comparison.identical.md#pointblank.Comparison.identical). To see how the failures break down by status, use [`status_counts`](../reference/Comparison.status_counts.md#pointblank.Comparison.status_counts).

## Examples

One station's reading changed, one station was dropped, and one was added.

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

comparison.n_failed
```

    3

A count is useful for a quick check, for instance to stop a script when too many rows differ:

``` python
if comparison.n_failed > 0.25 * comparison.n:
    print(f"{comparison.n_failed} of {comparison.n} rows differ")
```

    3 of 5 rows differ

To act on failures with thresholds and actions instead, compare the tables in a validation with [`Validate.tbl_match()`](../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match).
