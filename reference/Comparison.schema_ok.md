# Comparison.schema_ok

Whether the schema check passed.

`Comparison.schema_ok: bool`

A comparison checks the two tables' schemas before it compares their values. How a schema difference is treated depends on `schema=` in [`compare()`](../reference/compare.md#pointblank.compare). With `schema="strict"` (the default), [schema_ok](../reference/Comparison.schema_ok.md#pointblank.Comparison.schema_ok) is the same as [`SchemaDiff.matches`](../reference/SchemaDiff.md#pointblank.SchemaDiff.matches), and a `False` value means that every test unit fails, so [`n_passed`](../reference/Comparison.n_passed.md#pointblank.Comparison.n_passed) is `0`. With `schema="common"`, schema differences are only reported, so [schema_ok](../reference/Comparison.schema_ok.md#pointblank.Comparison.schema_ok) is always `True`.

This makes [schema_ok](../reference/Comparison.schema_ok.md#pointblank.Comparison.schema_ok) the property to check when a comparison fails and it's not clear whether the values or the columns are to blame. The details of the difference are in `schema_diff`.

## Examples

Here, the target table has the same rows as the source, plus a new `humidity` column.

``` python
import pointblank as pb
import polars as pl

readings = pl.DataFrame({"station": ["S1", "S2", "S3"], "temp": [12.1, 14.0, 9.8]})
readings_new = readings.with_columns(humidity=pl.Series([0.61, 0.58, 0.72]))

comparison = pb.compare(readings, readings_new, keys="station")

comparison.schema_ok, comparison.status_counts, comparison.n_passed
```

    (False, {'match': 3, 'changed': 0, 'missing': 0, 'extra': 0, 'dup_key': 0}, 0)

Every row has a status of `"match"`, but none of them pass, because the schema check failed. The cause is in the schema differences:

``` python
comparison.schema_diff.only_in_target
```

    ['humidity']

If the new column is expected, `schema="common"` compares the shared columns and lets the rows pass:

``` python
relaxed = pb.compare(readings, readings_new, keys="station", schema="common")

relaxed.schema_ok, relaxed.n_passed
```

    (True, 3)

The new column is still listed in `relaxed.schema_diff`, so the change remains visible in the report even though it no longer fails the comparison.
