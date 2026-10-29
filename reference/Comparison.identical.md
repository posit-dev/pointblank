# Comparison.identical

Whether the tables match under the comparison settings.

`Comparison.identical: bool`

A comparison is identical if the schema check passed (see [`schema_ok`](../reference/Comparison.schema_ok.md#pointblank.Comparison.schema_ok)) and no test unit failed. This is the single answer to "are these tables the same?", and it's also the truth value of the comparison, so `if comparison:` is the same as `if comparison.identical:`.

"Identical" is relative to the settings given to [`compare()`](../reference/compare.md#pointblank.compare). Columns left out with `columns=` aren't checked, values within `tolerance=` count as equal, values are compared after any `normalize=` operations, and with `schema="common"`, the tables can have different columns.

## Examples

Here, a reporting table stores the readings rounded to one decimal place.

``` python
import pointblank as pb
import polars as pl

raw = pl.DataFrame({"station": ["S1", "S2", "S3"], "temp": [12.14, 14.02, 9.81]})
rounded = pl.DataFrame({"station": ["S1", "S2", "S3"], "temp": [12.1, 14.0, 9.8]})

pb.compare(raw, rounded, keys="station").identical
```

    False

An exact comparison finds every row changed. Allowing for the rounding with `tolerance=` makes the tables match:

``` python
comparison = pb.compare(raw, rounded, keys="station", tolerance={"temp": 0.05})

comparison.identical
```

    True

Since a comparison has the same truth value, it can be used directly in a condition:

``` python
if comparison:
    print("The rounded table matches the raw readings.")
```

    The rounded table matches the raw readings.

A result of `True` here means "the same within 0.05 degrees", not "exactly the same". The settings belong with the result, and the comparison report lists them in its header.
