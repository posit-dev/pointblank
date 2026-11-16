# sentinel_values()

Get all sentinel values that encode missingness.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L239-L267)

``` python
sentinel_values()
```

The sentinel values are the keys of `reasons`, in the order they were given. This is the list to use when you need the codes themselves, for example to filter them out of a table before computing a summary statistic.

## Returns

`list`  
The keys of `reasons` (the actual values in the data that represent missingness). Note that this does *not* include `None` even when `null_is_missing=True`; use [`is_missing()`](../reference/pointblank.MissingSpec.is_missing.md) to test individual values.

## Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused", -97: "refused"})

spec.sentinel_values()
```

    [-99, -98, -97]

All three codes are listed, including the two that share the `"refused"` reason. Nulls aren't listed even though the spec treats them as missing (`null_is_missing=True` is the default), because they aren't values that appear in `reasons`.
