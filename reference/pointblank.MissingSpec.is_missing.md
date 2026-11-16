# is_missing()

Check whether a value should be considered missing under this spec.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L304-L340)

``` python
is_missing(value)
```

This applies the same rule that validation methods use with `missing=`: a value is missing if it's one of the declared sentinel values, or if it's null and the spec treats nulls as missing.

## Parameters

`value: Any`  
A value from the data.

## Returns

`bool`  
`True` if `value` is a declared sentinel value, or if `value` is null (`None` or a float `NaN`) and `null_is_missing=True`.

## Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(reasons={-99: "not_asked"}, null_is_missing=False)

print(spec.is_missing(-99))
print(spec.is_missing(None))
print(spec.is_missing(42))
```

    True
    False
    False

Only the sentinel `-99` counts as missing here. With `null_is_missing=False` a null doesn't count either, so in a `col_vals_*()` method it would be handled by `na_pass=` instead.
