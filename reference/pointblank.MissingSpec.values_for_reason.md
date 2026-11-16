# values_for_reason()

Get all sentinel values that correspond to a given reason.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L342-L372)

``` python
values_for_reason(reason)
```

## Parameters

`reason: str`  
A reason label.

## Returns

`list`  
All sentinel values mapped to `reason`, or an empty list if no sentinel value has that reason. Nulls are never included, even when `reason` is the spec's `null_reason`.

## Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(
    reasons={".": "system_missing", "._": "system_missing", ".R": "refused"}
)

spec.values_for_reason("system_missing")
```

    ['.', '._']

Both SAS-style codes for system missing values are returned. This is the reverse of [`reason_for()`](../reference/pointblank.MissingSpec.reason_for.md), which goes from a value to its reason.
