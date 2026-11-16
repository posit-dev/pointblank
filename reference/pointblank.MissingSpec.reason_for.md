# reason_for()

Get the reason label for a specific value.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L269-L302)

``` python
reason_for(value)
```

## Parameters

`value: Any`  
A value from the data.

## Returns

`str | None`  
The reason label if `value` is a declared sentinel value, `null_reason` if `value` is null (`None` or a float `NaN`) and `null_is_missing=True`, or `None` if the value is not considered missing.

## Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"})

print(spec.reason_for(-98))
print(spec.reason_for(None))
print(spec.reason_for(42))
```

    refused
    unknown
    None

The sentinel `-98` gives its declared reason, and a null gets the spec's `null_reason` (`"unknown"` by default). An ordinary value like `42` isn't missing, so its reason is `None`.
