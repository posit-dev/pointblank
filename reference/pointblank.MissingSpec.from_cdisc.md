# from_cdisc()

Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) for the standard HL7/CDISC null flavors (short alias).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L525-L557)

``` python
from_cdisc(**kwargs)
```

This is a shorter name for [`from_cdisc_null_flavors()`](../reference/pointblank.MissingSpec.from_cdisc_null_flavors.md), and it takes the same keyword arguments (`null_is_missing=`, `null_reason=`, and `description=`).

## Parameters

`**kwargs: Any`  
Keyword arguments passed on to [from_cdisc_null_flavors()](../reference/pointblank.MissingSpec.from_cdisc_null_flavors.md).

## Returns

`MissingSpec`  
A spec with the standard null flavor codes.

## Examples

``` python
import pointblank as pb

cdisc_missing = pb.MissingSpec.from_cdisc(null_reason="unknown")

cdisc_missing.reason_for(None)
```

    'unknown'

Here nulls get the `"unknown"` reason rather than the default `"no_information"`, which still places them in the `"unknown"` category.
