# DataScan.from_dict()

Restore a [DataScan](../reference/DataScan.md#pointblank.DataScan) from a dictionary.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L612-L666)

``` python
DataScan.from_dict(d)
```

The dictionary is the one produced by [`to_dict()`](../reference/DataScan.to_dict.md#pointblank.DataScan.to_dict). This reconstructs the profile without needing the original data.

## Parameters

`d: dict[str, Any]`  
A dictionary with `"metadata"` and `"columns"` keys, as produced by [`to_dict()`](../reference/DataScan.to_dict.md#pointblank.DataScan.to_dict).

## Returns

`DataScan`  
A restored [DataScan](../reference/DataScan.md#pointblank.DataScan) instance.
