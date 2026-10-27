# DataScan.from_dict()

Restore a [DataScan](../reference/DataScan.md#pointblank.DataScan) from a dictionary produced by `to_dict()`.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L605-L657)

``` python
DataScan.from_dict(d)
```

This reconstructs the profile without needing the original data.

## Parameters

`d: dict[str, Any]`  
A dictionary with `"metadata"` and `"columns"` keys, as produced by `to_dict()`.

## Returns

`DataScan`  
A restored [DataScan](../reference/DataScan.md#pointblank.DataScan) instance.
