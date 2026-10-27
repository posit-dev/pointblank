# DataScan.from_json()

Restore a [DataScan](../reference/DataScan.md#pointblank.DataScan) from a JSON string produced by `to_json()`.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L664-L679)

``` python
DataScan.from_json(json_string)
```

## Parameters

`json_string: str`  
A JSON string as produced by `to_json()`.

## Returns

`DataScan`  
A restored [DataScan](../reference/DataScan.md#pointblank.DataScan) instance.
