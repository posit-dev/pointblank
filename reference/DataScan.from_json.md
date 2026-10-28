# DataScan.from_json()

Restore a [DataScan](../reference/DataScan.md#pointblank.DataScan) from a JSON string.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L668-L685)

``` python
DataScan.from_json(json_string)
```

The JSON string is the one produced by [`to_json()`](../reference/DataScan.to_json.md#pointblank.DataScan.to_json).

## Parameters

`json_string: str`  
A JSON string as produced by [`to_json()`](../reference/DataScan.to_json.md#pointblank.DataScan.to_json).

## Returns

`DataScan`  
A restored [DataScan](../reference/DataScan.md#pointblank.DataScan) instance.
