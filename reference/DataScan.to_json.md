# DataScan.to_json()

Export the profile as a JSON string.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L504-L516)

``` python
DataScan.to_json()
```

The JSON is structured for round-trip persistence. Use [from_json()](../reference/DataScan.from_json.md#pointblank.DataScan.from_json) or [load_from_json()](../reference/DataScan.load_from_json.md#pointblank.DataScan.load_from_json) to restore a [DataScan](../reference/DataScan.md#pointblank.DataScan) from the output.

## Returns

`str`  
A JSON string representing the profile.
