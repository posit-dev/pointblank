# DataScan.load_from_json()

Load a [DataScan](../reference/DataScan.md#pointblank.DataScan) from a JSON file produced by [save_to_json()](../reference/DataScan.save_to_json.md#pointblank.DataScan.save_to_json).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/datascan.py#L676-L692)

``` python
DataScan.load_from_json(input_file)
```

## Parameters

`input_file: str`  
The path to the JSON file.

## Returns

`DataScan`  
A restored [DataScan](../reference/DataScan.md#pointblank.DataScan) instance.
