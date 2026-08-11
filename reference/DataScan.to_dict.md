## DataScan.to_dict()


Export the profile as a structured dictionary.


Usage

``` python
DataScan.to_dict()
```


The returned dictionary contains metadata (table name, row count, column list) plus per-column profile entries with their data type, statistics, and sample data. This format is designed for round-trip persistence: save it with [to_json()](DataScan.to_json.md#pointblank.DataScan.to_json) / [save_to_json()](DataScan.save_to_json.md#pointblank.DataScan.save_to_json) and restore with [from_dict()](Step.md#pointblank.Step.from_dict) / [from_json()](DataScan.from_json.md#pointblank.DataScan.from_json) / [load_from_json()](DataScan.load_from_json.md#pointblank.DataScan.load_from_json).


## Returns


`dict[str, Any]`  
A dictionary with keys `"metadata"` and `"columns"`.
