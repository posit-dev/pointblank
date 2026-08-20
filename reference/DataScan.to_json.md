# DataScan.to_json()


Export the profile as a JSON string.


Usage

``` python
DataScan.to_json()
```


The JSON is structured for round-trip persistence. Use [from_json()](DataScan.from_json.md#pointblank.DataScan.from_json) or [load_from_json()](DataScan.load_from_json.md#pointblank.DataScan.load_from_json) to restore a [DataScan](DataScan.md#pointblank.DataScan) from the output.


## Returns


`str`  
A JSON string representing the profile.
