## DataScan.from_dict()


Restore a [DataScan](DataScan.md#pointblank.DataScan) from a dictionary produced by [to_dict()](Step.md#pointblank.Step.to_dict).


Usage

``` python
DataScan.from_dict(d)
```


This reconstructs the profile without needing the original data.


## Parameters


`d: dict[str, Any]`  
A dictionary with `"metadata"` and `"columns"` keys, as produced by [to_dict()](Step.md#pointblank.Step.to_dict).


## Returns


`DataScan`  
A restored [DataScan](DataScan.md#pointblank.DataScan) instance.
