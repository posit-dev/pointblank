# Comparison.to_dict()


Get the comparison results as a JSON-serializable dict.


Usage

``` python
Comparison.to_dict(
    samples=True,
    limit=10,
)
```


## Parameters


`samples: bool = ``True`  
Should samples of differing rows be included? Set to `False` when the data might contain sensitive values (e.g., PII) and only counts and schema details may be stored.

`limit: int = ``10`  
The maximum number of sample rows per status.
