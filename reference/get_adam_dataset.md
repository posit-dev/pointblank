## get_adam_dataset()


Get the ADaM template for a specific dataset.


Usage

``` python
get_adam_dataset(name)
```


## Parameters


`name: str`  
Dataset name (e.g., `"ADSL"`, `"BDS"`, `"ADAE"`, `"ADTTE"`). This is case-insensitive.


## Returns


`ADaMDatasetTemplate`  
The structural template for the dataset.


## Raises


`KeyError`  
If the dataset is not supported.
