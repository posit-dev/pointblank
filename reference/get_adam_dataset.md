# get_adam_dataset()

Get the ADaM template for a specific dataset.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_adam_templates.py#L763-L785)

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
