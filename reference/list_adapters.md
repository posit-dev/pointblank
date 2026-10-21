# list_adapters()

List all registered adapters with their capabilities.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/adapters/_registry.py#L81-L102)

``` python
list_adapters()
```

## Returns

`dict`  
A dictionary mapping format names to adapter info dicts with keys:

- "class": the adapter class name
- "file_extensions": associated file extensions
- "supports_import": whether import is supported
- "supports_export": whether export is supported
