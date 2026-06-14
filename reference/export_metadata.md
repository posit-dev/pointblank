## export_metadata()


Export metadata to an external standard format.


Usage

``` python
export_metadata(
    source,
    destination=None,
    format="frictionless",
    **kwargs,
)
```


Converts a MetadataImport object to a standards-compliant representation (e.g., Frictionless Table Schema) and optionally writes it to a file.


## Parameters


`source: MetadataImport`  
The MetadataImport object to export.

`destination: str | Path | None = None`  
Optional file path to write the output. If `None`, returns the result as a dict (for JSON formats) or string.

`format: str = ``"frictionless"`  
Target format. Currently supported: `"frictionless"`.

`**kwargs: Any`  
Additional format-specific options.


## Returns


`dict | str`  
The exported metadata as a dict (JSON formats) or string.


## Raises


`ValueError`  
If the format is not supported.
