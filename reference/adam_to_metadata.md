## adam_to_metadata()


Convert an ADaM dataset template to a MetadataImport object.


Usage

``` python
adam_to_metadata(
    dataset,
    study_id=None,
)
```


## Parameters


`dataset: str`  
ADaM dataset name (e.g., `"ADSL"`, `"BDS"`, `"ADAE"`, `"ADTTE"`). This is case-insensitive.

`study_id: str | None = None`  
Optional study identifier.


## Returns


`MetadataImport`  
A MetadataImport representing the ADaM dataset template.
