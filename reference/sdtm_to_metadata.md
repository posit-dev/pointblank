## sdtm_to_metadata()


Convert an SDTM domain template to a [MetadataImport](MetadataImport.md#pointblank.MetadataImport) object.


Usage

``` python
sdtm_to_metadata(
    domain,
    study_id=None,
)
```


This allows using the standard metadata pipeline ([to_schema](MetadataImport.md#pointblank.MetadataImport.to_schema), [to_validate](Contract.md#pointblank.Contract.to_validate)) with SDTM domain specifications.


## Parameters


`domain: str`  
SDTM domain code (e.g., `"DM"`, `"AE"`, `"LB"`). This is case-insensitive.

`study_id: str | None = None`  
Optional study identifier to include in metadata.


## Returns


`MetadataImport`  
A [MetadataImport](MetadataImport.md#pointblank.MetadataImport) representing the SDTM domain template.
