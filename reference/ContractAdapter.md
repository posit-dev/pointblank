## ContractAdapter


Base class for contract import/export adapters.


Usage

``` python
ContractAdapter()
```


Subclass this to add support for a new external format.


## Attributes


`format_name: str`  
Short identifier for this format (e.g., `"json_schema"`).

`file_extensions: list[str]`  
File extensions associated with this format (e.g., `[".json"]`).

`supports_import: bool`  
Whether this adapter supports importing from the format.

`supports_export: bool`  
Whether this adapter supports exporting to the format.


## Methods

| Name | Description |
|----|----|
| [detect()](#detect) | Return True if this adapter can handle the given source. |
| [export_contract()](#export_contract) | Export to the external format. |
| [import_contract()](#import_contract) | Import from the external format. |

------------------------------------------------------------------------


#### detect()


Return True if this adapter can handle the given source.


Usage

``` python
detect(source)
```


##### Parameters


`source: Any`  
A file path string, dict, or Python object to inspect.


##### Returns


`bool`  
True if this adapter can handle the source.


------------------------------------------------------------------------


#### export_contract()


Export to the external format.


Usage

``` python
export_contract(validation_or_contract, destination=None, **kwargs)
```


##### Parameters


`validation_or_contract: Any`  
A [Validate](Validate.md#pointblank.Validate) or [Contract](Contract.md#pointblank.Contract) object to export.

`destination: str | None = None`  
Optional file path to write the output. If `None`, returns the result.

`**kwargs: Any`  
Format-specific options.


##### Returns


`str | dict`  
The exported content (string or dict), also written to file if destination given.


------------------------------------------------------------------------


#### import_contract()


Import from the external format.


Usage

``` python
import_contract(source, **kwargs)
```


##### Parameters


`source: Any`  
The source to import from (file path, dict, or Python object).

`**kwargs: Any`  
Format-specific options.


##### Returns


`ContractImport`  
The import result with columns, constraints, and metadata.
