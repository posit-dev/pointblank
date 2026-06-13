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


## Attributes

| Name | Description |
|----|----|
| [file_extensions](#file_extensions) | Built-in mutable sequence. |
| [format_name](#format_name) | str(object='') -\> str |
| [supports_export](#supports_export) | bool(x) -\> bool |
| [supports_import](#supports_import) | bool(x) -\> bool |

------------------------------------------------------------------------


#### file_extensions


Built-in mutable sequence.


`file_extensions: list[str] = []`  


If no argument is given, the constructor creates a new empty list. The argument must be an iterable if specified.


------------------------------------------------------------------------


#### format_name


str(object='') -\> str


`format_name: str = ``""`


str(bytes_or_buffer\[, encoding\[, errors\]\]) -\> str

Create a new string object from the given object. If encoding or errors is specified, then the object must expose a data buffer that will be decoded using the given encoding and error handler. Otherwise, returns the result of object.\_\_str\_\_() (if defined) or repr(object). encoding defaults to sys.getdefaultencoding(). errors defaults to 'strict'.


------------------------------------------------------------------------


#### supports_export


bool(x) -\> bool


`supports_export: bool = ``True`


Returns True when the argument x is true, False otherwise. The builtins True and False are the only two instances of the class bool. The class bool is a subclass of the class int, and cannot be subclassed.


------------------------------------------------------------------------


#### supports_import


bool(x) -\> bool


`supports_import: bool = ``True`


Returns True when the argument x is true, False otherwise. The builtins True and False are the only two instances of the class bool. The class bool is a subclass of the class int, and cannot be subclassed.


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
