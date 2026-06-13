## ContractImport


Result of importing an external contract/schema.


Usage

``` python
ContractImport()
```


Contains the parsed schema information, mapped validation steps, and any warnings about constraints that couldn't be fully translated.


## Parameters


`source_format: str`  
The adapter format name (e.g., `"json_schema"`, `"frictionless"`, etc.).

`source_path: str | None = None`  
File path if loaded from a file, else None.

`source_version: str | None = None`  
Version of the source format specification, if detectable.

`columns: list[tuple[str, str | None]] = list()`    
List of (column_name, dtype_string_or_None) tuples detected from the source.

`constraints: list[MappedConstraint] = list()`    
List of MappedConstraint objects ready for translation to Validate steps.

`metadata: dict[str, Any] = dict()`    
Any additional metadata from the source (title, description, etc.).

`warnings: list[str] = list()`    
Messages about constraints that couldn't be mapped.

`coverage: float = ``1.0`  
Fraction of source constraints successfully mapped (`0.0` to `1.0`).


## Methods

| Name | Description |
|----|----|
| [summary()](#summary) | Return a human-readable summary of what was imported. |
| [to_contract()](#to_contract) | Build a Contract object from the imported data. |
| [to_python()](#to_python) | Generate Python code string for the validation workflow. |
| [to_validate()](#to_validate) | Build a Validate object from the imported contract. |
| [to_yaml()](#to_yaml) | Generate Pointblank YAML configuration string. |

------------------------------------------------------------------------


#### summary()


Return a human-readable summary of what was imported.


Usage

``` python
summary()
```


##### Returns


`str`  
A formatted summary string.


------------------------------------------------------------------------


#### to_contract()


Build a Contract object from the imported data.


Usage

``` python
to_contract(name="imported_contract", **kwargs)
```


##### Parameters


`name: str = ``"imported_contract"`  
Name for the created [Contract](Contract.md#pointblank.Contract).

`**kwargs: Any`  
Additional keyword arguments passed to the [Contract](Contract.md#pointblank.Contract) constructor.


##### Returns


`Contract`  
A Contract object with schema and steps derived from the import.


------------------------------------------------------------------------


#### to_python()


Generate Python code string for the validation workflow.


Usage

``` python
to_python()
```


##### Returns


`str`  
Python source code that recreates the validation.


------------------------------------------------------------------------


#### to_validate()


Build a Validate object from the imported contract.


Usage

``` python
to_validate(data, **kwargs)
```


##### Parameters


`data: Any`  
The data table to validate.

`**kwargs: Any`  
Additional keyword arguments passed to the Validate constructor.


##### Returns


`Validate`  
A Validate object with all imported checks applied (not yet interrogated).


------------------------------------------------------------------------


#### to_yaml()


Generate Pointblank YAML configuration string.


Usage

``` python
to_yaml()
```


##### Returns


`str`  
YAML representation of the validation.
