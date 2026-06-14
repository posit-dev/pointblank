## MetadataPackage


A collection of [MetadataImport](MetadataImport.md#pointblank.MetadataImport) objects from a multi-dataset source.


Usage

``` python
MetadataPackage()
```


Used for multi-domain CDISC studies, Frictionless Data Packages, etc.


## Parameters


`name: str | None = None`  
Package name/identifier.

`items: dict[str, MetadataImport] = dict()`    
Named [MetadataImport](MetadataImport.md#pointblank.MetadataImport) objects.

`description: str | None = None`  
Description of the package.

`version: str | None = None`  
Package/study version.


## Methods

| Name | Description |
|----|----|
| [get_domain()](#get_domain) | Get metadata for a specific domain/dataset. |
| [keys()](#keys) | Get the names of all datasets/domains. |
| [summary()](#summary) | Return a human-readable summary of the package. |
| [values()](#values) | Get all MetadataImport objects. |

------------------------------------------------------------------------


#### get_domain()


Get metadata for a specific domain/dataset.


Usage

``` python
get_domain(name)
```


##### Parameters


`name: str`  
Domain or dataset name (e.g., `"DM"`, `"AE"`).


##### Returns


`MetadataImport`  
The metadata for the named domain.


##### Raises


`KeyError`  
If no domain with that name exists.


------------------------------------------------------------------------


#### keys()


Get the names of all datasets/domains.


Usage

``` python
keys()
```


------------------------------------------------------------------------


#### summary()


Return a human-readable summary of the package.


Usage

``` python
summary()
```


##### Returns


`str`  
Formatted summary string.


------------------------------------------------------------------------


#### values()


Get all MetadataImport objects.


Usage

``` python
values()
```
