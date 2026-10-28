# MetadataPackage

A collection of [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) objects from a multi-dataset source.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L493-L584)

``` python
MetadataPackage(
    name=None,
    items=dict(),
    description=None,
    version=None,
)
```

Used for multi-domain CDISC studies, Frictionless Data Packages, etc.

## Parameters

`name: str | None = None`  
Package name/identifier.

`items: dict[str, MetadataImport] = dict()`  
Named [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) objects.

`description: str | None = None`  
Description of the package.

`version: str | None = None`  
Package/study version.

## Methods

| Name | Description |
|----|----|
| [get_domain()](#get_domain) | Get metadata for a specific domain/dataset. |
| [keys()](#keys) | Get the names of all datasets/domains. |
| [values()](#values) | Get all MetadataImport objects. |
| [summary()](#summary) | Return a human-readable summary of the package. |

### get_domain()

Get metadata for a specific domain/dataset.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L536-L558)

``` python
get_domain(name)
```

#### Parameters

`name: str`  
Domain or dataset name (e.g., `"DM"`, `"AE"`).

#### Returns

`MetadataImport`  
The metadata for the named domain.

#### Raises

`KeyError`  
If no domain with that name exists.

### keys()

Get the names of all datasets/domains.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L528-L530)

``` python
keys()
```

### values()

Get all MetadataImport objects.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L532-L534)

``` python
values()
```

### summary()

Return a human-readable summary of the package.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L560-L578)

``` python
summary()
```

#### Returns

`str`  
Formatted summary string.
