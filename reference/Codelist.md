# Codelist

A controlled terminology / value set from an external standard.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L50-L112)

``` python
Codelist(
    name,
    codes=list(),
    label=None,
    version=None,
    source=None,
    extensible=False,
)
```

Represents a set of permitted values from standards like CDISC controlled terminology, SPSS value labels, DDI code schemes, etc.

## Parameters

`name: str`  
Codelist identifier.

`codes: list[CodelistEntry] = list()`  
List of codelist entries.

`label: str | None = None`  
Human-readable name for the codelist.

`version: str | None = None`  
Version of the terminology.

`source: str | None = None`  
Where this codelist comes from (e.g., `"CDISC CT 2024-09"`).

`extensible: bool = False`  
Whether additional values beyond the codelist are allowed.

## Methods

| Name | Description |
|----|----|
| [to_set()](#to_set) | Get the list of valid values (for col_vals_in_set). |
| [to_dict()](#to_dict) | Get a value → label mapping. |
| [extended_values()](#extended_values) | Get the values that extend an extensible codelist (sponsor extensions). |

### to_set()

Get the list of valid values (for col_vals_in_set).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L80-L88)

``` python
to_set()
```

#### Returns

`list`  
All non-deprecated values in the codelist.

### to_dict()

Get a value → label mapping.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L101-L109)

``` python
to_dict()
```

#### Returns

`dict`  
Mapping of value to human-readable label.

### extended_values()

Get the values that extend an extensible codelist (sponsor extensions).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L90-L99)

``` python
extended_values()
```

#### Returns

`list`  
The values of the entries marked as extended (e.g., with `def:ExtendedValue="Yes"` in Define-XML).
