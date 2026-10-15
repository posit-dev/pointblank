# CodelistEntry


A single entry in a codelist (controlled terminology).


Usage

``` python
CodelistEntry(
    value,
    label,
    description=None,
    synonyms=None,
    is_deprecated=False,
    is_extended=False,
)
```


## Parameters


`value: Any`  
The coded value.

`label: str`  
Human-readable label for the value.

`description: str | None = None`  
Extended description of this entry.

`synonyms: list[str] | None = None`  
Alternative terms for this entry.

`is_deprecated: bool = ``False`  
Whether this entry is deprecated.

`is_extended: bool = ``False`  
Whether this entry is a sponsor extension of an extensible codelist (declared in Define-XML with `def:ExtendedValue="Yes"`).
