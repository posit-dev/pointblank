## MissingValueCode


A structured missing value definition from an external standard.


Usage

``` python
MissingValueCode(
    value,
    label,
    category=None,
    reason=None,
)
```


In SPSS, SAS, and clinical data, missing values carry meaning (`REFUSED`, `NOT_APPLICABLE`, `NOT_ASKED`, etc.).


## Parameters


`value: Any`  
The sentinel value (e.g., `-99`, `".A"`, `""`).

`label: str`  
What this missing code means.

`category: str | None = None`  
Category of missingness (e.g., `"system_missing"`, `"user_missing"`).

`reason: str | None = None`  
Why data is missing.
