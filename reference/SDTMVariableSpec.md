## SDTMVariableSpec


Specification for a single variable in an SDTM domain template.


Usage

``` python
SDTMVariableSpec(
    name,
    label,
    dtype,
    role,
    required=False,
    max_length=None,
    controlled_term=None,
    core="Perm"
)
```


## Parameters


`name: str`  
Variable name (e.g., `"STUDYID"`, `"USUBJID"`).

`label: str`  
Variable label (e.g., `"Study Identifier"`).

`dtype: str`  
Expected data type (`"Char"` or `"Num"`).

`role: str`  
SDTM role: `"Identifier"`, `"Topic"`, `"Qualifier"`, `"Timing"`, `"Rule"`, or `"Record Qualifier"`.

`required: bool = ``False`  
Whether the variable is required (`Req="Yes"` in IG).

`max_length: int | None = None`  
Maximum character length for Char variables.

`controlled_term: str | None = None`  
Name of the associated controlled terminology codelist.

`core: str = ``"Perm"`  
SDTM core designation: `"Req"`, `"Exp"`, or `"Perm"`.
