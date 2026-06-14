## ADaMVariableSpec


Specification for a single variable in an ADaM dataset template.


Usage

``` python
ADaMVariableSpec(
    name,
    label,
    dtype,
    core="Perm",
    required=False,
    max_length=None,
    controlled_term=None,
    source=None,
    condition=None,
    is_population_flag=False
)
```


## Parameters


`name: str`  
Variable name (e.g., `"USUBJID"`, `"AVAL"`, `"PARAMCD"`).

`label: str`  
Variable label (e.g., `"Unique Subject Identifier"`).

`dtype: str`  
Expected data type (`"Char"` or `"Num"`).

`core: str = ``"Perm"`  
ADaM core designation: `"Req"` (required), `"Cond"` (conditionally required), or `"Perm"` (permissible).

`required: bool = ``False`  
Whether the variable is unconditionally required.

`max_length: int | None = None`  
Maximum character length for Char variables.

`controlled_term: str | None = None`  
Name of the associated controlled terminology codelist.

`source: str | None = None`  
Traceability: expected source (e.g., `"SDTM.DM"`, `"Derived"`).

`condition: str | None = None`  
For conditional variables, describes when they are required.

`is_population_flag: bool = ``False`  
Whether this is a population flag variable (e.g., `"SAFFL"`, `"ITTFL"`).
