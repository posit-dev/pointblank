## SDTMDomainTemplate


Structural template for an SDTM domain.


Usage

``` python
SDTMDomainTemplate(
    domain,
    label,
    description,
    domain_class,
    repeating,
    variables=list(),
    natural_keys=list()
)
```


## Parameters


`domain: str`  
Two-character domain code (e.g., `"DM"`, `"AE"`, `"LB"`).

`label: str`  
Domain label (e.g., `"Demographics"`, `"Adverse Events"`).

`description: str`  
Brief description of the domain's purpose.

`domain_class: str`  
SDTM observation class: `"Special Purpose"`, `"Events"`, `"Interventions"`, or `"Findings"`.

`repeating: bool`  
Whether the domain is a repeating (multi-row per subject) domain.

`variables: list[SDTMVariableSpec] = list()`    
Ordered list of variable specifications.

`natural_keys: list[str] = list()`    
List of variable names that form the natural key.


## Attributes

| Name | Description |
|----|----|
| [expected_variables](#expected_variables) | Get names of all expected (Exp core) variables. |
| [identifier_variables](#identifier_variables) | Get names of all Identifier-role variables. |
| [required_variables](#required_variables) | Get names of all required variables. |

------------------------------------------------------------------------


#### expected_variables


Get names of all expected (Exp core) variables.


`expected_variables: list[str]`


------------------------------------------------------------------------


#### identifier_variables


Get names of all Identifier-role variables.


`identifier_variables: list[str]`


------------------------------------------------------------------------


#### required_variables


Get names of all required variables.


`required_variables: list[str]`


## Methods

| Name | Description |
|----|----|
| [get_variable()](#get_variable) | Get a variable spec by name. |

------------------------------------------------------------------------


#### get_variable()


Get a variable spec by name.


Usage

``` python
get_variable(name)
```
