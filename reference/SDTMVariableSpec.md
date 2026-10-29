# SDTMVariableSpec

The specification of one variable in an SDTM domain template.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_sdtm_templates.py#L16-L83)

``` python
SDTMVariableSpec(
    name,
    label,
    dtype,
    role,
    required=False,
    max_length=None,
    controlled_term=None,
    core="Perm",
)
```

An [`SDTMDomainTemplate`](../reference/SDTMDomainTemplate.md#pointblank.SDTMDomainTemplate) holds an [SDTMVariableSpec](../reference/SDTMVariableSpec.md#pointblank.SDTMVariableSpec) for each variable of its domain, in its `variables` list. The specification gives what the SDTM Implementation Guide says about the variable: its label, its type, its role in the domain, its core designation, and the controlled terminology for its values. [`validate_sdtm()`](../reference/validate_sdtm.md#pointblank.validate_sdtm) uses `required`, `dtype`, `max_length` and `role` to decide which checks to make.

## Parameters

`name: str`  
The variable name, such as `"STUDYID"` or `"AESTDTC"`.

`label: str`  
The variable label, such as `"Study Identifier"`.

`dtype: str`  
The SDTM data type: `"Char"` for character variables or `"Num"` for numeric ones.

`role: str`  
The variable's SDTM role. The templates use `"Identifier"` (variables that identify the study, subject, domain or record), `"Topic"` (the focus of the observation, such as `AETERM`), `"Qualifier"` (variables that describe the observation) and `"Timing"` (dates, times and study days).

`required: bool = False`  
Whether the variable is required: it must be in the dataset and can't be null. In the templates, this is `True` exactly when `core` is `"Req"`. The default is `False`.

`max_length: int | None = None`  
The maximum length of the values of a character variable. The default is `None` (no maximum, as for numeric variables).

`controlled_term: str | None = None`  
The name of the controlled terminology codelist for the variable's values, such as `"SEX"` or `"NY"`. The default is `None`.

`core: str = "Perm"`  
The core designation: `"Req"` (required), `"Exp"` (expected: in the dataset, but values may be null) or `"Perm"` (permissible: included when there is data for it). The default is `"Perm"`.

## Examples

Specifications are usually taken from a template. Here is the one for `AESER` in the Adverse Events (AE) template:

``` python
import pointblank as pb

aeser = pb.get_sdtm_domain("AE").get_variable("AESER")

aeser
```

    SDTMVariableSpec(name='AESER', label='Serious Event', dtype='Char', role='Qualifier', required=False, max_length=2, controlled_term='NY', core='Exp')

`AESER` (whether the adverse event was serious) is an expected character variable of at most two characters, with values from the `NY` (No Yes Response) codelist.

## See Also

[SDTMDomainTemplate](../reference/SDTMDomainTemplate.md#pointblank.SDTMDomainTemplate)  
The template that holds the specifications of a domain's variables.

[ADaMVariableSpec](../reference/ADaMVariableSpec.md#pointblank.ADaMVariableSpec)  
The equivalent for ADaM dataset templates.
