# SDTMDomainTemplate

The structural template for an SDTM domain.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_sdtm_templates.py#L86-L257)

``` python
SDTMDomainTemplate(
    domain,
    label,
    description,
    domain_class,
    repeating,
    variables=list(),
    natural_keys=list(),
)
```

An [SDTMDomainTemplate](../reference/SDTMDomainTemplate.md#pointblank.SDTMDomainTemplate) describes the structure of one domain of the CDISC Study Data Tabulation Model (SDTM): its label and observation class, whether it has many records per subject, the variables that form its natural key, and the specifications of its variables (as [`SDTMVariableSpec`](../reference/SDTMVariableSpec.md#pointblank.SDTMVariableSpec) objects, in the SDTM order). Templates are obtained with [`get_sdtm_domain()`](../reference/get_sdtm_domain.md#pointblank.get_sdtm_domain).

## Parameters

`domain: str`  
The two-letter domain code, such as `"DM"`, `"AE"` or `"LB"`.

`label: str`  
The domain label, such as `"Demographics"` or `"Adverse Events"`.

`description: str`  
A short description of what the domain holds.

`domain_class: str`  
The SDTM observation class: `"Special Purpose"`, `"Events"`, `"Interventions"` or `"Findings"`.

`repeating: bool`  
Whether the domain can have more than one record per subject. Only DM, which has one record per subject, isn't repeating.

`variables: list[SDTMVariableSpec] = list()`  
The specifications of the domain's variables, in order. The default is an empty list.

`natural_keys: list[str] = list()`  
The names of the variables that together identify a record. The default is an empty list.

## Examples

Let's look at the template for the Adverse Events (AE) domain:

``` python
import pointblank as pb

ae = pb.get_sdtm_domain("AE")

print(ae.label, "|", ae.domain_class, "| repeating:", ae.repeating)
print(ae.description)
print(len(ae.variables), "variables")
```

    Adverse Events | Events | repeating: True
    Adverse events reported during the study.
    28 variables

The variables are in the `variables` list. Here are the first eight, with their type, role and core designation:

``` python
for var in ae.variables[:8]:
    print(f"{var.name:9}{var.dtype:5}{var.role:12}{var.core}")
```

    STUDYID  Char Identifier  Req
    DOMAIN   Char Identifier  Req
    USUBJID  Char Identifier  Req
    AESEQ    Num  Identifier  Req
    AEGRPID  Char Identifier  Perm
    AEREFID  Char Identifier  Perm
    AESPID   Char Identifier  Perm
    AETERM   Char Topic       Req

The template's properties collect the names of the variables with a given core designation or role, and [`get_variable()`](../reference/SDTMDomainTemplate.md#pointblank.SDTMDomainTemplate.get_variable) gets the specification of one variable.

## See Also

[get_sdtm_domain()](../reference/get_sdtm_domain.md#pointblank.get_sdtm_domain)  
Get the template for a domain.

[SDTMVariableSpec](../reference/SDTMVariableSpec.md#pointblank.SDTMVariableSpec)  
The specification of a variable in the template.

[ADaMDatasetTemplate](../reference/ADaMDatasetTemplate.md#pointblank.ADaMDatasetTemplate)  
The equivalent for ADaM datasets.

## Attributes

| Name | Description |
|----|----|
| [required_variables](#required_variables) | The names of the required variables, in order. |
| [expected_variables](#expected_variables) | The names of the expected variables, in order. |
| [identifier_variables](#identifier_variables) | The names of the variables with the Identifier role, in order. |

### required_variables

The names of the required variables, in order.

`required_variables: list[str]`

Required variables (core designation `"Req"`) must be in every dataset of the domain and can't be null. [`validate_sdtm()`](../reference/validate_sdtm.md#pointblank.validate_sdtm) adds a null check for each one, and [`validate_sdtm_structure()`](../reference/validate_sdtm_structure.md#pointblank.validate_sdtm_structure) reports the ones that are missing.

#### Examples

``` python
import pointblank as pb

pb.get_sdtm_domain("DM").required_variables
```

    ['STUDYID',
     'DOMAIN',
     'USUBJID',
     'SUBJID',
     'SITEID',
     'SEX',
     'ARMCD',
     'ARM',
     'COUNTRY']

These are the variables that every Demographics (DM) dataset must have. This is a property, so it's used without parentheses.

### expected_variables

The names of the expected variables, in order.

`expected_variables: list[str]`

Expected variables (core designation `"Exp"`) should be in every dataset of the domain, but their values may be null when the data wasn't collected or doesn't apply. [`validate_sdtm_structure()`](../reference/validate_sdtm_structure.md#pointblank.validate_sdtm_structure) reports the ones that are missing when `strict=True`.

#### Examples

``` python
import pointblank as pb

pb.get_sdtm_domain("DM").expected_variables
```

    ['RFSTDTC',
     'RFENDTC',
     'RFXSTDTC',
     'RFXENDTC',
     'AGE',
     'AGEU',
     'RACE',
     'ACTARMCD',
     'ACTARM']

For Demographics (DM), these include the reference dates (such as `RFSTDTC`), age, race and the actual arm. This is a property, so it's used without parentheses.

### identifier_variables

The names of the variables with the Identifier role, in order.

`identifier_variables: list[str]`

Identifier variables identify the study, the subject, the domain and the record, as with `STUDYID`, `USUBJID`, `DOMAIN` and the sequence number.

#### Examples

``` python
import pointblank as pb

pb.get_sdtm_domain("AE").identifier_variables
```

    ['STUDYID', 'DOMAIN', 'USUBJID', 'AESEQ', 'AEGRPID', 'AEREFID', 'AESPID']

Besides the identifiers that all domains share, the Adverse Events (AE) template has the sequence number `AESEQ` and the optional group, reference and sponsor identifiers. This is a property, so it's used without parentheses.

## Methods

| Name                            | Description                          |
|---------------------------------|--------------------------------------|
| [get_variable()](#get_variable) | Get the specification of a variable. |

### get_variable()

Get the specification of a variable.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_sdtm_templates.py#L221-L257)

``` python
get_variable(name)
```

#### Parameters

`name: str`  
The variable name. This is case-sensitive, and SDTM variable names are in upper case.

#### Returns

`SDTMVariableSpec | None`  
The specification of the variable, or `None` if the template has no variable with that name.

#### Examples

``` python
import pointblank as pb

dm = pb.get_sdtm_domain("DM")

dm.get_variable("AGE")
```

    SDTMVariableSpec(name='AGE', label='Age', dtype='Num', role='Qualifier', required=False, max_length=None, controlled_term=None, core='Exp')

`AGE` is an expected numeric variable, so it has no maximum length. A name that isn't in the template gives `None`:

``` python
print(dm.get_variable("age"))
```

    None

The lookup is case-sensitive, so the lower-case name doesn't match `AGE`.
