# ADaMDatasetTemplate

The structural template for an ADaM dataset.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_adam_templates.py#L93-L266)

``` python
ADaMDatasetTemplate(
    name,
    label,
    description,
    dataset_class,
    variables=list(),
    natural_keys=list(),
)
```

An [ADaMDatasetTemplate](../reference/ADaMDatasetTemplate.md#pointblank.ADaMDatasetTemplate) describes the structure of one type of CDISC Analysis Data Model (ADaM) dataset: its label and class, the variables that form its natural key, and the specifications of its variables (as [`ADaMVariableSpec`](../reference/ADaMVariableSpec.md#pointblank.ADaMVariableSpec) objects, in order). Templates are obtained with [`get_adam_dataset()`](../reference/get_adam_dataset.md#pointblank.get_adam_dataset).

## Parameters

`name: str`  
The template name: `"ADSL"`, `"BDS"`, `"ADAE"` or `"ADTTE"`.

`label: str`  
The dataset label, such as `"Subject Level Analysis Dataset"`.

`description: str`  
A short description of what the dataset holds and what a record is.

`dataset_class: str`  
The ADaM dataset class, which [`validate_adam()`](../reference/validate_adam.md#pointblank.validate_adam) uses to choose the checks for the type of dataset. In the templates, it's the same as `name`.

`variables: list[ADaMVariableSpec] = list()`  
The specifications of the dataset's variables, in order. The default is an empty list.

`natural_keys: list[str] = list()`  
The names of the variables that together identify a record. The default is an empty list.

## Examples

Let's look at the template for the Basic Data Structure (BDS), the structure of analysis datasets such as ADVS (vital signs) and ADLB (laboratory results):

``` python
import pointblank as pb

bds = pb.get_adam_dataset("BDS")

print(bds.label, "|", bds.dataset_class)
print(bds.description)
print(bds.natural_keys)
```

    Basic Data Structure | BDS
    One or more records per subject per analysis parameter per analysis timepoint. Used for efficacy, lab, vital signs, etc.
    ['STUDYID', 'USUBJID', 'PARAMCD', 'AVISIT', 'ADT']

A BDS dataset has a record for each analysis parameter (`PARAMCD`) at each visit and date, for each subject. The template's properties list the required and conditionally required variables:

``` python
print(bds.required_variables)
print(bds.conditional_variables)
```

    ['STUDYID', 'USUBJID', 'PARAMCD', 'PARAM', 'AVAL']
    ['TRT01P', 'AVALC', 'BASE', 'CHG', 'AVISIT', 'ADT', 'ABLFL']

Every BDS dataset needs the identifiers, the parameter and the analysis value. The conditional variables, such as the baseline value `BASE` and the change from baseline `CHG`, are needed only for some analyses.

## See Also

[get_adam_dataset()](../reference/get_adam_dataset.md#pointblank.get_adam_dataset)  
Get the template for a type of dataset.

[ADaMVariableSpec](../reference/ADaMVariableSpec.md#pointblank.ADaMVariableSpec)  
The specification of a variable in the template.

[SDTMDomainTemplate](../reference/SDTMDomainTemplate.md#pointblank.SDTMDomainTemplate)  
The equivalent for SDTM domains.

## Attributes

| Name | Description |
|----|----|
| [required_variables](#required_variables) | The names of the required variables, in order. |
| [conditional_variables](#conditional_variables) | The names of the conditionally required variables, in order. |
| [population_flags](#population_flags) | The names of the population flag variables, in order. |

### required_variables

The names of the required variables, in order.

`required_variables: list[str]`

Required variables (core designation `"Req"`) must be in every dataset of this type. [`validate_adam()`](../reference/validate_adam.md#pointblank.validate_adam) adds a null check for each one, and [`validate_adam_structure()`](../reference/validate_adam_structure.md#pointblank.validate_adam_structure) reports the ones that are missing.

#### Examples

``` python
import pointblank as pb

pb.get_adam_dataset("ADTTE").required_variables
```

    ['STUDYID', 'USUBJID', 'PARAMCD', 'PARAM', 'AVAL', 'STARTDT', 'ADT', 'CNSR']

A time-to-event dataset (ADTTE) needs the parameter, the time to the event (`AVAL`), the start and event dates, and the censoring indicator (`CNSR`). This is a property, so it's used without parentheses.

### conditional_variables

The names of the conditionally required variables, in order.

`conditional_variables: list[str]`

A conditionally required variable (core designation `"Cond"`) is needed only in some situations, which the variable's `condition` describes. [`validate_adam_structure()`](../reference/validate_adam_structure.md#pointblank.validate_adam_structure) reports the ones that are missing when `strict=True`.

#### Examples

``` python
import pointblank as pb

adsl = pb.get_adam_dataset("ADSL")

for name in adsl.conditional_variables[:4]:
    print(name, "->", adsl.get_variable(name).condition)
```

    TRT01A -> Required if actual differs from planned
    TRT01PN -> Required if treatment mapped to numeric
    TRTSDT -> Required if treatment start used in derivations
    TRTEDT -> Required if treatment end used in derivations

These are the first four conditional variables of ADSL, with the condition for each one. This is a property, so it's used without parentheses.

### population_flags

The names of the population flag variables, in order.

`population_flags: list[str]`

Population flags, such as `SAFFL` (safety) and `ITTFL` (intent-to-treat), say whether each subject is in an analysis population, with the values `"Y"` and `"N"`. Only the ADSL template has them.

#### Examples

``` python
import pointblank as pb

pb.get_adam_dataset("ADSL").population_flags
```

    ['SAFFL', 'ITTFL', 'EFFFL', 'RANDFL', 'ENRLFL', 'PPROTFL', 'COMPLFL']

[`validate_adam()`](../reference/validate_adam.md#pointblank.validate_adam) checks each of these flags that is a column of the data. This is a property, so it's used without parentheses.

## Methods

| Name                            | Description                          |
|---------------------------------|--------------------------------------|
| [get_variable()](#get_variable) | Get the specification of a variable. |

### get_variable()

Get the specification of a variable.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_adam_templates.py#L230-L266)

``` python
get_variable(name)
```

#### Parameters

`name: str`  
The variable name. This is case-sensitive, and ADaM variable names are in upper case.

#### Returns

`ADaMVariableSpec | None`  
The specification of the variable, or `None` if the template has no variable with that name.

#### Examples

``` python
import pointblank as pb

adtte = pb.get_adam_dataset("ADTTE")

adtte.get_variable("CNSR")
```

    ADaMVariableSpec(name='CNSR', label='Censor', dtype='Num', core='Req', required=True, max_length=None, controlled_term=None, source=None, condition=None, is_population_flag=False)

`CNSR` is the required censoring indicator of a time-to-event dataset. A name that isn't in the template gives `None`:

``` python
print(adtte.get_variable("AESEQ"))
```

    None

`AESEQ` belongs to the ADAE template, not to ADTTE.
