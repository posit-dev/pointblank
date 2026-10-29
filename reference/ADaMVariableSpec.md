# ADaMVariableSpec

The specification of one variable in an ADaM dataset template.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_adam_templates.py#L16-L90)

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
    is_population_flag=False,
)
```

An [`ADaMDatasetTemplate`](../reference/ADaMDatasetTemplate.md#pointblank.ADaMDatasetTemplate) holds an [ADaMVariableSpec](../reference/ADaMVariableSpec.md#pointblank.ADaMVariableSpec) for each variable of its dataset, in its `variables` list. The specification gives the variable's label, type and core designation, the condition under which a conditionally required variable is needed, where its values come from, and whether it is a population flag. [`validate_adam()`](../reference/validate_adam.md#pointblank.validate_adam) uses `required` and `is_population_flag` to decide which checks to make.

## Parameters

`name: str`  
The variable name, such as `"USUBJID"`, `"PARAMCD"` or `"AVAL"`.

`label: str`  
The variable label, such as `"Unique Subject Identifier"`.

`dtype: str`  
The data type: `"Char"` for character variables or `"Num"` for numeric ones (including dates, which ADaM stores as numbers).

`core: str = "Perm"`  
The core designation: `"Req"` (required), `"Cond"` (conditionally required: needed when the condition in `condition` holds) or `"Perm"` (permissible). The default is `"Perm"`.

`required: bool = False`  
Whether the variable is required in every dataset of this type. In the templates, this is `True` exactly when `core` is `"Req"`. The default is `False`.

`max_length: int | None = None`  
The maximum length of the values of a character variable. The default is `None` (no maximum).

`controlled_term: str | None = None`  
The name of the controlled terminology codelist for the variable's values, such as `"NY"` for the population flags. The default is `None`.

`source: str | None = None`  
Where the variable's values come from, such as `"SDTM.DM"` (copied from the Demographics domain) or `"Derived from SDTM.EX"`. The default is `None` (not specified).

`condition: str | None = None`  
For a conditionally required variable, a description of when it is required, such as `"Required if safety population defined"`. The default is `None`.

`is_population_flag: bool = False`  
Whether the variable is a population flag (such as `SAFFL` or `ITTFL`), whose values must be `"Y"` or `"N"`. The default is `False`.

## Examples

Specifications are usually taken from a template. Here is the one for the safety population flag, `SAFFL`, in the ADSL template:

``` python
import pointblank as pb

saffl = pb.get_adam_dataset("ADSL").get_variable("SAFFL")

saffl
```

    ADaMVariableSpec(name='SAFFL', label='Safety Population Flag', dtype='Char', core='Cond', required=False, max_length=1, controlled_term='NY', source=None, condition='Required if safety population defined', is_population_flag=True)

`SAFFL` is a one-character population flag with values from the `NY` codelist. It is conditionally required, and `condition` says when: whenever the study defines a safety population.

## See Also

[ADaMDatasetTemplate](../reference/ADaMDatasetTemplate.md#pointblank.ADaMDatasetTemplate)  
The template that holds the specifications of a dataset's variables.

[SDTMVariableSpec](../reference/SDTMVariableSpec.md#pointblank.SDTMVariableSpec)  
The equivalent for SDTM domain templates.
