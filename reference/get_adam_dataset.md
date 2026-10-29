# get_adam_dataset()

Get the template for a type of ADaM dataset.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_adam_templates.py#L925-L995)

``` python
get_adam_dataset(name)
```

Pointblank has templates for four types of CDISC Analysis Data Model (ADaM) dataset: ADSL (the subject-level analysis dataset), BDS (the Basic Data Structure, used by datasets such as ADVS and ADLB), ADAE (adverse events) and ADTTE (time to event). Each template is an [`ADaMDatasetTemplate`](../reference/ADaMDatasetTemplate.md#pointblank.ADaMDatasetTemplate): the dataset's label and class, the variables that form its natural key, and an [`ADaMVariableSpec`](../reference/ADaMVariableSpec.md#pointblank.ADaMVariableSpec) for each of its variables. The templates are what [`validate_adam()`](../reference/validate_adam.md#pointblank.validate_adam), [`validate_adam_structure()`](../reference/validate_adam_structure.md#pointblank.validate_adam_structure) and [`adam_to_metadata()`](../reference/adam_to_metadata.md#pointblank.adam_to_metadata) work from.

The templates cover the common variables of each type of dataset rather than every variable that the ADaM Implementation Guide describes. Each call returns a new template object, so changing one doesn't affect later calls.

## Parameters

`name: str`  
The template name: `"ADSL"`, `"BDS"`, `"ADAE"` or `"ADTTE"`. This is case-insensitive.

## Returns

`ADaMDatasetTemplate`  
The template for the dataset.

## Raises

`KeyError`  
If there is no template with that name. The error message lists the templates, as does [`list_adam_datasets()`](../reference/list_adam_datasets.md#pointblank.list_adam_datasets).

## Examples

Let's get the template for the subject-level analysis dataset:

``` python
import pointblank as pb

adsl = pb.get_adam_dataset("adsl")

print(adsl.label)
print(adsl.natural_keys)
print(adsl.required_variables)
```

    Subject Level Analysis Dataset
    ['STUDYID', 'USUBJID']
    ['STUDYID', 'USUBJID', 'SUBJID', 'SITEID', 'TRT01P']

ADSL has one record per subject, so the study and subject identifiers are its natural key. Besides the identifiers, the planned treatment `TRT01P` is required. The template also lists the population flags:

``` python
adsl.population_flags
```

    ['SAFFL', 'ITTFL', 'EFFFL', 'RANDFL', 'ENRLFL', 'PPROTFL', 'COMPLFL']

These flags (safety, intent-to-treat, efficacy and so on) are conditionally required or permissible, and [`validate_adam()`](../reference/validate_adam.md#pointblank.validate_adam) checks that the ones in a dataset hold only `"Y"` and `"N"`.

## See Also

[list_adam_datasets()](../reference/list_adam_datasets.md#pointblank.list_adam_datasets)  
List the dataset templates.

[ADaMDatasetTemplate](../reference/ADaMDatasetTemplate.md#pointblank.ADaMDatasetTemplate)  
The template that is returned.

[validate_adam()](../reference/validate_adam.md#pointblank.validate_adam)  
Build a validation plan from an ADaM dataset template.

[get_sdtm_domain()](../reference/get_sdtm_domain.md#pointblank.get_sdtm_domain)  
The equivalent for SDTM domains.
