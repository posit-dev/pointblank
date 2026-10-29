# list_adam_datasets()

List the ADaM dataset templates.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_adam_templates.py#L998-L1030)

``` python
list_adam_datasets()
```

These are the names that [`get_adam_dataset()`](../reference/get_adam_dataset.md#pointblank.get_adam_dataset), [`validate_adam()`](../reference/validate_adam.md#pointblank.validate_adam), [`validate_adam_structure()`](../reference/validate_adam_structure.md#pointblank.validate_adam_structure) and [`adam_to_metadata()`](../reference/adam_to_metadata.md#pointblank.adam_to_metadata) accept.

## Returns

`list[str]`  
The template names, in alphabetical order.

## Examples

Let's list the templates, together with each one's label:

``` python
import pointblank as pb

for name in pb.list_adam_datasets():
    print(f"{name:7}{pb.get_adam_dataset(name).label}")
```

    ADAE   Adverse Event Analysis Dataset
    ADSL   Subject Level Analysis Dataset
    ADTTE  Time-to-Event Analysis Dataset
    BDS    Basic Data Structure

BDS is a structure rather than a single dataset, so its template is used for any BDS dataset in a study, such as ADVS or ADLB.

## See Also

[get_adam_dataset()](../reference/get_adam_dataset.md#pointblank.get_adam_dataset)  
Get a dataset template.

[list_sdtm_domains()](../reference/list_sdtm_domains.md#pointblank.list_sdtm_domains)  
The equivalent for SDTM domains.
