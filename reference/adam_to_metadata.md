# adam_to_metadata()

Convert an ADaM dataset template to a [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) object.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_adam_validate.py#L19-L123)

``` python
adam_to_metadata(
    dataset,
    study_id=None,
)
```

The ADaM dataset templates (see [`get_adam_dataset()`](../reference/get_adam_dataset.md#pointblank.get_adam_dataset)) have their own classes, [`ADaMDatasetTemplate`](../reference/ADaMDatasetTemplate.md#pointblank.ADaMDatasetTemplate) and [`ADaMVariableSpec`](../reference/ADaMVariableSpec.md#pointblank.ADaMVariableSpec). The [adam_to_metadata()](../reference/adam_to_metadata.md#pointblank.adam_to_metadata) function converts a template to a [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport), the same object that [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) returns for a metadata file. That gives the template the methods of a [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport), such as [`to_schema()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_schema) and [`get_variable()`](../reference/MetadataImport.md#pointblank.MetadataImport.get_variable), and lets it be exported with [`export_metadata()`](../reference/export_metadata.md#pointblank.export_metadata).

Each variable in the template becomes a [`VariableMetadata`](../reference/VariableMetadata.md#pointblank.VariableMetadata) with the same name, label, `required` flag, maximum length and controlled terminology name. The variable's core designation (`"Req"`, `"Cond"` or `"Perm"`) goes in `cdisc_role`, and its expected source (such as `"SDTM.DM"`) goes in `adam_derivation`. ADaM's `"Num"` type becomes the `"Float64"` dtype and `"Char"` becomes `"String"`, so numeric dates such as `TRTSDT` (which ADaM stores as numbers of days) are `"Float64"`. The source version is `"IG 1.1"`. Calling `import_metadata(format="cdisc_adam", dataset=...)` gives the same result.

## Parameters

`dataset: str`  
The name of the ADaM dataset template: `"ADSL"`, `"BDS"`, `"ADAE"` or `"ADTTE"`. This is case-insensitive.

`study_id: str | None = None`  
A study identifier to record in the metadata's `study_id`. The default is `None`.

## Returns

`MetadataImport`  
The metadata for the dataset, with `source_format="cdisc_adam"`.

## Raises

`KeyError`  
If there is no template with that name.

## Examples

Let's convert the template for the subject-level analysis dataset, ADSL:

``` python
import pointblank as pb

adsl_meta = pb.adam_to_metadata("ADSL", study_id="STUDY01")

print(adsl_meta.dataset_label)
print(len(adsl_meta.variables), "variables")
```

    Subject Level Analysis Dataset
    30 variables

Each variable keeps what the template says about it, with the ADaM core designation in `cdisc_role`:

``` python
for name in ["USUBJID", "TRT01P", "SAFFL", "TRTSDT"]:
    var = adsl_meta.get_variable(name)
    print(f"{name:8}{var.dtype:8}{var.cdisc_role:5}{var.adam_derivation}")
```

    USUBJID String  Req  SDTM.DM
    TRT01P  String  Req  None
    SAFFL   String  Cond None
    TRTSDT  Float64 Cond Derived from SDTM.EX

`USUBJID` and `TRT01P` are required, while `SAFFL` (the safety population flag) and `TRTSDT` (the treatment start date) are conditionally required. `USUBJID` comes from the SDTM Demographics (DM) domain and `TRTSDT` is derived from Exposure (EX). Only the required variables get `required=True`, so the conditional ones don't get the null checks that [`to_validate()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_validate) adds for required variables.

## See Also

[get_adam_dataset()](../reference/get_adam_dataset.md#pointblank.get_adam_dataset)  
Get the template that is converted.

[validate_adam()](../reference/validate_adam.md#pointblank.validate_adam)  
Build a validation plan from an ADaM dataset template.

[sdtm_to_metadata()](../reference/sdtm_to_metadata.md#pointblank.sdtm_to_metadata)  
The equivalent for SDTM domain templates.

[MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport)  
The object that is returned.
