# sdtm_to_metadata()

Convert an SDTM domain template to a [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) object.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_sdtm_validate.py#L35-L156)

``` python
sdtm_to_metadata(
    domain,
    study_id=None,
)
```

The SDTM domain templates (see [`get_sdtm_domain()`](../reference/get_sdtm_domain.md#pointblank.get_sdtm_domain)) have their own classes, [`SDTMDomainTemplate`](../reference/SDTMDomainTemplate.md#pointblank.SDTMDomainTemplate) and [`SDTMVariableSpec`](../reference/SDTMVariableSpec.md#pointblank.SDTMVariableSpec). The [sdtm_to_metadata()](../reference/sdtm_to_metadata.md#pointblank.sdtm_to_metadata) function converts a template to a [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport), the same object that [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) returns for a metadata file. That gives the template the methods of a [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport), such as [`to_schema()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_schema) and [`get_variable()`](../reference/MetadataImport.md#pointblank.MetadataImport.get_variable), and lets it be exported with [`export_metadata()`](../reference/export_metadata.md#pointblank.export_metadata).

Each variable in the template becomes a [`VariableMetadata`](../reference/VariableMetadata.md#pointblank.VariableMetadata) with the same name, label, role, `required` flag, maximum length and controlled terminology name. SDTM's `"Num"` type becomes the `"Float64"` dtype and `"Char"` becomes `"String"`. The dataset name and label come from the domain, and the source version is `"IG 3.4"`. The template names the codelist of a variable (as in `controlled_term="SEX"`) but doesn't contain the codelist itself, so the result has no codelists. Calling `import_metadata(format="cdisc_sdtm", domain=...)` gives the same result.

## Parameters

`domain: str`  
The SDTM domain code, such as `"DM"`, `"AE"` or `"LB"`. This is case-insensitive.

`study_id: str | None = None`  
A study identifier to record in the metadata's `study_id`. The default is `None`.

## Returns

`MetadataImport`  
The metadata for the domain, with `source_format="cdisc_sdtm"`.

## Raises

`KeyError`  
If there is no template for the domain. See [`list_sdtm_domains()`](../reference/list_sdtm_domains.md#pointblank.list_sdtm_domains) for the domains that have one.

## Examples

Let's convert the template for the Demographics (DM) domain:

``` python
import pointblank as pb

dm_meta = pb.sdtm_to_metadata("DM", study_id="STUDY01")

print(dm_meta.dataset_label)
print(dm_meta.source_version)
print(dm_meta.variable_names[:6])
```

    Demographics
    IG 3.4
    ['STUDYID', 'DOMAIN', 'USUBJID', 'SUBJID', 'RFSTDTC', 'RFENDTC']

The metadata has all 26 variables of the template, in the template's order. Each variable keeps what the template says about it:

``` python
sex = dm_meta.get_variable("SEX")

print(sex.dtype, sex.required, sex.max_length, sex.controlled_term)
```

    String True 2 SEX

`SEX` is a required character variable of at most two characters, with values from the `SEX` codelist of CDISC Controlled Terminology. The metadata can be turned into a [`Schema`](../reference/Schema.md#pointblank.Schema) for use in [`Validate.col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match):

``` python
schema = dm_meta.to_schema()

schema.columns[:4]
```

    [('STUDYID', 'String'),
     ('DOMAIN', 'String'),
     ('USUBJID', 'String'),
     ('SUBJID', 'String')]

The schema lists every variable of the template, including the expected and permissible ones. A real DM dataset usually has only some of them, so a schema check against all 26 is stricter than SDTM requires. For checking a dataset against the template, [`validate_sdtm()`](../reference/validate_sdtm.md#pointblank.validate_sdtm) (which only checks the variables that are present) is usually the better fit.

## See Also

[get_sdtm_domain()](../reference/get_sdtm_domain.md#pointblank.get_sdtm_domain)  
Get the template that is converted.

[validate_sdtm()](../reference/validate_sdtm.md#pointblank.validate_sdtm)  
Build a validation plan from an SDTM domain template.

[adam_to_metadata()](../reference/adam_to_metadata.md#pointblank.adam_to_metadata)  
The equivalent for ADaM dataset templates.

[MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport)  
The object that is returned.
