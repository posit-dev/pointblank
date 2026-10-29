# get_sdtm_domain()

Get the template for an SDTM domain.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_sdtm_templates.py#L1471-L1541)

``` python
get_sdtm_domain(domain)
```

Pointblank has templates for eight domains of the CDISC Study Data Tabulation Model (SDTM), based on the SDTM Implementation Guide (SDTMIG) 3.4. Each template is an [`SDTMDomainTemplate`](../reference/SDTMDomainTemplate.md#pointblank.SDTMDomainTemplate): the domain's label and observation class, the variables that form its natural key, and an [`SDTMVariableSpec`](../reference/SDTMVariableSpec.md#pointblank.SDTMVariableSpec) for each of its variables, giving the variable's label, type, role, maximum length, controlled terminology and core designation. The templates are what [`validate_sdtm()`](../reference/validate_sdtm.md#pointblank.validate_sdtm), [`validate_sdtm_structure()`](../reference/validate_sdtm_structure.md#pointblank.validate_sdtm_structure) and [`sdtm_to_metadata()`](../reference/sdtm_to_metadata.md#pointblank.sdtm_to_metadata) work from.

The templates cover the common variables of each domain rather than every variable in the SDTMIG. Each call returns a new template object, so changing one doesn't affect later calls.

## Parameters

`domain: str`  
The two-letter domain code, such as `"DM"`, `"AE"`, `"LB"` or `"VS"`. This is case-insensitive.

## Returns

`SDTMDomainTemplate`  
The template for the domain.

## Raises

`KeyError`  
If there is no template for the domain. The error message lists the domains that have one, as does [`list_sdtm_domains()`](../reference/list_sdtm_domains.md#pointblank.list_sdtm_domains).

## Examples

Let's get the template for the Vital Signs (VS) domain:

``` python
import pointblank as pb

vs = pb.get_sdtm_domain("vs")

print(vs.label, "|", vs.domain_class, "| repeating:", vs.repeating)
print(vs.natural_keys)
```

    Vital Signs | Findings | repeating: True
    ['STUDYID', 'USUBJID', 'VSTESTCD', 'VSDTC', 'VSTPTNUM']

VS is a Findings domain with many records per subject, one for each test at each time point, so its natural key includes the test code, the date-time and the time point number. The template's properties list the variables by their core designation:

``` python
print(vs.required_variables)
print(vs.expected_variables)
```

    ['STUDYID', 'DOMAIN', 'USUBJID', 'VSSEQ', 'VSTESTCD', 'VSTEST']
    ['VSORRES', 'VSORRESU', 'VSSTRESC', 'VSSTRESN', 'VSSTRESU', 'VSBLFL', 'VISITNUM', 'VSDTC']

The required variables must be in every VS dataset and can't be null. The expected ones should be in the dataset but may be null, and the domain's other variables are permissible: they're included only when there is data for them.

## See Also

[list_sdtm_domains()](../reference/list_sdtm_domains.md#pointblank.list_sdtm_domains)  
List the domains that have a template.

[SDTMDomainTemplate](../reference/SDTMDomainTemplate.md#pointblank.SDTMDomainTemplate)  
The template that is returned.

[validate_sdtm()](../reference/validate_sdtm.md#pointblank.validate_sdtm)  
Build a validation plan from an SDTM domain template.

[get_adam_dataset()](../reference/get_adam_dataset.md#pointblank.get_adam_dataset)  
The equivalent for ADaM datasets.
