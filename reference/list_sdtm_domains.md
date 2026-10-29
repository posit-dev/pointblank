# list_sdtm_domains()

List the SDTM domains that have a template.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_sdtm_templates.py#L1544-L1577)

``` python
list_sdtm_domains()
```

These are the domain codes that [`get_sdtm_domain()`](../reference/get_sdtm_domain.md#pointblank.get_sdtm_domain), [`validate_sdtm()`](../reference/validate_sdtm.md#pointblank.validate_sdtm), [`validate_sdtm_structure()`](../reference/validate_sdtm_structure.md#pointblank.validate_sdtm_structure) and [`sdtm_to_metadata()`](../reference/sdtm_to_metadata.md#pointblank.sdtm_to_metadata) accept.

## Returns

`list[str]`  
The domain codes, in alphabetical order.

## Examples

Let's list the domains, together with the label and class of each one's template:

``` python
import pointblank as pb

for code in pb.list_sdtm_domains():
    template = pb.get_sdtm_domain(code)
    print(f"{code}  {template.label:26}{template.domain_class}")
```

    AE  Adverse Events            Events
    CM  Concomitant Medications   Interventions
    DM  Demographics              Special Purpose
    DS  Disposition               Events
    EX  Exposure                  Interventions
    LB  Laboratory Test Results   Findings
    MH  Medical History           Events
    VS  Vital Signs               Findings

The eight domains come from all four of the observation classes that the templates use: Special Purpose (DM), Events, Interventions and Findings.

## See Also

[get_sdtm_domain()](../reference/get_sdtm_domain.md#pointblank.get_sdtm_domain)  
Get the template for a domain.

[list_adam_datasets()](../reference/list_adam_datasets.md#pointblank.list_adam_datasets)  
The equivalent for ADaM datasets.
