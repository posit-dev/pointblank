## get_sdtm_domain()


Get the SDTM template for a specific domain.


Usage

``` python
get_sdtm_domain(domain)
```


## Parameters


`domain: str`  
Two-character domain code (e.g., `"DM"`, `"AE"`, `"LB"`, `"VS"`). This is case-insensitive.


## Returns


`SDTMDomainTemplate`  
The structural template for the domain.


## Raises


`KeyError`  
If the domain is not supported.


## Examples

``` python
from pointblank.metadata._sdtm_templates import get_sdtm_domain

dm = get_sdtm_domain("DM")
print(dm.required_variables)
# ['STUDYID', 'DOMAIN', 'USUBJID', 'SUBJID', 'ARMCD', 'ARM', 'COUNTRY']
```
