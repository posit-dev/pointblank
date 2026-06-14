## validate_sdtm()


Generate a comprehensive SDTM validation workflow for a dataset.


Usage

``` python
validate_sdtm(
    data,
    domain,
    study_id=None,
    check_dates=True,
    check_lengths=True,
    label=None,
    **kwargs
)
```


Creates a [Validate](Validate.md#pointblank.Validate) object with checks for:

- Schema conformance (required variables present with correct types)
- Required variables are non-null
- Variable length constraints (for Char variables)
- DOMAIN column value matches expected domain code
- ISO 8601 date format for -DTC timing variables
- Sequence number positivity and uniqueness per subject


## Parameters


`data: Any`  
The DataFrame to validate (Pandas or Polars).

`domain: str`  
SDTM domain code (e.g., `"DM"`, `"AE"`, `"LB"`). This is case-insensitive.

`study_id: str | None = None`  
Optional study identifier for the validation label.

`check_dates: bool = ``True`  
If `True`, validate ISO 8601 format for -DTC variables.

`check_lengths: bool = ``True`  
If `True`, validate string length constraints.

`label: str | None = None`  
Custom label for the [Validate](Validate.md#pointblank.Validate) object. Defaults to `"SDTM {domain} Validation"`.

`**kwargs: Any`  
Additional keyword arguments passed to the [Validate](Validate.md#pointblank.Validate) constructor.


## Returns


<a href="Validate.html#pointblank.Validate" class="gdls-link gdls-code"><code>Validate</code></a>  
A configured (but not yet interrogated) [Validate](Validate.md#pointblank.Validate) object.


## Examples

``` python
import pointblank as pb
from pointblank.metadata._sdtm_validate import validate_sdtm

validation = validate_sdtm(dm_data, domain="DM").interrogate()
```
