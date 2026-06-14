## validate_adam()


Generate a comprehensive ADaM validation workflow for a dataset.


Usage

``` python
validate_adam(
    data,
    dataset,
    study_id=None,
    check_population_flags=True,
    check_bds_structure=True,
    check_traceability=True,
    label=None,
    **kwargs
)
```


Creates a Validate object with checks for:

- Required variables present and non-null
- Population flag values (Y/N only, no nulls in flag columns)
- BDS structure: PARAMCD, PARAM, AVAL consistency
- ADTTE: CNSR values (0 or 1), AVAL \>= 0
- TRT01P/TRT01A consistency (non-null, single value per subject in ADSL)
- Traceability variable presence


## Parameters


`data: Any`  
The DataFrame to validate (pandas or polars).

`dataset: str`  
ADaM dataset name (e.g., `"ADSL"`, `"BDS"`, `"ADAE"`, `"ADTTE"`). This is case-insensitive.

`study_id: str | None = None`  
Optional study identifier for the validation label.

`check_population_flags: bool = ``True`  
If `True`, validate population flag columns (Y/N values only).

`check_bds_structure: bool = ``True`  
If `True`, validate BDS-specific structure (`PARAMCD`/`PARAM`/`AVAL`).

`check_traceability: bool = ``True`  
If `True`, check that traceability variables are non-null when present.

`label: str | None = None`  
Custom label for the Validate object.

`**kwargs: Any`  
Additional keyword arguments passed to the Validate constructor.


## Returns


`Validate`  
A configured (but not yet interrogated) Validate object.
