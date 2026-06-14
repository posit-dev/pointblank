## validate_sdtm_structure()


Validate the structural conformance of a dataset against an SDTM domain template.


Usage

``` python
validate_sdtm_structure(
    data,
    domain,
    strict=False,
)
```


Checks required variables, variable ordering, data types, and domainvalue consistency. Does not interrogate but rather returns a dict of findings.


## Parameters


`data: Any`  
A DataFrame (Pandas, Polars) to check.

`domain: str`  
SDTM domain code (e.g., `"DM"`, `"AE"`). This is case-insensitive.

`strict: bool = ``False`  
If `True`, also report missing Expected variables and unknown variables.


## Returns


`dict`  
A dictionary with keys:

- "domain": the domain code
- "valid": `True` if no required violations found
- "missing_required": list of missing required variable names
- "missing_expected": list of missing expected variable names (strict only)
- "unknown_variables": list of column names not in the template (strict only)
- "domain_mismatch": `True` if `DOMAIN` column doesn't match expected value
- "issues": list of human-readable issue strings
