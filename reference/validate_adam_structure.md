# validate_adam_structure()

Validate structural conformance of a dataset against an ADaM template.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_adam_templates.py#L799-L880)

``` python
validate_adam_structure(
    data,
    dataset,
    strict=False,
)
```

## Parameters

`data: Any`  
A DataFrame (pandas, polars) to check.

`dataset: str`  
ADaM dataset name (e.g., `"ADSL"`, `"BDS"`, `"ADAE"`, `"ADTTE"`). This is case-insensitive.

`strict: bool = False`  
If True, also reports missing conditional variables and unknown variables.

## Returns

`dict`  
Validation results with keys:

- "dataset": the dataset name
- "dataset_class": ADaM class
- "valid": True if no required violations found
- "missing_required": list of missing required variable names
- "missing_conditional": list of missing conditionally required variables (strict)
- "unknown_variables": list of unknown column names (strict)
- "population_flags_found": list of population flag variables present
- "issues": list of human-readable issue strings
