# from_spss()

Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) from SPSS-style user-defined missing values.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L629-L694)

``` python
from_spss(
    missing_values,
    labels=None,
    null_is_missing=True,
    null_reason="unknown",
    description="SPSS user-defined missing values",
)
```

SPSS supports up to 3 user-defined missing values per variable (plus a range). Pass the missing values (and optionally their value labels) to build a spec. Reason labels are derived from the labels when available, otherwise a `"missing_<value>"` placeholder is used. A range of missing values can't be given, so list the values in the range that occur in the data.

## Parameters

`missing_values: list`  
The sentinel values that SPSS marks as missing for the variable (e.g., `[-99, -98]`).

`labels: dict[Any, str] | None = None`  
Optional mapping of sentinel value to human-readable label (e.g., `{-99: "Refused"}`). Labels are slugified into reason identifiers (e.g., `"Not asked"` becomes `"not_asked"`).

`null_is_missing: bool = True`  
Whether actual null values should also be treated as missing. Default is `True`.

`null_reason: str = "unknown"`  
The reason label for actual null values. Default is `"unknown"`.

`description: str | None = "SPSS user-defined missing values"`  
Optional description. Default identifies the spec as SPSS user-defined missing values.

## Returns

`MissingSpec`  
A spec built from the SPSS missing values.

## Examples

``` python
import pointblank as pb

spss_missing = pb.MissingSpec.from_spss(
    missing_values=[-99, -98, 99],
    labels={-99: "Not asked", -98: "Refused"},
)

spss_missing.reasons
```

    {-99: 'not_asked', -98: 'refused', 99: 'missing_99'}

The labels become the reasons `"not_asked"` and `"refused"`. The value `99` has no label, so its reason is the placeholder `"missing_99"`. To read the missing values and labels straight from an SPSS file, use [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) and [`from_variable_metadata()`](../reference/pointblank.MissingSpec.from_variable_metadata.md) instead.
