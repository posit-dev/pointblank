# from_sas()

Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) for SAS special missing values.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L559-L627)

``` python
from_sas(
    reasons=None,
    include_underscore=True,
    null_is_missing=True,
    null_reason="system_missing",
    description="SAS special missing values",
)
```

SAS encodes missingness with `"."` (system missing), `"._"`, and `".A"` through `".Z"` (27 user-defined missing codes). This returns a spec covering all of them: `"."` and `"._"` get the reason `"system_missing"`, and each letter code gets a placeholder reason such as `"user_missing_a"`. Give the codes your study actually uses a meaningful reason with `reasons=`. The codes are strings, so the spec matches columns that hold them as text.

## Parameters

`reasons: dict[str, str] | None = None`  
Optional mapping of specific SAS missing codes to custom reason labels (e.g., `{".A": "not_applicable", ".B": "below_detection"}`). These override the defaults.

`include_underscore: bool = True`  
Whether to include the `"._"` special missing code. Default is `True`.

`null_is_missing: bool = True`  
Whether actual null values should also be treated as missing. Default is `True`.

`null_reason: str = "system_missing"`  
The reason label for actual null values. Default is `"system_missing"`.

`description: str | None = "SAS special missing values"`  
Optional description. Default identifies the spec as SAS special missing values.

## Returns

`MissingSpec`  
A spec covering the SAS special missing values.

## Examples

``` python
import pointblank as pb

sas_missing = pb.MissingSpec.from_sas(
    reasons={".A": "not_applicable", ".B": "below_detection"}
)

print(len(sas_missing.sentinel_values()))
print(sas_missing.reason_for(".A"))
print(sas_missing.reason_for(".C"))
print(sas_missing.values_for_reason("system_missing"))
```

    28
    not_applicable
    user_missing_c
    ['.', '._']

The spec has 28 codes: `"."`, `"._"`, and the 26 letters. `".A"` has the reason given in `reasons=`, while `".C"` keeps its placeholder. Nulls also get the `"system_missing"` reason, but they aren't listed by [values_for_reason()](../reference/pointblank.MissingSpec.values_for_reason.md) because they aren't sentinel values.
