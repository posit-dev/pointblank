# from_cdisc_null_flavors()

Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) for the standard HL7/CDISC *null flavors*.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L445-L522)

``` python
from_cdisc_null_flavors(
    null_is_missing=True,
    null_reason="no_information",
    description="CDISC/HL7 null flavors",
)
```

Clinical data uses standardized null flavor codes to record *why* a value is absent (e.g., `"NASK"` for "not asked", `"UNK"` for "unknown"). This returns a ready-to-use spec that maps the 14 standard codes to snake_case reason labels (`"NI"` to `"no_information"`, `"NA"` to `"not_applicable"`, `"ASKU"` to `"asked_but_unknown"`, and so on). It also defines three categories:

- `"unknown"`: `no_information`, `unknown`, `asked_but_unknown`, and `temporarily_unavailable`
- `"not_applicable"`: `not_applicable`, `not_asked`, and `not_present`
- `"boundary"`: `positive_infinity` and `negative_infinity`

The codes `"OTH"`, `"MSK"`, `"DER"`, `"QS"`, and `"TRC"` aren't in any category. The codes are strings, so the spec is meant for character columns. Note that `"NA"` is one of them: in a column where `"NA"` is a real value, use a spec of your own instead.

## Parameters

`null_is_missing: bool = True`  
Whether actual null values should also be treated as missing. Default is `True`.

`null_reason: str = "no_information"`  
The reason label for actual null values. Default is `"no_information"`, which places nulls in the `"unknown"` category.

`description: str | None = "CDISC/HL7 null flavors"`  
Optional description. Default identifies the spec as CDISC/HL7 null flavors.

## Returns

`MissingSpec`  
A spec with the standard null flavor codes.

## Examples

``` python
import pointblank as pb

cdisc_missing = pb.MissingSpec.from_cdisc_null_flavors()

print(cdisc_missing.reason_for("NASK"))
print(cdisc_missing.values_for_category("not_applicable"))
print(cdisc_missing.reason_for(None))
```

    not_asked
    ['NA', 'NASK', 'NP']
    no_information

`"NASK"` maps to `"not_asked"`, and the `"not_applicable"` category covers the three codes for a value that doesn't apply. A null gets the `"no_information"` reason, the same as the `"NI"` code. Pass the spec to a check with `missing=` so that these codes aren't tested as if they were real values.

## See Also

[from_cdisc()](../reference/pointblank.MissingSpec.from_cdisc.md)  
A shorter name for this method.
