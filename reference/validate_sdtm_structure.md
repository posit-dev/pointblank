# validate_sdtm_structure()

Check a dataset's variables against an SDTM domain template.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_sdtm_templates.py#L1580-L1735)

``` python
validate_sdtm_structure(
    data,
    domain,
    strict=False,
)
```

The [validate_sdtm_structure()](../reference/validate_sdtm_structure.md#pointblank.validate_sdtm_structure) function is a quick check of a dataset's structure, run before (or instead of) a full validation with [`validate_sdtm()`](../reference/validate_sdtm.md#pointblank.validate_sdtm). It checks that every required variable of the domain is a column of the dataset, and that the `DOMAIN` column holds only the domain code. It returns the findings as a dictionary straight away, without building a [`Validate`](../reference/Validate.md#pointblank.Validate) plan.

With `strict=True`, two more things are reported: the expected variables (core designation `"Exp"`) that are missing, and the columns that aren't variables in the template. These don't affect `"valid"`. SDTM expects an expected variable to be in the dataset even if all of its values are null, but a missing one is treated here as something to review rather than an error. A column that isn't in the template may still be a valid SDTM variable, since the templates only cover the common variables of each domain.

The column order and the data types aren't checked.

## Parameters

`data: Any`  
The dataset to check, as a Polars or Pandas DataFrame.

`domain: str`  
The SDTM domain code, such as `"DM"` or `"AE"`. This is case-insensitive.

`strict: bool = False`  
Whether to also report missing expected variables and columns that aren't in the template. The default is `False`.

## Returns

`dict`  
A dictionary with these keys:

- `"domain"`: the domain code, in upper case.
- `"valid"`: `True` if no required variable is missing and `DOMAIN` holds only the domain code.
- `"missing_required"`: the names of the required variables that aren't columns of the data.
- `"missing_expected"`: the names of the expected variables that aren't columns of the data (filled in only with `strict=True`).
- `"unknown_variables"`: the columns that aren't variables of the domain (filled in only with `strict=True`).
- `"domain_mismatch"`: `True` if the `DOMAIN` column has any value (including a null) other than the domain code.
- `"issues"`: a message for each of the problems above.

## Raises

`KeyError`  
If there is no template for the domain.

## Examples

Let's check a small Adverse Events (AE) dataset. It has all of the required AE variables except `AEDECOD`, and its `DOMAIN` column has one value from another domain:

``` python
import pointblank as pb
import polars as pl

ae = pl.DataFrame(
    {
        "STUDYID": ["STUDY01"] * 3,
        "DOMAIN": ["AE", "AE", "CM"],
        "USUBJID": ["STUDY01-001", "STUDY01-001", "STUDY01-002"],
        "AESEQ": [1, 2, 1],
        "AETERM": ["HEADACHE", "NAUSEA", "RASH"],
        "AESEV": ["MILD", "MILD", "MODERATE"],
        "AEFLAG": ["Y", "N", "Y"],
    }
)

result = pb.validate_sdtm_structure(ae, domain="AE")

print(result["valid"])
for issue in result["issues"]:
    print(issue)
```

    False
    Required variable 'AEDECOD' is missing
    DOMAIN column contains unexpected values: ['AE', 'CM'] (expected ['AE'])

The dataset isn't valid, and the issues say why. The extra column `AEFLAG` isn't mentioned, since that's only reported in strict mode:

``` python
result = pb.validate_sdtm_structure(ae, domain="AE", strict=True)

print(result["missing_expected"])
print(result["unknown_variables"])
```

    ['AEBODSYS', 'AESER', 'AEACN', 'AEREL', 'AEOUT', 'AESTDTC', 'AEENDTC']
    ['AEFLAG']

Strict mode adds the expected AE variables that the dataset doesn't have, such as `AESTDTC`, and the column that isn't an AE variable. `AESEV` is a permissible variable of the domain, so it isn't reported.

## See Also

[validate_sdtm()](../reference/validate_sdtm.md#pointblank.validate_sdtm)  
Build a validation plan from an SDTM domain template.

[get_sdtm_domain()](../reference/get_sdtm_domain.md#pointblank.get_sdtm_domain)  
Get the template that the dataset is checked against.

[validate_adam_structure()](../reference/validate_adam_structure.md#pointblank.validate_adam_structure)  
The equivalent for ADaM datasets.
