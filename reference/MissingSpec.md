# MissingSpec

Specification for structured missing values in a column.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L43-L771)

``` python
MissingSpec(
    reasons,
    categories=None,
    null_is_missing=True,
    null_reason="unknown",
    description=None,
)
```

Real-world data rarely encodes missingness as a single `null` value. Survey data distinguishes *refused* from *don't know* from *not applicable*; clinical data uses codes like `"NOT DONE"`; statistical packages use sentinel values such as `-99`, `".A"`, or `""`. A [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) captures these sentinel values, the *reason* each one represents, and how they should be handled during validation and analysis.

This brings the idea of *structured missingness* (a missing value carries a reason for its absence) into Pointblank's runtime validation layer. Without a spec, a sentinel like `-99` is just a number: a range check on an `age` column fails it, and a null check passes it, so both checks give the wrong answer. Declaring the sentinels once and passing the spec to each check keeps the codes out of the value checks, while still letting you check the codes themselves.

A spec can be used in these places:

- the `missing=` parameter of the `col_vals_*()` methods (such as [`Validate.col_vals_between()`](../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between)), where values that the spec treats as missing pass the check, so only the real values are tested
- [`Validate.col_pct_missing()`](../reference/Validate.col_pct_missing.md#pointblank.Validate.col_pct_missing), to limit how much of a column is missing, overall or for one reason or category
- [`Validate.col_missing_coded()`](../reference/Validate.col_missing_coded.md#pointblank.Validate.col_missing_coded), [`Validate.col_missing_only_coded()`](../reference/Validate.col_missing_only_coded.md#pointblank.Validate.col_missing_only_coded), and [`Validate.col_missing_consistent()`](../reference/Validate.col_missing_consistent.md#pointblank.Validate.col_missing_consistent), to check how the missing values are coded
- [`missing_vals_tbl()`](../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl), whose `missing=` mapping of column names to specs gives a breakdown of each column's missing values by reason
- YAML configurations, through a top-level `missing_specs:` block or an inline `missing:` mapping on a step

The factory methods build common specs: [`from_sas()`](../reference/pointblank.MissingSpec.from_sas.md) and [`from_spss()`](../reference/pointblank.MissingSpec.from_spss.md) for statistical packages, [`from_cdisc_null_flavors()`](../reference/pointblank.MissingSpec.from_cdisc_null_flavors.md) for clinical data, and [`from_variable_metadata()`](../reference/pointblank.MissingSpec.from_variable_metadata.md) for variables read with [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata).

## Parameters

`reasons: dict[Any, str]`  
A dictionary mapping sentinel values to reason labels. Keys are the actual values present in the data (e.g., `-99`, `"NA"`, `".A"`), and they must have the same type as the column: use `"-99"` rather than `-99` for a string column. Values are reason identifiers (e.g., `"refused"`, `"not_asked"`). Several sentinel values can share a reason. The dictionary can be empty if `null_is_missing=True`.

`categories: dict[str, list[str]] | None = None`  
Optional grouping of reasons into categories (e.g., an `"item_nonresponse"` category that groups `"refused"` and `"dont_know"`). Useful for aggregate reporting and for checking missingness rates by category. Each value is a list of reason labels that appear in `reasons` (or `null_reason`, when `null_is_missing=True`); any other label raises a `ValueError`. Default is `None`.

`null_is_missing: bool = True`  
Whether actual null values (`None`, and `NaN` in floating-point columns) should also be treated as missing, with the reason given by `null_reason`. Default is `True`. With `False`, nulls are left to the method's usual null handling (such as `na_pass=` in the `col_vals_*()` methods).

`null_reason: str = "unknown"`  
The reason label assigned to actual null values when `null_is_missing=True`. Default is `"unknown"`.

`description: str | None = None`  
Optional human-readable description of the overall missingness pattern. Default is `None`.

## Returns

`MissingSpec`  
A missing-value specification that can be passed to validation methods (via `missing=`) and to [`missing_vals_tbl()`](../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl).

## Examples

Suppose a survey records the respondent's age, with three codes for an age that wasn't given: `-99` when the question wasn't asked, `-98` when the respondent refused, and `-97` when they didn't know. The last two are both *item nonresponse* (the question was asked but not answered), so we'll group them in a category:

``` python
import pointblank as pb
import polars as pl

age_missing = pb.MissingSpec(
    reasons={-99: "not_asked", -98: "refused", -97: "dont_know"},
    categories={"item_nonresponse": ["refused", "dont_know"]},
)

print(age_missing.sentinel_values())
print(age_missing.reason_for(-98))
print(age_missing.values_for_category("item_nonresponse"))
```

    [-99, -98, -97]
    refused
    [-98, -97]

The spec lists its sentinel values, gives the reason for any of them, and finds the values in a category. Here's some survey data that uses these codes, along with a null and an age of `150` that's a genuine error:

``` python
survey = pl.DataFrame({"age": [34, -98, 51, -99, None, 27, 150, -97, 45, 62]})

pb.preview(survey)
```

|     | age *Int64* |
|-----|-------------|
| 1   | 34          |
| 2   | -98         |
| 3   | 51          |
| 4   | -99         |
| 5   | None        |
| 6   | 27          |
| 7   | 150         |
| 8   | -97         |
| 9   | 45          |
| 10  | 62          |

We'll check the ages with and without the spec, and then limit the item nonresponse to 10% of the rows:

``` python
validation = (
    pb.Validate(data=survey)
    .col_vals_between(columns="age", left=0, right=120)
    .col_vals_between(columns="age", left=0, right=120, missing=age_missing)
    .col_pct_missing(
        columns="age", missing=age_missing, category="item_nonresponse", max_pct=0.1
    )
    .interrogate()
)

validation
```

[TABLE]

Step 1 has 5 failing rows: the three codes and the null fail the range check along with the `150`, so the real error is lost among the codes. With `missing=age_missing`, step 2 lets the codes and the null pass and only the `150` fails. Step 3 looks at the codes instead of the ages: 2 of the 10 rows (`-98` and `-97`) are item nonresponse, which is 20% and over the 10% limit, so it fails. Together the steps separate bad values from values that are missing for a known reason, and check each on its own terms.

## See Also

[col_pct_missing()](../reference/Validate.col_pct_missing.md#pointblank.Validate.col_pct_missing)  
Limit the share of a column that is missing.

[col_missing_coded()](../reference/Validate.col_missing_coded.md#pointblank.Validate.col_missing_coded)  
Check that every missing value has a declared code.

[missing_vals_tbl()](../reference/missing_vals_tbl.md#pointblank.missing_vals_tbl)  
Show each column's missing values by reason.

[import_metadata()](../reference/import_metadata.md#pointblank.import_metadata)  
Read variable metadata, including missing-value codes, from a file.

## Functions

| Name | Description |
|----|----|
| [from_cdisc_null_flavors()](pointblank.MissingSpec.from_cdisc_null_flavors.md) | Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) for the standard HL7/CDISC *null flavors*. |
| [from_cdisc()](pointblank.MissingSpec.from_cdisc.md) | Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) for the standard HL7/CDISC null flavors (short alias). |
| [from_sas()](pointblank.MissingSpec.from_sas.md) | Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) for SAS special missing values. |
| [from_spss()](pointblank.MissingSpec.from_spss.md) | Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) from SPSS-style user-defined missing values. |
| [from_variable_metadata()](pointblank.MissingSpec.from_variable_metadata.md) | Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) from an imported variable's metadata. |
| [is_missing()](pointblank.MissingSpec.is_missing.md) | Check whether a value should be considered missing under this spec. |
| [reason_for()](pointblank.MissingSpec.reason_for.md) | Get the reason label for a specific value. |
| [sentinel_values()](pointblank.MissingSpec.sentinel_values.md) | Get all sentinel values that encode missingness. |
| [reasons_list()](pointblank.MissingSpec.reasons_list.md) | Get the distinct reason labels defined by this spec. |
| [values_for_reason()](pointblank.MissingSpec.values_for_reason.md) | Get all sentinel values that correspond to a given reason. |
| [values_for_category()](pointblank.MissingSpec.values_for_category.md) | Get all sentinel values whose reason falls in a given category. |
