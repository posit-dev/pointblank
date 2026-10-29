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

The factory methods build common specs: [`from_sas()`](../reference/MissingSpec.md#pointblank.MissingSpec.from_sas) and [`from_spss()`](../reference/MissingSpec.md#pointblank.MissingSpec.from_spss) for statistical packages, [`from_cdisc_null_flavors()`](../reference/MissingSpec.md#pointblank.MissingSpec.from_cdisc_null_flavors) for clinical data, and [`from_variable_metadata()`](../reference/MissingSpec.md#pointblank.MissingSpec.from_variable_metadata) for variables read with [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata).

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

## Methods

| Name | Description |
|----|----|
| [from_cdisc_null_flavors()](#from_cdisc_null_flavors) | Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) for the standard HL7/CDISC *null flavors*. |
| [from_cdisc()](#from_cdisc) | Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) for the standard HL7/CDISC null flavors (short alias). |
| [from_sas()](#from_sas) | Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) for SAS special missing values. |
| [from_spss()](#from_spss) | Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) from SPSS-style user-defined missing values. |
| [from_variable_metadata()](#from_variable_metadata) | Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) from an imported variable's metadata. |
| [is_missing()](#is_missing) | Check whether a value should be considered missing under this spec. |
| [reason_for()](#reason_for) | Get the reason label for a specific value. |
| [sentinel_values()](#sentinel_values) | Get all sentinel values that encode missingness. |
| [reasons_list()](#reasons_list) | Get the distinct reason labels defined by this spec. |
| [values_for_reason()](#values_for_reason) | Get all sentinel values that correspond to a given reason. |
| [values_for_category()](#values_for_category) | Get all sentinel values whose reason falls in a given category. |

### from_cdisc_null_flavors()

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

#### Parameters

`null_is_missing: bool = True`  
Whether actual null values should also be treated as missing. Default is `True`.

`null_reason: str = "no_information"`  
The reason label for actual null values. Default is `"no_information"`, which places nulls in the `"unknown"` category.

`description: str | None = "CDISC/HL7 null flavors"`  
Optional description. Default identifies the spec as CDISC/HL7 null flavors.

#### Returns

`MissingSpec`  
A spec with the standard null flavor codes.

#### Examples

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

#### See Also

[from_cdisc()](../reference/MissingSpec.md#pointblank.MissingSpec.from_cdisc)  
A shorter name for this method.

### from_cdisc()

Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) for the standard HL7/CDISC null flavors (short alias).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L525-L557)

``` python
from_cdisc(**kwargs)
```

This is a shorter name for [`from_cdisc_null_flavors()`](../reference/MissingSpec.md#pointblank.MissingSpec.from_cdisc_null_flavors), and it takes the same keyword arguments (`null_is_missing=`, `null_reason=`, and `description=`).

#### Parameters

`**kwargs: Any`  
Keyword arguments passed on to [from_cdisc_null_flavors()](../reference/MissingSpec.md#pointblank.MissingSpec.from_cdisc_null_flavors).

#### Returns

`MissingSpec`  
A spec with the standard null flavor codes.

#### Examples

``` python
import pointblank as pb

cdisc_missing = pb.MissingSpec.from_cdisc(null_reason="unknown")

cdisc_missing.reason_for(None)
```

    'unknown'

Here nulls get the `"unknown"` reason rather than the default `"no_information"`, which still places them in the `"unknown"` category.

### from_sas()

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

#### Parameters

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

#### Returns

`MissingSpec`  
A spec covering the SAS special missing values.

#### Examples

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

The spec has 28 codes: `"."`, `"._"`, and the 26 letters. `".A"` has the reason given in `reasons=`, while `".C"` keeps its placeholder. Nulls also get the `"system_missing"` reason, but they aren't listed by [values_for_reason()](../reference/MissingSpec.md#pointblank.MissingSpec.values_for_reason) because they aren't sentinel values.

### from_spss()

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

#### Parameters

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

#### Returns

`MissingSpec`  
A spec built from the SPSS missing values.

#### Examples

``` python
import pointblank as pb

spss_missing = pb.MissingSpec.from_spss(
    missing_values=[-99, -98, 99],
    labels={-99: "Not asked", -98: "Refused"},
)

spss_missing.reasons
```

    {-99: 'not_asked', -98: 'refused', 99: 'missing_99'}

The labels become the reasons `"not_asked"` and `"refused"`. The value `99` has no label, so its reason is the placeholder `"missing_99"`. To read the missing values and labels straight from an SPSS file, use [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) and [`from_variable_metadata()`](../reference/MissingSpec.md#pointblank.MissingSpec.from_variable_metadata) instead.

### from_variable_metadata()

Create a [MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec) from an imported variable's metadata.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L696-L771)

``` python
from_variable_metadata(
    variable,
    null_is_missing=True,
    null_reason="unknown",
)
```

This works with a [`VariableMetadata`](../reference/VariableMetadata.md#pointblank.VariableMetadata) object (as produced by [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) for SPSS, Stata, and SAS files). It reads the variable's `missing_values` and derives reason labels from `missing_value_labels` or `value_labels` when available. Values with neither get a `"missing_<value>"` placeholder. The [to_missing_spec()](../reference/VariableMetadata.md#pointblank.VariableMetadata.to_missing_spec) method of [VariableMetadata](../reference/VariableMetadata.md#pointblank.VariableMetadata) does the same thing.

#### Parameters

`variable: Any`  
A variable-metadata object exposing `missing_values` and (optionally) `missing_value_labels` / `value_labels` attributes.

`null_is_missing: bool = True`  
Whether actual null values should also be treated as missing. Default is `True`.

`null_reason: str = "unknown"`  
The reason label for actual null values. Default is `"unknown"`.

#### Returns

`MissingSpec | None`  
A spec built from the variable's missing values, or `None` if the variable declares no missing values.

#### Examples

In practice the variable comes from an imported file, but we'll build one here to show how it's read:

``` python
import pointblank as pb

q5 = pb.VariableMetadata(
    name="q5",
    missing_values=[-99, -98],
    value_labels={-99: "Not asked", -98: "Refused", 1: "Yes", 2: "No"},
)

spec = pb.MissingSpec.from_variable_metadata(q5)

print(spec.reasons)
print(spec.description)
```

    {-99: 'not_asked', -98: 'refused'}
    Imported missing values for 'q5'

Only the declared missing values become sentinels, and their value labels become the reasons. The value labels for `1` and `2` are ignored because those are real answers. A variable that declares no missing values gives `None` rather than a spec, so check for that before passing the result to `missing=`.

### is_missing()

Check whether a value should be considered missing under this spec.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L304-L340)

``` python
is_missing(value)
```

This applies the same rule that validation methods use with `missing=`: a value is missing if it's one of the declared sentinel values, or if it's null and the spec treats nulls as missing.

#### Parameters

`value: Any`  
A value from the data.

#### Returns

`bool`  
`True` if `value` is a declared sentinel value, or if `value` is null (`None` or a float `NaN`) and `null_is_missing=True`.

#### Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(reasons={-99: "not_asked"}, null_is_missing=False)

print(spec.is_missing(-99))
print(spec.is_missing(None))
print(spec.is_missing(42))
```

    True
    False
    False

Only the sentinel `-99` counts as missing here. With `null_is_missing=False` a null doesn't count either, so in a `col_vals_*()` method it would be handled by `na_pass=` instead.

### reason_for()

Get the reason label for a specific value.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L269-L302)

``` python
reason_for(value)
```

#### Parameters

`value: Any`  
A value from the data.

#### Returns

`str | None`  
The reason label if `value` is a declared sentinel value, `null_reason` if `value` is null (`None` or a float `NaN`) and `null_is_missing=True`, or `None` if the value is not considered missing.

#### Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"})

print(spec.reason_for(-98))
print(spec.reason_for(None))
print(spec.reason_for(42))
```

    refused
    unknown
    None

The sentinel `-98` gives its declared reason, and a null gets the spec's `null_reason` (`"unknown"` by default). An ordinary value like `42` isn't missing, so its reason is `None`.

### sentinel_values()

Get all sentinel values that encode missingness.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L239-L267)

``` python
sentinel_values()
```

The sentinel values are the keys of `reasons`, in the order they were given. This is the list to use when you need the codes themselves, for example to filter them out of a table before computing a summary statistic.

#### Returns

`list`  
The keys of `reasons` (the actual values in the data that represent missingness). Note that this does *not* include `None` even when `null_is_missing=True`; use [`is_missing()`](../reference/MissingSpec.md#pointblank.MissingSpec.is_missing) to test individual values.

#### Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused", -97: "refused"})

spec.sentinel_values()
```

    [-99, -98, -97]

All three codes are listed, including the two that share the `"refused"` reason. Nulls aren't listed even though the spec treats them as missing (`null_is_missing=True` is the default), because they aren't values that appear in `reasons`.

### reasons_list()

Get the distinct reason labels defined by this spec.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L411-L439)

``` python
reasons_list()
```

#### Returns

`list[str]`  
The distinct reason labels (in first-seen order), including `null_reason` when `null_is_missing=True`.

#### Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused", -97: "refused"})

spec.reasons_list()
```

    ['not_asked', 'refused', 'unknown']

The `"refused"` reason appears once even though two codes use it, and `"unknown"` comes last because it's the `null_reason` for nulls. These are the reason labels that `col_pct_missing(reason=)` accepts and that `categories` can refer to.

### values_for_reason()

Get all sentinel values that correspond to a given reason.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L342-L372)

``` python
values_for_reason(reason)
```

#### Parameters

`reason: str`  
A reason label.

#### Returns

`list`  
All sentinel values mapped to `reason`, or an empty list if no sentinel value has that reason. Nulls are never included, even when `reason` is the spec's `null_reason`.

#### Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(
    reasons={".": "system_missing", "._": "system_missing", ".R": "refused"}
)

spec.values_for_reason("system_missing")
```

    ['.', '._']

Both SAS-style codes for system missing values are returned. This is the reverse of [`reason_for()`](../reference/MissingSpec.md#pointblank.MissingSpec.reason_for), which goes from a value to its reason.

### values_for_category()

Get all sentinel values whose reason falls in a given category.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/missing.py#L374-L409)

``` python
values_for_category(category)
```

#### Parameters

`category: str`  
A category name defined in `categories`.

#### Returns

`list`  
All sentinel values whose reason label is in the given category. Returns an empty list if `categories` is `None` or the category is undefined.

#### Examples

``` python
import pointblank as pb

spec = pb.MissingSpec(
    reasons={-99: "not_asked", -98: "refused", -97: "dont_know"},
    categories={"item_nonresponse": ["refused", "dont_know"]},
)

print(spec.values_for_category("item_nonresponse"))
print(spec.values_for_category("design"))
```

    [-98, -97]
    []

The `"item_nonresponse"` category covers the `"refused"` and `"dont_know"` reasons, so its values are `-98` and `-97`. The `"design"` category isn't defined, and the result is an empty list rather than an error, so check the category name if you get no values back.
