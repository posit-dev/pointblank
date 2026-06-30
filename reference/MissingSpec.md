## MissingSpec


Specification for structured missing values in a column.


Usage

``` python
MissingSpec(
    reasons,
    categories=None,
    null_is_missing=True,
    null_reason="unknown",
    description=None
)
```


Real-world data rarely encodes missingness as a single `null` value. Survey data distinguishes *refused* from *don't know* from *not applicable*; clinical data uses codes like `"NOT DONE"`; statistical packages use sentinel values such as `-99`, `".A"`, or `""`. A [MissingSpec](MissingSpec.md#pointblank.MissingSpec) captures these sentinel values, the *reason* each one represents, and how they should be handled during validation and analysis.

This brings the idea of *structured missingness* (a missing value carries a reason for its absence) into Pointblank's runtime validation layer. Once defined, a [MissingSpec](MissingSpec.md#pointblank.MissingSpec) can be passed to validation methods (via `missing=`) to automatically exclude sentinel values from constraint checks, or used with dedicated methods like <a href="Validate.col_missing_coded.html#pointblank.Validate.col_missing_coded" class="gdls-link"><code>Validate.col_missing_coded()</code></a> and <a href="Validate.col_pct_missing.html#pointblank.Validate.col_pct_missing" class="gdls-link"><code>Validate.col_pct_missing()</code></a>.


## Parameters


`reasons: dict[Any, str]`  
A dictionary mapping sentinel values to reason labels. Keys are the actual values present in the data (e.g., `-99`, `"NA"`, `".A"`). Values are human-readable reason identifiers (e.g., `"refused"`, `"not_asked"`).

`categories: dict[str, list[str]] | None = None`  
Optional grouping of reasons into categories (e.g., an `"item_nonresponse"` category that groups `"refused"` and `"dont_know"`). Useful for aggregate reporting and for checking missingness rates by category. Each value is a list of reason labels that appear in `reasons`. Default is `None`.

`null_is_missing: bool = ``True`  
Whether actual null/`None`/`NaN` values should also be treated as missing (with reason given by [null_reason](MissingSpec.md#pointblank.MissingSpec.null_reason)). Default is `True`.

`null_reason: str = ``"unknown"`  
The reason label assigned to actual null values when `null_is_missing=True`. Default is `"unknown"`.

`description: str | None = None`  
Optional human-readable description of the overall missingness pattern. Default is `None`.


## Returns


`MissingSpec`  
A missing-value specification that can be attached to a `Field` (via `missing=`) or passed to validation methods.


## Examples

Define the missing-value codes for a survey `age` variable:

``` python
import pointblank as pb

age_missing = pb.MissingSpec(
    reasons={
        -99: "not_asked",       # Question wasn't asked to this participant
        -98: "refused",         # Participant declined to answer
        -97: "dont_know",       # Participant didn't know
        -96: "not_applicable",  # Question doesn't apply
    },
    categories={
        "item_nonresponse": ["refused", "dont_know"],
        "design": ["not_asked", "not_applicable"],
    },
)
```

The spec can then answer questions about its own structure:

``` python
age_missing.sentinel_values()              # [-99, -98, -97, -96]
age_missing.reason_for(-98)                # "refused"
age_missing.values_for_reason("refused")   # [-98]
age_missing.values_for_category("item_nonresponse")  # [-98, -97]
```


## Methods

| Name | Description |
|----|----|
| [from_cdisc()](#from_cdisc) | Alias for <a href="MissingSpec.html#pointblank.MissingSpec.from_cdisc_null_flavors" class="gdls-link"><code>from_cdisc_null_flavors()</code></a>. |
| [from_cdisc_null_flavors()](#from_cdisc_null_flavors) | Create a [MissingSpec](MissingSpec.md#pointblank.MissingSpec) for the standard HL7/CDISC *null flavors*. |
| [from_sas()](#from_sas) | Create a [MissingSpec](MissingSpec.md#pointblank.MissingSpec) for SAS special missing values. |
| [from_spss()](#from_spss) | Create a [MissingSpec](MissingSpec.md#pointblank.MissingSpec) from SPSS-style user-defined missing values. |
| [from_variable_metadata()](#from_variable_metadata) | Create a [MissingSpec](MissingSpec.md#pointblank.MissingSpec) from an imported variable's metadata. |
| [is_missing()](#is_missing) | Check whether a value should be considered missing under this spec. |
| [reason_for()](#reason_for) | Get the reason label for a specific value. |
| [reasons_list()](#reasons_list) | Get the distinct reason labels defined by this spec. |
| [sentinel_values()](#sentinel_values) | Get all sentinel values that encode missingness. |
| [values_for_category()](#values_for_category) | Get all sentinel values whose reason falls in a given category. |
| [values_for_reason()](#values_for_reason) | Get all sentinel values that correspond to a given reason. |

------------------------------------------------------------------------


#### from_cdisc()


Alias for <a href="MissingSpec.html#pointblank.MissingSpec.from_cdisc_null_flavors" class="gdls-link"><code>from_cdisc_null_flavors()</code></a>.


Usage

``` python
from_cdisc(**kwargs)
```


------------------------------------------------------------------------


#### from_cdisc_null_flavors()


Create a [MissingSpec](MissingSpec.md#pointblank.MissingSpec) for the standard HL7/CDISC *null flavors*.


Usage

``` python
from_cdisc_null_flavors(
    null_is_missing=True,
    null_reason="no_information",
    description="CDISC/HL7 null flavors"
)
```


Clinical data uses standardized null flavor codes to record *why* a value is absent (e.g., `"NASK"` for "not asked", `"UNK"` for "unknown"). This returns a ready-to-use spec mapping those codes to reason labels.


##### Parameters


`null_is_missing: bool = ``True`  
Whether actual null values should also be treated as missing. Default is `True`.

`null_reason: str = ``"no_information"`  
The reason label for actual null values. Default is `"no_information"`.

`description: str | None = ``"CDISC/HL7 null flavors"`  
Optional description. Default identifies the spec as CDISC/HL7 null flavors.


##### Returns


`MissingSpec`  
A spec with the standard null flavor codes.


##### Examples

``` python
import pointblank as pb

cdisc_missing = pb.MissingSpec.from_cdisc_null_flavors()
cdisc_missing.reason_for("NASK")   # "not_asked"
```

------------------------------------------------------------------------


#### from_sas()


Create a [MissingSpec](MissingSpec.md#pointblank.MissingSpec) for SAS special missing values.


Usage

``` python
from_sas(
    reasons=None,
    include_underscore=True,
    null_is_missing=True,
    null_reason="system_missing",
    description="SAS special missing values"
)
```


SAS encodes missingness with `"."` (system missing), `"._"`, and `".A"` through `".Z"` (27 user-defined missing codes). This returns a spec covering all of them; you can override the reason label for any specific code via `reasons=`.


##### Parameters


`reasons: dict[str, str] | None = None`  
Optional mapping of specific SAS missing codes to custom reason labels (e.g., `{".A": "not_applicable", ".B": "below_detection"}`). These override the defaults.

`include_underscore: bool = ``True`  
Whether to include the `"._"` special missing code. Default is `True`.

`null_is_missing: bool = ``True`  
Whether actual null values should also be treated as missing. Default is `True`.

`null_reason: str = ``"system_missing"`  
The reason label for actual null values. Default is `"system_missing"`.

`description: str | None = ``"SAS special missing values"`  
Optional description. Default identifies the spec as SAS special missing values.


##### Returns


`MissingSpec`  
A spec covering the SAS special missing values.


##### Examples

``` python
import pointblank as pb

sas_missing = pb.MissingSpec.from_sas(
    reasons={".A": "not_applicable", ".B": "below_detection"}
)
sas_missing.reason_for(".A")   # "not_applicable"
sas_missing.reason_for(".C")   # "user_missing_c"
```

------------------------------------------------------------------------


#### from_spss()


Create a [MissingSpec](MissingSpec.md#pointblank.MissingSpec) from SPSS-style user-defined missing values.


Usage

``` python
from_spss(
    missing_values,
    labels=None,
    null_is_missing=True,
    null_reason="unknown",
    description="SPSS user-defined missing values"
)
```


SPSS supports up to 3 user-defined missing values per variable (plus a range). Pass the missing values (and optionally their value labels) to build a spec. Reason labels are derived from the labels when available, otherwise a `"missing_<value>"` placeholder is used.


##### Parameters


`missing_values: list`  
The sentinel values that SPSS marks as missing for the variable (e.g., `[-99, -98]`).

`labels: dict[Any, str] | None = None`  
Optional mapping of sentinel value to human-readable label (e.g., `{-99: "Refused"}`). Labels are slugified into reason identifiers (e.g., `"Refused"` -\> `"refused"`).

`null_is_missing: bool = ``True`  
Whether actual null values should also be treated as missing. Default is `True`.

`null_reason: str = ``"unknown"`  
The reason label for actual null values. Default is `"unknown"`.

`description: str | None = ``"SPSS user-defined missing values"`  
Optional description. Default identifies the spec as SPSS user-defined missing values.


##### Returns


`MissingSpec`  
A spec built from the SPSS missing values.


##### Examples

``` python
import pointblank as pb

spss_missing = pb.MissingSpec.from_spss(
    missing_values=[-99, -98],
    labels={-99: "Not asked", -98: "Refused"},
)
spss_missing.reason_for(-98)   # "refused"
```

------------------------------------------------------------------------


#### from_variable_metadata()


Create a [MissingSpec](MissingSpec.md#pointblank.MissingSpec) from an imported variable's metadata.


Usage

``` python
from_variable_metadata(variable, null_is_missing=True, null_reason="unknown")
```


This works with a <a href="VariableMetadata.html#pointblank.VariableMetadata" class="gdls-link"><code>VariableMetadata</code></a> object (as produced by <a href="import_metadata.html#pointblank.import_metadata" class="gdls-link"><code>import_metadata()</code></a> for SPSS, Stata, and SAS files). It reads the variable's `missing_values` and derives reason labels from `missing_value_labels` or `value_labels` when available.


##### Parameters


`variable: Any`  
A variable-metadata object exposing `missing_values` and (optionally) `missing_value_labels` / `value_labels` attributes.

`null_is_missing: bool = ``True`  
Whether actual null values should also be treated as missing. Default is `True`.

`null_reason: str = ``"unknown"`  
The reason label for actual null values. Default is `"unknown"`.


##### Returns


`MissingSpec | None`  
A spec built from the variable's missing values, or `None` if the variable declares no missing values.


------------------------------------------------------------------------


#### is_missing()


Check whether a value should be considered missing under this spec.


Usage

``` python
is_missing(value)
```


##### Parameters


`value: Any`  
A value from the data.


##### Returns


`bool`  
`True` if `value` is a declared sentinel value, or if `value` is `None` and `null_is_missing=True`.


------------------------------------------------------------------------


#### reason_for()


Get the reason label for a specific value.


Usage

``` python
reason_for(value)
```


##### Parameters


`value: Any`  
A value from the data.


##### Returns


`str | None`  
The reason label if `value` is a declared sentinel value, [null_reason](MissingSpec.md#pointblank.MissingSpec.null_reason) if `value` is `None` and `null_is_missing=True`, or `None` if the value is not considered missing.


------------------------------------------------------------------------


#### reasons_list()


Get the distinct reason labels defined by this spec.


Usage

``` python
reasons_list()
```


##### Returns


`list[str]`  
The distinct reason labels (in first-seen order), including [null_reason](MissingSpec.md#pointblank.MissingSpec.null_reason) when `null_is_missing=True`.


------------------------------------------------------------------------


#### sentinel_values()


Get all sentinel values that encode missingness.


Usage

``` python
sentinel_values()
```


##### Returns


`list`  
The keys of `reasons` (the actual values in the data that represent missingness). Note that this does *not* include `None` even when `null_is_missing=True`; use <a href="MissingSpec.html#pointblank.MissingSpec.is_missing" class="gdls-link"><code>is_missing()</code></a> to test individual values.


------------------------------------------------------------------------


#### values_for_category()


Get all sentinel values whose reason falls in a given category.


Usage

``` python
values_for_category(category)
```


##### Parameters


`category: str`  
A category name defined in `categories`.


##### Returns


`list`  
All sentinel values whose reason label is in the given category. Returns an empty list if `categories` is `None` or the category is undefined.


------------------------------------------------------------------------


#### values_for_reason()


Get all sentinel values that correspond to a given reason.


Usage

``` python
values_for_reason(reason)
```


##### Parameters


`reason: str`  
A reason label.


##### Returns


`list`  
All sentinel values mapped to `reason`.
