## VariableMetadata


Metadata for a single variable/column, as imported from an external standard.


Usage

``` python
VariableMetadata()
```


## Parameters


`name: str`  
Variable/column name.

`label: str | None = None`  
Human-readable label.

`description: str | None = None`  
Longer description of the variable.

`dtype: str | None = None`  
Data type (mapped to Narwhals/Polars type names).

`role: str | None = None`  
Variable role (e.g., `"identifier"`, `"measure"`, `"classifier"`).

`required: bool = ``False`  
Whether the variable must be non-null.

`unique: bool = ``False`  
Whether all values must be distinct.

`min_val: float | None = None`  
Minimum allowed value (inclusive).

`max_val: float | None = None`  
Maximum allowed value (inclusive).

`min_length: int | None = None`  
Minimum string length.

`max_length: int | None = None`  
Maximum string length.

`pattern: str | None = None`  
Regex pattern that values must match.

`allowed_values: list[Any] | None = None`  
Explicit list of allowed values.

`codelist_ref: str | None = None`  
Reference to a named codelist.

`display_format: str | None = None`  
Display format from source system (e.g., `"F8.2"`, `"DATETIME20."`).

`value_labels: dict[Any, str] | None = None`  
Value-to-label mapping (e.g., `{1: "Male", 2: "Female"}`).

`missing_values: list[Any] | None = None`  
Sentinel values representing missingness (e.g., `-99`, `".A"`, `""`).

`missing_value_labels: dict[Any, str] | None = None`  
Labels for missing value sentinels (e.g., `"Refused"`, `"Not Applicable"`).

`origin: str | None = None`  
How the variable was created (`"CRF"`, `"Derived"`, `"Assigned"`).

`computational_method: str | None = None`  
Derivation algorithm for computed variables.

`controlled_term: str | None = None`  
CDISC controlled terminology reference.

`significant_digits: int | None = None`  
Number of significant digits.

`cdisc_domain: str | None = None`  
CDISC domain code (e.g., `"DM"`, `"AE"`, `"LB"`, `"VS"`).

`cdisc_role: str | None = None`  
CDISC variable role (`"Identifier"`, `"Topic"`, `"Timing"`, `"Qualifier"`, `"Rule"`).

`adam_derivation: str | None = None`  
ADaM derivation algorithm description.

`traceability_ref: str | None = None`  
ADaM traceability reference back to SDTM source.

`unit: str | None = None`  
Unit of measurement (e.g., `"kg"`, `"mmHg"`, `"years"`).

`unit_system: str | None = None`  
Unit system (e.g., `"SI"`, `"imperial"`, `"UDUNITS"`).


## Methods

| Name | Description |
|----|----|
| [to_missing_spec()](#to_missing_spec) | Build a <a href="MissingSpec.html#pointblank.MissingSpec" class="gdls-link"><code>MissingSpec</code></a> from this variable's missing values. |

------------------------------------------------------------------------


#### to_missing_spec()


Build a <a href="MissingSpec.html#pointblank.MissingSpec" class="gdls-link"><code>MissingSpec</code></a> from this variable's missing values.


Usage

``` python
to_missing_spec()
```


Reads `missing_values` and derives reason labels from `missing_value_labels` or `value_labels` when available.


##### Returns


`MissingSpec | None`  
A [MissingSpec](MissingSpec.md#pointblank.MissingSpec) for the variable, or `None` if no missing values are declared.
