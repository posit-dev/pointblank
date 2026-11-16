# from_variable_metadata()

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

## Parameters

`variable: Any`  
A variable-metadata object exposing `missing_values` and (optionally) `missing_value_labels` / `value_labels` attributes.

`null_is_missing: bool = True`  
Whether actual null values should also be treated as missing. Default is `True`.

`null_reason: str = "unknown"`  
The reason label for actual null values. Default is `"unknown"`.

## Returns

`MissingSpec | None`  
A spec built from the variable's missing values, or `None` if the variable declares no missing values.

## Examples

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
