# MissingValueCode

A declared missing-value code, as read from a metadata file.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L302-L365)

``` python
MissingValueCode(
    value,
    label,
    category=None,
    reason=None,
)
```

Many data formats let a dataset declare the values that stand for missing data: the user-defined missing values of an SPSS variable (such as `-99` for "Refused"), or the `missingValues` and `null` markers of a Frictionless or CSVW schema (such as `"NA"`). When [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) reads these declarations, it records each one as a [MissingValueCode](../reference/MissingValueCode.md#pointblank.MissingValueCode) in [`MetadataImport.missing_value_codes`](../reference/MetadataImport.md#pointblank.MetadataImport), a dictionary from column name to that column's list of codes.

These objects describe the codes. To use the codes in validation, build [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) objects with [`MetadataImport.missing_specs()`](../reference/MetadataImport.md#pointblank.MetadataImport.missing_specs), which works from each variable's [`missing_values`](../reference/VariableMetadata.md#pointblank.VariableMetadata) and their labels.

## Parameters

`value: Any`  
The value that stands for missing data, such as `-99` or `"NA"`.

`label: str`  
What the code means, such as `"Refused"`.

`category: str | None = None`  
The kind of missingness. The readers use `"user_missing"` for values declared in the file. The default is `None`.

`reason: str | None = None`  
Why the data is missing, if known. The default is `None`.

## Examples

The bundled Frictionless Data Package declares `"NA"` and `"N/A"` as missing-value markers for every field. Let's look at the codes recorded for the `amount` column:

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))

for code in meta.missing_value_codes["amount"]:
    print(code.value, "|", code.label, "|", code.category)
```

    NA | Missing value marker: 'NA' | user_missing
    N/A | Missing value marker: 'N/A' | user_missing

Each marker is recorded with a generated label and the `"user_missing"` category, since it was declared in the file. A code can also be created directly:

``` python
pb.MissingValueCode(value=-99, label="Refused", category="user_missing", reason="refused")
```

    MissingValueCode(value=-99, label='Refused', category='user_missing', reason='refused')

Creating codes yourself is mainly useful when you build a [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport) by hand to describe a dataset that doesn't come with a metadata file.

## See Also

[missing_specs()](../reference/MetadataImport.md#pointblank.MetadataImport.missing_specs)  
Build missing-value specs from imported metadata.

[MissingSpec](../reference/MissingSpec.md#pointblank.MissingSpec)  
Describe missing-value codes for use in validation.
