# MetadataImport

The metadata for a single dataset.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L585-L1108)

``` python
MetadataImport(
    source_format,
    source_path=None,
    source_version=None,
    dataset_name=None,
    dataset_label=None,
    dataset_description=None,
    creation_date=None,
    study_id=None,
    domain=None,
    variables=list(),
    codelists=dict(),
    missing_value_codes=dict(),
)
```

A [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) is what [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) returns for a file that describes one dataset. It holds the dataset-level information (name, label, study and domain), a [`VariableMetadata`](../reference/VariableMetadata.md#pointblank.VariableMetadata) for each variable, the codelists that the variables use, and any declared missing-value codes, all in the same form whatever the format of the file.

The methods turn the metadata into things Pointblank can use: [`to_validate()`](../reference/pointblank.MetadataImport.to_validate.md) builds a validation plan from the declared types and constraints, [`to_schema()`](../reference/pointblank.MetadataImport.to_schema.md) gives a [`Schema`](../reference/Schema.md#pointblank.Schema) of the column types, and [`missing_specs()`](../reference/pointblank.MetadataImport.missing_specs.md) builds [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) objects from the missing-value codes. Variables and codelists can be looked up with [`get_variable()`](../reference/pointblank.MetadataImport.get_variable.md) and [`get_codelist()`](../reference/pointblank.MetadataImport.get_codelist.md), and [`summary()`](../reference/pointblank.MetadataImport.summary.md) describes the whole dataset in text.

A [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) can also be created directly, which is a way to write down the rules for a dataset that doesn't come with a metadata file and still use the methods above. Calling `len()` on it gives the number of variables.

## Parameters

`source_format: str`  
The format the metadata was read from, such as `"frictionless"`, `"cdisc_define"` or `"spss"`. Any string can be used for metadata created directly.

`source_path: str | None = None`  
The path of the file that was read. The default is `None`.

`source_version: str | None = None`  
The version of the format or standard, such as `"Define-XML 2.0"`. The default is `None`.

`dataset_name: str | None = None`  
The name of the dataset, such as `"DM"`. The default is `None`.

`dataset_label: str | None = None`  
A human-readable label for the dataset, such as `"Demographics"`. The default is `None`.

`dataset_description: str | None = None`  
A longer description of the dataset. The default is `None`.

`creation_date: str | None = None`  
When the metadata was created, as given in the file. The default is `None`.

`study_id: str | None = None`  
The study identifier, for clinical data. The default is `None`.

`domain: str | None = None`  
The CDISC domain, such as `"DM"` or `"AE"`. The default is `None`.

`variables: list[VariableMetadata] = list()`  
The metadata of each variable, as a list of [`VariableMetadata`](../reference/VariableMetadata.md#pointblank.VariableMetadata) objects. The default is an empty list.

`codelists: dict[str, Codelist] = dict()`  
The codelists used by the variables, as a dictionary from codelist name to [`Codelist`](../reference/Codelist.md#pointblank.Codelist). The default is an empty dictionary.

`missing_value_codes: dict[str, list[MissingValueCode]] = dict()`  
The declared missing-value codes, as a dictionary from column name to a list of [`MissingValueCode`](../reference/MissingValueCode.md#pointblank.MissingValueCode) objects. The default is an empty dictionary.

## Examples

Let's import the bundled Frictionless Table Schema, which describes a table of sensor readings:

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("table_schema.json"))

print(len(meta))
print(meta.variable_names)
print(list(meta.codelists))
```

    6
    ['sensor_id', 'reading_time', 'temperature', 'pressure_hpa', 'battery_pct', 'status']
    ['status_enum']

The schema describes six variables, and the allowed values of `status` were read into a codelist named `"status_enum"`. Metadata can also be written down directly, here for a small survey table:

``` python
survey = pb.MetadataImport(
    source_format="custom",
    dataset_name="survey",
    dataset_label="Household survey",
    variables=[
        pb.VariableMetadata(name="respondent_id", dtype="String", required=True, unique=True),
        pb.VariableMetadata(name="age", dtype="Int64", required=True, min_val=18, max_val=99),
    ],
)

survey
```

    MetadataImport(source_format='custom', variables=2, codelists=0)

Either way, the result can be turned into a validation plan with [`to_validate()`](../reference/pointblank.MetadataImport.to_validate.md), which has an example of checking a table against its metadata.

## See Also

[import_metadata()](../reference/import_metadata.md#pointblank.import_metadata)  
Read a [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) from a file.

[MetadataPackage](../reference/MetadataPackage.md#pointblank.MetadataPackage)  
The metadata for several datasets.

[VariableMetadata](../reference/VariableMetadata.md#pointblank.VariableMetadata)  
The metadata of one variable.

## Attributes

| Name | Description |
|----|----|
| [variable_names](pointblank.MetadataImport.variable_names.md) | The names of all variables, in order. |

## Functions

| Name | Description |
|----|----|
| [to_validate()](pointblank.MetadataImport.to_validate.md) | Build a validation plan for a table from the metadata. |
| [to_schema()](pointblank.MetadataImport.to_schema.md) | Get a [Schema](../reference/Schema.md#pointblank.Schema) of the variables' names and types. |
| [get_variable()](pointblank.MetadataImport.get_variable.md) | Get the metadata of a variable by name. |
| [get_codelist()](pointblank.MetadataImport.get_codelist.md) | Get a codelist by name. |
| [missing_specs()](pointblank.MetadataImport.missing_specs.md) | Build a [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) for each variable with missing values. |
| [summary()](pointblank.MetadataImport.summary.md) | Get a text summary of the metadata. |
