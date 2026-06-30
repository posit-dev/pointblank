## MetadataImport


Parsed metadata from an external standard.


Usage

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
    missing_value_codes=dict()
)
```


Contains variable definitions, value labels, missing value codes, controlled terminologies, and dataset-level metadata: all mapped to Pointblank concepts.


## Parameters


`source_format: str`  
The format this metadata was imported from (e.g., `"spss"`, `"xpt"`, `"stata"`).

`source_path: str | None = None`  
Path to the source file, if imported from a file.

`source_version: str | None = None`  
Version of the source format/standard.

`dataset_name: str | None = None`  
Name of the dataset.

`dataset_label: str | None = None`  
Human-readable label for the dataset.

`dataset_description: str | None = None`  
Description of the dataset.

`creation_date: str | None = None`  
When the dataset/metadata was created.

`study_id: str | None = None`  
Study identifier (for clinical data).

`domain: str | None = None`  
Domain identifier (e.g., `"DM"`, `"AE"` for CDISC).

`variables: list[VariableMetadata] = list()`    
List of variable metadata definitions.

`codelists: dict[str, Codelist] = dict()`    
Named codelists (controlled terminologies).

`missing_value_codes: dict[str, list[MissingValueCode]] = dict()`    
Named missing value code definitions.


## Attributes

| Name | Description |
|----|----|
| [variable_names](#variable_names) | Get the list of all variable names. |

------------------------------------------------------------------------


#### variable_names


Get the list of all variable names.


`variable_names: list[str]`


## Methods

| Name | Description |
|----|----|
| [get_codelist()](#get_codelist) | Get a specific codelist by name. |
| [get_variable()](#get_variable) | Get metadata for a specific variable by name. |
| [missing_specs()](#missing_specs) | Auto-generate <a href="MissingSpec.html#pointblank.MissingSpec" class="gdls-link"><code>MissingSpec</code></a> objects for all variables. |
| [summary()](#summary) | Return a human-readable summary of the imported metadata. |
| [to_schema()](#to_schema) | Convert imported metadata to a Pointblank [Schema](Schema.md#pointblank.Schema) with `Field` objects. |
| [to_validate()](#to_validate) | Generate a [Validate](Validate.md#pointblank.Validate) workflow from the imported metadata. |

------------------------------------------------------------------------


#### get_codelist()


Get a specific codelist by name.


Usage

``` python
get_codelist(name)
```


##### Parameters


`name: str`  
The codelist name or identifier.


##### Returns


`Codelist`  
The requested codelist.


##### Raises


`KeyError`  
If no codelist with that name exists.


------------------------------------------------------------------------


#### get_variable()


Get metadata for a specific variable by name.


Usage

``` python
get_variable(name)
```


##### Parameters


`name: str`  
The variable name to look up.


##### Returns


`VariableMetadata`  
The metadata for the named variable.


##### Raises


`KeyError`  
If no variable with that name exists.


------------------------------------------------------------------------


#### missing_specs()


Auto-generate <a href="MissingSpec.html#pointblank.MissingSpec" class="gdls-link"><code>MissingSpec</code></a> objects for all variables.


Usage

``` python
missing_specs()
```


Builds a mapping of column name to [MissingSpec](MissingSpec.md#pointblank.MissingSpec) for every imported variable that declares missing values (e.g., SPSS user-defined missing values, SAS special missing). The result can be passed directly to validation methods (via `missing=`) or to <a href="missing_vals_tbl.html#pointblank.missing_vals_tbl" class="gdls-link"><code>missing_vals_tbl()</code></a>.


##### Returns


`dict[str, MissingSpec]`  
A mapping of column name to [MissingSpec](MissingSpec.md#pointblank.MissingSpec). Variables without declared missing values are omitted.


##### Examples

``` python
import pointblank as pb

meta = pb.import_metadata("survey.sav", format="spss")
specs = meta.missing_specs()

# Use the auto-generated specs in a missingness report
pb.missing_vals_tbl(data, missing=specs)
```

------------------------------------------------------------------------


#### summary()


Return a human-readable summary of the imported metadata.


Usage

``` python
summary()
```


##### Returns


`str`  
Formatted summary string.


------------------------------------------------------------------------


#### to_schema()


Convert imported metadata to a Pointblank [Schema](Schema.md#pointblank.Schema) with `Field` objects.


Usage

``` python
to_schema()
```


Maps variable metadata to appropriate `Field` types with constraints (min/max, allowed values, nullable, etc.).


##### Returns


`Schema`  
A Pointblank [Schema](Schema.md#pointblank.Schema) object with typed fields.


------------------------------------------------------------------------


#### to_validate()


Generate a [Validate](Validate.md#pointblank.Validate) workflow from the imported metadata.


Usage

``` python
to_validate(data, **kwargs)
```


Creates validation steps for all constraints found in the metadata: value ranges, allowed values, required fields, string lengths, etc.


##### Parameters


`data: Any`  
The DataFrame or table to validate.

`**kwargs: Any`  
Additional keyword arguments passed to the [Validate](Validate.md#pointblank.Validate) constructor.


##### Returns


<a href="Validate.html#pointblank.Validate" class="gdls-link gdls-code"><code>Validate</code></a>  
A configured (but not yet interrogated) [Validate](Validate.md#pointblank.Validate) object.
