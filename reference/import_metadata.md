## import_metadata()


Import metadata from an external standard or file.


Usage

``` python
import_metadata(
    source,
    format=None,
    **kwargs,
)
```


Reads metadata definitions from statistical package files (SPSS, SAS, Stata), standards documents (CDISC Define-XML, Frictionless), or scientific formats (NetCDF/CF) and returns a structured representation that can be converted to Pointblank validation workflows.


## Parameters


`source: str | Path | Any`  
Path to a metadata file, or an object containing metadata (e.g., an xarray Dataset). For file paths, the format will be auto-detected from the extension if not specified.

`format: str | None = None`  
Explicit format identifier. If None, auto-detected from the file extension. Supported formats: `"spss"`, `"sav"`, `"xpt"`, `"sas"`, `"stata"`, `"dta"`, `"frictionless"`, `"datapackage"`, `"table_schema"`, `"csvw"`, `"cdisc_define"`, `"define_xml"`, `"cdisc_ct"`.

`**kwargs: Any`  
Additional format-specific options passed to the reader.


## Returns


`MetadataImport | MetadataPackage`  
A MetadataImport for single-dataset sources, or a MetadataPackage for multi-dataset sources (e.g., multi-domain CDISC studies).


## Raises


`ValueError`  
If the format cannot be determined or is not supported.

`ImportError`  
If the required optional dependency is not installed.


## Examples

Import SPSS metadata and generate validation:

``` python
import pointblank as pb

meta = pb.import_metadata("survey_data.sav")
meta.summary()

# Convert to a validation workflow
validation = meta.to_validate(data=df).interrogate()
```

Import SAS Transport metadata:

``` python
meta = pb.import_metadata("clinical_data.xpt", format="xpt")
schema = meta.to_schema()
```
