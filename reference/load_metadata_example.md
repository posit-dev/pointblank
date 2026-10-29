# load_metadata_example()

Get the path to a bundled metadata example file.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_import.py#L77-L153)

``` python
load_metadata_example(name)
```

Pointblank ships a small collection of metadata files in the formats that [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) reads: CDISC Define-XML, CDISC Controlled Terminology, a Frictionless Data Package and Table Schema, CSVW, and SAS Transport (XPT). This function returns the path to one of them, so you can try out metadata import (and run the examples in this documentation) without supplying files of your own.

## Parameters

`name: str`  
The file name of the example, such as `"define.xml"`. See the *Available Examples* section for the full list. An unknown name raises a `ValueError` that lists the valid names.

## Returns

`Path`  
The path to the bundled file, which can be passed directly to [import_metadata()](../reference/import_metadata.md#pointblank.import_metadata).

## Available Examples

| Name | Format | Description |
|----|----|----|
| `"define.xml"` | `cdisc_define` | Define-XML 2.0 document with the DM and AE domains |
| `"sdtm_ct.xml"` | `cdisc_ct` | SDTM Controlled Terminology with common codelists |
| `"datapackage.json"` | `frictionless` | Frictionless Data Package for a transactions table |
| `"table_schema.json"` | `table_schema` | Frictionless Table Schema for sensor readings |
| `"weather_csvw.json"` | `csvw` | W3C CSVW metadata for weather observations |
| `"dm.xpt"` | `xpt` | SAS Transport file with a small SDTM Demographics dataset |

The XML formats need the `lxml` package and the XPT file needs `pyreadstat`. The JSON formats don't need any extra packages.

## Examples

Get the path to the bundled Define-XML document:

``` python
import pointblank as pb

define_path = pb.load_metadata_example("define.xml")

print(define_path.name)
print(define_path.exists())
```

    define.xml
    True

The result is a `pathlib.Path` to a file inside the installed package. Pass it to [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) to read the metadata:

``` python
package = pb.import_metadata(define_path)

for name in package.keys():
    meta = package[name]
    print(f"{name}: {meta.dataset_label} ({len(meta)} variables)")
```

    DM: Demographics (13 variables)
    AE: Adverse Events (10 variables)

The format was detected from the file's contents, and because the document describes two datasets the result is a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage) with one entry per dataset.

## See Also

[import_metadata()](../reference/import_metadata.md#pointblank.import_metadata)  
Read metadata from a file, such as one of these examples.
