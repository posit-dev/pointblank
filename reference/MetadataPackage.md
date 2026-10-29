# MetadataPackage

The metadata for several datasets, read from one source.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L1110-L1351)

``` python
MetadataPackage(
    name=None,
    items=dict(),
    description=None,
    version=None,
)
```

When a file describes more than one dataset, [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) returns a [MetadataPackage](../reference/MetadataPackage.md#pointblank.MetadataPackage) with a [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport) for each. This happens with a Define-XML document for a whole study (one entry per dataset, such as `"DM"` and `"AE"`), a Frictionless Data Package or CSVW table group with several tables, and a CDISC Controlled Terminology file (one entry per codelist).

The package works much like a read-only dictionary: `package["DM"]` gets an entry, `"DM" in package` checks for one, `len()` counts them, iterating gives their names, and [`keys()`](../reference/MetadataPackage.md#pointblank.MetadataPackage.keys) and [`values()`](../reference/MetadataPackage.md#pointblank.MetadataPackage.values) give the names and entries. [`get_domain()`](../reference/MetadataPackage.md#pointblank.MetadataPackage.get_domain) also gets an entry, with an error message that lists the available names, and [`summary()`](../reference/MetadataPackage.md#pointblank.MetadataPackage.summary) describes the package in text.

## Parameters

`name: str | None = None`  
The name of the package, such as the study OID of a Define-XML document. The default is `None`.

`items: dict[str, MetadataImport] = dict()`  
The entries, as a dictionary from dataset name to [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport). The default is an empty dictionary.

`description: str | None = None`  
A description of the package. The default is `None`.

`version: str | None = None`  
The version of the package or of the standard it follows. The default is `None`.

## Examples

The bundled Define-XML document describes two datasets of a clinical study:

``` python
import pointblank as pb

package = pb.import_metadata(pb.load_metadata_example("define.xml"))

print(len(package))
print("DM" in package)

for name in package:
    print(name, "-", package[name].dataset_label)
```

    2
    True
    DM - Demographics
    AE - Adverse Events

Each entry is a [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport), so it can be used like one. Here the Demographics metadata is used to check a small table:

``` python
import polars as pl

dm_data = pl.DataFrame(
    {
        "STUDYID": ["XYZ789"] * 3,
        "DOMAIN": ["DM"] * 3,
        "USUBJID": ["XYZ789-001", "XYZ789-002", "XYZ789-003"],
        "SUBJID": ["001", "002", "003"],
        "RFSTDTC": ["2024-01-15", "2024-01-20", "2024-02-01"],
        "RFENDTC": ["2024-06-15", "2024-06-20", "2024-07-01"],
        "SITEID": ["SITE01", "SITE01", "SITE02"],
        "AGE": [45, 62, 38],
        "AGEU": ["YEARS"] * 3,
        "SEX": ["M", "F", "M"],
        "RACE": ["WHITE", "ASIAN", "BLACK"],
        "ARMCD": ["DRUG", "PLACEBO", "DRUG"],
        "ARM": ["Active Drug 10mg", "Placebo", "Active Drug 10mg"],
    }
)

package["DM"].to_validate(data=dm_data).interrogate()
```

[TABLE]

The table has all the variables of the `DM` dataset with the declared types, and the required ones have values. The last step fails for one row, since `"BLACK"` isn't one of the values in the `Race` codelist (which uses `"BLACK OR AFRICAN AMERICAN"`).

## See Also

[import_metadata()](../reference/import_metadata.md#pointblank.import_metadata)  
Read metadata from a file.

[MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport)  
The metadata for a single dataset.

## Methods

| Name | Description |
|----|----|
| [get_domain()](#get_domain) | Get the metadata of a dataset by name. |
| [keys()](#keys) | Get the names of the datasets in the package. |
| [values()](#values) | Get the metadata of the datasets in the package. |
| [summary()](#summary) | Get a text summary of the package. |

### get_domain()

Get the metadata of a dataset by name.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L1267-L1307)

``` python
get_domain(name)
```

This does the same as `package[name]`, but a missing name gives an error message that lists the names in the package.

#### Parameters

`name: str`  
The name of the dataset, such as `"DM"` or `"AE"`.

#### Returns

`MetadataImport`  
The [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport) of the dataset.

#### Raises

`KeyError`  
If there's no dataset with that name.

#### Examples

``` python
import pointblank as pb

package = pb.import_metadata(pb.load_metadata_example("define.xml"))
ae = package.get_domain("AE")

print(ae.dataset_label)
print(ae.variable_names)
```

    Adverse Events
    ['STUDYID', 'DOMAIN', 'USUBJID', 'AESEQ', 'AETERM', 'AEDECOD', 'AESEV', 'AESER', 'AESTDTC', 'AEENDTC']

Asking for a dataset that isn't in the package, such as `package.get_domain("LB")`, raises a `KeyError` with the message `No domain/dataset named 'LB'. Available: ['DM', 'AE']`.

### keys()

Get the names of the datasets in the package.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L1218-L1239)

``` python
keys()
```

#### Returns

`KeysView`  
The names, in the order of the source file, as a dictionary keys view.

#### Examples

``` python
import pointblank as pb

ct = pb.import_metadata(pb.load_metadata_example("sdtm_ct.xml"))

list(ct.keys())
```

    ['Sex',
     'Severity/Intensity Scale for Adverse Events',
     'No Yes Response',
     'Race',
     'Route of Administration']

For a Controlled Terminology file the entries are codelists, so the names are codelist names. Each of them can be used to get an entry with `ct[name]`.

### values()

Get the metadata of the datasets in the package.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L1241-L1265)

``` python
values()
```

#### Returns

`ValuesView`  
The [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport) entries, in the order of the source file, as a dictionary values view.

#### Examples

``` python
import pointblank as pb

package = pb.import_metadata(pb.load_metadata_example("define.xml"))

for meta in package.values():
    print(meta.dataset_name, len(meta), list(meta.codelists))
```

    DM 13 ['Sex', 'Race']
    AE 10 ['Severity', 'No Yes Response']

This is the easiest way to do something with every dataset, such as counting variables or listing codelists as here. Use `package.items.items()` to get the names and entries together.

### summary()

Get a text summary of the package.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L1309-L1345)

``` python
summary()
```

The summary gives the package's name and description (when known), the number of datasets, and the number of variables in each. Printing a [MetadataPackage](../reference/MetadataPackage.md#pointblank.MetadataPackage) with `print()` shows the same summary. For the details of a dataset, use the [`summary()`](../reference/MetadataImport.md#pointblank.MetadataImport.summary) of its entry.

#### Returns

`str`  
The summary, as a string with one line per item.

#### Examples

``` python
import pointblank as pb

package = pb.import_metadata(pb.load_metadata_example("define.xml"))

print(package.summary())
```

    Metadata Package
      Name: STUDY.XYZ789
      Description: CDISC Define-XML 2.0 study metadata
      Datasets: 2

      [DM] 13 variables
      [AE] 10 variables

The package is named after the study in the Define-XML document, and holds 13 variables for the `DM` dataset and 10 for `AE`.
