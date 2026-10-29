# import_metadata()

Import metadata from a statistical-package file or a metadata standard.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_import.py#L323-L583)

``` python
import_metadata(
    source,
    format=None,
    **kwargs,
)
```

Many datasets come with a description of their variables: the variable labels, value labels and missing-value codes stored in an SPSS, Stata or SAS file, the field types and constraints in a Frictionless or CSVW schema, or the variable definitions and codelists in a CDISC Define-XML document. The [import_metadata()](../reference/import_metadata.md#pointblank.import_metadata) function reads that description (not the data itself) into a [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport) object, with one [`VariableMetadata`](../reference/VariableMetadata.md#pointblank.VariableMetadata) per variable and any codelists as [`Codelist`](../reference/Codelist.md#pointblank.Codelist) objects.

The imported metadata is most useful as the starting point for validation. Its [`to_validate()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_validate) method turns the declared types and constraints into a [`Validate`](../reference/Validate.md#pointblank.Validate) plan, [`to_schema()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_schema) gives a [`Schema`](../reference/Schema.md#pointblank.Schema) of the column types, and [`missing_specs()`](../reference/MetadataImport.md#pointblank.MetadataImport.missing_specs) turns declared missing-value codes into [`MissingSpec`](../reference/MissingSpec.md#pointblank.MissingSpec) objects. Use [`export_metadata()`](../reference/export_metadata.md#pointblank.export_metadata) to go the other way and write metadata out as a Frictionless Table Schema.

A source that describes several datasets (a Define-XML document for a whole study, a Data Package with several resources, or a Controlled Terminology file with several codelists) gives a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage) instead, which holds one [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) per dataset.

## Parameters

`source: str | Path | Any`  
The path to the metadata file, as a string or a `pathlib.Path`. Other kinds of objects raise a `TypeError`.

`format: str | None = None`  
The format of the file. If `None` (the default), the format is detected from the file: by extension for `.sav`, `.zsav`, `.dta` and `.xpt` files, and by content for `.json` files (Frictionless or CSVW) and `.xml` files (Define-XML or Controlled Terminology). Other extensions need an explicit format. See the *Supported Formats* section for the values.

`**kwargs: Any`  
Options for the reader of the chosen format. The Define-XML reader takes `dataset=` (such as `"DM"`) to return only that dataset as a [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport). The Controlled Terminology reader takes `codelist=` to return only one codelist, and the Frictionless reader takes `resource=` (a name or an index) to pick one resource from a Data Package. The `"cdisc_sdtm"` and `"cdisc_adam"` formats need `domain=` and `dataset=`, respectively, and also take `study_id=`.

## Returns

`MetadataImport | MetadataPackage`  
A [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) for a source that describes a single dataset, or a [MetadataPackage](../reference/MetadataPackage.md#pointblank.MetadataPackage) for a source that describes several.

## Raises

`ValueError`  
If the format can't be detected or isn't supported, or if a required option such as `domain=` is missing.

`FileNotFoundError`  
If the file doesn't exist.

`ImportError`  
If the format needs an optional package that isn't installed.

## Supported Formats

| `format=` | Reads | Needs |
|----|----|----|
| `"frictionless"` (or `"datapackage"`, `"table_schema"`) | Frictionless schema (`.json`) |  |
| `"csvw"` | W3C CSV on the Web metadata (`.json`) |  |
| `"cdisc_define"` (or `"define_xml"`) | CDISC Define-XML 2.0 or 2.1 (`.xml`) | `lxml` |
| `"cdisc_ct"` | CDISC Controlled Terminology in ODM-XML (`.xml`) | `lxml` |
| `"spss"` (or `"sav"`) | SPSS data file (`.sav`, `.zsav`) | `pyreadstat` |
| `"stata"` (or `"dta"`) | Stata data file (`.dta`) | `pyreadstat` |
| `"xpt"` (or `"sas"`) | SAS Transport file (`.xpt`) | `pyreadstat` |
| `"cdisc_sdtm"` | A built-in SDTM domain template |  |
| `"cdisc_adam"` | A built-in ADaM dataset template |  |

For SPSS, Stata and SAS files only the metadata is read, not the data. The last two formats don't read a file at all: `source=` isn't used (any string will do), and the template named by `domain=` or `dataset=` is returned, as with [`sdtm_to_metadata()`](../reference/sdtm_to_metadata.md#pointblank.sdtm_to_metadata) and [`adam_to_metadata()`](../reference/adam_to_metadata.md#pointblank.adam_to_metadata).

## Examples

Pointblank bundles a few metadata files to experiment with, which can be found with [`load_metadata_example()`](../reference/load_metadata_example.md#pointblank.load_metadata_example). Let's import a Frictionless Data Package that describes a table of sales transactions:

``` python
import pointblank as pb

meta = pb.import_metadata(pb.load_metadata_example("datapackage.json"))

meta
```

    MetadataImport(source_format='frictionless', variables=8, codelists=1)

The format was detected from the file's contents. The metadata describes eight variables and one codelist (the allowed values of `category`). Each variable's metadata can be looked up by name:

``` python
amount = meta.get_variable("amount")

print(amount.dtype, amount.required, amount.min_val, amount.max_val)
```

    Float64 True 0.01 99999.99

The `amount` variable must be a non-null float from `0.01` to `99999.99`. Constraints like these can be checked against a table with [`to_validate()`](../reference/MetadataImport.md#pointblank.MetadataImport.to_validate), which builds a validation plan from the metadata:

``` python
import datetime
import polars as pl

transactions = pl.DataFrame(
    {
        "transaction_id": ["TXN-00001", "TXN-00002", "TXN-00003", "TXN-00004"],
        "customer_id": ["CUST-101", "CUST-102", "CUST-101", "CUST-103"],
        "amount": [25.50, 1200.00, 0.00, 89.99],
        "quantity": [1, 3, 2, 1],
        "category": ["food", "electronics", "home", "toys"],
        "sale_date": [datetime.date(2024, 5, d) for d in (1, 1, 2, 3)],
        "discount_pct": [0.0, 10.0, 0.0, 5.0],
        "email": ["a@example.com", "b@example.com", "c@example.com", "not-an-email"],
    }
)

meta.to_validate(data=transactions).interrogate()
```

[TABLE]

The table has the declared column types, so the schema check in step 1 passes. Three later steps find a failing row each: the `amount` of `0.00` is below the minimum, `"toys"` isn't one of the allowed categories, and `"not-an-email"` doesn't match the pattern given for `email`.

A Define-XML document usually describes all the datasets in a study, so importing one gives a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage):

``` python
package = pb.import_metadata(pb.load_metadata_example("define.xml"))

package
```

    MetadataPackage(name='STUDY.XYZ789', datasets=2)

The package holds the Demographics (`DM`) and Adverse Events (`AE`) datasets, and `package["DM"]` gives the [MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport) for one of them. If only one dataset is needed, the `dataset=` option of the Define-XML reader returns it directly:

``` python
dm = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="DM")

print(dm.dataset_label)
print(dm.variable_names)
```

    Demographics
    ['STUDYID', 'DOMAIN', 'USUBJID', 'SUBJID', 'RFSTDTC', 'RFENDTC', 'SITEID', 'AGE', 'AGEU', 'SEX', 'RACE', 'ARMCD', 'ARM']

Options like `dataset=` are passed through to the reader for the file's format. The *Supported Formats* section lists the formats, and the `**kwargs` parameter describes the options that each reader accepts.

## See Also

[load_metadata_example()](../reference/load_metadata_example.md#pointblank.load_metadata_example)  
Get the path to a bundled metadata file.

[export_metadata()](../reference/export_metadata.md#pointblank.export_metadata)  
Export metadata as a Frictionless Table Schema.

[MetadataImport](../reference/MetadataImport.md#pointblank.MetadataImport)  
The metadata for a single dataset.

[MetadataPackage](../reference/MetadataPackage.md#pointblank.MetadataPackage)  
The metadata for several datasets.
