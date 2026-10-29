# Codelist

A set of permitted values, each with a label.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L93-L299)

``` python
Codelist(
    name,
    codes=list(),
    label=None,
    version=None,
    source=None,
    extensible=False,
)
```

A codelist (also called a value set or controlled terminology) lists the values that a variable may take and what each one means. When [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) reads a file, the codelists it finds become [Codelist](../reference/Codelist.md#pointblank.Codelist) objects in [`MetadataImport.codelists`](../reference/MetadataImport.md#pointblank.MetadataImport): the CodeList elements of a Define-XML document, the codelists of a CDISC Controlled Terminology file, the `enum` constraint of a Frictionless field (as a codelist named `"<field>_enum"`), and the value labels of an SPSS variable (as `"<variable>_values"`). Variables that use a codelist name it in their [`codelist_ref`](../reference/VariableMetadata.md#pointblank.VariableMetadata) and have its values in `allowed_values`.

A codelist's values are what a [`col_vals_in_set()`](../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set) check needs, and [`to_set()`](../reference/Codelist.md#pointblank.Codelist.to_set) gives them as a list. [`to_dict()`](../reference/Codelist.md#pointblank.Codelist.to_dict) maps each value to its label, and [`extended_values()`](../reference/Codelist.md#pointblank.Codelist.extended_values) lists the sponsor extensions of an extensible codelist.

## Parameters

`name: str`  
The name of the codelist, which is also its key in `MetadataImport.codelists`.

`codes: list[CodelistEntry] = list()`  
The entries of the codelist, as [`CodelistEntry`](../reference/CodelistEntry.md#pointblank.CodelistEntry) objects. The default is an empty list.

`label: str | None = None`  
A human-readable name for the codelist. The default is `None`.

`version: str | None = None`  
The version of the terminology, such as a CDISC Controlled Terminology release date. The default is `None`.

`source: str | None = None`  
Where the codelist came from, such as `"CDISC Define-XML"` or `"SPSS .sav"`. The default is `None`.

`extensible: bool = False`  
Whether values that aren't in the codelist may be added, as with CDISC codelists marked as extensible. A non-extensible codelist is a closed set. The default is `False`.

## Examples

The bundled Define-XML document defines a `Sex` codelist for the Demographics dataset. Let's get it from the imported metadata:

``` python
import pointblank as pb

dm = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="DM")
sex = dm.get_codelist("Sex")

print(sex.label, sex.source, sex.extensible, len(sex))
print(sex.to_dict())
```

    Sex CDISC Define-XML False 3
    {'M': 'Male', 'F': 'Female', 'U': 'Unknown'}

The codelist has three entries and isn't extensible, so `M`, `F` and `U` are the only values allowed. Its values can be used in a validation step directly:

``` python
import polars as pl

tbl = pl.DataFrame({"SEX": ["M", "F", "F", "X"]})

(
    pb.Validate(data=tbl)
    .col_vals_in_set(columns="SEX", set=sex.to_set())
    .interrogate()
)
```

[TABLE]

The `"X"` in the last row isn't in the codelist, so one test unit fails. A codelist can also be built by hand:

``` python
route = pb.Codelist(
    name="ROUTE",
    label="Route of Administration",
    extensible=True,
    codes=[
        pb.CodelistEntry(value="ORAL", label="Oral"),
        pb.CodelistEntry(value="INTRAVENOUS", label="Intravenous"),
        pb.CodelistEntry(value="IV", label="Intravenous (old code)", is_deprecated=True),
        pb.CodelistEntry(value="NASOGASTRIC TUBE", label="Nasogastric tube", is_extended=True),
    ],
)

print(len(route))
print(route.to_set())
print(route.extended_values())
```

    4
    ['ORAL', 'INTRAVENOUS', 'NASOGASTRIC TUBE']
    ['NASOGASTRIC TUBE']

`len()` counts all four entries, but [to_set()](../reference/Codelist.md#pointblank.Codelist.to_set) leaves out the deprecated `"IV"`, and [extended_values()](../reference/Codelist.md#pointblank.Codelist.extended_values) gives the one value that was added to the extensible codelist.

## See Also

[CodelistEntry](../reference/CodelistEntry.md#pointblank.CodelistEntry)  
A single entry in a codelist.

[get_codelist()](../reference/MetadataImport.md#pointblank.MetadataImport.get_codelist)  
Get a codelist from imported metadata.

## Methods

| Name | Description |
|----|----|
| [to_set()](#to_set) | Get the permitted values of the codelist. |
| [to_dict()](#to_dict) | Get a mapping of each value to its label. |
| [extended_values()](#extended_values) | Get the values that were added to an extensible codelist. |

### to_set()

Get the permitted values of the codelist.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L204-L230)

``` python
to_set()
```

The values are given in the order of the codelist's entries, leaving out any entry marked as deprecated. The result is ready to be used as the `set=` of [`col_vals_in_set()`](../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set).

#### Returns

`list`  
The values of all entries that aren't deprecated.

#### Examples

``` python
import pointblank as pb

dm = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="DM")

dm.get_codelist("Race").to_set()
```

    ['WHITE',
     'BLACK OR AFRICAN AMERICAN',
     'ASIAN',
     'AMERICAN INDIAN OR ALASKA NATIVE',
     'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER']

These are the five values that the `RACE` variable may take in the bundled Define-XML document. A table can be checked against them with `col_vals_in_set(columns="RACE", set=dm.get_codelist("Race").to_set())`.

### to_dict()

Get a mapping of each value to its label.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L271-L296)

``` python
to_dict()
```

Unlike [`to_set()`](../reference/Codelist.md#pointblank.Codelist.to_set), this includes the deprecated entries, so that every code that might appear in older data can still be decoded.

#### Returns

`dict`  
A dictionary with the coded values as keys and their labels as values.

#### Examples

``` python
import pointblank as pb

ae = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="AE")

ae.get_codelist("Severity").to_dict()
```

    {'MILD': 'Mild', 'MODERATE': 'Moderate', 'SEVERE': 'Severe'}

The mapping gives the meaning of each severity code used in the Adverse Events dataset. It can be used to add a column of labels to a table, for example with Polars' `pl.col("AESEV").replace_strict(mapping)`.

### extended_values()

Get the values that were added to an extensible codelist.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L232-L269)

``` python
extended_values()
```

An extensible codelist, such as many in CDISC Controlled Terminology, allows a study to add values of its own. In Define-XML these sponsor extensions are marked with `def:ExtendedValue="Yes"`, and they're read into entries with `is_extended=True`. This method lists those values, so they can be reviewed separately from the published ones.

#### Returns

`list`  
The values of the entries marked as extended. The list is empty if there are none.

#### Examples

``` python
import pointblank as pb

route = pb.Codelist(
    name="ROUTE",
    extensible=True,
    codes=[
        pb.CodelistEntry(value="ORAL", label="Oral"),
        pb.CodelistEntry(value="INTRAVENOUS", label="Intravenous"),
        pb.CodelistEntry(
            value="NASOGASTRIC TUBE", label="Nasogastric tube", is_extended=True
        ),
    ],
)

route.extended_values()
```

    ['NASOGASTRIC TUBE']

Only `"NASOGASTRIC TUBE"` was added to the codelist, so it's the only extended value. Extended values are still permitted values, and [`to_set()`](../reference/Codelist.md#pointblank.Codelist.to_set) includes them.
