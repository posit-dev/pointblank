# CodelistEntry

A single permitted value in a codelist.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L21-L90)

``` python
CodelistEntry(
    value,
    label,
    description=None,
    synonyms=None,
    is_deprecated=False,
    is_extended=False,
)
```

A [`Codelist`](../reference/Codelist.md#pointblank.Codelist) is a list of [CodelistEntry](../reference/CodelistEntry.md#pointblank.CodelistEntry) objects, each pairing a coded value (the value that appears in the data, such as `"F"`) with a label that says what it means (such as `"Female"`). Codelist entries are created when [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) reads a codelist, and can also be created directly to build a codelist by hand.

## Parameters

`value: Any`  
The coded value, as it appears in the data.

`label: str`  
A human-readable label for the value.

`description: str | None = None`  
A longer description of the entry. The default is `None`.

`synonyms: list[str] | None = None`  
Other terms with the same meaning, such as the synonyms listed in CDISC Controlled Terminology. The default is `None`.

`is_deprecated: bool = False`  
Whether the entry has been retired. A deprecated entry stays in the codelist (and in [`to_dict()`](../reference/Codelist.md#pointblank.Codelist.to_dict)) but is left out of the permitted values given by [`to_set()`](../reference/Codelist.md#pointblank.Codelist.to_set). The readers don't currently set this flag, but you can for codelists you build yourself. The default is `False`.

`is_extended: bool = False`  
Whether the entry is a sponsor extension to an extensible codelist: a value that isn't in the published terminology but was added for a study. Define-XML marks these with `def:ExtendedValue="Yes"`. The default is `False`.

## Examples

Each codelist imported from a file holds its entries in `codes`. Here are the entries of the `Sex` codelist in the bundled CDISC Controlled Terminology file:

``` python
import pointblank as pb

ct = pb.import_metadata(pb.load_metadata_example("sdtm_ct.xml"), codelist="Sex")
sex = ct.get_codelist("Sex")

for entry in sex.codes:
    print(entry.value, "->", entry.label)
```

    F -> Female
    M -> Male
    U -> Unknown
    UNDIFFERENTIATED -> Undifferentiated

The values are the codes used in the data, and the labels say what they mean. An entry can also be created directly, for example to build a codelist that isn't in any file:

``` python
entry = pb.CodelistEntry(value="NASOGASTRIC TUBE", label="Nasogastric tube", is_extended=True)

entry
```

    CodelistEntry(value='NASOGASTRIC TUBE', label='Nasogastric tube', description=None, synonyms=None, is_deprecated=False, is_extended=True)

Setting `is_extended=True` records that this value was added to an extensible codelist rather than taken from the published terminology, so [`extended_values()`](../reference/Codelist.md#pointblank.Codelist.extended_values) will list it.

## See Also

[Codelist](../reference/Codelist.md#pointblank.Codelist)  
The list of entries that a codelist is made of.
