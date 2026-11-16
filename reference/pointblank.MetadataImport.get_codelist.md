# get_codelist()

Get a codelist by name.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_types.py#L952-L989)

``` python
get_codelist(name)
```

The codelists are keyed by name in `codelists`, and a variable that uses one names it in its `codelist_ref`.

## Parameters

`name: str`  
The name of the codelist, such as `"Sex"`.

## Returns

`Codelist`  
The [`Codelist`](../reference/Codelist.md#pointblank.Codelist) with that name.

## Raises

`KeyError`  
If there's no codelist with that name. The error message lists the available names.

## Examples

``` python
import pointblank as pb

dm = pb.import_metadata(pb.load_metadata_example("define.xml"), dataset="DM")
race_ref = dm.get_variable("RACE").codelist_ref

dm.get_codelist(race_ref).to_set()
```

    ['WHITE',
     'BLACK OR AFRICAN AMERICAN',
     'ASIAN',
     'AMERICAN INDIAN OR ALASKA NATIVE',
     'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER']

Starting from the variable's `codelist_ref` finds the codelist that applies to `RACE`, and [`to_set()`](../reference/Codelist.md#pointblank.Codelist.to_set) gives its permitted values.
