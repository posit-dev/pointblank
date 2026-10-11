# SchemaDiff


Differences between the schemas of the source and target tables.


Usage

``` python
SchemaDiff(
    only_in_source=list(),
    only_in_target=list(),
    dtype_changed=list(),
    renamed=list(),
    case_only=list(),
    reordered=False,
)
```


Column names refer to the source table except in `only_in_target` (and the second element of `renamed` and `case_only` pairs), which refer to the target table.


## Attributes


`only_in_source: list[str]`  
Columns present in the source but not the target.

`only_in_target: list[str]`  
Columns present in the target but not the source.

`dtype_changed: list[tuple[str, str, str]]`  
`(column, source_dtype, target_dtype)` for shared columns whose data types differ.

`renamed: list[tuple[str, str]]`  
`(source_column, target_column)` pairs paired up through `column_map=`.

`case_only: list[tuple[str, str]]`  
`(source_column, target_column)` pairs whose names differ only by letter case. These are *not* paired up automatically (they also appear in `only_in_source`/`only_in_target`); use `column_map=` to compare them.

`reordered: bool`  
`True` if the shared columns appear in a different order in the two tables.


## Attributes

| Name | Description |
|----|----|
| [matches](#matches) | `True` if the schemas match (explicit renames through `column_map=` are allowed). |

------------------------------------------------------------------------


### matches


`True` if the schemas match (explicit renames through `column_map=` are allowed).


`matches: bool`
