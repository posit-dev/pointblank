# orphan_ids()

Find the key values in one dataset that are missing from another.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L616-L681)

``` python
orphan_ids(
    child,
    parent="DM",
    column="USUBJID",
)
```

In a study, the records of one dataset often refer to those of another. Each subject in AE should have a record in DM, for example. A value with nothing to refer to (an *orphan*) usually means that a record is missing or that a key was mistyped. The [orphan_ids()](../reference/pointblank.SubmissionPackage.orphan_ids.md) method finds these values, by default the subjects of a dataset that are missing from DM.

## Parameters

`child: str`  
The name of the dataset whose values are looked up, such as `"AE"`.

`parent: str = "DM"`  
The name of the dataset that the values should be in. The default is `"DM"`.

`column: str = "USUBJID"`  
The key column, which is compared between the two datasets. The default is `"USUBJID"`.

## Returns

`set`  
The values of `column` in `child` that aren't in `column` of `parent`. Null values are left out. If `child` has no such column, the set is empty. If `parent` has no such column, every value of `child` is returned.

## Raises

`KeyError`  
If `child` or `parent` isn't a dataset in the package.

## Examples

Let's make a package where the AE dataset has a subject, `S-009`, who isn't in DM:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"USUBJID": ["S-001", "S-002", "S-003"]})
ae = pl.DataFrame({"USUBJID": ["S-001", "S-009", "S-009"], "AESEQ": [1, 1, 2]})

study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae})

study.orphan_ids("AE")
```

    {'S-009'}

The orphan subject is found once, though they have two records in AE. The datasets that are compared can be changed with `parent=`. Here, a SUPPAE dataset holds supplemental qualifiers for adverse events, so each of its subjects should have a record in AE:

``` python
suppae = pl.DataFrame({"USUBJID": ["S-001", "S-002"], "QNAM": ["AETRTEM", "AETRTEM"]})
study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae, "SUPPAE": suppae})

print(study.orphan_ids("SUPPAE"))
print(study.orphan_ids("SUPPAE", parent="AE"))
```

    set()
    {'S-002'}

Both SUPPAE subjects are in DM, but `S-002` has no adverse events, so their supplemental qualifier has no AE record to belong to. A key other than `USUBJID` can be compared with `column=`.
