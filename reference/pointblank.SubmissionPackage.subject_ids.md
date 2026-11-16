# subject_ids()

Get the unique subject identifiers in a dataset.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L564-L614)

``` python
subject_ids(dataset="DM")
```

`USUBJID` identifies each subject across all the datasets of a study. DM has one record per subject, so its `USUBJID` values are the study's subjects, and the subjects in the other datasets should all be among them. The [subject_ids()](../reference/pointblank.SubmissionPackage.subject_ids.md) method gets these values for one dataset, which is useful for comparing the subjects of two datasets.

## Parameters

`dataset: str = "DM"`  
The name of the dataset, which is case-insensitive. The default is `"DM"`.

## Returns

`set`  
The different values of `USUBJID` in the dataset, leaving out nulls. The set is empty if the package has no dataset with that name, or if the dataset has no `USUBJID` column.

## Examples

Let's make a package where the AE dataset has two records for one subject:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"USUBJID": ["S-001", "S-002", "S-003"]})
ae = pl.DataFrame({"USUBJID": ["S-001", "S-001", "S-003"], "AESEQ": [1, 2, 1]})

study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae})

sorted(study.subject_ids("AE"))
```

    ['S-001', 'S-003']

Each subject appears once in the set, however many records they have. Sets can be compared directly, so here's a check that every subject with an adverse event is in DM:

``` python
study.subject_ids("AE") <= study.subject_ids()
```

    True

For a dataset that isn't in the package, [subject_ids()](../reference/pointblank.SubmissionPackage.subject_ids.md) returns an empty set rather than raising an error. To find the subjects that are missing from DM, use [`orphan_ids()`](../reference/pointblank.SubmissionPackage.orphan_ids.md).
