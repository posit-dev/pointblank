# summary()

Summarize the package's contents as text.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L1210-L1255)

``` python
summary()
```

The summary is a quick way to check what a package holds, such as after reading it with [`from_folder()`](../reference/pointblank.SubmissionPackage.from_folder.md). It has the study identifier, the standard and its version, the controlled terminology version, whether there is a Define-XML, and the names of the datasets. Lines for the study identifier, the terminology version and the Define-XML are left out when the package doesn't have them. Printing the package gives the same text.

## Returns

`str`  
The summary, over several lines.

## Examples

Let's make a package and summarize it:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"USUBJID": ["S-001", "S-002"]})
ae = pl.DataFrame({"USUBJID": ["S-001"], "AESEQ": [1]})

study = pb.SubmissionPackage(
    datasets={"DM": dm, "AE": ae}, study_id="STUDY01", ct_version="2024-09-27"
)

print(study.summary())
```

    Submission Package
      Study: STUDY01
      Standard: sdtmig 3.4
      CT version: 2024-09-27
      Datasets (2): AE, DM

The package has no `define`, so there's no line for a Define-XML. The datasets are listed in alphabetical order.
