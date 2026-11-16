# n_datasets

The number of datasets that were checked.

`n_datasets: int`

For a report of the built-in rules, this is the number of datasets that were given, and for a Validate-based report, the number of validations. For a CDISC CORE report, it is the number of datasets in CORE's `Dataset_Details`. Datasets that a rule requires but that are missing (such as TS) aren't counted. This is a property, so it's used without parentheses.

## Examples

Let's check a package of two datasets with the built-in rules:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"STUDYID": ["STUDY01"], "USUBJID": ["STUDY01-001"]})
ae = pl.DataFrame({"STUDYID": ["STUDY01"], "USUBJID": ["STUDY01-001"], "AESEQ": [1]})

report = pb.validate_sdtmig({"DM": dm, "AE": ae})

report.n_datasets
```

    2

The two datasets were checked, though some rules are about the TS and TA datasets, which the package doesn't have.
