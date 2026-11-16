# get_validation()

Get the validation of one dataset from a Validate-based report.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L1655-L1721)

``` python
get_validation(name)
```

The `"validate"` engine of the [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage) method [`validate_conformance()`](../reference/pointblank.SubmissionPackage.validate_conformance.md) makes a [`Validate`](../reference/Validate.md#pointblank.Validate) validation for each dataset. The [get_validation()](../reference/pointblank.ConformanceReport.get_validation.md) method gets one of them, so that its validation report or its failing rows can be looked at. Indexing the report (as in `report["AE"]`) does the same, but [get_validation()](../reference/pointblank.ConformanceReport.get_validation.md) gives a more helpful error for a name that isn't in the report. Reports of the built-in rules and of CDISC CORE don't have validations.

## Parameters

`name: str`  
The name of the dataset, such as `"AE"`. This is case-insensitive.

## Returns

`Validate`  
The validation of the dataset.

## Raises

`KeyError`  
If the report has no validation for that dataset. The error message lists the datasets that it has validations for.

## Examples

Let's check a package where the AE dataset has a subject, `STUDY01-009`, who isn't in DM:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"USUBJID": ["STUDY01-001", "STUDY01-002"], "DOMAIN": ["DM", "DM"]})
ae = pl.DataFrame(
    {
        "USUBJID": ["STUDY01-001", "STUDY01-009"],
        "DOMAIN": ["AE", "AE"],
        "AESEQ": [1, 1],
    }
)

study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae})
report = study.validate_conformance(engine="validate")

report.get_validation("ae")
```

[TABLE]

The last step checks that each `USUBJID` in AE is in DM, and it has one failing test unit. The validation is an ordinary [Validate](../reference/Validate.md#pointblank.Validate) object, so methods such as [`get_step_report()`](../reference/Validate.get_step_report.md#pointblank.Validate.get_step_report) can be used on it to see the failing record.
