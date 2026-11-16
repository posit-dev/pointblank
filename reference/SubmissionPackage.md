# SubmissionPackage

A study's datasets, gathered for CDISC conformance validation.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L141-L1264)

``` python
SubmissionPackage(
    datasets=dict(),
    define=None,
    ct_version=None,
    standard="sdtmig",
    standard_version="3.4",
    study_id=None,
)
```

A [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage) holds the datasets of a study (SDTM domains, SUPP- qualifiers, RELREC and ADaM datasets) together with the study's Define-XML. Where [`validate_sdtm()`](../reference/validate_sdtm.md#pointblank.validate_sdtm) and [`validate_adam()`](../reference/validate_adam.md#pointblank.validate_adam) look at one dataset at a time, a package knows about all of them, so it can check how they relate: for example, whether every subject in AE has a record in DM. It is the data-level counterpart of [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage), which holds the metadata of many datasets.

A package can be built from DataFrames, as here, or from a folder of XPT and Dataset-JSON files with [`from_folder()`](../reference/pointblank.SubmissionPackage.from_folder.md). Its [`validate_conformance()`](../reference/pointblank.SubmissionPackage.validate_conformance.md) method runs the conformance checks and returns a [`ConformanceReport`](../reference/ConformanceReport.md#pointblank.ConformanceReport). The package also has some helpers for looking across datasets, such as [`orphan_ids()`](../reference/pointblank.SubmissionPackage.orphan_ids.md), which finds the subjects in one dataset that are missing from another.

## Parameters

`datasets: dict[str, Any] = dict()`  
A dictionary that maps each dataset name (a domain code such as `"DM"`, `"AE"`, `"SUPPAE"` or `"ADSL"`) to the dataset, as a Polars or Pandas DataFrame. The names are case-insensitive and are stored in uppercase. The default is an empty dictionary.

`define: Any = None`  
The study's Define-XML, as a path to a `define.xml` file or as a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage) from [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata). With the built-in rules engine, it activates the rules that check the datasets against Define-XML. The CORE engine only accepts a path. The default is `None`.

`ct_version: str | None = None`  
The version of CDISC Controlled Terminology that the study uses (such as `"2024-03-29"`). It is shown by [`summary()`](../reference/pointblank.SubmissionPackage.summary.md) but doesn't affect the checks. To choose the terminology the built-in engine checks against, use the `ct_packages=` argument of [validate_conformance()](../reference/pointblank.SubmissionPackage.validate_conformance.md). The default is `None`.

`standard: str = "sdtmig"`  
The data standard that the datasets follow, `"sdtmig"` or `"adamig"`. The default is `"sdtmig"`.

`standard_version: str = "3.4"`  
The version of the standard, such as `"3.4"` for SDTMIG 3.4. The default is `"3.4"`.

`study_id: str | None = None`  
A study identifier. It is shown by [summary()](../reference/pointblank.SubmissionPackage.summary.md) and used in the labels of the validations made by `validate_conformance(engine="validate")`. The default is `None`.

## Examples

Let's build a package from a Demographics (DM) and an Adverse Events (AE) dataset. One of the adverse events belongs to subject `STUDY01-009`, who isn't in DM:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame(
    {
        "STUDYID": ["STUDY01"] * 3,
        "DOMAIN": ["DM"] * 3,
        "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
        "SEX": ["M", "F", "F"],
    }
)
ae = pl.DataFrame(
    {
        "STUDYID": ["STUDY01"] * 3,
        "DOMAIN": ["AE"] * 3,
        "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-009"],
        "AESEQ": [1, 1, 1],
        "AETERM": ["HEADACHE", "NAUSEA", "RASH"],
    }
)

study = pb.SubmissionPackage(datasets={"dm": dm, "AE": ae}, study_id="STUDY01")

print(study)
```

    Submission Package
      Study: STUDY01
      Standard: sdtmig 3.4
      Datasets (2): AE, DM

The dataset names are stored in uppercase, so `"dm"` became `DM`. The package can find the subject in AE that has no DM record:

``` python
study.orphan_ids("AE")
```

    {'STUDY01-009'}

The validation checks the whole package. The `"validate"` engine builds a [`Validate`](../reference/Validate.md#pointblank.Validate) for each dataset, with the structural checks from the SDTM templates and the checks across datasets:

``` python
report = study.validate_conformance(engine="validate")

print(report)
```

    ConformanceReport
      [AE] 12 steps -- FAIL (1 test units)
      [DM] 9 steps -- PASS

The AE validation has one failing test unit, from the check that each `USUBJID` in AE exists in DM. The default engine, the built-in rules engine, checks the package against the SDTMIG rules instead. See [`validate_conformance()`](../reference/pointblank.SubmissionPackage.validate_conformance.md) for how the engines differ.

## See Also

[validate_sdtmig()](../reference/validate_sdtmig.md#pointblank.validate_sdtmig)  
Check SDTM datasets against the SDTMIG rules, without building a package.

[validate_cdisc_submission()](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission)  
Check a submission with the CDISC CORE engine.

[ConformanceReport](../reference/ConformanceReport.md#pointblank.ConformanceReport)  
The report that validate_conformance() returns.

[MetadataPackage](../reference/MetadataPackage.md#pointblank.MetadataPackage)  
The metadata-level counterpart of a submission package.

## Attributes

| Name | Description |
|----|----|
| [domains](pointblank.SubmissionPackage.domains.md) | The names of the datasets in the package, in alphabetical order. |
| [metadata](pointblank.SubmissionPackage.metadata.md) | The package's Define-XML, as a [MetadataPackage](../reference/MetadataPackage.md#pointblank.MetadataPackage). |

## Functions

| Name | Description |
|----|----|
| [from_folder()](pointblank.SubmissionPackage.from_folder.md) | Build a [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage) from a folder of dataset files. |
| [validate_conformance()](pointblank.SubmissionPackage.validate_conformance.md) | Check the whole package for conformance to its CDISC standard. |
| [get_dataset()](pointblank.SubmissionPackage.get_dataset.md) | Get one of the package's datasets by name. |
| [subject_ids()](pointblank.SubmissionPackage.subject_ids.md) | Get the unique subject identifiers in a dataset. |
| [orphan_ids()](pointblank.SubmissionPackage.orphan_ids.md) | Find the key values in one dataset that are missing from another. |
| [summary()](pointblank.SubmissionPackage.summary.md) | Summarize the package's contents as text. |
