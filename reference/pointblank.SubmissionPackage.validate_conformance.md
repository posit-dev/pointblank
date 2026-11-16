# validate_conformance()

Check the whole package for conformance to its CDISC standard.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L685-L913)

``` python
validate_conformance(
    agency=None,
    engine="native",
    cross_dataset=True,
    thresholds=None,
    interrogate=True,
    *,
    standard=None,
    version=None,
    ct_packages=None,
    define_xml=None,
    controlled_terminology=None,
    core=None,
    core_cwd=None,
    cache=None,
    workdir=None,
)
```

A submission has to conform to the CDISC standard that it uses, both within each dataset and across the datasets. The [validate_conformance()](../reference/pointblank.SubmissionPackage.validate_conformance.md) method checks this for all the datasets of the package at once and returns a [`ConformanceReport`](../reference/ConformanceReport.md#pointblank.ConformanceReport). There are three ways of doing the checks, chosen with `engine=`:

- `"native"` (the default) runs Pointblank's bundled catalog of conformance rules for the package's standard and version, as [`validate_sdtmig()`](../reference/validate_sdtmig.md#pointblank.validate_sdtmig) does. The catalog has 362 rules for SDTMIG 3.4, which check the values of variables against their controlled terminology, the format of dates, the variables and domains that are required, and (if there's a Define-XML) the agreement between the datasets and their Define-XML. SDTMIG 3.4 is the only standard and version with a bundled catalog, so for any other (such as an ADaM package), the `"validate"` checks are run instead.
- `"validate"` builds a [`Validate`](../reference/Validate.md#pointblank.Validate) plan for each dataset, with the structural checks of [`validate_sdtm()`](../reference/validate_sdtm.md#pointblank.validate_sdtm) or [`validate_adam()`](../reference/validate_adam.md#pointblank.validate_adam), and (if `cross_dataset=True`) checks that the datasets agree with each other. These cross-dataset checks are that every `USUBJID` of an SDTM domain is in DM, that a SUPP- or RELREC record's `RDOMAIN` is a dataset in the package and its `USUBJID` is in DM, that a SUPP- record's `IDVAR` and `IDVARVAL` identify a record of its parent domain, that every `USUBJID` of ADSL is in DM, and that every `USUBJID` of another ADaM dataset is in ADSL. The rules catalog doesn't have these checks, so the `"validate"` engine is the one that finds orphan records.
- `"core"` runs the CDISC CORE engine (the `cdisc-rules-engine` command-line tool, which has to be installed separately), with the published CDISC and agency rules. The datasets are written out as XPT files (or read from the package's `source_folder`, if it was made with [`from_folder()`](../reference/pointblank.SubmissionPackage.from_folder.md)), CORE is run on them, and its report is read back. This is what [`validate_cdisc_submission()`](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission) does.

## Parameters

`agency: str | None = None`  
The regulatory agency that the submission is for, such as `"FDA"` or `"PMDA"`. This is a label that is shown in the report. It doesn't change which rules are run. The default is `None`.

`engine: str = "native"`  
The engine for the checks: `"native"`, `"validate"` or `"core"`. The default is `"native"`.

`cross_dataset: bool = True`  
Whether to add the cross-dataset checks to the `"validate"` engine's plans. The default is `True`.

`thresholds: Any = None`  
Thresholds for each of the `"validate"` engine's plans, given as for [`Validate`](../reference/Validate.md#pointblank.Validate). They set the severity (warning, error or critical) of a failing step. The default is `None`.

`interrogate: bool = True`  
Whether the `"validate"` engine interrogates its plans before returning them. The default is `True`.

`standard: str | None = None`  
The CDISC standard to check against, for the `"native"` and `"core"` engines. The default is the package's `standard`.

`version: str | None = None`  
The version of the standard, for the `"native"` and `"core"` engines. The default is the package's `standard_version`.

`ct_packages: list[str] | None = None`  
The bundled controlled terminology packages for the `"native"` engine, as a list of names such as `["sdtm-ct-2024-09-27"]`. The default is the latest bundled package.

`define_xml: Any = None`  
A Define-XML for the `"native"` engine, as a path or a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage). The default is the package's `define`. Without a Define-XML, the 12 rules that need one are not applicable.

`controlled_terminology: str | Sequence[str] | None = None`  
The controlled terminology package or packages for the `"core"` engine, such as `"sdtmct-2024-03-29"`. They are given to CORE's `-ct` option. The default is `None`.

`core: str | Sequence[str] | None = None`  
How to run CORE, for the `"core"` engine. This can be the path or name of the CORE executable, or a command as a list (such as `["python", "core.py"]`). With the default of `None`, the command is taken from the `POINTBLANK_CDISC_CORE` environment variable, or else CORE is looked for on the `PATH`.

`core_cwd: str | Path | None = None`  
The directory to run CORE from, for the `"core"` engine. This is needed when CORE is run from a checkout of its repository, since it looks for its `resources/` folder in the current directory. The default is `None`.

`cache: str | Path | None = None`  
The path to CORE's rules cache (its `-ca` option), for the `"core"` engine. The default is `None`.

`workdir: str | Path | None = None`  
A directory for the XPT files and the CORE report, for the `"core"` engine. With the default of `None`, a temporary directory is used and then deleted.

## Returns

`ConformanceReport`  
The report. It holds the results of the rules for the `"native"` engine, a validation for each dataset for the `"validate"` engine, and CORE's results for the `"core"` engine.

## Raises

`ValueError`  
If `engine` isn't one of `"native"`, `"validate"` or `"core"`.

`FileNotFoundError`  
If one of the `ct_packages` isn't bundled with Pointblank.

## Examples

Let's make a package with a DM and an AE dataset. A `SEX` value in DM isn't in the SEX codelist, and the second subject in AE (`STUDY01-009`) has no record in DM:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame(
    {
        "STUDYID": ["STUDY01"] * 3,
        "DOMAIN": ["DM"] * 3,
        "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
        "SUBJID": ["001", "002", "003"],
        "SITEID": ["S01"] * 3,
        "SEX": ["M", "F", "FEMALE"],
        "COUNTRY": ["USA"] * 3,
    }
)
ae = pl.DataFrame(
    {
        "STUDYID": ["STUDY01"] * 2,
        "DOMAIN": ["AE"] * 2,
        "USUBJID": ["STUDY01-001", "STUDY01-009"],
        "AESEQ": [1, 1],
        "AETERM": ["HEADACHE", "NAUSEA"],
        "AEDECOD": ["Headache", "Nausea"],
    }
)

study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae}, study_id="STUDY01")

report = study.validate_conformance()

print(report)
```

    ConformanceReport (Built-in Rules)
      sdtmig 3.4
      362 rules (fail=11, not_applicable=12, pass=339)
      11 issues -- FAIL

Of the 362 rules, 11 fail. The 12 that are not applicable need a Define-XML. Here are the failing ones:

``` python
for rule in report.rules(status="fail"):
    print(rule.rule_id, rule.dataset, rule.message)
```

    SDTM-007 DM SEX value is not in the SEX codelist.
    SDTM-032 DM DM domain is missing one or more required demographic variables (SEX, RACE, COUNTRY).
    SDTM-033 DM DM domain must include RFSTDTC (Subject Reference Start Date/Time) and RFENDTC (Subject Reference End Date/Time).
    SDTM-034 DM DM domain must include SITEID, AGE, and AGEU.
    SDTM-035 DM DM domain must include treatment arm variables: ARMCD, ARM, ACTARMCD, ACTARM.
    SDTM-037 AE AE domain must include AETERM, AEDECOD (MedDRA preferred term), and AEBODSYS (body system).
    SDTM-038 AE AE domain must include AESER (serious flag).
    SDTM-121 TS Required domain(s) missing: TS
    SDTM-122 TA Required domain(s) missing: TA
    SDTM-235 AE AESTDTC must be present in AE.
    SDTM-289 AE AESER must be present in AE.

Most of them are about variables that the datasets lack, such as `RACE` and `AESER`, and SDTM-121 and SDTM-122 fail since the package has no Trial Summary (TS) or Trial Arms (TA) dataset. SDTM-007 finds the `SEX` value of `"FEMALE"`. The rules don't look for orphan records, though, so for those the `"validate"` engine is needed:

``` python
report = study.validate_conformance(engine="validate")

for issue in report.issues():
    print(issue)
```

    {'dataset': 'AE', 'step': 14, 'assertion': 'specially', 'column': None, 'n_failed': 1, 'n': 2, 'severity': None}
    {'dataset': 'DM', 'step': 14, 'assertion': 'col_vals_expr', 'column': None, 'n_failed': 1, 'n': 3, 'severity': None}

The `"validate"` engine has a validation for each dataset. In AE, the last step is the check that each `USUBJID` is in DM, and it finds the one orphan record. In DM, the failing step is the length check on `SEX`, since `"FEMALE"` is longer than the two characters that SDTM allows. Each validation can be looked at with [`get_validation()`](../reference/pointblank.ConformanceReport.get_validation.md).

## See Also

[validate_sdtmig()](../reference/validate_sdtmig.md#pointblank.validate_sdtmig)  
Run the rules catalog on a dict of datasets.

[validate_cdisc_submission()](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission)  
Run CDISC CORE on a folder of datasets.

[ConformanceReport](../reference/ConformanceReport.md#pointblank.ConformanceReport)  
The report that is returned.
