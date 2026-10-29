# SubmissionPackage

A study's datasets, gathered for CDISC conformance validation.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L141-L1263)

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

A package can be built from DataFrames, as here, or from a folder of XPT and Dataset-JSON files with [`from_folder()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.from_folder). Its [`validate_conformance()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.validate_conformance) method runs the conformance checks and returns a [`ConformanceReport`](../reference/ConformanceReport.md#pointblank.ConformanceReport). The package also has some helpers for looking across datasets, such as [`orphan_ids()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.orphan_ids), which finds the subjects in one dataset that are missing from another.

## Parameters

`datasets: dict[str, Any] = dict()`  
A dictionary that maps each dataset name (a domain code such as `"DM"`, `"AE"`, `"SUPPAE"` or `"ADSL"`) to the dataset, as a Polars or Pandas DataFrame. The names are case-insensitive and are stored in uppercase. The default is an empty dictionary.

`define: Any = None`  
The study's Define-XML, as a path to a `define.xml` file or as a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage) from [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata). With the built-in rules engine, it activates the rules that check the datasets against Define-XML. The CORE engine only accepts a path. The default is `None`.

`ct_version: str | None = None`  
The version of CDISC Controlled Terminology that the study uses (such as `"2024-03-29"`). It is shown by [`summary()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.summary) but doesn't affect the checks. To choose the terminology the built-in engine checks against, use the `ct_packages=` argument of [validate_conformance()](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.validate_conformance). The default is `None`.

`standard: str = "sdtmig"`  
The data standard that the datasets follow, `"sdtmig"` or `"adamig"`. The default is `"sdtmig"`.

`standard_version: str = "3.4"`  
The version of the standard, such as `"3.4"` for SDTMIG 3.4. The default is `"3.4"`.

`study_id: str | None = None`  
A study identifier. It is shown by `summary()` and used in the labels of the validations made by `validate_conformance(engine="validate")`. The default is `None`.

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

The AE validation has one failing test unit, from the check that each `USUBJID` in AE exists in DM. The default engine, the built-in rules engine, checks the package against the SDTMIG rules instead. See [`validate_conformance()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.validate_conformance) for how the engines differ.

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
| [domains](#domains) | The names of the datasets in the package, in alphabetical order. |
| [metadata](#metadata) | The package's Define-XML, as a [MetadataPackage](../reference/MetadataPackage.md#pointblank.MetadataPackage). |

### domains

The names of the datasets in the package, in alphabetical order.

`domains: list[str]`

The names are the domain codes (such as `"AE"` and `"DM"`) under which the datasets were added, in uppercase. This is a property, so it's used without parentheses.

#### Examples

Let's make a package with three datasets and get their names:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"USUBJID": ["S-001", "S-002"]})
ae = pl.DataFrame({"USUBJID": ["S-001"], "AESEQ": [1]})
suppae = pl.DataFrame({"USUBJID": ["S-001"], "RDOMAIN": ["AE"]})

study = pb.SubmissionPackage(datasets={"dm": dm, "SUPPAE": suppae, "ae": ae})

study.domains
```

    ['AE', 'DM', 'SUPPAE']

The names are in uppercase and in alphabetical order, whatever order the datasets were given in. A name can also be tested with `in`, which ignores case (`"ae" in study` is `True`), and `len(study)` gives the number of datasets.

### metadata

The package's Define-XML, as a [MetadataPackage](../reference/MetadataPackage.md#pointblank.MetadataPackage).

`metadata: MetadataPackage | None`

If the package's `define` is a path to a `define.xml` file, the file is imported with [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata) the first time this property is used, and the result is kept for later use. If `define` is already a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage), that is returned. The metadata has a [`MetadataImport`](../reference/MetadataImport.md#pointblank.MetadataImport) for each dataset that the Define-XML describes, with its variables and codelists. The property is `None` if the package has no Define-XML. This is a property, so it's used without parentheses.

#### Examples

Let's make a package with the DM dataset and the Define-XML file that come with Pointblank:

``` python
import pointblank as pb
import pyreadstat

dm, _ = pyreadstat.read_xport(pb.load_metadata_example("dm.xpt"))

study = pb.SubmissionPackage(
    datasets={"DM": dm}, define=pb.load_metadata_example("define.xml")
)

print(study.metadata)
```

    Metadata Package
      Name: STUDY.XYZ789
      Description: CDISC Define-XML 2.0 study metadata
      Datasets: 2

      [DM] 13 variables
      [AE] 10 variables

The Define-XML describes two datasets, DM and AE, though the package only has DM. The metadata of a dataset can be looked at in more detail, such as the values allowed by the codelist of a variable:

``` python
race = study.metadata["DM"].get_variable("RACE")

race.allowed_values
```

    ['WHITE',
     'BLACK OR AFRICAN AMERICAN',
     'ASIAN',
     'AMERICAN INDIAN OR ALASKA NATIVE',
     'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER']

`BLACK` isn't one of them, though one of the subjects in DM has it as their `RACE`. Without a `define`, the property is `None`.

## Methods

| Name | Description |
|----|----|
| [from_folder()](#from_folder) | Build a [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage) from a folder of dataset files. |
| [validate_conformance()](#validate_conformance) | Check the whole package for conformance to its CDISC standard. |
| [get_dataset()](#get_dataset) | Get one of the package's datasets by name. |
| [subject_ids()](#subject_ids) | Get the unique subject identifiers in a dataset. |
| [orphan_ids()](#orphan_ids) | Find the key values in one dataset that are missing from another. |
| [summary()](#summary) | Summarize the package's contents as text. |

### from_folder()

Build a [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage) from a folder of dataset files.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L267-L403)

``` python
from_folder(
    path,
    define=None,
    standard="sdtmig",
    standard_version="3.4",
    ct_version=None,
    study_id=None,
)
```

Submission datasets are usually delivered as files, one per dataset, in a single folder. The [from_folder()](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.from_folder) method reads the folder so the datasets don't have to be loaded one by one. It reads these files (the folder's subfolders are not searched):

- SAS Transport files (`.xpt`), which need the `pyreadstat` package. The dataset name is the file name without its extension, in uppercase, so `dm.xpt` becomes `DM`.
- CDISC Dataset-JSON files (`.json`), in the Dataset-JSON 1.1 layout (top-level `columns` and [rows](../reference/Comparison.rows.md#pointblank.Comparison.rows)) or the older layout (`clinicalData` or `referenceData`). The dataset name is taken from the file's `name` (or its item group OID), or else from the file name. `.json` files that aren't Dataset-JSON are skipped.
- A Define-XML file: an `.xml` file whose name starts with `define`. It is used as the package's `define` unless `define=` is given.

The datasets are read into Pandas DataFrames. The package remembers the folder, so that `validate_conformance(engine="core")` can give the files straight to the CORE engine instead of writing the datasets out again.

#### Parameters

`path: str | Path`  
The path to the folder that holds the dataset files.

`define: str | Path | Any | None = None`  
The study's Define-XML, as a path to a `define.xml` file or a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage). If `None` (the default), a Define-XML file in the folder is used, if there is one.

`standard: str = "sdtmig"`  
The data standard that the datasets follow, `"sdtmig"` or `"adamig"`. The default is `"sdtmig"`.

`standard_version: str = "3.4"`  
The version of the standard. The default is `"3.4"`.

`ct_version: str | None = None`  
The version of CDISC Controlled Terminology that the study uses, which is shown by [`summary()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.summary). The default is `None`.

`study_id: str | None = None`  
A study identifier. The default is `None`.

#### Returns

`SubmissionPackage`  
A package with a dataset for each dataset file in the folder.

#### Raises

`NotADirectoryError`  
If `path` isn't a folder.

`ImportError`  
If the folder has an `.xpt` file and `pyreadstat` isn't installed.

#### Examples

Let's make a folder with a DM dataset as an XPT file and a Define-XML file for the study. Both files come with Pointblank:

``` python
import pointblank as pb
import shutil
import tempfile
from pathlib import Path

folder = Path(tempfile.mkdtemp())

shutil.copy(pb.load_metadata_example("dm.xpt"), folder / "dm.xpt")
shutil.copy(pb.load_metadata_example("define.xml"), folder / "define.xml")

study = pb.SubmissionPackage.from_folder(folder, study_id="XYZ789")

print(study)
```

    Submission Package
      Study: XYZ789
      Standard: sdtmig 3.4
      Define-XML: present
      Datasets (1): DM

The package has the `DM` dataset from `dm.xpt`, and it found the Define-XML file. The dataset is a Pandas DataFrame:

``` python
study.get_dataset("DM").head(3)
```

|  | STUDYID | DOMAIN | USUBJID | SUBJID | RFSTDTC | RFENDTC | SITEID | AGE | SEX | RACE | ARMCD | ARM |
|----|----|----|----|----|----|----|----|----|----|----|----|----|
| 0 | XYZ789 | DM | XYZ789-101 | 101 | 2024-01-15 | 2024-07-15 | S01 | 45.0 | M | WHITE | TRT | Active 10mg |
| 1 | XYZ789 | DM | XYZ789-102 | 102 | 2024-01-20 | 2024-07-20 | S01 | 62.0 | F | BLACK | PBO | Placebo |
| 2 | XYZ789 | DM | XYZ789-103 | 103 | 2024-02-01 | 2024-08-01 | S02 | 38.0 | M | ASIAN | TRT | Active 10mg |

Since a Define-XML file was found, the built-in rules engine also runs the rules that check the datasets against it. Here, one of the subjects has a `RACE` of `BLACK`, which isn't a term in the study's codelist for `RACE`:

``` python
report = study.validate_conformance()

[i["message"] for i in report.issues() if i["rule_type"] == "DEFINE_CODELIST_CHECK"]
```

    ['DM.RACE contains a value not in the Define-XML declared codelist.']

The other issues in the report come from rules that don't need Define-XML, such as the rules for variables that DM should have.

### validate_conformance()

Check the whole package for conformance to its CDISC standard.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L684-L912)

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

A submission has to conform to the CDISC standard that it uses, both within each dataset and across the datasets. The [validate_conformance()](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.validate_conformance) method checks this for all the datasets of the package at once and returns a [`ConformanceReport`](../reference/ConformanceReport.md#pointblank.ConformanceReport). There are three ways of doing the checks, chosen with `engine=`:

- `"native"` (the default) runs Pointblank's bundled catalog of conformance rules for the package's standard and version, as [`validate_sdtmig()`](../reference/validate_sdtmig.md#pointblank.validate_sdtmig) does. The catalog has 362 rules for SDTMIG 3.4, which check the values of variables against their controlled terminology, the format of dates, the variables and domains that are required, and (if there's a Define-XML) the agreement between the datasets and their Define-XML. SDTMIG 3.4 is the only standard and version with a bundled catalog, so for any other (such as an ADaM package), the `"validate"` checks are run instead.
- `"validate"` builds a [`Validate`](../reference/Validate.md#pointblank.Validate) plan for each dataset, with the structural checks of [`validate_sdtm()`](../reference/validate_sdtm.md#pointblank.validate_sdtm) or [`validate_adam()`](../reference/validate_adam.md#pointblank.validate_adam), and (if `cross_dataset=True`) checks that the datasets agree with each other. These cross-dataset checks are that every `USUBJID` of an SDTM domain is in DM, that a SUPP- or RELREC record's `RDOMAIN` is a dataset in the package and its `USUBJID` is in DM, that a SUPP- record's `IDVAR` and `IDVARVAL` identify a record of its parent domain, that every `USUBJID` of ADSL is in DM, and that every `USUBJID` of another ADaM dataset is in ADSL. The rules catalog doesn't have these checks, so the `"validate"` engine is the one that finds orphan records.
- `"core"` runs the CDISC CORE engine (the `cdisc-rules-engine` command-line tool, which has to be installed separately), with the published CDISC and agency rules. The datasets are written out as XPT files (or read from the package's `source_folder`, if it was made with [`from_folder()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.from_folder)), CORE is run on them, and its report is read back. This is what [`validate_cdisc_submission()`](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission) does.

#### Parameters

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

#### Returns

`ConformanceReport`  
The report. It holds the results of the rules for the `"native"` engine, a validation for each dataset for the `"validate"` engine, and CORE's results for the `"core"` engine.

#### Raises

`ValueError`  
If `engine` isn't one of `"native"`, `"validate"` or `"core"`.

`FileNotFoundError`  
If one of the `ct_packages` isn't bundled with Pointblank.

#### Examples

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

The `"validate"` engine has a validation for each dataset. In AE, the last step is the check that each `USUBJID` is in DM, and it finds the one orphan record. In DM, the failing step is the length check on `SEX`, since `"FEMALE"` is longer than the two characters that SDTM allows. Each validation can be looked at with [`get_validation()`](../reference/ConformanceReport.md#pointblank.ConformanceReport.get_validation).

#### See Also

[validate_sdtmig()](../reference/validate_sdtmig.md#pointblank.validate_sdtmig)  
Run the rules catalog on a dict of datasets.

[validate_cdisc_submission()](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission)  
Run CDISC CORE on a folder of datasets.

[ConformanceReport](../reference/ConformanceReport.md#pointblank.ConformanceReport)  
The report that is returned.

### get_dataset()

Get one of the package's datasets by name.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L446-L500)

``` python
get_dataset(name)
```

The dataset is returned as it was given to the package, so a Polars DataFrame stays a Polars DataFrame. Indexing the package (as in `study["DM"]`) also gets a dataset, but [get_dataset()](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.get_dataset) gives a more helpful error for a name that isn't in the package.

#### Parameters

`name: str`  
The name of the dataset, such as `"DM"`. This is case-insensitive.

#### Returns

`Any`  
The dataset, as a DataFrame.

#### Raises

`KeyError`  
If the package has no dataset with that name. The error message lists the names that the package has.

#### Examples

Let's make a package with two datasets and get one of them:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"USUBJID": ["S-001", "S-002"], "SEX": ["M", "F"]})
ae = pl.DataFrame({"USUBJID": ["S-001"], "AESEQ": [1], "AETERM": ["HEADACHE"]})

study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae})

study.get_dataset("ae")
```

shape: (1, 3)

| USUBJID | AESEQ | AETERM     |
|---------|-------|------------|
| str     | i64   | str        |
| "S-001" | 1     | "HEADACHE" |

The name is case-insensitive, so `"ae"` gets the `AE` dataset. Asking for a dataset that isn't in the package raises a `KeyError`:

``` python
try:
    study.get_dataset("LB")
except KeyError as e:
    print(e)
```

    "No dataset named 'LB'. Available: ['AE', 'DM']"

The message lists the datasets that the package has.

### subject_ids()

Get the unique subject identifiers in a dataset.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L563-L613)

``` python
subject_ids(dataset="DM")
```

`USUBJID` identifies each subject across all the datasets of a study. DM has one record per subject, so its `USUBJID` values are the study's subjects, and the subjects in the other datasets should all be among them. The [subject_ids()](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.subject_ids) method gets these values for one dataset, which is useful for comparing the subjects of two datasets.

#### Parameters

`dataset: str = "DM"`  
The name of the dataset, which is case-insensitive. The default is `"DM"`.

#### Returns

`set`  
The different values of `USUBJID` in the dataset, leaving out nulls. The set is empty if the package has no dataset with that name, or if the dataset has no `USUBJID` column.

#### Examples

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

For a dataset that isn't in the package, [subject_ids()](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.subject_ids) returns an empty set rather than raising an error. To find the subjects that are missing from DM, use [`orphan_ids()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.orphan_ids).

### orphan_ids()

Find the key values in one dataset that are missing from another.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L615-L680)

``` python
orphan_ids(
    child,
    parent="DM",
    column="USUBJID",
)
```

In a study, the records of one dataset often refer to those of another. Each subject in AE should have a record in DM, for example. A value with nothing to refer to (an *orphan*) usually means that a record is missing or that a key was mistyped. The [orphan_ids()](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.orphan_ids) method finds these values, by default the subjects of a dataset that are missing from DM.

#### Parameters

`child: str`  
The name of the dataset whose values are looked up, such as `"AE"`.

`parent: str = "DM"`  
The name of the dataset that the values should be in. The default is `"DM"`.

`column: str = "USUBJID"`  
The key column, which is compared between the two datasets. The default is `"USUBJID"`.

#### Returns

`set`  
The values of `column` in `child` that aren't in `column` of `parent`. Null values are left out. If `child` has no such column, the set is empty. If `parent` has no such column, every value of `child` is returned.

#### Raises

`KeyError`  
If `child` or `parent` isn't a dataset in the package.

#### Examples

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

### summary()

Summarize the package's contents as text.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L1209-L1254)

``` python
summary()
```

The summary is a quick way to check what a package holds, such as after reading it with [`from_folder()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.from_folder). It has the study identifier, the standard and its version, the controlled terminology version, whether there is a Define-XML, and the names of the datasets. Lines for the study identifier, the terminology version and the Define-XML are left out when the package doesn't have them. Printing the package gives the same text.

#### Returns

`str`  
The summary, over several lines.

#### Examples

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
