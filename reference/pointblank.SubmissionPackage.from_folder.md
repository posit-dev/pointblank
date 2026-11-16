# from_folder()

Build a [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage) from a folder of dataset files.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L268-L404)

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

Submission datasets are usually delivered as files, one per dataset, in a single folder. The [from_folder()](../reference/pointblank.SubmissionPackage.from_folder.md) method reads the folder so the datasets don't have to be loaded one by one. It reads these files (the folder's subfolders are not searched):

- SAS Transport files (`.xpt`), which need the `pyreadstat` package. The dataset name is the file name without its extension, in uppercase, so `dm.xpt` becomes `DM`.
- CDISC Dataset-JSON files (`.json`), in the Dataset-JSON 1.1 layout (top-level `columns` and [rows](../reference/Comparison.rows.md#pointblank.Comparison.rows)) or the older layout (`clinicalData` or `referenceData`). The dataset name is taken from the file's `name` (or its item group OID), or else from the file name. `.json` files that aren't Dataset-JSON are skipped.
- A Define-XML file: an `.xml` file whose name starts with `define`. It is used as the package's `define` unless `define=` is given.

The datasets are read into Pandas DataFrames. The package remembers the folder, so that `validate_conformance(engine="core")` can give the files straight to the CORE engine instead of writing the datasets out again.

## Parameters

`path: str | Path`  
The path to the folder that holds the dataset files.

`define: str | Path | Any | None = None`  
The study's Define-XML, as a path to a `define.xml` file or a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage). If `None` (the default), a Define-XML file in the folder is used, if there is one.

`standard: str = "sdtmig"`  
The data standard that the datasets follow, `"sdtmig"` or `"adamig"`. The default is `"sdtmig"`.

`standard_version: str = "3.4"`  
The version of the standard. The default is `"3.4"`.

`ct_version: str | None = None`  
The version of CDISC Controlled Terminology that the study uses, which is shown by [`summary()`](../reference/pointblank.SubmissionPackage.summary.md). The default is `None`.

`study_id: str | None = None`  
A study identifier. The default is `None`.

## Returns

`SubmissionPackage`  
A package with a dataset for each dataset file in the folder.

## Raises

`NotADirectoryError`  
If `path` isn't a folder.

`ImportError`  
If the folder has an `.xpt` file and `pyreadstat` isn't installed.

## Examples

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
