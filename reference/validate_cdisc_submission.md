# validate_cdisc_submission()

Check a CDISC submission with the CDISC CORE engine.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L2973-L3145)

``` python
validate_cdisc_submission(
    source,
    standard=None,
    version=None,
    define=None,
    controlled_terminology=None,
    agency=None,
    ct_version=None,
    study_id=None,
    core=None,
    core_cwd=None,
    cache=None,
    workdir=None,
)
```

CDISC CORE is CDISC's open-source engine for its conformance rules, and it has the published rules of CDISC and of the FDA and PMDA. The [validate_cdisc_submission()](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission) function runs CORE on a submission in a single call: it makes a [`SubmissionPackage`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage) from the `source` (unless it is one already), runs CORE on it, and reads CORE's results into a [`ConformanceReport`](../reference/ConformanceReport.md#pointblank.ConformanceReport). This is the same as calling the package's [`validate_conformance()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.validate_conformance) method with `engine="core"`.

CORE is a separate command-line tool (`cdisc-rules-engine`) that has to be installed. If it isn't on the `PATH`, give the command with `core=` or in the `POINTBLANK_CDISC_CORE` environment variable. For checks that need no other software, use [`validate_sdtmig()`](../reference/validate_sdtmig.md#pointblank.validate_sdtmig), which runs Pointblank's bundled SDTMIG rules.

## Parameters

`source: str | Path | dict | SubmissionPackage`  
The submission to check. This can be the path of a folder of datasets (`.xpt` or Dataset-JSON files, which are read with [`SubmissionPackage.from_folder()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.from_folder)), a dict of DataFrames keyed by dataset name, or a [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage).

`standard: str | None = None`  
The CDISC standard, such as `"sdtmig"`. The default is `"sdtmig"`, or the package's `standard` if `source` is a [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage).

`version: str | None = None`  
The version of the standard, such as `"3.4"`. The default is `"3.4"`, or the package's `standard_version` if `source` is a [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage).

`define: str | Path | Any | None = None`  
The path of the study's Define-XML, which CORE uses in its checks. For a folder, a `define*.xml` file in it is found without this. It is ignored if `source` is a [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage), whose own `define` is used. The default is `None`.

`controlled_terminology: str | Sequence[str] | None = None`  
The controlled terminology package or packages for CORE's `-ct` option, such as `"sdtmct-2024-03-29"`. The default is `None`.

`agency: str | None = None`  
The regulatory agency, such as `"FDA"` or `"PMDA"`, which is shown in the report. It doesn't change the rules that CORE runs. The default is `None`.

`ct_version: str | None = None`  
The controlled terminology version to record in the package. It is only shown in the package's summary and isn't given to CORE. It is ignored if `source` is a [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage). The default is `None`.

`study_id: str | None = None`  
A study identifier to record in the package. It is ignored if `source` is a [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage). The default is `None`.

`core: str | Sequence[str] | None = None`  
How to run CORE. This can be the path or name of the CORE executable, or a command as a list (such as `["python", "core.py"]`). With the default of `None`, the command is taken from the `POINTBLANK_CDISC_CORE` environment variable, or else CORE is looked for on the `PATH`.

`core_cwd: str | Path | None = None`  
The directory to run CORE from. This is needed when CORE is run from a checkout of its repository, since it looks for its `resources/` folder in the current directory. The default is `None`.

`cache: str | Path | None = None`  
The path to CORE's rules cache (its `-ca` option). The default is `None`.

`workdir: str | Path | None = None`  
A directory for the XPT files that the datasets are written to and for CORE's report. With the default of `None`, a temporary directory is used and then deleted.

## Returns

`ConformanceReport`  
A report of CORE's results, whose [`is_core`](../reference/ConformanceReport.md#pointblank.ConformanceReport.is_core) is `True`.

## Raises

`NotADirectoryError`  
If `source` is a path that isn't a folder.

`TypeError`  
If `source` isn't a path, a dict or a [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage).

## Examples

CORE has to be installed for this function, so these examples aren't run here. Let's check a folder of SDTM datasets against SDTMIG 3.4, for a submission to the FDA:

``` python
import pointblank as pb

report = pb.validate_cdisc_submission(
    "study_xyz/sdtm/",
    standard="sdtmig",
    version="3.4",
    controlled_terminology="sdtmct-2024-03-29",
    agency="FDA",
)

print(report)
```

Printing the report gives the standard, the version of CORE, the number of rules with each of CORE's statuses and the total number of issues. The issues and the failing records can then be looked at, and the report saved for review:

``` python
for issue in report.issues(status="ISSUE REPORTED"):
    print(issue["dataset"], issue["rule_id"], issue["issues"], issue["message"])

report.to_excel("study_xyz/conformance/core_report.xlsx")
```

If CORE was installed from a checkout of its repository, it has to be run from that directory, as in `core=["python", "core.py"], core_cwd="path/to/cdisc-rules-engine"`. A report that CORE wrote on its own can be read with [`ConformanceReport.from_core_report()`](../reference/ConformanceReport.md#pointblank.ConformanceReport.from_core_report), which has an example that runs without CORE.

## See Also

[validate_sdtmig()](../reference/validate_sdtmig.md#pointblank.validate_sdtmig)  
Check SDTM datasets with Pointblank's bundled rules.

[SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage)  
A set of datasets to check together.

[ConformanceReport](../reference/ConformanceReport.md#pointblank.ConformanceReport)  
The report that is returned.
