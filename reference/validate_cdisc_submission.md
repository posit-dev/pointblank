## validate_cdisc_submission()


Validate a CDISC submission with the CDISC CORE engine, in one call.


Usage

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
    workdir=None
)
```


Convenience wrapper that builds a <a href="SubmissionPackage.html#pointblank.SubmissionPackage" class="gdls-link"><code>SubmissionPackage</code></a> from `source` and runs <a href="SubmissionPackage.html#pointblank.SubmissionPackage" class="gdls-link"><code>validate_conformance()</code></a> with `engine="core"`. Requires an installed CORE executable (see [core](SDTMVariableSpec.md#pointblank.SDTMVariableSpec.core)).


## Parameters


`source: str | Path | dict | SubmissionPackage`  
The submission to validate. One of: a path to a folder of datasets (XPT / Dataset-JSON, with an optional `define.xml`), a mapping of dataset name to DataFrame, or an already-built [SubmissionPackage](SubmissionPackage.md#pointblank.SubmissionPackage).

`standard: str | None = None`  
The CDISC standard (e.g., `"sdtmig"`). Defaults to `"sdtmig"` (or the package's [standard](SubmissionPackage.md#pointblank.SubmissionPackage.standard) when `source` is a [SubmissionPackage](SubmissionPackage.md#pointblank.SubmissionPackage)).

`version: str | None = None`  
The standard version (e.g., `"3.4"`). Defaults to `"3.4"` (or the package's value).

`define: str | Path | Any | None = None`  
Optional Define-XML path (ignored when `source` is a [SubmissionPackage](SubmissionPackage.md#pointblank.SubmissionPackage) -- set it on the package instead). Auto-detected from a folder `source` when present.

`controlled_terminology: str | Sequence[str] | None = None`  
CT package name(s) for CORE's `-ct` (e.g., `"sdtmct-2024-03-29"`).

`agency: str | None = None`  
Optional agency rule-set selector recorded on the report.

`ct_version: str | None = None`  
Optional Controlled Terminology version pin recorded on the package.

`study_id: str | None = None`  
Optional study identifier.

`core: str | Sequence[str] | None = None`  
How to invoke CORE -- a path/name to the CORE executable, a full command prefix (e.g., `["python", "core.py"]`), or `None` to auto-discover via the `POINTBLANK_CDISC_CORE` environment variable and then `PATH`.

`core_cwd: str | Path | None = None`  
Working directory to run CORE from; required when invoking a repo checkout.

`cache: str | Path | None = None`  
Path to CORE's rules cache directory (`-ca`).

`workdir: str | Path | None = None`  
Directory for materialized XPT and the CORE report. If `None`, a temporary directory is used.


## Returns


`ConformanceReport`  
A CORE-form report ([is_core](ConformanceReport.md#pointblank.ConformanceReport.is_core) is `True`).


## Examples

``` python
import pointblank as pb

report = pb.validate_cdisc_submission(
    "study_xyz/sdtm/",
    standard="sdtmig",
    version="3.4",
    agency="FDA",
)
report.summary()
```
