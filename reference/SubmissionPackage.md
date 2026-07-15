## SubmissionPackage


A data-level model of a study submission package for CDISC conformance validation.


Usage

``` python
SubmissionPackage(
    datasets=dict(),
    define=None,
    ct_version=None,
    standard="sdtmig",
    standard_version="3.4",
    study_id=None
)
```


A [SubmissionPackage](SubmissionPackage.md#pointblank.SubmissionPackage) groups the datasets of a study (SDTM domains, SUPP- qualifiers, RELREC, and/or ADaM datasets) together with their Define-XML and Controlled Terminology context, and understands the *relationships* between them. This enables cross-dataset conformance checks -- referential integrity, SUPP- linkage, RELREC resolution, and ADaM ⇄ SDTM traceability -- that single-dataset validation cannot express.

This is the data-level analog of <a href="MetadataPackage.html#pointblank.MetadataPackage" class="gdls-link"><code>MetadataPackage</code></a>, which groups *metadata* for many datasets.


## Parameters


`datasets: dict[str, Any] = dict()`    
A mapping of dataset name (domain code, e.g., `"DM"`, `"AE"`, `"SUPPAE"`, `"ADSL"`) to the dataset itself (a Pandas or Polars DataFrame). Names are matched case-insensitively but conventionally uppercase.

`define: Any = None`  
Optional Define-XML context: a path to a `define.xml` file, or an already-imported <a href="MetadataPackage.html#pointblank.MetadataPackage" class="gdls-link"><code>MetadataPackage</code></a>. Used to supply variable definitions, codelists, and origins for define-context rules.

`ct_version: str | None = None`  
Optional Controlled Terminology version pin (e.g., `"2024-03-29"`), recorded for reproducible runs.

`standard: str = ``"sdtmig"`  
The data standard the package follows (`"sdtmig"` or `"adamig"`). Defaults to `"sdtmig"`.

`standard_version: str = ``"3.4"`  
The Implementation Guide version (e.g., `"3.4"` for SDTM IG). Defaults to `"3.4"`.

`study_id: str | None = None`  
Optional study identifier, used in report labels.


## Examples

Construct a package from in-memory DataFrames and validate conformance across it:

``` python
import pointblank as pb

study = pb.SubmissionPackage(
    datasets={"DM": dm_df, "AE": ae_df, "LB": lb_df},
    standard="sdtmig",
    standard_version="3.4",
)

report = study.validate_conformance()
report.summary()
```

Or ingest a folder of XPT files (Define-XML auto-detected if present):

``` python
study = pb.SubmissionPackage.from_folder("study_xyz/sdtm/")
report = study.validate_conformance(agency="FDA")
```


## Attributes

| Name | Description |
|----|----|
| [domains](#domains) | The names (domain codes) of all datasets in the package, sorted. |
| [metadata](#metadata) | The imported Define-XML metadata, if `define` was supplied. |

------------------------------------------------------------------------


#### domains


The names (domain codes) of all datasets in the package, sorted.


`domains: list[str]`


------------------------------------------------------------------------


#### metadata


The imported Define-XML metadata, if `define` was supplied.


`metadata: MetadataPackage | None`


Lazily imports the Define-XML document (via <a href="import_metadata.html#pointblank.import_metadata" class="gdls-link"><code>import_metadata()</code></a>) the first time it is accessed.


## Methods

| Name | Description |
|----|----|
| [from_folder()](#from_folder) | Build a [SubmissionPackage](SubmissionPackage.md#pointblank.SubmissionPackage) by ingesting a folder of datasets. |
| [get_dataset()](#get_dataset) | Get a dataset by name (case-insensitive). |
| [orphan_ids()](#orphan_ids) | Find values of `column` in `child` that do not exist in `parent`. |
| [subject_ids()](#subject_ids) | Get the set of `USUBJID` values in a dataset. |
| [summary()](#summary) | Return a human-readable summary of the package contents. |
| [validate_conformance()](#validate_conformance) | Validate CDISC conformance across the whole submission package. |

------------------------------------------------------------------------


#### from_folder()


Build a [SubmissionPackage](SubmissionPackage.md#pointblank.SubmissionPackage) by ingesting a folder of datasets.


Usage

``` python
from_folder(
    path,
    define=None,
    standard="sdtmig",
    standard_version="3.4",
    ct_version=None,
    study_id=None
)
```


Reads every SAS Transport (`.xpt`) and CDISC Dataset-JSON (`.json`) file in the folder, deriving the dataset name from the file stem (uppercased). If a `define.xml` is present in the folder and `define` is not supplied, it is picked up automatically.


##### Parameters


`path: str | Path`  
Path to a folder containing the study datasets.

`define: str | Path | Any | None = None`  
Optional Define-XML path or <a href="MetadataPackage.html#pointblank.MetadataPackage" class="gdls-link"><code>MetadataPackage</code></a>. If `None`, a `define.xml` in the folder is used when present.

`standard: str = ``"sdtmig"`  
The data standard (`"sdtmig"` or `"adamig"`). Defaults to `"sdtmig"`.

`standard_version: str = ``"3.4"`  
The Implementation Guide version. Defaults to `"3.4"`.

`ct_version: str | None = None`  
Optional Controlled Terminology version pin.

`study_id: str | None = None`  
Optional study identifier.


##### Returns


`SubmissionPackage`  
A package populated with the folder's datasets.


------------------------------------------------------------------------


#### get_dataset()


Get a dataset by name (case-insensitive).


Usage

``` python
get_dataset(name)
```


##### Parameters


`name: str`  
The dataset name / domain code.


##### Returns


`Any`  
The dataset (DataFrame).


##### Raises


`KeyError`  
If no dataset with that name exists.


------------------------------------------------------------------------


#### orphan_ids()


Find values of `column` in `child` that do not exist in `parent`.


Usage

``` python
orphan_ids(child, parent="DM", column="USUBJID")
```


This is the referential-integrity operator: e.g., subjects appearing in a finding domain that have no corresponding record in DM.


##### Parameters


`child: str`  
The referencing dataset (e.g., `"AE"`).

`parent: str = ``"DM"`  
The referenced dataset (e.g., `"DM"`). Defaults to `"DM"`.

`column: str = ``"USUBJID"`  
The key column to check. Defaults to `"USUBJID"`.


##### Returns


`set`  
The set of orphaned values (present in `child.column` but not `parent.column`).


------------------------------------------------------------------------


#### subject_ids()


Get the set of `USUBJID` values in a dataset.


Usage

``` python
subject_ids(dataset="DM")
```


##### Parameters


`dataset: str = ``"DM"`  
The dataset to read subject IDs from. Defaults to `"DM"` (the reference set of all enrolled subjects).


##### Returns


`set`  
The set of non-null `USUBJID` values, or an empty set if the dataset or column is absent.


------------------------------------------------------------------------


#### summary()


Return a human-readable summary of the package contents.


Usage

``` python
summary()
```


------------------------------------------------------------------------


#### validate_conformance()


Validate CDISC conformance across the whole submission package.


Usage

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
    workdir=None
)
```


Two engines are available:

- **`"native"`** (default) -- Pointblank's own checks. For each dataset this builds a <a href="Validate.html#pointblank.Validate" class="gdls-link"><code>Validate</code></a> plan combining the single-dataset structural checks (via <a href="validate_sdtm.html#pointblank.validate_sdtm" class="gdls-link"><code>validate_sdtm()</code></a> / <a href="validate_adam.html#pointblank.validate_adam" class="gdls-link"><code>validate_adam()</code></a>) and cross-dataset conformance checks (when `cross_dataset=True`):
  - **Referential integrity** -- every `USUBJID` in a finding/events/interventions domain exists in DM.
  - **SUPP- linkage** -- `RDOMAIN` references a present domain, `USUBJID` exists in DM, and `(USUBJID, IDVAR=IDVARVAL)` resolves to a record in the parent domain.
  - **RELREC** -- each relationship record's `RDOMAIN` is present and `USUBJID` exists in DM.
  - **ADaM ⇄ SDTM traceability** -- `ADSL.USUBJID ⊆ DM.USUBJID`, and every other ADaM dataset's `USUBJID ⊆ ADSL.USUBJID`.
- **`"core"`** -- hands the package to the external CDISC CORE engine (`cdisc-rules-engine`), which runs the authoritative conformance rule set, and ingests its results. Datasets are materialized to XPT (or the source folder is used directly for folder-ingested packages), CORE is invoked as a subprocess, and its JSON report becomes a CORE-form [ConformanceReport](ConformanceReport.md#pointblank.ConformanceReport). Requires an installed CORE executable (see [core](SDTMVariableSpec.md#pointblank.SDTMVariableSpec.core)).


##### Parameters


`agency: str | None = None`  
Optional agency rule-set selector (`"FDA"`, `"PMDA"`, or `None` for CDISC base rules).

`engine: str = ``"native"`  
`"native"` (the default) or `"core"`.

`cross_dataset: bool = ``True`  
(Validate-based engine only.) Whether to add cross-dataset conformance checks. Defaults to `True`.

`thresholds: Any = None`  
(Validate-based engine only.) Optional thresholds passed to each dataset's [Validate](Validate.md#pointblank.Validate) (maps failing test units onto Pointblank's warning/error/critical severity model).

`interrogate: bool = ``True`  
(Validate-based engine only.) Whether to interrogate (run) the validations before returning.

`standard: str | None = None`  
(CORE only.) Override the CDISC standard sent to CORE. Defaults to the package's [standard](SubmissionPackage.md#pointblank.SubmissionPackage.standard) (e.g., `"sdtmig"`).

`version: str | None = None`  
(CORE only.) Override the standard version. Defaults to the package's [standard_version](SubmissionPackage.md#pointblank.SubmissionPackage.standard_version) (e.g., `"3.4"`, sent to CORE hyphenated).

`controlled_terminology: str | Sequence[str] | None = None`  
(CORE only.) CT package name(s) for CORE's `-ct` (e.g., `"sdtmct-2024-03-29"`).

`core: str | Sequence[str] | None = None`  
(CORE only.) How to invoke CORE -- a path/name to the CORE executable, a full command prefix (e.g., `["python", "core.py"]`), or `None` to auto-discover via the `POINTBLANK_CDISC_CORE` environment variable and then `PATH`.

`core_cwd: str | Path | None = None`  
(CORE only.) Working directory to run CORE from; required when invoking a repo checkout (CORE resolves its bundled `resources/` relative to the current directory).

`cache: str | Path | None = None`  
(CORE only.) Path to CORE's rules cache directory (`-ca`).

`workdir: str | Path | None = None`  
(CORE only.) Directory for materialized XPT and the CORE report. If `None`, a temporary directory is used and cleaned up.


##### Returns


`ConformanceReport`  
A built-in engine report (per-dataset validations) or a CORE-form report, depending on `engine`.
