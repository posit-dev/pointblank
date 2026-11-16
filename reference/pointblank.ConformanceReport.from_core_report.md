# from_core_report()

Make a report from the JSON report of a CDISC CORE run.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L1440-L1543)

``` python
from_core_report(
    report,
    package=None,
    agency=None,
)
```

CDISC CORE can be run on its own (as `core validate ... -of JSON`), and its JSON report can be kept for later. The [from_core_report()](../reference/pointblank.ConformanceReport.from_core_report.md) class method reads such a report into a [ConformanceReport](../reference/ConformanceReport.md#pointblank.ConformanceReport), which gives it the same methods as the report of [`validate_cdisc_submission()`](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission). The JSON report has five sections: `Conformance_Details` (the standard, its version and the version of CORE), `Dataset_Details`, `Issue_Summary` (the number of issues for each rule and dataset), `Issue_Details` (one entry for each failing record) and `Rules_Report` (the status of each rule). A file written by [`to_json()`](../reference/pointblank.ConformanceReport.to_json.md) from a CORE report has the same sections, so it can be read back with this method.

## Parameters

`report: dict | ParsedCoreReport`  
The CORE report, as a dict (such as from `json.load()`) or as the already parsed report of an earlier [ConformanceReport](../reference/ConformanceReport.md#pointblank.ConformanceReport) (its `core` attribute).

`package: SubmissionPackage | None = None`  
The [`SubmissionPackage`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage) that CORE checked. The default is `None`.

`agency: str | None = None`  
The regulatory agency, such as `"FDA"`, which is only used as a label in the report. The default is `None`.

## Returns

`ConformanceReport`  
A report of CORE's results, whose [`is_core`](../reference/pointblank.ConformanceReport.is_core.md) is `True`.

## Examples

Let's write a short report by hand in CORE's format, for three rules on a DM and an AE dataset. One rule found an AE record whose subject isn't in DM, one rule passed and one was skipped:

``` python
import pointblank as pb

core_json = {
    "Conformance_Details": {
        "CORE_Engine_Version": "0.16.0",
        "Standard": "SDTMIG",
        "Version": "V3.4",
    },
    "Dataset_Details": [
        {"filename": "dm.xpt", "label": "Demographics", "length": 3},
        {"filename": "ae.xpt", "label": "Adverse Events", "length": 2},
    ],
    "Issue_Summary": [
        {
            "dataset": "ae.xpt",
            "core_id": "CORE-000201",
            "message": "USUBJID is not found in DM.USUBJID",
            "issues": 1,
        }
    ],
    "Issue_Details": [
        {
            "core_id": "CORE-000201",
            "message": "USUBJID is not found in DM.USUBJID",
            "dataset": "ae.xpt",
            "USUBJID": "STUDY01-009",
            "row": 2,
            "SEQ": 1,
            "variables": ["USUBJID"],
            "values": ["STUDY01-009"],
        }
    ],
    "Rules_Report": [
        {"core_id": "CORE-000201", "status": "ISSUE REPORTED"},
        {"core_id": "CORE-000351", "status": "SUCCESS"},
        {"core_id": "CORE-000087", "status": "SKIPPED"},
    ],
}

report = pb.ConformanceReport.from_core_report(core_json, agency="FDA")

print(report)
```

    ConformanceReport (CORE)
      Agency: FDA
      SDTMIG V3.4 -- CORE 0.16.0
      3 rules (ISSUE REPORTED=1, SKIPPED=1, SUCCESS=1)
      1 issues -- FAIL

The rule statuses are CORE's own. A real report would usually be read from a file, with `json.load()`. The failing record is in [`findings()`](../reference/pointblank.ConformanceReport.findings.md):

``` python
report.findings()
```

    [CoreFinding(rule_id='CORE-000201', message='USUBJID is not found in DM.USUBJID', dataset='ae.xpt', executability=None, usubjid='STUDY01-009', row=2, seq=1, variables=['USUBJID'], values=['STUDY01-009'])]

CORE gives the row of a record as a 1-based number, so `row=2` is the second record of the AE dataset.
