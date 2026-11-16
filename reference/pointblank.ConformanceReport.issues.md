# issues()

List the issues that the conformance check found.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L1816-L1940)

``` python
issues(
    severity=None,
    status=None,
)
```

An issue is a rule (or, for a Validate-based report, a validation step) that found something wrong. The list is a quick way to see what has to be fixed, and it can be turned into a DataFrame or saved. Each issue is a dict, whose keys depend on the form of the report.

## Parameters

`severity: str | None = None`  
For a Validate-based report, only keep the issues of this severity: `"warning"`, `"error"` or `"critical"`. The severity of a step comes from the `thresholds=` of [`validate_conformance()`](../reference/pointblank.SubmissionPackage.validate_conformance.md), so without thresholds no issue has a severity. This is ignored for the other forms of report. The default is `None`, which keeps all the issues.

`status: str | None = None`  
For a CDISC CORE report, only keep the issues of rules with this status, such as `"ISSUE REPORTED"`. This is ignored for the other forms of report. The default is `None`, which keeps all the issues.

## Returns

`list[dict]`  
For a report of the built-in rules, a dict for each rule with issues, with the keys `dataset`, `rule_id`, `rule_type`, `message`, `n_issues`, `sensitivity` (`"Error"` or `"Warning"`) and `status`.

For a CDISC CORE report, a dict for each rule and dataset with issues, with the keys `dataset`, `rule_id`, `message`, [issues](../reference/pointblank.ConformanceReport.issues.md) (the number of issues) and `status`.

For a Validate-based report, a dict for each step with failing test units, with the keys `dataset`, `step`, `assertion`, `column`, `n_failed`, `n` and `severity`.

## Examples

Let's check a small DM dataset with the built-in rules and list the issues as a DataFrame:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame(
    {
        "STUDYID": ["STUDY01"] * 3,
        "DOMAIN": ["DM"] * 3,
        "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
        "RFSTDTC": ["2024-01-15", "15JAN2024", "2024-01-17"],
        "AGE": [34, 51, -38],
        "SEX": ["M", "F", "FEMALE"],
    }
)

report = pb.validate_sdtmig({"DM": dm})

pl.DataFrame(report.issues()).select("rule_id", "dataset", "sensitivity", "n_issues")
```

shape: (10, 4)

| rule_id    | dataset | sensitivity | n_issues |
|------------|---------|-------------|----------|
| str        | str     | str         | i64      |
| "SDTM-007" | "DM"    | "Error"     | 1        |
| "SDTM-011" | "DM"    | "Error"     | 1        |
| "SDTM-031" | "DM"    | "Error"     | 1        |
| "SDTM-032" | "DM"    | "Error"     | 1        |
| "SDTM-033" | "DM"    | "Error"     | 1        |
| "SDTM-034" | "DM"    | "Error"     | 1        |
| "SDTM-035" | "DM"    | "Error"     | 1        |
| "SDTM-121" | "TS"    | "Error"     | 1        |
| "SDTM-122" | "TA"    | "Warning"   | 1        |
| "SDTM-195" | "DM"    | "Error"     | 1        |

The missing TS and TA datasets are given as the datasets of SDTM-121 and SDTM-122. Only SDTM-122 is of `"Warning"` sensitivity. A Validate-based report gives the severity of each step instead, when thresholds are set:

``` python
ae = pl.DataFrame({"USUBJID": ["STUDY01-001", "STUDY01-009"], "AESEQ": [1, 1]})
study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae})

report = study.validate_conformance(engine="validate", thresholds=pb.Thresholds(warning=1))

report.issues(severity="warning")
```

    [{'dataset': 'AE',
      'step': 5,
      'assertion': 'specially',
      'column': None,
      'n_failed': 1,
      'n': 2,
      'severity': 'warning'},
     {'dataset': 'DM',
      'step': 10,
      'assertion': 'col_vals_expr',
      'column': None,
      'n_failed': 1,
      'n': 3,
      'severity': 'warning'},
     {'dataset': 'DM',
      'step': 11,
      'assertion': 'col_vals_regex',
      'column': 'RFSTDTC',
      'n_failed': 1,
      'n': 3,
      'severity': 'warning'}]

The issues are the orphan subject in AE, and the length of `"FEMALE"` and the date `"15JAN2024"` in DM. Each of them is one failing test unit, which reaches the warning threshold.
