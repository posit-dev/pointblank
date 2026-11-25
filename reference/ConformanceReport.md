# ConformanceReport

The results of a CDISC conformance check.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L1337-L2970)

``` python
ConformanceReport(
    validations=dict(),
    package=None,
    agency=None,
    core=None,
    native_result=None,
)
```

A [ConformanceReport](../reference/ConformanceReport.md#pointblank.ConformanceReport) holds the results of checking a set of CDISC datasets for conformance to their standard. It is returned by [`validate_sdtmig()`](../reference/validate_sdtmig.md#pointblank.validate_sdtmig), [`validate_cdisc_submission()`](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission) and the [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage) method [`validate_conformance()`](../reference/pointblank.SubmissionPackage.validate_conformance.md), and is rarely made directly. The report has one of three forms, depending on the engine that did the checks:

- **Built-in rules** ([`is_rules`](../reference/pointblank.ConformanceReport.is_rules.md) is `True`): the results of Pointblank's bundled catalog of SDTMIG rules. Each rule has a status of `"pass"`, `"fail"`, `"error"` (the rule couldn't be evaluated), `"not_applicable"` (an input that it needs, such as a Define-XML, wasn't given) or `"not_supported"`. The rules that check values give the failing records, which are in [`findings_df()`](../reference/pointblank.ConformanceReport.findings_df.md) and [`get_findings_table()`](../reference/pointblank.ConformanceReport.get_findings_table.md).
- **CDISC CORE** ([`is_core`](../reference/pointblank.ConformanceReport.is_core.md) is `True`): the results of the CDISC CORE engine, with CORE's own rule statuses (such as `"SUCCESS"`, `"ISSUE REPORTED"` and `"SKIPPED"`). A report of this form can also be made from a saved CORE report with [`from_core_report()`](../reference/pointblank.ConformanceReport.from_core_report.md).
- **Validate-based**: a [`Validate`](../reference/Validate.md#pointblank.Validate) validation for each dataset, from `SubmissionPackage.validate_conformance(engine="validate")`. Each validation can be had with [`get_validation()`](../reference/pointblank.ConformanceReport.get_validation.md).

The methods [`summary()`](../reference/pointblank.ConformanceReport.summary.md), [`issues()`](../reference/pointblank.ConformanceReport.issues.md) and [`all_passed()`](../reference/pointblank.ConformanceReport.all_passed.md) work for all three forms. In a notebook, a report of the built-in rules is shown as the table of [`get_tabular_report()`](../reference/pointblank.ConformanceReport.get_tabular_report.md). A CORE report is shown as a short summary with a table of its issues, and a Validate-based report as the validation report of each dataset. Printing a report of any form gives a text summary.

## Parameters

`validations: dict[str, Validate] = dict()`  
The validation for each dataset, keyed by dataset name, for a Validate-based report. The default is an empty dict.

`package: SubmissionPackage | None = None`  
The [`SubmissionPackage`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage) that was checked. The default is `None`.

`agency: str | None = None`  
The regulatory agency, such as `"FDA"`, which is only used as a label in the report. The default is `None`.

`core: ParsedCoreReport | None = None`  
CORE's results, for a CDISC CORE report. The default is `None`.

`native_result: NativeConformanceResult | None = None`  
The results of the rules, for a report of the built-in rules. The default is `None`.

## Examples

Let's check a small DM dataset with the built-in rules. It has a date that isn't in the ISO 8601 format, a negative age and a `SEX` value that isn't in its codelist:

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

print(report)
```

    ConformanceReport (Built-in Rules)
      sdtmig 3-4
      362 rules (fail=10, not_applicable=12, pass=340)
      10 issues -- FAIL

Ten of the 362 rules fail. Most of them are about what the dataset lacks, such as the DM variables `SUBJID` and `RACE` and the TS and TA datasets. The failing records are found by three of the rules:

``` python
report.findings_df().select("rule_id", "row_index", "checked_column", "checked_value")
```

shape: (3, 4)

| rule_id    | row_index | checked_column | checked_value |
|------------|-----------|----------------|---------------|
| str        | i64       | str            | str           |
| "SDTM-007" | 2         | "SEX"          | "FEMALE"      |
| "SDTM-011" | 1         | "RFSTDTC"      | "15JAN2024"   |
| "SDTM-195" | 2         | "AGE"          | "-38"         |

The `row_index` of a finding is the 0-based position of the record in its dataset. To see the results of all the rules as a table, use [`get_tabular_report()`](../reference/pointblank.ConformanceReport.get_tabular_report.md).

## See Also

[validate_sdtmig()](../reference/validate_sdtmig.md#pointblank.validate_sdtmig)  
Check SDTM datasets with the built-in rules.

[validate_cdisc_submission()](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission)  
Check a folder of datasets with CDISC CORE.

[SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage)  
A set of datasets to check together.

## Attributes

| Name | Description |
|----|----|
| [n_datasets](pointblank.ConformanceReport.n_datasets.md) | The number of datasets that were checked. |
| [is_core](pointblank.ConformanceReport.is_core.md) | Whether the report holds the results of CDISC CORE. |
| [is_rules](pointblank.ConformanceReport.is_rules.md) | Whether the report holds the results of Pointblank's built-in rules. |

## Functions

| Name | Description |
|----|----|
| [all_passed()](pointblank.ConformanceReport.all_passed.md) | Check whether the datasets conform, with nothing failing. |
| [summary()](pointblank.ConformanceReport.summary.md) | Summarize the results of the conformance check as a dict. |
| [get_tabular_report()](pointblank.ConformanceReport.get_tabular_report.md) | Make a table of the results of the built-in rules. |
| [get_findings_table()](pointblank.ConformanceReport.get_findings_table.md) | Make a table of the failing records found by the built-in rules. |
| [rules()](pointblank.ConformanceReport.rules.md) | Get the result of each rule. |
| [issues()](pointblank.ConformanceReport.issues.md) | List the issues that the conformance check found. |
| [findings()](pointblank.ConformanceReport.findings.md) | Get the failing records that the conformance check found. |
| [findings_df()](pointblank.ConformanceReport.findings_df.md) | Get the failing records found by the built-in rules, as a DataFrame. |
| [get_validation()](pointblank.ConformanceReport.get_validation.md) | Get the validation of one dataset from a Validate-based report. |
| [to_json()](pointblank.ConformanceReport.to_json.md) | Save the report as a JSON file. |
| [to_excel()](pointblank.ConformanceReport.to_excel.md) | Save the report as an Excel workbook. |
| [from_core_report()](pointblank.ConformanceReport.from_core_report.md) | Make a report from the JSON report of a CDISC CORE run. |
