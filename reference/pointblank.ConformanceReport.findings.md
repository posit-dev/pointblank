# findings()

Get the failing records that the conformance check found.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L1942-L1998)

``` python
findings()
```

A rule that checks the values of a dataset (such as a codelist check on `SEX`) fails for particular records, and each of these is a finding. The [findings()](../reference/pointblank.ConformanceReport.findings.md) method gets them all as a list of objects. For the built-in rules, [`findings_df()`](../reference/pointblank.ConformanceReport.findings_df.md) gives the same findings as a DataFrame, which is often easier to work with.

## Returns

`list`  
For a report of the built-in rules, a `NativeRowFinding` for each failing record, with the attributes `rule_id`, `dataset`, `row` (the 0-based position of the record), `usubjid`, `checked_column`, `checked_value`, `context` (the record's identifying variables, such as `STUDYID`) and `message`. At most 100 findings are kept for each rule.

For a CDISC CORE report, a `CoreFinding` for each entry of CORE's `Issue_Details`, with the attributes `rule_id`, `message`, `dataset`, `executability`, `usubjid`, `row` (the 1-based position of the record), `seq`, `variables` and [values](../reference/MetadataPackage.md#pointblank.MetadataPackage.values).

For a Validate-based report, an empty list.

## Examples

Let's check a small DM dataset with the built-in rules:

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

for finding in report.findings():
    print(finding.rule_id, finding.row, finding.usubjid, finding.message)
```

    SDTM-007 2 STUDY01-003 SEX value is not in the SEX codelist.
    SDTM-011 1 STUDY01-002 RFSTDTC does not conform to ISO 8601 extended datetime format.
    SDTM-195 2 STUDY01-003 AGE must not be negative in DM.

Ten rules fail, but only these three check values. The others fail because a variable or a dataset is missing, so they have no failing records.
