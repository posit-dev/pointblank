# get_findings_table()

Make a table of the failing records found by the built-in rules.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L2490-L2660)

``` python
get_findings_table()
```

The table of [`get_tabular_report()`](../reference/pointblank.ConformanceReport.get_tabular_report.md) has a row for each rule, with its number of issues. The findings table goes into more detail, with a row for each failing record, so that a reviewer can see which subjects and values have to be fixed. It has the same findings as [`findings_df()`](../reference/pointblank.ConformanceReport.findings_df.md), with at most 100 for each rule.

The columns of the table are in two groups. Under *SDTM Rule Definition* are the rule's identifier, its domain and its description. Under *Finding* are the record's `USUBJID`, the variable that was checked, the record's row number (counting from 1) and the value that failed. The subtitle has the standard and its version and the numbers of rules that failed, passed and were not applicable.

## Returns

`GT`  
The table, as a [Great Tables](https://posit-dev.github.io/great-tables/) `GT` object.

## Raises

`TypeError`  
If the report isn't of the built-in rules.

`ValueError`  
If there are no failing records. This happens when no rule that checks values failed, even if other rules (such as those for missing variables) did.

## Examples

Let's check a small DM dataset with the built-in rules and make a table of its failing records:

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

report.get_findings_table()
```

    /opt/hostedtoolcache/Python/3.11.18/x64/lib/python3.11/site-packages/great_tables/_render_checks.py:37: RenderWarning: Rendering table with .cols_width() in Quarto may result in unexpected behavior. This is because Quarto performs custom table processing. Either use all percentage widths, or set .tab_options(quarto_disable_processing=True) to disable Quarto table processing.
      warnings.warn(

| Findings Report for CDISC Conformance |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| SDTMIG 3-4 · 10 failed (340 passed) · 12 n/a |  |  |  |  |  |  |  |
|  | SDTM Rule Definition |  |  | Finding |  |  |  |
|  | Rule | Domain | Description | USUBJID | Column | Row | Value |
| \#FF3300 | SDTM-007 | DM | SEX in DM must use values from the CDISC controlled terminology codelist SEX. | STUDY01-003 | SEX | 3 | FEMALE |
| \#FF3300 | SDTM-011 | DM | RFSTDTC in DM must be in ISO 8601 extended datetime format when present. | STUDY01-002 | RFSTDTC | 2 | 15JAN2024 |
| \#FF3300 | SDTM-195 | DM | AGE must not be negative in the DM dataset. | STUDY01-003 | AGE | 3 | -38 |

The third subject's record has two problems, a negative `AGE` and a `SEX` of `"FEMALE"`, so it's in the table twice. The second subject's `RFSTDTC` isn't in the ISO 8601 format.
