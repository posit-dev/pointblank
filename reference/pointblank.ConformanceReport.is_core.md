# is_core

Whether the report holds the results of CDISC CORE.

`is_core: bool`

This is `True` for the report of [`validate_cdisc_submission()`](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission), of `SubmissionPackage.validate_conformance(engine="core")` and of [`from_core_report()`](../reference/pointblank.ConformanceReport.from_core_report.md). Some methods only work for one form of report, so this tells code which form it has. This is a property, so it's used without parentheses.

## Examples

Let's check a DM dataset with the built-in rules and look at the form of the report:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"STUDYID": ["STUDY01"], "USUBJID": ["STUDY01-001"], "SEX": ["F"]})

report = pb.validate_sdtmig({"DM": dm})

print(report.is_core, report.is_rules)
```

    False True

The report is of the built-in rules, so [is_core](../reference/pointblank.ConformanceReport.is_core.md) is `False` and [`is_rules`](../reference/pointblank.ConformanceReport.is_rules.md) is `True`.
