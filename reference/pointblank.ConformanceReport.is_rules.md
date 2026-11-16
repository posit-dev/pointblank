# is_rules

Whether the report holds the results of Pointblank's built-in rules.

`is_rules: bool`

This is `True` for the report of [`validate_sdtmig()`](../reference/validate_sdtmig.md#pointblank.validate_sdtmig) and of [SubmissionPackage.validate_conformance()](../reference/pointblank.SubmissionPackage.validate_conformance.md) with its default engine (when the package's standard has a bundled rules catalog). Only these reports have [`findings_df()`](../reference/pointblank.ConformanceReport.findings_df.md), [`get_findings_table()`](../reference/pointblank.ConformanceReport.get_findings_table.md) and [`get_tabular_report()`](../reference/pointblank.ConformanceReport.get_tabular_report.md). This is a property, so it's used without parentheses.

## Examples

Let's check a package with each of the two Pointblank engines:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"STUDYID": ["STUDY01"], "USUBJID": ["STUDY01-001"], "SEX": ["F"]})
study = pb.SubmissionPackage(datasets={"DM": dm})

print(study.validate_conformance().is_rules)
print(study.validate_conformance(engine="validate").is_rules)
```

    True
    False

The `"validate"` engine gives a report of [`Validate`](../reference/Validate.md#pointblank.Validate) validations, so its [is_rules](../reference/pointblank.ConformanceReport.is_rules.md) is `False`, as is its [`is_core`](../reference/pointblank.ConformanceReport.is_core.md).
