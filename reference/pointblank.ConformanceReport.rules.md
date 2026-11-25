# rules()

Get the result of each rule.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L2002-L2066)

``` python
rules(status=None)
```

Every rule that was run has a result, whether it passed or not. The [rules()](../reference/pointblank.ConformanceReport.rules.md) method gets these results, optionally only those with a particular status. This is useful for listing the failing rules, or for finding out why some rules were not applicable.

## Parameters

`status: str | None = None`  
Only keep the rules with this status. For the built-in rules, this can be `"pass"`, `"fail"`, `"error"`, `"not_applicable"` or `"not_supported"`. For CDISC CORE, it is one of CORE's statuses, such as `"SUCCESS"`, `"ISSUE REPORTED"`, `"SKIPPED"` or `"EXECUTION ERROR"`. The default is `None`, which keeps all the rules.

## Returns

`list`  
For a report of the built-in rules, a `NativeRuleResult` for each rule, with the attributes `rule_id`, `rule_type`, `dataset`, `status`, `sensitivity`, `description`, `message`, `n_issues` and `row_findings` (its findings, as returned by [`findings()`](../reference/pointblank.ConformanceReport.findings.md)).

For a CDISC CORE report, a `CoreRuleResult` for each rule, with the attributes `rule_id`, `status`, `message`, `version`, `cdisc_rule_id` and `fda_rule_id`.

For a Validate-based report, an empty list.

## Examples

Let's check a small DM dataset with the built-in rules and look at the rules that are not applicable:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame(
    {
        "STUDYID": ["STUDY01"] * 3,
        "DOMAIN": ["DM"] * 3,
        "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
        "SEX": ["M", "F", "FEMALE"],
    }
)

report = pb.validate_sdtmig({"DM": dm})

not_applicable = report.rules(status="not_applicable")

print(len(not_applicable))
print({rule.message for rule in not_applicable})
```

    12
    {'Required input(s) not provided: DEFINE'}

The 12 rules all check the datasets against a Define-XML, which wasn't given. With the `define_xml=` argument of [`validate_sdtmig()`](../reference/validate_sdtmig.md#pointblank.validate_sdtmig), they would be run too.
