# summary()

Summarize the results of the conformance check as a dict.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L1723-L1814)

``` python
summary()
```

The summary has the counts that are shown when a report is printed, in a form that code can use, such as for logging the results of a run or for comparing two runs. Its keys depend on the form of the report.

## Returns

`dict`  
For a report of the built-in rules, a dict with the keys `standard`, `version`, `engine` (which is `"built-in"`), `ct_packages` (the controlled terminology packages that were used), `n_rules`, [status_counts](../reference/Comparison.status_counts.md#pointblank.Comparison.status_counts) (the number of rules with each status), `n_issues` (the total number of issues) and `all_passed`.

For a CDISC CORE report, a dict with the keys `standard`, `version`, `engine_version` (the version of CORE), `engine` (which is `"core"`), `n_rules`, [status_counts](../reference/Comparison.status_counts.md#pointblank.Comparison.status_counts), `n_issues`, [n_datasets](../reference/pointblank.ConformanceReport.n_datasets.md) and `all_passed`.

For a Validate-based report, a dict with an entry for each dataset. Each entry is a dict with the keys `n_steps`, `n_steps_failed`, `n_failed` (the number of failing test units) and `all_passed`.

## Examples

Let's check a small DM dataset with the built-in rules and summarize the results:

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

report.summary()
```

    {'standard': 'sdtmig',
     'version': '3-4',
     'engine': 'built-in',
     'ct_packages': ['sdtm-ct-2024-09-27'],
     'n_rules': 362,
     'status_counts': {'pass': 340, 'fail': 10, 'not_applicable': 12},
     'n_issues': 10,
     'all_passed': False}

The version is given as `"3-4"`, the form that the rules catalog uses. The issue count is the sum of the issues of all the rules: each failing record counts for a rule that checks values, and a failing rule counts once for a rule that checks the presence of variables or datasets.
