# to_json()

Save the report as a JSON file.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L2103-L2211)

``` python
to_json(path)
```

A JSON file keeps the results of a conformance check for later, such as for comparing the results of two runs or for passing them to another tool. For a report of the built-in rules or a Validate-based report, the file has two entries: `summary`, as from [`summary()`](../reference/pointblank.ConformanceReport.summary.md), and [issues](../reference/pointblank.ConformanceReport.issues.md), as from [`issues()`](../reference/pointblank.ConformanceReport.issues.md). For a CDISC CORE report, the file has the five sections of CORE's own JSON report, so it can be read by tools that read CORE reports, and back into a report with [`from_core_report()`](../reference/pointblank.ConformanceReport.from_core_report.md).

## Parameters

`path: str | Path`  
The path of the file to write, including its name. Directories in the path that don't exist are created.

## Returns

`Path`  
The path of the file that was written.

## Examples

Let's check a small DM dataset with the built-in rules and save the report in a temporary directory:

``` python
import json
import tempfile
from pathlib import Path

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

path = report.to_json(Path(tempfile.mkdtemp()) / "conformance" / "dm.json")

saved = json.loads(path.read_text())

print(saved["summary"]["status_counts"])
print(saved["issues"][0])
```

    {'pass': 342, 'fail': 8, 'not_applicable': 12}
    {'dataset': 'DM', 'rule_id': 'SDTM-007', 'rule_type': 'RECORD_CHECK', 'message': 'SEX value is not in the SEX codelist.', 'n_issues': 1, 'sensitivity': 'Error', 'status': 'fail'}

The `conformance` directory didn't exist, so it was created. Each issue is saved with all its keys, as [issues()](../reference/pointblank.ConformanceReport.issues.md) gives it.
