# to_excel()

Save the report as an Excel workbook.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L2211-L2382)

``` python
to_excel(path)
```

A workbook is a convenient way to share the results of a conformance check with people who review them in a spreadsheet. The sheets depend on the form of the report:

- For the built-in rules, `Rules_Report` has a row for each rule, with its type, dataset, status, sensitivity, number of issues and message. `Issues` has the rules with issues (as from [`issues()`](../reference/pointblank.ConformanceReport.issues.md)), and `Summary` has the entries of [`summary()`](../reference/pointblank.ConformanceReport.summary.md).
- For CDISC CORE, the sheets are `Issue_Summary`, `Issue_Details`, `Rules_Report` and `Conformance_Details`, like those of CORE's own Excel report.
- For a Validate-based report, `Issues` has the failing steps and `Summary` has a row for each dataset.

A sheet that would be empty, such as `Issues` when there are no issues, is left out. The `openpyxl` and `pandas` packages are needed for writing the workbook.

## Parameters

`path: str | Path`  
The path of the file to write, including its name (which usually ends in `.xlsx`). Directories in the path that don't exist are created.

## Returns

`Path`  
The path of the file that was written.

## Raises

`ImportError`  
If `openpyxl` or `pandas` isn't installed.

## Examples

Let's check a small DM dataset with the built-in rules and save the report in a temporary directory:

``` python
import tempfile
from pathlib import Path

import pandas as pd
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

path = report.to_excel(Path(tempfile.mkdtemp()) / "dm_conformance.xlsx")

pd.ExcelFile(path).sheet_names
```

    ['Rules_Report', 'Issues', 'Summary']

Here's the start of the `Rules_Report` sheet, whose rows are in the order of the rules catalog:

``` python
pd.read_excel(path, sheet_name="Rules_Report").head(3)
```

|  | Rule ID | Rule Type | Dataset | Status | Sensitivity | Issues | Message |
|----|----|----|----|----|----|----|----|
| 0 | SDTM-001 | DOMAIN_PRESENCE_CHECK | DM | pass | Error | 0 | DM (Demographics) domain is required in every ... |
| 1 | SDTM-002 | RECORD_CHECK | DM | pass | Error | 0 | STUDYID must not be null in any domain. |
| 2 | SDTM-003 | RECORD_CHECK | DM | pass | Error | 0 | DOMAIN must not be null in any SDTM dataset. |

The workbook has a row for each of the 362 rules, so the passing rules are recorded as well as the failing ones.
