# findings_df()

Get the failing records found by the built-in rules, as a DataFrame.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L2384-L2486)

``` python
findings_df()
```

Each row of the DataFrame is a record that failed a rule. As a DataFrame, the findings are easy to filter by rule, count by subject, save as a CSV file, or join back to the datasets to look at the records in full. The rules that check values (record checks, dataset contents checks and checks against the Define-XML's codelists) give findings. Rules that check whether variables or datasets are present have a number of issues but no findings. At most 100 findings are kept for each rule, though the rule's number of issues (as in [`issues()`](../reference/pointblank.ConformanceReport.issues.md)) counts all of its failing records.

## Returns

`polars.DataFrame`  
A DataFrame with a row for each finding and these columns:

- `rule_id`: the identifier of the rule, such as `"SDTM-007"`
- `dataset`: the dataset of the record, such as `"DM"`
- `row_index`: the 0-based position of the record in its dataset
- `usubjid`: the record's `USUBJID`, or `""` if it has none
- `checked_column`: the variable that was checked, such as `"SEX"`
- `checked_value`: the value that failed, as a string
- `description`: the description of the rule

If there are no findings, the DataFrame is empty but has the same columns.

## Raises

`TypeError`  
If the report isn't of the built-in rules. For a CDISC CORE report, use [`findings()`](../reference/pointblank.ConformanceReport.findings.md) instead.

## Examples

Let's check a small DM dataset with the built-in rules and get its failing records:

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

findings = report.findings_df()

findings.drop("description")
```

shape: (3, 6)

| rule_id    | dataset | row_index | usubjid       | checked_column | checked_value |
|------------|---------|-----------|---------------|----------------|---------------|
| str        | str     | i64       | str           | str            | str           |
| "SDTM-007" | "DM"    | 2         | "STUDY01-003" | "SEX"          | "FEMALE"      |
| "SDTM-011" | "DM"    | 1         | "STUDY01-002" | "RFSTDTC"      | "15JAN2024"   |
| "SDTM-195" | "DM"    | 2         | "STUDY01-003" | "AGE"          | "-38"         |

Since `row_index` is 0-based, it can be used to get the failing records from the dataset:

``` python
dm[findings["row_index"].unique().sort().to_list()]
```

shape: (2, 6)

| STUDYID   | DOMAIN | USUBJID       | RFSTDTC      | AGE | SEX      |
|-----------|--------|---------------|--------------|-----|----------|
| str       | str    | str           | str          | i64 | str      |
| "STUDY01" | "DM"   | "STUDY01-002" | "15JAN2024"  | 51  | "F"      |
| "STUDY01" | "DM"   | "STUDY01-003" | "2024-01-17" | -38 | "FEMALE" |

The second and third records have problems, and the first has none.
