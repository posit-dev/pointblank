# all_passed()

Check whether the datasets conform, with nothing failing.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L1609-L1650)

``` python
all_passed()
```

This gives a single answer for a conformance check, as is needed in a script or a pipeline that should stop when the data don't conform. What counts as passing depends on the form of the report:

- For the built-in rules, no rule has the status `"fail"`. A failing rule of `"Warning"` sensitivity counts too. Rules that are not applicable or couldn't be evaluated don't.
- For CDISC CORE, no rule reported an issue or an execution error.
- For a Validate-based report, every step of every validation passed (with no failing test units).

## Returns

`bool`  
`True` if nothing failed, otherwise `False`.

## Examples

Let's check a DM dataset with the built-in rules:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"STUDYID": ["STUDY01"], "USUBJID": ["STUDY01-001"], "SEX": ["F"]})

report = pb.validate_sdtmig({"DM": dm})

report.all_passed()
```

    False

The dataset lacks many of the variables that DM requires, and the TS and TA datasets are missing, so some rules fail. The rules that fail can be listed with [`rules()`](../reference/pointblank.ConformanceReport.rules.md), as in `report.rules(status="fail")`.
