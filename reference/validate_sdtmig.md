# validate_sdtmig()

Check SDTM datasets with Pointblank's bundled SDTMIG conformance rules.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L3148-L3332)

``` python
validate_sdtmig(
    datasets,
    version="3-4",
    ct_packages=None,
    define_xml=None,
    study_id=None,
)
```

Before SDTM datasets are submitted, they are checked for conformance to the SDTM Implementation Guide (SDTMIG). The [validate_sdtmig()](../reference/validate_sdtmig.md#pointblank.validate_sdtmig) function does this with a catalog of 362 rules for SDTMIG 3.4 that comes with Pointblank, so it needs no other software and no network access. It returns a [`ConformanceReport`](../reference/ConformanceReport.md#pointblank.ConformanceReport) with the result of each rule. For the published CDISC and agency rules, use [`validate_cdisc_submission()`](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission), which runs the CDISC CORE engine.

The rules are of these types:

- Record checks (221 rules) check the values of each record, such as that `SEX` uses the terms of its codelist, that dates are in the ISO 8601 format, that `AGE` isn't negative and that required variables aren't null. These rules give the failing records as findings.
- Variable metadata checks (116 rules) check that the required variables of a domain are present, that variables have the right type and that they are in the right order.
- Dataset contents checks (9 rules) check across the records of a dataset, such as that `STUDYID` has the same value in every record and that `AESEQ` is unique for each subject.
- Dataset metadata checks (3 rules) check that every dataset has `STUDYID`, `DOMAIN` and (for subject-level data) `USUBJID`.
- Domain presence checks (3 rules) check that the DM, TS and TA datasets are present.
- Define-XML checks (12 rules) check the datasets against their Define-XML, such as that every variable is declared in it and that values are from the codelists that it declares. These rules need `define_xml=`, and are not applicable without it.

Four of the rules are of `"Warning"` sensitivity: the presence of TA and of `USUBJID`, and the order of the identifier variables and of the sequence variable. All the others are of `"Error"` sensitivity. A rule for a domain that isn't one of the `datasets` (such as an AE rule when there's no AE dataset) has nothing to check, so it passes. The codelist checks ignore case, so `"m"` is accepted for `SEX`. Empty strings (which is how XPT files store missing character values) are treated as nulls. SUPP- and RELREC datasets aren't checked by the rules that apply to all domains, since they have a different structure (with `RDOMAIN` in place of `DOMAIN`).

The rules check each dataset on its own, so they don't check that the subjects of one dataset are in DM. For that, use the `"validate"` engine of the [SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage) method [`validate_conformance()`](../reference/SubmissionPackage.md#pointblank.SubmissionPackage.validate_conformance).

## Parameters

`datasets: dict`  
The SDTM datasets, as a dict of DataFrames keyed by domain name, such as `{"DM": dm, "AE": ae}`. The names are case-insensitive. The DataFrames can be Polars, Pandas or any other kind that Narwhals supports.

`version: str = "3-4"`  
The version of SDTMIG, as `"3-4"` or `"3.4"`. Only version 3.4 has a bundled catalog. The default is `"3-4"`.

`ct_packages: list[str] | None = None`  
The bundled controlled terminology packages for the codelist checks, as a list of names such as `["sdtm-ct-2024-09-27"]`. The default is the latest bundled package, which is `"sdtm-ct-2024-09-27"`.

`define_xml: Any = None`  
The study's Define-XML, as a path to a `define.xml` file or as a [`MetadataPackage`](../reference/MetadataPackage.md#pointblank.MetadataPackage) from [`import_metadata()`](../reference/import_metadata.md#pointblank.import_metadata). The default is `None`, which makes the 12 Define-XML rules not applicable.

`study_id: str | None = None`  
A study identifier, which is kept in the report's `package`. It isn't used by the rules and isn't shown in the report. The default is `None`.

## Returns

`ConformanceReport`  
The report, whose [`is_rules`](../reference/ConformanceReport.md#pointblank.ConformanceReport.is_rules) is `True`. In a notebook it is shown as the table of [`get_tabular_report()`](../reference/ConformanceReport.md#pointblank.ConformanceReport.get_tabular_report).

## Raises

`ValueError`  
If there is no bundled catalog for the `version`. The error message lists the versions that have one.

`FileNotFoundError`  
If one of the `ct_packages` isn't bundled with Pointblank.

## Examples

Here is a DM dataset with all the variables that SDTMIG requires for DM, but with three problems: the second subject's `RFSTDTC` isn't an ISO 8601 date, and the third subject has a negative `AGE` and a `SEX` of `"FEMALE"` (the SEX codelist has `"F"`):

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame(
    {
        "STUDYID": ["STUDY01"] * 3,
        "DOMAIN": ["DM"] * 3,
        "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
        "SUBJID": ["001", "002", "003"],
        "RFSTDTC": ["2024-01-15", "15JAN2024", "2024-01-17"],
        "RFENDTC": ["2024-06-15", "2024-06-16", "2024-06-17"],
        "SITEID": ["S01", "S01", "S02"],
        "AGE": [34, 51, -38],
        "AGEU": ["YEARS"] * 3,
        "SEX": ["M", "F", "FEMALE"],
        "RACE": ["WHITE", "ASIAN", "WHITE"],
        "ARMCD": ["DRUGA", "PBO", "DRUGA"],
        "ARM": ["Drug A", "Placebo", "Drug A"],
        "ACTARMCD": ["DRUGA", "PBO", "DRUGA"],
        "ACTARM": ["Drug A", "Placebo", "Drug A"],
        "COUNTRY": ["USA", "USA", "CAN"],
    }
)

report = pb.validate_sdtmig({"DM": dm})

print(report)
```

    ConformanceReport (Built-in Rules)
      sdtmig 3-4
      362 rules (fail=5, not_applicable=12, pass=345)
      5 issues -- FAIL

Five rules fail, and the 12 Define-XML rules are not applicable. Let's see which rules failed:

``` python
for rule in report.rules(status="fail"):
    print(rule.rule_id, rule.sensitivity, rule.message)
```

    SDTM-007 Error SEX value is not in the SEX codelist.
    SDTM-011 Error RFSTDTC does not conform to ISO 8601 extended datetime format.
    SDTM-121 Error Required domain(s) missing: TS
    SDTM-122 Warning Required domain(s) missing: TA
    SDTM-195 Error AGE must not be negative in DM.

The three problems are each found by a rule. The other two failures are for the TS and TA datasets, which every SDTM submission needs. The failing records can be had as a DataFrame with [`findings_df()`](../reference/ConformanceReport.md#pointblank.ConformanceReport.findings_df) or as a table with [`get_findings_table()`](../reference/ConformanceReport.md#pointblank.ConformanceReport.get_findings_table).

With a Define-XML, the Define-XML rules are run as well. Pointblank comes with a DM dataset as an XPT file and a Define-XML for it:

``` python
import pyreadstat

dm, _ = pyreadstat.read_xport(pb.load_metadata_example("dm.xpt"))

report = pb.validate_sdtmig({"DM": dm}, define_xml=pb.load_metadata_example("define.xml"))

print(report)
print([rule.rule_id for rule in report.rules(status="fail")])
```

    ConformanceReport (Built-in Rules)
      sdtmig 3-4
      362 rules (fail=7, pass=355)
      7 issues -- FAIL
    ['SDTM-008', 'SDTM-032', 'SDTM-034', 'SDTM-035', 'SDTM-057', 'SDTM-121', 'SDTM-122']

No rules are not applicable now. SDTM-057 is one of the Define-XML rules, and it finds a `RACE` of `"BLACK"`, which isn't in the codelist that the Define-XML declares for `RACE`. SDTM-008 finds the same value with the CDISC RACE codelist, and the other failures are for missing variables (such as `COUNTRY`) and the missing TS and TA datasets. The dataset was read as a Pandas DataFrame, which works as well as a Polars one.

## See Also

[ConformanceReport](../reference/ConformanceReport.md#pointblank.ConformanceReport)  
The report that is returned.

[validate_cdisc_submission()](../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission)  
Check a submission with the CDISC CORE engine.

[SubmissionPackage](../reference/SubmissionPackage.md#pointblank.SubmissionPackage)  
A set of datasets to check together.

[validate_sdtm()](../reference/validate_sdtm.md#pointblank.validate_sdtm)  
Build a validation plan for one dataset from an SDTM template.
