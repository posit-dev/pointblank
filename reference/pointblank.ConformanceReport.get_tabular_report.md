# get_tabular_report()

Make a table of the results of the built-in rules.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L2662-L2888)

``` python
get_tabular_report()
```

The table has a row for each rule in the catalog, so it shows at a glance which rules failed and how many issues each one found. It is how a report of the built-in rules is shown in a notebook. For the failing records themselves, use [`get_findings_table()`](../reference/pointblank.ConformanceReport.get_findings_table.md) or [`findings_df()`](../reference/pointblank.ConformanceReport.findings_df.md).

Each row has a colored bar for its status (green for a pass, red for a failure, amber for an error and gray for a rule that is not applicable or not supported), the rule's identifier, its domain, its type (such as `Record` for a check of values or `Variable` for a check that variables are present), its number of issues and its description. The failing rules come first, then the rules with errors, the passing rules and the rules that are not applicable or not supported. The title has a `PASS` or `FAIL` badge, and the subtitle has the standard and its version, the agency (if one was given) and the number of rules with each status.

## Returns

`GT`  
The table, as a [Great Tables](https://posit-dev.github.io/great-tables/) `GT` object.

## Raises

`TypeError`  
If the report isn't of the built-in rules.

## Examples

Let's check a small DM dataset with the built-in rules and make the table:

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

report.get_tabular_report()
```

| CDISC Conformance FAIL |  |  |  |  |  |
|----|----|----|----|----|----|
| SDTMIG 3-4 · 340 passed · 10 failed · 12 n/a |  |  |  |  |  |
|  | Rule | Domain | Type | Issues | Description |
| \#FF3300 | SDTM-007 | DM | Record | 1 | SEX in DM must use values from the CDISC controlled terminology codelist SEX. |
| \#FF3300 | SDTM-011 | DM | Record | 1 | RFSTDTC in DM must be in ISO 8601 extended datetime format when present. |
| \#FF3300 | SDTM-031 | DM | Variable | 1 | DM domain must contain all required Identifier and Topic variables. |
| \#FF3300 | SDTM-032 | DM | Variable | 1 | DM domain must contain required demographic variables (SEX, RACE, COUNTRY). |
| \#FF3300 | SDTM-033 | DM | Variable | 1 | DM domain must contain RFSTDTC and RFENDTC (reference start/end dates). |
| \#FF3300 | SDTM-034 | DM | Variable | 1 | DM domain must contain SITEID, AGE, and AGEU. |
| \#FF3300 | SDTM-035 | DM | Variable | 1 | DM domain must contain treatment arm variables (ARMCD, ARM, ACTARMCD, ACTARM). |
| \#FF3300 | SDTM-121 | TS | Domain | 1 | TS domain must be present in every SDTM submission. |
| \#FF3300 | SDTM-122 | TA | Domain | 1 | TA domain must be present in every SDTM submission. |
| \#FF3300 | SDTM-195 | DM | Record | 1 | AGE must not be negative in the DM dataset. |
| \#4CA64C | SDTM-001 | DM | Domain | 0 | DM (Demographics) domain is required in every SDTM submission. |
| \#4CA64C | SDTM-002 | DM | Record | 0 | STUDYID must not be null in any domain. |
| \#4CA64C | SDTM-003 | DM | Record | 0 | DOMAIN must not be null in any SDTM dataset. |
| \#4CA64C | SDTM-004 | DM | Record | 0 | USUBJID must not be null in any SDTM dataset. |
| \#4CA64C | SDTM-005 | DM | Dataset | 0 | STUDYID must be consistent (same value) across all records in a dataset. |
| \#4CA64C | SDTM-006 | DM | Dataset | 0 | DOMAIN must be consistent (same value) across all records in a dataset. |
| \#4CA64C | SDTM-008 | DM | Record | 0 | RACE in DM must use values from the CDISC controlled terminology codelist RACE. |
| \#4CA64C | SDTM-009 | DM | Record | 0 | COUNTRY in DM must use ISO 3166 alpha-3 country codes (CDISC COUNTRY codelist). |
| \#4CA64C | SDTM-010 | DM | Record | 0 | DMDTC in DM must be in ISO 8601 extended datetime format. |
| \#4CA64C | SDTM-012 | DM | Record | 0 | RFENDTC in DM must be in ISO 8601 extended datetime format when present. |
| \#4CA64C | SDTM-013 | DM | Record | 0 | DTHDTC in DM must be in ISO 8601 extended datetime format when present. |
| \#4CA64C | SDTM-014 | DM | Record | 0 | DTHFL in DM must use values from the NY codelist (Y or null). |
| \#4CA64C | SDTM-015 | DM | Record | 0 | SUBJID must not be null in DM. |
| \#4CA64C | SDTM-016 | AE | Record | 0 | AETERM must not be null in AE. |
| \#4CA64C | SDTM-017 | AE | Record | 0 | AEDECOD must not be null in AE. |
| \#4CA64C | SDTM-018 | AE | Record | 0 | AESTDTC in AE must be in ISO 8601 extended datetime format when present. |
| \#4CA64C | SDTM-019 | AE | Record | 0 | AEENDTC in AE must be in ISO 8601 extended datetime format when present. |
| \#4CA64C | SDTM-020 | AE | Record | 0 | AESER in AE must use values from the NY codelist when present. |
| \#4CA64C | SDTM-021 | AE | Record | 0 | AEOUT in AE must use values from the OUT (Outcome of Event) codelist when present. |
| \#4CA64C | SDTM-022 | LB | Record | 0 | LBDTC in LB must be in ISO 8601 extended datetime format when present. |
| \#4CA64C | SDTM-023 | LB | Record | 0 | LBTEST must not be null in LB. |
| \#4CA64C | SDTM-024 | LB | Record | 0 | LBTESTCD must not be null in LB. |
| \#4CA64C | SDTM-025 | VS | Record | 0 | VSTEST must not be null in VS. |
| \#4CA64C | SDTM-026 | VS | Record | 0 | VSDTC in VS must be in ISO 8601 extended datetime format when present. |
| \#4CA64C | SDTM-027 | DM | Metadata | 0 | USUBJID must be present in every subject-level SDTM domain. |
| \#4CA64C | SDTM-028 | DM | Metadata | 0 | STUDYID must be present in every SDTM domain. |
| \#4CA64C | SDTM-029 | DM | Metadata | 0 | DOMAIN must be present in every SDTM domain. |
| \#4CA64C | SDTM-030 | DM | Record | 0 | ETHNIC in DM must use values from the ETHNIC codelist when present. |
| \#4CA64C | SDTM-036 | AE | Variable | 0 | AE domain must contain required Identifier and sequence variables. |
| \#4CA64C | SDTM-037 | AE | Variable | 0 | AE domain must contain the AE term and dictionary-derived variables. |
| \#4CA64C | SDTM-038 | AE | Variable | 0 | AE domain must contain the seriousness variable (AESER). |
| \#4CA64C | SDTM-039 | VS | Variable | 0 | VS domain must contain required Identifier and test variables. |
| \#4CA64C | SDTM-040 | VS | Variable | 0 | VS domain must contain result variables (VSORRES, VSORRESU, VSSTRESC, VSSTRESN, VSSTRESU). |
| \#4CA64C | SDTM-041 | LB | Variable | 0 | LB domain must contain required Identifier and test variables. |
| \#4CA64C | SDTM-042 | LB | Variable | 0 | LB domain must contain result variables (LBORRES, LBORRESU, LBSTRESC, LBSTRESN, LBSTRESU). |
| \#4CA64C | SDTM-043 | EG | Variable | 0 | EG domain must contain required Identifier and test variables. |
| \#4CA64C | SDTM-044 | DM | Variable | 0 | In any domain, STUDYID must appear before DOMAIN, which must appear before USUBJID. |
| \#4CA64C | SDTM-045 | VS | Variable | 0 | In findings domains (VS, LB, EG), the sequence variable (xSEQ) must appear after USUBJID. |
| \#4CA64C | SDTM-046 | DM | Variable | 0 | In DM, AGE must be a numeric variable. |
| \#4CA64C | SDTM-047 | VS | Variable | 0 | In VS, VSSTRESN must be a numeric variable. |
| \#4CA64C | SDTM-048 | LB | Variable | 0 | In LB, LBSTRESN must be a numeric variable. |
| \#4CA64C | SDTM-061 | DM | Record | 0 | AGEU in DM must use AGEU codelist when present. |
| \#4CA64C | SDTM-062 | DM | Record | 0 | BRTHDTC in DM must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-063 | EX | Record | 0 | EXTRT must not be null in EX. |
| \#4CA64C | SDTM-064 | EX | Record | 0 | EXSTDTC in EX must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-065 | EX | Record | 0 | EXENDTC in EX must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-066 | EX | Record | 0 | EXDOSFRQ in EX must use FREQ codelist when present. |
| \#4CA64C | SDTM-067 | EX | Record | 0 | EXROUTE in EX must use ROUTE codelist when present. |
| \#4CA64C | SDTM-068 | EX | Record | 0 | EXDOSFRM in EX must use FRM codelist when present. |
| \#4CA64C | SDTM-069 | CM | Record | 0 | CMTRT must not be null in CM. |
| \#4CA64C | SDTM-070 | CM | Record | 0 | CMSTDTC in CM must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-071 | CM | Record | 0 | CMENDTC in CM must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-072 | CM | Record | 0 | CMROUTE in CM must use ROUTE codelist when present. |
| \#4CA64C | SDTM-073 | DS | Record | 0 | DSTERM must not be null in DS. |
| \#4CA64C | SDTM-074 | DS | Record | 0 | DSDECOD must not be null in DS. |
| \#4CA64C | SDTM-075 | DS | Record | 0 | DSSTDTC in DS must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-076 | DS | Record | 0 | DSSEQ must not be null in DS. |
| \#4CA64C | SDTM-077 | MH | Record | 0 | MHTERM must not be null in MH. |
| \#4CA64C | SDTM-078 | MH | Record | 0 | MHSTDTC in MH must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-079 | MH | Record | 0 | MHENDTC in MH must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-080 | SV | Record | 0 | VISITNUM must not be null in SV. |
| \#4CA64C | SDTM-081 | SV | Record | 0 | SVSTDTC in SV must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-082 | SV | Record | 0 | SVENDTC in SV must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-083 | EG | Record | 0 | EGTEST must not be null in EG. |
| \#4CA64C | SDTM-084 | EG | Record | 0 | EGTESTCD must not be null in EG. |
| \#4CA64C | SDTM-085 | EG | Record | 0 | EGDTC in EG must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-086 | AE | Record | 0 | AESEV in AE must use AESEV codelist when present. |
| \#4CA64C | SDTM-089 | AE | Record | 0 | AEACN in AE must use the ACN (Action Taken with Study Treatment) codelist when present. |
| \#4CA64C | SDTM-090 | AE | Record | 0 | AESEQ must not be null in AE. |
| \#4CA64C | SDTM-091 | VS | Record | 0 | VSTESTCD must not be null in VS. |
| \#4CA64C | SDTM-092 | VS | Record | 0 | VSSTRESU in VS must use VSRESU codelist when present. |
| \#4CA64C | SDTM-093 | VS | Record | 0 | VSBLFL in VS must use NY codelist when present. |
| \#4CA64C | SDTM-094 | LB | Record | 0 | LBSTRESU in LB must use the UNIT codelist when present. |
| \#4CA64C | SDTM-095 | LB | Record | 0 | LBBLFL in LB must use NY codelist when present. |
| \#4CA64C | SDTM-096 | LB | Record | 0 | LBNRIND in LB must use NRIND codelist when present. |
| \#4CA64C | SDTM-097 | PE | Record | 0 | PETEST must not be null in PE. |
| \#4CA64C | SDTM-098 | PE | Record | 0 | PEDTC in PE must conform to ISO 8601 when present. |
| \#4CA64C | SDTM-099 | IE | Record | 0 | IETEST must not be null in IE. |
| \#4CA64C | SDTM-100 | IE | Record | 0 | IECAT in IE must use IECAT codelist when present. |
| \#4CA64C | SDTM-101 | SUPP-- | Record | 0 | RDOMAIN must not be null in SUPP-- datasets. |
| \#4CA64C | SDTM-102 | SUPP-- | Record | 0 | QNAM must not be null in SUPP-- datasets. |
| \#4CA64C | SDTM-103 | SUPP-- | Record | 0 | QLABEL must not be null in SUPP-- datasets. |
| \#4CA64C | SDTM-104 | TS | Record | 0 | TSPARMCD must not be null in TS. |
| \#4CA64C | SDTM-105 | TS | Record | 0 | TSPARM must not be null in TS. |
| \#4CA64C | SDTM-106 | DM | Dataset | 0 | STUDYID must be consistent (same value) across all records within each domain. |
| \#4CA64C | SDTM-107 | DM | Dataset | 0 | DOMAIN must be consistent (same value) across all records within each dataset. |
| \#4CA64C | SDTM-108 | EX | Variable | 0 | EX domain must contain required identifier and treatment variables. |
| \#4CA64C | SDTM-109 | EX | Variable | 0 | EX domain must contain timing variables EXSTDTC and EXENDTC. |
| \#4CA64C | SDTM-110 | CM | Variable | 0 | CM domain must contain required identifier and treatment variables. |
| \#4CA64C | SDTM-112 | DS | Variable | 0 | DS domain must contain required identifier and disposition variables. |
| \#4CA64C | SDTM-113 | MH | Variable | 0 | MH domain must contain required identifier and history variables. |
| \#4CA64C | SDTM-114 | SV | Variable | 0 | SV domain must contain required identifier and visit variables. |
| \#4CA64C | SDTM-115 | TS | Variable | 0 | TS domain must contain required parameter variables. |
| \#4CA64C | SDTM-116 | IE | Variable | 0 | IE domain must contain required inclusion/exclusion criteria variables. |
| \#4CA64C | SDTM-117 | PE | Variable | 0 | PE domain must contain required physical examination variables. |
| \#4CA64C | SDTM-118 | EX | Variable | 0 | EXDOSE must be a numeric variable in EX. |
| \#4CA64C | SDTM-119 | SV | Variable | 0 | VISITNUM must be a numeric variable in SV. |
| \#4CA64C | SDTM-120 | EG | Variable | 0 | EGSTRESN must be a numeric variable in EG. |
| \#4CA64C | SDTM-123 | TA | Record | 0 | ARMCD must not be null in TA. |
| \#4CA64C | SDTM-124 | TA | Record | 0 | ARM must not be null in TA. |
| \#4CA64C | SDTM-125 | TA | Record | 0 | TAETORD must not be null in TA. |
| \#4CA64C | SDTM-126 | TA | Record | 0 | EPOCH must not be null in TA. |
| \#4CA64C | SDTM-127 | TE | Record | 0 | ETCD must not be null in TE. |
| \#4CA64C | SDTM-128 | TE | Record | 0 | ELEMENT must not be null in TE. |
| \#4CA64C | SDTM-129 | TV | Record | 0 | VISITNUM must not be null in TV. |
| \#4CA64C | SDTM-131 | TI | Record | 0 | IETESTCD must not be null in TI. |
| \#4CA64C | SDTM-132 | TI | Record | 0 | IETEST must not be null in TI. |
| \#4CA64C | SDTM-133 | TI | Record | 0 | IECAT in TI must use IECAT controlled terminology when present. |
| \#4CA64C | SDTM-136 | AE | Record | 0 | AESDTH in AE must use NY controlled terminology when present. |
| \#4CA64C | SDTM-137 | AE | Record | 0 | AESHOSP in AE must use NY controlled terminology when present. |
| \#4CA64C | SDTM-138 | AE | Record | 0 | AESLIFE in AE must use NY controlled terminology when present. |
| \#4CA64C | SDTM-139 | EX | Record | 0 | EXSEQ must not be null in EX. |
| \#4CA64C | SDTM-140 | CM | Record | 0 | CMSEQ must not be null in CM. |
| \#4CA64C | SDTM-141 | MH | Record | 0 | MHSEQ must not be null in MH. |
| \#4CA64C | SDTM-142 | EG | Record | 0 | EGSEQ must not be null in EG. |
| \#4CA64C | SDTM-143 | VS | Record | 0 | VSSEQ must not be null in VS. |
| \#4CA64C | SDTM-144 | LB | Record | 0 | LBSEQ must not be null in LB. |
| \#4CA64C | SDTM-145 | PE | Record | 0 | PESEQ must not be null in PE. |
| \#4CA64C | SDTM-146 | IE | Record | 0 | IESEQ must not be null in IE. |
| \#4CA64C | SDTM-147 | AE | Dataset | 0 | AESEQ must be unique within USUBJID in AE. |
| \#4CA64C | SDTM-148 | EX | Dataset | 0 | EXSEQ must be unique within USUBJID in EX. |
| \#4CA64C | SDTM-149 | CM | Dataset | 0 | CMSEQ must be unique within USUBJID in CM. |
| \#4CA64C | SDTM-150 | LB | Dataset | 0 | LBSEQ must be unique within USUBJID in LB. |
| \#4CA64C | SDTM-151 | VS | Dataset | 0 | VSSEQ must be unique within USUBJID in VS. |
| \#4CA64C | SDTM-152 | AE | Variable | 0 | AESTDY must be numeric type in AE. |
| \#4CA64C | SDTM-153 | AE | Variable | 0 | AEENDY must be numeric type in AE. |
| \#4CA64C | SDTM-154 | LB | Variable | 0 | LBDY must be numeric type in LB. |
| \#4CA64C | SDTM-155 | VS | Variable | 0 | VSDY must be numeric type in VS. |
| \#4CA64C | SDTM-156 | EX | Variable | 0 | EXSTDY must be numeric type in EX. |
| \#4CA64C | SDTM-157 | CM | Variable | 0 | CMSTDY must be numeric type in CM. |
| \#4CA64C | SDTM-158 | TA | Variable | 0 | TA must contain required variables: STUDYID, DOMAIN, ARMCD, ARM, TAETORD, EPOCH. |
| \#4CA64C | SDTM-159 | TE | Variable | 0 | TE must contain required variables: STUDYID, DOMAIN, ETCD, ELEMENT. |
| \#4CA64C | SDTM-160 | TV | Variable | 0 | TV must contain required variables: STUDYID, DOMAIN, VISITNUM. |
| \#4CA64C | SDTM-161 | TI | Variable | 0 | TI must contain required variables: STUDYID, DOMAIN, IETESTCD, IETEST, IECAT. |
| \#4CA64C | SDTM-163 | VS | Variable | 0 | VISITNUM must be present in VS. |
| \#4CA64C | SDTM-164 | LB | Variable | 0 | VISITNUM must be present in LB. |
| \#4CA64C | SDTM-165 | EG | Variable | 0 | VISITNUM must be present in EG. |
| \#4CA64C | SDTM-166 | SUPP-- | Record | 0 | QVAL must not be null in SUPP-- supplemental qualifier datasets. |
| \#4CA64C | SDTM-167 | TS | Record | 0 | TSSEQ must not be null in TS. |
| \#4CA64C | SDTM-168 | LB | Variable | 0 | LBSTNRLO must be numeric type in LB. |
| \#4CA64C | SDTM-169 | LB | Variable | 0 | LBSTNRHI must be numeric type in LB. |
| \#4CA64C | SDTM-172 | AE | Variable | 0 | AESEQ must be numeric type in AE. |
| \#4CA64C | SDTM-173 | EX | Variable | 0 | EXSEQ must be numeric type in EX. |
| \#4CA64C | SDTM-174 | CM | Variable | 0 | CMSEQ must be numeric type in CM. |
| \#4CA64C | SDTM-175 | MH | Variable | 0 | MHSEQ must be numeric type in MH. |
| \#4CA64C | SDTM-176 | VS | Variable | 0 | VSSEQ must be numeric type in VS. |
| \#4CA64C | SDTM-177 | LB | Variable | 0 | LBSEQ must be numeric type in LB. |
| \#4CA64C | SDTM-178 | EG | Variable | 0 | EGSEQ must be numeric type in EG. |
| \#4CA64C | SDTM-179 | PE | Variable | 0 | PESEQ must be numeric type in PE. |
| \#4CA64C | SDTM-180 | DS | Variable | 0 | DSSEQ must be numeric type in DS. |
| \#4CA64C | SDTM-181 | DM | Record | 0 | DOMAIN column value must equal 'DM' in the DM dataset. |
| \#4CA64C | SDTM-182 | AE | Record | 0 | DOMAIN column value must equal 'AE' in the AE dataset. |
| \#4CA64C | SDTM-183 | LB | Record | 0 | DOMAIN column value must equal 'LB' in the LB dataset. |
| \#4CA64C | SDTM-184 | VS | Record | 0 | DOMAIN column value must equal 'VS' in the VS dataset. |
| \#4CA64C | SDTM-185 | EX | Record | 0 | DOMAIN column value must equal 'EX' in the EX dataset. |
| \#4CA64C | SDTM-186 | CM | Record | 0 | DOMAIN column value must equal 'CM' in the CM dataset. |
| \#4CA64C | SDTM-187 | DS | Record | 0 | DOMAIN column value must equal 'DS' in the DS dataset. |
| \#4CA64C | SDTM-188 | MH | Record | 0 | DOMAIN column value must equal 'MH' in the MH dataset. |
| \#4CA64C | SDTM-189 | EG | Record | 0 | DOMAIN column value must equal 'EG' in the EG dataset. |
| \#4CA64C | SDTM-190 | TA | Record | 0 | DOMAIN column value must equal 'TA' in the TA dataset. |
| \#4CA64C | SDTM-191 | DM | Record | 0 | RFXSTDTC in DM must conform to ISO 8601 format when present (date of first exposure to trial treatment). |
| \#4CA64C | SDTM-192 | DM | Record | 0 | RFXENDTC in DM must conform to ISO 8601 format when present (date of last exposure to trial treatment). |
| \#4CA64C | SDTM-193 | DM | Record | 0 | RFICDTC in DM must conform to ISO 8601 format when present (date of informed consent). |
| \#4CA64C | SDTM-194 | DM | Record | 0 | RFPENDTC in DM must conform to ISO 8601 format when present (date of end of participation). |
| \#4CA64C | SDTM-197 | AE | Record | 0 | AESCONG (congenital anomaly flag) must use the NY codelist when present in AE. |
| \#4CA64C | SDTM-198 | AE | Record | 0 | AESDISAB (disability flag) must use the NY codelist when present in AE. |
| \#4CA64C | SDTM-199 | AE | Record | 0 | AESMIE (important medical event flag) must use the NY codelist when present in AE. |
| \#4CA64C | SDTM-200 | AE | Record | 0 | AECONTRT (concomitant treatment given flag) must use the NY codelist when present in AE. |
| \#4CA64C | SDTM-202 | MH | Record | 0 | MHPRESP (pre-specified) must use the NY codelist when present in MH. |
| \#4CA64C | SDTM-203 | MH | Record | 0 | MHOCCUR (occurrence) must use the NY codelist when present in MH. |
| \#4CA64C | SDTM-204 | EX | Record | 0 | EXOCCUR (occurrence) must use the NY codelist when present in EX. |
| \#4CA64C | SDTM-205 | EX | Variable | 0 | EXDOSTOT (total cumulative dose) must be a numeric variable in EX. |
| \#4CA64C | SDTM-206 | EX | Variable | 0 | EXDOSNO (dose number in a sequence) must be a numeric variable in EX. |
| \#4CA64C | SDTM-207 | CM | Record | 0 | CMOCCUR (occurrence) must use the NY codelist when present in CM. |
| \#4CA64C | SDTM-208 | CM | Variable | 0 | CMDOSE (dose per administration) must be a numeric variable in CM. |
| \#4CA64C | SDTM-210 | DS | Variable | 0 | DSSTDY (study day of disposition event) must be a numeric variable in DS. |
| \#4CA64C | SDTM-211 | PE | Record | 0 | PETESTCD (test short name) must not be null in the PE dataset. |
| \#4CA64C | SDTM-212 | PE | Variable | 0 | PESTRESN (numeric result in standard units) must be a numeric variable in PE. |
| \#4CA64C | SDTM-213 | IE | Record | 0 | IEORRES (original result) must not be null in the IE dataset. |
| \#4CA64C | SDTM-214 | IE | Record | 0 | IEDTC (date/time of IE assessment) must conform to ISO 8601 format when present in IE. |
| \#4CA64C | SDTM-215 | SV | Variable | 0 | SVSTDY (study day of visit start) must be a numeric variable in SV. |
| \#4CA64C | SDTM-216 | SV | Variable | 0 | SVENDY (study day of visit end) must be a numeric variable in SV. |
| \#4CA64C | SDTM-218 | TS | Variable | 0 | TSSEQ (sequence number) must be a numeric variable in TS. |
| \#4CA64C | SDTM-219 | LB | Record | 0 | LBFAST (fasting status) must use the NY codelist when present in LB. |
| \#4CA64C | SDTM-220 | LB | Record | 0 | LBCLSIG (clinically significant) must use the NY codelist when present in LB. |
| \#4CA64C | SDTM-221 | LB | Variable | 0 | LBLLOQ (lower limit of quantification) must be a numeric variable in LB. |
| \#4CA64C | SDTM-222 | TE | Record | 0 | TESTRL (trial element start rule) must not be null in the TE dataset. |
| \#4CA64C | SDTM-223 | TE | Variable | 0 | TESTRL (trial element start rule) must be present as a variable in the TE dataset. |
| \#4CA64C | SDTM-224 | TV | Record | 0 | TVSTRL (trial visit start rule) must not be null in the TV dataset. |
| \#4CA64C | SDTM-225 | TV | Variable | 0 | VISITNUM must be a numeric variable in TV. |
| \#4CA64C | SDTM-226 | TA | Variable | 0 | TAETORD (order of element within arm) must be a numeric variable in TA. |
| \#4CA64C | SDTM-227 | TA | Record | 0 | ETCD (element code) must not be null in the TA dataset. |
| \#4CA64C | SDTM-228 | IE | Record | 0 | DOMAIN column value must equal 'IE' in the IE dataset. |
| \#4CA64C | SDTM-229 | SV | Record | 0 | DOMAIN column value must equal 'SV' in the SV dataset. |
| \#4CA64C | SDTM-230 | PE | Record | 0 | DOMAIN column value must equal 'PE' in the PE dataset. |
| \#4CA64C | SDTM-231 | TS | Record | 0 | DOMAIN column value must equal 'TS' in the TS dataset. |
| \#4CA64C | SDTM-232 | TE | Record | 0 | DOMAIN column value must equal 'TE' in the TE dataset. |
| \#4CA64C | SDTM-233 | EG | Variable | 0 | EG domain must contain required result variables: EGORRES, EGSTRESC, EGSTRESN, EGSTRESU. |
| \#4CA64C | SDTM-234 | PE | Variable | 0 | PE domain must contain required result variables: PESTRESC, PESTRESN. |
| \#4CA64C | SDTM-235 | AE | Variable | 0 | AESTDTC (start date/time of adverse event) must be present as a variable in the AE dataset. |
| \#4CA64C | SDTM-236 | EX | Variable | 0 | EXDOSE (dose per administration) must be present as a variable in the EX dataset. |
| \#4CA64C | SDTM-238 | TV | Record | 0 | DOMAIN column value must equal 'TV' in the TV dataset. |
| \#4CA64C | SDTM-239 | TI | Record | 0 | DOMAIN column value must equal 'TI' in the TI dataset. |
| \#4CA64C | SDTM-241 | SUPPDM | Record | 0 | QNAM must not be null in SUPPDM. |
| \#4CA64C | SDTM-242 | SUPPDM | Record | 0 | QVAL must not be null in SUPPDM. |
| \#4CA64C | SDTM-243 | SUPPAE | Record | 0 | QNAM must not be null in SUPPAE. |
| \#4CA64C | SDTM-244 | SUPPAE | Record | 0 | QVAL must not be null in SUPPAE. |
| \#4CA64C | SDTM-245 | SUPPLB | Record | 0 | QNAM must not be null in SUPPLB. |
| \#4CA64C | SDTM-246 | SUPPDM | Record | 0 | RDOMAIN in SUPPDM must equal 'DM'. |
| \#4CA64C | SDTM-247 | SUPPAE | Record | 0 | RDOMAIN in SUPPAE must equal 'AE'. |
| \#4CA64C | SDTM-248 | SUPPLB | Record | 0 | RDOMAIN in SUPPLB must equal 'LB'. |
| \#4CA64C | SDTM-249 | SUPPVS | Record | 0 | RDOMAIN in SUPPVS must equal 'VS'. |
| \#4CA64C | SDTM-250 | SUPPVS | Record | 0 | QNAM must not be null in SUPPVS. |
| \#4CA64C | SDTM-251 | AE | Record | 0 | AESTDTC in AE must conform to ISO 8601 format when present. |
| \#4CA64C | SDTM-255 | EX | Record | 0 | EXSTDTC in EX must conform to ISO 8601 format when present. |
| \#4CA64C | SDTM-258 | MH | Record | 0 | MHENDTC in MH must conform to ISO 8601 format when present. |
| \#4CA64C | SDTM-259 | DS | Record | 0 | DSSTDTC in DS must conform to ISO 8601 format when present. |
| \#4CA64C | SDTM-261 | VS | Record | 0 | VSTESTCD (test short name) must not be null in VS. |
| \#4CA64C | SDTM-262 | LB | Record | 0 | LBTESTCD (test short name) must not be null in LB. |
| \#4CA64C | SDTM-263 | EG | Record | 0 | EGTESTCD (test short name) must not be null in EG. |
| \#4CA64C | SDTM-264 | VS | Record | 0 | VSTEST (test name) must not be null in VS. |
| \#4CA64C | SDTM-265 | LB | Record | 0 | LBTEST (test name) must not be null in LB. |
| \#4CA64C | SDTM-268 | EG | Record | 0 | EGORRES (original result) must not be null in EG. |
| \#4CA64C | SDTM-269 | PE | Record | 0 | PEORRES (original result) must not be null in PE. |
| \#4CA64C | SDTM-270 | LB | Variable | 0 | VISITNUM must be a numeric variable in LB. |
| \#4CA64C | SDTM-271 | AE | Record | 0 | AESEQ (sequence number) must not be null in AE. |
| \#4CA64C | SDTM-272 | CM | Record | 0 | CMSEQ (sequence number) must not be null in CM. |
| \#4CA64C | SDTM-273 | EX | Record | 0 | EXSEQ (sequence number) must not be null in EX. |
| \#4CA64C | SDTM-274 | MH | Record | 0 | MHSEQ (sequence number) must not be null in MH. |
| \#4CA64C | SDTM-275 | VS | Record | 0 | VSSEQ (sequence number) must not be null in VS. |
| \#4CA64C | SDTM-276 | LB | Record | 0 | LBSEQ (sequence number) must not be null in LB. |
| \#4CA64C | SDTM-277 | EG | Record | 0 | EGSEQ (sequence number) must not be null in EG. |
| \#4CA64C | SDTM-278 | PE | Record | 0 | PESEQ (sequence number) must not be null in PE. |
| \#4CA64C | SDTM-279 | DS | Record | 0 | DSSEQ (sequence number) must not be null in DS. |
| \#4CA64C | SDTM-280 | IE | Record | 0 | IESEQ (sequence number) must not be null in IE. |
| \#4CA64C | SDTM-281 | LB | Variable | 0 | LBSTNRLO (lower limit of reference range) must be a numeric variable in LB. |
| \#4CA64C | SDTM-282 | LB | Variable | 0 | LBSTNRHI (upper limit of reference range) must be a numeric variable in LB. |
| \#4CA64C | SDTM-283 | VS | Variable | 0 | VSSTRESN (numeric result in standard units) must be a numeric variable in VS. |
| \#4CA64C | SDTM-285 | VS | Record | 0 | VSNRIND (reference range indicator) must use values from the NRIND controlled terminology codelist when present in VS. |
| \#4CA64C | SDTM-286 | EG | Record | 0 | EGNRIND (reference range indicator) must use values from the NRIND controlled terminology codelist when present in EG. |
| \#4CA64C | SDTM-287 | PE | Record | 0 | PENRIND (reference range indicator) must use values from the NRIND controlled terminology codelist when present in PE. |
| \#4CA64C | SDTM-288 | EG | Variable | 0 | VISITNUM must be a numeric variable in EG. |
| \#4CA64C | SDTM-289 | AE | Variable | 0 | AESER (serious event flag) must be present in the AE dataset. |
| \#4CA64C | SDTM-290 | AE | Variable | 0 | AESEQ must be a numeric variable in AE. |
| \#4CA64C | SDTM-291 | VS | Variable | 0 | VSSEQ must be a numeric variable in VS. |
| \#4CA64C | SDTM-292 | LB | Variable | 0 | LBSEQ must be a numeric variable in LB. |
| \#4CA64C | SDTM-293 | EG | Variable | 0 | EGSEQ must be a numeric variable in EG. |
| \#4CA64C | SDTM-294 | CM | Variable | 0 | CMSEQ must be a numeric variable in CM. |
| \#4CA64C | SDTM-295 | DS | Variable | 0 | DSSEQ must be a numeric variable in DS. |
| \#4CA64C | SDTM-296 | EX | Variable | 0 | EXSEQ must be a numeric variable in EX. |
| \#4CA64C | SDTM-298 | MH | Variable | 0 | MHSEQ must be a numeric variable in MH. |
| \#4CA64C | SDTM-299 | SUPPVS | Record | 0 | QVAL must not be null in SUPPVS. |
| \#4CA64C | SDTM-300 | PE | Record | 0 | PETEST (physical exam test name) must not be null in PE. |
| \#4CA64C | SDTM-301 | AE | Record | 0 | AETERM (verbatim adverse event term) must not be null in AE. |
| \#4CA64C | SDTM-302 | AE | Record | 0 | AEDECOD (dictionary-derived preferred term) must not be null in AE. |
| \#4CA64C | SDTM-304 | CM | Record | 0 | CMTRT (verbatim medication name) must not be null in CM. |
| \#4CA64C | SDTM-306 | DS | Record | 0 | DSTERM (verbatim disposition term) must not be null in DS. |
| \#4CA64C | SDTM-307 | DS | Record | 0 | DSDECOD (dictionary-derived disposition term) must not be null in DS. |
| \#4CA64C | SDTM-308 | MH | Record | 0 | MHTERM (verbatim medical history term) must not be null in MH. |
| \#4CA64C | SDTM-309 | EX | Record | 0 | EXTRT (treatment name administered) must not be null in EX. |
| \#4CA64C | SDTM-313 | EG | Record | 0 | EGSTRESC (standardized result in character format) must not be null in EG. |
| \#4CA64C | SDTM-314 | PE | Record | 0 | PESTRESC (standardized result in character format) must not be null in PE. |
| \#4CA64C | SDTM-315 | VS | Variable | 0 | VSSTRESU column must be present in VS. |
| \#4CA64C | SDTM-316 | LB | Variable | 0 | LBSTRESU column must be present in LB. |
| \#4CA64C | SDTM-317 | EG | Variable | 0 | EGSTRESU column must be present in EG. |
| \#4CA64C | SDTM-318 | LB | Variable | 0 | LBSTRESN must be numeric type in LB. |
| \#4CA64C | SDTM-319 | EG | Variable | 0 | EGSTRESN must be numeric type in EG. |
| \#4CA64C | SDTM-320 | EX | Variable | 0 | EXDOSE must be numeric type in EX. |
| \#4CA64C | SDTM-321 | TA | Record | 0 | ARM (arm name) must not be null in TA. |
| \#4CA64C | SDTM-322 | TA | Record | 0 | EPOCH (epoch name) must not be null in TA. |
| \#4CA64C | SDTM-328 | TI | Record | 0 | IETEST (inclusion/exclusion criterion name) must not be null in TI. |
| \#4CA64C | SDTM-329 | TI | Record | 0 | IETESTCD (criterion short name) must not be null in TI. |
| \#4CA64C | SDTM-330 | TS | Record | 0 | TSPARMCD (trial summary parameter short name) must not be null in TS. |
| \#4CA64C | SDTM-331 | SV | Record | 0 | VISITNUM must not be null in SV. |
| \#4CA64C | SDTM-333 | SV | Record | 0 | SVSTDTC must conform to ISO 8601 date/time format when present in SV. |
| \#4CA64C | SDTM-334 | SV | Record | 0 | SVENDTC must conform to ISO 8601 date/time format when present in SV. |
| \#4CA64C | SDTM-337 | EG | Record | 0 | VISITNUM must not be null in EG. |
| \#4CA64C | SDTM-338 | PE | Record | 0 | VISITNUM must not be null in PE. |
| \#4CA64C | SDTM-339 | VS | Variable | 0 | VISITNUM must be numeric type in VS. |
| \#4CA64C | SDTM-340 | PE | Variable | 0 | VISITNUM must be numeric type in PE. |
| \#4CA64C | SDTM-341 | SUPPDM | Record | 0 | QLABEL must not be null in SUPPDM. |
| \#4CA64C | SDTM-342 | SUPPAE | Record | 0 | QLABEL must not be null in SUPPAE. |
| \#4CA64C | SDTM-343 | SUPPLB | Record | 0 | QLABEL must not be null in SUPPLB. |
| \#4CA64C | SDTM-344 | SUPPEG | Record | 0 | QNAM must not be null in SUPPEG. |
| \#4CA64C | SDTM-345 | SUPPEG | Record | 0 | QVAL must not be null in SUPPEG. |
| \#4CA64C | SDTM-346 | SUPPEG | Record | 0 | RDOMAIN must equal 'EG' in the SUPPEG dataset. |
| \#4CA64C | SDTM-347 | SUPPCM | Record | 0 | QNAM must not be null in SUPPCM. |
| \#4CA64C | SDTM-348 | SUPPCM | Record | 0 | RDOMAIN must equal 'CM' in the SUPPCM dataset. |
| \#4CA64C | SDTM-349 | SUPPDS | Record | 0 | QNAM must not be null in SUPPDS. |
| \#4CA64C | SDTM-350 | SUPPDS | Record | 0 | RDOMAIN must equal 'DS' in the SUPPDS dataset. |
| \#4CA64C | SDTM-351 | SUPPMH | Record | 0 | QNAM must not be null in SUPPMH. |
| \#4CA64C | SDTM-352 | SUPPMH | Record | 0 | RDOMAIN must equal 'MH' in the SUPPMH dataset. |
| \#4CA64C | SDTM-353 | AE | Variable | 0 | AESTDY (AE start study day) must be numeric type in AE. |
| \#4CA64C | SDTM-354 | AE | Variable | 0 | AEENDY (AE end study day) must be numeric type in AE. |
| \#4CA64C | SDTM-355 | CM | Variable | 0 | CMSTDY (CM start study day) must be numeric type in CM. |
| \#4CA64C | SDTM-356 | EX | Variable | 0 | EXSTDY (EX start study day) must be numeric type in EX. |
| \#4CA64C | SDTM-357 | MH | Variable | 0 | MHSTDY (MH start study day) must be numeric type in MH. |
| \#4CA64C | SDTM-358 | VS | Variable | 0 | VSSTDY (VS study day) must be numeric type in VS. |
| \#4CA64C | SDTM-359 | LB | Variable | 0 | LBSTDY (LB study day) must be numeric type in LB. |
| \#4CA64C | SDTM-360 | EG | Variable | 0 | EGSTDY (EG study day) must be numeric type in EG. |
| \#4CA64C | SDTM-361 | VS | Record | 0 | VSDTC in VS must conform to ISO 8601 format when present. |
| \#4CA64C | SDTM-362 | LB | Record | 0 | LBDTC in LB must conform to ISO 8601 format when present. |
| \#4CA64C | SDTM-363 | EG | Record | 0 | EGDTC in EG must conform to ISO 8601 format when present. |
| \#4CA64C | SDTM-364 | PE | Record | 0 | PEDTC in PE must conform to ISO 8601 format when present. |
| \#4CA64C | SDTM-365 | AE | Record | 0 | AEDTC in AE must conform to ISO 8601 format when present. |
| \#4CA64C | SDTM-366 | DM | Record | 0 | BRTHDTC in DM must conform to ISO 8601 format when present. |
| \#4CA64C | SDTM-368 | EX | Variable | 0 | EXENDY in EX must be a numeric type variable. |
| \#4CA64C | SDTM-369 | CM | Variable | 0 | CMENDY in CM must be a numeric type variable. |
| \#4CA64C | SDTM-370 | MH | Variable | 0 | MHENDY in MH must be a numeric type variable. |
| \#4CA64C | SDTM-374 | EX | Variable | 0 | EXDOSFRM (dose form) column must be present in the EX domain. |
| \#4CA64C | SDTM-377 | VS | Variable | 0 | VSDY in VS must be a numeric type variable. |
| \#4CA64C | SDTM-378 | LB | Variable | 0 | LBDY in LB must be a numeric type variable. |
| \#4CA64C | SDTM-379 | EG | Variable | 0 | EGDY in EG must be a numeric type variable. |
| \#4CA64C | SDTM-380 | PE | Variable | 0 | PEDY in PE must be a numeric type variable. |
| \#4CA64C | SDTM-381 | SUPPVS | Record | 0 | QLABEL must not be null in SUPPVS; the qualifier label is required in all supplemental datasets. |
| \#4CA64C | SDTM-382 | SUPPEG | Record | 0 | QLABEL must not be null in SUPPEG; the qualifier label is required in all supplemental datasets. |
| \#4CA64C | SDTM-383 | SUPPCM | Record | 0 | QLABEL must not be null in SUPPCM; the qualifier label is required in all supplemental datasets. |
| \#4CA64C | SDTM-384 | SUPPDS | Record | 0 | QLABEL must not be null in SUPPDS; the qualifier label is required in all supplemental datasets. |
| \#4CA64C | SDTM-385 | SUPPMH | Record | 0 | QLABEL must not be null in SUPPMH; the qualifier label is required in all supplemental datasets. |
| \#4CA64C | SDTM-394 | TS | Record | 0 | TSPARM must not be null in TS; the parameter name is required for every trial summary record. |
| \#4CA64C | SDTM-395 | TI | Record | 0 | IECAT must not be null in TI; the inclusion/exclusion category is required for each criterion record. |
| \#4CA64C | SDTM-396 | TA | Record | 0 | ARMCD must not be null in TA; the arm code is required for every trial arm record. |
| \#4CA64C | SDTM-397 | TA | Record | 0 | TAETORD must not be null in TA; the planned element order within the arm is required. |
| \#4CA64C | SDTM-407 | PE | Variable | 0 | PESEQ in PE must be a numeric type variable. |
| \#4CA64C | SDTM-408 | IE | Variable | 0 | IESEQ in IE must be a numeric type variable. |
| \#4CA64C | SDTM-409 | PE | Variable | 0 | PEENDY in PE must be a numeric type variable. |
| \#4CA64C | SDTM-410 | IE | Variable | 0 | IESTDY in IE must be a numeric type variable. |
| \#4CA64C | SDTM-411 | IE | Variable | 0 | IEENDY in IE must be a numeric type variable. |
| \#4CA64C | SDTM-412 | DS | Variable | 0 | DSENDY in DS must be a numeric type variable. |
| \#4CA64C | SDTM-421 | SUPPDM | Record | 0 | DOMAIN must equal 'SUPPDM' in the SUPPDM dataset; each record must carry the correct supplemental qualifier domain code. |
| \#4CA64C | SDTM-422 | SUPPAE | Record | 0 | DOMAIN must equal 'SUPPAE' in the SUPPAE dataset; each record must carry the correct supplemental qualifier domain code. |
| \#4CA64C | SDTM-423 | SUPPLB | Record | 0 | DOMAIN must equal 'SUPPLB' in the SUPPLB dataset; each record must carry the correct supplemental qualifier domain code. |
| \#4CA64C | SDTM-424 | SUPPVS | Record | 0 | DOMAIN must equal 'SUPPVS' in the SUPPVS dataset; each record must carry the correct supplemental qualifier domain code. |
| \#4CA64C | SDTM-425 | SUPPEG | Record | 0 | DOMAIN must equal 'SUPPEG' in the SUPPEG dataset; each record must carry the correct supplemental qualifier domain code. |
| \#4CA64C | SDTM-426 | SUPPCM | Record | 0 | DOMAIN must equal 'SUPPCM' in the SUPPCM dataset; each record must carry the correct supplemental qualifier domain code. |
| \#4CA64C | SDTM-427 | SUPPDS | Record | 0 | DOMAIN must equal 'SUPPDS' in the SUPPDS dataset; each record must carry the correct supplemental qualifier domain code. |
| \#4CA64C | SDTM-428 | SUPPMH | Record | 0 | DOMAIN must equal 'SUPPMH' in the SUPPMH dataset; each record must carry the correct supplemental qualifier domain code. |
| \#4CA64C | SDTM-432 | EX | Record | 0 | EXROUTE in EX must be a valid ROUTE codelist term when present. |
| \#4CA64C | SDTM-433 | EX | Record | 0 | EXDOSFRM in EX must be a valid FRM codelist term when present. |
| \#4CA64C | SDTM-434 | TA | Record | 0 | EPOCH in TA must be a valid EPOCH codelist term when present. |
| \#AAAAAA | SDTM-049 | DEFINE | Variable | 0 | Variable labels in the submission must match the labels specified in Define-XML. |
| \#AAAAAA | SDTM-050 | DEFINE | Variable | 0 | Variable data types in the submission must match the types declared in Define-XML. |
| \#AAAAAA | SDTM-051 | DEFINE | Define | 0 | Each variable in the DM dataset must be declared in Define-XML. |
| \#AAAAAA | SDTM-052 | DEFINE | Define | 0 | Each variable in the AE dataset must be declared in Define-XML. |
| \#AAAAAA | SDTM-053 | DEFINE | Define | 0 | Variables declared Mandatory in Define-XML must not contain null values in DM. |
| \#AAAAAA | SDTM-054 | DEFINE | Define | 0 | Variables declared Mandatory in Define-XML must not contain null values in AE. |
| \#AAAAAA | SDTM-055 | DEFINE | Define | 0 | Variable data types in DM must match the types declared in Define-XML. |
| \#AAAAAA | SDTM-056 | DEFINE | Codelist | 0 | SEX values in DM must be from the codelist declared in Define-XML. |
| \#AAAAAA | SDTM-057 | DEFINE | Codelist | 0 | RACE values in DM must be from the codelist declared in Define-XML. |
| \#AAAAAA | SDTM-058 | DEFINE | Codelist | 0 | ETHNIC values in DM must be from the codelist declared in Define-XML. |
| \#AAAAAA | SDTM-059 | DEFINE | Codelist | 0 | AESEV values in AE must be from the codelist declared in Define-XML. |
| \#AAAAAA | SDTM-060 | DEFINE | Codelist | 0 | AEOUT values in AE must be from the codelist declared in Define-XML. |
| CT: sdtm-ct-2024-09-27 |  |  |  |  |  |

The ten failing rules are at the top, followed by the passing ones. The table has all 362 rules, so it's long. Displaying `report` itself in a notebook gives the same table.
