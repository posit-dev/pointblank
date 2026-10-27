# CDISC Submission Conformance

Regulatory submissions require that every dataset in a study package passes CDISC conformance rules. Pointblank provides a built-in conformance engine that checks your SDTM datasets against the full SDTMIG rule catalog with no external dependencies (no subprocess, no CLI, no network). For those preparing a final submission gate, Pointblank can also delegate to the external [CDISC CORE engine](https://github.com/cdisc-org/cdisc-rules-engine).

Two engines are available:

- **Built-in** (primary): Pointblank's own engine runs 362 SDTMIG 3.4 rules against your datasets in-process. Results are immediately renderable in a notebook. No installation beyond Pointblank is required.
- **CORE** (advanced): delegates to the open-source CDISC CORE CLI, which runs the authoritative CDISC-certified rule set. Requires the CORE executable to be installed separately.

## SDTMIG Conformance

The built-in engine covers the complete SDTMIG 3.4 rule catalog, spanning per-record value checks, variable metadata, dataset-level constraints, domain presence, and Define-XML cross-references. The sections below walk through the three output surfaces (the tabular conformance report, the record-level findings table, and the programmatic findings DataFrame) and describe the rule catalog, controlled terminology handling, and optional inputs such as custom CT packages and Define-XML.

### Quick Start

[`validate_sdtmig()`](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig) is the entry point for SDTMIG conformance. Pass a dictionary mapping domain names to DataFrames and it returns a [`ConformanceReport`](../../reference/ConformanceReport.md#pointblank.ConformanceReport) that renders as a color-coded summary table in Jupyter and Quarto notebooks:

``` python
import polars as pl
import pointblank as pb

dm = pl.DataFrame({
    "STUDYID": ["STUDY01"] * 4,
    "DOMAIN":  ["DM"] * 4,
    "USUBJID": ["STUDY01-01-001", "STUDY01-01-002", "STUDY01-01-003", "STUDY01-01-004"],
    "SUBJID":  ["001", "002", "003", "004"],
    "RFSTDTC": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"],
    "SEX":     ["M", "F", "M", "F"],
    "RACE":    ["WHITE", "ASIAN", "WHITE", "BLACK OR AFRICAN AMERICAN"],
    "ETHNIC":  ["NOT HISPANIC OR LATINO"] * 4,
    "ARMCD":   ["TRT", "PBO", "TRT", "PBO"],
    "ARM":     ["Treatment", "Placebo", "Treatment", "Placebo"],
    "COUNTRY": ["USA"] * 4,
    "AGE":     [45, 62, 38, 55],
    "AGEU":    ["YEARS"] * 4,
    "DMDTC":   ["2024-01-10"] * 4,
    "DMDY":    [1] * 4,
    "SITEID":  ["01"] * 4,
})

report = pb.validate_sdtmig({"DM": dm})
report
```

| CDISC Conformance FAIL |  |  |  |  |  |
|----|----|----|----|----|----|
| SDTMIG 3-4 · 346 passed · 4 failed · 12 n/a |  |  |  |  |  |
|  | Rule | Domain | Type | Issues | Description |
| \#FF3300 | SDTM-033 | DM | Variable | 1 | DM domain must contain RFSTDTC and RFENDTC (reference start/end dates). |
| \#FF3300 | SDTM-035 | DM | Variable | 1 | DM domain must contain treatment arm variables (ARMCD, ARM, ACTARMCD, ACTARM). |
| \#FF3300 | SDTM-121 | TS | Domain | 1 | TS domain must be present in every SDTM submission. |
| \#FF3300 | SDTM-122 | TA | Domain | 1 | TA domain must be present in every SDTM submission. |
| \#4CA64C | SDTM-001 | DM | Domain | 0 | DM (Demographics) domain is required in every SDTM submission. |
| \#4CA64C | SDTM-002 | DM | Record | 0 | STUDYID must not be null in any domain. |
| \#4CA64C | SDTM-003 | DM | Record | 0 | DOMAIN must not be null in any SDTM dataset. |
| \#4CA64C | SDTM-004 | DM | Record | 0 | USUBJID must not be null in any SDTM dataset. |
| \#4CA64C | SDTM-005 | DM | Dataset | 0 | STUDYID must be consistent (same value) across all records in a dataset. |
| \#4CA64C | SDTM-006 | DM | Dataset | 0 | DOMAIN must be consistent (same value) across all records in a dataset. |
| \#4CA64C | SDTM-007 | DM | Record | 0 | SEX in DM must use values from the CDISC controlled terminology codelist SEX. |
| \#4CA64C | SDTM-008 | DM | Record | 0 | RACE in DM must use values from the CDISC controlled terminology codelist RACE. |
| \#4CA64C | SDTM-009 | DM | Record | 0 | COUNTRY in DM must use ISO 3166 alpha-3 country codes (CDISC COUNTRY codelist). |
| \#4CA64C | SDTM-010 | DM | Record | 0 | DMDTC in DM must be in ISO 8601 extended datetime format. |
| \#4CA64C | SDTM-011 | DM | Record | 0 | RFSTDTC in DM must be in ISO 8601 extended datetime format when present. |
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
| \#4CA64C | SDTM-031 | DM | Variable | 0 | DM domain must contain all required Identifier and Topic variables. |
| \#4CA64C | SDTM-032 | DM | Variable | 0 | DM domain must contain required demographic variables (SEX, RACE, COUNTRY). |
| \#4CA64C | SDTM-034 | DM | Variable | 0 | DM domain must contain SITEID, AGE, and AGEU. |
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
| \#4CA64C | SDTM-195 | DM | Record | 0 | AGE must not be negative in the DM dataset. |
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

Domain keys are matched case-insensitively (`"dm"` and `"DM"` are equivalent). Polars, pandas, and any other narwhals-compatible DataFrame are accepted. The report object itself renders the tabular summary in a notebook; read on to learn how to drill into failures and access findings programmatically.

### The Conformance Report

The tabular report shown above is produced by [`get_tabular_report()`](../../reference/ConformanceReport.md#pointblank.ConformanceReport.get_tabular_report). Each row is one rule from the SDTMIG 3.4 catalog; the colored bar on the left edge encodes its status:

| Color | Status | Meaning |
|----|----|----|
| Green | `pass` | The rule was evaluated and no violations were found |
| Red | `fail` | One or more records or datasets violated the rule |
| Amber | [error](../../reference/Validate.error.md#pointblank.Validate.error) | The rule could not be evaluated (unexpected data state) |
| Grey | `not_applicable` | The rule requires a dataset or variable that was not supplied |
| Grey | `not_supported` | The rule type is not yet implemented in the built-in engine |

Rules are sorted by severity: failures first, then errors, then passes, then not-applicable.

The header shows a `PASS` or `FAIL` badge alongside the standard, version, and a full status breakdown (e.g., `SDTMIG 3-4 · 410 passed · 4 failed · 12 n/a`).

Call `get_tabular_report()` to get the `GT` object directly if you need to embed it in a report pipeline:

``` python
gt = report.get_tabular_report()
gt
```

The `GT` object can be passed to any Great Tables export method (for example, `gt.as_raw_html()` to embed the table in a custom HTML report, or `gt.gtsave("report.png")` to render it as an image). When failures are present, use the findings surfaces below to investigate the specific records that triggered each rule.

### Findings Drill-Down

When rules fail, the tabular report shows how many records violated each rule but not which ones. [`get_findings_table()`](../../reference/ConformanceReport.md#pointblank.ConformanceReport.get_findings_table) provides that drill-down: one row per failing record, showing the subject, the specific column that violated the rule, the offending value, and the 1-based row number in the source dataset.

To see it in action, introduce a few deliberate violations:

``` python
dm_with_issues = pl.DataFrame({
    "STUDYID": ["STUDY01"] * 4,
    "DOMAIN":  ["DM"] * 4,
    "USUBJID": ["STUDY01-01-001", "STUDY01-01-002", "STUDY01-01-003", "STUDY01-01-004"],
    "SUBJID":  ["001", "002", "003", "004"],
    "RFSTDTC": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18"],
    "SEX":     ["M", "F", "UNKNOWN", "F"],      # "UNKNOWN" is not in the SEX codelist
    "RACE":    ["WHITE", "ASIAN", "WHITE", "BLACK OR AFRICAN AMERICAN"],
    "ETHNIC":  ["NOT HISPANIC OR LATINO"] * 4,
    "ARMCD":   ["TRT", "PBO", "TRT", "PBO"],
    "ARM":     ["Treatment", "Placebo", "Treatment", "Placebo"],
    "COUNTRY": ["USA"] * 4,
    "AGE":     [45, -5, 38, 55],               # -5 violates AGE >= 0
    "AGEU":    ["YEARS"] * 4,
    "DMDTC":   ["2024-01-10", "not-a-date", "2024-01-10", "2024-01-10"],  # invalid ISO date
    "DMDY":    [1] * 4,
    "SITEID":  ["01"] * 4,
})

report_with_issues = pb.validate_sdtmig({"DM": dm_with_issues})
report_with_issues.get_findings_table()
```

    /opt/hostedtoolcache/Python/3.11.17/x64/lib/python3.11/site-packages/great_tables/_render_checks.py:37: RenderWarning: Rendering table with .cols_width() in Quarto may result in unexpected behavior. This is because Quarto performs custom table processing. Either use all percentage widths, or set .tab_options(quarto_disable_processing=True) to disable Quarto table processing.
      warnings.warn(

| Findings Report for CDISC Conformance |  |  |  |  |  |  |  |
|----|----|----|----|----|----|----|----|
| SDTMIG 3-4 · 7 failed (343 passed) · 12 n/a |  |  |  |  |  |  |  |
|  | SDTM Rule Definition |  |  | Finding |  |  |  |
|  | Rule | Domain | Description | USUBJID | Column | Row | Value |
| \#FF3300 | SDTM-007 | DM | SEX in DM must use values from the CDISC controlled terminology codelist SEX. | STUDY01-01-003 | SEX | 3 | UNKNOWN |
| \#FF3300 | SDTM-010 | DM | DMDTC in DM must be in ISO 8601 extended datetime format. | STUDY01-01-002 | DMDTC | 2 | not-a-date |
| \#FF3300 | SDTM-195 | DM | AGE must not be negative in the DM dataset. | STUDY01-01-002 | AGE | 2 | -5 |

The findings table groups its columns into two spanners:

- **Rule**: `Domain` and `Description` identify which rule fired and where.
- **Finding**: `USUBJID` identifies the subject; `Column` and `Value` show exactly what was wrong; `Row` is the 1-based row number in the source dataset for quick lookup.

At most 100 findings per rule are shown in the table. The true total for each rule is always visible in `get_tabular_report()`. To export findings to a spreadsheet or join them back to your source data, use [findings_df()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.findings_df) instead.

### Programmatic Access with findings_df()

[`findings_df()`](../../reference/ConformanceReport.md#pointblank.ConformanceReport.findings_df) returns the same findings as a Polars DataFrame, which is better suited for filtering, grouping, exporting to CSV, or joining back to the source data:

``` python
df = report_with_issues.findings_df()
df
```

shape: (3, 7)

| rule_id | dataset | row_index | usubjid | checked_column | checked_value | description |
|----|----|----|----|----|----|----|
| str | str | i64 | str | str | str | str |
| "SDTM-007" | "DM" | 2 | "STUDY01-01-003" | "SEX" | "UNKNOWN" | "SEX in DM must use values from… |
| "SDTM-010" | "DM" | 1 | "STUDY01-01-002" | "DMDTC" | "not-a-date" | "DMDTC in DM must be in ISO 860… |
| "SDTM-195" | "DM" | 1 | "STUDY01-01-002" | "AGE" | "-5" | "AGE must not be negative in th… |

The DataFrame schema is:

| Column | Description |
|----|----|
| `rule_id` | CDISC CORE rule identifier (e.g., `"SDTM-007"`) |
| `dataset` | SDTM domain the failing record belongs to |
| `row_index` | 0-based row position in the source dataset |
| `usubjid` | Unique Subject Identifier (`USUBJID`) for the failing record |
| `checked_column` | The variable that violated the rule (e.g., `"SEX"`) |
| `checked_value` | The actual value found in `checked_column` |
| `description` | Human-readable rule description |

``` python
import polars as pl

# Filter to a specific rule
df.filter(pl.col("rule_id") == "SDTM-007")
```

shape: (1, 7)

| rule_id | dataset | row_index | usubjid | checked_column | checked_value | description |
|----|----|----|----|----|----|----|
| str | str | i64 | str | str | str | str |
| "SDTM-007" | "DM" | 2 | "STUDY01-01-003" | "SEX" | "UNKNOWN" | "SEX in DM must use values from… |

``` python
# Group by rule to count violations
df.group_by("rule_id").agg(pl.len().alias("n_violations")).sort("n_violations", descending=True)
```

shape: (3, 2)

| rule_id    | n_violations |
|------------|--------------|
| str        | u32          |
| "SDTM-007" | 1            |
| "SDTM-010" | 1            |
| "SDTM-195" | 1            |

The DataFrame is empty (with the same schema) when all rules pass, so it is safe to call unconditionally in a script. Use it to triage failures before running the more time-intensive CDISC CORE engine, or to generate a custom summary report tailored to your team's workflow.

### What the Rule Catalog Covers

The bundled SDTMIG 3.4 catalog contains 362 rules across seven types:

| Type | Description | Examples |
|----|----|----|
| `RECORD_CHECK` | Per-row value checks | Codelist membership, ISO 8601 dates, numeric ranges |
| `VARIABLE_METADATA_CHECK` | Variable presence and column ordering | USUBJID must precede domain-specific variables |
| `DATASET_METADATA_CHECK` | Dataset-level attributes | Required sort key order |
| `DATASET_CONTENTS_CHECK` | Dataset-level value constraints | All rows in a domain must share the same STUDYID |
| `DOMAIN_PRESENCE_CHECK` | Required or prohibited domain presence | DM must be present |
| `DEFINE_ITEM_METADATA_CHECK` | Variable declarations against Define-XML | Activated when `define_xml` is supplied |
| `DEFINE_CODELIST_CHECK` | Codelist values against Define-XML | Activated when `define_xml` is supplied |

Only `RECORD_CHECK` and `DATASET_CONTENTS_CHECK` rules produce row-level findings accessible via [findings_df()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.findings_df) and [get_findings_table()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.get_findings_table). The other types report a violation count in `get_tabular_report()` but do not have individual record detail.

Rules that require a domain or variable not present in your datasets are automatically marked `not_applicable` (they are not counted as failures). For example, a rule that checks `AESTDTC` in the AE domain is `not_applicable` when no AE dataset is supplied. Adding more domains to the dictionary passed to [validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig) will convert more rules from `not_applicable` to executable, giving a more complete conformance picture.

### Controlled Terminology

Codelist checks use the bundled CT package `sdtm-ct-2024-09-27`. Two important behaviors:

- **Case-insensitive matching**: a value of `"beats/min"` matches the codelist term `"BEATS/MIN"`. This avoids false positives for studies that applied mixed-case CT values.
- **SAS/XPT empty strings treated as null**: SAS Transport files encode character missing values as `""` (empty string) rather than `None`. The engine recognizes this and skips codelist and format checks for such cells, preventing the large volumes of false positives that occur with a naive string comparison.

Both behaviors apply automatically with no configuration. If you need to pin a different CT version or supply additional packages, see the next section.

### Supply a Custom CT Package

By default the most recent bundled CT package is used. Pass `ct_packages` to pin a specific version or supply additional packages:

``` python
report = pb.validate_sdtmig(
    {"DM": dm},
    ct_packages=["sdtm-ct-2024-09-27"],
)
```

Pinning the CT version is useful when your study was locked against a specific CDISC CT release and you want the conformance check to reflect that snapshot rather than the latest terms.

### Activating Define-XML Rules

Pass a path to `define.xml` to activate the Define-XML-aware rule types:

``` python
report = pb.validate_sdtmig(
    {"DM": dm, "AE": ae},
    define_xml="path/to/define.xml",
)
```

Without `define_xml`, the 76 `DEFINE_ITEM_METADATA_CHECK` and `DEFINE_CODELIST_CHECK` rules are marked `not_applicable`. Supplying the file allows the engine to verify that every variable present in your datasets is declared in Define-XML with the correct type and, where applicable, a valid codelist.

Define-XML also tells the CT checks about sponsor extensions. Many CDISC codelists are extensible (EPOCH, FREQ, ROUTE, and UNIT among them): a sponsor may add terms, provided Define-XML declares them with `def:ExtendedValue="Yes"` on the codelist item. When `define_xml` is supplied, such declared terms pass the codelist checks for extensible codelists. For example, a crossover study can use the epochs `TREATMENT 1` and `TREATMENT 2`. Without Define-XML, or for a codelist that isn't extensible (such as SEX), terms outside the codelist are still reported.

### Loading XPT Files

For datasets stored as SAS Transport (XPT) files, use `pyreadstat` to load them before passing to [validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig). Install it with `pip install pyreadstat`:

``` python
import pyreadstat
import polars as pl
import pointblank as pb

def load_xpt(path: str) -> pl.DataFrame:
    df, _ = pyreadstat.read_xport(path)
    return pl.from_pandas(df)

report = pb.validate_sdtmig(
    {
        "DM": load_xpt("sdtm/dm.xpt"),
        "AE": load_xpt("sdtm/ae.xpt"),
        "LB": load_xpt("sdtm/lb.xpt"),
        "VS": load_xpt("sdtm/vs.xpt"),
    },
    study_id="STUDY01",
)
```

Pointblank handles the SAS empty-string convention automatically, so XPT data does not require any preprocessing before being passed to [validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig). The same built-in handling applies whether data arrives from XPT files, in-memory DataFrames, or any other narwhals-compatible source.

## Prerequisites

[validate_sdtmig()](../../reference/validate_sdtmig.md#pointblank.validate_sdtmig) and the [ConformanceReport](../../reference/ConformanceReport.md#pointblank.ConformanceReport) methods require no additional dependencies beyond Pointblank itself. The optional extras are:

- **Reading XPT files** with `pyreadstat`: `pip install pyreadstat`
- **Writing XPT files** for CORE (automatic, handled internally): `pip install pyreadstat`
- **Excel export** with [to_excel()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.to_excel): `pip install openpyxl`

``` bash
pip install pointblank[cdisc]    # adds pyreadstat
pip install pointblank[excel]    # adds openpyxl
```

The CDISC CORE engine is not a Python dependency of Pointblank and must be installed separately when needed. See [Installing the CDISC CORE Engine](#installing-the-cdisc-core-engine) below.

## CDISC CORE Engine (Advanced)

The CDISC CORE engine runs the full authoritative conformance rule set and produces reports accepted by FDA and PMDA review tools. Use it as the final pre-submission gate after the built-in engine gives a clean result.

### Installing the CDISC CORE Engine

CORE can be obtained in three ways depending on your environment. The standalone executable is easiest for local use, Docker is preferred in containerized CI pipelines, and the repo checkout is useful when you need a specific unreleased version or want to inspect the rule definitions directly.

**Option 1: Standalone executable.** Download the pre-built binary from the [CDISC CORE releases page](https://github.com/cdisc-org/cdisc-rules-engine/releases). Place it somewhere on your `PATH` under the name `core`:

``` bash
chmod +x core
sudo mv core /usr/local/bin/
core --version
```

Once the binary is on `PATH`, Pointblank will discover it automatically with no additional configuration required.

**Option 2: Docker.** CDISC publishes an official Docker image that includes the engine and its full rules cache:

``` bash
docker pull cdisc/cdisc-rules-engine:latest

docker run --rm \
    -v /path/to/study/data:/data \
    cdisc/cdisc-rules-engine:latest \
    validate -s sdtmig -v 3-4 -d /data -of JSON -o /data/report
```

When running via Docker you will typically invoke CORE directly rather than through Pointblank's subprocess wrapper. The resulting JSON output can be loaded with `parse_core_report()` and wrapped in a [ConformanceReport](../../reference/ConformanceReport.md#pointblank.ConformanceReport) for further analysis.

**Option 3: Repo checkout.**

``` bash
git clone https://github.com/cdisc-org/cdisc-rules-engine.git
cd cdisc-rules-engine
pip install -r requirements.txt
python core.py --version
```

When using a repo checkout, pass the repo root as `core_cwd` so Pointblank sets the subprocess working directory correctly.

### Telling Pointblank Where to Find CORE

Pointblank discovers CORE through three mechanisms, tried in order:

1.  An explicit `core=` argument to [validate_cdisc_submission()](../../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission) or `validate_conformance(engine="core")`.
2.  The `POINTBLANK_CDISC_CORE` environment variable (e.g., `export POINTBLANK_CDISC_CORE="python /path/to/core.py"`).
3.  A `core` or `cdisc-rules-engine` executable on `PATH`.

The environment variable approach is convenient for CI systems where the CORE path differs between machines, while the explicit argument is useful in notebooks where you want the path to be self-documenting.

### Running CORE

[validate_cdisc_submission()](../../reference/validate_cdisc_submission.md#pointblank.validate_cdisc_submission) is the simplest way to run a CORE validation. It accepts an in-memory dictionary of DataFrames, a folder path, or an existing [SubmissionPackage](../../reference/SubmissionPackage.md#pointblank.SubmissionPackage):

``` python
report = pb.validate_cdisc_submission(
    {"DM": dm, "AE": ae},
    standard="sdtmig",
    version="3.4",
    agency="FDA",
)
```

For a folder of XPT files pass the path directly. Pointblank skips the materialization step and passes the folder straight to CORE:

``` python
report = pb.validate_cdisc_submission(
    "path/to/sdtm/",
    standard="sdtmig",
    version="3.4",
    agency="FDA",
)
```

For repo-checkout CORE, pass the command prefix and working directory:

``` python
report = pb.validate_cdisc_submission(
    {"DM": dm},
    standard="sdtmig",
    version="3.4",
    core=["python", "/path/to/cdisc-rules-engine/core.py"],
    core_cwd="/path/to/cdisc-rules-engine",
    cache="/path/to/cdisc-rules-engine/resources/cache",
)
```

In all cases the return value is a [ConformanceReport](../../reference/ConformanceReport.md#pointblank.ConformanceReport) with `is_core=True`. The sections below describe the accessor methods available on a CORE report.

### Working with a CORE ConformanceReport

The examples below use a captured real CORE report so that the code runs without requiring CORE to be installed in the docs environment. The structure is identical to what a live run produces.

``` python
import json
from pathlib import Path
from pointblank.metadata import parse_core_report, ConformanceReport

_fixtures = Path(pb.__file__).parent.parent / "tests" / "metadata_fixtures" / "cdisc_core"
raw = json.loads((_fixtures / "core_report_full.json").read_text())

report = ConformanceReport.from_core_report(raw, agency="FDA")
print(report)
```

    ConformanceReport (CORE)
      Agency: FDA
      SDTMIG V3.4 -- CORE 0.16.0
      430 rules (EXECUTION ERROR=2, ISSUE REPORTED=6, SKIPPED=344, SUCCESS=78)
      8 issues -- FAIL

**Overall result.** `all_passed()` returns `True` when no rules reported issues; [is_core](../../reference/ConformanceReport.md#pointblank.ConformanceReport.is_core) confirms the report originated from the CORE engine rather than the built-in engine:

``` python
print("All passed:", report.all_passed())
print("Is CORE report:", report.is_core)
```

    All passed: False
    Is CORE report: True

**Summary dictionary.** `summary()` returns run provenance and rule counts as a plain dictionary, useful for logging, assertions in CI scripts, or building a custom status page:

``` python
s = report.summary()

print(f"Standard:        {s['standard']} {s['version']}")
print(f"Engine:          {s['engine_version']}")
print(f"Rules evaluated: {s['n_rules']}")
print(f"Total issues:    {s['n_issues']}")
print()

print("Rule status breakdown:")
for status, count in sorted(s["status_counts"].items()):
    print(f"  {status:20s}: {count}")
```

    Standard:        SDTMIG V3.4
    Engine:          0.16.0
    Rules evaluated: 430
    Total issues:    8

    Rule status breakdown:
      EXECUTION ERROR     : 2
      ISSUE REPORTED      : 6
      SKIPPED             : 344
      SUCCESS             : 78

The [status_counts](../../reference/Comparison.status_counts.md#pointblank.Comparison.status_counts) dictionary enumerates every status bucket CORE reported. Use it to build pass/fail thresholds or trend charts across multiple submission runs.

**Issues.** [issues()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.issues) returns one record per (dataset, rule) pair that reported at least one issue. Pass a `status` filter to separate conformance violations from execution errors:

``` python
issues = report.issues()
print(f"Issue entries: {len(issues)}")

for issue in issues[:3]:
    print()
    print(f"  Dataset:  {issue['dataset']}")
    print(f"  Rule:     {issue['rule_id']}")
    print(f"  Message:  {issue['message']}")
    print(f"  Issues:   {issue['issues']}")
```

    Issue entries: 8

      Dataset:  STUDY
      Rule:     CORE-000581
      Message:  DM dataset is missing.
      Issues:   1

      Dataset:  TEST_DATASET
      Rule:     CORE-000357
      Message:  Supplemental qualifier dataset associated with a split dataset is greater than 8 characters in length
      Issues:   1

      Dataset:  TEST_DATASET
      Rule:     CORE-000510
      Message:  Split dataset name is not 3 or 4 characters in length
      Issues:   1

``` python
from pointblank.metadata._cdisc_core import STATUS_ISSUE, STATUS_ERROR

reported_issues = report.issues(status=STATUS_ISSUE)
print(f"Rules with issues: {len(reported_issues)}")

exec_errors = report.issues(status=STATUS_ERROR)
print(f"Execution errors:  {len(exec_errors)}")
```

    Rules with issues: 6
    Execution errors:  2

Execution errors (`STATUS_ERROR`) indicate that CORE could not evaluate a rule, typically because a required variable or dataset was absent or had an unexpected format. They are distinct from conformance issues (`STATUS_ISSUE`) where the rule ran successfully but found a violation.

**Row-level findings.** [findings()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.findings) returns the row-level detail from CORE's `Issue_Details` section, giving the exact record that triggered each rule:

``` python
findings = report.findings()
print(f"Row-level findings: {len(findings)}")

f = findings[0]
print()
print(f"Rule:          {f.rule_id}")
print(f"Dataset:       {f.dataset}")
print(f"Row:           {f.row}")
print(f"USUBJID:       {f.usubjid}")
print(f"Variables:     {f.variables}")
print(f"Values:        {f.values}")
```

    Row-level findings: 8

    Rule:          CORE-000357
    Dataset:       TEST_DATASET
    Row:           1
    USUBJID:       None
    Variables:     ['dataset_name']
    Values:        ['TEST_DATASET']

Each finding carries the `USUBJID`, the affected variables and their values, and the 1-based row number in the source dataset, matching the information shown in the built-in engine's findings table.

**Per-rule run results.** [rules()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.rules) returns one `CoreRuleResult` per rule with its run status:

``` python
from pointblank.metadata._cdisc_core import STATUS_SUCCESS, STATUS_SKIPPED

all_rules = report.rules()
print(f"Total rules:  {len(all_rules)}")
print(f"Successful:   {len(report.rules(status=STATUS_SUCCESS))}")
print(f"Skipped:      {len(report.rules(status=STATUS_SKIPPED))}")

failing = [r for r in all_rules if r.is_failing]
print(f"Failing:      {len(failing)}")
for r in failing:
    print(f"  {r.rule_id:15s} [{r.status}]  {(r.message or '')[:60]}")
```

    Total rules:  430
    Successful:   78
    Skipped:      344
    Failing:      8
      CORE-000357     [ISSUE REPORTED]  Supplemental qualifier dataset associated with a split datas
      CORE-000510     [ISSUE REPORTED]  Split dataset name is not 3 or 4 characters in length
      CORE-000539     [ISSUE REPORTED]  Split dataset is present but the two-Letter parent domain is
      CORE-000581     [ISSUE REPORTED]  DM dataset is missing.
      CORE-000598     [ISSUE REPORTED]  Dataset name does not begin with DOMAIN value
      CORE-000778     [ISSUE REPORTED]  Associated Persons non-supplemental qualifier dataset associ
      CORE-000929     [EXECUTION ERROR]  DOMAIN Code is not a published DOMAIN Code in CDISC Controll
      CORE-001081     [EXECUTION ERROR]  Variable role specified in the define-xml does not match the

Rules marked skipped are not failures. CORE skips rules when the dataset or variable they check is absent from the submission. A skipped rule is equivalent to `not_applicable` in the built-in engine and should not be treated as a conformance problem.

## Exporting Reports

Both built-in and CORE reports can be exported for archiving, sharing with a biometrics team, or loading into a review tool. Two formats are supported: JSON for round-trippable machine-readable output, and Excel for spreadsheet-based review workflows.

### JSON

`to_json()` saves the report as a JSON file. For CORE reports the structure mirrors CORE's native output and is parseable by `parse_core_report()`:

``` python
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    dest = report.to_json(Path(tmp) / "conformance_report.json")
    print(f"Written to: {dest.name}")
    print(f"File size:  {dest.stat().st_size:,} bytes")

    reloaded = json.loads(dest.read_text())
    reparsed = parse_core_report(reloaded)
    print(f"Rules after round-trip: {len(reparsed.rules)}")
```

    Written to: conformance_report.json
    File size:  113,079 bytes
    Rules after round-trip: 430

The round-trip fidelity means a JSON file written by Pointblank can be re-loaded later without re-running CORE, which is useful for archiving the state of a submission at a specific point in time.

### Excel

[to_excel()](../../reference/ConformanceReport.md#pointblank.ConformanceReport.to_excel) writes the report as a workbook. Requires `openpyxl`:

``` python
dest = report.to_excel("conformance_report.xlsx")
```

For CORE reports the workbook has four sheets: `Issue_Summary`, `Issue_Details`, `Rules_Report`, and `Conformance_Details`.

## Using Both Engines Together

The built-in engine provides fast feedback during development; CORE is the final gate. The typical workflow is to iterate with the built-in engine until it reports a clean result, then run CORE once as a pre-submission check:

``` python
# Step 1: fast iteration with the built-in engine
report = pb.validate_sdtmig({"DM": dm, "AE": ae}, study_id="STUDY01")

if not report.all_passed():
    print("Fix rule violations before running CORE:")
    df = report.findings_df()
    print(df.group_by("rule_id").agg(pl.len().alias("n")).sort("n", descending=True))
    raise SystemExit(1)

# Step 2: final gate with CDISC CORE
core_report = pb.validate_cdisc_submission(
    {"DM": dm, "AE": ae},
    standard="sdtmig",
    version="3.4",
    agency="FDA",
)

if not core_report.all_passed():
    failing = [r for r in core_report.rules() if r.is_failing]
    print(f"CORE found {len(failing)} failing rules.")
    core_report.to_json("core_report.json")
    raise SystemExit(1)

print("Submission passed all conformance checks.")
```

This pattern catches the majority of conformance problems early (when fixing them is cheap) and reserves the slower CORE invocation for the final verification step. Both reports can be exported and archived alongside the submission package.

## Running the Integration Tests

Pointblank ships integration tests for the full CORE pipeline. They are skipped automatically when CORE is not discoverable:

``` bash
# Run when CORE is on PATH:
pytest -m cdisc_core

# With a repo-checkout CORE:
export POINTBLANK_CDISC_CORE="python /path/to/core.py"
export POINTBLANK_CDISC_CORE_CWD="/path/to/cdisc-rules-engine"
pytest -m cdisc_core
```

The integration tests validate the full subprocess pipeline end-to-end, including XPT materialization, CORE invocation, JSON parsing, and report construction. Running them against your local CORE installation is a good sanity check after upgrading either Pointblank or the CORE engine.
