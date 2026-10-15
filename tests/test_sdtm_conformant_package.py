"""
A conformant SDTMIG 3.4 package must pass the native conformance catalog.

The package is small and hand-checked against the SDTMIG 3.4 domain specifications and CDISC CT
2024-09-27. It deliberately includes things that are conformant but that rules used to reject
(BUGS.md #29, #30, #33-#36, #42-#44): AE without VISITNUM, TI without IEINCLCR, LB with character
LBORNRLO/LBORNRHI and no LBSPEC, TV without ARMCD/ARM (every arm has the same visit schedule), a
VS record with a null VISITNUM, AEACN values DRUG WITHDRAWN/DRUG INTERRUPTED, the FREQ term QD,
and the EPOCH term CONTINUATION TREATMENT.
"""

from __future__ import annotations

import polars as pl
import pytest

import pointblank as pb

STUDY = "S001"
SUBJ = ["S001-001", "S001-002"]


def conformant_sdtm_package() -> dict[str, pl.DataFrame]:
    """A small SDTMIG 3.4 package that conforms to the IG (hand-checked)."""
    dm = pl.DataFrame(
        {
            "STUDYID": [STUDY] * 2,
            "DOMAIN": ["DM"] * 2,
            "USUBJID": SUBJ,
            "SUBJID": ["001", "002"],
            "RFSTDTC": ["2024-01-08", "2024-01-09"],
            "RFENDTC": ["2024-03-04", "2024-03-05"],
            "RFXSTDTC": ["2024-01-08", "2024-01-09"],
            "RFXENDTC": ["2024-03-04", "2024-03-05"],
            "RFICDTC": ["2024-01-02", "2024-01-03"],
            "RFPENDTC": ["2024-03-18", "2024-03-19"],
            "SITEID": ["001", "001"],
            "AGE": [45, 32],
            "AGEU": ["YEARS"] * 2,
            "SEX": ["M", "F"],
            "RACE": ["WHITE", "ASIAN"],
            "ETHNIC": ["NOT HISPANIC OR LATINO"] * 2,
            "ARMCD": ["A", "B"],
            "ARM": ["Drug A", "Placebo"],
            "ACTARMCD": ["A", "B"],
            "ACTARM": ["Drug A", "Placebo"],
            "COUNTRY": ["USA", "GBR"],
            "DMDTC": ["2024-01-02", "2024-01-03"],
            "DMDY": [-6, -6],
        }
    )
    # AE: no VISITNUM (Permissible in Events domains); AEACN uses C66767 terms
    ae = pl.DataFrame(
        {
            "STUDYID": [STUDY] * 2,
            "DOMAIN": ["AE"] * 2,
            "USUBJID": SUBJ,
            "AESEQ": [1, 1],
            "AETERM": ["HEADACHE", "NAUSEA"],
            "AEDECOD": ["Headache", "Nausea"],
            "AEBODSYS": ["Nervous system disorders", "Gastrointestinal disorders"],
            "AESEV": ["MILD", "MODERATE"],
            "AESER": ["N", "N"],
            "AEACN": ["DRUG WITHDRAWN", "DRUG INTERRUPTED"],
            "AEREL": ["RELATED", "NOT RELATED"],
            "AEOUT": ["RECOVERED/RESOLVED", "RECOVERED/RESOLVED"],
            "EPOCH": ["TREATMENT", "TREATMENT"],
            "AESTDTC": ["2024-01-15", "2024-01-20"],
            "AEENDTC": ["2024-01-17", "2024-01-22"],
            "AESTDY": [8, 12],
            "AEENDY": [10, 14],
        }
    )
    ex = pl.DataFrame(
        {
            "STUDYID": [STUDY] * 2,
            "DOMAIN": ["EX"] * 2,
            "USUBJID": SUBJ,
            "EXSEQ": [1, 1],
            "EXTRT": ["DRUG A", "PLACEBO"],
            "EXDOSE": [10.0, 0.0],
            "EXDOSU": ["mg", "mg"],
            "EXDOSFRM": ["TABLET", "TABLET"],
            "EXDOSFRQ": ["QD", "QD"],
            "EXROUTE": ["ORAL", "ORAL"],
            "EPOCH": ["TREATMENT", "CONTINUATION TREATMENT"],
            "EXSTDTC": ["2024-01-08", "2024-01-09"],
            "EXENDTC": ["2024-03-04", "2024-03-05"],
            "EXSTDY": [1, 1],
            "EXENDY": [57, 57],
        }
    )
    # LB: original-unit reference ranges are Char (LBORNRLO/LBORNRHI); standard ones are Num
    lb = pl.DataFrame(
        {
            "STUDYID": [STUDY] * 2,
            "DOMAIN": ["LB"] * 2,
            "USUBJID": SUBJ,
            "LBSEQ": [1, 1],
            "LBTESTCD": ["HGB", "HGB"],
            "LBTEST": ["Hemoglobin", "Hemoglobin"],
            "LBCAT": ["HEMATOLOGY"] * 2,
            "LBORRES": ["14.1", "12.9"],
            "LBORRESU": ["g/dL"] * 2,
            "LBORNRLO": ["13.5", "12.0"],
            "LBORNRHI": ["17.5", "15.5"],
            "LBSTRESC": ["14.1", "12.9"],
            "LBSTRESN": [14.1, 12.9],
            "LBSTRESU": ["g/dL"] * 2,
            "LBSTNRLO": [13.5, 12.0],
            "LBSTNRHI": [17.5, 15.5],
            "LBNRIND": ["NORMAL", "NORMAL"],
            "VISITNUM": [1.0, 1.0],
            "VISIT": ["SCREENING"] * 2,
            "EPOCH": ["SCREENING"] * 2,
            "LBDTC": ["2024-01-02", "2024-01-03"],
            "LBDY": [-6, -6],
        }
    )
    # VS: VISITNUM is Expected, so it's null for an unscheduled record
    vs = pl.DataFrame(
        {
            "STUDYID": [STUDY] * 3,
            "DOMAIN": ["VS"] * 3,
            "USUBJID": [SUBJ[0], SUBJ[0], SUBJ[1]],
            "VSSEQ": [1, 2, 1],
            "VSTESTCD": ["SYSBP", "SYSBP", "SYSBP"],
            "VSTEST": ["Systolic Blood Pressure"] * 3,
            "VSORRES": ["120", "118", "125"],
            "VSORRESU": ["mmHg"] * 3,
            "VSSTRESC": ["120", "118", "125"],
            "VSSTRESN": [120.0, 118.0, 125.0],
            "VSSTRESU": ["mmHg"] * 3,
            "VISITNUM": [1.0, None, 1.0],
            "VISIT": ["SCREENING", None, "SCREENING"],
            "EPOCH": ["SCREENING", "TREATMENT", "SCREENING"],
            "VSDTC": ["2024-01-02", "2024-01-20", "2024-01-03"],
            "VSDY": [-6, 13, -6],
        }
    )
    ta = pl.DataFrame(
        {
            "STUDYID": [STUDY] * 4,
            "DOMAIN": ["TA"] * 4,
            "ARMCD": ["A", "A", "B", "B"],
            "ARM": ["Drug A", "Drug A", "Placebo", "Placebo"],
            "TAETORD": [1, 2, 1, 2],
            "ETCD": ["SCRN", "DRUGA", "SCRN", "PBO"],
            "ELEMENT": ["Screening", "Drug A", "Screening", "Placebo"],
            "EPOCH": ["SCREENING", "TREATMENT", "SCREENING", "TREATMENT"],
        }
    )
    # TV: every arm has the same visit schedule, so ARMCD and ARM are null
    tv = pl.DataFrame(
        {
            "STUDYID": [STUDY] * 2,
            "DOMAIN": ["TV"] * 2,
            "VISITNUM": [1.0, 2.0],
            "VISIT": ["SCREENING", "WEEK 8"],
            "VISITDY": [-7, 56],
            "ARMCD": pl.Series([None, None], dtype=pl.String),
            "ARM": pl.Series([None, None], dtype=pl.String),
            "TVSTRL": ["7 days before first dose", "56 days after first dose"],
        }
    )
    # TI: SDTMIG 3.4 TI variables (no IEINCLCR)
    ti = pl.DataFrame(
        {
            "STUDYID": [STUDY] * 2,
            "DOMAIN": ["TI"] * 2,
            "IETESTCD": ["INCL01", "EXCL01"],
            "IETEST": ["Age 18 to 75 years", "Prior exposure to Drug A"],
            "IECAT": ["INCLUSION", "EXCLUSION"],
            "TIVERS": ["1", "1"],
        }
    )
    ts = pl.DataFrame(
        {
            "STUDYID": [STUDY] * 2,
            "DOMAIN": ["TS"] * 2,
            "TSSEQ": [1, 1],
            "TSPARMCD": ["PLANSUB", "TITLE"],
            "TSPARM": ["Planned Number of Subjects", "Trial Title"],
            "TSVAL": ["2", "A Study of Drug A"],
        }
    )
    return {
        "DM": dm,
        "AE": ae,
        "EX": ex,
        "LB": lb,
        "VS": vs,
        "TA": ta,
        "TV": tv,
        "TI": ti,
        "TS": ts,
    }


@pytest.fixture(scope="module")
def report():
    return pb.validate_sdtmig(conformant_sdtm_package())


def _rule(report, rule_id):
    return next(r for r in report.native_result.rule_results if r.rule_id == rule_id)


def test_conformant_package_fails_no_rules(report):
    failed = [
        f"{r.rule_id} ({r.dataset}): {r.description}"
        for r in report.native_result.rule_results
        if r.status == "fail"
    ]
    assert failed == []

    # The rules for every domain in the package ran
    ran = {r.dataset for r in report.native_result.rule_results if r.status == "pass"}
    assert set(conformant_sdtm_package()) <= ran


def test_incorrect_rules_are_not_in_the_catalog(report):
    rule_ids = {r.rule_id for r in report.native_result.rule_results}
    removed = {
        "SDTM-134",  # IEINCLCR (not a TI variable) codelist check
        "SDTM-162",  # VISITNUM required in AE (Permissible)
        "SDTM-170",  # LBORNRLO numeric (Char in SDTMIG; LBSTNRLO is checked by SDTM-168)
        "SDTM-171",  # LBORNRHI numeric (Char in SDTMIG; LBSTNRHI is checked by SDTM-169)
        "SDTM-240",  # IEINCLCR required in TI
        "SDTM-326",  # ARM not null in TV (null when every arm has the same schedule)
        "SDTM-327",  # ARMCD not null in TV (same)
        "SDTM-335",  # VISITNUM not null in VS (Expected)
        "SDTM-371",  # LBSPEC required in LB (Permissible)
        "SDTM-435",  # duplicate of SDTM-066
        "SDTM-087",  # AETOXGR has no CDISC codelist (the grading scale is the sponsor's)
        "SDTM-135",  # AEREL has no CDISC codelist (SDTMIG only gives ICH examples)
    }
    assert not removed & rule_ids


@pytest.mark.parametrize(
    "rule_id, domain, column, valid, invalid",
    [
        # Action Taken with Study Treatment (C66767)
        ("SDTM-089", "AE", "AEACN", "DRUG WITHDRAWN", "DISCONTINUED"),
        # Sex (C66731)
        ("SDTM-007", "DM", "SEX", "INTERSEX", "UNDIFFERENTIATED"),
        # Race (C74457), plus MULTIPLE, which SDTMIG prescribes for subjects of several races
        ("SDTM-008", "DM", "RACE", "MULTIPLE", "MIXED"),
        # Frequency (C71113): submission values, not synonyms like DAILY
        ("SDTM-066", "EX", "EXDOSFRQ", "EVERY 8 WEEKS", "DAILY"),
        # Epoch (C99079)
        ("SDTM-434", "TA", "EPOCH", "BLINDED TREATMENT", "DOUBLE-BLIND TREATMENT"),
    ],
)
def test_codelists_use_cdisc_submission_values(rule_id, domain, column, valid, invalid):
    for value, n_issues in [(valid, 0), (invalid, 1)]:
        package = conformant_sdtm_package()
        package[domain] = package[domain].with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(pl.lit(value))
            .otherwise(pl.col(column))
            .alias(column)
        )

        assert _rule(pb.validate_sdtmig(package), rule_id).n_issues == n_issues, value


def test_codelists_used_by_rules_and_templates_are_bundled():
    """A codelist missing from the CT bundle would make its checks pass every value."""
    import json

    from pointblank.metadata._adam_templates import get_adam_dataset, list_adam_datasets
    from pointblank.metadata._conformance import ControlledTerminology
    from pointblank.metadata._conformance.rule_loader import _DATA_DIR
    from pointblank.metadata._sdtm_templates import get_sdtm_domain, list_sdtm_domains

    ct = ControlledTerminology.load_default()

    used = {
        op["params"]["codelist"]
        for path in _DATA_DIR.glob("*.json")
        for rule in json.loads(path.read_text(encoding="utf-8"))["rules"]
        for op in rule["operations"]
        if "codelist" in op.get("params", {})
    }
    templates = [get_sdtm_domain(d) for d in list_sdtm_domains()]
    templates += [get_adam_dataset(d) for d in list_adam_datasets()]
    used |= {v.controlled_term for t in templates for v in t.variables if v.controlled_term}

    assert sorted(name for name in used if name not in ct) == []


@pytest.mark.parametrize(
    "codelist, values",
    [
        # Official CDISC CT 2024-09-27 values that the hand-made bundle lacked
        ("NY", {"N", "NA", "U", "Y"}),
        ("SEX", {"F", "INTERSEX", "M", "U"}),
        ("OUT", {"FATAL", "RECOVERED/RESOLVED", "NOT RECOVERED/NOT RESOLVED"}),
        ("ROUTE", {"ORAL", "INTRAVENOUS", "SUBCUTANEOUS"}),
        # ISO 3166-1 alpha-3, territories included
        ("COUNTRY", {"USA", "GBR", "PRI", "HKG"}),
    ],
)
def test_ct_bundle_values(codelist, values):
    from pointblank.metadata._conformance import ControlledTerminology

    terms = ControlledTerminology.load_default().get_codelist(codelist)
    assert values <= terms


def test_ct_bundle_has_no_variable_named_codelists():
    """Codelists are keyed by their CDISC short names (ACN, OUT, UNIT), not variable names."""
    from pointblank.metadata._conformance import ControlledTerminology

    ct = ControlledTerminology.load_default()
    for name in ("AEACN", "AEOUT", "AEREL", "AETOXGR", "LBSTRESU"):
        assert name not in ct
