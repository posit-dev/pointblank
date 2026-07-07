"""Tests for the CDISC submission-package conformance model."""

import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import pointblank as pb
from pointblank.metadata._submission import ConformanceReport, SubmissionPackage


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _dm(usubjids=("S1-001", "S1-002", "S1-003")):
    n = len(usubjids)
    return pd.DataFrame(
        {
            "STUDYID": ["S1"] * n,
            "DOMAIN": ["DM"] * n,
            "USUBJID": list(usubjids),
            "SUBJID": [u.split("-")[-1] for u in usubjids],
            "ARMCD": ["A", "B", "A"][:n],
            "ARM": ["Arm A", "Arm B", "Arm A"][:n],
            "COUNTRY": ["USA"] * n,
        }
    )


def _ae(usubjids):
    n = len(usubjids)
    return pd.DataFrame(
        {
            "STUDYID": ["S1"] * n,
            "DOMAIN": ["AE"] * n,
            "USUBJID": list(usubjids),
            "AESEQ": list(range(1, n + 1)),
            "AETERM": ["Headache", "Nausea", "Fever"][:n],
        }
    )


# ── Construction & accessors ─────────────────────────────────────────────────


def test_construction_and_accessors():
    study = SubmissionPackage(datasets={"dm": _dm(), "AE": _ae(["S1-001", "S1-002"])})
    # Keys normalized to uppercase
    assert study.domains == ["AE", "DM"]
    assert "DM" in study
    assert "dm" in study
    assert len(study) == 2
    assert study.get_dataset("dm").shape[0] == 3
    assert study["AE"].shape[0] == 2

    with pytest.raises(KeyError):
        study.get_dataset("LB")


def test_top_level_exports():
    assert pb.SubmissionPackage is SubmissionPackage
    assert pb.ConformanceReport is ConformanceReport


def test_subject_ids_and_orphans():
    study = SubmissionPackage(datasets={"DM": _dm(), "AE": _ae(["S1-001", "S1-999"])})
    assert study.subject_ids("DM") == {"S1-001", "S1-002", "S1-003"}
    assert study.subject_ids("MISSING") == set()
    assert study.orphan_ids("AE", "DM") == {"S1-999"}


def test_summary_and_repr():
    study = SubmissionPackage(datasets={"DM": _dm()}, ct_version="2024-03-29", study_id="XYZ")
    s = study.summary()
    assert "XYZ" in s
    assert "2024-03-29" in s
    assert "DM" in s
    assert "SubmissionPackage" in repr(study)


# ── Conformance: pass path ───────────────────────────────────────────────────


def test_conformance_clean_passes():
    study = SubmissionPackage(datasets={"DM": _dm(), "AE": _ae(["S1-001", "S1-002"])})
    report = study.validate_conformance()
    assert isinstance(report, ConformanceReport)
    assert report.all_passed()
    assert report.issues() == []
    assert report.n_datasets == 2
    summ = report.summary()
    assert summ["AE"]["all_passed"] is True
    assert summ["DM"]["n_steps"] > 0


# ── Conformance: USUBJID referential integrity ───────────────────────────────


def test_referential_integrity_flags_orphan_usubjid():
    study = SubmissionPackage(datasets={"DM": _dm(), "AE": _ae(["S1-001", "S1-999"])})
    report = study.validate_conformance()
    assert not report.all_passed()
    issues = report.issues()
    assert any(i["dataset"] == "AE" for i in issues)

    # The failing step is the referential specially() check
    ae = report["AE"]
    ref_steps = [s for s in ae.validation_info if s.brief and "exist in DM" in s.brief]
    assert len(ref_steps) == 1
    assert ref_steps[0].n_failed == 1


def test_no_dm_means_no_referential_check():
    # Without DM there is no reference set, so no referential check is added.
    study = SubmissionPackage(datasets={"AE": _ae(["S1-001", "S1-002"])})
    report = study.validate_conformance()
    ae = report["AE"]
    assert not any(s.brief and "exist in DM" in s.brief for s in ae.validation_info)


def test_polars_datasets_supported():
    study = SubmissionPackage(
        datasets={
            "DM": pl.from_pandas(_dm()),
            "AE": pl.from_pandas(_ae(["S1-001", "S1-999"])),
        }
    )
    report = study.validate_conformance()
    assert not report.all_passed()


# ── Conformance: SUPP-- linkage ──────────────────────────────────────────────


def _suppae(idvarvals):
    n = len(idvarvals)
    return pd.DataFrame(
        {
            "STUDYID": ["S1"] * n,
            "RDOMAIN": ["AE"] * n,
            "USUBJID": ["S1-001"] * n,
            "IDVAR": ["AESEQ"] * n,
            "IDVARVAL": [str(v) for v in idvarvals],
            "QNAM": ["AESOC"] * n,
            "QLABEL": ["Body System"] * n,
            "QVAL": ["x"] * n,
        }
    )


def test_supp_idvar_resolution_pass():
    ae = _ae(["S1-001"])  # AESEQ == 1
    study = SubmissionPackage(datasets={"DM": _dm(), "AE": ae, "SUPPAE": _suppae([1])})
    report = study.validate_conformance()
    supp = report["SUPPAE"]
    idvar_steps = [s for s in supp.validation_info if s.brief and "IDVAR" in s.brief]
    assert len(idvar_steps) == 1
    assert idvar_steps[0].n_failed == 0


def test_supp_idvar_resolution_flags_dangling_link():
    ae = _ae(["S1-001"])  # AESEQ == 1 only
    study = SubmissionPackage(datasets={"DM": _dm(), "AE": ae, "SUPPAE": _suppae([99])})
    report = study.validate_conformance()
    supp = report["SUPPAE"]
    idvar_steps = [s for s in supp.validation_info if s.brief and "IDVAR" in s.brief]
    assert idvar_steps[0].n_failed == 1


def test_supp_rdomain_must_be_present():
    supp = _suppae([1, 1])  # two rows
    supp["RDOMAIN"] = "ZZ"  # not a present domain
    study = SubmissionPackage(datasets={"DM": _dm(), "AE": _ae(["S1-001"]), "SUPPAE": supp})
    report = study.validate_conformance()
    supp_v = report["SUPPAE"]
    rdom_steps = [s for s in supp_v.validation_info if s.brief and "RDOMAIN" in s.brief]
    assert rdom_steps[0].n_failed == 2


# ── Conformance: ADaM traceability ───────────────────────────────────────────


def _adsl(usubjids=("S1-001", "S1-002")):
    n = len(usubjids)
    return pd.DataFrame(
        {
            "STUDYID": ["S1"] * n,
            "USUBJID": list(usubjids),
            "SUBJID": [u.split("-")[-1] for u in usubjids],
            "ARM": ["Arm A", "Arm B"][:n],
            "ARMCD": ["A", "B"][:n],
            "TRT01P": ["Arm A", "Arm B"][:n],
            "AGE": [40, 50][:n],
            "SEX": ["M", "F"][:n],
            "RACE": ["WHITE", "ASIAN"][:n],
            "COUNTRY": ["USA", "USA"][:n],
            "SAFFL": ["Y", "Y"][:n],
        }
    )


def test_adam_adsl_traces_to_dm():
    # ADSL has a subject not in DM
    study = SubmissionPackage(
        datasets={"DM": _dm(["S1-001", "S1-002"]), "ADSL": _adsl(["S1-001", "S1-777"])},
        standard="adamig",
        standard_version="1.1",
    )
    report = study.validate_conformance()
    adsl = report["ADSL"]
    trace = [s for s in adsl.validation_info if s.brief and "trace to DM" in s.brief]
    assert trace[0].n_failed == 1


def test_adam_dataset_traces_to_adsl():
    adae = pd.DataFrame(
        {
            "STUDYID": ["S1"],
            "USUBJID": ["S1-777"],  # not in ADSL
            "TRT01A": ["Arm A"],
            "AESEQ": [1],
            "AETERM": ["Headache"],
            "TRTEMFL": ["Y"],
        }
    )
    study = SubmissionPackage(
        datasets={"DM": _dm(), "ADSL": _adsl(["S1-001", "S1-002"]), "ADAE": adae},
        standard="adamig",
        standard_version="1.1",
    )
    report = study.validate_conformance()
    adae_v = report["ADAE"]
    trace = [s for s in adae_v.validation_info if s.brief and "trace to ADSL" in s.brief]
    assert trace[0].n_failed == 1


# ── cross_dataset=False disables the extra checks ────────────────────────────


def test_cross_dataset_can_be_disabled():
    study = SubmissionPackage(datasets={"DM": _dm(), "AE": _ae(["S1-001", "S1-999"])})
    report = study.validate_conformance(cross_dataset=False)
    ae = report["AE"]
    assert not any(s.brief and "exist in DM" in s.brief for s in ae.validation_info)


# ── Report API ───────────────────────────────────────────────────────────────


def test_report_issues_and_html():
    study = SubmissionPackage(datasets={"DM": _dm(), "AE": _ae(["S1-001", "S1-999"])})
    report = study.validate_conformance()
    issues = report.issues()
    assert all({"dataset", "step", "assertion", "n_failed"} <= set(i) for i in issues)
    html = report._repr_html_()
    assert "Conformance Report" in html
    assert "AE" in html
    assert "not-a-dataset".upper() not in html
    # repr text
    assert "ConformanceReport" in repr(report)


def test_report_agency_recorded():
    study = SubmissionPackage(datasets={"DM": _dm()})
    report = study.validate_conformance(agency="FDA")
    assert report.agency == "FDA"
    assert "FDA" in report._repr_html_()


# ── from_folder ingestion ────────────────────────────────────────────────────


def test_from_folder_xpt_and_define_autodetect():
    src = pb.load_metadata_example("dm.xpt")
    define_src = pb.load_metadata_example("define.xml")
    d = Path(tempfile.mkdtemp())
    shutil.copy(src, d / "dm.xpt")
    shutil.copy(define_src, d / "define.xml")

    study = SubmissionPackage.from_folder(d)
    assert study.domains == ["DM"]
    assert study.get_dataset("DM").shape[0] == 5
    # Define-XML auto-detected and lazily importable
    assert study.define is not None
    assert study.metadata is not None

    report = study.validate_conformance()
    assert report.all_passed()


def test_from_folder_rejects_non_directory():
    with pytest.raises(NotADirectoryError):
        SubmissionPackage.from_folder(pb.load_metadata_example("dm.xpt"))


def test_from_folder_dataset_json():
    # Dataset-JSON 1.1 columns/rows layout
    doc = {
        "datasetJSONVersion": "1.1.0",
        "name": "DM",
        "columns": [
            {"name": "STUDYID"},
            {"name": "DOMAIN"},
            {"name": "USUBJID"},
            {"name": "SUBJID"},
            {"name": "ARMCD"},
            {"name": "ARM"},
            {"name": "COUNTRY"},
        ],
        "rows": [
            ["S1", "DM", "S1-001", "001", "A", "Arm A", "USA"],
            ["S1", "DM", "S1-002", "002", "B", "Arm B", "USA"],
        ],
    }
    d = Path(tempfile.mkdtemp())
    (d / "dm.json").write_text(json.dumps(doc))
    study = SubmissionPackage.from_folder(d)
    assert study.domains == ["DM"]
    assert study.get_dataset("DM").shape == (2, 7)
    assert study.subject_ids("DM") == {"S1-001", "S1-002"}
