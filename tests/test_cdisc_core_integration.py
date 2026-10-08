"""Real end-to-end CDISC CORE engine integration tests.

These tests require the CDISC CORE engine to be installed and discoverable. They are auto-skipped
when CORE is absent (see the `pytest_collection_modifyitems` hook in `conftest.py`).

How to run:
    # Standalone executable on PATH (named 'core' or 'cdisc-rules-engine'):
    pytest -m cdisc_core

    # Explicit path / command via env var (supports repo-checkout invocations):
    POINTBLANK_CDISC_CORE="python /path/to/cdisc-rules-engine/core.py" \\
    POINTBLANK_CDISC_CORE_CWD="/path/to/cdisc-rules-engine" \\
    pytest -m cdisc_core

    # Optional: cache directory for the rules cache:
    POINTBLANK_CDISC_CORE_CACHE="/path/to/cdisc-rules-engine/resources/cache" \\
    pytest -m cdisc_core

These env vars are honoured automatically by pointblank (POINTBLANK_CDISC_CORE) and by this test
module (POINTBLANK_CDISC_CORE_CWD, POINTBLANK_CDISC_CORE_CACHE) so no command-line options are
needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

import pointblank as pb
from pointblank.metadata import ConformanceReport, CoreFinding, CoreRuleResult, parse_core_report

pytestmark = pytest.mark.cdisc_core

# ── Helpers ───────────────────────────────────────────────────────────────────

# Optional env vars that tell the integration tests how to invoke a repo-checkout CORE.
_CORE_CWD = os.environ.get("POINTBLANK_CDISC_CORE_CWD")
_CORE_CACHE = os.environ.get("POINTBLANK_CDISC_CORE_CACHE")


def _run_kwargs() -> dict:
    """Extra kwargs forwarded to validate_cdisc_submission / validate_conformance."""
    kw = {}
    if _CORE_CWD:
        kw["core_cwd"] = _CORE_CWD
    if _CORE_CACHE:
        kw["cache"] = _CORE_CACHE
    return kw


def _minimal_dm() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "STUDYID": ["STUDY01"] * 3,
            "DOMAIN": ["DM"] * 3,
            "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
            "SUBJID": ["001", "002", "003"],
            "RFSTDTC": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "ARMCD": ["A", "B", "A"],
            "ARM": ["Arm A", "Arm B", "Arm A"],
            "SEX": ["M", "F", "M"],
            "RACE": ["WHITE", "ASIAN", "BLACK OR AFRICAN AMERICAN"],
            "COUNTRY": ["USA", "USA", "USA"],
            "DMDTC": ["2024-01-01", "2024-01-02", "2024-01-03"],
        }
    )


@pytest.fixture(scope="module")
def core_report(tmp_path_factory) -> ConformanceReport:
    """Run the real CORE engine once per test-module and cache the result."""
    tmp = tmp_path_factory.mktemp("cdisc_core_integration")
    return pb.validate_cdisc_submission(
        {"DM": _minimal_dm()},
        standard="sdtmig",
        version="3.4",
        workdir=tmp / "workdir",
        **_run_kwargs(),
    )


# ── Basic shape and provenance ─────────────────────────────────────────────────


def test_report_is_core_form(core_report):
    assert core_report.is_core is True


def test_report_has_many_rules(core_report):
    # A healthy SDTMIG 3.4 run covers hundreds of rules.
    assert len(core_report.rules()) >= 100


def test_report_summary_keys_and_types(core_report):
    s = core_report.summary()
    assert isinstance(s["standard"], str)
    assert isinstance(s["version"], str)
    assert isinstance(s["engine_version"], str)
    assert isinstance(s["n_rules"], int) and s["n_rules"] >= 100
    assert isinstance(s["n_issues"], int)
    assert isinstance(s["status_counts"], dict)
    assert isinstance(s["all_passed"], bool)


def test_report_standard_version_recorded(core_report):
    s = core_report.summary()
    # CORE uppercases the standard and prepends "V" to the version.
    assert "SDTMIG" in s["standard"].upper()
    assert "3" in s["version"]


# ── Accessors ──────────────────────────────────────────────────────────────────


def test_issues_returns_list_of_dicts(core_report):
    issues = core_report.issues()
    assert isinstance(issues, list)
    if issues:
        assert all({"dataset", "rule_id", "message", "issues", "status"} <= set(i) for i in issues)


def test_findings_returns_core_finding_objects(core_report):
    findings = core_report.findings()
    assert isinstance(findings, list)
    if findings:
        assert all(isinstance(f, CoreFinding) for f in findings)
        assert all(f.rule_id and f.dataset for f in findings)


def test_rules_returns_core_rule_result_objects(core_report):
    all_rules = core_report.rules()
    assert all(isinstance(r, CoreRuleResult) for r in all_rules)
    # Every rule has a status string
    assert all(isinstance(r.status, str) and r.status for r in all_rules)


def test_rules_status_filter(core_report):
    from pointblank.metadata._cdisc_core import STATUS_SKIPPED, STATUS_SUCCESS

    success = core_report.rules(status=STATUS_SUCCESS)
    skipped = core_report.rules(status=STATUS_SKIPPED)
    assert all(r.status == STATUS_SUCCESS for r in success)
    assert all(r.status == STATUS_SKIPPED for r in skipped)
    # Together they should cover most of the rule set (a minimal DM won't fail everything)
    assert len(success) + len(skipped) > 0


def test_all_passed_is_bool(core_report):
    # For a minimal one-dataset DM, all_passed is either True or False (real run)
    assert isinstance(core_report.all_passed(), bool)


# ── Export methods ─────────────────────────────────────────────────────────────


def test_to_json_round_trips(core_report, tmp_path):
    dest = core_report.to_json(tmp_path / "report.json")
    assert dest.exists() and dest.stat().st_size > 0

    data = json.loads(dest.read_text())
    reparsed = parse_core_report(data)
    assert reparsed.standard == core_report.core.standard
    assert len(reparsed.rules) == len(core_report.core.rules)
    assert reparsed.n_total_issues == core_report.core.n_total_issues


def test_to_excel_sheets(core_report, tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    dest = core_report.to_excel(tmp_path / "report.xlsx")
    assert dest.exists() and dest.stat().st_size > 0
    wb = openpyxl.load_workbook(dest)
    assert "Rules_Report" in wb.sheetnames
    assert "Conformance_Details" in wb.sheetnames
    # Row count: header + one row per rule
    n_data_rows = wb["Rules_Report"].max_row - 1
    assert n_data_rows == len(core_report.core.rules)


# ── from_folder passthrough ────────────────────────────────────────────────────


def test_from_folder_with_core(tmp_path):
    """CORE reads existing XPT files directly when the package was ingested from a folder."""
    pytest.importorskip("pyreadstat")
    import shutil

    folder = tmp_path / "study"
    folder.mkdir()
    shutil.copy(pb.load_metadata_example("dm.xpt"), folder / "dm.xpt")

    pkg = pb.SubmissionPackage.from_folder(folder)
    assert pkg.source_folder == str(folder)

    rep = pkg.validate_conformance(
        engine="core",
        standard="sdtmig",
        version="3.4",
        workdir=tmp_path / "w",
        **_run_kwargs(),
    )
    assert rep.is_core is True
    assert len(rep.rules()) >= 100


# ── validate_cdisc_submission shortcuts ────────────────────────────────────────


def test_validate_cdisc_submission_accepts_package(tmp_path):
    pkg = pb.SubmissionPackage(
        datasets={"DM": _minimal_dm()},
        standard="sdtmig",
        standard_version="3.4",
    )
    rep = pb.validate_cdisc_submission(pkg, workdir=tmp_path / "w", **_run_kwargs())
    assert rep.is_core is True
    assert rep.package is pkg
