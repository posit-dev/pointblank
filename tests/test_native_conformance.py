"""Tests for the native CDISC conformance rule engine."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import pointblank as pb
from pointblank.metadata._conformance import (
    ControlledTerminology,
    NativeConformanceEngine,
    NativeConformanceResult,
    NativeRowFinding,
    NativeRuleResult,
    RuleLoader,
)
from pointblank.metadata._conformance.evaluator import evaluate_conditions, is_iso8601
from pointblank.metadata._conformance.operations import apply_operations
from pointblank.metadata._conformance.result import STATUS_FAIL, STATUS_PASS
import narwhals as nw


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _clean_dm(backend="polars"):
    data = {
        "STUDYID": ["S001", "S001"],
        "DOMAIN": ["DM", "DM"],
        "USUBJID": ["S001-001", "S001-002"],
        "SUBJID": ["001", "002"],
        "SEX": ["M", "F"],
        "RACE": ["WHITE", "ASIAN"],
        "COUNTRY": ["USA", "GBR"],
        "DMDTC": ["2024-01-01", "2024-01-02"],
    }
    if backend == "pandas":
        return pd.DataFrame(data)
    return pl.DataFrame(data)


@pytest.fixture
def ct():
    return ControlledTerminology.load_default()


@pytest.fixture
def engine():
    return NativeConformanceEngine("sdtmig", "3.4")


@pytest.fixture
def clean_result(engine):
    return engine.run({"DM": _clean_dm()})


# ── RuleLoader ────────────────────────────────────────────────────────────────


def test_rule_loader_loads_sdtmig_34():
    rules = RuleLoader.load("sdtmig", "3.4")
    assert len(rules) >= 20


def test_rule_loader_rule_types_coverage():
    rules = RuleLoader.load("sdtmig", "3.4")
    types = {r.rule_type for r in rules}
    assert "RECORD_CHECK" in types
    assert "DATASET_CONTENTS_CHECK" in types
    assert "DATASET_METADATA_CHECK" in types
    assert "DOMAIN_PRESENCE_CHECK" in types


def test_rule_loader_available():
    available = RuleLoader.available()
    assert ("sdtmig", "3.4") in available


def test_rule_loader_missing_raises():
    with pytest.raises(FileNotFoundError, match="No bundled rule catalog"):
        RuleLoader.load("imaginary", "99.0")


def test_rule_loader_filter_by_type():
    rules = RuleLoader.load("sdtmig", "3.4", rule_types=["RECORD_CHECK"])
    assert all(r.rule_type == "RECORD_CHECK" for r in rules)


def test_rule_loader_catalog_metadata():
    meta = RuleLoader.catalog_metadata("sdtmig", "3.4")
    assert "standard" in meta
    assert meta["standard"] == "sdtmig"
    assert "checksum" in meta


# ── ControlledTerminology ─────────────────────────────────────────────────────


def test_ct_load_default(ct):
    assert len(ct.packages) == 1
    assert "SEX" in ct


def test_ct_valid_term(ct):
    assert ct.is_valid("SEX", "M")
    assert ct.is_valid("SEX", "F")


def test_ct_invalid_term(ct):
    assert not ct.is_valid("SEX", "Q")


def test_ct_null_passes(ct):
    # Null values pass by convention (separate not-null rule handles them).
    assert ct.is_valid("SEX", None)


def test_ct_unknown_codelist_passes(ct):
    assert ct.is_valid("NONEXISTENT_CODELIST", "ANYTHING")


def test_ct_available():
    available = ControlledTerminology.available()
    assert len(available) >= 1


def test_ct_missing_package_raises():
    with pytest.raises(FileNotFoundError):
        ControlledTerminology.load(["no-such-package-2099-01-01"])


# ── Evaluator ─────────────────────────────────────────────────────────────────


def _nw_df(data: dict) -> nw.DataFrame:
    return nw.from_native(pl.DataFrame(data), eager_only=True)


def test_evaluator_is_null():
    df = _nw_df({"x": [1, None, 3]})
    mask = evaluate_conditions(df, {"all": [{"name": "x", "operator": "is_null", "value": None}]})
    assert mask.to_list() == [False, True, False]


def test_evaluator_is_not_null():
    df = _nw_df({"x": [1, None, 3]})
    mask = evaluate_conditions(
        df, {"all": [{"name": "x", "operator": "is_not_null", "value": None}]}
    )
    assert mask.to_list() == [True, False, True]


def test_evaluator_equal_to():
    df = _nw_df({"x": [1, 2, 3]})
    mask = evaluate_conditions(df, {"all": [{"name": "x", "operator": "equal_to", "value": 2}]})
    assert mask.to_list() == [False, True, False]


def test_evaluator_not_equal_to():
    df = _nw_df({"x": [1, 2, 3]})
    mask = evaluate_conditions(df, {"all": [{"name": "x", "operator": "not_equal_to", "value": 2}]})
    assert mask.to_list() == [True, False, True]


def test_evaluator_greater_than():
    df = _nw_df({"x": [1, 5, 3]})
    mask = evaluate_conditions(df, {"all": [{"name": "x", "operator": "greater_than", "value": 3}]})
    assert mask.to_list() == [False, True, False]


def test_evaluator_is_in():
    df = _nw_df({"x": ["A", "B", "C"]})
    mask = evaluate_conditions(
        df, {"all": [{"name": "x", "operator": "is_in", "value": ["A", "C"]}]}
    )
    assert mask.to_list() == [True, False, True]


def test_evaluator_not_in():
    df = _nw_df({"x": ["A", "B", "C"]})
    mask = evaluate_conditions(
        df, {"all": [{"name": "x", "operator": "not_in", "value": ["A", "C"]}]}
    )
    assert mask.to_list() == [False, True, False]


def test_evaluator_contains():
    df = _nw_df({"x": ["hello world", "foo", "world"]})
    mask = evaluate_conditions(
        df, {"all": [{"name": "x", "operator": "contains", "value": "world"}]}
    )
    assert mask.to_list() == [True, False, True]


def test_evaluator_any_combinator():
    df = _nw_df({"x": [1, 2, 3], "y": [10, 20, 30]})
    mask = evaluate_conditions(
        df,
        {
            "any": [
                {"name": "x", "operator": "equal_to", "value": 1},
                {"name": "y", "operator": "equal_to", "value": 30},
            ]
        },
    )
    assert mask.to_list() == [True, False, True]


def test_evaluator_not_combinator():
    df = _nw_df({"x": [1, 2, 3]})
    mask = evaluate_conditions(df, {"not": {"name": "x", "operator": "equal_to", "value": 2}})
    assert mask.to_list() == [True, False, True]


def test_evaluator_empty_conditions_returns_all_false():
    df = _nw_df({"x": [1, 2, 3]})
    mask = evaluate_conditions(df, {})
    assert all(not v for v in mask.to_list())


def test_evaluator_equal_to_column():
    df = _nw_df({"a": [1, 2, 3], "b": [1, 99, 3]})
    mask = evaluate_conditions(
        df, {"all": [{"name": "a", "operator": "equal_to_column", "value": "b"}]}
    )
    assert mask.to_list() == [True, False, True]


def test_iso8601_valid():
    assert is_iso8601("2024-01-15")
    assert is_iso8601("2024-01")
    assert is_iso8601("2024")
    assert is_iso8601("2024-01-15T10:30:00")
    assert is_iso8601("2024-01-15T10:30:00Z")
    assert is_iso8601("2024-01-15T10:30:00+05:30")


def test_iso8601_invalid():
    assert not is_iso8601("01/15/2024")
    assert not is_iso8601("not-a-date")
    assert not is_iso8601("")
    assert not is_iso8601(None)


# ── Operations ────────────────────────────────────────────────────────────────


def test_op_codelist_check_valid(ct):
    df = _nw_df({"SEX": ["M", "F", "Q"]})
    result = apply_operations(
        df, [{"operator": "codelist_check", "params": {"column": "SEX", "codelist": "SEX"}}], ct, {}
    )
    assert "_pb_SEX_valid" in result.columns
    assert result["_pb_SEX_valid"].to_list() == [True, True, False]


def test_op_codelist_check_null_passes(ct):
    df = _nw_df({"SEX": ["M", None, "F"]})
    result = apply_operations(
        df, [{"operator": "codelist_check", "params": {"column": "SEX", "codelist": "SEX"}}], ct, {}
    )
    assert result["_pb_SEX_valid"].to_list() == [True, True, True]


def test_op_codelist_check_missing_column(ct):
    df = _nw_df({"OTHER": ["A"]})
    result = apply_operations(
        df, [{"operator": "codelist_check", "params": {"column": "SEX", "codelist": "SEX"}}], ct, {}
    )
    assert result["_pb_SEX_valid"].to_list() == [True]


def test_op_consistency_check_consistent(ct):
    df = _nw_df({"STUDYID": ["S001", "S001", "S001"]})
    result = apply_operations(
        df, [{"operator": "consistency_check", "params": {"column": "STUDYID"}}], ct, {}
    )
    assert all(result["_pb_STUDYID_consistent"].to_list())


def test_op_consistency_check_inconsistent(ct):
    df = _nw_df({"STUDYID": ["S001", "S001", "S002"]})
    result = apply_operations(
        df, [{"operator": "consistency_check", "params": {"column": "STUDYID"}}], ct, {}
    )
    vals = result["_pb_STUDYID_consistent"].to_list()
    assert vals[2] is False  # S002 is the outlier


def test_op_iso8601_check_valid(ct):
    df = _nw_df({"DTC": ["2024-01-01", "not-a-date", None]})
    result = apply_operations(
        df, [{"operator": "iso8601_check", "params": {"column": "DTC"}}], ct, {}
    )
    vals = result["_pb_DTC_iso8601"].to_list()
    assert vals == [True, False, True]


def test_op_column_presence_present(ct):
    df = _nw_df({"USUBJID": ["U1"]})
    result = apply_operations(
        df, [{"operator": "column_presence", "params": {"column": "USUBJID"}}], ct, {}
    )
    assert result["_pb_USUBJID_present"].to_list() == [True]


def test_op_column_presence_absent(ct):
    df = _nw_df({"OTHER": ["X"]})
    result = apply_operations(
        df, [{"operator": "column_presence", "params": {"column": "USUBJID"}}], ct, {}
    )
    assert result["_pb_USUBJID_present"].to_list() == [False]


def test_op_unknown_operator_skipped(ct):
    df = _nw_df({"x": [1]})
    # Should not raise; unknown operators are silently skipped.
    result = apply_operations(df, [{"operator": "nonexistent_op", "params": {}}], ct, {})
    assert result.columns == ["x"]


# ── NativeConformanceEngine: clean data ───────────────────────────────────────


def test_engine_clean_dm_all_pass(clean_result):
    assert clean_result.all_passed


def test_engine_clean_dm_zero_issues(clean_result):
    assert clean_result.n_total_issues == 0


def test_engine_rule_count(clean_result):
    assert len(clean_result.rule_results) == 30


def test_engine_result_types(clean_result):
    assert isinstance(clean_result, NativeConformanceResult)
    assert all(isinstance(r, NativeRuleResult) for r in clean_result.rule_results)


def test_engine_status_counts(clean_result):
    counts = clean_result.status_counts()
    assert STATUS_PASS in counts
    assert counts.get(STATUS_FAIL, 0) == 0


def test_engine_findings_empty_for_clean_data(clean_result):
    assert clean_result.findings() == []


# ── NativeConformanceEngine: violations ───────────────────────────────────────


def test_engine_detects_invalid_sex(engine):
    dm = pl.DataFrame(
        {
            "STUDYID": ["S1"],
            "DOMAIN": ["DM"],
            "USUBJID": ["U1"],
            "SUBJID": ["1"],
            "SEX": ["Q"],
            "COUNTRY": ["USA"],
            "DMDTC": ["2024-01-01"],
        }
    )
    result = engine.run({"DM": dm})
    ids = {r.rule_id for r in result.rule_results if r.status == STATUS_FAIL}
    assert "SDTM-007" in ids


def test_engine_detects_invalid_country(engine):
    dm = pl.DataFrame(
        {
            "STUDYID": ["S1"],
            "DOMAIN": ["DM"],
            "USUBJID": ["U1"],
            "SUBJID": ["1"],
            "COUNTRY": ["XYZ"],
            "DMDTC": ["2024-01-01"],
        }
    )
    result = engine.run({"DM": dm})
    ids = {r.rule_id for r in result.rule_results if r.status == STATUS_FAIL}
    assert "SDTM-009" in ids


def test_engine_detects_bad_date(engine):
    dm = pl.DataFrame(
        {
            "STUDYID": ["S1"],
            "DOMAIN": ["DM"],
            "USUBJID": ["U1"],
            "SUBJID": ["1"],
            "DMDTC": ["not-a-date"],
        }
    )
    result = engine.run({"DM": dm})
    ids = {r.rule_id for r in result.rule_results if r.status == STATUS_FAIL}
    assert "SDTM-010" in ids


def test_engine_detects_null_usubjid(engine):
    dm = pl.DataFrame({"STUDYID": ["S1", "S1"], "DOMAIN": ["DM", "DM"], "USUBJID": ["U1", None]})
    result = engine.run({"DM": dm})
    ids = {r.rule_id for r in result.rule_results if r.status == STATUS_FAIL}
    assert "SDTM-004" in ids


def test_engine_detects_missing_dm_domain(engine):
    ae = pl.DataFrame({"STUDYID": ["S1"], "DOMAIN": ["AE"], "USUBJID": ["U1"]})
    result = engine.run({"AE": ae})
    ids = {r.rule_id for r in result.rule_results if r.status == STATUS_FAIL}
    assert "SDTM-001" in ids


def test_engine_detects_inconsistent_studyid(engine):
    dm = pl.DataFrame({"STUDYID": ["S1", "S2"], "DOMAIN": ["DM", "DM"], "USUBJID": ["U1", "U2"]})
    result = engine.run({"DM": dm})
    ids = {r.rule_id for r in result.rule_results if r.status == STATUS_FAIL}
    assert "SDTM-005" in ids


def test_engine_row_findings_populated(engine):
    dm = pl.DataFrame(
        {"STUDYID": ["S1"], "DOMAIN": ["DM"], "USUBJID": ["U1"], "SUBJID": ["1"], "SEX": ["Q"]}
    )
    result = engine.run({"DM": dm})
    findings = result.findings()
    assert len(findings) > 0
    f = findings[0]
    assert isinstance(f, NativeRowFinding)
    assert f.rule_id == "SDTM-007"
    assert f.row == 0


def test_engine_rules_status_filter(clean_result):
    passing = clean_result.rules(status=STATUS_PASS)
    assert all(r.status == STATUS_PASS for r in passing)


def test_engine_issues_list(engine):
    dm = pl.DataFrame(
        {"STUDYID": ["S1"], "DOMAIN": ["DM"], "USUBJID": ["U1"], "SUBJID": ["1"], "SEX": ["Q"]}
    )
    result = engine.run({"DM": dm})
    issues = result.issues()
    assert len(issues) > 0
    issue = next(i for i in issues if i["rule_id"] == "SDTM-007")
    assert issue["n_issues"] == 1
    assert issue["status"] == STATUS_FAIL


# ── Pandas backend ────────────────────────────────────────────────────────────


def test_engine_works_with_pandas(engine):
    dm = pd.DataFrame(
        {"STUDYID": ["S1"], "DOMAIN": ["DM"], "USUBJID": ["U1"], "SUBJID": ["1"], "SEX": ["Q"]}
    )
    result = engine.run({"DM": dm})
    ids = {r.rule_id for r in result.rule_results if r.status == STATUS_FAIL}
    assert "SDTM-007" in ids


# ── SubmissionPackage integration ─────────────────────────────────────────────


def test_submission_package_uses_rules_engine():
    pkg = pb.SubmissionPackage(
        datasets={"DM": _clean_dm()}, standard="sdtmig", standard_version="3.4"
    )
    report = pkg.validate_conformance()
    assert report.is_rules
    assert not report.is_core


def test_submission_package_summary_has_engine_key():
    pkg = pb.SubmissionPackage(datasets={"DM": _clean_dm()})
    report = pkg.validate_conformance()
    s = report.summary()
    assert s["engine"] == "native"
    assert "n_rules" in s
    assert "n_issues" in s


def test_submission_package_dirty_data_fails():
    dirty = pl.DataFrame(
        {"STUDYID": ["S1"], "DOMAIN": ["DM"], "USUBJID": ["U1"], "SUBJID": ["1"], "SEX": ["BAD"]}
    )
    pkg = pb.SubmissionPackage(datasets={"DM": dirty})
    report = pkg.validate_conformance()
    assert not report.all_passed()
    assert len(report.issues()) > 0


def test_submission_package_repr_shows_native_rules():
    pkg = pb.SubmissionPackage(datasets={"DM": _clean_dm()})
    report = pkg.validate_conformance()
    r = repr(report)
    assert "Native Rules" in r


def test_submission_package_to_json_rules(tmp_path):
    pkg = pb.SubmissionPackage(datasets={"DM": _clean_dm()})
    report = pkg.validate_conformance()
    dest = report.to_json(tmp_path / "r.json")
    data = json.loads(dest.read_text())
    assert data["summary"]["engine"] == "native"
    assert isinstance(data["issues"], list)


def test_submission_package_to_excel_rules(tmp_path):
    pytest.importorskip("openpyxl")
    import openpyxl

    pkg = pb.SubmissionPackage(datasets={"DM": _clean_dm()})
    report = pkg.validate_conformance()
    dest = report.to_excel(tmp_path / "r.xlsx")
    wb = openpyxl.load_workbook(dest)
    assert "Rules_Report" in wb.sheetnames
    assert "Summary" in wb.sheetnames
    # Every rule has a row.
    n_data_rows = wb["Rules_Report"].max_row - 1
    assert n_data_rows == len(report.rules())


def test_submission_package_rules_accessor():
    pkg = pb.SubmissionPackage(datasets={"DM": _clean_dm()})
    report = pkg.validate_conformance()
    rules = report.rules()
    assert len(rules) > 0
    assert all(isinstance(r, NativeRuleResult) for r in rules)


def test_submission_package_findings_accessor():
    dirty = pl.DataFrame({"STUDYID": ["S1"], "DOMAIN": ["DM"], "USUBJID": ["U1"], "SEX": ["BAD"]})
    pkg = pb.SubmissionPackage(datasets={"DM": dirty})
    report = pkg.validate_conformance()
    findings = report.findings()
    assert len(findings) > 0
    assert all(isinstance(f, NativeRowFinding) for f in findings)
