"""Tests for the CDISC CORE JSON report parser and CORE-backed ConformanceReport (PLAN_06 Phase 2).

These tests run entirely against captured CORE report fixtures — no CORE engine dependency. The
fixtures were produced by `core validate -s sdtmig -v 3-4 ... -of JSON` against CORE 0.16.0.
"""

import json
import sys
from pathlib import Path

import pytest

from pointblank.metadata import (
    ConformanceReport,
    CoreFinding,
    CoreRuleResult,
    ParsedCoreReport,
    parse_core_report,
)
from pointblank.metadata._cdisc_core import (
    STATUS_ERROR,
    STATUS_ISSUE,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    CoreExecutionError,
    CoreNotFoundError,
    _CoreRunner,
    _materialize_datasets,
    _normalize_version,
    _resolve_core_command,
    _write_xpt,
)

_FIXTURES = Path(__file__).parent / "metadata_fixtures" / "cdisc_core"


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text())


# A stand-in for the CORE CLI: copies a source JSON to `<stem>.<ext>` and records its argv to
# `<stem>.argv.json`, so tests can drive the runner and assert the built command — no real CORE.
_FAKE_CORE_TEMPLATE = '''\
import json, shutil, sys
args = sys.argv[1:]
def opt(flag):
    return args[args.index(flag) + 1] if flag in args else None
stem = opt("-o")
fmt = opt("-of") or "JSON"
ext = {{"JSON": "json", "XLSX": "xlsx", "CSV": "csv"}}[fmt]
with open(stem + ".argv.json", "w") as f:
    json.dump(args, f)
shutil.copy({src!r}, stem + "." + ext)
sys.exit({exit_code})
'''


def _make_fake_core(tmp_path: Path, src_json: Path, exit_code: int = 0) -> Path:
    script = tmp_path / "fakecore.py"
    script.write_text(_FAKE_CORE_TEMPLATE.format(src=str(src_json), exit_code=exit_code))
    return script


def _fake_runner(tmp_path: Path, exit_code: int = 0) -> _CoreRunner:
    src = _FIXTURES / "core_report_trimmed.json"
    script = _make_fake_core(tmp_path, src, exit_code=exit_code)
    return _CoreRunner(core=[sys.executable, str(script)])


@pytest.fixture
def full_report() -> dict:
    return _load("core_report_full.json")


@pytest.fixture
def trimmed_report() -> dict:
    return _load("core_report_trimmed.json")


# ── Parser ───────────────────────────────────────────────────────────────────


def test_parse_full_report_provenance(full_report):
    parsed = parse_core_report(full_report)
    assert isinstance(parsed, ParsedCoreReport)
    assert parsed.standard == "SDTMIG"
    assert parsed.version == "V3.4"
    assert parsed.engine_version == "0.16.0"
    assert len(parsed.rules) == 430
    assert len(parsed.datasets) == 1


def test_parse_status_counts(full_report):
    parsed = parse_core_report(full_report)
    counts = parsed.status_counts()
    assert counts == {
        STATUS_SKIPPED: 344,
        STATUS_SUCCESS: 78,
        STATUS_ISSUE: 6,
        STATUS_ERROR: 2,
    }


def test_parse_findings_typed(full_report):
    parsed = parse_core_report(full_report)
    assert all(isinstance(f, CoreFinding) for f in parsed.findings)
    f = parsed.findings[0]
    assert f.rule_id == "CORE-000357"
    assert f.dataset == "TEST_DATASET"
    assert f.row == 1
    assert f.variables == ["dataset_name"]
    assert f.values == ["TEST_DATASET"]
    # Empty-string sentinels normalized to None
    assert f.usubjid is None
    assert f.seq is None


def test_parse_rules_typed_and_is_failing(full_report):
    parsed = parse_core_report(full_report)
    assert all(isinstance(r, CoreRuleResult) for r in parsed.rules)
    failing = parsed.failing_rules()
    # 6 ISSUE REPORTED + 2 EXECUTION ERROR = 8 failing rules
    assert len(failing) == 8
    assert all(r.is_failing for r in failing)
    success = [r for r in parsed.rules if r.status == STATUS_SUCCESS]
    assert success and not success[0].is_failing


def test_parse_total_issues_and_all_passed(full_report):
    parsed = parse_core_report(full_report)
    assert parsed.n_total_issues == sum(s.issues for s in parsed.issue_summary)
    assert parsed.all_passed is False


def test_parse_accepts_already_parsed_via_report(trimmed_report):
    # from_core_report should accept both a dict and a ParsedCoreReport
    parsed = parse_core_report(trimmed_report)
    rep = ConformanceReport.from_core_report(parsed)
    assert rep.is_core
    assert rep.core is parsed


def test_all_passed_true_when_no_failing_rules():
    report = {
        "Conformance_Details": {"Standard": "SDTMIG", "Version": "V3.4"},
        "Dataset_Details": [],
        "Issue_Summary": [],
        "Issue_Details": [],
        "Rules_Report": [
            {"core_id": "CORE-1", "status": "SUCCESS"},
            {"core_id": "CORE-2", "status": "SKIPPED"},
        ],
    }
    parsed = parse_core_report(report)
    assert parsed.all_passed is True


def test_all_passed_falls_back_to_issue_summary_when_no_rules():
    report = {
        "Conformance_Details": {},
        "Issue_Summary": [{"dataset": "DM", "core_id": "CORE-1", "message": "x", "issues": 3}],
        "Issue_Details": [],
        "Rules_Report": [],
    }
    parsed = parse_core_report(report)
    assert parsed.all_passed is False


def test_parse_raises_on_non_dict():
    with pytest.raises(TypeError):
        parse_core_report(["not", "a", "dict"])


def test_parse_raises_on_unrecognized_dict():
    with pytest.raises(ValueError):
        parse_core_report({"foo": "bar", "baz": 1})


def test_parse_raw_report_ignores_results_data(full_report):
    # The --raw-report variant adds a results_data key; parser should ignore it.
    raw = dict(full_report)
    raw["results_data"] = [{"anything": 1}]
    parsed = parse_core_report(raw)
    assert len(parsed.rules) == 430


def test_as_int_coercion_edge_cases():
    report = {
        "Conformance_Details": {},
        "Issue_Summary": [{"dataset": "DM", "core_id": "C", "message": "m", "issues": ""}],
        "Issue_Details": [{"core_id": "C", "message": "m", "dataset": "DM", "row": "5"}],
        "Rules_Report": [],
    }
    parsed = parse_core_report(report)
    assert parsed.issue_summary[0].issues == 0  # "" -> 0
    assert parsed.findings[0].row == 5  # "5" -> 5


# ── ConformanceReport (CORE form) ──────────────────────────────────────────────


def test_report_is_core_and_all_passed(full_report):
    rep = ConformanceReport.from_core_report(full_report, agency="FDA")
    assert rep.is_core is True
    assert rep.agency == "FDA"
    assert rep.all_passed() is False
    assert rep.n_datasets == 1


def test_report_summary_core(full_report):
    rep = ConformanceReport.from_core_report(full_report)
    s = rep.summary()
    assert s["standard"] == "SDTMIG"
    assert s["version"] == "V3.4"
    assert s["engine_version"] == "0.16.0"
    assert s["n_rules"] == 430
    assert s["n_issues"] == 8
    assert s["all_passed"] is False
    assert s["status_counts"][STATUS_ISSUE] == 6


def test_report_issues_core(full_report):
    rep = ConformanceReport.from_core_report(full_report)
    issues = rep.issues()
    assert len(issues) == 8
    assert all({"dataset", "rule_id", "message", "issues", "status"} <= set(i) for i in issues)
    # every issue-summary rule id resolves to a status in the (consistent) full fixture
    assert all(i["status"] is not None for i in issues)


def test_report_issues_status_filter(full_report):
    rep = ConformanceReport.from_core_report(full_report)
    errs = rep.issues(status=STATUS_ERROR)
    assert {i["rule_id"] for i in errs} == {"CORE-000929", "CORE-001081"}
    assert all(i["status"] == STATUS_ERROR for i in errs)


def test_report_findings_and_rules_accessors(full_report):
    rep = ConformanceReport.from_core_report(full_report)
    findings = rep.findings()
    assert findings and isinstance(findings[0], CoreFinding)
    all_rules = rep.rules()
    assert len(all_rules) == 430
    issue_rules = rep.rules(status=STATUS_ISSUE)
    assert len(issue_rules) == 6
    assert all(r.status == STATUS_ISSUE for r in issue_rules)


def test_report_repr_and_html_core(full_report):
    rep = ConformanceReport.from_core_report(full_report, agency="FDA")
    text = repr(rep)
    assert "ConformanceReport (CORE)" in text
    assert "SDTMIG V3.4" in text
    assert "FAIL" in text
    html = rep._repr_html_()
    assert "CORE" in html
    assert "SDTMIG" in html
    assert "CORE-000357" in html  # an issue rule id appears in the table


def test_trimmed_fixture_is_internally_consistent(trimmed_report):
    # Every rule referenced by Issue_Summary must resolve to a status (no None) in the trimmed set.
    rep = ConformanceReport.from_core_report(trimmed_report)
    assert all(i["status"] is not None for i in rep.issues())


# ── Native/CORE separation ─────────────────────────────────────────────────────


def test_native_report_findings_rules_empty():
    # A native (non-CORE) report returns empty CORE accessors and is_core False.
    import pandas as pd

    import pointblank as pb

    dm = pd.DataFrame(
        {
            "STUDYID": ["S1"],
            "DOMAIN": ["DM"],
            "USUBJID": ["S1-001"],
            "SUBJID": ["001"],
            "ARMCD": ["A"],
            "ARM": ["Arm A"],
            "COUNTRY": ["USA"],
        }
    )
    rep = pb.SubmissionPackage(datasets={"DM": dm}).validate_conformance()
    assert rep.is_core is False
    assert rep.findings() == []
    assert rep.rules() == []


# ── Dataset materialization (_write_xpt / _materialize_datasets) ────────────────


def _dm_df():
    import pandas as pd

    return pd.DataFrame(
        {
            "STUDYID": ["S1", "S1"],
            "DOMAIN": ["DM", "DM"],
            "USUBJID": ["S1-001", "S1-002"],
        }
    )


def test_write_xpt_roundtrip(tmp_path):
    pyreadstat = pytest.importorskip("pyreadstat")
    df = _dm_df()
    out = _write_xpt(df, tmp_path / "dm.xpt", table_name="DM", file_label="Demographics")
    assert out.exists()
    back, meta = pyreadstat.read_xport(str(out))
    assert list(back.columns) == ["STUDYID", "DOMAIN", "USUBJID"]
    assert len(back) == 2
    assert meta.table_name == "DM"


def test_write_xpt_truncates_member_name(tmp_path):
    pyreadstat = pytest.importorskip("pyreadstat")
    df = _dm_df()
    out = _write_xpt(df, tmp_path / "x.xpt", table_name="SUPPLONGNAME")
    _back, meta = pyreadstat.read_xport(str(out))
    assert len(meta.table_name) <= 8
    assert meta.table_name == "SUPPLONG"


def test_write_xpt_accepts_polars(tmp_path):
    pytest.importorskip("pyreadstat")
    pl = pytest.importorskip("polars")
    df = pl.from_pandas(_dm_df())
    out = _write_xpt(df, tmp_path / "dm.xpt", table_name="DM")
    assert out.exists()


def test_materialize_datasets(tmp_path):
    pytest.importorskip("pyreadstat")
    written = _materialize_datasets({"DM": _dm_df(), "AE": _dm_df()}, tmp_path / "mat")
    assert set(written) == {"DM", "AE"}
    assert written["DM"].name == "dm.xpt"
    assert written["AE"].name == "ae.xpt"
    assert all(p.exists() for p in written.values())


# ── CORE command resolution / discovery ────────────────────────────────────────


def test_normalize_version():
    assert _normalize_version("3.4") == "3-4"
    assert _normalize_version("3-4") == "3-4"
    assert _normalize_version("1.1") == "1-1"


def test_resolve_core_command_explicit_str():
    assert _resolve_core_command("core") == ["core"]


def test_resolve_core_command_explicit_sequence():
    assert _resolve_core_command(["python", "core.py"]) == ["python", "core.py"]


def test_resolve_core_command_from_env(monkeypatch):
    monkeypatch.setenv("POINTBLANK_CDISC_CORE", "python /path/core.py")
    assert _resolve_core_command(None) == ["python", "/path/core.py"]


def test_resolve_core_command_not_found(monkeypatch):
    monkeypatch.delenv("POINTBLANK_CDISC_CORE", raising=False)
    # Force PATH discovery to fail
    monkeypatch.setattr("pointblank.metadata._cdisc_core.shutil.which", lambda name: None)
    with pytest.raises(CoreNotFoundError):
        _resolve_core_command(None)


# ── CORE runner (driven by a fake CORE script) ─────────────────────────────────


def test_runner_command_property(tmp_path):
    runner = _fake_runner(tmp_path)
    assert runner.command[0] == sys.executable
    assert runner.command[1].endswith("fakecore.py")


def test_run_validate_produces_and_parses(tmp_path):
    runner = _fake_runner(tmp_path)
    parsed = runner.validate_to_report(
        data_dir=tmp_path,
        standard="sdtmig",
        version="3.4",
        output_stem=tmp_path / "out",
    )
    assert isinstance(parsed, ParsedCoreReport)
    assert parsed.standard == "SDTMIG"
    assert len(parsed.rules) == 12


def test_run_validate_builds_expected_command(tmp_path):
    runner = _fake_runner(tmp_path)
    define = tmp_path / "define.xml"
    define.write_text("<define/>")
    runner.run_validate(
        data_dir=tmp_path / "data",
        standard="sdtmig",
        version="3.4",
        output_stem=tmp_path / "out",
        define_xml=define,
        controlled_terminology=["sdtmct-2024-03-29", "sdtmct-2023-12-15"],
        cache=tmp_path / "cache",
        raw_report=True,
    )
    argv = json.loads((tmp_path / "out.argv.json").read_text())
    # Positional/flag checks
    assert argv[0] == "validate"
    assert "-s" in argv and argv[argv.index("-s") + 1] == "sdtmig"
    assert argv[argv.index("-v") + 1] == "3-4"  # hyphenated
    assert argv[argv.index("-of") + 1] == "JSON"
    assert argv[argv.index("-dxp") + 1] == str(define)
    assert argv[argv.index("-ca") + 1] == str(tmp_path / "cache")
    assert argv.count("-ct") == 2
    assert "-rr" in argv


def test_run_validate_single_ct_string(tmp_path):
    runner = _fake_runner(tmp_path)
    runner.run_validate(
        data_dir=tmp_path,
        standard="sdtmig",
        version="3.4",
        output_stem=tmp_path / "out",
        controlled_terminology="sdtmct-2024-03-29",
    )
    argv = json.loads((tmp_path / "out.argv.json").read_text())
    assert argv.count("-ct") == 1
    assert argv[argv.index("-ct") + 1] == "sdtmct-2024-03-29"


def test_run_validate_unsupported_format(tmp_path):
    runner = _fake_runner(tmp_path)
    with pytest.raises(ValueError):
        runner.run_validate(
            data_dir=tmp_path,
            standard="sdtmig",
            version="3.4",
            output_stem=tmp_path / "out",
            output_format="PDF",
        )


def test_run_validate_nonzero_exit_raises(tmp_path):
    runner = _fake_runner(tmp_path, exit_code=2)
    with pytest.raises(CoreExecutionError):
        runner.run_validate(
            data_dir=tmp_path,
            standard="sdtmig",
            version="3.4",
            output_stem=tmp_path / "out",
        )


def test_run_validate_missing_output_raises(tmp_path):
    # A fake core that exits 0 but writes nothing.
    script = tmp_path / "noop.py"
    script.write_text("import sys; sys.exit(0)")
    runner = _CoreRunner(core=[sys.executable, str(script)])
    with pytest.raises(CoreExecutionError):
        runner.run_validate(
            data_dir=tmp_path,
            standard="sdtmig",
            version="3.4",
            output_stem=tmp_path / "out",
        )


def test_runner_passes_cwd(tmp_path):
    # CORE resolves its bundled resources relative to cwd; verify the runner honors cwd=.
    src = _FIXTURES / "core_report_trimmed.json"
    run_dir = tmp_path / "coreroot"
    run_dir.mkdir()
    script = tmp_path / "cwdcore.py"
    script.write_text(
        "import json, os, shutil, sys\n"
        "args = sys.argv[1:]\n"
        "stem = args[args.index('-o') + 1]\n"
        "open(stem + '.cwd.txt', 'w').write(os.getcwd())\n"
        f"shutil.copy({str(src)!r}, stem + '.json')\n"
    )
    runner = _CoreRunner(core=[sys.executable, str(script)], cwd=run_dir)
    assert runner.cwd == str(run_dir)
    runner.run_validate(
        data_dir=tmp_path,
        standard="sdtmig",
        version="3.4",
        output_stem=tmp_path / "out",
    )
    recorded = (tmp_path / "out.cwd.txt").read_text()
    assert Path(recorded).resolve() == run_dir.resolve()


def test_runner_launch_failure_raises(tmp_path):
    runner = _CoreRunner(core=[str(tmp_path / "does-not-exist-binary")])
    with pytest.raises(CoreExecutionError):
        runner.run_validate(
            data_dir=tmp_path,
            standard="sdtmig",
            version="3.4",
            output_stem=tmp_path / "out",
        )
