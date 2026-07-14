"""Smoke tests against the CDISC CDISCPILOT01 public reference dataset.

CDISCPILOT01 is the canonical public SDTM reference submission maintained by CDISC at:

  https://github.com/cdisc-org/sdtm-adam-pilot-project

Running these tests provides validation evidence that the Pointblank native conformance engine
produces sensible, correctly-scoped results on real SDTM data. This is an important bit of assurance
for pharmaceutical users.

How to obtain the data
----------------------
Download or clone the CDISC pilot repository and point CDISCPILOT01_PATH at the XPT folder:

    git clone https://github.com/cdisc-org/sdtm-adam-pilot-project
    export CDISCPILOT01_PATH=sdtm-adam-pilot-project/updated-pilot-submission-package/900172/m5/datasets/cdiscpilot01/tabulations/sdtm

Or set the variable directly when running pytest:

    CDISCPILOT01_PATH=/path/to/sdtm pytest tests/test_cdiscpilot01_smoke.py -v

These tests are skipped automatically when CDISCPILOT01_PATH is not set.

Validated against: SDTMIG 3.4 conformance catalog, 426 rules, CT 2024-09-27.
Study: CDISCPILOT01, Alzheimer's disease study, 306 subjects, 18 SDTM + 4 SUPP-- domains.
"""

from __future__ import annotations

import os
import pathlib
from collections import Counter

import pytest

# ── Path resolution ────────────────────────────────────────────────────────────

_PILOT_PATH_ENV = "CDISCPILOT01_PATH"
_PILOT_FALLBACK = pathlib.Path("/tmp/cdiscpilot01")

_PILOT_DOMAIN_FILES = ["dm", "ae", "cm", "ex", "lb", "vs"]  # minimum for a meaningful run

_pilot_path: pathlib.Path | None = None
if env := os.getenv(_PILOT_PATH_ENV):
    _pilot_path = pathlib.Path(env)
elif _PILOT_FALLBACK.exists() and any(_PILOT_FALLBACK.glob("dm.xpt")):
    _pilot_path = _PILOT_FALLBACK

_DATA_AVAILABLE = _pilot_path is not None and all(
    (_pilot_path / f"{d}.xpt").exists() for d in _PILOT_DOMAIN_FILES
)

pytestmark = pytest.mark.skipif(
    not _DATA_AVAILABLE,
    reason=(
        "CDISCPILOT01 XPT files not found. "
        f"Set {_PILOT_PATH_ENV}=/path/to/sdtm to enable these tests."
    ),
)


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def pilot_datasets():
    """Load all available CDISCPILOT01 XPT datasets into a {NAME: DataFrame} dict."""
    try:
        import pyreadstat
        import polars as pl
    except ImportError:
        pytest.skip("pyreadstat and polars are required for CDISCPILOT01 tests.")

    assert _pilot_path is not None
    datasets: dict = {}

    # Standard SDTM domains present in the pilot
    sdtm_domains = [
        "dm",
        "ae",
        "cm",
        "ex",
        "lb",
        "vs",
        "mh",
        "ds",
        "sv",
        "sc",
        "qs",
        "ta",
        "ti",
        "te",
        "tv",
        "eg",
        "fa",
    ]
    for name in sdtm_domains:
        f = _pilot_path / f"{name}.xpt"
        if f.exists():
            kw = {"encoding": "cp1252"} if name == "ts" else {}
            df_pd, _ = pyreadstat.read_xport(str(f), **kw)
            datasets[name.upper()] = pl.from_pandas(df_pd)

    # Trial summary (TS) — needs cp1252 encoding
    ts_f = _pilot_path / "ts.xpt"
    if ts_f.exists():
        df_pd, _ = pyreadstat.read_xport(str(ts_f), encoding="cp1252")
        datasets["TS"] = pl.from_pandas(df_pd)

    # SUPP-- datasets
    supp_names = ["suppdm", "suppae", "supplb", "suppvs", "suppeg", "suppcm", "suppds", "suppmh"]
    for name in supp_names:
        f = _pilot_path / f"{name}.xpt"
        if f.exists():
            df_pd, _ = pyreadstat.read_xport(str(f))
            datasets[name.upper()] = pl.from_pandas(df_pd)

    return datasets


@pytest.fixture(scope="module")
def pilot_report(pilot_datasets):
    """Run the native SDTMIG 3.4 engine against the full pilot and return the ConformanceReport."""
    import pointblank as pb

    return pb.validate_sdtmig(pilot_datasets)


# ── Basic result structure ─────────────────────────────────────────────────────


def test_report_is_rules_type(pilot_report):
    assert pilot_report.is_rules
    assert not pilot_report.is_core


def test_report_has_expected_rule_count(pilot_report):
    """426 rules in the SDTMIG 3.4 catalog after deduplication (as of 2025-07-14)."""
    assert len(pilot_report.native_result.rule_results) == 426


def test_report_overall_pass_rate(pilot_report):
    """Expect >= 93% pass rate on the reference pilot dataset."""
    results = pilot_report.native_result.rule_results
    passed = sum(1 for r in results if r.status == "pass")
    total_executed = sum(1 for r in results if r.status in ("pass", "fail"))
    pass_rate = passed / total_executed if total_executed else 0
    assert pass_rate >= 0.93, (
        f"Pass rate {pass_rate:.1%} is below 93% — check for regressions or CT expansion gaps."
    )


def test_report_total_issues_is_bounded(pilot_report):
    """Total issue count must stay well under 5,000 (pre-fix it was 127,357)."""
    total = sum(r.n_issues for r in pilot_report.native_result.rule_results)
    assert total < 5_000, (
        f"Total issues = {total:,}. "
        "This may indicate a regression in empty-string handling or duplicate rules."
    )


def test_no_duplicate_rule_ids(pilot_report):
    ids = [r.rule_id for r in pilot_report.native_result.rule_results]
    counts = Counter(ids)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    assert not duplicates, f"Duplicate rule IDs in results: {duplicates}"


# ── Domain coverage ────────────────────────────────────────────────────────────


def test_dm_domain_loaded(pilot_datasets):
    assert "DM" in pilot_datasets
    assert len(pilot_datasets["DM"]) == 306  # 306 subjects in the pilot


def test_core_domains_present(pilot_datasets):
    for domain in ["DM", "AE", "CM", "EX", "LB", "VS", "TA", "TS"]:
        assert domain in pilot_datasets, f"{domain} not loaded"


# ── Known legitimate findings in CDISCPILOT01 ─────────────────────────────────
#
# CDISCPILOT01 was created against SDTMIG 3.1.2 with CT from circa 2012.  The
# findings below reflect genuine differences between the pilot and current
# SDTMIG 3.4 / CT-2024-09-27, NOT bugs in the conformance engine.


def test_aerel_finding_is_present(pilot_report):
    """AEREL uses old CT terms (PROBABLE/POSSIBLE/REMOTE/NONE); current CT expects longer phrases.

    This is an expected, legitimate finding — the pilot predates the current AEREL codelist.
    If this test starts failing (0 issues) the AEREL rule may have been accidentally relaxed.
    """
    aerel = next(
        (r for r in pilot_report.native_result.rule_results if r.rule_id == "SDTM-135"),
        None,
    )
    assert aerel is not None, "SDTM-135 (AEREL CT check) not found in results"
    assert aerel.n_issues > 0, (
        "Expected AEREL findings against CDISCPILOT01 — "
        "old CT terms should not be silently accepted."
    )
    # Upper bound: all 1,191 AE records
    assert aerel.n_issues <= 1_200


def test_visitnum_in_ae_finding(pilot_report):
    """VISITNUM is absent from the pilot AE domain (it's optional in SDTM AE events)."""
    rule = next(
        (r for r in pilot_report.native_result.rule_results if r.rule_id == "SDTM-162"),
        None,
    )
    assert rule is not None
    assert rule.n_issues == 1  # one dataset (AE) fails the column-presence check


def test_lbornrlo_type_finding(pilot_report):
    """LBORNRLO is character in the pilot (stores 'SEE TEXT'); SDTMIG requires numeric type."""
    rule = next(
        (r for r in pilot_report.native_result.rule_results if r.rule_id == "SDTM-170"),
        None,
    )
    assert rule is not None
    assert rule.n_issues >= 1


# ── Engine correctness: empty-string handling ──────────────────────────────────


def test_lbblfl_false_positives_eliminated(pilot_report):
    """After SAS empty-string fix, LBBLFL (baseline flag) must not produce issues.

    Pre-fix: 50,347 issues (SAS missing values read as '' not null).
    Post-fix: 0 issues (empty strings treated as missing → skip codelist check).
    """
    rule = next(
        (r for r in pilot_report.native_result.rule_results if r.rule_id == "SDTM-095"),
        None,
    )
    assert rule is not None, "SDTM-095 (LBBLFL NY check) not found"
    assert rule.n_issues == 0, (
        f"SDTM-095 has {rule.n_issues} issues — SAS empty-string handling may be broken."
    )


def test_vsblfl_false_positives_eliminated(pilot_report):
    """VSBLFL baseline flag produces 0 issues after SAS empty-string fix (pre-fix: 26,860)."""
    rule = next(
        (r for r in pilot_report.native_result.rule_results if r.rule_id == "SDTM-093"),
        None,
    )
    assert rule is not None
    assert rule.n_issues == 0, (
        f"SDTM-093 has {rule.n_issues} issues — SAS empty-string handling may be broken."
    )


def test_rficdtc_false_positives_eliminated(pilot_report):
    """RFICDTC ISO 8601 check produces 0 issues (pre-fix: 306 issues from '' empty strings)."""
    rule = next(
        (r for r in pilot_report.native_result.rule_results if r.rule_id == "SDTM-193"),
        None,
    )
    # Rule may be not_applicable if RFICDTC column is absent; either way it must not produce
    # false positives from empty strings.
    if rule is not None and rule.status != "not_applicable":
        assert rule.n_issues == 0, (
            f"SDTM-193 has {rule.n_issues} issues — check SAS empty-string handling."
        )


# ── Engine correctness: case-insensitive CT comparison ────────────────────────


def test_epoch_ct_case_insensitive(pilot_report):
    """TA EPOCH values ('Screening', 'Treatment') must match the uppercase EPOCH codelist.

    Pre-fix: 8 issues (case-sensitive comparison failed 'Screening' vs 'SCREENING').
    Post-fix: 0 issues.
    """
    rule = next(
        (r for r in pilot_report.native_result.rule_results if r.rule_id == "SDTM-434"),
        None,
    )
    if rule is not None and rule.status != "not_applicable":
        assert rule.n_issues == 0, (
            f"SDTM-434 has {rule.n_issues} EPOCH CT issues — "
            "case-insensitive CT comparison may be broken."
        )


def test_vsstresu_case_insensitive(pilot_report):
    """VSSTRESU 'BEATS/MIN' must match codelist 'beats/min' via case-insensitive comparison."""
    rule = next(
        (r for r in pilot_report.native_result.rule_results if r.rule_id == "SDTM-092"),
        None,
    )
    if rule is not None and rule.status != "not_applicable":
        assert rule.n_issues == 0, (
            f"SDTM-092 has {rule.n_issues} VSSTRESU issues — "
            "case-insensitive CT comparison may be broken."
        )


# ── Engine correctness: SUPP-- excluded from catch-all rules ──────────────────


def test_domain_column_presence_excludes_supp(pilot_report):
    """SDTM-029 (DOMAIN column required) must not fire on SUPP-- datasets.

    SUPP-- datasets use RDOMAIN, not DOMAIN. Pre-fix: 4 issues (one per SUPP-- dataset).
    Post-fix: 0 issues.
    """
    rule = next(
        (r for r in pilot_report.native_result.rule_results if r.rule_id == "SDTM-029"),
        None,
    )
    assert rule is not None
    assert rule.n_issues == 0, (
        f"SDTM-029 has {rule.n_issues} issues — SUPP-- datasets are being incorrectly "
        "included in catch-all domain iteration."
    )


# ── get_tabular_report rendering ──────────────────────────────────────────────


def test_get_tabular_report_returns_gt(pilot_report):
    """get_tabular_report() must return a Great Tables GT object."""
    try:
        from great_tables import GT
    except ImportError:
        pytest.skip("great_tables not installed")
    gt = pilot_report.get_tabular_report()
    assert isinstance(gt, GT)


def test_repr_html_non_empty(pilot_report):
    html = pilot_report._repr_html_()
    assert html and len(html) > 500
