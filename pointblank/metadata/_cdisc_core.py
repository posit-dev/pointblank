"""Parser for CDISC CORE engine JSON reports (PLAN_06 Phase 2, Path A).

Pointblank wraps the open-source CDISC CORE engine (`cdisc-rules-engine`) as an external process:
datasets and Define-XML are handed to CORE, it runs the authoritative conformance rule set, and its
JSON report is parsed back into Pointblank's [`ConformanceReport`](`pointblank.ConformanceReport`).

This module implements the *parsing* half — turning CORE's JSON report into typed objects. The
subprocess runner and dataset materialization live elsewhere. The parser is written against the
JSON schema emitted by `core validate -of JSON` (verified against CORE 0.16.0):

- `Conformance_Details` — run provenance (standard, version, CT version, engine version, runtime).
- `Dataset_Details` — one entry per validated dataset (filename, label, path, size, row count).
- `Issue_Summary` — one entry per (dataset, rule) that reported issues, with a count.
- `Issue_Details` — row-level findings (rule id, message, dataset, USUBJID, row, variables, values).
- `Rules_Report` — the run status of every rule (`SKIPPED` / `SUCCESS` / `ISSUE REPORTED` /
  `EXECUTION ERROR`).

Note: the CORE JSON report carries **no severity field** (no Reject/Error/Warning/Notice). Pass/fail
gating therefore keys off rule *status*, not severity.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import Any

__all__ = [
    "CoreFinding",
    "CoreRuleResult",
    "CoreIssueSummary",
    "ParsedCoreReport",
    "parse_core_report",
    "STATUS_SUCCESS",
    "STATUS_SKIPPED",
    "STATUS_ISSUE",
    "STATUS_ERROR",
]

# Rule run-status values emitted by CORE in the `Rules_Report` section.
STATUS_SUCCESS = "SUCCESS"  # rule ran, data conformed
STATUS_SKIPPED = "SKIPPED"  # rule not applicable / required data absent
STATUS_ISSUE = "ISSUE REPORTED"  # rule ran, conformance issue(s) found
STATUS_ERROR = "EXECUTION ERROR"  # rule failed to execute

# Statuses that constitute a conformance failure (used for pass/fail gating).
_FAILING_STATUSES = frozenset({STATUS_ISSUE, STATUS_ERROR})


@dataclass
class CoreFinding:
    """A single row-level conformance finding from CORE's `Issue_Details`.

    Parameters
    ----------
    rule_id
        The CORE rule identifier (e.g., `"CORE-000357"`).
    message
        The human-readable rule message describing the issue.
    dataset
        The dataset (domain) the finding was raised against.
    executability
        CORE's executability note for the rule (e.g., `"fully executable"`).
    usubjid
        The `USUBJID` of the offending record, if applicable (may be empty).
    row
        The 1-based row number of the offending record, if applicable.
    seq
        The `--SEQ` value of the offending record, if applicable (may be empty).
    variables
        The variable name(s) implicated in the finding.
    values
        The offending value(s) corresponding to `variables`.
    """

    rule_id: str
    message: str
    dataset: str
    executability: str | None = None
    usubjid: str | None = None
    row: int | None = None
    seq: str | None = None
    variables: list[Any] = dataclass_field(default_factory=list)
    values: list[Any] = dataclass_field(default_factory=list)


@dataclass
class CoreIssueSummary:
    """A per-(dataset, rule) issue count from CORE's `Issue_Summary`.

    Parameters
    ----------
    dataset
        The dataset the issues were raised against (or `"STUDY"` for study-level issues).
    rule_id
        The CORE rule identifier.
    message
        The rule message.
    issues
        The number of issues reported for this (dataset, rule) pair.
    """

    dataset: str
    rule_id: str
    message: str
    issues: int


@dataclass
class CoreRuleResult:
    """The run status of a single rule from CORE's `Rules_Report`.

    Parameters
    ----------
    rule_id
        The CORE rule identifier (e.g., `"CORE-000001"`).
    status
        The run status: `"SUCCESS"`, `"SKIPPED"`, `"ISSUE REPORTED"`, or `"EXECUTION ERROR"`.
    message
        The rule message.
    version
        The rule version.
    cdisc_rule_id
        The corresponding CDISC rule identifier(s) (e.g., `"CG0176, TIG0405"`).
    fda_rule_id
        The corresponding FDA rule identifier(s), if any.
    """

    rule_id: str
    status: str
    message: str | None = None
    version: str | None = None
    cdisc_rule_id: str | None = None
    fda_rule_id: str | None = None

    @property
    def is_failing(self) -> bool:
        """Whether this rule's status constitutes a conformance failure."""
        return self.status in _FAILING_STATUSES


@dataclass
class ParsedCoreReport:
    """A parsed CDISC CORE JSON report.

    Parameters
    ----------
    details
        The `Conformance_Details` block (run provenance) as a plain dict.
    datasets
        The `Dataset_Details` entries as plain dicts.
    issue_summary
        Per-(dataset, rule) issue counts.
    findings
        Row-level findings.
    rules
        Per-rule run results.
    """

    details: dict[str, Any] = dataclass_field(default_factory=dict)
    datasets: list[dict[str, Any]] = dataclass_field(default_factory=list)
    issue_summary: list[CoreIssueSummary] = dataclass_field(default_factory=list)
    findings: list[CoreFinding] = dataclass_field(default_factory=list)
    rules: list[CoreRuleResult] = dataclass_field(default_factory=list)

    # ── Derived views ────────────────────────────────────────────────────────

    @property
    def standard(self) -> str | None:
        """The CDISC standard the run validated against (e.g., `"SDTMIG"`)."""
        return self.details.get("Standard")

    @property
    def version(self) -> str | None:
        """The standard version the run validated against (e.g., `"V3.4"`)."""
        return self.details.get("Version")

    @property
    def engine_version(self) -> str | None:
        """The CORE engine version that produced the report."""
        return self.details.get("CORE_Engine_Version")

    @property
    def n_total_issues(self) -> int:
        """Total number of issues across all (dataset, rule) pairs."""
        return sum(s.issues for s in self.issue_summary)

    def status_counts(self) -> dict[str, int]:
        """Count rules by run status."""
        counts: dict[str, int] = {}
        for r in self.rules:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def failing_rules(self) -> list[CoreRuleResult]:
        """The rules whose status constitutes a conformance failure."""
        return [r for r in self.rules if r.is_failing]

    @property
    def all_passed(self) -> bool:
        """Whether the run reported no conformance failures.

        A run passes when no rule has status `ISSUE REPORTED` or `EXECUTION ERROR`. If the report
        contains no `Rules_Report` (older/minimal reports), this falls back to the absence of any
        entries in `Issue_Summary`.
        """
        if self.rules:
            return not any(r.is_failing for r in self.rules)
        return len(self.issue_summary) == 0


def _as_int(value: Any) -> int | None:
    """Coerce a value to int, returning None for empty/unparseable input."""
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _empty_to_none(value: Any) -> Any:
    """Normalize CORE's empty-string sentinels to None."""
    return None if value == "" else value


def parse_core_report(report: dict[str, Any]) -> ParsedCoreReport:
    """Parse a CDISC CORE JSON report into a [`ParsedCoreReport`](`ParsedCoreReport`).

    Parameters
    ----------
    report
        The report as a `dict` (from `json.load` of CORE's `-of JSON` output). Both the standard
        report and the `--raw-report` variant are accepted; the extra `results_data` key in the raw
        variant is ignored.

    Returns
    -------
    ParsedCoreReport
        The parsed report with typed findings, per-rule results, and run provenance.

    Raises
    ------
    TypeError
        If `report` is not a dict.
    ValueError
        If `report` does not look like a CORE report (none of the expected sections present).
    """
    if not isinstance(report, dict):
        raise TypeError(f"Expected a CORE report dict, got {type(report).__name__}.")

    expected = {"Conformance_Details", "Issue_Details", "Rules_Report", "Issue_Summary"}
    if not expected & set(report.keys()):
        raise ValueError(
            "Input does not look like a CDISC CORE JSON report; expected one of "
            f"{sorted(expected)} among top-level keys, got {sorted(report.keys())}."
        )

    details = report.get("Conformance_Details") or {}
    datasets = list(report.get("Dataset_Details") or [])

    issue_summary = [
        CoreIssueSummary(
            dataset=item.get("dataset", ""),
            rule_id=item.get("core_id", ""),
            message=item.get("message", ""),
            issues=_as_int(item.get("issues")) or 0,
        )
        for item in (report.get("Issue_Summary") or [])
    ]

    findings = [
        CoreFinding(
            rule_id=item.get("core_id", ""),
            message=item.get("message", ""),
            dataset=item.get("dataset", ""),
            executability=_empty_to_none(item.get("executability")),
            usubjid=_empty_to_none(item.get("USUBJID")),
            row=_as_int(item.get("row")),
            seq=_empty_to_none(item.get("SEQ")),
            variables=list(item.get("variables") or []),
            values=list(item.get("values") or []),
        )
        for item in (report.get("Issue_Details") or [])
    ]

    rules = [
        CoreRuleResult(
            rule_id=item.get("core_id", ""),
            status=item.get("status", ""),
            message=_empty_to_none(item.get("message")),
            version=_empty_to_none(item.get("version")),
            cdisc_rule_id=_empty_to_none(item.get("cdisc_rule_id")),
            fda_rule_id=_empty_to_none(item.get("fda_rule_id")),
        )
        for item in (report.get("Rules_Report") or [])
    ]

    return ParsedCoreReport(
        details=details,
        datasets=datasets,
        issue_summary=issue_summary,
        findings=findings,
        rules=rules,
    )
