"""Result dataclasses for the native conformance engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Rule execution statuses
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_ERROR = "error"
STATUS_NOT_APPLICABLE = "not_applicable"
STATUS_NOT_SUPPORTED = "not_supported"


@dataclass
class NativeRowFinding:
    """A single row-level finding produced by a rule."""

    rule_id: str
    dataset: str
    row: int | None
    variables: list[str]
    values: list[Any]
    message: str


@dataclass
class NativeRuleResult:
    """The outcome of evaluating one rule against the dataset collection."""

    rule_id: str
    rule_type: str
    dataset: str
    status: str
    sensitivity: str = "Error"
    description: str = ""
    message: str | None = None
    n_issues: int = 0
    row_findings: list[NativeRowFinding] = field(default_factory=list)


@dataclass
class NativeConformanceResult:
    """Aggregated results of a native conformance run."""

    standard: str
    version: str
    ct_packages: list[str]
    rule_results: list[NativeRuleResult]

    @property
    def all_passed(self) -> bool:
        return not any(r.status == STATUS_FAIL for r in self.rule_results)

    @property
    def n_total_issues(self) -> int:
        return sum(r.n_issues for r in self.rule_results)

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for r in self.rule_results:
            counts[r.status] = counts.get(r.status, 0) + 1
        return counts

    def rules(self, status: str | None = None) -> list[NativeRuleResult]:
        if status is None:
            return list(self.rule_results)
        return [r for r in self.rule_results if r.status == status]

    def findings(self) -> list[NativeRowFinding]:
        out: list[NativeRowFinding] = []
        for r in self.rule_results:
            out.extend(r.row_findings)
        return out

    def issues(self) -> list[dict]:
        out: list[dict] = []
        for r in self.rule_results:
            if r.n_issues > 0:
                out.append(
                    {
                        "dataset": r.dataset,
                        "rule_id": r.rule_id,
                        "rule_type": r.rule_type,
                        "message": r.message or r.description,
                        "n_issues": r.n_issues,
                        "sensitivity": r.sensitivity,
                        "status": r.status,
                    }
                )
        return out
