"""Native CDISC conformance engine.

Runs bundled JSON rule catalogs against a collection of DataFrames using narwhals
expressions. No subprocesses, no external installs, no API calls at runtime.
"""

from __future__ import annotations

from typing import Any

import narwhals as nw

from pointblank.metadata._conformance.ct import ControlledTerminology
from pointblank.metadata._conformance.evaluator import EvaluationError, evaluate_conditions
from pointblank.metadata._conformance.operations import apply_operations
from pointblank.metadata._conformance.result import (
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_NOT_SUPPORTED,
    STATUS_PASS,
    NativeConformanceResult,
    NativeRowFinding,
    NativeRuleResult,
)
from pointblank.metadata._conformance.rule_loader import NativeRule, RuleLoader

# Rule types handled in Phase 1.
_SUPPORTED_TYPES = {
    "RECORD_CHECK",
    "DATASET_METADATA_CHECK",
    "DOMAIN_PRESENCE_CHECK",
    "DATASET_CONTENTS_CHECK",
}

# Maximum row-level findings to collect per rule (avoids blowing up memory on large datasets).
_MAX_FINDINGS = 100


class NativeConformanceEngine:
    """Run the bundled CDISC rule catalog against a collection of DataFrames.

    Parameters
    ----------
    standard
        The CDISC standard slug (e.g. `"sdtmig"`).
    version
        The standard version (e.g. `"3.4"`).
    ct_packages
        CT package slugs to load (e.g. `["sdtm-ct-2024-09-27"]`). If `None`, the most
        recent bundled CT package is used automatically.
    rule_types
        Optional list of rule types to evaluate. Defaults to all Phase 1 supported types.
    """

    def __init__(
        self,
        standard: str,
        version: str,
        ct_packages: list[str] | None = None,
        rule_types: list[str] | None = None,
    ) -> None:
        self.standard = standard
        self.version = version
        self._rules = RuleLoader.load(standard, version, rule_types=rule_types)
        if ct_packages is None:
            self._ct = ControlledTerminology.load_default()
        else:
            self._ct = ControlledTerminology.load(ct_packages)

    @property
    def ct_packages(self) -> list[str]:
        return self._ct.packages

    def run(self, datasets: dict[str, Any]) -> NativeConformanceResult:
        """Evaluate all rules against `datasets`.

        Parameters
        ----------
        datasets
            Mapping of domain name (e.g. `"DM"`) to a Pandas or Polars DataFrame.

        Returns
        -------
        NativeConformanceResult
        """
        nw_datasets: dict[str, nw.DataFrame] = {
            k.upper(): nw.from_native(v, eager_only=True) for k, v in datasets.items()
        }
        results: list[NativeRuleResult] = []
        for rule in self._rules:
            result = self._evaluate_rule(rule, nw_datasets)
            results.append(result)
        return NativeConformanceResult(
            standard=self.standard,
            version=self.version,
            ct_packages=self.ct_packages,
            rule_results=results,
        )

    # ── Rule dispatch ─────────────────────────────────────────────────────────

    def _evaluate_rule(
        self, rule: NativeRule, datasets: dict[str, nw.DataFrame]
    ) -> NativeRuleResult:
        if rule.rule_type not in _SUPPORTED_TYPES:
            return NativeRuleResult(
                rule_id=rule.core_id,
                rule_type=rule.rule_type,
                dataset="",
                status=STATUS_NOT_SUPPORTED,
                sensitivity=rule.sensitivity,
                description=rule.description,
            )

        handler = {
            "RECORD_CHECK": self._record_check,
            "DATASET_METADATA_CHECK": self._dataset_metadata_check,
            "DOMAIN_PRESENCE_CHECK": self._domain_presence_check,
            "DATASET_CONTENTS_CHECK": self._dataset_contents_check,
        }[rule.rule_type]

        try:
            return handler(rule, datasets)
        except Exception as exc:
            # Determine which domain the rule targets (best effort).
            domain = rule.domains[0] if rule.domains else ""
            return NativeRuleResult(
                rule_id=rule.core_id,
                rule_type=rule.rule_type,
                dataset=domain,
                status=STATUS_ERROR,
                sensitivity=rule.sensitivity,
                description=rule.description,
                message=str(exc),
            )

    # ── Rule type handlers ────────────────────────────────────────────────────

    def _record_check(
        self, rule: NativeRule, datasets: dict[str, nw.DataFrame]
    ) -> NativeRuleResult:
        """Per-row check: find rows where the condition tree evaluates to True (= violation)."""
        target_domains = rule.domains or list(datasets.keys())
        all_findings: list[NativeRowFinding] = []
        n_issues = 0

        for domain in target_domains:
            df = datasets.get(domain.upper())
            if df is None:
                continue
            df = apply_operations(df, rule.operations, self._ct, datasets)
            try:
                mask = evaluate_conditions(df, rule.conditions)
            except EvaluationError:
                continue
            failing_rows = [i for i, v in enumerate(mask.to_list()) if v]
            n_issues += len(failing_rows)
            for row_idx in failing_rows[:_MAX_FINDINGS]:
                variables = df.columns[:5]
                values = [str(df[c][row_idx]) for c in variables]
                all_findings.append(
                    NativeRowFinding(
                        rule_id=rule.core_id,
                        dataset=domain,
                        row=row_idx,
                        variables=variables,
                        values=values,
                        message=rule.message,
                    )
                )

        domain_label = ", ".join(target_domains) if target_domains else ""
        return NativeRuleResult(
            rule_id=rule.core_id,
            rule_type=rule.rule_type,
            dataset=domain_label,
            status=STATUS_FAIL if n_issues > 0 else STATUS_PASS,
            sensitivity=rule.sensitivity,
            description=rule.description,
            message=rule.message if n_issues > 0 else None,
            n_issues=n_issues,
            row_findings=all_findings,
        )

    def _dataset_contents_check(
        self, rule: NativeRule, datasets: dict[str, nw.DataFrame]
    ) -> NativeRuleResult:
        """Dataset-level value constraint check.

        Like RECORD_CHECK but the result is at dataset granularity (not per-row).
        """
        return self._record_check(rule, datasets)

    def _dataset_metadata_check(
        self, rule: NativeRule, datasets: dict[str, nw.DataFrame]
    ) -> NativeRuleResult:
        """Dataset-level metadata check (column presence, sort keys, label, etc.).

        Conditions reference computed columns added by operations (e.g. `$USUBJID_present`).
        """
        target_domains = rule.domains or list(datasets.keys())
        n_issues = 0
        first_failing_domain = ""

        for domain in target_domains:
            df = datasets.get(domain.upper())
            if df is None:
                continue
            # Operations add the computed columns that conditions reference.
            df = apply_operations(df, rule.operations, self._ct, datasets)
            # For metadata checks the condition is evaluated once against a single-row summary
            # DataFrame (all computed columns).  If any operation added a False column, the
            # condition fires.
            try:
                mask = evaluate_conditions(df, rule.conditions)
            except EvaluationError:
                continue
            if any(mask.to_list()):
                n_issues += 1
                if not first_failing_domain:
                    first_failing_domain = domain

        return NativeRuleResult(
            rule_id=rule.core_id,
            rule_type=rule.rule_type,
            dataset=first_failing_domain or (target_domains[0] if target_domains else ""),
            status=STATUS_FAIL if n_issues > 0 else STATUS_PASS,
            sensitivity=rule.sensitivity,
            description=rule.description,
            message=rule.message if n_issues > 0 else None,
            n_issues=n_issues,
        )

    def _domain_presence_check(
        self, rule: NativeRule, datasets: dict[str, nw.DataFrame]
    ) -> NativeRuleResult:
        """Check that a required domain is present (or that a prohibited domain is absent)."""
        params = rule.actions.get("params", {})
        required_domains: list[str] = params.get("required_domains", [])
        prohibited_domains: list[str] = params.get("prohibited_domains", [])
        present_domains = set(datasets.keys())

        missing = [d for d in required_domains if d.upper() not in present_domains]
        found_prohibited = [d for d in prohibited_domains if d.upper() in present_domains]

        issues = missing + found_prohibited
        n_issues = len(issues)
        message: str | None = None
        if missing:
            message = f"Required domain(s) missing: {', '.join(missing)}"
        elif found_prohibited:
            message = f"Prohibited domain(s) present: {', '.join(found_prohibited)}"

        domain_label = ", ".join(required_domains + prohibited_domains) or ""
        return NativeRuleResult(
            rule_id=rule.core_id,
            rule_type=rule.rule_type,
            dataset=domain_label,
            status=STATUS_FAIL if n_issues > 0 else STATUS_PASS,
            sensitivity=rule.sensitivity,
            description=rule.description,
            message=message,
            n_issues=n_issues,
        )
