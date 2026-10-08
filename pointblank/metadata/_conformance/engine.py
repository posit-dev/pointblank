"""Built-in CDISC conformance engine.

Runs bundled JSON rule catalogs against a collection of DataFrames using narwhals
expressions. No subprocesses, no external installs, no API calls at runtime.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import narwhals as nw

if TYPE_CHECKING:
    from pointblank.metadata._types import MetadataImport, MetadataPackage

from pointblank.metadata._conformance.ct import ControlledTerminology
from pointblank.metadata._conformance.evaluator import EvaluationError, evaluate_conditions
from pointblank.metadata._conformance.operations import apply_operations
from pointblank.metadata._conformance.result import (
    STATUS_ERROR,
    STATUS_FAIL,
    STATUS_NOT_APPLICABLE,
    STATUS_NOT_SUPPORTED,
    STATUS_PASS,
    NativeConformanceResult,
    NativeRowFinding,
    NativeRuleResult,
)
from pointblank.metadata._conformance.rule_loader import NativeRule, RuleLoader

# Rule types handled natively
_SUPPORTED_TYPES = {
    "RECORD_CHECK",
    "DATASET_METADATA_CHECK",
    "DOMAIN_PRESENCE_CHECK",
    "DATASET_CONTENTS_CHECK",
    "VARIABLE_METADATA_CHECK",
    "DEFINE_ITEM_METADATA_CHECK",
    "DEFINE_CODELIST_CHECK",
}

# SUPP-- and RELREC use RDOMAIN instead of DOMAIN and have a fixed non-standard structure.
# Catch-all rules (domains: []) must not automatically apply to them.
_STRUCTURAL_DATASETS = frozenset({"RELREC"})

# Maximum row-level findings to collect per rule (avoids blowing up memory on large datasets)
_MAX_FINDINGS = 100

# Columns checked as candidate identifiers when building a row finding (in priority order).
# The first one that's present in the dataset is included in `context`.
_CONTEXT_CANDIDATES = [
    "STUDYID",
    "DOMAIN",
    "SUBJID",
    "VISITNUM",
    "VISIT",
    "EPOCH",
    "AESEQ",
    "CMSEQ",
    "LBSEQ",
    "VSSEQ",
    "EXSEQ",
    "MHSEQ",
    "DSSEQ",
    "EGSEQ",
    "AETERM",
    "CMTRT",
    "LBTESTCD",
    "VSTESTCD",
    "EGTESTCD",
]


def _condition_columns(conditions: dict) -> list[str]:
    """Return all column names referenced in a conditions tree, depth-first."""
    cols: list[str] = []
    for key in ("all", "any"):
        for sub in conditions.get(key, []):
            cols.extend(_condition_columns(sub))
    name = conditions.get("name")
    if name:
        cols.append(name)
    return cols


def _build_row_finding(
    df: nw.DataFrame,
    row_idx: int,
    domain: str,
    operations: list[dict],
    conditions: dict,
    rule_id: str,
    message: str | None,
) -> "NativeRowFinding":
    """Build a NativeRowFinding with smart column selection.

    Captures USUBJID, the primary rule-checked column and its value, and a small set of
    identifying context columns (STUDYID, VISITNUM, test-code, etc.).
    """
    cols = set(df.columns)

    # USUBJID — the most important identifier for QC workflows
    raw_id = df["USUBJID"][row_idx] if "USUBJID" in cols else None
    usubjid = str(raw_id) if raw_id is not None else None

    # Primary checked column: first operation that names a column present in the dataset;
    # fall back to columns referenced directly in the conditions tree (e.g. rules with no ops).
    checked_col: str | None = None
    for op in operations:
        col = op.get("params", {}).get("column")
        if col and col in cols:
            checked_col = col
            break
    if checked_col is None:
        for col in _condition_columns(conditions):
            if col in cols:
                checked_col = col
                break

    checked_val: str | None = None
    if checked_col:
        raw = df[checked_col][row_idx]
        checked_val = str(raw) if raw is not None else ""

    # Context: a small set of identifying columns (excluding USUBJID and checked_col)
    context: dict[str, str] = {}
    for c in _CONTEXT_CANDIDATES:
        if c in cols and c != "USUBJID" and c != checked_col:
            raw = df[c][row_idx]
            if raw is not None:
                s = str(raw)
                if s and s != "None":
                    context[c] = s

    return NativeRowFinding(
        rule_id=rule_id,
        dataset=domain,
        row=row_idx,
        usubjid=usubjid,
        checked_column=checked_col,
        checked_value=checked_val,
        context=context,
        message=message,
    )


class NativeConformanceEngine:
    """Evaluate a bundled CDISC rule catalog against a collection of DataFrames.

    This is the low-level engine that powers [`validate_sdtmig()`](`pointblank.validate_sdtmig`).
    Most users should call that convenience function rather than instantiating this class
    directly. Use `NativeConformanceEngine` when you need fine-grained control over which rule
    types to run, or when integrating Pointblank's conformance engine into a larger pipeline.

    The engine loads the rule catalog for the requested standard and version from the bundled
    JSON files shipped with Pointblank, then evaluates each rule against the supplied datasets
    using narwhals expressions. No subprocesses, network calls, or external CDISC tools are
    involved.

    Supported rule types
    --------------------
    - ``RECORD_CHECK`` — per-row value checks; failing rows are collected as `NativeRowFinding`
      objects (up to 100 per rule).
    - ``VARIABLE_METADATA_CHECK`` — variable presence and column ordering.
    - ``DATASET_METADATA_CHECK`` — dataset-level attributes (sort keys, required sort order).
    - ``DATASET_CONTENTS_CHECK`` — dataset-level value constraints evaluated row-by-row.
    - ``DOMAIN_PRESENCE_CHECK`` — required or prohibited domain presence.
    - ``DEFINE_ITEM_METADATA_CHECK`` — variable declarations against Define-XML metadata.
    - ``DEFINE_CODELIST_CHECK`` — codelist values against Define-XML declarations.

    Parameters
    ----------
    standard
        CDISC standard slug (e.g., ``"sdtmig"``).
    version
        Standard version string (e.g., ``"3-4"``).
    ct_packages
        CT package slug(s) to load (e.g., ``["sdtm-ct-2024-09-27"]``). When ``None`` the
        most recent bundled CT package is loaded automatically.
    rule_types
        Optional allowlist of rule types to evaluate. When ``None`` all supported types are
        run. Pass a subset (e.g., ``["RECORD_CHECK"]``) to restrict the run.
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
        self._define_pkg: MetadataPackage | None = None

    @property
    def ct_packages(self) -> list[str]:
        return self._ct.packages

    def run(
        self,
        datasets: dict[str, Any],
        define_xml: Any = None,
    ) -> NativeConformanceResult:
        """Evaluate all rules against `datasets`.

        Parameters
        ----------
        datasets
            Mapping of domain name (e.g. `"DM"`) to a Pandas or Polars DataFrame.
        define_xml
            Optional Define-XML metadata. Accepted forms:

            * A file path (`str` or `pathlib.Path`): the file is parsed automatically.
            * A `MetadataPackage` object (already parsed).
            * A `MetadataImport` object (single domain).
            * `None`: no Define-XML; rules that require it return `STATUS_NOT_APPLICABLE`.

        Returns
        -------
        NativeConformanceResult
        """
        nw_datasets: dict[str, nw.DataFrame] = {
            k.upper(): nw.from_native(v, eager_only=True) for k, v in datasets.items()
        }
        self._define_pkg: MetadataPackage | None = self._load_define_xml(define_xml)
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

    def _load_define_xml(self, define_xml: Any) -> MetadataPackage | None:
        if define_xml is None:
            return None
        from pointblank.metadata._types import MetadataImport, MetadataPackage

        if isinstance(define_xml, MetadataPackage):
            return define_xml
        if isinstance(define_xml, MetadataImport):
            domain = (define_xml.domain or define_xml.dataset_name or "").upper()
            pkg = MetadataPackage()
            pkg.items[domain] = define_xml
            return pkg
        # Assume path
        try:
            from pointblank.metadata._readers_cdisc import _read_define_xml_metadata
        except ImportError:
            return None
        result = _read_define_xml_metadata(str(define_xml))
        from pointblank.metadata._types import MetadataPackage as _Pkg

        if isinstance(result, _Pkg):
            return result
        pkg = _Pkg()
        domain = (result.domain or result.dataset_name or "").upper()
        pkg.items[domain] = result
        return pkg

    def _define_meta_for(self, domain: str) -> MetadataImport | None:
        if self._define_pkg is None:
            return None
        return self._define_pkg.items.get(domain.upper())

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

        # Partially Executable rules require extra inputs.
        # "DEFINE" is satisfied by define_xml; all others must be present as DataFrame keys.
        if rule.executability == "Partially Executable":
            missing_ds = []
            for d in rule.datasets:
                if d.upper() == "DEFINE":
                    if self._define_pkg is None:
                        missing_ds.append(d)
                elif d.upper() not in datasets:
                    missing_ds.append(d)
            if missing_ds:
                return NativeRuleResult(
                    rule_id=rule.core_id,
                    rule_type=rule.rule_type,
                    dataset=", ".join(rule.datasets),
                    status=STATUS_NOT_APPLICABLE,
                    sensitivity=rule.sensitivity,
                    description=rule.description,
                    message=f"Required input(s) not provided: {', '.join(missing_ds)}",
                )

        handler = {
            "RECORD_CHECK": self._record_check,
            "DATASET_METADATA_CHECK": self._dataset_metadata_check,
            "DOMAIN_PRESENCE_CHECK": self._domain_presence_check,
            "DATASET_CONTENTS_CHECK": self._dataset_contents_check,
            "VARIABLE_METADATA_CHECK": self._variable_metadata_check,
            "DEFINE_ITEM_METADATA_CHECK": self._define_item_metadata_check,
            "DEFINE_CODELIST_CHECK": self._define_codelist_check,
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
        if rule.domains:
            target_domains = rule.domains
        else:
            # Exclude SUPP-- and RELREC from catch-all iteration; they have non-standard structure.
            target_domains = [
                k for k in datasets if not k.startswith("SUPP") and k not in _STRUCTURAL_DATASETS
            ]
        all_findings: list[NativeRowFinding] = []
        n_issues = 0

        for domain in target_domains:
            df = datasets.get(domain.upper())
            if df is None:
                continue
            define_meta = self._define_meta_for(domain)
            df = apply_operations(df, rule.operations, self._ct, datasets, define_meta)
            try:
                mask = evaluate_conditions(df, rule.conditions)
            except EvaluationError:
                continue
            failing_rows = [i for i, v in enumerate(mask.to_list()) if v]
            n_issues += len(failing_rows)
            for row_idx in failing_rows[:_MAX_FINDINGS]:
                all_findings.append(
                    _build_row_finding(
                        df=df,
                        row_idx=row_idx,
                        domain=domain,
                        operations=rule.operations,
                        conditions=rule.conditions,
                        rule_id=rule.core_id,
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
        if rule.domains:
            target_domains = rule.domains
        else:
            # Exclude SUPP-- and RELREC from catch-all iteration; they have non-standard structure.
            target_domains = [
                k for k in datasets if not k.startswith("SUPP") and k not in _STRUCTURAL_DATASETS
            ]
        n_issues = 0
        first_failing_domain = ""

        for domain in target_domains:
            df = datasets.get(domain.upper())
            if df is None:
                continue
            define_meta = self._define_meta_for(domain)
            # Operations add the computed columns that conditions reference.
            df = apply_operations(df, rule.operations, self._ct, datasets, define_meta)
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

    def _variable_metadata_check(
        self, rule: NativeRule, datasets: dict[str, nw.DataFrame]
    ) -> NativeRuleResult:
        """Variable-level metadata check (presence, order, type).

        Delegates to `_dataset_metadata_check`; semantically distinct from `DATASET_METADATA_CHECK`
        (which checks dataset-level attributes like sort keys or record count) but evaluated
        identically. Operations add scalar broadcast columns that conditions then test.
        """
        return self._dataset_metadata_check(rule, datasets)

    def _define_item_metadata_check(
        self, rule: NativeRule, datasets: dict[str, nw.DataFrame]
    ) -> NativeRuleResult:
        """Check submission variables against Define-XML item declarations.

        Operations add broadcast computed columns (e.g. `_pb_SEX_in_define`, `_pb_SEX_mandatory_ok`)
        using the domain's `MetadataImport`; conditions then test those columns. Behaves like a
        dataset-level metadata check.
        """
        return self._dataset_metadata_check(rule, datasets)

    def _define_codelist_check(
        self, rule: NativeRule, datasets: dict[str, nw.DataFrame]
    ) -> NativeRuleResult:
        """Check that codelist values in the submission match Define-XML declarations.

        Per-row check: `define_codelist_check` operations add `_pb_<col>_define_valid` columns;
        conditions flag rows where the value is outside the declared codelist.
        """
        return self._record_check(rule, datasets)

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
