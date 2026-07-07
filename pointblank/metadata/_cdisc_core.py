"""Parser for CDISC CORE engine JSON reports (PLAN_06 Phase 2, Path A).

Pointblank wraps the open-source CDISC CORE engine (`cdisc-rules-engine`) as an external process:
datasets and Define-XML are handed to CORE, it runs the authoritative conformance rule set, and its
JSON report is parsed back into Pointblank's [`ConformanceReport`](`pointblank.ConformanceReport`).

This module implements:

- the **parser** — turning CORE's JSON report into typed objects;
- **dataset materialization** (`_write_xpt` / `_materialize_datasets`) — writing in-memory
  DataFrames to a temp dir of SAS Transport (XPT) files that CORE can read; and
- the **subprocess runner** (`_CoreRunner`) — discovering an installed CORE executable / command and
  invoking its `validate` subcommand.

The parser is written against the JSON schema emitted by `core validate -of JSON` (verified against
CORE 0.16.0):

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

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "CoreFinding",
    "CoreRuleResult",
    "CoreIssueSummary",
    "ParsedCoreReport",
    "parse_core_report",
    "CoreNotFoundError",
    "CoreExecutionError",
    "STATUS_SUCCESS",
    "STATUS_SKIPPED",
    "STATUS_ISSUE",
    "STATUS_ERROR",
]

# Environment variable naming the CORE executable / command to invoke.
_CORE_ENV_VAR = "POINTBLANK_CDISC_CORE"

# Executable names to probe on PATH during auto-discovery.
_CORE_EXECUTABLE_NAMES = ("core", "cdisc-rules-engine")

# File extension CORE appends to the `-o` output stem, keyed by output format.
_OUTPUT_EXTENSIONS = {"JSON": "json", "XLSX": "xlsx", "CSV": "csv"}

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


# ── Dataset materialization (in-memory → XPT) ────────────────────────────────


def _write_xpt(data: Any, path: str | Path, table_name: str, file_label: str = "") -> Path:
    """Write a DataFrame to a SAS Transport (XPT) file that CORE can read.

    Accepts any narwhals-supported DataFrame (Pandas, Polars, …); it is converted to Pandas for
    `pyreadstat.write_xport`. The XPT member (table) name is uppercased and truncated to 8
    characters — the CDISC / classic-XPT limit — which is sufficient for all SDTM/ADaM domain names.

    Parameters
    ----------
    data
        The DataFrame to write.
    path
        Destination `.xpt` file path.
    table_name
        The dataset/domain name to record as the XPT member name (e.g., `"DM"`).
    file_label
        Optional dataset label.

    Returns
    -------
    Path
        The path written.

    Raises
    ------
    ImportError
        If `pyreadstat` is not installed.
    """
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "The 'pyreadstat' package is required to write XPT files for the CDISC CORE engine. "
            "Install it with: pip install pyreadstat"
        ) from None

    import narwhals as nw

    pdf = nw.from_native(data, eager_only=True).to_pandas()
    dest = Path(path)
    member = str(table_name).upper()[:8]
    pyreadstat.write_xport(pdf, str(dest), table_name=member, file_label=file_label or "")
    return dest


def _materialize_datasets(datasets: dict[str, Any], dest_dir: str | Path) -> dict[str, Path]:
    """Write each dataset in a mapping to `<name>.xpt` under `dest_dir`.

    Parameters
    ----------
    datasets
        A mapping of dataset name (domain code) to DataFrame.
    dest_dir
        Directory to write the XPT files into (created if needed).

    Returns
    -------
    dict[str, Path]
        A mapping of dataset name to the written XPT path.
    """
    out_dir = Path(dest_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, data in datasets.items():
        xpt_path = out_dir / f"{name.lower()}.xpt"
        _write_xpt(data, xpt_path, table_name=name)
        written[name] = xpt_path
    return written


# ── CORE subprocess runner ───────────────────────────────────────────────────


class CoreNotFoundError(RuntimeError):
    """Raised when no CDISC CORE executable / command can be discovered."""


class CoreExecutionError(RuntimeError):
    """Raised when the CDISC CORE process exits with a non-zero status."""


def _normalize_version(version: str) -> str:
    """Normalize a standard version to CORE's hyphenated form (e.g., ``3.4`` → ``3-4``)."""
    return str(version).replace(".", "-")


def _resolve_core_command(core: str | Sequence[str] | None) -> list[str]:
    """Resolve the base command used to invoke CORE.

    Resolution order:

    1. An explicit `core` argument — a path/name (`str`) or a full command prefix (sequence, e.g.
       `["python", "/path/core.py"]` for a repo checkout, or `["docker", "run", ...]`).
    2. The `POINTBLANK_CDISC_CORE` environment variable (split on whitespace to allow a command).
    3. A `core` / `cdisc-rules-engine` executable on `PATH`.

    Returns
    -------
    list[str]
        The command prefix (before the `validate` subcommand and its flags).

    Raises
    ------
    CoreNotFoundError
        If nothing resolves.
    """
    if core is not None:
        if isinstance(core, str):
            return [core]
        return list(core)

    env_val = os.environ.get(_CORE_ENV_VAR)
    if env_val:
        return env_val.split()

    for name in _CORE_EXECUTABLE_NAMES:
        found = shutil.which(name)
        if found:
            return [found]

    raise CoreNotFoundError(
        "Could not find the CDISC CORE engine. Install the CORE standalone executable "
        "(https://github.com/cdisc-org/cdisc-rules-engine/releases) and either put it on your "
        f"PATH as 'core', set the {_CORE_ENV_VAR} environment variable to its path (or a full "
        "command such as 'python /path/to/core.py'), or pass core=... explicitly. Note: the pip "
        "package 'cdisc-rules-engine' is a library only and ships neither the CLI nor the rules "
        "cache."
    )


class _CoreRunner:
    """Discovers and invokes the CDISC CORE engine as an external subprocess.

    Parameters
    ----------
    core
        How to invoke CORE. A path/name to the CORE executable (`str`), a full command prefix
        (sequence — e.g., `["python", "core.py"]` for a repo checkout), or `None` to auto-discover
        via the `POINTBLANK_CDISC_CORE` environment variable and then `PATH`.
    cwd
        Working directory to run CORE from. CORE resolves its bundled `resources/` (rules cache,
        report templates) *relative to the current directory*, so when invoking a repo checkout
        (`core.py`) this must be the repo root. Standalone executables bundle their resources and
        generally do not need this. If `None`, the current process directory is used.
    """

    def __init__(
        self,
        core: str | Sequence[str] | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self._command = _resolve_core_command(core)
        self._cwd = str(cwd) if cwd is not None else None

    @property
    def command(self) -> list[str]:
        """The resolved base command used to invoke CORE."""
        return list(self._command)

    @property
    def cwd(self) -> str | None:
        """The working directory CORE is run from (or `None` for the current directory)."""
        return self._cwd

    def run_validate(
        self,
        data_dir: str | Path,
        standard: str,
        version: str,
        output_stem: str | Path,
        define_xml: str | Path | None = None,
        controlled_terminology: str | Sequence[str] | None = None,
        output_format: str = "JSON",
        raw_report: bool = False,
        cache: str | Path | None = None,
        extra_args: Sequence[str] | None = None,
        timeout: float | None = None,
    ) -> Path:
        """Run `core validate` and return the path to the report it produced.

        Parameters
        ----------
        data_dir
            Directory of datasets to validate (XPT / Dataset-JSON).
        standard
            CDISC standard (e.g., `"sdtmig"`).
        version
            Standard version; hyphenated automatically (e.g., `"3.4"` → `"3-4"`).
        output_stem
            Output path *stem*. CORE appends the format extension (e.g., `.json`).
        define_xml
            Optional path to a `define.xml` (passed via `-dxp`; CORE ignores define files placed in
            the data directory).
        controlled_terminology
            Optional CT package name(s) (passed via one or more `-ct`).
        output_format
            `"JSON"` (default), `"XLSX"`, or `"CSV"`.
        raw_report
            If `True` (JSON only), request CORE's raw report via `-rr`.
        cache
            Optional path to CORE's rules cache directory (passed via `-ca`).
        extra_args
            Additional raw CLI arguments appended verbatim.
        timeout
            Optional subprocess timeout in seconds.

        Returns
        -------
        Path
            The path to the report file CORE produced.

        Raises
        ------
        CoreExecutionError
            If CORE exits non-zero or the expected output file is not produced.
        """
        fmt = output_format.upper()
        if fmt not in _OUTPUT_EXTENSIONS:
            raise ValueError(
                f"Unsupported output_format {output_format!r}. "
                f"Choose one of {sorted(_OUTPUT_EXTENSIONS)}."
            )

        stem = Path(output_stem)
        cmd = [
            *self._command,
            "validate",
            "-s",
            str(standard),
            "-v",
            _normalize_version(version),
            "-d",
            str(data_dir),
            "-of",
            fmt,
            "-o",
            str(stem),
        ]
        if define_xml is not None:
            cmd += ["-dxp", str(define_xml)]
        if controlled_terminology is not None:
            cts = (
                [controlled_terminology]
                if isinstance(controlled_terminology, str)
                else list(controlled_terminology)
            )
            for ct in cts:
                cmd += ["-ct", str(ct)]
        if cache is not None:
            cmd += ["-ca", str(cache)]
        if raw_report and fmt == "JSON":
            cmd += ["-rr"]
        if extra_args:
            cmd += list(extra_args)

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._cwd,
            )
        except FileNotFoundError as e:
            raise CoreExecutionError(
                f"Failed to launch CDISC CORE (command: {cmd[0]!r}): {e}"
            ) from None

        if proc.returncode != 0:
            raise CoreExecutionError(
                f"CDISC CORE exited with status {proc.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr}"
            )

        produced = Path(f"{stem}.{_OUTPUT_EXTENSIONS[fmt]}")
        if not produced.exists():
            raise CoreExecutionError(
                f"CDISC CORE completed but the expected report was not found at {produced}.\n"
                f"stdout:\n{proc.stdout}"
            )
        return produced

    def validate_to_report(
        self,
        data_dir: str | Path,
        standard: str,
        version: str,
        output_stem: str | Path,
        **kwargs: Any,
    ) -> ParsedCoreReport:
        """Run `core validate` (JSON) and parse the result into a `ParsedCoreReport`.

        Accepts the same keyword arguments as `run_validate` (except `output_format`, which is
        forced to `"JSON"`).
        """
        kwargs.pop("output_format", None)
        report_path = self.run_validate(
            data_dir=data_dir,
            standard=standard,
            version=version,
            output_stem=output_stem,
            output_format="JSON",
            **kwargs,
        )
        with open(report_path) as f:
            return parse_core_report(json.load(f))
