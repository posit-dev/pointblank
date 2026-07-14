"""Submission-package model for CDISC conformance validation.

This module implements the *submission-package* layer described in PLAN_06: a data-level
analog of [`MetadataPackage`](`pointblank.MetadataPackage`) that understands the relationships
*between* datasets in a study (referential integrity, SUPP-- linkage, RELREC, ADaM ⇄ SDTM
traceability) and drives Pointblank validation across the whole package.

Where [`validate_sdtm()`](`pointblank.validate_sdtm`) and
[`validate_adam()`](`pointblank.validate_adam`) validate a *single* dataset structurally, the
[`SubmissionPackage`](`pointblank.SubmissionPackage`) validates a study as a graph of related
datasets, adding the cross-dataset checks that today send sponsors to Pinnacle 21 / CDISC CORE.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from pointblank.metadata._cdisc_core import CoreFinding, CoreRuleResult, ParsedCoreReport
    from pointblank.metadata._conformance.result import NativeConformanceResult, NativeRowFinding, NativeRuleResult
    from pointblank.metadata._types import MetadataPackage
    from pointblank.validate import Validate

__all__ = [
    "SubmissionPackage",
    "ConformanceReport",
    "validate_cdisc_submission",
]


# ── Dataset-name classification helpers ──────────────────────────────────────


def _is_supp(name: str) -> bool:
    """Whether a dataset name is an SDTM Supplemental Qualifiers (SUPP--) dataset."""
    return name.upper().startswith("SUPP")


def _is_relrec(name: str) -> bool:
    """Whether a dataset name is the SDTM Related Records (RELREC) dataset."""
    return name.upper() == "RELREC"


def _is_adam(name: str) -> bool:
    """Whether a dataset name is an ADaM dataset (conventionally prefixed ``AD``)."""
    return name.upper().startswith("AD")


def _column_names(data: Any) -> list[str]:
    """Return the column names of a native table via narwhals."""
    import narwhals as nw

    return list(nw.from_native(data, eager_only=True).columns)


def _column_value_set(data: Any, column: str) -> set:
    """Return the set of non-null values in a column of a native table."""
    import narwhals as nw

    df = nw.from_native(data, eager_only=True)
    if column not in df.columns:
        return set()
    return {v for v in df[column].to_list() if v is not None}


# ── XPT / Dataset-JSON ingestion ─────────────────────────────────────────────


def _read_xpt_data(path: Path) -> Any:
    """Read a SAS Transport (XPT) file into a pandas DataFrame."""
    try:
        import pyreadstat
    except ImportError:
        raise ImportError(
            "The 'pyreadstat' package is required to read XPT files. "
            "Install it with: pip install pyreadstat"
        ) from None

    df, _meta = pyreadstat.read_xport(str(path))
    return df


def _read_dataset_json(path: Path) -> tuple[Any, str | None]:
    """Read a CDISC Dataset-JSON file into a pandas DataFrame.

    Supports both the Dataset-JSON 1.1 top-level ``columns``/``rows`` layout and the older
    ``clinicalData``/``referenceData`` → ``itemGroupData`` nesting.

    Returns
    -------
    tuple[Any, str | None]
        The DataFrame and the dataset name (domain) if discoverable, else ``None``.
    """
    import json

    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "The 'pandas' package is required to read Dataset-JSON files. "
            "Install it with: pip install pandas"
        ) from None

    with open(path) as f:
        doc = json.load(f)

    # ── Dataset-JSON 1.1: top-level columns + rows ──
    if isinstance(doc, dict) and "columns" in doc and "rows" in doc:
        col_names = [c.get("name") for c in doc["columns"]]
        df = pd.DataFrame(doc["rows"], columns=col_names)
        name = doc.get("name") or doc.get("itemGroupOID")
        return df, (str(name).upper() if name else None)

    # ── Older Dataset-JSON: clinicalData / referenceData → itemGroupData ──
    for section in ("clinicalData", "referenceData"):
        block = doc.get(section) if isinstance(doc, dict) else None
        if not block:
            continue
        item_groups = block.get("itemGroupData", {})
        for oid, group in item_groups.items():
            items = group.get("items", [])
            col_names = [it.get("name") for it in items]
            rows = group.get("itemData", [])
            df = pd.DataFrame(rows, columns=col_names)
            name = group.get("name") or oid
            # Strip a leading "IG." OID prefix if present
            if isinstance(name, str) and name.upper().startswith("IG."):
                name = name[3:]
            return df, (str(name).upper() if name else None)

    raise ValueError(
        f"Could not parse '{path.name}' as Dataset-JSON: expected top-level "
        f"'columns'/'rows' or a 'clinicalData'/'referenceData' section."
    )


@dataclass
class SubmissionPackage:
    """A data-level model of a study submission package for CDISC conformance validation.

    A `SubmissionPackage` groups the datasets of a study (SDTM domains, SUPP-- qualifiers,
    RELREC, and/or ADaM datasets) together with their Define-XML and Controlled Terminology
    context, and understands the *relationships* between them. This enables cross-dataset
    conformance checks — referential integrity, SUPP-- linkage, RELREC resolution, and
    ADaM ⇄ SDTM traceability — that single-dataset validation cannot express.

    This is the data-level analog of [`MetadataPackage`](`pointblank.MetadataPackage`), which
    groups *metadata* for many datasets.

    Parameters
    ----------
    datasets
        A mapping of dataset name (domain code, e.g., `"DM"`, `"AE"`, `"SUPPAE"`, `"ADSL"`) to
        the dataset itself (a Pandas or Polars DataFrame). Names are matched case-insensitively
        but conventionally uppercase.
    define
        Optional Define-XML context: a path to a `define.xml` file, or an already-imported
        [`MetadataPackage`](`pointblank.MetadataPackage`). Used to supply variable definitions,
        codelists, and origins for define-context rules.
    ct_version
        Optional Controlled Terminology version pin (e.g., `"2024-03-29"`), recorded for
        reproducible runs.
    standard
        The data standard the package follows (`"sdtmig"` or `"adamig"`). Defaults to `"sdtmig"`.
    standard_version
        The Implementation Guide version (e.g., `"3.4"` for SDTM IG). Defaults to `"3.4"`.
    study_id
        Optional study identifier, used in report labels.

    Examples
    --------
    Construct a package from in-memory DataFrames and validate conformance across it:

    ```python
    import pointblank as pb

    study = pb.SubmissionPackage(
        datasets={"DM": dm_df, "AE": ae_df, "LB": lb_df},
        standard="sdtmig",
        standard_version="3.4",
    )

    report = study.validate_conformance()
    report.summary()
    ```

    Or ingest a folder of XPT files (Define-XML auto-detected if present):

    ```python
    study = pb.SubmissionPackage.from_folder("study_xyz/sdtm/")
    report = study.validate_conformance(agency="FDA")
    ```
    """

    datasets: dict[str, Any] = dataclass_field(default_factory=dict)
    define: Any = None
    ct_version: str | None = None
    standard: str = "sdtmig"
    standard_version: str = "3.4"
    study_id: str | None = None

    def __post_init__(self) -> None:
        # Normalize dataset keys to uppercase for consistent lookup.
        self.datasets = {str(k).upper(): v for k, v in self.datasets.items()}
        self._metadata: MetadataPackage | None = None
        # Set by `from_folder`; lets the CORE engine read the on-disk datasets directly instead of
        # re-materializing them.
        self.source_folder: str | None = None

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_folder(
        cls,
        path: str | Path,
        define: str | Path | Any | None = None,
        standard: str = "sdtmig",
        standard_version: str = "3.4",
        ct_version: str | None = None,
        study_id: str | None = None,
    ) -> SubmissionPackage:
        """Build a `SubmissionPackage` by ingesting a folder of datasets.

        Reads every SAS Transport (`.xpt`) and CDISC Dataset-JSON (`.json`) file in the folder,
        deriving the dataset name from the file stem (uppercased). If a `define.xml` is present
        in the folder and `define` is not supplied, it is picked up automatically.

        Parameters
        ----------
        path
            Path to a folder containing the study datasets.
        define
            Optional Define-XML path or [`MetadataPackage`](`pointblank.MetadataPackage`). If
            `None`, a `define.xml` in the folder is used when present.
        standard
            The data standard (`"sdtmig"` or `"adamig"`). Defaults to `"sdtmig"`.
        standard_version
            The Implementation Guide version. Defaults to `"3.4"`.
        ct_version
            Optional Controlled Terminology version pin.
        study_id
            Optional study identifier.

        Returns
        -------
        SubmissionPackage
            A package populated with the folder's datasets.
        """
        folder = Path(path)
        if not folder.is_dir():
            raise NotADirectoryError(f"Not a directory: {folder}")

        datasets: dict[str, Any] = {}
        define_in_folder: Path | None = None

        for f in sorted(folder.iterdir()):
            if not f.is_file():
                continue
            suffix = f.suffix.lower()
            if suffix == ".xpt":
                datasets[f.stem.upper()] = _read_xpt_data(f)
            elif suffix == ".json":
                try:
                    df, name = _read_dataset_json(f)
                except ValueError:
                    # Not a Dataset-JSON file (could be Frictionless/CSVW); skip it.
                    continue
                datasets[(name or f.stem).upper()] = df
            elif suffix == ".xml" and f.name.lower().startswith("define"):
                define_in_folder = f

        if define is None and define_in_folder is not None:
            define = define_in_folder

        package = cls(
            datasets=datasets,
            define=define,
            ct_version=ct_version,
            standard=standard,
            standard_version=standard_version,
            study_id=study_id,
        )
        package.source_folder = str(folder)
        return package

    # ── Dataset graph accessors ──────────────────────────────────────────────

    @property
    def domains(self) -> list[str]:
        """The names (domain codes) of all datasets in the package, sorted."""
        return sorted(self.datasets.keys())

    def __contains__(self, name: str) -> bool:
        return name.upper() in self.datasets

    def __getitem__(self, name: str) -> Any:
        return self.datasets[name.upper()]

    def __len__(self) -> int:
        return len(self.datasets)

    def get_dataset(self, name: str) -> Any:
        """Get a dataset by name (case-insensitive).

        Parameters
        ----------
        name
            The dataset name / domain code.

        Returns
        -------
        Any
            The dataset (DataFrame).

        Raises
        ------
        KeyError
            If no dataset with that name exists.
        """
        key = name.upper()
        if key not in self.datasets:
            raise KeyError(f"No dataset named '{name}'. Available: {self.domains}")
        return self.datasets[key]

    @property
    def metadata(self) -> MetadataPackage | None:
        """The imported Define-XML metadata, if `define` was supplied.

        Lazily imports the Define-XML document (via
        [`import_metadata()`](`pointblank.import_metadata`)) the first time it is accessed.
        """
        if self._metadata is not None:
            return self._metadata
        if self.define is None:
            return None

        from pointblank.metadata._types import MetadataPackage

        if isinstance(self.define, MetadataPackage):
            self._metadata = self.define
        elif isinstance(self.define, (str, Path)):
            from pointblank.metadata._import import import_metadata

            imported = import_metadata(self.define, format="cdisc_define")
            # import_metadata returns a MetadataPackage for Define-XML
            self._metadata = imported if isinstance(imported, MetadataPackage) else None
        return self._metadata

    # ── Cross-dataset operators ──────────────────────────────────────────────

    def subject_ids(self, dataset: str = "DM") -> set:
        """Get the set of `USUBJID` values in a dataset.

        Parameters
        ----------
        dataset
            The dataset to read subject IDs from. Defaults to `"DM"` (the reference set of
            all enrolled subjects).

        Returns
        -------
        set
            The set of non-null `USUBJID` values, or an empty set if the dataset or column
            is absent.
        """
        key = dataset.upper()
        if key not in self.datasets:
            return set()
        return _column_value_set(self.datasets[key], "USUBJID")

    def orphan_ids(self, child: str, parent: str = "DM", column: str = "USUBJID") -> set:
        """Find values of `column` in `child` that do not exist in `parent`.

        This is the referential-integrity operator: e.g., subjects appearing in a finding
        domain that have no corresponding record in DM.

        Parameters
        ----------
        child
            The referencing dataset (e.g., `"AE"`).
        parent
            The referenced dataset (e.g., `"DM"`). Defaults to `"DM"`.
        column
            The key column to check. Defaults to `"USUBJID"`.

        Returns
        -------
        set
            The set of orphaned values (present in `child.column` but not `parent.column`).
        """
        child_vals = _column_value_set(self.get_dataset(child), column)
        parent_vals = _column_value_set(self.get_dataset(parent), column)
        return child_vals - parent_vals

    # ── Conformance validation ───────────────────────────────────────────────

    def validate_conformance(
        self,
        agency: str | None = None,
        engine: str = "native",
        cross_dataset: bool = True,
        thresholds: Any = None,
        interrogate: bool = True,
        *,
        standard: str | None = None,
        version: str | None = None,
        ct_packages: list[str] | None = None,
        define_xml: Any = None,
        controlled_terminology: str | Sequence[str] | None = None,
        core: str | Sequence[str] | None = None,
        core_cwd: str | Path | None = None,
        cache: str | Path | None = None,
        workdir: str | Path | None = None,
    ) -> ConformanceReport:
        """Validate CDISC conformance across the whole submission package.

        Two engines are available:

        - **`"native"`** (default) — Pointblank's own checks. For each dataset this builds a
          [`Validate`](`pointblank.Validate`) plan combining the single-dataset structural checks
          (via [`validate_sdtm()`](`pointblank.validate_sdtm`) /
          [`validate_adam()`](`pointblank.validate_adam`)) and cross-dataset conformance checks
          (when `cross_dataset=True`):
            - **Referential integrity** — every `USUBJID` in a finding/events/interventions
              domain exists in DM.
            - **SUPP-- linkage** — `RDOMAIN` references a present domain, `USUBJID` exists in DM,
              and `(USUBJID, IDVAR=IDVARVAL)` resolves to a record in the parent domain.
            - **RELREC** — each relationship record's `RDOMAIN` is present and `USUBJID` exists
              in DM.
            - **ADaM ⇄ SDTM traceability** — `ADSL.USUBJID ⊆ DM.USUBJID`, and every other
              ADaM dataset's `USUBJID ⊆ ADSL.USUBJID`.

        - **`"core"`** — hands the package to the external CDISC CORE engine
          (`cdisc-rules-engine`), which runs the authoritative conformance rule set, and ingests
          its results. Datasets are materialized to XPT (or the source folder is used directly for
          folder-ingested packages), CORE is invoked as a subprocess, and its JSON report becomes a
          CORE-form `ConformanceReport`. Requires an installed CORE executable (see `core`).

        Parameters
        ----------
        agency
            Optional agency rule-set selector (`"FDA"`, `"PMDA"`, or `None` for CDISC base
            rules). Recorded on the report; agency-specific business rule sets are a later
            phase, so this currently affects labeling only.
        engine
            `"native"` (default) or `"core"`.
        cross_dataset
            (Native only.) Whether to add cross-dataset conformance checks. Defaults to `True`.
        thresholds
            (Native only.) Optional thresholds passed to each dataset's `Validate` (maps failing
            test units onto Pointblank's warning/error/critical severity model).
        interrogate
            (Native only.) Whether to interrogate (run) the validations before returning.
        standard
            (CORE only.) Override the CDISC standard sent to CORE. Defaults to the package's
            `standard` (e.g., `"sdtmig"`).
        version
            (CORE only.) Override the standard version. Defaults to the package's
            `standard_version` (e.g., `"3.4"`, sent to CORE hyphenated).
        controlled_terminology
            (CORE only.) CT package name(s) for CORE's `-ct` (e.g., `"sdtmct-2024-03-29"`).
        core
            (CORE only.) How to invoke CORE — a path/name to the CORE executable, a full command
            prefix (e.g., `["python", "core.py"]`), or `None` to auto-discover via the
            `POINTBLANK_CDISC_CORE` environment variable and then `PATH`.
        core_cwd
            (CORE only.) Working directory to run CORE from; required when invoking a repo checkout
            (CORE resolves its bundled `resources/` relative to the current directory).
        cache
            (CORE only.) Path to CORE's rules cache directory (`-ca`).
        workdir
            (CORE only.) Directory for materialized XPT and the CORE report. If `None`, a temporary
            directory is used and cleaned up.

        Returns
        -------
        ConformanceReport
            A native-form report (per-dataset validations) or a CORE-form report, depending on
            `engine`.
        """
        if engine not in ("native", "core", "validate"):
            raise ValueError(f"engine must be 'native', 'validate', or 'core', got {engine!r}.")

        if engine == "core":
            return self._run_core_conformance(
                agency=agency,
                standard=standard,
                version=version,
                controlled_terminology=controlled_terminology,
                core=core,
                core_cwd=core_cwd,
                cache=cache,
                workdir=workdir,
            )

        # engine="validate" always uses the Validate-based approach (cross-dataset checks).
        if engine != "validate":
            # Try the rule-based native engine first (requires a bundled catalog).
            std = standard or self.standard
            ver = version or self.standard_version
            rules_report = self._run_rules_conformance(
                agency=agency, standard=std, version=ver, ct_packages=ct_packages,
                define_xml=define_xml,
            )
            if rules_report is not None:
                return rules_report

        # Fallback (or engine="validate"): Validate-based approach with cross-dataset checks.
        validations: dict[str, Validate] = {}

        for name in self.domains:
            validation = self._build_dataset_validation(
                name, cross_dataset=cross_dataset, thresholds=thresholds
            )
            if validation is None:
                continue
            if interrogate:
                validation = validation.interrogate()
            validations[name] = validation

        return ConformanceReport(validations=validations, package=self, agency=agency)

    def _run_rules_conformance(
        self,
        agency: str | None,
        standard: str,
        version: str,
        ct_packages: list[str] | None,
        define_xml: Any = None,
    ) -> ConformanceReport | None:
        """Run the native rule-based engine; returns None if no catalog is bundled."""
        from pointblank.metadata._conformance.engine import NativeConformanceEngine
        from pointblank.metadata._conformance.rule_loader import RuleLoader

        if not RuleLoader.catalog_path(standard, version).exists():
            return None

        engine = NativeConformanceEngine(
            standard=standard, version=version, ct_packages=ct_packages
        )
        # Prefer explicit define_xml argument; fall back to package's define path.
        _define = define_xml if define_xml is not None else (
            self.define if isinstance(self.define, (str, Path)) else None
        )
        result = engine.run(self.datasets, define_xml=_define)
        return ConformanceReport(native_result=result, package=self, agency=agency)

    def _run_core_conformance(
        self,
        agency: str | None,
        standard: str | None,
        version: str | None,
        controlled_terminology: str | Sequence[str] | None,
        core: str | Sequence[str] | None,
        core_cwd: str | Path | None,
        cache: str | Path | None,
        workdir: str | Path | None,
    ) -> ConformanceReport:
        """Run the CDISC CORE engine over the package and ingest its report."""
        import tempfile

        from pointblank.metadata._cdisc_core import _CoreRunner, _materialize_datasets

        std = standard or self.standard
        ver = version or self.standard_version
        define_xml = self.define if isinstance(self.define, (str, Path)) else None

        runner = _CoreRunner(core=core, cwd=core_cwd)

        tmp: tempfile.TemporaryDirectory | None = None
        if workdir is None:
            tmp = tempfile.TemporaryDirectory(prefix="pb_cdisc_core_")
            base = Path(tmp.name)
        else:
            base = Path(workdir)
            base.mkdir(parents=True, exist_ok=True)

        try:
            # Prefer reading the on-disk datasets directly for folder-ingested packages; otherwise
            # materialize the in-memory datasets to XPT.
            if self.source_folder and Path(self.source_folder).is_dir():
                data_dir: Path = Path(self.source_folder)
            else:
                data_dir = base / "data"
                _materialize_datasets(self.datasets, data_dir)

            parsed = runner.validate_to_report(
                data_dir=data_dir,
                standard=std,
                version=ver,
                output_stem=base / "core_report",
                define_xml=define_xml,
                controlled_terminology=controlled_terminology,
                cache=cache,
            )
        finally:
            if tmp is not None:
                tmp.cleanup()

        return ConformanceReport.from_core_report(parsed, package=self, agency=agency)

    def _build_dataset_validation(
        self,
        name: str,
        cross_dataset: bool,
        thresholds: Any,
    ) -> Validate | None:
        """Build the `Validate` plan for one dataset (structural + cross-dataset checks)."""
        from pointblank.validate import Validate

        data = self.datasets[name]

        # ── Single-dataset structural checks ──
        validation = self._structural_validation(name, data, thresholds)

        # If no structural template applied, start a bare Validate so cross-dataset checks can
        # still attach (e.g., SUPP-- or unknown domains that carry USUBJID).
        if validation is None:
            label = f"CDISC {name} conformance"
            if self.study_id:
                label = f"CDISC {name} — {self.study_id}"
            validation = Validate(data=data, label=label, thresholds=thresholds)

        if cross_dataset:
            self._add_cross_dataset_checks(validation, name, data)

        return validation

    def _structural_validation(self, name: str, data: Any, thresholds: Any) -> Validate | None:
        """Apply the appropriate single-dataset structural template, if any."""
        # ADaM datasets
        if _is_adam(name):
            from pointblank.metadata._adam_validate import validate_adam

            # Map a concrete dataset (e.g., ADLB) onto its structural class (BDS) when the exact
            # name has no template. ADSL/ADAE/ADTTE have direct templates; others are BDS-class.
            dataset_key = _adam_class_for(name)
            if dataset_key is None:
                return None
            try:
                return validate_adam(
                    data=data,
                    dataset=dataset_key,
                    study_id=self.study_id,
                    thresholds=thresholds,
                )
            except KeyError:
                return None

        # SUPP-- and RELREC have no single-dataset structural template here; handled by
        # cross-dataset checks. Everything else is treated as an SDTM domain.
        if _is_supp(name) or _is_relrec(name):
            return None

        from pointblank.metadata._sdtm_validate import validate_sdtm

        try:
            return validate_sdtm(
                data=data,
                domain=name,
                study_id=self.study_id,
                thresholds=thresholds,
            )
        except KeyError:
            # Not a domain we have a template for.
            return None

    def _add_cross_dataset_checks(self, validation: Validate, name: str, data: Any) -> None:
        """Attach cross-dataset conformance checks to a dataset's `Validate` plan."""
        cols = set(_column_names(data))
        has_dm = "DM" in self.datasets

        # ── SUPP-- linkage ──
        if _is_supp(name):
            self._add_supp_checks(validation, name, data, cols, has_dm)
            return

        # ── RELREC ──
        if _is_relrec(name):
            self._add_relrec_checks(validation, data, cols, has_dm)
            return

        # ── ADaM ⇄ SDTM traceability ──
        if _is_adam(name):
            self._add_adam_traceability(validation, name, cols)
            return

        # ── SDTM referential integrity: USUBJID must exist in DM ──
        if name != "DM" and has_dm and "USUBJID" in cols:
            valid_ids = self.subject_ids("DM")
            validation.specially(
                expr=_referential_expr("USUBJID", valid_ids),
                brief=f"USUBJID values exist in DM ({name} → DM)",
                dimension="consistency",
            )

    def _add_supp_checks(
        self, validation: Validate, name: str, data: Any, cols: set, has_dm: bool
    ) -> None:
        """SUPP-- (Supplemental Qualifiers) linkage checks."""
        present_domains = set(self.datasets.keys())

        # RDOMAIN must reference a domain present in the package.
        if "RDOMAIN" in cols:
            validation.specially(
                expr=_membership_expr("RDOMAIN", present_domains),
                brief=f"{name} RDOMAIN references a present domain",
                dimension="consistency",
            )

        # USUBJID must exist in DM.
        if has_dm and "USUBJID" in cols:
            validation.specially(
                expr=_referential_expr("USUBJID", self.subject_ids("DM")),
                brief=f"{name} USUBJID values exist in DM",
                dimension="consistency",
            )

        # (USUBJID, IDVAR=IDVARVAL) must resolve to a record in the parent domain.
        if {"RDOMAIN", "IDVAR", "IDVARVAL", "USUBJID"}.issubset(cols):
            validation.specially(
                expr=self._supp_idvar_expr(),
                brief=f"{name} IDVAR/IDVARVAL resolves in parent domain",
                dimension="consistency",
            )

    def _add_relrec_checks(
        self, validation: Validate, data: Any, cols: set, has_dm: bool
    ) -> None:
        """RELREC (Related Records) resolution checks (lightweight)."""
        present_domains = set(self.datasets.keys())
        if "RDOMAIN" in cols:
            validation.specially(
                expr=_membership_expr("RDOMAIN", present_domains),
                brief="RELREC RDOMAIN references a present domain",
                dimension="consistency",
            )
        if has_dm and "USUBJID" in cols:
            validation.specially(
                expr=_referential_expr("USUBJID", self.subject_ids("DM"), na_pass=True),
                brief="RELREC USUBJID values exist in DM",
                dimension="consistency",
            )

    def _add_adam_traceability(self, validation: Validate, name: str, cols: set) -> None:
        """ADaM ⇄ SDTM (and ADaM ⇄ ADSL) subject-level traceability checks."""
        if "USUBJID" not in cols:
            return

        if name == "ADSL":
            # ADSL subjects must trace to a DM record.
            if "DM" in self.datasets:
                validation.specially(
                    expr=_referential_expr("USUBJID", self.subject_ids("DM")),
                    brief="ADSL USUBJID values trace to DM",
                    dimension="consistency",
                )
        else:
            # Other ADaM datasets must trace to ADSL.
            if "ADSL" in self.datasets:
                validation.specially(
                    expr=_referential_expr("USUBJID", self.subject_ids("ADSL")),
                    brief=f"{name} USUBJID values trace to ADSL",
                    dimension="consistency",
                )

    def _supp_idvar_expr(self):
        """Build a `specially()` callable resolving SUPP IDVAR/IDVARVAL into parent records."""
        datasets = self.datasets

        def check(data: Any) -> list[bool]:
            import narwhals as nw

            df = nw.from_native(data, eager_only=True)
            rdomain = df["RDOMAIN"].to_list()
            idvar = df["IDVAR"].to_list()
            idvarval = df["IDVARVAL"].to_list()
            usubjid = df["USUBJID"].to_list()

            # Cache of (rdomain, idvar) -> set of (usubjid, str(value)) lookups.
            lookups: dict[tuple[str, str], set] = {}

            def _lookup(rdom: str, var: str) -> set | None:
                keycache = (rdom, var)
                if keycache in lookups:
                    return lookups[keycache]
                parent = datasets.get(rdom)
                if parent is None:
                    lookups[keycache] = None  # type: ignore[assignment]
                    return None
                pdf = nw.from_native(parent, eager_only=True)
                if var not in pdf.columns or "USUBJID" not in pdf.columns:
                    lookups[keycache] = None  # type: ignore[assignment]
                    return None
                pairs = {
                    (u, str(v))
                    for u, v in zip(pdf["USUBJID"].to_list(), pdf[var].to_list())
                    if v is not None
                }
                lookups[keycache] = pairs
                return pairs

            results: list[bool] = []
            for rdom, var, val, usub in zip(rdomain, idvar, idvarval, usubjid):
                # Rows with no IDVAR reference a whole-domain qualifier: pass.
                if var is None or (isinstance(var, str) and var.strip() == ""):
                    results.append(True)
                    continue
                pairs = _lookup(str(rdom).upper(), str(var))
                if pairs is None:
                    # Parent domain / variable absent: cannot resolve, flag as failure.
                    results.append(False)
                    continue
                results.append((usub, str(val)) in pairs)
            return results

        return check

    # ── Reporting helpers ────────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the package contents."""
        lines = ["Submission Package"]
        if self.study_id:
            lines.append(f"  Study: {self.study_id}")
        lines.append(f"  Standard: {self.standard} {self.standard_version}")
        if self.ct_version:
            lines.append(f"  CT version: {self.ct_version}")
        if self.define is not None:
            lines.append("  Define-XML: present")
        lines.append(f"  Datasets ({len(self.datasets)}): {', '.join(self.domains)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"SubmissionPackage(datasets={self.domains}, "
            f"standard={self.standard!r}, standard_version={self.standard_version!r})"
        )


# ── ADaM template resolution helpers ─────────────────────────────────────────


def _adam_template_names() -> set:
    """Names of ADaM datasets with a direct structural template."""
    from pointblank.metadata._adam_templates import list_adam_datasets

    return {n.upper() for n in list_adam_datasets()}


def _adam_class_for(name: str) -> str | None:
    """Map a concrete ADaM dataset name onto its structural class template.

    ADSL, ADAE, ADTTE have direct templates; occurrence/BDS datasets (ADLB, ADVS, ADEG, ...)
    validate against the generic BDS structure.
    """
    upper = name.upper()
    templates = _adam_template_names()
    if upper in templates:
        return upper
    # Occurrence-data ADaM datasets other than ADAE fall back to BDS structure.
    if "BDS" in templates:
        return "BDS"
    return None


# ── `specially()` expression factories ───────────────────────────────────────


def _referential_expr(column: str, valid_values: set, na_pass: bool = True):
    """Build a `specially()` callable: each row's `column` value is in `valid_values`.

    Null values pass when `na_pass` is `True` (null handling is a separate not-null check).
    """

    def check(data: Any) -> list[bool]:
        import narwhals as nw

        df = nw.from_native(data, eager_only=True)
        if column not in df.columns:
            return [True]
        return [
            (True if (v is None and na_pass) else (v in valid_values))
            for v in df[column].to_list()
        ]

    return check


def _membership_expr(column: str, valid_values: set, na_pass: bool = True):
    """Build a `specially()` callable: each row's `column` value is in `valid_values`.

    Distinct from `_referential_expr` only in intent (set membership vs. referential lookup);
    the value set is compared case-insensitively for domain codes.
    """
    upper_valid = {str(v).upper() for v in valid_values}

    def check(data: Any) -> list[bool]:
        import narwhals as nw

        df = nw.from_native(data, eager_only=True)
        if column not in df.columns:
            return [True]
        return [
            (True if (v is None and na_pass) else (str(v).upper() in upper_valid))
            for v in df[column].to_list()
        ]

    return check


@dataclass
class ConformanceReport:
    """The result of [`SubmissionPackage.validate_conformance()`](`pointblank.SubmissionPackage`).

    A `ConformanceReport` comes in one of two forms depending on the validation engine used:

    - **Native** (`engine="native"`, the default) — aggregates the per-dataset
      [`Validate`](`pointblank.Validate`) objects produced for the submission package. Each
      dataset's validation carries both its single-dataset structural checks and the cross-dataset
      conformance checks that reference it.
    - **CORE** (`engine="core"`) — wraps the results of the external CDISC CORE engine, holding its
      rule-ID-keyed findings, per-rule run statuses, and run provenance.

    The `all_passed()`, `summary()`, `issues()`, and rendering methods work for both forms; use the
    `is_core` property to tell them apart. CORE-backed reports additionally expose `findings()` and
    `rules()`.

    Parameters
    ----------
    validations
        A mapping of dataset name to its interrogated `Validate` object (native form).
    package
        The `SubmissionPackage` the report was produced from.
    agency
        The agency rule-set selector used for the run (or `None` for CDISC base rules).
    core
        The parsed CDISC CORE report (CORE form). `None` for native reports.
    """

    validations: dict[str, Validate] = dataclass_field(default_factory=dict)
    package: SubmissionPackage | None = None
    agency: str | None = None
    core: ParsedCoreReport | None = None
    native_result: NativeConformanceResult | None = None

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_core_report(
        cls,
        report: dict | ParsedCoreReport,
        package: SubmissionPackage | None = None,
        agency: str | None = None,
    ) -> ConformanceReport:
        """Build a CORE-backed `ConformanceReport` from a CDISC CORE JSON report.

        Parameters
        ----------
        report
            Either a raw CORE JSON report (`dict`, as produced by `core validate -of JSON`) or an
            already-parsed [`ParsedCoreReport`](`pointblank.metadata._cdisc_core.ParsedCoreReport`).
        package
            The `SubmissionPackage` the run was produced from, if any.
        agency
            The agency rule-set selector used for the run.

        Returns
        -------
        ConformanceReport
            A report in CORE form (`is_core` is `True`).
        """
        from pointblank.metadata._cdisc_core import ParsedCoreReport, parse_core_report

        parsed = report if isinstance(report, ParsedCoreReport) else parse_core_report(report)
        return cls(package=package, agency=agency, core=parsed)

    @property
    def is_core(self) -> bool:
        """Whether this report wraps CDISC CORE engine results (vs. native validations)."""
        return self.core is not None

    @property
    def is_rules(self) -> bool:
        """Whether this report was produced by the native rule-based conformance engine."""
        return self.native_result is not None

    def all_passed(self) -> bool:
        """Whether the run reported no conformance failures.

        For native reports, this is `True` when every check in every dataset passed with no failing
        test units. For CORE reports, this is `True` when no rule reported an issue or execution
        error.
        """
        if self.is_core:
            return self.core.all_passed
        if self.is_rules:
            return self.native_result.all_passed
        return all(v.all_passed() for v in self.validations.values())

    def __getitem__(self, name: str) -> Validate:
        return self.validations[name.upper()]

    def get_validation(self, name: str) -> Validate:
        """Get the `Validate` object for a single dataset (case-insensitive)."""
        key = name.upper()
        if key not in self.validations:
            raise KeyError(f"No validation for '{name}'. Available: {sorted(self.validations)}")
        return self.validations[key]

    def summary(self) -> dict:
        """Return a summary of the conformance run.

        Returns
        -------
        dict
            For a **native** report, a mapping of dataset name to a dict with keys `n_steps`,
            `n_steps_failed`, `n_failed` (failing test units), and `all_passed`.

            For a **CORE** report, a single dict with keys `standard`, `version`,
            `engine_version`, `n_rules`, `status_counts` (rule counts by run status), `n_issues`
            (total reported issues), `n_datasets`, and `all_passed`.
        """
        if self.is_core:
            core = self.core
            return {
                "standard": core.standard,
                "version": core.version,
                "engine_version": core.engine_version,
                "engine": "core",
                "n_rules": len(core.rules),
                "status_counts": core.status_counts(),
                "n_issues": core.n_total_issues,
                "n_datasets": len(core.datasets),
                "all_passed": core.all_passed,
            }

        if self.is_rules:
            nr = self.native_result
            return {
                "standard": nr.standard,
                "version": nr.version,
                "engine": "native",
                "ct_packages": nr.ct_packages,
                "n_rules": len(nr.rule_results),
                "status_counts": nr.status_counts(),
                "n_issues": nr.n_total_issues,
                "all_passed": nr.all_passed,
            }

        out: dict[str, dict] = {}
        for name, v in self.validations.items():
            steps = v.validation_info
            n_steps = len(steps)
            n_steps_failed = sum(1 for s in steps if not s.all_passed)
            n_failed = sum(int(s.n_failed or 0) for s in steps)
            out[name] = {
                "n_steps": n_steps,
                "n_steps_failed": n_steps_failed,
                "n_failed": n_failed,
                "all_passed": v.all_passed(),
            }
        return out

    def issues(self, severity: str | None = None, status: str | None = None) -> list[dict]:
        """Return the conformance issues found.

        Parameters
        ----------
        severity
            (Native reports only.) Optional severity filter: `"warning"`, `"error"`, or
            `"critical"`. Requires thresholds to have been set on the run. If `None`, all steps
            with failing test units are returned.
        status
            (CORE reports only.) Optional rule-status filter, e.g. `"ISSUE REPORTED"` or
            `"EXECUTION ERROR"`. If `None`, all reported issues are returned.

        Returns
        -------
        list[dict]
            For a **native** report, one dict per failing step, with keys `dataset`, `step`,
            `assertion`, `column`, `n_failed`, `n`, and `severity`.

            For a **CORE** report, one dict per (dataset, rule) with reported issues, with keys
            `dataset`, `rule_id`, `message`, `issues` (count), and `status`.
        """
        if self.is_rules:
            return self.native_result.issues()

        if self.is_core:
            # Look up each rule's run status by rule id.
            status_by_rule = {r.rule_id: r.status for r in self.core.rules}
            out: list[dict] = []
            for item in self.core.issue_summary:
                item_status = status_by_rule.get(item.rule_id)
                if status is not None and item_status != status:
                    continue
                out.append(
                    {
                        "dataset": item.dataset,
                        "rule_id": item.rule_id,
                        "message": item.message,
                        "issues": item.issues,
                        "status": item_status,
                    }
                )
            return out

        issues: list[dict] = []
        for name, v in self.validations.items():
            for s in v.validation_info:
                n_failed = int(s.n_failed or 0)
                if n_failed == 0:
                    continue
                sev = None
                if s.critical:
                    sev = "critical"
                elif s.error:
                    sev = "error"
                elif s.warning:
                    sev = "warning"
                if severity is not None and sev != severity:
                    continue
                issues.append(
                    {
                        "dataset": name,
                        "step": s.i,
                        "assertion": s.assertion_type,
                        "column": s.column,
                        "n_failed": n_failed,
                        "n": int(s.n or 0),
                        "severity": sev,
                    }
                )
        return issues

    def findings(self):
        """Return the row-level findings.

        For CORE reports, returns `CoreFinding` objects from CORE's `Issue_Details`.
        For native rule reports, returns `NativeRowFinding` objects.
        For Validate-based native reports, returns an empty list.
        """
        if self.is_core:
            return list(self.core.findings)
        if self.is_rules:
            return self.native_result.findings()
        return []

    def rules(self, status: str | None = None):
        """Return the per-rule run results.

        For CORE reports, returns `CoreRuleResult` objects.
        For native rule reports, returns `NativeRuleResult` objects.
        For Validate-based native reports, returns an empty list.

        Parameters
        ----------
        status
            Optional status filter. For CORE: e.g. `"SUCCESS"`, `"SKIPPED"`. For native rules:
            `"pass"`, `"fail"`, `"error"`, `"not_applicable"`, `"not_supported"`.
        """
        if self.is_core:
            if status is None:
                return list(self.core.rules)
            return [r for r in self.core.rules if r.status == status]
        if self.is_rules:
            return self.native_result.rules(status=status)
        return []

    @property
    def n_datasets(self) -> int:
        """Number of datasets validated."""
        if self.is_core:
            return len(self.core.datasets)
        if self.is_rules and self.package is not None:
            return len(self.package.datasets)
        return len(self.validations)

    def to_json(self, path: str | Path) -> Path:
        """Save the conformance report as a JSON file.

        For CORE reports the output mirrors the original CORE JSON structure (`Conformance_Details`,
        `Dataset_Details`, `Issue_Summary`, `Issue_Details`, `Rules_Report`), making the file
        readable by anything that parses a standard CORE report. For native reports the file
        contains `summary` and `issues` keys.

        Parameters
        ----------
        path
            Destination path (including filename). Parent directories are created if needed.

        Returns
        -------
        Path
            The path written.
        """
        import json

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if self.is_core:
            core = self.core
            data: dict = {
                "Conformance_Details": core.details,
                "Dataset_Details": core.datasets,
                "Issue_Summary": [
                    {
                        "dataset": s.dataset,
                        "core_id": s.rule_id,
                        "message": s.message,
                        "issues": s.issues,
                    }
                    for s in core.issue_summary
                ],
                "Issue_Details": [
                    {
                        "core_id": f.rule_id,
                        "message": f.message,
                        "dataset": f.dataset,
                        "executability": f.executability or "",
                        "USUBJID": f.usubjid or "",
                        "row": f.row,
                        "SEQ": f.seq or "",
                        "variables": f.variables,
                        "values": f.values,
                    }
                    for f in core.findings
                ],
                "Rules_Report": [
                    {
                        "core_id": r.rule_id,
                        "status": r.status,
                        "message": r.message or "",
                        "version": r.version or "",
                        "cdisc_rule_id": r.cdisc_rule_id or "",
                        "fda_rule_id": r.fda_rule_id or "",
                    }
                    for r in core.rules
                ],
            }
        else:
            data = {"summary": self.summary(), "issues": self.issues()}

        with open(dest, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return dest

    def to_excel(self, path: str | Path) -> Path:
        """Save the conformance report as an Excel workbook.

        For CORE reports the workbook contains sheets `Issue_Summary`, `Issue_Details`,
        `Rules_Report`, and `Conformance_Details`. For native reports the workbook contains
        `Issues` and `Summary`.

        Requires the `openpyxl` package (`pip install openpyxl` or
        `pip install 'pointblank[excel]'`).

        Parameters
        ----------
        path
            Destination path (including filename). Parent directories are created if needed.

        Returns
        -------
        Path
            The path written.

        Raises
        ------
        ImportError
            If `openpyxl` or `pandas` are not installed.
        """
        try:
            import openpyxl  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'openpyxl' package is required to export to Excel. "
                "Install it with: pip install openpyxl"
            ) from None
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "The 'pandas' package is required to export to Excel. "
                "Install it with: pip install pandas"
            ) from None

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        with pd.ExcelWriter(dest, engine="openpyxl") as writer:
            if self.is_core:
                core = self.core
                if core.issue_summary:
                    pd.DataFrame(
                        [
                            {
                                "Dataset": s.dataset,
                                "Rule ID": s.rule_id,
                                "Message": s.message,
                                "Issues": s.issues,
                            }
                            for s in core.issue_summary
                        ]
                    ).to_excel(writer, sheet_name="Issue_Summary", index=False)
                if core.findings:
                    pd.DataFrame(
                        [
                            {
                                "Rule ID": f.rule_id,
                                "Message": f.message,
                                "Dataset": f.dataset,
                                "USUBJID": f.usubjid or "",
                                "Row": f.row,
                                "SEQ": f.seq or "",
                                "Variables": ", ".join(str(v) for v in f.variables),
                                "Values": ", ".join(str(v) for v in f.values),
                            }
                            for f in core.findings
                        ]
                    ).to_excel(writer, sheet_name="Issue_Details", index=False)
                if core.rules:
                    pd.DataFrame(
                        [
                            {
                                "Rule ID": r.rule_id,
                                "Status": r.status,
                                "Message": r.message or "",
                                "Version": r.version or "",
                                "CDISC Rule ID": r.cdisc_rule_id or "",
                                "FDA Rule ID": r.fda_rule_id or "",
                            }
                            for r in core.rules
                        ]
                    ).to_excel(writer, sheet_name="Rules_Report", index=False)
                if core.details:
                    pd.DataFrame(
                        [{"Key": k, "Value": v} for k, v in core.details.items()]
                    ).to_excel(writer, sheet_name="Conformance_Details", index=False)
            elif self.is_rules:
                nr = self.native_result
                pd.DataFrame(
                    [
                        {
                            "Rule ID": r.rule_id,
                            "Rule Type": r.rule_type,
                            "Dataset": r.dataset,
                            "Status": r.status,
                            "Sensitivity": r.sensitivity,
                            "Issues": r.n_issues,
                            "Message": r.message or r.description,
                        }
                        for r in nr.rule_results
                    ]
                ).to_excel(writer, sheet_name="Rules_Report", index=False)
                issues = self.issues()
                if issues:
                    pd.DataFrame(issues).to_excel(writer, sheet_name="Issues", index=False)
                s = self.summary()
                pd.DataFrame(
                    [{"Key": k, "Value": str(v)} for k, v in s.items()]
                ).to_excel(writer, sheet_name="Summary", index=False)
            else:
                issues = self.issues()
                if issues:
                    pd.DataFrame(issues).to_excel(writer, sheet_name="Issues", index=False)
                pd.DataFrame(
                    [{"Dataset": name, **s} for name, s in self.summary().items()]
                ).to_excel(writer, sheet_name="Summary", index=False)

        return dest

    def _repr_html_(self) -> str:
        agency = f" — agency: {self.agency}" if self.agency else ""

        if self.is_rules:
            nr = self.native_result
            parts = [f"<h2>CDISC Conformance Report (Native Rules){agency}</h2>"]
            status = "PASS" if nr.all_passed else "FAIL"
            parts.append(
                f"<p><strong>{nr.standard} {nr.version}</strong> — "
                f"<strong>{status}</strong></p>"
            )
            counts = nr.status_counts()
            parts.append("<ul>")
            for st, n in sorted(counts.items()):
                parts.append(f"<li>{st}: {n}</li>")
            parts.append(f"<li><strong>Total issues:</strong> {nr.n_total_issues}</li>")
            parts.append("</ul>")
            failing = [r for r in nr.rule_results if r.n_issues > 0]
            if failing:
                parts.append(
                    "<table><tr><th>Dataset</th><th>Rule</th>"
                    "<th>Issues</th><th>Message</th></tr>"
                )
                for r in failing:
                    parts.append(
                        f"<tr><td>{r.dataset}</td><td>{r.rule_id}</td>"
                        f"<td>{r.n_issues}</td><td>{r.message or r.description}</td></tr>"
                    )
                parts.append("</table>")
            return "\n".join(parts)

        if self.is_core:
            core = self.core
            parts = [f"<h2>CDISC Conformance Report (CORE){agency}</h2>"]
            status = "PASS" if core.all_passed else "FAIL"
            parts.append(
                f"<p><strong>{core.standard} {core.version}</strong> — CORE "
                f"{core.engine_version} — <strong>{status}</strong></p>"
            )
            counts = core.status_counts()
            parts.append("<ul>")
            for st, n in sorted(counts.items()):
                parts.append(f"<li>{st}: {n}</li>")
            parts.append(f"<li><strong>Total issues:</strong> {core.n_total_issues}</li>")
            parts.append("</ul>")
            if core.issue_summary:
                parts.append("<table><tr><th>Dataset</th><th>Rule</th>"
                             "<th>Issues</th><th>Message</th></tr>")
                for item in core.issue_summary:
                    parts.append(
                        f"<tr><td>{item.dataset}</td><td>{item.rule_id}</td>"
                        f"<td>{item.issues}</td><td>{item.message}</td></tr>"
                    )
                parts.append("</table>")
            return "\n".join(parts)

        parts = [f"<h2>CDISC Conformance Report{agency}</h2>"]
        for name, v in self.validations.items():
            parts.append(f"<h3>{name}</h3>")
            try:
                report = v.get_tabular_report()
                html = getattr(report, "_repr_html_", lambda: str(report))()
                parts.append(html)
            except Exception:  # pragma: no cover - defensive
                parts.append("<p><em>(report unavailable)</em></p>")
        return "\n".join(parts)

    def __repr__(self) -> str:
        if self.is_rules:
            nr = self.native_result
            lines = ["ConformanceReport (Native Rules)"]
            if self.agency:
                lines.append(f"  Agency: {self.agency}")
            lines.append(f"  {nr.standard} {nr.version}")
            status = "PASS" if nr.all_passed else "FAIL"
            counts = nr.status_counts()
            counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            lines.append(f"  {len(nr.rule_results)} rules ({counts_str})")
            lines.append(f"  {nr.n_total_issues} issues — {status}")
            return "\n".join(lines)

        if self.is_core:
            core = self.core
            lines = ["ConformanceReport (CORE)"]
            if self.agency:
                lines.append(f"  Agency: {self.agency}")
            lines.append(f"  {core.standard} {core.version} — CORE {core.engine_version}")
            status = "PASS" if core.all_passed else "FAIL"
            counts = core.status_counts()
            counts_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
            lines.append(f"  {len(core.rules)} rules ({counts_str})")
            lines.append(f"  {core.n_total_issues} issues — {status}")
            return "\n".join(lines)

        lines = ["ConformanceReport"]
        if self.agency:
            lines.append(f"  Agency: {self.agency}")
        summary = self.summary()
        for name, s in summary.items():
            status = "PASS" if s["all_passed"] else f"FAIL ({s['n_failed']} test units)"
            lines.append(f"  [{name}] {s['n_steps']} steps — {status}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.__repr__()


# ── Module-level convenience entry point ──────────────────────────────────────


def validate_cdisc_submission(
    source: str | Path | dict | SubmissionPackage,
    standard: str | None = None,
    version: str | None = None,
    define: str | Path | Any | None = None,
    controlled_terminology: str | Sequence[str] | None = None,
    agency: str | None = None,
    ct_version: str | None = None,
    study_id: str | None = None,
    core: str | Sequence[str] | None = None,
    core_cwd: str | Path | None = None,
    cache: str | Path | None = None,
    workdir: str | Path | None = None,
) -> ConformanceReport:
    """Validate a CDISC submission with the CDISC CORE engine, in one call.

    Convenience wrapper that builds a [`SubmissionPackage`](`pointblank.SubmissionPackage`) from
    `source` and runs [`validate_conformance()`](`pointblank.SubmissionPackage`) with
    `engine="core"`. Requires an installed CORE executable (see `core`).

    Parameters
    ----------
    source
        The submission to validate. One of: a path to a folder of datasets (XPT / Dataset-JSON,
        with an optional `define.xml`), a mapping of dataset name to DataFrame, or an already-built
        `SubmissionPackage`.
    standard
        The CDISC standard (e.g., `"sdtmig"`). Defaults to `"sdtmig"` (or the package's `standard`
        when `source` is a `SubmissionPackage`).
    version
        The standard version (e.g., `"3.4"`). Defaults to `"3.4"` (or the package's value).
    define
        Optional Define-XML path (ignored when `source` is a `SubmissionPackage` — set it on the
        package instead). Auto-detected from a folder `source` when present.
    controlled_terminology
        CT package name(s) for CORE's `-ct` (e.g., `"sdtmct-2024-03-29"`).
    agency
        Optional agency rule-set selector recorded on the report.
    ct_version
        Optional Controlled Terminology version pin recorded on the package.
    study_id
        Optional study identifier.
    core
        How to invoke CORE — a path/name to the CORE executable, a full command prefix (e.g.,
        `["python", "core.py"]`), or `None` to auto-discover via the `POINTBLANK_CDISC_CORE`
        environment variable and then `PATH`.
    core_cwd
        Working directory to run CORE from; required when invoking a repo checkout.
    cache
        Path to CORE's rules cache directory (`-ca`).
    workdir
        Directory for materialized XPT and the CORE report. If `None`, a temporary directory is used.

    Returns
    -------
    ConformanceReport
        A CORE-form report (`is_core` is `True`).

    Examples
    --------
    ```python
    import pointblank as pb

    report = pb.validate_cdisc_submission(
        "study_xyz/sdtm/",
        standard="sdtmig",
        version="3.4",
        agency="FDA",
    )
    report.summary()
    ```
    """
    if isinstance(source, SubmissionPackage):
        package = source
    elif isinstance(source, dict):
        package = SubmissionPackage(
            datasets=source,
            define=define,
            standard=standard or "sdtmig",
            standard_version=version or "3.4",
            ct_version=ct_version,
            study_id=study_id,
        )
    elif isinstance(source, (str, Path)):
        folder = Path(source)
        if not folder.is_dir():
            raise NotADirectoryError(
                f"Expected a folder of datasets, but '{folder}' is not a directory."
            )
        package = SubmissionPackage.from_folder(
            folder,
            define=define,
            standard=standard or "sdtmig",
            standard_version=version or "3.4",
            ct_version=ct_version,
            study_id=study_id,
        )
    else:
        raise TypeError(
            "source must be a folder path, a {name: DataFrame} mapping, or a SubmissionPackage; "
            f"got {type(source).__name__}."
        )

    return package.validate_conformance(
        engine="core",
        agency=agency,
        standard=standard,
        version=version,
        controlled_terminology=controlled_terminology,
        core=core,
        core_cwd=core_cwd,
        cache=cache,
        workdir=workdir,
    )
