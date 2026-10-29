"""Submission-package model for CDISC conformance validation.

This module implements the *submission-package* layer described in PLAN_06: a data-level
analog of [`MetadataPackage`](`pointblank.MetadataPackage`) that understands the relationships
*between* datasets in a study (referential integrity, SUPP-- linkage, RELREC, ADaM ⇄ SDTM
traceability) and drives Pointblank validation across the whole package.

Where [`validate_sdtm()`](`pointblank.validate_sdtm`) and
[`validate_adam()`](`pointblank.validate_adam`) validate a *single* dataset structurally, the
[`SubmissionPackage`](`pointblank.SubmissionPackage`) validates a study as a graph of related
datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from great_tables import GT
    from pointblank.metadata._cdisc_core import ParsedCoreReport
    from pointblank.metadata._conformance.result import NativeConformanceResult
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
    """Whether a dataset name is an ADaM dataset (conventionally prefixed `AD`)."""
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

    Supports both the Dataset-JSON 1.1 top-level `columns`/`rows` layout and the older
    `clinicalData`/`referenceData` -> `itemGroupData` nesting.

    Returns
    -------
    tuple[Any, str | None]
        The DataFrame and the dataset name (domain) if discoverable, else `None`.
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
    """A study's datasets, gathered for CDISC conformance validation.

    A `SubmissionPackage` holds the datasets of a study (SDTM domains, SUPP-- qualifiers, RELREC
    and ADaM datasets) together with the study's Define-XML. Where
    [`validate_sdtm()`](`pointblank.validate_sdtm`) and
    [`validate_adam()`](`pointblank.validate_adam`) look at one dataset at a time, a package
    knows about all of them, so it can check how they relate: for example, whether every subject
    in AE has a record in DM. It is the data-level counterpart of
    [`MetadataPackage`](`pointblank.MetadataPackage`), which holds the metadata of many datasets.

    A package can be built from DataFrames, as here, or from a folder of XPT and Dataset-JSON
    files with [`from_folder()`](`pointblank.SubmissionPackage.from_folder`). Its
    [`validate_conformance()`](`pointblank.SubmissionPackage.validate_conformance`) method runs
    the conformance checks and returns a [`ConformanceReport`](`pointblank.ConformanceReport`).
    The package also has some helpers for looking across datasets, such as
    [`orphan_ids()`](`pointblank.SubmissionPackage.orphan_ids`), which finds the subjects in one
    dataset that are missing from another.

    Parameters
    ----------
    datasets
        A dictionary that maps each dataset name (a domain code such as `"DM"`, `"AE"`,
        `"SUPPAE"` or `"ADSL"`) to the dataset, as a Polars or Pandas DataFrame. The names are
        case-insensitive and are stored in uppercase. The default is an empty dictionary.
    define
        The study's Define-XML, as a path to a `define.xml` file or as a
        [`MetadataPackage`](`pointblank.MetadataPackage`) from
        [`import_metadata()`](`pointblank.import_metadata`). With the built-in rules engine, it
        activates the rules that check the datasets against Define-XML. The CORE engine only
        accepts a path. The default is `None`.
    ct_version
        The version of CDISC Controlled Terminology that the study uses (such as
        `"2024-03-29"`). It is shown by [`summary()`](`pointblank.SubmissionPackage.summary`) but
        doesn't affect the checks. To choose the terminology the built-in engine checks against,
        use the `ct_packages=` argument of `validate_conformance()`. The default is `None`.
    standard
        The data standard that the datasets follow, `"sdtmig"` or `"adamig"`. The default is
        `"sdtmig"`.
    standard_version
        The version of the standard, such as `"3.4"` for SDTMIG 3.4. The default is `"3.4"`.
    study_id
        A study identifier. It is shown by `summary()` and used in the labels of the validations
        made by `validate_conformance(engine="validate")`. The default is `None`.

    Examples
    --------
    Let's build a package from a Demographics (DM) and an Adverse Events (AE) dataset. One of the
    adverse events belongs to subject `STUDY01-009`, who isn't in DM:

    ```{python}
    import pointblank as pb
    import polars as pl

    dm = pl.DataFrame(
        {
            "STUDYID": ["STUDY01"] * 3,
            "DOMAIN": ["DM"] * 3,
            "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
            "SEX": ["M", "F", "F"],
        }
    )
    ae = pl.DataFrame(
        {
            "STUDYID": ["STUDY01"] * 3,
            "DOMAIN": ["AE"] * 3,
            "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-009"],
            "AESEQ": [1, 1, 1],
            "AETERM": ["HEADACHE", "NAUSEA", "RASH"],
        }
    )

    study = pb.SubmissionPackage(datasets={"dm": dm, "AE": ae}, study_id="STUDY01")

    print(study)
    ```

    The dataset names are stored in uppercase, so `"dm"` became `DM`. The package can find the
    subject in AE that has no DM record:

    ```{python}
    study.orphan_ids("AE")
    ```

    The validation checks the whole package. The `"validate"` engine builds a
    [`Validate`](`pointblank.Validate`) for each dataset, with the structural checks from the SDTM
    templates and the checks across datasets:

    ```{python}
    report = study.validate_conformance(engine="validate")

    print(report)
    ```

    The AE validation has one failing test unit, from the check that each `USUBJID` in AE exists
    in DM. The default engine, the built-in rules engine, checks the package against the SDTMIG
    rules instead. See
    [`validate_conformance()`](`pointblank.SubmissionPackage.validate_conformance`) for how the
    engines differ.

    See Also
    --------
    validate_sdtmig : Check SDTM datasets against the SDTMIG rules, without building a package.
    validate_cdisc_submission : Check a submission with the CDISC CORE engine.
    ConformanceReport : The report that validate_conformance() returns.
    MetadataPackage : The metadata-level counterpart of a submission package.
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
        """Build a `SubmissionPackage` from a folder of dataset files.

        Submission datasets are usually delivered as files, one per dataset, in a single folder.
        The `from_folder()` method reads the folder so the datasets don't have to be loaded one by
        one. It reads these files (the folder's subfolders are not searched):

        - SAS Transport files (`.xpt`), which need the `pyreadstat` package. The dataset name is
          the file name without its extension, in uppercase, so `dm.xpt` becomes `DM`.
        - CDISC Dataset-JSON files (`.json`), in the Dataset-JSON 1.1 layout (top-level `columns`
          and `rows`) or the older layout (`clinicalData` or `referenceData`). The dataset name is
          taken from the file's `name` (or its item group OID), or else from the file name.
          `.json` files that aren't Dataset-JSON are skipped.
        - A Define-XML file: an `.xml` file whose name starts with `define`. It is used as the
          package's `define` unless `define=` is given.

        The datasets are read into Pandas DataFrames. The package remembers the folder, so that
        `validate_conformance(engine="core")` can give the files straight to the CORE engine
        instead of writing the datasets out again.

        Parameters
        ----------
        path
            The path to the folder that holds the dataset files.
        define
            The study's Define-XML, as a path to a `define.xml` file or a
            [`MetadataPackage`](`pointblank.MetadataPackage`). If `None` (the default), a
            Define-XML file in the folder is used, if there is one.
        standard
            The data standard that the datasets follow, `"sdtmig"` or `"adamig"`. The default is
            `"sdtmig"`.
        standard_version
            The version of the standard. The default is `"3.4"`.
        ct_version
            The version of CDISC Controlled Terminology that the study uses, which is shown by
            [`summary()`](`pointblank.SubmissionPackage.summary`). The default is `None`.
        study_id
            A study identifier. The default is `None`.

        Returns
        -------
        SubmissionPackage
            A package with a dataset for each dataset file in the folder.

        Raises
        ------
        NotADirectoryError
            If `path` isn't a folder.
        ImportError
            If the folder has an `.xpt` file and `pyreadstat` isn't installed.

        Examples
        --------
        Let's make a folder with a DM dataset as an XPT file and a Define-XML file for the study.
        Both files come with Pointblank:

        ```{python}
        import pointblank as pb
        import shutil
        import tempfile
        from pathlib import Path

        folder = Path(tempfile.mkdtemp())

        shutil.copy(pb.load_metadata_example("dm.xpt"), folder / "dm.xpt")
        shutil.copy(pb.load_metadata_example("define.xml"), folder / "define.xml")

        study = pb.SubmissionPackage.from_folder(folder, study_id="XYZ789")

        print(study)
        ```

        The package has the `DM` dataset from `dm.xpt`, and it found the Define-XML file. The
        dataset is a Pandas DataFrame:

        ```{python}
        study.get_dataset("DM").head(3)
        ```

        Since a Define-XML file was found, the built-in rules engine also runs the rules that
        check the datasets against it. Here, one of the subjects has a `RACE` of `BLACK`, which
        isn't a term in the study's codelist for `RACE`:

        ```{python}
        report = study.validate_conformance()

        [i["message"] for i in report.issues() if i["rule_type"] == "DEFINE_CODELIST_CHECK"]
        ```

        The other issues in the report come from rules that don't need Define-XML, such as the
        rules for variables that DM should have.
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
        """The names of the datasets in the package, in alphabetical order.

        The names are the domain codes (such as `"AE"` and `"DM"`) under which the datasets were
        added, in uppercase. This is a property, so it's used without parentheses.

        Examples
        --------
        Let's make a package with three datasets and get their names:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame({"USUBJID": ["S-001", "S-002"]})
        ae = pl.DataFrame({"USUBJID": ["S-001"], "AESEQ": [1]})
        suppae = pl.DataFrame({"USUBJID": ["S-001"], "RDOMAIN": ["AE"]})

        study = pb.SubmissionPackage(datasets={"dm": dm, "SUPPAE": suppae, "ae": ae})

        study.domains
        ```

        The names are in uppercase and in alphabetical order, whatever order the datasets were
        given in. A name can also be tested with `in`, which ignores case (`"ae" in study` is
        `True`), and `len(study)` gives the number of datasets.
        """
        return sorted(self.datasets.keys())

    def __contains__(self, name: str) -> bool:
        return name.upper() in self.datasets

    def __getitem__(self, name: str) -> Any:
        return self.datasets[name.upper()]

    def __len__(self) -> int:
        return len(self.datasets)

    def get_dataset(self, name: str) -> Any:
        """Get one of the package's datasets by name.

        The dataset is returned as it was given to the package, so a Polars DataFrame stays a
        Polars DataFrame. Indexing the package (as in `study["DM"]`) also gets a dataset, but
        `get_dataset()` gives a more helpful error for a name that isn't in the package.

        Parameters
        ----------
        name
            The name of the dataset, such as `"DM"`. This is case-insensitive.

        Returns
        -------
        Any
            The dataset, as a DataFrame.

        Raises
        ------
        KeyError
            If the package has no dataset with that name. The error message lists the names that
            the package has.

        Examples
        --------
        Let's make a package with two datasets and get one of them:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame({"USUBJID": ["S-001", "S-002"], "SEX": ["M", "F"]})
        ae = pl.DataFrame({"USUBJID": ["S-001"], "AESEQ": [1], "AETERM": ["HEADACHE"]})

        study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae})

        study.get_dataset("ae")
        ```

        The name is case-insensitive, so `"ae"` gets the `AE` dataset. Asking for a dataset that
        isn't in the package raises a `KeyError`:

        ```{python}
        try:
            study.get_dataset("LB")
        except KeyError as e:
            print(e)
        ```

        The message lists the datasets that the package has.
        """
        key = name.upper()
        if key not in self.datasets:
            raise KeyError(f"No dataset named '{name}'. Available: {self.domains}")
        return self.datasets[key]

    @property
    def metadata(self) -> MetadataPackage | None:
        """The package's Define-XML, as a `MetadataPackage`.

        If the package's `define` is a path to a `define.xml` file, the file is imported with
        [`import_metadata()`](`pointblank.import_metadata`) the first time this property is used,
        and the result is kept for later use. If `define` is already a
        [`MetadataPackage`](`pointblank.MetadataPackage`), that is returned. The metadata has a
        [`MetadataImport`](`pointblank.MetadataImport`) for each dataset that the Define-XML
        describes, with its variables and codelists. The property is `None` if the package has no
        Define-XML. This is a property, so it's used without parentheses.

        Examples
        --------
        Let's make a package with the DM dataset and the Define-XML file that come with Pointblank:

        ```{python}
        import pointblank as pb
        import pyreadstat

        dm, _ = pyreadstat.read_xport(pb.load_metadata_example("dm.xpt"))

        study = pb.SubmissionPackage(
            datasets={"DM": dm}, define=pb.load_metadata_example("define.xml")
        )

        print(study.metadata)
        ```

        The Define-XML describes two datasets, DM and AE, though the package only has DM. The
        metadata of a dataset can be looked at in more detail, such as the values allowed by the
        codelist of a variable:

        ```{python}
        race = study.metadata["DM"].get_variable("RACE")

        race.allowed_values
        ```

        `BLACK` isn't one of them, though one of the subjects in DM has it as their `RACE`.
        Without a `define`, the property is `None`.
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
        """Get the unique subject identifiers in a dataset.

        `USUBJID` identifies each subject across all the datasets of a study. DM has one record
        per subject, so its `USUBJID` values are the study's subjects, and the subjects in the
        other datasets should all be among them. The `subject_ids()` method gets these values for
        one dataset, which is useful for comparing the subjects of two datasets.

        Parameters
        ----------
        dataset
            The name of the dataset, which is case-insensitive. The default is `"DM"`.

        Returns
        -------
        set
            The different values of `USUBJID` in the dataset, leaving out nulls. The set is empty
            if the package has no dataset with that name, or if the dataset has no `USUBJID`
            column.

        Examples
        --------
        Let's make a package where the AE dataset has two records for one subject:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame({"USUBJID": ["S-001", "S-002", "S-003"]})
        ae = pl.DataFrame({"USUBJID": ["S-001", "S-001", "S-003"], "AESEQ": [1, 2, 1]})

        study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae})

        sorted(study.subject_ids("AE"))
        ```

        Each subject appears once in the set, however many records they have. Sets can be
        compared directly, so here's a check that every subject with an adverse event is in DM:

        ```{python}
        study.subject_ids("AE") <= study.subject_ids()
        ```

        For a dataset that isn't in the package, `subject_ids()` returns an empty set rather than
        raising an error. To find the subjects that are missing from DM, use
        [`orphan_ids()`](`pointblank.SubmissionPackage.orphan_ids`).
        """
        key = dataset.upper()
        if key not in self.datasets:
            return set()
        return _column_value_set(self.datasets[key], "USUBJID")

    def orphan_ids(self, child: str, parent: str = "DM", column: str = "USUBJID") -> set:
        """Find the key values in one dataset that are missing from another.

        In a study, the records of one dataset often refer to those of another. Each subject in
        AE should have a record in DM, for example. A value with nothing to refer to (an
        *orphan*) usually means that a record is missing or that a key was mistyped. The
        `orphan_ids()` method finds these values, by default the subjects of a dataset that are
        missing from DM.

        Parameters
        ----------
        child
            The name of the dataset whose values are looked up, such as `"AE"`.
        parent
            The name of the dataset that the values should be in. The default is `"DM"`.
        column
            The key column, which is compared between the two datasets. The default is
            `"USUBJID"`.

        Returns
        -------
        set
            The values of `column` in `child` that aren't in `column` of `parent`. Null values
            are left out. If `child` has no such column, the set is empty. If `parent` has no such
            column, every value of `child` is returned.

        Raises
        ------
        KeyError
            If `child` or `parent` isn't a dataset in the package.

        Examples
        --------
        Let's make a package where the AE dataset has a subject, `S-009`, who isn't in DM:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame({"USUBJID": ["S-001", "S-002", "S-003"]})
        ae = pl.DataFrame({"USUBJID": ["S-001", "S-009", "S-009"], "AESEQ": [1, 1, 2]})

        study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae})

        study.orphan_ids("AE")
        ```

        The orphan subject is found once, though they have two records in AE. The datasets that
        are compared can be changed with `parent=`. Here, a SUPPAE dataset holds supplemental
        qualifiers for adverse events, so each of its subjects should have a record in AE:

        ```{python}
        suppae = pl.DataFrame({"USUBJID": ["S-001", "S-002"], "QNAM": ["AETRTEM", "AETRTEM"]})
        study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae, "SUPPAE": suppae})

        print(study.orphan_ids("SUPPAE"))
        print(study.orphan_ids("SUPPAE", parent="AE"))
        ```

        Both SUPPAE subjects are in DM, but `S-002` has no adverse events, so their supplemental
        qualifier has no AE record to belong to. A key other than `USUBJID` can be compared with
        `column=`.
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
        """Check the whole package for conformance to its CDISC standard.

        A submission has to conform to the CDISC standard that it uses, both within each dataset
        and across the datasets. The `validate_conformance()` method checks this for all the
        datasets of the package at once and returns a
        [`ConformanceReport`](`pointblank.ConformanceReport`). There are three ways of doing the
        checks, chosen with `engine=`:

        - `"native"` (the default) runs Pointblank's bundled catalog of conformance rules for the
          package's standard and version, as [`validate_sdtmig()`](`pointblank.validate_sdtmig`)
          does. The catalog has 362 rules for SDTMIG 3.4, which check the values of variables
          against their controlled terminology, the format of dates, the variables and domains
          that are required, and (if there's a Define-XML) the agreement between the datasets and
          their Define-XML. SDTMIG 3.4 is the only standard and version with a bundled catalog, so
          for any other (such as an ADaM package), the `"validate"` checks are run instead.
        - `"validate"` builds a [`Validate`](`pointblank.Validate`) plan for each dataset, with
          the structural checks of [`validate_sdtm()`](`pointblank.validate_sdtm`) or
          [`validate_adam()`](`pointblank.validate_adam`), and (if `cross_dataset=True`) checks
          that the datasets agree with each other. These cross-dataset checks are that every
          `USUBJID` of an SDTM domain is in DM, that a SUPP-- or RELREC record's `RDOMAIN` is a
          dataset in the package and its `USUBJID` is in DM, that a SUPP-- record's `IDVAR` and
          `IDVARVAL` identify a record of its parent domain, that every `USUBJID` of ADSL is in
          DM, and that every `USUBJID` of another ADaM dataset is in ADSL. The rules catalog
          doesn't have these checks, so the `"validate"` engine is the one that finds orphan
          records.
        - `"core"` runs the CDISC CORE engine (the `cdisc-rules-engine` command-line tool, which
          has to be installed separately), with the published CDISC and agency rules. The
          datasets are written out as XPT files (or read from the package's `source_folder`, if
          it was made with [`from_folder()`](`pointblank.SubmissionPackage.from_folder`)), CORE is
          run on them, and its report is read back. This is what
          [`validate_cdisc_submission()`](`pointblank.validate_cdisc_submission`) does.

        Parameters
        ----------
        agency
            The regulatory agency that the submission is for, such as `"FDA"` or `"PMDA"`. This is
            a label that is shown in the report. It doesn't change which rules are run. The default
            is `None`.
        engine
            The engine for the checks: `"native"`, `"validate"` or `"core"`. The default is
            `"native"`.
        cross_dataset
            Whether to add the cross-dataset checks to the `"validate"` engine's plans. The default
            is `True`.
        thresholds
            Thresholds for each of the `"validate"` engine's plans, given as for
            [`Validate`](`pointblank.Validate`). They set the severity (warning, error or critical)
            of a failing step. The default is `None`.
        interrogate
            Whether the `"validate"` engine interrogates its plans before returning them. The
            default is `True`.
        standard
            The CDISC standard to check against, for the `"native"` and `"core"` engines. The
            default is the package's `standard`.
        version
            The version of the standard, for the `"native"` and `"core"` engines. The default is the
            package's `standard_version`.
        ct_packages
            The bundled controlled terminology packages for the `"native"` engine, as a list of
            names such as `["sdtm-ct-2024-09-27"]`. The default is the latest bundled package.
        define_xml
            A Define-XML for the `"native"` engine, as a path or a
            [`MetadataPackage`](`pointblank.MetadataPackage`). The default is the package's
            `define`. Without a Define-XML, the 12 rules that need one are not applicable.
        controlled_terminology
            The controlled terminology package or packages for the `"core"` engine, such as
            `"sdtmct-2024-03-29"`. They are given to CORE's `-ct` option. The default is `None`.
        core
            How to run CORE, for the `"core"` engine. This can be the path or name of the CORE
            executable, or a command as a list (such as `["python", "core.py"]`). With the default
            of `None`, the command is taken from the `POINTBLANK_CDISC_CORE` environment variable,
            or else CORE is looked for on the `PATH`.
        core_cwd
            The directory to run CORE from, for the `"core"` engine. This is needed when CORE is
            run from a checkout of its repository, since it looks for its `resources/` folder in
            the current directory. The default is `None`.
        cache
            The path to CORE's rules cache (its `-ca` option), for the `"core"` engine. The default
            is `None`.
        workdir
            A directory for the XPT files and the CORE report, for the `"core"` engine. With the
            default of `None`, a temporary directory is used and then deleted.

        Returns
        -------
        ConformanceReport
            The report. It holds the results of the rules for the `"native"` engine, a validation
            for each dataset for the `"validate"` engine, and CORE's results for the `"core"`
            engine.

        Raises
        ------
        ValueError
            If `engine` isn't one of `"native"`, `"validate"` or `"core"`.
        FileNotFoundError
            If one of the `ct_packages` isn't bundled with Pointblank.

        Examples
        --------
        Let's make a package with a DM and an AE dataset. A `SEX` value in DM isn't in the SEX
        codelist, and the second subject in AE (`STUDY01-009`) has no record in DM:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame(
            {
                "STUDYID": ["STUDY01"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
                "SUBJID": ["001", "002", "003"],
                "SITEID": ["S01"] * 3,
                "SEX": ["M", "F", "FEMALE"],
                "COUNTRY": ["USA"] * 3,
            }
        )
        ae = pl.DataFrame(
            {
                "STUDYID": ["STUDY01"] * 2,
                "DOMAIN": ["AE"] * 2,
                "USUBJID": ["STUDY01-001", "STUDY01-009"],
                "AESEQ": [1, 1],
                "AETERM": ["HEADACHE", "NAUSEA"],
                "AEDECOD": ["Headache", "Nausea"],
            }
        )

        study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae}, study_id="STUDY01")

        report = study.validate_conformance()

        print(report)
        ```

        Of the 362 rules, 11 fail. The 12 that are not applicable need a Define-XML. Here are the
        failing ones:

        ```{python}
        for rule in report.rules(status="fail"):
            print(rule.rule_id, rule.dataset, rule.message)
        ```

        Most of them are about variables that the datasets lack, such as `RACE` and `AESER`, and
        SDTM-121 and SDTM-122 fail since the package has no Trial Summary (TS) or Trial Arms (TA)
        dataset. SDTM-007 finds the `SEX` value of `"FEMALE"`. The rules don't look for orphan
        records, though, so for those the `"validate"` engine is needed:

        ```{python}
        report = study.validate_conformance(engine="validate")

        for issue in report.issues():
            print(issue)
        ```

        The `"validate"` engine has a validation for each dataset. In AE, the last step is the
        check that each `USUBJID` is in DM, and it finds the one orphan record. In DM, the failing
        step is the length check on `SEX`, since `"FEMALE"` is longer than the two characters that
        SDTM allows. Each validation can be looked at with
        [`get_validation()`](`pointblank.ConformanceReport.get_validation`).

        See Also
        --------
        validate_sdtmig : Run the rules catalog on a dict of datasets.
        validate_cdisc_submission : Run CDISC CORE on a folder of datasets.
        ConformanceReport : The report that is returned.
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
                agency=agency,
                standard=std,
                version=ver,
                ct_packages=ct_packages,
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
        """Run the built-in rule-based engine; returns None if no catalog is bundled."""
        from pointblank.metadata._conformance.engine import NativeConformanceEngine
        from pointblank.metadata._conformance.rule_loader import RuleLoader

        if not RuleLoader.catalog_path(standard, version).exists():
            return None

        engine = NativeConformanceEngine(
            standard=standard, version=version, ct_packages=ct_packages
        )
        # Prefer explicit define_xml argument; fall back to the package's define (a path or an
        # imported MetadataPackage, both of which the engine accepts).
        _define = define_xml if define_xml is not None else self.define
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

    def _add_relrec_checks(self, validation: Validate, data: Any, cols: set, has_dm: bool) -> None:
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
        """Summarize the package's contents as text.

        The summary is a quick way to check what a package holds, such as after reading it with
        [`from_folder()`](`pointblank.SubmissionPackage.from_folder`). It has the study
        identifier, the standard and its version, the controlled terminology version, whether
        there is a Define-XML, and the names of the datasets. Lines for the study identifier, the
        terminology version and the Define-XML are left out when the package doesn't have them.
        Printing the package gives the same text.

        Returns
        -------
        str
            The summary, over several lines.

        Examples
        --------
        Let's make a package and summarize it:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame({"USUBJID": ["S-001", "S-002"]})
        ae = pl.DataFrame({"USUBJID": ["S-001"], "AESEQ": [1]})

        study = pb.SubmissionPackage(
            datasets={"DM": dm, "AE": ae}, study_id="STUDY01", ct_version="2024-09-27"
        )

        print(study.summary())
        ```

        The package has no `define`, so there's no line for a Define-XML. The datasets are listed
        in alphabetical order.
        """
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
            (True if (v is None and na_pass) else (v in valid_values)) for v in df[column].to_list()
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
    """The results of a CDISC conformance check.

    A `ConformanceReport` holds the results of checking a set of CDISC datasets for conformance
    to their standard. It is returned by [`validate_sdtmig()`](`pointblank.validate_sdtmig`),
    [`validate_cdisc_submission()`](`pointblank.validate_cdisc_submission`) and the
    `SubmissionPackage` method
    [`validate_conformance()`](`pointblank.SubmissionPackage.validate_conformance`), and is
    rarely made directly. The report has one of three forms, depending on the engine that
    did the checks:

    - **Built-in rules** ([`is_rules`](`pointblank.ConformanceReport.is_rules`) is `True`): the
      results of Pointblank's bundled catalog of SDTMIG rules. Each rule has a status of
      `"pass"`, `"fail"`, `"error"` (the rule couldn't be evaluated), `"not_applicable"` (an input
      that it needs, such as a Define-XML, wasn't given) or `"not_supported"`. The rules that
      check values give the failing records, which are in
      [`findings_df()`](`pointblank.ConformanceReport.findings_df`) and
      [`get_findings_table()`](`pointblank.ConformanceReport.get_findings_table`).
    - **CDISC CORE** ([`is_core`](`pointblank.ConformanceReport.is_core`) is `True`): the results
      of the CDISC CORE engine, with CORE's own rule statuses (such as `"SUCCESS"`,
      `"ISSUE REPORTED"` and `"SKIPPED"`). A report of this form can also be made from a saved
      CORE report with [`from_core_report()`](`pointblank.ConformanceReport.from_core_report`).
    - **Validate-based**: a [`Validate`](`pointblank.Validate`) validation for each dataset, from
      `SubmissionPackage.validate_conformance(engine="validate")`. Each validation can be had with
      [`get_validation()`](`pointblank.ConformanceReport.get_validation`).

    The methods [`summary()`](`pointblank.ConformanceReport.summary`),
    [`issues()`](`pointblank.ConformanceReport.issues`) and
    [`all_passed()`](`pointblank.ConformanceReport.all_passed`) work for all three forms. In a
    notebook, a report of the built-in rules is shown as the table of
    [`get_tabular_report()`](`pointblank.ConformanceReport.get_tabular_report`). A CORE report is
    shown as a short summary with a table of its issues, and a Validate-based report as the
    validation report of each dataset. Printing a report of any form gives a text summary.

    Parameters
    ----------
    validations
        The validation for each dataset, keyed by dataset name, for a Validate-based report. The
        default is an empty dict.
    package
        The [`SubmissionPackage`](`pointblank.SubmissionPackage`) that was checked. The default is
        `None`.
    agency
        The regulatory agency, such as `"FDA"`, which is only used as a label in the report. The
        default is `None`.
    core
        CORE's results, for a CDISC CORE report. The default is `None`.
    native_result
        The results of the rules, for a report of the built-in rules. The default is `None`.

    Examples
    --------
    Let's check a small DM dataset with the built-in rules. It has a date that isn't in the
    ISO 8601 format, a negative age and a `SEX` value that isn't in its codelist:

    ```{python}
    import pointblank as pb
    import polars as pl

    dm = pl.DataFrame(
        {
            "STUDYID": ["STUDY01"] * 3,
            "DOMAIN": ["DM"] * 3,
            "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
            "RFSTDTC": ["2024-01-15", "15JAN2024", "2024-01-17"],
            "AGE": [34, 51, -38],
            "SEX": ["M", "F", "FEMALE"],
        }
    )

    report = pb.validate_sdtmig({"DM": dm})

    print(report)
    ```

    Ten of the 362 rules fail. Most of them are about what the dataset lacks, such as the DM
    variables `SUBJID` and `RACE` and the TS and TA datasets. The failing records are found by
    three of the rules:

    ```{python}
    report.findings_df().select("rule_id", "row_index", "checked_column", "checked_value")
    ```

    The `row_index` of a finding is the 0-based position of the record in its dataset. To see the
    results of all the rules as a table, use
    [`get_tabular_report()`](`pointblank.ConformanceReport.get_tabular_report`).

    See Also
    --------
    validate_sdtmig : Check SDTM datasets with the built-in rules.
    validate_cdisc_submission : Check a folder of datasets with CDISC CORE.
    SubmissionPackage : A set of datasets to check together.
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
        """Make a report from the JSON report of a CDISC CORE run.

        CDISC CORE can be run on its own (as `core validate ... -of JSON`), and its JSON report
        can be kept for later. The `from_core_report()` class method reads such a report into a
        `ConformanceReport`, which gives it the same methods as the report of
        [`validate_cdisc_submission()`](`pointblank.validate_cdisc_submission`). The JSON report
        has five sections: `Conformance_Details` (the standard, its version and the version of
        CORE), `Dataset_Details`, `Issue_Summary` (the number of issues for each rule and
        dataset), `Issue_Details` (one entry for each failing record) and `Rules_Report` (the
        status of each rule). A file written by
        [`to_json()`](`pointblank.ConformanceReport.to_json`) from a CORE report has the same
        sections, so it can be read back with this method.

        Parameters
        ----------
        report
            The CORE report, as a dict (such as from `json.load()`) or as the already parsed report
            of an earlier `ConformanceReport` (its `core` attribute).
        package
            The [`SubmissionPackage`](`pointblank.SubmissionPackage`) that CORE checked. The
            default is `None`.
        agency
            The regulatory agency, such as `"FDA"`, which is only used as a label in the report.
            The default is `None`.

        Returns
        -------
        ConformanceReport
            A report of CORE's results, whose
            [`is_core`](`pointblank.ConformanceReport.is_core`) is `True`.

        Examples
        --------
        Let's write a short report by hand in CORE's format, for three rules on a DM and an AE
        dataset. One rule found an AE record whose subject isn't in DM, one rule passed and one
        was skipped:

        ```{python}
        import pointblank as pb

        core_json = {
            "Conformance_Details": {
                "CORE_Engine_Version": "0.16.0",
                "Standard": "SDTMIG",
                "Version": "V3.4",
            },
            "Dataset_Details": [
                {"filename": "dm.xpt", "label": "Demographics", "length": 3},
                {"filename": "ae.xpt", "label": "Adverse Events", "length": 2},
            ],
            "Issue_Summary": [
                {
                    "dataset": "ae.xpt",
                    "core_id": "CORE-000201",
                    "message": "USUBJID is not found in DM.USUBJID",
                    "issues": 1,
                }
            ],
            "Issue_Details": [
                {
                    "core_id": "CORE-000201",
                    "message": "USUBJID is not found in DM.USUBJID",
                    "dataset": "ae.xpt",
                    "USUBJID": "STUDY01-009",
                    "row": 2,
                    "SEQ": 1,
                    "variables": ["USUBJID"],
                    "values": ["STUDY01-009"],
                }
            ],
            "Rules_Report": [
                {"core_id": "CORE-000201", "status": "ISSUE REPORTED"},
                {"core_id": "CORE-000351", "status": "SUCCESS"},
                {"core_id": "CORE-000087", "status": "SKIPPED"},
            ],
        }

        report = pb.ConformanceReport.from_core_report(core_json, agency="FDA")

        print(report)
        ```

        The rule statuses are CORE's own. A real report would usually be read from a file, with
        `json.load()`. The failing record is in
        [`findings()`](`pointblank.ConformanceReport.findings`):

        ```{python}
        report.findings()
        ```

        CORE gives the row of a record as a 1-based number, so `row=2` is the second record of the
        AE dataset.
        """
        from pointblank.metadata._cdisc_core import ParsedCoreReport, parse_core_report

        parsed = report if isinstance(report, ParsedCoreReport) else parse_core_report(report)
        return cls(package=package, agency=agency, core=parsed)

    @property
    def is_core(self) -> bool:
        """Whether the report holds the results of CDISC CORE.

        This is `True` for the report of
        [`validate_cdisc_submission()`](`pointblank.validate_cdisc_submission`), of
        `SubmissionPackage.validate_conformance(engine="core")` and of
        [`from_core_report()`](`pointblank.ConformanceReport.from_core_report`). Some methods only
        work for one form of report, so this tells code which form it has. This is a property, so
        it's used without parentheses.

        Examples
        --------
        Let's check a DM dataset with the built-in rules and look at the form of the report:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame({"STUDYID": ["STUDY01"], "USUBJID": ["STUDY01-001"], "SEX": ["F"]})

        report = pb.validate_sdtmig({"DM": dm})

        print(report.is_core, report.is_rules)
        ```

        The report is of the built-in rules, so `is_core` is `False` and
        [`is_rules`](`pointblank.ConformanceReport.is_rules`) is `True`.
        """
        return self.core is not None

    @property
    def is_rules(self) -> bool:
        """Whether the report holds the results of Pointblank's built-in rules.

        This is `True` for the report of [`validate_sdtmig()`](`pointblank.validate_sdtmig`) and
        of `SubmissionPackage.validate_conformance()` with its default engine (when the package's
        standard has a bundled rules catalog). Only these reports have
        [`findings_df()`](`pointblank.ConformanceReport.findings_df`),
        [`get_findings_table()`](`pointblank.ConformanceReport.get_findings_table`) and
        [`get_tabular_report()`](`pointblank.ConformanceReport.get_tabular_report`). This is a
        property, so it's used without parentheses.

        Examples
        --------
        Let's check a package with each of the two Pointblank engines:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame({"STUDYID": ["STUDY01"], "USUBJID": ["STUDY01-001"], "SEX": ["F"]})
        study = pb.SubmissionPackage(datasets={"DM": dm})

        print(study.validate_conformance().is_rules)
        print(study.validate_conformance(engine="validate").is_rules)
        ```

        The `"validate"` engine gives a report of [`Validate`](`pointblank.Validate`) validations,
        so its `is_rules` is `False`, as is its
        [`is_core`](`pointblank.ConformanceReport.is_core`).
        """
        return self.native_result is not None

    def all_passed(self) -> bool:
        """Check whether the datasets conform, with nothing failing.

        This gives a single answer for a conformance check, as is needed in a script or a
        pipeline that should stop when the data don't conform. What counts as passing depends on
        the form of the report:

        - For the built-in rules, no rule has the status `"fail"`. A failing rule of `"Warning"`
          sensitivity counts too. Rules that are not applicable or couldn't be evaluated don't.
        - For CDISC CORE, no rule reported an issue or an execution error.
        - For a Validate-based report, every step of every validation passed (with no failing test
          units).

        Returns
        -------
        bool
            `True` if nothing failed, otherwise `False`.

        Examples
        --------
        Let's check a DM dataset with the built-in rules:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame({"STUDYID": ["STUDY01"], "USUBJID": ["STUDY01-001"], "SEX": ["F"]})

        report = pb.validate_sdtmig({"DM": dm})

        report.all_passed()
        ```

        The dataset lacks many of the variables that DM requires, and the TS and TA datasets are
        missing, so some rules fail. The rules that fail can be listed with
        [`rules()`](`pointblank.ConformanceReport.rules`), as in `report.rules(status="fail")`.
        """
        if self.is_core:
            return self.core.all_passed
        if self.is_rules:
            return self.native_result.all_passed
        return all(v.all_passed() for v in self.validations.values())

    def __getitem__(self, name: str) -> Validate:
        return self.validations[name.upper()]

    def get_validation(self, name: str) -> Validate:
        """Get the validation of one dataset from a Validate-based report.

        The `"validate"` engine of the `SubmissionPackage` method
        [`validate_conformance()`](`pointblank.SubmissionPackage.validate_conformance`) makes a
        [`Validate`](`pointblank.Validate`) validation for each dataset. The `get_validation()`
        method gets one of them, so that its validation report or its failing rows can be looked
        at. Indexing the report (as in `report["AE"]`) does the same, but
        `get_validation()` gives a more helpful error for a name that isn't in the report. Reports
        of the built-in rules and of CDISC CORE don't have validations.

        Parameters
        ----------
        name
            The name of the dataset, such as `"AE"`. This is case-insensitive.

        Returns
        -------
        Validate
            The validation of the dataset.

        Raises
        ------
        KeyError
            If the report has no validation for that dataset. The error message lists the datasets
            that it has validations for.

        Examples
        --------
        ```{python}
        #| echo: false
        #| output: false
        import pointblank as pb
        pb.config(
            report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False
        )
        ```
        Let's check a package where the AE dataset has a subject, `STUDY01-009`, who isn't in DM:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame({"USUBJID": ["STUDY01-001", "STUDY01-002"], "DOMAIN": ["DM", "DM"]})
        ae = pl.DataFrame(
            {
                "USUBJID": ["STUDY01-001", "STUDY01-009"],
                "DOMAIN": ["AE", "AE"],
                "AESEQ": [1, 1],
            }
        )

        study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae})
        report = study.validate_conformance(engine="validate")

        report.get_validation("ae")
        ```

        The last step checks that each `USUBJID` in AE is in DM, and it has one failing test unit.
        The validation is an ordinary `Validate` object, so methods such as
        [`get_step_report()`](`pointblank.Validate.get_step_report`) can be used on it to see the
        failing record.
        """
        key = name.upper()
        if key not in self.validations:
            raise KeyError(f"No validation for '{name}'. Available: {sorted(self.validations)}")
        return self.validations[key]

    def summary(self) -> dict:
        """Summarize the results of the conformance check as a dict.

        The summary has the counts that are shown when a report is printed, in a form that code
        can use, such as for logging the results of a run or for comparing two runs. Its keys
        depend on the form of the report.

        Returns
        -------
        dict
            For a report of the built-in rules, a dict with the keys `standard`, `version`,
            `engine` (which is `"built-in"`), `ct_packages` (the controlled terminology packages
            that were used), `n_rules`, `status_counts` (the number of rules with each status),
            `n_issues` (the total number of issues) and `all_passed`.

            For a CDISC CORE report, a dict with the keys `standard`, `version`, `engine_version`
            (the version of CORE), `engine` (which is `"core"`), `n_rules`, `status_counts`,
            `n_issues`, `n_datasets` and `all_passed`.

            For a Validate-based report, a dict with an entry for each dataset. Each entry is a
            dict with the keys `n_steps`, `n_steps_failed`, `n_failed` (the number of failing test
            units) and `all_passed`.

        Examples
        --------
        Let's check a small DM dataset with the built-in rules and summarize the results:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame(
            {
                "STUDYID": ["STUDY01"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
                "RFSTDTC": ["2024-01-15", "15JAN2024", "2024-01-17"],
                "AGE": [34, 51, -38],
                "SEX": ["M", "F", "FEMALE"],
            }
        )

        report = pb.validate_sdtmig({"DM": dm})

        report.summary()
        ```

        The version is given as `"3-4"`, the form that the rules catalog uses. The issue count is
        the sum of the issues of all the rules: each failing record counts for a rule that
        checks values, and a failing rule counts once for a rule that checks the presence of
        variables or datasets.
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
                "engine": "built-in",
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
        """List the issues that the conformance check found.

        An issue is a rule (or, for a Validate-based report, a validation step) that found
        something wrong. The list is a quick way to see what has to be fixed, and it can be
        turned into a DataFrame or saved. Each issue is a dict, whose keys depend on the form of
        the report.

        Parameters
        ----------
        severity
            For a Validate-based report, only keep the issues of this severity: `"warning"`,
            `"error"` or `"critical"`. The severity of a step comes from the `thresholds=` of
            [`validate_conformance()`](`pointblank.SubmissionPackage.validate_conformance`), so
            without thresholds no issue has a severity. This is ignored for the other forms of
            report. The default is `None`, which keeps all the issues.
        status
            For a CDISC CORE report, only keep the issues of rules with this status, such as
            `"ISSUE REPORTED"`. This is ignored for the other forms of report. The default is
            `None`, which keeps all the issues.

        Returns
        -------
        list[dict]
            For a report of the built-in rules, a dict for each rule with issues, with the keys
            `dataset`, `rule_id`, `rule_type`, `message`, `n_issues`, `sensitivity` (`"Error"` or
            `"Warning"`) and `status`.

            For a CDISC CORE report, a dict for each rule and dataset with issues, with the keys
            `dataset`, `rule_id`, `message`, `issues` (the number of issues) and `status`.

            For a Validate-based report, a dict for each step with failing test units, with the
            keys `dataset`, `step`, `assertion`, `column`, `n_failed`, `n` and `severity`.

        Examples
        --------
        Let's check a small DM dataset with the built-in rules and list the issues as a
        DataFrame:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame(
            {
                "STUDYID": ["STUDY01"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
                "RFSTDTC": ["2024-01-15", "15JAN2024", "2024-01-17"],
                "AGE": [34, 51, -38],
                "SEX": ["M", "F", "FEMALE"],
            }
        )

        report = pb.validate_sdtmig({"DM": dm})

        pl.DataFrame(report.issues()).select("rule_id", "dataset", "sensitivity", "n_issues")
        ```

        The missing TS and TA datasets are given as the datasets of SDTM-121 and SDTM-122. Only
        SDTM-122 is of `"Warning"` sensitivity. A Validate-based report gives the severity of
        each step instead, when thresholds are set:

        ```{python}
        ae = pl.DataFrame({"USUBJID": ["STUDY01-001", "STUDY01-009"], "AESEQ": [1, 1]})
        study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae})

        report = study.validate_conformance(engine="validate", thresholds=pb.Thresholds(warning=1))

        report.issues(severity="warning")
        ```

        The issues are the orphan subject in AE, and the length of `"FEMALE"` and the date
        `"15JAN2024"` in DM. Each of them is one failing test unit, which reaches the warning
        threshold.
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
        """Get the failing records that the conformance check found.

        A rule that checks the values of a dataset (such as a codelist check on `SEX`) fails for
        particular records, and each of these is a finding. The `findings()` method gets them all
        as a list of objects. For the built-in rules,
        [`findings_df()`](`pointblank.ConformanceReport.findings_df`) gives the same findings as a
        DataFrame, which is often easier to work with.

        Returns
        -------
        list
            For a report of the built-in rules, a `NativeRowFinding` for each failing record, with
            the attributes `rule_id`, `dataset`, `row` (the 0-based position of the record),
            `usubjid`, `checked_column`, `checked_value`, `context` (the record's identifying
            variables, such as `STUDYID`) and `message`. At most 100 findings are kept for each
            rule.

            For a CDISC CORE report, a `CoreFinding` for each entry of CORE's `Issue_Details`,
            with the attributes `rule_id`, `message`, `dataset`, `executability`, `usubjid`, `row`
            (the 1-based position of the record), `seq`, `variables` and `values`.

            For a Validate-based report, an empty list.

        Examples
        --------
        Let's check a small DM dataset with the built-in rules:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame(
            {
                "STUDYID": ["STUDY01"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
                "RFSTDTC": ["2024-01-15", "15JAN2024", "2024-01-17"],
                "AGE": [34, 51, -38],
                "SEX": ["M", "F", "FEMALE"],
            }
        )

        report = pb.validate_sdtmig({"DM": dm})

        for finding in report.findings():
            print(finding.rule_id, finding.row, finding.usubjid, finding.message)
        ```

        Ten rules fail, but only these three check values. The others fail because a variable or
        a dataset is missing, so they have no failing records.
        """
        if self.is_core:
            return list(self.core.findings)
        if self.is_rules:
            return self.native_result.findings()
        return []

    def rules(self, status: str | None = None):
        """Get the result of each rule.

        Every rule that was run has a result, whether it passed or not. The `rules()` method
        gets these results, optionally only those with a particular status. This is useful for
        listing the failing rules, or for finding out why some rules were not applicable.

        Parameters
        ----------
        status
            Only keep the rules with this status. For the built-in rules, this can be `"pass"`,
            `"fail"`, `"error"`, `"not_applicable"` or `"not_supported"`. For CDISC CORE, it is
            one of CORE's statuses, such as `"SUCCESS"`, `"ISSUE REPORTED"`, `"SKIPPED"` or
            `"EXECUTION ERROR"`. The default is `None`, which keeps all the rules.

        Returns
        -------
        list
            For a report of the built-in rules, a `NativeRuleResult` for each rule, with the
            attributes `rule_id`, `rule_type`, `dataset`, `status`, `sensitivity`, `description`,
            `message`, `n_issues` and `row_findings` (its findings, as returned by
            [`findings()`](`pointblank.ConformanceReport.findings`)).

            For a CDISC CORE report, a `CoreRuleResult` for each rule, with the attributes
            `rule_id`, `status`, `message`, `version`, `cdisc_rule_id` and `fda_rule_id`.

            For a Validate-based report, an empty list.

        Examples
        --------
        Let's check a small DM dataset with the built-in rules and look at the rules that are
        not applicable:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame(
            {
                "STUDYID": ["STUDY01"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
                "SEX": ["M", "F", "FEMALE"],
            }
        )

        report = pb.validate_sdtmig({"DM": dm})

        not_applicable = report.rules(status="not_applicable")

        print(len(not_applicable))
        print({rule.message for rule in not_applicable})
        ```

        The 12 rules all check the datasets against a Define-XML, which wasn't given. With the
        `define_xml=` argument of [`validate_sdtmig()`](`pointblank.validate_sdtmig`), they would
        be run too.
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
        """The number of datasets that were checked.

        For a report of the built-in rules, this is the number of datasets that were given, and
        for a Validate-based report, the number of validations. For a CDISC CORE report, it is
        the number of datasets in CORE's `Dataset_Details`. Datasets that a rule requires but
        that are missing (such as TS) aren't counted. This is a property, so it's used without
        parentheses.

        Examples
        --------
        Let's check a package of two datasets with the built-in rules:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame({"STUDYID": ["STUDY01"], "USUBJID": ["STUDY01-001"]})
        ae = pl.DataFrame({"STUDYID": ["STUDY01"], "USUBJID": ["STUDY01-001"], "AESEQ": [1]})

        report = pb.validate_sdtmig({"DM": dm, "AE": ae})

        report.n_datasets
        ```

        The two datasets were checked, though some rules are about the TS and TA datasets, which
        the package doesn't have.
        """
        if self.is_core:
            return len(self.core.datasets)
        if self.is_rules and self.package is not None:
            return len(self.package.datasets)
        return len(self.validations)

    def to_json(self, path: str | Path) -> Path:
        """Save the report as a JSON file.

        A JSON file keeps the results of a conformance check for later, such as for comparing
        the results of two runs or for passing them to another tool. For a report of the built-in
        rules or a Validate-based report, the file has two entries: `summary`, as from
        [`summary()`](`pointblank.ConformanceReport.summary`), and `issues`, as from
        [`issues()`](`pointblank.ConformanceReport.issues`). For a CDISC CORE report, the file has
        the five sections of CORE's own JSON report, so it can be read by tools that read CORE
        reports, and back into a report with
        [`from_core_report()`](`pointblank.ConformanceReport.from_core_report`).

        Parameters
        ----------
        path
            The path of the file to write, including its name. Directories in the path that don't
            exist are created.

        Returns
        -------
        Path
            The path of the file that was written.

        Examples
        --------
        Let's check a small DM dataset with the built-in rules and save the report in a temporary
        directory:

        ```{python}
        import json
        import tempfile
        from pathlib import Path

        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame(
            {
                "STUDYID": ["STUDY01"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
                "SEX": ["M", "F", "FEMALE"],
            }
        )

        report = pb.validate_sdtmig({"DM": dm})

        path = report.to_json(Path(tempfile.mkdtemp()) / "conformance" / "dm.json")

        saved = json.loads(path.read_text())

        print(saved["summary"]["status_counts"])
        print(saved["issues"][0])
        ```

        The `conformance` directory didn't exist, so it was created. Each issue is saved with all
        its keys, as `issues()` gives it.
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
        """Save the report as an Excel workbook.

        A workbook is a convenient way to share the results of a conformance check with people
        who review them in a spreadsheet. The sheets depend on the form of the report:

        - For the built-in rules, `Rules_Report` has a row for each rule, with its type, dataset,
          status, sensitivity, number of issues and message. `Issues` has the rules with issues
          (as from [`issues()`](`pointblank.ConformanceReport.issues`)), and `Summary` has the
          entries of [`summary()`](`pointblank.ConformanceReport.summary`).
        - For CDISC CORE, the sheets are `Issue_Summary`, `Issue_Details`, `Rules_Report` and
          `Conformance_Details`, like those of CORE's own Excel report.
        - For a Validate-based report, `Issues` has the failing steps and `Summary` has a row for
          each dataset.

        A sheet that would be empty, such as `Issues` when there are no issues, is left out. The
        `openpyxl` and `pandas` packages are needed for writing the workbook.

        Parameters
        ----------
        path
            The path of the file to write, including its name (which usually ends in `.xlsx`).
            Directories in the path that don't exist are created.

        Returns
        -------
        Path
            The path of the file that was written.

        Raises
        ------
        ImportError
            If `openpyxl` or `pandas` isn't installed.

        Examples
        --------
        Let's check a small DM dataset with the built-in rules and save the report in a temporary
        directory:

        ```{python}
        import tempfile
        from pathlib import Path

        import pandas as pd
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame(
            {
                "STUDYID": ["STUDY01"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
                "SEX": ["M", "F", "FEMALE"],
            }
        )

        report = pb.validate_sdtmig({"DM": dm})

        path = report.to_excel(Path(tempfile.mkdtemp()) / "dm_conformance.xlsx")

        pd.ExcelFile(path).sheet_names
        ```

        Here's the start of the `Rules_Report` sheet, whose rows are in the order of the rules
        catalog:

        ```{python}
        pd.read_excel(path, sheet_name="Rules_Report").head(3)
        ```

        The workbook has a row for each of the 362 rules, so the passing rules are recorded as
        well as the failing ones.
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
                pd.DataFrame([{"Key": k, "Value": str(v)} for k, v in s.items()]).to_excel(
                    writer, sheet_name="Summary", index=False
                )
            else:
                issues = self.issues()
                if issues:
                    pd.DataFrame(issues).to_excel(writer, sheet_name="Issues", index=False)
                pd.DataFrame(
                    [{"Dataset": name, **s} for name, s in self.summary().items()]
                ).to_excel(writer, sheet_name="Summary", index=False)

        return dest

    def findings_df(self):
        """Get the failing records found by the built-in rules, as a DataFrame.

        Each row of the DataFrame is a record that failed a rule. As a DataFrame, the findings
        are easy to filter by rule, count by subject, save as a CSV file, or join back to the
        datasets to look at the records in full. The rules that check values (record checks,
        dataset contents checks and checks against the Define-XML's codelists) give findings.
        Rules that check whether variables or datasets are present have a number of issues but no
        findings. At most 100 findings are kept for each rule, though the rule's number of issues
        (as in [`issues()`](`pointblank.ConformanceReport.issues`)) counts all of its failing
        records.

        Returns
        -------
        polars.DataFrame
            A DataFrame with a row for each finding and these columns:

            - `rule_id`: the identifier of the rule, such as `"SDTM-007"`
            - `dataset`: the dataset of the record, such as `"DM"`
            - `row_index`: the 0-based position of the record in its dataset
            - `usubjid`: the record's `USUBJID`, or `""` if it has none
            - `checked_column`: the variable that was checked, such as `"SEX"`
            - `checked_value`: the value that failed, as a string
            - `description`: the description of the rule

            If there are no findings, the DataFrame is empty but has the same columns.

        Raises
        ------
        TypeError
            If the report isn't of the built-in rules. For a CDISC CORE report, use
            [`findings()`](`pointblank.ConformanceReport.findings`) instead.

        Examples
        --------
        Let's check a small DM dataset with the built-in rules and get its failing records:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame(
            {
                "STUDYID": ["STUDY01"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
                "RFSTDTC": ["2024-01-15", "15JAN2024", "2024-01-17"],
                "AGE": [34, 51, -38],
                "SEX": ["M", "F", "FEMALE"],
            }
        )

        report = pb.validate_sdtmig({"DM": dm})

        findings = report.findings_df()

        findings.drop("description")
        ```

        Since `row_index` is 0-based, it can be used to get the failing records from the
        dataset:

        ```{python}
        dm[findings["row_index"].unique().sort().to_list()]
        ```

        The second and third records have problems, and the first has none.
        """
        import polars as pl

        if not self.is_rules:
            raise TypeError(
                "findings_df() is only available for built-in engine results. "
                "For CORE-backed reports, use findings() which returns CoreFinding objects."
            )

        rows: list[dict] = []
        for rule_result in self.native_result.rule_results:
            for f in rule_result.row_findings:
                rows.append(
                    {
                        "rule_id": f.rule_id,
                        "dataset": f.dataset,
                        "row_index": f.row if f.row is not None else -1,
                        "usubjid": f.usubjid or "",
                        "checked_column": f.checked_column or "",
                        "checked_value": f.checked_value or "",
                        "description": rule_result.description,
                    }
                )

        _SCHEMA = {
            "rule_id": pl.String,
            "dataset": pl.String,
            "row_index": pl.Int64,
            "usubjid": pl.String,
            "checked_column": pl.String,
            "checked_value": pl.String,
            "description": pl.String,
        }
        if not rows:
            return pl.DataFrame(schema=_SCHEMA)
        return pl.DataFrame(rows, schema=_SCHEMA)

    def get_findings_table(self) -> "GT":
        """Make a table of the failing records found by the built-in rules.

        The table of [`get_tabular_report()`](`pointblank.ConformanceReport.get_tabular_report`)
        has a row for each rule, with its number of issues. The findings table goes into more
        detail, with a row for each failing record, so that a reviewer can see which subjects and
        values have to be fixed. It has the same findings as
        [`findings_df()`](`pointblank.ConformanceReport.findings_df`), with at most 100 for each
        rule.

        The columns of the table are in two groups. Under *SDTM Rule Definition* are the rule's
        identifier, its domain and its description. Under *Finding* are the record's `USUBJID`,
        the variable that was checked, the record's row number (counting from 1) and the value
        that failed. The subtitle has the standard and its version and the numbers of rules that
        failed, passed and were not applicable.

        Returns
        -------
        GT
            The table, as a [Great Tables](https://posit-dev.github.io/great-tables/) `GT` object.

        Raises
        ------
        TypeError
            If the report isn't of the built-in rules.
        ValueError
            If there are no failing records. This happens when no rule that checks values failed,
            even if other rules (such as those for missing variables) did.

        Examples
        --------
        Let's check a small DM dataset with the built-in rules and make a table of its failing
        records:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame(
            {
                "STUDYID": ["STUDY01"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
                "RFSTDTC": ["2024-01-15", "15JAN2024", "2024-01-17"],
                "AGE": [34, 51, -38],
                "SEX": ["M", "F", "FEMALE"],
            }
        )

        report = pb.validate_sdtmig({"DM": dm})

        report.get_findings_table()
        ```

        The third subject's record has two problems, a negative `AGE` and a `SEX` of `"FEMALE"`,
        so it's in the table twice. The second subject's `RFSTDTC` isn't in the ISO 8601 format.
        """
        import polars as pl
        from great_tables import GT, from_column, google_font, html, loc, style

        if not self.is_rules:
            raise TypeError("get_findings_table() is only available for built-in engine results.")

        df = self.findings_df()
        if df.is_empty():
            raise ValueError("No row-level findings to display — all rules passed.")

        # Add a red status bar column (all findings are failures)
        df = df.with_columns(pl.lit("#FF3300").alias("status_color"))

        # 1-indexed row number for easy record lookup
        df = df.with_columns((pl.col("row_index") + 1).alias("row_1indexed"))

        # Reorder columns: color bar first, then rule info, then finding details
        df = df.select(
            [
                "status_color",
                "rule_id",
                "dataset",
                "description",
                "usubjid",
                "checked_column",
                "row_1indexed",
                "checked_value",
            ]
        )

        # Build header matching the tabular conformance report style
        nr = self.native_result
        counts = nr.status_counts()
        _LABEL = {
            "pass": "passed",
            "fail": "failed",
            "error": "error",
            "not_applicable": "n/a",
            "not_supported": "unsupported",
        }
        counts_parts = []
        n_failed = counts.get("fail", 0)
        n_passed = counts.get("pass", 0)
        if n_failed:
            counts_parts.append(f"{n_failed}&thinsp;failed&ensp;({n_passed}&thinsp;passed)")
        for st in ("error", "not_applicable", "not_supported"):
            n = counts.get(st, 0)
            if n:
                counts_parts.append(f"{n}&thinsp;{_LABEL[st]}")
        counts_str = "&ensp;·&ensp;".join(counts_parts)

        title_text = "Findings Report for CDISC Conformance"
        subtitle_text = f"{nr.standard.upper()}&thinsp;{nr.version}&ensp;·&ensp;{counts_str}"

        gt = (
            GT(df)
            .tab_header(title=html(title_text), subtitle=html(subtitle_text))
            .tab_spanner(
                label="Finding",
                columns=["usubjid", "checked_column", "row_1indexed", "checked_value"],
            )
            .tab_spanner(
                label="SDTM Rule Definition", columns=["rule_id", "dataset", "description"]
            )
            .cols_move_to_start(columns="status_color")
            .cols_label(
                status_color="",
                rule_id="Rule",
                dataset="Domain",
                description="Description",
                usubjid="USUBJID",
                checked_column="Column",
                row_1indexed="Row",
                checked_value="Value",
            )
            # Should be 904px wide, just like the validation report table.
            .cols_width(
                status_color="4px",
                rule_id="70px",
                dataset="70px",
                description="360px",
                usubjid="130px",
                checked_column="100px",
                row_1indexed="50px",
                checked_value="120px",
            )
            .tab_style(
                style=style.fill(color=from_column("status_color")),
                locations=loc.body(columns="status_color"),
            )
            .tab_style(
                style=style.text(color=from_column("status_color"), whitespace="nowrap"),
                locations=loc.body(columns="status_color"),
            )
            .tab_style(
                style=style.text(font=google_font("IBM Plex Mono"), size="11px"),
                locations=loc.body(),
            )
            .tab_style(
                style=style.css("padding-top: 2px; padding-bottom: 2px;"),
                locations=loc.body(),
            )
            .tab_style(
                style=style.css("overflow: hidden; text-overflow: ellipsis; white-space: nowrap;"),
                locations=loc.body(
                    columns=["usubjid", "checked_column", "row_1indexed", "checked_value"]
                ),
            )
            .opt_table_font(font=google_font("IBM Plex Sans"))
            .opt_align_table_header(align="left")
            .tab_options(table_font_size="90%")
        )

        return gt

    def get_tabular_report(self) -> "GT":
        """Make a table of the results of the built-in rules.

        The table has a row for each rule in the catalog, so it shows at a glance which rules
        failed and how many issues each one found. It is how a report of the built-in rules is
        shown in a notebook. For the failing records themselves, use
        [`get_findings_table()`](`pointblank.ConformanceReport.get_findings_table`) or
        [`findings_df()`](`pointblank.ConformanceReport.findings_df`).

        Each row has a colored bar for its status (green for a pass, red for a failure, amber for
        an error and gray for a rule that is not applicable or not supported), the rule's
        identifier, its domain, its type (such as `Record` for a check of values or `Variable` for
        a check that variables are present), its number of issues and its description. The
        failing rules come first, then the rules with errors, the passing rules and the rules
        that are not applicable or not supported. The title has a `PASS` or `FAIL` badge, and the
        subtitle has the standard and its version, the agency (if one was given) and the number
        of rules with each status.

        Returns
        -------
        GT
            The table, as a [Great Tables](https://posit-dev.github.io/great-tables/) `GT` object.

        Raises
        ------
        TypeError
            If the report isn't of the built-in rules.

        Examples
        --------
        Let's check a small DM dataset with the built-in rules and make the table:

        ```{python}
        import pointblank as pb
        import polars as pl

        dm = pl.DataFrame(
            {
                "STUDYID": ["STUDY01"] * 3,
                "DOMAIN": ["DM"] * 3,
                "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
                "RFSTDTC": ["2024-01-15", "15JAN2024", "2024-01-17"],
                "AGE": [34, 51, -38],
                "SEX": ["M", "F", "FEMALE"],
            }
        )

        report = pb.validate_sdtmig({"DM": dm})

        report.get_tabular_report()
        ```

        The ten failing rules are at the top, followed by the passing ones. The table has all 362
        rules, so it's long. Displaying `report` itself in a notebook gives the same table.
        """
        from importlib.metadata import version as _pkg_version

        import polars as pl
        from great_tables import GT, from_column, google_font, html, loc, style

        if not self.is_rules:
            raise TypeError(
                "get_tabular_report() is only available for built-in engine results. "
                "Use validate_conformance(engine='native') to obtain one."
            )

        nr = self.native_result

        _STATUS_COLORS = {
            "pass": "#4CA64C",
            "fail": "#FF3300",
            "error": "#EBBC14",
            "not_applicable": "#AAAAAA",
            "not_supported": "#AAAAAA",
        }
        _TYPE_LABELS = {
            "RECORD_CHECK": "Record",
            "VARIABLE_METADATA_CHECK": "Variable",
            "DATASET_CONTENTS_CHECK": "Dataset",
            "DOMAIN_PRESENCE_CHECK": "Domain",
            "DATASET_METADATA_CHECK": "Metadata",
            "DEFINE_ITEM_METADATA_CHECK": "Define",
            "DEFINE_CODELIST_CHECK": "Codelist",
        }
        _STATUS_PRIORITY = {
            "fail": 0,
            "error": 1,
            "pass": 2,
            "not_applicable": 3,
            "not_supported": 4,
        }

        rows = sorted(
            nr.rule_results,
            key=lambda r: (_STATUS_PRIORITY.get(r.status, 5), -r.n_issues),
        )

        def _fmt_dataset(ds: str) -> str:
            parts = [p.strip() for p in ds.split(",")]
            supp = [p for p in parts if p.startswith("SUPP")]
            other = [p for p in parts if not p.startswith("SUPP")]
            if len(supp) > 1:
                return ", ".join(other + ["SUPP--"]) if other else "SUPP--"
            return ds

        data = {
            "status_color": [_STATUS_COLORS.get(r.status, "#AAAAAA") for r in rows],
            "rule_id": [r.rule_id for r in rows],
            "dataset": [_fmt_dataset(r.dataset) for r in rows],
            "type": [_TYPE_LABELS.get(r.rule_type, r.rule_type) for r in rows],
            "n_issues": [r.n_issues for r in rows],
            "description": [r.description for r in rows],
        }
        df = pl.DataFrame(data)

        # indices of rows with at least one issue (for red-text styling)
        issue_indices = [i for i, r in enumerate(rows) if r.n_issues > 0]

        counts = nr.status_counts()
        counts_parts = []
        _LABEL = {
            "pass": "passed",
            "fail": "failed",
            "error": "error",
            "not_applicable": "n/a",
            "not_supported": "unsupported",
        }
        for st in ("pass", "fail", "error", "not_applicable", "not_supported"):
            n = counts.get(st, 0)
            if n:
                counts_parts.append(f"{n}&thinsp;{_LABEL[st]}")
        counts_str = "&ensp;·&ensp;".join(counts_parts)

        agency_part = f"&ensp;·&ensp;{self.agency}" if self.agency else ""
        subtitle_text = (
            f"{nr.standard.upper()}&thinsp;{nr.version}{agency_part}&ensp;·&ensp;{counts_str}"
        )

        ct_note = ""
        if nr.ct_packages:
            ct_note = "CT: " + " | ".join(nr.ct_packages)

        overall_passed = nr.all_passed
        status_label = "PASS" if overall_passed else "FAIL"
        status_color = "#4CA64C" if overall_passed else "#FF3300"
        status_html = (
            f'<span style="display:inline-block;padding:2px 9px;border-radius:10px;'
            f"background:{'#e8f5e9' if overall_passed else '#ffebee'};"
            f"color:{status_color};font-size:0.78em;font-weight:700;"
            f'letter-spacing:0.04em;">{status_label}</span>'
        )
        title_text = f"CDISC Conformance&ensp;{status_html}"

        gt_tbl = (
            GT(df, id="pb_conformance_tbl")
            .tab_header(
                title=html(title_text),
                subtitle=html(subtitle_text),
            )
            .opt_table_font(font=google_font(name="IBM Plex Sans"))
            .opt_align_table_header(align="left")
            # ── status color bar ──────────────────────────────────────────
            .tab_style(
                style=style.fill(color=from_column(column="status_color")),
                locations=loc.body(columns="status_color"),
            )
            .tab_style(
                style=style.text(color="transparent", size="0px"),
                locations=loc.body(columns="status_color"),
            )
            # ── monospace columns ─────────────────────────────────────────
            .tab_style(
                style=style.text(font=google_font(name="IBM Plex Mono"), size="11px"),
                locations=loc.body(
                    columns=["rule_id", "dataset", "type", "n_issues", "description"]
                ),
            )
            # ── issues column: red when non-zero ──────────────────────────
            .tab_style(
                style=style.text(color="#c62828", weight="bold"),
                locations=loc.body(columns="n_issues", rows=issue_indices),
            )
            # ── row height ────────────────────────────────────────────────
            .tab_style(
                style=style.css("padding-top: 2px; padding-bottom: 2px;"),
                locations=loc.body(),
            )
            # ── column labels ─────────────────────────────────────────────
            .cols_label(
                cases={
                    "status_color": "",
                    "rule_id": "Rule",
                    "dataset": "Domain",
                    "type": "Type",
                    "n_issues": "Issues",
                    "description": "Description",
                }
            )
            # ── column widths ─────────────────────────────────────────────
            # Should be 904px wide, just like the validation report table.
            .cols_width(
                cases={
                    "status_color": "4px",
                    "rule_id": "90px",
                    "dataset": "80px",
                    "type": "80px",
                    "n_issues": "50px",
                    "description": "600px",
                }
            )
            # ── alignment ─────────────────────────────────────────────────
            .cols_align(align="center", columns=["n_issues"])
            .tab_options(table_font_size="90%")
        )

        if ct_note:
            gt_tbl = gt_tbl.tab_source_note(
                source_note=html(f'<span style="font-size:0.82em;color:#9e9e9e;">{ct_note}</span>')
            )

        try:
            if _pkg_version("great_tables") >= "0.17.0":
                gt_tbl = gt_tbl.tab_options(quarto_disable_processing=True)
        except Exception:
            pass

        return gt_tbl

    def _repr_html_(self) -> str:
        agency = f" — agency: {self.agency}" if self.agency else ""

        if self.is_rules:
            return self.get_tabular_report()._repr_html_()

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
                parts.append(
                    "<table><tr><th>Dataset</th><th>Rule</th><th>Issues</th><th>Message</th></tr>"
                )
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
            lines = ["ConformanceReport (Built-in Rules)"]
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
    """Check a CDISC submission with the CDISC CORE engine.

    CDISC CORE is CDISC's open-source engine for its conformance rules, and it has the
    published rules of CDISC and of the FDA and PMDA. The `validate_cdisc_submission()` function
    runs CORE on a submission in a single call: it makes a
    [`SubmissionPackage`](`pointblank.SubmissionPackage`) from the `source` (unless it is one
    already), runs CORE on it, and reads CORE's results into a
    [`ConformanceReport`](`pointblank.ConformanceReport`). This is the same as calling the
    package's [`validate_conformance()`](`pointblank.SubmissionPackage.validate_conformance`)
    method with `engine="core"`.

    CORE is a separate command-line tool (`cdisc-rules-engine`) that has to be installed. If it
    isn't on the `PATH`, give the command with `core=` or in the `POINTBLANK_CDISC_CORE`
    environment variable. For checks that need no other software, use
    [`validate_sdtmig()`](`pointblank.validate_sdtmig`), which runs Pointblank's bundled SDTMIG
    rules.

    Parameters
    ----------
    source
        The submission to check. This can be the path of a folder of datasets (`.xpt` or
        Dataset-JSON files, which are read with
        [`SubmissionPackage.from_folder()`](`pointblank.SubmissionPackage.from_folder`)), a dict of
        DataFrames keyed by dataset name, or a `SubmissionPackage`.
    standard
        The CDISC standard, such as `"sdtmig"`. The default is `"sdtmig"`, or the package's
        `standard` if `source` is a `SubmissionPackage`.
    version
        The version of the standard, such as `"3.4"`. The default is `"3.4"`, or the package's
        `standard_version` if `source` is a `SubmissionPackage`.
    define
        The path of the study's Define-XML, which CORE uses in its checks. For a folder, a
        `define*.xml` file in it is found without this. It is ignored if `source` is a
        `SubmissionPackage`, whose own `define` is used. The default is `None`.
    controlled_terminology
        The controlled terminology package or packages for CORE's `-ct` option, such as
        `"sdtmct-2024-03-29"`. The default is `None`.
    agency
        The regulatory agency, such as `"FDA"` or `"PMDA"`, which is shown in the report. It
        doesn't change the rules that CORE runs. The default is `None`.
    ct_version
        The controlled terminology version to record in the package. It is only shown in the
        package's summary and isn't given to CORE. It is ignored if `source` is a
        `SubmissionPackage`. The default is `None`.
    study_id
        A study identifier to record in the package. It is ignored if `source` is a
        `SubmissionPackage`. The default is `None`.
    core
        How to run CORE. This can be the path or name of the CORE executable, or a command as a
        list (such as `["python", "core.py"]`). With the default of `None`, the command is taken
        from the `POINTBLANK_CDISC_CORE` environment variable, or else CORE is looked for on the
        `PATH`.
    core_cwd
        The directory to run CORE from. This is needed when CORE is run from a checkout of its
        repository, since it looks for its `resources/` folder in the current directory. The
        default is `None`.
    cache
        The path to CORE's rules cache (its `-ca` option). The default is `None`.
    workdir
        A directory for the XPT files that the datasets are written to and for CORE's report.
        With the default of `None`, a temporary directory is used and then deleted.

    Returns
    -------
    ConformanceReport
        A report of CORE's results, whose [`is_core`](`pointblank.ConformanceReport.is_core`) is
        `True`.

    Raises
    ------
    NotADirectoryError
        If `source` is a path that isn't a folder.
    TypeError
        If `source` isn't a path, a dict or a `SubmissionPackage`.

    Examples
    --------
    CORE has to be installed for this function, so these examples aren't run here. Let's check
    a folder of SDTM datasets against SDTMIG 3.4, for a submission to the FDA:

    ```python
    import pointblank as pb

    report = pb.validate_cdisc_submission(
        "study_xyz/sdtm/",
        standard="sdtmig",
        version="3.4",
        controlled_terminology="sdtmct-2024-03-29",
        agency="FDA",
    )

    print(report)
    ```

    Printing the report gives the standard, the version of CORE, the number of rules with each
    of CORE's statuses and the total number of issues. The issues and the failing records can
    then be looked at, and the report saved for review:

    ```python
    for issue in report.issues(status="ISSUE REPORTED"):
        print(issue["dataset"], issue["rule_id"], issue["issues"], issue["message"])

    report.to_excel("study_xyz/conformance/core_report.xlsx")
    ```

    If CORE was installed from a checkout of its repository, it has to be run from that
    directory, as in `core=["python", "core.py"], core_cwd="path/to/cdisc-rules-engine"`. A
    report that CORE wrote on its own can be read with
    [`ConformanceReport.from_core_report()`](`pointblank.ConformanceReport.from_core_report`),
    which has an example that runs without CORE.

    See Also
    --------
    validate_sdtmig : Check SDTM datasets with Pointblank's bundled rules.
    SubmissionPackage : A set of datasets to check together.
    ConformanceReport : The report that is returned.
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


def validate_sdtmig(
    datasets: dict,
    version: str = "3-4",
    ct_packages: list[str] | None = None,
    define_xml: Any = None,
    study_id: str | None = None,
) -> ConformanceReport:
    """Check SDTM datasets with Pointblank's bundled SDTMIG conformance rules.

    Before SDTM datasets are submitted, they are checked for conformance to the SDTM
    Implementation Guide (SDTMIG). The `validate_sdtmig()` function does this with a catalog of
    362 rules for SDTMIG 3.4 that comes with Pointblank, so it needs no other software and no
    network access. It returns a [`ConformanceReport`](`pointblank.ConformanceReport`) with the
    result of each rule. For the published CDISC and agency rules, use
    [`validate_cdisc_submission()`](`pointblank.validate_cdisc_submission`), which runs the
    CDISC CORE engine.

    The rules are of these types:

    - Record checks (221 rules) check the values of each record, such as that `SEX` uses the
      terms of its codelist, that dates are in the ISO 8601 format, that `AGE` isn't negative
      and that required variables aren't null. These rules give the failing records as
      findings.
    - Variable metadata checks (116 rules) check that the required variables of a domain are
      present, that variables have the right type and that they are in the right order.
    - Dataset contents checks (9 rules) check across the records of a dataset, such as that
      `STUDYID` has the same value in every record and that `AESEQ` is unique for each subject.
    - Dataset metadata checks (3 rules) check that every dataset has `STUDYID`, `DOMAIN` and
      (for subject-level data) `USUBJID`.
    - Domain presence checks (3 rules) check that the DM, TS and TA datasets are present.
    - Define-XML checks (12 rules) check the datasets against their Define-XML, such as that
      every variable is declared in it and that values are from the codelists that it declares.
      These rules need `define_xml=`, and are not applicable without it.

    Four of the rules are of `"Warning"` sensitivity: the presence of TA and of `USUBJID`, and
    the order of the identifier variables and of the sequence variable. All the others are of
    `"Error"` sensitivity. A rule for a domain that isn't one of the `datasets` (such as an AE
    rule when there's no AE dataset) has nothing to check, so it passes. The codelist checks
    ignore case, so `"m"` is accepted for `SEX`. Empty strings (which is how XPT files store
    missing character values) are treated as nulls. SUPP-- and RELREC datasets aren't checked
    by the rules that apply to all domains, since they have a different structure (with
    `RDOMAIN` in place of `DOMAIN`).

    The rules check each dataset on its own, so they don't check that the subjects of one
    dataset are in DM. For that, use the `"validate"` engine of the `SubmissionPackage` method
    [`validate_conformance()`](`pointblank.SubmissionPackage.validate_conformance`).

    Parameters
    ----------
    datasets
        The SDTM datasets, as a dict of DataFrames keyed by domain name, such as
        `{"DM": dm, "AE": ae}`. The names are case-insensitive. The DataFrames can be Polars,
        Pandas or any other kind that Narwhals supports.
    version
        The version of SDTMIG, as `"3-4"` or `"3.4"`. Only version 3.4 has a bundled catalog. The
        default is `"3-4"`.
    ct_packages
        The bundled controlled terminology packages for the codelist checks, as a list of names
        such as `["sdtm-ct-2024-09-27"]`. The default is the latest bundled package, which is
        `"sdtm-ct-2024-09-27"`.
    define_xml
        The study's Define-XML, as a path to a `define.xml` file or as a
        [`MetadataPackage`](`pointblank.MetadataPackage`) from
        [`import_metadata()`](`pointblank.import_metadata`). The default is `None`, which makes
        the 12 Define-XML rules not applicable.
    study_id
        A study identifier, which is kept in the report's `package`. It isn't used by the rules
        and isn't shown in the report. The default is `None`.

    Returns
    -------
    ConformanceReport
        The report, whose [`is_rules`](`pointblank.ConformanceReport.is_rules`) is `True`. In a
        notebook it is shown as the table of
        [`get_tabular_report()`](`pointblank.ConformanceReport.get_tabular_report`).

    Raises
    ------
    ValueError
        If there is no bundled catalog for the `version`. The error message lists the versions
        that have one.
    FileNotFoundError
        If one of the `ct_packages` isn't bundled with Pointblank.

    Examples
    --------
    Here is a DM dataset with all the variables that SDTMIG requires for DM, but with three
    problems: the second subject's `RFSTDTC` isn't an ISO 8601 date, and the third subject has a
    negative `AGE` and a `SEX` of `"FEMALE"` (the SEX codelist has `"F"`):

    ```{python}
    import pointblank as pb
    import polars as pl

    dm = pl.DataFrame(
        {
            "STUDYID": ["STUDY01"] * 3,
            "DOMAIN": ["DM"] * 3,
            "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
            "SUBJID": ["001", "002", "003"],
            "RFSTDTC": ["2024-01-15", "15JAN2024", "2024-01-17"],
            "RFENDTC": ["2024-06-15", "2024-06-16", "2024-06-17"],
            "SITEID": ["S01", "S01", "S02"],
            "AGE": [34, 51, -38],
            "AGEU": ["YEARS"] * 3,
            "SEX": ["M", "F", "FEMALE"],
            "RACE": ["WHITE", "ASIAN", "WHITE"],
            "ARMCD": ["DRUGA", "PBO", "DRUGA"],
            "ARM": ["Drug A", "Placebo", "Drug A"],
            "ACTARMCD": ["DRUGA", "PBO", "DRUGA"],
            "ACTARM": ["Drug A", "Placebo", "Drug A"],
            "COUNTRY": ["USA", "USA", "CAN"],
        }
    )

    report = pb.validate_sdtmig({"DM": dm})

    print(report)
    ```

    Five rules fail, and the 12 Define-XML rules are not applicable. Let's see which rules
    failed:

    ```{python}
    for rule in report.rules(status="fail"):
        print(rule.rule_id, rule.sensitivity, rule.message)
    ```

    The three problems are each found by a rule. The other two failures are for the TS and TA
    datasets, which every SDTM submission needs. The failing records can be had as a DataFrame
    with [`findings_df()`](`pointblank.ConformanceReport.findings_df`) or as a table with
    [`get_findings_table()`](`pointblank.ConformanceReport.get_findings_table`).

    With a Define-XML, the Define-XML rules are run as well. Pointblank comes with a DM dataset
    as an XPT file and a Define-XML for it:

    ```{python}
    import pyreadstat

    dm, _ = pyreadstat.read_xport(pb.load_metadata_example("dm.xpt"))

    report = pb.validate_sdtmig({"DM": dm}, define_xml=pb.load_metadata_example("define.xml"))

    print(report)
    print([rule.rule_id for rule in report.rules(status="fail")])
    ```

    No rules are not applicable now. SDTM-057 is one of the Define-XML rules, and it finds a
    `RACE` of `"BLACK"`, which isn't in the codelist that the Define-XML declares for `RACE`.
    SDTM-008 finds the same value with the CDISC RACE codelist, and the other failures are for
    missing variables (such as `COUNTRY`) and the missing TS and TA datasets. The dataset was
    read as a Pandas DataFrame, which works as well as a Polars one.

    See Also
    --------
    ConformanceReport : The report that is returned.
    validate_cdisc_submission : Check a submission with the CDISC CORE engine.
    SubmissionPackage : A set of datasets to check together.
    validate_sdtm : Build a validation plan for one dataset from an SDTM template.
    """

    # Normalise version separator (accept "3.4" or "3-4")
    _ver = version.replace(".", "-")
    pkg = SubmissionPackage(
        datasets=datasets,
        standard="sdtmig",
        standard_version=_ver,
        study_id=study_id,
    )
    report = pkg._run_rules_conformance(
        agency=None,
        standard="sdtmig",
        version=_ver,
        ct_packages=ct_packages,
        define_xml=define_xml,
    )
    if report is None:
        from pointblank.metadata._conformance.rule_loader import RuleLoader

        available = [v for s, v in RuleLoader.available() if s == "sdtmig"]
        raise ValueError(
            f"No bundled SDTMIG rule catalog for version {_ver!r}. "
            f"Available versions: {', '.join(available) or '(none)'}."
        )
    return report
