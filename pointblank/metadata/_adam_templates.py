from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

__all__ = [
    "ADaMDatasetTemplate",
    "ADaMVariableSpec",
    "get_adam_dataset",
    "list_adam_datasets",
    "validate_adam_structure",
]


@dataclass
class ADaMVariableSpec:
    """Specification for a single variable in an ADaM dataset template.

    Parameters
    ----------
    name
        Variable name (e.g., `"USUBJID"`, `"AVAL"`, `"PARAMCD"`).
    label
        Variable label (e.g., `"Unique Subject Identifier"`).
    dtype
        Expected data type (`"Char"` or `"Num"`).
    core
        ADaM core designation: `"Req"` (required), `"Cond"` (conditionally required), or `"Perm"`
        (permissible).
    required
        Whether the variable is unconditionally required.
    max_length
        Maximum character length for Char variables.
    controlled_term
        Name of the associated controlled terminology codelist.
    source
        Traceability: expected source (e.g., `"SDTM.DM"`, `"Derived"`).
    condition
        For conditional variables, describes when they are required.
    is_population_flag
        Whether this is a population flag variable (e.g., `"SAFFL"`, `"ITTFL"`).
    """

    name: str
    label: str
    dtype: str  # "Char" or "Num"
    core: str = "Perm"  # "Req", "Cond", "Perm"
    required: bool = False
    max_length: int | None = None
    controlled_term: str | None = None
    source: str | None = None
    condition: str | None = None
    is_population_flag: bool = False


@dataclass
class ADaMDatasetTemplate:
    """Structural template for an ADaM dataset.

    Parameters
    ----------
    name
        Dataset name (e.g., `"ADSL"`, `"ADVS"`, `"ADAE"`, `"ADTTE"`).
    label
        Dataset label (e.g., `"Subject Level Analysis Dataset"`).
    description
        Brief description of the dataset's purpose.
    dataset_class
        ADaM dataset class: `"ADSL"`, `"BDS"`, `"ADAE"`, or `"ADTTE"`.
    variables
        Ordered list of variable specifications.
    natural_keys
        List of variable names that form the natural key.
    """

    name: str
    label: str
    description: str
    dataset_class: str
    variables: list[ADaMVariableSpec] = dataclass_field(default_factory=list)
    natural_keys: list[str] = dataclass_field(default_factory=list)

    @property
    def required_variables(self) -> list[str]:
        """Get names of all required variables."""
        return [v.name for v in self.variables if v.required]

    @property
    def conditional_variables(self) -> list[str]:
        """Get names of all conditionally required variables."""
        return [v.name for v in self.variables if v.core == "Cond"]

    @property
    def population_flags(self) -> list[str]:
        """Get names of all population flag variables."""
        return [v.name for v in self.variables if v.is_population_flag]

    def get_variable(self, name: str) -> ADaMVariableSpec | None:
        """Get a variable spec by name."""
        for v in self.variables:
            if v.name == name:
                return v
        return None


# ─────────────────────────────────────────────────────────────────────────────
# ADaM Dataset Definitions (IG 1.1 / 1.3)
# ─────────────────────────────────────────────────────────────────────────────


def _adsl_template() -> ADaMDatasetTemplate:
    """ADSL: Subject-Level Analysis Dataset."""
    return ADaMDatasetTemplate(
        name="ADSL",
        label="Subject Level Analysis Dataset",
        description=(
            "One record per subject containing demographic, disposition, "
            "and population flag information for all subjects in the study."
        ),
        dataset_class="ADSL",
        natural_keys=["STUDYID", "USUBJID"],
        variables=[
            # ── Identifiers ──
            ADaMVariableSpec(
                "STUDYID",
                "Study Identifier",
                "Char",
                required=True,
                core="Req",
                max_length=20,
                source="SDTM.DM",
            ),
            ADaMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                required=True,
                core="Req",
                max_length=40,
                source="SDTM.DM",
            ),
            ADaMVariableSpec(
                "SUBJID",
                "Subject Identifier for the Study",
                "Char",
                required=True,
                core="Req",
                max_length=20,
                source="SDTM.DM",
            ),
            ADaMVariableSpec(
                "SITEID",
                "Study Site Identifier",
                "Char",
                required=True,
                core="Req",
                max_length=20,
                source="SDTM.DM",
            ),
            # ── Treatment Variables ──
            ADaMVariableSpec(
                "TRT01P",
                "Planned Treatment for Period 01",
                "Char",
                required=True,
                core="Req",
                max_length=200,
            ),
            ADaMVariableSpec(
                "TRT01A",
                "Actual Treatment for Period 01",
                "Char",
                core="Cond",
                max_length=200,
                condition="Required if actual differs from planned",
            ),
            ADaMVariableSpec(
                "TRT01PN",
                "Planned Treatment for Period 01 (N)",
                "Num",
                core="Cond",
                condition="Required if treatment mapped to numeric",
            ),
            ADaMVariableSpec("TRT01AN", "Actual Treatment for Period 01 (N)", "Num", core="Perm"),
            ADaMVariableSpec(
                "TRTSDTM",
                "Datetime of First Exposure to Treatment",
                "Num",
                core="Perm",
                source="Derived from SDTM.EX",
            ),
            ADaMVariableSpec(
                "TRTSDT",
                "Date of First Exposure to Treatment",
                "Num",
                core="Cond",
                source="Derived from SDTM.EX",
                condition="Required if treatment start used in derivations",
            ),
            ADaMVariableSpec(
                "TRTEDT",
                "Date of Last Exposure to Treatment",
                "Num",
                core="Cond",
                source="Derived from SDTM.EX",
                condition="Required if treatment end used in derivations",
            ),
            # ── Population Flags ──
            ADaMVariableSpec(
                "SAFFL",
                "Safety Population Flag",
                "Char",
                core="Cond",
                max_length=1,
                controlled_term="NY",
                is_population_flag=True,
                condition="Required if safety population defined",
            ),
            ADaMVariableSpec(
                "ITTFL",
                "Intent-To-Treat Population Flag",
                "Char",
                core="Cond",
                max_length=1,
                controlled_term="NY",
                is_population_flag=True,
                condition="Required if ITT population defined",
            ),
            ADaMVariableSpec(
                "EFFFL",
                "Efficacy Population Flag",
                "Char",
                core="Perm",
                max_length=1,
                controlled_term="NY",
                is_population_flag=True,
            ),
            ADaMVariableSpec(
                "RANDFL",
                "Randomized Population Flag",
                "Char",
                core="Perm",
                max_length=1,
                controlled_term="NY",
                is_population_flag=True,
            ),
            ADaMVariableSpec(
                "ENRLFL",
                "Enrolled Population Flag",
                "Char",
                core="Perm",
                max_length=1,
                controlled_term="NY",
                is_population_flag=True,
            ),
            ADaMVariableSpec(
                "PPROTFL",
                "Per-Protocol Population Flag",
                "Char",
                core="Perm",
                max_length=1,
                controlled_term="NY",
                is_population_flag=True,
            ),
            ADaMVariableSpec(
                "COMPLFL",
                "Completers Population Flag",
                "Char",
                core="Perm",
                max_length=1,
                controlled_term="NY",
                is_population_flag=True,
            ),
            # ── Demographics ──
            ADaMVariableSpec(
                "AGE",
                "Age",
                "Num",
                core="Cond",
                source="SDTM.DM",
                condition="Required if age used in analysis",
            ),
            ADaMVariableSpec(
                "AGEU",
                "Age Units",
                "Char",
                core="Cond",
                max_length=10,
                controlled_term="AGEU",
                source="SDTM.DM",
                condition="Required if AGE present",
            ),
            ADaMVariableSpec("AGEGR1", "Pooled Age Group 1", "Char", core="Perm", max_length=40),
            ADaMVariableSpec("AGEGR1N", "Pooled Age Group 1 (N)", "Num", core="Perm"),
            ADaMVariableSpec(
                "SEX",
                "Sex",
                "Char",
                core="Cond",
                max_length=2,
                controlled_term="SEX",
                source="SDTM.DM",
                condition="Required if sex used in analysis",
            ),
            ADaMVariableSpec(
                "RACE",
                "Race",
                "Char",
                core="Cond",
                max_length=60,
                controlled_term="RACE",
                source="SDTM.DM",
                condition="Required if race used in analysis",
            ),
            ADaMVariableSpec(
                "ETHNIC",
                "Ethnicity",
                "Char",
                core="Perm",
                max_length=40,
                controlled_term="ETHNIC",
                source="SDTM.DM",
            ),
            ADaMVariableSpec(
                "COUNTRY",
                "Country",
                "Char",
                core="Perm",
                max_length=3,
                controlled_term="COUNTRY",
                source="SDTM.DM",
            ),
            # ── Disposition ──
            ADaMVariableSpec(
                "DCSREAS",
                "Reason for Discontinuation from Study",
                "Char",
                core="Perm",
                max_length=200,
            ),
            ADaMVariableSpec(
                "DCTREAS",
                "Reason for Discontinuation from Treatment",
                "Char",
                core="Perm",
                max_length=200,
            ),
            # ── Study Dates ──
            ADaMVariableSpec(
                "RFSTDTC",
                "Subject Reference Start Date/Time",
                "Char",
                core="Perm",
                max_length=64,
                source="SDTM.DM",
            ),
            ADaMVariableSpec(
                "RFENDTC",
                "Subject Reference End Date/Time",
                "Char",
                core="Perm",
                max_length=64,
                source="SDTM.DM",
            ),
        ],
    )


def _bds_template() -> ADaMDatasetTemplate:
    """BDS: Basic Data Structure (e.g., ADVS, ADLB, ADEG).

    The BDS is the most common ADaM structure for analysis datasets containing one or more records
    per subject per analysis parameter per analysis timepoint.
    """
    return ADaMDatasetTemplate(
        name="BDS",
        label="Basic Data Structure",
        description=(
            "One or more records per subject per analysis parameter per "
            "analysis timepoint. Used for efficacy, lab, vital signs, etc."
        ),
        dataset_class="BDS",
        natural_keys=["STUDYID", "USUBJID", "PARAMCD", "AVISIT", "ADT"],
        variables=[
            # ── Identifiers ──
            ADaMVariableSpec(
                "STUDYID", "Study Identifier", "Char", required=True, core="Req", max_length=20
            ),
            ADaMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                required=True,
                core="Req",
                max_length=40,
            ),
            # ── Treatment (copied from ADSL) ──
            ADaMVariableSpec(
                "TRT01P",
                "Planned Treatment for Period 01",
                "Char",
                core="Cond",
                max_length=200,
                condition="Required if used in analysis",
            ),
            ADaMVariableSpec(
                "TRT01A", "Actual Treatment for Period 01", "Char", core="Perm", max_length=200
            ),
            ADaMVariableSpec("TRT01PN", "Planned Treatment for Period 01 (N)", "Num", core="Perm"),
            ADaMVariableSpec("TRT01AN", "Actual Treatment for Period 01 (N)", "Num", core="Perm"),
            # ── Parameter Variables ──
            ADaMVariableSpec(
                "PARAMCD", "Parameter Code", "Char", required=True, core="Req", max_length=8
            ),
            ADaMVariableSpec(
                "PARAM", "Parameter", "Char", required=True, core="Req", max_length=200
            ),
            ADaMVariableSpec("PARAMN", "Parameter (N)", "Num", core="Perm"),
            ADaMVariableSpec(
                "PARCAT1", "Parameter Category 1", "Char", core="Perm", max_length=200
            ),
            ADaMVariableSpec("PARCAT1N", "Parameter Category 1 (N)", "Num", core="Perm"),
            # ── Analysis Values ──
            ADaMVariableSpec("AVAL", "Analysis Value", "Num", required=True, core="Req"),
            ADaMVariableSpec(
                "AVALC",
                "Analysis Value (C)",
                "Char",
                core="Cond",
                max_length=200,
                condition="Required if character result needed",
            ),
            ADaMVariableSpec(
                "BASE",
                "Baseline Value",
                "Num",
                core="Cond",
                condition="Required if change from baseline analyzed",
            ),
            ADaMVariableSpec("BASEC", "Baseline Value (C)", "Char", core="Perm", max_length=200),
            ADaMVariableSpec(
                "CHG",
                "Change from Baseline",
                "Num",
                core="Cond",
                condition="Required if change from baseline analyzed",
            ),
            ADaMVariableSpec("PCHG", "Percent Change from Baseline", "Num", core="Perm"),
            # ── Analysis Timepoint ──
            ADaMVariableSpec(
                "AVISIT",
                "Analysis Visit",
                "Char",
                core="Cond",
                max_length=200,
                condition="Required if multiple timepoints",
            ),
            ADaMVariableSpec("AVISITN", "Analysis Visit (N)", "Num", core="Perm"),
            ADaMVariableSpec(
                "ADT", "Analysis Date", "Num", core="Cond", condition="Required if timing relevant"
            ),
            ADaMVariableSpec("ADY", "Analysis Relative Day", "Num", core="Perm"),
            ADaMVariableSpec("ATPT", "Analysis Timepoint", "Char", core="Perm", max_length=200),
            ADaMVariableSpec("ATPTN", "Analysis Timepoint (N)", "Num", core="Perm"),
            # ── Flags ──
            ADaMVariableSpec(
                "ABLFL",
                "Baseline Record Flag",
                "Char",
                core="Cond",
                max_length=1,
                controlled_term="NY",
                condition="Required if baseline value derived",
            ),
            ADaMVariableSpec(
                "ANL01FL",
                "Analysis Record Flag 01",
                "Char",
                core="Perm",
                max_length=1,
                controlled_term="NY",
            ),
            ADaMVariableSpec(
                "AENTMTFL",
                "Last Post-Baseline Obs Before/On Trt End",
                "Char",
                core="Perm",
                max_length=1,
                controlled_term="NY",
            ),
            # ── Traceability ──
            ADaMVariableSpec("SRCDOM", "Source Data Domain", "Char", core="Perm", max_length=8),
            ADaMVariableSpec("SRCVAR", "Source Data Variable", "Char", core="Perm", max_length=40),
            ADaMVariableSpec("SRCSEQ", "Source Data Sequence Number", "Num", core="Perm"),
            # ── Criterion Variables ──
            ADaMVariableSpec("CRIT1", "Analysis Criterion 1", "Char", core="Perm", max_length=200),
            ADaMVariableSpec(
                "CRIT1FL",
                "Criterion 1 Evaluation Result Flag",
                "Char",
                core="Perm",
                max_length=1,
                controlled_term="NY",
            ),
        ],
    )


def _adae_template() -> ADaMDatasetTemplate:
    """ADAE: Adverse Event Analysis Dataset."""
    return ADaMDatasetTemplate(
        name="ADAE",
        label="Adverse Event Analysis Dataset",
        description=(
            "One record per subject per adverse event per analysis need. "
            "Contains occurrence-based AE data with analysis flags."
        ),
        dataset_class="ADAE",
        natural_keys=["STUDYID", "USUBJID", "AETERM", "ASTDT"],
        variables=[
            # ── Identifiers ──
            ADaMVariableSpec(
                "STUDYID", "Study Identifier", "Char", required=True, core="Req", max_length=20
            ),
            ADaMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                required=True,
                core="Req",
                max_length=40,
            ),
            ADaMVariableSpec(
                "AESEQ", "Sequence Number", "Num", required=True, core="Req", source="SDTM.AE"
            ),
            # ── Treatment ──
            ADaMVariableSpec(
                "TRT01P",
                "Planned Treatment for Period 01",
                "Char",
                core="Cond",
                max_length=200,
                condition="Required if used in analysis",
            ),
            ADaMVariableSpec(
                "TRT01A",
                "Actual Treatment for Period 01",
                "Char",
                core="Cond",
                max_length=200,
                condition="Required if actual treatment used",
            ),
            ADaMVariableSpec(
                "TRTA",
                "Actual Treatment",
                "Char",
                core="Cond",
                max_length=200,
                condition="Required if period-specific treatment needed",
            ),
            ADaMVariableSpec("TRTAN", "Actual Treatment (N)", "Num", core="Perm"),
            # ── AE Variables (from SDTM) ──
            ADaMVariableSpec(
                "AETERM",
                "Reported Term for the Adverse Event",
                "Char",
                required=True,
                core="Req",
                max_length=200,
                source="SDTM.AE",
            ),
            ADaMVariableSpec(
                "AEDECOD",
                "Dictionary-Derived Term",
                "Char",
                required=True,
                core="Req",
                max_length=200,
                source="SDTM.AE",
            ),
            ADaMVariableSpec(
                "AEBODSYS",
                "Body System or Organ Class",
                "Char",
                core="Cond",
                max_length=200,
                source="SDTM.AE",
                condition="Required if body system used in analysis",
            ),
            ADaMVariableSpec(
                "AESEV",
                "Severity/Intensity",
                "Char",
                core="Perm",
                max_length=20,
                controlled_term="AESEV",
                source="SDTM.AE",
            ),
            ADaMVariableSpec(
                "AESER",
                "Serious Event",
                "Char",
                core="Cond",
                max_length=2,
                controlled_term="NY",
                source="SDTM.AE",
                condition="Required if SAE analyzed",
            ),
            ADaMVariableSpec(
                "AEREL", "Causality", "Char", core="Perm", max_length=40, source="SDTM.AE"
            ),
            ADaMVariableSpec(
                "AEACN",
                "Action Taken with Study Treatment",
                "Char",
                core="Perm",
                max_length=40,
                source="SDTM.AE",
            ),
            ADaMVariableSpec(
                "AEOUT",
                "Outcome of Adverse Event",
                "Char",
                core="Perm",
                max_length=40,
                source="SDTM.AE",
            ),
            # ── Analysis Dates ──
            ADaMVariableSpec(
                "ASTDT",
                "Analysis Start Date",
                "Num",
                core="Cond",
                condition="Required if onset timing analyzed",
            ),
            ADaMVariableSpec("ASTDTM", "Analysis Start Datetime", "Num", core="Perm"),
            ADaMVariableSpec("AENDT", "Analysis End Date", "Num", core="Perm"),
            ADaMVariableSpec("AENDTM", "Analysis End Datetime", "Num", core="Perm"),
            ADaMVariableSpec("ASTDY", "Analysis Start Relative Day", "Num", core="Perm"),
            ADaMVariableSpec("AENDY", "Analysis End Relative Day", "Num", core="Perm"),
            ADaMVariableSpec("ADURN", "AE Duration (N)", "Num", core="Perm"),
            ADaMVariableSpec("ADURU", "AE Duration Units", "Char", core="Perm", max_length=40),
            # ── Flags ──
            ADaMVariableSpec(
                "TRTEMFL",
                "Treatment Emergent Flag",
                "Char",
                core="Cond",
                max_length=1,
                controlled_term="NY",
                condition="Required for TEAE analysis",
            ),
            ADaMVariableSpec(
                "PREFL",
                "Pre-Treatment Flag",
                "Char",
                core="Perm",
                max_length=1,
                controlled_term="NY",
            ),
            ADaMVariableSpec("AREL", "Analysis Causality", "Char", core="Perm", max_length=40),
            ADaMVariableSpec(
                "CQ01NAM", "Customized Query 01 Name", "Char", core="Perm", max_length=200
            ),
            ADaMVariableSpec("SMQ01NAM", "SMQ 01 Name", "Char", core="Perm", max_length=200),
            # ── Severity Analysis ──
            ADaMVariableSpec(
                "ASEV", "Analysis Severity/Intensity", "Char", core="Perm", max_length=20
            ),
            ADaMVariableSpec("ASEVN", "Analysis Severity/Intensity (N)", "Num", core="Perm"),
        ],
    )


def _adtte_template() -> ADaMDatasetTemplate:
    """ADTTE: Time-to-Event Analysis Dataset."""
    return ADaMDatasetTemplate(
        name="ADTTE",
        label="Time-to-Event Analysis Dataset",
        description=(
            "One record per subject per analysis parameter for time-to-event "
            "analyses (e.g., overall survival, progression-free survival)."
        ),
        dataset_class="ADTTE",
        natural_keys=["STUDYID", "USUBJID", "PARAMCD"],
        variables=[
            # ── Identifiers ──
            ADaMVariableSpec(
                "STUDYID", "Study Identifier", "Char", required=True, core="Req", max_length=20
            ),
            ADaMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                required=True,
                core="Req",
                max_length=40,
            ),
            # ── Treatment ──
            ADaMVariableSpec(
                "TRT01P",
                "Planned Treatment for Period 01",
                "Char",
                core="Cond",
                max_length=200,
                condition="Required if used in analysis",
            ),
            ADaMVariableSpec(
                "TRT01A", "Actual Treatment for Period 01", "Char", core="Perm", max_length=200
            ),
            ADaMVariableSpec("TRT01PN", "Planned Treatment for Period 01 (N)", "Num", core="Perm"),
            ADaMVariableSpec("TRT01AN", "Actual Treatment for Period 01 (N)", "Num", core="Perm"),
            # ── Parameter Variables ──
            ADaMVariableSpec(
                "PARAMCD", "Parameter Code", "Char", required=True, core="Req", max_length=8
            ),
            ADaMVariableSpec(
                "PARAM", "Parameter", "Char", required=True, core="Req", max_length=200
            ),
            # ── Time-to-Event Variables ──
            ADaMVariableSpec("AVAL", "Analysis Value", "Num", required=True, core="Req"),
            ADaMVariableSpec(
                "STARTDT", "Time-to-Event Origin Date", "Num", required=True, core="Req"
            ),
            ADaMVariableSpec("ADT", "Analysis Date", "Num", required=True, core="Req"),
            ADaMVariableSpec("CNSR", "Censor", "Num", required=True, core="Req"),
            ADaMVariableSpec(
                "EVNTDESC",
                "Event Description",
                "Char",
                core="Cond",
                max_length=200,
                condition="Required for traceability",
            ),
            ADaMVariableSpec(
                "CNSDTDSC",
                "Censor Date Description",
                "Char",
                core="Cond",
                max_length=200,
                condition="Required for traceability",
            ),
            # ── Supporting Variables ──
            ADaMVariableSpec("SRCDOM", "Source Data Domain", "Char", core="Perm", max_length=8),
            ADaMVariableSpec("SRCVAR", "Source Data Variable", "Char", core="Perm", max_length=40),
            ADaMVariableSpec("SRCSEQ", "Source Data Sequence Number", "Num", core="Perm"),
        ],
    )


# Registry of all ADaM dataset templates
_ADAM_TEMPLATES: dict[str, callable] = {
    "ADSL": _adsl_template,
    "BDS": _bds_template,
    "ADAE": _adae_template,
    "ADTTE": _adtte_template,
}


def get_adam_dataset(name: str) -> ADaMDatasetTemplate:
    """Get the ADaM template for a specific dataset.

    Parameters
    ----------
    name
        Dataset name (e.g., `"ADSL"`, `"BDS"`, `"ADAE"`, `"ADTTE"`). This is case-insensitive.

    Returns
    -------
    ADaMDatasetTemplate
        The structural template for the dataset.

    Raises
    ------
    KeyError
        If the dataset is not supported.
    """
    name_upper = name.upper()
    if name_upper not in _ADAM_TEMPLATES:
        available = sorted(_ADAM_TEMPLATES.keys())
        raise KeyError(f"ADaM dataset '{name}' is not supported. Available datasets: {available}")
    return _ADAM_TEMPLATES[name_upper]()


def list_adam_datasets() -> list[str]:
    """List all available ADaM dataset template names.

    Returns
    -------
    list[str]
        Sorted list of dataset names.
    """
    return sorted(_ADAM_TEMPLATES.keys())


def validate_adam_structure(
    data: Any,
    dataset: str,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate structural conformance of a dataset against an ADaM template.

    Parameters
    ----------
    data
        A DataFrame (pandas, polars) to check.
    dataset
        ADaM dataset name (e.g., `"ADSL"`, `"BDS"`, `"ADAE"`, `"ADTTE"`). This is case-insensitive.
    strict
        If True, also reports missing conditional variables and unknown variables.

    Returns
    -------
    dict
        Validation results with keys:

        - "dataset": the dataset name
        - "dataset_class": ADaM class
        - "valid": True if no required violations found
        - "missing_required": list of missing required variable names
        - "missing_conditional": list of missing conditionally required variables (strict)
        - "unknown_variables": list of unknown column names (strict)
        - "population_flags_found": list of population flag variables present
        - "issues": list of human-readable issue strings
    """
    import narwhals as nw

    template = get_adam_dataset(dataset)
    df = nw.from_native(data, eager_only=True)
    columns = set(df.columns)

    issues: list[str] = []
    result: dict[str, Any] = {
        "dataset": dataset.upper(),
        "dataset_class": template.dataset_class,
        "valid": True,
        "missing_required": [],
        "missing_conditional": [],
        "unknown_variables": [],
        "population_flags_found": [],
        "issues": issues,
    }

    # Check required variables
    for var_name in template.required_variables:
        if var_name not in columns:
            result["missing_required"].append(var_name)
            issues.append(f"Required variable '{var_name}' is missing")

    if result["missing_required"]:
        result["valid"] = False

    # Check population flags present
    for var in template.variables:
        if var.is_population_flag and var.name in columns:
            result["population_flags_found"].append(var.name)

    # ADSL must have at least one population flag
    if template.dataset_class == "ADSL" and not result["population_flags_found"]:
        issues.append("ADSL should have at least one population flag (e.g., SAFFL, ITTFL)")

    # Strict mode checks
    if strict:
        for var in template.variables:
            if var.core == "Cond" and var.name not in columns:
                result["missing_conditional"].append(var.name)
                issues.append(
                    f"Conditionally required variable '{var.name}' is missing ({var.condition})"
                )

        template_names = {v.name for v in template.variables}
        for col in columns:
            if col not in template_names:
                result["unknown_variables"].append(col)
                issues.append(f"Variable '{col}' is not defined in {dataset.upper()} template")

    return result
