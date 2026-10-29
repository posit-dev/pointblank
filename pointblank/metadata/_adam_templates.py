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
    """The specification of one variable in an ADaM dataset template.

    An [`ADaMDatasetTemplate`](`pointblank.ADaMDatasetTemplate`) holds an `ADaMVariableSpec` for
    each variable of its dataset, in its `variables` list. The specification gives the variable's
    label, type and core designation, the condition under which a conditionally required variable
    is needed, where its values come from, and whether it is a population flag.
    [`validate_adam()`](`pointblank.validate_adam`) uses `required` and `is_population_flag` to
    decide which checks to make.

    Parameters
    ----------
    name
        The variable name, such as `"USUBJID"`, `"PARAMCD"` or `"AVAL"`.
    label
        The variable label, such as `"Unique Subject Identifier"`.
    dtype
        The data type: `"Char"` for character variables or `"Num"` for numeric ones (including
        dates, which ADaM stores as numbers).
    core
        The core designation: `"Req"` (required), `"Cond"` (conditionally required: needed when
        the condition in `condition` holds) or `"Perm"` (permissible). The default is `"Perm"`.
    required
        Whether the variable is required in every dataset of this type. In the templates, this is
        `True` exactly when `core` is `"Req"`. The default is `False`.
    max_length
        The maximum length of the values of a character variable. The default is `None` (no
        maximum).
    controlled_term
        The name of the controlled terminology codelist for the variable's values, such as `"NY"`
        for the population flags. The default is `None`.
    source
        Where the variable's values come from, such as `"SDTM.DM"` (copied from the Demographics
        domain) or `"Derived from SDTM.EX"`. The default is `None` (not specified).
    condition
        For a conditionally required variable, a description of when it is required, such as
        `"Required if safety population defined"`. The default is `None`.
    is_population_flag
        Whether the variable is a population flag (such as `SAFFL` or `ITTFL`), whose values must
        be `"Y"` or `"N"`. The default is `False`.

    Examples
    --------
    Specifications are usually taken from a template. Here is the one for the safety population
    flag, `SAFFL`, in the ADSL template:

    ```{python}
    import pointblank as pb

    saffl = pb.get_adam_dataset("ADSL").get_variable("SAFFL")

    saffl
    ```

    `SAFFL` is a one-character population flag with values from the `NY` codelist. It is
    conditionally required, and `condition` says when: whenever the study defines a safety
    population.

    See Also
    --------
    ADaMDatasetTemplate : The template that holds the specifications of a dataset's variables.
    SDTMVariableSpec : The equivalent for SDTM domain templates.
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
    """The structural template for an ADaM dataset.

    An `ADaMDatasetTemplate` describes the structure of one type of CDISC Analysis Data Model
    (ADaM) dataset: its label and class, the variables that form its natural key, and the
    specifications of its variables (as [`ADaMVariableSpec`](`pointblank.ADaMVariableSpec`)
    objects, in order). Templates are obtained with
    [`get_adam_dataset()`](`pointblank.get_adam_dataset`).

    Parameters
    ----------
    name
        The template name: `"ADSL"`, `"BDS"`, `"ADAE"` or `"ADTTE"`.
    label
        The dataset label, such as `"Subject Level Analysis Dataset"`.
    description
        A short description of what the dataset holds and what a record is.
    dataset_class
        The ADaM dataset class, which [`validate_adam()`](`pointblank.validate_adam`) uses to
        choose the checks for the type of dataset. In the templates, it's the same as `name`.
    variables
        The specifications of the dataset's variables, in order. The default is an empty list.
    natural_keys
        The names of the variables that together identify a record. The default is an empty list.

    Examples
    --------
    Let's look at the template for the Basic Data Structure (BDS), the structure of analysis
    datasets such as ADVS (vital signs) and ADLB (laboratory results):

    ```{python}
    import pointblank as pb

    bds = pb.get_adam_dataset("BDS")

    print(bds.label, "|", bds.dataset_class)
    print(bds.description)
    print(bds.natural_keys)
    ```

    A BDS dataset has a record for each analysis parameter (`PARAMCD`) at each visit and date,
    for each subject. The template's properties list the required and conditionally required
    variables:

    ```{python}
    print(bds.required_variables)
    print(bds.conditional_variables)
    ```

    Every BDS dataset needs the identifiers, the parameter and the analysis value. The
    conditional variables, such as the baseline value `BASE` and the change from baseline `CHG`,
    are needed only for some analyses.

    See Also
    --------
    get_adam_dataset : Get the template for a type of dataset.
    ADaMVariableSpec : The specification of a variable in the template.
    SDTMDomainTemplate : The equivalent for SDTM domains.
    """

    name: str
    label: str
    description: str
    dataset_class: str
    variables: list[ADaMVariableSpec] = dataclass_field(default_factory=list)
    natural_keys: list[str] = dataclass_field(default_factory=list)

    @property
    def required_variables(self) -> list[str]:
        """The names of the required variables, in order.

        Required variables (core designation `"Req"`) must be in every dataset of this type.
        [`validate_adam()`](`pointblank.validate_adam`) adds a null check for each one, and
        [`validate_adam_structure()`](`pointblank.validate_adam_structure`) reports the ones that
        are missing.

        Examples
        --------
        ```{python}
        import pointblank as pb

        pb.get_adam_dataset("ADTTE").required_variables
        ```

        A time-to-event dataset (ADTTE) needs the parameter, the time to the event (`AVAL`), the
        start and event dates, and the censoring indicator (`CNSR`). This is a property, so it's
        used without parentheses.
        """
        return [v.name for v in self.variables if v.required]

    @property
    def conditional_variables(self) -> list[str]:
        """The names of the conditionally required variables, in order.

        A conditionally required variable (core designation `"Cond"`) is needed only in some
        situations, which the variable's `condition` describes.
        [`validate_adam_structure()`](`pointblank.validate_adam_structure`) reports the ones that
        are missing when `strict=True`.

        Examples
        --------
        ```{python}
        import pointblank as pb

        adsl = pb.get_adam_dataset("ADSL")

        for name in adsl.conditional_variables[:4]:
            print(name, "->", adsl.get_variable(name).condition)
        ```

        These are the first four conditional variables of ADSL, with the condition for each one.
        This is a property, so it's used without parentheses.
        """
        return [v.name for v in self.variables if v.core == "Cond"]

    @property
    def population_flags(self) -> list[str]:
        """The names of the population flag variables, in order.

        Population flags, such as `SAFFL` (safety) and `ITTFL` (intent-to-treat), say whether each
        subject is in an analysis population, with the values `"Y"` and `"N"`. Only the ADSL
        template has them.

        Examples
        --------
        ```{python}
        import pointblank as pb

        pb.get_adam_dataset("ADSL").population_flags
        ```

        [`validate_adam()`](`pointblank.validate_adam`) checks each of these flags that is a
        column of the data. This is a property, so it's used without parentheses.
        """
        return [v.name for v in self.variables if v.is_population_flag]

    def get_variable(self, name: str) -> ADaMVariableSpec | None:
        """Get the specification of a variable.

        Parameters
        ----------
        name
            The variable name. This is case-sensitive, and ADaM variable names are in upper case.

        Returns
        -------
        ADaMVariableSpec | None
            The specification of the variable, or `None` if the template has no variable with that
            name.

        Examples
        --------
        ```{python}
        import pointblank as pb

        adtte = pb.get_adam_dataset("ADTTE")

        adtte.get_variable("CNSR")
        ```

        `CNSR` is the required censoring indicator of a time-to-event dataset. A name that isn't
        in the template gives `None`:

        ```{python}
        print(adtte.get_variable("AESEQ"))
        ```

        `AESEQ` belongs to the ADAE template, not to ADTTE.
        """
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
                "Reason for Discontinuation of Treatment",
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
                "Analysis Flag 01",
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
            ADaMVariableSpec("SRCDOM", "Source Data", "Char", core="Perm", max_length=8),
            ADaMVariableSpec("SRCVAR", "Source Variable", "Char", core="Perm", max_length=40),
            ADaMVariableSpec("SRCSEQ", "Source Sequence Number", "Num", core="Perm"),
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
            ADaMVariableSpec("ADURN", "Analysis Duration (N)", "Num", core="Perm"),
            ADaMVariableSpec(
                "ADURU", "Analysis Duration Units", "Char", core="Perm", max_length=40
            ),
            # ── Flags ──
            ADaMVariableSpec(
                "TRTEMFL",
                "Treatment Emergent Analysis Flag",
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
                "STARTDT", "Time-to-Event Origin Date for Subject", "Num", required=True, core="Req"
            ),
            ADaMVariableSpec("ADT", "Analysis Date", "Num", required=True, core="Req"),
            ADaMVariableSpec("CNSR", "Censor", "Num", required=True, core="Req"),
            ADaMVariableSpec(
                "EVNTDESC",
                "Event or Censoring Description",
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
            ADaMVariableSpec("SRCDOM", "Source Data", "Char", core="Perm", max_length=8),
            ADaMVariableSpec("SRCVAR", "Source Variable", "Char", core="Perm", max_length=40),
            ADaMVariableSpec("SRCSEQ", "Source Sequence Number", "Num", core="Perm"),
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
    """Get the template for a type of ADaM dataset.

    Pointblank has templates for four types of CDISC Analysis Data Model (ADaM) dataset: ADSL (the
    subject-level analysis dataset), BDS (the Basic Data Structure, used by datasets such as ADVS
    and ADLB), ADAE (adverse events) and ADTTE (time to event). Each template is an
    [`ADaMDatasetTemplate`](`pointblank.ADaMDatasetTemplate`): the dataset's label and class, the
    variables that form its natural key, and an [`ADaMVariableSpec`](`pointblank.ADaMVariableSpec`)
    for each of its variables. The templates are what
    [`validate_adam()`](`pointblank.validate_adam`),
    [`validate_adam_structure()`](`pointblank.validate_adam_structure`) and
    [`adam_to_metadata()`](`pointblank.adam_to_metadata`) work from.

    The templates cover the common variables of each type of dataset rather than every variable
    that the ADaM Implementation Guide describes. Each call returns a new template object, so
    changing one doesn't affect later calls.

    Parameters
    ----------
    name
        The template name: `"ADSL"`, `"BDS"`, `"ADAE"` or `"ADTTE"`. This is case-insensitive.

    Returns
    -------
    ADaMDatasetTemplate
        The template for the dataset.

    Raises
    ------
    KeyError
        If there is no template with that name. The error message lists the templates, as does
        [`list_adam_datasets()`](`pointblank.list_adam_datasets`).

    Examples
    --------
    Let's get the template for the subject-level analysis dataset:

    ```{python}
    import pointblank as pb

    adsl = pb.get_adam_dataset("adsl")

    print(adsl.label)
    print(adsl.natural_keys)
    print(adsl.required_variables)
    ```

    ADSL has one record per subject, so the study and subject identifiers are its natural key.
    Besides the identifiers, the planned treatment `TRT01P` is required. The template also lists
    the population flags:

    ```{python}
    adsl.population_flags
    ```

    These flags (safety, intent-to-treat, efficacy and so on) are conditionally required or
    permissible, and [`validate_adam()`](`pointblank.validate_adam`) checks that the ones in a
    dataset hold only `"Y"` and `"N"`.

    See Also
    --------
    list_adam_datasets : List the dataset templates.
    ADaMDatasetTemplate : The template that is returned.
    validate_adam : Build a validation plan from an ADaM dataset template.
    get_sdtm_domain : The equivalent for SDTM domains.
    """
    name_upper = name.upper()
    if name_upper not in _ADAM_TEMPLATES:
        available = sorted(_ADAM_TEMPLATES.keys())
        raise KeyError(f"ADaM dataset '{name}' is not supported. Available datasets: {available}")
    return _ADAM_TEMPLATES[name_upper]()


def list_adam_datasets() -> list[str]:
    """List the ADaM dataset templates.

    These are the names that [`get_adam_dataset()`](`pointblank.get_adam_dataset`),
    [`validate_adam()`](`pointblank.validate_adam`),
    [`validate_adam_structure()`](`pointblank.validate_adam_structure`) and
    [`adam_to_metadata()`](`pointblank.adam_to_metadata`) accept.

    Returns
    -------
    list[str]
        The template names, in alphabetical order.

    Examples
    --------
    Let's list the templates, together with each one's label:

    ```{python}
    import pointblank as pb

    for name in pb.list_adam_datasets():
        print(f"{name:7}{pb.get_adam_dataset(name).label}")
    ```

    BDS is a structure rather than a single dataset, so its template is used for any BDS dataset
    in a study, such as ADVS or ADLB.

    See Also
    --------
    get_adam_dataset : Get a dataset template.
    list_sdtm_domains : The equivalent for SDTM domains.
    """
    return sorted(_ADAM_TEMPLATES.keys())


def validate_adam_structure(
    data: Any,
    dataset: str,
    strict: bool = False,
) -> dict[str, Any]:
    """Check a dataset's variables against an ADaM dataset template.

    The `validate_adam_structure()` function is a quick check of a dataset's structure, run before
    (or instead of) a full validation with [`validate_adam()`](`pointblank.validate_adam`). It
    checks that every required variable of the template is a column of the dataset, and lists the
    population flags that the dataset has. It returns the findings as a dictionary straight away,
    without building a [`Validate`](`pointblank.Validate`) plan.

    For an ADSL dataset with no population flags at all, an issue is added to `"issues"`, but the
    dataset is still `"valid"`. With `strict=True`, two more things are reported: the
    conditionally required variables (core designation `"Cond"`) that are missing, and the columns
    that aren't variables in the template. These don't affect `"valid"` either, since whether a
    conditional variable is needed depends on the study, and the templates only cover the common
    variables of each type of dataset.

    The column order, the data types and the values aren't checked.

    Parameters
    ----------
    data
        The dataset to check, as a Polars or Pandas DataFrame.
    dataset
        The name of the ADaM dataset template: `"ADSL"`, `"BDS"`, `"ADAE"` or `"ADTTE"`. This is
        case-insensitive.
    strict
        Whether to also report missing conditional variables and columns that aren't in the
        template. The default is `False`.

    Returns
    -------
    dict
        A dictionary with these keys:

        - `"dataset"`: the template name, in upper case.
        - `"dataset_class"`: the ADaM class of the template.
        - `"valid"`: `True` if no required variable is missing.
        - `"missing_required"`: the names of the required variables that aren't columns of the
        data.
        - `"missing_conditional"`: the names of the conditional variables that aren't columns of
        the data (filled in only with `strict=True`).
        - `"unknown_variables"`: the columns that aren't variables in the template, in the order of
        the data's columns (filled in only with `strict=True`).
        - `"population_flags_found"`: the population flags that are columns of the data.
        - `"issues"`: a message for each of the problems above. With `strict=True`, the message for
        a missing conditional variable includes its condition.

    Raises
    ------
    KeyError
        If there is no template with that name.

    Examples
    --------
    Let's check a small ADSL dataset that lacks the site identifier, `SITEID`, a required ADSL
    variable:

    ```{python}
    import pointblank as pb
    import polars as pl

    adsl = pl.DataFrame(
        {
            "STUDYID": ["STUDY01"] * 3,
            "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
            "SUBJID": ["001", "002", "003"],
            "TRT01P": ["Drug A", "Placebo", "Drug A"],
            "SAFFL": ["Y", "Y", "N"],
            "BMIBL": [24.1, 31.7, 22.9],
        }
    )

    result = pb.validate_adam_structure(adsl, dataset="ADSL")

    print(result["valid"])
    print(result["missing_required"])
    print(result["population_flags_found"])
    ```

    The dataset isn't valid because of the missing `SITEID`, and it has one population flag,
    `SAFFL`. In strict mode, the conditional variables and the unknown columns are reported too:

    ```{python}
    result = pb.validate_adam_structure(adsl, dataset="ADSL", strict=True)

    print(result["missing_conditional"])
    print(result["unknown_variables"])
    ```

    The baseline BMI column, `BMIBL`, isn't in the template. Since ADaM allows a sponsor to add
    variables like this one, it's reported only so that it can be reviewed.

    See Also
    --------
    validate_adam : Build a validation plan from an ADaM dataset template.
    get_adam_dataset : Get the template that the dataset is checked against.
    validate_sdtm_structure : The equivalent for SDTM datasets.
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
        # Iterate over `df.columns` (not the set) to report columns in table order
        for col in df.columns:
            if col not in template_names:
                result["unknown_variables"].append(col)
                issues.append(f"Variable '{col}' is not defined in {dataset.upper()} template")

    return result
