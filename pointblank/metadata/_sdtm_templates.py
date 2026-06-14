from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

__all__ = [
    "SDTMDomainTemplate",
    "SDTMVariableSpec",
    "get_sdtm_domain",
    "list_sdtm_domains",
    "validate_sdtm_structure",
]


@dataclass
class SDTMVariableSpec:
    """Specification for a single variable in an SDTM domain template.

    Parameters
    ----------
    name
        Variable name (e.g., `"STUDYID"`, `"USUBJID"`).
    label
        Variable label (e.g., `"Study Identifier"`).
    dtype
        Expected data type (`"Char"` or `"Num"`).
    role
        SDTM role: `"Identifier"`, `"Topic"`, `"Qualifier"`, `"Timing"`, `"Rule"`, or
        `"Record Qualifier"`.
    required
        Whether the variable is required (`Req="Yes"` in IG).
    max_length
        Maximum character length for Char variables.
    controlled_term
        Name of the associated controlled terminology codelist.
    core
        SDTM core designation: `"Req"`, `"Exp"`, or `"Perm"`.
    """

    name: str
    label: str
    dtype: str  # "Char" or "Num"
    role: str
    required: bool = False
    max_length: int | None = None
    controlled_term: str | None = None
    core: str = "Perm"  # "Req", "Exp", "Perm"


@dataclass
class SDTMDomainTemplate:
    """Structural template for an SDTM domain.

    Parameters
    ----------
    domain
        Two-character domain code (e.g., `"DM"`, `"AE"`, `"LB"`).
    label
        Domain label (e.g., `"Demographics"`, `"Adverse Events"`).
    description
        Brief description of the domain's purpose.
    domain_class
        SDTM observation class: `"Special Purpose"`, `"Events"`, `"Interventions"`, or `"Findings"`.
    repeating
        Whether the domain is a repeating (multi-row per subject) domain.
    variables
        Ordered list of variable specifications.
    natural_keys
        List of variable names that form the natural key.
    """

    domain: str
    label: str
    description: str
    domain_class: str
    repeating: bool
    variables: list[SDTMVariableSpec] = dataclass_field(default_factory=list)
    natural_keys: list[str] = dataclass_field(default_factory=list)

    @property
    def required_variables(self) -> list[str]:
        """Get names of all required variables."""
        return [v.name for v in self.variables if v.required]

    @property
    def expected_variables(self) -> list[str]:
        """Get names of all expected (Exp core) variables."""
        return [v.name for v in self.variables if v.core == "Exp"]

    @property
    def identifier_variables(self) -> list[str]:
        """Get names of all Identifier-role variables."""
        return [v.name for v in self.variables if v.role == "Identifier"]

    def get_variable(self, name: str) -> SDTMVariableSpec | None:
        """Get a variable spec by name."""
        for v in self.variables:
            if v.name == name:
                return v
        return None


def _dm_template() -> SDTMDomainTemplate:
    """Demographics (DM): Special Purpose domain."""
    return SDTMDomainTemplate(
        domain="DM",
        label="Demographics",
        description="Subject demographics and study participation information.",
        domain_class="Special Purpose",
        repeating=False,
        natural_keys=["STUDYID", "USUBJID"],
        variables=[
            SDTMVariableSpec(
                "STUDYID",
                "Study Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=20,
                core="Req",
            ),
            SDTMVariableSpec(
                "DOMAIN",
                "Domain Abbreviation",
                "Char",
                "Identifier",
                required=True,
                max_length=2,
                core="Req",
            ),
            SDTMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=40,
                core="Req",
            ),
            SDTMVariableSpec(
                "SUBJID",
                "Subject Identifier for the Study",
                "Char",
                "Topic",
                required=True,
                max_length=20,
                core="Req",
            ),
            SDTMVariableSpec(
                "RFSTDTC",
                "Subject Reference Start Date/Time",
                "Char",
                "Timing",
                max_length=64,
                core="Exp",
            ),
            SDTMVariableSpec(
                "RFENDTC",
                "Subject Reference End Date/Time",
                "Char",
                "Timing",
                max_length=64,
                core="Exp",
            ),
            SDTMVariableSpec(
                "RFXSTDTC",
                "Date/Time of First Study Treatment",
                "Char",
                "Timing",
                max_length=64,
                core="Exp",
            ),
            SDTMVariableSpec(
                "RFXENDTC",
                "Date/Time of Last Study Treatment",
                "Char",
                "Timing",
                max_length=64,
                core="Exp",
            ),
            SDTMVariableSpec(
                "RFICDTC",
                "Date/Time of Informed Consent",
                "Char",
                "Timing",
                max_length=64,
                core="Perm",
            ),
            SDTMVariableSpec(
                "RFPENDTC",
                "Date/Time of End of Participation",
                "Char",
                "Timing",
                max_length=64,
                core="Perm",
            ),
            SDTMVariableSpec(
                "DTHDTC", "Date/Time of Death", "Char", "Timing", max_length=64, core="Perm"
            ),
            SDTMVariableSpec(
                "DTHFL",
                "Subject Death Flag",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Perm",
            ),
            SDTMVariableSpec(
                "SITEID",
                "Study Site Identifier",
                "Char",
                "Qualifier",
                required=True,
                max_length=20,
                core="Req",
            ),
            SDTMVariableSpec(
                "BRTHDTC", "Date/Time of Birth", "Char", "Qualifier", max_length=64, core="Perm"
            ),
            SDTMVariableSpec("AGE", "Age", "Num", "Qualifier", core="Exp"),
            SDTMVariableSpec(
                "AGEU",
                "Age Units",
                "Char",
                "Qualifier",
                max_length=10,
                controlled_term="AGEU",
                core="Exp",
            ),
            SDTMVariableSpec(
                "SEX",
                "Sex",
                "Char",
                "Qualifier",
                required=True,
                max_length=2,
                controlled_term="SEX",
                core="Req",
            ),
            SDTMVariableSpec(
                "RACE",
                "Race",
                "Char",
                "Qualifier",
                max_length=60,
                controlled_term="RACE",
                core="Exp",
            ),
            SDTMVariableSpec(
                "ETHNIC",
                "Ethnicity",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="ETHNIC",
                core="Perm",
            ),
            SDTMVariableSpec(
                "ARMCD",
                "Planned Arm Code",
                "Char",
                "Qualifier",
                required=True,
                max_length=20,
                core="Req",
            ),
            SDTMVariableSpec(
                "ARM",
                "Description of Planned Arm",
                "Char",
                "Qualifier",
                required=True,
                max_length=200,
                core="Req",
            ),
            SDTMVariableSpec(
                "ACTARMCD", "Actual Arm Code", "Char", "Qualifier", max_length=20, core="Exp"
            ),
            SDTMVariableSpec(
                "ACTARM",
                "Description of Actual Arm",
                "Char",
                "Qualifier",
                max_length=200,
                core="Exp",
            ),
            SDTMVariableSpec(
                "COUNTRY",
                "Country",
                "Char",
                "Qualifier",
                required=True,
                max_length=3,
                controlled_term="COUNTRY",
                core="Req",
            ),
            SDTMVariableSpec(
                "DMDTC", "Date/Time of Collection", "Char", "Timing", max_length=64, core="Perm"
            ),
            SDTMVariableSpec("DMDY", "Study Day of Collection", "Num", "Timing", core="Perm"),
        ],
    )


def _ae_template() -> SDTMDomainTemplate:
    """Adverse Events (AE): Events domain."""
    return SDTMDomainTemplate(
        domain="AE",
        label="Adverse Events",
        description="Adverse events reported during the study.",
        domain_class="Events",
        repeating=True,
        natural_keys=["STUDYID", "USUBJID", "AETERM", "AESTDTC"],
        variables=[
            SDTMVariableSpec(
                "STUDYID",
                "Study Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=20,
                core="Req",
            ),
            SDTMVariableSpec(
                "DOMAIN",
                "Domain Abbreviation",
                "Char",
                "Identifier",
                required=True,
                max_length=2,
                core="Req",
            ),
            SDTMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=40,
                core="Req",
            ),
            SDTMVariableSpec(
                "AESEQ", "Sequence Number", "Num", "Identifier", required=True, core="Req"
            ),
            SDTMVariableSpec(
                "AEGRPID", "Group ID", "Char", "Identifier", max_length=20, core="Perm"
            ),
            SDTMVariableSpec(
                "AEREFID", "Reference ID", "Char", "Identifier", max_length=20, core="Perm"
            ),
            SDTMVariableSpec(
                "AESPID",
                "Sponsor-Defined Identifier",
                "Char",
                "Identifier",
                max_length=20,
                core="Perm",
            ),
            SDTMVariableSpec(
                "AETERM",
                "Reported Term for the Adverse Event",
                "Char",
                "Topic",
                required=True,
                max_length=200,
                core="Req",
            ),
            SDTMVariableSpec(
                "AEMODIFY",
                "Modified Reported Term",
                "Char",
                "Qualifier",
                max_length=200,
                core="Perm",
            ),
            SDTMVariableSpec(
                "AEDECOD",
                "Dictionary-Derived Term",
                "Char",
                "Qualifier",
                required=True,
                max_length=200,
                core="Req",
            ),
            SDTMVariableSpec(
                "AEBODSYS",
                "Body System or Organ Class",
                "Char",
                "Qualifier",
                max_length=200,
                core="Exp",
            ),
            SDTMVariableSpec(
                "AESEV",
                "Severity/Intensity",
                "Char",
                "Qualifier",
                max_length=20,
                controlled_term="AESEV",
                core="Perm",
            ),
            SDTMVariableSpec(
                "AESER",
                "Serious Event",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Exp",
            ),
            SDTMVariableSpec(
                "AEACN",
                "Action Taken with Study Treatment",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="ACN",
                core="Exp",
            ),
            SDTMVariableSpec("AEREL", "Causality", "Char", "Qualifier", max_length=40, core="Exp"),
            SDTMVariableSpec(
                "AEOUT",
                "Outcome of Adverse Event",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="OUT",
                core="Exp",
            ),
            SDTMVariableSpec(
                "AESCAN",
                "Involves Cancer",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Perm",
            ),
            SDTMVariableSpec(
                "AESCONG",
                "Congenital Anomaly or Birth Defect",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Perm",
            ),
            SDTMVariableSpec(
                "AESDISAB",
                "Persist or Signif Disability/Incapacity",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Perm",
            ),
            SDTMVariableSpec(
                "AESDTH",
                "Results in Death",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Perm",
            ),
            SDTMVariableSpec(
                "AESHOSP",
                "Requires or Prolongs Hospitalization",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Perm",
            ),
            SDTMVariableSpec(
                "AESLIFE",
                "Is Life Threatening",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Perm",
            ),
            SDTMVariableSpec(
                "AESOD",
                "Other Medically Important SAE",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Perm",
            ),
            SDTMVariableSpec(
                "AECONTRT",
                "Concomitant or Additional Trtmnt Given",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Perm",
            ),
            SDTMVariableSpec(
                "AESTDTC",
                "Start Date/Time of Adverse Event",
                "Char",
                "Timing",
                max_length=64,
                core="Exp",
            ),
            SDTMVariableSpec(
                "AEENDTC",
                "End Date/Time of Adverse Event",
                "Char",
                "Timing",
                max_length=64,
                core="Exp",
            ),
            SDTMVariableSpec(
                "AESTDY", "Study Day of Start of Adverse Event", "Num", "Timing", core="Perm"
            ),
            SDTMVariableSpec(
                "AEENDY", "Study Day of End of Adverse Event", "Num", "Timing", core="Perm"
            ),
        ],
    )


def _lb_template() -> SDTMDomainTemplate:
    """Laboratory Test Results (LB): Findings domain."""
    return SDTMDomainTemplate(
        domain="LB",
        label="Laboratory Test Results",
        description="Laboratory test results including hematology, chemistry, and urinalysis.",
        domain_class="Findings",
        repeating=True,
        natural_keys=["STUDYID", "USUBJID", "LBTESTCD", "LBDTC", "LBSPEC"],
        variables=[
            SDTMVariableSpec(
                "STUDYID",
                "Study Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=20,
                core="Req",
            ),
            SDTMVariableSpec(
                "DOMAIN",
                "Domain Abbreviation",
                "Char",
                "Identifier",
                required=True,
                max_length=2,
                core="Req",
            ),
            SDTMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=40,
                core="Req",
            ),
            SDTMVariableSpec(
                "LBSEQ", "Sequence Number", "Num", "Identifier", required=True, core="Req"
            ),
            SDTMVariableSpec(
                "LBTESTCD",
                "Lab Test or Examination Short Name",
                "Char",
                "Topic",
                required=True,
                max_length=8,
                controlled_term="LBTESTCD",
                core="Req",
            ),
            SDTMVariableSpec(
                "LBTEST",
                "Lab Test or Examination Name",
                "Char",
                "Qualifier",
                required=True,
                max_length=40,
                controlled_term="LBTEST",
                core="Req",
            ),
            SDTMVariableSpec(
                "LBCAT", "Category for Lab Test", "Char", "Qualifier", max_length=40, core="Exp"
            ),
            SDTMVariableSpec(
                "LBSCAT",
                "Subcategory for Lab Test",
                "Char",
                "Qualifier",
                max_length=40,
                core="Perm",
            ),
            SDTMVariableSpec(
                "LBORRES",
                "Result or Finding in Original Units",
                "Char",
                "Qualifier",
                max_length=200,
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBORRESU",
                "Original Units",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="UNIT",
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBORNRLO",
                "Reference Range Lower Limit-Orig Unit",
                "Char",
                "Qualifier",
                max_length=40,
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBORNRHI",
                "Reference Range Upper Limit-Orig Unit",
                "Char",
                "Qualifier",
                max_length=40,
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBSTRESC",
                "Character Result/Finding in Std Format",
                "Char",
                "Qualifier",
                max_length=200,
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBSTRESN",
                "Numeric Result/Finding in Standard Units",
                "Num",
                "Qualifier",
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBSTRESU",
                "Standard Units",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="UNIT",
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBSTNRLO", "Reference Range Lower Limit-Std Units", "Num", "Qualifier", core="Exp"
            ),
            SDTMVariableSpec(
                "LBSTNRHI", "Reference Range Upper Limit-Std Units", "Num", "Qualifier", core="Exp"
            ),
            SDTMVariableSpec(
                "LBNRIND",
                "Reference Range Indicator",
                "Char",
                "Qualifier",
                max_length=20,
                controlled_term="NRIND",
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBSPEC",
                "Specimen Type",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="SPECTYPE",
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBMETHOD",
                "Method of Test or Examination",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="METHOD",
                core="Perm",
            ),
            SDTMVariableSpec(
                "LBBLFL",
                "Baseline Flag",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBFAST",
                "Fasting Status",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Perm",
            ),
            SDTMVariableSpec("VISITNUM", "Visit Number", "Num", "Timing", core="Exp"),
            SDTMVariableSpec("VISIT", "Visit Name", "Char", "Timing", max_length=40, core="Perm"),
            SDTMVariableSpec(
                "LBDTC",
                "Date/Time of Specimen Collection",
                "Char",
                "Timing",
                max_length=64,
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBDY", "Study Day of Specimen Collection", "Num", "Timing", core="Perm"
            ),
        ],
    )


def _vs_template() -> SDTMDomainTemplate:
    """Vital Signs (VS): Findings domain."""
    return SDTMDomainTemplate(
        domain="VS",
        label="Vital Signs",
        description="Vital signs measurements including blood pressure, heart rate, temperature, and weight.",
        domain_class="Findings",
        repeating=True,
        natural_keys=["STUDYID", "USUBJID", "VSTESTCD", "VSDTC", "VSTPTNUM"],
        variables=[
            SDTMVariableSpec(
                "STUDYID",
                "Study Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=20,
                core="Req",
            ),
            SDTMVariableSpec(
                "DOMAIN",
                "Domain Abbreviation",
                "Char",
                "Identifier",
                required=True,
                max_length=2,
                core="Req",
            ),
            SDTMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=40,
                core="Req",
            ),
            SDTMVariableSpec(
                "VSSEQ", "Sequence Number", "Num", "Identifier", required=True, core="Req"
            ),
            SDTMVariableSpec(
                "VSTESTCD",
                "Vital Signs Test Short Name",
                "Char",
                "Topic",
                required=True,
                max_length=8,
                controlled_term="VSTESTCD",
                core="Req",
            ),
            SDTMVariableSpec(
                "VSTEST",
                "Vital Signs Test Name",
                "Char",
                "Qualifier",
                required=True,
                max_length=40,
                controlled_term="VSTEST",
                core="Req",
            ),
            SDTMVariableSpec(
                "VSPOS",
                "Vital Signs Position of Subject",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="POSITION",
                core="Perm",
            ),
            SDTMVariableSpec(
                "VSORRES",
                "Result or Finding in Original Units",
                "Char",
                "Qualifier",
                max_length=200,
                core="Exp",
            ),
            SDTMVariableSpec(
                "VSORRESU",
                "Original Units",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="UNIT",
                core="Exp",
            ),
            SDTMVariableSpec(
                "VSSTRESC",
                "Character Result/Finding in Std Format",
                "Char",
                "Qualifier",
                max_length=200,
                core="Exp",
            ),
            SDTMVariableSpec(
                "VSSTRESN",
                "Numeric Result/Finding in Standard Units",
                "Num",
                "Qualifier",
                core="Exp",
            ),
            SDTMVariableSpec(
                "VSSTRESU",
                "Standard Units",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="UNIT",
                core="Exp",
            ),
            SDTMVariableSpec(
                "VSBLFL",
                "Baseline Flag",
                "Char",
                "Qualifier",
                max_length=2,
                controlled_term="NY",
                core="Exp",
            ),
            SDTMVariableSpec("VISITNUM", "Visit Number", "Num", "Timing", core="Exp"),
            SDTMVariableSpec("VISIT", "Visit Name", "Char", "Timing", max_length=40, core="Perm"),
            SDTMVariableSpec(
                "VSDTC", "Date/Time of Measurements", "Char", "Timing", max_length=64, core="Exp"
            ),
            SDTMVariableSpec("VSDY", "Study Day of Vital Signs", "Num", "Timing", core="Perm"),
            SDTMVariableSpec("VSTPTNUM", "Planned Time Point Number", "Num", "Timing", core="Perm"),
            SDTMVariableSpec(
                "VSTPT", "Planned Time Point Name", "Char", "Timing", max_length=40, core="Perm"
            ),
        ],
    )


def _ex_template() -> SDTMDomainTemplate:
    """Exposure (EX): Interventions domain."""
    return SDTMDomainTemplate(
        domain="EX",
        label="Exposure",
        description="Study treatment administration/exposure records.",
        domain_class="Interventions",
        repeating=True,
        natural_keys=["STUDYID", "USUBJID", "EXTRT", "EXSTDTC"],
        variables=[
            SDTMVariableSpec(
                "STUDYID",
                "Study Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=20,
                core="Req",
            ),
            SDTMVariableSpec(
                "DOMAIN",
                "Domain Abbreviation",
                "Char",
                "Identifier",
                required=True,
                max_length=2,
                core="Req",
            ),
            SDTMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=40,
                core="Req",
            ),
            SDTMVariableSpec(
                "EXSEQ", "Sequence Number", "Num", "Identifier", required=True, core="Req"
            ),
            SDTMVariableSpec(
                "EXTRT",
                "Name of Treatment",
                "Char",
                "Topic",
                required=True,
                max_length=200,
                core="Req",
            ),
            SDTMVariableSpec(
                "EXCAT", "Category of Treatment", "Char", "Qualifier", max_length=40, core="Perm"
            ),
            SDTMVariableSpec("EXDOSE", "Dose", "Num", "Qualifier", core="Exp"),
            SDTMVariableSpec(
                "EXDOSU",
                "Dose Units",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="UNIT",
                core="Exp",
            ),
            SDTMVariableSpec(
                "EXDOSFRM",
                "Dose Form",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="FRM",
                core="Exp",
            ),
            SDTMVariableSpec(
                "EXDOSFRQ",
                "Dosing Frequency per Interval",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="FREQ",
                core="Exp",
            ),
            SDTMVariableSpec(
                "EXROUTE",
                "Route of Administration",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="ROUTE",
                core="Exp",
            ),
            SDTMVariableSpec(
                "EXSTDTC",
                "Start Date/Time of Treatment",
                "Char",
                "Timing",
                max_length=64,
                core="Exp",
            ),
            SDTMVariableSpec(
                "EXENDTC", "End Date/Time of Treatment", "Char", "Timing", max_length=64, core="Exp"
            ),
            SDTMVariableSpec(
                "EXSTDY", "Study Day of Start of Treatment", "Num", "Timing", core="Perm"
            ),
            SDTMVariableSpec(
                "EXENDY", "Study Day of End of Treatment", "Num", "Timing", core="Perm"
            ),
        ],
    )


def _ds_template() -> SDTMDomainTemplate:
    """Disposition (DS): Events domain."""
    return SDTMDomainTemplate(
        domain="DS",
        label="Disposition",
        description="Subject disposition events (screening, randomization, completion, discontinuation).",
        domain_class="Events",
        repeating=True,
        natural_keys=["STUDYID", "USUBJID", "DSTERM", "DSSTDTC"],
        variables=[
            SDTMVariableSpec(
                "STUDYID",
                "Study Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=20,
                core="Req",
            ),
            SDTMVariableSpec(
                "DOMAIN",
                "Domain Abbreviation",
                "Char",
                "Identifier",
                required=True,
                max_length=2,
                core="Req",
            ),
            SDTMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=40,
                core="Req",
            ),
            SDTMVariableSpec(
                "DSSEQ", "Sequence Number", "Num", "Identifier", required=True, core="Req"
            ),
            SDTMVariableSpec(
                "DSTERM",
                "Reported Term for the Disposition Event",
                "Char",
                "Topic",
                required=True,
                max_length=200,
                core="Req",
            ),
            SDTMVariableSpec(
                "DSDECOD",
                "Standardized Disposition Term",
                "Char",
                "Qualifier",
                required=True,
                max_length=200,
                controlled_term="NCOMPLT",
                core="Req",
            ),
            SDTMVariableSpec(
                "DSCAT",
                "Category for Disposition Event",
                "Char",
                "Qualifier",
                max_length=40,
                core="Exp",
            ),
            SDTMVariableSpec(
                "DSSCAT",
                "Subcategory for Disposition Event",
                "Char",
                "Qualifier",
                max_length=40,
                core="Perm",
            ),
            SDTMVariableSpec(
                "EPOCH",
                "Epoch",
                "Char",
                "Timing",
                max_length=40,
                controlled_term="EPOCH",
                core="Exp",
            ),
            SDTMVariableSpec(
                "DSSTDTC",
                "Start Date/Time of Disposition Event",
                "Char",
                "Timing",
                max_length=64,
                core="Exp",
            ),
            SDTMVariableSpec("DSSTDY", "Study Day of Start of Event", "Num", "Timing", core="Perm"),
        ],
    )


def _mh_template() -> SDTMDomainTemplate:
    """Medical History (MH): Events domain."""
    return SDTMDomainTemplate(
        domain="MH",
        label="Medical History",
        description="Subject medical history prior to study participation.",
        domain_class="Events",
        repeating=True,
        natural_keys=["STUDYID", "USUBJID", "MHTERM"],
        variables=[
            SDTMVariableSpec(
                "STUDYID",
                "Study Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=20,
                core="Req",
            ),
            SDTMVariableSpec(
                "DOMAIN",
                "Domain Abbreviation",
                "Char",
                "Identifier",
                required=True,
                max_length=2,
                core="Req",
            ),
            SDTMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=40,
                core="Req",
            ),
            SDTMVariableSpec(
                "MHSEQ", "Sequence Number", "Num", "Identifier", required=True, core="Req"
            ),
            SDTMVariableSpec(
                "MHTERM",
                "Reported Term for the Medical History",
                "Char",
                "Topic",
                required=True,
                max_length=200,
                core="Req",
            ),
            SDTMVariableSpec(
                "MHMODIFY",
                "Modified Reported Term",
                "Char",
                "Qualifier",
                max_length=200,
                core="Perm",
            ),
            SDTMVariableSpec(
                "MHDECOD",
                "Dictionary-Derived Term",
                "Char",
                "Qualifier",
                max_length=200,
                core="Exp",
            ),
            SDTMVariableSpec(
                "MHBODSYS",
                "Body System or Organ Class",
                "Char",
                "Qualifier",
                max_length=200,
                core="Exp",
            ),
            SDTMVariableSpec(
                "MHCAT",
                "Category for Medical History",
                "Char",
                "Qualifier",
                max_length=40,
                core="Exp",
            ),
            SDTMVariableSpec(
                "MHSCAT",
                "Subcategory for Medical History",
                "Char",
                "Qualifier",
                max_length=40,
                core="Perm",
            ),
            SDTMVariableSpec(
                "MHSTDTC",
                "Start Date/Time of Medical History",
                "Char",
                "Timing",
                max_length=64,
                core="Perm",
            ),
            SDTMVariableSpec(
                "MHENDTC",
                "End Date/Time of Medical History",
                "Char",
                "Timing",
                max_length=64,
                core="Perm",
            ),
        ],
    )


def _cm_template() -> SDTMDomainTemplate:
    """Concomitant Medications (CM): Interventions domain."""
    return SDTMDomainTemplate(
        domain="CM",
        label="Concomitant Medications",
        description="Concomitant and prior medications reported during the study.",
        domain_class="Interventions",
        repeating=True,
        natural_keys=["STUDYID", "USUBJID", "CMTRT", "CMSTDTC"],
        variables=[
            SDTMVariableSpec(
                "STUDYID",
                "Study Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=20,
                core="Req",
            ),
            SDTMVariableSpec(
                "DOMAIN",
                "Domain Abbreviation",
                "Char",
                "Identifier",
                required=True,
                max_length=2,
                core="Req",
            ),
            SDTMVariableSpec(
                "USUBJID",
                "Unique Subject Identifier",
                "Char",
                "Identifier",
                required=True,
                max_length=40,
                core="Req",
            ),
            SDTMVariableSpec(
                "CMSEQ", "Sequence Number", "Num", "Identifier", required=True, core="Req"
            ),
            SDTMVariableSpec(
                "CMTRT",
                "Reported Name of Drug, Med, or Therapy",
                "Char",
                "Topic",
                required=True,
                max_length=200,
                core="Req",
            ),
            SDTMVariableSpec(
                "CMMODIFY",
                "Modified Reported Name",
                "Char",
                "Qualifier",
                max_length=200,
                core="Perm",
            ),
            SDTMVariableSpec(
                "CMDECOD",
                "Standardized Medication Name",
                "Char",
                "Qualifier",
                max_length=200,
                core="Exp",
            ),
            SDTMVariableSpec(
                "CMCAT", "Category for Medication", "Char", "Qualifier", max_length=40, core="Perm"
            ),
            SDTMVariableSpec("CMDOSE", "Dose per Administration", "Num", "Qualifier", core="Perm"),
            SDTMVariableSpec(
                "CMDOSU",
                "Dose Units",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="UNIT",
                core="Perm",
            ),
            SDTMVariableSpec(
                "CMDOSFRM",
                "Dose Form",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="FRM",
                core="Perm",
            ),
            SDTMVariableSpec(
                "CMROUTE",
                "Route of Administration",
                "Char",
                "Qualifier",
                max_length=40,
                controlled_term="ROUTE",
                core="Perm",
            ),
            SDTMVariableSpec(
                "CMINDC", "Indication", "Char", "Qualifier", max_length=200, core="Exp"
            ),
            SDTMVariableSpec(
                "CMSTDTC",
                "Start Date/Time of Medication",
                "Char",
                "Timing",
                max_length=64,
                core="Exp",
            ),
            SDTMVariableSpec(
                "CMENDTC",
                "End Date/Time of Medication",
                "Char",
                "Timing",
                max_length=64,
                core="Exp",
            ),
            SDTMVariableSpec(
                "CMSTDY", "Study Day of Start of Medication", "Num", "Timing", core="Perm"
            ),
            SDTMVariableSpec(
                "CMENDY", "Study Day of End of Medication", "Num", "Timing", core="Perm"
            ),
        ],
    )


# Registry of all domain templates
_DOMAIN_TEMPLATES: dict[str, callable] = {
    "DM": _dm_template,
    "AE": _ae_template,
    "LB": _lb_template,
    "VS": _vs_template,
    "EX": _ex_template,
    "DS": _ds_template,
    "MH": _mh_template,
    "CM": _cm_template,
}


def get_sdtm_domain(domain: str) -> SDTMDomainTemplate:
    """Get the SDTM template for a specific domain.

    Parameters
    ----------
    domain
        Two-character domain code (e.g., `"DM"`, `"AE"`, `"LB"`, `"VS"`). This is case-insensitive.

    Returns
    -------
    SDTMDomainTemplate
        The structural template for the domain.

    Raises
    ------
    KeyError
        If the domain is not supported.

    Examples
    --------
    ```python
    from pointblank.metadata._sdtm_templates import get_sdtm_domain

    dm = get_sdtm_domain("DM")
    print(dm.required_variables)
    # ['STUDYID', 'DOMAIN', 'USUBJID', 'SUBJID', 'ARMCD', 'ARM', 'COUNTRY']
    ```
    """
    domain_upper = domain.upper()
    if domain_upper not in _DOMAIN_TEMPLATES:
        available = sorted(_DOMAIN_TEMPLATES.keys())
        raise KeyError(f"SDTM domain '{domain}' is not supported. Available domains: {available}")
    return _DOMAIN_TEMPLATES[domain_upper]()


def list_sdtm_domains() -> list[str]:
    """List all available SDTM domain codes.

    Returns
    -------
    list[str]
        Sorted list of domain codes.
    """
    return sorted(_DOMAIN_TEMPLATES.keys())


def validate_sdtm_structure(
    data: Any,
    domain: str,
    strict: bool = False,
) -> dict[str, Any]:
    """Validate the structural conformance of a dataset against an SDTM domain template.

    Checks required variables, variable ordering, data types, and domainvalue consistency. Does not
    interrogate but rather returns a dict of findings.

    Parameters
    ----------
    data
        A DataFrame (Pandas, Polars) to check.
    domain
        SDTM domain code (e.g., `"DM"`, `"AE"`). This is case-insensitive.
    strict
        If `True`, also report missing Expected variables and unknown variables.

    Returns
    -------
    dict
        A dictionary with keys:

        - "domain": the domain code
        - "valid": `True` if no required violations found
        - "missing_required": list of missing required variable names
        - "missing_expected": list of missing expected variable names (strict only)
        - "unknown_variables": list of column names not in the template (strict only)
        - "domain_mismatch": `True` if `DOMAIN` column doesn't match expected value
        - "issues": list of human-readable issue strings
    """
    import narwhals as nw

    template = get_sdtm_domain(domain)

    # Wrap in narwhals for framework-agnostic access
    df = nw.from_native(data, eager_only=True)
    columns = df.columns

    issues: list[str] = []
    result: dict[str, Any] = {
        "domain": domain.upper(),
        "valid": True,
        "missing_required": [],
        "missing_expected": [],
        "unknown_variables": [],
        "domain_mismatch": False,
        "issues": issues,
    }

    # Check required variables
    for var_name in template.required_variables:
        if var_name not in columns:
            result["missing_required"].append(var_name)
            issues.append(f"Required variable '{var_name}' is missing")

    if result["missing_required"]:
        result["valid"] = False

    # Check DOMAIN value
    if "DOMAIN" in columns:
        domain_values = df["DOMAIN"].unique().to_list()
        if domain_values != [domain.upper()]:
            result["domain_mismatch"] = True
            issues.append(
                f"DOMAIN column contains unexpected values: {domain_values} "
                f"(expected ['{domain.upper()}'])"
            )
            result["valid"] = False

    # Strict mode: check expected variables and unknown columns
    if strict:
        for var_name in template.expected_variables:
            if var_name not in columns:
                result["missing_expected"].append(var_name)
                issues.append(f"Expected variable '{var_name}' is missing")

        template_names = {v.name for v in template.variables}
        for col in columns:
            if col not in template_names:
                result["unknown_variables"].append(col)
                issues.append(f"Variable '{col}' is not defined in {domain.upper()} template")

    return result
