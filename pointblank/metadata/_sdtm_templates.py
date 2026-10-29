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
    """The specification of one variable in an SDTM domain template.

    An [`SDTMDomainTemplate`](`pointblank.SDTMDomainTemplate`) holds an `SDTMVariableSpec` for each
    variable of its domain, in its `variables` list. The specification gives what the SDTM
    Implementation Guide says about the variable: its label, its type, its role in the domain, its
    core designation, and the controlled terminology for its values.
    [`validate_sdtm()`](`pointblank.validate_sdtm`) uses `required`, `dtype`, `max_length` and
    `role` to decide which checks to make.

    Parameters
    ----------
    name
        The variable name, such as `"STUDYID"` or `"AESTDTC"`.
    label
        The variable label, such as `"Study Identifier"`.
    dtype
        The SDTM data type: `"Char"` for character variables or `"Num"` for numeric ones.
    role
        The variable's SDTM role. The templates use `"Identifier"` (variables that identify the
        study, subject, domain or record), `"Topic"` (the focus of the observation, such as
        `AETERM`), `"Qualifier"` (variables that describe the observation) and `"Timing"` (dates,
        times and study days).
    required
        Whether the variable is required: it must be in the dataset and can't be null. In the
        templates, this is `True` exactly when `core` is `"Req"`. The default is `False`.
    max_length
        The maximum length of the values of a character variable. The default is `None` (no
        maximum, as for numeric variables).
    controlled_term
        The name of the controlled terminology codelist for the variable's values, such as `"SEX"`
        or `"NY"`. The default is `None`.
    core
        The core designation: `"Req"` (required), `"Exp"` (expected: in the dataset, but values may
        be null) or `"Perm"` (permissible: included when there is data for it). The default is
        `"Perm"`.

    Examples
    --------
    Specifications are usually taken from a template. Here is the one for `AESER` in the Adverse
    Events (AE) template:

    ```{python}
    import pointblank as pb

    aeser = pb.get_sdtm_domain("AE").get_variable("AESER")

    aeser
    ```

    `AESER` (whether the adverse event was serious) is an expected character variable of at most
    two characters, with values from the `NY` (No Yes Response) codelist.

    See Also
    --------
    SDTMDomainTemplate : The template that holds the specifications of a domain's variables.
    ADaMVariableSpec : The equivalent for ADaM dataset templates.
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
    """The structural template for an SDTM domain.

    An `SDTMDomainTemplate` describes the structure of one domain of the CDISC Study Data
    Tabulation Model (SDTM): its label and observation class, whether it has many records per
    subject, the variables that form its natural key, and the specifications of its variables (as
    [`SDTMVariableSpec`](`pointblank.SDTMVariableSpec`) objects, in the SDTM order). Templates are
    obtained with [`get_sdtm_domain()`](`pointblank.get_sdtm_domain`).

    Parameters
    ----------
    domain
        The two-letter domain code, such as `"DM"`, `"AE"` or `"LB"`.
    label
        The domain label, such as `"Demographics"` or `"Adverse Events"`.
    description
        A short description of what the domain holds.
    domain_class
        The SDTM observation class: `"Special Purpose"`, `"Events"`, `"Interventions"` or
        `"Findings"`.
    repeating
        Whether the domain can have more than one record per subject. Only DM, which has one record
        per subject, isn't repeating.
    variables
        The specifications of the domain's variables, in order. The default is an empty list.
    natural_keys
        The names of the variables that together identify a record. The default is an empty list.

    Examples
    --------
    Let's look at the template for the Adverse Events (AE) domain:

    ```{python}
    import pointblank as pb

    ae = pb.get_sdtm_domain("AE")

    print(ae.label, "|", ae.domain_class, "| repeating:", ae.repeating)
    print(ae.description)
    print(len(ae.variables), "variables")
    ```

    The variables are in the `variables` list. Here are the first eight, with their type, role and
    core designation:

    ```{python}
    for var in ae.variables[:8]:
        print(f"{var.name:9}{var.dtype:5}{var.role:12}{var.core}")
    ```

    The template's properties collect the names of the variables with a given core designation or
    role, and [`get_variable()`](`pointblank.SDTMDomainTemplate.get_variable`) gets the
    specification of one variable.

    See Also
    --------
    get_sdtm_domain : Get the template for a domain.
    SDTMVariableSpec : The specification of a variable in the template.
    ADaMDatasetTemplate : The equivalent for ADaM datasets.
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
        """The names of the required variables, in order.

        Required variables (core designation `"Req"`) must be in every dataset of the domain and
        can't be null. [`validate_sdtm()`](`pointblank.validate_sdtm`) adds a null check for each
        one, and [`validate_sdtm_structure()`](`pointblank.validate_sdtm_structure`) reports the
        ones that are missing.

        Examples
        --------
        ```{python}
        import pointblank as pb

        pb.get_sdtm_domain("DM").required_variables
        ```

        These are the variables that every Demographics (DM) dataset must have. This is a property,
        so it's used without parentheses.
        """
        return [v.name for v in self.variables if v.required]

    @property
    def expected_variables(self) -> list[str]:
        """The names of the expected variables, in order.

        Expected variables (core designation `"Exp"`) should be in every dataset of the domain, but
        their values may be null when the data wasn't collected or doesn't apply.
        [`validate_sdtm_structure()`](`pointblank.validate_sdtm_structure`) reports the ones that
        are missing when `strict=True`.

        Examples
        --------
        ```{python}
        import pointblank as pb

        pb.get_sdtm_domain("DM").expected_variables
        ```

        For Demographics (DM), these include the reference dates (such as `RFSTDTC`), age, race and
        the actual arm. This is a property, so it's used without parentheses.
        """
        return [v.name for v in self.variables if v.core == "Exp"]

    @property
    def identifier_variables(self) -> list[str]:
        """The names of the variables with the Identifier role, in order.

        Identifier variables identify the study, the subject, the domain and the record, as with
        `STUDYID`, `USUBJID`, `DOMAIN` and the sequence number.

        Examples
        --------
        ```{python}
        import pointblank as pb

        pb.get_sdtm_domain("AE").identifier_variables
        ```

        Besides the identifiers that all domains share, the Adverse Events (AE) template has the
        sequence number `AESEQ` and the optional group, reference and sponsor identifiers. This is
        a property, so it's used without parentheses.
        """
        return [v.name for v in self.variables if v.role == "Identifier"]

    def get_variable(self, name: str) -> SDTMVariableSpec | None:
        """Get the specification of a variable.

        Parameters
        ----------
        name
            The variable name. This is case-sensitive, and SDTM variable names are in upper case.

        Returns
        -------
        SDTMVariableSpec | None
            The specification of the variable, or `None` if the template has no variable with that
            name.

        Examples
        --------
        ```{python}
        import pointblank as pb

        dm = pb.get_sdtm_domain("DM")

        dm.get_variable("AGE")
        ```

        `AGE` is an expected numeric variable, so it has no maximum length. A name that isn't in
        the template gives `None`:

        ```{python}
        print(dm.get_variable("age"))
        ```

        The lookup is case-sensitive, so the lower-case name doesn't match `AGE`.
        """
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
                "Reference Range Lower Limit in Orig Unit",
                "Char",
                "Qualifier",
                max_length=40,
                core="Exp",
            ),
            SDTMVariableSpec(
                "LBORNRHI",
                "Reference Range Upper Limit in Orig Unit",
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
            SDTMVariableSpec(
                "DSSTDY", "Study Day of Start of Disposition Event", "Num", "Timing", core="Perm"
            ),
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
                "Start Date/Time of Medical History Event",
                "Char",
                "Timing",
                max_length=64,
                core="Perm",
            ),
            SDTMVariableSpec(
                "MHENDTC",
                "End Date/Time of Medical History Event",
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
    """Get the template for an SDTM domain.

    Pointblank has templates for eight domains of the CDISC Study Data Tabulation Model (SDTM),
    based on the SDTM Implementation Guide (SDTMIG) 3.4. Each template is an
    [`SDTMDomainTemplate`](`pointblank.SDTMDomainTemplate`): the domain's label and observation
    class, the variables that form its natural key, and an
    [`SDTMVariableSpec`](`pointblank.SDTMVariableSpec`) for each of its variables, giving the
    variable's label, type, role, maximum length, controlled terminology and core designation. The
    templates are what [`validate_sdtm()`](`pointblank.validate_sdtm`),
    [`validate_sdtm_structure()`](`pointblank.validate_sdtm_structure`) and
    [`sdtm_to_metadata()`](`pointblank.sdtm_to_metadata`) work from.

    The templates cover the common variables of each domain rather than every variable in the
    SDTMIG. Each call returns a new template object, so changing one doesn't affect later calls.

    Parameters
    ----------
    domain
        The two-letter domain code, such as `"DM"`, `"AE"`, `"LB"` or `"VS"`. This is
        case-insensitive.

    Returns
    -------
    SDTMDomainTemplate
        The template for the domain.

    Raises
    ------
    KeyError
        If there is no template for the domain. The error message lists the domains that have one,
        as does [`list_sdtm_domains()`](`pointblank.list_sdtm_domains`).

    Examples
    --------
    Let's get the template for the Vital Signs (VS) domain:

    ```{python}
    import pointblank as pb

    vs = pb.get_sdtm_domain("vs")

    print(vs.label, "|", vs.domain_class, "| repeating:", vs.repeating)
    print(vs.natural_keys)
    ```

    VS is a Findings domain with many records per subject, one for each test at each time point,
    so its natural key includes the test code, the date-time and the time point number. The
    template's properties list the variables by their core designation:

    ```{python}
    print(vs.required_variables)
    print(vs.expected_variables)
    ```

    The required variables must be in every VS dataset and can't be null. The expected ones should
    be in the dataset but may be null, and the domain's other variables are permissible: they're
    included only when there is data for them.

    See Also
    --------
    list_sdtm_domains : List the domains that have a template.
    SDTMDomainTemplate : The template that is returned.
    validate_sdtm : Build a validation plan from an SDTM domain template.
    get_adam_dataset : The equivalent for ADaM datasets.
    """
    domain_upper = domain.upper()
    if domain_upper not in _DOMAIN_TEMPLATES:
        available = sorted(_DOMAIN_TEMPLATES.keys())
        raise KeyError(f"SDTM domain '{domain}' is not supported. Available domains: {available}")
    return _DOMAIN_TEMPLATES[domain_upper]()


def list_sdtm_domains() -> list[str]:
    """List the SDTM domains that have a template.

    These are the domain codes that [`get_sdtm_domain()`](`pointblank.get_sdtm_domain`),
    [`validate_sdtm()`](`pointblank.validate_sdtm`),
    [`validate_sdtm_structure()`](`pointblank.validate_sdtm_structure`) and
    [`sdtm_to_metadata()`](`pointblank.sdtm_to_metadata`) accept.

    Returns
    -------
    list[str]
        The domain codes, in alphabetical order.

    Examples
    --------
    Let's list the domains, together with the label and class of each one's template:

    ```{python}
    import pointblank as pb

    for code in pb.list_sdtm_domains():
        template = pb.get_sdtm_domain(code)
        print(f"{code}  {template.label:26}{template.domain_class}")
    ```

    The eight domains come from all four of the observation classes that the templates use:
    Special Purpose (DM), Events, Interventions and Findings.

    See Also
    --------
    get_sdtm_domain : Get the template for a domain.
    list_adam_datasets : The equivalent for ADaM datasets.
    """
    return sorted(_DOMAIN_TEMPLATES.keys())


def validate_sdtm_structure(
    data: Any,
    domain: str,
    strict: bool = False,
) -> dict[str, Any]:
    """Check a dataset's variables against an SDTM domain template.

    The `validate_sdtm_structure()` function is a quick check of a dataset's structure, run before
    (or instead of) a full validation with [`validate_sdtm()`](`pointblank.validate_sdtm`). It
    checks that every required variable of the domain is a column of the dataset, and that the
    `DOMAIN` column holds only the domain code. It returns the findings as a dictionary straight
    away, without building a [`Validate`](`pointblank.Validate`) plan.

    With `strict=True`, two more things are reported: the expected variables (core designation
    `"Exp"`) that are missing, and the columns that aren't variables in the template. These don't
    affect `"valid"`. SDTM expects an expected variable to be in the dataset even if all of its
    values are null, but a missing one is treated here as something to review rather than an
    error. A column that isn't in the template may still be a valid SDTM variable, since the
    templates only cover the common variables of each domain.

    The column order and the data types aren't checked.

    Parameters
    ----------
    data
        The dataset to check, as a Polars or Pandas DataFrame.
    domain
        The SDTM domain code, such as `"DM"` or `"AE"`. This is case-insensitive.
    strict
        Whether to also report missing expected variables and columns that aren't in the template.
        The default is `False`.

    Returns
    -------
    dict
        A dictionary with these keys:

        - `"domain"`: the domain code, in upper case.
        - `"valid"`: `True` if no required variable is missing and `DOMAIN` holds only the domain
        code.
        - `"missing_required"`: the names of the required variables that aren't columns of the
        data.
        - `"missing_expected"`: the names of the expected variables that aren't columns of the data
        (filled in only with `strict=True`).
        - `"unknown_variables"`: the columns that aren't variables of the domain (filled in only
        with `strict=True`).
        - `"domain_mismatch"`: `True` if the `DOMAIN` column has any value (including a null) other
        than the domain code.
        - `"issues"`: a message for each of the problems above.

    Raises
    ------
    KeyError
        If there is no template for the domain.

    Examples
    --------
    Let's check a small Adverse Events (AE) dataset. It has all of the required AE variables
    except `AEDECOD`, and its `DOMAIN` column has one value from another domain:

    ```{python}
    import pointblank as pb
    import polars as pl

    ae = pl.DataFrame(
        {
            "STUDYID": ["STUDY01"] * 3,
            "DOMAIN": ["AE", "AE", "CM"],
            "USUBJID": ["STUDY01-001", "STUDY01-001", "STUDY01-002"],
            "AESEQ": [1, 2, 1],
            "AETERM": ["HEADACHE", "NAUSEA", "RASH"],
            "AESEV": ["MILD", "MILD", "MODERATE"],
            "AEFLAG": ["Y", "N", "Y"],
        }
    )

    result = pb.validate_sdtm_structure(ae, domain="AE")

    print(result["valid"])
    for issue in result["issues"]:
        print(issue)
    ```

    The dataset isn't valid, and the issues say why. The extra column `AEFLAG` isn't mentioned,
    since that's only reported in strict mode:

    ```{python}
    result = pb.validate_sdtm_structure(ae, domain="AE", strict=True)

    print(result["missing_expected"])
    print(result["unknown_variables"])
    ```

    Strict mode adds the expected AE variables that the dataset doesn't have, such as `AESTDTC`,
    and the column that isn't an AE variable. `AESEV` is a permissible variable of the domain, so
    it isn't reported.

    See Also
    --------
    validate_sdtm : Build a validation plan from an SDTM domain template.
    get_sdtm_domain : Get the template that the dataset is checked against.
    validate_adam_structure : The equivalent for ADaM datasets.
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
        domain_values = df["DOMAIN"].unique(maintain_order=True).to_list()
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
