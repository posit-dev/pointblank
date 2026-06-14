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


