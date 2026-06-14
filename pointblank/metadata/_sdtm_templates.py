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


