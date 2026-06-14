from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pointblank.schema import Schema
    from pointblank.validate import Validate

__all__ = [
    "CodelistEntry",
    "Codelist",
    "MissingValueCode",
    "VariableMetadata",
    "MetadataImport",
    "MetadataPackage",
]


@dataclass
class CodelistEntry:
    """A single entry in a codelist (controlled terminology).

    Parameters
    ----------
    value
        The coded value.
    label
        Human-readable label for the value.
    description
        Extended description of this entry.
    synonyms
        Alternative terms for this entry.
    is_deprecated
        Whether this entry is deprecated.
    """

    value: Any
    label: str
    description: str | None = None
    synonyms: list[str] | None = None
    is_deprecated: bool = False


@dataclass
class Codelist:
    """A controlled terminology / value set from an external standard.

    Represents a set of permitted values from standards like CDISC controlled terminology, SPSS
    value labels, DDI code schemes, etc.

    Parameters
    ----------
    name
        Codelist identifier.
    codes
        List of codelist entries.
    label
        Human-readable name for the codelist.
    version
        Version of the terminology.
    source
        Where this codelist comes from (e.g., `"CDISC CT 2024-09"`).
    extensible
        Whether additional values beyond the codelist are allowed.
    """

    name: str
    codes: list[CodelistEntry] = dataclass_field(default_factory=list)
    label: str | None = None
    version: str | None = None
    source: str | None = None
    extensible: bool = False

    def to_set(self) -> list:
        """Get the list of valid values (for col_vals_in_set).

        Returns
        -------
        list
            All non-deprecated values in the codelist.
        """
        return [entry.value for entry in self.codes if not entry.is_deprecated]

    def to_dict(self) -> dict:
        """Get a value → label mapping.

        Returns
        -------
        dict
            Mapping of value to human-readable label.
        """
        return {entry.value: entry.label for entry in self.codes}

    def __len__(self) -> int:
        return len(self.codes)


@dataclass
class MissingValueCode:
    """A structured missing value definition from an external standard.

    In SPSS, SAS, and clinical data, missing values carry meaning (`REFUSED`, `NOT_APPLICABLE`,
    `NOT_ASKED`, etc.).

    Parameters
    ----------
    value
        The sentinel value (e.g., `-99`, `".A"`, `""`).
    label
        What this missing code means.
    category
        Category of missingness (e.g., `"system_missing"`, `"user_missing"`).
    reason
        Why data is missing.
    """

    value: Any
    label: str
    category: str | None = None
    reason: str | None = None


