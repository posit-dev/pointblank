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


@dataclass
class VariableMetadata:
    """Metadata for a single variable/column, as imported from an external standard.

    Parameters
    ----------
    name
        Variable/column name.
    label
        Human-readable label.
    description
        Longer description of the variable.
    dtype
        Data type (mapped to Narwhals/Polars type names).
    role
        Variable role (e.g., `"identifier"`, `"measure"`, `"classifier"`).
    required
        Whether the variable must be non-null.
    unique
        Whether all values must be distinct.
    min_val
        Minimum allowed value (inclusive).
    max_val
        Maximum allowed value (inclusive).
    min_length
        Minimum string length.
    max_length
        Maximum string length.
    pattern
        Regex pattern that values must match.
    allowed_values
        Explicit list of allowed values.
    codelist_ref
        Reference to a named codelist.
    display_format
        Display format from source system (e.g., `"F8.2"`, `"DATETIME20."`).
    value_labels
        Value-to-label mapping (e.g., `{1: "Male", 2: "Female"}`).
    missing_values
        Sentinel values representing missingness (e.g., `-99`, `".A"`, `""`).
    missing_value_labels
        Labels for missing value sentinels (e.g., `"Refused"`, `"Not Applicable"`).
    origin
        How the variable was created (`"CRF"`, `"Derived"`, `"Assigned"`).
    computational_method
        Derivation algorithm for computed variables.
    controlled_term
        CDISC controlled terminology reference.
    significant_digits
        Number of significant digits.
    cdisc_domain
        CDISC domain code (e.g., `"DM"`, `"AE"`, `"LB"`, `"VS"`).
    cdisc_role
        CDISC variable role (`"Identifier"`, `"Topic"`, `"Timing"`, `"Qualifier"`, `"Rule"`).
    adam_derivation
        ADaM derivation algorithm description.
    traceability_ref
        ADaM traceability reference back to SDTM source.
    unit
        Unit of measurement (e.g., `"kg"`, `"mmHg"`, `"years"`).
    unit_system
        Unit system (e.g., `"SI"`, `"imperial"`, `"UDUNITS"`).
    """

    name: str
    label: str | None = None
    description: str | None = None
    dtype: str | None = None
    role: str | None = None

    # Constraints (map directly to validation steps)
    required: bool = False
    unique: bool = False
    min_val: float | None = None
    max_val: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None
    allowed_values: list[Any] | None = None
    codelist_ref: str | None = None

    # Statistical package metadata
    display_format: str | None = None
    value_labels: dict[Any, str] | None = None
    missing_values: list[Any] | None = None
    missing_value_labels: dict[Any, str] | None = None

    # Clinical/regulatory (CDISC)
    origin: str | None = None
    computational_method: str | None = None
    controlled_term: str | None = None
    significant_digits: int | None = None
    cdisc_domain: str | None = None
    cdisc_role: str | None = None
    adam_derivation: str | None = None
    traceability_ref: str | None = None

    # Units
    unit: str | None = None
    unit_system: str | None = None


@dataclass
class MetadataImport:
    """Parsed metadata from an external standard.

    Contains variable definitions, value labels, missing value codes, controlled terminologies, and
    dataset-level metadata: all mapped to Pointblank concepts.

    Parameters
    ----------
    source_format
        The format this metadata was imported from (e.g., `"spss"`, `"xpt"`, `"stata"`).
    source_path
        Path to the source file, if imported from a file.
    source_version
        Version of the source format/standard.
    dataset_name
        Name of the dataset.
    dataset_label
        Human-readable label for the dataset.
    dataset_description
        Description of the dataset.
    creation_date
        When the dataset/metadata was created.
    study_id
        Study identifier (for clinical data).
    domain
        Domain identifier (e.g., `"DM"`, `"AE"` for CDISC).
    variables
        List of variable metadata definitions.
    codelists
        Named codelists (controlled terminologies).
    missing_value_codes
        Named missing value code definitions.
    """

    source_format: str
    source_path: str | None = None
    source_version: str | None = None

    # Dataset-level metadata
    dataset_name: str | None = None
    dataset_label: str | None = None
    dataset_description: str | None = None
    creation_date: str | None = None
    study_id: str | None = None
    domain: str | None = None

    # Variable-level metadata
    variables: list[VariableMetadata] = dataclass_field(default_factory=list)

    # Controlled terminologies / codelists
    codelists: dict[str, Codelist] = dataclass_field(default_factory=dict)

    # Missing value definitions
    missing_value_codes: dict[str, list[MissingValueCode]] = dataclass_field(default_factory=dict)

    def to_schema(self) -> Schema:
        """Convert imported metadata to a Pointblank `Schema` with `Field` objects.

        Maps variable metadata to appropriate `Field` types with constraints (min/max, allowed
        values, nullable, etc.).

        Returns
        -------
        Schema
            A Pointblank `Schema` object with typed fields.
        """
        from pointblank.metadata._convert import _metadata_to_schema

        return _metadata_to_schema(self)

