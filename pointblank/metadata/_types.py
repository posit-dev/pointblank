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

    def to_validate(self, data: Any, **kwargs: Any) -> Validate:
        """Generate a `Validate` workflow from the imported metadata.

        Creates validation steps for all constraints found in the metadata: value ranges, allowed
        values, required fields, string lengths, etc.

        Parameters
        ----------
        data
            The DataFrame or table to validate.
        **kwargs
            Additional keyword arguments passed to the `Validate` constructor.

        Returns
        -------
        `Validate`
            A configured (but not yet interrogated) `Validate` object.
        """
        from pointblank.metadata._convert import _metadata_to_validate

        return _metadata_to_validate(self, data, **kwargs)

    def get_variable(self, name: str) -> VariableMetadata:
        """Get metadata for a specific variable by name.

        Parameters
        ----------
        name
            The variable name to look up.

        Returns
        -------
        VariableMetadata
            The metadata for the named variable.

        Raises
        ------
        KeyError
            If no variable with that name exists.
        """
        for var in self.variables:
            if var.name == name:
                return var
        raise KeyError(f"No variable named '{name}' in imported metadata")

    def get_codelist(self, name: str) -> Codelist:
        """Get a specific codelist by name.

        Parameters
        ----------
        name
            The codelist name or identifier.

        Returns
        -------
        Codelist
            The requested codelist.

        Raises
        ------
        KeyError
            If no codelist with that name exists.
        """
        if name not in self.codelists:
            raise KeyError(f"No codelist named '{name}'. Available: {list(self.codelists.keys())}")
        return self.codelists[name]

    @property
    def variable_names(self) -> list[str]:
        """Get the list of all variable names."""
        return [v.name for v in self.variables]

    def summary(self) -> str:
        """Return a human-readable summary of the imported metadata.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = []
        lines.append(f"Metadata Import ({self.source_format})")
        if self.source_path:
            lines.append(f"  Source: {self.source_path}")
        if self.dataset_name:
            lines.append(f"  Dataset: {self.dataset_name}")
        if self.dataset_label:
            lines.append(f"  Label: {self.dataset_label}")
        if self.domain:
            lines.append(f"  Domain: {self.domain}")

        lines.append(f"  Variables: {len(self.variables)}")
        lines.append(f"  Codelists: {len(self.codelists)}")

        # Show variable summary
        if self.variables:
            lines.append("")
            lines.append("  Variables:")
            for var in self.variables:
                dtype_str = f" ({var.dtype})" if var.dtype else ""
                label_str = f" — {var.label}" if var.label else ""
                constraints = []
                if var.required:
                    constraints.append("required")
                if var.unique:
                    constraints.append("unique")
                if var.min_val is not None or var.max_val is not None:
                    constraints.append(f"range=[{var.min_val}, {var.max_val}]")
                if var.allowed_values:
                    n = len(var.allowed_values)
                    constraints.append(f"{n} allowed values")
                if var.codelist_ref:
                    constraints.append(f"codelist={var.codelist_ref}")
                constraint_str = f" [{', '.join(constraints)}]" if constraints else ""
                lines.append(f"    {var.name}{dtype_str}{label_str}{constraint_str}")

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return (
            f"MetadataImport(source_format={self.source_format!r}, "
            f"variables={len(self.variables)}, "
            f"codelists={len(self.codelists)})"
        )

    def __len__(self) -> int:
        return len(self.variables)


@dataclass
class MetadataPackage:
    """A collection of `MetadataImport` objects from a multi-dataset source.

    Used for multi-domain CDISC studies, Frictionless Data Packages, etc.

    Parameters
    ----------
    name
        Package name/identifier.
    items
        Named `MetadataImport` objects.
    description
        Description of the package.
    version
        Package/study version.
    """

    name: str | None = None
    items: dict[str, MetadataImport] = dataclass_field(default_factory=dict)
    description: str | None = None
    version: str | None = None

    def __getitem__(self, key: str) -> MetadataImport:
        return self.items[key]

    def __contains__(self, key: str) -> bool:
        return key in self.items

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self):
        return iter(self.items)

    def keys(self):
        """Get the names of all datasets/domains."""
        return self.items.keys()

    def values(self):
        """Get all MetadataImport objects."""
        return self.items.values()

    def get_domain(self, name: str) -> MetadataImport:
        """Get metadata for a specific domain/dataset.

        Parameters
        ----------
        name
            Domain or dataset name (e.g., `"DM"`, `"AE"`).

        Returns
        -------
        MetadataImport
            The metadata for the named domain.

        Raises
        ------
        KeyError
            If no domain with that name exists.
        """
        if name not in self.items:
            raise KeyError(
                f"No domain/dataset named '{name}'. Available: {list(self.items.keys())}"
            )
        return self.items[name]

    def summary(self) -> str:
        """Return a human-readable summary of the package.

        Returns
        -------
        str
            Formatted summary string.
        """
        lines = []
        lines.append("Metadata Package")
        if self.name:
            lines.append(f"  Name: {self.name}")
        if self.description:
            lines.append(f"  Description: {self.description}")
        lines.append(f"  Datasets: {len(self.items)}")
        lines.append("")
        for name, meta in self.items.items():
            lines.append(f"  [{name}] {len(meta.variables)} variables")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

    def __repr__(self) -> str:
        return f"MetadataPackage(name={self.name!r}, datasets={len(self.items)})"
