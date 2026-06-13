from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pointblank.contract import Contract
    from pointblank.validate import Validate


@dataclass
class MappedConstraint:
    """A single constraint from an external format mapped to a Pointblank validation step.

    Parameters
    ----------
    method
        The Pointblank Validate method name (e.g., `"col_vals_gt"`).
    kwargs
        The keyword arguments to pass to that method.
    source_description
        Human-readable description of the original constraint in the source format.
    """

    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)
    source_description: str = ""

    def __repr__(self) -> str:
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"MappedConstraint({self.method!r}, {kwargs_str})"


@dataclass
class ContractImport:
    """Result of importing an external contract/schema.

    Contains the parsed schema information, mapped validation steps, and any warnings about
    constraints that couldn't be fully translated.

    Parameters
    ----------
    source_format
        The adapter format name (e.g., `"json_schema"`, `"frictionless"`, etc.).
    source_path
        File path if loaded from a file, else None.
    source_version
        Version of the source format specification, if detectable.
    columns
        List of (column_name, dtype_string_or_None) tuples detected from the source.
    constraints
        List of MappedConstraint objects ready for translation to Validate steps.
    metadata
        Any additional metadata from the source (title, description, etc.).
    warnings
        Messages about constraints that couldn't be mapped.
    coverage
        Fraction of source constraints successfully mapped (`0.0` to `1.0`).
    """

    source_format: str
    source_path: str | None = None
    source_version: str | None = None
    columns: list[tuple[str, str | None]] = field(default_factory=list)
    constraints: list[MappedConstraint] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    coverage: float = 1.0

    def to_validate(self, data: Any, **kwargs: Any) -> Validate:
        """Build a Validate object from the imported contract.

        Parameters
        ----------
        data
            The data table to validate.
        **kwargs
            Additional keyword arguments passed to the Validate constructor.

        Returns
        -------
        Validate
            A Validate object with all imported checks applied (not yet interrogated).
        """
        from pointblank.schema import Schema
        from pointblank.validate import Validate

        validation = Validate(data=data, **kwargs)

        # Add schema check if columns with types were detected
        schema_cols = {name: dtype for name, dtype in self.columns if dtype is not None}
        if schema_cols:
            schema = Schema(**schema_cols)
            validation = validation.col_schema_match(schema=schema)

        # Add each mapped constraint as a validation step
        for constraint in self.constraints:
            method = getattr(validation, constraint.method, None)
            if method is None:
                self.warnings.append(f"Validate has no method '{constraint.method}' — skipped.")
                continue
            validation = method(**constraint.kwargs)

        return validation

    def to_contract(self, name: str = "imported_contract", **kwargs: Any) -> Contract:
        """Build a Contract object from the imported data.

        Parameters
        ----------
        name
            Name for the created `Contract`.
        **kwargs
            Additional keyword arguments passed to the `Contract` constructor.

        Returns
        -------
        Contract
            A Contract object with schema and steps derived from the import.
        """
        from pointblank.contract import Contract, Step
        from pointblank.schema import Schema

        # Build schema from columns
        schema = None
        schema_cols = {col_name: dtype for col_name, dtype in self.columns if dtype is not None}
        if schema_cols:
            schema = Schema(**schema_cols)

        # Build steps from constraints
        steps = [Step(c.method, **c.kwargs) for c in self.constraints]

        contract_kwargs: dict[str, Any] = {
            "name": name,
            "schema": schema,
            "steps": steps,
        }
        # Merge in any metadata as description
        if "description" in self.metadata and "description" not in kwargs:
            contract_kwargs["description"] = self.metadata["description"]

        contract_kwargs.update(kwargs)
        return Contract(**contract_kwargs)

class ContractAdapter:
    """Base class for contract import/export adapters.

    Subclass this to add support for a new external format.

    Attributes
    ----------
    format_name
        Short identifier for this format (e.g., `"json_schema"`).
    file_extensions
        File extensions associated with this format (e.g., `[".json"]`).
    supports_import
        Whether this adapter supports importing from the format.
    supports_export
        Whether this adapter supports exporting to the format.
    """

    format_name: str = ""
    file_extensions: list[str] = []
    supports_import: bool = True
    supports_export: bool = True

    @staticmethod
    def detect(source: Any) -> bool:
        """Return True if this adapter can handle the given source.

        Parameters
        ----------
        source
            A file path string, dict, or Python object to inspect.

        Returns
        -------
        bool
            True if this adapter can handle the source.
        """
        raise NotImplementedError

