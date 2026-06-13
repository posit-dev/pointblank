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

