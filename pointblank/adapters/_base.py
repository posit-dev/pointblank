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


