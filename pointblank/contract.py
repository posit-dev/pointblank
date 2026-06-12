from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from pointblank.schema import Schema
    from pointblank.thresholds import Thresholds
    from pointblank.validate import Validate

__all__ = ["Step", "Contract"]


@dataclass
class Step:

    method: str
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __init__(self, method: str, **kwargs: Any) -> None:
        self.method = method
        self.kwargs = kwargs

    def __repr__(self) -> str:
        kwargs_str = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        if kwargs_str:
            return f"Step({self.method!r}, {kwargs_str})"
        return f"Step({self.method!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Step):
            return NotImplemented
        return self.method == other.method and self.kwargs == other.kwargs

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Step to a dictionary for YAML/JSON export.

        Returns
        -------
        dict
            A dictionary representation of this step.
        """
        if self.kwargs:
            return {self.method: dict(self.kwargs)}
        return {self.method: {}}

