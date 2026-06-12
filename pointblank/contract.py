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
    """A single validation step in a Contract, defined declaratively.

    This is the declarative equivalent of calling a validation method on Validate.
    Steps are stored as data and compiled into Validate method calls at enforcement time.

    Parameters
    ----------
    method
        The validation method name (e.g., "col_vals_gt", "rows_distinct").
    **kwargs
        All parameters that would be passed to the corresponding Validate method.

    Examples
    --------
    ```python
    import pointblank as pb

    # A step that checks column values are greater than 0
    step = pb.Step("col_vals_gt", columns="amount", value=0)

    # A step that checks rows are distinct by a key column
    step = pb.Step("rows_distinct", columns=["order_id"])

    # A step that checks a regex pattern
    step = pb.Step("col_vals_regex", columns="email", pattern=r"^[^@]+@[^@]+\\.[^@]+$")
    ```
    """

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Step:
        """Construct a Step from a dictionary (e.g., parsed from YAML).

        Parameters
        ----------
        data
            A dictionary with a single key (the method name) mapping to its kwargs dict.

        Returns
        -------
        Step
            A new Step instance.
        """
        if len(data) != 1:
            raise ValueError(
                f"Step dictionary must have exactly one key (the method name), got {len(data)} keys."
            )
        method = next(iter(data))
        kwargs = data[method]
        if kwargs is None:
            kwargs = {}
        if not isinstance(kwargs, dict):
            raise TypeError(
                f"Step kwargs must be a dictionary, got {type(kwargs).__name__} for method "
                f"'{method}'."
            )
        return cls(method, **kwargs)


