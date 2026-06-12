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


# Valid validation method names from the Validate class
_VALID_VALIDATION_METHODS = frozenset(
    {
        "col_vals_gt",
        "col_vals_lt",
        "col_vals_eq",
        "col_vals_ne",
        "col_vals_ge",
        "col_vals_le",
        "col_vals_between",
        "col_vals_outside",
        "col_vals_in_set",
        "col_vals_not_in_set",
        "col_vals_increasing",
        "col_vals_decreasing",
        "col_vals_null",
        "col_vals_not_null",
        "col_vals_regex",
        "col_vals_within_spec",
        "col_vals_expr",
        "col_exists",
        "col_pct_null",
        "col_schema_match",
        "col_count_match",
        "rows_distinct",
        "rows_complete",
        "row_count_match",
        "conjointly",
    }
)


@dataclass
class Contract:
    """A declarative boundary contract for pipeline data.

    A Contract defines what data must look like at a specific point in a pipeline.
    It combines a Schema (structural expectations) with validation Steps (semantic
    expectations) and metadata (ownership, versioning, directionality).

    Parameters
    ----------
    name
        A human-readable name for this contract (e.g., "raw_clickstream_feed").
    direction
        Either "source" (inbound data) or "target" (outbound data product).
        This is metadata for reporting and does not change validation behavior.
    schema
        A Schema object defining expected column names and types.
    steps
        A list of Step objects defining validation rules beyond schema checks.
    version
        Semantic version string for tracking contract evolution.
    owner
        Who is responsible for maintaining this contract.
    consumers
        Who depends on data conforming to this contract.
    description
        Optional longer description of the contract's purpose.
    thresholds
        Default thresholds for this contract's validations.
    on_violation
        What to do when the contract is violated: "warn", "raise", or "log".
        Default is "warn".

    Examples
    --------
    ```python
    import pointblank as pb

    source_contract = pb.Contract(
        name="raw_sales_feed",
        direction="source",
        schema=pb.Schema(
            order_id="String",
            amount_cents="Int64",
            currency="String",
        ),
        steps=[
            pb.Step("col_vals_not_null", columns=["order_id", "amount_cents"]),
            pb.Step("rows_distinct", columns=["order_id"]),
        ],
        version="1.0.0",
        owner="data-platform-team",
    )
    ```
    """

    name: str
    direction: Literal["source", "target"] = "source"
    schema: Schema | None = None
    steps: list[Step] = field(default_factory=list)
    version: str | None = None
    owner: str | None = None
    consumers: str | list[str] | None = None
    description: str | None = None
    thresholds: Thresholds | None = None
    on_violation: Literal["warn", "raise", "log"] = "warn"

    def __post_init__(self) -> None:
        # Validate direction
        if self.direction not in ("source", "target"):
            raise ValueError(
                f"Contract direction must be 'source' or 'target', got '{self.direction}'."
            )

        # Validate on_violation
        if self.on_violation not in ("warn", "raise", "log"):
            raise ValueError(
                f"Contract on_violation must be 'warn', 'raise', or 'log', "
                f"got '{self.on_violation}'."
            )

        # Validate name
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Contract name must be a non-empty string.")

        # Validate steps
        if not isinstance(self.steps, list):
            raise TypeError("Contract steps must be a list of Step objects.")
        for i, step in enumerate(self.steps):
            if not isinstance(step, Step):
                raise TypeError(
                    f"All items in steps must be Step objects, got {type(step).__name__} "
                    f"at index {i}."
                )
            if step.method not in _VALID_VALIDATION_METHODS:
                raise ValueError(
                    f"Unknown validation method '{step.method}' in step at index {i}. "
                    f"Valid methods are: {sorted(_VALID_VALIDATION_METHODS)}"
                )

    def to_validate(self, data: IntoDataFrame) -> Validate:
        """Compile this Contract into a Validate object ready for interrogation.

        This creates a Validate object with all schema checks and validation steps
        from this contract applied. The resulting Validate object has NOT been
        interrogated yet — call `.interrogate()` on it to execute the validation.

        Parameters
        ----------
        data
            The data table to validate against this contract.

        Returns
        -------
        Validate
            A Validate object with all contract checks applied (not yet interrogated).

        Examples
        --------
        ```python
        import pointblank as pb

        contract = pb.Contract(
            name="my_data",
            schema=pb.Schema(id="Int64", name="String"),
            steps=[pb.Step("col_vals_not_null", columns=["id"])],
        )

        # Create and interrogate
        validation = contract.to_validate(my_data).interrogate()
        ```
        """
        from pointblank.validate import Validate

        validation = Validate(
            data=data,
            tbl_name=self.name,
            label=f"Contract: {self.name}" + (f" v{self.version}" if self.version else ""),
            thresholds=self.thresholds,
            owner=self.owner,
            consumers=self.consumers,
            version=self.version,
        )

        # Add schema validation if a schema is defined
        if self.schema is not None:
            validation = validation.col_schema_match(schema=self.schema)

        # Add each step as a validation method call
        for step in self.steps:
            method = getattr(validation, step.method, None)
            if method is None:
                raise AttributeError(
                    f"Validate object has no method '{step.method}'. "
                    f"This may indicate a version mismatch."
                )
            validation = method(**step.kwargs)

        return validation

    def validate(self, data: IntoDataFrame) -> Validate:
        """Compile and interrogate this Contract against the provided data.

        This is a convenience method that calls `to_validate(data).interrogate()`.

        Parameters
        ----------
        data
            The data table to validate against this contract.

        Returns
        -------
        Validate
            An interrogated Validate object with results.
        """
        return self.to_validate(data).interrogate()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Contract to a dictionary for YAML/JSON export.

        Returns
        -------
        dict
            A dictionary representation of this contract.
        """
        result: dict[str, Any] = {"name": self.name}

        if self.direction != "source":
            result["direction"] = self.direction

        if self.version is not None:
            result["version"] = self.version

        if self.owner is not None:
            result["owner"] = self.owner

        if self.consumers is not None:
            result["consumers"] = self.consumers

        if self.description is not None:
            result["description"] = self.description

        if self.on_violation != "warn":
            result["on_violation"] = self.on_violation

        if self.schema is not None:
            result["schema"] = _schema_to_dict(self.schema)

        if self.steps:
            result["steps"] = [step.to_dict() for step in self.steps]

        if self.thresholds is not None:
            result["thresholds"] = _thresholds_to_dict(self.thresholds)

        return result

