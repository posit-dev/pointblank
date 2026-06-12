from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from narwhals.typing import IntoDataFrame

    from pointblank.contract import Contract
    from pointblank.thresholds import Actions, FinalActions, Thresholds
    from pointblank.validate import Validate

__all__ = ["Pipeline", "PipelineResult"]


@dataclass
class PipelineResult:
    """Result of a pipeline boundary validation run.

    Contains the validation results for both source and target boundaries,
    plus metadata about the run.

    Attributes
    ----------
    source_validation
        The Validate object for the source boundary (or None if no source contract).
    target_validation
        The Validate object for the target boundary (or None if no target contract).
    transform_output
        The transformed data (only available when using `Pipeline.run()`).
    passed
        True only if BOTH boundary validations pass (no critical failures).
    source_passed
        True if the source boundary validation passes.
    target_passed
        True if the target boundary validation passes.
    """

    source_validation: Validate | None = None
    target_validation: Validate | None = None
    transform_output: Any = None
    _source_passed: bool | None = None
    _target_passed: bool | None = None

    @property
    def source_passed(self) -> bool:
        """Whether the source boundary validation passed."""
        if self.source_validation is None:
            return True
        return self._check_passed(self.source_validation)

    @property
    def target_passed(self) -> bool:
        """Whether the target boundary validation passed."""
        if self.target_validation is None:
            return True
        return self._check_passed(self.target_validation)

    @property
    def passed(self) -> bool:
        """True only if BOTH boundaries pass (no critical threshold exceeded)."""
        return self.source_passed and self.target_passed

    def _check_passed(self, validation: Validate) -> bool:
        """Check if a validation passes (no critical failures).

        A validation "passes" if all test units pass across all steps.
        """
        if hasattr(validation, "all_passed"):
            return validation.all_passed()
        return True

    def get_report(self) -> str:
        """Get a combined text summary of both boundary validations.

        Returns
        -------
        str
            A summary string describing the boundary validation results.
        """
        lines = ["Pipeline Boundary Validation Results", "=" * 40]

        if self.source_validation is not None:
            lines.append("")
            lines.append(f"Source Boundary: {'PASSED' if self.source_passed else 'FAILED'}")
            lines.append("-" * 30)
            _append_validation_summary(lines, self.source_validation)

        if self.target_validation is not None:
            lines.append("")
            lines.append(f"Target Boundary: {'PASSED' if self.target_passed else 'FAILED'}")
            lines.append("-" * 30)
            _append_validation_summary(lines, self.target_validation)

        lines.append("")
        lines.append(f"Overall: {'PASSED' if self.passed else 'FAILED'}")

        return "\n".join(lines)

    def __repr__(self) -> str:
        parts = ["PipelineResult("]
        parts.append(f"  source={'validated' if self.source_validation else 'none'},")
        parts.append(f"  target={'validated' if self.target_validation else 'none'},")
        parts.append(f"  passed={self.passed}")
        parts.append(")")
        return "\n".join(parts)


def _append_validation_summary(lines: list[str], validation: Validate) -> None:
    """Append a summary of a Validate object to the lines list."""
    try:
        n_steps = len(validation.validation_info) if hasattr(validation, "validation_info") else 0
        # n_passed() returns dict[int, int] when called without scalar=True
        passed_dict = validation.n_passed() if hasattr(validation, "n_passed") else {}
        failed_dict = validation.n_failed() if hasattr(validation, "n_failed") else {}
        total_passed = sum(passed_dict.values()) if isinstance(passed_dict, dict) else 0
        total_failed = sum(failed_dict.values()) if isinstance(failed_dict, dict) else 0
        lines.append(f"  Steps: {n_steps}")
        lines.append(f"  Test units passed: {total_passed}")
        lines.append(f"  Test units failed: {total_failed}")
    except Exception:
        lines.append("  (unable to retrieve summary)")


@dataclass
class Pipeline:
    """Binds source and target contracts into a pipeline boundary enforcement unit.

    A Pipeline enforces data quality at both the ingestion point ("boundary in") and
    the output point ("boundary out") of a data transformation. It validates that data
    entering a pipeline meets source contract requirements, and that data leaving meets
    target contract requirements.

    Parameters
    ----------
    source
        The source (inbound) Contract.
    target
        The target (outbound) Contract, or None if only validating inbound data.
    thresholds
        Global thresholds applied to both boundary validations (overrides contract-level
        thresholds).
    actions
        Actions triggered on threshold exceedance at either boundary.
    final_actions
        Actions triggered after both validations complete.
    label
        A label for this pipeline (used in reports).
    short_circuit
        If True (default), skip the transform and target validation when source validation
        fails critically. Set to False to always run both validations.

    Examples
    --------
    ```python
    import pointblank as pb

    source_contract = pb.Contract(
        name="raw_data",
        direction="source",
        steps=[pb.Step("col_vals_not_null", columns=["id"])],
    )

    target_contract = pb.Contract(
        name="clean_data",
        direction="target",
        steps=[pb.Step("col_vals_not_null", columns=pb.everything())],
    )

    pipeline = pb.Pipeline(
        source=source_contract,
        target=target_contract,
    )

    # Validate source data
    source_result = pipeline.validate_source(raw_data)

    # Validate target data
    target_result = pipeline.validate_target(clean_data)

    # Or do it all in one shot
    result = pipeline.run(data=raw_data, transform=my_transform)
    ```
    """

    source: Contract | None = None
    target: Contract | None = None
    thresholds: Thresholds | None = None
    actions: Actions | None = None
    final_actions: FinalActions | None = None
    label: str | None = None
    short_circuit: bool = True

    def __post_init__(self) -> None:
        from pointblank.contract import Contract

        if self.source is not None and not isinstance(self.source, Contract):
            raise TypeError(
                f"Pipeline 'source' must be a Contract object, got {type(self.source).__name__}."
            )
        if self.target is not None and not isinstance(self.target, Contract):
            raise TypeError(
                f"Pipeline 'target' must be a Contract object, got {type(self.target).__name__}."
            )
        if self.source is None and self.target is None:
            raise ValueError("Pipeline must have at least one of 'source' or 'target' contracts.")

    def validate_source(self, data: IntoDataFrame) -> Validate:
        """Validate data against the source (inbound) contract.

        Parameters
        ----------
        data
            The incoming data to validate.

        Returns
        -------
        Validate
            An interrogated Validate object with results.

        Raises
        ------
        ValueError
            If no source contract is defined.
        RuntimeError
            If on_violation="raise" and validation fails.
        """
        if self.source is None:
            raise ValueError("No source contract defined for this pipeline.")

        validation = self._build_validation(self.source, data)
        validation = validation.interrogate()

        self._handle_violation(self.source, validation)

        return validation

    def validate_target(self, data: IntoDataFrame) -> Validate:
        """Validate data against the target (outbound) contract.

        Parameters
        ----------
        data
            The outgoing data to validate.

        Returns
        -------
        Validate
            An interrogated Validate object with results.

        Raises
        ------
        ValueError
            If no target contract is defined.
        RuntimeError
            If on_violation="raise" and validation fails.
        """
        if self.target is None:
            raise ValueError("No target contract defined for this pipeline.")

        validation = self._build_validation(self.target, data)
        validation = validation.interrogate()

        self._handle_violation(self.target, validation)

        return validation

    def run(
        self,
        data: IntoDataFrame,
        transform: Callable[[Any], Any],
    ) -> PipelineResult:
        """Run the full pipeline: validate source, transform, validate target.

        Parameters
        ----------
        data
            The input data for the pipeline.
        transform
            A callable that transforms the source data into target data.
            Must accept the data and return transformed data.

        Returns
        -------
        PipelineResult
            A result object containing both validations and the transform output.
        """
        result = PipelineResult()

        # Step 1: Validate source if a source contract is defined
        if self.source is not None:
            source_validation = self._build_validation(self.source, data)
            source_validation = source_validation.interrogate()
            result.source_validation = source_validation

            # Handle violation for source
            try:
                self._handle_violation(self.source, source_validation)
            except RuntimeError:
                if self.short_circuit:
                    # Source failed critically and short_circuit is True
                    return result
                # If not short_circuit, continue despite source failure

            # Check if source passed; if not and short_circuit is True, skip transform
            if self.short_circuit and not result.source_passed:
                return result

        # Step 2: Run the transform
        transformed_data = transform(data)
        result.transform_output = transformed_data

        # Step 3: Validate target if a target contract is defined
        if self.target is not None:
            target_validation = self._build_validation(self.target, transformed_data)
            target_validation = target_validation.interrogate()
            result.target_validation = target_validation

            # Handle violation for target
            try:
                self._handle_violation(self.target, target_validation)
            except RuntimeError:
                pass  # Already stored in result

        return result

    def _build_validation(self, contract: Contract, data: IntoDataFrame) -> Validate:
        """Build a Validate object from a contract, applying pipeline-level overrides.

        Parameters
        ----------
        contract
            The contract to build from.
        data
            The data to validate.

        Returns
        -------
        Validate
            A Validate object (not yet interrogated).
        """
        from pointblank.validate import Validate

        # Determine thresholds: pipeline-level overrides contract-level
        thresholds = self.thresholds if self.thresholds is not None else contract.thresholds

        validation = Validate(
            data=data,
            tbl_name=contract.name,
            label=(self.label or f"Pipeline: {contract.name}") + f" [{contract.direction}]",
            thresholds=thresholds,
            actions=self.actions,
            final_actions=self.final_actions,
            owner=contract.owner,
            consumers=contract.consumers,
            version=contract.version,
        )

        # Add schema validation if defined
        if contract.schema is not None:
            validation = validation.col_schema_match(schema=contract.schema)

        # Add all steps
        for step in contract.steps:
            method = getattr(validation, step.method, None)
            if method is None:
                raise AttributeError(
                    f"Validate object has no method '{step.method}'. "
                    f"This may indicate a version mismatch."
                )
            validation = method(**step.kwargs)

        return validation

    def _handle_violation(self, contract: Contract, validation: Validate) -> None:
        """Handle a contract violation based on the contract's on_violation setting.

        Parameters
        ----------
        contract
            The contract that was validated.
        validation
            The interrogated Validate object.

        Raises
        ------
        RuntimeError
            If on_violation="raise" and the validation has failures.
        """
        # Check if there are any failures
        has_failures = not validation.all_passed()

        if not has_failures:
            return

        if contract.on_violation == "raise":
            raise RuntimeError(
                f"Contract '{contract.name}' ({contract.direction}) violated: "
                f"validation has failing steps."
            )
        elif contract.on_violation == "warn":
            warnings.warn(
                f"Contract '{contract.name}' ({contract.direction}) violated: "
                f"validation has failing steps.",
                UserWarning,
                stacklevel=3,
            )
        elif contract.on_violation == "log":
            import logging

            logger = logging.getLogger("pointblank.contract")
            logger.warning(
                f"Contract '{contract.name}' ({contract.direction}) violated: "
                f"validation has failing steps."
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize the Pipeline to a dictionary for YAML/JSON export.

        Returns
        -------
        dict
            A dictionary representation of this pipeline.
        """
        from pointblank.contract import _thresholds_to_dict

        result: dict[str, Any] = {}

        if self.label is not None:
            result["label"] = self.label

        if self.thresholds is not None:
            result["thresholds"] = _thresholds_to_dict(self.thresholds)

        if self.short_circuit is not True:
            result["short_circuit"] = self.short_circuit

        pipeline_dict: dict[str, Any] = {"pipeline": result}

        if self.source is not None:
            pipeline_dict["source"] = self.source.to_dict()

        if self.target is not None:
            pipeline_dict["target"] = self.target.to_dict()

        return pipeline_dict

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Pipeline:
        """Construct a Pipeline from a dictionary (e.g., parsed from YAML).

        Parameters
        ----------
        data
            A dictionary representation of a pipeline.

        Returns
        -------
        Pipeline
            A new Pipeline instance.
        """
        from pointblank.contract import Contract, _dict_to_thresholds

        pipeline_meta = data.get("pipeline", {})
        label = pipeline_meta.get("label")
        short_circuit = pipeline_meta.get("short_circuit", True)

        # Parse thresholds
        thresholds = None
        thresholds_data = pipeline_meta.get("thresholds")
        if thresholds_data is not None:
            thresholds = _dict_to_thresholds(thresholds_data)

        # Parse source contract
        source = None
        source_data = data.get("source")
        if source_data is not None:
            if "direction" not in source_data:
                source_data["direction"] = "source"
            source = Contract.from_dict(source_data)

        # Parse target contract
        target = None
        target_data = data.get("target")
        if target_data is not None:
            if "direction" not in target_data:
                target_data["direction"] = "target"
            target = Contract.from_dict(target_data)

        return cls(
            source=source,
            target=target,
            thresholds=thresholds,
            label=label,
            short_circuit=short_circuit,
        )

    @classmethod
    def from_yaml(cls, path: str) -> Pipeline:
        """Load a Pipeline from a YAML file.

        Parameters
        ----------
        path
            Path to the YAML file.

        Returns
        -------
        Pipeline
            A new Pipeline instance.
        """
        from pathlib import Path

        import yaml

        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            raise ValueError(f"YAML file is empty: {path}")

        return cls.from_dict(data)

