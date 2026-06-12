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


