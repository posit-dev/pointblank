"""
Base infrastructure for data generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    pass

__all__ = ["GeneratorConfig"]


@dataclass
class GeneratorConfig:
    """
    Configuration for data generation.

    Parameters
    ----------
    n
        Number of rows to generate.
    seed
        Random seed for reproducibility.
    output
        Output format: "polars", "pandas", or "dict".
    country
        Country code for realistic data generation. Accepts ISO 3166-1 alpha-2 codes
        (e.g., `"US"`, `"DE"`, `"FR"`) or alpha-3 codes (e.g., `"USA"`, `"DEU"`).
        Default is `"US"`.
    max_unique_retries
        Maximum retries when generating unique values.
    """

    n: int = 100
    seed: int | None = None
    output: Literal["polars", "pandas", "dict"] = "polars"
    country: str = "US"
    max_unique_retries: int = 1000

    def __post_init__(self):
        if self.n < 0:
            raise ValueError(f"n must be non-negative, got {self.n}")
        if self.max_unique_retries < 1:
            raise ValueError(
                f"max_unique_retries must be at least 1, got {self.max_unique_retries}"
            )
