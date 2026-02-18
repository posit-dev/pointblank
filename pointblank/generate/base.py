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
        Output format: `"polars"`, `"pandas"`, or `"dict"`.
    country
        Country code(s) for realistic data generation. Accepts: (1) a single ISO 3166-1
        alpha-2/alpha-3 code (e.g., `"US"`, `"DEU"`), (2) a list of codes for uniform mixing (e.g.,
        `["US", "DE", "JP"]`), or (3) a dict mapping codes to positive weights (e.g.,
        `{"US": 60, "DE": 25}`). Weights are auto-normalized to sum to `1.0`. Default is `"US"`.
    shuffle
        When `country` specifies multiple countries, controls whether the output rows are randomly
        interleaved (`True`, the default) or grouped in contiguous country blocks (`False`). Has no
        effect when `country` is a single string.
    weighted
        When `True`, names and locations are sampled according to real-world frequency tiers (e.g.,
        common names like "James" appear far more often than rare names like "Zebediah"). When
        `False` (the default), all entries are equally likely. Only affects data files that have
        been migrated to the tiered format; flat-list data always uses uniform sampling.
    max_unique_retries
        Maximum number of retries when generating unique values. If the generator fails to produce a
        unique value after this many attempts, it will raise an error to prevent infinite loops.
    """

    n: int = 100
    seed: int | None = None
    output: Literal["polars", "pandas", "dict"] = "polars"
    country: str | list[str] | dict[str, float] = "US"
    shuffle: bool = True
    weighted: bool = False
    max_unique_retries: int = 1000

    def __post_init__(self):
        if self.n < 0:
            raise ValueError(f"n must be non-negative, got {self.n}")
        if self.max_unique_retries < 1:
            raise ValueError(
                f"max_unique_retries must be at least 1, got {self.max_unique_retries}"
            )
        # Validate country input
        _validate_country(self.country)


def _validate_country(country: str | list[str] | dict[str, float]) -> None:
    """Validate the `country` parameter."""
    if isinstance(country, str):
        return  # single country code — validated downstream by _normalize_country
    if isinstance(country, list):
        if len(country) == 0:
            raise ValueError("country list must contain at least one country code.")
        return
    if isinstance(country, dict):
        if len(country) == 0:
            raise ValueError("country dict must contain at least one country code.")
        for code, weight in country.items():
            if not isinstance(weight, (int, float)):
                raise ValueError(
                    f"country weight for '{code}' must be a number, got {type(weight).__name__}."
                )
            if weight <= 0:
                raise ValueError(f"country weight for '{code}' must be positive, got {weight}.")
        return
    raise TypeError(
        f"country must be a str, list[str], or dict[str, float], got {type(country).__name__}."
    )
