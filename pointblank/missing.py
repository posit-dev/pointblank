from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MissingSpec",
]


@dataclass
class MissingSpec:
    """
    Specification for structured missing values in a column.

    Real-world data rarely encodes missingness as a single `null` value. Survey data distinguishes
    *refused* from *don't know* from *not applicable*; clinical data uses codes like `"NOT DONE"`;
    statistical packages use sentinel values such as `-99`, `".A"`, or `""`. A `MissingSpec`
    captures these sentinel values, the *reason* each one represents, and how they should be
    handled during validation and analysis.

    This brings the idea of *structured missingness* (a missing value carries a reason for its
    absence) into Pointblank's runtime validation layer. Once defined, a `MissingSpec` can be
    passed to validation methods (via `missing=`) to automatically exclude sentinel values from
    constraint checks, or used with dedicated methods like
    [`Validate.col_missing_coded()`](`pointblank.Validate.col_missing_coded`) and
    [`Validate.col_pct_missing()`](`pointblank.Validate.col_pct_missing`).

    Parameters
    ----------
    reasons
        A dictionary mapping sentinel values to reason labels. Keys are the actual values present
        in the data (e.g., `-99`, `"NA"`, `".A"`). Values are human-readable reason identifiers
        (e.g., `"refused"`, `"not_asked"`).
    categories
        Optional grouping of reasons into categories (e.g., an `"item_nonresponse"` category that
        groups `"refused"` and `"dont_know"`). Useful for aggregate reporting and for checking
        missingness rates by category. Each value is a list of reason labels that appear in
        `reasons`. Default is `None`.
    null_is_missing
        Whether actual null/`None`/`NaN` values should also be treated as missing (with reason
        given by `null_reason`). Default is `True`.
    null_reason
        The reason label assigned to actual null values when `null_is_missing=True`. Default is
        `"unknown"`.
    description
        Optional human-readable description of the overall missingness pattern. Default is `None`.

    Returns
    -------
    MissingSpec
        A missing-value specification that can be attached to a `Field` (via `missing=`) or passed
        to validation methods.

    Examples
    --------
    Define the missing-value codes for a survey `age` variable:

    ```python
    import pointblank as pb

    age_missing = pb.MissingSpec(
        reasons={
            -99: "not_asked",       # Question wasn't asked to this participant
            -98: "refused",         # Participant declined to answer
            -97: "dont_know",       # Participant didn't know
            -96: "not_applicable",  # Question doesn't apply
        },
        categories={
            "item_nonresponse": ["refused", "dont_know"],
            "design": ["not_asked", "not_applicable"],
        },
    )
    ```

    The spec can then answer questions about its own structure:

    ```python
    age_missing.sentinel_values()              # [-99, -98, -97, -96]
    age_missing.reason_for(-98)                # "refused"
    age_missing.values_for_reason("refused")   # [-98]
    age_missing.values_for_category("item_nonresponse")  # [-98, -97]
    ```
    """

    reasons: dict[Any, str]
    categories: dict[str, list[str]] | None = None
    null_is_missing: bool = True
    null_reason: str = "unknown"
    description: str | None = field(default=None)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that the missing specification is internally consistent."""
        if not isinstance(self.reasons, dict):
            raise TypeError(
                f"reasons must be a dict mapping sentinel values to reason labels, "
                f"got {type(self.reasons).__name__}"
            )

        if len(self.reasons) == 0 and not self.null_is_missing:
            raise ValueError(
                "A MissingSpec must define at least one sentinel value in `reasons`, "
                "or set `null_is_missing=True`."
            )

        for value, reason in self.reasons.items():
            if not isinstance(reason, str):
                raise TypeError(
                    f"Reason labels must be strings, got {type(reason).__name__} "
                    f"for sentinel value {value!r}."
                )

        if not isinstance(self.null_reason, str):
            raise TypeError(
                f"null_reason must be a string, got {type(self.null_reason).__name__}."
            )

        if self.categories is not None:
            if not isinstance(self.categories, dict):
                raise TypeError(
                    f"categories must be a dict mapping category names to lists of reason "
                    f"labels, got {type(self.categories).__name__}."
                )

            known_reasons = set(self.reasons.values())
            if self.null_is_missing:
                known_reasons.add(self.null_reason)

            for category, reason_list in self.categories.items():
                if not isinstance(reason_list, (list, tuple)):
                    raise TypeError(
                        f"Category '{category}' must map to a list of reason labels, "
                        f"got {type(reason_list).__name__}."
                    )
                unknown = [r for r in reason_list if r not in known_reasons]
                if unknown:
                    raise ValueError(
                        f"Category '{category}' references unknown reason label(s) {unknown}. "
                        f"Known reasons are {sorted(known_reasons)}."
                    )

    def sentinel_values(self) -> list:
        """Get all sentinel values that encode missingness.

        Returns
        -------
        list
            The keys of `reasons` (the actual values in the data that represent missingness).
            Note that this does *not* include `None` even when `null_is_missing=True`; use
            [`is_missing()`](`pointblank.MissingSpec.is_missing`) to test individual values.
        """
        return list(self.reasons.keys())

    def reason_for(self, value: Any) -> str | None:
        """Get the reason label for a specific value.

        Parameters
        ----------
        value
            A value from the data.

        Returns
        -------
        str | None
            The reason label if `value` is a declared sentinel value, `null_reason` if `value`
            is `None` and `null_is_missing=True`, or `None` if the value is not considered
            missing.
        """
        if value is None:
            return self.null_reason if self.null_is_missing else None
        return self.reasons.get(value)

    def is_missing(self, value: Any) -> bool:
        """Check whether a value should be considered missing under this spec.

        Parameters
        ----------
        value
            A value from the data.

        Returns
        -------
        bool
            `True` if `value` is a declared sentinel value, or if `value` is `None` and
            `null_is_missing=True`.
        """
        if value is None:
            return self.null_is_missing
        return value in self.reasons

    def values_for_reason(self, reason: str) -> list:
        """Get all sentinel values that correspond to a given reason.

        Parameters
        ----------
        reason
            A reason label.

        Returns
        -------
        list
            All sentinel values mapped to `reason`.
        """
        return [v for v, r in self.reasons.items() if r == reason]

    def values_for_category(self, category: str) -> list:
        """Get all sentinel values whose reason falls in a given category.

        Parameters
        ----------
        category
            A category name defined in `categories`.

        Returns
        -------
        list
            All sentinel values whose reason label is in the given category. Returns an empty
            list if `categories` is `None` or the category is undefined.
        """
        if self.categories is None:
            return []
        reasons_in_cat = self.categories.get(category, [])
        return [v for v, r in self.reasons.items() if r in reasons_in_cat]

    def reasons_list(self) -> list[str]:
        """Get the distinct reason labels defined by this spec.

        Returns
        -------
        list[str]
            The distinct reason labels (in first-seen order), including `null_reason` when
            `null_is_missing=True`.
        """
        seen: dict[str, None] = {}
        for r in self.reasons.values():
            seen.setdefault(r, None)
        if self.null_is_missing:
            seen.setdefault(self.null_reason, None)
        return list(seen.keys())
