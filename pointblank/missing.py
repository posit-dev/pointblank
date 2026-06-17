from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "MissingSpec",
]


# Standard HL7/CDISC null flavors mapped to snake_case reason labels
_CDISC_NULL_FLAVORS: dict[str, str] = {
    "NI": "no_information",
    "NA": "not_applicable",
    "UNK": "unknown",
    "ASKU": "asked_but_unknown",
    "NAV": "temporarily_unavailable",
    "NASK": "not_asked",
    "OTH": "other",
    "PINF": "positive_infinity",
    "NINF": "negative_infinity",
    "MSK": "masked",
    "DER": "derived",
    "QS": "sufficient_quantity",
    "TRC": "trace",
    "NP": "not_present",
}


def _slugify(label: Any) -> str:
    """Convert a human-readable label into a snake_case reason identifier."""
    slug = re.sub(r"[^0-9a-zA-Z]+", "_", str(label).strip().lower()).strip("_")
    return slug or "missing"


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

    # ------------------------------------------------------------------
    # Factory methods (pre-built specs and metadata-import integration)
    # ------------------------------------------------------------------

    @classmethod
    def from_cdisc_null_flavors(
        cls,
        null_is_missing: bool = True,
        null_reason: str = "no_information",
        description: str | None = "CDISC/HL7 null flavors",
    ) -> "MissingSpec":
        """Create a `MissingSpec` for the standard HL7/CDISC *null flavors*.

        Clinical data uses standardized null flavor codes to record *why* a value is absent (e.g.,
        `"NASK"` for "not asked", `"UNK"` for "unknown"). This returns a ready-to-use spec mapping
        those codes to reason labels.

        Parameters
        ----------
        null_is_missing
            Whether actual null values should also be treated as missing. Default is `True`.
        null_reason
            The reason label for actual null values. Default is `"no_information"`.
        description
            Optional description. Default identifies the spec as CDISC/HL7 null flavors.

        Returns
        -------
        MissingSpec
            A spec with the standard null flavor codes.

        Examples
        --------
        ```python
        import pointblank as pb

        cdisc_missing = pb.MissingSpec.from_cdisc_null_flavors()
        cdisc_missing.reason_for("NASK")   # "not_asked"
        ```
        """
        reasons = dict(_CDISC_NULL_FLAVORS)
        categories = {
            "unknown": ["no_information", "unknown", "asked_but_unknown", "temporarily_unavailable"],
            "not_applicable": ["not_applicable", "not_asked", "not_present"],
            "boundary": ["positive_infinity", "negative_infinity"],
        }
        return cls(
            reasons=reasons,
            categories=categories,
            null_is_missing=null_is_missing,
            null_reason=null_reason,
            description=description,
        )

    # Convenient short alias
    @classmethod
    def from_cdisc(cls, **kwargs: Any) -> "MissingSpec":
        """Alias for [`from_cdisc_null_flavors()`](`pointblank.MissingSpec.from_cdisc_null_flavors`)."""
        return cls.from_cdisc_null_flavors(**kwargs)

    @classmethod
    def from_sas(
        cls,
        reasons: dict[str, str] | None = None,
        include_underscore: bool = True,
        null_is_missing: bool = True,
        null_reason: str = "system_missing",
        description: str | None = "SAS special missing values",
    ) -> "MissingSpec":
        """Create a `MissingSpec` for SAS special missing values.

        SAS encodes missingness with `"."` (system missing), `"._"`, and `".A"` through `".Z"` (27
        user-defined missing codes). This returns a spec covering all of them; you can override the
        reason label for any specific code via `reasons=`.

        Parameters
        ----------
        reasons
            Optional mapping of specific SAS missing codes to custom reason labels (e.g.,
            `{".A": "not_applicable", ".B": "below_detection"}`). These override the defaults.
        include_underscore
            Whether to include the `"._"` special missing code. Default is `True`.
        null_is_missing
            Whether actual null values should also be treated as missing. Default is `True`.
        null_reason
            The reason label for actual null values. Default is `"system_missing"`.
        description
            Optional description. Default identifies the spec as SAS special missing values.

        Returns
        -------
        MissingSpec
            A spec covering the SAS special missing values.

        Examples
        --------
        ```python
        import pointblank as pb

        sas_missing = pb.MissingSpec.from_sas(
            reasons={".A": "not_applicable", ".B": "below_detection"}
        )
        sas_missing.reason_for(".A")   # "not_applicable"
        sas_missing.reason_for(".C")   # "user_missing_c"
        ```
        """
        built: dict[Any, str] = {".": "system_missing"}
        if include_underscore:
            built["._"] = "system_missing"
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            built[f".{letter}"] = f"user_missing_{letter.lower()}"
        if reasons:
            for code, label in reasons.items():
                built[code] = label
        return cls(
            reasons=built,
            null_is_missing=null_is_missing,
            null_reason=null_reason,
            description=description,
        )

    @classmethod
    def from_spss(
        cls,
        missing_values: list,
        labels: dict[Any, str] | None = None,
        null_is_missing: bool = True,
        null_reason: str = "unknown",
        description: str | None = "SPSS user-defined missing values",
    ) -> "MissingSpec":
        """Create a `MissingSpec` from SPSS-style user-defined missing values.

        SPSS supports up to 3 user-defined missing values per variable (plus a range). Pass the
        missing values (and optionally their value labels) to build a spec. Reason labels are
        derived from the labels when available, otherwise a `"missing_<value>"` placeholder is used.

        Parameters
        ----------
        missing_values
            The sentinel values that SPSS marks as missing for the variable (e.g., `[-99, -98]`).
        labels
            Optional mapping of sentinel value to human-readable label (e.g., `{-99: "Refused"}`).
            Labels are slugified into reason identifiers (e.g., `"Refused"` -> `"refused"`).
        null_is_missing
            Whether actual null values should also be treated as missing. Default is `True`.
        null_reason
            The reason label for actual null values. Default is `"unknown"`.
        description
            Optional description. Default identifies the spec as SPSS user-defined missing values.

        Returns
        -------
        MissingSpec
            A spec built from the SPSS missing values.

        Examples
        --------
        ```python
        import pointblank as pb

        spss_missing = pb.MissingSpec.from_spss(
            missing_values=[-99, -98],
            labels={-99: "Not asked", -98: "Refused"},
        )
        spss_missing.reason_for(-98)   # "refused"
        ```
        """
        labels = labels or {}
        reasons = {
            value: (_slugify(labels[value]) if value in labels else f"missing_{_slugify(value)}")
            for value in missing_values
        }
        return cls(
            reasons=reasons,
            null_is_missing=null_is_missing,
            null_reason=null_reason,
            description=description,
        )

    @classmethod
    def from_variable_metadata(
        cls,
        variable: Any,
        null_is_missing: bool = True,
        null_reason: str = "unknown",
    ) -> "MissingSpec | None":
        """Create a `MissingSpec` from an imported variable's metadata.

        This works with a [`VariableMetadata`](`pointblank.VariableMetadata`) object (as produced by
        [`import_metadata()`](`pointblank.import_metadata`) for SPSS, Stata, and SAS files). It reads
        the variable's `missing_values` and derives reason labels from `missing_value_labels` or
        `value_labels` when available.

        Parameters
        ----------
        variable
            A variable-metadata object exposing `missing_values` and (optionally)
            `missing_value_labels` / `value_labels` attributes.
        null_is_missing
            Whether actual null values should also be treated as missing. Default is `True`.
        null_reason
            The reason label for actual null values. Default is `"unknown"`.

        Returns
        -------
        MissingSpec | None
            A spec built from the variable's missing values, or `None` if the variable declares no
            missing values.
        """
        missing_values = getattr(variable, "missing_values", None) or []
        if not missing_values:
            return None

        labels = getattr(variable, "missing_value_labels", None) or {}
        value_labels = getattr(variable, "value_labels", None) or {}

        reasons: dict[Any, str] = {}
        for value in missing_values:
            label = labels.get(value)
            if label is None:
                label = value_labels.get(value)
            reasons[value] = _slugify(label) if label else f"missing_{_slugify(value)}"

        return cls(
            reasons=reasons,
            null_is_missing=null_is_missing,
            null_reason=null_reason,
            description=f"Imported missing values for '{getattr(variable, 'name', 'variable')}'",
        )
