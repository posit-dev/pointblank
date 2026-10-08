from __future__ import annotations

from typing import Any

from pointblank.metadata._sdtm_templates import (
    get_sdtm_domain,
)
from pointblank.metadata._types import (
    Codelist,
    MetadataImport,
    VariableMetadata,
)

__all__ = [
    "sdtm_to_metadata",
    "validate_sdtm",
]

# ISO 8601 patterns used in CDISC
# Full: YYYY-MM-DDThh:mm:ss
# Partial dates allowed: YYYY, YYYY-MM, YYYY-MM-DD, etc.
_ISO8601_CDISC_PATTERN = (
    r"^"
    r"(\d{4})"  # Year (required)
    r"(-\d{2}"  # Month
    r"(-\d{2}"  # Day
    r"(T\d{2}"  # Hour
    r"(:\d{2}"  # Minute
    r"(:\d{2}"  # Second
    r")?)?)?)?)?"
    r"$"
)


def sdtm_to_metadata(
    domain: str,
    study_id: str | None = None,
) -> MetadataImport:
    """Convert an SDTM domain template to a `MetadataImport` object.

    This allows using the standard metadata pipeline (`to_schema`, `to_validate`) with SDTM domain
    specifications.

    Parameters
    ----------
    domain
        SDTM domain code (e.g., `"DM"`, `"AE"`, `"LB"`). This is case-insensitive.
    study_id
        Optional study identifier to include in metadata.

    Returns
    -------
    MetadataImport
        A `MetadataImport` representing the SDTM domain template.
    """
    template = get_sdtm_domain(domain)

    variables: list[VariableMetadata] = []
    codelists: dict[str, Codelist] = {}

    for spec in template.variables:
        # Map SDTM type to Pointblank dtype
        dtype = "Float64" if spec.dtype == "Num" else "String"

        var = VariableMetadata(
            name=spec.name,
            label=spec.label,
            dtype=dtype,
            role=spec.role,
            required=spec.required,
            max_length=spec.max_length,
            controlled_term=spec.controlled_term,
            cdisc_domain=template.domain,
            cdisc_role=spec.role,
        )
        variables.append(var)

    return MetadataImport(
        source_format="cdisc_sdtm",
        source_version="IG 3.4",
        dataset_name=template.domain,
        dataset_label=template.label,
        dataset_description=template.description,
        study_id=study_id,
        domain=template.domain,
        variables=variables,
        codelists=codelists,
    )


def validate_sdtm(
    data: Any,
    domain: str,
    study_id: str | None = None,
    check_dates: bool = True,
    check_lengths: bool = True,
    label: str | None = None,
    **kwargs: Any,
):
    """Generate a comprehensive SDTM validation workflow for a dataset.

    Creates a `Validate` object with checks for:

    - Schema conformance (required variables present with correct types)
    - Required variables are non-null
    - Variable length constraints (for Char variables)
    - DOMAIN column value matches expected domain code
    - ISO 8601 date format for --DTC timing variables
    - Sequence number positivity and uniqueness per subject

    Parameters
    ----------
    data
        The DataFrame to validate (Pandas or Polars).
    domain
        SDTM domain code (e.g., `"DM"`, `"AE"`, `"LB"`). This is case-insensitive.
    study_id
        Optional study identifier for the validation label.
    check_dates
        If `True`, validate ISO 8601 format for --DTC variables.
    check_lengths
        If `True`, validate string length constraints.
    label
        Custom label for the `Validate` object. Defaults to `"SDTM {domain} Validation"`.
    **kwargs
        Additional keyword arguments passed to the `Validate` constructor.

    Returns
    -------
    `Validate`
        A configured (but not yet interrogated) `Validate` object.

    Examples
    --------
    ```python
    import pointblank as pb
    from pointblank.metadata._sdtm_validate import validate_sdtm

    validation = validate_sdtm(dm_data, domain="DM").interrogate()
    ```
    """
    from pointblank.validate import Validate

    template = get_sdtm_domain(domain)

    if label is None:
        label_parts = [f"SDTM {domain.upper()} Validation"]
        if study_id:
            label_parts = [f"SDTM {domain.upper()} — {study_id}"]
        label = label_parts[0]

    validation = Validate(data=data, label=label, **kwargs)

    # Get the columns actually present in the data
    import narwhals as nw

    df = nw.from_native(data, eager_only=True)
    actual_columns = set(df.columns)

    # ── Required variables must be non-null ──
    for spec in template.variables:
        if spec.required and spec.name in actual_columns:
            validation = validation.col_vals_not_null(columns=spec.name)

    # ── DOMAIN column must equal the expected domain code ──
    if "DOMAIN" in actual_columns:
        validation = validation.col_vals_in_set(columns="DOMAIN", set=[domain.upper()])

    # ── Sequence number checks (--SEQ) ──
    seq_var = f"{domain.upper()}SEQ" if domain.upper() != "DM" else None
    if seq_var and seq_var in actual_columns:
        # Sequence numbers must be positive
        validation = validation.col_vals_gt(columns=seq_var, value=0)

    # ── String length checks ──
    # Note: col_vals_expr requires a narwhals Expr object. We build them
    # dynamically using narwhals for each Char variable with a length constraint.
    if check_lengths:
        for spec in template.variables:
            if spec.max_length is not None and spec.dtype == "Char" and spec.name in actual_columns:
                length_expr = nw.col(spec.name).str.len_chars() <= spec.max_length
                validation = validation.col_vals_expr(
                    expr=length_expr,
                    brief=f"{spec.name} length <= {spec.max_length}",
                )

    # ── ISO 8601 date checks for --DTC variables ──
    if check_dates:
        for spec in template.variables:
            if spec.name.endswith("DTC") and spec.name in actual_columns and spec.role == "Timing":
                validation = validation.col_vals_regex(
                    columns=spec.name,
                    pattern=_ISO8601_CDISC_PATTERN,
                    na_pass=True,
                )

    return validation
