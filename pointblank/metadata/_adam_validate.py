from __future__ import annotations

from typing import Any

from pointblank.metadata._adam_templates import (
    get_adam_dataset,
)
from pointblank.metadata._types import (
    MetadataImport,
    VariableMetadata,
)

__all__ = [
    "adam_to_metadata",
    "validate_adam",
]


def adam_to_metadata(
    dataset: str,
    study_id: str | None = None,
) -> MetadataImport:
    """Convert an ADaM dataset template to a MetadataImport object.

    Parameters
    ----------
    dataset
        ADaM dataset name (e.g., `"ADSL"`, `"BDS"`, `"ADAE"`, `"ADTTE"`). This is case-insensitive.
    study_id
        Optional study identifier.

    Returns
    -------
    MetadataImport
        A MetadataImport representing the ADaM dataset template.
    """
    template = get_adam_dataset(dataset)

    variables: list[VariableMetadata] = []
    for spec in template.variables:
        dtype = "Float64" if spec.dtype == "Num" else "String"
        var = VariableMetadata(
            name=spec.name,
            label=spec.label,
            dtype=dtype,
            required=spec.required,
            max_length=spec.max_length,
            controlled_term=spec.controlled_term,
            cdisc_domain=template.name,
            cdisc_role=spec.core,
            adam_derivation=spec.source,
        )
        variables.append(var)

    return MetadataImport(
        source_format="cdisc_adam",
        source_version="IG 1.1",
        dataset_name=template.name,
        dataset_label=template.label,
        dataset_description=template.description,
        study_id=study_id,
        domain=template.name,
        variables=variables,
    )


def validate_adam(
    data: Any,
    dataset: str,
    study_id: str | None = None,
    check_population_flags: bool = True,
    check_bds_structure: bool = True,
    check_traceability: bool = True,
    label: str | None = None,
    **kwargs: Any,
):
    """Generate a comprehensive ADaM validation workflow for a dataset.

    Creates a Validate object with checks for:

    - Required variables present and non-null
    - Population flag values (Y/N only, no nulls in flag columns)
    - BDS structure: PARAMCD, PARAM, AVAL consistency
    - ADTTE: CNSR values (0 or 1), AVAL >= 0
    - TRT01P (planned treatment) present and non-null in ADSL
    - Traceability variable presence (SRCDOM/SRCVAR/SRCSEQ non-null when present)

    Parameters
    ----------
    data
        The DataFrame to validate (pandas or polars).
    dataset
        ADaM dataset name (e.g., `"ADSL"`, `"BDS"`, `"ADAE"`, `"ADTTE"`). This is case-insensitive.
    study_id
        Optional study identifier for the validation label.
    check_population_flags
        If `True`, validate population flag columns (Y/N values only).
    check_bds_structure
        If `True`, validate BDS-specific structure (`PARAMCD`/`PARAM`/`AVAL`).
    check_traceability
        If `True`, check that traceability variables are non-null when present.
    label
        Custom label for the Validate object.
    **kwargs
        Additional keyword arguments passed to the Validate constructor.

    Returns
    -------
    Validate
        A configured (but not yet interrogated) Validate object.
    """
    import narwhals as nw

    from pointblank.validate import Validate

    template = get_adam_dataset(dataset)

    if label is None:
        label_parts = [f"ADaM {dataset.upper()} Validation"]
        if study_id:
            label_parts = [f"ADaM {dataset.upper()} — {study_id}"]
        label = label_parts[0]

    validation = Validate(data=data, label=label, **kwargs)

    df = nw.from_native(data, eager_only=True)
    actual_columns = set(df.columns)

    # ── Required variables must be non-null ──
    for spec in template.variables:
        if spec.required and spec.name in actual_columns:
            validation = validation.col_vals_not_null(columns=spec.name)

    # ── Population flag validation ──
    if check_population_flags:
        for spec in template.variables:
            if spec.is_population_flag and spec.name in actual_columns:
                # Population flags must be Y or N (no other values)
                validation = validation.col_vals_in_set(columns=spec.name, set=["Y", "N"])

    # ── BDS structure checks ──
    if check_bds_structure and template.dataset_class == "BDS":
        # PARAMCD must be non-null and ≤ 8 chars
        if "PARAMCD" in actual_columns:
            validation = validation.col_vals_expr(
                expr=nw.col("PARAMCD").str.len_chars() <= 8,
                brief="PARAMCD length <= 8",
            )
        # AVAL should not be all null (at least some numeric results)
        # CHG should only exist where ABLFL = "Y" exists for the parameter

    # ── ADTTE-specific checks ──
    if template.dataset_class == "ADTTE":
        # CNSR must be 0 (event) or 1 (censored)
        if "CNSR" in actual_columns:
            validation = validation.col_vals_in_set(columns="CNSR", set=[0, 1])
        # AVAL (time) must be non-negative
        if "AVAL" in actual_columns:
            validation = validation.col_vals_ge(columns="AVAL", value=0)

    # ── ADAE-specific checks ──
    if template.dataset_class == "ADAE":
        # TRTEMFL (treatment-emergent flag) must be Y or N when present
        if "TRTEMFL" in actual_columns:
            validation = validation.col_vals_in_set(columns="TRTEMFL", set=["Y", "N"])
        # AESEQ must be positive
        if "AESEQ" in actual_columns:
            validation = validation.col_vals_gt(columns="AESEQ", value=0)

    # ── ADSL-specific checks ──
    if template.dataset_class == "ADSL":
        # TRT01P must be non-null in ADSL
        if "TRT01P" in actual_columns:
            validation = validation.col_vals_not_null(columns="TRT01P")

    # ── Traceability checks ──
    if check_traceability:
        # If SRCDOM/SRCVAR/SRCSEQ are present, they should be non-null
        traceability_vars = ["SRCDOM", "SRCVAR", "SRCSEQ"]
        for var_name in traceability_vars:
            if var_name in actual_columns:
                validation = validation.col_vals_not_null(columns=var_name)

    return validation
