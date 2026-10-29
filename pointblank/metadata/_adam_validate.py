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
    """Convert an ADaM dataset template to a `MetadataImport` object.

    The ADaM dataset templates (see [`get_adam_dataset()`](`pointblank.get_adam_dataset`)) have
    their own classes, [`ADaMDatasetTemplate`](`pointblank.ADaMDatasetTemplate`) and
    [`ADaMVariableSpec`](`pointblank.ADaMVariableSpec`). The `adam_to_metadata()` function converts
    a template to a [`MetadataImport`](`pointblank.MetadataImport`), the same object that
    [`import_metadata()`](`pointblank.import_metadata`) returns for a metadata file. That gives the
    template the methods of a `MetadataImport`, such as
    [`to_schema()`](`pointblank.MetadataImport.to_schema`) and
    [`get_variable()`](`pointblank.MetadataImport.get_variable`), and lets it be exported with
    [`export_metadata()`](`pointblank.export_metadata`).

    Each variable in the template becomes a [`VariableMetadata`](`pointblank.VariableMetadata`)
    with the same name, label, `required` flag, maximum length and controlled terminology name.
    The variable's core designation (`"Req"`, `"Cond"` or `"Perm"`) goes in `cdisc_role`, and its
    expected source (such as `"SDTM.DM"`) goes in `adam_derivation`. ADaM's `"Num"` type becomes
    the `"Float64"` dtype and `"Char"` becomes `"String"`, so numeric dates such as `TRTSDT` (which
    ADaM stores as numbers of days) are `"Float64"`. The source version is `"IG 1.1"`. Calling
    `import_metadata(format="cdisc_adam", dataset=...)` gives the same result.

    Parameters
    ----------
    dataset
        The name of the ADaM dataset template: `"ADSL"`, `"BDS"`, `"ADAE"` or `"ADTTE"`. This is
        case-insensitive.
    study_id
        A study identifier to record in the metadata's `study_id`. The default is `None`.

    Returns
    -------
    MetadataImport
        The metadata for the dataset, with `source_format="cdisc_adam"`.

    Raises
    ------
    KeyError
        If there is no template with that name.

    Examples
    --------
    Let's convert the template for the subject-level analysis dataset, ADSL:

    ```{python}
    import pointblank as pb

    adsl_meta = pb.adam_to_metadata("ADSL", study_id="STUDY01")

    print(adsl_meta.dataset_label)
    print(len(adsl_meta.variables), "variables")
    ```

    Each variable keeps what the template says about it, with the ADaM core designation in
    `cdisc_role`:

    ```{python}
    for name in ["USUBJID", "TRT01P", "SAFFL", "TRTSDT"]:
        var = adsl_meta.get_variable(name)
        print(f"{name:8}{var.dtype:8}{var.cdisc_role:5}{var.adam_derivation}")
    ```

    `USUBJID` and `TRT01P` are required, while `SAFFL` (the safety population flag) and `TRTSDT`
    (the treatment start date) are conditionally required. `USUBJID` comes from the SDTM
    Demographics (DM) domain and `TRTSDT` is derived from Exposure (EX). Only the required
    variables get `required=True`, so the conditional ones don't get the null checks that
    [`to_validate()`](`pointblank.MetadataImport.to_validate`) adds for required variables.

    See Also
    --------
    get_adam_dataset : Get the template that is converted.
    validate_adam : Build a validation plan from an ADaM dataset template.
    sdtm_to_metadata : The equivalent for SDTM domain templates.
    MetadataImport : The object that is returned.
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
    """Build a validation plan for a dataset from an ADaM dataset template.

    The `validate_adam()` function reads an ADaM dataset template (see
    [`get_adam_dataset()`](`pointblank.get_adam_dataset`)) and turns it into a
    [`Validate`](`pointblank.Validate`) plan for an analysis dataset of that type. The plan isn't
    interrogated, so more steps can be added to it before calling
    [`interrogate()`](`pointblank.Validate.interrogate`).

    The steps are generated in this order, for the template variables that are columns of the
    data:

    1. [`col_vals_not_null()`](`pointblank.Validate.col_vals_not_null`) for each required variable
    (one with a core designation of `"Req"`).
    2. If `check_population_flags=True`, a
    [`col_vals_in_set()`](`pointblank.Validate.col_vals_in_set`) step for each population flag
    (such as `SAFFL` or `ITTFL`), checking that its values are `"Y"` or `"N"`. A null value fails,
    since ADaM doesn't allow nulls in population flags.
    3. Checks for the type of dataset:
        - BDS (if `check_bds_structure=True`): a
          [`col_vals_expr()`](`pointblank.Validate.col_vals_expr`) step checking that `PARAMCD`
          is no longer than eight characters. Null values pass.
        - ADTTE: `CNSR` must be `0` (an event) or `1` (censored), and `AVAL` (the time to the
          event) must be at least `0`.
        - ADAE: `TRTEMFL` must be `"Y"` or `"N"`, and `AESEQ` must be greater than `0`.
        - ADSL: `TRT01P` must not be null. This is already one of the required-variable checks,
          so it isn't added a second time.
    4. If `check_traceability=True`, a null check for each of the traceability variables `SRCDOM`,
    `SRCVAR` and `SRCSEQ`, which record the SDTM domain, variable and record that a value came
    from.

    Columns that aren't in the template are left alone, and the data types of the columns aren't
    checked. A template variable that isn't a column of the data gets no steps, so a dataset that
    lacks a required variable isn't reported for it. Use
    [`validate_adam_structure()`](`pointblank.validate_adam_structure`) to find missing variables,
    or add a [`col_exists()`](`pointblank.Validate.col_exists`) step to the plan.

    Parameters
    ----------
    data
        The dataset to validate, as a Polars or Pandas DataFrame.
    dataset
        The name of the ADaM dataset template: `"ADSL"`, `"BDS"`, `"ADAE"` or `"ADTTE"`. This is
        case-insensitive. Use `"BDS"` for any dataset with the Basic Data Structure, such as ADVS
        or ADLB.
    study_id
        A study identifier to use in the default label, which becomes
        `"ADaM {dataset} — {study_id}"`. It is ignored if `label=` is given. The default is `None`.
    check_population_flags
        Whether to check that the population flags hold only `"Y"` and `"N"`. The default is `True`.
    check_bds_structure
        Whether to add the `PARAMCD` length check for a BDS dataset. It has no effect for the other
        templates. The default is `True`.
    check_traceability
        Whether to add the null checks for the traceability variables. The default is `True`.
    label
        A label for the validation. The default is `"ADaM {dataset} Validation"`.
    **kwargs
        Other arguments for [`Validate`](`pointblank.Validate`), such as `tbl_name=` and
        `thresholds=`.

    Returns
    -------
    Validate
        The validation plan, not yet interrogated.

    Raises
    ------
    KeyError
        If there is no template with that name.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```
    Here is a small subject-level analysis dataset (ADSL) with two problems in its population
    flags: the fourth subject has no safety flag (`SAFFL`) and an intent-to-treat flag (`ITTFL`)
    of `"U"`:

    ```{python}
    import pointblank as pb
    import polars as pl

    adsl = pl.DataFrame(
        {
            "STUDYID": ["STUDY01"] * 4,
            "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003", "STUDY01-004"],
            "SUBJID": ["001", "002", "003", "004"],
            "SITEID": ["SITE01", "SITE01", "SITE02", "SITE02"],
            "TRT01P": ["Drug A", "Placebo", "Drug A", "Placebo"],
            "SAFFL": ["Y", "Y", "N", None],
            "ITTFL": ["Y", "Y", "Y", "U"],
        }
    )

    validation = pb.validate_adam(adsl, dataset="ADSL").interrogate()

    validation
    ```

    The five required ADSL variables get null checks, which all pass. The two population flags
    are checked for `"Y"` and `"N"`, and each of those steps has one failing test unit.

    For a time-to-event dataset (ADTTE), the plan also checks the censoring indicator and the
    time to the event. Here the second subject has a negative time, and the third has a `CNSR` of
    `2`:

    ```{python}
    adtte = pl.DataFrame(
        {
            "STUDYID": ["STUDY01"] * 3,
            "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
            "PARAMCD": ["OS"] * 3,
            "PARAM": ["Overall Survival (days)"] * 3,
            "AVAL": [120.0, -3.0, 365.0],
            "STARTDT": [19723.0, 19730.0, 19741.0],
            "ADT": [19843.0, 19727.0, 20106.0],
            "CNSR": [0, 1, 2],
        }
    )

    validation = pb.validate_adam(adtte, dataset="ADTTE", tbl_name="adtte").interrogate()

    validation
    ```

    The last two steps are the ADTTE checks, and each one finds its problem. The dates
    `STARTDT` and `ADT` are numbers of days since 1960-01-01, as is usual in ADaM datasets (which
    are often stored as SAS files).

    See Also
    --------
    validate_adam_structure : Check a dataset's variables against an ADaM template.
    get_adam_dataset : Get the template that the plan is built from.
    validate_sdtm : The equivalent for SDTM datasets.
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
        # PARAMCD must be ≤ 8 chars (nulls pass here, since they're caught by the null check)
        if "PARAMCD" in actual_columns:
            validation = validation.col_vals_expr(
                expr=(nw.col("PARAMCD").str.len_chars() <= 8) | nw.col("PARAMCD").is_null(),
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
        # TRT01P must be non-null in ADSL (skipped if the required-variable checks already did it)
        if "TRT01P" in actual_columns and "TRT01P" not in template.required_variables:
            validation = validation.col_vals_not_null(columns="TRT01P")

    # ── Traceability checks ──
    if check_traceability:
        # If SRCDOM/SRCVAR/SRCSEQ are present, they should be non-null
        traceability_vars = ["SRCDOM", "SRCVAR", "SRCSEQ"]
        for var_name in traceability_vars:
            if var_name in actual_columns:
                validation = validation.col_vals_not_null(columns=var_name)

    return validation
