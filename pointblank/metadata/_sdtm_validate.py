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

    The SDTM domain templates (see [`get_sdtm_domain()`](`pointblank.get_sdtm_domain`)) have their
    own classes, [`SDTMDomainTemplate`](`pointblank.SDTMDomainTemplate`) and
    [`SDTMVariableSpec`](`pointblank.SDTMVariableSpec`). The `sdtm_to_metadata()` function converts
    a template to a [`MetadataImport`](`pointblank.MetadataImport`), the same object that
    [`import_metadata()`](`pointblank.import_metadata`) returns for a metadata file. That gives the
    template the methods of a `MetadataImport`, such as
    [`to_schema()`](`pointblank.MetadataImport.to_schema`) and
    [`get_variable()`](`pointblank.MetadataImport.get_variable`), and lets it be exported with
    [`export_metadata()`](`pointblank.export_metadata`).

    Each variable in the template becomes a [`VariableMetadata`](`pointblank.VariableMetadata`)
    with the same name, label, role, `required` flag, maximum length and controlled terminology
    name. SDTM's `"Num"` type becomes the `"Float64"` dtype and `"Char"` becomes `"String"`. The
    dataset name and label come from the domain, and the source version is `"IG 3.4"`. The template
    names the codelist of a variable (as in `controlled_term="SEX"`) but doesn't contain the
    codelist itself, so the result has no codelists. Calling
    `import_metadata(format="cdisc_sdtm", domain=...)` gives the same result.

    Parameters
    ----------
    domain
        The SDTM domain code, such as `"DM"`, `"AE"` or `"LB"`. This is case-insensitive.
    study_id
        A study identifier to record in the metadata's `study_id`. The default is `None`.

    Returns
    -------
    MetadataImport
        The metadata for the domain, with `source_format="cdisc_sdtm"`.

    Raises
    ------
    KeyError
        If there is no template for the domain. See
        [`list_sdtm_domains()`](`pointblank.list_sdtm_domains`) for the domains that have one.

    Examples
    --------
    Let's convert the template for the Demographics (DM) domain:

    ```{python}
    import pointblank as pb

    dm_meta = pb.sdtm_to_metadata("DM", study_id="STUDY01")

    print(dm_meta.dataset_label)
    print(dm_meta.source_version)
    print(dm_meta.variable_names[:6])
    ```

    The metadata has all 26 variables of the template, in the template's order. Each variable keeps
    what the template says about it:

    ```{python}
    sex = dm_meta.get_variable("SEX")

    print(sex.dtype, sex.required, sex.max_length, sex.controlled_term)
    ```

    `SEX` is a required character variable of at most two characters, with values from the `SEX`
    codelist of CDISC Controlled Terminology. The metadata can be turned into a
    [`Schema`](`pointblank.Schema`) for use in
    [`Validate.col_schema_match()`](`pointblank.Validate.col_schema_match`):

    ```{python}
    schema = dm_meta.to_schema()

    schema.columns[:4]
    ```

    The schema lists every variable of the template, including the expected and permissible ones.
    A real DM dataset usually has only some of them, so a schema check against all 26 is stricter
    than SDTM requires. For checking a dataset against the template,
    [`validate_sdtm()`](`pointblank.validate_sdtm`) (which only checks the variables that are
    present) is usually the better fit.

    See Also
    --------
    get_sdtm_domain : Get the template that is converted.
    validate_sdtm : Build a validation plan from an SDTM domain template.
    adam_to_metadata : The equivalent for ADaM dataset templates.
    MetadataImport : The object that is returned.
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
    """Build a validation plan for a dataset from an SDTM domain template.

    The `validate_sdtm()` function reads the template for an SDTM domain (see
    [`get_sdtm_domain()`](`pointblank.get_sdtm_domain`)) and turns it into a
    [`Validate`](`pointblank.Validate`) plan for a dataset of that domain. The plan isn't
    interrogated, so more steps can be added to it before calling
    [`interrogate()`](`pointblank.Validate.interrogate`).

    The steps are generated in this order, for the template variables that are columns of the
    data:

    1. [`col_vals_not_null()`](`pointblank.Validate.col_vals_not_null`) for each required variable
    (one with a core designation of `"Req"`).
    2. [`col_vals_in_set()`](`pointblank.Validate.col_vals_in_set`) to check that `DOMAIN` holds
    only the domain code.
    3. [`col_vals_gt()`](`pointblank.Validate.col_vals_gt`) to check that the sequence number (as in
    `AESEQ`) is greater than `0`. DM has one record per subject and so has no sequence number.
    4. If `check_lengths=True`, a [`col_vals_expr()`](`pointblank.Validate.col_vals_expr`) step for
    each character variable, checking that its values are no longer than the template's maximum
    length. Null values pass.
    5. If `check_dates=True`, a [`col_vals_regex()`](`pointblank.Validate.col_vals_regex`) step for
    each `--DTC` timing variable (such as `RFSTDTC` or `AESTDTC`), checking that its values are
    ISO 8601 dates or date-times. Null values pass.

    Columns that aren't in the template are left alone, and the data types of the columns aren't
    checked. A template variable that isn't a column of the data gets no steps, so a dataset that
    lacks a required variable isn't reported for it. Use
    [`validate_sdtm_structure()`](`pointblank.validate_sdtm_structure`) to find missing variables,
    or add a [`col_exists()`](`pointblank.Validate.col_exists`) step to the plan (as in the second
    example below).

    The date check accepts the right-truncated forms that SDTM uses for partial dates: a year
    (`2024`), a year and month (`2024-03`), a full date (`2024-03-15`), and a date with a time to
    the hour, minute or second (`2024-03-15T10:30:00`). Dates in other formats (such as
    `03/15/2024`), fractional seconds, time zones and dates with missing middle components (such
    as `2024---15`) fail. The check is on the format only, so an impossible date such as
    `2024-13-45` passes.

    Parameters
    ----------
    data
        The dataset to validate, as a Polars or Pandas DataFrame.
    domain
        The SDTM domain code, such as `"DM"`, `"AE"` or `"LB"`. This is case-insensitive. See
        [`list_sdtm_domains()`](`pointblank.list_sdtm_domains`) for the domains that have a
        template.
    study_id
        A study identifier to use in the default label, which becomes
        `"SDTM {domain} — {study_id}"`. It is ignored if `label=` is given. The default is `None`.
    check_dates
        Whether to add the ISO 8601 format checks for the `--DTC` timing variables. The default is
        `True`.
    check_lengths
        Whether to add the maximum length checks for the character variables. The default is
        `True`.
    label
        A label for the validation. The default is `"SDTM {domain} Validation"`.
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
        If there is no template for the domain.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```
    Here is a small Demographics (DM) dataset with a few problems: the third row has the wrong
    `DOMAIN` and no `USUBJID`, the second row's `RFSTDTC` isn't an ISO 8601 date, and the fourth
    row's `SEX` is longer than the two characters that SDTM allows:

    ```{python}
    import pointblank as pb
    import polars as pl

    dm = pl.DataFrame(
        {
            "STUDYID": ["STUDY01"] * 4,
            "DOMAIN": ["DM", "DM", "AE", "DM"],
            "USUBJID": ["STUDY01-001", "STUDY01-002", None, "STUDY01-004"],
            "SUBJID": ["001", "002", "003", "004"],
            "RFSTDTC": ["2024-01-15", "03/15/2024", "2024-02", "2024-02-10T09:30"],
            "SEX": ["M", "F", "F", "MALE"],
        }
    )

    validation = pb.validate_sdtm(dm, domain="DM").interrogate()

    validation
    ```

    The plan starts with the null checks for the five required variables in the data, then checks
    `DOMAIN`, the lengths of all six columns, and the format of `RFSTDTC`. Four steps have a
    failing test unit, one for each problem. The partial date `2024-02` and the date-time
    `2024-02-10T09:30` both pass the date check, since SDTM allows dates to be truncated.

    The plan doesn't report the required DM variables that the dataset lacks, such as `SITEID` and
    `COUNTRY`. Since the plan isn't interrogated yet, a
    [`col_exists()`](`pointblank.Validate.col_exists`) step for the template's required variables
    can be added to it. This example uses an Adverse Events (AE) dataset and leaves out the length
    checks to keep the plan short:

    ```{python}
    ae = pl.DataFrame(
        {
            "STUDYID": ["STUDY01"] * 3,
            "DOMAIN": ["AE"] * 3,
            "USUBJID": ["STUDY01-001", "STUDY01-001", "STUDY01-002"],
            "AESEQ": [1, 2, 0],
            "AETERM": ["HEADACHE", "NAUSEA", "RASH"],
            "AESTDTC": ["2024-01-20", "2024-02-03", "2024-02"],
        }
    )

    required = pb.get_sdtm_domain("AE").required_variables

    validation = (
        pb.validate_sdtm(ae, domain="AE", check_lengths=False, tbl_name="ae")
        .col_exists(columns=required)
        .interrogate()
    )

    validation
    ```

    The `col_vals_gt()` step on `AESEQ` fails for the third row, whose sequence number is `0`. The
    `col_exists()` steps at the end show that `AEDECOD`, a required AE variable, is missing from
    the dataset. The `tbl_name=` argument was passed on to `Validate`.

    See Also
    --------
    validate_sdtm_structure : Check a dataset's variables against an SDTM template.
    get_sdtm_domain : Get the template that the plan is built from.
    validate_adam : The equivalent for ADaM datasets.
    validate_sdtmig : Check SDTM datasets against the SDTMIG conformance rules.
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
                # Null values pass: whether a variable may be null is a separate check
                length_expr = (nw.col(spec.name).str.len_chars() <= spec.max_length) | nw.col(
                    spec.name
                ).is_null()
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
