# validate_sdtm()

Build a validation plan for a dataset from an SDTM domain template.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_sdtm_validate.py#L159-L374)

``` python
validate_sdtm(
    data,
    domain,
    study_id=None,
    check_dates=True,
    check_lengths=True,
    label=None,
    **kwargs,
)
```

The [validate_sdtm()](../reference/validate_sdtm.md#pointblank.validate_sdtm) function reads the template for an SDTM domain (see [`get_sdtm_domain()`](../reference/get_sdtm_domain.md#pointblank.get_sdtm_domain)) and turns it into a [`Validate`](../reference/Validate.md#pointblank.Validate) plan for a dataset of that domain. The plan isn't interrogated, so more steps can be added to it before calling [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate).

The steps are generated in this order, for the template variables that are columns of the data:

1.  [`col_vals_not_null()`](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) for each required variable (one with a core designation of `"Req"`).
2.  [`col_vals_in_set()`](../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set) to check that `DOMAIN` holds only the domain code.
3.  [`col_vals_gt()`](../reference/Validate.col_vals_gt.md#pointblank.Validate.col_vals_gt) to check that the sequence number (as in `AESEQ`) is greater than `0`. DM has one record per subject and so has no sequence number.
4.  If `check_lengths=True`, a [`col_vals_expr()`](../reference/Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr) step for each character variable, checking that its values are no longer than the template's maximum length. Null values pass.
5.  If `check_dates=True`, a [`col_vals_regex()`](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) step for each `--DTC` timing variable (such as `RFSTDTC` or `AESTDTC`), checking that its values are ISO 8601 dates or date-times. Null values pass.

Columns that aren't in the template are left alone, and the data types of the columns aren't checked. A template variable that isn't a column of the data gets no steps, so a dataset that lacks a required variable isn't reported for it. Use [`validate_sdtm_structure()`](../reference/validate_sdtm_structure.md#pointblank.validate_sdtm_structure) to find missing variables, or add a [`col_exists()`](../reference/Validate.col_exists.md#pointblank.Validate.col_exists) step to the plan (as in the second example below).

The date check accepts the right-truncated forms that SDTM uses for partial dates: a year (`2024`), a year and month (`2024-03`), a full date (`2024-03-15`), and a date with a time to the hour, minute or second (`2024-03-15T10:30:00`). Dates in other formats (such as `03/15/2024`), fractional seconds, time zones and dates with missing middle components (such as `2024---15`) fail. The check is on the format only, so an impossible date such as `2024-13-45` passes.

## Parameters

`data: Any`  
The dataset to validate, as a Polars or Pandas DataFrame.

`domain: str`  
The SDTM domain code, such as `"DM"`, `"AE"` or `"LB"`. This is case-insensitive. See [`list_sdtm_domains()`](../reference/list_sdtm_domains.md#pointblank.list_sdtm_domains) for the domains that have a template.

`study_id: str | None = None`  
A study identifier to use in the default label, which becomes `"SDTM {domain} -- {study_id}"`. It is ignored if `label=` is given. The default is `None`.

`check_dates: bool = True`  
Whether to add the ISO 8601 format checks for the `--DTC` timing variables. The default is `True`.

`check_lengths: bool = True`  
Whether to add the maximum length checks for the character variables. The default is `True`.

`label: str | None = None`  
A label for the validation. The default is `"SDTM {domain} Validation"`.

`**kwargs: Any`  
Other arguments for [`Validate`](../reference/Validate.md#pointblank.Validate), such as `tbl_name=` and `thresholds=`.

## Returns

`Validate`  
The validation plan, not yet interrogated.

## Raises

`KeyError`  
If there is no template for the domain.

## Examples

Here is a small Demographics (DM) dataset with a few problems: the third row has the wrong `DOMAIN` and no `USUBJID`, the second row's `RFSTDTC` isn't an ISO 8601 date, and the fourth row's `SEX` is longer than the two characters that SDTM allows:

``` python
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

[TABLE]

The plan starts with the null checks for the five required variables in the data, then checks `DOMAIN`, the lengths of all six columns, and the format of `RFSTDTC`. Four steps have a failing test unit, one for each problem. The partial date `2024-02` and the date-time `2024-02-10T09:30` both pass the date check, since SDTM allows dates to be truncated.

The plan doesn't report the required DM variables that the dataset lacks, such as `SITEID` and `COUNTRY`. Since the plan isn't interrogated yet, a [`col_exists()`](../reference/Validate.col_exists.md#pointblank.Validate.col_exists) step for the template's required variables can be added to it. This example uses an Adverse Events (AE) dataset and leaves out the length checks to keep the plan short:

``` python
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

[TABLE]

The `col_vals_gt()` step on `AESEQ` fails for the third row, whose sequence number is `0`. The `col_exists()` steps at the end show that `AEDECOD`, a required AE variable, is missing from the dataset. The `tbl_name=` argument was passed on to [Validate](../reference/Validate.md#pointblank.Validate).

## See Also

[validate_sdtm_structure()](../reference/validate_sdtm_structure.md#pointblank.validate_sdtm_structure)  
Check a dataset's variables against an SDTM template.

[get_sdtm_domain()](../reference/get_sdtm_domain.md#pointblank.get_sdtm_domain)  
Get the template that the plan is built from.

[validate_adam()](../reference/validate_adam.md#pointblank.validate_adam)  
The equivalent for ADaM datasets.

[validate_sdtmig()](../reference/validate_sdtmig.md#pointblank.validate_sdtmig)  
Check SDTM datasets against the SDTMIG conformance rules.
