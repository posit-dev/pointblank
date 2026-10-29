# validate_adam()

Build a validation plan for a dataset from an ADaM dataset template.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_adam_validate.py#L126-L347)

``` python
validate_adam(
    data,
    dataset,
    study_id=None,
    check_population_flags=True,
    check_bds_structure=True,
    check_traceability=True,
    label=None,
    **kwargs,
)
```

The [validate_adam()](../reference/validate_adam.md#pointblank.validate_adam) function reads an ADaM dataset template (see [`get_adam_dataset()`](../reference/get_adam_dataset.md#pointblank.get_adam_dataset)) and turns it into a [`Validate`](../reference/Validate.md#pointblank.Validate) plan for an analysis dataset of that type. The plan isn't interrogated, so more steps can be added to it before calling [`interrogate()`](../reference/Validate.interrogate.md#pointblank.Validate.interrogate).

The steps are generated in this order, for the template variables that are columns of the data:

1.  [`col_vals_not_null()`](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) for each required variable (one with a core designation of `"Req"`).
2.  If `check_population_flags=True`, a [`col_vals_in_set()`](../reference/Validate.col_vals_in_set.md#pointblank.Validate.col_vals_in_set) step for each population flag (such as `SAFFL` or `ITTFL`), checking that its values are `"Y"` or `"N"`. A null value fails, since ADaM doesn't allow nulls in population flags.
3.  Checks for the type of dataset:
    - BDS (if `check_bds_structure=True`): a [`col_vals_expr()`](../reference/Validate.col_vals_expr.md#pointblank.Validate.col_vals_expr) step checking that `PARAMCD` is no longer than eight characters. Null values pass.
    - ADTTE: `CNSR` must be `0` (an event) or `1` (censored), and `AVAL` (the time to the event) must be at least `0`.
    - ADAE: `TRTEMFL` must be `"Y"` or `"N"`, and `AESEQ` must be greater than `0`.
    - ADSL: `TRT01P` must not be null. This is already one of the required-variable checks, so it isn't added a second time.
4.  If `check_traceability=True`, a null check for each of the traceability variables `SRCDOM`, `SRCVAR` and `SRCSEQ`, which record the SDTM domain, variable and record that a value came from.

Columns that aren't in the template are left alone, and the data types of the columns aren't checked. A template variable that isn't a column of the data gets no steps, so a dataset that lacks a required variable isn't reported for it. Use [`validate_adam_structure()`](../reference/validate_adam_structure.md#pointblank.validate_adam_structure) to find missing variables, or add a [`col_exists()`](../reference/Validate.col_exists.md#pointblank.Validate.col_exists) step to the plan.

## Parameters

`data: Any`  
The dataset to validate, as a Polars or Pandas DataFrame.

`dataset: str`  
The name of the ADaM dataset template: `"ADSL"`, `"BDS"`, `"ADAE"` or `"ADTTE"`. This is case-insensitive. Use `"BDS"` for any dataset with the Basic Data Structure, such as ADVS or ADLB.

`study_id: str | None = None`  
A study identifier to use in the default label, which becomes `"ADaM {dataset} -- {study_id}"`. It is ignored if `label=` is given. The default is `None`.

`check_population_flags: bool = True`  
Whether to check that the population flags hold only `"Y"` and `"N"`. The default is `True`.

`check_bds_structure: bool = True`  
Whether to add the `PARAMCD` length check for a BDS dataset. It has no effect for the other templates. The default is `True`.

`check_traceability: bool = True`  
Whether to add the null checks for the traceability variables. The default is `True`.

`label: str | None = None`  
A label for the validation. The default is `"ADaM {dataset} Validation"`.

`**kwargs: Any`  
Other arguments for [`Validate`](../reference/Validate.md#pointblank.Validate), such as `tbl_name=` and `thresholds=`.

## Returns

`Validate`  
The validation plan, not yet interrogated.

## Raises

`KeyError`  
If there is no template with that name.

## Examples

Here is a small subject-level analysis dataset (ADSL) with two problems in its population flags: the fourth subject has no safety flag (`SAFFL`) and an intent-to-treat flag (`ITTFL`) of `"U"`:

``` python
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

[TABLE]

The five required ADSL variables get null checks, which all pass. The two population flags are checked for `"Y"` and `"N"`, and each of those steps has one failing test unit.

For a time-to-event dataset (ADTTE), the plan also checks the censoring indicator and the time to the event. Here the second subject has a negative time, and the third has a `CNSR` of `2`:

``` python
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

[TABLE]

The last two steps are the ADTTE checks, and each one finds its problem. The dates `STARTDT` and `ADT` are numbers of days since 1960-01-01, as is usual in ADaM datasets (which are often stored as SAS files).

## See Also

[validate_adam_structure()](../reference/validate_adam_structure.md#pointblank.validate_adam_structure)  
Check a dataset's variables against an ADaM template.

[get_adam_dataset()](../reference/get_adam_dataset.md#pointblank.get_adam_dataset)  
Get the template that the plan is built from.

[validate_sdtm()](../reference/validate_sdtm.md#pointblank.validate_sdtm)  
The equivalent for SDTM datasets.
