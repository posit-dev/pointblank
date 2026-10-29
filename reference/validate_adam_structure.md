# validate_adam_structure()

Check a dataset's variables against an ADaM dataset template.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_adam_templates.py#L1033-L1187)

``` python
validate_adam_structure(
    data,
    dataset,
    strict=False,
)
```

The [validate_adam_structure()](../reference/validate_adam_structure.md#pointblank.validate_adam_structure) function is a quick check of a dataset's structure, run before (or instead of) a full validation with [`validate_adam()`](../reference/validate_adam.md#pointblank.validate_adam). It checks that every required variable of the template is a column of the dataset, and lists the population flags that the dataset has. It returns the findings as a dictionary straight away, without building a [`Validate`](../reference/Validate.md#pointblank.Validate) plan.

For an ADSL dataset with no population flags at all, an issue is added to `"issues"`, but the dataset is still `"valid"`. With `strict=True`, two more things are reported: the conditionally required variables (core designation `"Cond"`) that are missing, and the columns that aren't variables in the template. These don't affect `"valid"` either, since whether a conditional variable is needed depends on the study, and the templates only cover the common variables of each type of dataset.

The column order, the data types and the values aren't checked.

## Parameters

`data: Any`  
The dataset to check, as a Polars or Pandas DataFrame.

`dataset: str`  
The name of the ADaM dataset template: `"ADSL"`, `"BDS"`, `"ADAE"` or `"ADTTE"`. This is case-insensitive.

`strict: bool = False`  
Whether to also report missing conditional variables and columns that aren't in the template. The default is `False`.

## Returns

`dict`  
A dictionary with these keys:

- `"dataset"`: the template name, in upper case.
- `"dataset_class"`: the ADaM class of the template.
- `"valid"`: `True` if no required variable is missing.
- `"missing_required"`: the names of the required variables that aren't columns of the data.
- `"missing_conditional"`: the names of the conditional variables that aren't columns of the data (filled in only with `strict=True`).
- `"unknown_variables"`: the columns that aren't variables in the template, in the order of the data's columns (filled in only with `strict=True`).
- `"population_flags_found"`: the population flags that are columns of the data.
- `"issues"`: a message for each of the problems above. With `strict=True`, the message for a missing conditional variable includes its condition.

## Raises

`KeyError`  
If there is no template with that name.

## Examples

Let's check a small ADSL dataset that lacks the site identifier, `SITEID`, a required ADSL variable:

``` python
import pointblank as pb
import polars as pl

adsl = pl.DataFrame(
    {
        "STUDYID": ["STUDY01"] * 3,
        "USUBJID": ["STUDY01-001", "STUDY01-002", "STUDY01-003"],
        "SUBJID": ["001", "002", "003"],
        "TRT01P": ["Drug A", "Placebo", "Drug A"],
        "SAFFL": ["Y", "Y", "N"],
        "BMIBL": [24.1, 31.7, 22.9],
    }
)

result = pb.validate_adam_structure(adsl, dataset="ADSL")

print(result["valid"])
print(result["missing_required"])
print(result["population_flags_found"])
```

    False
    ['SITEID']
    ['SAFFL']

The dataset isn't valid because of the missing `SITEID`, and it has one population flag, `SAFFL`. In strict mode, the conditional variables and the unknown columns are reported too:

``` python
result = pb.validate_adam_structure(adsl, dataset="ADSL", strict=True)

print(result["missing_conditional"])
print(result["unknown_variables"])
```

    ['TRT01A', 'TRT01PN', 'TRTSDT', 'TRTEDT', 'ITTFL', 'AGE', 'AGEU', 'SEX', 'RACE']
    ['BMIBL']

The baseline BMI column, `BMIBL`, isn't in the template. Since ADaM allows a sponsor to add variables like this one, it's reported only so that it can be reviewed.

## See Also

[validate_adam()](../reference/validate_adam.md#pointblank.validate_adam)  
Build a validation plan from an ADaM dataset template.

[get_adam_dataset()](../reference/get_adam_dataset.md#pointblank.get_adam_dataset)  
Get the template that the dataset is checked against.

[validate_sdtm_structure()](../reference/validate_sdtm_structure.md#pointblank.validate_sdtm_structure)  
The equivalent for SDTM datasets.
