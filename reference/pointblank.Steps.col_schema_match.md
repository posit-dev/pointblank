# col_schema_match()

Add a `col_schema_match()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1829-L1874)

``` python
col_schema_match(
    schema,
    complete=True,
    in_order=True,
    case_sensitive_colnames=True,
    case_sensitive_dtypes=True,
    full_match_dtypes=True,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_schema_match()`](../reference/Validate.col_schema_match.md#pointblank.Validate.col_schema_match), which checks whether the table's columns and their types match a schema. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
