# col_vals_no_pii()

Add a `col_vals_no_pii()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1325-L1367)

``` python
col_vals_no_pii(
    columns,
    entities=("email", "phone", "credit_card"),
    na_pass=True,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_no_pii()`](../reference/Validate.col_vals_no_pii.md#pointblank.Validate.col_vals_no_pii), which checks whether text has no personal information (adding a validation step for each kind of information when the collection is added to a validation). It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
