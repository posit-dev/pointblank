# col_vals_within_spec()

Add a `col_vals_within_spec()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L960-L1003)

``` python
col_vals_within_spec(
    columns,
    spec,
    na_pass=False,
    missing=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_within_spec()`](../reference/Validate.col_vals_within_spec.md#pointblank.Validate.col_vals_within_spec), which checks whether column values fit a specification such as an email address, URL, or postal code. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
