# col_vals_in_language()

Add a `col_vals_in_language()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1223-L1274)

``` python
col_vals_in_language(
    columns,
    expected=None,
    expected_column=None,
    min_confidence=0.5,
    backend="langid",
    model=None,
    na_pass=False,
    cache=True,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_in_language()`](../reference/Validate.col_vals_in_language.md#pointblank.Validate.col_vals_in_language), which checks whether text is in the expected language. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
