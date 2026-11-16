# col_missing_consistent()

Add a `col_missing_consistent()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1787-L1828)

``` python
col_missing_consistent(
    columns,
    missing,
    when_reason,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_missing_consistent()`](../reference/Validate.col_missing_consistent.md#pointblank.Validate.col_missing_consistent), which checks whether related columns share a consistent pattern of missingness for a given reason. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
