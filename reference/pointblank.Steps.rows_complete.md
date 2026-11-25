# rows_complete()

Add a `rows_complete()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1747-L1784)

``` python
rows_complete(
    columns_subset=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.rows_complete()`](../reference/Validate.rows_complete.md#pointblank.Validate.rows_complete), which checks whether rows are complete, with no missing values across all columns or a subset of them. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
