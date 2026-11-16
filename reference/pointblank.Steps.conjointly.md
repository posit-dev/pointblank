# conjointly()

Add a `conjointly()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L2152-L2189)

``` python
conjointly(
    *exprs,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.conjointly()`](../reference/Validate.conjointly.md#pointblank.Validate.conjointly), which checks whether several row-wise conditions all hold for each row. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

The expressions are recorded together in the step's `exprs=` argument and passed back as positional arguments when the step is applied.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
