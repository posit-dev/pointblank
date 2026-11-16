# rows_near_distinct()

Add a `rows_near_distinct()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1650-L1695)

``` python
rows_near_distinct(
    columns,
    method="minhash",
    threshold=0.85,
    ngram=5,
    num_perm=128,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.rows_near_distinct()`](../reference/Validate.rows_near_distinct.md#pointblank.Validate.rows_near_distinct), which checks that no row's text is a near duplicate of another's. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
