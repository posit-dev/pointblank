# rows_not_contaminated()

Add a `rows_not_contaminated()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1696-L1745)

``` python
rows_not_contaminated(
    columns,
    reference,
    reference_columns=None,
    method="ngram",
    ngram=None,
    threshold=0.8,
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

The step records a call to [`Validate.rows_not_contaminated()`](../reference/Validate.rows_not_contaminated.md#pointblank.Validate.rows_not_contaminated), which checks that no row's text overlaps with a reference set of texts (such as a benchmark). It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
