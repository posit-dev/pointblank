# col_vals_ne()

Add a `col_vals_ne()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L417-L459)

``` python
col_vals_ne(
    columns,
    value,
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

The step records a call to [`Validate.col_vals_ne()`](../reference/Validate.col_vals_ne.md#pointblank.Validate.col_vals_ne), which checks whether column values are not equal to a fixed value or the values in another column. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
