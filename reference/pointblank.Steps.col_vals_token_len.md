# col_vals_token_len()

Add a `col_vals_token_len()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1275-L1322)

``` python
col_vals_token_len(
    columns,
    tokenizer,
    min_val=None,
    max_val=None,
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

The step records a call to [`Validate.col_vals_token_len()`](../reference/Validate.col_vals_token_len.md#pointblank.Validate.col_vals_token_len), which checks whether the length of text in tokens is within bounds. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
