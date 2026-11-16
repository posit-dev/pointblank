# col_vals_text_quality()

Add a `col_vals_text_quality()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1159-L1221)

``` python
col_vals_text_quality(
    columns,
    min_words=50,
    max_words=100000,
    min_mean_word_length=3,
    max_mean_word_length=10,
    max_symbol_ratio=0.1,
    max_bullet_lines=0.9,
    max_ellipsis_lines=0.3,
    min_alpha_words=0.8,
    min_stop_words=2,
    max_repeated_lines=0.3,
    stop_words=None,
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

The step records a call to [`Validate.col_vals_text_quality()`](../reference/Validate.col_vals_text_quality.md#pointblank.Validate.col_vals_text_quality), which checks text with the heuristic quality filters of web text (adding a validation step for each filter when the collection is added to a validation). It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
