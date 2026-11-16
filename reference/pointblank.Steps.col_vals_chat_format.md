# col_vals_chat_format()

Add a `col_vals_chat_format()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1104-L1157)

``` python
col_vals_chat_format(
    columns,
    roles=("system", "user", "assistant"),
    alternate=True,
    first_role=None,
    last_role=None,
    allow_empty=False,
    role_key="role",
    content_key="content",
    na_pass=False,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.col_vals_chat_format()`](../reference/Validate.col_vals_chat_format.md#pointblank.Validate.col_vals_chat_format), which checks whether each value is a well-formed conversation (a list of messages with roles). It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
