# tbl_match()

Add a `tbl_match()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L2046-L2100)

``` python
tbl_match(
    tbl_compare,
    keys=None,
    columns=None,
    column_map=None,
    tolerance=None,
    normalize=None,
    null_equal=True,
    schema="strict",
    dup_keys="flag",
    order_by=None,
    partitions=None,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.tbl_match()`](../reference/Validate.tbl_match.md#pointblank.Validate.tbl_match), which checks whether the table matches a comparison table. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
