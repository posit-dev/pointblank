# tbl_reconcile()

Add a `tbl_reconcile()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L2102-L2149)

``` python
tbl_reconcile(
    tbl_compare,
    keys=None,
    measures=None,
    agg="sum",
    tolerance=None,
    column_map=None,
    partitions=None,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.tbl_reconcile()`](../reference/Validate.tbl_reconcile.md#pointblank.Validate.tbl_reconcile), which checks whether aggregates of the table reconcile with those of a comparison table. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
