# data_freshness()

Add a `data_freshness()` step to the collection.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/steps.py#L1918-L1963)

``` python
data_freshness(
    column,
    max_age,
    reference_time=None,
    timezone=None,
    allow_tz_mismatch=False,
    allow_future=False,
    pre=None,
    thresholds=None,
    actions=None,
    brief=None,
    active=True,
    dimension=None,
)
```

The step records a call to [`Validate.data_freshness()`](../reference/Validate.data_freshness.md#pointblank.Validate.data_freshness), which checks whether the latest value in a datetime column is recent enough. It takes the same arguments as that method, so see its reference page for what each one does. Nothing is checked when the step is recorded: it runs when the collection is added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps) and the validation is interrogated.

## Returns

`Steps`  
This same [Steps](../reference/Steps.md#pointblank.Steps) object, with the new step added at the end, so that calls can be chained.
