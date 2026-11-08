# Ablation.drop_steps()

Add a variant that removes the rows failing any of the given steps.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L843-L919)

``` python
Ablation.drop_steps(
    steps,
    name=None,
)
```

This is the most flexible strategy: it adds one variant, with the rows that failed any of `steps=` removed. Use it to test a particular combination of checks (e.g., all the checks on one column), or to give a variant a name of its own.

## Parameters

`steps: int | list[int]`  
One or more step numbers. Each must be a row-level step that the ablation can use (active, and without a `pre=` function unless that function is marked with [`row_preserving()`](../reference/row_preserving.md#pointblank.row_preserving)).

`name: str | None = None`  
A name for the variant, shown in reports and used as its `id`. By default, it's named after the steps (e.g., `"Steps 1, 3"`, with the id `"steps_1_3"`).

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the variant added.

## Examples

Here's a table of temperature readings with some out-of-range values, checked against two different ranges. The 'model' estimates the average temperature and is evaluated by its error against the true average of `20`.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed, variant):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_between(columns="temp", left=-10, right=45)
    .col_vals_lt(columns="temp", value=100)
    .interrogate()
)

ablation = pb.Ablation(
    validation, evaluate=evaluate, metric="error", higher_is_better=False
).drop_steps([1, 2], name="Range checks")

ablation.plan()
```

| Ablation plan |  |
|----|----|
| 300 rows in the training table. Each version is evaluated 5 times, so the run will make **15** evaluations. |  |
| Version | Removed |
| Baseline |  |
| **Full table** Nothing removed | 0 0% |
| Variants |  |
| **Range checks** 1: Expect that values in `temp` should be between `-10` and `45`. 2: Expect that values in `temp` should be \< `100`. | 12 4.0% |
| Controls |  |
| **12 random rows** Rows removed at random, drawn anew for each repeat | 12 4.0% |

The plan has the full table, the one variant (removing the rows that failed either step), and its control, which removes the same number of rows at random.
