# Ablation.drop_each()

Add a variant for each step, removing only the rows that failed that step.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L503-L575)

``` python
Ablation.drop_each(steps=None)
```

Each variant answers the question 'does removing the rows that fail this check help, on its own?' Comparing the variants shows which checks matter to the model and which don't.

## Parameters

`steps: int | list[int] | None = None`  
The step numbers to add variants for. By default, every row-level step that the ablation can use gets one (the steps that can't be used are listed in the notes of the reports).

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the variants added.

## Examples

Here's a table of temperature readings with out-of-range and missing values. The 'model' estimates the average temperature, ignoring missing readings, and is evaluated by its error against the true average of `20`.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
for k in range(7, 300, 40):
    temps[k] = None
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed, variant):
    sample = train.drop_nulls().sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_not_null(columns="temp")
    .col_vals_between(columns="temp", left=-10, right=45, na_pass=True)
    .interrogate()
)

ablation = pb.Ablation(
    validation, evaluate=evaluate, metric="error", higher_is_better=False
).drop_each()

ablation.plan()
```

| Ablation plan |  |
|----|----|
| 300 rows in the training table. Each version is evaluated 5 times, so the run will make **25** evaluations. |  |
| Version | Removed |
| Baseline |  |
| **Full table** Nothing removed | 0 0% |
| Variants |  |
| **Step 1** Expect that all values in `temp` should not be Null. | 8 2.7% |
| **Step 2** Expect that values in `temp` should be between `-10` and `45`. | 12 4.0% |
| Controls |  |
| **8 random rows** Rows removed at random, drawn anew for each repeat | 8 2.7% |
| **12 random rows** Rows removed at random, drawn anew for each repeat | 12 4.0% |

There's a variant for each step. The two variants remove different numbers of rows, so each has its own control.
