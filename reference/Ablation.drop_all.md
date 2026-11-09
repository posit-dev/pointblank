# Ablation.drop_all()

Add a variant that removes the rows failing any step.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L1031-L1106)

``` python
Ablation.drop_all(steps=None)
```

This variant answers the question 'does the validation as a whole help?' Without [rows_distinct()](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) steps, its training table is the one that [`get_sundered_data()`](../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data) gives: the rows that passed every step.

## Parameters

`steps: list[int] | None = None`  
The step numbers to combine. By default, every row-level step that the ablation can use is combined.

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the variant added.

## Examples

Here's a table of temperature readings with out-of-range and missing values, validated with a step for each problem.

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
).drop_all()

ablation.plan()
```

| Ablation plan |  |
|----|----|
| 300 rows in the training table. Each version is evaluated 5 times, so the run will make **15** evaluations. |  |
| Version | Removed |
| Baseline |  |
| **Full table** Nothing removed | 0 0% |
| Variants |  |
| **All steps** 1: Expect that all values in `temp` should not be Null. 2: Expect that values in `temp` should be between `-10` and `45`. | 20 6.7% |
| Controls |  |
| **20 random rows** Rows removed at random, drawn anew for each repeat | 20 6.7% |

The one variant removes the 20 rows that failed either step.
