# Ablation.plan()

Show the versions of the table that the ablation will evaluate.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L2718-L2860)

``` python
Ablation.plan()
```

The plan is a dry run: it finds the rows that each variant removes, without evaluating anything. It lists the full table, each variant, and each control, with the number and share of rows they remove, and gives the number of evaluations that [`run()`](../reference/Ablation.run.md#pointblank.Ablation.run) will make in its header. Variants that won't be used (e.g., because they remove no rows) are listed in the notes at the bottom.

## Returns

`GT`  
A Great Tables object.

## Examples

Here's an ablation of a table of temperature readings, checked for missing values and for values out of range, and also for duplicate readings (there are none).

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
for k in range(7, 300, 40):
    temps[k] = None
readings = pl.DataFrame({"id": range(300), "temp": temps})

def evaluate(train, *, seed, variant):
    sample = train.drop_nulls().sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_not_null(columns="temp")
    .col_vals_between(columns="temp", left=-10, right=45, na_pass=True)
    .rows_distinct(columns_subset="id")
    .interrogate()
)

ablation = pb.Ablation(
    validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=10
).drop_each()

ablation.plan()
```

| Ablation plan |  |
|----|----|
| 300 rows in the training table. Each version is evaluated 10 times, so the run will make **50** evaluations. |  |
| Version | Removed |
| Baseline |  |
| **Full table** Nothing removed | 0 0% |
| Variants |  |
| **Step 1** Expect that all values in `temp` should not be Null. | 8 2.7% |
| **Step 2** Expect that values in `temp` should be between `-10` and `45`. | 12 4.0% |
| Controls |  |
| **8 random rows** Rows removed at random, drawn anew for each repeat | 8 2.7% |
| **12 random rows** Rows removed at random, drawn anew for each repeat | 12 4.0% |
| Step 3 isn't used: it removes no rows. |  |

No row failed step 3, so it has no variant, as the note says. With 10 repeats of the full table, two variants, and two controls, the run will make 50 evaluations.
