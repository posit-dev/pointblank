# Ablation.get_effects()

Get the results of the ablation as a table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L1174-L1273)

``` python
Ablation.get_effects()
```

The table has one row for each version of the training table and each metric: the full table first (as the baseline), then the variants. Use it to make plots, to compare ablations of different data, or to look at the metrics that the report doesn't show.

## Returns

`DataFrame`  
A Polars DataFrame (or a pandas DataFrame, if Polars isn't installed) with these columns:

- `variant`, `name`, `strategy`: the variant's id, name, and strategy (`"full"` for the full table)
- `steps`: the numbers of the steps whose failing rows were removed, as a list
- `n_removed`, `f_removed`: the number and share of rows removed
- `metric`, `mean`, `sd`: the metric, and its mean and standard deviation across the repeats
- `delta_vs_full`, `delta_vs_full_lower`, `delta_vs_full_upper`: the mean paired difference from the full table, with its interval at `conf_level=`
- `p_vs_full`, `p_adj_vs_full`: the p-value of that difference, and the p-value adjusted across the variants (as set by `adjust=`)
- `control_mean`: the mean of the metric for the variant's control
- `delta_vs_control`, `delta_vs_control_lower`, `delta_vs_control_upper`, `p_vs_control`, `p_adj_vs_control`: the same, for the difference from the control
- `verdict`: `"helps"`, `"hurts"`, or `"inconclusive"` for the metric named in `metric=` (missing for other metrics, and with fewer than 3 repeats)

The comparisons are missing for the full table, and the control columns are missing when `control=None`. The intervals and p-values need at least 2 repeats.

## Examples

Here's an ablation of a table of temperature readings with some out-of-range values.

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
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .drop_each()
    .run()
)

ablation.get_effects().select("variant", "mean", "delta_vs_control", "verdict")
```

shape: (2, 4)

| variant  | mean    | delta_vs_control | verdict |
|----------|---------|------------------|---------|
| str      | f64     | f64              | str     |
| "full"   | 7.392   | null             | null    |
| "step_1" | 0.10875 | -6.90125         | "helps" |

Removing the 12 rows that failed step 1 lowers the error by about 7 degrees compared with removing 12 random rows, so the verdict is that the step helps.
