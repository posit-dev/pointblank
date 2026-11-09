# Ablation.get_shapley()

Get the Shapley values of the steps as a table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L4269-L4373)

``` python
Ablation.get_shapley()
```

The Shapley values (from [`shapley()`](../reference/Ablation.shapley.md#pointblank.Ablation.shapley)) split the effect of removing every step's rows among the steps, giving each step its average contribution over the orders in which the steps could be added. This method gives them with their intervals and verdicts, for every metric.

## Returns

`DataFrame`  
A Polars DataFrame (or a pandas DataFrame, if Polars isn't installed) with a row for each step and metric, and these columns:

- `step`: the number of the step
- `n_failed`: the number of rows that failed the step
- `metric`: the metric
- [shapley](../reference/Ablation.shapley.md#pointblank.Ablation.shapley): the Shapley value of the step, the mean across the repeats
- `lower`, `upper`: its interval at `conf_level=` (with at least 2 repeats)
- `p`, `p_adj`: the p-value of the Shapley value (against no effect), and the p-value adjusted across the steps (as set by `adjust=`)
- `verdict`: `"helps"`, `"hurts"`, or `"inconclusive"` for the metric named in `metric=` (missing for other metrics, and with fewer than 3 repeats)

For each metric, the Shapley values add up to the effect of removing the rows of every step (compared with the control, or with the full table when `control=None`).

## Examples

Here's an ablation of a table of temperature readings with overlapping steps: step 1 catches the sentinel value `-999`, and step 2 catches the sentinels along with the readings in tenths of a degree. There are 10 repeats, since the sentinels make the error vary a lot from one repeat to the next.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
for k in range(7, 300, 40):
    temps[k] = -999
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_gt(columns="temp", value=-50)
    .col_vals_between(columns="temp", left=-50, right=50)
    .interrogate()
)

ablation = (
    pb.Ablation(
        validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=10
    )
    .shapley()
    .run()
)

ablation.get_shapley().select("step", "shapley", "lower", "upper", "verdict")
```

shape: (2, 5)

| step | shapley    | lower      | upper     | verdict |
|------|------------|------------|-----------|---------|
| i64  | f64        | f64        | f64       | str     |
| 1    | -6.14726   | -10.610548 | -1.683972 | "helps" |
| 2    | -17.432311 | -22.047903 | -12.81672 | "helps" |

Both steps lower the error, and step 2 gets the larger share, since it also catches the readings that step 1 misses.
