# Ablation.run()

Evaluate the full table, each variant, and each control.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L1702-L1816)

``` python
Ablation.run(
    max_workers=None,
    executor="thread",
)
```

The `evaluate=` function is called `repeats=` times for each version of the table, with the seeds `seed`, `seed + 1`, and so on. The full table is evaluated first, then each variant, then each control. A control is drawn anew for each repeat, so that the comparison with it doesn't depend on one random draw of rows. Once the run is done, the results are available from [`get_tabular_report()`](../reference/Ablation.get_tabular_report.md#pointblank.Ablation.get_tabular_report), [`get_effects()`](../reference/Ablation.get_effects.md#pointblank.Ablation.get_effects), and [`to_json()`](../reference/Ablation.to_json.md#pointblank.Ablation.to_json). Running again repeats every evaluation, except those in the cache (see `cache=`).

## Parameters

`max_workers: int | None = None`  
The number of evaluations to make at the same time. By default (or with `1`), they're made one after another. Models that already use several cores (e.g., with `n_jobs=-1`) usually gain little from this.

`executor: str = "thread"`  
How evaluations are made at the same time, when `max_workers=` is more than `1`. With `"thread"` (the default), they run in threads of this process, which suits models that release Python's global interpreter lock while training (as most libraries written in C, C++, or Rust do). With `"process"`, each runs in a separate process: the `evaluate=` function must then be one that can be pickled (e.g., defined at the top level of a module, rather than in a notebook or a lambda), and the training table must be a Polars, pandas, or PyArrow table.

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the results of the run.

## Details

The results are the same whatever `max_workers=` is: the evaluations are made in a different order, but each gets the same table and seed, and the results are put back in order. If an evaluation fails, the evaluations that haven't started are cancelled and the error is raised. With a cache, the results that were already made are kept, so running again picks up where the run stopped.

## Examples

Here's an ablation of a table of temperature readings with some out-of-range values. The `evaluate()` function also returns the number of rows it was given, which is a second metric in the results.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed, variant):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    return {"error": abs(sample["temp"].mean() - 20), "n_train": train.height}

validation = (
    pb.Validate(data=readings)
    .col_vals_between(columns="temp", left=-10, right=45)
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .drop_each()
    .run(max_workers=4)
)

ablation
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **7.392 ± 2.207** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| **Step 1** Expect that values in `temp` should be between `-10` and `45`. | 12 4.0% | 0.109 ± 0.080 Δ vs full −7.283 | −6.901 \[−8.634, −5.168\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjIuNSIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMy4yIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

After the run, the [Ablation](../reference/Ablation.md#pointblank.Ablation) object shows its report. The report shows the metric named in `metric=`; the other metrics are in the results of [get_effects()](../reference/Ablation.get_effects.md#pointblank.Ablation.get_effects).
