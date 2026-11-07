# Ablation.get_tabular_report()

Get a tabular report of the ablation's results.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L3033-L3251)

``` python
Ablation.get_tabular_report(
    metric=None,
    title=None,
)
```

The report has a row for each variant, with the rows it removed, the metric's mean and standard deviation across the repeats, and its difference from the control (or from the full table, if `control=None`). The difference is shown with its interval, as a bar around zero (on the same scale for every variant), and with the verdict as a badge. The variants of [`drop_all_but_each()`](../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each) are in a group of their own, showing what the left-out step's rows add to the other steps. The header gives the metric for the full table and the settings of the run, and notes at the bottom list the variants and steps that weren't used.

## Parameters

`metric: str | None = None`  
The metric to show. By default, it's the one that verdicts are based on (named in `metric=` when the ablation was created). Verdicts are only shown for that metric.

`title: str | None = None`  
An optional title replacing the default.

## Returns

`GT`  
A Great Tables object.

## Examples

Here's an ablation of a table of temperature readings with out-of-range and missing values.

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

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .drop_each()
    .run()
)

ablation.get_tabular_report(title="Which checks matter for the temperature model?")
```

| Which checks matter for the temperature model? |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **7.179 ± 2.440** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| **Step 1** Expect that all values in `temp` should not be Null. | 8 2.7% | 7.179 ± 2.440 Δ vs full +0.000 | +0.563 \[−4.216, +5.342\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMzEuNiIgeTE9IjYuMCIgeDI9IjczLjMiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTIuNSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **Step 2** Expect that values in `temp` should be between `-10` and `45`. | 12 4.0% | 0.069 ± 0.062 Δ vs full −7.109 | −7.974 \[−10.556, −5.393\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjYuNSIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxNS4yIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

Removing the readings that are out of range (step 2) helps, while removing the missing readings (step 1) makes no clear difference.
