# Ablation.drop_each()

Add a variant for each step, removing only the rows that failed that step.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L1077-L1169)

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

Here's a table of temperature readings with missing values and with readings that were recorded in tenths of a degree (so they're ten times too high). The 'model' estimates the average temperature, ignoring missing readings, and is evaluated by its error against the true average of `20`. The validation has a step for each problem, and a third step that flags readings below `19` degrees as suspect.

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
    .col_vals_gt(columns="temp", value=19, na_pass=True)
    .interrogate()
)

ablation = (
    pb.Ablation(
        validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=10
    )
    .drop_each()
    .run()
)

ablation
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **7.29 ± 1.83** (lower is better). Each version was evaluated 10 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| **Step 1** Expect that all values in `temp` should not be Null. | 8 2.7% | 7.29 ± 1.83 Δ vs full +0.00 | +0.11 \[−1.96, +2.17\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNDEuOSIgeTE9IjYuMCIgeDI9IjU4LjkiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTAuNCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **Step 2** Expect that values in `temp` should be between `-10` and `45`. | 12 4.0% | 0.07 ± 0.06 Δ vs full −7.22 | −8.04 \[−9.40, −6.68\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTEuNCIgeTE9IjYuMCIgeDI9IjIyLjUiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMTcuMCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Step 3** Expect that values in `temp` should be \> `19`. | 101 33.7% | 15.22 ± 3.01 Δ vs full +7.93 | +7.84 \[+4.50, +11.19\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNjguNSIgeTE9IjYuMCIgeDI9Ijk2LjAiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iODIuMiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

Each step tells a different story. Removing the missing readings (step 1) is inconclusive, since the model ignores them anyway. Removing the readings in tenths (step 2) helps a lot. And removing the readings below `19` degrees (step 3) hurts: they're good readings, and without them the estimate is too high. Step 3 is too strict for this model, and it removes about a third of the rows. Each variant removes a different number of rows, so each has its own control. With so few rows removed by steps 1 and 2, the ablation uses `repeats=10` to measure the effects precisely enough.

## See Also

[drop_all_but_each()](../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each)  
Remove the rows failing every step but one, for each step.

[drop_all()](../reference/Ablation.drop_all.md#pointblank.Ablation.drop_all)  
Remove the rows failing any step.

[shapley()](../reference/Ablation.shapley.md#pointblank.Ablation.shapley)  
Share the effect of the validation among its steps.
