# Ablation.drop_all()

Add a variant that removes the rows failing any step.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L1171-L1266)

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

Here's a table of temperature readings with missing values and with readings that were recorded in tenths of a degree (so they're ten times too high). The 'model' estimates the average temperature, ignoring missing readings, and is evaluated by its error against the true average of `20`. The validation has a step for each problem, and a third step that flags readings below `19` degrees as suspect. The ablation tests the validation as a whole, and then only its first two steps.

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
    .drop_all()
    .drop_all(steps=[1, 2])
    .run()
)

ablation
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **7.289 ± 1.827** (lower is better). Each version was evaluated 10 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| **All steps** 1: Expect that all values in `temp` should not be Null. 2: Expect that values in `temp` should be between `-10` and `45`. 3: Expect that values in `temp` should be \> `19`. | 121 40.3% | 1.196 ± 0.059 Δ vs full −6.093 | −3.519 \[−4.937, −2.100\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMjcuMSIgeTE9IjYuMCIgeDI9IjQwLjMiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMzMuNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **All of steps 1, 2** 1: Expect that all values in `temp` should not be Null. 2: Expect that values in `temp` should be between `-10` and `45`. | 20 6.7% | 0.065 ± 0.060 Δ vs full −7.224 | −8.012 \[−9.937, −6.087\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjEuOCIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMi45IiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

The validation as a whole helps: removing the 121 rows that failed any step lowers the error from about 7.3 degrees to about 1.2. But the validation does even better without its third step: removing only the 20 rows that failed steps 1 or 2 lowers the error to less than a tenth of a degree. A validation that helps can still have steps that hurt, and [`drop_each()`](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each) shows which ones they are.

## See Also

[drop_each()](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each)  
Remove the rows failing each step, one step at a time.

[drop_all_but_each()](../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each)  
Remove the rows failing every step but one, for each step.

[drop_steps()](../reference/Ablation.drop_steps.md#pointblank.Ablation.drop_steps)  
Remove the rows failing any of the given steps.
