# Ablation.drop_all_but_each()

Add a variant for each step, removing the rows that failed every other step.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L1005-L1115)

``` python
Ablation.drop_all_but_each(steps=None)
```

The variants of [`drop_each()`](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each) say whether each check helps on its own. This strategy asks a different question: 'does this check add anything, once the other checks have been applied?' Two checks can find mostly the same rows, so that each helps on its own but either one makes the other redundant. Each variant leaves out one step: it removes the rows that failed any of the other steps, and is compared with the variant that removes the rows of every step (which this method adds too, as with [`drop_all()`](../reference/Ablation.drop_all.md#pointblank.Ablation.drop_all)).

## Parameters

`steps: list[int] | None = None`  
The step numbers to use (at least two). By default, every row-level step that the ablation can use is included.

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the variants added.

## Details

The effect of a left-out step is what its rows add on top of the other steps: the difference between removing the rows of every step and removing the rows of the other steps (`delta_added` in [`get_effects()`](../reference/Ablation.get_effects.md#pointblank.Ablation.get_effects)). Removing the step's rows also means removing more rows, so with a control (the default), the comparison is with removing the rows of the other steps plus as many random rows as the step adds. Each variant has a control of its own for that.

The verdict is `"helps"` or `"hurts"` when the step's rows make a clear difference on top of the others. When they don't, but the step's rows help on their own (its variant from [drop_each()](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each) helps), the verdict is `"redundant"`: the other steps already remove what matters. So it's best to use this strategy together with [drop_each()](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each).

A variant that removes the same rows as another version of the table (e.g., when every row the step failed also failed another step) isn't evaluated again, and the results of that version are used.

## Examples

Here's a table of temperature readings where some readings were recorded in tenths of a degree, so they're ten times too high. Step 1 finds all of them, while step 2 has a looser limit and only finds the highest ones.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_between(columns="temp", left=-10, right=45)
    .col_vals_lt(columns="temp", value=200)
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .drop_each()
    .drop_all_but_each()
    .run()
)

ablation
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **7.392 ± 2.207** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. For leaving one out, Δ is what the left-out step's rows add to the other steps: removing every step's rows, compared with removing the other steps' rows, plus as many random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| Variants |  |  |  |
| **Step 1** Expect that values in `temp` should be between `-10` and `45`. | 12 4.0% | 0.109 ± 0.080 Δ vs full −7.283 | −6.901 \[−8.634, −5.168\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjIuNSIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMy4yIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| **Step 2** Expect that values in `temp` should be \< `200`. | 6 2.0% | 3.449 ± 1.759 Δ vs full −3.943 | −3.706 \[−4.900, −2.512\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMjMuOSIgeTE9IjYuMCIgeDI9IjM2LjYiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMzAuMyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **All steps** 1: Expect that values in `temp` should be between `-10` and `45`. 2: Expect that values in `temp` should be \< `200`. | 12 4.0% | 0.109 ± 0.080 Δ vs full −7.283 | −6.901 \[−8.634, −5.168\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjIuNSIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMy4yIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| Leave one out |  |  |  |
| **All steps but step 1** Leaves out step 1: Expect that values in `temp` should be between `-10` and `45`. Same rows as Step 2, so it isn't evaluated again | 6 2.0% | 3.449 ± 1.759 Δ vs full −3.943 | −2.871 \[−4.561, −1.181\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMjUuNyIgeTE9IjYuMCIgeDI9IjQzLjciIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMzQuNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **All steps but step 2** Leaves out step 2: Expect that values in `temp` should be \< `200`. Same rows as Step 1, so it isn't evaluated again | 12 4.0% | 0.109 ± 0.080 Δ vs full −7.283 | +0.000 \[+0.000, +0.000\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTAuMCIgeTE9IjYuMCIgeDI9IjUwLjAiIHkyPSI2LjAiIHN0cm9rZT0iI0U1QUIwMCIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTAuMCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0U1QUIwMCI+PC9jaXJjbGU+PC9zdmc+) redundant |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

Both steps help on their own. On top of step 2, step 1 still helps, since it removes the bad readings that step 2 misses. But step 2 is redundant: every row it finds also fails step 1, so leaving it out removes the same rows as using both steps.
