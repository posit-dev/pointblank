# Ablation.shapley()

Split the effect of removing the steps' rows fairly among the steps (Shapley values).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L1960-L2121)

``` python
Ablation.shapley(
    steps=None,
    permutations=200,
    seed=None,
)
```

When steps fail many of the same rows, the variants of [`drop_each()`](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each) can't say which step deserves the credit: each one looks useful on its own, and [`drop_all_but_each()`](../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each) shows each adding little on top of the others. Shapley values settle it the way game theory splits a payout among players: each step gets its average contribution over the orders in which the steps could be added. Steps that remove the same rows share the credit for them, and the values of the steps add up to the effect of removing every step's rows. This method adds the variants that the values need (one for each group of steps), and after the run, the report shows the values in a group of their own. They're also available from [`get_shapley()`](../reference/Ablation.get_shapley.md#pointblank.Ablation.get_shapley).

## Parameters

`steps: int | list[int] | None = None`  
The steps to split the effect among, as a list of step numbers. By default, it's every step that the ablation can use. Steps that no row failed are left out (with a note), and at least two steps must remain.

`permutations: int = 200`  
The number of random orders of the steps to average over, when there are too many steps to use every order. With `k` steps, there are `k!` orders: when that's at most `permutations=` (with the default of `200`, for up to 5 steps), every group of steps is evaluated and the values are exact.

`seed: int | None = None`  
The seed of the random orders. By default, it's the `seed=` of the ablation.

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the variants added (replacing those of an earlier [shapley()](../reference/Ablation.shapley.md#pointblank.Ablation.shapley)).

## Details

The value of a group of steps is the effect of removing the rows that fail any of them, compared with the control (or with the full table, with `control=None`), as for the verdicts of the other variants. The Shapley value of a step is the average of what it adds to the value of the steps before it, over the orders of the steps. The exact values need every group of the `k` steps (`2^k - 1` of them), and the estimated ones need the groups that start each random order. Each group is a variant (with its control) evaluated `repeats=` times, so the values can take many evaluations: check [`plan()`](../reference/Ablation.plan.md#pointblank.Ablation.plan) first. A variant of another strategy that removes the rows of the same steps is reused, and a group that removes the same rows as another isn't evaluated again.

The values are computed for each repeat, so each has an interval and a p-value across the repeats, and a verdict like that of a variant (`"helps"`, `"hurts"`, or `"inconclusive"`), with the p-values adjusted across the steps. The groups of steps are listed in the plan, and in the results of [`get_effects()`](../reference/Ablation.get_effects.md#pointblank.Ablation.get_effects) (with the strategy `"shapley"`), but not in the report.

## Examples

Here's a table of temperature readings, some of them recorded in tenths of a degree and some with the sentinel value `-999`. Step 1 catches the sentinels, and step 2 catches both kinds of bad readings, so the steps overlap.

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

ablation
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **17.80 ± 8.91** (lower is better). Each version was evaluated 10 times; Δ is the mean paired difference from removing the same number of random rows. For the Shapley values, Δ is each step's share of the effect of removing the rows of every step. |  |  |  |
| Variant | Removed | error | Δ vs random |
| Shapley values |  |  |  |
| **Step 1** Expect that values in `temp` should be \> `-50`. | 8 2.7% | 26% of the total Δ | −6.15 \[−10.61, −1.68\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMjcuOSIgeTE9IjYuMCIgeDI9IjQ2LjUiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMzcuMiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Step 2** Expect that values in `temp` should be between `-50` and `50`. | 20 6.7% | 74% of the total Δ | −17.43 \[−22.05, −12.82\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjMuMyIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMy42IiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |
| The Shapley values are exact, from every order of the 2 steps. They add up to −23.58, the effect of removing the rows of every step compared with removing as many random rows. |  |  |  |

Both steps remove the sentinels, so they share the credit for them equally. Step 2 also gets all the credit for the readings in tenths of a degree, which only it catches, so its Shapley value is the larger one. As the note at the bottom says, the two values add up to the effect of removing the rows of both steps.
