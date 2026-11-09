# Ablation.drop_steps()

Add a variant that removes the rows failing any of the given steps.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L976-L1075)

``` python
Ablation.drop_steps(
    steps,
    name=None,
)
```

This is the most flexible strategy: it adds one variant, with the rows that failed any of `steps=` removed. Use it to test a particular combination of checks (e.g., all the checks on one column), or to give a variant a name of its own.

## Parameters

`steps: int | list[int]`  
One or more step numbers. Each must be a row-level step that the ablation can use (active, and without a `pre=` function unless that function is marked with [`row_preserving()`](../reference/row_preserving.md#pointblank.row_preserving)).

`name: str | None = None`  
A name for the variant, shown in reports and used as its `id`. By default, it's named after the steps (e.g., `"Steps 1, 3"`, with the id `"steps_1_3"`).

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the variant added.

## Examples

Here's a table of temperature readings with two kinds of bad readings: some were recorded in tenths of a degree (so they're ten times too high), and some are the error code `-40` that the sensor reports when it fails. Step 1 checks for readings that are too high, and step 2 for readings that are too low. The 'model' estimates the average temperature and is evaluated by its error against the true average of `20`.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
for k in range(7, 300, 40):
    temps[k] = -40
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed, variant):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_lt(columns="temp", value=100)
    .col_vals_gt(columns="temp", value=-30)
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .drop_each()
    .drop_steps([1, 2], name="Both range checks")
    .run()
)

ablation
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **5.996 ± 1.857** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| **Step 1** Expect that values in `temp` should be \< `100`. | 12 4.0% | 1.374 ± 0.633 Δ vs full −4.622 | −4.474 \[−6.070, −2.878\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTUuNSIgeTE9IjYuMCIgeDI9IjMzLjYiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjQuNiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Step 2** Expect that values in `temp` should be \> `-30`. | 8 2.7% | 7.179 ± 2.440 Δ vs full +1.182 | +2.460 \[−3.168, +8.089\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMzIuMCIgeTE9IjYuMCIgeDI9Ijk2LjAiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjQuMCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **Both range checks** 1: Expect that values in `temp` should be \< `100`. 2: Expect that values in `temp` should be \> `-30`. | 20 6.7% | 0.069 ± 0.062 Δ vs full −5.927 | −3.738 \[−5.718, −1.758\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTcuNSIgeTE9IjYuMCIgeDI9IjQwLjAiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjguNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

Removing the readings in tenths (step 1) helps, and removing them along with the error codes (`"Both range checks"`) also helps, giving the lowest error: less than a tenth of a degree, down from about 6 degrees. But removing only the error codes (step 2) is inconclusive, with an interval that includes zero. The readings in tenths pull the average up while the error codes pull it down, so they partly offset each other: without the error codes, the readings in tenths pull the average up even more. The effect of a combination of steps isn't the sum of the effects of its steps, which is why it's worth testing combinations like this one.

## See Also

[drop_each()](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each)  
Remove the rows failing each step, one step at a time.

[drop_all()](../reference/Ablation.drop_all.md#pointblank.Ablation.drop_all)  
Remove the rows failing any step.
