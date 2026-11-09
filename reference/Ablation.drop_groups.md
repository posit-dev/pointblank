# Ablation.drop_groups()

Add a variant for each group of rows, removing the rows of that group.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L1385-L1502)

``` python
Ablation.drop_groups(
    column,
    values=None,
)
```

The other strategies remove the rows that failed validation steps. This one removes the rows with a value in a column (e.g., every record from one source, region, or batch), to find out whether a part of the data helps or hurts the model. A group that hurts is worth a closer look: it may have a problem that no check has found yet.

## Parameters

`column: str`  
The column whose values define the groups.

`values: Any = None`  
The values of `column=` to make variants for: one value or a list of them (`None` in a list stands for the rows where the column is missing). By default, every value in the column gets a variant, as long as there are no more than 50 of them.

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the variants added.

## Details

The variants are named after the column and value (e.g., `"source = web"`, with the id `"source=web"`), and the rows where the column is missing are a group of their own (`"source is missing"`). Like any other variant, a group is compared with removing the same number of random rows. A group with most of the table's rows gives a warning (see `max_removed=`), since without it the training table is mostly a different dataset.

## Examples

Here's a table of temperature readings from three sensors, where sensor `"c"` reads five degrees too high. No check of the validation finds this, since the readings are all in the expected range.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
sensors = ["a", "b", "c"] * 100
temps = [t + 5 if s == "c" else t for t, s in zip(temps, sensors)]
readings = pl.DataFrame({"sensor": sensors, "temp": temps})

def evaluate(train, *, seed):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_between(columns="temp", left=-10, right=45)
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .drop_groups("sensor")
    .run()
)

ablation
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **1.608 ± 0.239** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| **sensor = a** All the rows of this group | 100 33.3% | 2.309 ± 0.252 Δ vs full +0.701 | +0.664 \[+0.421, +0.907\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTkuOSIgeTE9IjYuMCIgeDI9IjcxLjIiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjUuNSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **sensor = b** All the rows of this group | 100 33.3% | 2.426 ± 0.253 Δ vs full +0.818 | +0.781 \[+0.439, +1.123\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNjAuMyIgeTE9IjYuMCIgeDI9Ijc2LjMiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjguMyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **sensor = c** All the rows of this group | 100 33.3% | 0.143 ± 0.061 Δ vs full −1.465 | −1.502 \[−1.965, −1.039\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjUuNyIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxNC44IiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

Removing the readings of sensor `"c"` helps, while removing either of the other sensors hurts, since it leaves a larger share of the readings from sensor `"c"`.

## See Also

[Stratified](../reference/ablation.Stratified.md#pointblank.ablation.Stratified)  
Draw the rows of controls in proportion to groups of rows.
