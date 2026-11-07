# ablation.Stratified

A control that removes random rows in proportion to groups of the table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L220-L318)

``` python
ablation.Stratified(by)
```

By default, an [`Ablation`](../reference/Ablation.md#pointblank.Ablation) compares each variant with a control that removes the same number of rows at random from anywhere in the table. When the table has groups that matter to the model (e.g., the source of each record, or the class of a classification target), a random draw can take more rows from one group than another just by chance. A stratified control removes rows from each group in proportion to its size, so the control has the same mix of groups as the full table, and the comparison with it isn't blurred by a change in that mix. Pass it to `control=` when creating the ablation.

## Parameters

`by: str | list[str] | tuple[str, …]`  
One or more columns of the training table that define the groups. With several columns, each combination of their values is a group, and missing values make a group of their own.

## Returns

`Stratified`  
A [Stratified](../reference/ablation.Stratified.md#pointblank.ablation.Stratified) object, for the `control=` parameter of [`Ablation`](../reference/Ablation.md#pointblank.Ablation).

## Details

Each repeat draws its rows anew. The number of rows drawn from each group is its share of the rows, rounded so that the total is the number the control removes (the groups with the largest remainders get the extra rows). A control for a [`drop_all_but_each()`](../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each) variant draws in proportion to the rows that the variant keeps.

## Examples

Here's a table of temperature readings from two sensors, where sensor `"b"` has a third of the readings and some of its readings are out of range. The 'model' estimates the average temperature, which is `20` degrees.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(2, 300, 24):
    temps[k] *= 10
readings = pl.DataFrame(
    {"sensor": ["a", "a", "b"] * 100, "temp": temps}
)

def evaluate(train, *, seed):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_between(columns="temp", left=-10, right=45)
    .interrogate()
)

ablation = pb.Ablation(
    validation,
    evaluate=evaluate,
    metric="error",
    higher_is_better=False,
    control=pb.ablation.Stratified(by="sensor"),
).drop_each()

ablation.plan()
```

| Ablation plan |  |
|----|----|
| 300 rows in the training table. Each version is evaluated 5 times, so the run will make **15** evaluations. |  |
| Version | Removed |
| Baseline |  |
| **Full table** Nothing removed | 0 0% |
| Variants |  |
| **Step 1** Expect that values in `temp` should be between `-10` and `45`. | 13 4.3% |
| Controls |  |
| **13 random rows** Rows removed at random within each group of sensor, drawn anew for each repeat | 13 4.3% |

The control removes as many rows as the variant, two thirds of them from sensor `"a"` and one third from sensor `"b"`, whatever the variant removes.
