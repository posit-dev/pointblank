# ablation.Stratified

A control that removes random rows in proportion to groups of the table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L393-L538)

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

Here's a table of temperature readings from two sensors in different places: sensor `"a"` has two thirds of the readings and reads about 20 degrees, while sensor `"b"` has the other third and reads about 40 degrees. One reading in ten comes from a device that was never calibrated (`calibrated_on` is missing), and those read 1.5 degrees too high. The 'model' estimates the average temperature across both sensors, and it's evaluated by its error against the true average of the calibrated readings. It's deterministic, so the only randomness in the ablation is in the rows that the controls remove.

``` python
import pointblank as pb
import polars as pl

sensors = ["a", "a", "b"] * 100
calibrated_on = [None if k % 10 == 3 else "2026-03-01" for k in range(300)]
temps = [
    (20 if s == "a" else 40) + (k * 37 % 11 - 5) * 0.6 + (1.5 if c is None else 0)
    for k, (s, c) in enumerate(zip(sensors, calibrated_on))
]
readings = pl.DataFrame({"sensor": sensors, "calibrated_on": calibrated_on, "temp": temps})
true_average = readings.drop_nulls()["temp"].mean()

def evaluate(train, *, seed):
    return abs(train["temp"].mean() - true_average)

validation = (
    pb.Validate(data=readings)
    .col_vals_not_null(columns="calibrated_on")
    .interrogate()
)

def ablate(control):
    return (
        pb.Ablation(
            validation,
            evaluate=evaluate,
            metric="error",
            higher_is_better=False,
            control=control,
        )
        .drop_each()
        .run()
    )

stratified = ablate(pb.ablation.Stratified(by="sensor"))
stratified
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **0.1484 ± 0.0000** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows (stratified by sensor). |  |  |  |
| Variant | Removed | error | Δ vs random |
| **Step 1** Expect that all values in `calibrated_on` should not be Null. | 30 10.0% | 0.0000 ± 0.0000 Δ vs full −0.1484 | −0.1396 \[−0.1868, −0.0923\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjcuMyIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxNS42IiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

Removing the 30 uncalibrated readings helps. Each repeat of the control removes 30 random readings, always 20 from sensor `"a"` and 10 from sensor `"b"`, so the control has the same mix of sensors as the full table. Here's the same ablation with the default control, which removes 30 readings from anywhere in the table:

``` python
pl.concat(
    [
        ablation.get_effects()
        .filter(pl.col("variant") == "step_1")
        .select(
            pl.lit(name).alias("control"),
            "control_mean",
            "delta_vs_control",
            "delta_vs_control_lower",
            "delta_vs_control_upper",
            "verdict",
        )
        for name, ablation in [
            ("size_matched", ablate("size_matched")),
            ("stratified", stratified),
        ]
    ]
)
```

shape: (2, 6)

| control | control_mean | delta_vs_control | delta_vs_control_lower | delta_vs_control_upper | verdict |
|----|----|----|----|----|----|
| str | f64 | f64 | f64 | f64 | str |
| "size_matched" | 0.113704 | -0.113704 | -0.230853 | 0.003446 | "inconclusive" |
| "stratified" | 0.139556 | -0.139556 | -0.186772 | -0.092339 | "helps" |

The default control's random draws take a different number of readings from each sensor in each repeat, and since the sensors read 20 degrees apart, that alone moves the average. The noise widens the interval to about \[−0.23, +0.003\], which includes zero, so the verdict is inconclusive. The stratified control removes that source of noise, and its interval of about \[−0.19, −0.09\] is narrow enough for the verdict to be that the step helps. Stratifying is worth it when the groups differ in what the model learns from them: with groups that are much alike, it makes little difference.

## See Also

[Ablation](../reference/Ablation.md#pointblank.Ablation)  
Measure how the rows that fail validation steps affect a model trained on the data.

[drop_groups()](../reference/Ablation.drop_groups.md#pointblank.Ablation.drop_groups)  
Remove the rows of each group, one group at a time.
