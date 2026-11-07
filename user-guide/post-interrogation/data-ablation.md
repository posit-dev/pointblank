# Data Ablation

A validation tells you which rows of a table fail its checks. If the table is training data for a model, the next question is whether those rows actually matter: does removing them make the model better, worse, or no different? Data ablation answers that by experiment. The model is trained on the full table and on versions of it with the failing rows removed, and each version is scored on the same held-out data. Pointblank's [`Ablation`](../../reference/Ablation.md#pointblank.Ablation) class runs that experiment from an interrogated validation and reports the effect of each step's rows.

Removing rows changes the model even when the rows were perfectly good, since there's less data to learn from. So each version is also compared to a *control*, which removes the same number of rows at random. A step whose rows are really a problem does better than its control, and a step whose rows are good data does worse.

This article uses [scikit-learn](https://scikit-learn.org/) for the model, so you'll need it installed (`pip install scikit-learn`) to run the examples. Pointblank itself doesn't depend on it: the model is just a function that you write.

## The Data and the Model

The `nycflights` dataset has a row for each flight that left New York City airports in 2013. We'll train a model to predict a flight's `air_time` (in minutes) from its `distance` (in miles) and the `month` and `hour` it left. Here we take 6,000 flights, keeping 2,000 of them aside as the held-out data for scoring the model.

``` python
import polars as pl

flights = (
    pb.load_dataset(dataset="nycflights")
    .drop_nulls("air_time")
    .select("month", "hour", "distance", "air_time")
    .sample(n=6000, seed=23)
)

holdout = flights.tail(2000)
```

To know what the right answers are, we'll add two problems to the other 4,000 flights. One flight in every 50 has its air time recorded in hours rather than minutes, and one in every 33 has a missing distance recorded as `0`:

``` python
train = (
    flights.head(4000)
    .with_row_index("row")
    .with_columns(
        air_time=pl.when(pl.col("row") % 50 == 7)
        .then(pl.col("air_time") / 60)
        .otherwise(pl.col("air_time")),
        distance=pl.when(pl.col("row") % 33 == 4)
        .then(0)
        .otherwise(pl.col("distance")),
    )
    .drop("row")
)

pb.preview(train)
```

| PolarsRows4,000Columns4 |  |  |  |  |
|----|----|----|----|----|
|  | month *Int64* | hour *Int64* | distance *Int64* | air_time *Float64* |
| 1 | 3 | 16 | 1391 | 210.0 |
| 2 | 8 | 15 | 1089 | 145.0 |
| 3 | 5 | 16 | 762 | 123.0 |
| 4 | 10 | 8 | 340 | 61.0 |
| 5 | 5 | 17 | 0 | 56.0 |
| 3996 | 12 | 12 | 544 | 88.0 |
| 3997 | 7 | 17 | 502 | 78.0 |
| 3998 | 6 | 15 | 0 | 112.0 |
| 3999 | 11 | 6 | 1598 | 183.0 |
| 4000 | 2 | 16 | 944 | 135.0 |

The validation of the training data has a check for each problem, plus a third one that looks reasonable but isn't: it treats any flight longer than 2,500 miles as an error, though flights from New York to San Francisco and Honolulu are longer than that.

``` python
validation = (
    pb.Validate(data=train, tbl_name="flights_train")
    .col_vals_between(
        columns="air_time",
        left=20,
        right=700,
        brief="Air time is between 20 and 700 minutes",
    )
    .col_vals_gt(
        columns="distance",
        value=0,
        brief="Distance is greater than zero",
    )
    .col_vals_le(
        columns="distance",
        value=2500,
        brief="Distance is at most 2,500 miles",
    )
    .interrogate()
)

validation
```

[TABLE]

Each of the three steps has failing rows. The report says how many, but not whether removing them would help the model.

## Writing the `evaluate` Function

An ablation needs a function that trains the model on a version of the training table and scores it. Pointblank calls the function with the table as its first argument (the same type of table as the validation's data, a Polars DataFrame here) and a `seed=` keyword. It returns the score, or a dictionary of scores if there's more than one:

``` python
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

features = ["month", "hour", "distance"]


def evaluate(train, *, seed):
    model = RandomForestRegressor(
        n_estimators=50, min_samples_leaf=5, random_state=seed, n_jobs=-1
    )
    model.fit(train.select(features).to_numpy(), train["air_time"].to_numpy())

    predicted = model.predict(holdout.select(features).to_numpy())
    actual = holdout["air_time"].to_numpy()

    return {
        "mae": mean_absolute_error(actual, predicted),
        "r2": r2_score(actual, predicted),
    }
```

There are a few things to keep in mind when writing this function:

- the held-out data is fixed: it's the yardstick for every version of the training table, so it's never ablated
- the function should give the same result each time it's called with the same table and seed, so use the seed for any randomness in training (here, the forest's `random_state=`)
- the function can also take a `variant=` keyword, which gets an [`AblationVariant`](../../reference/AblationVariant.md#pointblank.AblationVariant) describing the version of the table being evaluated (its id, the steps whose rows were removed, and the number of rows removed)

## Planning the Ablation

An [Ablation](../../reference/Ablation.md#pointblank.Ablation) is built in the same way as a validation: you add variants with its methods, and nothing is evaluated until you call [`Ablation.run()`](../../reference/Ablation.run.md#pointblank.Ablation.run). The [`Ablation.drop_each()`](../../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each) method adds a variant for each step, removing the rows that failed it, and [`Ablation.drop_all()`](../../reference/Ablation.drop_all.md#pointblank.Ablation.drop_all) adds one that removes the rows that failed any step (the failing set from [`Validate.get_sundered_data()`](../../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data)).

The model is scored on the mean absolute error (`"mae"`), so lower is better. The [`Ablation.plan()`](../../reference/Ablation.plan.md#pointblank.Ablation.plan) method shows what the run will do, without evaluating anything:

``` python
ablation = (
    pb.Ablation(
        validation,
        evaluate=evaluate,
        metric="mae",
        higher_is_better=False,
        label="Air time model",
    )
    .drop_each()
    .drop_all()
)

ablation.plan()
```

| Ablation plan Air time model |  |
|----|----|
| 4,000 rows in the training table. Each version is evaluated 5 times, so the run will make **45** evaluations. |  |
| Version | Removed |
| Baseline |  |
| **Full table** Nothing removed | 0 0% |
| Variants |  |
| **Step 1** Air time is between 20 and 700 minutes | 80 2.0% |
| **Step 2** Distance is greater than zero | 122 3.0% |
| **Step 3** Distance is at most 2,500 miles | 182 4.5% |
| **All steps** 1: Air time is between 20 and 700 minutes 2: Distance is greater than zero 3: Distance is at most 2,500 miles | 380 9.5% |
| Controls |  |
| **80 random rows** Rows removed at random, drawn anew for each repeat | 80 2.0% |
| **122 random rows** Rows removed at random, drawn anew for each repeat | 122 3.0% |
| **182 random rows** Rows removed at random, drawn anew for each repeat | 182 4.5% |
| **380 random rows** Rows removed at random, drawn anew for each repeat | 380 9.5% |

Each version of the table, and each control, is evaluated `repeats=5` times by default, with the seeds `0` to `4`. Variants that remove the same number of rows share a control, and a variant that removes no rows isn't evaluated at all (the plan has a note saying so).

## Running the Ablation

The [`Ablation.run()`](../../reference/Ablation.run.md#pointblank.Ablation.run) method does the evaluations and returns the [Ablation](../../reference/Ablation.md#pointblank.Ablation), which then shows its report:

``` python
ablation.run()
```

| Ablation report Air time model |  |  |  |
|----|----|----|----|
| mae on the full table (4,000 rows): **9.122 ± 0.059** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | mae | Δ vs random |
| **Step 1** Air time is between 20 and 700 minutes | 80 2.0% | 7.946 ± 0.085 Δ vs full −1.176 | −1.217 \[−1.404, −1.030\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMTYuMiIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMC4xIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| **Step 2** Distance is greater than zero | 122 3.0% | 9.153 ± 0.078 Δ vs full +0.031 | +0.039 \[−0.117, +0.195\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNDYuMiIgeTE9IjYuMCIgeDI9IjU2LjQiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTEuMyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **Step 3** Distance is at most 2,500 miles | 182 4.5% | 9.617 ± 0.078 Δ vs full +0.495 | +0.312 \[+0.131, +0.493\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTQuMyIgeTE9IjYuMCIgeDI9IjY2LjEiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjAuMiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **All steps** 1: Air time is between 20 and 700 minutes 2: Distance is greater than zero 3: Distance is at most 2,500 miles | 380 9.5% | 8.487 ± 0.010 Δ vs full −0.635 | −0.786 \[−0.943, −0.629\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTkuMSIgeTE9IjYuMCIgeDI9IjI5LjQiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjQuMiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

The report has a row for each variant. The **mae** column has the mean and standard deviation of the metric over the repeats, and the change from the full table. The **Δ vs random** column is the main result: the mean difference between the variant and its control, with its interval and a verdict. A variant *helps* if it does better than its control, *hurts* if it does worse, and is *inconclusive* if the difference could be down to chance.

Here, the results match what we know about the data:

- removing the flights with air times in hours (step 1) helps a lot, more than removing the same number of random flights
- removing the flights with a zero distance (step 2) makes no clear difference: the problem is real, but it doesn't measurably affect this model
- removing the flights longer than 2,500 miles (step 3) hurts: they're good data, and without them the model can't predict the long flights in the held-out data
- removing every failing row helps, but less than step 1 alone, because it includes the rows from step 3

Each version and its control are evaluated with the same seeds, and the differences are taken between runs with the same seed. This pairing means that the variation from the seed itself (here, the randomness in growing the forest) mostly cancels out.

## Comparing Other Combinations of Steps

The [`Ablation.drop_steps()`](../../reference/Ablation.drop_steps.md#pointblank.Ablation.drop_steps) method adds a variant that removes the rows failing any of the steps you give it. Let's try the validation without its third step. Adding a variant clears the results, so the ablation needs to be run again:

``` python
ablation.drop_steps([1, 2], name="Without step 3").run()

ablation.get_tabular_report()
```

| Ablation report Air time model |  |  |  |
|----|----|----|----|
| mae on the full table (4,000 rows): **9.122 ± 0.059** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | mae | Δ vs random |
| **Step 1** Air time is between 20 and 700 minutes | 80 2.0% | 7.946 ± 0.085 Δ vs full −1.176 | −1.217 \[−1.404, −1.030\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMTYuMiIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMC4xIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| **Step 2** Distance is greater than zero | 122 3.0% | 9.153 ± 0.078 Δ vs full +0.031 | +0.039 \[−0.117, +0.195\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNDYuMiIgeTE9IjYuMCIgeDI9IjU2LjQiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTEuMyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **Step 3** Distance is at most 2,500 miles | 182 4.5% | 9.617 ± 0.078 Δ vs full +0.495 | +0.312 \[+0.131, +0.493\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTQuMyIgeTE9IjYuMCIgeDI9IjY2LjEiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjAuMiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **All steps** 1: Air time is between 20 and 700 minutes 2: Distance is greater than zero 3: Distance is at most 2,500 miles | 380 9.5% | 8.487 ± 0.010 Δ vs full −0.635 | −0.786 \[−0.943, −0.629\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTkuMSIgeTE9IjYuMCIgeDI9IjI5LjQiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjQuMiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Without step 3** 1: Air time is between 20 and 700 minutes 2: Distance is greater than zero | 200 5.0% | 8.012 ± 0.030 Δ vs full −1.110 | −1.108 \[−1.178, −1.038\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTEuNCIgeTE9IjYuMCIgeDI9IjE2LjAiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMTMuNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

The two checks that find real problems do about as well together as step 1 alone, and better than all three steps. So the third step should be dropped from the cleaning process (or its limit raised), even though it might still be a useful check to report on.

## Getting the Results

The [`Ablation.get_effects()`](../../reference/Ablation.get_effects.md#pointblank.Ablation.get_effects) method returns the results as a table, with a row for each variant and metric. Along with the differences from the controls, it has the differences from the full table, the interval bounds, and the p-values:

``` python
effects = ablation.get_effects()
mae = effects.filter(pl.col("metric") == "mae")

mae.select("name", "mean", "delta_vs_control", "p_adj_vs_control", "verdict")
```

shape: (6, 5)

| name             | mean     | delta_vs_control | p_adj_vs_control | verdict        |
|------------------|----------|------------------|------------------|----------------|
| str              | f64      | f64              | f64              | str            |
| "Full table"     | 9.122164 | null             | null             | null           |
| "Step 1"         | 7.945899 | -1.217097        | 0.000219         | "helps"        |
| "Step 2"         | 9.152884 | 0.039022         | 0.5248           | "inconclusive" |
| "Step 3"         | 9.617069 | 0.311815         | 0.017424         | "hurts"        |
| "All steps"      | 8.487428 | -0.78607         | 0.00047          | "helps"        |
| "Without step 3" | 8.011877 | -1.107792        | 0.000008         | "helps"        |

The verdicts are only given for the metric named in `metric=`, but the other metrics are evaluated too. The report can show any of them:

``` python
ablation.get_tabular_report(metric="r2")
```

| Ablation report Air time model |  |  |  |
|----|----|----|----|
| r2 on the full table (4,000 rows): **0.9761 ± 0.0014**. Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. Verdicts are based on mae, so none are shown here. |  |  |  |
| Variant | Removed | r2 | Δ vs random |
| **Step 1** Air time is between 20 and 700 minutes | 80 2.0% | 0.9822 ± 0.0021 Δ vs full +0.0062 | +0.0064 \[+0.0028, +0.0101\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTYuNCIgeTE9IjYuMCIgeDI9IjczLjMiIHkyPSI2LjAiIHN0cm9rZT0iIzY2NjY2NiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjQuOCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzY2NjY2NiI+PC9jaXJjbGU+PC9zdmc+) |
| **Step 2** Distance is greater than zero | 122 3.0% | 0.9754 ± 0.0021 Δ vs full −0.0007 | −0.0015 \[−0.0060, +0.0029\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMzYuMSIgeTE9IjYuMCIgeDI9IjU2LjgiIHkyPSI2LjAiIHN0cm9rZT0iIzY2NjY2NiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNDYuNCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzY2NjY2NiI+PC9jaXJjbGU+PC9zdmc+) |
| **Step 3** Distance is at most 2,500 miles | 182 4.5% | 0.9573 ± 0.0004 Δ vs full −0.0188 | −0.0158 \[−0.0200, −0.0117\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjMuMSIgeTI9IjYuMCIgc3Ryb2tlPSIjNjY2NjY2IiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMy42IiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNjY2NjY2Ij48L2NpcmNsZT48L3N2Zz4=) |
| **All steps** 1: Air time is between 20 and 700 minutes 2: Distance is greater than zero 3: Distance is at most 2,500 miles | 380 9.5% | 0.9625 ± 0.0001 Δ vs full −0.0136 | −0.0098 \[−0.0157, −0.0040\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTMuOSIgeTE9IjYuMCIgeDI9IjQwLjgiIHkyPSI2LjAiIHN0cm9rZT0iIzY2NjY2NiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjcuMyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzY2NjY2NiI+PC9jaXJjbGU+PC9zdmc+) |
| **Without step 3** 1: Air time is between 20 and 700 minutes 2: Distance is greater than zero | 200 5.0% | 0.9809 ± 0.0011 Δ vs full +0.0048 | +0.0046 \[+0.0025, +0.0067\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTUuOCIgeTE9IjYuMCIgeDI9IjY1LjUiIHkyPSI2LjAiIHN0cm9rZT0iIzY2NjY2NiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjAuNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzY2NjY2NiI+PC9jaXJjbGU+PC9zdmc+) |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

The R² tells the same story for steps 1 and 3, but it rates removing all the failing rows as worse than removing random ones. That's because it's based on squared errors, so the large errors on the long flights count for more. Metrics can disagree like this, so it's worth deciding on the metric for the verdicts before running the ablation, rather than picking the one with the most favorable result.

The [`Ablation.to_json()`](../../reference/Ablation.to_json.md#pointblank.Ablation.to_json) method (or [`Ablation.to_dict()`](../../reference/Ablation.to_dict.md#pointblank.Ablation.to_dict)) gives everything about the ablation: its settings, the variants, every evaluation's results, and the effects. This is useful for keeping a record of the experiment alongside the model.

## How the Comparisons Are Made

For each variant, the difference from its control is computed for each repeat, and the report shows the mean of these differences with a paired *t*-interval (95% by default, set with `conf_level=`). The p-values are adjusted for the number of variants with the Holm method, and the verdicts use the adjusted p-values: a variant helps or hurts only if its adjusted p-value is below `1 - conf_level`. You can turn off the adjustment with `adjust=None`.

With fewer than three repeats, there's too little data for a useful interval, so the report gives no verdicts (and [Ablation](../../reference/Ablation.md#pointblank.Ablation) warns about this). More repeats give narrower intervals, at the cost of more evaluations. A few other options are worth knowing about:

- `control=None` compares variants only to the full table, which saves evaluations but leaves the effect of removing *any* rows mixed in with the effect of removing these rows
- `max_removed=` (by default `0.5`) sets the share of rows above which a variant gives a warning, since a version of the table missing most of its rows is a different dataset rather than an ablation of this one
- `duplicates=` and `duplicates_order_by=` say which copy of a duplicate row a [`Validate.rows_distinct()`](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step should keep (by default the first one, as with [`Validate.get_row_flags()`](../../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags)); see [Sundering Validated Data](sundering.md) for more on these

Only steps that check rows can be used in an ablation. Steps that check the table as a whole (like [`Validate.col_exists()`](../../reference/Validate.col_exists.md#pointblank.Validate.col_exists)), inactive steps, and steps using a `pre=` function that isn't marked with [`row_preserving()`](../../reference/row_preserving.md#pointblank.row_preserving) are left out, and the plan and report have a note for each step that couldn't be used.

## Conclusion

A validation can say which rows break its rules, but only an experiment can say whether those rows hurt the model trained on them. An [Ablation](../../reference/Ablation.md#pointblank.Ablation) runs that experiment with controls, repeats, and intervals, so you can find the checks that are worth enforcing, the ones that don't matter, and the ones that remove good data. It works with any model that you can wrap in an `evaluate` function, and with any validation whose steps check rows.
