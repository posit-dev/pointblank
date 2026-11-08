# Data Ablation

A validation tells you which rows of a table fail its checks. If the table is training data for a model, the next question is whether those rows actually matter: does removing them make the model better, worse, or no different? Data ablation answers that by experiment. The model is trained on the full table and on versions of it with the failing rows removed, and each version is scored on the same held-out data. Pointblank's [`Ablation`](../../reference/Ablation.md#pointblank.Ablation) class runs that experiment from an interrogated validation and reports the effect of each step's rows.

Removing rows changes the model even when the rows were perfectly good, since there's less data to learn from. So each version is also compared to a *control*, which removes the same number of rows at random. A step whose rows are really a problem does better than its control, and a step whose rows are good data does worse.

This article uses [scikit-learn](https://scikit-learn.org/) for the model, so you'll need it installed (`pip install "pointblank[ml]"` installs it along with Pointblank) to run the examples. An ablation doesn't depend on it: the model is just a function that you write.

## The Data and the Model

The `nycflights` dataset has a row for each flight that left New York City airports in 2013. We'll train a model to predict a flight's `air_time` (in minutes) from its `distance` (in miles) and the `month` and `hour` it left. Here we take 6,000 flights, keeping 2,000 of them aside as the held-out data for scoring the model. The `origin` airport isn't used by the model, but we'll use it later to look at groups of flights.

``` python
import json

import polars as pl

flights = (
    pb.load_dataset(dataset="nycflights")
    .drop_nulls("air_time")
    .select("month", "hour", "origin", "distance", "air_time")
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

| PolarsRows4,000Columns5 |  |  |  |  |  |
|----|----|----|----|----|----|
|  | month *Int64* | hour *Int64* | origin *String* | distance *Int64* | air_time *Float64* |
| 1 | 3 | 16 | JFK | 1391 | 210.0 |
| 2 | 8 | 15 | JFK | 1089 | 145.0 |
| 3 | 5 | 16 | LGA | 762 | 123.0 |
| 4 | 10 | 8 | JFK | 340 | 61.0 |
| 5 | 5 | 17 | JFK | 0 | 56.0 |
| 3996 | 12 | 12 | LGA | 544 | 88.0 |
| 3997 | 7 | 17 | LGA | 502 | 78.0 |
| 3998 | 6 | 15 | EWR | 0 | 112.0 |
| 3999 | 11 | 6 | JFK | 1598 | 183.0 |
| 4000 | 2 | 16 | JFK | 944 | 135.0 |

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

## Leaving One Step Out

A step's variant shows the effect of its rows on their own, but the steps of a validation are used together, and their rows overlap. The [`Ablation.drop_all_but_each()`](../../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each) method asks what each step adds to the others: for each step, it adds a variant that removes the rows failing every *other* step. The difference between that variant and the one removing every step's rows is what the left-out step's rows add on top of the rest:

``` python
ablation.drop_all_but_each().run()
```

| Ablation report Air time model |  |  |  |
|----|----|----|----|
| mae on the full table (4,000 rows): **9.122 ± 0.059** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. For leaving one out, Δ is what the left-out step's rows add to the other steps: removing every step's rows, compared with removing the other steps' rows, plus as many random rows. |  |  |  |
| Variant | Removed | mae | Δ vs random |
| Variants |  |  |  |
| **Step 1** Air time is between 20 and 700 minutes | 80 2.0% | 7.946 ± 0.085 Δ vs full −1.176 | −1.217 \[−1.404, −1.030\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMTYuMiIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMC4xIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| **Step 2** Distance is greater than zero | 122 3.0% | 9.153 ± 0.078 Δ vs full +0.031 | +0.039 \[−0.117, +0.195\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNDYuMiIgeTE9IjYuMCIgeDI9IjU2LjQiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTEuMyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **Step 3** Distance is at most 2,500 miles | 182 4.5% | 9.617 ± 0.078 Δ vs full +0.495 | +0.312 \[+0.131, +0.493\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTQuMyIgeTE9IjYuMCIgeDI9IjY2LjEiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjAuMiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **All steps** 1: Air time is between 20 and 700 minutes 2: Distance is greater than zero 3: Distance is at most 2,500 miles | 380 9.5% | 8.487 ± 0.010 Δ vs full −0.635 | −0.786 \[−0.943, −0.629\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTkuMSIgeTE9IjYuMCIgeDI9IjI5LjQiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjQuMiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Without step 3** 1: Air time is between 20 and 700 minutes 2: Distance is greater than zero | 200 5.0% | 8.012 ± 0.030 Δ vs full −1.110 | −1.108 \[−1.178, −1.038\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTEuNCIgeTE9IjYuMCIgeDI9IjE2LjAiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMTMuNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| Leave one out |  |  |  |
| **All steps but step 1** Leaves out step 1: Air time is between 20 and 700 minutes | 304 7.6% | 9.620 ± 0.051 Δ vs full +0.498 | −1.186 \[−1.262, −1.109\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iOC42IiB5MT0iNi4wIiB4Mj0iMTMuNyIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMS4yIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| **All steps but step 2** Leaves out step 2: Distance is greater than zero | 260 6.5% | 8.488 ± 0.017 Δ vs full −0.634 | −0.002 \[−0.040, +0.037\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNDguNyIgeTE9IjYuMCIgeDI9IjUxLjIiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNDkuOSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **All steps but step 3** Leaves out step 3: Distance is at most 2,500 miles Same rows as Without step 3, so it isn't evaluated again | 200 5.0% | 8.012 ± 0.030 Δ vs full −1.110 | +0.428 \[+0.287, +0.570\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTkuNCIgeTE9IjYuMCIgeDI9IjY4LjciIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjQuMCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

These variants have a group of their own in the report, and their **Δ** is what the left-out step's rows add, compared with removing as many random rows on top of the other steps' rows. Step 1's rows help on top of the others and step 3's rows hurt, as they do on their own, while step 2's rows make no clear difference either way. Leaving out step 3 removes the same rows as the `"Without step 3"` variant, so that version of the table isn't evaluated again.

A step whose rows help on their own but add nothing on top of the other steps gets the verdict *redundant*: the other steps already remove the rows that matter. Such a step can be dropped from the cleaning without losing anything, even though its variant helps.

## Removing Groups of Rows

The variants don't have to come from the steps. The [`Ablation.drop_groups()`](../../reference/Ablation.drop_groups.md#pointblank.Ablation.drop_groups) method adds a variant for each value of a column, removing the rows with that value. This shows whether some part of the data (a source, a region, a period) is worse than the rest, which a validation can't always say. Here's an ablation that removes the flights from each of the three airports:

``` python
by_origin = (
    pb.Ablation(
        validation,
        evaluate=evaluate,
        metric="mae",
        higher_is_better=False,
        label="Air time model",
    )
    .drop_groups("origin")
    .run()
)

by_origin
```

| Ablation report Air time model |  |  |  |
|----|----|----|----|
| mae on the full table (4,000 rows): **9.12 ± 0.06** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | mae | Δ vs random |
| **origin = EWR** All the rows of this group | 1,449 36.2% | 10.04 ± 0.04 Δ vs full +0.92 | +0.68 \[+0.42, +0.94\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNzAuNyIgeTE9IjYuMCIgeDI9Ijk2LjAiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iODMuMyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **origin = JFK** All the rows of this group | 1,303 32.6% | 9.70 ± 0.06 Δ vs full +0.57 | +0.22 \[−0.19, +0.64\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNDAuNiIgeTE9IjYuMCIgeDI9IjgxLjEiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjAuOSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **origin = LGA** All the rows of this group | 1,248 31.2% | 9.51 ± 0.09 Δ vs full +0.39 | −0.11 \[−0.33, +0.10\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMzQuMSIgeTE9IjYuMCIgeDI9IjU0LjciIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNDQuNCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

None of the airports' flights help the model when they're removed, so none of them are a problem. Removing the flights from Newark (`EWR`) hurts more than removing as many random flights, so they seem to have something to teach the model that the flights from the other airports don't. The `values=` parameter picks the groups to try, for a column with many distinct values.

## Sweeping a Threshold

A step's variant tests one setting of its check, but the setting itself is often the question: is 2,500 miles the right limit, or would a higher one do better? The [`Ablation.sweep_step()`](../../reference/Ablation.sweep_step.md#pointblank.Ablation.sweep_step) method adds a variant for each of a list of values of one of a step's parameters, each removing the rows that fail the step with that value. Here's the limit of step 3 at four values:

``` python
by_limit = (
    pb.Ablation(
        validation,
        evaluate=evaluate,
        metric="mae",
        higher_is_better=False,
        label="Air time model",
    )
    .sweep_step(3, "value", [1500, 2000, 2500, 3000])
    .run()
)

by_limit
```

| Ablation report Air time model |  |  |  |
|----|----|----|----|
| mae on the full table (4,000 rows): **9.12 ± 0.06** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | mae | Δ vs random |
| Sweep: Step 3, varying value |  |  |  |
| **Step 3 with value=1500** The rows that fail step 3 with `value=1500` | 850 21.2% | 28.55 ± 0.16 Δ vs full +19.43 | +19.20 \[+18.97, +19.43\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iOTQuOSIgeTE9IjYuMCIgeDI9Ijk2LjAiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iOTUuNSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **Step 3 with value=2000** The rows that fail step 3 with `value=2000` | 608 15.2% | 16.19 ± 0.08 Δ vs full +7.07 | +6.76 \[+6.53, +7.00\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNjUuNSIgeTE9IjYuMCIgeDI9IjY2LjYiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjYuMCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **Step 3 with value=2500** The rows that fail step 3 with `value=2500` | 182 4.5% | 9.62 ± 0.08 Δ vs full +0.49 | +0.31 \[+0.13, +0.49\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTAuMyIgeTE9IjYuMCIgeDI9IjUxLjIiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTAuNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **Step 3 with value=3000** The rows that fail step 3 with `value=3000` | 7 0.2% | 9.37 ± 0.05 Δ vs full +0.25 | +0.26 \[+0.23, +0.30\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTAuNSIgeTE9IjYuMCIgeDI9IjUwLjciIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTAuNiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |
| ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDYwIiBoZWlnaHQ9IjIzMCIgdmlld2JveD0iMCAwIDQ2MCAyMzAiIHN0eWxlPSJkaXNwbGF5OiBibG9jazsgbWFyZ2luLXRvcDogNHB4OyI+PHRleHQgeD0iMCIgeT0iMTMiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsgZm9udC1zaXplOiAxMXB4OyBmaWxsOiAjMzMzMzMzOyI+U3RlcCAzLCB2YXJ5aW5nIHZhbHVlOiDOlCBtYWUgdnMgZnVsbCAobG93ZXIgaXMgYmV0dGVyKTwvdGV4dD48bGluZSB4MT0iNTgiIHkxPSI0NCIgeDI9IjU4IiB5Mj0iMTkwIiBzdHJva2U9IiNCQkJCQkIiPjwvbGluZT48bGluZSB4MT0iNTgiIHkxPSIxOTAiIHgyPSI0NDQiIHkyPSIxOTAiIHN0cm9rZT0iI0JCQkJCQiI+PC9saW5lPjxsaW5lIHgxPSI1OCIgeTE9IjE2Mi4xIiB4Mj0iNDQ0IiB5Mj0iMTYyLjEiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PGxpbmUgeDE9IjU4IiB5MT0iMTkwLjAiIHgyPSI0NDQiIHkyPSIxOTAuMCIgc3Ryb2tlPSIjRUVFRUVFIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1MyIgeT0iMTkzLjAiIHRleHQtYW5jaG9yPSJlbmQiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPuKIkjU8L3RleHQ+PHRleHQgeD0iNTMiIHk9IjE2NS4xIiB0ZXh0LWFuY2hvcj0iZW5kIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4rMDwvdGV4dD48bGluZSB4MT0iNTgiIHkxPSIxMzQuMyIgeDI9IjQ0NCIgeTI9IjEzNC4zIiBzdHJva2U9IiNFRUVFRUUiIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjUzIiB5PSIxMzcuMyIgdGV4dC1hbmNob3I9ImVuZCIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+KzU8L3RleHQ+PGxpbmUgeDE9IjU4IiB5MT0iMTA2LjQiIHgyPSI0NDQiIHkyPSIxMDYuNCIgc3Ryb2tlPSIjRUVFRUVFIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1MyIgeT0iMTA5LjQiIHRleHQtYW5jaG9yPSJlbmQiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPisxMDwvdGV4dD48bGluZSB4MT0iNTgiIHkxPSI3OC41IiB4Mj0iNDQ0IiB5Mj0iNzguNSIgc3Ryb2tlPSIjRUVFRUVFIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1MyIgeT0iODEuNSIgdGV4dC1hbmNob3I9ImVuZCIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+KzE1PC90ZXh0PjxsaW5lIHgxPSI1OCIgeTE9IjUwLjciIHgyPSI0NDQiIHkyPSI1MC43IiBzdHJva2U9IiNFRUVFRUUiIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjUzIiB5PSI1My43IiB0ZXh0LWFuY2hvcj0iZW5kIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4rMjA8L3RleHQ+PHRleHQgeD0iNzIuMyIgeT0iMjAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4xNTAwPC90ZXh0Pjx0ZXh0IHg9IjE5MS40IiB5PSIyMDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPjIwMDA8L3RleHQ+PHRleHQgeD0iMzEwLjYiIHk9IjIwMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+MjUwMDwvdGV4dD48dGV4dCB4PSI0MjkuNyIgeT0iMjAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4zMDAwPC90ZXh0Pjx0ZXh0IHg9IjI1MS4wIiB5PSIyMjQiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPnZhbHVlIChzdGVwIDMpPC90ZXh0Pjxwb2x5bGluZSBwb2ludHM9IjcyLjMsMTYwLjkgMTkxLjQsMTYwLjQgMzEwLjYsMTYxLjEgNDI5LjcsMTYyLjIiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzg4ODg4OCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjQsMyI+PC9wb2x5bGluZT48cG9seWxpbmUgcG9pbnRzPSI3Mi4zLDUzLjggMTkxLjQsMTIyLjcgMzEwLjYsMTU5LjQgNDI5LjcsMTYwLjciIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzMzMzMzMyIgc3Ryb2tlLXdpZHRoPSIxLjUiPjwvcG9seWxpbmU+PGxpbmUgeDE9IjcyLjMiIHkxPSI1NC45IiB4Mj0iNzIuMyIgeTI9IjUyLjgiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNzIuMyIgY3k9IjUzLjgiIHI9IjMuNSIgZmlsbD0iI0NGMTQyQiI+PHRpdGxlPlN0ZXAgMyB3aXRoIHZhbHVlPTE1MDA6ICsxOS40MzwvdGl0bGU+PC9jaXJjbGU+PGxpbmUgeDE9IjE5MS40IiB5MT0iMTIzLjEiIHgyPSIxOTEuNCIgeTI9IjEyMi4zIiBzdHJva2U9IiNDRjE0MkIiIHN0cm9rZS13aWR0aD0iMiI+PC9saW5lPjxjaXJjbGUgY3g9IjE5MS40IiBjeT0iMTIyLjciIHI9IjMuNSIgZmlsbD0iI0NGMTQyQiI+PHRpdGxlPlN0ZXAgMyB3aXRoIHZhbHVlPTIwMDA6ICs3LjA3PC90aXRsZT48L2NpcmNsZT48bGluZSB4MT0iMzEwLjYiIHkxPSIxNjAuMSIgeDI9IjMxMC42IiB5Mj0iMTU4LjYiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMzEwLjYiIGN5PSIxNTkuNCIgcj0iMy41IiBmaWxsPSIjQ0YxNDJCIj48dGl0bGU+U3RlcCAzIHdpdGggdmFsdWU9MjUwMDogKzAuNDk8L3RpdGxlPjwvY2lyY2xlPjxsaW5lIHgxPSI0MjkuNyIgeTE9IjE2MS4yIiB4Mj0iNDI5LjciIHkyPSIxNjAuMiIgc3Ryb2tlPSIjQ0YxNDJCIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSI0MjkuNyIgY3k9IjE2MC43IiByPSIzLjUiIGZpbGw9IiNDRjE0MkIiPjx0aXRsZT5TdGVwIDMgd2l0aCB2YWx1ZT0zMDAwOiArMC4yNTwvdGl0bGU+PC9jaXJjbGU+PGxpbmUgeDE9IjAiIHkxPSIyNiIgeDI9IjE4IiB5Mj0iMjYiIHN0cm9rZT0iIzg4ODg4OCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjQsMyI+PC9saW5lPjx0ZXh0IHg9IjIyIiB5PSIyOSIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+UmVtb3ZpbmcgYXMgbWFueSByYW5kb20gcm93czwvdGV4dD48L3N2Zz4=) |  |  |  |

The variants of a sweep are a group of their own in the report, and below the table, a curve shows the change from the full table at each value, with a dashed line for the controls (removing as many random rows). The lower the limit, the more long flights are removed and the worse the model gets. Even a limit of 3,000 miles, which only removes the seven flights to Honolulu, hurts. No limit helps, so this check shouldn't be used to clean the data at all.

A check whose limit is a model's score is the same question: a score is a matter of degree, and where to draw the line is a choice. A [`Validate.col_vals_scored()`](../../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) step (see [Scored Validation](../advanced-validation/scored-validation.md)) gives each row a score and keeps the scores. The [`Ablation.sweep_score()`](../../reference/Ablation.sweep_score.md#pointblank.Ablation.sweep_score) method then removes more and more of the rows with the highest (or lowest) scores, whatever the step's own bound. Here, the score of a flight is the average speed implied by its distance and air time, which is far too high for the flights with their air times in hours:

``` python
def speed(batch):
    return [row["distance"] / (row["air_time"] / 60) for row in batch]


scored_validation = (
    pb.Validate(data=train, tbl_name="flights_train")
    .col_vals_scored(
        columns=["distance", "air_time"],
        scorer=speed,
        max_val=600,
        brief="Implied speed is at most 600 mph",
    )
    .interrogate()
)

by_speed = (
    pb.Ablation(
        scored_validation,
        evaluate=evaluate,
        metric="mae",
        higher_is_better=False,
        label="Air time model",
    )
    .sweep_score(1, quantiles=[0.01, 0.02, 0.03, 0.05, 0.1, 0.2])
    .run()
)

by_speed
```

| Ablation report Air time model |  |  |  |
|----|----|----|----|
| mae on the full table (4,000 rows): **9.122 ± 0.059** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | mae | Δ vs random |
| Sweep: Step 1, removing the highest-scoring rows |  |  |  |
| **Step 1: highest 1% of scores** The 40 rows with the highest scores (of 4,000 scored rows) | 40 1.0% | 8.231 ± 0.052 Δ vs full −0.891 | −0.940 \[−1.017, −0.863\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTUuMiIgeTE9IjYuMCIgeDI9IjIwLjUiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMTcuOSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Step 1: highest 2% of scores** The 80 rows with the highest scores (of 4,000 scored rows) | 80 2.0% | 7.951 ± 0.036 Δ vs full −1.171 | −1.212 \[−1.345, −1.078\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMTMuMSIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSI4LjYiIGN5PSI2LjAiIHI9IjMiIGZpbGw9IiM0Q0E2NEMiPjwvY2lyY2xlPjwvc3ZnPg==) helps |
| **Step 1: highest 3% of scores** The 120 rows with the highest scores (of 4,000 scored rows) | 120 3.0% | 8.276 ± 0.025 Δ vs full −0.846 | −0.866 \[−1.049, −0.684\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTQuMSIgeTE9IjYuMCIgeDI9IjI2LjYiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjAuNCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Step 1: highest 5% of scores** The 200 rows with the highest scores (of 4,000 scored rows) | 200 5.0% | 8.500 ± 0.020 Δ vs full −0.622 | −0.620 \[−0.721, −0.518\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMjUuMyIgeTE9IjYuMCIgeDI9IjMyLjMiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjguOCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Step 1: highest 10% of scores** The 400 rows with the highest scores (of 4,000 scored rows) | 400 10.0% | 8.973 ± 0.032 Δ vs full −0.149 | −0.148 \[−0.343, +0.047\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMzguMyIgeTE9IjYuMCIgeDI9IjUxLjYiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNDQuOSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **Step 1: highest 20% of scores** The 800 rows with the highest scores (of 4,000 scored rows) | 800 20.0% | 9.802 ± 0.025 Δ vs full +0.679 | +0.360 \[+0.159, +0.561\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTUuNCIgeTE9IjYuMCIgeDI9IjY5LjIiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjIuMyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |
| ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDYwIiBoZWlnaHQ9IjIzMCIgdmlld2JveD0iMCAwIDQ2MCAyMzAiIHN0eWxlPSJkaXNwbGF5OiBibG9jazsgbWFyZ2luLXRvcDogNHB4OyI+PHRleHQgeD0iMCIgeT0iMTMiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsgZm9udC1zaXplOiAxMXB4OyBmaWxsOiAjMzMzMzMzOyI+U3RlcCAxLCByZW1vdmluZyB0aGUgaGlnaGVzdC1zY29yaW5nIHJvd3M6IM6UIG1hZSB2cyBmdWxsIChsb3dlciBpcyBiZXR0ZXIpPC90ZXh0PjxsaW5lIHgxPSI1OCIgeTE9IjQ0IiB4Mj0iNTgiIHkyPSIxOTAiIHN0cm9rZT0iI0JCQkJCQiI+PC9saW5lPjxsaW5lIHgxPSI1OCIgeTE9IjE5MCIgeDI9IjQ0NCIgeTI9IjE5MCIgc3Ryb2tlPSIjQkJCQkJCIj48L2xpbmU+PGxpbmUgeDE9IjU4IiB5MT0iOTIuNyIgeDI9IjQ0NCIgeTI9IjkyLjciIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PGxpbmUgeDE9IjU4IiB5MT0iMTkwLjAiIHgyPSI0NDQiIHkyPSIxOTAuMCIgc3Ryb2tlPSIjRUVFRUVFIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1MyIgeT0iMTkzLjAiIHRleHQtYW5jaG9yPSJlbmQiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPuKIkjI8L3RleHQ+PGxpbmUgeDE9IjU4IiB5MT0iMTQxLjMiIHgyPSI0NDQiIHkyPSIxNDEuMyIgc3Ryb2tlPSIjRUVFRUVFIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1MyIgeT0iMTQ0LjMiIHRleHQtYW5jaG9yPSJlbmQiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPuKIkjE8L3RleHQ+PHRleHQgeD0iNTMiIHk9Ijk1LjciIHRleHQtYW5jaG9yPSJlbmQiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPiswPC90ZXh0PjxsaW5lIHgxPSI1OCIgeTE9IjQ0LjAiIHgyPSI0NDQiIHkyPSI0NC4wIiBzdHJva2U9IiNFRUVFRUUiIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjUzIiB5PSI0Ny4wIiB0ZXh0LWFuY2hvcj0iZW5kIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4rMTwvdGV4dD48dGV4dCB4PSI3Mi4zIiB5PSIyMDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPjElPC90ZXh0Pjx0ZXh0IHg9IjkxLjEiIHk9IjIwMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+MiU8L3RleHQ+PHRleHQgeD0iMTA5LjkiIHk9IjIwMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+MyU8L3RleHQ+PHRleHQgeD0iMTQ3LjUiIHk9IjIwMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+NSU8L3RleHQ+PHRleHQgeD0iMjQxLjYiIHk9IjIwMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+MTAlPC90ZXh0Pjx0ZXh0IHg9IjQyOS43IiB5PSIyMDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPjIwJTwvdGV4dD48dGV4dCB4PSIyNTEuMCIgeT0iMjI0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij5TaGFyZSBvZiB0aGUgc2NvcmVkIHJvd3MgcmVtb3ZlZDwvdGV4dD48cG9seWxpbmUgcG9pbnRzPSI3Mi4zLDkwLjMgOTEuMSw5MC43IDEwOS45LDkxLjcgMTQ3LjUsOTIuOCAyNDEuNiw5Mi43IDQyOS43LDc3LjEiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzg4ODg4OCIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1kYXNoYXJyYXk9IjQsMyI+PC9wb2x5bGluZT48cG9seWxpbmUgcG9pbnRzPSI3Mi4zLDEzNi4wIDkxLjEsMTQ5LjYgMTA5LjksMTMzLjggMTQ3LjUsMTIyLjkgMjQxLjYsOTkuOSA0MjkuNyw1OS42IiBmaWxsPSJub25lIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMS41Ij48L3BvbHlsaW5lPjxsaW5lIHgxPSI3Mi4zIiB5MT0iMTM4LjYiIHgyPSI3Mi4zIiB5Mj0iMTMzLjQiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNzIuMyIgY3k9IjEzNi4wIiByPSIzLjUiIGZpbGw9IiM0Q0E2NEMiPjx0aXRsZT5TdGVwIDE6IGhpZ2hlc3QgMSUgb2Ygc2NvcmVzOiDiiJIwLjg5MTwvdGl0bGU+PC9jaXJjbGU+PGxpbmUgeDE9IjkxLjEiIHkxPSIxNTQuMiIgeDI9IjkxLjEiIHkyPSIxNDUuMSIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSI5MS4xIiBjeT0iMTQ5LjYiIHI9IjMuNSIgZmlsbD0iIzRDQTY0QyI+PHRpdGxlPlN0ZXAgMTogaGlnaGVzdCAyJSBvZiBzY29yZXM6IOKIkjEuMTcxPC90aXRsZT48L2NpcmNsZT48bGluZSB4MT0iMTA5LjkiIHkxPSIxMzguMCIgeDI9IjEwOS45IiB5Mj0iMTI5LjciIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMTA5LjkiIGN5PSIxMzMuOCIgcj0iMy41IiBmaWxsPSIjNENBNjRDIj48dGl0bGU+U3RlcCAxOiBoaWdoZXN0IDMlIG9mIHNjb3Jlczog4oiSMC44NDY8L3RpdGxlPjwvY2lyY2xlPjxsaW5lIHgxPSIxNDcuNSIgeTE9IjEyNi44IiB4Mj0iMTQ3LjUiIHkyPSIxMTkuMSIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxNDcuNSIgY3k9IjEyMi45IiByPSIzLjUiIGZpbGw9IiM0Q0E2NEMiPjx0aXRsZT5TdGVwIDE6IGhpZ2hlc3QgNSUgb2Ygc2NvcmVzOiDiiJIwLjYyMjwvdGl0bGU+PC9jaXJjbGU+PGxpbmUgeDE9IjI0MS42IiB5MT0iMTAyLjgiIHgyPSIyNDEuNiIgeTI9Ijk3LjEiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjQxLjYiIGN5PSI5OS45IiByPSIzLjUiIGZpbGw9IiM5OTk5OTkiPjx0aXRsZT5TdGVwIDE6IGhpZ2hlc3QgMTAlIG9mIHNjb3Jlczog4oiSMC4xNDk8L3RpdGxlPjwvY2lyY2xlPjxsaW5lIHgxPSI0MjkuNyIgeTE9IjYyLjYiIHgyPSI0MjkuNyIgeTI9IjU2LjYiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNDI5LjciIGN5PSI1OS42IiByPSIzLjUiIGZpbGw9IiNDRjE0MkIiPjx0aXRsZT5TdGVwIDE6IGhpZ2hlc3QgMjAlIG9mIHNjb3JlczogKzAuNjc5PC90aXRsZT48L2NpcmNsZT48bGluZSB4MT0iMCIgeTE9IjI2IiB4Mj0iMTgiIHkyPSIyNiIgc3Ryb2tlPSIjODg4ODg4IiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWRhc2hhcnJheT0iNCwzIj48L2xpbmU+PHRleHQgeD0iMjIiIHk9IjI5IiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij5SZW1vdmluZyBhcyBtYW55IHJhbmRvbSByb3dzPC90ZXh0Pjwvc3ZnPg==) |  |  |  |

Each variant removes a share of the rows with the highest scores (the step has only a `max_val=`, so it's the highest that are removed; `remove="lowest"` would remove the lowest instead). The curve is lowest at 2%, which is the share of flights with their air times in hours: removing fewer leaves some of them in, and removing more starts to remove good flights. By 20%, the sweep hurts. A sweep can also use the values of a column of the table as the scores (`sweep_score("column_name")`), or cutoffs on the scale of the scores rather than shares of the rows (`values=`).

## Injecting Defects

The variants so far remove rows that a check flags, which asks whether the flagged rows hurt the model. The opposite question is just as useful: if a defect got into the data, how much would it hurt? The [`Ablation.inject()`](../../reference/Ablation.inject.md#pointblank.Ablation.inject) method answers it by adding the defect to clean rows (those that fail no step) and evaluating the corrupted table. The defect can be one of four kinds:

- `"null"`: the values of a column are made missing
- `"shuffle"`: the values of a column are moved between rows, so each is a valid value in the wrong row
- `"truncate"`: text values are cut to half their length
- `"duplicate"`: rows are copied

or a function, which is applied to the values of a column. Here, the defect is the problem that step 1 catches, air times in hours, made by a function that divides the air time by 60:

``` python
injected = (
    pb.Ablation(
        validation,
        evaluate=evaluate,
        metric="mae",
        higher_is_better=False,
        label="Air time model",
    )
    .inject(lambda minutes: minutes / 60, step=1, name="hours", rates=[0.01, 0.02, 0.05, 0.1])
    .run()
)

injected
```

| Ablation report Air time model |  |  |  |
|----|----|----|----|
| mae on the full table (4,000 rows): **9.12 ± 0.06** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from the full table. |  |  |  |
| Variant | Removed | mae | Δ vs full |
| Injected: hours on air_time |  |  |  |
| **Hours on air_time: 1%** 40 clean rows have `hours()` applied to `air_time` (checked by step 1) | 40 1.0% corrupted | 9.89 ± 0.06 | +0.77 \[+0.71, +0.83\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTMuMiIgeTE9IjYuMCIgeDI9IjUzLjciIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTMuNCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **Hours on air_time: 2%** 80 clean rows have `hours()` applied to `air_time` (checked by step 1) | 80 2.0% corrupted | 10.62 ± 0.05 | +1.50 \[+1.45, +1.56\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTYuNSIgeTE9IjYuMCIgeDI9IjU3LjAiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTYuNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **Hours on air_time: 5%** 200 clean rows have `hours()` applied to `air_time` (checked by step 1) | 200 5.0% corrupted | 13.58 ± 0.08 | +4.46 \[+4.40, +4.52\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNjkuNyIgeTE9IjYuMCIgeDI9IjcwLjIiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjkuOSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **Hours on air_time: 10%** 400 clean rows have `hours()` applied to `air_time` (checked by step 1) | 400 10.0% corrupted | 19.32 ± 0.12 | +10.20 \[+10.11, +10.30\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iOTUuMSIgeTE9IjYuMCIgeDI9Ijk2LjAiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iOTUuNiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |
| ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDYwIiBoZWlnaHQ9IjIzMCIgdmlld2JveD0iMCAwIDQ2MCAyMzAiIHN0eWxlPSJkaXNwbGF5OiBibG9jazsgbWFyZ2luLXRvcDogNHB4OyI+PHRleHQgeD0iMCIgeT0iMTMiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsgZm9udC1zaXplOiAxMXB4OyBmaWxsOiAjMzMzMzMzOyI+aG91cnMgb24gYWlyX3RpbWU6IM6UIG1hZSB2cyBmdWxsIChsb3dlciBpcyBiZXR0ZXIpPC90ZXh0PjxsaW5lIHgxPSI1OCIgeTE9IjQ0IiB4Mj0iNTgiIHkyPSIxOTAiIHN0cm9rZT0iI0JCQkJCQiI+PC9saW5lPjxsaW5lIHgxPSI1OCIgeTE9IjE5MCIgeDI9IjQ0NCIgeTI9IjE5MCIgc3Ryb2tlPSIjQkJCQkJCIj48L2xpbmU+PGxpbmUgeDE9IjU4IiB5MT0iMTgyLjQiIHgyPSI0NDQiIHkyPSIxODIuNCIgc3Ryb2tlPSIjOTk5OTk5IiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1MyIgeT0iMTg1LjQiIHRleHQtYW5jaG9yPSJlbmQiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPiswPC90ZXh0PjxsaW5lIHgxPSI1OCIgeTE9IjEzNi4zIiB4Mj0iNDQ0IiB5Mj0iMTM2LjMiIHN0cm9rZT0iI0VFRUVFRSIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PHRleHQgeD0iNTMiIHk9IjEzOS4zIiB0ZXh0LWFuY2hvcj0iZW5kIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4rNTwvdGV4dD48bGluZSB4MT0iNTgiIHkxPSI5MC4xIiB4Mj0iNDQ0IiB5Mj0iOTAuMSIgc3Ryb2tlPSIjRUVFRUVFIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1MyIgeT0iOTMuMSIgdGV4dC1hbmNob3I9ImVuZCIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+KzEwPC90ZXh0PjxsaW5lIHgxPSI1OCIgeTE9IjQ0LjAiIHgyPSI0NDQiIHkyPSI0NC4wIiBzdHJva2U9IiNFRUVFRUUiIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjUzIiB5PSI0Ny4wIiB0ZXh0LWFuY2hvcj0iZW5kIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4rMTU8L3RleHQ+PHRleHQgeD0iNzIuMyIgeT0iMjAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4xJTwvdGV4dD48dGV4dCB4PSIxMTIuMCIgeT0iMjAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4yJTwvdGV4dD48dGV4dCB4PSIyMzEuMSIgeT0iMjAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij41JTwvdGV4dD48dGV4dCB4PSI0MjkuNyIgeT0iMjAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4xMCU8L3RleHQ+PHRleHQgeD0iMjUxLjAiIHk9IjIyNCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+U2hhcmUgb2YgdGhlIHJvd3MgY29ycnVwdGVkPC90ZXh0Pjxwb2x5bGluZSBwb2ludHM9IjcyLjMsMTc1LjMgMTEyLjAsMTY4LjUgMjMxLjEsMTQxLjIgNDI5LjcsODguMyIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjEuNSI+PC9wb2x5bGluZT48bGluZSB4MT0iNzIuMyIgeTE9IjE3NS44IiB4Mj0iNzIuMyIgeTI9IjE3NC44IiBzdHJva2U9IiNDRjE0MkIiIHN0cm9rZS13aWR0aD0iMiI+PC9saW5lPjxjaXJjbGUgY3g9IjcyLjMiIGN5PSIxNzUuMyIgcj0iMy41IiBmaWxsPSIjQ0YxNDJCIj48dGl0bGU+SG91cnMgb24gYWlyX3RpbWU6IDElOiArMC43NzwvdGl0bGU+PC9jaXJjbGU+PGxpbmUgeDE9IjExMi4wIiB5MT0iMTY5LjAiIHgyPSIxMTIuMCIgeTI9IjE2OC4wIiBzdHJva2U9IiNDRjE0MkIiIHN0cm9rZS13aWR0aD0iMiI+PC9saW5lPjxjaXJjbGUgY3g9IjExMi4wIiBjeT0iMTY4LjUiIHI9IjMuNSIgZmlsbD0iI0NGMTQyQiI+PHRpdGxlPkhvdXJzIG9uIGFpcl90aW1lOiAyJTogKzEuNTA8L3RpdGxlPjwvY2lyY2xlPjxsaW5lIHgxPSIyMzEuMSIgeTE9IjE0MS44IiB4Mj0iMjMxLjEiIHkyPSIxNDAuNyIgc3Ryb2tlPSIjQ0YxNDJCIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIyMzEuMSIgY3k9IjE0MS4yIiByPSIzLjUiIGZpbGw9IiNDRjE0MkIiPjx0aXRsZT5Ib3VycyBvbiBhaXJfdGltZTogNSU6ICs0LjQ2PC90aXRsZT48L2NpcmNsZT48bGluZSB4MT0iNDI5LjciIHkxPSI4OS4yIiB4Mj0iNDI5LjciIHkyPSI4Ny40IiBzdHJva2U9IiNDRjE0MkIiIHN0cm9rZS13aWR0aD0iMiI+PC9saW5lPjxjaXJjbGUgY3g9IjQyOS43IiBjeT0iODguMyIgcj0iMy41IiBmaWxsPSIjQ0YxNDJCIj48dGl0bGU+SG91cnMgb24gYWlyX3RpbWU6IDEwJTogKzEwLjIwPC90aXRsZT48L2NpcmNsZT48L3N2Zz4=) |  |  |  |

Each rate is a share of the rows to corrupt, and each is a variant. The rows corrupted at a rate include those at the lower rates, so the curve shows the same defect getting worse. Since no rows are removed, there's no control: each variant is compared with the full table. Even 1% of the flights in hours makes the error almost a minute larger, and the harm grows steadily with the share.

The `step=` says which step checks for the defect (and its column is the one corrupted, unless `column=` says otherwise). The table must be a Polars, pandas, or PyArrow table, as the defect is made in memory, and the injected rows are the same in every repeat (they're chosen from the ablation's `seed=`).

## Suggesting Thresholds

A step's thresholds say how many failing rows are too many, but the right number depends on how much the failures cost. With a curve like the one above, that can be worked out: the [`Ablation.suggest_thresholds()`](../../reference/Ablation.suggest_thresholds.md#pointblank.Ablation.suggest_thresholds) method fits a line through the harm of each injected defect against the share of rows that fail its step, and gives the share at which the harm reaches a `tolerance=` (in the units of the metric):

``` python
injected.suggest_thresholds(tolerance=1)
```

    {1: Thresholds(warning=0.005, error=0.01, critical=None)}

If the model can lose at most a minute of mean absolute error, step 1 should give an error once 1% of its rows fail (with a warning at half that). The suggestions are dictionaries of [`Thresholds`](../../reference/Thresholds.md#pointblank.Thresholds) by step, ready for the `thresholds=` of the steps. A step whose defect doesn't clearly hurt the model, or that doesn't catch its defect (a shuffled distance, for example, is still greater than zero), gets no suggestion, with a warning that says why.

## Sharing the Effect Among the Steps

Removing every step's rows has an effect, but the steps' rows overlap and work together, so their effects on their own don't add up to it. The *Shapley value* of a step is its fair share of the whole effect: its average effect when it's added to every combination of the other steps. The [`Ablation.shapley()`](../../reference/Ablation.shapley.md#pointblank.Ablation.shapley) method adds a variant for each combination that's needed (reusing those already in the ablation), and the report shows a Shapley value for each step:

``` python
shares = (
    pb.Ablation(
        validation,
        evaluate=evaluate,
        metric="mae",
        higher_is_better=False,
        label="Air time model",
    )
    .shapley()
    .run()
)

shares
```

| Ablation report Air time model |  |  |  |
|----|----|----|----|
| mae on the full table (4,000 rows): **9.122 ± 0.059** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. For the Shapley values, Δ is each step's share of the effect of removing the rows of every step. |  |  |  |
| Variant | Removed | mae | Δ vs random |
| Shapley values |  |  |  |
| **Step 1** Air time is between 20 and 700 minutes | 80 2.0% | 148% of the total Δ | −1.166 \[−1.308, −1.024\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMTQuMCIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSI5LjAiIGN5PSI2LjAiIHI9IjMiIGZpbGw9IiM0Q0E2NEMiPjwvY2lyY2xlPjwvc3ZnPg==) helps |
| **Step 2** Distance is greater than zero | 122 3.0% | -1% of the total Δ | +0.011 \[−0.092, +0.114\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNDYuOCIgeTE9IjYuMCIgeDI9IjU0LjAiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTAuNCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **Step 3** Distance is at most 2,500 miles | 182 4.5% | -47% of the total Δ | +0.368 \[+0.291, +0.446\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNjAuMiIgeTE9IjYuMCIgeDI9IjY1LjciIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjMuMCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |
| The Shapley values are exact, from every order of the 3 steps. They add up to −0.786, the effect of removing the rows of every step compared with removing as many random rows. |  |  |  |

The Shapley values add up to the effect of removing the rows of every step, so each step gets a share of the total (in the **mae** column). Step 1's rows bring the error down by more than the whole cleaning does, and step 3's rows take back almost half of that, so the shares are above 100% and below zero. Step 2's share is close to nothing. The [`Ablation.get_shapley()`](../../reference/Ablation.get_shapley.md#pointblank.Ablation.get_shapley) method gives the values as a table, with their intervals and p-values for every metric.

With three steps, there are seven combinations to evaluate, and the values are exact. The number of combinations doubles with each step, so with more steps than `permutations=` allows (by default, those with at most 200 orders, or five steps), the values are estimated from that many random orders of the steps instead, and the report says so.

## Getting the Results

The [`Ablation.get_effects()`](../../reference/Ablation.get_effects.md#pointblank.Ablation.get_effects) method returns the results as a table, with a row for each variant and metric. Along with the differences from the controls, it has the differences from the full table, the interval bounds, and the p-values:

``` python
effects = ablation.get_effects()
mae = effects.filter(pl.col("metric") == "mae")

mae.select("name", "mean", "delta_vs_control", "p_adj_vs_control", "verdict")
```

shape: (9, 5)

| name | mean | delta_vs_control | p_adj_vs_control | verdict |
|----|----|----|----|----|
| str | f64 | f64 | f64 | str |
| "Full table" | 9.122164 | null | null | null |
| "Step 1" | 7.945899 | -1.217097 | 0.000219 | "helps" |
| "Step 2" | 9.152884 | 0.039022 | 0.5248 | "inconclusive" |
| "Step 3" | 9.617069 | 0.311815 | 0.017424 | "hurts" |
| "All steps" | 8.487428 | -0.78607 | 0.00047 | "helps" |
| "Without step 3" | 8.011877 | -1.107792 | 0.000008 | "helps" |
| "All steps but step 1" | 9.620453 | null | null | "helps" |
| "All steps but step 2" | 8.488381 | null | null | "inconclusive" |
| "All steps but step 3" | 8.011877 | null | null | "hurts" |

The verdicts are only given for the metric named in `metric=`, but the other metrics are evaluated too. The report can show any of them:

``` python
ablation.get_tabular_report(metric="r2")
```

| Ablation report Air time model |  |  |  |
|----|----|----|----|
| r2 on the full table (4,000 rows): **0.9761 ± 0.0014**. Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. For leaving one out, Δ is what the left-out step's rows add to the other steps: removing every step's rows, compared with removing the other steps' rows, plus as many random rows. Verdicts are based on mae, so none are shown here. |  |  |  |
| Variant | Removed | r2 | Δ vs random |
| Variants |  |  |  |
| **Step 1** Air time is between 20 and 700 minutes | 80 2.0% | 0.9822 ± 0.0021 Δ vs full +0.0062 | +0.0064 \[+0.0028, +0.0101\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTYuMCIgeTE9IjYuMCIgeDI9IjcxLjciIHkyPSI2LjAiIHN0cm9rZT0iIzY2NjY2NiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjMuOSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzY2NjY2NiI+PC9jaXJjbGU+PC9zdmc+) |
| **Step 2** Distance is greater than zero | 122 3.0% | 0.9754 ± 0.0021 Δ vs full −0.0007 | −0.0015 \[−0.0060, +0.0029\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMzcuMCIgeTE9IjYuMCIgeDI9IjU2LjMiIHkyPSI2LjAiIHN0cm9rZT0iIzY2NjY2NiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNDYuNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzY2NjY2NiI+PC9jaXJjbGU+PC9zdmc+) |
| **Step 3** Distance is at most 2,500 miles | 182 4.5% | 0.9573 ± 0.0004 Δ vs full −0.0188 | −0.0158 \[−0.0200, −0.0117\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNy4wIiB5MT0iNi4wIiB4Mj0iMjQuOSIgeTI9IjYuMCIgc3Ryb2tlPSIjNjY2NjY2IiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxNi4wIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNjY2NjY2Ij48L2NpcmNsZT48L3N2Zz4=) |
| **All steps** 1: Air time is between 20 and 700 minutes 2: Distance is greater than zero 3: Distance is at most 2,500 miles | 380 9.5% | 0.9625 ± 0.0001 Δ vs full −0.0136 | −0.0098 \[−0.0157, −0.0040\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTYuMyIgeTE9IjYuMCIgeDI9IjQxLjQiIHkyPSI2LjAiIHN0cm9rZT0iIzY2NjY2NiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjguOCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzY2NjY2NiI+PC9jaXJjbGU+PC9zdmc+) |
| **Without step 3** 1: Air time is between 20 and 700 minutes 2: Distance is greater than zero | 200 5.0% | 0.9809 ± 0.0011 Δ vs full +0.0048 | +0.0046 \[+0.0025, +0.0067\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTUuNCIgeTE9IjYuMCIgeDI9IjY0LjUiIHkyPSI2LjAiIHN0cm9rZT0iIzY2NjY2NiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjAuMCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzY2NjY2NiI+PC9jaXJjbGU+PC9zdmc+) |
| Leave one out |  |  |  |
| **All steps but step 1** Leaves out step 1: Air time is between 20 and 700 minutes | 304 7.6% | 0.9572 ± 0.0003 Δ vs full −0.0189 | +0.0055 \[+0.0050, +0.0060\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNjAuOCIgeTE9IjYuMCIgeDI9IjYyLjkiIHkyPSI2LjAiIHN0cm9rZT0iIzY2NjY2NiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjEuOCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzY2NjY2NiI+PC9jaXJjbGU+PC9zdmc+) |
| **All steps but step 2** Leaves out step 2: Distance is greater than zero | 260 6.5% | 0.9624 ± 0.0000 Δ vs full −0.0136 | +0.0001 \[−0.0001, +0.0003\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNDkuOSIgeTE9IjYuMCIgeDI9IjUwLjYiIHkyPSI2LjAiIHN0cm9rZT0iIzY2NjY2NiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTAuMiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzY2NjY2NiI+PC9jaXJjbGU+PC9zdmc+) |
| **All steps but step 3** Leaves out step 3: Distance is at most 2,500 miles Same rows as Without step 3, so it isn't evaluated again | 200 5.0% | 0.9809 ± 0.0011 Δ vs full +0.0048 | −0.0172 \[−0.0214, −0.0130\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjIuMSIgeTI9IjYuMCIgc3Ryb2tlPSIjNjY2NjY2IiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMy4wIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNjY2NjY2Ij48L2NpcmNsZT48L3N2Zz4=) |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

The R² tells the same story for steps 1 and 3, but it rates removing all the failing rows as worse than removing random ones. That's because it's based on squared errors, so the large errors on the long flights count for more. Metrics can disagree like this, so it's worth deciding on the metric for the verdicts before running the ablation, rather than picking the one with the most favorable result.

The [`Ablation.to_json()`](../../reference/Ablation.to_json.md#pointblank.Ablation.to_json) method (or [`Ablation.to_dict()`](../../reference/Ablation.to_dict.md#pointblank.Ablation.to_dict)) gives everything about the ablation: its settings, the variants, every evaluation's results, and the effects. This is useful for keeping a record of the experiment alongside the model.

The [`Ablation.annotate()`](../../reference/Ablation.annotate.md#pointblank.Ablation.annotate) method puts the results where the checks are: it adds a note to each step of the validation that has a variant, with its effect and verdict (and what its rows add on top of the other steps, with [`Ablation.drop_all_but_each()`](../../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each)). The notes appear in the validation's report:

``` python
ablation.annotate()

validation
```

[TABLE]

The validation can also be a newer one with the same steps (e.g., an interrogation of next month's training data), passed as `annotate(validation=...)`.

## A Ready-Made `evaluate` Function

Many `evaluate` functions for tabular models are much the same: pick the features, encode them, fit a model, and score it on the held-out data. The [`ablation.sklearn_evaluator()`](../../reference/ablation.sklearn_evaluator.md#pointblank.ablation.sklearn_evaluator) function makes one from the held-out data and the target. By default it fits a histogram-based gradient boosting model, which handles missing values and string features, and scores it with `"r2"` for a regression (or `"accuracy"` for classification). Any scikit-learn scorers can be used instead:

``` python
evaluate_boosting = pb.ablation.sklearn_evaluator(
    holdout,
    target="air_time",
    features=["month", "hour", "distance"],
    scoring=["neg_mean_absolute_error", "r2"],
)

boosting = (
    pb.Ablation(
        validation,
        evaluate=evaluate_boosting,
        metric="neg_mean_absolute_error",
        label="Air time model (gradient boosting)",
    )
    .drop_each()
    .run()
)

boosting
```

| Ablation report Air time model (gradient boosting) |  |  |  |
|----|----|----|----|
| neg_mean_absolute_error on the full table (4,000 rows): **−9.471 ± 0.000** (higher is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | neg_mean_absolute_error | Δ vs random |
| **Step 1** Air time is between 20 and 700 minutes | 80 2.0% | −8.373 ± 0.000 Δ vs full +1.098 | +1.152 \[+1.082, +1.222\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iOTAuNyIgeTE9IjYuMCIgeDI9Ijk2LjAiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iOTMuNCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Step 2** Distance is greater than zero | 122 3.0% | −9.498 ± 0.000 Δ vs full −0.028 | +0.082 \[+0.028, +0.137\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNTEuMCIgeTE9IjYuMCIgeDI9IjU1LjIiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTMuMSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Step 3** Distance is at most 2,500 miles | 182 4.5% | −9.723 ± 0.000 Δ vs full −0.252 | −0.158 \[−0.202, −0.114\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNDIuNCIgeTE9IjYuMCIgeDI9IjQ1LjciIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNDQuMSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

The scorers of scikit-learn are such that higher is better (which is why the mean absolute error is negated), so the default `higher_is_better=True` is the right one here. With this model, removing the flights with a zero distance helps too. Whether a problem in the data matters depends on the model, which is why an ablation is worth running with the model that will be trained on the data. Pass `estimator=` to use another model or pipeline: it's cloned for each evaluation, with its `random_state` set to the seed.

## Running Larger Ablations

The ablations here make a few dozen evaluations, but with a model that takes minutes to train, a run can take hours. A few things help with that.

Before a long run, it's worth checking that the `evaluate` function uses its seed for all of its randomness, since the paired comparisons rely on it. The [`Ablation.check_determinism()`](../../reference/Ablation.check_determinism.md#pointblank.Ablation.check_determinism) method evaluates the full table twice with the same seed, and warns if the results differ:

``` python
ablation.check_determinism()
```

    True

The `max_workers=` parameter of [`Ablation.run()`](../../reference/Ablation.run.md#pointblank.Ablation.run) makes the evaluations in parallel. They run in threads by default, which suits models that release Python's lock while they train (most of those in compiled libraries do), and `executor="process"` runs them in separate processes instead (this needs an `evaluate` function defined at the top level of a module). The random forest here already uses every core with `n_jobs=-1`, so it wouldn't gain much.

The `cache=` parameter of [Ablation](../../reference/Ablation.md#pointblank.Ablation) keeps each evaluation's result in a directory, so that a run that stops partway through (or a run with an added variant) only makes the evaluations it doesn't already have. A result is found again only for the same data, the same rows removed, and the same seed, so a change to the table gives new evaluations. A change to the model doesn't, since Pointblank can't see inside the `evaluate` function, so give it an `evaluate_version=` that you change along with the model:

``` python
ablation = pb.Ablation(
    validation,
    evaluate=evaluate,
    metric="mae",
    higher_is_better=False,
    cache=".ablation-cache",
    evaluate_version="random forest, 50 trees",
).drop_each()

ablation.run(max_workers=4)
```

The report says how many of the evaluations came from the cache.

## Evaluating Somewhere Else

Sometimes the model can't be trained in the Python session that has the validation: it's trained on a cluster, by another team, or in another language. The [`Ablation.export()`](../../reference/Ablation.export.md#pointblank.Ablation.export) method writes every version of the training table that a run would evaluate to a directory, as Parquet files, along with a `manifest.json` that lists each evaluation (its version of the table, its file, and its seed) and a `results_template.csv` to fill in. With `format="indices"`, it writes the row numbers that each version removes instead, for a table that's too large to copy.

Here, the evaluations are made by reading the files back and calling the same `evaluate` function, but any process that writes a row of metrics for each evaluation would do:

``` python
import tempfile
from pathlib import Path

plan = pb.Ablation(
    validation, metric="mae", higher_is_better=False, repeats=3
).drop_each(steps=1)

directory = plan.export(Path(tempfile.mkdtemp()) / "air-time-ablation")

manifest = json.loads((directory / "manifest.json").read_text())

results = pl.DataFrame(
    [
        {
            "variant": run["variant"],
            "repeat": run["repeat"],
            "seed": run["seed"],
            **evaluate(pl.read_parquet(directory / run["file"]), seed=run["seed"]),
        }
        for run in manifest["runs"]
    ]
)

results.write_csv(directory / "results.csv")

plan.load_results(directory)
```

| Ablation report |  |  |  |
|----|----|----|----|
| mae on the full table (4,000 rows): **9.129 ± 0.038** (lower is better). Each version was evaluated 3 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | mae | Δ vs random |
| **Step 1** Air time is between 20 and 700 minutes | 80 2.0% | 7.954 ± 0.113 Δ vs full −1.175 | −1.185 \[−1.688, −0.682\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMzEuNCIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxNy43IiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |
| The results were read with `load_results()`, from evaluations made elsewhere. |  |  |  |

The [`Ablation.load_results()`](../../reference/Ablation.load_results.md#pointblank.Ablation.load_results) method reads the results (from the directory, a CSV file, or a table) and computes the effects, just as a run would. It checks that there's a result for every evaluation, and, with the manifest, that the results are for the same data and the same plan.

## Ablations in YAML

A validation written in YAML can have an `ablation:` block, with the settings of the [Ablation](../../reference/Ablation.md#pointblank.Ablation) and its strategies (the methods that add variants). The `evaluate` function is given as the import path of a function in a module:

``` yaml
tbl: flights_train.parquet
tbl_name: flights_train
steps:
- col_vals_between:
    columns: air_time
    left: 20
    right: 700
- col_vals_gt:
    columns: distance
    value: 0
- col_vals_le:
    columns: distance
    value: 2500
ablation:
  evaluate: air_time_model:evaluate
  metric: mae
  higher_is_better: false
  label: Air time model
  strategies:
  - drop_each
  - drop_all_but_each
  - drop_groups:
      column: origin
```

The [`yaml_ablation()`](../../reference/yaml_ablation.md#pointblank.yaml_ablation) function interrogates the validation and returns the [Ablation](../../reference/Ablation.md#pointblank.Ablation), ready to run. Importing the function runs Python code, so it's only done for a configuration that you trust, with `allow_python=True`. Otherwise, the function can be passed in directly with `evaluate=`, as it is here (along with the training table, in place of the file):

``` python
config = """
tbl: flights_train.parquet
tbl_name: flights_train
steps:
- col_vals_between:
    columns: air_time
    left: 20
    right: 700
- col_vals_gt:
    columns: distance
    value: 0
- col_vals_le:
    columns: distance
    value: 2500
ablation:
  evaluate: air_time_model:evaluate
  metric: mae
  higher_is_better: false
  label: Air time model
  strategies:
  - drop_each
  - drop_all_but_each
  - drop_groups:
      column: origin
"""

pb.yaml_ablation(config, set_tbl=train, evaluate=evaluate).plan()
```

| Ablation plan Air time model |  |
|----|----|
| 4,000 rows in the training table. Each version is evaluated 5 times, so the run will make **105** evaluations. |  |
| Version | Removed |
| Baseline |  |
| **Full table** Nothing removed | 0 0% |
| Variants |  |
| **Step 1** Expect that values in `air_time` should be between `20` and `700`. | 80 2.0% |
| **Step 2** Expect that values in `distance` should be \> `0`. | 122 3.0% |
| **Step 3** Expect that values in `distance` should be \<= `2500`. | 182 4.5% |
| **All steps** 1: Expect that values in `air_time` should be between `20` and `700`. 2: Expect that values in `distance` should be \> `0`. 3: Expect that values in `distance` should be \<= `2500`. | 380 9.5% |
| **origin = EWR** All the rows of this group | 1,449 36.2% |
| **origin = JFK** All the rows of this group | 1,303 32.6% |
| **origin = LGA** All the rows of this group | 1,248 31.2% |
| Leave one out |  |
| **All steps but step 1** Leaves out step 1: Expect that values in `air_time` should be between `20` and `700`. | 304 7.6% |
| **All steps but step 2** Leaves out step 2: Expect that values in `distance` should be \> `0`. | 260 6.5% |
| **All steps but step 3** Leaves out step 3: Expect that values in `distance` should be \<= `2500`. | 200 5.0% |
| Controls |  |
| **80 random rows** Rows removed at random, drawn anew for each repeat | 80 2.0% |
| **122 random rows** Rows removed at random, drawn anew for each repeat | 122 3.0% |
| **182 random rows** Rows removed at random, drawn anew for each repeat | 182 4.5% |
| **380 random rows** Rows removed at random, drawn anew for each repeat | 380 9.5% |
| **1,248 random rows** Rows removed at random, drawn anew for each repeat | 1,248 31.2% |
| **1,303 random rows** Rows removed at random, drawn anew for each repeat | 1,303 32.6% |
| **1,449 random rows** Rows removed at random, drawn anew for each repeat | 1,449 36.2% |
| **All steps but step 1, plus 76 random rows** The rows of steps 2, 3, plus random rows drawn anew for each repeat | 380 9.5% |
| **All steps but step 2, plus 120 random rows** The rows of steps 1, 3, plus random rows drawn anew for each repeat | 380 9.5% |
| **All steps but step 3, plus 180 random rows** The rows of steps 1, 2, plus random rows drawn anew for each repeat | 380 9.5% |

The sweeps, [inject](../../reference/Ablation.inject.md#pointblank.Ablation.inject), and [shapley](../../reference/Ablation.shapley.md#pointblank.Ablation.shapley) are strategies too, with their arguments by name (e.g., `- sweep_step: {step: 3, param: value, values: [1500, 2000, 2500, 3000]}` or `- inject: {corruption: "null", step: 2}`). In YAML, a defect must be one of the four kinds by name, not a function.

The block is ignored by [`yaml_interrogate()`](../../reference/yaml_interrogate.md#pointblank.yaml_interrogate), so the same file can be used for validating the data, and [`yaml_to_python()`](../../reference/yaml_to_python.md#pointblank.yaml_to_python) writes out the equivalent Python code, ablation included.

## How the Comparisons Are Made

For each variant, the difference from its control is computed for each repeat, and the report shows the mean of these differences with a paired *t*-interval (95% by default, set with `conf_level=`). The p-values are adjusted for the number of variants with the Holm method, and the verdicts use the adjusted p-values: a variant helps or hurts only if its adjusted p-value is below `1 - conf_level`. You can turn off the adjustment with `adjust=None`. Injected defects remove no rows, so they're always compared with the full table, and the Shapley values have intervals and p-values of their own (from the values of each repeat), adjusted for the number of steps.

With fewer than three repeats, there's too little data for a useful interval, so the report gives no verdicts (and [Ablation](../../reference/Ablation.md#pointblank.Ablation) warns about this). More repeats give narrower intervals, at the cost of more evaluations. A few other options are worth knowing about:

- `control=None` compares variants only to the full table, which saves evaluations but leaves the effect of removing *any* rows mixed in with the effect of removing these rows
- `control=pb.ablation.Stratified("origin")` (see [`ablation.Stratified`](../../reference/ablation.Stratified.md#pointblank.ablation.Stratified)) draws the random rows of each control from each group in proportion to its size, so that a control doesn't take more rows from one group than another by chance (worth using when the groups matter to the model, like the classes of a classification target)
- `max_removed=` (by default `0.5`) sets the share of rows above which a variant gives a warning, since a version of the table missing most of its rows is a different dataset rather than an ablation of this one
- `duplicates=` and `duplicates_order_by=` say which copy of a duplicate row a [`Validate.rows_distinct()`](../../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step should keep (by default the first one, as with [`Validate.get_row_flags()`](../../reference/Validate.get_row_flags.md#pointblank.Validate.get_row_flags)); see [Sundering Validated Data](sundering.md) for more on these

Only steps that check rows can be used in an ablation. Steps that check the table as a whole (like [`Validate.col_exists()`](../../reference/Validate.col_exists.md#pointblank.Validate.col_exists)), inactive steps, and steps using a `pre=` function that isn't marked with [`row_preserving()`](../../reference/row_preserving.md#pointblank.row_preserving) are left out, and the plan and report have a note for each step that couldn't be used.

## Conclusion

A validation can say which rows break its rules, but only an experiment can say whether those rows hurt the model trained on them. An [Ablation](../../reference/Ablation.md#pointblank.Ablation) runs that experiment with controls, repeats, and intervals, so you can find the checks that are worth enforcing, the ones that don't matter, the ones that are redundant, the ones that remove good data, and where to set their limits. Injecting defects shows how much a problem would cost (and gives thresholds to match), and Shapley values share the effect of the whole cleaning fairly among the steps. It works with any model that you can wrap in an `evaluate` function (or train elsewhere, with `export()` and [load_results()](../../reference/Ablation.load_results.md#pointblank.Ablation.load_results)), and with any validation whose steps check rows.
