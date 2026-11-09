# ablation.sklearn_evaluator()

Make an `evaluate=` function that trains a scikit-learn model and scores it on held-out data.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L6289-L6520)

``` python
ablation.sklearn_evaluator(
    holdout,
    target,
    features=None,
    estimator=None,
    task="auto",
    scoring=None,
)
```

An [`Ablation`](../reference/Ablation.md#pointblank.Ablation) needs a function that trains a model on a version of the training table and scores it. For tabular models, that function is often the same: pick the features, encode them, fit a model, and score it on fixed held-out data. This function makes one, so that an ablation can start with a few lines. It needs scikit-learn, which can be installed with `pip install "pointblank[ml]"`.

## Parameters

`holdout: Any`  
The held-out data that every version of the training table is scored on, with the same columns as the training table (any table that the validation could use). It's never ablated. Its rows with a missing target are left out.

`target: str`  
The column to predict.

`features: list[str] | None = None`  
The columns to predict from. By default, every column except `target=` with a type that can be used: numbers, booleans, strings, and categoricals (there's a warning listing the other columns, which are left out).

`estimator: Any = None`  
The scikit-learn estimator (or pipeline) to train. It's cloned for each evaluation, with each `random_state` parameter (including those of the steps of a pipeline) set to the seed of the evaluation. By default, it's a histogram-based gradient boosting model (`HistGradientBoostingClassifier` or `HistGradientBoostingRegressor`), which handles missing values and treats string and categorical features as categories.

`task: str = "auto"`  
The kind of model: `"classification"`, `"regression"`, or `"auto"` (the default), which takes a target of strings, booleans, or categories (or of integers with at most 10 distinct values) as classes, and any other numeric target as a regression.

`scoring: Any = None`  
How the model is scored: the name of a scikit-learn scorer (e.g., `"roc_auc"` or `"neg_mean_absolute_error"`), a scorer function `scorer(estimator, X, y)`, or a list or dictionary of them to get several metrics. By default, it's `"accuracy"` for classification and `"r2"` for regression.

## Returns

`Callable`  
A function for the `evaluate=` parameter of [Ablation](../reference/Ablation.md#pointblank.Ablation), which returns a dictionary of scores (named by the scorers). It can be pickled, so it works with `run(executor="process")` as long as the estimator can be pickled.

## Details

String and categorical features are encoded as integer codes, with the categories found in each version of the training table (a category that's missing from it, like a missing value, is encoded as a missing value). Numeric and boolean features are used as numbers. Rows of the training table with a missing target are left out of training.

The scorers of scikit-learn are such that higher is better (that's why the error metrics are negated, as in `"neg_mean_absolute_error"`), so the default `higher_is_better=True` of [Ablation](../reference/Ablation.md#pointblank.Ablation) is the right one for them. With several scorers, use `metric=` to say which one the verdicts are based on.

## Examples

Here's a table of the temperatures of a process at different pressures, where one temperature in ten was recorded in degrees Fahrenheit rather than Celsius. The held-out data is clean. The model predicts the temperature from the pressure, and it's scored in two ways: with R² (which the verdicts are based on) and with the mean absolute error.

``` python
import pointblank as pb
import polars as pl

def readings(n, offset):
    pressure = [((k + offset) * 37 % 101) / 10 for k in range(n)]
    return pl.DataFrame(
        {"pressure": pressure, "temp": [20 + 3 * p + (k % 7) / 7 for k, p in enumerate(pressure)]}
    )

train = readings(400, offset=0).with_row_index().with_columns(
    temp=pl.when(pl.col("index") % 10 == 3).then(pl.col("temp") * 9 / 5 + 32).otherwise("temp")
).drop("index")
holdout = readings(200, offset=500)

validation = (
    pb.Validate(data=train)
    .col_vals_between(columns="temp", left=15, right=60)
    .interrogate()
)

evaluate = pb.ablation.sklearn_evaluator(
    holdout,
    target="temp",
    features=["pressure"],
    scoring=["r2", "neg_mean_absolute_error"],
)

ablation = pb.Ablation(validation, evaluate=evaluate, metric="r2").drop_each().run()

ablation
```

| Ablation report |  |  |  |
|----|----|----|----|
| r2 on the full table (400 rows): **−0.0926 ± 0.0000** (higher is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | r2 | Δ vs random |
| **Step 1** Expect that values in `temp` should be between `15` and `60`. | 40 10.0% | 0.9981 ± 0.0000 Δ vs full +1.0906 | +1.1308 \[+1.0136, +1.2480\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iODcuNCIgeTE9IjYuMCIgeDI9Ijk2LjAiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iOTEuNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

The default model, a gradient boosting model, is badly thrown off by the readings in degrees Fahrenheit: its R² on the held-out data is below zero, so it does worse than always predicting the average temperature. Removing those readings brings its R² close to `1`. The report for the other metric shows the same change in degrees:

``` python
ablation.get_tabular_report(metric="neg_mean_absolute_error")
```

| Ablation report |  |  |  |
|----|----|----|----|
| neg_mean_absolute_error on the full table (400 rows): **−6.050 ± 0.000**. Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. Verdicts are based on r2, so none are shown here. |  |  |  |
| Variant | Removed | neg_mean_absolute_error | Δ vs random |
| **Step 1** Expect that values in `temp` should be between `15` and `60`. | 40 10.0% | −0.300 ± 0.000 Δ vs full +5.750 | +5.844 \[+5.580, +6.107\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iOTIuMCIgeTE9IjYuMCIgeDI9Ijk2LjAiIHkyPSI2LjAiIHN0cm9rZT0iIzY2NjY2NiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iOTQuMCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzY2NjY2NiI+PC9jaXJjbGU+PC9zdmc+) |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

The mean absolute error drops from about 6 degrees to about 0.3 (the metric is negated, so that higher is better, as for every scikit-learn scorer). Whether a problem in the data matters depends on the model, though. Here's the same ablation with a robust linear model, which gives less weight to the readings far from the line that it fits:

``` python
from sklearn.linear_model import HuberRegressor

evaluate = pb.ablation.sklearn_evaluator(
    holdout,
    target="temp",
    features=["pressure"],
    estimator=HuberRegressor(),
    scoring=["r2", "neg_mean_absolute_error"],
)

pb.Ablation(validation, evaluate=evaluate, metric="r2").drop_each().run()
```

| Ablation report |  |  |  |
|----|----|----|----|
| r2 on the full table (400 rows): **0.9989 ± 0.0000** (higher is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | r2 | Δ vs random |
| **Step 1** Expect that values in `temp` should be between `15` and `60`. | 40 10.0% | 0.9989 ± 0.0000 Δ vs full +0.0000 | +0.0001 \[+0.0000, +0.0001\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNzUuNiIgeTE9IjYuMCIgeDI9Ijk2LjAiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iODUuOCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

This model is barely affected by the readings in degrees Fahrenheit, with an R² of about 0.999 either way. Removing them still helps, and the verdict says so, since the difference is consistent across the repeats. But the difference is tiny (about 0.00005), and the mean absolute error only drops from about 0.253 degrees to 0.245. A clear verdict doesn't mean that the effect is large, so look at the size of the difference too.

## See Also

[Ablation](../reference/Ablation.md#pointblank.Ablation)  
Measure how the rows that fail validation steps affect a model trained on the data.
