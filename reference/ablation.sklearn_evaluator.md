# ablation.sklearn_evaluator()

Make an `evaluate=` function that trains a scikit-learn model and scores it on held-out data.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L5993-L6184)

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

Here's a table of the temperatures of a process at different pressures, where some of the temperatures were recorded in the wrong units. The held-out data is clean.

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

ablation = pb.Ablation(
    validation,
    evaluate=pb.ablation.sklearn_evaluator(holdout, target="temp"),
    metric="r2",
).drop_each()

ablation.run()
```

| Ablation report |  |  |  |
|----|----|----|----|
| r2 on the full table (400 rows): **−0.0926 ± 0.0000** (higher is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | r2 | Δ vs random |
| **Step 1** Expect that values in `temp` should be between `15` and `60`. | 40 10.0% | 0.9981 ± 0.0000 Δ vs full +1.0906 | +1.1308 \[+1.0136, +1.2480\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iODcuNCIgeTE9IjYuMCIgeDI9Ijk2LjAiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iOTEuNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |

The model is scored with R² by default, for a regression. Removing the readings in the wrong units improves it, compared with removing as many random readings.
