# Ablation.suggest_thresholds()

Suggest thresholds for the steps, from what their failing test units cost the model.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L4374-L4527)

``` python
Ablation.suggest_thresholds(tolerance)
```

Thresholds are usually picked by feel: is 1% of missing values a warning, or an error? With the variants of [`inject()`](../reference/Ablation.inject.md#pointblank.Ablation.inject), the ablation has measured how much the model gets worse as a defect gets more common. This method turns that into thresholds: for each step that checks for an injected defect (given as `step=` to [inject()](../reference/Ablation.inject.md#pointblank.Ablation.inject)), the error threshold is the share of failing test units at which the model would get worse by `tolerance=`, and the warning threshold is half of that.

## Parameters

`tolerance: float`  
How much worse the model can get before it's an error, in the units of the metric that the verdicts are based on (e.g., `0.01` for an accuracy that can drop by a percentage point).

## Returns

`dict[int, Thresholds]`  
The suggested thresholds, by step number, as [`Thresholds`](../reference/Thresholds.md#pointblank.Thresholds) objects with the `warning=` and `error=` levels set as shares of the test units. They can be given to the steps (e.g., as the `thresholds=` of the validation method).

## Details

For each rate of a defect, the step is checked again on the corrupted table, which gives the share of its test units that the defect makes fail (beyond those that failed already). The cost of a failing test unit is then the slope of the harm to the model (how much worse the metric is than on the full table) against that share, fitted through zero over the rates. The error threshold is `tolerance=` divided by that slope, rounded to two significant digits.

A step gets no thresholds (with a warning saying why) when the defect makes none of its test units fail, when the defect doesn't make the model worse (or when none of the verdicts are `"hurts"`, with at least 3 repeats), or when even failing every test unit would cost less than `tolerance=`. When a step has several injected defects, its thresholds come from the costliest one. The thresholds are only as good as the defects are realistic: a defect that's injected into random clean rows can cost less (or more) than the same defect in the rows where it really happens.

## Examples

Here's a table of temperature readings with no missing values, and a 'model' that estimates the average temperature, filling in a missing reading with `0`. Missing readings are injected at three rates.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed):
    sample = train.fill_null(0).sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_not_null(columns="temp")
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .inject("null", step=1)
    .run()
)

ablation.suggest_thresholds(tolerance=0.5)
```

    {1: Thresholds(warning=0.013, error=0.025, critical=None)}

Each percent of missing readings adds about 0.2 degrees to the error, so the error threshold is the share of missing readings that adds 0.5 degrees: about 2.5%.
