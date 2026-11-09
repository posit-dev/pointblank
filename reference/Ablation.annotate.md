# Ablation.annotate()

Add the results of the ablation to the steps of the validation, as notes.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L4227-L4387)

``` python
Ablation.annotate(validation=None)
```

Each step with a variant of its own (from [`drop_each()`](../reference/Ablation.drop_each.md#pointblank.Ablation.drop_each), or a [`drop_steps()`](../reference/Ablation.drop_steps.md#pointblank.Ablation.drop_steps) variant with just that step) gets a note with the effect of removing its failing rows and the verdict. A step with a variant from [`drop_all_but_each()`](../reference/Ablation.drop_all_but_each.md#pointblank.Ablation.drop_all_but_each) also gets the effect of its rows on top of the other steps, and a step with a Shapley value (from [`shapley()`](../reference/Ablation.shapley.md#pointblank.Ablation.shapley)) gets that value. The notes appear in the validation's report, so that the report shows which checks matter to the model, next to the checks themselves. They're written in the language of that validation (its `lang=`), with the numbers formatted for its `locale=`.

## Parameters

`validation: Validate | None = None`  
The validation to add the notes to. By default, it's the validation of the ablation. Another validation (e.g., of a newer version of the data, with the same steps) can be given, as long as each step used in the ablation is the same kind of check of the same column.

## Returns

`Validate`  
The validation, with the notes added (any notes from an earlier [annotate()](../reference/Ablation.annotate.md#pointblank.Ablation.annotate) are replaced).

## Examples

Here's an ablation of a table of temperature readings with out-of-range and missing values.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
for k in range(7, 300, 40):
    temps[k] = None
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed):
    sample = train.drop_nulls().sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_not_null(columns="temp")
    .col_vals_between(columns="temp", left=-10, right=45, na_pass=True)
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .drop_each()
    .run()
)

ablation.annotate()
```

[TABLE]

The validation report now has a note on each step, with the change in the error from removing its failing rows and the verdict.

## See Also

[get_tabular_report()](../reference/Ablation.get_tabular_report.md#pointblank.Ablation.get_tabular_report)  
Get a tabular report of the ablation's results.
