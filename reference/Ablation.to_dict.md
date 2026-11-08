# Ablation.to_dict()

Get the settings and results of the ablation as a JSON-serializable dict.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L4357-L4437)

``` python
Ablation.to_dict()
```

## Returns

`dict[str, Any]`  
A dictionary with the ablation's settings (`"settings"`), fingerprints of the data and of the plan (`"fingerprints"`, as in the manifest of [`export()`](../reference/Ablation.export.md#pointblank.Ablation.export)), the versions of the table that were evaluated (`"variants"`, including the full table and the controls), the variants that weren't evaluated because they remove the same rows as another version (`"same_as"`, from each to that version), the metrics from every evaluation (`"runs"`, one entry per version and repeat, with its seed and whether its result came from the cache), the results (`"effects"`, with the same fields as the columns of [`get_effects()`](../reference/Ablation.get_effects.md#pointblank.Ablation.get_effects)), the Shapley values (`"shapley"`, with the same fields as the columns of [`get_shapley()`](../reference/Ablation.get_shapley.md#pointblank.Ablation.get_shapley), or `None` without [`shapley()`](../reference/Ablation.shapley.md#pointblank.Ablation.shapley)), and the notes about variants and steps that weren't used (`"notes"`).

## Examples

Here's an ablation of a table of temperature readings, with three repeats.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed, variant):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_between(columns="temp", left=-10, right=45)
    .interrogate()
)

ablation = (
    pb.Ablation(
        validation, evaluate=evaluate, metric="error", higher_is_better=False, repeats=3
    )
    .drop_each()
    .run()
)

ablation.to_dict()["runs"][:4]
```

    [{'variant': 'full',
      'repeat': 0,
      'seed': 0,
      'metrics': {'error': 6.587999999999997},
      'cached': False},
     {'variant': 'full',
      'repeat': 1,
      'seed': 1,
      'metrics': {'error': 9.003999999999998},
      'cached': False},
     {'variant': 'full',
      'repeat': 2,
      'seed': 2,
      'metrics': {'error': 3.913999999999998},
      'cached': False},
     {'variant': 'step_1',
      'repeat': 0,
      'seed': 0,
      'metrics': {'error': 0.06875000000000142},
      'cached': False}]

Each run records the version of the table, the repeat, the seed, the metrics, and whether the result came from the cache, so the runs can be stored and analyzed again later.
