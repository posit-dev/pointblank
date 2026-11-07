# Ablation.to_json()

Get the settings and results of the ablation as a JSON string.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L2967-L3031)

``` python
Ablation.to_json(indent=2)
```

The structure is the same as that of [`to_dict()`](../reference/Ablation.to_dict.md#pointblank.Ablation.to_dict). Saving it alongside a model's training data makes a record of which checks mattered to the model, and of how that was measured.

## Parameters

`indent: int | None = 2`  
The number of spaces to indent nested levels by. Use `None` for the most compact output, on a single line.

## Returns

`str`  
The settings and results of the ablation as a JSON string.

## Examples

Here's an ablation of a table of temperature readings, with three repeats.

``` python
import json
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

print(json.dumps(json.loads(ablation.to_json())["settings"], indent=2))
```

    {
      "label": null,
      "metric": "error",
      "metrics": [
        "error"
      ],
      "higher_is_better": false,
      "repeats": 3,
      "seed": 0,
      "control": "size_matched",
      "duplicates": "keep_first",
      "duplicates_order_by": null,
      "max_removed": 0.5,
      "conf_level": 0.95,
      "adjust": "holm",
      "evaluate_version": null,
      "n_rows": 300,
      "tbl_name": null
    }

The settings record everything needed to repeat the ablation with the same `evaluate()` function, including the seed.
