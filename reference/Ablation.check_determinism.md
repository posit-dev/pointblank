# Ablation.check_determinism()

Check that `evaluate=` gives the same result twice for the same table and seed.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L3290-L3394)

``` python
Ablation.check_determinism(
    seed=None,
    tolerance=0.0,
)
```

The comparisons of an ablation are paired: each version of the table is evaluated with the same seeds, so that the randomness of training (e.g., the initial weights of a model, or the order of its training rows) cancels out in the differences. That only works if the seed controls all of that randomness. This method evaluates the full table twice with the same seed and warns if the results differ, which is worth checking before a long run.

## Parameters

`seed: int | None = None`  
The seed to evaluate with. By default, it's the `seed=` of the ablation.

`tolerance: float = 0.0`  
The largest difference in a metric that still counts as the same. Differences from floating-point rounding (e.g., from a model that sums its trees in parallel, in whichever order they finish) always count as the same: these are relative differences of up to `1e-9`.

## Returns

`bool`  
`True` if both evaluations gave the same metrics (within `tolerance=`), `False` (with a warning that lists the metrics that differ) if they didn't.

## Examples

Here's an `evaluate()` function that forgets to use its seed, so that its model starts from a different random weight every time.

``` python
import random
import warnings
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed):
    weight = random.Random().uniform(0.9, 1.1)
    return abs(train["temp"].mean() * weight - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_between(columns="temp", left=-10, right=45)
    .interrogate()
)

ablation = pb.Ablation(validation, evaluate=evaluate)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    deterministic = ablation.check_determinism()

print(deterministic)
print(caught[0].message)
```

    False
    `evaluate=` isn't deterministic: given the full table and the seed 0 twice, it returned different results (score: 1.7476182306362595 and 0.5255053742129512). Use the seed for every source of randomness in training, or the ablation's comparisons will be noisier than they need to be.

Creating the generator with `random.Random(seed)` would fix it, and the check would then return `True`.

## See Also

[run()](../reference/Ablation.run.md#pointblank.Ablation.run)  
Evaluate the full table, each variant, and each control.
