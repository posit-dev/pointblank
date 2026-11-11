# SyntheticCheck.to_dict()

Get the results as a dictionary (that can be written as JSON).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/synthetic.py#L519-L561)

``` python
SyntheticCheck.to_dict()
```

## Returns

`dict`  
The sizes of the tables, the settings, and the results of the checks of fidelity, diversity, and copies, with the issues found.

## Examples

``` python
import json

import pointblank as pb
import polars as pl

seed = pl.DataFrame({"name": ["Ana", "Ben", "Cleo"], "age": [34, 51, 27]})
synthetic = pl.DataFrame({"name": ["Dan", "Ben", "Eve"], "age": [40, 51, 27]})

results = pb.check_synthetic(synthetic, seed).to_dict()
print(json.dumps({"copies": results["copies"], "issues": results["issues"]}, indent=2))
```

    {
      "copies": {
        "exact": 1,
        "near": null,
        "share": 0.3333333333333333
      },
      "issues": [
        "1 row of the synthetic data (33.3%) copies a seed row (1 the same)."
      ]
    }

The dictionary holds only plain values, so it can be written as JSON and kept with the synthetic data, or compared with the results for another batch of it.
