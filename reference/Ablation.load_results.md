# Ablation.load_results()

Read the results of evaluations made elsewhere, and compute the effects.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L2078-L2260)

``` python
Ablation.load_results(
    results,
    manifest=None,
)
```

This is the second half of an ablation split with [`export()`](../reference/Ablation.export.md#pointblank.Ablation.export): once each version of the table has been trained and scored, this method reads the metrics and computes the results, as [`run()`](../reference/Ablation.run.md#pointblank.Ablation.run) would. The ablation must have the same variants and settings as the one that was exported (e.g., it's made by the same code).

## Parameters

`results: Any`  
The metrics of the evaluations: the path of a CSV file, the path of a directory with a `results.csv` file (e.g., the directory written by `export()`), or a table (e.g., a Polars or pandas DataFrame). It needs a `variant` column (the id of the version of the table), a `seed` column, and a column for each metric. A `repeat` column is checked if it's there, and any other column is a metric.

`manifest: str | os.PathLike[str] | None = None`  
The path of the `manifest.json` file written by `export()` (or of its directory). By default, a manifest in the same directory as the results file is used if there is one. Its fingerprints are checked against the ablation, which makes sure that the results are for the same data and the same versions of the table.

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the results.

## Details

There must be exactly one row for each evaluation in the plan, with a value for every metric. The results are checked in the same way as those from `evaluate=`: when there's more than one metric, `metric=` must name the one that verdicts are based on.

## Examples

Here's an ablation of a table of temperature readings, exported for training elsewhere.

``` python
import tempfile
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
readings = pl.DataFrame({"temp": temps})

validation = (
    pb.Validate(data=readings)
    .col_vals_between(columns="temp", left=-10, right=45)
    .interrogate()
)

ablation = pb.Ablation(validation, metric="error", higher_is_better=False).drop_each()
directory = ablation.export(tempfile.mkdtemp())
```

Here, the 'training' happens in the same session, reading each table from the manifest and filling in the template, but it could be done anywhere.

``` python
import json

manifest = json.loads((directory / "manifest.json").read_text())

def train_and_score(table, seed):
    sample = table.sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

results = pl.DataFrame(
    [
        {
            "variant": run["variant"],
            "seed": run["seed"],
            "error": train_and_score(pl.read_parquet(directory / run["file"]), run["seed"]),
        }
        for run in manifest["runs"]
    ]
)
results.write_csv(directory / "results.csv")

ablation.load_results(directory)
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **7.392 ± 2.207** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| **Step 1** Expect that values in `temp` should be between `-10` and `45`. | 12 4.0% | 0.109 ± 0.080 Δ vs full −7.283 | −6.901 \[−8.634, −5.168\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjIuNSIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMy4yIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |
| The results were read with `load_results()`, from evaluations made elsewhere. |  |  |  |

The results were found in the directory, next to the manifest, and the report is the same as a run with the same evaluations would give.
