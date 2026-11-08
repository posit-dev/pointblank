# Ablation.export()

Write out the versions of the table to evaluate, for training them elsewhere.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L3155-L3318)

``` python
Ablation.export(
    path,
    format="parquet",
    overwrite=False,
)
```

When the model can't be trained in this Python session (e.g., it's trained on a cluster, by another team, or in another language), the ablation can be split in two. This method writes every version of the training table that a run would evaluate to a directory, with a manifest that says which seed to train each one with. Once the evaluations are done, [`load_results()`](../reference/Ablation.load_results.md#pointblank.Ablation.load_results) reads the metrics back and computes the results, just as [`run()`](../reference/Ablation.run.md#pointblank.Ablation.run) would.

## Parameters

`path: str | os.PathLike[str]`  
The directory to write to. It's created if it doesn't exist.

`format: str = "parquet"`  
How the versions of the table are written. With `"parquet"` (the default), each is a Parquet file of the table with its rows removed. With `"indices"`, each is a JSON file with the numbers of the rows that are removed (counting from `0`, in the order of the table), which is much smaller when the table is large and the evaluations can read the full table themselves. For database tables, which have no row order of their own, the rows are numbered in the order of their values, so `"parquet"` is the safer choice.

`overwrite: bool = False`  
Should an earlier export in the same directory be replaced? By default, there's an error if the directory already has a manifest.

## Returns

`Path`  
The directory that was written to.

## Details

The directory has:

- `manifest.json`: the settings of the ablation, the versions of the table, every evaluation to make (the version, the repeat, the seed, and the file of its table), and fingerprints of the data and of the plan, which [load_results()](../reference/Ablation.load_results.md#pointblank.Ablation.load_results) checks
- `tables/`: the versions of the table, one file for each variant (shared by its repeats) and one for each repeat of each control, since a control's random rows change from one repeat to the next
- `results_template.csv`: a table with a row for each evaluation and an empty column for the metric, to be filled in and saved as `results.csv`

Each evaluation trains a model on its table with its seed and scores it on the same fixed holdout data, as the `evaluate=` function of a run would. Add a column to the results for each metric, if there's more than one.

## Examples

Here's an ablation of a table of temperature readings, with no `evaluate=` function since the evaluations are made elsewhere.

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

pl.read_csv(directory / "results_template.csv").head()
```

shape: (5, 4)

| variant | repeat | seed | error |
|---------|--------|------|-------|
| str     | i64    | i64  | str   |
| "full"  | 0      | 0    | null  |
| "full"  | 1      | 1    | null  |
| "full"  | 2      | 2    | null  |
| "full"  | 3      | 3    | null  |
| "full"  | 4      | 4    | null  |

The template has a row for each of the 15 evaluations (5 repeats of the full table, the variant, and its control), each with the seed to train with.
