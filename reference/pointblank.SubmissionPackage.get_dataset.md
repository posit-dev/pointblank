# get_dataset()

Get one of the package's datasets by name.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/metadata/_submission.py#L447-L501)

``` python
get_dataset(name)
```

The dataset is returned as it was given to the package, so a Polars DataFrame stays a Polars DataFrame. Indexing the package (as in `study["DM"]`) also gets a dataset, but [get_dataset()](../reference/pointblank.SubmissionPackage.get_dataset.md) gives a more helpful error for a name that isn't in the package.

## Parameters

`name: str`  
The name of the dataset, such as `"DM"`. This is case-insensitive.

## Returns

`Any`  
The dataset, as a DataFrame.

## Raises

`KeyError`  
If the package has no dataset with that name. The error message lists the names that the package has.

## Examples

Let's make a package with two datasets and get one of them:

``` python
import pointblank as pb
import polars as pl

dm = pl.DataFrame({"USUBJID": ["S-001", "S-002"], "SEX": ["M", "F"]})
ae = pl.DataFrame({"USUBJID": ["S-001"], "AESEQ": [1], "AETERM": ["HEADACHE"]})

study = pb.SubmissionPackage(datasets={"DM": dm, "AE": ae})

study.get_dataset("ae")
```

shape: (1, 3)

| USUBJID | AESEQ | AETERM     |
|---------|-------|------------|
| str     | i64   | str        |
| "S-001" | 1     | "HEADACHE" |

The name is case-insensitive, so `"ae"` gets the `AE` dataset. Asking for a dataset that isn't in the package raises a `KeyError`:

``` python
try:
    study.get_dataset("LB")
except KeyError as e:
    print(e)
```

    "No dataset named 'LB'. Available: ['AE', 'DM']"

The message lists the datasets that the package has.
