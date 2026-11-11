# SyntheticCheck

The results of [`check_synthetic()`](../reference/check_synthetic.md#pointblank.check_synthetic).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/synthetic.py#L249-L797)

``` python
SyntheticCheck(
    values,
    numeric,
    columns,
    text,
    near,
    threshold,
    ngram,
    num_perm,
    fidelity_columns,
)
```

## Attributes

`rows`  
The number of rows of the synthetic data and of the seed.

`fidelity`  
A dictionary for each column of `fidelity_columns=`, with the PSI and (for a numeric column) the KS statistic and its p-value of the synthetic data, compared with the seed, the level of the change (`"stable"`, `"moderate"`, `"major"`, or `None` when there are too few values), and the shares of missing values in each table.

`duplicates`  
The number and share of the rows of each table that repeat an earlier row of the same table (on `columns=`).

`diversity`  
A dictionary for each column of `text=` and each length of n-gram (1 and 2), with the distinct-n of each table (on as many rows of each), and that of the synthetic data as a share of the seed's (`ratio`).

`copies`  
The number of synthetic rows that are the same as a seed row (`exact`) and, with `text=` and `near=True`, the number of other synthetic rows whose text is the same or nearly the same as a seed row's (`near`), with their share of the synthetic rows.

`issues`  
The problems found, as messages: major changes in distribution (a PSI of at least 0.2), changes of at least 10 percentage points in the share of missing values, more repeated rows, text that's less diverse, and copies. A moderate change isn't an issue, but it's shown in the report.

## Examples

Here's a seed of four people, and four synthetic people made from it. The second synthetic person is the same as the second real one.

``` python
import pointblank as pb
import polars as pl

seed = pl.DataFrame({"name": ["Ana", "Ben", "Cleo", "Dev"], "age": [34, 51, 27, 45]})
synthetic = pl.DataFrame({"name": ["Dan", "Ben", "Eve", "Fay"], "age": [40, 51, 29, 44]})

check = pb.check_synthetic(synthetic, seed)
check.copies
```

    {'exact': 1, 'near': None, 'share': 0.25}

One synthetic row (a quarter of them) is a copy. The ages are compared too. Four rows are too few for a PSI, so the level of the change is `None`, but there's still a KS statistic:

``` python
check.fidelity
```

    [{'column': 'age',
      'kind': 'numeric',
      'psi': None,
      'ks': 0.25,
      'ks_p_value': 1.0,
      'level': None,
      'missing_seed': 0.0,
      'missing_synthetic': 0.0}]

Every problem is in [issues](../reference/ConformanceReport.md#pointblank.ConformanceReport.issues), as a message:

``` python
check.issues
```

    ['1 row of the synthetic data (25.0%) copies a seed row (1 the same).']

The `ok` property is `True` only when there are no issues, so a pipeline can stop when it's `False`.
