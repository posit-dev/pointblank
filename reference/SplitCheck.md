# SplitCheck

The results of [`check_splits()`](../reference/check_splits.md#pointblank.check_splits).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/splits.py#L234-L745)

``` python
SplitCheck(
    values,
    schema,
    columns,
    label,
    near,
    threshold,
    ngram,
    num_perm,
    drift_columns,
)
```

## Attributes

`splits`  
The number of rows of each split, by name.

`overlap`  
A dictionary for each pair of splits (an earlier split `first` and a later one `second`), with the number of rows of the later split that are also in the earlier one (`exact`) and, with `near=True`, the number of other rows of the later split that are nearly the same as one in the earlier split (`near`).

`labels`  
The check of the labels (or `None` without `label=`): the counts of the labels in each split, the labels of the first split that each other split doesn't have (`missing`), those of each other split that the first doesn't have (`unseen`), and the PSI of the shares of the labels of each other split, compared with the first.

`drift`  
A dictionary for each column of `drift_columns=` and each split after the first, with the PSI and (for a numeric column) the KS statistic and its p-value, compared with the first split, and the level of the change (`"stable"`, `"moderate"`, `"major"`, or `None` when there are too few values).

`issues`  
The problems found, as messages: overlapping rows, missing or unseen labels, and major changes in distribution (a PSI of at least 0.2). A moderate change isn't an issue, but it's shown in the report.
