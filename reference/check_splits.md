# check_splits()

Check the splits of a dataset for leaks, label coverage, and drift.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/splits.py#L34-L231)

``` python
check_splits(
    train=None,
    valid=None,
    test=None,
    *,
    splits=None,
    columns=None,
    label=None,
    near=False,
    threshold=0.8,
    ngram=5,
    num_perm=128,
    drift_columns=None,
)
```

A model is trained on one split of a dataset and judged on another, so the judgment is only fair if the splits are independent and alike. This checks three ways that they can fail to be:

- **Overlap.** A row of a later split (e.g., of the test split) that's also in an earlier one (e.g., in the training split) leaks: the model is judged on data it has seen, so it looks better than it is. Rows overlap if they have the same values of `columns=`; with `near=True`, rows whose text is nearly the same (a Jaccard similarity of their word n-grams of at least `threshold=`, found with MinHash) overlap too, which catches copies that differ by a few words, by case, or by punctuation.
- **Labels.** With `label=`, every label of the first split should be in the others (or the model is never judged on it), no split should have a label that isn't in the first (the model never learns it), and the shares of the labels should be about the same.
- **Drift.** The distribution of each column of `drift_columns=` in each split is compared with that in the first split, with the population stability index (PSI, from binned shares of the values) and, for a numeric column, the Kolmogorov-Smirnov (KS) statistic. A PSI of at least 0.1 is a moderate change, and one of at least 0.2 is a major change.

The splits can be tables of any kind that Pointblank can validate (or the paths of files). The columns that are checked are read into memory, so a split in a database is read in full.

## Parameters

`train: Any = None`  
The splits, in that order (give at least two). Or use `splits=` instead.

`valid: Any = None`  
The splits, in that order (give at least two). Or use `splits=` instead.

`test: Any = None`  
The splits, in that order (give at least two). Or use `splits=` instead.

`splits: Mapping[str, Any] | None = None`  
A dictionary of the splits, by name, in order (the first one is the one that the others are compared with).

`columns: str | list[str] | None = None`  
The columns whose values identify a row, for the check of overlap: a column, or a list of them. By default, they're all the columns that the splits share, except for `label=`.

`label: str | None = None`  
The column with the labels, if any. A numeric label with more than 50 distinct values (e.g., the target of a regression) is checked for drift only.

`near: bool = False`  
Whether to find rows whose text is nearly the same, as well as rows that are the same.

`threshold: float = 0.8`  
The smallest Jaccard similarity of two rows' word n-grams for them to be nearly the same (with `near=True`).

`ngram: int = 5`  
The number of words of the n-grams (with `near=True`).

`num_perm: int = 128`  
The number of MinHash permutations (with `near=True`).

`drift_columns: str | list[str] | bool | None = None`  
The columns to check for drift. By default, they're the numeric columns that the splits share and the others with few distinct values in the first split (at most 50, and at most half as many as its rows, which leaves out free text and identifiers), except for `label=`. Use `False` to check none.

## Returns

`SplitCheck`  
The results, with a report from [`get_tabular_report()`](../reference/SplitCheck.get_tabular_report.md#pointblank.SplitCheck.get_tabular_report) and the overlapping rows from [`get_overlap()`](../reference/SplitCheck.get_overlap.md#pointblank.SplitCheck.get_overlap).

## Examples

Here are the training and test splits of a dataset of support tickets, labelled with their topic. Two of the test tickets are copies of training tickets: one exactly, and one with different case and punctuation.

``` python
import pointblank as pb
import polars as pl

topics = ["billing", "login", "shipping", "billing", "returns"]
train = pl.DataFrame(
    {
        "text": [f"Ticket {k}: question number {k} about {topics[k % 5]}" for k in range(40)],
        "topic": [topics[k % 5] for k in range(40)],
        "length": [20 + k % 7 for k in range(40)],
    }
)
test = pl.DataFrame(
    {
        "text": [
            "Ticket 3: question number 3 about billing",
            "ticket 7 -- question NUMBER 7 about shipping!",
        ]
        + [f"New ticket {k} about a lost parcel" for k in range(10)],
        "topic": ["billing", "shipping"] + ["shipping"] * 10,
        "length": [20, 22] + [40 + k for k in range(10)],
    }
)

check = pb.check_splits(
    train=train, test=test, columns="text", label="topic", near=True, ngram=3
)
check
```

| Split check |  |
|----|----|
| train: 40 rows · test: 12 rows **4 issues found.** |  |
| Check | Result |
| Overlap |  |
| Rows of **test** also in **train** compared on `text` | LEAK 2 of 12 rows (16.7%) 1 the same, 1 nearly the same |
| Labels |  |
| Labels of **train** in `topic` | BASE billing 40% · login 20% · returns 20% · shipping 20% 4 labels |
| Labels of **test** in `topic`, compared with train | ISSUE shipping 92% · billing 8% missing 'login', 'returns'; PSI 4.63 |
| Drift |  |
| `length` in **test** compared with train | MAJOR PSI 5.6 numeric; KS 0.833 (p = 0) |
| Rows are nearly the same if the Jaccard similarity of their 3-word n-grams is at least 0.8 (found with MinHash, with 128 permutations). |  |
| PSI is the population stability index of the shares of the values (binned by the deciles of 'train', for a numeric column): a change is moderate from 0.1 and major from 0.2. KS is the Kolmogorov-Smirnov statistic. |  |

The report shows both copies, and that the test split has none of the tickets about logins or returns, so the model would never be judged on those. The tickets are also much longer in the test split. The overlapping rows can be had as a table:

``` python
check.get_overlap("train", "test")
```

shape: (2, 4)

| train_row | test_row | overlap | text                             |
|-----------|----------|---------|----------------------------------|
| i64       | i64      | str     | str                              |
| 4         | 1        | "exact" | "Ticket 3: question number 3 ab… |
| 8         | 2        | "near"  | "ticket 7 -- question NUMBER 7 … |

And `check.issues` lists the problems, so that a pipeline can stop when there are any.
