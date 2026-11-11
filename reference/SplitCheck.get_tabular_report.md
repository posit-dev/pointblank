# SplitCheck.get_tabular_report()

Get a tabular report of the checks of the splits.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/splits.py#L584-L795)

``` python
SplitCheck.get_tabular_report(
    title=None,
    lang=None,
    locale=None,
)
```

The report has a group of rows for each check: the overlap of each pair of splits, the labels of each split, and the drift of each column in each split after the first. Each row has a badge with its result, and the header says how many issues there are.

## Parameters

`title: str | None = None`  
An optional title replacing the default.

`lang: str | None = None`  
The language of the report's text, as a two-letter language code (e.g., `"fr"` or `"de"`; see the `lang=` argument of [`Validate`](../reference/Validate.md#pointblank.Validate) for the languages). By default, the report is in English. The names of the splits and the columns, and the labels, are shown as they are.

`locale: str | None = None`  
The locale for the numbers in the report (e.g., `"de-CH"`), which sets the decimal mark and the thousands separator. By default, it's `lang=`.

## Returns

`GT`  
A Great Tables object.

## Examples

``` python
import pointblank as pb
import polars as pl

train = pl.DataFrame({"x": list(range(100)), "y": ["a", "b"] * 50})
test = pl.DataFrame({"x": list(range(50, 100)), "y": ["a"] * 50})

pb.check_splits(train=train, test=test, label="y").get_tabular_report()
```

| Split check |  |
|----|----|
| train: 100 rows · test: 50 rows **4 issues found.** |  |
| Check | Result |
| Overlap |  |
| Rows of **test** also in **train** compared on `x` | LEAK 50 of 50 rows (100.0%) 50 the same |
| Labels |  |
| Labels of **train** in `y` | BASE a 50% · b 50% 2 labels |
| Labels of **test** in `y`, compared with train | ISSUE a 100% missing 'b'; PSI 4.6 |
| Drift |  |
| `x` in **test** compared with train | MAJOR PSI 3.8 numeric; KS 0.5 (p = 0) |
| PSI is the population stability index of the shares of the values (binned by the deciles of 'train', for a numeric column): a change is moderate from 0.1 and major from 0.2. KS is the Kolmogorov-Smirnov statistic. |  |

Each row of the test split is also in the training split, the test split has none of the label `"b"`, and its values of `x` are larger.
