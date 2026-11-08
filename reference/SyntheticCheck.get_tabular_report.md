# SyntheticCheck.get_tabular_report()

Get a tabular report of the checks of the synthetic data.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/synthetic.py#L513-L689)

``` python
SyntheticCheck.get_tabular_report(title=None)
```

The report has a group of rows for each check: the fidelity of each column, the diversity of the rows and of the text, and the copies of seed rows. Each row has a badge with its result, and the header says how many issues there are.

## Parameters

`title: str | None = None`  
An optional title replacing the default.

## Returns

`GT`  
A Great Tables object.

## Examples

``` python
import pointblank as pb
import polars as pl

seed = pl.DataFrame({"x": list(range(100)), "y": ["a", "b"] * 50})
synthetic = pl.DataFrame({"x": [k % 10 for k in range(100)], "y": ["a", "b"] * 50})

pb.check_synthetic(synthetic, seed).get_tabular_report()
```

| Synthetic data check |  |
|----|----|
| synthetic: 100 rows · seed: 100 rows **3 issues found.** |  |
| Check | Result |
| Fidelity |  |
| `x` compared with the seed | MAJOR PSI 8.28 numeric; KS 0.9 (p = 0); missing 0.0% (the seed: 0.0%) |
| `y` compared with the seed | STABLE PSI 0 categorical; missing 0.0% (the seed: 0.0%) |
| Diversity |  |
| Rows that repeat an earlier row compared on `x`, `y` | MORE 90 of 100 rows (90.0%) the seed: 0 of 100 rows (0.0%) |
| Copies |  |
| Synthetic rows that copy a seed row compared on `x`, `y` | COPIES 100 of 100 rows (100.0%) 100 the same |
| PSI is the population stability index of the shares of the values (binned by the deciles of the seed, for a numeric column): a change is moderate from 0.1 and major from 0.2. KS is the Kolmogorov-Smirnov statistic. A change of 10 percentage points or more in the share of missing values is an issue. |  |
| A share of repeated rows 5 percentage points or more above the seed's is an issue. |  |

The synthetic values of `x` are all small, so the synthetic rows repeat each other, and every one of them is a copy of a seed row.
