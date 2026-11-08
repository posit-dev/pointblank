# check_synthetic()

Check synthetic data against its seed for fidelity, diversity, and copies.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/synthetic.py#L47-L246)

``` python
check_synthetic(
    synthetic,
    seed,
    *,
    columns=None,
    text=None,
    near=True,
    threshold=0.8,
    ngram=5,
    num_perm=128,
    fidelity_columns=None,
)
```

Synthetic data (made by a generative model, an LLM, or a program that perturbs real rows) is meant to be like the real data that it was made from (its *seed*) without being a copy of it. This checks three ways that it can fail:

- **Fidelity.** The distribution of each column of `fidelity_columns=` in the synthetic data is compared with that in the seed, with the population stability index (PSI, from binned shares of the values) and, for a numeric column, the Kolmogorov-Smirnov (KS) statistic. A PSI of at least 0.1 is a moderate change, and one of at least 0.2 is a major change. A change of at least 10 percentage points in the share of missing values of a column is an issue too.
- **Diversity.** A generator can collapse to a few kinds of rows. The share of the rows that repeat an earlier row (on `columns=`) shouldn't be more than 5 percentage points higher than the seed's, and, for each column of `text=`, the distinct-1 and distinct-2 of the text (the share of the words, and of the pairs of words, that are distinct) should be at least 90% of the seed's. Distinct-n is lower for more text, so it's measured on as many rows of each table (sampled, with a fixed seed, from the larger one).
- **Copies.** A synthetic row that's the same as a seed row (on `columns=`) leaks the real data, which matters when the seed is private. With `text=` and `near=True`, a row whose text is the same or nearly the same as a seed row's (a Jaccard similarity of their word n-grams of at least `threshold=`, found with MinHash) is a copy too, which catches copies that differ by a few words, by case, or by punctuation.

The tables can be of any kind that Pointblank can validate (or the paths of files). The columns that are checked are read into memory, so a table in a database is read in full.

## Parameters

`synthetic: Any`  
The synthetic data.

`seed: Any`  
The real data that the synthetic data was made from (and is compared with).

`columns: str | list[str] | None = None`  
The columns whose values identify a row, for the checks of copies and of repeated rows: a column, or a list of them. By default, they're all the columns of the synthetic data that the seed has too.

`text: str | list[str] | None = None`  
The columns of free text, if any: a column, or a list of them. Their diversity is measured, and with `near=True`, synthetic rows whose text is nearly the same as a seed row's are copies.

`near: bool = True`  
Whether to find synthetic rows whose text (of `text=`) is nearly the same as a seed row's, as well as rows that are the same.

`threshold: float = 0.8`  
The smallest Jaccard similarity of two texts' word n-grams for them to be nearly the same (with `near=True`).

`ngram: int = 5`  
The number of words of the n-grams (with `near=True`).

`num_perm: int = 128`  
The number of MinHash permutations (with `near=True`).

`fidelity_columns: str | list[str] | bool | None = None`  
The columns whose distributions are compared. By default, they're the numeric columns that the tables share and the others with few distinct values in the seed (at most 50, and at most half as many as its rows, which leaves out free text and identifiers). Use `False` to compare none.

## Returns

`SyntheticCheck`  
The results, with a report from [`get_tabular_report()`](../reference/SyntheticCheck.get_tabular_report.md#pointblank.SyntheticCheck.get_tabular_report) and the copied rows from [`get_copies()`](../reference/SyntheticCheck.get_copies.md#pointblank.SyntheticCheck.get_copies).

## Examples

Here's a seed of product reviews, and synthetic reviews that a model made from it. The model copied two of the reviews (one with a different case and punctuation), wrote the rest from a few templates, gave fewer low ratings, and left out the product of many reviews.

``` python
import pointblank as pb
import polars as pl

things = ["kettle", "lamp", "blender", "chair", "fan", "desk", "toaster", "rug"]
moods = ["works well", "broke after a week", "is too loud", "looks great", "was a bargain"]
seed = pl.DataFrame(
    {
        "review": [
            f"My {things[k % 8]} {moods[k % 5]} and arrived on day {k}" for k in range(40)
        ],
        "stars": [1 + k % 5 for k in range(40)],
        "product": [things[k % 8] for k in range(40)],
    }
)
synthetic = pl.DataFrame(
    {
        "review": [
            "My kettle works well and arrived on day 0",
            "MY LAMP BROKE AFTER A WEEK, and arrived on day 1!",
            *[f"The {things[k % 2]} is good" for k in range(38)],
        ],
        "stars": [1, 2, *[4 + k % 2 for k in range(38)]],
        "product": ["kettle", "lamp", *[things[k % 2] if k % 3 else None for k in range(38)]],
    }
)

check = pb.check_synthetic(synthetic, seed, text="review", ngram=3)
check
```

| Synthetic data check |  |
|----|----|
| synthetic: 40 rows · seed: 40 rows **7 issues found.** |  |
| Check | Result |
| Fidelity |  |
| `stars` compared with the seed | MAJOR PSI 2.72 numeric; KS 0.55 (p = 7e-06); missing 0.0% (the seed: 0.0%) |
| `product` compared with the seed | MAJOR PSI 6.38 categorical; missing 32.5% (the seed: 0.0%) |
| Diversity |  |
| Rows that repeat an earlier row compared on `review`, `stars`, `product` | MORE 34 of 40 rows (85.0%) the seed: 0 of 40 rows (0.0%) |
| Distinct-1 of `review` the share of the words that are distinct | LOW 0.105 the seed: 0.168 (62% of it); on 40 rows of each |
| Distinct-2 of `review` the share of the pairs of words that are distinct | LOW 0.152 the seed: 0.298 (51% of it); on 40 rows of each |
| Copies |  |
| Synthetic rows that copy a seed row compared on `review`, `stars`, `product`, and on the text of `review` | COPIES 2 of 40 rows (5.0%) 1 the same, 1 with nearly the same text |
| PSI is the population stability index of the shares of the values (binned by the deciles of the seed, for a numeric column): a change is moderate from 0.1 and major from 0.2. KS is the Kolmogorov-Smirnov statistic. A change of 10 percentage points or more in the share of missing values is an issue. |  |
| A share of repeated rows 5 percentage points or more above the seed's is an issue, as is a distinct-n below 90% of the seed's (measured on as many rows of each table, sampled from the larger one). |  |
| Texts are nearly the same if the Jaccard similarity of their 3-word n-grams is at least 0.8 (found with MinHash, with 128 permutations). |  |

The report shows that the ratings and the products are distributed differently in the synthetic data, that a third of its products are missing, that its reviews repeat each other and use far fewer words, and that two of them copy a review of the seed. The copies can be had as a table:

``` python
check.get_copies()
```

shape: (2, 6)

| seed_row | synthetic_row | copy | review | stars | product |
|----|----|----|----|----|----|
| i64 | i64 | str | str | i64 | str |
| 1 | 1 | "exact" | "My kettle works well and arriv… | 1 | "kettle" |
| 2 | 2 | "near" | "MY LAMP BROKE AFTER A WEEK, an… | 2 | "lamp" |

And `check.issues` lists the problems, so that a pipeline can stop when there are any.
