# Ablation.sweep_score()

Add a variant for each cutoff of a score, removing the rows on the wrong side of it.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L1334-L1535)

``` python
Ablation.sweep_score(
    score,
    quantiles=None,
    values=None,
    remove=None,
)
```

A score (e.g., from a quality classifier) doesn't say where to draw the line between the rows to keep and those to remove. This strategy measures it: each variant removes more of the lowest-scoring rows than the last, so the results trace how the model changes as the cutoff gets stricter (a dose-response curve). The report plots the curve, along with what removing as many random rows does. The cutoffs are given as shares of the scored rows to remove (`quantiles=`) or as score values (`values=`).

## Parameters

`score: int | str`  
Where the scores come from: the number of a [`col_vals_scored()`](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) step that kept its scores (`keep_scores=True`, the default), or the name of a numeric column of the table.

`quantiles: list[float] | None = None`  
The shares of the scored rows to remove, each greater than `0` and at most `1` (e.g., `0.1` removes the tenth of the rows with the lowest scores). This is the default, with the shares `0.05`, `0.1`, `0.2`, `0.3`, `0.4`, and `0.5`.

`values: list[float] | None = None`  
Score cutoffs to use instead of `quantiles=`: each variant removes the rows that score below the cutoff (or above it, with `remove="highest"`).

`remove: str | None = None`  
Which rows to remove: `"lowest"` (the lowest-scoring rows) or `"highest"`. By default, it's `"highest"` for a step with only `max_val=` (e.g., a toxicity or perplexity score, where a high score is bad) and `"lowest"` otherwise.

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the variants added.

## Details

The rows without a score (a missing value, a row the scorer gave no score, or a row outside a segmented step's segment) are never removed, and a note says how many there are. With `quantiles=`, the rows are ranked by their score, and ties are broken by row order, so each share removes a set number of rows. Each point of the sweep is a variant like any other: it's compared with removing the same number of random rows and gets a verdict. A point that removes no rows is left out, and a point that removes the same rows as another version of the table isn't evaluated again.

## Examples

Here's a table of temperature readings, some of them recorded in tenths of a degree, and a scorer that rates how plausible each reading is (from `0` to `1`). The 'model' estimates the average temperature and is evaluated by its error against the true average of `20`.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
readings = pl.DataFrame({"temp": temps})

def plausibility(rows):
    return [1 / (1 + abs(row["temp"] - 20) / 5) for row in rows]

def evaluate(train, *, seed):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_scored(columns="temp", scorer=plausibility, min_val=0.5)
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .sweep_score(1, quantiles=[0.02, 0.04, 0.1, 0.2])
    .run()
)

ablation
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **7.392 ± 2.207** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| Sweep: Step 1, removing the lowest-scoring rows |  |  |  |
| **Step 1: lowest 2% of scores** The 6 rows with the lowest scores (of 300 scored rows) | 6 2.0% | 3.449 ± 1.759 Δ vs full −3.943 | −3.706 \[−4.900, −2.512\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMjguOCIgeTE9IjYuMCIgeDI9IjM5LjEiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMzQuMCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Step 1: lowest 4% of scores** The 12 rows with the lowest scores (of 300 scored rows) | 12 4.0% | 0.109 ± 0.080 Δ vs full −7.283 | −6.901 \[−8.634, −5.168\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTIuNyIgeTE9IjYuMCIgeDI9IjI3LjciIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjAuMiIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| **Step 1: lowest 10% of scores** The 30 rows with the lowest scores (of 300 scored rows) | 30 10.0% | 0.113 ± 0.104 Δ vs full −7.279 | −8.524 \[−10.645, −6.403\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjIuMyIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMy4yIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| **Step 1: lowest 20% of scores** The 60 rows with the lowest scores (of 300 scored rows) | 60 20.0% | 0.111 ± 0.079 Δ vs full −7.281 | −7.245 \[−9.099, −5.392\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMTAuNyIgeTE9IjYuMCIgeDI9IjI2LjciIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMTguNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |
| ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDYwIiBoZWlnaHQ9IjIzMCIgdmlld2JveD0iMCAwIDQ2MCAyMzAiIHN0eWxlPSJkaXNwbGF5OiBibG9jazsgbWFyZ2luLXRvcDogNHB4OyI+PHRleHQgeD0iMCIgeT0iMTMiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsgZm9udC1zaXplOiAxMXB4OyBmaWxsOiAjMzMzMzMzOyI+U3RlcCAxLCByZW1vdmluZyB0aGUgbG93ZXN0LXNjb3Jpbmcgcm93czogzpQgZXJyb3IgdnMgZnVsbCAobG93ZXIgaXMgYmV0dGVyKTwvdGV4dD48bGluZSB4MT0iNTgiIHkxPSI0NCIgeDI9IjU4IiB5Mj0iMTkwIiBzdHJva2U9IiNCQkJCQkIiPjwvbGluZT48bGluZSB4MT0iNTgiIHkxPSIxOTAiIHgyPSI0NDQiIHkyPSIxOTAiIHN0cm9rZT0iI0JCQkJCQiI+PC9saW5lPjxsaW5lIHgxPSI1OCIgeTE9IjgwLjUiIHgyPSI0NDQiIHkyPSI4MC41IiBzdHJva2U9IiM5OTk5OTkiIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjxsaW5lIHgxPSI1OCIgeTE9IjE5MC4wIiB4Mj0iNDQ0IiB5Mj0iMTkwLjAiIHN0cm9rZT0iI0VFRUVFRSIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PHRleHQgeD0iNTMiIHk9IjE5My4wIiB0ZXh0LWFuY2hvcj0iZW5kIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij7iiJIxNTwvdGV4dD48bGluZSB4MT0iNTgiIHkxPSIxNTMuNSIgeDI9IjQ0NCIgeTI9IjE1My41IiBzdHJva2U9IiNFRUVFRUUiIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjUzIiB5PSIxNTYuNSIgdGV4dC1hbmNob3I9ImVuZCIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+4oiSMTA8L3RleHQ+PGxpbmUgeDE9IjU4IiB5MT0iMTE3LjAiIHgyPSI0NDQiIHkyPSIxMTcuMCIgc3Ryb2tlPSIjRUVFRUVFIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1MyIgeT0iMTIwLjAiIHRleHQtYW5jaG9yPSJlbmQiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPuKIkjU8L3RleHQ+PHRleHQgeD0iNTMiIHk9IjgzLjUiIHRleHQtYW5jaG9yPSJlbmQiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPiswPC90ZXh0PjxsaW5lIHgxPSI1OCIgeTE9IjQ0LjAiIHgyPSI0NDQiIHkyPSI0NC4wIiBzdHJva2U9IiNFRUVFRUUiIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjUzIiB5PSI0Ny4wIiB0ZXh0LWFuY2hvcj0iZW5kIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4rNTwvdGV4dD48dGV4dCB4PSI3Mi4zIiB5PSIyMDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPjIlPC90ZXh0Pjx0ZXh0IHg9IjExMi4wIiB5PSIyMDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPjQlPC90ZXh0Pjx0ZXh0IHg9IjIzMS4xIiB5PSIyMDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPjEwJTwvdGV4dD48dGV4dCB4PSI0MjkuNyIgeT0iMjAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4yMCU8L3RleHQ+PHRleHQgeD0iMjUxLjAiIHk9IjIyNCIgdGV4dC1hbmNob3I9Im1pZGRsZSIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+U2hhcmUgb2YgdGhlIHNjb3JlZCByb3dzIHJlbW92ZWQ8L3RleHQ+PHBvbHlsaW5lIHBvaW50cz0iNzIuMyw4Mi4yIDExMi4wLDgzLjMgMjMxLjEsNzEuNCA0MjkuNyw4MC44IiBmaWxsPSJub25lIiBzdHJva2U9IiM4ODg4ODgiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtZGFzaGFycmF5PSI0LDMiPjwvcG9seWxpbmU+PHBvbHlsaW5lIHBvaW50cz0iNzIuMywxMDkuMyAxMTIuMCwxMzMuNyAyMzEuMSwxMzMuNiA0MjkuNywxMzMuNyIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjEuNSI+PC9wb2x5bGluZT48bGluZSB4MT0iNzIuMyIgeTE9IjEzOS43IiB4Mj0iNzIuMyIgeTI9Ijc4LjkiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNzIuMyIgY3k9IjEwOS4zIiByPSIzLjUiIGZpbGw9IiM0Q0E2NEMiPjx0aXRsZT5TdGVwIDE6IGxvd2VzdCAyJSBvZiBzY29yZXM6IOKIkjMuOTQzPC90aXRsZT48L2NpcmNsZT48bGluZSB4MT0iMTEyLjAiIHkxPSIxNTMuNCIgeDI9IjExMi4wIiB5Mj0iMTE0LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMTEyLjAiIGN5PSIxMzMuNyIgcj0iMy41IiBmaWxsPSIjNENBNjRDIj48dGl0bGU+U3RlcCAxOiBsb3dlc3QgNCUgb2Ygc2NvcmVzOiDiiJI3LjI4MzwvdGl0bGU+PC9jaXJjbGU+PGxpbmUgeDE9IjIzMS4xIiB5MT0iMTUzLjUiIHgyPSIyMzEuMSIgeTI9IjExMy44IiBzdHJva2U9IiM0Q0E2NEMiIHN0cm9rZS13aWR0aD0iMiI+PC9saW5lPjxjaXJjbGUgY3g9IjIzMS4xIiBjeT0iMTMzLjYiIHI9IjMuNSIgZmlsbD0iIzRDQTY0QyI+PHRpdGxlPlN0ZXAgMTogbG93ZXN0IDEwJSBvZiBzY29yZXM6IOKIkjcuMjc5PC90aXRsZT48L2NpcmNsZT48bGluZSB4MT0iNDI5LjciIHkxPSIxNTMuNiIgeDI9IjQyOS43IiB5Mj0iMTEzLjciIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNDI5LjciIGN5PSIxMzMuNyIgcj0iMy41IiBmaWxsPSIjNENBNjRDIj48dGl0bGU+U3RlcCAxOiBsb3dlc3QgMjAlIG9mIHNjb3Jlczog4oiSNy4yODE8L3RpdGxlPjwvY2lyY2xlPjxsaW5lIHgxPSIwIiB5MT0iMjYiIHgyPSIxOCIgeTI9IjI2IiBzdHJva2U9IiM4ODg4ODgiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtZGFzaGFycmF5PSI0LDMiPjwvbGluZT48dGV4dCB4PSIyMiIgeT0iMjkiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPlJlbW92aW5nIGFzIG1hbnkgcmFuZG9tIHJvd3M8L3RleHQ+PC9zdmc+) |  |  |  |

Removing the lowest-scoring 4% of the rows takes out every bad reading, and the curve below the report shows the error dropping to near zero there. Removing more rows than that makes no further difference.
