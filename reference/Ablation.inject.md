# Ablation.inject()

Add a variant for each rate of a defect, put into clean rows of the table.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L1679-L1958)

``` python
Ablation.inject(
    corruption,
    column=None,
    step=None,
    rates=None,
    name=None,
)
```

The other strategies remove the rows that failed a check, to see whether they matter. This one runs the experiment the other way around: it takes rows that pass every check and gives them a defect (e.g., a missing value) at a few rates, to see how much the defect costs the model. The results trace how the model gets worse as the defect gets more common (a dose-response curve), and the report plots the curve. When the defect is one that a step checks for (given as `step=`), this measures what a failing test unit of the step costs, which [`suggest_thresholds()`](../reference/Ablation.suggest_thresholds.md#pointblank.Ablation.suggest_thresholds) turns into thresholds for the step.

## Parameters

`corruption: str | Callable[[Any], Any]`  
The defect to put into the rows:

- `"null"`: replace the value with a missing value
- `"shuffle"`: swap the values of the corrupted rows among themselves (e.g., to shuffle the labels of a classification target)
- `"truncate"`: cut a text value to the first half of its characters
- `"duplicate"`: add a copy of each corrupted row (whole rows, so there's no `column=`)

It can also be a function that takes a value and returns the corrupted value (e.g., `lambda x: x * 10`, for a reading recorded in the wrong unit).

`column: str | None = None`  
The column to corrupt. By default, it's the column that `step=` checks (if it checks a single column).

`step: int | None = None`  
The number of the step that checks for the defect (e.g., a [`col_vals_not_null()`](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) step, for `"null"`). It must be a row-level step that the ablation can use. [suggest_thresholds()](../reference/Ablation.suggest_thresholds.md#pointblank.Ablation.suggest_thresholds) only suggests thresholds for the steps given here.

`rates: list[float] | None = None`  
The shares of the table's rows to corrupt, each greater than `0` and at most `1`. By default, they're `0.01`, `0.05`, and `0.1`.

`name: str | None = None`  
A name for the defect, used in the ids and names of the variants (e.g., `"unit_error"`). By default, it's `corruption=`, or the name of the function.

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the variants added.

## Details

The corrupted rows are drawn at random (with the ablation's `seed=`) from the clean rows: the rows that fail none of the ablation's steps and have a value in `column=` (a text value with at least one character, for `"truncate"`). The draws are nested, so the rows corrupted at one rate are also corrupted at every higher rate, and they're the same in every repeat. A rate that needs more rows than there are clean rows, or that rounds to no rows, is left out, with a note. The copies of `"duplicate"` rows are added at the end of the table. With `"shuffle"`, the values move around in a ring, so no row keeps its own value (though it can get an equal one).

Each variant is compared with the full table rather than with a control, since no rows are removed. A verdict of `"hurts"` means that the defect makes the model clearly worse. The table is corrupted in memory, so this strategy needs a Polars, pandas, or PyArrow table (or a Polars `LazyFrame`). The versions of the table can be exported with [`export()`](../reference/Ablation.export.md#pointblank.Ablation.export), but only as Parquet files.

## Examples

Here's a table of temperature readings with no missing values, and a 'model' that estimates the average temperature, filling in a missing reading with `0`. It's evaluated by its error against the true average of `20`.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed):
    sample = train.fill_null(0).sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_not_null(columns="temp")
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .inject("null", step=1)
    .run()
)

ablation
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **0.092 ± 0.062** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from the full table. |  |  |  |
| Variant | Removed | error | Δ vs full |
| Injected: missing values in temp |  |  |  |
| **Missing values in temp: 1%** 3 clean rows get a missing value in `temp` (checked by step 1) | 3 1.0% corrupted | 0.232 ± 0.127 | +0.140 \[−0.048, +0.327\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNDkuMSIgeTE9IjYuMCIgeDI9IjU2LjMiIHkyPSI2LjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNTIuNyIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzk5OTk5OSI+PC9jaXJjbGU+PC9zdmc+) inconclusive |
| **Missing values in temp: 5%** 15 clean rows get a missing value in `temp` (checked by step 1) | 15 5.0% corrupted | 1.119 ± 0.157 | +1.026 \[+0.792, +1.260\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNjUuNCIgeTE9IjYuMCIgeDI9Ijc0LjUiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNjkuOSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| **Missing values in temp: 10%** 30 clean rows get a missing value in `temp` (checked by step 1) | 30 10.0% corrupted | 2.071 ± 0.331 | +1.978 \[+1.585, +2.371\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iODAuNyIgeTE9IjYuMCIgeDI9Ijk2LjAiIHkyPSI2LjAiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iODguNCIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iI0NGMTQyQiI+PC9jaXJjbGU+PC9zdmc+) hurts |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |
| ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDYwIiBoZWlnaHQ9IjIzMCIgdmlld2JveD0iMCAwIDQ2MCAyMzAiIHN0eWxlPSJkaXNwbGF5OiBibG9jazsgbWFyZ2luLXRvcDogNHB4OyI+PHRleHQgeD0iMCIgeT0iMTMiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsgZm9udC1zaXplOiAxMXB4OyBmaWxsOiAjMzMzMzMzOyI+bWlzc2luZyB2YWx1ZXMgaW4gdGVtcDogzpQgZXJyb3IgdnMgZnVsbCAobG93ZXIgaXMgYmV0dGVyKTwvdGV4dD48bGluZSB4MT0iNTgiIHkxPSI0NCIgeDI9IjU4IiB5Mj0iMTkwIiBzdHJva2U9IiNCQkJCQkIiPjwvbGluZT48bGluZSB4MT0iNTgiIHkxPSIxOTAiIHgyPSI0NDQiIHkyPSIxOTAiIHN0cm9rZT0iI0JCQkJCQiI+PC9saW5lPjxsaW5lIHgxPSI1OCIgeTE9IjE1My41IiB4Mj0iNDQ0IiB5Mj0iMTUzLjUiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PGxpbmUgeDE9IjU4IiB5MT0iMTkwLjAiIHgyPSI0NDQiIHkyPSIxOTAuMCIgc3Ryb2tlPSIjRUVFRUVFIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1MyIgeT0iMTkzLjAiIHRleHQtYW5jaG9yPSJlbmQiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPuKIkjE8L3RleHQ+PHRleHQgeD0iNTMiIHk9IjE1Ni41IiB0ZXh0LWFuY2hvcj0iZW5kIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4rMDwvdGV4dD48bGluZSB4MT0iNTgiIHkxPSIxMTcuMCIgeDI9IjQ0NCIgeTI9IjExNy4wIiBzdHJva2U9IiNFRUVFRUUiIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjUzIiB5PSIxMjAuMCIgdGV4dC1hbmNob3I9ImVuZCIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+KzE8L3RleHQ+PGxpbmUgeDE9IjU4IiB5MT0iODAuNSIgeDI9IjQ0NCIgeTI9IjgwLjUiIHN0cm9rZT0iI0VFRUVFRSIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PHRleHQgeD0iNTMiIHk9IjgzLjUiIHRleHQtYW5jaG9yPSJlbmQiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPisyPC90ZXh0PjxsaW5lIHgxPSI1OCIgeTE9IjQ0LjAiIHgyPSI0NDQiIHkyPSI0NC4wIiBzdHJva2U9IiNFRUVFRUUiIHN0cm9rZS13aWR0aD0iMSI+PC9saW5lPjx0ZXh0IHg9IjUzIiB5PSI0Ny4wIiB0ZXh0LWFuY2hvcj0iZW5kIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4rMzwvdGV4dD48dGV4dCB4PSI3Mi4zIiB5PSIyMDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPjElPC90ZXh0Pjx0ZXh0IHg9IjIzMS4xIiB5PSIyMDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPjUlPC90ZXh0Pjx0ZXh0IHg9IjQyOS43IiB5PSIyMDMiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPjEwJTwvdGV4dD48dGV4dCB4PSIyNTEuMCIgeT0iMjI0IiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij5TaGFyZSBvZiB0aGUgcm93cyBjb3JydXB0ZWQ8L3RleHQ+PHBvbHlsaW5lIHBvaW50cz0iNzIuMywxNDguNCAyMzEuMSwxMTYuMCA0MjkuNyw4MS4zIiBmaWxsPSJub25lIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMS41Ij48L3BvbHlsaW5lPjxsaW5lIHgxPSI3Mi4zIiB5MT0iMTU1LjIiIHgyPSI3Mi4zIiB5Mj0iMTQxLjYiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iNzIuMyIgY3k9IjE0OC40IiByPSIzLjUiIGZpbGw9IiM5OTk5OTkiPjx0aXRsZT5NaXNzaW5nIHZhbHVlcyBpbiB0ZW1wOiAxJTogKzAuMTQwPC90aXRsZT48L2NpcmNsZT48bGluZSB4MT0iMjMxLjEiIHkxPSIxMjQuNiIgeDI9IjIzMS4xIiB5Mj0iMTA3LjUiIHN0cm9rZT0iI0NGMTQyQiIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMjMxLjEiIGN5PSIxMTYuMCIgcj0iMy41IiBmaWxsPSIjQ0YxNDJCIj48dGl0bGU+TWlzc2luZyB2YWx1ZXMgaW4gdGVtcDogNSU6ICsxLjAyNjwvdGl0bGU+PC9jaXJjbGU+PGxpbmUgeDE9IjQyOS43IiB5MT0iOTUuNiIgeDI9IjQyOS43IiB5Mj0iNjYuOSIgc3Ryb2tlPSIjQ0YxNDJCIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSI0MjkuNyIgY3k9IjgxLjMiIHI9IjMuNSIgZmlsbD0iI0NGMTQyQiI+PHRpdGxlPk1pc3NpbmcgdmFsdWVzIGluIHRlbXA6IDEwJTogKzEuOTc4PC90aXRsZT48L2NpcmNsZT48L3N2Zz4=) |  |  |  |

Each missing reading pulls the estimate down, so the error grows with the share of missing readings, by about 0.2 degrees for each percent. That's the cost of a failing test unit of step 1, which [suggest_thresholds()](../reference/Ablation.suggest_thresholds.md#pointblank.Ablation.suggest_thresholds) uses.
