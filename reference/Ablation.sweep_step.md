# Ablation.sweep_step()

Add a variant for each value of a step's parameter, removing the rows that fail it.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/ablation.py#L1711-L1861)

``` python
Ablation.sweep_step(
    step,
    param,
    values,
)
```

A check's bounds are often picked by hand: is `min_val=0.5` the right cutoff, or would `0.3` keep more useful rows? This strategy measures it. For each value, the step is run again on the table with `param=` set to that value, and the variant removes the rows that fail it. The results trace how the model changes as the check gets stricter, and the report plots the curve, along with what removing as many random rows does.

## Parameters

`step: int`  
The number of the step to vary. It must be a row-level step that the ablation can use (active, and without a `pre=` function unless that function is marked with [`row_preserving()`](../reference/row_preserving.md#pointblank.row_preserving)).

`param: str`  
The name of the parameter of the step's validation method to vary (e.g., `"min_val"` for [`col_vals_scored()`](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored), or `"right"` for [`col_vals_between()`](../reference/Validate.col_vals_between.md#pointblank.Validate.col_vals_between)). It can't be one of the parameters that don't change which rows fail (`thresholds=`, `actions=`, `brief=`, `active=`, and `dimension=`), nor `pre=` or `segments=`.

`values: list[float]`  
The values of `param=` to use, as numbers.

## Returns

`Ablation`  
The [Ablation](../reference/Ablation.md#pointblank.Ablation) object, with the variants added.

## Details

The step is run again as it was set up, with only `param=` changed, so a [col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored) step takes its scores from the score cache instead of scoring the rows again. For a segmented step, only the step's own segment is checked. A value that the validation method doesn't accept (e.g., a `min_val=` above the step's `max_val=`) gives an error. As with any other variant, a value whose step fails no rows is left out, and a value that removes the same rows as another version of the table isn't evaluated again.

## Examples

Here's a table of temperature readings, some of them recorded in tenths of a degree, with a check that the readings are below an upper limit. The 'model' estimates the average temperature and is evaluated by its error against the true average of `20`.

``` python
import pointblank as pb
import polars as pl

temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
for k in range(0, 300, 25):
    temps[k] *= 10
readings = pl.DataFrame({"temp": temps})

def evaluate(train, *, seed):
    sample = train.sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)

validation = (
    pb.Validate(data=readings)
    .col_vals_lt(columns="temp", value=100)
    .interrogate()
)

ablation = (
    pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
    .sweep_step(1, param="value", values=[21, 50, 190, 220])
    .run()
)

ablation
```

| Ablation report |  |  |  |
|----|----|----|----|
| error on the full table (300 rows): **7.392 ± 2.207** (lower is better). Each version was evaluated 5 times; Δ is the mean paired difference from removing the same number of random rows. |  |  |  |
| Variant | Removed | error | Δ vs random |
| Sweep: Step 1, varying value |  |  |  |
| **Step 1 with value=21** The rows that fail step 1 with `value=21` | 117 39.0% | 1.191 ± 0.065 Δ vs full −6.201 | −5.803 \[−8.436, −3.169\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNS4xIiB5MT0iNi4wIiB4Mj0iMzMuMSIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxOS4xIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| **Step 1 with value=50** The rows that fail step 1 with `value=50` | 12 4.0% | 0.109 ± 0.080 Δ vs full −7.283 | −6.901 \[−8.634, −5.168\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iNC4wIiB5MT0iNi4wIiB4Mj0iMjIuNSIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIxMy4yIiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| **Step 1 with value=190** The rows that fail step 1 with `value=190` | 7 2.3% | 2.493 ± 1.666 Δ vs full −4.899 | −4.978 \[−7.564, −2.392\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iOS43IiB5MT0iNi4wIiB4Mj0iMzcuMyIgeTI9IjYuMCIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSIyMy41IiBjeT0iNi4wIiByPSIzIiBmaWxsPSIjNENBNjRDIj48L2NpcmNsZT48L3N2Zz4=) helps |
| **Step 1 with value=220** The rows that fail step 1 with `value=220` | 2 0.7% | 5.764 ± 1.863 Δ vs full −1.628 | −2.045 \[−2.706, −1.385\] ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAwIiBoZWlnaHQ9IjEyIiB2aWV3Ym94PSIwIDAgMTAwIDEyIiBzdHlsZT0idmVydGljYWwtYWxpZ246IG1pZGRsZTsiPjxsaW5lIHgxPSI1MC4wIiB5MT0iMCIgeDI9IjUwLjAiIHkyPSIxMiIgc3Ryb2tlPSIjQkJCQkJCIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48bGluZSB4MT0iMzUuNiIgeTE9IjYuMCIgeDI9IjQyLjYiIHkyPSI2LjAiIHN0cm9rZT0iIzRDQTY0QyIgc3Ryb2tlLXdpZHRoPSIyIj48L2xpbmU+PGNpcmNsZSBjeD0iMzkuMSIgY3k9IjYuMCIgcj0iMyIgZmlsbD0iIzRDQTY0QyI+PC9jaXJjbGU+PC9zdmc+) helps |
| Intervals are 95% paired t-intervals, not adjusted for multiple comparisons; verdicts use Holm-adjusted p-values. |  |  |  |
| ![](data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDYwIiBoZWlnaHQ9IjIzMCIgdmlld2JveD0iMCAwIDQ2MCAyMzAiIHN0eWxlPSJkaXNwbGF5OiBibG9jazsgbWFyZ2luLXRvcDogNHB4OyI+PHRleHQgeD0iMCIgeT0iMTMiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsgZm9udC1zaXplOiAxMXB4OyBmaWxsOiAjMzMzMzMzOyI+U3RlcCAxLCB2YXJ5aW5nIHZhbHVlOiDOlCBlcnJvciB2cyBmdWxsIChsb3dlciBpcyBiZXR0ZXIpPC90ZXh0PjxsaW5lIHgxPSI1OCIgeTE9IjQ0IiB4Mj0iNTgiIHkyPSIxOTAiIHN0cm9rZT0iI0JCQkJCQiI+PC9saW5lPjxsaW5lIHgxPSI1OCIgeTE9IjE5MCIgeDI9IjQ0NCIgeTI9IjE5MCIgc3Ryb2tlPSIjQkJCQkJCIj48L2xpbmU+PGxpbmUgeDE9IjU4IiB5MT0iOTAuMCIgeDI9IjQ0NCIgeTI9IjkwLjAiIHN0cm9rZT0iIzk5OTk5OSIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PGxpbmUgeDE9IjU4IiB5MT0iMTgyLjAiIHgyPSI0NDQiIHkyPSIxODIuMCIgc3Ryb2tlPSIjRUVFRUVFIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1MyIgeT0iMTg1LjAiIHRleHQtYW5jaG9yPSJlbmQiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPuKIkjEwPC90ZXh0PjxsaW5lIHgxPSI1OCIgeTE9IjEzNi4wIiB4Mj0iNDQ0IiB5Mj0iMTM2LjAiIHN0cm9rZT0iI0VFRUVFRSIgc3Ryb2tlLXdpZHRoPSIxIj48L2xpbmU+PHRleHQgeD0iNTMiIHk9IjEzOS4wIiB0ZXh0LWFuY2hvcj0iZW5kIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij7iiJI1PC90ZXh0Pjx0ZXh0IHg9IjUzIiB5PSI5My4wIiB0ZXh0LWFuY2hvcj0iZW5kIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4rMDwvdGV4dD48bGluZSB4MT0iNTgiIHkxPSI0NC4wIiB4Mj0iNDQ0IiB5Mj0iNDQuMCIgc3Ryb2tlPSIjRUVFRUVFIiBzdHJva2Utd2lkdGg9IjEiPjwvbGluZT48dGV4dCB4PSI1MyIgeT0iNDcuMCIgdGV4dC1hbmNob3I9ImVuZCIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+KzU8L3RleHQ+PHRleHQgeD0iNzIuMyIgeT0iMjAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4yMTwvdGV4dD48dGV4dCB4PSIxMjQuNCIgeT0iMjAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij41MDwvdGV4dD48dGV4dCB4PSIzNzUuOCIgeT0iMjAzIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIiBzdHlsZT0iZm9udC1mYW1pbHk6ICYjMzk7SUJNIFBsZXggU2FucyYjMzk7LCBzYW5zLXNlcmlmOyBmb250LXNpemU6IDEwcHg7IGZpbGw6ICM1NTU1NTU7Ij4xOTA8L3RleHQ+PHRleHQgeD0iNDI5LjciIHk9IjIwMyIgdGV4dC1hbmNob3I9Im1pZGRsZSIgc3R5bGU9ImZvbnQtZmFtaWx5OiAmIzM5O0lCTSBQbGV4IFNhbnMmIzM5Oywgc2Fucy1zZXJpZjsgZm9udC1zaXplOiAxMHB4OyBmaWxsOiAjNTU1NTU1OyI+MjIwPC90ZXh0Pjx0ZXh0IHg9IjI1MS4wIiB5PSIyMjQiIHRleHQtYW5jaG9yPSJtaWRkbGUiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPnZhbHVlIChzdGVwIDEpPC90ZXh0Pjxwb2x5bGluZSBwb2ludHM9IjcyLjMsOTMuNyAxMjQuNCw5My41IDM3NS44LDg5LjMgNDI5LjcsODYuMiIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjODg4ODg4IiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWRhc2hhcnJheT0iNCwzIj48L3BvbHlsaW5lPjxwb2x5bGluZSBwb2ludHM9IjcyLjMsMTQ3LjEgMTI0LjQsMTU3LjAgMzc1LjgsMTM1LjEgNDI5LjcsMTA1LjAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iIzMzMzMzMyIgc3Ryb2tlLXdpZHRoPSIxLjUiPjwvcG9seWxpbmU+PGxpbmUgeDE9IjcyLjMiIHkxPSIxNzIuNiIgeDI9IjcyLjMiIHkyPSIxMjEuNSIgc3Ryb2tlPSIjNENBNjRDIiBzdHJva2Utd2lkdGg9IjIiPjwvbGluZT48Y2lyY2xlIGN4PSI3Mi4zIiBjeT0iMTQ3LjEiIHI9IjMuNSIgZmlsbD0iIzRDQTY0QyI+PHRpdGxlPlN0ZXAgMSB3aXRoIHZhbHVlPTIxOiDiiJI2LjIwMTwvdGl0bGU+PC9jaXJjbGU+PGxpbmUgeDE9IjEyNC40IiB5MT0iMTgxLjkiIHgyPSIxMjQuNCIgeTI9IjEzMi4yIiBzdHJva2U9IiM0Q0E2NEMiIHN0cm9rZS13aWR0aD0iMiI+PC9saW5lPjxjaXJjbGUgY3g9IjEyNC40IiBjeT0iMTU3LjAiIHI9IjMuNSIgZmlsbD0iIzRDQTY0QyI+PHRpdGxlPlN0ZXAgMSB3aXRoIHZhbHVlPTUwOiDiiJI3LjI4MzwvdGl0bGU+PC9jaXJjbGU+PGxpbmUgeDE9IjM3NS44IiB5MT0iMTY4LjIiIHgyPSIzNzUuOCIgeTI9IjEwMi4wIiBzdHJva2U9IiM0Q0E2NEMiIHN0cm9rZS13aWR0aD0iMiI+PC9saW5lPjxjaXJjbGUgY3g9IjM3NS44IiBjeT0iMTM1LjEiIHI9IjMuNSIgZmlsbD0iIzRDQTY0QyI+PHRpdGxlPlN0ZXAgMSB3aXRoIHZhbHVlPTE5MDog4oiSNC44OTk8L3RpdGxlPjwvY2lyY2xlPjxsaW5lIHgxPSI0MjkuNyIgeTE9IjEyOS42IiB4Mj0iNDI5LjciIHkyPSI4MC40IiBzdHJva2U9IiM0Q0E2NEMiIHN0cm9rZS13aWR0aD0iMiI+PC9saW5lPjxjaXJjbGUgY3g9IjQyOS43IiBjeT0iMTA1LjAiIHI9IjMuNSIgZmlsbD0iIzRDQTY0QyI+PHRpdGxlPlN0ZXAgMSB3aXRoIHZhbHVlPTIyMDog4oiSMS42Mjg8L3RpdGxlPjwvY2lyY2xlPjxsaW5lIHgxPSIwIiB5MT0iMjYiIHgyPSIxOCIgeTI9IjI2IiBzdHJva2U9IiM4ODg4ODgiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtZGFzaGFycmF5PSI0LDMiPjwvbGluZT48dGV4dCB4PSIyMiIgeT0iMjkiIHN0eWxlPSJmb250LWZhbWlseTogJiMzOTtJQk0gUGxleCBTYW5zJiMzOTssIHNhbnMtc2VyaWY7IGZvbnQtc2l6ZTogMTBweDsgZmlsbDogIzU1NTU1NTsiPlJlbW92aW5nIGFzIG1hbnkgcmFuZG9tIHJvd3M8L3RleHQ+PC9zdmc+) |  |  |  |

A limit of `50` removes every bad reading. A limit of `21` also removes many good readings (the warmer ones), which leaves the average too low, and the limits of `190` and `220` miss some of the bad readings.

Every limit gets the verdict that it helps, though, since each verdict compares a point of the sweep with its own control (removing as many random rows), not with the other points. Even the limit of `21` does better than removing 117 random rows, which would leave most of the bad readings in. To pick a limit, compare the points' metrics with each other, as the curve below the report does: the error is lowest at `50`.

## See Also

[sweep_score()](../reference/Ablation.sweep_score.md#pointblank.Ablation.sweep_score)  
Remove the rows on the wrong side of each cutoff of a score.
