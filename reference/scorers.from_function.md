# scorers.from_function()

Make a scorer from a function, with a name, a version, and a batch size.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/scorers.py#L324-L393)

``` python
scorers.from_function(
    fn,
    name=None,
    version=None,
    batch_size=None,
    per_row=False,
)
```

Any function of a batch of rows can be used as a scorer in [Validate.col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored). This wraps one to give it the attributes of the [`Scorer`](../reference/scorers.Scorer.md#pointblank.scorers.Scorer) protocol: a stable name for the score cache and the report, a version that's part of the key of the cache, and the number of rows to score at a time. With `per_row=True`, it also wraps a function that scores one row at a time.

## Parameters

`fn: Callable[…, Any]`  
The function. It takes a list of rows (dictionaries of the values of the columns of the step) and returns a score for each, or with `per_row=True`, it takes one row and returns its score. A score is a number, or `None` for a row that can't be scored.

`name: str | None = None`  
A stable name for the scorer. A function defined at the top level of a module is named after its module and its name by default; a lambda or a function defined inside another function has no name unless it's given here, and its scores can only be cached in memory.

`version: str | None = None`  
A version of the scorer (e.g., of its model). Scores in the cache are kept by name and version, so changing the version means scoring the rows again.

`batch_size: int | None = None`  
The number of rows that the function gets at a time (`256` by default).

`per_row: bool = False`  
Whether the function scores one row at a time (`True`) or a list of rows (`False`, the default).

## Returns

`Scorer`  
A scorer for [Validate.col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored).

## Examples

Here, a function that scores one row (a ratio of two columns) becomes a scorer with a name and a version.

``` python
import pointblank as pb
import polars as pl

tbl = pl.DataFrame({"clicks": [5, 40, 2, 18], "views": [100, 200, 120, 90]})

ctr = pb.scorers.from_function(
    lambda row: row["clicks"] / row["views"], name="ctr", version="1", per_row=True
)

pb.Validate(data=tbl).col_vals_scored(
    columns=["clicks", "views"], scorer=ctr, min_val=0.03
).interrogate()
```

[TABLE]

The scorer's name, `ctr`, is shown in the report. One row has a click-through rate below 3%.
