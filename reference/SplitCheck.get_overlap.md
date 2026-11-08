# SplitCheck.get_overlap()

Get the rows of a split that are also in an earlier split.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/splits.py#L433-L489)

``` python
SplitCheck.get_overlap(
    first=None,
    second=None,
)
```

## Parameters

`first: str | None = None`  
The names of the earlier and later splits. By default, they're the first two.

`second: str | None = None`  
The names of the earlier and later splits. By default, they're the first two.

## Returns

`DataFrame`  
A Polars DataFrame (or a pandas one, if Polars isn't installed) with a row for each row of `second` that's also in `first`: the row numbers in each split (from 1), the kind of overlap (`"exact"` or `"near"`), and the values of the columns that were compared (from the later split).

## Examples

``` python
import pointblank as pb
import polars as pl

train = pl.DataFrame({"text": ["a cat", "a dog", "a bird"]})
test = pl.DataFrame({"text": ["a fish", "a dog"]})

pb.check_splits(train=train, test=test).get_overlap()
```

shape: (1, 4)

| train_row | test_row | overlap | text    |
|-----------|----------|---------|---------|
| i64       | i64      | str     | str     |
| 2         | 2        | "exact" | "a dog" |

The second row of the test split is the same as the second row of the training split.
