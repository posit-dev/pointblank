# SplitCheck.to_dict()

Get the results as a dictionary (that can be written as JSON).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/splits.py#L527-L582)

``` python
SplitCheck.to_dict()
```

## Returns

`dict`  
The splits and their sizes, the settings, and the results of the checks of overlap, labels, and drift, with the issues found.

## Examples

``` python
import pointblank as pb
import polars as pl

train = pl.DataFrame({"text": ["a cat", "a dog", "a bird"], "label": [1, 1, 2]})
test = pl.DataFrame({"text": ["a fish", "a dog"], "label": [2, 1]})

results = pb.check_splits(train=train, test=test, label="label").to_dict()
results["labels"]["counts"]
```

    {'train': [{'label': 1, 'n': 2}, {'label': 2, 'n': 1}],
     'test': [{'label': 1, 'n': 1}, {'label': 2, 'n': 1}]}

The counts of the labels are lists of records, rather than dictionaries keyed by label, so that labels that aren't strings (like these integers) are kept when the results are written as JSON.
