# scorers.Scorer

The protocol of the scorers used by [Validate.col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored).

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/scorers.py#L71-L130)

``` python
scorers.Scorer()
```

A scorer is a callable that takes a batch of rows and returns one score for each. Each row is a dictionary of the values of the columns of the step (e.g., `{"query": ..., "passage": ...}`), and the scores are numbers (or `None` for a row that can't be scored), in the order of the rows. Any function that does this is a scorer, as are the adapters of `pb.scorers` (like [`classifier()`](../reference/scorers.classifier.md#pointblank.scorers.classifier)).

A scorer can have three attributes, all optional:

- `name`: a stable name for the scorer, which identifies it in the score cache and in the report (a function defined at the top level of a module is named after the function)
- `version`: a version, which is part of the key of the score cache, so that the scores of an older version of a model aren't used for a newer one
- `batch_size`: the number of rows to give the scorer at a time (`256` if it doesn't say)

[`from_function()`](../reference/scorers.from_function.md#pointblank.scorers.from_function) sets these attributes for a function, and can also wrap a function that scores one row at a time.

## Examples

Here's a scorer that gives each row the share of its words that are longer than three letters, as a plain function of a batch of rows.

``` python
import pointblank as pb
import polars as pl

def long_words(batch):
    scores = []
    for row in batch:
        words = row["text"].split()
        scores.append(sum(len(w) > 3 for w in words) / len(words))
    return scores

tbl = pl.DataFrame(
    {"text": ["the cat sat on a mat", "careful thinking pays off", "no", "ok go"]}
)

pb.Validate(data=tbl).col_vals_scored(
    columns="text", scorer=long_words, min_val=0.5
).interrogate()
```

[TABLE]

Three of the four rows have a score below `0.5`, so they fail.
