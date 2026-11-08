# scorers.embedding_similarity()

Score text by its similarity to reference texts, with an embedding model.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/scorers.py#L473-L550)

``` python
scorers.embedding_similarity(
    reference,
    embedder,
    column=None,
    name=None,
    version=None,
    batch_size=None,
)
```

Each row's text is embedded, and its score is the cosine similarity of its embedding to the closest of the embeddings of the reference texts: close to `1` for text like one of the references, and lower for text that isn't. It's a way to check that text is on topic, or to find text that's too close to a set of texts that shouldn't be repeated (with `max_val=`).

## Parameters

`reference: str | list[str]`  
A reference text, or a list of them.

`embedder: Any`  
The embedding model: the name of a [sentence-transformers](https://sbert.net) model (e.g., `"all-MiniLM-L6-v2"`), which needs the `sentence-transformers` package, a model with an `encode()` method that takes a list of texts, or a function that takes a list of texts and returns their embeddings (a list of vectors or a 2D array).

`column: str | None = None`  
The column with the text to score. It can be left out if the step has one column.

`name: str | None = None`  
A name for the scorer, for the score cache and the report. By default, it's made from the embedder (if it's a model name or a function at the top level of a module), the references, and `column=`. With a model object or a lambda as the embedder, the scores can only be cached in memory unless there's a name.

`version: str | None = None`  
A version (e.g., of the model), which is part of the key of the score cache.

`batch_size: int | None = None`  
The number of rows embedded at a time (`64` by default).

## Returns

`Scorer`  
A scorer for [Validate.col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored).

## Examples

A real embedder would be a model like `"all-MiniLM-L6-v2"`. To keep this example self-contained, the embedding of a text here counts its letters.

``` python
import pointblank as pb
import polars as pl

def letter_counts(texts):
    return [[t.lower().count(c) for c in "abcdefghijklmnopqrstuvwxyz"] for t in texts]

tbl = pl.DataFrame(
    {"title": ["data validation", "validating data", "zzz", "quality of data"]}
)

on_topic = pb.scorers.embedding_similarity(
    reference="data validation", embedder=letter_counts
)

pb.Validate(data=tbl).col_vals_scored(
    columns="title", scorer=on_topic, min_val=0.6
).interrogate()
```

[TABLE]

The title `"zzz"` has nothing in common with the reference, so it's the one that fails.
