# scorers.reranker()

Score how relevant a document is to a query, with a reranking model.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/scorers.py#L665-L747)

``` python
scorers.reranker(
    provider="cohere",
    model=None,
    query=None,
    document=None,
    api_key=None,
    name=None,
    version=None,
    batch_size=None,
)
```

A reranker scores a pair of texts: how well a document (e.g., a passage retrieved for a search) answers a query. This is a check of the pairs of a retrieval dataset or of the results of a search: rows with a low score pair a query with a passage that doesn't answer it. The step needs two columns, the query and the document.

## Parameters

`provider: str = "cohere"`  
Where the model runs: `"cohere"` (the Cohere Rerank API, which needs the `cohere` package and an API key) or `"sentence-transformers"` (a cross-encoder model that runs locally, which needs the `sentence-transformers` package).

`model: str | None = None`  
The model. For Cohere, the name of a rerank model (`"rerank-v3.5"` by default). For sentence-transformers, the name of a cross-encoder model (`"cross-encoder/ms-marco-MiniLM-L6-v2"` by default) or a model with a `predict()` method that takes a list of (query, document) pairs.

`query: str | None = None`  
The column of the queries. By default, the first column of the step.

`document: str | None = None`  
The column of the documents. By default, the first column of the step that isn't the query column.

`api_key: str | None = None`  
The Cohere API key. By default, it's read from the `CO_API_KEY` environment variable. The key is never written in the code or the YAML made from a validation.

`name: str | None = None`  
A name for the scorer, for the score cache and the report. By default, it's made from the provider, the model, and the columns.

`version: str | None = None`  
A version (e.g., of the model), which is part of the key of the score cache.

`batch_size: int | None = None`  
The number of pairs scored at a time (`100` by default). For Cohere, the pairs of a batch with the same query are sent in one request.

## Returns

`Scorer`  
A scorer for [Validate.col_vals_scored()](../reference/Validate.col_vals_scored.md#pointblank.Validate.col_vals_scored).

## Details

The scores of different models aren't on the same scale: Cohere's relevance scores are between `0` and `1`, while a cross-encoder may give unbounded logits. Look at the scores (they're kept in the step's results with `keep_scores=True`) before choosing a threshold, or use [`Ablation.sweep_score()`](../reference/Ablation.sweep_score.md#pointblank.Ablation.sweep_score) to see how the threshold affects a model trained on the data.

## Examples

This checks that the passages of a retrieval dataset are relevant to their queries, with Cohere's reranker (an API key is needed).

``` python
import pointblank as pb

validation = (
    pb.Validate(data=pairs)
    .col_vals_scored(
        columns=["query", "passage"],
        scorer=pb.scorers.reranker(provider="cohere", model="rerank-v3.5"),
        min_val=0.5,
    )
    .interrogate()
)
```

The same check with a cross-encoder that runs locally would use `pb.scorers.reranker(provider="sentence-transformers")`. Its scores are on another scale, so the threshold would be different.
