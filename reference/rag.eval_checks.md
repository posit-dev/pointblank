# rag.eval_checks()

Make validation steps that check the evaluation set of a RAG system.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/rag.py#L309-L477)

``` python
rag.eval_checks(
    query,
    gold_doc_id=None,
    corpus=None,
    corpus_id=None,
    passage=None,
    scorer=None,
    min_relevance=None,
    near_duplicates=True,
    pre=None,
    thresholds=None,
    actions=None,
    active=True,
)
```

A RAG system is evaluated with a set of queries, each with the document that has its answer (the *gold* document) and often the passage of it. The scores of the evaluation can only be trusted if the set itself is sound: every query has text, no query is asked twice (a near duplicate counts twice in the scores), every gold document is in the corpus (a query whose document was removed or renamed can never be answered), and each gold passage is relevant to its query. This function makes a validation step for each:

- every query has text (not missing, and not only whitespace): a `col_vals_regex()` step
- no query is a near duplicate of another, unless `near_duplicates=False`: a `rows_near_distinct()` step
- every query has a gold document, with `gold_doc_id=`: a `col_vals_not_null()` step
- every gold document is in the corpus, with `corpus=`: a `col_vals_in_table()` step
- every passage is relevant to its query, with `passage=`, `scorer=`, and `min_relevance=`: a `col_vals_scored()` step (with a reranker, like [`pb.scorers.reranker()`](../reference/scorers.reranker.md#pointblank.scorers.reranker), or another scorer of a query and a passage)

## Parameters

`query: str`  
The column with the queries.

`gold_doc_id: str | None = None`  
The column with the id of the gold document of each query.

`corpus: Any = None`  
The corpus (any table that `col_vals_in_table()` takes, such as a DataFrame, a database table, or the path of a file) to look up the gold documents in. It needs `gold_doc_id=`.

`corpus_id: str | None = None`  
The column of the corpus with the document ids (by default, the same name as `gold_doc_id=`).

`passage: str | None = None`  
The column with the gold passage of each query, for the check of relevance.

`scorer: Callable[…, Any] | None = None`  
The scorer of the relevance of a passage to its query, given the values of `query=` and `passage=` (in that order).

`min_relevance: float | None = None`  
The lowest score of relevance that a passage can have.

`near_duplicates: bool | float = True`  
Whether to check for near-duplicate queries (`True` by default), or the threshold of the similarity of two queries for them to be near duplicates (the `threshold=` of `rows_near_distinct()`, `0.85` with `True`).

`pre: Callable | None = None`  
A function that transforms the table before each step, as in the validation methods.

`thresholds: int | float | bool | tuple | dict | Thresholds | None = None`  
The thresholds of the steps (by default, those of the validation).

`actions: Actions | None = None`  
The actions of the steps.

`active: bool | Callable = True`  
Whether the steps are active.

## Returns

`Steps`  
The steps, to be added to a validation with [`Validate.add_steps()`](../reference/Validate.add_steps.md#pointblank.Validate.add_steps).

## Examples

Let's check an evaluation set of three queries against a corpus of three documents.

``` python
import pointblank as pb
import polars as pl

corpus = pl.DataFrame({"doc_id": ["account", "billing", "shipping"]})

queries = pl.DataFrame(
    {
        "query": [
            "How do I reset my password?",
            "Where can I download an invoice?",
            "how do I reset my password",
        ],
        "doc_id": ["account", "invoices", "account"],
    }
)

(
    pb.Validate(data=queries, tbl_name="queries")
    .add_steps(pb.rag.eval_checks("query", gold_doc_id="doc_id", corpus=corpus))
    .interrogate()
)
```

[TABLE]

The first and third queries are the same once normalized, so both fail step 2, and the second query's gold document (`invoices`) isn't in the corpus. A check of relevance needs a scorer, such as `scorer=pb.scorers.reranker(provider="cohere")` with `passage=` and `min_relevance=`.
