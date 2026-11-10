"""
Scorers for `Validate.col_vals_scored()`: functions that give each row of a table a score.

A scorer is any callable that takes a batch of rows (a list of dictionaries, one per row, with
the values of the columns of the step) and returns a score for each, in order. This module has
adapters for common kinds of scores (similarity of embeddings, rerankers, text classifiers,
language identification, and perplexity), each of which uses an optional dependency that's only
imported when the scorer is first used. Pointblank never trains a model: the scorers use models
that are given to them.
"""

from __future__ import annotations

import datetime
import decimal
import functools
import hashlib
import inspect
import itertools
import json
import math
import os
import sqlite3
import threading
import weakref
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

__all__ = [
    "Scorer",
    "from_function",
    "embedding_similarity",
    "reranker",
    "classifier",
    "langid",
    "perplexity",
    "text_stat",
    "token_count",
    "clear_cache",
]

# The number of rows that a scorer gets at a time, if it doesn't say
_DEFAULT_BATCH_SIZE = 256

# The largest number of scores kept in memory for one scorer (the oldest are dropped first)
_MEMORY_CACHE_LIMIT = 1_000_000

# The file of a score cache on disk, in the directory given as `cache=`
CACHE_FILE = "scores.sqlite"

# The adapters that can be written in YAML as `scorer: {<name>: {<arguments>}}`
ADAPTERS = (
    "embedding_similarity",
    "reranker",
    "classifier",
    "langid",
    "perplexity",
    "text_stat",
    "token_count",
)

_RERANK_PROVIDERS = ("cohere", "sentence-transformers")
_DEFAULT_RERANK_MODELS = {
    "cohere": "rerank-v3.5",
    "sentence-transformers": "cross-encoder/ms-marco-MiniLM-L6-v2",
}
_LANGID_BACKENDS = ("langid", "fasttext")


@runtime_checkable
class Scorer(Protocol):
    """
    The protocol of the scorers used by `Validate.col_vals_scored()`.

    A scorer is a callable that takes a batch of rows and returns one score for each. Each row is
    a dictionary of the values of the columns of the step (e.g., `{"query": ..., "passage": ...}`),
    and the scores are numbers (or `None` for a row that can't be scored), in the order of the
    rows. Any function that does this is a scorer, as are the adapters of `pb.scorers` (like
    [`classifier()`](`pointblank.scorers.classifier`)).

    A scorer can have three attributes, all optional:

    - `name`: a stable name for the scorer, which identifies it in the score cache and in the
      report (a function defined at the top level of a module is named after the function)
    - `version`: a version, which is part of the key of the score cache, so that the scores of an
      older version of a model aren't used for a newer one
    - `batch_size`: the number of rows to give the scorer at a time (`256` if it doesn't say)

    [`from_function()`](`pointblank.scorers.from_function`) sets these attributes for a function,
    and can also wrap a function that scores one row at a time.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Here's a scorer that gives each row the share of its words that are longer than three
    letters, as a plain function of a batch of rows.

    ```{python}
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

    Three of the four rows have a score below `0.5`, so they fail.
    """

    def __call__(
        self, batch: list[dict[str, Any]]
    ) -> Sequence[float | None]: ...  # pragma: no cover


# ---------------------------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------------------------


def _is_literal(value: Any) -> bool:
    """Check that a value can be written as a Python literal (and in YAML)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_literal(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_literal(v) for k, v in value.items())
    return False


def _stable_name(fn: Any) -> str | None:
    """
    Get the qualified name of a function defined at the top level of a module.

    A lambda, a function defined inside another function, a method of an object, and a callable
    object have no stable name: two of them can have the same name and score differently.
    """
    if not (inspect.isfunction(fn) or inspect.isbuiltin(fn)):
        return None
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname or "<" in qualname or "." in qualname:
        return None
    return f"{module}.{qualname}"


def _digest(text: str, size: int = 4) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def _literal_code(value: Any) -> str:
    """Write a literal as Python code, with strings in double quotes (as in `Validate.to_code()`)."""
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_literal_code(v) for v in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(f"{_literal_code(k)}: {_literal_code(v)}" for k, v in value.items())
            + "}"
        )
    return repr(value)


def _describe(value: Any) -> str | None:
    """Describe a setting of an adapter in its name, or give `None` if it has no stable name."""
    if _is_literal(value):
        text = repr(list(value) if isinstance(value, tuple) else value)
        return text if len(text) <= 60 else f"<{_digest(text)}>"
    return _stable_name(value)


class _Adapter:
    """
    The base of the scorers made by `pb.scorers`.

    `_params` holds the arguments that were given (those that aren't the default), which is how
    the scorer is written as code (`pb.scorers.<kind>(...)`) and in YAML (`{<kind>: {...}}`).
    """

    _kind = ""
    _label = ""
    _default_batch_size = _DEFAULT_BATCH_SIZE

    def __init__(
        self,
        params: dict[str, Any],
        settings: dict[str, Any],
        name: str | None,
        version: str | None,
        batch_size: int | None,
    ):
        if name is not None and (not isinstance(name, str) or not name):
            raise TypeError("`name=` must be a non-empty string.")
        if version is not None and not isinstance(version, (str, int, float)):
            raise TypeError("`version=` must be a string or a number.")
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1
        ):
            raise ValueError("`batch_size=` must be a positive integer.")
        self._params = {k: v for k, v in params.items()}
        for key, value in (("name", name), ("version", version), ("batch_size", batch_size)):
            if value is not None:
                self._params[key] = value
        self.name = name if name is not None else self._settings_name(settings)
        self.version = None if version is None else str(version)
        self.batch_size = batch_size if batch_size is not None else self._default_batch_size

    def _settings_name(self, settings: dict[str, Any]) -> str | None:
        """Name the scorer by the settings that change its scores (`None` if one can't be named)."""
        parts = []
        for key, value in settings.items():
            if value is None:
                continue
            text = _describe(value)
            if text is None:
                return None
            parts.append(f"{key}={text}")
        return f"{self._kind}({', '.join(parts)})"

    def _is_literal(self) -> bool:
        return _is_literal(self._params)

    def _to_code(self) -> str:
        args = ", ".join(f"{k}={_literal_code(v)}" for k, v in self._params.items())
        return f"pb.scorers.{self._kind}({args})"

    def _to_yaml(self) -> dict[str, Any]:
        params = {k: list(v) if isinstance(v, tuple) else v for k, v in self._params.items()}
        return {self._kind: params}

    def __repr__(self) -> str:
        if self._is_literal():
            return self._to_code()
        args = ", ".join(
            f"{k}={_literal_code(v)}" if _is_literal(v) else f"{k}=<{type(v).__name__}>"
            for k, v in self._params.items()
        )
        return f"pb.scorers.{self._kind}({args})"

    def __deepcopy__(self, memo: dict[int, Any]) -> _Adapter:
        # A segmented step is copied for each segment: the copies share the scorer (and the model
        # it has loaded), as its settings never change
        return self

    def __call__(self, batch: list[dict[str, Any]]) -> Sequence[float | None]:
        raise NotImplementedError  # pragma: no cover


def _install_hint(what: str, package: str, error: ImportError) -> ImportError:
    return ImportError(
        f"{what} needs the {package} package, which can be installed with "
        f"`pip install {package}` ({error})."
    )


def _text_of(record: dict[str, Any], column: str | None, kind: str) -> str:
    """Get the text that an adapter scores from a row."""
    if column is None:
        if len(record) != 1:
            raise ValueError(
                f"{kind}() scores the text of one column, and the step has the columns "
                f"{', '.join(map(repr, record))}: choose one with `column=`."
            )
        value = next(iter(record.values()))
    else:
        if column not in record:
            raise ValueError(
                f"{kind}() scores the column {column!r} (`column=`), which isn't one of the "
                f"columns of the step ({', '.join(map(repr, record))})."
            )
        value = record[column]
    return value if isinstance(value, str) else str(value)


def _check_column(column: Any) -> None:
    if column is not None and (not isinstance(column, str) or not column):
        raise TypeError("`column=` must be the name of a column.")


class _FunctionScorer(_Adapter):
    _kind = "from_function"

    def __init__(self, fn, name, version, batch_size, per_row):
        if not callable(fn):
            raise TypeError(f"`fn=` must be a function, got {type(fn).__name__}.")
        if not isinstance(per_row, bool):
            raise TypeError("`per_row=` must be `True` or `False`.")
        self._fn = fn
        self._per_row = per_row
        params = {"fn": fn}
        if per_row:
            params["per_row"] = True
        super().__init__(params, {}, name, version, batch_size)
        if name is None:
            self.name = _stable_name(fn)
        self._label = name or getattr(fn, "__name__", None) or type(fn).__name__

    def __call__(self, batch):
        if self._per_row:
            return [self._fn(record) for record in batch]
        return self._fn(batch)


def from_function(
    fn: Callable[..., Any],
    name: str | None = None,
    version: str | None = None,
    batch_size: int | None = None,
    per_row: bool = False,
) -> Scorer:
    """
    Make a scorer from a function, with a name, a version, and a batch size.

    Any function of a batch of rows can be used as a scorer in `Validate.col_vals_scored()`. This
    wraps one to give it the attributes of the [`Scorer`](`pointblank.scorers.Scorer`) protocol:
    a stable name for the score cache and the report, a version that's part of the key of the
    cache, and the number of rows to score at a time. With `per_row=True`, it also wraps a
    function that scores one row at a time.

    Parameters
    ----------
    fn
        The function. It takes a list of rows (dictionaries of the values of the columns of the
        step) and returns a score for each, or with `per_row=True`, it takes one row and returns
        its score. A score is a number, or `None` for a row that can't be scored.
    name
        A stable name for the scorer. A function defined at the top level of a module is named
        after its module and its name by default; a lambda or a function defined inside another
        function has no name unless it's given here, and its scores can only be cached in memory.
    version
        A version of the scorer (e.g., of its model). Scores in the cache are kept by name and
        version, so changing the version means scoring the rows again.
    batch_size
        The number of rows that the function gets at a time (`256` by default).
    per_row
        Whether the function scores one row at a time (`True`) or a list of rows (`False`, the
        default).

    Returns
    -------
    Scorer
        A scorer for `Validate.col_vals_scored()`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Here, a function that scores one row (a ratio of two columns) becomes a scorer with a name
    and a version.

    ```{python}
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

    The scorer's name, `ctr`, is shown in the report. One row has a click-through rate below 3%.
    """
    return _FunctionScorer(fn, name, version, batch_size, per_row)


def _to_vectors(output: Any, n: int, what: str) -> list[list[float]]:
    """Turn the output of an embedder (an array, a tensor, or a list of lists) into vectors."""
    if hasattr(output, "tolist"):
        output = output.tolist()
    vectors = [[float(x) for x in row] for row in output]
    if len(vectors) != n:
        raise ValueError(f"The embedder returned {len(vectors)} vectors for {n} {what}.")
    return vectors


def _unit(vector: list[float]) -> list[float] | None:
    norm = math.sqrt(sum(x * x for x in vector))
    return None if norm == 0 else [x / norm for x in vector]


class _EmbeddingSimilarity(_Adapter):
    _kind = "embedding_similarity"
    _label = "EMBEDDING"
    _default_batch_size = 64

    def __init__(self, reference, embedder, column, name, version, batch_size):
        references = [reference] if isinstance(reference, str) else reference
        if not isinstance(references, (list, tuple)) or not references:
            raise TypeError("`reference=` must be a string or a non-empty list of strings.")
        if not all(isinstance(r, str) for r in references):
            raise TypeError("`reference=` must be a string or a non-empty list of strings.")
        if not (isinstance(embedder, str) or callable(getattr(embedder, "encode", None))):
            if not callable(embedder):
                raise TypeError(
                    "`embedder=` must be the name of a sentence-transformers model, a model with "
                    f"an `encode()` method, or a function, got {type(embedder).__name__}."
                )
        _check_column(column)
        self._references = list(references)
        self._embedder = embedder
        self._column = column
        self._encode = None
        self._reference_vectors = None
        params = {"reference": reference, "embedder": embedder}
        if column is not None:
            params["column"] = column
        reference_text = json.dumps(self._references, ensure_ascii=False)
        settings = {
            "embedder": embedder,
            "reference": (
                f"{len(self._references)} text{'' if len(self._references) == 1 else 's'}, "
                f"{_digest(reference_text)}"
            ),
            "column": column,
        }
        super().__init__(params, settings, name, version, batch_size)

    def _encoder(self):
        if self._encode is None:
            embedder = self._embedder
            if isinstance(embedder, str):
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as e:
                    raise _install_hint(
                        "embedding_similarity() with a model name", "sentence-transformers", e
                    ) from e
                self._encode = SentenceTransformer(embedder).encode
            elif callable(getattr(embedder, "encode", None)):
                self._encode = embedder.encode
            else:
                self._encode = embedder
        return self._encode

    def __call__(self, batch):
        texts = [_text_of(record, self._column, self._kind) for record in batch]
        encode = self._encoder()
        if self._reference_vectors is None:
            vectors = _to_vectors(encode(self._references), len(self._references), "references")
            self._reference_vectors = [v for v in map(_unit, vectors) if v is not None]
            if not self._reference_vectors:
                raise ValueError("The embedder gave the references only zero vectors.")
        scores = []
        for vector in map(_unit, _to_vectors(encode(texts), len(texts), "rows")):
            if vector is None:
                scores.append(None)
            else:
                scores.append(
                    max(sum(a * b for a, b in zip(vector, r)) for r in self._reference_vectors)
                )
        return scores


def embedding_similarity(
    reference: str | list[str],
    embedder: Any,
    column: str | None = None,
    name: str | None = None,
    version: str | None = None,
    batch_size: int | None = None,
) -> Scorer:
    """
    Score text by its similarity to reference texts, with an embedding model.

    Each row's text is embedded, and its score is the cosine similarity of its embedding to the
    closest of the embeddings of the reference texts: close to `1` for text like one of the
    references, and lower for text that isn't. It's a way to check that text is on topic, or to
    find text that's too close to a set of texts that shouldn't be repeated (with `max_val=`).

    Parameters
    ----------
    reference
        A reference text, or a list of them.
    embedder
        The embedding model: the name of a
        [sentence-transformers](https://sbert.net) model (e.g., `"all-MiniLM-L6-v2"`), which
        needs the `sentence-transformers` package, a model with an `encode()` method that takes a
        list of texts, or a function that takes a list of texts and returns their embeddings (a
        list of vectors or a 2D array).
    column
        The column with the text to score. It can be left out if the step has one column.
    name
        A name for the scorer, for the score cache and the report. By default, it's made from the
        embedder (if it's a model name or a function at the top level of a module), the
        references, and `column=`. With a model object or a lambda as the embedder, the scores
        can only be cached in memory unless there's a name.
    version
        A version (e.g., of the model), which is part of the key of the score cache.
    batch_size
        The number of rows embedded at a time (`64` by default).

    Returns
    -------
    Scorer
        A scorer for `Validate.col_vals_scored()`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    A real embedder would be a model like `"all-MiniLM-L6-v2"`. To keep this example
    self-contained, the embedding of a text here counts its letters.

    ```{python}
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

    The title `"zzz"` has nothing in common with the reference, so it's the one that fails.
    """
    return _EmbeddingSimilarity(reference, embedder, column, name, version, batch_size)


class _Reranker(_Adapter):
    _kind = "reranker"
    _label = "RERANKER"
    _default_batch_size = 100

    def __init__(self, provider, model, query, document, api_key, name, version, batch_size):
        if provider not in _RERANK_PROVIDERS:
            raise ValueError(
                f"`provider=` must be one of {', '.join(map(repr, _RERANK_PROVIDERS))}, "
                f"got {provider!r}."
            )
        if model is not None and not isinstance(model, str):
            if provider == "cohere":
                raise TypeError("`model=` must be the name of a Cohere rerank model.")
            if not callable(getattr(model, "predict", None)):
                raise TypeError(
                    "`model=` must be the name of a cross-encoder model or a model with a "
                    "`predict()` method."
                )
        for value, param in ((query, "query"), (document, "document")):
            if value is not None and (not isinstance(value, str) or not value):
                raise TypeError(f"`{param}=` must be the name of a column.")
        if query is not None and query == document:
            raise ValueError("`query=` and `document=` must be different columns.")
        self._provider = provider
        self._model = model if model is not None else _DEFAULT_RERANK_MODELS[provider]
        self._query = query
        self._document = document
        self._api_key = api_key
        self._client = None
        params = {"provider": provider}
        for key, value in (("model", model), ("query", query), ("document", document)):
            if value is not None:
                params[key] = value
        settings = {
            "provider": provider,
            "model": self._model,
            "query": query,
            "document": document,
        }
        # The API key is left out of the code, the YAML, and the name of the scorer
        super().__init__(params, settings, name, version, batch_size)

    def _columns(self, record: dict[str, Any]) -> tuple[str, str]:
        keys = list(record)
        query = self._query
        document = self._document
        if query is None:
            others = [k for k in keys if k != document]
            if not others:
                raise ValueError("reranker() needs a query column and a document column.")
            query = others[0]
        if document is None:
            others = [k for k in keys if k != query]
            if not others:
                raise ValueError("reranker() needs a query column and a document column.")
            document = others[0]
        for column, param in ((query, "query"), (document, "document")):
            if column not in record:
                raise ValueError(
                    f"reranker() uses the column {column!r} (`{param}=`), which isn't one of the "
                    f"columns of the step ({', '.join(map(repr, keys))})."
                )
        return query, document

    def _get_client(self):
        if self._client is None:
            if self._provider == "cohere":
                try:
                    import cohere
                except ImportError as e:
                    raise _install_hint('reranker(provider="cohere")', "cohere", e) from e
                if self._api_key is not None:
                    self._client = cohere.ClientV2(api_key=self._api_key)
                else:
                    self._client = cohere.ClientV2()
            elif isinstance(self._model, str):
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as e:
                    raise _install_hint(
                        'reranker(provider="sentence-transformers")', "sentence-transformers", e
                    ) from e
                self._client = CrossEncoder(self._model)
            else:
                self._client = self._model
        return self._client

    def __call__(self, batch):
        pairs = []
        for record in batch:
            query, document = self._columns(record)
            pairs.append((str(record[query]), str(record[document])))
        client = self._get_client()
        if self._provider == "sentence-transformers":
            return client.predict(pairs)
        scores: list[float | None] = [None] * len(pairs)
        by_query: dict[str, list[int]] = {}
        for i, (query, _) in enumerate(pairs):
            by_query.setdefault(query, []).append(i)
        for query, indices in by_query.items():
            response = client.rerank(
                model=self._model,
                query=query,
                documents=[pairs[i][1] for i in indices],
                top_n=len(indices),
            )
            for result in response.results:
                scores[indices[result.index]] = result.relevance_score
        return scores


def reranker(
    provider: str = "cohere",
    model: str | None = None,
    query: str | None = None,
    document: str | None = None,
    api_key: str | None = None,
    name: str | None = None,
    version: str | None = None,
    batch_size: int | None = None,
) -> Scorer:
    """
    Score how relevant a document is to a query, with a reranking model.

    A reranker scores a pair of texts: how well a document (e.g., a passage retrieved for a
    search) answers a query. This is a check of the pairs of a retrieval dataset or of the
    results of a search: rows with a low score pair a query with a passage that doesn't answer it.
    The step needs two columns, the query and the document.

    Parameters
    ----------
    provider
        Where the model runs: `"cohere"` (the Cohere Rerank API, which needs the `cohere` package
        and an API key) or `"sentence-transformers"` (a cross-encoder model that runs locally,
        which needs the `sentence-transformers` package).
    model
        The model. For Cohere, the name of a rerank model (`"rerank-v3.5"` by default). For
        sentence-transformers, the name of a cross-encoder model
        (`"cross-encoder/ms-marco-MiniLM-L6-v2"` by default) or a model with a `predict()` method
        that takes a list of (query, document) pairs.
    query
        The column of the queries. By default, the first column of the step.
    document
        The column of the documents. By default, the first column of the step that isn't the
        query column.
    api_key
        The Cohere API key. By default, it's read from the `CO_API_KEY` environment variable. The
        key is never written in the code or the YAML made from a validation.
    name
        A name for the scorer, for the score cache and the report. By default, it's made from the
        provider, the model, and the columns.
    version
        A version (e.g., of the model), which is part of the key of the score cache.
    batch_size
        The number of pairs scored at a time (`100` by default). For Cohere, the pairs of a batch
        with the same query are sent in one request.

    Returns
    -------
    Scorer
        A scorer for `Validate.col_vals_scored()`.

    Details
    -------
    The scores of different models aren't on the same scale: Cohere's relevance scores are
    between `0` and `1`, while a cross-encoder may give unbounded logits. Look at the scores
    (they're kept in the step's results with `keep_scores=True`) before choosing a threshold, or
    use [`Ablation.sweep_score()`](`pointblank.Ablation.sweep_score`) to see how the threshold
    affects a model trained on the data.

    Examples
    --------
    This checks that the passages of a retrieval dataset are relevant to their queries, with
    Cohere's reranker (an API key is needed).

    ```python
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

    The same check with a cross-encoder that runs locally would use
    `pb.scorers.reranker(provider="sentence-transformers")`. Its scores are on another scale, so
    the threshold would be different.
    """
    return _Reranker(provider, model, query, document, api_key, name, version, batch_size)


class _Classifier(_Adapter):
    _kind = "classifier"
    _label = "CLASSIFIER"
    _default_batch_size = 32

    def __init__(
        self, hf_model, label, column, function_to_apply, device, name, version, batch_size
    ):
        if not isinstance(hf_model, str) and not callable(hf_model):
            raise TypeError(
                "`hf_model=` must be the name of a Hugging Face model or a text-classification "
                f"pipeline, got {type(hf_model).__name__}."
            )
        if label is not None and not isinstance(label, str):
            raise TypeError("`label=` must be the name of a label of the model.")
        if function_to_apply is not None and function_to_apply not in (
            "sigmoid",
            "softmax",
            "none",
        ):
            raise ValueError('`function_to_apply=` must be "sigmoid", "softmax", or "none".')
        _check_column(column)
        self._hf_model = hf_model
        self._target = label
        self._column = column
        self._function_to_apply = function_to_apply
        self._device = device
        self._pipeline = None
        params = {"hf_model": hf_model}
        for key, value in (
            ("label", label),
            ("column", column),
            ("function_to_apply", function_to_apply),
            ("device", device),
        ):
            if value is not None:
                params[key] = value
        model_name = hf_model
        if not isinstance(hf_model, str):
            model_name = getattr(getattr(hf_model, "model", None), "name_or_path", None)
            model_name = model_name if isinstance(model_name, str) and model_name else hf_model
        settings = {
            "hf_model": model_name,
            "label": label,
            "column": column,
            "function_to_apply": function_to_apply,
        }
        super().__init__(params, settings, name, version, batch_size)

    def _get_pipeline(self):
        if self._pipeline is None:
            if isinstance(self._hf_model, str):
                try:
                    from transformers import pipeline
                except ImportError as e:
                    raise _install_hint("classifier() with a model name", "transformers", e) from e
                kwargs: dict[str, Any] = {"task": "text-classification", "model": self._hf_model}
                if self._device is not None:
                    kwargs["device"] = self._device
                self._pipeline = pipeline(**kwargs)
            else:
                self._pipeline = self._hf_model
        return self._pipeline

    def __call__(self, batch):
        texts = [_text_of(record, self._column, self._kind) for record in batch]
        kwargs: dict[str, Any] = {"top_k": None, "truncation": True}
        if self._function_to_apply is not None:
            kwargs["function_to_apply"] = self._function_to_apply
        outputs = self._get_pipeline()(texts, **kwargs)
        if len(texts) == 1 and outputs and isinstance(outputs[0], dict):
            outputs = [outputs]
        scores = []
        for output in outputs:
            items = [output] if isinstance(output, dict) else list(output)
            if self._target is None:
                if len(items) != 1:
                    labels = ", ".join(repr(item["label"]) for item in items)
                    raise ValueError(
                        f"The model has the labels {labels}: choose the one to score with `label=`."
                    )
                scores.append(items[0]["score"])
                continue
            match = [item["score"] for item in items if item["label"] == self._target]
            if not match:
                labels = ", ".join(repr(item["label"]) for item in items)
                raise ValueError(
                    f"The model has no label {self._target!r} (`label=`); its labels are {labels}."
                )
            scores.append(match[0])
        return scores


def classifier(
    hf_model: Any,
    label: str | None = None,
    column: str | None = None,
    function_to_apply: str | None = None,
    device: Any = None,
    name: str | None = None,
    version: str | None = None,
    batch_size: int | None = None,
) -> Scorer:
    """
    Score text with a text-classification model from Hugging Face.

    A text classifier scores text for a label: a quality or toxicity classifier, a sentiment
    model, or a classifier of topics. The score of a row is the score that the model gives its
    text for `label=` (with a regression model that has one output, like an educational quality
    classifier, there's no label to choose). It needs the `transformers` package (and a backend
    for it, like PyTorch) unless it's given a pipeline that's already made.

    Parameters
    ----------
    hf_model
        The model: the name of a model on the Hugging Face Hub (e.g.,
        `"HuggingFaceFW/fineweb-edu-classifier"`), or a `transformers` text-classification
        pipeline (or any function that takes a list of texts and returns, for each, a list of
        `{"label": ..., "score": ...}` dictionaries).
    label
        The label whose score is the score of a row. It can be left out for a model with one
        output.
    column
        The column with the text to score. It can be left out if the step has one column.
    function_to_apply
        How the model's outputs are turned into scores: `"sigmoid"`, `"softmax"`, or `"none"`
        (the raw outputs, e.g., for a regression model). By default, the pipeline chooses.
    device
        The device that the model runs on (e.g., `"cuda"`, `"mps"`, or `-1` for the CPU), when
        the model is given by name.
    name
        A name for the scorer, for the score cache and the report. By default, it's made from the
        model and the other settings.
    version
        A version (e.g., a revision of the model), which is part of the key of the score cache.
    batch_size
        The number of texts classified at a time (`32` by default).

    Returns
    -------
    Scorer
        A scorer for `Validate.col_vals_scored()`.

    Examples
    --------
    This checks the educational quality of the documents of a pretraining dataset with the
    FineWeb-Edu classifier, a regression model that scores text from `0` to `5`.

    ```python
    import pointblank as pb

    validation = (
        pb.Validate(data=documents)
        .col_vals_scored(
            columns="text",
            scorer=pb.scorers.classifier(
                hf_model="HuggingFaceFW/fineweb-edu-classifier", function_to_apply="none"
            ),
            min_val=2.5,
            thresholds=(0.3, 0.5),
        )
        .interrogate()
    )
    ```

    For a classifier with several labels, like a toxicity model, `label=` chooses the label to
    score and `max_val=` would keep the rows that score low for it.
    """
    return _Classifier(
        hf_model, label, column, function_to_apply, device, name, version, batch_size
    )


def _lang_matches(lang: str, expected: list[str]) -> bool:
    """Check a language code against the expected ones (`"en"` matches `"en"`, `"en_US"`, ...)."""
    lang = lang.lower()
    base = lang.replace("-", "_").split("_")[0]
    return any(lang == code or base == code for code in expected)


class _LangId(_Adapter):
    _kind = "langid"
    _label = "LANGID"

    def __init__(
        self, expected, backend, model, column, expected_column, name, version, batch_size
    ):
        if (expected is None) == (expected_column is None):
            raise ValueError("langid() needs one of `expected=` and `expected_column=`.")
        if expected_column is not None and (
            not isinstance(expected_column, str) or not expected_column
        ):
            raise TypeError("`expected_column=` must be the name of a column.")
        codes = [] if expected is None else [expected] if isinstance(expected, str) else expected
        if expected is not None and (not isinstance(codes, (list, tuple)) or not codes):
            raise TypeError("`expected=` must be a language code or a list of them.")
        if not all(isinstance(c, str) and c for c in codes):
            raise TypeError("`expected=` must be a language code or a list of them.")
        if isinstance(backend, str):
            if backend not in _LANGID_BACKENDS:
                raise ValueError(
                    f'`backend=` must be "langid", "fasttext", or a function, got {backend!r}.'
                )
            if backend == "fasttext" and not isinstance(model, (str, os.PathLike)):
                raise ValueError(
                    '`backend="fasttext"` needs the path of a fastText language identification '
                    "model (e.g., `lid.176.bin`) as `model=`."
                )
        elif not callable(backend):
            raise TypeError(
                f'`backend=` must be "langid", "fasttext", or a function, got {type(backend).__name__}.'
            )
        _check_column(column)
        self._expected = [c.lower() for c in codes]
        self._expected_column = expected_column
        self._backend = backend
        self._model = model
        self._column = column
        self._rank = None
        params: dict[str, Any] = (
            {"expected": expected} if expected is not None else {"expected_column": expected_column}
        )
        if backend != "langid":
            params["backend"] = backend
        if model is not None:
            params["model"] = str(model) if isinstance(model, os.PathLike) else model
        if column is not None:
            params["column"] = column
        settings = {
            "expected": sorted(self._expected) or None,
            "expected_column": expected_column,
            "backend": backend,
            "model": params.get("model"),
            "column": column,
        }
        super().__init__(params, settings, name, version, batch_size)

    def _text_and_expected(self, record: dict[str, Any]) -> tuple[str, list[str]]:
        """Get the text of a row, and the languages that it's expected to be in."""
        if self._expected_column is None:
            return _text_of(record, self._column, self._kind), self._expected
        if self._expected_column not in record:
            raise ValueError(
                f"langid() reads the expected language from the column "
                f"{self._expected_column!r} (`expected_column=`), which isn't one of the columns "
                f"of the step ({', '.join(map(repr, record))}): add it to `columns=`."
            )
        others = {k: v for k, v in record.items() if k != self._expected_column}
        text = _text_of(others if self._column is None else record, self._column, self._kind)
        codes = record[self._expected_column]
        codes = [codes] if isinstance(codes, str) else list(codes)
        return text, [str(c).lower() for c in codes]

    def _get_rank(self):
        if self._rank is None:
            if self._backend == "langid":
                try:
                    from langid.langid import LanguageIdentifier
                    from langid.langid import model as langid_model
                except ImportError as e:
                    raise _install_hint("langid()", "langid", e) from e
                identifier = LanguageIdentifier.from_modelstring(langid_model, norm_probs=True)
                self._rank = identifier.rank
            elif self._backend == "fasttext":
                try:
                    import fasttext
                except ImportError as e:
                    raise _install_hint('langid(backend="fasttext")', "fasttext", e) from e
                ft_model = fasttext.load_model(str(self._model))

                def rank(text):
                    labels, probs = ft_model.predict(text.replace("\n", " "), k=-1)
                    return [
                        (label.removeprefix("__label__"), float(p))
                        for label, p in zip(labels, probs)
                    ]

                self._rank = rank
            else:
                self._rank = self._backend
        return self._rank

    def __call__(self, batch):
        rank = self._get_rank()
        scores = []
        for record in batch:
            text, expected = self._text_and_expected(record)
            ranking = rank(text)
            pairs = ranking.items() if isinstance(ranking, Mapping) else ranking
            scores.append(sum(float(p) for lang, p in pairs if _lang_matches(str(lang), expected)))
        return scores


def langid(
    expected: str | list[str] | None = None,
    backend: str | Callable[[str], Any] = "langid",
    model: str | None = None,
    column: str | None = None,
    expected_column: str | None = None,
    name: str | None = None,
    version: str | None = None,
    batch_size: int | None = None,
) -> Scorer:
    """
    Score text by how likely it is to be in the expected languages.

    The score of a row is the probability, according to a language identification model, that
    its text is in one of the expected languages. It's a check that a corpus or a column of
    free text is in the languages it should be, with `min_val=` as the confidence needed (e.g.,
    `0.5`).

    Parameters
    ----------
    expected
        The expected language (an ISO 639 code like `"en"`), or a list of them. A code matches
        the model's languages with the same code or with a region or script after it (`"en"`
        matches `"en_US"`). Give this or `expected_column=`.
    backend
        The language identification model: `"langid"` (the default, which needs the `langid`
        package), `"fasttext"` (a fastText model, which needs the `fasttext` package and the path
        of the model as `model=`), or a function that takes a text and returns its probabilities
        by language (a dictionary or a list of `(language, probability)` pairs).
    model
        The path of the fastText model (e.g., `lid.176.bin`), with `backend="fasttext"`.
    column
        The column with the text to score. It can be left out if the step has one column (or two,
        with one of them as `expected_column=`).
    expected_column
        A column with the expected language of each row, instead of `expected=` (for a
        multilingual corpus, where each document is labelled with its language). The column must
        be one of the columns of the step.
    name
        A name for the scorer, for the score cache and the report. By default, it's made from the
        expected languages (or `expected_column=`), the backend, and the model.
    version
        A version, which is part of the key of the score cache.
    batch_size
        The number of texts scored at a time (`256` by default).

    Returns
    -------
    Scorer
        A scorer for `Validate.col_vals_scored()`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    A real check would use the `langid` package or a fastText model. To keep this example
    self-contained, the backend is a toy function that guesses the language from a few words.

    ```{python}
    import pointblank as pb
    import polars as pl

    def guess(text):
        words = set(text.lower().split())
        if words & {"the", "and", "is"}:
            return {"en": 0.9, "fr": 0.1}
        if words & {"le", "et", "est"}:
            return {"en": 0.1, "fr": 0.9}
        return {"en": 0.5, "fr": 0.5}

    tbl = pl.DataFrame(
        {"review": ["the soup is good", "le pain est bon", "and the bread", "ok"]}
    )

    pb.Validate(data=tbl).col_vals_scored(
        columns="review", scorer=pb.scorers.langid(expected="en", backend=guess), min_val=0.6
    ).interrogate()
    ```

    The review in French fails, and so does `"ok"`, which can't be told apart. In a corpus with
    documents in several languages, the expected language of each row can come from a column,
    with `expected_column=`:

    ```{python}
    tbl = pl.DataFrame(
        {
            "review": ["the soup is good", "le pain est bon", "le vin est bon"],
            "lang": ["en", "fr", "en"],
        }
    )

    pb.Validate(data=tbl).col_vals_scored(
        columns=["review", "lang"],
        scorer=pb.scorers.langid(expected_column="lang", backend=guess),
        min_val=0.6,
    ).interrogate()
    ```

    The third review is labelled as English, but it's in French, so it fails.
    """
    return _LangId(expected, backend, model, column, expected_column, name, version, batch_size)


class _Perplexity(_Adapter):
    _kind = "perplexity"
    _label = "PERPLEXITY"

    def __init__(self, fn, kenlm_model, column, per_row, name, version, batch_size):
        if (fn is None) == (kenlm_model is None):
            raise ValueError("perplexity() needs one of `fn=` and `kenlm_model=`.")
        if fn is not None and not callable(fn):
            raise TypeError(f"`fn=` must be a function, got {type(fn).__name__}.")
        if kenlm_model is not None and not isinstance(kenlm_model, (str, os.PathLike)):
            raise TypeError("`kenlm_model=` must be the path of a KenLM model.")
        if not isinstance(per_row, bool):
            raise TypeError("`per_row=` must be `True` or `False`.")
        _check_column(column)
        self._fn = fn
        self._kenlm_path = None if kenlm_model is None else str(kenlm_model)
        self._column = column
        self._per_row = per_row
        self._kenlm = None
        params: dict[str, Any] = {}
        if fn is not None:
            params["fn"] = fn
        else:
            params["kenlm_model"] = self._kenlm_path
        if column is not None:
            params["column"] = column
        if not per_row:
            params["per_row"] = False
        settings = {
            "fn": fn,
            "kenlm_model": self._kenlm_path,
            "column": column,
        }
        super().__init__(params, settings, name, version, batch_size)

    def __call__(self, batch):
        texts = [_text_of(record, self._column, self._kind) for record in batch]
        if self._kenlm_path is not None:
            if self._kenlm is None:
                try:
                    import kenlm
                except ImportError as e:
                    raise _install_hint("perplexity(kenlm_model=)", "kenlm", e) from e
                self._kenlm = kenlm.Model(self._kenlm_path)
            return [self._kenlm.perplexity(text) for text in texts]
        if self._per_row:
            return [self._fn(text) for text in texts]
        return self._fn(texts)


def perplexity(
    fn: Callable[..., Any] | None = None,
    kenlm_model: str | None = None,
    column: str | None = None,
    per_row: bool = True,
    name: str | None = None,
    version: str | None = None,
    batch_size: int | None = None,
) -> Scorer:
    """
    Score text by its perplexity under a language model.

    Perplexity measures how surprising a text is to a language model: it's low for fluent text
    like the text the model was trained on, and high for garbled text, boilerplate in another
    language, or encoding errors. A check of perplexity keeps the text that's fluent, with
    `max_val=` as the highest perplexity allowed. The model is a function given as `fn=` (e.g.,
    one that runs a small causal language model) or a [KenLM](https://github.com/kpu/kenlm)
    model, which needs the `kenlm` package.

    Parameters
    ----------
    fn
        A function that gives the perplexity of a text (or with `per_row=False`, of each of a
        list of texts).
    kenlm_model
        The path of a KenLM model (an `.arpa` or `.bin` file), instead of `fn=`.
    column
        The column with the text to score. It can be left out if the step has one column.
    per_row
        Whether `fn=` takes one text (`True`, the default) or a list of texts (`False`).
    name
        A name for the scorer, for the score cache and the report. By default, it's made from the
        function (if it's defined at the top level of a module) or the path of the KenLM model.
    version
        A version (e.g., of the model), which is part of the key of the score cache.
    batch_size
        The number of texts scored at a time (`256` by default).

    Returns
    -------
    Scorer
        A scorer for `Validate.col_vals_scored()`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    A real model would be a language model. To keep this example self-contained, the
    "perplexity" of a text here is higher when it has fewer vowels, as garbled text often does.

    ```{python}
    import pointblank as pb
    import polars as pl

    def toy_perplexity(text):
        letters = [c for c in text.lower() if c.isalpha()]
        vowels = sum(c in "aeiou" for c in letters)
        return len(letters) / max(vowels, 1)

    tbl = pl.DataFrame(
        {"text": ["a plain sentence", "xkcd qwrt zzpf", "another fine line", "ok then"]}
    )

    pb.Validate(data=tbl).col_vals_scored(
        columns="text", scorer=pb.scorers.perplexity(fn=toy_perplexity), max_val=4
    ).interrogate()
    ```

    The garbled text has the highest score, above `max_val=4`, so it's the row that fails.
    """
    return _Perplexity(fn, kenlm_model, column, per_row, name, version, batch_size)


class _TextStat(_Adapter):
    _kind = "text_stat"
    _default_batch_size = 1024

    def __init__(self, stat, column, stop_words, name, version, batch_size):
        from pointblank._text import STATS, STOP_WORDS

        if stat not in STATS:
            raise ValueError(f"`stat=` must be one of {', '.join(map(repr, STATS))}, got {stat!r}.")
        if stop_words is not None:
            if isinstance(stop_words, str) or not isinstance(stop_words, (list, tuple)):
                raise TypeError("`stop_words=` must be a list of words.")
            if not stop_words or not all(isinstance(w, str) and w for w in stop_words):
                raise TypeError("`stop_words=` must be a list of words.")
            if stat != "stop_words":
                raise ValueError('`stop_words=` can only be used with `stat="stop_words"`.')
        _check_column(column)
        self._stat = stat
        self._column = column
        self._stop_words = STOP_WORDS if stop_words is None else tuple(stop_words)
        params: dict[str, Any] = {"stat": stat}
        if column is not None:
            params["column"] = column
        if stop_words is not None:
            params["stop_words"] = list(stop_words)
        settings = {"stat": stat, "column": column, "stop_words": params.get("stop_words")}
        super().__init__(params, settings, name, version, batch_size)
        self._label = stat.upper().replace("_", " ")

    def __call__(self, batch):
        from pointblank._text import text_stat

        return [
            text_stat(_text_of(record, self._column, self._kind), self._stat, self._stop_words)
            for record in batch
        ]


def text_stat(
    stat: str,
    column: str | None = None,
    stop_words: list[str] | None = None,
    name: str | None = None,
    version: str | None = None,
    batch_size: int | None = None,
) -> Scorer:
    """
    Score text by a simple statistic, like its number of words or its share of bullet lines.

    These are the statistics of the quality filters that are used to clean web text for language
    models (those of the Gopher models, by Rae et al., 2021). They need no model, so they're fast,
    and they catch much of the text that isn't prose: lists of links, menus, tables of numbers,
    and boilerplate. `~~pointblank.Validate.col_vals_text_quality()` adds a step for each of
    them, with the bounds used for Gopher.

    Words are the text split on whitespace, and lines are the text split on newlines (blank
    lines are left out). The statistics are:

    - `"char_count"`: the number of characters
    - `"word_count"`: the number of words
    - `"mean_word_length"`: the mean number of characters of the words
    - `"symbol_ratio"`: the number of hash symbols (`#`) and ellipses for each word
    - `"bullet_lines"`: the share of lines that start with a bullet point
    - `"ellipsis_lines"`: the share of lines that end with an ellipsis
    - `"alpha_words"`: the share of words with at least one letter
    - `"stop_words"`: the number of distinct stop words in the text
    - `"repeated_lines"`: the share of lines that are the same as an earlier line

    A text without words has no score for the statistics that are shares of its words (and
    without lines, for those that are shares of its lines).

    Parameters
    ----------
    stat
        The statistic, one of those above.
    column
        The column with the text to score. It can be left out if the step has one column.
    stop_words
        The stop words, with `stat="stop_words"`. By default, they're the eight English words
        of the Gopher filters (`the`, `be`, `to`, `of`, `and`, `that`, `have`, and `with`).
    name
        A name for the scorer, for the score cache and the report. By default, it's made from the
        statistic and its settings.
    version
        A version, which is part of the key of the score cache.
    batch_size
        The number of texts scored at a time (`1024` by default).

    Returns
    -------
    Scorer
        A scorer for `Validate.col_vals_scored()`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's check that each document of a small corpus has at least eight words, and that no more
    than half of its lines are bullet points.

    ```{python}
    import pointblank as pb
    import polars as pl

    tbl = pl.DataFrame(
        {
            "text": [
                "The museum opens at nine and closes at five on weekdays.",
                "Home | About | Contact",
                "- Bread\\n- Milk\\n- Eggs\\nA short list of what to buy for the weekend.",
            ]
        }
    )

    (
        pb.Validate(data=tbl)
        .col_vals_scored(columns="text", scorer=pb.scorers.text_stat("word_count"), min_val=8)
        .col_vals_scored(
            columns="text", scorer=pb.scorers.text_stat("bullet_lines"), max_val=0.5
        )
        .interrogate()
    )
    ```

    The menu in the second row has only five words, so it fails step 1, and three of the four
    lines of the shopping list are bullet points, so it fails step 2.
    """
    return _TextStat(stat, column, stop_words, name, version, batch_size)


def _count_tokens(output: Any) -> int:
    """The number of tokens in the output of a tokenizer (a count, a list, or an encoding)."""
    if isinstance(output, Mapping) and "input_ids" in output:
        output = output["input_ids"]
    for attribute in ("ids", "input_ids"):
        if hasattr(output, attribute) and not isinstance(output, (list, tuple)):
            output = getattr(output, attribute)
    if isinstance(output, bool):
        raise TypeError("the tokenizer returned a boolean")
    if isinstance(output, int):
        return output
    if hasattr(output, "tolist"):
        output = output.tolist()
    return len(output)


class _TokenCount(_Adapter):
    _kind = "token_count"
    _label = "TOKENS"
    _default_batch_size = 512

    def __init__(self, tokenizer, column, name, version, batch_size):
        if isinstance(tokenizer, str):
            if not tokenizer or tokenizer == "tiktoken:":
                raise ValueError("`tokenizer=` must name a tokenizer.")
        elif not (callable(tokenizer) or callable(getattr(tokenizer, "encode", None))):
            raise TypeError(
                "`tokenizer=` must be the name of a tokenizer, an object with an `encode()` "
                f"method, or a function, got {type(tokenizer).__name__}."
            )
        _check_column(column)
        self._tokenizer = tokenizer
        self._column = column
        self._encode = None
        params: dict[str, Any] = {"tokenizer": tokenizer}
        if column is not None:
            params["column"] = column
        settings = {"tokenizer": tokenizer, "column": column}
        super().__init__(params, settings, name, version, batch_size)

    def _get_encode(self) -> Callable[[str], Any]:
        if self._encode is not None:
            return self._encode
        tokenizer = self._tokenizer
        if tokenizer == "whitespace":
            self._encode = str.split
        elif isinstance(tokenizer, str) and tokenizer.startswith("tiktoken:"):
            try:
                import tiktoken
            except ImportError as e:
                raise _install_hint(f"token_count({tokenizer!r})", "tiktoken", e) from e
            encoding = tokenizer.removeprefix("tiktoken:")
            try:
                enc = tiktoken.get_encoding(encoding)
            except ValueError:
                # (the name of a model, like `gpt-4o`, rather than of an encoding)
                enc = tiktoken.encoding_for_model(encoding)
            self._encode = functools.partial(enc.encode, disallowed_special=())
        elif isinstance(tokenizer, str):
            try:
                from transformers import AutoTokenizer
            except ImportError as e:
                raise _install_hint(f"token_count({tokenizer!r})", "transformers", e) from e
            hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer)
            self._encode = functools.partial(hf_tokenizer.encode, add_special_tokens=False)
        elif callable(getattr(tokenizer, "encode", None)):
            self._encode = tokenizer.encode
        else:
            self._encode = tokenizer
        return self._encode

    def __call__(self, batch):
        encode = self._get_encode()
        return [
            float(_count_tokens(encode(_text_of(record, self._column, self._kind))))
            for record in batch
        ]


def token_count(
    tokenizer: str | Any,
    column: str | None = None,
    name: str | None = None,
    version: str | None = None,
    batch_size: int | None = None,
) -> Scorer:
    """
    Score text by its number of tokens.

    A language model reads text as tokens, and most have a limit on how many they can take, so a
    corpus for training (or a set of prompts) usually needs texts with neither too few tokens nor
    too many. The score of a row is the number of tokens of its text, with the tokenizer of a
    model, which makes `min_val=` and `max_val=` bounds on the length of each text in tokens.

    Parameters
    ----------
    tokenizer
        The tokenizer. It can be:

        - the name of a Hugging Face tokenizer (e.g., `"gpt2"` or `"bert-base-uncased"`), which
          needs the `transformers` package (special tokens aren't counted)
        - `"tiktoken:"` and the name of a `tiktoken` encoding or model (e.g.,
          `"tiktoken:cl100k_base"` or `"tiktoken:gpt-4o"`), which needs the `tiktoken` package
        - `"whitespace"`, to count the words of the text, split on whitespace
        - a tokenizer object with an `encode()` method (from `tiktoken`, `tokenizers`, or
          `transformers`)
        - a function that takes a text and returns its tokens, or the number of them
    column
        The column with the text to score. It can be left out if the step has one column.
    name
        A name for the scorer, for the score cache and the report. A tokenizer given by name
        names the scorer; a tokenizer object or a function needs a `name=` for the scores to be
        kept on disk.
    version
        A version, which is part of the key of the score cache.
    batch_size
        The number of texts scored at a time (`512` by default).

    Returns
    -------
    Scorer
        A scorer for `Validate.col_vals_scored()`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    A real check would use the tokenizer of a model, such as `"tiktoken:cl100k_base"`. To keep
    this example self-contained, the tokenizer here splits text into words and punctuation.

    ```{python}
    import re

    import pointblank as pb
    import polars as pl

    def tokenize(text):
        return re.findall(r"\\w+|[^\\w\\s]", text)

    tbl = pl.DataFrame(
        {
            "prompt": [
                "Summarize the article in two sentences.",
                "Hi",
                "Translate this paragraph into French, keeping the names as they are.",
            ]
        }
    )

    pb.Validate(data=tbl).col_vals_scored(
        columns="prompt",
        scorer=pb.scorers.token_count(tokenize),
        min_val=3,
        max_val=12,
    ).interrogate()
    ```

    The second prompt has one token and the third has thirteen, so both fail.
    """
    return _TokenCount(tokenizer, column, name, version, batch_size)


# ---------------------------------------------------------------------------------------------
# Score cache
# ---------------------------------------------------------------------------------------------

_lock = threading.Lock()
_named_scores: dict[tuple[str, str | None], dict[str, float | None]] = {}
_unnamed_scores: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def clear_cache() -> None:
    """
    Clear the scores that `Validate.col_vals_scored()` keeps in memory.

    With `cache=True` (the default of `~~pointblank.Validate.col_vals_scored()`), the scores of a
    scorer are kept in memory for the rest of the session, so that interrogating the same data again
    (as in an ablation, or after a change to another step) doesn't score it again. This clears them,
    for all the scorers. A cache on disk (`cache=` as a directory) isn't changed: delete its
    directory to clear it.

    Returns
    -------
    None
        Nothing; the scores in memory are deleted.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    ```

    After this, the next interrogation of a scored step scores its rows again.

    ```{python}
    import pointblank as pb

    pb.scorers.clear_cache()
    ```

    This is useful when a scorer's model has changed without a change to its name or version.
    """
    with _lock:
        _named_scores.clear()
        _unnamed_scores.clear()


def scorer_identity(scorer: Any) -> tuple[str | None, str | None]:
    """Get the name and the version of a scorer (the name is `None` if it has no stable name)."""
    name = getattr(scorer, "name", None)
    if name is None and not isinstance(scorer, _Adapter):
        name = _stable_name(scorer)
    version = getattr(scorer, "version", None)
    return (
        None if name is None else str(name),
        None if version is None else str(version),
    )


def scorer_label(scorer: Any) -> str:
    """Get a short label of a scorer for the report."""
    if isinstance(scorer, _Adapter) and scorer._label:
        return scorer._label
    name = getattr(scorer, "name", None)
    if isinstance(name, str) and name:
        return name.rsplit(".", 1)[-1]
    return getattr(scorer, "__name__", None) or type(scorer).__name__


def scorer_batch_size(scorer: Any) -> int:
    size = getattr(scorer, "batch_size", None)
    if isinstance(size, int) and not isinstance(size, bool) and size >= 1:
        return size
    return _DEFAULT_BATCH_SIZE


def check_cache(scorer: Any, cache: Any) -> None:
    """Check the `cache=` of a scored step (a directory needs a scorer with a stable name)."""
    if isinstance(cache, bool):
        return
    if not isinstance(cache, (str, os.PathLike)):
        raise TypeError(
            "`cache=` must be `True`, `False`, or the path of a directory for the cache, got "
            f"{type(cache).__name__}."
        )
    name, _ = scorer_identity(scorer)
    if name is None:
        raise ValueError(
            "A score cache on disk needs a scorer with a stable name, so that its scores can be "
            "found again in another session; give it one with "
            "`pb.scorers.from_function(fn, name=...)` (or the `name=` of an adapter)."
        )


class _MemoryStore:
    def __init__(self, entries: dict[str, float | None]):
        self._entries = entries

    def get_many(self, keys: list[str]) -> dict[str, float | None]:
        entries = self._entries
        return {k: entries[k] for k in keys if k in entries}

    def put_many(self, items: dict[str, float | None]) -> None:
        with _lock:
            self._entries.update(items)
            excess = len(self._entries) - _MEMORY_CACHE_LIMIT
            if excess > 0:
                for key in list(itertools.islice(self._entries, excess)):
                    del self._entries[key]

    def close(self) -> None:
        pass


class _DiskStore:
    def __init__(self, directory: str | os.PathLike, name: str, version: str | None):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._name = name
        self._version = "" if version is None else version
        self._connection = sqlite3.connect(path / CACHE_FILE)
        with self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS scores (scorer TEXT NOT NULL, version TEXT NOT NULL, "
                "key TEXT NOT NULL, score REAL, PRIMARY KEY (scorer, version, key))"
            )

    def get_many(self, keys: list[str]) -> dict[str, float | None]:
        found: dict[str, float | None] = {}
        for start in range(0, len(keys), 500):
            chunk = keys[start : start + 500]
            rows = self._connection.execute(
                "SELECT key, score FROM scores WHERE scorer = ? AND version = ? AND key IN "
                f"({', '.join('?' * len(chunk))})",
                [self._name, self._version, *chunk],
            )
            found.update(rows)
        return found

    def put_many(self, items: dict[str, float | None]) -> None:
        with self._connection:
            self._connection.executemany(
                "INSERT OR REPLACE INTO scores VALUES (?, ?, ?, ?)",
                [(self._name, self._version, k, v) for k, v in items.items()],
            )

    def close(self) -> None:
        self._connection.close()


def _store(scorer: Any, cache: Any):
    if cache is False or cache is None:
        return None
    name, version = scorer_identity(scorer)
    if cache is True:
        with _lock:
            if name is not None:
                return _MemoryStore(_named_scores.setdefault((name, version), {}))
            try:
                return _MemoryStore(_unnamed_scores.setdefault(scorer, {}))
            except TypeError:
                # An object that can't be weakly referenced isn't cached
                return None
    check_cache(scorer, cache)
    return _DiskStore(cache, name, version)


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------


def python_value(value: Any) -> Any:
    """Turn a value of a row into a plain Python value, with `None` for a missing value."""
    if value is None:
        return None
    if isinstance(value, float):
        return None if math.isnan(value) else value
    kind = type(value)
    if kind.__name__ in ("NAType", "NaTType"):
        return None
    if kind.__module__.split(".")[0] == "numpy" and getattr(value, "ndim", None) == 0:
        return python_value(value.item())
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, decimal.Decimal):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(map(repr, value))
    return repr(value)


def record_key(record: dict[str, Any]) -> str:
    """Get the key of a row in the score cache: a hash of its values, by column name."""
    text = json.dumps(
        record, sort_keys=True, default=_json_default, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


class ScorerOutputError(ValueError):
    """A scorer returned something other than a score for each row."""


def _check_scores(output: Any, n: int) -> list[float | None]:
    if hasattr(output, "tolist") and not isinstance(output, (list, tuple)):
        output = output.tolist()
    if isinstance(output, (str, bytes, Mapping)):
        raise ScorerOutputError(f"it returned {type(output).__name__} instead of a list of scores")
    if not isinstance(output, Sequence):
        try:
            output = list(output)
        except TypeError:
            raise ScorerOutputError(
                f"it returned {type(output).__name__} instead of a list of scores"
            ) from None
    if len(output) != n:
        raise ScorerOutputError(f"it returned {len(output)} scores for {n} rows")
    scores: list[float | None] = []
    for value in output:
        value = python_value(value)
        if value is None:
            scores.append(None)
            continue
        if isinstance(value, (str, bytes)):
            raise ScorerOutputError(f"it returned a score that isn't a number ({value!r})")
        try:
            score = float(value)
        except (TypeError, ValueError):
            raise ScorerOutputError(
                f"it returned a score that isn't a number ({value!r})"
            ) from None
        scores.append(None if math.isnan(score) else score)
    return scores


def score_records(
    scorer: Any, records: list[dict[str, Any]], cache: Any
) -> tuple[list[float | None], dict[str, int]]:
    """
    Score rows (each a dictionary of plain values, none of them missing), with the cache.

    Rows with the same values are scored once. The scores are written to the cache after each
    batch, so an interrupted run picks up where it left off. Returns the scores, in the order of
    the rows, and the numbers of distinct rows, of those scored, and of those found in the cache.
    """
    keys = [record_key(record) for record in records]
    unique: dict[str, dict[str, Any]] = {}
    for key, record in zip(keys, records):
        unique.setdefault(key, record)

    store = _store(scorer, cache)
    try:
        found = store.get_many(list(unique)) if store is not None else {}
        n_cached = len(found)
        todo = [(k, r) for k, r in unique.items() if k not in found]
        size = scorer_batch_size(scorer)
        for start in range(0, len(todo), size):
            batch = todo[start : start + size]
            scores = _check_scores(scorer([r for _, r in batch]), len(batch))
            new = {k: s for (k, _), s in zip(batch, scores)}
            if store is not None:
                store.put_many(new)
            found.update(new)
    finally:
        if store is not None:
            store.close()
    counts = {"n_distinct": len(unique), "n_scored": len(unique) - n_cached, "n_cached": n_cached}
    return [found[k] for k in keys], counts
