"""
Checks of the data of retrieval-augmented generation (RAG): the corpus and the evaluation set.

Each function here makes a [`Steps`](`pointblank.Steps`) collection of ordinary validation steps,
to be added to a validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`). The
steps have the same reports, thresholds, and actions as any other, and each one can be changed or
left out like any step of a validation.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Callable

from pointblank import text as _text
from pointblank.steps import Steps

if TYPE_CHECKING:
    from pointblank._typing import SegmentSpec
    from pointblank.actions import Actions
    from pointblank.thresholds import Thresholds

__all__ = ["corpus_checks", "eval_checks"]

# A value has text if it has a character that isn't whitespace (written with the features of
# regular expressions that every backend has)
_HAS_TEXT = r"[^ \t\r\n]"


def _check_column_name(column: Any, param: str) -> None:
    if not isinstance(column, str) or not column:
        raise TypeError(f"`{param}=` must be the name of a column.")


def _near_threshold(near_duplicates: Any) -> float | None:
    """The threshold of the check of near duplicates (`None` to leave the check out)."""
    if near_duplicates is False or near_duplicates is None:
        return None
    if near_duplicates is True:
        return 0.85
    if isinstance(near_duplicates, bool) or not isinstance(near_duplicates, (int, float)):
        raise TypeError("`near_duplicates=` must be `True`, `False`, or a threshold from 0 to 1.")
    if not 0 < near_duplicates <= 1:
        raise ValueError("`near_duplicates=` must be greater than 0 and at most 1.")
    return float(near_duplicates)


def _freshness_cutoff(max_age: Any, reference_time: Any) -> tuple[Any, Any]:
    """
    The reference time, and the earliest time a chunk can have been updated (the reference time
    less `max_age=`).
    """
    from pointblank.validate import _parse_max_age, _parse_reference_time

    age = _parse_max_age(max_age)
    if reference_time is None:
        # (naive, in UTC, as Pointblank treats naive datetimes)
        reference = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    elif isinstance(reference_time, str):
        reference = _parse_reference_time(reference_time)
    elif isinstance(reference_time, (datetime.datetime, datetime.date)):
        reference = reference_time
    else:
        raise TypeError(
            "`reference_time=` must be a datetime, a date, or a string in ISO 8601 format, "
            f"got {type(reference_time).__name__}."
        )
    if not isinstance(reference, datetime.datetime):
        # A date: the cutoff is the date `max_age=` before it (in whole days)
        return reference, reference - datetime.timedelta(days=age.days)
    return reference, reference - age


def _describe_freshness(max_age: Any, reference: Any) -> str:
    if isinstance(max_age, str):
        age = max_age.strip()
    else:
        from pointblank.validate import _timedelta_to_duration_str

        age = _timedelta_to_duration_str(max_age)
    if isinstance(reference, datetime.datetime):
        when = reference.strftime("%Y-%m-%d %H:%M")
    else:
        when = reference.isoformat()
    return f"Every chunk was updated in the {age} before {when}."


def corpus_checks(
    text: str,
    chunk_id: str | None = None,
    metadata: str | list[str] | None = None,
    min_tokens: int | None = None,
    max_tokens: int | None = None,
    tokenizer: str | Any = "whitespace",
    near_duplicates: bool | float = True,
    updated_at: str | None = None,
    max_age: str | datetime.timedelta | None = None,
    reference_time: datetime.datetime | datetime.date | str | None = None,
    pre: Callable | None = None,
    segments: SegmentSpec | None = None,
    thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
    actions: Actions | None = None,
    active: bool | Callable = True,
) -> Steps:
    """
    Make validation steps that check the corpus of a RAG system.

    The answers of a retrieval-augmented generation (RAG) system are only as good as the chunks
    of text it retrieves. Common problems of a corpus are chunks with no text (from a failed
    extraction of a PDF, say), chunks too short to be useful or too long for the model, the same
    text in many chunks (which crowds out the others in the results of a search), chunks without
    the metadata that the system needs (such as the source to cite, or the access control list
    that says who can see it), and documents that haven't been updated in a long time. This
    function makes a validation step for each of them:

    -  every chunk has text (not missing, and not only whitespace): a
      `~~pointblank.Validate.col_vals_regex()` step
    - the chunk ids are unique, with `chunk_id=`: a `~~pointblank.Validate.rows_distinct()` step
    -  every chunk has a value of each column of `metadata=`: a
      `~~pointblank.Validate.col_vals_not_null()` step for each
    - the chunks have between `min_tokens=` and `max_tokens=` tokens, with either: a step of
      [`pb.text.token_length()`](`pointblank.text.token_length`)
    - no chunk is a near duplicate of another, unless `near_duplicates=False`: a
      `~~pointblank.Validate.rows_near_distinct()` step
    - every chunk was updated in the `max_age=` before the reference time, with `updated_at=`: a
      `~~pointblank.Validate.col_vals_ge()` step

    Checks of the metadata of access control (like a column of the groups that can see a chunk)
    matter for security as much as for quality: a chunk without one may be shown to anyone, or
    to no one, depending on how the system treats a missing value.

    Parameters
    ----------
    text
        The column with the text of the chunks.
    chunk_id
        The column with the id of each chunk, which must be unique.
    metadata
        The column (or a list of them) that every chunk needs a value of, such as the source
        document, its URL, or the access control list.
    min_tokens, max_tokens
        The fewest and the most tokens that a chunk can have. The check is left out if both are
        `None` (the default).
    tokenizer
        The tokenizer of the check of the number of tokens, as in
        [`pb.text.token_length()`](`pointblank.text.token_length`) (by default, `"whitespace"`,
        which counts words). Use the tokenizer of the embedding model or the language model of the
        system for an accurate count.
    near_duplicates
        Whether to check for near duplicates (`True` by default), or the threshold of the
        similarity of two chunks for them to be near duplicates (the `threshold=` of
        `~~pointblank.Validate.rows_near_distinct()`, `0.85` with `True`).
    updated_at
        The column with the time each chunk (or its document) was last updated, for the check of
        freshness. It needs `max_age=`.
    max_age
        How long ago a chunk can have been updated, as a string like `"90 days"` or
        `"2 weeks"`, or a `datetime.timedelta`.
    reference_time
        The time that the age of a chunk is measured from. By default, it's the time that the
        steps are made (in UTC, without a time zone). The cutoff (the reference time less
        `max_age=`) is fixed when the steps are made, so a collection of steps kept for later has
        the same cutoff each time it's used. For a column with a time zone, give a reference time
        with one (e.g., `datetime.now(timezone.utc)`). For a column of dates, a date can be
        given, and the cutoff is then a date too.
    pre
        A function that transforms the table before each step, as in the validation methods.
    segments
        The segments of the steps, as in the validation methods.
    thresholds
        The thresholds of the steps (by default, those of the validation).
    actions
        The actions of the steps.
    active
        Whether the steps are active.

    Returns
    -------
    Steps
        The steps, to be added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's check a small corpus of chunks of help articles, each with an id, a source, and the
    groups that can see it.

    ```{python}
    import datetime

    import pointblank as pb
    import polars as pl

    chunks = pl.DataFrame(
        {
            "chunk_id": ["a-1", "a-2", "b-1", "b-1", "c-1"],
            "text": [
                "To reset your password, open Settings and choose Security.",
                "   ",
                "Invoices can be downloaded as PDF files from the Billing page.",
                "Invoices can be downloaded as PDF files from the Billing page!",
                "Our offices are closed on public holidays.",
            ],
            "source": ["help/account", "help/account", "help/billing", "help/billing", None],
            "acl": ["all", "all", "finance", "finance", None],
            "updated_at": [datetime.date(2026, 10, 1)] * 4 + [datetime.date(2023, 5, 1)],
        }
    )

    (
        pb.Validate(data=chunks, tbl_name="chunks")
        .add_steps(
            pb.rag.corpus_checks(
                "text",
                chunk_id="chunk_id",
                metadata=["source", "acl"],
                updated_at="updated_at",
                max_age="365 days",
                reference_time=datetime.date(2026, 11, 1),
            )
        )
        .interrogate()
    )
    ```

    The second chunk is only whitespace, the id `b-1` is used twice (for two chunks that are near
    duplicates, as well), and the last chunk has no source or access control list and hasn't been
    updated in over three years.
    """
    _check_column_name(text, "text")
    if chunk_id is not None:
        _check_column_name(chunk_id, "chunk_id")
    if metadata is None:
        metadata = []
    elif isinstance(metadata, str):
        metadata = [metadata]
    if not isinstance(metadata, (list, tuple)):
        raise TypeError("`metadata=` must be the name of a column or a list of them.")
    for column in metadata:
        _check_column_name(column, "metadata")
    _text._check_bound(min_tokens, "min_tokens")
    _text._check_bound(max_tokens, "max_tokens")
    threshold = _near_threshold(near_duplicates)
    if updated_at is not None:
        _check_column_name(updated_at, "updated_at")
        if max_age is None:
            raise ValueError("The check of freshness (`updated_at=`) needs `max_age=`.")
        reference, cutoff = _freshness_cutoff(max_age, reference_time)
    elif max_age is not None:
        raise ValueError("`max_age=` needs the column of the times of update (`updated_at=`).")

    common = {
        "pre": pre,
        "segments": segments,
        "thresholds": thresholds,
        "actions": actions,
        "active": active,
    }

    steps = Steps().col_vals_regex(
        columns=text, pattern=_HAS_TEXT, brief=f"Every chunk has text in `{text}`.", **common
    )
    if chunk_id is not None:
        steps.rows_distinct(
            columns_subset=chunk_id, brief=f"The chunk ids in `{chunk_id}` are unique.", **common
        )
    for column in metadata:
        steps.col_vals_not_null(
            columns=column, brief=f"Every chunk has a value of `{column}`.", **common
        )
    if min_tokens is not None or max_tokens is not None:
        if min_tokens is None:
            bounds = f"at most {max_tokens:,}"
        elif max_tokens is None:
            bounds = f"at least {min_tokens:,}"
        else:
            bounds = f"{min_tokens:,} to {max_tokens:,}"
        length = _text.token_length(
            text,
            tokenizer=tokenizer,
            min_val=min_tokens,
            max_val=max_tokens,
            brief=f"Every chunk has {bounds} tokens.",
            **common,
        )
        steps._steps.extend(length._steps)
    if threshold is not None:
        steps.rows_near_distinct(
            columns=text,
            threshold=threshold,
            brief="No chunk is a near duplicate of another.",
            **common,
        )
    if updated_at is not None:
        steps.col_vals_ge(
            columns=updated_at,
            value=cutoff,
            brief=_describe_freshness(max_age, reference),
            **common,
        )
    return steps


def eval_checks(
    query: str,
    gold_doc_id: str | None = None,
    corpus: Any = None,
    corpus_id: str | None = None,
    passage: str | None = None,
    scorer: Callable[..., Any] | None = None,
    min_relevance: float | None = None,
    near_duplicates: bool | float = True,
    pre: Callable | None = None,
    thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
    actions: Actions | None = None,
    active: bool | Callable = True,
) -> Steps:
    """
    Make validation steps that check the evaluation set of a RAG system.

    A RAG system is evaluated with a set of queries, each with the document that has its answer
    (the *gold* document) and often the passage of it. The scores of the evaluation can only be
    trusted if the set itself is sound: every query has text, no query is asked twice (a near
    duplicate counts twice in the scores), every gold document is in the corpus (a query whose
    document was removed or renamed can never be answered), and each gold passage is relevant to
    its query. This function makes a validation step for each:

    -  every query has text (not missing, and not only whitespace): a
      `~~pointblank.Validate.col_vals_regex()` step
    - no query is a near duplicate of another, unless `near_duplicates=False`: a
      `~~pointblank.Validate.rows_near_distinct()` step
    -  every query has a gold document, with `gold_doc_id=`: a
      `~~pointblank.Validate.col_vals_not_null()` step
    -  every gold document is in the corpus, with `corpus=`: a
      `~~pointblank.Validate.col_vals_in_table()` step
    - every passage is relevant to its query, with `passage=`, `scorer=`, and `min_relevance=`: a
      `~~pointblank.Validate.col_vals_scored()` step (with a reranker, like
      [`pb.scorers.reranker()`](`pointblank.scorers.reranker`), or another scorer of a query and
      a passage)

    Parameters
    ----------
    query
        The column with the queries.
    gold_doc_id
        The column with the id of the gold document of each query.
    corpus
        The corpus (any table that `~~pointblank.Validate.col_vals_in_table()` takes, such as a
        DataFrame, a database table, or the path of a file) to look up the gold documents in. It
        needs `gold_doc_id=`.
    corpus_id
        The column of the corpus with the document ids (by default, the same name as
        `gold_doc_id=`).
    passage
        The column with the gold passage of each query, for the check of relevance.
    scorer
        The scorer of the relevance of a passage to its query, given the values of `query=` and
        `passage=` (in that order).
    min_relevance
        The lowest score of relevance that a passage can have.
    near_duplicates
        Whether to check for near-duplicate queries (`True` by default), or the threshold of the
        similarity of two queries for them to be near duplicates (the `threshold=` of
        `~~pointblank.Validate.rows_near_distinct()`, `0.85` with `True`).
    pre
        A function that transforms the table before each step, as in the validation methods.
    thresholds
        The thresholds of the steps (by default, those of the validation).
    actions
        The actions of the steps.
    active
        Whether the steps are active.

    Returns
    -------
    Steps
        The steps, to be added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`).

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's check an evaluation set of three queries against a corpus of three documents.

    ```{python}
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

    The first and third queries are the same once normalized, so both fail step 2, and the second
    query's gold document (`invoices`) isn't in the corpus. A check of relevance needs a scorer,
    such as `scorer=pb.scorers.reranker(provider="cohere")` with `passage=` and
    `min_relevance=`.
    """
    _check_column_name(query, "query")
    if gold_doc_id is not None:
        _check_column_name(gold_doc_id, "gold_doc_id")
    if corpus is not None and gold_doc_id is None:
        raise ValueError("The check of the gold documents (`corpus=`) needs `gold_doc_id=`.")
    if corpus_id is not None:
        _check_column_name(corpus_id, "corpus_id")
        if corpus is None:
            raise ValueError("`corpus_id=` needs the corpus (`corpus=`).")
    relevance = [passage, scorer, min_relevance]
    if any(value is not None for value in relevance):
        if passage is None or scorer is None or min_relevance is None:
            raise ValueError(
                "The check of relevance needs `passage=`, `scorer=`, and `min_relevance=`."
            )
        _check_column_name(passage, "passage")
        if not callable(scorer):
            raise TypeError("`scorer=` must be a scorer (a callable).")
        if isinstance(min_relevance, bool) or not isinstance(min_relevance, (int, float)):
            raise TypeError("`min_relevance=` must be a number.")
    threshold = _near_threshold(near_duplicates)

    common = {"pre": pre, "thresholds": thresholds, "actions": actions, "active": active}

    steps = Steps().col_vals_regex(
        columns=query, pattern=_HAS_TEXT, brief=f"Every query has text in `{query}`.", **common
    )
    if threshold is not None:
        steps.rows_near_distinct(
            columns=query,
            threshold=threshold,
            brief="No query is a near duplicate of another.",
            **common,
        )
    if gold_doc_id is not None:
        steps.col_vals_not_null(
            columns=gold_doc_id,
            brief=f"Every query has a gold document in `{gold_doc_id}`.",
            **common,
        )
    if corpus is not None:
        steps.col_vals_in_table(
            columns=gold_doc_id,
            ref_table=corpus,
            ref_column=corpus_id or gold_doc_id,
            brief="Every gold document is in the corpus.",
            **common,
        )
    if passage is not None:
        steps.col_vals_scored(
            columns=[query, passage],
            scorer=scorer,
            min_val=min_relevance,
            brief=f"Every passage in `{passage}` is relevant to its query.",
            **common,
        )
    return steps
