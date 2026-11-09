# rag.corpus_checks()

Make validation steps that check the corpus of a RAG system.

Usage

[Source](https://github.com/posit-dev/pointblank/blob/main/pointblank/rag.py#L88-L308)

``` python
rag.corpus_checks(
    text,
    chunk_id=None,
    metadata=None,
    min_tokens=None,
    max_tokens=None,
    tokenizer="whitespace",
    near_duplicates=True,
    updated_at=None,
    max_age=None,
    reference_time=None,
    pre=None,
    segments=None,
    thresholds=None,
    actions=None,
    active=True,
)
```

The answers of a retrieval-augmented generation (RAG) system are only as good as the chunks of text it retrieves. Common problems of a corpus are chunks with no text (from a failed extraction of a PDF, say), chunks too short to be useful or too long for the model, the same text in many chunks (which crowds out the others in the results of a search), chunks without the metadata that the system needs (such as the source to cite, or the access control list that says who can see it), and documents that haven't been updated in a long time. This function makes a validation step for each of them:

- every chunk has text (not missing, and not only whitespace): a [col_vals_regex()](../reference/Validate.col_vals_regex.md#pointblank.Validate.col_vals_regex) step
- the chunk ids are unique, with `chunk_id=`: a [rows_distinct()](../reference/Validate.rows_distinct.md#pointblank.Validate.rows_distinct) step
- every chunk has a value of each column of `metadata=`: a [col_vals_not_null()](../reference/Validate.col_vals_not_null.md#pointblank.Validate.col_vals_not_null) step for each
- the chunks have between `min_tokens=` and `max_tokens=` tokens, with either: a step of [`pb.text.token_length()`](../reference/text.token_length.md#pointblank.text.token_length)
- no chunk is a near duplicate of another, unless `near_duplicates=False`: a [rows_near_distinct()](../reference/Validate.rows_near_distinct.md#pointblank.Validate.rows_near_distinct) step
- every chunk was updated in the `max_age=` before the reference time, with `updated_at=`: a [col_vals_ge()](../reference/Validate.col_vals_ge.md#pointblank.Validate.col_vals_ge) step

Checks of the metadata of access control (like a column of the groups that can see a chunk) matter for security as much as for quality: a chunk without one may be shown to anyone, or to no one, depending on how the system treats a missing value.

## Parameters

`text: str`  
The column with the text of the chunks.

`chunk_id: str | None = None`  
The column with the id of each chunk, which must be unique.

`metadata: str | list[str] | None = None`  
The column (or a list of them) that every chunk needs a value of, such as the source document, its URL, or the access control list.

`min_tokens: int | None = None`  
The fewest and the most tokens that a chunk can have. The check is left out if both are `None` (the default).

`max_tokens: int | None = None`  
The fewest and the most tokens that a chunk can have. The check is left out if both are `None` (the default).

`tokenizer: str | Any = "whitespace"`  
The tokenizer of the check of the number of tokens, as in [`pb.text.token_length()`](../reference/text.token_length.md#pointblank.text.token_length) (by default, `"whitespace"`, which counts words). Use the tokenizer of the embedding model or the language model of the system for an accurate count.

`near_duplicates: bool | float = True`  
Whether to check for near duplicates (`True` by default), or the threshold of the similarity of two chunks for them to be near duplicates (the `threshold=` of [rows_near_distinct()](../reference/Validate.rows_near_distinct.md#pointblank.Validate.rows_near_distinct), `0.85` with `True`).

`updated_at: str | None = None`  
The column with the time each chunk (or its document) was last updated, for the check of freshness. It needs `max_age=`.

`max_age: str | datetime.timedelta | None = None`  
How long ago a chunk can have been updated, as a string like `"90 days"` or `"2 weeks"`, or a `datetime.timedelta`.

`reference_time: datetime.datetime | datetime.date | str | None = None`  
The time that the age of a chunk is measured from. By default, it's the time that the steps are made (in UTC, without a time zone). The cutoff (the reference time less `max_age=`) is fixed when the steps are made, so a collection of steps kept for later has the same cutoff each time it's used. For a column with a time zone, give a reference time with one (e.g., `datetime.now(timezone.utc)`). For a column of dates, a date can be given, and the cutoff is then a date too.

`pre: Callable | None = None`  
A function that transforms the table before each step, as in the validation methods.

`segments: SegmentSpec | None = None`  
The segments of the steps, as in the validation methods.

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

Let's check a small corpus of chunks of help articles, each with an id, a source, and the groups that can see it.

``` python
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

[TABLE]

The second chunk is only whitespace, the id `b-1` is used twice (for two chunks that are near duplicates, as well), and the last chunk has no source or access control list and hasn't been updated in over three years.
