# Checking RAG Data

A retrieval-augmented generation (RAG) system answers a question by searching a corpus of text for the chunks most like it and giving them to a language model with the question. Its answers can only be as good as those chunks. A chunk with no text (from a PDF whose text couldn't be extracted), the same paragraph in many chunks, a chunk without the source that the answer should cite, or a document that's years out of date all make the answers worse, and none of them are errors that the system reports.

The evaluation set of a RAG system (questions with the documents that answer them) has problems of its own: a question asked twice counts twice in the scores, and a question whose document was removed from the corpus can never be answered. The `pb.rag` functions make validation steps for both, so they're checked with the same reports, thresholds, and actions as any other table.

## The Corpus

Let's make a small corpus of chunks of help articles. Each chunk has an id, its text, the article it's from, the groups of users that can see it (its access control list), and the date the article was last updated.

``` python
import datetime

import polars as pl

chunks = pl.DataFrame(
    {
        "chunk_id": ["acct-1", "acct-2", "bill-1", "bill-2", "bill-2", "ship-1", "hol-1"],
        "text": [
            "To reset your password, open Settings, choose Security, and select Reset password.",
            "Two-step verification can be turned on from the Security page of your Settings.",
            "Invoices can be downloaded as PDF files from the Billing page of your account.",
            "Invoices can be downloaded as PDF files from the Billing page of your account!",
            "Refunds are paid to the card used for the order within five working days.",
            " ",
            "Our offices are closed on public holidays.",
        ],
        "source": [
            "help/account",
            "help/account",
            "help/billing",
            "help/billing",
            "help/billing",
            "help/shipping",
            None,
        ],
        "acl": ["all", "all", "finance", "finance", "finance", "all", None],
        "updated_at": [datetime.date(2026, 9, 14)] * 6 + [datetime.date(2022, 12, 1)],
    }
)

pb.preview(chunks)
```

| PolarsRows7Columns5 |  |  |  |  |  |
|----|----|----|----|----|----|
|  | chunk_id *String* | text *String* | source *String* | acl *String* | updated_at *Date* |
| 1 | acct-1 | To reset your password, open Settings, choose Security, and select Reset password. | help/account | all | 2026-09-14 |
| 2 | acct-2 | Two-step verification can be turned on from the Security page of your Settings. | help/account | all | 2026-09-14 |
| 3 | bill-1 | Invoices can be downloaded as PDF files from the Billing page of your account. | help/billing | finance | 2026-09-14 |
| 4 | bill-2 | Invoices can be downloaded as PDF files from the Billing page of your account! | help/billing | finance | 2026-09-14 |
| 5 | bill-2 | Refunds are paid to the card used for the order within five working days. | help/billing | finance | 2026-09-14 |
| 6 | ship-1 |  | help/shipping | all | 2026-09-14 |
| 7 | hol-1 | Our offices are closed on public holidays. | None | None | 2022-12-01 |

## Checking the Corpus

The [`pb.rag.corpus_checks()`](../../reference/rag.corpus_checks.md#pointblank.rag.corpus_checks) function makes a step for each check of the chunks. It returns a [`Steps`](../../reference/Steps.md#pointblank.Steps) collection, which is added to a validation with [add_steps()](../../reference/Validate.add_steps.md#pointblank.Validate.add_steps). We'll give it the column of the text and the other columns it needs for each check: the ids, the metadata that every chunk must have, and the dates of update (with how old a chunk can be). We'll also ask for chunks of at least five words.

``` python
validation = (
    pb.Validate(data=chunks, tbl_name="chunks")
    .add_steps(
        pb.rag.corpus_checks(
            "text",
            chunk_id="chunk_id",
            metadata=["source", "acl"],
            min_tokens=5,
            updated_at="updated_at",
            max_age="104 weeks",
            reference_time=datetime.date(2026, 11, 1),
        )
    )
    .interrogate()
)

validation
```

[TABLE]

The steps check, in order (and the brief of each says what it checks):

- **Text.** Every chunk has text: it isn't missing, and it isn't only whitespace. The chunk of the shipping article is a single space.
- **Unique ids.** No two chunks have the same id. The id `bill-2` is used for two different chunks, so a citation of `bill-2` could mean either.
- **Metadata.** Every chunk has a value of each column of `metadata=`, with a step for each. The chunk about holidays has no source to cite and no access control list. A chunk without an access control list is a problem of security as much as of quality: it may be shown to anyone, or to no one, depending on how the system treats a missing value.
- **Length.** Every chunk has from `min_tokens=` to `max_tokens=` tokens. The chunk of the shipping article has none, so it fails this step too. The tokens are words by default; with `tokenizer=`, they're the tokens of a model (as in [col_vals_token_len()](../../reference/Validate.col_vals_token_len.md#pointblank.Validate.col_vals_token_len)).
- **Near duplicates.** No chunk is nearly the same as another, which would crowd the other chunks out of the results of a search. The two chunks about invoices differ only in their punctuation. This is a [rows_near_distinct()](../../reference/Validate.rows_near_distinct.md#pointblank.Validate.rows_near_distinct) step, and `near_duplicates=` sets its threshold of similarity (or turns it off).
- **Freshness.** Every chunk was updated in the `max_age=` before the reference time. The article about holidays was last updated almost four years ago.

The checks of the ids, the length, and freshness are left out unless their columns or bounds are given. By default, the reference time is the time the steps are made, so that the cutoff of the check of freshness is fixed in the steps.

## Checking the Evaluation Set

An evaluation set has questions, each with the id of the document that answers it (its *gold* document) and often the passage of that document with the answer. The [`pb.rag.eval_checks()`](../../reference/rag.eval_checks.md#pointblank.rag.eval_checks) function checks that every question has text, that no question is a near duplicate of another, that every question has a gold document, and that every gold document is in the corpus.

It can also check that each gold passage is relevant to its question, with a scorer of the two. A reranker, like [`pb.scorers.reranker()`](../../reference/scorers.reranker.md#pointblank.scorers.reranker), is made for this. To keep this page self-contained, we'll use a scorer that gives the share of the words of a passage that are in its question.

``` python
import re

questions = pl.DataFrame(
    {
        "question": [
            "How do I reset my password?",
            "Where can I download an invoice?",
            "how do I reset my password",
            "When are refunds paid?",
        ],
        "chunk_id": ["acct-1", "bill-1", "acct-1", "bill-9"],
        "passage": [
            "open Settings, choose Security, and select Reset password",
            "Invoices can be downloaded as PDF files",
            "open Settings, choose Security, and select Reset password",
            "Our offices are closed on public holidays.",
        ],
    }
)


def words(text):
    return re.findall(r"\w+", text.lower())


def word_overlap(rows):
    scores = []
    for row in rows:
        question, passage = set(words(row["question"])), words(row["passage"])
        scores.append(sum(word in question for word in passage) / len(passage))
    return scores


validation = (
    pb.Validate(data=questions, tbl_name="questions")
    .add_steps(
        pb.rag.eval_checks(
            "question",
            gold_doc_id="chunk_id",
            corpus=chunks,
            passage="passage",
            scorer=pb.scorers.from_function(word_overlap, name="word_overlap"),
            min_relevance=0.2,
        )
    )
    .interrogate()
)

validation
```

[TABLE]

The first and third questions are the same once they're in lower case and without punctuation, so both fail step 2. The last question's gold document, `bill-9`, isn't in the corpus (step 4), and its passage is about something else (step 5). The second passage fails step 5 as well, since it shares only one word with its question (*invoices* and *downloaded* aren't the words *invoice* and *download*). That shows the limits of counting words: a reranker would see that a passage about downloading invoices answers it. The corpus can be any table that [col_vals_in_table()](../../reference/Validate.col_vals_in_table.md#pointblank.Validate.col_vals_in_table) takes, and `corpus_id=` names its column of ids if it's different from the column of the evaluation set.

## Which Checks Matter?

Like the [checks of text data](text-checks.md), these are ordinary validation steps. The rows that fail can be removed with [get_sundered_data()](../../reference/Validate.get_sundered_data.md#pointblank.Validate.get_sundered_data), and an [`Ablation`](../post-interrogation/data-ablation.md) can measure how much each check matters to the quality of the answers, by evaluating the system with a corpus that has the failing chunks of each step removed. Every check runs on any backend, so a corpus in a database or a PySpark DataFrame can be checked where it is (the checks of length, near duplicates, and relevance collect the distinct values of their columns, score them in Python, and join the results back to the table).

## Conclusion

A RAG system is only as good as its corpus and its evaluation set. The `pb.rag.corpus_checks()` function checks that the chunks have text, unique ids, the metadata the system needs, a useful length, no near duplicates, and recent updates; `pb.rag.eval_checks()` checks that the questions have text and aren't repeated, that their gold documents are in the corpus, and that their gold passages are relevant to them.
