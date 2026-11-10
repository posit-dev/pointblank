import datetime

import polars as pl
import pytest

import pointblank as pb
from tests.test_text_checks import _TBL_TYPES, _spark_tbl, _tbl

_CHUNKS = pl.DataFrame(
    {
        "chunk_id": ["a-1", "a-2", "b-1", "b-1", "c-1", "d-1"],
        "text": [
            "To reset your password, open Settings and choose Security.",
            " \t ",
            "Invoices can be downloaded as PDF files from the Billing page.",
            "Invoices can be downloaded as PDF files from the Billing page!",
            "Our offices are closed on public holidays.",
            None,
        ],
        "source": ["help/account", "help/account", "help/billing", "help/billing", None, "x"],
        "acl": ["all", "all", "finance", None, None, "all"],
        "updated_at": [datetime.datetime(2026, 10, 1)] * 4
        + [datetime.datetime(2023, 5, 1), datetime.datetime(2026, 10, 1)],
    }
)

_QUERIES = pl.DataFrame(
    {
        "query": [
            "How do I reset my password?",
            "Where can I download an invoice?",
            "how do I reset my password",
            "",
        ],
        "doc_id": ["a-1", "z-9", "a-1", None],
        "passage": ["Reset your password in Settings.", "Billing page.", "Settings.", "x"],
    }
)


def _table(tbl_type, tbl):
    if tbl_type != "pyspark" or "updated_at" not in tbl.columns:
        return _tbl(tbl_type, tbl)
    # (a PySpark DataFrame with a column of timestamps)
    spark = _spark_tbl(tbl.drop("updated_at")).sparkSession
    schema = ", ".join(f"{name} string" for name in tbl.columns if name != "updated_at")
    return spark.createDataFrame(tbl.rows(), schema=schema + ", updated_at timestamp")


def _failed(validation):
    return [step.n_failed for step in validation.validation_info]


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_corpus_checks(tbl_type):
    steps = pb.rag.corpus_checks(
        "text",
        chunk_id="chunk_id",
        metadata=["source", "acl"],
        min_tokens=8,
        updated_at="updated_at",
        max_age="365 days",
        reference_time=datetime.datetime(2026, 11, 1),
    )

    validation = pb.Validate(data=_table(tbl_type, _CHUNKS)).add_steps(steps).interrogate()

    assert [step.assertion_type for step in validation.validation_info] == [
        "col_vals_regex",
        "rows_distinct",
        "col_vals_not_null",
        "col_vals_not_null",
        "col_vals_scored",
        "rows_near_distinct",
        "col_vals_ge",
    ]
    # (whitespace and missing text fail; the near duplicates are chunks 3 and 4; the short ones
    # are chunks 2, 5, and 6 (a missing text fails)
    assert _failed(validation) == [2, 2, 1, 2, 3, 2, 1]
    assert [step.brief for step in validation.validation_info] == [
        "Every chunk has text in `text`.",
        "The chunk ids in `chunk_id` are unique.",
        "Every chunk has a value of `source`.",
        "Every chunk has a value of `acl`.",
        "Every chunk has at least 8 tokens.",
        "No chunk is a near duplicate of another.",
        "Every chunk was updated in the 365 days before 2026-11-01 00:00.",
    ]
    assert all(not step.eval_error for step in validation.validation_info)


def test_corpus_checks_defaults_and_options():
    # By default: text, and near duplicates
    steps = pb.rag.corpus_checks("text")
    assert [step.method for step in steps._steps] == ["col_vals_regex", "rows_near_distinct"]
    assert steps._steps[1].kwargs["threshold"] == 0.85

    steps = pb.rag.corpus_checks(
        "text", near_duplicates=0.5, metadata="acl", max_tokens=10, thresholds=0.1
    )
    assert [step.method for step in steps._steps] == [
        "col_vals_regex",
        "col_vals_not_null",
        "col_vals_token_len",
        "rows_near_distinct",
    ]
    assert steps._steps[2].kwargs["brief"] == "Every chunk has at most 10 tokens."
    assert steps._steps[3].kwargs["threshold"] == 0.5
    assert all(step.kwargs["thresholds"] == 0.1 for step in steps._steps)

    both = pb.rag.corpus_checks("text", min_tokens=2, max_tokens=1000, near_duplicates=False)
    assert both._steps[1].kwargs["brief"] == "Every chunk has 2 to 1,000 tokens."
    assert len(both) == 2


def test_corpus_checks_freshness():
    # A date as the reference time gives a date as the cutoff
    steps = pb.rag.corpus_checks(
        "text",
        near_duplicates=False,
        updated_at="day",
        max_age=datetime.timedelta(days=30, hours=5),
        reference_time=datetime.date(2026, 11, 1),
    )
    assert steps._steps[1].kwargs["value"] == datetime.date(2026, 10, 2)
    assert steps._steps[1].kwargs["brief"].startswith("Every chunk was updated in the ")
    assert steps._steps[1].kwargs["brief"].endswith(" before 2026-11-01.")

    # By default, the reference time is now (in UTC, without a time zone)
    before = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
    steps = pb.rag.corpus_checks("text", updated_at="t", max_age="2 days")
    cutoff = steps._steps[-1].kwargs["value"]
    assert cutoff.tzinfo is None
    assert before - datetime.timedelta(days=2) <= cutoff

    # A reference time with a time zone, for a column with one
    aware = datetime.datetime(2026, 11, 1, tzinfo=datetime.timezone.utc)
    table = pl.DataFrame(
        {
            "text": ["a b c", "d e f"],
            "t": [aware - datetime.timedelta(days=1), aware - datetime.timedelta(days=9)],
        }
    )
    steps = pb.rag.corpus_checks(
        "text", near_duplicates=False, updated_at="t", max_age="1 week", reference_time=aware
    )
    validation = pb.Validate(data=table).add_steps(steps).interrogate()
    assert _failed(validation) == [0, 1]

    iso = pb.rag.corpus_checks(
        "text", updated_at="t", max_age="1 day", reference_time="2026-11-01T12:00:00"
    )
    assert iso._steps[-1].kwargs["value"] == datetime.datetime(2026, 10, 31, 12)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"text": 1}, TypeError, "`text=` must be the name"),
        ({"chunk_id": ""}, TypeError, "`chunk_id=` must be the name"),
        ({"metadata": 1}, TypeError, "`metadata=` must be the name of a column or a list"),
        ({"metadata": ["a", 2]}, TypeError, "`metadata=` must be the name"),
        ({"min_tokens": "5"}, TypeError, "`min_tokens=` must be a number"),
        ({"near_duplicates": "yes"}, TypeError, "`near_duplicates=` must be"),
        ({"near_duplicates": 1.5}, ValueError, "at most 1"),
        ({"updated_at": "t"}, ValueError, "needs `max_age=`"),
        ({"max_age": "1 day"}, ValueError, "needs the column of the times"),
        ({"updated_at": "t", "max_age": "1 day", "reference_time": 1}, TypeError, "datetime"),
        ({"updated_at": "t", "max_age": "soon"}, ValueError, "Invalid max_age format"),
    ],
)
def test_corpus_checks_bad_arguments(kwargs, error, message):
    kwargs = {"text": "text", **kwargs}
    with pytest.raises(error, match=message):
        pb.rag.corpus_checks(**kwargs)


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_eval_checks(tbl_type):
    corpus = _CHUNKS.select("chunk_id")

    def overlap(batch):
        # The share of the words of the passage that are in the query
        scores = []
        for row in batch:
            query = set(row["query"].lower().replace("?", "").split())
            passage = row["passage"].lower().replace(".", "").split()
            scores.append(sum(word in query for word in passage) / len(passage))
        return scores

    steps = pb.rag.eval_checks(
        "query",
        gold_doc_id="doc_id",
        corpus=corpus,
        corpus_id="chunk_id",
        passage="passage",
        scorer=overlap,
        min_relevance=0.1,
    )

    validation = pb.Validate(data=_tbl(tbl_type, _QUERIES)).add_steps(steps).interrogate()

    assert [step.assertion_type for step in validation.validation_info] == [
        "col_vals_regex",
        "rows_near_distinct",
        "col_vals_not_null",
        "col_vals_in_table",
        "col_vals_scored",
    ]
    # (the empty query fails; queries 1 and 3 are near duplicates; the last query has no gold
    # document, and 'z-9' isn't in the corpus; only the first passage shares a word with its query)
    assert _failed(validation)[:4] == [1, 2, 1, 2]
    assert validation.validation_info[4].n_passed == 1


def test_eval_checks_defaults():
    steps = pb.rag.eval_checks("query", near_duplicates=False)
    assert [step.method for step in steps._steps] == ["col_vals_regex"]

    steps = pb.rag.eval_checks("query", gold_doc_id="doc_id", corpus=_CHUNKS)
    assert steps._steps[-1].kwargs["ref_column"] == "doc_id"


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"query": None}, TypeError, "`query=` must be the name"),
        ({"corpus": _CHUNKS}, ValueError, "needs `gold_doc_id=`"),
        ({"gold_doc_id": "d", "corpus_id": "c"}, ValueError, "needs the corpus"),
        ({"passage": "p"}, ValueError, "needs `passage=`, `scorer=`, and `min_relevance=`"),
        (
            {"passage": "p", "scorer": "rerank", "min_relevance": 0.5},
            TypeError,
            "`scorer=` must be a scorer",
        ),
        (
            {"passage": "p", "scorer": len, "min_relevance": "high"},
            TypeError,
            "`min_relevance=` must be a number",
        ),
    ],
)
def test_eval_checks_bad_arguments(kwargs, error, message):
    kwargs = {"query": "query", **kwargs}
    with pytest.raises(error, match=message):
        pb.rag.eval_checks(**kwargs)
