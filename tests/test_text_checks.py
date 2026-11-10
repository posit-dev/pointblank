import json
import os

import pandas as pd
import polars as pl
import pytest

import pointblank as pb
from pointblank import _text
from pointblank._interrogation import CLUSTER_COLUMN, MATCH_COLUMN, REASON_COLUMN

_TBL_TYPES = ["polars", "lazy", "pandas", "duckdb", "sqlite", "pyspark"]

# Documents 1, 3, and 5 are the same once normalized; 2 and 6 differ by a word; 7 is missing
_DOCS = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6, 7],
        "text": [
            "The library opens at nine in the morning and closes at six in the evening.",
            "The trail to the summit is steep, so bring water and good walking shoes.",
            "The library opens at nine in the morning and closes at six in the evening!",
            "Tickets for the museum are free for children under the age of twelve.",
            "the LIBRARY opens at nine in the morning, and closes at six in the evening",
            "The trail to the summit is steep, so bring water and sturdy walking shoes.",
            None,
        ],
    }
)

_QUESTION = "What is the name of the largest moon of the planet Saturn in our solar system?"

_TRAIN = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5],
        "text": [
            "Rivers carry water from the hills down to the sea over many years.",
            f"Quiz: {_QUESTION} Answer: Titan.",
            "Bread is made of flour, water, salt, and yeast, and baked in an oven.",
            "How many bones are there in the body of an adult human being?",
            None,
        ],
    }
)

_BENCHMARK = pl.DataFrame(
    {"question": [_QUESTION, "How many bones are there in the body of an adult human being?"]}
)


def _chat(*messages) -> str:
    return json.dumps([{"role": role, "content": content} for role, content in messages])


_CHATS = pl.DataFrame(
    {
        "id": [1, 2, 3, 4, 5, 6],
        "messages": [
            _chat(("user", "Hi!"), ("assistant", "Hello!")),
            _chat(("user", "Hi!"), ("user", "Hello?")),
            _chat(("system", "Be brief."), ("user", "Capital of France?"), ("assistant", "Paris.")),
            _chat(("user", "What is 2 + 2?"), ("bot", "4")),
            "not json",
            None,
        ],
    }
)


def _spark_tbl(tbl: pl.DataFrame):
    if os.environ.get("SKIP_PYSPARK_TESTS", "").lower() in ("true", "1", "yes"):
        pytest.skip("PySpark tests disabled via SKIP_PYSPARK_TESTS environment variable")
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.appName("PointblankTextTests")
            .master("local[1]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
    except Exception:  # pragma: no cover
        pytest.skip("PySpark can't start")
    spark_types = {pl.Int64: "long", pl.Float64: "double", pl.String: "string"}
    schema = ", ".join(f"{name} {spark_types[dtype]}" for name, dtype in tbl.schema.items())
    return spark.createDataFrame(tbl.rows(), schema=schema)


def _tbl(tbl_type: str, tbl: pl.DataFrame):
    if tbl_type == "polars":
        return tbl
    if tbl_type == "lazy":
        return tbl.lazy()
    if tbl_type == "pandas":
        # (a non-default index, which the results must follow)
        return tbl.to_pandas().set_axis([10 * (i + 1) for i in range(tbl.height)])
    if tbl_type == "duckdb":
        ibis = pytest.importorskip("ibis")
        return ibis.memtable(tbl)
    if tbl_type == "sqlite":
        ibis = pytest.importorskip("ibis")
        return ibis.sqlite.connect().create_table("text_tbl", tbl.to_pandas())
    return _spark_tbl(tbl)


def _as_polars(tbl) -> pl.DataFrame:
    if isinstance(tbl, pl.DataFrame):
        return tbl
    if isinstance(tbl, pl.LazyFrame):
        return tbl.collect()
    if isinstance(tbl, pd.DataFrame):
        return pl.from_pandas(tbl)
    if hasattr(tbl, "toPandas"):
        return pl.from_pandas(tbl.toPandas())
    return tbl.to_polars()


def _failing(validation: pb.Validate, i: int = 1, extra: str | None = None):
    """The ids of the failing rows of a step (with a results column, as a dict)."""
    extract = _as_polars(validation.get_data_extracts(i=i, frame=True))
    if extra is None:
        return sorted(extract["id"].to_list())
    return dict(sorted(zip(extract["id"].to_list(), extract[extra].to_list())))


def _units(validation: pb.Validate, i: int = 1):
    step = validation.validation_info[i - 1]
    assert step.eval_error is None, validation.get_notes(i)
    return step.n, step.n_passed, step.n_failed


def _note(validation: pb.Validate, key: str, i: int = 1) -> str:
    return validation.get_note(i, key, format="text")


def _error_note(validation: pb.Validate, i: int = 1) -> str:
    step = validation.validation_info[i - 1]
    assert step.eval_error
    return " ".join(str(note["text"]) for note in (step.notes or {}).values())


# ---------------------------------------------------------------------------------------------
# The text utilities
# ---------------------------------------------------------------------------------------------


def test_normalize_and_ngrams():
    assert _text.normalize("  The LIBRARY, opens!\n at nine…") == "the library opens at nine"
    assert _text.words("Ça va? Très bien.") == ["ça", "va", "très", "bien"]
    assert _text.ngrams(["a", "b", "c"], 2) == ["a b", "b c"]
    # A text shorter than the n-gram is one n-gram
    assert _text.ngrams(["a", "b"], 5) == ["a b"]
    # (an empty text is one empty n-gram)
    assert _text.ngrams([], 5) == [""]
    assert _text.join_values(["a", 1, 2.5]) == "a\n1\n2.5"


def test_signatures_without_numpy_are_the_same(monkeypatch):
    shingles = [_text.shingle_hashes(t, 3) for t in _DOCS["text"].drop_nulls()]
    with_numpy = _text.signatures(shingles, 16)
    monkeypatch.setattr(_text, "_numpy", lambda: None)
    assert _text.signatures(shingles, 16) == with_numpy


@pytest.mark.parametrize("threshold", [0.3, 0.5, 0.85, 0.95])
@pytest.mark.parametrize("num_perm", [16, 128])
def test_lsh_params(threshold, num_perm):
    bands, rows = _text.lsh_params(threshold, num_perm)
    assert bands >= 1 and rows >= 1 and bands * rows <= num_perm


def test_near_duplicate_groups():
    texts = _DOCS["text"].drop_nulls().to_list()

    groups, n_compared = _text.near_duplicate_groups(texts, "exact", 0.85, 5, 128)
    assert groups == [0, 1, 0, 2, 0, 3]
    assert n_compared == 0

    groups, _ = _text.near_duplicate_groups(texts, "minhash", 0.85, 5, 128)
    assert groups == [0, 1, 0, 2, 0, 3]

    groups, n_compared = _text.near_duplicate_groups(texts, "minhash", 0.5, 5, 128)
    assert groups == [0, 1, 0, 2, 0, 1]
    assert n_compared >= 1


def test_near_duplicate_groups_are_connected():
    # A is like B and B is like C (but A isn't like C), so all three are one group (the
    # similarities are 0.83, 0.71, and 0.57)
    a = "one two three four five six seven eight nine ten"
    b = "one two three four five six seven eight nine ten eleven twelve"
    c = "three four five six seven eight nine ten eleven twelve thirteen fourteen"
    shingles = [_text.shingle_hashes(t, 1) for t in (a, b, c)]
    assert _text.jaccard(shingles[0], shingles[1]) >= 0.6
    assert _text.jaccard(shingles[1], shingles[2]) >= 0.6
    assert _text.jaccard(shingles[0], shingles[2]) < 0.6

    groups, _ = _text.near_duplicate_groups([a, b, c], "minhash", 0.6, 1, 256)
    assert groups == [0, 0, 0]


def test_near_duplicate_groups_never_join_pairs_below_the_threshold():
    import random

    rng = random.Random(7)
    vocabulary = [f"w{k}" for k in range(30)]
    texts = [" ".join(rng.choices(vocabulary, k=12)) for _ in range(60)]
    shingles = [_text.shingle_hashes(t, 2) for t in texts]
    groups, _ = _text.near_duplicate_groups(texts, "minhash", 0.4, 2, 64)

    # Every text of a group of two is like the other one
    for group in set(groups):
        members = [k for k, g in enumerate(groups) if g == group]
        if len(members) == 2:
            assert _text.jaccard(shingles[members[0]], shingles[members[1]]) >= 0.4


def test_reference_index_ngram():
    index = _text.ReferenceIndex([_QUESTION, "Name a prime"], "ngram", 13)

    assert index.match(f"Quiz: {_QUESTION.upper()} Answer: Titan.") == 0
    # Twelve words of the question aren't enough
    assert index.match(" ".join(_QUESTION.split()[:12])) is None
    # A reference text shorter than the n-gram is found whole in a text, and a text shorter than
    # the n-gram is the same as a reference text
    assert index.match("Please name a prime, if you can.") == 1
    assert index.match("name a PRIME") == 1
    assert index.match("name a") is None
    assert index.match("") is None


def test_reference_index_exact_and_minhash():
    index = _text.ReferenceIndex([_QUESTION], "exact", 13)
    assert index.match(_QUESTION.lower() + "!!") == 0
    assert index.match(f"Quiz: {_QUESTION}") is None

    index = _text.ReferenceIndex([_QUESTION], "minhash", 2, threshold=0.5)
    assert index.match(_QUESTION.replace("largest", "biggest")) == 0
    assert index.match("Rivers carry water from the hills down to the sea.") is None


_RULES = dict(
    roles=("system", "user", "assistant"),
    system_role="system",
    alternate=True,
    first_role=None,
    last_role=None,
    allow_empty=False,
    role_key="role",
    content_key="content",
)


@pytest.mark.parametrize(
    "value, rules, reason",
    [
        (_chat(("user", "a"), ("assistant", "b")), {}, None),
        # A list of messages, or a dictionary with one
        ([{"role": "user", "content": "a"}], {}, None),
        ({"messages": [{"role": "user", "content": "a"}]}, {}, None),
        ("not json", {}, "not_messages"),
        ('{"role": "user"}', {}, "not_messages"),
        (42, {}, "not_messages"),
        ("[]", {}, "no_messages"),
        ('[{"content": "a"}]', {}, "bad_message"),
        ('["a"]', {}, "bad_message"),
        ('[{"role": 1, "content": "a"}]', {}, "bad_message"),
        (_chat(("user", "a"), ("bot", "b")), {}, "unknown_role"),
        (_chat(("user", "a"), ("system", "b")), {}, "system_position"),
        (_chat(("assistant", "a"), ("user", "b")), {}, None),
        (_chat(("assistant", "a"), ("user", "b")), {"first_role": "user"}, "first_role"),
        (_chat(("system", "a"), ("user", "b")), {"first_role": "user"}, None),
        (_chat(("system", "a")), {"first_role": "user"}, "first_role"),
        (_chat(("user", "a"), ("user", "b")), {}, "not_alternating"),
        (_chat(("user", "a"), ("user", "b")), {"alternate": False}, None),
        (_chat(("user", "a")), {"last_role": "assistant"}, "last_role"),
        (_chat(("user", "a"), ("assistant", " ")), {}, "empty_content"),
        ('[{"role": "user"}]', {}, "empty_content"),
        ('[{"role": "user", "content": []}]', {}, "empty_content"),
        (_chat(("user", "a"), ("assistant", "")), {"allow_empty": True}, None),
        # Without a system role, "system" is an ordinary role
        (
            _chat(("user", "a"), ("system", "b")),
            {"roles": ("user", "system"), "system_role": None},
            None,
        ),
        (
            '[{"speaker": "user", "text": "a"}]',
            {"role_key": "speaker", "content_key": "text"},
            None,
        ),
    ],
)
def test_chat_reason(value, rules, reason):
    assert _text.chat_reason(value, **{**_RULES, **rules}) == reason


def test_text_stats():
    text = "The cat sat.\n- one\n- one\nAnd then...\n\n#tag"
    assert _text.text_stat(text, "char_count") == len(text)
    assert _text.text_stat(text, "word_count") == 10
    assert _text.text_stat(text, "bullet_lines") == pytest.approx(2 / 5)
    assert _text.text_stat(text, "ellipsis_lines") == pytest.approx(1 / 5)
    assert _text.text_stat(text, "repeated_lines") == pytest.approx(1 / 5)
    assert _text.text_stat(text, "symbol_ratio") == pytest.approx(2 / 10)
    assert _text.text_stat(text, "alpha_words") == pytest.approx(8 / 10)
    assert _text.text_stat(text, "stop_words") == 2
    assert _text.text_stat("", "mean_word_length") is None
    assert _text.text_stat("ab abcd", "mean_word_length") == 3


# ---------------------------------------------------------------------------------------------
# rows_near_distinct()
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_rows_near_distinct(tbl_type):
    validation = (
        pb.Validate(data=_tbl(tbl_type, _DOCS))
        .rows_near_distinct(columns="text")
        .rows_near_distinct(columns="text", threshold=0.5)
        .rows_near_distinct(columns="text", method="exact")
        .interrogate()
    )

    # The missing document passes
    assert _units(validation, 1) == (7, 4, 3)
    assert _units(validation, 2) == (7, 2, 5)
    assert _units(validation, 3) == (7, 4, 3)
    assert _failing(validation, 1, CLUSTER_COLUMN) == {1: 1, 3: 1, 5: 1}
    assert _failing(validation, 2, CLUSTER_COLUMN) == {1: 1, 2: 2, 3: 1, 5: 1, 6: 2}

    assert _note(validation, "near_duplicates", 2) == (
        "2 groups of near duplicates, with 5 rows (the largest has 3 rows); MinHash of 5-word "
        "n-grams, with a Jaccard similarity of at least 0.5. The group of each row is in the "
        "`pb_cluster_` column."
    )
    assert "normalized" in _note(validation, "near_duplicates", 3)


def test_rows_near_distinct_notes():
    validation = (
        pb.Validate(data=_DOCS)
        .rows_near_distinct(columns="text")
        .rows_near_distinct(columns="id")
        .interrogate()
    )

    assert _note(validation, "near_duplicates", 1) == (
        "1 group of near duplicates, with 3 rows; MinHash of 5-word n-grams, with a Jaccard "
        "similarity of at least 0.85. The group of each row is in the `pb_cluster_` column."
    )
    assert _note(validation, "near_duplicates", 2).startswith("No near duplicates")


def test_rows_near_distinct_counts_repeated_rows():
    # Copies of one text are near duplicates of each other (and are compared once)
    tbl = pl.DataFrame({"id": [1, 2, 3], "text": ["a b c", "x y z", "A b c."]})

    validation = pb.Validate(data=tbl).rows_near_distinct(columns="text").interrogate()

    assert _units(validation) == (3, 1, 2)
    assert _failing(validation) == [1, 3]


@pytest.mark.parametrize("tbl_type", ["polars", "pandas", "duckdb"])
def test_rows_near_distinct_several_columns(tbl_type):
    # The text of a row is its values, joined; a row with a missing value passes
    tbl = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "title": ["Opening hours", "Opening hours", "Directions", None],
            "body": ["Nine to six.", "Nine to six!", "Nine to six.", "Nine to six."],
        }
    )

    validation = (
        pb.Validate(data=_tbl(tbl_type, tbl))
        .rows_near_distinct(columns=["title", "body"], method="exact")
        .interrogate()
    )

    assert _units(validation) == (4, 2, 2)
    assert _failing(validation) == [1, 2]
    assert validation.validation_info[0].column == ["title", "body"]


@pytest.mark.parametrize("tbl_type", ["polars", "lazy", "duckdb"])
def test_rows_near_distinct_row_flags_and_sundering(tbl_type):
    validation = (
        pb.Validate(data=_tbl(tbl_type, _DOCS)).rows_near_distinct(columns="text").interrogate()
    )

    # By default, the first row of each group is kept
    flags = _as_polars(validation.get_row_flags()).sort("id")
    assert flags.filter(pl.col("pb_failed_1_"))["id"].to_list() == [3, 5]
    flags = _as_polars(validation.get_row_flags(duplicates="keep_last")).sort("id")
    assert flags.filter(pl.col("pb_failed_1_"))["id"].to_list() == [1, 3]
    flags = _as_polars(validation.get_row_flags(duplicates="flag_all")).sort("id")
    assert flags.filter(pl.col("pb_failed_1_"))["id"].to_list() == [1, 3, 5]

    passing = _as_polars(validation.get_sundered_data(type="pass", duplicates="keep_first"))
    assert sorted(passing["id"].to_list()) == [1, 2, 4, 6, 7]
    # (as for `rows_distinct()`, the step isn't used for sundering by default)
    assert _as_polars(validation.get_sundered_data(type="pass")).height == 7


def test_rows_near_distinct_with_segments():
    tbl = _DOCS.with_columns(g=pl.Series(["a", "a", "b", "b", "a", "a", "b"]))

    validation = (
        pb.Validate(data=tbl).rows_near_distinct(columns="text", segments="g").interrogate()
    )

    # Rows 1 and 5 are in segment `a`, and row 3 is alone in segment `b`
    assert _units(validation, 1) == (4, 2, 2)
    assert _units(validation, 2) == (3, 3, 0)


def test_rows_near_distinct_step_report():
    validation = (
        pb.Validate(data=_DOCS).rows_near_distinct(columns="text", threshold=0.5).interrogate()
    )

    html = validation.get_step_report(i=1).as_raw_html()
    assert "pb_cluster_" in html
    assert "max jaccard" in html

    report = validation.get_tabular_report().as_raw_html()
    assert "≥0.5" in report
    assert "MINHASH 5-GRAM" in report


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"columns": 1}, TypeError, "columns"),
        ({"columns": ["text", "text"]}, ValueError, "more than once"),
        ({"columns": "text", "method": "simhash"}, ValueError, "method"),
        ({"columns": "text", "threshold": 0}, ValueError, "threshold"),
        ({"columns": "text", "threshold": 1.5}, ValueError, "threshold"),
        ({"columns": "text", "threshold": "high"}, TypeError, "threshold"),
        ({"columns": "text", "ngram": 0}, ValueError, "ngram"),
        ({"columns": "text", "ngram": 2.5}, TypeError, "ngram"),
        ({"columns": "text", "num_perm": True}, TypeError, "num_perm"),
    ],
)
def test_rows_near_distinct_bad_arguments(kwargs, error, message):
    with pytest.raises(error, match=message):
        pb.Validate(data=_DOCS).rows_near_distinct(**kwargs)


def test_rows_near_distinct_evaluation_errors():
    validation = (
        pb.Validate(data=_DOCS.with_columns(pb_cluster_=pl.lit(1)))
        .rows_near_distinct(columns="nope")
        .rows_near_distinct(columns="text")
        .col_vals_not_null(columns="id")
        .interrogate()
    )

    assert "nope" in _error_note(validation, 1)
    assert "pb_cluster_" in _error_note(validation, 2)
    # The other steps are still interrogated
    assert validation.validation_info[2].all_passed


# ---------------------------------------------------------------------------------------------
# rows_not_contaminated()
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_rows_not_contaminated(tbl_type):
    validation = (
        pb.Validate(data=_tbl(tbl_type, _TRAIN))
        .rows_not_contaminated(columns="text", reference=_BENCHMARK, reference_columns="question")
        .rows_not_contaminated(columns="text", reference=_BENCHMARK["question"].to_list(), ngram=5)
        .rows_not_contaminated(
            columns="text", reference=_BENCHMARK, reference_columns="question", method="exact"
        )
        .rows_not_contaminated(
            columns="text",
            reference={"bench": _BENCHMARK.rename({"question": "text"}), "other": ["Bread is"]},
            method="minhash",
            threshold=0.5,
        )
        .interrogate()
    )

    # The question of row 2 is within a longer text; row 4 is a whole question (a 13-gram)
    assert _units(validation, 1) == (5, 3, 2)
    assert _failing(validation, 1, MATCH_COLUMN) == {
        2: _BENCHMARK["question"][0],
        4: _BENCHMARK["question"][1],
    }
    assert _units(validation, 2) == (5, 3, 2)
    assert _failing(validation, 3) == [4]
    # (row 2 has the 5-grams of the question, and only four more)
    assert _failing(validation, 4) == [2, 4]


def test_rows_not_contaminated_notes():
    validation = (
        pb.Validate(data=_TRAIN)
        .rows_not_contaminated(
            columns="text",
            reference={"bench": _BENCHMARK, "quiz": ["Answer: Titan"]},
            reference_columns="question",
        )
        .rows_not_contaminated(columns="text", reference=["Nothing like this at all"])
        .interrogate()
    )

    # Row 2 matches the first text of the reference that it overlaps with
    assert _note(validation, "contamination", 1) == (
        "2 rows overlap with the reference (of 3 texts): 2 with `bench`. The matching reference "
        "text of each row is in the `pb_match_` column."
    )
    assert _note(validation, "contamination", 2).startswith("No rows overlap")


def test_rows_not_contaminated_reference_kinds(tmp_path):
    path = tmp_path / "bench.csv"
    _BENCHMARK.write_csv(path)

    validation = (
        pb.Validate(data=_TRAIN)
        .rows_not_contaminated(columns="text", reference=str(path), reference_columns="question")
        .rows_not_contaminated(columns="text", reference=path, reference_columns="question")
        .rows_not_contaminated(
            columns="text",
            reference=[_BENCHMARK.lazy(), _BENCHMARK.to_pandas()],
            reference_columns="question",
        )
        .rows_not_contaminated(columns="text", reference=(_QUESTION,))
        .interrogate()
    )

    assert [_failing(validation, i) for i in range(1, 5)] == [[2, 4]] * 3 + [[2]]
    report = validation.get_tabular_report().as_raw_html()
    assert "bench.csv" in report
    assert "2 TABLES" in report
    assert "1 TEXTS" in report or "1 TEXT" in report


def test_rows_not_contaminated_evaluation_errors(tmp_path):
    validation = (
        pb.Validate(data=_TRAIN)
        .rows_not_contaminated(columns="text", reference=_BENCHMARK)
        .rows_not_contaminated(columns="text", reference=str(tmp_path / "missing.csv"))
        .rows_not_contaminated(columns="nope", reference=["a"])
        .interrogate()
    )

    assert "set `reference_columns=`" in _error_note(validation, 1)
    assert "couldn't be read" in _error_note(validation, 2)
    assert "nope" in _error_note(validation, 3)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"reference": None}, ValueError, "reference"),
        ({"reference": []}, ValueError, "empty"),
        ({"reference": {}}, ValueError, "empty"),
        ({"reference": ["a"], "method": "fuzzy"}, ValueError, "method"),
        ({"reference": ["a"], "ngram": 0}, ValueError, "ngram"),
        ({"reference": ["a"], "threshold": 2}, ValueError, "threshold"),
        ({"reference": ["a"], "reference_columns": 3}, TypeError, "reference_columns"),
    ],
)
def test_rows_not_contaminated_bad_arguments(kwargs, error, message):
    with pytest.raises(error, match=message):
        pb.Validate(data=_TRAIN).rows_not_contaminated(columns="text", **kwargs)


def test_rows_not_contaminated_step_report():
    validation = (
        pb.Validate(data=_TRAIN)
        .rows_not_contaminated(columns="text", reference=_BENCHMARK, reference_columns="question")
        .interrogate()
    )

    html = validation.get_step_report(i=1).as_raw_html()
    assert "pb_match_" in html
    assert "13-grams" in html
    assert "13-GRAM" in validation.get_tabular_report().as_raw_html()


# ---------------------------------------------------------------------------------------------
# col_vals_chat_format()
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_col_vals_chat_format(tbl_type):
    validation = (
        pb.Validate(data=_tbl(tbl_type, _CHATS))
        .col_vals_chat_format(columns="messages")
        .col_vals_chat_format(columns="messages", last_role="assistant", na_pass=True)
        .interrogate()
    )

    assert _units(validation, 1) == (6, 2, 4)
    assert _failing(validation, 1, REASON_COLUMN) == {
        2: "not_alternating",
        4: "unknown_role",
        5: "not_messages",
        6: None,
    }
    assert _units(validation, 2) == (6, 3, 3)
    assert _failing(validation, 2) == [2, 4, 5]


def test_col_vals_chat_format_note():
    validation = (
        pb.Validate(data=_CHATS)
        .col_vals_chat_format(columns="messages")
        .col_vals_chat_format(columns="messages", na_pass=True)
        .interrogate()
    )

    assert _note(validation, "chat_format", 1) == (
        "3 rows aren't well-formed conversations: 1 has roles that don't alternate "
        "(`not_alternating`), 1 isn't a list of messages (`not_messages`), and 1 has a role that "
        "isn't allowed (`unknown_role`). The problem with each row is in the `pb_reason_` "
        "column. 1 row had a missing value and failed."
    )
    assert "passed with `na_pass=True`" in _note(validation, "chat_format", 2)


def test_col_vals_chat_format_lists_of_messages():
    # Polars keeps a conversation as a list of structs, and pandas as a list of dictionaries
    conversations = [
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        [{"role": "assistant", "content": "Hello"}],
        [],
    ]
    for tbl in (
        pl.DataFrame({"id": [1, 2, 3], "messages": conversations}),
        pd.DataFrame({"id": [1, 2, 3], "messages": conversations}),
    ):
        validation = (
            pb.Validate(data=tbl)
            .col_vals_chat_format(columns="messages", first_role="user")
            .interrogate()
        )
        assert _failing(validation, 1, REASON_COLUMN) == {2: "first_role", 3: "no_messages"}


def test_col_vals_chat_format_columns():
    tbl = _CHATS.with_columns(messages_2=pl.col("messages"))

    validation = (
        pb.Validate(data=tbl)
        .col_vals_chat_format(columns=pb.starts_with("messages"), na_pass=True)
        .col_vals_chat_format(columns=["messages", "messages_2"], roles=["user", "assistant"])
        .interrogate()
    )

    assert len(validation.validation_info) == 4
    assert [s.column for s in validation.validation_info] == ["messages", "messages_2"] * 2
    # Without "system" in `roles=`, a system message is an unknown role
    assert _failing(validation, 3, REASON_COLUMN)[3] == "unknown_role"


def test_col_vals_chat_format_lazy_needs_text():
    pytest.importorskip("ibis")
    import ibis

    tbl = ibis.memtable(pl.DataFrame({"id": [1], "messages": [[{"role": "user", "content": "a"}]]}))

    validation = pb.Validate(data=tbl).col_vals_chat_format(columns="messages").interrogate()

    assert "JSON" in _error_note(validation)


def test_col_vals_chat_format_report():
    validation = (
        pb.Validate(data=_CHATS)
        .col_vals_chat_format(columns="messages", first_role="user", last_role="assistant")
        .interrogate()
    )

    report = validation.get_tabular_report().as_raw_html()
    assert "ALTERNATING" in report
    assert "FIRST: USER" in report
    assert "LAST: ASSISTANT" in report
    assert "pb_reason_" in validation.get_step_report(i=1).as_raw_html()


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"roles": "user"}, TypeError, "roles"),
        ({"roles": []}, ValueError, "roles"),
        ({"roles": ["user", 1]}, TypeError, "roles"),
        ({"first_role": "bot"}, ValueError, "first_role"),
        ({"last_role": "bot"}, ValueError, "last_role"),
        ({"alternate": "yes"}, ValueError, "alternate"),
        ({"role_key": ""}, TypeError, "role_key"),
        ({"content_key": 1}, TypeError, "content_key"),
    ],
)
def test_col_vals_chat_format_bad_arguments(kwargs, error, message):
    with pytest.raises(error, match=message):
        pb.Validate(data=_CHATS).col_vals_chat_format(columns="messages", **kwargs)


# ---------------------------------------------------------------------------------------------
# Plans: Steps, YAML, and serialization
# ---------------------------------------------------------------------------------------------


def test_text_checks_in_steps():
    steps = (
        pb.Steps()
        .rows_near_distinct(columns="text", method="exact")
        .rows_not_contaminated(columns="text", reference=[_QUESTION])
    )

    validation = pb.Validate(data=_TRAIN).add_steps(steps).interrogate()

    assert [s.assertion_type for s in validation.validation_info] == [
        "rows_near_distinct",
        "rows_not_contaminated",
    ]
    assert _failing(validation, 2) == [2]
    assert "method='exact'" in repr(steps)
    # (a `method=` argument doesn't clash with the method of the step)
    assert pb.Step("rows_near_distinct", method="exact").kwargs == {"method": "exact"}


def test_text_checks_in_yaml(tmp_path):
    path = tmp_path / "bench.csv"
    _BENCHMARK.write_csv(path)

    config = f"""
    tbl: train
    steps:
    - rows_near_distinct:
        columns: text
        threshold: 0.5
    - rows_not_contaminated:
        columns: text
        reference: {path}
        reference_columns: question
    - rows_not_contaminated:
        columns: text
        reference:
          quiz:
          - "Answer: Titan"
    - col_vals_chat_format:
        columns: text
        roles: [user, assistant]
        last_role: assistant
    """

    validation = pb.yaml_interrogate(config, set_tbl=_TRAIN)

    assert _failing(validation, 2) == [2, 4]
    assert _failing(validation, 3) == [2]
    assert _units(validation, 4) == (5, 0, 5)

    # The plan is written back as it was given
    rebuilt = pb.yaml_interrogate(validation.to_yaml(), set_tbl=_TRAIN)
    assert [s.n_failed for s in rebuilt.validation_info] == [
        s.n_failed for s in validation.validation_info
    ]


def test_text_checks_to_code_placeholder():
    validation = pb.Validate(data=_TRAIN).rows_not_contaminated(
        columns="text", reference=_BENCHMARK, reference_columns="question"
    )

    with pytest.warns(UserWarning, match="reference that cannot be serialized"):
        code = validation.to_code()

    assert "reference=your_reference" in code
    assert 'reference_columns="question"' in code


# ---------------------------------------------------------------------------------------------
# col_vals_text_quality(), col_vals_in_language(), col_vals_token_len(), and col_vals_no_pii()
# ---------------------------------------------------------------------------------------------

_WEB = pl.DataFrame(
    {
        "text": [
            "The museum is open from nine to five, and the entry is free for children under the "
            "age of twelve, so it's a good place to spend the day with family.",
            "Home\nAbout\nHome\nAbout\nHome\nContact us",
            "#sale #deals #shoes #free #today ... click to see more of the best deals",
            "- eggs\n- milk\n- bread\n- butter\n- apples\n- the rest of the list of things to buy",
            None,
        ]
    }
)

_WEB = _WEB.with_columns(id=pl.int_range(1, pl.len() + 1))


def _failing_by_step(validation: pb.Validate) -> list[list[int]]:
    return [_failing(validation, i) for i in range(1, len(validation.validation_info) + 1)]


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_col_vals_text_quality(tbl_type):
    validation = (
        pb.Validate(data=_tbl(tbl_type, _WEB))
        .col_vals_text_quality(columns="text", min_words=10, max_bullet_lines=0.5)
        .interrogate()
    )

    assert [s.assertion_type for s in validation.validation_info] == ["col_vals_scored"] * 8
    assert _failing_by_step(validation) == [[2], [], [3], [4], [], [4], [2], [2]]
    # A missing text passes
    assert all(s.n == 5 for s in validation.validation_info)


def test_col_vals_text_quality_briefs_and_bounds():
    validation = (
        pb.Validate(data=_WEB)
        .col_vals_text_quality(
            columns="text",
            min_words=None,
            max_words=500,
            min_mean_word_length=None,
            max_mean_word_length=None,
            max_symbol_ratio=None,
            max_bullet_lines=None,
            max_ellipsis_lines=None,
            min_alpha_words=0.75,
            min_stop_words=1,
            max_repeated_lines=None,
            stop_words=["est", "le"],
        )
        .interrogate()
    )

    assert [s.brief for s in validation.validation_info] == [
        "At most 500 words in `text`.",
        "At least 75% of the words have a letter in `text`.",
        "At least 1 distinct stop word in `text`.",
    ]
    assert [(s.values["min_val"], s.values["max_val"]) for s in validation.validation_info] == [
        (None, 500),
        (0.75, None),
        (1, None),
    ]
    # None of the texts has a French stop word
    assert _failing(validation, 3) == [1, 2, 3, 4]

    defaults = pb.Validate(data=_WEB).col_vals_text_quality(columns="text")
    assert [s.brief for s in defaults.validation_info] == [
        "Between 50 and 100,000 words in `text`.",
        "A mean word length of between 3 and 10 characters in `text`.",
        "At most 0.1 hash symbols and ellipses for each word in `text`.",
        "At most 90% of the lines start with a bullet point in `text`.",
        "At most 30% of the lines end with an ellipsis in `text`.",
        "At least 80% of the words have a letter in `text`.",
        "At least 2 distinct stop words in `text`.",
        "At most 30% of the lines repeat an earlier line in `text`.",
    ]
    assert all(s.na_pass for s in defaults.validation_info)
    assert all(s.values["cache"] is False for s in defaults.validation_info)


def test_col_vals_text_quality_columns_and_settings():
    tbl = _WEB.with_columns(title=pl.lit("A title"))

    validation = pb.Validate(data=tbl).col_vals_text_quality(
        columns=["text", "title"],
        min_words=10,
        na_pass=False,
        thresholds=(1, 2),
        active=False,
        pre=lambda df: df,
        dimension="text_quality",
    )

    assert [s.column for s in validation.validation_info] == [["text"]] * 8 + [["title"]] * 8
    assert validation.validation_info[8].brief == "Between 10 and 100,000 words in `title`."
    assert all(s.active is False for s in validation.validation_info)
    assert all(s.thresholds.warning == 1 for s in validation.validation_info)
    assert all(s.na_pass is False for s in validation.validation_info)
    assert all(s.pre is not None for s in validation.validation_info)
    assert all(s.dimension == "text_quality" for s in validation.validation_info)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"columns": 1}, TypeError, "`columns=` must be the name of a column"),
        ({"columns": []}, TypeError, "`columns=` must be the name of a column"),
        ({"columns": ["text", "text"]}, ValueError, "the same column more than once"),
        ({"min_words": "50"}, TypeError, "`min_words=` must be a number"),
        ({"max_symbol_ratio": True}, TypeError, "`max_symbol_ratio=` must be a number"),
        ({"min_words": 10, "max_words": 5}, ValueError, "`min_words=` must not be greater"),
        (
            {"min_mean_word_length": 8, "max_mean_word_length": 4},
            ValueError,
            "`min_mean_word_length=` must not be greater",
        ),
        ({"stop_words": "the"}, TypeError, "`stop_words=` must be a list"),
        ({"brief": 1}, TypeError, "`brief=` must be a string or a boolean"),
        ({"na_pass": "yes"}, ValueError, "na_pass"),
    ],
)
def test_col_vals_text_quality_bad_arguments(kwargs, error, message):
    kwargs = {"columns": "text", **kwargs}
    validation = pb.Validate(data=_WEB)
    with pytest.raises(error, match=message):
        validation.col_vals_text_quality(**kwargs)
    # A bad setting adds no steps
    assert validation.validation_info == []


def _guess(text):
    words = set(text.lower().split())
    if words & {"the", "and", "is"}:
        return {"en": 0.9, "fr": 0.1}
    if words & {"le", "et", "est"}:
        return {"en": 0.1, "fr": 0.9}
    return {"en": 0.5, "fr": 0.5}


_LANGS = pl.DataFrame(
    {
        "text": ["the soup is good", "le pain est bon", "the bread is fresh", "ok"],
        "lang": ["en", "fr", "fr", "en"],
        "id": [1, 2, 3, 4],
    }
)


def test_col_vals_in_language():
    validation = (
        pb.Validate(data=_LANGS)
        .col_vals_in_language(columns="text", expected="en", backend=_guess)
        .col_vals_in_language(columns="text", expected=["en", "fr"], backend=_guess)
        .col_vals_in_language(
            columns="text", expected_column="lang", backend=_guess, min_confidence=0.6
        )
        .interrogate()
    )

    assert [s.assertion_type for s in validation.validation_info] == ["col_vals_scored"] * 3
    assert _failing_by_step(validation) == [[2], [], [3, 4]]
    assert validation.validation_info[2].column == ["text", "lang"]
    assert validation.validation_info[2].values["min_val"] == 0.6
    assert [s.brief for s in validation.validation_info] == [
        "The text of `text` is in `en` (probability of at least 0.5).",
        "The text of `text` is in `en` or `fr` (probability of at least 0.5).",
        "The text of `text` is in the language of `lang` (probability of at least 0.6).",
    ]


def test_col_vals_in_language_columns():
    tbl = _LANGS.with_columns(title=pl.lit("le titre"))

    validation = (
        pb.Validate(data=tbl)
        .col_vals_in_language(
            columns=["text", "title"], expected_column="lang", backend=_guess, brief=False
        )
        .interrogate()
    )

    assert [s.column for s in validation.validation_info] == [
        ["text", "lang"],
        ["title", "lang"],
    ]
    assert _failing_by_step(validation) == [[3], [1, 4]]
    assert all(s.brief is None for s in validation.validation_info)


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({}, ValueError, "needs one of `expected=` and `expected_column=`"),
        ({"expected": "en", "min_confidence": 2}, ValueError, "between 0 and 1"),
        ({"expected": "en", "min_confidence": "high"}, TypeError, "must be a number"),
        ({"expected": "en", "columns": 1}, TypeError, "`columns=` must be the name"),
        ({"expected_column": "text"}, ValueError, "can't be one of the columns of text"),
    ],
)
def test_col_vals_in_language_bad_arguments(kwargs, error, message):
    kwargs = {"columns": "text", "backend": _guess, **kwargs}
    validation = pb.Validate(data=_LANGS)
    with pytest.raises(error, match=message):
        validation.col_vals_in_language(**kwargs)
    assert validation.validation_info == []


def test_col_vals_token_len():
    prompts = pl.DataFrame(
        {
            "prompt": ["Summarize the article.", "Hi", "a b c d e f", None],
            "reply": ["Sure.", "Hello there, how can I help?", "ok", "Yes"],
            "id": [1, 2, 3, 4],
        }
    )

    validation = (
        pb.Validate(data=prompts)
        .col_vals_token_len(columns="prompt", tokenizer="whitespace", min_val=2)
        .col_vals_token_len(columns="prompt", tokenizer=list, max_val=11, na_pass=True)
        .col_vals_token_len(columns=["prompt", "reply"], tokenizer="whitespace", min_val=2)
        .interrogate()
    )

    assert [s.assertion_type for s in validation.validation_info] == ["col_vals_scored"] * 4
    assert _failing_by_step(validation) == [[2, 4], [1], [2, 4], [1, 3, 4]]
    assert [s.brief for s in validation.validation_info] == [
        "At least 2 tokens in `prompt`.",
        "At most 11 tokens in `prompt`.",
        "At least 2 tokens in `prompt`.",
        "At least 2 tokens in `reply`.",
    ]

    one = pb.Validate(data=prompts).col_vals_token_len(
        columns="prompt", tokenizer="whitespace", min_val=1, max_val=1, brief="Step {step}: {auto}"
    )
    assert one.validation_info[0].brief == "Step {step}: Between 1 and 1 token in `prompt`."


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({}, ValueError, "needs `min_val=`, `max_val=`, or both"),
        ({"max_val": "10"}, TypeError, "`max_val=` must be a number"),
        ({"min_val": 5, "max_val": 2}, ValueError, "must not be greater than `max_val=`"),
        (
            {"tokenizer": 1, "max_val": 10},
            TypeError,
            "`tokenizer=` must be the name of a tokenizer",
        ),
    ],
)
def test_col_vals_token_len_bad_arguments(kwargs, error, message):
    kwargs = {"columns": "prompt", "tokenizer": "whitespace", **kwargs}
    validation = pb.Validate(data=_LANGS)
    with pytest.raises(error, match=message):
        validation.col_vals_token_len(**kwargs)
    assert validation.validation_info == []


_TICKETS = pl.DataFrame(
    {
        "text": [
            "The app crashes when I open the settings page.",
            "Please email me at jo.smith@example.com when it's fixed.",
            "Call me back on (555) 123-4567 after five.",
            "I was charged twice on card 4111 1111 1111 1111.",
            "Order 2024-11-05 of 3 items, for 12345 points.",
            None,
        ],
        "note": [
            "From 192.168.0.1",
            "SSN 123-45-6789",
            "IBAN GB82 WEST 1234 5698 7654 32",
            "Employee EMP-123456",
            "Version 1.2.3",
            "Fine",
        ],
        "id": [1, 2, 3, 4, 5, 6],
    }
)


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_col_vals_no_pii(tbl_type):
    validation = pb.Validate(data=_tbl(tbl_type, _TICKETS)).col_vals_no_pii(columns="text")
    validation = validation.interrogate()

    assert [s.assertion_type for s in validation.validation_info] == ["col_vals_regex"] * 3
    assert [s.brief for s in validation.validation_info] == [
        "No email addresses in `text`.",
        "No phone numbers in `text`.",
        "No credit card numbers in `text`.",
    ]
    assert all(s.values["inverse"] for s in validation.validation_info)
    assert _failing_by_step(validation) == [[2], [3], [4]]


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_col_vals_no_pii_other_entities(tbl_type):
    entities = {"ip_address": None, "us_ssn": None, "iban": None, "employee": r"EMP-[0-9]{6}"}

    validation = (
        pb.Validate(data=_tbl(tbl_type, _TICKETS))
        .col_vals_no_pii(columns=["text", "note"], entities=entities)
        .interrogate()
    )

    assert len(validation.validation_info) == 8
    assert _failing_by_step(validation)[4:] == [[1], [2], [3], [4]]
    assert _failing_by_step(validation)[:4] == [[], [], [], []]
    assert validation.validation_info[7].brief == "No values that look like `employee` in `note`."


def test_col_vals_no_pii_missing_values_and_settings():
    validation = (
        pb.Validate(data=_TICKETS)
        .col_vals_no_pii(
            columns="text", entities="email", na_pass=False, thresholds=(1, 2), dimension="privacy"
        )
        .interrogate()
    )

    assert _failing(validation) == [2, 6]
    assert validation.validation_info[0].thresholds.warning == 1
    assert validation.validation_info[0].dimension == "privacy"


@pytest.mark.parametrize(
    "kwargs, error, message",
    [
        ({"columns": []}, TypeError, "`columns=` must be the name"),
        ({"columns": ["text", 1]}, TypeError, "`columns=` must be the name"),
        ({"entities": ["email", "name"]}, ValueError, "Unknown entity 'name'"),
        ({"entities": {"name": None}}, ValueError, "Unknown entity 'name'"),
        ({"entities": {"id": 1}}, TypeError, "The pattern of the entity 'id'"),
        ({"entities": []}, ValueError, "`entities=` is empty"),
        ({"entities": 1}, TypeError, "`entities=` must be a list"),
    ],
)
def test_col_vals_no_pii_bad_arguments(kwargs, error, message):
    kwargs = {"columns": "text", **kwargs}
    validation = pb.Validate(data=_TICKETS)
    with pytest.raises(error, match=message):
        validation.col_vals_no_pii(**kwargs)
    assert validation.validation_info == []


def test_text_checks_as_steps():
    steps = (
        pb.Steps()
        .col_vals_text_quality("body", min_words=10)
        .col_vals_in_language("body", expected_column="language", backend=_guess)
        .col_vals_token_len("body", tokenizer="whitespace", max_val=100)
        .col_vals_no_pii("body")
    )

    # Each method is one step of the collection, with only its settings that aren't defaults
    assert len(steps) == 4
    assert repr(steps).splitlines()[1:] == [
        "  1. col_vals_text_quality(columns='body', min_words=10)",
        f"  2. col_vals_in_language(columns='body', expected_column='language', backend={_guess!r})",
        "  3. col_vals_token_len(columns='body', tokenizer='whitespace', max_val=100)",
        "  4. col_vals_no_pii(columns='body')",
    ]

    tbl = _WEB.rename({"text": "words"}).with_columns(lang=pl.lit("en"))
    validation = pb.Validate(data=tbl).add_steps(
        steps, columns_map={"body": "words", "language": "lang"}
    )

    assert len(validation.validation_info) == 8 + 1 + 1 + 3
    assert validation.validation_info[8].column == ["words", "lang"]
    assert validation.validation_info[-1].brief == "No credit card numbers in `words`."

    # `exclude=` counts the steps of the collection, so one position leaves out a whole check
    fewer = pb.Validate(data=tbl).add_steps(
        steps, columns_map={"body": "words", "language": "lang"}, exclude=[1, "col_vals_no_pii"]
    )
    assert len(fewer.validation_info) == 2


def test_text_checks_in_yaml():
    config = """
tbl: web
steps:
  - col_vals_text_quality:
      columns: text
      min_words: 10
      max_bullet_lines: 0.5
  - col_vals_token_len:
      columns: text
      tokenizer: whitespace
      max_val: 20
  - col_vals_no_pii:
      columns: [text]
      entities:
        email: null
        employee: EMP-[0-9]{6}
      dimension: privacy
"""
    validation = pb.yaml_interrogate(config, set_tbl=_WEB)

    assert len(validation.validation_info) == 8 + 1 + 2
    assert _failing_by_step(validation)[:8] == [[2], [], [3], [4], [], [4], [2], [2]]
    assert validation.validation_info[8].brief == "At most 20 tokens in `text`."
    assert validation.validation_info[10].brief == "No values that look like `employee` in `text`."
    assert validation.validation_info[10].dimension == "privacy"


def test_text_steps_round_trip_through_yaml():
    validation = (
        pb.Validate(data=_WEB, tbl_name="web")
        .col_vals_text_quality(columns="text", min_words=10)
        .col_vals_token_len(columns="text", tokenizer="whitespace", min_val=3)
        .col_vals_no_pii(columns="text")
        .interrogate()
    )

    rebuilt = pb.yaml_interrogate(validation.to_yaml(), set_tbl=_WEB)

    assert [s.brief for s in rebuilt.validation_info] == [
        s.brief for s in validation.validation_info
    ]
    assert [s.n_failed for s in rebuilt.validation_info] == [
        s.n_failed for s in validation.validation_info
    ]


def test_text_module_is_gone():
    assert not hasattr(pb, "text")
