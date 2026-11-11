import copy
import datetime
import decimal
import os
import sqlite3
import sys
import types

import narwhals as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest

import pointblank as pb
from pointblank import scorers
from pointblank.yaml import YAMLValidationError

# Scores are the length of `t` over 10: 0.3, 1.0, (none: a missing value), 0.6, and 0.3 (the
# last row repeats the first, so it's scored once)
_TBL = pl.DataFrame(
    {
        "t": ["abc", "abcdefghij", None, "abcdef", "abc"],
        "u": [1, 2, 3, 4, 5],
    }
)

_TBL_TYPES = ["polars", "lazy", "pandas", "duckdb", "sqlite", "pyspark"]

# The sizes of the batches that the scorers below were given
_CALLS: list[int] = []


def _length(batch):
    _CALLS.append(len(batch))
    return [len(row["t"]) / 10 for row in batch]


def _length_v2(batch):
    _CALLS.append(len(batch))
    return [len(row["t"]) / 100 for row in batch]


@pytest.fixture(autouse=True)
def _fresh_cache():
    scorers.clear_cache()
    _CALLS.clear()
    yield
    scorers.clear_cache()


def _spark_tbl(tbl: pl.DataFrame):
    if os.environ.get("SKIP_PYSPARK_TESTS", "").lower() in ("true", "1", "yes"):
        pytest.skip("PySpark tests disabled via SKIP_PYSPARK_TESTS environment variable")
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.appName("PointblankScorerTests")
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


def _tbl(tbl_type: str, tbl: pl.DataFrame = _TBL):
    if tbl_type == "polars":
        return tbl
    if tbl_type == "lazy":
        return tbl.lazy()
    if tbl_type == "pandas":
        return tbl.to_pandas()
    if tbl_type == "duckdb":
        ibis = pytest.importorskip("ibis")
        return ibis.memtable(tbl)
    if tbl_type == "sqlite":
        ibis = pytest.importorskip("ibis")
        return ibis.sqlite.connect().create_table("scored_tbl", tbl.to_pandas())
    return _spark_tbl(tbl)


def _scored(tbl=_TBL, scorer=_length, **kwargs) -> pb.Validate:
    kwargs.setdefault("min_val", 0.5)
    return pb.Validate(data=tbl).col_vals_scored(columns="t", scorer=scorer, **kwargs).interrogate()


def _error(validation: pb.Validate) -> str:
    assert validation.validation_info[0].eval_error
    return validation.get_note(1, "eval_error", format="text")


# ---------------------------------------------------------------------------------------------
# Scoring and the results of a step
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_col_vals_scored(tbl_type):
    validation = _scored(_tbl(tbl_type))
    vi = validation.validation_info[0]

    assert not vi.eval_error
    assert (vi.n, vi.n_passed, vi.n_failed) == (5, 2, 3)
    # The distinct rows with a value are scored once
    assert sum(_CALLS) == 3
    assert vi.val_info == {
        "scorer": "tests.test_scorers._length",
        "version": None,
        "label": "_length",
        "n_distinct": 3,
        "n_scored": 3,
        "n_cached": 0,
        "n_rows": 5,
        "n_unscored": 1,
        "min": 0.3,
        "max": 1.0,
        # The median counts each row (the repeated row twice); SQLite has no median
        "median": None if tbl_type == "sqlite" else pytest.approx(0.45),
    }
    median = "" if tbl_type == "sqlite" else " (median 0.45)"
    assert validation.get_note(1, "scores", format="text") == (
        "Scored by tests.test_scorers._length: 3 distinct rows scored. "
        f"Scores from 0.3 to 1{median}. 1 row had no score (a "
        "missing value, or none from the scorer) and failed (`na_pass=True` lets it pass)."
    )

    # The scores are kept next to the results
    results = nw.from_native(vi.tbl_checked).lazy().collect()
    assert "pb_score_" in results.columns
    by_u = {u: None if s != s else s for u, s in results.select("u", "pb_score_").sort("u").rows()}
    assert by_u == {1: 0.3, 2: 1.0, 3: None, 4: 0.6, 5: 0.3}


def test_col_vals_scored_bounds_and_na_pass():
    assert _scored(min_val=0.5).n_passed(i=1, scalar=True) == 2
    assert _scored(min_val=0.5, na_pass=True).n_passed(i=1, scalar=True) == 3
    assert _scored(min_val=None, max_val=0.5).n_passed(i=1, scalar=True) == 2
    assert _scored(min_val=0.3, max_val=0.6).n_passed(i=1, scalar=True) == 3
    # The bounds are inclusive
    assert _scored(min_val=1.0).n_passed(i=1, scalar=True) == 1


def test_col_vals_scored_rows_without_a_score():
    def some_none(batch):
        return [None if row["t"] == "abc" else len(row["t"]) for row in batch]

    validation = _scored(scorer=some_none, min_val=0)
    vi = validation.validation_info[0]
    assert (vi.n_passed, vi.n_failed) == (2, 3)
    assert vi.val_info["n_unscored"] == 3
    assert "3 rows had no score" in validation.get_note(1, "scores", format="text")
    assert "failed (`na_pass=True` lets them pass)" in validation.get_note(
        1, "scores", format="text"
    )
    passing = _scored(scorer=some_none, min_val=0, na_pass=True)
    assert passing.n_passed(i=1, scalar=True) == 5
    assert "passed with `na_pass=True`." in passing.get_note(1, "scores", format="text")
    assert "passed with <code>na_pass=True</code>." in passing.get_note(
        1, "scores", format="markdown"
    )

    # NaN and numpy numbers are scores too (NaN is no score)
    def numpy_scores(batch):
        return np.array([np.nan if row["t"] == "abc" else len(row["t"]) for row in batch])

    vi = _scored(scorer=numpy_scores, min_val=0).validation_info[0]
    assert (vi.n_passed, vi.val_info["n_unscored"]) == (2, 3)


def test_col_vals_scored_without_keeping_scores():
    validation = _scored(keep_scores=False)
    vi = validation.validation_info[0]
    assert vi.n_passed == 2
    assert "pb_score_" not in nw.from_native(vi.tbl_checked).columns
    assert "pb_score_" not in validation.get_data_extracts(i=1, frame=True).columns
    assert not any(c.startswith("pb_score_") for c in validation.get_row_flags().columns)


def test_col_vals_scored_with_several_columns():
    def ratio(batch):
        return [len(row["t"]) / row["u"] for row in batch]

    validation = (
        pb.Validate(data=_TBL)
        .col_vals_scored(columns=["t", "u"], scorer=ratio, min_val=1)
        .interrogate()
    )
    # 3/1, 10/2, 6/4, and 3/5 (the last row is no longer a repeat of the first)
    assert (validation.n_passed(i=1, scalar=True), validation.n_failed(i=1, scalar=True)) == (3, 2)
    assert validation.validation_info[0].val_info["n_distinct"] == 4


def test_col_vals_scored_pandas_with_a_custom_index():
    df = _TBL.to_pandas()
    df.index = [10, 20, 30, 40, 50]
    vi = _scored(df).validation_info[0]
    assert vi.tbl_checked["pb_score_"].tolist()[:2] == [0.3, 1.0]
    assert vi.tbl_checked["pb_is_good_"].tolist() == [False, True, False, True, False]
    assert list(vi.tbl_checked.index) == [10, 20, 30, 40, 50]


def test_col_vals_scored_pandas_missing_values():
    df = pd.DataFrame({"t": ["abc", pd.NA, "abcdef"], "u": pd.array([1, pd.NA, 3], dtype="Int64")})

    def total(batch):
        return [len(row["t"]) + row["u"] for row in batch]

    validation = (
        pb.Validate(data=df).col_vals_scored(columns=["t", "u"], scorer=total, min_val=5)
    ).interrogate()
    vi = validation.validation_info[0]
    assert (vi.n_passed, vi.n_failed, vi.val_info["n_unscored"]) == (1, 2, 1)


def test_col_vals_scored_with_dates_and_decimals():
    tbl = pl.DataFrame(
        {
            "d": [datetime.date(2024, 1, 1), datetime.date(2024, 6, 1)],
            "x": [decimal.Decimal("1.5"), decimal.Decimal("2.5")],
        }
    )

    def month(batch):
        return [row["d"].month + float(row["x"]) for row in batch]

    validation = (
        pb.Validate(data=tbl).col_vals_scored(columns=["d", "x"], scorer=month, min_val=5)
    ).interrogate()
    assert validation.n_passed(i=1, scalar=True) == 1


def test_col_vals_scored_extracts_get_row_flags_and_reports():
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_not_null(columns="t")
        .col_vals_scored(columns="t", scorer=_length, min_val=0.5)
        .interrogate()
    )

    extract = validation.get_data_extracts(i=2, frame=True)
    assert extract.columns == ["_row_num_", "t", "u", "pb_score_"]
    assert extract["pb_score_"].to_list() == [0.3, None, 0.3]

    flags = validation.get_row_flags()
    assert flags["pb_score_2_"].to_list() == [0.3, 1.0, None, 0.6, 0.3]
    assert flags["pb_failed_2_"].to_list() == [True, False, True, False, True]
    assert "pb_score_1_" not in flags.columns

    html = validation.get_tabular_report().as_raw_html()
    assert "≥0.5" in html
    assert "_length</span>" in html
    assert "Scored by <code>tests.test_scorers._length</code>" in html

    step_html = validation.get_step_report(i=2).as_raw_html()
    assert "pb_score_" in step_html
    assert "_length(t) &ge; 0.5" in step_html


def test_col_vals_scored_badge_of_a_long_name():
    scorer = pb.scorers.from_function(_length, name="a_rather_long_name_for_a_scorer")
    html = _scored(scorer=scorer).get_tabular_report().as_raw_html()
    assert "a_rather_long_name_for_…</span>" in html


def test_col_vals_scored_batches():
    scorer = pb.scorers.from_function(_length, batch_size=2)
    _scored(scorer=scorer)
    assert _CALLS == [2, 1]

    # A function can also say its batch size itself (an attribute that isn't a positive integer
    # is ignored)
    def sized(batch):
        _CALLS.append(len(batch))
        return [1.0] * len(batch)

    _CALLS.clear()
    sized.batch_size = 1
    _scored(scorer=sized, cache=False)
    assert _CALLS == [1, 1, 1]
    _CALLS.clear()
    sized.batch_size = True
    _scored(scorer=sized, cache=False)
    assert _CALLS == [3]


def test_col_vals_scored_segments():
    tbl = _TBL.with_columns(g=pl.Series(["a", "a", "b", "b", "b"]))
    scorer = pb.scorers.perplexity(fn=lambda text: len(text) / 10)
    validation = (
        pb.Validate(data=tbl)
        .col_vals_scored(columns="t", scorer=scorer, min_val=0.5, segments="g")
        .interrogate()
    )
    assert [vi.n_passed for vi in validation.validation_info] == [1, 1]
    # The segments share the scorer (and the model it has loaded)
    assert copy.deepcopy(scorer) is scorer
    assert {id(vi.values["scorer"]) for vi in validation.validation_info} == {id(scorer)}


def test_col_vals_scored_with_pre():
    validation = _scored(pre=lambda df: df.filter(pl.col("u") > 1))
    assert (validation.n(i=1, scalar=True), validation.n_passed(i=1, scalar=True)) == (4, 2)


# ---------------------------------------------------------------------------------------------
# Errors of a scorer, which stop the step
# ---------------------------------------------------------------------------------------------


def test_col_vals_scored_errors_of_the_scorer():
    def fails(batch):
        raise RuntimeError("the model is down\nmore detail")

    assert _error(_scored(scorer=fails)) == (
        "The scorer fails of the col_vals_scored() step failed (RuntimeError: the model is down)."
    )
    assert _error(_scored(scorer=lambda batch: [1.0])) == (
        "The scorer <lambda> of the col_vals_scored() step can't be used: it returned 1 scores "
        "for 3 rows."
    )
    assert "a score that isn't a number ('high')" in _error(
        _scored(scorer=lambda batch: ["high"] * len(batch))
    )
    assert "a score that isn't a number ([1])" in _error(
        _scored(scorer=lambda batch: [[1]] * len(batch))
    )
    assert "it returned dict instead of a list of scores" in _error(
        _scored(scorer=lambda batch: {"a": 1})
    )
    assert "it returned str instead of a list of scores" in _error(
        _scored(scorer=lambda batch: "1")
    )
    assert "it returned int instead of a list of scores" in _error(_scored(scorer=lambda batch: 1))
    # A generator is a list of scores
    vi = _scored(scorer=lambda batch: (1.0 for _ in batch)).validation_info[0]
    assert not vi.eval_error and vi.n_passed == 4


def test_col_vals_scored_errors_of_the_table():
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_scored(columns=["t", "v", "w"], scorer=_length, min_val=0.5)
        .interrogate()
    )
    assert _error(validation) == (
        "The columns 'v', 'w' of the col_vals_scored() step aren't in the table."
    )
    validation = (
        pb.Validate(data=_TBL).col_vals_scored(columns="v", scorer=_length, min_val=0.5)
    ).interrogate()
    assert _error(validation) == "The column 'v' of the col_vals_scored() step isn't in the table."

    assert "rename it first" in _error(_scored(_TBL.with_columns(pb_score_=pl.lit(1.0))))
    assert _CALLS == []


def test_col_vals_scored_missing_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "langid", None)
    monkeypatch.setitem(sys.modules, "langid.langid", None)
    message = _error(_scored(scorer=pb.scorers.langid(expected="en")))
    assert message.startswith(
        "The col_vals_scored() step couldn't be run: langid() needs the langid package, which "
        "can be installed with `pip install langid`"
    )


def test_col_vals_scored_unsupported_backend(monkeypatch):
    from pointblank import _interrogation

    pytest.importorskip("ibis")
    lazy = nw.from_native(pl.LazyFrame({"t": ["abc"]}))
    with pytest.raises(_interrogation.StepEvaluationError, match="not a"):
        _interrogation._scored_lookup(lazy, ["t"], [{"t": "abc"}], [0.3])


# ---------------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------------


def test_col_vals_scored_argument_errors():
    validate = pb.Validate(data=_TBL)

    for columns in (1, [], [""], ["t", 2]):
        with pytest.raises(TypeError, match="`columns=` must be"):
            validate.col_vals_scored(columns=columns, scorer=_length, min_val=0)
    with pytest.raises(ValueError, match="the same column more than once"):
        validate.col_vals_scored(columns=["t", "t"], scorer=_length, min_val=0)
    with pytest.raises(TypeError, match="`scorer=` must be a function"):
        validate.col_vals_scored(columns="t", scorer="model", min_val=0)
    with pytest.raises(TypeError, match="`min_val=` must be a number, got bool"):
        validate.col_vals_scored(columns="t", scorer=_length, min_val=True)
    with pytest.raises(TypeError, match="`max_val=` must be a number, got str"):
        validate.col_vals_scored(columns="t", scorer=_length, max_val="1")
    with pytest.raises(ValueError, match="needs a bound"):
        validate.col_vals_scored(columns="t", scorer=_length)
    with pytest.raises(ValueError, match="must not be greater than"):
        validate.col_vals_scored(columns="t", scorer=_length, min_val=2, max_val=1)
    with pytest.raises(ValueError):
        validate.col_vals_scored(columns="t", scorer=_length, min_val=0, na_pass="yes")
    with pytest.raises(ValueError):
        validate.col_vals_scored(columns="t", scorer=_length, min_val=0, keep_scores=1)
    with pytest.raises(TypeError, match="`cache=` must be"):
        validate.col_vals_scored(columns="t", scorer=_length, min_val=0, cache=1)
    with pytest.raises(ValueError, match="needs a scorer with a stable name"):
        validate.col_vals_scored(columns="t", scorer=lambda b: b, min_val=0, cache="dir")
    assert validate.validation_info == []

    # Bounds can be any real numbers
    validate.col_vals_scored(columns="t", scorer=_length, min_val=np.float64(0.5), max_val=1)
    assert len(validate.validation_info) == 1


# ---------------------------------------------------------------------------------------------
# The score cache
# ---------------------------------------------------------------------------------------------


def test_memory_cache():
    _scored()
    assert sum(_CALLS) == 3
    vi = _scored().validation_info[0]
    # The same scorer on the same rows isn't called again
    assert sum(_CALLS) == 3
    assert (vi.val_info["n_scored"], vi.val_info["n_cached"]) == (0, 3)
    assert "3 distinct rows, with every score from the cache." in _scored().get_note(
        1, "scores", format="text"
    )

    # New rows are the only ones scored
    validation = _scored(pl.DataFrame({"t": ["abc", "abcd", "abcde"]}))
    assert sum(_CALLS) == 5
    assert "3 distinct rows: 2 scored and 1 from the cache." in validation.get_note(
        1, "scores", format="text"
    )
    _scored(_TBL.with_columns(t=pl.col("t") + "x"))
    assert sum(_CALLS) == 8

    # `cache=False` scores every time
    _scored(cache=False)
    assert sum(_CALLS) == 11

    pb.scorers.clear_cache()
    _scored()
    assert sum(_CALLS) == 14


def test_memory_cache_by_name_and_version():
    _scored(scorer=pb.scorers.from_function(_length, version="1"))
    _scored(scorer=pb.scorers.from_function(_length, version="1"))
    assert sum(_CALLS) == 3
    # A new version is scored again
    _scored(scorer=pb.scorers.from_function(_length, version="2"))
    assert sum(_CALLS) == 6
    # No version is a version of its own
    _scored()
    assert sum(_CALLS) == 9
    # Two functions with the same name share their scores (so a name must be stable)
    _scored(scorer=pb.scorers.from_function(_length_v2, name="tests.test_scorers._length"))
    assert sum(_CALLS) == 9


def test_memory_cache_of_unnamed_scorers():
    def make():
        def inner(batch):
            _CALLS.append(len(batch))
            return [1.0] * len(batch)

        return inner

    scorer = make()
    _scored(scorer=scorer)
    _scored(scorer=scorer)
    # The same function object is cached...
    assert sum(_CALLS) == 3
    # ...but another one with the same name isn't
    _scored(scorer=make())
    assert sum(_CALLS) == 6

    class NoWeakref:
        __slots__ = ()

        def __call__(self, batch):
            _CALLS.append(len(batch))
            return [1.0] * len(batch)

    scorer = NoWeakref()
    _scored(scorer=scorer)
    _scored(scorer=scorer)
    assert sum(_CALLS) == 12
    assert scorers.scorer_label(scorer) == "NoWeakref"


def test_memory_cache_limit(monkeypatch):
    monkeypatch.setattr(scorers, "_MEMORY_CACHE_LIMIT", 2)
    _scored()
    store = scorers._named_scores[("tests.test_scorers._length", None)]
    assert len(store) == 2
    _scored()
    assert sum(_CALLS) == 4


def test_disk_cache(tmp_path):
    cache = tmp_path / "scores"
    vi = _scored(cache=cache).validation_info[0]
    assert vi.val_info["n_scored"] == 3
    assert (cache / scorers.CACHE_FILE).exists()

    # Another session (nothing in memory) finds the scores on disk
    pb.scorers.clear_cache()
    vi = _scored(cache=str(cache)).validation_info[0]
    assert sum(_CALLS) == 3
    assert vi.val_info["n_cached"] == 3

    # A new version is scored again, and kept next to the older one
    _scored(scorer=pb.scorers.from_function(_length, version=2), cache=cache)
    assert sum(_CALLS) == 6
    with sqlite3.connect(cache / scorers.CACHE_FILE) as connection:
        rows = connection.execute("SELECT scorer, version, COUNT(*) FROM scores GROUP BY 1, 2")
        assert sorted(rows) == [
            ("tests.test_scorers._length", "", 3),
            ("tests.test_scorers._length", "2", 3),
        ]

    # The code and YAML of the step keep the directory
    validation = pb.Validate(data=_TBL).col_vals_scored(
        columns="t", scorer=_length, min_val=0.5, cache=cache
    )
    with pytest.warns(UserWarning, match="placeholder `your_scorer`"):
        assert f"cache: {cache}" in validation.to_yaml()


def test_disk_cache_resumes_after_a_failure(tmp_path):
    def flaky(batch):
        _CALLS.append(len(batch))
        if len(_CALLS) == 2:
            raise RuntimeError("timeout")
        return [len(row["t"]) / 10 for row in batch]

    scorer = pb.scorers.from_function(flaky, name="flaky", batch_size=2)
    assert _scored(scorer=scorer, cache=tmp_path).validation_info[0].eval_error
    # The first batch was kept, so only the rest is scored
    vi = _scored(scorer=scorer, cache=tmp_path).validation_info[0]
    assert not vi.eval_error
    assert _CALLS == [2, 1, 1]
    assert (vi.val_info["n_cached"], vi.val_info["n_scored"]) == (2, 1)


def test_disk_cache_with_many_rows(tmp_path):
    tbl = pl.DataFrame({"t": [f"row {k}" for k in range(1200)]})
    assert _scored(tbl, cache=tmp_path).n_passed(i=1, scalar=True) == 1200
    pb.scorers.clear_cache()
    vi = _scored(tbl, cache=tmp_path).validation_info[0]
    assert vi.val_info["n_cached"] == 1200


def test_record_keys():
    key = scorers.record_key
    assert key({"a": 1, "b": "x"}) == key({"b": "x", "a": 1})
    assert key({"a": 1}) != key({"a": "1"})
    assert key({"a": 1}) != key({"b": 1})
    for value in (
        datetime.datetime(2024, 1, 1, 12),
        datetime.time(12, 30),
        datetime.timedelta(seconds=90),
        decimal.Decimal("1.10"),
        b"\x00\x01",
        np.array([1, 2]),
        {3, 1},
        object(),
    ):
        assert len(key({"a": value})) == 32
    assert key({"a": decimal.Decimal("1.10")}) != key({"a": decimal.Decimal("1.1")})

    assert scorers.python_value(np.int64(3)) == 3
    assert type(scorers.python_value(np.float32(1.5))) is float
    assert scorers.python_value(np.float64("nan")) is None
    assert scorers.python_value(pd.NA) is None
    assert scorers.python_value(pd.NaT) is None
    assert scorers.python_value(float("nan")) is None


# ---------------------------------------------------------------------------------------------
# from_function() and the Scorer protocol
# ---------------------------------------------------------------------------------------------


def test_from_function():
    scorer = pb.scorers.from_function(_length)
    assert isinstance(scorer, pb.scorers.Scorer)
    assert (scorer.name, scorer.version, scorer.batch_size) == (
        "tests.test_scorers._length",
        None,
        256,
    )
    assert scorers.scorer_label(scorer) == "_length"
    assert scorer([{"t": "ab"}]) == [0.2]

    scorer = pb.scorers.from_function(
        lambda row: len(row["t"]), name="len", version=2, batch_size=10, per_row=True
    )
    assert (scorer.name, scorer.version, scorer.batch_size) == ("len", "2", 10)
    assert scorers.scorer_label(scorer) == "len"
    assert scorer([{"t": "ab"}, {"t": "abc"}]) == [2, 3]

    # A lambda has no stable name, but is labelled by its name in the report
    scorer = pb.scorers.from_function(lambda batch: [1.0] * len(batch))
    assert scorer.name is None
    assert scorers.scorer_label(scorer) == "<lambda>"
    assert "fn=<function>" in repr(scorer)
    assert not scorer._is_literal()


def test_from_function_errors():
    with pytest.raises(TypeError, match="`fn=` must be a function, got int"):
        pb.scorers.from_function(1)
    with pytest.raises(TypeError, match="`per_row=` must be"):
        pb.scorers.from_function(_length, per_row=1)
    with pytest.raises(TypeError, match="`name=` must be a non-empty string"):
        pb.scorers.from_function(_length, name="")
    with pytest.raises(TypeError, match="`version=` must be a string or a number"):
        pb.scorers.from_function(_length, version=["1"])
    for batch_size in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="`batch_size=` must be a positive integer"):
            pb.scorers.from_function(_length, batch_size=batch_size)


def test_scorer_identity():
    class Model:
        name = "pkg.models.Quality"
        version = 3

        def __call__(self, batch):
            return [1.0] * len(batch)

    assert scorers.scorer_identity(Model()) == ("pkg.models.Quality", "3")
    assert scorers.scorer_label(Model()) == "Quality"
    assert scorers.scorer_identity(_length) == ("tests.test_scorers._length", None)
    assert scorers.scorer_identity(len) == ("builtins.len", None)

    def nested(batch):
        return batch

    assert scorers.scorer_identity(nested) == (None, None)
    assert scorers.scorer_identity(Model().__call__) == (None, None)
    assert scorers.scorer_batch_size(Model()) == 256


# ---------------------------------------------------------------------------------------------
# The adapters, with stand-ins for the packages they use
# ---------------------------------------------------------------------------------------------


def _module(monkeypatch, name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def _ppl(text):
    return float(len(text))


def _letters(texts):
    return [[t.count(c) for c in "abc"] for t in texts]


def test_embedding_similarity():
    tbl = pl.DataFrame({"t": ["abc", "aaa", "zzz", "cba"]})
    scorer = pb.scorers.embedding_similarity(reference=["abc", "bbb"], embedder=_letters)
    validation = (
        pb.Validate(data=tbl).col_vals_scored(columns="t", scorer=scorer, min_val=0.9)
    ).interrogate()
    vi = validation.validation_info[0]
    scores = nw.from_native(vi.tbl_checked)["pb_score_"].to_list()
    assert scores == [pytest.approx(1.0), pytest.approx(3**-0.5), None, pytest.approx(1.0)]
    assert (vi.n_passed, vi.n_failed) == (2, 2)

    assert scorer.name.startswith(
        "embedding_similarity(embedder=tests.test_scorers._letters, reference='2 texts, "
    )
    assert scorer.batch_size == 64
    assert scorers.scorer_label(scorer) == "EMBEDDING"
    assert scorer._is_literal() is False
    assert repr(scorer) == (
        'pb.scorers.embedding_similarity(reference=["abc", "bbb"], embedder=<function>)'
    )

    # Different references are a different scorer
    other = pb.scorers.embedding_similarity(reference="abc", embedder=_letters)
    assert other.name != scorer.name and "'1 text, " in other.name

    # A model with an `encode()` method has no stable name
    model = types.SimpleNamespace(encode=_letters)
    scorer = pb.scorers.embedding_similarity(reference="abc", embedder=model)
    assert scorer.name is None
    assert scorer([{"t": "abc"}]) == [pytest.approx(1.0)]


def test_embedding_similarity_with_a_model_name(monkeypatch):
    loaded = []

    class SentenceTransformer:
        def __init__(self, name):
            loaded.append(name)

        def encode(self, texts):
            return np.array(_letters(texts), dtype=float)

    _module(monkeypatch, "sentence_transformers", SentenceTransformer=SentenceTransformer)
    scorer = pb.scorers.embedding_similarity(reference="abc", embedder="mini", column="t")
    assert scorer.name.startswith("embedding_similarity(embedder='mini', reference='1 text, ")
    assert scorer.name.endswith(", column='t')")
    assert scorer._is_literal()
    assert scorer._to_code() == (
        'pb.scorers.embedding_similarity(reference="abc", embedder="mini", column="t")'
    )
    assert scorer._to_yaml() == {
        "embedding_similarity": {"reference": "abc", "embedder": "mini", "column": "t"}
    }
    assert scorer([{"t": "abc", "u": 1}, {"t": "cab", "u": 2}]) == [
        pytest.approx(1.0),
        pytest.approx(1.0),
    ]
    scorer([{"t": "abc", "u": 1}])
    # The model and the references are loaded once
    assert loaded == ["mini"]

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    scorer = pb.scorers.embedding_similarity(reference="abc", embedder="mini")
    with pytest.raises(ImportError, match="pip install sentence-transformers"):
        scorer([{"t": "abc"}])


def test_embedding_similarity_errors():
    for reference in (1, [], ["a", 2]):
        with pytest.raises(TypeError, match="`reference=` must be"):
            pb.scorers.embedding_similarity(reference=reference, embedder=_letters)
    with pytest.raises(TypeError, match="`embedder=` must be .* got int"):
        pb.scorers.embedding_similarity(reference="a", embedder=1)
    with pytest.raises(TypeError, match="`column=` must be the name of a column"):
        pb.scorers.embedding_similarity(reference="a", embedder=_letters, column=1)

    scorer = pb.scorers.embedding_similarity(reference="zzz", embedder=_letters)
    with pytest.raises(ValueError, match="only zero vectors"):
        scorer([{"t": "abc"}])
    scorer = pb.scorers.embedding_similarity(reference="abc", embedder=lambda texts: [[1, 0]])
    with pytest.raises(ValueError, match="returned 1 vectors for 2 rows"):
        scorer([{"t": "abc"}, {"t": "b"}])
    scorer = pb.scorers.embedding_similarity(reference="abc", embedder=_letters)
    with pytest.raises(ValueError, match=r"choose one with `column=`"):
        scorer([{"t": "abc", "u": "b"}])
    scorer = pb.scorers.embedding_similarity(reference="abc", embedder=_letters, column="v")
    with pytest.raises(ValueError, match=r"the column 'v' \(`column=`\), which isn't one"):
        scorer([{"t": "abc"}])


def test_reranker_with_cohere(monkeypatch):
    clients = []

    class ClientV2:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.requests = []
            clients.append(self)

        def rerank(self, model, query, documents, top_n):
            self.requests.append((model, query, documents, top_n))
            # The results are sorted by relevance, not in the order of the documents
            results = [
                types.SimpleNamespace(index=i, relevance_score=len(d) / 10)
                for i, d in enumerate(documents)
            ]
            return types.SimpleNamespace(results=results[::-1])

    _module(monkeypatch, "cohere", ClientV2=ClientV2)
    scorer = pb.scorers.reranker(api_key="secret")
    assert scorer.name == "reranker(provider='cohere', model='rerank-v3.5')"
    assert scorer.batch_size == 100
    assert "secret" not in scorer._to_code()
    assert scorer._to_yaml() == {"reranker": {"provider": "cohere"}}

    tbl = pl.DataFrame({"q": ["a", "b", "a"], "doc": ["xx", "xxxxxx", "xxxx"]})
    validation = (
        pb.Validate(data=tbl).col_vals_scored(columns=["q", "doc"], scorer=scorer, min_val=0.3)
    ).interrogate()
    assert nw.from_native(validation.validation_info[0].tbl_checked)["pb_score_"].to_list() == [
        0.2,
        0.6,
        0.4,
    ]
    # The documents of a query are sent together
    (client,) = clients
    assert client.kwargs == {"api_key": "secret"}
    assert client.requests == [
        ("rerank-v3.5", "a", ["xx", "xxxx"], 2),
        ("rerank-v3.5", "b", ["xxxxxx"], 1),
    ]

    # Without a key, the client finds it itself
    pb.scorers.reranker(document="q")([{"q": "a", "doc": "xx"}])
    assert clients[-1].kwargs == {}
    assert clients[-1].requests[0][1:3] == ("xx", ["a"])

    monkeypatch.setitem(sys.modules, "cohere", None)
    with pytest.raises(ImportError, match="pip install cohere"):
        pb.scorers.reranker()([{"q": "a", "doc": "b"}])


def test_reranker_with_sentence_transformers(monkeypatch):
    class CrossEncoder:
        def __init__(self, name):
            self.name = name

        def predict(self, pairs):
            return np.array([len(d) - len(q) for q, d in pairs], dtype=float)

    _module(monkeypatch, "sentence_transformers", CrossEncoder=CrossEncoder)
    scorer = pb.scorers.reranker(provider="sentence-transformers", query="q", document="d")
    assert scorer.name == (
        "reranker(provider='sentence-transformers', "
        "model='cross-encoder/ms-marco-MiniLM-L6-v2', query='q', document='d')"
    )
    assert scorer([{"d": "abc", "q": "a", "x": 1}]).tolist() == [2.0]

    model = CrossEncoder("local")
    scorer = pb.scorers.reranker(provider="sentence-transformers", model=model)
    assert scorer.name is None
    assert scorer._get_client() is model

    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(ImportError, match="pip install sentence-transformers"):
        pb.scorers.reranker(provider="sentence-transformers")([{"q": "a", "d": "b"}])


def test_reranker_errors():
    with pytest.raises(ValueError, match="`provider=` must be one of 'cohere'"):
        pb.scorers.reranker(provider="openai")
    with pytest.raises(TypeError, match="name of a Cohere rerank model"):
        pb.scorers.reranker(model=object())
    with pytest.raises(TypeError, match="a `predict\\(\\)` method"):
        pb.scorers.reranker(provider="sentence-transformers", model=object())
    with pytest.raises(TypeError, match="`query=` must be the name of a column"):
        pb.scorers.reranker(query="")
    with pytest.raises(TypeError, match="`document=` must be the name of a column"):
        pb.scorers.reranker(document=1)
    with pytest.raises(ValueError, match="must be different columns"):
        pb.scorers.reranker(query="a", document="a")

    scorer = pb.scorers.reranker()
    with pytest.raises(ValueError, match="needs a query column and a document column"):
        scorer._columns({"q": "a"})
    with pytest.raises(ValueError, match="needs a query column and a document column"):
        pb.scorers.reranker(document="q")._columns({"q": "a"})
    with pytest.raises(ValueError, match=r"the column 'd' \(`document=`\)"):
        pb.scorers.reranker(document="d")._columns({"q": "a", "x": "b"})
    assert pb.scorers.reranker(query="x")._columns({"q": "a", "x": "b"}) == ("x", "q")


def test_classifier(monkeypatch):
    calls = []

    def pipe(texts, **kwargs):
        calls.append(kwargs)
        out = [
            [{"label": "POS", "score": len(t) / 10}, {"label": "NEG", "score": 0.5}] for t in texts
        ]
        # A pipeline gives one text's labels without the outer list
        return out[0] if len(texts) == 1 else out

    pipe.model = types.SimpleNamespace(name_or_path="org/sentiment")
    scorer = pb.scorers.classifier(hf_model=pipe, label="POS", function_to_apply="softmax")
    assert scorer.name == (
        "classifier(hf_model='org/sentiment', label='POS', function_to_apply='softmax')"
    )
    assert scorer.batch_size == 32
    assert scorers.scorer_label(scorer) == "CLASSIFIER"
    assert scorer([{"t": "abc"}, {"t": "a"}]) == [0.3, 0.1]
    assert scorer([{"t": "abcd"}]) == [0.4]
    assert calls[0] == {"top_k": None, "truncation": True, "function_to_apply": "softmax"}

    with pytest.raises(ValueError, match=r"has the labels 'POS', 'NEG': choose the one"):
        pb.scorers.classifier(hf_model=pipe)([{"t": "a"}, {"t": "b"}])
    with pytest.raises(ValueError, match=r"no label 'MIXED' \(`label=`\); its labels are 'POS'"):
        pb.scorers.classifier(hf_model=pipe, label="MIXED")([{"t": "a"}, {"t": "b"}])

    # A regression model has one output, with no label to choose
    def regression(texts, **kwargs):
        return [{"label": "LABEL_0", "score": float(len(t))} for t in texts]

    assert pb.scorers.classifier(hf_model=regression).name is None
    assert pb.scorers.classifier(hf_model=regression)([{"t": "ab"}, {"t": "a"}]) == [2.0, 1.0]


def test_classifier_with_a_function():
    # A function of just the texts gets none of the pipeline's keyword arguments
    def labels(texts):
        return [[{"label": "POS", "score": len(t) / 10}] for t in texts]

    scorer = pb.scorers.classifier(hf_model=labels, label="POS")
    assert scorer([{"t": "abc"}, {"t": "a"}]) == [0.3, 0.1]
    assert scorer([{"t": "ab"}]) == [0.2]

    # It gets the ones it takes
    calls = []

    def truncating(texts, truncation=False):
        calls.append(truncation)
        return [{"label": "LABEL_0", "score": float(len(t))} for t in texts]

    assert pb.scorers.classifier(hf_model=truncating)([{"t": "ab"}, {"t": "a"}]) == [2.0, 1.0]
    assert calls == [True]

    def applying(texts, *, function_to_apply):
        calls.append(function_to_apply)
        return [[{"label": "POS", "score": 0.5}] for _ in texts]

    scorer = pb.scorers.classifier(hf_model=applying, label="POS", function_to_apply="sigmoid")
    assert scorer([{"t": "a"}, {"t": "b"}]) == [0.5, 0.5]
    assert calls[-1] == "sigmoid"

    # As a scorer, it works through `col_vals_scored()`
    validation = (
        pb.Validate(data=pl.DataFrame({"t": ["abc", "a", "abcde"]}))
        .col_vals_scored(
            columns="t", scorer=pb.scorers.classifier(hf_model=labels, label="POS"), min_val=0.3
        )
        .interrogate()
    )
    info = validation.validation_info[0]
    assert info.eval_error is not True
    assert (info.n_passed, info.n_failed) == (2, 1)

    # `function_to_apply=` needs a function that takes it
    with pytest.raises(ValueError, match="`function_to_apply=` needs a model name or a pipeline"):
        pb.scorers.classifier(hf_model=labels, function_to_apply="softmax")

    # A callable without a signature to inspect gets all of them
    assert scorers._keywords_taken(type, {"top_k": None}) == {"top_k": None}


def test_classifier_with_a_model_name(monkeypatch):
    made = []

    def pipeline(**kwargs):
        made.append(kwargs)
        return lambda texts, **kw: [[{"label": "edu", "score": 2.5}] for _ in texts]

    _module(monkeypatch, "transformers", pipeline=pipeline)
    scorer = pb.scorers.classifier(hf_model="org/edu", device="mps", version="r1")
    assert scorer.name == "classifier(hf_model='org/edu')"
    assert scorer.version == "r1"
    assert scorer._to_code() == (
        'pb.scorers.classifier(hf_model="org/edu", device="mps", version="r1")'
    )
    assert scorer([{"t": "a"}, {"t": "b"}]) == [2.5, 2.5]
    scorer([{"t": "c"}, {"t": "d"}])
    assert made == [{"task": "text-classification", "model": "org/edu", "device": "mps"}]
    pb.scorers.classifier(hf_model="org/edu")([{"t": "a"}, {"t": "b"}])
    assert made[-1] == {"task": "text-classification", "model": "org/edu"}

    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="pip install transformers"):
        pb.scorers.classifier(hf_model="org/edu")([{"t": "a"}])


def test_classifier_errors():
    with pytest.raises(TypeError, match="`hf_model=` must be .* got int"):
        pb.scorers.classifier(hf_model=1)
    with pytest.raises(TypeError, match="`label=` must be"):
        pb.scorers.classifier(hf_model="m", label=1)
    with pytest.raises(ValueError, match="`function_to_apply=` must be"):
        pb.scorers.classifier(hf_model="m", function_to_apply="relu")
    with pytest.raises(TypeError, match="`column=` must be"):
        pb.scorers.classifier(hf_model="m", column="")


def test_langid(monkeypatch):
    def guess(text):
        return {"en_US": 0.6, "en_GB": 0.2, "fr": 0.2} if "the" in text else [("fr", 0.9)]

    scorer = pb.scorers.langid(expected="EN", backend=guess)
    # A function defined inside another one has no stable name
    assert scorer.name is None
    assert scorer([{"t": "the cat"}, {"t": "le chat"}]) == [pytest.approx(0.8), 0.0]
    assert pb.scorers.langid(expected=["fr", "en"], backend=guess)([{"t": "the"}]) == [
        pytest.approx(1.0)
    ]

    class LanguageIdentifier:
        @classmethod
        def from_modelstring(cls, model, norm_probs):
            assert (model, norm_probs) == ("MODEL", True)
            return types.SimpleNamespace(rank=lambda text: [("en", 0.7), ("de", 0.3)])

    package = _module(monkeypatch, "langid")
    package.langid = _module(
        monkeypatch, "langid.langid", LanguageIdentifier=LanguageIdentifier, model="MODEL"
    )
    scorer = pb.scorers.langid(expected=["en", "de"])
    assert scorer.name == "langid(expected=['de', 'en'], backend='langid')"
    assert scorer._to_yaml() == {"langid": {"expected": ["en", "de"]}}
    assert scorer([{"t": "x"}]) == [pytest.approx(1.0)]


def test_langid_with_fasttext(monkeypatch, tmp_path):
    loaded = []

    class Model:
        def predict(self, text, k):
            assert "\n" not in text and k == -1
            return ("__label__en", "__label__fr"), np.array([0.75, 0.25])

    def load_model(path):
        loaded.append(path)
        return Model()

    _module(monkeypatch, "fasttext", load_model=load_model)
    path = tmp_path / "lid.176.bin"
    scorer = pb.scorers.langid(expected="fr", backend="fasttext", model=path)
    assert scorer._to_code() == (
        f'pb.scorers.langid(expected="fr", backend="fasttext", model="{path}")'
    )
    assert scorer([{"t": "a\nb"}, {"t": "c"}]) == [0.25, 0.25]
    assert loaded == [str(path)]

    monkeypatch.setitem(sys.modules, "fasttext", None)
    with pytest.raises(ImportError, match="pip install fasttext"):
        pb.scorers.langid(expected="fr", backend="fasttext", model="m.bin")([{"t": "a"}])


def test_langid_errors():
    for expected in (1, [], ["en", ""]):
        with pytest.raises(TypeError, match="`expected=` must be a language code"):
            pb.scorers.langid(expected=expected)
    with pytest.raises(ValueError, match='`backend=` must be "langid", "fasttext", or a function'):
        pb.scorers.langid(expected="en", backend="cld3")
    with pytest.raises(TypeError, match="got int"):
        pb.scorers.langid(expected="en", backend=1)
    with pytest.raises(ValueError, match="needs the path of a fastText"):
        pb.scorers.langid(expected="en", backend="fasttext")


def test_perplexity(monkeypatch):
    def ppl(text):
        return float(len(text))

    scorer = pb.scorers.perplexity(fn=ppl)
    assert scorer.name is None
    assert pb.scorers.perplexity(fn=_ppl).name == "perplexity(fn=tests.test_scorers._ppl)"
    assert scorer([{"t": "abc"}]) == [3.0]
    batched = pb.scorers.perplexity(fn=lambda texts: [len(t) for t in texts], per_row=False)
    assert batched([{"t": "ab"}, {"t": "a"}]) == [2, 1]
    assert "per_row=False" in repr(batched)

    loaded = []

    class Model:
        def __init__(self, path):
            loaded.append(path)

        def perplexity(self, text):
            return 10.0 * len(text)

    _module(monkeypatch, "kenlm", Model=Model)
    scorer = pb.scorers.perplexity(kenlm_model="en.arpa", column="t")
    assert scorer.name == "perplexity(kenlm_model='en.arpa', column='t')"
    assert scorer._to_yaml() == {"perplexity": {"kenlm_model": "en.arpa", "column": "t"}}
    assert scorer([{"t": "ab", "u": 1}]) == [20.0]
    scorer([{"t": "a", "u": 1}])
    assert loaded == ["en.arpa"]

    monkeypatch.setitem(sys.modules, "kenlm", None)
    with pytest.raises(ImportError, match="pip install kenlm"):
        pb.scorers.perplexity(kenlm_model="en.arpa")([{"t": "a"}])


def test_perplexity_errors():
    with pytest.raises(ValueError, match="needs one of `fn=` and `kenlm_model=`"):
        pb.scorers.perplexity()
    with pytest.raises(ValueError, match="needs one of `fn=` and `kenlm_model=`"):
        pb.scorers.perplexity(fn=len, kenlm_model="m")
    with pytest.raises(TypeError, match="`fn=` must be a function"):
        pb.scorers.perplexity(fn="m")
    with pytest.raises(TypeError, match="`kenlm_model=` must be the path"):
        pb.scorers.perplexity(kenlm_model=1)
    with pytest.raises(TypeError, match="`per_row=` must be"):
        pb.scorers.perplexity(fn=len, per_row=None)


def test_adapter_names_of_long_settings():
    scorer = pb.scorers.langid(expected=[f"l{k}" for k in range(30)], backend="langid")
    # A setting too long to show is named by a hash of it
    assert scorer.name.startswith("langid(expected=<") and len(scorer.name) < 60


def test_langid_with_expected_column():
    def guess(text):
        return {"en": 0.9, "fr": 0.1} if "the" in text else {"en": 0.2, "fr": 0.8}

    scorer = pb.scorers.langid(expected_column="lang", backend=guess)
    batch = [
        {"t": "the cat", "lang": "en"},
        {"t": "le chat", "lang": "EN"},
        {"t": "le chat", "lang": ["fr", "en"]},
    ]
    assert scorer(batch) == [pytest.approx(0.9), pytest.approx(0.2), pytest.approx(1.0)]
    # With a third column, `column=` names the text
    named = pb.scorers.langid(expected_column="lang", backend=guess, column="t")
    assert named([{"t": "the", "lang": "fr", "id": 1}]) == [pytest.approx(0.1)]

    with pytest.raises(ValueError, match="choose one with `column=`"):
        scorer([{"t": "the", "lang": "fr", "id": 1}])
    with pytest.raises(ValueError, match="reads the expected language from the column 'lang'"):
        scorer([{"t": "the"}])

    validation = (
        pb.Validate(data=pl.DataFrame({"t": ["the cat", "le chat"], "lang": ["en", "en"]}))
        .col_vals_scored(columns=["t", "lang"], scorer=scorer, min_val=0.5)
        .interrogate()
    )
    assert validation.n_failed(i=1, scalar=True) == 1

    langid_scorer = pb.scorers.langid(expected_column="lang")
    assert langid_scorer._to_yaml() == {"langid": {"expected_column": "lang"}}
    assert langid_scorer.name == "langid(expected_column='lang', backend='langid')"


def test_langid_expected_column_errors():
    with pytest.raises(ValueError, match="needs one of `expected=` and `expected_column=`"):
        pb.scorers.langid()
    with pytest.raises(ValueError, match="needs one of `expected=` and `expected_column=`"):
        pb.scorers.langid(expected="en", expected_column="lang")
    with pytest.raises(TypeError, match="`expected_column=` must be the name of a column"):
        pb.scorers.langid(expected_column="")


def test_text_stat():
    words = pb.scorers.text_stat("word_count")
    assert words.name == "text_stat(stat='word_count')"
    assert words([{"t": "one two three"}, {"t": ""}]) == [3.0, 0.0]
    assert words._label == "WORD COUNT"

    length = pb.scorers.text_stat("mean_word_length", column="t")
    # A text without words has no mean word length
    assert length([{"t": "ab abcd", "u": 1}, {"t": " ", "u": 2}]) == [3.0, None]

    stops = pb.scorers.text_stat("stop_words", stop_words=["le", "la"])
    assert stops([{"t": "Le chat et la souris, la nuit."}]) == [2.0]
    assert stops._to_yaml() == {"text_stat": {"stat": "stop_words", "stop_words": ["le", "la"]}}
    assert stops._to_code() == 'pb.scorers.text_stat(stat="stop_words", stop_words=["le", "la"])'

    validation = (
        pb.Validate(data=pl.DataFrame({"t": ["a b c d", "a b", None]}))
        .col_vals_scored(columns="t", scorer=words, min_val=3, na_pass=True)
        .interrogate()
    )
    assert validation.n_failed(i=1, scalar=True) == 1


def test_text_stat_errors():
    with pytest.raises(ValueError, match="`stat=` must be one of 'char_count'"):
        pb.scorers.text_stat("letters")
    for stop_words in ("the", [], ["the", ""], 1):
        with pytest.raises(TypeError, match="`stop_words=` must be a list of words"):
            pb.scorers.text_stat("stop_words", stop_words=stop_words)
    with pytest.raises(ValueError, match='can only be used with `stat="stop_words"`'):
        pb.scorers.text_stat("word_count", stop_words=["the"])
    with pytest.raises(TypeError, match="`column=` must be"):
        pb.scorers.text_stat("word_count", column=1)
    with pytest.raises(ValueError, match="choose one with `column=`"):
        pb.scorers.text_stat("word_count")([{"t": "a", "u": "b"}])


def test_token_count(monkeypatch):
    words = pb.scorers.token_count("whitespace")
    assert words.name == "token_count(tokenizer='whitespace')"
    assert words._label == "TOKENS"
    assert words([{"t": "one two  three"}, {"t": ""}]) == [3.0, 0.0]

    # A function can give the tokens, or their number
    assert pb.scorers.token_count(list)([{"t": "abc"}]) == [3.0]
    assert pb.scorers.token_count(len)([{"t": "abcd"}]) == [4.0]

    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    class Tokenizer:
        def encode(self, text):
            return Encoding(list(text))

    assert pb.scorers.token_count(Tokenizer(), column="t")([{"t": "ab", "u": 1}]) == [2.0]
    assert pb.scorers.token_count(lambda t: {"input_ids": [1, 2]})([{"t": "x"}]) == [2.0]
    assert pb.scorers.token_count(lambda t: np.arange(5))([{"t": "x"}]) == [5.0]

    calls = []

    class TiktokenEncoding:
        def encode(self, text, disallowed_special):
            assert disallowed_special == ()
            return text.split("-")

    def get_encoding(name):
        calls.append(("get_encoding", name))
        if name != "cl100k_base":
            raise ValueError(name)
        return TiktokenEncoding()

    def encoding_for_model(name):
        calls.append(("encoding_for_model", name))
        return TiktokenEncoding()

    _module(
        monkeypatch, "tiktoken", get_encoding=get_encoding, encoding_for_model=encoding_for_model
    )
    assert pb.scorers.token_count("tiktoken:cl100k_base")([{"t": "a-b-c"}]) == [3.0]
    assert pb.scorers.token_count("tiktoken:gpt-4o")([{"t": "a-b"}]) == [2.0]
    assert calls[-2:] == [("get_encoding", "gpt-4o"), ("encoding_for_model", "gpt-4o")]

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, name):
            assert name == "org/model"
            return types.SimpleNamespace(
                encode=lambda text, add_special_tokens: [0] * (len(text) + add_special_tokens)
            )

    _module(monkeypatch, "transformers", AutoTokenizer=AutoTokenizer)
    scorer = pb.scorers.token_count("org/model")
    assert scorer([{"t": "abc"}]) == [3.0]
    assert scorer._to_yaml() == {"token_count": {"tokenizer": "org/model"}}

    monkeypatch.setitem(sys.modules, "tiktoken", None)
    with pytest.raises(ImportError, match="pip install tiktoken"):
        pb.scorers.token_count("tiktoken:cl100k_base")([{"t": "a"}])
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match="pip install transformers"):
        pb.scorers.token_count("org/model")([{"t": "a"}])


def test_token_count_errors():
    for tokenizer in ("", "tiktoken:"):
        with pytest.raises(ValueError, match="`tokenizer=` must name a tokenizer"):
            pb.scorers.token_count(tokenizer)
    with pytest.raises(TypeError, match="`tokenizer=` must be the name of a tokenizer.*got int"):
        pb.scorers.token_count(1)
    with pytest.raises(TypeError, match="returned a boolean"):
        pb.scorers.token_count(lambda t: True)([{"t": "a"}])


def test_text_adapters_in_yaml():
    config = """
    tbl: docs
    steps:
    - col_vals_scored:
        columns: t
        scorer:
          text_stat:
            stat: word_count
        min_val: 3
    - col_vals_scored:
        columns: t
        scorer:
          token_count:
            tokenizer: whitespace
        max_val: 3
    """
    docs = pl.DataFrame({"t": ["a b", "a b c", "a b c d"]})

    validation = pb.yaml_interrogate(config, set_tbl=docs)

    assert [s.n_failed for s in validation.validation_info] == [1, 1]
    assert "token_count:\n        tokenizer: whitespace" in validation.to_yaml()


# ---------------------------------------------------------------------------------------------
# Code and YAML
# ---------------------------------------------------------------------------------------------


def test_col_vals_scored_to_code_and_yaml():
    validation = pb.Validate(data=_TBL).col_vals_scored(
        columns=["t", "u"],
        scorer=pb.scorers.perplexity(kenlm_model="en.arpa", column="t"),
        max_val=100,
        na_pass=True,
        cache=False,
        keep_scores=False,
    )
    code = validation.to_code()
    assert (
        '.col_vals_scored(columns=["t", "u"], scorer=pb.scorers.perplexity('
        'kenlm_model="en.arpa", column="t"), max_val=100, cache=False, keep_scores=False, '
        "na_pass=True)"
    ) in code
    yaml_text = validation.to_yaml()
    assert (
        "- col_vals_scored:\n"
        "    columns:\n"
        "    - t\n"
        "    - u\n"
        "    scorer:\n"
        "      perplexity:\n"
        "        kenlm_model: en.arpa\n"
        "        column: t\n"
        "    max_val: 100\n"
    ) in yaml_text

    # A function is a placeholder
    validation = pb.Validate(data=_TBL).col_vals_scored(columns="t", scorer=_length, min_val=0.5)
    with pytest.warns(UserWarning, match="placeholder `your_scorer`"):
        code = validation.to_code()
    assert "scorer=your_scorer" in code
    with pytest.warns(UserWarning, match="placeholder `your_scorer`"):
        yaml_text = validation.to_yaml()
    assert "python: your_scorer" in yaml_text


def test_yaml_interrogate_with_an_adapter(monkeypatch):
    def rank(text):
        return [("en", 1.0)] if "abc" in text else [("en", 0.2)]

    class LanguageIdentifier:
        @classmethod
        def from_modelstring(cls, model, norm_probs):
            return types.SimpleNamespace(rank=rank)

    package = _module(monkeypatch, "langid")
    package.langid = _module(
        monkeypatch, "langid.langid", LanguageIdentifier=LanguageIdentifier, model="MODEL"
    )
    validation = pb.Validate(data=_TBL).col_vals_scored(
        columns="t", scorer=pb.scorers.langid(expected="en"), min_val=0.5
    )
    yaml_text = validation.to_yaml()
    assert "scorer:\n      langid:\n        expected: en\n" in yaml_text

    result = pb.yaml_interrogate(yaml_text, set_tbl=_TBL)
    vi = result.validation_info[0]
    assert vi.values["scorer"].name == "langid(expected=['en'], backend='langid')"
    # Every text but the missing one has "abc" in it
    assert (vi.n_passed, vi.n_failed) == (4, 1)

    for spec, message in (
        ("{gpt: {model: x}}", "must be written as `{<adapter>: {...}}`"),
        ("{langid: {expected: en}, perplexity: {fn: x}}", "must be written as"),
        ("{langid: en}", "The settings of the `langid` scorer must be a dictionary"),
        ("{langid: {expected: 1}}", "The `langid` scorer couldn't be made"),
        ("{langid: {language: en}}", "The `langid` scorer couldn't be made"),
    ):
        text = (
            "tbl: small_table\nsteps:\n- col_vals_scored:\n    columns: t\n"
            f"    scorer: {spec}\n    min_val: 0.5\n"
        )
        with pytest.raises(YAMLValidationError, match=message):
            pb.yaml_interrogate(text, set_tbl=_TBL)
