import json
import os
import pickle
import random
import warnings

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

import pointblank as pb
from pointblank.ablation import (
    _betainc,
    _holm,
    _NumberFormat,
    _paired,
    _removed_html,
    _t_p_value,
    _t_quantile,
)

# A table of 100 rows: 10 have a negative `x` (rows that hurt the model below), 10 others are
# 'gold' rows (rows that help it), and 10 others have a missing `y` (rows that don't matter)
_N = 100
_TBL = pl.DataFrame(
    {
        "rid": list(range(_N)),
        "x": [-1.0 if k % 10 == 0 else float(k) for k in range(_N)],
        "g": ["gold" if k % 10 == 1 else "plain" for k in range(_N)],
        "y": [None if k % 10 == 2 else k * 0.5 for k in range(_N)],
    }
)

_TBL_TYPES = ["polars", "lazy", "pandas", "duckdb", "sqlite", "pyspark"]


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
        return ibis.sqlite.connect().create_table("abl_tbl", tbl.to_pandas())
    return _spark_tbl(tbl)


def _spark_tbl(tbl: pl.DataFrame):
    if os.environ.get("SKIP_PYSPARK_TESTS", "").lower() in ("true", "1", "yes"):
        pytest.skip("PySpark tests disabled via SKIP_PYSPARK_TESTS environment variable")
    pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    try:
        spark = (
            SparkSession.builder.appName("PointblankAblationTests")
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


def _counts(train) -> tuple[int, int, int]:
    """The number of rows, of rows with a negative `x`, and of 'gold' rows in a table."""
    counts = nw.from_native(train).select(
        n=nw.len(),
        bad=(nw.col("x") < 0).cast(nw.Int64).sum(),
        gold=(nw.col("g") == "gold").cast(nw.Int64).sum(),
    )
    if isinstance(counts, nw.LazyFrame):
        counts = counts.collect()
    # `to_list()` gives Python values (`row()` can give pyarrow scalars, which older pyarrow
    # versions can't pass to `int()`)
    return tuple(int(counts[name].to_list()[0]) for name in ("n", "bad", "gold"))


def _evaluate(train, *, seed, variant):
    """A 'model' whose score falls with each negative-`x` row and rises with each 'gold' row."""
    _, bad, gold = _counts(train)
    noise = random.Random(seed).gauss(0, 0.3)
    return -bad + gold + noise


def _validation(tbl):
    return (
        pb.Validate(data=tbl)
        .col_vals_ge(columns="x", value=0)  # removes the rows that hurt
        .col_vals_ne(columns="g", value="gold")  # removes the rows that help
        .col_vals_not_null(columns="y")  # removes rows that don't matter
        .col_exists(columns="x")
        .interrogate()
    )


# ----------------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prob, df, expected",
    [
        (0.975, 1, 12.706205),
        (0.975, 2, 4.302653),
        (0.975, 4, 2.776445),
        (0.975, 10, 2.228139),
        (0.95, 4, 2.131847),
        (0.995, 29, 2.756386),
        (0.975, 1000, 1.962339),
    ],
)
def test_t_quantile(prob, df, expected):
    assert _t_quantile(prob, df) == pytest.approx(expected, abs=1e-5)


@pytest.mark.parametrize(
    "t, df, expected",
    [
        (2.0, 10, 0.07338803),
        (-2.0, 10, 0.07338803),
        (0.0, 5, 1.0),
        (1.0, 1, 0.5),
        (2.776445, 4, 0.05),
        (10.0, 3, 0.00212840),
        (float("inf"), 3, 0.0),
    ],
)
def test_t_p_value(t, df, expected):
    assert _t_p_value(t, df) == pytest.approx(expected, abs=1e-6)


def test_betainc():
    assert _betainc(2.0, 3.0, 0.0) == 0.0
    assert _betainc(2.0, 3.0, 1.0) == 1.0
    # I_x(1, 1) = x and I_x(a, 1) = x^a
    assert _betainc(1.0, 1.0, 0.3) == pytest.approx(0.3)
    assert _betainc(2.5, 1.0, 0.4) == pytest.approx(0.4**2.5)
    # I_x(a, b) = 1 - I_(1-x)(b, a)
    assert _betainc(2.0, 5.0, 0.7) == pytest.approx(1 - _betainc(5.0, 2.0, 0.3))


def test_holm():
    assert _holm([0.01, 0.04, None, 0.03]) == pytest.approx([0.03, 0.06, None, 0.06])
    assert _holm([0.5, 0.6]) == pytest.approx([1.0, 1.0])
    assert _holm([None, None]) == [None, None]
    assert _holm([]) == []
    # The adjusted p-values keep the order of the p-values
    assert _holm([0.02, 0.01, 0.03]) == pytest.approx([0.04, 0.03, 0.04])


def test_paired():
    mean, lower, upper, p = _paired([3.0, 5.0, 4.0], [1.0, 2.0, 2.0], conf_level=0.95)
    # The differences are 2, 3, and 2: mean 7/3, sd 0.57735, se 0.33333
    assert mean == pytest.approx(7 / 3)
    assert lower == pytest.approx(7 / 3 - 4.302653 / 3, abs=1e-5)
    assert upper == pytest.approx(7 / 3 + 4.302653 / 3, abs=1e-5)
    assert p == pytest.approx(_t_p_value(7.0, 2))

    # One pair has no interval or p-value
    assert _paired([2.0], [1.0], conf_level=0.95) == (1.0, None, None, None)

    # Differences that are all the same have a zero-width interval
    assert _paired([2.0, 3.0], [1.0, 2.0], conf_level=0.95) == (1.0, 1.0, 1.0, 0.0)
    assert _paired([1.0, 2.0], [1.0, 2.0], conf_level=0.95) == (0.0, 0.0, 0.0, 1.0)


def test_number_format():
    fmt = _NumberFormat([0.85, 0.9])
    assert fmt.number(0.85) == "0.8500"
    assert fmt.delta(-0.01234) == "−0.0123"
    assert fmt.delta(0.01) == "+0.0100"
    assert fmt.value(0.85, None) == "0.8500"
    assert fmt.value(0.85, 0.01) == "0.8500 ± 0.0100"
    assert _NumberFormat([1234.5]).number(1234.5) == "1,234"
    assert _NumberFormat([0.0]).decimals == 3


@pytest.mark.parametrize(
    "f_removed, expected", [(0.0, ">0%<"), (0.0005, ">&lt;0.1%<"), (0.123, ">12.3%<")]
)
def test_removed_html(f_removed, expected):
    assert expected in _removed_html(5, f_removed)


# ----------------------------------------------------------------------------------------------
# Running
# ----------------------------------------------------------------------------------------------


def test_ablation_effects_and_verdicts():
    ablation = (
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, metric="score", repeats=5, seed=1)
        .drop_each()
        .drop_all()
    )
    assert repr(ablation) == "Ablation(4 variants, 5 repeats, control='size_matched'; not run)"

    assert ablation.run() is ablation
    assert repr(ablation).endswith("; run)")

    effects = ablation.get_effects()
    assert isinstance(effects, pl.DataFrame)
    assert effects.columns == [
        "variant",
        "name",
        "strategy",
        "steps",
        "left_out",
        "group",
        "sweep",
        "value",
        "n_removed",
        "f_removed",
        "metric",
        "mean",
        "sd",
        "delta_vs_full",
        "delta_vs_full_lower",
        "delta_vs_full_upper",
        "p_vs_full",
        "p_adj_vs_full",
        "control_mean",
        "delta_vs_control",
        "delta_vs_control_lower",
        "delta_vs_control_upper",
        "p_vs_control",
        "p_adj_vs_control",
        "delta_added",
        "delta_added_lower",
        "delta_added_upper",
        "p_added",
        "p_adj_added",
        "verdict",
    ]
    assert effects["steps"].dtype == pl.List(pl.Int64)
    rows = {row["variant"]: row for row in effects.iter_rows(named=True)}

    assert list(rows) == ["full", "step_1", "step_2", "step_3", "all"]
    assert rows["full"]["strategy"] == "full"
    assert rows["full"]["n_removed"] == 0
    assert rows["full"]["delta_vs_full"] is None
    assert rows["full"]["verdict"] is None

    assert [rows[v]["n_removed"] for v in ("step_1", "step_2", "step_3", "all")] == [10, 10, 10, 30]
    assert rows["all"]["f_removed"] == pytest.approx(0.3)
    assert rows["all"]["steps"] == [1, 2, 3]

    # The seeds are shared, so the noise cancels out in the differences from the full table
    assert rows["step_1"]["delta_vs_full"] == pytest.approx(10)
    assert rows["step_2"]["delta_vs_full"] == pytest.approx(-10)
    assert rows["step_3"]["delta_vs_full"] == pytest.approx(0)
    assert rows["step_1"]["delta_vs_full_lower"] == pytest.approx(10)
    assert rows["step_1"]["p_vs_full"] == 0.0
    assert rows["step_3"]["p_vs_full"] == 1.0

    assert rows["step_1"]["verdict"] == "helps"
    assert rows["step_2"]["verdict"] == "hurts"
    assert rows["step_3"]["verdict"] == "inconclusive"
    # Removing the 'gold' rows cancels out removing the rows that hurt
    assert rows["all"]["verdict"] == "inconclusive"

    # The variants that remove 10 rows share a control
    assert rows["step_1"]["control_mean"] == rows["step_2"]["control_mean"]
    assert rows["step_1"]["control_mean"] != rows["all"]["control_mean"]
    for variant in ("step_1", "step_2", "step_3", "all"):
        row = rows[variant]
        assert row["delta_vs_control"] == pytest.approx(row["mean"] - row["control_mean"])
        assert row["delta_vs_control_lower"] < row["delta_vs_control"]
        assert row["delta_vs_control"] < row["delta_vs_control_upper"]
        assert row["p_vs_control"] <= row["p_adj_vs_control"] <= 1

    # Holm: the smallest p-value is multiplied by the number of variants
    p_values = [rows[v]["p_vs_control"] for v in ("step_1", "step_2", "step_3", "all")]
    adjusted = [rows[v]["p_adj_vs_control"] for v in ("step_1", "step_2", "step_3", "all")]
    assert adjusted == pytest.approx(_holm(p_values))
    assert min(adjusted) == pytest.approx(min(4 * min(p_values), 1))


def test_ablation_runs_are_paired_and_reproducible():
    calls = []

    def evaluate(train, *, seed, variant):
        n, bad, gold = _counts(train)
        calls.append((variant.id, seed, n, bad, gold))
        return _evaluate(train, seed=seed, variant=variant)

    validation = _validation(_TBL)
    ablation = pb.Ablation(validation, evaluate=evaluate, repeats=3, seed=10).drop_each(steps=1)
    ablation.run()

    assert [(c[0], c[1]) for c in calls] == [
        ("full", 10),
        ("full", 11),
        ("full", 12),
        ("step_1", 10),
        ("step_1", 11),
        ("step_1", 12),
        ("control_10", 10),
        ("control_10", 11),
        ("control_10", 12),
    ]
    assert {c[2:] for c in calls if c[0] == "full"} == {(100, 10, 10)}
    assert {c[2:] for c in calls if c[0] == "step_1"} == {(90, 0, 10)}
    assert {c[2] for c in calls if c[0] == "control_10"} == {90}

    # The runs give the same results each time, and the controls depend on the seed
    first = ablation.to_dict()
    calls.clear()
    ablation.run()
    assert ablation.to_dict() == first
    other = pb.Ablation(validation, evaluate=evaluate, repeats=3, seed=11).drop_each(steps=1)
    assert other.run().to_dict()["runs"] != first["runs"]

    assert first["runs"][0] == {
        "variant": "full",
        "repeat": 0,
        "seed": 10,
        "metrics": {"score": pytest.approx(_evaluate(_TBL, seed=10, variant=None))},
        "cached": False,
    }


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_ablation_training_tables(tbl_type):
    tbl = _tbl(tbl_type)
    seen = []

    def evaluate(train, *, seed, variant):
        seen.append((variant.id, type(train), _counts(train)))
        return _evaluate(train, seed=seed, variant=variant)

    repeats = 3 if tbl_type == "pyspark" else 4
    ablation = pb.Ablation(_validation(tbl), evaluate=evaluate, repeats=repeats).drop_steps(
        [1, 2], name="x and g"
    )
    ablation.run()

    # The training tables have the same type as the target table
    assert {t for _, t, _ in seen} == {type(tbl)}
    assert {c for v, _, c in seen if v == "full"} == {(100, 10, 10)}
    assert {c for v, _, c in seen if v == "x and g"} == {(80, 0, 0)}
    assert {c[0] for v, _, c in seen if v == "control_20"} == {80}

    effects = ablation.get_effects()
    assert effects["variant"].to_list() == ["full", "x and g"]
    assert effects["delta_vs_full"][1] == pytest.approx(0)


def test_ablation_pandas_row_labels_are_kept():
    tbl = _TBL.to_pandas()
    tbl.index = [f"r{k}" for k in range(_N)]
    labels = {}

    def evaluate(train, *, seed, variant):
        labels[variant.id] = list(train.index)
        return 0.0

    pb.Ablation(_validation(tbl), evaluate=evaluate, repeats=1).drop_each(steps=1).run()

    assert labels["full"] == list(tbl.index)
    assert labels["step_1"] == [f"r{k}" for k in range(_N) if k % 10 != 0]


def test_ablation_mutating_evaluate_does_not_change_the_data():
    tbl = _TBL.to_pandas()

    def evaluate(train, *, seed, variant):
        train["x"] = 0.0
        return 0.0

    validation = _validation(tbl)
    pb.Ablation(validation, evaluate=evaluate, repeats=1).drop_each(steps=1).run()
    assert (validation.data["x"] < 0).sum() == 10


def test_ablation_with_row_preserving_pre_and_segments():
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_ge(
            columns="x",
            value=0,
            pre=pb.row_preserving(lambda df: df.filter(pl.col("rid") < 50)),
        )
        .col_vals_ne(columns="g", value="gold", segments=("y", [0.5, 5.5, 10.5]))
        .col_vals_ge(columns="x", value=0, pre=lambda df: df)
        .interrogate()
    )
    seen = {}

    def evaluate(train, *, seed, variant):
        seen[variant.id] = sorted(train["rid"].to_list())
        return 0.0

    ablation = pb.Ablation(validation, evaluate=evaluate, repeats=1).drop_each()

    # The step whose `pre=` isn't marked can't be used
    with pytest.raises(ValueError, match="can't use step 5: it uses `pre=` without"):
        ablation.drop_steps(5)
    ablation.run()

    # Step 1 only checked the first 50 rows; steps 2 to 4 are the segments of the second step
    assert sorted(seen) == [
        "control_1",
        "control_5",
        "full",
        "step_1",
        "step_2",
        "step_3",
        "step_4",
    ]
    assert set(range(_N)) - set(seen["step_1"]) == {0, 10, 20, 30, 40}
    assert set(range(_N)) - set(seen["step_2"]) == {1}
    assert set(range(_N)) - set(seen["step_3"]) == {11}
    assert set(range(_N)) - set(seen["step_4"]) == {21}
    assert ablation.to_dict()["notes"] == [
        "Step 5 isn't used: it uses `pre=` without `pb.row_preserving()`."
    ]


def test_ablation_metrics():
    def evaluate(train, *, seed, variant):
        score = _evaluate(train, seed=seed, variant=variant)
        return {"score": score, "loss": -score, "n": _counts(train)[0]}

    ablation = pb.Ablation(
        _validation(_TBL), evaluate=evaluate, metric="loss", higher_is_better=False, repeats=3
    ).drop_each(steps=[1, 2])
    ablation.run()

    effects = ablation.get_effects()
    assert effects["metric"].unique(maintain_order=True).to_list() == ["score", "loss", "n"]
    verdicts = dict(
        effects.filter(pl.col("metric") == "loss").select("variant", "verdict").iter_rows()
    )
    assert verdicts == {"full": None, "step_1": "helps", "step_2": "hurts"}
    assert effects.filter(pl.col("metric") != "loss")["verdict"].null_count() == 6
    assert ablation.to_dict()["settings"]["metrics"] == ["score", "loss", "n"]

    # The report can show any metric (verdicts are only shown for the primary one)
    html = ablation.get_tabular_report(metric="n").as_raw_html()
    assert "Verdicts are based on loss" in html
    assert "helps" not in html
    # `higher_is_better=False` is for `loss` and says nothing about the other metrics
    assert "is better" not in html
    assert "(lower is better)" in ablation.get_tabular_report().as_raw_html()
    with pytest.raises(ValueError, match="There's no metric 'accuracy'"):
        ablation.get_tabular_report(metric="accuracy")


def test_ablation_metric_naming():
    def evaluate(train, *, seed):
        return {"auc": 0.5 + seed / 100}

    ablation = pb.Ablation(_validation(_TBL), evaluate=evaluate, repeats=2).drop_each(steps=1)
    ablation.run()
    assert ablation.to_dict()["settings"]["metric"] == "auc"

    # A number gets the name in `metric=`, or "score"
    for metric, name in [(None, "score"), ("acc", "acc")]:
        ablation = pb.Ablation(
            _validation(_TBL), evaluate=lambda train, seed: 1.0, metric=metric, repeats=2
        ).drop_each(steps=1)
        assert ablation.run().get_effects()["metric"].unique().to_list() == [name]


@pytest.mark.parametrize(
    "result, error, match",
    [
        ("0.9", TypeError, "must return a number or a dictionary of numbers, but it returned str"),
        (True, TypeError, "returned bool"),
        ({}, ValueError, "returned an empty dictionary"),
        ({"a": float("nan")}, ValueError, "returned nan for the metric 'a'"),
        ({"a": float("inf")}, ValueError, "must be a finite number"),
        ({"a": None}, ValueError, "returned None for the metric 'a'"),
        ({1: 0.5}, TypeError, "metric names from `evaluate=` must be strings"),
        ({"a": 1.0, "b": 2.0}, ValueError, "returned several metrics"),
    ],
)
def test_ablation_bad_results(result, error, match):
    ablation = pb.Ablation(
        _validation(_TBL), evaluate=lambda train, seed: result, repeats=1
    ).drop_each(steps=1)
    with pytest.raises(error, match=match):
        ablation.run()


def test_ablation_inconsistent_results():
    def evaluate(train, *, seed, variant):
        return {"a": 1.0} if variant.id == "full" else {"b": 1.0}

    ablation = pb.Ablation(_validation(_TBL), evaluate=evaluate, repeats=1).drop_each(steps=1)
    with pytest.raises(ValueError, match="must return the same metrics every time"):
        ablation.run()

    ablation = pb.Ablation(
        _validation(_TBL), evaluate=lambda train, seed: {"a": 1.0}, metric="b", repeats=1
    ).drop_each(steps=1)
    with pytest.raises(ValueError, match="didn't return the metric 'b'"):
        ablation.run()


def test_ablation_numpy_results():
    np = pytest.importorskip("numpy")

    ablation = pb.Ablation(
        _validation(_TBL), evaluate=lambda train, seed: np.float32(0.5), repeats=1
    ).drop_each(steps=1)
    ablation.run()
    assert ablation.get_effects()["mean"].to_list() == [0.5, 0.5]
    json.loads(ablation.to_json())


def test_ablation_without_control():
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate, control=None).drop_each()
    calls = []
    ablation.evaluate = lambda train, seed, variant: calls.append(variant.id) or _evaluate(
        train, seed=seed, variant=variant
    )
    ablation.run()

    assert not any(c.startswith("control") for c in calls)
    effects = ablation.get_effects()
    assert effects["control_mean"].null_count() == effects.height
    assert effects["delta_vs_control"].null_count() == effects.height
    # The verdicts are based on the comparison with the full table
    assert effects["verdict"].to_list() == [None, "helps", "hurts", "inconclusive"]

    html = ablation.get_tabular_report().as_raw_html()
    assert "Δ vs full" in html
    assert "Δ vs random" not in html


def test_ablation_without_adjustment():
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate, adjust=None).drop_each()
    effects = ablation.run().get_effects()
    assert effects["p_vs_control"].to_list() == effects["p_adj_vs_control"].to_list()
    html = ablation.get_tabular_report().as_raw_html()
    assert "verdicts use unadjusted p-values" in html


def test_ablation_few_repeats():
    with pytest.warns(UserWarning, match="won't give verdicts"):
        ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=2)
    effects = ablation.drop_each(steps=1).run().get_effects()
    assert effects["verdict"].to_list() == [None, None]
    assert effects["delta_vs_control_lower"][1] is not None

    with pytest.warns(UserWarning):
        ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=1)
    effects = ablation.drop_each(steps=1).run().get_effects()
    assert effects["sd"].to_list() == [None, None]
    assert effects["delta_vs_control_lower"][1] is None
    assert effects["p_vs_control"][1] is None
    html = ablation.get_tabular_report().as_raw_html()
    assert "no verdicts with fewer than 3 repeats" in html


def test_ablation_evaluate_without_variant_argument():
    seeds = []

    def evaluate(train, seed):
        seeds.append(seed)
        return 1.0

    pb.Ablation(_validation(_TBL), evaluate=evaluate, repeats=2, seed=5).drop_each(1).run()
    assert seeds == [5, 6] * 3

    received = []

    def evaluate_kwargs(train, **kwargs):
        received.append(sorted(kwargs))
        return 1.0

    pb.Ablation(_validation(_TBL), evaluate=evaluate_kwargs, repeats=1).drop_each(1).run()
    assert received == [["seed", "variant"]] * 3


# ----------------------------------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------------------------------


def test_ablation_variant_ids_and_dedup():
    ablation = (
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=3)
        .drop_steps(1)
        .drop_steps([3, 1])
        .drop_each()
        .drop_all(steps=[1, 3])
        .drop_all()
        .drop_steps([2, 3], name="g and y")
    )
    ablation.run()
    variants = ablation.to_dict()["variants"]

    assert [(v["id"], v["name"], v["strategy"], v["steps"]) for v in variants] == [
        ("full", "Full table", "full", []),
        ("steps_1", "Step 1", "drop_steps", [1]),
        ("steps_1_3", "Steps 1, 3", "drop_steps", [1, 3]),
        ("step_2", "Step 2", "drop_each", [2]),
        ("step_3", "Step 3", "drop_each", [3]),
        ("all", "All steps", "drop_all", [1, 2, 3]),
        ("g and y", "g and y", "drop_steps", [2, 3]),
        ("control_10", "10 random rows", "control", []),
        ("control_20", "20 random rows", "control", []),
        ("control_30", "30 random rows", "control", []),
    ]
    assert [v["is_control"] for v in variants] == [False] * 7 + [True] * 3
    # A variant removing the same rows as another one isn't used (without a note when it has the
    # same name)
    assert ablation.to_dict()["notes"] == [
        "All of steps 1, 3 isn't used: it removes the same rows as Steps 1, 3."
    ]

    # Calling a strategy again doesn't add anything
    ablation.drop_each()
    assert len(ablation.to_dict()["variants"]) == 10

    with pytest.raises(ValueError, match="There's already a variant called 'g and y'"):
        ablation.drop_steps(1, name="g and y")
    for name in ("full", "control_10"):
        with pytest.raises(ValueError, match="is reserved"):
            ablation.drop_steps(2, name=name)
    with pytest.raises(TypeError, match="non-empty string"):
        ablation.drop_steps(2, name="")


def test_ablation_variant_passed_to_evaluate():
    variants = {}

    def evaluate(train, *, seed, variant):
        variants[variant.id] = variant
        return 0.0

    pb.Ablation(_validation(_TBL), evaluate=evaluate, repeats=1).drop_all().run()

    assert variants["full"] == pb.AblationVariant(id="full", strategy="full", name="Full table")
    assert variants["all"] == pb.AblationVariant(
        id="all",
        strategy="drop_all",
        name="All steps",
        steps=(1, 2, 3),
        n_removed=30,
        f_removed=0.3,
    )
    assert variants["control_30"].is_control
    assert variants["control_30"].n_removed == 30
    assert variants["control_30"].steps == ()
    with pytest.raises(AttributeError):
        variants["all"].n_removed = 3
    assert pickle.loads(pickle.dumps(variants["all"])) == variants["all"]


def test_ablation_step_errors():
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_ge(columns="x", value=0)
        .col_vals_ge(columns="x", value=0, active=False)
        .col_exists(columns="x")
        .rows_distinct(columns_subset="g")
        .interrogate()
    )
    ablation = pb.Ablation(validation, evaluate=_evaluate, duplicates=None)

    for steps, match in [
        (2, "can't use step 2: it's inactive"),
        (3, "can't use step 3: it doesn't check rows"),
        (4, "can't use step 4: no `duplicates=` policy was given"),
        (9, "can't use step 9: the validation has no step 9"),
        ([1, 9], "can't use step 9"),
    ]:
        with pytest.raises(ValueError, match=match):
            ablation.drop_steps(steps)
    with pytest.raises(ValueError, match="`drop_each\\(\\)` can't use step 2"):
        ablation.drop_each(steps=2)
    with pytest.raises(ValueError, match="`drop_all\\(\\)` can't use step 2"):
        ablation.drop_all(steps=[1, 2])
    with pytest.raises(ValueError, match="at least one step"):
        ablation.drop_steps([])
    with pytest.raises(TypeError, match="must be integers"):
        ablation.drop_steps([1, "2"])
    with pytest.raises(TypeError, match="must be integers"):
        ablation.drop_steps([True])
    with pytest.raises(TypeError, match="a step number or a list of them"):
        ablation.drop_steps("1")

    # With a `duplicates=` policy, the `rows_distinct()` step can be used
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=1).drop_steps(4)
    ablation.run()
    assert ablation.get_effects()["n_removed"].to_list() == [0, 98]


def test_ablation_without_usable_steps():
    validation = pb.Validate(data=_TBL).col_exists(columns="x").interrogate()
    ablation = pb.Ablation(validation, evaluate=_evaluate)
    with pytest.raises(ValueError, match="`drop_each\\(\\)` has no steps to use"):
        ablation.drop_each()
    with pytest.raises(ValueError, match="There are no variants to evaluate"):
        ablation.run()

    # A variant removing no rows isn't used, and then there's nothing to compare
    validation = pb.Validate(data=_TBL).col_vals_ge(columns="rid", value=0).interrogate()
    ablation = pb.Ablation(validation, evaluate=_evaluate).drop_each()
    with pytest.raises(ValueError, match="None of the variants remove any rows"):
        ablation.run()
    assert "Step 1 isn't used: it removes no rows." in ablation.plan().as_raw_html()


def test_ablation_removing_every_row():
    validation = pb.Validate(data=_TBL).col_vals_lt(columns="rid", value=0).interrogate()
    ablation = pb.Ablation(validation, evaluate=_evaluate).drop_each()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        html = ablation.plan().as_raw_html()
    assert "Step 1 isn't used: it removes every row." in html


def test_ablation_max_removed_warning():
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate, max_removed=0.2).drop_each()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ablation.plan()
    ablation.drop_all()
    with pytest.warns(UserWarning, match="All steps removes 30.0% of the rows, more than"):
        ablation.plan()
    # The rows are found once until variants are added
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ablation.plan()


def test_ablation_adding_variants_resets_results():
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=3).drop_each(steps=1)
    ablation.run()
    ablation.drop_each(steps=2)
    with pytest.raises(ValueError, match="Call `run\\(\\)` \\(or `load_results\\(\\)`\\) first"):
        ablation.get_effects()
    ablation.run()
    assert ablation.get_effects()["variant"].to_list() == ["full", "step_1", "step_2"]


# ----------------------------------------------------------------------------------------------
# Arguments
# ----------------------------------------------------------------------------------------------


def test_ablation_argument_errors():
    validation = _validation(_TBL)

    for kwargs, error, match in [
        ({"validation": _TBL}, TypeError, "must be a `Validate` object"),
        ({"evaluate": 3}, TypeError, "`evaluate=` must be a function"),
        ({"metric": 3}, TypeError, "`metric=` must be a string"),
        ({"higher_is_better": "yes"}, TypeError, "`higher_is_better=`"),
        ({"repeats": 0}, ValueError, "`repeats=` must be a positive integer"),
        ({"repeats": 2.5}, ValueError, "`repeats=` must be a positive integer"),
        ({"seed": 1.5}, TypeError, "`seed=` must be an integer"),
        ({"control": "random"}, ValueError, "`control=` must be one of 'size_matched'"),
        ({"max_removed": 0}, ValueError, "`max_removed=` must be a number in"),
        ({"max_removed": 1.5}, ValueError, "`max_removed=` must be a number in"),
        ({"conf_level": 1}, ValueError, "`conf_level=` must be a number in"),
        ({"adjust": "bonferroni"}, ValueError, "`adjust=` must be one of 'holm'"),
        ({"label": 3}, TypeError, "`label=` must be a string"),
        ({"duplicates": "keep"}, ValueError, "`duplicates=` must be one of"),
        ({"duplicates_order_by": "zzz"}, ValueError, "'zzz' aren't in the target table"),
    ]:
        args = {"validation": validation, "evaluate": _evaluate, **kwargs}
        with pytest.raises(error, match=match):
            pb.Ablation(**args)

    with pytest.raises(ValueError, match="hasn't been interrogated"):
        pb.Ablation(pb.Validate(data=_TBL).col_vals_ge(columns="x", value=0), evaluate=_evaluate)

    validation = (
        pb.Validate(data=_TBL)
        .col_vals_ge(columns="x", value=0)
        .interrogate(collect_tbl_checked=False)
    )
    with pytest.raises(ValueError, match="aren't available"):
        pb.Ablation(validation, evaluate=_evaluate)


def test_ablation_duplicates_order_by():
    tbl = _TBL.with_columns(
        g=pl.when(pl.col("rid") < 4).then(pl.lit("dup")).otherwise(pl.col("rid").cast(pl.String))
    )
    validation = pb.Validate(data=tbl).rows_distinct(columns_subset="g").interrogate()
    kept = {}

    def evaluate(train, *, seed, variant):
        kept[variant.id] = sorted(train.filter(pl.col("g") == "dup")["rid"].to_list())
        return 0.0

    for duplicates, order_by, expected in [
        ("keep_first", None, [0]),
        ("keep_last", None, [3]),
        ("keep_first", "x", [0]),
        ("keep_last", "y", [3]),
    ]:
        pb.Ablation(
            validation,
            evaluate=evaluate,
            repeats=1,
            duplicates=duplicates,
            duplicates_order_by=order_by,
        ).drop_each().run()
        assert kept["step_1"] == expected

    ablation = pb.Ablation(
        validation, evaluate=evaluate, repeats=1, duplicates="flag_all", max_removed=1
    ).drop_each()
    ablation.run()
    assert kept["step_1"] == []
    assert ablation.to_dict()["settings"]["duplicates"] == "flag_all"


# ----------------------------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------------------------


def test_ablation_plan():
    ablation = (
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=4, label="Train v7")
        .drop_each()
        .drop_all()
    )
    html = ablation.plan().as_raw_html()

    assert "Ablation plan" in html
    assert "Train v7" in html
    # The full table, 4 variants, and 2 controls (removing 10 and 30 rows), 4 times each
    assert "<strong>28</strong> evaluations" in html
    assert html.count("random rows") == 2
    assert "Expect that values in <code>x</code> should be &gt;= <code>0</code>." in html
    assert "1: Expect that values in <code>x</code>" in html
    assert "Baseline" in html and "Variants" in html and "Controls" in html

    # The plan doesn't evaluate anything
    with pytest.raises(ValueError, match="Call `run\\(\\)` \\(or `load_results\\(\\)`\\) first"):
        ablation.get_tabular_report()
    assert ablation._repr_html_() == html


def test_ablation_plan_long_variant_and_brief():
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_ge(columns="x", value=0, brief="No <negative> x")
        .col_vals_ne(columns="g", value="gold")
        .col_vals_not_null(columns="y")
        .col_vals_lt(columns="x", value=95)
        .interrogate()
    )
    html = pb.Ablation(validation, evaluate=_evaluate).drop_all().plan().as_raw_html()
    assert "No &lt;negative&gt; x" in html
    assert "and 1 more steps" in html


def test_ablation_tabular_report():
    ablation = (
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, label="Train v7").drop_each().run()
    )
    report = ablation.get_tabular_report()
    html = report.as_raw_html()

    assert "Ablation report" in html
    assert "Train v7" in html
    assert "score on the full table (100 rows)" in html
    assert "(higher is better)" in html
    assert "Each version was evaluated 5 times" in html
    assert "Δ vs random" in html
    assert html.count("<svg") == 3
    for verdict in ("helps", "hurts", "inconclusive"):
        assert f">{verdict}</span>" in html
    assert "Intervals are 95% paired t-intervals, not adjusted" in html
    assert "Holm-adjusted" in html
    assert ablation._repr_html_() == html

    html = ablation.get_tabular_report(title="My <title>").as_raw_html()
    assert "My &lt;title&gt;" in html
    assert "Ablation report" not in html


def test_ablation_to_json():
    ablation = pb.Ablation(
        _validation(_TBL), evaluate=_evaluate, repeats=3, seed=4, label="v7"
    ).drop_each()
    with pytest.raises(ValueError, match="Call `run\\(\\)` \\(or `load_results\\(\\)`\\) first"):
        ablation.to_json()
    ablation.run()

    data = json.loads(ablation.to_json())
    assert list(data) == [
        "settings",
        "fingerprints",
        "variants",
        "same_as",
        "runs",
        "effects",
        "notes",
    ]
    assert set(data["fingerprints"]) == {"data", "plan"}
    assert data["same_as"] == {}
    assert data["settings"] == {
        "label": "v7",
        "metric": "score",
        "metrics": ["score"],
        "higher_is_better": True,
        "repeats": 3,
        "seed": 4,
        "control": "size_matched",
        "duplicates": "keep_first",
        "duplicates_order_by": None,
        "max_removed": 0.5,
        "conf_level": 0.95,
        "adjust": "holm",
        "evaluate_version": None,
        "n_rows": 100,
        "tbl_name": None,
    }
    # The full table, 3 variants, and one control, 3 times each
    assert len(data["runs"]) == 15
    assert len(data["effects"]) == 4
    assert data["effects"][1]["steps"] == [1]
    assert data["effects"] == ablation.get_effects().to_dicts()
    assert ablation.to_json(indent=None).count("\n") == 0


def test_ablation_effects_with_pandas(monkeypatch):
    import pointblank.ablation as ablation_module

    monkeypatch.setattr(ablation_module, "_select_df_lib", lambda preference: pd)
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=3).drop_each()
    effects = ablation.run().get_effects()

    assert isinstance(effects, pd.DataFrame)
    assert effects["variant"].tolist() == ["full", "step_1", "step_2", "step_3"]
    assert effects["n_removed"].dtype == "int64"
    assert effects["delta_vs_full"].dtype == "float64"
    assert effects["delta_vs_full"].isna().tolist() == [True, False, False, False]
    assert effects["steps"].tolist() == [[], [1], [2], [3]]
    assert "helps" in ablation.get_tabular_report().as_raw_html()


# ----------------------------------------------------------------------------------------------
# Leaving one out and groups
# ----------------------------------------------------------------------------------------------


def _effects_by_variant(ablation, metric=None):
    effects = ablation.get_effects()
    if metric is not None:
        effects = effects.filter(pl.col("metric") == metric)
    return {row["variant"]: row for row in effects.iter_rows(named=True)}


def test_ablation_drop_all_but_each():
    ablation = (
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=5)
        .drop_each()
        .drop_all_but_each()
    )
    variants = {v["id"]: v for v in ablation.run().to_dict()["variants"]}

    assert variants["all_but_1"] == {
        "id": "all_but_1",
        "strategy": "drop_all_but_each",
        "name": "All steps but step 1",
        "steps": [2, 3],
        "n_removed": 20,
        "f_removed": 0.2,
        "is_control": False,
        "group": None,
        "group_column": None,
        "left_out": 1,
        "sweep": None,
        "value": None,
    }
    # Each left-out variant has a control of its own: its rows plus as many random rows as the
    # left-out step adds
    control = variants["control_all_but_1"]
    assert (control["steps"], control["n_removed"], control["left_out"]) == ([2, 3], 30, 1)
    assert control["name"] == "All steps but step 1, plus 10 random rows"
    assert "all" in variants

    rows = _effects_by_variant(ablation)
    # Step 1's rows hurt the model, so adding them to the other steps' rows helps
    assert rows["all_but_1"]["delta_added"] > 5
    assert rows["all_but_1"]["verdict"] == "helps"
    assert rows["all_but_2"]["delta_added"] < -5
    assert rows["all_but_2"]["verdict"] == "hurts"
    assert rows["all_but_3"]["verdict"] == "inconclusive"
    # The left-out variants are compared with the full table too, but not with a control of
    # their own size
    assert rows["all_but_1"]["delta_vs_full"] is not None
    assert rows["all_but_1"]["delta_vs_control"] is None
    assert rows["step_1"]["delta_added"] is None
    assert rows["all_but_1"]["p_adj_added"] >= rows["all_but_1"]["p_added"]

    # The control of a left-out variant always removes its rows
    removed = ablation._resolve().removed["all_but_1"]
    for r in range(5):
        drawn = ablation._control_rows("control_all_but_1", r)
        assert removed < drawn and len(drawn) == 30

    html = ablation.get_tabular_report().as_raw_html()
    assert "Leave one out" in html and "Variants" in html
    assert "Δ is what the left-out step" in html


def test_ablation_drop_all_but_each_redundant_step():
    # Step 2's rows are a subset of step 1's
    tbl = _TBL.with_columns(x2=pl.when(pl.col("rid") % 20 == 0).then(-1.0).otherwise(1.0))
    validation = (
        pb.Validate(data=tbl)
        .col_vals_ge(columns="x", value=0)
        .col_vals_ge(columns="x2", value=0)
        .interrogate()
    )
    ablation = (
        pb.Ablation(validation, evaluate=_evaluate, repeats=5).drop_each().drop_all_but_each()
    )
    ablation.run()
    data = ablation.to_dict()

    # Leaving out step 2 removes the rows of step 1, and leaving out step 1 removes the rows of
    # step 2, so neither is evaluated again
    assert data["same_as"] == {"all_but_2": "step_1", "all_but_1": "step_2"}
    assert {run["variant"] for run in data["runs"]}.isdisjoint({"all_but_1", "all_but_2"})
    # Leaving out step 2 removes as many rows as every step, so it has no control
    assert "control_all_but_2" not in {v["id"] for v in data["variants"]}

    rows = _effects_by_variant(ablation)
    assert rows["step_1"]["verdict"] == "helps"
    assert rows["step_2"]["verdict"] == "helps"
    assert rows["all_but_1"]["verdict"] == "helps"
    assert rows["all_but_2"]["delta_added"] == 0
    assert rows["all_but_2"]["verdict"] == "redundant"
    assert rows["all_but_2"]["mean"] == rows["step_1"]["mean"]

    html = ablation.get_tabular_report().as_raw_html()
    assert "redundant" in html
    assert "Step 1" in html


def test_ablation_drop_all_but_each_without_control():
    ablation = (
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=4, control=None)
        .drop_each()
        .drop_all_but_each()
        .run()
    )
    ids = [v["id"] for v in ablation.to_dict()["variants"]]
    assert not any(i.startswith("control_") for i in ids)

    rows = _effects_by_variant(ablation)
    # Without a control, the added effect is every step's rows compared with the other steps'
    assert rows["all_but_1"]["delta_added"] == pytest.approx(
        rows["all"]["mean"] - rows["all_but_1"]["mean"]
    )
    assert rows["all_but_1"]["verdict"] == "helps"
    assert "only the other steps" in ablation.get_tabular_report().as_raw_html()


def test_ablation_drop_all_but_each_with_steps():
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=3).drop_all_but_each(
        steps=[2, 1]
    )
    ids = [v["id"] for v in ablation.run().to_dict()["variants"]]
    assert ids[:4] == ["full", "all_1_2", "all_but_1_of_1_2", "all_but_2_of_1_2"]
    names = {v["id"]: v["name"] for v in ablation.to_dict()["variants"]}
    assert names["all_but_2_of_1_2"] == "Steps 1, 2 but step 2"

    with pytest.raises(ValueError, match="at least two steps"):
        pb.Ablation(_validation(_TBL), evaluate=_evaluate).drop_all_but_each(steps=[1])
    validation = pb.Validate(data=_TBL).col_vals_ge(columns="x", value=0).interrogate()
    with pytest.raises(ValueError, match="at least two steps"):
        pb.Ablation(validation, evaluate=_evaluate).drop_all_but_each()


def test_ablation_drop_all_but_each_left_out_step_without_rows():
    # Only step 1's rows fail, so leaving it out removes no rows
    tbl = _TBL.with_columns(y=pl.col("y").fill_null(0.0), g=pl.lit("plain"))
    validation = (
        pb.Validate(data=tbl)
        .col_vals_ge(columns="x", value=0)
        .col_vals_not_null(columns="y")
        .interrogate()
    )
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).drop_all_but_each()
    ablation.run()
    data = ablation.to_dict()

    # Leaving out step 1 is the same as the full table, and leaving out step 2 is the same as
    # removing every step's rows
    assert data["same_as"] == {"all_but_1": "full", "all_but_2": "all"}
    rows = _effects_by_variant(ablation)
    assert rows["all_but_1"]["n_removed"] == 0
    assert rows["all_but_1"]["delta_added"] > 5
    # The note about `all_but_2` names the variant it's the same as
    html = ablation.get_tabular_report().as_raw_html()
    assert "Same rows as Full table, so it isn't evaluated again" in html
    assert "Same rows as All steps, so it isn't evaluated again" in html


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_ablation_drop_groups(tbl_type):
    tbl = _tbl(tbl_type)
    ablation = pb.Ablation(_validation(tbl), evaluate=_evaluate, repeats=3).drop_groups("g")
    ablation.run()
    variants = {v["id"]: v for v in ablation.to_dict()["variants"]}

    assert variants["g=gold"]["name"] == "g = gold"
    assert variants["g=gold"]["group"] == "gold"
    assert variants["g=gold"]["group_column"] == "g"
    assert variants["g=gold"]["n_removed"] == 10
    # The 'plain' group has 90 of the 100 rows, more than `max_removed=` allows without a warning
    # (the warning is given when the ablation is first resolved, here by `run()`)
    assert variants["g=plain"]["n_removed"] == 90

    rows = _effects_by_variant(ablation)
    assert rows["g=gold"]["verdict"] == "hurts"
    assert rows["g=gold"]["steps"] == []


def test_ablation_drop_groups_values():
    tbl = _TBL.with_columns(
        src=pl.when(pl.col("rid") < 30)
        .then(pl.lit("web"))
        .when(pl.col("rid") < 60)
        .then(pl.lit("app"))
        .otherwise(None)
    )
    validation = pb.Validate(data=tbl).col_vals_ge(columns="x", value=0).interrogate()

    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).drop_groups("src")
    variants = [(v["id"], v["name"], v["n_removed"]) for v in ablation.run().to_dict()["variants"]]
    # The values are sorted, with the missing values last
    assert variants[1:4] == [
        ("src=app", "src = app", 30),
        ("src=web", "src = web", 30),
        ("src=null", "src is missing", 40),
    ]

    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).drop_groups(
        "src", values=[None, "web"]
    )
    ids = [v["id"] for v in ablation.run().to_dict()["variants"]]
    assert ids[1:3] == ["src=null", "src=web"]

    # A value that isn't in the column removes no rows
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).drop_groups(
        "src", values="mail"
    )
    with pytest.raises(ValueError, match="None of the variants remove any rows"):
        ablation.run()
    assert ablation._resolve().notes == ["src = mail isn't used: it removes no rows."]

    # Integer groups
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).drop_groups(
        "rid", values=[0, 1]
    )
    assert [v["group"] for v in ablation.run().to_dict()["variants"][1:3]] == [0, 1]


def test_ablation_drop_groups_errors():
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate)
    with pytest.raises(TypeError, match="must be a column name"):
        ablation.drop_groups(["g"])
    with pytest.raises(ValueError, match="isn't in the table"):
        ablation.drop_groups("nope")
    with pytest.raises(ValueError, match="more than 50"):
        ablation.drop_groups("rid")
    with pytest.raises(ValueError, match="at least one value"):
        ablation.drop_groups("g", values=[])
    with pytest.raises(TypeError, match="must be single values"):
        ablation.drop_groups("g", values=[["gold"]])

    tbl = pl.DataFrame({"x": [1.0, -1.0, 2.0], "tags": [["a"], ["b"], ["a"]]})
    validation = pb.Validate(data=tbl).col_vals_ge(columns="x", value=0).interrogate()
    with pytest.raises(TypeError, match="can't be grouped by the column 'tags'"):
        pb.Ablation(validation, evaluate=_evaluate).drop_groups("tags")


# ----------------------------------------------------------------------------------------------
# Sweeps
# ----------------------------------------------------------------------------------------------

_SCORER_CALLS = []


def _x_score(batch):
    """A scorer of the rows' `x` (a hundredth of it), counting the rows it's given."""
    _SCORER_CALLS.append(len(batch))
    return [row["x"] / 100 for row in batch]


def _scored_validation(tbl, **kwargs):
    kwargs = {"min_val": 0, **kwargs}
    return (
        pb.Validate(data=tbl)
        .col_vals_ge(columns="x", value=0)
        .col_vals_scored(columns="x", scorer=_x_score, cache=False, **kwargs)
        .interrogate()
    )


def _sweep_points(ablation):
    return [
        (v["id"], v["strategy"], v["sweep"], v["value"], v["n_removed"])
        for v in ablation.to_dict()["variants"]
        if v["sweep"] is not None
    ]


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_ablation_sweep_score(tbl_type):
    tbl = _tbl(tbl_type)
    validation = _scored_validation(tbl)
    repeats = 3
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=repeats).sweep_score(
        2, quantiles=[0.1, 0.25]
    )
    ablation.run()
    assert _sweep_points(ablation) == [
        ("step_2:lowest_0.1", "sweep_score", "step_2:lowest", 0.1, 10),
        ("step_2:lowest_0.25", "sweep_score", "step_2:lowest", 0.25, 25),
    ]
    rows = _effects_by_variant(ablation)
    # The 10 lowest scores are the rows with a negative `x`, which hurt the model
    assert rows["step_2:lowest_0.1"]["verdict"] == "helps"
    assert rows["step_2:lowest_0.1"]["sweep"] == "step_2:lowest"
    assert rows["step_2:lowest_0.1"]["value"] == 0.1
    assert rows["step_2:lowest_0.1"]["steps"] == []
    assert rows["step_2:lowest_0.1"]["name"] == "Step 2: lowest 10% of scores"

    # The removed rows are those with the lowest scores (here, of `x`)
    removed = ablation._resolve().removed
    xs = _TBL["x"].to_list()
    assert sorted(xs[k] for k in removed["step_2:lowest_0.1"]) == [-1.0] * 10
    assert max(xs[k] for k in removed["step_2:lowest_0.25"]) == 16.0


def test_ablation_sweep_score_values_and_direction():
    validation = _scored_validation(_TBL)
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3)
    ablation.sweep_score(2, values=[0, 0.05])
    ablation.sweep_score(2, values=[0.95], remove="highest")
    ablation.sweep_score(2, quantiles=[0.05], remove="highest")
    assert _sweep_points(ablation.run()) == [
        ("step_2:below_0", "sweep_score", "step_2:below", 0.0, 10),
        ("step_2:below_0.05", "sweep_score", "step_2:below", 0.05, 14),
        ("step_2:above_0.95", "sweep_score", "step_2:above", 0.95, 4),
        ("step_2:highest_0.05", "sweep_score", "step_2:highest", 0.05, 5),
    ]
    names = {v.id: v.name for v in ablation._variants}
    assert names["step_2:below_0.05"] == "Step 2: scores below 0.05"
    assert names["step_2:highest_0.05"] == "Step 2: highest 5% of scores"
    # Removing the highest scores of 95 to 99 is the same as removing the top 5%, minus row 95
    assert ablation._resolve().same_as == {}

    # With only `max_val=`, a high score is bad, so the highest scores are removed by default
    validation = _scored_validation(_TBL, min_val=None, max_val=0.9)
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).sweep_score(
        2, quantiles=[0.1]
    )
    assert [spec.id for spec in ablation._specs] == ["step_2:highest_0.1"]
    xs = _TBL["x"].to_list()
    assert sorted(xs[k] for k in ablation._resolve().removed["step_2:highest_0.1"]) == [
        float(k) for k in range(89, 100) if k % 10
    ]


def test_ablation_sweep_score_ties_and_rounding():
    tbl = pl.DataFrame({"s": [0.5, 0.1, 0.5, 0.1, 0.9, None, 0.3, 0.7], "x": [1.0] * 8})
    validation = pb.Validate(data=tbl).col_vals_ge(columns="x", value=0).interrogate()
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).sweep_score(
        "s", quantiles=[0.2, 0.5, 1]
    )
    removed = {spec.id: sorted(ablation._sweep_rows[spec.id]) for spec in ablation._specs}
    # Of the 7 scored rows, 20% is 1.4 rows (1 row), and ties are broken by row order
    assert removed == {
        "s:lowest_0.2": [1],
        "s:lowest_0.5": [0, 1, 3, 6],
        "s:lowest_1": [0, 1, 2, 3, 4, 6, 7],
    }
    assert ablation._notes == [
        "1 row has no score in the column `s`, so the sweep never removes it."
    ]
    details = ablation._sweep_details
    assert details["s:lowest_0.5"] == "The 4 rows with the lowest scores (of 7 scored rows)"


def test_ablation_sweep_score_column_with_missing_values(tmp_path):
    validation = _validation(_TBL)
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).sweep_score("y", values=[10])
    ablation.run()
    # The rows with a missing `y` are never removed
    removed = ablation._resolve().removed["y:below_10"]
    assert len(removed) == 18
    assert all(_TBL["y"][k] is not None for k in removed)
    assert ablation._notes == [
        "10 rows have no score in the column `y`, so the sweep never removes them."
    ]

    # A pandas table with NaN for the missing values
    tbl = _TBL.to_pandas()
    validation = _validation(tbl)
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).sweep_score("y", values=[10])
    assert len(ablation._resolve().removed["y:below_10"]) == 18


def test_ablation_sweep_score_with_a_segmented_step():
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_scored(columns="x", scorer=_x_score, min_val=0, segments="g", cache=False)
        .interrogate()
    )
    # Step 1 is the 'gold' segment, step 2 the 'plain' one
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).sweep_score(
        2, quantiles=[0.5]
    )
    removed = ablation._resolve().removed["step_2:lowest_0.5"]
    assert len(removed) == 45
    assert all(_TBL["g"][k] == "plain" for k in removed)
    assert ablation._notes == ["10 rows have no score in step 2, so the sweep never removes them."]


def test_ablation_sweep_score_same_rows_and_no_rows():
    validation = _scored_validation(_TBL)
    ablation = (
        pb.Ablation(validation, evaluate=_evaluate, repeats=3)
        .drop_each()
        .sweep_score(2, values=[-0.5, 0, 0.001])
    )
    ablation.run()
    resolved = ablation._resolve()
    # Removing the scores below 0 removes the rows that fail step 1 (and step 2), and so does
    # removing those below 0.001, so they aren't evaluated again
    assert resolved.same_as == {"step_2:below_0": "step_1", "step_2:below_0.001": "step_1"}
    assert "Step 2: scores below -0.5 isn't used: it removes no rows." in resolved.notes
    rows = _effects_by_variant(ablation)
    assert rows["step_2:below_0"]["mean"] == rows["step_1"]["mean"]

    # A point that removes every row isn't used
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).sweep_score(
        2, quantiles=[0.5, 1]
    )
    assert "Step 2: lowest 100% of scores isn't used: it removes every row." in (
        ablation._resolve().notes
    )


def test_ablation_sweep_score_errors():
    validation = _scored_validation(_TBL)
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3)
    with pytest.raises(TypeError, match=r"must be the number of a `col_vals_scored\(\)` step"):
        ablation.sweep_score(True)
    with pytest.raises(TypeError, match=r"must be the number of a `col_vals_scored\(\)` step"):
        ablation.sweep_score(2.0)
    with pytest.raises(ValueError, match="either `quantiles=` or `values=`"):
        ablation.sweep_score(2, quantiles=[0.1], values=[0])
    with pytest.raises(ValueError, match="must be 'lowest' or 'highest'"):
        ablation.sweep_score(2, remove="low")
    with pytest.raises(ValueError, match=r"isn't a `col_vals_scored\(\)` step"):
        ablation.sweep_score(1)
    with pytest.raises(ValueError, match="the validation has no step 7"):
        ablation.sweep_score(7)
    with pytest.raises(ValueError, match="The column 'nope' isn't in the table"):
        ablation.sweep_score("nope")
    with pytest.raises(TypeError, match="must be numeric to be used as a score"):
        ablation.sweep_score("g")
    with pytest.raises(ValueError, match="greater than 0 and at most 1"):
        ablation.sweep_score(2, quantiles=[0])
    with pytest.raises(ValueError, match="greater than 0 and at most 1"):
        ablation.sweep_score(2, quantiles=[1.5])
    with pytest.raises(TypeError, match="must be a list of numbers"):
        ablation.sweep_score(2, values=0.5)
    with pytest.raises(ValueError, match="at least one value"):
        ablation.sweep_score(2, values=[])
    with pytest.raises(TypeError, match="must be finite numbers"):
        ablation.sweep_score(2, values=[True])
    with pytest.raises(TypeError, match="must be finite numbers"):
        ablation.sweep_score(2, values=[float("nan")])
    with pytest.raises(ValueError, match="different from each other"):
        ablation.sweep_score(2, values=[0.1, 0.1])
    assert ablation._specs == []

    # A step that didn't keep its scores
    validation = _scored_validation(_TBL, keep_scores=False)
    with pytest.raises(ValueError, match="kept its scores"):
        pb.Ablation(validation, evaluate=_evaluate).sweep_score(2)


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_ablation_sweep_step(tbl_type):
    tbl = _tbl(tbl_type)
    ablation = pb.Ablation(_validation(tbl), evaluate=_evaluate, repeats=3).sweep_step(
        1, param="value", values=[0, 5, 30]
    )
    ablation.run()
    assert _sweep_points(ablation) == [
        ("step_1:value=0", "sweep_step", "step_1:value", 0.0, 10),
        ("step_1:value=5", "sweep_step", "step_1:value", 5.0, 14),
        ("step_1:value=30", "sweep_step", "step_1:value", 30.0, 37),
    ]
    rows = _effects_by_variant(ablation)
    assert rows["step_1:value=0"]["verdict"] == "helps"
    assert rows["step_1:value=0"]["name"] == "Step 1 with value=0"
    assert ablation._sweep_details["step_1:value=5"] == ("The rows that fail step 1 with `value=5`")


def test_ablation_sweep_step_of_a_scored_step_uses_the_cache():
    pb.scorers.clear_cache()
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_scored(columns="x", scorer=_x_score, min_val=0, thresholds=0.1, brief="Scored")
        .interrogate()
    )
    _SCORER_CALLS.clear()
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).sweep_step(
        1, param="min_val", values=[0, 0.1, 0.5]
    )
    assert [len(ablation._sweep_rows[spec.id]) for spec in ablation._specs] == [10, 19, 55]
    # The scores came from the score cache
    assert _SCORER_CALLS == []
    # Running the step again with `min_val=` above `max_val=` fails
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_scored(columns="x", scorer=_x_score, min_val=0, max_val=0.5)
        .interrogate()
    )
    with pytest.raises(ValueError, match="Step 1 can't be run with `min_val=0.9`: `min_val=`"):
        pb.Ablation(validation, evaluate=_evaluate).sweep_step(1, "min_val", [0.9])


def test_ablation_sweep_step_with_segments_and_pre():
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_ge(columns="x", value=0, segments="g")
        .col_vals_ge(columns="x", value=0, pre=pb.row_preserving(lambda df: df))
        .interrogate()
    )
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).sweep_step(
        2, param="value", values=[20]
    )
    # Only the rows of the step's segment ('plain') are checked
    removed = ablation._sweep_rows["step_2:value=20"]
    assert len(removed) == 10 + 8 + 8
    assert all(_TBL["g"][k] == "plain" for k in removed)

    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).sweep_step(
        3, param="value", values=[1.5]
    )
    assert len(ablation._sweep_rows["step_3:value=1.5"]) == 11


def test_ablation_sweep_step_errors():
    validation = _validation(_TBL)
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3)
    with pytest.raises(TypeError, match="must be a step number"):
        ablation.sweep_step("1", "value", [1])
    with pytest.raises(ValueError, match="can't use step 4: it doesn't check rows"):
        ablation.sweep_step(4, "value", [1])
    with pytest.raises(TypeError, match="must be the name of a parameter"):
        ablation.sweep_step(1, None, [1])
    for param in ("thresholds", "pre", "segments", "self", "nope"):
        with pytest.raises(ValueError, match=f"can't vary '{param}'"):
            ablation.sweep_step(1, param, [1])
    with pytest.raises(TypeError, match="must be finite numbers"):
        ablation.sweep_step(1, "value", ["a"])
    with pytest.raises(ValueError, match="Step 1 can't be run with `columns=1`"):
        ablation.sweep_step(1, "columns", [1])
    assert ablation._specs == []


def test_ablation_sweep_reports():
    validation = _scored_validation(_TBL)
    ablation = (
        pb.Ablation(validation, evaluate=_evaluate, repeats=3)
        .drop_each(steps=[1])
        .sweep_score(2, quantiles=[0.1, 0.2, 0.3])
        .sweep_step(1, param="value", values=[5])
    )
    plan = ablation.plan().as_raw_html()
    assert "Sweep: Step 2, removing the lowest-scoring rows" in plan
    assert "Sweep: Step 1, varying value" in plan
    assert "The 20 rows with the lowest scores (of 100 scored rows)" in plan
    assert "The rows that fail step 1 with <code>value=5</code>" in plan
    # The groups are in order: the variants, then each sweep
    assert plan.index(">Variants<") < plan.index("Sweep: Step 2") < plan.index("Sweep: Step 1")

    ablation.run()
    report = ablation.get_tabular_report().as_raw_html()
    assert report.count("<svg width='460'") == 2
    assert "Removing as many random rows" in report
    assert "Share of the scored rows removed" in report
    assert ">10%<" in report and ">30%<" in report
    assert "value (step 1)" in report
    # The sweeps aren't used to annotate the validation
    annotated = ablation.annotate()
    assert "ablation" in annotated.validation_info[0].notes
    assert "ablation" not in (annotated.validation_info[1].notes or {})

    # Without a control, there's no dashed line
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3, control=None).sweep_score(
        2, quantiles=[0.1]
    )
    report = ablation.run().get_tabular_report().as_raw_html()
    assert report.count("<svg width='460'") == 1
    assert "Removing as many random rows" not in report

    # The effects, and the JSON of the results
    effects = ablation.get_effects()
    assert effects["sweep"].to_list() == [None, "step_2:lowest"]
    assert effects["value"].to_list() == [None, 0.1]
    variants = json.loads(ablation.to_json())["variants"]
    assert variants[1]["sweep"] == "step_2:lowest" and variants[1]["value"] == 0.1


def test_sweep_svg_with_one_point_and_no_intervals():
    from pointblank.ablation import _sweep_svg

    sweep = {"label": "S", "axis": "A", "percent": False}
    row = {
        "value": 1.0,
        "name": "P",
        "delta_vs_full": 0.0,
        "delta_vs_full_lower": None,
        "delta_vs_full_upper": None,
        "control_mean": 1.0,
        "verdict": None,
    }
    svg = _sweep_svg(sweep, [row], baseline=1.0, fmt=_NumberFormat([1.0]), metric="m", direction="")
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "stroke-dasharray='4,3'" in svg


def test_nice_ticks():
    from pointblank.ablation import _nice_ticks

    assert _nice_ticks(-10.0, 1.2) == [-10.0, -5.0, 0.0, 5.0]
    assert _nice_ticks(0.0, 1.0) == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert 0.0 in _nice_ticks(-0.013, 0.004)


def test_yaml_ablation_sweeps():
    config = """
tbl: small_table
steps:
- col_vals_ge:
    columns: x
    value: 0
ablation:
  repeats: 3
  strategies:
  - sweep_step:
      step: 1
      param: value
      values: [0, 5]
  - sweep_score:
      score: x
      quantiles: [0.1]
"""
    ablation = pb.yaml_ablation(config, evaluate=_evaluate, set_tbl=_TBL)
    assert [spec.id for spec in ablation._specs] == [
        "step_1:value=0",
        "step_1:value=5",
        "x:lowest_0.1",
    ]
    code = pb.yaml_to_python(config)
    assert '    .sweep_step(step=1, param="value", values=[0, 5])' in code
    assert '    .sweep_score(score="x", quantiles=[0.1])' in code

    bad = config.replace("      param: value\n", "")
    with pytest.raises(
        pb.yaml.YAMLValidationError, match="needs the parameter\\(s\\) \\['param'\\]"
    ):
        pb.yaml_ablation(bad, evaluate=_evaluate, set_tbl=_TBL)


# ----------------------------------------------------------------------------------------------
# Stratified controls
# ----------------------------------------------------------------------------------------------


def test_allocate():
    from pointblank.ablation import _allocate

    assert _allocate(10, [50, 30, 20]) == [5, 3, 2]
    assert _allocate(13, [200, 100]) == [9, 4]
    # The largest remainders get the extra rows, then the first groups
    assert _allocate(2, [1, 1, 1]) == [1, 1, 0]
    assert _allocate(5, [10, 0, 10]) == [3, 0, 2]
    assert _allocate(0, [3, 4]) == [0, 0]
    assert _allocate(3, [0, 0]) == [0, 0]
    assert sum(_allocate(7, [13, 29, 1, 57])) == 7


def test_stratified():
    from pointblank.ablation import Stratified

    assert pb.ablation.Stratified is Stratified
    assert Stratified("g").by == ("g",)
    assert Stratified(by=["g", "y"]).by == ("g", "y")
    assert repr(Stratified("g")) == "Stratified(by='g')"
    assert repr(Stratified(["g", "y"])) == "Stratified(by=['g', 'y'])"
    assert Stratified("g") == Stratified(["g"])

    for by in ([], "", 1, ["g", 2], None):
        with pytest.raises(TypeError, match="must be a column name or a list"):
            Stratified(by)
    with pytest.raises(ValueError, match="must be different"):
        Stratified(["g", "g"])
    with pytest.raises(ValueError, match="columns aren't in the table: 'nope'"):
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, control=Stratified(["g", "nope"]))


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_ablation_stratified_control(tbl_type):
    from pointblank.ablation import Stratified

    tbl = _tbl(tbl_type)
    ablation = pb.Ablation(
        _validation(tbl), evaluate=_evaluate, repeats=3, control=Stratified("g")
    ).drop_each(steps=[1, 3])
    ablation.run()

    # 10 of the 100 rows are 'gold', so a control of 10 rows always removes exactly one of them
    gold = {k for k in range(_N) if k % 10 == 1}
    for r in range(3):
        drawn = ablation._control_rows("control_10", r)
        assert len(drawn) == 10
        assert len(drawn & gold) == 1
    assert ablation.to_dict()["settings"]["control"] == {"stratified": ["g"]}
    assert "stratified by g" in ablation.get_tabular_report().as_raw_html()
    assert repr(ablation).endswith("control=Stratified(by='g'); run)")


def test_ablation_stratified_control_draws():
    from pointblank.ablation import Stratified

    unstratified = pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=3).drop_each(1)
    stratified = pb.Ablation(
        _validation(_TBL), evaluate=_evaluate, repeats=3, control=Stratified(["g", "y"])
    ).drop_each(1)
    two = pb.Ablation(
        _validation(_TBL), evaluate=_evaluate, repeats=3, control=Stratified("g")
    ).drop_each(1)
    for ablation in (unstratified, stratified, two):
        ablation._resolve()
    # The draws of a stratified control differ from the unstratified ones, and depend on the
    # columns, but each is reproducible
    draws = [a._control_rows("control_10", 0) for a in (unstratified, stratified, two)]
    assert len(set(draws)) == 3
    assert stratified._control_rows("control_10", 0) == draws[1]

    # Missing values are a group of their own
    strata = stratified._strata_rows()
    assert sum(len(s) for s in strata) == _N
    assert sorted(k for s in strata for k in s) == list(range(_N))

    # A left-out variant's control draws its random rows in proportion to the rows it keeps
    ablation = pb.Ablation(
        _validation(_TBL), evaluate=_evaluate, repeats=3, control=Stratified("g")
    ).drop_all_but_each()
    ablation._resolve()
    removed = ablation._resolve().removed["all_but_1"]
    drawn = ablation._control_rows("control_all_but_1", 0)
    assert removed < drawn and len(drawn) == 30

    tbl = pl.DataFrame({"x": [1.0, -1.0, 2.0, 3.0], "tags": [["a"], ["b"], ["a"], []]})
    validation = pb.Validate(data=tbl).col_vals_ge(columns="x", value=0).interrogate()
    ablation = pb.Ablation(
        validation, evaluate=_evaluate, repeats=3, control=Stratified("tags")
    ).drop_each()
    with pytest.raises(TypeError, match="can't be grouped by the columns of the `Stratified`"):
        ablation._control_rows("control_1", 0)


# ----------------------------------------------------------------------------------------------
# Running in parallel and the cache
# ----------------------------------------------------------------------------------------------


def test_ablation_run_in_threads():
    def make():
        return (
            pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=4)
            .drop_each()
            .drop_all_but_each()
        )

    sequential = make().run().to_dict()
    threaded = make().run(max_workers=4).to_dict()
    assert threaded == sequential
    assert make().run(max_workers=1).to_dict() == sequential


@pytest.mark.parametrize("tbl_type", ["polars", "pandas"])
def test_ablation_run_in_processes(tbl_type):
    tbl = _tbl(tbl_type)

    def make():
        return pb.Ablation(_validation(tbl), evaluate=_evaluate, repeats=3).drop_each()

    assert make().run(max_workers=2, executor="process").to_dict() == make().run().to_dict()


def test_ablation_run_argument_errors(tmp_path):
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=3).drop_each()
    for max_workers in (0, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="`max_workers=` must be a positive integer"):
            ablation.run(max_workers=max_workers)
    with pytest.raises(ValueError, match="`executor=` must be one of 'thread', 'process'"):
        ablation.run(executor="cluster")

    # A process pool needs a function that can be pickled and a table that can be sent
    lambda_ablation = pb.Ablation(
        _validation(_TBL), evaluate=lambda train, *, seed: 1.0, repeats=3
    ).drop_each()
    with pytest.raises(TypeError, match="can be sent to another process"):
        lambda_ablation.run(max_workers=2, executor="process")
    # ... which isn't checked when the evaluations aren't made in parallel
    lambda_ablation.run(executor="process")

    ablation = pb.Ablation(_validation(_tbl("duckdb")), evaluate=_evaluate, repeats=3).drop_each()
    with pytest.raises(ValueError, match="for Polars, pandas, and PyArrow tables"):
        ablation.run(max_workers=2, executor="process")

    no_evaluate = pb.Ablation(_validation(_TBL), repeats=3).drop_each()
    with pytest.raises(ValueError, match="There's no `evaluate=` function"):
        no_evaluate.run()


def test_ablation_run_in_threads_errors(tmp_path):
    calls = []

    def evaluate(train, *, seed, variant):
        calls.append(variant.id)
        if variant.id == "step_2" and seed == 1:
            raise RuntimeError("training failed")
        return _evaluate(train, seed=seed, variant=variant)

    ablation = pb.Ablation(
        _validation(_TBL), evaluate=evaluate, repeats=3, cache=tmp_path / "cache"
    ).drop_each()
    with pytest.raises(RuntimeError, match="training failed"):
        ablation.run(max_workers=2)
    assert ablation._effects is None
    n_cached = len(list((tmp_path / "cache").glob("*.json")))
    assert 0 < n_cached < len(ablation._tasks())

    # Running again only makes the evaluations that aren't in the cache
    calls.clear()

    def fixed(train, *, seed, variant):
        calls.append(variant.id)
        return _evaluate(train, seed=seed, variant=variant)

    ablation.evaluate = fixed
    ablation.run(max_workers=2)
    assert len(calls) == len(ablation._tasks()) - n_cached
    assert sum(run["cached"] for run in ablation.to_dict()["runs"]) == n_cached

    # Bad results from a thread are checked as they are without threads
    bad = pb.Ablation(
        _validation(_TBL), evaluate=lambda train, *, seed: "high", repeats=3
    ).drop_each()
    with pytest.raises(TypeError, match="must return a number or a dictionary"):
        bad.run(max_workers=2)


def test_ablation_cache(tmp_path):
    cache = tmp_path / "cache"
    calls = []

    def evaluate(train, *, seed, variant):
        calls.append((variant.id, seed))
        return {"score": _evaluate(train, seed=seed, variant=variant), "n": len(train)}

    def make(**kwargs):
        return pb.Ablation(
            _validation(_TBL), evaluate=evaluate, metric="score", repeats=3, cache=cache, **kwargs
        ).drop_each()

    first = make().run()
    n_tasks = len(first._tasks())
    assert len(calls) == n_tasks
    assert len(list(cache.glob("*.json"))) == n_tasks
    assert not any(run["cached"] for run in first.to_dict()["runs"])
    assert "from the cache" not in first.get_tabular_report().as_raw_html()

    # A new ablation with the same data, versions, and seeds gets every result from the cache
    calls.clear()
    plan = make().plan().as_raw_html()
    assert f"{n_tasks} of them are already in the cache" in plan
    second = make().run()
    assert calls == []
    assert all(run["cached"] for run in second.to_dict()["runs"])
    assert second.get_effects().equals(first.get_effects())
    assert f"{n_tasks} of the {n_tasks} evaluations came from the cache" in (
        second.get_tabular_report().as_raw_html()
    )
    # ... and so does a run in threads
    third = make().run(max_workers=3)
    assert calls == [] and third.get_effects().equals(first.get_effects())

    # A different `evaluate_version=` doesn't use the results of another version
    make(evaluate_version="v2").run()
    assert len(calls) == n_tasks
    assert len(list(cache.glob("*.json"))) == 2 * n_tasks

    # A cache entry that can't be read is evaluated again and replaced
    calls.clear()
    entries = sorted(cache.glob("*.json"))
    entries[0].write_text("not json")
    entries[1].write_text(json.dumps({"result": "high"}))
    make().run()
    make(evaluate_version="v2").run()
    assert len(calls) == 2
    assert json.loads(entries[0].read_text())["result"] is not None

    # A different seed or table isn't in the cache
    calls.clear()
    make(seed=1).run()
    # The repeats with the seeds 1 and 2 are in the cache, except for the control: its rows are
    # drawn for this ablation's repeats, so they're new
    assert len(calls) == 4 + 3
    calls.clear()
    other = _TBL.with_columns(y=pl.col("y") * 2)
    pb.Ablation(
        _validation(other), evaluate=evaluate, metric="score", repeats=3, cache=cache
    ).drop_each().run()
    assert len(calls) == n_tasks


def test_ablation_cache_shares_identical_versions(tmp_path):
    calls = []

    def evaluate(train, *, seed, variant):
        calls.append(variant.id)
        return _evaluate(train, seed=seed, variant=variant)

    # Steps 1 and 4 fail the same rows, but their variants have different names
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_ge(columns="x", value=0)
        .col_vals_gt(columns="x", value=-0.5)
        .interrogate()
    )
    ablation = (
        pb.Ablation(validation, evaluate=evaluate, repeats=3, cache=tmp_path, control=None)
        .drop_steps(1, name="first")
        .drop_steps(2, name="second")
    )
    ablation.run(max_workers=2)
    # The second variant's evaluations are the same as the first's (same rows and seeds)
    assert calls.count("first") + calls.count("second") == 3
    runs = ablation.to_dict()["runs"]
    assert sum(run["cached"] for run in runs) == 3


def test_ablation_cache_errors(tmp_path):
    with pytest.raises(TypeError, match="`cache=` must be the path of a directory"):
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, cache=1)
    file = tmp_path / "file.txt"
    file.write_text("x")
    with pytest.raises(ValueError, match="`cache=` must be a directory"):
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, cache=file)
    with pytest.raises(TypeError, match="`evaluate_version=` must be a string"):
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, evaluate_version=2)

    ablation = pb.Ablation(
        _validation(_TBL), evaluate=_evaluate, repeats=3, cache=str(tmp_path / "new" / "cache")
    ).drop_each(1)
    ablation.run()
    assert len(list((tmp_path / "new" / "cache").glob("*.json"))) == 9
    assert not list((tmp_path / "new" / "cache").glob("*.tmp"))


def test_ablation_evaluate_version_in_report():
    ablation = pb.Ablation(
        _validation(_TBL), evaluate=_evaluate, repeats=3, evaluate_version="2024-06 (xgb 2.0)"
    ).drop_each(1)
    html = ablation.run().get_tabular_report().as_raw_html()
    assert "Version of <code>evaluate</code>: 2024-06 (xgb 2.0)" in html
    assert ablation.to_dict()["settings"]["evaluate_version"] == "2024-06 (xgb 2.0)"


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
def test_table_fingerprint(tbl_type):
    from pointblank.ablation import _table_fingerprint

    tbl = _tbl(tbl_type)
    fingerprint = _table_fingerprint(tbl)
    assert fingerprint == _table_fingerprint(tbl)
    assert len(fingerprint) == 64
    if tbl_type in ("polars", "lazy"):
        # A lazy table has the fingerprint of the table it collects to
        assert fingerprint == _table_fingerprint(_TBL)
        assert fingerprint != _table_fingerprint(_TBL.with_columns(x=pl.col("x") + 1))
        assert fingerprint != _table_fingerprint(_TBL.reverse())
        assert fingerprint != _table_fingerprint(_TBL.rename({"y": "z"}))
    if tbl_type == "pandas":
        assert fingerprint != _table_fingerprint(tbl.assign(x=tbl["x"] + 1))
        # The row labels don't matter
        assert fingerprint == _table_fingerprint(tbl.set_axis(range(100, 200)))


def test_table_fingerprint_unhashable_values():
    from pointblank.ablation import _table_fingerprint

    tbl = pd.DataFrame({"x": [1, 2], "tags": [["a"], ["b", "c"]]})
    assert _table_fingerprint(tbl) == _table_fingerprint(tbl.copy())
    assert _table_fingerprint(tbl) != _table_fingerprint(tbl.assign(tags=[["a"], ["b"]]))


# ----------------------------------------------------------------------------------------------
# Checking the evaluate function
# ----------------------------------------------------------------------------------------------


def test_ablation_check_determinism():
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ablation.check_determinism() is True
        assert ablation.check_determinism(seed=12) is True

    state = {"calls": 0}

    def noisy(train, *, seed):
        state["calls"] += 1
        return {"score": float(state["calls"]), "n": 1.0}

    ablation = pb.Ablation(_validation(_TBL), evaluate=noisy, metric="score")
    with pytest.warns(
        UserWarning, match=r"isn't deterministic: .* the seed 0 twice.*\(score: 1.0 and 2.0\)"
    ):
        assert ablation.check_determinism() is False
    # A difference within the tolerance counts as the same
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ablation.check_determinism(tolerance=1.5) is True

    # Rounding differences count as the same, even without a tolerance
    def rounding(train, *, seed):
        state["calls"] += 1
        return 9.155649164204585 if state["calls"] % 2 else 9.155649164204583

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert pb.Ablation(_validation(_TBL), evaluate=rounding).check_determinism() is True

    def changing_metrics(train, *, seed):
        state["calls"] += 1
        return {f"m{state['calls']}": 1.0}

    state["calls"] = 0
    ablation = pb.Ablation(_validation(_TBL), evaluate=changing_metrics, metric="m1")
    with pytest.warns(UserWarning, match="the first time and"):
        assert ablation.check_determinism() is False

    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate)
    with pytest.raises(TypeError, match="`seed=` must be an integer"):
        ablation.check_determinism(seed="1")
    for tolerance in (-1, "0", None):
        with pytest.raises(ValueError, match="`tolerance=` must be a number"):
            ablation.check_determinism(tolerance=tolerance)
    with pytest.raises(ValueError, match="There's no `evaluate=` function to check"):
        pb.Ablation(_validation(_TBL)).check_determinism()
    with pytest.raises(TypeError, match="must return a number"):
        pb.Ablation(_validation(_TBL), evaluate=lambda train, *, seed: None).check_determinism()


# ----------------------------------------------------------------------------------------------
# Exporting and loading results
# ----------------------------------------------------------------------------------------------


def _score_exported(directory, metric="score", format="parquet"):
    """Evaluate each exported version of the table, as an external process would."""
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = []
    for run in manifest["runs"]:
        if format == "parquet":
            table = pl.read_parquet(directory / run["file"])
        else:
            payload = json.loads((directory / run["file"]).read_text())
            assert payload["n_rows"] == _N
            table = _TBL.filter(~pl.col("rid").is_in(payload["removed"]))
        rows.append(
            {
                "variant": run["variant"],
                "repeat": run["repeat"],
                "seed": run["seed"],
                metric: _evaluate(table, seed=run["seed"], variant=None),
            }
        )
    return pl.DataFrame(rows)


@pytest.mark.parametrize("tbl_type", _TBL_TYPES)
@pytest.mark.parametrize("format", ["parquet", "indices"])
def test_ablation_export_and_load_results(tmp_path, tbl_type, format):
    tbl = _tbl(tbl_type)

    def make(evaluate=None):
        return (
            pb.Ablation(_validation(tbl), evaluate=evaluate, repeats=3, label="v1")
            .drop_each()
            .drop_all_but_each()
        )

    directory = make().export(tmp_path / "export", format=format)
    assert directory == tmp_path / "export"
    results = _score_exported(directory, format=format)
    results.write_csv(directory / "results.csv")

    loaded = make().load_results(directory)
    expected = make(evaluate=_evaluate).run()
    assert loaded.get_effects().equals(expected.get_effects())
    assert "read with <code>load_results()</code>" in loaded.get_tabular_report().as_raw_html()
    assert loaded.to_dict()["runs"] == expected.to_dict()["runs"]


def test_ablation_export_contents(tmp_path):
    ablation = (
        pb.Ablation(_validation(_TBL), metric="mae", higher_is_better=False, repeats=3, seed=7)
        .drop_each()
        .drop_steps(1, name="Bad X/rows")
    )
    directory = ablation.export(str(tmp_path))
    assert isinstance(directory, type(tmp_path))

    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["format"] == "pointblank-ablation"
    assert manifest["format_version"] == 1
    assert manifest["pointblank_version"] == pb.__version__
    assert manifest["table_format"] == "parquet"
    assert set(manifest["fingerprints"]) == {"data", "plan"}
    assert manifest["settings"]["metric"] == "mae"
    assert manifest["settings"]["seed"] == 7
    assert manifest["results"] == {
        "file": "results.csv",
        "template": "results_template.csv",
        "columns": ["variant", "repeat", "seed", "mae"],
    }
    assert "Bad X/rows isn't used: it removes the same rows as Step 1." in manifest["notes"]
    assert [v["id"] for v in manifest["variants"]] == [
        "full",
        "step_1",
        "step_2",
        "step_3",
        "control_10",
    ]

    # Each variant is one file for all of its repeats, and each repeat of a control has its own
    runs = manifest["runs"]
    assert len(runs) == 15
    assert runs[0] == {"variant": "full", "repeat": 0, "seed": 7, "file": "tables/full.parquet"}
    assert {run["file"] for run in runs if run["variant"] == "step_1"} == {"tables/step_1.parquet"}
    assert [run["file"] for run in runs if run["variant"] == "control_10"] == [
        "tables/control_10_r0.parquet",
        "tables/control_10_r1.parquet",
        "tables/control_10_r2.parquet",
    ]
    assert sorted(p.name for p in (directory / "tables").iterdir()) == [
        "control_10_r0.parquet",
        "control_10_r1.parquet",
        "control_10_r2.parquet",
        "full.parquet",
        "step_1.parquet",
        "step_2.parquet",
        "step_3.parquet",
    ]
    assert pl.read_parquet(directory / "tables" / "step_1.parquet").height == 90
    assert pl.read_parquet(directory / "tables" / "full.parquet").equals(_TBL)

    template = pl.read_csv(directory / "results_template.csv")
    assert template.columns == ["variant", "repeat", "seed", "mae"]
    assert template.height == 15
    assert template["mae"].null_count() == 15
    assert template.row(0) == ("full", 0, 7, None)

    # An export doesn't overwrite another one unless asked to
    with pytest.raises(FileExistsError, match="already an exported ablation"):
        ablation.export(directory)
    ablation.export(directory, format="indices", overwrite=True)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["table_format"] == "indices"
    payload = json.loads((directory / "tables" / "step_1.json").read_text())
    assert payload == {
        "variant": "step_1",
        "repeats": [0, 1, 2],
        "n_rows": 100,
        "removed": list(range(0, 100, 10)),
    }
    control = json.loads((directory / "tables" / "control_10_r1.json").read_text())
    assert control["repeats"] == [1]
    assert sorted(control["removed"]) == sorted(ablation._control_rows("control_10", 1))


def test_ablation_export_file_names(tmp_path):
    from pointblank.ablation import _unique_slugs

    assert _unique_slugs(["full", "a/b", "a b", "A_B", "..", "g=x"]) == {
        "full": "full",
        "a/b": "a_b",
        "a b": "a_b_2",
        "A_B": "A_B_3",
        "..": "variant",
        "g=x": "g_x",
    }

    tbl = _TBL.with_columns(
        src=pl.when(pl.col("rid") < 50).then(pl.lit("Web")).otherwise(pl.lit("web"))
    )
    validation = pb.Validate(data=tbl).col_vals_ge(columns="x", value=0).interrogate()
    ablation = pb.Ablation(validation, repeats=3, control=None).drop_groups("src")
    directory = ablation.export(tmp_path)
    files = sorted(p.name for p in (directory / "tables").iterdir())
    assert files == ["full.parquet", "src_Web.parquet", "src_web_2.parquet"]


def test_ablation_export_errors(tmp_path):
    ablation = pb.Ablation(_validation(_TBL), repeats=3)
    with pytest.raises(ValueError, match="There are no variants to evaluate"):
        ablation.export(tmp_path)
    ablation.drop_each()
    with pytest.raises(ValueError, match="`format=` must be one of 'parquet', 'indices'"):
        ablation.export(tmp_path, format="csv")
    with pytest.raises(TypeError, match="`path=` must be the path of a directory"):
        ablation.export(1)


def test_ablation_load_results_sources(tmp_path):
    def make():
        return pb.Ablation(_validation(_TBL), metric="score", repeats=3).drop_each(steps=[1, 2])

    directory = make().export(tmp_path / "export")
    results = _score_exported(directory)
    expected = pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=3).drop_each(steps=[1, 2])
    expected = expected.run().get_effects()

    # A table (Polars, pandas, or lazy), with or without a `repeat` column
    assert make().load_results(results).get_effects().equals(expected)
    assert make().load_results(results.to_pandas()).get_effects().equals(expected)
    assert make().load_results(results.drop("repeat").lazy()).get_effects().equals(expected)
    # A CSV file anywhere, with the columns in any order, and the manifest given
    results.select("score", "seed", "variant").write_csv(tmp_path / "scores.csv")
    loaded = make().load_results(tmp_path / "scores.csv", manifest=directory)
    assert loaded.get_effects().equals(expected)
    loaded = make().load_results(str(tmp_path / "scores.csv"), manifest=directory / "manifest.json")
    assert loaded.get_effects().equals(expected)
    # Missing values written in other ways and a byte-order mark
    text = (tmp_path / "scores.csv").read_text()
    (tmp_path / "bom.csv").write_text("﻿" + text, encoding="utf-8")
    assert make().load_results(tmp_path / "bom.csv").get_effects().equals(expected)

    # Several metrics
    two = results.with_columns(n=pl.lit(1.0))
    ablation = pb.Ablation(_validation(_TBL), metric="score", repeats=3).drop_each(steps=[1, 2])
    assert ablation.load_results(two).get_effects()["metric"].unique().sort().to_list() == [
        "n",
        "score",
    ]
    ablation = pb.Ablation(_validation(_TBL), repeats=3).drop_each(steps=[1, 2])
    with pytest.raises(ValueError, match="The results have several metrics"):
        ablation.load_results(two)
    ablation = pb.Ablation(_validation(_TBL), metric="auc", repeats=3).drop_each(steps=[1, 2])
    with pytest.raises(ValueError, match="The results don't have the metric 'auc'"):
        ablation.load_results(results)

    # Loading results replaces those of a run, and a run replaces loaded results
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=3).drop_each(steps=[1, 2])
    ablation.load_results(results)
    assert ablation._loaded
    ablation.run()
    assert not ablation._loaded
    assert "load_results" not in ablation.get_tabular_report().as_raw_html()


def test_ablation_load_results_checks_manifest(tmp_path):
    directory = (
        pb.Ablation(_validation(_TBL), metric="score", repeats=3).drop_each().export(tmp_path)
    )
    results = _score_exported(directory)
    results.write_csv(directory / "results.csv")

    # Different data
    other = _validation(_TBL.with_columns(y=pl.col("y") + 1))
    with pytest.raises(ValueError, match="The results are for different data"):
        pb.Ablation(other, metric="score", repeats=3).drop_each().load_results(directory)
    # A different plan
    for kwargs in ({"repeats": 3, "seed": 1}, {"repeats": 3, "control": None}):
        ablation = pb.Ablation(_validation(_TBL), metric="score", **kwargs).drop_each()
        with pytest.raises(ValueError, match="The results are for a different plan"):
            ablation.load_results(directory)
    ablation = pb.Ablation(_validation(_TBL), metric="score", repeats=3).drop_each(steps=[1, 2])
    with pytest.raises(ValueError, match="The results are for a different plan"):
        ablation.load_results(directory)
    # ... which the results themselves may not show, without the manifest
    with pytest.raises(ValueError, match="isn't one of the evaluations"):
        ablation.load_results(results)

    # Settings that don't change the evaluations can be different
    ablation = pb.Ablation(
        _validation(_TBL), metric="score", repeats=3, conf_level=0.9, label="other"
    ).drop_each()
    ablation.load_results(directory)

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "manifest.json").write_text("{")
    with pytest.raises(ValueError, match="isn't a valid manifest"):
        ablation.load_results(results, manifest=bad)
    (bad / "manifest.json").write_text('{"format": "other"}')
    with pytest.raises(ValueError, match="isn't a manifest written by `Ablation.export\\(\\)`"):
        ablation.load_results(results, manifest=bad / "manifest.json")
    with pytest.raises(FileNotFoundError, match="There's no manifest"):
        ablation.load_results(results, manifest=tmp_path / "nowhere")
    with pytest.raises(TypeError, match="`manifest=` must be the path"):
        ablation.load_results(results, manifest=1)


def test_ablation_load_results_errors(tmp_path):
    def make():
        return pb.Ablation(_validation(_TBL), metric="score", repeats=3).drop_each(steps=1)

    directory = make().export(tmp_path / "export")
    results = _score_exported(directory)
    n = results.height

    def check(frame, match, error=ValueError):
        with pytest.raises(error, match=match):
            make().load_results(frame)

    check(results.drop("variant"), "The results have no 'variant' column")
    check(results.drop("seed"), "The results have no 'seed' column")
    check(results.select("variant", "seed"), "The results have no metrics")
    check(
        results.with_columns(variant=pl.when(pl.int_range(n) == 2).then(None).otherwise("variant")),
        "Row 3 of the results has no variant",
    )
    check(
        results.with_columns(seed=pl.lit("x")),
        "Row 1 of the results has the seed 'x', which isn't an integer",
    )
    check(
        results.with_columns(seed=pl.col("seed") + 0.5),
        "Row 1 of the results has the seed 0.5, which isn't an integer",
    )
    check(
        results.with_columns(variant=pl.lit("step_9")),
        "Row 1 of the results is for 'step_9' with the seed 0, which isn't one",
    )
    check(
        pl.concat([results, results.head(1)]),
        "more than one row for 'full' with the seed 0",
    )
    check(
        results.with_columns(repeat=pl.col("repeat") + 1),
        "Row 1 of the results has the repeat 1, but the seed 0 is for repeat 0",
    )
    check(
        results.with_columns(score=pl.when(pl.int_range(n) == 4).then(None).otherwise("score")),
        "Row 5 of the results has no value for 'score'",
    )
    check(
        results.with_columns(
            score=pl.when(pl.int_range(n) == 0).then(float("inf")).otherwise("score")
        ),
        "Row 1 of the results has inf for 'score'",
    )
    check(
        results.with_columns(score=pl.lit("high")),
        "The results have 'high' as a metric, which isn't a number",
    )
    check(results.head(2), "missing 7 of the 9 evaluations: 'full' \\(seed 2\\), 'step_1'")
    check(
        results.head(0),
        "missing 9 of the 9 evaluations: .*, and 4 more\\.",
    )
    check(42, "`results=` must be the path of a CSV file", error=TypeError)
    check(tmp_path / "nowhere.csv", "There's no file of results", error=FileNotFoundError)
    check(tmp_path, "There's no file of results", error=FileNotFoundError)

    # Values in a CSV file: missing values written as text, and integers written as floats
    csv_results = results.with_columns(seed=pl.col("seed").cast(pl.Float64)).with_columns(
        score=pl.when(pl.int_range(n) == 1)
        .then(pl.lit("NA"))
        .otherwise(pl.col("score").cast(pl.String))
    )
    csv_results.write_csv(tmp_path / "results.csv")
    check(tmp_path, "Row 2 of the results has no value for 'score'")


def test_parse_values():
    from pointblank.ablation import _parse_float, _parse_int

    assert _parse_int(3, "x") == 3
    assert _parse_int("3", "x") == 3
    assert _parse_int(" 3.0 ", "x") == 3
    assert _parse_int(3.0, "x") == 3
    with pytest.raises(ValueError, match="x None, which isn't an integer"):
        _parse_int(None, "x")
    with pytest.raises(ValueError, match="isn't an integer"):
        _parse_int(float("nan"), "x")

    assert _parse_float(None) is None
    assert _parse_float(" 1.5 ") == 1.5
    for missing in ("", "NA", "nan", "NULL", "None", float("nan"), pd.NA):
        assert _parse_float(missing) is None
    assert _parse_float(2) == 2.0
    with pytest.raises(ValueError, match="isn't a number"):
        _parse_float([1])


# ----------------------------------------------------------------------------------------------
# Notes in the validation
# ----------------------------------------------------------------------------------------------


def test_ablation_annotate():
    validation = _validation(_TBL)
    ablation = (
        pb.Ablation(validation, evaluate=_evaluate, repeats=5).drop_each().drop_all_but_each()
    )
    with pytest.raises(ValueError, match="There are no results yet"):
        ablation.annotate()
    ablation.run()

    assert ablation.annotate() is validation
    notes = {step.i: step.notes for step in validation.validation_info}
    assert set(notes[1]) == {"ablation"}
    text = notes[1]["ablation"]["text"]
    assert text.startswith("Ablation (score, higher is better): removing the 10 rows that failed")
    assert "compared with removing as many random rows: helps" in text
    assert "on top of steps 2, 3, this step's rows change it by" in text
    assert notes[1]["ablation"]["markdown"].count("**helps**") == 2
    assert "hurts" in notes[2]["ablation"]["text"]
    assert notes[4] is None or "ablation" not in notes[4]

    # The notes appear in the validation's report
    assert (
        "removing the 10 rows that failed this step"
        in validation.get_tabular_report().as_raw_html()
    )

    # Annotating again replaces the notes
    ablation.annotate()
    assert set(validation.validation_info[0].notes) == {"ablation"}


def test_ablation_annotate_other_validation():
    ablation = pb.Ablation(
        _validation(_TBL), evaluate=_evaluate, repeats=3, control=None
    ).drop_each(steps=[1, 2])
    ablation.drop_groups("g", values="gold").run()

    newer = _validation(_TBL.head(50))
    assert ablation.annotate(newer) is newer
    text = newer.validation_info[0].notes["ablation"]["text"]
    assert "compared with the full table" in text
    # Steps without a variant of their own (and groups) have no note
    assert newer.validation_info[2].notes is None or "ablation" not in (
        newer.validation_info[2].notes
    )

    different = (
        pb.Validate(data=_TBL)
        .col_vals_ge(columns="x", value=0)
        .col_vals_ne(columns="y", value=0)
        .interrogate()
    )
    with pytest.raises(ValueError, match="its step 2 isn't a `col_vals_ne\\(\\)` step of the same"):
        ablation.annotate(different)
    with pytest.raises(TypeError, match="`validation=` must be a `Validate` object"):
        ablation.annotate("validation")


# ----------------------------------------------------------------------------------------------
# YAML
# ----------------------------------------------------------------------------------------------

_YAML_ABLATION = """
tbl: small_table
tbl_name: training
steps:
- col_vals_ge:
    columns: x
    value: 0
- col_vals_ne:
    columns: g
    value: gold
- col_vals_not_null:
    columns: y
ablation:
  metric: score
  repeats: 3
  seed: 2
  label: YAML ablation
  evaluate_version: v1
  control:
    stratified: g
  strategies:
    - drop_each
    - drop_all_but_each:
        steps: [1, 2]
    - drop_steps:
        steps: [1, 3]
        name: x and y
    - drop_groups:
        column: g
        values: gold
"""


def test_yaml_ablation():
    ablation = pb.yaml_ablation(_YAML_ABLATION, evaluate=_evaluate, set_tbl=_TBL)
    assert isinstance(ablation, pb.Ablation)
    assert ablation._effects is None
    assert ablation.validation.tbl_name == "training"
    assert (ablation.metric, ablation.repeats, ablation.seed) == ("score", 3, 2)
    assert ablation.label == "YAML ablation"
    assert ablation.evaluate_version == "v1"
    assert ablation.control == pb.ablation.Stratified("g")

    expected = (
        pb.Ablation(
            _validation(_TBL),
            evaluate=_evaluate,
            metric="score",
            repeats=3,
            seed=2,
            control=pb.ablation.Stratified("g"),
        )
        .drop_each()
        .drop_all_but_each(steps=[1, 2])
        .drop_steps([1, 3], name="x and y")
        .drop_groups("g", values="gold")
    )
    assert ablation.run().get_effects().equals(expected.run().get_effects())

    # The block is ignored when interrogating, and is valid YAML for `validate_yaml()`
    validation = pb.yaml_interrogate(_YAML_ABLATION, set_tbl=_TBL)
    assert validation.n_failed() == {1: 10, 2: 10, 3: 10}
    pb.validate_yaml(_YAML_ABLATION)


def test_yaml_ablation_settings():
    config = """
tbl: small_table
steps:
- col_vals_ge:
    columns: x
    value: 0
ablation:
  higher_is_better: false
  control: null
  duplicates: keep_last
  max_removed: 0.25
  conf_level: 0.9
  adjust: null
  strategies: [drop_each]
"""
    ablation = pb.yaml_ablation(config, evaluate=_evaluate, set_tbl=_TBL)
    assert ablation.higher_is_better is False
    assert ablation.control is None
    assert ablation.duplicates == "keep_last"
    assert (ablation.max_removed, ablation.conf_level, ablation.adjust) == (0.25, 0.9, None)
    assert ablation.repeats == 5
    assert [spec.id for spec in ablation._specs] == ["step_1"]

    # Without strategies, the ablation has no variants yet
    config = "tbl: small_table\nsteps:\n- rows_distinct\nablation:\n  repeats: 3\n"
    ablation = pb.yaml_ablation(config, evaluate=_evaluate, set_tbl=_TBL)
    assert ablation._specs == []
    ablation.drop_each()
    # A `size_matched` control written out
    config = config.replace("repeats: 3", "control: size_matched")
    assert pb.yaml_ablation(config, set_tbl=_TBL).control == "size_matched"


def test_yaml_ablation_evaluate_from_config(tmp_path, monkeypatch):
    (tmp_path / "ablation_models.py").write_text(
        "from tests.test_ablation import _evaluate\n"
        "class Models:\n"
        "    evaluate = staticmethod(_evaluate)\n"
        "not_a_function = 3\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    from tests.test_ablation import _evaluate as evaluate

    config = _YAML_ABLATION.replace(
        "  metric: score", "  evaluate: ablation_models:_evaluate\n  metric: score"
    )

    # The function is Python code, so it's only imported with `allow_python=True`
    with pytest.raises(pb.yaml.YAMLValidationError, match="is Python code, which isn't run"):
        pb.yaml_ablation(config, set_tbl=_TBL)
    assert pb.yaml_ablation(config, set_tbl=_TBL, allow_python=True).evaluate is evaluate
    # ... unless `evaluate=` is given instead
    other = pb.yaml_ablation(config, set_tbl=_TBL, evaluate=_evaluate)
    assert other.evaluate is _evaluate
    # `yaml_interrogate()` ignores it
    pb.yaml_interrogate(config, set_tbl=_TBL)

    nested = config.replace("ablation_models:_evaluate", "ablation_models:Models.evaluate")
    assert pb.yaml_ablation(nested, set_tbl=_TBL, allow_python=True).evaluate is evaluate

    for path, match in (
        ("no_such_module_xyz:f", "can't be imported"),
        ("ablation_models:nope", "has no 'nope'"),
        ("ablation_models:not_a_function", "must be a function, got int"),
    ):
        bad = config.replace("ablation_models:_evaluate", path)
        with pytest.raises(pb.yaml.YAMLValidationError, match=match):
            pb.yaml_ablation(bad, set_tbl=_TBL, allow_python=True)

    # A `python:` block
    block = _YAML_ABLATION.replace(
        "  metric: score",
        "  evaluate:\n    python: |\n      lambda train, *, seed: float(len(train)) + seed\n"
        "  metric: score",
    )
    with pytest.raises(pb.yaml.YAMLValidationError, match="a `python:` block"):
        pb.yaml_ablation(block, set_tbl=_TBL)
    ablation = pb.yaml_ablation(block, set_tbl=_TBL, allow_python=True)
    assert ablation.evaluate(_TBL, seed=1) == 101.0


@pytest.mark.parametrize(
    "block, match",
    [
        ("  repeats: 3\n  bogus: 1\n", "Unknown key\\(s\\) in 'ablation': \\['bogus'\\]"),
        ("  - drop_each\n", "'ablation' must be a mapping"),
        ("  strategies: drop_each\n", "must be a list of strategies"),
        ("  strategies: [drop_rows]\n", "Unknown ablation strategy 'drop_rows'"),
        ("  strategies:\n  - drop_each: [1]\n", "parameters of the `drop_each` strategy must be"),
        ("  strategies:\n  - drop_groups: {}\n", "`drop_groups` strategy needs the parameter"),
        (
            "  strategies:\n  - drop_steps:\n      name: a\n",
            "needs the parameter\\(s\\) \\['steps'\\]",
        ),
        ("  strategies:\n  - drop_each:\n      step: 1\n", "Unknown parameter\\(s\\) \\['step'\\]"),
        ("  strategies:\n  - {drop_each: null, drop_all: null}\n", "a mapping of one name"),
        ("  control: random\n", "'ablation.control' must be `size_matched`"),
        ("  control:\n    by: g\n", "'ablation.control' must be `size_matched`"),
        ("  evaluate: models.evaluate\n", "must be the import path of a function"),
        ("  evaluate: 3\n", "must be the import path of a function"),
        ("  evaluate:\n    code: x\n", "must be the import path of a function"),
    ],
)
def test_yaml_ablation_block_errors(block, match):
    config = "tbl: small_table\nsteps:\n- rows_distinct\nablation:\n" + block
    with pytest.raises(pb.yaml.YAMLValidationError, match=match):
        pb.yaml_ablation(config, evaluate=_evaluate)
    # The block is checked whenever the config is loaded
    with pytest.raises(pb.yaml.YAMLValidationError, match=match):
        pb.validate_yaml(config)


def test_yaml_ablation_errors():
    config = "tbl: small_table\nsteps:\n- rows_distinct\n"
    with pytest.raises(pb.yaml.YAMLValidationError, match="has no `ablation:` block"):
        pb.yaml_ablation(config, evaluate=_evaluate)
    with pytest.raises(TypeError, match="`evaluate=` must be a function"):
        pb.yaml_ablation(config + "ablation: {}\n", evaluate="f")
    # Settings are checked by `Ablation`
    with pytest.raises(ValueError, match="`repeats=` must be a positive integer"):
        pb.yaml_ablation(config + "ablation:\n  repeats: 0\n", evaluate=_evaluate)
    # Strategies are checked by the methods
    with pytest.raises(ValueError, match="can't use step 5"):
        pb.yaml_ablation(
            config + "ablation:\n  strategies:\n  - drop_each:\n      steps: 5\n",
            evaluate=_evaluate,
        )
    # Python code is checked before the data is loaded
    with pytest.raises(pb.yaml.YAMLValidationError, match="is Python code"):
        pb.yaml_ablation("tbl: nowhere.csv\nsteps:\n- rows_distinct\nablation:\n  evaluate: a:b\n")


def test_yaml_to_python_with_ablation():
    config = _YAML_ABLATION.replace(
        "  metric: score", "  evaluate: my_models.training:evaluate\n  metric: score"
    )
    code = pb.yaml_to_python(config)
    assert "from my_models.training import evaluate\n" in code
    assert "validation = (\n    pb.Validate(" in code
    assert (
        "ablation = (\n"
        "    pb.Ablation(\n"
        "        validation,\n"
        "        evaluate=evaluate,\n"
        '        metric="score",\n'
        "        repeats=3,\n"
        "        seed=2,\n"
        '        label="YAML ablation",\n'
        '        evaluate_version="v1",\n'
        '        control=pb.ablation.Stratified(by="g"),\n'
        "    )\n"
        "    .drop_each()\n"
        "    .drop_all_but_each(steps=[1, 2])\n"
        '    .drop_steps(steps=[1, 3], name="x and y")\n'
        '    .drop_groups(column="g", values="gold")\n'
        ")\n"
    ) in code

    # The generated code runs (with the function defined and the data swapped)
    source = code.removeprefix("```python\n").removesuffix("\n```")
    source = source.replace("from my_models.training import evaluate\n", "")
    source = source.replace('pb.load_dataset("small_table", tbl_type="polars")', "tbl")
    namespace = {"evaluate": _evaluate, "tbl": _TBL}
    exec(source, namespace)
    assert namespace["ablation"].run().get_effects().height == 9

    # A nested name, and a `python:` block
    nested = config.replace("my_models.training:evaluate", "my_models:Models.evaluate")
    assert "from my_models import Models\n" in pb.yaml_to_python(nested)
    assert "evaluate=Models.evaluate," in pb.yaml_to_python(nested)
    block = _YAML_ABLATION.replace(
        "  metric: score",
        "  evaluate:\n    python: |\n      lambda train, *, seed: 1.0\n  metric: score",
    )
    assert "        evaluate=lambda train, *, seed: 1.0,\n" in pb.yaml_to_python(block)

    # Without an `ablation:` block, the code is unchanged
    plain = pb.yaml_to_python("tbl: small_table\nsteps:\n- rows_distinct\n")
    assert "\n(\n    pb.Validate(" in plain and "Ablation" not in plain


# ----------------------------------------------------------------------------------------------
# scikit-learn
# ----------------------------------------------------------------------------------------------


def _sklearn_tables():
    """Training and holdout tables where the target is `2 * a + b`, with some bad targets."""

    def make(n, offset):
        a = [((k + offset) * 37 % 101) / 10 for k in range(n)]
        return pl.DataFrame(
            {
                "a": a,
                "b": [float((k + offset) % 5) for k in range(n)],
                "kind": [["x", "y", "z"][(k + offset) % 3] for k in range(n)],
                "flag": [(k + offset) % 2 == 0 for k in range(n)],
                "when": [None] * n,
                "target": [2 * v + (k + offset) % 5 for k, v in enumerate(a)],
            }
        ).with_columns(pl.col("when").cast(pl.Date))

    train = (
        make(300, 0)
        .with_row_index()
        .with_columns(
            target=pl.when(pl.col("index") % 10 == 3)
            .then(pl.col("target") * 50)
            .otherwise("target")
        )
        .drop("index")
    )
    return train, make(150, 1000)


def test_sklearn_evaluator_regression():
    pytest.importorskip("sklearn")

    train, holdout = _sklearn_tables()
    with pytest.warns(UserWarning, match="doesn't use the columns 'when'"):
        evaluate = pb.ablation.sklearn_evaluator(holdout, target="target")
    assert evaluate.task == "regression"
    assert evaluate.features == ["a", "b", "kind", "flag"]
    assert evaluate.kinds == {
        "a": "numeric",
        "b": "numeric",
        "kind": "categorical",
        "flag": "numeric",
    }
    assert repr(evaluate).startswith("sklearn_evaluator(target='target'")

    result = evaluate(train, seed=1)
    assert set(result) == {"r2"}
    assert result == evaluate(train, seed=1)
    clean = train.filter(pl.col("target") < 100)
    assert evaluate(clean, seed=1)["r2"] > 0.95 > result["r2"]

    # The same scores from a pandas or lazy table, and for a pandas holdout
    assert evaluate(train.to_pandas(), seed=1) == pytest.approx(result)
    assert evaluate(train.lazy(), seed=1) == pytest.approx(result)
    pandas_evaluate = pb.ablation.sklearn_evaluator(
        holdout.drop("when").to_pandas(), target="target"
    )
    assert pandas_evaluate(train.drop("when"), seed=1) == pytest.approx(result)

    # It can be pickled, for `executor="process"`
    assert pickle.loads(pickle.dumps(evaluate))(train, seed=1) == pytest.approx(result)

    validation = pb.Validate(data=train).col_vals_lt(columns="target", value=100).interrogate()
    ablation = pb.Ablation(validation, evaluate=evaluate, repeats=3).drop_each().run()
    assert _effects_by_variant(ablation)["step_1"]["verdict"] == "helps"


def test_sklearn_evaluator_classification():
    pytest.importorskip("sklearn")

    train, holdout = _sklearn_tables()
    train = train.with_columns(
        label=pl.when(pl.col("a") > 5).then(pl.lit("hi")).otherwise(pl.lit("lo"))
    )
    holdout = holdout.with_columns(
        label=pl.when(pl.col("a") > 5).then(pl.lit("hi")).otherwise(pl.lit("lo"))
    )

    evaluate = pb.ablation.sklearn_evaluator(
        holdout, target="label", features=["a", "kind"], scoring=["accuracy", "f1_macro"]
    )
    assert evaluate.task == "classification"
    result = evaluate(train, seed=0)
    assert set(result) == {"accuracy", "f1_macro"}
    assert result["accuracy"] > 0.95

    # Integer targets with few values are classes, and with many they're a regression
    with_ints = train.with_columns(cls=(pl.col("a") > 5).cast(pl.Int64))
    holdout_ints = holdout.with_columns(cls=(pl.col("a") > 5).cast(pl.Int64))
    assert pb.ablation.sklearn_evaluator(holdout_ints, target="cls", features="a").task == (
        "classification"
    )
    assert (
        pb.ablation.sklearn_evaluator(holdout_ints, target="cls", features=["a"])(
            with_ints, seed=0
        )["accuracy"]
        > 0.95
    )
    many = holdout.with_columns(n=pl.int_range(pl.len()))
    assert pb.ablation.sklearn_evaluator(many, target="n", features=["a"]).task == "regression"

    # Boolean targets are classes
    def with_parity(tbl):
        return tbl.with_columns(parity=pl.col("flag").cast(pl.Float64))

    flag = pb.ablation.sklearn_evaluator(with_parity(holdout), target="flag", features=["parity"])
    assert flag.task == "classification"
    assert flag(with_parity(train), seed=0)["accuracy"] == 1.0

    # A category that's only in the holdout data is treated as a missing value
    odd = holdout.with_columns(kind=pl.lit("new"))
    assert "accuracy" in pb.ablation.sklearn_evaluator(odd, target="label", features=["a", "kind"])(
        train, seed=0
    )


def test_sklearn_evaluator_estimator_and_scoring():
    sklearn = pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import make_scorer, mean_absolute_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train, holdout = _sklearn_tables()
    train, holdout = train.drop("when"), holdout.drop("when")

    linear = pb.ablation.sklearn_evaluator(
        holdout,
        target="target",
        features=["a", "b"],
        estimator=make_pipeline(StandardScaler(), LinearRegression()),
        scoring={"mae": make_scorer(mean_absolute_error, greater_is_better=False), "r2": "r2"},
    )
    clean = train.filter(pl.col("target") < 100)
    result = linear(clean, seed=0)
    assert set(result) == {"mae", "r2"}
    assert result["mae"] == pytest.approx(0, abs=1e-9)
    assert result["r2"] == pytest.approx(1.0)

    # The estimator is cloned, with its random states set to the seed
    forest = RandomForestRegressor(n_estimators=5, random_state=123)
    evaluate = pb.ablation.sklearn_evaluator(
        holdout, target="target", features=["a", "b"], estimator=forest
    )
    assert evaluate._model(7, {}).random_state == 7
    assert forest.random_state == 123
    assert not hasattr(forest, "estimators_")
    assert evaluate(train, seed=7) == evaluate(train, seed=7)
    assert evaluate(train, seed=7) != evaluate(train, seed=8)
    pipeline = pb.ablation.sklearn_evaluator(
        holdout,
        target="target",
        features=["a"],
        estimator=make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=3)),
    )
    assert pipeline._model(5, {}).get_params()["randomforestregressor__random_state"] == 5

    def score(estimator, X, y):
        return -abs(estimator.predict(X) - y).max()

    custom = pb.ablation.sklearn_evaluator(
        holdout, target="target", features=["a", "b"], estimator=LinearRegression(), scoring=score
    )
    assert set(custom(clean, seed=0)) == {"score"}
    assert sklearn.__version__


def test_sklearn_evaluator_missing_values():
    pytest.importorskip("sklearn")

    train, holdout = _sklearn_tables()
    train = train.drop("when").with_columns(
        target=pl.when(pl.int_range(pl.len()) % 7 == 0).then(None).otherwise("target"),
        kind=pl.when(pl.int_range(pl.len()) % 11 == 0).then(None).otherwise("kind"),
        a=pl.when(pl.int_range(pl.len()) % 13 == 0).then(None).otherwise("a"),
    )
    holdout = holdout.drop("when").with_columns(
        target=pl.when(pl.int_range(pl.len()) % 9 == 0).then(None).otherwise("target")
    )
    evaluate = pb.ablation.sklearn_evaluator(holdout, target="target")
    assert len(evaluate.holdout) == 150 - 17
    assert "r2" in evaluate(train, seed=0)
    # Categorical and enum features
    categorical = train.with_columns(kind=pl.col("kind").cast(pl.Categorical))
    evaluate = pb.ablation.sklearn_evaluator(
        holdout.with_columns(kind=pl.col("kind").cast(pl.Categorical)), target="target"
    )
    assert evaluate.kinds["kind"] == "categorical"
    assert "r2" in evaluate(categorical, seed=0)


def test_sklearn_evaluator_errors():
    pytest.importorskip("sklearn")

    train, holdout = _sklearn_tables()
    make = pb.ablation.sklearn_evaluator
    with pytest.raises(TypeError, match="`target=` must be a column name"):
        make(holdout, target=["target"])
    with pytest.raises(ValueError, match="no column 'nope' \\(`target=`\\)"):
        make(holdout, target="nope")
    with pytest.raises(ValueError, match="`task=` must be one of"):
        make(holdout, target="target", task="clustering")
    with pytest.raises(ValueError, match="no column 'nope' \\(`features=`\\)"):
        make(holdout, target="target", features=["a", "nope"])
    with pytest.raises(ValueError, match="can't also be a feature"):
        make(holdout, target="target", features=["a", "target"])
    with pytest.raises(TypeError, match="The features 'when' have types that can't be used"):
        make(holdout, target="target", features=["a", "when"])
    with pytest.raises(ValueError, match="There are no features"):
        make(holdout.select("target"), target="target")
    with pytest.raises(ValueError, match="no rows with a value of 'when'"):
        make(holdout, target="when", features=["a"], task="classification")
    with pytest.raises(TypeError, match="isn't a class or a number"):
        make(holdout.with_columns(when=pl.date(2024, 1, 1)), target="when", features=["a"])
    with pytest.raises(TypeError, match="A regression needs a numeric target"):
        make(holdout, target="kind", features=["a"], task="regression")
    with pytest.raises(ValueError, match="'nope' isn't the name of a scikit-learn scorer"):
        make(holdout, target="target", features=["a"], scoring="nope")
    for scoring in ([], {}, 3, ["r2", 3]):
        with pytest.raises(TypeError, match="`scoring=`|list of scorer names"):
            make(holdout, target="target", features=["a"], scoring=scoring)
    with pytest.raises(TypeError, match="The scorer 'x' must be"):
        make(holdout, target="target", features=["a"], scoring={"x": 3})

    evaluate = make(holdout, target="target", features=["a", "b"])
    with pytest.raises(ValueError, match="The training table has no column 'b'"):
        evaluate(train.drop("b"), seed=0)


def test_sklearn_evaluator_without_sklearn(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sklearn" or name.startswith("sklearn."):
            raise ImportError("No module named 'sklearn'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match='pip install "pointblank\\[ml\\]"'):
        pb.ablation.sklearn_evaluator(_TBL, target="x")
