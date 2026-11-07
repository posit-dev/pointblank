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
    n, bad, gold = counts.row(0)
    return int(n), int(bad), int(gold)


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
    with pytest.raises(ValueError, match="Call `run\\(\\)` first"):
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
    with pytest.raises(ValueError, match="Call `run\\(\\)` first"):
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
    with pytest.raises(ValueError, match="Call `run\\(\\)` first"):
        ablation.to_json()
    ablation.run()

    data = json.loads(ablation.to_json())
    assert list(data) == ["settings", "variants", "runs", "effects", "notes"]
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
