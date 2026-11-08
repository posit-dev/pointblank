import json
import math
import random
import warnings

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

import pointblank as pb
from pointblank.ablation import _corrupted_html, _function_name, _is_missing
from tests.test_ablation import _TBL, _evaluate, _tbl, _validation

# The rows of `_TBL` that fail none of the steps of `_validation()`: those with `k % 10` from 3
# to 9 (70 rows)
_CLEAN = [k for k in range(100) if k % 10 >= 3]


def _frame(train) -> pl.DataFrame:
    frame = nw.from_native(train)
    if isinstance(frame, nw.LazyFrame):
        frame = frame.collect()
    # (pandas has NaN for missing floats)
    return pl.from_dict(frame.to_dict(as_series=False)).fill_nan(None)


_SEEN = {}


def _null_evaluate(train, *, seed, variant):
    """A 'model' that loses a point for each missing `x`."""
    frame = _frame(train)
    _SEEN[variant.id] = frame
    return -frame["x"].null_count() + random.Random(seed).gauss(0, 0.1)


def _inject_effects(ablation):
    return {row["variant"]: row for row in ablation.get_effects().iter_rows(named=True)}


# ----------------------------------------------------------------------------------------------
# inject()
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tbl_type", ["polars", "lazy", "pandas"])
def test_inject_null(tbl_type):
    _SEEN.clear()
    ablation = (
        pb.Ablation(_validation(_tbl(tbl_type)), evaluate=_null_evaluate, repeats=3)
        .inject("null", column="x", step=1, rates=[0.05, 0.1])
        .run()
    )

    effects = _inject_effects(ablation)
    assert list(effects) == ["full", "inject:null:x_0.05", "inject:null:x_0.1"]
    low, high = effects["inject:null:x_0.05"], effects["inject:null:x_0.1"]
    assert low["name"] == "Missing values in x: 5%"
    assert (low["strategy"], low["sweep"], low["value"]) == ("inject", "inject:null:x", 0.05)
    assert (low["corruption"], low["column"], low["checked_by"]) == ("null", "x", 1)
    assert (low["n_corrupted"], high["n_corrupted"]) == (5, 10)
    assert (low["n_removed"], low["f_removed"]) == (0, 0.0)
    # A defect is compared with the full table, as it has no control
    assert low["delta_vs_full"] == pytest.approx(-5, abs=0.5)
    assert high["delta_vs_full"] == pytest.approx(-10, abs=0.5)
    assert low["delta_vs_control"] is None and low["control_mean"] is None
    assert low["verdict"] == high["verdict"] == "hurts"
    assert effects["full"]["n_corrupted"] == 0

    # The corrupted rows are clean rows, nested across the rates, and the same in every repeat
    nulls = {}
    for variant_id in ("inject:null:x_0.05", "inject:null:x_0.1"):
        frame = _SEEN[variant_id]
        assert frame.height == 100
        nulls[variant_id] = set(frame.filter(pl.col("x").is_null())["rid"].to_list())
    assert nulls["inject:null:x_0.05"] < nulls["inject:null:x_0.1"]
    assert nulls["inject:null:x_0.1"] <= set(_CLEAN)
    assert _SEEN["full"]["x"].null_count() == 0
    # Only rows with no control are used: just the full table's
    assert [v.id for v in ablation._control_variants] == []

    html = ablation.get_tabular_report().as_raw_html()
    assert "Injected: missing values in x" in html
    assert "5.0% corrupted" in html
    assert "For injected defects" not in html  # (there's no control to compare the others with)
    assert "Share of the rows corrupted" in html
    plan = ablation.plan().as_raw_html()
    assert "10.0% corrupted" in plan
    assert "5 clean rows get a missing value in <code>x</code> (checked by step 1)" in plan


def test_inject_with_other_variants_and_seed():
    ablation = (
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=3)
        .drop_each(steps=1)
        .inject("null", column="y")
    )
    resolved = ablation._resolve()
    assert [v.id for v, _ in resolved.variants] == [
        "step_1",
        "inject:null:y_0.01",
        "inject:null:y_0.05",
        "inject:null:y_0.1",
    ]
    # Only the variant that removes rows gets a control
    assert [c.id for c in resolved.controls] == ["control_10"]
    assert resolved.variant_control == {"step_1": "control_10"}

    ablation.run()
    html = ablation.get_tabular_report().as_raw_html()
    assert "For injected defects, Δ is the difference from the full table" in html

    rows = ablation._injections["inject:null:y_0.1"].rows
    other = pb.Ablation(_validation(_TBL), evaluate=_evaluate, seed=7).inject("null", column="y")
    assert other._injections["inject:null:y_0.1"].rows != rows
    again = pb.Ablation(_validation(_TBL), evaluate=_evaluate).inject("null", column="y")
    assert again._injections["inject:null:y_0.1"].rows == rows


def test_inject_shuffle_truncate_duplicate():
    seen = {}

    def evaluate(train, *, seed, variant):
        seen[variant.id] = _frame(train)
        return 0.0

    ablation = (
        pb.Ablation(_validation(_TBL), evaluate=evaluate, repeats=1)
        .inject("shuffle", column="y", rates=[0.1])
        .inject("truncate", column="g", rates=[0.05])
        .inject("duplicate", rates=[0.02])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ablation.run()

    full = seen["full"]
    shuffled = seen["inject:shuffle:y_0.1"]
    changed = full.join(shuffled, on="rid", suffix="_new").filter(pl.col("y") != pl.col("y_new"))
    # Every corrupted row gets another row's value (the values are all different)
    assert changed.height == 10
    assert sorted(changed["y"].to_list()) == sorted(changed["y_new"].to_list())
    assert set(changed["rid"].to_list()) <= set(_CLEAN)

    truncated = seen["inject:truncate:g_0.05"]
    assert truncated["g"].value_counts().sort("g").rows() == [
        ("gold", 10),
        ("pl", 5),
        ("plain", 85),
    ]

    duplicated = seen["inject:duplicate_0.02"]
    assert duplicated.height == 102
    copies = duplicated["rid"].to_list()[100:]
    assert len(set(copies)) == 2 and set(copies) <= set(_CLEAN)

    effects = _inject_effects(ablation)
    assert effects["inject:duplicate_0.02"]["column"] is None
    assert effects["inject:duplicate_0.02"]["name"] == "Duplicated rows: 2%"
    assert "2.0% copied" in ablation.plan().as_raw_html()
    assert ablation._sweeps["inject:duplicate"]["axis"] == "Share of the rows copied"


def test_inject_pandas_and_pyarrow_keep_their_types():
    seen = {}

    def evaluate(train, *, seed, variant):
        seen[variant.id] = train
        return 0.0

    pdf = _TBL.to_pandas()
    pdf.index = [f"r{k}" for k in range(100)]
    ablation = (
        pb.Ablation(_validation(pdf), evaluate=evaluate, repeats=1)
        .inject(lambda x: x * 10, column="x", rates=[0.1])
        .inject("duplicate", rates=[0.01])
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ablation.run()
    scaled = seen["inject:custom:x_0.1"]
    assert isinstance(scaled, pd.DataFrame)
    assert list(scaled.index) == list(pdf.index)
    assert scaled["x"].dtype == "float64"
    assert (scaled["x"] != pdf["x"]).sum() == 10
    assert len(seen["inject:duplicate_0.01"]) == 101

    import pyarrow as pa

    ablation = pb.Ablation(_validation(_TBL.to_arrow()), evaluate=evaluate, repeats=1).inject(
        "shuffle", column="rid", rates=[0.05]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ablation.run()
    shuffled = seen["inject:shuffle:rid_0.05"]
    assert isinstance(shuffled, pa.Table)
    assert shuffled.schema.field("rid").type == pa.int64()
    assert shuffled["rid"].to_pylist() != list(range(100))


def test_inject_functions():
    def unit_error(x):
        return x * 10

    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate).inject(
        unit_error, step=1, rates=[0.1]
    )
    variant = ablation._resolve().variants[0][0]
    assert variant.id == "inject:unit_error:x_0.1"
    assert variant.name == "Unit_error on x: 10%"
    assert (variant.corruption, variant.column, variant.checked_by) == ("function", "x", 1)
    injection = ablation._injections[variant.id]
    assert all(v == k * 10.0 for k, v in zip(injection.rows, injection.values))

    named = pb.Ablation(_validation(_TBL), evaluate=_evaluate).inject(
        lambda x: -x, column="x", name="sign_flip", rates=[0.1]
    )
    assert list(named._injections) == ["inject:sign_flip:x_0.1"]

    def fails(x):
        raise RuntimeError("boom")

    with pytest.raises(ValueError, match="function raised an error for the value .*: boom"):
        pb.Ablation(_validation(_TBL), evaluate=_evaluate).inject(fails, column="x")
    with pytest.raises(TypeError, match="can't be put in the column 'x'"):
        pb.Ablation(_validation(_TBL), evaluate=_evaluate).inject(lambda x: "text", column="x")


def test_inject_notes_for_rates():
    small = pl.DataFrame({"x": [float(k) for k in range(30)]})
    validation = pb.Validate(data=small).col_vals_lt(columns="x", value=20).interrogate()

    ablation = pb.Ablation(validation, evaluate=_evaluate).inject(
        "null", column="x", rates=[0.01, 0.5, 0.9]
    )

    # 1% of 30 rows rounds to no rows, and 90% needs more than the 20 clean rows
    assert [spec.id for spec in ablation._specs] == ["inject:null:x_0.5"]
    assert ablation._notes == [
        "Missing values in x: 1% isn't used: 1% of the 30 rows is too few rows to corrupt.",
        "Missing values in x: 90% isn't used: it needs 27 clean rows, but there are only 20.",
    ]
    # Shuffling needs at least 2 rows
    shuffle = pb.Ablation(validation, evaluate=_evaluate).inject(
        "shuffle", column="x", rates=[0.03, 0.1]
    )
    assert [spec.id for spec in shuffle._specs] == ["inject:shuffle:x_0.1"]


def test_inject_skips_missing_values_and_empty_text():
    tbl = pl.DataFrame(
        {"t": ["", "a", None, "abcd", "xy"] * 4, "n": [None, 1.0, 2.0, float("nan"), 3.0] * 4}
    )
    validation = pb.Validate(data=tbl).col_exists(columns="t").interrogate()

    # Only the 12 values with characters can be truncated (to their first half)
    ablation = pb.Ablation(validation, evaluate=_evaluate).inject(
        "truncate", column="t", rates=[0.6, 0.65]
    )
    assert ablation._notes == [
        "Truncated values of t: 65% isn't used: it needs 13 clean rows, but there are only 12."
    ]
    injection = ablation._injections["inject:truncate:t_0.6"]
    assert sorted(injection.rows) == [k for k in range(20) if k % 5 in (1, 3, 4)]
    assert sorted(set(injection.values)) == ["", "ab", "x"]

    # A missing value can't be made missing (NaN counts as missing)
    nulls = pb.Ablation(validation, evaluate=_evaluate).inject("null", column="n", rates=[0.65])
    assert nulls._notes == [
        "Missing values in n: 65% isn't used: it needs 13 clean rows, but there are only 12."
    ]


def test_is_missing_and_function_name():
    assert _is_missing(None) and _is_missing(float("nan")) and _is_missing(pd.NA)
    assert _is_missing(pd.NaT)
    assert not _is_missing(0) and not _is_missing("") and not _is_missing([1, 2])
    # (an array's comparison has no single truth value)
    assert not _is_missing(__import__("numpy").array([1.0, 2.0]))
    assert _function_name(lambda x: x) == "custom"
    assert _function_name(abs) == "abs"
    assert _function_name(type("F", (), {"__call__": lambda self, x: x})()) == "custom"


def test_corrupted_html():
    from pointblank.ablation import AblationVariant

    variant = AblationVariant(id="v", strategy="inject", name="V", corruption="null", n_corrupted=5)
    assert "5.0% corrupted" in _corrupted_html(variant, 100)
    copied = AblationVariant(
        id="v", strategy="inject", name="V", corruption="duplicate", n_corrupted=1
    )
    assert "&lt;0.1% copied" in _corrupted_html(copied, 10_000)


def test_inject_errors():
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate)

    with pytest.raises(ValueError, match="`corruption=` must be one of 'null'"):
        ablation.inject("zero", column="x")
    with pytest.raises(TypeError, match="must be the name of a defect or a function"):
        ablation.inject(1, column="x")
    with pytest.raises(TypeError, match="`name=` must be a non-empty string"):
        ablation.inject("null", column="x", name="")
    with pytest.raises(TypeError, match="`step=` must be a step number"):
        ablation.inject("null", step="1")
    with pytest.raises(ValueError, match="can't use step 4: it doesn't check rows"):
        ablation.inject("null", step=4)
    with pytest.raises(ValueError, match="`column=` isn't used with"):
        ablation.inject("duplicate", column="x")
    with pytest.raises(ValueError, match="needs the column to corrupt, as `column=`.$"):
        ablation.inject("null")
    with pytest.raises(TypeError, match="`column=` must be the name of a column"):
        ablation.inject("null", column=["x"])
    with pytest.raises(ValueError, match="The column 'nope' isn't in the table"):
        ablation.inject("null", column="nope")
    with pytest.raises(TypeError, match='`corruption="truncate"` needs a text column'):
        ablation.inject("truncate", column="x")
    with pytest.raises(ValueError, match="greater than 0 and at most 1, got 0"):
        ablation.inject("null", column="x", rates=[0, 0.1])
    with pytest.raises(ValueError, match="greater than 0 and at most 1, got 1.5"):
        ablation.inject("null", column="x", rates=[1.5])
    with pytest.raises(TypeError, match="`rates=` must be a list"):
        ablation.inject("null", column="x", rates=0.1)

    # A step that doesn't check a single column needs `column=`
    validation = pb.Validate(data=_TBL).rows_distinct(columns_subset=["rid", "x"]).interrogate()
    with pytest.raises(ValueError, match="step 1 doesn't check a single column"):
        pb.Ablation(validation, evaluate=_evaluate).inject("null", step=1)


@pytest.mark.parametrize("tbl_type", ["duckdb", "sqlite", "pyspark"])
def test_inject_needs_an_in_memory_table(tbl_type):
    if tbl_type != "pyspark":
        pytest.importorskip("ibis")
    ablation = pb.Ablation(_validation(_tbl(tbl_type)), evaluate=_evaluate)
    with pytest.raises(ValueError, match="corrupts the table in memory"):
        ablation.inject("null", column="x")


def test_inject_cache_and_export(tmp_path):
    calls = []

    def evaluate(train, *, seed, variant):
        calls.append(variant.id)
        return -float(_frame(train)["x"].null_count())

    def make():
        return (
            pb.Ablation(_validation(_TBL), evaluate=evaluate, repeats=3, cache=tmp_path / "c")
            .drop_each(steps=1)
            .inject("null", column="x", rates=[0.05])
            .inject("null", column="y", rates=[0.05])
        )

    first = make()
    first.run()
    n_calls = len(calls)
    assert n_calls == 15
    calls.clear()
    second = make().run()
    assert calls == []
    assert second.get_effects()["mean"].to_list() == first.get_effects()["mean"].to_list()
    # The defects have keys of their own, and the variants that remove rows keep theirs
    variants = {v.id: v for v, _ in first._resolve().variants}
    keys = {first._task_key(variants[i], 0) for i in variants}
    assert len(keys) == 3
    assert first._task_key(variants["step_1"], 0) == first._cache_key(
        first._resolve().removed["step_1"], first.seed
    )
    assert "already in the cache" in make().plan().as_raw_html()

    with pytest.raises(ValueError, match='`format="indices"` can only describe'):
        first.export(tmp_path / "idx", format="indices")
    root = first.export(tmp_path / "pq")
    manifest = json.loads((root / "manifest.json").read_text())
    file = next(run["file"] for run in manifest["runs"] if run["variant"] == "inject:null:x_0.05")
    table = pl.read_parquet(root / file)
    assert table.height == 100 and table["x"].null_count() == 5

    # The results of the export can be loaded back
    results = pl.DataFrame(
        {
            "variant": [run["variant"] for run in manifest["runs"]],
            "repeat": [run["repeat"] for run in manifest["runs"]],
            "seed": [run["seed"] for run in manifest["runs"]],
            "score": [
                -float(pl.read_parquet(root / run["file"])["x"].null_count())
                for run in manifest["runs"]
            ],
        }
    )
    results.write_csv(root / "results.csv")
    loaded = make().load_results(root)
    effects = _inject_effects(loaded)
    assert effects["inject:null:x_0.05"]["delta_vs_full"] == pytest.approx(-5)


def test_inject_lazy_frame_export(tmp_path):
    ablation = pb.Ablation(_validation(_TBL.lazy()), evaluate=_evaluate, repeats=1).inject(
        "shuffle", column="y", rates=[0.1]
    )
    root = ablation.export(tmp_path)
    tables = sorted((root / "tables").iterdir())
    shuffled = pl.read_parquet(next(t for t in tables if "shuffle" in t.name))
    assert shuffled["y"].to_list() != _TBL["y"].to_list()
    assert sorted(shuffled["y"].drop_nulls()) == sorted(_TBL["y"].drop_nulls())


# ----------------------------------------------------------------------------------------------
# suggest_thresholds()
# ----------------------------------------------------------------------------------------------


def _readings():
    temps = [20 + (k * 37 % 11 - 5) * 0.6 for k in range(300)]
    return pl.DataFrame({"temp": temps, "t2": temps})


def _temp_evaluate(train, *, seed):
    sample = train.fill_null(0).sample(fraction=1.0, with_replacement=True, seed=seed)
    return abs(sample["temp"].mean() - 20)


def test_suggest_thresholds():
    validation = (
        pb.Validate(data=_readings())
        .col_vals_not_null(columns="temp")
        .col_vals_not_null(columns="t2")
        .interrogate()
    )
    ablation = (
        pb.Ablation(validation, evaluate=_temp_evaluate, metric="error", higher_is_better=False)
        .inject("null", step=1)
        .run()
    )

    thresholds = ablation.suggest_thresholds(tolerance=0.5)

    assert thresholds == {1: pb.Thresholds(warning=0.013, error=0.025)}
    # Every injected missing value fails the step
    assert [ablation._caught[k] for k in sorted(ablation._caught)] == pytest.approx(
        [0.01, 0.05, 0.1]
    )
    # A smaller tolerance gives smaller thresholds
    assert ablation.suggest_thresholds(tolerance=0.2)[1].error == pytest.approx(0.01, rel=0.1)
    # Failing every test unit costs less than the tolerance
    with pytest.warns(UserWarning, match="step 1: even with every test unit failing"):
        assert ablation.suggest_thresholds(tolerance=100) == {}

    for tolerance, error in ((0, ValueError), (-1, ValueError), ("a", TypeError)):
        with pytest.raises(error, match="`tolerance=`"):
            ablation.suggest_thresholds(tolerance=tolerance)


def test_suggest_thresholds_skipped_steps():
    validation = (
        pb.Validate(data=_readings())
        .col_vals_not_null(columns="temp")
        .col_vals_gt(columns="t2", value=0)
        .col_vals_not_null(columns="t2")
        .interrogate()
    )
    ablation = (
        pb.Ablation(validation, evaluate=_temp_evaluate, metric="error", higher_is_better=False)
        # The model doesn't use `t2`, and a shuffled reading still passes step 2
        .inject("null", step=3)
        .inject("shuffle", column="t2", step=2)
        .inject("null", step=1, rates=[0.05, 0.1])
        .run()
    )

    with pytest.warns(UserWarning) as record:
        thresholds = ablation.suggest_thresholds(tolerance=0.5)

    assert list(thresholds) == [1]
    message = str(record[0].message)
    assert "step 2: the shuffled values of t2 make none of its test units fail" in message
    assert "step 3: the missing values in t2 don't" in message


def test_suggest_thresholds_unclear_harm():
    validation = pb.Validate(data=_readings()).col_vals_not_null(columns="temp").interrogate()

    def evaluate(train, *, seed, variant):
        noise = random.Random(f"{seed}:{variant.id}").gauss(0, 0.05)
        return 0.0005 * train["temp"].null_count() + noise

    ablation = (
        pb.Ablation(validation, evaluate=evaluate, metric="error", higher_is_better=False)
        .inject("null", step=1)
        .run()
    )

    # The defect makes the model a bit worse, but none of the differences are clear
    with pytest.warns(UserWarning, match="don't clearly make the model worse"):
        assert ablation.suggest_thresholds(tolerance=0.01) == {}


def test_suggest_thresholds_needs_steps():
    ablation = (
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=3)
        .inject("null", column="x")
        .run()
    )
    with pytest.raises(ValueError, match="needs `inject\\(\\)` variants with the step"):
        ablation.suggest_thresholds(tolerance=1)
    with pytest.raises(ValueError, match="There are no results yet"):
        pb.Ablation(_validation(_TBL), evaluate=_evaluate).inject(
            "null", step=1
        ).suggest_thresholds(tolerance=1)


def test_suggest_thresholds_higher_is_better_and_segments():
    tbl = pl.DataFrame({"x": [float(k % 7) for k in range(200)], "s": ["a", "b"] * 100})

    def evaluate(train, *, seed):
        return -float(nw.from_native(train)["x"].null_count())

    validation = pb.Validate(data=tbl).col_vals_not_null(columns="x", segments="s").interrogate()
    ablation = pb.Ablation(validation, evaluate=evaluate, repeats=3).inject(
        "null", step=1, rates=[0.1]
    )
    ablation.run()

    # The step checks only the segment "a", so it catches about half of the missing values
    share = ablation._caught_share("inject:null:x_0.1")
    n_a = sum(1 for k in ablation._injections["inject:null:x_0.1"].rows if k % 2 == 0)
    assert share == pytest.approx(n_a / 100)
    thresholds = ablation.suggest_thresholds(tolerance=2)
    expected = float(f"{2 / (20 / share):.2g}")
    assert thresholds[1].error == pytest.approx(expected)


# ----------------------------------------------------------------------------------------------
# shapley()
# ----------------------------------------------------------------------------------------------


def _exact_shapley(value, players):
    k = len(players)
    phi = {}
    for i in players:
        others = [j for j in players if j != i]
        total = 0.0
        for size in range(k):
            weight = math.factorial(size) * math.factorial(k - size - 1) / math.factorial(k)
            for before in __import__("itertools").combinations(others, size):
                total += weight * (value[tuple(sorted((*before, i)))] - value[before])
        phi[i] = total
    return phi


@pytest.mark.parametrize("control", ["size_matched", None])
def test_shapley_exact(control):
    ablation = (
        pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=4, control=control)
        .shapley()
        .run()
    )
    resolved = ablation._resolve()
    assert sorted(resolved.shapley) == [(1,), (1, 2), (1, 2, 3), (1, 3), (2,), (2, 3), (3,)]
    assert [v.name for v, _ in resolved.variants][:4] == [
        "Step 1",
        "Step 2",
        "Step 3",
        "Steps 1, 2",
    ]

    shapley = ablation.get_shapley()
    assert shapley.columns == [
        "step",
        "n_failed",
        "metric",
        "shapley",
        "lower",
        "upper",
        "p",
        "p_adj",
        "verdict",
    ]
    rows = {row["step"]: row for row in shapley.iter_rows(named=True)}
    assert [rows[i]["n_failed"] for i in (1, 2, 3)] == [10, 10, 10]

    # The values match the exact formula, from the effects of the groups of steps
    effects = _inject_effects(ablation)
    comparison = "delta_vs_control" if control else "delta_vs_full"
    value = {(): 0.0}
    for coalition, variant_id in resolved.shapley.items():
        value[coalition] = effects[variant_id][comparison]
    expected = _exact_shapley(value, (1, 2, 3))
    for i in (1, 2, 3):
        assert rows[i]["shapley"] == pytest.approx(expected[i])
    # ... and add up to the effect of removing every step's rows
    assert sum(rows[i]["shapley"] for i in (1, 2, 3)) == pytest.approx(value[(1, 2, 3)])
    assert rows[1]["verdict"] == "helps" and rows[2]["verdict"] == "hurts"
    assert rows[1]["shapley"] == pytest.approx(10, abs=1.5)

    # The groups of steps have no verdict, and are only in the report through the values
    assert all(row["verdict"] is None for row in effects.values() if row["strategy"] == "shapley")
    html = ablation.get_tabular_report().as_raw_html()
    assert "Shapley values" in html and "Shapley: groups" not in html
    assert "The Shapley values are exact, from every order of the 3 steps" in html
    # The rows of steps 1 and 2 have opposite effects, so their shares of the total aren't shown
    assert "of the total Δ" not in html
    assert "Shapley: groups of steps" in ablation.plan().as_raw_html()
    shares = pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=4, control=control).shapley(
        steps=[1, 3]
    )
    shares_html = shares.run().get_tabular_report().as_raw_html()
    assert "of the total Δ" in shares_html

    result = ablation.to_dict()
    json.dumps(result)
    assert [row["step"] for row in result["shapley"]] == [1, 2, 3]

    notes = {step.i: step.notes for step in ablation.annotate().validation_info}
    assert (
        "its Shapley value (its share of the effect of removing every step's rows) is"
        in (notes[1]["ablation"]["text"])
    )


@pytest.mark.parametrize("tbl_type", ["duckdb", "pyspark"])
def test_shapley_on_other_tables(tbl_type):
    if tbl_type != "pyspark":
        pytest.importorskip("ibis")
    ablation = (
        pb.Ablation(_validation(_tbl(tbl_type)), evaluate=_evaluate, repeats=3)
        .shapley(steps=[1, 2])
        .run()
    )
    rows = ablation.get_shapley()
    assert rows["shapley"].to_list() == pytest.approx([10, -10], abs=1.5)
    assert rows["verdict"].to_list() == ["helps", "hurts"]


def test_shapley_reuses_variants_and_dedups_rows():
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_ge(columns="x", value=0)
        .col_vals_gt(columns="x", value=-0.5)  # the same rows as step 1
        .col_vals_ne(columns="g", value="gold")
        .interrogate()
    )
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).drop_each(steps=[1, 2])
    ablation.drop_steps([1, 3], name="both").shapley()
    resolved = ablation._resolve()

    assert resolved.shapley[(1,)] == "step_1"
    assert resolved.shapley[(2,)] == "step_2"
    assert resolved.shapley[(1, 3)] == "both"
    assert resolved.shapley[(1, 2)] == "shapley_1_2"
    # Removing the rows of steps 1 and 2 is the same as removing those of step 1
    assert resolved.same_as["shapley_1_2"] == "step_1"
    assert resolved.same_as["shapley_1_2_3"] == "both"

    ablation.run()
    rows = {row["step"]: row for row in ablation.get_shapley().iter_rows(named=True)}
    # Steps 1 and 2 remove the same rows, so they get the same credit
    assert rows[1]["shapley"] == pytest.approx(rows[2]["shapley"])


def test_shapley_sampled_orders():
    validation = (
        pb.Validate(data=_TBL)
        .col_vals_ge(columns="x", value=0)
        .col_vals_ne(columns="g", value="gold")
        .col_vals_not_null(columns="y")
        .col_vals_lt(columns="rid", value=95)
        .interrogate()
    )
    ablation = pb.Ablation(validation, evaluate=_evaluate, repeats=3).shapley(
        permutations=5, seed=3
    )
    info = ablation._shapley
    assert not info["exact"] and len(info["orders"]) == 5
    # Only the groups that start an order are needed
    expected = {tuple(sorted(o[:m])) for o in info["orders"] for m in range(1, 5)}
    assert set(ablation._resolve().shapley) == expected
    assert len(expected) < 15

    ablation.run()
    rows = ablation.get_shapley()
    effects = _inject_effects(ablation)
    total = effects[ablation._resolve().shapley[(1, 2, 3, 4)]]["delta_vs_control"]
    assert rows["shapley"].sum() == pytest.approx(total)
    assert "estimated from 5 random orders of the 4 steps (seed 3)" in (
        ablation.get_tabular_report().as_raw_html()
    )
    # The same seed gives the same orders
    again = pb.Ablation(validation, evaluate=_evaluate).shapley(permutations=5, seed=3)
    assert again._shapley["orders"] == info["orders"]


def test_shapley_steps_and_errors():
    ablation = pb.Ablation(_validation(_TBL), evaluate=_evaluate)

    ablation.shapley(steps=[1, 2])
    assert ablation._shapley["steps"] == (1, 2)
    assert ablation._shapley["exact"]
    assert repr(ablation) == (
        "Ablation(0 variants, Shapley values of 2 steps, 5 repeats, control='size_matched'; "
        "not run)"
    )

    tbl = _TBL.with_columns(z=pl.lit(1))
    validation = (
        pb.Validate(data=tbl)
        .col_vals_ge(columns="x", value=0)
        .col_vals_ge(columns="z", value=0)  # no rows fail it
        .interrogate()
    )
    with pytest.raises(ValueError, match="needs at least two steps that rows failed"):
        pb.Ablation(validation, evaluate=_evaluate).shapley()
    lonely = pb.Ablation(validation, evaluate=_evaluate)
    with pytest.raises(ValueError):
        lonely.shapley()
    assert "Step 2 has no Shapley value: no rows failed it." in lonely._notes

    every = (
        pb.Validate(data=_TBL)
        .col_vals_lt(columns="rid", value=50)
        .col_vals_ge(columns="rid", value=50)
        .interrogate()
    )
    with pytest.raises(ValueError, match="together, they remove every row"):
        pb.Ablation(every, evaluate=_evaluate).shapley()
    with pytest.raises(ValueError, match="`permutations=` must be a positive integer"):
        ablation.shapley(permutations=0)
    with pytest.raises(TypeError, match="`seed=` must be an integer"):
        ablation.shapley(seed="1")
    with pytest.raises(ValueError, match="can't use step 4"):
        ablation.shapley(steps=[1, 4])
    with pytest.raises(ValueError, match="is reserved"):
        ablation.drop_steps(1, name="shapley_1")

    run = pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=3).drop_each().run()
    with pytest.raises(ValueError, match="There are no Shapley values"):
        run.get_shapley()
    assert run.to_dict()["shapley"] is None
    # Calling `shapley()` again replaces the groups, and the results
    run.shapley(steps=[1, 2])
    assert run._effects is None


def test_shapley_with_pandas_and_few_repeats(monkeypatch):
    import pointblank.ablation as ablation_module

    monkeypatch.setattr(ablation_module, "_select_df_lib", lambda preference: pd)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ablation = (
            pb.Ablation(_validation(_TBL), evaluate=_evaluate, repeats=2)
            .shapley(steps=[1, 2])
            .run()
        )
    shapley = ablation.get_shapley()
    assert isinstance(shapley, pd.DataFrame)
    assert shapley["step"].dtype == "int64"
    assert shapley["shapley"].dtype == "float64"
    assert shapley["verdict"].isna().all()


def test_shapley_with_several_metrics():
    def evaluate(train, *, seed, variant):
        score = _evaluate(train, seed=seed, variant=variant)
        return {"score": score, "other": -score}

    ablation = (
        pb.Ablation(_validation(_TBL), evaluate=evaluate, metric="score", repeats=3)
        .shapley(steps=[1, 2])
        .run()
    )
    shapley = ablation.get_shapley()
    score = shapley.filter(pl.col("metric") == "score")
    other = shapley.filter(pl.col("metric") == "other")
    assert score["shapley"].to_list() == pytest.approx([-v for v in other["shapley"].to_list()])
    assert other["verdict"].is_null().all()
    assert "Shapley values" in ablation.get_tabular_report(metric="other").as_raw_html()


# ----------------------------------------------------------------------------------------------
# YAML
# ----------------------------------------------------------------------------------------------


def test_yaml_ablation_inject_and_shapley():
    config = """
tbl: small_table
steps:
- col_vals_ge:
    columns: x
    value: 0
- col_vals_ne:
    columns: g
    value: gold
ablation:
  repeats: 3
  strategies:
  - inject:
      corruption: "null"
      step: 1
      rates: [0.05]
  - shapley:
      permutations: 10
"""
    ablation = pb.yaml_ablation(config, evaluate=_evaluate, set_tbl=_TBL)
    assert [spec.id for spec in ablation._specs] == ["inject:null:x_0.05"]
    assert ablation._shapley["steps"] == (1, 2)
    code = pb.yaml_to_python(config)
    assert '    .inject(corruption="null", step=1, rates=[0.05])' in code
    assert "    .shapley(permutations=10)" in code

    bad = config.replace('      corruption: "null"\n', "")
    with pytest.raises(pb.yaml.YAMLValidationError, match="\\['corruption'\\]"):
        pb.yaml_ablation(bad, evaluate=_evaluate, set_tbl=_TBL)
