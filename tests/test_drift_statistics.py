import polars as pl
import pytest

from pointblank.datascan import (
    DataScan,
    DataScanDiff,
    _compute_ks_statistic,
    _compute_psi_categorical,
    _compute_psi_numeric,
)


# ── PSI numeric ─────────────────────────────────────────────────────────────


def test_psi_numeric_identical():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    psi = _compute_psi_numeric(vals, vals)
    assert psi is not None
    assert psi == pytest.approx(0.0, abs=1e-3)


def test_psi_numeric_slight_shift():
    base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    cur = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5]
    psi = _compute_psi_numeric(base, cur)
    assert psi is not None
    assert psi < 0.25


def test_psi_numeric_major_shift():
    base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    cur = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    psi = _compute_psi_numeric(base, cur)
    assert psi is not None
    assert psi > 0.25


def test_psi_numeric_too_few_values():
    assert _compute_psi_numeric([1.0, 2.0], [3.0, 4.0]) is None


def test_psi_numeric_empty():
    assert _compute_psi_numeric([], []) is None


# ── PSI categorical ─────────────────────────────────────────────────────────


def test_psi_categorical_identical():
    freqs = {"a": 50, "b": 30, "c": 20}
    psi = _compute_psi_categorical(freqs, freqs)
    assert psi is not None
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_psi_categorical_shifted():
    base = {"a": 50, "b": 30, "c": 20}
    cur = {"a": 20, "b": 30, "c": 50}
    psi = _compute_psi_categorical(base, cur)
    assert psi is not None
    assert psi > 0.0


def test_psi_categorical_new_category():
    base = {"a": 50, "b": 50}
    cur = {"a": 40, "b": 40, "c": 20}
    psi = _compute_psi_categorical(base, cur)
    assert psi is not None
    assert psi > 0.0


def test_psi_categorical_empty():
    assert _compute_psi_categorical({}, {}) is None


# ── KS statistic ────────────────────────────────────────────────────────────


def test_ks_identical():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = _compute_ks_statistic(vals, vals)
    assert result is not None
    assert result["statistic"] == 0.0
    assert result["p_value"] == 1.0


def test_ks_different_distributions():
    base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    cur = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]
    result = _compute_ks_statistic(base, cur)
    assert result is not None
    assert result["statistic"] == 1.0
    assert result["p_value"] < 0.05


def test_ks_too_few_values():
    assert _compute_ks_statistic([1.0], [2.0]) is None


# ── Integration: DataScanDiff with drift scores ─────────────────────────────


def test_drift_scores_numeric_column():
    base = pl.DataFrame({"x": list(range(100))})
    cur = pl.DataFrame({"x": list(range(50, 150))})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    x_diff = next(d for d in diff.column_diffs if d.colname == "x")

    assert "psi" in x_diff.drift_scores
    assert "ks_statistic" in x_diff.drift_scores
    assert "ks_p_value" in x_diff.drift_scores
    assert x_diff.drift_scores["psi"] > 0.0


def test_drift_scores_categorical_column():
    base = pl.DataFrame({"cat": ["a"] * 50 + ["b"] * 30 + ["c"] * 20})
    cur = pl.DataFrame({"cat": ["a"] * 20 + ["b"] * 30 + ["c"] * 50})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    cat_diff = next(d for d in diff.column_diffs if d.colname == "cat")

    assert "psi" in cat_diff.drift_scores
    assert "ks_statistic" not in cat_diff.drift_scores


def test_drift_scores_identical_data():
    df = pl.DataFrame({"x": list(range(20)), "y": ["a"] * 10 + ["b"] * 10})
    scan = DataScan(data=df)
    diff = scan.compare(scan)

    for d in diff.column_diffs:
        assert d.drift_scores == {}, f"{d.colname} has unexpected drift scores"
    assert not diff.has_changes


def test_drift_scores_no_raw_data():
    df = pl.DataFrame({"x": list(range(20))})
    scan = DataScan(data=df)
    restored = DataScan.from_dict(scan.to_dict())

    diff_data = DataScan(data=pl.DataFrame({"x": list(range(50, 70))}))
    diff = diff_data.compare(restored)

    x_diff = next(d for d in diff.column_diffs if d.colname == "x")
    assert "ks_statistic" not in x_diff.drift_scores
    assert "psi" not in x_diff.drift_scores


def test_drift_scores_in_to_dict():
    base = pl.DataFrame({"x": list(range(100))})
    cur = pl.DataFrame({"x": list(range(50, 150))})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    d = diff.to_dict()

    assert "drift_scores" in d
    assert "x" in d["drift_scores"]
    assert "psi" in d["drift_scores"]["x"]


def test_drift_scores_in_tabular_report():
    from great_tables import GT

    base = pl.DataFrame({"x": list(range(100))})
    cur = pl.DataFrame({"x": list(range(50, 150))})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    report = diff.get_tabular_report()
    assert isinstance(report, GT)


def test_drift_scores_not_in_report_when_absent():
    from great_tables import GT

    scan = DataScan.from_dict(
        {
            "metadata": {"row_count": 5, "columns": ["x"]},
            "columns": [
                {"colname": "x", "coltype": "Int64", "statistics": {}, "sample_data": []},
            ],
        }
    )
    diff = scan.compare(scan)
    report = diff.get_tabular_report()
    assert isinstance(report, GT)


def test_drift_type_changed_no_scores():
    base = pl.DataFrame({"x": [1, 2, 3]})
    cur = pl.DataFrame({"x": ["a", "b", "c"]})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    x_diff = next(d for d in diff.column_diffs if d.colname == "x")
    assert x_diff.status == "type_changed"
    assert x_diff.drift_scores == {}


def test_drift_added_removed_no_scores():
    base = pl.DataFrame({"a": [1, 2]})
    cur = pl.DataFrame({"b": [3, 4]})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    for d in diff.column_diffs:
        assert d.drift_scores == {}


# ── PSI interpretation thresholds ────────────────────────────────────────────


def test_psi_no_drift_threshold():
    """PSI < 0.1 indicates no significant drift."""
    base = list(range(100))
    cur = [x + 1 for x in base]
    base_f = [float(x) for x in base]
    cur_f = [float(x) for x in cur]
    psi = _compute_psi_numeric(base_f, cur_f)
    assert psi is not None
    assert psi < 0.1


def test_psi_major_drift_threshold():
    """PSI > 0.25 indicates significant population shift."""
    import random

    random.seed(42)
    base = [random.gauss(0, 1) for _ in range(200)]
    cur = [random.gauss(5, 1) for _ in range(200)]
    psi = _compute_psi_numeric(base, cur)
    assert psi is not None
    assert psi > 0.25


def test_boolean_column_drift():
    base = pl.DataFrame({"flag": [True] * 80 + [False] * 20})
    cur = pl.DataFrame({"flag": [True] * 50 + [False] * 50})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    flag_diff = next(d for d in diff.column_diffs if d.colname == "flag")
    assert "psi" in flag_diff.drift_scores
    assert flag_diff.drift_scores["psi"] > 0.0


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_nulls_in_numeric_column():
    base = pl.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]})
    cur = pl.DataFrame(
        {
            "x": [
                100.0,
                None,
                300.0,
                400.0,
                500.0,
                600.0,
                700.0,
                800.0,
                900.0,
                1000.0,
                1100.0,
                1200.0,
            ]
        }
    )

    diff = DataScan(data=cur).compare(DataScan(data=base))
    x_diff = next(d for d in diff.column_diffs if d.colname == "x")
    assert "psi" in x_diff.drift_scores
    assert x_diff.drift_scores["psi"] > 0.0


def test_nulls_in_categorical_column():
    base = pl.DataFrame({"cat": ["a", "a", "b", None, "b", "a", "b", "a", "b", "a"]})
    cur = pl.DataFrame({"cat": ["c", "c", "c", None, None, "c", "a", "a", "a", "a"]})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    cat_diff = next(d for d in diff.column_diffs if d.colname == "cat")
    assert "psi" in cat_diff.drift_scores


def test_all_null_column():
    base = pl.DataFrame({"x": pl.Series([None, None, None], dtype=pl.Float64)})
    cur = pl.DataFrame({"x": pl.Series([None, None, None], dtype=pl.Float64)})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    x_diff = next(d for d in diff.column_diffs if d.colname == "x")
    assert x_diff.drift_scores == {}


def test_constant_numeric_column():
    base = pl.DataFrame({"x": [5.0] * 20})
    cur = pl.DataFrame({"x": [5.0] * 20})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    x_diff = next(d for d in diff.column_diffs if d.colname == "x")
    assert "psi" not in x_diff.drift_scores


def test_single_category():
    base = pl.DataFrame({"cat": ["a"] * 20})
    cur = pl.DataFrame({"cat": ["a"] * 20})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    cat_diff = next(d for d in diff.column_diffs if d.colname == "cat")
    assert cat_diff.drift_scores == {}


def test_category_disappears():
    base = {"a": 50, "b": 30, "c": 20}
    cur = {"a": 80, "b": 20}
    psi = _compute_psi_categorical(base, cur)
    assert psi is not None
    assert psi > 0.0


def test_category_appears():
    base = {"a": 80, "b": 20}
    cur = {"a": 50, "b": 30, "c": 20}
    psi = _compute_psi_categorical(base, cur)
    assert psi is not None
    assert psi > 0.0


def test_asymmetric_sizes():
    base = pl.DataFrame({"x": list(range(20))})
    cur = pl.DataFrame({"x": list(range(1000))})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    x_diff = next(d for d in diff.column_diffs if d.colname == "x")
    assert "psi" in x_diff.drift_scores
    assert "ks_statistic" in x_diff.drift_scores


def test_negative_and_zero_values():
    base = pl.DataFrame({"x": [-10.0, -5.0, 0.0, 5.0, 10.0, -8.0, -3.0, 2.0, 7.0, 12.0]})
    cur = pl.DataFrame({"x": [-100.0, -50.0, 0.0, 50.0, 100.0, -80.0, -30.0, 20.0, 70.0, 120.0]})

    diff = DataScan(data=cur).compare(DataScan(data=base))
    x_diff = next(d for d in diff.column_diffs if d.colname == "x")
    assert "psi" in x_diff.drift_scores
    assert x_diff.drift_scores["psi"] > 0.0


def test_psi_is_non_negative():
    import random

    random.seed(99)
    for _ in range(10):
        base = [random.gauss(0, 1) for _ in range(100)]
        cur = [random.gauss(random.uniform(-2, 2), 1) for _ in range(100)]
        psi = _compute_psi_numeric(base, cur)
        if psi is not None:
            assert psi >= 0.0, f"PSI should be non-negative, got {psi}"


def test_ks_overlapping_distributions():
    import random

    random.seed(42)
    base = [random.gauss(0, 1) for _ in range(100)]
    cur = [random.gauss(0.5, 1) for _ in range(100)]
    result = _compute_ks_statistic(base, cur)
    assert result is not None
    assert 0.0 < result["statistic"] < 1.0
    assert 0.0 < result["p_value"] < 1.0


def test_mixed_column_types():
    base = pl.DataFrame(
        {
            "num": list(range(20)),
            "cat": ["a"] * 10 + ["b"] * 10,
            "flag": [True] * 15 + [False] * 5,
        }
    )
    cur = pl.DataFrame(
        {
            "num": list(range(50, 70)),
            "cat": ["a"] * 5 + ["b"] * 10 + ["c"] * 5,
            "flag": [True] * 10 + [False] * 10,
        }
    )

    diff = DataScan(data=cur).compare(DataScan(data=base))

    num_diff = next(d for d in diff.column_diffs if d.colname == "num")
    assert "psi" in num_diff.drift_scores
    assert "ks_statistic" in num_diff.drift_scores

    cat_diff = next(d for d in diff.column_diffs if d.colname == "cat")
    assert "psi" in cat_diff.drift_scores
    assert "ks_statistic" not in cat_diff.drift_scores

    flag_diff = next(d for d in diff.column_diffs if d.colname == "flag")
    assert "psi" in flag_diff.drift_scores
