import json

import pandas as pd
import polars as pl
import pytest

import pointblank as pb
from pointblank.datascan import DataScan, DataScanDiff


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df():
    return pl.DataFrame(
        {
            "id": [1, 2, 3, None],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "score": [95.5, 82.3, None, 91.0],
            "active": [True, False, True, True],
        }
    )


@pytest.fixture
def scan(sample_df):
    return DataScan(data=sample_df, tbl_name="test_table")


# ── to_dict ───────────────────────────────────────────────────────────────────


def test_to_dict_structure(scan):
    d = scan.to_dict()
    assert "metadata" in d
    assert "columns" in d
    assert d["metadata"]["table_name"] == "test_table"
    assert d["metadata"]["row_count"] == 4
    assert d["metadata"]["columns"] == ["id", "name", "score", "active"]
    assert len(d["columns"]) == 4


def test_to_dict_column_entries(scan):
    d = scan.to_dict()
    col_entry = d["columns"][0]
    assert "colname" in col_entry
    assert "coltype" in col_entry
    assert "sample_data" in col_entry
    assert "statistics" in col_entry
    assert isinstance(col_entry["statistics"], dict)


def test_to_dict_no_svg_icons(scan):
    d = scan.to_dict()
    json_str = json.dumps(d, default=str)
    assert "<svg" not in json_str


def test_to_dict_statistics_present(scan):
    d = scan.to_dict()
    id_col = next(c for c in d["columns"] if c["colname"] == "id")
    stats = id_col["statistics"]
    assert "n_missing" in stats
    assert "n_unique" in stats
    assert "mean" in stats


# ── from_dict round-trip ──────────────────────────────────────────────────────


def test_from_dict_round_trip(scan):
    d = scan.to_dict()
    restored = DataScan.from_dict(d)

    assert restored.tbl_name == scan.tbl_name
    assert restored.profile.row_count == scan.profile.row_count
    assert restored.profile.columns == scan.profile.columns
    assert len(restored.profile.column_profiles) == len(scan.profile.column_profiles)


def test_from_dict_preserves_stats(scan):
    d = scan.to_dict()
    restored = DataScan.from_dict(d)

    for orig, rest in zip(scan.profile.column_profiles, restored.profile.column_profiles):
        orig_stats = {s.name: s.val for s in orig.statistics}
        rest_stats = {s.name: s.val for s in rest.statistics}
        assert orig_stats == rest_stats, f"Stats mismatch for column {orig.colname}"


def test_from_dict_preserves_sample_data(scan):
    d = scan.to_dict()
    restored = DataScan.from_dict(d)

    for orig, rest in zip(scan.profile.column_profiles, restored.profile.column_profiles):
        assert list(orig.sample_data) == list(rest.sample_data)


def test_from_dict_no_original_data(scan):
    d = scan.to_dict()
    restored = DataScan.from_dict(d)
    assert restored.nw_data is None


# ── to_json / from_json ──────────────────────────────────────────────────────


def test_json_round_trip(scan):
    json_str = scan.to_json()
    restored = DataScan.from_json(json_str)

    assert restored.tbl_name == scan.tbl_name
    assert restored.profile.row_count == scan.profile.row_count
    assert len(restored.profile.column_profiles) == len(scan.profile.column_profiles)


def test_to_json_is_valid_json(scan):
    json_str = scan.to_json()
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert "metadata" in parsed
    assert "columns" in parsed


# ── save_to_json / load_from_json ─────────────────────────────────────────────


def test_file_round_trip(scan, tmp_path):
    filepath = str(tmp_path / "scan.json")
    scan.save_to_json(filepath)

    loaded = DataScan.load_from_json(filepath)
    assert loaded.tbl_name == scan.tbl_name
    assert loaded.profile.row_count == scan.profile.row_count

    for orig, rest in zip(scan.profile.column_profiles, loaded.profile.column_profiles):
        orig_stats = {s.name: s.val for s in orig.statistics}
        rest_stats = {s.name: s.val for s in rest.statistics}
        assert orig_stats == rest_stats


def test_saved_file_is_readable_json(scan, tmp_path):
    filepath = str(tmp_path / "scan.json")
    scan.save_to_json(filepath)

    with open(filepath) as f:
        content = f.read()
    parsed = json.loads(content)
    assert parsed["metadata"]["table_name"] == "test_table"


# ── Pandas input ──────────────────────────────────────────────────────────────


def test_pandas_round_trip():
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": ["a", "b", "c"],
        }
    )
    scan = DataScan(data=df, tbl_name="pandas_test")
    d = scan.to_dict()
    restored = DataScan.from_dict(d)

    assert restored.profile.row_count == 3
    assert len(restored.profile.column_profiles) == 2


# ── compare: no changes ──────────────────────────────────────────────────────


def test_compare_identical(scan):
    diff = scan.compare(scan)
    assert not diff.has_changes
    assert diff.columns_added == []
    assert diff.columns_removed == []
    assert diff.columns_type_changed == []
    assert diff.row_count_diff == (4, 4)


def test_compare_identical_loaded(scan):
    restored = DataScan.from_dict(scan.to_dict())
    diff = scan.compare(restored)
    assert not diff.has_changes


# ── compare: schema drift ────────────────────────────────────────────────────


def test_compare_column_added():
    base = DataScan(data=pl.DataFrame({"a": [1, 2]}))
    cur = DataScan(data=pl.DataFrame({"a": [1, 2], "b": [3, 4]}))

    diff = cur.compare(base)
    assert diff.columns_added == ["b"]
    assert diff.columns_removed == []
    assert diff.has_changes


def test_compare_column_removed():
    base = DataScan(data=pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
    cur = DataScan(data=pl.DataFrame({"a": [1, 2]}))

    diff = cur.compare(base)
    assert diff.columns_added == []
    assert diff.columns_removed == ["b"]
    assert diff.has_changes


def test_compare_column_type_changed():
    base = DataScan(data=pl.DataFrame({"a": [1, 2, 3]}))
    cur = DataScan(data=pl.DataFrame({"a": ["x", "y", "z"]}))

    diff = cur.compare(base)
    assert diff.columns_type_changed == ["a"]
    assert diff.has_changes


# ── compare: statistical drift ───────────────────────────────────────────────


def test_compare_stat_changes():
    base = DataScan(data=pl.DataFrame({"x": [1, 2, 3]}))
    cur = DataScan(data=pl.DataFrame({"x": [10, 20, 30]}))

    diff = cur.compare(base)
    assert diff.has_changes

    x_diff = next(d for d in diff.column_diffs if d.colname == "x")
    assert "mean" in x_diff.stat_diffs
    baseline_mean, current_mean = x_diff.stat_diffs["mean"]
    assert baseline_mean == 2.0
    assert current_mean == 20.0


def test_compare_null_rate_change():
    base = DataScan(data=pl.DataFrame({"x": [1, 2, 3]}))
    cur = DataScan(data=pl.DataFrame({"x": [1, None, None]}))

    diff = cur.compare(base)
    x_diff = next(d for d in diff.column_diffs if d.colname == "x")
    assert "n_missing" in x_diff.stat_diffs
    assert x_diff.stat_diffs["n_missing"] == (0, 2)


def test_compare_row_count_diff():
    base = DataScan(data=pl.DataFrame({"x": [1, 2, 3]}))
    cur = DataScan(data=pl.DataFrame({"x": [1, 2, 3, 4, 5]}))

    diff = cur.compare(base)
    assert diff.row_count_diff == (3, 5)


# ── compare: to_dict ─────────────────────────────────────────────────────────


def test_compare_to_dict():
    base = DataScan(data=pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
    cur = DataScan(data=pl.DataFrame({"a": [10, 20], "c": [5, 6]}))

    diff = cur.compare(base)
    d = diff.to_dict()

    assert "row_count" in d
    assert "columns_added" in d
    assert "columns_removed" in d
    assert "stat_diffs" in d
    assert "c" in d["columns_added"]
    assert "b" in d["columns_removed"]
    assert "a" in d["stat_diffs"]


# ── compare: tabular report ──────────────────────────────────────────────────


def test_compare_tabular_report():
    from great_tables import GT

    base = DataScan(data=pl.DataFrame({"a": [1, 2], "b": [3, 4]}), tbl_name="v1")
    cur = DataScan(data=pl.DataFrame({"a": [10, 20], "c": [5, 6]}), tbl_name="v2")

    diff = cur.compare(base)
    report = diff.get_tabular_report()
    assert isinstance(report, GT)


def test_compare_tabular_report_no_changes(scan):
    diff = scan.compare(scan)
    report = diff.get_tabular_report()
    assert isinstance(report, type(report))


# ── compare: repr ─────────────────────────────────────────────────────────────


def test_compare_repr():
    base = DataScan(data=pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
    cur = DataScan(data=pl.DataFrame({"a": [10, 20], "c": [5, 6]}))

    diff = cur.compare(base)
    r = repr(diff)
    assert "DataScanDiff" in r
    assert "added=1" in r
    assert "removed=1" in r


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_from_dict_missing_table_name():
    d = {
        "metadata": {"row_count": 0, "columns": []},
        "columns": [],
    }
    restored = DataScan.from_dict(d)
    assert restored.tbl_name is None


def test_compare_both_empty():
    base = DataScan(data=pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}))
    cur = DataScan(data=pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)}))

    diff = cur.compare(base)
    assert not diff.has_changes


def test_from_dict_preserves_bool_freqs():
    df = pl.DataFrame({"flag": [True, False, True]})
    scan = DataScan(data=df)
    d = scan.to_dict()
    restored = DataScan.from_dict(d)

    orig_stats = {s.name: s.val for s in scan.profile.column_profiles[0].statistics}
    rest_stats = {s.name: s.val for s in restored.profile.column_profiles[0].statistics}
    assert orig_stats["freqs"] == rest_stats["freqs"]
