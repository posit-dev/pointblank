from __future__ import annotations

import datetime
import json

import ibis
import narwhals as nw
import pandas as pd
import polars as pl
import pytest

import pointblank as pb
from pointblank.compare import Comparison, SchemaDiff, compare


# Source and target tables covering every row status: ids 1 and 4 match (4 via NaN == null),
# id 2 changed, id 3 missing, id 5 extra, id 6 duplicated in the source, and one null key per side
SOURCE = {
    "id": [1, 2, 3, 4, 6, 6, None],
    "amount": [10.0, 20.0, 30.0, float("nan"), 1.0, 2.0, 9.0],
    "region": ["N", "S", "E", "W", "x", "y", "z"],
}
TARGET = {
    "id": [1, 2, 4, 5, 6, None],
    "amount": [10.0, 20.5, None, 50.0, 1.0, 8.0],
    "region": ["N", "S", "W", "N", "x", "q"],
}

EXPECTED_COUNTS = {"match": 2, "changed": 1, "missing": 2, "extra": 2, "dup_key": 2}


def _make(backend: str, data: dict, name: str):
    df = pl.DataFrame(data)
    if backend == "polars":
        return df
    if backend == "pandas":
        return df.to_pandas()
    if backend == "polars_lazy":
        return df.lazy()
    if backend == "duckdb":
        con = _make.con  # one connection so both tables can be joined in DuckDB
        return con.create_table(name, df, overwrite=True)
    raise ValueError(backend)


_make.con = ibis.duckdb.connect()

BACKENDS = ["polars", "pandas", "polars_lazy", "duckdb"]


def _native_rows(frame) -> list[dict]:
    return nw.from_native(frame).rows(named=True)


# ----------------------------------------------------------------------------------------------
# Row statuses and counts
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_keyed_status_counts_across_backends(backend):
    cmp = compare(_make(backend, SOURCE, "s"), _make(backend, TARGET, "t"), keys="id")

    assert isinstance(cmp, Comparison)
    assert cmp.status_counts == EXPECTED_COUNTS
    assert cmp.column_mismatches == {"amount": 1, "region": 0}
    assert cmp.n_source_rows == 7
    assert cmp.n_target_rows == 6
    assert cmp.n == 9
    assert cmp.n_passed == 2
    assert cmp.n_failed == 7
    assert not cmp.identical
    assert not bool(cmp)


@pytest.mark.parametrize("backend", BACKENDS)
def test_keyed_samples_across_backends(backend):
    cmp = compare(_make(backend, SOURCE, "s"), _make(backend, TARGET, "t"), keys="id")

    changed = _native_rows(cmp.rows("changed"))
    assert changed == [
        {
            "id": 2,
            "amount_source": 20.0,
            "amount_target": 20.5,
            "region_source": "S",
            "region_target": "S",
        }
    ]

    missing = _native_rows(cmp.rows("missing"))
    assert {r["region"] for r in missing} == {"E", "z"}  # id 3, plus the null-key source row

    extra = _native_rows(cmp.rows("extra"))
    assert {r["region"] for r in extra} == {"N", "q"}  # id 5, plus the null-key target row

    dup = _native_rows(cmp.rows("dup_key"))
    assert dup == [{"id": 6, "n_source": 2, "n_target": 1}]

    assert {r["id"] for r in _native_rows(cmp.rows("match"))} == {1, 4}


def test_samples_keep_eager_input_type():
    s, t = pl.DataFrame(SOURCE), pl.DataFrame(TARGET)
    assert isinstance(compare(s, t, keys="id").rows("changed"), pl.DataFrame)
    assert isinstance(
        compare(s.to_pandas(), t.to_pandas(), keys="id").rows("changed"), pd.DataFrame
    )
    # Lazy database backends are collected to Polars
    assert isinstance(
        compare(_make("duckdb", SOURCE, "s"), _make("duckdb", TARGET, "t"), keys="id").rows(
            "missing"
        ),
        pl.DataFrame,
    )


def test_mixed_backends_are_aligned():
    # A Pandas source (with a float key, because of the null) against a Polars lazy target
    cmp = compare(pl.DataFrame(SOURCE).to_pandas(), pl.DataFrame(TARGET).lazy(), keys="id")
    assert cmp.status_counts == EXPECTED_COUNTS


def test_identical_tables():
    df = pl.DataFrame({"id": [1, 2, 3], "x": ["a", "b", None]})
    cmp = compare(df, df.clone(), keys="id")

    assert cmp.identical
    assert bool(cmp)
    assert cmp.status_counts == {"match": 3, "changed": 0, "missing": 0, "extra": 0, "dup_key": 0}
    assert cmp.n_failed == 0
    assert len(cmp.rows("changed")) == 0


def test_empty_tables():
    df = pl.DataFrame({"id": [], "x": []}, schema={"id": pl.Int64, "x": pl.String})
    cmp = compare(df, df, keys="id")
    assert cmp.n == 0
    assert cmp.identical


def test_composite_keys():
    s = pl.DataFrame({"k1": [1, 1, 2], "k2": ["a", "b", "a"], "v": [1, 2, 3]})
    t = pl.DataFrame({"k1": [1, 1, 2], "k2": ["a", "b", "b"], "v": [1, 5, 3]})
    cmp = compare(s, t, keys=["k1", "k2"])

    assert cmp.status_counts == {"match": 1, "changed": 1, "missing": 1, "extra": 1, "dup_key": 0}
    assert _native_rows(cmp.rows("changed")) == [{"k1": 1, "k2": "b", "v_source": 2, "v_target": 5}]


def test_rows_limit_and_cache():
    s = pl.DataFrame({"id": list(range(100)), "v": [0] * 100})
    t = pl.DataFrame({"id": list(range(100)), "v": [1] * 100})
    cmp = compare(s, t, keys="id")

    assert cmp.n_changed == 100
    assert len(cmp.rows("changed", limit=5)) == 5
    assert cmp.rows("changed", limit=5) is cmp.rows("changed", limit=5)

    with pytest.raises(ValueError, match="status"):
        cmp.rows("bogus")


# ----------------------------------------------------------------------------------------------
# Duplicate and null keys
# ----------------------------------------------------------------------------------------------


def test_dup_key_units_use_larger_occurrence_count():
    s = pl.DataFrame({"id": [1, 1, 1, 2], "v": [1, 1, 1, 2]})
    t = pl.DataFrame({"id": [1, 1, 2, 3, 3], "v": [1, 1, 2, 3, 3]})
    cmp = compare(s, t, keys="id")

    # key 1: max(3, 2) = 3 units; key 3: max(0, 2) = 2 units
    assert cmp.n_dup_key == 5
    assert cmp.n_dup_key_values == 2
    assert cmp.n_match == 1
    assert cmp.n_source_rows == 4
    assert cmp.n_target_rows == 5
    dup = sorted(_native_rows(cmp.rows("dup_key")), key=lambda r: r["id"])
    assert dup == [{"id": 1, "n_source": 3, "n_target": 2}, {"id": 3, "n_source": 0, "n_target": 2}]


def test_dup_keys_compare_mode_reports_pairwise_differences():
    s = pl.DataFrame({"id": [1, 1, 2, 2], "a": [1, 1, 5, 5], "b": ["x", "x", "y", "y"]})
    t = pl.DataFrame({"id": [1, 2], "a": [1, 6], "b": ["x", "y"]})
    cmp = compare(s, t, keys="id", dup_keys="compare")

    # Status counts are the same as with "flag"
    assert cmp.n_dup_key == 4
    html = cmp.get_tabular_report().as_raw_html()
    assert "all pairs identical" in html  # key 1
    assert "pairs differ in" in html  # key 2 (column a)


def test_dup_keys_compare_with_composite_keys_only_uses_duplicated_keys():
    s = pl.DataFrame({"k1": [1, 1, 2], "k2": ["a", "a", "b"], "v": [1, 1, 1]})
    t = pl.DataFrame({"k1": [1, 2, 1], "k2": ["a", "b", "b"], "v": [2, 1, 1]})
    cmp = compare(s, t, keys=["k1", "k2"], dup_keys="compare")
    pairs = cmp._dup_pairs(limit=10)
    assert {(r["k1"], r["k2"]) for r in pairs.rows(named=True)} == {(1, "a")}


def test_null_keys_count_as_missing_and_extra():
    s = pl.DataFrame({"id": [1, None, None], "v": [1, 2, 3]})
    t = pl.DataFrame({"id": [1, None], "v": [1, 2]})
    cmp = compare(s, t, keys="id")
    assert cmp.status_counts == {"match": 1, "changed": 0, "missing": 2, "extra": 1, "dup_key": 0}


# ----------------------------------------------------------------------------------------------
# Positional mode
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["polars", "pandas", "polars_lazy"])
def test_positional_mode(backend):
    s = {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]}
    t = {"a": [1, 2, 9], "b": ["w", "X", "y"]}
    cmp = compare(_make(backend, s, "s"), _make(backend, t, "t"))

    assert cmp.mode == "positional"
    assert cmp.status_counts == {"match": 1, "changed": 2, "missing": 1, "extra": 0, "dup_key": 0}
    changed = _native_rows(cmp.rows("changed"))
    assert sorted(r["_row_num_"] for r in changed) == [2, 3]
    assert _native_rows(cmp.rows("missing"))[0]["_row_num_"] == 4


def test_positional_mode_requires_order_on_unordered_backends():
    s = _make("duckdb", {"a": [1, 2]}, "ps")
    t = _make("duckdb", {"a": [1, 3]}, "pt")
    with pytest.raises(ValueError, match="no defined row order"):
        compare(s, t)

    cmp = compare(s, t, order_by="a")
    assert cmp.status_counts["match"] == 1
    assert cmp.status_counts["changed"] == 1


# ----------------------------------------------------------------------------------------------
# Value comparison: nulls, NaN, tolerance, normalization, dtype differences
# ----------------------------------------------------------------------------------------------


def test_null_equal():
    s = pl.DataFrame({"id": [1, 2], "v": [None, 1]})
    t = pl.DataFrame({"id": [1, 2], "v": [None, None]})
    assert compare(s, t, keys="id").status_counts["changed"] == 1
    assert compare(s, t, keys="id", null_equal=False).status_counts["changed"] == 2


def test_nan_is_treated_as_null():
    s = pl.DataFrame({"id": [1, 2], "v": [float("nan"), float("nan")]})
    t = pl.DataFrame({"id": [1, 2], "v": [None, float("nan")]})
    assert compare(s, t, keys="id").identical


@pytest.mark.parametrize(
    "tolerance, n_changed",
    [
        # diffs: 1.0 -> 1.4 (0.4), 100.0 -> 100.9 (0.9), 5.0 -> 5.6 (0.6)
        (None, 3),
        (0.5, 2),  # global absolute tolerance: covers 0.4 only
        ({"atol": 0.5}, 2),
        ({"rtol": 0.01}, 2),  # 1% of |source|: covers 100 -> 100.9 only
        ({"atol": 0.5, "rtol": 0.01}, 1),  # bounds 0.51, 1.5, 0.55: only 5.0 -> 5.6 differs
        ({"v": 0.5}, 2),  # per-column
        ({"v": {"rtol": 0.01}}, 2),
    ],
)
def test_numeric_tolerance(tolerance, n_changed):
    s = pl.DataFrame({"id": [1, 2, 3], "v": [1.0, 100.0, 5.0]})
    t = pl.DataFrame({"id": [1, 2, 3], "v": [1.4, 100.9, 5.6]})
    assert compare(s, t, keys="id", tolerance=tolerance).n_changed == n_changed


def test_global_tolerance_skips_non_numeric_columns():
    s = pl.DataFrame({"id": [1], "v": [1.0], "s": ["a"]})
    t = pl.DataFrame({"id": [1], "v": [1.2], "s": ["b"]})
    cmp = compare(s, t, keys="id", tolerance=0.5)
    assert cmp.column_mismatches == {"v": 0, "s": 1}


@pytest.mark.parametrize("backend", ["polars", "pandas", "duckdb"])
@pytest.mark.parametrize("tol", ["2s", datetime.timedelta(seconds=2), {"atol": "2s"}])
def test_datetime_tolerance(backend, tol):
    base = datetime.datetime(2024, 1, 1)
    s = {"id": [1, 2], "ts": [base, base]}
    t = {
        "id": [1, 2],
        "ts": [base + datetime.timedelta(seconds=1), base + datetime.timedelta(seconds=5)],
    }
    cmp = compare(
        _make(backend, s, "ts_s"), _make(backend, t, "ts_t"), keys="id", tolerance={"ts": tol}
    )
    assert cmp.n_changed == 1


def test_invalid_tolerance():
    s = pl.DataFrame({"id": [1], "v": [1.0]})
    with pytest.raises(ValueError, match="duration"):
        compare(s, s, keys="id", tolerance={"v": "soon"})
    with pytest.raises(ValueError, match="Unknown tolerance"):
        compare(s, s, keys="id", tolerance={"v": {"abs": 1}})
    with pytest.raises(ValueError, match="isn't in the source"):
        compare(s, s, keys="id", tolerance={"nope": 1})


@pytest.mark.parametrize("backend", ["polars", "pandas", "duckdb"])
def test_normalize(backend):
    s = {"id": [1, 2, 3], "name": [" Alice ", "BOB", "c  d"], "x": [1.234, 2.0, 3.0]}
    t = {"id": [1, 2, 3], "name": ["alice", "bob", "c d"], "x": [1.23, 2.0, 3.0]}
    S, T = _make(backend, s, "ns"), _make(backend, t, "nt")

    assert compare(S, T, keys="id").n_changed == 3
    cmp = compare(
        S,
        T,
        keys="id",
        normalize={"name": ["strip", "lower", "collapse_whitespace"], "x": {"round": 2}},
    )
    assert cmp.n_changed == 0
    # Only lowercasing fixes "BOB"; reported values are the original (unnormalized) ones
    lowered = compare(S, T, keys="id", normalize={"name": "lower"})
    assert lowered.n_changed == 2
    names = {r["name_source"] for r in _native_rows(lowered.rows("changed"))}
    assert names == {" Alice ", "c  d"}


def test_invalid_normalize():
    s = pl.DataFrame({"id": [1], "v": ["a"]})
    with pytest.raises(ValueError, match="Unknown normalization"):
        compare(s, s, keys="id", normalize={"v": ["squash"]})
    with pytest.raises(ValueError, match="number of digits"):
        compare(s, s, keys="id", normalize={"v": ["round"]})


def test_values_compared_across_dtypes():
    # Integer vs float compares numerically; string vs integer compares as strings
    s = pl.DataFrame({"id": [1, 2], "n": [1, 2], "code": ["1", "2"]})
    t = pl.DataFrame({"id": [1, 2], "n": [1.0, 2.5], "code": [1, 3]})
    cmp = compare(s, t, keys="id", schema="common")
    assert cmp.column_mismatches == {"n": 1, "code": 1}


# ----------------------------------------------------------------------------------------------
# Schema handling
# ----------------------------------------------------------------------------------------------


def test_schema_diff_and_strict_mode():
    s = pl.DataFrame({"id": [1, 2], "a": [1, 2], "b": ["x", "y"], "Gone": [0, 0]})
    t = pl.DataFrame(
        {"id": [1, 2], "b": ["x", "y"], "a": [1.0, 2.0], "gone": [0, 0], "new": [1, 1]}
    )
    cmp = compare(s, t, keys="id")

    diff = cmp.schema_diff
    assert isinstance(diff, SchemaDiff)
    assert diff.only_in_source == ["Gone"]
    assert diff.only_in_target == ["gone", "new"]
    assert diff.dtype_changed == [("a", "Int64", "Float64")]
    assert diff.case_only == [("Gone", "gone")]
    assert diff.reordered
    assert not diff.matches

    # Values of shared columns all match, but the strict schema check fails every row
    assert cmp.n_match == 2
    assert cmp.n_passed == 0
    assert cmp.n_failed == 2
    assert not cmp.identical

    common = compare(s, t, keys="id", schema="common")
    assert common.n_passed == 2
    assert common.identical


def test_column_map():
    s = pl.DataFrame({"id": [1, 2], "cust_nm": ["a", "b"]})
    t = pl.DataFrame({"cust_id": [1, 2], "customer_name": ["a", "c"]})
    cmp = compare(s, t, keys="id", column_map={"id": "cust_id", "cust_nm": "customer_name"})

    assert cmp.schema_diff.matches
    assert cmp.schema_diff.renamed == [("id", "cust_id"), ("cust_nm", "customer_name")]
    assert cmp.n_changed == 1
    assert _native_rows(cmp.rows("changed")) == [
        {"id": 2, "cust_nm_source": "b", "cust_nm_target": "c"}
    ]
    extra = compare(
        s,
        pl.DataFrame({"cust_id": [3], "customer_name": ["z"]}),
        keys="id",
        column_map={"id": "cust_id", "cust_nm": "customer_name"},
    ).rows("extra")
    assert _native_rows(extra) == [{"id": 3, "customer_name": "z"}]


def test_columns_subset_limits_comparison_and_schema_scope():
    s = pl.DataFrame({"id": [1], "a": [1], "b": [1], "only_s": [1]})
    t = pl.DataFrame({"id": [1], "a": [1], "b": [2], "only_t": [1]})
    cmp = compare(s, t, keys="id", columns=["a"])

    assert cmp.compared_columns == ["a"]
    assert cmp.schema_diff.matches
    assert cmp.identical


def test_argument_errors():
    s = pl.DataFrame({"id": [1], "v": [1]})
    t = pl.DataFrame({"key": [1], "v": [1]})
    with pytest.raises(ValueError, match="isn't in the target"):
        compare(s, t, keys="id")
    with pytest.raises(ValueError, match="isn't in the source"):
        compare(s, s, keys="nope")
    with pytest.raises(ValueError, match="column_map"):
        compare(s, t, keys="id", column_map={"nope": "key"})
    with pytest.raises(ValueError, match="column_map"):
        compare(s, t, keys="id", column_map={"id": "nope"})
    with pytest.raises(ValueError, match="columns="):
        compare(s, s, keys="id", columns=["nope"])
    with pytest.raises(ValueError, match="schema="):
        compare(s, s, keys="id", schema="loose")
    with pytest.raises(ValueError, match="dup_keys="):
        compare(s, s, keys="id", dup_keys="drop")


# ----------------------------------------------------------------------------------------------
# Inputs: callables and files (lazily scanned)
# ----------------------------------------------------------------------------------------------


def test_callable_inputs():
    cmp = compare(lambda: pl.DataFrame(SOURCE), lambda: pl.DataFrame(TARGET), keys="id")
    assert cmp.status_counts == EXPECTED_COUNTS


def test_csv_and_parquet_paths_are_scanned_lazily(tmp_path):
    src_csv = tmp_path / "source.csv"
    tgt_parquet = tmp_path / "target.parquet"
    pl.DataFrame(SOURCE).write_csv(src_csv)
    pl.DataFrame(TARGET).write_parquet(tgt_parquet)

    cmp = compare(str(src_csv), str(tgt_parquet), keys="id")
    assert cmp.status_counts == EXPECTED_COUNTS
    assert cmp.source_name == str(src_csv)
    assert isinstance(cmp._src.to_native(), pl.LazyFrame)

    # Positional mode works on lazily scanned files, which keep their row order
    assert compare(str(src_csv), str(src_csv)).identical


def test_validate_ingest_stays_eager(tmp_path):
    from pointblank.validate import _process_data

    path = tmp_path / "data.csv"
    pl.DataFrame({"a": [1]}).write_csv(path)
    assert isinstance(_process_data(str(path)), pl.DataFrame)
    assert isinstance(_process_data(str(path), lazy=True), pl.LazyFrame)

    pq_dir = tmp_path / "pq"
    pq_dir.mkdir()
    pl.DataFrame({"a": [1]}).write_parquet(pq_dir / "one.parquet")
    pl.DataFrame({"a": [2]}).write_parquet(pq_dir / "two.parquet")
    lazy = _process_data(str(pq_dir / "*.parquet"), lazy=True)
    assert isinstance(lazy, pl.LazyFrame)
    assert lazy.collect().height == 2
    assert isinstance(_process_data(str(pq_dir), lazy=True), pl.LazyFrame)


# ----------------------------------------------------------------------------------------------
# Summaries, export, and report
# ----------------------------------------------------------------------------------------------


def test_column_summary():
    cmp = compare(pl.DataFrame(SOURCE), pl.DataFrame(TARGET), keys="id")
    summary = cmp.column_summary()
    assert summary.columns == [
        "column",
        "target_column",
        "dtype_source",
        "dtype_target",
        "n_mismatch",
        "f_mismatch",
    ]
    rows = {r["column"]: r for r in summary.iter_rows(named=True)}
    assert rows["amount"]["n_mismatch"] == 1
    assert rows["amount"]["f_mismatch"] == pytest.approx(1 / 3)
    assert rows["region"]["n_mismatch"] == 0


def test_to_dict_and_json():
    cmp = compare(pl.DataFrame(SOURCE), pl.DataFrame(TARGET), keys="id")

    d = cmp.to_dict(limit=1)
    assert d["status_counts"] == EXPECTED_COUNTS
    assert d["schema_diff"]["matches"] is True
    assert set(d["samples"]) == {"changed", "missing", "extra", "dup_key"}
    assert all(len(v) == 1 for v in d["samples"].values())
    assert d["samples"]["changed"][0]["amount_target"] == 20.5

    # Samples can be left out (e.g., when data may contain PII)
    assert "samples" not in cmp.to_dict(samples=False)

    parsed = json.loads(cmp.to_json())
    assert parsed["n"] == 9


def test_to_json_handles_temporal_values():
    s = pl.DataFrame({"id": [1], "ts": [datetime.datetime(2024, 1, 1)]})
    t = pl.DataFrame({"id": [1], "ts": [datetime.datetime(2024, 1, 2)]})
    parsed = json.loads(compare(s, t, keys="id").to_json())
    assert parsed["samples"]["changed"][0]["ts_source"] == "2024-01-01T00:00:00"


def test_report_sections():
    cmp = compare(
        pl.DataFrame(SOURCE),
        pl.DataFrame(TARGET),
        keys="id",
        source_name="orders_src",
        target_name="orders_tgt",
    )
    html = cmp.get_tabular_report().as_raw_html()

    for text in [
        "Table Comparison",
        "orders_src",
        "orders_tgt",
        "DIFFERENT",
        "Column mismatches (1 of 2 columns)",
        "Changed rows (1 row)",
        "Missing from target (2 rows)",
        "Extra in target (2 rows)",
        "Duplicate keys (1 key)",
        "(+0.5)",
        "null key",
    ]:
        assert text in html, text

    assert "Changed rows (1 row)" in cmp._repr_html_()


def test_report_sampling_label_and_title():
    s = pl.DataFrame({"id": list(range(30)), "v": [0] * 30})
    t = pl.DataFrame({"id": list(range(30)), "v": [1] * 30})
    html = compare(s, t, keys="id").get_tabular_report(limit=5, title="My diff").as_raw_html()
    assert "Changed rows (showing 5 of 30 rows)" in html
    assert "My diff" in html


def test_report_identical_and_schema_sections():
    df = pl.DataFrame({"id": [1, 2], "v": [1, 2]})
    html = compare(df, df, keys="id").get_tabular_report().as_raw_html()
    assert "IDENTICAL" in html
    assert "All 2 rows match." in html

    t = pl.DataFrame({"id": [1, 2], "w": [1, 2], "v": [1.0, 2.0]})
    html = compare(df, t, keys="id").get_tabular_report().as_raw_html()
    assert "Schema differences" in html
    assert "SCHEMA DIFFERS" in html
    assert "only in target" in html


def test_repr():
    cmp = compare(pl.DataFrame(SOURCE), pl.DataFrame(TARGET), keys="id")
    assert repr(cmp).startswith("Comparison('source' vs 'target': match=2, changed=1")


def test_public_exports():
    assert pb.compare is compare
    assert pb.Comparison is Comparison
    assert pb.SchemaDiff is SchemaDiff


# ----------------------------------------------------------------------------------------------
# Real file and connection-string inputs (ported from the tests of the former `Compare` stub)
# ----------------------------------------------------------------------------------------------


def test_compare_csv_file_with_itself():
    cmp = compare("data_raw/small_table.csv", "data_raw/small_table.csv")
    assert cmp.identical
    assert cmp.n == 13


def test_compare_parquet_file_with_itself():
    cmp = compare("tests/tbl_files/tbl_xyz.parquet", "tests/tbl_files/tbl_xyz.parquet")
    assert cmp.identical


def test_compare_connection_strings_across_backends():
    import os

    from pointblank.validate import get_data_path

    sqlite_path = os.path.abspath("tests/tbl_files/tbl_xyz.sqlite")
    parquet_path = "tests/tbl_files/tbl_xyz.parquet"

    # SQLite vs Parquet: different backends are brought onto a common one
    cmp = compare(f"sqlite:///{sqlite_path}::tbl_xyz", parquet_path, order_by=["x", "y", "z"])
    assert cmp.identical
    assert cmp.n_match == cmp.n_source_rows == cmp.n_target_rows

    # DuckDB connection string vs the same table as CSV
    duckdb_path = get_data_path("small_table", "duckdb")
    cmp = compare(
        f"duckdb:///{duckdb_path}::small_table",
        "data_raw/small_table.csv",
        keys=["date_time", "a", "b"],
        schema="common",
    )
    assert cmp.n_source_rows == 13
    assert cmp.n_target_rows == 13
    # All values agree; `small_table` has one fully duplicated row, flagged on both sides
    assert cmp.status_counts == {"match": 11, "changed": 0, "missing": 0, "extra": 0, "dup_key": 2}
    assert all(n == 0 for n in cmp.column_mismatches.values())


def test_compare_mixed_file_types():
    cmp = compare("data_raw/small_table.csv", "tests/tbl_files/tbl_xyz.parquet", schema="common")
    assert not cmp.schema_diff.matches
    assert cmp.compared_columns == []


# ----------------------------------------------------------------------------------------------
# Partitioned comparison (`partitions=`)
# ----------------------------------------------------------------------------------------------


_DUCKDB_CONNECTIONS: list = []  # keep connections alive for the relations' lifetime


def _duckdb_relations(s: dict, t: dict):
    import duckdb

    con = duckdb.connect()
    _DUCKDB_CONNECTIONS.append(con)
    s_df, t_df = pl.DataFrame(s), pl.DataFrame(t)  # noqa: F841 (referenced by name in SQL)
    return con.sql("select * from s_df"), con.sql("select * from t_df")


@pytest.mark.parametrize("backend", ["polars", "polars_lazy", "duckdb", "duckdb_relation"])
@pytest.mark.parametrize("partitions", [2, 5])
def test_partitioned_results_equal_single_pass(backend, partitions):
    target = {**TARGET, "cid": TARGET["id"]}
    target = {k: v for k, v in target.items() if k != "id"}
    if backend == "duckdb_relation":
        s, t = _duckdb_relations(SOURCE, target)
    else:
        s, t = _make(backend, SOURCE, "ps"), _make(backend, target, "pt")
    kwargs = dict(keys="id", column_map={"id": "cid"}, dup_keys="compare")

    single = compare(s, t, **kwargs)
    parted = compare(s, t, partitions=partitions, **kwargs)

    assert parted.partitions == partitions
    assert parted.status_counts == single.status_counts == EXPECTED_COUNTS
    assert parted.column_mismatches == single.column_mismatches
    assert (parted.n_source_rows, parted.n_target_rows) == (7, 6)
    for status in ("changed", "missing", "extra", "dup_key"):
        assert len(parted.rows(status)) == len(single.rows(status))
    # (one extract row per duplicated key value)
    assert len(nw.from_native(parted._failing_rows(100))) == len(
        nw.from_native(single._failing_rows(100))
    )
    assert parted._target_pass_table() is None
    assert "duplicate keys compared pairwise" in parted.get_tabular_report().as_raw_html()
    assert parted.to_dict()["partitions"] == partitions


def test_partitioned_composite_keys_and_sample_limit():
    s = pl.DataFrame({"k1": [i // 2 for i in range(80)], "k2": ["a", "b"] * 40, "v": [0] * 80})
    t = s.with_columns(v=pl.lit(1))
    cmp = compare(s, t, keys=["k1", "k2"], partitions=4)
    assert cmp.n_changed == 80
    assert len(cmp.rows("changed", limit=25)) == 25
    assert len(nw.from_native(cmp._failing_rows(30))) == 30


def test_partitions_argument_errors():
    s = pl.DataFrame({"id": [1], "v": [1]})
    with pytest.raises(ValueError, match="requires `keys=`"):
        compare(s, s, partitions=2)
    with pytest.raises(ValueError, match="requires `keys=`"):
        compare(s, s, keys="*", partitions=2)
    with pytest.raises(ValueError, match="positive integer"):
        compare(s, s, keys="id", partitions=0)
    with pytest.raises(ValueError, match="supported for Polars, DuckDB, and Ibis"):
        compare(s.to_pandas(), s.to_pandas(), keys="id", partitions=2)
    with pytest.raises(ValueError, match="isn't in the target"):
        compare(s, s.rename({"id": "x"}), keys="id", partitions=2)
    # A single partition is an ordinary comparison
    assert compare(s, s, keys="id", partitions=1).partitions == 1


def test_path_objects_name_the_tables(tmp_path):
    path = tmp_path / "source.csv"
    pl.DataFrame({"id": [1]}).write_csv(path)
    cmp = compare(path, path, keys="id")
    assert cmp.source_name == cmp.target_name == str(path)
