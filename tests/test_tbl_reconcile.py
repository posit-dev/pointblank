from __future__ import annotations

import json
import re
import warnings

import ibis
import pandas as pd
import polars as pl
import pytest

import pointblank as pb

# The source system's orders and the loaded orders: `north` matches (but in a different number of
# rows), `south` matches, `east` has a different total, `west` is missing, `central` is extra, and
# both tables have a null-region group with the same total
SOURCE = pl.DataFrame(
    {
        "region": ["north", "north", "south", "east", "west", None],
        "amount": [100.0, 250.0, 75.5, 30.0, 12.0, 5.0],
        "qty": [1, 2, 3, 4, 5, 6],
    }
)
TARGET = pl.DataFrame(
    {
        "area": ["north", "north", "north", "south", "east", "central", None],
        "amt": [100.0, 200.0, 50.0, 75.5, 31.0, 8.0, 5.0],
        "qty": [1, 1, 1, 3, 4, 2, 6],
    }
)
COLUMN_MAP = {"region": "area", "amount": "amt"}


def _step(validation, i: int = 1):
    return validation.validation_info[i - 1]


def _comparison(validation, i: int = 1):
    return _step(validation, i).val_info["comparison"]


def _make(backend: str):
    if backend == "polars":
        return SOURCE, TARGET
    if backend == "pandas":
        return SOURCE.to_pandas(), TARGET.to_pandas()
    if backend == "polars_lazy":
        return SOURCE.lazy(), TARGET.lazy()
    con = ibis.duckdb.connect()
    return con.create_table("s", SOURCE), con.create_table("t", TARGET)


# ----------------------------------------------------------------------------------------------
# Reconciliation results
# ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["polars", "pandas", "polars_lazy", "duckdb"])
def test_keyed_sum(backend):
    source, target = _make(backend)
    v = (
        pb.Validate(data=target)
        .tbl_reconcile(tbl_compare=source, keys="region", measures="amount", column_map=COLUMN_MAP)
        .interrogate()
    )
    cmp = _comparison(v)
    # north, south, and the null-region group reconcile; east changed; west missing; central extra
    assert cmp.status_counts == {"match": 3, "changed": 1, "missing": 1, "extra": 1, "dup_key": 0}
    assert (_step(v).n, _step(v).n_passed, _step(v).n_failed) == (6, 3, 3)

    extract = pl.DataFrame(v.get_data_extracts(i=1, frame=True))
    rows = {r["region"]: r for r in extract.iter_rows(named=True)}
    assert rows["east"]["_status_"] == "changed"
    assert (rows["east"]["amount_sum_source"], rows["east"]["amount_sum_target"]) == (30.0, 31.0)
    assert rows["west"]["_status_"] == "missing"
    assert rows["central"]["_status_"] == "extra"


def test_null_key_groups_are_matched():
    v = (
        pb.Validate(data=TARGET)
        .tbl_reconcile(tbl_compare=SOURCE, keys="region", measures="amount", column_map=COLUMN_MAP)
        .interrogate()
    )
    matched = _comparison(v).rows("match")["region"].to_list()
    assert None in matched


def test_grand_totals_and_tolerance():
    # Totals: 472.5 (source) vs 469.5 (target)
    def step(**kwargs):
        v = (
            pb.Validate(data=TARGET)
            .tbl_reconcile(tbl_compare=SOURCE, measures="amount", column_map=COLUMN_MAP, **kwargs)
            .interrogate()
        )
        return _step(v)

    assert (step().n, step().all_passed) == (1, False)
    assert step(tolerance=5).all_passed
    assert step(tolerance={"amount": 5}).all_passed
    assert not step(tolerance={"amount": 1}).all_passed
    assert step(tolerance={"rtol": 0.01}).all_passed


def test_row_counts_without_measures():
    v = (
        pb.Validate(data=TARGET)
        .tbl_reconcile(tbl_compare=SOURCE, keys="region", agg="count", column_map=COLUMN_MAP)
        .interrogate()
    )
    cmp = _comparison(v)
    assert cmp.compared_columns == ["n_rows"]
    # north has 2 vs 3 rows; south, east, and null match; west missing; central extra
    assert cmp.status_counts == {"match": 3, "changed": 1, "missing": 1, "extra": 1, "dup_key": 0}


def test_multiple_measures_and_aggregations():
    v = (
        pb.Validate(data=TARGET)
        .tbl_reconcile(
            tbl_compare=SOURCE,
            keys="region",
            measures=["amount", "qty"],
            agg={"amount": ["sum", "max"], "qty": "mean"},
            column_map=COLUMN_MAP,
        )
        .interrogate()
    )
    cmp = _comparison(v)
    assert cmp.compared_columns == ["amount_sum", "amount_max", "qty_mean"]
    # north: same sum but different max (250 vs 200) and mean qty (1.5 vs 1)
    assert cmp.column_mismatches == {"amount_sum": 1, "amount_max": 2, "qty_mean": 1}


@pytest.mark.parametrize("agg, expected_changed", [("min", 2), ("max", 2), ("count", 1)])
def test_other_aggregations(agg, expected_changed):
    v = (
        pb.Validate(data=TARGET)
        .tbl_reconcile(
            tbl_compare=SOURCE, keys="region", measures="amount", agg=agg, column_map=COLUMN_MAP
        )
        .interrogate()
    )
    assert _comparison(v).n_changed == expected_changed


def test_partitions_give_identical_results():
    con = ibis.duckdb.connect()
    s, t = con.create_table("s", SOURCE), con.create_table("t", TARGET)
    v = (
        pb.Validate(data=t)
        .tbl_reconcile(tbl_compare=s, keys="region", measures="amount", column_map=COLUMN_MAP)
        .tbl_reconcile(
            tbl_compare=s, keys="region", measures="amount", column_map=COLUMN_MAP, partitions=3
        )
        .interrogate()
    )
    assert _comparison(v, 1).status_counts == _comparison(v, 2).status_counts
    assert _comparison(v, 2).partitions == 3


def test_file_paths(tmp_path):
    SOURCE.write_csv(tmp_path / "s.csv")
    TARGET.write_parquet(tmp_path / "t.parquet")
    v = (
        pb.Validate(data=str(tmp_path / "t.parquet"))
        .tbl_reconcile(
            tbl_compare=str(tmp_path / "s.csv"),
            keys="region",
            measures="amount",
            column_map=COLUMN_MAP,
        )
        .interrogate()
    )
    assert _step(v).n_passed == 3


def test_empty_tables():
    empty_s = SOURCE.head(0)
    empty_t = TARGET.head(0)
    v = (
        pb.Validate(data=empty_t)
        .tbl_reconcile(tbl_compare=empty_s, keys="region", measures="amount", column_map=COLUMN_MAP)
        .interrogate()
    )
    assert (_step(v).n, _step(v).all_passed) == (1, True)


# ----------------------------------------------------------------------------------------------
# Errors
# ----------------------------------------------------------------------------------------------


def test_argument_errors():
    v = pb.Validate(data=TARGET)
    with pytest.raises(ValueError, match="measures="):
        v.tbl_reconcile(tbl_compare=SOURCE, keys="region")
    with pytest.raises(ValueError, match="agg="):
        v.tbl_reconcile(tbl_compare=SOURCE, measures="amount", agg="median")
    with pytest.raises(ValueError, match="agg="):
        v.tbl_reconcile(tbl_compare=SOURCE, measures="amount", agg={"amount": ["sum", "bogus"]})
    with pytest.raises(ValueError, match="requires `keys=`"):
        v.tbl_reconcile(tbl_compare=SOURCE, measures="amount", partitions=2)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"keys": "region", "measures": "amount"}, "isn't in the target"),  # no column_map
        ({"keys": "nope", "measures": "amt", "column_map": {}}, "isn't in the comparison"),
        ({"measures": "region", "column_map": COLUMN_MAP}, "isn't numeric"),
        ({"measures": "amount", "tolerance": {"qty": 1}, "column_map": COLUMN_MAP}, "measures="),
    ],
)
def test_setup_errors_are_eval_errors(kwargs, message):
    v = pb.Validate(data=TARGET).tbl_reconcile(tbl_compare=SOURCE, **kwargs).interrogate()
    assert _step(v).eval_error
    assert message in v.get_notes(i=1, format="text")[0]


# ----------------------------------------------------------------------------------------------
# Reporting and serialization
# ----------------------------------------------------------------------------------------------


def test_reports():
    v = (
        pb.Validate(data=TARGET, tbl_name="loaded_orders")
        .tbl_reconcile(
            tbl_compare=SOURCE, keys="region", measures="amount", column_map=COLUMN_MAP, tolerance=1
        )
        .interrogate()
    )
    report = v.get_tabular_report().as_raw_html()
    assert "EXTERNAL TABLE" in report
    assert "sum &middot; by region &middot; tol" in report

    step_report = v.get_step_report(i=1).as_raw_html()
    assert "tbl_reconcile(" in step_report
    # Test units are groups: counts, sections, and the aggregation chip say so
    assert "loaded_orders</strong> (5 groups)" in step_report
    assert "2 of 6 groups fail" in step_report  # east's 30 vs 31 is within the tolerance
    assert "Missing from target (1 group)" in step_report
    assert "<code>sum</code> by <code>region</code>" in step_report
    assert "schema: common columns" not in step_report

    no_tolerance = (
        pb.Validate(data=TARGET)
        .tbl_reconcile(tbl_compare=SOURCE, keys="region", measures="amount", column_map=COLUMN_MAP)
        .interrogate()
    )
    assert "Changed groups (1 group)" in no_tolerance.get_step_report(i=1).as_raw_html()

    totals = (
        pb.Validate(data=TARGET)
        .tbl_reconcile(tbl_compare=SOURCE, measures="amount", column_map=COLUMN_MAP)
        .interrogate()
    )
    assert "<code>sum</code> of whole table" in totals.get_step_report(i=1).as_raw_html()

    french = pb.Validate(data=TARGET, lang="fr").tbl_reconcile(
        tbl_compare=SOURCE, keys="region", measures="amount", column_map=COLUMN_MAP
    )
    french_report = french.interrogate().get_step_report(i=1).as_raw_html()
    assert "Absentes de la cible (1 groupe)" in french_report
    assert "Groupes modifiés" in french_report


def test_json_report():
    v = (
        pb.Validate(data=TARGET)
        .tbl_reconcile(tbl_compare=SOURCE, keys="region", measures="amount", column_map=COLUMN_MAP)
        .interrogate()
    )
    values = json.loads(v.get_json_report())[0]["values"]
    assert values["tbl_compare"] == "<polars table>"
    assert values["measures"] == "amount"
    assert values["comparison"]["status_counts"]["changed"] == 1


def test_to_code_yaml_and_steps(tmp_path):
    src_path, tgt_path = tmp_path / "s.csv", tmp_path / "t.csv"
    SOURCE.write_csv(src_path)
    TARGET.write_csv(tgt_path)

    v = pb.Validate(data=TARGET).tbl_reconcile(
        tbl_compare=str(src_path),
        keys="region",
        measures=["amount", "qty"],
        agg={"amount": "sum", "qty": "max"},
        column_map=COLUMN_MAP,
        tolerance={"amount": 0.5},
    )
    code = v.to_code()
    assert f'.tbl_reconcile(tbl_compare="{src_path}", keys="region"' in code
    assert "agg=" in code

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_yaml = pb.yaml_interrogate(v.to_yaml().replace("tbl: your_data", f"tbl: {tgt_path}"))
    direct = v.interrogate()
    assert _step(from_yaml).values == _step(direct).values
    assert _step(from_yaml).n_passed == _step(direct).n_passed

    steps = pb.Steps().tbl_reconcile(
        SOURCE, keys="region", measures="amount", column_map=COLUMN_MAP
    )
    via_steps = pb.Validate(data=TARGET).add_steps(steps).interrogate()
    assert _step(via_steps).n_passed == 3
    assert "agg" not in repr(steps)  # the default isn't shown


def test_threshold_and_dimension():
    v = (
        pb.Validate(data=TARGET, thresholds=pb.Thresholds(warning=0.25, error=0.75))
        .tbl_reconcile(tbl_compare=SOURCE, keys="region", measures="amount", column_map=COLUMN_MAP)
        .interrogate()
    )
    assert _step(v).warning is True
    assert _step(v).error is False
    assert _step(v).dimension in (None, "consistency")
    report = v.get_tabular_report().as_raw_html()
    assert re.search(r"tbl_reconcile", report)


def test_pandas_with_polars_comparison_table():
    v = (
        pb.Validate(data=TARGET.to_pandas())
        .tbl_reconcile(tbl_compare=SOURCE, keys="region", measures="amount", column_map=COLUMN_MAP)
        .interrogate()
    )
    assert _step(v).n_passed == 3
    assert isinstance(v.get_data_extracts(i=1, frame=True), (pl.DataFrame, pd.DataFrame))
