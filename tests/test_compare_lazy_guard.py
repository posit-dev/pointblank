"""
Guard test: comparing lazy tables (Polars LazyFrames, CSV/Parquet paths, Ibis/DuckDB tables, and
DuckDB relations) never loads a whole table into Python.

Every point where a lazy query is materialized is intercepted (narwhals' `LazyFrame.collect`,
Polars' `LazyFrame.collect`, Ibis' `to_pyarrow`/`to_polars`/`to_pandas`/`execute`, and DuckDB
relations' `pl`/`arrow`/`df`/`fetchall`), and the number of rows each one returns is recorded.
The tables have tens of thousands of rows with thousands of rows in every status, while everything
the engine legitimately collects is either a one-row aggregate or a sample bounded by a `limit`.
So any collection larger than `MAX_ROWS` means a table (or a whole status) was loaded in full.
"""

from __future__ import annotations

import json

import duckdb
import ibis
import narwhals.dataframe as nw_dataframe
import polars as pl
import pytest
from click.testing import CliRunner

import pointblank as pb

N = 30_000
MAX_ROWS = 100  # the largest bounded sample requested in these tests


def _tables() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Source/target with thousands of rows in every status (incl. duplicate and null keys)."""
    ids = pl.int_range(0, N, eager=True)
    source = pl.DataFrame(
        {
            "id": ids,
            "amount": (ids * 7 % 1000).cast(pl.Float64),
            "region": (ids % 4).cast(pl.String),
        }
    )
    source = source.with_columns(
        id=pl.when(pl.col("id") % 97 == 5).then(None).otherwise(pl.col("id"))  # null keys
    )
    target = (
        source.filter(pl.col("id").is_null() | (pl.col("id") % 10 != 1))  # ~10% missing
        .with_columns(
            amount=pl.when(pl.col("id") % 10 == 2)  # ~10% changed
            .then(pl.col("amount") + 1)
            .otherwise("amount")
        )
        .vstack(source.head(3000).with_columns(id=pl.col("id") + N))  # extra rows
        .vstack(source.filter(pl.col("id") % 50 == 3))  # duplicated keys
    )
    return source, target


@pytest.fixture
def collected(monkeypatch) -> list[int]:
    """
    Record the row count of every lazy-to-eager materialization. (Eager Polars operations also run
    through `LazyFrame.collect()` internally, so tests clear the record after building inputs.)
    """
    sizes: list[int] = []

    def wrap(cls, name, count=len):
        original = getattr(cls, name)

        def wrapper(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            sizes.append(count(out))
            return out

        monkeypatch.setattr(cls, name, wrapper)

    wrap(nw_dataframe.LazyFrame, "collect")
    wrap(pl.LazyFrame, "collect", count=lambda out: out.height)
    for name in ("to_pyarrow", "to_polars", "to_pandas", "execute"):
        wrap(ibis.expr.types.Table, name)
    for name in ("pl", "arrow", "df", "fetchall"):
        wrap(duckdb.DuckDBPyRelation, name)
    return sizes


def _lazy_inputs(kind: str, tmp_path, con_holder: list):
    source, target = _tables()
    if kind == "polars_lazy":
        return source.lazy(), target.lazy()
    if kind == "csv":
        source.write_csv(tmp_path / "s.csv")
        target.write_csv(tmp_path / "t.csv")
        return str(tmp_path / "s.csv"), str(tmp_path / "t.csv")
    if kind == "parquet":
        source.write_parquet(tmp_path / "s.parquet")
        target.write_parquet(tmp_path / "t.parquet")
        return str(tmp_path / "s.parquet"), str(tmp_path / "t.parquet")
    if kind == "ibis_duckdb":
        con = ibis.duckdb.connect()
        con_holder.append(con)
        return con.create_table("s", source), con.create_table("t", target)
    if kind == "duckdb_relation":
        con = duckdb.connect()
        con_holder.append(con)
        source.write_parquet(tmp_path / "s.parquet")
        target.write_parquet(tmp_path / "t.parquet")
        return con.read_parquet(str(tmp_path / "s.parquet")), con.read_parquet(
            str(tmp_path / "t.parquet")
        )
    raise ValueError(kind)


KINDS = ["polars_lazy", "csv", "parquet", "ibis_duckdb", "duckdb_relation"]


def _exercise_comparison(cmp) -> None:
    """Use every part of a `Comparison` that reads data."""
    cmp.get_tabular_report().as_raw_html()
    for status in ("changed", "missing", "extra", "dup_key", "match"):
        cmp.rows(status)  # default limit=100
    cmp.column_summary()
    cmp.to_json()
    cmp._failing_rows(limit=MAX_ROWS)
    cmp._dup_pairs(limit=10)


@pytest.mark.parametrize("partitions", [None, 3])
@pytest.mark.parametrize("kind", KINDS)
def test_compare_never_collects_full_tables(kind, partitions, tmp_path, collected):
    con_holder: list = []
    source, target = _lazy_inputs(kind, tmp_path, con_holder)
    collected.clear()  # only measure the comparison (building the inputs collects eagerly)

    cmp = pb.compare(source, target, keys="id", dup_keys="compare", partitions=partitions)
    _exercise_comparison(cmp)

    # Every status has thousands of rows, so this is a real comparison of a large table...
    assert min(cmp.n_changed, cmp.n_missing, cmp.n_extra, cmp.n_dup_key) > 500
    # ...and yet nothing larger than a bounded sample was ever materialized
    assert collected, "no materializations were recorded (is the guard working?)"
    assert max(collected) <= MAX_ROWS, sorted(collected)[-5:]


@pytest.mark.parametrize("kind", ["polars_lazy", "ibis_duckdb"])
def test_tbl_match_never_collects_full_tables(kind, tmp_path, collected):
    con_holder: list = []
    source, target = _lazy_inputs(kind, tmp_path, con_holder)
    collected.clear()

    validation = (
        pb.Validate(data=target)
        .tbl_match(tbl_compare=source, keys="id", dup_keys="compare")
        .interrogate(extract_limit=MAX_ROWS)
    )
    validation.get_step_report(i=1, limit=10).as_raw_html()
    validation.get_data_extracts(i=1, frame=True)
    json.loads(validation.get_json_report())

    assert validation.validation_info[0].n > N
    assert max(collected) <= MAX_ROWS, sorted(collected)[-5:]


def test_cli_never_collects_full_tables(tmp_path, collected):
    from pointblank.cli import cli

    _lazy_inputs("csv", tmp_path, [])
    collected.clear()
    for engine in ("polars", "duckdb"):
        result = CliRunner().invoke(
            cli,
            ["compare", str(tmp_path / "s.csv"), str(tmp_path / "t.csv"), "-k", "id"]
            + ["--engine", engine, "--output-html", str(tmp_path / "r.html"), "--format", "json"],
        )
        assert result.exit_code == 1, result.output
    assert max(collected) <= MAX_ROWS, sorted(collected)[-5:]


def test_guard_detects_a_full_collection(tmp_path, collected):
    """The guard itself: loading a lazy table in full is recorded as a large collection."""
    source, _ = _lazy_inputs("polars_lazy", tmp_path, [])
    collected.clear()
    source.collect()
    assert max(collected) >= N
