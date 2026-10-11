from __future__ import annotations

import json

import click
import polars as pl
import pytest
from click.testing import CliRunner

from pointblank.cli import (
    _parse_compare_column_map,
    _parse_compare_normalize,
    _parse_compare_tolerance,
    cli,
)


SOURCE = pl.DataFrame(
    {"id": [1, 2, 3, 4], "amount": [10.0, 20.0, 30.0, 40.0], "region": ["N", "S", "E", "W"]}
)
TARGET = pl.DataFrame(
    {"id": [1, 2, 4, 5], "amount": [10.0, 20.5, 40.0, 50.0], "region": ["N", "S", "W", "N"]}
)


@pytest.fixture
def files(tmp_path):
    src, tgt = tmp_path / "source.csv", tmp_path / "target.csv"
    SOURCE.write_csv(src)
    TARGET.write_csv(tgt)
    return str(src), str(tgt)


def _run(*args):
    return CliRunner().invoke(cli, ["compare", *args])


def test_identical_tables_exit_zero(files):
    src, _ = files
    result = _run(src, src, "--key", "id")
    assert result.exit_code == 0, result.output
    assert "IDENTICAL" in result.output


def test_differences_exit_one_with_summary(files):
    src, tgt = files
    result = _run(src, tgt, "-k", "id")
    assert result.exit_code == 1
    for text in [
        "Table Comparison",
        "DIFFERENT",
        "Row status",
        "Column mismatches",
        "Changed rows (1)",
        "Missing from target (1)",
        "Extra in target (1)",
        "20.5",
    ]:
        assert text in result.output, text


def test_no_exit_code(files):
    src, tgt = files
    assert _run(src, tgt, "-k", "id", "--no-exit-code").exit_code == 0


def test_tolerance_option(files):
    src, tgt = files
    result = _run(src, tgt, "-k", "id", "-t", "amount=1", "--format", "json")
    assert json.loads(result.output)["status_counts"]["changed"] == 0


def test_json_output_and_files(files, tmp_path):
    src, tgt = files
    html_path, json_path = tmp_path / "diff.html", tmp_path / "diff.json"
    result = _run(
        src,
        tgt,
        "-k",
        "id",
        "--format",
        "json",
        "--output-html",
        str(html_path),
        "--output-json",
        str(json_path),
    )
    assert result.exit_code == 1

    data = json.loads(result.output)  # stdout is pure JSON
    assert data["status_counts"] == {
        "match": 2,
        "changed": 1,
        "missing": 1,
        "extra": 1,
        "dup_key": 0,
    }
    assert data["samples"]["changed"][0]["amount_target"] == 20.5
    assert json.loads(json_path.read_text()) == data
    assert "Table Comparison" in html_path.read_text()


def test_no_samples(files):
    src, tgt = files
    data = json.loads(_run(src, tgt, "-k", "id", "--format", "json", "--no-samples").output)
    assert "samples" not in data


def test_positional_multiset_and_partitions(files):
    src, tgt = files
    positional = json.loads(_run(src, tgt, "--format", "json").output)
    assert positional["mode"] == "positional"

    multiset = json.loads(_run(src, src, "-k", "*", "--format", "json").output)
    assert multiset["mode"] == "multiset"
    assert multiset["identical"] is True

    single = json.loads(_run(src, tgt, "-k", "id", "--format", "json").output)
    partitioned = json.loads(
        _run(src, tgt, "-k", "id", "--partitions", "3", "--format", "json").output
    )
    assert partitioned["partitions"] == 3
    assert partitioned["status_counts"] == single["status_counts"]


def test_schema_and_column_map_options(tmp_path, files):
    src, _ = files
    renamed = tmp_path / "renamed.csv"
    SOURCE.rename({"region": "area"}).write_csv(renamed)

    strict = _run(src, str(renamed), "-k", "id", "--format", "json")
    assert json.loads(strict.output)["identical"] is False

    mapped = _run(src, str(renamed), "-k", "id", "--column-map", "region=area")
    assert mapped.exit_code == 0

    common = json.loads(
        _run(src, str(renamed), "-k", "id", "--schema", "common", "--format", "json").output
    )
    assert common["identical"] is True
    assert common["compared_columns"] == ["amount"]


def test_builtin_dataset(files):
    result = _run("small_table", "small_table", "--format", "json")
    assert result.exit_code == 0
    assert json.loads(result.output)["n"] == 13


def test_errors_exit_two(files):
    src, tgt = files
    result = _run(src, tgt, "-k", "nope")
    assert result.exit_code == 2
    assert "Error" in result.output

    result = _run(src, "does_not_exist.csv", "-k", "id")
    assert result.exit_code == 2

    # Bad option values are usage errors (also exit code 2)
    assert _run(src, tgt, "-k", "id", "-t", "amount=bogus:1").exit_code == 2
    assert _run(src, tgt, "-k", "id", "--column-map", "region").exit_code == 2
    assert _run(src, tgt, "-k", "id", "--partitions", "0").exit_code == 2


def test_compare_listed_in_help():
    result = CliRunner().invoke(cli, ["--help"])
    assert "compare" in result.output


# ----------------------------------------------------------------------------------------------
# Option parsing
# ----------------------------------------------------------------------------------------------


def test_parse_tolerance():
    assert _parse_compare_tolerance(()) is None
    assert _parse_compare_tolerance(("0.5",)) == 0.5
    assert _parse_compare_tolerance(("rtol:1e-3",)) == {"rtol": 0.001}
    assert _parse_compare_tolerance(("amount=atol:0.1,rtol:0.01", "ts=1s")) == {
        "amount": {"atol": 0.1, "rtol": 0.01},
        "ts": "1s",
    }
    with pytest.raises(click.BadParameter, match="either"):
        _parse_compare_tolerance(("0.5", "amount=1"))
    with pytest.raises(click.BadParameter, match="Unknown tolerance"):
        _parse_compare_tolerance(("amount=abs:1",))


def test_parse_normalize_and_column_map():
    assert _parse_compare_normalize(()) is None
    assert _parse_compare_normalize(("name=strip,lower", "x=round:2")) == {
        "name": ["strip", "lower"],
        "x": [{"round": 2}],
    }
    with pytest.raises(click.BadParameter):
        _parse_compare_normalize(("name",))

    assert _parse_compare_column_map(()) is None
    assert _parse_compare_column_map(("a=b", "c = d")) == {"a": "b", "c": "d"}
    with pytest.raises(click.BadParameter):
        _parse_compare_column_map(("a=",))


def test_duckdb_engine(files, tmp_path):
    src, tgt = files
    parquet = tmp_path / "target.parquet"
    TARGET.write_parquet(parquet)
    polars_result = json.loads(_run(src, tgt, "-k", "id", "--format", "json").output)
    for target in (tgt, str(parquet)):
        result = _run(
            src, target, "-k", "id", "--engine", "duckdb", "--partitions", "2", "--format", "json"
        )
        data = json.loads(result.output)
        assert data["status_counts"] == polars_result["status_counts"]
    assert _run(src, "nope.csv", "-k", "id", "--engine", "duckdb").exit_code == 2
