"""
Parity checks: `pointblank.compare()` against results recorded from datacompy, an established
table-comparison library (used as a behavioral reference only).

The fixtures and expected results in `tests/tbl_files/compare_parity/` are produced by
`scripts/generate_compare_parity_fixtures.py`. The mapping between the two tools:

| datacompy                          | Pointblank                              |
|------------------------------------|-----------------------------------------|
| `df1_unq_rows` (rows only in df1)  | `n_missing` (source rows not in target) |
| `df2_unq_rows` (rows only in df2)  | `n_extra` (target rows not in source)   |
| `intersect_rows`                   | `n_match + n_changed`                   |
| `count_matching_rows()`            | `n_match`                               |
| `column_stats[...]["unequal_cnt"]` | `column_mismatches[...]`                |
| `df1_unq_columns()`                | `schema_diff.only_in_source`            |
| `df2_unq_columns()`                | `schema_diff.only_in_target`            |
| `join_columns`                     | `keys=`                                 |
| `abs_tol` / `rel_tol`              | `tolerance={"atol": ...}` / `"rtol"`    |
| `ignore_spaces` / `ignore_case`    | `normalize={col: ["strip", "lower"]}`   |

Known, intentional differences (covered by the fixtures' design):

- **Duplicate keys**: datacompy pairs up rows with a duplicated key by their order of appearance;
  Pointblank flags them as `dup_key` and doesn't align them. Parity is checked on the other keys.
- **Relative tolerance**: datacompy scales `rel_tol` by the target value (`np.isclose(df1, df2)`);
  Pointblank scales it by the source (expected) value. This only matters for differences right
  at the boundary, which the fixtures avoid.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import ibis
import pandas as pd
import polars as pl
import pytest

import pointblank as pb

FIXTURES = Path(__file__).parent / "tbl_files" / "compare_parity"
EXPECTED = json.loads((FIXTURES / "expected.json").read_text())
CASES = sorted(EXPECTED["cases"])
STRING_COLUMNS = ["region", "name", "day"]


def _load(case: str, backend: str):
    src_path, tgt_path = FIXTURES / f"{case}_source.csv", FIXTURES / f"{case}_target.csv"
    if backend == "paths":
        return str(src_path), str(tgt_path)
    if backend == "pandas":
        return pd.read_csv(src_path), pd.read_csv(tgt_path)
    src, tgt = pl.read_csv(src_path), pl.read_csv(tgt_path)
    if backend == "polars":
        return src, tgt
    con = ibis.duckdb.connect()
    return con.create_table("src", src), con.create_table("tgt", tgt)


def _compare_kwargs(case: str) -> dict:
    """Translate the datacompy settings of a case into `pb.compare()` arguments."""
    settings = EXPECTED["cases"][case]["settings"]
    kwargs: dict = {"keys": settings["join_columns"], "schema": "common"}
    tol = {}
    if settings.get("abs_tol"):
        tol["atol"] = settings["abs_tol"]
    if settings.get("rel_tol"):
        tol["rtol"] = settings["rel_tol"]
    if tol:
        kwargs["tolerance"] = tol
    ops = []
    if settings.get("ignore_spaces"):
        ops.append("strip")
    if settings.get("ignore_case"):
        ops.append("lower")
    if ops:
        kwargs["normalize"] = {col: ops for col in STRING_COLUMNS}
    return kwargs


def _assert_parity(cmp, expected: dict) -> None:
    assert cmp.n_missing == expected["n_df1_unq_rows"]
    assert cmp.n_extra == expected["n_df2_unq_rows"]
    assert cmp.n_match + cmp.n_changed == expected["n_intersect_rows"]
    assert cmp.n_match == expected["n_matching_rows"]
    assert cmp.column_mismatches == expected["unequal_cnt"]
    assert sorted(cmp.schema_diff.only_in_source) == expected["df1_unq_columns"]
    assert sorted(cmp.schema_diff.only_in_target) == expected["df2_unq_columns"]


@pytest.mark.parametrize("backend", ["pandas", "polars", "duckdb", "paths"])
@pytest.mark.parametrize("case", CASES)
def test_parity_with_datacompy(case, backend):
    entry = EXPECTED["cases"][case]
    cmp = pb.compare(*_load(case, backend), **_compare_kwargs(case))

    if case == "duplicates":
        # Duplicated keys are flagged rather than paired up; everything else matches datacompy
        assert cmp.n_dup_key_values == len(entry["duplicated_keys"])
        assert sorted(cmp.rows("dup_key")["id"].to_list()) == entry["duplicated_keys"]
        _assert_parity(cmp, entry["datacompy_without_duplicates"])
    else:
        assert cmp.n_dup_key == 0
        _assert_parity(cmp, entry["datacompy"])


@pytest.mark.parametrize("case", ["basic", "composite_keys"])
def test_parity_with_partitions(case):
    cmp = pb.compare(*_load(case, "paths"), partitions=3, **_compare_kwargs(case))
    _assert_parity(cmp, EXPECTED["cases"][case]["datacompy"])


def test_recorded_results_match_installed_datacompy():
    """When datacompy is installed, re-run it on the fixtures to check the recorded results."""
    pytest.importorskip("datacompy")

    script = Path(__file__).parent.parent / "scripts" / "generate_compare_parity_fixtures.py"
    spec = importlib.util.spec_from_file_location("parity_fixtures", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for case in CASES:
        entry = EXPECTED["cases"][case]
        src, tgt = _load(case, "pandas")
        assert module._datacompy_results(src, tgt, entry["settings"]) == entry["datacompy"], case
