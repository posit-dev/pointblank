"""
Tests for `Validate.tbl_match()` as backed by the table-comparison engine (`pointblank.compare`):
row-level test units, comparison settings, step reports, extracts, sundering, and serialization.
"""

from __future__ import annotations

import json
import warnings

import ibis
import pandas as pd
import polars as pl
import pytest

import pointblank as pb
from pointblank._interrogation import tbl_match as tbl_match_legacy


SOURCE = pl.DataFrame({"id": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 4.0], "s": ["a", "b", "c", "d"]})
# Target: id 2 changed, id 3 missing, id 5 extra
TARGET = pl.DataFrame({"id": [1, 2, 4, 5], "x": [1.0, 2.5, 4.0, 5.0], "s": ["a", "b", "d", "e"]})


def _step(validation: pb.Validate, i: int = 1):
    return validation.validation_info[i - 1]


# ----------------------------------------------------------------------------------------------
# Verdict parity with the previous (single test unit) implementation
# ----------------------------------------------------------------------------------------------

_BASE = {"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"], "c": [4.0, 5.0, 6.0, 7.0]}

PARITY_CASES = {
    "identical": (_BASE, _BASE),
    "value_differs": (_BASE, {**_BASE, "c": [4.0, 5.5, 6.0, 7.0]}),
    "fewer_rows": (_BASE, {k: v[:2] for k, v in _BASE.items()}),
    "more_rows": ({k: v[:2] for k, v in _BASE.items()}, _BASE),
    "renamed_column": (_BASE, {"a": _BASE["a"], "b": _BASE["b"], "d": _BASE["c"]}),
    "column_order": (_BASE, {"a": _BASE["a"], "c": _BASE["c"], "b": _BASE["b"]}),
    "column_case": (_BASE, {"A": _BASE["a"], "b": _BASE["b"], "c": _BASE["c"]}),
    "extra_column": (_BASE, {**_BASE, "d": [0, 0, 0, 0]}),
    "int_vs_float": (_BASE, {**_BASE, "a": [1.0, 2.0, 3.0, 4.0]}),
    "nan_vs_null": (
        {"a": [1, 2], "c": [float("nan"), 1.0]},
        {"a": [1, 2], "c": [None, 1.0]},
    ),
    "nulls_equal": ({"a": [1, None], "b": [None, "x"]}, {"a": [1, None], "b": [None, "x"]}),
    "null_vs_value": ({"a": [1, None]}, {"a": [1, 2]}),
    "row_order": (_BASE, {k: list(reversed(v)) for k, v in _BASE.items()}),
}


@pytest.mark.parametrize("backend", ["polars", "pandas"])
@pytest.mark.parametrize("case", list(PARITY_CASES))
def test_default_verdict_matches_legacy_implementation(case, backend):
    data_dict, compare_dict = PARITY_CASES[case]
    data, comparison = pl.DataFrame(data_dict), pl.DataFrame(compare_dict)
    if backend == "pandas":
        data, comparison = data.to_pandas(), comparison.to_pandas()

    expected = tbl_match_legacy(data_tbl=data, tbl_compare=comparison)
    validation = pb.Validate(data=data).tbl_match(tbl_compare=comparison).interrogate()

    assert validation.all_passed() is expected


def test_empty_tables_have_a_single_test_unit():
    schema = {"a": pl.Int64, "b": pl.String}
    empty = pl.DataFrame({"a": [], "b": []}, schema=schema)

    v = pb.Validate(empty).tbl_match(empty.clone()).interrogate()
    assert (_step(v).n, _step(v).n_passed, _step(v).all_passed) == (1, 1, True)

    other = pl.DataFrame({"a": [], "c": []}, schema={"a": pl.Int64, "c": pl.String})
    v = pb.Validate(empty).tbl_match(other).interrogate()
    assert (_step(v).n, _step(v).n_failed, _step(v).all_passed) == (1, 1, False)


# ----------------------------------------------------------------------------------------------
# Row-level test units and comparison settings
# ----------------------------------------------------------------------------------------------


def test_keyed_test_units_and_fractional_thresholds():
    v = (
        pb.Validate(TARGET, thresholds=pb.Thresholds(warning=0.5, error=0.8))
        .tbl_match(SOURCE, keys="id")
        .interrogate()
    )
    step = _step(v)

    # ids 1..5: 1 and 4 match; 2 changed, 3 missing, 5 extra
    assert (step.n, step.n_passed, step.n_failed) == (5, 2, 3)
    assert step.f_failed == pytest.approx(0.6)
    assert step.warning is True
    assert step.error is False
    assert v.validation_info[0].val_info["comparison"].status_counts["changed"] == 1


def test_settings_are_passed_to_the_comparison():
    v = (
        pb.Validate(TARGET)
        .tbl_match(SOURCE, keys="id", tolerance={"x": 1}, columns=["x"])
        .interrogate()
    )
    # With the tolerance, id 2 (2.0 -> 2.5) matches; only missing + extra remain
    assert (_step(v).n_passed, _step(v).n_failed) == (3, 2)


def test_column_map_and_common_schema():
    renamed = TARGET.rename({"s": "label"}).with_columns(extra=pl.lit(0))
    v = (
        pb.Validate(renamed)
        .tbl_match(SOURCE, keys="id", column_map={"s": "label"})
        .tbl_match(SOURCE, keys="id", column_map={"s": "label"}, schema="common")
        .interrogate()
    )
    # The extra column fails the strict schema check (all units fail) but not the "common" one
    assert _step(v, 1).n_passed == 0
    assert _step(v, 2).n_passed == 2


def test_multiset_mode():
    data = pl.DataFrame({"a": [2, 1, 1], "b": ["y", "x", "x"]})
    comparison = pl.DataFrame({"a": [1, 2, 1], "b": ["x", "y", "x"]})
    v = pb.Validate(data).tbl_match(comparison, keys="*").interrogate()
    assert v.all_passed()
    assert _step(v).n == 3

    v = pb.Validate(data).tbl_match(comparison.head(2), keys="*").interrogate()
    assert (_step(v).n_passed, _step(v).n_failed) == (2, 1)


def test_tolerance_not_allowed_in_multiset_mode():
    v = pb.Validate(TARGET).tbl_match(SOURCE, keys="*", tolerance=0.1).interrogate()
    assert _step(v).eval_error


def test_invalid_comparison_is_an_eval_error_with_note():
    v = pb.Validate(TARGET).tbl_match(SOURCE, keys="nope").interrogate()
    step = _step(v)
    assert step.eval_error
    assert "nope" in v.get_notes(i=1, format="text")[0]
    # The step report can't be produced, but nothing raises
    assert v.get_step_report(i=1) is None


def test_invalid_setting_values_raise_early():
    with pytest.raises(ValueError, match="schema="):
        pb.Validate(TARGET).tbl_match(SOURCE, schema="loose")
    with pytest.raises(ValueError, match="dup_keys="):
        pb.Validate(TARGET).tbl_match(SOURCE, dup_keys="drop")


def test_callable_and_path_comparison_tables(tmp_path):
    path = tmp_path / "source.csv"
    SOURCE.write_csv(path)

    v = (
        pb.Validate(TARGET)
        .tbl_match(lambda: SOURCE, keys="id")
        .tbl_match(str(path), keys="id")
        .interrogate()
    )
    assert _step(v, 1).n_passed == _step(v, 2).n_passed == 2


def test_database_tables():
    con = ibis.duckdb.connect()
    src = con.create_table("src", SOURCE)
    tgt = con.create_table("tgt", TARGET)
    v = pb.Validate(tgt).tbl_match(src, keys="id").interrogate()
    assert (_step(v).n, _step(v).n_passed) == (5, 2)
    # Extracts are collected from the database comparison too
    assert len(v.get_data_extracts(i=1, frame=True)) == 3


# ----------------------------------------------------------------------------------------------
# Extracts and sundering
# ----------------------------------------------------------------------------------------------


def test_data_extract_contains_failing_rows_of_both_tables():
    v = pb.Validate(TARGET).tbl_match(SOURCE, keys="id").interrogate()
    extract = v.get_data_extracts(i=1, frame=True)

    assert extract.columns == ["id", "_status_", "x_source", "x_target", "s_source", "s_target"]
    rows = {r["id"]: r for r in extract.iter_rows(named=True)}
    assert rows[2]["_status_"] == "changed"
    assert (rows[2]["x_source"], rows[2]["x_target"]) == (2.0, 2.5)
    assert rows[3]["_status_"] == "missing"
    assert rows[5]["_status_"] == "extra"


def test_extract_limit_and_no_extracts():
    s = pl.DataFrame({"id": list(range(50)), "v": [0] * 50})
    t = pl.DataFrame({"id": list(range(50)), "v": [1] * 50})
    v = pb.Validate(t).tbl_match(s, keys="id").interrogate(extract_limit=7)
    assert len(v.get_data_extracts(i=1, frame=True)) == 7

    v = pb.Validate(t).tbl_match(s, keys="id").interrogate(collect_extracts=False)
    assert v.get_data_extracts(i=1, frame=True) is None


def test_extract_includes_null_and_duplicate_keys():
    s = pl.DataFrame({"id": [1, 2, 2, None], "v": [1, 2, 3, 4]})
    t = pl.DataFrame({"id": [1, 2, None], "v": [1, 2, 5]})
    v = pb.Validate(t).tbl_match(s, keys="id").interrogate()
    statuses = sorted(v.get_data_extracts(i=1, frame=True)["_status_"].to_list())
    assert statuses == ["dup_key", "extra", "missing"]


@pytest.mark.parametrize("to_pandas", [False, True])
def test_sundering_with_tbl_match(to_pandas):
    data = TARGET.to_pandas() if to_pandas else TARGET
    comparison = SOURCE.to_pandas() if to_pandas else SOURCE
    v = (
        pb.Validate(data)
        .tbl_match(comparison, keys="id")
        .col_vals_lt(columns="x", value=3.0)
        .interrogate()
    )
    passed = pl.DataFrame(v.get_sundered_data(type="pass"))
    failed = pl.DataFrame(v.get_sundered_data(type="fail"))

    # id 1 matches and x < 3; id 4 matches but fails x < 3; ids 2 (changed) and 5 (extra)
    # fail tbl_match; id 3 (missing) isn't a target row
    assert passed["id"].to_list() == [1]
    assert failed["id"].to_list() == [2, 4, 5]


def test_sundering_positional_with_order_by():
    data = pl.DataFrame({"k": [3, 1, 2], "v": ["c", "a", "X"]})
    comparison = pl.DataFrame({"k": [1, 2, 3], "v": ["a", "b", "c"]})
    v = pb.Validate(data).tbl_match(comparison, order_by="k").interrogate()

    assert (_step(v).n_passed, _step(v).n_failed) == (2, 1)
    # Original row order is kept; only the row aligned with k=2 differs
    assert v.get_sundered_data(type="pass")["k"].to_list() == [3, 1]
    assert v.get_sundered_data(type="fail")["k"].to_list() == [2]


def test_sundering_skips_steps_without_row_aligned_results():
    # Mixed backends: the comparison runs in Polars, so the step can't be aligned with the
    # pandas data and is left out of sundering (all rows pass the remaining steps)
    v = pb.Validate(TARGET.to_pandas()).tbl_match(SOURCE.lazy(), keys="id").interrogate()
    assert _step(v).tbl_checked is None
    assert len(v.get_sundered_data(type="pass")) == 4


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------


def test_step_report():
    v = pb.Validate(TARGET, tbl_name="orders").tbl_match(SOURCE, keys="id").interrogate()
    html = v.get_step_report(i=1).as_raw_html()

    for text in [
        "Report for Validation Step 1",
        "tbl_match(keys=&#x27;id&#x27;)",
        "orders",
        "Changed rows (1 row)",
        "Missing from target (1 row)",
        "Extra in target (1 row)",
        "(+0.5)",
    ]:
        assert text in html, text

    # Without the step header, the comparison report keeps its own title
    html = v.get_step_report(i=1, header=None).as_raw_html()
    assert "Table Comparison" in html
    assert "Report for Validation Step" not in html

    # A custom header template
    html = v.get_step_report(i=1, header="Custom: {title}").as_raw_html()
    assert "Custom:" in html


def test_step_report_dup_keys_compare():
    s = pl.DataFrame({"id": [1, 1], "v": [1, 2]})
    t = pl.DataFrame({"id": [1], "v": [1]})
    v = pb.Validate(t).tbl_match(s, keys="id", dup_keys="compare").interrogate()
    assert "pairs differ in" in v.get_step_report(i=1).as_raw_html()


def test_validation_report_values_badge():
    v = (
        pb.Validate(TARGET)
        .tbl_match(SOURCE)
        .tbl_match(SOURCE, keys="id", tolerance=0.1, schema="common")
        .tbl_match(SOURCE, keys="*")
        .interrogate()
    )
    html = v.get_tabular_report().as_raw_html()
    assert html.count("EXTERNAL TABLE") == 3
    assert "key: id &middot; tol &middot; common schema" in html
    assert "multiset" in html


def test_json_report_includes_comparison_results():
    v = pb.Validate(TARGET).tbl_match(SOURCE, keys="id").interrogate()
    values = json.loads(v.get_json_report())[0]["values"]

    assert values["tbl_compare"] == "<polars table>"
    assert values["keys"] == "id"
    assert values["comparison"]["status_counts"] == {
        "match": 2,
        "changed": 1,
        "missing": 1,
        "extra": 1,
        "dup_key": 0,
    }
    assert set(values["comparison"]["samples"]) == {"changed", "missing", "extra"}

    # Row values stay out of the JSON when extracts aren't collected (e.g., for PII)
    v = pb.Validate(TARGET).tbl_match(SOURCE, keys="id").interrogate(collect_extracts=False)
    values = json.loads(v.get_json_report())[0]["values"]
    assert "samples" not in values["comparison"]
    assert values["comparison"]["n_failed"] == 3


# ----------------------------------------------------------------------------------------------
# Serialization: to_code(), to_yaml(), YAML interrogation, and Steps
# ----------------------------------------------------------------------------------------------


def test_to_code_and_to_yaml(tmp_path):
    path = str(tmp_path / "source.csv")
    SOURCE.write_csv(path)
    v = pb.Validate(TARGET).tbl_match(path, keys="id", tolerance={"x": {"atol": 0.1}})

    code = v.to_code()
    assert f'.tbl_match(tbl_compare="{path}", keys="id", tolerance=' in code

    yaml_str = v.to_yaml()
    assert "tbl_compare: " + path in yaml_str
    assert "atol: 0.1" in yaml_str

    # An in-memory comparison table becomes a placeholder (with a warning)
    with pytest.warns(UserWarning, match="placeholder"):
        code = pb.Validate(TARGET).tbl_match(SOURCE).to_code()
    assert "tbl_compare=your_comparison_table" in code


def test_yaml_round_trip(tmp_path):
    src_path, tgt_path = str(tmp_path / "source.csv"), str(tmp_path / "target.csv")
    SOURCE.write_csv(src_path)
    TARGET.write_csv(tgt_path)

    v = pb.Validate(TARGET).tbl_match(src_path, keys="id", normalize={"s": ["upper"]})
    yaml_str = v.to_yaml().replace("tbl: your_data", f"tbl: {tgt_path}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from_yaml = pb.yaml_interrogate(yaml_str)
    direct = v.interrogate()

    assert _step(from_yaml).values == _step(direct).values
    assert (_step(from_yaml).n, _step(from_yaml).n_passed) == (_step(direct).n, 2)


def test_steps_api():
    steps = pb.Steps().tbl_match(SOURCE, keys="id", schema="common")
    v = pb.Validate(TARGET).add_steps(steps).interrogate()
    assert _step(v).values["keys"] == "id"
    assert _step(v).values["schema"] == "common"
    assert (_step(v).n, _step(v).n_passed) == (5, 2)
    assert "schema" not in repr(pb.Steps().tbl_match(SOURCE))


def test_pandas_target_with_polars_comparison():
    v = pb.Validate(TARGET.to_pandas()).tbl_match(SOURCE, keys="id").interrogate()
    assert (_step(v).n, _step(v).n_passed) == (5, 2)
    extract = v.get_data_extracts(i=1, frame=True)
    assert isinstance(extract, (pl.DataFrame, pd.DataFrame))
    assert len(extract) == 3
