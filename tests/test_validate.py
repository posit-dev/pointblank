import pathlib

import pprint
import sys
import re
from unittest.mock import patch
import pytest
import random
import itertools
from functools import partial
import contextlib

import pandas as pd
import polars as pl
import ibis
from datetime import datetime

import great_tables as GT
import narwhals as nw

from pointblank.validate import (
    Actions,
    get_column_count,
    get_row_count,
    load_dataset,
    missing_vals_tbl,
    _normalize_reporting_language,
    PointblankConfig,
    preview,
    Validate,
    _create_table_time_html,
    _create_table_type_html,
    _fmt_lg,
    _get_default_title_text,
    _process_title_text,
    _ValidationInfo,
)
from pointblank.thresholds import Thresholds
from pointblank.schema import Schema, _get_schema_validation_info
from pointblank.column import (
    col,
    starts_with,
    ends_with,
    contains,
    matches,
    everything,
    first_n,
    last_n,
)


TBL_LIST = [
    "tbl_pd",
    "tbl_pl",
    "tbl_parquet",
    "tbl_duckdb",
    "tbl_sqlite",
]

TBL_MISSING_LIST = [
    "tbl_missing_pd",
    "tbl_missing_pl",
    "tbl_missing_parquet",
    "tbl_missing_duckdb",
    "tbl_missing_sqlite",
]

TBL_DATES_TIMES_TEXT_LIST = [
    "tbl_dates_times_text_pd",
    "tbl_dates_times_text_pl",
    "tbl_dates_times_text_parquet",
    "tbl_dates_times_text_duckdb",
    "tbl_dates_times_text_sqlite",
]


@pytest.fixture
def tbl_pd():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_missing_pd():
    return pd.DataFrame({"x": [1, 2, pd.NA, 4], "y": [4, pd.NA, 6, 7], "z": [8, pd.NA, 8, 8]})


@pytest.fixture
def tbl_dates_times_text_pd():
    return pd.DataFrame(
        {
            "date": ["2021-01-01", "2021-02-01", pd.NA],
            "dttm": ["2021-01-01 00:00:00", pd.NA, "2021-02-01 00:00:00"],
            "text": [pd.NA, "5-egh-163", "8-kdg-938"],
        }
    )


@pytest.fixture
def tbl_pl():
    return pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_missing_pl():
    return pl.DataFrame({"x": [1, 2, None, 4], "y": [4, None, 6, 7], "z": [8, None, 8, 8]})


@pytest.fixture
def tbl_dates_times_text_pl():
    return pl.DataFrame(
        {
            "date": ["2021-01-01", "2021-02-01", None],
            "dttm": ["2021-01-01 00:00:00", None, "2021-02-01 00:00:00"],
            "text": [None, "5-egh-163", "8-kdg-938"],
        }
    )


@pytest.fixture
def tbl_parquet():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.parquet"
    return ibis.read_parquet(file_path)


@pytest.fixture
def tbl_missing_parquet():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz_missing.parquet"
    return ibis.read_parquet(file_path)


@pytest.fixture
def tbl_dates_times_text_parquet():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_dates_times_text.parquet"
    return ibis.read_parquet(file_path)


@pytest.fixture
def tbl_duckdb():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.ddb"
    return ibis.connect(f"duckdb://{file_path}").table("tbl_xyz")


@pytest.fixture
def tbl_missing_duckdb():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz_missing.ddb"
    return ibis.connect(f"duckdb://{file_path}").table("tbl_xyz_missing")


@pytest.fixture
def tbl_dates_times_text_duckdb():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_dates_times_text.ddb"
    return ibis.connect(f"duckdb://{file_path}").table("tbl_dates_times_text")


@pytest.fixture
def tbl_sqlite():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.sqlite"
    return ibis.sqlite.connect(file_path).table("tbl_xyz")


@pytest.fixture
def tbl_missing_sqlite():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz_missing.sqlite"
    return ibis.sqlite.connect(file_path).table("tbl_xyz_missing")


@pytest.fixture
def tbl_dates_times_text_sqlite():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_dates_times_text.sqlite"
    return ibis.sqlite.connect(file_path).table("tbl_dates_times_text")


@pytest.fixture
def tbl_pl_variable_names():

    return pl.DataFrame(
        {
            "word": ["apple", "banana"],
            "low_numbers": [1, 2],
            "high_numbers": [13500, 95000],
            "low_floats": [41.6, 41.2],
            "high_floats": [41.6, 41.2],
            "superhigh_floats": [23453.23, 32453532.33],
            "date": ["2021-01-01", "2021-01-02"],
            "datetime": ["2021-01-01 00:00:00", "2021-01-02 00:00:00"],
            "bools": [True, False],
        }
    )


@pytest.fixture
def tbl_pd_variable_names():

    return pd.DataFrame(
        {
            "word": ["apple", "banana"],
            "low_numbers": [1, 2],
            "high_numbers": [13500, 95000],
            "low_floats": [41.6, 41.2],
            "high_floats": [41.6, 41.2],
            "superhigh_floats": [23453.23, 32453532.33],
            "date": ["2021-01-01", "2021-01-02"],
            "datetime": ["2021-01-01 00:00:00", "2021-01-02 00:00:00"],
            "bools": [True, False],
        }
    )


@pytest.fixture
def tbl_memtable_variable_names():
    return ibis.memtable(
        pd.DataFrame(
            {
                "word": ["apple", "banana"],
                "low_numbers": [1, 2],
                "high_numbers": [13500, 95000],
                "low_floats": [41.6, 41.2],
                "high_floats": [41.6, 41.2],
                "superhigh_floats": [23453.23, 32453532.33],
                "date": ["2021-01-01", "2021-01-02"],
                "datetime": ["2021-01-01 00:00:00", "2021-01-02 00:00:00"],
                "bools": [True, False],
            }
        )
    )


@pytest.fixture
def tbl_schema_tests():
    return pl.DataFrame(
        {
            "a": ["apple", "banana", "cherry", "date"],
            "b": [1, 6, 3, 5],
            "c": [1.1, 2.2, 3.3, 4.4],
        }
    )


def test_normalize_reporting_language():

    assert _normalize_reporting_language(lang=None) == "en"
    assert _normalize_reporting_language(lang="en") == "en"
    assert _normalize_reporting_language(lang="IT") == "it"

    # Raise if `lang` value is invalid
    with pytest.raises(ValueError):
        _normalize_reporting_language(lang="invalid")
        _normalize_reporting_language(lang="fr-CA")


def test_validate_class():

    validate = Validate(tbl_pd)

    assert validate.data == tbl_pd
    assert validate.tbl_name is None
    assert validate.label is None
    assert validate.thresholds == Thresholds()
    assert validate.actions is None
    assert validate.lang == "en"
    assert validate.locale == "en"
    assert validate.time_start is None
    assert validate.time_end is None
    assert validate.validation_info == []


def test_validate_class_lang_locale():

    validate_1 = Validate(tbl_pd, lang="fr", locale="fr-CA")

    assert validate_1.lang == "fr"
    assert validate_1.locale == "fr-CA"

    validate_2 = Validate(tbl_pd, lang="de", locale=None)

    assert validate_2.lang == "de"
    assert validate_2.locale == "de"

    # Raise if `lang` value is invalid
    with pytest.raises(ValueError):
        Validate(tbl_pd, lang="invalid")


def test_validation_info():

    v = _ValidationInfo(
        i=1,
        i_o=1,
        step_id="col_vals_gt",
        sha1="a",
        assertion_type="col_vals_gt",
        column="x",
        values=0,
        inclusive=True,
        na_pass=False,
        thresholds=Thresholds(),
        label=None,
        brief=None,
        autobrief=None,
        active=True,
        eval_error=False,
        all_passed=True,
        n=4,
        n_passed=4,
        n_failed=0,
        f_passed=1.0,
        f_failed=0.0,
        warning=None,
        error=None,
        critical=None,
        time_processed="2021-08-01T00:00:00",
        proc_duration_s=0.0,
    )

    assert v.i == 1
    assert v.i_o == 1
    assert v.step_id == "col_vals_gt"
    assert v.sha1 == "a"
    assert v.assertion_type == "col_vals_gt"
    assert v.column == "x"
    assert v.values == 0
    assert v.inclusive is True
    assert v.na_pass is False
    assert v.thresholds == Thresholds()
    assert v.label is None
    assert v.brief is None
    assert v.autobrief is None
    assert v.active is True
    assert v.eval_error is False
    assert v.all_passed is True
    assert v.n == 4
    assert v.n_passed == 4
    assert v.n_failed == 0
    assert v.f_passed == 1.0
    assert v.f_failed == 0.0
    assert v.warning is None
    assert v.error is None
    assert v.critical is None

    assert isinstance(v.time_processed, str)
    assert isinstance(v.proc_duration_s, float)


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_col_vals_all_passing(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(columns="x", value=0).interrogate()

    if tbl_fixture not in ["tbl_parquet", "tbl_duckdb", "tbl_sqlite"]:
        assert v.data.shape == (4, 3)
        assert str(v.data["x"].dtype).lower() == "int64"
        assert str(v.data["y"].dtype).lower() == "int64"
        assert str(v.data["z"].dtype).lower() == "int64"

    # There is a single validation check entry in the `validation_info` attribute
    assert len(v.validation_info) == 1

    # The single step had no failing test units so the `all_passed` attribute is `True`
    assert v.all_passed()

    # Test other validation types for all passing behavior in single steps
    assert Validate(tbl).col_vals_lt(columns="x", value=5).interrogate().all_passed()
    assert Validate(tbl).col_vals_eq(columns="z", value=8).interrogate().all_passed()
    assert Validate(tbl).col_vals_ge(columns="x", value=1).interrogate().all_passed()
    assert Validate(tbl).col_vals_le(columns="x", value=4).interrogate().all_passed()
    assert Validate(tbl).col_vals_between(columns="x", left=0, right=5).interrogate().all_passed()
    assert Validate(tbl).col_vals_outside(columns="x", left=-5, right=0).interrogate().all_passed()
    assert (
        Validate(tbl).col_vals_in_set(columns="x", set=[1, 2, 3, 4, 5]).interrogate().all_passed()
    )
    assert Validate(tbl).col_vals_not_in_set(columns="x", set=[5, 6, 7]).interrogate().all_passed()


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_plan_and_interrogation(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Create a validation plan
    v = Validate(tbl).col_vals_gt(columns="x", value=0)

    # A single validation step was added to the plan so `validation_info` has a single entry
    assert len(v.validation_info) == 1

    # Extract the `validation_info` object to check its attributes
    val_info = v.validation_info[0]

    assert [
        attr
        for attr in val_info.__dict__.keys()
        if not attr.startswith("__") and not attr.endswith("__")
    ] == [
        "i",
        "i_o",
        "step_id",
        "sha1",
        "assertion_type",
        "column",
        "values",
        "inclusive",
        "na_pass",
        "pre",
        "thresholds",
        "actions",
        "label",
        "brief",
        "autobrief",
        "active",
        "eval_error",
        "all_passed",
        "n",
        "n_passed",
        "n_failed",
        "f_passed",
        "f_failed",
        "warning",
        "error",
        "critical",
        "tbl_checked",
        "extract",
        "val_info",
        "time_processed",
        "proc_duration_s",
    ]

    # Check the attributes of the `validation_info` object
    assert val_info.i is None
    assert val_info.i_o == 1
    assert val_info.assertion_type == "col_vals_gt"
    assert val_info.column == "x"
    assert val_info.values == 0
    assert val_info.na_pass is False
    assert val_info.thresholds == Thresholds()
    assert val_info.actions is None
    assert val_info.label is None
    assert val_info.brief is None
    assert val_info.autobrief is None
    assert val_info.active is True
    assert val_info.eval_error is None
    assert val_info.all_passed is None
    assert val_info.n is None
    assert val_info.n_passed is None
    assert val_info.n_failed is None
    assert val_info.f_passed is None
    assert val_info.f_failed is None
    assert val_info.warning is None
    assert val_info.error is None
    assert val_info.critical is None
    assert val_info.tbl_checked is None
    assert val_info.extract is None
    assert val_info.val_info is None
    assert val_info.time_processed is None
    assert val_info.proc_duration_s is None

    # Interrogate the validation plan
    v_int = v.interrogate()

    # The length of the validation info list is still 1
    assert len(v_int.validation_info) == 1

    # Extract the validation info object to check its attributes
    val_info_int = v.validation_info[0]

    # The attribute names of `validation_info` object are the same as before
    assert [
        attr
        for attr in val_info_int.__dict__.keys()
        if not attr.startswith("__") and not attr.endswith("__")
    ] == [
        "i",
        "i_o",
        "step_id",
        "sha1",
        "assertion_type",
        "column",
        "values",
        "inclusive",
        "na_pass",
        "pre",
        "thresholds",
        "actions",
        "label",
        "brief",
        "autobrief",
        "active",
        "eval_error",
        "all_passed",
        "n",
        "n_passed",
        "n_failed",
        "f_passed",
        "f_failed",
        "warning",
        "error",
        "critical",
        "tbl_checked",
        "extract",
        "val_info",
        "time_processed",
        "proc_duration_s",
    ]

    # Check the attributes of the `validation_info` object
    assert val_info.i == 1
    assert val_info.assertion_type == "col_vals_gt"
    assert val_info.column == "x"
    assert val_info.values == 0
    assert val_info.na_pass is False
    assert val_info.thresholds == Thresholds()
    assert val_info.actions is None
    assert val_info.label is None
    assert val_info.brief is None
    assert val_info.autobrief is not None
    assert val_info.active is True
    assert val_info.eval_error is None
    assert val_info.all_passed is True
    assert val_info.n == 4
    assert val_info.n_passed == 4
    assert val_info.n_failed == 0
    assert val_info.f_passed == 1.0
    assert val_info.f_failed == 0.0
    assert val_info.warning is None
    assert val_info.error is None
    assert val_info.critical is None
    assert val_info.tbl_checked is not None
    assert val_info.val_info is None
    assert isinstance(val_info.time_processed, str)
    assert val_info.proc_duration_s > 0.0


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_attr_getters(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(columns="x", value=0).interrogate()

    # Get the total number of test units as a dictionary
    n_dict = v.n()
    assert len(n_dict) == 1
    assert n_dict.keys() == {1}
    assert n_dict[1] == 4

    # Get the number of passing test units
    n_passed_dict = v.n_passed()
    assert len(n_passed_dict) == 1
    assert n_passed_dict.keys() == {1}
    assert n_passed_dict[1] == 4

    # Get the number of failing test units
    n_failed_dict = v.n_failed()
    assert len(n_failed_dict) == 1
    assert n_failed_dict.keys() == {1}
    assert n_failed_dict[1] == 0

    # Get the fraction of passing test units
    f_passed_dict = v.f_passed()
    assert len(f_passed_dict) == 1
    assert f_passed_dict.keys() == {1}
    assert f_passed_dict[1] == 1.0

    # Get the fraction of failing test units
    f_failed_dict = v.f_failed()
    assert len(f_failed_dict) == 1
    assert f_failed_dict.keys() == {1}
    assert f_failed_dict[1] == 0.0

    # Get the 'warning' status
    warning_dict = v.warning()
    assert len(warning_dict) == 1
    assert warning_dict.keys() == {1}
    assert warning_dict[1] is None

    # Get the 'error' status
    error_dict = v.error()
    assert len(error_dict) == 1
    assert error_dict.keys() == {1}
    assert error_dict[1] is None

    # Get the 'critical' status
    critical_dict = v.critical()
    assert len(critical_dict) == 1
    assert critical_dict.keys() == {1}
    assert critical_dict[1] is None


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_attr_getters_no_dict(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(columns="x", value=0).interrogate()

    # Get the total number of test units as a dictionary
    n_val = v.n(i=1, scalar=True)
    assert n_val == 4

    # Get the number of passing test units
    n_passed_val = v.n_passed(i=1, scalar=True)
    assert n_passed_val == 4

    # Get the number of failing test units
    n_failed_val = v.n_failed(i=1, scalar=True)
    assert n_failed_val == 0

    # Get the fraction of passing test units
    f_passed_val = v.f_passed(i=1, scalar=True)
    assert f_passed_val == 1.0

    # Get the fraction of failing test units
    f_failed_val = v.f_failed(i=1, scalar=True)
    assert f_failed_val == 0.0

    # Get the 'warning' status
    warning_val = v.warning(i=1, scalar=True)
    assert warning_val is None

    # Get the 'error' status
    error_val = v.error(i=1, scalar=True)
    assert error_val is None

    # Get the 'critical' status
    critical_val = v.critical(i=1, scalar=True)
    assert critical_val is None


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_get_json_report(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    v = Validate(tbl).col_vals_gt(columns="x", value=0).interrogate()

    assert v.get_json_report() != v.get_json_report(
        exclude_fields=["time_processed", "proc_duration_s"]
    )

    # A ValueError is raised when `use_fields=` includes invalid fields
    with pytest.raises(ValueError):
        v.get_json_report(use_fields=["invalid_field"])

    # A ValueError is raised when `exclude_fields=` includes invalid fields
    with pytest.raises(ValueError):
        v.get_json_report(exclude_fields=["invalid_field"])

    # A ValueError is raised `use_fields=` and `exclude_fields=` are both provided
    with pytest.raises(ValueError):
        v.get_json_report(use_fields=["i"], exclude_fields=["i_o"])


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_report_interrogate_snap(request, tbl_fixture, snapshot):

    tbl = request.getfixturevalue(tbl_fixture)

    report = (
        Validate(tbl)
        .col_vals_gt(columns="x", value=0)
        .interrogate()
        .get_json_report(exclude_fields=["time_processed", "proc_duration_s"])
    )

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(report, "validation_report.json")


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_report_no_interrogate_snap(request, tbl_fixture, snapshot):

    tbl = request.getfixturevalue(tbl_fixture)

    report = (
        Validate(tbl)
        .col_vals_gt(columns="x", value=0)
        .get_json_report(exclude_fields=["time_processed", "proc_duration_s"])
    )

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(report, "validation_report.json")


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_report_use_fields_snap(request, tbl_fixture, snapshot):

    tbl = request.getfixturevalue(tbl_fixture)

    report = (
        Validate(tbl)
        .col_vals_gt(columns="x", value=0)
        .get_json_report(
            use_fields=[
                "i",
                "assertion_type",
                "all_passed",
                "n",
                "f_passed",
                "f_failed",
            ]
        )
    )

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(report, "validation_report.json")


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_report_json_no_steps(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).get_json_report() == "[]"
    assert Validate(tbl).interrogate().get_json_report() == "[]"


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_check_column_input(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Raise a ValueError when `columns=` is not a string
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_lt(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_eq(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ne(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ge(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_le(columns=9, value=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_between(columns=9, left=0, right=5)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_outside(columns=9, left=-5, right=0)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_in_set(columns=9, set=[1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_not_in_set(columns=9, set=[5, 6, 7])
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_null(columns=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_not_null(columns=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_exists(columns=9)


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_check_column_input_with_col(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Check that using `col(column_name)` in `columns=` is allowed and doesn't raise an error
    Validate(tbl).col_vals_gt(columns=col("x"), value=0).interrogate()
    Validate(tbl).col_vals_lt(columns=col("x"), value=0).interrogate()
    Validate(tbl).col_vals_eq(columns=col("x"), value=0).interrogate()
    Validate(tbl).col_vals_ne(columns=col("x"), value=0).interrogate()
    Validate(tbl).col_vals_ge(columns=col("x"), value=0).interrogate()
    Validate(tbl).col_vals_le(columns=col("x"), value=0).interrogate()
    Validate(tbl).col_vals_between(columns=col("x"), left=0, right=5).interrogate()
    Validate(tbl).col_vals_outside(columns=col("x"), left=-5, right=0).interrogate()
    Validate(tbl).col_vals_in_set(columns=col("x"), set=[1, 2, 3, 4, 5]).interrogate()
    Validate(tbl).col_vals_not_in_set(columns=col("x"), set=[5, 6, 7]).interrogate()
    Validate(tbl).col_vals_null(columns=col("x")).interrogate()
    Validate(tbl).col_vals_not_null(columns=col("x")).interrogate()
    Validate(tbl).col_exists(columns=col("x")).interrogate()


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_check_na_pass_input(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Raise a ValueError when `na_pass=` is not a boolean
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_lt(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_eq(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ne(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ge(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_le(columns="x", value=0, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_between(columns="x", left=0, right=5, na_pass=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_outside(columns="x", left=-5, right=0, na_pass=9)


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_check_thresholds_input(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Check that allowed forms for `thresholds=` don't raise an error
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=1)
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=0.1)
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(0.1))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(0.1, 0.2))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(0.1, 0.2, 0.3))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(0.1, 2, 0.3))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, 2))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, 3, 4))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, 0.3, 4))
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds={"warning": 0.1})
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds={"error": 0.1})
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds={"critical": 0.1})
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds={"warning": 0.05, "critical": 0.1})
    Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=Thresholds())
    Validate(tbl).col_vals_gt(
        columns="x", value=0, thresholds=Thresholds(warning=0.1, error=0.2, critical=0.3)
    )
    Validate(tbl).col_vals_gt(
        columns="x", value=0, thresholds=Thresholds(warning=1, error=2, critical=3)
    )

    # Raise a ValueError when `thresholds=` is not one of the allowed types
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds="invalid")
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=[1, 2, 3])
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=-2)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, 2, 3, 4))
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=())
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, -2))
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, thresholds=(1, [2], 3))
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(
            columns="x", value=0, thresholds={"warnings": 0.05, "critical": 0.1}
        )
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(
            columns="x", value=0, thresholds={"warning": 0.05, "critical": -0.1}
        )
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(
            columns="x", value=0, thresholds={"warning": "invalid", "critical": 3}
        )


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_check_active_input(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Raise a ValueError when `active=` is not a boolean
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_lt(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_eq(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ne(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_ge(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_le(columns="x", value=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_between(columns="x", left=0, right=5, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_outside(columns="x", left=-5, right=0, active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_in_set(columns="x", set=[1, 2, 3, 4, 5], active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_not_in_set(columns="x", set=[5, 6, 7], active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_null(columns="x", active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_not_null(columns="x", active=9)
    with pytest.raises(ValueError):
        Validate(tbl).col_exists(columns="x", active=9)


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_check_thresholds_inherit(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Check that the `thresholds=` argument is inherited from Validate, in those steps where
    # it is not explicitly provided (is `None`)
    v = (
        Validate(tbl, thresholds=Thresholds(warning=1, error=2, critical=3))
        .col_vals_gt(columns="x", value=0)
        .col_vals_gt(columns="x", value=0, thresholds=0.5)
        .col_vals_lt(columns="x", value=2)
        .col_vals_lt(columns="x", value=2, thresholds=0.5)
        .col_vals_eq(columns="z", value=4)
        .col_vals_eq(columns="z", value=4, thresholds=0.5)
        .col_vals_ne(columns="z", value=6)
        .col_vals_ne(columns="z", value=6, thresholds=0.5)
        .col_vals_ge(columns="z", value=8)
        .col_vals_ge(columns="z", value=8, thresholds=0.5)
        .col_vals_le(columns="z", value=10)
        .col_vals_le(columns="z", value=10, thresholds=0.5)
        .col_vals_between(columns="x", left=0, right=5)
        .col_vals_between(columns="x", left=0, right=5, thresholds=0.5)
        .col_vals_outside(columns="x", left=-5, right=0)
        .col_vals_outside(columns="x", left=-5, right=0, thresholds=0.5)
        .col_vals_in_set(columns="x", set=[1, 2, 3, 4, 5])
        .col_vals_in_set(columns="x", set=[1, 2, 3, 4, 5], thresholds=0.5)
        .col_vals_not_in_set(columns="x", set=[5, 6, 7])
        .col_vals_not_in_set(columns="x", set=[5, 6, 7], thresholds=0.5)
        .col_vals_null(columns="x")
        .col_vals_null(columns="x", thresholds=0.5)
        .col_vals_not_null(columns="x")
        .col_vals_not_null(columns="x", thresholds=0.5)
        .col_exists(columns="x")
        .col_exists(columns="x", thresholds=0.5)
        .interrogate()
    )

    # `col_vals_gt()` - inherited
    assert v.validation_info[0].thresholds.warning == 1
    assert v.validation_info[0].thresholds.error == 2
    assert v.validation_info[0].thresholds.critical == 3

    # `col_vals_gt()` - overridden
    assert v.validation_info[1].thresholds.warning == 0.5
    assert v.validation_info[1].thresholds.error is None
    assert v.validation_info[1].thresholds.critical is None

    # `col_vals_lt()` - inherited
    assert v.validation_info[2].thresholds.warning == 1
    assert v.validation_info[2].thresholds.error == 2
    assert v.validation_info[2].thresholds.critical == 3

    # `col_vals_lt()` - overridden
    assert v.validation_info[3].thresholds.warning == 0.5
    assert v.validation_info[3].thresholds.error is None
    assert v.validation_info[3].thresholds.critical is None

    # `col_vals_eq()` - inherited
    assert v.validation_info[4].thresholds.warning == 1
    assert v.validation_info[4].thresholds.error == 2
    assert v.validation_info[4].thresholds.critical == 3

    # `col_vals_eq()` - overridden
    assert v.validation_info[5].thresholds.warning == 0.5
    assert v.validation_info[5].thresholds.error is None
    assert v.validation_info[5].thresholds.critical is None

    # `col_vals_ne()` - inherited
    assert v.validation_info[6].thresholds.warning == 1
    assert v.validation_info[6].thresholds.error == 2
    assert v.validation_info[6].thresholds.critical == 3

    # `col_vals_ne()` - overridden
    assert v.validation_info[7].thresholds.warning == 0.5
    assert v.validation_info[7].thresholds.error is None
    assert v.validation_info[7].thresholds.critical is None

    # `col_vals_ge()` - inherited
    assert v.validation_info[8].thresholds.warning == 1
    assert v.validation_info[8].thresholds.error == 2
    assert v.validation_info[8].thresholds.critical == 3

    # `col_vals_ge()` - overridden
    assert v.validation_info[9].thresholds.warning == 0.5
    assert v.validation_info[9].thresholds.error is None
    assert v.validation_info[9].thresholds.critical is None

    # `col_vals_le()` - inherited
    assert v.validation_info[10].thresholds.warning == 1
    assert v.validation_info[10].thresholds.error == 2
    assert v.validation_info[10].thresholds.critical == 3

    # `col_vals_le()` - overridden
    assert v.validation_info[11].thresholds.warning == 0.5
    assert v.validation_info[11].thresholds.error is None
    assert v.validation_info[11].thresholds.critical is None

    # `col_vals_between()` - inherited
    assert v.validation_info[12].thresholds.warning == 1
    assert v.validation_info[12].thresholds.error == 2
    assert v.validation_info[12].thresholds.critical == 3

    # `col_vals_between()` - overridden
    assert v.validation_info[13].thresholds.warning == 0.5
    assert v.validation_info[13].thresholds.error is None
    assert v.validation_info[13].thresholds.critical is None

    # `col_vals_outside()` - inherited
    assert v.validation_info[14].thresholds.warning == 1
    assert v.validation_info[14].thresholds.error == 2
    assert v.validation_info[14].thresholds.critical == 3

    # `col_vals_outside()` - overridden
    assert v.validation_info[15].thresholds.warning == 0.5
    assert v.validation_info[15].thresholds.error is None
    assert v.validation_info[15].thresholds.critical is None

    # `col_vals_in_set()` - inherited
    assert v.validation_info[16].thresholds.warning == 1
    assert v.validation_info[16].thresholds.error == 2
    assert v.validation_info[16].thresholds.critical == 3

    # `col_vals_in_set()` - overridden
    assert v.validation_info[17].thresholds.warning == 0.5
    assert v.validation_info[17].thresholds.error is None
    assert v.validation_info[17].thresholds.critical is None

    # `col_vals_not_in_set()` - inherited
    assert v.validation_info[18].thresholds.warning == 1
    assert v.validation_info[18].thresholds.error == 2
    assert v.validation_info[18].thresholds.critical == 3

    # `col_vals_not_in_set()` - overridden
    assert v.validation_info[19].thresholds.warning == 0.5
    assert v.validation_info[19].thresholds.error is None
    assert v.validation_info[19].thresholds.critical is None

    # `col_vals_null()` - inherited
    assert v.validation_info[20].thresholds.warning == 1
    assert v.validation_info[20].thresholds.error == 2
    assert v.validation_info[20].thresholds.critical == 3

    # `col_vals_null()` - overridden
    assert v.validation_info[21].thresholds.warning == 0.5
    assert v.validation_info[21].thresholds.error is None
    assert v.validation_info[21].thresholds.critical is None

    # `col_vals_not_null()` - inherited
    assert v.validation_info[22].thresholds.warning == 1
    assert v.validation_info[22].thresholds.error == 2
    assert v.validation_info[22].thresholds.critical == 3

    # `col_vals_not_null()` - overridden
    assert v.validation_info[23].thresholds.warning == 0.5
    assert v.validation_info[23].thresholds.error is None
    assert v.validation_info[23].thresholds.critical is None

    # `col_exists()` - inherited
    assert v.validation_info[24].thresholds.warning == 1
    assert v.validation_info[24].thresholds.error == 2
    assert v.validation_info[24].thresholds.critical == 3

    # `col_exists()` - overridden
    assert v.validation_info[25].thresholds.warning == 0.5
    assert v.validation_info[25].thresholds.error is None
    assert v.validation_info[25].thresholds.critical is None


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_autobriefs(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    schema = Schema(columns=["x", "y", "z"])

    # Perform every type of validation step in ways that exercise the autobriefs
    v = (
        Validate(tbl)
        .col_vals_gt(columns="x", value=0)
        .col_vals_gt(columns="x", value=col("y"))
        .col_vals_lt(columns="x", value=2)
        .col_vals_lt(columns="x", value=col("y"))
        .col_vals_eq(columns="z", value=4)
        .col_vals_eq(columns="z", value=col("y"))
        .col_vals_ne(columns="z", value=6)
        .col_vals_ne(columns="z", value=col("y"))
        .col_vals_ge(columns="z", value=8)
        .col_vals_ge(columns="z", value=col("y"))
        .col_vals_le(columns="z", value=10)
        .col_vals_le(columns="z", value=col("y"))
        .col_vals_between(columns="x", left=0, right=5)
        .col_vals_between(columns="x", left=col("y"), right=5)
        .col_vals_between(columns="x", left=0, right=col("z"))
        .col_vals_between(columns="x", left=col("y"), right=col("z"))
        .col_vals_outside(columns="x", left=-5, right=0)
        .col_vals_outside(columns="x", left=col("y"), right=0)
        .col_vals_outside(columns="x", left=-5, right=col("z"))
        .col_vals_outside(columns="x", left=col("y"), right=col("z"))
        .col_vals_in_set(columns="x", set=[1, 2])
        .col_vals_in_set(columns="x", set=[1, 2, 3])
        .col_vals_in_set(columns="x", set=[1, 2, 3, 4])
        .col_vals_in_set(columns="x", set=[1, 2, 3, 4, 5])
        .col_vals_not_in_set(columns="x", set=[1, 2])
        .col_vals_not_in_set(columns="x", set=[1, 2, 3])
        .col_vals_not_in_set(columns="x", set=[1, 2, 3, 4])
        .col_vals_not_in_set(columns="x", set=[1, 2, 3, 4, 5])
        .col_vals_null(columns="x")
        .col_vals_not_null(columns="x")
        .col_exists(columns="x")
        .rows_distinct()
        .rows_distinct(columns_subset=["x", "y"])
        .col_schema_match(schema=schema)
        .row_count_match(count=5)
        .col_count_match(count=3)
        .interrogate()
    )

    # `col_vals_gt()`
    assert v.validation_info[0].autobrief == "Expect that values in `x` should be > `0`."

    # `col_vals_gt()` - column literal
    assert v.validation_info[1].autobrief == "Expect that values in `x` should be > `y`."

    # `col_vals_lt()`
    assert v.validation_info[2].autobrief == "Expect that values in `x` should be < `2`."

    # `col_vals_lt()` - column literal
    assert v.validation_info[3].autobrief == "Expect that values in `x` should be < `y`."

    # `col_vals_eq()`
    assert v.validation_info[4].autobrief == "Expect that values in `z` should be == `4`."

    # `col_vals_eq()` - column literal
    assert v.validation_info[5].autobrief == "Expect that values in `z` should be == `y`."

    # `col_vals_ne()`
    assert v.validation_info[6].autobrief == "Expect that values in `z` should be != `6`."

    # `col_vals_ne()` - column literal
    assert v.validation_info[7].autobrief == "Expect that values in `z` should be != `y`."

    # `col_vals_ge()`
    assert v.validation_info[8].autobrief == "Expect that values in `z` should be >= `8`."

    # `col_vals_ge()` - column literal
    assert v.validation_info[9].autobrief == "Expect that values in `z` should be >= `y`."

    # `col_vals_le()`
    assert v.validation_info[10].autobrief == "Expect that values in `z` should be <= `10`."

    # `col_vals_le()` - column literal
    assert v.validation_info[11].autobrief == "Expect that values in `z` should be <= `y`."

    # `col_vals_between()`
    assert (
        v.validation_info[12].autobrief
        == "Expect that values in `x` should be between `0` and `5`."
    )

    # `col_vals_between()` - left column literal
    assert (
        v.validation_info[13].autobrief
        == "Expect that values in `x` should be between `y` and `5`."
    )

    # `col_vals_between()` - right column literal
    assert (
        v.validation_info[14].autobrief
        == "Expect that values in `x` should be between `0` and `z`."
    )

    # `col_vals_between()` - left and right column literal
    assert (
        v.validation_info[15].autobrief
        == "Expect that values in `x` should be between `y` and `z`."
    )

    # `col_vals_outside()`
    assert (
        v.validation_info[16].autobrief
        == "Expect that values in `x` should not be between `-5` and `0`."
    )

    # `col_vals_outside()` - left column literal
    assert (
        v.validation_info[17].autobrief
        == "Expect that values in `x` should not be between `y` and `0`."
    )

    # `col_vals_outside()` - right column literal
    assert (
        v.validation_info[18].autobrief
        == "Expect that values in `x` should not be between `-5` and `z`."
    )

    # `col_vals_outside()` - left and right column literal
    assert (
        v.validation_info[19].autobrief
        == "Expect that values in `x` should not be between `y` and `z`."
    )

    # `col_vals_in_set()` - 2 elements
    assert (
        v.validation_info[20].autobrief
        == "Expect that values in `x` should be in the set of `1`, `2`."
    )

    # `col_vals_in_set()` - 3 elements
    assert (
        v.validation_info[21].autobrief
        == "Expect that values in `x` should be in the set of `1`, `2`, `3`."
    )

    # `col_vals_in_set()` - 4 elements
    assert (
        v.validation_info[22].autobrief
        == "Expect that values in `x` should be in the set of `1`, `2`, `3`, and 1 more."
    )

    # `col_vals_in_set()` - 5 elements
    assert (
        v.validation_info[23].autobrief
        == "Expect that values in `x` should be in the set of `1`, `2`, `3`, and 2 more."
    )

    # `col_vals_not_in_set()` - 2 elements
    assert (
        v.validation_info[24].autobrief
        == "Expect that values in `x` should not be in the set of `1`, `2`."
    )

    # `col_vals_not_in_set()` - 3 elements
    assert (
        v.validation_info[25].autobrief
        == "Expect that values in `x` should not be in the set of `1`, `2`, `3`."
    )

    # `col_vals_not_in_set()` - 4 elements
    assert (
        v.validation_info[26].autobrief
        == "Expect that values in `x` should not be in the set of `1`, `2`, `3`, and 1 more."
    )

    # `col_vals_not_in_set()` - 5 elements
    assert (
        v.validation_info[27].autobrief
        == "Expect that values in `x` should not be in the set of `1`, `2`, `3`, and 2 more."
    )

    # `col_vals_null()`
    assert v.validation_info[28].autobrief == "Expect that all values in `x` should be Null."

    # `col_vals_not_null()`
    assert v.validation_info[29].autobrief == "Expect that all values in `x` should not be Null."

    # `col_exists()`
    assert v.validation_info[30].autobrief == "Expect that column `x` exists."

    # `rows_distinct()`
    assert v.validation_info[31].autobrief == "Expect entirely distinct rows across all columns."

    # `rows_distinct()` - subset of columns
    assert v.validation_info[32].autobrief == "Expect entirely distinct rows across `x`, `y`."

    # `col_schema_match()`
    assert v.validation_info[33].autobrief == "Expect that column schemas match."

    # `row_count_match()`
    assert v.validation_info[34].autobrief == "Expect that the row count is exactly `5`."

    # `col_count_match()`
    assert v.validation_info[35].autobrief == "Expect that the column count is exactly `3`."


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_actions_inherit_case(request, tbl_fixture, capsys):

    tbl = request.getfixturevalue(tbl_fixture)

    # Check that the `actions=` argument is inherited from Validate
    (
        Validate(
            data=tbl,
            thresholds=Thresholds(warning=1, error=2, critical=3),
            actions=Actions(critical="notification"),
        )
        .col_vals_gt(columns="x", value=10000)
        .interrogate()
    )

    # Capture the output and verify that "notification" was printed to the console
    captured = capsys.readouterr()
    assert "notification" in captured.out


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_actions_override_case(request, tbl_fixture, capsys):

    tbl = request.getfixturevalue(tbl_fixture)

    # Check that the `actions=` argument is *not* inherited from Validate
    (
        Validate(
            data=tbl,
            thresholds=Thresholds(warning=1, error=2, critical=3),
            actions=Actions(critical="notification"),
        )
        .col_vals_gt(columns="x", value=10000, actions=Actions(critical="notification override"))
        .interrogate()
    )

    # Capture the output and verify that "notification override" was printed to the console
    captured = capsys.readouterr()
    assert "notification override" in captured.out


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_actions_multiple_actions_inherit(request, tbl_fixture, capsys):

    def notify():
        print("NOTIFIER")

    tbl = request.getfixturevalue(tbl_fixture)

    (
        Validate(
            data=tbl,
            thresholds=Thresholds(warning=1, error=2, critical=3),
            actions=Actions(critical=["notification", notify]),
        )
        .col_vals_gt(columns="x", value=10000)
        .interrogate()
    )

    # Capture the output and verify that "notification" was printed to the console
    captured = capsys.readouterr()
    assert "notification" in captured.out
    assert "NOTIFIER" in captured.out

    # Verify that "notification" is emitted before "NOTIFIER"
    notification_index = captured.out.index("notification")
    notifier_index = captured.out.index("NOTIFIER")
    assert notification_index < notifier_index


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_actions_multiple_actions_override(request, tbl_fixture, capsys):

    def notify():
        print("NOTIFIER")

    def notify_step():
        print("NOTIFY STEP")

    tbl = request.getfixturevalue(tbl_fixture)

    (
        Validate(
            data=tbl,
            thresholds=Thresholds(warning=1, error=2, critical=3),
            actions=Actions(critical=["notification", notify]),
        )
        .col_vals_gt(
            columns="x", value=10000, actions=Actions(critical=["step notify", notify_step])
        )
        .interrogate()
    )

    # Capture the output and verify that "notification" was printed to the console
    captured = capsys.readouterr()
    assert "step notify" in captured.out
    assert "NOTIFY STEP" in captured.out

    # Verify that "step notify" is emitted before "NOTIFY STEP"
    notification_index = captured.out.index("step notify")
    notifier_index = captured.out.index("NOTIFY STEP")
    assert notification_index < notifier_index


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_actions_multiple_actions_step_only(request, tbl_fixture, capsys):

    def notify_step():
        print("NOTIFY STEP")

    tbl = request.getfixturevalue(tbl_fixture)

    (
        Validate(
            data=tbl,
            thresholds=Thresholds(warning=1, error=2, critical=3),
        )
        .col_vals_gt(
            columns="x", value=10000, actions=Actions(critical=["step notify", notify_step])
        )
        .interrogate()
    )

    # Capture the output and verify that "notification" was printed to the console
    captured = capsys.readouterr()
    assert "step notify" in captured.out
    assert "NOTIFY STEP" in captured.out

    # Verify that "step notify" is emitted before "NOTIFY STEP"
    notification_index = captured.out.index("step notify")
    notifier_index = captured.out.index("NOTIFY STEP")
    assert notification_index < notifier_index


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_actions_inherit_none(request, tbl_fixture, capsys):

    tbl = request.getfixturevalue(tbl_fixture)

    (
        Validate(
            data=tbl,
            thresholds=Thresholds(warning=1, error=2, critical=3),
            actions=Actions(critical=None),
        )
        .col_vals_gt(columns="x", value=10000)
        .interrogate()
    )

    # Capture the output and verify that nothing was printed to the console
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_actions_override_none(request, tbl_fixture, capsys):

    tbl = request.getfixturevalue(tbl_fixture)

    (
        Validate(
            data=tbl,
            thresholds=Thresholds(warning=1, error=2, critical=3),
            actions=Actions(critical=None),
        )
        .col_vals_gt(columns="x", value=10000, actions=Actions(critical=None))
        .interrogate()
    )

    # Capture the output and verify that nothing was printed to the console
    captured = capsys.readouterr()
    assert captured.out == ""


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_actions_step_only_none(request, tbl_fixture, capsys):

    tbl = request.getfixturevalue(tbl_fixture)

    (
        Validate(
            data=tbl,
            thresholds=Thresholds(warning=1, error=2, critical=3),
        )
        .col_vals_gt(columns="x", value=10000, actions=Actions(critical=None))
        .interrogate()
    )

    # Capture the output and verify that nothing was printed to the console
    captured = capsys.readouterr()
    assert captured.out == ""


def test_validation_with_preprocessing_pd(tbl_pd):

    v = (
        Validate(tbl_pd)
        .col_vals_eq(columns="z", value=8)
        .col_vals_eq(columns="z", value=16, pre=lambda df: df.assign(z=df["z"] * 2))
        .interrogate()
    )

    assert v.n_passed()[1] == 4
    assert v.n_passed()[2] == 4


def test_validation_with_preprocessing_pd_use_nw(tbl_pd):

    v = (
        Validate(tbl_pd)
        .col_vals_eq(columns="z", value=8)
        .col_vals_eq(columns="z", value=16, pre=lambda dfn: dfn.with_columns(z=nw.col("z") * 2))
        .interrogate()
    )

    assert v.n_passed()[1] == 4
    assert v.n_passed()[2] == 4


def test_validation_with_preprocessing_with_fn_pd(tbl_pd):

    def multiply_z_by_two(df):
        return df.assign(z=df["z"] * 2)

    v = (
        Validate(tbl_pd)
        .col_vals_eq(columns="z", value=8)
        .col_vals_eq(columns="z", value=16, pre=multiply_z_by_two)
        .interrogate()
    )

    assert v.n_passed()[1] == 4
    assert v.n_passed()[2] == 4


def test_validation_with_preprocessing_pl(tbl_pl):

    v = (
        Validate(tbl_pl)
        .col_vals_eq(columns="z", value=8)
        .col_vals_eq(columns="z", value=16, pre=lambda df: df.with_columns(z=pl.col("z") * 2))
        .interrogate()
    )

    assert v.n_passed()[1] == 4
    assert v.n_passed()[2] == 4


def test_validation_with_preprocessing_pl_use_nw(tbl_pl):

    v = (
        Validate(tbl_pl)
        .col_vals_eq(columns="z", value=8)
        .col_vals_eq(columns="z", value=16, pre=lambda dfn: dfn.with_columns(z=nw.col("z") * 2))
        .interrogate()
    )

    assert v.n_passed()[1] == 4
    assert v.n_passed()[2] == 4


def test_validation_with_preprocessing_with_fn_pl(tbl_pl):

    def multiply_z_by_two(df):
        return df.with_columns(z=pl.col("z") * 2)

    v = (
        Validate(tbl_pl)
        .col_vals_eq(columns="z", value=8)
        .col_vals_eq(columns="z", value=16, pre=multiply_z_by_two)
        .interrogate()
    )

    assert v.n_passed()[1] == 4
    assert v.n_passed()[2] == 4


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_gt(request, tbl_fixture):

    pl.DataFrame({"x": [1, 2, None, 4], "y": [4, None, 6, 7], "z": [8, None, 8, 8]})

    tbl = request.getfixturevalue(tbl_fixture)

    validation_1 = Validate(tbl).col_vals_gt(columns="x", value=0).interrogate()

    assert validation_1.n_passed(i=1, scalar=True) == 3
    assert validation_1.n_failed(i=1, scalar=True) == 1

    validation_2 = Validate(tbl).col_vals_gt(columns="x", value=0, na_pass=True).interrogate()

    assert validation_2.n_passed(i=1, scalar=True) == 4
    assert validation_2.n_failed(i=1, scalar=True) == 0


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_lt(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation_1 = Validate(tbl).col_vals_lt(columns="x", value=10).interrogate()

    assert validation_1.n_passed(i=1, scalar=True) == 3
    assert validation_1.n_failed(i=1, scalar=True) == 1

    validation_2 = Validate(tbl).col_vals_lt(columns="x", value=10, na_pass=True).interrogate()

    assert validation_2.n_passed(i=1, scalar=True) == 4
    assert validation_2.n_failed(i=1, scalar=True) == 0


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_eq(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation_1 = Validate(tbl).col_vals_eq(columns="z", value=8).interrogate()

    assert validation_1.n_passed(i=1, scalar=True) == 3
    assert validation_1.n_failed(i=1, scalar=True) == 1

    validation_2 = Validate(tbl).col_vals_eq(columns="z", value=8, na_pass=True).interrogate()

    assert validation_2.n_passed(i=1, scalar=True) == 4
    assert validation_2.n_failed(i=1, scalar=True) == 0


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_ne(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation_1 = Validate(tbl).col_vals_ne(columns="z", value=7).interrogate()

    assert validation_1.n_passed(i=1, scalar=True) == 3
    assert validation_1.n_failed(i=1, scalar=True) == 1

    validation_2 = Validate(tbl).col_vals_ne(columns="z", value=7, na_pass=True).interrogate()

    assert validation_2.n_passed(i=1, scalar=True) == 4
    assert validation_2.n_failed(i=1, scalar=True) == 0


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_ge(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation_1 = Validate(tbl).col_vals_ge(columns="x", value=1).interrogate()

    assert validation_1.n_passed(i=1, scalar=True) == 3
    assert validation_1.n_failed(i=1, scalar=True) == 1

    validation_2 = Validate(tbl).col_vals_ge(columns="x", value=1, na_pass=True).interrogate()

    assert validation_2.n_passed(i=1, scalar=True) == 4
    # assert validation_2.n_failed(i=1, scalar=True) == 0


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_le(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation_1 = Validate(tbl).col_vals_le(columns="x", value=4).interrogate()

    assert validation_1.n_passed(i=1, scalar=True) == 3
    assert validation_1.n_failed(i=1, scalar=True) == 1

    validation_2 = Validate(tbl).col_vals_le(columns="x", value=4, na_pass=True).interrogate()

    assert validation_2.n_passed(i=1, scalar=True) == 4
    assert validation_2.n_failed(i=1, scalar=True) == 0


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_between(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation_1 = Validate(tbl).col_vals_between(columns="x", left=1, right=4).interrogate()

    assert validation_1.n_passed(i=1, scalar=True) == 3
    assert validation_1.n_failed(i=1, scalar=True) == 1

    validation_2 = (
        Validate(tbl).col_vals_between(columns="x", left=1, right=4, na_pass=True).interrogate()
    )

    assert validation_2.n_passed(i=1, scalar=True) == 4
    assert validation_2.n_failed(i=1, scalar=True) == 0

    validation_3 = (
        Validate(tbl).col_vals_between(columns="x", left=11, right=14, na_pass=False).interrogate()
    )

    assert validation_3.n_passed(i=1, scalar=True) == 0
    assert validation_3.n_failed(i=1, scalar=True) == 4

    validation_4 = (
        Validate(tbl).col_vals_between(columns="x", left=11, right=14, na_pass=True).interrogate()
    )

    assert validation_4.n_passed(i=1, scalar=True) == 1
    assert validation_4.n_failed(i=1, scalar=True) == 3

    validtion_5 = (
        Validate(tbl)
        .col_vals_between(columns="x", left=1, right=4, inclusive=(False, True), na_pass=True)
        .interrogate()
    )

    assert validtion_5.n_passed(i=1, scalar=True) == 3
    assert validtion_5.n_failed(i=1, scalar=True) == 1

    validation_6 = (
        Validate(tbl)
        .col_vals_between(columns="x", left=1, right=4, inclusive=(True, False), na_pass=True)
        .interrogate()
    )

    assert validation_6.n_passed(i=1, scalar=True) == 3
    assert validation_6.n_failed(i=1, scalar=True) == 1

    validation_7 = (
        Validate(tbl)
        .col_vals_between(columns="x", left=1, right=4, inclusive=(False, False), na_pass=True)
        .interrogate()
    )

    assert validation_7.n_passed(i=1, scalar=True) == 2
    assert validation_7.n_failed(i=1, scalar=True) == 2

    validation_8 = (
        Validate(tbl)
        .col_vals_between(columns="x", left=1, right=4, inclusive=(False, False), na_pass=False)
        .interrogate()
    )

    assert validation_8.n_passed(i=1, scalar=True) == 1
    assert validation_8.n_failed(i=1, scalar=True) == 3


@pytest.mark.parametrize("tbl_fixture", TBL_MISSING_LIST)
def test_col_vals_outside(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation_1 = Validate(tbl).col_vals_outside(columns="x", left=5, right=8).interrogate()

    assert validation_1.n_passed(i=1, scalar=True) == 3
    assert validation_1.n_failed(i=1, scalar=True) == 1

    validation_2 = (
        Validate(tbl).col_vals_outside(columns="x", left=5, right=8, na_pass=True).interrogate()
    )

    assert validation_2.n_passed(i=1, scalar=True) == 4
    assert validation_2.n_failed(i=1, scalar=True) == 0

    validation_3 = (
        Validate(tbl).col_vals_outside(columns="x", left=4, right=8, na_pass=False).interrogate()
    )

    assert validation_3.n_passed(i=1, scalar=True) == 2
    assert validation_3.n_failed(i=1, scalar=True) == 2

    validation_4 = (
        Validate(tbl).col_vals_outside(columns="x", left=-4, right=1, na_pass=False).interrogate()
    )

    assert validation_4.n_passed(i=1, scalar=True) == 2
    assert validation_4.n_failed(i=1, scalar=True) == 2

    validation_5 = (
        Validate(tbl)
        .col_vals_outside(columns="x", left=1, right=4, inclusive=(True, True), na_pass=False)
        .interrogate()
    )

    assert validation_5.n_passed(i=1, scalar=True) == 0
    assert validation_5.n_failed(i=1, scalar=True) == 4

    validation_6 = (
        Validate(tbl)
        .col_vals_outside(columns="x", left=1, right=4, inclusive=(True, True), na_pass=True)
        .interrogate()
    )

    assert validation_6.n_passed(i=1, scalar=True) == 1
    assert validation_6.n_failed(i=1, scalar=True) == 3

    validation_7 = (
        Validate(tbl)
        .col_vals_outside(columns="x", left=4, right=8, inclusive=(False, True), na_pass=False)
        .interrogate()
    )

    assert validation_7.n_passed(i=1, scalar=True) == 3
    assert validation_7.n_failed(i=1, scalar=True) == 1

    validation_8 = (
        Validate(tbl)
        .col_vals_outside(columns="x", left=-4, right=1, inclusive=(True, False), na_pass=False)
        .interrogate()
    )

    assert validation_8.n_passed(i=1, scalar=True) == 3
    assert validation_8.n_failed(i=1, scalar=True) == 1

    validation_9 = (
        Validate(tbl)
        .col_vals_outside(columns="x", left=1, right=4, inclusive=(False, False), na_pass=True)
        .interrogate()
    )

    assert validation_9.n_passed(i=1, scalar=True) == 3
    assert validation_9.n_failed(i=1, scalar=True) == 1

    validation_10 = (
        Validate(tbl)
        .col_vals_outside(columns="x", left=1, right=4, inclusive=(False, False), na_pass=False)
        .interrogate()
    )

    assert validation_10.n_passed(i=1, scalar=True) == 2
    assert validation_10.n_failed(i=1, scalar=True) == 2


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_col_vals_in_set(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_in_set(columns="x", set=[1, 2, 3, 4])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_in_set(columns="x", set=[0, 1, 2, 3, 4, 5, 6])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_in_set(columns="x", set=[1.0, 2.0, 3.0, 4.0])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_in_set(columns="x", set=[1.00001, 2.00001, 3.00001, 4.00001])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_in_set(columns="x", set=[-1, -2, -3, -4])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_col_vals_not_in_set(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_not_in_set(columns="x", set=[5, 6, 7])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_not_in_set(columns="x", set=[0, 1, 2, 3, 4, 5, 6])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_not_in_set(columns="x", set=[1.0, 2.0, 3.0, 4.0])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_not_in_set(columns="x", set=[1.00001, 2.00001, 3.00001, 4.00001])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_not_in_set(columns="x", set=[-1, -2, -3, -4])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )


@pytest.mark.parametrize("tbl_fixture", TBL_DATES_TIMES_TEXT_LIST)
def test_col_vals_regex(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_regex(columns="text", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}")
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_regex(columns="text", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}", na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_regex(columns="text", pattern=r"^[0-9]-[a-z]{3}-[0-9]{3}$", na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )


def test_col_vals_expr_polars_tbl():

    df = load_dataset(tbl_type="polars")

    pl_expr = (pl.col("c") > pl.col("a")) & (pl.col("d") > pl.col("c"))
    nw_expr = (nw.col("c") > nw.col("a")) & (nw.col("d") > nw.col("c"))

    assert (
        Validate(data=df).col_vals_expr(expr=pl_expr).interrogate().n_passed(i=1, scalar=True) == 6
    )
    assert (
        Validate(data=df).col_vals_expr(expr=pl_expr).interrogate().n_failed(i=1, scalar=True) == 5
    )

    assert (
        Validate(data=df).col_vals_expr(expr=nw_expr).interrogate().n_passed(i=1, scalar=True) == 6
    )
    assert (
        Validate(data=df).col_vals_expr(expr=nw_expr).interrogate().n_failed(i=1, scalar=True) == 5
    )


def test_col_vals_expr_pandas_tbl():

    df = load_dataset(tbl_type="pandas")

    pd_expr = lambda df: (df["c"] > df["a"]) & (df["d"] > df["c"])  # noqa
    nw_expr = (nw.col("c") > nw.col("a")) & (nw.col("d") > nw.col("c"))

    assert (
        Validate(data=df).col_vals_expr(expr=pd_expr).interrogate().n_passed(i=1, scalar=True) == 6
    )
    assert (
        Validate(data=df).col_vals_expr(expr=pd_expr).interrogate().n_failed(i=1, scalar=True) == 7
    )

    assert (
        Validate(data=df).col_vals_expr(expr=nw_expr).interrogate().n_passed(i=1, scalar=True) == 6
    )
    assert (
        Validate(data=df).col_vals_expr(expr=nw_expr).interrogate().n_failed(i=1, scalar=True) == 7
    )


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_rows_distinct(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).rows_distinct().interrogate().n_passed(i=1, scalar=True) == 4
    assert (
        Validate(tbl)
        .rows_distinct(columns_subset=["x", "y"])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .rows_distinct(columns_subset=["y", "z"])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .rows_distinct(columns_subset=["x", "z"])
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl).rows_distinct(columns_subset="x").interrogate().n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl).rows_distinct(columns_subset="y").interrogate().n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl).rows_distinct(columns_subset="z").interrogate().n_passed(i=1, scalar=True)
        == 0
    )


def test_col_schema_match():

    tbl = pl.DataFrame(
        {
            "a": ["apple", "banana", "cherry", "date"],
            "b": [1, 6, 3, 5],
            "c": [1.1, 2.2, 3.3, 4.4],
        }
    )

    # Completely correct schema supplied to `columns=`
    schema = Schema(columns=[("a", "String"), ("b", "Int64"), ("c", "Float64")])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Completely correct schema supplied to `columns=` (using dictionary)
    schema = Schema(columns={"a": "String", "b": "Int64", "c": "Float64"})
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Completely correct schema (using kwargs)
    schema = Schema(columns={"a": "String", "b": "Int64", "c": "Float64"})
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Schema produced using the tbl object (supplied to `tbl=`)
    schema = Schema(tbl=tbl)
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Having an incorrect dtype in supplied schema
    schema = Schema(columns=[("a", "wrong"), ("b", "Int64"), ("c", "Float64")])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Schema expressed in a different order (yet complete)
    schema = Schema(columns=[("b", "Int64"), ("c", "Float64"), ("a", "String")])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Schema expressed in a different order (yet complete) - wrong column name
    schema = Schema(columns=[("b", "Int64"), ("c", "Float64"), ("wrong", "String")])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Schema has duplicate column/dtype
    schema = Schema(columns=[("a", "String"), ("a", "String"), ("b", "Int64"), ("c", "Float64")])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Schema has duplicate column/dtype - wrong column name
    schema = Schema(
        columns=[("a", "String"), ("a", "String"), ("wrong", "Int64"), ("c", "Float64")]
    )
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Supplied schema is a subset of the actual schema (in the correct order)
    schema = Schema(columns=[("b", "Int64"), ("c", "Float64")])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Supplied schema is a subset of the actual schema (in the correct order) - wrong column name
    schema = Schema(columns=[("wrong", "Int64"), ("c", "Float64")])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Supplied schema is a subset of the actual schema but in a different order
    schema = Schema(columns=[("c", "Float64"), ("b", "Int64")])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Supplied schema is a subset of the actual schema but in a different order - wrong column name
    schema = Schema(columns=[("wrong", "Float64"), ("b", "Int64")])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Completely correct schema supplied to `columns=` except for the case mismatch in colnames
    schema = Schema(columns=[("a", "String"), ("B", "Int64"), ("C", "Float64")])
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, case_sensitive_colnames=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema, complete=True, in_order=False, case_sensitive_colnames=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema, complete=False, in_order=True, case_sensitive_colnames=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema, complete=False, in_order=False, case_sensitive_colnames=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Completely correct schema supplied to `columns=` except for the case mismatch in dtypes
    schema = Schema(columns=[("a", "string"), ("b", "INT64"), ("c", "FloaT64")])
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, case_sensitive_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=True, in_order=False, case_sensitive_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=True, case_sensitive_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema, complete=False, in_order=False, case_sensitive_dtypes=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Completely correct schema supplied to `columns=` except for the case mismatch in
    # colnames and dtypes
    schema = Schema(columns=[("A", "string"), ("b", "INT64"), ("C", "FloaT64")])
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, case_sensitive_colnames=False, case_sensitive_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=True,
            in_order=False,
            case_sensitive_colnames=False,
            case_sensitive_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=True,
            case_sensitive_colnames=False,
            case_sensitive_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=False,
            case_sensitive_colnames=False,
            case_sensitive_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Matching dtypes with substrings in the supplied schema (`full_match_dtypes=False` case)
    schema = Schema(columns=[("a", "Str"), ("b", "Int"), ("c", "Float64")])
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=True, in_order=False, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=True, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=False, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Matching dtypes with substrings in the supplied schema (`full_match_dtypes=True` case)
    schema = Schema(columns=[("a", "Str"), ("b", "Int"), ("c", "Float64")])
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, full_match_dtypes=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=True, in_order=False, full_match_dtypes=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=True, full_match_dtypes=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=False, full_match_dtypes=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Matching dtypes with substrings in the supplied schema and using case-insensitive matching
    schema = Schema(columns=[("a", "str"), ("b", "Int"), ("c", "float64")])
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, case_sensitive_dtypes=False, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=True,
            in_order=False,
            case_sensitive_dtypes=False,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=True,
            case_sensitive_dtypes=False,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=False,
            case_sensitive_dtypes=False,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Matching dtypes with substrings in the supplied schema and using case-insensitive matching
    # (`case_sensitive_dtypes=True` case)
    schema = Schema(columns=[("a", "str"), ("b", "Int"), ("c", "float64")])
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, case_sensitive_dtypes=True, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=True,
            in_order=False,
            case_sensitive_dtypes=True,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=True,
            case_sensitive_dtypes=True,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=False,
            case_sensitive_dtypes=True,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_row_count_match(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).row_count_match(count=4).interrogate().n_passed(i=1, scalar=True) == 1

    assert (
        Validate(tbl)
        .row_count_match(count=3, inverse=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    assert Validate(tbl).row_count_match(count=tbl).interrogate().n_passed(i=1, scalar=True) == 1

@pytest.mark.parametrize(('val','e', 'exc'),
                         [
                             ((-1, 5), ValueError, 'If passing a tuple to `tol`, both bounds should be positive. Each limit is'),
                             ([100, 5], TypeError, '`tol` must be a tuple of limits'),
                             ((5, -1), ValueError, 'If passing a tuple to `tol`, both bounds should be positive. Each limit is'),
                             ((None, .05), TypeError, 'Each element passed to `tol` must be an integer if defining upper and lower'),
                             (('fooval', 100), TypeError, 'Each element passed to `tol` must be an integer if defining upper and lower'),
                             (-1, ValueError, 'Value passed to `tol` must be positive'),
                         ])
def test_invalid_row_count_tol(val : float | tuple[int, int], e: Exception, exc :str) -> None:
    data = pl.DataFrame({"foocol": [1, 2, 3]})

    with pytest.raises(expected_exception=e, match = exc):
        Validate(data=data).row_count_match(count = 3, tol=val)


def test_row_count_example_tol() -> None:
    small_table = load_dataset("small_table")
    smaller_small_table = small_table.sample(n = 12) # within the lower bound
    (
        Validate(data=smaller_small_table)
        .row_count_match(count=13,tol=(2, 0)) # minus 2 but plus 0, ie. 11-13
        .interrogate()
        .assert_passing()
    )

    (
            Validate(data=smaller_small_table)
            .row_count_match(count=13,tol=.05) # .05% tolerance of 13
            .interrogate()
            .assert_passing()
        )

    even_smaller_table = small_table.sample(n = 2)
    with pytest.raises(AssertionError):
        (
            Validate(data=even_smaller_table)
            .row_count_match(count=13,tol=5) # plus or minus 5; this test will fail
            .interrogate()
            .assert_passing()
        )

@pytest.mark.parametrize(('nrows','target_count','tol','should_pass'),
                         [
                            (98, 100, .05, True),
                            (98, 100, 5, True),
                            (104, 100, (5, 5), True),
                            (0, 100, .05, False),
                            (0, 100, 5, False),
                            (0, 100, (5, 5), False),
                            (98, 100, .95, True),
                         ])
def test_row_count_tol(nrows : int, target_count: int, tol : float | tuple[int, int], should_pass : bool) -> None:
    data = pl.DataFrame({"foocol": [random.random()] * nrows})

    catcher = contextlib.nullcontext if should_pass else partial(pytest.raises, AssertionError, match = "All tests did not pass")

    with catcher():
        Validate(data = data).row_count_match(
            count = target_count,
            tol = tol
        ).interrogate().assert_passing()

@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_col_count_match(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_count_match(count=3).interrogate().n_passed(i=1, scalar=True) == 1

    assert (
        Validate(tbl)
        .col_count_match(count=8, inverse=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    assert Validate(tbl).col_count_match(count=tbl).interrogate().n_passed(i=1, scalar=True) == 1


def test_col_schema_match_list_of_dtypes():

    tbl = pl.DataFrame(
        {
            "a": ["apple", "banana", "cherry", "date"],
            "b": [1, 6, 3, 5],
            "c": [1.1, 2.2, 3.3, 4.4],
        }
    )

    # Completely correct schema supplied, using 1-element lists for dtypes
    schema = Schema(columns=[("a", ["String"]), ("b", ["Int64"]), ("c", ["Float64"])])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Completely correct schema supplied, using 1-element lists for dtypes (using dict for schema)
    schema = Schema(columns={"a": ["String"], "b": ["Int64"], "c": ["Float64"]})
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Completely correct schema supplied, using 1-element lists for dtypes (using kwargs for schema)
    schema = Schema(a=["String"], b=["Int64"], c=["Float64"])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Having one of two dtypes being correct in 2-element lists for dtypes
    schema = Schema(
        columns=[("a", ["str", "String"]), ("b", ["Int64", "Int"]), ("c", ["Float64", "float"])]
    )
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Having one of two dtypes being correct in 2-element lists for dtypes (using dict for schema)
    schema = Schema(
        columns={"a": ["str", "String"], "b": ["Int64", "Int"], "c": ["Float64", "float"]}
    )
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Having one of two dtypes being correct in 2-element lists for dtypes (using kwargs for schema)
    schema = Schema(a=["str", "String"], b=["Int64", "Int"], c=["Float64", "float"])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Having mix of scalars and lists for dtypes
    schema = Schema(columns=[("a", "String"), ("b", ["Int64"]), ("c", ["float", "Float64"])])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Having duplicate items in dtype lists is allowed
    schema = Schema(
        columns=[
            ("a", ["str", "String", "str"]),
            ("b", ["Int64", "Int64"]),
            ("c", ["Float64", "Float64", "float"]),
        ]
    )
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Having all incorrect dtypes in a list of dtypes
    schema = Schema(
        columns=[
            ("a", ["wrong", "incorrect"]),
            ("b", ["Int64", "int"]),
            ("c", ["float", "Float64"]),
        ]
    )
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Schema expressed in a different order (yet complete)
    schema = Schema(columns=[("b", ["Int64", "int"]), ("c", ["float", "Float64"]), ("a", "String")])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Schema expressed in a different order (yet complete) - wrong column name
    schema = Schema(
        columns=[("b", ["int", "Int64"]), ("c", ["float", "Float64"]), ("wrong", ["String", "str"])]
    )
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Schema has duplicate column/dtype
    schema = Schema(
        columns=[
            ("a", ["String", "str"]),
            ("a", ["str", "String"]),
            ("b", ["Int64", "int"]),
            ("c", ["Float64", "float"]),
        ]
    )
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Schema has duplicate column/dtype - wrong dtypes in one case
    schema = Schema(
        columns=[
            ("a", ["String", "str"]),
            ("a", ["wrong", "Wrong"]),
            ("b", ["Int64", "int"]),
            ("c", ["Float64", "float"]),
        ]
    )
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Schema has duplicate column/dtype - wrong dtypes in both cases
    schema = Schema(
        columns=[
            ("a", ["wrong", "Wrong"]),
            ("a", ["wrong", "Wrong"]),
            ("b", ["Int64", "int"]),
            ("c", ["Float64", "float"]),
        ]
    )
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Schema has duplicate column/dtype - wrong column name
    schema = Schema(
        columns=[
            ("a", ["String", "str"]),
            ("a", ["str", "String"]),
            ("wrong", ["Int64", "int"]),
            ("c", ["Float64", "float"]),
        ]
    )
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Supplied schema is a subset of the actual schema (in the correct order)
    schema = Schema(columns=[("b", ["Int64", "int"]), ("c", ["float", "Float64"])])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Supplied schema is a subset of the actual schema (in the correct order) - wrong column name
    schema = Schema(columns=[("wrong", ["Int64", "int"]), ("c", ["Float64", "float"])])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Supplied schema is a subset of the actual schema but in a different order
    schema = Schema(columns=[("c", ["float", "Float64"]), ("b", ["Int64", "int"])])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Supplied schema is a subset of the actual schema but in a different order - wrong column name
    schema = Schema(columns=[("wrong", ["float", "Float64"]), ("b", ["Int64", "int"])])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Completely correct schema supplied to `columns=` except for the case mismatch in colnames
    schema = Schema(
        columns=[("a", ["String", "str"]), ("B", ["int", "Int64"]), ("C", ["float", "Float64"])]
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, case_sensitive_colnames=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema, complete=True, in_order=False, case_sensitive_colnames=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema, complete=False, in_order=True, case_sensitive_colnames=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema, complete=False, in_order=False, case_sensitive_colnames=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Completely correct schema supplied to `columns=` except for the case mismatch in dtypes
    schema = Schema(
        columns=[("a", ["string", "STR"]), ("b", ["INT64", "INT"]), ("c", ["FloaT64", "float"])]
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, case_sensitive_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=True, in_order=False, case_sensitive_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=True, case_sensitive_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema, complete=False, in_order=False, case_sensitive_dtypes=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Completely correct schema supplied to `columns=` except for the case mismatch in
    # colnames and dtypes
    schema = Schema(
        columns=[("A", ["string", "STR"]), ("b", ["INT64", "int"]), ("C", ["FloaT64", "float"])]
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, case_sensitive_colnames=False, case_sensitive_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=True,
            in_order=False,
            case_sensitive_colnames=False,
            case_sensitive_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=True,
            case_sensitive_colnames=False,
            case_sensitive_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=False,
            case_sensitive_colnames=False,
            case_sensitive_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Matching dtypes with substrings in the supplied schema (`full_match_dtypes=False` case)
    schema = Schema(
        columns=[("a", ["Str", "num"]), ("b", ["Int", "string"]), ("c", ["Float64", "real"])]
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=True, in_order=False, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=True, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=False, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Matching dtypes with substrings in the supplied schema (`full_match_dtypes=True` case)
    schema = Schema(
        columns=[("a", ["Str", "St"]), ("b", ["Int", "In"]), ("c", ["Float64", "Floa"])]
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, full_match_dtypes=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=True, in_order=False, full_match_dtypes=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=True, full_match_dtypes=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=False, full_match_dtypes=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Matching dtypes with substrings in the supplied schema and using case-insensitive matching
    schema = Schema(
        columns=[("a", ["str", "s"]), ("b", ["Int", "num"]), ("c", ["float64", "float80"])]
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, case_sensitive_dtypes=False, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=True,
            in_order=False,
            case_sensitive_dtypes=False,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=True,
            case_sensitive_dtypes=False,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=False,
            case_sensitive_dtypes=False,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Matching dtypes with substrings in the supplied schema and using case-insensitive matching
    # (`case_sensitive_dtypes=True` case)
    schema = Schema(
        columns=[("a", ["str", "str2"]), ("b", ["Int", "Inte"]), ("c", "float64", "float")]
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, case_sensitive_dtypes=True, full_match_dtypes=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=True,
            in_order=False,
            case_sensitive_dtypes=True,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=True,
            case_sensitive_dtypes=True,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema,
            complete=False,
            in_order=False,
            case_sensitive_dtypes=True,
            full_match_dtypes=False,
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )


def test_col_schema_match_columns_only():

    tbl = pl.DataFrame(
        {
            "a": ["apple", "banana", "cherry", "date"],
            "b": [1, 6, 3, 5],
            "c": [1.1, 2.2, 3.3, 4.4],
        }
    )

    # Completely correct schema supplied to `columns=` as a list of strings
    schema = Schema(columns=["a", "b", "c"])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Completely correct schema supplied to `columns=` as a list of 1-element tuples
    schema = Schema(columns=[("a",), ("b",), ("c",)])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Schema columns expressed in a different order (yet complete)
    schema = Schema(columns=["b", "c", "a"])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Schema columns expressed in a different order (yet complete) - wrong column name
    schema = Schema(columns=["b", "c", "wrong"])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Schema of columns has a duplicate column
    schema = Schema(columns=["a", "a", "b", "c"])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Schema columns has duplicate column and a wrong column name
    schema = Schema(columns=["a", "a", "wrong", "c"])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Supplied columns are a subset of the actual columns (but in the correct order)
    schema = Schema(columns=["b", "c"])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Supplied columns are a subset of the actual column (in correct order) - has wrong column name
    schema = Schema(columns=["wrong", "c"])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Supplied columns are a subset of the actual schema but in a different order
    schema = Schema(columns=["c", "b"])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Supplied columns are a subset of actual columns but in a different order - wrong column name
    schema = Schema(columns=["wrong", "b"])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=True, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, in_order=False, complete=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )

    # Completely correct column names except for case mismatches
    schema = Schema(columns=["a", "B", "C"])
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, case_sensitive_colnames=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema, complete=True, in_order=False, case_sensitive_colnames=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema, complete=False, in_order=True, case_sensitive_colnames=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(
            schema=schema, complete=False, in_order=False, case_sensitive_colnames=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Single (but correct) column supplied to `columns=` as a string
    schema = Schema(columns="a")
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=True, in_order=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )

    # Single (but correct) column supplied to `columns=` as a tuple within a list
    schema = Schema(columns=[("a",)])
    assert (
        Validate(data=tbl).col_schema_match(schema=schema).interrogate().n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=True, in_order=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(data=tbl)
        .col_schema_match(schema=schema, complete=False, in_order=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_validation_with_selector_helper_functions(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Create a large validation plan and interrogate the input table
    v = (
        Validate(tbl)
        .col_vals_gt(columns=col("low_numbers"), value=0)  # 1
        .col_vals_lt(columns=col(ends_with("NUMBERS")), value=200000)  # 2 & 3
        .col_vals_between(
            columns=col(ends_with("FLOATS") - contains("superhigh")), left=0, right=100
        )  # 4 & 5
        .col_vals_ge(columns=col(ends_with("floats") | matches("num")), value=0)  # 6, 7, 8, 9, 10
        .col_vals_le(
            columns=col(everything() - last_n(3) - first_n(1)), value=4e7
        )  # 11, 12, 13, 14, 15
        .col_vals_in_set(
            columns=col(starts_with("w") & ends_with("d")), set=["apple", "banana"]
        )  # 16
        .col_vals_outside(columns=col(~first_n(1) & ~last_n(7)), left=10, right=15)  # 17
        .col_vals_regex(columns=col("word"), pattern="a")  # 18
        .interrogate()
    )

    # Check the length of the validation plan
    assert len(v.validation_info) == 18

    # Check the assertion type across all validation steps
    assert [v.validation_info[i].assertion_type for i in range(18)] == [
        "col_vals_gt",
        "col_vals_lt",
        "col_vals_lt",
        "col_vals_between",
        "col_vals_between",
        "col_vals_ge",
        "col_vals_ge",
        "col_vals_ge",
        "col_vals_ge",
        "col_vals_ge",
        "col_vals_le",
        "col_vals_le",
        "col_vals_le",
        "col_vals_le",
        "col_vals_le",
        "col_vals_in_set",
        "col_vals_outside",
        "col_vals_regex",
    ]

    # Check column names across all validation steps
    assert [v.validation_info[i].column for i in range(18)] == [
        "low_numbers",
        "low_numbers",
        "high_numbers",
        "low_floats",
        "high_floats",
        "low_numbers",
        "high_numbers",
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "low_numbers",
        "high_numbers",
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "word",
        "low_numbers",
        "word",
    ]

    # Check values across all validation steps
    assert [v.validation_info[i].values for i in range(18)] == [
        0,
        200000,
        200000,
        (0, 100),
        (0, 100),
        0,
        0,
        0,
        0,
        0,
        4e7,
        4e7,
        4e7,
        4e7,
        4e7,
        ["apple", "banana"],
        (10, 15),
        "a",
    ]

    # Check that all validation steps are active
    assert [v.validation_info[i].active for i in range(18)] == [True] * 18

    # Check that all validation steps have no evaluation errors
    assert [v.validation_info[i].eval_error for i in range(18)] == [None] * 18

    # Check that all validation steps have passed
    assert [v.validation_info[i].all_passed for i in range(18)] == [True] * 18

    # Check that all test unit counts and passing counts are correct (2)
    assert [v.validation_info[i].n for i in range(18)] == [2] * 18
    assert [v.validation_info[i].n_passed for i in range(18)] == [2] * 18


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_validation_with_single_selectors(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Use `starts_with()` selector

    v = Validate(tbl).col_vals_gt(columns=starts_with("low"), value=0).interrogate()
    assert len(v.validation_info) == 2
    assert v.validation_info[0].eval_error is None
    assert v.validation_info[1].eval_error is None
    assert v.validation_info[0].all_passed is True
    assert v.validation_info[1].all_passed is True
    assert [v.validation_info[i].column for i in range(2)] == ["low_numbers", "low_floats"]

    # Use `ends_with()` selector

    v = Validate(tbl).col_vals_gt(columns=ends_with("floats"), value=0).interrogate()
    assert len(v.validation_info) == 3
    assert v.validation_info[0].eval_error is None
    assert v.validation_info[1].eval_error is None
    assert v.validation_info[2].eval_error is None
    assert v.validation_info[0].all_passed is True
    assert v.validation_info[1].all_passed is True
    assert v.validation_info[2].all_passed is True
    assert [v.validation_info[i].column for i in range(2)] == ["low_floats", "high_floats"]

    # Use `ends_with()` selector

    v = Validate(tbl).col_vals_gt(columns=ends_with("floats"), value=0).interrogate()
    assert len(v.validation_info) == 3
    assert v.validation_info[0].eval_error is None
    assert v.validation_info[1].eval_error is None
    assert v.validation_info[2].eval_error is None
    assert v.validation_info[0].all_passed is True
    assert v.validation_info[1].all_passed is True
    assert v.validation_info[2].all_passed is True
    assert [v.validation_info[i].column for i in range(2)] == ["low_floats", "high_floats"]

    # Use `contains()` selector

    v = Validate(tbl).col_vals_gt(columns=contains("numbers"), value=0).interrogate()
    assert len(v.validation_info) == 2
    assert v.validation_info[0].eval_error is None
    assert v.validation_info[1].eval_error is None
    assert v.validation_info[0].all_passed is True
    assert v.validation_info[1].all_passed is True
    assert [v.validation_info[i].column for i in range(2)] == ["low_numbers", "high_numbers"]

    # Use `matches()` selector

    v = Validate(tbl).col_vals_gt(columns=matches("_"), value=0).interrogate()
    assert len(v.validation_info) == 5
    for i in range(5):
        assert v.validation_info[i].eval_error is None
        assert v.validation_info[i].n == 2
        assert v.validation_info[i].n_passed == 2
        assert v.validation_info[i].active is True
        assert v.validation_info[i].assertion_type == "col_vals_gt"
    assert [v.validation_info[i].column for i in range(5)] == [
        "low_numbers",
        "high_numbers",
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]

    # Use `everything()` selector

    v = Validate(tbl).col_exists(columns=everything()).interrogate()
    assert len(v.validation_info) == 9
    for i in range(9):
        assert v.validation_info[i].eval_error is None
        assert v.validation_info[i].n == 1
        assert v.validation_info[i].n_passed == 1
        assert v.validation_info[i].assertion_type == "col_exists"
    assert [v.validation_info[i].column for i in range(9)] == [
        "word",
        "low_numbers",
        "high_numbers",
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    # Use `first_n()` selector

    v = Validate(tbl).col_vals_in_set(columns=first_n(1), set=["apple", "banana"]).interrogate()
    assert len(v.validation_info) == 1
    assert v.validation_info[0].column == "word"
    assert v.validation_info[0].eval_error is None
    assert v.validation_info[0].n == 2
    assert v.validation_info[0].n_passed == 2

    # Use `last_n()` selector

    v = Validate(tbl).col_vals_ge(columns=last_n(1, offset=3), value=1000).interrogate()

    assert len(v.validation_info) == 1
    assert v.validation_info[0].column == "superhigh_floats"
    assert v.validation_info[0].eval_error is None
    assert v.validation_info[0].n == 2
    assert v.validation_info[0].n_passed == 2


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_validation_with_single_selectors_across_validations(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # `col_vals_gt()`

    v_col = Validate(tbl).col_vals_gt(columns=col("low_numbers"), value=0).interrogate()
    v_sel = Validate(tbl).col_vals_gt(columns=starts_with("low"), value=0).interrogate()
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 2

    # `col_vals_lt()`

    v_col = Validate(tbl).col_vals_lt(columns=col("low_numbers"), value=200000).interrogate()
    v_sel = Validate(tbl).col_vals_lt(columns=starts_with("low"), value=200000).interrogate()
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 2

    # `col_vals_ge()`

    v_col = Validate(tbl).col_vals_ge(columns=col("low_numbers"), value=0).interrogate()
    v_sel = Validate(tbl).col_vals_ge(columns=starts_with("low"), value=0).interrogate()
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 2

    # `col_vals_le()`

    v_col = Validate(tbl).col_vals_le(columns=col("low_numbers"), value=200000).interrogate()
    v_sel = Validate(tbl).col_vals_le(columns=starts_with("low"), value=200000).interrogate()
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 2

    # `col_vals_eq()`

    v_col = Validate(tbl).col_vals_eq(columns=col("low_numbers"), value=0).interrogate()
    v_sel = Validate(tbl).col_vals_eq(columns=starts_with("low"), value=0).interrogate()
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 2

    # `col_vals_ne()`

    v_col = Validate(tbl).col_vals_ne(columns=col("low_numbers"), value=0).interrogate()
    v_sel = Validate(tbl).col_vals_ne(columns=starts_with("low"), value=0).interrogate()
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 2

    # `col_vals_between()`

    v_col = (
        Validate(tbl)
        .col_vals_between(columns=col("low_numbers"), left=0, right=200000)
        .interrogate()
    )
    v_sel = (
        Validate(tbl)
        .col_vals_between(columns=starts_with("low"), left=0, right=200000)
        .interrogate()
    )
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 2

    # `col_vals_outside()`

    v_col = (
        Validate(tbl)
        .col_vals_outside(columns=col("low_numbers"), left=0, right=200000)
        .interrogate()
    )
    v_sel = (
        Validate(tbl)
        .col_vals_outside(columns=starts_with("low"), left=0, right=200000)
        .interrogate()
    )
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 2

    # `col_vals_in_set()`

    v_col = (
        Validate(tbl).col_vals_in_set(columns=col("word"), set=["apple", "banana"]).interrogate()
    )
    v_sel = (
        Validate(tbl)
        .col_vals_in_set(columns=starts_with("w"), set=["apple", "banana"])
        .interrogate()
    )
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 1

    # `col_vals_not_in_set()`

    v_col = (
        Validate(tbl)
        .col_vals_not_in_set(columns=col("word"), set=["apple", "banana"])
        .interrogate()
    )
    v_sel = (
        Validate(tbl)
        .col_vals_not_in_set(columns=starts_with("w"), set=["apple", "banana"])
        .interrogate()
    )
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 1

    # `col_vals_null()`

    v_col = Validate(tbl).col_vals_null(columns=col("word")).interrogate()
    v_sel = Validate(tbl).col_vals_null(columns=starts_with("w")).interrogate()
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 1

    # `col_vals_not_null()`

    v_col = Validate(tbl).col_vals_not_null(columns=col("word")).interrogate()
    v_sel = Validate(tbl).col_vals_not_null(columns=starts_with("w")).interrogate()
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 1

    # `col_vals_regex()`

    v_col = Validate(tbl).col_vals_regex(columns=col("word"), pattern="a").interrogate()
    v_sel = Validate(tbl).col_vals_regex(columns=starts_with("w"), pattern="a").interrogate()
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 1

    # `col_exists()`

    v_col = Validate(tbl).col_exists(columns=col("word")).interrogate()
    v_sel = Validate(tbl).col_exists(columns=starts_with("w")).interrogate()
    assert len(v_col.validation_info) == 1
    assert len(v_sel.validation_info) == 1


def test_validation_with_selector_helper_functions_using_pre(tbl_pl_variable_names):

    # Create a validation plan and interrogate the input table
    v = (
        Validate(tbl_pl_variable_names)
        .col_vals_gt(
            columns=col(starts_with("higher")),
            value=100,
            pre=lambda df: df.with_columns(
                higher_floats=pl.col("high_floats") * 10,
                even_higher_floats=pl.col("high_floats") * 100,
            ),
        )
        .interrogate()
    )

    # Check the length of the validation plan
    assert len(v.validation_info) == 1

    # Check properties of the validation step
    assert v.validation_info[0].assertion_type == "col_vals_gt"
    assert v.validation_info[0].column == "higher_floats"
    assert v.validation_info[0].values == 100
    assert v.validation_info[0].active is True
    assert v.validation_info[0].eval_error is None
    assert v.validation_info[0].all_passed is True
    assert v.validation_info[0].n == 2
    assert v.validation_info[0].n_passed == 2
    assert v.validation_info[0].pre is not None

    # Create a slightly different validation plan and interrogate the input table; this will:
    # - have two validation steps (matches both new columns produced via `pre=`)
    # - will succeed in the first but not in the second (+ would fail with any of the start columns)
    v = (
        Validate(tbl_pl_variable_names)
        .col_vals_between(
            columns=col(contains("higher")),
            left=100,
            right=1000,
            pre=lambda df: df.with_columns(
                higher_floats=pl.col("high_floats") * 10,
                even_higher_floats=pl.col("high_floats") * 100,
            ),
        )
        .interrogate()
    )

    # Check the length of the validation plan
    assert len(v.validation_info) == 2

    # Check properties of the first (all passing) validation step
    assert v.validation_info[0].assertion_type == "col_vals_between"
    assert v.validation_info[0].column == "higher_floats"
    assert v.validation_info[0].values == (100, 1000)
    assert v.validation_info[0].active is True
    assert v.validation_info[0].eval_error is None
    assert v.validation_info[0].all_passed is True
    assert v.validation_info[0].n == 2
    assert v.validation_info[0].n_passed == 2
    assert v.validation_info[0].pre is not None

    # Check properties of the second (all failing) validation step
    assert v.validation_info[1].assertion_type == "col_vals_between"
    assert v.validation_info[1].column == "even_higher_floats"
    assert v.validation_info[1].values == (100, 1000)
    assert v.validation_info[1].active is True
    assert v.validation_info[1].eval_error is None
    assert v.validation_info[1].all_passed is False
    assert v.validation_info[1].n == 2
    assert v.validation_info[1].n_passed == 0
    assert v.validation_info[1].pre is not None


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_validation_with_selector_helper_functions_no_match(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Create a validation that evaluates with no issues in the first and third steps but has
    # an evaluation failure in the second step because a column selector fails to resolve any
    # table columns
    v = (
        Validate(tbl)
        .col_vals_le(columns="high_floats", value=100)
        .col_vals_gt(columns=col(contains("not_present")), value=10)
        .col_vals_lt(columns="low_numbers", value=5)
        .interrogate()
    )

    # Check the length of the validation plan
    assert len(v.validation_info) == 3

    # Check properties of the first (all passing, okay eval) validation step
    assert v.validation_info[0].assertion_type == "col_vals_le"
    assert v.validation_info[0].column == "high_floats"
    assert v.validation_info[0].values == 100
    assert v.validation_info[0].active is True
    assert v.validation_info[0].eval_error is None
    assert v.validation_info[0].all_passed is True
    assert v.validation_info[0].n == 2
    assert v.validation_info[0].n_passed == 2
    assert v.validation_info[0].pre is None

    # Check properties of the second (eval failure) validation step
    assert v.validation_info[1].assertion_type == "col_vals_gt"
    assert v.validation_info[1].column == "Contains(text='not_present', case_sensitive=False)"
    assert v.validation_info[1].values == 10
    assert v.validation_info[1].active is False
    assert v.validation_info[1].eval_error is True
    assert v.validation_info[1].all_passed is None
    assert v.validation_info[1].n is None
    assert v.validation_info[1].n_passed is None
    assert v.validation_info[1].pre is None

    # Check properties of the third (all passing, okay eval) validation step
    assert v.validation_info[2].assertion_type == "col_vals_lt"
    assert v.validation_info[2].column == "low_numbers"
    assert v.validation_info[2].values == 5
    assert v.validation_info[2].active is True
    assert v.validation_info[2].eval_error is None
    assert v.validation_info[2].all_passed is True
    assert v.validation_info[2].n == 2
    assert v.validation_info[2].n_passed == 2
    assert v.validation_info[2].pre is None


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_validation_with_selector_helper_functions_no_match_snap(request, tbl_fixture, snapshot):

    tbl = request.getfixturevalue(tbl_fixture)

    # Create a validation that evaluates with no issues in the first and third steps but has
    # an evaluation failure in the second step because a column selector fails to resolve any
    # table columns
    v = (
        Validate(tbl, tbl_name="example_table", label="Simple pointblank validation example")
        .col_vals_le(columns="high_floats", value=100)
        .col_vals_gt(columns=col(contains("not_present")), value=10)
        .col_vals_lt(columns="low_numbers", value=5)
        .interrogate()
    )

    html_str = v.get_tabular_report().as_raw_html()

    # Define the regex pattern to match the entire <td> tag with class "gt_sourcenote"
    pattern = r'<tfoot class="gt_sourcenotes">.*?</tfoot>'

    # Use re.sub to remove the tag
    edited_report_html_str = re.sub(pattern, "", html_str, flags=re.DOTALL)

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(edited_report_html_str, "selector_helper_functions_no_match.html")


@pytest.mark.parametrize("tbl_fixture", TBL_DATES_TIMES_TEXT_LIST)
def test_interrogate_first_n(request, tbl_fixture):

    if tbl_fixture not in [
        "tbl_dates_times_text_parquet",
        "tbl_dates_times_text_duckdb",
        "tbl_dates_times_text_sqlite",
    ]:

        tbl = request.getfixturevalue(tbl_fixture)

        validation = (
            Validate(tbl)
            .col_vals_regex(columns="text", pattern=r"^[a-z]{3}")
            .interrogate(get_first_n=2)
        )

        # Expect that the extracts table has 2 entries out of 3 failures
        assert validation.n_failed(i=1, scalar=True) == 3
        assert len(nw.from_native(validation.get_data_extracts(i=1, frame=True)).rows()) == 2
        assert len(nw.from_native(validation.get_data_extracts(i=1, frame=True)).columns) == 4


@pytest.mark.parametrize("tbl_fixture", TBL_DATES_TIMES_TEXT_LIST)
def test_interrogate_sample_n(request, tbl_fixture):

    if tbl_fixture not in [
        "tbl_dates_times_text_parquet",
        "tbl_dates_times_text_duckdb",
        "tbl_dates_times_text_sqlite",
    ]:

        tbl = request.getfixturevalue(tbl_fixture)

        validation = (
            Validate(tbl)
            .col_vals_regex(columns="text", pattern=r"^[a-z]{3}")
            .interrogate(sample_n=2)
        )

        # Expect that the extracts table has 2 entries out of 3 failures
        assert validation.n_failed(i=1, scalar=True) == 3
        assert len(nw.from_native(validation.get_data_extracts(i=1, frame=True)).rows()) == 2
        assert len(nw.from_native(validation.get_data_extracts(i=1, frame=True)).columns) == 4


@pytest.mark.parametrize(
    "tbl_fixture, sample_frac, expected",
    [
        ("tbl_dates_times_text_pd", 0, 0),
        # ("tbl_dates_times_text_pd", 0.20, 1), # sampling is different in Pandas DFs
        ("tbl_dates_times_text_pd", 0.35, 1),
        # ("tbl_dates_times_text_pd", 0.50, 2), # sampling is different in Pandas DFs
        ("tbl_dates_times_text_pd", 0.75, 2),
        ("tbl_dates_times_text_pd", 1.00, 3),
        ("tbl_dates_times_text_pl", 0, 0),
        ("tbl_dates_times_text_pl", 0.20, 0),
        ("tbl_dates_times_text_pl", 0.35, 1),
        ("tbl_dates_times_text_pl", 0.50, 1),
        ("tbl_dates_times_text_pl", 0.75, 2),
        ("tbl_dates_times_text_pl", 1.00, 3),
    ],
)
def test_interrogate_sample_frac(request, tbl_fixture, sample_frac, expected):

    tbl = request.getfixturevalue(tbl_fixture)

    validation = (
        Validate(tbl)
        .col_vals_regex(columns="text", pattern=r"^[a-z]{3}")
        .interrogate(sample_frac=sample_frac)
    )

    # Expect that the extracts table has 2 entries out of 3 failures
    assert validation.n_failed(i=1, scalar=True) == 3
    assert len(nw.from_native(validation.get_data_extracts(i=1, frame=True)).rows()) == expected
    assert len(nw.from_native(validation.get_data_extracts(i=1, frame=True)).columns) == 4


@pytest.mark.parametrize("tbl_fixture", ["tbl_dates_times_text_pd", "tbl_dates_times_text_pl"])
def test_interrogate_sample_frac_with_sample_limit(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation = (
        Validate(tbl)
        .col_vals_regex(columns="text", pattern=r"^[a-z]{3}")
        .interrogate(sample_frac=0.8, sample_limit=1)
    )

    # Expect that the extracts table has 2 entries out of 3 failures
    assert validation.n_failed(i=1, scalar=True) == 3
    assert len(nw.from_native(validation.get_data_extracts(i=1, frame=True)).rows()) == 1
    assert len(nw.from_native(validation.get_data_extracts(i=1, frame=True)).columns) == 4


@pytest.mark.parametrize("tbl_fixture", TBL_DATES_TIMES_TEXT_LIST)
def test_col_vals_null(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_vals_null(columns="text").interrogate().n_passed(i=1, scalar=True) == 1


@pytest.mark.parametrize("tbl_fixture", TBL_DATES_TIMES_TEXT_LIST)
def test_col_vals_not_null(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl).col_vals_not_null(columns="text").interrogate().n_passed(i=1, scalar=True)
        == 2
    )


@pytest.mark.parametrize("tbl_fixture", TBL_DATES_TIMES_TEXT_LIST)
def test_col_exists(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    assert Validate(tbl).col_exists(columns="text").interrogate().n_passed(i=1, scalar=True) == 1
    assert Validate(tbl).col_exists(columns="invalid").interrogate().n_passed(i=1, scalar=True) == 0


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_validation_types(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation = Validate(tbl).col_vals_gt(columns="x", value=0).interrogate()

    # Check that the `validation` object is a Validate object
    assert isinstance(validation, Validate)

    # Check that using the `get_tabular_report()` returns a GT object
    assert isinstance(validation.get_tabular_report(), GT.GT)


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_interrogate_raise_on_get_first_and_sample(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="z", value=10).interrogate(get_first_n=2, sample_n=4)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="z", value=10).interrogate(get_first_n=2, sample_frac=0.5)
    with pytest.raises(ValueError):
        Validate(tbl).col_vals_gt(columns="z", value=10).interrogate(sample_n=2, sample_frac=0.5)


def test_get_data_extracts(tbl_missing_pd):

    validation = (
        Validate(tbl_missing_pd)
        .col_vals_gt(columns="x", value=1)
        .col_vals_lt(columns="y", value=10)
        .interrogate()
    )

    extracts_all = validation.get_data_extracts()
    extracts_1 = validation.get_data_extracts(i=1)
    extracts_2 = validation.get_data_extracts(i=2)

    assert isinstance(extracts_all, dict)
    assert isinstance(extracts_1, dict)
    assert isinstance(extracts_2, dict)
    assert len(extracts_all) == 2
    assert len(extracts_1) == 1
    assert len(extracts_2) == 1

    extracts_1_df = validation.get_data_extracts(i=1, frame=True)
    extracts_2_df = validation.get_data_extracts(i=2, frame=True)

    assert isinstance(extracts_1_df, pd.DataFrame)
    assert isinstance(extracts_2_df, pd.DataFrame)


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_interrogate_with_active_inactive(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation = (
        Validate(tbl)
        .col_vals_eq(columns="z", value=8)
        .col_vals_lt(columns="y", value=10, active=False)
        .interrogate()
    )

    assert validation.validation_info[0].active is True
    assert validation.validation_info[1].active is False
    assert validation.validation_info[0].proc_duration_s is not None
    assert validation.validation_info[1].proc_duration_s is not None
    assert validation.validation_info[0].time_processed is not None
    assert validation.validation_info[1].time_processed is not None
    assert validation.validation_info[0].all_passed is True
    assert validation.validation_info[1].all_passed is None
    assert validation.validation_info[0].n == 4
    assert validation.validation_info[1].n is None
    assert validation.validation_info[0].n_passed == 4
    assert validation.validation_info[1].n_passed is None
    assert validation.validation_info[0].n_failed == 0
    assert validation.validation_info[1].n_failed is None
    assert validation.validation_info[0].warning is None
    assert validation.validation_info[1].warning is None
    assert validation.validation_info[0].error is None
    assert validation.validation_info[1].error is None
    assert validation.validation_info[0].critical is None
    assert validation.validation_info[1].critical is None
    assert validation.validation_info[1].extract is None
    assert validation.validation_info[1].extract is None


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl"])
def test_get_sundered_data(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # This validation will:
    # - pass completely for all rows in the `col_vals_eq()` step
    # - fail for row 0 in `col_vals_gt()` step
    # - fail for row 3 in `col_vals_lt()` step
    # when error rows are considered across all steps, only rows 1 and 2 free of errors;
    # an 'error row' is a row with a test unit that has failed in any of the row-based steps
    # and all of the validation steps here are row-based
    validation = (
        Validate(tbl)
        .col_vals_eq(columns="z", value=8)
        .col_vals_gt(columns="y", value=4)
        .col_vals_lt(columns="x", value=4)
        .interrogate()
    )

    sundered_data_pass = validation.get_sundered_data(type="pass")  # this is the default
    sundered_data_fail = validation.get_sundered_data(type="fail")

    assert isinstance(sundered_data_pass, pd.DataFrame if tbl_fixture == "tbl_pd" else pl.DataFrame)
    assert isinstance(sundered_data_fail, pd.DataFrame if tbl_fixture == "tbl_pd" else pl.DataFrame)

    # Check properties of the passed data piece
    assert len(nw.from_native(sundered_data_pass).rows()) == 2
    assert len(nw.from_native(sundered_data_pass).columns) == 3
    assert nw.from_native(sundered_data_pass).columns == ["x", "y", "z"]

    # Check the rows of the passed data piece
    passed_data_rows = nw.from_native(sundered_data_pass).rows()
    assert passed_data_rows[0] == (2, 5, 8)
    assert passed_data_rows[1] == (3, 6, 8)

    # Check properties of the failed data piece
    assert len(nw.from_native(sundered_data_fail).rows()) == 2
    assert len(nw.from_native(sundered_data_fail).columns) == 3
    assert nw.from_native(sundered_data_fail).columns == ["x", "y", "z"]

    # Check the rows of the failed data piece
    failed_data_rows = nw.from_native(sundered_data_fail).rows()
    assert failed_data_rows[0] == (1, 4, 8)
    assert failed_data_rows[1] == (4, 7, 8)


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl"])
def test_get_sundered_data_empty_frame(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # Remove all rows from the table
    tbl = tbl.head(0)

    validation = Validate(tbl).col_exists(columns="z").interrogate()

    sundered_data_pass = validation.get_sundered_data(type="pass")
    sundered_data_fail = validation.get_sundered_data(type="fail")

    # Check properties of the passed data piece
    assert len(nw.from_native(sundered_data_pass).rows()) == 0
    assert len(nw.from_native(sundered_data_pass).columns) == 3
    assert nw.from_native(sundered_data_pass).columns == ["x", "y", "z"]

    # Check properties of the failed data piece
    assert len(nw.from_native(sundered_data_fail).rows()) == 0
    assert len(nw.from_native(sundered_data_fail).columns) == 3
    assert nw.from_native(sundered_data_fail).columns == ["x", "y", "z"]


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl"])
def test_get_sundered_data_no_validation_steps(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    validation = Validate(tbl).interrogate()

    sundered_data_pass = validation.get_sundered_data(type="pass")
    sundered_data_fail = validation.get_sundered_data(type="fail")

    # Check properties of the passed data piece
    assert len(nw.from_native(sundered_data_pass).rows()) == 4
    assert len(nw.from_native(sundered_data_pass).columns) == 3
    assert nw.from_native(sundered_data_pass).columns == ["x", "y", "z"]

    # Check properties of the failed data piece
    assert len(nw.from_native(sundered_data_fail).rows()) == 0
    assert len(nw.from_native(sundered_data_fail).columns) == 3
    assert nw.from_native(sundered_data_fail).columns == ["x", "y", "z"]


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl"])
def test_get_sundered_data_mix_of_step_types(request, tbl_fixture):

    tbl = request.getfixturevalue(tbl_fixture)

    # This sundering from this validation will effectively be the same as in the
    # `test_get_sundered_data()` test; steps 3 and 4 are not included in the sundering process:
    # - step 3 is not included because it is not row-based (it checks for a column's existence)
    # - step 4 is not included because it is inactive (if active, it would have failed all rows)
    # - the remaining steps are row-based the parameters of the steps are the same as in the
    #   `test_get_sundered_data()` test
    validation = (
        Validate(tbl)
        .col_vals_eq(columns="z", value=8)
        .col_vals_gt(columns="y", value=4)
        .col_exists(columns="z")  # <- this step is not row-based so not included when sundering
        .col_vals_eq(columns="z", value=7, active=False)  # <- this step is inactive so not included
        .col_vals_lt(columns="x", value=4)
        .interrogate()
    )

    sundered_data_pass = validation.get_sundered_data(type="pass")
    sundered_data_fail = validation.get_sundered_data(type="fail")

    # Check properties of the passed data piece
    assert isinstance(sundered_data_pass, pd.DataFrame if tbl_fixture == "tbl_pd" else pl.DataFrame)
    assert isinstance(sundered_data_fail, pd.DataFrame if tbl_fixture == "tbl_pd" else pl.DataFrame)

    # Check properties of the passed data piece
    assert len(nw.from_native(sundered_data_pass).rows()) == 2
    assert len(nw.from_native(sundered_data_pass).columns) == 3
    assert nw.from_native(sundered_data_pass).columns == ["x", "y", "z"]

    # Check the rows of the passed data piece
    passed_data_rows = nw.from_native(sundered_data_pass).rows()
    assert passed_data_rows[0] == (2, 5, 8)
    assert passed_data_rows[1] == (3, 6, 8)

    # Check properties of the failed data piece
    assert len(nw.from_native(sundered_data_fail).rows()) == 2
    assert len(nw.from_native(sundered_data_fail).columns) == 3
    assert nw.from_native(sundered_data_fail).columns == ["x", "y", "z"]

    # Check the rows of the failed data piece
    failed_data_rows = nw.from_native(sundered_data_fail).rows()
    assert failed_data_rows[0] == (1, 4, 8)
    assert failed_data_rows[1] == (4, 7, 8)


def test_comprehensive_validation_report_html_snap(snapshot):

    validation = (
        Validate(
            data=load_dataset(),
            tbl_name="small_table",
            label="Simple pointblank validation example",
            thresholds=Thresholds(warning=0.10, error=0.25, critical=0.35),
        )
        .col_vals_gt(columns="d", value=100)
        .col_vals_lt(columns="c", value=5)
        .col_vals_eq(columns="a", value=3)
        .col_vals_ne(columns="c", value=10, na_pass=True)
        .col_vals_le(columns="a", value=7)
        .col_vals_ge(columns="d", value=500, na_pass=True)
        .col_vals_between(columns="c", left=0, right=10, na_pass=True)
        .col_vals_outside(columns="a", left=8, right=9, inclusive=(False, True))
        .col_vals_eq(columns="a", value=10, active=False)
        .col_vals_ge(columns="a", value=20, pre=lambda dfn: dfn.with_columns(nw.col("a") * 20))
        .col_vals_gt(
            columns="new", value=20, pre=lambda dfn: dfn.with_columns(new=nw.col("a") * 15)
        )
        .col_vals_in_set(columns="f", set=["low", "mid", "high"])
        .col_vals_not_in_set(columns="f", set=["l", "h", "m"])
        .col_vals_null(columns="c")
        .col_vals_not_null(columns="date_time")
        .col_vals_regex(columns="b", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}")
        .col_exists(columns="z")
        .col_schema_match(schema=Schema(columns=[("a", "Int64")]), complete=False, in_order=False)
        .row_count_match(count=13)
        .row_count_match(count=2, inverse=True)
        .col_count_match(count=8)
        .col_count_match(count=2, inverse=True)
        .rows_distinct()
        .rows_distinct(columns_subset=["a", "b", "c"])
        .col_vals_expr(expr=pl.col("d") > pl.col("a"))
        .interrogate()
    )

    html_str = validation.get_tabular_report().as_raw_html()

    # Define the regex pattern to match the entire <td> tag with class "gt_sourcenote"
    pattern = r'<tfoot class="gt_sourcenotes">.*?</tfoot>'

    # Use re.sub to remove the tag
    edited_report_html_str = re.sub(pattern, "", html_str, flags=re.DOTALL)

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(edited_report_html_str, "comprehensive_validation_report.html")


def test_no_interrogation_validation_report_html_snap(snapshot):

    validation = (
        Validate(
            data=load_dataset(),
            tbl_name="small_table",
            label="Simple pointblank validation example",
            thresholds=Thresholds(warning=0.10, error=0.25, critical=0.35),
        )
        .col_vals_gt(columns="d", value=100)
        .col_vals_lt(columns="c", value=5)
        .col_vals_eq(columns="a", value=3)
        .col_vals_ne(columns="c", value=10, na_pass=True)
        .col_vals_le(columns="a", value=7)
        .col_vals_ge(columns="d", value=500, na_pass=True)
        .col_vals_between(columns="c", left=0, right=10, na_pass=True)
        .col_vals_outside(columns="a", left=8, right=9, inclusive=(False, True))
        .col_vals_eq(columns="a", value=10, active=False)
        .col_vals_ge(columns="a", value=20, pre=lambda dfn: dfn.with_columns(nw.col("a") * 20))
        .col_vals_gt(
            columns="new", value=20, pre=lambda dfn: dfn.with_columns(new=nw.col("a") * 15)
        )
        .col_vals_in_set(columns="f", set=["low", "mid", "high"])
        .col_vals_not_in_set(columns="f", set=["l", "h", "m"])
        .col_vals_null(columns="c")
        .col_vals_not_null(columns="date_time")
        .col_vals_regex(columns="b", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}")
        .col_exists(columns="z")
    )

    html_str = validation.get_tabular_report().as_raw_html()

    # Define the regex pattern to match the entire <td> tag with class "gt_sourcenote"
    pattern = r'<tfoot class="gt_sourcenotes">.*?</tfoot>'

    # Use re.sub to remove the tag
    edited_report_html_str = re.sub(pattern, "", html_str, flags=re.DOTALL)

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(edited_report_html_str, "no_interrogation_validation_report.html")


def test_no_steps_validation_report_html_snap(snapshot):

    validation = Validate(
        data=load_dataset(),
        tbl_name="small_table",
        thresholds=Thresholds(warning=0.10, error=0.25, critical=0.35),
    )

    html_str = validation.get_tabular_report().as_raw_html()

    # Use the snapshot fixture to create and save the snapshot
    snapshot.assert_match(html_str, "no_steps_validation_report.html")


def test_no_steps_validation_report_html_with_interrogate():

    validation = Validate(
        data=load_dataset(),
        tbl_name="small_table",
        thresholds=Thresholds(warning=0.10, error=0.25, critical=0.35),
    )

    assert (
        validation.interrogate().get_tabular_report().as_raw_html()
        == validation.get_tabular_report().as_raw_html()
    )


def test_load_dataset():

    # Load the default dataset (`small_table`) and verify it's a Polars DataFrame
    tbl = load_dataset()
    assert isinstance(tbl, pl.DataFrame)

    # Load the default dataset (`small_table`) and verify it's a Pandas DataFrame
    tbl = load_dataset(tbl_type="pandas")
    assert isinstance(tbl, pd.DataFrame)

    # Load the `game_revenue` dataset and verify it's a Polars DataFrame
    tbl = load_dataset(dataset="game_revenue")
    assert isinstance(tbl, pl.DataFrame)

    # Load the `game_revenue` dataset and verify it's a Pandas DataFrame
    tbl = load_dataset(dataset="game_revenue", tbl_type="pandas")
    assert isinstance(tbl, pd.DataFrame)

    # Load the `nycflights` dataset and verify it's a Polars DataFrame
    tbl = load_dataset(dataset="nycflights")
    assert isinstance(tbl, pl.DataFrame)

    # Load the `nycflights` dataset and verify it's a Pandas DataFrame
    tbl = load_dataset(dataset="nycflights", tbl_type="pandas")
    assert isinstance(tbl, pd.DataFrame)


def test_load_dataset_invalid():

    # A ValueError is raised when an invalid dataset name is provided
    with pytest.raises(ValueError):
        load_dataset(dataset="invalid_dataset")

    # A ValueError is raised when an invalid table type is provided
    with pytest.raises(ValueError):
        load_dataset(tbl_type="invalid_tbl_type")


def test_load_dataset_no_pandas():

    # Mock the absence of the pandas library
    with patch.dict(sys.modules, {"pandas": None}):
        # A ValueError is raised when `tbl_type="pandas"` and the `pandas` package is not installed
        with pytest.raises(ImportError):
            load_dataset(tbl_type="pandas")


def test_load_dataset_no_polars():

    # Mock the absence of the polars library
    with patch.dict(sys.modules, {"polars": None}):
        # A ValueError is raised when `tbl_type="pandas"` and the `pandas` package is not installed
        with pytest.raises(ImportError):
            load_dataset(tbl_type="polars")


def test_process_title_text():

    assert _process_title_text(title=None, tbl_name=None) == ""
    assert _process_title_text(title=":default:", tbl_name=None) == _get_default_title_text()
    assert _process_title_text(title=":none:", tbl_name=None) == ""
    assert _process_title_text(title=":tbl_name:", tbl_name="tbl_name") == "<code>tbl_name</code>"
    assert _process_title_text(title=":tbl_name:", tbl_name=None) == ""
    assert _process_title_text(title="*Title*", tbl_name=None) == "<p><em>Title</em></p>\n"


@pytest.mark.parametrize(
    "input_value, expected_output",
    [
        (0, "0"),
        (1, "1.00"),
        (5, "5.00"),
        (10, "10.0"),
        (100, "100"),
        (999, "999"),
        (1000, "1.00K"),
        (10000, "10.0K"),
        (100000, "100K"),
        (999999, "1,000K"),
        (1000000, "1.00M"),
        (10000000, "10.0M"),
        (100000000, "100M"),
        (999999999, "1,000M"),
        (1000000000, "1.00B"),
        (10000000000, "10.0B"),
        (100000000000, "100B"),
    ],
)
def test_fmt_lg(input_value, expected_output):
    assert _fmt_lg(input_value) == expected_output


def test_create_table_time_html():

    datetime_0 = datetime(2021, 1, 1, 0, 0, 0, 0)
    datetime_1_min_later = datetime(2021, 1, 1, 0, 1, 0, 0)

    assert _create_table_time_html(time_start=None, time_end=None) == ""
    assert "div" in _create_table_time_html(time_start=datetime_0, time_end=datetime_1_min_later)


def test_create_table_type_html():

    # def _create_table_type_html(tbl_type: str | None, tbl_name: str | None)

    assert _create_table_type_html(tbl_type=None, tbl_name="tbl_name") == ""
    assert _create_table_type_html(tbl_type="invalid", tbl_name="tbl_name") == ""
    assert "span" in _create_table_type_html(tbl_type="pandas", tbl_name="tbl_name")
    assert "span" in _create_table_type_html(tbl_type="pandas", tbl_name=None)
    assert _create_table_type_html(
        tbl_type="pandas", tbl_name="tbl_name"
    ) != _create_table_type_html(tbl_type="pandas", tbl_name=None)


def test_pointblank_config_class():

    # Test the default configuration
    config = PointblankConfig()

    assert config.report_incl_header is True
    assert config.report_incl_footer is True
    assert config.preview_incl_header is True

    assert (
        str(config)
        == "PointblankConfig(report_incl_header=True, report_incl_footer=True, preview_incl_header=True)"
    )


def test_preview_no_fail_pd_table():

    small_table = load_dataset(dataset="small_table", tbl_type="pandas")

    preview(small_table)
    preview(small_table, n_head=2)
    preview(small_table, n_tail=2)
    preview(small_table, n_head=2, n_tail=2)


def test_preview_no_fail_pl_table():

    small_table = load_dataset(dataset="small_table", tbl_type="polars")

    preview(small_table)
    preview(small_table, n_head=2)
    preview(small_table, n_tail=2)
    preview(small_table, n_head=2, n_tail=2)


def test_preview_no_fail_duckdb_table():

    small_table = load_dataset(dataset="small_table", tbl_type="duckdb")

    preview(small_table)
    preview(small_table, n_head=2)
    preview(small_table, n_tail=2)
    preview(small_table, n_head=2, n_tail=2)


def test_preview_large_head_tail_pd_table():

    small_table = load_dataset(dataset="small_table", tbl_type="pandas")
    preview(small_table, n_head=10, n_tail=10)


def test_preview_large_head_tail_pl_table():

    small_table = load_dataset(dataset="small_table", tbl_type="polars")
    preview(small_table, n_head=10, n_tail=10)


def test_preview_large_head_tail_duckdb_table():

    small_table = load_dataset(dataset="small_table", tbl_type="duckdb")
    preview(small_table, n_head=10, n_tail=10)


def test_preview_fails_head_tail_exceed_limit():

    small_table = load_dataset(dataset="small_table", tbl_type="pandas")

    with pytest.raises(ValueError):
        preview(small_table, n_head=100, n_tail=100)  # default limit is 50

    preview(small_table, n_head=100, n_tail=100, limit=300)


def test_preview_no_polars_duckdb_table():

    small_table = load_dataset(dataset="small_table", tbl_type="duckdb")

    # Mock the absence of the Polars library, which is the default library for making
    # a table for the preview; this should not raise an error since Pandas is the
    # fallback library and is available
    with patch.dict(sys.modules, {"polars": None}):
        preview(small_table)

    # Mock the absence of the Pandas library, which is a secondary library for making
    # a table for the preview; this should not raise an error since Polars is the default
    # library and is available
    with patch.dict(sys.modules, {"pandas": None}):
        preview(small_table)

    # Mock the absence of both the Polars and Pandas libraries, which are the libraries
    # for making a table for the preview; this should raise an error since there are no
    # libraries available to make a table for the preview
    with patch.dict(sys.modules, {"polars": None, "pandas": None}):
        with pytest.raises(ImportError):
            preview(small_table)


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_preview_with_columns_subset_no_fail(tbl_type):

    tbl = load_dataset(dataset="game_revenue", tbl_type=tbl_type)

    preview(tbl, columns_subset="player_id")
    preview(tbl, columns_subset=["player_id"])
    preview(tbl, columns_subset=["player_id", "item_name", "item_revenue"])
    preview(tbl, columns_subset=col("player_id"))
    preview(tbl, columns_subset=col(matches("player_id")))
    preview(tbl, columns_subset=col(matches("_id")))
    preview(tbl, columns_subset=starts_with("item"))
    preview(tbl, columns_subset=ends_with("revenue"))
    preview(tbl, columns_subset=matches("_id"))
    preview(tbl, columns_subset=contains("_"))
    preview(tbl, columns_subset=everything())
    preview(tbl, columns_subset=col(starts_with("item") | matches("player")))
    preview(tbl, columns_subset=col(first_n(2) | last_n(2)))
    preview(tbl, columns_subset=col(everything() - last_n(2)))
    preview(tbl, columns_subset=col(~first_n(2)))


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_preview_with_columns_subset_failing(tbl_type):

    tbl = load_dataset(dataset="game_revenue", tbl_type=tbl_type)

    with pytest.raises(ValueError):
        preview(tbl, columns_subset="player_id", n_head=100, n_tail=100)
    with pytest.raises(ValueError):
        preview(tbl, columns_subset="fake_id")
    with pytest.raises(ValueError):
        preview(tbl, columns_subset=["fake_id", "item_name", "item_revenue"])
    with pytest.raises(ValueError):
        preview(tbl, columns_subset=col(matches("fake_id")))


def test_missing_vals_tbl_no_fail_pd_table():

    small_table = load_dataset(dataset="small_table", tbl_type="pandas")
    missing_vals_tbl(small_table)

    game_revenue = load_dataset(dataset="game_revenue", tbl_type="pandas")
    missing_vals_tbl(game_revenue)

    nycflights = load_dataset(dataset="nycflights", tbl_type="pandas")
    missing_vals_tbl(nycflights)


def test_missing_vals_tbl_no_fail_pl_table():

    small_table = load_dataset(dataset="small_table", tbl_type="polars")
    missing_vals_tbl(small_table)

    game_revenue = load_dataset(dataset="game_revenue", tbl_type="polars")
    missing_vals_tbl(game_revenue)

    nycflights = load_dataset(dataset="nycflights", tbl_type="polars")
    missing_vals_tbl(nycflights)


def test_missing_vals_tbl_no_fail_duckdb_table():

    small_table = load_dataset(dataset="small_table", tbl_type="duckdb")
    missing_vals_tbl(small_table)

    game_revenue = load_dataset(dataset="game_revenue", tbl_type="duckdb")
    missing_vals_tbl(game_revenue)

    nycflights = load_dataset(dataset="nycflights", tbl_type="duckdb")
    missing_vals_tbl(nycflights)


def test_missing_vals_tbl_no_pandas():

    # Mock the absence of the pandas library
    with patch.dict(sys.modules, {"pandas": None}):

        # The function should not raise an error if a Polars table is provided
        small_table = load_dataset(dataset="small_table", tbl_type="polars")
        missing_vals_tbl(small_table)


def test_missing_vals_tbl_no_polars():

    # Mock the absence of the polars library
    with patch.dict(sys.modules, {"polars": None}):

        # The function should not raise an error if a Pandas table is provided
        small_table = load_dataset(dataset="small_table", tbl_type="pandas")
        missing_vals_tbl(small_table)


def test_missing_vals_tbl_using_ibis_no_pandas():

    # Mock the absence of the pandas library
    with patch.dict(sys.modules, {"pandas": None}):

        # The function should not raise an error if an Ibis backend table is provided
        small_table = load_dataset(dataset="small_table", tbl_type="duckdb")
        missing_vals_tbl(small_table)


def test_missing_vals_tbl_using_ibis_no_polars():

    # Mock the absence of the polars library
    with patch.dict(sys.modules, {"polars": None}):

        # The function should not raise an error if an Ibis backend table is provided
        small_table = load_dataset(dataset="small_table", tbl_type="duckdb")
        missing_vals_tbl(small_table)


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_get_column_count(tbl_type):

    small_table = load_dataset(dataset="small_table", tbl_type=tbl_type)
    game_revenue = load_dataset(dataset="game_revenue", tbl_type=tbl_type)
    nycflights = load_dataset(dataset="nycflights", tbl_type=tbl_type)

    assert get_column_count(small_table) == 8
    assert get_column_count(game_revenue) == 11
    assert get_column_count(nycflights) == 18


def test_get_column_count_failing():

    with pytest.raises(ValueError):
        get_column_count(None)
    with pytest.raises(ValueError):
        get_column_count("not a table")


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_get_row_count(tbl_type):

    small_table = load_dataset(dataset="small_table", tbl_type=tbl_type)
    game_revenue = load_dataset(dataset="game_revenue", tbl_type=tbl_type)
    nycflights = load_dataset(dataset="nycflights", tbl_type=tbl_type)

    assert get_row_count(small_table) == 13
    assert get_row_count(game_revenue) == 2000
    assert get_row_count(nycflights) == 336776


def test_get_row_count_failing():

    with pytest.raises(ValueError):
        get_row_count(None)
    with pytest.raises(ValueError):
        get_row_count("not a table")


def test_get_row_count_no_polars_duckdb_table():

    small_table = load_dataset(dataset="small_table", tbl_type="duckdb")

    # Mock the absence of the Polars library, which is the default library for making
    # a table for the preview; this should not raise an error since Pandas is the
    # fallback library and is available
    with patch.dict(sys.modules, {"polars": None}):
        assert get_row_count(small_table) == 13

    # Mock the absence of the Pandas library, which is a secondary library for making
    # a table for the preview; this should not raise an error since Polars is the default
    # library and is available
    with patch.dict(sys.modules, {"pandas": None}):
        assert get_row_count(small_table) == 13

    # Mock the absence of both the Polars and Pandas libraries, which are the libraries
    # for making a table for the preview; this should raise an error since there are no
    # libraries available to make a table for the preview
    with patch.dict(sys.modules, {"polars": None, "pandas": None}):
        with pytest.raises(ImportError):
            assert get_row_count(small_table) == 13


@pytest.mark.parametrize("tbl_type", ["pandas", "polars"])
def test_get_step_report_no_fail(tbl_type):

    small_table = load_dataset(dataset="small_table", tbl_type=tbl_type)

    validation = (
        Validate(small_table)
        .col_vals_gt(columns="a", value=0)
        .col_vals_lt(columns="a", value=10)
        .col_vals_eq(columns="c", value=8)
        .col_vals_ne(columns="d", value=100)
        .col_vals_le(columns="a", value=6)
        .col_vals_ge(columns="d", value=500)
        .col_vals_between(columns="a", left=2, right=10)
        .col_vals_outside(columns="a", left=7, right=20)
        .col_vals_in_set(columns="f", set=["low", "mid", "high"])
        .col_vals_not_in_set(columns="f", set=["l", "mid", "m"])
        .col_vals_null(columns="b")
        .col_vals_not_null(columns="date_time")
        .col_vals_regex(columns="b", pattern=r"[0-9]-[a-z]{3}-[0-9]{3}")
        .col_schema_match(schema=Schema(columns=[("a", "Int64")]), complete=True, in_order=True)
        .col_schema_match(schema=Schema(columns=[("a", "Int64")]), complete=True, in_order=False)
        .col_schema_match(schema=Schema(columns=[("a", "Int64")]), complete=False, in_order=True)
        .col_schema_match(schema=Schema(columns=[("a", "Int64")]), complete=False, in_order=False)
        .interrogate()
    )

    for i in range(1, 18):
        assert isinstance(validation.get_step_report(i=i), GT.GT)


def test_get_step_report_failing_inputs():

    small_table = load_dataset(dataset="small_table", tbl_type="pandas")

    validation = Validate(small_table).col_vals_gt(columns="a", value=0).interrogate()

    with pytest.raises(ValueError):
        validation.get_step_report(i=0)

    with pytest.raises(ValueError):
        validation.get_step_report(i=2)


def test_get_step_report_inactive_step():

    small_table = load_dataset(dataset="small_table", tbl_type="pandas")

    validation = Validate(small_table).col_vals_gt(columns="a", value=0, active=False).interrogate()

    validation.get_step_report(i=1) == "This validation step is inactive."


def test_get_step_report_non_supported_steps():

    small_table = load_dataset(dataset="small_table", tbl_type="pandas")

    validation = (
        Validate(small_table).rows_distinct().rows_distinct(columns_subset=["a", "b"]).interrogate()
    )

    assert validation.get_step_report(i=1) is None
    assert validation.get_step_report(i=2) is None


@pytest.mark.parametrize(
    "schema",
    [
        Schema(columns=[("a", ["String", "Int64"])]),
        Schema(columns=[("a", ["String", "Int64"]), ("b", "Int64")]),
        Schema(columns=[("a", ["String", "Int64"]), ("b", "Int64"), ("c", "Float64")]),
        Schema(columns=[("a", ["Str", "Int64"])]),
        Schema(columns=[("a", ["String", "Int64"]), ("b", "Int")]),
        Schema(columns=[("a", ["String", "Int64"]), ("b", "Int"), ("c", "Float64")]),
        Schema(columns=[("a", ["String", "Int64"]), ("d", "Float64")]),
        Schema(columns=[("a", ["String", "Int64"]), ("b", "Int64"), ("z", "Float64")]),
        Schema(
            columns=[("a", ["String", "Int64"]), ("b", "Int64"), ("c", "Float64"), ("z", "Float64")]
        ),
    ],
)
def test_get_step_report_schema_checks(schema):

    tbl = pl.DataFrame(
        {
            "a": ["apple", "banana", "cherry", "date"],
            "b": [1, 6, 3, 5],
            "c": [1.1, 2.2, 3.3, 4.4],
        }
    )

    for in_order in [True, False]:
        validation = (
            Validate(data=tbl)
            .col_schema_match(schema=schema, complete=True, in_order=in_order)
            .interrogate()
        )

        assert isinstance(validation.get_step_report(i=1), GT.GT)


def get_schema_info(
    data_tbl,
    schema,
    passed=True,
    complete=True,
    in_order=True,
    case_sensitive_colnames=True,
    case_sensitive_dtypes=True,
    full_match_dtypes=True,
):
    return _get_schema_validation_info(
        data_tbl=data_tbl,
        schema=schema,
        passed=passed,
        complete=complete,
        in_order=in_order,
        case_sensitive_colnames=case_sensitive_colnames,
        case_sensitive_dtypes=case_sensitive_dtypes,
        full_match_dtypes=full_match_dtypes,
    )


def assert_schema_cols(schema_info, expectations):
    (
        expected_columns_found,
        expected_columns_not_found,
        expected_columns_unmatched,
    ) = expectations

    assert (
        schema_info["columns_found"] == expected_columns_found
    ), f"Expected {expected_columns_found}, but got {schema_info['columns_found']}"
    assert (
        schema_info["columns_not_found"] == expected_columns_not_found
    ), f"Expected {expected_columns_not_found}, but got {schema_info['columns_not_found']}"
    assert (
        schema_info["columns_unmatched"] == expected_columns_unmatched
    ), f"Expected {expected_columns_unmatched}, but got {schema_info['columns_unmatched']}"


def assert_col_dtype_match(schema_info, column):
    if column not in schema_info["columns"]:
        assert False
    assert schema_info["columns"][column]["dtype_matched"]


def assert_col_dtype_mismatch(schema_info, column):
    if column not in schema_info["columns"]:
        assert False
    assert not schema_info["columns"][column]["dtype_matched"]


def assert_col_index_match(schema_info, column):
    if column not in schema_info["columns"]:
        assert False
    assert schema_info["columns"][column]["index_matched"]


def assert_col_index_mismatch(schema_info, column):
    if column not in schema_info["columns"]:
        assert False
    assert not schema_info["columns"][column]["index_matched"]


def assert_col_dtype_absent(schema_info, column):
    if column not in schema_info["columns"]:
        assert False
    assert not schema_info["columns"][column]["dtype_present"]


def assert_columns_full_set(schema_info):
    assert schema_info["columns_full_set"]


def assert_columns_subset(schema_info):
    assert schema_info["columns_subset"]


def assert_columns_not_a_set(schema_info):
    assert not schema_info["columns_full_set"] and not schema_info["columns_subset"]


def assert_columns_matched_in_order(schema_info, reverse=False):
    if reverse:
        assert not schema_info["columns_matched_in_order"]
    else:
        assert schema_info["columns_matched_in_order"]
    return


def assert_columns_matched_any_order(schema_info, reverse=False):
    if reverse:
        assert not schema_info["columns_matched_any_order"]
    else:
        assert schema_info["columns_matched_any_order"]
    return


def schema_info_str(schema_info):
    return pprint.pformat(schema_info, sort_dicts=False, width=100)


def test_get_schema_validation_info(tbl_schema_tests, snapshot):

    # Note regarding the input in the `assert_schema_cols()` testing function
    #
    # The main input is a tuple of three lists:
    # - the first list contains the target columns matched to expected columns
    # - the second list contains the target columns not matched by the expected columns
    # - the third list holds the expected columns having no match to the target columns
    #
    # target columns = columns in the data table
    # expected columns = columns in the supplied schema

    # 1. Schema matches completely and in order; dtypes all correct
    schema = Schema(
        columns=[
            ("a", "String"),
            ("b", "Int64"),
            ("c", "Float64"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "b")
    assert_col_dtype_match(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_01-0.txt")

    # 2. Schema matches completely; option taken to match any of two different dtypes for
    # column "a", but all dtypes correct
    schema = Schema(
        columns=[
            ("a", ["String", "Int64"]),
            ("b", "Int64"),
            ("c", "Float64"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "b")
    assert_col_dtype_match(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_02-0.txt")

    # 3. Schema has all three columns accounted for but in an incorrect order; dtypes correct
    schema = Schema(
        columns=[
            ("b", "Int64"),
            ("a", "String"),
            ("c", "Float64"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info)
    assert_col_dtype_match(schema_info, "b")
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "c")
    assert_col_index_mismatch(schema_info, "a")
    assert_col_index_mismatch(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_03-0.txt")

    # 4. Schema has all three columns accounted for but in an incorrect order; option taken to
    # match any of two different dtypes for column "a", but all dtypes correct
    schema = Schema(
        columns=[
            ("b", "Int64"),
            ("a", ["Int64", "String"]),
            ("c", "Float64"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info)
    assert_col_dtype_match(schema_info, "b")
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "c")
    assert_col_index_mismatch(schema_info, "a")
    assert_col_index_mismatch(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_04-0.txt")

    # 5. Schema has all three columns matching, correct order; no dtypes provided
    schema = Schema(
        columns=[
            ("a",),
            ("b",),
            ("c",),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_absent(schema_info, "a")
    assert_col_dtype_absent(schema_info, "b")
    assert_col_dtype_absent(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_05-0.txt")

    # 6. Schema has all three columns matching, correct order; incorrect dtypes
    schema = Schema(
        columns=[
            ("a", ["invalid", "invalid"]),
            ("b", "invalid"),
            ("c", "invalid"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "b")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_06-0.txt")

    # 7. Schema has 2/3 columns matching, correct order; incorrect dtypes
    schema = Schema(
        columns=[
            ("a", ["invalid", "invalid"]),
            ("c", "invalid"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_07-0.txt")

    # 8. Schema has 2/3 columns matching, incorrect order; incorrect dtypes
    schema = Schema(
        columns=[
            ("c", "invalid"),
            ("a", ["invalid", "invalid"]),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_mismatch(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_08-0.txt")

    # 9. Schema has single column match; incorrect dtype
    schema = Schema(
        columns=[
            ("c", "invalid"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["c"], ["a", "b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_mismatch(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_09-0.txt")

    # 10. Schema is empty
    schema = Schema(columns=[])
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, ([], ["a", "b", "c"], []))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_10-0.txt")

    # 11. Schema has complete match of columns plus an additional, unmatched column
    schema = Schema(
        columns=[("a", ["String", "Int64"]), ("b", "Int64"), ("c", "Float64"), ("d", "String")]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], ["d"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "b")
    assert_col_dtype_match(schema_info, "c")
    assert_col_dtype_mismatch(schema_info, "d")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")
    assert_col_index_mismatch(schema_info, "d")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_11-0.txt")

    # 12. Schema has partial match of columns (in right order) plus an additional, unmatched column
    schema = Schema(columns=[("a", ["String", "Int64"]), ("c", "Float64"), ("d", "String")])
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "c"], ["b"], ["d"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "c")
    assert_col_dtype_mismatch(schema_info, "d")
    assert_col_index_match(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")
    assert_col_index_mismatch(schema_info, "d")
    assert_col_index_match(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")
    assert_col_index_mismatch(schema_info, "d")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_12-0.txt")

    # 13. Schema has no matches to any column names
    schema = Schema(
        columns=[
            ("x", "String"),
            ("y", "Int64"),
            ("z", "Float64"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, ([], ["a", "b", "c"], ["x", "y", "z"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_index_mismatch(schema_info, "x")
    assert_col_index_mismatch(schema_info, "y")
    assert_col_index_mismatch(schema_info, "z")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_13-0.txt")

    # 14. Schema has all columns matching in case-insensitive manner, correct order; dtypes
    # all correct
    schema = Schema(
        columns=[
            ("A", "String"),
            ("B", "Int64"),
            ("C", "Float64"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, ([], ["a", "b", "c"], ["A", "B", "C"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "A")
    assert_col_dtype_mismatch(schema_info, "B")
    assert_col_dtype_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "A")
    assert_col_index_mismatch(schema_info, "B")
    assert_col_index_mismatch(schema_info, "C")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_14-0.txt")

    # 14-1. Using `case_sensitive_colnames=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests, schema=schema, case_sensitive_colnames=False
    )
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "A")
    assert_col_dtype_match(schema_info, "B")
    assert_col_dtype_match(schema_info, "C")
    assert_col_index_match(schema_info, "A")
    assert_col_index_match(schema_info, "B")
    assert_col_index_match(schema_info, "C")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_14-1.txt")

    # 15. Schema has all columns matching in case-insensitive manner, correct order; dtypes
    # all correct
    schema = Schema(
        columns=[
            ("B", "Int64"),
            ("A", "String"),
            ("C", "Float64"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, ([], ["a", "b", "c"], ["B", "A", "C"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "B")
    assert_col_dtype_mismatch(schema_info, "A")
    assert_col_dtype_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "B")
    assert_col_index_mismatch(schema_info, "A")
    assert_col_index_mismatch(schema_info, "C")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_15-0.txt")

    # 15-1. Using `case_sensitive_colnames=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests, schema=schema, case_sensitive_colnames=False
    )
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info)
    assert_col_dtype_match(schema_info, "A")
    assert_col_dtype_match(schema_info, "B")
    assert_col_dtype_match(schema_info, "C")
    assert_col_index_mismatch(schema_info, "B")
    assert_col_index_mismatch(schema_info, "A")
    assert_col_index_match(schema_info, "C")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_15-1.txt")

    # 16. Schema has 2/3 columns matching in case-insensitive manner, correct order; dtypes
    # all correct
    schema = Schema(
        columns=[
            ("A", "String"),
            ("C", "Float64"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, ([], ["a", "b", "c"], ["A", "C"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "A")
    assert_col_dtype_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "A")
    assert_col_index_mismatch(schema_info, "C")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_16-0.txt")

    # 16-1. Using `case_sensitive_colnames=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests, schema=schema, case_sensitive_colnames=False
    )
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "A")
    assert_col_dtype_match(schema_info, "C")
    assert_col_index_match(schema_info, "A")
    assert_col_index_mismatch(schema_info, "C")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_16-1.txt")

    # 17. Schema has 2/3 columns matching in case-insensitive manner, incorrect order; dtypes
    # all correct
    schema = Schema(
        columns=[
            ("C", "Float64"),
            ("A", "String"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, ([], ["a", "b", "c"], ["C", "A"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "C")
    assert_col_dtype_mismatch(schema_info, "A")
    assert_col_index_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "A")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_17-0.txt")

    # 17-1. Using `case_sensitive_colnames=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests, schema=schema, case_sensitive_colnames=False
    )
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info)
    assert_col_dtype_match(schema_info, "C")
    assert_col_dtype_match(schema_info, "A")
    assert_col_index_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "A")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_17-1.txt")

    # 18. Schema has one column matching in case-insensitive manner; dtypes is correct
    schema = Schema(
        columns=[
            ("C", "Float64"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, ([], ["a", "b", "c"], ["C"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "C")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_18-0.txt")

    # 18-1. Using `case_sensitive_colnames=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests, schema=schema, case_sensitive_colnames=False
    )
    assert_schema_cols(schema_info, (["c"], ["a", "b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "C")
    assert_col_index_mismatch(schema_info, "C")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_18-1.txt")

    # 19. Schema has all three columns matching, correct order; dtypes don't match case
    # of actual dtypes
    schema = Schema(
        columns=[
            ("a", "string"),
            ("b", "int64"),
            ("c", "float64"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "b")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_19-0.txt")

    # 19-1. Using `case_sensitive_dtypes=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests, schema=schema, case_sensitive_dtypes=False
    )
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "b")
    assert_col_dtype_match(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_19-1.txt")

    # 20. Schema has all three columns matching, correct order; dtypes are substrings
    # of actual dtypes
    schema = Schema(
        columns=[
            ("a", "Str"),
            ("b", "Int"),
            ("c", "Float"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "b")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_20-0.txt")

    # 20-1. Using `full_match_dtypes=False`
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema, full_match_dtypes=False)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "b")
    assert_col_dtype_match(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_20-1.txt")

    # 21. Schema has all three columns matching, correct order; dtypes are substrings of
    # actual dtypes where case doesn't match
    schema = Schema(
        columns=[
            ("a", "str"),
            ("b", "int"),
            ("c", "float"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "b")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_21-0.txt")

    # 21-1. Using `case_sensitive_dtypes=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests, schema=schema, case_sensitive_dtypes=False
    )
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "b")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_21-1.txt")

    # 21-2. Using `full_match_dtypes=False`
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema, full_match_dtypes=False)
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "b")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_21-2.txt")

    # 21-3. Using `case_sensitive_dtypes=False` and `full_match_dtypes=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests,
        schema=schema,
        case_sensitive_dtypes=False,
        full_match_dtypes=False,
    )
    assert_schema_cols(schema_info, (["a", "b", "c"], [], []))
    assert_columns_full_set(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "b")
    assert_col_dtype_match(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_match(schema_info, "b")
    assert_col_index_match(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_21-3.txt")

    # 22. Schema has all 2/3 columns matching, missing one, correct order; dtypes don't match case
    # of actual dtypes
    schema = Schema(
        columns=[
            ("a", "string"),
            ("c", "float64"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_22-0.txt")

    # 22-1. Using `case_sensitive_dtypes=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests, schema=schema, case_sensitive_dtypes=False
    )
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_22-1.txt")

    # 23. Schema has all 2/3 columns matching, missing one, correct order; dtypes are substrings
    # of actual dtypes
    schema = Schema(
        columns=[
            ("a", "Str"),
            ("c", "Float"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_23-0.txt")

    # 23-1. Using `full_match_dtypes=False`
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema, full_match_dtypes=False)
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_23-1.txt")

    # 24. Schema has all 2/3 columns matching, missing one, correct order; dtypes are substrings
    # of actual dtypes where case doesn't match
    schema = Schema(
        columns=[
            ("a", "str"),
            ("c", "float"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_24-0.txt")

    # 24-1. Using `case_sensitive_dtypes=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests, schema=schema, case_sensitive_dtypes=False
    )
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_24-1.txt")

    # 24-2. Using `full_match_dtypes=False`
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema, full_match_dtypes=False)
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "a")
    assert_col_dtype_mismatch(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_24-2.txt")

    # 24-3. Using `case_sensitive_dtypes=False` and `full_match_dtypes=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests,
        schema=schema,
        case_sensitive_dtypes=False,
        full_match_dtypes=False,
    )
    assert_schema_cols(schema_info, (["a", "c"], ["b"], []))
    assert_columns_subset(schema_info)
    assert_columns_matched_in_order(schema_info)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "a")
    assert_col_dtype_match(schema_info, "c")
    assert_col_index_match(schema_info, "a")
    assert_col_index_mismatch(schema_info, "c")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_24-3.txt")

    # 25. Schema has all 2/3 columns matching, missing one, an unmatched column, correct order for
    # the matching set; dtypes are substrings of actual dtypes where case doesn't match
    schema = Schema(
        columns=[
            ("A", "str"),
            ("C", "float"),
            ("X", "int"),
        ]
    )
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema)
    assert_schema_cols(schema_info, ([], ["a", "b", "c"], ["A", "C", "X"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "A")
    assert_col_dtype_mismatch(schema_info, "C")
    assert_col_dtype_mismatch(schema_info, "X")
    assert_col_index_mismatch(schema_info, "A")
    assert_col_index_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "X")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_25-0.txt")

    # 25-1. Using `case_sensitive_colnames=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests, schema=schema, case_sensitive_colnames=False
    )
    assert_schema_cols(schema_info, (["a", "c"], ["b"], ["X"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "A")
    assert_col_dtype_mismatch(schema_info, "C")
    assert_col_dtype_mismatch(schema_info, "X")
    assert_col_index_match(schema_info, "A")
    assert_col_index_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "X")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_25-1.txt")

    # 25-2. Using `case_sensitive_dtypes=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests, schema=schema, case_sensitive_dtypes=False
    )
    assert_schema_cols(schema_info, ([], ["a", "b", "c"], ["A", "C", "X"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "A")
    assert_col_dtype_mismatch(schema_info, "C")
    assert_col_dtype_mismatch(schema_info, "X")
    assert_col_index_mismatch(schema_info, "A")
    assert_col_index_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "X")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_25-2.txt")

    # 25-3. Using `full_match_dtypes=False`
    schema_info = get_schema_info(data_tbl=tbl_schema_tests, schema=schema, full_match_dtypes=False)
    assert_schema_cols(schema_info, ([], ["a", "b", "c"], ["A", "C", "X"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "A")
    assert_col_dtype_mismatch(schema_info, "C")
    assert_col_dtype_mismatch(schema_info, "X")
    assert_col_index_mismatch(schema_info, "A")
    assert_col_index_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "X")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_25-3.txt")

    # 25-4. Using `case_sensitive_colnames=False` and `case_sensitive_dtypes=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests,
        schema=schema,
        case_sensitive_colnames=False,
        case_sensitive_dtypes=False,
    )
    assert_schema_cols(schema_info, (["a", "c"], ["b"], ["X"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_mismatch(schema_info, "A")
    assert_col_dtype_mismatch(schema_info, "C")
    assert_col_dtype_mismatch(schema_info, "X")
    assert_col_index_match(schema_info, "A")
    assert_col_index_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "X")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_25-4.txt")

    # 25-5. Using `case_sensitive_colnames=False`, `case_sensitive_dtypes=False`, and
    # `full_match_dtypes=False`
    schema_info = get_schema_info(
        data_tbl=tbl_schema_tests,
        schema=schema,
        case_sensitive_colnames=False,
        case_sensitive_dtypes=False,
        full_match_dtypes=False,
    )
    assert_schema_cols(schema_info, (["a", "c"], ["b"], ["X"]))
    assert_columns_not_a_set(schema_info)
    assert_columns_matched_in_order(schema_info, reverse=True)
    assert_columns_matched_any_order(schema_info, reverse=True)
    assert_col_dtype_match(schema_info, "A")
    assert_col_dtype_match(schema_info, "C")
    assert_col_dtype_mismatch(schema_info, "X")
    assert_col_index_match(schema_info, "A")
    assert_col_index_mismatch(schema_info, "C")
    assert_col_index_mismatch(schema_info, "X")

    # Take snapshot of schema info object
    snapshot.assert_match(schema_info_str(schema_info), "schema_info_25-5.txt")


def test_get_val_info(tbl_schema_tests):

    # 1. Schema matches completely and in order; dtypes all correct
    schema = Schema(
        columns=[
            ("a", "String"),
            ("b", "Int64"),
            ("c", "Float64"),
        ]
    )

    # Create a validation object
    validation = Validate(data=tbl_schema_tests).col_schema_match(schema=schema).interrogate()

    # Get the validation info from the first (and only) element of `validation_info` using
    # the `get_val_info()` method
    val_info = validation.validation_info[0].get_val_info()

    # Check that the `val_info` is a dictionary
    assert isinstance(val_info, dict)


def test_get_schema_step_report_01(tbl_schema_tests, snapshot):

    # 1. Schema matches completely and in order; dtypes all correct
    schema = Schema(
        columns=[
            ("a", "String"),
            ("b", "Int64"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_01-0.txt")


def test_get_schema_step_report_01_1(tbl_schema_tests, snapshot):

    # 1-1. Schema matches completely and in order; dtypes all correct
    # - use `complete=False` / `in_order=True`
    schema = Schema(
        columns=[
            ("a", "String"),
            ("b", "Int64"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=False,  # non-default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_01-1.txt")


def test_get_schema_step_report_01_2(tbl_schema_tests, snapshot):

    # 1-2. Schema matches completely and in order; dtypes all correct
    # - use `complete=True` / `in_order=False`
    schema = Schema(
        columns=[
            ("a", "String"),
            ("b", "Int64"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=False,  # non-default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_01-2.txt")


def test_get_schema_step_report_01_3(tbl_schema_tests, snapshot):

    # 1-3. Schema matches completely and in order; dtypes all correct
    # - use `complete=False` / `in_order=False`
    schema = Schema(
        columns=[
            ("a", "String"),
            ("b", "Int64"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=False,  # non-default
            in_order=False,  # non-default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_01-3.txt")


def test_get_schema_step_report_02(tbl_schema_tests, snapshot):

    # 2. Schema matches completely; option taken to match any of two different dtypes for column
    # "a", but all dtypes correct
    schema = Schema(
        columns=[
            ("a", ["String", "Int64"]),
            ("b", "Int64"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_02-0.txt")


def test_get_schema_step_report_02_1(tbl_schema_tests, snapshot):

    # 2-1. Schema matches completely; option taken to match any of two different dtypes for column
    # "a", but all dtypes correct
    # - use `complete=False` / `in_order=True`
    schema = Schema(
        columns=[
            ("a", ["String", "Int64"]),
            ("b", "Int64"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=False,  # non-default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_02-1.txt")


def test_get_schema_step_report_02_2(tbl_schema_tests, snapshot):

    # 2-2. Schema matches completely; option taken to match any of two different dtypes for column
    # "a", but all dtypes correct
    # - use `complete=True` / `in_order=False`
    schema = Schema(
        columns=[
            ("a", ["String", "Int64"]),
            ("b", "Int64"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=False,  # non-default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_02-2.txt")


def test_get_schema_step_report_02_3(tbl_schema_tests, snapshot):

    # 2-3. Schema matches completely; option taken to match any of two different dtypes for column
    # "a", but all dtypes correct
    # - use `complete=False` / `in_order=False`
    schema = Schema(
        columns=[
            ("a", ["String", "Int64"]),
            ("b", "Int64"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=False,  # non-default
            in_order=False,  # non-default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_02-3.txt")


def test_get_schema_step_report_03(tbl_schema_tests, snapshot):

    # 3. Schema has all three columns accounted for but in an incorrect order; dtypes correct
    schema = Schema(
        columns=[
            ("b", "Int64"),
            ("a", "String"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_03-0.txt")


def test_get_schema_step_report_03_1(tbl_schema_tests, snapshot):

    # 3-1. Schema has all three columns accounted for but in an incorrect order; dtypes correct
    # - use `complete=False` / `in_order=True`
    schema = Schema(
        columns=[
            ("b", "Int64"),
            ("a", "String"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=False,  # non-default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_03-1.txt")


def test_get_schema_step_report_03_2(tbl_schema_tests, snapshot):

    # 3-2. Schema has all three columns accounted for but in an incorrect order; dtypes correct
    # - use `complete=True` / `in_order=False`
    schema = Schema(
        columns=[
            ("b", "Int64"),
            ("a", "String"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=False,  # non-default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_03-2.txt")


def test_get_schema_step_report_03_3(tbl_schema_tests, snapshot):

    # 3-3. Schema has all three columns accounted for but in an incorrect order; dtypes correct
    # - use `complete=False` / `in_order=False`
    schema = Schema(
        columns=[
            ("b", "Int64"),
            ("a", "String"),
            ("c", "Float64"),
        ]
    )

    # Use `col_schema_match()` validation method to perform schema check
    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=False,  # non-default
            in_order=False,  # non-default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_03-3.txt")


def test_get_schema_step_report_04(tbl_schema_tests, snapshot):

    # 4. Schema has all three columns accounted for but in an incorrect order; option taken to match
    # any of two different dtypes for column "a", but all dtypes correct
    schema = Schema(
        columns=[
            ("b", "Int64"),
            ("a", ["Int64", "String"]),
            ("c", "Float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_04-0.txt")


def test_get_schema_step_report_05(tbl_schema_tests, snapshot):

    # 5. Schema has all three columns matching, correct order; no dtypes provided
    schema = Schema(
        columns=[
            ("a",),
            ("b",),
            ("c",),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_05-0.txt")


def test_get_schema_step_report_06(tbl_schema_tests, snapshot):

    # 6. Schema has all three columns matching, correct order; incorrect dtypes
    schema = Schema(
        columns=[
            ("a", ["invalid", "invalid"]),
            ("b", "invalid"),
            ("c", "invalid"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_06-0.txt")


def test_get_schema_step_report_07(tbl_schema_tests, snapshot):

    # 7. Schema has 2/3 columns matching, correct order; incorrect dtypes
    schema = Schema(
        columns=[
            ("a", ["invalid", "invalid"]),
            ("c", "invalid"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_07-0.txt")


def test_get_schema_step_report_08(tbl_schema_tests, snapshot):

    # 8. Schema has 2/3 columns matching, incorrect order; incorrect dtypes
    schema = Schema(
        columns=[
            ("c", "invalid"),
            ("a", ["invalid", "invalid"]),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_08-0.txt")


def test_get_schema_step_report_09(tbl_schema_tests, snapshot):

    # 9. Schema has single column match; incorrect dtype
    schema = Schema(
        columns=[
            ("c", "invalid"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_09-0.txt")


def test_get_schema_step_report_10(tbl_schema_tests, snapshot):

    # 10. Schema is empty
    schema = Schema(columns=[])

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_10-0.txt")


def test_get_schema_step_report_11(tbl_schema_tests, snapshot):

    # 11. Schema has complete match of columns plus an additional, unmatched column
    schema = Schema(
        columns=[("a", ["String", "Int64"]), ("b", "Int64"), ("c", "Float64"), ("d", "String")]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_11-0.txt")


def test_get_schema_step_report_12(tbl_schema_tests, snapshot):

    # 12. Schema has partial match of columns (in right order) plus an additional, unmatched column
    schema = Schema(columns=[("a", ["String", "Int64"]), ("c", "Float64"), ("d", "String")])

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_12-0.txt")


def test_get_schema_step_report_13(tbl_schema_tests, snapshot):

    # 13. Schema has no matches to any column names
    schema = Schema(
        columns=[
            ("x", "String"),
            ("y", "Int64"),
            ("z", "Float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_13-0.txt")


def test_get_schema_step_report_14(tbl_schema_tests, snapshot):

    # 14. Schema has all columns matching in case-insensitive manner, correct order; dtypes
    # all correct
    schema = Schema(
        columns=[
            ("A", "String"),
            ("B", "Int64"),
            ("C", "Float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_14-0.txt")


def test_get_schema_step_report_14_1(tbl_schema_tests, snapshot):

    # 14-1. Using `case_sensitive_colnames=False`
    schema = Schema(
        columns=[
            ("A", "String"),
            ("B", "Int64"),
            ("C", "Float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=False,  # non-default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_14-1.txt")


def test_get_schema_step_report_15(tbl_schema_tests, snapshot):

    # 15. Schema has all columns matching in case-insensitive manner, correct order; dtypes
    # all correct
    schema = Schema(
        columns=[
            ("B", "Int64"),
            ("A", "String"),
            ("C", "Float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_15-0.txt")


def test_get_schema_step_report_15_1(tbl_schema_tests, snapshot):

    # 15-1. Using `case_sensitive_colnames=False`
    schema = Schema(
        columns=[
            ("B", "Int64"),
            ("A", "String"),
            ("C", "Float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=False,  # non-default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_15-1.txt")


def test_get_schema_step_report_16(tbl_schema_tests, snapshot):

    # 16. Schema has 2/3 columns matching in case-insensitive manner, correct order; dtypes
    # all correct
    schema = Schema(
        columns=[
            ("A", "String"),
            ("C", "Float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_16-0.txt")


def test_get_schema_step_report_16_1(tbl_schema_tests, snapshot):

    # 16-1. Using `case_sensitive_colnames=False`
    schema = Schema(
        columns=[
            ("A", "String"),
            ("C", "Float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=False,  # non-default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_16-1.txt")


def test_get_schema_step_report_17(tbl_schema_tests, snapshot):

    # 17. Schema has 2/3 columns matching in case-insensitive manner, incorrect order; dtypes
    # all correct
    schema = Schema(
        columns=[
            ("C", "Float64"),
            ("A", "String"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_17-0.txt")


def test_get_schema_step_report_17_1(tbl_schema_tests, snapshot):

    # 17-1. Using `case_sensitive_colnames=False`
    schema = Schema(
        columns=[
            ("C", "Float64"),
            ("A", "String"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=False,  # non-default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_17-1.txt")


def test_get_schema_step_report_18(tbl_schema_tests, snapshot):

    # 18. Schema has one column matching in case-insensitive manner; dtype is correct
    schema = Schema(
        columns=[
            ("C", "Float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_18-0.txt")


def test_get_schema_step_report_18_1(tbl_schema_tests, snapshot):

    # 18-1. Using `case_sensitive_colnames=False`
    schema = Schema(
        columns=[
            ("C", "Float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=False,  # non-default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_18-1.txt")


def test_get_schema_step_report_19(tbl_schema_tests, snapshot):

    # 19. Schema has all three columns matching, correct order; dtypes don't match case of
    # actual dtypes
    schema = Schema(
        columns=[
            ("a", "string"),
            ("b", "int64"),
            ("c", "float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_19-0.txt")


def test_get_schema_step_report_19_1(tbl_schema_tests, snapshot):

    # 19-1. Using `case_sensitive_colnames=False`
    schema = Schema(
        columns=[
            ("a", "string"),
            ("b", "int64"),
            ("c", "float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=False,  # non-default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_19-1.txt")


def test_get_schema_step_report_20(tbl_schema_tests, snapshot):

    # 20. Schema has all three columns matching, correct order; dtypes are substrings of
    # actual dtypes
    schema = Schema(
        columns=[
            ("a", "Str"),
            ("b", "Int"),
            ("c", "Float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_20-0.txt")


def test_get_schema_step_report_20_1(tbl_schema_tests, snapshot):

    # 20-1. Using `full_match_dtypes=False`
    schema = Schema(
        columns=[
            ("a", "Str"),
            ("b", "Int"),
            ("c", "Float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=False,  # non-default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_20-1.txt")


def test_get_schema_step_report_21(tbl_schema_tests, snapshot):

    # 21. Schema has all three columns matching, correct order; dtypes are substrings of actual
    # dtypes where case doesn't match
    schema = Schema(
        columns=[
            ("a", "str"),
            ("b", "int"),
            ("c", "float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_21-0.txt")


def test_get_schema_step_report_21_1(tbl_schema_tests, snapshot):

    # 21-1. Using `case_sensitive_dtypes=False`
    schema = Schema(
        columns=[
            ("a", "str"),
            ("b", "int"),
            ("c", "float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=False,  # non-default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_21-1.txt")


def test_get_schema_step_report_21_2(tbl_schema_tests, snapshot):

    # 21-2. Using `full_match_dtypes=False`
    schema = Schema(
        columns=[
            ("a", "str"),
            ("b", "int"),
            ("c", "float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=False,  # non-default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_21-2.txt")


def test_get_schema_step_report_21_3(tbl_schema_tests, snapshot):

    # 21-3. Using `case_sensitive_dtypes=False` and `full_match_dtypes=False`
    schema = Schema(
        columns=[
            ("a", "str"),
            ("b", "int"),
            ("c", "float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=False,  # non-default
            full_match_dtypes=False,  # non-default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_21-3.txt")


def test_get_schema_step_report_22(tbl_schema_tests, snapshot):

    # 22. Schema has all 2/3 columns matching, missing one, correct order; dtypes don't match
    # case of actual dtypes
    schema = Schema(
        columns=[
            ("a", "string"),
            ("c", "float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_22-0.txt")


def test_get_schema_step_report_22_1(tbl_schema_tests, snapshot):

    # 22-1. Using `case_sensitive_dtypes=False`
    schema = Schema(
        columns=[
            ("a", "string"),
            ("c", "float64"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=False,  # non-default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_22-1.txt")


def test_get_schema_step_report_23(tbl_schema_tests, snapshot):

    # 23. Schema has all 2/3 columns matching, missing one, correct order; dtypes are substrings
    # of actual dtypes
    schema = Schema(
        columns=[
            ("a", "Str"),
            ("c", "Float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_23-0.txt")


def test_get_schema_step_report_23_1(tbl_schema_tests, snapshot):

    # 23-1. Using `full_match_dtypes=False`
    schema = Schema(
        columns=[
            ("a", "Str"),
            ("c", "Float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=False,  # non-default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_23-1.txt")


def test_get_schema_step_report_24(tbl_schema_tests, snapshot):

    # 24. Schema has all 2/3 columns matching, missing one, correct order; dtypes are substrings
    # of actual dtypes where case doesn't match
    schema = Schema(
        columns=[
            ("a", "str"),
            ("c", "float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_24-0.txt")


def test_get_schema_step_report_24_1(tbl_schema_tests, snapshot):

    # 24-1. Using `case_sensitive_dtypes=False`
    schema = Schema(
        columns=[
            ("a", "str"),
            ("c", "float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=False,  # non-default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_24-1.txt")


def test_get_schema_step_report_24_2(tbl_schema_tests, snapshot):

    # 24-2. Using `full_match_dtypes=False`
    schema = Schema(
        columns=[
            ("a", "str"),
            ("c", "float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=False,  # non-default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_24-2.txt")


def test_get_schema_step_report_24_3(tbl_schema_tests, snapshot):

    # 24-3. Using `case_sensitive_dtypes=False` and `full_match_dtypes=False`
    schema = Schema(
        columns=[
            ("a", "str"),
            ("c", "float"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=False,  # non-default
            full_match_dtypes=False,  # non-default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_24-3.txt")


def test_get_schema_step_report_25(tbl_schema_tests, snapshot):

    # 25. Schema has all 2/3 columns matching, missing one, an unmatched column, correct
    # order for the matching set; dtypes are substrings of actual dtypes where case doesn't match
    schema = Schema(
        columns=[
            ("A", "str"),
            ("C", "float"),
            ("X", "int"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_25-0.txt")


def test_get_schema_step_report_25_1(tbl_schema_tests, snapshot):

    # 25-1. Using `case_sensitive_colnames=False`
    schema = Schema(
        columns=[
            ("A", "str"),
            ("C", "float"),
            ("X", "int"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=False,  # non-default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_25-1.txt")


def test_get_schema_step_report_25_2(tbl_schema_tests, snapshot):

    # 25-2. Using `case_sensitive_dtypes=False`
    schema = Schema(
        columns=[
            ("A", "str"),
            ("C", "float"),
            ("X", "int"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=False,  # non-default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_25-2.txt")


def test_get_schema_step_report_25_3(tbl_schema_tests, snapshot):

    # 25-3. Using `full_match_dtypes=False`
    schema = Schema(
        columns=[
            ("A", "str"),
            ("C", "float"),
            ("X", "int"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=True,  # default
            case_sensitive_dtypes=True,  # default
            full_match_dtypes=False,  # non-default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_25-3.txt")


def test_get_schema_step_report_25_4(tbl_schema_tests, snapshot):

    # 25-4. Using `case_sensitive_colnames=False` and `case_sensitive_dtypes=False`
    schema = Schema(
        columns=[
            ("A", "str"),
            ("C", "float"),
            ("X", "int"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=False,  # non-default
            case_sensitive_dtypes=False,  # non-default
            full_match_dtypes=True,  # default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_25-4.txt")


def test_get_schema_step_report_25_5(tbl_schema_tests, snapshot):

    # 25-5. Using `case_sensitive_colnames=False`, `case_sensitive_dtypes=False`, and
    # `full_match_dtypes=False`
    schema = Schema(
        columns=[
            ("A", "str"),
            ("C", "float"),
            ("X", "int"),
        ]
    )

    validation = (
        Validate(data=tbl_schema_tests)
        .col_schema_match(
            schema=schema,
            complete=True,  # default
            in_order=True,  # default
            case_sensitive_colnames=False,  # non-default
            case_sensitive_dtypes=False,  # non-default
            full_match_dtypes=False,  # non-default
        )
        .interrogate()
    )

    report_df = validation.get_step_report(i=-99)

    # Take snapshot of the report DataFrame
    snapshot.assert_match(str(report_df), "schema_step_report_25-5.txt")


@pytest.mark.parametrize(("tbl", "should_pass"), itertools.product(TBL_LIST, [True, False]))
def test_assert_passing(request, tbl: str, *, should_pass: bool) -> None:
    tbl = request.getfixturevalue(tbl)

    if should_pass:
        val = 0  # should always pass
        catcher = contextlib.nullcontext
    else:
        val = 100  # should always fail
        catcher = partial(pytest.raises, AssertionError, match="All tests did not pass")

    v = Validate(tbl).col_vals_gt(columns="x", value=val).interrogate()

    try:
        assert v.all_passed() == should_pass
    except AssertionError:
        pytest.mark.skip(reason="Unexpected result invalidating the test. Please review.")

    with catcher():
        v.assert_passing()  # should not raise since all passing


def test_assert_passing_example() -> None:
    tbl = pl.DataFrame(
        {
            "a": [1, 2, 9, 5],
            "b": [5, 6, 10, 3],
            "c": ["a", "b", "a", "a"],
        }
    )

    validation = (
        Validate(data=tbl)
        .col_vals_gt(columns="a", value=0)
        .col_vals_lt(columns="b", value=9)  # this step will not pass
        .col_vals_in_set(columns="c", set=["a", "b"])
        .interrogate()
    )
    with pytest.raises(AssertionError, match="All tests did not pass"):
        validation.assert_passing()

    passing_validation = (
        Validate(data=tbl)
        .col_vals_gt(columns="a", value=0)
        # now, the invalid step passes
        .col_vals_in_set(columns="c", set=["a", "b"])
        .interrogate()
    )

    passing_validation.assert_passing()
