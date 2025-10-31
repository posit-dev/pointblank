import os
import sys
import tempfile
from unittest.mock import patch

import narwhals as nw
import pandas as pd
import polars as pl
import pytest

from pointblank._utils_llms_txt import (
    _get_api_and_examples_text,
    _get_api_text,
    _get_examples_text,
)
from pointblank._utils import (
    _check_any_df_lib,
    _check_column_exists,
    _check_column_type,
    _check_invalid_fields,
    _column_subset_test_prep,
    _column_test_prep,
    _convert_to_narwhals,
    _copy_dataframe,
    _count_null_values_in_column,
    _count_true_values_in_column,
    _derive_bounds,
    _derive_single_bound,
    _format_to_float_value,
    _format_to_integer_value,
    _get_assertion_from_fname,
    _get_column_dtype,
    _get_fn_name,
    _get_tbl_type,
    _is_date_or_datetime_dtype,
    _is_duration_dtype,
    _is_lazy_frame,
    _is_lib_present,
    _is_narwhals_table,
    _is_numeric_dtype,
    _is_value_a_df,
    _pivot_to_dict,
    _process_ibis_through_narwhals,
    _select_df_lib,
    transpose_dicts,
)
from pointblank.validate import load_dataset

# Import ibis conditionally for tests that need it
try:
    import ibis
except ImportError:
    ibis = None


@pytest.fixture
def tbl_pd():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_missing_pd():
    return pd.DataFrame({"x": [1, 2, pd.NA, 4], "y": [4, pd.NA, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_pl():
    return pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_missing_pl():
    return pl.DataFrame({"x": [1, 2, None, 4], "y": [4, None, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_multiple_types_pd():
    return pd.DataFrame(
        {
            "int": [1, 2, 3, 4],
            "float": [4.0, 5.0, 6.0, 7.0],
            "str": ["a", "b", "c", "d"],
            "bool": [True, False, True, False],
            "date": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]),
            "datetime": pd.to_datetime(
                [
                    "2021-01-01 00:00:00",
                    "2021-01-02 00:00:00",
                    "2021-01-03 00:00:00",
                    "2021-01-04 00:00:00",
                ]
            ),
            "timedelta": pd.to_timedelta(["1 days", "2 days", "3 days", "4 days"]),
        }
    )


@pytest.fixture
def tbl_multiple_types_pl():
    # Create a Polars DataFrame with multiple data types (int, float, str, date, datetime, timedelta)
    return pl.DataFrame(
        {
            "int": [1, 2, 3, 4],
            "float": [4.0, 5.0, 6.0, 7.0],
            "str": ["a", "b", "c", "d"],
            "bool": [True, False, True, False],
            "date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"],
            "datetime": [
                "2021-01-01 00:00:00",
                "2021-01-02 00:00:00",
                "2021-01-03 00:00:00",
                "2021-01-04 00:00:00",
            ],
            "timedelta": [1, 2, 3, 4],
        }
    ).with_columns(
        date=pl.col("date").str.strptime(pl.Date, "%Y-%m-%d"),
        datetime=pl.col("datetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
        timedelta=pl.duration(days=pl.col("timedelta")),
    )


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_convert_to_narwhals(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    assert isinstance(dfn, nw.DataFrame)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_double_convert_to_narwhals(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)
    dfn_2 = _convert_to_narwhals(dfn)

    assert isinstance(dfn_2, nw.DataFrame)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_pd", "tbl_pl"],
)
def test_check_column_exists_no_error(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    _check_column_exists(dfn=dfn, column="x")
    _check_column_exists(dfn=dfn, column="y")
    _check_column_exists(dfn=dfn, column="z")


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_missing_pd", "tbl_missing_pl"],
)
def test_check_column_exists_missing_values_no_error(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    _check_column_exists(dfn=dfn, column="x")
    _check_column_exists(dfn=dfn, column="y")
    _check_column_exists(dfn=dfn, column="z")


def test_is_numeric_dtype():
    assert _is_numeric_dtype(dtype="int")
    assert _is_numeric_dtype(dtype="float")
    assert _is_numeric_dtype(dtype="int64")
    assert _is_numeric_dtype(dtype="float64")


def test_is_date_or_datetime_dtype():
    assert _is_date_or_datetime_dtype(dtype="datetime")
    assert _is_date_or_datetime_dtype(dtype="date")
    assert _is_date_or_datetime_dtype(dtype="datetime(time_unit='ns', time_zone=none)")
    assert _is_date_or_datetime_dtype(dtype="datetime(time_unit='us', time_zone=none)")


def test_is_duration_dtype():
    assert _is_duration_dtype(dtype="duration")
    assert _is_duration_dtype(dtype="duration(time_unit='ns')")
    assert _is_duration_dtype(dtype="duration(time_unit='us')")


def test_get_column_dtype_pd(tbl_multiple_types_pd):
    dfn = _convert_to_narwhals(tbl_multiple_types_pd)

    assert _get_column_dtype(dfn=dfn, column="int") == "int64"
    assert _get_column_dtype(dfn=dfn, column="float") == "float64"
    assert _get_column_dtype(dfn=dfn, column="str") == "string"
    assert _get_column_dtype(dfn=dfn, column="bool") == "boolean"
    assert _get_column_dtype(dfn=dfn, column="date") == "datetime(time_unit='ns', time_zone=none)"
    assert (
        _get_column_dtype(dfn=dfn, column="datetime") == "datetime(time_unit='ns', time_zone=none)"
    )
    assert _get_column_dtype(dfn=dfn, column="timedelta") == "duration(time_unit='ns')"

    assert _get_column_dtype(dfn=dfn, column="int", lowercased=False) == "Int64"
    assert _get_column_dtype(dfn=dfn, column="float", lowercased=False) == "Float64"
    assert _get_column_dtype(dfn=dfn, column="str", lowercased=False) == "String"


def test_get_column_dtype_pl(tbl_multiple_types_pl):
    dfn = _convert_to_narwhals(tbl_multiple_types_pl)

    assert _get_column_dtype(dfn=dfn, column="int") == "int64"
    assert _get_column_dtype(dfn=dfn, column="float") == "float64"
    assert _get_column_dtype(dfn=dfn, column="str") == "string"
    assert _get_column_dtype(dfn=dfn, column="bool") == "boolean"
    assert _get_column_dtype(dfn=dfn, column="date") == "date"
    assert (
        _get_column_dtype(dfn=dfn, column="datetime") == "datetime(time_unit='us', time_zone=none)"
    )
    assert _get_column_dtype(dfn=dfn, column="timedelta") == "duration(time_unit='us')"

    assert _get_column_dtype(dfn=dfn, column="int", lowercased=False) == "Int64"
    assert _get_column_dtype(dfn=dfn, column="float", lowercased=False) == "Float64"
    assert _get_column_dtype(dfn=dfn, column="str", lowercased=False) == "String"


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_multiple_types_pd", "tbl_multiple_types_pl"],
)
def test_check_column_type(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    _check_column_type(dfn=dfn, column="int", allowed_types=["numeric"])
    _check_column_type(dfn=dfn, column="int", allowed_types=["numeric", "str"])
    _check_column_type(dfn=dfn, column="int", allowed_types=["numeric", "str", "bool"])

    _check_column_type(dfn=dfn, column="float", allowed_types=["numeric"])
    _check_column_type(dfn=dfn, column="float", allowed_types=["numeric", "str"])
    _check_column_type(dfn=dfn, column="float", allowed_types=["numeric", "str", "bool"])

    _check_column_type(dfn=dfn, column="str", allowed_types=["str"])
    _check_column_type(dfn=dfn, column="str", allowed_types=["str", "numeric"])
    _check_column_type(dfn=dfn, column="str", allowed_types=["str", "numeric", "bool"])

    _check_column_type(dfn=dfn, column="bool", allowed_types=["bool"])
    _check_column_type(dfn=dfn, column="bool", allowed_types=["bool", "numeric"])
    _check_column_type(dfn=dfn, column="bool", allowed_types=["bool", "numeric", "str"])

    _check_column_type(dfn=dfn, column="datetime", allowed_types=["datetime"])
    _check_column_type(dfn=dfn, column="datetime", allowed_types=["datetime", "str"])
    _check_column_type(dfn=dfn, column="datetime", allowed_types=["datetime", "str", "numeric"])

    _check_column_type(dfn=dfn, column="timedelta", allowed_types=["duration"])
    _check_column_type(dfn=dfn, column="timedelta", allowed_types=["duration", "str"])
    _check_column_type(dfn=dfn, column="timedelta", allowed_types=["duration", "str", "numeric"])


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_multiple_types_pd", "tbl_multiple_types_pl"],
)
def test_check_column_type_raises(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="int", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="float", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="str", allowed_types=["numeric"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="bool", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="date", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="datetime", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="timedelta", allowed_types=["str"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="int", allowed_types=["bool"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="float", allowed_types=["bool"])
    with pytest.raises(TypeError):
        _check_column_type(dfn=dfn, column="str", allowed_types=["bool"])


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_multiple_types_pd", "tbl_multiple_types_pl"],
)
def test_check_column_type_raises_invalid_input(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    dfn = _convert_to_narwhals(tbl)

    with pytest.raises(ValueError):
        _check_column_type(dfn=dfn, column="int", allowed_types=[])

    with pytest.raises(ValueError):
        _check_column_type(
            dfn=dfn, column="int", allowed_types=["numeric", "str", "bool", "invalid"]
        )


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_multiple_types_pd", "tbl_multiple_types_pl"],
)
def test_check_column_test_prep(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    _column_test_prep(df=tbl, column="int", allowed_types=["numeric"])
    _column_test_prep(df=tbl, column="float", allowed_types=["numeric"])
    _column_test_prep(df=tbl, column="str", allowed_types=["str"])
    _column_test_prep(df=tbl, column="bool", allowed_types=["bool"])
    _column_test_prep(df=tbl, column="date", allowed_types=["datetime"])
    _column_test_prep(df=tbl, column="datetime", allowed_types=["datetime"])
    _column_test_prep(df=tbl, column="timedelta", allowed_types=["duration"])

    # Using `allowed_types=None` bypasses the type check
    _column_test_prep(df=tbl, column="int", allowed_types=None)


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_multiple_types_pd", "tbl_multiple_types_pl"],
)
def test_check_column_test_prep_raises(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    # No types in `allowed_types` match the column data type
    with pytest.raises(TypeError):
        _column_test_prep(
            df=tbl, column="int", allowed_types=["str", "bool", "datetime", "duration"]
        )

    # Column not present in DataFrame
    with pytest.raises(ValueError):
        _column_test_prep(df=tbl, column="invalid", allowed_types=["numeric"])


@pytest.mark.parametrize("tbl_type", ["polars", "duckdb"])
def test_count_true_values_in_column(tbl_type):
    data = load_dataset(dataset="small_table", tbl_type=tbl_type)

    assert _count_true_values_in_column(tbl=data, column="e") == 8
    assert _count_true_values_in_column(tbl=data, column="e", inverse=True) == 5


@pytest.mark.parametrize("tbl_type", ["polars", "duckdb"])
def test_count_null_values_in_column(tbl_type):
    data = load_dataset(dataset="small_table", tbl_type=tbl_type)

    assert _count_null_values_in_column(tbl=data, column="c") == 2


def test_format_to_integer_value():
    assert _format_to_integer_value(0) == "0"
    assert _format_to_integer_value(0.3) == "0"
    assert _format_to_integer_value(0.7) == "1"
    assert _format_to_integer_value(1) == "1"
    assert _format_to_integer_value(10) == "10"
    assert _format_to_integer_value(100) == "100"
    assert _format_to_integer_value(1000) == "1,000"
    assert _format_to_integer_value(10000) == "10,000"
    assert _format_to_integer_value(100000) == "100,000"
    assert _format_to_integer_value(1000000) == "1,000,000"
    assert _format_to_integer_value(-232323) == "\u2212" + "232,323"

    assert _format_to_integer_value(-232323, locale="de") == "\u2212" + "232.323"
    assert _format_to_integer_value(-232323, locale="fi") == "\u2212" + "232 323"
    assert _format_to_integer_value(-232323, locale="fr") == "\u2212" + "232" + "\u202f" + "323"


def test_format_to_integer_value_error():
    with pytest.raises(TypeError):
        _format_to_integer_value([5])

    with pytest.raises(ValueError):
        _format_to_integer_value(5, locale="invalid")


def test_format_to_float_value():
    assert _format_to_float_value(0) == "0.00"
    assert _format_to_float_value(0, decimals=0) == "0"
    assert _format_to_float_value(0.3343) == "0.33"
    assert _format_to_float_value(0.7) == "0.70"
    assert _format_to_float_value(1) == "1.00"
    assert _format_to_float_value(1000) == "1,000.00"
    assert _format_to_float_value(10000) == "10,000.00"
    assert _format_to_float_value(-232323.11) == "\u2212" + "232,323.11"

    assert _format_to_float_value(-232323.11, locale="de") == "\u2212" + "232.323,11"
    assert _format_to_float_value(-232323.11, locale="fi") == "\u2212" + "232 323,11"
    assert _format_to_float_value(-232323.11, locale="fr") == "\u2212" + "232" + "\u202f" + "323,11"


def test_format_to_float_value_error():
    with pytest.raises(TypeError):
        _format_to_float_value([5])

    with pytest.raises(ValueError):
        _format_to_float_value(5, locale="invalid")


def test_format_to_integer_float_no_df_lib():
    # Mock the absence of the both the Pandas and Polars libraries
    with patch.dict(sys.modules, {"pandas": None, "polars": None}):
        assert _format_to_integer_value(1000) == "1,000"
        assert _format_to_float_value(1000) == "1,000.00"


def test_format_to_integer_float_only_polars(monkeypatch):
    # Mock the absence of the Pandas library
    monkeypatch.delitem(sys.modules, "pandas", raising=False)
    assert _format_to_integer_value(1000) == "1,000"
    assert _format_to_float_value(1000) == "1,000.00"


def test_get_fn_name():
    def get_name():
        return _get_fn_name()

    assert get_name() == "get_name"


def test_get_assertion_from_fname():
    def col_vals_gt():
        return _get_assertion_from_fname()

    def col_vals_lt():
        return _get_assertion_from_fname()

    def col_vals_eq():
        return _get_assertion_from_fname()

    def col_vals_ne():
        return _get_assertion_from_fname()

    def col_vals_ge():
        return _get_assertion_from_fname()

    def col_vals_le():
        return _get_assertion_from_fname()

    def col_vals_between():
        return _get_assertion_from_fname()

    def col_vals_outside():
        return _get_assertion_from_fname()

    def col_vals_in_set():
        return _get_assertion_from_fname()

    def col_vals_not_in_set():
        return _get_assertion_from_fname()

    assert col_vals_gt() == "gt"
    assert col_vals_lt() == "lt"
    assert col_vals_eq() == "eq"
    assert col_vals_ne() == "ne"
    assert col_vals_ge() == "ge"
    assert col_vals_le() == "le"
    assert col_vals_between() == "between"
    assert col_vals_outside() == "outside"
    assert col_vals_in_set() == "in_set"
    assert col_vals_not_in_set() == "not_in_set"


def test_check_invalid_fields():
    with pytest.raises(ValueError):
        _check_invalid_fields(
            fields=["invalid"], valid_fields=["numeric", "str", "bool", "datetime", "duration"]
        )

    with pytest.raises(ValueError):
        _check_invalid_fields(
            fields=["numeric", "str", "bool", "datetime", "duration", "invalid"],
            valid_fields=["numeric", "str", "bool", "datetime", "duration"],
        )


def test_select_df_lib():
    # Mock the absence of the both the Pandas and Polars libraries
    with patch.dict(sys.modules, {"pandas": None, "polars": None}):
        # An ImportError is raised when the `pandas` and `polars` packages are not installed
        with pytest.raises(ImportError):
            _select_df_lib()

    # Mock the absence of the Pandas library
    with patch.dict(sys.modules, {"pandas": None}):
        # The Polars library is selected when the `pandas` package is not installed
        assert _select_df_lib(preference="polars") == pl
        assert _select_df_lib(preference="pandas") == pl

    # Mock the absence of the Polars library
    with patch.dict(sys.modules, {"polars": None}):
        # The Pandas library is selected when the `polars` package is not installed
        assert _select_df_lib(preference="pandas") == pd
        assert _select_df_lib(preference="polars") == pd

    # Where both the Pandas and Polars libraries are available
    assert _select_df_lib(preference="pandas") == pd
    assert _select_df_lib(preference="polars") == pl


def test_get_tbl_type():
    assert _get_tbl_type(pd.DataFrame()) == "pandas"
    assert _get_tbl_type(pl.DataFrame()) == "polars"


def test_get_api_text():
    assert isinstance(_get_api_text(), str)


def test_get_examples_text():
    assert isinstance(_get_examples_text(), str)


def test_get_api_and_examples_text():
    assert isinstance(_get_api_and_examples_text(), str)


def test_transpose_dicts():
    # Test with empty list
    assert transpose_dicts([]) == {}

    # Test with single dict
    result = transpose_dicts([{"a": 1, "b": 2}])
    expected = {"a": [1], "b": [2]}
    assert result == expected

    # Test with multiple dicts with same keys
    result = transpose_dicts([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    expected = {"a": [1, 3], "b": [2, 4]}
    assert result == expected

    # Test with multiple dicts with different keys (missing values become None)
    result = transpose_dicts([{"a": 1, "b": 2}, {"a": 3, "c": 4}])
    expected = {"a": [1, 3], "b": [2, None], "c": [None, 4]}
    assert result == expected


def test_derive_single_bound():
    # Test with valid inputs
    assert _derive_single_bound(100, 0.1) == 10  # tol < 1, so tol * ref
    assert _derive_single_bound(100, 1.5) == 1  # tol >= 1, so int(tol)
    assert _derive_single_bound(100, 2) == 2  # integer tolerance
    assert _derive_single_bound(50, 0.5) == 25  # half of reference

    # Test error cases
    with pytest.raises(TypeError):
        _derive_single_bound(100, "invalid")

    with pytest.raises(TypeError):
        _derive_single_bound(100, [1, 2])

    with pytest.raises(ValueError):
        _derive_single_bound(100, -1)

    with pytest.raises(ValueError):
        _derive_single_bound(100, -0.5)


def test_derive_bounds():
    # Test with single tolerance value
    result = _derive_bounds(100, 0.1)
    expected = (10, 10)
    assert result == expected

    # Test with tuple tolerance
    result = _derive_bounds(100, (0.1, 0.2))
    expected = (10, 20)
    assert result == expected

    # Test with tuple of integers
    result = _derive_bounds(100, (2, 3))
    expected = (2, 3)
    assert result == expected


def test_is_narwhals_table():
    # Test with actual Polars/Pandas DataFrames (not narwhals wrapped)
    pd_df = pd.DataFrame({"x": [1, 2, 3]})
    pl_df = pl.DataFrame({"x": [1, 2, 3]})
    assert not _is_narwhals_table(pd_df)
    assert not _is_narwhals_table(pl_df)

    # Test with narwhals wrapped DataFrame
    nw_df = nw.from_native(pd_df)
    assert _is_narwhals_table(nw_df)

    # Test with non-DataFrame objects
    assert not _is_narwhals_table("string")
    assert not _is_narwhals_table(123)
    assert not _is_narwhals_table([1, 2, 3])


def test_is_lazy_frame():
    # Test with regular DataFrames
    pd_df = pd.DataFrame({"x": [1, 2, 3]})
    pl_df = pl.DataFrame({"x": [1, 2, 3]})
    assert not _is_lazy_frame(pd_df)
    assert not _is_lazy_frame(pl_df)

    # Test with lazy frame
    pl_lazy = pl.DataFrame({"x": [1, 2, 3]}).lazy()
    assert _is_lazy_frame(pl_lazy)

    # Test with narwhals lazy frame
    nw_lazy = nw.from_native(pl_lazy)
    assert _is_lazy_frame(nw_lazy)

    # Test with non-DataFrame objects
    assert not _is_lazy_frame("string")
    assert not _is_lazy_frame(123)


def test_is_lib_present():
    # Test with libraries that should be present
    assert _is_lib_present("sys")
    assert _is_lib_present("os")
    assert _is_lib_present("pandas")
    assert _is_lib_present("polars")

    # Test with library that should not be present
    assert not _is_lib_present("nonexistent_library")


def test_check_any_df_lib():
    # Should not raise an error since both pandas and polars are available
    _check_any_df_lib("test_method")

    # Test error case when neither pandas nor polars is available
    with patch.dict(sys.modules, {"pandas": None, "polars": None}):
        with pytest.raises(ImportError):
            _check_any_df_lib("test_method")


def test_is_value_a_df():
    # Test with valid DataFrames
    pd_df = pd.DataFrame({"x": [1, 2, 3]})
    pl_df = pl.DataFrame({"x": [1, 2, 3]})
    assert _is_value_a_df(pd_df)
    assert _is_value_a_df(pl_df)

    # Test with non-DataFrame objects
    assert not _is_value_a_df("string")
    assert not _is_value_a_df(123)
    assert not _is_value_a_df([1, 2, 3])
    assert not _is_value_a_df({"x": [1, 2, 3]})


@pytest.mark.parametrize(
    "tbl_fixture",
    ["tbl_multiple_types_pd", "tbl_multiple_types_pl"],
)
def test_column_subset_test_prep(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    # Test with valid column subset
    dfn = _column_subset_test_prep(df=tbl, columns_subset=["int", "str"])
    assert isinstance(dfn, nw.DataFrame)

    # Test with None columns_subset
    dfn = _column_subset_test_prep(df=tbl, columns_subset=None)
    assert isinstance(dfn, nw.DataFrame)

    # Test with empty columns_subset
    dfn = _column_subset_test_prep(df=tbl, columns_subset=[])
    assert isinstance(dfn, nw.DataFrame)

    # Test with check_exists=False (should not raise even with invalid columns)
    dfn = _column_subset_test_prep(df=tbl, columns_subset=["invalid"], check_exists=False)
    assert isinstance(dfn, nw.DataFrame)

    # Test error case with invalid column when check_exists=True
    with pytest.raises(ValueError):
        _column_subset_test_prep(df=tbl, columns_subset=["invalid"], check_exists=True)


def test_pivot_to_dict():
    # Test basic functionality
    input_dict = {
        "col1": {"key1": "value1", "key2": "value2"},
        "col2": {"key1": "value3", "key3": "value4"},
    }

    result = _pivot_to_dict(input_dict)
    expected = {"key1": ["value1", "value3"], "key2": ["value2", None], "key3": [None, "value4"]}
    assert result == expected

    # Test with empty dict
    assert _pivot_to_dict({}) == {}

    # Test with single column
    input_dict = {"col1": {"key1": "value1", "key2": "value2"}}
    result = _pivot_to_dict(input_dict)
    expected = {"key1": ["value1"], "key2": ["value2"]}
    assert result == expected


def test_process_ibis_through_narwhals():
    # Test with non-Ibis table type
    pd_df = pd.DataFrame({"x": [1, 2, 3]})
    result_data, result_type = _process_ibis_through_narwhals(pd_df, "pandas")
    assert result_data is pd_df
    assert result_type == "pandas"

    # Test with mock Ibis table type
    # Since we can't easily create real Ibis tables in tests, we'll mock the behavior
    with patch("pointblank._utils.nw.from_native") as mock_nw:
        mock_nw.return_value = "mocked_narwhals_df"

        # Test successful Narwhals wrapping
        result_data, result_type = _process_ibis_through_narwhals("mock_ibis_table", "duckdb")
        assert result_data == "mocked_narwhals_df"
        assert result_type == "narwhals"

        # Test fallback when Narwhals fails
        mock_nw.side_effect = Exception("Narwhals failed")
        result_data, result_type = _process_ibis_through_narwhals("mock_ibis_table", "duckdb")
        assert result_data == "mock_ibis_table"
        assert result_type == "duckdb"


def test_get_tbl_type_additional():
    """Test invalid input detection in _get_tbl_type()."""
    with pytest.raises(TypeError):
        _get_tbl_type("invalid_data")


def test_get_tbl_type_pyspark():
    """Test detection of PySpark DataFrames."""

    class MockPySparkNamespace:
        def __str__(self):
            return "pyspark.sql.module"

    class MockPySparkDF:
        def __native_namespace__(self):
            return MockPySparkNamespace()

    mock_pyspark_df = MockPySparkDF()

    with patch("pointblank._utils.nw.from_native", return_value=mock_pyspark_df):
        # Create a non-Ibis object to trigger the regular DataFrame path
        mock_data = type("MockNonIbisData", (), {})()
        result = _get_tbl_type(mock_data)
        assert result == "pyspark"


def test_get_column_dtype_raw_parameter():
    """Test the `raw=` parameter of _get_column_dtype()."""

    tbl = pd.DataFrame({"int_col": [1, 2, 3]})
    dfn = _convert_to_narwhals(tbl)

    # Test raw=True as this should return the raw dtype from the schema
    raw_dtype = _get_column_dtype(dfn=dfn, column="int_col", raw=True)

    assert raw_dtype is not None  # Just check that it returns something


def test_get_tbl_type_ibis_memtable():
    """Test detection of Ibis memtable tables."""

    # Create an actual Ibis memtable
    df_pd = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})
    ibis_table = ibis.memtable(df_pd)

    # Test that it's detected as a memtable
    result = _get_tbl_type(ibis_table)
    assert result == "memtable"


def test_get_tbl_type_ibis_parquet():
    """Test detection of Ibis parquet tables."""

    # Create a temporary parquet file
    df_pd = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
        tmp_path = tmp.name
        df_pd.to_parquet(tmp_path)

    try:
        # Read it with ibis
        conn = ibis.duckdb.connect()
        ibis_table = conn.read_parquet(tmp_path)

        # Test that it's detected as parquet
        result = _get_tbl_type(ibis_table)
        assert result == "parquet"
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_get_tbl_type_ibis_duckdb():
    """Test detection of plain DuckDB Ibis tables."""

    # Create a DuckDB table that's not a memtable or Parquet
    conn = ibis.duckdb.connect()

    # Create a table directly in DuckDB
    ibis_table = conn.create_table(
        "test_table", pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}), overwrite=True
    )

    # Test that it's detected as duckdb
    result = _get_tbl_type(ibis_table)

    assert result == "duckdb"


def test_copy_dataframe_exception_handling():
    """Test _copy_dataframe() with DataFrames that don't support standard copy methods."""

    # Test with a normal DataFrame (should use .copy())
    df_pd = pd.DataFrame({"x": [1, 2, 3]})
    copied = _copy_dataframe(df_pd)

    assert copied is not df_pd  # Should be a different object
    assert copied.equals(df_pd)  # But with same values

    # Test with a Polars DataFrame (should use .clone())
    df_pl = pl.DataFrame({"x": [1, 2, 3]})
    copied_pl = _copy_dataframe(df_pl)

    assert copied_pl is not df_pl
    assert copied_pl.equals(df_pl)

    # Test with a mock object that has copy() but it raises an exception
    class MockDFWithBrokenCopy:
        def copy(self):
            raise RuntimeError("Copy failed!")

        def clone(self):
            raise RuntimeError("Clone failed!")

        def select(self, col):
            raise RuntimeError("Select failed!")

    mock_df = MockDFWithBrokenCopy()

    # Should fall back to deepcopy or return original
    result = _copy_dataframe(mock_df)

    # Since deepcopy might work or fail, we just verify no exception is raised
    assert result is not None

    # Test with an object that can't be deepcopied either
    class UncopyableObject:
        def copy(self):
            raise RuntimeError("Copy failed!")

        def clone(self):
            raise RuntimeError("Clone failed!")

        def select(self, col):
            raise RuntimeError("Select failed!")

        def __deepcopy__(self, memo):
            raise RuntimeError("Deepcopy failed!")

    uncopyable = UncopyableObject()

    # Should return the original object without raising an exception
    result = _copy_dataframe(uncopyable)

    assert result is uncopyable  # Should return the original
