from __future__ import annotations

import datetime
import pytest
import narwhals as nw
import polars as pl

from pointblank.scan_profile import (
    _TypeMap,
    ColumnProfile,
    _DateProfile,
    _BoolProfile,
    _StringProfile,
    _NumericProfile,
    _DataProfile,
    _as_physical,
)
from pointblank.scan_profile_stats import (
    NMissing,
    NUnique,
    MinStat,
    MaxStat,
    MeanStat,
    FreqStat,
)


# ──────────────────────────────────────────────────────────────────────
# _TypeMap
# ──────────────────────────────────────────────────────────────────────


def test_typemap_is_illegal_struct():
    assert _TypeMap.is_illegal("Struct{'a': Int64}") is True


def test_typemap_is_illegal_non_struct():
    assert _TypeMap.is_illegal("Int64") is False
    assert _TypeMap.is_illegal("String") is False


def test_typemap_fetch_prof_map_returns_dict():
    m = _TypeMap.fetch_prof_map()
    assert _TypeMap.NUMERIC in m
    assert _TypeMap.STRING in m
    assert _TypeMap.DATE in m
    assert _TypeMap.BOOL in m
    assert m[_TypeMap.NUMERIC] is _NumericProfile
    assert m[_TypeMap.STRING] is _StringProfile
    assert m[_TypeMap.DATE] is _DateProfile
    assert m[_TypeMap.BOOL] is _BoolProfile


def test_typemap_fetch_profile_numeric():
    cls = _TypeMap.fetch_profile("Int64")
    assert cls is _NumericProfile


def test_typemap_fetch_profile_float():
    cls = _TypeMap.fetch_profile("Float64")
    assert cls is _NumericProfile


def test_typemap_fetch_profile_string():
    cls = _TypeMap.fetch_profile("String")
    assert cls is _StringProfile


def test_typemap_fetch_profile_categorical():
    cls = _TypeMap.fetch_profile("Categorical")
    assert cls is _StringProfile


def test_typemap_fetch_profile_date():
    cls = _TypeMap.fetch_profile("Date")
    assert cls is _DateProfile


def test_typemap_fetch_profile_bool():
    cls = _TypeMap.fetch_profile("Boolean")
    assert cls is _BoolProfile


def test_typemap_fetch_icon_numeric():
    icon = _TypeMap.fetch_icon(_TypeMap.NUMERIC)
    assert isinstance(icon, str)
    assert len(icon) > 0


def test_typemap_fetch_icon_string():
    icon = _TypeMap.fetch_icon(_TypeMap.STRING)
    assert isinstance(icon, str)


def test_typemap_fetch_icon_date():
    icon = _TypeMap.fetch_icon(_TypeMap.DATE)
    assert isinstance(icon, str)


def test_typemap_fetch_icon_bool():
    icon = _TypeMap.fetch_icon(_TypeMap.BOOL)
    assert isinstance(icon, str)


def test_typemap_fetch_icon_unknown_key():
    from enum import Enum

    class FakeType(Enum):
        UNKNOWN = ("unknown",)

    icon = _TypeMap.fetch_icon(FakeType.UNKNOWN)
    assert isinstance(icon, str)


# ──────────────────────────────────────────────────────────────────────
# ColumnProfile
# ──────────────────────────────────────────────────────────────────────


def test_column_profile_init():
    cp = ColumnProfile(colname="mycol", coltype="Int64")
    assert cp.colname == "mycol"
    assert cp.coltype == "Int64"
    assert cp.statistics == []


def test_column_profile_sample_data_setter_sequence():
    cp = ColumnProfile(colname="a", coltype="Int64")
    cp.sample_data = [1, 2, 3]
    assert cp.sample_data == [1, 2, 3]


def test_column_profile_sample_data_setter_tuple():
    cp = ColumnProfile(colname="a", coltype="Int64")
    cp.sample_data = (1, 2, 3)
    assert cp.sample_data == (1, 2, 3)


def test_column_profile_spawn_profile():
    cp = ColumnProfile(colname="a", coltype="Float64")
    cp.sample_data = [1.0, 2.0]
    spawned = cp.spawn_profile(_NumericProfile)
    assert isinstance(spawned, _NumericProfile)
    assert spawned.colname == "a"
    assert spawned.coltype == "Float64"
    assert spawned.sample_data == [1.0, 2.0]


def test_column_profile_calc_stats():
    df = nw.from_native(pl.DataFrame({"_col": [1, 2, None]}))
    cp = ColumnProfile(colname="_col", coltype="Int64")
    cp.sample_data = [1, 2]
    cp.calc_stats(df)
    names = [s.name for s in cp.statistics]
    assert "n_missing" in names
    assert "n_unique" in names


# ──────────────────────────────────────────────────────────────────────
# _DateProfile
# ──────────────────────────────────────────────────────────────────────


def test_date_profile_calc_stats():
    df = nw.from_native(
        pl.DataFrame({"mydate": [datetime.date(2020, 1, 1), datetime.date(2021, 6, 15)]})
    )
    dp = _DateProfile(colname="mydate", coltype="Date")
    dp.sample_data = [datetime.date(2020, 1, 1)]
    dp.calc_stats(df)
    names = [s.name for s in dp.statistics]
    assert "min" in names
    assert "max" in names


def test_date_profile_type():
    assert _DateProfile._type is _TypeMap.DATE


# ──────────────────────────────────────────────────────────────────────
# _BoolProfile
# ──────────────────────────────────────────────────────────────────────


def test_bool_profile_calc_stats():
    df = nw.from_native(pl.DataFrame({"mybool": [True, False, True, True]}))
    bp = _BoolProfile(colname="mybool", coltype="Boolean")
    bp.sample_data = [True]
    bp.calc_stats(df)
    assert len(bp.statistics) == 1
    assert isinstance(bp.statistics[0], FreqStat)
    freqs = bp.statistics[0].val
    assert "True" in freqs
    assert "False" in freqs
    assert freqs["True"] == 3
    assert freqs["False"] == 1


def test_bool_profile_type():
    assert _BoolProfile._type is _TypeMap.BOOL


# ──────────────────────────────────────────────────────────────────────
# _StringProfile
# ──────────────────────────────────────────────────────────────────────


def test_string_profile_calc_stats():
    df = nw.from_native(pl.DataFrame({"mystr": ["hello", "world", "foo", "bar", "baz"]}))
    sp = _StringProfile(colname="mystr", coltype="String")
    sp.sample_data = ["hello"]
    sp.calc_stats(df)
    names = [s.name for s in sp.statistics]
    assert "mean" in names
    assert "median" in names
    assert "std" in names
    assert "min" in names
    assert "max" in names
    assert "p05" in names
    assert "q_1" in names
    assert "q_3" in names
    assert "p95" in names
    assert "iqr" in names


def test_string_profile_type():
    assert _StringProfile._type is _TypeMap.STRING


# ──────────────────────────────────────────────────────────────────────
# _NumericProfile
# ──────────────────────────────────────────────────────────────────────


def test_numeric_profile_calc_stats():
    df = nw.from_native(pl.DataFrame({"mynum": [1.0, 2.0, 3.0, 4.0, 5.0]}))
    np_ = _NumericProfile(colname="mynum", coltype="Float64")
    np_.sample_data = [1.0]
    np_.calc_stats(df)
    names = [s.name for s in np_.statistics]
    assert "mean" in names
    assert "median" in names
    assert "std" in names
    assert "min" in names
    assert "max" in names
    assert "p05" in names
    assert "q_1" in names
    assert "q_3" in names
    assert "p95" in names
    assert "iqr" in names


def test_numeric_profile_type():
    assert _NumericProfile._type is _TypeMap.NUMERIC


# ──────────────────────────────────────────────────────────────────────
# _DataProfile
# ──────────────────────────────────────────────────────────────────────


def test_data_profile_init():
    dp = _DataProfile(
        table_name="mytable",
        columns=["a", "b"],
        implementation=nw.Implementation.POLARS,
    )
    assert dp.table_name == "mytable"
    assert dp.columns == ["a", "b"]
    assert dp.column_profiles == []


def test_data_profile_set_row_count_eager():
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3, 4, 5]}))
    dp = _DataProfile(table_name=None, columns=["a"], implementation=nw.Implementation.POLARS)
    dp.set_row_count(df)
    assert dp.row_count == 5


def test_data_profile_set_row_count_lazy():
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}).lazy())
    dp = _DataProfile(table_name=None, columns=["a"], implementation=nw.Implementation.POLARS)
    dp.set_row_count(df)
    assert dp.row_count == 3


def test_data_profile_as_dataframe_numeric():
    df = nw.from_native(pl.DataFrame({"mynum": [1.0, 2.0, 3.0, 4.0, 5.0]}))
    dp = _DataProfile(table_name="t", columns=["mynum"], implementation=nw.Implementation.POLARS)
    dp.set_row_count(df)
    np_ = _NumericProfile(colname="mynum", coltype="Float64")
    np_.sample_data = [1.0, 2.0]
    np_.calc_stats(df)
    dp.column_profiles.append(np_)
    result = dp.as_dataframe()
    assert "colname" in result.columns
    assert "coltype" in result.columns
    assert "mean" in result.columns


def test_data_profile_as_dataframe_multiple_profiles():
    df = nw.from_native(pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]}))
    dp = _DataProfile(table_name="t", columns=["a", "b"], implementation=nw.Implementation.POLARS)
    dp.set_row_count(df)

    np_ = _NumericProfile(colname="a", coltype="Float64")
    np_.sample_data = [1.0]
    df_a = nw.from_native(pl.DataFrame({"a": [1.0, 2.0, 3.0]}))
    np_.calc_stats(df_a)
    dp.column_profiles.append(np_)

    sp = _StringProfile(colname="b", coltype="String")
    sp.sample_data = ["x"]
    df_b = nw.from_native(pl.DataFrame({"b": ["x", "y", "z"]}))
    sp.calc_stats(df_b)
    dp.column_profiles.append(sp)

    result = dp.as_dataframe(strict=False)
    assert len(result) == 2


# ──────────────────────────────────────────────────────────────────────
# _as_physical
# ──────────────────────────────────────────────────────────────────────


def test_as_physical_with_eager_dataframe():
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}))
    result = _as_physical(df)
    assert isinstance(result, nw.DataFrame)


def test_as_physical_with_lazy_frame():
    df = nw.from_native(pl.DataFrame({"a": [1, 2, 3]}).lazy())
    result = _as_physical(df)
    assert isinstance(result, nw.DataFrame)
    assert len(result) == 3
