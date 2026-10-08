import pytest

from pointblank.validate import Validate
from pointblank.column import (
    Column,
    ColumnExpression,
    ColumnLiteral,
    ColumnSelectorNarwhals,
    ReferenceColumn,
    col,
    ref,
    StartsWith,
    EndsWith,
    Contains,
    Matches,
    Everything,
    FirstN,
    LastN,
    starts_with,
    ends_with,
    contains,
    matches,
    everything,
    first_n,
    last_n,
    AndSelector,
    OrSelector,
    SubSelector,
    NotSelector,
    expr_col,
)

import pandas as pd
import polars as pl
import ibis
import narwhals.selectors as ncs


@pytest.fixture
def tbl_pl():
    return pl.DataFrame(
        {
            "a": [1, 2, None, 4, 5, None, None, 8, None],
            "b": [4, None, 6, 7, 8, None, None, None, 12],
            "c": [None, 8, 8, 8, 8, None, 8, None, None],
            "d": [None, 8, 8, 8, 8, None, 8, None, None],
            "e": [9, 9, 9, 9, 9, 9, 9, 9, 9],
            "f": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "g": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def tbl_pd():
    return pd.DataFrame(
        {
            "a": [1, 2, pd.NA, 4, 5, pd.NA, pd.NA, 8, pd.NA],
            "b": [4, pd.NA, 6, 7, 8, pd.NA, pd.NA, pd.NA, 12],
            "c": [pd.NA, 8, 8, 8, 8, pd.NA, 8, pd.NA, pd.NA],
            "d": [pd.NA, 8, 8, 8, 8, pd.NA, 8, pd.NA, pd.NA],
            "e": [9, 9, 9, 9, 9, 9, 9, 9, 9],
            "f": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            "g": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def tbl_memtable():
    return ibis.memtable(
        pd.DataFrame(
            {
                "a": [1, 2, pd.NA, 4, 5, pd.NA, pd.NA, 8, pd.NA],
                "b": [4, pd.NA, 6, 7, 8, pd.NA, pd.NA, pd.NA, 12],
                "c": [pd.NA, 8, 8, 8, 8, pd.NA, 8, pd.NA, pd.NA],
                "d": [pd.NA, 8, 8, 8, 8, pd.NA, 8, pd.NA, pd.NA],
                "e": [9, 9, 9, 9, 9, 9, 9, 9, 9],
                "f": [1, 1, 1, 1, 1, 1, 1, 1, 1],
                "g": [1, 1, 1, 1, 1, 1, 1, 1, 1],
            }
        )
    )


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
def df_pd():
    return pd.DataFrame(
        {
            "a": [5, 7, None, 3, 9, 4],
            "b": [6, 3, 0, 5, 8, 2],
            "c": [10, 4, 8, 9, 10, 5],
            "d": [8, 2, None, 1, 7, 6],
            "e": [True, False, True, False, True, False],
            "f": [False, True, False, True, False, True],
        }
    )


@pytest.fixture
def df_pl():
    return pl.DataFrame(
        {
            "a": [5, 7, None, 3, 9, 4],
            "b": [6, 3, 0, 5, 8, 2],
            "c": [10, 4, 8, 9, 10, 5],
            "d": [8, 2, None, 1, 7, 6],
            "e": [True, False, True, False, True, False],
            "f": [False, True, False, True, False, True],
        }
    )


@pytest.fixture
def df_ibis():
    df = pd.DataFrame(
        {
            "a": [5, 7, None, 3, 9, 4],
            "b": [6, 3, 0, 5, 8, 2],
            "c": [10, 4, 8, 9, 10, 5],
            "d": [8, 2, None, 1, 7, 6],
            "e": [True, False, True, False, True, False],
            "f": [False, True, False, True, False, True],
        }
    )
    return ibis.memtable(df)


def test_column_class():
    col1 = Column("col1")
    assert col1.exprs == "col1"
    assert str(col1) == "col1"


def test_col_function():
    col1 = col("col1")
    assert col1.exprs == "col1"
    assert str(col1) == "col1"


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_gt_col(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_gt(columns="b", value=col("a"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="b", value=col("a"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="a", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="a", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_gt(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    # assert (
    #     Validate(tbl)
    #     .col_vals_gt(columns="e", value=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1, scalar=True)
    #     == 5
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_lt_col(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_lt(columns="a", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="a", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="b", value=col("a"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="b", value=col("a"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_lt(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    # assert (
    #     Validate(tbl)
    #     .col_vals_lt(columns="e", value=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1, scalar=True)
    #     == 0
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_eq_col(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_eq(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="e", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="e", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="d", value=col("e"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="e", value=col("f"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="e", value=col("f"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="f", value=col("g"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_eq(columns="f", value=col("g"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_ne_col(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_ne(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 4
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="d", value=col("e"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=col("f"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=col("f"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="d", value=7, na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="d", value=7, na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=10, na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=10, na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=9, na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_ne(columns="e", value=9, na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_ge_col(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_ge(columns="b", value=col("a"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="b", value=col("a"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="d", value=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="d", value=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_ge(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    # assert (
    #     Validate(tbl)
    #     .col_vals_ge(columns="e", value=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1, scalar=True)
    #     == 5
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_le_col(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_le(columns="a", value=col("b"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="a", value=col("b"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="b", value=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="b", value=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="c", value=col("e"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_le(columns="d", value=col("e"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 5
    )
    # TODO: Fix this Pandas failure (TypeError: boolean value of NA is ambiguous)
    # assert (
    #    Validate(tbl)
    #    .col_vals_le(columns="e", value=col("d"), na_pass=False)
    #    .interrogate()
    #    .n_passed(i=1, scalar=True)
    #    == 0
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_between_col(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=0, right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=0, right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=8, na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("a"), right=8, na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="c", left=col("b"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="c", left=col("b"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("c"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_between(columns="b", left=col("c"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 7
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_between(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    # TODO: Fix this Pandas failure (TypeError: boolean value of NA is ambiguous)
    # assert (
    #     Validate(tbl)
    #     .col_vals_between(columns="e", left=col("c"), right=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1, scalar=True)
    #     == 0
    # )


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd", "tbl_pl", "tbl_memtable"])
def test_col_vals_outside_col(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 7
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=col("c"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=0, right=col("c"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=0, right=col("c"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=8, na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("a"), right=8, na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 1
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("a"), right=8, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 7
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="c", left=col("b"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="c", left=col("b"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=col("d"), inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 0
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="c", left=col("b"), right=9, inclusive=(True, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 6
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("c"), right=col("d"), na_pass=False)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 2
    )
    assert (
        Validate(tbl)
        .col_vals_outside(columns="b", left=col("c"), right=col("d"), na_pass=True)
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 8
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=False
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 3
    )
    assert (
        Validate(tbl)
        .col_vals_outside(
            columns="b", left=col("c"), right=col("d"), inclusive=(False, False), na_pass=True
        )
        .interrogate()
        .n_passed(i=1, scalar=True)
        == 9
    )
    # TODO: Fix this Pandas failure (TypeError: boolean value of NA is ambiguous)
    # assert (
    #     Validate(tbl)
    #     .col_vals_between(columns="e", left=col("c"), right=col("d"), na_pass=False)
    #     .interrogate()
    #     .n_passed(i=1, scalar=True)
    #     == 0
    # )


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_selector_classes(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    # StartsWith tests

    assert Column(exprs=StartsWith("low")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
    ]
    assert Column(exprs=StartsWith("LOW")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
    ]
    assert Column(exprs=StartsWith("LOW", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert Column(exprs=StartsWith("not_present")).resolve(columns=tbl.columns) == []
    assert Column(exprs=StartsWith("low")).exprs == StartsWith(text="low")

    # EndsWith tests

    assert Column(exprs=EndsWith("floats")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]
    assert Column(exprs=EndsWith("FLOATS")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]
    assert Column(exprs=EndsWith("FLOATS", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert Column(exprs=EndsWith("not_present")).resolve(columns=tbl.columns) == []
    assert Column(exprs=EndsWith("floats")).exprs == EndsWith(text="floats")

    # Contains tests

    assert Column(exprs=Contains("numbers")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert Column(exprs=Contains("NUMBERS")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert Column(exprs=Contains("NUMBERS", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert Column(exprs=Contains("not_present")).resolve(columns=tbl.columns) == []
    assert Column(exprs=Contains("numbers")).exprs == Contains(text="numbers")

    # Matches tests

    assert Column(exprs=Matches("at")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
    ]
    assert Column(exprs=Matches("numb")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert Column(exprs=Matches("NUMB")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert Column(exprs=Matches("NUMB", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert Column(exprs=Matches(r"^..._numbers$")).resolve(columns=tbl.columns) == ["low_numbers"]
    assert Column(exprs=Matches("not_present")).resolve(columns=tbl.columns) == []
    assert Column(exprs=Matches("at")).exprs == Matches(pattern="at")

    # Everything tests

    assert Column(exprs=Everything()).resolve(columns=tbl.columns) == list(tbl.columns)
    assert Column(exprs=Everything()).exprs == Everything()

    # FirstN tests

    assert Column(exprs=FirstN(3)).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
    ]
    assert Column(exprs=FirstN(9)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert Column(exprs=FirstN(20)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert Column(exprs=FirstN(0)).resolve(columns=tbl.columns) == []
    assert Column(exprs=FirstN(3, offset=1)).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
        "low_floats",
    ]
    assert Column(exprs=FirstN(3, offset=10)).resolve(columns=tbl.columns) == []
    assert Column(exprs=FirstN(0, offset=3)).resolve(columns=tbl.columns) == []
    assert Column(exprs=FirstN(0, offset=10)).resolve(columns=tbl.columns) == []
    assert Column(exprs=FirstN(5, offset=-1)).resolve(columns=tbl.columns) == []

    assert Column(exprs=FirstN(3)).exprs == FirstN(n=3)

    # LastN tests

    assert Column(exprs=LastN(3)).resolve(columns=tbl.columns) == [
        "date",
        "datetime",
        "bools",
    ]
    assert Column(exprs=LastN(9)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert Column(exprs=LastN(20)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert Column(exprs=LastN(0)).resolve(columns=tbl.columns) == []
    assert Column(exprs=LastN(3, offset=1)).resolve(columns=tbl.columns) == [
        "superhigh_floats",
        "date",
        "datetime",
    ]
    assert Column(exprs=LastN(3, offset=10)).resolve(columns=tbl.columns) == []
    assert Column(exprs=LastN(0, offset=3)).resolve(columns=tbl.columns) == []
    assert Column(exprs=LastN(0, offset=10)).resolve(columns=tbl.columns) == []
    assert Column(exprs=LastN(5, offset=-1)).resolve(columns=tbl.columns) == []

    assert Column(exprs=LastN(3)).exprs == LastN(n=3)


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_selector_helper_functions(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert col("not_present").resolve(columns=tbl.columns) == ["not_present"]

    # starts_with() tests

    assert starts_with("low").resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
    ]
    assert col(starts_with("low")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
    ]
    assert col(starts_with("LOW")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
    ]
    assert col(starts_with("LOW", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert col(starts_with("not_present")).resolve(columns=tbl.columns) == []
    assert col(starts_with("low")).exprs == StartsWith(text="low")

    # ends_with() tests

    assert ends_with("floats").resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]
    assert col(ends_with("floats")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]
    assert col(ends_with("FLOATS")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]
    assert col(ends_with("FLOATS", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert col(ends_with("not_present")).resolve(columns=tbl.columns) == []
    assert col(ends_with("floats")).exprs == EndsWith(text="floats")

    # contains() tests

    assert contains("numbers").resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert col(contains("numbers")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert col(contains("NUMBERS")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert col(contains("NUMBERS", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert col(contains("not_present")).resolve(columns=tbl.columns) == []
    assert col(contains("numbers")).exprs == Contains(text="numbers")

    # matches() tests

    assert matches("at").resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
    ]
    assert col(matches("at")).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
    ]
    assert col(matches("numb")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert col(matches("NUMB")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
    ]
    assert col(matches("NUMB", case_sensitive=True)).resolve(columns=tbl.columns) == []
    assert col(matches(r"^..._numbers$")).resolve(columns=tbl.columns) == ["low_numbers"]
    assert col(matches("not_present")).resolve(columns=tbl.columns) == []
    assert col(matches("at")).exprs == Matches(pattern="at")

    # everything() tests

    assert col(everything()).resolve(columns=tbl.columns) == list(tbl.columns)
    assert col(everything()).exprs == Everything()

    # first_n() tests

    assert col(first_n(3)).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
    ]
    assert col(first_n(9)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert col(first_n(20)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert col(first_n(0)).resolve(columns=tbl.columns) == []
    assert col(first_n(3, offset=1)).resolve(columns=tbl.columns) == [
        "low_numbers",
        "high_numbers",
        "low_floats",
    ]
    assert col(first_n(3, offset=10)).resolve(columns=tbl.columns) == []
    assert col(first_n(0, offset=3)).resolve(columns=tbl.columns) == []
    assert col(first_n(0, offset=10)).resolve(columns=tbl.columns) == []
    assert col(first_n(5, offset=-1)).resolve(columns=tbl.columns) == []

    assert col(first_n(3)).exprs == FirstN(n=3)

    # last_n() tests

    assert col(last_n(3)).resolve(columns=tbl.columns) == [
        "date",
        "datetime",
        "bools",
    ]
    assert col(last_n(9)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert col(last_n(20)).resolve(columns=tbl.columns) == list(tbl.columns)
    assert col(last_n(0)).resolve(columns=tbl.columns) == []
    assert col(last_n(3, offset=1)).resolve(columns=tbl.columns) == [
        "superhigh_floats",
        "date",
        "datetime",
    ]
    assert col(last_n(3, offset=10)).resolve(columns=tbl.columns) == []
    assert col(last_n(0, offset=3)).resolve(columns=tbl.columns) == []
    assert col(last_n(0, offset=10)).resolve(columns=tbl.columns) == []
    assert col(last_n(5, offset=-1)).resolve(columns=tbl.columns) == []

    assert col(last_n(3)).exprs == LastN(n=3)


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_selector_set_ops_classes(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert Column(exprs=AndSelector(StartsWith("low"), EndsWith("floats"))).resolve(
        columns=tbl.columns
    ) == ["low_floats"]

    assert Column(exprs=OrSelector(StartsWith("low"), EndsWith("floats"))).resolve(
        columns=tbl.columns
    ) == ["low_numbers", "low_floats", "high_floats", "superhigh_floats"]

    assert Column(exprs=SubSelector(StartsWith("low"), EndsWith("floats"))).resolve(
        columns=tbl.columns
    ) == ["low_numbers"]

    assert Column(exprs=NotSelector(StartsWith("low"))).resolve(columns=tbl.columns) == [
        "word",
        "high_numbers",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert Column(exprs=NotSelector(EndsWith("floats"))).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
        "date",
        "datetime",
        "bools",
    ]


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_selector_set_ops_functions(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert col(starts_with("low") & ends_with("floats")).resolve(columns=tbl.columns) == [
        "low_floats"
    ]

    assert col(starts_with("low") | ends_with("floats")).resolve(columns=tbl.columns) == [
        "low_numbers",
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]

    assert col(starts_with("low") - ends_with("floats")).resolve(columns=tbl.columns) == [
        "low_numbers"
    ]

    assert col(~starts_with("low")).resolve(columns=tbl.columns) == [
        "word",
        "high_numbers",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert col(~ends_with("floats")).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
        "date",
        "datetime",
        "bools",
    ]

    assert col(~contains("numbers")).resolve(columns=tbl.columns) == [
        "word",
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert col(~matches("at")).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
        "bools",
    ]

    assert col(~everything()).resolve(columns=tbl.columns) == []

    assert col(~first_n(3)).resolve(columns=tbl.columns) == [
        "low_floats",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert col(~first_n(3, offset=1)).resolve(columns=tbl.columns) == [
        "word",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert col(~last_n(3)).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
        "low_floats",
        "high_floats",
        "superhigh_floats",
    ]

    assert col(~last_n(3, offset=1)).resolve(columns=tbl.columns) == [
        "word",
        "low_numbers",
        "high_numbers",
        "low_floats",
        "high_floats",
        "bools",
    ]


@pytest.mark.parametrize(
    "tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names", "tbl_memtable_variable_names"]
)
def test_selector_set_ops_functions_complex(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    assert col(starts_with("low") & ends_with("floats") | contains("numbers")).resolve(
        columns=tbl.columns
    ) == ["low_numbers", "high_numbers", "low_floats"]

    assert col(
        starts_with("low") & ends_with("floats") | contains("numbers") - matches("numb")
    ).resolve(columns=tbl.columns) == ["low_floats"]

    assert col(ends_with("floats") | starts_with("low") - matches("numb")).resolve(
        columns=tbl.columns
    ) == ["low_floats", "high_floats", "superhigh_floats"]

    assert col((starts_with("low") & ends_with("floats")) | contains("numbers")).resolve(
        columns=tbl.columns
    ) == ["low_numbers", "high_numbers", "low_floats"]

    assert col(everything() - (starts_with("low") & ends_with("floats"))).resolve(
        columns=tbl.columns
    ) == [
        "word",
        "low_numbers",
        "high_numbers",
        "high_floats",
        "superhigh_floats",
        "date",
        "datetime",
        "bools",
    ]

    assert col(everything() - starts_with("low") & ends_with("floats")).resolve(
        columns=tbl.columns
    ) == [
        "high_floats",
        "superhigh_floats",
    ]


@pytest.mark.parametrize("tbl_fixture", ["tbl_pd_variable_names", "tbl_pl_variable_names"])
def test_nw_selectors(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    # Test with single Narwhals column selector
    validation = Validate(data=tbl).col_exists(columns=ncs.numeric()).interrogate()

    assert len(validation.n()) == 5

    # Test with single Narwhals column selector within `col()`
    validation = Validate(data=tbl).col_exists(columns=col(ncs.numeric())).interrogate()

    assert len(validation.n()) == 5

    # Test that multiple Narwhals column selectors work within `col()`
    validation = (
        Validate(data=tbl).col_exists(columns=col(ncs.numeric() | ncs.boolean())).interrogate()
    )

    assert len(validation.n()) == 6


def test_expr_col_creation():
    # Test basic creation
    col_expr = expr_col("a")
    assert isinstance(col_expr, ColumnExpression)
    assert col_expr.column_name == "a"
    assert col_expr.operation is None
    assert col_expr.left is None
    assert col_expr.right is None


def test_comparison_operators():
    # Test greater than
    expr = expr_col("a") > 5
    assert expr.operation == "gt"
    assert expr.left.column_name == "a"
    assert expr.right == 5

    # Test less than
    expr = expr_col("a") < 5
    assert expr.operation == "lt"
    assert expr.left.column_name == "a"
    assert expr.right == 5

    # Test equal to
    expr = expr_col("a") == 5
    assert expr.operation == "eq"
    assert expr.left.column_name == "a"
    assert expr.right == 5

    # Test not equal to
    expr = expr_col("a") != 5
    assert expr.operation == "ne"
    assert expr.left.column_name == "a"
    assert expr.right == 5

    # Test greater than or equal to
    expr = expr_col("a") >= 5
    assert expr.operation == "ge"
    assert expr.left.column_name == "a"
    assert expr.right == 5

    # Test less than or equal to
    expr = expr_col("a") <= 5
    assert expr.operation == "le"
    assert expr.left.column_name == "a"
    assert expr.right == 5


def test_arithmetic_operators():
    # Test addition
    expr = expr_col("a") + 5
    assert expr.operation == "add"
    assert expr.left.column_name == "a"
    assert expr.right == 5

    # Test subtraction
    expr = expr_col("a") - 5
    assert expr.operation == "sub"
    assert expr.left.column_name == "a"
    assert expr.right == 5

    # Test multiplication
    expr = expr_col("a") * 5
    assert expr.operation == "mul"
    assert expr.left.column_name == "a"
    assert expr.right == 5

    # Test division
    expr = expr_col("a") / 5
    assert expr.operation == "div"
    assert expr.left.column_name == "a"
    assert expr.right == 5


def test_column_to_column_operations():
    # Test comparison between columns
    expr = expr_col("a") > expr_col("b")
    assert expr.operation == "gt"
    assert expr.left.column_name == "a"
    assert expr.right.column_name == "b"

    # Test arithmetic between columns
    expr = expr_col("a") + expr_col("b")
    assert expr.operation == "add"
    assert expr.left.column_name == "a"
    assert expr.right.column_name == "b"


def test_complex_expressions():
    # Test complex nested expression
    expr = (expr_col("a") + expr_col("b")) > expr_col("c")
    assert expr.operation == "gt"
    assert expr.left.operation == "add"
    assert expr.left.left.column_name == "a"
    assert expr.left.right.column_name == "b"
    assert expr.right.column_name == "c"

    # Test another complex expression
    expr = expr_col("a") > (expr_col("b") + expr_col("c"))
    assert expr.operation == "gt"
    assert expr.left.column_name == "a"
    assert expr.right.operation == "add"
    assert expr.right.left.column_name == "b"
    assert expr.right.right.column_name == "c"


def test_null_operations():
    # Test is_null
    expr = expr_col("a").is_null()
    assert expr.operation == "is_null"
    assert expr.left.column_name == "a"
    assert expr.right is None

    # Test is_not_null
    expr = expr_col("a").is_not_null()
    assert expr.operation == "is_not_null"
    assert expr.left.column_name == "a"
    assert expr.right is None


def test_logical_operations():
    # Test AND
    expr = (expr_col("a") > 5) & (expr_col("b") < 10)
    assert expr.operation == "and"
    assert expr.left.operation == "gt"
    assert expr.right.operation == "lt"

    # Test OR
    expr = (expr_col("a") > 5) | (expr_col("b") < 10)
    assert expr.operation == "or"
    assert expr.left.operation == "gt"
    assert expr.right.operation == "lt"


def test_to_polars_expr():
    # Test basic column reference
    expr = expr_col("a")
    polars_expr = expr.to_polars_expr()
    assert 'col("a")' in str(polars_expr)  # More flexible check

    # Test comparison
    expr = expr_col("a") > 5
    polars_expr = expr.to_polars_expr()
    # Use a more flexible assertion that just checks for key components
    assert 'col("a")' in str(polars_expr)
    assert ">" in str(polars_expr)
    assert "5" in str(polars_expr)

    # Test arithmetic
    expr = expr_col("a") + expr_col("b")
    polars_expr = expr.to_polars_expr()
    assert 'col("a")' in str(polars_expr)
    assert 'col("b")' in str(polars_expr)
    assert "+" in str(polars_expr)

    # Test complex expression
    expr = (expr_col("a") + expr_col("b")) < expr_col("c")
    polars_expr = expr.to_polars_expr()
    assert 'col("a")' in str(polars_expr)
    assert 'col("b")' in str(polars_expr)
    assert 'col("c")' in str(polars_expr)
    assert "+" in str(polars_expr)
    assert "<" in str(polars_expr)

    # Test is_null
    expr = expr_col("a").is_null()
    polars_expr = expr.to_polars_expr()
    assert 'col("a")' in str(polars_expr)
    assert "is_null" in str(polars_expr)

    # Test is_not_null
    expr = expr_col("a").is_not_null()
    polars_expr = expr.to_polars_expr()
    assert 'col("a")' in str(polars_expr)
    assert "is_not_null" in str(polars_expr)

    # Test logical operations
    expr = (expr_col("a") > 5) & (expr_col("b") < 10)  # Add parentheses!
    polars_expr = expr.to_polars_expr()
    assert 'col("a")' in str(polars_expr)
    assert 'col("b")' in str(polars_expr)
    assert ">" in str(polars_expr)
    assert "<" in str(polars_expr)
    assert "&" in str(polars_expr)
    assert "5" in str(polars_expr)
    assert "10" in str(polars_expr)


def test_to_pandas_expr(df_pd):
    # Test basic column reference
    expr = expr_col("a")
    pandas_expr = expr.to_pandas_expr(df_pd)
    assert isinstance(pandas_expr, pd.Series)
    assert pandas_expr.equals(df_pd["a"])

    # Test comparison
    expr = expr_col("a") > 5
    pandas_expr = expr.to_pandas_expr(df_pd)
    assert isinstance(pandas_expr, pd.Series)
    assert pandas_expr.equals(df_pd["a"] > 5)

    # Test arithmetic
    expr = expr_col("a") + expr_col("b")
    pandas_expr = expr.to_pandas_expr(df_pd)
    assert isinstance(pandas_expr, pd.Series)
    assert pandas_expr.equals(df_pd["a"] + df_pd["b"])

    # Test complex expression
    expr = (expr_col("a") + expr_col("b")) < expr_col("c")
    pandas_expr = expr.to_pandas_expr(df_pd)
    assert isinstance(pandas_expr, pd.Series)
    assert pandas_expr.equals((df_pd["a"] + df_pd["b"]) < df_pd["c"])


def test_is_null_pandas_not_supported(df_pd):
    # Test that is_null raises NotImplementedError for pandas
    expr = expr_col("a").is_null()
    with pytest.raises(NotImplementedError):
        expr.to_pandas_expr(df_pd)

    # Test that is_not_null raises NotImplementedError for pandas
    expr = expr_col("a").is_not_null()
    with pytest.raises(NotImplementedError):
        expr.to_pandas_expr(df_pd)


def test_to_ibis_expr(df_ibis):
    # Test basic column reference
    expr = expr_col("a")
    ibis_expr = expr.to_ibis_expr(df_ibis)
    assert ibis_expr.get_name() == "a"

    # # Test comparison
    # expr = expr_col("a") > 5
    # ibis_expr = expr.to_ibis_expr(df_ibis)
    # assert str(ibis_expr.op()) == "Greater(ref_0, Literal(5))"

    # # Test arithmetic
    # expr = expr_col("a") + expr_col("b")
    # ibis_expr = expr.to_ibis_expr(df_ibis)
    # assert str(ibis_expr.op()).startswith("Add(")

    # # Test complex expression
    # expr = (expr_col("a") + expr_col("b")) < expr_col("c")
    # ibis_expr = expr.to_ibis_expr(df_ibis)
    # assert str(ibis_expr.op()).startswith("Less(")

    # # Test is_null
    # expr = expr_col("a").is_null()
    # ibis_expr = expr.to_ibis_expr(df_ibis)
    # assert str(ibis_expr.op()).startswith("IsNull(")

    # # Test is_not_null
    # expr = expr_col("a").is_not_null()
    # ibis_expr = expr.to_ibis_expr(df_ibis)
    # assert str(ibis_expr.op()).startswith("Not(")

    # # Test logical operations
    # expr = expr_col("a") > 5 & expr_col("b") < 10
    # ibis_expr = expr.to_ibis_expr(df_ibis)
    # assert str(ibis_expr.op()).startswith("And(")

    # expr = expr_col("a") > 5 | expr_col("b") < 10
    # ibis_expr = expr.to_ibis_expr(df_ibis)
    # assert str(ibis_expr.op()).startswith("Or(")


def test_invalid_operations():
    # Test invalid operation
    expr = ColumnExpression(column_name="a", operation="invalid", left=expr_col("a"), right=5)
    with pytest.raises(ValueError, match="Unsupported operation"):
        expr.to_polars_expr()

    with pytest.raises(ValueError, match="Unsupported operation"):
        expr.to_pandas_expr(pd.DataFrame({"a": [1, 2, 3]}))

    with pytest.raises(ValueError, match="Unsupported operation"):
        expr.to_ibis_expr(ibis.memtable(pd.DataFrame({"a": [1, 2, 3]})))

    # Test invalid state
    expr = ColumnExpression(operation=None, left=None, right=None)
    with pytest.raises(ValueError, match="Invalid expression state"):
        expr.to_polars_expr()

    with pytest.raises(ValueError, match="Invalid expression state"):
        expr.to_ibis_expr(ibis.memtable(pd.DataFrame({"a": [1, 2, 3]})))


def test_polars_evaluation(df_pl):
    # Test greater than
    expr = expr_col("a") > 5
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 2

    # Test column to column comparison
    expr = expr_col("a") > expr_col("b")
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 3

    # Test arithmetic
    expr = expr_col("a") + expr_col("b") < expr_col("c")
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 1

    # Test is_null
    expr = expr_col("a").is_null()
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 1

    # Test complex expression
    expr = (expr_col("a") > 4) & (expr_col("b") < 6)
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 1


def test_pandas_evaluation(df_pd):
    # Test greater than
    expr = expr_col("a") > 5
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 2

    # Test column to column comparison
    expr = expr_col("a") > expr_col("b")
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 3

    # Test arithmetic
    expr = expr_col("a") + expr_col("b") < expr_col("c")
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 1

    # TODO: fix as errors with 'ValueError: Unsupported operation: and'
    # # Test complex expression
    # expr = (expr_col("a") > 4) & (expr_col("b") < 6)
    # result = df_pd[expr.to_pandas_expr(df_pd)]
    # assert len(result) == 2


def test_ibis_evaluation(df_ibis):
    # Test greater than
    expr = expr_col("a") > 5
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 2

    # Test column to column comparison
    expr = expr_col("a") > expr_col("b")
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 3

    # Test arithmetic
    expr = expr_col("a") + expr_col("b") < expr_col("c")
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 1

    # Test is_null
    expr = expr_col("a").is_null()
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 1

    # Test complex expression
    expr = (expr_col("a") > 4) & (expr_col("b") < 6)
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 1


def test_eq_operation_polars(df_pl):
    # Test equality operation
    expr = expr_col("a") == 5
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 1
    assert result["a"][0] == 5

    # Test equality between columns
    expr = expr_col("a") == expr_col("b")
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 0  # No rows where a equals b


def test_eq_operation_pandas(df_pd):
    # Test equality operation
    expr = expr_col("a") == 5
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 1
    assert result["a"].iloc[0] == 5

    # Test equality between columns
    expr = expr_col("a") == expr_col("b")
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 0  # No rows where a equals b


def test_eq_operation_ibis(df_ibis):
    # Test equality operation
    expr = expr_col("a") == 5
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 1
    assert result["a"].iloc[0] == 5

    # Test equality between columns
    expr = expr_col("a") == expr_col("b")
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 0  # No rows where a equals b


def test_ne_operation_polars(df_pl):
    # Test inequality operation
    expr = expr_col("a") != 5
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 4  # All non-null rows except where a=5

    # Test inequality between columns
    expr = expr_col("a") != expr_col("b")
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 5  # All non-null rows since no a equals b


def test_ne_operation_pandas(df_pd):
    # Test inequality operation
    expr = expr_col("a") != 5
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 5  # All non-null rows except where a=5

    # Test inequality between columns
    expr = expr_col("a") != expr_col("b")
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 6  # All non-null rows since no a equals b


def test_ne_operation_ibis(df_ibis):
    # Test inequality operation
    expr = expr_col("a") != 5
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 4  # All non-null rows except where a=5

    # Test inequality between columns
    expr = expr_col("a") != expr_col("b")
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 5  # All non-null rows since no a equals b


def test_ge_operation_polars(df_pl):
    # Test greater than or equal operation
    expr = expr_col("a") >= 5
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 3  # Rows where a >= 5

    # Test greater than or equal between columns
    expr = expr_col("a") >= expr_col("b")
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 3  # Rows where a >= b


def test_ge_operation_pandas(df_pd):
    # Test greater than or equal operation
    expr = expr_col("a") >= 5
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 3  # Rows where a >= 5

    # Test greater than or equal between columns
    expr = expr_col("a") >= expr_col("b")
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 3  # Rows where a >= b


def test_ge_operation_ibis(df_ibis):
    # Test greater than or equal operation
    expr = expr_col("a") >= 5
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 3  # Rows where a >= 5

    # Test greater than or equal between columns
    expr = expr_col("a") >= expr_col("b")
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 3  # Rows where a >= b


def test_le_operation_polars(df_pl):
    # Test less than or equal operation
    expr = expr_col("a") <= 5
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 3  # Rows where a <= 5

    # Test less than or equal between columns
    expr = expr_col("a") <= expr_col("b")
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 2  # Rows where a <= b


def test_le_operation_pandas(df_pd):
    # Test less than or equal operation
    expr = expr_col("a") <= 5
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 3  # Rows where a <= 5

    # Test less than or equal between columns
    expr = expr_col("a") <= expr_col("b")
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 2  # Rows where a <= b


def test_le_operation_ibis(df_ibis):
    # Test less than or equal operation
    expr = expr_col("a") <= 5
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 3  # Rows where a <= 5

    # Test less than or equal between columns
    expr = expr_col("a") <= expr_col("b")
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 2  # Rows where a <= b


def test_sub_operation_polars(df_pl):
    # Test subtraction operation with filter
    expr = (expr_col("a") - 3) > 2
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 2  # Rows where (a - 3) > 2

    # Test subtraction between columns
    expr = (expr_col("a") - expr_col("b")) > 0
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 3  # Rows where (a - b) > 0

    # Test direct subtraction result
    expr = expr_col("a") - expr_col("b")
    result = df_pl.with_columns(result=expr.to_polars_expr())
    assert result["result"][1] == 4  # 7 - 3 = 4


def test_sub_operation_pandas(df_pd):
    # Test subtraction operation with filter
    expr = (expr_col("a") - 3) > 2
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 2  # Rows where (a - 3) > 2

    # Test subtraction between columns
    expr = (expr_col("a") - expr_col("b")) > 0
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 3  # Rows where (a - b) > 0

    # Test direct subtraction result
    expr = expr_col("a") - expr_col("b")
    result = expr.to_pandas_expr(df_pd)
    assert result.iloc[1] == 4  # 7 - 3 = 4


def test_sub_operation_ibis(df_ibis):
    # Test subtraction operation with filter
    expr = (expr_col("a") - 3) > 2
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 2  # Rows where (a - 3) > 2

    # Test subtraction between columns
    expr = (expr_col("a") - expr_col("b")) > 0
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 3  # Rows where (a - b) > 0


def test_mul_operation_polars(df_pl):
    # Test multiplication operation with filter
    expr = (expr_col("a") * 2) > 10
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 2  # Rows where (a * 2) > 10

    # Test multiplication between columns
    expr = (expr_col("a") * expr_col("b")) > 20
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 3  # Rows where (a * b) > 20

    # Test direct multiplication result
    expr = expr_col("a") * 2
    result = df_pl.with_columns(result=expr.to_polars_expr())
    assert result["result"][0] == 10  # 5 * 2 = 10


def test_mul_operation_pandas(df_pd):
    # Test multiplication operation with filter
    expr = (expr_col("a") * 2) > 10
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 2  # Rows where (a * 2) > 10

    # Test multiplication between columns
    expr = (expr_col("a") * expr_col("b")) > 20
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 3  # Rows where (a * b) > 20

    # Test direct multiplication result
    expr = expr_col("a") * 2
    result = expr.to_pandas_expr(df_pd)
    assert result.iloc[0] == 10  # 5 * 2 = 10


def test_mul_operation_ibis(df_ibis):
    # Test multiplication operation with filter
    expr = (expr_col("a") * 2) > 10
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 2  # Rows where (a * 2) > 10

    # Test multiplication between columns
    expr = (expr_col("a") * expr_col("b")) > 20
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 3  # Rows where (a * b) > 20


def test_div_operation_polars(df_pl):
    # Test division operation with filter
    expr = (expr_col("a") / 2) > 3
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 2  # Rows where (a / 2) > 3

    # Test division between columns
    expr = (expr_col("a") / expr_col("b")) > 1
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 3  # Rows where (a / b) > 1

    # Test direct division result
    expr = expr_col("a") / 5
    result = df_pl.with_columns(result=expr.to_polars_expr())
    assert result["result"][0] == 1.0  # 5 / 5 = 1.0


def test_div_operation_pandas(df_pd):
    # Test division operation with filter
    expr = (expr_col("a") / 2) > 3
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 2  # Rows where (a / 2) > 3

    # Test division between columns
    expr = (expr_col("a") / expr_col("b")) > 1
    result = df_pd[expr.to_pandas_expr(df_pd)]
    assert len(result) == 3  # Rows where (a / b) > 1

    # Test direct division result
    expr = expr_col("a") / 5
    result = expr.to_pandas_expr(df_pd)
    assert result.iloc[0] == 1.0  # 5 / 5 = 1.0


def test_div_operation_ibis(df_ibis):
    # Test division operation with filter
    expr = (expr_col("a") / 2) > 3
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 2  # Rows where (a / 2) > 3

    # Test division between columns
    expr = (expr_col("a") / expr_col("b")) > 1
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 3  # Rows where (a / b) > 1


def test_and_operation_polars(df_pl):
    # Test logical AND operation
    expr = (expr_col("a") > 5) & (expr_col("b") < 6)
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 1  # Rows where a > 5 AND b < 6

    # Test logical AND with boolean columns
    expr = expr_col("e") & expr_col("f")
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 0  # No rows where both e and f are True


def test_and_operation_ibis(df_ibis):
    # Test logical AND operation
    expr = (expr_col("a") > 5) & (expr_col("b") < 6)
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 1  # Rows where a > 5 AND b < 6

    # Test logical AND with boolean columns
    expr = expr_col("e") & expr_col("f")
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 0  # No rows where both e and f are True


def test_or_operation_polars(df_pl):
    # Test logical OR operation
    expr = (expr_col("a") > 8) | (expr_col("b") < 1)
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 2  # Rows where a > 8 OR b < 1

    # Test logical OR with boolean columns
    expr = expr_col("e") | expr_col("f")
    result = df_pl.filter(expr.to_polars_expr())
    assert len(result) == 6  # All rows have either e or f as True


def test_or_operation_ibis(df_ibis):
    # Test logical OR operation
    expr = (expr_col("a") > 8) | (expr_col("b") < 1)
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 2  # Rows where a > 8 OR b < 1

    # Test logical OR with boolean columns
    expr = expr_col("e") | expr_col("f")
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 6  # All rows have either e or f as True


def test_is_not_null_operation_ibis(df_ibis):
    """Test is_not_null operation specifically for Ibis backend."""
    # Test basic is_not_null operation
    expr = expr_col("a").is_not_null()
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 5  # Should match all non-null values in column a

    # Test is_not_null in combination with other operations
    expr = expr_col("a").is_not_null() & (expr_col("b") > 2)
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 4  # Non-null a values where b > 2

    # Test compound expression with is_not_null
    expr = (expr_col("a") > 4) | expr_col("a").is_not_null()
    result = df_ibis.filter(expr.to_ibis_expr(df_ibis)).execute()
    assert len(result) == 5  # All non-null values (condition always true for non-nulls)

    # Test that the result doesn't include null values
    has_nulls = any(pd.isna(result["a"]))
    assert not has_nulls


def test_and_operation_pandas_raises_error(df_pd):
    # Test that logical AND raises ValueError for pandas
    expr = (expr_col("a") > 5) & (expr_col("b") < 6)
    with pytest.raises(ValueError):
        df_pd[expr.to_pandas_expr(df_pd)]


def test_or_operation_pandas_raises_error(df_pd):
    # Test that logical OR raises ValueError for pandas
    expr = (expr_col("a") > 8) | (expr_col("b") < 1)
    with pytest.raises(ValueError):
        df_pd[expr.to_pandas_expr(df_pd)]


def test_ref_function():
    r = ref("my_column")
    assert isinstance(r, ReferenceColumn)
    assert r.column_name == "my_column"


def test_reference_column_repr():
    r = ref("sales")
    assert repr(r) == "ref('sales')"
    r2 = ReferenceColumn(column_name="total")
    assert repr(r2) == "ref('total')"


def test_column_repr_with_selector():
    c = Column(exprs=StartsWith("low"))
    assert repr(c) == repr(StartsWith(text="low"))


def test_column_literal_resolve_directly():
    literal = ColumnLiteral(exprs="col_a")
    result = literal.resolve(["col_a", "col_b", "col_c"])
    assert result == ["col_a"]
    assert literal.name == "col_a"
    assert repr(literal) == "col_a"


def test_column_literal_from_col():
    literal = col("my_col")
    assert isinstance(literal, ColumnLiteral)
    assert literal.name == "my_col"
    assert repr(literal) == "my_col"
    result = literal.resolve(["my_col", "other"])
    assert result == ["my_col"]


def test_column_selector_narwhals_resolve_without_table():
    selector = ColumnSelectorNarwhals(exprs=ncs.numeric())
    with pytest.raises(ValueError, match="ColumnSelectorNarwhals requires a table"):
        selector.resolve(columns=["a", "b"])


def test_column_selector_narwhals_resolve_with_table():
    tbl = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    validation = Validate(data=tbl).col_vals_ge(columns=ncs.numeric(), value=0).interrogate()
    assert validation.n_passed(i=1, scalar=True) == 2


def test_starts_with_standalone():
    sw = starts_with("low")
    result = sw.resolve(["low_num", "high_num", "low_float"])
    assert result == ["low_num", "low_float"]


def test_ends_with_standalone():
    ew = ends_with("_num")
    result = ew.resolve(["low_num", "high_num", "low_float"])
    assert result == ["low_num", "high_num"]


def test_contains_standalone():
    ct = contains("_num")
    result = ct.resolve(["low_num", "high_num", "low_float"])
    assert result == ["low_num", "high_num"]


def test_matches_standalone():
    mt = matches("^low")
    result = mt.resolve(["low_num", "high_num", "low_float"])
    assert result == ["low_num", "low_float"]
    empty = mt.resolve(["high_num", "high_float"])
    assert empty == []


def test_everything_standalone():
    cols = ["a", "b", "c"]
    ev = everything()
    assert ev.resolve(cols) == cols


def test_first_n_standalone():
    cols = ["a", "b", "c", "d"]
    fn = first_n(2)
    assert fn.resolve(cols) == ["a", "b"]


def test_last_n_standalone():
    cols = ["a", "b", "c", "d"]
    ln = last_n(2)
    assert ln.resolve(cols) == ["c", "d"]


def test_selector_and_operator():
    sw = starts_with("a")
    ew = ends_with("a")
    combined = sw & ew
    assert isinstance(combined, AndSelector)
    result = combined.resolve(["aa", "ab", "ba", "bb"])
    assert result == ["aa"]


def test_selector_or_operator():
    sw = starts_with("a")
    ew = ends_with("b")
    combined = sw | ew
    assert isinstance(combined, OrSelector)
    result = set(combined.resolve(["aa", "ab", "ba", "bb"]))
    assert result == {"aa", "ab", "bb"}


def test_selector_sub_operator():
    ev = everything()
    sw = starts_with("a")
    combined = ev - sw
    assert isinstance(combined, SubSelector)
    result = combined.resolve(["aa", "ab", "ba", "bb"])
    assert result == ["ba", "bb"]


def test_selector_invert_operator():
    sw = starts_with("a")
    combined = ~sw
    assert isinstance(combined, NotSelector)
    result = combined.resolve(["aa", "ab", "ba", "bb"])
    assert result == ["ba", "bb"]


def test_expr_col_invalid_polars_operation():
    expr = ColumnExpression(operation="invalid_op", left=ColumnExpression(column_name="a"), right=5)
    with pytest.raises(ValueError, match="Unsupported operation"):
        expr.to_polars_expr()


def test_expr_col_invalid_state_polars():
    expr = ColumnExpression(operation=None, column_name=None, left=None, right=5)
    with pytest.raises(ValueError, match="Invalid expression state"):
        expr.to_polars_expr()


def test_expr_col_to_pandas_base_case(df_pd):
    expr = expr_col("b")
    result = expr.to_pandas_expr(df_pd)
    assert len(result) == len(df_pd["b"])
    assert result[0] == df_pd["b"][0]


def test_expr_col_to_pandas_nested_expression(df_pd):
    expr = expr_col("b") + expr_col("c")
    result = expr.to_pandas_expr(df_pd)
    expected = df_pd["b"] + df_pd["c"]
    assert len(result) == len(expected)


def test_expr_col_is_null_pandas_raises(df_pd):
    expr = expr_col("a").is_null()
    with pytest.raises(NotImplementedError, match="is_null"):
        expr.to_pandas_expr(df_pd)


def test_expr_col_is_not_null_pandas_raises(df_pd):
    expr = expr_col("a").is_not_null()
    with pytest.raises(NotImplementedError, match="is_not_null"):
        expr.to_pandas_expr(df_pd)


def test_expr_col_to_ibis_base_case(df_ibis):
    expr = expr_col("a")
    result = expr.to_ibis_expr(df_ibis)
    assert result is not None


def test_expr_col_all_operators_create_expressions():
    a = expr_col("a")
    assert isinstance(a > 5, ColumnExpression)
    assert isinstance(a < 5, ColumnExpression)
    assert isinstance(a == 5, ColumnExpression)
    assert isinstance(a != 5, ColumnExpression)
    assert isinstance(a >= 5, ColumnExpression)
    assert isinstance(a <= 5, ColumnExpression)
    assert isinstance(a + 5, ColumnExpression)
    assert isinstance(a - 5, ColumnExpression)
    assert isinstance(a * 5, ColumnExpression)
    assert isinstance(a / 5, ColumnExpression)
    assert isinstance(a.is_null(), ColumnExpression)
    assert isinstance(a.is_not_null(), ColumnExpression)
    assert isinstance(a | expr_col("b"), ColumnExpression)
    assert isinstance(a & expr_col("b"), ColumnExpression)
