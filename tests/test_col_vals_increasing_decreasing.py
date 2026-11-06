import datetime
import math
from decimal import Decimal

import ibis
import pandas as pd
import polars as pl
import pytest

import pointblank as pb


def test_strictly_increasing_passes():
    """Test that strictly increasing values pass validation."""
    tbl = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    validation = pb.Validate(data=tbl).col_vals_increasing(columns="a").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 5
    assert validation.n_failed(i=1, scalar=True) == 0
    assert validation.f_passed(i=1, scalar=True) == 1.0


def test_strictly_increasing_with_stationary_fails():
    """Test that stationary values fail when allow_stationary=False."""
    tbl = pl.DataFrame({"a": [1, 2, 2, 3, 4]})

    validation = pb.Validate(data=tbl).col_vals_increasing(columns="a").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 4  # First value and values after stationary
    assert validation.n_failed(i=1, scalar=True) == 1  # The repeated value
    assert validation.f_failed(i=1, scalar=True) == 0.2


def test_allow_stationary_passes():
    """Test that stationary values pass when allow_stationary=True."""
    tbl = pl.DataFrame({"a": [1, 2, 2, 3, 4]})

    validation = (
        pb.Validate(data=tbl).col_vals_increasing(columns="a", allow_stationary=True).interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == 5
    assert validation.n_failed(i=1, scalar=True) == 0
    assert validation.f_passed(i=1, scalar=True) == 1.0


def test_increasing_decreasing_values_fail():
    """Test that decreasing values fail validation."""
    tbl = pl.DataFrame({"a": [1, 2, 3, 2, 4]})

    validation = pb.Validate(data=tbl).col_vals_increasing(columns="a").interrogate()

    assert validation.n_failed(i=1, scalar=True) == 1  # The value that decreased


def test_increasing_decreasing_tol_allows_small_decreases():
    """Test that decreasing_tol allows small decreases."""
    tbl = pl.DataFrame({"a": [10, 12, 11, 13, 15]})  # 12 to 11 is -1

    validation = (
        pb.Validate(data=tbl).col_vals_increasing(columns="a", decreasing_tol=1.0).interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == 5
    assert validation.n_failed(i=1, scalar=True) == 0


def test_increasing_decreasing_tol_fails_large_decreases():
    """Test that decreasing_tol still fails large decreases."""
    tbl = pl.DataFrame({"a": [10, 12, 8, 13, 15]})  # 12 to 8 is -4

    validation = (
        pb.Validate(data=tbl).col_vals_increasing(columns="a", decreasing_tol=1.0).interrogate()
    )

    assert validation.n_failed(i=1, scalar=True) == 1  # The large decrease


def test_increasing_na_pass_false_fails_na_values():
    """Test that NA values fail when na_pass=False."""
    tbl = pl.DataFrame({"a": [1, 2, None, 4, 5]})

    validation = pb.Validate(data=tbl).col_vals_increasing(columns="a", na_pass=False).interrogate()

    assert validation.n_failed(i=1, scalar=True) == 1  # The None value


def test_increasing_na_pass_true_passes_na_values():
    """Test that NA values pass when na_pass=True."""
    tbl = pl.DataFrame({"a": [1, 2, None, 4, 5]})

    validation = pb.Validate(data=tbl).col_vals_increasing(columns="a", na_pass=True).interrogate()

    assert validation.n_passed(i=1, scalar=True) == 5
    assert validation.n_failed(i=1, scalar=True) == 0


def test_increasing_multiple_columns():
    """Test validation on multiple columns."""
    tbl = pl.DataFrame({"a": [1, 2, 3], "b": [10, 20, 30], "c": [5, 5, 6]})

    validation = pb.Validate(data=tbl).col_vals_increasing(columns=["a", "b"]).interrogate()

    # Both columns should pass
    assert validation.n_passed(i=1, scalar=True) == 3  # Column a
    assert validation.n_passed(i=2, scalar=True) == 3  # Column b


def test_strictly_decreasing_passes():
    """Test that strictly decreasing values pass validation."""
    tbl = pl.DataFrame({"a": [5, 4, 3, 2, 1]})

    validation = pb.Validate(data=tbl).col_vals_decreasing(columns="a").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 5
    assert validation.n_failed(i=1, scalar=True) == 0
    assert validation.f_passed(i=1, scalar=True) == 1.0


def test_strictly_decreasing_with_stationary_fails():
    """Test that stationary values fail when allow_stationary=False."""
    tbl = pl.DataFrame({"a": [4, 3, 3, 2, 1]})

    validation = pb.Validate(data=tbl).col_vals_decreasing(columns="a").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 4  # First value and values after stationary
    assert validation.n_failed(i=1, scalar=True) == 1  # The repeated value


def test_decreasing_allow_stationary_passes():
    """Test that stationary values pass when allow_stationary=True."""
    tbl = pl.DataFrame({"a": [4, 3, 3, 2, 1]})

    validation = (
        pb.Validate(data=tbl).col_vals_decreasing(columns="a", allow_stationary=True).interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == 5
    assert validation.n_failed(i=1, scalar=True) == 0
    assert validation.f_passed(i=1, scalar=True) == 1.0


def test_decreasing_increasing_values_fail():
    """Test that increasing values fail validation."""
    tbl = pl.DataFrame({"a": [4, 3, 2, 3, 1]})

    validation = pb.Validate(data=tbl).col_vals_decreasing(columns="a").interrogate()

    assert validation.n_failed(i=1, scalar=True) == 1  # The value that increased


def test_decreasing_increasing_tol_allows_small_increases():
    """Test that increasing_tol allows small increases."""
    tbl = pl.DataFrame({"a": [15, 13, 14, 12, 10]})  # 13 to 14 is +1

    validation = (
        pb.Validate(data=tbl).col_vals_decreasing(columns="a", increasing_tol=1.0).interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == 5
    assert validation.n_failed(i=1, scalar=True) == 0


def test_decreasing_increasing_tol_fails_large_increases():
    """Test that increasing_tol still fails large increases."""
    tbl = pl.DataFrame({"a": [15, 13, 18, 12, 10]})  # 13 to 18 is +5

    validation = (
        pb.Validate(data=tbl).col_vals_decreasing(columns="a", increasing_tol=1.0).interrogate()
    )

    assert validation.n_failed(i=1, scalar=True) == 1  # The large increase


def test_decreasing_na_pass_false_fails_na_values():
    """Test that NA values fail when na_pass=False."""
    tbl = pl.DataFrame({"a": [5, 4, None, 2, 1]})

    validation = pb.Validate(data=tbl).col_vals_decreasing(columns="a", na_pass=False).interrogate()

    assert validation.n_failed(i=1, scalar=True) == 1  # The None value


def test_decreasing_na_pass_true_passes_na_values():
    """Test that NA values pass when na_pass=True."""
    tbl = pl.DataFrame({"a": [5, 4, None, 2, 1]})

    validation = pb.Validate(data=tbl).col_vals_decreasing(columns="a", na_pass=True).interrogate()

    assert validation.n_passed(i=1, scalar=True) == 5
    assert validation.n_failed(i=1, scalar=True) == 0


def test_decreasing_multiple_columns():
    """Test validation on multiple columns."""
    tbl = pl.DataFrame({"a": [3, 2, 1], "b": [30, 20, 10], "c": [6, 5, 5]})

    validation = pb.Validate(data=tbl).col_vals_decreasing(columns=["a", "b"]).interrogate()

    # Both columns should pass
    assert validation.n_passed(i=1, scalar=True) == 3  # Column a
    assert validation.n_passed(i=2, scalar=True) == 3  # Column b


# Edge case tests


def test_single_value_always_passes():
    """Test that a single value always passes (no previous value to compare)."""
    tbl = pl.DataFrame({"a": [5]})

    validation_inc = pb.Validate(data=tbl).col_vals_increasing(columns="a").interrogate()
    validation_dec = pb.Validate(data=tbl).col_vals_decreasing(columns="a").interrogate()

    assert validation_inc.n_passed(i=1, scalar=True) == 1
    assert validation_dec.n_passed(i=1, scalar=True) == 1


def test_two_values_increasing():
    """Test with exactly two values for increasing."""
    tbl = pl.DataFrame({"a": [1, 2]})

    validation = pb.Validate(data=tbl).col_vals_increasing(columns="a").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 2
    assert validation.n_failed(i=1, scalar=True) == 0


def test_two_values_decreasing():
    """Test with exactly two values for decreasing."""
    tbl = pl.DataFrame({"a": [2, 1]})

    validation = pb.Validate(data=tbl).col_vals_decreasing(columns="a").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 2
    assert validation.n_failed(i=1, scalar=True) == 0


def test_datetime_values_increasing():
    """Test with datetime values for increasing."""
    tbl = pl.DataFrame(
        {
            "date": [
                datetime.datetime(2023, 1, 1),
                datetime.datetime(2023, 1, 2),
                datetime.datetime(2023, 1, 3),
            ]
        }
    )

    validation = pb.Validate(data=tbl).col_vals_increasing(columns="date").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 3
    assert validation.n_failed(i=1, scalar=True) == 0


def test_float_values():
    """Test with float values."""
    tbl = pl.DataFrame({"a": [1.1, 2.2, 3.3, 4.4]})

    validation = pb.Validate(data=tbl).col_vals_increasing(columns="a").interrogate()

    assert validation.n_passed(i=1, scalar=True) == 4
    assert validation.n_failed(i=1, scalar=True) == 0


def test_increasing_decreasing_polars_lazyframe():
    """Test that Polars LazyFrames give the same results as DataFrames."""
    data = {"a": [1.0, 2.0, None, 5.0, 2.0], "b": [5, 4, 4, 1, 2]}

    for tbl in [pl.DataFrame(data), pl.LazyFrame(data)]:
        validation = (
            pb.Validate(data=tbl)
            .col_vals_increasing(columns="a")
            .col_vals_increasing(columns="a", na_pass=True)
            .col_vals_decreasing(columns="b")
            .col_vals_decreasing(columns="b", allow_stationary=True)
            .interrogate()
        )

        assert validation.n_passed(scalar=False) == {1: 3, 2: 4, 3: 3, 4: 4}
        assert validation.n_failed(scalar=False) == {1: 2, 2: 1, 3: 2, 4: 1}

        # The failing rows are extracted without the helper columns
        extract = validation.get_data_extracts(i=1, frame=True)
        assert extract["a"].to_list() == [None, 2.0]
        assert "pb_row_index_" not in extract.columns


def _ordering_test_tables(data: dict) -> dict:
    """The same data as Polars (eager and lazy), pandas, DuckDB, and memtable tables."""
    # An all-null column would get the `Null` dtype, which DuckDB can't store
    tbl = pl.DataFrame(data).cast({pl.Null: pl.Float64})
    return {
        "polars": tbl,
        "polars_lazy": tbl.lazy(),
        "pandas": tbl.to_pandas(),
        "duckdb": ibis.duckdb.connect().create_table("tbl", tbl),
        "memtable": ibis.memtable(tbl),
    }


# Rows are stored out of order; in `t` order, `a` is [1, 2, None, 5, 3] and `b` is [5, 4, None, 1, 3]
SHUFFLED = {"t": [4, 1, 3, 2, 5], "a": [5.0, 1.0, None, 2.0, 3.0], "b": [1.0, 5.0, None, 4.0, 3.0]}


@pytest.mark.parametrize("tbl_type", ["polars", "polars_lazy", "pandas", "duckdb", "memtable"])
def test_order_by(tbl_type):
    """`order_by=` sets the row order on every backend; the 5 is compared with the 2."""
    tbl = _ordering_test_tables(SHUFFLED)[tbl_type]

    validation = (
        pb.Validate(data=tbl)
        .col_vals_increasing(columns="a", order_by="t")
        .col_vals_increasing(columns="a", order_by=["t"], na_pass=True)
        .col_vals_decreasing(columns="b", order_by="t")
        .interrogate()
    )

    # In `t` order: 1 (first), 2, None, 5 (vs 2), 3 (fails vs 5); and for `b`, the 1 is compared
    # with the 4, and the 3 fails against the 1
    assert validation.n_passed(scalar=False) == {1: 3, 2: 4, 3: 3}
    assert validation.n_failed(scalar=False) == {1: 2, 2: 1, 3: 2}


def test_order_by_extracts_original_rows():
    """The failing rows are reported against the table's own rows."""
    validation = (
        pb.Validate(data=pl.DataFrame(SHUFFLED))
        .col_vals_increasing(columns="a", order_by="t")
        .interrogate()
    )

    extract = validation.get_data_extracts(i=1, frame=True)
    assert extract["_row_num_"].to_list() == [3, 5]
    assert extract["t"].to_list() == [3, 5]


@pytest.mark.parametrize("tbl_type", ["duckdb", "memtable"])
def test_no_row_order_without_order_by_is_eval_error(tbl_type):
    """Tables without a row order need `order_by=`; the other steps are still interrogated."""
    tbl = _ordering_test_tables(SHUFFLED)[tbl_type]

    validation = (
        pb.Validate(data=tbl)
        .col_vals_increasing(columns="a")
        .col_vals_decreasing(columns="b")
        .col_vals_not_null(columns="t")
        .interrogate()
    )

    for step in validation.validation_info[:2]:
        assert step.eval_error is True
        assert "order_by=" in step.notes["eval_error"]["text"]
    assert validation.n_passed(i=3, scalar=True) == 5


def test_order_by_missing_column_is_eval_error():
    validation = (
        pb.Validate(data=pl.DataFrame(SHUFFLED))
        .col_vals_increasing(columns="a", order_by=["t", "zzz"])
        .interrogate()
    )

    step = validation.validation_info[0]
    assert step.eval_error is True
    assert "'zzz'" in step.notes["eval_error"]["text"]


@pytest.mark.parametrize("order_by", [1, ["t", 2], ("t",)])
def test_order_by_invalid_input(order_by):
    with pytest.raises(ValueError, match="order_by="):
        pb.Validate(data=pl.DataFrame(SHUFFLED)).col_vals_increasing(columns="a", order_by=order_by)


@pytest.mark.parametrize(
    "values, n_passed, n_failed",
    [
        ([1, 2, None, 5, 3], 3, 2),  # the 5 is compared with the 2; the 3 fails against the 5
        ([5, None, 3], 1, 2),  # the 3 fails against the 5, across the null
        ([1, None, None, 4, 0], 2, 3),  # several nulls in a row
        ([None, None, 3, 4, 2], 2, 3),  # leading nulls fail; the 3 is the starting point
        ([None, None], 0, 2),
    ],
)
@pytest.mark.parametrize("tbl_type", ["polars", "polars_lazy", "pandas", "duckdb", "memtable"])
def test_compared_with_last_non_null_value(tbl_type, values, n_passed, n_failed):
    data = {"t": list(range(len(values))), "a": [None if v is None else float(v) for v in values]}
    tbl = _ordering_test_tables(data)[tbl_type]

    validation = (
        pb.Validate(data=tbl)
        .col_vals_increasing(columns="a", order_by="t")
        .col_vals_decreasing(columns="a", order_by="t", increasing_tol=100)
        .interrogate()
    )

    assert validation.n_passed(i=1, scalar=True) == n_passed
    assert validation.n_failed(i=1, scalar=True) == n_failed

    # Any increase is within the tolerance, so only the nulls fail
    n_null = values.count(None)
    assert validation.n_failed(i=2, scalar=True) == n_null


def test_internal_column_names_dont_collide():
    """User columns named like the helper columns are neither used nor dropped."""
    data = {
        "a": [1.0, 2.0, None, 5.0, 3.0],
        "pb_row_index_": [9, 9, 9, 9, 9],
        "pb_values_": [9, 9, 9, 9, 9],
        "pb_previous_": [9, 9, 9, 9, 9],
    }

    for tbl in [pl.DataFrame(data), pl.LazyFrame(data)]:
        validation = pb.Validate(data=tbl).col_vals_increasing(columns="a").interrogate()

        assert validation.n_passed(i=1, scalar=True) == 3
        assert validation.n_failed(i=1, scalar=True) == 2

        extract = validation.get_data_extracts(i=1, frame=True)
        for name in ["pb_row_index_", "pb_values_", "pb_previous_"]:
            assert extract[name].to_list() == [9, 9]


def test_order_by_serialization():
    validation = (
        pb.Validate(data=pl.DataFrame(SHUFFLED))
        .col_vals_increasing(columns="a", order_by="t")
        .col_vals_decreasing(columns="b", order_by=["t", "a"])
    )

    code = validation.to_code()
    assert "order_by='t'" in code or 'order_by="t"' in code
    assert "order_by=['t', 'a']" in code or 'order_by=["t", "a"]' in code

    yaml_text = validation.to_yaml()
    assert "order_by: t" in yaml_text

    # `set_tbl=` replaces the table, but the placeholder table name must still be loadable
    yaml_text = yaml_text.replace("tbl: your_data", "tbl: small_table")
    revalidated = pb.yaml_interrogate(yaml_text, set_tbl=pl.DataFrame(SHUFFLED))
    assert revalidated.n_passed(scalar=False) == {1: 3, 2: 3}


ALL_TBL_TYPES = ["polars", "polars_lazy", "pandas", "pyarrow", "duckdb", "memtable"]


def _typed_test_table(tbl_type: str, values: list, dtype=None):
    """A table with an order column `t` and the values in `x`, for every backend."""
    tbl = pl.DataFrame({"t": list(range(len(values))), "x": pl.Series(values, dtype=dtype)})
    if tbl_type == "pyarrow":
        return tbl.to_arrow()
    return _ordering_test_tables(tbl.to_dict(as_series=False) if dtype is None else tbl)[tbl_type]


def _monotonic_counts(tbl, method: str, **kwargs) -> tuple[int, int]:
    validation = getattr(pb.Validate(data=tbl), method)(columns="x", order_by="t", **kwargs)
    validation = validation.interrogate()
    assert validation.validation_info[0].eval_error is not True
    return validation.n_passed(i=1, scalar=True), validation.n_failed(i=1, scalar=True)


def pa_table(data: dict):
    return pl.DataFrame(data).to_arrow()


# With `-99` and `-98` as codes, the codes pass and are skipped like nulls: the next value is
# compared with the last real value before the code (#135)
@pytest.mark.parametrize(
    "values, increasing, decreasing",
    [
        ([1, 3, -99, 2], (3, 1), (3, 1)),  # the 2 fails against the 3, not the -99
        ([1, -99, 3, -98, 4], (5, 0), (3, 2)),
        ([-99, 1, 2], (3, 0), (2, 1)),  # a leading code passes; the 1 is the starting point
        ([1.0, math.nan, 3.0, -99.0, None, 2.0], (5, 1), (5, 1)),  # NaN and null are missing too
    ],
)
@pytest.mark.parametrize("tbl_type", ALL_TBL_TYPES)
def test_missing_codes_are_skipped(tbl_type, values, increasing, decreasing):
    data = {"t": list(range(len(values))), "x": [None if v is None else float(v) for v in values]}
    tbl = pa_table(data) if tbl_type == "pyarrow" else _ordering_test_tables(data)[tbl_type]
    spec = pb.MissingSpec(reasons={-99: "refused", -98: "don't know"})

    assert _monotonic_counts(tbl, "col_vals_increasing", missing=spec) == increasing
    assert _monotonic_counts(tbl, "col_vals_decreasing", missing=spec) == decreasing


@pytest.mark.parametrize("tbl_type", ALL_TBL_TYPES)
def test_missing_codes_without_null_is_missing(tbl_type):
    """With `null_is_missing=False`, nulls are left to `na_pass=` but still skipped."""
    data = {"t": [0, 1, 2, 3, 4], "x": [1.0, 3.0, -99.0, None, 2.0]}
    tbl = pa_table(data) if tbl_type == "pyarrow" else _ordering_test_tables(data)[tbl_type]
    spec = pb.MissingSpec(reasons={-99: "refused"}, null_is_missing=False)

    # The null fails, and the 2 fails against the 3
    assert _monotonic_counts(tbl, "col_vals_increasing", missing=spec) == (3, 2)
    assert _monotonic_counts(tbl, "col_vals_increasing", missing=spec, na_pass=True) == (4, 1)


DATES = [datetime.date(2024, 1, d) for d in (1, 3, 2)] + [None, datetime.date(2024, 1, 5)]
DATETIMES = [datetime.datetime(2024, 1, d, 9, 30) for d in (1, 3, 2)] + [
    None,
    datetime.datetime(2024, 1, 5, 9, 30),
]
DURATIONS = [datetime.timedelta(hours=h) for h in (1, 3, 2)] + [None, datetime.timedelta(hours=5)]
DECIMALS = [Decimal("1.10"), Decimal("3.10"), Decimal("2.10"), None, Decimal("5.10")]


# Dates and datetimes are compared rather than subtracted, which pandas and PyArrow couldn't do
# (#136). In `t` order the values go 1, 3, 2, null, 5: the 2 and the null fail when increasing;
# the 3, the null and the 5 fail when decreasing
@pytest.mark.parametrize(
    "values, dtype",
    [
        (DATES, None),
        (DATETIMES, None),
        (DATETIMES, pl.Datetime("us", "UTC")),
        (DURATIONS, None),
        (DECIMALS, pl.Decimal(10, 2)),
        (DECIMALS, pl.Decimal(38, 2)),  # subtracting these overflowed the precision on PyArrow
    ],
    ids=["date", "datetime", "datetime-tz", "duration", "decimal", "decimal-38"],
)
@pytest.mark.parametrize("tbl_type", ALL_TBL_TYPES)
def test_non_float_column_types(tbl_type, values, dtype):
    if values is DURATIONS and tbl_type in ("duckdb", "memtable"):
        pytest.skip("Ibis gives durations the interval type, which can't be collected to Arrow")

    tbl = _typed_test_table(tbl_type, values, dtype)

    assert _monotonic_counts(tbl, "col_vals_increasing") == (3, 2)
    assert _monotonic_counts(tbl, "col_vals_increasing", na_pass=True) == (4, 1)
    assert _monotonic_counts(tbl, "col_vals_decreasing") == (2, 3)
    assert _monotonic_counts(tbl, "col_vals_decreasing", allow_stationary=True) == (2, 3)


@pytest.mark.parametrize("tbl_type", ALL_TBL_TYPES)
def test_decimal_column_with_tolerance(tbl_type):
    tbl = _typed_test_table(tbl_type, DECIMALS, pl.Decimal(38, 2))

    # The drop from 3.10 to 2.10 is within the tolerance
    assert _monotonic_counts(tbl, "col_vals_increasing", decreasing_tol=1) == (4, 1)
    assert _monotonic_counts(tbl, "col_vals_decreasing", increasing_tol=1) == (2, 3)


def test_pandas_object_decimal_column():
    """Decimals in a pandas `object` column (as from `read_sql()`) are compared as they are."""
    tbl = pd.DataFrame({"x": pd.Series(DECIMALS, dtype=object)})

    for kwargs, expected in [({}, (3, 2)), ({"decreasing_tol": 1}, (4, 1))]:
        validation = pb.Validate(data=tbl).col_vals_increasing(columns="x", **kwargs).interrogate()
        assert (validation.n_passed(i=1, scalar=True), validation.n_failed(i=1, scalar=True)) == (
            expected
        )


def _eval_error_note(tbl, method: str, **kwargs) -> str:
    validation = getattr(pb.Validate(data=tbl), method)(columns="x", **kwargs).interrogate()
    step = validation.validation_info[0]
    assert step.eval_error is True
    return step.notes["eval_error"]["text"]


@pytest.mark.parametrize(
    "values", [["a", "c", "b"], [True, False, True]], ids=["string", "boolean"]
)
@pytest.mark.parametrize("tbl_type", ALL_TBL_TYPES)
@pytest.mark.parametrize("method", ["col_vals_increasing", "col_vals_decreasing"])
def test_unorderable_column_is_eval_error(tbl_type, values, method):
    """A column that isn't numeric or temporal is an evaluation error, not a crash (#136)."""
    tbl = _typed_test_table(tbl_type, values)

    note = _eval_error_note(tbl, method, order_by="t")
    assert f"{method}() can't check" in note


@pytest.mark.parametrize(
    "values", [DATES, DATETIMES, DURATIONS], ids=["date", "datetime", "duration"]
)
@pytest.mark.parametrize("tbl_type", ["polars", "pandas", "pyarrow", "duckdb"])
def test_tolerance_on_temporal_column_is_eval_error(tbl_type, values):
    """The tolerance is a number, so it needs a numeric column."""
    if values is DURATIONS and tbl_type == "duckdb":
        pytest.skip("Ibis gives durations the interval type, which can't be collected to Arrow")

    tbl = _typed_test_table(tbl_type, values)

    note = _eval_error_note(tbl, "col_vals_increasing", order_by="t", decreasing_tol=1)
    assert "decreasing_tol= needs a numeric column" in note
    note = _eval_error_note(tbl, "col_vals_decreasing", order_by="t", increasing_tol=0.5)
    assert "increasing_tol= needs a numeric column" in note


def test_unorderable_pandas_object_column_is_eval_error():
    tbl = pd.DataFrame({"x": pd.Series([1, None, "a"], dtype=object)})

    note = _eval_error_note(tbl, "col_vals_increasing")
    assert "can't be put in order" in note


def test_pandas_object_date_column_with_nulls():
    values = [datetime.date(2024, 1, 2), None, datetime.date(2024, 1, 1)]
    tbl = pd.DataFrame({"t": [1, 2, 3], "x": pd.Series(values, dtype=object)})

    if int(pd.__version__.split(".")[0]) >= 3:
        # pandas 3 can order dates alongside nulls in an `object` column
        assert _monotonic_counts(tbl, "col_vals_increasing", na_pass=True) == (2, 1)
    else:
        # pandas 2 can't, and the step is an evaluation error
        note = _eval_error_note(tbl, "col_vals_increasing")
        assert "can't be put in order" in note
