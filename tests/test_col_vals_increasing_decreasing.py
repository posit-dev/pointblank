import datetime

import ibis
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
        "pb_lagged_difference_": [9, 9, 9, 9, 9],
    }

    for tbl in [pl.DataFrame(data), pl.LazyFrame(data)]:
        validation = pb.Validate(data=tbl).col_vals_increasing(columns="a").interrogate()

        assert validation.n_passed(i=1, scalar=True) == 3
        assert validation.n_failed(i=1, scalar=True) == 2

        extract = validation.get_data_extracts(i=1, frame=True)
        assert extract["pb_row_index_"].to_list() == [9, 9]
        assert extract["pb_lagged_difference_"].to_list() == [9, 9]


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
