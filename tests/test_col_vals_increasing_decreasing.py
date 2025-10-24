import datetime

import polars as pl

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
