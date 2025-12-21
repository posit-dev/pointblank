import pytest

from pointblank import Validate
import polars as pl
from pointblank._agg import load_validation_method_grid, is_valid_agg


@pytest.fixture
def simple_pl() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 1, 1, None],
            "b": [2, 2, 2, None],
            "c": [3, 3, 3, None],
        }
    )


@pytest.mark.parametrize(
    "tol",
    [
        (0, 0),
        (1, 1),
        (100, 100),
        0,
    ],
)
def test_sums_old(tol, simple_pl) -> None:
    v = Validate(simple_pl).col_sum_eq("a", 3, tol=tol).interrogate()

    v.assert_passing()

    v.get_tabular_report()


# TODO: Expand expression types
# TODO: Expand table types
@pytest.mark.parametrize(
    ("method", "vals"),
    [
        # Sum -> 3, 6, 9
        ("col_sum_eq", (3, 6, 9)),
        ("col_sum_gt", (2, 5, 8)),
        ("col_sum_ge", (3, 6, 9)),
        ("col_sum_lt", (4, 7, 10)),
        ("col_sum_le", (3, 6, 9)),
        # Average -> 1, 2, 3
        ("col_avg_eq", (1, 2, 3)),
        ("col_avg_gt", (0, 1, 2)),
        ("col_avg_ge", (1, 2, 3)),
        ("col_avg_lt", (2, 3, 4)),
        ("col_avg_le", (1, 2, 3)),
        # Standard Deviation -> 0, 0, 0
        ("col_sd_eq", (0, 0, 0)),
        ("col_sd_gt", (-1, -1, -1)),
        ("col_sd_ge", (0, 0, 0)),
        ("col_sd_lt", (1, 1, 1)),
        ("col_sd_le", (0, 0, 0)),
    ],
)
def test_aggs(simple_pl: pl.DataFrame, method: str, vals: tuple[int, int, int]):
    v = Validate(simple_pl)
    for col, val in zip(["a", "b", "c"], vals):
        v = getattr(v, method)(col, val)
    v = v.interrogate()

    v.assert_passing()


@pytest.fixture
def simple_pl() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": [1, 1, 1, None],
            "b": [2, 2, 2, None],
            "c": [3, 3, 3, None],
        }
    )


@pytest.fixture
def varied_pl() -> pl.DataFrame:
    """DataFrame with varied values for testing standard deviation"""
    return pl.DataFrame(
        {
            "low_variance": [5, 5, 5, 5, 5],
            "high_variance": [1, 5, 10, 15, 20],
            "mixed": [1, 2, 3, 4, 5],
        }
    )


@pytest.fixture
def edge_case_pl() -> pl.DataFrame:
    """DataFrame with edge cases: single value, all nulls, mixed nulls"""
    return pl.DataFrame(
        {
            "single_value": [42, None, None, None],
            "all_nulls": [None, None, None, None],
            "mostly_nulls": [1, None, None, None],
            "no_nulls": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def negative_pl() -> pl.DataFrame:
    """DataFrame with negative numbers"""
    return pl.DataFrame(
        {
            "all_negative": [-1, -2, -3, -4],
            "mixed_signs": [-2, -1, 1, 2],
            "zeros": [0, 0, 0, 0],
        }
    )


@pytest.fixture
def large_values_pl() -> pl.DataFrame:
    """DataFrame with large values to test numerical stability"""
    return pl.DataFrame(
        {
            "large": [1_000_000, 1_000_000, 1_000_000],
            "very_large": [1e10, 1e10, 1e10],
            "small_decimals": [0.001, 0.002, 0.003],
        }
    )


# Original test
@pytest.mark.parametrize(
    ("method", "vals"),
    [
        ("col_sum_eq", (3, 6, 9)),
        ("col_sum_gt", (2, 5, 8)),
        ("col_sum_ge", (3, 6, 9)),
        ("col_sum_lt", (4, 7, 10)),
        ("col_sum_le", (3, 6, 9)),
        ("col_avg_eq", (1, 2, 3)),
        ("col_avg_gt", (0, 1, 2)),
        ("col_avg_ge", (1, 2, 3)),
        ("col_avg_lt", (2, 3, 4)),
        ("col_avg_le", (1, 2, 3)),
        ("col_sd_eq", (0, 0, 0)),
        ("col_sd_gt", (-1, -1, -1)),
        ("col_sd_ge", (0, 0, 0)),
        ("col_sd_lt", (1, 1, 1)),
        ("col_sd_le", (0, 0, 0)),
    ],
)
def test_aggs(simple_pl: pl.DataFrame, method: str, vals: tuple[int, int, int]):
    v = Validate(simple_pl)
    for col, val in zip(["a", "b", "c"], vals):
        getattr(v, method)(col, val)
    v = v.interrogate()
    v.assert_passing()


# Test with varied standard deviations
def test_aggs_with_variance(varied_pl: pl.DataFrame):
    v = Validate(varied_pl)

    # Low variance column should have SD close to 0
    v.col_sd_lt("low_variance", 0.1)
    v.col_sd_eq("low_variance", 0)

    # High variance column
    v.col_sd_gt("high_variance", 5)

    # Mixed values
    v.col_sd_ge("mixed", 1)

    v = v.interrogate()
    v.assert_passing()


# Test negative numbers
@pytest.mark.parametrize(
    ("method", "col", "val", "should_pass"),
    [
        # Negative sums
        ("col_sum_eq", "all_negative", -10, True),
        ("col_sum_lt", "all_negative", -9, True),
        ("col_sum_gt", "all_negative", -11, True),
        # Mixed signs sum to zero
        ("col_sum_eq", "mixed_signs", 0, True),
        # Zeros
        ("col_sum_eq", "zeros", 0, True),
        ("col_avg_eq", "zeros", 0, True),
        ("col_sd_eq", "zeros", 0, True),
        # Negative averages
        ("col_avg_eq", "all_negative", -2.5, True),
        ("col_avg_lt", "all_negative", -2, True),
    ],
)
def test_negative_values(
    negative_pl: pl.DataFrame, method: str, col: str, val: float, should_pass: bool
):
    v = Validate(negative_pl)
    v = getattr(v, method)(col, val).interrogate()

    if should_pass:
        v.assert_passing()
    else:
        with pytest.raises(AssertionError):
            v.assert_passing()


# Test edge cases with nulls
@pytest.mark.parametrize(
    ("method", "col", "val", "should_handle"),
    [
        # Single non-null value
        ("col_sum_eq", "single_value", 42, True),
        ("col_avg_eq", "single_value", 42, True),
        ("col_sd_eq", "single_value", 0, True),  # SD of single value is 0
        # Mostly nulls
        ("col_sum_eq", "mostly_nulls", 1, True),
        ("col_avg_eq", "mostly_nulls", 1, True),
        # No nulls
        ("col_sum_eq", "no_nulls", 10, True),
        ("col_avg_eq", "no_nulls", 2.5, True),
    ],
)
@pytest.mark.xfail(reason="Have some work to do here")
def test_edge_cases_with_nulls(
    edge_case_pl: pl.DataFrame, method: str, col: str, val: float, should_handle: bool
):
    v = Validate(edge_case_pl)
    v = getattr(v, method)(col, val)
    v = v.interrogate()
    v.assert_passing()


# Test boundary conditions
@pytest.mark.parametrize(
    ("method", "col", "exact_val", "just_below", "just_above"),
    [
        ("col_sum", "a", 3, 2.99, 3.01),
        ("col_avg", "b", 2, 1.99, 2.01),
        ("col_sd", "c", 0, -0.01, 0.01),
    ],
)
def test_boundary_conditions(
    simple_pl: pl.DataFrame,
    method: str,
    col: str,
    exact_val: float,
    just_below: float,
    just_above: float,
):
    # Test exact equality
    v = Validate(simple_pl)
    getattr(v, f"{method}_eq")(col, exact_val)
    v.interrogate().assert_passing()

    # Test greater than (just below should pass)
    v = Validate(simple_pl)
    getattr(v, f"{method}_gt")(col, just_below)
    v.interrogate().assert_passing()

    # Test less than (just above should pass)
    v = Validate(simple_pl)
    getattr(v, f"{method}_lt")(col, just_above)
    v.interrogate().assert_passing()

    # Test greater than or equal
    v = Validate(simple_pl)
    getattr(v, f"{method}_ge")(col, exact_val)
    v.interrogate().assert_passing()

    # Test less than or equal
    v = Validate(simple_pl)
    getattr(v, f"{method}_le")(col, exact_val)
    v.interrogate().assert_passing()


# Test large values
def test_large_values(large_values_pl: pl.DataFrame):
    v = Validate(large_values_pl)

    # Large values
    v = v.col_sum_eq("large", 3_000_000)
    v = v.col_avg_eq("large", 1_000_000)

    # Very large values
    v = v.col_sum_eq("very_large", 3e10)
    v = v.col_avg_eq("very_large", 1e10)

    # Small decimals
    v = v.col_sum_eq("small_decimals", 0.006)
    v = v.col_avg_eq("small_decimals", 0.002)

    v = v.interrogate()
    v.assert_passing()


# Test multiple assertions on same column
def test_multiple_assertions_same_column(simple_pl: pl.DataFrame):
    v = Validate(simple_pl)

    # Multiple checks on column 'a'
    v = v.col_sum_eq("a", 3)
    v = v.col_sum_ge("a", 3)
    v = v.col_sum_le("a", 3)
    v = v.col_avg_eq("a", 1)
    v = v.col_sd_eq("a", 0)

    v = v.interrogate()
    v.assert_passing()


# Test chaining all comparison operators
def test_all_operators_chained(simple_pl: pl.DataFrame):
    v = Validate(simple_pl)

    # Test all operators work together
    v = v.col_sum_gt("a", 2)
    v = v.col_sum_lt("a", 4)
    v = v.col_avg_ge("b", 2)
    v = v.col_avg_le("b", 2)
    v = v.col_sd_eq("c", 0)

    v = v.interrogate()
    v.assert_passing()


# Test failure cases
@pytest.mark.parametrize(
    ("method", "col", "val"),
    [
        ("col_sum_eq", "a", 999),  # Wrong sum
        ("col_sum_gt", "a", 10),  # Sum not greater
        ("col_avg_lt", "b", 1),  # Avg not less than
        ("col_sd_gt", "c", 5),  # SD not greater
    ],
)
def test_expected_failures(simple_pl: pl.DataFrame, method: str, col: str, val: float):
    v = Validate(simple_pl)

    v = getattr(v, method)(col, val).interrogate()

    with pytest.raises(AssertionError):
        v.assert_passing()


# Test with floating point precision
def test_floating_point_precision():
    df = pl.DataFrame(
        {
            "precise": [1.1, 2.2, 3.3],
            "imprecise": [0.1 + 0.2, 0.2 + 0.3, 0.3 + 0.4],  # Classic floating point issues
        }
    )

    v: Validate = Validate(df)

    # Sum might not be exactly 6.6 due to floating point
    v = v.col_sum_ge("precise", 6.5)
    v = v.col_sum_le("precise", 6.7)

    v = v.interrogate()
    v.assert_passing()


# Test with extreme standard deviations
def test_extreme_standard_deviations():
    df = pl.DataFrame(
        {
            "uniform": [5, 5, 5, 5, 5],
            "extreme_range": [1, 1000, 1, 1000, 1],
        }
    )

    Validate(df).col_sd_eq("uniform", 0).col_sd_gt(
        "extreme_range", 400
    ).interrogate().assert_passing()


def test_all_methods_can_be_accessed():
    v = Validate(pl.DataFrame())

    for meth in load_validation_method_grid():
        assert hasattr(v, meth)


def test_invalid_agg():
    assert not is_valid_agg("not_a_real_method")
    assert is_valid_agg("col_sum_eq")


if __name__ == "__main__":
    pytest.main([__file__, "-sv"])
