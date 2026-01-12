import pytest

from pointblank import Validate, ref
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


# =====================
# Reference Data Tests
# =====================


@pytest.fixture
def reference_data() -> pl.DataFrame:
    """Reference data for comparison tests."""
    return pl.DataFrame(
        {
            "a": [1, 1, 1],  # sum=3, avg=1, sd=0
            "b": [2, 2, 2],  # sum=6, avg=2, sd=0
            "c": [3, 3, 3],  # sum=9, avg=3, sd=0
        }
    )


@pytest.fixture
def matching_data() -> pl.DataFrame:
    """Data that matches the reference data."""
    return pl.DataFrame(
        {
            "a": [1, 1, 1],  # sum=3, avg=1, sd=0
            "b": [2, 2, 2],  # sum=6, avg=2, sd=0
            "c": [3, 3, 3],  # sum=9, avg=3, sd=0
        }
    )


@pytest.fixture
def different_data() -> pl.DataFrame:
    """Data with different values than reference."""
    return pl.DataFrame(
        {
            "a": [2, 2, 2],  # sum=6, avg=2, sd=0
            "b": [3, 3, 3],  # sum=9, avg=3, sd=0
            "c": [4, 4, 4],  # sum=12, avg=4, sd=0
        }
    )


def test_ref_sum_eq_matching(matching_data, reference_data):
    """Test that sum matches between identical data and reference."""
    v = (
        Validate(data=matching_data, reference=reference_data)
        .col_sum_eq("a", ref("a"))
        .col_sum_eq("b", ref("b"))
        .col_sum_eq("c", ref("c"))
        .interrogate()
    )
    v.assert_passing()


def test_ref_avg_eq_matching(matching_data, reference_data):
    """Test that avg matches between identical data and reference."""
    v = (
        Validate(data=matching_data, reference=reference_data)
        .col_avg_eq("a", ref("a"))
        .col_avg_eq("b", ref("b"))
        .col_avg_eq("c", ref("c"))
        .interrogate()
    )
    v.assert_passing()


def test_ref_sd_eq_matching(matching_data, reference_data):
    """Test that sd matches between identical data and reference."""
    v = (
        Validate(data=matching_data, reference=reference_data)
        .col_sd_eq("a", ref("a"))
        .col_sd_eq("b", ref("b"))
        .col_sd_eq("c", ref("c"))
        .interrogate()
    )
    v.assert_passing()


def test_ref_sum_gt(different_data, reference_data):
    """Test that sum of different data is greater than reference."""
    # different_data.a sum=6 > reference_data.a sum=3
    v = (
        Validate(data=different_data, reference=reference_data)
        .col_sum_gt("a", ref("a"))
        .interrogate()
    )
    v.assert_passing()


def test_ref_sum_lt(reference_data, different_data):
    """Test that sum is less than reference."""
    # reference_data.a sum=3 < different_data.a sum=6 (using different as reference)
    v = (
        Validate(data=reference_data, reference=different_data)
        .col_sum_lt("a", ref("a"))
        .interrogate()
    )
    v.assert_passing()


def test_ref_different_columns():
    """Test comparing different columns between data and reference."""
    data = pl.DataFrame({"x": [1, 2, 3]})  # sum=6
    reference = pl.DataFrame({"y": [2, 2, 2]})  # sum=6

    v = Validate(data=data, reference=reference).col_sum_eq("x", ref("y")).interrogate()
    v.assert_passing()


def test_ref_with_tolerance():
    """Test reference data comparison with tolerance."""
    data = pl.DataFrame({"a": [10, 11, 12]})  # sum=33
    reference = pl.DataFrame({"a": [10, 10, 10]})  # sum=30

    # Without tolerance, this should fail
    v_fail = Validate(data=data, reference=reference).col_sum_eq("a", ref("a")).interrogate()
    with pytest.raises(AssertionError):
        v_fail.assert_passing()

    # With 10% tolerance, this should pass (30 +/- 3 includes 33)
    v_pass = (
        Validate(data=data, reference=reference).col_sum_eq("a", ref("a"), tol=0.1).interrogate()
    )
    v_pass.assert_passing()


def test_ref_without_reference_data_raises():
    """Test that using ref() without reference data raises an error."""
    data = pl.DataFrame({"a": [1, 2, 3]})

    v = Validate(data=data).col_sum_eq("a", ref("a"))

    with pytest.raises(ValueError, match="Cannot use ref"):
        v.interrogate()


def test_ref_avg_comparisons():
    """Test all avg comparison operators with reference data."""
    data = pl.DataFrame({"value": [5, 5, 5]})  # avg=5
    ref_equal = pl.DataFrame({"value": [5, 5, 5]})  # avg=5
    ref_lower = pl.DataFrame({"value": [3, 3, 3]})  # avg=3
    ref_higher = pl.DataFrame({"value": [7, 7, 7]})  # avg=7

    # Test eq
    Validate(data=data, reference=ref_equal).col_avg_eq(
        "value", ref("value")
    ).interrogate().assert_passing()

    # Test gt (5 > 3)
    Validate(data=data, reference=ref_lower).col_avg_gt(
        "value", ref("value")
    ).interrogate().assert_passing()

    # Test ge (5 >= 5)
    Validate(data=data, reference=ref_equal).col_avg_ge(
        "value", ref("value")
    ).interrogate().assert_passing()

    # Test lt (5 < 7)
    Validate(data=data, reference=ref_higher).col_avg_lt(
        "value", ref("value")
    ).interrogate().assert_passing()

    # Test le (5 <= 5)
    Validate(data=data, reference=ref_equal).col_avg_le(
        "value", ref("value")
    ).interrogate().assert_passing()


def test_ref_multiple_columns_single_reference():
    """Test validating multiple columns against a single reference column."""
    data = pl.DataFrame(
        {
            "col_a": [10, 10, 10],  # sum=30
            "col_b": [10, 10, 10],  # sum=30
            "col_c": [10, 10, 10],  # sum=30
        }
    )
    reference = pl.DataFrame({"baseline": [10, 10, 10]})  # sum=30

    v = (
        Validate(data=data, reference=reference)
        .col_sum_eq("col_a", ref("baseline"))
        .col_sum_eq("col_b", ref("baseline"))
        .col_sum_eq("col_c", ref("baseline"))
        .interrogate()
    )
    v.assert_passing()


def test_ref_mixed_validation():
    """Test mixing reference-based and literal-value validations."""
    data = pl.DataFrame({"a": [1, 2, 3]})  # sum=6, avg=2
    reference = pl.DataFrame({"a": [1, 2, 3]})  # sum=6

    v = (
        Validate(data=data, reference=reference)
        .col_sum_eq("a", ref("a"))  # Reference-based
        .col_sum_eq("a", 6)  # Literal value
        .col_avg_eq("a", 2)  # Literal value
        .interrogate()
    )
    v.assert_passing()


# Tests for automatic reference column inference (when value is None)
def test_auto_ref_sum_eq():
    """Test automatic reference inference when value is not provided."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})  # sum: a=6, b=15
    reference = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})  # sum: a=6, b=15

    # When value is None, it should default to ref("a") and ref("b")
    v = (
        Validate(data=data, reference=reference)
        .col_sum_eq("a")  # No value provided, should use ref("a")
        .col_sum_eq("b")  # No value provided, should use ref("b")
        .interrogate()
    )
    v.assert_passing()


def test_auto_ref_avg_eq():
    """Test automatic reference inference for avg comparison."""
    data = pl.DataFrame({"x": [10, 20, 30]})  # avg=20
    reference = pl.DataFrame({"x": [10, 20, 30]})  # avg=20

    v = Validate(data=data, reference=reference).col_avg_eq("x").interrogate()
    v.assert_passing()


def test_auto_ref_sd_eq():
    """Test automatic reference inference for sd comparison."""
    data = pl.DataFrame({"val": [2, 4, 4, 4, 6]})
    reference = pl.DataFrame({"val": [2, 4, 4, 4, 6]})

    v = Validate(data=data, reference=reference).col_sd_eq("val").interrogate()
    v.assert_passing()


def test_auto_ref_gt():
    """Test automatic reference with greater than comparison."""
    data = pl.DataFrame({"a": [10, 20, 30]})  # sum=60
    reference = pl.DataFrame({"a": [1, 2, 3]})  # sum=6

    # data.a sum (60) > reference.a sum (6)
    v = Validate(data=data, reference=reference).col_sum_gt("a").interrogate()
    v.assert_passing()


def test_auto_ref_lt():
    """Test automatic reference with less than comparison."""
    data = pl.DataFrame({"a": [1, 2, 3]})  # sum=6
    reference = pl.DataFrame({"a": [10, 20, 30]})  # sum=60

    # data.a sum (6) < reference.a sum (60)
    v = Validate(data=data, reference=reference).col_sum_lt("a").interrogate()
    v.assert_passing()


def test_auto_ref_with_tolerance():
    """Test automatic reference with tolerance."""
    data = pl.DataFrame({"a": [11, 22, 33]})  # sum=66
    reference = pl.DataFrame({"a": [10, 20, 30]})  # sum=60

    # sum difference is 6, which is 10% of 60
    v = Validate(data=data, reference=reference).col_sum_eq("a", tol=0.11).interrogate()
    v.assert_passing()


def test_auto_ref_multiple_columns():
    """Test automatic reference with multiple columns."""
    data = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    reference = pl.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})

    v = (
        Validate(data=data, reference=reference)
        .col_sum_eq("a")
        .col_sum_eq("b")
        .col_sum_eq("c")
        .col_avg_eq("a")
        .col_avg_eq("b")
        .col_avg_eq("c")
        .interrogate()
    )
    v.assert_passing()


def test_auto_ref_no_reference_data_raises():
    """Test that using no value without reference data raises an error."""
    data = pl.DataFrame({"a": [1, 2, 3]})

    # Error is raised when calling the method, not at interrogate time
    with pytest.raises(ValueError, match="value.*required"):
        Validate(data=data).col_sum_eq("a")  # No value and no reference


def test_auto_ref_mixed_explicit_and_auto():
    """Test mixing explicit ref() and automatic inference."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    reference = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    v = (
        Validate(data=data, reference=reference)
        .col_sum_eq("a")  # Auto: ref("a")
        .col_sum_eq("b", ref("b"))  # Explicit ref
        .col_sum_eq("a", 6)  # Literal value
        .interrogate()
    )
    v.assert_passing()


def test_auto_ref_column_list():
    """Test automatic reference with a list of columns."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    reference = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]})

    # Passing a list of columns should use ref(col) for each
    v = Validate(data=data, reference=reference).col_sum_eq(["a", "b", "c"]).interrogate()
    v.assert_passing()


# =====================================================
# Parameterized Auto-Reference Tests (All Agg Methods)
# =====================================================


@pytest.fixture
def auto_ref_equal_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Matching data and reference for equality tests."""
    data = pl.DataFrame(
        {
            "val": [1.0, 2.0, 3.0, 4.0],  # sum=10, avg=2.5, sd≈1.29
        }
    )
    reference = pl.DataFrame(
        {
            "val": [1.0, 2.0, 3.0, 4.0],  # same as data
        }
    )
    return data, reference


@pytest.fixture
def auto_ref_greater_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Data with larger values than reference for gt/ge tests."""
    data = pl.DataFrame(
        {
            "val": [10.0, 20.0, 30.0, 40.0],  # sum=100, avg=25, sd≈12.91
        }
    )
    reference = pl.DataFrame(
        {
            "val": [1.0, 2.0, 3.0, 4.0],  # sum=10, avg=2.5, sd≈1.29
        }
    )
    return data, reference


@pytest.fixture
def auto_ref_lesser_data() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Data with smaller values than reference for lt/le tests."""
    data = pl.DataFrame(
        {
            "val": [1.0, 2.0, 3.0, 4.0],  # sum=10, avg=2.5, sd≈1.29
        }
    )
    reference = pl.DataFrame(
        {
            "val": [10.0, 20.0, 30.0, 40.0],  # sum=100, avg=25, sd≈12.91
        }
    )
    return data, reference


# Test all equality methods (_eq) with automatic reference inference
@pytest.mark.parametrize(
    "method",
    [
        "col_sum_eq",
        "col_avg_eq",
        "col_sd_eq",
    ],
)
def test_auto_ref_eq_methods_with_equal_data(method: str, auto_ref_equal_data):
    """Test all _eq methods pass when data equals reference (auto-inference)."""
    data, reference = auto_ref_equal_data
    v = Validate(data=data, reference=reference)
    v = getattr(v, method)("val")  # No value provided, should auto-infer ref("val")
    v = v.interrogate()
    v.assert_passing()


# Test all greater-than methods (_gt) with auto-reference
@pytest.mark.parametrize(
    "method",
    [
        "col_sum_gt",
        "col_avg_gt",
        "col_sd_gt",
    ],
)
def test_auto_ref_gt_methods(method: str, auto_ref_greater_data):
    """Test all _gt methods pass when data > reference (auto-inference)."""
    data, reference = auto_ref_greater_data
    v = Validate(data=data, reference=reference)
    v = getattr(v, method)("val")  # No value provided, should auto-infer ref("val")
    v = v.interrogate()
    v.assert_passing()


# Test all greater-or-equal methods (_ge) with auto-reference
@pytest.mark.parametrize(
    "method",
    [
        "col_sum_ge",
        "col_avg_ge",
        "col_sd_ge",
    ],
)
def test_auto_ref_ge_methods_with_equal_data(method: str, auto_ref_equal_data):
    """Test all _ge methods pass when data == reference (auto-inference)."""
    data, reference = auto_ref_equal_data
    v = Validate(data=data, reference=reference)
    v = getattr(v, method)("val")
    v = v.interrogate()
    v.assert_passing()


@pytest.mark.parametrize(
    "method",
    [
        "col_sum_ge",
        "col_avg_ge",
        "col_sd_ge",
    ],
)
def test_auto_ref_ge_methods_with_greater_data(method: str, auto_ref_greater_data):
    """Test all _ge methods pass when data > reference (auto-inference)."""
    data, reference = auto_ref_greater_data
    v = Validate(data=data, reference=reference)
    v = getattr(v, method)("val")
    v = v.interrogate()
    v.assert_passing()


# Test all less-than methods (_lt) with auto-reference
@pytest.mark.parametrize(
    "method",
    [
        "col_sum_lt",
        "col_avg_lt",
        "col_sd_lt",
    ],
)
def test_auto_ref_lt_methods(method: str, auto_ref_lesser_data):
    """Test all _lt methods pass when data < reference (auto-inference)."""
    data, reference = auto_ref_lesser_data
    v = Validate(data=data, reference=reference)
    v = getattr(v, method)("val")
    v = v.interrogate()
    v.assert_passing()


# Test all less-or-equal methods (_le) with auto-reference
@pytest.mark.parametrize(
    "method",
    [
        "col_sum_le",
        "col_avg_le",
        "col_sd_le",
    ],
)
def test_auto_ref_le_methods_with_equal_data(method: str, auto_ref_equal_data):
    """Test all _le methods pass when data == reference (auto-inference)."""
    data, reference = auto_ref_equal_data
    v = Validate(data=data, reference=reference)
    v = getattr(v, method)("val")
    v = v.interrogate()
    v.assert_passing()


@pytest.mark.parametrize(
    "method",
    [
        "col_sum_le",
        "col_avg_le",
        "col_sd_le",
    ],
)
def test_auto_ref_le_methods_with_lesser_data(method: str, auto_ref_lesser_data):
    """Test all _le methods pass when data < reference (auto-inference)."""
    data, reference = auto_ref_lesser_data
    v = Validate(data=data, reference=reference)
    v = getattr(v, method)("val")
    v = v.interrogate()
    v.assert_passing()


# Test all methods raise error when no reference data
@pytest.mark.parametrize("method", load_validation_method_grid())
def test_auto_ref_all_methods_raise_without_reference(method: str):
    """Test that all agg methods raise ValueError when value=None and no reference data."""
    data = pl.DataFrame({"val": [1, 2, 3]})

    with pytest.raises(ValueError, match="value.*required"):
        v = Validate(data=data)
        getattr(v, method)("val")  # No value and no reference


# Test all methods work with explicit ref() even without auto-inference
@pytest.mark.parametrize(
    "method",
    [
        "col_sum_eq",
        "col_avg_eq",
        "col_sd_eq",
    ],
)
def test_auto_ref_explicit_ref_still_works(method: str, auto_ref_equal_data):
    """Test that explicit ref() still works alongside auto-inference."""
    data, reference = auto_ref_equal_data
    v = Validate(data=data, reference=reference)

    # Explicit ref("val") should work the same as omitting value
    v = getattr(v, method)("val", ref("val"))
    v = v.interrogate()
    v.assert_passing()


# Test auto-reference with tolerance for all eq methods
@pytest.mark.parametrize(
    "method",
    [
        "col_sum_eq",
        "col_avg_eq",
        "col_sd_eq",
    ],
)
def test_auto_ref_eq_methods_with_tolerance(method: str):
    """Test all _eq methods with tolerance and auto-reference.

    Note: Tolerance is calculated as int(tol * ref), so we need values large
    enough that the tolerance doesn't truncate to 0. For ref=100 and tol=0.1,
    we get int(0.1 * 100) = 10, allowing a tolerance of 10 units.
    """
    # Use larger values so tolerance calculation works (int(tol * ref) > 0)
    # Data avg=110, sum=440, sd≈12.91
    data = pl.DataFrame({"val": [100.0, 105.0, 115.0, 120.0]})

    # Reference avg=100, sum=400, sd≈12.91
    reference = pl.DataFrame({"val": [90.0, 95.0, 105.0, 110.0]})

    v = Validate(data=data, reference=reference)
    v = getattr(v, method)("val", tol=0.15)  # 15% tolerance
    v = v.interrogate()
    v.assert_passing()


# Test auto-reference with multiple columns for all eq methods
@pytest.mark.parametrize(
    "method",
    [
        "col_sum_eq",
        "col_avg_eq",
        "col_sd_eq",
    ],
)
def test_auto_ref_eq_methods_multiple_columns(method: str):
    """Test auto-reference works when passing a list of columns."""
    data = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    reference = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})

    v = Validate(data=data, reference=reference)

    # Pass list of columns; should auto-infer ref(col) for each
    v = getattr(v, method)(["a", "b", "c"])
    v = v.interrogate()
    v.assert_passing()


# Test expected failures with auto-reference
@pytest.mark.parametrize(
    ("method", "data_vals", "ref_vals"),
    [
        # eq should fail when values differ
        ("col_sum_eq", [1, 2, 3], [10, 20, 30]),
        ("col_avg_eq", [1, 2, 3], [10, 20, 30]),
        ("col_sd_eq", [1, 2, 3], [1, 1, 1]),  # Different variance
        # gt should fail when data <= reference
        ("col_sum_gt", [1, 2, 3], [10, 20, 30]),
        ("col_avg_gt", [1, 2, 3], [10, 20, 30]),
        # lt should fail when data >= reference
        ("col_sum_lt", [10, 20, 30], [1, 2, 3]),
        ("col_avg_lt", [10, 20, 30], [1, 2, 3]),
    ],
)
def test_auto_ref_expected_failures(method: str, data_vals: list, ref_vals: list):
    """Test that auto-reference correctly fails when conditions are not met."""
    data = pl.DataFrame({"val": data_vals})
    reference = pl.DataFrame({"val": ref_vals})

    v = Validate(data=data, reference=reference)
    v = getattr(v, method)("val")
    v = v.interrogate()

    with pytest.raises(AssertionError):
        v.assert_passing()


# Test mixing auto-reference and explicit values in same validation
@pytest.mark.parametrize(
    "method",
    [
        "col_sum_eq",
        "col_avg_eq",
        "col_sd_eq",
    ],
)
def test_auto_ref_mixed_with_explicit_values(method: str, auto_ref_equal_data):
    """Test mixing auto-reference (value=None) with explicit numeric values."""
    data, reference = auto_ref_equal_data

    v = Validate(data=data, reference=reference)

    # First call with auto-reference
    v = getattr(v, method)("val")

    # Second call with explicit ref()
    v = getattr(v, method)("val", ref("val"))

    v = v.interrogate()
    v.assert_passing()


# =============================================================================
# Tests for validation report display formatting
# =============================================================================


def test_agg_report_columns_display():
    """Test that the COLUMNS column displays column names without list brackets."""
    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    validation = Validate(data).col_sum_gt(columns="a", value=10).interrogate()

    report = validation.get_tabular_report()
    html = report.as_raw_html()

    # Should display 'a' not '['a']'
    assert "['a']" not in html

    # The column name should be present in the HTML
    assert ">a<" in html


def test_agg_report_values_no_tolerance():
    """Test that VALUES column shows just the value when tolerance is 0."""
    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    validation = Validate(data).col_sum_gt(columns="a", value=10).interrogate()

    report = validation.get_tabular_report()
    html = report.as_raw_html()

    # Should display just '10' and no tolerance info
    assert ">10<" in html

    # Should NOT contain 'tol=' since tolerance is 0
    assert "tol=0" not in html


def test_agg_report_values_with_symmetric_tolerance():
    """Test that VALUES column shows value and tolerance on separate lines."""
    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    validation = Validate(data).col_avg_eq(columns="a", value=3, tol=0.5).interrogate()

    report = validation.get_tabular_report()
    html = report.as_raw_html()

    # Should display value and tolerance separated by <br/>
    assert "3<br/>tol=0.5" in html


def test_agg_report_values_with_asymmetric_tolerance():
    """Test that VALUES column shows value and asymmetric tolerance tuple."""
    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    validation = Validate(data).col_sd_le(columns="a", value=2.0, tol=(0.1, 0.2)).interrogate()

    report = validation.get_tabular_report()
    html = report.as_raw_html()

    # Should display value and asymmetric tolerance separated by <br/>
    assert "2.0<br/>tol=(0.1, 0.2)" in html


def test_agg_report_values_with_reference_column():
    """Test that VALUES column shows ref('column') for reference columns."""
    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    ref_data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    validation = (
        Validate(data, reference=ref_data).col_sum_eq(columns="a", value=ref("a")).interrogate()
    )

    report = validation.get_tabular_report()
    html = report.as_raw_html()

    # Should display ref('a') for the reference column
    assert "ref('a')" in html or "ref(&#x27;a&#x27;)" in html  # HTML may escape quotes


def test_agg_report_values_implicit_reference():
    """Test that VALUES column shows ref('column') for implicit reference."""
    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    ref_data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})

    # When value is omitted with reference data, it should default to ref('a')
    validation = Validate(data, reference=ref_data).col_sum_eq(columns="a").interrogate()

    report = validation.get_tabular_report()
    html = report.as_raw_html()

    # Should display ref('a') for the implicit reference column
    assert "ref('a')" in html or "ref(&#x27;a&#x27;)" in html  # HTML may escape quotes


def test_agg_report_multiple_steps_formatting():
    """Test that multiple aggregation steps all display correctly."""
    data = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})

    validation = (
        Validate(data)
        .col_sum_gt(columns="a", value=10)  # No tolerance
        .col_avg_eq(columns="b", value=30, tol=0.1)  # Symmetric tolerance
        .col_sd_le(columns="a", value=2.0, tol=(0.1, 0.2))  # Asymmetric tolerance
        .interrogate()
    )

    report = validation.get_tabular_report()
    html = report.as_raw_html()

    # Step 1: Just the value (no tolerance)
    assert ">10<" in html

    # Step 2: Value with symmetric tolerance
    assert "30<br/>tol=0.1" in html

    # Step 3: Value with asymmetric tolerance
    assert "2.0<br/>tol=(0.1, 0.2)" in html
