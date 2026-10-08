from pointblank import Validate
from collections.abc import Callable

## IMPORTANT: READ THIS
# This test file is unique, it's designed to create the doctests for the `col_*` aggregate functions.
# Since we generate the docs dynamically using the `make pyi` command, we need a store of doctests to
# inform the examples in the docstring. The `scripts/generate_agg_validate_pyi.py` script will use these
# to create the examples.

## How to add a new test:
# 1. Create a test titled `test_<agg_function>` OR add to the existing.
# 2. Mark the function with the `@_test` decorator to ensure the pyi gen can find it.
# 3. Run `uv run pytest tests/test_agg_doctests.py` to run these tests and ensure they pass.
# 4. Run `make pyi` to update the `pyi` files to reflect the new examples.

_TEST_FUNCTION_REGISTRY: dict[str, Callable] = {}


def _test(fn):
    nm: str = fn.__name__
    name_no_test = nm.removeprefix("test_")
    _TEST_FUNCTION_REGISTRY[name_no_test] = fn
    return fn


@_test
def test_col_sum_eq():
    """Test col_sum_eq"""
    import polars as pl

    # We check this column sums to 15
    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_sum_eq("a", 15)
    v.assert_passing()


@_test
def test_col_sum_eq():
    """Test col_sum_eq"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_sum_eq("a", 15)
    v.assert_passing()


@_test
def test_col_sum_gt():
    """Test col_sum_gt"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_sum_gt("a", 10)
    v.assert_passing()


@_test
def test_col_sum_ge():
    """Test col_sum_ge"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_sum_ge("a", 15)
    v.assert_passing()


@_test
def test_col_sum_lt():
    """Test col_sum_lt"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_sum_lt("a", 20)
    v.assert_passing()


@_test
def test_col_sum_le():
    """Test col_sum_le"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_sum_le("a", 15)
    v.assert_passing()


@_test
def test_col_avg_eq():
    """Test col_avg_eq"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_avg_eq("a", 3)
    v.assert_passing()


@_test
def test_col_avg_gt():
    """Test col_avg_gt"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_avg_gt("a", 2)
    v.assert_passing()


@_test
def test_col_avg_ge():
    """Test col_avg_ge"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_avg_ge("a", 3)
    v.assert_passing()


@_test
def test_col_avg_lt():
    """Test col_avg_lt"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_avg_lt("a", 5)
    v.assert_passing()


@_test
def test_col_avg_le():
    """Test col_avg_le"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_avg_le("a", 3)
    v.assert_passing()


@_test
def test_col_sd_eq():
    """Test col_sd_eq"""
    import polars as pl

    data = pl.DataFrame({"a": [2, 4, 6, 8, 10]})
    v = Validate(data).col_sd_eq("a", 3.1622776601683795)
    v.assert_passing()


@_test
def test_col_sd_gt():
    """Test col_sd_gt"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_sd_gt("a", 1)
    v.assert_passing()


@_test
def test_col_sd_ge():
    """Test col_sd_ge"""
    import polars as pl

    data = pl.DataFrame({"a": [2, 4, 4, 4, 6]})
    v = Validate(data).col_sd_ge("a", 1.4142135623730951)
    v.assert_passing()


@_test
def test_col_sd_lt():
    """Test col_sd_lt"""
    import polars as pl

    data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
    v = Validate(data).col_sd_lt("a", 2)
    v.assert_passing()


@_test
def test_col_sd_le():
    """Test col_sd_le"""
    import polars as pl

    data = pl.DataFrame({"a": [2, 4, 4, 4, 6]})
    v = Validate(data).col_sd_le("a", 1.4142135623730951)
    v.assert_passing()
