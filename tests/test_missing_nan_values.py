import math

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

import pointblank as pb

NAN = math.nan


def _table(tbl_type: str, data: dict):
    if tbl_type == "polars":
        return pl.DataFrame(data)
    if tbl_type == "pandas":
        return pd.DataFrame(data)
    if tbl_type == "pyarrow":
        return pa.table({k: pa.array(v, type=pa.float64()) for k, v in data.items()})
    if tbl_type == "duckdb":
        import ibis

        return ibis.duckdb.connect().create_table("tbl", pl.DataFrame(data).to_arrow())
    if tbl_type == "sqlite":
        import ibis

        return ibis.sqlite.connect().create_table("tbl", pd.DataFrame(data))
    raise ValueError(tbl_type)


def _counts(validation) -> tuple[int, int]:
    return (
        validation.n_passed(i=1, scalar=True),
        validation.n_failed(i=1, scalar=True),
    )


TABLE_TYPES = ["polars", "pandas", "pyarrow"]


# `null_is_missing` is documented as treating null, None and NaN as missing, but NaN is not a null in
# Polars and PyArrow, so those backends need the NaN checked on its own.
@pytest.mark.parametrize("tbl_type", TABLE_TYPES)
def test_nan_is_missing_in_col_vals_with_missing_spec(tbl_type) -> None:
    tbl = _table(tbl_type, {"n": [1.0, -99.0, 3.0, None, 5.0, -99.0, NAN]})
    spec = pb.MissingSpec(reasons={-99.0: "refused"})

    validation = pb.Validate(tbl).col_vals_gt("n", 0, missing=spec).interrogate()

    assert _counts(validation) == (7, 0)


@pytest.mark.parametrize("tbl_type", TABLE_TYPES)
def test_nan_is_missing_in_col_missing_only_coded(tbl_type) -> None:
    tbl = _table(tbl_type, {"n": [1.0, -99.0, NAN, None, -5.0]})
    spec = pb.MissingSpec(reasons={-99.0: "refused"})

    validation = (
        pb.Validate(tbl).col_missing_only_coded("n", missing=spec, allowed=[1.0]).interrogate()
    )

    assert _counts(validation) == (4, 1)


@pytest.mark.parametrize("tbl_type", TABLE_TYPES)
def test_nan_counts_as_the_null_reason_in_col_missing_consistent(tbl_type) -> None:
    tbl = _table(
        tbl_type,
        {"a": [-99.0, NAN, -99.0, 2.0, 1.0], "b": [-99.0, -99.0, NAN, 2.0, -99.0]},
    )
    spec = pb.MissingSpec(reasons={-99.0: "refused"}, null_reason="refused")

    validation = (
        pb.Validate(tbl)
        .col_missing_consistent(["a", "b"], missing=spec, when_reason="refused")
        .interrogate()
    )

    assert _counts(validation) == (4, 1)


@pytest.mark.parametrize("tbl_type", TABLE_TYPES)
def test_nan_is_uncoded_missingness_in_col_missing_coded(tbl_type) -> None:
    tbl = _table(tbl_type, {"n": [1.0, -99.0, NAN, None, 5.0]})
    spec = pb.MissingSpec(reasons={-99.0: "refused"})

    validation = pb.Validate(tbl).col_missing_coded("n", missing=spec).interrogate()

    assert _counts(validation) == (3, 2)


@pytest.mark.parametrize("tbl_type", TABLE_TYPES)
@pytest.mark.parametrize("max_pct, passes", [(0.4, False), (0.7, True)])
def test_nan_is_missing_in_col_pct_missing(tbl_type, max_pct, passes) -> None:
    tbl = _table(tbl_type, {"n": [1.0, NAN, None]})
    spec = pb.MissingSpec(reasons={-99.0: "refused"})

    validation = pb.Validate(tbl).col_pct_missing("n", missing=spec, max_pct=max_pct).interrogate()

    assert validation.all_passed() is passes


# Integers and decimals can't hold NaN, and Polars has no `is_nan()` for decimals, so those columns
# are only checked for nulls
@pytest.mark.parametrize(
    "dtype", [pl.Int64, pl.Decimal(10, 2)], ids=lambda dtype: str(dtype).split("(")[0]
)
def test_missing_spec_checks_on_columns_that_cant_hold_nan(dtype) -> None:
    tbl = pl.DataFrame(
        {"a": [1, -99, None, 5], "b": [2, -99, None, None]}, schema={"a": dtype, "b": dtype}
    )
    spec = pb.MissingSpec(reasons={-99: "refused"})

    validation = (
        pb.Validate(tbl)
        .col_vals_gt("a", 0, missing=spec)
        .col_missing_only_coded("a", missing=spec, min_val=0)
        .col_missing_consistent(["a", "b"], missing=spec, when_reason=spec.null_reason)
        .col_missing_coded("a", missing=spec)
        .col_pct_missing("a", missing=spec, max_pct=0.5)
        .interrogate()
    )

    assert [
        (validation.n_passed(i=i, scalar=True), validation.n_failed(i=i, scalar=True))
        for i in range(1, 6)
    ] == [(4, 0), (4, 0), (3, 1), (3, 1), (1, 0)]


# NaN is not a null in Polars, PyArrow and DuckDB, and it's greater than every number there, so
# `col_vals_between()` and `col_vals_outside()` have to treat it as missing on their own, whether it's
# the value or one of the bounds
@pytest.mark.parametrize("tbl_type", TABLE_TYPES + ["duckdb", "sqlite"])
@pytest.mark.parametrize(
    "method, data, left, right, expected",
    [
        ("between", {"x": [1.0, 7.0, NAN]}, 0, 5, [(1, 2), (2, 1)]),
        ("between", {"x": [2.0, 2.0], "lo": [NAN, 0.0]}, pb.col("lo"), 5, [(1, 1), (2, 0)]),
        ("between", {"x": [2.0, 2.0], "hi": [NAN, 5.0]}, 0, pb.col("hi"), [(1, 1), (2, 0)]),
        ("outside", {"x": [1.0, 7.0, NAN]}, 0, 5, [(1, 2), (2, 1)]),
        ("outside", {"x": [-1.0, -1.0], "lo": [NAN, 0.0]}, pb.col("lo"), 5, [(1, 1), (2, 0)]),
        ("outside", {"x": [9.0, 9.0], "hi": [NAN, 5.0]}, 0, pb.col("hi"), [(1, 1), (2, 0)]),
    ],
    ids=[
        "between-value",
        "between-low-column",
        "between-high-column",
        "outside-value",
        "outside-low-column",
        "outside-high-column",
    ],
)
def test_nan_is_missing_in_range_checks(tbl_type, method, data, left, right, expected) -> None:
    # The first count is with `na_pass=False`, the second with `na_pass=True`; the NaN row only
    # passes with `na_pass=True`
    counts = []
    for na_pass in [False, True]:
        validation = pb.Validate(_table(tbl_type, data))
        getattr(validation, f"col_vals_{method}")("x", left, right, na_pass=na_pass)
        counts.append(_counts(validation.interrogate()))

    assert counts == expected
