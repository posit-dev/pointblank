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
