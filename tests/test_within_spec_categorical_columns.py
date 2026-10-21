import pandas as pd
import polars as pl
import pytest

from pointblank import Validate

EMAILS = ["a@example.com", None, "not-an-email", "b@example.org"]
CODES = ["DEUTDEFF", None, "nope", "deutdeff"]


def _table(tbl_type: str, values):
    if tbl_type == "polars_str":
        return pl.DataFrame({"a": values})
    if tbl_type == "polars_categorical":
        return pl.DataFrame({"a": values}, schema={"a": pl.Categorical})
    if tbl_type == "polars_enum":
        enum = pl.Enum(sorted({v for v in values if v is not None}))
        return pl.DataFrame({"a": values}, schema={"a": enum})
    if tbl_type == "polars_lazy_enum":
        enum = pl.Enum(sorted({v for v in values if v is not None}))
        return pl.LazyFrame({"a": values}, schema={"a": enum})
    if tbl_type == "pandas_category":
        return pd.DataFrame({"a": pd.Categorical(values)})
    raise ValueError(tbl_type)


TBL_TYPES = [
    "polars_str",
    "polars_categorical",
    "polars_enum",
    "polars_lazy_enum",
    "pandas_category",
]


@pytest.mark.parametrize("tbl_type", TBL_TYPES)
@pytest.mark.parametrize("na_pass, expected", [(False, (2, 2)), (True, (3, 1))])
def test_col_vals_within_spec_email_on_categorical_columns(tbl_type, na_pass, expected) -> None:
    validation = (
        Validate(_table(tbl_type, EMAILS))
        .col_vals_within_spec(columns="a", spec="email", na_pass=na_pass)
        .interrogate()
    )

    assert (
        validation.n_passed(i=1, scalar=True),
        validation.n_failed(i=1, scalar=True),
    ) == expected


@pytest.mark.parametrize("tbl_type", TBL_TYPES)
def test_col_vals_within_spec_swift_on_categorical_columns(tbl_type) -> None:
    # SWIFT/BIC codes are upper-cased before they are matched
    validation = (
        Validate(_table(tbl_type, CODES))
        .col_vals_within_spec(columns="a", spec="swift")
        .interrogate()
    )

    assert (
        validation.n_passed(i=1, scalar=True),
        validation.n_failed(i=1, scalar=True),
    ) == (2, 2)
