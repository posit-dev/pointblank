import pandas as pd
import polars as pl
import pytest

from pointblank import Validate

VALUES = ["ax", None, "b", "ax"]


def _table(tbl_type: str):
    if tbl_type == "polars_str":
        return pl.DataFrame({"a": VALUES})
    if tbl_type == "polars_categorical":
        return pl.DataFrame({"a": VALUES}, schema={"a": pl.Categorical})
    if tbl_type == "polars_enum":
        return pl.DataFrame({"a": VALUES}, schema={"a": pl.Enum(["ax", "b"])})
    if tbl_type == "polars_lazy_enum":
        return pl.LazyFrame({"a": VALUES}, schema={"a": pl.Enum(["ax", "b"])})
    if tbl_type == "pandas_category":
        return pd.DataFrame({"a": pd.Categorical(VALUES)})
    raise ValueError(tbl_type)


@pytest.mark.parametrize(
    "tbl_type",
    ["polars_str", "polars_categorical", "polars_enum", "polars_lazy_enum", "pandas_category"],
)
@pytest.mark.parametrize("na_pass, expected", [(False, (2, 2)), (True, (3, 1))])
def test_col_vals_regex_on_categorical_columns(tbl_type, na_pass, expected) -> None:
    validation = (
        Validate(_table(tbl_type))
        .col_vals_regex(columns="a", pattern="^a", na_pass=na_pass)
        .interrogate()
    )

    assert (
        validation.n_passed(i=1, scalar=True),
        validation.n_failed(i=1, scalar=True),
    ) == expected
