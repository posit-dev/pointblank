import pandas as pd
import polars as pl
import pytest

from pointblank import Validate

VALUES = ["ab", None, "abc", "", "abcd"]


def _table(tbl_type: str):
    if tbl_type == "polars_str":
        return pl.DataFrame({"s": VALUES})
    if tbl_type == "polars_categorical":
        return pl.DataFrame({"s": VALUES}, schema={"s": pl.Categorical})
    if tbl_type == "polars_enum":
        return pl.DataFrame({"s": VALUES}, schema={"s": pl.Enum(["ab", "abc", "", "abcd"])})
    if tbl_type == "polars_lazy_enum":
        return pl.LazyFrame({"s": VALUES}, schema={"s": pl.Enum(["ab", "abc", "", "abcd"])})
    if tbl_type == "pandas_category":
        return pd.DataFrame({"s": pd.Categorical(VALUES)})
    raise ValueError(tbl_type)


@pytest.mark.parametrize(
    "tbl_type",
    ["polars_str", "polars_categorical", "polars_enum", "polars_lazy_enum", "pandas_category"],
)
@pytest.mark.parametrize(
    "kwargs, na_pass, expected",
    [
        ({"min_val": 2}, False, (3, 2)),
        ({"max_val": 3}, False, (3, 2)),
        ({"min_val": 2, "max_val": 3}, True, (3, 2)),
    ],
)
def test_col_vals_str_len_on_categorical_columns(tbl_type, kwargs, na_pass, expected) -> None:
    validation = (
        Validate(_table(tbl_type))
        .col_vals_str_len(columns="s", na_pass=na_pass, **kwargs)
        .interrogate()
    )

    assert (
        validation.n_passed(i=1, scalar=True),
        validation.n_failed(i=1, scalar=True),
    ) == expected
