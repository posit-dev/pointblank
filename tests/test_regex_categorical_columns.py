import pandas as pd
import polars as pl
import pyarrow as pa
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
    if tbl_type == "pandas_str":
        return pd.DataFrame({"a": VALUES})
    if tbl_type == "pandas_category":
        return pd.DataFrame({"a": pd.Categorical(VALUES)})
    if tbl_type == "pandas_pyarrow_dictionary":
        return pd.DataFrame({"a": pd.Categorical(VALUES)}).astype(
            {"a": pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string()))}
        )
    raise ValueError(tbl_type)


# `^[a-z]+$` would also match a missing value if it were cast to the string "nan" (as Narwhals < 2.27
# does for a pandas `category` column), and with `inverse=True` a missing value is left to `na_pass=`
@pytest.mark.parametrize(
    "tbl_type",
    [
        "polars_str",
        "polars_categorical",
        "polars_enum",
        "polars_lazy_enum",
        "pandas_str",
        "pandas_category",
        "pandas_pyarrow_dictionary",
    ],
)
@pytest.mark.parametrize(
    "pattern, inverse, na_pass, expected",
    [
        ("^a", False, False, (2, 2)),
        ("^a", False, True, (3, 1)),
        ("^a", True, False, (1, 3)),
        ("^a", True, True, (2, 2)),
        ("^[a-z]+$", False, False, (3, 1)),
        ("^[a-z]+$", False, True, (4, 0)),
        ("^[a-z]+$", True, False, (0, 4)),
        ("^[a-z]+$", True, True, (1, 3)),
    ],
)
def test_col_vals_regex_on_categorical_columns(
    tbl_type, pattern, inverse, na_pass, expected
) -> None:
    validation = (
        Validate(_table(tbl_type))
        .col_vals_regex(columns="a", pattern=pattern, na_pass=na_pass, inverse=inverse)
        .interrogate()
    )

    assert (
        validation.n_passed(i=1, scalar=True),
        validation.n_failed(i=1, scalar=True),
    ) == expected
