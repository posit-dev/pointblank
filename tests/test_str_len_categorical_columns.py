import narwhals as nw
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from pointblank import Validate
from pointblank._interrogation import _string_expr

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
    if tbl_type == "pandas_pyarrow_dictionary":
        return pd.DataFrame({"s": pd.Categorical(VALUES)}).astype(
            {"s": pd.ArrowDtype(pa.dictionary(pa.int32(), pa.string()))}
        )
    raise ValueError(tbl_type)


@pytest.mark.parametrize(
    "tbl_type",
    [
        "polars_str",
        "polars_categorical",
        "polars_enum",
        "polars_lazy_enum",
        "pandas_category",
        "pandas_pyarrow_dictionary",
    ],
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


@pytest.mark.parametrize(
    "tbl_type",
    ["polars_categorical", "polars_lazy_enum", "pandas_category", "pandas_pyarrow_dictionary"],
)
def test_string_expr_keeps_missing_values_missing(tbl_type) -> None:
    # Narwhals < 2.27 casts a missing value in a pandas `category` column to the string "nan"
    nw_tbl = nw.from_native(_table(tbl_type))
    result = nw_tbl.select(s=_string_expr(nw_tbl, "s"), is_null=_string_expr(nw_tbl, "s").is_null())
    if isinstance(result, nw.LazyFrame):
        result = result.collect()

    assert result["is_null"].to_list() == [False, True, False, False, False]
    assert result["s"].dtype == nw.String
