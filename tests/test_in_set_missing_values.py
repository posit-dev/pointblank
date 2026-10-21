import pandas as pd
import polars as pl
import pytest

from pointblank import Validate

STRINGS = ["x", None, "y", "z"]


def _table(tbl_type: str):
    """A column with the values `x`, missing, `y`, `z` (or 1, missing, 2, 3), for set `["x"]`/`[1]`."""
    if tbl_type == "polars_str":
        return pl.DataFrame({"a": STRINGS}), ["x"]
    if tbl_type == "polars_float":
        return pl.DataFrame({"a": [1.0, None, 2.0, 3.0]}), [1.0]
    if tbl_type == "polars_enum":
        return pl.DataFrame({"a": STRINGS}, schema={"a": pl.Enum(["x", "y", "z"])}), ["x"]
    if tbl_type == "pandas_float":
        return pd.DataFrame({"a": [1.0, None, 2.0, 3.0]}), [1.0]
    if tbl_type == "pandas_object":
        return pd.DataFrame({"a": STRINGS}), ["x"]
    if tbl_type == "pandas_Int64":
        return pd.DataFrame({"a": pd.array([1, None, 2, 3], dtype="Int64")}), [1]
    raise ValueError(tbl_type)


@pytest.mark.parametrize(
    "tbl_type",
    ["polars_str", "polars_float", "polars_enum", "pandas_float", "pandas_object", "pandas_Int64"],
)
def test_col_vals_in_set_missing_value_fails_in_sundered_data_and_extracts(tbl_type) -> None:
    tbl, set_values = _table(tbl_type)

    validation = Validate(tbl).col_vals_in_set(columns="a", set=set_values).interrogate()

    assert (
        validation.n_passed(i=1, scalar=True),
        validation.n_failed(i=1, scalar=True),
    ) == (1, 3)
    assert len(validation.get_sundered_data(type="pass")) == 1
    assert len(validation.get_sundered_data(type="fail")) == 3
    assert len(validation.get_data_extracts(i=1, frame=True)) == 3
