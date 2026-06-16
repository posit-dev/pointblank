import polars as pl
import pandas as pd
import pytest
from great_tables import GT

import pointblank as pb


@pytest.fixture
def tbl_pl():
    return pl.DataFrame(
        {
            "age": [34, -98, 41, -99, 29, -98, 55, None],
            "income": [50000, -99, -1, None, 42000, -99, 38000, 61000],
        }
    )


@pytest.fixture
def specs():
    return {
        "age": pb.MissingSpec(reasons={-99: "not_asked", -98: "refused", -97: "dont_know"}),
        "income": pb.MissingSpec(reasons={-99: "not_asked", -1: "below_threshold"}),
    }


class TestStructuredMissingTbl:
    def test_returns_gt(self, tbl_pl, specs):
        result = pb.missing_vals_tbl(tbl_pl, missing=specs)
        assert isinstance(result, GT)

    def test_reason_columns_present(self, tbl_pl, specs):
        html = pb.missing_vals_tbl(tbl_pl, missing=specs).as_raw_html()
        for token in [
            "Not Asked",
            "Refused",
            "Dont Know",
            "Below Threshold",
            "Unknown",
            "Complete",
            "Total N",
        ]:
            assert token in html

    def test_counts_correct(self, tbl_pl):
        # age: total 8 -> refused 2 (25%), not_asked 1 (12%), dont_know 0 (0%),
        #      unknown/null 1 (12%), complete 4 (50%)
        spec = pb.MissingSpec(reasons={-99: "not_asked", -98: "refused", -97: "dont_know"})
        html = pb.missing_vals_tbl(tbl_pl, missing={"age": spec}).as_raw_html()
        assert "4 (50%)" in html  # complete
        assert "2 (25%)" in html  # refused
        assert "1 (12%)" in html  # not_asked / unknown
        assert "0 (0%)" in html  # dont_know

    def test_null_excluded_when_spec_says_so(self):
        # null_is_missing=False -> the null is counted as complete, no Unknown column
        tbl = pl.DataFrame({"age": [34, -98, 41, None]})
        spec = pb.MissingSpec(reasons={-98: "refused"}, null_is_missing=False)
        html = pb.missing_vals_tbl(tbl, missing={"age": spec}).as_raw_html()
        assert "Unknown" not in html
        # complete = 3 (null + 2 reals) of 4 = 75%
        assert "3 (75%)" in html

    def test_pandas_input(self, specs):
        tbl = pd.DataFrame(
            {
                "age": [34, -98, 41, -99, 29, -98, 55, None],
                "income": [50000, -99, -1, None, 42000, -99, 38000, 61000],
            }
        )
        result = pb.missing_vals_tbl(tbl, missing=specs)
        assert isinstance(result, GT)

    def test_default_behavior_unchanged(self, tbl_pl):
        # No missing= -> the original sector heatmap path
        result = pb.missing_vals_tbl(tbl_pl)
        assert isinstance(result, GT)

    def test_missing_must_be_dict_of_specs(self, tbl_pl):
        with pytest.raises(TypeError):
            pb.missing_vals_tbl(tbl_pl, missing={"age": {-99: "x"}})

    def test_unknown_column_raises(self, tbl_pl):
        spec = pb.MissingSpec(reasons={-99: "not_asked"})
        with pytest.raises(ValueError, match="not found"):
            pb.missing_vals_tbl(tbl_pl, missing={"nonexistent": spec})
