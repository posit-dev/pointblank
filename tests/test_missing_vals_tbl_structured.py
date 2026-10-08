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
        # Coded reason labels keep their raw input form (snake_case), grouped under a spanner
        for token in [
            "not_asked",
            "refused",
            "dont_know",
            "below_threshold",
            "Complete",
            "Null",  # fixed column for raw nulls (not a reason)
            "Missing Reasons",  # spanner over the coded reason columns only
        ]:
            assert token in html
        # Labels are not prettified to Title Case
        assert "Not Asked" not in html and "Below Threshold" not in html
        # The redundant "Total N" column was removed (row count is in the header)
        assert "Total N" not in html

    def test_null_is_fixed_column_not_a_reason(self, tbl_pl, specs):
        # Raw nulls appear in a fixed "Null" column, not as an "unknown" reason under the spanner
        html = pb.missing_vals_tbl(tbl_pl, missing=specs).as_raw_html()
        assert "Null" in html
        assert "unknown" not in html  # the null_reason label is not shown
        # "Null" is a fixed column to the right of the coded reasons
        gt = pb.missing_vals_tbl(tbl_pl, missing=specs)
        cols = list(gt._tbl_data.columns)
        assert cols[-1] == "null"
        assert cols.index("null") > cols.index("below_threshold")

    def test_no_null_column_when_null_not_missing(self):
        # null_is_missing=False -> no "Null" column and no "unknown" text
        tbl = pl.DataFrame({"age": [34, -98, 41, None]})
        spec = {"age": pb.MissingSpec(reasons={-98: "refused"}, null_is_missing=False)}
        gt = pb.missing_vals_tbl(tbl, missing=spec)
        assert "null" not in list(gt._tbl_data.columns)
        html = gt.as_raw_html()
        assert "unknown" not in html

    def test_null_column_em_dash_when_not_applicable(self):
        # When one spec counts nulls and another doesn't, the Null column shows an em dash for the
        # column whose spec sets null_is_missing=False
        tbl = pl.DataFrame({"a": [1, -99, None], "b": [1, -99, None]})
        specs = {
            "a": pb.MissingSpec(reasons={-99: "not_asked"}),  # null_is_missing=True
            "b": pb.MissingSpec(reasons={-99: "not_asked"}, null_is_missing=False),
        }
        gt = pb.missing_vals_tbl(tbl, missing=specs)
        null_vals = list(gt._tbl_data["null"])
        # column "a" counts its 1 null; column "b" is not applicable (em dash)
        assert null_vals[0] == "1 (33%)"
        assert null_vals[1] == "—"

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
        assert "unknown" not in html
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


class TestMissingHeatmap:
    def test_heatmap_returns_gt(self, tbl_pl, specs):
        result = pb.missing_vals_tbl(tbl_pl, missing=specs, as_heatmap=True)
        assert isinstance(result, GT)

    def test_heatmap_title_and_labels(self, tbl_pl, specs):
        html = pb.missing_vals_tbl(tbl_pl, missing=specs, as_heatmap=True).as_raw_html()
        assert "Missing Pattern Heatmap" in html
        assert "refused" in html and "below_threshold" in html
        assert "Missing Reasons" in html  # spanner over reason columns
        assert "%" in html  # proportions formatted as percentages

    def test_heatmap_pandas(self, specs):
        tbl = pd.DataFrame(
            {
                "age": [34, -98, 41, -99, 29, -98, 55, None],
                "income": [50000, -99, -1, None, 42000, -99, 38000, 61000],
            }
        )
        assert isinstance(pb.missing_vals_tbl(tbl, missing=specs, as_heatmap=True), GT)

    def test_as_heatmap_ignored_without_missing(self, tbl_pl):
        # as_heatmap only applies with missing=; default sector view still returned
        assert isinstance(pb.missing_vals_tbl(tbl_pl, as_heatmap=True), GT)


class TestStyledLikeOriginal:
    """The structured/heatmap outputs should reuse the original report's title style and the
    monospaced left Column column."""

    def test_table_mode_styling(self, tbl_pl, specs):
        html = pb.missing_vals_tbl(tbl_pl, missing=specs).as_raw_html()
        # Monospaced font present (left Column column + value columns)
        assert "IBM Plex Mono" in html
        # Header carries the table type + dimensions subtitle (as the default report does)
        assert "rows" in html.lower() or "columns" in html.lower()
        # Plain title (no shrunk font-size wrapper as before)
        assert "<div style='font-size: 14px;'>Missing Values by Reason" not in html

    def test_heatmap_mode_styling(self, tbl_pl, specs):
        html = pb.missing_vals_tbl(tbl_pl, missing=specs, as_heatmap=True).as_raw_html()
        assert "IBM Plex Mono" in html
        assert "<div style='font-size: 14px;'>Missing Pattern Heatmap" not in html


class TestNonApplicableReasons:
    """Reasons not defined in a column's spec should render as an em dash, not '0 (0%)'."""

    def test_table_mode_em_dash(self, tbl_pl, specs):
        html = pb.missing_vals_tbl(tbl_pl, missing=specs).as_raw_html()
        # age has no "below_threshold"; income has no "refused"/"dont_know" -> 3 em dashes
        assert html.count("—") == 3
        # age DOES define "dont_know" but observes none -> should still show "0 (0%)"
        assert "0 (0%)" in html

    def test_heatmap_mode_em_dash(self, tbl_pl, specs):
        html = pb.missing_vals_tbl(tbl_pl, missing=specs, as_heatmap=True).as_raw_html()
        assert html.count("—") == 3

    def test_single_spec_no_em_dash(self):
        # With one spec, every reason in the union applies -> no em dashes
        tbl = pl.DataFrame({"age": [34, -98, 41, -99]})
        spec = {"age": pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"})}
        html = pb.missing_vals_tbl(tbl, missing=spec).as_raw_html()
        assert "—" not in html
