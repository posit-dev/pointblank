import polars as pl
import pytest

import pointblank as pb


@pytest.fixture
def survey_tbl():
    # 8 rows: ages with sentinel codes
    #  -99 = not_asked, -98 = refused, -97 = dont_know
    # values: 34, -98, 41, -99, 29, -98, 55, 38
    #  -> 2 refused, 1 not_asked, 0 dont_know, 5 real -> 3/8 = 0.375 missing
    return pl.DataFrame({"age": [34, -98, 41, -99, 29, -98, 55, 38]})


@pytest.fixture
def age_missing():
    return pb.MissingSpec(
        reasons={-99: "not_asked", -98: "refused", -97: "dont_know"},
        categories={"item_nonresponse": ["refused", "dont_know"], "design": ["not_asked"]},
    )


def _single_step_passed(validation):
    info = validation.validation_info[0]
    return info.all_passed


class TestColPctMissing:
    def test_overall_pass(self, survey_tbl, age_missing):
        validation = (
            pb.Validate(data=survey_tbl)
            .col_pct_missing(columns="age", missing=age_missing, max_pct=0.5)
            .interrogate()
        )
        assert _single_step_passed(validation) is True

    def test_overall_fail(self, survey_tbl, age_missing):
        validation = (
            pb.Validate(data=survey_tbl)
            .col_pct_missing(columns="age", missing=age_missing, max_pct=0.30)
            .interrogate()
        )
        # 3/8 = 0.375 > 0.30 -> fail
        assert _single_step_passed(validation) is False

    def test_by_reason_refused(self, survey_tbl, age_missing):
        # 2/8 = 0.25 refused
        passing = (
            pb.Validate(data=survey_tbl)
            .col_pct_missing(columns="age", missing=age_missing, reason="refused", max_pct=0.25)
            .interrogate()
        )
        failing = (
            pb.Validate(data=survey_tbl)
            .col_pct_missing(columns="age", missing=age_missing, reason="refused", max_pct=0.20)
            .interrogate()
        )
        assert _single_step_passed(passing) is True
        assert _single_step_passed(failing) is False

    def test_by_reason_zero(self, survey_tbl, age_missing):
        # no dont_know values -> 0% always passes
        validation = (
            pb.Validate(data=survey_tbl)
            .col_pct_missing(columns="age", missing=age_missing, reason="dont_know", max_pct=0.0)
            .interrogate()
        )
        assert _single_step_passed(validation) is True

    def test_by_category(self, survey_tbl, age_missing):
        # item_nonresponse = refused + dont_know = 2/8 = 0.25
        passing = (
            pb.Validate(data=survey_tbl)
            .col_pct_missing(
                columns="age", missing=age_missing, category="item_nonresponse", max_pct=0.25
            )
            .interrogate()
        )
        assert _single_step_passed(passing) is True

    def test_nulls_counted(self, age_missing):
        tbl = pl.DataFrame({"age": [34, None, 41, -98, 29, 38, 55, 38]})
        # null_is_missing=True by default: 1 null + 1 refused = 2/8 = 0.25
        validation = (
            pb.Validate(data=tbl)
            .col_pct_missing(columns="age", missing=age_missing, max_pct=0.25)
            .interrogate()
        )
        assert _single_step_passed(validation) is True

    def test_nulls_excluded_when_spec_says_so(self):
        spec = pb.MissingSpec(reasons={-98: "refused"}, null_is_missing=False)
        tbl = pl.DataFrame({"age": [34, None, None, -98, 29, 38, 55, 38]})
        # only -98 counts: 1/8 = 0.125
        validation = (
            pb.Validate(data=tbl)
            .col_pct_missing(columns="age", missing=spec, max_pct=0.125)
            .interrogate()
        )
        assert _single_step_passed(validation) is True

    def test_reason_and_category_mutually_exclusive(self, survey_tbl, age_missing):
        with pytest.raises(ValueError, match="Only one of"):
            pb.Validate(data=survey_tbl).col_pct_missing(
                columns="age",
                missing=age_missing,
                reason="refused",
                category="item_nonresponse",
                max_pct=0.5,
            )

    def test_max_pct_bounds(self, survey_tbl, age_missing):
        with pytest.raises(ValueError, match="max_pct"):
            pb.Validate(data=survey_tbl).col_pct_missing(
                columns="age", missing=age_missing, max_pct=1.5
            )

    def test_missing_must_be_missingspec(self, survey_tbl):
        with pytest.raises(TypeError):
            pb.Validate(data=survey_tbl).col_pct_missing(
                columns="age", missing={-99: "not_asked"}, max_pct=0.5
            )

    def test_multiple_columns(self, age_missing):
        tbl = pl.DataFrame(
            {"a": [1, -98, 3, 4], "b": [-99, -99, 3, 4]}
        )
        validation = (
            pb.Validate(data=tbl)
            .col_pct_missing(columns=["a", "b"], missing=age_missing, max_pct=0.5)
            .interrogate()
        )
        assert len(validation.validation_info) == 2

    def test_report_renders(self, survey_tbl, age_missing):
        # The validation report should build without error (exercises icon + value rendering)
        validation = (
            pb.Validate(data=survey_tbl)
            .col_pct_missing(columns="age", missing=age_missing, max_pct=0.5, brief=True)
            .interrogate()
        )
        gt = validation.get_tabular_report()
        assert gt is not None
