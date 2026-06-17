import polars as pl
import pytest

import pointblank as pb


@pytest.fixture
def age_missing():
    return pb.MissingSpec(reasons={-99: "not_asked", -98: "refused", -97: "dont_know"})


class TestColMissingCoded:
    def test_passes_when_all_coded(self, age_missing):
        # All absence expressed as sentinels; no raw nulls
        tbl = pl.DataFrame({"age": [34, -98, 41, -99, 29, -97, 55, 38]})
        validation = (
            pb.Validate(data=tbl)
            .col_missing_coded(columns="age", missing=age_missing)
            .interrogate()
        )
        info = validation.validation_info[0]
        assert info.n == 8
        assert info.n_failed == 0
        assert info.all_passed is True

    def test_fails_on_raw_null(self, age_missing):
        tbl = pl.DataFrame({"age": [34, -98, 41, None, 29, -99, 55, None]})
        validation = (
            pb.Validate(data=tbl)
            .col_missing_coded(columns="age", missing=age_missing)
            .interrogate()
        )
        info = validation.validation_info[0]
        assert info.n == 8
        assert info.n_failed == 2  # two raw nulls
        assert info.all_passed is False

    def test_sentinels_pass(self, age_missing):
        # only sentinels and reals, no nulls -> all pass
        tbl = pl.DataFrame({"age": [-99, -98, -97, -99]})
        validation = (
            pb.Validate(data=tbl)
            .col_missing_coded(columns="age", missing=age_missing)
            .interrogate()
        )
        assert validation.validation_info[0].all_passed is True

    def test_missing_must_be_missingspec(self):
        tbl = pl.DataFrame({"age": [1, 2, 3]})
        with pytest.raises(TypeError):
            pb.Validate(data=tbl).col_missing_coded(columns="age", missing={-99: "x"})

    def test_multiple_columns(self, age_missing):
        tbl = pl.DataFrame({"a": [1, None, 3], "b": [-99, 2, 3]})
        validation = (
            pb.Validate(data=tbl)
            .col_missing_coded(columns=["a", "b"], missing=age_missing)
            .interrogate()
        )
        assert len(validation.validation_info) == 2
        assert validation.validation_info[0].n_failed == 1  # column a has a null
        assert validation.validation_info[1].n_failed == 0  # column b has none

    def test_report_renders_with_brief(self, age_missing):
        tbl = pl.DataFrame({"age": [34, None, 41]})
        validation = (
            pb.Validate(data=tbl)
            .col_missing_coded(columns="age", missing=age_missing, brief=True)
            .interrogate()
        )
        gt = validation.get_tabular_report()
        assert gt is not None


class TestAutobriefTranslations:
    """Exercise the autobrief text builders across languages (no KeyError)."""

    @pytest.mark.parametrize("lang", ["en", "fr", "de", "ja", "ar", "zh-Hans", "fa", "he"])
    def test_col_missing_coded_brief_langs(self, age_missing, lang):
        tbl = pl.DataFrame({"age": [34, None, 41]})
        validation = (
            pb.Validate(data=tbl, lang=lang)
            .col_missing_coded(columns="age", missing=age_missing, brief=True)
            .interrogate()
        )
        assert validation.validation_info[0].autobrief

    @pytest.mark.parametrize("lang", ["en", "fr", "de", "ja", "ar", "zh-Hans", "fa", "he"])
    def test_col_pct_missing_brief_langs(self, age_missing, lang):
        tbl = pl.DataFrame({"age": [34, -98, 41, -99]})
        validation = (
            pb.Validate(data=tbl, lang=lang)
            .col_pct_missing(columns="age", missing=age_missing, max_pct=0.5, brief=True)
            .interrogate()
        )
        assert validation.validation_info[0].autobrief
