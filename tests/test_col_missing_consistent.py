import polars as pl
import pandas as pd
import pytest

import pointblank as pb


@pytest.fixture
def spec():
    return pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"})


def _info(v):
    return v.validation_info[0]


class TestColMissingConsistent:
    def test_basic_inconsistency(self, spec):
        tbl = pl.DataFrame(
            {"income_source": [1, -99, 2, -99], "income_amount": [50000, -99, 42000, 38000]}
        )
        v = (
            pb.Validate(data=tbl)
            .col_missing_consistent(
                columns=["income_source", "income_amount"], missing=spec, when_reason="not_asked"
            )
            .interrogate()
        )
        info = _info(v)
        assert info.n == 4
        assert info.n_failed == 1  # last row: only one column is -99

    def test_all_consistent_passes(self, spec):
        tbl = pl.DataFrame({"a": [1, -99, 2, -99], "b": [5, -99, 6, -99]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_consistent(columns=["a", "b"], missing=spec, when_reason="not_asked")
            .interrogate()
        )
        assert _info(v).n_failed == 0

    def test_null_reason_consistency(self):
        # when_reason == null_reason, null_is_missing True -> nulls count
        spec = pb.MissingSpec(reasons={-98: "refused"}, null_reason="unknown")
        tbl = pl.DataFrame({"a": [1, None, None], "b": [5, None, 6]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_consistent(columns=["a", "b"], missing=spec, when_reason="unknown")
            .interrogate()
        )
        # row2 both null -> ok; row3 only a null -> fail
        assert _info(v).n_failed == 1

    def test_three_columns(self, spec):
        tbl = pl.DataFrame({"a": [-99, 1, -99], "b": [-99, 2, -99], "c": [-99, 3, 7]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_consistent(columns=["a", "b", "c"], missing=spec, when_reason="not_asked")
            .interrogate()
        )
        # row1 all -99 ok; row2 none ok; row3 a,b -99 but c=7 -> fail
        assert _info(v).n_failed == 1

    def test_requires_two_columns(self, spec):
        tbl = pl.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="at least two columns"):
            pb.Validate(data=tbl).col_missing_consistent(
                columns=["a"], missing=spec, when_reason="not_asked"
            )

    def test_missing_must_be_spec(self):
        tbl = pl.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(TypeError):
            pb.Validate(data=tbl).col_missing_consistent(
                columns=["a", "b"], missing={-99: "x"}, when_reason="not_asked"
            )

    def test_pandas_backend(self, spec):
        tbl = pd.DataFrame({"a": [1, -99, -99], "b": [5, -99, 6]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_consistent(columns=["a", "b"], missing=spec, when_reason="not_asked")
            .interrogate()
        )
        assert _info(v).n_failed == 1

    def test_report_and_step_report(self, spec):
        tbl = pl.DataFrame({"a": [1, -99, -99], "b": [5, -99, 6]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_consistent(columns=["a", "b"], missing=spec, when_reason="not_asked")
            .interrogate()
        )
        assert v.get_tabular_report() is not None
        # step report (row-based extract path) should build without error
        assert v.get_step_report(i=1) is not None

    @pytest.mark.parametrize("lang", ["en", "fr", "de", "ja", "ar", "zh-Hans"])
    def test_brief_langs(self, spec, lang):
        tbl = pl.DataFrame({"a": [1, -99, -99], "b": [5, -99, 6]})
        v = (
            pb.Validate(data=tbl, lang=lang)
            .col_missing_consistent(
                columns=["a", "b"], missing=spec, when_reason="not_asked", brief=True
            )
            .interrogate()
        )
        assert _info(v).autobrief
