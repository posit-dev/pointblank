import polars as pl
import pandas as pd
import pytest

import pointblank as pb


@pytest.fixture
def spec():
    return pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"})


def _info(v):
    return v.validation_info[0]


class TestColMissingOnlyCoded:
    def test_flags_undocumented_code(self, spec):
        # -95 is undocumented; reals in [0,120]; -99/-98 documented
        tbl = pl.DataFrame({"age": [34, -98, 41, -95, 29, -99, 55]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_only_coded(columns="age", missing=spec, min_val=0, max_val=120)
            .interrogate()
        )
        info = _info(v)
        assert info.n == 7
        assert info.n_failed == 1  # only -95

    def test_all_documented_or_real_passes(self, spec):
        tbl = pl.DataFrame({"age": [34, -98, 41, -99, 29]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_only_coded(columns="age", missing=spec, min_val=0, max_val=120)
            .interrogate()
        )
        assert _info(v).n_failed == 0

    def test_allowed_set(self, spec):
        tbl = pl.DataFrame({"grade": [1, 2, -99, 3, -95, -98]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_only_coded(columns="grade", missing=spec, allowed=[1, 2, 3])
            .interrogate()
        )
        # -95 is undocumented -> 1 failure
        assert _info(v).n_failed == 1

    def test_null_documented_when_null_is_missing(self):
        spec = pb.MissingSpec(reasons={-99: "not_asked"}, null_is_missing=True)
        tbl = pl.DataFrame({"age": [34, None, -99, 200]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_only_coded(columns="age", missing=spec, min_val=0, max_val=120)
            .interrogate()
        )
        # null passes (documented as unknown), -99 passes, 200 out of range -> fail
        assert _info(v).n_failed == 1

    def test_null_fails_when_not_missing(self):
        spec = pb.MissingSpec(reasons={-99: "not_asked"}, null_is_missing=False)
        tbl = pl.DataFrame({"age": [34, None, -99, 41]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_only_coded(columns="age", missing=spec, min_val=0, max_val=120)
            .interrogate()
        )
        # null is neither documented nor a real value -> fail
        assert _info(v).n_failed == 1

    def test_requires_a_real_value_constraint(self, spec):
        tbl = pl.DataFrame({"age": [1, 2, 3]})
        with pytest.raises(ValueError, match="at least one of"):
            pb.Validate(data=tbl).col_missing_only_coded(columns="age", missing=spec)

    def test_missing_must_be_spec(self):
        tbl = pl.DataFrame({"age": [1, 2, 3]})
        with pytest.raises(TypeError):
            pb.Validate(data=tbl).col_missing_only_coded(
                columns="age", missing={-99: "x"}, min_val=0, max_val=10
            )

    def test_pandas_backend(self, spec):
        tbl = pd.DataFrame({"age": [34, -98, -95, 200]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_only_coded(columns="age", missing=spec, min_val=0, max_val=120)
            .interrogate()
        )
        # -95 undocumented, 200 out of range -> 2 failures
        assert _info(v).n_failed == 2

    def test_report_and_step_report(self, spec):
        tbl = pl.DataFrame({"age": [34, -98, -95, 41]})
        v = (
            pb.Validate(data=tbl)
            .col_missing_only_coded(columns="age", missing=spec, min_val=0, max_val=120, brief=True)
            .interrogate()
        )
        assert v.get_tabular_report() is not None
        assert v.get_step_report(i=1) is not None

    @pytest.mark.parametrize("lang", ["en", "fr", "de", "ja", "ar", "zh-Hans"])
    def test_brief_langs(self, spec, lang):
        tbl = pl.DataFrame({"age": [34, -95]})
        v = (
            pb.Validate(data=tbl, lang=lang)
            .col_missing_only_coded(columns="age", missing=spec, min_val=0, max_val=120, brief=True)
            .interrogate()
        )
        assert _info(v).autobrief
