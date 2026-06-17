import polars as pl
import pandas as pd
import pytest

import pointblank as pb


@pytest.fixture
def spec():
    return pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"})


@pytest.fixture
def spec_no_null():
    return pb.MissingSpec(reasons={-99: "not_asked", -98: "refused"}, null_is_missing=False)


def _info(v):
    return v.validation_info[0]


class TestMissingExclusion:
    def test_between_excludes_sentinels_and_nulls(self, spec):
        tbl = pl.DataFrame({"age": [34, -98, 41, -99, 29, 200, 55, None]})
        v = (
            pb.Validate(data=tbl)
            .col_vals_between(columns="age", left=0, right=120, missing=spec)
            .interrogate()
        )
        info = _info(v)
        assert info.n == 8
        # only 200 is a real out-of-range value
        assert info.n_failed == 1

    def test_gt_excludes(self, spec):
        tbl = pl.DataFrame({"age": [34, -98, 41, -99, 29, 55]})
        v = pb.Validate(data=tbl).col_vals_gt(columns="age", value=0, missing=spec).interrogate()
        assert _info(v).n_failed == 0

    def test_null_not_excluded_when_spec_says_so(self, spec_no_null):
        # null_is_missing=False -> nulls are NOT excluded; with na_pass default False, null fails gt
        tbl = pl.DataFrame({"age": [34, -98, None, 41]})
        v = (
            pb.Validate(data=tbl)
            .col_vals_gt(columns="age", value=0, missing=spec_no_null)
            .interrogate()
        )
        # -98 excluded (passes); null fails (na_pass False); reals pass -> 1 failure
        assert _info(v).n_failed == 1

    def test_in_set_excludes_sentinels(self, spec):
        tbl = pl.DataFrame({"grade": [1, 2, -99, 3, -98, 9]})
        v = (
            pb.Validate(data=tbl)
            .col_vals_in_set(columns="grade", set=[1, 2, 3], missing=spec)
            .interrogate()
        )
        # 9 is the only real value not in the set
        assert _info(v).n_failed == 1

    def test_regex_excludes_string_sentinels(self):
        spec = pb.MissingSpec(reasons={"N/A": "not_applicable", "REF": "refused"})
        tbl = pl.DataFrame({"code": ["AB12", "N/A", "CD34", "REF", "bad code"]})
        v = (
            pb.Validate(data=tbl)
            .col_vals_regex(columns="code", pattern=r"^[A-Z]{2}[0-9]{2}$", missing=spec)
            .interrogate()
        )
        # "bad code" is the only real non-matching value
        assert _info(v).n_failed == 1

    def test_no_missing_param_unchanged(self):
        tbl = pl.DataFrame({"age": [34, -98, 41]})
        v = pb.Validate(data=tbl).col_vals_gt(columns="age", value=0).interrogate()
        # -98 is a real value < 0 -> fails when missing= not used
        assert _info(v).n_failed == 1

    def test_pandas_backend(self, spec):
        tbl = pd.DataFrame({"age": [34, -98, 41, -99, 200]})
        v = (
            pb.Validate(data=tbl)
            .col_vals_between(columns="age", left=0, right=120, missing=spec)
            .interrogate()
        )
        assert _info(v).n_failed == 1

    def test_report_renders(self, spec):
        tbl = pl.DataFrame({"age": [34, -98, 41, 200]})
        v = (
            pb.Validate(data=tbl)
            .col_vals_between(columns="age", left=0, right=120, missing=spec)
            .interrogate()
        )
        assert v.get_tabular_report() is not None
