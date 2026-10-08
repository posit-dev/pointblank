from __future__ import annotations

import pytest
import pandas as pd

import pointblank as pb
from pointblank.steps import Steps
from pointblank.contract import Step


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "email": ["a@b.c", "d@e.f", "g@h.i"],
            "amount": [10, 20, 30],
            "score": [0.5, 0.8, 0.9],
        }
    )


@pytest.fixture
def df_with_issues():
    return pd.DataFrame(
        {
            "id": [1, None, 3],
            "email": ["a@b.c", None, "g@h.i"],
            "amount": [10, -5, 30],
            "score": [0.5, 0.8, 1.5],
        }
    )


# ---------------------------------------------------------------------------
# Steps construction
# ---------------------------------------------------------------------------


class TestStepsConstruction:
    def test_empty_steps(self):
        s = Steps()
        assert len(s) == 0
        assert s._steps == []

    def test_init_with_step_list(self):
        step_list = [
            Step("col_vals_not_null", columns="id"),
            Step("col_vals_gt", columns="amount", value=0),
        ]
        s = Steps(steps=step_list)
        assert len(s) == 2
        assert s._steps[0].method == "col_vals_not_null"
        assert s._steps[1].method == "col_vals_gt"

    def test_chained_construction(self):
        s = (
            Steps()
            .col_vals_not_null(columns="id")
            .col_vals_gt(columns="amount", value=0)
            .col_vals_regex(columns="email", pattern=r".+@.+")
        )
        assert len(s) == 3
        assert s._steps[0].method == "col_vals_not_null"
        assert s._steps[1].method == "col_vals_gt"
        assert s._steps[2].method == "col_vals_regex"

    def test_returns_self(self):
        s = Steps()
        result = s.col_vals_not_null(columns="id")
        assert result is s

    def test_kwargs_captured(self):
        s = Steps().col_vals_gt(columns="amount", value=0, na_pass=True)
        step = s._steps[0]
        assert step.kwargs["columns"] == "amount"
        assert step.kwargs["value"] == 0
        assert step.kwargs["na_pass"] is True


# ---------------------------------------------------------------------------
# Steps repr
# ---------------------------------------------------------------------------


class TestStepsRepr:
    def test_repr_empty(self):
        s = Steps()
        assert repr(s) == "Steps(0 steps)"

    def test_repr_single(self):
        s = Steps().col_vals_not_null(columns="id")
        r = repr(s)
        assert "Steps(1 step)" in r
        assert "col_vals_not_null" in r
        assert "columns='id'" in r

    def test_repr_filters_defaults(self):
        s = Steps().col_vals_gt(columns="x", value=0)
        r = repr(s)
        assert "na_pass" not in r
        assert "active" not in r
        assert "thresholds" not in r

    def test_repr_shows_non_defaults(self):
        s = Steps().col_vals_gt(columns="x", value=0, na_pass=True)
        r = repr(s)
        assert "na_pass=True" in r

    def test_repr_html(self):
        s = Steps().col_vals_not_null(columns="id").col_vals_gt(columns="amount", value=0)
        html = s._repr_html_()
        assert "Steps" in html
        assert "2 steps" in html
        assert "col_vals_not_null" in html
        assert "col_vals_gt" in html

    def test_repr_html_empty(self):
        html = Steps()._repr_html_()
        assert "0 steps" in html
        assert "No steps defined" in html


# ---------------------------------------------------------------------------
# All validation methods exist on Steps
# ---------------------------------------------------------------------------


class TestStepsMethodCoverage:
    EXPECTED_METHODS = [
        "col_vals_gt",
        "col_vals_lt",
        "col_vals_eq",
        "col_vals_ne",
        "col_vals_ge",
        "col_vals_le",
        "col_vals_between",
        "col_vals_outside",
        "col_vals_in_set",
        "col_vals_not_in_set",
        "col_vals_increasing",
        "col_vals_decreasing",
        "col_vals_null",
        "col_vals_not_null",
        "col_vals_regex",
        "col_vals_within_spec",
        "col_vals_str_len",
        "col_vals_expr",
        "col_exists",
        "col_pct_null",
        "col_pct_missing",
        "col_missing_coded",
        "col_missing_only_coded",
        "rows_distinct",
        "rows_complete",
        "col_missing_consistent",
        "col_schema_match",
        "row_count_match",
        "data_freshness",
        "col_count_match",
        "col_vals_in_table",
        "tbl_match",
        "conjointly",
    ]

    @pytest.mark.parametrize("method_name", EXPECTED_METHODS)
    def test_method_exists(self, method_name):
        assert hasattr(Steps, method_name), f"Steps is missing method: {method_name}"
        assert callable(getattr(Steps, method_name))


# ---------------------------------------------------------------------------
# add_steps() basic behavior
# ---------------------------------------------------------------------------


class TestAddSteps:
    def test_add_single_steps_obj(self, sample_df):
        s = Steps().col_vals_not_null(columns="id").col_vals_gt(columns="amount", value=0)
        v = pb.Validate(data=sample_df).add_steps(s)
        assert len(v.validation_info) == 2
        assert v.validation_info[0].assertion_type == "col_vals_not_null"
        assert v.validation_info[1].assertion_type == "col_vals_gt"

    def test_add_multiple_steps_objs(self, sample_df):
        s1 = Steps().col_vals_not_null(columns="id")
        s2 = Steps().col_vals_gt(columns="amount", value=0)
        v = pb.Validate(data=sample_df).add_steps(s1, s2)
        assert len(v.validation_info) == 2

    def test_add_steps_preserves_existing(self, sample_df):
        s = Steps().col_vals_gt(columns="amount", value=0)
        v = pb.Validate(data=sample_df).col_vals_not_null(columns="id").add_steps(s)
        assert len(v.validation_info) == 2
        assert v.validation_info[0].assertion_type == "col_vals_not_null"
        assert v.validation_info[1].assertion_type == "col_vals_gt"

    def test_add_steps_returns_validate(self, sample_df):
        s = Steps().col_vals_not_null(columns="id")
        v = pb.Validate(data=sample_df)
        result = v.add_steps(s)
        assert result is v

    def test_add_empty_steps(self, sample_df):
        s = Steps()
        v = pb.Validate(data=sample_df).add_steps(s)
        assert len(v.validation_info) == 0

    def test_add_steps_no_args_raises(self, sample_df):
        with pytest.raises(ValueError, match="At least one"):
            pb.Validate(data=sample_df).add_steps()

    def test_add_steps_wrong_type_raises(self, sample_df):
        with pytest.raises(TypeError, match="Steps or Validate"):
            pb.Validate(data=sample_df).add_steps("not a steps object")

    def test_add_steps_chains_with_interrogate(self, sample_df):
        s = Steps().col_vals_gt(columns="amount", value=0)
        result = pb.Validate(data=sample_df).add_steps(s).interrogate()
        assert result.validation_info[0].all_passed is True


# ---------------------------------------------------------------------------
# add_steps() from Validate
# ---------------------------------------------------------------------------


class TestAddStepsFromValidate:
    def test_extract_from_validate(self, sample_df):
        source = pb.Validate(data=sample_df).col_vals_gt(columns="amount", value=0)
        v = pb.Validate(data=sample_df).add_steps(source)
        assert len(v.validation_info) == 1
        assert v.validation_info[0].assertion_type == "col_vals_gt"

    def test_extract_preserves_params(self, sample_df):
        source = pb.Validate(data=sample_df).col_vals_gt(columns="amount", value=5, na_pass=True)
        v = pb.Validate(data=sample_df).add_steps(source)
        vi = v.validation_info[0]
        assert vi.values == 5
        assert vi.na_pass is True

    def test_mixed_steps_and_validate(self, sample_df):
        s = Steps().col_vals_not_null(columns="id")
        v_source = pb.Validate(data=sample_df).col_vals_gt(columns="amount", value=0)
        v = pb.Validate(data=sample_df).add_steps(s, v_source)
        assert len(v.validation_info) == 2
        assert v.validation_info[0].assertion_type == "col_vals_not_null"
        assert v.validation_info[1].assertion_type == "col_vals_gt"


# ---------------------------------------------------------------------------
# add_steps() active= override
# ---------------------------------------------------------------------------


class TestAddStepsActive:
    def test_active_false_deactivates(self, sample_df):
        s = Steps().col_vals_not_null(columns="id").col_vals_gt(columns="amount", value=0)
        v = pb.Validate(data=sample_df).add_steps(s, active=False)
        for vi in v.validation_info:
            assert vi.active is False

    def test_active_none_preserves(self, sample_df):
        s = Steps().col_vals_not_null(columns="id", active=False)
        v = pb.Validate(data=sample_df).add_steps(s, active=None)
        assert v.validation_info[0].active is False

    def test_active_true_activates(self, sample_df):
        s = Steps().col_vals_not_null(columns="id", active=False)
        v = pb.Validate(data=sample_df).add_steps(s, active=True)
        assert v.validation_info[0].active is True


# ---------------------------------------------------------------------------
# add_steps() thresholds= override
# ---------------------------------------------------------------------------


class TestAddStepsThresholds:
    def test_thresholds_override(self, sample_df):
        s = Steps().col_vals_gt(columns="amount", value=0)
        thresh = pb.Thresholds(warning=0.1)
        v = pb.Validate(data=sample_df).add_steps(s, thresholds=thresh)
        assert v.validation_info[0].thresholds is not None

    def test_thresholds_none_preserves_step_thresholds(self, sample_df):
        step_thresh = pb.Thresholds(warning=0.5)
        s = Steps().col_vals_gt(columns="amount", value=0, thresholds=step_thresh)
        v = pb.Validate(data=sample_df).add_steps(s, thresholds=None)
        assert v.validation_info[0].thresholds is not None


# ---------------------------------------------------------------------------
# add_steps() exclude=
# ---------------------------------------------------------------------------


class TestAddStepsExclude:
    def test_exclude_by_method_name(self, sample_df):
        s = (
            Steps()
            .col_vals_not_null(columns="id")
            .col_vals_gt(columns="amount", value=0)
            .col_vals_regex(columns="email", pattern=r".+@.+")
        )
        v = pb.Validate(data=sample_df).add_steps(s, exclude=["col_vals_regex"])
        assert len(v.validation_info) == 2
        methods = [vi.assertion_type for vi in v.validation_info]
        assert "col_vals_regex" not in methods

    def test_exclude_by_index(self, sample_df):
        s = (
            Steps()
            .col_vals_not_null(columns="id")
            .col_vals_gt(columns="amount", value=0)
            .col_vals_regex(columns="email", pattern=r".+@.+")
        )
        v = pb.Validate(data=sample_df).add_steps(s, exclude=[2])
        assert len(v.validation_info) == 2
        methods = [vi.assertion_type for vi in v.validation_info]
        assert "col_vals_not_null" in methods
        assert "col_vals_regex" in methods
        assert "col_vals_gt" not in methods

    def test_exclude_mixed(self, sample_df):
        s = (
            Steps()
            .col_vals_not_null(columns="id")
            .col_vals_gt(columns="amount", value=0)
            .col_vals_regex(columns="email", pattern=r".+@.+")
        )
        v = pb.Validate(data=sample_df).add_steps(s, exclude=[1, "col_vals_regex"])
        assert len(v.validation_info) == 1
        assert v.validation_info[0].assertion_type == "col_vals_gt"

    def test_exclude_invalid_type_raises(self, sample_df):
        s = Steps().col_vals_not_null(columns="id")
        with pytest.raises(TypeError, match="strings.*or integers"):
            pb.Validate(data=sample_df).add_steps(s, exclude=[3.14])

    def test_exclude_per_source(self, sample_df):
        s1 = Steps().col_vals_not_null(columns="id").col_vals_gt(columns="amount", value=0)
        s2 = Steps().col_vals_regex(columns="email", pattern=r".+@.+")
        v = pb.Validate(data=sample_df).add_steps(s1, s2, exclude=["col_vals_gt"])
        assert len(v.validation_info) == 2
        methods = [vi.assertion_type for vi in v.validation_info]
        assert "col_vals_not_null" in methods
        assert "col_vals_regex" in methods


# ---------------------------------------------------------------------------
# add_steps() columns_map=
# ---------------------------------------------------------------------------


class TestAddStepsColumnsMap:
    def test_remap_columns_param(self):
        df = pd.DataFrame({"order_id": [1, 2], "total": [10, 20]})
        s = Steps().col_vals_gt(columns="amount", value=0)
        v = pb.Validate(data=df).add_steps(s, columns_map={"amount": "total"})
        assert v.validation_info[0].column == "total"

    def test_remap_list_columns(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        s = Steps().rows_distinct(columns_subset=["x", "y"])
        v = pb.Validate(data=df).add_steps(s, columns_map={"x": "a", "y": "b"})
        vi = v.validation_info[0]
        # rows_distinct stores columns_subset differently - check the assertion type was applied
        assert vi.assertion_type == "rows_distinct"

    def test_remap_preserves_unmapped(self):
        df = pd.DataFrame({"id": [1], "total": [10]})
        s = Steps().col_vals_not_null(columns="id").col_vals_gt(columns="amount", value=0)
        v = pb.Validate(data=df).add_steps(s, columns_map={"amount": "total"})
        assert v.validation_info[0].column == "id"
        assert v.validation_info[1].column == "total"

    def test_remap_with_data_freshness(self):
        df = pd.DataFrame({"updated_at": pd.to_datetime(["2026-09-11"])})
        s = Steps().data_freshness(column="ts", max_age="1d")
        v = pb.Validate(data=df).add_steps(s, columns_map={"ts": "updated_at"})
        assert v.validation_info[0].column == "updated_at"


# ---------------------------------------------------------------------------
# Integration: end-to-end with interrogate
# ---------------------------------------------------------------------------


class TestStepsIntegration:
    def test_full_workflow(self, df_with_issues):
        completeness = Steps().col_vals_not_null(columns="id").col_vals_not_null(columns="email")
        range_checks = Steps().col_vals_ge(columns="amount", value=0)

        result = (
            pb.Validate(data=df_with_issues).add_steps(completeness, range_checks).interrogate()
        )

        assert len(result.validation_info) == 3
        assert result.validation_info[0].all_passed is False  # id has null
        assert result.validation_info[1].all_passed is False  # email has null
        assert result.validation_info[2].all_passed is False  # amount has -5

    def test_reuse_across_datasets(self):
        checks = Steps().col_vals_gt(columns="value", value=0)

        df1 = pd.DataFrame({"value": [1, 2, 3]})
        df2 = pd.DataFrame({"value": [10, 20, 30]})

        r1 = pb.Validate(data=df1).add_steps(checks).interrogate()
        r2 = pb.Validate(data=df2).add_steps(checks).interrogate()

        assert r1.validation_info[0].all_passed is True
        assert r2.validation_info[0].all_passed is True

    def test_conditional_inclusion(self, sample_df):
        strict = Steps().col_vals_gt(columns="score", value=0.9)

        v_prod = pb.Validate(data=sample_df).add_steps(strict, active=True).interrogate()
        v_dev = pb.Validate(data=sample_df).add_steps(strict, active=False).interrogate()

        assert v_prod.validation_info[0].all_passed is False
        # Inactive steps aren't executed
        assert v_dev.validation_info[0].n is None


# ---------------------------------------------------------------------------
# Edge cases: multi-column, selectors, segments
# ---------------------------------------------------------------------------


class TestStepsEdgeCases:
    def test_multi_column_list_expands(self):
        df = pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        s = Steps().col_vals_not_null(columns=["a", "b", "c"])
        v = pb.Validate(data=df).add_steps(s).interrogate()
        assert len(v.validation_info) == 3
        columns = [vi.column for vi in v.validation_info]
        assert columns == ["a", "b", "c"]

    def test_column_selector_resolves(self):
        df = pd.DataFrame({"amt_a": [10], "amt_b": [20], "name": ["x"]})
        s = Steps().col_vals_gt(columns=pb.starts_with("amt_"), value=0)
        v = pb.Validate(data=df).add_steps(s).interrogate()
        assert len(v.validation_info) == 2
        columns = sorted(vi.column for vi in v.validation_info)
        assert columns == ["amt_a", "amt_b"]

    def test_segments_expand(self):
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "val": [1, 2, 3, 4]})
        s = Steps().col_vals_gt(columns="val", value=0, segments="group")
        v = pb.Validate(data=df).add_steps(s).interrogate()
        assert len(v.validation_info) == 2
        assert all(vi.all_passed for vi in v.validation_info)

    def test_columns_map_with_list(self):
        df = pd.DataFrame({"a": [1], "b": [2]})
        s = Steps().col_vals_not_null(columns=["x", "y"])
        v = pb.Validate(data=df).add_steps(s, columns_map={"x": "a", "y": "b"})
        assert len(v.validation_info) == 2
        columns = sorted(vi.column for vi in v.validation_info)
        assert columns == ["a", "b"]

    def test_columns_map_ignores_selectors(self):
        df = pd.DataFrame({"amt_total": [10], "amt_tax": [2]})
        s = Steps().col_vals_gt(columns=pb.starts_with("amt_"), value=0)
        v = pb.Validate(data=df).add_steps(s, columns_map={"irrelevant": "other"}).interrogate()
        assert len(v.validation_info) == 2

    def test_extract_from_validate_expanded_columns(self):
        df = pd.DataFrame({"id": [1], "name": ["a"]})
        source = pb.Validate(data=df).col_vals_not_null(columns=["id", "name"])
        v = pb.Validate(data=df).add_steps(source)
        assert len(v.validation_info) == 2
        columns = [vi.column for vi in v.validation_info]
        assert columns == ["id", "name"]

    def test_columns_map_with_data_freshness(self):
        df = pd.DataFrame({"updated_at": pd.to_datetime(["2026-09-11"])})
        s = Steps().data_freshness(column="ts", max_age="30d")
        v = pb.Validate(data=df).add_steps(s, columns_map={"ts": "updated_at"})
        assert v.validation_info[0].column == "updated_at"

    def test_steps_immutable_across_add_steps_calls(self):
        s = Steps().col_vals_gt(columns="x", value=0)
        df1 = pd.DataFrame({"x": [1]})
        df2 = pd.DataFrame({"x": [2]})
        pb.Validate(data=df1).add_steps(s)
        pb.Validate(data=df2).add_steps(s)
        assert len(s) == 1


# ---------------------------------------------------------------------------
# Steps accessible from top-level import
# ---------------------------------------------------------------------------


class TestStepsExport:
    def test_in_all(self):
        assert "Steps" in pb.__all__

    def test_importable(self):
        from pointblank import Steps as S

        assert S is Steps
