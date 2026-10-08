"""Tests for validate.py helper functions and uncovered branches."""

from __future__ import annotations

import datetime

import polars as pl
import pytest

import pointblank as pb
from pointblank.validate import (
    _base_dimension_from_assertion_type,
    _column_to_name,
    _coalesce_plan_steps,
    _create_text_col_missing_coded,
    _create_text_col_missing_consistent,
    _create_text_col_missing_only_coded,
    _create_text_col_pct_missing,
    _create_text_data_freshness,
    _create_text_str_len,
    _format_timedelta,
    _get_dimension_label,
    _health_score_color,
    _parse_max_age,
    _parse_reference_time,
    _parse_timezone,
    _render_code_value,
    _render_columns_arg,
    _render_schema_code,
    _render_step_code,
    _render_thresholds_code,
    _schema_to_yaml,
    _thresholds_as_dict,
    _transform_auto_brief,
    _UnserializablePlaceholder,
    _validation_info_to_step,
    _value_to_yaml,
    Validate,
)
from pointblank.column import Column, col


# ─── _parse_reference_time ───────────────────────────────────────────────────────


class TestParseReferenceTime:
    def test_iso_format(self):
        result = _parse_reference_time("2024-01-15T10:30:00")
        assert result == datetime.datetime(2024, 1, 15, 10, 30, 0)

    def test_iso_format_with_timezone(self):
        result = _parse_reference_time("2024-01-15T10:30:00+00:00")
        assert result.tzinfo is not None

    def test_fallback_format_space_separated(self):
        result = _parse_reference_time("2024-01-15 10:30:00")
        assert result == datetime.datetime(2024, 1, 15, 10, 30, 0)

    def test_fallback_format_date_only(self):
        result = _parse_reference_time("2024-01-15")
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError, match="Could not parse reference_time"):
            _parse_reference_time("not-a-date")

    def test_fallback_format_with_tz(self):
        result = _parse_reference_time("2024-01-15T10:30:00+0530")
        assert result.tzinfo is not None


# ─── _create_text_str_len ────────────────────────────────────────────────────────


class TestCreateTextStrLen:
    def test_min_and_max(self):
        text = _create_text_str_len("en", "name", {"min_val": 3, "max_val": 10})
        assert "between 3 and 10" in text

    def test_min_only(self):
        text = _create_text_str_len("en", "name", {"min_val": 3})
        assert "at least 3" in text

    def test_max_only(self):
        text = _create_text_str_len("en", "name", {"max_val": 10})
        assert "at most 10" in text

    def test_failure_text(self):
        text = _create_text_str_len("en", "name", {"min_val": 3}, for_failure=True)
        assert "at least 3" in text


# ─── _create_text_data_freshness ─────────────────────────────────────────────────


class TestCreateTextDataFreshness:
    def test_expect_text(self):
        text = _create_text_data_freshness(
            lang="en",
            column="created_at",
            value={"max_age": datetime.timedelta(hours=24)},
        )
        assert "created_at" in text

    def test_failure_text_with_age(self):
        text = _create_text_data_freshness(
            lang="en",
            column="updated_at",
            value={
                "max_age": datetime.timedelta(hours=1),
                "age": datetime.timedelta(hours=3),
            },
            for_failure=True,
        )
        assert "updated_at" in text

    def test_failure_text_without_age(self):
        text = _create_text_data_freshness(
            lang="en",
            column="col",
            value={"max_age": datetime.timedelta(hours=1)},
            for_failure=True,
        )
        assert "unknown" in text


# ─── _create_text_col_pct_missing ────────────────────────────────────────────────


class TestCreateTextColPctMissing:
    def test_basic(self):
        text = _create_text_col_pct_missing("en", "age", {"max_pct": 0.1})
        assert "age" in text

    def test_failure(self):
        text = _create_text_col_pct_missing("en", "age", {"max_pct": 0.1}, for_failure=True)
        assert "age" in text


# ─── _create_text_col_missing_coded ──────────────────────────────────────────────


class TestCreateTextColMissingCoded:
    def test_basic(self):
        text = _create_text_col_missing_coded("en", "age")
        assert "age" in text

    def test_failure(self):
        text = _create_text_col_missing_coded("en", "age", for_failure=True)
        assert "age" in text


# ─── _create_text_col_missing_only_coded ─────────────────────────────────────────


class TestCreateTextColMissingOnlyCoded:
    def test_basic(self):
        text = _create_text_col_missing_only_coded("en", "age")
        assert "age" in text

    def test_failure(self):
        text = _create_text_col_missing_only_coded("en", "age", for_failure=True)
        assert "age" in text


# ─── _create_text_col_missing_consistent ─────────────────────────────────────────


class TestCreateTextColMissingConsistent:
    def test_with_list_columns(self):
        text = _create_text_col_missing_consistent("en", ["a", "b"], {"when_reason": "not_asked"})
        assert "not_asked" in text

    def test_with_single_column(self):
        text = _create_text_col_missing_consistent("en", "a", {"when_reason": "refused"})
        assert "refused" in text

    def test_failure(self):
        text = _create_text_col_missing_consistent(
            "en", ["a", "b"], {"when_reason": "not_asked"}, for_failure=True
        )
        assert "not_asked" in text


# ─── _base_dimension_from_assertion_type ─────────────────────────────────────────


class TestBaseDimensionFromAssertionType:
    def test_none_returns_none(self):
        assert _base_dimension_from_assertion_type(None) is None

    def test_known_type(self):
        result = _base_dimension_from_assertion_type("col_vals_gt")
        assert result is not None

    def test_aggregate_method(self):
        assert _base_dimension_from_assertion_type("col_sum_eq") == "validity"
        assert _base_dimension_from_assertion_type("col_avg_gt") == "validity"
        assert _base_dimension_from_assertion_type("col_sd_le") == "validity"

    def test_unknown_type(self):
        assert _base_dimension_from_assertion_type("totally_made_up") == "unknown"


# ─── _get_dimension_label ────────────────────────────────────────────────────────


class TestGetDimensionLabel:
    def test_known_dimension(self):
        label = _get_dimension_label("validity", "en")
        assert isinstance(label, str)
        assert len(label) > 0

    def test_custom_dimension(self):
        label = _get_dimension_label("my_custom_dim", "en")
        assert label == "My Custom Dim"

    def test_none_dimension(self):
        label = _get_dimension_label(None, "en")
        assert isinstance(label, str)


# ─── _health_score_color ─────────────────────────────────────────────────────────


class TestHealthScoreColor:
    def test_high_score_green(self):
        assert _health_score_color(95) == "#2E7D32"

    def test_medium_score_amber(self):
        assert _health_score_color(80) == "#A15C00"

    def test_low_score_red(self):
        assert _health_score_color(50) == "#C62828"

    def test_boundary_90(self):
        assert _health_score_color(90) == "#2E7D32"

    def test_boundary_75(self):
        assert _health_score_color(75) == "#A15C00"


# ─── _UnserializablePlaceholder ──────────────────────────────────────────────────


class TestUnserializablePlaceholder:
    def test_eq(self):
        a = _UnserializablePlaceholder(code="None", note="test note")
        b = _UnserializablePlaceholder(code="True", note="test note")
        assert a == b

    def test_neq(self):
        a = _UnserializablePlaceholder(code="None", note="note a")
        b = _UnserializablePlaceholder(code="None", note="note b")
        assert a != b

    def test_hash(self):
        a = _UnserializablePlaceholder(code="None", note="test note")
        assert hash(a) == hash("test note")

    def test_yaml_value_defaults_to_code(self):
        p = _UnserializablePlaceholder(code="None", note="test")
        assert p.yaml_value == "None"

    def test_yaml_value_explicit(self):
        p = _UnserializablePlaceholder(code="None", note="test", yaml_value="null")
        assert p.yaml_value == "null"


# ─── _thresholds_as_dict ─────────────────────────────────────────────────────────


class TestThresholdsAsDict:
    def test_none(self):
        assert _thresholds_as_dict(None) == {}

    def test_with_values(self):
        t = pb.Thresholds(warning=0.1, error=0.25)
        result = _thresholds_as_dict(t)
        assert result == {"warning": 0.1, "error": 0.25}

    def test_empty_thresholds(self):
        t = pb.Thresholds()
        result = _thresholds_as_dict(t)
        assert result == {}


# ─── _column_to_name ─────────────────────────────────────────────────────────────


class TestColumnToName:
    def test_string(self):
        assert _column_to_name("my_col") == "my_col"

    def test_column_with_exprs(self):
        c = col("x")
        assert _column_to_name(c) == "x"

    def test_none_for_complex(self):
        assert _column_to_name(42) is None


# ─── _render_code_value ──────────────────────────────────────────────────────────


class TestRenderCodeValue:
    def test_placeholder(self):
        p = _UnserializablePlaceholder(code="lambda df: df", note="test")
        assert _render_code_value(p) == "lambda df: df"

    def test_bool(self):
        assert _render_code_value(True) == "True"
        assert _render_code_value(False) == "False"

    def test_none(self):
        assert _render_code_value(None) == "None"

    def test_string(self):
        assert _render_code_value("hello") == '"hello"'

    def test_int(self):
        assert _render_code_value(42) == "42"

    def test_float(self):
        assert _render_code_value(3.14) == "3.14"

    def test_datetime(self):
        dt = datetime.datetime(2024, 1, 15, 10, 30, 0)
        assert _render_code_value(dt) == '"2024-01-15T10:30:00"'

    def test_date(self):
        d = datetime.date(2024, 1, 15)
        assert _render_code_value(d) == '"2024-01-15"'

    def test_column(self):
        c = col("x")
        result = _render_code_value(c)
        assert result == 'pb.col("x")'

    def test_thresholds(self):
        t = pb.Thresholds(warning=0.1)
        result = _render_code_value(t)
        assert "pb.Thresholds" in result

    def test_schema(self):
        s = pb.Schema(columns=[("a", "Int64")])
        result = _render_code_value(s)
        assert "pb.Schema" in result

    def test_tuple(self):
        assert _render_code_value((1, 2)) == "(1, 2)"

    def test_single_tuple(self):
        assert _render_code_value((1,)) == "(1,)"

    def test_list(self):
        assert _render_code_value([1, 2]) == "[1, 2]"

    def test_reference_column(self):
        from pointblank.column import ReferenceColumn

        rc = ReferenceColumn(column_name="d")
        result = _render_code_value(rc)
        assert result == 'pb.ref("d")'


# ─── _render_thresholds_code ─────────────────────────────────────────────────────


class TestRenderThresholdsCode:
    def test_single_level(self):
        t = pb.Thresholds(warning=0.1)
        result = _render_thresholds_code(t)
        assert result == "pb.Thresholds(warning=0.1)"

    def test_multiple_levels(self):
        t = pb.Thresholds(warning=0.1, error=0.25, critical=0.5)
        result = _render_thresholds_code(t)
        assert "warning=0.1" in result
        assert "error=0.25" in result
        assert "critical=0.5" in result


# ─── _render_schema_code ─────────────────────────────────────────────────────────


class TestRenderSchemaCode:
    def test_none(self):
        assert _render_schema_code(None) == "pb.Schema(columns=[])"

    def test_single_column_with_dtype(self):
        s = pb.Schema(columns=[("a", "Int64")])
        result = _render_schema_code(s)
        assert '("a", "Int64")' in result

    def test_column_without_dtype(self):
        s = pb.Schema(columns=[("a",)])
        result = _render_schema_code(s)
        assert '("a",)' in result

    def test_column_with_dtype_list(self):
        s = pb.Schema(columns=[("a", ["Int64", "Float64"])])
        result = _render_schema_code(s)
        assert '["Int64", "Float64"]' in result


# ─── _render_columns_arg ─────────────────────────────────────────────────────────


class TestRenderColumnsArg:
    def test_single_item_list(self):
        assert _render_columns_arg(["a"]) == '"a"'

    def test_multi_item_list(self):
        result = _render_columns_arg(["a", "b"])
        assert result == '["a", "b"]'

    def test_string(self):
        assert _render_columns_arg("a") == '"a"'


# ─── _render_step_code ──────────────────────────────────────────────────────────


class TestRenderStepCode:
    def test_basic(self):
        result = _render_step_code("col_vals_gt", {"columns": "d", "value": 100})
        assert result == '    .col_vals_gt(columns="d", value=100)'

    def test_columns_subset(self):
        result = _render_step_code("rows_distinct", {"columns_subset": ["a"]})
        assert result == '    .rows_distinct(columns_subset="a")'


# ─── _value_to_yaml ─────────────────────────────────────────────────────────────


class TestValueToYaml:
    def test_placeholder_null(self):
        p = _UnserializablePlaceholder(code="None", note="test", yaml_value="null")
        warnings_out = []
        assert _value_to_yaml(p, warnings_out) is None

    def test_placeholder_with_value(self):
        p = _UnserializablePlaceholder(code="[]", note="test", yaml_value="[]")
        warnings_out = []
        result = _value_to_yaml(p, warnings_out)
        assert result == "[]"
        assert len(warnings_out) == 1

    def test_bool(self):
        assert _value_to_yaml(True, []) is True

    def test_none(self):
        assert _value_to_yaml(None, []) is None

    def test_int(self):
        assert _value_to_yaml(42, []) == 42

    def test_string(self):
        assert _value_to_yaml("hello", []) == "hello"

    def test_datetime(self):
        dt = datetime.datetime(2024, 1, 15)
        assert _value_to_yaml(dt, []) == "2024-01-15T00:00:00"

    def test_date(self):
        d = datetime.date(2024, 1, 15)
        assert _value_to_yaml(d, []) == "2024-01-15"

    def test_reference_column(self):
        from pointblank.column import ReferenceColumn

        rc = ReferenceColumn(column_name="d")
        result = _value_to_yaml(rc, [])
        assert result == {"python": "pb.ref('d')"}

    def test_column(self):
        c = col("x")
        result = _value_to_yaml(c, [])
        assert result == {"python": "pb.col('x')"}

    def test_thresholds(self):
        t = pb.Thresholds(warning=0.1)
        result = _value_to_yaml(t, [])
        assert result == {"warning": 0.1}

    def test_schema(self):
        s = pb.Schema(columns=[("a", "Int64")])
        result = _value_to_yaml(s, [])
        assert "columns" in result

    def test_tuple(self):
        result = _value_to_yaml((1, 2), [])
        assert result == [1, 2]

    def test_list(self):
        result = _value_to_yaml([1, 2], [])
        assert result == [1, 2]


# ─── _schema_to_yaml ────────────────────────────────────────────────────────────


class TestSchemaToYaml:
    def test_with_dtype(self):
        s = pb.Schema(columns=[("a", "Int64")])
        result = _schema_to_yaml(s)
        assert result == {"columns": [["a", "Int64"]]}

    def test_without_dtype(self):
        s = pb.Schema(columns=[("a",)])
        result = _schema_to_yaml(s)
        assert result == {"columns": [["a"]]}

    def test_with_dtype_list(self):
        s = pb.Schema(columns=[("a", ["Int64", "Float64"])])
        result = _schema_to_yaml(s)
        assert result == {"columns": [["a", ["Int64", "Float64"]]]}

    def test_none(self):
        result = _schema_to_yaml(None)
        assert result == {"columns": []}


# ─── _coalesce_plan_steps ───────────────────────────────────────────────────────


class TestCoalescePlanSteps:
    def test_merge_adjacent_same_method(self):
        steps = [
            ("col_vals_not_null", {"columns": "a"}),
            ("col_vals_not_null", {"columns": "b"}),
        ]
        result = _coalesce_plan_steps(steps)
        assert len(result) == 1
        assert result[0][1]["columns"] == ["a", "b"]

    def test_no_merge_different_methods(self):
        steps = [
            ("col_vals_not_null", {"columns": "a"}),
            ("col_vals_gt", {"columns": "b", "value": 0}),
        ]
        result = _coalesce_plan_steps(steps)
        assert len(result) == 2

    def test_no_merge_different_params(self):
        steps = [
            ("col_vals_gt", {"columns": "a", "value": 0}),
            ("col_vals_gt", {"columns": "b", "value": 10}),
        ]
        result = _coalesce_plan_steps(steps)
        assert len(result) == 2

    def test_three_columns_coalesced(self):
        steps = [
            ("col_vals_not_null", {"columns": "a"}),
            ("col_vals_not_null", {"columns": "b"}),
            ("col_vals_not_null", {"columns": "c"}),
        ]
        result = _coalesce_plan_steps(steps)
        assert len(result) == 1
        assert result[0][1]["columns"] == ["a", "b", "c"]

    def test_empty_steps(self):
        assert _coalesce_plan_steps([]) == []


# ─── col_vals_str_len via Validate ──────────────────────────────────────────────


class TestColValsStrLen:
    def test_min_val(self):
        df = pl.DataFrame({"name": ["abc", "ab", "abcde"]})
        v = Validate(data=df).col_vals_str_len(columns="name", min_val=3).interrogate()
        assert v.n_passed(i=1, scalar=True) == 2

    def test_max_val(self):
        df = pl.DataFrame({"name": ["abc", "ab", "abcde"]})
        v = Validate(data=df).col_vals_str_len(columns="name", max_val=3).interrogate()
        assert v.n_passed(i=1, scalar=True) == 2

    def test_min_and_max(self):
        df = pl.DataFrame({"name": ["abc", "ab", "abcde"]})
        v = Validate(data=df).col_vals_str_len(columns="name", min_val=2, max_val=4).interrogate()
        assert v.n_passed(i=1, scalar=True) == 2

    def test_no_min_or_max_raises(self):
        df = pl.DataFrame({"name": ["abc"]})
        with pytest.raises(ValueError, match="At least one of"):
            Validate(data=df).col_vals_str_len(columns="name")

    def test_na_pass(self):
        df = pl.DataFrame({"name": ["abc", None, "abcde"]})
        v = (
            Validate(data=df)
            .col_vals_str_len(columns="name", min_val=3, na_pass=True)
            .interrogate()
        )
        assert v.n_passed(i=1, scalar=True) == 3


# ─── to_code() / to_yaml() coverage for uncovered branches ─────────────────────


class TestSerializationBranches:
    def test_str_len_serialization(self):
        df = pl.DataFrame({"name": ["abc"]})
        v = Validate(data=df, tbl_name="test").col_vals_str_len(
            columns="name", min_val=2, max_val=5
        )
        code = v.to_code()
        assert "col_vals_str_len" in code
        assert "min_val=2" in code
        assert "max_val=5" in code

    def test_within_spec_serialization(self):
        df = pl.DataFrame({"email": ["test@example.com"]})
        v = Validate(data=df, tbl_name="test").col_vals_within_spec(columns="email", spec="email")
        code = v.to_code()
        assert "col_vals_within_spec" in code
        assert 'spec="email"' in code

    def test_increasing_serialization(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        v = Validate(data=df, tbl_name="test").col_vals_increasing(columns="x")
        code = v.to_code()
        assert "col_vals_increasing" in code

    def test_decreasing_serialization(self):
        df = pl.DataFrame({"x": [3, 2, 1]})
        v = Validate(data=df, tbl_name="test").col_vals_decreasing(columns="x")
        code = v.to_code()
        assert "col_vals_decreasing" in code

    def test_conjointly_serialization(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        import warnings

        v = Validate(data=df, tbl_name="test").conjointly(lambda df: df["a"] > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = v.to_code()
        assert "conjointly" in code

    def test_col_vals_expr_serialization(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        import warnings

        v = Validate(data=df, tbl_name="test").col_vals_expr(expr=lambda df: df["a"] > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = v.to_code()
        assert "col_vals_expr" in code

    def test_rows_distinct_with_columns_serialization(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        v = Validate(data=df, tbl_name="test").rows_distinct(columns_subset=["a", "b"])
        code = v.to_code()
        assert "rows_distinct" in code

    def test_rows_complete_serialization(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        v = Validate(data=df, tbl_name="test").rows_complete()
        code = v.to_code()
        assert "rows_complete" in code

    def test_col_count_match_inverse(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        v = Validate(data=df, tbl_name="test").col_count_match(count=3, inverse=True)
        code = v.to_code()
        assert "col_count_match" in code
        assert "inverse=True" in code

    def test_row_count_match_inverse(self):
        df = pl.DataFrame({"a": [1, 2]})
        v = Validate(data=df, tbl_name="test").row_count_match(count=5, inverse=True)
        code = v.to_code()
        assert "row_count_match" in code
        assert "inverse=True" in code

    def test_data_freshness_with_all_params(self):
        df = pl.DataFrame({"dt": [datetime.datetime(2024, 1, 1)]})
        v = Validate(data=df, tbl_name="test").data_freshness(
            column="dt",
            max_age="1 day",
            reference_time="2024-01-02T00:00:00",
            timezone="UTC",
            allow_tz_mismatch=True,
        )
        code = v.to_code()
        assert "data_freshness" in code
        assert "timezone" in code
        assert "allow_tz_mismatch=True" in code

    def test_active_false_serialization(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        v = Validate(data=df, tbl_name="test").col_vals_gt(columns="a", value=0, active=False)
        code = v.to_code()
        assert "active=False" in code

    def test_active_callable_serialization(self):
        import warnings

        df = pl.DataFrame({"a": [1, 2, 3]})
        v = Validate(data=df, tbl_name="test").col_vals_gt(
            columns="a", value=0, active=lambda df: True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = v.to_code()
        assert "col_vals_gt" in code

    def test_brief_serialization(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        v = Validate(data=df, tbl_name="test").col_vals_gt(
            columns="a", value=0, brief="Custom brief"
        )
        code = v.to_code()
        assert "Custom brief" in code

    def test_segments_serialization(self):
        df = pl.DataFrame({"a": [1, 2, 3], "grp": ["x", "y", "x"]})
        v = Validate(data=df, tbl_name="test").col_vals_gt(columns="a", value=0, segments="grp")
        code = v.to_code()
        assert "segments" in code

    def test_aggregate_method_with_tol_serialization(self):
        df = pl.DataFrame({"d": [100, 200, 300]})
        v = Validate(data=df, tbl_name="test").col_sum_eq(columns="d", value=600, tol=10)
        code = v.to_code()
        assert "col_sum_eq" in code
        assert "tol=10" in code

    def test_to_yaml_str_len(self):
        df = pl.DataFrame({"name": ["abc"]})
        v = Validate(data=df, tbl_name="test").col_vals_str_len(
            columns="name", min_val=2, max_val=5
        )
        yaml_str = v.to_yaml()
        assert "col_vals_str_len" in yaml_str

    def test_to_yaml_increasing(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        v = Validate(data=df, tbl_name="test").col_vals_increasing(columns="x")
        yaml_str = v.to_yaml()
        assert "col_vals_increasing" in yaml_str

    def test_to_yaml_col_count_match_inverse(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        v = Validate(data=df, tbl_name="test").col_count_match(count=3, inverse=True)
        yaml_str = v.to_yaml()
        assert "col_count_match" in yaml_str
        assert "inverse: true" in yaml_str

    def test_to_code_with_lang(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, tbl_name="test", lang="de").col_vals_gt(columns="a", value=0)
        code = v.to_code()
        assert 'lang="de"' in code

    def test_to_code_with_consumers(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, tbl_name="test", consumers=["team-a"]).col_vals_gt(
            columns="a", value=0
        )
        code = v.to_code()
        assert "consumers" in code

    def test_to_code_with_locale(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, tbl_name="test", lang="en", locale="en_US").col_vals_gt(
            columns="a", value=0
        )
        code = v.to_code()
        assert 'locale="en_US"' in code

    def test_to_code_with_brief_true(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, tbl_name="test", brief=True).col_vals_gt(columns="a", value=0)
        code = v.to_code()
        assert "brief=True" in code


# ─── _validation_info_to_step edge cases ────────────────────────────────────────


class TestValidationInfoToStep:
    def test_regex_with_inverse(self):
        df = pl.DataFrame({"b": ["abc", "123"]})
        v = Validate(data=df, tbl_name="test").col_vals_regex(
            columns="b", pattern="^[a-z]+$", inverse=True
        )
        code = v.to_code()
        assert "inverse=True" in code

    def test_schema_match_non_default_flags(self):
        df = pl.DataFrame({"A": [1], "b": [2]})
        schema = pb.Schema(columns=[("A", "Int64")])
        v = Validate(data=df, tbl_name="test").col_schema_match(
            schema=schema, complete=False, in_order=False
        )
        code = v.to_code()
        assert "complete=False" in code
        assert "in_order=False" in code

    def test_increasing_with_allow_stationary(self):
        df = pl.DataFrame({"x": [1, 1, 2, 3]})
        v = Validate(data=df, tbl_name="test").col_vals_increasing(
            columns="x", allow_stationary=True
        )
        code = v.to_code()
        assert "allow_stationary=True" in code

    def test_rows_distinct_single_column(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        v = Validate(data=df, tbl_name="test").rows_distinct(columns_subset="a")
        code = v.to_code()
        assert "rows_distinct" in code
        assert 'columns_subset="a"' in code

    def test_step_level_actions_serialization(self):
        import warnings

        df = pl.DataFrame({"a": [1, 2, 3]})
        v = Validate(data=df, tbl_name="test").col_vals_gt(
            columns="a", value=0, actions=pb.Actions(warning="warn")
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = v.to_code()
        assert "col_vals_gt" in code

    def test_to_code_with_actions_warning(self):
        import warnings

        df = pl.DataFrame({"a": [1, 2, 3]})
        v = Validate(data=df, tbl_name="test", actions=pb.Actions(warning="warn")).col_vals_gt(
            columns="a", value=0
        )
        with pytest.warns(UserWarning, match="could not be fully serialized"):
            code = v.to_code()
        assert "col_vals_gt" in code


# ─── to_yaml() top-level args coverage ─────────────────────────────────────────


class TestToYamlTopLevelArgs:
    def test_with_thresholds(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, tbl_name="test", thresholds=pb.Thresholds(warning=0.1)).col_vals_gt(
            columns="a", value=0
        )
        yaml_str = v.to_yaml()
        assert "thresholds" in yaml_str

    def test_with_brief(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, tbl_name="test", brief=True).col_vals_gt(columns="a", value=0)
        yaml_str = v.to_yaml()
        assert "brief: true" in yaml_str

    def test_with_lang(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, tbl_name="test", lang="de").col_vals_gt(columns="a", value=0)
        yaml_str = v.to_yaml()
        assert "lang: de" in yaml_str

    def test_with_locale(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, tbl_name="test", lang="en", locale="en_US").col_vals_gt(
            columns="a", value=0
        )
        yaml_str = v.to_yaml()
        assert "locale: en_US" in yaml_str

    def test_with_owner_and_version(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, tbl_name="test", owner="data-team", version="1.0").col_vals_gt(
            columns="a", value=0
        )
        yaml_str = v.to_yaml()
        assert "owner: data-team" in yaml_str
        assert "version: '1.0'" in yaml_str

    def test_with_consumers(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, tbl_name="test", consumers=["team-a"]).col_vals_gt(
            columns="a", value=0
        )
        yaml_str = v.to_yaml()
        assert "consumers" in yaml_str

    def test_with_actions_warning(self):
        import warnings

        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, tbl_name="test", actions=pb.Actions(warning="warn")).col_vals_gt(
            columns="a", value=0
        )
        with pytest.warns(UserWarning, match="could not be fully serialized"):
            v.to_yaml()


# ─── _parse_timezone / _parse_max_age edge cases ───────────────────────────────


class TestParseTimezone:
    def test_invalid_timezone_raises(self):
        from pointblank.validate import _parse_timezone

        with pytest.raises(ValueError, match="Invalid timezone"):
            _parse_timezone("Not/A/Real/Zone")


class TestParseMaxAge:
    def test_invalid_type_raises(self):
        from pointblank.validate import _parse_max_age

        with pytest.raises(TypeError, match="must be a string or timedelta"):
            _parse_max_age(42)


# ─── data_freshness invalid reference_time ──────────────────────────────────────


class TestDataFreshnessEdgeCases:
    def test_invalid_reference_time_type(self):
        df = pl.DataFrame({"dt": [datetime.datetime(2024, 1, 1)]})
        with pytest.raises(TypeError, match="must be a string or datetime"):
            Validate(data=df).data_freshness(column="dt", max_age="1 day", reference_time=42)


# ─── col_missing_consistent string columns input ───────────────────────────────


class TestColMissingConsistentStringInput:
    def test_string_columns_converted_to_list(self):
        from pointblank.missing import MissingSpec

        spec = MissingSpec(reasons={-99: "not_asked"})
        df = pl.DataFrame({"a": [1, -99], "b": [2, -99]})
        with pytest.raises(ValueError, match="requires at least two columns"):
            Validate(data=df).col_missing_consistent(
                columns="a", missing=spec, when_reason="not_asked"
            )


# ─── _format_timedelta ──────────────────────────────────────────────────────────


class TestFormatTimedelta:
    def test_seconds(self):
        assert _format_timedelta(datetime.timedelta(seconds=30)) == "30.0s"

    def test_sub_second(self):
        assert _format_timedelta(datetime.timedelta(seconds=0.5)) == "0.5s"

    def test_minutes(self):
        assert _format_timedelta(datetime.timedelta(minutes=5)) == "5.0m"

    def test_minutes_fractional(self):
        result = _format_timedelta(datetime.timedelta(seconds=90))
        assert result == "1.5m"

    def test_hours(self):
        assert _format_timedelta(datetime.timedelta(hours=3)) == "3.0h"

    def test_hours_fractional(self):
        result = _format_timedelta(datetime.timedelta(hours=1, minutes=30))
        assert result == "1.5h"

    def test_days_exact(self):
        assert _format_timedelta(datetime.timedelta(days=2)) == "2d"

    def test_days_with_hours(self):
        result = _format_timedelta(datetime.timedelta(days=1, hours=6))
        assert result == "1d 6.0h"

    def test_weeks(self):
        result = _format_timedelta(datetime.timedelta(weeks=2))
        assert result == "2w"

    def test_boundary_under_minute(self):
        assert _format_timedelta(datetime.timedelta(seconds=59)) == "59.0s"

    def test_boundary_exactly_one_minute(self):
        assert _format_timedelta(datetime.timedelta(minutes=1)) == "1.0m"

    def test_boundary_exactly_one_hour(self):
        assert _format_timedelta(datetime.timedelta(hours=1)) == "1.0h"

    def test_boundary_exactly_one_day(self):
        assert _format_timedelta(datetime.timedelta(days=1)) == "1d"

    def test_boundary_exactly_one_week(self):
        assert _format_timedelta(datetime.timedelta(weeks=1)) == "1w"


# ─── _parse_max_age ─────────────────────────────────────────────────────────────


class TestParseMaxAge:
    def test_timedelta_passthrough(self):
        td = datetime.timedelta(hours=1)
        assert _parse_max_age(td) is td

    def test_simple_hours(self):
        result = _parse_max_age("24 hours")
        assert result == datetime.timedelta(hours=24)

    def test_simple_day(self):
        result = _parse_max_age("1 day")
        assert result == datetime.timedelta(days=1)

    def test_simple_minutes(self):
        result = _parse_max_age("30 minutes")
        assert result == datetime.timedelta(minutes=30)

    def test_simple_seconds(self):
        result = _parse_max_age("60 seconds")
        assert result == datetime.timedelta(seconds=60)

    def test_simple_weeks(self):
        result = _parse_max_age("2 weeks")
        assert result == datetime.timedelta(weeks=2)

    def test_compound_expression(self):
        result = _parse_max_age("2 hours 15 minutes")
        assert result == datetime.timedelta(hours=2, minutes=15)

    def test_compound_no_spaces(self):
        result = _parse_max_age("1day6h")
        assert result == datetime.timedelta(days=1, hours=6)

    def test_abbreviations_sec(self):
        assert _parse_max_age("30sec") == datetime.timedelta(seconds=30)

    def test_abbreviations_min(self):
        assert _parse_max_age("5min") == datetime.timedelta(minutes=5)

    def test_abbreviations_hr(self):
        assert _parse_max_age("2hr") == datetime.timedelta(hours=2)

    def test_abbreviations_d(self):
        assert _parse_max_age("3d") == datetime.timedelta(days=3)

    def test_abbreviations_wk(self):
        assert _parse_max_age("1wk") == datetime.timedelta(weeks=1)

    def test_abbreviations_single_char(self):
        assert _parse_max_age("5s") == datetime.timedelta(seconds=5)
        assert _parse_max_age("5m") == datetime.timedelta(minutes=5)
        assert _parse_max_age("5h") == datetime.timedelta(hours=5)
        assert _parse_max_age("5d") == datetime.timedelta(days=5)
        assert _parse_max_age("5w") == datetime.timedelta(weeks=5)

    def test_fractional_value(self):
        result = _parse_max_age("1.5 hours")
        assert result == datetime.timedelta(hours=1.5)

    def test_invalid_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown time unit"):
            _parse_max_age("5 fortnights")

    def test_no_match_raises(self):
        with pytest.raises(ValueError, match="Invalid max_age format"):
            _parse_max_age("foo bar")

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="must be a string or timedelta"):
            _parse_max_age(42)

    def test_whitespace_handling(self):
        result = _parse_max_age("  24 hours  ")
        assert result == datetime.timedelta(hours=24)

    def test_plural_forms(self):
        assert _parse_max_age("1 second") == datetime.timedelta(seconds=1)
        assert _parse_max_age("2 secs") == datetime.timedelta(seconds=2)
        assert _parse_max_age("1 minute") == datetime.timedelta(minutes=1)
        assert _parse_max_age("2 mins") == datetime.timedelta(minutes=2)
        assert _parse_max_age("1 hour") == datetime.timedelta(hours=1)
        assert _parse_max_age("2 hrs") == datetime.timedelta(hours=2)
        assert _parse_max_age("1 week") == datetime.timedelta(weeks=1)
        assert _parse_max_age("2 wks") == datetime.timedelta(weeks=2)


# ─── _parse_timezone ─────────────────────────────────────────────────────────────


class TestParseTimezone:
    def test_iana_utc(self):
        tz = _parse_timezone("UTC")
        now = datetime.datetime.now(tz)
        assert now.utcoffset() == datetime.timedelta(0)

    def test_iana_named(self):
        tz = _parse_timezone("America/New_York")
        assert tz is not None

    def test_positive_offset_simple(self):
        tz = _parse_timezone("+5")
        offset = datetime.datetime(2024, 1, 1, tzinfo=tz).utcoffset()
        assert offset == datetime.timedelta(hours=5)

    def test_negative_offset_simple(self):
        tz = _parse_timezone("-7")
        offset = datetime.datetime(2024, 1, 1, tzinfo=tz).utcoffset()
        assert offset == datetime.timedelta(hours=-7)

    def test_offset_with_colon(self):
        tz = _parse_timezone("+05:30")
        offset = datetime.datetime(2024, 1, 1, tzinfo=tz).utcoffset()
        assert offset == datetime.timedelta(hours=5, minutes=30)

    def test_negative_offset_with_colon(self):
        tz = _parse_timezone("-07:00")
        offset = datetime.datetime(2024, 1, 1, tzinfo=tz).utcoffset()
        assert offset == datetime.timedelta(hours=-7)

    def test_unsigned_offset(self):
        tz = _parse_timezone("5")
        offset = datetime.datetime(2024, 1, 1, tzinfo=tz).utcoffset()
        assert offset == datetime.timedelta(hours=5)

    def test_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid timezone"):
            _parse_timezone("Not/A/Real/Zone")

    def test_whitespace_stripped(self):
        tz = _parse_timezone("  +5  ")
        offset = datetime.datetime(2024, 1, 1, tzinfo=tz).utcoffset()
        assert offset == datetime.timedelta(hours=5)


# ─── _transform_auto_brief ──────────────────────────────────────────────────────


class TestTransformAutoBrief:
    def test_true_becomes_auto(self):
        assert _transform_auto_brief(True) == "{auto}"

    def test_false_becomes_none(self):
        assert _transform_auto_brief(False) is None

    def test_string_passthrough(self):
        assert _transform_auto_brief("custom brief") == "custom brief"

    def test_none_passthrough(self):
        assert _transform_auto_brief(None) is None


# ─── _coalesce_plan_steps (non-adjacent) ─────────────────────────────────────────


class TestCoalescePlanStepsNonAdjacent:
    def test_interleaved_methods_not_merged(self):
        steps = [
            ("col_vals_not_null", {"columns": "a"}),
            ("col_vals_gt", {"columns": "b", "value": 0}),
            ("col_vals_not_null", {"columns": "c"}),
        ]
        result = _coalesce_plan_steps(steps)
        assert len(result) == 3

    def test_adjacent_with_different_extra_params_not_merged(self):
        steps = [
            ("col_vals_gt", {"columns": "a", "value": 0}),
            ("col_vals_gt", {"columns": "b", "value": 0, "na_pass": True}),
        ]
        result = _coalesce_plan_steps(steps)
        assert len(result) == 2

    def test_mixed_coalescible_and_not(self):
        steps = [
            ("col_vals_not_null", {"columns": "a"}),
            ("col_vals_not_null", {"columns": "b"}),
            ("col_vals_gt", {"columns": "c", "value": 0}),
            ("col_vals_gt", {"columns": "d", "value": 0}),
        ]
        result = _coalesce_plan_steps(steps)
        assert len(result) == 2
        assert result[0][1]["columns"] == ["a", "b"]
        assert result[1][1]["columns"] == ["c", "d"]

    def test_single_step(self):
        steps = [("col_vals_not_null", {"columns": "a"})]
        result = _coalesce_plan_steps(steps)
        assert len(result) == 1
        assert result[0][1]["columns"] == "a"


# ─── Threshold normalization via public API ──────────────────────────────────────


class TestThresholdNormalization:
    def test_bare_float(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, thresholds=0.1)
        assert v.thresholds is not None

    def test_tuple_two_levels(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, thresholds=(0.1, 0.25))
        assert v.thresholds is not None

    def test_tuple_three_levels(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, thresholds=(0.1, 0.25, 0.5))
        assert v.thresholds is not None

    def test_dict_form(self):
        df = pl.DataFrame({"a": [1]})
        v = Validate(data=df, thresholds={"warning": 0.1, "error": 0.25})
        assert v.thresholds is not None

    def test_thresholds_object(self):
        df = pl.DataFrame({"a": [1]})
        t = pb.Thresholds(warning=0.1)
        v = Validate(data=df, thresholds=t)
        assert v.thresholds is t


# ─── Multi-step integration ──────────────────────────────────────────────────────


class TestMultiStepIntegration:
    def test_diverse_chain_interrogate_and_query(self):
        df = pl.DataFrame(
            {
                "a": [1, 2, 3, 4, 5],
                "b": ["x", "y", "x", "y", "z"],
                "c": [10.0, 20.0, None, 40.0, 50.0],
            }
        )
        v = (
            Validate(data=df)
            .col_vals_gt(columns="a", value=0)
            .col_vals_in_set(columns="b", set=["x", "y", "z"])
            .col_vals_not_null(columns="c")
            .col_vals_between(columns="a", left=1, right=5)
            .rows_distinct()
            .interrogate()
        )
        assert v.n_passed(i=1, scalar=True) == 5
        assert v.n_passed(i=2, scalar=True) == 5
        assert v.n_passed(i=3, scalar=True) == 4
        assert v.n_failed(i=3, scalar=True) == 1
        assert v.n_passed(i=4, scalar=True) == 5

    def test_all_passed_false_with_failures(self):
        df = pl.DataFrame({"a": [1, 2, -1]})
        v = Validate(data=df).col_vals_gt(columns="a", value=0).interrogate()
        assert v.all_passed() is False

    def test_all_passed_true(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        v = Validate(data=df).col_vals_gt(columns="a", value=0).interrogate()
        assert v.all_passed() is True

    def test_get_data_extracts(self):
        df = pl.DataFrame({"a": [1, 2, -1, 3]})
        v = Validate(data=df).col_vals_gt(columns="a", value=0).interrogate()
        extracts = v.get_data_extracts(i=1)
        assert len(extracts) == 1

    def test_get_sundered_data_pass(self):
        df = pl.DataFrame({"a": [1, 2, -1, 3], "b": [10, 20, 30, 40]})
        v = Validate(data=df).col_vals_gt(columns="a", value=0).interrogate()
        passed = v.get_sundered_data(type="pass")
        assert len(passed) == 3

    def test_get_sundered_data_fail(self):
        df = pl.DataFrame({"a": [1, 2, -1, 3], "b": [10, 20, 30, 40]})
        v = Validate(data=df).col_vals_gt(columns="a", value=0).interrogate()
        failed = v.get_sundered_data(type="fail")
        assert len(failed) == 1


# ─── to_code/to_yaml round-trips for more assertion types ──────────────────────


class TestSerializationRoundTrips:
    def _exec_code(self, code, data):
        namespace = {}
        exec(code.replace("your_data", "data"), {"pb": pb, "data": data}, namespace)
        return namespace["validation"]

    def test_within_spec_roundtrip(self):
        df = pl.DataFrame({"email": ["test@example.com"]})
        v = Validate(data=df, tbl_name="test").col_vals_within_spec(columns="email", spec="email")
        code = v.to_code()
        rebuilt = self._exec_code(code, df)
        assert rebuilt.validation_info[0].assertion_type == "col_vals_within_spec"

    def test_str_len_roundtrip(self):
        df = pl.DataFrame({"name": ["abc"]})
        v = Validate(data=df, tbl_name="test").col_vals_str_len(
            columns="name", min_val=2, max_val=10
        )
        code = v.to_code()
        rebuilt = self._exec_code(code, df)
        assert rebuilt.validation_info[0].assertion_type == "col_vals_str_len"
        assert rebuilt.validation_info[0].values["min_val"] == 2
        assert rebuilt.validation_info[0].values["max_val"] == 10

    def test_increasing_decreasing_roundtrip(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        v = (
            Validate(data=df, tbl_name="test")
            .col_vals_increasing(columns="x")
            .col_vals_decreasing(columns="x")
        )
        code = v.to_code()
        rebuilt = self._exec_code(code, df)
        assert rebuilt.validation_info[0].assertion_type == "col_vals_increasing"
        assert rebuilt.validation_info[1].assertion_type == "col_vals_decreasing"

    def test_col_pct_missing_not_serializable(self):
        from pointblank.missing import MissingSpec

        spec = MissingSpec(reasons={-99: "not_asked"})
        df = pl.DataFrame({"age": [25, -99, 30]})
        v = Validate(data=df, tbl_name="test").col_pct_missing(
            columns="age", missing=spec, max_pct=0.5
        )
        with pytest.warns(UserWarning, match="col_pct_missing"):
            code = v.to_code()
        assert "col_pct_missing" not in code

    def test_col_missing_coded_not_serializable(self):
        from pointblank.missing import MissingSpec

        spec = MissingSpec(reasons={-99: "not_asked"})
        df = pl.DataFrame({"age": [25, -99, 30]})
        v = Validate(data=df, tbl_name="test").col_missing_coded(columns="age", missing=spec)
        with pytest.warns(UserWarning, match="col_missing_coded"):
            code = v.to_code()
        assert "col_missing_coded" not in code

    def test_col_missing_consistent_not_serializable(self):
        from pointblank.missing import MissingSpec

        spec = MissingSpec(reasons={-99: "not_asked"})
        df = pl.DataFrame({"a": [1, -99], "b": [2, -99]})
        v = Validate(data=df, tbl_name="test").col_missing_consistent(
            columns=["a", "b"], missing=spec, when_reason="not_asked"
        )
        with pytest.warns(UserWarning, match="col_missing_consistent"):
            code = v.to_code()
        assert "col_missing_consistent" not in code

    def test_yaml_within_spec(self):
        df = pl.DataFrame({"email": ["test@example.com"]})
        v = Validate(data=df, tbl_name="test").col_vals_within_spec(columns="email", spec="email")
        yaml_str = v.to_yaml()
        assert "col_vals_within_spec" in yaml_str
        assert "email" in yaml_str

    def test_yaml_str_len(self):
        df = pl.DataFrame({"name": ["abc"]})
        v = Validate(data=df, tbl_name="test").col_vals_str_len(
            columns="name", min_val=2, max_val=10
        )
        yaml_str = v.to_yaml()
        assert "col_vals_str_len" in yaml_str
        assert "min_val: 2" in yaml_str
        assert "max_val: 10" in yaml_str

    def test_aggregate_with_tol_roundtrip(self):
        df = pl.DataFrame({"d": [100, 200, 300]})
        v = Validate(data=df, tbl_name="test").col_sum_eq(columns="d", value=600, tol=10)
        code = v.to_code()
        rebuilt = self._exec_code(code, df)
        assert rebuilt.validation_info[0].assertion_type == "col_sum_eq"

    def test_data_freshness_yaml(self):
        df = pl.DataFrame({"dt": [datetime.datetime(2024, 1, 1)]})
        v = Validate(data=df, tbl_name="test").data_freshness(column="dt", max_age="1 day")
        yaml_str = v.to_yaml()
        assert "data_freshness" in yaml_str
        assert "max_age" in yaml_str


# ─── data_freshness with timezone offsets (end-to-end) ──────────────────────────


class TestDataFreshnessTimezones:
    def test_with_utc_timezone(self):
        import datetime as dt

        now = dt.datetime.now(dt.timezone.utc)
        df = pl.DataFrame({"ts": [now - dt.timedelta(hours=1)]})
        v = (
            Validate(data=df)
            .data_freshness(column="ts", max_age="2 hours", timezone="UTC")
            .interrogate()
        )
        assert v.n_passed(i=1, scalar=True) == 1

    def test_with_numeric_offset(self):
        import datetime as dt

        tz = dt.timezone(dt.timedelta(hours=-5))
        now = dt.datetime.now(tz)
        df = pl.DataFrame({"ts": [now - dt.timedelta(minutes=30)]})
        v = (
            Validate(data=df)
            .data_freshness(column="ts", max_age="1 hour", timezone="-5")
            .interrogate()
        )
        assert v.n_passed(i=1, scalar=True) == 1


# ─── col_vals_str_len extended scenarios ─────────────────────────────────────────


class TestColValsStrLenExtended:
    def test_multi_column(self):
        df = pl.DataFrame({"a": ["abc", "ab"], "b": ["xyz", "xy"]})
        v = Validate(data=df).col_vals_str_len(columns=["a", "b"], min_val=2).interrogate()
        assert v.n_passed(i=1, scalar=True) == 2
        assert v.n_passed(i=2, scalar=True) == 2

    def test_with_thresholds(self):
        df = pl.DataFrame({"name": ["a", "ab", "abc"]})
        v = (
            Validate(data=df)
            .col_vals_str_len(columns="name", min_val=2, thresholds=pb.Thresholds(warning=0.5))
            .interrogate()
        )
        assert v.n_failed(i=1, scalar=True) == 1

    def test_with_brief(self):
        df = pl.DataFrame({"name": ["abc"]})
        v = Validate(data=df).col_vals_str_len(columns="name", min_val=1, brief="Custom brief")
        assert v.validation_info[0].brief == "Custom brief"

    def test_with_pandas(self):
        import pandas as pd

        df = pd.DataFrame({"name": ["abc", "ab", "abcde"]})
        v = Validate(data=df).col_vals_str_len(columns="name", min_val=3).interrogate()
        assert v.n_passed(i=1, scalar=True) == 2
