"""Tests for data quality dimensions and health scoring (PLAN_09)."""

import pytest

import pointblank as pb
from pointblank.validate import (
    Validate,
    FinalActions,
    get_validation_summary,
    global_config,
    _infer_dimension_from_assertion_type,
    _base_dimension_from_assertion_type,
    _compute_dimension_scores,
    _compute_health_score,
    _aggregate_dimension_units,
    _validation_info_as_dict,
    _get_dimension_label,
)


@pytest.fixture(autouse=True)
def reset_global_config():
    """Save and restore the dimension-related global config around each test."""
    saved = (
        global_config.dimension_map,
        global_config.dimension_weights,
        global_config.dimension_thresholds,
    )
    global_config.dimension_map = None
    global_config.dimension_weights = None
    global_config.dimension_thresholds = None
    yield
    (
        global_config.dimension_map,
        global_config.dimension_weights,
        global_config.dimension_thresholds,
    ) = saved


@pytest.fixture
def simple_df():
    pl = pytest.importorskip("polars")
    # `a`: 1 null (completeness 4/5); `b`: 2 non-positive (validity 3/5); `id`: distinct
    return pl.DataFrame(
        {
            "a": [1, 2, 3, None, 5],
            "b": [10, -3, 5, 0, 8],
            "id": [1, 2, 3, 4, 5],
        }
    )


# ---------------------------------------------------------------------------
# Dimension inference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "assertion_type,expected",
    [
        ("col_vals_not_null", "completeness"),
        ("col_pct_null", "completeness"),
        ("rows_complete", "completeness"),
        ("col_vals_gt", "validity"),
        ("col_vals_regex", "validity"),
        ("col_schema_match", "validity"),
        ("rows_distinct", "uniqueness"),
        ("col_missing_consistent", "consistency"),
        ("conjointly", "consistency"),
        ("tbl_match", "consistency"),
        ("data_freshness", "timeliness"),
        ("row_count_match", "volume"),
        ("col_count_match", "volume"),
        # dynamically generated aggregate methods -> validity via regex fallback
        ("col_sum_eq", "validity"),
        ("col_avg_gt", "validity"),
        ("col_sd_le", "validity"),
    ],
)
def test_infer_dimension_from_assertion_type(assertion_type, expected):
    assert _infer_dimension_from_assertion_type(assertion_type) == expected


def test_infer_dimension_none_and_unknown():
    assert _infer_dimension_from_assertion_type(None) is None
    assert _infer_dimension_from_assertion_type("not_a_real_assertion") == "unknown"


def test_auto_inference_on_steps(simple_df):
    v = (
        Validate(data=simple_df)
        .col_vals_not_null(columns="a")
        .col_vals_gt(columns="b", value=0)
        .rows_distinct(columns_subset=["id"])
        .col_sum_eq(columns="id", value=15)
    )
    dims = [s.dimension for s in v.validation_info]
    assert dims == ["completeness", "validity", "uniqueness", "validity"]


def test_per_step_dimension_override(simple_df):
    v = (
        Validate(data=simple_df)
        .col_vals_gt(columns="b", value=0, dimension="consistency")
        .col_vals_gt(columns="b", value=0)
    )
    assert v.validation_info[0].dimension == "consistency"
    assert v.validation_info[1].dimension == "validity"


def test_agg_method_dimension_override(simple_df):
    v = Validate(data=simple_df).col_sum_eq(columns="id", value=15, dimension="volume")
    assert v.validation_info[0].dimension == "volume"


def test_config_dimension_map_remap(simple_df):
    pb.config(dimension_map={"col_vals_gt": "consistency"})
    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0)
    assert v.validation_info[0].dimension == "consistency"


def test_config_dimension_map_does_not_override_explicit(simple_df):
    pb.config(dimension_map={"col_vals_gt": "consistency"})
    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0, dimension="validity")
    # Explicit per-step dimension wins over the config remap
    assert v.validation_info[0].dimension == "validity"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_dimension_scores(simple_df):
    v = (
        Validate(data=simple_df)
        .col_vals_not_null(columns="a")  # completeness 4/5 = 80
        .col_vals_gt(columns="b", value=0)  # validity 3/5 = 60
        .rows_distinct(columns_subset=["id"])  # uniqueness 5/5 = 100
        .interrogate()
    )
    scores = v.get_dimension_scores()
    assert scores == {"completeness": 80.0, "validity": 60.0, "uniqueness": 100.0}


def test_health_score_unweighted(simple_df):
    v = (
        Validate(data=simple_df)
        .col_vals_not_null(columns="a")
        .col_vals_gt(columns="b", value=0)
        .rows_distinct(columns_subset=["id"])
        .interrogate()
    )
    # (4 + 3 + 5) / (5 + 5 + 5) = 12/15 = 80.0
    assert v.get_health_score() == 80.0


def test_health_score_weighted(simple_df):
    pb.config(dimension_weights={"validity": 3.0})
    v = (
        Validate(data=simple_df)
        .col_vals_not_null(columns="a")  # completeness 4/5
        .col_vals_gt(columns="b", value=0)  # validity 3/5, weight 3
        .interrogate()
    )
    # num = 1*4 + 3*3 = 13 ; den = 1*5 + 3*5 = 20 -> 65.0
    assert v.get_health_score() == 65.0


def test_scores_empty_when_not_interrogated(simple_df):
    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0)
    assert v.get_dimension_scores() == {}
    assert v.get_health_score() == 100.0


def test_inactive_steps_excluded_from_scoring(simple_df):
    v = (
        Validate(data=simple_df)
        .col_vals_gt(columns="b", value=0, active=False)  # would be validity 3/5
        .col_vals_not_null(columns="a")  # completeness 4/5
        .interrogate()
    )
    scores = v.get_dimension_scores()
    assert "validity" not in scores
    assert scores == {"completeness": 80.0}


def test_aggregate_dimension_units(simple_df):
    v = (
        Validate(data=simple_df)
        .col_vals_not_null(columns="a")
        .col_vals_gt(columns="b", value=0)
        .interrogate()
    )
    agg = _aggregate_dimension_units(v.validation_info)
    assert agg == {"completeness": [4, 5], "validity": [3, 5]}


def test_compute_helpers_directly(simple_df):
    v = (
        Validate(data=simple_df)
        .col_vals_not_null(columns="a")
        .col_vals_gt(columns="b", value=0)
        .interrogate()
    )
    assert _compute_dimension_scores(v.validation_info) == {
        "completeness": 80.0,
        "validity": 60.0,
    }
    assert _compute_health_score(v.validation_info) == 70.0
    assert _compute_health_score(v.validation_info, dimension_weights={"validity": 3.0}) == 65.0


# ---------------------------------------------------------------------------
# assert_dimension_scores
# ---------------------------------------------------------------------------


def test_assert_dimension_scores_pass(simple_df):
    v = Validate(data=simple_df).col_vals_not_null(columns="a").interrogate()
    # completeness is 80, so a 75 minimum passes (no exception)
    v.assert_dimension_scores(thresholds={"completeness": 75})


def test_assert_dimension_scores_raises(simple_df):
    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0).interrogate()
    with pytest.raises(AssertionError, match="validity"):
        v.assert_dimension_scores(thresholds={"validity": 90})


def test_assert_dimension_scores_uses_config_default(simple_df):
    pb.config(dimension_thresholds={"validity": 90})
    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0).interrogate()
    with pytest.raises(AssertionError):
        v.assert_dimension_scores()


def test_assert_dimension_scores_no_thresholds_is_noop(simple_df):
    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0).interrogate()
    # No thresholds set anywhere -> no exception
    v.assert_dimension_scores()


def test_assert_dimension_scores_custom_message(simple_df):
    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0).interrogate()
    with pytest.raises(AssertionError, match="custom failure"):
        v.assert_dimension_scores(thresholds={"validity": 90}, message="custom failure")


def test_assert_dimension_scores_ignores_absent_dimension(simple_df):
    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0).interrogate()
    # 'timeliness' is not present in the plan, so it is ignored
    v.assert_dimension_scores(thresholds={"timeliness": 100})


# ---------------------------------------------------------------------------
# Report + dict integration
# ---------------------------------------------------------------------------


def test_dimension_in_validation_info_dict(simple_df):
    v = Validate(data=simple_df).col_vals_not_null(columns="a").interrogate()
    d = _validation_info_as_dict(v.validation_info)
    assert "dimension" in d
    assert d["dimension"] == ["completeness"]


def test_dimension_in_json_report(simple_df):
    import json

    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0).interrogate()
    report = json.loads(v.get_json_report())
    assert report[0]["dimension"] == "validity"


def test_dimension_via_yaml(simple_df):
    yaml_cfg = """
tbl: small_table
steps:
  - col_vals_gt:
      columns: d
      value: 0
      dimension: consistency
"""
    v = pb.yaml_interrogate(yaml_cfg)
    assert v.validation_info[0].dimension == "consistency"


def test_report_includes_dimension_badges_and_scores(simple_df):
    v = (
        Validate(data=simple_df, tbl_name="demo")
        .col_vals_not_null(columns="a")
        .col_vals_gt(columns="b", value=0)
        .interrogate()
    )
    html = v.get_tabular_report().as_raw_html()
    # No standalone dimension column; instead a compact badge with a tooltip on the step number
    assert 'title="Completeness"' in html
    assert 'title="Validity"' in html
    assert ">CM<" in html  # completeness abbreviation badge
    assert ">VA<" in html  # validity abbreviation badge
    assert "Health Score" in html


def test_report_renders_without_interrogation(simple_df):
    v = Validate(data=simple_df).col_vals_not_null(columns="a")
    html = v.get_tabular_report().as_raw_html()
    # Dimension badge still shown for the plan, but no health-score block yet
    assert 'title="Completeness"' in html
    assert "Health Score" not in html


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------


def test_get_scorecard_renders(simple_df):
    v = (
        Validate(data=simple_df, tbl_name="orders")
        .col_vals_not_null(columns="a")
        .col_vals_gt(columns="b", value=0)
        .rows_distinct(columns_subset=["id"])
        .interrogate()
    )
    html = v.get_scorecard().as_raw_html()
    assert "Health Score" in html
    assert "80%" in html
    assert "Completeness" in html
    assert "Dimension Scores" in html


def test_get_scorecard_empty(simple_df):
    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0)  # not interrogated
    # Should render a minimal table without raising
    assert len(v.get_scorecard().as_raw_html()) > 0


def test_get_scorecard_title_none(simple_df):
    v = Validate(data=simple_df).col_vals_not_null(columns="a").interrogate()
    html = v.get_scorecard(title=":none:").as_raw_html()
    assert len(html) > 0


# ---------------------------------------------------------------------------
# Validation summary integration
# ---------------------------------------------------------------------------


def test_validation_summary_includes_scores():
    pl = pytest.importorskip("polars")
    captured = {}

    def capture():
        summary = get_validation_summary()
        captured["dimension_scores"] = summary["dimension_scores"]
        captured["overall_health_score"] = summary["overall_health_score"]

    df = pl.DataFrame({"a": [1, 2, None, 4, 5], "b": [10, -3, 5, 0, 8]})
    (
        Validate(data=df, final_actions=FinalActions(capture))
        .col_vals_not_null(columns="a")
        .col_vals_gt(columns="b", value=0)
        .interrogate()
    )

    assert captured["dimension_scores"] == {"completeness": 80.0, "validity": 60.0}
    assert captured["overall_health_score"] == 70.0


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_dimension_label_english():
    assert _get_dimension_label("completeness", "en") == "Completeness"
    assert _get_dimension_label("validity", "en") == "Validity"


def test_dimension_label_localized():
    assert _get_dimension_label("completeness", "fr") == "Complétude"
    assert _get_dimension_label("validity", "de") == "Gültigkeit"


def test_dimension_label_localized_non_latin():
    # Non-Latin languages are also translated (not falling back to English)
    assert _get_dimension_label("completeness", "ja") == "完全性"
    assert _get_dimension_label("timeliness", "ru") == "Своевременность"


def test_dimension_label_fallback_to_english_for_missing_language():
    # A language code without an explicit translation falls back to English
    assert _get_dimension_label("completeness", "xx") == "Completeness"


def test_dimension_translation_keys_cover_all_languages():
    # Every dimension/scoring report string must be translated for all supported languages
    from pointblank._constants import REPORTING_LANGUAGES
    from pointblank._constants_translations import VALIDATION_REPORT_TEXT

    keys = [
        "report_col_dimension",
        "report_col_score",
        "report_health_score",
        "report_dimension_scores",
        "dimension_completeness",
        "dimension_consistency",
        "dimension_validity",
        "dimension_uniqueness",
        "dimension_timeliness",
        "dimension_volume",
        "dimension_unknown",
    ]
    for key in keys:
        langs = set(VALIDATION_REPORT_TEXT[key])
        missing = [lang for lang in REPORTING_LANGUAGES if lang not in langs]
        assert not missing, f"{key} is missing translations for: {missing}"


def test_dimension_label_custom_dimension_title_cased():
    # A custom, unmapped dimension is presented in a readable title-cased form
    assert _get_dimension_label("business_rules", "en") == "Business Rules"


def test_custom_dimension_scored_and_rendered():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2, 3, None, 5]})
    v = (
        Validate(data=df, tbl_name="t")
        .col_vals_not_null(columns="a", dimension="business_rules")
        .interrogate()
    )
    assert v.get_dimension_scores() == {"business_rules": 80.0}
    html = v.get_tabular_report().as_raw_html()
    # Custom dimension: tooltip shows the title-cased name; badge derives a two-letter code
    assert 'title="Business Rules"' in html
    assert ">BU<" in html


def test_eval_error_step_excluded_from_scoring():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    v = (
        Validate(df)
        .col_vals_gt(columns="nonexistent", value=0)  # eval_error -> excluded
        .col_vals_gt(columns="b", value=0)  # validity 3/3
        .interrogate()
    )

    # The broken step must not drag the score to 0
    assert v.get_dimension_scores() == {"validity": 100.0}
    assert v.get_health_score() == 100.0


def test_only_eval_error_yields_empty_scores():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2, 3]})
    v = Validate(df).col_vals_gt(columns="nonexistent", value=0).interrogate()

    assert v.get_dimension_scores() == {}
    assert v.get_health_score() == 100.0


def test_base_dimension_ignores_global_config(simple_df):
    pb.config(dimension_map={"col_vals_gt": "consistency"})

    # The base inference ignores the global override...
    assert _base_dimension_from_assertion_type("col_vals_gt") == "validity"

    # ...while the effective inference respects it
    assert _infer_dimension_from_assertion_type("col_vals_gt") == "consistency"


def test_to_code_emits_dimension_only_when_overridden(simple_df):
    v = (
        Validate(data=simple_df)
        .col_vals_gt(columns="b", value=0)  # inferred validity -> no dimension=
        .col_vals_gt(columns="b", value=0, dimension="consistency")  # override -> dimension=
        .col_vals_not_null(columns="a")  # inferred completeness -> no dimension=
    )
    code = v.to_code()

    assert code.count("dimension=") == 1
    assert 'dimension="consistency"' in code


def test_to_code_round_trip_preserves_dimension(simple_df):
    v = (
        Validate(data=simple_df)
        .col_vals_gt(columns="b", value=0, dimension="consistency")
        .col_vals_not_null(columns="a")
    )
    code = v.to_code()
    ns = {"pb": pb, "your_data": simple_df}
    exec(code.replace("validation\n", ""), ns)
    rebuilt = ns["validation"].interrogate()
    dims = [s.dimension for s in rebuilt.validation_info]

    assert dims == ["consistency", "completeness"]


def test_to_yaml_preserves_dimension_override(simple_df):
    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0, dimension="consistency")
    yaml_text = v.to_yaml()

    assert "dimension: consistency" in yaml_text


def test_assert_dimension_scores_auto_interrogates(simple_df):
    # Not interrogated yet; the assertion should interrogate and then evaluate
    v = Validate(data=simple_df).col_vals_gt(columns="b", value=0)  # validity 3/5 = 60
    with pytest.raises(AssertionError, match="validity"):
        v.assert_dimension_scores(thresholds={"validity": 90})


def test_custom_dimension_name_is_html_escaped():
    pl = pytest.importorskip("polars")
    df = pl.DataFrame({"a": [1, 2, 3]})
    v = (
        Validate(df, tbl_name="t")
        .col_vals_not_null(columns="a", dimension='weird"<x>')
        .interrogate()
    )
    html = v.get_tabular_report().as_raw_html()

    # The custom name is title-cased for display; the raw (unescaped) form must not leak into the
    # HTML, and the escaped entities must be present instead
    assert 'Weird"<X>' not in html
    assert "Weird&quot;&lt;X&gt;" in html


def test_health_score_negative_weight_clamped(simple_df):
    v = (
        Validate(data=simple_df)
        .col_vals_not_null(columns="a")
        .col_vals_gt(columns="b", value=0)
        .interrogate()
    )
    # A negative weight must not push the score outside [0, 100]
    score = _compute_health_score(v.validation_info, dimension_weights={"validity": -1.0})

    assert 0.0 <= score <= 100.0


def test_health_score_all_zero_weights(simple_df):
    v = (
        Validate(data=simple_df)
        .col_vals_not_null(columns="a")
        .col_vals_gt(columns="b", value=0)
        .interrogate()
    )

    # All-zero weights -> no denominator -> defaults to 100.0 (same as the no-data case)
    assert (
        _compute_health_score(
            v.validation_info, dimension_weights={"completeness": 0.0, "validity": 0.0}
        )
        == 100.0
    )


def test_scorecard_message_distinguishes_interrogation_state(simple_df):
    # Interrogated but nothing scorable (all steps inactive) -> "no validation steps" message
    v_inter = (
        Validate(data=simple_df, tbl_name="t")
        .col_vals_gt(columns="b", value=0, active=False)
        .interrogate()
    )
    html_inter = v_inter.get_scorecard().as_raw_html()

    assert "No Interrogation Performed" not in html_inter
    assert "NO VALIDATION STEPS" in html_inter

    # Not interrogated -> "no interrogation performed" message
    v_noninter = Validate(data=simple_df, tbl_name="t").col_vals_gt(columns="b", value=0)

    assert "No Interrogation Performed" in v_noninter.get_scorecard().as_raw_html()
