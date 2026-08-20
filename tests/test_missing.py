import pytest

import pointblank as pb
from pointblank.missing import MissingSpec, _slugify


class TestMissingSpecConstruction:
    """Tests for MissingSpec construction and validation."""

    def test_minimal_spec(self):
        spec = MissingSpec(reasons={-99: "not_asked"})
        assert spec.reasons == {-99: "not_asked"}
        assert spec.categories is None
        assert spec.null_is_missing is True
        assert spec.null_reason == "unknown"
        assert spec.description is None

    def test_full_spec(self):
        spec = MissingSpec(
            reasons={-99: "not_asked", -98: "refused", -97: "dont_know"},
            categories={"item_nonresponse": ["refused", "dont_know"], "design": ["not_asked"]},
            null_is_missing=False,
            null_reason="system",
            description="Standard survey codes",
        )
        assert spec.null_is_missing is False
        assert spec.null_reason == "system"
        assert spec.description == "Standard survey codes"

    def test_exported_from_top_level(self):
        assert pb.MissingSpec is MissingSpec

    def test_reasons_must_be_dict(self):
        with pytest.raises(TypeError):
            MissingSpec(reasons=[-99, -98])  # type: ignore[arg-type]

    def test_empty_reasons_requires_null_is_missing(self):
        # OK: empty reasons but null_is_missing=True
        MissingSpec(reasons={}, null_is_missing=True)
        # Not OK: empty reasons and null_is_missing=False
        with pytest.raises(ValueError):
            MissingSpec(reasons={}, null_is_missing=False)

    def test_reason_labels_must_be_strings(self):
        with pytest.raises(TypeError):
            MissingSpec(reasons={-99: 1})  # type: ignore[dict-item]

    def test_category_must_reference_known_reasons(self):
        with pytest.raises(ValueError, match="unknown reason"):
            MissingSpec(
                reasons={-99: "not_asked"},
                categories={"bad": ["nonexistent"]},
            )

    def test_category_can_reference_null_reason(self):
        spec = MissingSpec(
            reasons={-99: "not_asked"},
            categories={"all_absent": ["not_asked", "unknown"]},
            null_is_missing=True,
        )
        assert spec.values_for_category("all_absent") == [-99]

    def test_categories_must_be_dict(self):
        with pytest.raises(TypeError):
            MissingSpec(reasons={-99: "not_asked"}, categories=["not_asked"])  # type: ignore[arg-type]


class TestMissingSpecMethods:
    @pytest.fixture
    def spec(self):
        return MissingSpec(
            reasons={-99: "not_asked", -98: "refused", -97: "dont_know", -96: "not_applicable"},
            categories={
                "item_nonresponse": ["refused", "dont_know"],
                "design": ["not_asked", "not_applicable"],
            },
        )

    def test_sentinel_values(self, spec):
        assert spec.sentinel_values() == [-99, -98, -97, -96]

    def test_reason_for(self, spec):
        assert spec.reason_for(-98) == "refused"
        assert spec.reason_for(5) is None

    def test_reason_for_null(self, spec):
        assert spec.reason_for(None) == "unknown"
        spec_no_null = MissingSpec(reasons={-99: "not_asked"}, null_is_missing=False)
        assert spec_no_null.reason_for(None) is None

    def test_is_missing(self, spec):
        assert spec.is_missing(-99) is True
        assert spec.is_missing(42) is False
        assert spec.is_missing(None) is True

    def test_is_missing_null_excluded(self):
        spec = MissingSpec(reasons={-99: "not_asked"}, null_is_missing=False)
        assert spec.is_missing(None) is False

    def test_values_for_reason(self, spec):
        assert spec.values_for_reason("refused") == [-98]
        assert spec.values_for_reason("nonexistent") == []

    def test_values_for_category(self, spec):
        assert spec.values_for_category("item_nonresponse") == [-98, -97]
        assert spec.values_for_category("design") == [-99, -96]
        assert spec.values_for_category("nonexistent") == []

    def test_values_for_category_no_categories(self):
        spec = MissingSpec(reasons={-99: "not_asked"})
        assert spec.values_for_category("anything") == []

    def test_reasons_list(self, spec):
        assert spec.reasons_list() == [
            "not_asked",
            "refused",
            "dont_know",
            "not_applicable",
            "unknown",
        ]

    def test_reasons_list_no_null(self):
        spec = MissingSpec(reasons={-99: "a", -98: "b"}, null_is_missing=False)
        assert spec.reasons_list() == ["a", "b"]


class TestMissingSpecValidationEdgeCases:
    def test_null_reason_must_be_string(self):
        with pytest.raises(TypeError, match="null_reason must be a string"):
            MissingSpec(reasons={-99: "not_asked"}, null_reason=99)  # type: ignore[arg-type]

    def test_category_value_must_be_list(self):
        with pytest.raises(TypeError, match="must map to a list"):
            MissingSpec(
                reasons={-99: "not_asked"},
                categories={"bad": "not_a_list"},  # type: ignore[dict-item]
            )


class TestMissingSpecFactoryMethods:
    def test_from_cdisc_null_flavors_defaults(self):
        spec = MissingSpec.from_cdisc_null_flavors()
        assert spec.null_is_missing is True
        assert spec.null_reason == "no_information"
        assert spec.description == "CDISC/HL7 null flavors"
        assert spec.reason_for("NASK") == "not_asked"
        assert spec.reason_for("UNK") == "unknown"
        assert "NI" in spec.reasons

    def test_from_cdisc_null_flavors_custom(self):
        spec = MissingSpec.from_cdisc_null_flavors(null_is_missing=False, null_reason="ni")
        assert spec.null_is_missing is False
        assert spec.null_reason == "ni"

    def test_from_cdisc_alias(self):
        spec = MissingSpec.from_cdisc()
        assert spec.reason_for("NASK") == "not_asked"

    def test_from_sas_defaults(self):
        spec = MissingSpec.from_sas()
        assert spec.reason_for(".") == "system_missing"
        assert spec.reason_for("._") == "system_missing"
        assert spec.reason_for(".A") == "user_missing_a"
        assert spec.reason_for(".Z") == "user_missing_z"
        assert spec.null_reason == "system_missing"

    def test_from_sas_exclude_underscore(self):
        spec = MissingSpec.from_sas(include_underscore=False)
        assert spec.reason_for("._") is None

    def test_from_sas_custom_reasons(self):
        spec = MissingSpec.from_sas(reasons={".A": "not_applicable"})
        assert spec.reason_for(".A") == "not_applicable"
        assert spec.reason_for(".B") == "user_missing_b"

    def test_from_spss_with_labels(self):
        spec = MissingSpec.from_spss(
            missing_values=[-99, -98],
            labels={-99: "Not asked", -98: "Refused"},
        )
        assert spec.reason_for(-99) == "not_asked"
        assert spec.reason_for(-98) == "refused"

    def test_from_spss_without_labels(self):
        spec = MissingSpec.from_spss(missing_values=[-99, -98])
        assert spec.reason_for(-99) == "missing_99"
        assert spec.reason_for(-98) == "missing_98"

    def test_from_spss_empty(self):
        spec = MissingSpec.from_spss(missing_values=[])
        assert spec.sentinel_values() == []

    def test_from_variable_metadata_with_missing_values(self):
        class FakeVar:
            name = "age"
            missing_values = [-99, -98]
            missing_value_labels = {-99: "Not asked"}
            value_labels = {-98: "Refused"}

        spec = MissingSpec.from_variable_metadata(FakeVar())
        assert spec is not None
        assert spec.reason_for(-99) == "not_asked"
        assert spec.reason_for(-98) == "refused"
        assert "age" in spec.description

    def test_from_variable_metadata_no_missing_values(self):
        class FakeVar:
            missing_values = []
            missing_value_labels = {}
            value_labels = {}

        result = MissingSpec.from_variable_metadata(FakeVar())
        assert result is None

    def test_from_variable_metadata_fallback_label(self):
        class FakeVar:
            name = "q1"
            missing_values = [-99]
            missing_value_labels = None
            value_labels = None

        spec = MissingSpec.from_variable_metadata(FakeVar())
        assert spec is not None
        assert spec.reason_for(-99) == "missing_99"


class TestSlugify:
    def test_normal_string(self):
        assert _slugify("Not Asked") == "not_asked"

    def test_already_snake_case(self):
        assert _slugify("refused") == "refused"

    def test_special_chars(self):
        assert _slugify("Don't Know!") == "don_t_know"

    def test_numeric_label(self):
        assert _slugify(-99) == "99"

    def test_empty_string(self):
        assert _slugify("") == "missing"

    def test_leading_trailing_separators(self):
        assert _slugify("  hello world  ") == "hello_world"


class TestMissingSpecSentinelValues:
    def test_returns_keys(self):
        spec = MissingSpec(reasons={-99: "not_asked", -98: "refused"})
        assert spec.sentinel_values() == [-99, -98]

    def test_empty_reasons_null_is_missing_true(self):
        spec = MissingSpec(reasons={})
        assert spec.sentinel_values() == []

    def test_string_sentinel_values(self):
        spec = MissingSpec(reasons={"NA": "not_applicable", "UNK": "unknown"})
        assert spec.sentinel_values() == ["NA", "UNK"]


class TestMissingSpecReasonFor:
    def test_known_value(self):
        spec = MissingSpec(reasons={-99: "not_asked", -98: "refused"})
        assert spec.reason_for(-99) == "not_asked"
        assert spec.reason_for(-98) == "refused"

    def test_none_value_null_is_missing_true(self):
        spec = MissingSpec(reasons={-99: "not_asked"}, null_reason="unknown")
        assert spec.reason_for(None) == "unknown"

    def test_none_value_null_is_missing_false(self):
        spec = MissingSpec(reasons={-99: "not_asked"}, null_is_missing=False)
        assert spec.reason_for(None) is None

    def test_unknown_value_returns_none(self):
        spec = MissingSpec(reasons={-99: "not_asked"})
        assert spec.reason_for(42) is None


class TestMissingSpecIsMissing:
    def test_sentinel_value_is_missing(self):
        spec = MissingSpec(reasons={-99: "not_asked"})
        assert spec.is_missing(-99) is True

    def test_non_sentinel_value_not_missing(self):
        spec = MissingSpec(reasons={-99: "not_asked"})
        assert spec.is_missing(0) is False

    def test_none_null_is_missing_true(self):
        spec = MissingSpec(reasons={-99: "not_asked"}, null_is_missing=True)
        assert spec.is_missing(None) is True

    def test_none_null_is_missing_false(self):
        spec = MissingSpec(reasons={-99: "not_asked"}, null_is_missing=False)
        assert spec.is_missing(None) is False


class TestMissingSpecValuesForReason:
    def test_single_match(self):
        spec = MissingSpec(reasons={-99: "not_asked", -98: "refused"})
        assert spec.values_for_reason("refused") == [-98]

    def test_multiple_matches(self):
        spec = MissingSpec(reasons={-99: "not_asked", -98: "not_asked", -97: "refused"})
        assert spec.values_for_reason("not_asked") == [-99, -98]

    def test_no_match(self):
        spec = MissingSpec(reasons={-99: "not_asked"})
        assert spec.values_for_reason("refused") == []


class TestMissingSpecValuesForCategory:
    def test_returns_values_in_category(self):
        spec = MissingSpec(
            reasons={-99: "not_asked", -98: "refused", -97: "dont_know"},
            categories={"item_nonresponse": ["refused", "dont_know"]},
        )
        assert set(spec.values_for_category("item_nonresponse")) == {-98, -97}

    def test_unknown_category_returns_empty(self):
        spec = MissingSpec(
            reasons={-99: "not_asked"},
            categories={"design": ["not_asked"]},
        )
        assert spec.values_for_category("no_such_category") == []

    def test_no_categories_returns_empty(self):
        spec = MissingSpec(reasons={-99: "not_asked"})
        assert spec.values_for_category("design") == []


class TestMissingSpecReasonsList:
    def test_distinct_reasons_include_null_reason(self):
        spec = MissingSpec(
            reasons={-99: "not_asked", -98: "refused"},
            null_is_missing=True,
            null_reason="unknown",
        )
        reasons = spec.reasons_list()
        assert "not_asked" in reasons
        assert "refused" in reasons
        assert "unknown" in reasons

    def test_null_reason_excluded_when_null_not_missing(self):
        spec = MissingSpec(
            reasons={-99: "not_asked"},
            null_is_missing=False,
        )
        reasons = spec.reasons_list()
        assert "unknown" not in reasons

    def test_duplicate_reason_labels_appear_once(self):
        spec = MissingSpec(
            reasons={-99: "not_asked", -98: "not_asked"},
            null_is_missing=False,
        )
        assert spec.reasons_list().count("not_asked") == 1


class TestMissingSpecValidationEmptyReasons:
    def test_empty_reasons_with_null_false_raises(self):
        with pytest.raises(ValueError, match="at least one sentinel value"):
            MissingSpec(reasons={}, null_is_missing=False)
