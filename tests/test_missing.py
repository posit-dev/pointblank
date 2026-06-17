import pytest

import pointblank as pb
from pointblank.missing import MissingSpec


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
