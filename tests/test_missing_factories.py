import pytest

import pointblank as pb
from pointblank.missing import MissingSpec, _slugify
from pointblank.metadata import VariableMetadata, MetadataImport


class TestSlugify:
    @pytest.mark.parametrize(
        "label,expected",
        [
            ("Refused", "refused"),
            ("Not Applicable", "not_applicable"),
            ("DON'T KNOW", "don_t_know"),
            ("  spaced  ", "spaced"),
            (-99, "99"),
            ("", "missing"),
        ],
    )
    def test_slugify(self, label, expected):
        assert _slugify(label) == expected


class TestFromCdisc:
    def test_standard_codes(self):
        spec = MissingSpec.from_cdisc_null_flavors()
        assert spec.reason_for("NASK") == "not_asked"
        assert spec.reason_for("UNK") == "unknown"
        assert spec.reason_for("PINF") == "positive_infinity"
        assert spec.reason_for("NA") == "not_applicable"

    def test_categories(self):
        spec = MissingSpec.from_cdisc_null_flavors()
        assert set(spec.values_for_category("boundary")) == {"PINF", "NINF"}
        assert "NASK" in spec.values_for_category("not_applicable")

    def test_alias(self):
        assert MissingSpec.from_cdisc().reason_for("MSK") == "masked"

    def test_null_handling(self):
        spec = MissingSpec.from_cdisc_null_flavors()
        assert spec.null_is_missing is True
        assert spec.reason_for(None) == "no_information"

    def test_exported_via_top_level(self):
        assert pb.MissingSpec.from_cdisc_null_flavors().reason_for("NI") == "no_information"


class TestFromSas:
    def test_defaults(self):
        spec = MissingSpec.from_sas()
        assert spec.reason_for(".") == "system_missing"
        assert spec.reason_for(".A") == "user_missing_a"
        assert spec.reason_for(".Z") == "user_missing_z"
        assert spec.reason_for("._") == "system_missing"

    def test_overrides(self):
        spec = MissingSpec.from_sas(reasons={".A": "not_applicable", ".B": "below_detection"})
        assert spec.reason_for(".A") == "not_applicable"
        assert spec.reason_for(".B") == "below_detection"
        assert spec.reason_for(".C") == "user_missing_c"  # default preserved

    def test_no_underscore(self):
        spec = MissingSpec.from_sas(include_underscore=False)
        assert spec.reason_for("._") is None
        # 26 letters + "." = 27 sentinels
        assert len(spec.sentinel_values()) == 27


class TestFromSpss:
    def test_with_labels(self):
        spec = MissingSpec.from_spss(
            missing_values=[-99, -98], labels={-99: "Not asked", -98: "Refused"}
        )
        assert spec.reason_for(-99) == "not_asked"
        assert spec.reason_for(-98) == "refused"

    def test_without_labels(self):
        spec = MissingSpec.from_spss(missing_values=[-99, -1])
        assert spec.reason_for(-99) == "missing_99"
        assert spec.reason_for(-1) == "missing_1"


class TestFromVariableMetadata:
    def test_uses_missing_value_labels(self):
        var = VariableMetadata(
            name="age",
            dtype="Int64",
            missing_values=[-99, -98],
            missing_value_labels={-99: "Not asked", -98: "Refused"},
        )
        spec = MissingSpec.from_variable_metadata(var)
        assert spec.reason_for(-98) == "refused"

    def test_falls_back_to_value_labels(self):
        var = VariableMetadata(
            name="age",
            dtype="Int64",
            missing_values=[-99],
            value_labels={-99: "Not Asked", 1: "Yes"},
        )
        spec = MissingSpec.from_variable_metadata(var)
        assert spec.reason_for(-99) == "not_asked"

    def test_no_missing_returns_none(self):
        var = VariableMetadata(name="id", dtype="Int64")
        assert MissingSpec.from_variable_metadata(var) is None

    def test_to_missing_spec_method(self):
        var = VariableMetadata(name="age", dtype="Int64", missing_values=[-99])
        assert var.to_missing_spec().is_missing(-99) is True


class TestMetadataImportMissingSpecs:
    def test_missing_specs_mapping(self):
        v1 = VariableMetadata(
            name="age",
            dtype="Int64",
            missing_values=[-99, -98],
            missing_value_labels={-99: "Not asked", -98: "Refused"},
        )
        v2 = VariableMetadata(name="id", dtype="Int64")  # no missing values
        meta = MetadataImport(source_format="spss", variables=[v1, v2])

        specs = meta.missing_specs()
        assert list(specs.keys()) == ["age"]  # id omitted (no missing values)
        assert specs["age"].reason_for(-99) == "not_asked"

    def test_specs_usable_in_validation(self):
        import polars as pl

        v = VariableMetadata(
            name="age", dtype="Int64", missing_values=[-99], missing_value_labels={-99: "Not asked"}
        )
        meta = MetadataImport(source_format="spss", variables=[v])
        specs = meta.missing_specs()

        tbl = pl.DataFrame({"age": [34, -99, 200]})
        validation = (
            pb.Validate(data=tbl)
            .col_vals_between(columns="age", left=0, right=120, missing=specs["age"])
            .interrogate()
        )
        # -99 excluded; only 200 fails
        assert validation.validation_info[0].n_failed == 1
