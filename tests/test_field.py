import pytest

from pointblank.field import (
    Field,
    IntField,
    FloatField,
    StringField,
    BoolField,
    DateField,
    DatetimeField,
    TimeField,
    DurationField,
    int_field,
    float_field,
    string_field,
    bool_field,
    date_field,
    datetime_field,
    time_field,
    duration_field,
    AVAILABLE_PRESETS,
)


class TestIntField:
    """Tests for IntField and int_field()."""

    def test_int_field_defaults(self):
        """Test IntField with default values."""
        field = int_field()
        assert field.dtype == "Int64"
        assert field.min_val is None
        assert field.max_val is None
        assert field.nullable is False
        assert field.unique is False

    def test_int_field_with_constraints(self):
        """Test IntField with min/max constraints."""
        field = int_field(min_val=0, max_val=100)
        assert field.min_val == 0
        assert field.max_val == 100

    def test_int_field_with_allowed_values(self):
        """Test IntField with allowed values."""
        field = int_field(allowed=[1, 2, 3, 4, 5])
        assert field.allowed == [1, 2, 3, 4, 5]
        assert field.has_allowed_values() is True

    def test_int_field_with_custom_dtype(self):
        """Test IntField with different integer dtypes."""
        field = int_field(dtype="UInt8")
        assert field.dtype == "UInt8"

        field = int_field(dtype="Int32")
        assert field.dtype == "Int32"

    def test_int_field_invalid_dtype_raises_error(self):
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            int_field(dtype="Float64")

    def test_int_field_min_greater_than_max_raises_error(self):
        """Test that min_val > max_val raises ValueError."""
        with pytest.raises(ValueError, match="min_val .* cannot be greater than max_val"):
            int_field(min_val=100, max_val=50)

    def test_int_field_empty_allowed_raises_error(self):
        """Test that empty allowed list raises ValueError."""
        with pytest.raises(ValueError, match="allowed list cannot be empty"):
            int_field(allowed=[])

    def test_int_field_nullable(self):
        """Test IntField with nullable settings."""
        field = int_field(nullable=True, null_probability=0.2)
        assert field.nullable is True
        assert field.null_probability == 0.2

    def test_int_field_helper_returns_int_field(self):
        """Test that int_field() returns IntField instance."""
        field = int_field()
        assert isinstance(field, IntField)

    def test_int_field_is_numeric(self):
        """Test helper methods."""
        field = int_field()
        assert field.is_numeric() is True
        assert field.is_integer() is True
        assert field.is_float() is False


class TestFloatField:
    """Tests for FloatField and float_field()."""

    def test_float_field_defaults(self):
        """Test FloatField with default values."""
        field = float_field()
        assert field.dtype == "Float64"
        assert field.min_val is None
        assert field.max_val is None

    def test_float_field_with_constraints(self):
        """Test FloatField with min/max constraints."""
        field = float_field(min_val=0.0, max_val=1.0)
        assert field.min_val == 0.0
        assert field.max_val == 1.0

    def test_float_field_with_custom_dtype(self):
        """Test FloatField with Float32 dtype."""
        field = float_field(dtype="Float32")
        assert field.dtype == "Float32"

    def test_float_field_invalid_dtype_raises_error(self):
        """Test that invalid dtype raises ValueError."""
        with pytest.raises(ValueError, match="Invalid dtype"):
            float_field(dtype="Int64")

    def test_float_field_is_numeric(self):
        """Test helper methods."""
        field = float_field()
        assert field.is_numeric() is True
        assert field.is_float() is True
        assert field.is_integer() is False


class TestStringField:
    """Tests for StringField and string_field()."""

    def test_string_field_defaults(self):
        """Test StringField with default values."""
        field = string_field()
        assert field.dtype == "String"
        assert field.min_length is None
        assert field.max_length is None
        assert field.pattern is None
        assert field.preset is None

    def test_string_field_with_length_constraints(self):
        """Test StringField with length constraints."""
        field = string_field(min_length=5, max_length=20)
        assert field.min_length == 5
        assert field.max_length == 20

    def test_string_field_with_pattern(self):
        """Test StringField with regex pattern."""
        field = string_field(pattern=r"^[A-Z]{3}-\d{4}$")
        assert field.pattern == r"^[A-Z]{3}-\d{4}$"
        assert field.has_pattern() is True

    def test_string_field_with_preset(self):
        """Test StringField with preset."""
        field = string_field(preset="email")
        assert field.preset == "email"
        assert field.has_preset() is True

    def test_string_field_with_allowed_values(self):
        """Test StringField with allowed values."""
        field = string_field(allowed=["active", "pending", "inactive"])
        assert field.allowed == ["active", "pending", "inactive"]
        assert field.has_allowed_values() is True

    def test_string_field_invalid_preset_raises_error(self):
        """Test that invalid preset raises ValueError."""
        with pytest.raises(ValueError, match="Unknown preset"):
            string_field(preset="invalid_preset")

    def test_string_field_incompatible_options_raises_error(self):
        """Test that incompatible options raise ValueError."""
        with pytest.raises(ValueError, match="Only one of preset, pattern, or allowed"):
            string_field(preset="email", pattern=r"^\d+$")

        with pytest.raises(ValueError, match="Only one of preset, pattern, or allowed"):
            string_field(preset="name", allowed=["a", "b"])

        with pytest.raises(ValueError, match="Only one of preset, pattern, or allowed"):
            string_field(pattern=r"^\d+$", allowed=["1", "2"])

    def test_string_field_negative_length_raises_error(self):
        """Test that negative length raises ValueError."""
        with pytest.raises(ValueError, match="min_length must be non-negative"):
            string_field(min_length=-1)

        with pytest.raises(ValueError, match="max_length must be non-negative"):
            string_field(max_length=-1)

    def test_string_field_min_greater_than_max_length_raises_error(self):
        """Test that min_length > max_length raises ValueError."""
        with pytest.raises(ValueError, match="min_length .* cannot be greater than max_length"):
            string_field(min_length=20, max_length=5)


class TestBoolField:
    """Tests for BoolField and bool_field()."""

    def test_bool_field_defaults(self):
        """Test BoolField with default values."""
        field = bool_field()
        assert field.dtype == "Boolean"
        assert field.nullable is False

    def test_bool_field_nullable(self):
        """Test BoolField with nullable settings."""
        field = bool_field(nullable=True, null_probability=0.1)
        assert field.nullable is True
        assert field.null_probability == 0.1

    def test_bool_field_is_boolean(self):
        """Test helper methods."""
        field = bool_field()
        assert field.is_boolean() is True
        assert field.is_numeric() is False


class TestDateField:
    """Tests for DateField and date_field()."""

    def test_date_field_defaults(self):
        """Test DateField with default values."""
        field = date_field()
        assert field.dtype == "Date"
        assert field.min_date is None
        assert field.max_date is None

    def test_date_field_with_constraints(self):
        """Test DateField with min/max constraints."""
        field = date_field(min_date="2020-01-01", max_date="2025-12-31")
        assert field.min_date == "2020-01-01"
        assert field.max_date == "2025-12-31"

    def test_date_field_min_greater_than_max_raises_error(self):
        """Test that min_date > max_date raises ValueError."""
        with pytest.raises(ValueError, match="min_date .* cannot be greater than max_date"):
            date_field(min_date="2025-01-01", max_date="2020-01-01")

    def test_date_field_is_temporal(self):
        """Test helper methods."""
        field = date_field()
        assert field.is_temporal() is True
        assert field.is_numeric() is False


class TestDatetimeField:
    """Tests for DatetimeField and datetime_field()."""

    def test_datetime_field_defaults(self):
        """Test DatetimeField with default values."""
        field = datetime_field()
        assert field.dtype == "Datetime"
        assert field.min_date is None
        assert field.max_date is None

    def test_datetime_field_with_constraints(self):
        """Test DatetimeField with min/max constraints."""
        field = datetime_field(min_date="2024-01-01T00:00:00", max_date="2024-12-31T23:59:59")
        assert field.min_date == "2024-01-01T00:00:00"
        assert field.max_date == "2024-12-31T23:59:59"


class TestTimeField:
    """Tests for TimeField and time_field()."""

    def test_time_field_defaults(self):
        """Test TimeField with default values."""
        field = time_field()
        assert field.dtype == "Time"
        assert field.min_time is None
        assert field.max_time is None

    def test_time_field_with_constraints(self):
        """Test TimeField with min/max constraints."""
        field = time_field(min_time="09:00:00", max_time="17:00:00")
        assert field.min_time == "09:00:00"
        assert field.max_time == "17:00:00"

    def test_time_field_min_greater_than_max_raises_error(self):
        """Test that min_time > max_time raises ValueError."""
        with pytest.raises(ValueError, match="min_time .* cannot be greater than max_time"):
            time_field(min_time="17:00:00", max_time="09:00:00")


class TestDurationField:
    """Tests for DurationField and duration_field()."""

    def test_duration_field_defaults(self):
        """Test DurationField with default values."""
        field = duration_field()
        assert field.dtype == "Duration"
        assert field.min_duration is None
        assert field.max_duration is None

    def test_duration_field_with_timedelta(self):
        """Test DurationField with timedelta constraints."""
        from datetime import timedelta

        field = duration_field(min_duration=timedelta(minutes=1), max_duration=timedelta(hours=2))
        assert field.min_duration == timedelta(minutes=1)
        assert field.max_duration == timedelta(hours=2)


class TestBaseFieldValidation:
    """Tests for base Field validation."""

    def test_null_probability_out_of_range_raises_error(self):
        """Test that null_probability outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="null_probability must be between"):
            int_field(nullable=True, null_probability=1.5)

        with pytest.raises(ValueError, match="null_probability must be between"):
            string_field(nullable=True, null_probability=-0.1)

    def test_null_probability_without_nullable_raises_error(self):
        """Test that null_probability > 0 without nullable raises ValueError."""
        with pytest.raises(ValueError, match="null_probability > 0 requires nullable=True"):
            int_field(null_probability=0.5)


class TestFieldHelperMethods:
    """Tests for Field helper methods across all types."""

    def test_is_numeric(self):
        """Test is_numeric() method."""
        assert int_field().is_numeric() is True
        assert float_field().is_numeric() is True
        assert string_field().is_numeric() is False
        assert bool_field().is_numeric() is False

    def test_is_integer(self):
        """Test is_integer() method."""
        assert int_field().is_integer() is True
        assert int_field(dtype="UInt32").is_integer() is True
        assert float_field().is_integer() is False
        assert string_field().is_integer() is False

    def test_is_float(self):
        """Test is_float() method."""
        assert float_field().is_float() is True
        assert float_field(dtype="Float32").is_float() is True
        assert int_field().is_float() is False

    def test_is_string(self):
        """Test is_string() method."""
        assert string_field().is_string() is True
        assert int_field().is_string() is False

    def test_is_boolean(self):
        """Test is_boolean() method."""
        assert bool_field().is_boolean() is True
        assert int_field().is_boolean() is False

    def test_is_temporal(self):
        """Test is_temporal() method."""
        assert date_field().is_temporal() is True
        assert datetime_field().is_temporal() is True
        assert duration_field().is_temporal() is True
        assert time_field().is_temporal() is True
        assert int_field().is_temporal() is False

    def test_has_custom_generator(self):
        """Test has_custom_generator() method."""
        gen = lambda: "value"
        assert string_field(generator=gen).has_custom_generator() is True
        assert string_field().has_custom_generator() is False


class TestAvailablePresets:
    """Tests for the AVAILABLE_PRESETS constant."""

    def test_common_presets_are_available(self):
        """Test that common presets are in the available set."""
        common_presets = [
            "name",
            "email",
            "address",
            "phone_number",
            "company",
            "url",
            "text",
            "uuid4",
        ]
        for preset in common_presets:
            assert preset in AVAILABLE_PRESETS

    def test_all_available_presets_can_create_field(self):
        """Test that all available presets can create a valid StringField."""
        for preset in AVAILABLE_PRESETS:
            field = string_field(preset=preset)
            assert field.preset == preset


class TestFieldTypeInheritance:
    """Tests to verify field class inheritance."""

    def test_all_field_types_inherit_from_field(self):
        """Test that all field types inherit from Field."""
        assert isinstance(int_field(), Field)
        assert isinstance(float_field(), Field)
        assert isinstance(string_field(), Field)
        assert isinstance(bool_field(), Field)
        assert isinstance(date_field(), Field)
        assert isinstance(datetime_field(), Field)
        assert isinstance(time_field(), Field)
        assert isinstance(duration_field(), Field)

    def test_field_types_are_correct_class(self):
        """Test that field functions return the correct class type."""
        assert isinstance(int_field(), IntField)
        assert isinstance(float_field(), FloatField)
        assert isinstance(string_field(), StringField)
        assert isinstance(bool_field(), BoolField)
        assert isinstance(date_field(), DateField)
        assert isinstance(datetime_field(), DatetimeField)
        assert isinstance(time_field(), TimeField)
        assert isinstance(duration_field(), DurationField)
