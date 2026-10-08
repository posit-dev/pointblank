import pytest

import polars as pl
import pandas as pd

from pointblank.schema import Schema
from pointblank.field import (
    BoolField,
    DateField,
    DatetimeField,
    DurationField,
    FloatField,
    IntField,
    StringField,
)
from pointblank.generate.inference import (
    infer_fields_from_table,
    _detect_preset,
    _classify_dtype,
    _validate_preset_values,
    _validate_numeric_as_latitude,
    _validate_numeric_as_longitude,
    _detect_numeric_role,
)
from pointblank.schema import schema_from_tbl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_field(schema: Schema, col_name: str):
    """Get the Field object for a column from an inferred schema."""
    for name, field_obj in schema.columns:
        if name == col_name:
            return field_obj
    raise KeyError(f"Column '{col_name}' not found in schema")


# ---------------------------------------------------------------------------
# Basic smoke test
# ---------------------------------------------------------------------------


class TestFromDataframeSmoke:
    def test_polars_basic(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        schema = Schema.from_table(df)
        assert schema.columns is not None
        assert len(schema.columns) == 2

    def test_pandas_basic(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        schema = Schema.from_table(df)
        assert schema.columns is not None
        assert len(schema.columns) == 2

    def test_infer_constraints_false(self):
        """When infer_constraints=False, behave like Schema(tbl=df)."""
        df = pl.DataFrame({"a": [1, 2, 3]})
        schema = Schema.from_table(df, infer_constraints=False)
        # Should have dtype tuples, not Field objects
        assert schema.columns is not None
        name, dtype = schema.columns[0]
        assert name == "a"
        assert isinstance(dtype, str)


# ---------------------------------------------------------------------------
# Integer inference
# ---------------------------------------------------------------------------


class TestIntInference:
    def test_min_max(self):
        df = pl.DataFrame({"val": [10, 20, 30, 40, 50]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "val")
        assert isinstance(field, IntField)
        assert field.min_val == 10
        assert field.max_val == 50

    def test_unique_detection(self):
        df = pl.DataFrame({"id": list(range(25))})
        schema = Schema.from_table(df)
        field = _get_field(schema, "id")
        assert isinstance(field, IntField)
        assert field.unique is True

    def test_categorical_detection(self):
        df = pl.DataFrame({"rating": [1, 2, 3, 2, 1, 3, 2, 1]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "rating")
        assert isinstance(field, IntField)
        assert field.allowed == [1, 2, 3]
        assert field.min_val is None  # not set when allowed is used
        assert field.max_val is None

    def test_categorical_threshold_respected(self):
        """Values above categorical threshold should use min/max instead."""
        df = pl.DataFrame({"val": list(range(25))})
        schema = Schema.from_table(df, categorical_threshold=10)
        field = _get_field(schema, "val")
        assert isinstance(field, IntField)
        assert field.allowed is None
        assert field.min_val == 0
        assert field.max_val == 24

    def test_nullable(self):
        df = pl.DataFrame({"val": [1, 2, None, 4, None]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "val")
        assert isinstance(field, IntField)
        assert field.nullable is True
        assert field.null_probability == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# Float inference
# ---------------------------------------------------------------------------


class TestFloatInference:
    def test_min_max(self):
        df = pl.DataFrame({"val": [1.5, 2.5, 3.5]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "val")
        assert isinstance(field, FloatField)
        assert field.min_val == pytest.approx(1.5)
        assert field.max_val == pytest.approx(3.5)

    def test_nullable(self):
        df = pl.DataFrame({"val": [1.0, None, 3.0, None, 5.0]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "val")
        assert isinstance(field, FloatField)
        assert field.nullable is True
        assert field.null_probability == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# String inference
# ---------------------------------------------------------------------------


class TestStringInference:
    def test_categorical(self):
        df = pl.DataFrame({"status": ["active", "pending", "active", "inactive"]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "status")
        assert isinstance(field, StringField)
        assert field.allowed == ["active", "inactive", "pending"]

    def test_length_constraints(self):
        """Non-categorical strings get min/max length."""
        values = [f"item_{i:04d}" for i in range(30)]
        df = pl.DataFrame({"code": values})
        schema = Schema.from_table(df)
        field = _get_field(schema, "code")
        assert isinstance(field, StringField)
        assert field.min_length == 9
        assert field.max_length == 9
        assert field.allowed is None

    def test_unique_detection(self):
        values = [f"unique_{i}" for i in range(25)]
        df = pl.DataFrame({"key": values})
        schema = Schema.from_table(df)
        field = _get_field(schema, "key")
        assert isinstance(field, StringField)
        assert field.unique is True


# ---------------------------------------------------------------------------
# Boolean inference
# ---------------------------------------------------------------------------


class TestBoolInference:
    def test_p_true(self):
        df = pl.DataFrame({"flag": [True, True, True, False]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "flag")
        assert isinstance(field, BoolField)
        assert field.p_true == pytest.approx(0.75, abs=0.01)

    def test_nullable(self):
        df = pl.DataFrame({"flag": [True, None, False, None]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "flag")
        assert isinstance(field, BoolField)
        assert field.nullable is True
        assert field.null_probability == pytest.approx(0.5, abs=0.01)


# ---------------------------------------------------------------------------
# Date inference
# ---------------------------------------------------------------------------


class TestDateInference:
    def test_min_max(self):
        from datetime import date

        df = pl.DataFrame({"d": [date(2020, 1, 1), date(2023, 6, 15), date(2024, 12, 31)]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "d")
        assert isinstance(field, DateField)
        assert field.min_date == date(2020, 1, 1)
        assert field.max_date == date(2024, 12, 31)


# ---------------------------------------------------------------------------
# Preset detection
# ---------------------------------------------------------------------------


class TestPresetDetection:
    def test_email_by_name(self):
        assert _detect_preset("email", ["a@b.com", "c@d.org"]) == "email"
        assert _detect_preset("user_email", ["a@b.com"]) == "email"

    def test_email_validation_fails(self):
        """Column named email but values don't look like emails."""
        assert _detect_preset("email", ["not-an-email", "just-text"]) is None

    def test_phone(self):
        assert _detect_preset("phone_number", ["+1-555-1234"]) == "phone_number"
        assert _detect_preset("telephone", ["555-0100"]) == "phone_number"

    def test_url(self):
        assert _detect_preset("url", ["https://example.com"]) == "url"
        assert _detect_preset("website_url", ["http://test.org"]) == "url"

    def test_url_validation_fails(self):
        assert _detect_preset("url", ["/path/to/file", "relative"]) is None

    def test_uuid(self):
        assert _detect_preset("uuid", ["550e8400-e29b-41d4-a716-446655440000"]) == "uuid4"

    def test_city(self):
        assert _detect_preset("city", ["New York", "London"]) == "city"

    def test_no_match(self):
        assert _detect_preset("foobar", ["val1", "val2"]) is None

    # --- CamelCase / PascalCase support (R pointblank learning) ---
    def test_camel_case_email(self):
        assert _detect_preset("userEmail", ["a@b.com", "c@d.org"]) == "email"
        assert _detect_preset("EmailAddress", ["x@y.com"]) == "email"

    def test_camel_case_first_name(self):
        assert _detect_preset("firstName", ["Alice", "Bob"]) == "first_name"
        assert _detect_preset("FirstName", ["Alice", "Bob"]) == "first_name"

    def test_camel_case_last_name(self):
        assert _detect_preset("lastName", ["Smith", "Jones"]) == "last_name"

    def test_camel_case_phone(self):
        assert _detect_preset("phoneNumber", ["555-1234"]) == "phone_number"
        assert _detect_preset("mobilePhone", ["555-1234"]) == "phone_number"

    def test_camel_case_city(self):
        assert _detect_preset("homeCity", ["NYC"]) == "city"

    def test_dot_case_email(self):
        """Dot-separated column names (common in R data)."""
        assert _detect_preset("user.email", ["a@b.com"]) == "email"

    def test_abbreviations_from_r(self):
        """Abbreviations recognized by R pointblank's column_roles.R."""
        assert _detect_preset("addr", ["123 Main St"]) == "address"
        assert _detect_preset("prov", ["Ontario"]) == "state"
        assert _detect_preset("ctry", ["US", "CA"]) == "country_code_2"


# ---------------------------------------------------------------------------
# Lat/Lon numeric role detection (R pointblank learning)
# ---------------------------------------------------------------------------


class TestNumericRoleDetection:
    def test_latitude_float_column(self):
        """Float column named 'lat' with values in [-90, 90] → latitude preset."""
        df = pl.DataFrame({"lat": [40.7, 34.0, 51.5, -33.9, 48.8]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "lat")
        assert isinstance(field, StringField)
        assert field.preset == "latitude"

    def test_longitude_float_column(self):
        """Float column named 'longitude' with values in [-180, 180] → longitude preset."""
        df = pl.DataFrame({"longitude": [-74.0, -118.2, -0.1, 151.2, 2.3]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "longitude")
        assert isinstance(field, StringField)
        assert field.preset == "longitude"

    def test_lat_out_of_bounds_no_preset(self):
        """Values outside lat bounds should NOT trigger the preset."""
        df = pl.DataFrame({"lat": [100.0, 200.0, 300.0, 400.0, 500.0]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "lat")
        # Should fall back to FloatField since values aren't valid latitudes
        assert isinstance(field, FloatField)

    def test_generic_float_not_affected(self):
        """A float column without a geo name should stay FloatField."""
        df = pl.DataFrame({"price": [9.99, 19.99, 29.99]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "price")
        assert isinstance(field, FloatField)


# ---------------------------------------------------------------------------
# ID column detection (R pointblank learning)
# ---------------------------------------------------------------------------


class TestIdColumnDetection:
    def test_user_id_unique(self):
        """Columns with 'id' in name + unique values → unique=True."""
        df = pl.DataFrame({"user_id": list(range(1, 51))})
        schema = Schema.from_table(df)
        field = _get_field(schema, "user_id")
        assert isinstance(field, IntField)
        assert field.unique is True

    def test_record_id_camel_case(self):
        """CamelCase ID columns detected."""
        df = pl.DataFrame({"recordId": list(range(100, 150))})
        schema = Schema.from_table(df)
        field = _get_field(schema, "recordId")
        assert isinstance(field, IntField)
        assert field.unique is True


# ---------------------------------------------------------------------------
# Dtype classification
# ---------------------------------------------------------------------------


class TestDtypeClassification:
    @pytest.mark.parametrize(
        "dtype_str,expected",
        [
            ("Int64", "int"),
            ("int32", "int"),
            ("UInt8", "int"),
            ("Float64", "float"),
            ("float32", "float"),
            ("String", "string"),
            ("Utf8", "string"),
            ("Boolean", "bool"),
            ("Date", "date"),
            ("Datetime", "datetime"),
            ("Duration", "duration"),
            ("object", "string"),
        ],
    )
    def test_classification(self, dtype_str, expected):
        assert _classify_dtype(dtype_str) == expected


# ---------------------------------------------------------------------------
# Round-trip: from_table -> generate
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_generate_produces_valid_data(self):
        df = pl.DataFrame(
            {
                "id": list(range(1, 51)),
                "name": [f"person_{i}" for i in range(50)],
                "score": [float(i) * 1.5 for i in range(50)],
                "active": [i % 3 != 0 for i in range(50)],
            }
        )
        schema = Schema.from_table(df)
        result = schema.generate(n=20, seed=123)

        assert result.shape == (20, 4)
        assert "id" in result.columns
        assert "name" in result.columns
        assert "score" in result.columns
        assert "active" in result.columns

    def test_generate_respects_constraints(self):
        df = pl.DataFrame(
            {
                "val": list(range(100, 200)),
                "status": ["a", "b", "c"] * 33 + ["a"],
            }
        )
        schema = Schema.from_table(df)
        result = schema.generate(n=50, seed=7)

        # Integer values should be within inferred range
        assert result["val"].min() >= 100
        assert result["val"].max() <= 199

        # Status should only contain allowed values
        assert set(result["status"].to_list()).issubset({"a", "b", "c"})

    def test_sample_size_parameter(self):
        """sample_size should not raise and should still produce a valid schema."""
        df = pl.DataFrame({"x": list(range(1000))})
        schema = Schema.from_table(df, sample_size=50)
        field = _get_field(schema, "x")
        assert isinstance(field, IntField)
        # With sampling, the range might be narrower, but it should still work
        result = schema.generate(n=10, seed=1)
        assert result.shape == (10, 1)

    def test_pandas_round_trip(self):
        df = pd.DataFrame(
            {
                "a": list(range(1, 31)),
                "b": [float(i) * 0.5 for i in range(30)],
                "c": ["x", "y", "z"] * 10,
            }
        )
        schema = Schema.from_table(df)
        result = schema.generate(n=10, seed=23)
        assert result.shape == (10, 3)


# ---------------------------------------------------------------------------
# Datetime inference
# ---------------------------------------------------------------------------


class TestDatetimeInference:
    def test_min_max(self):
        from datetime import datetime

        df = pl.DataFrame({"ts": [datetime(2020, 1, 1, 12, 0), datetime(2023, 6, 15, 18, 30)]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "ts")
        assert isinstance(field, DatetimeField)
        assert field.min_date == datetime(2020, 1, 1, 12, 0)
        assert field.max_date == datetime(2023, 6, 15, 18, 30)

    def test_nullable(self):
        from datetime import datetime

        df = pl.DataFrame({"ts": [datetime(2022, 1, 1), None, datetime(2022, 12, 31)]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "ts")
        assert isinstance(field, DatetimeField)
        assert field.nullable is True
        assert field.null_probability > 0


# ---------------------------------------------------------------------------
# Duration inference
# ---------------------------------------------------------------------------


class TestDurationInference:
    def test_min_max(self):
        from datetime import timedelta

        df = pl.DataFrame({"elapsed": [timedelta(hours=1), timedelta(hours=5), timedelta(days=2)]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "elapsed")
        assert isinstance(field, DurationField)
        assert field.min_duration == timedelta(hours=1)
        assert field.max_duration == timedelta(days=2)

    def test_nullable(self):
        from datetime import timedelta

        df = pl.DataFrame({"elapsed": [timedelta(seconds=30), None, timedelta(minutes=10)]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "elapsed")
        assert isinstance(field, DurationField)
        assert field.nullable is True
        assert field.null_probability > 0


# ---------------------------------------------------------------------------
# schema_from_tbl() functional form
# ---------------------------------------------------------------------------


class TestSchemaFromTblFunction:
    def test_basic(self):
        df = pl.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
        schema = schema_from_tbl(df)
        assert schema.columns is not None
        assert len(schema.columns) == 2

    def test_passes_kwargs(self):
        """Ensure keyword arguments are forwarded to from_table."""
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5] * 10})
        schema = schema_from_tbl(df, categorical_threshold=10)
        field = _get_field(schema, "val")
        assert isinstance(field, IntField)
        assert field.allowed == [1, 2, 3, 4, 5]

    def test_detect_presets_false(self):
        """schema_from_tbl with detect_presets=False skips preset detection."""
        df = pl.DataFrame({"email": [f"user{i}@test.com" for i in range(50)]})
        schema = schema_from_tbl(df, detect_presets=False)
        field = _get_field(schema, "email")
        assert isinstance(field, StringField)
        assert field.preset is None


# ---------------------------------------------------------------------------
# detect_presets=False on from_dataframe
# ---------------------------------------------------------------------------


class TestDetectPresetsDisabled:
    def test_string_column_no_preset(self):
        """With detect_presets=False, string columns get length constraints instead of presets."""
        df = pl.DataFrame({"phone": [f"+1-555-{i:04d}" for i in range(100)]})
        schema = Schema.from_table(df, detect_presets=False)
        field = _get_field(schema, "phone")
        assert isinstance(field, StringField)
        assert field.preset is None
        # Should have length constraints instead
        assert field.min_length is not None or field.max_length is not None


# ---------------------------------------------------------------------------
# Fractional categorical_threshold
# ---------------------------------------------------------------------------


class TestFractionalCategoricalThreshold:
    def test_fraction_of_rows(self):
        """A float threshold is treated as fraction of total rows."""
        # 100 rows, 10 unique values → 10% of rows = threshold of 10 at 0.1
        df = pl.DataFrame({"x": list(range(10)) * 10})
        schema = Schema.from_table(df, categorical_threshold=0.1)
        field = _get_field(schema, "x")
        assert isinstance(field, IntField)
        assert field.allowed == list(range(10))

    def test_fraction_below_unique_count(self):
        """When unique count exceeds fractional threshold, use min/max instead."""
        # 50 rows, 25 unique → 0.1 * 50 = threshold of 5, so NOT categorical
        df = pl.DataFrame({"x": list(range(25)) * 2})
        schema = Schema.from_table(df, categorical_threshold=0.1)
        field = _get_field(schema, "x")
        assert isinstance(field, IntField)
        assert field.allowed is None
        assert field.min_val == 0
        assert field.max_val == 24


# ---------------------------------------------------------------------------
# Nullable lat/lon
# ---------------------------------------------------------------------------


class TestNullableLatLon:
    def test_nullable_latitude(self):
        """Lat column with nulls still detects preset and records null_probability."""
        df = pl.DataFrame({"lat": [40.7, None, 51.5, None, -33.9]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "lat")
        assert isinstance(field, StringField)
        assert field.preset == "latitude"
        assert field.nullable is True
        assert field.null_probability > 0

    def test_nullable_longitude(self):
        """Lon column with nulls still detects preset and records null_probability."""
        df = pl.DataFrame({"longitude": [-74.0, None, 139.7, 13.4, None]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "longitude")
        assert isinstance(field, StringField)
        assert field.preset == "longitude"
        assert field.nullable is True


# ---------------------------------------------------------------------------
# _validate_preset_values edge cases
# ---------------------------------------------------------------------------


class TestValidatePresetValues:
    def test_empty_list_returns_false(self):
        assert _validate_preset_values("email", []) is False

    def test_email_valid(self):
        assert _validate_preset_values("email", ["a@b.com", "c@d.org", "x@y.io"]) is True

    def test_email_invalid_below_threshold(self):
        # Only 1/4 matches → 25% < 70% threshold
        assert _validate_preset_values("email", ["a@b.com", "nope", "bad", "wrong"]) is False

    def test_uuid_valid(self):
        uuids = ["550e8400-e29b-41d4-a716-446655440000"] * 5
        assert _validate_preset_values("uuid4", uuids) is True

    def test_uuid_invalid(self):
        assert _validate_preset_values("uuid4", ["not-a-uuid", "also-not"]) is False

    def test_ipv4_valid(self):
        assert _validate_preset_values("ipv4", ["192.168.1.1", "10.0.0.1"]) is True

    def test_ipv4_invalid(self):
        assert _validate_preset_values("ipv4", ["hello", "world"]) is False

    def test_url_valid(self):
        assert _validate_preset_values("url", ["https://x.com", "http://y.org"]) is True

    def test_url_invalid(self):
        assert _validate_preset_values("url", ["/path/to", "relative"]) is False

    def test_unknown_preset_trusts_name(self):
        """Presets without a validator return True (trust name heuristic)."""
        assert _validate_preset_values("city", ["New York", "London"]) is True
        assert _validate_preset_values("first_name", ["Alice", "Bob"]) is True


# ---------------------------------------------------------------------------
# _validate_numeric_as_latitude / _validate_numeric_as_longitude
# ---------------------------------------------------------------------------


class TestValidateNumericGeographic:
    def test_latitude_valid(self):
        assert _validate_numeric_as_latitude([40.7, -33.9, 51.5, 0.0]) is True

    def test_latitude_invalid(self):
        assert _validate_numeric_as_latitude([100.0, 200.0, 300.0]) is False

    def test_latitude_empty(self):
        assert _validate_numeric_as_latitude([]) is False

    def test_latitude_all_none(self):
        assert _validate_numeric_as_latitude([None, None]) is False

    def test_latitude_mixed_with_none(self):
        """Nones are filtered out; only non-null values are checked."""
        assert _validate_numeric_as_latitude([40.7, None, -33.9, None]) is True

    def test_longitude_valid(self):
        assert _validate_numeric_as_longitude([-74.0, 139.7, 0.0, 180.0]) is True

    def test_longitude_invalid(self):
        assert _validate_numeric_as_longitude([200.0, 300.0, -200.0]) is False

    def test_longitude_empty(self):
        assert _validate_numeric_as_longitude([]) is False


# ---------------------------------------------------------------------------
# _detect_numeric_role
# ---------------------------------------------------------------------------


class TestDetectNumericRole:
    def test_latitude_detected(self):
        assert _detect_numeric_role("lat", [40.7, -33.9]) == "latitude"

    def test_longitude_detected(self):
        assert _detect_numeric_role("lon", [-74.0, 139.7]) == "longitude"

    def test_lat_out_of_bounds(self):
        assert _detect_numeric_role("lat", [200.0, 300.0]) is None

    def test_non_geo_name(self):
        assert _detect_numeric_role("price", [40.7, -33.9]) is None

    def test_camel_case_latitude(self):
        assert _detect_numeric_role("userLatitude", [45.0, -10.0]) == "latitude"


# ---------------------------------------------------------------------------
# Unsupported table type error
# ---------------------------------------------------------------------------


class TestUnsupportedTableType:
    def test_raises_type_error(self):
        """Passing a non-DataFrame object raises TypeError."""
        with pytest.raises(TypeError, match="not a DataFrame"):
            infer_fields_from_table({"not": "a dataframe"})


# ---------------------------------------------------------------------------
# Pandas null-handling edge cases
# ---------------------------------------------------------------------------


class TestPandasNullHandling:
    def test_nan_treated_as_none(self):
        """Pandas NaN values should be treated as nulls."""
        import numpy as np

        df = pd.DataFrame({"x": [1.0, np.nan, 3.0, np.nan, 5.0]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "x")
        assert isinstance(field, FloatField)
        assert field.nullable is True
        assert field.null_probability == pytest.approx(0.4, abs=0.01)

    def test_nat_treated_as_none(self):
        """Pandas NaT values in datetime columns should be treated as nulls."""
        df = pd.DataFrame({"ts": pd.to_datetime(["2020-01-01", pd.NaT, "2022-12-31"])})
        schema = Schema.from_table(df)
        field = _get_field(schema, "ts")
        assert isinstance(field, DatetimeField)
        assert field.nullable is True

    def test_string_column_with_none(self):
        """Pandas string columns with None should record null_probability."""
        df = pd.DataFrame({"name": ["Alice", None, "Charlie", None, "Eve"]})
        schema = Schema.from_table(df)
        field = _get_field(schema, "name")
        assert isinstance(field, StringField)
        assert field.nullable is True
        assert field.null_probability == pytest.approx(0.4, abs=0.01)


# ---------------------------------------------------------------------------
# Ibis/DuckDB table inference
# ---------------------------------------------------------------------------


class TestIbisTableInference:
    def test_duckdb_memtable(self):
        """Ibis memtable (DuckDB backend) should work with from_table."""
        ibis = pytest.importorskip("ibis")
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": ["a", "b", "c", "a", "b"]})
        tbl = ibis.memtable(df)
        schema = Schema.from_table(tbl)
        assert schema.columns is not None
        assert len(schema.columns) == 2
        field_x = _get_field(schema, "x")
        assert isinstance(field_x, IntField)

    def test_duckdb_table(self):
        """DuckDB native table should work with schema_from_tbl."""
        ibis = pytest.importorskip("ibis")
        conn = ibis.duckdb.connect()
        df = pd.DataFrame({"id": list(range(1, 21)), "value": [float(i) * 1.5 for i in range(20)]})
        conn.create_table("test_infer", df, overwrite=True)
        tbl = conn.table("test_infer")
        schema = schema_from_tbl(tbl)
        assert schema.columns is not None
        field_id = _get_field(schema, "id")
        assert isinstance(field_id, IntField)
        field_val = _get_field(schema, "value")
        assert isinstance(field_val, FloatField)
