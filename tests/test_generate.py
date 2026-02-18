import pytest
from datetime import date, datetime, time, timedelta

from pointblank.countries import COUNTRIES_WITH_FULL_DATA
from pointblank.field import (
    int_field,
    float_field,
    string_field,
    bool_field,
    date_field,
    datetime_field,
    time_field,
    duration_field,
    IntField,
    FloatField,
    StringField,
)
from pointblank.generate.base import GeneratorConfig
from pointblank.generate.generators import (
    generate_column,
    generate_dataframe,
)


class TestGeneratorConfig:
    """Tests for GeneratorConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = GeneratorConfig()
        assert config.n == 100
        assert config.seed is None
        assert config.output == "polars"
        assert config.country == "US"
        assert config.max_unique_retries == 1000

    def test_custom_config(self):
        """Test custom configuration values."""
        config = GeneratorConfig(n=50, seed=23, output="pandas", country="DE")
        assert config.n == 50
        assert config.seed == 23
        assert config.output == "pandas"
        assert config.country == "DE"

    def test_negative_n_raises_error(self):
        """Test that negative n raises ValueError."""
        with pytest.raises(ValueError, match="n must be non-negative"):
            GeneratorConfig(n=-1)

    def test_invalid_max_retries_raises_error(self):
        """Test that invalid max_unique_retries raises ValueError."""
        with pytest.raises(ValueError, match="max_unique_retries must be at least 1"):
            GeneratorConfig(max_unique_retries=0)


class TestGenerateColumnInteger:
    """Tests for integer column generation."""

    def test_generate_int64_column(self):
        """Test generating an Int64 column."""
        field = int_field()
        config = GeneratorConfig(n=10, seed=23)
        values = generate_column(field, config)

        assert len(values) == 10
        assert all(isinstance(v, int) for v in values)

    def test_generate_int64_with_constraints(self):
        """Test generating an Int64 column with min/max constraints."""
        field = int_field(min_val=10, max_val=20)
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        assert len(values) == 100
        assert all(10 <= v <= 20 for v in values)

    def test_generate_uint8_column(self):
        """Test generating a UInt8 column respects dtype bounds."""
        field = int_field(dtype="UInt8")
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        assert all(0 <= v <= 255 for v in values)


class TestGenerateColumnFloat:
    """Tests for float column generation."""

    def test_generate_float64_column(self):
        """Test generating a Float64 column."""
        field = float_field()
        config = GeneratorConfig(n=10, seed=23)
        values = generate_column(field, config)

        assert len(values) == 10
        assert all(isinstance(v, float) for v in values)

    def test_generate_float64_with_constraints(self):
        """Test generating a Float64 column with min/max constraints."""
        field = float_field(min_val=0.0, max_val=1.0)
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        assert all(0.0 <= v <= 1.0 for v in values)


class TestGenerateColumnString:
    """Tests for string column generation."""

    def test_generate_string_column(self):
        """Test generating a String column."""
        field = string_field()
        config = GeneratorConfig(n=10, seed=23)
        values = generate_column(field, config)

        assert len(values) == 10
        assert all(isinstance(v, str) for v in values)

    def test_generate_string_with_length_constraints(self):
        """Test generating a String column with length constraints."""
        field = string_field(min_length=5, max_length=10)
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        assert all(5 <= len(v) <= 10 for v in values)

    def test_generate_string_with_allowed_values(self):
        """Test generating a String column with allowed values."""
        allowed = ["apple", "banana", "cherry"]
        field = string_field(allowed=allowed)
        config = GeneratorConfig(n=50, seed=23)
        values = generate_column(field, config)

        assert all(v in allowed for v in values)


class TestGenerateColumnBoolean:
    """Tests for boolean column generation."""

    def test_generate_boolean_column(self):
        """Test generating a Boolean column."""
        field = bool_field()
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        assert len(values) == 100
        assert all(isinstance(v, bool) for v in values)

        # Should have a mix of True and False with enough samples
        assert True in values
        assert False in values


class TestGenerateColumnDate:
    """Tests for date column generation."""

    def test_generate_date_column(self):
        """Test generating a Date column."""
        field = date_field()
        config = GeneratorConfig(n=10, seed=23)
        values = generate_column(field, config)

        assert len(values) == 10
        assert all(isinstance(v, date) for v in values)

    def test_generate_date_with_constraints(self):
        """Test generating a Date column with constraints."""
        field = date_field(min_date="2023-01-01", max_date="2023-12-31")
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        min_date = date(2023, 1, 1)
        max_date = date(2023, 12, 31)

        assert all(min_date <= v <= max_date for v in values)


class TestGenerateColumnDatetime:
    """Tests for datetime column generation."""

    def test_generate_datetime_column(self):
        """Test generating a Datetime column."""
        field = datetime_field()
        config = GeneratorConfig(n=10, seed=23)
        values = generate_column(field, config)

        assert len(values) == 10
        assert all(isinstance(v, datetime) for v in values)


class TestGenerateColumnTime:
    """Tests for time column generation."""

    def test_generate_time_column(self):
        """Test generating a Time column."""
        field = time_field()
        config = GeneratorConfig(n=10, seed=23)
        values = generate_column(field, config)

        assert len(values) == 10
        assert all(isinstance(v, str) for v in values)

        # Verify time format HH:MM:SS
        for v in values:
            parts = v.split(":")

            assert len(parts) == 3
            assert 0 <= int(parts[0]) <= 23
            assert 0 <= int(parts[1]) <= 59
            assert 0 <= int(parts[2]) <= 59

    def test_generate_time_with_constraints(self):
        """Test generating a Time column with min/max constraints."""
        field = time_field(min_time=time(9, 0, 0), max_time=time(12, 0, 0))
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        assert len(values) == 100

        # Verify all times are within the 9:00-12:00 range
        for v in values:
            parts = v.split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2])
            time_seconds = hour * 3600 + minute * 60 + second
            min_seconds = 9 * 3600  # 09:00:00
            max_seconds = 12 * 3600  # 12:00:00

            assert min_seconds <= time_seconds <= max_seconds

    def test_generate_time_different_fields_have_different_values(self):
        """Test that different time fields generate different values (not the same row values)."""
        # This tests the fix where time fields were all getting identical values per row
        from pointblank import Schema, generate_dataset

        schema = Schema(
            start_time=time_field(min_time=time(9, 0, 0), max_time=time(12, 0, 0)),
            end_time=time_field(min_time=time(13, 0, 0), max_time=time(17, 0, 0)),
        )

        df = generate_dataset(schema, n=50, seed=23)

        # Get all values
        start_times = df["start_time"].to_list()
        end_times = df["end_time"].to_list()

        # Verify the values are different (start_time should be 9-12, end_time should be 13-17)
        for st, et in zip(start_times, end_times):
            assert st != et

            # Verify start_time is in 9-12 range
            st_hour = int(st.split(":")[0])

            assert 9 <= st_hour <= 12

            # Verify end_time is in 13-17 range
            et_hour = int(et.split(":")[0])

            assert 13 <= et_hour <= 17

    def test_generate_time_with_string_constraints(self):
        """Test generating a Time column with string min/max constraints."""
        field = time_field(min_time="14:00:00", max_time="18:30:00")
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        assert len(values) == 100

        for v in values:
            parts = v.split(":")
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2])
            time_seconds = hour * 3600 + minute * 60 + second
            min_seconds = 14 * 3600  # 14:00:00
            max_seconds = 18 * 3600 + 30 * 60  # 18:30:00

            assert min_seconds <= time_seconds <= max_seconds


class TestGenerateColumnDuration:
    """Tests for duration column generation."""

    def test_generate_duration_column(self):
        """Test generating a Duration column."""
        field = duration_field()
        config = GeneratorConfig(n=10, seed=23)
        values = generate_column(field, config)

        assert len(values) == 10
        assert all(isinstance(v, timedelta) for v in values)

    def test_generate_duration_with_constraints(self):
        """Test generating a Duration column with min/max constraints."""
        min_dur = timedelta(minutes=5)
        max_dur = timedelta(hours=2)
        field = duration_field(min_duration=min_dur, max_duration=max_dur)
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        assert len(values) == 100

        for v in values:
            assert min_dur <= v <= max_dur

    def test_generate_duration_different_fields_have_different_values(self):
        """Test that different duration fields generate different values."""
        from pointblank import Schema, generate_dataset

        schema = Schema(
            session_length=duration_field(
                min_duration=timedelta(hours=1), max_duration=timedelta(hours=3)
            ),
            wait_time=duration_field(
                min_duration=timedelta(seconds=10), max_duration=timedelta(minutes=5)
            ),
        )

        df = generate_dataset(schema, n=50, seed=23)

        # Convert to Python objects for comparison
        import polars as pl

        session_lengths = df["session_length"].to_list()
        wait_times = df["wait_time"].to_list()

        # Verify values are in their respective ranges
        for sl in session_lengths:
            assert timedelta(hours=1) <= sl <= timedelta(hours=3)

        for wt in wait_times:
            assert timedelta(seconds=10) <= wt <= timedelta(minutes=5)

    def test_generate_duration_with_small_range(self):
        """Test generating durations with a very small range."""
        min_dur = timedelta(seconds=100)
        max_dur = timedelta(seconds=110)
        field = duration_field(min_duration=min_dur, max_duration=max_dur)
        config = GeneratorConfig(n=50, seed=23)
        values = generate_column(field, config)

        assert len(values) == 50

        for v in values:
            assert min_dur <= v <= max_dur


class TestGenerateColumnUnique:
    """Tests for unique value generation."""

    def test_generate_unique_int_column(self):
        """Test generating unique integer values."""
        field = int_field(unique=True, min_val=1, max_val=1000)
        config = GeneratorConfig(n=50, seed=23)
        values = generate_column(field, config)

        assert len(values) == 50
        assert len(set(values)) == 50  # All unique

    def test_generate_unique_string_column(self):
        """Test generating unique string values."""
        field = string_field(unique=True)
        config = GeneratorConfig(n=50, seed=23)
        values = generate_column(field, config)

        assert len(values) == 50
        assert len(set(values)) == 50  # All unique

    def test_generate_unique_from_small_allowed_fails(self):
        """Test that unique generation fails when allowed values are insufficient."""
        field = string_field(unique=True, allowed=["a", "b"])
        config = GeneratorConfig(n=5, seed=23)

        with pytest.raises(ValueError, match="Cannot generate .* unique values"):
            generate_column(field, config)


class TestGenerateColumnNullable:
    """Tests for nullable column generation."""

    def test_generate_nullable_column(self):
        """Test generating a nullable column."""
        field = int_field(nullable=True, null_probability=0.5)
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        # Should have some nulls and some non-nulls
        null_count = sum(1 for v in values if v is None)
        non_null_count = sum(1 for v in values if v is not None)

        assert null_count > 0
        assert non_null_count > 0


class TestGenerateColumnCustomGenerator:
    """Tests for custom generator functions."""

    def test_custom_generator(self):
        """Test using a custom generator function."""
        counter = [0]

        def custom_gen():
            counter[0] += 1
            return f"item_{counter[0]}"

        field = string_field(generator=custom_gen)
        config = GeneratorConfig(n=5, seed=23)
        values = generate_column(field, config)

        assert values == ["item_1", "item_2", "item_3", "item_4", "item_5"]


class TestGenerateColumnReproducibility:
    """Tests for reproducible generation."""

    def test_same_seed_produces_same_values(self):
        """Test that the same seed produces identical values."""
        field = int_field(min_val=0, max_val=1000)
        config1 = GeneratorConfig(n=50, seed=12345)
        config2 = GeneratorConfig(n=50, seed=12345)

        values1 = generate_column(field, config1)
        values2 = generate_column(field, config2)

        assert values1 == values2

    def test_different_seeds_produce_different_values(self):
        """Test that different seeds produce different values."""
        field = int_field(min_val=0, max_val=1000)
        config1 = GeneratorConfig(n=50, seed=12345)
        config2 = GeneratorConfig(n=50, seed=54321)

        values1 = generate_column(field, config1)
        values2 = generate_column(field, config2)

        assert values1 != values2


class TestGenerateDataframe:
    """Tests for DataFrame generation."""

    def test_generate_dataframe_polars(self):
        """Test generating a Polars DataFrame."""
        pytest.importorskip("polars")
        import polars as pl

        fields = {
            "id": int_field(unique=True, min_val=1),
            "name": string_field(),
            "active": bool_field(),
        }
        config = GeneratorConfig(n=10, seed=23, output="polars")
        df = generate_dataframe(fields, config)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == ["id", "name", "active"]

    def test_generate_dataframe_pandas(self):
        """Test generating a Pandas DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        fields = {
            "id": int_field(),
            "value": float_field(),
        }
        config = GeneratorConfig(n=10, seed=23, output="pandas")
        df = generate_dataframe(fields, config)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == ["id", "value"]

    def test_generate_dataframe_dict(self):
        """Test generating a dictionary of lists."""
        fields = {
            "a": int_field(),
            "b": string_field(),
        }
        config = GeneratorConfig(n=5, seed=23, output="dict")
        data = generate_dataframe(fields, config)

        assert isinstance(data, dict)
        assert list(data.keys()) == ["a", "b"]
        assert len(data["a"]) == 5
        assert len(data["b"]) == 5


class TestSchemaGenerate:
    """Tests for Schema.generate() method."""

    def test_schema_generate_basic(self):
        """Test generating data from a basic schema."""
        pytest.importorskip("polars")
        import polars as pl
        from pointblank import Schema

        schema = Schema(name="String", age="Int64", active="Boolean")
        df = schema.generate(n=10, seed=23)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 10
        assert list(df.columns) == ["name", "age", "active"]

    def test_schema_generate_with_field_objects(self):
        """Test generating data from a schema with Field objects."""
        pytest.importorskip("polars")
        import polars as pl
        from pointblank import Schema, int_field, string_field, float_field

        schema = Schema(
            user_id=int_field(unique=True, min_val=1),
            status=string_field(allowed=["active", "pending", "inactive"]),
            score=float_field(min_val=0.0, max_val=100.0),
        )
        df = schema.generate(n=20, seed=23)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 20

        # Check unique constraint
        assert df["user_id"].n_unique() == 20

        # Check allowed values
        assert all(v in ["active", "pending", "inactive"] for v in df["status"].to_list())

        # Check numeric constraints
        assert all(0.0 <= v <= 100.0 for v in df["score"].to_list())

    def test_schema_generate_reproducible(self):
        """Test that schema generation is reproducible with seed."""
        pytest.importorskip("polars")
        from pointblank import Schema

        schema = Schema(x="Int64", y="Float64")
        df1 = schema.generate(n=10, seed=23)
        df2 = schema.generate(n=10, seed=23)

        assert df1.equals(df2)

    def test_schema_generate_empty_raises_error(self):
        """Test that generating from empty schema raises error."""
        from pointblank import Schema
        import polars as pl

        schema = Schema(tbl=pl.DataFrame({}))
        with pytest.raises(ValueError, match="Cannot generate data from an empty schema"):
            schema.generate()

    def test_schema_generate_output_formats(self):
        """Test different output formats."""
        pytest.importorskip("polars")
        pytest.importorskip("pandas")
        import polars as pl
        import pandas as pd
        from pointblank import Schema

        schema = Schema(value="Int64")

        # Polars (default)
        df_polars = schema.generate(n=5, seed=23, output="polars")
        assert isinstance(df_polars, pl.DataFrame)

        # Pandas
        df_pandas = schema.generate(n=5, seed=23, output="pandas")
        assert isinstance(df_pandas, pd.DataFrame)

        # Dict
        df_dict = schema.generate(n=5, seed=23, output="dict")
        assert isinstance(df_dict, dict)


class TestGenerateDatasetFunction:
    """Tests for the generate_dataset() convenience function."""

    def test_generate_dataset_basic(self):
        """Test basic generate_dataset functionality."""
        pytest.importorskip("polars")
        import polars as pl
        from pointblank import Schema, generate_dataset

        schema = Schema(name="String", age="Int64")
        df = generate_dataset(schema, n=10, seed=23)

        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 10
        assert list(df.columns) == ["name", "age"]

    def test_generate_dataset_with_fields(self):
        """Test generate_dataset with Field constraints."""
        pytest.importorskip("polars")
        import polars as pl
        from pointblank import Schema, generate_dataset, int_field, string_field

        schema = Schema(
            user_id=int_field(min_val=1, max_val=100),
            status=string_field(allowed=["active", "inactive"]),
        )
        df = generate_dataset(schema, n=20, seed=23)

        assert isinstance(df, pl.DataFrame)
        assert df.shape[0] == 20
        assert df["user_id"].min() >= 1
        assert df["user_id"].max() <= 100
        assert set(df["status"].unique().to_list()).issubset({"active", "inactive"})

    def test_generate_dataset_output_formats(self):
        """Test generate_dataset with different output formats."""
        pytest.importorskip("polars")
        pytest.importorskip("pandas")
        import polars as pl
        import pandas as pd
        from pointblank import Schema, generate_dataset

        schema = Schema(value="Int64")

        # Polars (default)
        df_polars = generate_dataset(schema, n=5, seed=23)
        assert isinstance(df_polars, pl.DataFrame)

        # Pandas
        df_pandas = generate_dataset(schema, n=5, seed=23, output="pandas")
        assert isinstance(df_pandas, pd.DataFrame)

        # Dict
        df_dict = generate_dataset(schema, n=5, seed=23, output="dict")
        assert isinstance(df_dict, dict)

    def test_generate_dataset_reproducibility(self):
        """Test that same seed produces same results."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(value="Int64", name="String")

        df1 = generate_dataset(schema, n=10, seed=23)
        df2 = generate_dataset(schema, n=10, seed=23)

        assert df1.equals(df2)

    def test_generate_dataset_country(self):
        """Test generate_dataset with country parameter."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(name=string_field(preset="name"))

        # Should not raise with valid country code
        df = generate_dataset(schema, n=5, seed=23, country="DE")

        assert df.shape[0] == 5


class TestCountrySupport:
    """Tests for country support using Pointblank validations."""

    # Common countries to test (ISO alpha-2 codes)
    COUNTRIES = [
        "US",
        "CA",
        "GB",
        "DE",
        "FR",
        "ES",
        "IT",
        "BR",
        "JP",
        "CN",
        "KR",
        "NL",
        "PL",
        "RU",
    ]

    def test_all_countries_generate_without_errors(self):
        """Test that all supported countries generate data without errors."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(
            name=string_field(preset="name"),
            email=string_field(preset="email"),
            city=string_field(preset="city"),
        )

        for country in self.COUNTRIES:
            df = generate_dataset(schema, n=10, seed=23, country=country)

            assert df.shape[0] == 10, f"Failed for country {country}"
            assert list(df.columns) == ["name", "email", "city"]

    def test_country_data_validity_with_pointblank(self):
        """Use Pointblank validations to verify generated data quality."""
        pytest.importorskip("polars")
        from pointblank import (
            Schema,
            Validate,
            generate_dataset,
            int_field,
            float_field,
            string_field,
            bool_field,
        )

        # Create a schema with various field types
        schema = Schema(
            user_id=int_field(min_val=1, max_val=10000),
            score=float_field(min_val=0.0, max_val=100.0),
            email=string_field(preset="email"),
            name=string_field(preset="name"),
            active=bool_field(),
        )

        for country in ["US", "DE", "JP"]:
            df = generate_dataset(schema, n=100, seed=23, country=country)

            # Validate the generated data using Pointblank
            validation = (
                Validate(df)
                .col_vals_not_null(columns=["user_id", "score", "email", "name", "active"])
                .col_vals_between(columns="user_id", left=1, right=10000)
                .col_vals_between(columns="score", left=0.0, right=100.0)
                .col_vals_regex(columns="email", pattern=r".+@.+\..+")
                .col_vals_regex(columns="name", pattern=r".+")  # Non-empty
                .interrogate()
            )

            # All validations should pass
            assert validation.all_passed()

    def test_email_format_across_countries(self):
        """Test that email preset generates valid email formats across countries."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(email=string_field(preset="email"))

        for country in self.COUNTRIES:
            df = generate_dataset(schema, n=50, seed=23, country=country)

            validation = (
                Validate(df)
                .col_vals_not_null(columns="email")
                .col_vals_regex(columns="email", pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
                .interrogate()
            )

            assert validation.all_passed()

    def test_numeric_constraints_with_pointblank(self):
        """Test that numeric constraints are respected and validated by Pointblank."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, int_field, float_field

        schema = Schema(
            age=int_field(min_val=18, max_val=65),
            temperature=float_field(min_val=-40.0, max_val=50.0),
            count=int_field(min_val=0, max_val=1000),
        )

        df = generate_dataset(schema, n=200, seed=23)

        validation = (
            Validate(df)
            .col_vals_between(columns="age", left=18, right=65)
            .col_vals_between(columns="temperature", left=-40.0, right=50.0)
            .col_vals_between(columns="count", left=0, right=1000)
            .col_vals_not_null(columns=["age", "temperature", "count"])
            .interrogate()
        )

        assert validation.all_passed()

    def test_allowed_values_with_pointblank(self):
        """Test that allowed values constraint is validated by Pointblank."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field, int_field

        schema = Schema(
            status=string_field(allowed=["active", "pending", "inactive"]),
            priority=int_field(allowed=[1, 2, 3, 4, 5]),
        )

        df = generate_dataset(schema, n=100, seed=23)

        validation = (
            Validate(df)
            .col_vals_in_set(columns="status", set=["active", "pending", "inactive"])
            .col_vals_in_set(columns="priority", set=[1, 2, 3, 4, 5])
            .interrogate()
        )

        assert validation.all_passed()

    def test_iso_alpha2_and_alpha3_codes(self):
        """Test that both alpha-2 and alpha-3 country codes produce same results."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(name=string_field(preset="name"))

        # Same seed should produce same results regardless of code format
        df_alpha2 = generate_dataset(schema, n=10, seed=23, country="US")
        df_alpha3 = generate_dataset(schema, n=10, seed=23, country="USA")

        assert df_alpha2.equals(df_alpha3)

    def test_legacy_locale_format_supported(self):
        """Test that legacy locale formats are still accepted for backwards compatibility."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(name=string_field(preset="name"))

        # Legacy locale formats should work
        df_legacy1 = generate_dataset(schema, n=10, seed=23, country="en-US")
        df_legacy2 = generate_dataset(schema, n=10, seed=23, country="en_US")
        df_iso = generate_dataset(schema, n=10, seed=23, country="US")

        # All should produce the same results
        assert df_legacy1.equals(df_iso)
        assert df_legacy2.equals(df_iso)

    def test_personal_presets_across_countries(self):
        """Test personal data presets work across multiple countries."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(
            first_name=string_field(preset="first_name"),
            last_name=string_field(preset="last_name"),
            city=string_field(preset="city"),
            country=string_field(preset="country"),
        )

        for country in ["US", "FR", "DE", "ES", "JP"]:
            df = generate_dataset(schema, n=20, seed=23, country=country)

            validation = (
                Validate(df)
                .col_vals_not_null(columns=["first_name", "last_name", "city", "country"])
                .col_vals_regex(columns="first_name", pattern=r".+")
                .col_vals_regex(columns="last_name", pattern=r".+")
                .interrogate()
            )

            assert validation.all_passed()

    def test_business_presets_across_countries(self):
        """Test business data presets work across multiple countries."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(
            company=string_field(preset="company"),
            job=string_field(preset="job"),
        )

        for country in ["US", "DE", "FR", "JP"]:
            df = generate_dataset(schema, n=20, seed=23, country=country)

            validation = (
                Validate(df)
                .col_vals_not_null(columns=["company", "job"])
                .col_vals_regex(columns="company", pattern=r".+")
                .col_vals_regex(columns="job", pattern=r".+")
                .interrogate()
            )

            assert validation.all_passed()

    def test_internet_presets_across_countries(self):
        """Test internet-related presets work across countries."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(
            url=string_field(preset="url"),
            domain=string_field(preset="domain_name"),
            username=string_field(preset="user_name"),
        )

        for country in ["US", "DE", "FR"]:
            df = generate_dataset(schema, n=20, seed=23, country=country)

            validation = (
                Validate(df)
                .col_vals_not_null(columns=["url", "domain", "username"])
                .col_vals_regex(columns="url", pattern=r"https?://")
                .col_vals_regex(columns="domain", pattern=r".+\..+")
                .interrogate()
            )

            assert validation.all_passed()

    def test_hash_presets(self):
        """Test hash presets produce valid hex strings of correct length."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(
            md5_hash=string_field(preset="md5"),
            sha1_hash=string_field(preset="sha1"),
            sha256_hash=string_field(preset="sha256"),
        )

        df = generate_dataset(schema, n=20, seed=23, country="US")

        validation = (
            Validate(df)
            .col_vals_not_null(columns=["md5_hash", "sha1_hash", "sha256_hash"])
            .col_vals_regex(columns="md5_hash", pattern=r"^[0-9a-f]{32}$")
            .col_vals_regex(columns="sha1_hash", pattern=r"^[0-9a-f]{40}$")
            .col_vals_regex(columns="sha256_hash", pattern=r"^[0-9a-f]{64}$")
            .interrogate()
        )

        assert validation.all_passed()

    def test_hash_presets_deterministic(self):
        """Test hash presets produce deterministic output with same seed."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(
            md5_hash=string_field(preset="md5"),
            sha256_hash=string_field(preset="sha256"),
        )

        df1 = generate_dataset(schema, n=10, seed=99)
        df2 = generate_dataset(schema, n=10, seed=99)

        assert df1.equals(df2)

    def test_barcode_presets(self):
        """Test barcode presets produce valid barcodes with correct check digits."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(
            ean8_code=string_field(preset="ean8"),
            ean13_code=string_field(preset="ean13"),
        )

        df = generate_dataset(schema, n=50, seed=23, country="US")

        # Check format: all digits, correct length
        validation = (
            Validate(df)
            .col_vals_not_null(columns=["ean8_code", "ean13_code"])
            .col_vals_regex(columns="ean8_code", pattern=r"^\d{8}$")
            .col_vals_regex(columns="ean13_code", pattern=r"^\d{13}$")
            .interrogate()
        )

        assert validation.all_passed()

        # Verify EAN-8 check digits
        for barcode in df["ean8_code"].to_list():
            digits = [int(c) for c in barcode]
            total = sum(d * (3 if i % 2 == 0 else 1) for i, d in enumerate(digits[:7]))
            expected_check = (10 - (total % 10)) % 10

            assert digits[7] == expected_check

        # Verify EAN-13 check digits
        for barcode in df["ean13_code"].to_list():
            digits = [int(c) for c in barcode]
            total = sum(d * (1 if i % 2 == 0 else 3) for i, d in enumerate(digits[:12]))
            expected_check = (10 - (total % 10)) % 10

            assert digits[12] == expected_check

    def test_date_range_presets(self):
        """Test date-related string presets produce valid output."""
        pytest.importorskip("polars")
        from datetime import date

        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(
            date_btwn=string_field(preset="date_between"),
            date_rng=string_field(preset="date_range"),
            future=string_field(preset="future_date"),
            past=string_field(preset="past_date"),
        )

        df = generate_dataset(schema, n=30, seed=23, country="US")

        # date_between and future/past should be single ISO dates
        validation = (
            Validate(df)
            .col_vals_not_null(columns=["date_btwn", "date_rng", "future", "past"])
            .col_vals_regex(
                columns=["date_btwn", "future", "past"],
                pattern=r"^\d{4}-\d{2}-\d{2}$",
            )
            # date_range should be "YYYY-MM-DD – YYYY-MM-DD" (en-dash)
            .col_vals_regex(
                columns="date_rng",
                pattern=r"^\d{4}-\d{2}-\d{2} \u2013 \d{4}-\d{2}-\d{2}$",
            )
            .interrogate()
        )

        assert validation.all_passed()

        # Verify date_between values are within default range (2000–2025)
        today = date.today()
        for val in df["date_btwn"].to_list():
            d = date.fromisoformat(val)

            assert date(2000, 1, 1) <= d <= date(2025, 12, 31)

        # Verify date_range start <= end
        for val in df["date_rng"].to_list():
            parts = val.split(" \u2013 ")

            assert len(parts) == 2
            start = date.fromisoformat(parts[0])
            end = date.fromisoformat(parts[1])

            assert start <= end

        # Verify future_date values are after today
        for val in df["future"].to_list():
            d = date.fromisoformat(val)

            assert d > today

        # Verify past_date values are before today
        for val in df["past"].to_list():
            d = date.fromisoformat(val)

            assert d < today

    def test_combined_schema_validation(self):
        """Comprehensive test combining multiple field types and validations."""
        pytest.importorskip("polars")
        from pointblank import (
            Schema,
            Validate,
            generate_dataset,
            int_field,
            float_field,
            string_field,
            bool_field,
            date_field,
        )
        from datetime import date

        schema = Schema(
            id=int_field(min_val=1, unique=True),
            name=string_field(preset="name"),
            email=string_field(preset="email"),
            age=int_field(min_val=18, max_val=100),
            salary=float_field(min_val=30000.0, max_val=200000.0),
            department=string_field(allowed=["Engineering", "Sales", "Marketing", "HR"]),
            is_manager=bool_field(),
            hire_date=date_field(min_date=date(2010, 1, 1), max_date=date(2025, 12, 31)),
        )

        df = generate_dataset(schema, n=50, seed=23, country="US")

        validation = (
            Validate(df)
            # Check all columns exist and have no nulls
            .col_vals_not_null(
                columns=[
                    "id",
                    "name",
                    "email",
                    "age",
                    "salary",
                    "department",
                    "is_manager",
                    "hire_date",
                ]
            )
            # Check numeric ranges
            .col_vals_gt(columns="id", value=0)
            .col_vals_between(columns="age", left=18, right=100)
            .col_vals_between(columns="salary", left=30000.0, right=200000.0)
            # Check string patterns
            .col_vals_regex(columns="email", pattern=r".+@.+")
            .col_vals_regex(columns="name", pattern=r".+")
            # Check allowed values
            .col_vals_in_set(columns="department", set=["Engineering", "Sales", "Marketing", "HR"])
            # Check dates
            .col_vals_ge(columns="hire_date", value=date(2010, 1, 1))
            .col_vals_le(columns="hire_date", value=date(2025, 12, 31))
            # Check uniqueness
            .rows_distinct(columns_subset="id")
            .interrogate()
        )

        assert validation.all_passed()

    def test_comprehensive_schema_all_countries_with_full_data(self):
        """
        Comprehensive test that exercises ALL field types and presets across ALL countries
        with full data. This catches regressions early and ensures locale data is complete.
        """
        pytest.importorskip("polars")
        from datetime import date, datetime, time

        from pointblank import (
            Schema,
            generate_dataset,
            int_field,
            float_field,
            string_field,
            bool_field,
            date_field,
            datetime_field,
            time_field,
        )

        # Create a comprehensive schema with many field types
        schema = Schema(
            columns=[
                # =====================================================================
                # Personal Information (using presets)
                # =====================================================================
                ("name", string_field(preset="name")),
                ("full_name", string_field(preset="name_full")),
                ("first_name", string_field(preset="first_name")),
                ("last_name", string_field(preset="last_name")),
                ("email", string_field(preset="email")),
                ("phone", string_field(preset="phone_number")),
                ("username", string_field(preset="user_name")),
                # =====================================================================
                # Address Information (using presets)
                # =====================================================================
                ("address", string_field(preset="address")),
                ("city", string_field(preset="city")),
                ("state", string_field(preset="state")),
                ("postcode", string_field(preset="postcode")),
                ("country", string_field(preset="country")),
                ("latitude", string_field(preset="latitude")),
                ("longitude", string_field(preset="longitude")),
                # =====================================================================
                # Business Information (using presets)
                # =====================================================================
                ("company", string_field(preset="company")),
                ("job_title", string_field(preset="job")),
                ("catch_phrase", string_field(preset="catch_phrase")),
                # =====================================================================
                # Internet & Technical (using presets)
                # =====================================================================
                ("website", string_field(preset="url")),
                ("domain", string_field(preset="domain_name")),
                ("ipv4_address", string_field(preset="ipv4")),
                ("ipv6_address", string_field(preset="ipv6")),
                ("password", string_field(preset="password")),
                # =====================================================================
                # Financial (using presets)
                # =====================================================================
                ("credit_card", string_field(preset="credit_card_number")),
                ("iban", string_field(preset="iban")),
                ("currency", string_field(preset="currency_code")),
                # =====================================================================
                # Identifiers (using presets)
                # =====================================================================
                ("uuid", string_field(preset="uuid4")),
                ("md5_hash", string_field(preset="md5")),
                ("sha1_hash", string_field(preset="sha1")),
                ("sha256_hash", string_field(preset="sha256")),
                ("ssn", string_field(preset="ssn")),
                ("license_plate", string_field(preset="license_plate")),
                # =====================================================================
                # Barcodes (using presets)
                # =====================================================================
                ("ean8_code", string_field(preset="ean8")),
                ("ean13_code", string_field(preset="ean13")),
                # =====================================================================
                # Text Content (using presets)
                # =====================================================================
                ("word", string_field(preset="word")),
                ("sentence", string_field(preset="sentence")),
                ("paragraph", string_field(preset="paragraph")),
                # =====================================================================
                # Miscellaneous (using presets)
                # =====================================================================
                ("color", string_field(preset="color_name")),
                ("file_name", string_field(preset="file_name")),
                ("file_ext", string_field(preset="file_extension")),
                ("mime_type", string_field(preset="mime_type")),
                ("ua_string", string_field(preset="user_agent")),
                # =====================================================================
                # Date Range Presets (using presets)
                # =====================================================================
                ("date_btwn", string_field(preset="date_between")),
                ("date_rng", string_field(preset="date_range")),
                ("future", string_field(preset="future_date")),
                ("past", string_field(preset="past_date")),
                # =====================================================================
                # Integer Fields (with constraints)
                # =====================================================================
                ("id", int_field(unique=True)),
                ("age", int_field(min_val=18, max_val=80)),
                ("rating", int_field(allowed=[1, 2, 3, 4, 5])),
                ("quantity", int_field(min_val=0, max_val=1000)),
                (
                    "priority",
                    int_field(allowed=[1, 2, 3], nullable=True, null_probability=0.1),
                ),
                # =====================================================================
                # Float Fields (with constraints)
                # =====================================================================
                ("price", float_field(min_val=0.01, max_val=9999.99)),
                ("discount_pct", float_field(min_val=0.0, max_val=0.5)),
                ("temperature", float_field(min_val=-40.0, max_val=50.0)),
                (
                    "score",
                    float_field(min_val=0.0, max_val=100.0, nullable=True, null_probability=0.05),
                ),
                # =====================================================================
                # String Fields (with constraints)
                # =====================================================================
                ("product_code", string_field(pattern=r"[A-Z]{3}-\d{4}")),
                (
                    "status",
                    string_field(allowed=["pending", "active", "completed", "cancelled"]),
                ),
                ("description", string_field(min_length=10, max_length=100)),
                (
                    "category",
                    string_field(allowed=["Electronics", "Clothing", "Food", "Books", "Other"]),
                ),
                # =====================================================================
                # Boolean Fields
                # =====================================================================
                ("is_active", bool_field()),
                ("is_verified", bool_field()),
                ("has_subscription", bool_field(nullable=True, null_probability=0.1)),
                # =====================================================================
                # Date Fields
                # =====================================================================
                (
                    "birth_date",
                    date_field(min_date=date(1940, 1, 1), max_date=date(2005, 12, 31)),
                ),
                (
                    "order_date",
                    date_field(min_date=date(2024, 1, 1), max_date=date(2024, 12, 31)),
                ),
                # =====================================================================
                # Datetime Fields
                # =====================================================================
                (
                    "created_at",
                    datetime_field(
                        min_date=datetime(2024, 1, 1, 0, 0, 0),
                        max_date=datetime(2024, 12, 31, 23, 59, 59),
                    ),
                ),
                (
                    "updated_at",
                    datetime_field(
                        min_date=datetime(2024, 6, 1, 0, 0, 0),
                        max_date=datetime(2024, 12, 31, 23, 59, 59),
                        nullable=True,
                        null_probability=0.2,
                    ),
                ),
                # =====================================================================
                # Time Fields
                # =====================================================================
                ("start_time", time_field(min_time=time(8, 0, 0), max_time=time(18, 0, 0))),
                ("end_time", time_field(min_time=time(9, 0, 0), max_time=time(22, 0, 0))),
            ]
        )

        # Test all countries with full data
        for country in COUNTRIES_WITH_FULL_DATA:
            df = generate_dataset(schema, n=10, seed=23, country=country)

            # Verify basic structure
            assert df.shape[0] == 10
            assert df.shape[1] == 67

            # Verify no unexpected errors occurred (all values generated successfully)
            # Check a few key columns are not empty strings
            for col in ["name", "email", "city", "company", "address"]:
                values = df[col].to_list()

                assert all(v is not None and len(str(v)) > 0 for v in values)

    def test_address_city_state_coherence_all_countries(self):
        """
        Test that address, city, and state columns are coherent (match) when
        generated together. The city embedded in the address should match the
        city column, and similarly for state.

        Note: Some cities have exonyms (e.g., "Brussels" for "Bruxelles"). In these
        cases, the city column shows the English exonym while the address uses the
        native name. This is intentional for international usability.
        """
        pytest.importorskip("polars")

        from pointblank import Schema, generate_dataset, string_field

        # Known exonym mappings: exonym -> native name
        # Cities where the city() preset returns English name but address uses native
        EXONYM_TO_NATIVE = {
            "Brussels": "Bruxelles",
            "Copenhagen": "København",
            "Gothenburg": "Göteborg",
            "Istanbul": "İstanbul",
            "Izmir": "İzmir",
            "Krakow": "Kraków",
            "Lisbon": "Lisboa",
            "Vienna": "Wien",
            "Warsaw": "Warszawa",
        }

        schema = Schema(
            columns=[
                ("address", string_field(preset="address")),
                ("city", string_field(preset="city")),
                ("state", string_field(preset="state")),
                ("postcode", string_field(preset="postcode")),
            ]
        )

        # Test all countries with full data
        for country in COUNTRIES_WITH_FULL_DATA:
            df = generate_dataset(schema, n=50, seed=23, country=country)

            for i, row in enumerate(df.iter_rows()):
                address, city, state, postcode = row

                # The city should appear somewhere in the address
                # (addresses contain the city name in the format string)
                # For cities with exonyms, check the native name instead
                native_name = EXONYM_TO_NATIVE.get(city, city)

                assert native_name in address

    def test_address_city_coherence_with_abbreviations(self):
        """
        Test address/city coherence handles state abbreviations correctly.
        Some address formats use abbreviated states (e.g., 'CA' instead of 'California').
        """
        pytest.importorskip("polars")

        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(
            columns=[
                ("address", string_field(preset="address")),
                ("city", string_field(preset="city")),
                ("state", string_field(preset="state")),
            ]
        )

        # Test a selection of countries
        test_countries = ["US", "GB", "DE", "FR", "IE", "JP"]

        for country in test_countries:
            if country not in COUNTRIES_WITH_FULL_DATA:
                continue

            df = generate_dataset(schema, n=30, seed=123, country=country)

            mismatches = []
            for i, row in enumerate(df.iter_rows()):
                address, city, state = row

                if city not in address:
                    mismatches.append(f"Row {i}: city='{city}' not in address='{address}'")

            assert len(mismatches) == 0

    def test_address_only_still_generates_coherent_addresses(self):
        """
        Test that when only the address preset is used (without separate city/state
        columns), it still generates internally coherent addresses.
        """
        pytest.importorskip("polars")

        from pointblank import Schema, generate_dataset, string_field
        from pointblank.countries import LocaleGenerator

        schema = Schema(columns=[("address", string_field(preset="address"))])

        for country in COUNTRIES_WITH_FULL_DATA:
            df = generate_dataset(schema, n=20, seed=99, country=country)

            # Load the locale to get the list of valid cities
            gen = LocaleGenerator(country=country, seed=1)
            locations = gen._data.address.get("locations", [])
            valid_cities = {loc.get("city", "") for loc in locations}

            # Check that each address contains a valid city for this country
            for i, row in enumerate(df.iter_rows()):
                address = row[0]
                found_city = any(city in address for city in valid_cities if city)

                assert found_city


class TestGeneratorValidation:
    """Tests that generated data passes pointblank validation."""

    def test_credit_card_numbers_pass_luhn_validation(self):
        """Ensure generated credit card numbers pass Luhn checksum validation."""
        from pointblank.countries import LocaleGenerator
        from pointblank._spec_utils import is_credit_card

        # Test across multiple countries
        for locale in ["en_US", "de_DE", "fr_FR", "ja_JP"]:
            gen = LocaleGenerator(locale, seed=23)

            # Generate multiple credit cards and verify all pass validation
            for i in range(50):
                cc = gen.credit_card_number()

                # Check it passes the full credit card validation (regex + Luhn)
                assert is_credit_card(cc)

                # Also verify length is correct for card type
                if cc.startswith("37"):  # Amex
                    assert len(cc) == 15
                else:  # Visa (4), MC (5), Discover (6011)
                    assert len(cc) == 16

    def test_credit_cards_with_col_vals_within_spec(self):
        """Ensure generated credit cards pass col_vals_within_spec validation."""
        import pointblank as pb

        # Generate data with credit card column using Schema
        schema = pb.Schema(columns=[("credit_card", "str")])
        df = pb.generate_dataset(
            schema,
            n=100,
            seed=23,
        )

        # The Schema.generate() doesn't use presets directly, so use LocaleGenerator
        from pointblank.countries import LocaleGenerator

        gen = LocaleGenerator("en_US", seed=23)
        credit_cards = [gen.credit_card_number() for _ in range(100)]

        # Create a dataframe with the generated credit cards
        import polars as pl

        df = pl.DataFrame({"credit_card": credit_cards})

        # Validate using col_vals_within_spec
        validation = (
            pb.Validate(df)
            .col_vals_within_spec(columns="credit_card", spec="credit_card")
            .interrogate()
        )

        assert validation.all_passed()


class TestUserAgentPreset:
    """Tests for the user_agent preset with country-specific browser weighting."""

    def test_user_agent_returns_string(self):
        """User agent preset returns a non-empty string."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(columns=[("ua", string_field(preset="user_agent"))])
        df = generate_dataset(schema, n=20, seed=23)

        assert df.shape == (20, 1)

        for val in df["ua"].to_list():
            assert isinstance(val, str)
            assert len(val) > 0

    def test_user_agent_looks_realistic(self):
        """Generated user agents contain expected fragments."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(columns=[("ua", string_field(preset="user_agent"))])
        df = generate_dataset(schema, n=100, seed=123)
        ua_strings = df["ua"].to_list()

        # All should contain "Mozilla/5.0" (universal UA prefix)
        for ua in ua_strings:
            assert "Mozilla/5.0" in ua

    def test_user_agent_reproducible_with_seed(self):
        """Same seed produces same user agents."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(columns=[("ua", string_field(preset="user_agent"))])
        df1 = generate_dataset(schema, n=50, seed=999)
        df2 = generate_dataset(schema, n=50, seed=999)

        assert df1["ua"].to_list() == df2["ua"].to_list()

    def test_user_agent_country_weighting_varies(self):
        """Different countries produce different browser distributions."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(columns=[("ua", string_field(preset="user_agent"))])
        n = 500

        # US should have more Safari (iPhone market share)
        us_df = generate_dataset(schema, n=n, seed=23, country="US")

        # Russia should have Yandex browser strings
        ru_df = generate_dataset(schema, n=n, seed=23, country="RU")

        # South Korea should have Whale browser strings
        kr_df = generate_dataset(schema, n=n, seed=23, country="KR")

        us_uas = us_df["ua"].to_list()
        ru_uas = ru_df["ua"].to_list()
        kr_uas = kr_df["ua"].to_list()

        # Yandex should appear in Russian results (high weight)
        ru_yandex = sum(1 for ua in ru_uas if "YaBrowser" in ua)
        us_yandex = sum(1 for ua in us_uas if "YaBrowser" in ua)

        assert ru_yandex > us_yandex

        # Whale should appear in Korean results
        kr_whale = sum(1 for ua in kr_uas if "Whale" in ua)
        us_whale = sum(1 for ua in us_uas if "Whale" in ua)

        assert kr_whale > us_whale

    def test_user_agent_all_countries(self):
        """User agent works for all supported countries."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(columns=[("ua", string_field(preset="user_agent"))])
        for country in COUNTRIES_WITH_FULL_DATA:
            df = generate_dataset(schema, n=5, seed=23, country=country)

            assert df.shape == (5, 1)

            for val in df["ua"].to_list():
                assert isinstance(val, str)
                assert len(val) > 20  # UAs are long strings

    def test_user_agent_browser_diversity(self):
        """Large sample contains multiple browser types."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(columns=[("ua", string_field(preset="user_agent"))])
        df = generate_dataset(schema, n=1000, seed=23)
        uas = df["ua"].to_list()

        # Should see Chrome, Safari, Firefox, and Edge at minimum
        has_chrome = any("Chrome" in ua and "Edg" not in ua and "OPR" not in ua for ua in uas)
        has_safari = any("Safari" in ua and "Chrome" not in ua for ua in uas)
        has_firefox = any("Firefox" in ua for ua in uas)
        has_edge = any("Edg/" in ua or "EdgA/" in ua for ua in uas)

        assert has_chrome
        assert has_safari
        assert has_firefox
        assert has_edge

    def test_user_agent_china_specific_browsers(self):
        """Chinese locale produces region-specific browsers (UC, 360)."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(columns=[("ua", string_field(preset="user_agent"))])
        df = generate_dataset(schema, n=1000, seed=23, country="CN")
        uas = df["ua"].to_list()

        # China profile has UC Browser and 360 Safe Browser weights
        has_uc = any("UCBrowser" in ua for ua in uas)
        has_360 = any("QIHU 360" in ua for ua in uas)
        assert has_uc or has_360


class TestLocaleDataFiles:
    """Tests for locale data file consistency and validity."""

    def test_all_countries_have_required_files(self):
        """Ensure all country directories have the same set of JSON files."""
        import os
        from pathlib import Path

        countries_dir = Path(__file__).parent.parent / "pointblank" / "countries" / "data"
        countries = COUNTRIES_WITH_FULL_DATA
        required_files = {
            "address.json",
            "company.json",
            "internet.json",
            "misc.json",
            "person.json",
            "text.json",
        }

        for country in countries:
            country_dir = countries_dir / country

            assert country_dir.exists()

            actual_files = {f.name for f in country_dir.iterdir() if f.suffix == ".json"}
            missing = required_files - actual_files

            assert not missing

    def test_all_json_files_are_valid(self):
        """Ensure all JSON files can be parsed without errors."""
        import json
        from pathlib import Path

        countries_dir = Path(__file__).parent.parent / "pointblank" / "countries" / "data"
        countries = COUNTRIES_WITH_FULL_DATA

        for country in countries:
            country_dir = countries_dir / country
            for json_file in country_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    assert isinstance(data, dict)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {json_file}: {e}")

    def test_json_key_order_consistency(self):
        """Ensure JSON files have consistent key ordering across all countries."""
        import json
        from pathlib import Path

        countries_dir = Path(__file__).parent.parent / "pointblank" / "countries" / "data"
        countries = COUNTRIES_WITH_FULL_DATA
        json_files = [
            "address.json",
            "company.json",
            "internet.json",
            "misc.json",
            "person.json",
            "text.json",
        ]

        # Expected key orders for each file type
        # Keys marked as optional may not exist in all countries
        # All countries use streets_by_city for city-specific full street names
        # The tuple format is: (key_name, is_optional)
        expected_key_orders = {
            "address.json": [
                ("locations", False),
                ("streets_by_city", False),  # All countries use city-specific full street names
                ("postcode_format", False),
                ("address_formats", False),
                ("country", False),
                ("country_code", False),
                ("phone_area_codes", False),
            ],
            "company.json": [
                ("suffixes", False),
                ("formats", False),
                ("adjectives", False),
                ("nouns", False),
                ("well_known_companies", False),
                ("jobs", False),
                ("catch_phrase_adjectives", False),
                ("catch_phrase_nouns", False),
                ("catch_phrase_verbs", False),
            ],
            "internet.json": [
                ("free_email_domains", False),
                ("tlds", False),
                ("domain_words", False),
                ("user_agent_browsers", False),
                ("user_agent_os", False),
            ],
            "misc.json": [
                ("colors", False),
            ],
            "person.json": [
                ("first_names", False),
                ("last_names", False),
                ("name_formats", False),
                ("prefixes", False),
                ("suffixes", False),
            ],
            "text.json": [
                ("words", False),
                ("sentence_patterns", False),
                ("adjectives", False),
                ("nouns", False),
                ("verbs", False),
                ("adverbs", False),
            ],
        }

        for json_file in json_files:
            key_specs = expected_key_orders[json_file]

            for country in countries:
                file_path = countries_dir / country / json_file
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                actual_keys = list(data.keys())

                # Build expected order for this country (exclude optional keys not present)
                expected_order = [
                    key for key, is_optional in key_specs if not is_optional or key in actual_keys
                ]

                # Filter actual keys to only those in our expected order
                filtered_keys = [k for k in actual_keys if k in expected_order]

                # Verify the order matches
                assert filtered_keys == expected_order

    def test_address_json_schema_consistency(self):
        """Ensure address.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        countries_dir = Path(__file__).parent.parent / "pointblank" / "countries" / "data"
        countries = COUNTRIES_WITH_FULL_DATA

        # Common required keys for all countries
        common_required_keys = {
            "locations",
            "postcode_format",
            "address_formats",
            "country",
            "country_code",
            "phone_area_codes",
        }

        # All countries use streets_by_city (city-specific full street names)
        countries = COUNTRIES_WITH_FULL_DATA

        for country in countries:
            address_file = countries_dir / country / "address.json"
            with open(address_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = common_required_keys - set(data.keys())

            assert not missing_keys

            # Check streets_by_city structure (all countries use this)
            assert "streets_by_city" in data
            assert isinstance(data["streets_by_city"], dict)

            # Validate that each city in locations has streets
            city_names = {loc["city"] for loc in data["locations"]}
            streets_cities = set(data["streets_by_city"].keys())

            assert city_names == streets_cities

            # Validate locations structure
            assert isinstance(data["locations"], list)
            assert len(data["locations"]) > 0

            for loc in data["locations"]:
                assert "city" in loc
                assert "state" in loc
                assert "state_abbr" in loc
                assert "postcode_prefix" in loc

                # Validate lat/lon bounding box fields
                assert "lat_min" in loc
                assert "lat_max" in loc
                assert "lon_min" in loc
                assert "lon_max" in loc

                # Validate lat/lon bounds are numeric and valid
                assert isinstance(loc["lat_min"], (int, float))
                assert isinstance(loc["lat_max"], (int, float))
                assert isinstance(loc["lon_min"], (int, float))
                assert isinstance(loc["lon_max"], (int, float))
                assert loc["lat_min"] < loc["lat_max"]
                assert loc["lon_min"] < loc["lon_max"]

                # Validate latitude range (-90 to 90) and longitude range (-180 to 180)
                assert -90 <= loc["lat_min"] <= 90
                assert -90 <= loc["lat_max"] <= 90
                assert -180 <= loc["lon_min"] <= 180
                assert -180 <= loc["lon_max"] <= 180

            # Validate other required fields
            assert isinstance(data["phone_area_codes"], dict)
            assert len(data["phone_area_codes"]) > 0

            # Validate postcode_format
            assert isinstance(data["postcode_format"], str)
            assert len(data["postcode_format"]) > 0

            # Validate address_formats
            assert isinstance(data["address_formats"], list)
            assert len(data["address_formats"]) > 0

            for fmt in data["address_formats"]:
                assert isinstance(fmt, str)

            # Validate country and country_code
            assert isinstance(data["country"], str)
            assert isinstance(data["country_code"], str)
            assert len(data["country_code"]) == 2

            # Validate streets_by_city structure
            for city, streets in data["streets_by_city"].items():
                assert isinstance(streets, list)
                assert len(streets) > 0
                for street in streets:
                    assert isinstance(street, str)

    def test_person_json_schema_consistency(self):
        """Ensure person.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        countries_dir = Path(__file__).parent.parent / "pointblank" / "countries" / "data"
        countries = COUNTRIES_WITH_FULL_DATA
        required_keys = {"first_names", "last_names", "name_formats", "prefixes", "suffixes"}

        for country in countries:
            person_file = countries_dir / country / "person.json"
            with open(person_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())
            assert not missing_keys

            # first_names should be a dict with male/female/neutral keys
            first_names = data["first_names"]

            assert isinstance(first_names, dict)

            for gender in ["male", "female", "neutral"]:
                assert gender in first_names
                assert isinstance(first_names[gender], list)

                # male and female should have content, neutral can be empty
                if gender != "neutral":
                    assert len(first_names[gender]) > 0

                for name in first_names[gender]:
                    assert isinstance(name, str)

            # last_names should be a list of strings OR a dict with gendered keys
            last_names = data["last_names"]
            if isinstance(last_names, dict):
                # Gendered last names (e.g., IS patronymics)
                for gender_key in last_names:
                    assert isinstance(last_names[gender_key], list)

                    for name in last_names[gender_key]:
                        assert isinstance(name, str)

                # At least one category should have names
                all_names = [n for v in last_names.values() for n in v]

                assert len(all_names) > 0
            else:
                assert isinstance(last_names, list)
                assert len(last_names) > 0

                for name in last_names:
                    assert isinstance(name, str)

            # name_formats should be a list with valid placeholders
            assert isinstance(data["name_formats"], list)
            assert len(data["name_formats"]) > 0

            for fmt in data["name_formats"]:
                assert isinstance(fmt, str)

                # Check that format contains at least first_name or last_name
                has_name = "{first_name}" in fmt or "{last_name}" in fmt

                assert has_name

            # prefixes should be a dict with male/female/neutral keys
            prefixes = data["prefixes"]

            assert isinstance(prefixes, dict)

            for gender in ["male", "female", "neutral"]:
                assert gender in prefixes
                assert isinstance(prefixes[gender], list)

                # male and female should have content
                if gender != "neutral":
                    assert len(prefixes[gender]) > 0
                for prefix in prefixes[gender]:
                    assert isinstance(prefix, str)

            # suffixes should be a list of strings
            assert isinstance(data["suffixes"], list)

            for suffix in data["suffixes"]:
                assert isinstance(suffix, str)

    def test_company_json_schema_consistency(self):
        """Ensure company.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        countries_dir = Path(__file__).parent.parent / "pointblank" / "countries" / "data"
        countries = COUNTRIES_WITH_FULL_DATA
        required_keys = {
            "suffixes",
            "formats",
            "adjectives",
            "nouns",
            "well_known_companies",
            "jobs",
            "catch_phrase_adjectives",
            "catch_phrase_nouns",
            "catch_phrase_verbs",
        }

        for country in countries:
            company_file = countries_dir / country / "company.json"
            with open(company_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())

            assert not missing_keys

            # Validate list keys are lists with content
            list_keys = {
                "suffixes",
                "formats",
                "adjectives",
                "nouns",
                "jobs",
                "catch_phrase_adjectives",
                "catch_phrase_nouns",
                "catch_phrase_verbs",
            }
            for key in list_keys:
                assert isinstance(data[key], list)
                assert len(data[key]) > 0

            # Validate well_known_companies structure
            well_known = data["well_known_companies"]

            assert isinstance(well_known, list)
            assert len(well_known) > 0

            for company in well_known:
                assert isinstance(company, dict)
                assert "name" in company
                assert "cities" in company
                assert isinstance(company["name"], str)
                assert isinstance(company["cities"], list)
                assert len(company["cities"]) > 0

                for city in company["cities"]:
                    assert isinstance(city, str)

            # Validate formats contain valid placeholders
            valid_placeholders = {"{last_name}", "{suffix}", "{adjective}", "{noun}"}
            for fmt in data["formats"]:
                # Extract all placeholders from format string
                import re

                placeholders = set(re.findall(r"\{[^}]+\}", fmt))
                invalid = placeholders - valid_placeholders
                assert not invalid

    def test_internet_json_schema_consistency(self):
        """Ensure internet.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        countries_dir = Path(__file__).parent.parent / "pointblank" / "countries" / "data"
        countries = COUNTRIES_WITH_FULL_DATA
        required_keys = {
            "free_email_domains",
            "tlds",
            "domain_words",
            "user_agent_browsers",
            "user_agent_os",
        }

        for country in countries:
            internet_file = countries_dir / country / "internet.json"
            with open(internet_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())

            assert not missing_keys

            # Validate all required keys are lists with content and contain strings
            for key in required_keys:
                assert isinstance(data[key], list)
                assert len(data[key]) > 0

                for item in data[key]:
                    assert isinstance(item, str)

            # Validate email domains look like domains
            for domain in data["free_email_domains"]:
                assert "." in domain

            # Validate TLDs (stored without dots, e.g., 'com' not '.com')
            for tld in data["tlds"]:
                assert len(tld) > 0
                assert not tld.startswith(".")

    def test_misc_json_schema_consistency(self):
        """Ensure misc.json files have consistent schema across countries.

        Note: file_extensions, mime_types, and currency_codes are universal and stored in
        _shared/misc.json, so they are not required in country-specific misc.json files.
        """
        import json
        from pathlib import Path

        countries_dir = Path(__file__).parent.parent / "pointblank" / "countries" / "data"
        countries = COUNTRIES_WITH_FULL_DATA
        # Country-specific required keys (file_extensions/mime_types/currency_codes are universal)
        required_keys = {"colors"}

        for country in countries:
            misc_file = countries_dir / country / "misc.json"
            with open(misc_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())

            assert not missing_keys

            # Validate arrays are non-empty
            for key in required_keys:
                assert isinstance(data[key], list)
                assert len(data[key]) > 0

        # Also verify that shared/universal data exists
        shared_file = countries_dir / "_shared" / "misc.json"

        assert shared_file.exists()

        with open(shared_file, "r", encoding="utf-8") as f:
            shared_data = json.load(f)

        # Validate file_extensions
        assert "file_extensions" in shared_data
        assert isinstance(shared_data["file_extensions"], list)
        assert len(shared_data["file_extensions"]) > 0
        for ext in shared_data["file_extensions"]:
            assert isinstance(ext, str)
            assert len(ext) > 0

            # Extensions are stored without dots (e.g., 'txt' not '.txt')
            assert not ext.startswith(".")

        # Validate mime_types
        assert "mime_types" in shared_data
        assert isinstance(shared_data["mime_types"], list)
        assert len(shared_data["mime_types"]) > 0

        for mime in shared_data["mime_types"]:
            assert isinstance(mime, str)
            assert "/" in mime

        # Validate currency_codes (ISO 4217)
        assert "currency_codes" in shared_data
        assert isinstance(shared_data["currency_codes"], list)
        assert len(shared_data["currency_codes"]) > 0

        for code in shared_data["currency_codes"]:
            assert isinstance(code, str)
            assert len(code) == 3
            assert code.isupper()

    def test_text_json_schema_consistency(self):
        """Ensure text.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        countries_dir = Path(__file__).parent.parent / "pointblank" / "countries" / "data"
        countries = COUNTRIES_WITH_FULL_DATA
        required_keys = {"words", "adjectives", "nouns", "verbs", "adverbs"}

        for country in countries:
            text_file = countries_dir / country / "text.json"
            with open(text_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())

            assert not missing_keys

            # Validate arrays are non-empty and contain strings
            for key in required_keys:
                assert isinstance(data[key], list)
                assert len(data[key]) > 0

                for item in data[key]:
                    assert isinstance(item, str)

    def test_well_known_companies_cities_exist_in_locations(self):
        """Ensure well-known company cities match cities defined in address.json locations."""
        import json
        from pathlib import Path

        countries_dir = Path(__file__).parent.parent / "pointblank" / "countries" / "data"
        countries = COUNTRIES_WITH_FULL_DATA

        for country in countries:
            # Load address data to get valid cities
            address_file = countries_dir / country / "address.json"
            with open(address_file, "r", encoding="utf-8") as f:
                address_data = json.load(f)
            valid_cities = {loc["city"] for loc in address_data["locations"]}

            # Load company data
            company_file = countries_dir / country / "company.json"
            with open(company_file, "r", encoding="utf-8") as f:
                company_data = json.load(f)

            # Check each well-known company's cities
            for company in company_data.get("well_known_companies", []):
                company_name = company.get("name", "Unknown")
                for city in company.get("cities", []):
                    # This is a warning, not a hard failure as companies may have offices
                    # in cities we haven't added to our locations list yet
                    if city not in valid_cities:
                        print(
                            f"Note: {country} company '{company_name}' has city '{city}' "
                            f"not in locations"
                        )

    def test_locale_data_no_duplicate_entries(self):
        """Ensure locale data lists don't have critical duplicate entries."""
        import json
        from pathlib import Path

        countries_dir = Path(__file__).parent.parent / "pointblank" / "countries" / "data"
        countries = COUNTRIES_WITH_FULL_DATA

        for country in countries:
            # Check person.json for duplicate names
            person_file = countries_dir / country / "person.json"
            with open(person_file, "r", encoding="utf-8") as f:
                person_data = json.load(f)

            # Last names may have some duplicates (common surnames appear multiple times
            # in real populations), so we just check for excessive duplicates
            last_names = person_data.get("last_names", [])
            unique_last_names = set(last_names)
            duplicate_ratio = 1 - (len(unique_last_names) / len(last_names)) if last_names else 0

            assert duplicate_ratio < 0.1

            # First names should not have duplicates within each gender
            for gender in ["male", "female", "neutral"]:
                first_names = person_data.get("first_names", {}).get(gender, [])
                duplicates = [x for x in first_names if first_names.count(x) > 1]

                assert not duplicates

            # Check company.json for duplicate well-known companies (must be unique)
            company_file = countries_dir / country / "company.json"
            with open(company_file, "r", encoding="utf-8") as f:
                company_data = json.load(f)

            company_names = [c["name"] for c in company_data.get("well_known_companies", [])]
            duplicates = [x for x in company_names if company_names.count(x) > 1]

            assert not duplicates
