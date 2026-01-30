import pytest
from datetime import date, datetime

from pointblank.locales import COUNTRIES_WITH_FULL_DATA
from pointblank.field import (
    int_field,
    float_field,
    string_field,
    bool_field,
    date_field,
    datetime_field,
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
            assert validation.all_passed(), (
                f"Validation failed for country {country}: {validation.get_sundered_data()}"
            )

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

            assert validation.all_passed(), f"Email validation failed for country {country}"

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

            assert validation.all_passed(), f"Personal presets failed for country {country}"

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

            assert validation.all_passed(), f"Business presets failed for country {country}"

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

            assert validation.all_passed(), f"Internet presets failed for country {country}"

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

        assert validation.all_passed(), "Combined schema validation failed"

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
                ("ssn", string_field(preset="ssn")),
                ("license_plate", string_field(preset="license_plate")),
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
            assert df.shape[1] == 57

            # Verify no unexpected errors occurred (all values generated successfully)
            # Check a few key columns are not empty strings
            for col in ["name", "email", "city", "company", "address"]:
                values = df[col].to_list()

                assert all(v is not None and len(str(v)) > 0 for v in values), (
                    f"Column {col} has empty values for {country}"
                )


class TestGeneratorValidation:
    """Tests that generated data passes pointblank validation."""

    def test_credit_card_numbers_pass_luhn_validation(self):
        """Ensure generated credit card numbers pass Luhn checksum validation."""
        from pointblank.locales import LocaleGenerator
        from pointblank._spec_utils import is_credit_card

        # Test across multiple locales
        for locale in ["en_US", "de_DE", "fr_FR", "ja_JP"]:
            gen = LocaleGenerator(locale, seed=23)

            # Generate multiple credit cards and verify all pass validation
            for i in range(50):
                cc = gen.credit_card_number()

                # Check it passes the full credit card validation (regex + Luhn)
                assert is_credit_card(cc), (
                    f"Credit card '{cc}' generated for {locale} failed validation"
                )

                # Also verify length is correct for card type
                if cc.startswith("37"):  # Amex
                    assert len(cc) == 15, f"Amex card should be 15 digits: {cc}"
                else:  # Visa (4), MC (5), Discover (6011)
                    assert len(cc) == 16, f"Non-Amex card should be 16 digits: {cc}"

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
        from pointblank.locales import LocaleGenerator

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

        assert validation.all_passed(), "Generated credit cards failed col_vals_within_spec"


class TestLocaleDataFiles:
    """Tests for locale data file consistency and validity."""

    def test_all_countries_have_required_files(self):
        """Ensure all country directories have the same set of JSON files."""
        import os
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
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
            country_dir = locales_dir / country
            assert country_dir.exists(), f"Country directory {country} does not exist"

            actual_files = {f.name for f in country_dir.iterdir() if f.suffix == ".json"}
            missing = required_files - actual_files
            extra = actual_files - required_files

            assert not missing, f"Country {country} is missing files: {missing}"
            # Extra files are allowed but we note them
            if extra:
                print(f"Note: Country {country} has extra files: {extra}")

    def test_all_json_files_are_valid(self):
        """Ensure all JSON files can be parsed without errors."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = COUNTRIES_WITH_FULL_DATA

        for country in countries:
            country_dir = locales_dir / country
            for json_file in country_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    assert isinstance(data, dict), f"{json_file} should contain a JSON object"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {json_file}: {e}")

    def test_json_key_order_consistency(self):
        """Ensure JSON files have consistent key ordering across all countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
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
                file_path = locales_dir / country / json_file
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
                assert filtered_keys == expected_order, (
                    f"{country}/{json_file}: Key order mismatch. "
                    f"Expected {expected_order}, got {filtered_keys}"
                )

    def test_address_json_schema_consistency(self):
        """Ensure address.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
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
            address_file = locales_dir / country / "address.json"
            with open(address_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = common_required_keys - set(data.keys())
            assert not missing_keys, f"address.json for {country} is missing keys: {missing_keys}"

            # Check streets_by_city structure (all countries use this)
            assert "streets_by_city" in data, f"{country}: should have 'streets_by_city'"
            assert isinstance(data["streets_by_city"], dict), (
                f"{country}: streets_by_city should be a dict"
            )
            # Validate that each city in locations has streets
            city_names = {loc["city"] for loc in data["locations"]}
            streets_cities = set(data["streets_by_city"].keys())
            assert city_names == streets_cities, (
                f"{country}: streets_by_city cities should match location cities. "
                f"Missing: {city_names - streets_cities}, Extra: {streets_cities - city_names}"
            )

            # Validate locations structure
            assert isinstance(data["locations"], list), f"{country}: locations should be a list"
            assert len(data["locations"]) > 0, f"{country}: locations should not be empty"

            for loc in data["locations"]:
                assert "city" in loc, f"{country}: each location should have 'city'"
                assert "state" in loc, f"{country}: each location should have 'state'"
                assert "state_abbr" in loc, f"{country}: each location should have 'state_abbr'"
                assert "postcode_prefix" in loc, (
                    f"{country}: each location should have 'postcode_prefix'"
                )
                # Validate lat/lon bounding box fields
                assert "lat_min" in loc, f"{country}: each location should have 'lat_min'"
                assert "lat_max" in loc, f"{country}: each location should have 'lat_max'"
                assert "lon_min" in loc, f"{country}: each location should have 'lon_min'"
                assert "lon_max" in loc, f"{country}: each location should have 'lon_max'"
                # Validate lat/lon bounds are numeric and valid
                assert isinstance(loc["lat_min"], (int, float)), (
                    f"{country}: lat_min should be numeric"
                )
                assert isinstance(loc["lat_max"], (int, float)), (
                    f"{country}: lat_max should be numeric"
                )
                assert isinstance(loc["lon_min"], (int, float)), (
                    f"{country}: lon_min should be numeric"
                )
                assert isinstance(loc["lon_max"], (int, float)), (
                    f"{country}: lon_max should be numeric"
                )
                assert loc["lat_min"] < loc["lat_max"], (
                    f"{country}/{loc['city']}: lat_min should be less than lat_max"
                )
                assert loc["lon_min"] < loc["lon_max"], (
                    f"{country}/{loc['city']}: lon_min should be less than lon_max"
                )
                # Validate latitude range (-90 to 90) and longitude range (-180 to 180)
                assert -90 <= loc["lat_min"] <= 90, f"{country}/{loc['city']}: lat_min out of range"
                assert -90 <= loc["lat_max"] <= 90, f"{country}/{loc['city']}: lat_max out of range"
                assert -180 <= loc["lon_min"] <= 180, (
                    f"{country}/{loc['city']}: lon_min out of range"
                )
                assert -180 <= loc["lon_max"] <= 180, (
                    f"{country}/{loc['city']}: lon_max out of range"
                )

            # Validate other required fields
            assert isinstance(data["phone_area_codes"], dict), (
                f"{country}: phone_area_codes should be a dict"
            )
            assert len(data["phone_area_codes"]) > 0, (
                f"{country}: phone_area_codes should not be empty"
            )

            # Validate postcode_format
            assert isinstance(data["postcode_format"], str), (
                f"{country}: postcode_format should be a string"
            )
            assert len(data["postcode_format"]) > 0, (
                f"{country}: postcode_format should not be empty"
            )

            # Validate address_formats
            assert isinstance(data["address_formats"], list), (
                f"{country}: address_formats should be a list"
            )
            assert len(data["address_formats"]) > 0, (
                f"{country}: address_formats should not be empty"
            )
            for fmt in data["address_formats"]:
                assert isinstance(fmt, str), f"{country}: each address_format should be a string"

            # Validate country and country_code
            assert isinstance(data["country"], str), f"{country}: country should be a string"
            assert isinstance(data["country_code"], str), (
                f"{country}: country_code should be a string"
            )
            assert len(data["country_code"]) == 2, (
                f"{country}: country_code should be 2 characters (ISO 3166-1 alpha-2)"
            )

            # Validate streets_by_city structure
            for city, streets in data["streets_by_city"].items():
                assert isinstance(streets, list), f"{country}/{city}: streets should be a list"
                assert len(streets) > 0, f"{country}/{city}: streets should not be empty"
                for street in streets:
                    assert isinstance(street, str), (
                        f"{country}/{city}: each street should be a string"
                    )

    def test_person_json_schema_consistency(self):
        """Ensure person.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = COUNTRIES_WITH_FULL_DATA
        required_keys = {"first_names", "last_names", "name_formats", "prefixes", "suffixes"}

        for country in countries:
            person_file = locales_dir / country / "person.json"
            with open(person_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())
            assert not missing_keys, f"person.json for {country} is missing keys: {missing_keys}"

            # first_names should be a dict with male/female/neutral keys
            first_names = data["first_names"]
            assert isinstance(first_names, dict), f"{country}: first_names should be a dict"
            for gender in ["male", "female", "neutral"]:
                assert gender in first_names, f"{country}: first_names should have '{gender}' key"
                assert isinstance(first_names[gender], list), (
                    f"{country}: first_names[{gender}] should be a list"
                )
                # male and female should have content, neutral can be empty
                if gender != "neutral":
                    assert len(first_names[gender]) > 0, (
                        f"{country}: first_names[{gender}] should not be empty"
                    )
                for name in first_names[gender]:
                    assert isinstance(name, str), (
                        f"{country}: each first name in {gender} should be a string"
                    )

            # last_names should be a list of strings
            assert isinstance(data["last_names"], list), f"{country}: last_names should be a list"
            assert len(data["last_names"]) > 0, f"{country}: last_names should not be empty"
            for name in data["last_names"]:
                assert isinstance(name, str), f"{country}: each last_name should be a string"

            # name_formats should be a list with valid placeholders
            assert isinstance(data["name_formats"], list), (
                f"{country}: name_formats should be a list"
            )
            assert len(data["name_formats"]) > 0, f"{country}: name_formats should not be empty"
            valid_name_placeholders = {"{first_name}", "{last_name}"}
            for fmt in data["name_formats"]:
                assert isinstance(fmt, str), f"{country}: each name_format should be a string"
                # Check that format contains at least first_name or last_name
                has_name = "{first_name}" in fmt or "{last_name}" in fmt
                assert has_name, f"{country}: name_format '{fmt}' should contain a name placeholder"

            # prefixes should be a dict with male/female/neutral keys
            prefixes = data["prefixes"]
            assert isinstance(prefixes, dict), f"{country}: prefixes should be a dict"
            for gender in ["male", "female", "neutral"]:
                assert gender in prefixes, f"{country}: prefixes should have '{gender}' key"
                assert isinstance(prefixes[gender], list), (
                    f"{country}: prefixes[{gender}] should be a list"
                )
                # male and female should have content
                if gender != "neutral":
                    assert len(prefixes[gender]) > 0, (
                        f"{country}: prefixes[{gender}] should not be empty"
                    )
                for prefix in prefixes[gender]:
                    assert isinstance(prefix, str), (
                        f"{country}: each prefix in {gender} should be a string"
                    )

            # suffixes should be a list of strings
            assert isinstance(data["suffixes"], list), f"{country}: suffixes should be a list"
            for suffix in data["suffixes"]:
                assert isinstance(suffix, str), f"{country}: each suffix should be a string"

    def test_company_json_schema_consistency(self):
        """Ensure company.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
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
            company_file = locales_dir / country / "company.json"
            with open(company_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())
            assert not missing_keys, f"company.json for {country} is missing keys: {missing_keys}"

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
                assert isinstance(data[key], list), f"{country}: {key} should be a list"
                assert len(data[key]) > 0, f"{country}: {key} should not be empty"

            # Validate well_known_companies structure
            well_known = data["well_known_companies"]
            assert isinstance(well_known, list), f"{country}: well_known_companies should be a list"
            assert len(well_known) > 0, f"{country}: well_known_companies should not be empty"

            for company in well_known:
                assert isinstance(company, dict), (
                    f"{country}: each well_known_company should be a dict"
                )
                assert "name" in company, f"{country}: each well_known_company should have 'name'"
                assert "cities" in company, (
                    f"{country}: each well_known_company should have 'cities'"
                )
                assert isinstance(company["name"], str), (
                    f"{country}: company name should be a string"
                )
                assert isinstance(company["cities"], list), (
                    f"{country}: company cities should be a list"
                )
                assert len(company["cities"]) > 0, (
                    f"{country}: company '{company['name']}' should have at least one city"
                )
                for city in company["cities"]:
                    assert isinstance(city, str), (
                        f"{country}: each city in '{company['name']}' should be a string"
                    )

            # Validate formats contain valid placeholders
            valid_placeholders = {"{last_name}", "{suffix}", "{adjective}", "{noun}"}
            for fmt in data["formats"]:
                # Extract all placeholders from format string
                import re

                placeholders = set(re.findall(r"\{[^}]+\}", fmt))
                invalid = placeholders - valid_placeholders
                assert not invalid, f"{country}: format '{fmt}' has invalid placeholders: {invalid}"

    def test_internet_json_schema_consistency(self):
        """Ensure internet.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = COUNTRIES_WITH_FULL_DATA
        required_keys = {
            "free_email_domains",
            "tlds",
            "domain_words",
            "user_agent_browsers",
            "user_agent_os",
        }

        for country in countries:
            internet_file = locales_dir / country / "internet.json"
            with open(internet_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())
            assert not missing_keys, f"internet.json for {country} is missing keys: {missing_keys}"

            # Validate all required keys are lists with content and contain strings
            for key in required_keys:
                assert isinstance(data[key], list), f"{country}: {key} should be a list"
                assert len(data[key]) > 0, f"{country}: {key} should not be empty"
                for item in data[key]:
                    assert isinstance(item, str), (
                        f"{country}: each item in {key} should be a string"
                    )

            # Validate email domains look like domains
            for domain in data["free_email_domains"]:
                assert "." in domain, f"{country}: email domain '{domain}' should contain a dot"

            # Validate TLDs (stored without dots, e.g., 'com' not '.com')
            for tld in data["tlds"]:
                assert len(tld) > 0, f"{country}: TLD should not be empty"
                assert not tld.startswith("."), (
                    f"{country}: TLD '{tld}' should not start with a dot"
                )

    def test_misc_json_schema_consistency(self):
        """Ensure misc.json files have consistent schema across countries.

        Note: file_extensions, mime_types, and currency_codes are universal and stored in
        _shared/misc.json, so they are not required in country-specific misc.json files.
        """
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = COUNTRIES_WITH_FULL_DATA
        # Country-specific required keys (file_extensions/mime_types/currency_codes are universal)
        required_keys = {"colors"}

        for country in countries:
            misc_file = locales_dir / country / "misc.json"
            with open(misc_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())
            assert not missing_keys, f"misc.json for {country} is missing keys: {missing_keys}"

            # Validate arrays are non-empty
            for key in required_keys:
                assert isinstance(data[key], list), f"{country}: {key} should be a list"
                assert len(data[key]) > 0, f"{country}: {key} should not be empty"

        # Also verify that shared/universal data exists
        shared_file = locales_dir / "_shared" / "misc.json"
        assert shared_file.exists(), "_shared/misc.json should exist with universal data"
        with open(shared_file, "r", encoding="utf-8") as f:
            shared_data = json.load(f)

        # Validate file_extensions
        assert "file_extensions" in shared_data, "_shared should have file_extensions"
        assert isinstance(shared_data["file_extensions"], list), "file_extensions should be a list"
        assert len(shared_data["file_extensions"]) > 0, "file_extensions should not be empty"
        for ext in shared_data["file_extensions"]:
            assert isinstance(ext, str), "each file_extension should be a string"
            assert len(ext) > 0, f"file_extension should not be empty"
            # Extensions are stored without dots (e.g., 'txt' not '.txt')
            assert not ext.startswith("."), f"file_extension '{ext}' should not start with a dot"

        # Validate mime_types
        assert "mime_types" in shared_data, "_shared should have mime_types"
        assert isinstance(shared_data["mime_types"], list), "mime_types should be a list"
        assert len(shared_data["mime_types"]) > 0, "mime_types should not be empty"
        for mime in shared_data["mime_types"]:
            assert isinstance(mime, str), "each mime_type should be a string"
            assert "/" in mime, f"mime_type '{mime}' should contain a slash"

        # Validate currency_codes (ISO 4217)
        assert "currency_codes" in shared_data, "_shared should have currency_codes"
        assert isinstance(shared_data["currency_codes"], list), "currency_codes should be a list"
        assert len(shared_data["currency_codes"]) > 0, "currency_codes should not be empty"
        for code in shared_data["currency_codes"]:
            assert isinstance(code, str), "each currency_code should be a string"
            assert len(code) == 3, f"currency_code '{code}' should be 3 characters (ISO 4217)"
            assert code.isupper(), f"currency_code '{code}' should be uppercase"

    def test_text_json_schema_consistency(self):
        """Ensure text.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = COUNTRIES_WITH_FULL_DATA
        required_keys = {"words", "adjectives", "nouns", "verbs", "adverbs"}

        for country in countries:
            text_file = locales_dir / country / "text.json"
            with open(text_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())
            assert not missing_keys, f"text.json for {country} is missing keys: {missing_keys}"

            # Validate arrays are non-empty and contain strings
            for key in required_keys:
                assert isinstance(data[key], list), f"{country}: {key} should be a list"
                assert len(data[key]) > 0, f"{country}: {key} should not be empty"
                for item in data[key]:
                    assert isinstance(item, str), (
                        f"{country}: each item in {key} should be a string"
                    )

    def test_well_known_companies_cities_exist_in_locations(self):
        """Ensure well-known company cities match cities defined in address.json locations."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = COUNTRIES_WITH_FULL_DATA

        for country in countries:
            # Load address data to get valid cities
            address_file = locales_dir / country / "address.json"
            with open(address_file, "r", encoding="utf-8") as f:
                address_data = json.load(f)
            valid_cities = {loc["city"] for loc in address_data["locations"]}

            # Load company data
            company_file = locales_dir / country / "company.json"
            with open(company_file, "r", encoding="utf-8") as f:
                company_data = json.load(f)

            # Check each well-known company's cities
            for company in company_data.get("well_known_companies", []):
                company_name = company.get("name", "Unknown")
                for city in company.get("cities", []):
                    # This is a warning, not a hard failure - companies may have offices
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

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = COUNTRIES_WITH_FULL_DATA

        for country in countries:
            # Check person.json for duplicate names
            person_file = locales_dir / country / "person.json"
            with open(person_file, "r", encoding="utf-8") as f:
                person_data = json.load(f)

            # Last names may have some duplicates (common surnames appear multiple times
            # in real populations), so we just check for excessive duplicates
            last_names = person_data.get("last_names", [])
            unique_last_names = set(last_names)
            duplicate_ratio = 1 - (len(unique_last_names) / len(last_names)) if last_names else 0
            assert duplicate_ratio < 0.1, (
                f"{country}: person.json has too many duplicate last_names "
                f"({duplicate_ratio:.1%} duplicates)"
            )

            # First names should not have duplicates within each gender
            for gender in ["male", "female", "neutral"]:
                first_names = person_data.get("first_names", {}).get(gender, [])
                duplicates = [x for x in first_names if first_names.count(x) > 1]
                assert not duplicates, (
                    f"{country}: person.json has duplicate {gender} first_names: {set(duplicates)}"
                )

            # Check company.json for duplicate well-known companies (must be unique)
            company_file = locales_dir / country / "company.json"
            with open(company_file, "r", encoding="utf-8") as f:
                company_data = json.load(f)

            company_names = [c["name"] for c in company_data.get("well_known_companies", [])]
            duplicates = [x for x in company_names if company_names.count(x) > 1]
            assert not duplicates, (
                f"{country}: company.json has duplicate well_known_companies: {set(duplicates)}"
            )

    def test_locale_data_statistics(self):
        """Print statistics about locale data for review."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = COUNTRIES_WITH_FULL_DATA

        print("\n=== Locale Data Statistics ===")
        for country in countries:
            country_dir = locales_dir / country

            # Address stats
            with open(country_dir / "address.json", "r", encoding="utf-8") as f:
                addr = json.load(f)
            locations = len(addr.get("locations", []))
            # Count streets from streets_by_city (sum of all city street lists)
            streets_by_city = addr.get("streets_by_city", {})
            streets = sum(len(v) for v in streets_by_city.values()) if streets_by_city else 0
            states = len(addr.get("phone_area_codes", {}))

            # Person stats
            with open(country_dir / "person.json", "r", encoding="utf-8") as f:
                person = json.load(f)
            first_names = len(person.get("first_names", []))
            last_names = len(person.get("last_names", []))

            # Text stats
            with open(country_dir / "text.json", "r", encoding="utf-8") as f:
                text = json.load(f)
            words = len(text.get("words", []))
            adjectives = len(text.get("adjectives", []))
            nouns = len(text.get("nouns", []))

            print(f"\n{country}:")
            print(f"  Address: {locations} locations, {streets} streets, {states} states/regions")
            print(f"  Person: {first_names} first names, {last_names} last names")
            print(f"  Text: {words} words, {adjectives} adjectives, {nouns} nouns")
