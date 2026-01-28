import pytest
from datetime import date, datetime

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

        df1 = generate_dataset(schema, n=10, seed=42)
        df2 = generate_dataset(schema, n=10, seed=42)

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
        df_alpha2 = generate_dataset(schema, n=10, seed=42, country="US")
        df_alpha3 = generate_dataset(schema, n=10, seed=42, country="USA")

        assert df_alpha2.equals(df_alpha3)

    def test_legacy_locale_format_supported(self):
        """Test that legacy locale formats are still accepted for backwards compatibility."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(name=string_field(preset="name"))

        # Legacy locale formats should work
        df_legacy1 = generate_dataset(schema, n=10, seed=42, country="en-US")
        df_legacy2 = generate_dataset(schema, n=10, seed=42, country="en_US")
        df_iso = generate_dataset(schema, n=10, seed=42, country="US")

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


class TestLocaleDataFiles:
    """Tests for locale data file consistency and validity."""

    def test_all_countries_have_required_files(self):
        """Ensure all country directories have the same set of JSON files."""
        import os
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = ["US", "DE", "FR", "JP"]
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
        countries = ["US", "DE", "FR", "JP"]

        for country in countries:
            country_dir = locales_dir / country
            for json_file in country_dir.glob("*.json"):
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    assert isinstance(data, dict), f"{json_file} should contain a JSON object"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON in {json_file}: {e}")

    def test_address_json_schema_consistency(self):
        """Ensure address.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = ["US", "DE", "FR", "JP"]
        required_keys = {
            "locations",
            "street_names",
            "postcode_format",
            "address_formats",
            "country",
            "country_code",
            "phone_area_codes",
        }

        for country in countries:
            address_file = locales_dir / country / "address.json"
            with open(address_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())
            assert not missing_keys, f"address.json for {country} is missing keys: {missing_keys}"

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

            # Validate other required fields
            assert isinstance(data["street_names"], list), (
                f"{country}: street_names should be a list"
            )
            assert len(data["street_names"]) > 0, f"{country}: street_names should not be empty"
            assert isinstance(data["phone_area_codes"], dict), (
                f"{country}: phone_area_codes should be a dict"
            )

    def test_person_json_schema_consistency(self):
        """Ensure person.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = ["US", "DE", "FR", "JP"]
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

            # last_names should be a list
            assert isinstance(data["last_names"], list), f"{country}: last_names should be a list"
            assert len(data["last_names"]) > 0, f"{country}: last_names should not be empty"

            # name_formats should be a list
            assert isinstance(data["name_formats"], list), (
                f"{country}: name_formats should be a list"
            )
            assert len(data["name_formats"]) > 0, f"{country}: name_formats should not be empty"

            # prefixes should be a dict with male/female/neutral keys
            prefixes = data["prefixes"]
            assert isinstance(prefixes, dict), f"{country}: prefixes should be a dict"
            for gender in ["male", "female", "neutral"]:
                assert gender in prefixes, f"{country}: prefixes should have '{gender}' key"
                assert isinstance(prefixes[gender], list), (
                    f"{country}: prefixes[{gender}] should be a list"
                )

            # suffixes should be a list
            assert isinstance(data["suffixes"], list), f"{country}: suffixes should be a list"

    def test_company_json_schema_consistency(self):
        """Ensure company.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = ["US", "DE", "FR", "JP"]
        required_keys = {
            "suffixes",
            "formats",
            "adjectives",
            "nouns",
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

            # Validate all required keys are lists with content
            for key in required_keys:
                assert isinstance(data[key], list), f"{country}: {key} should be a list"
                assert len(data[key]) > 0, f"{country}: {key} should not be empty"

    def test_internet_json_schema_consistency(self):
        """Ensure internet.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = ["US", "DE", "FR", "JP"]
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

            # Validate all required keys are lists with content
            for key in required_keys:
                assert isinstance(data[key], list), f"{country}: {key} should be a list"
                assert len(data[key]) > 0, f"{country}: {key} should not be empty"

    def test_misc_json_schema_consistency(self):
        """Ensure misc.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = ["US", "DE", "FR", "JP"]
        required_keys = {"colors", "file_extensions", "mime_types", "currency_codes"}

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

    def test_text_json_schema_consistency(self):
        """Ensure text.json files have consistent schema across countries."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = ["US", "DE", "FR", "JP"]
        required_keys = {"words", "adjectives", "nouns", "verbs", "adverbs"}

        for country in countries:
            text_file = locales_dir / country / "text.json"
            with open(text_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            missing_keys = required_keys - set(data.keys())
            assert not missing_keys, f"text.json for {country} is missing keys: {missing_keys}"

            # Validate arrays are non-empty
            for key in required_keys:
                assert isinstance(data[key], list), f"{country}: {key} should be a list"
                assert len(data[key]) > 0, f"{country}: {key} should not be empty"

    def test_locale_data_statistics(self):
        """Print statistics about locale data for review."""
        import json
        from pathlib import Path

        locales_dir = Path(__file__).parent.parent / "pointblank" / "locales" / "data"
        countries = ["US", "DE", "FR", "JP"]

        print("\n=== Locale Data Statistics ===")
        for country in countries:
            country_dir = locales_dir / country

            # Address stats
            with open(country_dir / "address.json", "r", encoding="utf-8") as f:
                addr = json.load(f)
            locations = len(addr.get("locations", []))
            streets = len(addr.get("street_names", []))
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
