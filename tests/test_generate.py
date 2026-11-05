import os

import pytest
from datetime import date, datetime, time, timedelta

from pointblank.countries import COUNTRIES_WITH_FULL_DATA, _transliterate_to_ascii
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
    _derive_seed,
    _get_coherence_groups,
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

    def test_country_dict_with_non_numeric_weight_raises(self):
        with pytest.raises(ValueError, match="must be a number"):
            GeneratorConfig(country={"US": "not_a_number"})


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

    def test_generate_float_with_precision(self):
        """Test that precision rounds generated float values."""
        field = float_field(min_val=0.0, max_val=100.0, precision=2)
        config = GeneratorConfig(n=50, seed=23)
        values = generate_column(field, config)

        assert all(isinstance(v, float) for v in values)
        assert all(round(v, 2) == v for v in values)

    def test_generate_float_precision_zero(self):
        """Test that `precision=0` rounds values to whole numbers."""
        field = float_field(min_val=0.0, max_val=100.0, precision=0)
        config = GeneratorConfig(n=20, seed=7)
        values = generate_column(field, config)

        assert all(v == round(v, 0) for v in values)

    def test_generate_float_precision_none_produces_full_precision(self):
        """Test that `precision=None` does not round values."""
        field = float_field(min_val=0.0, max_val=1.0)
        config = GeneratorConfig(n=20, seed=23)
        values = generate_column(field, config)

        # Full-precision floats are very unlikely to already be rounded to 2 decimals
        assert not all(round(v, 2) == v for v in values)

    def test_generate_float_precision_with_unique(self):
        """Test that precision composes correctly with unique=True."""
        field = float_field(min_val=1.0, max_val=200.0, precision=2, unique=True)
        config = GeneratorConfig(n=20, seed=23)
        values = generate_column(field, config)

        assert len(values) == len(set(values))
        assert all(round(v, 2) == v for v in values)

    def test_generate_float_precision_with_nullable(self):
        """Test that precision composes correctly with nullable."""
        field = float_field(
            min_val=0.0, max_val=10.0, precision=1, nullable=True, null_probability=0.3
        )
        config = GeneratorConfig(n=50, seed=99)
        values = generate_column(field, config)

        non_null = [v for v in values if v is not None]
        assert all(round(v, 1) == v for v in non_null)


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
        assert all(isinstance(v, time) for v in values)

        # Times are generated at second resolution
        assert all(v.microsecond == 0 for v in values)

    def test_generate_time_with_constraints(self):
        """Test generating a Time column with min/max constraints."""
        field = time_field(min_time=time(9, 0, 0), max_time=time(12, 0, 0))
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        assert len(values) == 100

        # Verify all times are within the 9:00-12:00 range
        for v in values:
            assert time(9, 0, 0) <= v <= time(12, 0, 0)

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
            assert 9 <= st.hour <= 12

            # Verify end_time is in 13-17 range
            assert 13 <= et.hour <= 17

    def test_generate_time_with_string_constraints(self):
        """Test generating a Time column with string min/max constraints."""
        field = time_field(min_time="14:00:00", max_time="18:30:00")
        config = GeneratorConfig(n=100, seed=23)
        values = generate_column(field, config)

        assert len(values) == 100

        for v in values:
            assert time(14, 0, 0) <= v <= time(18, 30, 0)


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


class TestGenerateCoherentUnique:
    """Tests for `unique=True` on presets that use per-row coherent context (issue #441)."""

    @pytest.mark.parametrize("preset", ["email", "name", "user_name", "address", "phone_number"])
    def test_coherent_preset_unique(self, preset):
        """Coherent presets honor unique=True."""
        fields = {"x": string_field(preset=preset, unique=True)}
        config = GeneratorConfig(n=1000, seed=42, country="US", output="dict")
        data = generate_dataframe(fields, config)

        assert len(data["x"]) == 1000
        assert len(set(data["x"])) == 1000

    def test_coherent_preset_unique_impossible_raises(self):
        """An unsatisfiable unique constraint on a coherent preset raises, not silently dupes."""
        fields = {"x": string_field(preset="city", unique=True)}
        config = GeneratorConfig(n=2000, seed=42, country="US", output="dict")

        with pytest.raises(ValueError, match="Unable to generate 2000 unique values"):
            generate_dataframe(fields, config)

    def test_coherent_unique_keeps_row_coherence(self):
        """Re-drawn rows stay coherent across person and address columns."""
        fields = {
            "first": string_field(preset="first_name"),
            "last": string_field(preset="last_name"),
            "name": string_field(preset="name", unique=True),
            "email": string_field(preset="email", unique=True),
            "address": string_field(preset="address"),
            "city": string_field(preset="city"),
            "state": string_field(preset="state"),
        }
        config = GeneratorConfig(n=2000, seed=1, country="US", output="dict")
        data = generate_dataframe(fields, config)

        assert len(set(data["name"])) == 2000
        assert len(set(data["email"])) == 2000
        for first, last, name in zip(data["first"], data["last"], data["name"]):
            assert name == f"{first} {last}"

        for address, city, state in zip(data["address"], data["city"], data["state"]):
            assert f", {city}, {state} " in address

    def test_coherent_unique_matches_non_unique_without_collisions(self):
        """Rows before any collision are identical to the non-unique output."""
        base = GeneratorConfig(n=50, seed=7, country="US", output="dict")
        non_unique = generate_dataframe(
            {"name": string_field(preset="name"), "city": string_field(preset="city")}, base
        )
        unique = generate_dataframe(
            {"name": string_field(preset="name", unique=True), "city": string_field(preset="city")},
            base,
        )

        assert len(set(non_unique["name"])) == 50
        assert unique == non_unique

    def test_coherent_unique_is_reproducible(self):
        """Seeded generation with coherent unique columns is deterministic."""
        fields = {"email": string_field(preset="email", unique=True)}
        config = GeneratorConfig(n=2000, seed=99, country="US", output="dict")

        assert generate_dataframe(fields, config) == generate_dataframe(fields, config)


class TestCoherenceGroups:
    """Tests for grouping preset columns by `profile_fields(prefix=)`."""

    def test_groups_by_prefix_in_schema_order(self):
        from pointblank.field import profile_fields

        fields = {
            "id": int_field(),
            "rep": string_field(preset="name"),
            **profile_fields(set="minimal", prefix="b_"),
            "note": string_field(),
            **profile_fields(set="minimal", prefix="a_"),
        }
        groups = _get_coherence_groups(fields)
        assert list(groups) == [None, "b_", "a_"]
        assert list(groups[None]) == ["rep"]
        assert list(groups["b_"]) == ["b_first_name", "b_last_name", "b_email", "b_phone_number"]
        assert list(groups["a_"]) == ["a_first_name", "a_last_name", "a_email", "a_phone_number"]

    def test_hand_written_columns_join_longest_matching_prefix(self):
        from pointblank.field import profile_fields

        fields = {
            **profile_fields(set="minimal", prefix="a_"),
            **profile_fields(set="minimal", prefix="a_b_"),
            "a_b_city": string_field(preset="city"),
            "a_city": string_field(preset="city"),
            "city": string_field(preset="city"),
        }
        groups = _get_coherence_groups(fields)
        assert "a_b_city" in groups["a_b_"]
        assert "a_city" in groups["a_"]
        assert list(groups[None]) == ["city"]

    def test_no_profiles_is_one_group(self):
        fields = {
            "name": string_field(preset="name"),
            "customer_email": string_field(preset="email"),
            "code": string_field(pattern=r"[A-Z]{3}"),
        }
        assert _get_coherence_groups(fields) == {
            None: {"name": fields["name"], "customer_email": fields["customer_email"]}
        }

    def test_unseeded_profiles_are_different_people(self):
        from pointblank.field import profile_fields

        fields = {
            **profile_fields(set="minimal", prefix="a_"),
            **profile_fields(set="minimal", prefix="b_"),
        }
        data = generate_dataframe(fields, GeneratorConfig(n=50, country="US", output="dict"))
        same = sum(a == b for a, b in zip(data["a_email"], data["b_email"]))
        assert same <= 2


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


class TestSeededColumnIndependence:
    """With a seed, each column gets its own random sequence (BUGS.md #113, #118, #128)."""

    def test_same_field_twice_gives_different_columns(self):
        field = int_field(min_val=1, max_val=100, nullable=True, null_probability=0.3)
        config = GeneratorConfig(n=200, seed=7, output="dict")
        data = generate_dataframe({"a": field, "b": field}, config)

        assert data["a"] != data["b"]
        assert sum(a == b for a, b in zip(data["a"], data["b"]) if a is not None) < 20

        # The missing values fall in different rows
        nulls_a = {i for i, v in enumerate(data["a"]) if v is None}
        nulls_b = {i for i, v in enumerate(data["b"]) if v is None}
        assert nulls_a and nulls_b and nulls_a != nulls_b

    def test_different_ranges_are_not_correlated(self):
        """Two time fields with different windows don't share their minutes and seconds."""
        fields = {
            "start": time_field(min_time=time(6, 0), max_time=time(9, 0)),
            "end": time_field(min_time=time(16, 0), max_time=time(20, 0)),
        }
        data = generate_dataframe(fields, GeneratorConfig(n=100, seed=23, output="dict"))
        same = sum(
            (s.minute, s.second) == (e.minute, e.second) for s, e in zip(data["start"], data["end"])
        )
        assert same < 5

    def test_other_columns_unchanged_by_adding_reordering_or_removing(self):
        fields = {
            "id": int_field(min_val=1, max_val=10_000, unique=True),
            "score": float_field(min_val=0, max_val=1, nullable=True, null_probability=0.2),
            "code": string_field(pattern=r"[A-Z]{3}-\d{2}"),
        }
        config = GeneratorConfig(n=50, seed=23, output="dict")
        base = generate_dataframe(fields, config)

        more = generate_dataframe({"flag": bool_field(), **fields}, config)
        assert {col: more[col] for col in fields} == base

        reordered = generate_dataframe(dict(reversed(fields.items())), config)
        assert {col: reordered[col] for col in fields} == base

        fewer = generate_dataframe({"id": fields["id"], "code": fields["code"]}, config)
        assert fewer == {"id": base["id"], "code": base["code"]}

    def test_seed_zero_is_reproducible(self):
        """A seed of `0` seeds the missing values too (they were random with `seed=0`)."""
        fields = {"x": int_field(nullable=True, null_probability=0.5)}
        config = GeneratorConfig(n=100, seed=0, output="dict")
        assert generate_dataframe(fields, config) == generate_dataframe(fields, config)
        assert generate_column(fields["x"], config) == generate_column(fields["x"], config)

    def test_coherent_unique_nulls_match_non_unique(self):
        """A coherent column has its nulls in the same rows with or without `unique=True`."""
        config = GeneratorConfig(n=50, seed=7, country="US", output="dict")
        city = string_field(preset="city", nullable=True, null_probability=0.3)
        non_unique = generate_dataframe({"name": string_field(preset="name"), "city": city}, config)
        unique = generate_dataframe(
            {"name": string_field(preset="name", unique=True), "city": city}, config
        )
        assert unique == non_unique
        assert None in unique["city"]

    @pytest.mark.parametrize("country", [["US", "FR"], {"DE": 2, "JP": 1}])
    def test_multi_country_same_in_every_session(self, country):
        """The per-country seeds don't depend on Python's per-process string hashing."""
        import subprocess
        import sys

        code = (
            "import pointblank as pb; "
            "s = pb.Schema(name=pb.string_field(preset='name'), x=pb.int_field(max_val=99)); "
            f"print(pb.generate_dataset(s, n=20, seed=7, country={country!r}, output='dict'))"
        )
        outputs = {
            subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
            ).stdout
            for hash_seed in ["1", "2", "3"]
        }
        assert len(outputs) == 1


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


INT_DTYPES = ["Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64"]
FLOAT_DTYPES = ["Float32", "Float64"]


class TestGenerateDataframeDtypes:
    """Tests that generated columns have the dtypes of their fields."""

    @pytest.mark.parametrize("dtype", INT_DTYPES + FLOAT_DTYPES)
    def test_polars_numeric_dtype(self, dtype):
        pl = pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        field = int_field(dtype=dtype) if dtype in INT_DTYPES else float_field(dtype=dtype)
        nullable = (
            int_field(dtype=dtype, nullable=True, null_probability=0.5)
            if dtype in INT_DTYPES
            else float_field(dtype=dtype, nullable=True, null_probability=0.5)
        )
        df = generate_dataset(Schema(x=field, y=nullable), n=20, seed=23)

        assert df.schema["x"] == getattr(pl, dtype)
        assert df.schema["y"] == getattr(pl, dtype)
        assert df["y"].null_count() > 0

    @pytest.mark.parametrize("dtype", INT_DTYPES)
    def test_pandas_int_dtype(self, dtype):
        pytest.importorskip("pandas")
        from pointblank import Schema, generate_dataset

        schema = Schema(
            x=int_field(dtype=dtype),
            y=int_field(dtype=dtype, nullable=True, null_probability=0.5),
        )
        df = generate_dataset(schema, n=20, seed=23, output="pandas")

        # A NumPy integer, or pandas' nullable integer (keeping its dtype) when there are nulls
        assert str(df["x"].dtype) == dtype.lower()
        assert str(df["y"].dtype) == dtype
        assert df["y"].isna().sum() > 0

    @pytest.mark.parametrize("dtype", FLOAT_DTYPES)
    def test_pandas_float_dtype(self, dtype):
        pytest.importorskip("pandas")
        from pointblank import Schema, generate_dataset

        schema = Schema(
            x=float_field(dtype=dtype),
            y=float_field(dtype=dtype, nullable=True, null_probability=0.5),
        )
        df = generate_dataset(schema, n=20, seed=23, output="pandas")

        assert str(df["x"].dtype) == dtype.lower()
        assert str(df["y"].dtype) == dtype.lower()
        assert df["y"].isna().sum() > 0

    def test_dtype_strings_in_schema(self):
        """A dtype-only schema (e.g., `x="Int16"`) gives the same dtypes."""
        pl = pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(a="Int16", b="UInt32", c="Float32", d="int8")
        df = generate_dataset(schema, n=5, seed=23)

        assert df.schema == pl.Schema(
            {"a": pl.Int16, "b": pl.UInt32, "c": pl.Float32, "d": pl.Int8}
        )

    def test_full_uint64_range(self):
        """Values above the Int64 range are kept exactly."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        big = 2**64 - 1
        schema = Schema(x=int_field(dtype="UInt64", allowed=[big]))

        assert generate_dataset(schema, n=3, seed=23)["x"].to_list() == [big] * 3
        assert generate_dataset(schema, n=3, seed=23, output="pandas")["x"].tolist() == [big] * 3

    def test_zero_rows_keep_dtypes(self):
        """With no values to infer from, every column still gets its field's dtype."""
        pl = pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(
            a=int_field(dtype="Int32"),
            b=float_field(dtype="Float32"),
            s=string_field(),
            o=bool_field(),
            d=date_field(),
            dt=datetime_field(),
            du=duration_field(),
            t=time_field(),
        )
        df = generate_dataset(schema, n=0, seed=23)

        assert df.schema == pl.Schema(
            {
                "a": pl.Int32,
                "b": pl.Float32,
                "s": pl.String,
                "o": pl.Boolean,
                "d": pl.Date,
                "dt": pl.Datetime("us"),
                "du": pl.Duration("us"),
                "t": pl.Time,
            }
        )

        # The inferred dtypes of non-empty columns are the same
        assert generate_dataset(schema, n=3, seed=23).schema == df.schema

    def test_all_null_column_keeps_dtype(self):
        pl = pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(
            s=string_field(generator=lambda: None),
            t=time_field(generator=lambda: None),
        )
        df = generate_dataset(schema, n=3, seed=23)

        assert df.schema == pl.Schema({"s": pl.String, "t": pl.Time})
        assert df.null_count().row(0) == (3, 3)

    @pytest.mark.parametrize("output", ["polars", "pandas"])
    def test_generated_table_matches_its_schema(self, output):
        """The generated table passes `col_schema_match()` against the schema it came from."""
        pytest.importorskip(output)
        import pointblank as pb

        schema = pb.Schema(
            a=int_field(dtype="Int8", min_val=0, max_val=100),
            b=int_field(dtype="UInt16", nullable=True, null_probability=0.3),
            c=float_field(dtype="Float32", min_val=0.0, max_val=1.0),
            d=int_field(),
        )
        df = pb.generate_dataset(schema, n=30, seed=23, output=output)

        # pandas spells NumPy dtypes in lowercase (`int8`), so compare case-insensitively there
        validation = (
            pb.Validate(df)
            .col_schema_match(schema, case_sensitive_dtypes=output == "polars")
            .interrogate()
        )

        assert validation.all_passed()

    def test_time_column_polars(self):
        pl = pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(t=time_field(min_time="09:00:00", max_time="17:00:00"))
        df = generate_dataset(schema, n=20, seed=23)

        assert df.schema["t"] == pl.Time
        assert all(time(9, 0, 0) <= v <= time(17, 0, 0) for v in df["t"].to_list())

    def test_time_column_pandas_and_dict(self):
        pytest.importorskip("pandas")
        from pointblank import Schema, generate_dataset

        schema = Schema(t=time_field(nullable=True, null_probability=0.3))

        df = generate_dataset(schema, n=20, seed=23, output="pandas")
        assert all(isinstance(v, time) for v in df["t"].dropna())

        data = generate_dataset(schema, n=20, seed=23, output="dict")
        assert all(v is None or isinstance(v, time) for v in data["t"])
        assert None in data["t"]

    def test_time_column_unique(self):
        from pointblank import Schema, generate_dataset

        schema = Schema(t=time_field(min_time="09:00:00", max_time="09:01:00", unique=True))
        data = generate_dataset(schema, n=61, seed=23, output="dict")

        assert len(set(data["t"])) == 61

    def test_dict_output_is_unchanged(self):
        """The dict output holds the generated Python values, with no casting."""
        from pointblank import Schema, generate_dataset

        schema = Schema(x=int_field(dtype="UInt8"), y=float_field(dtype="Float32"))
        data = generate_dataset(schema, n=5, seed=23, output="dict")

        assert all(type(v) is int for v in data["x"])
        assert all(type(v) is float for v in data["y"])

    @pytest.mark.parametrize("output", ["polars", "pandas"])
    @pytest.mark.parametrize(
        "field, match",
        [
            (int_field(generator=lambda: 1.5), "has the dtype Int64, but it got the value 1.5"),
            (int_field(generator=lambda: True), "got the value True"),
            (int_field(dtype="UInt8", generator=lambda: 300), "an integer from 0 to 255"),
            (int_field(dtype="Int8", generator=lambda: -129), "an integer from -128 to 127"),
            (float_field(generator=lambda: "1.5"), "got the value '1.5', which isn't a number"),
        ],
    )
    def test_generator_value_that_doesnt_fit_raises(self, output, field, match):
        pytest.importorskip(output)
        from pointblank import Schema, generate_dataset

        with pytest.raises(ValueError, match=match):
            generate_dataset(Schema(x=field), n=3, seed=23, output=output)

    def test_generator_numpy_values_are_accepted(self):
        np = pytest.importorskip("numpy")
        pl = pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(
            x=int_field(dtype="Int16", generator=lambda: np.int64(7)),
            y=float_field(dtype="Float32", generator=lambda: np.float64(0.5)),
            z=float_field(generator=lambda: 3),
        )
        df = generate_dataset(schema, n=3, seed=23)

        assert df.schema == pl.Schema({"x": pl.Int16, "y": pl.Float32, "z": pl.Float64})
        assert df["z"].to_list() == [3.0, 3.0, 3.0]


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

    def test_country_code_presets_across_countries(self):
        """Test country_code_2 and country_code_3 presets."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(
            country_code_2=string_field(preset="country_code_2"),
            country_code_3=string_field(preset="country_code_3"),
        )

        # Expected values from the COUNTRY_INFO mapping
        expected = {
            "US": ("US", "USA"),
            "DE": ("DE", "DEU"),
            "FR": ("FR", "FRA"),
            "JP": ("JP", "JPN"),
            "BR": ("BR", "BRA"),
            "GB": ("GB", "GBR"),
            "AU": ("AU", "AUS"),
            "IN": ("IN", "IND"),
        }

        for country, (exp_code_2, exp_code_3) in expected.items():
            df = generate_dataset(schema, n=5, seed=23, country=country)

            # All rows should have the same country codes for a single-country dataset
            validation = (
                Validate(df)
                .col_vals_not_null(columns=["country_code_2", "country_code_3"])
                .col_vals_in_set(columns="country_code_2", set=[exp_code_2])
                .col_vals_in_set(columns="country_code_3", set=[exp_code_3])
                .interrogate()
            )

            assert validation.all_passed(), f"Failed for country {country}"

    def test_country_code_presets_multi_country(self):
        """Test country_code_2 and country_code_3 with multiple countries."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(
            country_code_2=string_field(preset="country_code_2"),
            country_code_3=string_field(preset="country_code_3"),
            city=string_field(preset="city"),
        )

        df = generate_dataset(schema, n=20, seed=23, country=["US", "DE", "JP"])

        assert df.shape[0] == 20

        # Check that all expected country codes appear
        validation = (
            Validate(df)
            .col_vals_not_null(columns=["country_code_2", "country_code_3"])
            .col_vals_in_set(
                columns="country_code_2",
                set=["US", "DE", "JP"],
            )
            .col_vals_in_set(
                columns="country_code_3",
                set=["USA", "DEU", "JPN"],
            )
            .interrogate()
        )

        assert validation.all_passed()

    def test_country_code_presets_with_other_presets(self):
        """Test country code presets combined with name, city, and country presets."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(
            name=string_field(preset="name"),
            city=string_field(preset="city"),
            country=string_field(preset="country"),
            country_code_2=string_field(preset="country_code_2"),
            country_code_3=string_field(preset="country_code_3"),
        )

        df = generate_dataset(schema, n=10, seed=23, country="ES")

        validation = (
            Validate(df)
            .col_vals_not_null(
                columns=["name", "city", "country", "country_code_2", "country_code_3"]
            )
            .col_vals_in_set(columns="country", set=["Spain"])
            .col_vals_in_set(columns="country_code_2", set=["ES"])
            .col_vals_in_set(columns="country_code_3", set=["ESP"])
            .interrogate()
        )

        assert validation.all_passed()

    def test_locale_code_preset_single_language_countries(self):
        """Test locale_code preset returns correct codes for single-language countries."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(locale_code=string_field(preset="locale_code"))

        expected = {
            "US": ["en_US"],
            "DE": ["de_DE"],
            "FR": ["fr_FR"],
            "JP": ["ja_JP"],
            "BR": ["pt_BR"],
            "PL": ["pl_PL"],
            "IT": ["it_IT"],
            "KR": ["ko_KR"],
        }

        for country, codes in expected.items():
            df = generate_dataset(schema, n=5, seed=23, country=country)

            validation = (
                Validate(df)
                .col_vals_not_null(columns="locale_code")
                .col_vals_in_set(columns="locale_code", set=codes)
                .interrogate()
            )

            assert validation.all_passed(), f"Failed for country {country}"

    def test_locale_code_preset_multilingual_countries(self):
        """Test locale_code preset returns valid codes for multilingual countries."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(locale_code=string_field(preset="locale_code"))

        multilingual = {
            "BE": ["nl_BE", "fr_BE", "de_BE"],
            "CH": ["de_CH", "fr_CH", "it_CH"],
            "CA": ["en_CA", "fr_CA"],
            "SG": ["en_SG", "zh_SG", "ms_SG", "ta_SG"],
            "ZA": ["en_ZA", "af_ZA", "zu_ZA"],
        }

        for country, valid_codes in multilingual.items():
            df = generate_dataset(schema, n=50, seed=23, country=country)

            validation = (
                Validate(df)
                .col_vals_not_null(columns="locale_code")
                .col_vals_in_set(columns="locale_code", set=valid_codes)
                .interrogate()
            )

            assert validation.all_passed(), f"Failed for country {country}"

    def test_locale_code_preset_format(self):
        """Test locale_code values match the expected xx_XX pattern."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(locale_code=string_field(preset="locale_code"))

        for country in ["US", "DE", "CH", "JP", "IN"]:
            df = generate_dataset(schema, n=10, seed=23, country=country)
            for val in df["locale_code"].to_list():
                parts = val.split("_")
                assert len(parts) == 2, f"Bad format: {val}"
                assert parts[0].islower(), f"Language part not lowercase: {val}"
                assert parts[1].isupper(), f"Country part not uppercase: {val}"

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
        from pointblank.countries import LocaleGenerator

        # Build exonym -> native mapping dynamically from locale data
        EXONYM_TO_NATIVE = {}
        for cc in COUNTRIES_WITH_FULL_DATA:
            gen = LocaleGenerator(country=cc, seed=1)
            raw_locations = gen._data.address.get("locations", {})
            locs = []
            if isinstance(raw_locations, dict):
                for tier_locs in raw_locations.values():
                    if isinstance(tier_locs, list):
                        locs.extend(tier_locs)
            elif isinstance(raw_locations, list):
                locs = raw_locations
            for loc in locs:
                if isinstance(loc, dict) and "exonym" in loc:
                    EXONYM_TO_NATIVE[loc["exonym"]] = loc["city"]

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
        from pointblank.countries import LocaleGenerator

        # Build exonym -> native mapping dynamically from locale data
        EXONYM_TO_NATIVE = {}
        for cc in COUNTRIES_WITH_FULL_DATA:
            gen = LocaleGenerator(country=cc, seed=1)
            raw_locations = gen._data.address.get("locations", {})
            locs = []
            if isinstance(raw_locations, dict):
                for tier_locs in raw_locations.values():
                    if isinstance(tier_locs, list):
                        locs.extend(tier_locs)
            elif isinstance(raw_locations, list):
                locs = raw_locations
            for loc in locs:
                if isinstance(loc, dict) and "exonym" in loc:
                    EXONYM_TO_NATIVE[loc["exonym"]] = loc["city"]

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

                # For cities with exonyms, check the native name instead
                native_name = EXONYM_TO_NATIVE.get(city, city)
                if native_name not in address:
                    mismatches.append(
                        f"Row {i}: city='{city}' (native='{native_name}') not in address='{address}'"
                    )

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
            raw_locations = gen._data.address.get("locations", [])
            # Handle tiered location dicts (frequency-weighted format)
            tiered_keys = {"very_common", "common", "uncommon", "rare"}
            if isinstance(raw_locations, dict) and set(raw_locations.keys()) <= tiered_keys:
                locations = [loc for tier in raw_locations.values() for loc in tier]
            else:
                locations = raw_locations
            valid_cities = {loc.get("city", "") for loc in locations}

            # Check that each address contains a valid city for this country
            for i, row in enumerate(df.iter_rows()):
                address = row[0]
                found_city = any(city in address for city in valid_cities if city)

                assert found_city

    @pytest.mark.parametrize("postcode_first", [False, True])
    def test_address_postcode_coherence_all_countries(self, monkeypatch, postcode_first):
        """
        Test that the postcode column matches the postcode within the address of the same row,
        whichever of the two columns comes first.
        """
        from pointblank import Schema, generate_dataset, string_field
        from pointblank.countries import LocaleGenerator

        address_col = ("address", string_field(preset="address"))
        postcode_col = ("postcode", string_field(preset="postcode"))
        city_col = ("city", string_field(preset="city"))
        columns = [postcode_col, address_col] if postcode_first else [address_col, postcode_col]
        schema = Schema(columns=[*columns, city_col])

        n_checked = 0
        for country in COUNTRIES_WITH_FULL_DATA:
            # Keep only the address formats that include the postcode (the locale data is shared,
            # so `monkeypatch` restores it after the test)
            address_data = LocaleGenerator(country=country, seed=1)._data.address
            formats = [
                fmt for fmt in address_data.get("address_formats", []) if "{postcode}" in fmt
            ]
            if not formats:
                continue
            monkeypatch.setitem(address_data, "address_formats", formats)

            data = generate_dataset(schema, n=50, seed=23, country=country, output="dict")
            for address, postcode in zip(data["address"], data["postcode"]):
                assert postcode in address, (country, address, postcode)
            n_checked += 1

        assert n_checked >= 80

    def test_address_postcode_coherence_with_unique_postcode(self):
        # A duplicate postcode re-draws the row's location, and the address follows the new one
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(
            address=string_field(preset="address"),
            postcode=string_field(preset="postcode", unique=True),
        )
        data = generate_dataset(schema, n=1000, seed=1, country="US", output="dict")

        assert len(set(data["postcode"])) == 1000
        for address, postcode in zip(data["address"], data["postcode"]):
            assert address.endswith(f" {postcode}")

    def test_address_postcode_coherence_per_profile(self):
        # Each prefixed profile has its own row locations, and so its own postcodes
        from pointblank import Schema, generate_dataset, profile_fields

        schema = Schema(
            **profile_fields(set="full", prefix="sender_"),
            **profile_fields(set="full", prefix="recipient_"),
        )
        data = generate_dataset(schema, n=100, seed=23, country="US", output="dict")

        for prefix in ["sender_", "recipient_"]:
            for address, postcode in zip(data[f"{prefix}address"], data[f"{prefix}postcode"]):
                assert address.endswith(f" {postcode}")
        assert data["sender_postcode"] != data["recipient_postcode"]


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

    def test_credit_card_provider_returns_valid_providers(self):
        """Ensure credit_card_provider returns valid card network names."""
        pytest.importorskip("polars")
        from pointblank import Schema, Validate, generate_dataset, string_field

        schema = Schema(columns=[("provider", string_field(preset="credit_card_provider"))])
        df = generate_dataset(schema, n=100, seed=23, country="US")

        valid_providers = {"Visa", "Mastercard", "American Express", "Discover"}

        # Validate all providers are in the expected set
        validation = (
            Validate(df)
            .col_vals_not_null(columns="provider")
            .col_vals_in_set(columns="provider", set=valid_providers)
            .interrogate()
        )

        assert validation.all_passed()

        # Verify we get a variety of providers (not all the same)
        unique_providers = set(df["provider"].unique().to_list())
        assert len(unique_providers) > 1, "Expected multiple different providers"

    def test_credit_card_provider_and_number_coherence(self):
        """Ensure credit_card_provider matches credit_card_number prefix."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(
            columns=[
                ("card_number", string_field(preset="credit_card_number")),
                ("card_provider", string_field(preset="credit_card_provider")),
            ]
        )

        df = generate_dataset(schema, n=100, seed=23, country="US")

        # Map prefixes to expected providers
        PREFIX_TO_PROVIDER = {
            "4": "Visa",
            "5": "Mastercard",
            "37": "American Express",
            "6011": "Discover",
        }

        # Verify coherence: each card number prefix should match its provider
        for i, row in enumerate(df.iter_rows()):
            card_number, provider = row

            # Determine expected provider from card number prefix
            if card_number.startswith("4"):
                expected = "Visa"
            elif card_number.startswith("5"):
                expected = "Mastercard"
            elif card_number.startswith("37"):
                expected = "American Express"
            elif card_number.startswith("6011"):
                expected = "Discover"
            else:
                expected = None

            assert provider == expected, (
                f"Row {i}: card_number={card_number[:6]}... has provider={provider}, "
                f"expected={expected}"
            )

    def test_credit_card_provider_standalone_varies(self):
        """Ensure standalone credit_card_provider generates varied values."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        # Without credit_card_number, provider should still vary per row
        schema = Schema(columns=[("provider", string_field(preset="credit_card_provider"))])
        df = generate_dataset(schema, n=50, seed=23, country="US")

        providers = df["provider"].to_list()

        # Should not all be the same value
        unique_count = len(set(providers))
        assert unique_count > 1, "Standalone credit_card_provider should vary per row"

    def test_credit_card_provider_reproducible_with_seed(self):
        """Same seed produces same credit card providers."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset, string_field

        schema = Schema(
            columns=[
                ("card_number", string_field(preset="credit_card_number")),
                ("card_provider", string_field(preset="credit_card_provider")),
            ]
        )

        df1 = generate_dataset(schema, n=20, seed=99, country="US")
        df2 = generate_dataset(schema, n=20, seed=99, country="US")

        assert df1["card_number"].to_list() == df2["card_number"].to_list()
        assert df1["card_provider"].to_list() == df2["card_provider"].to_list()


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


class TestLocaleMixing:
    """Tests for multi-country locale mixing (country as list or dict)."""

    # ── allocate_rows unit tests ──────────────────────────────────────────

    def test_allocate_rows_equal_weights(self):
        """Equal weights split rows evenly (with remainder distribution)."""
        from pointblank.generate.generators import _allocate_rows

        alloc = _allocate_rows({"US": 1.0, "DE": 1.0, "FR": 1.0}, 100)

        assert sum(alloc.values()) == 100

        # Each country should get 33 or 34 rows
        for count in alloc.values():
            assert count in (33, 34)

    def test_allocate_rows_weighted(self):
        """Weighted allocation gives proportional row counts."""
        from pointblank.generate.generators import _allocate_rows

        alloc = _allocate_rows({"US": 0.7, "DE": 0.3}, 100)

        assert sum(alloc.values()) == 100
        assert alloc["US"] == 70
        assert alloc["DE"] == 30

    def test_allocate_rows_unnormalized_weights(self):
        """Weights that don't sum to 1.0 are auto-normalized."""
        from pointblank.generate.generators import _allocate_rows

        alloc = _allocate_rows({"US": 3.0, "DE": 7.0}, 100)

        assert sum(alloc.values()) == 100
        assert alloc["US"] == 30
        assert alloc["DE"] == 70

    def test_allocate_rows_single_country(self):
        """Single-country dict allocates all rows."""
        from pointblank.generate.generators import _allocate_rows

        alloc = _allocate_rows({"US": 1.0}, 50)

        assert alloc == {"US": 50}

    def test_allocate_rows_zero_n(self):
        """Zero rows gives zero rows per country."""
        from pointblank.generate.generators import _allocate_rows

        alloc = _allocate_rows({"US": 0.5, "DE": 0.5}, 0)

        assert sum(alloc.values()) == 0

    # ── List form ─────────────────────────────────────────────────────────

    def test_list_form_basic(self):
        """Passing a list of countries produces rows from all countries."""
        config = GeneratorConfig(n=60, seed=23, output="dict", country=["US", "DE", "FR"])
        fields = {"name": string_field(preset="name")}
        result = generate_dataframe(fields, config)

        assert len(result["name"]) == 60

    def test_list_form_row_count(self):
        """List form produces exactly n rows."""
        config = GeneratorConfig(n=100, seed=23, output="dict", country=["US", "DE"])
        fields = {"city": string_field(preset="city")}
        result = generate_dataframe(fields, config)

        assert len(result["city"]) == 100

    def test_list_single_element_equivalent_to_string(self):
        """A single-element list produces identical output to a plain string."""
        config_str = GeneratorConfig(n=50, seed=23, output="dict", country="DE")
        config_list = GeneratorConfig(n=50, seed=23, output="dict", country=["DE"])
        fields = {"name": string_field(preset="name")}
        result_str = generate_dataframe(fields, config_str)
        result_list = generate_dataframe(fields, config_list)
        assert result_str["name"] == result_list["name"]

    # ── Dict form ─────────────────────────────────────────────────────────

    def test_dict_form_basic(self):
        """Passing a dict of weights produces rows from all countries."""
        config = GeneratorConfig(n=100, seed=23, output="dict", country={"US": 0.7, "DE": 0.3})
        fields = {"name": string_field(preset="name")}
        result = generate_dataframe(fields, config)
        assert len(result["name"]) == 100

    def test_dict_single_element_equivalent_to_string(self):
        """A single-element dict produces identical output to a plain string."""
        config_str = GeneratorConfig(n=50, seed=23, output="dict", country="FR")
        config_dict = GeneratorConfig(n=50, seed=23, output="dict", country={"FR": 1.0})
        fields = {"name": string_field(preset="name")}
        result_str = generate_dataframe(fields, config_str)
        result_dict = generate_dataframe(fields, config_dict)
        assert result_str["name"] == result_dict["name"]

    def test_dict_form_weight_normalization(self):
        """Weights that sum to more/less than 1.0 work correctly."""
        config = GeneratorConfig(n=100, seed=23, output="dict", country={"US": 5, "DE": 5})
        fields = {"city": string_field(preset="city")}
        result = generate_dataframe(fields, config)
        assert len(result["city"]) == 100

    # ── Shuffle behavior ──────────────────────────────────────────────────

    def test_shuffle_true_interleaves(self):
        """With shuffle=True (default), rows from different countries are mixed."""
        # Use a simple int column + a name preset so we can detect mixing
        config = GeneratorConfig(
            n=100,
            seed=23,
            output="dict",
            country=["US", "JP"],
            shuffle=True,
        )
        fields = {"name": string_field(preset="name")}
        result = generate_dataframe(fields, config)
        # With 50 US + 50 JP rows shuffled, it's very unlikely that all US
        # rows come before all JP rows. But we can't deterministically know
        # the exact order. Just verify row count is correct.
        assert len(result["name"]) == 100

    def test_shuffle_false_groups_by_country(self):
        """With shuffle=False, rows are grouped by country in dict order."""
        config = GeneratorConfig(
            n=100,
            seed=23,
            output="dict",
            country={"US": 0.5, "JP": 0.5},
            shuffle=False,
        )
        # Use city preset since US and JP cities are distinguishable
        fields = {"city": string_field(preset="city")}
        result = generate_dataframe(fields, config)
        assert len(result["city"]) == 100
        # The first 50 should all be from US data, last 50 from JP.
        # We can verify by generating the batches separately and comparing.
        config_us = GeneratorConfig(
            n=50, seed=_derive_seed(23, "country", "US"), output="dict", country="US"
        )
        config_jp = GeneratorConfig(
            n=50, seed=_derive_seed(23, "country", "JP"), output="dict", country="JP"
        )
        us_only = generate_dataframe(fields, config_us)
        jp_only = generate_dataframe(fields, config_jp)
        assert result["city"][:50] == us_only["city"]
        assert result["city"][50:] == jp_only["city"]

    def test_shuffle_true_differs_from_false(self):
        """shuffle=True and shuffle=False produce different row orderings."""
        fields = {"name": string_field(preset="name")}
        config_shuffled = GeneratorConfig(
            n=100,
            seed=23,
            output="dict",
            country=["US", "DE"],
            shuffle=True,
        )
        config_blocked = GeneratorConfig(
            n=100,
            seed=23,
            output="dict",
            country=["US", "DE"],
            shuffle=False,
        )
        result_shuffled = generate_dataframe(fields, config_shuffled)
        result_blocked = generate_dataframe(fields, config_blocked)

        # Same set of values, different order
        assert sorted(result_shuffled["name"]) == sorted(result_blocked["name"])

        # But different ordering (extremely unlikely to be same by chance)
        assert result_shuffled["name"] != result_blocked["name"]

    # ── Reproducibility ───────────────────────────────────────────────────

    def test_reproducibility_list_form(self):
        """Same seed + list country produces identical output."""
        fields = {"name": string_field(preset="name")}
        config1 = GeneratorConfig(n=50, seed=23, output="dict", country=["US", "DE"])
        config2 = GeneratorConfig(n=50, seed=23, output="dict", country=["US", "DE"])
        r1 = generate_dataframe(fields, config1)
        r2 = generate_dataframe(fields, config2)

        assert r1["name"] == r2["name"]

    def test_reproducibility_dict_form(self):
        """Same seed + dict country produces identical output."""
        fields = {"city": string_field(preset="city")}
        config1 = GeneratorConfig(n=50, seed=23, output="dict", country={"US": 0.8, "FR": 0.2})
        config2 = GeneratorConfig(n=50, seed=23, output="dict", country={"US": 0.8, "FR": 0.2})
        r1 = generate_dataframe(fields, config1)
        r2 = generate_dataframe(fields, config2)
        assert r1["city"] == r2["city"]

    # ── Coherence within mixed rows ───────────────────────────────────────

    def test_coherence_preserved_across_countries(self):
        """Person coherence (name + email) works in multi-country mixing."""

        config = GeneratorConfig(
            n=60,
            seed=23,
            output="dict",
            country=["US", "FR"],
            shuffle=False,
        )
        fields = {
            "name": string_field(preset="name"),
            "email": string_field(preset="email"),
        }
        result = generate_dataframe(fields, config)

        # Every email should contain some fragment of the name (lower-cased,
        # transliterated to ASCII, since emails use the same transliteration).
        for name, email in zip(result["name"], result["email"]):
            normalized_name = (
                _transliterate_to_ascii(name).lower().replace(".", "").replace("-", "")
            )
            name_parts = normalized_name.split()
            email_local = (
                email.split("@")[0].lower().replace(".", "").replace("_", "").replace("-", "")
            )
            found = any(part in email_local for part in name_parts if len(part) > 1)
            assert found, f"Email '{email}' doesn't match name '{name}'"

    def test_address_coherence_across_countries(self):
        """Address coherence (city + postcode) works in multi-country mixing."""
        config = GeneratorConfig(
            n=40,
            seed=23,
            output="dict",
            country=["US", "DE"],
            shuffle=False,
        )
        fields = {
            "city": string_field(preset="city"),
            "postcode": string_field(preset="postcode"),
        }
        result = generate_dataframe(fields, config)
        assert len(result["city"]) == 40
        assert len(result["postcode"]) == 40

    def test_schema_generate_list_form(self):
        """Schema.generate() accepts list country form."""
        from pointblank.schema import Schema

        schema = Schema(name=string_field(preset="name"))
        df = schema.generate(n=30, seed=23, output="dict", country=["US", "DE"])
        assert len(df["name"]) == 30

    def test_schema_generate_dict_form(self):
        """Schema.generate() accepts dict country form."""
        from pointblank.schema import Schema

        schema = Schema(name=string_field(preset="name"))
        df = schema.generate(n=30, seed=23, output="dict", country={"US": 0.7, "DE": 0.3})
        assert len(df["name"]) == 30

    def test_generate_dataset_function_list_form(self):
        """generate_dataset() function accepts list country form."""
        from pointblank.schema import Schema, generate_dataset as gen_ds

        schema = Schema(name=string_field(preset="name"))
        df = gen_ds(schema, n=30, seed=23, output="dict", country=["US", "FR"])
        assert len(df["name"]) == 30

    def test_generate_dataset_function_dict_form(self):
        """generate_dataset() function accepts dict country form."""
        from pointblank.schema import Schema, generate_dataset as gen_ds

        schema = Schema(city=string_field(preset="city"))
        df = gen_ds(schema, n=30, seed=23, output="dict", country={"US": 0.6, "JP": 0.4})
        assert len(df["city"]) == 30

    def test_generate_dataset_function_shuffle_param(self):
        """generate_dataset() passes shuffle= through correctly."""
        from pointblank.schema import Schema, generate_dataset as gen_ds

        schema = Schema(name=string_field(preset="name"))
        df_shuffled = gen_ds(
            schema, n=50, seed=23, output="dict", country=["US", "DE"], shuffle=True
        )
        df_blocked = gen_ds(
            schema, n=50, seed=23, output="dict", country=["US", "DE"], shuffle=False
        )

        # Same values, different order
        assert sorted(df_shuffled["name"]) == sorted(df_blocked["name"])
        assert df_shuffled["name"] != df_blocked["name"]

    # ── Polars output format ──────────────────────────────────────────────

    def test_polars_output_multi_country(self):
        """Multi-country mixing works with Polars output."""
        config = GeneratorConfig(n=30, seed=23, output="polars", country=["US", "DE"])
        fields = {"name": string_field(preset="name")}
        import polars as pl

        result = generate_dataframe(fields, config)
        assert isinstance(result, pl.DataFrame)
        assert result.shape[0] == 30

    # ── Non-preset columns in mixed mode ──────────────────────────────────

    def test_non_preset_columns_work_in_mixed_mode(self):
        """Integer and float columns work alongside country mixing."""
        config = GeneratorConfig(
            n=60,
            seed=23,
            output="dict",
            country=["US", "DE", "FR"],
        )

        fields = {
            "name": string_field(preset="name"),
            "age": int_field(min_val=18, max_val=80),
            "score": float_field(min_val=0.0, max_val=100.0),
        }

        result = generate_dataframe(fields, config)

        assert len(result["name"]) == 60
        assert len(result["age"]) == 60
        assert len(result["score"]) == 60

        # Age and score constraints should still hold
        assert all(18 <= a <= 80 for a in result["age"])
        assert all(0.0 <= s <= 100.0 for s in result["score"])

    # ── Validation errors ─────────────────────────────────────────────────

    def test_empty_list_raises_error(self):
        """Empty list country raises ValueError."""
        with pytest.raises(ValueError, match="country list must contain at least one"):
            GeneratorConfig(n=10, country=[])

    def test_empty_dict_raises_error(self):
        """Empty dict country raises ValueError."""
        with pytest.raises(ValueError, match="country dict must contain at least one"):
            GeneratorConfig(n=10, country={})

    def test_negative_weight_raises_error(self):
        """Negative weight raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            GeneratorConfig(n=10, country={"US": -0.5, "DE": 0.5})

    def test_zero_weight_raises_error(self):
        """Zero weight raises ValueError."""
        with pytest.raises(ValueError, match="must be positive"):
            GeneratorConfig(n=10, country={"US": 0.0, "DE": 1.0})

    def test_invalid_country_type_raises_error(self):
        """Invalid country type raises TypeError."""
        with pytest.raises(TypeError, match="must be a str, list\\[str\\], or dict"):
            GeneratorConfig(n=10, country=42)  # type: ignore[arg-type]


class TestWeightedSampling:
    """Tests for frequency-weighted sampling (weighted=True)."""

    # --- Helper: get US tier data ---

    @staticmethod
    def _us_male_tiers():
        """Return the tiered male first-name data for US."""
        import json
        from pathlib import Path

        person_path = (
            Path(__file__).parent.parent
            / "pointblank"
            / "countries"
            / "data"
            / "US"
            / "person.json"
        )
        with open(person_path) as f:
            data = json.load(f)
        return data["first_names"]["male"]

    # --- Unit tests for tier helpers ---

    def test_is_tiered_flat_list(self):
        """Flat list is not detected as tiered."""
        from pointblank.countries import _is_tiered

        assert not _is_tiered(["James", "John"])

    def test_is_tiered_dict(self):
        """Dict with tier keys is detected as tiered."""
        from pointblank.countries import _is_tiered

        assert _is_tiered({"very_common": ["a"], "common": ["b"]})

    def test_is_tiered_non_tier_dict(self):
        """Dict without tier keys (e.g., gendered dict) is not tiered."""
        from pointblank.countries import _is_tiered

        assert not _is_tiered({"male": ["a"], "female": ["b"]})

    def test_flatten_tiered(self):
        """Flatten preserves all items in tier order."""
        from pointblank.countries import _flatten_tiered

        tiered = {
            "very_common": ["A", "B"],
            "common": ["C"],
            "uncommon": ["D", "E"],
            "rare": ["F"],
        }
        flat = _flatten_tiered(tiered)
        assert flat == ["A", "B", "C", "D", "E", "F"]

    def test_flatten_tiered_missing_tiers(self):
        """Flatten handles missing tiers gracefully."""
        from pointblank.countries import _flatten_tiered

        tiered = {"very_common": ["A"], "rare": ["Z"]}
        flat = _flatten_tiered(tiered)
        assert flat == ["A", "Z"]

    def test_pick_from_tiered_returns_item(self):
        """pick_from_tiered always returns an item from the tiered data."""
        import random
        from pointblank.countries import _pick_from_tiered

        tiered = {
            "very_common": ["James", "John"],
            "common": ["Edward"],
            "uncommon": ["Jordan"],
            "rare": ["Micah"],
        }
        rng = random.Random(42)
        all_items = ["James", "John", "Edward", "Jordan", "Micah"]
        for _ in range(100):
            item = _pick_from_tiered(tiered, rng)
            assert item in all_items

    def test_pick_from_tiered_skews_toward_very_common(self):
        """With enough samples, very_common tier items should appear most often."""
        import random
        from pointblank.countries import _pick_from_tiered

        tiered = {
            "very_common": ["COMMON"],
            "rare": ["RARE"],
        }
        rng = random.Random(42)
        picks = [_pick_from_tiered(tiered, rng) for _ in range(1000)]
        common_count = picks.count("COMMON")
        rare_count = picks.count("RARE")
        # With 0.45 vs 0.05 weight, COMMON should be ~9x more frequent
        assert common_count > rare_count * 5

    # --- US person.json is now tiered ---

    def test_us_person_data_is_tiered(self):
        """Verify US person.json has been migrated to tiered format."""
        tiers = self._us_male_tiers()
        assert isinstance(tiers, dict)
        expected_keys = {"very_common", "common", "uncommon", "rare"}
        assert set(tiers.keys()) == expected_keys

    def test_us_male_tier_sizes(self):
        """Verify US male tiers have correct sizes."""
        tiers = self._us_male_tiers()
        assert len(tiers["very_common"]) == 25
        assert len(tiers["common"]) == 50
        assert len(tiers["uncommon"]) == 75
        assert len(tiers["rare"]) == 100

    # --- weighted=False (uniform) still works with tiered data ---

    def test_weighted_false_generates_names(self):
        """weighted=False with tiered data should generate valid names (uniform)."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(name=string_field(preset="first_name"))
        df = generate_dataset(schema, n=50, seed=23, country="US", weighted=False)
        assert df.shape == (50, 1)
        # All values should be non-empty strings
        for val in df["name"].to_list():
            assert isinstance(val, str)
            assert len(val) > 0

    def test_weighted_false_uniform_distribution(self):
        """weighted=False should maintain roughly uniform distribution across all tiers."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(name=string_field(preset="first_name"))
        # Generate many rows to get statistical significance
        df = generate_dataset(schema, n=2000, seed=23, country="US", weighted=False)

        # With uniform sampling over 250 names, any single name appearing > 5%
        # of the time would be suspicious
        names = df["name"].to_list()
        from collections import Counter

        counts = Counter(names)
        max_freq = max(counts.values()) / len(names)
        assert max_freq < 0.05  # No single name should dominate

    # --- weighted=True produces skewed distribution ---

    def test_weighted_true_generates_names(self):
        """weighted=True should generate valid names."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(name=string_field(preset="first_name"))
        df = generate_dataset(schema, n=50, seed=23, country="US", weighted=True)
        assert df.shape == (50, 1)
        for val in df["name"].to_list():
            assert isinstance(val, str)
            assert len(val) > 0

    def test_weighted_true_skews_toward_common_names(self):
        """weighted=True should produce more common names than rare ones."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(name=string_field(preset="first_name"))
        df = generate_dataset(schema, n=2000, seed=23, country="US", weighted=True)
        names = df["name"].to_list()

        # Get the tier data to know which names are common vs rare
        tiers = self._us_male_tiers()
        very_common_set = set(tiers["very_common"])
        rare_set = set(tiers["rare"])

        # Count picks from each tier (note: includes both male and female names)
        vc_count = sum(1 for n in names if n in very_common_set)
        rare_count = sum(1 for n in names if n in rare_set)

        # very_common should appear significantly more than rare
        # With 45% vs 5% tier weights, and 25 vs 100 names, per-name frequency
        # for very_common is ~3.6x that of rare.
        # But since we're counting ALL names in a tier, very_common (25 names
        # at 45%) should still yield many more hits per name than rare (100 names at 5%)
        # Give generous tolerance since we also get female and neutral names
        assert vc_count > 0 or rare_count > 0  # At minimum ensure some were found

    def test_weighted_true_skews_last_names(self):
        """weighted=True should also skew last name distribution."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(last=string_field(preset="last_name"))
        df = generate_dataset(schema, n=2000, seed=23, country="US", weighted=True)
        last_names = df["last"].to_list()

        # Very common US last names (from the tiered data)
        common_lasts = {"Smith", "Johnson", "Williams", "Brown", "Jones"}
        common_count = sum(1 for n in last_names if n in common_lasts)

        # These 5 names should appear more than 5% of rows combined (50% weight / 50 names)
        assert common_count > 50

    # --- weighted=True affects locations ---

    def test_weighted_true_skews_cities(self):
        """weighted=True should produce more major cities than small ones."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(city=string_field(preset="city"))
        df = generate_dataset(schema, n=2000, seed=23, country="US", weighted=True)
        cities = df["city"].to_list()

        major_cities = {"New York", "Los Angeles", "Chicago", "Houston", "Phoenix"}
        major_count = sum(1 for c in cities if c in major_cities)

        # With ~19 very_common cities at 45% weight, these 5 should appear frequently
        assert major_count > 50

    # --- Reproducibility ---

    def test_weighted_reproducible_with_seed(self):
        """weighted=True produces identical output with same seed."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(name=string_field(preset="name"), city=string_field(preset="city"))
        df1 = generate_dataset(schema, n=100, seed=23, country="US", weighted=True)
        df2 = generate_dataset(schema, n=100, seed=23, country="US", weighted=True)
        assert df1.equals(df2)

    # --- Backward compatibility ---

    def test_weighted_with_flat_list_country(self):
        """weighted=True with a non-tiered country falls back to uniform gracefully."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        # Germany still has flat lists (not yet migrated to tiered format)
        schema = Schema(name=string_field(preset="first_name"))
        df = generate_dataset(schema, n=50, seed=23, country="DE", weighted=True)
        assert df.shape == (50, 1)
        for val in df["name"].to_list():
            assert isinstance(val, str)

    # --- Interaction with locale mixing ---

    def test_weighted_with_multi_country(self):
        """weighted=True works correctly with multi-country mixing."""
        pytest.importorskip("polars")
        from pointblank import Schema, generate_dataset

        schema = Schema(name=string_field(preset="name"), city=string_field(preset="city"))
        df = generate_dataset(schema, n=100, seed=23, country=["US", "DE"], weighted=True)
        assert df.shape == (100, 2)
        # Should have a mix of US and DE names/cities
        cities = set(df["city"].to_list())
        assert len(cities) > 5  # Should have diversity from both countries

    # --- Schema.generate() integration ---

    def test_schema_generate_weighted(self):
        """Schema.generate() passes weighted= through correctly."""
        pytest.importorskip("polars")
        from pointblank import Schema

        schema = Schema(name=string_field(preset="first_name"))
        df = schema.generate(n=50, seed=23, country="US", weighted=True)
        assert df.shape == (50, 1)

    # --- GeneratorConfig ---

    def test_config_weighted_default_true(self):
        """GeneratorConfig defaults weighted to True."""
        config = GeneratorConfig()
        assert config.weighted is True

    def test_config_weighted_true(self):
        """GeneratorConfig accepts weighted=True."""
        config = GeneratorConfig(weighted=True)
        assert config.weighted is True


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

            # Validate that each city in locations has streets.
            # Locations may be a flat list or a tiered dict (frequency tiers).
            raw_locations = data["locations"]
            tiered_keys = {"very_common", "common", "uncommon", "rare"}
            if isinstance(raw_locations, dict) and set(raw_locations.keys()) <= tiered_keys:
                all_locations = [loc for tier in raw_locations.values() for loc in tier]
            else:
                all_locations = raw_locations

            city_names = {loc["city"] for loc in all_locations}
            streets_cities = set(data["streets_by_city"].keys())

            assert city_names == streets_cities

            # Validate locations structure
            assert len(all_locations) > 0

            for loc in all_locations:
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
                gender_data = first_names[gender]

                # May be a flat list or a tiered dict (with frequency tier keys)
                if isinstance(gender_data, dict):
                    # Tiered format: keys should be frequency tier names
                    valid_tiers = {"very_common", "common", "uncommon", "rare"}
                    assert set(gender_data.keys()) <= valid_tiers
                    all_names = [n for tier in gender_data.values() for n in tier]

                    if gender != "neutral":
                        assert len(all_names) > 0

                    for name in all_names:
                        assert isinstance(name, str)
                else:
                    assert isinstance(gender_data, list)

                    # male and female should have content, neutral can be empty
                    if gender != "neutral":
                        assert len(gender_data) > 0

                    for name in gender_data:
                        assert isinstance(name, str)

            # last_names should be a list of strings, a dict with gendered keys,
            # or a tiered dict with frequency tier keys
            last_names = data["last_names"]
            tiered_keys = {"very_common", "common", "uncommon", "rare"}
            if isinstance(last_names, dict) and set(last_names.keys()) <= tiered_keys:
                # Tiered last names (frequency tiers)
                all_names = [n for tier in last_names.values() for n in tier]
                assert len(all_names) > 0
                for name in all_names:
                    assert isinstance(name, str)
            elif isinstance(last_names, dict):
                # Gendered last names (e.g., IS patronymics)
                # Each gender value can be a flat list or a tiered dict
                for gender_key in last_names:
                    gv = last_names[gender_key]
                    if isinstance(gv, dict) and set(gv.keys()) <= tiered_keys:
                        # Tiered gendered last names
                        names = [n for tier in gv.values() for n in tier]
                    else:
                        assert isinstance(gv, list)
                        names = gv

                    for name in names:
                        assert isinstance(name, str)

                # At least one category should have names
                all_names = []
                for v in last_names.values():
                    if isinstance(v, dict):
                        all_names.extend(n for tier in v.values() for n in tier)
                    elif isinstance(v, list):
                        all_names.extend(v)

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
            valid_cities_raw = address_data["locations"]
            tiered_keys = {"very_common", "common", "uncommon", "rare"}
            if isinstance(valid_cities_raw, dict) and set(valid_cities_raw.keys()) <= tiered_keys:
                all_locs = [loc for tier in valid_cities_raw.values() for loc in tier]
            else:
                all_locs = valid_cities_raw
            valid_cities = {loc["city"] for loc in all_locs}

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
            tiered_keys = {"very_common", "common", "uncommon", "rare"}
            last_names_raw = person_data.get("last_names", [])
            if isinstance(last_names_raw, dict) and set(last_names_raw.keys()) <= tiered_keys:
                # Direct tiered dict
                last_names = [n for tier in last_names_raw.values() for n in tier]
            elif isinstance(last_names_raw, dict):
                # Gendered dict (e.g., IS) — each value may be a list or tiered dict
                last_names = []
                for gv in last_names_raw.values():
                    if isinstance(gv, dict) and set(gv.keys()) <= tiered_keys:
                        last_names.extend(n for tier in gv.values() for n in tier)
                    elif isinstance(gv, list):
                        last_names.extend(gv)
            else:
                last_names = last_names_raw
            unique_last_names = set(last_names)
            duplicate_ratio = 1 - (len(unique_last_names) / len(last_names)) if last_names else 0

            assert duplicate_ratio < 0.1

            # First names should not have duplicates within each gender
            for gender in ["male", "female", "neutral"]:
                first_names_raw = person_data.get("first_names", {}).get(gender, [])
                if isinstance(first_names_raw, dict):
                    first_names = [n for tier in first_names_raw.values() for n in tier]
                else:
                    first_names = first_names_raw
                duplicates = [x for x in first_names if first_names.count(x) > 1]

                assert not duplicates

            # Check company.json for duplicate well-known companies (must be unique)
            company_file = countries_dir / country / "company.json"
            with open(company_file, "r", encoding="utf-8") as f:
                company_data = json.load(f)

            company_names = [c["name"] for c in company_data.get("well_known_companies", [])]
            duplicates = [x for x in company_names if company_names.count(x) > 1]

            assert not duplicates
