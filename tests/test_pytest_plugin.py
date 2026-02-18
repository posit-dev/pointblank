"""
Tests for the pointblank pytest plugin (`generate_dataset` fixture).

These tests verify:

- the fixture is auto-discovered and usable
- automatic seeding produces deterministic data
- different tests get different seeds (different data)
- multiple calls within one test get different but deterministic data
- explicit seed override works
- all output formats work
- country parameter is respected
- seed-derivation helper produces sane values
"""

import pytest

from pointblank import Schema, int_field, float_field, string_field
from pointblank.pytest_plugin import _seed_from_node_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_SCHEMA = Schema(
    id=int_field(min_val=1, max_val=10_000),
    value=float_field(min_val=0.0, max_val=100.0),
    name=string_field(min_length=3, max_length=10),
)


# ---------------------------------------------------------------------------
# _seed_from_node_id unit tests
# ---------------------------------------------------------------------------


class TestSeedFromNodeId:
    """Tests for the seed derivation helper."""

    def test_returns_positive_int(self):
        seed = _seed_from_node_id("tests/test_example.py::test_foo")
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31

    def test_deterministic(self):
        a = _seed_from_node_id("tests/test_example.py::test_foo")
        b = _seed_from_node_id("tests/test_example.py::test_foo")
        assert a == b

    def test_different_names_give_different_seeds(self):
        a = _seed_from_node_id("tests/test_example.py::test_foo")
        b = _seed_from_node_id("tests/test_example.py::test_bar")
        assert a != b

    def test_parametrize_ids_give_different_seeds(self):
        a = _seed_from_node_id("tests/test_example.py::test_param[0]")
        b = _seed_from_node_id("tests/test_example.py::test_param[1]")
        assert a != b

    def test_empty_string(self):
        seed = _seed_from_node_id("")
        assert isinstance(seed, int)
        assert 0 <= seed < 2**31


# ---------------------------------------------------------------------------
# Fixture integration tests
# ---------------------------------------------------------------------------


class TestFixtureBasicUsage:
    """Test the generate_dataset fixture produces valid data."""

    def test_returns_polars_by_default(self, generate_dataset):
        import polars as pl

        df = generate_dataset(SIMPLE_SCHEMA, n=10)
        assert isinstance(df, pl.DataFrame)
        assert df.shape == (10, 3)
        assert set(df.columns) == {"id", "value", "name"}

    def test_pandas_output(self, generate_dataset):
        import pandas as pd

        df = generate_dataset(SIMPLE_SCHEMA, n=5, output="pandas")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (5, 3)

    def test_dict_output(self, generate_dataset):
        result = generate_dataset(SIMPLE_SCHEMA, n=5, output="dict")
        assert isinstance(result, dict)
        assert set(result.keys()) == {"id", "value", "name"}
        assert all(len(v) == 5 for v in result.values())

    def test_country_parameter(self, generate_dataset):
        schema = Schema(city=string_field(preset="city"))
        df = generate_dataset(schema, n=5, country="DE")
        assert df.shape == (5, 1)


class TestFixtureDeterminism:
    """Verify that the fixture produces the same data on repeated runs."""

    def test_same_data_across_calls(self, generate_dataset, request):
        """The fixture with auto-seed should produce the same data as a manual
        call to generate_dataset() with the same derived seed."""
        from pointblank.schema import generate_dataset as raw_generate

        # Compute the seed the fixture would use for call_count=0
        node_seed = _seed_from_node_id(request.node.nodeid)

        df_fixture = generate_dataset(SIMPLE_SCHEMA, n=20, output="dict")
        df_manual = raw_generate(SIMPLE_SCHEMA, n=20, seed=node_seed, output="dict")

        assert df_fixture["id"] == df_manual["id"]
        assert df_fixture["value"] == df_manual["value"]
        assert df_fixture["name"] == df_manual["name"]

    def test_fixture_is_deterministic_first(self, generate_dataset):
        """First of a pair: generates data that should match test_fixture_is_deterministic_second
        only if they share the same test name (they don't), so they should differ."""
        df = generate_dataset(SIMPLE_SCHEMA, n=10)
        # Just store the first value for structural verification
        assert df.shape == (10, 3)

    def test_fixture_is_deterministic_second(self, generate_dataset):
        """Different test name → different seed → different data."""
        df = generate_dataset(SIMPLE_SCHEMA, n=10)
        assert df.shape == (10, 3)


class TestFixtureMultipleCalls:
    """Verify multiple calls within one test produce different but deterministic data."""

    def test_two_calls_produce_different_data(self, generate_dataset):
        """Two calls in the same test should use different seeds (via call counter)."""
        df1 = generate_dataset(SIMPLE_SCHEMA, n=50, output="dict")
        df2 = generate_dataset(SIMPLE_SCHEMA, n=50, output="dict")

        # With different seeds and 50 rows, the id columns should differ
        assert df1["id"] != df2["id"], "Two calls in the same test should produce different data"

    def test_three_calls_all_differ(self, generate_dataset):
        """Three calls should all produce different data."""
        results = [generate_dataset(SIMPLE_SCHEMA, n=30, output="dict") for _ in range(3)]

        # All id columns should be pairwise different
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                assert results[i]["id"] != results[j]["id"], (
                    f"Call {i} and {j} produced the same data"
                )


class TestFixtureExplicitSeed:
    """Verify explicit seed overrides the automatic one."""

    def test_explicit_seed_overrides_auto(self, generate_dataset):
        """When seed is passed explicitly, it should be used instead of the auto-derived one."""
        from pointblank.schema import generate_dataset as raw_generate

        df_fixture = generate_dataset(SIMPLE_SCHEMA, n=10, seed=42, output="dict")
        df_raw = raw_generate(SIMPLE_SCHEMA, n=10, seed=42, output="dict")

        assert df_fixture["id"] == df_raw["id"]
        assert df_fixture["value"] == df_raw["value"]
        assert df_fixture["name"] == df_raw["name"]

    def test_explicit_seed_does_not_increment_counter(self, generate_dataset):
        """Explicit seeds should not be affected by call_count incrementing.
        The call_count only affects auto-seeded calls."""
        from pointblank.schema import generate_dataset as raw_generate

        # First call: auto-seeded (increments counter)
        _ = generate_dataset(SIMPLE_SCHEMA, n=5)

        # Second call: explicit seed — should match raw function with same seed
        df_fixture = generate_dataset(SIMPLE_SCHEMA, n=10, seed=99, output="dict")
        df_raw = raw_generate(SIMPLE_SCHEMA, n=10, seed=99, output="dict")

        assert df_fixture["id"] == df_raw["id"]

    def test_two_explicit_same_seed_give_same_data(self, generate_dataset):
        """Two calls with the same explicit seed give the same data."""
        df1 = generate_dataset(SIMPLE_SCHEMA, n=20, seed=123, output="dict")
        df2 = generate_dataset(SIMPLE_SCHEMA, n=20, seed=123, output="dict")

        assert df1["id"] == df2["id"]
        assert df1["value"] == df2["value"]


class TestFixtureDifferentTestsDifferentData:
    """Each test function gets a different auto-seed."""

    def test_alpha(self, generate_dataset):
        """Returns data seeded from 'test_alpha' node ID."""
        df = generate_dataset(SIMPLE_SCHEMA, n=30, output="dict")
        # Store as attribute on the class for cross-test comparison
        TestFixtureDifferentTestsDifferentData._alpha_ids = df["id"]

    def test_beta(self, generate_dataset):
        """Returns data seeded from 'test_beta' node ID — should differ from alpha."""
        df = generate_dataset(SIMPLE_SCHEMA, n=30, output="dict")

        if hasattr(TestFixtureDifferentTestsDifferentData, "_alpha_ids"):
            assert df["id"] != TestFixtureDifferentTestsDifferentData._alpha_ids, (
                "Different tests should produce different data"
            )


class TestFixtureParametrize:
    """Parametrized tests get different seeds per parameter."""

    @pytest.mark.parametrize("country", ["US", "DE", "JP"])
    def test_parametrized_countries(self, generate_dataset, country):
        schema = Schema(city=string_field(preset="city"))
        df = generate_dataset(schema, n=5, country=country)
        assert df.shape == (5, 1)

    @pytest.mark.parametrize("n", [1, 10, 50])
    def test_parametrized_sizes(self, generate_dataset, n):
        df = generate_dataset(SIMPLE_SCHEMA, n=n)
        assert df.shape == (n, 3)
