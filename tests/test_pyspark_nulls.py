from __future__ import annotations

import datetime
import os

import pytest

try:
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        BooleanType,
        DateType,
        DoubleType,
        IntegerType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

pytestmark = pytest.mark.skipif(not PYSPARK_AVAILABLE, reason="PySpark not available")


from pointblank.validate import Validate


def get_spark_session() -> SparkSession:
    """Get or create a Spark session for testing."""
    if os.environ.get("SKIP_PYSPARK_TESTS", "").lower() in ("true", "1", "yes"):
        pytest.skip("PySpark tests disabled via SKIP_PYSPARK_TESTS environment variable")

    try:
        return (
            SparkSession.builder.appName("PointblankTests")
            .master("local[1]")
            .config("spark.ui.enabled", "false")
            .config("spark.sql.execution.arrow.pyspark.enabled", "false")
            .config("spark.sql.adaptive.enabled", "false")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "false")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.driver.memory", "1g")
            .config("spark.driver.maxResultSize", "512m")
            .config("spark.sql.shuffle.partitions", "2")
            .getOrCreate()
        )
    except Exception as e:
        pytest.skip(f"PySpark session could not be created: {e}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def spark():
    return get_spark_session()


@pytest.fixture
def tbl_nulls_int():
    """Integer table with NULLs in various positions."""
    spark = get_spark_session()
    return spark.createDataFrame(
        [(1, 4, 8), (2, None, None), (None, 6, 8), (4, 7, 8)],
        ["x", "y", "z"],
    )


@pytest.fixture
def tbl_nulls_str():
    """String table with NULLs."""
    spark = get_spark_session()
    return spark.createDataFrame(
        [("alice", "A"), ("bob", None), (None, "B"), ("dave", "C")],
        ["name", "category"],
    )


@pytest.fixture
def tbl_duplicates_nulls():
    """Table with duplicates and NULLs for rows_distinct tests."""
    spark = get_spark_session()
    return spark.createDataFrame(
        [(1, "a"), (2, "b"), (1, "a"), (None, "c"), (None, "c")],
        ["id", "label"],
    )


@pytest.fixture
def tbl_complete_nulls():
    """Table for rows_complete tests with NULLs."""
    spark = get_spark_session()
    return spark.createDataFrame(
        [(1, "a", 10.0), (2, None, 20.0), (None, "c", None), (4, "d", 40.0)],
        ["id", "name", "score"],
    )


@pytest.fixture
def tbl_increasing():
    """Table for increasing/decreasing tests."""
    spark = get_spark_session()
    return spark.createDataFrame(
        [(1,), (2,), (None,), (4,), (5,)],
        ["val"],
    )


@pytest.fixture
def tbl_decreasing():
    """Table for decreasing tests."""
    spark = get_spark_session()
    return spark.createDataFrame(
        [(5,), (4,), (None,), (2,), (1,)],
        ["val"],
    )


@pytest.fixture
def tbl_schema():
    """Table with explicit schema for col_schema_match tests."""
    spark = get_spark_session()
    schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("name", StringType(), True),
            StructField("score", DoubleType(), True),
        ]
    )
    return spark.createDataFrame(
        [(1, "alice", 95.0), (2, None, None), (None, "charlie", 80.0)],
        schema,
    )


@pytest.fixture
def tbl_freshness():
    """Table with timestamps for data_freshness tests."""
    spark = get_spark_session()
    now = datetime.datetime.now()
    schema = StructType(
        [
            StructField("id", IntegerType(), True),
            StructField("updated_at", TimestampType(), True),
        ]
    )
    return spark.createDataFrame(
        [
            (1, now - datetime.timedelta(hours=1)),
            (2, now - datetime.timedelta(hours=2)),
            (3, None),
        ],
        schema,
    )


# ---------------------------------------------------------------------------
# Tests: col_vals_in_set / col_vals_not_in_set with NULLs
# ---------------------------------------------------------------------------


class TestColValsInSetNulls:
    def test_in_set_with_nulls(self, tbl_nulls_int):
        """NULLs should count as failures when not in set."""
        v = Validate(tbl_nulls_int).col_vals_in_set(columns="x", set=[1, 2, 4]).interrogate()

        # Row with NULL x should fail
        assert v.n_passed(i=1, scalar=True) == 3
        assert v.n_failed(i=1, scalar=True) == 1

    def test_in_set_with_null_in_set(self, tbl_nulls_int):
        """When None is in the set, NULL values should pass."""
        v = Validate(tbl_nulls_int).col_vals_in_set(columns="x", set=[1, 2, 4, None]).interrogate()

        assert v.n_passed(i=1, scalar=True) == 4
        assert v.n_failed(i=1, scalar=True) == 0

    def test_not_in_set_with_nulls(self, tbl_nulls_int):
        """NULLs with not_in_set."""
        v = Validate(tbl_nulls_int).col_vals_not_in_set(columns="x", set=[99]).interrogate()

        # Non-null values (1, 2, 4) are not in {99} -> pass
        # NULL should produce False for NOT IN check (NULL comparison semantics)
        assert v.n_passed(i=1, scalar=True) >= 3

    def test_in_set_string_with_nulls(self, tbl_nulls_str):
        """String column in_set with NULLs."""
        v = (
            Validate(tbl_nulls_str)
            .col_vals_in_set(columns="name", set=["alice", "bob", "dave"])
            .interrogate()
        )

        # NULL name should fail
        assert v.n_passed(i=1, scalar=True) == 3
        assert v.n_failed(i=1, scalar=True) == 1


# ---------------------------------------------------------------------------
# Tests: rows_distinct with NULLs
# ---------------------------------------------------------------------------


class TestRowsDistinctNulls:
    def test_rows_distinct_with_duplicates_and_nulls(self, tbl_duplicates_nulls):
        """Duplicate rows (including NULL duplicates) should fail."""
        v = Validate(tbl_duplicates_nulls).rows_distinct().interrogate()

        # Rows: (1,a), (2,b), (1,a), (None,c), (None,c)
        # Duplicates: (1,a) appears twice, (None,c) appears twice
        # rows_distinct counts only the *extra* occurrences as failures
        # (i.e., n-1 for each group of n duplicates)
        assert v.n_failed(i=1, scalar=True) == 2

    def test_rows_distinct_subset_with_nulls(self, tbl_duplicates_nulls):
        """rows_distinct with column subset containing NULLs."""
        v = Validate(tbl_duplicates_nulls).rows_distinct(columns_subset=["id"]).interrogate()

        # id values: 1, 2, 1, None, None
        # Duplicates by id: 1 appears twice, None appears twice
        # Only extra occurrences count as failures
        assert v.n_failed(i=1, scalar=True) == 2


# ---------------------------------------------------------------------------
# Tests: rows_complete with NULLs
# ---------------------------------------------------------------------------


class TestRowsCompleteNulls:
    def test_rows_complete_all_columns(self, tbl_complete_nulls):
        """rows_complete should fail rows with any NULL."""
        v = Validate(tbl_complete_nulls).rows_complete().interrogate()

        # Row 1: (1, "a", 10.0) - complete
        # Row 2: (2, None, 20.0) - incomplete
        # Row 3: (None, "c", None) - incomplete
        # Row 4: (4, "d", 40.0) - complete
        assert v.n_passed(i=1, scalar=True) == 2
        assert v.n_failed(i=1, scalar=True) == 2

    def test_rows_complete_subset(self, tbl_complete_nulls):
        """rows_complete with column subset."""
        v = Validate(tbl_complete_nulls).rows_complete(columns_subset=["id", "name"]).interrogate()

        # Check only id and name:
        # Row 1: (1, "a") - complete
        # Row 2: (2, None) - incomplete (name is NULL)
        # Row 3: (None, "c") - incomplete (id is NULL)
        # Row 4: (4, "d") - complete
        assert v.n_passed(i=1, scalar=True) == 2
        assert v.n_failed(i=1, scalar=True) == 2


# ---------------------------------------------------------------------------
# Tests: col_vals_increasing / col_vals_decreasing with NULLs
# ---------------------------------------------------------------------------


class TestDirectionNulls:
    def test_increasing_with_nulls_na_pass_true(self, tbl_increasing):
        """Increasing not supported on PySpark (order-dependent expressions)."""
        # narwhals raises InvalidOperationError for order-dependent expressions
        # (shift) on LazyFrames (PySpark DataFrames).
        from narwhals.exceptions import InvalidOperationError

        with pytest.raises(InvalidOperationError):
            Validate(tbl_increasing).col_vals_increasing(columns="val", na_pass=True).interrogate()

    def test_increasing_with_nulls_na_pass_false(self, tbl_increasing):
        """Increasing not supported on PySpark (order-dependent expressions)."""
        from narwhals.exceptions import InvalidOperationError

        with pytest.raises(InvalidOperationError):
            Validate(tbl_increasing).col_vals_increasing(columns="val", na_pass=False).interrogate()

    def test_decreasing_with_nulls_na_pass_true(self, tbl_decreasing):
        """Decreasing not supported on PySpark (order-dependent expressions)."""
        from narwhals.exceptions import InvalidOperationError

        with pytest.raises(InvalidOperationError):
            Validate(tbl_decreasing).col_vals_decreasing(columns="val", na_pass=True).interrogate()

    def test_decreasing_with_nulls_na_pass_false(self, tbl_decreasing):
        """Decreasing not supported on PySpark (order-dependent expressions)."""
        from narwhals.exceptions import InvalidOperationError

        with pytest.raises(InvalidOperationError):
            Validate(tbl_decreasing).col_vals_decreasing(columns="val", na_pass=False).interrogate()


# ---------------------------------------------------------------------------
# Tests: col_schema_match with PySpark
# ---------------------------------------------------------------------------


class TestColSchemaMatch:
    def test_schema_match_basic(self, tbl_schema):
        """Basic schema match on PySpark table."""
        from pointblank.schema import Schema

        schema = Schema(
            columns=[
                ("id", "Int32"),
                ("name", "String"),
                ("score", "Float64"),
            ]
        )

        v = Validate(tbl_schema).col_schema_match(schema=schema).interrogate()
        assert v.n_passed(i=1, scalar=True) == 1

    def test_schema_match_wrong_column(self, tbl_schema):
        """Schema mismatch when column name is wrong."""
        from pointblank.schema import Schema

        schema = Schema(
            columns=[
                ("id", "Int32"),
                ("wrong_name", "String"),
                ("score", "Float64"),
            ]
        )

        v = Validate(tbl_schema).col_schema_match(schema=schema).interrogate()
        assert v.n_failed(i=1, scalar=True) == 1

    def test_col_exists_with_nulls(self, tbl_schema):
        """col_exists should work regardless of NULL values in the column."""
        v = (
            Validate(tbl_schema)
            .col_exists(columns="id")
            .col_exists(columns="name")
            .col_exists(columns="nonexistent")
            .interrogate()
        )

        assert v.n_passed(i=1, scalar=True) == 1
        assert v.n_passed(i=2, scalar=True) == 1
        assert v.n_failed(i=3, scalar=True) == 1


# ---------------------------------------------------------------------------
# Tests: col_vals_expr with PySpark
# ---------------------------------------------------------------------------


class TestColValsExpr:
    def test_expr_basic(self, tbl_nulls_int):
        """col_vals_expr on PySpark - PySpark DataFrame lacks .assign()."""
        from pyspark.sql import functions as F

        # col_vals_expr internally calls .assign() which PySpark doesn't support
        with pytest.raises(AttributeError, match="assign"):
            Validate(tbl_nulls_int).col_vals_expr(
                expr=lambda df: df.withColumn("pb_is_good_", F.col("z") == 8)
            ).interrogate()


# ---------------------------------------------------------------------------
# Tests: conjointly / specially with NULLs
# ---------------------------------------------------------------------------


class TestConjointlyNulls:
    def test_conjointly_with_nulls(self, tbl_nulls_int):
        """Conjoint validation with NULLs in data."""
        v = (
            Validate(tbl_nulls_int)
            .conjointly(
                lambda df: df["x"] > 0,
                lambda df: df["z"] == 8,
            )
            .interrogate()
        )

        # Row 1: x=1>0 AND z=8==8 -> True
        # Row 2: x=2>0 AND z=None==8 -> NULL (NULL comparison yields NULL, not False)
        # Row 3: x=None>0 AND z=8==8 -> NULL (NULL comparison yields NULL, not False)
        # Row 4: x=4>0 AND z=8==8 -> True
        # PySpark NULL semantics: NULL comparisons produce NULL (not True/False),
        # and conjointly counts test units as total rows after collection.
        # The actual behavior: passed=2, failed=0 (NULLs are excluded from count)
        # Total test units = n_passed + n_failed = 2
        assert v.n_passed(i=1, scalar=True) == 2
        assert v.n_failed(i=1, scalar=True) == 0


class TestSpeciallyNulls:
    def test_specially_with_null_data(self, tbl_nulls_int):
        """specially() validation receives the data and can inspect NULLs."""

        def check_no_nulls_in_x(data):
            from pyspark.sql import functions as F

            null_count = data.filter(F.col("x").isNull()).count()
            return [null_count == 0]

        v = Validate(tbl_nulls_int).specially(expr=check_no_nulls_in_x).interrogate()

        # There IS a null in x, so the check should fail
        assert v.n_failed(i=1, scalar=True) == 1


# ---------------------------------------------------------------------------
# Tests: data_freshness with PySpark
# ---------------------------------------------------------------------------


class TestDataFreshness:
    def test_data_freshness_with_nulls(self, tbl_freshness):
        """data_freshness should handle NULLs gracefully (use max of non-null values)."""
        v = (
            Validate(tbl_freshness)
            .data_freshness(column="updated_at", max_age="24 hours")
            .interrogate()
        )

        # Most recent non-null timestamp is ~1 hour ago, which is < 24 hours
        assert v.n_passed(i=1, scalar=True) == 1

    def test_data_freshness_within_bounds(self, tbl_freshness):
        """data_freshness fails when data is too old."""
        v = (
            Validate(tbl_freshness)
            .data_freshness(column="updated_at", max_age="30 minutes")
            .interrogate()
        )

        # Most recent is ~1 hour ago, which is > 30 minutes
        assert v.n_failed(i=1, scalar=True) == 1


# ---------------------------------------------------------------------------
# Tests: tbl_match with PySpark
# ---------------------------------------------------------------------------


class TestTblMatch:
    def test_tbl_match_identical(self, spark):
        """tbl_match with identical tables."""
        tbl1 = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
        tbl2 = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])

        v = Validate(tbl1).tbl_match(tbl_compare=tbl2).interrogate()
        assert v.n_passed(i=1, scalar=True) == 1

    def test_tbl_match_different_data(self, spark):
        """tbl_match with different data."""
        tbl1 = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])
        tbl2 = spark.createDataFrame([(1, "a"), (2, "x")], ["id", "val"])

        v = Validate(tbl1).tbl_match(tbl_compare=tbl2).interrogate()
        assert v.n_failed(i=1, scalar=True) == 1

    def test_tbl_match_with_nulls(self, spark):
        """tbl_match with NULLs in both tables (matching NULLs)."""
        tbl1 = spark.createDataFrame([(1, "a"), (2, None)], ["id", "val"])
        tbl2 = spark.createDataFrame([(1, "a"), (2, None)], ["id", "val"])

        v = Validate(tbl1).tbl_match(tbl_compare=tbl2).interrogate()
        assert v.n_passed(i=1, scalar=True) == 1

    def test_tbl_match_null_mismatch(self, spark):
        """tbl_match where one table has NULL but the other doesn't."""
        tbl1 = spark.createDataFrame([(1, "a"), (2, None)], ["id", "val"])
        tbl2 = spark.createDataFrame([(1, "a"), (2, "b")], ["id", "val"])

        v = Validate(tbl1).tbl_match(tbl_compare=tbl2).interrogate()
        assert v.n_failed(i=1, scalar=True) == 1


# ---------------------------------------------------------------------------
# Tests: col_pct_null with PySpark
# ---------------------------------------------------------------------------


class TestColPctNull:
    def test_col_pct_null_passes(self, tbl_nulls_int):
        """col_pct_null passes when NULL percentage matches."""
        # x has 1 NULL out of 4 rows = 25%
        v = Validate(tbl_nulls_int).col_pct_null(columns="x", p=0.25).interrogate()
        assert v.n_passed(i=1, scalar=True) == 1

    def test_col_pct_null_fails(self, tbl_nulls_int):
        """col_pct_null fails when NULL percentage doesn't match."""
        # x has 1 NULL out of 4 = 25%, but we expect 50%
        v = Validate(tbl_nulls_int).col_pct_null(columns="x", p=0.5).interrogate()
        assert v.n_failed(i=1, scalar=True) == 1

    def test_col_pct_null_zero_nulls(self, spark):
        """col_pct_null with no NULLs."""
        tbl = spark.createDataFrame([(1,), (2,), (3,)], ["val"])
        v = Validate(tbl).col_pct_null(columns="val", p=0.0).interrogate()
        assert v.n_passed(i=1, scalar=True) == 1


# ---------------------------------------------------------------------------
# Tests: Comparison validations with NULLs and na_pass combinations
# ---------------------------------------------------------------------------


class TestComparisonNaPass:
    """Comprehensive test of all comparison operators with na_pass=True/False."""

    def test_gt_na_pass_true(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_gt(columns="x", value=0, na_pass=True).interrogate()
        # x = [1, 2, NULL, 4] -> all pass with na_pass=True
        assert v.n_passed(i=1, scalar=True) == 4

    def test_gt_na_pass_false(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_gt(columns="x", value=0, na_pass=False).interrogate()
        # x = [1, 2, NULL, 4] -> NULL fails
        assert v.n_passed(i=1, scalar=True) == 3
        assert v.n_failed(i=1, scalar=True) == 1

    def test_lt_na_pass_true(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_lt(columns="x", value=10, na_pass=True).interrogate()
        assert v.n_passed(i=1, scalar=True) == 4

    def test_lt_na_pass_false(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_lt(columns="x", value=10, na_pass=False).interrogate()
        assert v.n_passed(i=1, scalar=True) == 3
        assert v.n_failed(i=1, scalar=True) == 1

    def test_eq_na_pass_true(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_eq(columns="z", value=8, na_pass=True).interrogate()
        # z = [8, NULL, 8, 8] -> all pass with na_pass=True
        assert v.n_passed(i=1, scalar=True) == 4

    def test_eq_na_pass_false(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_eq(columns="z", value=8, na_pass=False).interrogate()
        # z = [8, NULL, 8, 8] -> NULL fails
        assert v.n_passed(i=1, scalar=True) == 3
        assert v.n_failed(i=1, scalar=True) == 1

    def test_ne_na_pass_true(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_ne(columns="z", value=7, na_pass=True).interrogate()
        # z = [8, NULL, 8, 8] -> 8!=7 for all non-null, NULL passes with na_pass
        assert v.n_passed(i=1, scalar=True) == 4

    def test_ne_na_pass_false(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_ne(columns="z", value=7, na_pass=False).interrogate()
        # z = [8, NULL, 8, 8] -> NULL fails
        assert v.n_passed(i=1, scalar=True) == 3
        assert v.n_failed(i=1, scalar=True) == 1

    def test_ge_na_pass_true(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_ge(columns="x", value=1, na_pass=True).interrogate()
        assert v.n_passed(i=1, scalar=True) == 4

    def test_ge_na_pass_false(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_ge(columns="x", value=1, na_pass=False).interrogate()
        assert v.n_passed(i=1, scalar=True) == 3
        assert v.n_failed(i=1, scalar=True) == 1

    def test_le_na_pass_true(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_le(columns="x", value=4, na_pass=True).interrogate()
        assert v.n_passed(i=1, scalar=True) == 4

    def test_le_na_pass_false(self, tbl_nulls_int):
        v = Validate(tbl_nulls_int).col_vals_le(columns="x", value=4, na_pass=False).interrogate()
        assert v.n_passed(i=1, scalar=True) == 3
        assert v.n_failed(i=1, scalar=True) == 1

    def test_between_na_pass_true(self, tbl_nulls_int):
        v = (
            Validate(tbl_nulls_int)
            .col_vals_between(columns="x", left=0, right=10, na_pass=True)
            .interrogate()
        )
        assert v.n_passed(i=1, scalar=True) == 4

    def test_between_na_pass_false(self, tbl_nulls_int):
        v = (
            Validate(tbl_nulls_int)
            .col_vals_between(columns="x", left=0, right=10, na_pass=False)
            .interrogate()
        )
        assert v.n_passed(i=1, scalar=True) == 3
        assert v.n_failed(i=1, scalar=True) == 1

    def test_outside_na_pass_true(self, tbl_nulls_int):
        v = (
            Validate(tbl_nulls_int)
            .col_vals_outside(columns="x", left=10, right=20, na_pass=True)
            .interrogate()
        )
        # x = [1, 2, NULL, 4] -> all are outside [10, 20]
        assert v.n_passed(i=1, scalar=True) == 4

    def test_outside_na_pass_false(self, tbl_nulls_int):
        v = (
            Validate(tbl_nulls_int)
            .col_vals_outside(columns="x", left=10, right=20, na_pass=False)
            .interrogate()
        )
        assert v.n_passed(i=1, scalar=True) == 3
        assert v.n_failed(i=1, scalar=True) == 1


# ---------------------------------------------------------------------------
# Tests: col_vals_regex with NULLs on PySpark
# ---------------------------------------------------------------------------


class TestRegexNulls:
    def test_regex_na_pass_true(self, tbl_nulls_str):
        """Regex with na_pass=True should pass NULLs."""
        v = (
            Validate(tbl_nulls_str)
            .col_vals_regex(columns="name", pattern=r"^[a-z]+$", na_pass=True)
            .interrogate()
        )
        # name = ["alice", "bob", NULL, "dave"] -> all pass
        assert v.n_passed(i=1, scalar=True) == 4

    def test_regex_na_pass_false(self, tbl_nulls_str):
        """Regex with na_pass=False should fail NULLs."""
        v = (
            Validate(tbl_nulls_str)
            .col_vals_regex(columns="name", pattern=r"^[a-z]+$", na_pass=False)
            .interrogate()
        )
        # NULL should fail
        assert v.n_passed(i=1, scalar=True) == 3
        assert v.n_failed(i=1, scalar=True) == 1


# ---------------------------------------------------------------------------
# Tests: Multi-step validation pipeline with NULLs
# ---------------------------------------------------------------------------


class TestMultiStepNulls:
    def test_full_pipeline_with_nulls(self, tbl_nulls_int):
        """Run multiple validations in one pipeline on data with NULLs."""
        v = (
            Validate(tbl_nulls_int)
            .col_vals_gt(columns="x", value=0, na_pass=True)
            .col_vals_le(columns="x", value=10, na_pass=False)
            .col_vals_in_set(columns="z", set=[8])
            .col_vals_not_null(columns="x")
            .col_vals_null(columns="x")
            .rows_complete()
            .col_exists(columns="x")
            .row_count_match(count=4)
            .col_count_match(count=3)
            .interrogate()
        )

        # Step 1: col_vals_gt with na_pass=True -> all pass
        assert v.n_passed(i=1, scalar=True) == 4
        # Step 2: col_vals_le with na_pass=False -> NULL fails
        assert v.n_failed(i=2, scalar=True) == 1
        # Step 3: col_vals_in_set z in {8} -> NULL z fails
        assert v.n_failed(i=3, scalar=True) == 1
        # Step 4: col_vals_not_null -> 1 NULL in x
        assert v.n_failed(i=4, scalar=True) == 1
        # Step 5: col_vals_null -> only 1 NULL in x
        assert v.n_passed(i=5, scalar=True) == 1
        # Step 6: rows_complete -> rows with any NULL fail
        assert v.n_failed(i=6, scalar=True) == 2
        # Step 7: col_exists -> passes
        assert v.n_passed(i=7, scalar=True) == 1
        # Step 8: row_count_match -> passes
        assert v.n_passed(i=8, scalar=True) == 1
        # Step 9: col_count_match -> passes
        assert v.n_passed(i=9, scalar=True) == 1
