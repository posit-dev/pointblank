import pandas as pd
import polars as pl
import pytest

import pointblank as pb


# ── Single-column FK ──────────────────────────────────────────────────────────


def test_single_col_all_pass():
    ref = pl.DataFrame({"id": [1, 2, 3, 4, 5]})
    tbl = pl.DataFrame({"customer_id": [1, 2, 3]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="customer_id", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 3
    assert v.n_failed(i=1, scalar=True) == 0


def test_single_col_some_fail():
    ref = pl.DataFrame({"id": [1, 2, 3]})
    tbl = pl.DataFrame({"customer_id": [1, 2, 99, 100]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="customer_id", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 2


def test_single_col_all_fail():
    ref = pl.DataFrame({"id": [1, 2, 3]})
    tbl = pl.DataFrame({"customer_id": [10, 20, 30]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="customer_id", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 0
    assert v.n_failed(i=1, scalar=True) == 3


def test_single_col_string_values():
    ref = pl.DataFrame({"code": ["A", "B", "C"]})
    tbl = pl.DataFrame({"product_code": ["A", "B", "D"]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="product_code", ref_table=ref, ref_column="code")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


def test_same_column_name():
    ref = pl.DataFrame({"id": [1, 2, 3]})
    tbl = pl.DataFrame({"id": [1, 2, 99]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="id", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


# ── na_pass behavior ─────────────────────────────────────────────────────────


def test_na_pass_false_nulls_fail():
    ref = pl.DataFrame({"id": [1, 2, 3]})
    tbl = pl.DataFrame({"fk": [1, None, 3]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id", na_pass=False)
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


def test_na_pass_true_nulls_pass():
    ref = pl.DataFrame({"id": [1, 2, 3]})
    tbl = pl.DataFrame({"fk": [1, None, 3]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id", na_pass=True)
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 3
    assert v.n_failed(i=1, scalar=True) == 0


def test_na_pass_true_with_failing_rows():
    ref = pl.DataFrame({"id": [1, 2, 3]})
    tbl = pl.DataFrame({"fk": [1, None, 99]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id", na_pass=True)
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


# ── Composite keys ───────────────────────────────────────────────────────────


def test_composite_key_all_pass():
    ref = pl.DataFrame({"region": ["US", "US", "EU"], "sku": ["A1", "B2", "A1"]})
    tbl = pl.DataFrame({"region": ["US", "EU"], "sku": ["A1", "A1"]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(
            columns=["region", "sku"],
            ref_table=ref,
            ref_column=["region", "sku"],
        )
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 0


def test_composite_key_some_fail():
    ref = pl.DataFrame({"region": ["US", "US", "EU"], "sku": ["A1", "B2", "A1"]})
    tbl = pl.DataFrame({"region": ["US", "EU", "US"], "sku": ["A1", "A1", "C3"]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(
            columns=["region", "sku"],
            ref_table=ref,
            ref_column=["region", "sku"],
        )
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


def test_composite_key_different_column_names():
    ref = pl.DataFrame({"r": ["US", "EU"], "s": ["A1", "A1"]})
    tbl = pl.DataFrame({"region": ["US", "EU", "US"], "sku": ["A1", "A1", "B2"]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(
            columns=["region", "sku"],
            ref_table=ref,
            ref_column=["r", "s"],
        )
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


def test_composite_key_na_pass_true():
    ref = pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    tbl = pl.DataFrame({"a": [1, None, 3], "b": ["x", None, "z"]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(
            columns=["a", "b"],
            ref_table=ref,
            ref_column=["a", "b"],
            na_pass=True,
        )
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


# ── Callable ref_table ────────────────────────────────────────────────────────


def test_callable_ref_table():
    def get_ref():
        return pl.DataFrame({"id": [10, 20, 30]})

    tbl = pl.DataFrame({"fk": [10, 20, 99]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=get_ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


# ── Cross-backend: Polars ↔ Pandas ───────────────────────────────────────────


def test_polars_tbl_pandas_ref():
    ref = pd.DataFrame({"id": [1, 2, 3]})
    tbl = pl.DataFrame({"fk": [1, 2, 99]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


def test_pandas_tbl_polars_ref():
    ref = pl.DataFrame({"id": [1, 2, 3]})
    tbl = pd.DataFrame({"fk": [1, 2, 99]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


# ── Cross-backend: DuckDB ────────────────────────────────────────────────────


@pytest.fixture
def duckdb_connection():
    duckdb = pytest.importorskip("duckdb")
    conn = duckdb.connect()
    yield conn
    conn.close()


def test_polars_tbl_duckdb_ref(duckdb_connection):
    ibis = pytest.importorskip("ibis")
    conn = ibis.duckdb.from_connection(duckdb_connection)
    ref = conn.create_table("ref_ids", obj=pd.DataFrame({"id": [1, 2, 3]}))

    tbl = pl.DataFrame({"fk": [1, 2, 99]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


def test_duckdb_tbl_polars_ref(duckdb_connection):
    ibis = pytest.importorskip("ibis")
    conn = ibis.duckdb.from_connection(duckdb_connection)
    tbl = conn.create_table("orders", obj=pd.DataFrame({"fk": [1, 2, 99]}))

    ref = pl.DataFrame({"id": [1, 2, 3]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


# ── Duplicate ref values don't inflate results ───────────────────────────────


def test_ref_duplicates_handled():
    ref = pl.DataFrame({"id": [1, 1, 2, 2, 3, 3]})
    tbl = pl.DataFrame({"fk": [1, 2, 3]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 3
    assert v.n_failed(i=1, scalar=True) == 0


# ── Empty tables ──────────────────────────────────────────────────────────────


def test_empty_data_table():
    ref = pl.DataFrame({"id": [1, 2, 3]})
    tbl = pl.DataFrame({"fk": pl.Series([], dtype=pl.Int64)})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n(i=1, scalar=True) == 0


def test_empty_ref_table_all_fail():
    ref = pl.DataFrame({"id": pl.Series([], dtype=pl.Int64)})
    tbl = pl.DataFrame({"fk": [1, 2]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 0
    assert v.n_failed(i=1, scalar=True) == 2


# ── Validation errors ────────────────────────────────────────────────────────


def test_mismatched_column_counts():
    ref = pl.DataFrame({"a": [1], "b": [2]})
    tbl = pl.DataFrame({"x": [1]})

    with pytest.raises(ValueError, match="same length"):
        pb.Validate(data=tbl).col_vals_in_table(
            columns="x",
            ref_table=ref,
            ref_column=["a", "b"],
        )


# ── Thresholds ────────────────────────────────────────────────────────────────


def test_threshold_warn():
    ref = pl.DataFrame({"id": [1, 2, 3]})
    tbl = pl.DataFrame({"fk": [1, 2, 99, 100]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(
            columns="fk",
            ref_table=ref,
            ref_column="id",
            thresholds=pb.Thresholds(warning=0.3),
        )
        .interrogate()
    )

    assert v.n_failed(i=1, scalar=True) == 2
    assert v.all_passed() is False


# ── Multiple steps ────────────────────────────────────────────────────────────


def test_multiple_in_table_steps():
    customers = pl.DataFrame({"id": [1, 2, 3]})
    products = pl.DataFrame({"sku": ["A", "B", "C"]})

    orders = pl.DataFrame({
        "customer_id": [1, 2, 99],
        "product_sku": ["A", "B", "D"],
    })

    v = (
        pb.Validate(data=orders)
        .col_vals_in_table(columns="customer_id", ref_table=customers, ref_column="id")
        .col_vals_in_table(columns="product_sku", ref_table=products, ref_column="sku")
        .interrogate()
    )

    assert v.n_failed(i=1, scalar=True) == 1
    assert v.n_failed(i=2, scalar=True) == 1


# ── Method chaining ──────────────────────────────────────────────────────────


def test_chaining_with_other_validations():
    ref = pl.DataFrame({"id": [1, 2, 3]})
    tbl = pl.DataFrame({"fk": [1, 2, 3], "value": [10, 20, 30]})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id")
        .col_vals_gt(columns="value", value=0)
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 3
    assert v.n_passed(i=2, scalar=True) == 3
