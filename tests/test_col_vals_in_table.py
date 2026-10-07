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

    orders = pl.DataFrame(
        {
            "customer_id": [1, 2, 99],
            "product_sku": ["A", "B", "D"],
        }
    )

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


@pytest.mark.parametrize(
    "tbl_values, ref_values",
    [([1.0, 2.0, 99.0], [1, 2, 3]), ([1, 2, 99], [1.0, 2.0, 3.0])],
    ids=["float_tbl_int_ref", "int_tbl_float_ref"],
)
def test_single_col_int_float_mismatch(tbl_values, ref_values):
    # Polars >= 2.0 no longer coerces between integers and floats in `is_in()`
    ref = pl.DataFrame({"id": ref_values})
    tbl = pl.DataFrame({"fk": tbl_values})

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns="fk", ref_table=ref, ref_column="id")
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


# ── Composite keys across backends and key dtypes ────────────────────────────


def _to_backend(df: pl.DataFrame, backend: str):
    if backend == "pandas":
        return df.to_pandas()
    if backend == "duckdb":
        ibis = pytest.importorskip("ibis")
        return ibis.memtable(df.to_arrow())
    return df


@pytest.mark.parametrize("backend", ["polars", "pandas", "duckdb"])
def test_composite_key_backends(backend):
    ref = _to_backend(pl.DataFrame({"a": [1, 2, 9], "b": ["x", "y", "z"]}), backend)
    tbl = _to_backend(pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}), backend)

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns=["a", "b"], ref_table=ref, ref_column=["a", "b"])
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


@pytest.mark.parametrize("backend", ["polars", "pandas", "duckdb"])
@pytest.mark.parametrize(
    "tbl_dtype, ref_values, ref_dtype, n_passed",
    [
        (pl.Int64, [1.0, 2.0, 9.0], pl.Float64, 2),
        (pl.Int64, [1.0, 2.5, 9.0], pl.Float64, 1),  # 2.5 can't match an integer key
        (pl.Float64, [1, 2, 9], pl.Int64, 2),
        (pl.Int32, [1, 2, 9], pl.Int64, 2),
        (pl.UInt8, [1, 2, 9], pl.Int64, 2),
        (pl.Float32, [1.0, 2.0, 9.0], pl.Float64, 2),
    ],
    ids=["int_float", "int_float_fractional", "float_int", "int32_int64", "uint8_int64", "f32_f64"],
)
def test_composite_key_numeric_dtype_mismatch(backend, tbl_dtype, ref_values, ref_dtype, n_passed):
    ref = pl.DataFrame({"id": pl.Series(ref_values, dtype=ref_dtype), "k": ["x", "y", "z"]})
    tbl = pl.DataFrame({"fk": pl.Series([1, 2, 3], dtype=tbl_dtype), "k": ["x", "y", "z"]})

    v = (
        pb.Validate(data=_to_backend(tbl, backend))
        .col_vals_in_table(
            columns=["fk", "k"], ref_table=_to_backend(ref, backend), ref_column=["id", "k"]
        )
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == n_passed
    assert v.n_failed(i=1, scalar=True) == 3 - n_passed


def test_composite_key_decimal_float_mismatch():
    ref = pl.DataFrame({"id": [1.5, 2.0], "k": ["x", "y"]})
    tbl = pl.DataFrame({"fk": ["1.5", "2.0", "3.0"], "k": ["x", "y", "z"]}).with_columns(
        pl.col("fk").str.to_decimal(scale=1)
    )

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns=["fk", "k"], ref_table=ref, ref_column=["id", "k"])
        .interrogate()
    )

    assert v.n_passed(i=1, scalar=True) == 2
    assert v.n_failed(i=1, scalar=True) == 1


def test_composite_key_cast_does_not_change_table():
    from pointblank._interrogation import interrogate_in_table

    ref = pl.DataFrame({"id": [1.0, 2.0], "k": ["x", "y"]})
    tbl = pl.DataFrame({"fk": [1, 2, 3], "k": ["x", "y", "z"]})

    result = interrogate_in_table(tbl, ["fk", "k"], ref, ["id", "k"], na_pass=False)

    assert result.schema["fk"] == pl.Int64
    assert result["pb_is_good_"].to_list() == [True, True, False]


def test_composite_key_pandas_non_default_index():
    from pointblank._interrogation import interrogate_in_table

    ref = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    tbl = pd.DataFrame({"a": [3, 1, 2, 1], "b": ["z", "x", "y", "q"]}, index=[10, 7, 99, 3])

    result = interrogate_in_table(tbl, ["a", "b"], ref, ["a", "b"], na_pass=False)

    assert result["pb_is_good_"].tolist() == [False, True, True, False]
    assert result.index.tolist() == [10, 7, 99, 3]


@pytest.mark.parametrize("backend", ["polars", "pandas", "duckdb"])
@pytest.mark.parametrize("composite", [False, True])
def test_incompatible_key_types_eval_error(backend, composite):
    ref = _to_backend(pl.DataFrame({"id": ["1", "2"], "k": ["x", "y"]}), backend)
    tbl = _to_backend(pl.DataFrame({"fk": [1, 2, 3], "k": ["x", "y", "z"]}), backend)

    columns, ref_column = (["fk", "k"], ["id", "k"]) if composite else ("fk", "id")

    v = (
        pb.Validate(data=tbl)
        .col_vals_in_table(columns=columns, ref_table=ref, ref_column=ref_column)
        .interrogate()
    )

    assert v.validation_info[0].eval_error is True
