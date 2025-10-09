import pathlib
import sys

import pytest
from unittest.mock import patch

import narwhals as nw
from pointblank.schema import Schema, _check_schema_match
from pointblank.validate import load_dataset

import pandas as pd
import polars as pl
import ibis


TBL_LIST = [
    "tbl_pd",
    "tbl_pl",
    "tbl_parquet",
    "tbl_duckdb",
    "tbl_sqlite",
]


@pytest.fixture
def tbl_pd():
    return pd.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_pl():
    return pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})


@pytest.fixture
def tbl_parquet():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.parquet"
    return ibis.read_parquet(file_path)


@pytest.fixture
def tbl_duckdb():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.ddb"
    return ibis.connect(f"duckdb://{file_path}").table("tbl_xyz")


@pytest.fixture
def tbl_sqlite():
    file_path = pathlib.Path.cwd() / "tests" / "tbl_files" / "tbl_xyz.sqlite"
    return ibis.sqlite.connect(file_path).table("tbl_xyz")


def test_schema_str(capfd):
    schema = Schema(columns=[("a", "int"), ("b", "str")])
    print(schema)
    captured = capfd.readouterr()
    expected_output = "Pointblank Schema\n  a: int\n  b: str\n"
    assert captured.out == expected_output


def test_schema_str_no_data_type(capfd):
    schema = Schema(columns=[("a",), ("b", "str")])
    print(schema)
    captured = capfd.readouterr()
    expected_output = "Pointblank Schema\n  a: <ANY>\n  b: str\n"
    assert captured.out == expected_output


def test_schema_repr():
    schema = Schema(columns=[("a", "int"), ("b", "str")])
    expected_repr = "Schema(columns=[('a', 'int'), ('b', 'str')])"
    assert repr(schema) == expected_repr


def test_equivalent_inputs():
    schema_1 = Schema(columns=[("a", "int"), ("b", "str")])
    schema_2 = Schema(columns={"a": "int", "b": "str"})
    schema_3 = Schema(a="int", b="str")

    assert schema_1.columns == schema_2.columns
    assert schema_2.columns == schema_3.columns


def test_schema_only_columns_equivalent_inputs():
    schema_1 = Schema(columns=["a", "b"])
    schema_2 = Schema(columns=[("a",), ("b",)])

    assert schema_1.columns == schema_2.columns


def test_schema_single_column_equivalent_inputs():
    schema_1 = Schema(columns=["a"])
    schema_2 = Schema(columns="a")
    schema_3 = Schema(columns=[("a",)])

    assert schema_1.columns == schema_2.columns
    assert schema_2.columns == schema_3.columns


def test_schema_from_pd_table():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="pandas"))
    assert schema.columns == [
        ("date_time", "datetime64[ns]"),
        ("date", "datetime64[ns]"),
        ("a", "int64"),
        ("b", "object"),
        ("c", "float64"),
        ("d", "float64"),
        ("e", "bool"),
        ("f", "object"),
    ]

    assert str(type(schema.tbl)) == "<class 'pandas.core.frame.DataFrame'>"


def test_schema_from_pl_table():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="polars"))
    assert schema.columns == [
        ("date_time", "Datetime(time_unit='us', time_zone=None)"),
        ("date", "Date"),
        ("a", "Int64"),
        ("b", "String"),
        ("c", "Int64"),
        ("d", "Float64"),
        ("e", "Boolean"),
        ("f", "String"),
    ]

    assert str(type(schema.tbl)) == "<class 'polars.dataframe.frame.DataFrame'>"


def test_schema_from_parquet_table(tbl_parquet):
    schema = Schema(tbl=tbl_parquet)

    assert schema.columns == [
        ("x", "int64"),
        ("y", "int64"),
        ("z", "int64"),
    ]

    assert str(type(schema.tbl)) == "<class 'ibis.expr.types.relations.Table'>"


def test_schema_from_duckdb_table():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="duckdb"))

    target_types: dict[str, tuple[str | tuple[str, ...], ...]] = {
        "date_time": ("timestamp(6)", "timestamp"),
        "date": "date",
        "a": "int64",
        "b": "string",
        "c": "int64",
        "d": "float64",
        "e": "boolean",
        "f": "string",
    }

    for target, real in zip(target_types, schema.columns):
        # check if the column name is in the target_types dict
        if target in target_types:
            # check if the real type is in the expected types
            if isinstance(target_types[target], tuple):
                assert real[1] in target_types[target]
            else:
                assert real[1] == target_types[target]
        else:
            raise AssertionError

    assert str(type(schema.tbl)) == "<class 'ibis.expr.types.relations.Table'>"


def test_schema_from_sqlite_table(tbl_sqlite):
    schema = Schema(tbl=tbl_sqlite)

    assert schema.columns == [
        ("x", "int64"),
        ("y", "int64"),
        ("z", "int64"),
    ]

    assert str(type(schema.tbl)) == "<class 'ibis.expr.types.relations.Table'>"


def test_get_tbl_type_small_table():
    schema_pd = Schema(tbl=load_dataset(dataset="small_table", tbl_type="pandas"))
    schema_pl = Schema(tbl=load_dataset(dataset="small_table", tbl_type="polars"))
    schema_duckdb = Schema(tbl=load_dataset(dataset="small_table", tbl_type="duckdb"))

    assert schema_pd.get_tbl_type() == "pandas"
    assert schema_pl.get_tbl_type() == "polars"
    assert schema_duckdb.get_tbl_type() == "duckdb"


def test_get_column_list_small_table():
    schema_pd = Schema(tbl=load_dataset(dataset="small_table", tbl_type="pandas"))
    schema_pl = Schema(tbl=load_dataset(dataset="small_table", tbl_type="polars"))
    schema_duckdb = Schema(tbl=load_dataset(dataset="small_table", tbl_type="duckdb"))

    schemas = [schema_pd, schema_pl, schema_duckdb]

    for schema in schemas:
        assert schema.get_column_list() == [
            "date_time",
            "date",
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
        ]


def test_get_dtype_list_small_table_pd():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="pandas"))

    assert schema.get_dtype_list() == [
        "datetime64[ns]",
        "datetime64[ns]",
        "int64",
        "object",
        "float64",
        "float64",
        "bool",
        "object",
    ]


def test_get_dtype_list_small_table_pl():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="polars"))

    assert schema.get_dtype_list() == [
        "Datetime(time_unit='us', time_zone=None)",
        "Date",
        "Int64",
        "String",
        "Int64",
        "Float64",
        "Boolean",
        "String",
    ]


def test_get_dtype_list_small_table_duckdb():
    schema = Schema(tbl=load_dataset(dataset="small_table", tbl_type="duckdb"))

    target_types: tuple[str | tuple[str, ...], ...] = (
        ("timestamp(6)", "timestamp"),
        "date",
        "int64",
        "string",
        "int64",
        "float64",
        "boolean",
        "string",
    )

    for target, real in zip(target_types, schema.get_dtype_list()):
        if isinstance(target, tuple):
            assert real in target
        else:
            assert real == target


def test_get_dtype_list_game_revenue_pd():
    schema = Schema(tbl=load_dataset(dataset="game_revenue", tbl_type="pandas"))

    assert schema.get_dtype_list() == [
        "object",
        "object",
        "datetime64[ns, UTC]",
        "datetime64[ns, UTC]",
        "object",
        "object",
        "float64",
        "float64",
        "datetime64[ns]",
        "object",
        "object",
    ]


def test_get_dtype_list_game_revenue_pl():
    schema = Schema(tbl=load_dataset(dataset="game_revenue", tbl_type="polars"))

    assert schema.get_dtype_list() == [
        "String",
        "String",
        "Datetime(time_unit='us', time_zone='UTC')",
        "Datetime(time_unit='us', time_zone='UTC')",
        "String",
        "String",
        "Float64",
        "Float64",
        "Date",
        "String",
        "String",
    ]


def test_get_dtype_list_game_revenue_duckdb():
    schema = Schema(tbl=load_dataset(dataset="game_revenue", tbl_type="duckdb"))

    assert schema.get_dtype_list() == [
        "string",
        "string",
        "timestamp('UTC', 6)",
        "timestamp('UTC', 6)",
        "string",
        "string",
        "float64",
        "float64",
        "date",
        "string",
        "string",
    ]


def test_schema_coercion_pd_to_pl():
    schema_pd = Schema(tbl=load_dataset(dataset="small_table", tbl_type="pandas"))
    schema_pl = schema_pd.get_schema_coerced(to="polars")

    assert schema_pd.columns == [
        ("date_time", "datetime64[ns]"),
        ("date", "datetime64[ns]"),
        ("a", "int64"),
        ("b", "object"),
        ("c", "float64"),
        ("d", "float64"),
        ("e", "bool"),
        ("f", "object"),
    ]

    assert schema_pl.columns == [
        ("date_time", "Datetime(time_unit='ns', time_zone=None)"),
        ("date", "Datetime(time_unit='ns', time_zone=None)"),
        ("a", "Int64"),
        ("b", "String"),
        ("c", "Float64"),
        ("d", "Float64"),
        ("e", "Boolean"),
        ("f", "String"),
    ]

    assert str(type(schema_pl.tbl)) == "<class 'polars.dataframe.frame.DataFrame'>"


def test_schema_coercion_pl_to_pd():
    schema_pl = Schema(tbl=load_dataset(dataset="small_table", tbl_type="polars"))
    schema_pd = schema_pl.get_schema_coerced(to="pandas")

    assert schema_pl.columns == [
        ("date_time", "Datetime(time_unit='us', time_zone=None)"),
        ("date", "Date"),
        ("a", "Int64"),
        ("b", "String"),
        ("c", "Int64"),
        ("d", "Float64"),
        ("e", "Boolean"),
        ("f", "String"),
    ]

    assert schema_pd.columns == [
        ("date_time", "datetime64[us]"),
        ("date", "datetime64[ms]"),
        ("a", "int64"),
        ("b", "object"),
        ("c", "float64"),
        ("d", "float64"),
        ("e", "bool"),
        ("f", "object"),
    ]

    assert str(type(schema_pd.tbl)) == "<class 'pandas.core.frame.DataFrame'>"


def test_schema_coercion_raises_no_tbl():
    schema = Schema(columns=[("a", "int"), ("b", "str")])

    with pytest.raises(ValueError):
        schema.get_schema_coerced(to="polars")

    with pytest.raises(ValueError):
        schema.get_schema_coerced(to="pandas")


def test_schema_coercion_raises_no_lib():
    schema_pd = Schema(tbl=load_dataset(dataset="small_table", tbl_type="pandas"))
    schema_pl = Schema(tbl=load_dataset(dataset="small_table", tbl_type="polars"))

    # Mock the absence of the polars library
    with patch.dict(sys.modules, {"polars": None}):
        with pytest.raises(ImportError):
            schema_pd.get_schema_coerced(to="polars")

    # Mock the absence of the pandas library
    with patch.dict(sys.modules, {"pandas": None}):
        with pytest.raises(ImportError):
            schema_pl.get_schema_coerced(to="pandas")

    # Mock the absence of the pyarrow library
    with patch.dict(sys.modules, {"pyarrow": None}):
        with pytest.raises(ImportError):
            schema_pl.get_schema_coerced(to="pandas")


@pytest.mark.parametrize("tbl_fixture", TBL_LIST)
def test_schema_input_errors(request, tbl_fixture):
    tbl = request.getfixturevalue(tbl_fixture)

    with pytest.raises(ValueError):
        Schema()

    with pytest.raises(ValueError):
        Schema(tbl=tbl, columns=[("a", "int")])

    with pytest.raises(ValueError):
        Schema(columns=1)

    with pytest.raises(ValueError):
        Schema(tbl=tbl, a="int")

    with pytest.raises(ValueError):
        Schema(columns=("a", "int", "extra"))

    with pytest.raises(ValueError):
        Schema(columns=[("a", "int"), ["b", "str"]])

    with pytest.raises(ValueError):
        Schema(columns=("a", "int"))

    with pytest.raises(ValueError):
        Schema(columns=(1, "int"))


def test_schema_from_narwhals_lazy_frame():
    """Test schema extraction from a Narwhals lazy frame."""
    # Create a polars lazy frame and convert to narwhals
    tbl_pl_lazy = pl.LazyFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})
    tbl_nw_lazy = nw.from_native(tbl_pl_lazy)

    # Create schema from narwhals lazy frame
    schema = Schema(tbl=tbl_nw_lazy)

    assert schema.columns is not None
    assert len(schema.columns) == 3
    assert schema.columns[0][0] == "x"
    assert schema.columns[1][0] == "y"
    assert schema.columns[2][0] == "z"


def test_schema_from_narwhals_eager_frame():
    """Test schema extraction from a Narwhals eager/non-lazy frame."""
    tbl_pl = pl.DataFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})
    tbl_nw = nw.from_native(tbl_pl)

    # Create schema from narwhals eager frame
    schema = Schema(tbl=tbl_nw)

    assert schema.columns is not None
    assert len(schema.columns) == 3
    assert schema.columns[0][0] == "x"
    assert schema.columns[1][0] == "y"
    assert schema.columns[2][0] == "z"


def test_schema_from_polars_lazy_frame():
    """Test schema extraction from a Polars LazyFrame directly."""
    tbl_pl_lazy = pl.LazyFrame({"x": [1, 2, 3, 4], "y": [4, 5, 6, 7], "z": [8, 8, 8, 8]})

    # Create schema from polars lazy frame
    schema = Schema(tbl=tbl_pl_lazy)

    assert schema.columns is not None
    assert len(schema.columns) == 3
    assert schema.columns[0][0] == "x"
    assert schema.columns[1][0] == "y"
    assert schema.columns[2][0] == "z"


def test_check_schema_match_basic():
    """Test the _check_schema_match() function with basic validation."""
    # Create a simple DataFrame
    tbl = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    # Create a matching schema - use actual pandas dtypes or partial match
    schema = Schema(columns=[("x", "int"), ("y", "int")])

    # Should pass with partial dtype matching (full_match_dtypes=False)
    assert _check_schema_match(data_tbl=tbl, schema=schema, full_match_dtypes=False) is True

    # Create a non-matching schema (wrong column name)
    schema_wrong = Schema(columns=[("x", "int"), ("z", "int")])
    assert _check_schema_match(data_tbl=tbl, schema=schema_wrong, full_match_dtypes=False) is False

    # Create a schema with wrong dtype
    schema_wrong_dtype = Schema(columns=[("x", "str"), ("y", "int")])
    assert (
        _check_schema_match(data_tbl=tbl, schema=schema_wrong_dtype, full_match_dtypes=False)
        is False
    )


def test_check_schema_match_complete_option():
    """Test the _check_schema_match() function with complete= parameter."""
    tbl = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]})

    # Partial schema (only 2 of 3 columns)
    schema_partial = Schema(columns=[("x", "int"), ("y", "int")])

    # Should fail when complete=True
    assert (
        _check_schema_match(
            data_tbl=tbl, schema=schema_partial, complete=True, full_match_dtypes=False
        )
        is False
    )

    # Should pass when complete=False
    assert (
        _check_schema_match(
            data_tbl=tbl, schema=schema_partial, complete=False, full_match_dtypes=False
        )
        is True
    )


def test_check_schema_match_in_order_option():
    """Test the _check_schema_match() function with in_order= parameter."""
    tbl = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

    # Schema with columns in different order
    schema_wrong_order = Schema(columns=[("y", "int"), ("x", "int")])

    # Should fail when in_order=True
    assert (
        _check_schema_match(
            data_tbl=tbl, schema=schema_wrong_order, in_order=True, full_match_dtypes=False
        )
        is False
    )

    # Should pass when in_order=False
    assert (
        _check_schema_match(
            data_tbl=tbl, schema=schema_wrong_order, in_order=False, full_match_dtypes=False
        )
        is True
    )
