import json
from pathlib import Path

import pandas as pd
import polars as pl
import pytest

import pointblank as pb


# ── Basic output structure ───────────────────────────────────────────────────


def test_returns_dict():
    v = pb.Validate(data=pl.DataFrame({"a": [1]})).col_vals_gt(columns="a", value=0)
    schema = v.to_json_schema()
    assert isinstance(schema, dict)


def test_has_json_schema_meta_keys():
    v = pb.Validate(data=pl.DataFrame({"a": [1]})).col_vals_gt(columns="a", value=0)
    schema = v.to_json_schema()
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert schema["type"] == "object"
    assert "properties" in schema


# ── Individual step mappings ─────────────────────────────────────────────────


def test_col_vals_gt():
    v = pb.Validate(data=pl.DataFrame({"x": [1]})).col_vals_gt(columns="x", value=5)
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["exclusiveMinimum"] == 5


def test_col_vals_ge():
    v = pb.Validate(data=pl.DataFrame({"x": [1]})).col_vals_ge(columns="x", value=10)
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["minimum"] == 10


def test_col_vals_lt():
    v = pb.Validate(data=pl.DataFrame({"x": [1]})).col_vals_lt(columns="x", value=100)
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["exclusiveMaximum"] == 100


def test_col_vals_le():
    v = pb.Validate(data=pl.DataFrame({"x": [1]})).col_vals_le(columns="x", value=50)
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["maximum"] == 50


def test_col_vals_eq():
    v = pb.Validate(data=pl.DataFrame({"x": [1]})).col_vals_eq(columns="x", value=42)
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["const"] == 42


def test_col_vals_not_null():
    v = pb.Validate(data=pl.DataFrame({"x": [1]})).col_vals_not_null(columns="x")
    schema = v.to_json_schema()
    assert "x" in schema.get("required", [])


def test_col_vals_in_set():
    v = pb.Validate(data=pl.DataFrame({"x": ["a"]})).col_vals_in_set(
        columns="x", set=["a", "b", "c"]
    )
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["enum"] == ["a", "b", "c"]


def test_col_vals_regex():
    v = pb.Validate(data=pl.DataFrame({"x": ["abc"]})).col_vals_regex(
        columns="x", pattern="^[a-z]+"
    )
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["pattern"] == "^[a-z]+"


def test_col_vals_between():
    v = pb.Validate(data=pl.DataFrame({"x": [5]})).col_vals_between(columns="x", left=0, right=100)
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["minimum"] == 0
    assert schema["properties"]["x"]["maximum"] == 100


# ── Type enrichment ──────────────────────────────────────────────────────────


def test_integer_type_enrichment():
    v = pb.Validate(data=pl.DataFrame({"x": [1, 2]})).col_vals_gt(columns="x", value=0)
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["type"] == "integer"


def test_float_type_enrichment():
    v = pb.Validate(data=pl.DataFrame({"x": [1.5, 2.5]})).col_vals_gt(columns="x", value=0)
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["type"] == "number"


def test_string_type_enrichment():
    v = pb.Validate(data=pl.DataFrame({"x": ["a", "b"]})).col_vals_regex(columns="x", pattern=".")
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["type"] == "string"


def test_boolean_type_enrichment():
    df = pl.DataFrame({"flag": [True, False]})
    v = pb.Validate(data=df).col_vals_not_null(columns="flag")
    schema = v.to_json_schema()
    assert schema["properties"]["flag"]["type"] == "boolean"


# ── Multiple steps on the same column ────────────────────────────────────────


def test_combined_constraints():
    v = (
        pb.Validate(data=pl.DataFrame({"x": [5]}))
        .col_vals_ge(columns="x", value=0)
        .col_vals_le(columns="x", value=100)
        .col_vals_not_null(columns="x")
    )
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["minimum"] == 0
    assert schema["properties"]["x"]["maximum"] == 100
    assert "x" in schema["required"]


# ── Steps with no JSON Schema equivalent are silently skipped ────────────────


def test_unsupported_steps_skipped():
    v = (
        pb.Validate(data=pl.DataFrame({"x": [1], "y": [2]}))
        .col_vals_gt(columns="x", value=0)
        .rows_distinct()
    )
    schema = v.to_json_schema()
    assert "x" in schema["properties"]


# ── No data attached ─────────────────────────────────────────────────────────


def test_no_data_no_type_enrichment():
    schema = pb.export_contract(
        pb.Validate(data=pl.DataFrame({"x": [1]})).col_vals_gt(columns="x", value=0),
        format="json_schema",
    )
    assert "type" not in schema.get("properties", {}).get("x", {})


# ── Multiple columns ────────────────────────────────────────────────────────


def test_multiple_columns():
    df = pl.DataFrame({"a": [1], "b": ["x"], "c": [1.5]})
    v = (
        pb.Validate(data=df)
        .col_vals_gt(columns="a", value=0)
        .col_vals_regex(columns="b", pattern="^[a-z]")
        .col_vals_le(columns="c", value=10.0)
    )
    schema = v.to_json_schema()
    assert "a" in schema["properties"]
    assert "b" in schema["properties"]
    assert "c" in schema["properties"]
    assert schema["properties"]["a"]["type"] == "integer"
    assert schema["properties"]["b"]["type"] == "string"
    assert schema["properties"]["c"]["type"] == "number"


# ── File output ──────────────────────────────────────────────────────────────


def test_writes_file(tmp_path):
    df = pl.DataFrame({"x": [1]})
    v = pb.Validate(data=df).col_vals_gt(columns="x", value=0)
    path = tmp_path / "schema.json"

    result = v.to_json_schema(path=str(path))

    assert path.exists()
    with open(path) as f:
        saved = json.load(f)
    assert result == saved


def test_creates_parent_directories(tmp_path):
    df = pl.DataFrame({"x": [1]})
    v = pb.Validate(data=df).col_vals_gt(columns="x", value=0)
    path = tmp_path / "nested" / "dir" / "schema.json"

    v.to_json_schema(path=str(path))
    assert path.exists()


def test_file_is_valid_json(tmp_path):
    df = pl.DataFrame({"x": [1]})
    v = pb.Validate(data=df).col_vals_gt(columns="x", value=0)
    path = tmp_path / "schema.json"

    v.to_json_schema(path=str(path))

    with open(path) as f:
        parsed = json.load(f)
    assert parsed["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_path_as_pathlib(tmp_path):
    df = pl.DataFrame({"x": [1]})
    v = pb.Validate(data=df).col_vals_gt(columns="x", value=0)
    path = tmp_path / "schema.json"

    v.to_json_schema(path=path)
    assert path.exists()


def test_no_path_returns_dict_only():
    df = pl.DataFrame({"x": [1]})
    v = pb.Validate(data=df).col_vals_gt(columns="x", value=0)
    result = v.to_json_schema()
    assert isinstance(result, dict)


# ── Pandas backend ───────────────────────────────────────────────────────────


def test_pandas_input():
    df = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    v = (
        pb.Validate(data=df)
        .col_vals_gt(columns="x", value=0)
        .col_vals_regex(columns="y", pattern="^[a-z]")
    )
    schema = v.to_json_schema()
    assert schema["properties"]["x"]["exclusiveMinimum"] == 0
    assert schema["properties"]["y"]["pattern"] == "^[a-z]"


# ── Empty validation plan ───────────────────────────────────────────────────


def test_empty_plan():
    v = pb.Validate(data=pl.DataFrame({"x": [1]}))
    schema = v.to_json_schema()
    assert isinstance(schema, dict)
    assert schema["type"] == "object"
