import pickle
import sys
import tempfile
from pathlib import Path

import polars as pl
import pytest

import pointblank as pb

VALIDATIONS_DIR = Path(__file__).parent.parent / "pointblank" / "data" / "validations"
sys.path.insert(0, str(VALIDATIONS_DIR))

from generate_test_files import (
    create_test_data,
    create_validation_examples,
    save_validation_files,
)


def test_create_test_data_shape():
    df = create_test_data()
    assert df.shape == (10, 4)
    assert set(df.columns) == {"a", "b", "c", "d"}


def test_create_test_data_types():
    df = create_test_data()
    assert df["a"].dtype == pl.Int32 or df["a"].dtype in (pl.Int64, pl.Int32)
    assert df["c"].dtype == pl.String or df["c"].dtype == pl.Utf8


def test_create_test_data_values():
    df = create_test_data()
    assert df["a"].to_list() == list(range(1, 11))
    assert set(df["c"].unique().to_list()) == {"x", "y"}


def test_create_validation_examples_keys():
    validations = create_validation_examples()
    expected_keys = {
        "simple_preprocessing",
        "complex_preprocessing",
        "narwhals_function",
        "multiple_steps",
        "pandas_compatible",
        "no_preprocessing",
    }
    assert set(validations.keys()) == expected_keys


def test_create_validation_examples_types():
    validations = create_validation_examples()
    for name, v in validations.items():
        assert isinstance(v, pb.Validate), f"{name} is not a Validate instance"


def test_create_validation_examples_step_counts():
    validations = create_validation_examples()
    assert len(validations["simple_preprocessing"].validation_info) == 2
    assert len(validations["complex_preprocessing"].validation_info) == 2
    assert len(validations["narwhals_function"].validation_info) == 1
    assert len(validations["multiple_steps"].validation_info) == 3
    assert len(validations["pandas_compatible"].validation_info) == 1
    assert len(validations["no_preprocessing"].validation_info) == 3


def test_save_validation_files_creates_pkl(tmp_path):
    validations = {"no_preprocessing": create_validation_examples()["no_preprocessing"]}
    save_validation_files(validations, tmp_path)
    assert (tmp_path / "no_preprocessing.pkl").exists()


def test_save_validation_files_pkl_loadable(tmp_path):
    validations = {"no_preprocessing": create_validation_examples()["no_preprocessing"]}
    save_validation_files(validations, tmp_path)
    with open(tmp_path / "no_preprocessing.pkl", "rb") as f:
        loaded = pickle.load(f)
    assert isinstance(loaded, pb.Validate)


def test_save_validation_files_creates_json(tmp_path):
    validations = {"no_preprocessing": create_validation_examples()["no_preprocessing"]}
    save_validation_files(validations, tmp_path)
    assert (tmp_path / "no_preprocessing.json").exists()


def test_save_validation_files_creates_output_dir(tmp_path):
    nested = tmp_path / "new_dir" / "subdir"
    validations = {"no_preprocessing": create_validation_examples()["no_preprocessing"]}
    save_validation_files(validations, nested)
    assert nested.is_dir()
    assert (nested / "no_preprocessing.pkl").exists()


def test_save_validation_files_all_keys(tmp_path):
    validations = create_validation_examples()
    save_validation_files(validations, tmp_path)
    expected = {
        "simple_preprocessing",
        "complex_preprocessing",
        "narwhals_function",
        "multiple_steps",
        "pandas_compatible",
        "no_preprocessing",
    }
    for name in expected:
        assert (tmp_path / f"{name}.pkl").exists(), f"Missing {name}.pkl"
