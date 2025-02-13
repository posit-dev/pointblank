import pytest
import sys

from unittest.mock import patch

from pointblank.validate import load_dataset
from pointblank.datascan import DataScan


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_datascan_class(tbl_type):

    dataset = load_dataset(dataset="small_table", tbl_type=tbl_type)
    scanner = DataScan(data=dataset)

    assert scanner.data.equals(dataset)
    assert scanner.tbl_name is None
    assert scanner.profile is not None
    assert isinstance(scanner.profile, dict)

    if tbl_type == "duckdb":
        assert scanner.tbl_type == "duckdb"
        assert scanner.tbl_category == "ibis"
        assert scanner.data_alt is None

    if tbl_type == "polars":
        assert scanner.tbl_type == "polars"
        assert scanner.tbl_category == "dataframe"
        assert scanner.data_alt is not None

    if tbl_type == "pandas":
        assert scanner.tbl_type == "pandas"
        assert scanner.tbl_category == "dataframe"
        assert scanner.data_alt is not None


def test_datascan_class_use_tbl_name():

    dataset = load_dataset(dataset="small_table")
    scanner = DataScan(data=dataset, tbl_name="my_small_table")

    assert scanner.tbl_name == "my_small_table"


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_datascan_no_fail(tbl_type):

    small_table = load_dataset(dataset="small_table", tbl_type=tbl_type)
    DataScan(data=small_table)

    game_revenue = load_dataset(dataset="game_revenue", tbl_type=tbl_type)
    DataScan(data=game_revenue)

    nycflights = load_dataset(dataset="nycflights", tbl_type=tbl_type)
    DataScan(data=nycflights)


@pytest.mark.parametrize("tbl_type", ["pandas", "polars", "duckdb"])
def test_datascan_dict_output(tbl_type):

    dataset = load_dataset(dataset="small_table", tbl_type=tbl_type)
    scanner = DataScan(data=dataset)

    assert isinstance(scanner.profile, dict)

    scan_dict = scanner.get_profile()

    assert isinstance(scan_dict, dict)

    assert scanner.profile == scan_dict


def test_datascan_json_output():

    dataset = load_dataset(dataset="small_table")
    scanner = DataScan(data=dataset)

    profile_json = scanner.to_json()

    assert isinstance(profile_json, str)


def test_datascan_json_file_output(tmp_path):

    dataset = load_dataset(dataset="small_table")
    scanner = DataScan(data=dataset)

    profile_json = scanner.to_json()

    file_path = tmp_path / "profile.json"
    scanner.save_to_json(output_file=file_path)

    assert file_path.exists()
    assert file_path.is_file()

    with open(file_path, "r") as f:
        file_content = f.read()

    assert profile_json == file_content


def test_datascan_class_raises():
    with pytest.raises(TypeError):
        DataScan(data="not a DataFrame or Ibis Table")

    with pytest.raises(TypeError):
        DataScan(data=123)

    with pytest.raises(TypeError):
        DataScan(data=[1, 2, 3])


def test_datascan_ibis_table_no_polars():

    # Mock the absence of the polars library
    with patch.dict(sys.modules, {"polars": None}):

        small_table = load_dataset(dataset="small_table", tbl_type="duckdb")
        DataScan(data=small_table)
