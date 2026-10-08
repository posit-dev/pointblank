import pytest

from pointblank._utils_check_args import (
    _check_active_input,
    _check_boolean_input,
    _check_column,
    _check_value_float_int,
    _check_set_types,
    _check_pre,
    _check_thresholds,
)
from pointblank.thresholds import Thresholds
from pointblank.column import col, starts_with


def test_check_boolean_input():
    assert _check_boolean_input(param=True, param_name="test") is None

    with pytest.raises(ValueError):
        _check_boolean_input(param="not a boolean", param_name="test")


def test_check_active_input():
    assert _check_active_input(param=True, param_name="active") is None
    assert _check_active_input(param=False, param_name="active") is None
    assert _check_active_input(param=lambda tbl: True, param_name="active") is None

    with pytest.raises(ValueError):
        _check_active_input(param="not valid", param_name="active")
    with pytest.raises(ValueError):
        _check_active_input(param=42, param_name="active")


def test_check_column():
    assert _check_column(column="test") is None
    assert _check_column(column=["one", "two", "three"]) is None
    assert _check_column(column=col(starts_with("test"))) is None
    assert _check_column(column=col("test")) is None

    with pytest.raises(ValueError):
        _check_column(column=123)
    with pytest.raises(ValueError):
        _check_column(column=[1.0, 3.0])


def test_check_value_float_int():
    assert _check_value_float_int(value=1.0) is None
    assert _check_value_float_int(value=1) is None

    with pytest.raises(ValueError):
        _check_value_float_int(value="not a number")
    with pytest.raises(ValueError):
        _check_value_float_int(value=True)


def test_check_set_types():
    assert _check_set_types(set=[1.0, 1, "test"]) is None
    assert _check_set_types(set=(1.0, 1, "test")) is None
    assert _check_set_types(set={1.0, 1, "test"}) is None

    with pytest.raises(ValueError):
        _check_set_types(set=[1.0, 1, Thresholds])
    with pytest.raises(ValueError):
        _check_set_types(set=[1.0, 1, True])


def test_check_pre():
    def test_func():
        pass

    assert _check_pre(pre=test_func) is None
    assert _check_pre(pre=None) is None
    assert _check_pre(pre=lambda x: x + 1) is None

    with pytest.raises(ValueError):
        _check_pre(pre="not a function")


def test_check_thresholds():
    assert _check_thresholds(thresholds=1.0) is None
    assert _check_thresholds(thresholds=1) is None
    assert _check_thresholds(thresholds=(1.0, 2.0)) is None
    assert _check_thresholds(thresholds=(1, 2)) is None
    assert _check_thresholds(thresholds={"warning": 2.0}) is None
    assert _check_thresholds(thresholds={"error": 2}) is None
    assert _check_thresholds(thresholds=None) is None
    assert _check_thresholds(thresholds=True) is None
    assert _check_thresholds(thresholds=Thresholds()) is None
    assert _check_thresholds(thresholds=Thresholds(True, False, 0.34)) is None

    with pytest.raises(ValueError):
        _check_thresholds(thresholds=-1.0)
    with pytest.raises(ValueError):
        _check_thresholds(thresholds=-1)
    with pytest.raises(ValueError):
        _check_thresholds(thresholds=(-1.0, 2.0))
    with pytest.raises(ValueError):
        _check_thresholds(thresholds=(-1, 2))
    with pytest.raises(ValueError):
        _check_thresholds(thresholds={-1.0: 2.0})
    with pytest.raises(ValueError):
        _check_thresholds(thresholds={"invalid_key": 2})
    with pytest.raises(ValueError):
        _check_thresholds(thresholds="not a threshold")
    with pytest.raises(ValueError):
        _check_thresholds(thresholds=[1, 2, 3])

    class TestThresholds:
        pass

    with pytest.raises(ValueError):
        _check_thresholds(thresholds=TestThresholds())
