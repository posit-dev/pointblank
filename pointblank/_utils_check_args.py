from __future__ import annotations

from typing import Callable

import narwhals as nw

from pointblank.column import Column, ColumnSelector
from pointblank.thresholds import Thresholds


def _check_boolean_input(param: bool, param_name: str):
    """
    Check that input value is a boolean.

    Parameters
    ----------
    param
        The input value to check for a boolean value.
    param_name
        The name of the parameter being checked. This is used in the error message.

    Raises
    ------
    ValueError
        When `param=` is not a boolean value.
    """
    if not isinstance(param, bool):
        raise ValueError(f"`{param_name}=` must be a boolean value.")


def _check_column(column: str | list[str]):
    """
    Check the input value of the `column=` parameter.

    Parameters
    ----------
    column
        The column to validate.

    Raises
    ------
    ValueError
        When `column` is not a string.
    """

    if isinstance(column, list):
        if not all(isinstance(col, str) for col in column):
            raise ValueError("If a list is supplied to `column=` all elements must be strings.")
        return

    if isinstance(column, (Column, ColumnSelector)):
        return

    if isinstance(column, nw.selectors.Selector):
        return

    if not isinstance(column, str):
        raise ValueError("`column=` must be a string.")


# TODO: allow for checking of dates/datetimes
def _check_value_float_int(value: float | int | any):
    """
    Check that input value of the `value=` parameter is a float or integer.

    Parameters
    ----------
    value
        The value to compare against in a validation.

    Raises
    ------
    ValueError
        When `value` is not a float or integer.
    """

    from pointblank.column import Column

    if not isinstance(value, (float, int, Column)) or isinstance(value, bool):
        raise ValueError("`value=` must be a float, integer, or reference to a column.")


def _check_set_types(set: list[float | int | str]):
    """
    Check that input value of the `set=` parameter is a list of floats, integers, or strings.

    Parameters
    ----------
    set
        The set of values to compare against in a validation.

    Raises
    ------
    ValueError
        When `set` is not a list of floats or integers.
    """
    if not all(isinstance(value, (float, int, str)) for value in set):
        raise ValueError("`set=` must be a list of floats, integers, or strings.")

    if any(isinstance(value, bool) for value in set):
        raise ValueError("`set=` must not contain boolean values.")


def _check_pre(pre: Callable | None):
    """
    Check that input value of the `pre=` parameter is a callable function.

    Parameters
    ----------
    pre
        The preprocessing function to apply to the table.

    Raises
    ------
    ValueError
        When `pre` is not a callable function.
    """
    if pre is not None and not isinstance(pre, Callable):
        raise ValueError("`pre=` must be a callable function.")


def _check_thresholds(thresholds: int | float | tuple | dict | Thresholds | None):
    """
    Check that input value of the `thresholds=` parameter is a valid threshold.

    Parameters
    ----------
    thresholds
        The threshold value or values.

    Raises
    ------
    ValueError
        When `thresholds` is not a valid threshold.
    """

    if thresholds is None or isinstance(thresholds, Thresholds):
        return

    if isinstance(thresholds, (int, float)):
        if thresholds < 0:
            raise ValueError(
                "If an int or float is supplied to `thresholds=` it must be a non-negative value."
            )

    if isinstance(thresholds, tuple):
        if len(thresholds) > 3:
            raise ValueError(
                "If a tuple is supplied to `thresholds=` it must have at most three elements."
            )
        if not all(isinstance(threshold, (int, float)) for threshold in thresholds):
            raise ValueError(
                "If a tuple is supplied to `thresholds=` all elements must be integers or floats."
            )
        if any(threshold < 0 for threshold in thresholds):
            raise ValueError(
                "If a tuple is supplied to `thresholds=` all elements must be non-negative."
            )

    if isinstance(thresholds, dict):
        # Check keys for invalid entries and raise a ValueError if any are found
        invalid_keys = set(thresholds.keys()) - {"warning", "error", "critical"}

        if invalid_keys:
            raise ValueError(f"Invalid keys in the thresholds dictionary: {invalid_keys}")

        # Get values as a list and raise a ValueError for any non-integer or non-float values
        values = list(thresholds.values())

        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError(
                "If a dict is supplied to `thresholds=` all values must be integers or floats."
            )

        # Raise a ValueError if any values are negative
        if any(value < 0 for value in values):
            raise ValueError(
                "If a dict is supplied to `thresholds=` all values must be non-negative."
            )

    # Raise a ValueError if the thresholds argument is not valid (also accept None)
    if thresholds is not None and not isinstance(thresholds, (int, float, tuple, dict, Thresholds)):
        raise ValueError("The thresholds argument is not valid.")
