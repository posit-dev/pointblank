from __future__ import annotations

import inspect
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import narwhals as nw
from great_tables import GT
from narwhals.dependencies import is_narwhals_dataframe, is_narwhals_lazyframe
from great_tables.gt import _get_column_of_values

from pointblank._constants import ASSERTION_TYPE_METHOD_MAP, GENERAL_COLUMN_TYPES, IBIS_BACKENDS
from pointblank.column import Column, ColumnLiteral, ColumnSelector, ColumnSelectorNarwhals, col

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals.typing import IntoFrame, IntoFrameT

    from pointblank._typing import AbsoluteBounds, Tolerance


def transpose_dicts(list_of_dicts: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if not list_of_dicts:
        return {}

    # Get all unique keys across all dictionaries
    all_keys = set()
    for d in list_of_dicts:
        all_keys.update(d.keys())

    result = defaultdict(list)
    for d in list_of_dicts:
        for key in all_keys:
            result[key].append(d.get(key))  # None is default for missing keys

    return dict(result)


# TODO: doctest
def _derive_single_bound(ref: int, tol: int | float) -> int:
    """Derive a single bound using the reference."""
    if not isinstance(tol, float | int):
        raise TypeError("Tolerance must be a number or a tuple of numbers.")
    if tol < 0:
        raise ValueError("Tolerance must be non-negative.")
    return int(tol * ref) if tol < 1 else int(tol)


# TODO: doctest
def _derive_bounds(ref: int, tol: Tolerance) -> AbsoluteBounds:
    """Validate and extract the absolute bounds of the tolerance."""
    if isinstance(tol, tuple):
        return (_derive_single_bound(ref, tol[0]), _derive_single_bound(ref, tol[1]))

    bound = _derive_single_bound(ref, tol)
    return bound, bound


def _get_tbl_type(data: Any) -> str:
    type_str = str(type(data))

    ibis_tbl = "ibis.expr.types.relations.Table" in type_str

    if not ibis_tbl:
        # TODO: in a later release of Narwhals, there will be a method for getting the namespace:
        # `get_native_namespace()`
        try:
            df_ns_str = str(nw.from_native(data).__native_namespace__())
        except Exception as e:
            raise TypeError("The `data` object is not a DataFrame or Ibis Table.") from e

        # Detect through regex if the table is a polars, pandas, or Spark DataFrame
        if re.search(r"polars", df_ns_str, re.IGNORECASE):
            return "polars"
        elif re.search(r"pandas", df_ns_str, re.IGNORECASE):
            return "pandas"
        elif re.search(r"pyspark", df_ns_str, re.IGNORECASE):
            return "pyspark"

    # If ibis is present, then get the table's backend name
    ibis_present = _is_lib_present(lib_name="ibis")

    if ibis_present:
        import ibis

        # TODO: Getting the backend 'name' is currently a bit brittle right now; as it is,
        #       we either extract the backend name from the table name or get the backend name
        #       from the get_backend() method and name attribute

        backend = ibis.get_backend(data).name

        # Try using the get_name() method to get the table name, this is important for elucidating
        # the original table type since it sometimes gets handled by duckdb

        if backend == "duckdb":
            try:
                tbl_name = data.get_name()
            except AttributeError:  # pragma: no cover
                tbl_name = None

            if tbl_name is not None:
                if "memtable" in tbl_name:
                    return "memtable"

                if "read_parquet" in tbl_name:
                    return "parquet"

            else:  # pragma: no cover
                return "duckdb"

        return backend

    return "unknown"  # pragma: no cover


def _process_ibis_through_narwhals(data: Any, tbl_type: str) -> tuple[Any, str]:
    """
    Process Ibis tables through Narwhals to unify the processing pathway.

    This function takes an Ibis table and wraps it with Narwhals, allowing
    all downstream processing to use the unified Narwhals API instead of
    Ibis-specific code paths.

    Parameters
    ----------
    data
        The data table, potentially an Ibis table
    tbl_type
        The detected table type

    Returns
    -------
    tuple[Any, str]
        A tuple of (processed_data, updated_tbl_type) where:
        - processed_data is the Narwhals-wrapped table if it was Ibis, otherwise original data
        - updated_tbl_type is "narwhals" if it was Ibis, otherwise original tbl_type
    """
    # Check if this is an Ibis table type
    if tbl_type in IBIS_BACKENDS:
        try:
            # Wrap with Narwhals
            narwhals_wrapped = nw.from_native(data)
            return narwhals_wrapped, "narwhals"
        except Exception:
            # If Narwhals can't handle it, fall back to original approach
            return data, tbl_type

    return data, tbl_type


def _is_narwhals_table(data: Any) -> bool:
    # Check if the data is a Narwhals DataFrame
    type_str = str(type(data)).lower()

    if "narwhals" in type_str:
        # If the object is not a Narwhals DataFrame, return False
        return True

    return False


def _is_lazy_frame(data: Any) -> bool:
    # Check if the data is a Polars or Narwhals DataFrame
    type_str = str(type(data)).lower()

    if "polars" not in type_str and "narwhals" not in type_str:
        # If the object is neither a Polars nor a Narwhals DataFrame, return False
        return False

    # Check if the data is a lazy frame
    return "lazy" in type_str


def _is_lib_present(lib_name: str) -> bool:
    import importlib

    try:
        importlib.import_module(lib_name)
        return True
    except ImportError:
        return False


def _check_any_df_lib(method_used: str) -> None:
    # Determine whether Pandas or Polars is available
    pd = None
    try:
        import pandas as pd
    except ImportError:
        pass

    pl = None
    try:
        import polars as pl
    except ImportError:
        pass

    # If neither Pandas nor Polars is available, raise an ImportError
    if pd is None and pl is None:
        raise ImportError(
            f"Using the `{method_used}()` method requires either the "
            "Polars or the Pandas library to be installed."
        )


def _is_value_a_df(value: Any) -> bool:
    try:
        ns = nw.get_native_namespace(value)
        if "polars" in str(ns) or "pandas" in str(ns) or "pyspark" in str(ns):
            return True
        else:  # pragma: no cover
            return False
    except (AttributeError, TypeError):
        return False


def _select_df_lib(preference: str = "polars") -> Any:
    # Determine whether Pandas is available
    pd = None
    try:
        import pandas as pd
    except ImportError:
        pass

    # Determine whether Polars is available
    pl = None
    try:
        import polars as pl
    except ImportError:
        pass

    # TODO: replace this with the `_check_any_df_lib()` function, introduce `method_used=` param
    # If neither Pandas nor Polars is available, raise an ImportError
    if pd is None and pl is None:
        raise ImportError(
            "Generating a report with the `get_tabular_report()` method requires either the "
            "Polars or the Pandas library to be installed."
        )

    # Return the library based on preference, if both are available
    if pd is not None and pl is not None:
        if preference == "polars":
            return pl
        else:
            return pd

    return pl if pl is not None else pd


# TODO: Good argument exceptions should be handled by caller
def _copy_dataframe(df: IntoFrameT) -> IntoFrameT:
    """
    Create a copy of a DataFrame, handling different DataFrame types.

    This function attempts to create a proper copy of the DataFrame using
    the most appropriate method for each DataFrame type.
    """
    # Try standard copy methods first
    if hasattr(df, "copy") and callable(getattr(df, "copy")):
        try:
            return df.copy()
        except Exception:
            pass

    if hasattr(df, "clone") and callable(getattr(df, "clone")):
        try:
            return df.clone()
        except Exception:
            pass

    # Try the select('*') approach for DataFrames that support it
    # This works well for PySpark and other SQL-like DataFrames
    if hasattr(df, "select") and callable(getattr(df, "select")):
        try:
            return df.select("*")
        except Exception:
            pass

    # For DataFrames that can't be copied, return original
    # This provides some protection while avoiding crashes
    try:
        import copy

        return copy.deepcopy(df)
    except Exception:  # pragma: no cover
        # If all else fails, return the original DataFrame
        # This is better than crashing the validation
        return df  # pragma: no cover


# TODO: Should straight up remove this
def _convert_to_narwhals(df: IntoFrame) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
    # Convert the DataFrame to a format that narwhals can work with
    result = nw.from_native(df)
    assert is_narwhals_dataframe(result) or is_narwhals_lazyframe(result)
    return result


def _check_column_exists(dfn: nw.DataFrame[Any] | nw.LazyFrame[Any], column: str) -> None:
    """
    Check if a column exists in a DataFrame.

    Parameters
    ----------
    dfn
        A Narwhals DataFrame or LazyFrame.
    column
        The column to check for existence.

    Raises
    ------
    ValueError
        When the column is not found in the DataFrame.
    """

    if column not in dfn.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame.")


def _count_true_values_in_column(
    tbl: IntoFrame,
    column: str,
    inverse: bool = False,
) -> int:
    """
    Count the number of `True` values in a specified column of a table.

    Parameters
    ----------
    tbl
        A Narwhals-compatible DataFrame or table-like object.
    column
        The column in which to count the `True` values.
    inverse
        If `True`, count the number of `False` values instead.

    Returns
    -------
    int
        The count of `True` (or `False`) values in the specified column.
    """

    # Convert the DataFrame to a Narwhals DataFrame (no detrimental effect if
    # already a Narwhals DataFrame)
    tbl_nw = nw.from_native(tbl)

    # Filter the table based on the column and whether we want to count True or False values
    tbl_filtered = tbl_nw.filter(nw.col(column) if not inverse else ~nw.col(column))

    # Always collect table if it is a LazyFrame; this is required to get the row count
    if is_narwhals_lazyframe(tbl_filtered):
        tbl_filtered = tbl_filtered.collect()

    return len(tbl_filtered)


def _count_null_values_in_column(
    tbl: IntoFrame,
    column: str,
) -> int:
    """
    Count the number of Null values in a specified column of a table.

    Parameters
    ----------
    tbl
        A Narwhals-compatible DataFrame or table-like object.
    column
        The column in which to count the Null values.

    Returns
    -------
    int
        The count of Null values in the specified column.
    """

    # Convert the DataFrame to a Narwhals DataFrame (no detrimental effect if
    # already a Narwhals DataFrame)
    tbl_nw = nw.from_native(tbl)

    # Filter the table to get rows where the specified column is Null
    tbl_filtered = tbl_nw.filter(nw.col(column).is_null())

    # Always collect table if it is a LazyFrame; this is required to get the row count
    if is_narwhals_lazyframe(tbl_filtered):
        tbl_filtered = tbl_filtered.collect()

    return len(tbl_filtered)


def _is_numeric_dtype(dtype: str) -> bool:
    """
    Check if a given data type string represents a numeric type.

    Parameters
    ----------
    dtype
        The data type string to check.

    Returns
    -------
    bool
        `True` if the data type is numeric, `False` otherwise.
    """
    # Define the regular expression pattern for numeric data types
    numeric_pattern = re.compile(r"^(int|float)\d*$")
    return bool(numeric_pattern.match(dtype))


def _is_date_or_datetime_dtype(dtype: str) -> bool:
    """
    Check if a given data type string represents a date or datetime type.

    Parameters
    ----------
    dtype
        The data type string to check.

    Returns
    -------
    bool
        `True` if the data type is date or datetime, `False` otherwise.
    """
    # Define the regular expression pattern for date or datetime data types
    date_pattern = re.compile(r"^(date|datetime).*$")
    return bool(date_pattern.match(dtype))


def _is_duration_dtype(dtype: str) -> bool:
    """
    Check if a given data type string represents a duration type.

    Parameters
    ----------
    dtype
        The data type string to check.

    Returns
    -------
    bool
        `True` if the data type is a duration, `False` otherwise.
    """
    # Define the regular expression pattern for duration data types
    duration_pattern = re.compile(r"^duration.*$")
    return bool(duration_pattern.match(dtype))


def _get_column_dtype(
    dfn: nw.DataFrame[Any] | nw.LazyFrame[Any],
    column: str,
    raw: bool = False,
    lowercased: bool = True,
) -> str | nw.dtypes.DType | None:
    """
    Get the data type of a column in a DataFrame.

    Parameters
    ----------
    dfn
        A Narwhals DataFrame.
    column
        The column from which to get the data type.
    raw
        If `True`, return the raw DType object (or None if column not found).
    lowercased
        If `True`, return the data type string in lowercase.

    Returns
    -------
    str | nw.dtypes.DType | None
        The data type of the column as a string, or the raw DType object if `raw=True`.
    """

    if raw:  # pragma: no cover
        return dfn.collect_schema().get(column)

    column_dtype_str = str(dfn.collect_schema().get(column))

    if lowercased:
        return column_dtype_str.lower()

    return column_dtype_str


def _check_column_type(
    dfn: nw.DataFrame[Any] | nw.LazyFrame[Any], column: str, allowed_types: list[str]
) -> None:
    """
    Check if a column is of a certain data type.

    Parameters
    ----------
    dfn
        A Narwhals DataFrame.
    column
        The column to check for data type.
    dtype
        The data type to check for. These are shorthand types and the following are supported:
        - `"numeric"`: Numeric data types (`int`, `float`)
        - `"str"`: String data type
        - `"bool"`: Boolean data type
        - `"datetime"`: Date or Datetime data type
        - `"duration"`: Duration data type

    Raises
    ------
    TypeError
        When the column is not of the specified data type.
    """

    # Get the data type of the column as a lowercase string
    column_dtype = str(dfn.collect_schema().get(column)).lower()

    # If `allowed_types` is empty, raise a ValueError
    if not allowed_types:
        raise ValueError("No allowed types specified.")

    # If any of the supplied `allowed_types` are not in the `GENERAL_COLUMN_TYPES` list,
    # raise a ValueError
    _check_invalid_fields(fields=allowed_types, valid_fields=GENERAL_COLUMN_TYPES)

    if _is_numeric_dtype(dtype=column_dtype) and "numeric" not in allowed_types:
        raise TypeError(f"Column '{column}' is numeric.")

    if column_dtype == "string" and "str" not in allowed_types:
        raise TypeError(f"Column '{column}' is a string.")

    if column_dtype == "boolean" and "bool" not in allowed_types:
        raise TypeError(f"Column '{column}' is a boolean.")

    if _is_date_or_datetime_dtype(dtype=column_dtype) and "datetime" not in allowed_types:
        raise TypeError(f"Column '{column}' is a date or datetime.")

    if _is_duration_dtype(dtype=column_dtype) and "duration" not in allowed_types:
        raise TypeError(f"Column '{column}' is a duration.")


def _column_test_prep(
    df: IntoFrame, column: str, allowed_types: list[str] | None, check_exists: bool = True
) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
    # Convert the DataFrame to a format that narwhals can work with.
    dfn = _convert_to_narwhals(df=df)

    # Check if the column exists
    if check_exists:
        _check_column_exists(dfn=dfn, column=column)

    # Check if the column is of the allowed types. Raise a TypeError if not.
    if allowed_types:
        _check_column_type(dfn=dfn, column=column, allowed_types=allowed_types)

    return dfn


def _column_subset_test_prep(
    df: IntoFrame, columns_subset: list[str] | None, check_exists: bool = True
) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
    # Convert the DataFrame to a format that narwhals can work with.
    dfn = _convert_to_narwhals(df=df)

    # Check whether all columns exist
    if check_exists and columns_subset:
        for column in columns_subset:
            _check_column_exists(dfn=dfn, column=column)

    return dfn


_PBUnresolvedColumn = str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals
_PBResolvedColumn = Column | ColumnLiteral | ColumnSelectorNarwhals | list[Column] | list[str]


def _resolve_columns(columns: _PBUnresolvedColumn) -> _PBResolvedColumn:
    # If `columns` is a ColumnSelector or Narwhals selector, call `col()` on it to later
    # resolve the columns
    if isinstance(columns, (ColumnSelector, nw.selectors.Selector)):
        columns = col(columns)

    # If `columns` is Column value or a string, place it in a list for iteration
    if isinstance(columns, (Column, str)):
        columns = [columns]

    return columns


def _get_fn_name() -> str | None:
    # Get the current function name
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return None
    return frame.f_back.f_code.co_name


def _get_assertion_from_fname() -> str | None:
    # Get the current function name
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:
        return None
    func_name = frame.f_back.f_code.co_name

    # Use the `ASSERTION_TYPE_METHOD_MAP` dictionary to get the assertion type
    return ASSERTION_TYPE_METHOD_MAP.get(func_name)


def _check_invalid_fields(fields: list[str], valid_fields: list[str]):
    """
    Check if any fields in the list are not in the valid fields list.

    Parameters
    ----------
    fields
        The list of fields to check.
    valid_fields
        The list of valid fields.

    Raises
    ------
    ValueError
        If any field in the list is not in the valid fields list.
    """
    for field in fields:
        if field not in valid_fields:
            raise ValueError(f"Invalid field: {field}")


def _format_to_integer_value(x: int | float, locale: str = "en") -> str:
    """
    Format a numeric value as an integer according to a locale's specifications.

    Parameters
    ----------
    value
        The value to format.

    Returns
    -------
    str
        The formatted integer value.
    """

    if not isinstance(x, (int, float)):
        raise TypeError("The `x=` value must be an integer or float.")

    # Use the built-in Python formatting if Polars isn't present
    if not _is_lib_present(lib_name="polars"):
        return f"{x:,d}"

    import polars as pl

    # Format the value as an integer value
    gt = GT(pl.DataFrame({"x": [x]})).fmt_integer(columns="x", locale=locale)
    formatted_vals = _get_column_of_values(gt, column_name="x", context="html")

    return formatted_vals[0]


def _format_to_float_value(
    x: int | float,
    decimals: int = 2,
    n_sigfig: int | None = None,
    compact: bool = False,
    locale: str = "en",
) -> str:
    """
    Format a numeric value as a float value according to a locale's specifications.

    Parameters
    ----------
    value
        The value to format.

    Returns
    -------
    str
        The formatted float value.
    """

    if not isinstance(x, (int, float)):
        raise TypeError("The `x=` value must be an integer or float.")

    # Use the built-in Python formatting if Polars isn't present
    if not _is_lib_present(lib_name="polars"):
        return f"{x:,.{decimals}f}"

    import polars as pl

    # Format the value as a float value
    gt = GT(pl.DataFrame({"x": [x]})).fmt_number(
        columns="x", decimals=decimals, n_sigfig=n_sigfig, compact=compact, locale=locale
    )
    formatted_vals = _get_column_of_values(gt, column_name="x", context="html")

    return formatted_vals[0]


def _pivot_to_dict(col_dict: Mapping[str, Any]):  # TODO : Type hint and unit test
    result_dict = {}
    for _col, sub_dict in col_dict.items():
        for key, value in sub_dict.items():
            # add columns fields not present
            if key not in result_dict:
                result_dict[key] = [None] * len(col_dict)
            result_dict[key][list(col_dict.keys()).index(_col)] = value
    return result_dict
