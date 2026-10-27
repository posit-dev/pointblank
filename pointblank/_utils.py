from __future__ import annotations

import ast
import functools
import inspect
import linecache
import math
import numbers
import re
import sys
import textwrap
from collections import defaultdict
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Collection

import narwhals as nw
from great_tables import GT
from great_tables.gt import _get_column_of_values
from narwhals.dependencies import is_narwhals_dataframe, is_narwhals_lazyframe, is_polars_lazyframe
from narwhals.utils import Implementation

from pointblank._constants import ASSERTION_TYPE_METHOD_MAP, GENERAL_COLUMN_TYPES, IBIS_BACKENDS
from pointblank.column import Column, ColumnLiteral, ColumnSelector, ColumnSelectorNarwhals, col

if TYPE_CHECKING:
    from collections.abc import Mapping

    from narwhals.dtypes import DType
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

        try:
            backend = ibis.get_backend(data).name
        except Exception:  # sometimes this will fail. Not an expert on why - Tyler Riccio
            namespace = nw.get_native_namespace(nw.from_native(data))
            backend = Implementation.from_native_namespace(namespace).name
            return backend.lower()

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


def _is_real_number(value: Any) -> bool:
    # `Decimal` isn't registered as `numbers.Real`; `bool` is (as an `Integral`) but isn't a number
    # for the purposes of set membership
    return isinstance(value, (numbers.Real, Decimal)) and not isinstance(value, bool)


def _is_float_value(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, numbers.Integral)


def _is_in(column: str, values: Collection[Any], dtype: DType | None) -> nw.Expr:
    """
    Build `nw.col(column).is_in(values)`, aligning numeric values with the column's dtype.

    Polars < 2.0 coerced the column and the values of `is_in()` to a common supertype (e.g., an
    integer column checked against `[1.0, 2.0]`), whereas Polars >= 2.0 raises an
    `InvalidOperationError` when the dtypes differ. Narwhals passes `is_in()` through to the
    backend unchanged, so the values are aligned here instead. Only numeric columns checked against
    purely numeric values are adjusted; anything else (strings, temporal values, mixed-type
    collections) is passed through as is, keeping whatever behavior the backend has for it.

    Parameters
    ----------
    column
        The name of the column to check.
    values
        The values to check membership against. A `None` value is passed through as is.
    dtype
        The Narwhals dtype of `column`. If `None` (dtype unknown), no alignment is performed.

    Returns
    -------
    nw.Expr
        A boolean expression.
    """
    values = list(values)
    non_null = [v for v in values if v is not None]

    if (
        dtype is None
        or not dtype.is_numeric()
        or not non_null
        or not all(_is_real_number(v) for v in non_null)
    ):
        return nw.col(column).is_in(values)

    # NOTE: See https://docs.pola.rs/releases/upgrade/2/#make-coercion-casts-for-is_in-strict-instead-of-lossy
    if dtype.is_float():
        # Compare as floats (an integer or `Decimal` value can't be checked against float data)
        return nw.col(column).is_in([None if v is None else float(v) for v in values])

    if dtype.is_integer() and not all(isinstance(v, numbers.Integral) for v in non_null):
        # A value can only ever match an integer if it is integral, so integral values are
        # converted to integers and all others (fractional values, NaN, infinity, values outside of
        # the 64-bit range) are dropped; this is exact, unlike casting the column to `Float64`
        # (which is lossy above 2**53)
        int_values: list[Any] = [
            None if v is None else int(v)
            for v in values
            if v is None or (math.isfinite(v) and v == int(v) and -(2**63) <= v < 2**64)
        ]
        return nw.col(column).is_in(int_values)

    if dtype.is_decimal() and any(_is_float_value(v) for v in non_null):
        # A `Decimal` column checked against float values is compared as `Float64`
        float_values = [None if v is None else float(v) for v in values]
        return nw.col(column).cast(nw.Float64).is_in(float_values)

    # Integer values against integer data, and integer or `Decimal` values against `Decimal` data,
    # are accepted as is
    return nw.col(column).is_in(values)


def _with_row_index(tbl: Any, name: str) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
    """
    Add a zero-based row index column to a table, following the table's row order.

    Narwhals requires an `order_by=` column for row indexes on LazyFrames. Polars LazyFrames do
    have a positional row order though (and ordering by a column would number the rows in sorted
    order rather than in their actual order), so the native Polars row index is used for those.
    Other LazyFrames (e.g., database tables) have no inherent row order, so they're ordered by
    their first column, with ties broken by the other columns that can be sorted (numeric,
    temporal, string, and boolean ones) so that the numbering doesn't vary from run to run. A
    results column (`pb_is_good_`) isn't used, so a table is numbered the same with or without it.

    Parameters
    ----------
    tbl
        A native table or a Narwhals DataFrame/LazyFrame.
    name
        The name of the row index column.

    Returns
    -------
    nw.DataFrame | nw.LazyFrame
        The Narwhals table with the row index as its first column.
    """
    tbl_nw = nw.from_native(tbl)
    if isinstance(tbl_nw, nw.DataFrame):
        return tbl_nw.with_row_index(name=name)
    native = tbl_nw.to_native()
    if is_polars_lazyframe(native):
        return nw.from_native(native.with_row_index(name=name))
    schema = tbl_nw.collect_schema()
    first_col, *other_cols = schema.names()
    tie_breakers = [
        c
        for c in other_cols
        if c != "pb_is_good_"
        and (
            schema[c].is_numeric()
            or schema[c].is_temporal()
            or schema[c] in (nw.String, nw.Boolean)
        )
    ]
    return tbl_nw.with_row_index(name=name, order_by=[first_col, *tie_breakers])


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

    # For LazyFrames, use aggregation to get the count without materializing all filtered rows
    if is_narwhals_lazyframe(tbl_filtered):
        return int(tbl_filtered.select(nw.len()).collect().item())

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

    # Use aggregation to count null values without materializing the full frame
    # Cast to Int32 before sum to support PySpark which can't sum booleans
    result = tbl_nw.select(nw.col(column).is_null().cast(nw.Int32).sum())

    if is_narwhals_lazyframe(result):
        return int(result.collect().item())
    return int(result.item())


def _count_validation_units(
    tbl: IntoFrame,
    column: str,
) -> tuple[int, int, int, int]:
    """
    Compute the row count and pass/fail/null counts for a results table in a single pass.

    Given a results table with a boolean `column` (typically ``pb_is_good_``), this returns
    the total number of rows, the number of `True` values (passing test units), the number of
    `False` values (failing test units), and the number of Null values.

    Computing all four quantities in one aggregation is important for LazyFrames: otherwise each
    separate count would trigger its own `collect()`, re-executing the entire (potentially
    expensive) lazy plan multiple times.

    Parameters
    ----------
    tbl
        A Narwhals-compatible DataFrame or table-like object.
    column
        The boolean column to summarize.

    Returns
    -------
    tuple[int, int, int, int]
        A tuple of ``(n, n_passed, n_failed, n_null)``.
    """

    # Convert the DataFrame to a Narwhals DataFrame (no detrimental effect if
    # already a Narwhals DataFrame)
    tbl_nw = nw.from_native(tbl)

    # Build a single aggregation that computes all counts at once. Casting booleans to Int32
    # before summing is required for backends like PySpark (which can't sum booleans), and the
    # sums naturally ignore Null values (so `n_passed`/`n_failed` exclude nulls).
    result = tbl_nw.select(
        nw.len().alias("n"),
        nw.col(column).cast(nw.Int32).sum().alias("n_passed"),
        (~nw.col(column)).cast(nw.Int32).sum().alias("n_failed"),
        nw.col(column).is_null().cast(nw.Int32).sum().alias("n_null"),
    )

    if is_narwhals_lazyframe(result):
        result = result.collect()

    row = result.rows(named=True)[0]

    n = int(row["n"])
    n_passed = int(row["n_passed"] or 0)
    n_failed = int(row["n_failed"] or 0)
    n_null = int(row["n_null"] or 0)

    return n, n_passed, n_failed, n_null


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


_PBUnresolvedColumn = (
    str | Collection[str] | Collection[Column] | Column | ColumnSelector | ColumnSelectorNarwhals
)
_PBResolvedColumn = (
    Column | ColumnLiteral | ColumnSelectorNarwhals | Collection[Column] | Collection[str]
)


def _resolve_columns(columns: _PBUnresolvedColumn) -> _PBResolvedColumn:
    # If `columns` is a ColumnSelector or Narwhals selector, call `col()` on it to later
    # resolve the columns
    if isinstance(columns, (ColumnSelector, nw.selectors.Selector)):
        columns = col(columns)

    # If `columns` is Column value or a string, place it in a list for iteration
    if isinstance(columns, (Column, str)):
        columns: list[Column] | list[str] = [columns]

    return columns


def _narwhals_expr_nodes(expr: nw.Expr) -> list[Any]:
    # Narwhals doesn't have a public API for inspecting expressions so this relies on the
    # `_nodes` attribute (present in recent versions); callers must guard against failure
    nodes = list(expr._nodes)
    for node in list(nodes):
        for arg in (*node.exprs, *node.kwargs.values()):
            if isinstance(arg, nw.Expr):
                nodes.extend(_narwhals_expr_nodes(arg))
    return nodes


def _get_expr_root_names(expr: Any) -> list[str]:
    """
    Get the names of the columns referenced by a `col_vals_expr()` expression.

    Polars, Narwhals, and Ibis expressions are supported. An empty list is returned when the
    columns can't be determined (e.g., for callables, PySpark columns, or Polars expressions that
    use `pl.all()` or regex column selection).
    """
    try:
        module = type(expr).__module__.split(".")[0]

        if module == "polars":
            names = expr.meta.root_names()

        elif isinstance(expr, nw.Expr):
            names = [
                name
                for node in _narwhals_expr_nodes(expr)
                if node.name == "col"
                for name in node.kwargs.get("names", [])
            ]

        elif module == "ibis" and hasattr(expr, "op"):
            import ibis.expr.operations as ops

            names = [field.name for field in expr.op().find(ops.Field)]

        else:
            return []

        # De-duplicate while keeping the order in which the columns were referenced
        return list(dict.fromkeys(str(name) for name in names))

    except Exception:
        return []


def _get_expr_alias(expr: Any) -> str | None:
    """
    Get the name explicitly given to a `col_vals_expr()` expression through an alias.

    Supported are Polars (`.alias()`), Narwhals (`.alias()`), and Ibis (`.name()`) expressions.
    `None` is returned when the expression wasn't given a name or the name can't be determined.
    """
    try:
        module = type(expr).__module__.split(".")[0]

        if module == "polars":
            # Only report a name if one was set (otherwise the output name is just the name of
            # the first column in the expression)
            if expr.meta.eq(expr.meta.undo_aliases()):
                return None
            return expr.meta.output_name(raise_if_undetermined=False)

        if isinstance(expr, nw.Expr):
            aliases = [node.kwargs["name"] for node in expr._nodes if node.name == "alias"]
            return str(aliases[-1]) if aliases else None

        if module == "ibis" and hasattr(expr, "op"):
            import ibis.expr.operations as ops

            op = expr.op()
            return str(op.name) if isinstance(op, ops.Alias) else None

    except Exception:
        return None

    return None


@functools.lru_cache(maxsize=8)
def _parse_source(source: str) -> ast.Module:
    # Several steps are usually defined in the same file or notebook cell, so cache the parsing
    return ast.parse(source)


def _get_source_span(
    source: str,
    start: tuple[int, int],
    end: tuple[int, int],
    limit: tuple[int, int],
    expand_parens: bool = False,
) -> str:
    # AST positions are given as line numbers and UTF-8 byte offsets within the lines, so work
    # with the encoded source
    lines = source.encode("utf-8").splitlines(keepends=True)
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))
    source_bytes = b"".join(lines)

    def _offset(position: tuple[int, int]) -> int:
        return line_starts[position[0] - 1] + position[1]

    i_start, i_end, i_limit = _offset(start), _offset(end), _offset(limit)

    # Widen the span to take in any pairs of parentheses enclosing it (the opening one being
    # beyond `limit`, so that the parentheses of the call itself are excluded)
    while expand_parens:
        before = source_bytes[i_limit:i_start].rstrip()
        after = source_bytes[i_end:].lstrip()
        if len(before.lstrip()) > 1 and before.endswith(b"(") and after.startswith(b")"):
            i_start = len(before) - 1 + i_limit
            i_end = len(source_bytes) - len(after) + 1
        else:
            break

    # Pad the first line with spaces so that it lines up with any continuation lines
    line_start = source_bytes.rfind(b"\n", 0, i_start) + 1
    padding = " " * len(source_bytes[line_start:i_start].decode("utf-8"))

    return padding + source_bytes[i_start:i_end].decode("utf-8")


def _get_caller_arg_source(method_name: str, arg_name: str, arg_index: int = 0) -> str | None:
    """
    Get the source text of an argument as written in the user's call to a method.

    This is to be called directly from within the method (e.g., `Validate.col_vals_expr()`). The
    call is located through the exact source position of the caller's current instruction, which
    requires Python 3.11 or later. `None` is returned whenever the text can't be determined with
    certainty: on earlier Python versions, when the source isn't available (e.g., in the standard
    Python REPL or with `exec()`), when the method was called from within pointblank (e.g., when
    building a validation from YAML), or when the argument was supplied through `*args` or
    `**kwargs` unpacking.
    """
    if sys.version_info < (3, 11):
        return None

    frame = inspect.currentframe()

    try:
        # The frame of the code that called the method that called this function
        caller = frame.f_back.f_back

        if caller.f_globals.get("__name__", "").split(".")[0] == "pointblank":
            return None

        _, end_lineno, _, end_col_offset = list(caller.f_code.co_positions())[caller.f_lasti // 2]

        filename = caller.f_code.co_filename
        linecache.checkcache(filename)
        source = "".join(linecache.getlines(filename, caller.f_globals))

        if not source or end_lineno is None or end_col_offset is None:
            return None

        # The call that ends exactly where the caller's current instruction ends must be the only
        # one there and it must be a call of the method
        calls = [
            node
            for node in ast.walk(_parse_source(source))
            if isinstance(node, ast.Call)
            and (node.end_lineno, node.end_col_offset) == (end_lineno, end_col_offset)
        ]
        if len(calls) != 1:
            return None

        call = calls[0]
        func_name = getattr(call.func, "attr", getattr(call.func, "id", None))
        if func_name != method_name:
            return None

        if any(isinstance(arg, ast.Starred) for arg in call.args) or any(
            keyword.arg is None for keyword in call.keywords
        ):
            return None

        keywords = [keyword for keyword in call.keywords if keyword.arg == arg_name]

        if keywords:
            # The keyword's source (e.g., `expr=(...)`) includes any parentheses around the value,
            # which aren't part of the value's own node
            start = (keywords[0].lineno, keywords[0].col_offset)
            end = (keywords[0].end_lineno, keywords[0].end_col_offset)
        elif len(call.args) > arg_index:
            arg_node = call.args[arg_index]
            start = (arg_node.lineno, arg_node.col_offset)
            end = (arg_node.end_lineno, arg_node.end_col_offset)
        else:
            return None

        arg_source = _get_source_span(
            source=source,
            start=start,
            end=end,
            # Parentheses beyond the end of the method name belong to the call itself
            limit=(call.func.end_lineno, call.func.end_col_offset),
            expand_parens=not keywords,
        )

        # Remove the indentation of any continuation lines (the first line is padded so that it
        # gets the same treatment) and then the `expr=` part of a keyword argument
        arg_source = textwrap.dedent(arg_source).strip()
        if keywords:
            arg_source = arg_source[len(arg_name) :].lstrip().removeprefix("=").lstrip()

        return arg_source or None

    except Exception:
        return None

    finally:
        del frame


def _get_fn_name() -> str | None:
    # Get the current function name
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:  # pragma: no cover
        return None
    return frame.f_back.f_code.co_name


def _get_assertion_from_fname() -> str | None:
    # Get the current function name
    frame = inspect.currentframe()
    if frame is None or frame.f_back is None:  # pragma: no cover
        return None
    func_name = frame.f_back.f_code.co_name

    # Use the `ASSERTION_TYPE_METHOD_MAP` dictionary to get the assertion type
    return ASSERTION_TYPE_METHOD_MAP.get(func_name)


def _check_invalid_fields(fields: list[str], valid_fields: list[str]) -> None:
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
