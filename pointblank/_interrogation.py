from __future__ import annotations

import datetime
import functools
import numbers
import operator
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import narwhals as nw
from narwhals.dependencies import (
    is_narwhals_dataframe,
    is_narwhals_lazyframe,
    is_pandas_dataframe,
    is_polars_dataframe,
    is_polars_lazyframe,
)

from pointblank._constants import IBIS_BACKENDS
from pointblank._spec_utils import (
    check_credit_card,
    check_iban,
    check_isbn,
    check_postal_code,
    check_vin,
)
from pointblank._typing import AbsoluteBounds
from pointblank._utils import (
    _column_test_prep,
    _convert_to_narwhals,
    _get_tbl_type,
    _is_in,
    _is_lazy_frame,
    _is_lib_present,
    _with_row_index,
)
from pointblank.column import Column

if TYPE_CHECKING:
    from narwhals.typing import Frame, IntoFrame


class StepEvaluationError(Exception):
    """
    A validation step can't be evaluated against the table (e.g., an invalid expression).

    `Validate.interrogate()` catches this and marks the step with `eval_error=True`, using the
    message as the step's note, instead of letting it stop the whole interrogation.
    """


def _first_line(error: BaseException) -> str:
    """Get a short description of an error (some backends add query plans after the first line)."""
    message = str(error).strip().splitlines()
    return f"{type(error).__name__}: {message[0]}" if message else type(error).__name__


def _native_lib(obj: Any) -> str:
    """Get the name of the library an object comes from (e.g., `"polars"`, `"pandas"`)."""
    return type(obj).__module__.split(".")[0]


def _align_decimal_compare_val(nw_tbl: Any, column: str, compare_val: Any) -> Any:
    """
    Compare a `Decimal` value with a float column as a float.

    pandas compares a float column with a `Decimal` value element by element, which raises
    `decimal.InvalidOperation` as soon as the column has a `NaN`; the other backends compare the
    two as floats anyway. A `Decimal` value against a decimal or integer column is left as is,
    except on an Ibis version that can't compare a decimal column with a `Decimal` literal, where
    the value is compared as a float as well.
    """
    if not isinstance(compare_val, Decimal):
        return compare_val

    dtype = nw_tbl.collect_schema().get(column)
    if dtype is not None and dtype.is_float():
        return float(compare_val)

    if (
        dtype is not None
        and dtype.is_decimal()
        and nw_tbl.implementation.is_ibis()
        and not _ibis_compares_decimal_literal(nw_tbl.to_native(), column)
    ):
        return float(compare_val)

    return compare_val


def _ibis_compares_decimal_literal(ibis_table: Any, column: str) -> bool:
    """
    Check whether Ibis can compare a decimal column with a `Decimal` literal.

    Ibis 11 types a `Decimal` literal as a decimal without precision and scale, which it then
    refuses to compare with a `decimal(p, s)` column (`IbisTypeError: ... are not comparable`);
    Ibis 12 accepts the comparison.
    """
    try:
        ibis_table[column] > Decimal(0)
    except Exception:
        return False
    return True


def _safe_modify_datetime_compare_val(data_frame: Any, column: str, compare_val: Any) -> Any:
    """
    Safely modify datetime comparison values for LazyFrame compatibility.

    This function handles the case where we can't directly slice LazyFrames
    to get column dtypes for datetime conversion.
    """
    try:
        # First try to get column dtype from schema for LazyFrames
        column_dtype = None

        if is_narwhals_lazyframe(data_frame):
            schema = data_frame.collect_schema()
            column_dtype = schema.get(column)
        elif is_narwhals_dataframe(data_frame):
            schema = data_frame.schema
            column_dtype = schema.get(column)

        # If we got a dtype from schema, use it
        if column_dtype is not None:
            # Create a mock column object for _modify_datetime_compare_val
            class MockColumn:
                def __init__(self, dtype) -> None:
                    self.dtype = dtype

            mock_column = MockColumn(column_dtype)
            return _modify_datetime_compare_val(tgt_column=mock_column, compare_val=compare_val)

        # Fallback: try collecting a small sample if possible
        try:
            sample = data_frame.head(1).collect()
            if hasattr(sample, "dtypes") and column in sample.columns:
                # For pandas-like dtypes
                column_dtype = sample.dtypes[column] if hasattr(sample, "dtypes") else None
                if column_dtype:

                    class MockColumn:
                        def __init__(self, dtype) -> None:
                            self.dtype = dtype

                    mock_column = MockColumn(column_dtype)
                    return _modify_datetime_compare_val(
                        tgt_column=mock_column, compare_val=compare_val
                    )
        except Exception:
            pass

        # Final fallback: try direct access (for eager DataFrames)
        try:
            if hasattr(data_frame, "dtypes") and column in data_frame.columns:
                column_dtype = data_frame.dtypes[column]

                class MockColumn:
                    def __init__(self, dtype) -> None:
                        self.dtype = dtype

                mock_column = MockColumn(column_dtype)
                return _modify_datetime_compare_val(tgt_column=mock_column, compare_val=compare_val)
        except Exception:
            pass

    except Exception:  # pragma: no cover
        pass

    # If all else fails, return the original compare_val
    return compare_val


def _safe_is_nan_or_null_expr(data_frame: IntoFrame, column_expr: nw.Expr, column_name: str) -> Any:
    """
    Create an expression that safely checks for both Null and NaN values.

    This function handles the case where `is_nan()` is not supported for certain data types (like
    strings or Polars decimals) or backends (like `SQLite` via Ibis) by checking the backend type
    and column type first. Only float columns can hold NaN, so only those get the `is_nan()` check.

    Parameters
    ----------
    data_frame
        The data frame to get schema information from.
    column_expr
        The narwhals column expression to check.
    column_name
        The name of the column.

    Returns
    -------
    Any
        A narwhals expression that returns `True` for Null or NaN values.
    """
    null_check = column_expr.is_null()

    df: Frame = nw.from_native(data_frame, allow_series=False)

    schema: nw.Schema = df.collect_schema()

    dtype: nw.dtypes.DType = schema[column_name]

    if not dtype.is_float():
        return null_check

    # Some Ibis backends (e.g. SQLite) don't support is_nan(), so only check nulls for those
    if df.implementation.is_ibis() and not _ibis_backend_supports_is_nan(df.to_native()):
        return null_check

    return null_check | column_expr.is_nan()


def _ibis_backend_supports_is_nan(ibis_table: Any) -> bool:
    """
    Check whether the backend of an Ibis table can evaluate `is_nan()` (e.g., DuckDB can but
    SQLite can't).
    """
    try:
        import ibis.expr.operations as ops

        return bool(ibis_table.get_backend().has_operation(ops.IsNan))
    except Exception:
        return False


def _conjointly_expression_error(i: int, reason: str) -> StepEvaluationError:
    return StepEvaluationError(
        f"Expression {i} passed to conjointly() couldn't be evaluated ({reason})."
    )


class ConjointlyValidation:
    def __init__(self, data_tbl, expressions: type[nw.Expr], threshold, tbl_type) -> None:
        self.data_tbl = data_tbl
        self.expressions = expressions
        self.threshold = threshold

        # Detect the table type
        if tbl_type in (None, "local"):
            # Detect the table type using _get_tbl_type()
            self.tbl_type = _get_tbl_type(data=data_tbl)
        else:
            self.tbl_type = tbl_type

        # Conjointly validation requires eager frames (uses .to_series())
        # Collect lazy frames here since this validation inherently needs materialized data
        if _is_lazy_frame(self.data_tbl):
            self.data_tbl = self.data_tbl.collect()

    def get_test_results(self):
        """Evaluate all expressions and combine them conjointly."""

        if "polars" in self.tbl_type:
            get_results = self._get_polars_results
        elif "pandas" in self.tbl_type:
            get_results = self._get_pandas_results
        elif "duckdb" in self.tbl_type or "ibis" in self.tbl_type or self.tbl_type in IBIS_BACKENDS:
            get_results = self._get_ibis_results
        elif "pyspark" in self.tbl_type:
            get_results = self._get_pyspark_results
        else:  # pragma: no cover
            raise NotImplementedError(f"Support for {self.tbl_type} is not yet implemented")

        try:
            results_tbl = get_results()
            return _fill_null_as_failing(results_tbl, source="conjointly() expressions")
        except StepEvaluationError:
            raise
        except Exception as e:
            raise StepEvaluationError(
                f"The conjointly() expressions couldn't be evaluated ({_first_line(e)})."
            ) from e

    def _resolve_expressions(
        self, is_native: Callable[[Any], bool], converter: str, *converter_args: Any
    ) -> list[Any]:
        """
        Turn each expression into a native boolean expression (or Series) for the table.

        Each expression is a function that, given the table, returns either a native expression
        or a `pb.expr_col()` expression. The latter (which can also be built without the table) is
        converted with its `converter` method. An expression that can't be resolved raises a
        `StepEvaluationError` rather than being skipped, which would make it pass every row.
        """
        if not self.expressions:  # pragma: no cover
            raise StepEvaluationError("conjointly() needs at least one expression.")

        resolved = []
        for i, expr_fn in enumerate(self.expressions, start=1):
            error = None
            try:
                result = expr_fn(self.data_tbl)
            except Exception as e:
                result, error = None, e

            if result is not None and is_native(result):
                resolved.append(result)
                continue

            col_expr = result
            if not hasattr(col_expr, converter):
                try:
                    col_expr = expr_fn(None)
                except Exception:
                    col_expr = None

            if hasattr(col_expr, converter):
                try:
                    resolved.append(getattr(col_expr, converter)(*converter_args))
                    continue
                except Exception as e:
                    reason = _first_line(e)
            elif error is not None:
                reason = _first_line(error)
            else:
                reason = f"it returned a {type(result).__name__}, not a boolean expression"

            raise _conjointly_expression_error(i, reason)

        return resolved

    def _get_polars_results(self):
        """Process expressions for Polars DataFrames."""
        import polars as pl

        resolved = self._resolve_expressions(
            lambda x: isinstance(x, (pl.Expr, pl.Series)), "to_polars_expr"
        )

        # Evaluate expressions to Series so they can be combined with any Series results
        series = []
        for i, x in enumerate(resolved, start=1):
            try:
                series.append(
                    x if isinstance(x, pl.Series) else self.data_tbl.select(x).to_series()
                )
            except Exception as e:
                raise _conjointly_expression_error(i, _first_line(e)) from e
        return self.data_tbl.with_columns(pb_is_good_=functools.reduce(operator.and_, series))

    def _get_pandas_results(self):
        """Process expressions for pandas DataFrames."""
        import pandas as pd

        resolved = self._resolve_expressions(
            lambda x: isinstance(x, pd.Series), "to_pandas_expr", self.data_tbl
        )

        results_tbl = self.data_tbl.copy()
        results_tbl["pb_is_good_"] = functools.reduce(operator.and_, resolved)
        return results_tbl

    def _get_ibis_results(self):
        """Process expressions for Ibis tables (including DuckDB)."""
        import ibis

        resolved = self._resolve_expressions(
            lambda x: isinstance(x, ibis.expr.types.Value), "to_ibis_expr", self.data_tbl
        )
        return self.data_tbl.mutate(pb_is_good_=functools.reduce(operator.and_, resolved))

    def _get_pyspark_results(self):
        """Process expressions for PySpark DataFrames."""
        resolved = self._resolve_expressions(
            lambda x: _native_lib(x) == "pyspark" and type(x).__name__ == "Column",
            "to_pyspark_expr",
            self.data_tbl,
        )
        return self.data_tbl.withColumn("pb_is_good_", functools.reduce(operator.and_, resolved))


class SpeciallyValidation:
    def __init__(self, data_tbl, expression, threshold, tbl_type) -> None:
        self.data_tbl = data_tbl
        self.expression = expression
        self.threshold = threshold

        # Detect the table type
        if tbl_type in (None, "local"):
            # Detect the table type using _get_tbl_type()
            self.tbl_type = _get_tbl_type(data=data_tbl)
        else:
            self.tbl_type = tbl_type

        # User-defined specially() functions expect eager frames for backwards compatibility
        if _is_lazy_frame(self.data_tbl):
            self.data_tbl = self.data_tbl.collect()

    def get_test_results(self) -> Any | list[bool]:
        """Evaluate the expression get either a list of booleans or a results table."""

        # Get the expression and inspect whether there is a `data` argument
        expression = self.expression

        import inspect

        # During execution of `specially` validation
        sig = inspect.signature(expression)
        params = list(sig.parameters.keys())

        # Execute the function based on its signature
        if len(params) == 0:
            # No parameters: call without arguments
            result = expression()
        elif len(params) == 1:
            # One parameter: pass the data table
            data_tbl = self.data_tbl
            result = expression(data_tbl)
        else:
            # More than one parameter: this doesn't match either allowed signature
            raise ValueError(
                f"The function provided to 'specially()' should have either no parameters or a "
                f"single 'data' parameter, but it has {len(params)} parameters: {params}"
            )

        # Determine if the object is a DataFrame by inspecting the string version of its type
        if (
            "pandas" in str(type(result))
            or "polars" in str(type(result))
            or "ibis" in str(type(result))
        ):
            # Get the type of the table
            tbl_type = _get_tbl_type(data=result)

            if "pandas" in tbl_type:
                # If it's a Pandas DataFrame, check if the last column is a boolean column
                last_col = result.iloc[:, -1]

                import pandas as pd

                if last_col.dtype == bool or pd.api.types.is_bool_dtype(last_col):
                    # If the last column is a boolean column, rename it as `pb_is_good_`
                    result = result.rename(columns={result.columns[-1]: "pb_is_good_"})
            elif "polars" in tbl_type:
                # If it's a Polars DataFrame, check if the last column is a boolean column
                last_col_name = result.columns[-1]
                last_col_dtype = result.schema[last_col_name]

                import polars as pl

                if last_col_dtype == pl.Boolean:
                    # If the last column is a boolean column, rename it as `pb_is_good_`
                    result = result.rename({last_col_name: "pb_is_good_"})
            elif tbl_type in IBIS_BACKENDS:
                # If it's an Ibis table, check if the last column is a boolean column
                last_col_name = result.columns[-1]
                result_schema = result.schema()
                is_last_col_bool = str(result_schema[last_col_name]) == "boolean"

                if is_last_col_bool:
                    # If the last column is a boolean column, rename it as `pb_is_good_`
                    result = result.rename(pb_is_good_=last_col_name)

            else:  # pragma: no cover
                raise NotImplementedError(f"Support for {tbl_type} is not yet implemented")

        elif isinstance(result, bool):
            # If it's a single boolean, return that as a list
            return [result]

        elif isinstance(result, list):
            # If it's a list, check that it is a boolean list
            if all(isinstance(x, bool) for x in result):
                # If it's a list of booleans, return it as is
                return result
            else:
                # If it's not a list of booleans, raise an error
                raise TypeError("The result is not a list of booleans.")
        else:  # pragma: no cover
            # If it's not a DataFrame or a list, raise an error
            raise TypeError("The result is not a DataFrame or a list of booleans.")

        # Return the results table or list of booleans
        return result


@dataclass
class NumberOfTestUnits:
    """
    Count the number of test units in a column.
    """

    df: Any  # Can be IntoFrame or Ibis table
    column: str

    def get_test_units(self, tbl_type: str) -> int:
        if (
            tbl_type == "pandas"
            or tbl_type == "polars"
            or tbl_type == "pyspark"
            or tbl_type == "local"
        ):
            # Convert the DataFrame to a format that narwhals can work with and:
            #  - check if the column exists
            dfn = _column_test_prep(
                df=self.df, column=self.column, allowed_types=None, check_exists=False
            )

            # Use lazy-compatible row count (avoids collecting the entire frame)
            if is_narwhals_lazyframe(dfn):
                return dfn.select(nw.len()).collect().item()

            assert is_narwhals_dataframe(dfn)
            return len(dfn)

        if tbl_type in IBIS_BACKENDS:
            # Get the count of test units and convert to a native format
            # TODO: check whether pandas or polars is available
            return self.df.count().to_polars()

        raise ValueError(f"Unsupported table type: {tbl_type}")  # pragma: no cover


def _get_compare_expr_nw(compare: Any) -> Any:
    if isinstance(compare, Column):
        if not isinstance(compare.exprs, str):
            raise ValueError("The column expression must be a string.")  # pragma: no cover
        return nw.col(compare.exprs)
    return compare


def _check_nulls_across_columns_nw(table, columns_subset):
    # Get all column names from the table
    column_names = columns_subset if columns_subset else table.columns

    # Build the expression by combining each column's `is_null()` with OR operations
    null_expr = functools.reduce(
        lambda acc, col: acc | nw.col(col).is_null() if acc is not None else nw.col(col).is_null(),
        column_names,
        None,
    )

    # Add the expression as a new column to the table
    result = table.with_columns(_any_is_null_=null_expr)

    return result


def _modify_datetime_compare_val(tgt_column: Any, compare_val: Any) -> Any:
    tgt_col_dtype_str = str(tgt_column.dtype).lower()

    if compare_val is isinstance(compare_val, Column):  # pragma: no cover
        return compare_val

    # Get the type of `compare_expr` and convert, if necessary, to the type of the column
    compare_type_str = str(type(compare_val)).lower()

    if "datetime.datetime" in compare_type_str:
        compare_type = "datetime"
    elif "datetime.date" in compare_type_str:
        compare_type = "date"
    else:
        compare_type = "other"

    if "datetime" in tgt_col_dtype_str:
        tgt_col_dtype = "datetime"
    elif "date" in tgt_col_dtype_str or "object" in tgt_col_dtype_str:
        # Object type is used for date columns in Pandas
        tgt_col_dtype = "date"
    else:
        tgt_col_dtype = "other"

    # Handle each combination of `compare_type` and `tgt_col_dtype`, coercing only the
    # `compare_expr` to the type of the column
    if compare_type == "datetime" and tgt_col_dtype == "date":
        # Assume that `compare_expr` is a datetime.datetime object and strip the time part
        # to get a date object
        compare_expr = compare_val.date()

    elif compare_type == "date" and tgt_col_dtype == "datetime":
        import datetime

        # Assume that `compare_expr` is a `datetime.date` object so add in the time part
        # to get a `datetime.datetime` object
        compare_expr = datetime.datetime.combine(compare_val, datetime.datetime.min.time())

    else:
        return compare_val

    return compare_expr


def _fill_null_as_failing(tbl: Any, source: str) -> Any:
    """
    Check that the `pb_is_good_` results are boolean, and make null results fail.

    Null results come from expressions on null values (e.g., `pl.col("a") > 1` where `a` is null).
    Non-boolean results raise a `StepEvaluationError` instead of being judged by truthiness.
    """
    tbl_nw = nw.from_native(tbl)
    dtype = tbl_nw.collect_schema()["pb_is_good_"]

    # pandas gives an `object` column for booleans mixed with nulls (e.g., `.str.contains()`)
    is_boolean = dtype == nw.Boolean or (
        dtype == nw.Object
        and {type(v).__name__ for v in tbl_nw.get_column("pb_is_good_").drop_nulls().to_list()}
        <= {"bool", "bool_"}
    )
    if not is_boolean:
        raise StepEvaluationError(
            f"The {source} must evaluate to boolean values, but the result has the {dtype} type."
        )

    return tbl_nw.with_columns(
        pb_is_good_=nw.col("pb_is_good_").fill_null(False).cast(nw.Boolean)
    ).to_native()


def _apply_row_expr(data_tbl: Any, expr: Any) -> Any:
    """Add the result of a `col_vals_expr()` expression to the table as `pb_is_good_`."""
    # A function (e.g., a lambda for a pandas table) is given the table and returns the result;
    # Ibis deferred expressions (`_.a > 1`) are callable but are resolved by Ibis itself
    if callable(expr) and _native_lib(expr) != "ibis":
        expr = expr(data_tbl)

    # Narwhals expressions work on every backend
    if isinstance(expr, nw.Expr):
        return nw.from_native(data_tbl).with_columns(pb_is_good_=expr).to_native()

    tbl_lib = _native_lib(data_tbl)
    expr_lib = _native_lib(expr)

    if tbl_lib == "polars" and expr_lib == "polars":
        return data_tbl.with_columns(pb_is_good_=expr)
    if tbl_lib == "pandas" and expr_lib in ("pandas", "numpy"):
        return data_tbl.assign(pb_is_good_=expr)
    if tbl_lib == "ibis" and expr_lib == "ibis":
        return data_tbl.mutate(pb_is_good_=expr)
    if tbl_lib == "pyspark" and expr_lib == "pyspark":
        return data_tbl.withColumn("pb_is_good_", expr)

    raise StepEvaluationError(
        f"The expr= expression is a {expr_lib} {type(expr).__name__}, which can't be used with a "
        f'{tbl_lib} table. Use a Narwhals expression (e.g., nw.col("a") > 1) or one from {tbl_lib}.'
    )


def col_vals_expr(data_tbl: Any, expr: Any, tbl_type: str = "local") -> Any:
    """Check if values in a column evaluate to True for a given predicate expression."""
    try:
        results_tbl = _apply_row_expr(data_tbl=data_tbl, expr=expr)
        return _fill_null_as_failing(results_tbl, source="expr= expression")
    except StepEvaluationError:
        raise
    except Exception as e:
        raise StepEvaluationError(
            f"The expr= expression couldn't be evaluated ({_first_line(e)})."
        ) from e


def rows_complete(data_tbl: IntoFrame, columns_subset: list[str] | None) -> Any:
    """
    Check if rows in a DataFrame are complete (no null values).

    This function replaces the RowsComplete dataclass for direct usage.
    """
    return interrogate_rows_complete(
        tbl=data_tbl,
        columns_subset=columns_subset,
    )


def col_exists(data_tbl: IntoFrame, column: str) -> bool:
    """
    Check if a column exists in a DataFrame.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.

    Returns
    -------
    bool
        `True` if the column exists, `False` otherwise.
    """
    return col_exists_details(data_tbl=data_tbl, column=column)["passed"]


def col_exists_details(data_tbl: IntoFrame, column: str) -> dict:
    """
    Check if a column exists in a DataFrame, returning the details for reporting.

    The returned dictionary has the result (`passed`) along with the `column` checked and the
    columns of the table (`tbl_columns`).
    """
    tbl = _convert_to_narwhals(df=data_tbl)
    tbl_columns = list(tbl.columns)
    return {"passed": column in tbl_columns, "column": column, "tbl_columns": tbl_columns}


def col_schema_match(
    data_tbl: IntoFrame,
    schema: Any,
    complete: bool,
    in_order: bool,
    case_sensitive_colnames: bool,
    case_sensitive_dtypes: bool,
    full_match_dtypes: bool,
    threshold: int,
) -> bool:
    """
    Check if DataFrame schema matches expected schema.
    """
    from pointblank.schema import _check_schema_match

    return _check_schema_match(
        data_tbl=data_tbl,
        schema=schema,
        complete=complete,
        in_order=in_order,
        case_sensitive_colnames=case_sensitive_colnames,
        case_sensitive_dtypes=case_sensitive_dtypes,
        full_match_dtypes=full_match_dtypes,
    )


def row_count_match(
    data_tbl: IntoFrame, count: Any, inverse: bool, abs_tol_bounds: AbsoluteBounds
) -> bool:
    """
    Check if DataFrame row count matches expected count.
    """
    return row_count_match_details(
        data_tbl=data_tbl, count=count, inverse=inverse, abs_tol_bounds=abs_tol_bounds
    )["passed"]


def row_count_match_details(
    data_tbl: IntoFrame, count: Any, inverse: bool, abs_tol_bounds: AbsoluteBounds
) -> dict:
    """
    Check if DataFrame row count matches expected count, returning the details for reporting.

    The returned dictionary has the result (`passed`) along with the observed row count
    (`actual`), the expected `count`, and the range of matching row counts (`min_val`, `max_val`).
    """
    from pointblank.validate import get_row_count

    row_count: int = get_row_count(data=data_tbl)
    lower_abs_limit, upper_abs_limit = abs_tol_bounds
    min_val: int = count - lower_abs_limit
    max_val: int = count + upper_abs_limit

    in_range = row_count >= min_val and row_count <= max_val

    return {
        "passed": not in_range if inverse else in_range,
        "actual": row_count,
        "count": count,
        "min_val": min_val,
        "max_val": max_val,
        "inverse": inverse,
    }


def col_pct_null(
    data_tbl: IntoFrame, column: str, p: float, bound_finder: Callable[[int], AbsoluteBounds]
) -> bool:
    """Check if the percentage of null vales are within p given the absolute bounds."""
    return col_pct_null_details(data_tbl=data_tbl, column=column, p=p, bound_finder=bound_finder)[
        "passed"
    ]


def col_pct_null_details(
    data_tbl: IntoFrame, column: str, p: float, bound_finder: Callable[[int], AbsoluteBounds]
) -> dict:
    """
    Check if the percentage of null values are within p given the absolute bounds, returning the
    details for reporting.

    The returned dictionary has the result (`passed`) along with the number of null values
    (`n_null`), the number of rows (`total_rows`), the expected percentage (`p`), and the range of
    passing null counts (`min_null`, `max_null`).
    """
    nw_frame = nw.from_native(data_tbl)

    # Use lazy-compatible aggregation to get total rows and null count
    # Cast boolean to Int32 before sum to support PySpark which can't sum booleans
    if is_narwhals_lazyframe(nw_frame):
        stats = nw_frame.select(
            total_rows=nw.len(),
            n_null=nw.col(column).is_null().cast(nw.Int32).sum(),
        ).collect()
        total_rows: int = int(stats["total_rows"][0])
        n_null: int = int(stats["n_null"][0])
    else:
        assert is_narwhals_dataframe(nw_frame)
        total_rows = int(nw_frame.select(nw.len()).item())
        n_null = int(nw_frame.select(nw.col(column).is_null().cast(nw.Int32).sum()).item())

    abs_target: float = round(total_rows * p)
    lower_bound, upper_bound = bound_finder(abs_target)
    min_null = abs_target - lower_bound
    max_null = abs_target + upper_bound

    return {
        "passed": n_null >= min_null and n_null <= max_null,
        "n_null": n_null,
        "total_rows": total_rows,
        "p": p,
        "min_null": min_null,
        "max_null": max_null,
    }


def col_pct_missing(
    data_tbl: IntoFrame,
    column: str,
    sentinels: list,
    count_null: bool,
    max_pct: float,
) -> bool:
    """Check that the percentage of missing values in a column does not exceed `max_pct`.

    Missing values are those equal to one of the `sentinels` and, when `count_null=True`, actual
    null (or NaN) values. The percentage is computed over the total number of rows.
    """
    return col_pct_missing_details(
        data_tbl=data_tbl,
        column=column,
        sentinels=sentinels,
        count_null=count_null,
        max_pct=max_pct,
    )["passed"]


def col_pct_missing_details(
    data_tbl: IntoFrame,
    column: str,
    sentinels: list,
    count_null: bool,
    max_pct: float,
) -> dict:
    """
    Check that the percentage of missing values in a column does not exceed `max_pct`, returning
    the details for reporting.

    The returned dictionary has the result (`passed`) along with the number of missing values
    (`n_missing`), the number of rows (`total_rows`), the percentage of missing values
    (`pct_missing`), and `max_pct`. The number of rows isn't counted when nothing counts as missing
    (it's `None` then).
    """
    nw_frame = nw.from_native(data_tbl)

    # Build a boolean expression that flags missing values
    missing_expr = None
    if sentinels:
        missing_expr = _is_in(column, sentinels, nw_frame.collect_schema()[column])
    if count_null:
        null_expr = _safe_is_nan_or_null_expr(data_tbl, nw.col(column), column)
        missing_expr = null_expr if missing_expr is None else (missing_expr | null_expr)

    if missing_expr is None:
        # Nothing counts as missing under this spec/filter
        return {
            "passed": 0.0 <= max_pct,
            "n_missing": 0,
            "total_rows": None,
            "pct_missing": 0.0,
            "max_pct": max_pct,
        }

    # Cast boolean to Int32 before sum to support PySpark which can't sum booleans
    if is_narwhals_lazyframe(nw_frame):
        stats = nw_frame.select(
            total_rows=nw.len(),
            n_missing=missing_expr.cast(nw.Int32).sum(),
        ).collect()
        total_rows: int = int(stats["total_rows"][0])
        n_missing: int = int(stats["n_missing"][0])
    else:
        assert is_narwhals_dataframe(nw_frame)
        total_rows = int(nw_frame.select(nw.len()).item())
        n_missing = int(nw_frame.select(missing_expr.cast(nw.Int32).sum()).item())

    if total_rows == 0:
        return {
            "passed": True,
            "n_missing": n_missing,
            "total_rows": total_rows,
            "pct_missing": None,
            "max_pct": max_pct,
        }

    pct_missing = n_missing / total_rows
    return {
        "passed": pct_missing <= max_pct,
        "n_missing": n_missing,
        "total_rows": total_rows,
        "pct_missing": pct_missing,
        "max_pct": max_pct,
    }


def col_count_match(data_tbl: IntoFrame, count: Any, inverse: bool) -> bool:
    """
    Check if DataFrame column count matches expected count.
    """
    return col_count_match_details(data_tbl=data_tbl, count=count, inverse=inverse)["passed"]


def col_count_match_details(data_tbl: IntoFrame, count: Any, inverse: bool) -> dict:
    """
    Check if DataFrame column count matches expected count, returning the details for reporting.

    The returned dictionary has the result (`passed`) along with the observed column count
    (`actual`) and the expected `count`.
    """
    from pointblank.validate import get_column_count

    col_count: int = get_column_count(data=data_tbl)

    return {
        "passed": col_count != count if inverse else col_count == count,
        "actual": col_count,
        "count": count,
        "inverse": inverse,
    }


def _coerce_to_common_backend(data_tbl: Any, tbl_compare: Any) -> tuple[Any, Any]:
    """
    Coerce two tables to the same backend if they differ.

    If the tables to compare have different backends (e.g., one is Polars and one is Pandas),
    the data table is materialized first if it's a database (Ibis) table, and the comparison table
    is then converted directly to the data table's (resulting) backend. Converting directly (rather
    than through pandas) keeps dtypes intact: e.g., a nullable integer column stays an integer.

    Parameters
    ----------
    data_tbl
        The primary table (materialized to Polars, or pandas, if it's a database table).
    tbl_compare
        The comparison table (converted to match the data table's backend).

    Returns
    -------
    tuple[Any, Any]
        Both tables, on a common backend where possible.

    Raises
    ------
    StepEvaluationError
        If a table can't be converted.
    """
    data_backend = _get_tbl_type(data_tbl)
    compare_backend = _get_tbl_type(tbl_compare)

    if data_backend == compare_backend:
        return data_tbl, tbl_compare

    database_backends = set(IBIS_BACKENDS)
    eager_backend = "polars" if _is_lib_present("polars") else "pandas"

    def convert(tbl: Any, from_backend: str, to_backend: str, which: str) -> Any:
        try:
            if from_backend in database_backends:
                return tbl.to_polars() if to_backend == "polars" else tbl.to_pandas()
            if from_backend == "pandas" and to_backend == "polars":
                import polars as pl

                return pl.from_pandas(tbl)
            if from_backend == "polars" and to_backend == "pandas":
                return tbl.to_pandas()
        except Exception as e:
            raise StepEvaluationError(
                f"The {which} table couldn't be converted from {from_backend} to {to_backend} "
                f"for the comparison ({_first_line(e)})."
            ) from e
        # Other backends (e.g., PySpark) are left as they are
        return tbl

    # Materialize a database data table first, so that the comparison table is converted to the
    # backend the data table ends up with
    if data_backend in database_backends:
        data_tbl = convert(data_tbl, data_backend, eager_backend, "target")
        data_backend = eager_backend

    if compare_backend != data_backend:
        tbl_compare = convert(tbl_compare, compare_backend, data_backend, "comparison")

    return data_tbl, tbl_compare


def conjointly_validation(
    data_tbl: IntoFrame, expressions: Any, threshold: int, tbl_type: str = "local"
) -> Any:
    """
    Perform conjoint validation using multiple expressions.
    """
    # Create a ConjointlyValidation instance and get the results
    conjointly_instance = ConjointlyValidation(
        data_tbl=data_tbl,
        expressions=expressions,
        threshold=threshold,
        tbl_type=tbl_type,
    )

    return conjointly_instance.get_test_results()


# TODO: we can certainly simplify this
def interrogate_gt(tbl: IntoFrame, column: str, compare: Any, na_pass: bool) -> Any:
    """Greater than interrogation."""
    return _interrogate_comparison_base(tbl, column, compare, na_pass, "gt")


def interrogate_lt(tbl: IntoFrame, column: str, compare: Any, na_pass: bool) -> Any:
    """Less than interrogation."""
    return _interrogate_comparison_base(tbl, column, compare, na_pass, "lt")


def interrogate_ge(tbl: IntoFrame, column: str, compare: Any, na_pass: bool) -> Any:
    """Greater than or equal interrogation."""
    return _interrogate_comparison_base(tbl, column, compare, na_pass, "ge")


def interrogate_le(tbl: IntoFrame, column: str, compare: Any, na_pass: bool) -> Any:
    """Less than or equal interrogation."""
    return _interrogate_comparison_base(tbl, column, compare, na_pass, "le")


def interrogate_eq(tbl: IntoFrame, column: str, compare: Any, na_pass: bool) -> Any:
    """Equal interrogation."""
    return _interrogate_comparison_base(tbl, column, compare, na_pass, "eq")


def interrogate_ne(tbl: IntoFrame, column: str, compare: Any, na_pass: bool) -> Any:
    """Not equal interrogation."""
    return _interrogate_comparison_base(tbl, column, compare, na_pass, "ne")


def interrogate_between(
    tbl: IntoFrame, column: str, low: Any, high: Any, inclusive: tuple[bool, bool], na_pass: bool
) -> Any:
    """Between interrogation."""

    low_val = _get_compare_expr_nw(compare=low)
    high_val = _get_compare_expr_nw(compare=high)

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))
    _check_comparable(nw_tbl, column, low)
    _check_comparable(nw_tbl, column, high)
    low_val = _safe_modify_datetime_compare_val(nw_tbl, column, low_val)
    high_val = _safe_modify_datetime_compare_val(nw_tbl, column, high_val)
    low_val = _align_decimal_compare_val(nw_tbl, column, low_val)
    high_val = _align_decimal_compare_val(nw_tbl, column, high_val)

    result_tbl = nw_tbl.with_columns(
        pb_is_good_1=_safe_is_nan_or_null_expr(nw_tbl, nw.col(column), column),  # val is missing
        pb_is_good_2=(  # lb is missing in Column
            _safe_is_nan_or_null_expr(nw_tbl, nw.col(low.name), low.name)
            if isinstance(low, Column)
            else nw.lit(False)
        ),
        pb_is_good_3=(  # ub is missing in Column
            _safe_is_nan_or_null_expr(nw_tbl, nw.col(high.name), high.name)
            if isinstance(high, Column)
            else nw.lit(False)
        ),
        pb_is_good_4=nw.lit(na_pass),  # Pass if any Null in lb, val, or ub
    )

    if inclusive[0]:
        result_tbl = result_tbl.with_columns(pb_is_good_5=nw.col(column) >= low_val)
    else:
        result_tbl = result_tbl.with_columns(pb_is_good_5=nw.col(column) > low_val)

    if inclusive[1]:
        result_tbl = result_tbl.with_columns(pb_is_good_6=nw.col(column) <= high_val)
    else:
        result_tbl = result_tbl.with_columns(pb_is_good_6=nw.col(column) < high_val)

    # A missing value or bound fails both comparisons, so only `na_pass=` decides the row
    # (`NaN` is greater than every number on Polars and DuckDB, so `x <= NaN` would be `True`)
    any_missing = nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3")

    result_tbl = result_tbl.with_columns(
        pb_is_good_5=(
            nw.when(nw.col("pb_is_good_5").is_null() | any_missing)
            .then(nw.lit(False))
            .otherwise(nw.col("pb_is_good_5"))
        )
    )

    result_tbl = result_tbl.with_columns(
        pb_is_good_6=(
            nw.when(nw.col("pb_is_good_6").is_null() | any_missing)
            .then(nw.lit(False))
            .otherwise(nw.col("pb_is_good_6"))
        )
    )

    result_tbl = result_tbl.with_columns(
        pb_is_good_=(
            (
                (nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3"))
                & nw.col("pb_is_good_4")
            )
            | (nw.col("pb_is_good_5") & nw.col("pb_is_good_6"))
        )
    ).drop(
        "pb_is_good_1",
        "pb_is_good_2",
        "pb_is_good_3",
        "pb_is_good_4",
        "pb_is_good_5",
        "pb_is_good_6",
    )

    return result_tbl.to_native()


def interrogate_outside(
    tbl: IntoFrame, column: str, low: Any, high: Any, inclusive: tuple[bool, bool], na_pass: bool
) -> Any:
    """Outside range interrogation."""

    low_val = _get_compare_expr_nw(compare=low)
    high_val = _get_compare_expr_nw(compare=high)

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))
    _check_comparable(nw_tbl, column, low)
    _check_comparable(nw_tbl, column, high)
    low_val = _safe_modify_datetime_compare_val(nw_tbl, column, low_val)
    high_val = _safe_modify_datetime_compare_val(nw_tbl, column, high_val)
    low_val = _align_decimal_compare_val(nw_tbl, column, low_val)
    high_val = _align_decimal_compare_val(nw_tbl, column, high_val)

    result_tbl = nw_tbl.with_columns(
        pb_is_good_1=_safe_is_nan_or_null_expr(nw_tbl, nw.col(column), column),  # val is missing
        pb_is_good_2=(  # lb is missing in Column
            _safe_is_nan_or_null_expr(nw_tbl, nw.col(low.name), low.name)
            if isinstance(low, Column)
            else nw.lit(False)
        ),
        pb_is_good_3=(  # ub is missing in Column
            _safe_is_nan_or_null_expr(nw_tbl, nw.col(high.name), high.name)
            if isinstance(high, Column)
            else nw.lit(False)
        ),
        pb_is_good_4=nw.lit(na_pass),  # Pass if any Null in lb, val, or ub
    )

    # Note: Logic is inverted for "outside"; when inclusive[0] is True,
    # we want values < low_val (not <= low_val) to be "outside"
    if inclusive[0]:
        result_tbl = result_tbl.with_columns(pb_is_good_5=nw.col(column) < low_val)
    else:
        result_tbl = result_tbl.with_columns(pb_is_good_5=nw.col(column) <= low_val)

    if inclusive[1]:
        result_tbl = result_tbl.with_columns(pb_is_good_6=nw.col(column) > high_val)
    else:
        result_tbl = result_tbl.with_columns(pb_is_good_6=nw.col(column) >= high_val)

    # A missing value or bound fails both comparisons, so only `na_pass=` decides the row
    # (`NaN` is greater than every number on Polars and DuckDB, so `x <= NaN` would be `True`)
    any_missing = nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3")

    result_tbl = result_tbl.with_columns(
        pb_is_good_5=nw.when(nw.col("pb_is_good_5").is_null() | any_missing)
        .then(False)
        .otherwise(nw.col("pb_is_good_5")),
        pb_is_good_6=nw.when(nw.col("pb_is_good_6").is_null() | any_missing)
        .then(False)
        .otherwise(nw.col("pb_is_good_6")),
    )

    result_tbl = result_tbl.with_columns(
        pb_is_good_=(
            (
                (nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3"))
                & nw.col("pb_is_good_4")
            )
            | (
                (nw.col("pb_is_good_5") & ~nw.col("pb_is_good_3"))
                | (nw.col("pb_is_good_6")) & ~nw.col("pb_is_good_2")
            )
        )
    ).drop(
        "pb_is_good_1",
        "pb_is_good_2",
        "pb_is_good_3",
        "pb_is_good_4",
        "pb_is_good_5",
        "pb_is_good_6",
    )

    return result_tbl.to_native()


def interrogate_isin(tbl: IntoFrame, column: str, set_values: Any) -> Any:
    """In set interrogation."""

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))

    can_be_null: bool = None in set_values
    # `is_in()` gives null for a missing value on Polars and nullable pandas dtypes, which is
    # neither passing nor failing in the sundered data and the extracts
    base_expr: nw.Expr = _is_in(column, set_values, nw_tbl.collect_schema()[column]).fill_null(
        False
    )
    if can_be_null:
        # `NaN` counts as missing too, as in the comparison methods
        base_expr = base_expr | _safe_is_nan_or_null_expr(nw_tbl, nw.col(column), column)

    result_tbl = nw_tbl.with_columns(pb_is_good_=base_expr)
    return result_tbl.to_native()


def _value_kind(value: Any) -> str | None:
    """Classify a Python comparison value like `_dtype_kind()` does dtypes (`None` if other)."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, numbers.Number):
        return "numeric"
    if isinstance(value, datetime.date):  # includes datetimes
        return "temporal"
    if isinstance(value, str):
        return "string"
    return None


def _check_comparable(nw_tbl: Any, column: str, compare: Any) -> None:
    """
    Raise a `StepEvaluationError` if a column can't be compared with a value or another column.

    Comparing, e.g., a numeric column with a string crashes on some backends and silently gives
    wrong results on others (pandas compares `1 == "x"` as False; Polars compares dates with
    integers). Columns that aren't in the table are skipped (their own error is clearer), as are
    types that can't be classified. Date and datetime columns can be compared with date strings.
    """
    schema = nw_tbl.collect_schema()
    if column not in schema:
        return
    kind = _dtype_kind(schema[column])

    if isinstance(compare, Column):
        if not isinstance(compare.exprs, str) or compare.exprs not in schema:
            return
        other_kind = _dtype_kind(schema[compare.exprs])
        other = f"column '{compare.exprs}' ({schema[compare.exprs]})"
    else:
        other_kind = _value_kind(compare)
        other = f"the value {compare!r}"
        if kind == "temporal" and other_kind == "string":
            return

    if kind is not None and other_kind is not None and kind != other_kind:
        raise StepEvaluationError(
            f"Column '{column}' ({schema[column]}) can't be compared with {other}."
        )


def _dtype_kind(dtype: Any) -> str | None:
    """Classify a Narwhals dtype as numeric, string, temporal, or boolean (`None` if other)."""
    if dtype.is_numeric():
        return "numeric"
    if dtype.is_temporal():
        return "temporal"
    if dtype in (nw.String, nw.Categorical, nw.Enum):
        return "string"
    if dtype == nw.Boolean:
        return "boolean"
    return None


def _check_key_dtypes_compatible(
    columns: list[str], ref_columns: list[str], tbl_schema: Any, ref_schema: Any
) -> None:
    """Raise a `TypeError` if a key column can never match its reference column's values."""
    for c, rc in zip(columns, ref_columns):
        kind, ref_kind = _dtype_kind(tbl_schema[c]), _dtype_kind(ref_schema[rc])
        if kind is not None and ref_kind is not None and kind != ref_kind:
            raise TypeError(
                f"Column '{c}' ({tbl_schema[c]}) and reference column '{rc}' "
                f"({ref_schema[rc]}) have incompatible types."
            )


def _common_numeric_key_dtype(dtype: Any, ref_dtype: Any) -> Any:
    """Get the dtype that mismatched numeric join keys should be cast to (`None` if no cast)."""
    if dtype == ref_dtype or not (dtype.is_numeric() and ref_dtype.is_numeric()):
        return None
    if dtype.is_integer() and ref_dtype.is_integer():
        # `UInt64` values can't all be represented by `Int64` (or vice versa), so such a pair is
        # left to the backend
        if dtype == nw.UInt64 or ref_dtype == nw.UInt64:
            return None
        return nw.Int64
    return nw.Float64


def interrogate_in_table(
    tbl: IntoFrame,
    columns: str | list[str],
    ref_tbl: IntoFrame,
    ref_columns: str | list[str],
    na_pass: bool,
) -> Any:
    """Referential integrity interrogation.

    Checks that each value (or composite key) in the target column(s) exists in the reference
    table's column(s). Returns the table with a `pb_is_good_` boolean column.

    Heterogeneous backends (e.g. MySQL table + Polars DataFrame) are handled via
    `_coerce_to_common_backend()`, which materializes the lighter side to match the heavier side's
    backend before comparison.
    """
    # Coerce both tables to the same backend
    tbl, ref_tbl = _coerce_to_common_backend(tbl, ref_tbl)

    nw_tbl = nw.from_native(tbl)
    nw_ref = nw.from_native(ref_tbl)

    # Normalize to lists
    single_col = isinstance(columns, str)
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(ref_columns, str):
        ref_columns = [ref_columns]

    if len(columns) != len(ref_columns):
        raise ValueError(
            f"columns and ref_column must have the same length, "
            f"got {len(columns)} and {len(ref_columns)}."
        )

    tbl_schema = nw_tbl.collect_schema()
    ref_schema = nw_ref.collect_schema()
    _check_key_dtypes_compatible(columns, ref_columns, tbl_schema, ref_schema)

    if single_col:
        # Single-column path: extract distinct ref values, use is_in()
        col_name = columns[0]
        ref_col_name = ref_columns[0]

        ref_unique = nw_ref.select(nw.col(ref_col_name)).unique()
        if isinstance(ref_unique, nw.LazyFrame):  # pragma: no cover
            ref_unique = ref_unique.collect()
        ref_values = ref_unique.get_column(ref_col_name).to_list()
        ref_values_clean = [v for v in ref_values if v is not None]

        # A missing key (null, or `NaN` as in the comparison methods) never counts as found, even
        # when the reference column has missing values too: pandas stores those as `NaN`, which
        # `is_in()` would match; so it passes only with `na_pass=True`, on every backend
        in_ref = _is_in(col_name, ref_values_clean, tbl_schema[col_name]).fill_null(False)
        is_missing = _safe_is_nan_or_null_expr(nw_tbl, nw.col(col_name), col_name)
        expr: nw.Expr = (in_ref & ~is_missing) | (is_missing & na_pass)

        result_tbl = nw_tbl.with_columns(pb_is_good_=expr)
        return result_tbl.to_native()

    # Composite-key path: left-join with distinct ref keys + marker
    ref_keys = nw_ref.select(ref_columns).unique()
    if isinstance(ref_keys, nw.LazyFrame):  # pragma: no cover
        ref_keys = ref_keys.collect()

    # Rename ref columns to match target columns (required for join)
    rename_map = {src: tgt for src, tgt in zip(ref_columns, columns) if src != tgt}
    if rename_map:
        ref_keys = ref_keys.rename(rename_map)

    ref_keys = ref_keys.with_columns(nw.lit(True).alias("__pb_ref_matched__"))

    # Materialize lazy main table for the join
    if isinstance(nw_tbl, nw.LazyFrame):  # pragma: no cover
        nw_tbl = nw_tbl.collect()

    # Joins require the key columns on both sides to have the same dtype, so mismatched numeric
    # keys (e.g., an integer key against a float reference key) are cast to a common dtype; this
    # only applies to the join keys, the returned table keeps its original dtypes
    key_casts = {
        c: dtype
        for c, rc in zip(columns, ref_columns)
        if (dtype := _common_numeric_key_dtype(tbl_schema[c], ref_schema[rc])) is not None
    }
    tbl_keys = nw_tbl.select(columns)
    if key_casts:
        tbl_keys = tbl_keys.with_columns(nw.col(c).cast(d) for c, d in key_casts.items())
        ref_keys = ref_keys.with_columns(nw.col(c).cast(d) for c, d in key_casts.items())

    joined = tbl_keys.join(ref_keys, on=columns, how="left")

    # The marker is `True` or null after the left join; pandas stores that as an `object` column,
    # so it is cast to make it a proper boolean column
    matched_series = joined.get_column("__pb_ref_matched__").fill_null(value=False).cast(nw.Boolean)

    # As with a single column, a key with a missing value never counts as found (the joins of
    # Polars and DuckDB match `NaN` to `NaN`, while pandas treats it as null); a key that is
    # missing altogether passes with `na_pass=True`
    is_missing = [_safe_is_nan_or_null_expr(nw_tbl, nw.col(c), c) for c in columns]
    missing_flags = nw_tbl.select(
        nw.any_horizontal(*is_missing, ignore_nulls=False).alias("__pb_any_missing__"),
        nw.all_horizontal(*is_missing, ignore_nulls=False).alias("__pb_all_missing__"),
    )
    matched_series = matched_series & ~missing_flags.get_column("__pb_any_missing__")

    if na_pass:
        matched_series = matched_series | missing_flags.get_column("__pb_all_missing__")

    result_tbl = nw_tbl.with_columns(pb_is_good_=matched_series)
    return result_tbl.to_native()


def interrogate_notin(tbl: IntoFrame, column: str, set_values: Any, na_pass: bool = False) -> Any:
    """Not in set interrogation."""

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))
    # Missing values fail unless `na_pass=True`; `is_in()` gives null for them on most backends
    # (and False for NaN in pandas), so make the result a plain boolean
    in_set = _is_in(column, set_values, nw_tbl.collect_schema()[column]).fill_null(False)
    is_null = _safe_is_nan_or_null_expr(nw_tbl, nw.col(column), column)
    result_tbl = nw_tbl.with_columns(pb_is_good_=(~in_set & ~is_null) | (is_null & na_pass))
    return result_tbl.to_native()


def interrogate_regex(
    tbl: IntoFrame, column: str, values: dict[str, Any] | str, na_pass: bool
) -> Any:
    """Regex interrogation."""

    # Handle both old and new formats for backward compatibility
    if isinstance(values, str):
        pattern = values
        inverse = False
    else:
        pattern = values["pattern"]
        inverse = values["inverse"]

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))

    column_expr = _string_expr(nw_tbl, column)

    result_tbl = nw_tbl.with_columns(
        pb_is_good_1=nw.col(column).is_null() & na_pass,
        pb_is_good_2=column_expr.str.contains(pattern, literal=False).fill_null(False),
    )

    # Apply inverse logic if needed
    if inverse:
        # Use explicit boolean logic instead of bitwise NOT for pandas compatibility; a missing
        # value doesn't pass either way, so it's left to `na_pass=` (through `pb_is_good_1`)
        result_tbl = result_tbl.with_columns(
            pb_is_good_2=nw.when(nw.col("pb_is_good_2") | nw.col(column).is_null())
            .then(False)
            .otherwise(True)
        )

    result_tbl = result_tbl.with_columns(
        pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2")
    ).drop("pb_is_good_1", "pb_is_good_2")

    return result_tbl.to_native()


def _string_expr(nw_tbl: Any, column: str) -> nw.Expr:
    """
    The column as an expression that the `.str` methods can be used on.

    Polars has no string methods for Categorical and Enum columns (nor pandas for pyarrow
    dictionary columns), so those are cast to strings. Narwhals < 2.27 casts a missing value in a
    pandas `category` column to the string "nan", so missing values are kept missing.
    """
    if nw_tbl.collect_schema()[column] not in (nw.Categorical, nw.Enum):
        return nw.col(column)
    return nw.when(~nw.col(column).is_null()).then(nw.col(column).cast(nw.String))


def interrogate_str_len(tbl: IntoFrame, column: str, values: dict, na_pass: bool) -> Any:
    """String length interrogation."""

    min_val = values.get("min_val")
    max_val = values.get("max_val")

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))

    column_expr = _string_expr(nw_tbl, column)

    result_tbl = nw_tbl.with_columns(
        pb_is_good_1=nw.col(column).is_null() & na_pass,
        pb_is_good_2=nw.lit(True),
        pb_is_good_3=nw.lit(True),
    )

    if min_val is not None:
        result_tbl = result_tbl.with_columns(
            pb_is_good_2=(column_expr.str.len_chars() >= min_val).fill_null(False)
        )

    if max_val is not None:
        result_tbl = result_tbl.with_columns(
            pb_is_good_3=(column_expr.str.len_chars() <= max_val).fill_null(False)
        )

    result_tbl = result_tbl.with_columns(
        pb_is_good_=(nw.col("pb_is_good_1") | (nw.col("pb_is_good_2") & nw.col("pb_is_good_3")))
    ).drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")

    return result_tbl.to_native()


def interrogate_within_spec(
    tbl: IntoFrame, column: str, values: dict[str, Any], na_pass: bool
) -> Any:
    """Within specification interrogation."""
    from pointblank._spec_utils import (
        regex_email,
        regex_ipv4_address,
        regex_ipv6_address,
        regex_mac,
        regex_phone,
        regex_swift_bic,
        regex_url,
    )

    spec = values["spec"]
    spec_lower = spec.lower()

    # Parse spec for country-specific formats
    country = None
    if "[" in spec and "]" in spec:
        # Extract country code from spec like "postal_code[US]" or "iban[DE]"
        base_spec = spec[: spec.index("[")]
        country = spec[spec.index("[") + 1 : spec.index("]")]
        spec_lower = base_spec.lower()

    # Convert to Narwhals for cross-backend compatibility
    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))

    # Regex-based specifications can use Narwhals directly (no materialization needed)
    regex_specs = {
        "email": regex_email(),
        "url": regex_url(),
        "phone": regex_phone(),
        "ipv4": regex_ipv4_address(),
        "ipv4_address": regex_ipv4_address(),
        "ipv6": regex_ipv6_address(),
        "ipv6_address": regex_ipv6_address(),
        "mac": regex_mac(),
        "mac_address": regex_mac(),
        "swift": regex_swift_bic(),
        "swift_bic": regex_swift_bic(),
        "bic": regex_swift_bic(),
    }

    if spec_lower in regex_specs:
        # Use regex validation through Narwhals (works for all backends including Ibis!)
        pattern = regex_specs[spec_lower]

        col_expr = _string_expr(nw_tbl, column)

        # For SWIFT/BIC, need to uppercase first
        if spec_lower in ("swift", "swift_bic", "bic"):
            col_expr = col_expr.str.to_uppercase()

        result_tbl = nw_tbl.with_columns(
            pb_is_good_1=nw.col(column).is_null() & na_pass,
            pb_is_good_2=col_expr.str.contains(f"^{pattern}$", literal=False).fill_null(False),
        )

        result_tbl = result_tbl.with_columns(
            pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2")
        ).drop("pb_is_good_1", "pb_is_good_2")

        return result_tbl.to_native()

    # For specifications requiring checksums or complex logic:
    # Auto-detect Ibis tables and use database-native validation when available
    native_tbl = nw_tbl.to_native()
    is_ibis = hasattr(native_tbl, "execute")

    # Use database-native validation for VIN and credit_card when using Ibis
    if is_ibis and spec_lower == "vin":  # pragma: no cover
        # Route to database-native VIN validation
        return interrogate_within_spec_db(tbl, column, values, na_pass)
    elif is_ibis and spec_lower in ("credit_card", "creditcard"):  # pragma: no cover
        # Route to database-native credit card validation
        return interrogate_credit_card_db(tbl, column, values, na_pass)

    # For non-Ibis tables or other specs, materialize data and use Python validation; a Polars
    # LazyFrame is collected here (it can't be indexed by column) and given back as a LazyFrame
    is_polars_lazy = is_polars_lazyframe(native_tbl)
    if is_polars_lazy:
        nw_tbl = nw_tbl.collect()

    # Get the column data as a list
    col_data: Any = nw_tbl.select(column).to_native()

    # Convert to list based on backend - type varies so use duck typing
    if hasattr(col_data, "to_list"):  # pragma: no cover  # Polars
        col_list = col_data[column].to_list()  # type: ignore[index]
    elif hasattr(col_data, "tolist"):  # pragma: no cover  # Pandas
        col_list = col_data[column].tolist()  # type: ignore[index]
    else:  # pragma: no cover  # For Ibis tables, we need to execute the query first
        try:
            # Try to execute if it's an Ibis table
            if hasattr(col_data, "execute"):
                col_data_exec = col_data.execute()  # type: ignore[operator]
                if hasattr(col_data_exec, "to_list"):  # Polars result
                    col_list = col_data_exec[column].to_list()
                elif hasattr(col_data_exec, "tolist"):  # Pandas result
                    col_list = col_data_exec[column].tolist()
                else:
                    col_list = list(col_data_exec[column])
            else:
                col_list = list(col_data[column])
        except Exception:
            # Fallback to direct list conversion
            col_list = list(col_data[column])

    assert isinstance(col_list, list)

    # Validate based on spec type (checksum-based validations)
    if spec_lower in ("isbn", "isbn-10", "isbn-13"):
        is_valid_list = check_isbn(col_list)
    elif spec_lower == "vin":
        is_valid_list = check_vin(col_list)
    elif spec_lower in ("credit_card", "creditcard"):
        is_valid_list = check_credit_card(col_list)
    elif spec_lower == "iban":
        is_valid_list = check_iban(col_list, country=country)
    elif spec_lower in ("postal_code", "postalcode", "postcode", "zip"):
        if country is None:
            raise ValueError("Country code required for postal code validation")
        is_valid_list = check_postal_code(col_list, country=country)
    else:
        raise ValueError(f"Unknown specification type: {spec}")

    # Create result table with validation results
    # For Ibis tables, execute to get a materialized dataframe first
    native_tbl = nw_tbl.to_native()
    if hasattr(native_tbl, "execute"):  # pragma: no cover
        native_tbl = native_tbl.execute()

    # Add validation column: convert native table to Series, then back through Narwhals
    if is_polars_dataframe(native_tbl):
        import polars as pl

        native_tbl = native_tbl.with_columns(pb_is_good_2=pl.Series(is_valid_list))
    elif is_pandas_dataframe(native_tbl):
        import pandas as pd

        native_tbl["pb_is_good_2"] = pd.Series(is_valid_list, index=native_tbl.index)
    else:
        raise NotImplementedError(
            f"Backend type not supported: {type(native_tbl)}"
        )  # pragma: no cover

    result_tbl = nw.from_native(native_tbl)  # Handle NA values and combine validation results
    result_tbl = result_tbl.with_columns(
        pb_is_good_1=nw.col(column).is_null() & na_pass,
    )

    result_tbl = result_tbl.with_columns(
        pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2")
    ).drop("pb_is_good_1", "pb_is_good_2")

    if is_polars_lazy:
        return result_tbl.to_native().lazy()

    return result_tbl.to_native()


def interrogate_within_spec_db(  # pragma: no cover
    tbl: IntoFrame, column: str, values: dict[str, Any], na_pass: bool
) -> Any:
    """
    Database-native specification validation (proof of concept).

    This function uses Ibis expressions to perform validation entirely in SQL,
    avoiding data materialization for remote database tables. Currently only
    supports VIN validation as a proof of concept.

    Parameters
    ----------
    tbl
        The table to interrogate (must be an Ibis table).
    column
        The column to validate.
    values
        Dictionary containing 'spec' key with specification type.
    na_pass
        Whether to pass null values.

    Returns
    -------
    Any
        Result table with pb_is_good_ column indicating validation results.

    Notes
    -----
    This is a proof-of-concept implementation demonstrating database-native
    validation. It translates complex Python validation logic (regex, checksums)
    into SQL expressions that can be executed directly in the database.
    """
    spec = values["spec"]
    spec_lower = spec.lower()

    # Check if this is an Ibis table
    native_tbl: Any = tbl
    if is_narwhals_dataframe(tbl) or is_narwhals_lazyframe(tbl):
        native_tbl = tbl.to_native()

    is_ibis = hasattr(native_tbl, "execute")

    if not is_ibis:
        # Fall back to regular implementation for non-Ibis tables
        return interrogate_within_spec(tbl, column, values, na_pass)

    # Route to appropriate database-native validation
    if spec_lower == "credit_card":
        return interrogate_credit_card_db(tbl, column, values, na_pass)
    elif spec_lower != "vin":
        raise NotImplementedError(
            f"Database-native validation for '{spec}' not yet implemented. "
            "Currently 'vin' and 'credit_card' are supported in interrogate_within_spec_db(). "
            "Use interrogate_within_spec() for other specifications."
        )

    # VIN validation using Ibis expressions (database-native)
    # Implementation based on ISO 3779 standard with check digit algorithm
    try:
        import ibis
    except ImportError:
        raise ImportError("Ibis is required for database-native validation")

    # VIN transliteration map (character to numeric value for checksum)
    # Based on ISO 3779 standard for VIN check digit calculation
    transliteration = {
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7,
        "H": 8,
        "J": 1,
        "K": 2,
        "L": 3,
        "M": 4,
        "N": 5,
        "P": 7,
        "R": 9,
        "S": 2,
        "T": 3,
        "U": 4,
        "V": 5,
        "W": 6,
        "X": 7,
        "Y": 8,
        "Z": 9,
        "0": 0,
        "1": 1,
        "2": 2,
        "3": 3,
        "4": 4,
        "5": 5,
        "6": 6,
        "7": 7,
        "8": 8,
        "9": 9,
    }

    # Position weights for checksum calculation
    weights = [8, 7, 6, 5, 4, 3, 2, 10, 0, 9, 8, 7, 6, 5, 4, 3, 2]

    # Get the column as an Ibis expression
    col_expr = native_tbl[column]  # type: ignore[index]

    # Basic checks: length must be 17, no invalid characters (I, O, Q)
    valid_length = col_expr.length() == 17
    no_invalid_chars = (
        ~col_expr.upper().contains("I")
        & ~col_expr.upper().contains("O")
        & ~col_expr.upper().contains("Q")
    )

    # Calculate checksum using Ibis expressions
    # For each position, extract character, transliterate to number, multiply by weight, sum
    checksum = ibis.literal(0)

    for pos in range(17):
        if pos == 8:  # Position 9 (0-indexed 8) is the check digit itself
            continue

        # Extract character at position (1-indexed for substr)
        char = col_expr.upper().substr(pos, 1)

        # Build a case expression for transliteration using ibis.cases()
        # Add final else condition for invalid characters
        conditions = [(char == ch, num) for ch, num in transliteration.items()]
        value = ibis.cases(*conditions, else_=0)  # Default: invalid char = 0 (will fail validation)

        # Multiply by weight and add to checksum
        checksum = checksum + (value * weights[pos])  # type: ignore[operator]

    # Check digit calculation: checksum % 11
    # If result is 10, check digit should be 'X', otherwise it's the digit itself
    expected_check = checksum % 11  # type: ignore[operator]
    actual_check_char = col_expr.upper().substr(8, 1)  # Position 9 (0-indexed 8)

    # Validate check digit using ibis.cases()
    check_digit_valid = ibis.cases(
        (expected_check == 10, actual_check_char == "X"),
        (expected_check < 10, actual_check_char == expected_check.cast(str)),
        else_=False,
    )

    # Combine all validation checks
    is_valid = valid_length & no_invalid_chars & check_digit_valid

    # Handle NULL values
    if na_pass:
        # NULL values should pass when na_pass=True
        is_valid = col_expr.isnull() | is_valid
    else:
        # NULL values should explicitly fail when na_pass=False
        # Use fill_null to convert NULL results to False
        is_valid = is_valid.fill_null(False)

    # Add validation column to table
    result_tbl = native_tbl.mutate(pb_is_good_=is_valid)  # type: ignore[union-attr]

    return result_tbl


def interrogate_credit_card_db(  # pragma: no cover
    tbl: IntoFrame, column: str, values: dict[str, str], na_pass: bool
) -> Any:
    """
    Database-native credit card validation using Luhn algorithm in SQL.

    This function implements the Luhn checksum algorithm entirely in SQL using
    Ibis expressions, avoiding data materialization for remote database tables.
    This is a unique implementation that validates credit card numbers directly
    in the database.

    Parameters
    ----------
    tbl
        The table to interrogate (must be an Ibis table).
    column
        The column to validate.
    values
        Dictionary containing 'spec' key (should be 'credit_card').
    na_pass
        Whether to pass null values.

    Returns
    -------
    Any
        Result table with pb_is_good_ column indicating validation results.

    Notes
    -----
    The Luhn algorithm works as follows:
    1. Remove spaces and hyphens from the card number
    2. Starting from the rightmost digit, double every second digit
    3. If doubled digit > 9, subtract 9
    4. Sum all digits
    5. Valid if sum % 10 == 0

    This implementation translates the entire algorithm into SQL expressions.
    """
    # Check if this is an Ibis table
    native_tbl = tbl
    if hasattr(tbl, "to_native"):
        native_tbl = tbl.to_native() if callable(tbl.to_native) else tbl  # type: ignore[operator]

    is_ibis = hasattr(native_tbl, "execute")

    if not is_ibis:
        # Fall back to regular implementation for non-Ibis tables
        return interrogate_within_spec(tbl, column, values, na_pass)

    try:
        import ibis
    except ImportError:
        raise ImportError("Ibis is required for database-native validation")

    # Get the column as an Ibis expression
    col_expr = native_tbl[column]  # type: ignore[index]

    # Step 1: Clean the input and remove spaces and hyphens
    # First check format: only digits, spaces, and hyphens allowed
    valid_chars = col_expr.re_search(r"^[0-9\s\-]+$").notnull()

    # Clean: remove spaces and hyphens
    clean_card = col_expr.replace(" ", "").replace("-", "")

    # Step 2: Check length (13-19 digits after cleaning)
    card_length = clean_card.length()
    valid_length = (card_length >= 13) & (card_length <= 19)

    # Step 3: Luhn algorithm implementation in SQL
    # We'll process each digit position and calculate the checksum
    # Starting from the right, double every second digit

    # Initialize checksum
    checksum = ibis.literal(0)

    # Process up to 19 digits (maximum credit card length)
    for pos in range(19):
        # Calculate position from right (0 = rightmost)
        pos_from_right = pos

        # Extract digit at this position from the right
        # substr with negative index or using length - pos
        digit_pos = card_length - pos_from_right
        digit_char = clean_card.substr(digit_pos - 1, 1)

        # Convert character to integer (using case statement)
        digit_val = ibis.cases(
            (digit_char == "0", 0),
            (digit_char == "1", 1),
            (digit_char == "2", 2),
            (digit_char == "3", 3),
            (digit_char == "4", 4),
            (digit_char == "5", 5),
            (digit_char == "6", 6),
            (digit_char == "7", 7),
            (digit_char == "8", 8),
            (digit_char == "9", 9),
            else_=-1,  # Invalid character
        )

        # Check if this position should be processed (within card length)
        in_range = digit_pos > 0

        # Double every second digit (odd positions from right, 0-indexed)
        should_double = (pos_from_right % 2) == 1

        # Calculate contribution to checksum
        # If should_double: double the digit, then if > 9 subtract 9
        doubled = digit_val * 2  # type: ignore[operator]
        adjusted = ibis.cases(
            (should_double & (doubled > 9), doubled - 9),
            (should_double, doubled),
            else_=digit_val,
        )

        # Add to checksum only if in range
        contribution = ibis.cases(
            (in_range, adjusted),
            else_=0,
        )

        checksum = checksum + contribution  # type: ignore[operator]

    # Step 4: Valid if checksum % 10 == 0
    luhn_valid = (checksum % 10) == 0  # type: ignore[operator]

    # Combine all validation checks
    is_valid = valid_chars & valid_length & luhn_valid

    # Handle NULL values
    if na_pass:
        # NULL values should pass when na_pass=True
        is_valid = col_expr.isnull() | is_valid
    else:
        # NULL values should explicitly fail when na_pass=False
        is_valid = is_valid.fill_null(False)

    # Add validation column to table
    result_tbl = native_tbl.mutate(pb_is_good_=is_valid)  # type: ignore[union-attr]

    return result_tbl


def interrogate_null(tbl: IntoFrame, column: str) -> Any:
    """Null interrogation."""

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))
    result_tbl = nw_tbl.with_columns(pb_is_good_=nw.col(column).is_null())
    return result_tbl.to_native()


def interrogate_not_null(tbl: IntoFrame, column: str) -> Any:
    """Not null interrogation."""

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))
    result_tbl = nw_tbl.with_columns(pb_is_good_=~nw.col(column).is_null())
    return result_tbl.to_native()


def apply_missing_exclusion(results_tbl: IntoFrame, column: str, spec: Any) -> Any:
    """Mark rows with structured-missing values as passing.

    Given a `results_tbl` that already carries a boolean `pb_is_good_` column, force that column to
    `True` for any row whose value in `column` is a declared sentinel of `spec` (a `MissingSpec`),
    or a null when `spec.null_is_missing` is `True`. This implements the `missing=` exclusion on
    `col_vals_*` validation methods: sentinel/missing values are excluded from the check (they pass)
    so that only the "real" values are validated.
    """
    mask = _missing_spec_mask(results_tbl, column, spec)
    if mask is None:
        return results_tbl

    nw_tbl = nw.from_native(results_tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))
    nw_tbl = nw_tbl.with_columns(pb_is_good_=(nw.col("pb_is_good_") | mask))
    return nw_tbl.to_native()


def _missing_spec_mask(tbl: IntoFrame, column: str, spec: Any) -> nw.Expr | None:
    """
    Get a null-free expression that's `True` where `column` holds a code of `spec` (a
    `MissingSpec`), or a null (or NaN) when `spec.null_is_missing` is `True`.

    Returns `None` when the spec declares nothing that could match.
    """
    sentinels = spec.sentinel_values()

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))

    # Note `is_in()` yields null for null inputs, and OR-ing a null into `pb_is_good_` would corrupt
    # a failing row (False | null = null under Kleene logic), so the sentinel mask is explicitly
    # filled with `False` for null rows.
    mask = None
    if sentinels:
        mask = _is_in(column, sentinels, nw_tbl.collect_schema()[column]).fill_null(False)
    if spec.null_is_missing:
        null_expr = _safe_is_nan_or_null_expr(tbl, nw.col(column), column)
        mask = null_expr if mask is None else (mask | null_expr)

    return mask


def interrogate_missing_only_coded(
    tbl: IntoFrame,
    column: str,
    sentinels: list,
    count_null: bool,
    allowed: list | None,
    min_val: Any,
    max_val: Any,
) -> Any:
    """Missing-only-coded interrogation.

    A row passes when its value is either a declared sentinel (a documented missing code), a null
    (when `count_null=True`), or a legitimate "real" value — one in `allowed` or within the
    `[min_val, max_val]` range. Any other value is treated as an *undocumented* code and fails.
    """
    nw_tbl = nw.from_native(tbl)
    dtype = nw_tbl.collect_schema()[column]

    good = None

    def _or(expr):
        nonlocal good
        good = expr if good is None else (good | expr)

    if sentinels:
        _or(_is_in(column, sentinels, dtype).fill_null(False))
    if count_null:
        _or(_safe_is_nan_or_null_expr(tbl, nw.col(column), column))
    if allowed:
        _or(_is_in(column, allowed, dtype).fill_null(False))
    if min_val is not None or max_val is not None:
        range_expr = nw.lit(True)
        if min_val is not None:
            range_expr = range_expr & (nw.col(column) >= min_val)
        if max_val is not None:
            range_expr = range_expr & (nw.col(column) <= max_val)
        _or(range_expr.fill_null(False))

    if good is None:
        good = nw.lit(False)

    result_tbl = nw_tbl.with_columns(pb_is_good_=good)
    return result_tbl.to_native()


def interrogate_missing_consistent(
    tbl: IntoFrame, columns: list[str], sentinels: list, count_null: bool
) -> Any:
    """Cross-column missing-consistency interrogation.

    Given a set of related `columns`, a row passes when the "missing for a given reason" status is
    consistent across all of them: either *none* of the columns carry the reason, or *all* of them
    do. A row fails when some-but-not-all of the columns are missing for that reason. Missingness
    for the reason is encoded by the `sentinels` values (and, when `count_null=True`, actual nulls).
    """
    nw_tbl = nw.from_native(tbl)
    schema = nw_tbl.collect_schema()
    n_cols = len(columns)

    count_expr = None
    for c in columns:
        if sentinels:
            col_expr = _is_in(c, sentinels, schema[c]).fill_null(False)
        else:
            col_expr = nw.lit(False)  # noqa
        if count_null:
            col_expr = col_expr | _safe_is_nan_or_null_expr(tbl, nw.col(c), c)
        col_count = col_expr.cast(nw.Int32)
        count_expr = col_count if count_expr is None else (count_expr + col_count)

    result_tbl = nw_tbl.with_columns(_n_reason_=count_expr)
    result_tbl = result_tbl.with_columns(
        pb_is_good_=((nw.col("_n_reason_") == 0) | (nw.col("_n_reason_") == n_cols))
    )
    result_tbl = result_tbl.drop("_n_reason_")
    return result_tbl.to_native()


def interrogate_missing_coded(tbl: IntoFrame, column: str) -> Any:
    """Missing-coded interrogation.

    A row passes when its value is *not* a raw null (or NaN). Under the structured-missingness
    model, every absence should be expressed with an explicit sentinel code (which is non-null), so
    a raw null represents *uncoded* missingness and fails the test unit.
    """
    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))
    result_tbl = nw_tbl.with_columns(
        pb_is_good_=~_safe_is_nan_or_null_expr(tbl, nw.col(column), column)
    )
    return result_tbl.to_native()


def _unused_column_name(name: str, existing: list[str]) -> str:
    """Get an internal column name that doesn't collide with any of the table's columns."""
    while name in existing:
        name = f"_{name}"
    return name


def _with_previous_value(
    nw_tbl: nw.DataFrame[Any] | nw.LazyFrame[Any], column: str, order_by: list[str] | None
) -> tuple[nw.DataFrame[Any] | nw.LazyFrame[Any], str, list[str]]:
    """
    Add a column with the last non-null value of `column` in an earlier row.

    The previous value is null up to and including the first non-null value. Rows are taken in
    `order_by=` order if given; otherwise DataFrames and Polars LazyFrames use their row order, and
    other tables (e.g., database tables), which have no row order, raise a `StepEvaluationError`.

    Returns the table, the name of the previous-value column, and the internal columns to drop.
    """
    existing = nw_tbl.collect_schema().names()
    temp_columns = []

    if order_by:
        not_found = [c for c in order_by if c not in existing]
        if not_found:
            raise StepEvaluationError(
                f"The order_by= column(s) {', '.join(repr(c) for c in not_found)} aren't in the "
                "table."
            )
        order = order_by
    elif isinstance(nw_tbl, nw.DataFrame) or is_polars_lazyframe(nw_tbl.to_native()):
        row_index = _unused_column_name("pb_row_index_", existing)
        nw_tbl = _with_row_index(nw_tbl, name=row_index)
        order = [row_index]
        temp_columns.append(row_index)
    else:
        raise StepEvaluationError(
            "This table has no inherent row order (e.g., it's a database table), so use order_by= "
            'to give the column(s) that put the rows in order (e.g., order_by="date").'
        )

    group, filled, previous = (
        _unused_column_name(name, existing) for name in ("pb_group_", "pb_filled_", "pb_previous_")
    )
    temp_columns += [group, filled, previous]

    # Forward-fill the column without `fill_null(strategy=)` (not supported by every backend): a
    # running count of non-null values puts each non-null value in its own group, together with
    # the nulls that follow it, so the group's maximum is that value
    nw_tbl = nw_tbl.with_columns(
        **{group: (~nw.col(column).is_null()).cast(nw.Int64).cum_sum().over(order_by=order)}
    )
    nw_tbl = nw_tbl.with_columns(**{filled: nw.col(column).max().over(group)})
    nw_tbl = nw_tbl.with_columns(**{previous: nw.col(filled).shift(1).over(order_by=order)})

    return nw_tbl, previous, temp_columns


def _interrogate_monotonic(
    tbl: IntoFrame,
    column: str,
    increasing: bool,
    allow_stationary: bool,
    tol: float,
    na_pass: bool,
    order_by: list[str] | None,
    missing: Any = None,
) -> Any:
    """
    Check that column values increase (or decrease) from one row to the next.

    Each non-missing value is compared with the last non-missing value before it, so missing
    values don't break the sequence. Missing values pass only with `na_pass=True`, and the first
    non-missing value passes. A nonzero `tol=` allows movement in the wrong direction up to that
    size (and stationary values); it needs a numeric column.

    A NaN counts as missing like a null does (on Polars, PyArrow and most Ibis backends a NaN is
    not a null, and it compares greater than every number, so it'd otherwise pass or fail on its
    own and break the sequence for the values after it). So do the codes of a `missing=`
    `MissingSpec` (the caller then marks those rows as passing with `apply_missing_exclusion()`).
    """
    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))
    dtype = nw_tbl.collect_schema()[column]

    if tol != 0 and dtype in (nw.Date, nw.Datetime, nw.Duration):
        tol_arg = "decreasing_tol" if increasing else "increasing_tol"
        raise StepEvaluationError(
            f"{tol_arg}= needs a numeric column, and column '{column}' has dtype {dtype}."
        )

    # Work from a copy of the column with the missing values (NaN and any `MissingSpec` codes)
    # turned into nulls, so they're all handled the same way (in the forward fill and in the
    # `na_pass=` check)
    values = _unused_column_name("pb_values_", nw_tbl.collect_schema().names())
    is_missing = _safe_is_nan_or_null_expr(tbl, nw.col(column), column)
    codes = _missing_spec_mask(tbl, column, missing) if missing is not None else None
    if codes is not None:
        is_missing = is_missing | codes
    value_expr = nw.col(column)
    if tol != 0 and dtype in (nw.Decimal, nw.Object):
        # The tolerance is a float, and subtracting Decimals can overflow their precision (PyArrow)
        # or fail on Decimal objects (pandas)
        value_expr = value_expr.cast(nw.Float64)
    elif dtype == nw.Duration and nw_tbl.implementation.is_pyarrow():
        # PyArrow can't take the maximum of a duration in the forward fill, but the durations of a
        # column share a unit, so their integer counts are in the same order
        value_expr = value_expr.cast(nw.Int64)

    try:
        nw_tbl = nw_tbl.with_columns(**{values: nw.when(~is_missing).then(value_expr)})
        result_tbl, previous, temp_columns = _with_previous_value(nw_tbl, values, order_by)
        temp_columns.append(values)

        # Without a tolerance the values are compared rather than subtracted, so dates and
        # datetimes work on every backend
        current, before = nw.col(values), nw.col(previous)
        if tol != 0:
            # The movement in the expected direction
            movement = current - before if increasing else before - current
            good_condition = movement >= -abs(tol)
        elif allow_stationary:
            good_condition = current >= before if increasing else current <= before
        else:
            good_condition = current > before if increasing else current < before

        result_tbl = result_tbl.with_columns(
            pb_is_good_=nw.when(current.is_null())
            .then(nw.lit(na_pass))
            # The first non-missing value has nothing to be compared with
            .otherwise(nw.when(before.is_null()).then(nw.lit(True)).otherwise(good_condition))
        )
    except (TypeError, ValueError) as e:
        # A pandas `object` column can hold values that pandas can't order (e.g., mixed types, or
        # dates alongside nulls) or convert to floats for the tolerance
        if dtype != nw.Object:
            raise
        raise StepEvaluationError(
            f"The values of column '{column}' (a pandas object column) can't be put in order "
            f"({e}). Convert the column to a numeric, date or datetime dtype."
        ) from e

    return result_tbl.drop(*temp_columns).to_native()


def interrogate_increasing(
    tbl: IntoFrame,
    column: str,
    allow_stationary: bool,
    decreasing_tol: float,
    na_pass: bool,
    order_by: list[str] | None = None,
    missing: Any = None,
) -> Any:
    """Increasing interrogation (see `_interrogate_monotonic()`)."""
    return _interrogate_monotonic(
        tbl, column, True, allow_stationary, decreasing_tol, na_pass, order_by, missing
    )


def interrogate_decreasing(
    tbl: IntoFrame,
    column: str,
    allow_stationary: bool,
    increasing_tol: float,
    na_pass: bool,
    order_by: list[str] | None = None,
    missing: Any = None,
) -> Any:
    """Decreasing interrogation (see `_interrogate_monotonic()`)."""
    return _interrogate_monotonic(
        tbl, column, False, allow_stationary, increasing_tol, na_pass, order_by, missing
    )


def _interrogate_comparison_base(
    tbl: IntoFrame, column: str, compare: Any, na_pass: bool, operator: str
) -> Any:
    """
    Unified base function for comparison operations (gt, ge, lt, le, eq, ne).

    Parameters
    ----------
    tbl
        The table to interrogate.
    column
        The column to check.
    compare
        The value to compare against.
    na_pass
        Whether to pass null values.
    operator
        The comparison operator: 'gt', 'ge', 'lt', 'le', 'eq', 'ne'.

    Returns
    -------
    Any
        The result table with `pb_is_good_` column indicating the passing test units.
    """

    compare_expr = _get_compare_expr_nw(compare=compare)

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))
    _check_comparable(nw_tbl, column, compare)
    compare_expr = _safe_modify_datetime_compare_val(nw_tbl, column, compare_expr)
    compare_expr = _align_decimal_compare_val(nw_tbl, column, compare_expr)

    compared_columns = [column, compare.name] if isinstance(compare, Column) else [column]
    eval_tbl = _with_pandas_object_columns_converted(nw_tbl, compared_columns)

    # Create the comparison expression based on the operator
    column_expr = nw.col(column)
    if operator == "gt":
        comparison = column_expr > compare_expr
    elif operator == "ge":
        comparison = column_expr >= compare_expr
    elif operator == "lt":
        comparison = column_expr < compare_expr
    elif operator == "le":
        comparison = column_expr <= compare_expr
    elif operator == "eq":
        comparison = column_expr == compare_expr
    elif operator == "ne":
        comparison = column_expr != compare_expr
    else:
        raise ValueError(  # pragma: no cover
            f"Invalid operator: {operator}. Must be one of: 'gt', 'ge', 'lt', 'le', 'eq', 'ne'"
        )

    column_missing = _safe_is_nan_or_null_expr(eval_tbl, nw.col(column), column)
    compare_missing = (
        _safe_is_nan_or_null_expr(eval_tbl, nw.col(compare.name), compare.name)
        if isinstance(compare, Column)
        else nw.lit(False)
    )

    # A missing value on either side passes only with `na_pass=True`; the comparison itself is
    # only used where both values are present (with NumPy-backed pandas, `x != NaN` is True)
    result_tbl = eval_tbl.with_columns(
        pb_is_good_1=column_missing & na_pass,
        pb_is_good_2=compare_missing & na_pass,
        pb_is_good_3=comparison & ~column_missing & ~compare_missing,
    )

    result_tbl = result_tbl.with_columns(
        pb_is_good_3=(
            nw.when(nw.col("pb_is_good_3").is_null())
            .then(nw.lit(False))
            .otherwise(nw.col("pb_is_good_3"))
        )
    )

    result_tbl = result_tbl.with_columns(
        pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3")
    ).drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")

    if eval_tbl is not nw_tbl:
        # Keep the original (unconverted) columns in the results table
        return nw_tbl.to_native().assign(pb_is_good_=result_tbl["pb_is_good_"].to_native())

    return result_tbl.to_native()


def _with_pandas_object_columns_converted(
    nw_tbl: nw.DataFrame[Any] | nw.LazyFrame[Any], columns: list[str]
) -> nw.DataFrame[Any] | nw.LazyFrame[Any]:
    """
    Convert pandas `object` columns to their nullable dtypes (e.g., `Int64`) for comparing.

    Comparing a NumPy-backed column with an `object` column holding `pd.NA` raises ("boolean
    value of NA is ambiguous"), whereas with nullable dtypes the comparison gives `<NA>`, which is
    then handled as a missing value.
    """
    native = nw_tbl.to_native()
    if not is_pandas_dataframe(native):
        return nw_tbl

    schema = nw_tbl.collect_schema()
    object_columns = [c for c in columns if c in schema and schema[c] == nw.Object]
    if not object_columns:
        return nw_tbl

    return nw.from_native(native.assign(**{c: native[c].convert_dtypes() for c in object_columns}))


def interrogate_rows_distinct(data_tbl: IntoFrame, columns_subset: list[str] | None) -> Any:
    """
    Check if rows in a DataFrame are distinct.

    Parameters
    ----------
    data_tbl
        A data table.
    columns_subset
        A list of columns to check for distinctness.
    threshold
        The maximum number of failing test units to allow.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    Any
        A DataFrame with a `pb_is_good_` column indicating which rows pass the test.
    """
    tbl = nw.from_native(data_tbl)
    assert is_narwhals_dataframe(tbl) or is_narwhals_lazyframe(tbl)

    # Get the column subset to use for the test
    if columns_subset is None:
        columns_subset = tbl.collect_schema().names() if is_narwhals_lazyframe(tbl) else tbl.columns

    # Check for duplicate rows using null-safe approaches
    # The previous group_by + join approach failed because NULL != NULL in join semantics,
    # causing rows with nulls to be excluded from both PASS and FAIL counts.
    if is_narwhals_dataframe(tbl):
        # For DataFrames, is_duplicated() correctly treats nulls as equal
        result = tbl.with_columns(pb_is_good_=~tbl.select(columns_subset).is_duplicated())
        return result.to_native()
    elif is_narwhals_lazyframe(tbl):
        # For LazyFrames, use a window function which treats nulls as equal
        result = tbl.with_columns(pb_is_good_=nw.len().over(columns_subset) == 1)
        return result.to_native()
    else:  # pragma: no cover
        msg = f"Expected DataFrame or LazyFrame, got {type(tbl)}"
        raise TypeError(msg)


def interrogate_rows_complete(tbl: IntoFrame, columns_subset: list[str] | None) -> Any:
    """Rows complete interrogation."""
    nw_tbl = nw.from_native(tbl)

    # Determine the number of null values in each row (column subsets are handled in
    # the `_check_nulls_across_columns_nw()` function)
    result_tbl = _check_nulls_across_columns_nw(table=nw_tbl, columns_subset=columns_subset)

    # Failing rows will have the value `True` in the generated column, so we need to negate
    # the result to get the passing rows
    result_tbl = result_tbl.with_columns(pb_is_good_=~nw.col("_any_is_null_"))
    result_tbl = result_tbl.drop("_any_is_null_")

    return result_tbl.to_native()


# The column of the number of judges that passed each row of a `prompt()` step with several models
VOTES_COLUMN = "pb_votes_"

# How the verdicts of several judges are combined into the result of a row
PROMPT_AGGREGATES = ("majority", "all", "any")


def _aggregate_votes(votes: int, n_judges: int, aggregate: str) -> bool:
    """Whether a row passes, given the number of judges that passed it."""
    if aggregate == "all":
        return votes == n_judges
    if aggregate == "any":
        return votes > 0
    # A majority (more than half; a tie fails)
    return 2 * votes > n_judges


def _prompt_verdicts(  # pragma: no cover
    tbl: Any,
    columns_subset: list[str] | None,
    ai_config: dict[str, Any],
    provider: str,
    model: str,
    total_rows: int,
) -> list[bool]:
    """The verdict of one model on each row of the table (a row it didn't judge fails)."""
    import logging

    from pointblank._utils_ai import (
        _AIValidationEngine,
        _BatchConfig,
        _DataBatcher,
        _LLMConfig,
        _PromptBuilder,
        _ValidationResponseParser,
    )

    logger = logging.getLogger(__name__)

    batch_size = ai_config.get("batch_size", 1000)
    max_concurrent = ai_config.get("max_concurrent", 3)
    attachments = ai_config.get("attachments", [])

    # Set up LLM configuration (api_key will be loaded from environment)
    llm_config = _LLMConfig(
        provider=provider,
        model=model,
        api_key=None,  # Will be loaded from environment variables
        verify_ssl=True,  # Default to verifying SSL certificates
    )

    # Set up batch configuration
    batch_config = _BatchConfig(size=batch_size, max_concurrent=max_concurrent)

    # Create optimized data batcher
    batcher = _DataBatcher(data=tbl, columns=columns_subset, config=batch_config)

    # Create batches with signature mapping for optimization
    batches, signature_mapping = batcher.create_batches()
    logger.info(f"Created {len(batches)} batches for AI validation")

    # Log optimization stats
    if hasattr(batcher, "get_reduction_stats"):
        stats = batcher.get_reduction_stats()
        if stats.get("reduction_percentage", 0) > 0:
            logger.info(
                f"Optimization: {stats['original_rows']} → {stats['unique_rows']} rows ({stats['reduction_percentage']:.1f}% reduction)"
            )

    # Create prompt builder
    prompt_builder = _PromptBuilder(ai_config["prompt"])

    # Create AI validation engine
    engine = _AIValidationEngine(llm_config, attachments=attachments)

    # Run AI validation synchronously (chatlas is synchronous)
    batch_results = engine.validate_batches(
        batches=batches, prompt_builder=prompt_builder, max_concurrent=max_concurrent
    )

    # Parse and combine results with signature mapping optimization
    parser = _ValidationResponseParser(total_rows=total_rows)
    combined_results = parser.combine_batch_results(batch_results, signature_mapping)

    # Debug: Log table info and combined results
    logger.debug("🏁 Final result conversion:")
    logger.debug(f"   - Table length: {total_rows}")
    logger.debug(
        f"   - Combined results keys: {sorted(combined_results.keys()) if combined_results else 'None'}"
    )

    # Default to False if row wasn't processed
    validation_results = [bool(combined_results.get(i, False)) for i in range(total_rows)]

    logger.debug(f"   - Final passed count: {sum(validation_results)}")
    logger.debug(f"   - Final failed count: {len(validation_results) - sum(validation_results)}")

    return validation_results


def _with_prompt_columns(native_tbl: Any, columns: dict[str, list]) -> Any:  # pragma: no cover
    """Add the result columns of a `prompt()` step to a Polars or pandas table."""
    if hasattr(native_tbl, "with_columns"):  # Polars
        import polars as pl

        return native_tbl.with_columns(
            **{name: pl.Series(name, values) for name, values in columns.items()}
        )

    if hasattr(native_tbl, "assign"):  # Pandas
        import pandas as pd

        # Align by position: the frame's index need not be 0..n-1
        return native_tbl.assign(
            **{
                name: pd.Series(
                    values, dtype=bool if name == "pb_is_good_" else "int64", index=native_tbl.index
                )
                for name, values in columns.items()
            }
        )

    # Generic fallback
    result_tbl = native_tbl.copy() if hasattr(native_tbl, "copy") else native_tbl
    for name, values in columns.items():
        result_tbl[name] = values
    return result_tbl


def interrogate_prompt(  # pragma: no cover
    tbl: IntoFrame,
    columns_subset: list[str] | None,
    ai_config: dict[str, Any],
    info: dict[str, Any] | None = None,
) -> Any:
    """
    AI-powered interrogation of rows.

    With several models (the `judges` of `ai_config`), each judges every row, and their verdicts
    are combined with its `aggregate` rule; the number of judges that passed each row is in a
    `pb_votes_` column. The verdicts of each model are put in `info` (if given), for the note of
    the step and for `Validate.calibrate_prompt()`.
    """
    import logging

    logger = logging.getLogger(__name__)

    # AI validation inherently needs materialized data (sends rows to LLM)
    # Collect lazy frames here since this cannot operate lazily
    if _is_lazy_frame(tbl):
        tbl = tbl.collect()

    # Convert to narwhals early for consistent row counting
    nw_tbl = nw.from_native(tbl)
    # Get row count - for LazyFrame we need to use select/collect
    if is_narwhals_lazyframe(nw_tbl):
        row_count = nw_tbl.select(nw.len()).collect().item()
        assert isinstance(row_count, int)
        total_rows = row_count
    else:
        assert is_narwhals_dataframe(nw_tbl)
        total_rows = len(nw_tbl)

    judges = ai_config.get("judges") or [f"{ai_config['llm_provider']}:{ai_config['llm_model']}"]
    aggregate = ai_config.get("aggregate", "majority")

    try:
        verdicts = {}
        for judge in judges:
            provider, model = judge.split(":", 1)
            verdicts[judge] = _prompt_verdicts(
                tbl, columns_subset, ai_config, provider, model, total_rows
            )

        columns: dict[str, list] = {}
        if len(judges) == 1:
            columns["pb_is_good_"] = verdicts[judges[0]]
        else:
            votes = [sum(row) for row in zip(*verdicts.values())]
            columns["pb_is_good_"] = [
                _aggregate_votes(vote, len(judges), aggregate) for vote in votes
            ]
            columns[VOTES_COLUMN] = votes

        if info is not None:
            info["judges"] = verdicts
            info["aggregate"] = aggregate

        # Add the result columns to the table (as a Polars or pandas table)
        result_tbl = _with_prompt_columns(nw_tbl.to_native(), columns)

        logger.info(
            f"AI validation completed. {sum(columns['pb_is_good_'])} rows passed out of "
            f"{total_rows}"
        )

        return result_tbl

    except StepEvaluationError:
        raise

    except ImportError as e:
        # A missing package (e.g., the provider's SDK); the message says how to install it
        raise StepEvaluationError(f"The prompt() step couldn't be run: {e}") from e

    except Exception as e:
        # Anything else that stops the step as a whole (e.g., an invalid model configuration)
        raise StepEvaluationError(f"The prompt() step couldn't be run ({_first_line(e)}).") from e


# The column of the scores of a `col_vals_scored()` step, next to `pb_is_good_`
SCORE_COLUMN = "pb_score_"


def _score_plain_rows(
    scorer: Any, rows: list[dict[str, Any]], columns: list[str], cache: Any
) -> tuple[list[float | None], dict[str, Any]]:
    """
    Score rows of values, leaving out (with a score of `None`) those with a missing value.

    Errors of the scorer (or of a missing package it needs) stop the step, with a note.
    """
    from pointblank import scorers

    records = []
    positions = []
    for i, row in enumerate(rows):
        record = {c: scorers.python_value(row[c]) for c in columns}
        if any(v is None for v in record.values()):
            continue
        records.append(record)
        positions.append(i)

    label = scorers.scorer_label(scorer)
    try:
        scored, counts = scorers.score_records(scorer, records, cache)
    except ImportError as e:
        raise StepEvaluationError(f"The col_vals_scored() step couldn't be run: {e}") from e
    except scorers.ScorerOutputError as e:
        raise StepEvaluationError(
            f"The scorer {label} of the col_vals_scored() step can't be used: {e}."
        ) from e
    except Exception as e:
        raise StepEvaluationError(
            f"The scorer {label} of the col_vals_scored() step failed ({_first_line(e)})."
        ) from e

    scores: list[float | None] = [None] * len(rows)
    for position, score in zip(positions, scored):
        scores[position] = score
    return scores, counts


def _scored_lookup(nw_tbl: Any, columns: list[str], rows: list[dict], scores: list) -> Any:
    """Make a table of the distinct rows that have a score, on the backend of a lazy table."""
    keep = [k for k, score in enumerate(scores) if score is not None]
    return _value_lookup(
        nw_tbl,
        columns,
        [rows[k] for k in keep],
        {SCORE_COLUMN: ([scores[k] for k in keep], "float64")},
        step="col_vals_scored",
    )


def _value_lookup(
    nw_tbl: Any,
    columns: list[str],
    rows: list[dict],
    values: dict[str, tuple[list, str]],
    step: str,
) -> Any:
    """
    Make a table of distinct rows of `columns` with values computed for them, on the backend of a
    lazy table (an Ibis table or a PySpark DataFrame), to join to it.

    `values` maps each new column to its values (one for each row) and its type (`"float64"`,
    `"int64"`, or `"string"`).
    """
    native = nw_tbl.to_native()

    if nw_tbl.implementation == nw.Implementation.IBIS:
        import ibis

        data_schema = native.schema()
        schema = {c: data_schema[c] for c in columns}
        data = {c: [row[c] for row in rows] for c in columns}
        for name, (column_values, dtype) in values.items():
            schema[name] = dtype
            data[name] = list(column_values)
        return nw.from_native(ibis.memtable(data, schema=ibis.schema(schema)))

    if nw_tbl.implementation.is_pyspark() or nw_tbl.implementation.is_pyspark_connect():
        from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

        spark_types = {"float64": DoubleType, "int64": LongType, "string": StringType}
        fields = {field.name: field for field in native.schema.fields}
        schema = StructType(
            [fields[c] for c in columns]
            + [StructField(name, spark_types[dtype](), True) for name, (_, dtype) in values.items()]
        )
        columns_values = [column_values for column_values, _ in values.values()]
        data = [
            tuple(row[c] for c in columns) + tuple(v[k] for v in columns_values)
            for k, row in enumerate(rows)
        ]
        return nw.from_native(native.sparkSession.createDataFrame(data, schema=schema))

    raise StepEvaluationError(
        f"{step}() can check Polars and pandas tables, Ibis tables, and PySpark DataFrames, not a "
        f"{nw_tbl.implementation} table."
    )


def interrogate_scored(
    tbl: IntoFrame, columns: list[str], config: dict[str, Any], na_pass: bool
) -> tuple[Any, dict[str, Any]]:
    """
    Score each row with a model and check the scores against bounds.

    The scorer gets the values of `columns` for each distinct row that has no missing value (a
    row with a missing value, or that the scorer gives no score, passes if `na_pass=True`). The
    results table has the scores in `pb_score_` (unless `keep_scores=False`) and the test results
    in `pb_is_good_`. An eager table is scored in Python, in order. A database (Ibis) table or a
    PySpark DataFrame has its distinct rows collected and scored, and the scores are joined back
    to it on its own backend.

    Returns the results table and the information on the scores for the step (`val_info`).
    """
    from pointblank import scorers

    scorer = config["scorer"]
    min_val = config.get("min_val")
    max_val = config.get("max_val")
    cache = config.get("cache", True)

    nw_tbl = nw.from_native(tbl)
    names = list(nw_tbl.collect_schema().names())
    missing = [c for c in columns if c not in names]
    if missing:
        listed = ", ".join(repr(c) for c in missing)
        raise StepEvaluationError(
            f"The column{'s' if len(missing) > 1 else ''} {listed} of the col_vals_scored() step "
            f"{'aren' if len(missing) > 1 else 'isn'}'t in the table."
        )
    if SCORE_COLUMN in names:
        raise StepEvaluationError(
            f"The table has a column {SCORE_COLUMN!r}, which is the name of the scores of a "
            "col_vals_scored() step: rename it first (e.g., with `pre=`)."
        )

    name, version = scorers.scorer_identity(scorer)
    info: dict[str, Any] = {
        "scorer": name,
        "version": version,
        "label": scorers.scorer_label(scorer),
    }

    was_polars_lazy = is_polars_lazyframe(nw_tbl.to_native())
    if was_polars_lazy:
        nw_tbl = nw_tbl.collect()

    if isinstance(nw_tbl, nw.DataFrame):
        rows = nw_tbl.select(columns).rows(named=True)
        scores, counts = _score_plain_rows(scorer, rows, columns, cache)
        native = nw_tbl.to_native()
        if is_polars_dataframe(native):
            import polars as pl

            native = native.with_columns(pl.Series(SCORE_COLUMN, scores, dtype=pl.Float64))
        elif is_pandas_dataframe(native):
            import pandas as pd

            # Align by position: the frame's index need not be 0..n-1
            native = native.assign(
                **{SCORE_COLUMN: pd.Series(scores, dtype="float64", index=native.index)}
            )
        else:  # pragma: no cover
            native = nw_tbl.with_columns(
                nw.new_series(SCORE_COLUMN, scores, nw.Float64(), backend=nw_tbl.implementation)
            ).to_native()
        scored_tbl = nw.from_native(native)

        present = [s for s in scores if s is not None]
        info.update(counts)
        info["n_rows"] = len(scores)
        info["n_unscored"] = len(scores) - len(present)
        info["min"] = min(present) if present else None
        info["max"] = max(present) if present else None
        info["median"] = statistics.median(present) if present else None
    else:
        distinct = nw_tbl.select(columns).unique().collect()
        rows = distinct.rows(named=True)
        scores, counts = _score_plain_rows(scorer, rows, columns, cache)
        lookup = _scored_lookup(nw_tbl, columns, rows, scores)
        scored_tbl = nw_tbl.join(lookup, on=columns, how="left")
        info.update(counts)
        info.update(_lazy_score_stats(scored_tbl))

    score = nw.col(SCORE_COLUMN)
    in_bounds = None
    if min_val is not None:
        in_bounds = score >= min_val
    if max_val is not None:
        below_max = score <= max_val
        in_bounds = below_max if in_bounds is None else in_bounds & below_max
    results = scored_tbl.with_columns(
        pb_is_good_=nw.when(score.is_null()).then(nw.lit(bool(na_pass))).otherwise(in_bounds)
    )
    if not config.get("keep_scores", True):
        results = results.drop(SCORE_COLUMN)
    if was_polars_lazy:
        results = results.lazy()

    return results.to_native(), info


def _lazy_score_stats(scored_tbl: Any) -> dict[str, Any]:
    """Get the number of rows, those without a score, and the range of the scores, on a backend."""
    score = nw.col(SCORE_COLUMN)
    exprs = [
        nw.len().alias("n_rows"),
        score.is_null().cast(nw.Int64).sum().alias("n_unscored"),
        score.min().alias("min"),
        score.max().alias("max"),
    ]
    try:
        stats = scored_tbl.select(*exprs, score.median().alias("median")).collect()
    except Exception:
        # A backend without a median (e.g., SQLite)
        stats = scored_tbl.select(*exprs).collect()
    row = stats.rows(named=True)[0]
    out = {k: row.get(k) for k in ("n_rows", "n_unscored", "min", "max", "median")}
    for key in ("n_rows", "n_unscored"):
        out[key] = int(out[key] or 0)
    for key in ("min", "max", "median"):
        out[key] = None if out[key] is None else float(out[key])
    return out


# ---------------------------------------------------------------------------------------------
# Text checks: `rows_near_distinct()`, `rows_not_contaminated()`, and `col_vals_chat_format()`
# ---------------------------------------------------------------------------------------------

# The columns that the text checks add to their results, next to `pb_is_good_`
CLUSTER_COLUMN = "pb_cluster_"
MATCH_COLUMN = "pb_match_"
REASON_COLUMN = "pb_reason_"


def _check_step_columns(nw_tbl: Any, columns: list[str], step: str, added: str) -> None:
    """Check that the columns of a step are in the table, and that the table hasn't `added`."""
    names = list(nw_tbl.collect_schema().names())
    missing = [c for c in columns if c not in names]
    if missing:
        listed = ", ".join(repr(c) for c in missing)
        raise StepEvaluationError(
            f"The column{'s' if len(missing) > 1 else ''} {listed} of the {step}() step "
            f"{'aren' if len(missing) > 1 else 'isn'}'t in the table."
        )
    if added in names:
        raise StepEvaluationError(
            f"The table has a column {added!r}, which is the name of a column of the results of "
            f"a {step}() step: rename it first (e.g., with `pre=`)."
        )


def _row_values(row: dict[str, Any], columns: list[str]) -> list[Any] | None:
    """The plain values of a row's columns, or `None` if any is missing."""
    from pointblank.scorers import python_value

    values = [python_value(row[c]) for c in columns]
    return None if any(v is None for v in values) else values


def _distinct_with_counts(nw_tbl: Any, columns: list[str]) -> list[tuple[dict[str, Any], int]]:
    """
    Get the distinct rows of `columns` of a lazy table that have no missing value, with the number
    of rows of each, sorted by their values (so that the order doesn't vary from run to run).
    """
    counted = (
        nw_tbl.filter(*[~nw.col(c).is_null() for c in columns])
        .group_by(columns)
        .agg(nw.len().alias("pb_n_"))
        .collect()
    )
    out = []
    for row in counted.rows(named=True):
        values = _row_values(row, columns)
        if values is not None:
            out.append(({c: row[c] for c in columns}, int(row["pb_n_"])))
    out.sort(key=lambda item: [repr(item[0][c]) for c in columns])
    return out


def _with_eager_columns(nw_tbl: Any, values: dict[str, tuple[list, str]]) -> Any:
    """Add columns of values (one for each row, by position) to an eager table."""
    native = nw_tbl.to_native()
    if is_polars_dataframe(native):
        import polars as pl

        types = {"float64": pl.Float64, "int64": pl.Int64, "string": pl.String}
        native = native.with_columns(
            [pl.Series(name, v, dtype=types[dtype]) for name, (v, dtype) in values.items()]
        )
    elif is_pandas_dataframe(native):
        import pandas as pd

        # Align by position: the frame's index need not be 0..n-1
        types = {"float64": "float64", "int64": "Int64", "string": "object"}
        native = native.assign(
            **{
                name: pd.Series(v, dtype=types[dtype], index=native.index)
                for name, (v, dtype) in values.items()
            }
        )
    else:  # pragma: no cover
        types = {"float64": nw.Float64(), "int64": nw.Int64(), "string": nw.String()}
        native = nw_tbl.with_columns(
            *[
                nw.new_series(name, v, types[dtype], backend=nw_tbl.implementation)
                for name, (v, dtype) in values.items()
            ]
        ).to_native()
    return nw.from_native(native)


def _check_lazy_backend(nw_tbl: Any, step: str) -> None:
    if not (
        nw_tbl.implementation == nw.Implementation.IBIS
        or nw_tbl.implementation.is_pyspark()
        or nw_tbl.implementation.is_pyspark_connect()
    ):
        raise StepEvaluationError(
            f"{step}() can check Polars and pandas tables, Ibis tables, and PySpark DataFrames, "
            f"not a {nw_tbl.implementation} table."
        )


def _with_row_text_values(
    tbl: IntoFrame,
    columns: list[str],
    step: str,
    added: dict[str, str],
    compute: Callable[[list[str], list[int]], dict[str, list]],
) -> tuple[Any, bool, list[int]]:
    """
    Add columns computed from the text of each row (the values of `columns`, joined by newlines).

    `compute` gets the texts of the distinct rows that have no missing value, with the number of
    rows of each (one for each row of an eager table), and gives a value of each column in `added`
    (name to type) for each. An eager table is computed in place (a Polars LazyFrame is collected,
    and is made lazy again at the end); an Ibis table or a PySpark DataFrame has its distinct rows
    collected, and the values are joined back to it on its own backend.

    Returns the table with the new columns (as Narwhals), whether it was a Polars LazyFrame, and
    the numbers of rows of the texts given to `compute`.
    """
    from pointblank._text import join_values

    nw_tbl = nw.from_native(tbl)
    for name in added:
        _check_step_columns(nw_tbl, columns, step, name)

    was_polars_lazy = is_polars_lazyframe(nw_tbl.to_native())
    if was_polars_lazy:
        nw_tbl = nw_tbl.collect()

    if isinstance(nw_tbl, nw.DataFrame):
        rows = nw_tbl.select(columns).rows(named=True)
        positions = []
        texts = []
        for k, row in enumerate(rows):
            values = _row_values(row, columns)
            if values is not None:
                positions.append(k)
                texts.append(join_values(values))
        computed = compute(texts, [1] * len(texts))
        out: dict[str, tuple[list, str]] = {}
        for name, dtype in added.items():
            column_values: list[Any] = [None] * len(rows)
            for position, value in zip(positions, computed[name]):
                column_values[position] = value
            out[name] = (column_values, dtype)
        return _with_eager_columns(nw_tbl, out), was_polars_lazy, [1] * len(texts)

    _check_lazy_backend(nw_tbl, step)
    distinct = _distinct_with_counts(nw_tbl, columns)
    texts = [join_values([row[c] for c in columns]) for row, _ in distinct]
    counts = [n for _, n in distinct]
    computed = compute(texts, counts)
    keep = [k for k in range(len(distinct)) if any(computed[name][k] is not None for name in added)]
    lookup = _value_lookup(
        nw_tbl,
        columns,
        [distinct[k][0] for k in keep],
        {name: ([computed[name][k] for k in keep], dtype) for name, dtype in added.items()},
        step=step,
    )
    return nw_tbl.join(lookup, on=columns, how="left"), was_polars_lazy, counts


def interrogate_near_distinct(
    tbl: IntoFrame, columns: list[str], config: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """
    Find the rows whose text is a near duplicate of another row's.

    The text of a row is the values of `columns`, joined by newlines. Rows in a group of near
    duplicates (see `pointblank._text.near_duplicate_groups()`) fail, and their group is in the
    `pb_cluster_` column of the results (numbered from 1); a row with a missing value passes.

    Returns the results table and the information on the groups for the step's note.
    """
    from pointblank._text import near_duplicate_groups

    info: dict[str, Any] = {}

    def compute(texts: list[str], counts: list[int]) -> dict[str, list]:
        groups, n_compared = near_duplicate_groups(
            texts,
            method=config["method"],
            threshold=config["threshold"],
            ngram=config["ngram"],
            num_perm=config["num_perm"],
        )
        sizes: dict[int, int] = {}
        for group, n in zip(groups, counts):
            sizes[group] = sizes.get(group, 0) + n
        # Number the groups of more than one row from 1, in the order of their first text
        numbers: dict[int, int] = {}
        clusters: list[Any] = []
        for group in groups:
            if sizes[group] < 2:
                clusters.append(None)
                continue
            clusters.append(numbers.setdefault(group, len(numbers) + 1))
        dup_sizes = [size for group, size in sizes.items() if size >= 2]
        info.update(
            {
                "n_clusters": len(dup_sizes),
                "n_in_clusters": sum(dup_sizes),
                "largest": max(dup_sizes) if dup_sizes else 0,
                "n_compared": n_compared,
            }
        )
        return {CLUSTER_COLUMN: clusters}

    nw_tbl, was_polars_lazy, _ = _with_row_text_values(
        tbl, columns, "rows_near_distinct", {CLUSTER_COLUMN: "int64"}, compute
    )
    results = nw_tbl.with_columns(pb_is_good_=nw.col(CLUSTER_COLUMN).is_null())
    if was_polars_lazy:
        results = results.lazy()
    return results.to_native(), info


def reference_texts(
    reference: Any, reference_columns: list[str] | None, columns: list[str]
) -> tuple[list[str], list[str]]:
    """
    Get the texts of the reference of `~~pointblank.Validate.rows_not_contaminated()` and the name
    of the reference that each comes from.

    The reference is a table (or the path of a file, read like the data of a validation), a list
    of them (named `1`, `2`, ...), a dictionary of them by name, or a list of texts. The text of a
    row of a table is the values of `reference_columns` (or of `columns`), joined by newlines.
    """
    from pointblank._text import join_values
    from pointblank.validate import _process_data

    if isinstance(reference, dict):
        items = [(str(name), value) for name, value in reference.items()]
    elif isinstance(reference, (list, tuple)) and all(isinstance(v, str) for v in reference):
        return [v for v in reference], ["texts"] * len(reference)
    elif isinstance(reference, (list, tuple)):
        items = [(str(k), value) for k, value in enumerate(reference, start=1)]
    else:
        items = [("reference", reference)]

    ref_columns = reference_columns or columns
    texts: list[str] = []
    names: list[str] = []
    for name, value in items:
        if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
            texts.extend(value)
            names.extend([name] * len(value))
            continue
        try:
            table = nw.from_native(_process_data(value))
        except Exception as e:
            raise StepEvaluationError(
                f"The reference {name!r} of the rows_not_contaminated() step couldn't be read "
                f"({_first_line(e)})."
            ) from e
        schema_names = list(table.collect_schema().names())
        missing = [c for c in ref_columns if c not in schema_names]
        if missing:
            raise StepEvaluationError(
                f"The reference {name!r} of the rows_not_contaminated() step has no column "
                f"{', '.join(repr(c) for c in missing)} (set `reference_columns=`)."
            )
        selected = table.select(ref_columns)
        if isinstance(selected, nw.LazyFrame):
            selected = selected.collect()
        for row in selected.rows(named=True):
            values = _row_values(row, ref_columns)
            if values is not None:
                texts.append(join_values(values))
                names.append(name)
    return texts, names


def interrogate_not_contaminated(
    tbl: IntoFrame, columns: list[str], config: dict[str, Any]
) -> tuple[Any, dict[str, Any]]:
    """
    Find the rows whose text overlaps with a reference (e.g., the questions of a benchmark).

    A row fails if its text (the values of `columns`, joined by newlines) overlaps with a text of
    the reference (see `pointblank._text.ReferenceIndex`), and the first reference text that it
    overlaps with is in the `pb_match_` column of the results. A row with a missing value passes.

    Returns the results table and the information on the overlap for the step's note.
    """
    from pointblank._text import ReferenceIndex

    ref_texts, ref_names = reference_texts(
        config["reference"], config.get("reference_columns"), columns
    )
    index = ReferenceIndex(
        ref_texts,
        method=config["method"],
        ngram=config["ngram"],
        threshold=config["threshold"],
        num_perm=config["num_perm"],
    )
    info: dict[str, Any] = {"n_reference": len(ref_texts)}

    def compute(texts: list[str], counts: list[int]) -> dict[str, list]:
        matches: list[Any] = []
        by_name: dict[str, int] = {}
        for text, n in zip(texts, counts):
            k = index.match(text)
            if k is None:
                matches.append(None)
                continue
            matches.append(ref_texts[k])
            by_name[ref_names[k]] = by_name.get(ref_names[k], 0) + n
        info["n_contaminated"] = sum(by_name.values())
        info["by_reference"] = by_name
        return {MATCH_COLUMN: matches}

    nw_tbl, was_polars_lazy, _ = _with_row_text_values(
        tbl, columns, "rows_not_contaminated", {MATCH_COLUMN: "string"}, compute
    )
    results = nw_tbl.with_columns(pb_is_good_=nw.col(MATCH_COLUMN).is_null())
    if was_polars_lazy:
        results = results.lazy()
    return results.to_native(), info


def interrogate_chat_format(
    tbl: IntoFrame, column: str, config: dict[str, Any], na_pass: bool
) -> tuple[Any, dict[str, Any]]:
    """
    Check that each row of a column is a well-formed chat conversation.

    The value of a row is a list of messages (e.g., a Polars list of structs) or the JSON text of
    one. The code of the first problem found with a conversation is in the `pb_reason_` column of
    the results (see `pointblank._text.CHAT_REASONS`), and a row with a problem fails. A missing
    value passes if `na_pass=True`. On an Ibis table or a PySpark DataFrame, the conversations
    must be stored as JSON text.

    Returns the results table and the counts of each problem for the step's note.
    """
    from pointblank._text import chat_reason
    from pointblank.scorers import python_value

    def check(value: Any) -> str | None:
        return chat_reason(
            value,
            roles=config["roles"],
            system_role=config["system_role"],
            alternate=config["alternate"],
            first_role=config["first_role"],
            last_role=config["last_role"],
            allow_empty=config["allow_empty"],
            role_key=config["role_key"],
            content_key=config["content_key"],
        )

    nw_tbl = nw.from_native(tbl)
    _check_step_columns(nw_tbl, [column], "col_vals_chat_format", REASON_COLUMN)

    was_polars_lazy = is_polars_lazyframe(nw_tbl.to_native())
    if was_polars_lazy:
        nw_tbl = nw_tbl.collect()

    reasons_count: dict[str, int] = {}
    if isinstance(nw_tbl, nw.DataFrame):
        values = nw_tbl.get_column(column).to_list()
        reasons: list[Any] = []
        n_missing = 0
        for value in values:
            value = python_value(value) if not isinstance(value, (list, tuple)) else value
            if value is None:
                reasons.append(None)
                n_missing += 1
                continue
            reason = check(value)
            reasons.append(reason)
            if reason is not None:
                reasons_count[reason] = reasons_count.get(reason, 0) + 1
        nw_tbl = _with_eager_columns(nw_tbl, {REASON_COLUMN: (reasons, "string")})
    else:
        _check_lazy_backend(nw_tbl, "col_vals_chat_format")
        if nw_tbl.collect_schema()[column] != nw.String:
            raise StepEvaluationError(
                "col_vals_chat_format() checks the conversations of a database table or a PySpark "
                f"DataFrame as JSON text, and the column {column!r} is of type "
                f"{nw_tbl.collect_schema()[column]} (store the messages as JSON, or check an "
                "eager table)."
            )
        distinct = _distinct_with_counts(nw_tbl, [column])
        reasons = []
        for row, n in distinct:
            reason = check(row[column])
            reasons.append(reason)
            if reason is not None:
                reasons_count[reason] = reasons_count.get(reason, 0) + n
        keep = [k for k, reason in enumerate(reasons) if reason is not None]
        lookup = _value_lookup(
            nw_tbl,
            [column],
            [distinct[k][0] for k in keep],
            {REASON_COLUMN: ([reasons[k] for k in keep], "string")},
            step="col_vals_chat_format",
        )
        nw_tbl = nw_tbl.join(lookup, on=[column], how="left")
        n_missing = int(
            nw_tbl.select(nw.col(column).is_null().cast(nw.Int64).sum().alias("n"))
            .collect()
            .item(0, "n")
            or 0
        )

    results = nw_tbl.with_columns(
        pb_is_good_=nw.when(nw.col(column).is_null())
        .then(nw.lit(bool(na_pass)))
        .otherwise(nw.col(REASON_COLUMN).is_null())
    )
    if was_polars_lazy:
        results = results.lazy()
    return results.to_native(), {"reasons": reasons_count, "n_missing": n_missing}


def data_freshness(
    data_tbl: IntoFrame,
    column: str,
    max_age: Any,  # datetime.timedelta
    reference_time: Any | None,  # datetime.datetime | None
    timezone: str | None,
    allow_tz_mismatch: bool,
    allow_future: Any = False,  # bool | datetime.timedelta
) -> dict:
    """
    Check if the most recent datetime value in a column is within the allowed max_age.

    Parameters
    ----------
    data_tbl
        The data table to check.
    column
        The datetime column to check.
    max_age
        The maximum allowed age as a timedelta.
    reference_time
        The reference time to compare against (None = use current time).
    timezone
        The timezone to use for interpretation.
    allow_tz_mismatch
        Whether to suppress timezone mismatch warnings.
    allow_future
        How far the most recent datetime may be later than the reference time: `False` (none),
        `True` (any amount), or a timedelta.

    Returns
    -------
    dict
        A dictionary containing:
        - 'passed': bool, whether the validation passed
        - 'max_datetime': the maximum datetime found in the column
        - 'reference_time': the reference time used
        - 'age': the calculated age (timedelta)
        - 'max_age': the maximum allowed age
        - 'tz_warning': any timezone warning message
        - 'future': whether the most recent datetime is later than the reference time
    """
    import datetime

    nw_frame = nw.from_native(data_tbl)

    result = {
        "passed": False,
        "max_datetime": None,
        "reference_time": None,
        "age": None,
        "max_age": max_age,
        "tz_warning": None,
        "column_empty": False,
        "future": False,
    }

    # Get the maximum datetime value from the column using lazy-compatible aggregation
    try:
        max_val_result = nw_frame.select(nw.col(column).max())
        if is_narwhals_lazyframe(max_val_result):  # pragma: no cover
            max_val_result = max_val_result.collect()
        max_datetime_raw = max_val_result.item()

        if max_datetime_raw is None:
            result["column_empty"] = True
            result["passed"] = False
            return result

        # Convert to Python datetime if needed
        if hasattr(max_datetime_raw, "to_pydatetime"):
            # Pandas Timestamp
            max_datetime = max_datetime_raw.to_pydatetime()
        elif hasattr(max_datetime_raw, "isoformat"):
            # Already a datetime-like object
            max_datetime = max_datetime_raw
        else:  # pragma: no cover
            # Try to parse as string or handle other types
            max_datetime = datetime.datetime.fromisoformat(str(max_datetime_raw))

        result["max_datetime"] = max_datetime

    except Exception as e:  # pragma: no cover
        result["error"] = str(e)
        result["passed"] = False
        return result

    # Determine the reference time
    # We'll set the reference time after we know the timezone awareness of the data
    if reference_time is None:
        ref_time = None  # Will be set below based on data timezone awareness
    else:
        ref_time = reference_time

    # Handle timezone awareness/naivete
    max_dt_aware = _is_datetime_aware(max_datetime)

    # Helper to parse timezone string (supports IANA names and offsets like "-7", "-07:00")
    def _get_tz_from_string(tz_str: str) -> datetime.tzinfo:
        import re

        # Check for offset formats: "-7", "+5", "-07:00", "+05:30", etc.
        offset_pattern = r"^([+-]?)(\d{1,2})(?::(\d{2}))?$"
        match = re.match(offset_pattern, tz_str.strip())

        if match:
            sign_str, hours_str, minutes_str = match.groups()
            hours = int(hours_str)
            minutes = int(minutes_str) if minutes_str else 0

            total_minutes = hours * 60 + minutes
            if sign_str == "-":
                total_minutes = -total_minutes

            return datetime.timezone(datetime.timedelta(minutes=total_minutes))

        # Try IANA timezone names (zoneinfo is standard in Python 3.9+)
        try:
            return ZoneInfo(tz_str)
        except KeyError:  # pragma: no cover
            # Invalid timezone name, fall back to UTC
            return datetime.timezone.utc

    # If ref_time is None (no reference_time provided), set it based on data awareness
    if ref_time is None:
        if max_dt_aware:
            # Data is timezone-aware, use timezone-aware now
            if timezone:
                ref_time = datetime.datetime.now(_get_tz_from_string(timezone))
            else:
                # Default to UTC when data is aware but no timezone specified
                ref_time = datetime.datetime.now(datetime.timezone.utc)
        else:
            # Data is naive, use naive local time for comparison
            if timezone:
                # If user specified timezone, use it for reference
                ref_time = datetime.datetime.now(_get_tz_from_string(timezone))
            else:
                # No timezone specified and data is naive -> use naive local time
                ref_time = datetime.datetime.now()

    result["reference_time"] = ref_time
    ref_dt_aware = _is_datetime_aware(ref_time)

    # Track timezone warnings - use keys for translation lookup
    tz_warning_key = None

    if max_dt_aware != ref_dt_aware:
        if not allow_tz_mismatch:
            if max_dt_aware and not ref_dt_aware:
                tz_warning_key = "data_freshness_tz_warning_aware_naive"
            else:
                tz_warning_key = "data_freshness_tz_warning_naive_aware"
        result["tz_warning_key"] = tz_warning_key

    # Make both comparable
    try:
        if max_dt_aware and not ref_dt_aware:
            # Add timezone to reference time
            if timezone:
                try:
                    ref_time = ref_time.replace(tzinfo=ZoneInfo(timezone))
                except KeyError:  # pragma: no cover
                    ref_time = ref_time.replace(tzinfo=datetime.timezone.utc)
            else:
                # Assume UTC
                ref_time = ref_time.replace(tzinfo=datetime.timezone.utc)

        elif not max_dt_aware and ref_dt_aware:
            # Localize the max_datetime if we have a timezone
            if timezone:
                try:
                    max_datetime = max_datetime.replace(tzinfo=ZoneInfo(timezone))
                except KeyError:  # pragma: no cover
                    # Remove timezone from reference for comparison
                    ref_time = ref_time.replace(tzinfo=None)
            else:
                # Remove timezone from reference for comparison
                ref_time = ref_time.replace(tzinfo=None)

        # Calculate the age
        age = ref_time - max_datetime
        result["age"] = age
        result["reference_time"] = ref_time

        # Check if within max_age; a negative age means the data is later than the reference time,
        # which fails unless `allow_future=` permits it
        result["future"] = age < datetime.timedelta(0)
        if result["future"] and allow_future is not True:
            future_tol = datetime.timedelta(0) if allow_future is False else allow_future
            result["passed"] = -age <= future_tol
        else:
            result["passed"] = age <= max_age

    except Exception as e:  # pragma: no cover
        result["error"] = str(e)
        result["passed"] = False

    return result


def _is_datetime_aware(dt: Any) -> bool:
    """Check if a datetime object is timezone-aware."""
    if dt is None:
        return False
    if hasattr(dt, "tzinfo"):
        return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None
    return False
