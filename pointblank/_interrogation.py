from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.dependencies import (
    is_narwhals_dataframe,
    is_narwhals_lazyframe,
    is_pandas_dataframe,
    is_polars_dataframe,
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
)
from pointblank.column import Column

if TYPE_CHECKING:
    from narwhals.typing import IntoFrame


def _safe_modify_datetime_compare_val(data_frame: Any, column: str, compare_val: Any) -> Any:
    """
    Safely modify datetime comparison values for LazyFrame compatibility.

    This function handles the case where we can't directly slice LazyFrames
    to get column dtypes for datetime conversion.
    """
    try:
        # First try to get column dtype from schema for LazyFrames
        column_dtype = None

        if hasattr(data_frame, "collect_schema"):
            schema = data_frame.collect_schema()
            column_dtype = schema.get(column)
        elif hasattr(data_frame, "schema"):
            schema = data_frame.schema
            column_dtype = schema.get(column)

        # If we got a dtype from schema, use it
        if column_dtype is not None:
            # Create a mock column object for _modify_datetime_compare_val
            class MockColumn:
                def __init__(self, dtype):
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
                        def __init__(self, dtype):
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
                    def __init__(self, dtype):
                        self.dtype = dtype

                mock_column = MockColumn(column_dtype)
                return _modify_datetime_compare_val(tgt_column=mock_column, compare_val=compare_val)
        except Exception:
            pass

    except Exception:
        pass

    # If all else fails, return the original compare_val
    return compare_val


def _safe_is_nan_or_null_expr(
    data_frame: Any, column_expr: Any, column_name: str | None = None
) -> Any:
    """
    Create an expression that safely checks for both Null and NaN values.

    This function handles the case where `is_nan()` is not supported for certain data types (like
    strings) or backends (like `SQLite` via Ibis) by checking the backend type and column type
    first.

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
    # Always check for null values
    null_check = column_expr.is_null()

    # For Ibis backends, many don't support `is_nan()` so we stick to Null checks only;
    # use `narwhals.get_native_namespace()` for reliable backend detection
    try:
        native_namespace = nw.get_native_namespace(data_frame)

        # If it's an Ibis backend, only check for null values
        # The namespace is the actual module, so we check its name
        if hasattr(native_namespace, "__name__") and "ibis" in native_namespace.__name__:
            return null_check
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    # For non-Ibis backends, try to use `is_nan()` if the column type supports it
    try:
        if hasattr(data_frame, "collect_schema"):
            schema = data_frame.collect_schema()
        elif hasattr(data_frame, "schema"):
            schema = data_frame.schema
        else:  # pragma: no cover
            schema = None  # pragma: no cover

        if schema and column_name:
            column_dtype = schema.get(column_name)
            if column_dtype:
                dtype_str = str(column_dtype).lower()

                # Check if it's a numeric type that supports NaN
                is_numeric = any(
                    num_type in dtype_str for num_type in ["float", "double", "f32", "f64"]
                )

                if is_numeric:
                    try:
                        # For numeric types, try to check both Null and NaN
                        return null_check | column_expr.is_nan()
                    except Exception:
                        # If `is_nan()` fails for any reason, fall back to Null only
                        pass
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

    # Fallback: just check Null values
    return null_check


class ConjointlyValidation:
    def __init__(self, data_tbl, expressions, threshold, tbl_type):
        self.data_tbl = data_tbl
        self.expressions = expressions
        self.threshold = threshold

        # Detect the table type
        if tbl_type in (None, "local"):
            # Detect the table type using _get_tbl_type()
            self.tbl_type = _get_tbl_type(data=data_tbl)
        else:
            self.tbl_type = tbl_type

    def get_test_results(self):
        """Evaluate all expressions and combine them conjointly."""

        if "polars" in self.tbl_type:
            return self._get_polars_results()
        elif "pandas" in self.tbl_type:
            return self._get_pandas_results()
        elif "duckdb" in self.tbl_type or "ibis" in self.tbl_type:
            return self._get_ibis_results()
        elif "pyspark" in self.tbl_type:
            return self._get_pyspark_results()
        else:  # pragma: no cover
            raise NotImplementedError(f"Support for {self.tbl_type} is not yet implemented")

    def _get_polars_results(self):
        """Process expressions for Polars DataFrames."""
        import polars as pl

        polars_results = []  # Changed from polars_expressions to polars_results

        for expr_fn in self.expressions:
            try:
                # First try direct evaluation with native expressions
                expr_result = expr_fn(self.data_tbl)
                if isinstance(expr_result, pl.Expr):
                    # This is a Polars expression, we'll evaluate it later
                    polars_results.append(("expr", expr_result))
                elif isinstance(expr_result, pl.Series):
                    # This is a boolean Series from lambda function
                    polars_results.append(("series", expr_result))
                else:
                    raise TypeError("Not a valid Polars expression or series")
            except Exception as e:
                try:
                    # Try to get a ColumnExpression
                    col_expr = expr_fn(None)
                    if hasattr(col_expr, "to_polars_expr"):
                        polars_expr = col_expr.to_polars_expr()
                        polars_results.append(("expr", polars_expr))
                    else:  # pragma: no cover
                        raise TypeError(f"Cannot convert {type(col_expr)} to Polars expression")
                except Exception as e:  # pragma: no cover
                    print(f"Error evaluating expression: {e}")

        # Combine results with AND logic
        if polars_results:
            # Convert everything to Series for consistent handling
            series_results = []
            for result_type, result_value in polars_results:
                if result_type == "series":
                    series_results.append(result_value)
                elif result_type == "expr":
                    # Evaluate the expression on the DataFrame to get a Series
                    evaluated_series = self.data_tbl.select(result_value).to_series()
                    series_results.append(evaluated_series)

            # Combine all boolean Series with AND logic
            final_result = series_results[0]
            for series in series_results[1:]:
                final_result = final_result & series

            # Create results table with boolean column
            results_tbl = self.data_tbl.with_columns(pb_is_good_=final_result)
            return results_tbl

        # Default case
        results_tbl = self.data_tbl.with_columns(pb_is_good_=pl.lit(True))  # pragma: no cover
        return results_tbl  # pragma: no cover

    def _get_pandas_results(self):
        """Process expressions for pandas DataFrames."""
        import pandas as pd

        pandas_series = []

        for expr_fn in self.expressions:
            try:
                # First try direct evaluation with pandas DataFrame
                expr_result = expr_fn(self.data_tbl)

                # Check that it's a pandas Series with bool dtype
                if isinstance(expr_result, pd.Series):
                    if expr_result.dtype == bool or pd.api.types.is_bool_dtype(expr_result):
                        pandas_series.append(expr_result)
                    else:  # pragma: no cover
                        raise TypeError(
                            f"Expression returned Series of type {expr_result.dtype}, expected bool"
                        )
                else:  # pragma: no cover
                    raise TypeError(f"Expression returned {type(expr_result)}, expected pd.Series")

            except Exception as e:
                try:
                    # Try as a ColumnExpression (for pb.expr_col style)
                    col_expr = expr_fn(None)

                    if hasattr(col_expr, "to_pandas_expr"):
                        # Watch for NotImplementedError here and re-raise it
                        try:
                            pandas_expr = col_expr.to_pandas_expr(self.data_tbl)
                            pandas_series.append(pandas_expr)
                        except NotImplementedError as nie:  # pragma: no cover
                            # Re-raise NotImplementedError with the original message
                            raise NotImplementedError(str(nie))
                    else:  # pragma: no cover
                        raise TypeError(f"Cannot convert {type(col_expr)} to pandas Series")
                except NotImplementedError as nie:  # pragma: no cover
                    # Re-raise NotImplementedError
                    raise NotImplementedError(str(nie))
                except Exception as nested_e:  # pragma: no cover
                    print(f"Error evaluating pandas expression: {e} -> {nested_e}")

        # Combine results with AND logic
        if pandas_series:
            final_result = pandas_series[0]
            for series in pandas_series[1:]:
                final_result = final_result & series

            # Create results table with boolean column
            results_tbl = self.data_tbl.copy()
            results_tbl["pb_is_good_"] = final_result
            return results_tbl

        # Default case
        results_tbl = self.data_tbl.copy()  # pragma: no cover
        results_tbl["pb_is_good_"] = pd.Series(  # pragma: no cover
            [True] * len(self.data_tbl), index=self.data_tbl.index
        )
        return results_tbl  # pragma: no cover

    def _get_ibis_results(self):
        """Process expressions for Ibis tables (including DuckDB)."""
        import ibis

        ibis_expressions = []

        for expr_fn in self.expressions:
            # Strategy 1: Try direct evaluation with native Ibis expressions
            try:
                expr_result = expr_fn(self.data_tbl)

                # Check if it's a valid Ibis expression
                if hasattr(expr_result, "_ibis_expr"):  # pragma: no cover
                    ibis_expressions.append(expr_result)
                    continue  # Skip to next expression if this worked
            except Exception:  # pragma: no cover
                pass  # Silently continue to Strategy 2

            # Strategy 2: Try with ColumnExpression
            try:  # pragma: no cover
                # Skip this strategy if we don't have an expr_col implementation
                if not hasattr(self, "to_ibis_expr"):
                    continue

                col_expr = expr_fn(None)

                # Skip if we got None
                if col_expr is None:
                    continue

                # Convert ColumnExpression to Ibis expression
                if hasattr(col_expr, "to_ibis_expr"):
                    ibis_expr = col_expr.to_ibis_expr(self.data_tbl)
                    ibis_expressions.append(ibis_expr)
            except Exception:  # pragma: no cover
                # Silent failure where we already tried both strategies
                pass

        # Combine expressions
        if ibis_expressions:  # pragma: no cover
            try:
                final_result = ibis_expressions[0]
                for expr in ibis_expressions[1:]:
                    final_result = final_result & expr

                # Create results table with boolean column
                results_tbl = self.data_tbl.mutate(pb_is_good_=final_result)
                return results_tbl
            except Exception as e:
                print(f"Error combining Ibis expressions: {e}")

        # Default case
        results_tbl = self.data_tbl.mutate(pb_is_good_=ibis.literal(True))
        return results_tbl

    def _get_pyspark_results(self):
        """Process expressions for PySpark DataFrames."""
        from pyspark.sql import functions as F

        pyspark_columns = []

        for expr_fn in self.expressions:
            try:
                # First try direct evaluation with PySpark DataFrame
                expr_result = expr_fn(self.data_tbl)

                # Check if it's a PySpark Column
                if hasattr(expr_result, "_jc"):  # PySpark Column has _jc attribute
                    pyspark_columns.append(expr_result)
                else:
                    raise TypeError(
                        f"Expression returned {type(expr_result)}, expected PySpark Column"
                    )  # pragma: no cover

            except Exception as e:
                try:
                    # Try as a ColumnExpression (for pb.expr_col style)
                    col_expr = expr_fn(None)

                    if hasattr(col_expr, "to_pyspark_expr"):
                        # Convert to PySpark expression
                        pyspark_expr = col_expr.to_pyspark_expr(self.data_tbl)
                        pyspark_columns.append(pyspark_expr)
                    else:
                        raise TypeError(
                            f"Cannot convert {type(col_expr)} to PySpark Column"
                        )  # pragma: no cover
                except Exception as nested_e:
                    print(f"Error evaluating PySpark expression: {e} -> {nested_e}")

        # Combine results with AND logic
        if pyspark_columns:
            final_result = pyspark_columns[0]
            for col in pyspark_columns[1:]:
                final_result = final_result & col

            # Create results table with boolean column
            results_tbl = self.data_tbl.withColumn("pb_is_good_", final_result)
            return results_tbl

        # Default case
        results_tbl = self.data_tbl.withColumn("pb_is_good_", F.lit(True))
        return results_tbl


class SpeciallyValidation:
    def __init__(self, data_tbl, expression, threshold, tbl_type):
        self.data_tbl = data_tbl
        self.expression = expression
        self.threshold = threshold

        # Detect the table type
        if tbl_type in (None, "local"):
            # Detect the table type using _get_tbl_type()
            self.tbl_type = _get_tbl_type(data=data_tbl)
        else:
            self.tbl_type = tbl_type

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
                    result.rename(columns={result.columns[-1]: "pb_is_good_"}, inplace=True)
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

            # Handle LazyFrames which don't have len()
            if is_narwhals_lazyframe(dfn):
                dfn = dfn.collect()

            assert is_narwhals_dataframe(dfn)
            return len(dfn)

        if tbl_type in IBIS_BACKENDS:
            # Get the count of test units and convert to a native format
            # TODO: check whether pandas or polars is available
            return self.df.count().to_polars()  # type: ignore[union-attr]

        raise ValueError(f"Unsupported table type: {tbl_type}")


def _get_compare_expr_nw(compare: Any) -> Any:
    if isinstance(compare, Column):
        if not isinstance(compare.exprs, str):
            raise ValueError("The column expression must be a string.")  # pragma: no cover
        return nw.col(compare.exprs)
    return compare


def _column_has_null_values(table: nw.DataFrame[Any] | nw.LazyFrame[Any], column: str) -> bool:
    try:
        # Try the standard null_count() method (DataFrame)
        null_count = (table.select(column).null_count())[column][0]  # type: ignore[union-attr]
    except AttributeError:
        # For LazyFrames, collect first then get null count
        try:
            collected = table.select(column).collect()  # type: ignore[union-attr]
            null_count = (collected.null_count())[column][0]
        except Exception:
            # Fallback: check if any values are null
            try:
                result = table.select(nw.col(column).is_null().sum().alias("null_count")).collect()  # type: ignore[union-attr]
                null_count = result["null_count"][0]
            except Exception:
                # Last resort: return False (assume no nulls)
                return False

    return null_count is not None and null_count > 0


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


def col_vals_expr(data_tbl: Any, expr: Any, tbl_type: str = "local") -> Any:
    """Check if values in a column evaluate to True for a given predicate expression."""
    if tbl_type == "local":
        # Check the type of expression provided
        if "narwhals" in str(type(expr)) and "expr" in str(type(expr)):
            expression_type = "narwhals"
        elif "polars" in str(type(expr)) and "expr" in str(type(expr)):
            expression_type = "polars"
        else:
            expression_type = "pandas"

        # Determine whether this is a Pandas or Polars table
        tbl_type_detected = _get_tbl_type(data=data_tbl)
        df_lib_name = "polars" if "polars" in tbl_type_detected else "pandas"

        if expression_type == "narwhals":
            tbl_nw = _convert_to_narwhals(df=data_tbl)
            tbl_nw = tbl_nw.with_columns(pb_is_good_=expr)
            return tbl_nw.to_native()

        if df_lib_name == "polars" and expression_type == "polars":
            return data_tbl.with_columns(pb_is_good_=expr)

        if df_lib_name == "pandas" and expression_type == "pandas":
            return data_tbl.assign(pb_is_good_=expr)

    # For remote backends, return original table (placeholder)
    return data_tbl  # pragma: no cover


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
    tbl = _convert_to_narwhals(df=data_tbl)
    return column in tbl.columns


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
    from pointblank.validate import get_row_count

    row_count: int = get_row_count(data=data_tbl)
    lower_abs_limit, upper_abs_limit = abs_tol_bounds
    min_val: int = count - lower_abs_limit
    max_val: int = count + upper_abs_limit

    if inverse:
        return not (row_count >= min_val and row_count <= max_val)
    else:
        return row_count >= min_val and row_count <= max_val


def col_pct_null(
    data_tbl: IntoFrame, column: str, p: float, bound_finder: Callable[[int], AbsoluteBounds]
) -> bool:
    """Check if the percentage of null vales are within p given the absolute bounds."""
    nw_frame = nw.from_native(data_tbl)
    # Handle LazyFrames by collecting them first
    if is_narwhals_lazyframe(nw_frame):
        nw_frame = nw_frame.collect()

    assert is_narwhals_dataframe(nw_frame)

    # We cast as int because it could come back as an arbitary type. For example if the backend
    # is numpy-like, we might get a scalar from `item()`. `int()` expects a certain signature though
    # and `object` does not satisfy so we have to go with the type ignore.
    total_rows: object = nw_frame.select(nw.len()).item()
    total_rows: int = int(total_rows)  # type: ignore

    abs_target: float = round(total_rows * p)
    lower_bound, upper_bound = bound_finder(abs_target)

    # Count null values (see above comment on typing shenanigans)
    n_null: object = nw_frame.select(nw.col(column).is_null().sum()).item()
    n_null: int = int(n_null)  # type: ignore

    return n_null >= (abs_target - lower_bound) and n_null <= (abs_target + upper_bound)


def col_count_match(data_tbl: IntoFrame, count: Any, inverse: bool) -> bool:
    """
    Check if DataFrame column count matches expected count.
    """
    from pointblank.validate import get_column_count

    if not inverse:
        return get_column_count(data=data_tbl) == count
    else:
        return get_column_count(data=data_tbl) != count


def _coerce_to_common_backend(data_tbl: Any, tbl_compare: Any) -> tuple[Any, Any]:
    """
    Coerce two tables to the same backend if they differ.

    If the tables to compare have different backends (e.g., one is Polars and one is Pandas),
    this function will convert the comparison table to match the data table's backend.
    This ensures consistent dtype handling during comparison.

    Parameters
    ----------
    data_tbl
        The primary table (backend is preserved).
    tbl_compare
        The comparison table (may be converted to match data_tbl's backend).

    Returns
    -------
    tuple[Any, Any]
        Both tables, with tbl_compare potentially converted to data_tbl's backend.
    """
    # Get backend types for both tables
    data_backend = _get_tbl_type(data_tbl)
    compare_backend = _get_tbl_type(tbl_compare)

    # If backends match, no conversion needed
    if data_backend == compare_backend:
        return data_tbl, tbl_compare

    # Define database backends (Ibis tables that need materialization)
    database_backends = {"duckdb", "sqlite", "postgres", "mysql", "snowflake", "bigquery"}

    #
    # If backends differ, convert tbl_compare to match data_tbl's backend
    #

    # Handle Ibis/database tables: materialize them to match the target backend
    if compare_backend in database_backends:
        # Materialize to Polars if data table is Polars, otherwise Pandas
        if data_backend == "polars":
            try:
                tbl_compare = tbl_compare.to_polars()
                compare_backend = "polars"
            except Exception:
                # Fallback: materialize to Pandas, then convert to Polars
                try:
                    tbl_compare = tbl_compare.execute()
                    compare_backend = "pandas"
                except Exception:
                    try:
                        tbl_compare = tbl_compare.to_pandas()
                        compare_backend = "pandas"
                    except Exception:
                        pass
        else:
            # Materialize to Pandas for Pandas or other backends
            try:
                tbl_compare = tbl_compare.execute()  # Returns Pandas DataFrame
                compare_backend = "pandas"
            except Exception:
                try:
                    tbl_compare = tbl_compare.to_pandas()
                    compare_backend = "pandas"
                except Exception:
                    pass

    if data_backend in database_backends:
        # If data table itself is a database backend, materialize to Polars
        # (Polars is the default modern backend for optimal performance)
        try:
            data_tbl = data_tbl.to_polars()
            data_backend = "polars"
        except Exception:
            # Fallback to Pandas if Polars conversion fails
            try:
                data_tbl = data_tbl.execute()
                data_backend = "pandas"
            except Exception:
                try:
                    data_tbl = data_tbl.to_pandas()
                    data_backend = "pandas"
                except Exception:
                    pass

    # Now handle the Polars/Pandas conversions
    if data_backend == "polars" and compare_backend == "pandas":
        try:
            import polars as pl

            tbl_compare = pl.from_pandas(tbl_compare)
        except Exception:
            # If conversion fails, return original tables
            pass

    elif data_backend == "pandas" and compare_backend == "polars":
        try:
            tbl_compare = tbl_compare.to_pandas()
        except Exception:
            # If conversion fails, return original tables
            pass

    return data_tbl, tbl_compare


def tbl_match(data_tbl: IntoFrame, tbl_compare: IntoFrame) -> bool:
    """
    Check if two tables match exactly in schema, row count, and data.

    This function performs a comprehensive comparison between two tables,
    checking progressively stricter conditions from least to most stringent:

    1. Column count match
    2. Row count match
    3. Schema match (case-insensitive column names, any order)
    4. Schema match (case-insensitive column names, correct order)
    5. Schema match (case-sensitive column names, correct order)
    6. Data match: compares values column-by-column

    If the two tables have different backends (e.g., one is Polars and one is Pandas),
    the comparison table will be automatically coerced to match the data table's backend
    before comparison. This ensures consistent dtype handling.

    Parameters
    ----------
    data_tbl
        The target table to validate.
    tbl_compare
        The comparison table to validate against.

    Returns
    -------
    bool
        True if tables match completely, False otherwise.
    """
    from pointblank.schema import Schema, _check_schema_match
    from pointblank.validate import get_column_count, get_row_count

    # Coerce to common backend if needed
    data_tbl, tbl_compare = _coerce_to_common_backend(data_tbl, tbl_compare)

    # Convert both tables to narwhals for compatibility
    tbl = _convert_to_narwhals(df=data_tbl)
    tbl_cmp = _convert_to_narwhals(df=tbl_compare)

    # Stage 1: Check column count (least stringent)
    col_count_matching = get_column_count(data=data_tbl) == get_column_count(data=tbl_compare)

    if not col_count_matching:
        return False

    # Stage 2: Check row count
    row_count_matching = get_row_count(data=data_tbl) == get_row_count(data=tbl_compare)

    if not row_count_matching:
        return False

    # Stage 3: Check schema match for case-insensitive column names, any order
    schema = Schema(tbl=tbl_compare)

    col_schema_matching_any_order = _check_schema_match(
        data_tbl=data_tbl,
        schema=schema,
        complete=True,
        in_order=False,
        case_sensitive_colnames=False,
        case_sensitive_dtypes=False,
        full_match_dtypes=False,
    )

    if not col_schema_matching_any_order:
        return False

    # Stage 4: Check schema match for case-insensitive column names, correct order
    col_schema_matching_in_order = _check_schema_match(
        data_tbl=data_tbl,
        schema=schema,
        complete=True,
        in_order=True,
        case_sensitive_colnames=False,
        case_sensitive_dtypes=False,
        full_match_dtypes=False,
    )

    if not col_schema_matching_in_order:
        return False

    # Stage 5: Check schema match for case-sensitive column names, correct order
    col_schema_matching_exact = _check_schema_match(
        data_tbl=data_tbl,
        schema=schema,
        complete=True,
        in_order=True,
        case_sensitive_colnames=True,
        case_sensitive_dtypes=False,
        full_match_dtypes=False,
    )

    if not col_schema_matching_exact:
        return False

    # Stage 6: Check for exact data by cell across matched columns (most stringent)
    # Handle edge case where both tables have zero rows (they match)
    if get_row_count(data=data_tbl) == 0:
        return True

    column_count = get_column_count(data=data_tbl)

    # Compare column-by-column
    for i in range(column_count):
        # Get column name
        col_name = tbl.columns[i]

        # Get column data from both tables
        col_data_1 = tbl.select(col_name)
        col_data_2 = tbl_cmp.select(col_name)

        # Convert to native format for comparison
        # We need to collect if lazy frames
        if is_narwhals_lazyframe(col_data_1):
            col_data_1 = col_data_1.collect()

        if is_narwhals_lazyframe(col_data_2):
            col_data_2 = col_data_2.collect()

        # Convert to native and then to lists for comparison
        # Native frames could be Polars, Pandas, or Ibis - use Any for dynamic access
        col_1_native: Any = col_data_1.to_native()
        col_2_native: Any = col_data_2.to_native()

        # Extract values as lists for comparison
        # Note: We use hasattr for runtime detection but maintain Any typing
        values_1: list[Any]
        values_2: list[Any]
        if hasattr(col_1_native, "to_list"):  # Polars DataFrame
            values_1 = col_1_native[col_name].to_list()  # type: ignore[index]
            values_2 = col_2_native[col_name].to_list()  # type: ignore[index]

        elif hasattr(col_1_native, "tolist"):  # Pandas DataFrame
            values_1 = col_1_native[col_name].tolist()  # type: ignore[index]
            values_2 = col_2_native[col_name].tolist()  # type: ignore[index]

        elif hasattr(col_1_native, "collect"):  # Ibis
            values_1 = col_1_native[col_name].to_pandas().tolist()  # type: ignore[index]
            values_2 = col_2_native[col_name].to_pandas().tolist()  # type: ignore[index]

        else:
            # Fallback: try direct comparison
            values_1 = list(col_1_native[col_name])  # type: ignore[index]
            values_2 = list(col_2_native[col_name])  # type: ignore[index]

        # Compare the two lists element by element, handling NaN/None
        if len(values_1) != len(values_2):
            return False

        for v1, v2 in zip(values_1, values_2):
            # Handle None/NaN comparisons and check both None and NaN
            # Note: When Pandas NaN is converted to Polars, it may become None
            v1_is_null = v1 is None
            v2_is_null = v2 is None

            # Check if v1 is NaN
            if not v1_is_null:
                try:
                    import math

                    if math.isnan(v1):
                        v1_is_null = True
                except (TypeError, ValueError):
                    pass

            # Check if v2 is NaN
            if not v2_is_null:
                try:
                    import math

                    if math.isnan(v2):
                        v2_is_null = True
                except (TypeError, ValueError):
                    pass

            # If both are null (None or NaN), they match
            if v1_is_null and v2_is_null:
                continue

            # If only one is null, they don't match
            if v1_is_null or v2_is_null:
                return False

            # Direct comparison: handle lists/arrays separately
            try:
                if v1 != v2:
                    return False
            except (TypeError, ValueError):
                # If direct comparison fails (e.g., for lists/arrays), try element-wise comparison
                try:
                    if isinstance(v1, list) and isinstance(v2, list):
                        if v1 != v2:
                            return False
                    elif hasattr(v1, "__eq__") and hasattr(v2, "__eq__"):
                        # For array-like objects, check if they're equal
                        if not (v1 == v2).all() if hasattr((v1 == v2), "all") else v1 == v2:
                            return False
                    else:
                        return False
                except Exception:
                    return False

    return True


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

    nw_tbl = nw.from_native(tbl)
    assert is_narwhals_dataframe(nw_tbl) or is_narwhals_lazyframe(nw_tbl)

    if isinstance(compare, Column):
        compare_expr = _get_compare_expr_nw(compare=compare)

        result_tbl = nw_tbl.with_columns(
            pb_is_good_1=nw.col(column).is_null() & na_pass,
            pb_is_good_2=(
                nw.col(compare.name).is_null() & na_pass
                if isinstance(compare, Column)
                else nw.lit(False)
            ),
        )

        result_tbl = result_tbl.with_columns(
            pb_is_good_3=(~nw.col(compare.name).is_null() & ~nw.col(column).is_null())
        )

        if is_pandas_dataframe(result_tbl.to_native()):
            # For Pandas, handle potential NA comparison issues
            try:
                result_tbl = result_tbl.with_columns(
                    pb_is_good_4=nw.col(column) == compare_expr,
                )
            except (TypeError, ValueError) as e:
                # Handle Pandas NA comparison issues
                if "boolean value of NA is ambiguous" in str(e):
                    # Work around Pandas NA comparison issue by using Null checks first
                    result_tbl = result_tbl.with_columns(
                        pb_is_good_4_tmp=(
                            # Both Null: True (they're equal)
                            (nw.col(column).is_null() & nw.col(compare.name).is_null())
                            |
                            # Both not Null and values are equal: use string conversion
                            # as a fallback
                            (
                                (~nw.col(column).is_null() & ~nw.col(compare.name).is_null())
                                & (
                                    nw.col(column).cast(nw.String)
                                    == nw.col(compare.name).cast(nw.String)
                                )
                            )
                        )
                    )
                    result_tbl = result_tbl.rename({"pb_is_good_4_tmp": "pb_is_good_4"})
                elif "cannot compare" in str(e).lower():
                    # Handle genuine type incompatibility - native_df type varies by backend
                    native_df = result_tbl.to_native()
                    col_dtype = str(native_df[column].dtype)  # type: ignore[index]
                    compare_dtype = str(native_df[compare.name].dtype)  # type: ignore[index]

                    raise TypeError(
                        f"Cannot compare columns '{column}' (dtype: {col_dtype}) and "
                        f"'{compare.name}' (dtype: {compare_dtype}). "
                        f"Column types are incompatible for equality comparison. "
                        f"Ensure both columns have compatible data types (both numeric, "
                        f"both string, or both datetime) before comparing."
                    ) from e
                else:
                    raise  # Re-raise unexpected errors

            result_tbl = result_tbl.with_columns(
                pb_is_good_=nw.col("pb_is_good_1")
                | nw.col("pb_is_good_2")
                | (nw.col("pb_is_good_4") & ~nw.col("pb_is_good_1") & ~nw.col("pb_is_good_2"))
            )

        else:
            # For non-Pandas backends (Polars, Ibis, etc.), handle type incompatibility
            try:
                result_tbl = result_tbl.with_columns(
                    pb_is_good_4=nw.col(column) == compare_expr,
                )
            except (TypeError, ValueError, Exception) as e:
                # Handle type compatibility issues for all backends
                error_msg = str(e).lower()
                if (
                    "cannot compare" in error_msg
                    or "type" in error_msg
                    and ("mismatch" in error_msg or "incompatible" in error_msg)
                    or "dtype" in error_msg
                    or "conversion" in error_msg
                    and "failed" in error_msg
                ):
                    # Get column types for a descriptive error message - native type varies by backend
                    col_dtype = "unknown"
                    compare_dtype = "unknown"
                    try:
                        native_df = result_tbl.to_native()
                        if hasattr(native_df, "dtypes"):
                            col_dtype = str(native_df.dtypes.get(column, "unknown"))  # type: ignore[union-attr]
                            compare_dtype = str(native_df.dtypes.get(compare.name, "unknown"))  # type: ignore[union-attr]
                        elif hasattr(native_df, "schema"):
                            col_dtype = str(native_df.schema.get(column, "unknown"))  # type: ignore[union-attr]
                            compare_dtype = str(native_df.schema.get(compare.name, "unknown"))  # type: ignore[union-attr]
                    except Exception:
                        pass

                    raise TypeError(
                        f"Cannot compare columns '{column}' (dtype: {col_dtype}) and "
                        f"'{compare.name}' (dtype: {compare_dtype}). "
                        f"Column types are incompatible for equality comparison. "
                        f"Ensure both columns have compatible data types (both numeric, "
                        f"both string, or both datetime) before comparing."
                    ) from e
                else:
                    raise  # Re-raise unexpected errors

            result_tbl = result_tbl.with_columns(
                pb_is_good_=nw.col("pb_is_good_1")
                | nw.col("pb_is_good_2")
                | (nw.col("pb_is_good_4") & ~nw.col("pb_is_good_1") & ~nw.col("pb_is_good_2"))
            )

        return result_tbl.drop(
            "pb_is_good_1", "pb_is_good_2", "pb_is_good_3", "pb_is_good_4"
        ).to_native()

    else:
        compare_expr = _get_compare_expr_nw(compare=compare)
        compare_expr = _safe_modify_datetime_compare_val(nw_tbl, column, compare_expr)

        result_tbl = nw_tbl.with_columns(
            pb_is_good_1=nw.col(column).is_null() & na_pass,
            pb_is_good_2=(
                nw.col(compare.name).is_null() & na_pass
                if isinstance(compare, Column)
                else nw.lit(False)
            ),
        )

        # Handle type incompatibility for literal value comparisons
        try:
            result_tbl = result_tbl.with_columns(pb_is_good_3=nw.col(column) == compare_expr)
        except (TypeError, ValueError, Exception) as e:
            # Handle type compatibility issues for column vs literal comparisons
            error_msg = str(e).lower()
            if (
                "cannot compare" in error_msg
                or "type" in error_msg
                and ("mismatch" in error_msg or "incompatible" in error_msg)
                or "dtype" in error_msg
                or "conversion" in error_msg
                and "failed" in error_msg
            ):
                # Get column type for a descriptive error message - native type varies by backend
                col_dtype = "unknown"
                try:
                    native_df = result_tbl.to_native()
                    if hasattr(native_df, "dtypes"):
                        col_dtype = str(native_df.dtypes.get(column, "unknown"))  # type: ignore[union-attr]
                    elif hasattr(native_df, "schema"):
                        col_dtype = str(native_df.schema.get(column, "unknown"))  # type: ignore[union-attr]
                except Exception:
                    pass

                compare_type = type(compare).__name__
                compare_value = str(compare)

                raise TypeError(
                    f"Cannot compare column '{column}' (dtype: {col_dtype}) with "
                    f"literal value '{compare_value}' (type: {compare_type}). "
                    f"Column type and literal value type are incompatible for equality comparison. "
                    f"Ensure the column data type is compatible with the comparison value "
                    f"(e.g., numeric column with numeric value, string column with string value)."
                ) from e
            else:
                raise  # Re-raise unexpected errors

        result_tbl = result_tbl.with_columns(
            pb_is_good_3=(
                nw.when(nw.col("pb_is_good_3").is_null())
                .then(nw.lit(False))
                .otherwise(nw.col("pb_is_good_3"))
            )
        )

        result_tbl = result_tbl.with_columns(
            pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3")
        )

        return result_tbl.drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3").to_native()


def interrogate_ne(tbl: IntoFrame, column: str, compare: Any, na_pass: bool) -> Any:
    """Not equal interrogation."""

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))

    # Determine if the reference and comparison columns have any null values
    ref_col_has_null_vals = _column_has_null_values(table=nw_tbl, column=column)

    if isinstance(compare, Column):
        compare_name = compare.name if isinstance(compare, Column) else compare
        cmp_col_has_null_vals = _column_has_null_values(table=nw_tbl, column=compare_name)
    else:
        cmp_col_has_null_vals = False

    # If neither column has null values, we can proceed with the comparison
    # without too many complications
    if not ref_col_has_null_vals and not cmp_col_has_null_vals:
        if isinstance(compare, Column):
            compare_expr = _get_compare_expr_nw(compare=compare)

            # Handle type incompatibility for column comparisons
            try:
                return nw_tbl.with_columns(
                    pb_is_good_=nw.col(column) != compare_expr,
                ).to_native()
            except (TypeError, ValueError, Exception) as e:
                # Handle type compatibility issues for column vs column comparisons
                error_msg = str(e).lower()
                if (
                    "cannot compare" in error_msg
                    or "type" in error_msg
                    and ("mismatch" in error_msg or "incompatible" in error_msg)
                    or "dtype" in error_msg
                    or "conversion" in error_msg
                    and "failed" in error_msg
                    or "boolean value of na is ambiguous" in error_msg
                ):
                    # Get column types for a descriptive error message
                    try:
                        native_df = nw_tbl.to_native()
                        if hasattr(native_df, "dtypes"):
                            col_dtype = str(native_df.dtypes.get(column, "unknown"))
                            compare_dtype = str(native_df.dtypes.get(compare.name, "unknown"))
                        elif hasattr(native_df, "schema"):
                            col_dtype = str(native_df.schema.get(column, "unknown"))
                            compare_dtype = str(native_df.schema.get(compare.name, "unknown"))
                        else:
                            col_dtype = "unknown"
                            compare_dtype = "unknown"
                    except Exception:
                        col_dtype = "unknown"
                        compare_dtype = "unknown"

                    raise TypeError(
                        f"Cannot compare columns '{column}' (dtype: {col_dtype}) and "
                        f"'{compare.name}' (dtype: {compare_dtype}). "
                        f"Column types are incompatible for inequality comparison. "
                        f"Ensure both columns have compatible data types (both numeric, "
                        f"both string, or both datetime) before comparing."
                    ) from e
                else:
                    raise  # Re-raise unexpected errors

        else:
            compare_expr = _safe_modify_datetime_compare_val(nw_tbl, column, compare)

            # Handle type incompatibility for literal comparisons
            try:
                return nw_tbl.with_columns(
                    pb_is_good_=nw.col(column) != nw.lit(compare_expr),
                ).to_native()
            except (TypeError, ValueError, Exception) as e:
                # Handle type compatibility issues for column vs literal comparisons
                error_msg = str(e).lower()
                if (
                    "cannot compare" in error_msg
                    or "type" in error_msg
                    and ("mismatch" in error_msg or "incompatible" in error_msg)
                    or "dtype" in error_msg
                    or "conversion" in error_msg
                    and "failed" in error_msg
                ):
                    # Get column type for a descriptive error message
                    try:
                        native_df = nw_tbl.to_native()
                        if hasattr(native_df, "dtypes"):
                            col_dtype = str(native_df.dtypes.get(column, "unknown"))
                        elif hasattr(native_df, "schema"):
                            col_dtype = str(native_df.schema.get(column, "unknown"))
                        else:
                            col_dtype = "unknown"
                    except Exception:
                        col_dtype = "unknown"

                    compare_type = type(compare).__name__
                    compare_value = str(compare)

                    raise TypeError(
                        f"Cannot compare column '{column}' (dtype: {col_dtype}) with "
                        f"literal value '{compare_value}' (type: {compare_type}). "
                        f"Column type and literal value type are incompatible for inequality comparison. "
                        f"Ensure the column data type is compatible with the comparison value "
                        f"(e.g., numeric column with numeric value, string column with string value)."
                    ) from e
                else:
                    raise  # Re-raise unexpected errors

    # If either column has Null values, we need to handle the comparison
    # much more carefully since we can't inadvertently compare Null values
    # to non-Null values

    if isinstance(compare, Column):
        compare_expr = _get_compare_expr_nw(compare=compare)

        # CASE 1: the reference column has Null values but the comparison column does not
        if ref_col_has_null_vals and not cmp_col_has_null_vals:
            if is_pandas_dataframe(nw_tbl.to_native()):
                try:
                    result_tbl = nw_tbl.with_columns(
                        pb_is_good_1=nw.col(column).is_null(),
                        pb_is_good_2=nw.col(column) != nw.col(compare.name),
                    )
                except (TypeError, ValueError) as e:
                    # Handle Pandas type compatibility issues
                    if (
                        "boolean value of NA is ambiguous" in str(e)
                        or "cannot compare" in str(e).lower()
                    ):
                        # Get column types for a descriptive error message
                        native_df = nw_tbl.to_native()
                        col_dtype = str(native_df[column].dtype)
                        compare_dtype = str(native_df[compare.name].dtype)

                        raise TypeError(
                            f"Cannot compare columns '{column}' (dtype: {col_dtype}) and "
                            f"'{compare.name}' (dtype: {compare_dtype}). "
                            f"Column types are incompatible for inequality comparison. "
                            f"Ensure both columns have compatible data types (both numeric, "
                            f"both string, or both datetime) before comparing."
                        ) from e
                    else:
                        raise  # Re-raise unexpected errors

            else:
                try:
                    result_tbl = nw_tbl.with_columns(
                        pb_is_good_1=nw.col(column).is_null(),
                        pb_is_good_2=nw.col(column) != nw.col(compare.name),
                    )
                except (TypeError, ValueError, Exception) as e:
                    # Handle type compatibility issues for non-Pandas backends
                    error_msg = str(e).lower()
                    if (
                        "cannot compare" in error_msg
                        or "type" in error_msg
                        and ("mismatch" in error_msg or "incompatible" in error_msg)
                        or "dtype" in error_msg
                        or "conversion" in error_msg
                        and "failed" in error_msg
                    ):
                        # Get column types for a descriptive error message
                        try:
                            native_df = nw_tbl.to_native()
                            if hasattr(native_df, "dtypes"):
                                col_dtype = str(native_df.dtypes.get(column, "unknown"))
                                compare_dtype = str(native_df.dtypes.get(compare.name, "unknown"))
                            elif hasattr(native_df, "schema"):
                                col_dtype = str(native_df.schema.get(column, "unknown"))
                                compare_dtype = str(native_df.schema.get(compare.name, "unknown"))
                            else:
                                col_dtype = "unknown"
                                compare_dtype = "unknown"
                        except Exception:
                            col_dtype = "unknown"
                            compare_dtype = "unknown"

                        raise TypeError(
                            f"Cannot compare columns '{column}' (dtype: {col_dtype}) and "
                            f"'{compare.name}' (dtype: {compare_dtype}). "
                            f"Column types are incompatible for inequality comparison. "
                            f"Ensure both columns have compatible data types (both numeric, "
                            f"both string, or both datetime) before comparing."
                        ) from e
                    else:
                        raise  # Re-raise unexpected errors

            if not na_pass:
                result_tbl = result_tbl.with_columns(
                    pb_is_good_2=nw.col("pb_is_good_2") & ~nw.col("pb_is_good_1")
                )

            if is_polars_dataframe(nw_tbl.to_native()):
                # There may be Null values in the `pb_is_good_2` column, change those to
                # True if `na_pass=` is True, False otherwise

                result_tbl = result_tbl.with_columns(
                    pb_is_good_2=nw.when(nw.col("pb_is_good_2").is_null())
                    .then(False)
                    .otherwise(nw.col("pb_is_good_2")),
                )

                if na_pass:
                    result_tbl = result_tbl.with_columns(
                        pb_is_good_2=(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                    )
            else:
                # General case (non-Polars): handle na_pass=True properly
                if na_pass:
                    result_tbl = result_tbl.with_columns(
                        pb_is_good_2=(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                    )

            return (
                result_tbl.with_columns(pb_is_good_=nw.col("pb_is_good_2"))
                .drop("pb_is_good_1", "pb_is_good_2")
                .to_native()
            )

        # CASE 2: the comparison column has Null values but the reference column does not
        elif not ref_col_has_null_vals and cmp_col_has_null_vals:
            if is_pandas_dataframe(nw_tbl.to_native()):
                try:
                    result_tbl = nw_tbl.with_columns(
                        pb_is_good_1=nw.col(column) != nw.lit(compare.name),
                        pb_is_good_2=nw.col(compare.name).is_null(),
                    )
                except (TypeError, ValueError) as e:
                    # Handle Pandas type compatibility issues
                    if (
                        "boolean value of NA is ambiguous" in str(e)
                        or "cannot compare" in str(e).lower()
                    ):
                        # Get column types for a descriptive error message
                        native_df = nw_tbl.to_native()
                        col_dtype = str(native_df[column].dtype)
                        compare_dtype = str(native_df[compare.name].dtype)

                        raise TypeError(
                            f"Cannot compare columns '{column}' (dtype: {col_dtype}) and "
                            f"'{compare.name}' (dtype: {compare_dtype}). "
                            f"Column types are incompatible for inequality comparison. "
                            f"Ensure both columns have compatible data types (both numeric, "
                            f"both string, or both datetime) before comparing."
                        ) from e
                    else:
                        raise  # Re-raise unexpected errors

            else:
                try:
                    result_tbl = nw_tbl.with_columns(
                        pb_is_good_1=nw.col(column) != nw.col(compare.name),
                        pb_is_good_2=nw.col(compare.name).is_null(),
                    )
                except (TypeError, ValueError, Exception) as e:
                    # Handle type compatibility issues for non-Pandas backends
                    error_msg = str(e).lower()
                    if (
                        "cannot compare" in error_msg
                        or "type" in error_msg
                        and ("mismatch" in error_msg or "incompatible" in error_msg)
                        or "dtype" in error_msg
                        or "conversion" in error_msg
                        and "failed" in error_msg
                    ):
                        # Get column types for a descriptive error message
                        try:
                            native_df = nw_tbl.to_native()
                            if hasattr(native_df, "dtypes"):
                                col_dtype = str(native_df.dtypes.get(column, "unknown"))
                                compare_dtype = str(native_df.dtypes.get(compare.name, "unknown"))
                            elif hasattr(native_df, "schema"):
                                col_dtype = str(native_df.schema.get(column, "unknown"))
                                compare_dtype = str(native_df.schema.get(compare.name, "unknown"))
                            else:
                                col_dtype = "unknown"
                                compare_dtype = "unknown"
                        except Exception:
                            col_dtype = "unknown"
                            compare_dtype = "unknown"

                        raise TypeError(
                            f"Cannot compare columns '{column}' (dtype: {col_dtype}) and "
                            f"'{compare.name}' (dtype: {compare_dtype}). "
                            f"Column types are incompatible for inequality comparison. "
                            f"Ensure both columns have compatible data types (both numeric, "
                            f"both string, or both datetime) before comparing."
                        ) from e
                    else:
                        raise  # Re-raise unexpected errors

            if not na_pass:
                result_tbl = result_tbl.with_columns(
                    pb_is_good_1=nw.col("pb_is_good_1") & ~nw.col("pb_is_good_2")
                )

            if is_polars_dataframe(nw_tbl.to_native()):
                if na_pass:
                    result_tbl = result_tbl.with_columns(
                        pb_is_good_1=(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                    )
            else:
                # General case (non-Polars): handle `na_pass=True` properly
                if na_pass:
                    result_tbl = result_tbl.with_columns(
                        pb_is_good_1=(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                    )

            return (
                result_tbl.with_columns(pb_is_good_=nw.col("pb_is_good_1"))
                .drop("pb_is_good_1", "pb_is_good_2")
                .to_native()
            )

        # CASE 3: both columns have Null values and there may potentially be cases where
        # there could even be Null/Null comparisons
        elif ref_col_has_null_vals and cmp_col_has_null_vals:
            try:
                result_tbl = nw_tbl.with_columns(
                    pb_is_good_1=nw.col(column).is_null(),
                    pb_is_good_2=nw.col(compare.name).is_null(),
                    pb_is_good_3=nw.col(column) != nw.col(compare.name),
                )
            except (TypeError, ValueError, Exception) as e:
                # Handle type compatibility issues for column vs column comparisons
                error_msg = str(e).lower()
                if (
                    "cannot compare" in error_msg
                    or "type" in error_msg
                    and ("mismatch" in error_msg or "incompatible" in error_msg)
                    or "dtype" in error_msg
                    or "conversion" in error_msg
                    and "failed" in error_msg
                    or "boolean value of na is ambiguous" in error_msg
                ):
                    # Get column types for a descriptive error message
                    try:
                        native_df = nw_tbl.to_native()
                        if hasattr(native_df, "dtypes"):
                            col_dtype = str(native_df.dtypes.get(column, "unknown"))
                            compare_dtype = str(native_df.dtypes.get(compare.name, "unknown"))
                        elif hasattr(native_df, "schema"):
                            col_dtype = str(native_df.schema.get(column, "unknown"))
                            compare_dtype = str(native_df.schema.get(compare.name, "unknown"))
                        else:
                            col_dtype = "unknown"
                            compare_dtype = "unknown"
                    except Exception:
                        col_dtype = "unknown"
                        compare_dtype = "unknown"

                    raise TypeError(
                        f"Cannot compare columns '{column}' (dtype: {col_dtype}) and "
                        f"'{compare.name}' (dtype: {compare_dtype}). "
                        f"Column types are incompatible for inequality comparison. "
                        f"Ensure both columns have compatible data types (both numeric, "
                        f"both string, or both datetime) before comparing."
                    ) from e
                else:
                    raise  # Re-raise unexpected errors

            if not na_pass:
                result_tbl = result_tbl.with_columns(
                    pb_is_good_3=nw.col("pb_is_good_3")
                    & ~nw.col("pb_is_good_1")
                    & ~nw.col("pb_is_good_2")
                )

            if is_polars_dataframe(nw_tbl.to_native()):
                if na_pass:
                    result_tbl = result_tbl.with_columns(
                        pb_is_good_3=(
                            nw.when(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                            .then(True)
                            .otherwise(False)
                        )
                    )
            else:
                # General case (non-Polars): handle na_pass=True properly
                if na_pass:
                    result_tbl = result_tbl.with_columns(
                        pb_is_good_3=(
                            nw.when(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                            .then(True)
                            .otherwise(nw.col("pb_is_good_3"))
                        )
                    )

            return (
                result_tbl.with_columns(pb_is_good_=nw.col("pb_is_good_3"))
                .drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")
                .to_native()
            )

    else:
        # Case where the reference column contains null values
        if ref_col_has_null_vals:
            # Create individual cases for Pandas and Polars
            compare_expr = _safe_modify_datetime_compare_val(nw_tbl, column, compare)

            if is_pandas_dataframe(nw_tbl.to_native()):
                try:
                    result_tbl = nw_tbl.with_columns(
                        pb_is_good_1=nw.col(column).is_null(),
                        pb_is_good_2=nw.col(column) != nw.lit(compare_expr),
                    )
                except (TypeError, ValueError) as e:
                    # Handle Pandas type compatibility issues for literal comparisons
                    if (
                        "boolean value of NA is ambiguous" in str(e)
                        or "cannot compare" in str(e).lower()
                    ):
                        # Get column type for a descriptive error message
                        native_df = nw_tbl.to_native()
                        col_dtype = str(native_df[column].dtype)
                        compare_type = type(compare).__name__
                        compare_value = str(compare)

                        raise TypeError(
                            f"Cannot compare column '{column}' (dtype: {col_dtype}) with "
                            f"literal value '{compare_value}' (type: {compare_type}). "
                            f"Column type and literal value type are incompatible for inequality comparison. "
                            f"Ensure the column data type is compatible with the comparison value "
                            f"(e.g., numeric column with numeric value, string column with string value)."
                        ) from e
                    else:
                        raise  # Re-raise unexpected errors

                if not na_pass:
                    result_tbl = result_tbl.with_columns(
                        pb_is_good_2=nw.col("pb_is_good_2") & ~nw.col("pb_is_good_1")
                    )

                return (
                    result_tbl.with_columns(pb_is_good_=nw.col("pb_is_good_2"))
                    .drop("pb_is_good_1", "pb_is_good_2")
                    .to_native()
                )

            elif is_polars_dataframe(nw_tbl.to_native()):
                result_tbl = nw_tbl.with_columns(
                    pb_is_good_1=nw.col(column).is_null(),  # val is Null in Column
                    pb_is_good_2=nw.lit(na_pass),  # Pass if any Null in val or compare
                )

                try:
                    result_tbl = result_tbl.with_columns(
                        pb_is_good_3=nw.col(column) != nw.lit(compare_expr)
                    )
                except (TypeError, ValueError, Exception) as e:
                    # Handle type compatibility issues for literal comparisons
                    error_msg = str(e).lower()
                    if (
                        "cannot compare" in error_msg
                        or "type" in error_msg
                        and ("mismatch" in error_msg or "incompatible" in error_msg)
                        or "dtype" in error_msg
                        or "conversion" in error_msg
                        and "failed" in error_msg
                    ):
                        # Get column type for a descriptive error message
                        try:
                            native_df = nw_tbl.to_native()
                            if hasattr(native_df, "dtypes"):
                                col_dtype = str(native_df.dtypes.get(column, "unknown"))
                            elif hasattr(native_df, "schema"):
                                col_dtype = str(native_df.schema.get(column, "unknown"))
                            else:
                                col_dtype = "unknown"
                        except Exception:
                            col_dtype = "unknown"

                        compare_type = type(compare).__name__
                        compare_value = str(compare)

                        raise TypeError(
                            f"Cannot compare column '{column}' (dtype: {col_dtype}) with "
                            f"literal value '{compare_value}' (type: {compare_type}). "
                            f"Column type and literal value type are incompatible for inequality comparison. "
                            f"Ensure the column data type is compatible with the comparison value "
                            f"(e.g., numeric column with numeric value, string column with string value)."
                        ) from e
                    else:
                        raise  # Re-raise unexpected errors

                result_tbl = result_tbl.with_columns(
                    pb_is_good_=(
                        (nw.col("pb_is_good_1") & nw.col("pb_is_good_2"))
                        | (nw.col("pb_is_good_3") & ~nw.col("pb_is_good_1"))
                    )
                )

                result_tbl = result_tbl.drop(
                    "pb_is_good_1", "pb_is_good_2", "pb_is_good_3"
                ).to_native()

                return result_tbl

            else:
                # Generic case for other DataFrame types (PySpark, etc.)
                # Use similar logic to Polars but handle potential differences
                result_tbl = nw_tbl.with_columns(
                    pb_is_good_1=nw.col(column).is_null(),  # val is Null in Column
                    pb_is_good_2=nw.lit(na_pass),  # Pass if any Null in val or compare
                )

                try:
                    result_tbl = result_tbl.with_columns(
                        pb_is_good_3=nw.col(column) != nw.lit(compare_expr)
                    )
                except (TypeError, ValueError, Exception) as e:
                    # Handle type compatibility issues for literal comparisons
                    error_msg = str(e).lower()
                    if (
                        "cannot compare" in error_msg
                        or "type" in error_msg
                        and ("mismatch" in error_msg or "incompatible" in error_msg)
                        or "dtype" in error_msg
                        or "conversion" in error_msg
                        and "failed" in error_msg
                    ):
                        # Get column type for a descriptive error message
                        try:
                            native_df = nw_tbl.to_native()
                            if hasattr(native_df, "dtypes"):
                                col_dtype = str(native_df.dtypes.get(column, "unknown"))
                            elif hasattr(native_df, "schema"):
                                col_dtype = str(native_df.schema.get(column, "unknown"))
                            else:
                                col_dtype = "unknown"
                        except Exception:
                            col_dtype = "unknown"

                        compare_type = type(compare).__name__
                        compare_value = str(compare)

                        raise TypeError(
                            f"Cannot compare column '{column}' (dtype: {col_dtype}) with "
                            f"literal value '{compare_value}' (type: {compare_type}). "
                            f"Column type and literal value type are incompatible for inequality comparison. "
                            f"Ensure the column data type is compatible with the comparison value "
                            f"(e.g., numeric column with numeric value, string column with string value)."
                        ) from e
                    else:
                        raise  # Re-raise unexpected errors

                result_tbl = result_tbl.with_columns(
                    pb_is_good_=(
                        (nw.col("pb_is_good_1") & nw.col("pb_is_good_2"))
                        | (nw.col("pb_is_good_3") & ~nw.col("pb_is_good_1"))
                    )
                )

                return result_tbl.drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3").to_native()


def interrogate_between(
    tbl: IntoFrame, column: str, low: Any, high: Any, inclusive: tuple[bool, bool], na_pass: bool
) -> Any:
    """Between interrogation."""

    low_val = _get_compare_expr_nw(compare=low)
    high_val = _get_compare_expr_nw(compare=high)

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))
    low_val = _safe_modify_datetime_compare_val(nw_tbl, column, low_val)
    high_val = _safe_modify_datetime_compare_val(nw_tbl, column, high_val)

    result_tbl = nw_tbl.with_columns(
        pb_is_good_1=nw.col(column).is_null(),  # val is Null in Column
        pb_is_good_2=(  # lb is Null in Column
            nw.col(low.name).is_null() if isinstance(low, Column) else nw.lit(False)
        ),
        pb_is_good_3=(  # ub is Null in Column
            nw.col(high.name).is_null() if isinstance(high, Column) else nw.lit(False)
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

    result_tbl = result_tbl.with_columns(
        pb_is_good_5=(
            nw.when(nw.col("pb_is_good_5").is_null())
            .then(nw.lit(False))
            .otherwise(nw.col("pb_is_good_5"))
        )
    )

    result_tbl = result_tbl.with_columns(
        pb_is_good_6=(
            nw.when(nw.col("pb_is_good_6").is_null())
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
    low_val = _safe_modify_datetime_compare_val(nw_tbl, column, low_val)
    high_val = _safe_modify_datetime_compare_val(nw_tbl, column, high_val)

    result_tbl = nw_tbl.with_columns(
        pb_is_good_1=nw.col(column).is_null(),  # val is Null in Column
        pb_is_good_2=(  # lb is Null in Column
            nw.col(low.name).is_null() if isinstance(low, Column) else nw.lit(False)
        ),
        pb_is_good_3=(  # ub is Null in Column
            nw.col(high.name).is_null() if isinstance(high, Column) else nw.lit(False)
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

    result_tbl = result_tbl.with_columns(
        pb_is_good_5=nw.when(nw.col("pb_is_good_5").is_null())
        .then(False)
        .otherwise(nw.col("pb_is_good_5")),
        pb_is_good_6=nw.when(nw.col("pb_is_good_6").is_null())
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
    base_expr: nw.Expr = nw.col(column).is_in(set_values)
    if can_be_null:
        base_expr = base_expr | nw.col(column).is_null()

    result_tbl = nw_tbl.with_columns(pb_is_good_=base_expr)
    return result_tbl.to_native()


def interrogate_notin(tbl: IntoFrame, column: str, set_values: Any) -> Any:
    """Not in set interrogation."""

    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))
    result_tbl = nw_tbl.with_columns(
        pb_is_good_=nw.col(column).is_in(set_values),
    ).with_columns(pb_is_good_=~nw.col("pb_is_good_"))
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
    result_tbl = nw_tbl.with_columns(
        pb_is_good_1=nw.col(column).is_null() & na_pass,
        pb_is_good_2=nw.col(column).str.contains(pattern, literal=False).fill_null(False),
    )

    # Apply inverse logic if needed
    if inverse:
        # Use explicit boolean logic instead of bitwise NOT for pandas compatibility
        result_tbl = result_tbl.with_columns(
            pb_is_good_2=nw.when(nw.col("pb_is_good_2")).then(False).otherwise(True)
        )

    result_tbl = result_tbl.with_columns(
        pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2")
    ).drop("pb_is_good_1", "pb_is_good_2")

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

        # For SWIFT/BIC, need to uppercase first
        if spec_lower in ("swift", "swift_bic", "bic"):
            col_expr = nw.col(column).str.to_uppercase()
        else:
            col_expr = nw.col(column)

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
    if is_ibis and spec_lower == "vin":
        # Route to database-native VIN validation
        return interrogate_within_spec_db(tbl, column, values, na_pass)
    elif is_ibis and spec_lower in ("credit_card", "creditcard"):
        # Route to database-native credit card validation
        return interrogate_credit_card_db(tbl, column, values, na_pass)

    # For non-Ibis tables or other specs, materialize data and use Python validation
    # Get the column data as a list
    col_data: Any = nw_tbl.select(column).to_native()

    # Convert to list based on backend - type varies so use duck typing
    if hasattr(col_data, "to_list"):  # Polars
        col_list = col_data[column].to_list()  # type: ignore[index]
    elif hasattr(col_data, "tolist"):  # Pandas
        col_list = col_data[column].tolist()  # type: ignore[index]
    else:  # For Ibis tables, we need to execute the query first
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
    if hasattr(native_tbl, "execute"):
        native_tbl = native_tbl.execute()

    # Add validation column: convert native table to Series, then back through Narwhals
    if is_polars_dataframe(native_tbl):
        import polars as pl

        native_tbl = native_tbl.with_columns(pb_is_good_2=pl.Series(is_valid_list))
    elif is_pandas_dataframe(native_tbl):
        import pandas as pd

        native_tbl["pb_is_good_2"] = pd.Series(is_valid_list, index=native_tbl.index)
    else:
        raise NotImplementedError(f"Backend type not supported: {type(native_tbl)}")

    result_tbl = nw.from_native(native_tbl)  # Handle NA values and combine validation results
    result_tbl = result_tbl.with_columns(
        pb_is_good_1=nw.col(column).is_null() & na_pass,
    )

    result_tbl = result_tbl.with_columns(
        pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2")
    ).drop("pb_is_good_1", "pb_is_good_2")

    return result_tbl.to_native()


def interrogate_within_spec_db(
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


def interrogate_credit_card_db(
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


def interrogate_increasing(
    tbl: IntoFrame, column: str, allow_stationary: bool, decreasing_tol: float, na_pass: bool
) -> Any:
    """
    Increasing interrogation.

    Checks whether column values are increasing row by row.

    Parameters
    ----------
    tbl
        The table to interrogate.
    column
        The column to check.
    allow_stationary
        Whether to allow consecutive equal values (stationary phases).
    decreasing_tol
        Optional tolerance for negative movement (decreasing values).
    na_pass
        Whether NA/null values should be considered as passing.

    Returns
    -------
    Any
        The table with a `pb_is_good_` column indicating pass/fail for each row.
    """
    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))

    # Create a lagged difference column
    result_tbl = nw_tbl.with_columns(pb_lagged_difference_=nw.col(column) - nw.col(column).shift(1))

    # Build the condition based on allow_stationary and decreasing_tol
    if allow_stationary or decreasing_tol != 0:
        # Allow stationary (diff >= 0) or within tolerance
        threshold = -abs(decreasing_tol) if decreasing_tol != 0 else 0
        good_condition = nw.col("pb_lagged_difference_") >= threshold
    else:
        # Strictly increasing (diff > 0)
        good_condition = nw.col("pb_lagged_difference_") > 0

    # Apply the validation logic
    # The logic is:
    # 1. If lagged_diff is null AND current value is NOT null -> pass (first row or after NA)
    # 2. If current value is null -> apply na_pass
    # 3. Otherwise -> apply the good_condition
    result_tbl = result_tbl.with_columns(
        pb_is_good_=nw.when(nw.col("pb_lagged_difference_").is_null() & ~nw.col(column).is_null())
        .then(nw.lit(True))  # First row or row after NA (can't validate)
        .otherwise(
            nw.when(nw.col(column).is_null())
            .then(nw.lit(na_pass))  # Handle NA values in current row
            .otherwise(good_condition)
        )
    )

    return result_tbl.drop("pb_lagged_difference_").to_native()


def interrogate_decreasing(
    tbl: IntoFrame, column: str, allow_stationary: bool, increasing_tol: float, na_pass: bool
) -> Any:
    """
    Decreasing interrogation.

    Checks whether column values are decreasing row by row.

    Parameters
    ----------
    tbl
        The table to interrogate.
    column
        The column to check.
    allow_stationary
        Whether to allow consecutive equal values (stationary phases).
    increasing_tol
        Optional tolerance for positive movement (increasing values).
    na_pass
        Whether NA/null values should be considered as passing.

    Returns
    -------
    Any
        The table with a `pb_is_good_` column indicating pass/fail for each row.
    """
    nw_tbl = nw.from_native(tbl)
    assert isinstance(nw_tbl, (nw.DataFrame, nw.LazyFrame))

    # Create a lagged difference column
    result_tbl = nw_tbl.with_columns(pb_lagged_difference_=nw.col(column) - nw.col(column).shift(1))

    # Build the condition based on allow_stationary and increasing_tol
    if allow_stationary or increasing_tol != 0:
        # Allow stationary (diff <= 0) or within tolerance
        threshold = abs(increasing_tol) if increasing_tol != 0 else 0
        good_condition = nw.col("pb_lagged_difference_") <= threshold
    else:
        # Strictly decreasing (diff < 0)
        good_condition = nw.col("pb_lagged_difference_") < 0

    # Apply the validation logic
    # The logic is:
    # 1. If lagged_diff is null AND current value is NOT null -> pass (first row or after NA)
    # 2. If current value is null -> apply na_pass
    # 3. Otherwise -> apply the good_condition
    result_tbl = result_tbl.with_columns(
        pb_is_good_=nw.when(nw.col("pb_lagged_difference_").is_null() & ~nw.col(column).is_null())
        .then(nw.lit(True))  # First row or row after NA (can't validate)
        .otherwise(
            nw.when(nw.col(column).is_null())
            .then(nw.lit(na_pass))  # Handle NA values in current row
            .otherwise(good_condition)
        )
    )

    return result_tbl.drop("pb_lagged_difference_").to_native()


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
    compare_expr = _safe_modify_datetime_compare_val(nw_tbl, column, compare_expr)

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

    result_tbl = nw_tbl.with_columns(
        pb_is_good_1=_safe_is_nan_or_null_expr(nw_tbl, nw.col(column), column) & na_pass,
        pb_is_good_2=(
            _safe_is_nan_or_null_expr(nw_tbl, nw.col(compare.name), compare.name) & na_pass
            if isinstance(compare, Column)
            else nw.lit(False)
        ),
        pb_is_good_3=comparison & ~_safe_is_nan_or_null_expr(nw_tbl, nw.col(column), column),
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

    return result_tbl.to_native()


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
        columns_subset = tbl.columns

    # Create a count of duplicates using group_by approach
    # Group by the columns of interest and count occurrences
    # Handle DataFrame and LazyFrame separately for proper type narrowing
    if is_narwhals_dataframe(tbl):
        count_tbl = tbl.group_by(columns_subset).agg(nw.len().alias("pb_count_"))
        result = tbl.join(count_tbl, on=columns_subset, how="left")
        result = result.with_columns(pb_is_good_=nw.col("pb_count_") == 1).drop("pb_count_")
        return result.to_native()
    elif is_narwhals_lazyframe(tbl):
        count_tbl = tbl.group_by(columns_subset).agg(nw.len().alias("pb_count_"))
        result = tbl.join(count_tbl, on=columns_subset, how="left")
        result = result.with_columns(pb_is_good_=nw.col("pb_count_") == 1).drop("pb_count_")
        return result.to_native()
    else:
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


def interrogate_prompt(
    tbl: IntoFrame, columns_subset: list[str] | None, ai_config: dict[str, Any]
) -> Any:
    """AI-powered interrogation of rows."""
    import logging

    logger = logging.getLogger(__name__)

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

    try:
        # Import AI validation modules
        from pointblank._utils_ai import (
            _AIValidationEngine,
            _BatchConfig,
            _DataBatcher,
            _LLMConfig,
            _PromptBuilder,
            _ValidationResponseParser,
        )

        # Extract AI configuration
        prompt = ai_config["prompt"]
        llm_provider = ai_config["llm_provider"]
        llm_model = ai_config["llm_model"]
        batch_size = ai_config.get("batch_size", 1000)
        max_concurrent = ai_config.get("max_concurrent", 3)

        # Set up LLM configuration (api_key will be loaded from environment)
        llm_config = _LLMConfig(
            provider=llm_provider,
            model=llm_model,
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
        prompt_builder = _PromptBuilder(prompt)

        # Create AI validation engine
        engine = _AIValidationEngine(llm_config)

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

        # Create a boolean column for validation results
        validation_results = []
        for i in range(total_rows):
            # Default to False if row wasn't processed
            result = combined_results.get(i, False)
            validation_results.append(result)

            # Debug: Log first few conversions
            if i < 5 or total_rows - i <= 2:
                logger.debug(f"   Row {i}: {result} (from combined_results.get({i}, False))")

        logger.debug(f"   - Final validation_results length: {len(validation_results)}")
        logger.debug(f"   - Final passed count: {sum(validation_results)}")
        logger.debug(
            f"   - Final failed count: {len(validation_results) - sum(validation_results)}"
        )

        # Add the pb_is_good_ column by creating a proper boolean Series
        # First convert to native to work with the underlying data frame
        native_tbl = nw_tbl.to_native()

        # Create the result table with the boolean column
        if hasattr(native_tbl, "with_columns"):  # Polars
            import polars as pl

            result_tbl = native_tbl.with_columns(pb_is_good_=pl.Series(validation_results))

        elif hasattr(native_tbl, "assign"):  # Pandas
            import pandas as pd

            result_tbl = native_tbl.assign(pb_is_good_=pd.Series(validation_results, dtype=bool))

        else:
            # Generic fallback
            result_tbl = native_tbl.copy() if hasattr(native_tbl, "copy") else native_tbl
            result_tbl["pb_is_good_"] = validation_results

        logger.info(
            f"AI validation completed. {sum(validation_results)} rows passed out of {len(validation_results)}"
        )

        return result_tbl

    except ImportError as e:
        logger.error(f"Missing dependencies for AI validation: {e}")
        logger.error("Install required packages: pip install openai anthropic aiohttp")

        # Return all False results as fallback (nw_tbl and total_rows defined at function start)
        native_tbl = nw_tbl.to_native()
        validation_results = [False] * total_rows

        if hasattr(native_tbl, "with_columns"):  # Polars
            import polars as pl

            result_tbl = native_tbl.with_columns(pb_is_good_=pl.Series(validation_results))

        elif hasattr(native_tbl, "assign"):  # Pandas
            import pandas as pd

            result_tbl = native_tbl.assign(pb_is_good_=pd.Series(validation_results, dtype=bool))

        else:
            # Fallback
            result_tbl = native_tbl.copy() if hasattr(native_tbl, "copy") else native_tbl
            result_tbl["pb_is_good_"] = validation_results

        return result_tbl

    except Exception as e:
        logger.error(f"AI validation failed: {e}")

        # Return all False results as fallback (nw_tbl and total_rows defined at function start)
        native_tbl = nw_tbl.to_native()
        validation_results = [False] * total_rows

        if hasattr(native_tbl, "with_columns"):  # Polars
            import polars as pl

            result_tbl = native_tbl.with_columns(pb_is_good_=pl.Series(validation_results))

        elif hasattr(native_tbl, "assign"):  # Pandas
            import pandas as pd

            result_tbl = native_tbl.assign(pb_is_good_=pd.Series(validation_results, dtype=bool))

        else:
            # Fallback
            result_tbl = native_tbl.copy() if hasattr(native_tbl, "copy") else native_tbl
            result_tbl["pb_is_good_"] = validation_results

        return result_tbl
