from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import narwhals as nw
from narwhals.dependencies import is_pandas_dataframe, is_polars_dataframe
from narwhals.typing import FrameT

from pointblank._constants import IBIS_BACKENDS
from pointblank._typing import AbsoluteBounds
from pointblank._utils import (
    _column_test_prep,
    _convert_to_narwhals,
    _get_tbl_type,
)
from pointblank.column import Column


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


def _safe_is_nan_or_null_expr(data_frame: Any, column_expr: Any, column_name: str = None) -> Any:
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
    except Exception:
        pass

    # For non-Ibis backends, try to use `is_nan()` if the column type supports it
    try:
        if hasattr(data_frame, "collect_schema"):
            schema = data_frame.collect_schema()
        elif hasattr(data_frame, "schema"):
            schema = data_frame.schema
        else:
            schema = None

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
    except Exception:
        pass

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
                    )

            except Exception as e:
                try:
                    # Try as a ColumnExpression (for pb.expr_col style)
                    col_expr = expr_fn(None)

                    if hasattr(col_expr, "to_pyspark_expr"):
                        # Convert to PySpark expression
                        pyspark_expr = col_expr.to_pyspark_expr(self.data_tbl)
                        pyspark_columns.append(pyspark_expr)
                    else:
                        raise TypeError(f"Cannot convert {type(col_expr)} to PySpark Column")
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

    def get_test_results(self) -> any | list[bool]:
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

    df: FrameT
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
            if hasattr(dfn, "collect"):
                dfn = dfn.collect()

            return len(dfn)

        if tbl_type in IBIS_BACKENDS:
            # Get the count of test units and convert to a native format
            # TODO: check whether pandas or polars is available
            return self.df.count().to_polars()


def _get_compare_expr_nw(compare: Any) -> Any:
    if isinstance(compare, Column):
        if not isinstance(compare.exprs, str):
            raise ValueError("The column expression must be a string.")  # pragma: no cover
        return nw.col(compare.exprs)
    return compare


def _column_has_null_values(table: FrameT, column: str) -> bool:
    try:
        # Try the standard null_count() method
        null_count = (table.select(column).null_count())[column][0]
    except AttributeError:
        # For LazyFrames, collect first then get null count
        try:
            collected = table.select(column).collect()
            null_count = (collected.null_count())[column][0]
        except Exception:
            # Fallback: check if any values are null
            try:
                result = table.select(nw.col(column).is_null().sum().alias("null_count")).collect()
                null_count = result["null_count"][0]
            except Exception:
                # Last resort: return False (assume no nulls)
                return False

    if null_count is None or null_count == 0:
        return False

    return True


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


def _modify_datetime_compare_val(tgt_column: any, compare_val: any) -> any:
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


def col_vals_expr(data_tbl: FrameT, expr, tbl_type: str = "local"):
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
    return data_tbl


def rows_complete(data_tbl: FrameT, columns_subset: list[str] | None):
    """
    Check if rows in a DataFrame are complete (no null values).

    This function replaces the RowsComplete dataclass for direct usage.
    """
    tbl = _convert_to_narwhals(df=data_tbl)

    return interrogate_rows_complete(
        tbl=tbl,
        columns_subset=columns_subset,
    )


def col_exists(data_tbl: FrameT, column: str) -> bool:
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
    data_tbl: FrameT,
    schema,
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


def row_count_match(data_tbl: FrameT, count, inverse: bool, abs_tol_bounds) -> bool:
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
    data_tbl: FrameT, column: str, p: float, bound_finder: Callable[[int], AbsoluteBounds]
) -> bool:
    """Check if the pct of null vales are within p given the absolute bounds.

    Args:
        data_tbl (FrameT): Data
        column (str): Column in the data.
        p (float): Percentage of null values out of the total allowed.
        bound_finder (Callable[[int], AbsoluteBounds]): Function that takes a target number of
            null values and returns the absolute bounds.

    Returns:
        bool: _description_
    """
    # TODO: Shouldn't be passing the whole dataframe for things like this.
    # Extract the absolute target to use with the absolute bounds.
    total_rows: int = data_tbl[column].len()
    abs_target: float = round(total_rows * p)
    lower_bound, upper_bound = bound_finder(abs_target)

    n_null: int = nw.from_native(data_tbl).select(nw.col(column).is_null().sum()).item()

    return n_null >= (abs_target - lower_bound) and n_null <= (abs_target + upper_bound)


def col_count_match(data_tbl: FrameT, count, inverse: bool) -> bool:
    """
    Check if DataFrame column count matches expected count.
    """
    from pointblank.validate import get_column_count

    if not inverse:
        return get_column_count(data=data_tbl) == count
    else:
        return get_column_count(data=data_tbl) != count


def conjointly_validation(data_tbl: FrameT, expressions, threshold: int, tbl_type: str = "local"):
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


def interrogate_gt(tbl: FrameT, column: str, compare: any, na_pass: bool) -> FrameT:
    """Greater than interrogation."""
    return _interrogate_comparison_base(tbl, column, compare, na_pass, "gt")


def interrogate_lt(tbl: FrameT, column: str, compare: any, na_pass: bool) -> FrameT:
    """Less than interrogation."""
    return _interrogate_comparison_base(tbl, column, compare, na_pass, "lt")


def interrogate_ge(tbl: FrameT, column: str, compare: any, na_pass: bool) -> FrameT:
    """Greater than or equal interrogation."""
    return _interrogate_comparison_base(tbl, column, compare, na_pass, "ge")


def interrogate_le(tbl: FrameT, column: str, compare: any, na_pass: bool) -> FrameT:
    """Less than or equal interrogation."""
    return _interrogate_comparison_base(tbl, column, compare, na_pass, "le")


def interrogate_eq(tbl: FrameT, column: str, compare: any, na_pass: bool) -> FrameT:
    """Equal interrogation."""

    nw_tbl = nw.from_native(tbl)

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
                    # Handle genuine type incompatibility
                    native_df = result_tbl.to_native()
                    col_dtype = str(native_df[column].dtype)
                    compare_dtype = str(native_df[compare.name].dtype)

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
                    # Get column types for a descriptive error message
                    try:
                        native_df = result_tbl.to_native()
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
                # Get column type for a descriptive error message
                try:
                    native_df = result_tbl.to_native()
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


def interrogate_ne(tbl: FrameT, column: str, compare: any, na_pass: bool) -> FrameT:
    """Not equal interrogation."""

    nw_tbl = nw.from_native(tbl)

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
    tbl: FrameT, column: str, low: any, high: any, inclusive: tuple, na_pass: bool
) -> FrameT:
    """Between interrogation."""

    low_val = _get_compare_expr_nw(compare=low)
    high_val = _get_compare_expr_nw(compare=high)

    nw_tbl = nw.from_native(tbl)
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
    tbl: FrameT, column: str, low: any, high: any, inclusive: tuple, na_pass: bool
) -> FrameT:
    """Outside range interrogation."""

    low_val = _get_compare_expr_nw(compare=low)
    high_val = _get_compare_expr_nw(compare=high)

    nw_tbl = nw.from_native(tbl)
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

    # Note: Logic is inverted for "outside" - when inclusive[0] is True,
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


def interrogate_isin(tbl: FrameT, column: str, set_values: any) -> FrameT:
    """In set interrogation."""

    nw_tbl = nw.from_native(tbl)

    can_be_null: bool = None in set_values
    base_expr: nw.Expr = nw.col(column).is_in(set_values)
    if can_be_null:
        base_expr = base_expr | nw.col(column).is_null()

    result_tbl = nw_tbl.with_columns(pb_is_good_=base_expr)
    return result_tbl.to_native()


def interrogate_notin(tbl: FrameT, column: str, set_values: any) -> FrameT:
    """Not in set interrogation."""

    nw_tbl = nw.from_native(tbl)
    result_tbl = nw_tbl.with_columns(
        pb_is_good_=nw.col(column).is_in(set_values),
    ).with_columns(pb_is_good_=~nw.col("pb_is_good_"))
    return result_tbl.to_native()


def interrogate_regex(tbl: FrameT, column: str, values: dict | str, na_pass: bool) -> FrameT:
    """Regex interrogation."""

    # Handle both old and new formats for backward compatibility
    if isinstance(values, str):
        pattern = values
        inverse = False
    else:
        pattern = values["pattern"]
        inverse = values["inverse"]

    nw_tbl = nw.from_native(tbl)
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


def interrogate_null(tbl: FrameT, column: str) -> FrameT:
    """Null interrogation."""

    nw_tbl = nw.from_native(tbl)
    result_tbl = nw_tbl.with_columns(pb_is_good_=nw.col(column).is_null())
    return result_tbl.to_native()


def interrogate_not_null(tbl: FrameT, column: str) -> FrameT:
    """Not null interrogation."""

    nw_tbl = nw.from_native(tbl)
    result_tbl = nw_tbl.with_columns(pb_is_good_=~nw.col(column).is_null())
    return result_tbl.to_native()


def _interrogate_comparison_base(
    tbl: FrameT, column: str, compare: any, na_pass: bool, operator: str
) -> FrameT:
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
    FrameT
        The result table with `pb_is_good_` column indicating the passing test units.
    """

    compare_expr = _get_compare_expr_nw(compare=compare)

    nw_tbl = nw.from_native(tbl)
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


def interrogate_rows_distinct(data_tbl: FrameT, columns_subset: list[str] | None) -> FrameT:
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
    FrameT
        A DataFrame with a `pb_is_good_` column indicating which rows pass the test.
    """
    tbl = nw.from_native(data_tbl)

    # Get the column subset to use for the test
    if columns_subset is None:
        columns_subset = tbl.columns

    # Create a count of duplicates using group_by approach
    # Group by the columns of interest and count occurrences
    count_tbl = tbl.group_by(columns_subset).agg(nw.len().alias("pb_count_"))

    # Join back to original table to get count for each row
    tbl = tbl.join(count_tbl, on=columns_subset, how="left")

    # Passing rows will have the value `1` (no duplicates, so True), otherwise False applies
    tbl = tbl.with_columns(pb_is_good_=nw.col("pb_count_") == 1).drop("pb_count_")

    return tbl.to_native()


def interrogate_rows_complete(tbl: FrameT, columns_subset: list[str] | None) -> FrameT:
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


def interrogate_prompt(tbl: FrameT, columns_subset: list[str] | None, ai_config: dict) -> FrameT:
    """AI-powered interrogation of rows."""
    import logging

    logger = logging.getLogger(__name__)

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
        parser = _ValidationResponseParser(total_rows=len(tbl))
        combined_results = parser.combine_batch_results(batch_results, signature_mapping)

        # Debug: Log table info and combined results
        logger.debug("🏁 Final result conversion:")
        logger.debug(f"   - Table length: {len(tbl)}")
        logger.debug(
            f"   - Combined results keys: {sorted(combined_results.keys()) if combined_results else 'None'}"
        )

        # Convert results to narwhals format
        nw_tbl = nw.from_native(tbl)

        # Create a boolean column for validation results
        validation_results = []
        for i in range(len(tbl)):
            # Default to False if row wasn't processed
            result = combined_results.get(i, False)
            validation_results.append(result)

            # Debug: Log first few conversions
            if i < 5 or len(tbl) - i <= 2:
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

        # Return all False results as fallback
        nw_tbl = nw.from_native(tbl)
        native_tbl = nw_tbl.to_native()
        validation_results = [False] * len(tbl)

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

        # Return all False results as fallback
        nw_tbl = nw.from_native(tbl)
        native_tbl = nw_tbl.to_native()
        validation_results = [False] * len(tbl)

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
