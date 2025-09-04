from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import narwhals as nw
from narwhals.dependencies import is_pandas_dataframe, is_polars_dataframe
from narwhals.typing import FrameT

from pointblank._constants import IBIS_BACKENDS
from pointblank._utils import (
    _column_subset_test_prep,
    _column_test_prep,
    _convert_to_narwhals,
    _get_tbl_type,
)
from pointblank.column import Column
from pointblank.schema import Schema
from pointblank.thresholds import _threshold_check

if TYPE_CHECKING:
    from pointblank._typing import AbsoluteTolBounds


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


@dataclass
class Interrogator:
    """
    Compare values against a single value, a set of values, or a range of values.

    Parameters
    ----------
    x
        The values to compare.
    column
        The column to check.
    columns_subset
        The subset of columns to use for the check.
    compare
        The value to compare against. Used in the following interrogations:
        - 'gt' for greater than
        - 'lt' for less than
        - 'eq' for equal to
        - 'ne' for not equal to
        - 'ge' for greater than or equal to
        - 'le' for less than or equal to
    set
        The set of values to compare against. Used in the following interrogations:
        - 'isin' for values in the set
        - 'notin' for values not in the set
    pattern
        The regular expression pattern to compare against. Used in the following:
        - 'regex' for values that match the pattern
    low
        The lower bound of the range of values to compare against. Used in the following:
        - 'between' for values between the range
        - 'outside' for values outside the range
    high
        The upper bound of the range of values to compare against. Used in the following:
        - 'between' for values between the range
        - 'outside' for values outside the range
    inclusive
        A tuple of booleans that state which bounds are inclusive. The position of the boolean
        corresponds to the value in the following order: (low, high). Used in the following:
        - 'between' for values between the range
        - 'outside' for values outside the range
    na_pass
        `True` to pass test units with missing values, `False` otherwise.
    tbl_type
        The type of table to use for the assertion. This is used to determine the backend for the
        assertion. The default is 'local' but it can also be any of the table types in the
        `IBIS_BACKENDS` constant.

    Returns
    -------
    list[bool]
        A list of booleans where `True` indicates a passing test unit.
    """

    x: nw.DataFrame | Any
    column: str = None
    columns_subset: list[str] = None
    compare: float | int | list[float | int] = None
    set: list[float | int] = None
    pattern: str = None
    low: float | int | list[float | int] = None
    high: float | int | list[float | int] = None
    inclusive: tuple[bool, bool] = None
    na_pass: bool = False
    tbl_type: str = "local"

    def __post_init__(self):
        """
        Post-initialization to process Ibis tables through Narwhals.

        This converts Ibis tables to Narwhals-wrapped tables to unify
        the processing pathway and reduce code branching.
        """
        # Import the processing function
        from pointblank._utils import _process_ibis_through_narwhals

        # Process Ibis tables through Narwhals
        self.x, self.tbl_type = _process_ibis_through_narwhals(self.x, self.tbl_type)

    def _comparison_base(self, operator: str) -> FrameT | Any:
        """
        Unified base method for comparison operations (gt, ge, lt, le).

        Parameters
        ----------
        operator
            The comparison operator: 'gt', 'ge', 'lt', or 'le'.

        Returns
        -------
        FrameT | Any
            The result table with `pb_is_good_` column indicating the passing test units.
        """
        compare_expr = _get_compare_expr_nw(compare=self.compare)
        compare_expr = _safe_modify_datetime_compare_val(self.x, self.column, compare_expr)

        # Create the comparison expression based on the operator
        column_expr = nw.col(self.column)
        if operator == "gt":
            comparison = column_expr > compare_expr
        elif operator == "ge":
            comparison = column_expr >= compare_expr
        elif operator == "lt":
            comparison = column_expr < compare_expr
        elif operator == "le":
            comparison = column_expr <= compare_expr
        else:
            raise ValueError(  # pragma: no cover
                f"Invalid operator: {operator}. Must be one of: 'gt', 'ge', 'lt', 'le'"
            )

        return (
            self.x.with_columns(
                pb_is_good_1=_safe_is_nan_or_null_expr(self.x, nw.col(self.column), self.column)
                & self.na_pass,
                pb_is_good_2=(
                    _safe_is_nan_or_null_expr(self.x, nw.col(self.compare.name), self.compare.name)
                    & self.na_pass
                    if isinstance(self.compare, Column)
                    else nw.lit(False)
                ),
                pb_is_good_3=comparison
                & ~_safe_is_nan_or_null_expr(self.x, nw.col(self.column), self.column),
            )
            .with_columns(
                pb_is_good_3=(
                    nw.when(nw.col("pb_is_good_3").is_null())
                    .then(nw.lit(False))
                    .otherwise(nw.col("pb_is_good_3"))
                )
            )
            .with_columns(
                pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3")
            )
            .drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")
            .to_native()
        )

    def gt(self) -> FrameT | Any:
        return self._comparison_base("gt")

    def lt(self) -> FrameT | Any:
        return self._comparison_base("lt")

    def ge(self) -> FrameT | Any:
        return self._comparison_base("ge")

    def le(self) -> FrameT | Any:
        return self._comparison_base("le")

    def eq(self) -> FrameT | Any:
        # All backends now use Narwhals (including former Ibis tables) ---------

        if isinstance(self.compare, Column):
            compare_expr = _get_compare_expr_nw(compare=self.compare)

            tbl = self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=(
                    nw.col(self.compare.name).is_null() & self.na_pass
                    if isinstance(self.compare, Column)
                    else nw.lit(False)
                ),
            )

            tbl = tbl.with_columns(
                pb_is_good_3=(~nw.col(self.compare.name).is_null() & ~nw.col(self.column).is_null())
            )

            if is_pandas_dataframe(tbl.to_native()):
                tbl = tbl.with_columns(
                    pb_is_good_4=nw.col(self.column) - compare_expr,
                )

                tbl = tbl.with_columns(
                    pb_is_good_=nw.col("pb_is_good_1")
                    | nw.col("pb_is_good_2")
                    | (nw.col("pb_is_good_4") == 0 & ~nw.col("pb_is_good_3").is_null())
                )

            else:
                tbl = tbl.with_columns(
                    pb_is_good_4=nw.col(self.column) == compare_expr,
                )

                tbl = tbl.with_columns(
                    pb_is_good_=nw.col("pb_is_good_1")
                    | nw.col("pb_is_good_2")
                    | (nw.col("pb_is_good_4") & ~nw.col("pb_is_good_1") & ~nw.col("pb_is_good_2"))
                )

            return tbl.drop(
                "pb_is_good_1", "pb_is_good_2", "pb_is_good_3", "pb_is_good_4"
            ).to_native()

        else:
            compare_expr = _get_compare_expr_nw(compare=self.compare)

            compare_expr = _safe_modify_datetime_compare_val(self.x, self.column, compare_expr)

            tbl = self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=(
                    nw.col(self.compare.name).is_null() & self.na_pass
                    if isinstance(self.compare, Column)
                    else nw.lit(False)
                ),
            )

            tbl = tbl.with_columns(pb_is_good_3=nw.col(self.column) == compare_expr)

            tbl = tbl.with_columns(
                pb_is_good_3=(
                    nw.when(nw.col("pb_is_good_3").is_null())
                    .then(nw.lit(False))
                    .otherwise(nw.col("pb_is_good_3"))
                )
            )

            tbl = tbl.with_columns(
                pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3")
            )

            return tbl.drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3").to_native()

    def ne(self) -> FrameT | Any:
        # All backends now use Narwhals (including former Ibis tables) ---------

        # Determine if the reference and comparison columns have any null values
        ref_col_has_null_vals = _column_has_null_values(table=self.x, column=self.column)

        if isinstance(self.compare, Column):
            compare_name = self.compare.name if isinstance(self.compare, Column) else self.compare
            cmp_col_has_null_vals = _column_has_null_values(table=self.x, column=compare_name)
        else:
            cmp_col_has_null_vals = False

        # If neither column has null values, we can proceed with the comparison
        # without too many complications
        if not ref_col_has_null_vals and not cmp_col_has_null_vals:
            if isinstance(self.compare, Column):
                compare_expr = _get_compare_expr_nw(compare=self.compare)

                return self.x.with_columns(
                    pb_is_good_=nw.col(self.column) != compare_expr,
                ).to_native()

            else:
                compare_expr = _safe_modify_datetime_compare_val(self.x, self.column, self.compare)

                return self.x.with_columns(
                    pb_is_good_=nw.col(self.column) != nw.lit(compare_expr),
                ).to_native()

        # If either column has null values, we need to handle the comparison
        # much more carefully since we can't inadverdently compare null values
        # to non-null values

        if isinstance(self.compare, Column):
            compare_expr = _get_compare_expr_nw(compare=self.compare)

            # CASE 1: the reference column has null values but the comparison column does not
            if ref_col_has_null_vals and not cmp_col_has_null_vals:
                if is_pandas_dataframe(self.x.to_native()):
                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column).is_null(),
                        pb_is_good_2=nw.lit(self.column) != nw.col(self.compare.name),
                    )

                else:
                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column).is_null(),
                        pb_is_good_2=nw.col(self.column) != nw.col(self.compare.name),
                    )

                if not self.na_pass:
                    tbl = tbl.with_columns(
                        pb_is_good_2=nw.col("pb_is_good_2") & ~nw.col("pb_is_good_1")
                    )

                if is_polars_dataframe(self.x.to_native()):
                    # There may be Null values in the pb_is_good_2 column, change those to
                    # True if na_pass is True, False otherwise

                    tbl = tbl.with_columns(
                        pb_is_good_2=nw.when(nw.col("pb_is_good_2").is_null())
                        .then(False)
                        .otherwise(nw.col("pb_is_good_2")),
                    )

                    if self.na_pass:
                        tbl = tbl.with_columns(
                            pb_is_good_2=(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                        )
                else:
                    # General case (non-Polars): handle na_pass=True properly
                    if self.na_pass:
                        tbl = tbl.with_columns(
                            pb_is_good_2=(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                        )

                return (
                    tbl.with_columns(pb_is_good_=nw.col("pb_is_good_2"))
                    .drop("pb_is_good_1", "pb_is_good_2")
                    .to_native()
                )

            # CASE 2: the comparison column has null values but the reference column does not
            elif not ref_col_has_null_vals and cmp_col_has_null_vals:
                if is_pandas_dataframe(self.x.to_native()):
                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column) != nw.lit(self.compare.name),
                        pb_is_good_2=nw.col(self.compare.name).is_null(),
                    )

                else:
                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column) != nw.col(self.compare.name),
                        pb_is_good_2=nw.col(self.compare.name).is_null(),
                    )

                if not self.na_pass:
                    tbl = tbl.with_columns(
                        pb_is_good_1=nw.col("pb_is_good_1") & ~nw.col("pb_is_good_2")
                    )

                if is_polars_dataframe(self.x.to_native()):
                    if self.na_pass:
                        tbl = tbl.with_columns(
                            pb_is_good_1=(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                        )
                else:
                    # General case (non-Polars): handle na_pass=True properly
                    if self.na_pass:
                        tbl = tbl.with_columns(
                            pb_is_good_1=(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                        )

                return (
                    tbl.with_columns(pb_is_good_=nw.col("pb_is_good_1"))
                    .drop("pb_is_good_1", "pb_is_good_2")
                    .to_native()
                )

            # CASE 3: both columns have null values and there may potentially be cases where
            # there could even be null/null comparisons
            elif ref_col_has_null_vals and cmp_col_has_null_vals:
                tbl = self.x.with_columns(
                    pb_is_good_1=nw.col(self.column).is_null(),
                    pb_is_good_2=nw.col(self.compare.name).is_null(),
                    pb_is_good_3=nw.col(self.column) != nw.col(self.compare.name),
                )

                if not self.na_pass:
                    tbl = tbl.with_columns(
                        pb_is_good_3=nw.col("pb_is_good_3")
                        & ~nw.col("pb_is_good_1")
                        & ~nw.col("pb_is_good_2")
                    )

                if is_polars_dataframe(self.x.to_native()):
                    if self.na_pass:
                        tbl = tbl.with_columns(
                            pb_is_good_3=(
                                nw.when(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                                .then(True)
                                .otherwise(False)
                            )
                        )
                else:
                    # General case (non-Polars): handle na_pass=True properly
                    if self.na_pass:
                        tbl = tbl.with_columns(
                            pb_is_good_3=(
                                nw.when(nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
                                .then(True)
                                .otherwise(nw.col("pb_is_good_3"))
                            )
                        )

                return (
                    tbl.with_columns(pb_is_good_=nw.col("pb_is_good_3"))
                    .drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3")
                    .to_native()
                )

        else:
            # Case where the reference column contains null values
            if ref_col_has_null_vals:
                # Create individual cases for Pandas and Polars

                compare_expr = _safe_modify_datetime_compare_val(self.x, self.column, self.compare)

                if is_pandas_dataframe(self.x.to_native()):
                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column).is_null(),
                        pb_is_good_2=nw.lit(self.column) != nw.lit(compare_expr),
                    )

                    if not self.na_pass:
                        tbl = tbl.with_columns(
                            pb_is_good_2=nw.col("pb_is_good_2") & ~nw.col("pb_is_good_1")
                        )

                    return (
                        tbl.with_columns(pb_is_good_=nw.col("pb_is_good_2"))
                        .drop("pb_is_good_1", "pb_is_good_2")
                        .to_native()
                    )

                elif is_polars_dataframe(self.x.to_native()):
                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column).is_null(),  # val is Null in Column
                        pb_is_good_2=nw.lit(self.na_pass),  # Pass if any Null in val or compare
                    )

                    tbl = tbl.with_columns(pb_is_good_3=nw.col(self.column) != nw.lit(compare_expr))

                    tbl = tbl.with_columns(
                        pb_is_good_=(
                            (nw.col("pb_is_good_1") & nw.col("pb_is_good_2"))
                            | (nw.col("pb_is_good_3") & ~nw.col("pb_is_good_1"))
                        )
                    )

                    tbl = tbl.drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3").to_native()

                    return tbl

                else:
                    # Generic case for other DataFrame types (PySpark, etc.)
                    # Use similar logic to Polars but handle potential differences
                    tbl = self.x.with_columns(
                        pb_is_good_1=nw.col(self.column).is_null(),  # val is Null in Column
                        pb_is_good_2=nw.lit(self.na_pass),  # Pass if any Null in val or compare
                    )

                    tbl = tbl.with_columns(pb_is_good_3=nw.col(self.column) != nw.lit(compare_expr))

                    tbl = tbl.with_columns(
                        pb_is_good_=(
                            (nw.col("pb_is_good_1") & nw.col("pb_is_good_2"))
                            | (nw.col("pb_is_good_3") & ~nw.col("pb_is_good_1"))
                        )
                    )

                    return tbl.drop("pb_is_good_1", "pb_is_good_2", "pb_is_good_3").to_native()

    def between(self) -> FrameT | Any:
        # All backends now use Narwhals (including former Ibis tables) ---------

        low_val = _get_compare_expr_nw(compare=self.low)
        high_val = _get_compare_expr_nw(compare=self.high)

        low_val = _safe_modify_datetime_compare_val(self.x, self.column, low_val)
        high_val = _safe_modify_datetime_compare_val(self.x, self.column, high_val)

        tbl = self.x.with_columns(
            pb_is_good_1=nw.col(self.column).is_null(),  # val is Null in Column
            pb_is_good_2=(  # lb is Null in Column
                nw.col(self.low.name).is_null() if isinstance(self.low, Column) else nw.lit(False)
            ),
            pb_is_good_3=(  # ub is Null in Column
                nw.col(self.high.name).is_null() if isinstance(self.high, Column) else nw.lit(False)
            ),
            pb_is_good_4=nw.lit(self.na_pass),  # Pass if any Null in lb, val, or ub
        )

        if self.inclusive[0]:
            tbl = tbl.with_columns(pb_is_good_5=nw.col(self.column) >= low_val)
        else:
            tbl = tbl.with_columns(pb_is_good_5=nw.col(self.column) > low_val)

        if self.inclusive[1]:
            tbl = tbl.with_columns(pb_is_good_6=nw.col(self.column) <= high_val)
        else:
            tbl = tbl.with_columns(pb_is_good_6=nw.col(self.column) < high_val)

        tbl = tbl.with_columns(
            pb_is_good_5=(
                nw.when(nw.col("pb_is_good_5").is_null())
                .then(nw.lit(False))
                .otherwise(nw.col("pb_is_good_5"))
            )
        )

        tbl = tbl.with_columns(
            pb_is_good_6=(
                nw.when(nw.col("pb_is_good_6").is_null())
                .then(nw.lit(False))
                .otherwise(nw.col("pb_is_good_6"))
            )
        )

        tbl = (
            tbl.with_columns(
                pb_is_good_=(
                    (
                        (nw.col("pb_is_good_1") | nw.col("pb_is_good_2") | nw.col("pb_is_good_3"))
                        & nw.col("pb_is_good_4")
                    )
                    | (nw.col("pb_is_good_5") & nw.col("pb_is_good_6"))
                )
            )
            .drop(
                "pb_is_good_1",
                "pb_is_good_2",
                "pb_is_good_3",
                "pb_is_good_4",
                "pb_is_good_5",
                "pb_is_good_6",
            )
            .to_native()
        )

        return tbl

    def outside(self) -> FrameT | Any:
        # All backends now use Narwhals (including former Ibis tables) ---------

        low_val = _get_compare_expr_nw(compare=self.low)
        high_val = _get_compare_expr_nw(compare=self.high)

        low_val = _get_compare_expr_nw(compare=self.low)
        high_val = _get_compare_expr_nw(compare=self.high)

        low_val = _safe_modify_datetime_compare_val(self.x, self.column, low_val)
        high_val = _safe_modify_datetime_compare_val(self.x, self.column, high_val)

        tbl = self.x.with_columns(
            pb_is_good_1=nw.col(self.column).is_null(),  # val is Null in Column
            pb_is_good_2=(  # lb is Null in Column
                nw.col(self.low.name).is_null() if isinstance(self.low, Column) else nw.lit(False)
            ),
            pb_is_good_3=(  # ub is Null in Column
                nw.col(self.high.name).is_null() if isinstance(self.high, Column) else nw.lit(False)
            ),
            pb_is_good_4=nw.lit(self.na_pass),  # Pass if any Null in lb, val, or ub
        )

        if self.inclusive[0]:
            tbl = tbl.with_columns(pb_is_good_5=nw.col(self.column) < low_val)
        else:
            tbl = tbl.with_columns(pb_is_good_5=nw.col(self.column) <= low_val)

        if self.inclusive[1]:
            tbl = tbl.with_columns(pb_is_good_6=nw.col(self.column) > high_val)
        else:
            tbl = tbl.with_columns(pb_is_good_6=nw.col(self.column) >= high_val)

        tbl = tbl.with_columns(
            pb_is_good_5=nw.when(nw.col("pb_is_good_5").is_null())
            .then(False)
            .otherwise(nw.col("pb_is_good_5")),
            pb_is_good_6=nw.when(nw.col("pb_is_good_6").is_null())
            .then(False)
            .otherwise(nw.col("pb_is_good_6")),
        )

        tbl = (
            tbl.with_columns(
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
            )
            .drop(
                "pb_is_good_1",
                "pb_is_good_2",
                "pb_is_good_3",
                "pb_is_good_4",
                "pb_is_good_5",
                "pb_is_good_6",
            )
            .to_native()
        )

        return tbl

    def isin(self) -> FrameT | Any:
        # All backends now use Narwhals (including former Ibis tables) ---------

        can_be_null: bool = None in self.set

        base_expr: nw.Expr = nw.col(self.column).is_in(self.set)
        if can_be_null:
            base_expr = base_expr | nw.col(self.column).is_null()

        return self.x.with_columns(pb_is_good_=base_expr).to_native()

    def notin(self) -> FrameT | Any:
        # All backends now use Narwhals (including former Ibis tables) ---------

        return (
            self.x.with_columns(
                pb_is_good_=nw.col(self.column).is_in(self.set),
            )
            .with_columns(pb_is_good_=~nw.col("pb_is_good_"))
            .to_native()
        )

    def regex(self) -> FrameT | Any:
        # All backends now use Narwhals (including former Ibis tables) ---------

        return (
            self.x.with_columns(
                pb_is_good_1=nw.col(self.column).is_null() & self.na_pass,
                pb_is_good_2=nw.when(~nw.col(self.column).is_null())
                .then(nw.col(self.column).str.contains(pattern=self.pattern))
                .otherwise(False),
            )
            .with_columns(pb_is_good_=nw.col("pb_is_good_1") | nw.col("pb_is_good_2"))
            .drop("pb_is_good_1", "pb_is_good_2")
            .to_native()
        )

    def null(self) -> FrameT | Any:
        # All backends now use Narwhals (including former Ibis tables) ---------

        return self.x.with_columns(
            pb_is_good_=nw.col(self.column).is_null(),
        ).to_native()

    def not_null(self) -> FrameT | Any:
        # All backends now use Narwhals (including former Ibis tables) ---------

        return self.x.with_columns(
            pb_is_good_=~nw.col(self.column).is_null(),
        ).to_native()

    def rows_distinct(self) -> FrameT | Any:
        # All backends now use Narwhals (including former Ibis tables) ---------

        tbl = self.x

        # Get the column subset to use for the test
        if self.columns_subset is None:
            columns_subset = tbl.columns
        else:
            columns_subset = self.columns_subset

        # Create a count of duplicates using group_by approach like Ibis backend
        # Group by the columns of interest and count occurrences
        count_tbl = tbl.group_by(columns_subset).agg(nw.len().alias("pb_count_"))

        # Join back to original table to get count for each row
        tbl = tbl.join(count_tbl, on=columns_subset, how="left")

        # Passing rows will have the value `1` (no duplicates, so True), otherwise False applies
        tbl = tbl.with_columns(pb_is_good_=nw.col("pb_count_") == 1).drop("pb_count_")

        return tbl.to_native()

    def rows_complete(self) -> FrameT | Any:
        # All backends now use Narwhals (including former Ibis tables) ---------

        tbl = self.x

        # Determine the number of null values in each row (column subsets are handled in
        # the `_check_nulls_across_columns_nw()` function)
        tbl = _check_nulls_across_columns_nw(table=tbl, columns_subset=self.columns_subset)

        # Failing rows will have the value `True` in the generated column, so we need to negate
        # the result to get the passing rows
        tbl = tbl.with_columns(pb_is_good_=~nw.col("_any_is_null_"))
        tbl = tbl.drop("_any_is_null_")

        # Convert the table to a native format
        return tbl.to_native()


@dataclass
class ColValsCompareOne:
    """
    Compare values in a table column against a single value.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    value
        A value to check against.
    na_pass
        `True` to pass test units with missing values, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    assertion_method
        The type of assertion ('gt' for greater than, 'lt' for less than).
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    value: float | int
    na_pass: bool
    threshold: int
    assertion_method: str
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):
        if self.tbl_type == "local":
            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )
        else:
            # For remote backends (Ibis), pass the table as is since Interrogator now handles Ibis through Narwhals
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.assertion_method == "gt":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).gt()
        elif self.assertion_method == "lt":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).lt()
        elif self.assertion_method == "eq":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).eq()
        elif self.assertion_method == "ne":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).ne()
        elif self.assertion_method == "ge":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).ge()
        elif self.assertion_method == "le":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).le()
        elif self.assertion_method == "null":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                tbl_type=self.tbl_type,
            ).null()
        elif self.assertion_method == "not_null":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                compare=self.value,
                tbl_type=self.tbl_type,
            ).not_null()
        else:
            raise ValueError(
                """Invalid comparison type. Use:
                - `gt` for greater than,
                - `lt` for less than,
                - `eq` for equal to,
                - `ne` for not equal to,
                - `ge` for greater than or equal to,
                - `le` for less than or equal to,
                - `null` for null values, or
                - `not_null` for not null values.
                """
            )

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
        )


@dataclass
class ColValsCompareTwo:
    """
    General routine to compare values in a column against two values.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    value1
        A value to check against.
    value2
        A value to check against.
    inclusive
        A tuple of booleans that state which bounds are inclusive. The position of the boolean
        corresponds to the value in the following order: (value1, value2).
    na_pass
        `True` to pass test units with missing values, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    assertion_method
        The type of assertion ('between' for between two values and 'outside' for outside two
        values).
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    value1: float | int
    value2: float | int
    inclusive: tuple[bool, bool]
    na_pass: bool
    threshold: int
    assertion_method: str
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):
        if self.tbl_type == "local":
            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        else:
            # For remote backends (Ibis), pass the table as is since Interrogator now handles Ibis through Narwhals
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.assertion_method == "between":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                low=self.value1,
                high=self.value2,
                inclusive=self.inclusive,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).between()
        elif self.assertion_method == "outside":
            self.test_unit_res = Interrogator(
                x=tbl,
                column=self.column,
                low=self.value1,
                high=self.value2,
                inclusive=self.inclusive,
                na_pass=self.na_pass,
                tbl_type=self.tbl_type,
            ).outside()
        else:
            raise ValueError(
                """Invalid assertion type. Use:
                - `between` for values between two values, or
                - `outside` for values outside two values."""
            )

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
        )


@dataclass
class ColValsCompareSet:
    """
    General routine to compare values in a column against a set of values.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    values
        A set of values to check against.
    threshold
        The maximum number of failing test units to allow.
    inside
        `True` to check if the values are inside the set, `False` to check if the values are
        outside the set.
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    values: list[float | int]
    threshold: int
    inside: bool
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):
        if self.tbl_type == "local":
            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )
        else:
            # For remote backends (Ibis), pass the table as is since Interrogator now handles Ibis through Narwhals
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        if self.inside:
            self.test_unit_res = Interrogator(
                x=tbl, column=self.column, set=self.values, tbl_type=self.tbl_type
            ).isin()
        else:
            self.test_unit_res = Interrogator(
                x=tbl, column=self.column, set=self.values, tbl_type=self.tbl_type
            ).notin()

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
        )


@dataclass
class ColValsRegex:
    """
    Check if values in a column match a regular expression pattern.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    pattern
        The regular expression pattern to check against.
    na_pass
        `True` to pass test units with missing values, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    allowed_types
        The allowed data types for the column.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    pattern: str
    na_pass: bool
    threshold: int
    allowed_types: list[str]
    tbl_type: str = "local"

    def __post_init__(self):
        if self.tbl_type == "local":
            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_test_prep(
                df=self.data_tbl, column=self.column, allowed_types=self.allowed_types
            )
        else:
            # For remote backends (Ibis), pass the table as is since Interrogator now handles Ibis through Narwhals
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        self.test_unit_res = Interrogator(
            x=tbl,
            column=self.column,
            pattern=self.pattern,
            na_pass=self.na_pass,
            tbl_type=self.tbl_type,
        ).regex()

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
        )


@dataclass
class ColValsExpr:
    """
    Check if values in a column evaluate to True for a given predicate expression.

    Parameters
    ----------
    data_tbl
        A data table.
    expr
        The expression to check against.
    threshold
        The maximum number of failing test units to allow.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    expr: str
    threshold: int
    tbl_type: str = "local"

    def __post_init__(self):
        if self.tbl_type == "local":
            # Check the type of expression provided
            if "narwhals" in str(type(self.expr)) and "expr" in str(type(self.expr)):
                expression_type = "narwhals"
            elif "polars" in str(type(self.expr)) and "expr" in str(type(self.expr)):
                expression_type = "polars"
            else:
                expression_type = "pandas"

            # Determine whether this is a Pandas or Polars table
            tbl_type = _get_tbl_type(data=self.data_tbl)

            df_lib_name = "polars" if "polars" in tbl_type else "pandas"

            if expression_type == "narwhals":
                tbl_nw = _convert_to_narwhals(df=self.data_tbl)
                tbl_nw = tbl_nw.with_columns(pb_is_good_=self.expr)
                tbl = tbl_nw.to_native()
                self.test_unit_res = tbl

                return self

            if df_lib_name == "polars" and expression_type == "polars":
                self.test_unit_res = self.data_tbl.with_columns(pb_is_good_=self.expr)

            if df_lib_name == "pandas" and expression_type == "pandas":
                self.test_unit_res = self.data_tbl.assign(pb_is_good_=self.expr)

            return self

    def get_test_results(self):
        return self.test_unit_res


@dataclass
class ColExistsHasType:
    """
    Check if a column exists in a DataFrame or has a certain data type.

    Parameters
    ----------
    data_tbl
        A data table.
    column
        The column to check.
    threshold
        The maximum number of failing test units to allow.
    assertion_method
        The type of assertion ('exists' for column existence).
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    column: str
    threshold: int
    assertion_method: str
    tbl_type: str = "local"

    def __post_init__(self):
        if self.tbl_type == "local":
            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _convert_to_narwhals(df=self.data_tbl)
        else:
            # For remote backends (Ibis), pass the table as is since Narwhals can handle it
            tbl = _convert_to_narwhals(df=self.data_tbl)

        if self.assertion_method == "exists":
            res = int(self.column in tbl.columns)

        self.test_unit_res = res

    def get_test_results(self):
        return self.test_unit_res


@dataclass
class RowsDistinct:
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
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    columns_subset: list[str] | None
    threshold: int
    tbl_type: str = "local"

    def __post_init__(self):
        if self.tbl_type == "local":
            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_subset_test_prep(df=self.data_tbl, columns_subset=self.columns_subset)

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        else:
            # For remote backends (Ibis), pass the table as is since Interrogator now handles Ibis through Narwhals
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        self.test_unit_res = Interrogator(
            x=tbl,
            columns_subset=self.columns_subset,
            tbl_type=self.tbl_type,
        ).rows_distinct()

    def get_test_results(self):
        return self.test_unit_res

    def test(self):
        # Get the number of failing test units by counting instances of `False` in the `pb_is_good_`
        # column and then determine if the test passes overall by comparing the number of failing
        # test units to the threshold for failing test units

        results_list = nw.from_native(self.test_unit_res)["pb_is_good_"].to_list()

        return _threshold_check(
            failing_test_units=results_list.count(False), threshold=self.threshold
        )


@dataclass
class RowsComplete:
    """
    Check if rows in a DataFrame are complete.

    Parameters
    ----------
    data_tbl
        A data table.
    columns_subset
        A list of columns to check for completeness.
    threshold
        The maximum number of failing test units to allow.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    columns_subset: list[str] | None
    threshold: int
    tbl_type: str = "local"

    def __post_init__(self):
        if self.tbl_type == "local":
            # Convert the DataFrame to a format that narwhals can work with, and:
            #  - check if the `column=` exists
            #  - check if the `column=` type is compatible with the test
            tbl = _column_subset_test_prep(df=self.data_tbl, columns_subset=self.columns_subset)

        # TODO: For Ibis backends, check if the column exists and if the column type is compatible;
        #       for now, just pass the table as is
        else:
            # For remote backends (Ibis), pass the table as is since Interrogator now handles Ibis through Narwhals
            tbl = self.data_tbl

        # Collect results for the test units; the results are a list of booleans where
        # `True` indicates a passing test unit
        self.test_unit_res = Interrogator(
            x=tbl,
            columns_subset=self.columns_subset,
            tbl_type=self.tbl_type,
        ).rows_complete()

    def get_test_results(self):
        return self.test_unit_res


@dataclass
class ColSchemaMatch:
    """
    Check if a column exists in a DataFrame or has a certain data type.

    Parameters
    ----------
    data_tbl
        A data table.
    schema
        A schema to check against.
    complete
        `True` to check if the schema is complete, `False` otherwise.
    in_order
        `True` to check if the schema is in order, `False` otherwise.
    case_sensitive_colnames
        `True` to perform column-name matching in a case-sensitive manner, `False` otherwise.
    case_sensitive_dtypes
        `True` to perform data-type matching in a case-sensitive manner, `False` otherwise.
    full_match_dtypes
        `True` to perform a full match of data types, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT | Any
    schema: any
    complete: bool
    in_order: bool
    case_sensitive_colnames: bool
    case_sensitive_dtypes: bool
    full_match_dtypes: bool
    threshold: int

    def __post_init__(self):
        schema_expect = self.schema
        schema_actual = Schema(tbl=self.data_tbl)

        if self.complete and self.in_order:
            # Check if the schema is complete and in order (most restrictive check)
            # complete: True, in_order: True
            res = schema_expect._compare_schema_columns_complete_in_order(
                other=schema_actual,
                case_sensitive_colnames=self.case_sensitive_colnames,
                case_sensitive_dtypes=self.case_sensitive_dtypes,
                full_match_dtypes=self.full_match_dtypes,
            )

        elif not self.complete and not self.in_order:
            # Check if the schema is at least a subset, and, order of columns does not matter
            # complete: False, in_order: False
            res = schema_expect._compare_schema_columns_subset_any_order(
                other=schema_actual,
                case_sensitive_colnames=self.case_sensitive_colnames,
                case_sensitive_dtypes=self.case_sensitive_dtypes,
                full_match_dtypes=self.full_match_dtypes,
            )

        elif self.complete:
            # Check if the schema is complete, but the order of columns does not matter
            # complete: True, in_order: False
            res = schema_expect._compare_schema_columns_complete_any_order(
                other=schema_actual,
                case_sensitive_colnames=self.case_sensitive_colnames,
                case_sensitive_dtypes=self.case_sensitive_dtypes,
                full_match_dtypes=self.full_match_dtypes,
            )

        else:
            # Check if the schema is a subset (doesn't need to be complete) and in order
            # complete: False, in_order: True
            res = schema_expect._compare_schema_columns_subset_in_order(
                other=schema_actual,
                case_sensitive_colnames=self.case_sensitive_colnames,
                case_sensitive_dtypes=self.case_sensitive_dtypes,
                full_match_dtypes=self.full_match_dtypes,
            )

        self.test_unit_res = res

    def get_test_results(self):
        return self.test_unit_res


@dataclass
class RowCountMatch:
    """
    Check if rows in a DataFrame either match or don't match a fixed value.

    Parameters
    ----------
    data_tbl
        A data table.
    count
        The fixed row count to check against.
    inverse
        `True` to check if the row count does not match the fixed value, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    count: int
    inverse: bool
    threshold: int
    abs_tol_bounds: AbsoluteTolBounds
    tbl_type: str = "local"

    def __post_init__(self):
        from pointblank.validate import get_row_count

        row_count: int = get_row_count(data=self.data_tbl)

        lower_abs_limit, upper_abs_limit = self.abs_tol_bounds
        min_val: int = self.count - lower_abs_limit
        max_val: int = self.count + upper_abs_limit

        if self.inverse:
            res: bool = not (row_count >= min_val and row_count <= max_val)
        else:
            res: bool = row_count >= min_val and row_count <= max_val

        self.test_unit_res = res

    def get_test_results(self):
        return self.test_unit_res


@dataclass
class ColCountMatch:
    """
    Check if columns in a DataFrame either match or don't match a fixed value.

    Parameters
    ----------
    data_tbl
        A data table.
    count
        The fixed column count to check against.
    inverse
        `True` to check if the column count does not match the fixed value, `False` otherwise.
    threshold
        The maximum number of failing test units to allow.
    tbl_type
        The type of table to use for the assertion.

    Returns
    -------
    bool
        `True` when test units pass below the threshold level for failing test units, `False`
        otherwise.
    """

    data_tbl: FrameT
    count: int
    inverse: bool
    threshold: int
    tbl_type: str = "local"

    def __post_init__(self):
        from pointblank.validate import get_column_count

        if not self.inverse:
            res = get_column_count(data=self.data_tbl) == self.count
        else:
            res = get_column_count(data=self.data_tbl) != self.count

        self.test_unit_res = res

    def get_test_results(self):
        return self.test_unit_res


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

        polars_expressions = []

        for expr_fn in self.expressions:
            try:
                # First try direct evaluation with native Polars expressions
                expr_result = expr_fn(self.data_tbl)
                if isinstance(expr_result, pl.Expr):
                    polars_expressions.append(expr_result)
                else:
                    raise TypeError("Not a valid Polars expression")
            except Exception as e:
                try:
                    # Try to get a ColumnExpression
                    col_expr = expr_fn(None)
                    if hasattr(col_expr, "to_polars_expr"):
                        polars_expr = col_expr.to_polars_expr()
                        polars_expressions.append(polars_expr)
                    else:  # pragma: no cover
                        raise TypeError(f"Cannot convert {type(col_expr)} to Polars expression")
                except Exception as e:  # pragma: no cover
                    print(f"Error evaluating expression: {e}")

        # Combine results with AND logic
        if polars_expressions:
            final_result = polars_expressions[0]
            for expr in polars_expressions[1:]:
                final_result = final_result & expr

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
                # Silent failure - we already tried both strategies
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
            # More than one parameter - this doesn't match either allowed signature
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
