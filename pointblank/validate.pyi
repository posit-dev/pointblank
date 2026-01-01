from pointblank import Actions, Thresholds
from pointblank._utils import _PBUnresolvedColumn
from pointblank.column import Column, ReferenceColumn
from pointblank._typing import Tolerance

from collections.abc import Collection
from dataclasses import dataclass
from great_tables import GT
from narwhals.typing import FrameT, IntoFrame
from pathlib import Path
from pointblank._typing import SegmentSpec, Tolerance
from pointblank._utils import _PBUnresolvedColumn
from pointblank.column import Column, ColumnSelector, ColumnSelectorNarwhals, ReferenceColumn
from pointblank.schema import Schema
from pointblank.thresholds import Actions, FinalActions, Thresholds
from typing import Any, Callable, Literal, ParamSpec, TypeVar

__all__ = [
    "Validate",
    "load_dataset",
    "read_file",
    "write_file",
    "config",
    "connect_to_table",
    "print_database_tables",
    "preview",
    "missing_vals_tbl",
    "get_action_metadata",
    "get_column_count",
    "get_data_path",
    "get_row_count",
    "get_validation_summary",
]

P = ParamSpec("P")
R = TypeVar("R")

def get_action_metadata() -> dict | None: ...
def get_validation_summary() -> dict | None: ...
@dataclass
class PointblankConfig:
    report_incl_header: bool = ...
    report_incl_footer: bool = ...
    report_incl_footer_timings: bool = ...
    report_incl_footer_notes: bool = ...
    preview_incl_header: bool = ...
    def __repr__(self) -> str: ...

def config(
    report_incl_header: bool = True,
    report_incl_footer: bool = True,
    report_incl_footer_timings: bool = True,
    report_incl_footer_notes: bool = True,
    preview_incl_header: bool = True,
) -> PointblankConfig: ...
def load_dataset(
    dataset: Literal["small_table", "game_revenue", "nycflights", "global_sales"] = "small_table",
    tbl_type: Literal["polars", "pandas", "duckdb"] = "polars",
) -> FrameT | Any: ...
def read_file(filepath: str | Path) -> Validate: ...
def write_file(
    validation: Validate,
    filename: str,
    path: str | None = None,
    keep_tbl: bool = False,
    keep_extracts: bool = False,
    quiet: bool = False,
) -> None: ...
def get_data_path(
    dataset: Literal["small_table", "game_revenue", "nycflights", "global_sales"] = "small_table",
    file_type: Literal["csv", "parquet", "duckdb"] = "csv",
) -> str: ...
def preview(
    data: FrameT | Any,
    columns_subset: str | list[str] | Column | None = None,
    n_head: int = 5,
    n_tail: int = 5,
    limit: int = 50,
    show_row_numbers: bool = True,
    max_col_width: int = 250,
    min_tbl_width: int = 500,
    incl_header: bool = None,
) -> GT: ...
def missing_vals_tbl(data: FrameT | Any) -> GT: ...
def get_column_count(data: FrameT | Any) -> int: ...
def get_row_count(data: FrameT | Any) -> int: ...
@dataclass
class _ValidationInfo:
    @classmethod
    def from_agg_validator(
        cls,
        assertion_type: str,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> _ValidationInfo: ...
    i: int | None = ...
    i_o: int | None = ...
    step_id: str | None = ...
    sha1: str | None = ...
    assertion_type: str | None = ...
    column: Any | None = ...
    values: Any | list[any] | tuple | None = ...
    inclusive: tuple[bool, bool] | None = ...
    na_pass: bool | None = ...
    pre: Callable | None = ...
    segments: Any | None = ...
    thresholds: Thresholds | None = ...
    actions: Actions | None = ...
    label: str | None = ...
    brief: str | None = ...
    autobrief: str | None = ...
    active: bool | None = ...
    eval_error: bool | None = ...
    all_passed: bool | None = ...
    n: int | None = ...
    n_passed: int | None = ...
    n_failed: int | None = ...
    f_passed: int | None = ...
    f_failed: int | None = ...
    warning: bool | None = ...
    error: bool | None = ...
    critical: bool | None = ...
    failure_text: str | None = ...
    tbl_checked: FrameT | None = ...
    extract: FrameT | None = ...
    val_info: dict[str, any] | None = ...
    time_processed: str | None = ...
    proc_duration_s: float | None = ...
    notes: dict[str, dict[str, str]] | None = ...
    def get_val_info(self) -> dict[str, any]: ...
    def _add_note(self, key: str, markdown: str, text: str | None = None) -> None: ...
    def _get_notes(self, format: str = "dict") -> dict[str, dict[str, str]] | list[str] | None: ...
    def _get_note(self, key: str, format: str = "dict") -> dict[str, str] | str | None: ...
    def _has_notes(self) -> bool: ...

def connect_to_table(connection_string: str) -> Any: ...
def print_database_tables(connection_string: str) -> list[str]: ...
@dataclass
class Validate:
    data: FrameT | Any
    reference: IntoFrame | None = ...
    tbl_name: str | None = ...
    label: str | None = ...
    thresholds: int | float | bool | tuple | dict | Thresholds | None = ...
    actions: Actions | None = ...
    final_actions: FinalActions | None = ...
    brief: str | bool | None = ...
    lang: str | None = ...
    locale: str | None = ...
    col_names = ...
    col_types = ...
    time_start = ...
    time_end = ...
    validation_info = ...
    def __post_init__(self) -> None: ...
    def _add_agg_validation(
        self,
        *,
        assertion_type: str,
        columns: str | Collection[str],
        value,
        tol: int = 0,
        thresholds=None,
        brief: bool = False,
        actions=None,
        active: bool = True,
    ): ...
    def set_tbl(
        self, tbl: FrameT | Any, tbl_name: str | None = None, label: str | None = None
    ) -> Validate: ...
    def _repr_html_(self) -> str: ...
    def col_vals_gt(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_lt(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_eq(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_ne(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_ge(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_le(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_between(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        left: float | int | Column,
        right: float | int | Column,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_outside(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        left: float | int | Column,
        right: float | int | Column,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_in_set(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        set: Collection[Any],
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_not_in_set(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        set: Collection[Any],
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_increasing(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        allow_stationary: bool = False,
        decreasing_tol: float | None = None,
        na_pass: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_decreasing(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        allow_stationary: bool = False,
        increasing_tol: float | None = None,
        na_pass: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_null(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_not_null(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_regex(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        pattern: str,
        na_pass: bool = False,
        inverse: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_within_spec(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        spec: str,
        na_pass: bool = False,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_vals_expr(
        self,
        expr: Any,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_exists(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_pct_null(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        p: float,
        tol: Tolerance = 0,
        thresholds: int | float | None | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def rows_distinct(
        self,
        columns_subset: str | list[str] | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def rows_complete(
        self,
        columns_subset: str | list[str] | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def prompt(
        self,
        prompt: str,
        model: str,
        columns_subset: str | list[str] | None = None,
        batch_size: int = 1000,
        max_concurrent: int = 3,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_schema_match(
        self,
        schema: Schema,
        complete: bool = True,
        in_order: bool = True,
        case_sensitive_colnames: bool = True,
        case_sensitive_dtypes: bool = True,
        full_match_dtypes: bool = True,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def row_count_match(
        self,
        count: int | FrameT | Any,
        tol: Tolerance = 0,
        inverse: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def col_count_match(
        self,
        count: int | FrameT | Any,
        inverse: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def tbl_match(
        self,
        tbl_compare: FrameT | Any,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def conjointly(
        self,
        *exprs: Callable,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def specially(
        self,
        expr: Callable,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool = True,
    ) -> Validate: ...
    def interrogate(
        self,
        collect_extracts: bool = True,
        collect_tbl_checked: bool = True,
        get_first_n: int | None = None,
        sample_n: int | None = None,
        sample_frac: int | float | None = None,
        extract_limit: int = 500,
    ) -> Validate: ...
    def all_passed(self) -> bool: ...
    def assert_passing(self) -> None: ...
    def assert_below_threshold(
        self, level: str = "warning", i: int | None = None, message: str | None = None
    ) -> None: ...
    def above_threshold(self, level: str = "warning", i: int | None = None) -> bool: ...
    def n(self, i: int | list[int] | None = None, scalar: bool = False) -> dict[int, int] | int: ...
    def n_passed(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, int] | int: ...
    def n_failed(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, int] | int: ...
    def f_passed(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, float] | float: ...
    def f_failed(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, float] | float: ...
    def warning(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, bool] | bool: ...
    def error(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, bool] | bool: ...
    def critical(
        self, i: int | list[int] | None = None, scalar: bool = False
    ) -> dict[int, bool] | bool: ...
    def get_data_extracts(
        self, i: int | list[int] | None = None, frame: bool = False
    ) -> dict[int, FrameT | None] | FrameT | None: ...
    def get_json_report(
        self, use_fields: list[str] | None = None, exclude_fields: list[str] | None = None
    ) -> str: ...
    def get_sundered_data(self, type: str = "pass") -> FrameT: ...
    def get_notes(
        self, i: int, format: str = "dict"
    ) -> dict[str, dict[str, str]] | list[str] | None: ...
    def get_note(self, i: int, key: str, format: str = "dict") -> dict[str, str] | str | None: ...
    def get_tabular_report(
        self,
        title: str | None = ":default:",
        incl_header: bool | None = None,
        incl_footer: bool | None = None,
        incl_footer_timings: bool | None = None,
        incl_footer_notes: bool | None = None,
    ) -> GT: ...
    def get_step_report(
        self,
        i: int,
        columns_subset: str | list[str] | Column | None = None,
        header: str = ":default:",
        limit: int | None = 10,
    ) -> GT: ...
    def _add_validation(self, validation_info): ...
    def _evaluate_column_exprs(self, validation_info): ...
    def _evaluate_segments(self, validation_info): ...
    def _get_validation_dict(self, i: int | list[int] | None, attr: str) -> dict[int, int]: ...
    def _execute_final_actions(self) -> None: ...
    def _get_highest_severity_level(self): ...
    # === GENERATED START ===
    def col_sum_eq(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sum to a value eq some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_sum_eq("a", 15)
        >>> v.assert_passing()
        """
        ...

    def col_sum_gt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sum to a value gt some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_sum_gt("a", 10)
        >>> v.assert_passing()
        """
        ...

    def col_sum_ge(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sum to a value ge some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_sum_ge("a", 15)
        >>> v.assert_passing()
        """
        ...

    def col_sum_lt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sum to a value lt some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_sum_lt("a", 20)
        >>> v.assert_passing()
        """
        ...

    def col_sum_le(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sum to a value le some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_sum_le("a", 15)
        >>> v.assert_passing()
        """
        ...

    def col_avg_eq(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column avg to a value eq some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_avg_eq("a", 3)
        >>> v.assert_passing()
        """
        ...

    def col_avg_gt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column avg to a value gt some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_avg_gt("a", 2)
        >>> v.assert_passing()
        """
        ...

    def col_avg_ge(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column avg to a value ge some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_avg_ge("a", 3)
        >>> v.assert_passing()
        """
        ...

    def col_avg_lt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column avg to a value lt some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_avg_lt("a", 5)
        >>> v.assert_passing()
        """
        ...

    def col_avg_le(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column avg to a value le some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_avg_le("a", 3)
        >>> v.assert_passing()
        """
        ...

    def col_sd_eq(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sd to a value eq some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [2, 4, 6, 8, 10]})
        >>> v = Validate(data).col_sd_eq("a", 3.1622776601683795)
        >>> v.assert_passing()
        """
        ...

    def col_sd_gt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sd to a value gt some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_sd_gt("a", 1)
        >>> v.assert_passing()
        """
        ...

    def col_sd_ge(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sd to a value ge some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [2, 4, 4, 4, 6]})
        >>> v = Validate(data).col_sd_ge("a", 1.4142135623730951)
        >>> v.assert_passing()
        """
        ...

    def col_sd_lt(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sd to a value lt some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [1, 2, 3, 4, 5]})
        >>> v = Validate(data).col_sd_lt("a", 2)
        >>> v.assert_passing()
        """
        ...

    def col_sd_le(
        self,
        columns: _PBUnresolvedColumn,
        value: float | Column | ReferenceColumn | None = None,
        tol: Tolerance = 0,
        thresholds: float | bool | tuple | dict | Thresholds | None = None,
        brief: str | bool = False,
        actions: Actions | None = None,
        active: bool = True,
    ) -> Validate:
        """Assert the values in a column sd to a value le some `value`.

        Args:
            columns (_PBUnresolvedColumn): Column or collection of columns to validate.
            value (float | Column | ReferenceColumn | None): Target value to validate against.
                If None and reference data is set on the Validate object, defaults to
                ref(column) to compare against the same column in the reference data.
            tol (Tolerance, optional): Tolerance for validation distance to target. Defaults to 0.
            thresholds (float | bool | tuple | dict | Thresholds | None, optional): Custom thresholds for
                the bounds. See examples for usage. Defaults to None.
            brief (str | bool, optional): Explanation of validation operation. Defaults to False.
            actions (Actions | None, optional): Actions to take after validation. Defaults to None.
            active (bool, optional): Whether to activate the validation. Defaults to True.

        Returns:
            Validate: A `Validate` instance with the new validation method added.

        Examples:
        >>> import polars as pl
        >>>
        >>> data = pl.DataFrame({"a": [2, 4, 4, 4, 6]})
        >>> v = Validate(data).col_sd_le("a", 1.4142135623730951)
        >>> v.assert_passing()
        """
        ...

    # === GENERATED END ===
