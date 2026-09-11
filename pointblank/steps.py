from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Any, Callable

from pointblank.contract import Step

if TYPE_CHECKING:
    from collections.abc import Collection

    from pointblank._typing import SegmentSpec, Tolerance
    from pointblank.actions import Actions
    from pointblank.column import Column, ColumnSelector, ColumnSelectorNarwhals
    from pointblank.missing import MissingSpec
    from pointblank.schema import Schema
    from pointblank.thresholds import Thresholds
    from pointblank.validate import Validate

__all__ = ["Steps"]


class Steps:
    """A reusable collection of validation step definitions.

    A `Steps` object records validation steps without binding them to data, thresholds, or
    metadata. It mirrors the validation method API on
    [`Validate`](`pointblank.Validate`) so that defining steps feels identical, but
    the result is a portable recipe that can be imported into any number of pipelines via
    [`Validate.add_steps()`](`pointblank.Validate.add_steps`).

    Parameters
    ----------
    steps
        An optional list of [`Step`](`pointblank.Step`) objects to initialize with.

    Examples
    --------
    ```python
    import pointblank as pb

    completeness = (
        pb.Steps()
        .col_vals_not_null(columns=pb.ends_with("_id"))
        .col_vals_not_null(columns="email")
    )

    positive_amounts = (
        pb.Steps()
        .col_vals_ge(columns=pb.starts_with("amt_"), value=0)
        .col_vals_gt(columns="amt_total", value=0)
    )

    validation = (
        pb.Validate(data=orders, label="Order quality")
        .add_steps(completeness, positive_amounts)
        .interrogate()
    )
    ```
    """

    def __init__(self, steps: list[Step] | None = None) -> None:
        self._steps: list[Step] = list(steps) if steps is not None else []

    def _add(self, method: str, **kwargs: Any) -> Steps:
        self._steps.append(Step(method, **kwargs))
        return self

    def __len__(self) -> int:
        return len(self._steps)

    @staticmethod
    def _is_default(key: str, value: Any) -> bool:
        _DEFAULTS = {
            "na_pass": False,
            "inverse": False,
            "active": True,
            "allow_stationary": False,
            "allow_tz_mismatch": False,
            "complete": True,
            "in_order": True,
            "case_sensitive_colnames": True,
            "case_sensitive_dtypes": True,
            "full_match_dtypes": True,
            "inclusive": (True, True),
        }
        if value is None:
            return True
        if key in _DEFAULTS:
            return value == _DEFAULTS[key]
        return False

    def __repr__(self) -> str:
        lines = []
        for i, step in enumerate(self._steps):
            shown = {k: v for k, v in step.kwargs.items() if not self._is_default(k, v)}
            if shown:
                kwargs_str = ", ".join(f"{k}={v!r}" for k, v in shown.items())
                lines.append(f"  {i + 1}. {step.method}({kwargs_str})")
            else:
                lines.append(f"  {i + 1}. {step.method}()")
        header = f"Steps({len(self._steps)} step{'s' if len(self._steps) != 1 else ''})"
        if not lines:
            return header
        return header + "\n" + "\n".join(lines)

    def _repr_html_(self) -> str:
        from pointblank._utils_html import _create_steps_html

        return _create_steps_html(self)

    # -- Validation methods -------------------------------------------------------
    # Each method mirrors the corresponding Validate method signature but simply
    # records the call as a Step for later application via add_steps().

    def col_vals_gt(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_gt",
            columns=columns,
            value=value,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_lt(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_lt",
            columns=columns,
            value=value,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_eq(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_eq",
            columns=columns,
            value=value,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_ne(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_ne",
            columns=columns,
            value=value,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_ge(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_ge",
            columns=columns,
            value=value,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_le(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        value: float | int | Column,
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_le",
            columns=columns,
            value=value,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_between(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        left: float | int | Column,
        right: float | int | Column,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_between",
            columns=columns,
            left=left,
            right=right,
            inclusive=inclusive,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_outside(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        left: float | int | Column,
        right: float | int | Column,
        inclusive: tuple[bool, bool] = (True, True),
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_outside",
            columns=columns,
            left=left,
            right=right,
            inclusive=inclusive,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_in_set(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        set: Collection[Any],
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_in_set",
            columns=columns,
            set=set,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_not_in_set(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        set: Collection[Any],
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_not_in_set",
            columns=columns,
            set=set,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_increasing(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        allow_stationary: bool = False,
        decreasing_tol: float | None = None,
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_increasing",
            columns=columns,
            allow_stationary=allow_stationary,
            decreasing_tol=decreasing_tol,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_decreasing(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        allow_stationary: bool = False,
        increasing_tol: float | None = None,
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_decreasing",
            columns=columns,
            allow_stationary=allow_stationary,
            increasing_tol=increasing_tol,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_null(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_null",
            columns=columns,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_not_null(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_not_null",
            columns=columns,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_regex(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        pattern: str,
        na_pass: bool = False,
        inverse: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_regex",
            columns=columns,
            pattern=pattern,
            na_pass=na_pass,
            inverse=inverse,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_within_spec(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        spec: str,
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_within_spec",
            columns=columns,
            spec=spec,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_str_len(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        min_val: int | None = None,
        max_val: int | None = None,
        na_pass: bool = False,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_str_len",
            columns=columns,
            min_val=min_val,
            max_val=max_val,
            na_pass=na_pass,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_expr(
        self,
        expr: Any,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_expr",
            expr=expr,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_exists(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_exists",
            columns=columns,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_pct_null(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        p: float,
        tol: Tolerance = 0,
        thresholds: int | float | None | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_pct_null",
            columns=columns,
            p=p,
            tol=tol,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_pct_missing(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        missing: MissingSpec,
        max_pct: float,
        reason: str | None = None,
        category: str | None = None,
        thresholds: int | float | None | bool | tuple | dict | Thresholds = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_pct_missing",
            columns=columns,
            missing=missing,
            max_pct=max_pct,
            reason=reason,
            category=category,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_missing_coded(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        missing: MissingSpec,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_missing_coded",
            columns=columns,
            missing=missing,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_missing_only_coded(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        missing: MissingSpec,
        allowed: Collection[Any] | None = None,
        min_val: float | int | None = None,
        max_val: float | int | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_missing_only_coded",
            columns=columns,
            missing=missing,
            allowed=allowed,
            min_val=min_val,
            max_val=max_val,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def rows_distinct(
        self,
        columns_subset: str | list[str] | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "rows_distinct",
            columns_subset=columns_subset,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def rows_complete(
        self,
        columns_subset: str | list[str] | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "rows_complete",
            columns_subset=columns_subset,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_missing_consistent(
        self,
        columns: list[str],
        missing: MissingSpec,
        when_reason: str,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_missing_consistent",
            columns=columns,
            missing=missing,
            when_reason=when_reason,
            pre=pre,
            segments=segments,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_schema_match(
        self,
        schema: Schema,
        complete: bool = True,
        in_order: bool = True,
        case_sensitive_colnames: bool = True,
        case_sensitive_dtypes: bool = True,
        full_match_dtypes: bool = True,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_schema_match",
            schema=schema,
            complete=complete,
            in_order=in_order,
            case_sensitive_colnames=case_sensitive_colnames,
            case_sensitive_dtypes=case_sensitive_dtypes,
            full_match_dtypes=full_match_dtypes,
            pre=pre,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def row_count_match(
        self,
        count: int | Any,
        tol: Tolerance = 0,
        inverse: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "row_count_match",
            count=count,
            tol=tol,
            inverse=inverse,
            pre=pre,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def data_freshness(
        self,
        column: str,
        max_age: str | datetime.timedelta,
        reference_time: datetime.datetime | str | None = None,
        timezone: str | None = None,
        allow_tz_mismatch: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "data_freshness",
            column=column,
            max_age=max_age,
            reference_time=reference_time,
            timezone=timezone,
            allow_tz_mismatch=allow_tz_mismatch,
            pre=pre,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_count_match(
        self,
        count: int | Any,
        inverse: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_count_match",
            count=count,
            inverse=inverse,
            pre=pre,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def col_vals_in_table(
        self,
        columns: str | list[str],
        ref_table: Any,
        ref_column: str | list[str],
        na_pass: bool = False,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "col_vals_in_table",
            columns=columns,
            ref_table=ref_table,
            ref_column=ref_column,
            na_pass=na_pass,
            pre=pre,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def tbl_match(
        self,
        tbl_compare: Any,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "tbl_match",
            tbl_compare=tbl_compare,
            pre=pre,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def conjointly(
        self,
        *exprs: Callable,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        return self._add(
            "conjointly",
            exprs=exprs,
            pre=pre,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )
