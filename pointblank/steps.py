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

__all__ = ["Steps"]


class Steps:
    """A reusable collection of validation step definitions.

    A `Steps` object is a list of validation steps that isn't attached to any table. You build it
    with the same methods you'd call on a [`Validate`](`pointblank.Validate`) object
    (`col_vals_not_null()`, `rows_distinct()`, `col_vals_regex()`, and so on), but each call only
    records the step. Nothing is checked until the collection is added to a validation with
    [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and that validation is
    interrogated.

    This makes it possible to write a group of checks once and use it in many places. Many tables
    share the same rules: every table with a customer ID needs that ID to be present and unique,
    and every table with an email column needs valid addresses. Without `Steps`, those rules get
    copied into each validation and drift apart over time. With `Steps`, you define them in one
    place (a shared module, for example), name them for what they check (`id_checks`,
    `contact_checks`), and add them to whichever validations need them. Each validation keeps its
    own table, thresholds, actions, and metadata, and can mix shared steps with its own.

    Since the methods have the same signatures as the `Validate` methods they mirror, you get the
    same argument names, defaults, and editor completion, and a misspelled argument raises a
    `TypeError` as soon as the step is recorded.

    Parameters
    ----------
    steps
        An optional list of [`Step`](`pointblank.Step`) objects to start the collection with.
        This is useful when step definitions come from data (a configuration file, for example)
        rather than from method calls.

    Returns
    -------
    Steps
        A `Steps` object. Add steps to it with its validation methods, and apply it with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`).

    Recording Steps
    ---------------
    Each validation method records one step and returns the same `Steps` object, so calls can be
    chained. All of the row-based and table-level validation methods are available, from
    `col_vals_gt()` to `tbl_reconcile()`. The aggregation methods (`col_sum_*()`, `col_avg_*()`,
    and `col_sd_*()`), `specially()`, and `prompt()` aren't, so add those directly to the
    validation.

    Because each method adds to the object it's called on, a collection isn't copied when you add
    to it. Calling a method on a shared collection (e.g., `extended = id_checks.col_exists(...)`)
    adds the step to `id_checks` itself, and `extended` is the same object. To build variations,
    keep the shared parts in separate collections and add several of them to a validation.

    Printing a `Steps` object lists its steps with the arguments that differ from the defaults,
    and `len()` gives the number of steps.

    Applying Steps to a Validation
    ------------------------------
    [`Validate.add_steps()`](`pointblank.Validate.add_steps`) appends the recorded steps to a
    validation, in order, as if each method had been called on it directly. It takes any number of
    `Steps` objects (and also `Validate` objects, whose steps it copies), and has options for
    adapting the steps to the validation at hand:

    - `columns_map=` renames columns in the steps, so that the same collection works on tables that
    name a column differently (e.g., `{"customer_id": "cust_id"}`)
    - `thresholds=` and `active=` replace the thresholds and active state of every added step
    - `exclude=` skips steps by method name or by their 1-based position in a collection

    Steps, Step, and Contract
    -------------------------
    A [`Step`](`pointblank.Step`) is a single step stored as data: a method name and a dictionary
    of arguments. A `Steps` object holds a list of them, built with method calls instead. A
    [`Contract`](`pointblank.Contract`) also holds a list of `Step` objects, together with a schema
    and metadata (an owner, a version, consumers), and can be saved as YAML. Use `Steps` for
    reusable building blocks inside Python code, and a `Contract` for an agreement about a table
    that's shared, versioned, and enforced at a pipeline boundary. The two work together: a
    contract compiled with [`to_validate()`](`pointblank.Contract.to_validate`) is an ordinary
    validation, so collections can be added to it with `add_steps()`.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_footer_timings=False, preview_incl_header=False)
    ```

    Let's build two collections of checks for customer data. The first covers identifiers (each
    customer ID must be present and appear only once), and the second covers contact details
    (email addresses must look valid, and the country must be one of three codes).

    ```{python}
    import pointblank as pb
    import polars as pl

    id_checks = (
        pb.Steps()
        .col_vals_not_null(columns="customer_id")
        .rows_distinct(columns_subset="customer_id")
    )

    contact_checks = (
        pb.Steps()
        .col_vals_regex(columns="email", pattern=r"^[^@]+@[^@]+\\.[a-z]+$")
        .col_vals_in_set(columns="country", set=["CA", "MX", "US"])
    )

    id_checks
    ```

    Displaying a collection lists its steps and their arguments. No table is involved yet.

    Now let's apply both collections to a table of customers from a CRM system. The validation is
    created as usual, and `add_steps()` appends the four recorded steps before interrogation.

    ```{python}
    crm_customers = pl.DataFrame(
        {
            "customer_id": [1, 2, 3, 3, 5],
            "email": ["a@x.com", "b@y.org", "c-at-z.com", "d@w.net", "e@v.io"],
            "country": ["US", "CA", "US", "MX", "BR"],
        }
    )

    (
        pb.Validate(data=crm_customers, tbl_name="crm_customers")
        .add_steps(id_checks, contact_checks)
        .interrogate()
    )
    ```

    The report has one step per recorded step, in the order they were added. Customer ID `3`
    appears twice, so both of its rows fail the `rows_distinct` step. The address `"c-at-z.com"`
    fails the email pattern, and the country `"BR"` isn't in the allowed set.

    The same collections can check a table from a billing system, where the ID column is called
    `cust_id`. The `columns_map=` argument renames the column in the added steps, and
    `thresholds=` gives every added step a warning threshold of one failing test unit.

    ```{python}
    billing_accounts = pl.DataFrame(
        {
            "cust_id": [10, 11, None, 13],
            "email": ["f@x.com", "g@y.com", "h@z.com", "i@w"],
            "country": ["US", "US", "CA", "US"],
        }
    )

    (
        pb.Validate(data=billing_accounts, tbl_name="billing_accounts")
        .add_steps(
            id_checks,
            contact_checks,
            columns_map={"customer_id": "cust_id"},
            thresholds=pb.Thresholds(warning=1),
        )
        .interrogate()
    )
    ```

    The ID steps now check `cust_id`, and one account is missing its ID. The address `"i@w"`
    has no domain suffix, so it fails the email pattern. Both failing steps reach the warning
    threshold.

    See Also
    --------
    Validate.add_steps : Add the steps in one or more collections to a validation.
    Step : A single validation step stored as data.
    Contract : A schema, steps, and metadata that make up an agreement about a table.
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
            # `tbl_match()` comparison settings
            "null_equal": True,
            "dup_keys": "flag",
            "agg": "sum",
        }
        if value is None:
            return True
        if key == "schema" and value == "strict":
            # (`col_schema_match()` also has a `schema=` argument, but never a string value)
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
        """Add a `col_vals_gt()` step to the collection.

        The step records a call to [`Validate.col_vals_gt()`](`pointblank.Validate.col_vals_gt`),
        which checks whether column values are greater than a fixed value or the values in another
        column. It takes the same arguments as that method, so see its reference page for what each
        one does. Nothing is checked when the step is recorded: it runs when the collection is added
        to a validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the
        validation is interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_lt()` step to the collection.

        The step records a call to [`Validate.col_vals_lt()`](`pointblank.Validate.col_vals_lt`),
        which checks whether column values are less than a fixed value or the values in another
        column. It takes the same arguments as that method, so see its reference page for what each
        one does. Nothing is checked when the step is recorded: it runs when the collection is added
        to a validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the
        validation is interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_eq()` step to the collection.

        The step records a call to [`Validate.col_vals_eq()`](`pointblank.Validate.col_vals_eq`),
        which checks whether column values are equal to a fixed value or the values in another
        column. It takes the same arguments as that method, so see its reference page for what each
        one does. Nothing is checked when the step is recorded: it runs when the collection is added
        to a validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the
        validation is interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_ne()` step to the collection.

        The step records a call to [`Validate.col_vals_ne()`](`pointblank.Validate.col_vals_ne`),
        which checks whether column values are not equal to a fixed value or the values in another
        column. It takes the same arguments as that method, so see its reference page for what each
        one does. Nothing is checked when the step is recorded: it runs when the collection is added
        to a validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the
        validation is interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_ge()` step to the collection.

        The step records a call to [`Validate.col_vals_ge()`](`pointblank.Validate.col_vals_ge`),
        which checks whether column values are greater than or equal to a fixed value or the values
        in another column. It takes the same arguments as that method, so see its reference page for
        what each one does. Nothing is checked when the step is recorded: it runs when the
        collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_le()` step to the collection.

        The step records a call to [`Validate.col_vals_le()`](`pointblank.Validate.col_vals_le`),
        which checks whether column values are less than or equal to a fixed value or the values in
        another column. It takes the same arguments as that method, so see its reference page for
        what each one does. Nothing is checked when the step is recorded: it runs when the
        collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_between()` step to the collection.

        The step records a call to
        [`Validate.col_vals_between()`](`pointblank.Validate.col_vals_between`), which checks
        whether column values lie between two bounds, given as fixed values or other columns. It
        takes the same arguments as that method, so see its reference page for what each one does.
        Nothing is checked when the step is recorded: it runs when the collection is added to a
        validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation
        is interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_outside()` step to the collection.

        The step records a call to
        [`Validate.col_vals_outside()`](`pointblank.Validate.col_vals_outside`), which checks
        whether column values lie outside two bounds, given as fixed values or other columns. It
        takes the same arguments as that method, so see its reference page for what each one does.
        Nothing is checked when the step is recorded: it runs when the collection is added to a
        validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation
        is interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_in_set()` step to the collection.

        The step records a call to
        [`Validate.col_vals_in_set()`](`pointblank.Validate.col_vals_in_set`), which checks whether
        column values belong to a set of allowed values. It takes the same arguments as that method,
        so see its reference page for what each one does. Nothing is checked when the step is
        recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_not_in_set()` step to the collection.

        The step records a call to
        [`Validate.col_vals_not_in_set()`](`pointblank.Validate.col_vals_not_in_set`), which checks
        whether column values avoid a set of disallowed values. It takes the same arguments as that
        method, so see its reference page for what each one does. Nothing is checked when the step
        is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
        return self._add(
            "col_vals_not_in_set",
            columns=columns,
            set=set,
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

    def col_vals_increasing(
        self,
        columns: str | list[str] | Column | ColumnSelector | ColumnSelectorNarwhals,
        allow_stationary: bool = False,
        decreasing_tol: float | None = None,
        na_pass: bool = False,
        order_by: str | list[str] | None = None,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        """Add a `col_vals_increasing()` step to the collection.

        The step records a call to
        [`Validate.col_vals_increasing()`](`pointblank.Validate.col_vals_increasing`), which checks
        whether column values increase from row to row. It takes the same arguments as that method,
        so see its reference page for what each one does. Nothing is checked when the step is
        recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
        return self._add(
            "col_vals_increasing",
            columns=columns,
            allow_stationary=allow_stationary,
            decreasing_tol=decreasing_tol,
            na_pass=na_pass,
            order_by=order_by,
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
        order_by: str | list[str] | None = None,
        missing: MissingSpec | None = None,
        pre: Callable | None = None,
        segments: SegmentSpec | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        """Add a `col_vals_decreasing()` step to the collection.

        The step records a call to
        [`Validate.col_vals_decreasing()`](`pointblank.Validate.col_vals_decreasing`), which checks
        whether column values decrease from row to row. It takes the same arguments as that method,
        so see its reference page for what each one does. Nothing is checked when the step is
        recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
        return self._add(
            "col_vals_decreasing",
            columns=columns,
            allow_stationary=allow_stationary,
            increasing_tol=increasing_tol,
            na_pass=na_pass,
            order_by=order_by,
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
        """Add a `col_vals_null()` step to the collection.

        The step records a call to
        [`Validate.col_vals_null()`](`pointblank.Validate.col_vals_null`), which checks whether
        column values are all Null. It takes the same arguments as that method, so see its reference
        page for what each one does. Nothing is checked when the step is recorded: it runs when the
        collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_not_null()` step to the collection.

        The step records a call to
        [`Validate.col_vals_not_null()`](`pointblank.Validate.col_vals_not_null`), which checks
        whether column values are free of Nulls. It takes the same arguments as that method, so see
        its reference page for what each one does. Nothing is checked when the step is recorded: it
        runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_regex()` step to the collection.

        The step records a call to
        [`Validate.col_vals_regex()`](`pointblank.Validate.col_vals_regex`), which checks whether
        column values match a regular expression. It takes the same arguments as that method, so see
        its reference page for what each one does. Nothing is checked when the step is recorded: it
        runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_within_spec()` step to the collection.

        The step records a call to
        [`Validate.col_vals_within_spec()`](`pointblank.Validate.col_vals_within_spec`), which
        checks whether column values fit a specification such as an email address, URL, or postal
        code. It takes the same arguments as that method, so see its reference page for what each
        one does. Nothing is checked when the step is recorded: it runs when the collection is added
        to a validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the
        validation is interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_str_len()` step to the collection.

        The step records a call to
        [`Validate.col_vals_str_len()`](`pointblank.Validate.col_vals_str_len`), which checks
        whether the lengths of string values fall within given bounds. It takes the same arguments
        as that method, so see its reference page for what each one does. Nothing is checked when
        the step is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_expr()` step to the collection.

        The step records a call to
        [`Validate.col_vals_expr()`](`pointblank.Validate.col_vals_expr`), which checks column
        values with a custom expression. It takes the same arguments as that method, so see its
        reference page for what each one does. Nothing is checked when the step is recorded: it runs
        when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_exists()` step to the collection.

        The step records a call to [`Validate.col_exists()`](`pointblank.Validate.col_exists`),
        which checks whether one or more columns exist in the table. It takes the same arguments as
        that method, so see its reference page for what each one does. Nothing is checked when the
        step is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_pct_null()` step to the collection.

        The step records a call to [`Validate.col_pct_null()`](`pointblank.Validate.col_pct_null`),
        which checks whether the percentage of Null values in a column matches an expected value. It
        takes the same arguments as that method, so see its reference page for what each one does.
        Nothing is checked when the step is recorded: it runs when the collection is added to a
        validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation
        is interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_pct_missing()` step to the collection.

        The step records a call to
        [`Validate.col_pct_missing()`](`pointblank.Validate.col_pct_missing`), which checks whether
        the percentage of structured missing values in a column stays within a limit. It takes the
        same arguments as that method, so see its reference page for what each one does. Nothing is
        checked when the step is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_missing_coded()` step to the collection.

        The step records a call to
        [`Validate.col_missing_coded()`](`pointblank.Validate.col_missing_coded`), which checks
        whether all missing values in a column are coded (no uncoded Nulls). It takes the same
        arguments as that method, so see its reference page for what each one does. Nothing is
        checked when the step is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_missing_only_coded()` step to the collection.

        The step records a call to
        [`Validate.col_missing_only_coded()`](`pointblank.Validate.col_missing_only_coded`), which
        checks whether a column contains only documented missing-value codes and legitimate values.
        It takes the same arguments as that method, so see its reference page for what each one
        does. Nothing is checked when the step is recorded: it runs when the collection is added to
        a validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the
        validation is interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `rows_distinct()` step to the collection.

        The step records a call to
        [`Validate.rows_distinct()`](`pointblank.Validate.rows_distinct`), which checks whether rows
        are distinct, across all columns or a subset of them. It takes the same arguments as that
        method, so see its reference page for what each one does. Nothing is checked when the step
        is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `rows_complete()` step to the collection.

        The step records a call to
        [`Validate.rows_complete()`](`pointblank.Validate.rows_complete`), which checks whether rows
        are complete, with no missing values across all columns or a subset of them. It takes the
        same arguments as that method, so see its reference page for what each one does. Nothing is
        checked when the step is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_missing_consistent()` step to the collection.

        The step records a call to
        [`Validate.col_missing_consistent()`](`pointblank.Validate.col_missing_consistent`), which
        checks whether related columns share a consistent pattern of missingness for a given reason.
        It takes the same arguments as that method, so see its reference page for what each one
        does. Nothing is checked when the step is recorded: it runs when the collection is added to
        a validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the
        validation is interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_schema_match()` step to the collection.

        The step records a call to
        [`Validate.col_schema_match()`](`pointblank.Validate.col_schema_match`), which checks
        whether the table's columns and their types match a schema. It takes the same arguments as
        that method, so see its reference page for what each one does. Nothing is checked when the
        step is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `row_count_match()` step to the collection.

        The step records a call to
        [`Validate.row_count_match()`](`pointblank.Validate.row_count_match`), which checks whether
        the table's row count matches an expected count or the count of another table. It takes the
        same arguments as that method, so see its reference page for what each one does. Nothing is
        checked when the step is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `data_freshness()` step to the collection.

        The step records a call to
        [`Validate.data_freshness()`](`pointblank.Validate.data_freshness`), which checks whether
        the latest value in a datetime column is recent enough. It takes the same arguments as that
        method, so see its reference page for what each one does. Nothing is checked when the step
        is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_count_match()` step to the collection.

        The step records a call to
        [`Validate.col_count_match()`](`pointblank.Validate.col_count_match`), which checks whether
        the table's column count matches an expected count or the count of another table. It takes
        the same arguments as that method, so see its reference page for what each one does. Nothing
        is checked when the step is recorded: it runs when the collection is added to a validation
        with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        """Add a `col_vals_in_table()` step to the collection.

        The step records a call to
        [`Validate.col_vals_in_table()`](`pointblank.Validate.col_vals_in_table`), which checks
        whether column values exist in a column of a reference table (referential integrity). It
        takes the same arguments as that method, so see its reference page for what each one does.
        Nothing is checked when the step is recorded: it runs when the collection is added to a
        validation with [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation
        is interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
        keys: str | list[str] | None = None,
        columns: str | list[str] | None = None,
        column_map: dict[str, str] | None = None,
        tolerance: Any = None,
        normalize: dict[str, Any] | None = None,
        null_equal: bool = True,
        schema: str = "strict",
        dup_keys: str = "flag",
        order_by: str | list[str] | None = None,
        partitions: int | None = None,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        """Add a `tbl_match()` step to the collection.

        The step records a call to [`Validate.tbl_match()`](`pointblank.Validate.tbl_match`), which
        checks whether the table matches a comparison table. It takes the same arguments as that
        method, so see its reference page for what each one does. Nothing is checked when the step
        is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
        return self._add(
            "tbl_match",
            tbl_compare=tbl_compare,
            keys=keys,
            columns=columns,
            column_map=column_map,
            tolerance=tolerance,
            normalize=normalize,
            null_equal=null_equal,
            schema=schema,
            dup_keys=dup_keys,
            order_by=order_by,
            partitions=partitions,
            pre=pre,
            thresholds=thresholds,
            actions=actions,
            brief=brief,
            active=active,
            dimension=dimension,
        )

    def tbl_reconcile(
        self,
        tbl_compare: Any,
        keys: str | list[str] | None = None,
        measures: str | list[str] | None = None,
        agg: str | dict[str, Any] = "sum",
        tolerance: Any = None,
        column_map: dict[str, str] | None = None,
        partitions: int | None = None,
        pre: Callable | None = None,
        thresholds: int | float | bool | tuple | dict | Thresholds | None = None,
        actions: Actions | None = None,
        brief: str | bool | None = None,
        active: bool | Callable = True,
        dimension: str | None = None,
    ) -> Steps:
        """Add a `tbl_reconcile()` step to the collection.

        The step records a call to
        [`Validate.tbl_reconcile()`](`pointblank.Validate.tbl_reconcile`), which checks whether
        aggregates of the table reconcile with those of a comparison table. It takes the same
        arguments as that method, so see its reference page for what each one does. Nothing is
        checked when the step is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
        return self._add(
            "tbl_reconcile",
            tbl_compare=tbl_compare,
            keys=keys,
            measures=measures,
            agg=agg,
            tolerance=tolerance,
            column_map=column_map,
            partitions=partitions,
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
        """Add a `conjointly()` step to the collection.

        The step records a call to [`Validate.conjointly()`](`pointblank.Validate.conjointly`),
        which checks whether several row-wise conditions all hold for each row. It takes the same
        arguments as that method, so see its reference page for what each one does. Nothing is
        checked when the step is recorded: it runs when the collection is added to a validation with
        [`Validate.add_steps()`](`pointblank.Validate.add_steps`) and the validation is
        interrogated.

        The expressions are recorded together in the step's `exprs=` argument and passed back as
        positional arguments when the step is applied.

        Returns
        -------
        Steps
            This same `Steps` object, with the new step added at the end, so that calls
            can be chained.
        """
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
