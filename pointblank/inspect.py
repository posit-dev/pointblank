from __future__ import annotations

from typing import Any, Callable


def has_columns(*columns: str | list[str]) -> Callable[[Any], bool]:
    """
    Check whether one or more columns exist in a table.

    This function returns a callable that, when given a table, checks whether all specified columns
    are present. It is primarily designed for use with the `active=` parameter of validation
    methods. When a validation step has `active=has_columns("col_a", "col_b")`, the step will be
    skipped (made inactive) if either `col_a` or `col_b` is missing from the target table.

    The callable is evaluated against the original table *before* any `pre=` processing is
    applied. This means the column check is performed on the raw input data, not on a
    pre-processed version of it.

    A note is attached to any skipped step in the validation report explaining which columns were
    not found.

    Parameters
    ----------
    *columns
        One or more column names to check for in the table. Each argument can be a string or a
        list of strings. All specified columns must be present for the callable to return `True`.

    Returns
    -------
    Callable[[Any], bool]
        A callable that accepts a table and returns `True` if every column in `columns` exists
        in the table, `False` otherwise.

    Raises
    ------
    ValueError
        If no column names are provided.
    TypeError
        If any of the provided column names is not a string or list of strings.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Using `has_columns()` with the `active=` parameter to conditionally run a validation step:

    ```{python}
    import pointblank as pb
    import polars as pl

    tbl = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns="a", value=0, active=pb.has_columns("a"))
        .col_vals_gt(columns="a", value=0, active=pb.has_columns("z"))
        .interrogate()
    )

    validation
    ```

    The first step ran because column `a` exists. The second step was skipped because column `z` is
    missing, and the report note explains which column was not found.

    When checking for multiple columns, the step is only active when *all* columns are present:

    ```{python}
    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns="a", value=0, active=pb.has_columns("a", "b"))
        .col_vals_gt(columns="a", value=0, active=pb.has_columns("a", "x", "y"))
        .interrogate()
    )

    validation
    ```

    The first step is active because both `a` and `b` exist. The second step is skipped because `x`
    and `y` are missing.

    Column names can also be provided as a list:

    ```{python}
    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns="a", value=0, active=pb.has_columns(["a", "b"]))
        .interrogate()
    )

    validation
    ```
    """

    # Flatten: accept a mix of individual strings and lists of strings
    flat_columns: list[str] = []
    for c in columns:
        if isinstance(c, str):
            flat_columns.append(c)
        elif isinstance(c, list):
            for item in c:
                if not isinstance(item, str):
                    raise TypeError(
                        "Column names provided to `has_columns()` must be strings, "
                        f"got {type(item).__name__} inside a list."
                    )
                flat_columns.append(item)
        else:
            raise TypeError(
                f"Column names provided to `has_columns()` must be strings or lists of strings, "
                f"got {type(c).__name__}."
            )

    if len(flat_columns) == 0:
        raise ValueError("At least one column name must be provided to `has_columns()`.")

    columns = tuple(flat_columns)

    def _check(tbl: Any) -> bool:
        import narwhals as nw

        tbl_nw = nw.from_native(tbl)
        tbl_columns = set(tbl_nw.columns)
        result = all(col in tbl_columns for col in columns)

        if not result:
            missing = sorted(col for col in columns if col not in tbl_columns)
            _check._reason = {
                "key": "active_check_missing_columns",
                "params": {"columns": missing},
            }
        else:
            _check._reason = None

        return result

    _check._reason = None
    return _check


def has_rows(
    count: int | None = None, *, min: int | None = None, max: int | None = None
) -> Callable[[Any], bool]:
    """
    Check whether a table has a certain number of rows.

    The `has_rows()` function returns a callable that, when given a table, checks whether the row
    count satisfies a specified condition. It is designed for use with the `active=` parameter of
    validation methods so that a validation step can be conditionally skipped when the target table
    is too small, too large, or empty.

    The callable supports several modes:

    - **exact count**: `has_rows(count=N)` returns `True` only if the table has exactly `N` rows.
    - **minimum**: `has_rows(min=N)` returns `True` if the table has at least `N` rows.
    - **maximum**: `has_rows(max=N)` returns `True` if the table has at most `N` rows.
    - **range**: `has_rows(min=A, max=B)` returns `True` if the row count falls within `[A, B]`.
    - **non-empty**: `has_rows()` (no arguments) returns `True` if the table has at least one row.

    A note is attached to any skipped step in the validation report explaining the row count
    condition that was not met.

    The callable is evaluated against the original table *before* any `pre=` processing is
    applied. This means the column check is performed on the raw input data, not on a
    pre-processed version of it.

    Parameters
    ----------
    count
        The exact number of rows the table should have. Cannot be used together with `min=` or
        `max=`.
    min
        The minimum number of rows (inclusive) the table should have. Can be used alone or with
        `max=`.
    max
        The maximum number of rows (inclusive) the table should have. Can be used alone or with
        `min=`.

    Returns
    -------
    Callable[[Any], bool]
        A callable that accepts a table and returns `True` if the row count satisfies the specified
        condition, `False` otherwise. When the callable returns `False`, it stores diagnostic
        information that is used to generate a descriptive note in the validation report.

    How It Works
    ------------
    When [`interrogate()`](`pointblank.Validate.interrogate`) is called, each validation step whose
    `active=` parameter is a callable will have that callable evaluated with the target table. If
    the callable returns `False`, the step is deactivated and an explanatory note is added to the
    validation report. The note is locale-aware: if the
    [`Validate`](`pointblank.Validate`) object was created with a non-English `locale=`, the note
    will be translated accordingly.

    Examples
    --------
    ```{python}
    #| echo: false
    #| output: false
    import pointblank as pb
    pb.config(report_incl_header=False, report_incl_footer=False, preview_incl_header=False)
    ```

    Skip a validation step if the table is empty:

    ```{python}
    import pointblank as pb
    import polars as pl

    tbl = pl.DataFrame({"x": [1, 2, 3]})
    empty_tbl = pl.DataFrame({"x": []})

    validation = (
        pb.Validate(data=empty_tbl)
        .col_vals_gt(columns="x", value=0, active=pb.has_rows())
        .interrogate()
    )

    validation
    ```

    The step was skipped because the table has no rows.

    Only run a step when the table has at least a minimum number of rows:

    ```{python}
    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns="x", value=0, active=pb.has_rows(min=100))
        .interrogate()
    )

    validation
    ```

    The step was skipped because the table has only 3 rows, which is fewer than the required
    minimum of `100`.

    You can also check for an exact count or a range:

    ```{python}
    validation = (
        pb.Validate(data=tbl)
        .col_vals_gt(columns="x", value=0, active=pb.has_rows(count=3))
        .col_vals_gt(columns="x", value=0, active=pb.has_rows(min=2, max=10))
        .col_vals_gt(columns="x", value=0, active=pb.has_rows(count=100))
        .interrogate()
    )

    validation
    ```

    The first two steps ran because the table has exactly 3 rows (matching `count=3`) and falls
    within the range `[2, 10]`. The third step was skipped because `3` does not equal `100`.
    """

    if count is not None and (min is not None or max is not None):
        raise ValueError("`count=` cannot be used together with `min=` or `max=` in `has_rows()`.")

    if count is not None and count < 0:
        raise ValueError("`count=` must be a non-negative integer in `has_rows()`.")
    if min is not None and min < 0:
        raise ValueError("`min=` must be a non-negative integer in `has_rows()`.")
    if max is not None and max < 0:
        raise ValueError("`max=` must be a non-negative integer in `has_rows()`.")
    if min is not None and max is not None and min > max:
        raise ValueError("`min=` cannot be greater than `max=` in `has_rows()`.")

    def _check(tbl: Any) -> bool:
        from pointblank.validate import get_row_count

        n_rows = get_row_count(tbl)
        result = True
        reason = None

        if count is not None:
            result = n_rows == count
            if not result:
                reason = {
                    "key": "active_check_row_count_exact",
                    "params": {"expected": count, "found": n_rows},
                }
        elif min is not None and max is not None:
            result = min <= n_rows <= max
            if not result:
                reason = {
                    "key": "active_check_row_count_range",
                    "params": {"min": min, "max": max, "found": n_rows},
                }
        elif min is not None:
            result = n_rows >= min
            if not result:
                reason = {
                    "key": "active_check_row_count_min",
                    "params": {"min": min, "found": n_rows},
                }
        elif max is not None:
            result = n_rows <= max
            if not result:
                reason = {
                    "key": "active_check_row_count_max",
                    "params": {"max": max, "found": n_rows},
                }
        else:
            # Default: table has at least one row
            result = n_rows > 0
            if not result:
                reason = {
                    "key": "active_check_table_empty",
                    "params": {},
                }

        _check._reason = reason
        return result

    _check._reason = None
    return _check
