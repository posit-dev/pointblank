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

