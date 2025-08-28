from __future__ import annotations

import inspect
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import narwhals as nw
from great_tables import GT
from great_tables.gt import _get_column_of_values
from narwhals.typing import FrameT

from pointblank._constants import ASSERTION_TYPE_METHOD_MAP, GENERAL_COLUMN_TYPES, IBIS_BACKENDS

if TYPE_CHECKING:
    from collections.abc import Mapping

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


def _derive_single_bound(ref: int, tol: int | float) -> int:
    """Derive a single bound using the reference."""
    if not isinstance(tol, float | int):
        raise TypeError("Tolerance must be a number or a tuple of numbers.")
    if tol < 0:
        raise ValueError("Tolerance must be non-negative.")
    return int(tol * ref) if tol < 1 else int(tol)


def _derive_bounds(ref: int, tol: Tolerance) -> AbsoluteBounds:
    """Validate and extract the absolute bounds of the tolerance."""
    if isinstance(tol, tuple):
        return tuple(_derive_single_bound(ref, t) for t in tol)

    bound = _derive_single_bound(ref, tol)
    return bound, bound


def _get_tbl_type(data: FrameT | Any) -> str:
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

        backend = ibis.get_backend(data).name

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

            else:
                return "duckdb"

        return backend

    return "unknown"  # pragma: no cover


def _process_ibis_through_narwhals(data: FrameT | Any, tbl_type: str) -> tuple[FrameT | Any, str]:
    """
    Process Ibis tables through Narwhals to unify the processing pathway.

    This function takes an Ibis table and wraps it with Narwhals, allowing
    all downstream processing to use the unified Narwhals API instead of
    Ibis-specific code paths.

    Parameters
    ----------
    data : FrameT | Any
        The data table, potentially an Ibis table
    tbl_type : str
        The detected table type

    Returns
    -------
    tuple[FrameT | Any, str]
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


def _is_narwhals_table(data: any) -> bool:
    # Check if the data is a Narwhals DataFrame
    type_str = str(type(data)).lower()

    if "narwhals" in type_str:
        # If the object is not a Narwhals DataFrame, return False
        return True

    return False


def _is_lazy_frame(data: any) -> bool:
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
    try:
        import pandas as pd
    except ImportError:
        pd = None

    try:
        import polars as pl
    except ImportError:
        pl = None

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
    try:
        import pandas as pd
    except ImportError:
        pd = None

    # Determine whether Pandas is available
    try:
        import polars as pl
    except ImportError:
        pl = None

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


def _convert_to_narwhals(df: FrameT) -> nw.DataFrame:
    # Convert the DataFrame to a format that narwhals can work with
    return nw.from_native(df)


def _check_column_exists(dfn: nw.DataFrame, column: str) -> None:
    """
    Check if a column exists in a DataFrame.

    Parameters
    ----------
    dfn
        A Narwhals DataFrame.
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
    tbl: FrameT,
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

    # Always collect table if it is a LazyFrame; this is required to get the row count
    if _is_lazy_frame(tbl_filtered):
        tbl_filtered = tbl_filtered.collect()

    return len(tbl_filtered)


def _count_null_values_in_column(
    tbl: FrameT,
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

    # Filter the table to get rows where the specified column is Null
    tbl_filtered = tbl_nw.filter(nw.col(column).is_null())

    # Always collect table if it is a LazyFrame; this is required to get the row count
    if _is_lazy_frame(tbl_filtered):
        tbl_filtered = tbl_filtered.collect()

    return len(tbl_filtered)


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
    dfn: nw.DataFrame, column: str, raw: bool = False, lowercased: bool = True
) -> str:
    """
    Get the data type of a column in a DataFrame.

    Parameters
    ----------
    dfn
        A Narwhals DataFrame.
    column
        The column from which to get the data type.
    raw
        If `True`, return the raw data type string.
    lowercased
        If `True`, return the data type string in lowercase.

    Returns
    -------
    str
        The data type of the column.
    """

    if raw:  # pragma: no cover
        return dfn.collect_schema().get(column)

    column_dtype_str = str(dfn.collect_schema().get(column))

    if lowercased:
        return column_dtype_str.lower()

    return column_dtype_str


def _check_column_type(dfn: nw.DataFrame, column: str, allowed_types: list[str]) -> None:
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
    df: FrameT, column: str, allowed_types: list[str] | None, check_exists: bool = True
) -> nw.DataFrame:
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
    df: FrameT, columns_subset: list[str] | None, check_exists: bool = True
) -> nw.DataFrame:
    # Convert the DataFrame to a format that narwhals can work with.
    dfn = _convert_to_narwhals(df=df)

    # Check whether all columns exist
    if check_exists and columns_subset:
        for column in columns_subset:
            _check_column_exists(dfn=dfn, column=column)

    return dfn


def _get_fn_name() -> str:
    # Get the current function name
    fn_name = inspect.currentframe().f_back.f_code.co_name

    return fn_name


def _get_assertion_from_fname() -> str:
    # Get the current function name
    func_name = inspect.currentframe().f_back.f_code.co_name

    # Use the `ASSERTION_TYPE_METHOD_MAP` dictionary to get the assertion type
    assertion = ASSERTION_TYPE_METHOD_MAP.get(func_name)

    return assertion


def _check_invalid_fields(fields: list[str], valid_fields: list[str]):
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


def get_api_details(module, exported_list):
    """
    Retrieve the signatures and docstrings of the functions/classes in the exported list.

    Parameters
    ----------
    module : module
        The module from which to retrieve the functions/classes.
    exported_list : list
        A list of function/class names as strings.

    Returns
    -------
    str
        A string containing the combined class name, signature, and docstring.
    """
    api_text = ""

    for fn in exported_list:
        # Split the attribute path to handle nested attributes
        parts = fn.split(".")
        obj = module
        for part in parts:
            obj = getattr(obj, part)

        # Get the name of the object
        obj_name = obj.__name__

        # Get the function signature
        sig = inspect.signature(obj)

        # Get the docstring
        doc = obj.__doc__

        # Combine the class name, signature, and docstring
        api_text += f"{obj_name}{sig}\n{doc}\n\n"

    return api_text


def _get_api_text() -> str:
    """
    Get the API documentation for the Pointblank library.

    Returns
    -------
    str
        The API documentation for the Pointblank library.
    """

    import pointblank

    sep_line = "-" * 70

    api_text = (
        f"{sep_line}\nThis is the API documentation for the Pointblank library.\n{sep_line}\n\n"
    )

    #
    # Lists of exported functions and methods in different families
    #

    validate_exported = [
        "Validate",
        "Thresholds",
        "Actions",
        "FinalActions",
        "Schema",
        "DraftValidation",
    ]

    val_steps_exported = [
        "Validate.col_vals_gt",
        "Validate.col_vals_lt",
        "Validate.col_vals_ge",
        "Validate.col_vals_le",
        "Validate.col_vals_eq",
        "Validate.col_vals_ne",
        "Validate.col_vals_between",
        "Validate.col_vals_outside",
        "Validate.col_vals_in_set",
        "Validate.col_vals_not_in_set",
        "Validate.col_vals_null",
        "Validate.col_vals_not_null",
        "Validate.col_vals_regex",
        "Validate.col_vals_expr",
        "Validate.col_exists",
        "Validate.rows_distinct",
        "Validate.rows_complete",
        "Validate.col_schema_match",
        "Validate.row_count_match",
        "Validate.col_count_match",
        "Validate.conjointly",
        "Validate.specially",
    ]

    column_selection_exported = [
        "col",
        "starts_with",
        "ends_with",
        "contains",
        "matches",
        "everything",
        "first_n",
        "last_n",
        "expr_col",
    ]

    segments_exported = [
        "seg_group",
    ]

    interrogation_exported = [
        "Validate.interrogate",
        "Validate.get_tabular_report",
        "Validate.get_step_report",
        "Validate.get_json_report",
        "Validate.get_sundered_data",
        "Validate.get_data_extracts",
        "Validate.all_passed",
        "Validate.assert_passing",
        "Validate.assert_below_threshold",
        "Validate.above_threshold",
        "Validate.n",
        "Validate.n_passed",
        "Validate.n_failed",
        "Validate.f_passed",
        "Validate.f_failed",
        "Validate.warning",
        "Validate.error",
        "Validate.critical",
    ]

    inspect_exported = [
        "DataScan",
        "preview",
        "col_summary_tbl",
        "missing_vals_tbl",
        "assistant",
        "load_dataset",
        "get_data_path",
        "connect_to_table",
    ]

    yaml_exported = [
        "yaml_interrogate",
        "validate_yaml",
    ]

    utility_exported = [
        "get_column_count",
        "get_row_count",
        "get_action_metadata",
        "get_validation_summary",
        "config",
    ]

    prebuilt_actions_exported = [
        "send_slack_notification",
    ]

    validate_desc = """When peforming data validation, you'll need the `Validate` class to get the
process started. It's given the target table and you can optionally provide some metadata and/or
failure thresholds (using the `Thresholds` class or through shorthands for this task). The
`Validate` class has numerous methods for defining validation steps and for obtaining
post-interrogation metrics and data."""

    val_steps_desc = """Validation steps can be thought of as sequential validations on the target
data. We call `Validate`'s validation methods to build up a validation plan: a collection of steps
that, in the aggregate, provides good validation coverage."""

    column_selection_desc = """A flexible way to select columns for validation is to use the `col()`
function along with column selection helper functions. A combination of `col()` + `starts_with()`,
`matches()`, etc., allows for the selection of multiple target columns (mapping a validation across
many steps). Furthermore, the `col()` function can be used to declare a comparison column (e.g.,
for the `value=` argument in many `col_vals_*()` methods) when you can't use a fixed value
for comparison."""

    segments_desc = (
        """Combine multiple values into a single segment using `seg_*()` helper functions."""
    )

    interrogation_desc = """The validation plan is put into action when `interrogate()` is called.
The workflow for performing a comprehensive validation is then: (1) `Validate()`, (2) adding
validation steps, (3) `interrogate()`. After interrogation of the data, we can view a validation
report table (by printing the object or using `get_tabular_report()`), extract key metrics, or we
can split the data based on the validation results (with `get_sundered_data()`)."""

    inspect_desc = """The *Inspection and Assistance* group contains functions that are helpful for
getting to grips on a new data table. Use the `DataScan` class to get a quick overview of the data,
`preview()` to see the first and last few rows of a table, `col_summary_tbl()` for a column-level
summary of a table, `missing_vals_tbl()` to see where there are missing values in a table, and
`get_column_count()`/`get_row_count()` to get the number of columns and rows in a table. Several
datasets included in the package can be accessed via the `load_dataset()` function. Finally, the
`config()` utility lets us set global configuration parameters. Want to chat with an assistant? Use
the `assistant()` function to get help with Pointblank."""

    yaml_desc = """The *YAML* group contains functions that allow for the use of YAML to orchestrate
validation workflows. The `yaml_interrogate()` function can be used to run a validation workflow from
YAML strings or files. The `validate_yaml()` function checks if the YAML configuration
passes its own validity checks."""

    utility_desc = """The Utility Functions group contains functions that are useful for accessing
metadata about the target data. Use `get_column_count()` or `get_row_count()` to get the number of
columns or rows in a table. The `get_action_metadata()` function is useful when building custom
actions since it returns metadata about the validation step that's triggering the action. Lastly,
the `config()` utility lets us set global configuration parameters."""

    prebuilt_actions_desc = """The Prebuilt Actions group contains a function that can be used to
send a Slack notification when validation steps exceed failure threshold levels or just to provide a
summary of the validation results, including the status, number of steps, passing and failing steps,
table information, and timing details."""

    #
    # Add headings (`*_desc` text) and API details for each family of functions/methods
    #

    api_text += f"""\n## The Validate family\n\n{validate_desc}\n\n"""
    api_text += get_api_details(module=pointblank, exported_list=validate_exported)

    api_text += f"""\n## The Validation Steps family\n\n{val_steps_desc}\n\n"""
    api_text += get_api_details(module=pointblank, exported_list=val_steps_exported)

    api_text += f"""\n## The Column Selection family\n\n{column_selection_desc}\n\n"""
    api_text += get_api_details(module=pointblank, exported_list=column_selection_exported)

    api_text += f"""\n## The Segments family\n\n{segments_desc}\n\n"""
    api_text += get_api_details(module=pointblank, exported_list=segments_exported)

    api_text += f"""\n## The Interrogation and Reporting family\n\n{interrogation_desc}\n\n"""
    api_text += get_api_details(module=pointblank, exported_list=interrogation_exported)

    api_text += f"""\n## The Inspection and Assistance family\n\n{inspect_desc}\n\n"""
    api_text += get_api_details(module=pointblank, exported_list=inspect_exported)

    api_text += f"""\n## The YAML family\n\n{yaml_desc}\n\n"""
    api_text += get_api_details(module=pointblank, exported_list=yaml_exported)

    api_text += f"""\n## The Utility Functions family\n\n{utility_desc}\n\n"""
    api_text += get_api_details(module=pointblank, exported_list=utility_exported)

    api_text += f"""\n## The Prebuilt Actions family\n\n{prebuilt_actions_desc}\n\n"""
    api_text += get_api_details(module=pointblank, exported_list=prebuilt_actions_exported)

    # Modify language syntax in all code cells
    api_text = api_text.replace("{python}", "python")

    # Remove code cells that contain `#| echo: false` (i.e., don't display the code)
    api_text = re.sub(r"```python\n\s*.*\n\s*.*\n.*\n.*\n.*```\n\s*", "", api_text)

    return api_text


def _get_examples_text() -> str:
    """
    Get the examples for the Pointblank library. These examples are extracted from the Quarto
    documents in the `docs/demos` directory.

    Returns
    -------
    str
        The examples for the Pointblank library.
    """

    sep_line = "-" * 70

    examples_text = (
        f"{sep_line}\nThis is a set of examples for the Pointblank library.\n{sep_line}\n\n"
    )

    # A large set of examples is available in the docs/demos directory, and each of the
    # subdirectories contains a different example (in the form of a Quarto document)

    example_dirs = [
        "01-starter",
        "02-advanced",
        "03-data-extracts",
        "04-sundered-data",
        "05-step-report-column-check",
        "06-step-report-schema-check",
        "apply-checks-to-several-columns",
        "check-row-column-counts",
        "checks-for-missing",
        "col-vals-custom-expr",
        "column-selector-functions",
        "comparisons-across-columns",
        "expect-no-duplicate-rows",
        "expect-no-duplicate-values",
        "expect-text-pattern",
        "failure-thresholds",
        "mutate-table-in-step",
        "numeric-comparisons",
        "schema-check",
        "set-membership",
        "using-parquet-data",
    ]

    for example_dir in example_dirs:
        link = f"https://posit-dev.github.io/pointblank/demos/{example_dir}/"

        # Read in the index.qmd file for each example
        with open(f"docs/demos/{example_dir}/index.qmd", "r") as f:
            example_text = f.read()

            # Remove the first eight lines of the example text (contains the YAML front matter)
            example_text = "\n".join(example_text.split("\n")[8:])

            # Extract the title of the example (the line beginning with `###`)
            title = re.search(r"### (.*)", example_text).group(1)

            # The next line with text is the short description of the example
            desc = re.search(r"(.*)\.", example_text).group(1)

            # Get all of the Python code blocks in the example
            # these can be identified as starting with ```python and ending with ```
            code_blocks = re.findall(r"```python\n(.*?)```", example_text, re.DOTALL)

            # Wrap each code block with a leading ```python and trailing ```
            code_blocks = [f"```python\n{code}```" for code in code_blocks]

            # Collapse all code blocks into a single string
            code_text = "\n\n".join(code_blocks)

            # Add the example title, description, and code to the examples text
            examples_text += f"### {title} ({link})\n\n{desc}\n\n{code_text}\n\n"

    return examples_text


def _get_api_and_examples_text() -> str:
    """
    Get the combined API and examples text for the Pointblank library.

    Returns
    -------
    str
        The combined API and examples text for the Pointblank library.
    """

    api_text = _get_api_text()
    examples_text = _get_examples_text()

    return f"{api_text}\n\n{examples_text}"


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
    for col, sub_dict in col_dict.items():
        for key, value in sub_dict.items():
            # add columns fields not present
            if key not in result_dict:
                result_dict[key] = [None] * len(col_dict)
            result_dict[key][list(col_dict.keys()).index(col)] = value
    return result_dict
