import inspect
import re
from pathlib import Path


def get_api_details(module, exported_list) -> str:
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

        # Fallback for dynamically generated aggregation methods that might not have
        # their docstrings properly attached yet
        if not doc and obj_name.startswith("col_") and "_" in obj_name:
            # Check if this looks like a dynamically generated aggregation method
            # (e.g., col_sum_gt, col_avg_eq, col_sd_le)
            parts_name = obj_name.split("_")
            if (
                len(parts_name) == 3
                and parts_name[1] in ["sum", "avg", "sd"]
                and parts_name[2] in ["gt", "ge", "lt", "le", "eq"]
            ):
                try:
                    from pointblank.validate import _generate_agg_docstring

                    doc = _generate_agg_docstring(obj_name)
                except Exception:
                    # If we can't generate the docstring, just use what we have
                    pass

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
        "Validate.col_vals_increasing",
        "Validate.col_vals_decreasing",
        "Validate.col_vals_null",
        "Validate.col_vals_not_null",
        "Validate.col_vals_regex",
        "Validate.col_vals_within_spec",
        "Validate.col_vals_expr",
        "Validate.col_sum_gt",
        "Validate.col_sum_lt",
        "Validate.col_sum_ge",
        "Validate.col_sum_le",
        "Validate.col_sum_eq",
        "Validate.col_avg_gt",
        "Validate.col_avg_lt",
        "Validate.col_avg_ge",
        "Validate.col_avg_le",
        "Validate.col_avg_eq",
        "Validate.col_sd_gt",
        "Validate.col_sd_lt",
        "Validate.col_sd_ge",
        "Validate.col_sd_le",
        "Validate.col_sd_eq",
        "Validate.rows_distinct",
        "Validate.rows_complete",
        "Validate.col_exists",
        "Validate.col_pct_null",
        "Validate.data_freshness",
        "Validate.col_schema_match",
        "Validate.row_count_match",
        "Validate.col_count_match",
        "Validate.tbl_match",
        "Validate.conjointly",
        "Validate.specially",
        "Validate.prompt",
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
        "Validate.set_tbl",
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
        "print_database_tables",
    ]

    yaml_exported = [
        "yaml_interrogate",
        "validate_yaml",
        "yaml_to_python",
    ]

    utility_exported = [
        "get_column_count",
        "get_row_count",
        "get_action_metadata",
        "get_validation_summary",
        "write_file",
        "read_file",
        "config",
    ]

    test_data_generation_exported = [
        "generate_dataset",
        "int_field",
        "float_field",
        "string_field",
        "bool_field",
        "date_field",
        "datetime_field",
        "time_field",
        "duration_field",
        "profile_fields",
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
validation workflows. The `yaml_interrogate()` function can be used to run a validation workflow
from YAML strings or files. The `validate_yaml()` function checks if the YAML configuration passes
its own validity checks. The `yaml_to_python()` function converts YAML configuration to equivalent
Python code."""

    utility_desc = """The Utility Functions group contains functions that are useful for accessing
metadata about the target data. Use `get_column_count()` or `get_row_count()` to get the number of
columns or rows in a table. The `get_action_metadata()` function is useful when building custom
actions since it returns metadata about the validation step that's triggering the action. Lastly,
the `config()` utility lets us set global configuration parameters."""

    test_data_generation_desc = """Generate synthetic test data based on schema definitions. Use
`generate_dataset()` to create data from a `Schema` object. The helper functions define typed fields
with constraints for realistic test data generation."""

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

    api_text += f"""\n## The Test Data Generation family\n\n{test_data_generation_desc}\n\n"""
    api_text += get_api_details(module=pointblank, exported_list=test_data_generation_exported)

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
    documents in the `examples` directory.

    Returns
    -------
    str
        The examples for the Pointblank library.
    """

    sep_line = "-" * 70

    examples_text = (
        f"{sep_line}\nThis is a set of examples for the Pointblank library.\n{sep_line}\n\n"
    )

    # Examples are organized in the examples/ directory under category subdirectories,
    # each containing Quarto documents with title and description in YAML front matter

    examples_dir = Path("examples")

    # Collect all .qmd files from category subdirectories, sorted for deterministic order
    example_files = sorted(examples_dir.glob("*/*.qmd"))

    for example_file in example_files:
        # Build the link URL from the file path (e.g., examples/01-getting-started/starter.html)
        link = (
            f"https://posit-dev.github.io/pointblank/"
            f"{example_file.parent.name}/{example_file.stem}.html"
        )

        example_text = example_file.read_text()

        # Extract title and description from YAML front matter
        title_match = re.search(r'^title:\s*"(.+?)"', example_text, re.MULTILINE)
        desc_match = re.search(r'^description:\s*"(.+?)"', example_text, re.MULTILINE)

        if not title_match or not desc_match:
            continue

        title = title_match.group(1)
        desc = desc_match.group(1)

        # Get the plain ```python code blocks (not ```{python} executable blocks)
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
