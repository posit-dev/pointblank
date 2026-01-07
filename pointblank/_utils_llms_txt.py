import inspect
import re
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

try:
    import requests

    SCRAPING_AVAILABLE = True
except ImportError:
    SCRAPING_AVAILABLE = False


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
            title_match = re.search(r"### (.*)", example_text)
            assert title_match is not None
            title = title_match.group(1)

            # The next line with text is the short description of the example
            desc_match = re.search(r"(.*)\.", example_text)
            assert desc_match is not None
            desc = desc_match.group(1)

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


def scrape_examples_index(base_url: str = "https://posit-dev.github.io/pointblank/") -> list[dict]:
    """
    Parse the examples index page from local .qmd file to extract demo titles and descriptions.

    Parameters
    ----------
    base_url : str
        The base URL of the Pointblank documentation site.

    Returns
    -------
    list[dict]
        A list of dictionaries with 'title', 'description', and 'url' keys.
    """
    examples = []

    # Read from local file
    qmd_path = Path(__file__).parent.parent / "docs" / "demos" / "index.qmd"

    if not qmd_path.exists():
        # Fallback to web scraping if local file doesn't exist
        if not SCRAPING_AVAILABLE:
            raise ImportError(
                "requests is required for web scraping. Install it with: pip install requests"
            )
        demos_url = urljoin(base_url, "demos/")
        response = requests.get(demos_url)
        response.raise_for_status()
        content = response.text
    else:
        with open(qmd_path, "r") as f:
            content = f.read()

    # Pattern to match the example structure in the .qmd file:
    # [Title](./path/index.qmd)
    # ... potentially an image ...
    # <p ...>Description</p>

    # First, get the grid-based examples with images
    grid_pattern = r"\[([^\]]+)\]\(\./([^)]+)/index\.qmd\).*?<p[^>]*>(.*?)</p>"
    matches = re.findall(grid_pattern, content, re.DOTALL)

    for title, path, description in matches:
        url = urljoin(base_url, f"demos/{path}/")
        # Clean up description
        desc_clean = re.sub(r"<[^>]+>", "", description).strip()
        examples.append({"title": title.strip(), "description": desc_clean, "url": url})

    # Also get the list-style examples (after the <hr>)
    list_pattern = r"\[([^\]]+)\]\(\./([^)]+)/index\.qmd\)<br>\s*([^\n]+)"
    list_matches = re.findall(list_pattern, content)

    for title, path, description in list_matches:
        url = urljoin(base_url, f"demos/{path}/")
        examples.append({"title": title.strip(), "description": description.strip(), "url": url})

    return examples


def scrape_api_reference_index(
    base_url: str = "https://posit-dev.github.io/pointblank/",
) -> list[dict]:
    """
    Parse the API reference index page from local .qmd file to extract function/class names and descriptions.

    Parameters
    ----------
    base_url : str
        The base URL of the Pointblank documentation site.

    Returns
    -------
    list[dict]
        A list of dictionaries with 'title', 'description', and 'url' keys.
    """
    api_items = []

    # Read from local file
    qmd_path = Path(__file__).parent.parent / "docs" / "reference" / "index.qmd"

    if not qmd_path.exists():
        # Fallback to web scraping if local file doesn't exist
        if not SCRAPING_AVAILABLE:
            raise ImportError(
                "requests is required for web scraping. Install it with: pip install requests"
            )
        reference_url = urljoin(base_url, "reference/")
        response = requests.get(reference_url)
        response.raise_for_status()
        content = response.text
    else:
        with open(qmd_path, "r") as f:
            content = f.read()

    # Pattern to match the API reference structure in the .qmd file:
    # | [Function](path.qmd#anchor) | Description |

    table_row_pattern = r"\| \[([^\]]+)\]\(([^)]+)\) \| ([^\|]+) \|"
    matches = re.findall(table_row_pattern, content)

    for title, path, description in matches:
        # Extract just the filename without the anchor and change .qmd to .html
        file_path = path.split("#")[0]
        if file_path.endswith(".qmd"):
            file_path = file_path[:-4] + ".html"
        url = urljoin(base_url, f"reference/{file_path}")

        api_items.append({"title": title.strip(), "description": description.strip(), "url": url})

    return api_items


def generate_llms_txt(
    base_url: str = "https://posit-dev.github.io/pointblank/",
    include_user_guide: bool = True,
) -> str:
    """
    Generate the llms.txt content for the Pointblank project.

    Parameters
    ----------
    base_url : str
        The base URL of the Pointblank documentation site.
    include_user_guide : bool
        Whether to include user guide pages in the output.

    Returns
    -------
    str
        The llms.txt formatted content.
    """
    if not SCRAPING_AVAILABLE:
        raise ImportError(
            "requests is required for web scraping. Install it with: pip install requests"
        )

    lines = ["# Pointblank", "", "## Docs", ""]

    # Add examples section
    try:
        examples = scrape_examples_index(base_url)
        if examples:
            lines.append("### Examples")
            lines.append("")
            for ex in examples:
                desc = f": {ex['description']}" if ex["description"] else ""
                lines.append(f"- [{ex['title']}]({ex['url']}){desc}")
            lines.append("")
    except Exception as e:
        print(f"Warning: Failed to scrape examples index: {e}")

    # Add API reference section
    try:
        api_items = scrape_api_reference_index(base_url)
        if api_items:
            lines.append("### API Reference")
            lines.append("")
            for item in api_items:
                desc = f": {item['description']}" if item["description"] else ""
                lines.append(f"- [{item['title']}]({item['url']}){desc}")
            lines.append("")
    except Exception as e:
        print(f"Warning: Failed to scrape API reference: {e}")

    # If user guide is requested, scrape it too
    if include_user_guide:
        try:
            user_guide_items = scrape_user_guide_index(base_url)
            if user_guide_items:
                lines.append("### User Guide")
                lines.append("")
                for item in user_guide_items:
                    desc = f": {item['description']}" if item["description"] else ""
                    lines.append(f"- [{item['title']}]({item['url']}){desc}")
        except Exception as e:
            print(f"Warning: Failed to scrape user guide: {e}")

    return "\n".join(lines)


def scrape_user_guide_index(
    base_url: str = "https://posit-dev.github.io/pointblank/",
) -> list[dict]:
    """
    Get the user guide pages from local directory listing.

    Parameters
    ----------
    base_url : str
        The base URL of the Pointblank documentation site.

    Returns
    -------
    list[dict]
        A list of dictionaries with 'title', 'description', and 'url' keys.
    """
    guide_items = []

    # Read from local directory
    user_guide_dir = Path(__file__).parent.parent / "docs" / "user-guide"

    if not user_guide_dir.exists():
        return guide_items

    # Get all .qmd files (excluding index.qmd)
    qmd_files = sorted([f for f in user_guide_dir.glob("*.qmd") if f.name != "index.qmd"])

    for qmd_file in qmd_files:
        # Read the file to extract title
        with open(qmd_file, "r") as f:
            content = f.read()

        # Try to extract title from YAML frontmatter
        title_match = re.search(r'^title:\s*["\']?([^"\'\n]+)["\']?', content, re.MULTILINE)
        if title_match:
            title = title_match.group(1).strip()
        else:
            # Fallback to filename
            title = qmd_file.stem.replace("-", " ").title()

        # Try to extract first paragraph as description (optional)
        # Skip code blocks and look for first real content
        description = ""

        url = urljoin(base_url, f"user-guide/{qmd_file.stem}.html")

        guide_items.append({"title": title, "description": description, "url": url})

    return guide_items


def generate_llms_full_txt(output_path: Optional[str] = None) -> str:
    """
    Generate the llms-full.txt content using the existing api-docs.txt file or by generating
    the API and examples text.

    Parameters
    ----------
    output_path : str, optional
        Path to save the generated content. If None, content is returned but not saved.

    Returns
    -------
    str
        The llms-full.txt formatted content.
    """
    # Try to use existing api-docs.txt first
    api_docs_path = Path(__file__).parent / "data" / "api-docs.txt"

    if api_docs_path.exists():
        with open(api_docs_path, "r") as f:
            content = f.read()
    else:
        # Generate the content
        content = _get_api_and_examples_text()

    if output_path:
        with open(output_path, "w") as f:
            f.write(content)

    return content


def main():
    """
    Main function to generate both llms.txt and llms-full.txt files.
    """
    # Generate llms.txt
    print("Generating llms.txt...")
    try:
        llms_content = generate_llms_txt()
        llms_path = Path(__file__).parent.parent / "docs" / "llms.txt"
        with open(llms_path, "w") as f:
            f.write(llms_content)
        print(f"✓ Generated {llms_path}")
    except Exception as e:
        print(f"✗ Failed to generate llms.txt: {e}")

    # Generate llms-full.txt
    print("\nGenerating llms-full.txt...")
    try:
        llms_full_path = Path(__file__).parent.parent / "docs" / "llms-full.txt"
        generate_llms_full_txt(str(llms_full_path))
        print(f"✓ Generated {llms_full_path}")
    except Exception as e:
        print(f"✗ Failed to generate llms-full.txt: {e}")


if __name__ == "__main__":
    main()
