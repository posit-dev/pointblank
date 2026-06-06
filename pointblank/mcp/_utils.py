"""Utility functions for the Pointblank MCP server."""

import json
import logging
import math
import re
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pointblank as pb
from pointblank.mcp._config import (
    _ALLOWED_DATA_EXTENSIONS,
    _MAX_FILE_SIZE_BYTES,
    HAS_PANDAS,
    HAS_POLARS,
    TESTING_MODE,
    pd,
    pl,
)

logger = logging.getLogger(__name__)

# Regex for valid resource IDs: alphanumeric, underscores, hyphens, max 128 chars
_VALID_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]{1,128}$")


def validate_resource_id(resource_id: str, resource_type: str = "resource") -> str:
    """
    Validate a user-provided resource ID (DataFrame ID or Validator ID).

    Prevents injection of special characters, excessively long IDs,
    or empty strings.
    """
    if not resource_id or not resource_id.strip():
        raise ValueError(f"{resource_type} ID cannot be empty.")

    resource_id = resource_id.strip()

    if not _VALID_ID_PATTERN.match(resource_id):
        raise ValueError(
            f"Invalid {resource_type} ID: '{resource_id}'. "
            f"IDs must be 1-128 characters using only letters, digits, underscores, and hyphens."
        )

    return resource_id


def validate_input_path(input_path: str) -> Path:
    """
    Validate and resolve an input file path, preventing path traversal attacks.

    Ensures the path:
    - Is resolved to an absolute path
    - Does not contain traversal sequences
    - Has an allowed file extension
    - Does not exceed the maximum file size
    """
    p = Path(input_path).resolve()

    # Block obvious traversal patterns in the raw input
    if ".." in Path(input_path).parts:
        raise ValueError("Path traversal ('..') is not allowed in file paths.")

    # Validate extension
    if p.suffix.lower() not in _ALLOWED_DATA_EXTENSIONS:
        raise ValueError(
            f"File type '{p.suffix}' is not allowed. "
            f"Supported: {', '.join(sorted(_ALLOWED_DATA_EXTENSIONS))}"
        )

    # Check existence
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    if not p.is_file():
        raise ValueError(f"Path is not a regular file: {p}")

    # Check file size
    file_size = p.stat().st_size
    if file_size > _MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds the "
            f"{_MAX_FILE_SIZE_BYTES / 1024 / 1024:.0f} MB limit."
        )

    return p


def validate_output_path(output_path: str, allowed_extensions: set[str]) -> Path:
    """
    Validate an output file path, preventing writes to dangerous locations.

    Ensures the path:
    - Is resolved to an absolute path
    - Does not contain traversal sequences
    - Has an allowed file extension
    - Parent directory exists or can be created
    """
    p = Path(output_path).resolve()

    # Block traversal
    if ".." in Path(output_path).parts:
        raise ValueError("Path traversal ('..') is not allowed in file paths.")

    # Validate extension
    if p.suffix.lower() not in allowed_extensions:
        raise ValueError(
            f"File type '{p.suffix}' is not allowed. "
            f"Supported: {', '.join(sorted(allowed_extensions))}"
        )

    # Don't allow writing into system directories
    _blocked_prefixes = ("/etc", "/usr", "/bin", "/sbin", "/var", "/sys", "/proc")
    for prefix in _blocked_prefixes:
        if str(p).startswith(prefix):
            raise ValueError(f"Writing to system directory '{prefix}' is not allowed.")

    # Ensure parent directory exists
    p.parent.mkdir(parents=True, exist_ok=True)

    return p


def get_available_backends() -> list[str]:
    """Get list of available DataFrame backends."""
    backends = []
    if HAS_PANDAS:
        backends.append("pandas")
    if HAS_POLARS:
        backends.append("polars")
    return backends


def save_dataframe_to_csv(df: Any, output_path: Path) -> None:
    """Save DataFrame to CSV in a backend-agnostic way."""
    if HAS_PANDAS and hasattr(df, "to_csv") and hasattr(df, "index"):
        df.to_csv(output_path, index=False)
    elif HAS_POLARS and hasattr(df, "write_csv"):
        df.write_csv(output_path)
    else:
        if HAS_PANDAS:
            if hasattr(df, "to_pandas"):
                df.to_pandas().to_csv(output_path, index=False)
            else:
                pd.DataFrame(df).to_csv(output_path, index=False)
        else:
            raise TypeError(f"Unsupported DataFrame type '{type(df).__name__}' for CSV export.")


def open_browser_conditionally(url: str) -> None:
    """Open browser only if not in testing mode."""
    if not TESTING_MODE:
        webbrowser.open(url)
    else:
        logger.debug(f"Browser opening suppressed in testing mode for: {url}")


def save_html_and_open(html_content: str, title: str, filename_prefix: str) -> str:
    """
    Wrap HTML content in a styled document, save to a file, and open in browser.

    Returns a user-facing message about the result.
    """
    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
        }}
        .table-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow-x: auto;
        }}
    </style>
</head>
<body>
    <div class="table-container">
        {html_content}
    </div>
</body>
</html>
"""
    html_filename = f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    html_path = Path.cwd() / html_filename

    if TESTING_MODE:
        return (
            f"HTML generated (file creation skipped during testing)\n\nFile location: {html_path}"
        )

    try:
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(full_html)
        open_browser_conditionally(f"file://{html_path}")
        return f"HTML saved and opened in default browser!\n\nFile location: {html_path}"
    except Exception as e:
        return f"Error saving HTML file: {str(e)}"


def load_dataframe_from_path(input_path: str, backend: str = "auto") -> Any:
    """Load DataFrame from file using specified backend or auto-detect."""
    p_path = validate_input_path(input_path)

    # Auto-detect backend
    if backend == "auto":
        if HAS_PANDAS:
            backend = "pandas"
        elif HAS_POLARS:
            backend = "polars"
        else:
            raise ImportError("No DataFrame library available. Install pandas or polars.")

    # Load with specified backend
    if backend == "pandas":
        if not HAS_PANDAS:
            raise ImportError("Pandas not available. Install with: pip install pandas")

        if p_path.suffix.lower() == ".csv":
            return pd.read_csv(p_path)
        elif p_path.suffix.lower() in [".xls", ".xlsx"]:
            return pd.read_excel(p_path, engine="openpyxl")
        elif p_path.suffix.lower() == ".parquet":
            return pd.read_parquet(p_path)
        elif p_path.suffix.lower() == ".json":
            return pd.read_json(p_path)
        elif p_path.suffix.lower() == ".jsonl":
            return pd.read_json(p_path, lines=True)
    elif backend == "polars":
        if not HAS_POLARS:
            raise ImportError("Polars not available. Install with: pip install polars")

        if p_path.suffix.lower() == ".csv":
            return pl.read_csv(p_path)
        elif p_path.suffix.lower() == ".parquet":
            return pl.read_parquet(p_path)
        elif p_path.suffix.lower() == ".json":
            return pl.read_json(p_path)
        elif p_path.suffix.lower() == ".jsonl":
            return pl.read_ndjson(p_path)
        elif p_path.suffix.lower() in [".xls", ".xlsx"]:
            if HAS_PANDAS:
                return pd.read_excel(p_path, engine="openpyxl")
            else:
                raise ValueError(
                    "Excel files require pandas. Install with: pip install pandas openpyxl"
                )
    else:
        raise ValueError(f"Unsupported backend: {backend}. Available: {get_available_backends()}")

    raise ValueError(
        f"Unsupported file type: {p_path.suffix}. Please use CSV, Excel, Parquet, JSON, or JSONL."
    )


def generate_python_code_for_validator(
    validator: pb.Validate, validator_id: str, df_path: Optional[str] = None
) -> str:
    """Generate Python code equivalent for reproducing the validation using fluent interface."""
    code_lines = [
        "# Generated Python code for Pointblank validation",
        "import pointblank as pb",
        "",
        "# Load your data",
    ]

    if df_path:
        code_lines.extend(
            [
                f"# Original file: {df_path}",
                f"df = pb.load_dataset('{df_path}')  # Adjust path as needed",
            ]
        )
    else:
        code_lines.extend(
            [
                "# Replace 'your_data.csv' with your actual data file",
                "df = pb.load_dataset('your_data.csv')",
            ]
        )

    validation_methods = []

    try:
        json_report = validator.get_json_report()
        validation_data = json.loads(json_report)

        for step in validation_data:
            assertion_type = step.get("assertion_type", "")
            column = step.get("column", "")
            values = step.get("values", None)

            if assertion_type == "rows_distinct":
                validation_methods.append("    .rows_distinct()")
            elif assertion_type == "col_vals_not_null":
                if column:
                    validation_methods.append(f"    .col_vals_not_null(columns='{column}')")
            elif assertion_type == "col_vals_between":
                if column and values and len(values) >= 2:
                    left, right = values[0], values[1]
                    validation_methods.append(
                        f"    .col_vals_between(columns='{column}', left={left}, right={right})"
                    )
            elif assertion_type == "col_vals_ge":
                if column and values is not None:
                    validation_methods.append(
                        f"    .col_vals_ge(columns='{column}', value={values})"
                    )
            elif assertion_type == "col_vals_gt":
                if column and values is not None:
                    validation_methods.append(
                        f"    .col_vals_gt(columns='{column}', value={values})"
                    )
            elif assertion_type == "col_vals_le":
                if column and values is not None:
                    validation_methods.append(
                        f"    .col_vals_le(columns='{column}', value={values})"
                    )
            elif assertion_type == "col_vals_lt":
                if column and values is not None:
                    validation_methods.append(
                        f"    .col_vals_lt(columns='{column}', value={values})"
                    )
            elif assertion_type == "col_vals_in_set":
                if column and values:
                    set_values = repr(values) if isinstance(values, list) else f"[{repr(values)}]"
                    validation_methods.append(
                        f"    .col_vals_in_set(columns='{column}', set={set_values})"
                    )
            elif assertion_type == "col_vals_regex":
                if column and values:
                    pattern = values if isinstance(values, str) else str(values)
                    validation_methods.append(
                        f"    .col_vals_regex(columns='{column}', pattern=r'{pattern}')"
                    )
            elif assertion_type == "col_exists":
                if column:
                    cols = repr(column) if isinstance(column, list) else f"'{column}'"
                    validation_methods.append(f"    .col_exists(columns={cols})")

    except Exception as e:
        validation_methods.append(f"    # Error reconstructing validation steps: {e}")
        validation_methods.append("    # Please manually add your validation steps")

    code_lines.extend(
        [
            "",
            "# Create validator, add validation steps, and interrogate",
            "validator = (",
            "    pb.Validate(df)",
        ]
    )
    code_lines.extend(validation_methods)
    code_lines.extend(
        ["    .interrogate()", ")", "", "# View HTML report", "validator.get_tabular_report()"]
    )

    return "\n".join(code_lines)


def generate_validation_report_html(validator: pb.Validate, validator_id: str) -> str:
    """
    Generate an HTML report table for validation results and save to file.
    Returns the file path.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pointblank_validation_report_{validator_id}_{timestamp}.html"
    file_path = Path.cwd() / filename

    if TESTING_MODE:
        logger.debug(f"Skipping HTML file generation during testing: {filename}")
        return str(file_path.resolve())

    gt_report = validator.get_tabular_report()
    html_content = gt_report.as_raw_html()

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    logger.info(f"Validation report HTML saved to: {file_path}")
    return str(file_path.resolve())


def clean_for_json_serialization(obj: Any) -> Any:
    """
    Recursively clean an object to ensure it can be JSON serialized by converting
    problematic values like NaN and infinity.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return str(obj)
        else:
            return obj
    elif isinstance(obj, dict):
        return {key: clean_for_json_serialization(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json_serialization(item) for item in obj]
    else:
        return obj


def get_supported_validations(validator):
    """Get the supported validation methods from a validator instance."""
    return {
        "col_vals_lt": validator.col_vals_lt,
        "col_vals_gt": validator.col_vals_gt,
        "col_vals_le": validator.col_vals_le,
        "col_vals_ge": validator.col_vals_ge,
        "col_vals_eq": validator.col_vals_eq,
        "col_vals_ne": validator.col_vals_ne,
        "col_vals_between": validator.col_vals_between,
        "col_vals_outside": validator.col_vals_outside,
        "col_vals_in_set": validator.col_vals_in_set,
        "col_vals_not_in_set": validator.col_vals_not_in_set,
        "col_vals_null": validator.col_vals_null,
        "col_vals_not_null": validator.col_vals_not_null,
        "col_vals_regex": validator.col_vals_regex,
        "col_vals_expr": validator.col_vals_expr,
        "col_count_match": validator.col_count_match,
        "col_exists": validator.col_exists,
        "rows_distinct": validator.rows_distinct,
        "rows_complete": validator.rows_complete,
        "row_count_match": validator.row_count_match,
        "conjointly": validator.conjointly,
        "col_schema_match": validator.col_schema_match,
    }


SUPPORTED_VALIDATION_TYPES = [
    "col_vals_lt",
    "col_vals_gt",
    "col_vals_le",
    "col_vals_ge",
    "col_vals_eq",
    "col_vals_ne",
    "col_vals_between",
    "col_vals_outside",
    "col_vals_in_set",
    "col_vals_not_in_set",
    "col_vals_null",
    "col_vals_not_null",
    "col_vals_regex",
    "col_vals_expr",
    "col_count_match",
    "col_exists",
    "rows_distinct",
    "rows_complete",
    "prompt",
    "row_count_match",
    "conjointly",
    "col_schema_match",
]
