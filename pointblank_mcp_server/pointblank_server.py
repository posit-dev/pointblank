import json
import logging
import math
import os
import sys
import uuid
import webbrowser
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Dict,
    Optional,
    Union,
)

from fastmcp import Context, FastMCP
from fastmcp.prompts.prompt import Message

import pointblank as pb

# Detect if we're running in a test environment
TESTING_MODE = (
    "pytest" in sys.modules
    or os.environ.get("PYTEST_CURRENT_TEST") is not None
    or os.environ.get("POINTBLANK_TESTING") == "true"
)

# Try to import Pandas, but make it optional
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False

# Try to import other DataFrame libraries
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    pl = None
    HAS_POLARS = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("pointblank_mcp_server.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info(f"MCP Server starting at {datetime.now()}")
logger.info(f"Available DataFrame backends: pandas={HAS_PANDAS}, polars={HAS_POLARS}")
logger.info("Core Pointblank visualization features available")


# Type alias for DataFrame: can be Pandas or Polars or other
if HAS_PANDAS:
    DataFrameType = pd.DataFrame
else:
    DataFrameType = Any  # Fallback to Any if pandas not available


# --- Lifespan Context: manage DataFrames and Validators ---
@dataclass
class AppContext:
    # Stores loaded DataFrames: {df_id: DataFrame}
    loaded_dataframes: Dict[str, Any] = field(default_factory=dict)
    # Stores active Pointblank Validators: {validator_id: Validate}
    active_validators: Dict[str, pb.Validate] = field(default_factory=dict)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    context = AppContext()
    yield context
    context.loaded_dataframes.clear()
    context.active_validators.clear()


# Create FastMCP instance with version-aware dependency handling
def _create_fastmcp_instance():
    """Create FastMCP instance with backwards compatibility for dependencies parameter."""
    try:
        # Try to get FastMCP version to determine if the `dependencies=` parameter is available
        import fastmcp

        version_str = getattr(fastmcp, "__version__", "0.0.0")

        # Parse version and assume format like "2.11.4"
        version_parts = version_str.split(".")
        if len(version_parts) >= 3:
            major, minor, patch = (
                int(version_parts[0]),
                int(version_parts[1]),
                int(version_parts[2]),
            )

            # For versions >=2.11.4, the `dependencies=` parameter is deprecated
            if (
                (major > 2)
                or (major == 2 and minor > 11)
                or (major == 2 and minor == 11 and patch >= 4)
            ):
                return FastMCP(
                    "PointblankMCP",
                    lifespan=app_lifespan,
                )

        # For older versions, use the `dependencies=` parameter
        return FastMCP(
            "PointblankMCP",
            lifespan=app_lifespan,
            dependencies=["pointblank", "openpyxl"],
        )

    except Exception:
        # Fallback: try without dependencies first (newer approach)
        # If that fails, try with dependencies (older approach)
        try:
            return FastMCP(
                "PointblankMCP",
                lifespan=app_lifespan,
            )
        except Exception:
            return FastMCP(
                "PointblankMCP",
                lifespan=app_lifespan,
                dependencies=["pointblank", "openpyxl"],
            )


mcp = _create_fastmcp_instance()


def _get_available_backends():
    """Get list of available DataFrame backends."""
    backends = []
    if HAS_PANDAS:
        backends.append("pandas")
    if HAS_POLARS:
        backends.append("polars")
    return backends


def _save_dataframe_to_csv(df: Any, output_path: Path) -> None:
    """Save DataFrame to CSV in a backend-agnostic way."""
    if HAS_PANDAS and hasattr(df, "to_csv") and hasattr(df, "index"):
        # Pandas DataFrame
        df.to_csv(output_path, index=False)
    elif HAS_POLARS and hasattr(df, "write_csv"):
        # Polars DataFrame
        df.write_csv(output_path)
    else:
        # Fallback: try to convert to pandas if available
        if HAS_PANDAS:
            if hasattr(df, "to_pandas"):
                # Polars to pandas conversion
                df.to_pandas().to_csv(output_path, index=False)
            else:
                # Try direct pandas constructor
                pd.DataFrame(df).to_csv(output_path, index=False)
        else:
            raise TypeError(f"Unsupported DataFrame type '{type(df).__name__}' for CSV export.")


def _open_browser_conditionally(url: str) -> None:
    """Open browser only if not in testing mode."""
    if not TESTING_MODE:
        webbrowser.open(url)
    else:
        logger.debug(f"Browser opening suppressed in testing mode for: {url}")


def _load_dataframe_from_path(input_path: str, backend: str = "auto") -> Any:
    """Load DataFrame from file using specified backend or auto-detect."""
    p_path = Path(input_path)
    if not p_path.exists():
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

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
            # Polars doesn't directly support Excel, fall back to pandas if available
            if HAS_PANDAS:
                return pd.read_excel(p_path, engine="openpyxl")
            else:
                raise ValueError(
                    "Excel files require pandas. Install with: pip install pandas openpyxl"
                )
    else:
        raise ValueError(f"Unsupported backend: {backend}. Available: {_get_available_backends()}")

    raise ValueError(
        f"Unsupported file type: {p_path.suffix}. Please use CSV, Excel, Parquet, JSON, or JSONL."
    )


def _generate_python_code_for_validator(
    validator: pb.Validate, validator_id: str, df_path: Optional[str] = None
) -> str:
    """
    Generate Python code equivalent for reproducing the validation using fluent interface.
    """
    # Start building the Python code
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

    # Get validation steps from the validator's method chain
    validation_methods = []

    try:
        # Access the validator's validation steps
        # Reconstruct based on the report data
        json_report = validator.get_json_report()
        validation_data = json.loads(json_report)

        for step in validation_data:
            assertion_type = step.get("assertion_type", "")
            column = step.get("column", "")
            values = step.get("values", None)
            inclusive = step.get("inclusive", None)

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
                    if isinstance(column, list):
                        cols = repr(column)
                    else:
                        cols = f"'{column}'"
                    validation_methods.append(f"    .col_exists(columns={cols})")

    except Exception as e:
        validation_methods.append(f"    # Error reconstructing validation steps: {e}")
        validation_methods.append("    # Please manually add your validation steps")

    # Build the fluent interface chain
    code_lines.extend(
        [
            "",
            "# Create validator, add validation steps, and interrogate",
            "validator = (",
            "    pb.Validate(df)",
        ]
    )

    # Add all validation methods
    code_lines.extend(validation_methods)

    # Close the chain with interrogate
    code_lines.extend(
        ["    .interrogate()", ")", "", "# View HTML report", "validator.get_tabular_report()"]
    )

    return "\n".join(code_lines)


def _generate_validation_report_html(validator: pb.Validate, validator_id: str) -> str:
    """
    Generate an HTML report table for validation results and save to file.
    Uses Pointblank's built-in get_tabular_report() and as_raw_html() methods.
    Returns the file path.
    """
    try:
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pointblank_validation_report_{validator_id}_{timestamp}.html"
        file_path = Path.cwd() / filename

        # Skip file generation during testing
        if TESTING_MODE:
            logger.debug(f"Skipping HTML file generation during testing: {filename}")
            return str(file_path.resolve())  # Return path but don't create file

        # Get the validation report as a GT table
        gt_report = validator.get_tabular_report()

        # Get the raw HTML from the GT table
        html_content = gt_report.as_raw_html()

        # Fix encoding issues with corrupted em dash characters
        # Use byte sequences to avoid encoding issues in the source code
        corrupted_sequences = [
            b"\xe2\x80\x94".decode("utf-8"),  # The â€" corruption pattern
            "\u2014",  # Unicode em dash
            "—",  # Literal em dash
        ]

        # Replace all problematic sequences with HTML entity
        for seq in corrupted_sequences:
            html_content = html_content.replace(seq, "&mdash;")

        # Ensure proper UTF-8 content
        if isinstance(html_content, bytes):
            html_content = html_content.decode("utf-8", errors="replace")

        # Save HTML file with explicit UTF-8 encoding
        with open(file_path, "w", encoding="utf-8", newline="") as f:
            f.write(html_content)

        logger.info(f"Validation report HTML saved to: {file_path}")
        return str(file_path.resolve())

    except Exception as e:
        logger.error(f"Error generating validation HTML report using get_tabular_report(): {e}")
        # If get_tabular_report() fails, we still need a fallback
        raise RuntimeError(f"Could not generate validation report HTML: {e}")


@dataclass
class DataFrameInfo:
    df_id: str
    shape: tuple
    columns: list


@mcp.tool(
    name="load_dataframe",
    description="Load a DataFrame from a CSV, Excel or Parquet file into the server's context.",
    tags={"Data Management"},
)
async def load_dataframe(
    ctx: Context,
    input_path: Annotated[str, "Path to the input CSV, Excel or Parquet file."],
    df_id: Optional[
        Annotated[
            str, "Optional ID for the DataFrame. If not provided, a new ID will be generated."
        ]
    ] = None,
    backend: Annotated[
        str,
        "DataFrame backend to use: 'auto', 'pandas', or 'polars'. Default is 'auto' (uses pandas if available, then polars).",
    ] = "auto",
) -> DataFrameInfo:
    """
    Loads a DataFrame from the specified CSV, Excel, or Parquet file into the server's context.
    Assigns a unique ID to the DataFrame for later reference.
    If df_id is not provided, a new one will be generated.
    Supports multiple DataFrame backends (pandas, polars) for flexibility.
    Returns the DataFrame ID and basic information (shape, columns).
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Check available backends
    available_backends = _get_available_backends()
    if not available_backends:
        raise RuntimeError("No DataFrame library available. Install pandas or polars.")

    # Inform user about available backends
    await ctx.report_progress(10, 100, f"Available backends: {', '.join(available_backends)}")

    df = _load_dataframe_from_path(input_path, backend)

    effective_df_id = df_id if df_id else f"df_{uuid.uuid4().hex[:8]}"

    if effective_df_id in app_ctx.loaded_dataframes:
        raise ValueError(
            f"DataFrame ID '{effective_df_id}' already exists. Choose a different ID or omit to generate a new one."
        )

    app_ctx.loaded_dataframes[effective_df_id] = df

    # Get DataFrame info in a backend-agnostic way
    shape = df.shape
    columns = list(df.columns)

    await ctx.report_progress(
        100, 100, f"DataFrame loaded with {backend} backend: {shape[0]} rows, {shape[1]} columns"
    )

    return DataFrameInfo(
        df_id=effective_df_id,
        shape=(int(shape[0]), int(shape[1])),  # Convert to Python ints
        columns=columns,
    )


@mcp.tool(
    name="list_available_backends",
    description="List available DataFrame backends (pandas, polars) installed in the environment.",
    tags={"Data Management"},
)
async def list_available_backends(ctx: Context) -> Dict[str, Any]:
    """
    Returns information about available DataFrame backends and their capabilities.
    """
    backends = _get_available_backends()

    backend_info = {}
    for backend in backends:
        if backend == "pandas":
            backend_info["pandas"] = {
                "available": True,
                "supports": ["CSV", "Excel", "Parquet", "JSON"],
                "excel_engine": "openpyxl" if HAS_PANDAS else None,
            }
        elif backend == "polars":
            backend_info["polars"] = {
                "available": True,
                "supports": ["CSV", "Parquet"],
                "notes": "Excel requires fallback to pandas",
            }

    if not backends:
        backend_info["warning"] = "No DataFrame backends available. Install pandas or polars."

    return {
        "available_backends": backends,
        "backend_details": backend_info,
        "recommendation": "pandas"
        if "pandas" in backends
        else ("polars" if "polars" in backends else "install pandas or polars"),
    }


@mcp.tool(
    name="list_loaded_dataframes",
    description="List all DataFrames currently loaded in the server context.",
    tags={"Data Management"},
)
async def list_loaded_dataframes(ctx: Context) -> Dict[str, Any]:
    """
    Returns information about all DataFrames currently loaded in the server.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    dataframes_info = {}
    for df_id, df in app_ctx.loaded_dataframes.items():
        try:
            shape = df.shape
            columns = list(df.columns)
            # Detect DataFrame type
            df_type = "pandas" if hasattr(df, "to_csv") and hasattr(df, "index") else "polars"

            dataframes_info[df_id] = {
                "shape": [int(shape[0]), int(shape[1])],  # Convert to Python ints
                "columns": columns,
                "column_count": len(columns),
                "row_count": int(shape[0]),  # Convert to Python int
                "backend": df_type,
                "memory_usage_mb": round(float(df.memory_usage(deep=True).sum()) / 1024 / 1024, 2)
                if df_type == "pandas"
                else "N/A",
            }
        except Exception as e:
            dataframes_info[df_id] = {
                "error": f"Failed to get info: {str(e)}",
                "backend": "unknown",
            }

    result = {
        "loaded_dataframes": dataframes_info,
        "total_count": len(dataframes_info),
        "memory_usage_summary": {
            "total_mb": sum(
                info.get("memory_usage_mb", 0)
                for info in dataframes_info.values()
                if isinstance(info.get("memory_usage_mb"), (int, float))
            )
        },
    }

    # Clean the result to ensure all numpy types are converted
    return clean_for_json_serialization(result)


@mcp.tool(
    name="analyze_data_quality",
    description="Analyze data quality using Pointblank's DataScan class.",
    tags={"Data Analysis"},
)
async def analyze_data_quality(
    ctx: Context,
    df_id: Annotated[str, "ID of the DataFrame to analyze."],
) -> Dict[str, Any]:
    """
    Analyze data quality using Pointblank's built-in DataScan functionality.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if df_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame ID '{df_id}' not found.")

    df = app_ctx.loaded_dataframes[df_id]

    await ctx.report_progress(20, 100, "Starting data quality analysis...")

    try:
        # Use Pointblank's DataScan class for profiling
        scanner = pb.DataScan(data=df)

        await ctx.report_progress(60, 100, "Running DataScan analysis...")

        # Get the JSON representation which properly handles numpy types
        profile_json = scanner.to_json()

        await ctx.report_progress(80, 100, "Processing results...")

        # Parse it back to a dictionary and clean any problematic values
        profile_results = json.loads(profile_json)
        cleaned_results = clean_for_json_serialization(profile_results)

        await ctx.report_progress(100, 100, "Data quality analysis complete!")

        return {"status": "success", "df_id": df_id, "analysis": cleaned_results}

    except Exception as e:
        error_msg = f"Error during data quality analysis: {str(e)}"
        await ctx.report_progress(100, 100, error_msg)
        raise ValueError(error_msg)


@mcp.tool(
    name="list_active_validators",
    description="List all validators currently active in the server context.",
    tags={"Validation"},
)
async def list_active_validators(ctx: Context) -> Dict[str, Any]:
    """
    Returns information about all active validators in the server.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    validators_info = {}
    for validator_id, validator in app_ctx.active_validators.items():
        try:
            # Get validator metadata
            table_name = getattr(validator, "tbl_name", "Unknown")
            label = getattr(validator, "label", "No label")

            # Check if interrogated
            is_interrogated = (
                hasattr(validator, "time_processed") and validator.time_processed is not None
            )

            # Count validation steps (this is a rough estimate)
            step_count = len(getattr(validator, "_validation_set", []))

            validators_info[validator_id] = {
                "table_name": table_name,
                "label": label,
                "is_interrogated": is_interrogated,
                "validation_steps_count": step_count,
                "last_processed": str(getattr(validator, "time_processed", "Never")),
            }
        except Exception as e:
            validators_info[validator_id] = {"error": f"Failed to get info: {str(e)}"}

    return {
        "active_validators": validators_info,
        "total_count": len(validators_info),
        "interrogated_count": sum(
            1 for info in validators_info.values() if info.get("is_interrogated", False)
        ),
    }


@mcp.tool(
    name="delete_dataframe",
    description="Remove a DataFrame from the server context to free up memory.",
    tags={"Data Management"},
)
async def delete_dataframe(
    ctx: Context,
    df_id: Annotated[str, "ID of the DataFrame to delete."],
) -> Dict[str, str]:
    """
    Removes a DataFrame from the server context and frees up memory.
    Also removes any validators that were using this DataFrame.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if df_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame ID '{df_id}' not found.")

    # Remove the DataFrame
    del app_ctx.loaded_dataframes[df_id]

    # Find and remove validators that might be using this DataFrame
    validators_to_remove = []
    for validator_id, validator in app_ctx.active_validators.items():
        # This is a heuristic as we can't easily determine which DataFrame a validator uses
        # In a more sophisticated implementation, we'd track this relationship
        try:
            if hasattr(validator, "tbl_name") and df_id in validator.tbl_name:
                validators_to_remove.append(validator_id)
        except Exception:
            pass

    removed_validators = 0
    for validator_id in validators_to_remove:
        del app_ctx.active_validators[validator_id]
        removed_validators += 1

    message = f"DataFrame '{df_id}' deleted successfully."
    if removed_validators > 0:
        message += f" Also removed {removed_validators} associated validator(s)."

    await ctx.report_progress(100, 100, message)

    return {"status": "success", "message": message, "removed_validators": removed_validators}


@mcp.tool(
    name="delete_validator",
    description="Remove a validator from the server context.",
    tags={"Validation"},
)
async def delete_validator(
    ctx: Context,
    validator_id: Annotated[str, "ID of the validator to delete."],
) -> Dict[str, str]:
    """
    Removes a validator from the server context.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if validator_id not in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{validator_id}' not found.")

    # Get validator info before deleting
    validator = app_ctx.active_validators[validator_id]
    table_name = getattr(validator, "tbl_name", "Unknown")

    # Remove the validator
    del app_ctx.active_validators[validator_id]

    message = f"Validator '{validator_id}' (table: {table_name}) deleted successfully."
    await ctx.report_progress(100, 100, message)

    return {"status": "success", "message": message}


def clean_for_json_serialization(obj: Any) -> Any:
    """
    Recursively clean an object to ensure it can be JSON serialized by converting
    problematic values like NaN and infinity.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        elif math.isinf(obj):
            return str(obj)  # "inf" or "-inf"
        else:
            return obj
    elif isinstance(obj, dict):
        return {key: clean_for_json_serialization(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [clean_for_json_serialization(item) for item in obj]
    else:
        return obj


@mcp.tool(
    name="test_simple_dataframe_access",
    description="Simple test to check if DataFrame access causes serialization issues.",
    tags={"Debug"},
)
async def test_simple_dataframe_access(
    ctx: Context,
    df_id: Annotated[str, "ID of the DataFrame to test."],
) -> Dict[str, Any]:
    """
    Simple test function to check DataFrame access.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if df_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame ID '{df_id}' not found.")

    df = app_ctx.loaded_dataframes[df_id]

    # Return minimal info without accessing df.shape or any other potentially problematic attributes
    return {
        "status": "success",
        "df_id": df_id,
        "columns": list(df.columns),
        "column_count": len(df.columns),
    }


@mcp.tool(
    name="profile_dataframe_original",
    description="Generate comprehensive data profiling insights for a loaded DataFrame.",
    tags={"Data Analysis"},
)
async def profile_dataframe(
    ctx: Context,
    df_id: Annotated[str, "ID of the DataFrame to profile."],
    include_correlations: Annotated[
        bool, "Whether to include correlation analysis for numeric columns."
    ] = True,
    sample_size: Annotated[
        int, "Maximum number of rows to sample for profiling (0 = all rows)."
    ] = 10000,
) -> Dict[str, Any]:
    """
    Generates comprehensive data profiling insights using Pointblank's DataScan class.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if df_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame ID '{df_id}' not found.")

    df = app_ctx.loaded_dataframes[df_id]

    await ctx.report_progress(10, 100, "Starting data profiling...")

    # Sample data if needed
    if sample_size > 0 and df.shape[0] > sample_size:
        if hasattr(df, "sample"):  # pandas
            df_sample = df.sample(n=sample_size, random_state=42)
        else:  # polars
            df_sample = df.sample(n=sample_size, seed=42)
        await ctx.report_progress(20, 100, f"Sampling {sample_size} rows for analysis...")
    else:
        df_sample = df

    await ctx.report_progress(50, 100, "Running Pointblank DataScan...")

    try:
        # Use Pointblank's DataScan class for profiling
        scanner = pb.DataScan(data=df_sample)

        await ctx.report_progress(80, 100, "Converting to JSON...")

        # Get the JSON representation which properly handles numpy types
        profile_json = scanner.to_json()

        # Parse it back to a dictionary and clean any problematic values
        profile_results = json.loads(profile_json)
        cleaned_results = clean_for_json_serialization(profile_results)

        await ctx.report_progress(100, 100, "Data profiling complete!")

        return cleaned_results

    except Exception as e:
        error_msg = f"Error during data profiling: {str(e)}"
        await ctx.report_progress(100, 100, error_msg)
        raise ValueError(error_msg)


def _profile_column(df: Any, column: str) -> Dict[str, Any]:
    """Profile a single column and return statistics."""
    try:
        col_data = df[column]

        # Basic info
        profile = {
            "dtype": str(col_data.dtype),
            "non_null_count": int(col_data.count()) if hasattr(col_data, "count") else "N/A",
            "null_count": int(col_data.isnull().sum()) if hasattr(col_data, "isnull") else "N/A",
            "unique_count": int(col_data.nunique()) if hasattr(col_data, "nunique") else "N/A",
        }

        # Add null percentage
        if isinstance(profile["null_count"], int) and len(col_data) > 0:
            profile["null_percentage"] = round(profile["null_count"] / len(col_data) * 100, 2)

        # Type-specific analysis
        if hasattr(col_data, "dtype"):
            if "int" in str(col_data.dtype) or "float" in str(col_data.dtype):
                # Numeric column
                profile.update(
                    {
                        "min": float(col_data.min()) if hasattr(col_data, "min") else None,
                        "max": float(col_data.max()) if hasattr(col_data, "max") else None,
                        "mean": float(col_data.mean()) if hasattr(col_data, "mean") else None,
                        "std": float(col_data.std()) if hasattr(col_data, "std") else None,
                        "quartiles": [
                            float(col_data.quantile(0.25))
                            if hasattr(col_data, "quantile")
                            else None,
                            float(col_data.quantile(0.5))
                            if hasattr(col_data, "quantile")
                            else None,
                            float(col_data.quantile(0.75))
                            if hasattr(col_data, "quantile")
                            else None,
                        ]
                        if hasattr(col_data, "quantile")
                        else None,
                    }
                )
            elif "object" in str(col_data.dtype) or "string" in str(col_data.dtype):
                # String column
                if hasattr(col_data, "str"):
                    try:
                        profile.update(
                            {
                                "avg_length": float(col_data.str.len().mean())
                                if hasattr(col_data.str, "len")
                                else None,
                                "min_length": int(col_data.str.len().min())
                                if hasattr(col_data.str, "len")
                                else None,
                                "max_length": int(col_data.str.len().max())
                                if hasattr(col_data.str, "len")
                                else None,
                            }
                        )
                    except Exception:
                        pass

                # Most common values
                try:
                    if hasattr(col_data, "value_counts"):
                        top_values = col_data.value_counts().head(5)
                        if hasattr(top_values, "to_dict"):
                            profile["top_values"] = top_values.to_dict()
                        else:
                            profile["top_values"] = dict(top_values)
                except Exception:
                    pass

        return profile
    except Exception as e:
        return {"error": str(e)}


def _generate_validation_suggestions(column: str, profile: Dict[str, Any]) -> list:
    """Generate suggested validation rules based on column profile."""
    suggestions = []

    try:
        # Null checks
        if profile.get("null_count", 0) == 0:
            suggestions.append(
                {
                    "validation_type": "col_vals_not_null",
                    "params": {"columns": column},
                    "reason": "Column has no null values",
                }
            )
        elif profile.get("null_percentage", 0) > 20:
            suggestions.append(
                {
                    "validation_type": "col_vals_null",
                    "params": {"columns": column},
                    "reason": f"High null percentage ({profile.get('null_percentage', 0)}%)",
                }
            )

        # Numeric validations
        if profile.get("min") is not None and profile.get("max") is not None:
            min_val = profile["min"]
            max_val = profile["max"]

            # Range validation
            suggestions.append(
                {
                    "validation_type": "col_vals_between",
                    "params": {"columns": column, "left": min_val, "right": max_val},
                    "reason": f"Values range from {min_val} to {max_val}",
                }
            )

            # Positive values check
            if min_val >= 0:
                suggestions.append(
                    {
                        "validation_type": "col_vals_ge",
                        "params": {"columns": column, "value": 0},
                        "reason": "All values are non-negative",
                    }
                )

        # String validations
        if "top_values" in profile and len(profile["top_values"]) <= 10:
            # With a limited set of values suggest `in_set` validation
            values_list = list(profile["top_values"].keys())
            suggestions.append(
                {
                    "validation_type": "col_vals_in_set",
                    "params": {"columns": column, "set_": values_list},
                    "reason": f"Column has limited distinct values: {values_list[:3]}...",
                }
            )

        # Length validations for strings
        if profile.get("min_length") is not None and profile.get("max_length") is not None:
            min_len = profile["min_length"]
            max_len = profile["max_length"]
            if min_len > 0:
                suggestions.append(
                    {
                        "validation_type": "col_vals_expr",
                        "params": {"columns": column, "expr": f"_.str.len() >= {min_len}"},
                        "reason": f"Minimum string length is {min_len}",
                    }
                )

    except Exception:
        pass

    return suggestions


def _detect_data_quality_issues(df: Any, column_profiles: Dict[str, Any]) -> list:
    """Detect potential data quality issues."""
    issues = []

    try:
        total_rows = df.shape[0]

        for col, profile in column_profiles.items():
            if "error" in profile:
                issues.append(
                    {
                        "column": col,
                        "issue_type": "analysis_error",
                        "severity": "warning",
                        "description": f"Could not analyze column: {profile['error']}",
                    }
                )
                continue

            # High null percentage
            null_pct = profile.get("null_percentage", 0)
            if null_pct > 50:
                issues.append(
                    {
                        "column": col,
                        "issue_type": "high_null_rate",
                        "severity": "critical",
                        "description": f"Column has {null_pct}% null values",
                    }
                )
            elif null_pct > 20:
                issues.append(
                    {
                        "column": col,
                        "issue_type": "moderate_null_rate",
                        "severity": "warning",
                        "description": f"Column has {null_pct}% null values",
                    }
                )

            # Low cardinality (potential categorical)
            unique_count = profile.get("unique_count")
            if unique_count is not None and unique_count < 10 and total_rows > 100:
                issues.append(
                    {
                        "column": col,
                        "issue_type": "low_cardinality",
                        "severity": "info",
                        "description": f"Column has only {unique_count} unique values (potential categorical)",
                    }
                )

            # High cardinality (potential identifier)
            if unique_count is not None and unique_count > total_rows * 0.9:
                issues.append(
                    {
                        "column": col,
                        "issue_type": "high_cardinality",
                        "severity": "info",
                        "description": f"Column has {unique_count}/{total_rows} unique values (potential identifier)",
                    }
                )

            # Constant column
            if unique_count == 1:
                issues.append(
                    {
                        "column": col,
                        "issue_type": "constant_column",
                        "severity": "warning",
                        "description": "Column has only one unique value",
                    }
                )

    except Exception as e:
        issues.append(
            {
                "column": "general",
                "issue_type": "analysis_error",
                "severity": "error",
                "description": f"Error during data quality analysis: {str(e)}",
            }
        )

    return issues


def _compute_correlations(df: Any) -> Dict[str, Any]:
    """Compute correlation matrix for numeric columns."""
    try:
        # Get numeric columns
        if hasattr(df, "select_dtypes"):  # Pandas
            numeric_df = df.select_dtypes(include=["number"])
        else:  # Polars: basic approach
            numeric_cols = [
                col
                for col in df.columns
                if "int" in str(df[col].dtype) or "float" in str(df[col].dtype)
            ]
            numeric_df = df.select(numeric_cols) if numeric_cols else None

        if numeric_df is None or numeric_df.shape[1] < 2:
            return {"message": "Not enough numeric columns for correlation analysis"}

        # Compute correlation matrix
        if hasattr(numeric_df, "corr"):  # Pandas
            corr_matrix = numeric_df.corr()

            # Find strong correlations
            strong_correlations = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i + 1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_value = corr_matrix.iloc[i, j]

                    if abs(corr_value) > 0.7:  # Strong correlation threshold
                        strong_correlations.append(
                            {
                                "column1": col1,
                                "column2": col2,
                                "correlation": round(float(corr_value), 3),
                                "strength": "strong" if abs(corr_value) > 0.8 else "moderate",
                            }
                        )

            return {
                "correlation_matrix": corr_matrix.round(3).to_dict(),
                "strong_correlations": strong_correlations,
                "numeric_columns": list(numeric_df.columns),
            }
        else:
            return {"message": "Correlation analysis not available for this DataFrame type"}

    except Exception as e:
        return {"error": str(e)}


@mcp.tool(
    name="apply_validation_template",
    description="Apply a pre-built validation template to a validator for common data quality patterns.",
    tags={"Validation"},
)
async def apply_validation_template(
    ctx: Context,
    validator_id: Annotated[str, "ID of the Validator to apply template to."],
    template_name: Annotated[
        str,
        "Name of the validation template to apply. Available: 'basic_quality', 'financial_data', 'customer_data', 'sensor_data', 'survey_data'.",
    ],
    column_mapping: Annotated[
        Dict[str, str],
        "Mapping of template column names to actual DataFrame column names. E.g., {'amount': 'price', 'id': 'customer_id'}.",
    ],
    thresholds: Annotated[
        Optional[Dict[str, float]],
        "Optional custom thresholds for validation steps. E.g., {'warning': 0.05, 'error': 0.1}.",
    ] = None,
) -> Dict[str, Any]:
    """
    Applies a pre-built validation template with common data quality checks.
    Templates include ready-made validation rules for typical data scenarios.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if validator_id not in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{validator_id}' not found.")

    # Get validation template
    template = _get_validation_template(template_name)
    if not template:
        available_templates = [
            "basic_quality",
            "financial_data",
            "customer_data",
            "sensor_data",
            "survey_data",
        ]
        raise ValueError(f"Unknown template '{template_name}'. Available: {available_templates}")

    validator = app_ctx.active_validators[validator_id]
    applied_validations = []

    await ctx.report_progress(10, 100, f"Applying {template_name} template...")

    # Apply each validation in the template
    for i, validation_rule in enumerate(template["validations"]):
        try:
            # Map template column names to actual column names
            mapped_params = {}
            for key, value in validation_rule["params"].items():
                if key == "columns" and value in column_mapping:
                    mapped_params[key] = column_mapping[value]
                elif isinstance(value, str) and value in column_mapping:
                    mapped_params[key] = column_mapping[value]
                else:
                    mapped_params[key] = value

            # Add custom thresholds if provided
            if thresholds:
                mapped_params.update(thresholds)

            # Get the validation method
            validation_type = validation_rule["validation_type"]
            supported_validations = _get_supported_validations(validator)

            if validation_type in supported_validations:
                validation_method = supported_validations[validation_type]
                validation_method(**mapped_params)

                applied_validations.append(
                    {
                        "validation_type": validation_type,
                        "params": mapped_params,
                        "description": validation_rule.get("description", ""),
                    }
                )

                await ctx.report_progress(
                    10 + (i + 1) * 80 // len(template["validations"]),
                    100,
                    f"Applied {validation_type}...",
                )
            else:
                applied_validations.append(
                    {
                        "validation_type": validation_type,
                        "params": mapped_params,
                        "error": f"Unsupported validation type: {validation_type}",
                    }
                )

        except Exception as e:
            applied_validations.append(
                {
                    "validation_type": validation_rule["validation_type"],
                    "params": validation_rule["params"],
                    "error": str(e),
                }
            )

    await ctx.report_progress(100, 100, f"Template {template_name} applied successfully!")

    return {
        "template_name": template_name,
        "template_description": template["description"],
        "applied_validations": applied_validations,
        "total_validations": len(applied_validations),
        "successful_validations": len([v for v in applied_validations if "error" not in v]),
    }


def _get_validation_template(template_name: str) -> Optional[Dict[str, Any]]:
    """Get a predefined validation template."""
    templates = {
        "basic_quality": {
            "description": "Basic data quality checks for any dataset",
            "validations": [
                {
                    "validation_type": "col_exists",
                    "params": {"columns": "id"},
                    "description": "Check that ID column exists",
                },
                {
                    "validation_type": "col_vals_not_null",
                    "params": {"columns": "id"},
                    "description": "ID column should not have null values",
                },
                {
                    "validation_type": "rows_distinct",
                    "params": {},
                    "description": "Check for duplicate rows",
                },
            ],
        },
        "financial_data": {
            "description": "Validation template for financial/transaction data",
            "validations": [
                {
                    "validation_type": "col_vals_not_null",
                    "params": {"columns": "amount"},
                    "description": "Amount should not be null",
                },
                {
                    "validation_type": "col_vals_gt",
                    "params": {"columns": "amount", "value": 0},
                    "description": "Amount should be positive",
                },
                {
                    "validation_type": "col_vals_not_null",
                    "params": {"columns": "date"},
                    "description": "Transaction date should not be null",
                },
                {
                    "validation_type": "col_vals_not_null",
                    "params": {"columns": "account_id"},
                    "description": "Account ID should not be null",
                },
            ],
        },
        "customer_data": {
            "description": "Validation template for customer/user data",
            "validations": [
                {
                    "validation_type": "col_vals_not_null",
                    "params": {"columns": "customer_id"},
                    "description": "Customer ID should not be null",
                },
                {
                    "validation_type": "col_vals_regex",
                    "params": {"columns": "email", "pattern": r"^[^@]+@[^@]+\.[^@]+$"},
                    "description": "Email should be in valid format",
                },
                {
                    "validation_type": "col_vals_between",
                    "params": {"columns": "age", "left": 0, "right": 120},
                    "description": "Age should be between 0 and 120",
                },
                {
                    "validation_type": "col_vals_in_set",
                    "params": {"columns": "status", "set_": ["active", "inactive", "pending"]},
                    "description": "Status should be one of predefined values",
                },
            ],
        },
        "sensor_data": {
            "description": "Validation template for IoT/sensor data",
            "validations": [
                {
                    "validation_type": "col_vals_not_null",
                    "params": {"columns": "timestamp"},
                    "description": "Timestamp should not be null",
                },
                {
                    "validation_type": "col_vals_not_null",
                    "params": {"columns": "sensor_id"},
                    "description": "Sensor ID should not be null",
                },
                {
                    "validation_type": "col_vals_between",
                    "params": {"columns": "temperature", "left": -50, "right": 100},
                    "description": "Temperature should be in reasonable range",
                },
                {
                    "validation_type": "col_vals_ge",
                    "params": {"columns": "humidity", "value": 0},
                    "description": "Humidity should be non-negative",
                },
                {
                    "validation_type": "col_vals_le",
                    "params": {"columns": "humidity", "value": 100},
                    "description": "Humidity should not exceed 100%",
                },
            ],
        },
        "survey_data": {
            "description": "Validation template for survey/questionnaire data",
            "validations": [
                {
                    "validation_type": "col_vals_not_null",
                    "params": {"columns": "response_id"},
                    "description": "Response ID should not be null",
                },
                {
                    "validation_type": "col_vals_between",
                    "params": {"columns": "satisfaction_score", "left": 1, "right": 10},
                    "description": "Satisfaction score should be between 1 and 10",
                },
                {
                    "validation_type": "col_vals_in_set",
                    "params": {
                        "columns": "completion_status",
                        "set_": ["complete", "partial", "abandoned"],
                    },
                    "description": "Completion status should be one of predefined values",
                },
            ],
        },
    }

    return templates.get(template_name)


def _get_supported_validations(validator):
    """Get the supported validation methods from a validator instance."""
    return {
        # Column value validations
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
        # Check existence of a column
        "col_exists": validator.col_exists,
        # Row validations
        "rows_distinct": validator.rows_distinct,
        "rows_complete": validator.rows_complete,
        "row_count_match": validator.row_count_match,
        # Other specialized validations
        "conjointly": validator.conjointly,
        "col_schema_match": validator.col_schema_match,
    }


@mcp.tool(
    name="server_health_check",
    description="Get comprehensive server health and status information.",
    tags={"Server Management"},
)
async def server_health_check(ctx: Context) -> Dict[str, Any]:
    """
    Returns comprehensive server health information including:
    - resource usage and capacity
    - backend availability
    - active resources count
    - system information
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    # Get current time
    current_time = datetime.now().isoformat()

    # Count resources
    dataframes_count = len(app_ctx.loaded_dataframes)
    validators_count = len(app_ctx.active_validators)

    # Calculate memory usage
    total_memory_mb = 0
    dataframe_details = []

    for df_id, df in app_ctx.loaded_dataframes.items():
        try:
            if hasattr(df, "memory_usage"):  # pandas
                memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
                total_memory_mb += memory_mb
                dataframe_details.append(
                    {
                        "df_id": df_id,
                        "shape": [int(df.shape[0]), int(df.shape[1])],  # Convert to Python ints
                        "memory_mb": round(memory_mb, 2),
                        "backend": "pandas",
                    }
                )
            else:  # polars or other
                # Estimate memory for polars (rough approximation)
                estimated_mb = (
                    df.shape[0] * df.shape[1] * 8 / 1024 / 1024
                )  # 8 bytes per value estimate
                total_memory_mb += estimated_mb
                dataframe_details.append(
                    {
                        "df_id": df_id,
                        "shape": [int(df.shape[0]), int(df.shape[1])],  # Convert to Python ints
                        "memory_mb": round(estimated_mb, 2),
                        "backend": "polars/other",
                    }
                )
        except Exception as e:
            dataframe_details.append({"df_id": df_id, "error": str(e), "backend": "unknown"})

    # Get system information
    import platform
    import sys

    health_info = {
        "timestamp": current_time,
        "server_status": "healthy",
        "system_info": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
        },
        "backend_status": {
            "pandas_available": HAS_PANDAS,
            "polars_available": HAS_POLARS,
            "fastmcp_loaded": True,  # If we're running, FastMCP is loaded
            "pointblank_loaded": True,  # If we're running, Pointblank is loaded
        },
        "resource_usage": {
            "total_dataframes": dataframes_count,
            "total_validators": validators_count,
            "total_memory_mb": round(total_memory_mb, 2),
            "dataframe_details": dataframe_details,
        },
        "capabilities": {
            "supported_file_formats": ["CSV", "JSON", "JSONL", "Parquet"]
            + (["Excel"] if HAS_PANDAS else []),
            "validation_types_count": len(_get_supported_validations_list()),
            "templates_available": [
                "basic_quality",
                "financial_data",
                "customer_data",
                "sensor_data",
                "survey_data",
            ],
        },
    }

    # Add warnings if any
    warnings = []
    if not HAS_PANDAS and not HAS_POLARS:
        warnings.append("No DataFrame backends available")
        health_info["server_status"] = "degraded"

    if total_memory_mb > 1000:  # > 1GB
        warnings.append(f"High memory usage: {total_memory_mb:.1f}MB")

    if dataframes_count > 50:
        warnings.append(f"Many DataFrames loaded: {dataframes_count}")

    if warnings:
        health_info["warnings"] = warnings

    logger.info(
        f"Health check performed: {dataframes_count} DataFrames, {validators_count} validators, {total_memory_mb:.1f}MB memory"
    )

    return health_info


def _get_supported_validations_list():
    """Get list of supported validation types."""
    return [
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


@dataclass
class ValidatorInfo:
    validator_id: str


@mcp.tool(
    name="create_validator",
    description="Create a Pointblank Validator for a previously loaded DataFrame.",
    tags={"Validation"},
)
def create_validator(
    ctx: Context,
    df_id: Annotated[str, "ID of the DataFrame to validate."],
    validator_id: Annotated[
        Optional[str],
        "Optional ID for the Validator. If not provided, a new ID will be generated.",
    ] = None,
    table_name: Annotated[
        Optional[str],
        "Optional name for the table within Pointblank reports. If not provided, a default name will be used.",
    ] = None,
    validator_label: Annotated[
        Optional[str],
        "Optional descriptive label for the Validator. If not provided, a default label will be used.",
    ] = None,  # Corresponds to 'label' in pb.Validate
    thresholds_dict: Annotated[
        Optional[Dict[str, Union[int, float]]],
        "Optional thresholds for validation failures. Example: {'warning': 0.1, 'error': 5, 'critical': 0.10}. "
        "If not provided, no thresholds will be set.",
    ] = None,  # Corresponds to 'thresholds' in pb.Validate, e.g. {"warning": 1, "error": 20, "critical": 0.10}
    actions_dict: Optional[Dict[str, Any]] = None,  # Simplified, for pb.Actions
    final_actions_dict: Optional[Dict[str, Any]] = None,  # Simplified, for pb.FinalActions
    brief: Optional[bool] = None,
    lang: Optional[str] = None,
    locale: Optional[str] = None,
) -> ValidatorInfo:
    """
    Creates a Pointblank Validator for a previously loaded DataFrame.
    Assigns a unique ID to the Validator for adding validation steps.
    If validator_id is not provided, a new one will be generated.
    'df_id' must refer to a DataFrame loaded via 'load_dataframe'.
    'table_name' is an optional name for the table within Pointblank reports.
    'validator_label' is an optional descriptive label for the validator.
    'thresholds_dict' can be like {"warning": 0.1, "error": 5} to set failure thresholds.
    Returns the Validator ID.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if df_id not in app_ctx.loaded_dataframes:
        raise ValueError(
            f"DataFrame ID '{df_id}' not found. Please load it first using 'load_dataframe'."
        )

    df = app_ctx.loaded_dataframes[df_id]

    effective_validator_id = validator_id if validator_id else f"validator_{uuid.uuid4().hex[:8]}"

    if effective_validator_id in app_ctx.active_validators:
        raise ValueError(
            f"Validator ID '{effective_validator_id}' already exists. Choose a different ID or omit to generate a new one."
        )

    actual_table_name = table_name if table_name else f"table_for_{df_id}"
    actual_validator_label = (
        validator_label if validator_label else f"Validation for {actual_table_name}"
    )

    # Construct Thresholds, Actions, FinalActions if dicts are provided
    pb_thresholds = None
    if thresholds_dict:
        try:
            pb_thresholds = pb.Thresholds(**thresholds_dict)
        except Exception as e:
            raise ValueError(f"Error creating pb.Thresholds from thresholds_dict: {e}")

    # Note: pb.Actions and pb.FinalActions might require more complex construction
    # For simplicity, we're assuming direct kwarg passing or simple structures.
    # This part might need refinement based on how pb.Actions/pb.FinalActions are instantiated.
    pb_actions = None
    if actions_dict:
        try:
            # Example: if pb.Actions takes specific function handlers
            # This is a placeholder and likely needs more specific handling
            pb_actions = pb.Actions(
                **actions_dict
            )  # This assumes pb.Actions can be created this way
        except Exception as e:
            print(f"Could not create pb.Actions from actions_dict: {e}. Passing None.")

    pb_final_actions = None
    if final_actions_dict:
        try:
            pb_final_actions = pb.FinalActions(**final_actions_dict)  # Placeholder
        except Exception as e:
            print(f"Could not create pb.FinalActions from final_actions_dict: {e}. Passing None.")

    validator_instance_params = {
        "data": df,
        "tbl_name": actual_table_name,
        "label": actual_validator_label,
    }

    if pb_thresholds:
        validator_instance_params["thresholds"] = pb_thresholds
    if pb_actions:
        validator_instance_params["actions"] = pb_actions
    if pb_final_actions:
        validator_instance_params["final_actions"] = pb_final_actions
    if brief is not None:
        validator_instance_params["brief"] = brief
    if lang:
        validator_instance_params["lang"] = lang
    if locale:
        validator_instance_params["locale"] = locale

    validator_instance = pb.Validate(**validator_instance_params)
    app_ctx.active_validators[effective_validator_id] = validator_instance

    return ValidatorInfo(validator_id=effective_validator_id)


@dataclass
class ValidationStepInfo:
    validator_id: str
    status: str


@mcp.tool(
    name="add_validation_step",
    description="Add a validation step to an existing Pointblank Validator.",
    tags={"Validation"},
)
def add_validation_step(
    ctx: Context,
    validator_id: Annotated[str, "ID of the Validator to add a step to."],
    validation_type: Annotated[
        str,
        "Type of validation to perform. Supported types include: 'col_vals_lt', 'col_vals_gt', 'col_vals_between', 'col_exists', 'rows_distinct', etc.",
    ],
    params: Annotated[
        Dict[str, Any],
        "Parameters for the validation function. This should match the expected parameters for the Pointblank validation method.",
    ],
    actions_config: Optional[Dict[str, Any]] = None,  # Placeholder for simplified action definition
) -> ValidationStepInfo:
    """
    Adds a validation step to an existing Pointblank Validator.
    'validator_id' must refer to a validator created via 'create_validator'.
    'validation_type' specifies the Pointblank validation function to call
      (e.g., 'col_vals_lt', 'col_vals_between', 'col_vals_in_set', 'col_exists', 'rows_distinct').
    'params' is a dictionary of parameters for that validation function.
    'actions_config' (optional) can be used to define simple actions (currently basic support).
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if validator_id not in app_ctx.active_validators:
        raise ValueError(
            f"Validator ID '{validator_id}' not found. Please create it first using 'create_validator'."
        )

    validator = app_ctx.active_validators[validator_id]

    # --- Define supported validation types and their methods from pb.Validate ---
    # This mapping allows dynamic dispatch and can be extended
    # Methods are called on the 'validator' (pb.Validate instance)
    supported_validations = {
        # Column value validations
        "col_vals_lt": validator.col_vals_lt,  # less than a value
        "col_vals_gt": validator.col_vals_gt,  # greater than a value
        "col_vals_le": validator.col_vals_le,  # less or equal
        "col_vals_ge": validator.col_vals_ge,  # greater or equal
        "col_vals_eq": validator.col_vals_eq,  # equal to a value
        "col_vals_ne": validator.col_vals_ne,  # not equal to a value
        "col_vals_between": validator.col_vals_between,  # data lies between two values left=val, right=val
        "col_vals_outside": validator.col_vals_outside,  # data is outside two values
        "col_vals_in_set": validator.col_vals_in_set,  # values in a set e.g. [1,2,3]
        "col_vals_not_in_set": validator.col_vals_not_in_set,  # values not in a set
        "col_vals_null": validator.col_vals_null,  # null values
        "col_vals_not_null": validator.col_vals_not_null,  # not null values
        "col_vals_regex": validator.col_vals_regex,  # values match a regular expresion
        "col_vals_expr": validator.col_vals_expr,  # Validate column values using a custom expression
        "col_count_match": validator.col_count_match,  # Validate whether the column count of the table matches a specified count.
        # Check existence of a column
        "col_exists": validator.col_exists,
        # Row validations
        "rows_distinct": validator.rows_distinct,  # check distinct rows in a table
        "rows_complete": validator.rows_complete,  # check for no nulls in rows across specified columns
        "prompt": validator.prompt,  # AI-powered validation of rows using LLMs
        "row_count_match": validator.row_count_match,  # check if number of rows in the table matches a fixed value
        # Other specialized validations
        "conjointly": validator.conjointly,  # For multiple column conditions
        "col_schema_match": validator.col_schema_match,  # Do columns in the table (and their types) match a predefined schema?
    }

    if validation_type not in supported_validations:
        raise ValueError(
            f"Unsupported validation_type: '{validation_type}'. Supported types include: {list(supported_validations.keys())}"
        )

    validation_method = supported_validations[validation_type]

    # Simplified actions handling (can be expanded)
    # pb.Validate methods expect an 'actions' parameter which is an instance of pb.Actions
    # This is a placeholder for how one might construct it.
    # A more robust solution would deserialize a dict into pb.Actions object.
    current_params = {**params}

    # Handle parameter name mapping for reserved keywords
    # Pointblank uses 'set' but we use 'set_' in JSON to avoid Python reserved keyword issues
    if "set_" in current_params:
        current_params["set"] = current_params.pop("set_")

    if actions_config:
        # Example: actions_config = {"warn": 0.1} might translate to
        # actions = pb.Actions(warn_fraction=0.1)
        # For now, if a method expects 'actions', it should be in params directly
        # or handled here explicitly if simple shorthands are desired.
        # This is a complex area to generalize perfectly via JSON.
        # Let's assume 'actions' if needed is part of 'params' and is a pb.Actions object
        # or the LLM constructs the params for methods that take thresholds directly.
        # For now, we'll pass 'params' as is.
        # If 'actions' is a direct parameter of the validation_method, it should be in 'params'.
        pass  # No special action processing here yet, assuming 'params' has all needed args
    try:
        validation_method(**current_params)
    except TypeError as e:
        raise ValueError(
            f"Error calling validation method '{validation_type}' with params {current_params}. Original error: {e}. Check parameter names and types against Pointblank's API for the '{validation_type}' method of the 'Validate' class."
        )
    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred while adding validation step '{validation_type}': {e}"
        )

    return ValidationStepInfo(
        validator_id=validator_id,
        status=f"Validation step '{validation_type}' added successfully.",
    )


@dataclass
class ValidationOutput:
    status: str
    message: str
    output_file: Optional[str] = None


@mcp.tool(
    name="get_validation_step_output",
    description="Retrieve output for a validation step and save it to a CSV file.",
    tags={"Validation"},
)
async def get_validation_step_output(
    ctx: Context,
    validator_id: Annotated[str, "ID of the Validator to retrieve output from."],
    output_path: Annotated[
        str,
        "Path to save the output file. Must end with .csv.",
    ],
    sundered_type: Annotated[
        str,
        "Mode 2: Retrieve all 'pass' or 'fail' rows for the *entire* validation run. Only used if 'step_index' is not provided.",
    ] = "fail",
    step_index: Annotated[
        Optional[int],
        "Mode 1: Retrieve data for a *specific* step by its index (starting from 0). If used, 'sundered_type' is ignored.",
    ] = None,
) -> ValidationOutput:
    """
    Retrieves validation output and saves it to a CSV file. This function has two modes:
    1.  Specific Step Extract: Provide a 'step_index' to get the data extract (e.g., failing rows) for that specific step.
    2.  Overall Sundered Data: Omit 'step_index' and use 'sundered_type' ('pass' or 'fail') to get all rows that met that condition across all validation steps.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if validator_id not in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{validator_id}' not found.")
    validator = app_ctx.active_validators[validator_id]

    p_output_path = Path(output_path)
    if p_output_path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file format '{p_output_path.suffix}'. Please use '.csv'.")

    if step_index is not None and step_index < 0:
        raise ValueError("The 'step_index' cannot be a negative number.")

    try:
        if not getattr(validator, "time_processed", None):
            await ctx.warning(
                f"Validator '{validator_id}' has not been interrogated. Interrogating now."
            )

        p_output_path.parent.mkdir(parents=True, exist_ok=True)
        message = ""
        data_extract_df = None

        # Pathway 1: Get data for a single, specific validation step.
        if step_index is not None:
            data_extract_df = validator.get_data_extracts(i=step_index, frame=True)
            if data_extract_df is None or data_extract_df.empty:
                message = f"No data extract available for step {step_index}. This may mean all rows passed this validation step."
                data_extract_df = None  # Ensure it's None if empty
            else:
                message = f"Data extract for step {step_index} retrieved."

        # Pathway 2: Get all 'fail' or 'pass' data from the entire validation run.
        else:
            data_extract_df = validator.get_sundered_data(type=sundered_type)
            if data_extract_df is None or data_extract_df.empty:
                message = f"No sundered data available for type '{sundered_type}'."
                data_extract_df = None  # Ensure it's None if empty
            else:
                message = f"Sundered data for type '{sundered_type}' retrieved."

        if data_extract_df is None:
            return ValidationOutput(
                status="success",
                message=message,
                output_file=None,
            )

        # Save to CSV using backend-agnostic method
        _save_dataframe_to_csv(data_extract_df, p_output_path)
        message = f"Data extract saved to {p_output_path.resolve()}"

        await ctx.report_progress(100, 100, message)

        return ValidationOutput(
            status="success",
            message=message,
            output_file=str(p_output_path.resolve()),
        )

    except Exception as e:
        raise RuntimeError(f"Error getting output for validator '{validator_id}': {e}")


@mcp.tool(
    name="interrogate_validator",
    description="Run validations and return a JSON summary with Python code equivalent.",
    tags={"Validation"},
)
async def interrogate_validator(
    ctx: Context,
    validator_id: Annotated[str, "ID of the Validator to interrogate."],
) -> Dict[str, Any]:
    """
    Runs validations and returns a JSON summary.
    Also generates an HTML report table that opens in browser.
    Provides Python code equivalent for reproducing the validation.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if validator_id not in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{validator_id}' not found.")

    validator = app_ctx.active_validators[validator_id]

    try:
        validator.interrogate()
        json_report_str = validator.get_json_report()

        # Generate HTML report table and open in browser
        try:
            html_report_path = _generate_validation_report_html(validator, validator_id)
            _open_browser_conditionally(f"file://{html_report_path}")
            await ctx.report_progress(
                50, 100, f"Validation report opened in browser: {html_report_path}"
            )
        except Exception as html_error:
            logger.warning(f"Could not generate HTML report: {html_error}")

        # Generate Python code equivalent
        try:
            python_code = _generate_python_code_for_validator(validator, validator_id)
            await ctx.report_progress(75, 100, "Generated Python code equivalent for validation")
        except Exception as code_error:
            logger.warning(f"Could not generate Python code: {code_error}")
            python_code = "# Error generating Python code equivalent"

    except Exception as e:
        raise RuntimeError(f"Error during validator interrogation: {e}")

    report_data = json.loads(json_report_str)

    # Enhanced output with Python code
    output_dict = {
        "validation_summary": report_data,
        "python_code": python_code,
        "html_report_path": html_report_path if "html_report_path" in locals() else None,
        "instructions": {
            "html_report": "Interactive validation report opened in your browser",
            "python_code": "Use the provided Python code to reproduce this validation in your own scripts",
            "customization": "Modify the Python code to adjust validation rules, thresholds, or add new checks",
        },
    }

    await ctx.report_progress(100, 100, "Validation complete! Check browser for detailed report.")

    return output_dict


@mcp.prompt(
    name="prompt_load_dataframe",
    description="Prompt to load a DataFrame from a file into the server's context for validation.",
    tags={"Data Management"},
)
def prompt_load_dataframe(
    input_path: str = "Path to the input CSV, Excel or Parquet file.",
    df_id: Optional[
        str
    ] = "Optional ID for the DataFrame. If not provided, a new ID will be generated.",
) -> tuple:
    return (
        Message(
            "I can load your data from a file into my context for validation.",
            role="assistant",
        ),
        Message(
            f"Please call `load_dataframe` with input_path='{input_path}'. "
            f"You can optionally provide a `df_id` (e.g., '{df_id}') to name this dataset, "
            "or I will generate one for you. Make a note of the returned `df_id` for subsequent steps.",
            role="user",
        ),
    )


@mcp.prompt(
    name="prompt_create_validator",
    description="Prompt to create a Pointblank Validator for a loaded DataFrame.",
    tags={"Validation"},
)
def prompt_create_validator(
    df_id: Annotated[str, "ID of the DataFrame to validate."] = "df_default",
    validator_id: Annotated[
        Optional[str],
        "Optional ID for the Validator. If not provided, a new ID will be generated.",
    ] = "validator_default",
    table_name: Annotated[
        Optional[str],
        "Optional name for the table within Pointblank reports. If not provided, a default name will be used.",
    ] = "data_table",
    validator_label: Annotated[
        Optional[str],
        "Optional descriptive label for the Validator. If not provided, a default label will be used.",
    ] = "Validator",
    thresholds_dict_example: Annotated[
        Optional[Dict[str, Union[int, float]]],
        "Example thresholds for validation failures. If not provided, a default example will be used.",
    ] = None,
) -> tuple:
    """
    Prompt guiding the LLM to create a Pointblank Validator object.
    Includes an example for thresholds_dict.
    """
    thresholds_msg_example = (
        thresholds_dict_example if thresholds_dict_example else {"warning": 0.05, "error": 10}
    )

    return (
        Message(
            "Once your data is loaded (using its `df_id`), I can create a 'Validator' object to define data quality checks.",
            role="assistant",
        ),
        Message(
            f"Please call `create_validator` using the `df_id` of your loaded data (e.g., '{df_id}').\n"
            f"You can optionally provide:\n"
            f"- `validator_id` (e.g., '{validator_id}') to name this validator instance.\n"
            f"- `table_name` (e.g., '{table_name}') as a reference name for the data table in reports.\n"
            f"- `validator_label` (e.g., '{validator_label}') for a descriptive label.\n"
            f"- `thresholds_dict` (e.g., {thresholds_msg_example}) to set global failure thresholds for validation steps.\n"
            f"- Other optional parameters like `actions_dict`, `final_actions_dict`, `brief`, `lang`, `locale` can also be specified if needed.\n"
            "Make a note of the returned `validator_id` to use when adding validation steps.",
            role="user",
        ),
    )


@mcp.prompt(
    name="prompt_add_validation_step_example",
    description="Prompt to add a validation step to a Pointblank Validator.",
    tags={"Validation"},
)
def prompt_add_validation_step_example() -> tuple:
    return (
        Message(
            "I can add various validation steps to your validator. "
            "You'll need to specify the 'validator_id', 'validation_type', and 'params' for the step. "
            "For example, to check if values in column 'age' are less than 100 for validator 'validator_123':",
            role="assistant",
        ),
        Message(
            "Please call `add_validation_step` with validator_id='validator_123', "
            "validation_type='col_vals_lt', and params={'columns': 'age', 'value': 100}. "
            "Note: Parameter names within 'params' (like 'columns', 'value', 'left', 'right', 'set_', etc.) must exactly match what the specific Pointblank validation function expects.\n"
            "Other examples:\n"
            "- For 'col_vals_between': params={'columns': 'score', 'left': 0, 'right': 100, 'inclusive': [True, True]}\n"
            "- For 'col_vals_in_set': params={'columns': 'grade', 'set_': ['A', 'B', 'C']} (Note: Pointblank uses 'set_' for this method's list of values)\n"
            "- For 'col_exists': params={'columns': 'user_id'}\n"
            "Refer to the Pointblank Python API for the 'Validate' class for available `validation_type` (method names) and their specific `params`.",
            role="user",
        ),
    )


@mcp.prompt(
    name="prompt_get_validation_step_output",
    description="Prompt to get validation output by specifying either a step index or a sundered type.",
    tags={"Validation"},
)
def prompt_get_validation_step_output(
    validator_id: Annotated[str, "Example ID of the Validator."] = "validator_123",
    step_index: Annotated[
        Optional[int],
        "Example step index for the first mode of operation.",
    ] = 0,
    sundered_type: Annotated[
        Optional[str], "Example sundered type ('pass' or 'fail') for the second mode of operation."
    ] = "fail",
) -> tuple:
    """
    Guides the LLM to get a validation output CSV by choosing one of two modes:
    1.  By a specific step index.
    2.  By the overall sundered data type ('pass' or 'fail').
    """
    return (
        Message(
            "I can extract validation data in two different ways. You must choose one: "
            "either get data for a *specific step* by its index, or get *all passed or failed rows* from the entire validation run.",
            role="assistant",
        ),
        Message(
            f"Please call the `get_validation_step_output` tool using only **one** of the following mutually exclusive options:\n\n"
            f"**OPTION 1: Get data for a specific step**\n"
            f"To get the data extract for step number {step_index}, use the `step_index` parameter. For example:\n"
            f"`get_validation_step_output(validator_id='{validator_id}', step_index={step_index}, output_path='step_{step_index}_data.csv')`\n\n"
            f"**OPTION 2: Get all passed or failed data**\n"
            f"To get all rows that '{sundered_type}' across all validation steps, use the `sundered_type` parameter. For example:\n"
            f"`get_validation_step_output(validator_id='{validator_id}', sundered_type='{sundered_type}', output_path='all_{sundered_type}_rows.csv')`",
            role="user",
        ),
    )


@mcp.prompt(
    name="prompt_interrogate_validator",
    description="Prompt to run validations and generate reports with Python code.",
    tags={"Validation"},
)
def prompt_interrogate_validator(
    validator_id: Annotated[str, "ID of the Validator to interrogate."],
) -> tuple:
    """
    Prompt guiding the LLM to run validations and generate HTML reports with Python code equivalent.
    """
    return (
        Message(
            "After all desired validation steps have been added to a validator, I can run the interrogation process. This will execute all checks and generate comprehensive reports.",
            role="assistant",
        ),
        Message(
            f"Please call `interrogate_validator` with the `validator_id` (e.g., '{validator_id}').\n"
            f"This will:\n"
            f"• Execute all validation checks and return a JSON summary\n"
            f"• Generate an interactive HTML report that opens in your browser\n"
            f"• Provide Python code equivalent for reproducing the validation\n"
            f"• Give you the flexibility to customize and extend the validation in your own scripts",
            role="user",
        ),
    )


# --- Core Pointblank Table Functions ---


@mcp.tool()
async def preview_table(
    ctx: Context,
    dataframe_id: str,
    n_head: int = 5,
    n_tail: int = 5,
    limit: int = 1000,
    show_row_numbers: bool = True,
) -> str:
    """
    Display a preview of the DataFrame showing rows from top and bottom.

    Uses Pointblank's built-in preview() function to generate a nicely formatted
    table view with column types and a sample of the data.
    """
    try:
        # Get the DataFrame
        app_ctx: AppContext = ctx.request_context.lifespan_context

        if dataframe_id not in app_ctx.loaded_dataframes:
            return f"❌ Error: DataFrame '{dataframe_id}' not found. Load a DataFrame first."

        data = app_ctx.loaded_dataframes[dataframe_id]

        # Use Pointblank's preview function
        import pointblank as pb

        gt_table = pb.preview(
            data, n_head=n_head, n_tail=n_tail, limit=limit, show_row_numbers=show_row_numbers
        )

        # Convert to HTML string for display
        html_output = gt_table.as_raw_html()

        # Save HTML to file for viewing

        # Create a complete HTML document with nice styling
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DataFrame Preview: {dataframe_id}</title>
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
        {html_output}
    </div>
</body>
</html>
"""

        # Save HTML to file for viewing
        try:
            # Save to a user-friendly location
            html_filename = (
                f"pointblank_preview_{dataframe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            )
            html_path = Path.cwd() / html_filename

            # Skip file generation during testing
            if TESTING_MODE:
                browser_msg = f"HTML preview generated (file creation skipped during testing)\n\nFile location: {html_path}"
            else:
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(full_html)

                # Open in default browser
                try:
                    _open_browser_conditionally(f"file://{html_path}")
                    browser_msg = f"HTML preview saved and opened in default browser!\n\nFile location: {html_path}"
                except Exception as browser_error:
                    browser_msg = f"HTML preview saved to: {html_path}\n\n📖 Could not open browser automatically: {str(browser_error)}\nPlease open the file manually in your browser."

        except Exception as e:
            browser_msg = f"Error saving HTML file: {str(e)}"

        return f"✅ Table preview generated successfully!\n\n{browser_msg}\n\nShowing {n_head} head + {n_tail} tail rows from {data.shape[0]:,} total rows with {data.shape[1]} columns."

    except Exception as e:
        logger.error(f"Error creating table preview: {e}")
        return f"❌ Error creating preview: {str(e)}"


@mcp.tool()
async def missing_values_table(
    ctx: Context,
    dataframe_id: str,
) -> str:
    """
    Generate a table showing missing values analysis for the DataFrame.

    Uses Pointblank's built-in missing_vals_tbl() function to show
    missing value patterns and statistics.
    """
    try:
        # Get the DataFrame
        app_ctx: AppContext = ctx.request_context.lifespan_context

        if dataframe_id not in app_ctx.loaded_dataframes:
            return f"❌ Error: DataFrame '{dataframe_id}' not found. Load a DataFrame first."

        data = app_ctx.loaded_dataframes[dataframe_id]

        # Use Pointblank's missing_vals_tbl function
        import pointblank as pb

        gt_table = pb.missing_vals_tbl(data)

        # Convert to HTML string for display
        html_output = gt_table.as_raw_html()

        # Save HTML to file for viewing

        # Create a complete HTML document with nice styling
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Missing Values Analysis: {dataframe_id}</title>
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
        {html_output}
    </div>
</body>
</html>
"""

        # Save HTML to file for viewing
        try:
            # Save to a user-friendly location
            html_filename = f"pointblank_missing_values_{dataframe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            html_path = Path.cwd() / html_filename

            # Skip file generation during testing
            if TESTING_MODE:
                browser_msg = f"HTML missing values analysis generated (file creation skipped during testing)\n\nFile location: {html_path}"
            else:
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(full_html)

                # Open in default browser
                try:
                    _open_browser_conditionally(f"file://{html_path}")
                    browser_msg = f"HTML missing values analysis saved and opened in default browser!\n\nFile location: {html_path}"
                except Exception as browser_error:
                    browser_msg = f"HTML analysis saved to: {html_path}\n\n📖 Could not open browser automatically: {str(browser_error)}\nPlease open the file manually in your browser."

        except Exception as e:
            browser_msg = f"Error saving HTML file: {str(e)}"

        return f"✅ Missing values analysis generated successfully!\n\n{browser_msg}\n\nDataset: {data.shape[0]:,} rows × {data.shape[1]} columns"

    except Exception as e:
        logger.error(f"Error creating missing values table: {e}")
        return f"❌ Error creating missing values analysis: {str(e)}"


@mcp.tool()
async def column_summary_table(
    ctx: Context,
    dataframe_id: str,
    table_name: str = None,
) -> str:
    """
    Generate a comprehensive column-level summary of the DataFrame.

    Uses Pointblank's built-in col_summary_tbl() function to provide detailed
    statistics including data types, missing values, and descriptive statistics.
    """
    try:
        # Get the DataFrame
        app_ctx: AppContext = ctx.request_context.lifespan_context

        if dataframe_id not in app_ctx.loaded_dataframes:
            return f"❌ Error: DataFrame '{dataframe_id}' not found. Load a DataFrame first."

        data = app_ctx.loaded_dataframes[dataframe_id]

        # Use Pointblank's col_summary_tbl function
        import pointblank as pb

        gt_table = pb.col_summary_tbl(data, tbl_name=table_name if table_name else dataframe_id)

        # Convert to HTML string for display
        html_output = gt_table.as_raw_html()

        # Save HTML to file for viewing

        # Create a complete HTML document with nice styling
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Column Summary: {dataframe_id}</title>
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
        {html_output}
    </div>
</body>
</html>
"""

        # Save HTML to file for viewing
        try:
            # Save to a user-friendly location
            html_filename = f"pointblank_column_summary_{dataframe_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            html_path = Path.cwd() / html_filename

            # Skip file generation during testing
            if TESTING_MODE:
                browser_msg = f"HTML column summary generated (file creation skipped during testing)\n\nFile location: {html_path}"
            else:
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write(full_html)

                # Open in default browser
                try:
                    _open_browser_conditionally(f"file://{html_path}")
                    browser_msg = f"HTML column summary saved and opened in default browser!\n\nFile location: {html_path}"
                except Exception as browser_error:
                    browser_msg = f"HTML summary saved to: {html_path}\n\n📖 Could not open browser automatically: {str(browser_error)}\nPlease open the file manually in your browser."

        except Exception as e:
            browser_msg = f"Error saving HTML file: {str(e)}"

        return f"✅ Column summary table generated successfully!\n\n{browser_msg}\n\nDataset: {data.shape[0]:,} rows × {data.shape[1]} columns"

    except Exception as e:
        logger.error(f"Error creating column summary table: {e}")
        return f"❌ Error creating column summary: {str(e)}"


@mcp.tool(
    name="draft_validation_plan",
    description="Generate an AI-powered validation plan using Pointblank's DraftValidation class.",
    tags={"Validation", "AI"},
)
async def draft_validation_plan(
    ctx: Context,
    dataframe_id: Annotated[str, "ID of the DataFrame to generate validation plan for."],
    model: Annotated[
        str,
        "AI model to use in format 'provider:model' (e.g., 'anthropic:claude-sonnet-4-5', 'openai:gpt-4'). Supported providers: anthropic, openai, ollama, bedrock.",
    ] = "anthropic:claude-sonnet-4-5",
    api_key: Annotated[
        Optional[str],
        "API key for the model provider. If not provided, will try to load from environment variables or .env file.",
    ] = None,
) -> Dict[str, Any]:
    """
    Uses Pointblank's DraftValidation class to generate an AI-powered validation plan.
    This provides a much more sophisticated and data-aware validation strategy than templates.

    The AI analyzes your data structure, types, ranges, and patterns to generate
    appropriate validation rules automatically.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if dataframe_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame ID '{dataframe_id}' not found.")

    df = app_ctx.loaded_dataframes[dataframe_id]

    await ctx.report_progress(
        10, 100, f"Initializing AI validation plan generation with {model}..."
    )

    try:
        # Check if DraftValidation dependencies are available
        try:
            from pointblank.draft import DraftValidation
        except ImportError as e:
            return {
                "error": "DraftValidation not available",
                "message": "The DraftValidation feature requires additional dependencies. Install with: pip install pointblank[generate]",
                "details": str(e),
            }

        await ctx.report_progress(
            30, 100, "Analyzing data structure and generating validation plan..."
        )

        # Generate the validation plan using AI
        draft_validator = DraftValidation(data=df, model=model, api_key=api_key)

        await ctx.report_progress(80, 100, "Processing AI-generated validation plan...")

        # Get the generated validation plan
        validation_plan = str(draft_validator)

        await ctx.report_progress(90, 100, "Formatting results...")

        # Extract just the Python code from the response
        code_start = validation_plan.find("```python")
        code_end = validation_plan.find("```", code_start + 9)

        if code_start != -1 and code_end != -1:
            python_code = validation_plan[code_start + 9 : code_end].strip()
        else:
            python_code = validation_plan

        await ctx.report_progress(100, 100, "AI validation plan generated successfully!")

        # Format the response with enhanced presentation
        formatted_response = f"""## **🤖 AI-Generated Validation Plan**

**Model Used:** {model}
**DataFrame:** {dataframe_id} ({df.shape[0]:,} rows × {df.shape[1]} columns)

### **📋 Generated Validation Code:**

<details>
<summary><strong>🔍 View Complete Python Code</strong></summary>

```python
{python_code}
```
</details>

### **🎯 Next Steps:**
1. **Review the generated plan** - The AI has analyzed your data structure and suggested appropriate validations
2. **Customize as needed** - Adjust thresholds, add business-specific rules, or remove unnecessary checks
3. **Copy and adapt** - Replace `your_data` with your actual DataFrame variable
4. **Run the validation** - Execute the code to validate your data

### **💡 Key Features:**
- **Schema validation** ensuring column types match expectations
- **Range checks** for numeric values based on actual data distribution
- **Null value handling** customized to your data's missing value patterns
- **Business logic** inferred from data characteristics
- **Row and column count validation** ensuring data integrity

The AI has analyzed your specific dataset and generated validation rules tailored to your data's characteristics!
"""

        return {
            "status": "success",
            "model_used": model,
            "dataframe_id": dataframe_id,
            "validation_plan": python_code,
            "formatted_response": formatted_response,
            "raw_response": validation_plan,
        }

    except Exception as e:
        error_msg = f"Error generating AI validation plan: {str(e)}"
        logger.error(error_msg)
        await ctx.report_progress(100, 100, error_msg)

        return {
            "status": "error",
            "error": error_msg,
            "suggestion": "Check your API key and model availability. For Anthropic models, ensure ANTHROPIC_API_KEY is set in environment or .env file.",
        }


@mcp.tool()
def validation_assistant(
    ctx: Context,
    dataframe_id: str,
    validation_goal: str = "general data quality",
) -> str:
    """
    Interactive assistant to help create a validation plan for your data.

    This tool walks you through creating appropriate validation rules based on
    your data characteristics and validation goals.
    """
    try:
        # Get the DataFrame
        app_ctx: AppContext = ctx.request_context.lifespan_context

        if dataframe_id not in app_ctx.loaded_dataframes:
            return f"❌ Error: DataFrame '{dataframe_id}' not found. Load a DataFrame first."

        data = app_ctx.loaded_dataframes[dataframe_id]

        # Analyze the DataFrame structure

        # Get basic info about the data
        if hasattr(data, "shape"):
            rows, cols = data.shape
        elif hasattr(data, "count"):
            rows = data.count().collect()[0, 0] if hasattr(data.count(), "collect") else len(data)
            cols = len(data.columns)
        else:
            rows, cols = "unknown", len(data.columns) if hasattr(data, "columns") else "unknown"

        # Get column information
        column_info = []
        for col in data.columns:
            if hasattr(data, "dtypes"):
                if hasattr(data.dtypes, "items"):  # pandas
                    dtype = str(data.dtypes[col])
                else:  # polars
                    dtype = str(data.dtypes[data.columns.index(col)])
            else:
                dtype = "unknown"
            column_info.append(f"  - {col}: {dtype}")

        # Generate validation suggestions based on goal
        suggestions = []

        if validation_goal in ["general", "data_quality"]:
            suggestions.extend(
                [
                    "# Basic Data Quality Checks",
                    "validator = pb.Validate(data)",
                    f".col_exists(columns={data.columns})  # Ensure all expected columns exist",
                ]
            )

            # Add type-specific suggestions
            for col in data.columns:
                if any(keyword in col.lower() for keyword in ["id", "key"]):
                    suggestions.append(
                        f".col_vals_not_null(columns='{col}')  # ID columns should not be null"
                    )
                    suggestions.append(
                        f".col_vals_unique(columns='{col}')  # ID columns should be unique"
                    )
                elif any(keyword in col.lower() for keyword in ["email", "mail"]):
                    suggestions.append(
                        f".col_vals_regex(columns='{col}', regex=r'[^@]+@[^@]+\\.[^@]+')  # Email format"
                    )
                elif any(keyword in col.lower() for keyword in ["phone", "tel"]):
                    suggestions.append(
                        f".col_vals_regex(columns='{col}', regex=r'\\+?[\\d\\s\\-\\(\\)]+')  # Phone format"
                    )
                elif any(keyword in col.lower() for keyword in ["date", "time"]):
                    suggestions.append(
                        f".col_vals_not_null(columns='{col}')  # Date columns should not be null"
                    )

        if validation_goal in ["general", "completeness"]:
            suggestions.extend(
                [
                    "# Completeness Checks",
                    ".col_vals_not_null(columns=['critical_column1', 'critical_column2'])  # Critical fields must be complete",
                    ".row_count_match(count=expected_count)  # Verify expected number of records",
                ]
            )

        if validation_goal in ["general", "consistency"]:
            suggestions.extend(
                [
                    "# Consistency Checks",
                    ".col_vals_in_set(columns='status', set=['active', 'inactive', 'pending'])  # Valid status values",
                    ".col_vals_between(columns='age', left=0, right=120)  # Reasonable age range",
                ]
            )

        if validation_goal in ["general", "accuracy"]:
            suggestions.extend(
                [
                    "# Accuracy Checks",
                    ".col_vals_gt(columns='price', value=0)  # Prices should be positive",
                    ".col_vals_le(columns='discount_pct', value=100)  # Discounts <= 100%",
                ]
            )

        # Add interrogation
        suggestions.append(".interrogate()  # Execute all validation steps")

        suggestion_text = "\n".join(suggestions)

        response = f"""
🔍 **Validation Assistant for DataFrame '{dataframe_id}'**

📊 **Data Overview:**
- Rows: {rows}
- Columns: {cols}
- Column Details:
{chr(10).join(column_info)}

🎯 **Validation Goal:** {validation_goal}

📋 **Suggested Validation Plan:**

```python
{suggestion_text}
```

💡 **Next Steps:**
1. Review the suggested validation rules above
2. Customize the rules based on your specific business requirements
3. Use the `create_validator` tool to implement these checks
4. Run `interrogate_validator` to execute the validation

❓ **Need More Help?**
- Use `validation_goal='data_quality'` for basic data quality checks
- Use `validation_goal='completeness'` for missing data validation
- Use `validation_goal='consistency'` for value range/format validation
- Use `validation_goal='accuracy'` for business rule validation

Would you like me to create a validator with these suggestions? Use the `create_validator` tool with the above validation steps!
"""

        return response

    except Exception as e:
        logger.error(f"Error in validation assistant: {e}")
        return f"❌ Error in validation assistant: {str(e)}"


@mcp.tool(
    name="get_pointblank_api_reference",
    description="Get API reference for Pointblank validation methods and common patterns.",
    tags={"Reference"},
)
async def get_pointblank_api_reference(
    ctx: Context,
    category: Annotated[
        str,
        "Category of API reference: 'validation_methods', 'data_types', 'thresholds', 'common_patterns', or 'all'",
    ] = "validation_methods",
) -> str:
    """
    Provides comprehensive API reference for Pointblank validation methods and patterns.
    Helps ensure correct parameter usage and method signatures.
    """

    validation_methods = {
        "col_vals_ge": {
            "description": "Check that column values are greater than or equal to value",
            "parameters": {
                "columns": "str | list - Column name(s) to check",
                "value": "float - Minimum value",
            },
            "example": '.col_vals_ge(columns="price", value=0)',
        },
        "col_vals_gt": {
            "description": "Check that column values are greater than value",
            "parameters": {
                "columns": "str | list - Column name(s) to check",
                "value": "float - Value to compare against",
            },
            "example": '.col_vals_gt(columns="quantity", value=0)',
        },
        "col_vals_le": {
            "description": "Check that column values are less than or equal to value",
            "parameters": {
                "columns": "str | list - Column name(s) to check",
                "value": "float - Maximum value",
            },
            "example": '.col_vals_le(columns="percentage", value=100)',
        },
        "col_vals_lt": {
            "description": "Check that column values are less than value",
            "parameters": {
                "columns": "str | list - Column name(s) to check",
                "value": "float - Value to compare against",
            },
            "example": '.col_vals_lt(columns="score", value=100)',
        },
        "col_vals_null": {
            "description": "Check that column values are null/missing",
            "parameters": {"columns": "str | list - Column name(s) to check"},
            "example": '.col_vals_null(columns="empty")',
        },
        "col_vals_not_null": {
            "description": "Check that column values are not null/missing",
            "parameters": {"columns": "str | list - Column name(s) to check"},
            "example": '.col_vals_not_null(columns="email")',
        },
        "col_vals_between": {
            "description": "Check that column values are within a numeric range",
            "parameters": {
                "columns": "str | list - Column name(s) to check",
                "left": "float - Lower bound (inclusive by default)",
                "right": "float - Upper bound (inclusive by default)",
                "inclusive": "tuple[bool, bool] - (left_inclusive, right_inclusive)",
            },
            "example": '.col_vals_between(columns="age", left=0, right=120)',
        },
        "col_vals_in_set": {
            "description": "Check that column values are in allowed set",
            "parameters": {
                "columns": "str | list - Column name(s) to check",
                "set": "list - List of allowed values",
            },
            "example": '.col_vals_in_set(columns="status", set=["active", "inactive"])',
        },
        "col_vals_outside": {
            "description": "Check that column values are outside of a specified set",
            "parameters": {
                "columns": "str | list - Column name(s) to check",
                "set": "list - List of allowed values",
            },
            "example": '.col_vals_outside(columns="status", set=["active", "inactive"])',
        },
        "col_vals_regex": {
            "description": "Check that column values match regex pattern",
            "parameters": {
                "columns": "str | list - Column name(s) to check",
                "pattern": "str - Regular expression pattern",
            },
            "example": '.col_vals_regex(columns="email", pattern=r"[^@]+@[^@]+\\.[^@]+")',
        },
        "col_exists": {
            "description": "Check that specified columns exist in the table",
            "parameters": {"columns": "str | list - Column name(s) to check"},
            "example": '.col_exists(columns=["name", "email", "age"])',
        },
        "rows_distinct": {
            "description": "Check that all rows in the table are unique",
            "parameters": {
                "columns_subset": "str | list - Column name(s) for constraining uniqueness"
            },
            "example": ".rows_distinct()",
        },
        "rows_complete": {
            "description": "Check that all rows in the table are complete (no missing values)",
            "parameters": {
                "columns_subset": "str | list - Column name(s) for constraining completeness"
            },
            "example": ".rows_complete()",
        },
    }

    data_types_info = """
    **Column Data Type Expectations:**
    - Numeric validations (between, gt, ge, lt, le): Work with int, float columns
    - String validations (regex, in_set): Work with object/string columns
    - Null checks (not_null, null): Work with any column type
    - Set validations (in_set, not_in_set): Work with any comparable type
    """

    thresholds_info = """
    **Threshold Configuration:**
    ```python
    thresholds = {
        "warning": 0.05,    # 5% failures trigger warning
        "error": 0.10,      # 10% failures trigger error
        "critical": 0.15    # 15% failures trigger critical
    }
    ```

    **Threshold Levels:**
    - warning: Minor data quality issues
    - error: Significant problems requiring attention
    - critical: Severe issues that stop processing
    """

    common_patterns = """
    **Common Validation Patterns:**

    1. **Data Integrity:**
    ```python
    .col_vals_not_null(columns="id")
    .rows_distinct()
    .col_exists(columns=["required_field1", "required_field2"])
    ```

    2. **Business Rules:**
    ```python
    .col_vals_between(columns="age", left=0, right=120)
    .col_vals_in_set(columns="status", set=["active", "inactive", "pending"])
    .col_vals_ge(columns="price", value=0)
    ```

    3. **Format Validation:**
    ```python
    .col_vals_regex(columns="email", pattern=r"[^@]+@[^@]+\\.[^@]+")
    .col_vals_regex(columns="phone", pattern=r"\\d{3}-\\d{3}-\\d{4}")
    ```

    4. **Range Validation:**
    ```python
    .col_vals_between(columns="percentage", left=0, right=100)
    .col_vals_between(columns="year", left=1900, right=2030)
    ```
    """

    if category == "validation_methods":
        result = "# 🔧 **Pointblank Validation Methods Reference**\n\n"
        for method_name, info in validation_methods.items():
            result += f"## `{method_name}`\n"
            result += f"**Description:** {info['description']}\n\n"
            result += "**Parameters:**\n"
            for param, desc in info["parameters"].items():
                result += f"- `{param}`: {desc}\n"
            result += f"\n**Example:** `{info['example']}`\n\n---\n\n"
        return result

    elif category == "data_types":
        return f"# 📊 **Data Types Reference**\n\n{data_types_info}"

    elif category == "thresholds":
        return f"# ⚠️ **Thresholds Reference**\n\n{thresholds_info}"

    elif category == "common_patterns":
        return f"# 🎯 **Common Patterns Reference**\n\n{common_patterns}"

    elif category == "all":
        # Build complete reference
        validation_methods_ref = "# 🔧 **Pointblank Validation Methods Reference**\n\n"
        for method_name, info in validation_methods.items():
            validation_methods_ref += f"## `{method_name}`\n"
            validation_methods_ref += f"**Description:** {info['description']}\n\n"
            validation_methods_ref += "**Parameters:**\n"
            for param, desc in info["parameters"].items():
                validation_methods_ref += f"- `{param}`: {desc}\n"
            validation_methods_ref += f"\n**Example:** `{info['example']}`\n\n---\n\n"

        return f"""# 📚 **Complete Pointblank API Reference**

{validation_methods_ref}

{data_types_info}

{thresholds_info}

{common_patterns}

**💡 Tips:**
- Always check parameter names match exactly (e.g., 'columns' not 'column')
- Use list for multiple columns: `columns=["col1", "col2"]`
- String values in sets need quotes: `set=["value1", "value2"]`
- Numeric ranges are inclusive by default
- Regular expressions need proper escaping
"""

    else:
        return f"❌ Unknown category '{category}'. Use: validation_methods, data_types, thresholds, common_patterns, or all"


def _format_validation_plan_summary(validator_steps: list) -> str:
    """
    Format validation steps into a readable summary with code block.
    """
    plan_summary = "## **🎯 Netflix Dataset Validation Plan**\n\n"

    # Group by category
    integrity_checks = []
    business_logic = []

    step_descriptions = {
        "col_vals_not_null": "Ensures no missing values",
        "rows_distinct": "Checks for duplicate entries",
        "col_vals_in_set": "Validates against allowed values",
        "col_vals_between": "Validates numeric ranges",
        "col_vals_ge": "Ensures values are non-negative",
        "col_vals_regex": "Validates format patterns",
    }

    for i, step in enumerate(validator_steps, 1):
        step_type = step.get("validation_type", "unknown")
        columns = step.get("params", {}).get("columns", "table")
        description = step_descriptions.get(step_type, f"Validates {step_type}")

        if step_type in ["col_vals_not_null", "rows_distinct", "col_exists"]:
            integrity_checks.append(
                f"{i}. **{columns.title() if isinstance(columns, str) else 'Data'} Check** - {description}"
            )
        else:
            business_logic.append(
                f"{i}. **{columns.title() if isinstance(columns, str) else 'Value'} Validation** - {description}"
            )

    # Add categories
    if integrity_checks:
        plan_summary += "### **Data Integrity Checks:**\n"
        for check in integrity_checks:
            plan_summary += f"- {check}\n"
        plan_summary += "\n"

    if business_logic:
        plan_summary += "### **Business Logic Validations:**\n"
        for validation in business_logic:
            plan_summary += f"- {validation}\n"
        plan_summary += "\n"

    # Add code block
    plan_summary += """<details>
<summary><strong>📋 View Pointblank Code</strong></summary>

```python
import pointblank as pb

# Create validator
validator = (
    pb.Validate(data=df, tbl_name="Netflix Movies and TV Shows")
    .col_vals_not_null(columns="show_id")
    .rows_distinct()
    .col_vals_not_null(columns="title")
    .col_vals_in_set(columns="type", set=["Movie", "TV Show"])
    .col_vals_between(columns="release_year", left=1900, right=2025)
    .col_vals_between(columns="rating", left=0, right=10)
    .col_vals_ge(columns="vote_count", value=0)
    .col_vals_between(columns="vote_average", left=0, right=10)
    .col_vals_ge(columns="budget", value=0)
    .col_vals_ge(columns="revenue", value=0)
)

# Run validation
results = validator.interrogate()
```
</details>

"""

    return plan_summary


if __name__ == "__main__":
    mcp.run(transport="stdio")
