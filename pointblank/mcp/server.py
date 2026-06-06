"""Pointblank MCP Server — tool and prompt definitions."""

import json
import platform
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Annotated,
    Any,
    AsyncIterator,
    Dict,
    Optional,
    Union,
)

from fastmcp import Context, FastMCP

import pointblank as pb
from pointblank.mcp._config import HAS_PANDAS, HAS_POLARS, logger
from pointblank.mcp._prompts import (
    prompt_add_validation_step_example,
    prompt_create_validator,
    prompt_get_validation_step_output,
    prompt_interrogate_validator,
    prompt_load_dataframe,
)
from pointblank.mcp._templates import AVAILABLE_TEMPLATES, get_validation_template
from pointblank.mcp._utils import (
    SUPPORTED_VALIDATION_TYPES,
    clean_for_json_serialization,
    generate_python_code_for_validator,
    generate_validation_report_html,
    get_available_backends,
    get_supported_validations,
    load_dataframe_from_path,
    open_browser_conditionally,
    save_dataframe_to_csv,
    save_html_and_open,
    validate_output_path,
    validate_resource_id,
)

logger.info(f"MCP Server starting at {datetime.now()}")
logger.info(f"Available DataFrame backends: pandas={HAS_PANDAS}, polars={HAS_POLARS}")


# --- Lifespan Context ---


@dataclass
class AppContext:
    loaded_dataframes: Dict[str, Any] = field(default_factory=dict)
    active_validators: Dict[str, pb.Validate] = field(default_factory=dict)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    context = AppContext()
    yield context
    context.loaded_dataframes.clear()
    context.active_validators.clear()


# --- FastMCP Instance ---


def _create_fastmcp_instance():
    """Create FastMCP instance with backwards compatibility for dependencies parameter."""
    try:
        import fastmcp

        version_str = getattr(fastmcp, "__version__", "0.0.0")
        version_parts = version_str.split(".")
        if len(version_parts) >= 3:
            major, minor, patch = (
                int(version_parts[0]),
                int(version_parts[1]),
                int(version_parts[2]),
            )
            if (
                (major > 2)
                or (major == 2 and minor > 11)
                or (major == 2 and minor == 11 and patch >= 4)
            ):
                return FastMCP("PointblankMCP", lifespan=app_lifespan)

        return FastMCP(
            "PointblankMCP", lifespan=app_lifespan, dependencies=["pointblank", "openpyxl"]
        )

    except Exception:
        try:
            return FastMCP("PointblankMCP", lifespan=app_lifespan)
        except Exception:
            return FastMCP(
                "PointblankMCP", lifespan=app_lifespan, dependencies=["pointblank", "openpyxl"]
            )


mcp = _create_fastmcp_instance()


# --- Register Prompts ---

mcp.prompt(
    name="prompt_load_dataframe",
    description="Prompt to load a DataFrame from a file into the server's context for validation.",
    tags={"Data Management"},
)(prompt_load_dataframe)

mcp.prompt(
    name="prompt_create_validator",
    description="Prompt to create a Pointblank Validator for a loaded DataFrame.",
    tags={"Validation"},
)(prompt_create_validator)

mcp.prompt(
    name="prompt_add_validation_step_example",
    description="Prompt to add a validation step to a Pointblank Validator.",
    tags={"Validation"},
)(prompt_add_validation_step_example)

mcp.prompt(
    name="prompt_get_validation_step_output",
    description="Prompt to get validation output by specifying either a step index or a sundered type.",
    tags={"Validation"},
)(prompt_get_validation_step_output)

mcp.prompt(
    name="prompt_interrogate_validator",
    description="Prompt to run validations and generate reports with Python code.",
    tags={"Validation"},
)(prompt_interrogate_validator)


# =============================================================================
# TOOLS: Data Management
# =============================================================================


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
        "DataFrame backend to use: 'auto', 'pandas', or 'polars'. Default is 'auto'.",
    ] = "auto",
) -> DataFrameInfo:
    """Load a DataFrame from file into the server's context."""
    app_ctx: AppContext = ctx.request_context.lifespan_context

    available_backends = get_available_backends()
    if not available_backends:
        raise RuntimeError("No DataFrame library available. Install pandas or polars.")

    await ctx.report_progress(10, 100, f"Available backends: {', '.join(available_backends)}")

    df = load_dataframe_from_path(input_path, backend)

    effective_df_id = df_id if df_id else f"df_{uuid.uuid4().hex[:8]}"
    effective_df_id = validate_resource_id(effective_df_id, "DataFrame")

    if effective_df_id in app_ctx.loaded_dataframes:
        raise ValueError(
            f"DataFrame ID '{effective_df_id}' already exists. Choose a different ID or omit to generate a new one."
        )

    app_ctx.loaded_dataframes[effective_df_id] = df

    shape = df.shape
    columns = list(df.columns)

    await ctx.report_progress(
        100, 100, f"DataFrame loaded with {backend} backend: {shape[0]} rows, {shape[1]} columns"
    )

    return DataFrameInfo(
        df_id=effective_df_id,
        shape=(int(shape[0]), int(shape[1])),
        columns=columns,
    )


@mcp.tool(
    name="list_available_backends",
    description="List available DataFrame backends (pandas, polars) installed in the environment.",
    tags={"Data Management"},
)
async def list_available_backends(ctx: Context) -> Dict[str, Any]:
    """Returns information about available DataFrame backends and their capabilities."""
    backends = get_available_backends()

    backend_info = {}
    for b in backends:
        if b == "pandas":
            backend_info["pandas"] = {
                "available": True,
                "supports": ["CSV", "Excel", "Parquet", "JSON"],
            }
        elif b == "polars":
            backend_info["polars"] = {
                "available": True,
                "supports": ["CSV", "Parquet", "JSON"],
                "notes": "Excel requires fallback to pandas",
            }

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
    """Returns information about all DataFrames currently loaded in the server."""
    app_ctx: AppContext = ctx.request_context.lifespan_context

    dataframes_info = {}
    for df_id, df in app_ctx.loaded_dataframes.items():
        try:
            shape = df.shape
            columns = list(df.columns)
            df_type = "pandas" if hasattr(df, "to_csv") and hasattr(df, "index") else "polars"

            dataframes_info[df_id] = {
                "shape": [int(shape[0]), int(shape[1])],
                "columns": columns,
                "column_count": len(columns),
                "row_count": int(shape[0]),
                "backend": df_type,
            }
        except Exception as e:
            dataframes_info[df_id] = {
                "error": f"Failed to get info: {str(e)}",
                "backend": "unknown",
            }

    return clean_for_json_serialization(
        {
            "loaded_dataframes": dataframes_info,
            "total_count": len(dataframes_info),
        }
    )


@mcp.tool(
    name="delete_dataframe",
    description="Remove a DataFrame from the server context to free up memory.",
    tags={"Data Management"},
)
async def delete_dataframe(
    ctx: Context,
    df_id: Annotated[str, "ID of the DataFrame to delete."],
) -> Dict[str, str]:
    """Removes a DataFrame from the server context and frees up memory."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    df_id = validate_resource_id(df_id, "DataFrame")

    if df_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame ID '{df_id}' not found.")

    del app_ctx.loaded_dataframes[df_id]

    message = f"DataFrame '{df_id}' deleted successfully."
    await ctx.report_progress(100, 100, message)
    return {"status": "success", "message": message}


# =============================================================================
# TOOLS: Data Analysis
# =============================================================================


@mcp.tool(
    name="analyze_data_quality",
    description="Analyze data quality using Pointblank's DataScan class.",
    tags={"Data Analysis"},
)
async def analyze_data_quality(
    ctx: Context,
    df_id: Annotated[str, "ID of the DataFrame to analyze."],
) -> Dict[str, Any]:
    """Analyze data quality using Pointblank's built-in DataScan functionality."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    df_id = validate_resource_id(df_id, "DataFrame")

    if df_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame ID '{df_id}' not found.")

    df = app_ctx.loaded_dataframes[df_id]

    await ctx.report_progress(20, 100, "Starting data quality analysis...")

    scanner = pb.DataScan(data=df)

    await ctx.report_progress(60, 100, "Running DataScan analysis...")

    profile_json = scanner.to_json()

    await ctx.report_progress(80, 100, "Processing results...")

    profile_results = json.loads(profile_json)
    cleaned_results = clean_for_json_serialization(profile_results)

    await ctx.report_progress(100, 100, "Data quality analysis complete!")

    return {"status": "success", "df_id": df_id, "analysis": cleaned_results}


@mcp.tool(
    name="profile_dataframe",
    description="Generate comprehensive data profiling insights for a loaded DataFrame.",
    tags={"Data Analysis"},
)
async def profile_dataframe(
    ctx: Context,
    df_id: Annotated[str, "ID of the DataFrame to profile."],
    sample_size: Annotated[
        int, "Maximum number of rows to sample for profiling (0 = all rows)."
    ] = 10000,
) -> Dict[str, Any]:
    """Generates comprehensive data profiling insights using Pointblank's DataScan class."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    df_id = validate_resource_id(df_id, "DataFrame")

    if df_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame ID '{df_id}' not found.")

    df = app_ctx.loaded_dataframes[df_id]

    await ctx.report_progress(10, 100, "Starting data profiling...")

    if sample_size > 0 and df.shape[0] > sample_size:
        if hasattr(df, "sample"):
            df_sample = df.sample(n=sample_size, random_state=23)
        else:
            df_sample = df.sample(n=sample_size, seed=23)
        await ctx.report_progress(20, 100, f"Sampling {sample_size} rows for analysis...")
    else:
        df_sample = df

    await ctx.report_progress(50, 100, "Running Pointblank DataScan...")

    scanner = pb.DataScan(data=df_sample)

    await ctx.report_progress(80, 100, "Converting to JSON...")

    profile_json = scanner.to_json()
    profile_results = json.loads(profile_json)
    cleaned_results = clean_for_json_serialization(profile_results)

    await ctx.report_progress(100, 100, "Data profiling complete!")
    return cleaned_results


# =============================================================================
# TOOLS: Validation
# =============================================================================


@mcp.tool(
    name="list_active_validators",
    description="List all validators currently active in the server context.",
    tags={"Validation"},
)
async def list_active_validators(ctx: Context) -> Dict[str, Any]:
    """Returns information about all active validators in the server."""
    app_ctx: AppContext = ctx.request_context.lifespan_context

    validators_info = {}
    for validator_id, validator in app_ctx.active_validators.items():
        try:
            table_name = getattr(validator, "tbl_name", "Unknown")
            label = getattr(validator, "label", "No label")
            is_interrogated = (
                hasattr(validator, "time_processed") and validator.time_processed is not None
            )
            step_count = len(getattr(validator, "_validation_set", []))

            validators_info[validator_id] = {
                "table_name": table_name,
                "label": label,
                "is_interrogated": is_interrogated,
                "validation_steps_count": step_count,
            }
        except Exception as e:
            validators_info[validator_id] = {"error": f"Failed to get info: {str(e)}"}

    return {
        "active_validators": validators_info,
        "total_count": len(validators_info),
    }


@mcp.tool(
    name="delete_validator",
    description="Remove a validator from the server context.",
    tags={"Validation"},
)
async def delete_validator(
    ctx: Context,
    validator_id: Annotated[str, "ID of the validator to delete."],
) -> Dict[str, str]:
    """Removes a validator from the server context."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    validator_id = validate_resource_id(validator_id, "Validator")

    if validator_id not in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{validator_id}' not found.")

    del app_ctx.active_validators[validator_id]

    message = f"Validator '{validator_id}' deleted successfully."
    await ctx.report_progress(100, 100, message)
    return {"status": "success", "message": message}


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
    validator_id: Annotated[Optional[str], "Optional ID for the Validator."] = None,
    table_name: Annotated[Optional[str], "Optional name for the table within reports."] = None,
    validator_label: Annotated[Optional[str], "Optional descriptive label."] = None,
    thresholds_dict: Annotated[
        Optional[Dict[str, Union[int, float]]],
        "Optional thresholds. Example: {'warning': 0.1, 'error': 5, 'critical': 0.10}.",
    ] = None,
    brief: Optional[bool] = None,
    lang: Optional[str] = None,
    locale: Optional[str] = None,
) -> ValidatorInfo:
    """Creates a Pointblank Validator for a previously loaded DataFrame."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    df_id = validate_resource_id(df_id, "DataFrame")

    if df_id not in app_ctx.loaded_dataframes:
        raise ValueError(
            f"DataFrame ID '{df_id}' not found. Please load it first using 'load_dataframe'."
        )

    df = app_ctx.loaded_dataframes[df_id]

    effective_validator_id = validator_id if validator_id else f"validator_{uuid.uuid4().hex[:8]}"
    effective_validator_id = validate_resource_id(effective_validator_id, "Validator")

    if effective_validator_id in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{effective_validator_id}' already exists.")

    actual_table_name = table_name if table_name else f"table_for_{df_id}"
    actual_label = validator_label if validator_label else f"Validation for {actual_table_name}"

    validator_params = {
        "data": df,
        "tbl_name": actual_table_name,
        "label": actual_label,
    }

    if thresholds_dict:
        validator_params["thresholds"] = pb.Thresholds(**thresholds_dict)
    if brief is not None:
        validator_params["brief"] = brief
    if lang:
        validator_params["lang"] = lang
    if locale:
        validator_params["locale"] = locale

    validator_instance = pb.Validate(**validator_params)
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
        "Type of validation to perform (e.g., 'col_vals_lt', 'col_vals_between', 'rows_distinct').",
    ],
    params: Annotated[
        Dict[str, Any],
        "Parameters for the validation function matching Pointblank's API.",
    ],
) -> ValidationStepInfo:
    """Adds a validation step to an existing Pointblank Validator."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    validator_id = validate_resource_id(validator_id, "Validator")

    if validator_id not in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{validator_id}' not found.")

    validator = app_ctx.active_validators[validator_id]
    supported = get_supported_validations(validator)

    if validation_type not in supported:
        raise ValueError(
            f"Unsupported validation_type: '{validation_type}'. Supported: {list(supported.keys())}"
        )

    current_params = {**params}
    # Handle 'set_' -> 'set' mapping for Python reserved keyword
    if "set_" in current_params:
        current_params["set"] = current_params.pop("set_")

    try:
        supported[validation_type](**current_params)
    except TypeError as e:
        raise ValueError(f"Error calling '{validation_type}' with params {current_params}: {e}")

    return ValidationStepInfo(
        validator_id=validator_id,
        status=f"Validation step '{validation_type}' added successfully.",
    )


@mcp.tool(
    name="apply_validation_template",
    description="Apply a pre-built validation template to a validator.",
    tags={"Validation"},
)
async def apply_validation_template(
    ctx: Context,
    validator_id: Annotated[str, "ID of the Validator to apply template to."],
    template_name: Annotated[
        str,
        f"Template name. Available: {', '.join(AVAILABLE_TEMPLATES)}.",
    ],
    column_mapping: Annotated[
        Dict[str, str],
        "Mapping of template column names to actual DataFrame column names.",
    ],
) -> Dict[str, Any]:
    """Applies a pre-built validation template with common data quality checks."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    validator_id = validate_resource_id(validator_id, "Validator")

    if validator_id not in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{validator_id}' not found.")

    template = get_validation_template(template_name)
    if not template:
        raise ValueError(f"Unknown template '{template_name}'. Available: {AVAILABLE_TEMPLATES}")

    validator = app_ctx.active_validators[validator_id]
    supported = get_supported_validations(validator)
    applied_validations = []

    await ctx.report_progress(10, 100, f"Applying {template_name} template...")

    for i, rule in enumerate(template["validations"]):
        try:
            mapped_params = {}
            for key, value in rule["params"].items():
                if key == "columns" and value in column_mapping:
                    mapped_params[key] = column_mapping[value]
                elif isinstance(value, str) and value in column_mapping:
                    mapped_params[key] = column_mapping[value]
                else:
                    mapped_params[key] = value

            vtype = rule["validation_type"]
            if vtype in supported:
                supported[vtype](**mapped_params)
                applied_validations.append(
                    {
                        "validation_type": vtype,
                        "params": mapped_params,
                        "description": rule.get("description", ""),
                    }
                )

            await ctx.report_progress(
                10 + (i + 1) * 80 // len(template["validations"]), 100, f"Applied {vtype}..."
            )
        except Exception as e:
            applied_validations.append(
                {
                    "validation_type": rule["validation_type"],
                    "error": str(e),
                }
            )

    await ctx.report_progress(100, 100, f"Template {template_name} applied!")

    return {
        "template_name": template_name,
        "template_description": template["description"],
        "applied_validations": applied_validations,
        "total_validations": len(applied_validations),
        "successful_validations": len([v for v in applied_validations if "error" not in v]),
    }


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
    output_path: Annotated[str, "Path to save the output file. Must end with .csv."],
    sundered_type: Annotated[str, "Retrieve all 'pass' or 'fail' rows."] = "fail",
    step_index: Annotated[
        Optional[int], "Specific step index (0-based). Overrides sundered_type."
    ] = None,
) -> ValidationOutput:
    """Retrieves validation output and saves it to a CSV file."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    validator_id = validate_resource_id(validator_id, "Validator")

    if validator_id not in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{validator_id}' not found.")
    validator = app_ctx.active_validators[validator_id]

    p_output_path = validate_output_path(output_path, {".csv"})

    if step_index is not None and step_index < 0:
        raise ValueError("The 'step_index' cannot be a negative number.")

    if not getattr(validator, "time_processed", None):
        await ctx.warning(
            f"Validator '{validator_id}' has not been interrogated. Interrogating now."
        )

    message = ""
    data_extract_df = None

    if step_index is not None:
        data_extract_df = validator.get_data_extracts(i=step_index, frame=True)
        if data_extract_df is None or data_extract_df.empty:
            message = f"No data extract available for step {step_index}."
            data_extract_df = None
        else:
            message = f"Data extract for step {step_index} retrieved."
    else:
        data_extract_df = validator.get_sundered_data(type=sundered_type)
        if data_extract_df is None or data_extract_df.empty:
            message = f"No sundered data available for type '{sundered_type}'."
            data_extract_df = None
        else:
            message = f"Sundered data for type '{sundered_type}' retrieved."

    if data_extract_df is None:
        return ValidationOutput(status="success", message=message, output_file=None)

    save_dataframe_to_csv(data_extract_df, p_output_path)
    message = f"Data extract saved to {p_output_path.resolve()}"
    await ctx.report_progress(100, 100, message)

    return ValidationOutput(
        status="success", message=message, output_file=str(p_output_path.resolve())
    )


@mcp.tool(
    name="interrogate_validator",
    description="Run validations and return a JSON summary with Python code equivalent.",
    tags={"Validation"},
)
async def interrogate_validator(
    ctx: Context,
    validator_id: Annotated[str, "ID of the Validator to interrogate."],
) -> Dict[str, Any]:
    """Runs validations and returns a JSON summary with HTML report and Python code."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    validator_id = validate_resource_id(validator_id, "Validator")

    if validator_id not in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{validator_id}' not found.")

    validator = app_ctx.active_validators[validator_id]

    validator.interrogate()
    json_report_str = validator.get_json_report()

    html_report_path = None
    try:
        html_report_path = generate_validation_report_html(validator, validator_id)
        open_browser_conditionally(f"file://{html_report_path}")
        await ctx.report_progress(50, 100, "Validation report opened in browser")
    except Exception as html_error:
        logger.warning(f"Could not generate HTML report: {html_error}")

    python_code = "# Error generating Python code equivalent"
    try:
        python_code = generate_python_code_for_validator(validator, validator_id)
        await ctx.report_progress(75, 100, "Generated Python code equivalent")
    except Exception as code_error:
        logger.warning(f"Could not generate Python code: {code_error}")

    report_data = json.loads(json_report_str)

    await ctx.report_progress(100, 100, "Validation complete!")

    return {
        "validation_summary": report_data,
        "python_code": python_code,
        "html_report_path": html_report_path,
        "instructions": {
            "html_report": "Interactive validation report opened in your browser",
            "python_code": "Use the provided Python code to reproduce this validation in your own scripts",
        },
    }


# =============================================================================
# TOOLS: Table Visualization
# =============================================================================


@mcp.tool()
async def preview_table(
    ctx: Context,
    dataframe_id: str,
    n_head: int = 5,
    n_tail: int = 5,
    limit: int = 1000,
    show_row_numbers: bool = True,
) -> str:
    """Display a preview of the DataFrame showing rows from top and bottom."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    dataframe_id = validate_resource_id(dataframe_id, "DataFrame")

    if dataframe_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame '{dataframe_id}' not found. Load a DataFrame first.")

    data = app_ctx.loaded_dataframes[dataframe_id]

    gt_table = pb.preview(
        data, n_head=n_head, n_tail=n_tail, limit=limit, show_row_numbers=show_row_numbers
    )

    html_output = gt_table.as_raw_html()
    browser_msg = save_html_and_open(
        html_output,
        title=f"DataFrame Preview: {dataframe_id}",
        filename_prefix=f"pointblank_preview_{dataframe_id}",
    )

    return f"Table preview generated successfully!\n\n{browser_msg}\n\nShowing {n_head} head + {n_tail} tail rows from {data.shape[0]:,} total rows with {data.shape[1]} columns."


@mcp.tool()
async def missing_values_table(ctx: Context, dataframe_id: str) -> str:
    """Generate a table showing missing values analysis for the DataFrame."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    dataframe_id = validate_resource_id(dataframe_id, "DataFrame")

    if dataframe_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame '{dataframe_id}' not found. Load a DataFrame first.")

    data = app_ctx.loaded_dataframes[dataframe_id]

    gt_table = pb.missing_vals_tbl(data)
    html_output = gt_table.as_raw_html()
    browser_msg = save_html_and_open(
        html_output,
        title=f"Missing Values Analysis: {dataframe_id}",
        filename_prefix=f"pointblank_missing_values_{dataframe_id}",
    )

    return f"Missing values analysis generated!\n\n{browser_msg}\n\nDataset: {data.shape[0]:,} rows x {data.shape[1]} columns"


@mcp.tool()
async def column_summary_table(ctx: Context, dataframe_id: str, table_name: str = None) -> str:
    """Generate a comprehensive column-level summary of the DataFrame."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    dataframe_id = validate_resource_id(dataframe_id, "DataFrame")

    if dataframe_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame '{dataframe_id}' not found. Load a DataFrame first.")

    data = app_ctx.loaded_dataframes[dataframe_id]

    gt_table = pb.col_summary_tbl(data, tbl_name=table_name if table_name else dataframe_id)
    html_output = gt_table.as_raw_html()
    browser_msg = save_html_and_open(
        html_output,
        title=f"Column Summary: {dataframe_id}",
        filename_prefix=f"pointblank_column_summary_{dataframe_id}",
    )

    return f"Column summary table generated!\n\n{browser_msg}\n\nDataset: {data.shape[0]:,} rows x {data.shape[1]} columns"


# =============================================================================
# TOOLS: AI & Assistant
# =============================================================================


@mcp.tool(
    name="draft_validation_plan",
    description="Generate an AI-powered validation plan using Pointblank's DraftValidation class.",
    tags={"Validation", "AI"},
)
async def draft_validation_plan(
    ctx: Context,
    dataframe_id: Annotated[str, "ID of the DataFrame to generate validation plan for."],
    model: Annotated[str, "AI model in format 'provider:model'."] = "anthropic:claude-sonnet-4-5",
    api_key: Annotated[Optional[str], "API key for the model provider."] = None,
) -> Dict[str, Any]:
    """Uses Pointblank's DraftValidation to generate an AI-powered validation plan."""
    app_ctx: AppContext = ctx.request_context.lifespan_context
    dataframe_id = validate_resource_id(dataframe_id, "DataFrame")

    if dataframe_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame ID '{dataframe_id}' not found.")

    df = app_ctx.loaded_dataframes[dataframe_id]

    await ctx.report_progress(10, 100, f"Initializing AI validation plan with {model}...")

    try:
        from pointblank.draft import DraftValidation
    except ImportError as e:
        return {
            "error": "DraftValidation not available",
            "message": "Install with: pip install pointblank[generate]",
            "details": str(e),
        }

    await ctx.report_progress(30, 100, "Analyzing data and generating validation plan...")

    draft_validator = DraftValidation(data=df, model=model, api_key=api_key)

    await ctx.report_progress(80, 100, "Processing AI-generated validation plan...")

    validation_plan = str(draft_validator)

    code_start = validation_plan.find("```python")
    code_end = validation_plan.find("```", code_start + 9)
    python_code = (
        validation_plan[code_start + 9 : code_end].strip()
        if code_start != -1 and code_end != -1
        else validation_plan
    )

    await ctx.report_progress(100, 100, "AI validation plan generated!")

    return {
        "status": "success",
        "model_used": model,
        "dataframe_id": dataframe_id,
        "validation_plan": python_code,
        "raw_response": validation_plan,
    }


@mcp.tool()
def validation_assistant(
    ctx: Context,
    dataframe_id: str,
    validation_goal: str = "general",
) -> str:
    """
    Generate data-aware validation suggestions based on actual column types and statistics.

    Analyzes your data's structure (types, null counts, value ranges, cardinality)
    to produce actionable Pointblank validation code tailored to your dataset.
    """
    app_ctx: AppContext = ctx.request_context.lifespan_context
    dataframe_id = validate_resource_id(dataframe_id, "DataFrame")

    if dataframe_id not in app_ctx.loaded_dataframes:
        raise ValueError(f"DataFrame '{dataframe_id}' not found. Load a DataFrame first.")

    data = app_ctx.loaded_dataframes[dataframe_id]
    rows, cols = data.shape

    # Analyze each column with actual data characteristics
    column_analyses = []
    validation_steps = []

    for col in data.columns:
        # Get dtype
        if hasattr(data, "dtypes") and hasattr(data.dtypes, "items"):  # pandas
            dtype = str(data.dtypes[col])
        elif hasattr(data, "dtypes"):  # polars
            dtype = str(data.dtypes[data.columns.index(col)])
        else:
            dtype = "unknown"

        # Count nulls
        try:
            if hasattr(data[col], "isnull"):  # pandas
                null_count = int(data[col].isnull().sum())
            elif hasattr(data[col], "null_count"):  # polars
                null_count = int(data[col].null_count())
            else:
                null_count = 0
        except Exception:
            null_count = 0

        null_pct = round(null_count / rows * 100, 1) if rows > 0 else 0

        col_info = {"name": col, "dtype": dtype, "null_count": null_count, "null_pct": null_pct}

        # Numeric analysis
        is_numeric = any(t in dtype.lower() for t in ["int", "float", "decimal", "numeric"])
        if is_numeric:
            try:
                if hasattr(data[col], "min"):
                    col_info["min"] = float(data[col].min())
                    col_info["max"] = float(data[col].max())
            except Exception:
                pass

        # Cardinality for string/categorical
        is_string = any(
            t in dtype.lower() for t in ["object", "string", "utf8", "str", "categorical"]
        )
        if is_string or (not is_numeric):
            try:
                if hasattr(data[col], "nunique"):
                    col_info["n_unique"] = int(data[col].nunique())
                elif hasattr(data[col], "n_unique"):
                    col_info["n_unique"] = int(data[col].n_unique())
            except Exception:
                pass

        column_analyses.append(col_info)

        # Generate validation suggestions based on actual data characteristics
        if null_count == 0:
            validation_steps.append(f"    .col_vals_not_null(columns='{col}')  # Currently 0% null")
        elif null_pct > 50:
            # Don't validate not-null for mostly-null columns
            pass

        if is_numeric and "min" in col_info and "max" in col_info:
            min_val = col_info["min"]
            max_val = col_info["max"]
            if min_val >= 0:
                validation_steps.append(
                    f"    .col_vals_ge(columns='{col}', value=0)  # Range: [{min_val}, {max_val}]"
                )
            else:
                validation_steps.append(
                    f"    .col_vals_between(columns='{col}', left={min_val}, right={max_val})"
                )

        if is_string and "n_unique" in col_info:
            n_unique = col_info["n_unique"]
            if n_unique <= 10 and rows > 20:
                # Low cardinality - suggest in_set
                try:
                    if hasattr(data[col], "dropna"):  # pandas
                        unique_vals = list(data[col].dropna().unique()[:10])
                    elif hasattr(data[col], "drop_nulls"):  # polars
                        unique_vals = data[col].drop_nulls().unique().to_list()[:10]
                    else:
                        unique_vals = None

                    if unique_vals:
                        validation_steps.append(
                            f"    .col_vals_in_set(columns='{col}', set={repr(unique_vals)})"
                        )
                except Exception:
                    pass

    # Build column details section
    col_details = []
    for ca in column_analyses:
        detail = f"  - **{ca['name']}** ({ca['dtype']}): {ca['null_pct']}% null"
        if "min" in ca:
            detail += f", range [{ca['min']}, {ca['max']}]"
        if "n_unique" in ca:
            detail += f", {ca['n_unique']} unique values"
        col_details.append(detail)

    # Build validation code
    code = "import pointblank as pb\n\nvalidator = (\n    pb.Validate(data)\n"
    if validation_steps:
        code += "\n".join(validation_steps)
    else:
        code += "    # No automatic suggestions - add your own validation steps"
    code += "\n    .interrogate()\n)"

    response = f"""**Validation Assistant for '{dataframe_id}'**

**Data Overview:** {rows:,} rows x {cols} columns

**Column Analysis:**
{chr(10).join(col_details)}

**Suggested Validation Plan ({len(validation_steps)} rules):**

```python
{code}
```

**Next Steps:**
1. Review and customize the suggestions above
2. Use `create_validator` + `add_validation_step` to implement
3. Or use `draft_validation_plan` for AI-powered suggestions
"""

    return response


# =============================================================================
# TOOLS: Server Management
# =============================================================================


@mcp.tool(
    name="server_health_check",
    description="Get comprehensive server health and status information.",
    tags={"Server Management"},
)
async def server_health_check(ctx: Context) -> Dict[str, Any]:
    """Returns server health information."""
    app_ctx: AppContext = ctx.request_context.lifespan_context

    dataframes_count = len(app_ctx.loaded_dataframes)
    validators_count = len(app_ctx.active_validators)

    total_memory_mb = 0
    for df in app_ctx.loaded_dataframes.values():
        try:
            if hasattr(df, "memory_usage"):
                total_memory_mb += df.memory_usage(deep=True).sum() / 1024 / 1024
            else:
                total_memory_mb += df.shape[0] * df.shape[1] * 8 / 1024 / 1024
        except Exception:
            pass

    health_info = {
        "timestamp": datetime.now().isoformat(),
        "server_status": "healthy",
        "system_info": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "backend_status": {
            "pandas_available": HAS_PANDAS,
            "polars_available": HAS_POLARS,
        },
        "resource_usage": {
            "total_dataframes": dataframes_count,
            "total_validators": validators_count,
            "total_memory_mb": round(total_memory_mb, 2),
        },
        "capabilities": {
            "supported_file_formats": ["CSV", "JSON", "JSONL", "Parquet"]
            + (["Excel"] if HAS_PANDAS else []),
            "validation_types_count": len(SUPPORTED_VALIDATION_TYPES),
            "templates_available": AVAILABLE_TEMPLATES,
        },
    }

    warnings = []
    if not HAS_PANDAS and not HAS_POLARS:
        warnings.append("No DataFrame backends available")
        health_info["server_status"] = "degraded"
    if total_memory_mb > 1000:
        warnings.append(f"High memory usage: {total_memory_mb:.1f}MB")
    if warnings:
        health_info["warnings"] = warnings

    return health_info


@mcp.tool(
    name="get_pointblank_api_reference",
    description="Get API reference for Pointblank validation methods and common patterns.",
    tags={"Reference"},
)
async def get_pointblank_api_reference(
    ctx: Context,
    category: Annotated[
        str,
        "Category: 'validation_methods', 'thresholds', 'common_patterns', or 'all'",
    ] = "validation_methods",
) -> str:
    """Provides API reference for Pointblank validation methods and patterns."""

    validation_methods_ref = """# Pointblank Validation Methods

## Value Comparisons
- `col_vals_gt(columns, value)` - Greater than
- `col_vals_ge(columns, value)` - Greater than or equal
- `col_vals_lt(columns, value)` - Less than
- `col_vals_le(columns, value)` - Less than or equal
- `col_vals_eq(columns, value)` - Equal to
- `col_vals_ne(columns, value)` - Not equal to
- `col_vals_between(columns, left, right)` - Within range (inclusive)
- `col_vals_outside(columns, left, right)` - Outside range

## Set Membership
- `col_vals_in_set(columns, set)` - Values in allowed set
- `col_vals_not_in_set(columns, set)` - Values not in set

## Null Checks
- `col_vals_null(columns)` - Values are null
- `col_vals_not_null(columns)` - Values are not null

## Pattern Matching
- `col_vals_regex(columns, pattern)` - Match regex pattern

## Table Checks
- `col_exists(columns)` - Column exists
- `rows_distinct()` - All rows unique
- `rows_complete()` - No nulls in specified columns
- `row_count_match(count)` - Expected row count
- `col_count_match(count)` - Expected column count
- `col_schema_match(schema)` - Schema validation
"""

    thresholds_ref = """# Thresholds

```python
thresholds = {"warning": 0.05, "error": 0.10, "critical": 0.15}
```

- **warning**: Minor issues (e.g., 5% failures)
- **error**: Significant problems (e.g., 10% failures)
- **critical**: Severe issues that stop processing
"""

    patterns_ref = """# Common Patterns

```python
# Data Integrity
.col_vals_not_null(columns="id")
.rows_distinct()
.col_exists(columns=["required_field1", "required_field2"])

# Business Rules
.col_vals_between(columns="age", left=0, right=120)
.col_vals_in_set(columns="status", set=["active", "inactive"])
.col_vals_ge(columns="price", value=0)

# Format Validation
.col_vals_regex(columns="email", pattern=r"[^@]+@[^@]+\\.[^@]+")
```
"""

    if category == "validation_methods":
        return validation_methods_ref
    elif category == "thresholds":
        return thresholds_ref
    elif category == "common_patterns":
        return patterns_ref
    elif category == "all":
        return f"{validation_methods_ref}\n{thresholds_ref}\n{patterns_ref}"
    else:
        return f"Unknown category '{category}'. Use: validation_methods, thresholds, common_patterns, or all"
