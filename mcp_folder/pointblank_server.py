from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import AsyncIterator, Dict, Any, Optional, Union  # Keep Union for now
from pathlib import Path
import uuid
import json
import pandas as pd
import pointblank as pb
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.fastmcp.prompts import base


# --- Lifespan Context: manage DataFrames and Validators ---
@dataclass
class AppContext:
    # Stores loaded DataFrames: {df_id: DataFrame}
    loaded_dataframes: Dict[str, pd.DataFrame] = field(default_factory=dict)
    # Stores active Pointblank Validators: {validator_id: Validate}
    active_validators: Dict[str, pb.Validate] = field(default_factory=dict)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    context = AppContext()
    yield context
    context.loaded_dataframes.clear()
    context.active_validators.clear()


mcp = FastMCP(
    "FlexiblePointblankMCP",
    lifespan=app_lifespan,
    dependencies=["pandas", "pointblank", "openpyxl", "great_tables", "polars"],
)


def _load_dataframe_from_path(input_path: str) -> pd.DataFrame:
    p_path = Path(input_path)
    if not p_path.exists():
        raise FileNotFoundError(f"Input file '{input_path}' not found.")
    if p_path.suffix.lower() == ".csv":
        return pd.read_csv(p_path)
    elif p_path.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(p_path, engine="openpyxl")
    else:
        raise ValueError(f"Unsupported file type: {p_path.suffix}. Please use CSV or Excel.")


@mcp.tool()
def load_dataframe(input_path: str, df_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Loads a DataFrame from the specified CSV or Excel file into the server's context.
    Assigns a unique ID to the DataFrame for later reference.
    If df_id is not provided, a new one will be generated.
    Returns the DataFrame ID and basic information (shape, columns).
    """
    ctx: Context = mcp.get_context()
    app_ctx: AppContext = ctx.request_context.lifespan_context
    df = _load_dataframe_from_path(input_path)

    effective_df_id = df_id if df_id else f"df_{uuid.uuid4().hex[:8]}"

    if effective_df_id in app_ctx.loaded_dataframes:
        raise ValueError(
            f"DataFrame ID '{effective_df_id}' already exists. Choose a different ID or omit to generate a new one."
        )

    app_ctx.loaded_dataframes[effective_df_id] = df

    return {
        "df_id": effective_df_id,
        "status": "DataFrame loaded successfully.",
        "shape": df.shape,
        "columns": list(df.columns),
    }


@mcp.tool()
def create_validator(
    df_id: str,
    validator_id: Optional[str] = None,
    table_name: Optional[str] = None,  # Corresponds to 'name' in pb.Validate
    validator_label: Optional[str] = None,  # Corresponds to 'label' in pb.Validate
    thresholds_dict: Optional[
        Dict[str, Union[int, float]]
    ] = None,  # e.g. {"warning": 1, "error": 20, "critical": 0.10}
    actions_dict: Optional[Dict[str, Any]] = None,  # Simplified, for pb.Actions
    final_actions_dict: Optional[Dict[str, Any]] = None,  # Simplified, for pb.FinalActions
    brief: Optional[bool] = None,
    lang: Optional[str] = None,
    locale: Optional[str] = None,
) -> Dict[str, str]:
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
    ctx: Context = mcp.get_context()
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

    return {"validator_id": effective_validator_id, "status": "Validator created successfully."}


@mcp.tool()
def add_validation_step(
    validator_id: str,
    validation_type: str,
    params: Dict[str, Any],
    actions_config: Optional[Dict[str, Any]] = None,  # Placeholder for simplified action definition
) -> Dict[str, str]:
    """
    Adds a validation step to an existing Pointblank Validator.
    'validator_id' must refer to a validator created via 'create_validator'.
    'validation_type' specifies the Pointblank validation function to call
      (e.g., 'col_vals_lt', 'col_vals_between', 'col_vals_in_set', 'col_exists', 'rows_distinct').
    'params' is a dictionary of parameters for that validation function.
    'actions_config' (optional) can be used to define simple actions (currently basic support).
    """
    ctx: Context = mcp.get_context()
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
        "col_vals_lte": validator.col_vals_le,  # less or equal
        "col_vals_gte": validator.col_vals_ge,  # greater or equal
        "col_vals_equal": validator.col_vals_eq,  # equal to a value
        "col_vals_not_equal": validator.col_vals_ne,  # not equal to a value
        "col_vals_between": validator.col_vals_between,  # data lies between two values left=val, right=val
        "col_vals_not_between": validator.col_vals_outside,  # data is outside two values
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
        "rows_distinct": validator.rows_distinct,  # distinc rows in a table
        "rows_complete": validator.rows_complete,  # Check for no NAs in specified columns
        "row_count_match": validator.row_count_match,  # Check if number of rows in the table matches a fixed value
        # Other specialized validations
        "conjointly": validator.conjointly,  # For multiple column conditions
        "col_schema_match": validator.col_schema_match,  # Do columns in the table (and their types) match a predefined schema? columns=[("a", "String"), ("b", "Int64"), ("c", "Float64")]
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

    return {
        "validator_id": validator_id,
        "status": f"Validation step '{validation_type}' added successfully.",
    }


@mcp.tool()
async def get_validation_step_output(
    validator_id: str,
    output_path: str,
    step_index: Optional[int] = None,  # Made step_index optional
    sundered_type: str = "fail",
) -> Dict[str, Any]:
    """
    Retrieves output for a validation and saves it to a file.
    If 'step_index' is provided, it fetches the data extract for that specific step (e.g., failing rows).
    If 'step_index' is NOT provided, it fetches all rows that failed ('fail') or passed ('pass') any step.
    The output format is determined by the file extension of 'output_path' (.csv or .png).
    """
    ctx: Context = mcp.get_context()
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if validator_id not in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{validator_id}' not found.")
    validator = app_ctx.active_validators[validator_id]

    p_output_path = Path(output_path)
    output_format = p_output_path.suffix.lower()

    if not output_format:
        if step_index:
            # Default to PNG for specific step reports if no extension is given
            output_format = ".png"
            p_output_path = p_output_path.with_suffix(".png")
        else:
            # Default to CSV for sundered data
            output_format = ".csv"
            p_output_path = p_output_path.with_suffix(".csv")
        await ctx.warning(
            f"No file extension provided. Defaulting to {output_format}. Saving to: {p_output_path}"
        )

    try:
        # Ensure validator has been interrogated.
        if not getattr(validator, "time_processed", None):
            await ctx.warning(
                f"Validator '{validator_id}' has not been interrogated. Interrogating now."
            )

        p_output_path.parent.mkdir(parents=True, exist_ok=True)
        message = ""
        data_extract_df = None

        # --- Logic for CSV output ---
        if output_format == ".csv":
            if step_index:
                # --- Get extract for a SPECIFIC step ---
                data_extract_df = validator.get_data_extracts(i=step_index, frame=True)
                if data_extract_df is None:
                    message = f"No data extract available for step {step_index}. This may mean all rows passed validation."
                else:
                    message = f"Data extract for step {step_index} retrieved."
            else:
                # --- Get all sundered data (pass or fail) ---
                data_extract_df = validator.get_sundered_data(type=sundered_type)
                if data_extract_df is None:
                    message = f"No sundered data available for type '{sundered_type}'."
                else:
                    message = f"Sundered data for type '{sundered_type}' retrieved."

            if data_extract_df is None:
                return {"status": "success", "message": message, "output_file": None}

            # Save the retrieved dataframe
            if isinstance(data_extract_df, (pd.DataFrame)):
                data_extract_df.to_csv(p_output_path, index=False)
                message = f"Data extract saved to {p_output_path.resolve()}"
            else:
                raise TypeError(
                    f"Unsupported DataFrame type '{type(data_extract_df).__name__}' for CSV export."
                )

        # --- Logic for PNG output (requires a step_index) ---
        elif output_format == ".png":
            if not step_index:
                raise ValueError(
                    "A 'step_index' is required to generate a PNG report for a specific step."
                )

            step_report_obj = validator.get_step_report(i=step_index)

            if step_report_obj is None:
                num_steps = len(validator.n())
                raise ValueError(
                    f"No report found for step index {step_index}. Validator has {num_steps} step(s)."
                )

            if not hasattr(step_report_obj, "save"):
                raise TypeError(
                    "The visual report object for this step does not have a .save() method."
                )

            step_report_obj.save(str(p_output_path))
            message = f"Visual report for step {step_index} saved to {p_output_path.resolve()}"

        else:
            raise ValueError(
                f"Unsupported file format '{output_format}'. Please use '.csv' or '.png'."
            )

        await ctx.report_progress(100, 100, message)
        return {
            "status": "success",
            "message": message,
            "output_file": str(p_output_path.resolve()),
        }

    except Exception as e:
        raise RuntimeError(f"Error getting output for validator '{validator_id}': {e}")


@mcp.tool()
async def interrogate_validator(
    validator_id: str, report_file_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Runs validations and returns a JSON summary.
    Optionally saves the report to 'report_file_path'.
    If path ends with .csv, saves as CSV. If .png, saves as PNG image.
    """
    ctx: Context = mcp.get_context()
    app_ctx: AppContext = ctx.request_context.lifespan_context

    if validator_id not in app_ctx.active_validators:
        raise ValueError(f"Validator ID '{validator_id}' not found.")

    validator = app_ctx.active_validators[validator_id]

    try:
        # interrogate() modifies the validator in place and returns self
        validator.interrogate()
        json_report_str = validator.get_json_report()
    except Exception as e:
        raise RuntimeError(f"Error during validator interrogation: {e}")

    output_dict = {"validation_summary": json_report_str}

    if report_file_path:
        p_report_file_path = Path(report_file_path)
        p_report_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_saved_path = None

        try:
            if report_file_path.lower().endswith(".csv"):
                report_data = json.loads(json_report_str)
                # The JSON report is a list of dicts, suitable for DataFrame
                df_report = pd.DataFrame(report_data)
                df_report.to_csv(p_report_file_path, index=False)
                file_saved_path = str(p_report_file_path.resolve())
                output_dict["csv_report_saved_to"] = file_saved_path

            elif report_file_path.lower().endswith(".png"):
                # Assumes get_tabular_report() returns a GreatTable-like object with .save()
                tabular_report_obj = validator.get_tabular_report()
                if hasattr(tabular_report_obj, "save"):
                    tabular_report_obj.save(str(p_report_file_path))
                    file_saved_path = str(p_report_file_path.resolve())
                    output_dict["png_report_saved_to"] = file_saved_path
                else:
                    # Fallback if get_tabular_report() returns a DataFrame, try to use great_tables explicitly
                    try:
                        from great_tables import GT

                        gt_table = GT(
                            tabular_report_obj
                        )  # Assumes tabular_report_obj is a DataFrame

                        gt_table.save(file=str(p_report_file_path))
                        file_saved_path = str(p_report_file_path.resolve())
                        output_dict["png_report_saved_to"] = file_saved_path
                    except ImportError:
                        err_msg = "Package 'great_tables' not installed. Cannot save as PNG from DataFrame."
                        print(err_msg)
                        output_dict["report_save_error"] = err_msg
                    except Exception as e_gt:
                        err_msg = f"Failed to save PNG using great_tables from DataFrame: {e_gt}"
                        print(err_msg)
                        output_dict["report_save_error"] = err_msg
            else:
                err_msg = "Unsupported report file extension. Use .csv or .png."
                print(err_msg)
                output_dict["report_save_error"] = err_msg

            if file_saved_path:
                await ctx.report_progress(100, 100, f"Report saved to {file_saved_path}")

        except Exception as e:
            error_msg = f"Failed to save report to {report_file_path}: {e}"
            print(error_msg)
            output_dict["report_save_error"] = error_msg

    return output_dict


# --- Prompt Templates (Updated Terminology) ---
@mcp.prompt()
def prompt_load_dataframe(input_path: str, df_id: Optional[str] = "my_data") -> tuple:
    return (
        base.AssistantMessage("I can load your data from a file into my context for validation."),
        base.UserMessage(
            f"Please call `load_dataframe` with input_path='{input_path}'. "
            f"You can optionally provide a `df_id` (e.g., '{df_id}') to name this dataset, "
            "or I will generate one for you. Make a note of the returned `df_id` for subsequent steps."
        ),
    )


@mcp.prompt()
def prompt_create_validator(
    df_id: str,
    validator_id: Optional[str] = "val_default",
    table_name: Optional[str] = "data_table",
    validator_label: Optional[str] = "My Validation",
    thresholds_dict_example: Optional[Dict[str, Union[int, float]]] = None,  # For hint
) -> tuple:
    """
    Prompt guiding the LLM to create a Pointblank Validator object.
    Includes an example for thresholds_dict.
    """
    thresholds_msg_example = (
        thresholds_dict_example if thresholds_dict_example else {"warning": 0.05, "error": 10}
    )  # Default example

    return (
        base.AssistantMessage(
            "Once your data is loaded (using its `df_id`), I can create a 'Validator' object to define data quality checks."
        ),
        base.UserMessage(
            f"Please call `create_validator` using the `df_id` of your loaded data (e.g., '{df_id}').\n"
            f"You can optionally provide:\n"
            f"- `validator_id` (e.g., '{validator_id}') to name this validator instance.\n"
            f"- `table_name` (e.g., '{table_name}') as a reference name for the data table in reports.\n"
            f"- `validator_label` (e.g., '{validator_label}') for a descriptive label.\n"
            f"- `thresholds_dict` (e.g., {thresholds_msg_example}) to set global failure thresholds for validation steps.\n"
            f"- Other optional parameters like `actions_dict`, `final_actions_dict`, `brief`, `lang`, `locale` can also be specified if needed.\n"
            "Make a note of the returned `validator_id` to use when adding validation steps."
        ),
    )


@mcp.prompt()
def prompt_add_validation_step_example() -> tuple:
    return (
        base.AssistantMessage(
            "I can add various validation steps to your validator. "
            "You'll need to specify the 'validator_id', 'validation_type', and 'params' for the step. "
            "For example, to check if values in column 'age' are less than 100 for validator 'validator_123':"
        ),
        base.UserMessage(
            "Please call `add_validation_step` with validator_id='validator_123', "
            "validation_type='col_vals_lt', and params={'columns': 'age', 'value': 100}. "
            "Note: Parameter names within 'params' (like 'columns', 'value', 'left', 'right', 'set_', etc.) must exactly match what the specific Pointblank validation function expects.\n"
            "Other examples:\n"
            "- For 'col_vals_between': params={'columns': 'score', 'left': 0, 'right': 100, 'inclusive': [True, True]}\n"
            "- For 'col_vals_in_set': params={'columns': 'grade', 'set_': ['A', 'B', 'C']} (Note: Pointblank uses 'set_' for this method's list of values)\n"
            "- For 'col_exists': params={'columns': 'user_id'}\n"
            "Refer to the Pointblank Python API for the 'Validate' class for available `validation_type` (method names) and their specific `params`."
        ),
    )


@mcp.prompt()
def prompt_get_validation_step_output(
    validator_id: str, step_index: int = 1, output_path: str = "step_1_extract.csv"
) -> tuple:
    """
    Guides the LLM to get an output for a specific validation step,
    explaining the choice between a CSV data extract and a PNG visual report.
    """
    return (
        base.AssistantMessage(
            "For any validation step, I can either extract the relevant data rows (e.g., the rows that failed) as a CSV file, "
            "or I can generate a visual summary of the step as a PNG image."
        ),
        base.UserMessage(
            f"Please get the output for step number {step_index} from validator '{validator_id}'.\n"
            f"To get the data extract, provide an `output_path` ending in `.csv` (e.g., 'step_{step_index}_failures.csv'). This is the default.\n"
            f"To get the visual report, provide an `output_path` ending in `.png` (e.g., 'step_{step_index}_report.png').\n\n"
            f"To proceed, call `get_validation_step_output` with the correct `validator_id`, `step_index`, and `output_path`."
        ),
    )


@mcp.prompt()
def prompt_interrogate_validator(
    validator_id: str, report_file_path: Optional[str] = "validation_summary.csv"
) -> tuple:
    """
    Prompt guiding the LLM to run validations and optionally save the report.
    """
    return (
        base.AssistantMessage(
            "After all desired validation steps have been added to a validator, I can run the interrogation process. This will execute all checks."
        ),
        base.UserMessage(
            f"Please call `interrogate_validator` with the `validator_id` (e.g., '{validator_id}').\n"
            f"The main result will be a JSON string summarizing all validation steps.\n"
            f"Optionally, you can specify `report_file_path` to save the report to a file. "
            f"For example:\n"
            f"- To save as a CSV file: `report_file_path='{report_file_path}'`\n"
            f"- To save as a PNG image: `report_file_path='validation_summary.png'`\n"
            f"If `report_file_path` is not provided, the report will not be saved to a file."
        ),
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
