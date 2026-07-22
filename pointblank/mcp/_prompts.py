"""MCP prompt definitions for the Pointblank MCP server."""

from typing import Annotated, Dict, Optional, Union

from fastmcp.prompts.prompt import Message


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


def prompt_create_validator(
    df_id: Annotated[str, "ID of the DataFrame to validate."] = "df_default",
    validator_id: Annotated[
        Optional[str],
        "Optional ID for the Validator. If not provided, a new ID will be generated.",
    ] = "validator_default",
    table_name: Annotated[
        Optional[str],
        "Optional name for the table within Pointblank reports.",
    ] = "data_table",
    validator_label: Annotated[
        Optional[str],
        "Optional descriptive label for the Validator.",
    ] = "Validator",
    thresholds_dict_example: Annotated[
        Optional[Dict[str, Union[int, float]]],
        "Example thresholds for validation failures.",
    ] = None,
) -> tuple:
    """Prompt guiding the LLM to create a Pointblank Validator object."""
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
    """Guides the LLM to get a validation output CSV."""
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


def prompt_interrogate_validator(
    validator_id: Annotated[str, "ID of the Validator to interrogate."] = "validator_123",
) -> tuple:
    """Prompt guiding the LLM to run validations and generate reports."""
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
