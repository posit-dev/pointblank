# MCP Pointblank Server Documentation

This document provides documentation for the `pointblank_server.py` script located in the `pointblank_mcp_server` folder. This server leverages the FastMCP framework to expose data validation functionalities of the Pointblank library, allowing for robust data quality checks through a structured interface.

## Overview

`pointblank_server.py` implements a FastMCP server that acts as a bridge to the Pointblank data validation library. It allows users (or LLM agents) to load datasets, define validation rules, execute these validations, and retrieve detailed reports and data extracts. This enables programmatic and interactive data quality assessment within environments that can interface with an MCP server.

## Features

- **Load Data**: Supports loading datasets from CSV and Excel files into the server's memory.
- **Create Validators**: Allows the creation of Pointblank `Validate` objects, which are central to defining and running validation plans.
- **Define Validation Steps**: Offers a comprehensive set of validation rules that can be added to a validator, covering various aspects like column value checks, schema validation, row properties, etc.
- **Execute Validations**: Provides a mechanism to "interrogate" a validator, running all defined checks against the associated dataset.
- **Retrieve Results**:
  - Get a JSON summary of the entire validation process.
  - Save detailed validation reports in CSV or PNG format.
  - Extract data subsets (e.g., failing rows) for specific validation steps as CSV.
  - Save visual summaries of individual validation steps as PNG images.
- **LLM Guidance**: Includes prompt templates to guide Language Models in utilizing the server's tools effectively.

## Core Components

- **`AppContext`**: A dataclass that manages the state within the server's lifespan. It holds:
  - `loaded_dataframes`: A dictionary storing pandas DataFrames loaded by the user, keyed by a unique `df_id`.
  - `active_validators`: A dictionary storing Pointblank `Validate` objects, keyed by a unique `validator_id`.
- **`mcp = FastMCP(...)`**: Initializes the FastMCP server, configuring its name, lifespan context (using `AppContext`), and dependencies (`pandas`, `pointblank`, `openpyxl`, `great_tables`, `polars`).

## Available Tools (MCP Tools)

The server exposes the following tools that can be called by an MCP client:

### 1. `load_dataframe`

- **Purpose**: Loads a DataFrame from a specified CSV or Excel file into the server's context.
- **Parameters**:
  - `input_path` (str): The file path to the CSV or Excel file.
  - `df_id` (Optional[str]): An optional custom ID for the DataFrame. If not provided, a unique ID is generated.
- **Returns**: A dictionary containing:
  - `df_id` (str): The ID of the loaded DataFrame.
  - `status` (str): Confirmation message.
  - `shape` (tuple): The dimensions (rows, columns) of the DataFrame.
  - `columns` (list): A list of column names.
- **Raises**: `FileNotFoundError` if the path is invalid, `ValueError` if the file type is unsupported or `df_id` already exists.

### 2. `create_validator`

- **Purpose**: Creates a Pointblank `Validate` object for a previously loaded DataFrame.
- **Parameters**:
  - `df_id` (str): The ID of the DataFrame to validate (must be loaded first).
  - `validator_id` (Optional[str]): An optional custom ID for the validator. If not provided, a unique ID is generated.
  - `table_name` (Optional[str]): An optional name for the table, used in Pointblank reports.
  - `validator_label` (Optional[str]): An optional descriptive label for the validator.
  - `thresholds_dict` (Optional[Dict[str, Union[int, float]]]): Optional dictionary to set global failure thresholds (e.g., `{"warning": 0.1, "error": 5}`).
  - `actions_dict` (Optional[Dict[str, Any]]): Configuration for Pointblank actions.
  - `final_actions_dict` (Optional[Dict[str, Any]]): Configuration for Pointblank final actions.
  - `brief` (Optional[bool]): Pointblank specific option.
  - `lang` (Optional[str]): Pointblank specific option for language.
  - `locale` (Optional[str]): Pointblank specific option for locale.
- **Returns**: A dictionary containing:
  - `validator_id` (str): The ID of the created validator.
  - `status` (str): Confirmation message.
- **Raises**: `ValueError` if `df_id` is not found or `validator_id` already exists.

### 3. `add_validation_step`

- **Purpose**: Adds a specific validation rule/step to an existing Pointblank validator.
- **Parameters**:
  - `validator_id` (str): The ID of the validator to which the step should be added.
  - `validation_type` (str): The name of the Pointblank validation function to call (e.g., `col_vals_lt`, `rows_distinct`). This corresponds to methods available in the Pointblank `Validate` class.
  - `params` (Dict[str, Any]): A dictionary of parameters required by the specified `validation_type` function. Parameter names and types must match the Pointblank API.
  - `actions_config` (Optional[Dict[str, Any]]): Optional simplified action definition (currently basic support).
- **Returns**: A dictionary containing:
  - `validator_id` (str): The ID of the validator.
  - `status` (str): Confirmation message.
- **Raises**: `ValueError` if `validator_id` is not found, `validation_type` is unsupported, or if there's a `TypeError` due to incorrect `params` for the chosen validation method.

**Note on `validation_type` and `params`**: The `validation_type` string must be one of the keys in the `supported_validations` dictionary within the tool's implementation (e.g., "col_vals_lt", "col_exists", "rows_distinct"). The `params` dictionary must then provide the arguments that the corresponding Pointblank `Validate` method expects. For example, for `validation_type='col_vals_lt'`, `params` would be `{'columns': 'column_name', 'value': 100}`.

### 4. `get_validation_step_output`

- **Purpose**: Retrieves and saves the output for a specific validation step from an interrogated validator.
- **Parameters**:
  - `validator_id` (str): The ID of the interrogated validator.
  - `step_index` (int): The 1-based index of the validation step for which to retrieve output.
  - `output_path` (str): The file path where the output should be saved.
    - If ends with `.csv`: Saves the data extract (e.g., rows that failed or passed the validation). Defaults to this if no extension.
    - If ends with `.png`: Saves a visual summary report for the step.
  - `sundered_type` (str, default="fail"): Specifies whether to retrieve "fail" or "pass" data for CSV extracts using `get_sundered_data`.
- **Returns**: A dictionary containing status, message, and the resolved path to the output file.
- **Raises**: `ValueError` if `validator_id` is not found or output format is unsupported. `RuntimeError` for other issues during output generation.
- **Note**: The validator should ideally be interrogated before calling this, though the tool attempts to interrogate if not already done.

### 5. `interrogate_validator`

- **Purpose**: Runs all defined validation steps for a given validator and returns a summary. Optionally saves a detailed report.
- **Parameters**:
  - `validator_id` (str): The ID of the validator to interrogate.
  - `report_file_path` (Optional[str]): If provided, the path to save the validation report.
    - If ends with `.csv`: Saves the report as a CSV file.
    - If ends with `.png`: Saves the report as a PNG image (requires `great_tables`).
- **Returns**: A dictionary containing:
  - `validation_summary` (str): A JSON string representing the validation report.
  - Optionally, `csv_report_saved_to` or `png_report_saved_to` if `report_file_path` was provided and successful.
  - Optionally, `report_save_error` if saving failed.
- **Raises**: `ValueError` if `validator_id` is not found. `RuntimeError` for errors during interrogation.

## Prompt Templates (MCP Prompts)

The server includes prompt templates to assist LLMs in correctly formulating calls to the available tools. These are decorated with `@mcp.prompt()`.

- **`prompt_load_dataframe`**: Guides on how to use the `load_dataframe` tool.
- **`prompt_create_validator`**: Guides on how to use the `create_validator` tool, including examples for parameters like `thresholds_dict`.
- **`prompt_add_validation_step_example`**: Provides examples and guidance for using the `add_validation_step` tool, emphasizing the correct structure for `validation_type` and `params`.
- **`prompt_get_validation_step_output`**: Explains how to use `get_validation_step_output` for both CSV data extracts and PNG visual reports.
- **`prompt_interrogate_validator`**: Guides on how to use the `interrogate_validator` tool and its reporting options.

## Usage Workflow Example

A typical interaction with the server might follow these steps:

1.  **Load Data**:

    ```json
    // Call to load_dataframe
    {
      "tool_name": "load_dataframe",
      "arguments": {
        "input_path": "path/to/your/data.csv",
        "df_id": "my_dataset"
      }
    }
    // Server returns: {"df_id": "my_dataset", ...}
    ```

2.  **Create Validator**:

    ```json
    // Call to create_validator
    {
      "tool_name": "create_validator",
      "arguments": {
        "df_id": "my_dataset",
        "validator_id": "my_validator",
        "table_name": "Sales Data",
        "validator_label": "Initial Quality Checks"
      }
    }
    // Server returns: {"validator_id": "my_validator", ...}
    ```

3.  **Add Validation Steps**:

    ```json
    // Call to add_validation_step (example: check 'age' column values are less than 120)
    {
      "tool_name": "add_validation_step",
      "arguments": {
        "validator_id": "my_validator",
        "validation_type": "col_vals_lt",
        "params": { "columns": "age", "value": 120 }
      }
    }
    // Server returns: {"validator_id": "my_validator", "status": "Validation step 'col_vals_lt' added successfully."}
    ```

    ```json
    // Call to add_validation_step (example: check 'email' column for non-null values)
    {
      "tool_name": "add_validation_step",
      "arguments": {
        "validator_id": "my_validator",
        "validation_type": "col_vals_not_null",
        "params": { "columns": "email" }
      }
    }
    // Server returns: {"validator_id": "my_validator", "status": "Validation step 'col_vals_not_null' added successfully."}
    ```

4.  **Interrogate Validator**:

    ```json
    // Call to interrogate_validator and save a CSV report
    {
      "tool_name": "interrogate_validator",
      "arguments": {
        "validator_id": "my_validator",
        "report_file_path": "output/validation_report.csv"
      }
    }
    // Server returns: {"validation_summary": "{...json...}", "csv_report_saved_to": "output/validation_report.csv"}
    ```

5.  **Get Specific Step Output (Optional)**:
    ```json
    // Call to get_validation_step_output for the first step (index 1), expecting failing rows
    {
      "tool_name": "get_validation_step_output",
      "arguments": {
        "validator_id": "my_validator",
        "step_index": 1,
        "output_path": "output/age_validation_failures.csv",
        "sundered_type": "fail"
      }
    }
    // Server returns: {"status": "success", ..., "output_file": "output/age_validation_failures.csv"}
    ```

## Purpose for Pull Request

This `pointblank_server.py` script and its associated MCP interface provide a significant enhancement for data quality workflows. By exposing the comprehensive validation capabilities of the Pointblank library through a standardized MCP, it enables:

- **Automated Data Quality Checks**: Integration into automated data pipelines for continuous monitoring.
- **LLM-Driven Data Analysis**: Allows Language Models to interactively define and execute data validation tasks, making complex data quality assessments more accessible.
- **Standardized Interface**: Offers a consistent way to perform data validations, regardless of the underlying data source format (as long as it can be loaded into a pandas DataFrame).
- **Reproducible Validations**: Validation plans can be easily versioned and reused.

The inclusion of this server will empower users and automated systems to ensure higher data quality with greater ease and flexibility.
