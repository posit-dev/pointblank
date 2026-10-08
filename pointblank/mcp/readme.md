# Pointblank MCP Server

This package implements a FastMCP server that exposes the Pointblank data validation library through the Model Context Protocol (MCP). It allows LLM agents and other clients to load datasets, define validation rules, execute validations, and retrieve reports.

## Module Structure

```
pointblank/mcp/
├── __init__.py       # Package init, exports `mcp`
├── __main__.py       # Entry point: `python -m pointblank.mcp`
├── _config.py        # Shared config: backend detection, logging, constants
├── _prompts.py       # MCP prompt definitions
├── _templates.py     # Validation templates (basic_quality, financial_data, etc.)
├── _utils.py         # Utility functions: path validation, ID validation, helpers
├── server.py         # FastMCP instance, tool definitions
└── readme.md         # This file
```

## Running the Server

```bash
python -m pointblank.mcp
```

The server runs over stdio transport by default.

## Available Tools (19)

### Data Management
- **`load_dataframe`** — Load a CSV, Excel, or Parquet file into the server context.
- **`list_loaded_dataframes`** — List all DataFrames currently in memory.
- **`list_available_backends`** — Show available DataFrame backends (pandas/polars).
- **`delete_dataframe`** — Remove a DataFrame from memory.

### Data Analysis
- **`profile_dataframe`** — Profile a DataFrame using Pointblank's DataScan (column-level statistics, null counts, value distributions). Supports optional row sampling.

### Validation
- **`create_validator`** — Create a Pointblank Validator for a loaded DataFrame.
- **`add_validation_step`** — Add a validation rule to an existing Validator.
- **`apply_validation_template`** — Apply a pre-built validation template (basic_quality, financial_data, customer_data, sensor_data, survey_data).
- **`interrogate_validator`** — Run all validation steps and return JSON summary + HTML report + Python code equivalent.
- **`get_validation_step_output`** — Export pass/fail rows for a validation step to CSV.
- **`list_active_validators`** — List all validators in the server context.
- **`delete_validator`** — Remove a validator from memory.

### Table Visualization
- **`preview_table`** — Display head + tail rows of a DataFrame as an HTML table.
- **`missing_values_table`** — Generate a missing values analysis table.
- **`column_summary_table`** — Generate a column-level summary table.

### AI & Assistant
- **`validation_assistant`** — Generate data-aware validation suggestions by analyzing actual column types, null rates, value ranges, and cardinality.
- **`draft_validation_plan`** — Use AI (via Pointblank's DraftValidation) to generate a validation plan.

### Reference & Management
- **`get_pointblank_api_reference`** — Get API docs for validation methods, thresholds, or common patterns.
- **`server_health_check`** — Server health, backend status, resource usage, capabilities.

## Prompts (5)

- `prompt_load_dataframe` — Guide for loading data
- `prompt_create_validator` — Guide for creating validators
- `prompt_add_validation_step_example` — Guide for adding validation steps
- `prompt_get_validation_step_output` — Guide for retrieving output
- `prompt_interrogate_validator` — Guide for running validations

## Security

- **Path validation**: Input/output paths are validated against traversal attacks, allowed extensions, size limits, and blocked system directories.
- **Resource ID validation**: All DataFrame and Validator IDs must match `^[a-zA-Z0-9_\-]{1,128}$`.
- **File size limit**: 500 MB maximum for loaded files.

## Workflow Example

```python
# 1. Load data
load_dataframe(input_path="data.csv", df_id="sales")

# 2. Create validator
create_validator(df_id="sales", validator_id="sales_check")

# 3. Add validation steps
add_validation_step(validator_id="sales_check", validation_type="col_vals_not_null", params={"columns": "id"})
add_validation_step(validator_id="sales_check", validation_type="col_vals_ge", params={"columns": "price", "value": 0})

# 4. Run validations
interrogate_validator(validator_id="sales_check")

# 5. Export failures
get_validation_step_output(validator_id="sales_check", output_path="failures.csv", step_index=2)
```
